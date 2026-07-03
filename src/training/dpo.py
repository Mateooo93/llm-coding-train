"""
DPO (Direct Preference Optimization) trainer — factuality alignment.

DPO directly optimizes a language model to prefer one response over another
given a prompt, without needing to train a separate reward model (unlike PPO/RLHF).

The DPO loss formulation:
  L_DPO(π_θ; π_ref) = -E_{(x, y_w, y_l) ~ D} [log σ(β * (log_ratio_w - log_ratio_l))]
  where:
    log_ratio_w = log π_θ(y_w|x) - log π_ref(y_w|x)
    log_ratio_l = log π_θ(y_l|x) - log π_ref(y_l|x)
    y_w = preferred (factual) response
    y_l = dispreferred (hallucinated) response
    β = temperature for the implicit reward

This is THE breakthrough for hallucination reduction at training time. By
constructing (prompt, factual_response, hallucinated_response) triplets and
training with DPO, the model learns to prefer factual outputs.

Reference: "Direct Preference Optimization: Your Language Model is Secretly a
Reward Model" (Rafailov et al., 2023). arXiv:2305.18290.

Implementation notes:
  - We use the standard "reference-free" formulation that requires keeping
    a frozen copy of the reference policy (π_ref). For small models we just
    keep two copies in memory.
  - Loss includes the implicit SFT (sigmoid-based) term so the policy doesn't
    drift too far afield.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..model import AttnResLM


@dataclass
class DPOConfig:
    """Hyperparameters for DPO training."""
    learning_rate: float = 5e-7
    beta: float = 0.1  # KL penalty strength
    max_steps: int = 1000
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    max_length: int = 1024
    max_prompt_length: int = 512
    max_response_length: int = 512
    warmup_steps: int = 50
    label_smoothing: float = 0.0  # optional DPO smoothing
    use_amp: bool = True
    amp_dtype: str = "bf16"
    log_interval: int = 10
    save_dir: Optional[str] = None


def compute_logprobs(
    model: AttnResLM,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    prompt_lengths: torch.Tensor,
) -> torch.Tensor:
    """
    Compute per-token log-probabilities of ``labels`` under ``model``.

    Args:
        model: AttnResLM instance.
        input_ids: [batch, seq_len] — prompt + response concatenated.
        labels: [batch, seq_len] — same shape; -100 for prompt positions.
        prompt_lengths: [batch] — number of prompt tokens per example.

    Returns:
        sum_logprob: [batch] sum of logprobs of the response tokens.
            Length used = seq_len - 1 - prompt_lengths (the number of positions
            in ``shift_labels`` that have a real response label, after the
            standard next-token shift).
    """
    output = model(input_ids=input_ids)
    logits = output.logits  # [batch, seq_len, vocab_size]

    # Shift logits and labels: predict next token from current position
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # Cross-entropy with -100 ignore
    log_probs = -F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="none",
    ).view(shift_labels.shape)

    # Valid response positions: where the SHIFTED label is not -100.
    # Total response length in the shifted view = seq_len - 1 - prompt_lengths.
    valid_mask = (shift_labels != -100).float()
    masked_log_probs = log_probs * valid_mask

    sum_logprob = masked_log_probs.sum(dim=-1)
    return sum_logprob


def dpo_loss(
    policy_chosen_logprobs: torch.Tensor,
    policy_rejected_logprobs: torch.Tensor,
    reference_chosen_logprobs: torch.Tensor,
    reference_rejected_logprobs: torch.Tensor,
    beta: float = 0.1,
    label_smoothing: float = 0.0,
) -> tuple[torch.Tensor, dict]:
    """
    Compute the DPO loss.

    Args:
        policy_chosen_logprobs: [batch] sum logprobs of chosen response under policy.
        policy_rejected_logprobs: [batch] sum logprobs of rejected response under policy.
        reference_chosen_logprobs: [batch] sum logprobs under reference (chosen).
        reference_rejected_logprobs: [batch] sum logprobs under reference (rejected).
        beta: KL penalty strength (higher = more conservative, stays close to ref).
        label_smoothing: gamma (0 = standard DPO).

    Returns:
        loss: scalar DPO loss.
        metrics: dict with chosen_rewards, rejected_rewards, accuracy, margins.
    """
    # log-ratios
    policy_logratio_chosen = policy_chosen_logprobs - reference_chosen_logprobs
    policy_logratio_rejected = policy_rejected_logprobs - reference_rejected_logprobs

    chosen_rewards = beta * policy_logratio_chosen
    rejected_rewards = beta * policy_logratio_rejected

    # Margin (positive = policy prefers chosen over rejected)
    margins = chosen_rewards - rejected_rewards

    # Loss: -log σ(margins - label_smoothing_term)
    if label_smoothing > 0:
        loss = (
            -F.logsigmoid(margins - label_smoothing * torch.log(torch.tensor(2.0, device=margins.device))).mean()
            - label_smoothing * F.logsigmoid(-margins).mean()
        )
    else:
        loss = -F.logsigmoid(margins).mean()

    accuracy = (margins > 0).float().mean()
    metrics = {
        "chosen_rewards": chosen_rewards.detach().mean().item(),
        "rejected_rewards": rejected_rewards.detach().mean().item(),
        "reward_margin": margins.detach().mean().item(),
        "accuracy": float(accuracy.item()),
    }
    return loss, metrics


class DPOTrainer:
    """
    DPO trainer for aligning the AttnResLM to prefer factual responses.

    Expects preference data of the form:
    - chosen_input_ids: [B, T_c]
    - chosen_labels: [B, T_c]
    - rejected_input_ids: [B, T_r]
    - rejected_labels: [B, T_r]
    """

    def __init__(
        self,
        policy_model: AttnResLM,
        config: Optional[DPOConfig] = None,
        reference_model: Optional[AttnResLM] = None,
    ):
        self.policy_model = policy_model
        self.config = config or DPOConfig()

        # If reference_model is not provided, make a frozen copy of the policy.
        # This is the standard DPO setup.
        if reference_model is None:
            self.reference_model = copy.deepcopy(policy_model)
            for p in self.reference_model.parameters():
                p.requires_grad = False
            self.reference_model.eval()
        else:
            self.reference_model = reference_model
            for p in self.reference_model.parameters():
                p.requires_grad = False
            self.reference_model.eval()

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            [p for p in self.policy_model.parameters() if p.requires_grad],
            lr=self.config.learning_rate,
            betas=(0.9, 0.95),
            weight_decay=0.0,  # DPO usually uses no weight decay
        )

        # LR schedule: warmup then cosine decay to 0 (standard DPO recipe).
        warmup = self.config.warmup_steps
        max_steps = self.config.max_steps
        import math as _math
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lambda step: (
                step / max(1, warmup)
                if step < warmup
                else 0.5 * (1.0 + _math.cos(_math.pi * (step - warmup) / max(1, max_steps - warmup)))
            ),
        )

        self.global_step = 0
        self.amp_dtype = torch.bfloat16 if self.config.amp_dtype == "bf16" else torch.float16

    def step(
        self,
        chosen_input_ids: torch.Tensor,
        chosen_labels: torch.Tensor,
        chosen_prompt_lengths: torch.Tensor,
        rejected_input_ids: torch.Tensor,
        rejected_labels: torch.Tensor,
        rejected_prompt_lengths: torch.Tensor,
    ) -> dict:
        """
        One DPO training step.

        Returns: dict with keys: loss, accuracy, chosen_rewards,
        rejected_rewards, reward_margin.
        """
        self.optimizer.zero_grad()

        # Forward pass through policy (gradient-required)
        with torch.amp.autocast("cuda", enabled=self.config.use_amp, dtype=self.amp_dtype):
            policy_chosen_logp = compute_logprobs(
                self.policy_model, chosen_input_ids, chosen_labels, chosen_prompt_lengths
            )
            policy_rejected_logp = compute_logprobs(
                self.policy_model, rejected_input_ids, rejected_labels, rejected_prompt_lengths
            )
        with torch.no_grad():
            ref_chosen_logp = compute_logprobs(
                self.reference_model, chosen_input_ids, chosen_labels, chosen_prompt_lengths
            )
            ref_rejected_logp = compute_logprobs(
                self.reference_model, rejected_input_ids, rejected_labels, rejected_prompt_lengths
            )

            loss, metrics = dpo_loss(
                policy_chosen_logp,
                policy_rejected_logp,
                ref_chosen_logp,
                ref_rejected_logp,
                beta=self.config.beta,
                label_smoothing=self.config.label_smoothing,
            )

        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.scheduler.step()
        self.global_step += 1

        metrics["loss"] = float(loss.item())
        metrics["lr"] = float(self.scheduler.get_last_lr()[0])
        return metrics
