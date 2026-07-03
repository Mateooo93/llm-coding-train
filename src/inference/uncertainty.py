"""
Uncertainty-Aware Decoding — entropy-based hallucination detection.

Provides two complementary signals for estimating the model's confidence:

  1. **Entropy** of the next-token distribution: high entropy = uncertain.
  2. **Monte Carlo Dropout** confidence interval: run multiple forward passes
     with dropout enabled and measure variance of probabilities.

When either signal exceeds a configurable threshold, the caller can:
  - re-sample the token (default: rejection sampling),
  - surface a "low confidence" flag for argument-tuning,
  - request RAG/context lookup.

Reference: Survey on LLM Hallucinations (Huang et al., 2025), section on
"Uncertainty-Aware Decoding" and "Monte Carlo Dropout at inference".
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass
class UncertaintySignal:
    """Per-token uncertainty estimate."""
    entropy: float  # Shannon entropy of next-token distribution
    top1_prob: float
    top1_token: int
    is_uncertain: bool  # True if above threshold


def entropy_of_distribution(logits: torch.Tensor) -> torch.Tensor:
    """
    Compute Shannon entropy (in nats) of the categorical distribution.

    Args:
        logits: [batch, vocab_size] or [batch, seq_len, vocab_size].

    Returns:
        entropy: [batch] (or [batch, seq_len]).
    """
    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum(dim=-1)
    return entropy


def evaluate_uncertainty(
    logits: torch.Tensor,
    entropy_threshold: float = 2.5,  # nats (max = ln(vocab))
) -> UncertaintySignal:
    """
    Single-pass entropy-based uncertainty estimate.

    Args:
        logits: [batch, vocab_size] (last-token logits for autoregressive).
        entropy_threshold: above this, mark as uncertain.

    Returns:
        UncertaintySignal with entropy, top1 prob, top1 token, and is_uncertain.
    """
    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum(dim=-1)
    top1_prob, top1_token = probs.max(dim=-1)

    # For batch=1 (generation) this returns scalar ints
    is_uncertain = bool(entropy[0].item() > entropy_threshold)
    return UncertaintySignal(
        entropy=float(entropy[0].item()),
        top1_prob=float(top1_prob[0].item()),
        top1_token=int(top1_token[0].item()),
        is_uncertain=is_uncertain,
    )


def monte_carlo_uncertainty(
    model,
    input_ids: torch.Tensor,
    n_samples: int = 5,
    entropy_threshold: float = 2.5,
) -> dict:
    """
    Monte Carlo dropout uncertainty estimate.

    Activates model dropout, runs ``n_samples`` forward passes, and measures
    variance of the next-token distribution. Higher variance = less confident.

    Args:
        model: the AttnResLM model (must have dropout layers).
        input_ids: [batch, seq_len] input ids.
        n_samples: number of MC forward passes (5–10 is typical).
        entropy_threshold: threshold for marking uncertain.

    Returns:
        dict with keys: mean_entropy, std_top_token, mean_top_token, is_uncertain.
    """
    if n_samples < 2:
        raise ValueError("n_samples must be >= 2 for Monte Carlo uncertainty")

    model.train()  # enable dropout
    probs_history = []
    with torch.no_grad():
        for _ in range(n_samples):
            output = model(input_ids)
            last_logits = output.logits[:, -1, :]
            probs = F.softmax(last_logits, dim=-1)
            probs_history.append(probs)
    model.eval()

    stacked = torch.stack(probs_history, dim=0)  # [n_samples, batch, vocab]
    mean_probs = stacked.mean(dim=0)

    log_mean = torch.log(mean_probs.clamp(min=1e-12))
    mean_entropy = float((-(mean_probs * log_mean).sum(dim=-1)).mean().item())

    argmaxes = stacked.argmax(dim=-1)  # [n_samples, batch]
    most_common = None
    if argmaxes.shape[1] == 1:
        # Single-sample batch: get the most common predicted token
        most_common = int(argmaxes.flatten().mode().values.item())
        agreement = float((argmaxes[:, 0] == most_common).float().mean().item())
    else:
        # Batched: average agreement of each sample's argmax with the first sample
        agreement = float(
            (argmaxes == argmaxes[0:1]).all(dim=-1).float().mean().item()
        )

    is_uncertain = mean_entropy > entropy_threshold or agreement < 0.6
    return {
        "mean_entropy": mean_entropy,
        "top_token_agreement": agreement,
        "most_common_token": most_common,
        "is_uncertain": is_uncertain,
    }
