"""
Mixture-of-Experts (MoE) Layer — for future scaling.

MoE allows scaling total parameter count while keeping active compute per token
low by routing each token to only a subset of experts (top-k).

The Kimi Linear architecture (48B total / 3B active) uses MoE + AttnRes.
This implementation provides a standard sparse MoE with:
  - Router/gating network: linear projection to num_experts logits
  - Top-k selection: pick top-k experts per token
  - Expert MLPs: independent SwiGLU MLPs per expert
  - Load balancing loss: auxiliary loss to encourage uniform expert usage

This is a stub for future use — the prototype uses dense MLPs.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mlp import SwiGLUMLP


class MoELayer(nn.Module):
    """
    Sparse Mixture-of-Experts layer with top-k routing.

    Each token is routed to the top-k highest-scoring experts out of N total.
    A load balancing auxiliary loss encourages uniform expert utilization.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int = 8,
        num_experts_per_tok: int = 2,
        router_init_std: float = 0.02,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok

        # Router/gating network: produces a score for each expert
        self.router = nn.Linear(hidden_size, num_experts, bias=False)
        nn.init.normal_(self.router.weight, mean=0.0, std=router_init_std)

        # Expert MLPs — each is an independent SwiGLU
        self.experts = nn.ModuleList([
            SwiGLUMLP(hidden_size, intermediate_size)
            for _ in range(num_experts)
        ])

        # Store the last computed load balancing loss for the trainer to use
        self.aux_loss: torch.Tensor = torch.tensor(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, hidden_size]

        Returns:
            output: [batch, seq_len, hidden_size]
        """
        batch_size, seq_len, hidden_size = x.shape
        # Flatten to [num_tokens, hidden_size] for routing
        flat_x = x.view(-1, hidden_size)  # [B*T, D]
        num_tokens = flat_x.shape[0]

        # Compute router scores: [num_tokens, num_experts]
        router_logits = self.router(flat_x)

        # Top-k selection
        top_k_logits, top_k_indices = torch.topk(
            router_logits, self.num_experts_per_tok, dim=-1
        )
        # Softmax over selected experts only
        top_k_weights = F.softmax(top_k_logits, dim=-1)  # [num_tokens, k]

        # Compute load balancing auxiliary loss
        # Encourages uniform routing by penalizing deviation from uniform distribution
        # See Switch Transformer / GShard for details
        # Compute load balancing auxiliary loss
        # Encourages uniform routing by penalizing deviation from uniform distribution
        # See Switch Transformer / GShard for details
        # NOTE: router_prob must be computed OUTSIDE no_grad so gradients flow
        # to the router through the aux_loss. Only the token counts are detached.
        with torch.no_grad():
            # Fraction of tokens routed to each expert (detached — no grad needed)
            expert_mask = F.one_hot(
                top_k_indices, self.num_experts
            ).sum(dim=1)  # [num_tokens, num_experts]
            tokens_per_expert = expert_mask.sum(dim=0)  # [num_experts]
            fraction_per_expert = tokens_per_expert.float() / num_tokens
        # router_prob needs gradients to train the router
        router_prob = F.softmax(router_logits, dim=-1).mean(dim=0)  # [num_experts]
        # Load balancing loss: N * Σ(f_i * P_i) where f_i = fraction of tokens per expert
        # and P_i = average router probability for expert i
        # Minimized when all f_i and P_i are 1/N (uniform)
        self.aux_loss = self.num_experts * torch.sum(
            fraction_per_expert * router_prob
        )

        # Dispatch tokens to experts and combine results
        # For efficiency, we process each expert's assigned tokens in a batch
        output = torch.zeros_like(flat_x)

        for expert_idx in range(self.num_experts):
            # Find tokens routed to this expert: [num_assigned]
            token_indices = torch.where(top_k_indices == expert_idx)[0]
            if token_indices.numel() == 0:
                continue

            # Find which slot (0..k-1) this expert was selected in
            slot_mask = (top_k_indices == expert_idx)  # [num_tokens, k]
            slot_indices = slot_mask.nonzero(as_tuple=True)[1]  # which slot per token

            # Get the expert weights for these tokens
            # top_k_weights[token_indices, slot_indices] gives the weight
            expert_weights = top_k_weights[token_indices, slot_indices]  # [num_assigned]

            # Process through the expert MLP
            expert_input = flat_x[token_indices]  # [num_assigned, D]
            expert_output = self.experts[expert_idx](expert_input)  # [num_assigned, D]

            # Scale by routing weight and scatter back
            output[token_indices] += expert_weights.unsqueeze(-1) * expert_output

        return output.view(batch_size, seq_len, hidden_size)
