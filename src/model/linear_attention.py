"""
Linear Attention — O(L) sub-quadratic attention using feature map approximation.

Replaces the softmax kernel with a feature map φ(·) = ELU(x) + 1, allowing the
attention computation to be reordered via matrix multiplication associativity.

This implementation materializes a [B, H, T, T] scores matrix for the prototype,
so the forward pass is O(L²·d) — same complexity as standard attention minus
the softmax overhead. To achieve true O(L·d²) complexity, a chunkwise parallel
implementation with custom Triton kernels is needed (see flash-linear-attention).

The recurrent view enables O(1) inference state (no growing KV-cache):
    S_i = S_{i-1} + φ(k_i)^T · v_i
    o_i = φ(q_i) · S_i / z_i
    z_i = z_{i-1} + φ(k_i)^T

NOTE: KV-cache / recurrent-state incremental inference is planned as a followup.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attn_res import apply_rope, RoPECache


def elu_plus_one(x: torch.Tensor) -> torch.Tensor:
    """
    ELU+1 feature map: φ(x) = ELU(x) + 1.

    Ensures non-negativity (required for linear attention to work as a kernel
    approximation of softmax). With α=1, ELU(x)+1 is always ≥ 0 and smooth.
    """
    return F.elu(x) + 1.0


class LinearAttention(nn.Module):
    """
    Linear Attention with GQA support and optional RoPE.

    Predictable complexity ceiling: O(L² · d) in the current implementation
    (materializes T×T matrix). True O(L · d²) requires chunkwise parallel
    kernels (see https://github.com/fla-org/flash-linear-attention).
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        use_rope: bool = False,
        rope_theta: float = 10000.0,
        max_position_embeddings: int = 2048,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.use_rope = use_rope
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        self.num_heads_per_group = num_attention_heads // num_key_value_heads
        self.q_proj_dim = num_attention_heads * head_dim
        self.kv_proj_dim = num_key_value_heads * head_dim

        self.q_proj = nn.Linear(hidden_size, self.q_proj_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.kv_proj_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.kv_proj_dim, bias=False)
        self.o_proj = nn.Linear(self.q_proj_dim, hidden_size, bias=False)

        if self.use_rope:
            self._rope_cache = RoPECache(
                head_dim=self.head_dim,
                theta=self.rope_theta,
                max_position_embeddings=self.max_position_embeddings,
            )
        else:
            self._rope_cache = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch, seq_len, hidden_size]
            attention_mask: padding mask [batch, seq_len].
            position_ids: explicit position IDs (optional, for RoPE).

        Returns:
            output: [batch, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = q.view(batch_size, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if self.use_rope:
            cos, sin = self._rope_cache.get(seq_len, hidden_states.device, hidden_states.dtype)
            q, k = apply_rope(q, k, cos, sin)

        q = elu_plus_one(q)
        k = elu_plus_one(k)

        if self.num_heads_per_group > 1:
            k = k.repeat_interleave(self.num_heads_per_group, dim=1)
            v = v.repeat_interleave(self.num_heads_per_group, dim=1)

        q = q.transpose(-1, -2)
        k = k.transpose(-1, -2)

        k_cumsum = torch.cumsum(k, dim=-1)
        scores = torch.matmul(q.transpose(-1, -2), k)
        causal_mask = torch.tril(
            torch.ones(seq_len, seq_len, device=hidden_states.device, dtype=torch.bool)
        )
        scores = scores.masked_fill(~causal_mask, 0.0)

        if attention_mask is not None and attention_mask.dim() == 2:
            pad_mask = attention_mask[:, None, None, :].to(torch.bool)
            scores = scores.masked_fill(~pad_mask, 0.0)

        normalizer = (q * k_cumsum).sum(dim=-2).clamp(min=1e-6)
        output = torch.matmul(scores, v) / normalizer.unsqueeze(-1)

        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len, self.q_proj_dim)
        return self.o_proj(output)
