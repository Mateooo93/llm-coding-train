"""
Grouped Query Attention (GQA) with Rotary Position Embeddings (RoPE).

GQA balances the quality of Multi-Head Attention with the KV-cache efficiency
of Multi-Query Attention by sharing KV heads across groups of query heads.
This is the standard choice for modern LLMs (Llama 2/3, Mistral, Gemma).

NOTE: KV-cache support is planned as a followup. For now, this module is
intentionally kept simple with single forward calls; the inference techniques
in src/inference/ work on top of the simple model.generate() loop.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attn_res import apply_rope, RoPECache


class GQAAttention(nn.Module):
    """
    Grouped Query Attention with RoPE.

    - num_attention_heads query heads (Q)
    - num_key_value_heads KV heads (K, V), shared across query head groups
    - RoPE applied to Q and K for positional encoding
    - Causal masking for autoregressive decoding
    - Uses PyTorch's SDPA with Flash Attention backends when available
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        rope_theta: float = 10000.0,
        max_position_embeddings: int = 2048,
        attention_dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        self.num_heads_per_group = num_attention_heads // num_key_value_heads
        self.q_proj_dim = num_attention_heads * head_dim
        self.kv_proj_dim = num_key_value_heads * head_dim

        self.q_proj = nn.Linear(hidden_size, self.q_proj_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.kv_proj_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.kv_proj_dim, bias=False)
        self.o_proj = nn.Linear(self.q_proj_dim, hidden_size, bias=False)

        self.dropout = attention_dropout

        self._rope_cache = RoPECache(
            head_dim=self.head_dim,
            theta=self.rope_theta,
            max_position_embeddings=self.max_position_embeddings,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch, seq_len, hidden_size]
            attention_mask: padding mask [batch, seq_len] or [batch, seq_len, seq_len].
            position_ids: [batch, seq_len] (optional, for explicit positions during incremental decoding).

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

        cos, sin = self._rope_cache.get(seq_len, hidden_states.device, hidden_states.dtype)
        q, k = apply_rope(q, k, cos, sin)

        if self.num_heads_per_group > 1:
            k = k.repeat_interleave(self.num_heads_per_group, dim=1)
            v = v.repeat_interleave(self.num_heads_per_group, dim=1)

        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"),
                       device=hidden_states.device, dtype=hidden_states.dtype),
            diagonal=1,
        )

        if attention_mask is not None:
            if attention_mask.dim() == 2:
                padding_mask = attention_mask[:, None, None, :]
                causal_mask = causal_mask + padding_mask
            elif attention_mask.dim() == 3:
                causal_mask = causal_mask + attention_mask

        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=causal_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.q_proj_dim)

        return self.o_proj(attn_output)
