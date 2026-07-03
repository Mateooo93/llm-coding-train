"""
Attention Residuals (AttnRes) — the core innovation.

Standard PreNorm transformers accumulate layer outputs with fixed unit weights:
    h_l = h_{l-1} + Layer(h_{l-1})

This causes uncontrolled hidden-state growth with depth, progressively diluting
each layer's contribution. AttnRes replaces this with softmax attention over
preceding layer outputs, allowing each layer to selectively aggregate earlier
representations with learned, input-dependent weights.

Block AttnRes partitions layers into blocks and attends over block-level
representations, reducing the memory footprint from O(L) to O(N) while
preserving most of the gains.

Reference: "Attention Residuals" (Chen et al., Moonshot AI, 2026)
           arXiv:2603.15031
           https://github.com/MoonshotAI/Attention-Residuals
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    Unlike LayerNorm, RMSNorm does not center the activations (no mean subtraction),
    which provides equivalent training performance with lower computational overhead.
    Standard choice in modern LLMs (Llama, Mistral, Gemma).
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return (self.weight * hidden_states.to(input_dtype)).to(input_dtype)


def precompute_rope_frequencies(
    head_dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Precompute the rotation frequencies for RoPE.

    Returns a tensor of shape [max_seq_len, head_dim] containing the
    cos/sin frequencies for each position.

    RoPE encodes relative position by rotating query/key vectors:
        q_rotated = q * cos(theta) + rotate_half(q) * sin(theta)
    """
    assert head_dim % 2 == 0, f"head_dim must be even, got {head_dim}"

    # Frequency for each dimension pair: theta_i = 1 / (theta^(2i/d))
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device, dtype=dtype) / head_dim))
    # Position indices: [0, 1, ..., max_seq_len-1]
    positions = torch.arange(max_seq_len, device=device, dtype=dtype)
    # Outer product: [max_seq_len, head_dim/2]
    freqs = torch.outer(positions, inv_freq)
    # Duplicate to match head_dim: [max_seq_len, head_dim]
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb


class RoPECache:
    """
    Shared RoPE cache for attention modules.

    Lazily computes and caches cos/sin tensors for RoPE. Shared between
    GQAAttention and LinearAttention to avoid code duplication and
    prevent the two implementations from diverging.
    """

    def __init__(self, head_dim: int, theta: float = 10000.0, max_position_embeddings: int = 2048):
        self.head_dim = head_dim
        self.theta = theta
        self.max_position_embeddings = max_position_embeddings
        self._cache: Optional[tuple[torch.Tensor, torch.Tensor]] = None
        self._device: Optional[torch.device] = None

    def get(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        """
        Get cached cos/sin tensors, computing them if needed.

        Returns: (cos, sin) each of shape [seq_len, head_dim].
        """
        if (
            self._cache is not None
            and self._device == device
            and self._cache[0].shape[0] >= seq_len
        ):
            cos, sin = self._cache
            return cos[:seq_len].to(dtype), sin[:seq_len].to(dtype)

        freqs = precompute_rope_frequencies(
            self.head_dim,
            max(self.max_position_embeddings, seq_len),
            theta=self.theta,
            device=device,
            dtype=torch.float32,
        )
        cos, sin = freqs.cos(), freqs.sin()
        self._cache = (cos, sin)
        self._device = device
        return cos[:seq_len].to(dtype), sin[:seq_len].to(dtype)


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply Rotary Position Embeddings to query and key tensors.

    Args:
        q: [batch, num_heads, seq_len, head_dim]
        k: [batch, num_kv_heads, seq_len, head_dim]
        cos: [seq_len, head_dim]
        sin: [seq_len, head_dim]

    Returns:
        Rotated q and k tensors with same shapes.
    """
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        """Rotate the second half of the last dimension."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    # Reshape cos/sin for broadcasting: [1, 1, seq_len, head_dim]
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)

    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot


class BlockAttnRes(nn.Module):
    """
    Block Attention Residuals — the core innovation.

    Replaces fixed unit-weight residual connections with learned, input-dependent
    softmax attention over preceding block-level representations.

    Within each block, standard additive residuals are used (intra-block).
    Across blocks, learned attention weights selectively aggregate earlier
    representations (inter-block), giving each layer the ability to "choose"
    which earlier representations to draw from.

    The mechanism uses a learnable pseudo-query vector w_l ∈ R^d for each
    sublayer. The pseudo-query computes attention scores over normalized
    block representations, and softmax produces the aggregation weights.

    Memory: O(N * d) per token where N = num_blocks (vs O(L * d) for full AttnRes)
    Compute: O(N * d) per sublayer — negligible compared to attention/MLP
    """

    def __init__(self, hidden_size: int, init_scale: float = 0.02, eps: float = 1e-6):
        super().__init__()
        # Pseudo-query vector: a single learnable vector that "queries" all
        # preceding block representations to determine aggregation weights.
        # Implemented as a Linear(d, 1) for efficiency, then squeeze.
        self.proj = nn.Linear(hidden_size, 1, bias=False)

        # Normalization applied to block representations before computing
        # attention scores. This stabilizes the pseudo-query dot product.
        self.norm = RMSNorm(hidden_size, eps=eps)

        # Initialize pseudo-query with small values for stable early training
        nn.init.normal_(self.proj.weight, mean=0.0, std=init_scale)

        self.hidden_size = hidden_size

    def forward(
        self,
        block_reps: List[torch.Tensor],
        partial_block: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute the aggregated hidden state using Block AttnRes.

        Args:
            block_reps: List of N tensors, each [batch, seq_len, hidden_size].
                        These are completed block-level representations from
                        preceding blocks.
            partial_block: [batch, seq_len, hidden_size] or None — the current
                           intra-block partial sum. None at the start of a new
                           block (before any sublayer output has been added).

        Returns:
            aggregated: [batch, seq_len, hidden_size] — the weighted sum of
                        all block representations (plus partial if present),
                        with softmax attention weights.

        The aggregation formula:
            V = stack(block_reps + [partial_block])    # [N+1, B, T, D]
            K = RMSNorm(V)                              # normalized keys
            logits = einsum('d, nbtd -> nbt', w, K)     # pseudo-query dot product
            alpha = softmax(logits, dim=0)              # attention weights over blocks
            h = einsum('nbt, nbtd -> btd', alpha, V)    # weighted aggregation
        """
        if partial_block is None:
            # Start of a new block with no partial accumulation yet.
            # Attend only over completed block representations.
            if len(block_reps) == 0:
                raise RuntimeError(
                    "BlockAttnRes has no block_reps and partial_block is None — "
                    "the first sublayer must have an initial hidden state."
                )
            V = torch.stack(block_reps, dim=0)  # [N, B, T, D]
        elif len(block_reps) == 0:
            # No preceding blocks — just return the partial block as-is
            return partial_block
        else:
            # Stack all representations: [N+1, B, T, D]
            all_reps = block_reps + [partial_block]
            V = torch.stack(all_reps, dim=0)

        # Normalize for stable attention scores
        K = self.norm(V)  # [N(+1), B, T, D]

        # Compute attention logits using the pseudo-query vector
        # proj.weight has shape [1, D], squeeze to get w ∈ R^D
        w = self.proj.weight.squeeze(0)  # [D]

        # logits[n, b, t] = w · K[n, b, t, :]  →  [N(+1), B, T]
        logits = torch.einsum("d,nbtd->nbt", w, K)

        # Softmax over the block dimension (dim=0) — each position independently
        # chooses how much to weight each preceding block
        alpha = F.softmax(logits, dim=0)  # [N(+1), B, T]

        # Weighted aggregation: h[b, t, d] = Σ_n alpha[n, b, t] * V[n, b, t, d]
        aggregated = torch.einsum("nbt,nbtd->btd", alpha, V)  # [B, T, D]

        return aggregated


class ResidualState:
    """
    Manages the residual connection state across layers.

    For standard residuals: simply passes h_{l-1} + Layer(h_{l-1}).
    For AttnRes: maintains a list of completed block representations
    and the current intra-block partial sum.

    This is not an nn.Module — it's a lightweight state container that
    each transformer block reads from and writes to during the forward pass.
    """

    def __init__(self, initial_hidden: torch.Tensor, use_attn_res: bool):
        self.use_attn_res = use_attn_res
        self.block_reps: List[torch.Tensor] = []  # completed block representations
        self.partial_block: torch.Tensor = initial_hidden  # current intra-block accumulation
        self.sublayer_idx: int = 0  # counts sublayers (attn + mlp) within current block

    def add_output(self, output: torch.Tensor):
        """
        Add a sublayer output to the residual stream.

        If partial_block is None (start of a new block), the output becomes
        the initial partial_block. Otherwise, standard additive accumulation.
        """
        if self.partial_block is None:
            self.partial_block = output
        else:
            self.partial_block = self.partial_block + output

    def maybe_close_block(self, sublayers_per_block: int):
        """
        Check if we've completed a block and save its representation.

        After saving, partial_block is reset to None so the next block starts
        fresh — preventing double-counting of the same representation as both
        a completed block and the current partial.
        """
        self.sublayer_idx += 1
        if self.use_attn_res and self.sublayer_idx >= sublayers_per_block:
            # Save the completed block representation
            self.block_reps.append(self.partial_block)
            # Reset partial_block so the next block starts fresh from the
            # next sublayer's output (matching the paper's pseudocode)
            self.partial_block = None
            self.sublayer_idx = 0

    def get_final_hidden(self) -> torch.Tensor:
        """Get the final hidden state after all layers."""
        # If partial_block is None (block just closed at the very end),
        # fall back to the last completed block representation.
        if self.partial_block is not None:
            return self.partial_block
        if len(self.block_reps) > 0:
            return self.block_reps[-1]
        raise RuntimeError("No hidden state available — model produced no output.")
