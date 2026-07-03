"""
Transformer block integrating Block AttnRes.

Each block contains:
  1. Self-attention sublayer (GQA + RoPE)
  2. MLP sublayer (SwiGLU)

With AttnRes, the residual connection is replaced by Block AttnRes:
  - Before each sublayer, Block AttnRes aggregates the current partial block
    with all preceding completed block representations.
  - After each sublayer, the output is added to the intra-block partial sum.
  - At block boundaries, the completed block representation is saved.

Without AttnRes (use_attn_res=False), standard PreNorm residuals are used:
  h = h + Attn(Norm(h))
  h = h + MLP(Norm(h))
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional

from .attention import GQAAttention
from .linear_attention import LinearAttention
from .mlp import SwiGLUMLP
from .attn_res import BlockAttnRes, RMSNorm, ResidualState
from .config import AttnResConfig


class AttnResTransformerBlock(nn.Module):
    """
    A single transformer layer with Block AttnRes integration.

    The block applies attention and MLP sublayers. When AttnRes is enabled,
    BlockAttnRes modules provide the residual aggregation. When disabled,
    standard additive residuals are used.
    """

    def __init__(self, config: AttnResConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        # Sublayer normalizations (PreNorm)
        self.attn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Attention sublayer — type depends on config and layer index
        # In hybrid mode, every hybrid_full_attention_interval-th layer uses
        # full softmax attention; the rest use linear attention (O(L) complexity).
        # This mirrors Kimi Linear's architecture: full attention for precise
        # retrieval, linear attention for efficient long-context processing.
        use_full_attention = True
        if config.attention_type == "linear":
            use_full_attention = False
        elif config.attention_type == "hybrid":
            # Every Nth layer uses full attention, others use linear
            use_full_attention = (layer_idx % config.hybrid_full_attention_interval == 0)

        if use_full_attention:
            self.attn = GQAAttention(
                hidden_size=config.hidden_size,
                num_attention_heads=config.num_attention_heads,
                num_key_value_heads=config.num_key_value_heads,
                head_dim=config.head_dim,
                rope_theta=config.rope_theta,
                max_position_embeddings=config.max_position_embeddings,
                attention_dropout=config.attention_dropout,
            )
            self.is_linear_attention = False
        else:
            self.attn = LinearAttention(
                hidden_size=config.hidden_size,
                num_attention_heads=config.num_attention_heads,
                num_key_value_heads=config.num_key_value_heads,
                head_dim=config.head_dim,
                use_rope=config.linear_attention_use_rope,
                rope_theta=config.rope_theta,
                max_position_embeddings=config.max_position_embeddings,
            )
            self.is_linear_attention = True

        # MLP sublayer (SwiGLU or MoE)
        if config.use_moe and layer_idx % config.moe_layer_interval == 0:
            from .moe import MoELayer
            self.mlp = MoELayer(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                num_experts=config.num_experts,
                num_experts_per_tok=config.num_experts_per_tok,
            )
        else:
            self.mlp = SwiGLUMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
            )

        # AttnRes modules — one for each sublayer (attention + MLP)
        # Each has its own learnable pseudo-query vector
        if config.use_attn_res:
            self.attn_res = BlockAttnRes(
                hidden_size=config.hidden_size,
                init_scale=config.attn_res_init_scale,
                eps=config.rms_norm_eps,
            )
            self.mlp_res = BlockAttnRes(
                hidden_size=config.hidden_size,
                init_scale=config.attn_res_init_scale,
                eps=config.rms_norm_eps,
            )
        else:
            self.attn_res = None
            self.mlp_res = None

        # Gradient checkpointing toggle — mirrored from AttnResLM.forward pass.
        # When True (and self.training), `forward` wraps `_compute_block` in
        # torch.utils.checkpoint.checkpoint(..., use_reentrant=False). The use
        # of a pure-functional helper that does NOT mutate the input block_reps
        # list / sublayer_idx scalar is what makes checkpointing safe to use
        # alongside the otherwise-mutable ResidualState.
        self._gc_enabled: bool = False

    def _compute_block(
        self,
        initial_partial: Optional[torch.Tensor],
        block_reps: list,
        sublayer_idx: int,
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.Tensor],
    ) -> tuple[Optional[torch.Tensor], list, int]:
        """Run both sublayers (attn + MLP) as a pure function.

        Returns the new (partial_block, block_reps, sublayer_idx) tuple. The
        caller is responsible for writing these back to the live
        ResidualState. This function NEVER mutates its inputs, which is what
        makes it safe to wrap in torch.utils.checkpoint.checkpoint(...
        use_reentrant=False): on backward replay the original inputs are
        re-passed and the same computation is re-run deterministically.
        """
        # Local copies so we never touch the caller's list / int / tensor.
        cur_reps = list(block_reps)
        cur_partial = initial_partial
        cur_idx = int(sublayer_idx)

        # ── Attention sublayer ──────────────────────────────────
        if self.attn_res is not None:
            h_in = self.attn_res(cur_reps, cur_partial)
        else:
            h_in = cur_partial

        attn_out = self.attn(self.attn_norm(h_in), attention_mask=attention_mask, position_ids=position_ids)

        if cur_partial is None:
            cur_partial = attn_out
        else:
            cur_partial = cur_partial + attn_out

        cur_idx += 1
        if self.config.use_attn_res and cur_idx >= self.config.sublayers_per_block:
            cur_reps.append(cur_partial)
            cur_partial = None
            cur_idx = 0

        # ── MLP sublayer ────────────────────────────────────────
        if self.mlp_res is not None:
            h_in = self.mlp_res(cur_reps, cur_partial)
        else:
            h_in = cur_partial

        mlp_out = self.mlp(self.mlp_norm(h_in))

        if cur_partial is None:
            cur_partial = mlp_out
        else:
            cur_partial = cur_partial + mlp_out

        cur_idx += 1
        if self.config.use_attn_res and cur_idx >= self.config.sublayers_per_block:
            cur_reps.append(cur_partial)
            cur_partial = None
            cur_idx = 0

        return cur_partial, cur_reps, cur_idx

    def forward(
        self,
        residual_state: ResidualState,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Process this layer, updating the residual state.

        With gradient checkpointing enabled and self.training, the whole
        block (attn + MLP sublayers) is wrapped in
        torch.utils.checkpoint.checkpoint(..., use_reentrant=False). This
        drops all per-attention / per-MLP intermediates from the autograd
        graph between the block's input boundary and its output boundary,
        saving ~10x activation memory at the cost of ~30% slower backward.

        Without checkpointing, the layer runs the compute directly.

        Args:
            residual_state: The shared residual state across all layers.
            attention_mask: Optional attention mask for padding.
            position_ids: Optional position IDs for RoPE.
        """
        use_gc = bool(getattr(self, "_gc_enabled", False)) and self.training

        if use_gc:
            new_partial, new_reps, new_idx = torch.utils.checkpoint.checkpoint(
                self._compute_block,
                residual_state.partial_block,
                residual_state.block_reps,
                residual_state.sublayer_idx,
                attention_mask,
                position_ids,
                use_reentrant=False,
                preserve_rng_state=True,
            )
        else:
            new_partial, new_reps, new_idx = self._compute_block(
                residual_state.partial_block,
                residual_state.block_reps,
                residual_state.sublayer_idx,
                attention_mask,
                position_ids,
            )

        residual_state.partial_block = new_partial
        residual_state.block_reps = new_reps
        residual_state.sublayer_idx = new_idx
