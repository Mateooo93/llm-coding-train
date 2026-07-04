"""
AttnRes Language Model — the full decoder-only transformer with Block AttnRes.

Architecture:
    Input IDs → Token Embedding → [AttnRes Transformer Blocks] → Final RMSNorm → LM Head → Logits

Each transformer block uses Block AttnRes instead of standard residual connections,
allowing layers to selectively aggregate earlier representations with learned,
input-dependent softmax attention weights.

The model supports:
  - Gradient checkpointing for memory-efficient training
  - Tied word embeddings (share weights between input embedding and LM head)
  - Dense MLPs or Mixture-of-Experts (MoE) for future scaling
  - HuggingFace-compatible save/load
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import AttnResConfig
from .attn_res import RMSNorm, ResidualState
from .transformer_block import AttnResTransformerBlock

# Importing the HF wrapper registers `AttnresLMConfigHF` /
# `AttnResLMForCausalLM` with `transformers.AutoConfig` so HF can route
# `from_pretrained()` calls without `trust_remote_code=True`. The wrapper
# is a thin adaptor — `AttnResLM` stays the canonical class.
from . import hf_wrapper  # noqa: F401


@dataclass
class AttnResOutput:
    """Output from the AttnRes LM forward pass."""
    logits: torch.Tensor  # [batch, seq_len, vocab_size]
    loss: Optional[torch.Tensor] = None  # scalar, if labels provided
    aux_loss: Optional[torch.Tensor] = None  # MoE load balancing loss


class AttnResLM(nn.Module):
    """
    Full AttnRes Language Model.

    A decoder-only transformer that replaces standard residual connections with
    Block Attention Residuals (AttnRes) — the core architectural innovation.

    Usage:
        config = AttnResConfig(...)
        model = AttnResLM(config)
        output = model(input_ids, labels=labels)
    """

    def __init__(self, config: AttnResConfig):
        super().__init__()
        self.config = config

        # Token embedding
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        nn.init.normal_(self.embed_tokens.weight, mean=0.0, std=0.02)

        # Embedding dropout
        self.embed_dropout = nn.Dropout(config.hidden_dropout)

        # Transformer blocks
        self.layers = nn.ModuleList([
            AttnResTransformerBlock(config, layer_idx=i)
            for i in range(config.num_hidden_layers)
        ])

        # Final normalization before LM head
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # LM head (tied with embeddings if configured)
        if config.tie_word_embeddings:
            self.lm_head = None  # use embed_tokens.weight
        else:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
            nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)

        # Gradient checkpointing is supported alongside AttnRes via the
        # per-block snapshot/restore pattern in AttnResTransformerBlock.forward
        # (use_reentrant=False). The flag is mutable at runtime via
        # gradient_checkpointing_enable() / disable() — it's NOT permanently
        # disabled when use_attn_res=True (an earlier version had a guard here,
        # but the per-block checkpoint refactor removes the original concern).
        self.gradient_checkpointing = config.gradient_checkpointing
        # State mirrors to each block so AttnResTransformerBlock.forward can
        # decide locally whether to wrap its compute in checkpoint().
        for layer in self.layers:
            layer._gc_enabled = self.gradient_checkpointing

        # Initialize weights (called AFTER all modules are created, so it
        # can selectively skip modules that have custom init like AttnRes)
        self._init_weights()

    def gradient_checkpointing_enable(self) -> None:
        """Enable gradient checkpointing at runtime (memory-efficient training)."""
        self.gradient_checkpointing = True
        for layer in self.layers:
            layer._gc_enabled = True

    def gradient_checkpointing_disable(self) -> None:
        """Disable gradient checkpointing at runtime."""
        self.gradient_checkpointing = False
        for layer in self.layers:
            layer._gc_enabled = False

    def _init_weights(self):
        """
        Initialize weights with small normal values, ones for norms.

        Skips AttnRes projection layers — they have their own custom init
        (attn_res_init_scale) set in BlockAttnRes.__init__.
        """
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                # Skip modules with custom init: AttnRes pseudo-query projections
                # (attn_res_init_scale) and MoE router (router_init_std)
                if "attn_res" in name or "mlp_res" in name or "router" in name:
                    continue
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_lm_head_weight(self) -> torch.Tensor:
        """Get the LM head weight matrix (tied or untied)."""
        if self.lm_head is not None:
            return self.lm_head.weight
        return self.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[AttnResOutput, tuple]:
        """
        Args:
            input_ids: [batch, seq_len] token IDs
            attention_mask: [batch, seq_len] — 1 for real tokens, 0 for padding
            position_ids: [batch, seq_len] — explicit position IDs (optional)
            labels: [batch, seq_len] — target token IDs for loss computation
            return_dict: if True, return AttnResOutput; else return tuple

        Returns:
            AttnResOutput with logits, loss, and optional aux_loss
        """
        batch_size, seq_len = input_ids.shape

        # Position IDs
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)

        # Token embeddings
        hidden_states = self.embed_tokens(input_ids)  # [B, T, D]
        hidden_states = self.embed_dropout(hidden_states)

        # Initialize residual state — tracks block representations for AttnRes
        residual_state = ResidualState(
            initial_hidden=hidden_states,
            use_attn_res=self.config.use_attn_res,
        )

        # Process through all transformer blocks
        # The residual state is threaded through all layers, accumulating
        # block representations and maintaining the intra-block partial sum.
        total_aux_loss = torch.tensor(0.0, device=hidden_states.device, dtype=hidden_states.dtype)

        for layer in self.layers:
            # Gradient checkpointing (when enabled on the layer) is delegated to
            # AttnResTransformerBlock.forward, which wraps `_compute_block` with
            # torch.utils.checkpoint.checkpoint(..., use_reentrant=False). This
            # allows GC to coexist with AttnRes despite ResidualState mutation,
            # because the per-block helper is a pure functional on tensors.
            layer(residual_state, attention_mask=attention_mask, position_ids=position_ids)

            # Accumulate MoE auxiliary loss
            if hasattr(layer.mlp, 'aux_loss'):
                total_aux_loss = total_aux_loss + layer.mlp.aux_loss

        # Final hidden state and normalization
        final_hidden = residual_state.get_final_hidden()
        final_hidden = self.norm(final_hidden)

        # LM head → logits
        lm_head_weight = self.get_lm_head_weight()
        logits = F.linear(final_hidden, lm_head_weight)  # [B, T, vocab_size]

        # Compute loss if labels are provided
        loss = None
        if labels is not None:
            # Shift: predict next token from current
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        # Average aux loss across layers
        if self.config.use_moe and total_aux_loss.item() != 0.0:
            aux_loss = total_aux_loss / max(1, self.config.num_hidden_layers)
        else:
            aux_loss = None

        if not return_dict:
            output = (logits,)
            if loss is not None:
                output += (loss,)
            if aux_loss is not None:
                output += (aux_loss,)
            return output

        return AttnResOutput(logits=logits, loss=loss, aux_loss=aux_loss)

    def num_parameters(self, exclude_embeddings: bool = False) -> int:
        """Count total parameters."""
        if exclude_embeddings:
            return sum(p.numel() for n, p in self.named_parameters()
                       if "embed" not in n and "lm_head" not in n)
        return sum(p.numel() for p in self.parameters())

    def save_pretrained(self, path: str):
        """Save model in a HuggingFace-compatible format."""
        import os
        import json
        os.makedirs(path, exist_ok=True)
        # Save config as JSON
        from dataclasses import asdict
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(asdict(self.config), f, indent=2)
        # Save model state dict
        torch.save(self.state_dict(), os.path.join(path, "pytorch_model.bin"))

    @classmethod
    def from_pretrained(cls, path: str) -> "AttnResLM":
        """Load model from a saved checkpoint."""
        import os
        import json
        with open(os.path.join(path, "config.json")) as f:
            config_dict = json.load(f)
        config = AttnResConfig(**config_dict)
        model = cls(config)
        state_dict = torch.load(os.path.join(path, "pytorch_model.bin"), map_location="cpu")
        model.load_state_dict(state_dict)
        return model
