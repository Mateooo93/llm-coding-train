"""
HuggingFace `PreTrainedModel` wrapper around our `AttnResLM`.

This wrapper exists so that the rest of the HuggingFace ecosystem treats
our custom architecture AS IF it were any other HF causal LM:

  - `AutoModelForCausalLM.from_pretrained("oars344/attnres-phase1",
        trust_remote_code=True)` returns a `~AttnResLMForCausalLM`
  - `bitsandbytes` 4-bit quantization can be applied via the standard
    `from_pretrained(..., quantization_config=...)` path
  - `peft.get_peft_model(...)` finds `nn.Linear` modules inside
    `self.model.layers[i].attn.{q,k,v,o}_proj`, `.mlp.{gate,up,down}_proj`
    for LoRA injection, plus the special `attn_res.proj` / `mlp_res.proj`
    pseudo-query projections inside `BlockAttnRes`.
  - `transformers.Trainer` accepts the model because `forward()` returns
    `CausalLMOutputWithPast` (with `loss` populated when labels are given).
  - `Trainer.push_to_hub()` on a PEFT model pushes ONLY the adapter
    weights (`adapter_model.safetensors` + `adapter_config.json`).

`AttnResConfig` (the dataclass in `src/model/config.py`) stays untouched
— it's the canonical Phase-1 / production config. We mirror its fields
into `AttnResLMConfigHF` (a `PretrainedConfig` subclass) for HF
serialization, and bridge back to `AttnResConfig` inside the wrapper.

The model code from `src/model/model.py` is NOT modified — this wrapper
purely *adapts* it. Phase 1's `AttnResLM.from_pretrained()` path keeps
working exactly as it does today.

Usage from the CLI side (see `src/training/train_phase2.py`):

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    base = AutoModelForCausalLM.from_pretrained(
        "oars344/attnres-phase1",
        trust_remote_code=True,
    )
    lora = LoraConfig(r=16, target_modules=[
        "q_proj","k_proj","v_proj","o_proj",
        "gate_proj","up_proj","down_proj",
        "attn_res.proj","mlp_res.proj",   # <-- Task-specific residual re-routing
    ], task_type="CAUSAL_LM")
    model = get_peft_model(base, lora)
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Optional

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    CausalLMOutputWithPast,
    PretrainedConfig,
    PreTrainedModel,
)


# Default AttnRes config dict (mirrors `src.model.config.AttnResConfig()` defaults).
_ATTNRES_DATACLASS_DEFAULTS: dict = dict(
    vocab_size=50304,
    hidden_size=768,
    intermediate_size=2048,
    num_hidden_layers=12,
    num_attention_heads=12,
    num_key_value_heads=4,
    head_dim=64,
    attention_type="full",
    hybrid_full_attention_interval=6,
    linear_attention_use_rope=True,
    rope_theta=10000.0,
    max_position_embeddings=2048,
    use_attn_res=True,
    attn_res_block_size=4,
    attn_res_init_scale=0.02,
    rms_norm_eps=1e-6,
    attention_dropout=0.0,
    hidden_dropout=0.0,
    tie_word_embeddings=True,
    use_moe=False,
    num_experts=8,
    num_experts_per_tok=2,
    moe_layer_interval=2,
    gradient_checkpointing=False,
    pad_token_id=0,
    bos_token_id=1,
    eos_token_id=2,
)


class AttnResLMConfigHF(PretrainedConfig):
    """
    `PretrainedConfig` mirror of the canonical `AttnResConfig` dataclass.

    We don't change `src.model.config.AttnResConfig` itself (Phase 1 still
    uses it directly) — we just expose all of its fields as HF config
    fields so `config.json` round-trips correctly through HF Hub I/O.

    The `to_attnres_config()` method bridges back to the dataclass for
    use by `AttnResLM.__init__`.
    """

    model_type: str = "attnres"

    def __init__(self, **kwargs):
        # Set every AttnRes field provided, falling back to dataclass defaults
        # for anything missing. This makes HF's "merge with defaults" logic
        # work cleanly even when the JSON omits non-HF-standard fields like
        # `attn_res_block_size`.
        merged = dict(_ATTNRES_DATACLASS_DEFAULTS)
        merged.update({k: v for k, v in kwargs.items() if k != "torch_dtype"})
        # Strip unknown keys (HF sometimes injects `torch_dtype`, `_attn_implementation`, etc.)
        merged = {k: v for k, v in merged.items()
                  if k in _ATTNRES_DATACLASS_DEFAULTS}
        super().__init__(**merged)
        # Re-apply the merged values so the HF config object actually
        # exposes the AttnRes-specific fields after `super().__init__()`
        # strips them.
        for k, v in merged.items():
            setattr(self, k, v)

    def to_attnres_config(self):
        """Build the canonical `AttnResConfig` dataclass from this HF config."""
        from .config import AttnResConfig
        kwargs = {k: getattr(self, k) for k in _ATTNRES_DATACLASS_DEFAULTS}
        # `head_dim=0` sentinel triggers auto-derivation in AttnResConfig.__post_init__
        if kwargs.get("head_dim") == 0:
            kwargs["head_dim"] = 0
        return AttnResConfig(**kwargs)


class AttnResLMForCausalLM(PreTrainedModel):
    """
    HF-compatible wrapper around `AttnResLM` that returns
    `CausalLMOutputWithPast` and exposes a HF-style `self.model` /
    `self.lm_head` layout.

    Architecture invariants preserved:

      - Underlying `AttnResLM` is the SAME class as Phase 1 (no code
        duplication, no parallel inference path).
      - Weights-listing is unchanged: every leaf module name
        (`layers.0.attn.q_proj`, etc.) is exactly what HF would emit
        for a Llama-style model — PEFT pattern-matches against these
        names for `target_modules`.

    Notes about ties:

      When `tie_word_embeddings=True`, our `AttnResLM` has `self.lm_head = None`
      and uses `embed_tokens.weight` directly via `F.linear`. HF expects a
      `self.lm_head` attribute on causal LMs (PEFT probes for it). We expose
      a *proxy* `lm_head` downstream — `self.lm_head.weight` is the same
      tensor as `self.model.embed_tokens.weight`, kept in sync via
      `_tie_weights()`. PEFT sees an actual weight tensor to wrap if a
      user includes `"lm_head"` in `target_modules`, but the default
      list does not.
    """

    config_class = AttnResLMConfigHF
    base_model_prefix = "model"
    _no_split_modules = ["AttnResTransformerBlock"]
    _supports_gradient_checkpointing = True

    def __init__(self, config: AttnResLMConfigHF):
        super().__init__(config)
        # Lazy import so this module is importable even if torch is the only
        # dependency (HF's PretrainedConfig construct-only path).
        from torch import nn
        from .model import AttnResLM
        self.model = AttnResLM(config.to_attnres_config())
        # Build `self.lm_head` BEFORE `post_init()` so HF's tie pass
        # (`_tied_weights_keys = ["lm_head.weight"]`) finds a real module
        # to operate on, NOT a `None` placeholder. This is the canonical
        # Llama-style pattern: lm_head is instantiated in __init__, then
        # _tie_weights() overwrites `.weight` with the embed tensor.
        # An earlier draft did the inverse (set `lm_head = None`,
        # build the proxy inside _tie_weights), which left the tie pass
        # running against `None` and silently no-op-ing.
        attnres_config = config.to_attnres_config()
        if attnres_config.tie_word_embeddings:
            embed = self.model.embed_tokens.weight
            self.lm_head = nn.Linear(embed.shape[1], embed.shape[0], bias=False)
            self.lm_head.weight = embed  # storage-share from the start
        else:
            # AttnResLM already has a real lm_head and it's a child of
            # self.model — re-export it at the wrapper level so HF's
            # attribute-probing finds it.
            self.lm_head = self.model.lm_head
        self.post_init()

    def _tie_weights(self) -> None:
        """HF calls this from `post_init()`. Re-assert the embed↔lm_head tie.

        Storage-shared Tensors are stable across reloads, but if anything
        ever mutates `embed_tokens.weight` (e.g. weight tying after a
        checkpoint load, an explicit `_init_weights`), this refresh
        guarantees the lm_head proxy stays in sync.
        """
        if self.config.to_attnres_config().tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """Standard HF causal LM forward.

        Hands off to the underlying `AttnResLM` and wraps its custom
        `AttnResOutput` dataclass into a HF `CausalLMOutputWithPast`.
        Adds `aux_loss` (MoE load-balancing) to `loss` when present.
        """
        # `return_dict=True` is the only path we use; AttnResLM.forward
        # already returns AttnResOutput in that mode.
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            labels=labels,
            return_dict=True,
        )
        loss = outputs.loss
        if loss is not None and outputs.aux_loss is not None:
            # Phase 1 uses custom Trainer which sums aux_loss in separately;
            # for HF Trainer we sum it into the headline loss so the
            # optimizer gets the correct total gradient signal.
            loss = loss + outputs.aux_loss

        return CausalLMOutputWithPast(
            loss=loss,
            logits=outputs.logits,
            past_key_values=None,        # KV-cache not implemented yet
        )

    def gradient_checkpointing_enable(self, **kwargs):
        """Delegate to the underlying AttnResLM (it already implements this)."""
        self.model.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self):
        self.model.gradient_checkpointing_disable()

    def save_pretrained(self, save_directory, **kwargs):
        """Save the underlying AttnResLM, then write the HF config.json.

        AttnResLM.save_pretrained() writes `config.json` via
        `dataclasses.asdict`, which produces the dataclass schema —
        *almost* identical to the HF schema today (both are flat
        key/value with all AttnResConfig fields). We re-write
        `config.json` with the HF-schema dict so Phase 2's
        `from_pretrained()` can deserialize via `AttnResLMConfigHF`,
        AND so the schema is robust to future dataclass-only fields
        that the HF wrapper can't recover from a dataclass-only dump.
        """
        import os, json
        os.makedirs(save_directory, exist_ok=True)
        self.model.save_pretrained(save_directory)
        # Overwrite config.json with HF schema.
        with open(os.path.join(save_directory, "config.json"), "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """Standard HF entrypoint. AutoModel routing happens at registration
        time (below) — see `AutoConfig.register("attnres", AttnResLMConfigHF)`.
        """
        return super().from_pretrained(pretrained_model_name_or_path, **kwargs)


# ──────────────────────────────────────────────────────────────────────────
# Local AutoConfig / AutoModelForCausalLM registration
#
# Registering at module-import time means Phase 2's standard
#   `AutoModelForCausalLM.from_pretrained("oars344/attnres-phase1")`
# resolves to `AttnResLMForCausalLM` automatically — without needing
# `trust_remote_code=True` and without uploading modeling code to the
# hub. The moment any caller does `from src.model.model import AttnResLM`
# (which is what `src/model/__init__.py` and `train_phase1.py` already do),
# this registration kicks in and HF can route by `model_type="attnres"`.
# ──────────────────────────────────────────────────────────────────────────
AutoConfig.register("attnres", AttnResLMConfigHF)
AutoModelForCausalLM.register(AttnResLMConfigHF, AttnResLMForCausalLM)
