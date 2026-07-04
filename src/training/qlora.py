"""
QLoRA helpers for Phase 2 — 4-bit base + LoRA adapters on a HF AutoModel.

Used by `src/training/train_phase2.py`. Kept thin so the CLI script owns
the training loop + argparse surface; this module just handles the
model-construction choreography:

  1) bitsandbytes 4-bit quantization config (NF4 + double-quant + bf16 compute)
  2) PEFT LoRA config (r/alpha/dropout/target_modules)
  3) Tokenizer pad_token fallback (most base models ship without one)
  4) `prepare_model_for_kbit_training(...)`  — casts LayerNorm + LM head to fp32
  5) `get_peft_model(...)`  — wraps base in LoRA adapters
  6) `model.enable_input_require_grads()`  — required so PEFT adapters can
     still receive gradient signal through the frozen quantized base when
     gradient checkpointing is enabled. **Without this line, QLoRA + GC
     silently trains nothing** — a variant of the same loss-of-gradient
     failure mode we debugged in Phase 1's BlockAttnRes early-return bug.

Designed around `deepreinforce-ai/Ornith-1.0-9B` (Llama-architecture 9B
reasoning / agentic-coding model — see `config/base_models.yaml`) but the
helpers work on any HF AutoModelForCausalLM with a Llama- or Qwen-style
attention / MLP module layout. Override `--lora-target-modules` from the
CLI for fused-QKV or non-Llama architectures.
"""
from __future__ import annotations

import os
from typing import List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


# Default LoRA target modules for Llama- / Qwen-style transformer blocks. These
# are the seven linear projections in each block and match the QLoRA paper's
# recommended configuration for coding / reasoning tasks.
DEFAULT_LORA_TARGETS: List[str] = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


def build_bnb_config(amp_dtype: str = "bf16") -> BitsAndBytesConfig:
    """Construct the bitsandbytes 4-bit config for QLoRA.

    `compute_dtype` is *tied to the training autocast dtype* so 4-bit
    dequantization at each matmul hits a dtype the forward pass is
    actually running in. Mixing bnb fp16-compute with bf16 autocast
    (or vice versa) is a common source of silent numerical drift and
    occasional NaN losses on QLoRA runs — so we make the caller
    specify `--amp-dtype` once and use it for both.

    Args:
        amp_dtype: "bf16" (Ampere+ recommended) or "fp16" (T4 / older GPUs
            where software-emulated bf16 is slow/buggy).

    Returns:
        BitsAndBytesConfig suitable for `AutoModelForCausalLM.from_pretrained(
            quantization_config=..., device_map="auto")`.
    """
    if amp_dtype not in ("bf16", "fp16"):
        raise ValueError(
            f"amp_dtype must be 'bf16' or 'fp16' (got {amp_dtype!r}). "
            "Plain fp32 is not supported by bitsandbytes 4-bit quant."
        )
    compute_dtype = torch.bfloat16 if amp_dtype == "bf16" else torch.float16
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",          # NormalFloat4 — best for LLM weights
        bnb_4bit_use_double_quant=True,     # nested quant of the quant scales
        bnb_4bit_compute_dtype=compute_dtype,
    )


def build_lora_config(
    r: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
    bias: str = "none",
) -> LoraConfig:
    """Construct the PEFT LoRA config.

    Defaults target *all* linear projections in each transformer block
    (q/k/v/o attention + gate/up/down MLP) — QLoRA paper's recommended
    configuration for coding/reasoning. ~0.5-1% of base params in trainable
    adapters for a 9B base at r=16.

    Args:
        r: LoRA rank. Higher = more capacity, slower, more memory.
            Sane values: 8 (light), 16 (default), 32 (high), 64 (max).
        alpha: LoRA scaling. Convention: alpha ≈ 2×r. alpha/r is the
            effective scale of adapter updates relative to base.
        dropout: Adapter dropout. Default 0.05 mirrors QLoRA paper.
        target_modules: Names of modules to wrap. Pass `None` for the
            Llama-style default. For Phi / non-Llama, override via --lora-
            target-modules on the CLI.
        bias: "none" (recommended), "all", or "lora_only".
    """
    if target_modules is None:
        target_modules = list(DEFAULT_LORA_TARGETS)
    if r < 1:
        raise ValueError(f"LoRA rank r must be >= 1, got {r}")
    if alpha < 1:
        raise ValueError(f"LoRA alpha must be >= 1, got {alpha}")
    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        bias=bias,
        task_type="CAUSAL_LM",
    )


def prepare_qlora_model(
    base_model_id: str,
    lora_config: LoraConfig,
    bnb_config: BitsAndBytesConfig,
    *,
    use_gradient_checkpointing: bool = True,
    cache_dir: Optional[str] = None,
    token: Optional[str] = None,
    trust_remote_code: bool = False,
) -> Tuple:
    """Load tokenizer + base (4-bit) + LoRA wrappers, returning (model, tokenizer).

    Performs the canonical QLoRA prep dance in the order required for
    correct gradient flow:

      1) Load the tokenizer. If it doesn't have `pad_token`, set it to
         `eos_token` — many base / agent models (Ornith included) ship
         with no pad_token because their pretraining corpus didn't need
         collation, but QLoRA fine-tuning does.
      2) Load `AutoModelForCausalLM.from_pretrained(..., quantization_config=...)`
         with `device_map="auto"` so accelerate places layers intelligently.
         Disable `model.config.use_cache` — required when gradient
         checkpointing is on, otherwise static KV cache collides with
         the recomputed graph.
      3) `prepare_model_for_kbit_training(...)` — handles two important
         things: (a) cast LayerNorm parameters + the LM output head to
         fp32 so 4-bit quantization doesn't blow them up numerically;
         (b) enable gradient checkpointing if requested.
      4) `get_peft_model(base, lora_config)` — wraps the base in LoRA
         adapters. The base stays frozen; only adapters + the cast-fp32
         LayerNorm/head get `requires_grad=True`.
      5) `model.enable_input_require_grads()` if gradient checkpointing
         is on — this one-line shim is CRITICAL. With the quantized
         base frozen AND input embeddings unused by any "live" param,
         gradient checkpointing's recompute drops the embedding's
         gradient entirely, which means the model's only path to the
         adapters (whose input flows through the embedding) is severed.
         Calling this propagates `requires_grad` from the LoRA wrappers
         back up to the input embedding, restoring the gradient chain.

    Returns:
        (model, tokenizer) ready for `transformers.Trainer` (or any other
        standard PyTorch training loop).
    """
    hf_token = (
        token
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
    )

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_id,
        cache_dir=cache_dir,
        token=hf_token,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None:
        # Almost all agent / code base models lack pad_token. QLoRA needs
        # one for batched training (otherwise PadWithNothingError from
        # the data collator).
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        cache_dir=cache_dir,
        token=hf_token,
        trust_remote_code=trust_remote_code,
        device_map="auto",
    )
    # Disable the static KV cache — incompatible with gradient checkpointing
    # (the recompute would write to a shared cache, corrupting later passes).
    base_model.config.use_cache = False

    base_model = prepare_model_for_kbit_training(
        base_model, use_gradient_checkpointing=use_gradient_checkpointing
    )

    model = get_peft_model(base_model, lora_config)

    if use_gradient_checkpointing:
        # The CRITICAL shim — see step (5) above. Without this, PEFT
        # adapters receive no gradient under gradient checkpointing.
        # (Symptom: loss stays constant at the random-init value, or
        #  loss decreases for the first ~5 steps then freezes — both
        #  point to this.)
        model.enable_input_require_grads()

    return model, tokenizer


def count_trainable_parameters(model) -> Tuple[int, int, float]:
    """Return (trainable, total, percent) — convenient for the `[rank-0] Model:
    X.X% trainable` diagnostic line we use to confirm adapters loaded.

    The expected percent for a 9B base + r=16 LoRA on all linear layers
    is roughly 0.3–0.6%. If you see 0% after `prepare_qlora_model()`,
    you forgot the `enable_input_require_grads()` shim.
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    pct = 100.0 * trainable / max(1, total)
    return trainable, total, pct
