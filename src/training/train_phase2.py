"""
Phase 2: QLoRA fine-tuning on top of an existing HuggingFace base model.

This is the Phase-2 sibling of `train_phase1.py`. The two scripts are NOT
interchangeable because Phase 1 and Phase 2 work on different model classes
and train them differently:

  * **Phase 1** trains an `AttnResLM` from scratch (custom architecture
    with MoE-friendly aux_loss + BlockAttnRes). It uses the project's
    custom `trainer.py::Trainer` because AttnResLM has model-specific code
    that HF Trainer doesn't know about.

  * **Phase 2** fine-tunes a HuggingFace `AutoModelForCausalLM` (currently
    `deepreinforce-ai/Ornith-1.0-9B`) via bitsandbytes 4-bit quantization
    + PEFT LoRA adapters. It uses the HuggingFace `Trainer` because that
    integrates with PEFT out of the box, including the model's
    `push_to_hub()` automatically pushing ONLY the LoRA adapter weights
    (`adapter_model.safetensors` + `adapter_config.json`) — never the 9B
    base, which is too heavy and isn't yours to redistribute.

Cost target:
  * **~$0 on Colab T4** — tight, batch=1, seq=1024, GC=on. Works but slow.
  * **~$5 on Modal A10G** — comfortable memory headroom for a real run.

CLI surface mirrors `train_phase1.py` (so the notebook can swap cell 7 to
launch Phase 2 without editing anything in Cell 5) plus QLoRA-specific
flags (`--base-model`, `--lora-r`, `--lora-alpha`, `--lora-dropout`,
`--lora-target-modules`). Plus `--smoke` for a 50-step pipeline-only run.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Iterator, List, Optional

import torch
from torch.utils.data import IterableDataset
from transformers import (
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

# Local QLoRA helpers (defined in src/training/qlora.py).
from .qlora import (
    build_bnb_config,
    build_lora_config,
    build_lora_config_for_attnres,
    count_trainable_parameters,
    prepare_lora_model,
    prepare_qlora_model,
)


# ---------------------------------------------------------------------------
# Single source of truth for argparse defaults. Used BOTH as the `default=`
# value for each `_parse_args` flag AND as the comparison target for the
# `_maybe_load_yaml_overlay` "did the user override this flag?" check.
# Adding a flag? Add it here AND in `_parse_args` together.
# ---------------------------------------------------------------------------
_DEFAULTS: dict = {
    # Phase 2's BASE is our own AttnResLM from Phase 1 — so the fine-tune
    # actually exercises the architectural breakthroughs (AttnRes residuals
    # + sub-quadratic / hybrid attention) rather than inheriting Ornith-
    # style standard softmax attention. Override to a HF AutoModel repo
    # id (e.g. `deepreinforce-ai/Ornith-1.0-9B`) to fall back to QLoRA-on-
    # someone-else's-base; the dispatcher below picks the right prep
    # function based on `config.model_type`.
    "base_model": "oars344/attnres-phase1",
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "lora_target_modules": "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    "lora_bias": "none",
    "max_steps": 2000,
    "warmup_steps": 50,
    "learning_rate": 2e-4,
    "weight_decay": 0.0,
    "max_grad_norm": 1.0,
    "batch_size": 1,
    "grad_accum": 16,
    "gc": True,
    "amp_dtype": "bf16",
    "dataset": "HuggingFaceFW/fineweb-edu",
    "dataset_config": "sample-10BT",
    "dataset_split": "train",
    "max_sequences": 4096,
    "seq_len": 1024,
    "output_dir": "./outputs/ornith-qlora",
    "save_interval": 1000,
    "log_interval": 10,
    "push_to_hub": False,
    "repo_id": None,
    "hf_token": None,
    "seed": 42,
    "smoke": False,
}


# ---------------------------------------------------------------------------
# CLI parsing
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 2 QLoRA fine-tune of a HF base model "
                    "(default: deepreinforce-ai/Ornith-1.0-9B)."
    )

    # YAML config overlay (matches the Phase 1 surface so notebook Cell 7 works).
    p.add_argument("--config", type=str, default=None,
                   help="YAML defaults file. Optional — all flags can be set via CLI.")

    # Base model + QLoRA knobs.
    p.add_argument("--base-model", type=str, default=_DEFAULTS["base_model"],
                   help="HF repo id for the base model. Defaults to Ornith-1.0-9B.")
    p.add_argument("--lora-r", type=int, default=_DEFAULTS["lora_r"],
                   help="LoRA rank. 8=light, 16=default, 32=high, 64=max.")
    p.add_argument("--lora-alpha", type=int, default=_DEFAULTS["lora_alpha"],
                   help="LoRA alpha. Convention is alpha ≈ 2×r.")
    p.add_argument("--lora-dropout", type=float, default=_DEFAULTS["lora_dropout"],
                   help="Adapter dropout (default 0.05 mirrors QLoRA paper).")
    p.add_argument("--lora-target-modules", type=str,
                   default=_DEFAULTS["lora_target_modules"],
                   help="Comma-separated module names to wrap with LoRA. "
                        "Default is the Llama-style 7-linears set.")
    p.add_argument("--lora-bias", choices=("none", "all", "lora_only"),
                   default=_DEFAULTS["lora_bias"])

    # Training hyperparameters (mirror Phase 1).
    p.add_argument("--max-steps", type=int, default=_DEFAULTS["max_steps"])
    p.add_argument("--warmup-steps", type=int, default=_DEFAULTS["warmup_steps"])
    p.add_argument("--learning-rate", type=float, default=_DEFAULTS["learning_rate"],
                   help="QLoRA paper recommends 2e-4 for adapter LR. Higher than "
                        "full-finetune because adapters start at zero init.")
    p.add_argument("--weight-decay", type=float, default=_DEFAULTS["weight_decay"])
    p.add_argument("--max-grad-norm", type=float, default=_DEFAULTS["max_grad_norm"])
    p.add_argument("--batch-size", type=int, default=_DEFAULTS["batch_size"],
                   help="Per-device train batch size. 1 for 9B + 4-bit + "
                        "GC on A10G. Increase only if you have headroom.")
    p.add_argument("--grad-accum", type=int, default=_DEFAULTS["grad_accum"],
                   help="Gradient accumulation steps to compensate for batch=1. "
                        "Effective batch = batch_size × grad_accum × world_size.")
    p.add_argument("--gc", action="store_true", default=_DEFAULTS["gc"],
                   help="Enable gradient checkpointing (recommended for QLoRA).")
    p.add_argument("--no-gc", dest="gc", action="store_false",
                   help="Disable gradient checkpointing (faster, but only "
                        "viable on 24+ GB GPUs with no other big tensors).")
    p.add_argument("--amp-dtype", choices=("bf16", "fp16"),
                   default=_DEFAULTS["amp_dtype"],
                   help="bf16 for Ampere+ (A10G/A100). fp16 for T4/older.")

    # Dataset (matches Phase 1's streaming FineWeb-Edu path).
    p.add_argument("--dataset", type=str, default=_DEFAULTS["dataset"])
    p.add_argument("--dataset-config", type=str, default=_DEFAULTS["dataset_config"])
    p.add_argument("--dataset-split", type=str, default=_DEFAULTS["dataset_split"])
    p.add_argument("--max-sequences", type=int, default=_DEFAULTS["max_sequences"],
                   help="Cap on streamed sequences (FineWeb is infinite).")
    p.add_argument("--seq-len", type=int, default=_DEFAULTS["seq_len"],
                   help="Context window — fixed-length chunks for next-token loss.")

    # Output.
    p.add_argument("--output-dir", type=str, default=_DEFAULTS["output_dir"])
    p.add_argument("--save-interval", type=int, default=_DEFAULTS["save_interval"],
                   help="Save PEFT adapters every N steps (0 = disable).")
    p.add_argument("--log-interval", type=int, default=_DEFAULTS["log_interval"])
    p.add_argument("--push-to-hub", action="store_true", default=_DEFAULTS["push_to_hub"],
                   help="Push LoRA adapters to HuggingFace at end-of-training.")
    p.add_argument("--repo-id", type=str, default=_DEFAULTS["repo_id"],
                   help="HF repo id, e.g. 'oars344/ornith-9b-attnres'.")
    p.add_argument("--hf-token", type=str, default=_DEFAULTS["hf_token"],
                   help="HF token (overrides HF_TOKEN env var).")
    p.add_argument("--seed", type=int, default=_DEFAULTS["seed"])

    # Pipeline-validation mode.
    p.add_argument("--quantize-base", action="store_true",
                   default=False,
                   help="Apply bitsandbytes 4-bit quant to the base (only "
                        "useful for larger bases; the 114M default AttnResLM "
                        "fits comfortably in bf16 without quant). Ignored if "
                        "the base is an HF AutoModel like Ornith, where the "
                        "QLoRA path always quantizes.")
    # Validation gates (default ON, with escape hatch for power users).
    p.add_argument("--no-validate", dest="validate", action="store_false",
                   default=True,
                   help="Disable pre-flight checks (lm_head-in-targets + tied "
                        "embeddings footgun, and the post-load adapter-count "
                        "assertion). Use only when you've audited the "
                        "interaction yourself.")
    p.add_argument("--smoke", action="store_true", default=_DEFAULTS["smoke"],
                   help="Run a 50-step pipeline-validation pass instead of the "
                        "full training. Disables hub push, sets save_interval=0, "
                        "appends '/smoke' to output-dir. Use to validate the "
                        "whole stack before committing compute dollars.")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Streaming FineWeb-Edu → tokenized → fixed-length chunks
# ---------------------------------------------------------------------------

class _Phase2StreamingChunks(IterableDataset):
    """Streaming FineWeb-Edu → tokenized → fixed-length next-token chunks.

    Mirrors `src/data/dataset.py::StreamingTextDataset`'s role but uses
    HF `AutoTokenizer` instead of tiktoken — Phase 2 fine-tunes a HF base
    whose vocabulary and special tokens differ from the AttnResLM one.

    Each yielded sample is a dict with:
        input_ids : torch.LongTensor [seq_len]
        labels    : torch.LongTensor [seq_len]    (input_ids shifted by 1, with
                                                  the last position masked to -100
                                                  so loss ignores the partly-
                                                  filled tail)

    Note: deliberately NO `__len__` method. IterableDataset that emit
    unbounded streams should report their length as NotImplementedError
    and let HF Trainer drive stepping via `max_steps` × accumulation.
    Reporting a misleading finite length confuses Trainer's per-epoch
    progress bar and can over-estimate total_steps.
    """

    def __init__(
        self,
        dataset_name: str,
        dataset_config: str,
        dataset_split: str,
        tokenizer,
        seq_len: int,
        max_sequences: int,
        seed: int,
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.dataset_split = dataset_split
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.max_sequences = max_sequences
        # seed captured for reproducibility; worker shuffling could be added later
        self._seed = seed

    def __iter__(self) -> Iterator[dict]:
        from datasets import load_dataset

        hf_token = (
            os.environ.get("HF_TOKEN")
            or os.environ.get("HUGGINGFACE_TOKEN")
        )
        ds = load_dataset(
            self.dataset_name,
            self.dataset_config,
            split=self.dataset_split,
            streaming=True,
            token=hf_token,
        )

        eos = self.tokenizer.eos_token_id
        if eos is None:
            raise ValueError(
                f"Tokenizer for {self.dataset_name} has no eos_token_id; "
                "this breaks the chunk boundary handling."
            )

        buf: List[int] = []
        emitted = 0
        for sample in ds:
            text = sample.get("text")
            if not text:
                continue
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            buf.extend(ids)
            buf.append(eos)

            # Emit as many complete seq_len+1 chunks as we have buffered.
            while len(buf) >= self.seq_len + 1:
                chunk = buf[: self.seq_len + 1]
                buf = buf[self.seq_len:]
                x = torch.tensor(chunk[:-1], dtype=torch.long)
                # Mask the last "shift" position to -100 so cross-entropy
                # ignores it (the final token has no next-token target).
                y = torch.tensor(chunk[1:] + [-100], dtype=torch.long)
                yield {"input_ids": x, "labels": y}
                emitted += 1
                if 0 < self.max_sequences <= emitted:
                    return


def _collate(batch_items: List[dict], pad_token_id: int) -> dict:
    """Right-pad a batch of variable-length items to the max length in-batch.

    Streaming chunks are emitted as exactly `seq_len` long, so padding is a
    no-op in practice. Kept here for safety in case shorter chunks ever
    slip through (e.g. trajectory data on Phase 2.5).
    """
    max_len = max(len(it["input_ids"]) for it in batch_items)
    input_ids = torch.full((len(batch_items), max_len), pad_token_id, dtype=torch.long)
    labels = torch.full((len(batch_items), max_len), -100, dtype=torch.long)
    attention_mask = torch.zeros((len(batch_items), max_len), dtype=torch.long)
    for i, it in enumerate(batch_items):
        L = len(it["input_ids"])
        input_ids[i, :L] = it["input_ids"]
        labels[i, :L] = it["labels"]
        attention_mask[i, :L] = 1
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }


# ---------------------------------------------------------------------------
# Losses.json callback — schema matches Phase 1 verbatim so notebook Cell 6
# can keep working with structured-dict-per-entry format.
# ---------------------------------------------------------------------------

class JsonLossCallback(TrainerCallback):
    """Records the training loss curve to `<output_dir>/losses.json`.

    Schema (intentionally identical to `Trainer._write_losses_json`):
        {
          "config": { ... TrainingArguments fields ... },
          "step":   [
            { "step": N,  "loss": float, "lr": float, "elapsed_s": float },
            ...
          ]
        }

    Caveat: Cell 10 in `notebooks/01_train_attnres_prototype.ipynb` was
    written assuming `step` is a flat list of floats. This is a pre-existing
    Phase-1 bug — Phase 1's `_write_losses_json` ALSO writes structured
    dicts. Plotting that cell needs a separate fix in the notebook; we
    preserve Phase-1's structured schema here for consistency rather than
    bloat the training loop with a redundant flat-list sidecar.
    """

    def __init__(self):
        self.history: List[dict] = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero:
            return
        if not logs:
            return
        entry = {
            "step": state.global_step,
            "loss": float(logs.get("loss", 0.0) or 0.0),
            "lr": float(logs.get("learning_rate", 0.0) or 0.0),
        }
        if "train_runtime" in logs:
            entry["elapsed_s"] = float(logs["train_runtime"])
        if "train_tokens_per_second" in logs:
            entry["tokens_per_sec"] = float(logs["train_tokens_per_second"])
        self.history.append(entry)

    def on_train_end(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return
        os.makedirs(args.output_dir, exist_ok=True)
        # Drop `_`-prefixed "private" fields and callables — Phase 1 mirror.
        # `default=str` on json.dump covers enum-like values
        # (e.g. SchedulerType.COSINE) that the strict Phase-1-ish filter
        # would otherwise drop silently, causing downstream Cell 10's
        # `step[]` consumption to silently lose field access.
        config_dict = {
            k: v for k, v in vars(args).items()
            if not k.startswith("_") and not callable(v)
        }
        path = os.path.join(args.output_dir, "losses.json")
        with open(path, "w") as f:
            json.dump(
                {"config": config_dict, "step": self.history},
                f, indent=2, default=str,
            )
        print(f"[phase-2] Wrote loss log: {path} ({len(self.history)} entries)")


# ---------------------------------------------------------------------------
# Base auto-dispatch helper
# ---------------------------------------------------------------------------

def _peek_base_config(repo_id: str, hf_token: Optional[str]) -> dict:
    """Peek at `config.json` at `repo_id` (or local path) for tier-1 fields.

    Returns a dict with `model_type` (string, e.g. `"attnres"`, `"llama"`,
    `"qwen2"`) and `tie_word_embeddings` (bool). These are the only fields
    Phase 2's pre-flight dispatcher reads; everything else stays out of
    the round-trip-cost-critical peek path.

    For local paths, just reads the local `config.json`. For HF hub repos,
    uses `huggingface_hub.hf_hub_download` to grab just the JSON file.
    Defaults to `{"model_type": "unknown", "tie_word_embeddings": False}`
    on any error so the caller falls back to the safe (QLoRA) path.
    """
    import json
    try:
        if os.path.isdir(repo_id):
            with open(os.path.join(repo_id, "config.json")) as f:
                cfg = json.load(f)
        else:
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(
                repo_id=repo_id, filename="config.json", token=hf_token,
            )
            with open(path) as f:
                cfg = json.load(f)
        return {
            "model_type": cfg.get("model_type", "unknown"),
            "tie_word_embeddings": bool(cfg.get("tie_word_embeddings", False)),
        }
    except Exception as e:
        print(f"[phase-2] WARN: could not peek config for {repo_id}: {e}. "
              f"Falling back to standard QLoRA prep.")
        return {"model_type": "unknown", "tie_word_embeddings": False}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _maybe_apply_smoke_overrides(args: argparse.Namespace) -> None:
    """If --smoke, run a tiny 50-step pipeline-only pass and disable side-effects."""
    if not args.smoke:
        return
    print(
        "[phase-2] --smoke mode: max_steps=50, save_interval=0, "
        "push-to-hub OFF. Use this to validate the whole stack before "
        "committing the $5 Modal budget."
    )
    args.max_steps = 50
    args.save_interval = 0
    args.push_to_hub = False
    if not args.output_dir.endswith("/smoke"):
        args.output_dir = args.output_dir.rstrip("/") + "/smoke"


def _maybe_load_yaml_overlay(args: argparse.Namespace) -> None:
    """If --config is set, overlay those defaults onto args (CLI flags win).

    Detection rule: a flag was overridden by the user iff the parsed value
    differs from `_DEFAULTS[flag]`. If they match, the YAML value applies.
    """
    if not args.config:
        return
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"--config file not found: {args.config}")
    import yaml
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f) or {}

    mapping = {
        "model": {
            "base_model": "base_model",
            "amp_dtype": "amp_dtype",
            "use_gradient_checkpointing": "_gc_yaml",
        },
        "lora": {
            "r": "lora_r",
            "alpha": "lora_alpha",
            "dropout": "lora_dropout",
            "target_modules": "_lora_targets_yaml",
            "bias": "lora_bias",
        },
        "training": {
            "learning_rate": "learning_rate",
            "max_steps": "max_steps",
            "warmup_steps": "warmup_steps",
            "batch_size": "batch_size",
            "gradient_accumulation_steps": "grad_accum",
            "save_interval": "save_interval",
            "log_interval": "log_interval",
            "output_dir": "output_dir",
            "push_to_hub": "push_to_hub",
            "repo_id": "repo_id",
            "max_sequences": "max_sequences",
            "seq_len": "seq_len",
        },
    }
    for block_name, fields in mapping.items():
        block = cfg.get(block_name, {}) or {}
        for yaml_key, attr_name in fields.items():
            if yaml_key not in block:
                continue
            yaml_value = block[yaml_key]
            # Special keys are applied post-hoc; skip direct overwrite for now.
            if attr_name.startswith("_"):
                continue
            # Apply YAML value only if the user didn't override the relevant CLI flag.
            if getattr(args, attr_name, None) == _DEFAULTS.get(attr_name):
                setattr(args, attr_name, yaml_value)

    # Apply special keys.
    if "use_gradient_checkpointing" in (cfg.get("model") or {}):
        args.gc = bool(cfg["model"]["use_gradient_checkpointing"])
    if "target_modules" in (cfg.get("lora") or {}):
        targets = cfg["lora"]["target_modules"]
        if isinstance(targets, list):
            args.lora_target_modules = ",".join(str(t) for t in targets)


def main() -> int:
    args = _parse_args()

    _maybe_load_yaml_overlay(args)
    _maybe_apply_smoke_overrides(args)

    # Pre-flight: warn if user is on T4 / pre-Ampere with bf16 (slow fallback).
    if torch.cuda.is_available():
        try:
            cap_major = torch.cuda.get_device_capability()[0]
        except Exception:
            cap_major = 0
        if cap_major < 8 and args.amp_dtype == "bf16":
            print(
                "[phase-2] WARNING: GPU compute capability is <8 "
                f"(CC {cap_major}.x — pre-Ampere, e.g. T4). bf16 is "
                "software-emulated here and will be 2-3× slower than fp16. "
                "Suggested fix: re-run with --amp-dtype fp16."
            )

    print(
        f"[phase-2] base={args.base_model}  LoRA r={args.lora_r}α={args.lora_alpha} "
        f"  amp={args.amp_dtype}  gc={args.gc}  "
        f"bs={args.batch_size}×{args.grad_accum}=eff{args.batch_size*args.grad_accum}"
    )

    # 4-bit + LoRA prep.
    bnb_config = build_bnb_config(amp_dtype=args.amp_dtype)
    lora_targets = [t.strip() for t in args.lora_target_modules.split(",") if t.strip()]

    print(
        f"[phase-2] Loading {args.base_model} + PEFT/LoRA "
        f"(this downloads once + caches; ~3 min first time)..."
    )
    t0 = time.time()
    # Auto-dispatch: peek at the base's `model_type` + `tie_word_embeddings`
    # from `config.json` at the hub before committing to a prep function.
    # Avoids a wasted 4-bit quant pass on a small / non-quantized base
    # (114M AttnResLM), and avoids skipping quant when the user pointed
    # Phase 2 at a 9B HF AutoModel (Ornith).
    base_meta = _peek_base_config(args.base_model, args.hf_token)
    base_model_type = base_meta["model_type"]

    # Pre-flight: PEFT footgun. If the user puts `lm_head` in target_modules
    # AND the base has tied embeddings, PEFT will replace lm_head's nn.Linear
    # with `lora.Linear` whose `base_layer.weight` is *copied* (not
    # storage-shared). The embed↔lm_head tie silently breaks. Refuse to
    # proceed unless the caller passes --no-validate.
    if args.validate and "lm_head" in lora_targets and base_meta.get("tie_word_embeddings", False):
        raise RuntimeError(
            "[phase-2] ABORT: `--lora-target-modules` includes \"lm_head\" AND "
            "the base has `tie_word_embeddings=True`. PEFT will silently "
            "break the embed↔lm_head weight tie (LoRA wraps copy-storage, "
            "not the original param). Either:\n"
            "  (a) remove `lm_head` from `--lora-target-modules`, OR\n"
            "  (b) rerun with `--no-validate` to bypass (you take responsibility "
            "      for the busted tie)."
        )
    if base_model_type == "attnres":
        # No 4-bit quant — our 114M base fits comfortably in bf16 on T4/A10G.
        # AttnRes-aware target list adds attn_res.proj / mlp_res.proj unless
        # the user overrode via --lora-target-modules on the CLI.
        if lora_targets == [t.strip() for t in _DEFAULTS["lora_target_modules"].split(",") if t.strip()]:
            from .qlora import DEFAULT_LORA_TARGETS_ATTNRES, DEFAULT_LORA_TARGETS
            if set(lora_targets) == set(DEFAULT_LORA_TARGETS):
                lora_targets = list(DEFAULT_LORA_TARGETS_ATTNRES)
        lora_config = build_lora_config_for_attnres(
            r=args.lora_r,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
            target_modules=lora_targets,
            bias=args.lora_bias,
        )
        model, tokenizer = prepare_lora_model(
            args.base_model,
            lora_config=lora_config,
            use_gradient_checkpointing=args.gc,
            token=args.hf_token,
            quantize=args.quantize_base,
            bnb_compute_dtype=(torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16),
        )
    else:
        # Standard QLoRA path: 4-bit NF4 + double quant + LoRA on the
        # 7 Llama-style linears. `lora_targets` is honored verbatim.
        lora_config = build_lora_config(
            r=args.lora_r,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
            target_modules=lora_targets,
            bias=args.lora_bias,
        )
        model, tokenizer = prepare_qlora_model(
            args.base_model,
            lora_config=lora_config,
            bnb_config=bnb_config,
            use_gradient_checkpointing=args.gc,
            token=args.hf_token,
        )
    trainable, total, pct = count_trainable_parameters(model)
    print(f"[phase-2] Trainable: {trainable:,} / {total:,}  ({pct:.2f}%)")

    # Post-load sanity check: if PEFT matched ~0 modules, the user has a typo
    # in `--lora-target-modules` and the run would train nothing. Surface it
    # loud-and-clear instead of letting the run log "looks fine" for 2000
    # steps while loss stays constant.
    if pct < 0.1:
        raise RuntimeError(
            f"LoRA adapters only matched {pct:.2f}% of params — check "
            "`--lora-target-modules` for typos. PEFT silently matched almost "
            "nothing and the run would train nothing. Aborting before "
            "wasting compute dollars."
        )

    print(f"[phase-2] Base+adapter load took {time.time()-t0:.1f}s")

    # Streaming FineWeb-Edu dataset.
    train_ds = _Phase2StreamingChunks(
        dataset_name=args.dataset,
        dataset_config=args.dataset_config,
        dataset_split=args.dataset_split,
        tokenizer=tokenizer,
        seq_len=args.seq_len,
        max_sequences=args.max_sequences,
        seed=args.seed,
    )

    # If push-to-hub is requested and user didn't set output_dir, route
    # adapter cache to a stable local path before push.
    if args.push_to_hub and args.repo_id and not args.output_dir:
        args.output_dir = f"./hf_cache_{args.repo_id.replace('/', '_')}"

    # HuggingFace Trainer args.
    targs = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        lr_scheduler_type="cosine",
        gradient_checkpointing=args.gc,
        bf16=(args.amp_dtype == "bf16"),
        fp16=(args.amp_dtype == "fp16"),
        logging_steps=args.log_interval,
        # Avoid Trainer error when save_steps <= 0 by routing 0 → very large.
        save_steps=max(args.save_interval, 1) if args.save_interval > 0 else 10**9,
        save_total_limit=2,
        report_to=[],                  # wandb/HF not auto-enabled — pass if user wants
        push_to_hub=args.push_to_hub,
        hub_model_id=args.repo_id if args.push_to_hub else None,
        hub_token=args.hf_token or os.environ.get("HF_TOKEN"),
        # Safety default for any base model whose license is unclear;
        # user can opt into public push by passing --public override
        # (not implemented in this CLI surface — would require adding
        # `--public` add_argument).
        hub_private_repo=True,
        seed=args.seed,
        # DDP safety on Modal / Colab: 0 worker dataloader, no fork surprise.
        dataloader_num_workers=0,
        remove_unused_columns=False,    # we keep input_ids/labels/attention_mask
        logging_first_step=True,
    )

    # Custom collator for safety (in case streaming chunks ever emit variable length).
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    def _collator(batch_items):
        return _collate(batch_items, pad_id)

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        data_collator=_collator,
        callbacks=[JsonLossCallback()],
    )

    print(f"[phase-2] Starting training: {args.max_steps} steps, "
          f"effective batch={args.batch_size * args.grad_accum}")
    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    print(f"[phase-2] Training complete in {elapsed:.1f}s "
          f"({elapsed/max(1,args.max_steps):.2f}s/step)")

    # Push adapters — trainer.push_to_hub() on a PEFT model pushes ONLY the
    # adapter weights (`adapter_model.safetensors` + `adapter_config.json`),
    # never the 9B base.
    if args.push_to_hub and args.repo_id:
        trainer.push_to_hub()
        print(
            f"[phase-2] Pushed QLoRA adapters to "
            f"https://huggingface.co/{args.repo_id}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
