"""
Phase 1 training CLI for AttnRes LM.

This is the canonical entrypoint for Stage-1 pretraining. It replaces the
~280-line inline training loop that previously lived inside the Jupyter
notebook (01_train_attnres_prototype.ipynb). The notebook is now a thin
launcher that does:

    !python -m src.training.train_phase1 \
        --config config/small_prototype.yaml \
        --batch-size 8 --max-steps 1000 --gc \
        --output-dir /content/attnres-runs/phase1 \
        --push-to-hub --repo-id mateooo93/attnres-phase1

Run from the repo root after `pip install -e .`. Multi-GPU:

    torchrun --nproc_per_node=$N_GPUS -m src.training.train_phase1 ...

Supports:
  - Single-GPU (T4 16 GB) with --gc to fit OOM-prone configs.
  - Multi-GPU via torchrun (each rank gets its own AttnResLM wrapped in DDP).
  - HuggingFace Hub push at end-of-training (creates repo, uploads folder).
  - JSON logs at <output_dir>/losses.json for the notebook loss-curve cell.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Optional

import torch
import yaml


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 1 pretraining for AttnRes LM")
    p.add_argument("--config", type=str, default="config/small_prototype.yaml",
                   help="YAML config for AttnResConfig + training defaults")
    p.add_argument("--batch-size", type=int, default=4,
                   help="Per-device batch size")
    p.add_argument("--grad-accum", type=int, default=4,
                   help="Gradient accumulation steps (effective batch = batch_size * grad_accum * world_size)")
    p.add_argument("--max-steps", type=int, default=10000)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--amp-dtype", type=str, default="bf16",
                   choices=("bf16", "fp16"), help="Mixed-precision dtype for autocast")
    p.add_argument("--gc", action="store_true",
                   help="Enable gradient checkpointing (saves VRAM, slower training)")
    p.add_argument("--num-workers", type=int, default=2,
                   help="DataLoader workers (0 for DDP safety on Colab)")
    p.add_argument("--dataset", type=str, default="HuggingFaceFW/fineweb-edu",
                   help="HuggingFace dataset name")
    p.add_argument("--dataset-config", type=str, default="sample-10BT")
    p.add_argument("--dataset-split", type=str, default="train")
    p.add_argument("--max-sequences", type=int, default=4096,
                   help="Cap on streamed sequences (FineWeb is infinite)")
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--output-dir", type=str, default="./checkpoints/phase1")
    p.add_argument("--save-interval", type=int, default=1000)
    p.add_argument("--log-interval", type=int, default=10)
    p.add_argument("--push-to-hub", action="store_true",
                   help="Upload final checkpoint to HuggingFace Hub")
    p.add_argument("--repo-id", type=str, default=None,
                   help="HF repo id, e.g. 'mateooo93/attnres-phase1'")
    p.add_argument("--hf-token", type=str, default=None,
                   help="HF token (overrides HF_TOKEN env var)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _build_config(args: argparse.Namespace):
    """Build (AttnResConfig, TrainingConfig) by overlaying CLI args onto YAML."""
    from src.model.config import AttnResConfig
    from src.training.trainer import TrainingConfig

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")
    with open(args.config, "r") as f:
        raw = yaml.safe_load(f) or {}

    model_dict = raw.get("model", {})
    # Allow the YAML to specify AttnResConfig fields directly under `model:`
    model_config = AttnResConfig(**model_dict)

    train_dict = raw.get("training", {})
    train_dict.update({
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "max_steps": args.max_steps,
        "warmup_steps": args.warmup_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "max_grad_norm": args.max_grad_norm,
        "amp_dtype": args.amp_dtype,
        "log_interval": args.log_interval,
        "save_interval": args.save_interval,
        "output_dir": args.output_dir,
    })
    training_config = TrainingConfig(**train_dict)
    return model_config, training_config


def _build_dataset_and_loader(args: argparse.Namespace, rank: int, world_size: int):
    """Build a (StreamingTextDataset, DataLoader) pair.

    On DDP, we max out at the same number of sequences but shard implicitly
    by seeding the DataLoader's shuffle differently per rank.
    """
    from src.data.dataset import StreamingTextDataset, create_dataloader

    dataset = StreamingTextDataset(
        dataset_name=args.dataset,
        dataset_config=args.dataset_config,
        split=args.dataset_split,
        max_length=args.seq_len,
        max_sequences=args.max_sequences,
    )

    # On DDP, give each rank a deterministic seed offset so they pull
    # different shuffled mini-batches.
    shuffle_seed = args.seed + rank
    loader = create_dataloader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers if world_size == 1 else 0,  # DDP safety
        pin_memory=True,
    )
    if shuffle_seed != args.seed:
        # Re-seed the RNG so the DataLoader's RandomSampler differs per rank
        torch.manual_seed(shuffle_seed)
    return dataset, loader


def _init_distributed(rank: int, world_size: int) -> Optional[torch.nn.parallel.DistributedDataParallel]:
    """Initialize torch.distributed if launched via torchrun.

    Returns the DDP-wrapped model on rank 0 / non-zero ranks, or None on
    single-GPU (world_size==1) so callers know to skip DDP setup.
    """
    if world_size == 1:
        return None
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    torch.distributed.init_process_group(backend=backend)
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
    return None  # actual wrapping happens in main after model assigned to device


def _maybe_wrap_ddp(model: torch.nn.Module, rank: int, world_size: int):
    """Wrap model in DDP if running in a multi-GPU context.

    Uses `find_unused_parameters=True` as defense-in-depth against future
    variations that introduce conditional param usage (MoE with routing
    that sometimes skips an expert, gradient checkpointing on a subset
    of layers, mixed-attention hybrid patterns, etc.). The marginal
    overhead is small for this 12-layer / 114M-param prototype and the
    forward-compatibility win is worth it. The previously hard crash
    (params 10-13 with grad=0 due to BlockAttnRes early-return on layer
    0) was the proximate trigger; the underlying root-cause fix lives
    in src/model/attn_res.py (BlockAttnRes.forward now always stacks
    every available representation so all params participate in the
    autograd graph).
    """
    if world_size == 1 or not torch.distributed.is_initialized():
        return model
    if torch.cuda.is_available():
        model = model.to(rank)
    return torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[rank] if torch.cuda.is_available() else None,
        output_device=rank if torch.cuda.is_available() else None,
        find_unused_parameters=True,
    )


def _save_losses_json(training_config, model, output_dir: str, world_size: int) -> Optional[str]:
    """Write a small losses.json the notebook can plot.

    Trainer already saves its own checkpoint tree; this is a flat sidecar
    for the notebook's loss-curve cell. Only rank 0 writes it.
    """
    if world_size > 1 and torch.distributed.get_rank() != 0:
        return None

    losses_path = os.path.join(output_dir, "losses.json")
    # Trainer writes its own progress; we attach a stub list so the notebook
    # has a guaranteed JSON to read even if Trainer didn't emit one.
    os.makedirs(output_dir, exist_ok=True)
    if not os.path.exists(losses_path):
        with open(losses_path, "w") as f:
            json.dump({"config": training_config.__dict__, "step": []}, f, indent=2)
    return losses_path


def _push_to_hub(model, output_dir: str, repo_id: str, hf_token: Optional[str], model_config) -> None:
    """Upload the final checkpoint directory to HuggingFace Hub.

    Requires `huggingface_hub` (already in requirements.txt).

    Phase 1 saves the model + tokenizer into `output_dir/final/`
    (`_save_final` writes everything into that subdir so the rest of
    `output_dir/` can stay clean — losses.json, intermediate checkpoints
    if any). For Phase 2's
    `AutoModelForCausalLM.from_pretrained(repo)` to find `config.json`
    + `pytorch_model.bin` directly at the repo root, we upload
    `output_dir/final/` *as the repo root* — not `output_dir/`. The
    README model card is written inside `final/README.md` so it lands
    at the repo root after the upload.
    """
    from huggingface_hub import HfApi, create_repo

    if hf_token is None:
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

    api = HfApi(token=hf_token)

    # Create repo if needed (idempotent). Default to private for safety;
    # user can flip with --public later.
    create_repo(repo_id, token=hf_token, exist_ok=True, private=False, repo_type="model")

    final_path = os.path.join(output_dir, "final")

    # Write a small model card so the repo is self-describing. Place it
    # INSIDE `final/` so it lifts to the repo root after we upload that
    # subdir as the root.
    readme_path = os.path.join(final_path, "README.md")
    with open(readme_path, "w") as f:
        f.write(_model_card(repo_id, model, model_config))

    # Verification step — log every file we'll push, so a "missing
    # tokenizer" or "missing config.json" surfaces immediately instead
    # of as a confusing from_pretrained crash hours later in Phase 2.
    files = sorted(os.listdir(final_path))
    print(f"[rank-0] Pushing {len(files)} files to {repo_id}: {files}")

    # Upload the `final/` subdir *as the repo root* so Phase 2's
    # `from_pretrained(repo)` finds config.json + model weights at root.
    api.upload_folder(
        folder_path=final_path,
        repo_id=repo_id,
        commit_message="Phase 1 training complete",
        token=hf_token,
    )
    print(f"[rank-0] Pushed checkpoint to https://huggingface.co/{repo_id}")


def _model_card(repo_id: str, model, model_config) -> str:
    n_params_m = model.num_parameters() / 1e6 if hasattr(model, "num_parameters") else 0
    return f"""---
tags:
- attnres
- transformer
- pytorch
license: apache-2.0
---

# {repo_id}

Phase 1 checkpoint of the **AttnRes** language model — a decoder-only
transformer that replaces standard residual connections with **Block
Attention Residuals** (learned, input-dependent softmax attention over
preceding block representations). See [`Attention Residuals` (Chen
et al., Moonshot AI)](https://github.com/MoonshotAI/Attention-Residuals).

## Architecture

```yaml
hidden_size: {model_config.hidden_size}
num_layers: {model_config.num_hidden_layers}
num_attention_heads: {model_config.num_attention_heads}
num_key_value_heads: {model_config.num_key_value_heads}
vocab_size: {model_config.vocab_size}
use_attn_res: {model_config.use_attn_res}
sublayers_per_block: {model_config.sublayers_per_block}
```

Total parameters: {n_params_m:.1f}M

## Usage

This repo is auto-registered in `transformers.AutoModelForCausalLM` via our
`AttnResLMForCausalLM` wrapper (`src/model/hf_wrapper.py`). When you load
it through any of our training / inference scripts, the wrapper's
`AutoConfig.register("attnres", ...)` call has already run, so a plain
`from_pretrained(repo_id)` call just works — **no `trust_remote_code=True`
required** and no modeling files uploaded to the Hub. For fully external
usage from a fresh Python session that hasn't imported our wrapper, pass
`trust_remote_code=True` (or `import src.model.hf_wrapper` once before
loading):

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

repo_id = "oars344/attnres-phase1"

tokenizer = AutoTokenizer.from_pretrained(repo_id)
model = AutoModelForCausalLM.from_pretrained(
    repo_id,
    torch_dtype=torch.bfloat16,    # 114M params → ~228 MB in bf16
    device_map="auto",
)
model.eval()

prompt = "Once upon a time"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=32, do_sample=True, top_p=0.95)

print(tokenizer.decode(output[0], skip_special_tokens=True))
```

### Inspecting the raw `AttnResLM`

The HF wrapper stores the underlying `AttnResLM` under `model.model`.
The learned pseudo-query projections inside each block's
`BlockAttnRes` live at `model.model.layers[i].{attn_res,mlp_res}.proj`:

```python
# Probing the learned residual aggregation weights:
projection = model.model.layers[0].attn_res.proj.weight  # [1, hidden_size]
print(f"Layer-0 attn_res pseudo-query shape: {tuple(projection.shape)}")
# `mlp_res.proj` is the corresponding projection for the MLP-side residual.
```

### Phase 2 fine-tuning with LoRA / QLoRA

Adapters can be trained on top of this base via
`src/training/train_phase2.py`. AttnRes-aware target modules include the
seven standard Llama-style linears **plus** the BlockAttnRes pseudo-query
projections (`attn_res.proj`, `mlp_res.proj`), so LoRA can re-route the
residual stream for downstream tasks:

```python
from peft import LoraConfig, get_peft_model

attnres_targets = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
    "attn_res.proj", "mlp_res.proj",
]
model = get_peft_model(
    model,
    LoraConfig(r=16, lora_alpha=32, task_type="CAUSAL_LM",
               target_modules=attnres_targets),
)
model.print_trainable_parameters()
```
"""


def _save_final(model, output_dir: str, rank: int, world_size: int) -> None:
    """Save final model + config + tokenizer to <output_dir>/final/ (rank 0 on DDP).

    The tokenizer save is the key bridge to Phase 2: Phase 1's training
    uses `tiktoken` directly (BPE from GPT-2's released merges),
    mathematically identical to HF's `GPT2TokenizerFast`. Saving the HF
    tokenizer alongside the model means Phase 2's
    `AutoTokenizer.from_pretrained("oars344/attnres-phase1")` works
    without us writing a tiktoken→HF shim. The 47-row gap between
    AttnResLM's `vocab_size=50304` and the tokenizer's 50257 is fine:
    embedding rows 50257–50303 are never indexed and stay at their
    init-weight values.
    """
    if world_size > 1 and torch.distributed.get_rank() != 0:
        return
    final_path = os.path.join(output_dir, "final")
    inner = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    inner.save_pretrained(final_path)
    # Step 1: re-write config.json in HF-compatible AttnResLMConfigHF schema,
    # so Phase 2's `AutoModelForCausalLM.from_pretrained()` can deserialize.
    # Robust against future dataclass-only fields.
    _save_hf_compatible_config(final_path, inner.config)
    # Step 2: save a HF-compatible tokenizer next to the model so Phase 2's
    # `AutoTokenizer.from_pretrained()` works without a tiktoken shim.
    _save_hf_compatible_tokenizer(final_path)
    print(f"[rank-{rank}] Saved final model + tokenizer to {final_path}")


def _save_hf_compatible_config(final_path: str, attnres_config) -> None:
    """Overwrite config.json with the HF-compatible `AttnResLMConfigHF` schema.

    `AttnResLM.save_pretrained()` writes config.json in the dataclass
    schema (via `dataclasses.asdict`). For Phase 2's
    `AutoModelForCausalLM.from_pretrained()` to deserialize via the
    `AttnResLMConfigHF` PretrainedConfig, we rewrite config.json with
    the HF schema. Today's dataclass and HF schemas happen to share
    keys, but the HF schema version is the canonical source of truth
    and any future dataclass-only field stays cleanly separated.
    """
    try:
        from dataclasses import asdict
        from src.model.hf_wrapper import AttnResLMConfigHF
        hf_config = AttnResLMConfigHF(**asdict(attnres_config))
        with open(os.path.join(final_path, "config.json"), "w") as f:
            json.dump(hf_config.to_dict(), f, indent=2)
    except Exception as e:
        print(f"[WARN] Failed to write HF-compatible config.json: {e}. "
              f"Phase 2 may fail to deserialize via from_pretrained() unless fixed.")


def _save_hf_compatible_tokenizer(final_path: str) -> None:
    """Save a HuggingFace `GPT2TokenizerFast` to `final_path`.

    Tiktoken's `gpt2` encoding has the SAME byte-level BPE vocab + merges
    as HF's `gpt2` tokenizer. We instantiate the HF tokenizer from disk
    (no size difference in vocab: 50257 either way) and save it next to
    the model so Phase 2's standard HF tooling can load both.

    We also pin `model_max_length` to whatever the just-saved
    `AttnResConfig.max_position_embeddings` was — without this, HF
    defaults `model_max_length=1024` and Phase 2's data collator can
    silently truncate sequences at 1024 even if `args.seq_len=2048`.
    """
    try:
        from transformers import GPT2TokenizerFast
        tok = GPT2TokenizerFast.from_pretrained("gpt2")
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        # Pin model_max_length to the model's actual context — read from
        # the just-saved config.json (HF schema written above).
        cfg_path = os.path.join(final_path, "config.json")
        max_len = 1024
        if os.path.exists(cfg_path):
            try:
                with open(cfg_path) as f:
                    max_len = json.load(f).get("max_position_embeddings", 1024)
            except Exception:
                pass
        tok.model_max_length = max_len
        tok.save_pretrained(final_path)
    except Exception as e:
        # Don't fail the save if the tokenizer write fails — Phase 1
        # already finished training and we don't want a typo in a
        # this-side helper to wipe the checkpoint. Print loud warning
        # so the user knows to investigate.
        print(f"[WARN] Failed to save HF tokenizer to {final_path}: {e}. "
              f"Phase 2 will need local tiktoken shim unless this is fixed.")


def main() -> int:
    args = _parse_args()

    # Distributed init from torchrun
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))

    if world_size > 1:
        _init_distributed(rank, world_size)

    torch.manual_seed(args.seed + rank)

    # Build configs
    model_config, training_config = _build_config(args)

    # Build model
    from src.model.model import AttnResLM
    model = AttnResLM(model_config)

    # Enable gradient checkpointing before moving to GPU so autograd hooks
    # are wired correctly across all blocks.
    if args.gc:
        model.gradient_checkpointing_enable()
        if rank == 0:
            print(f"[rank-{rank}] Gradient checkpointing ENABLED "
                  f"(~30% slower backward, ~10x lower VRAM activations)")

    # Move to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if world_size > 1 and torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
    model = model.to(device)

    # Wrap with DDP after .to(device)
    model = _maybe_wrap_ddp(model, local_rank, world_size)

    # Build dataset/loader
    dataset, train_loader = _build_dataset_and_loader(args, rank, world_size)

    if rank == 0:
        print(f"[rank-{rank}] Model: {model_config.num_hidden_layers} layers, "
              f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params")
        print(f"[rank-{rank}] Dataset: {len(dataset):,} sequences "
              f"({len(dataset) * args.seq_len / 1e6:.1f}M tokens)")
        print(f"[rank-{rank}] Effective batch = "
              f"{args.batch_size} * {args.grad_accum} * {world_size} = "
              f"{args.batch_size * args.grad_accum * world_size}")

    # Build Trainer (existing one — we delegate, not duplicate)
    from src.training.trainer import Trainer
    trainer = Trainer(
        model=model,
        config=training_config,
        train_dataloader=train_loader,
        eval_dataloader=None,
    )

    # Train
    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    if rank == 0:
        print(f"[rank-{rank}] Training complete in {elapsed:.1f}s")

    # Save final + losses.json stub + (optional) push to hub — all rank-0
    _save_final(model, args.output_dir, rank, world_size)
    _save_losses_json(training_config, model, args.output_dir, world_size)

    if args.push_to_hub and args.repo_id:
        if rank == 0:
            # Reload the unwrapped module object for save_pretrained in HF push
            unwrapped = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
            _push_to_hub(unwrapped, args.output_dir, args.repo_id, args.hf_token, model_config)
        if world_size > 1:
            torch.distributed.barrier()

    if world_size > 1:
        torch.distributed.destroy_process_group()

    return 0


if __name__ == "__main__":
    sys.exit(main())
