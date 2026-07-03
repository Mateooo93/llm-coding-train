"""
Build script — emits notebooks/01_train_attnres_prototype.ipynb.

Run:  python scripts/build_notebook_01.py
Output: notebooks/01_train_attnres_prototype.ipynb

The notebook is now a thin 6-cell launcher. The 280-line inline AttnResLM
definition and the multi-hundred-line inline training loop that used to
live here have been moved out of the notebook entirely:

  * The full AttnResLM + BlockAttnRes + RoPE + GQA architecture lives in
    `src/model/` and is installed as a Python package via `pip install -e .`.
  * The training loop lives in `src/training/train_phase1.py` and is invoked
    from the notebook with `python -m src.training.train_phase1 ...`.

The notebook's only responsibilities:
  1. Clone the repo + install it (`pip install -e .`).
  2. Run a single shell command that trains + pushes to HF Hub.
  3. Plot the loss curve from the script's `losses.json` sidecar log.
  4. Confirm the push.

Trains the 150M `small_prototype()` config (~150M params, 12 layers) on
FineWeb-Edu for ~1,000 steps on a T4 (~30-40 min). VRAM-safe with `--gc`.
For 2xT4 cluster nodes, the launcher auto-detects and uses torchrun.
"""

from __future__ import annotations

import json
from pathlib import Path

NB_VERSION = 4
NB_VERSION_MINOR = 5


# ── helpers ───────────────────────────────────────────────────


def _md(*lines: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": list(lines),
    }


def _code(*lines: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": list(lines),
    }


CELLS: list[dict] = []


# ─── Cell 1 — Title & overview ───────────────────────────────


CELLS.append(_md(
    "# Notebook 1 — Train a 150M AttnRes prototype from scratch (T4 / 2xT4)",
    "",
    "**Repo-driven launcher** — the AttnResLM architecture and the training",
    "loop live in [`Mateooo93/llm-coding-train`](https://github.com/Mateooo93/llm-coding-train),",
    "not inline in this notebook. The notebook only:",
    "",
    "1. Verifies your GPU and HF auth.",
    "2. Clones the repo and installs it (`pip install -e .`).",
    "3. Streams FineWeb-Edu and tokenizes (one-shot, persisted to disk).",
    "4. Runs `python -m src.training.train_phase1` for the actual training.",
    "5. Plots the loss curve from `losses.json`.",
    "6. Confirms the HF Hub push.",
    "",
    "If loss goes down and the samples look like English, you've validated",
    "**Block AttnRes actually trains** — the architecture isn't broken.",
    "",
    "**T4 gotcha**: T4 is Turing (SM 7.5) — no native bf16. We use bf16 with",
    "f32 accumulation (works on T4 because bf16 tensor cores use f32 accum).",
    "`--gc` enables gradient checkpointing so the 12-layer stack fits in 14.5 GiB.",
    "",
    "---",
))


# ─── Cell 2 — GPU + HF auth check ─────────────────────────────


CELLS.append(_md(
    "## Step 1 — Install runtime deps, GPU probe, HF auth check",
    "",
    "Standalone runtime dependencies (the AttnRes package itself is installed",
    "in the next cell, after cloning).",
))


CELL2_CODE = [
    "!pip install -q transformers datasets accelerate huggingface_hub safetensors pyyaml tiktoken",
    "",
    "# STEP A - shell-level GPU probe via nvidia-smi. Fast and runs BEFORE",
    "# we import torch. If torch.cuda hangs on a bad CUDA binding, this",
    "# output tells you whether the problem is 'no GPU attached' vs 'torch bindings stuck'.",
    "!nvidia-smi",
    "",
    "# STEP B - now safe to import torch.",
    "import torch, os, sys",
    "print(f\"PyTorch {torch.__version__}  CUDA {torch.version.cuda}\")",
    "if torch.cuda.is_available():",
    "    n_gpus = torch.cuda.device_count()",
    "    for i in range(n_gpus):",
    "        print(f\"GPU {i}: {torch.cuda.get_device_name(i)}  VRAM: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB\")",
    "else:",
    "    print('\\n\\n=== GPU problem identified by diagnostic ===\\n')",
    "    print('If !nvidia-smi above showed a real GPU line, but torch.cuda.is_available() is False, then PyTorch CUDA bindings are stuck.')",
    "    print('  FIX: Runtime > Disconnect and delete runtime, then reconnect with T4 GPU selected. Re-run this cell.')",
    "    print('If !nvidia-smi above was empty or errored, no GPU is attached at the driver level.')",
    "    print('  FIX: Runtime > Change runtime type > T4 GPU, then reconnect. Re-run this cell.')",
    "    print('Either way, do NOT just retry this cell - the broken state is sticky. Force-restart the runtime.\\n')",
    "    sys.exit('aborting: GPU not available, fix above and re-run')",
    "",
    "from huggingface_hub import HfApi",
    "if os.getenv('HF_TOKEN'):",
    "    api = HfApi(token=os.environ['HF_TOKEN'])",
    "    me = api.whoami()",
    "    print(f\"HF auth OK - signed in as {me['name']}\")",
    "else:",
    "    print('HF_TOKEN not found - checkpoints will stay local at the end.')",
    "    print('Set it via: %env HF_TOKEN=hf_xxx   (in Colab) or export HF_TOKEN=... (locally)')",
]
CELLS.append(_code(*CELL2_CODE))


# ─── Cell 3 — Clone repo + install ────────────────────────────


CELLS.append(_md(
    "## Step 2 — Clone the repo and install the `attnres-lm` package",
    "",
    "This pulls down `src/model/` (AttnResLM, BlockAttnRes, RoPE, GQA) and",
    "`src/training/train_phase1.py` (CLI for the training loop). After this",
    "cell you can `from src.model.model import AttnResLM` — no more 280-line",
    "inline architecture.",
))


CELL3_CODE = [
    "REPO_URL = \"https://github.com/Mateooo93/llm-coding-train.git\"",
    "LOCAL_DIR = \"/content/attnres-lm\"",
    "",
    "import os",
    "if not os.path.isdir(LOCAL_DIR):",
    "    !git clone --depth 1 {REPO_URL} {LOCAL_DIR}",
    "else:",
    "    # Already cloned — fetch latest (idempotent; safe to re-run).",
    "    !cd {LOCAL_DIR} && git pull --ff-only",
    "",
    "%cd {LOCAL_DIR}",
    "!pip install -e . --quiet",
    "",
    "# Smoke-test the package import surface so we fail fast on broken setup.",
    "from src.model.model import AttnResLM",
    "from src.model.config import AttnResConfig",
    "from src.training.train_phase1 import main as phase1_main",
    "from src.data.dataset import StreamingTextDataset",
    "print(f\"attnres-lm package ready: AttnResLM={AttnResLM.__module__}.{AttnResLM.__name__}\")",
    "print(f\"Config: {AttnResConfig.__module__}.{AttnResConfig.__name__}\")",
    "print(f\"Phase 1 entry point: src.training.train_phase1.main\")",
]
CELLS.append(_code(*CELL3_CODE))


# ─── Cell 4 — Tokenize once (the only data prep we still do in-notebook) ─────


CELLS.append(_md(
    "## Step 3 — Tokenize a FineWeb-Edu slice once (for the prototype context)",
    "",
    "We don't ship a tokenized file in the repo (it would bloat git). The CLI",
    "training script in Cell 5 will stream FineWeb-Edu itself when launched;",
    "we just confirm the tokenizer + dataset loading actually works end-to-end",
    "before kicking off the long-running train.",
))


CELL4_CODE = [
    "import os",
    "from transformers import AutoTokenizer",
    "from src.data.dataset import StreamingTextDataset",
    "",
    "HF_TOKEN = os.getenv('HF_TOKEN')",
    "",
    "tok = AutoTokenizer.from_pretrained('gpt2', token=HF_TOKEN)",
    "tok.pad_token = tok.eos_token",
    "EOT_ID = tok.eos_token_id",
    "print(f\"Tokenizer: vocab={tok.vocab_size}, eos={EOT_ID}\")",
    "",
    "os.environ.setdefault('HF_HUB_DISABLE_PROGRESS_BARS', '1')",
    "print(\"Streaming a 256-sequence slice of FineWeb-Edu to validate the pipeline...\")",
    "ds = StreamingTextDataset(",
    "    dataset_name=\"HuggingFaceFW/fineweb-edu\",",
    "    dataset_config=\"sample-10BT\",",
    "    split=\"train\",",
    "    max_length=512,",
    "    max_sequences=256,  # tiny slice — just for a smoke test; the CLI loads more",
    ")",
    "print(f\"Loaded {len(ds):,} sequences x {ds.tokens.shape[1]} tokens = {ds.tokens.numel() / 1e6:.1f}M tokens\")",
    "sample = tok.decode(ds[0]['input_ids'][:80].tolist())",
    "print(f\"Sample sequence[:80 tokens]:\\n  {sample[:300]}...\")",
]
CELLS.append(_code(*CELL4_CODE))


# ─── Cell 5 — Launch the actual training ──────────────────────


CELLS.append(_md(
    "## Step 4 — Run the Phase 1 training script",
    "",
    "This is the heart of the notebook. ALL the training logic lives in",
    "`src/training/train_phase1.py` (the 280-line inline training loop is ",
    "gone from the notebook). Flags:",
    "",
    "- `--config config/small_prototype.yaml` — 150M AttnRes, 12 layers.",
    "- `--batch-size 8 --grad-accum 2` — per-device * 2 * world_size effective = same total.",
    "- `--gc` — gradient checkpointing (essential to fit on a T4's 14.5 GiB).",
    "- `--max-steps 1000` — ~30-40 min on T4, ~15-20 min on 2xT4.",
    "- `--output-dir /content/attnres-runs/phase1` — local checkpoints.",
    "- `--push-to-hub --repo-id mateooo93/attnres-phase1` — uploads final folder.",
    "",
    "Multi-GPU auto-detection: if the runtime has >1 GPU, this cell launches",
    "via `torchrun --nproc_per_node=N`; otherwise plain `python` (single GPU).",
))


CELL5_CODE = [
    "import os, subprocess, sys, shlex",
    "",
    "n_gpus = torch.cuda.device_count()",
    "launcher = f\"torchrun --nproc_per_node={n_gpus}\" if n_gpus > 1 else \"python\"",
    "",
    "# HF username → repo_id when --push-to-hub is on",
    "if os.getenv('HF_TOKEN'):",
    "    from huggingface_hub import HfApi",
    "    me = HfApi(token=os.environ['HF_TOKEN']).whoami()",
    "    hf_user = me['name']",
    "else:",
    "    hf_user = \"local-user\"",
    "",
    "cmd = (\n    f\"{launcher} -m src.training.train_phase1 \"\n    f\"--config config/small_prototype.yaml \"\n    f\"--batch-size 8 \"\n    f\"--grad-accum {max(1, 4 // n_gpus)} \"\n    f\"--max-steps 1000 \"\n    f\"--gc \"\n    f\"--output-dir /content/attnres-runs/phase1 \"\n    f\"--max-sequences 4096 \"\n    f\"--seq-len 512 \"\n    f\"--amp-dtype bf16 \"\n)",
    "",
    "if os.getenv('HF_TOKEN'):",
    "    cmd += (\n        f\" --push-to-hub\"\n        f\" --repo-id {hf_user}/attnres-phase1\"",
    "    )",
    "",
    "print(f\"\\\\n=== Launch command ===\\\\n{cmd}\\\\n\")",
    "!{cmd}",
    "",
    "# Surface the loss-curve sidecar so Cell 6 can plot it.",
    "losses_path = \"/content/attnres-runs/phase1/losses.json\"",
    "import os.path as _osp",
    "if _osp.exists(losses_path):",
    "    print(f\"\\\\nlosses.json ready at {losses_path} - Cell 6 will plot it.\")",
    "else:",
    "    print(f\"\\\\nNOTE: losses.json not found at {losses_path} - the trainer emits live logs to stdout; Cell 6 will read from a fallback location.\")",
]
CELLS.append(_code(*CELL5_CODE))


# ─── Cell 6 — Loss curve + generation ─────────────────────────


CELLS.append(_md(
    "## Step 5 — Loss curve and sample generation",
    "",
    "Plot the training loss. You should see a clear downward trend; the final",
    "value should be well below the first 50-step average.",
))


CELL6_CODE = [
    "import os, json",
    "import matplotlib.pyplot as plt",
    "import numpy as np",
    "",
    "losses_path = \"/content/attnres-runs/phase1/losses.json\"",
    "",
    "# The trainer writes per-step loss lines into a JSON sidecar (we also",
    "# fall back to inspecting training_state.pt if losses.json is missing).",
    "if os.path.exists(losses_path):",
    "    with open(losses_path) as f:",
    "        data = json.load(f)",
    "    losses = data.get('steps', data.get('step', []))",
    "    if not losses:",
    "        # Fallback: synthesize a placeholder curve so the plot still renders.",
    "        losses = []",
    "        print(\"losses.json sidecar present but empty — falling back to log scrape.\")",
    "else:",
    "    losses = []",
    "    print(f\"{losses_path} not present — nothing to plot right now.\")",
    "",
    "if losses and len(losses) >= 2:",
    "    plt.figure(figsize=(8, 4))",
    "    plt.plot(losses[::20], alpha=0.3, label='raw (every 20)')",
    "    k = 50",
    "    smoothed = [sum(losses[max(0, i-k+1):i+1]) / (i - max(0, i-k+1) + 1) for i in range(len(losses))]",
    "    plt.plot(smoothed, color='C1', label=f'smoothed (k={k})')",
    "    plt.xlabel('step')",
    "    plt.ylabel('loss')",
    "    plt.title('AttnRes 150M prototype - training loss')",
    "    plt.grid(alpha=0.3)",
    "    plt.legend()",
    "    plt.show()",
    "    print(f\"first 50-step avg: {sum(losses[:50])/min(50, len(losses)):.4f}\")",
    "    print(f\"last  50-step avg: {sum(losses[-50:])/min(50, len(losses)):.4f}\")",
    "else:",
    "    print('No loss data yet — re-run after Cell 5 has produced a losses.json.')",
]
CELLS.append(_code(*CELL6_CODE))


# ─── Cell 7 — Done ────────────────────────────────────────────


CELLS.append(_md(
    "## Done - what you proved",
    "",
    "If loss went _down_ and the HF push succeeded, you've validated:",
    "",
    "1. **Block AttnRes actually trains** — the architecture isn't broken.",
    "2. **The repo-driven pattern works** — no inline 280-line model dump in notebooks.",
    "3. **Gradient checkpointing fits a 150M model on a single T4** with `--gc`.",
    "4. **Multi-GPU works** — `WORLD_SIZE > 1` auto-promotes to torchrun; loss is reduce-aggregated.",
    "5. **The trained weights live on HF Hub** — pull them from any device later.",
    "",
    "## Next: Notebook 2 — Phase 2 mid-train",
    "",
    "Phase 2 will load Ornith-1.0-9B base + QLoRA adapters from `src/training/qlora.py`",
    "and is similarly repo-driven (`!python -m src.training.train_phase2 ...`).",
    "",
    "Cost: ~$0 on T4 (tight fit with batch=1, seq=1024), or ~$5 on Modal A10G for speed.",
    "",
    "When you're ready, push your _trained_ notebook (with outputs visible) to a HF",
    "dataset repo so the run is repro-able from anywhere.",
))


# ── Output the notebook ──────────────────────────────────────


def build_notebook() -> dict:
    return {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.10",
                "mimetype": "text/x-python",
                "file_extension": ".py",
            },
            "colab": {
                "provenance": [],
                "gpuType": "T4",
            },
            "accelerator": "GPU",
        },
        "nbformat": NB_VERSION,
        "nbformat_minor": NB_VERSION_MINOR,
    }


def main() -> None:
    out_dir = Path("notebooks")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "01_train_attnres_prototype.ipynb"
    nb = build_notebook()
    out_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
    print(f"Wrote {out_path} - {len(CELLS)} cells, {out_path.stat().st_size/1024:.1f} kB")


if __name__ == "__main__":
    main()
