#!/usr/bin/env python3
"""
Download datasets for the AttnRes LM training pipeline.

SECURITY: Set the HF_TOKEN environment variable before running. NEVER hardcode it.
  export HF_TOKEN=hf_xxx...      # bash / zsh
  set -x HF_TOKEN hf_xxx...      # fish
  put it in a local .env file (gitignored) and source it

Dataset routing:
  - Large pre-training corpora are streamed (chunkwise) — only a configurable
    number of samples are pulled to the local cloud cache.
  - Mid-training / DPO / evaluation datasets are downloaded fully (small enough).

USAGE:
  # Pre-training: stream FineWeb-Edu sample-10BT (10B tokens) + StarCoder (python+shell)
  python scripts/download_data.py --phase pretrain --max-samples 2000000

  # Mid-training: download OpenHermes-2.5 + Hermes function-calling dataset.
  python scripts/download_data.py --phase midtrain

  # DPO: ultrafeedback_binarized + OpenCodeInterpreter preferences.
  python scripts/download_data.py --phase dpo

  # Eval: HumanEval + SWE-bench Lite (test sets).
  python scripts/download_data.py --phase eval
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _login_if_token():
    """Login to Hugging Face Hub if HF_TOKEN env var is set.

    SECURITY: token is read from environment variable ONLY. Never accept
    tokens via CLI args or hardcode them. Never print the token.
    """
    try:
        from src.data.env import get_hf_token
        token = get_hf_token()
    except ImportError:
        token = os.environ.get("HF_TOKEN")

    if not token:
        print(
            "WARNING: HF_TOKEN not set. Only public datasets will work.\n"
            "  Set the env var to access gated datasets:\n"
            "    export HF_TOKEN=hf_xxx...\n"
            "  To avoid leaking the token, keep it in an untracked .env file:\n"
            "    echo 'HF_TOKEN=hf_xxx...' > .env  # gitignored"
        )
        return False
    # Avoid logging the token!
    print(f"Logging in to Hugging Face Hub (token length: {len(token)} chars)")
    try:
        from huggingface_hub import login
        login(token=token, add_to_git_credential=False)
        return True
    except ImportError:
        print("huggingface_hub not installed; skipping login")
        return False


def load_phase_datasets(phase: str, config_path: str) -> list:
    """Load the dataset spec list for ``phase`` from the YAML config."""
    import yaml
    p = Path(config_path)
    if not p.exists():
        raise FileNotFoundError(
            f"Dataset config not found: {config_path}. "
            f"The default is `config/datasets.yaml` relative to the project root."
        )
    with open(p, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg.get(phase) or []


def download_phase(
    phase: str,
    max_samples: Optional[int],
    cache_dir: Path,
    output_dir: Path,
    datasets: list,
) -> None:
    """Download datasets for one phase and write summaries to disk."""
    from datasets import load_dataset

    out_dir = output_dir / phase
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries = []

    for spec in datasets:
        print(f"\n[{phase}] {spec.get('name', spec['id'])}")
        print(f"  id: {spec['id']}  config: {spec.get('config')}  split: {spec['split']}")
        try:
            ds = load_dataset(
                spec["id"],
                spec.get("config"),
                split=spec["split"],
                streaming=spec.get("streaming", False),
                cache_dir=str(cache_dir),
                trust_remote_code=False,
            )
        except Exception as e:
            print(f"  FAILED to load: {type(e).__name__}: {e}")
            summaries.append({**spec, "status": "failed", "error": str(e)})
            continue

        if spec.get("streaming", False) and max_samples is not None:
            # Take only the first N samples from the stream — saves download + memory
            ds = ds.take(max_samples)

        # Materialize to a JSONL shard. Stream-write line by line so we never
        # hold the entire dataset in memory.
        shard_path = out_dir / f"{spec['id'].replace('/', '_')}.jsonl"
        try:
            n = 0
            with open(shard_path, "w", encoding="utf-8") as f:
                text_field = spec.get("text_field", "text")
                for ex in ds:
                    text = ex.get(text_field) or ex.get("content") or ""
                    if not text.strip():
                        continue
                    f.write(text.replace("\n", " ")[:4096] + "\n")
                    n += 1
            print(f"  wrote {n} lines to {shard_path}")
            summaries.append({**spec, "status": "ok", "samples": n, "shard": str(shard_path)})
        except Exception as e:
            print(f"  FAILED during write: {type(e).__name__}: {e}")
            summaries.append({**spec, "status": "error", "error": str(e)})

    import json
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"\n[{phase}] Summary: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", type=str, required=True,
        choices=["pretrain", "midtrain", "dpo", "eval"],
        help="Which training phase to download for",
    )
    parser.add_argument(
        "--max-samples", type=int, default=2_000_000,
        help="For streaming datasets, take at most this many samples (default 2M).",
    )
    parser.add_argument(
        "--cache-dir", type=str, default="./data/raw",
        help="Where HuggingFace stores raw downloaded shards.",
    )
    parser.add_argument(
        "--output-dir", type=str, default="./data/processed",
        help="Where JSONL shards get written.",
    )
    parser.add_argument(
        "--config", type=str, default="config/datasets.yaml",
        help="YAML config with dataset specs per phase.",
    )
    args = parser.parse_args()

    _login_if_token()

    cache_dir = Path(args.cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Phase:    {args.phase}")
    print(f"Config:   {args.config}")
    print(f"Cache:    {cache_dir}")
    print(f"Output:   {output_dir}")

    datasets = load_phase_datasets(args.phase, args.config)
    if not datasets:
        print(f"ERROR: no datasets defined for phase '{args.phase}' in {args.config}")
        sys.exit(1)

    download_phase(args.phase, args.max_samples, cache_dir, output_dir, datasets)


if __name__ == "__main__":
    main()
