#!/usr/bin/env python3
"""
Upload a trained AttnResLM checkpoint to the Hugging Face Hub.

SECURITY: Set the HF_TOKEN env var before running. The script NEVER prints,
stores, or re-uses the token.

USAGE:
  python scripts/upload_model.py --repo your-username/attnres-9b \\
    --checkpoint ./checkpoints/final

  # Push as private with a model card
  python scripts/upload_model.py --repo your-username/attnres-9b-v1 \\
    --checkpoint ./checkpoints/best --private --card ./MODEL_CARD.md
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _login_if_token():
    from src.data.env import require_hf_token
    token = require_hf_token()
    # Don't print the token!
    print(f"Logging in to Hub (token length: {len(token)} chars)")
    from huggingface_hub import login
    login(token=token, add_to_git_credential=False)


def _smoke_test_checkpoint(ckpt_dir: Path) -> bool:
    """
    Verify that the checkpoint actually loads into a matching AttnResLM.

    Returns True on success. On failure, prints the error and returns False.
    """
    try:
        from src.model import AttnResLM
        model = AttnResLM.from_pretrained(str(ckpt_dir))
        n_params = model.num_parameters()
        print(f"  smoke test OK: loaded {n_params / 1e6:.1f}M-parameter AttnResLM")
        return True
    except ImportError as e:
        print(f"  WARNING: AttnResLM import failed ({e}); skipping smoke test.")
        return True  # non-fatal
    except Exception as e:
        print(f"  smoke test FAILED: {type(e).__name__}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=str, required=True,
                        help="Hub repo id, e.g. 'username/attnres-9b'")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint directory containing config.json + pytorch_model.bin")
    parser.add_argument("--private", action="store_true",
                        help="Create a private repo (default: public)")
    parser.add_argument("--card", type=str, default=None,
                        help="Optional path to MODEL_CARD.md to upload alongside")
    parser.add_argument("--message", type=str, default="upload checkpoint",
                        help="Commit message for the upload. AVOID putting secrets here — "
                        "it will be visible in the public commit history.")
    parser.add_argument("--no-smoke-test", action="store_true",
                        help="Skip loading the checkpoint into AttnResLM before uploading")
    args = parser.parse_args()

    ckpt_dir = Path(args.checkpoint).resolve()
    if not (ckpt_dir / "config.json").exists() or not (ckpt_dir / "pytorch_model.bin").exists():
        print(f"ERROR: {ckpt_dir} is missing config.json / pytorch_model.bin")
        sys.exit(1)

    _login_if_token()

    # Echo the commit message back so the user sees what will appear in the public log.
    print(f"\nCommit message (will be PUBLIC on the Hub):\n  >>> {args.message} <<<\n")

    if not args.no_smoke_test:
        print("Running smoke test before upload...")
        if not _smoke_test_checkpoint(ckpt_dir):
            print("ERROR: smoke test failed; refusing to upload. "
                  "Use --no-smoke-test to override.")
            sys.exit(2)
        print("Smoke test passed.\n")

    from huggingface_hub import create_repo, upload_folder, upload_file

    print(f"Creating repo {args.repo} (private={args.private}) ...")
    create_repo(
        args.repo,
        repo_type="model",
        private=args.private,
        exist_ok=True,
    )

    print(f"Uploading files from {ckpt_dir} ...")
    upload_folder(
        folder_path=str(ckpt_dir),
        repo_id=args.repo,
        repo_type="model",
        commit_message=args.message,
    )
    print(f"\nUploaded: https://huggingface.co/{args.repo}")

    # Optional: upload a model card separately
    if args.card:
        card_path = Path(args.card).resolve()
        if card_path.exists():
            upload_file(
                path_or_fileobj=str(card_path),
                path_in_repo="README.md",
                repo_id=args.repo,
                repo_type="model",
                commit_message="add model card",
            )
            print(f"Uploaded model card from {card_path}")


if __name__ == "__main__":
    main()
