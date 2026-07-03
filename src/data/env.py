"""
Secure environment-variable loading for the data pipeline.

The HF_TOKEN is read from the environment ONLY. We provide a small loader
utility that reads a local .env file (untracked) without printing the
secret, so it can be sourced before a training run without leaking.
"""

from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(env_path: str | os.PathLike = ".env") -> None:
    """
    Read KEY=value pairs from ``env_path`` and set them in os.environ ONLY
    if they are NOT already set. Lines starting with '#' and blank lines are
    ignored. Values are NEVER printed to stdout/stderr.

    Args:
        env_path: path to a .env file. By default "./.env" relative to CWD.
    """
    p = Path(env_path)
    if not p.exists():
        return
    with open(p, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            # Strip quoting if present
            if (
                len(value) >= 2
                and value[0] == value[-1]
                and value[0] in ("'", '"')
            ):
                value = value[1:-1]
            if key not in os.environ:
                os.environ[key] = value


def get_hf_token() -> str | None:
    """
    Return the Hugging Face token from the environment or None.

    Tries, in order:
      1. HF_TOKEN (Hugging Face standard)
      2. HUGGING_FACE_HUB_TOKEN (alternate)

    Always returns None rather than raising so that scripts can give a
    helpful message about setting the env var.
    """
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def require_hf_token() -> str:
    """Like get_hf_token but raise with a helpful message if missing."""
    token = get_hf_token()
    if not token:
        raise RuntimeError(
            "HF_TOKEN environment variable is not set.\n"
            "  - On macOS/Linux:  export HF_TOKEN=hf_xxx...\n"
            "  - On Windows:      set HF_TOKEN=hf_xxx...\n"
            "  - Or put it in a gitignored .env file and run:\n"
            "        from src.data.env import load_dotenv; load_dotenv()"
        )
    return token
