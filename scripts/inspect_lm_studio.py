"""
Inspect a running LM Studio server (or any OpenAI-compatible endpoint) —
print loaded model id, sampling defaults, and a sample completion so the user
knows exactly what they're working against.

Usage:
    python scripts/inspect_lm_studio.py
    python scripts/inspect_lm_studio.py --base-url http://127.0.0.1:1234/v1
    python scripts/inspect_lm_studio.py --prompt "Write fibonacci" --max-tokens 200
"""

from __future__ import annotations

import argparse
import json
import sys
from urllib import request, error

DEFAULT_BASE_URL = "http://127.0.0.1:1234"


def _post(base_url: str, path: str, payload: dict, timeout: int = 30) -> dict:
    url = base_url.rstrip("/") + path
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _get(base_url: str, path: str, timeout: int = 30) -> dict:
    url = base_url.rstrip("/") + path
    with request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def list_models(base_url: str) -> dict:
    return _get(base_url, "/v1/models")


def sample_completion(
    base_url: str,
    model_id: str,
    prompt: str,
    max_tokens: int,
) -> dict:
    """Use the chat completions endpoint to get one sample."""
    return _post(
        base_url,
        "/v1/chat/completions",
        {
            "model": model_id,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": 0.6,
            "top_p": 0.95,
        },
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--prompt", default="Write a Python function that returns the n-th Fibonacci number.")
    ap.add_argument("--max-tokens", type=int, default=300)
    ap.add_argument("--skip-sample", action="store_true", help="Don't run a completion test, just list models")
    args = ap.parse_args()

    print(f"Connecting to {args.base_url} ...\n")

    # 1. List loaded models
    try:
        info = list_models(args.base_url)
    except (error.URLError, error.HTTPError, ConnectionError) as e:
        print(f"[error] Could not reach {args.base_url}/v1/models: {e}")
        print("      Is LM Studio running? Is 'Local Server' enabled?")
        return 1

    print("== Loaded models ==")
    print(json.dumps(info, indent=2))
    print()

    # Extract first model id for a sample
    models = info.get("data") or []
    if not models:
        print("[warn] No models returned by /v1/models.")
        return 0

    model_id = models[0].get("id", "<unknown>")
    print(f"== Sample completion using model_id={model_id!r}")  # noqa
    if args.skip_sample:
        return 0

    try:
        result = sample_completion(args.base_url, model_id, args.prompt, args.max_tokens)
    except (error.URLError, error.HTTPError, ConnectionError) as e:
        print(f"[error] Sample completion failed: {e}")
        return 1

    # 2. Print the assistant's reply and any tool_calls (Ornith emits these)
    choices = result.get("choices") or []
    if not choices:
        print("[warn] No choices returned.")
        return 0
    msg = choices[0].get("message", {})
    print(f"-- reasoning_content --\n{msg.get('reasoning_content', '(none)')}\n")
    print(f"-- content --\n{msg.get('content', '(empty)')}\n")
    if msg.get("tool_calls"):
        print(f"-- tool_calls --\n{json.dumps(msg['tool_calls'], indent=2)}\n")

    # 3. Show usage so the user knows what their context budget is
    print("-- usage --")
    print(json.dumps(result.get("usage", {}), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
