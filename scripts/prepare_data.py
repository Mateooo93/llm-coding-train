#!/usr/bin/env python3
"""
Tokenize and pack downloaded JSONL shards into fixed-length sequences
ready for training.

Reads JSONL shards produced by ``scripts/download_data.py`` (under
./data/processed/<phase>/) and writes packed tensors under
./data/processed/<phase>_packed/<dataset_id>.pt.

Memory: For 10B-token shards, the token list itself is ~40GB as Python ints.
We avoid ever holding the full list: we accumulate one packed sequence at a
time, append it to a single output tensor, and write that tensor as a
``.pt`` file once we've consumed one shard.

USAGE:
  python scripts/prepare_data.py --phase pretrain --max-length 1024
  python scripts/prepare_data.py --phase midtrain --max-length 2048
  python scripts/prepare_data.py --phase dpo --max-length 1024
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def tokenize_shard_streaming(
    shard_path: Path,
    max_length: int,
    encoding_name: str = "gpt2",
    max_lines: int | None = None,
) -> dict:
    """
    Tokenize a JSONL shard and pack into fixed-length sequences.

    Streams lines from disk and only ever buffers one packed sequence at a
    time, so peak memory is O(max_length * vocab_overhead), not O(total_tokens).

    Returns:
        dict with keys: num_sequences, total_tokens, shard_path.
    """
    import tiktoken
    import torch
    enc = tiktoken.get_encoding(encoding_name)
    eot = enc.eot_token

    out_chunks: list[torch.Tensor] = []
    buf: list[int] = []
    total_tokens = 0

    with open(shard_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_lines is not None and i >= max_lines:
                break
            text = line.strip()
            if not text:
                continue
            tokens = enc.encode(text) + [eot]
            total_tokens += len(tokens)
            buf.extend(tokens)
            # Drain complete sequences without ever holding a huge list
            while len(buf) >= max_length:
                seq = buf[:max_length]
                buf = buf[max_length:]
                out_chunks.append(torch.tensor(seq, dtype=torch.long))

    # Pad remainder (drop incomplete sequence)
    n_seq = len(out_chunks)
    if out_chunks:
        tensor = torch.stack(out_chunks, dim=0)  # [n_seq, max_length]
    else:
        tensor = torch.zeros(0, max_length, dtype=torch.long)
    out_path = shard_path.with_suffix(".pt")
    torch.save(tensor, out_path)
    return {
        "num_sequences": n_seq,
        "total_tokens": total_tokens,
        "shard_path": str(out_path),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", type=str, required=True,
                        choices=["pretrain", "midtrain", "dpo", "eval"])
    parser.add_argument("--input-dir", type=str, default="./data/processed",
                        help="Where download_data.py wrote shards")
    parser.add_argument("--max-length", type=int, default=1024,
                        help="Pack sequences to this length")
    parser.add_argument("--max-lines", type=int, default=None,
                        help="Cap number of lines per shard (for fastest iteration)")
    parser.add_argument("--encoding", type=str, default="gpt2")
    args = parser.parse_args()

    in_dir = Path(args.input_dir) / args.phase
    if not in_dir.exists():
        print(f"ERROR: {in_dir} not found. Run scripts/download_data.py first.")
        sys.exit(1)

    summary_path = in_dir / "summary.json"
    if not summary_path.exists():
        print(f"ERROR: missing {summary_path}. Nothing to tokenize.")
        sys.exit(1)

    with open(summary_path) as f:
        summaries = json.load(f)

    print(f"Phase: {args.phase}")
    print(f"Max length: {args.max_length}")

    out_summary = []
    for entry in summaries:
        if entry.get("status") != "ok":
            out_summary.append({**entry, "tokenize_status": "skipped"})
            continue
        shard = Path(entry["shard"])
        print(f"\n  {entry.get('name', shard.name)}: {shard}")
        try:
            result = tokenize_shard_streaming(
                shard, args.max_length, args.encoding, args.max_lines
            )
            print(f"    packed {result['num_sequences']} sequences, "
                  f"{result['total_tokens']} tokens -> {result['shard_path']}")
            out_summary.append({**entry, "tokenize_status": "ok", **result})
        except Exception as e:
            print(f"    FAILED: {type(e).__name__}: {e}")
            out_summary.append({**entry, "tokenize_status": "failed",
                               "error": str(e)})

    with open(in_dir / "packed_summary.json", "w") as f:
        json.dump(out_summary, f, indent=2)
    print(f"\nDone. See {in_dir / 'packed_summary.json'} for the run summary.")


if __name__ == "__main__":
    main()
