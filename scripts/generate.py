#!/usr/bin/env python3
"""
Generate text from a trained AttnRes LM checkpoint.

Usage:
    python scripts/generate.py --checkpoint ./checkpoints/best --prompt "Once upon a time"
    python scripts/generate.py --checkpoint ./checkpoints/best --prompt "The future of AI is" --temperature 0.8 --top-k 50
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.model import AttnResLM
from src.inference import generate_text
from src.data.dataset import get_tokenizer


def main():
    parser = argparse.ArgumentParser(description="Generate text with the AttnRes LM")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint directory")
    parser.add_argument("--prompt", type=str, default="Once upon a time",
                        help="Text prompt to generate from")
    parser.add_argument("--max-tokens", type=int, default=200,
                        help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature (0 = greedy, 1.0 = normal)")
    parser.add_argument("--top-k", type=int, default=50,
                        help="Top-k sampling (0 to disable)")
    parser.add_argument("--top-p", type=float, default=0.9,
                        help="Top-p nucleus sampling threshold")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device to use (auto, cpu, cuda)")
    args = parser.parse_args()

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model = AttnResLM.from_pretrained(args.checkpoint)
    model = model.to(device)
    print(f"Model loaded: {model.num_parameters() / 1e6:.1f}M parameters")

    # Tokenizer
    tokenizer = get_tokenizer()

    # Generate
    print(f"\nPrompt: {args.prompt}")
    print(f"Generating {args.max_tokens} tokens (temp={args.temperature}, top_k={args.top_k}, top_p={args.top_p})...\n")

    text = generate_text(
        model, args.prompt, tokenizer,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k if args.top_k > 0 else None,
        top_p=args.top_p,
        device=device,
    )

    print(f"{'='*60}")
    print(text)
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
