#!/usr/bin/env python3
"""
Training script for the AttnRes LM.

Usage:
    python scripts/train.py --config small           # Train the 150M prototype
    python scripts/train.py --config small --no-attn-res  # Ablation: standard residuals
    python scripts/train.py --config medium          # Train the 1B model
    python scripts/train.py --config target          # Train the 9B target (needs multi-GPU)

For Colab/Kaggle (single T4 16GB):
    python scripts/train.py --config small --max-steps 5000 --batch-size 4
"""

import argparse
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.model import AttnResConfig, AttnResLM
from src.model.config import small_prototype, small_hybrid, medium_1b, target_9b, target_9b_moe
from src.data import TextDataset, create_dataloader
from src.training.trainer import Trainer, TrainingConfig


def get_config(name: str) -> AttnResConfig:
    """Get a preset configuration by name."""
    configs = {
        "small": small_prototype,
        "small_hybrid": small_hybrid,
        "medium": medium_1b,
        "target": target_9b,
        "target_moe": target_9b_moe,
    }
    if name not in configs:
        raise ValueError(f"Unknown config: {name}. Choose from {list(configs.keys())}")
    return configs[name]()


def main():
    parser = argparse.ArgumentParser(description="Train the AttnRes LM")
    parser.add_argument("--config", type=str, default="small",
                        choices=["small", "small_hybrid", "medium", "target", "target_moe"],
                        help="Model configuration preset")
    parser.add_argument("--data", type=str, default=None,
                        help="Path to training data (text file)")
    parser.add_argument("--eval-data", type=str, default=None,
                        help="Path to evaluation data (text file)")
    parser.add_argument("--output-dir", type=str, default="./checkpoints",
                        help="Directory to save checkpoints")
    parser.add_argument("--max-steps", type=int, default=10000,
                        help="Maximum training steps")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Per-device batch size")
    parser.add_argument("--grad-accum", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--warmup", type=int, default=100,
                        help="Warmup steps")
    parser.add_argument("--no-attn-res", action="store_true",
                        help="Disable AttnRes (use standard residuals) — for ablation")
    parser.add_argument("--use-wandb", action="store_true",
                        help="Enable WandB logging")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Build model config
    config = get_config(args.config)
    if args.no_attn_res:
        config.use_attn_res = False
        print("⚠️  AttnRes DISABLED — using standard residual connections (ablation mode)")

    # Print model info
    param_info = config.estimate_num_params()
    print(f"\n{'='*60}")
    print(f"AttnRes LM — Model Configuration")
    print(f"{'='*60}")
    print(f"  Config: {args.config}")
    print(f"  Hidden size: {config.hidden_size}")
    print(f"  Layers: {config.num_hidden_layers}")
    print(f"  Heads: {config.num_attention_heads} (Q) / {config.num_key_value_heads} (KV)")
    print(f"  AttnRes: {'enabled' if config.use_attn_res else 'disabled'}")
    if config.use_attn_res:
        print(f"  AttnRes blocks: {config.num_blocks} (block_size={config.attn_res_block_size})")
    print(f"  MoE: {'enabled' if config.use_moe else 'disabled'}")
    print(f"  Estimated params: {param_info['total_M']:.1f}M ({param_info['total_B']:.2f}B)")
    print(f"  Max position: {config.max_position_embeddings}")
    print(f"{'='*60}\n")

    # Build model
    model = AttnResLM(config)
    actual_params = model.num_parameters()
    print(f"Actual parameters: {actual_params / 1e6:.1f}M ({actual_params / 1e9:.3f}B)")

    # Build datasets
    if args.data is None:
        print("No data path provided. Using a small synthetic dataset for testing.")
        # Create a tiny synthetic dataset for testing
        texts = [
            "The quick brown fox jumps over the lazy dog. " * 20,
            "In machine learning, attention mechanisms allow models to focus on relevant parts. " * 20,
            "Residual connections help train deep networks by providing gradient pathways. " * 20,
            "Attention residuals replace fixed accumulation with learned depth-wise selection. " * 20,
        ] * 100
        train_dataset = TextDataset(texts=texts, max_length=config.max_position_embeddings)
        eval_dataset = TextDataset(texts=texts[:20], max_length=config.max_position_embeddings)
    else:
        train_dataset = TextDataset(file_path=args.data, max_length=config.max_position_embeddings)
        eval_dataset = TextDataset(file_path=args.eval_data, max_length=config.max_position_embeddings) if args.eval_data else None

    print(f"Training sequences: {len(train_dataset)}")

    train_loader = create_dataloader(train_dataset, batch_size=args.batch_size, shuffle=True)
    eval_loader = create_dataloader(eval_dataset, batch_size=args.batch_size, shuffle=False) if eval_dataset else None

    # Training config
    train_config = TrainingConfig(
        learning_rate=args.lr,
        warmup_steps=args.warmup,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        output_dir=args.output_dir,
        use_wandb=args.use_wandb,
    )

    # Print training info
    print(f"\nTraining config:")
    print(f"  Effective batch size: {args.batch_size * args.grad_accum}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Max steps: {args.max_steps}")
    print(f"  Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU memory: {mem:.1f} GB")
    print()

    # Train!
    trainer = Trainer(model, train_config, train_loader, eval_loader)
    trainer.train()


if __name__ == "__main__":
    main()
