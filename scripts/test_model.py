#!/usr/bin/env python3
"""
Quick test script — verify the model works before training.

Checks:
  1. Model initialization and parameter count
  2. Forward pass produces correct output shapes
  3. Backward pass computes gradients
  4. AttnRes vs standard residual comparison
  5. Text generation works
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.model import AttnResConfig, AttnResLM
from src.model.config import small_prototype
from src.inference import generate_text
from src.data.dataset import get_tokenizer


def test_forward_pass():
    """Test that forward pass produces correct shapes."""
    print("Testing forward pass...")
    config = small_prototype()
    model = AttnResLM(config)

    batch_size, seq_len = 2, 128
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    labels = input_ids.clone()

    output = model(input_ids, labels=labels)

    assert output.logits.shape == (batch_size, seq_len, config.vocab_size), \
        f"Expected logits shape {(batch_size, seq_len, config.vocab_size)}, got {output.logits.shape}"
    assert output.loss is not None, "Loss should be computed when labels are provided"
    assert output.loss.dim() == 0, "Loss should be a scalar"

    print(f"  ✓ Logits shape: {output.logits.shape}")
    print(f"  ✓ Loss: {output.loss.item():.4f}")
    print(f"  ✓ Parameters: {model.num_parameters() / 1e6:.1f}M")


def test_backward_pass():
    """Test that backward pass computes gradients for all parameters."""
    print("\nTesting backward pass...")
    config = small_prototype()
    config.num_hidden_layers = 4  # smaller for faster test
    model = AttnResLM(config)

    input_ids = torch.randint(0, config.vocab_size, (2, 64))
    labels = input_ids.clone()

    output = model(input_ids, labels=labels)
    output.loss.backward()

    # Check that all parameters have gradients
    no_grad = []
    for name, param in model.named_parameters():
        if param.grad is None:
            no_grad.append(name)
        elif param.grad.abs().sum().item() == 0:
            no_grad.append(f"{name} (zero grad)")

    if no_grad:
        print(f"  ⚠️  Parameters without gradients: {no_grad}")
    else:
        print("  ✓ All parameters have non-zero gradients")


def test_attn_res_vs_standard():
    """Compare AttnRes vs standard residuals — both should run without errors."""
    print("\nTesting AttnRes vs standard residuals...")

    for use_attn_res in [True, False]:
        config = small_prototype()
        config.use_attn_res = use_attn_res
        config.num_hidden_layers = 4
        model = AttnResLM(config)

        input_ids = torch.randint(0, config.vocab_size, (2, 64))
        labels = input_ids.clone()

        output = model(input_ids, labels=labels)
        label = "AttnRes" if use_attn_res else "Standard"
        print(f"  ✓ {label} residuals — loss: {output.loss.item():.4f}")


def test_generation():
    """Test text generation works end-to-end."""
    print("\nTesting text generation...")
    config = small_prototype()
    config.num_hidden_layers = 4
    model = AttnResLM(config)
    model.eval()

    tokenizer = get_tokenizer()
    prompt = "Hello, world!"

    text = generate_text(
        model, prompt, tokenizer,
        max_new_tokens=20,
        temperature=0.8,
        top_k=50,
    )

    assert len(text) > len(prompt), "Generated text should be longer than prompt"
    print(f"  ✓ Generated: {text[:100]}...")


def test_param_counts():
    """Print parameter counts for all preset configs."""
    print("\nParameter counts for preset configurations:")
    from src.model.config import small_prototype, small_hybrid, medium_1b, target_9b, target_9b_moe

    for name, config_fn in [
        ("small (prototype)", small_prototype),
        ("small_hybrid", small_hybrid),
        ("medium (1B)", medium_1b),
        ("target (9B dense)", target_9b),
        ("target_moe (30B/3B active)", target_9b_moe),
    ]:
        config = config_fn()
        info = config.estimate_num_params()
        attn_res = "AttnRes" if config.use_attn_res else "standard"
        moe = f"MoE({config.num_experts}e)" if config.use_moe else "dense"
        print(f"  {name:30s}: {info['total_M']:>10.1f}M params ({attn_res}, {moe})")


def main():
    print("=" * 60)
    print("AttnRes LM — Model Test Suite")
    print("=" * 60)

    test_forward_pass()
    test_backward_pass()
    test_attn_res_vs_standard()
    test_generation()
    test_param_counts()

    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()
