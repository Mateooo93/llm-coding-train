# AttnRes LM — A Revolutionary Language Model with Attention Residuals

A from-scratch implementation of a decoder-only LLM featuring **Block Attention Residuals (AttnRes)** — a novel architecture that replaces fixed unit-weight residual connections with learned, input-dependent softmax attention over preceding layer outputs.

> **The Problem:** Standard PreNorm transformers accumulate all layer outputs with fixed unit weights (`h_l = h_{l-1} + Layer(h_{l-1})`). This causes uncontrolled hidden-state growth with depth, progressively diluting each layer's contribution.
>
> **The Solution:** AttnRes replaces this with softmax attention over preceding block-level representations, allowing each layer to *selectively aggregate* earlier representations with learned, input-dependent weights.

Based on the paper ["Attention Residuals" (Chen et al., Moonshot AI, 2026)](https://arxiv.org/abs/2603.15031), which showed AttnRes is performance-equivalent to a baseline model trained with **1.25× more compute**.

## Architecture

```
Input IDs → Token Embedding
         → [AttnRes Transformer Block × N]
              ├── Block AttnRes (attention aggregation)
              ├── GQA Self-Attention (RoPE)
              ├── Block AttnRes (MLP aggregation)
              └── SwiGLU MLP (or MoE)
         → Final RMSNorm
         → LM Head → Logits
```

### Key Innovations

| Feature | Standard LLM | AttnRes LM |
|---|---|---|
| **Residual connections** | Fixed unit weights: `h + Layer(h)` | Learned softmax attention over blocks |
| **Depth-wise selection** | None — all layers weighted equally | Each layer *chooses* which earlier representations to use |
| **Hidden-state growth** | Uncontrolled with depth | Bounded by attention normalization |
| **Gradient distribution** | Uneven across depth | More uniform (shown in paper) |

### Modern Architecture Stack

- **RoPE** — Rotary Position Embeddings
- **GQA** — Grouped Query Attention (efficient KV-cache)
- **SwiGLU** — Swish-Gated Linear Unit activation
- **RMSNorm** — Pre-normalization without mean-centering
- **Block AttnRes** — Learned depth-wise residual aggregation
- **Linear Attention** — O(L) sub-quadratic attention option (elu+1 feature map)
- **Hybrid Attention** — Mix of full softmax + linear attention layers (like Kimi Linear)
- **MoE-ready** — Mixture-of-Experts stub for future scaling

### Two Breakthroughs Combined

| Innovation | What it changes | Benefit |
|---|---|---|
| **AttnRes** | Residual connections (depth-wise aggregation) | Each layer selects which earlier blocks to draw from — 1.25× compute equivalent |
| **Linear Attention** | Attention mechanism (sequence-wise scaling) | O(L) instead of O(L²) — sub-quadratic long context, inspired by SubQ/SSA |
| **Hybrid mode** | Mixes both attention types per layer | Full attention for precise retrieval + linear for long-context efficiency |

## Quickstart

### Install

```bash
pip install -e .
```

### Test the model

```bash
python scripts/test_model.py
```

### Train the prototype (~150M params, fits on a T4)

```bash
# Using synthetic data for testing
python scripts/train.py --config small --max-steps 100

# With real data
python scripts/train.py --config small --data ./data/train.txt --max-steps 5000
```

### Generate text

```bash
python scripts/generate.py --checkpoint ./checkpoints/small/final --prompt "Once upon a time"
```

### Ablation: Compare AttnRes vs standard residuals

```bash
# With AttnRes (the innovation)
python scripts/train.py --config small --output-dir ./checkpoints/attn_res

# Without AttnRes (standard baseline)
python scripts/train.py --config small --no-attn-res --output-dir ./checkpoints/standard
```

## Model Configurations

| Config | Parameters | Hidden | Layers | AttnRes Blocks | Target Hardware |
|---|---|---|---|---|---|
| `small` | ~150M | 768 | 12 | 6 | Single T4 (16GB) |
| `small_hybrid` | ~150M | 768 | 12 | 6 | Single T4 (16GB) — hybrid attention |
| `medium` | ~1B | 2048 | 24 | 8 | Single A100 (40GB) |
| `target` | ~9B | 4096 | 36 | 9 | 8× 24GB GPUs |
| `target_moe` | ~30B total / 3B active | 4096 | 36 | 9 | 8× 24GB GPUs + expert parallel |

## Project Structure

```
attnres-lm/
├── src/
│   ├── model/
│   │   ├── config.py           # Model configuration + presets
│   │   ├── attn_res.py         # ★ Block AttnRes — the core innovation
│   │   ├── attention.py        # GQA with RoPE (standard O(L²) attention)
│   │   ├── linear_attention.py # ★ Linear Attention — O(L) sub-quadratic
│   │   ├── mlp.py              # SwiGLU MLP
│   │   ├── moe.py              # Mixture-of-Experts (future scaling)
│   │   ├── transformer_block.py # Block integrating AttnRes + attention type
│   │   └── model.py            # Full AttnRes LM
│   ├── data/
│   │   └── dataset.py          # Text dataset + tokenization
│   ├── training/
│   │   └── trainer.py          # Training loop with AMP, grad accum, checkpointing
│   └── inference/
│       └── generate.py         # Text generation (greedy, top-k, top-p)
├── config/
│   ├── small_prototype.yaml    # 150M config
│   ├── target_9b.yaml          # 9B dense config
│   └── target_9b_moe.yaml      # 30B/3B MoE config
├── scripts/
│   ├── train.py                # Training entry point
│   ├── generate.py             # Generation entry point
│   └── test_model.py           # Quick model validation
├── tests/
│   └── test_model.py           # Unit tests
└── requirements.txt
```

## How Block AttnRes Works

```
Standard Residual:              Block AttnRes:

  h₀ → Layer₀ → h₁               h₀ ──┐
  h₁ → Layer₁ → h₂               h₁ ──┤ (block 0)
  h₂ → Layer₂ → h₃               h₂ ──┘
  ...                              ↓ save block rep
  hₙ₋₁ → Layerₙ₋₁ → hₙ           h₃ ← AttnRes(block₀, block₁, ...) ← Layer
                                   ↓
                                  h₄ ← AttnRes(block₀, block₁, ...) ← Layer
                                   ↓ save block rep
                                  ...

  All layers weighted equally    Each layer selects which
  (fixed unit weight)            blocks to attend to (learned weights)
```

The pseudo-query vector `w_l ∈ R^d` for each sublayer computes:
```
α_{l,i} = softmax(w_l · RMSNorm(block_rep_i))   # attention weight for block i
h_l = Σ_i α_{l,i} · block_rep_i                  # weighted aggregation
```

## Roadmap

- [x] Core AttnRes + Block AttnRes implementation
- [x] GQA attention with RoPE
- [x] SwiGLU MLP
- [x] MoE layer (stub for future scaling)
- [x] Training pipeline (AMP, grad accumulation, checkpointing)
- [x] Text generation (greedy, top-k, top-p)
- [ ] KV-cache for efficient generation
- [ ] Multi-GPU training (accelerate/DeepSpeed config)
- [ ] Custom 128K vocabulary tokenizer
- [ ] Data pipeline for large-scale pretraining
- [ ] Evaluation benchmarks (HellaSwag, MMLU, etc.)
- [ ] Scaling experiments: AttnRes vs standard at multiple scales

## References

- **Attention Residuals** — Chen et al., Moonshot AI, 2026. [arXiv:2603.15031](https://arxiv.org/abs/2603.15031)
- [Official code](https://github.com/MoonshotAI/Attention-Residuals)
- RoPE — Su et al., 2021. [arXiv:2104.09864](https://arxiv.org/abs/2104.09864)
- GQA — Ainslie et al., 2023. [arXiv:2305.13245](https://arxiv.org/abs/2305.13245)
- SwiGLU — Shazeer, 2020. [arXiv:2002.05202](https://arxiv.org/abs/2002.05202)
- RMSNorm — Zhang & Sennrich, 2019. [arXiv:1910.07467](https://arxiv.org/abs/1910.07467)

## License

This project (code and any model checkpoints released in this repo) is licensed under the [Apache License 2.0](LICENSE). Copyright 2025 The AttnRes LM Authors.

### Data Attribution

The `notebooks/01_train_attnres_prototype.ipynb` notebook trains on data sampled from [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) (`sample-10BT` config), which is distributed under the [Open Data Commons Attribution License (ODC-By) 1.0](https://opendatacommons.org/licenses/by/). FineWeb-Edu is built on top of [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb), itself derived from [Common Crawl](https://commoncrawl.org/). Model weights derived from this data are released under Apache 2.0 in this repository; downstream users are invited (but not required under ODC-By) to credit FineWeb-Edu when redistributing derived artifacts.
