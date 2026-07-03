"""
Configuration for the AttnRes Language Model.

This config supports both small prototype models (~150M) and the full 9B target,
with Block AttnRes as a drop-in replacement for standard residual connections.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AttnResConfig:
    """
    Configuration for the AttnRes LM.

    Architecture: decoder-only transformer with:
      - RoPE positional embeddings
      - Grouped Query Attention (GQA)
      - SwiGLU MLP
      - RMSNorm (PreNorm)
      - Block Attention Residuals (AttnRes) — the core innovation

    AttnRes replaces fixed unit-weight residual connections with learned,
    input-dependent softmax attention over preceding block-level representations.
    """

    # ── Vocabulary ──────────────────────────────────────────────
    vocab_size: int = 50304  # rounded to 64 multiple for efficiency (GPT-2 BPE)
    # ── Model dimensions ───────────────────────────────────────
    hidden_size: int = 768  # d_model
    intermediate_size: int = 2048  # d_ff (SwiGLU: ~8/3 * hidden_size, rounded)
    num_hidden_layers: int = 12

    # ── Attention ───────────────────────────────────────────────
    num_attention_heads: int = 12  # query heads
    num_key_value_heads: int = 4  # KV heads for GQA (must divide num_attention_heads)
    head_dim: int = 64  # dimension per head (hidden_size / num_attention_heads)

    # Attention type: "full" (standard softmax O(L²)), "linear" (O(L) linear
    # attention with elu+1 feature map), or "hybrid" (mix of both, like Kimi
    # Linear / Jamba). "hybrid" uses full attention every N layers and linear
    # attention in between, combining precise retrieval with long-context efficiency.
    attention_type: str = "full"  # "full", "linear", or "hybrid"
    hybrid_full_attention_interval: int = 6  # in hybrid mode, every Nth layer uses full attention
    linear_attention_use_rope: bool = True  # apply RoPE in linear attention layers

    # ── RoPE ────────────────────────────────────────────────────
    rope_theta: float = 10000.0  # base frequency for RoPE
    max_position_embeddings: int = 2048

    # ── AttnRes (the core innovation) ──────────────────────────
    use_attn_res: bool = True  # toggle AttnRes vs standard residuals
    attn_res_block_size: int = 4  # layers per AttnRes block (block_size // 2 = sublayers per block)
    attn_res_init_scale: float = 0.02  # init scale for pseudo-query vectors

    # ── Normalization ───────────────────────────────────────────
    rms_norm_eps: float = 1e-6

    # ── Dropout ─────────────────────────────────────────────────
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0

    # ── Embedding ───────────────────────────────────────────────
    tie_word_embeddings: bool = True  # share weights between input embeddings and LM head

    # ── MoE (for future scaling — not active in prototype) ──────
    use_moe: bool = False
    num_experts: int = 8
    num_experts_per_tok: int = 2
    moe_layer_interval: int = 2  # every Nth layer uses MoE instead of dense MLP

    # ── Training helpers ────────────────────────────────────────
    gradient_checkpointing: bool = False

    # ── Misc ────────────────────────────────────────────────────
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2

    def __post_init__(self):
        """Validate configuration consistency."""
        assert self.attention_type in ("full", "linear", "hybrid"), (
            f"attention_type ({self.attention_type}) must be 'full', 'linear', or 'hybrid'"
        )
        assert self.hidden_size % self.num_attention_heads == 0, (
            f"hidden_size ({self.hidden_size}) must be divisible by "
            f"num_attention_heads ({self.num_attention_heads})"
        )
        assert self.num_attention_heads % self.num_key_value_heads == 0, (
            f"num_attention_heads ({self.num_attention_heads}) must be divisible by "
            f"num_key_value_heads ({self.num_key_value_heads}) for GQA"
        )
        # head_dim can be explicitly set or derived
        computed_head_dim = self.hidden_size // self.num_attention_heads
        if self.head_dim == 0:
            self.head_dim = computed_head_dim
        assert self.head_dim == computed_head_dim, (
            f"head_dim ({self.head_dim}) must equal "
            f"hidden_size // num_attention_heads ({computed_head_dim})"
        )
        if self.use_attn_res:
            assert self.attn_res_block_size >= 2, (
                f"attn_res_block_size ({self.attn_res_block_size}) must be >= 2"
            )
            # block_size counts both ATTN + MLP sublayers, so layers_per_block = block_size // 2
            assert self.attn_res_block_size % 2 == 0, (
                f"attn_res_block_size ({self.attn_res_block_size}) must be even "
                f"(counts both attention and MLP sublayers)"
            )

    @property
    def sublayers_per_block(self) -> int:
        """Number of sublayers (attention or MLP) per AttnRes block."""
        return self.attn_res_block_size // 2

    @property
    def num_blocks(self) -> int:
        """Total number of AttnRes blocks."""
        return self.num_hidden_layers * 2 // self.attn_res_block_size

    def estimate_num_params(self) -> dict:
        """Estimate parameter count for this configuration."""
        d = self.hidden_size
        d_ff = self.intermediate_size
        n_layers = self.num_hidden_layers
        vocab = self.vocab_size

        # Token embedding (counted once if tied)
        embedding = vocab * d
        if not self.tie_word_embeddings:
            lm_head = vocab * d
        else:
            lm_head = 0

        # Per-layer attention: Q, K, V, O projections
        qkv = d * (self.num_attention_heads * self.head_dim)  # Q
        qkv += d * (self.num_key_value_heads * self.head_dim)  # K
        qkv += d * (self.num_key_value_heads * self.head_dim)  # V
        qkv += (self.num_attention_heads * self.head_dim) * d  # O
        attn_per_layer = qkv

        # Per-layer SwiGLU MLP: 3 projections (gate, up, down)
        # SwiGLU: gate(d -> d_ff), up(d -> d_ff), down(d_ff -> d)
        mlp_per_layer = 3 * d * d_ff

        # Per-layer norms (2 RMSNorm per layer: attn + MLP)
        norms_per_layer = 4 * d  # 2 norms, each has weight of size d

        # AttnRes pseudo-query vectors: 2 per layer (one for attn, one for MLP)
        attn_res_per_layer = 2 * d if self.use_attn_res else 0

        total_per_layer = attn_per_layer + mlp_per_layer + norms_per_layer + attn_res_per_layer
        total = embedding + lm_head + n_layers * total_per_layer

        return {
            "embedding": embedding,
            "lm_head": lm_head,
            "attention_per_layer": attn_per_layer,
            "mlp_per_layer": mlp_per_layer,
            "norms_per_layer": norms_per_layer,
            "attn_res_per_layer": attn_res_per_layer,
            "total_per_layer": total_per_layer,
            "layers_total": n_layers * total_per_layer,
            "total": total,
            "total_M": total / 1e6,
            "total_B": total / 1e9,
        }


# ── Preset configurations ──────────────────────────────────────

def small_prototype() -> AttnResConfig:
    """~150M parameter model for validating AttnRes on limited hardware (T4/Colab)."""
    return AttnResConfig(
        vocab_size=50304,
        hidden_size=768,
        intermediate_size=2048,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_key_value_heads=4,
        head_dim=64,
        max_position_embeddings=1024,
        use_attn_res=True,
        attn_res_block_size=4,  # 2 layers per block → 6 blocks total
        tie_word_embeddings=True,
        attention_type="full",  # standard attention for the prototype
    )


def medium_1b() -> AttnResConfig:
    """~1B parameter model for more serious experiments."""
    return AttnResConfig(
        vocab_size=50304,
        hidden_size=2048,
        intermediate_size=5632,
        num_hidden_layers=24,
        num_attention_heads=16,
        num_key_value_heads=4,
        head_dim=128,
        max_position_embeddings=4096,
        rope_theta=10000.0,
        use_attn_res=True,
        attn_res_block_size=6,  # 3 layers per block → 8 blocks total
        tie_word_embeddings=False,
    )


def target_9b() -> AttnResConfig:
    """
    ~9B parameter target model — the full revolutionary architecture.
    Requires multi-GPU training (8x24GB minimum with ZeRO-3 + grad checkpointing).
    """
    return AttnResConfig(
        vocab_size=128256,  # larger vocab like Llama 3
        hidden_size=4096,
        intermediate_size=14336,
        num_hidden_layers=36,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
        max_position_embeddings=8192,
        rope_theta=500000.0,
        use_attn_res=True,
        attn_res_block_size=8,  # 4 layers per block → 9 blocks total
        attn_res_init_scale=0.01,
        tie_word_embeddings=False,
        gradient_checkpointing=True,
    )


def target_9b_moe() -> AttnResConfig:
    """
    ~30B total / 3B active MoE model with AttnRes — matching the Kimi Linear approach.
    Each MoE layer has 8 experts, 2 active per token.
    Uses hybrid attention (mix of full softmax + linear attention), like Kimi Linear.
    """
    return AttnResConfig(
        vocab_size=128256,
        hidden_size=4096,
        intermediate_size=14336,
        num_hidden_layers=36,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
        max_position_embeddings=8192,
        rope_theta=500000.0,
        use_attn_res=True,
        attn_res_block_size=8,
        attn_res_init_scale=0.01,
        tie_word_embeddings=False,
        gradient_checkpointing=True,
        use_moe=True,
        num_experts=8,
        num_experts_per_tok=2,
        moe_layer_interval=2,
        attention_type="hybrid",  # mix of full + linear attention (Kimi Linear style)
        hybrid_full_attention_interval=6,  # every 6th layer uses full softmax attention
        linear_attention_use_rope=True,
    )


def small_hybrid() -> AttnResConfig:
    """
    ~150M model with hybrid attention + AttnRes.
    Mixes full softmax attention (every 4th layer) with linear attention (others).
    Best of both worlds: precise retrieval from full attention + long-context
    efficiency from linear attention. Inspired by Kimi Linear's architecture.
    """
    return AttnResConfig(
        vocab_size=50304,
        hidden_size=768,
        intermediate_size=2048,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_key_value_heads=4,
        head_dim=64,
        max_position_embeddings=2048,
        use_attn_res=True,
        attn_res_block_size=4,
        tie_word_embeddings=True,
        attention_type="hybrid",
        hybrid_full_attention_interval=4,  # every 4th layer uses full attention
        linear_attention_use_rope=True,
    )
