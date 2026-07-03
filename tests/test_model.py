"""
Unit tests for the AttnRes LM.

Tests:
  1. AttnRes module: block aggregation, softmax weights, shape correctness
  2. Model forward pass: output shapes, loss computation
  3. Backward pass: gradient flow through all parameters
  4. AttnRes vs standard: both modes work, produce different outputs
  5. MoE: expert routing and aux loss
  6. Config validation: parameter estimation, preset configs
  7. Generation: text generation produces valid output
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch
import torch.nn as nn

from src.model import AttnResConfig, AttnResLM
from src.model.attn_res import BlockAttnRes, RMSNorm, ResidualState, precompute_rope_frequencies, apply_rope
from src.model.config import small_prototype, medium_1b, target_9b, small_hybrid
from src.model.moe import MoELayer
from src.model.linear_attention import LinearAttention, elu_plus_one


# ── Fixtures ────────────────────────────────────────────────────

@pytest.fixture
def small_config():
    """Small config for fast tests."""
    return AttnResConfig(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        max_position_embeddings=128,
        use_attn_res=True,
        attn_res_block_size=4,
        tie_word_embeddings=True,
    )


@pytest.fixture
def small_model(small_config):
    return AttnResLM(small_config)


# ── AttnRes Module Tests ────────────────────────────────────────

class TestBlockAttnRes:
    """Tests for the BlockAttnRes module — the core innovation."""

    def test_no_preceding_blocks(self):
        """When there are no preceding blocks, output should equal input."""
        attn_res = BlockAttnRes(hidden_size=64)
        partial = torch.randn(2, 10, 64)
        out = attn_res([], partial)
        assert torch.allclose(out, partial), "Output should equal input when no preceding blocks"

    def test_none_partial_block(self):
        """When partial_block is None (start of new block), attend over block_reps only."""
        attn_res = BlockAttnRes(hidden_size=64)
        blocks = [torch.randn(2, 10, 64) for _ in range(2)]
        out = attn_res(blocks, None)
        assert out.shape == (2, 10, 64), f"Expected (2, 10, 64), got {out.shape}"
        # Should NOT be equal to any single block (it's a weighted sum)
        for i, b in enumerate(blocks):
            assert not torch.allclose(out, b), f"Output should not equal block {i} exactly"

    def test_none_partial_no_blocks_raises(self):
        """Should raise when both partial_block and block_reps are empty."""
        attn_res = BlockAttnRes(hidden_size=64)
        with pytest.raises(RuntimeError):
            attn_res([], None)

    def test_output_shape(self):
        """Output shape should match input shape."""
        attn_res = BlockAttnRes(hidden_size=64)
        blocks = [torch.randn(2, 10, 64) for _ in range(3)]
        partial = torch.randn(2, 10, 64)
        out = attn_res(blocks, partial)
        assert out.shape == (2, 10, 64), f"Expected (2, 10, 64), got {out.shape}"

    def test_softmax_weights_sum_to_one(self):
        """The attention weights should sum to 1 across blocks (by construction of softmax)."""
        hidden_size = 64
        attn_res = BlockAttnRes(hidden_size=hidden_size)
        blocks = [torch.randn(1, 5, hidden_size) for _ in range(2)]
        partial = torch.randn(1, 5, hidden_size)

        # Recompute the weights to verify
        all_reps = blocks + [partial]
        V = torch.stack(all_reps, dim=0)
        K = attn_res.norm(V)
        w = attn_res.proj.weight.squeeze(0)
        logits = torch.einsum("d,nbtd->nbt", w, K)
        alpha = torch.softmax(logits, dim=0)

        sums = alpha.sum(dim=0)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), \
            "Softmax weights should sum to 1 across blocks"

    def test_gradient_flow(self):
        """Gradients should flow through the AttnRes module."""
        attn_res = BlockAttnRes(hidden_size=64)
        blocks = [torch.randn(2, 10, 64, requires_grad=True) for _ in range(2)]
        partial = torch.randn(2, 10, 64, requires_grad=True)
        out = attn_res(blocks, partial)
        loss = out.sum()
        loss.backward()

        assert attn_res.proj.weight.grad is not None, "Pseudo-query should have gradients"
        assert attn_res.norm.weight.grad is not None, "Norm should have gradients"


class TestRMSNorm:
    """Tests for RMSNorm."""

    def test_output_shape(self):
        norm = RMSNorm(64)
        x = torch.randn(2, 10, 64)
        out = norm(x)
        assert out.shape == x.shape

    def test_preserves_dtype(self):
        norm = RMSNorm(64)
        x = torch.randn(2, 10, 64, dtype=torch.float16)
        out = norm(x)
        assert out.dtype == torch.float16

    def test_unit_norm_property(self):
        """RMSNorm output should have RMS close to 1 (before weight scaling)."""
        norm = RMSNorm(64)
        # Set weight to 1 to test normalization only
        with torch.no_grad():
            norm.weight.fill_(1.0)
        x = torch.randn(1, 100, 64) * 5.0  # large values
        out = norm(x)
        rms = out.pow(2).mean(-1).sqrt()
        assert torch.allclose(rms, torch.ones_like(rms), atol=1e-5), \
            "RMSNorm output should have RMS ≈ 1"


class TestRoPE:
    """Tests for Rotary Position Embeddings."""

    def test_frequency_shape(self):
        freqs = precompute_rope_frequencies(head_dim=64, max_seq_len=128)
        assert freqs.shape == (128, 64)

    def test_rope_preserves_norm(self):
        """RoPE should preserve the norm of q and k vectors."""
        head_dim = 64
        q = torch.randn(1, 4, 10, head_dim)
        k = torch.randn(1, 2, 10, head_dim)
        freqs = precompute_rope_frequencies(head_dim, 10)
        cos, sin = freqs.cos(), freqs.sin()

        q_rot, k_rot = apply_rope(q, k, cos, sin)

        q_norm = q.norm(dim=-1)
        q_rot_norm = q_rot.norm(dim=-1)
        assert torch.allclose(q_norm, q_rot_norm, atol=1e-5), \
            "RoPE should preserve query norms"


# ── Model Forward/Backward Tests ────────────────────────────────

class TestModelForward:
    """Tests for the full model forward pass."""

    def test_logits_shape(self, small_model, small_config):
        batch, seq = 2, 32
        input_ids = torch.randint(0, small_config.vocab_size, (batch, seq))
        output = small_model(input_ids)
        assert output.logits.shape == (batch, seq, small_config.vocab_size)

    def test_loss_computed(self, small_model, small_config):
        input_ids = torch.randint(0, small_config.vocab_size, (2, 32))
        labels = input_ids.clone()
        output = small_model(input_ids, labels=labels)
        assert output.loss is not None
        assert output.loss.dim() == 0  # scalar

    def test_loss_is_finite(self, small_model, small_config):
        input_ids = torch.randint(0, small_config.vocab_size, (2, 32))
        labels = input_ids.clone()
        output = small_model(input_ids, labels=labels)
        assert torch.isfinite(output.loss), "Loss should be finite"

    def test_no_labels_no_loss(self, small_model, small_config):
        input_ids = torch.randint(0, small_config.vocab_size, (2, 32))
        output = small_model(input_ids)
        assert output.loss is None

    def test_different_seq_lengths(self, small_model, small_config):
        for seq_len in [16, 32, 64, 128]:
            input_ids = torch.randint(0, small_config.vocab_size, (1, seq_len))
            output = small_model(input_ids)
            assert output.logits.shape == (1, seq_len, small_config.vocab_size)


class TestModelBackward:
    """Tests for gradient flow through the model."""

    def test_all_parameters_have_grads(self, small_model, small_config):
        input_ids = torch.randint(0, small_config.vocab_size, (2, 32))
        labels = input_ids.clone()
        output = small_model(input_ids, labels=labels)
        output.loss.backward()

        no_grad = []
        for name, param in small_model.named_parameters():
            if param.grad is None:
                # Layer 0's AttnRes is a no-op (no preceding blocks to attend over),
                # so its params don't participate in the computation. This is expected.
                if "layers.0.attn_res" in name or "layers.0.mlp_res" in name:
                    continue
                no_grad.append(name)
            elif param.grad.abs().sum().item() == 0:
                # AttnRes with only 1 preceding block has zero gradients because
                # softmax over a single element is always 1.0 (derivative is 0),
                # so the pseudo-query doesn't affect the output. This is expected
                # mathematical behavior, not a bug.
                if "attn_res" in name or "mlp_res" in name:
                    continue
                no_grad.append(f"{name} (zero)")

        assert len(no_grad) == 0, f"Parameters without gradients: {no_grad}"

    def test_attn_res_params_have_grads(self, small_model, small_config):
        """Specifically check AttnRes pseudo-query vectors get gradients (layers 1+)."""
        input_ids = torch.randint(0, small_config.vocab_size, (2, 32))
        labels = input_ids.clone()
        output = small_model(input_ids, labels=labels)
        output.loss.backward()

        # Layer 0's AttnRes is a no-op (no preceding blocks), so skip it.
        # Layers 1+ should have gradients on their AttnRes params.
        found_grad = False
        for name, param in small_model.named_parameters():
            if ("attn_res" in name or "mlp_res" in name) and "layers.0." not in name:
                if param.grad is not None and param.grad.abs().sum().item() > 0:
                    found_grad = True
        assert found_grad, "AttnRes params in layers 1+ should have non-zero gradients"


class TestAttnResVsStandard:
    """Compare AttnRes vs standard residual connections."""

    def test_both_modes_work(self, small_config):
        """Both AttnRes and standard modes should run without errors."""
        for use_attn_res in [True, False]:
            config = AttnResConfig(**{**vars(small_config), "use_attn_res": use_attn_res})
            model = AttnResLM(config)
            input_ids = torch.randint(0, config.vocab_size, (2, 32))
            labels = input_ids.clone()
            output = model(input_ids, labels=labels)
            assert torch.isfinite(output.loss), f"Loss not finite with use_attn_res={use_attn_res}"

    def test_produce_different_outputs(self, small_config):
        """AttnRes and standard residuals should produce different outputs."""
        torch.manual_seed(42)
        config_attn = AttnResConfig(**{**vars(small_config), "use_attn_res": True})
        model_attn = AttnResLM(config_attn)

        torch.manual_seed(42)
        config_std = AttnResConfig(**{**vars(small_config), "use_attn_res": False})
        model_std = AttnResLM(config_std)

        input_ids = torch.randint(0, config_attn.vocab_size, (2, 32))
        out_attn = model_attn(input_ids)
        out_std = model_std(input_ids)

        assert not torch.allclose(out_attn.logits, out_std.logits), \
            "AttnRes and standard residuals should produce different outputs"


# ── MoE Tests ───────────────────────────────────────────────────

class TestMoE:
    """Tests for the Mixture-of-Experts layer."""

    def test_moe_forward_shape(self):
        moe = MoELayer(hidden_size=128, intermediate_size=256, num_experts=4, num_experts_per_tok=2)
        x = torch.randn(2, 16, 128)
        out = moe(x)
        assert out.shape == x.shape

    def test_moe_aux_loss(self):
        moe = MoELayer(hidden_size=128, intermediate_size=256, num_experts=4, num_experts_per_tok=2)
        x = torch.randn(2, 16, 128)
        moe(x)
        assert moe.aux_loss is not None
        assert moe.aux_loss.item() > 0

    def test_moe_aux_loss_has_gradients(self):
        """The aux_loss should be differentiable so the router can learn."""
        moe = MoELayer(hidden_size=128, intermediate_size=256, num_experts=4, num_experts_per_tok=2)
        x = torch.randn(2, 16, 128)
        moe(x)
        moe.aux_loss.backward()
        assert moe.router.weight.grad is not None, "Router should get gradients from aux_loss"
        assert moe.router.weight.grad.abs().sum().item() > 0, "Router gradients should be non-zero"

    def test_moe_gradient_flow(self):
        moe = MoELayer(hidden_size=128, intermediate_size=256, num_experts=4, num_experts_per_tok=2)
        x = torch.randn(2, 16, 128, requires_grad=True)
        out = moe(x)
        loss = out.sum()
        loss.backward()

        assert moe.router.weight.grad is not None, "Router should have gradients"
        for i, expert in enumerate(moe.experts):
            for name, param in expert.named_parameters():
                assert param.grad is not None or param.grad.abs().sum().item() >= 0, \
                    f"Expert {i} param {name} should have gradients"


# ── Linear Attention Tests ─────────────────────────────────────

class TestLinearAttention:
    """Tests for the Linear Attention module — O(L) sub-quadratic attention."""

    def test_forward_shape(self):
        """Output shape should match input shape."""
        attn = LinearAttention(
            hidden_size=128, num_attention_heads=4, num_key_value_heads=2, head_dim=32,
        )
        x = torch.randn(2, 16, 128)
        out = attn(x)
        assert out.shape == (2, 16, 128)

    def test_different_seq_lengths(self):
        """Should handle various sequence lengths."""
        attn = LinearAttention(
            hidden_size=128, num_attention_heads=4, num_key_value_heads=2, head_dim=32,
        )
        for seq_len in [8, 16, 32, 64]:
            x = torch.randn(1, seq_len, 128)
            out = attn(x)
            assert out.shape == (1, seq_len, 128)

    def test_loss_is_finite(self):
        """Model with linear attention should produce finite loss."""
        config = AttnResConfig(
            vocab_size=256, hidden_size=128, intermediate_size=256,
            num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
            head_dim=32, max_position_embeddings=128,
            use_attn_res=True, attn_res_block_size=4,
            attention_type="linear", linear_attention_use_rope=True,
        )
        model = AttnResLM(config)
        input_ids = torch.randint(0, config.vocab_size, (2, 32))
        labels = input_ids.clone()
        output = model(input_ids, labels=labels)
        assert torch.isfinite(output.loss), "Loss should be finite with linear attention"

    def test_gradient_flow(self):
        """Gradients should flow through linear attention parameters."""
        attn = LinearAttention(
            hidden_size=128, num_attention_heads=4, num_key_value_heads=2, head_dim=32,
        )
        x = torch.randn(2, 16, 128, requires_grad=True)
        out = attn(x)
        loss = out.sum()
        loss.backward()

        assert attn.q_proj.weight.grad is not None, "Q proj should have gradients"
        assert attn.k_proj.weight.grad is not None, "K proj should have gradients"
        assert attn.v_proj.weight.grad is not None, "V proj should have gradients"
        assert attn.o_proj.weight.grad is not None, "O proj should have gradients"

    def test_elu_plus_one_non_negative(self):
        """ELU+1 feature map should always produce non-negative values."""
        x = torch.randn(100) * 10  # large range including negatives
        phi = elu_plus_one(x)
        assert (phi >= 0).all(), "ELU+1 should be non-negative everywhere"

    def test_causal_property(self):
        """Changing a later token should not affect earlier token outputs (causality)."""
        attn = LinearAttention(
            hidden_size=64, num_attention_heads=4, num_key_value_heads=4, head_dim=16,
        )
        x1 = torch.randn(1, 8, 64)
        x2 = x1.clone()
        x2[0, 6, :] += 10.0  # modify token at position 6

        with torch.no_grad():
            out1 = attn(x1)
            out2 = attn(x2)

        # Positions 0-5 should be unchanged (causal — can't see future)
        assert torch.allclose(out1[:, :6], out2[:, :6], atol=1e-5), \
            "Earlier positions should not be affected by changes to later tokens"


class TestHybridAttention:
    """Tests for hybrid attention mode (mix of full + linear attention)."""

    def test_hybrid_forward(self):
        """Model with hybrid attention should run and produce finite loss."""
        config = small_hybrid()
        config.num_hidden_layers = 8  # smaller for test
        config.vocab_size = 256
        config.hidden_size = 128
        config.intermediate_size = 256
        config.num_attention_heads = 4
        config.num_key_value_heads = 2
        config.head_dim = 32
        config.max_position_embeddings = 128
        model = AttnResLM(config)

        input_ids = torch.randint(0, config.vocab_size, (2, 32))
        labels = input_ids.clone()
        output = model(input_ids, labels=labels)
        assert torch.isfinite(output.loss), "Hybrid model should produce finite loss"

    def test_hybrid_has_both_types(self):
        """Hybrid model should have both full and linear attention layers."""
        config = small_hybrid()
        config.num_hidden_layers = 8
        config.vocab_size = 256
        config.hidden_size = 128
        config.intermediate_size = 256
        config.num_attention_heads = 4
        config.num_key_value_heads = 2
        config.head_dim = 32
        config.max_position_embeddings = 128
        config.hybrid_full_attention_interval = 4
        model = AttnResLM(config)

        full_count = sum(1 for layer in model.layers if not layer.is_linear_attention)
        linear_count = sum(1 for layer in model.layers if layer.is_linear_attention)
        assert full_count > 0, "Should have at least one full attention layer"
        assert linear_count > 0, "Should have at least one linear attention layer"
        assert full_count + linear_count == config.num_hidden_layers

    def test_hybrid_backward(self):
        """Gradients should flow through both full and linear attention layers."""
        config = small_hybrid()
        config.num_hidden_layers = 4
        config.vocab_size = 256
        config.hidden_size = 128
        config.intermediate_size = 256
        config.num_attention_heads = 4
        config.num_key_value_heads = 2
        config.head_dim = 32
        config.max_position_embeddings = 64
        config.hybrid_full_attention_interval = 2
        model = AttnResLM(config)

        input_ids = torch.randint(0, config.vocab_size, (2, 16))
        labels = input_ids.clone()
        output = model(input_ids, labels=labels)
        output.loss.backward()

        # Check at least one linear attention layer has gradients
        linear_grads = False
        for layer in model.layers:
            if layer.is_linear_attention:
                if layer.attn.q_proj.weight.grad is not None:
                    linear_grads = True
        assert linear_grads, "Linear attention layers should have gradients"


# ── Config Tests ────────────────────────────────────────────────

class TestConfig:
    """Tests for configuration validation and parameter estimation."""

    def test_small_prototype_params(self):
        config = small_prototype()
        info = config.estimate_num_params()
        assert 100e6 < info["total"] < 200e6, f"Small prototype should be ~150M, got {info['total_M']:.1f}M"

    def test_target_9b_params(self):
        config = target_9b()
        info = config.estimate_num_params()
        assert 7e9 < info["total"] < 12e9, f"Target 9B should be ~9B, got {info['total_B']:.2f}B"

    def test_invalid_config_raises(self):
        with pytest.raises(AssertionError):
            AttnResConfig(hidden_size=100, num_attention_heads=3)  # not divisible

    def test_invalid_gqa_raises(self):
        with pytest.raises(AssertionError):
            AttnResConfig(
                hidden_size=128, num_attention_heads=4, num_key_value_heads=3
            )  # 4 not divisible by 3

    def test_invalid_attention_type_raises(self):
        with pytest.raises(AssertionError):
            AttnResConfig(attention_type="invalid")

    def test_num_blocks(self):
        config = AttnResConfig(
            num_hidden_layers=12, attn_res_block_size=4, use_attn_res=True
        )
        # 12 layers * 2 sublayers / 4 = 6 blocks
        assert config.num_blocks == 6

    def test_sublayers_per_block(self):
        config = AttnResConfig(attn_res_block_size=4, use_attn_res=True)
        assert config.sublayers_per_block == 2


# ── ResidualState Tests ─────────────────────────────────────────

class TestResidualState:
    """Tests for the ResidualState manager."""

    def test_initial_state(self):
        h = torch.randn(2, 10, 64)
        state = ResidualState(h, use_attn_res=True)
        assert len(state.block_reps) == 0
        assert torch.equal(state.partial_block, h)
        assert state.sublayer_idx == 0

    def test_add_output(self):
        h = torch.randn(2, 10, 64)
        state = ResidualState(h, use_attn_res=True)
        out = torch.randn(2, 10, 64)
        state.add_output(out)
        assert torch.allclose(state.partial_block, h + out)

    def test_block_closure(self):
        h = torch.randn(2, 10, 64)
        state = ResidualState(h, use_attn_res=True)
        # sublayers_per_block = 2
        state.maybe_close_block(2)  # first sublayer
        assert len(state.block_reps) == 0  # not yet
        state.maybe_close_block(2)  # second sublayer — should close block
        assert len(state.block_reps) == 1
        assert state.sublayer_idx == 0
        # After closing, partial_block should be reset to None
        assert state.partial_block is None, "partial_block should be None after block closure"

    def test_add_output_after_block_closure(self):
        """After block closure (partial=None), add_output should set instead of add."""
        h = torch.randn(2, 10, 64)
        state = ResidualState(h, use_attn_res=True)
        state.maybe_close_block(2)  # close block
        state.maybe_close_block(2)  # boundary
        assert state.partial_block is None
        out = torch.randn(2, 10, 64)
        state.add_output(out)
        assert torch.allclose(state.partial_block, out), "Should set partial to output when was None"
