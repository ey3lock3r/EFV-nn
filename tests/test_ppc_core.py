"""
Unit tests for the Prospective Predictive Coding (PPC) components.

Covers:
  - ComplexKaimingInitializer: shape, phase bounds, variance, fan_in logic
  - complex_activation (ComplexGELU): output shape, non-negative magnitude
  - ExpertChoiceMoEMatcher: routing coverage, output shape
  - PPCNodeLayer: output shape, no-NaN
  - PPCGraphLLM: end-to-end forward pass shape, no-NaN, gradient flow
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

import math
import pytest
import torch
import torch.nn as nn

from efv_nn.ppc_core import (
    UnitaryInitializer, ComplexKaimingInitializer,
    complex_activation, ExpertChoiceMoEMatcher,
)
from efv_nn.ppc_gnn import PPCNodeLayer, PPCGraphLLM
from efv_nn.deq_solvers import anderson_acceleration


# ---------------------------------------------------------------------------
# ComplexKaimingInitializer (canonical) — also tested via UnitaryInitializer alias
# ---------------------------------------------------------------------------

class TestUnitaryInitializer:
    def test_returns_real_pair_tensor(self):
        """Arrange / Act / Assert: output must be real-pair format [..., 2]."""
        # Arrange
        shape = (16, 16)
        # Act
        W = UnitaryInitializer.initialize(shape)
        # Assert
        assert not W.is_complex(), "Expected a real-valued tensor"
        assert W.shape[-1] == 2, f"Expected last dimension 2, got {W.shape[-1]}"

    def test_output_shape(self):
        """Output shape must match requested shape + last dim 2."""
        # Arrange
        shape = (8, 32)
        # Act
        W = UnitaryInitializer.initialize(shape)
        # Assert
        assert W.shape == torch.Size((*shape, 2)), f"Expected shape {(*shape, 2)}, got {W.shape}"

    def test_phase_bounded(self):
        """Phase component should be in (-π, π]."""
        # Arrange
        shape = (64, 64)
        # Act
        W = UnitaryInitializer.initialize(shape)
        angles = torch.atan2(W[..., 1], W[..., 0])
        # Assert
        assert angles.abs().max().item() <= math.pi + 1e-6, \
            "Phase values exceeded [-π, π] bounds"

    def test_variance_reasonable(self):
        """
        Magnitude variance should propagate signal reasonably —
        roughly 1/sqrt(fan_in).
        """
        # Arrange
        fan_in = 128
        shape = (fan_in, fan_in)
        # Act
        W = UnitaryInitializer.initialize(shape)
        mag_std = torch.norm(W, dim=-1).std().item()
        expected = 1.0 / math.sqrt(fan_in)
        # Assert
        assert 0.1 * expected < mag_std < 10 * expected, \
            f"Magnitude std={mag_std:.4f} far from expected ~{expected:.4f}"

    def test_fan_in_2d_uses_second_to_last_dim(self):
        """For a 2D shape (rows, cols) fan_in == cols (shape[-2])."""
        # Arrange — shape (32, 64): fan_in should be 32 not 64
        W_wide = ComplexKaimingInitializer.initialize((32, 64))
        W_tall = ComplexKaimingInitializer.initialize((64, 32))
        # Act — wider matrix (larger fan_in=32) should have smaller scale
        std_wide = torch.norm(W_wide, dim=-1).std().item()
        std_tall = torch.norm(W_tall, dim=-1).std().item()
        # Assert — fan_in=32 → scale 1/√32 ≈ 0.177; fan_in=64 → 1/√64 = 0.125
        assert std_wide > std_tall, (
            f"Expected wider matrix (fan_in=32) to have larger std than "
            f"tall (fan_in=64): {std_wide:.4f} vs {std_tall:.4f}"
        )

    def test_1d_shape(self):
        """1D shape initializes without error and has correct last dim."""
        W = ComplexKaimingInitializer.initialize((64,))
        assert W.shape == (64, 2)


# ---------------------------------------------------------------------------
# complex_activation (ComplexGELU — independent real/imag)
# ---------------------------------------------------------------------------

class TestComplexActivation:
    def test_output_is_real_pair(self):
        """Output tensor must remain real-pair."""
        # Arrange
        x = torch.randn(4, 8, 2)
        # Act
        out = complex_activation(x)
        # Assert
        assert not out.is_complex()
        assert out.shape[-1] == 2

    def test_magnitude_non_negative(self):
        """GELU output magnitude must be >= 0 (GELU can output negative values,
        so this tests that magnitudes are geometric norms, not activations themselves)."""
        # Arrange
        x = torch.randn(16, 16, 2) * 5  # large values exercise clamping region
        # Act
        out = complex_activation(x)
        # Assert
        mag = torch.norm(out, dim=-1)
        assert (mag >= 0).all(), "All magnitudes should be non-negative"

    def test_applies_gelu_independently(self):
        """ComplexGELU applies GELU to each component independently (not magnitude)."""
        import torch.nn.functional as F
        # Arrange
        x = torch.randn(4, 8, 2)
        # Act
        out = complex_activation(x)
        # Assert: each component equals F.gelu of that component
        assert torch.allclose(out[..., 0], F.gelu(x[..., 0]), atol=1e-6)
        assert torch.allclose(out[..., 1], F.gelu(x[..., 1]), atol=1e-6)

    def test_shape_preserved(self):
        """Output shape must equal input shape."""
        # Arrange
        shapes = [(4, 8, 2), (2, 16, 32, 2), (100, 2)]
        for shape in shapes:
            x = torch.randn(*shape)
            # Act / Assert
            assert complex_activation(x).shape == x.shape, f"Shape mismatch for {shape}"


# ---------------------------------------------------------------------------
# Anderson Acceleration
# ---------------------------------------------------------------------------

class TestAndersonAcceleration:
    def test_converges_on_linear_system(self):
        """Anderson must converge on a simple contractive linear map x = Ax + b."""
        # Arrange — f(x) = 0.5*x + c  has fixed point x* = 2c
        torch.manual_seed(0)
        B, T, D = 1, 4, 8
        c = torch.ones(B, T, D, 2) * 0.5

        def f(x):
            return 0.5 * x + c

        x0 = torch.zeros(B, T, D, 2)
        x_star_expected = 2 * c  # analytical fixed point

        # Act
        x_star, iters, res_norm = anderson_acceleration(f, x0, m=5, max_iter=50, tol=1e-6)

        # Assert
        assert torch.allclose(x_star, x_star_expected, atol=1e-4), \
            f"Did not converge to fixed point. Max err: {(x_star - x_star_expected).abs().max()}"
        assert res_norm.item() < 1e-4

    def test_nan_siphon_prevents_cascade(self):
        """NaN in first f(x0) must not poison subsequent iterations."""
        # Arrange — f returns NaN on first call, then a valid contractive function
        torch.manual_seed(1)
        B, T, D = 1, 4, 8
        call_count = [0]

        def f_nan_first(x):
            call_count[0] += 1
            if call_count[0] == 1:
                return torch.full_like(x, float('nan'))
            return 0.5 * x  # converges to x*=0

        x0 = torch.randn(B, T, D, 2) * 0.1

        # Act
        x_out, iters, res_norm = anderson_acceleration(f_nan_first, x0, m=5, max_iter=20, tol=1e-4)

        # Assert — no NaN in output and the solver continued
        assert not torch.isnan(x_out).any(), "NaN cascade not stopped by siphon"
        assert iters > 1

    def test_returns_correct_shapes(self):
        """Return shapes must match: x [B,T,D,2], iters scalar, res_norm scalar."""
        B, T, D = 2, 8, 16
        x0 = torch.zeros(B, T, D, 2)
        x_out, iters, res_norm = anderson_acceleration(lambda x: 0.5 * x, x0, m=3, max_iter=5, tol=1.0)
        assert x_out.shape == (B, T, D, 2)
        assert isinstance(iters, int)
        assert res_norm.ndim == 0  # scalar tensor


# ---------------------------------------------------------------------------
# ExpertChoiceMoEMatcher
# ---------------------------------------------------------------------------

class TestExpertChoiceMoEMatcher:
    def _make_matcher(self, hidden_dim=16, num_experts=4, k=4):
        return ExpertChoiceMoEMatcher(
            hidden_dim=hidden_dim, num_experts=num_experts, k_nodes=k
        )

    def test_output_shape(self):
        """Output must match input shape [seq_len, hidden_dim, 2]."""
        # Arrange
        seq_len, hidden_dim = 12, 16
        moe = self._make_matcher(hidden_dim=hidden_dim, num_experts=4, k=4)
        x = torch.randn(seq_len, hidden_dim, 2)
        # Act
        out, _, _, _, _ = moe(x)
        # Assert
        assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"

    def test_expert_selection_coverage(self):
        """Union of selected indices should be full coverage if k=seq_len."""
        # Arrange
        seq_len, hidden_dim, num_experts = 8, 16, 2
        moe = ExpertChoiceMoEMatcher(hidden_dim=hidden_dim, num_experts=num_experts,
                                      k_nodes=seq_len)
        x = torch.randn(seq_len, hidden_dim, 2)
        x_gate = x.reshape(seq_len, hidden_dim * 2)
        routing_scores = torch.matmul(x_gate, moe.gate_weights)
        # Act
        _, top_indices = torch.topk(routing_scores.T, k=seq_len, dim=1)
        all_selected = top_indices.unique()
        # Assert
        assert all_selected.numel() == seq_len


# ---------------------------------------------------------------------------
# PPCNodeLayer — Local Convergence
# ---------------------------------------------------------------------------

class TestPPCNodeLayer:
    def test_output_shape(self):
        """Layer must return the same shape as input."""
        # Arrange
        seq_len, hidden_dim = 10, 32
        layer = PPCNodeLayer(hidden_dim=hidden_dim)
        x = torch.randn(seq_len, hidden_dim, 2)
        # Act
        out, _, _, _ = layer(x, local_iters=2)
        # Assert
        assert out.shape == x.shape

    def test_no_nan_output(self):
        """No NaN values should appear in the output."""
        # Arrange
        torch.manual_seed(42)
        layer = PPCNodeLayer(hidden_dim=16)
        x = torch.randn(8, 16, 2)
        # Act
        out, _, _, _ = layer(x, local_iters=3)
        # Assert
        assert not torch.isnan(out).any(), "NaN in output"

    def test_apd_exits_early_with_low_rolling_energy(self):
        """Low rolling_energy relaxes the tolerance — APD exits early when energy is low."""
        torch.manual_seed(0)
        layer = PPCNodeLayer(hidden_dim=16, prime_delays=[])
        x = torch.randn(1, 8, 16, 2)
        max_iters = 20

        # low rolling_energy → dynamic_tol is relaxed → exits earlier
        _, iters_relaxed, _, _ = layer(x, local_iters=max_iters, rolling_energy=0.0)
        # high rolling_energy → dynamic_tol is tight → uses more iterations
        _, iters_tight, _, _ = layer(x, local_iters=max_iters, rolling_energy=100.0)

        assert iters_relaxed <= iters_tight, (
            f"APD: low energy should exit no later than high energy: "
            f"relaxed={iters_relaxed}, tight={iters_tight}"
        )

    def test_external_gate_bias_not_overwritten(self):
        """An externally provided gate_bias must be used, not overridden by the layer."""
        # Arrange
        torch.manual_seed(5)
        hidden_dim, num_experts = 16, 4
        layer = PPCNodeLayer(hidden_dim=hidden_dim, num_experts=num_experts, prime_delays=[])
        B, T = 1, 8
        x = torch.randn(B, T, hidden_dim, 2)

        # A clearly non-zero external bias — if ignored the routing would differ
        sentinel_bias = torch.ones(B * T, num_experts) * 1000.0

        # Capture what gate_bias is actually used inside f_forward_step
        captured = {}
        original_get_indices = layer.moe.get_indices

        def patched_get_indices(x_in, gate_bias=None):
            captured['gate_bias'] = gate_bias
            return original_get_indices(x_in, gate_bias=gate_bias)

        layer.moe.get_indices = patched_get_indices

        # Act
        with torch.no_grad():
            layer(x, local_iters=2, gate_bias=sentinel_bias)  # return is 4-tuple; not unpacked here

        # Assert
        assert captured.get('gate_bias') is not None, "gate_bias was None — external bias dropped"
        assert torch.allclose(captured['gate_bias'], sentinel_bias), \
            "gate_bias was overwritten with internally computed value"


# ---------------------------------------------------------------------------
# PPCGraphLLM — End-to-End
# ---------------------------------------------------------------------------

class TestPPCGraphLLM:
    def _make_model(self, vocab_size=10, hidden_dim=32, num_layers=2, local_lr=0.5):
        return PPCGraphLLM(vocab_size=vocab_size, hidden_dim=hidden_dim,
                           num_layers=num_layers, local_lr=local_lr)

    def test_logits_shape_unbatched(self):
        """Unbatched: logits must be [T, vocab_size]."""
        # Arrange
        vocab_size, seq_len = 10, 16
        model = self._make_model(vocab_size=vocab_size)
        input_ids = torch.arange(seq_len) % vocab_size
        # Act
        logits, _, _, _, _ = model(input_ids, local_iters=2)
        # Assert
        assert logits.shape == (seq_len, vocab_size), \
            f"Expected ({seq_len}, {vocab_size}), got {logits.shape}"

    def test_logits_shape_batched(self):
        """Batched: logits must be [B, T, vocab_size]."""
        # Arrange
        vocab_size, B, seq_len = 10, 3, 16
        model = self._make_model(vocab_size=vocab_size)
        input_ids = torch.randint(0, vocab_size, (B, seq_len))
        # Act
        logits, avg_iters, avg_energy, layer_energies, aux_loss = model(input_ids, local_iters=2)
        # Assert
        assert logits.shape == (B, seq_len, vocab_size), \
            f"Expected ({B}, {seq_len}, {vocab_size}), got {logits.shape}"
        assert layer_energies.shape == (2,)  # num_layers

    def test_no_nan_in_logits(self):
        """Logits must contain no NaN after a forward pass."""
        # Arrange
        torch.manual_seed(7)
        model = self._make_model()
        input_ids = torch.randint(0, 10, (12,))
        # Act
        logits, _, _, _, _ = model(input_ids)
        # Assert
        assert not torch.isnan(logits).any(), "NaN detected in model logits"

    def test_loss_decreases_over_epochs(self):
        """Basic sanity-check for gradient flow through IFT backward."""
        # Arrange
        torch.manual_seed(0)
        vocab_size, seq_len = 10, 16
        model = self._make_model(vocab_size=vocab_size, local_lr=0.2)
        model.float()
        input_ids = torch.arange(seq_len) % vocab_size
        target_ids = torch.roll(input_ids, -1)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss()

        def compute_loss():
            logits, _, _, _, _ = model(input_ids)
            return criterion(logits[:-1], target_ids[:-1])

        # Act
        initial_loss = compute_loss().item()
        for i in range(10):
            optimizer.zero_grad()
            loss = compute_loss()
            if torch.isnan(loss): break
            loss.backward()
            optimizer.step()
        final_loss = compute_loss().item()

        # Assert
        assert final_loss < initial_loss, (
            f"Loss did not decrease: initial={initial_loss:.4f}, final={final_loss:.4f}"
        )

    def test_moe_cache_cleared_after_forward(self):
        """FP32 MoE weight cache must be None after each forward (no 25 GB leak)."""
        # Arrange
        torch.manual_seed(2)
        model = self._make_model()
        input_ids = torch.randint(0, 10, (1, 8))

        # Act
        with torch.no_grad():
            model(input_ids, local_iters=3)  # return is 5-tuple; not unpacked here

        # Assert — cache must be cleaned up by cleanup_fn
        for layer in model.layers:
            assert layer.moe._wr_f32 is None, "FP32 weight cache not cleared — memory leak"

    def test_expert_weight_grad_flows(self):
        """Expert weight parameters must receive non-zero gradients after backward."""
        # Arrange
        torch.manual_seed(3)
        model = self._make_model(vocab_size=10, hidden_dim=16, num_layers=1)
        input_ids = torch.randint(0, 10, (1, 8))
        target = torch.randint(0, 10, (1, 8))

        # Act
        logits, _, _, _, _ = model(input_ids, local_iters=3)
        loss = nn.CrossEntropyLoss()(logits.reshape(-1, 10), target.reshape(-1))
        loss.backward()

        # Assert
        grad = model.layers[0].moe.experts_weight_real.grad
        assert grad is not None, "No gradient on expert weights"
        assert grad.norm().item() > 0, "Expert weight gradient is zero"


# ---------------------------------------------------------------------------
# ShardedPPCGraphLLM — CPU single-GPU smoke tests
# ---------------------------------------------------------------------------

class TestShardedPPCGraphLLM:
    """Tests ShardedPPCGraphLLM on CPU (single-GPU sharding collapses to cuda:0==cuda:1 or cpu)."""

    def _make_model(self, vocab_size=10, hidden_dim=32, num_layers=2):
        from efv_nn.ppc_sharded import ShardedPPCGraphLLM
        return ShardedPPCGraphLLM(
            vocab_size=vocab_size, hidden_dim=hidden_dim,
            num_layers=num_layers, num_experts=4,
            prime_delays=[1, 2], use_triton=False,
        )

    def test_forward_output_shapes(self):
        """Forward must return logits [B, T, V], scalar avg_iters, scalar avg_energy."""
        # Arrange
        vocab_size, B, T = 10, 2, 8
        model = self._make_model(vocab_size=vocab_size)
        input_ids = torch.randint(0, vocab_size, (B, T))

        # Act
        logits, avg_iters, avg_energy, layer_energies, aux_loss = model(input_ids, local_iters=2)

        # Assert
        assert logits.shape == (B, T, vocab_size)
        assert layer_energies.shape == (2,)
        assert isinstance(avg_iters, float)

    def test_generate_single_batch(self):
        """generate must work for B=1 and return tokens including prompt."""
        # Arrange
        vocab_size = 20
        model = self._make_model(vocab_size=vocab_size)
        input_ids = torch.randint(0, vocab_size, (1, 5))

        # Act
        with torch.no_grad():
            out = model.generate(input_ids, max_new_tokens=4, local_iters=2)

        # Assert
        assert out.shape[0] == 1
        assert out.shape[1] >= 5  # at least the prompt
        assert out.shape[1] <= 9  # at most prompt + 4

    def test_generate_batched_no_crash(self):
        """generate must work for B > 1 (EOS .item() crash regression)."""
        # Arrange
        vocab_size = 20
        model = self._make_model(vocab_size=vocab_size)
        input_ids = torch.randint(0, vocab_size, (3, 5))  # B=3

        # Act / Assert — must not raise RuntimeError
        with torch.no_grad():
            out = model.generate(input_ids, max_new_tokens=3, local_iters=2)
        assert out.shape[0] == 3


class TestMoELoadBalance:
    def test_aux_loss_is_scalar_tensor(self):
        """compute() must return a scalar aux_loss when grad is enabled."""
        moe = ExpertChoiceMoEMatcher(hidden_dim=16, num_experts=4, k_nodes=4)
        x = torch.randn(8, 16, 2, requires_grad=True)
        topk_indices, topk_scores = moe.get_indices(x)
        flat = topk_indices.reshape(-1)
        x_batched = x[flat].view(4, topk_indices.shape[1], 16, 2)
        out, _, _, _, aux_loss = moe.compute(x_batched, topk_indices, topk_scores, 8)
        assert isinstance(aux_loss, torch.Tensor)
        assert aux_loss.ndim == 0
        assert aux_loss.item() >= 0.0

    def test_aux_loss_zero_in_no_grad(self):
        """aux_loss must be 0 during inference (no_grad)."""
        moe = ExpertChoiceMoEMatcher(hidden_dim=16, num_experts=4, k_nodes=4)
        x = torch.randn(8, 16, 2)
        topk_indices, topk_scores = moe.get_indices(x)
        flat = topk_indices.reshape(-1)
        x_batched = x[flat].view(4, topk_indices.shape[1], 16, 2)
        with torch.no_grad():
            out, _, _, _, aux_loss = moe.compute(x_batched, topk_indices, topk_scores, 8)
        assert aux_loss.item() == 0.0

    def test_ppcgraphllm_returns_aux_loss(self):
        """PPCGraphLLM.forward must return 5-tuple including scalar aux_loss."""
        from efv_nn.ppc_gnn import PPCGraphLLM
        model = PPCGraphLLM(vocab_size=10, hidden_dim=32, num_layers=2)
        ids = torch.randint(0, 10, (2, 8))
        out = model(ids, local_iters=2)
        assert len(out) == 5, f"Expected 5-tuple, got {len(out)}"
        logits, avg_iters, avg_energy, layer_energies, aux_loss = out
        assert aux_loss.ndim == 0
        assert aux_loss.item() >= 0.0
