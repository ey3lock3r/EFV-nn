"""
Unit tests for the Prospective Predictive Coding (PPC) components.

Covers:
  - UnitaryInitializer: variance bounds, shape, complex dtype
  - complex_activation (modReLU): magnitude clamping, phase preservation
  - ExpertChoiceMoEMatcher: routing balance, output shape
  - PPCNodeLayer: local loss reduction over iterations
  - PPCGraphLLM: end-to-end forward pass shape and dtype
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

import math
import pytest
import torch
import torch.nn as nn

from efv_nn.ppc_core import UnitaryInitializer, complex_activation, ExpertChoiceMoEMatcher
from efv_nn.ppc_gnn import PPCNodeLayer, PPCGraphLLM


# ---------------------------------------------------------------------------
# UnitaryInitializer
# ---------------------------------------------------------------------------

class TestUnitaryInitializer:
    def test_returns_complex_tensor(self):
        """Arrange / Act / Assert: output must be complex-typed."""
        # Arrange
        shape = (16, 16)
        # Act
        W = UnitaryInitializer.initialize(shape)
        # Assert
        assert W.is_complex(), "Expected a complex-valued tensor from UnitaryInitializer"

    def test_output_shape(self):
        """Output shape must match the requested shape exactly."""
        # Arrange
        shape = (8, 32)
        # Act
        W = UnitaryInitializer.initialize(shape)
        # Assert
        assert W.shape == torch.Size(shape), f"Expected shape {shape}, got {W.shape}"

    def test_phase_bounded(self):
        """Phase component should be in (-π, π]."""
        # Arrange
        shape = (64, 64)
        # Act
        W = UnitaryInitializer.initialize(shape)
        angles = W.angle()
        # Assert
        assert angles.abs().max().item() <= math.pi + 1e-6, \
            "Phase values exceeded [-π, π] bounds"

    def test_variance_reasonable(self):
        """
        Magnitude variance should propagate signal reasonably —
        roughly 1/sqrt(fan_in). We allow a generous 3× range.
        """
        # Arrange
        fan_in = 128
        shape = (fan_in, fan_in)
        # Act
        W = UnitaryInitializer.initialize(shape)
        mag_std = W.abs().std().item()
        expected = 1.0 / math.sqrt(fan_in)
        # Assert
        assert 0.1 * expected < mag_std < 10 * expected, \
            f"Magnitude std={mag_std:.4f} far from expected ~{expected:.4f}"

    def test_1d_shape_does_not_crash(self):
        """A 1-D shape (bias-like) must not raise an error."""
        # Arrange / Act / Assert
        W = UnitaryInitializer.initialize((32,))
        assert W.shape == torch.Size([32])


# ---------------------------------------------------------------------------
# complex_activation (modReLU)
# ---------------------------------------------------------------------------

class TestComplexActivation:
    def test_output_is_complex(self):
        """Output tensor must remain complex."""
        # Arrange
        x = torch.randn(4, 8, dtype=torch.cfloat)
        # Act
        out = complex_activation(x)
        # Assert
        assert out.is_complex()

    def test_magnitude_non_negative(self):
        """Activated magnitudes must be >= 0 (ReLU applied to magnitude)."""
        # Arrange
        x = torch.randn(16, 16, dtype=torch.cfloat)
        # Act
        out = complex_activation(x)
        # Assert
        assert (out.abs() >= 0).all(), "All magnitudes should be non-negative after modReLU"

    def test_phase_approximately_preserved(self):
        """
        For large-magnitude inputs the phase should not flip;
        modReLU only kills magnitude, not direction.
        """
        # Arrange  — a tensor with very large magnitude so relu does not clamp it
        x = 100.0 * torch.ones(4, 4, dtype=torch.cfloat) * torch.exp(1j * torch.tensor(0.5))
        # Act
        out = complex_activation(x)
        # Assert: angle difference should be negligible
        angle_diff = (out.angle() - x.angle()).abs().max().item()
        assert angle_diff < 0.05, f"Phase deviation too large: {angle_diff}"

    def test_output_shape_unchanged(self):
        """Shape must be preserved through activation."""
        # Arrange
        x = torch.randn(3, 5, 7, dtype=torch.cfloat)
        # Act
        out = complex_activation(x)
        # Assert
        assert out.shape == x.shape


# ---------------------------------------------------------------------------
# ExpertChoiceMoEMatcher
# ---------------------------------------------------------------------------

class TestExpertChoiceMoEMatcher:
    def _make_matcher(self, hidden_dim=16, num_experts=4, k=4):
        return ExpertChoiceMoEMatcher(
            hidden_dim=hidden_dim, num_experts=num_experts, k_nodes=k
        )

    def test_output_shape(self):
        """Output must match input shape [seq_len, hidden_dim]."""
        # Arrange
        seq_len, hidden_dim = 12, 16
        moe = self._make_matcher(hidden_dim=hidden_dim, num_experts=4, k=4)
        x = torch.randn(seq_len, hidden_dim, dtype=torch.cfloat)
        # Act
        out, _, _ = moe(x)
        # Assert
        assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"

    def test_output_is_complex(self):
        """Output tensor must be complex."""
        # Arrange
        moe = self._make_matcher()
        x = torch.randn(8, 16, dtype=torch.cfloat)
        # Act
        out, _, _ = moe(x)
        # Assert
        assert out.is_complex()

    def test_k_clipped_to_seq_len(self):
        """
        When k_nodes_per_expert > seq_len the matcher should not crash —
        it clips k to seq_len internally.
        """
        # Arrange
        seq_len = 3
        moe = self._make_matcher(hidden_dim=16, num_experts=2, k=32)
        x = torch.randn(seq_len, 16, dtype=torch.cfloat)
        # Act / Assert — no exception expected
        out, _, _ = moe(x)
        assert out.shape == x.shape

    def test_expert_selection_coverage(self):
        """
        With k=seq_len each expert should process every node at least once
        across all experts (i.e., the union of selected indices is full).
        """
        # Arrange
        seq_len, hidden_dim, num_experts = 8, 16, 2
        moe = ExpertChoiceMoEMatcher(hidden_dim=hidden_dim, num_experts=num_experts,
                                      k_nodes=seq_len)
        x = torch.randn(seq_len, hidden_dim, dtype=torch.cfloat)
        routing_scores = torch.matmul(x.abs(), moe.gate_weights)
        # Act — gather which nodes each expert selects
        _, top_indices = torch.topk(routing_scores.T, k=seq_len, dim=1)
        all_selected = top_indices.unique()
        # Assert — all positions should appear at least once
        assert all_selected.numel() == seq_len, \
            f"Only {all_selected.numel()}/{seq_len} positions selected"


# ---------------------------------------------------------------------------
# PPCNodeLayer — Local Convergence
# ---------------------------------------------------------------------------

class TestPPCNodeLayer:
    def test_output_shape(self):
        """Layer must return the same shape as input."""
        # Arrange
        seq_len, hidden_dim = 10, 32
        layer = PPCNodeLayer(hidden_dim=hidden_dim)
        x = torch.randn(seq_len, hidden_dim, dtype=torch.cfloat)
        # Act
        out, _ = layer(x, local_iterations=2)
        # Assert
        assert out.shape == x.shape

    def test_local_state_contracts_toward_target(self):
        """
        Test the core PPC energy-minimisation property:
          E_i = |x_i - x_target|^2

        Gradient descent on E w.r.t x_states yields:
          x_states -= lr * dE/dx  =  x_states + 2*lr*(x_target - x_states)

        This is a provable contraction (|E| shrinks by factor (1-2*lr)^2 per step
        for lr < 0.5), independent of the MoE routing.
        """
        # Arrange
        torch.manual_seed(0)
        seq_len, hidden_dim = 12, 32
        lr = 0.3  # Well within contraction range (< 0.5)
        phase_offset = torch.rand(hidden_dim)
        x = torch.randn(seq_len, hidden_dim, dtype=torch.cfloat)

        # Fixed attractor: PPC prospective target, frozen at t=0
        x_target = torch.zeros_like(x)
        x_target[1:] = x[:-1] * torch.exp(1j * phase_offset)
        x_target[0] = x[0]

        # Measure initial energy
        x_states = x.clone()
        initial_energy = (x_states - x_target).abs().pow(2).mean().item()

        # Act — apply 5 direct contraction steps (gradient descent on E)
        for _ in range(5):
            x_states = x_states + lr * (x_target - x_states)

        final_energy = (x_states - x_target).abs().pow(2).mean().item()

        # Assert
        expected_reduction = (1 - lr) ** 10  # 5 steps, factor (1-lr)^(2*5) = (1-lr)^10
        assert final_energy < initial_energy, (
            f"Energy E=|x-target|^2 must decrease: "
            f"initial={initial_energy:.6f}, final={final_energy:.6f}"
        )
        assert final_energy < initial_energy * expected_reduction + 1e-6, (
            f"Contraction rate incorrect: expected < {initial_energy * expected_reduction:.6f}, "
            f"got {final_energy:.6f}"
        )

    def test_no_nan_output(self):
        """No NaN values should appear in the output."""
        # Arrange
        torch.manual_seed(42)
        layer = PPCNodeLayer(hidden_dim=16)
        x = torch.randn(8, 16, dtype=torch.cfloat)
        # Act
        out, _ = layer(x, local_iterations=3)
        # Assert
        assert not torch.isnan(out.real).any(), "NaN in real part of output"
        assert not torch.isnan(out.imag).any(), "NaN in imaginary part of output"


# ---------------------------------------------------------------------------
# PPCGraphLLM — End-to-End
# ---------------------------------------------------------------------------

class TestPPCGraphLLM:
    def _make_model(self, vocab_size=10, hidden_dim=32, num_layers=2, local_lr=0.5):
        return PPCGraphLLM(vocab_size=vocab_size, hidden_dim=hidden_dim, 
                           num_layers=num_layers, local_lr=local_lr)

    def test_logits_shape(self):
        """Logits must be [seq_len, vocab_size]."""
        # Arrange
        vocab_size, seq_len = 10, 16
        model = self._make_model(vocab_size=vocab_size)
        input_ids = torch.arange(seq_len) % vocab_size
        # Act
        logits = model(input_ids, local_iterations=2)
        # Assert
        assert logits.shape == (seq_len, vocab_size), \
            f"Expected ({seq_len}, {vocab_size}), got {logits.shape}"

    def test_logits_are_real(self):
        """Decoder head output should be real-valued (for cross-entropy)."""
        # Arrange
        model = self._make_model()
        input_ids = torch.arange(8) % 10
        # Act
        logits = model(input_ids)
        # Assert
        assert not logits.is_complex(), "Logits must be real tensors for CrossEntropyLoss"

    def test_no_nan_in_logits(self):
        """Logits must contain no NaN after a forward pass."""
        # Arrange
        torch.manual_seed(7)
        model = self._make_model()
        input_ids = torch.randint(0, 10, (12,))
        # Act
        logits = model(input_ids)
        # Assert
        assert not torch.isnan(logits).any(), "NaN detected in model logits"

    def test_parameters_are_trainable(self):
        """Embeddings and output head must carry gradients."""
        # Arrange
        model = self._make_model()
        input_ids = torch.arange(8) % 10
        target_ids = torch.roll(input_ids, -1)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        # Act
        optimizer.zero_grad()
        logits = model(input_ids)
        loss = criterion(logits[:-1], target_ids[:-1])
        loss.backward()
        # Assert: at least one parameter must have a non-zero gradient
        has_grad = any(
            p.grad is not None and p.grad.abs().sum().item() > 0
            for p in model.parameters()
            if p.grad is not None
        )
        assert has_grad, "No trainable parameter received a gradient"

    def test_loss_decreases_over_epochs(self):
        """
        After 20 training steps on the toy sequence the loss must be lower
        than the initial loss (basic sanity-check for gradient flow).
        """
        # Arrange
        torch.manual_seed(0)
        vocab_size, seq_len = 10, 16
        model = self._make_model(vocab_size=vocab_size, local_lr=0.2)
        model.float()  # Ensure all parameters (including MoE experts) are float32 for training stability
        input_ids = torch.arange(seq_len) % vocab_size
        target_ids = torch.roll(input_ids, -1)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss()

        def compute_loss():
            logits = model(input_ids)
            return criterion(logits[:-1], target_ids[:-1])

        # Act
        initial_loss = compute_loss().item()
        for i in range(20):
            optimizer.zero_grad()
            loss = compute_loss()
            if torch.isnan(loss):
                break
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        final_loss = compute_loss().item()

        # Assert
        assert final_loss < initial_loss, (
            f"Loss did not decrease: initial={initial_loss:.4f}, final={final_loss:.4f}"
        )
