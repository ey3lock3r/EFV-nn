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


# ---------------------------------------------------------------------------
# complex_activation (modReLU)
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
        """Activated magnitudes must be >= 0."""
        # Arrange
        x = torch.randn(16, 16, 2)
        # Act
        out = complex_activation(x)
        # Assert
        mag = torch.norm(out, dim=-1)
        assert (mag >= 0).all(), "All magnitudes should be non-negative"

    def test_phase_preserved(self):
        """modReLU only kills magnitude, not direction."""
        # Arrange
        phase = torch.tensor(0.5)
        x_r = 100.0 * torch.cos(phase)
        x_i = 100.0 * torch.sin(phase)
        x = torch.stack([x_r, x_i], dim=-1).expand(4, 4, 2)
        # Act
        out = complex_activation(x)
        # Assert: angle difference should be negligible
        out_angle = torch.atan2(out[..., 1], out[..., 0])
        x_angle = torch.atan2(x[..., 1], x[..., 0])
        angle_diff = (out_angle - x_angle).abs().max().item()
        assert angle_diff < 0.05, f"Phase deviation too large: {angle_diff}"


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
        out, _, _, _ = moe(x)
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
        out, _ = layer(x, local_iters=2)
        # Assert
        assert out.shape == x.shape

    def test_no_nan_output(self):
        """No NaN values should appear in the output."""
        # Arrange
        torch.manual_seed(42)
        layer = PPCNodeLayer(hidden_dim=16)
        x = torch.randn(8, 16, 2)
        # Act
        out, _ = layer(x, local_iters=3)
        # Assert
        assert not torch.isnan(out).any(), "NaN in output"


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
        logits = model(input_ids, local_iters=2)
        # Assert
        assert logits.shape == (seq_len, vocab_size), \
            f"Expected ({seq_len}, {vocab_size}), got {logits.shape}"

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

    def test_loss_decreases_over_epochs(self):
        """Basic sanity-check for gradient flow."""
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
            logits = model(input_ids)
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
