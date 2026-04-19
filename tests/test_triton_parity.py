"""
Numerical Parity Tests for Triton Kernels.

Verifies that each Triton kernel produces bit-identical results (within FP32 tolerance)
to the equivalent Python/PyTorch implementation.
"""
import pytest
import torch
import math

# Skip all tests if CUDA is not available
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton tests")


@pytest.fixture
def device():
    return "cuda:0"


@pytest.fixture
def dims():
    """Standard PPC-GNN dimensions."""
    return {"B": 2, "T": 256, "D": 1024}


# ============================================================
# Kernel 1: Phase Rotation Parity
# ============================================================
class TestPhaseRotationParity:
    def _python_phase_rotation(self, x_states, cos_p, sin_p):
        """Python reference implementation."""
        B, T, D, _ = x_states.shape
        x_target = torch.zeros_like(x_states)
        x_prev = x_states[:, :-1]
        prev_r, prev_i = x_prev[..., 0], x_prev[..., 1]
        rot_r = prev_r * cos_p - prev_i * sin_p
        rot_i = prev_r * sin_p + prev_i * cos_p
        x_target[:, 1:] = torch.stack([rot_r, rot_i], dim=-1)
        x_target[:, 0] = x_states[:, 0]
        return x_target

    def test_parity(self, device, dims):
        from efv_nn.triton_kernels import fused_phase_rotation

        B, T, D = dims["B"], dims["T"], dims["D"]
        x_states = torch.randn(B, T, D, 2, device=device, dtype=torch.float32)
        phase = torch.rand(D, device=device) * 2 * math.pi
        cos_p = torch.cos(phase)
        sin_p = torch.sin(phase)

        ref = self._python_phase_rotation(x_states, cos_p, sin_p)
        tri = fused_phase_rotation(x_states, cos_p, sin_p)

        assert torch.allclose(ref, tri, atol=1e-5, rtol=1e-5), \
            f"Max diff: {(ref - tri).abs().max().item()}"


# ============================================================
# Kernel 2: OCNS Delay Parity
# ============================================================
class TestOCNSDelayParity:
    def _python_ocns_delay(self, x_states, delay_gains, prime_delays):
        """Python reference (slicing version)."""
        x_eff = x_states.clone()
        for idx, tau in enumerate(prime_delays):
            gr, gi = delay_gains[idx, ..., 0], delay_gains[idx, ..., 1]
            dr = x_states[:, :-tau, ..., 0]
            di = x_states[:, :-tau, ..., 1]
            x_eff[:, tau:, ..., 0] += (dr * gr - di * gi)
            x_eff[:, tau:, ..., 1] += (dr * gi + di * gr)
        return x_eff

    def test_parity(self, device, dims):
        from efv_nn.triton_kernels import fused_ocns_delay

        B, T, D = dims["B"], dims["T"], dims["D"]
        prime_delays = [1, 2, 3, 5]
        x_states = torch.randn(B, T, D, 2, device=device, dtype=torch.float32)
        delay_gains = torch.randn(len(prime_delays), D, 2, device=device, dtype=torch.float32) * 0.01

        ref = self._python_ocns_delay(x_states, delay_gains, prime_delays)
        tri = fused_ocns_delay(x_states, delay_gains, prime_delays)

        assert torch.allclose(ref, tri, atol=1e-5, rtol=1e-5), \
            f"Max diff: {(ref - tri).abs().max().item()}"

    def test_parity_7_delays(self, device, dims):
        """Test with extended Fibonacci delays."""
        from efv_nn.triton_kernels import fused_ocns_delay

        B, T, D = dims["B"], dims["T"], dims["D"]
        prime_delays = [1, 2, 3, 5, 8, 13, 21]
        x_states = torch.randn(B, T, D, 2, device=device, dtype=torch.float32)
        delay_gains = torch.randn(len(prime_delays), D, 2, device=device, dtype=torch.float32) * 0.01

        ref = self._python_ocns_delay(x_states, delay_gains, prime_delays)
        tri = fused_ocns_delay(x_states, delay_gains, prime_delays)

        assert torch.allclose(ref, tri, atol=1e-5, rtol=1e-5), \
            f"Max diff: {(ref - tri).abs().max().item()}"


# ============================================================
# Kernel 3: State Update Parity
# ============================================================
class TestStateUpdateParity:
    def _python_state_update(self, x_states, step, lr):
        """Python reference."""
        x = x_states.clone()
        x.add_(torch.clamp(step, -10.0, 10.0), alpha=lr)
        return x

    def test_parity(self, device, dims):
        from efv_nn.triton_kernels import fused_state_update

        B, T, D = dims["B"], dims["T"], dims["D"]
        x_states = torch.randn(B, T, D, 2, device=device, dtype=torch.float32)
        step = torch.randn(B, T, D, 2, device=device, dtype=torch.float32) * 20  # some values > 10
        lr = 0.35

        ref = self._python_state_update(x_states, step, lr)
        # fused_state_update is in-place
        fused_state_update(x_states, step, lr)

        assert torch.allclose(ref, x_states, atol=1e-5, rtol=1e-5), \
            f"Max diff: {(ref - x_states).abs().max().item()}"


# ============================================================
# Kernel 4: Normalize + Activate Parity
# ============================================================
class TestNormalizeActivateParity:
    def _python_normalize_activate(self, output, counts, bias):
        """Python reference (ModReLU)."""
        normed = output / counts.clamp(min=1)
        mag = torch.norm(normed, dim=-1)
        safe_mag = mag.clamp(min=1e-8)
        unit_phase = normed / safe_mag.unsqueeze(-1)
        activated_mag = torch.relu(mag + bias)
        return activated_mag.unsqueeze(-1) * unit_phase

    def test_parity(self, device, dims):
        from efv_nn.triton_kernels import fused_normalize_activate

        B, T, D = dims["B"], dims["T"], dims["D"]
        B_T = B * T
        output = torch.randn(B_T, D, 2, device=device, dtype=torch.float32)
        counts = torch.randint(1, 5, (B_T, 1, 1), device=device, dtype=torch.float32)
        bias = torch.randn(D, device=device, dtype=torch.float32) * 0.1

        ref = self._python_normalize_activate(output, counts, bias)
        tri = fused_normalize_activate(output, counts, bias)

        assert torch.allclose(ref, tri, atol=1e-4, rtol=1e-4), \
            f"Max diff: {(ref - tri).abs().max().item()}"
