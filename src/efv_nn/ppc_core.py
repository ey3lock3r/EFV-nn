import torch
import torch.nn as nn
import torch.nn.functional as F
import bitsandbytes.nn as bnb_nn
import math


class ComplexKaimingInitializer:
    """
    Initializes complex parameters with non-negative magnitudes and uniform phases.

    Produces tensors where:
      - |W| ~ HalfNormal(scale = gain / sqrt(fan_in))  → always non-negative
      - angle(W) ~ Uniform[0, 2π)                      → uniform phase

    This preserves signal variance across layers in the complex domain.

    Note: A true *unitary* matrix requires W^H W = I (norm-preserving).
    For that, use `torch.linalg.qr` on a random complex matrix. This initializer
    provides variance control but does not enforce unitarity.
    """

    @staticmethod
    def initialize(shape: tuple, gain: float = 1.0) -> torch.Tensor:
        if len(shape) > 1:
            fan_in = shape[-2]
        elif len(shape) == 1:
            fan_in = shape[0]
        else:
            fan_in = 1

        scale = gain / math.sqrt(fan_in)
        # torch.abs(randn) → HalfNormal → always non-negative magnitudes
        magnitude = torch.abs(torch.randn(shape)) * scale
        phase = torch.rand(shape) * 2 * math.pi
        return magnitude * torch.exp(1j * phase)


# Keep the old name as an alias so existing callers are not broken.
UnitaryInitializer = ComplexKaimingInitializer


class ModReLU(nn.Module):
    """
    Learnable mod-ReLU activation for complex-valued tensors.

    Based on Arjovsky et al. (2016): "Unitary Evolution Recurrent Neural Networks".
    Formula: modReLU(z) = ReLU(|z| + b) * (z / max(|z|, ε))

    A learnable bias `b` per hidden dimension controls the activation threshold:
    - b > 0: all magnitudes are activated (nearly linear)
    - b < 0: small-magnitude components are zeroed out (sparse activation)
    - b = 0: threshold at zero (equivalent to magnitude ReLU)
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: complex tensor [..., hidden_dim]
        returns: complex tensor of same shape
        """
        magnitude = x.abs()                            # [..., D], real ≥ 0
        # Stabilise division by clamping the denominator (avoids 0/0 in backward)
        safe_mag = magnitude.clamp(min=1e-8)
        unit_phase = x / safe_mag                      # unit complex vector
        activated_mag = torch.relu(magnitude + self.bias)
        return activated_mag * unit_phase


def complex_activation(x: torch.Tensor) -> torch.Tensor:
    """
    Stateless mod-ReLU with zero bias — used where a Module is not appropriate.

    For learnable thresholding, use `ModReLU(hidden_dim)` instead.
    """
    magnitude = x.abs()
    safe_mag = magnitude.clamp(min=1e-8)   # stabilise angle backward
    unit_phase = x / safe_mag
    return torch.relu(magnitude) * unit_phase


class ExpertChoiceMoEMatcher(nn.Module):
    def __init__(self, hidden_dim, num_experts=16, k_nodes=None):
        super().__init__()
        self.hidden_dim, self.num_experts = hidden_dim, num_experts
        # k_nodes=None → compute dynamically as B_T // num_experts each forward call
        self.k_nodes_default = k_nodes
        # Issue 6 Fix: gate uses full complex info [real || imag] → hidden_dim*2 inputs
        self.gate_weights = nn.Parameter(
            torch.randn(hidden_dim * 2, num_experts) / math.sqrt(hidden_dim * 2)
        )

        init_complex = ComplexKaimingInitializer.initialize((num_experts, hidden_dim, hidden_dim))
        # OPTIMIZATION: FP16 View Trick — store in FP16, unpack JIT to FP32 for matmul.
        self.experts_weight_real = nn.Parameter(torch.view_as_real(init_complex).half())

        self.activation = ModReLU(hidden_dim)

    def forward(self, x):
        B_T, D = x.shape
        # Issue 7 Fix: dynamic k_nodes scales with batch size
        k_nodes = self.k_nodes_default if self.k_nodes_default is not None else max(1, B_T // self.num_experts)

        # Issue 6 Fix: use full complex signal [real || imag] for routing
        x_gate_input = torch.cat([x.real, x.imag], dim=-1)  # [B_T, 2D]
        scores = torch.matmul(x_gate_input, self.gate_weights)  # [B_T, num_experts]
        topk_scores, topk_indices = torch.topk(scores, k_nodes, dim=0)  # [k_nodes, num_experts]

        output = torch.zeros_like(x)
        counts = torch.zeros(B_T, 1, device=x.device)

        for i in range(self.num_experts):
            idx = topk_indices[:, i]
            # Just-In-Time FP32 unpack for stable matmul
            w_expert = torch.view_as_complex(self.experts_weight_real[i].float())
            y_expert = torch.matmul(x[idx], w_expert)

            output.index_add_(0, idx, y_expert * topk_scores[:, i:i+1])
            counts.index_add_(0, idx, torch.ones(k_nodes, 1, device=x.device))

        # Issue 4 Fix: return topk_scores so transpose_forward can weight correctly
        return self.activation(output / counts.clamp(min=1)), topk_indices, topk_scores, counts

    def transpose_forward(self, residual, topk_indices, topk_scores, counts):
        """Jacobian-Hermitian pass: computes J^H r = Σ_i s_i * W_i^H * r[idx_i].
        Must include topk_scores to match the weighted forward pass Jacobian."""
        B_T, D = residual.shape
        out = torch.zeros_like(residual)

        for i in range(self.num_experts):
            idx = topk_indices[:, i]
            w_expert = torch.view_as_complex(self.experts_weight_real[i].float())

            # Issue 4 Fix: weight by gate scores to match forward Jacobian exactly
            expert_grad = torch.matmul(residual[idx], w_expert.adjoint()) * topk_scores[:, i:i+1]
            out.index_add_(0, idx, expert_grad)

        return out / counts.clamp(min=1)

