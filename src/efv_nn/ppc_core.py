import torch
import torch.nn as nn
import torch.nn.functional as F
import bitsandbytes.nn as bnb_nn
import math


class ComplexKaimingInitializer:
    """
    Initializes interleaved-real parameters [..., 2] with non-negative magnitudes and uniform phases.
    """

    @staticmethod
    def initialize(shape: tuple, gain: float = 1.0) -> torch.Tensor:
        if len(shape) > 1:
            fan_in = shape[-1] if len(shape) == 1 else shape[-2]
        else:
            fan_in = 1

        scale = gain / math.sqrt(fan_in)
        magnitude = torch.abs(torch.randn(shape)) * scale
        phase = torch.rand(shape) * 2 * math.pi
        
        # Interleaved real: [magnitude*cos(phase), magnitude*sin(phase)]
        real = magnitude * torch.cos(phase)
        imag = magnitude * torch.sin(phase)
        return torch.stack([real, imag], dim=-1)


# Keep the old name as an alias so existing callers are not broken.
UnitaryInitializer = ComplexKaimingInitializer


class ModReLU(nn.Module):
    """
    Learnable mod-ReLU activation for interleaved-real tensors [..., 2].
    Formula: modReLU(z) = ReLU(|z| + b) * (z / max(|z|, ε))
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: interleaved-real tensor [..., hidden_dim, 2]
        returns: interleaved-real tensor of same shape
        """
        # Magnitude |z| = sqrt(r^2 + i^2)
        magnitude = torch.norm(x, dim=-1)              # [..., D]
        # Stabilise division
        safe_mag = magnitude.clamp(min=1e-8)
        unit_phase = x / safe_mag.unsqueeze(-1)        # [..., D, 2]
        
        activated_mag = torch.relu(magnitude + self.bias)
        return activated_mag.unsqueeze(-1) * unit_phase


def complex_activation(x: torch.Tensor) -> torch.Tensor:
    """
    Stateless mod-ReLU with zero bias for interleaved-real tensors [..., 2].
    """
    magnitude = torch.norm(x, dim=-1)
    safe_mag = magnitude.clamp(min=1e-8)
    unit_phase = x / safe_mag.unsqueeze(-1)
    return torch.relu(magnitude).unsqueeze(-1) * unit_phase


class ExpertChoiceMoEMatcher(nn.Module):
    def __init__(self, hidden_dim, num_experts=16, k_nodes=None):
        super().__init__()
        self.hidden_dim, self.num_experts = hidden_dim, num_experts
        self.k_nodes_default = k_nodes
        # Gate weights: [2*hidden_dim, num_experts] - gate remains real-valued
        self.gate_weights = nn.Parameter(
            torch.randn(hidden_dim * 2, num_experts) / math.sqrt(hidden_dim * 2)
        )

        # Experts stored in interleaved real [num_experts, hidden_dim, hidden_dim, 2]
        # Using half precision for storage, unpacking to float32 for matmul
        init_real = ComplexKaimingInitializer.initialize((num_experts, hidden_dim, hidden_dim))
        self.experts_weight_real = nn.Parameter(init_real.half())

        self.activation = ModReLU(hidden_dim)

    def forward(self, x):
        """
        x: [B_T, D, 2] interleaved real
        """
        B_T, D, _ = x.shape
        k_nodes = self.k_nodes_default if self.k_nodes_default is not None else max(1, B_T // self.num_experts)

        # Gate uses full complex signal [real || imag]
        x_gate_input = x.reshape(B_T, D * 2)  # [B_T, 2D]
        scores = torch.matmul(x_gate_input, self.gate_weights)  # [B_T, num_experts]
        topk_scores, topk_indices = torch.topk(scores, k_nodes, dim=0)  # [k_nodes, num_experts]

        output = torch.zeros_like(x)
        counts = torch.zeros(B_T, 1, 1, device=x.device)

        for i in range(self.num_experts):
            idx = topk_indices[:, i]
            # Manual Complex Matmul: (x_r + i*x_i)(w_r + i*w_i) = (x_r*w_r - x_i*w_i) + i*(x_r*w_i + x_i*w_r)
            x_subset = x[idx] # [k_nodes, D, 2]
            w_expert = self.experts_weight_real[i].float() # [D, D, 2]
            
            x_r, x_i = x_subset[..., 0], x_subset[..., 1]
            w_r, w_i = w_expert[..., 0], w_expert[..., 1]
            
            y_r = torch.matmul(x_r, w_r) - torch.matmul(x_i, w_i)
            y_i = torch.matmul(x_r, w_i) + torch.matmul(x_i, w_r)
            y_expert = torch.stack([y_r, y_i], dim=-1) # [k_nodes, D, 2]

            output.index_add_(0, idx, y_expert * topk_scores[:, i:i+1].unsqueeze(-1))
            counts.index_add_(0, idx, torch.ones(k_nodes, 1, 1, device=x.device))

        return self.activation(output / counts.clamp(min=1)), topk_indices, topk_scores, counts

    def transpose_forward(self, residual, topk_indices, topk_scores, counts):
        """
        Jacobian-Hermitian pass using manual complex math.
        Computes J^H r = Σ_i s_i * W_i^H * r[idx_i]
        """
        B_T, D, _ = residual.shape
        out = torch.zeros_like(residual)

        for i in range(self.num_experts):
            idx = topk_indices[:, i]
            w_expert = self.experts_weight_real[i].float() # [D, D, 2]
            y_subset = residual[idx] # [k_nodes, D, 2]
            
            # W^H = [W_r^T, -W_i^T].
            # (y_r + i*y_i)(w_r^T - i*w_i^T) = (y_r*w_r^T + y_i*w_i^T) + i*(y_i*w_r^T - y_r*w_i^T)
            y_r, y_i = y_subset[..., 0], y_subset[..., 1]
            w_r_t = w_expert[..., 0].transpose(-2, -1)
            w_i_t = w_expert[..., 1].transpose(-2, -1)
            
            grad_r = torch.matmul(y_r, w_r_t) + torch.matmul(y_i, w_i_t)
            grad_i = torch.matmul(y_i, w_r_t) - torch.matmul(y_r, w_i_t)
            expert_grad = torch.stack([grad_r, grad_i], dim=-1) # [k_nodes, D, 2]
            
            out.index_add_(0, idx, expert_grad * topk_scores[:, i:i+1].unsqueeze(-1))

        return out / counts.clamp(min=1)

