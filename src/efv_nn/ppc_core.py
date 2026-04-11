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
        [B_T, D, 2] interleaved real -> [B_T, D, 2]
        Vectorized Expert Choice MoE
        """
        original_dtype = x.dtype
        # Logic: Experts run in autocast-selected precision (f16), but accumulate in f32
        B_T, D, _ = x.shape
        k_nodes = self.k_nodes_default if self.k_nodes_default is not None else max(1, B_T // self.num_experts)

        # 1. Gating
        x_gate_input = x.reshape(B_T, D * 2).to(dtype=self.gate_weights.dtype)
        scores = torch.matmul(x_gate_input, self.gate_weights)  # [B_T, num_experts]
        topk_scores, topk_indices = torch.topk(scores, k_nodes, dim=0)  # [k_node, num_experts]

        # 2. Gather All Tokens for BMM
        # topk_indices: [k, E] -> flat: [E*k]
        flat_indices = topk_indices.T.reshape(-1)
        # gathered_x: [E*k, D, 2] -> [E, k, D, 2]
        x_batched = x[flat_indices].view(self.num_experts, k_nodes, D, 2)

        # 3. Vectorized Complex BMM: (xr + i*xi)(wr + i*wi)
        xr, xi = x_batched[..., 0], x_batched[..., 1]
        # experts_weight_real: [E, D, D, 2]
        wr, wi = self.experts_weight_real[..., 0], self.experts_weight_real[..., 1]
        
        # Restore experts to f32 if they were half-stored, but allow autocast to handle matmul
        wr, wi = wr.to(x_batched.dtype), wi.to(x_batched.dtype)

        yr = torch.matmul(xr, wr) - torch.matmul(xi, wi)
        yi = torch.matmul(xr, wi) + torch.matmul(xi, wr)
        y_all = torch.stack([yr, yi], dim=-1) # [E, k, D, 2]

        # 4. Score Weighting
        # topk_scores: [k, E] -> weights: [E, k, 1, 1]
        weights = topk_scores.T.reshape(self.num_experts, k_nodes, 1, 1).to(y_all.dtype)
        y_weighted = (y_all * weights).reshape(self.num_experts * k_nodes, D, 2)

        # 5. Precision-Balanced Accumulation
        output = torch.zeros((B_T, D, 2), device=x.device, dtype=torch.float32)
        counts = torch.zeros((B_T, 1, 1), device=x.device, dtype=torch.float32)
        
        # Final accumulation MUST be float32 for stable PPC
        # Cache ones for speed: size is self.num_experts * k_nodes
        num_updates = self.num_experts * k_nodes
        if not hasattr(self, '_ones_buf') or self._ones_buf.shape[0] != num_updates or self._ones_buf.device != x.device:
            self.register_buffer('_ones_buf', torch.ones(num_updates, 1, 1, device=x.device, dtype=torch.float32), persistent=False)
            
        output.index_add_(0, flat_indices, y_weighted.float())
        counts.index_add_(0, flat_indices, self._ones_buf)

        # Activation
        res = self.activation(output / counts.clamp(min=1))
        return res.to(original_dtype), topk_indices, topk_scores, counts

    def transpose_forward(self, residual, topk_indices, topk_scores, counts):
        """
        Jacobian-Hermitian pass: Sigma s_i * W_i^H * r[idx_i]
        Fully Vectorized.
        """
        original_dtype = residual.dtype
        B_T, D, _ = residual.shape
        k_nodes = topk_indices.shape[0]

        flat_indices = topk_indices.T.reshape(-1)
        res_batched = residual[flat_indices].view(self.num_experts, k_nodes, D, 2)

        # W^H = [W_r^T, -W_i^T].
        # (yr + i*yi)(wr^T - i*wi^T) = (yr*wr^T + yi*wi^T) + i*(yi*wr^T - yr*wi^T)
        yr, yi = res_batched[..., 0], res_batched[..., 1]
        wr_t = self.experts_weight_real[..., 0].transpose(-2, -1).to(residual.dtype)
        wi_t = self.experts_weight_real[..., 1].transpose(-2, -1).to(residual.dtype)
        
        grad_r = torch.matmul(yr, wr_t) + torch.matmul(yi, wi_t)
        grad_i = torch.matmul(yi, wr_t) - torch.matmul(yr, wi_t)
        grad_all = torch.stack([grad_r, grad_i], dim=-1) # [E, k, D, 2]

        # Weighting
        weights = topk_scores.T.reshape(self.num_experts, k_nodes, 1, 1).to(grad_all.dtype)
        grad_weighted = (grad_all * weights).reshape(self.num_experts * k_nodes, D, 2)

        out = torch.zeros((B_T, D, 2), device=residual.device, dtype=torch.float32)
        out.index_add_(0, flat_indices, grad_weighted.float())

        res = out / counts.clamp(min=1)
        return res.to(original_dtype)

