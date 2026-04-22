import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ComplexKaimingInitializer:
    """
    Initializes interleaved-real parameters [..., 2] with non-negative magnitudes and uniform phases.
    """

    @staticmethod
    def initialize(shape: tuple, gain: float = 1.0, device='cpu', dtype=torch.float32) -> torch.Tensor:
        """
        Memory-efficient initialization using in-place operations to prevent 
        RAM spikes on large-parameter models (3.2B+).
        """
        if len(shape) > 1:
            fan_in = shape[-1] if len(shape) == 1 else shape[-2]
        else:
            fan_in = 1

        scale = gain / math.sqrt(fan_in)
        
        # Pre-allocate the final interleaved-real tensor [..., 2]
        # This saves us from creating separate 'real' and 'imag' temporary tensors.
        out = torch.empty((*shape, 2), device=device, dtype=dtype)
        
        # 1. Compute Phase in-place in the second channel
        # Use out[..., 1] as temporary storage for phase
        phase = out[..., 1].uniform_(0, 2 * math.pi)
        
        # 2. Compute Magnitude in-place in the first channel
        mag = out[..., 0].normal_(0, 1).abs_().mul_(scale)
        
        # 3. Transform to Complex (In-place)
        # Final Real: mag * cos(phase)
        # Final Imag: mag * sin(phase)
        # We must copy phase/mag because we are about to overwrite them
        p_tmp = phase.clone()
        m_tmp = mag.clone()
        
        out[..., 0] = m_tmp * torch.cos(p_tmp)
        out[..., 1] = m_tmp * torch.sin(p_tmp)
        
        return out


# Keep the old name as an alias so existing callers are not broken.
UnitaryInitializer = ComplexKaimingInitializer


class ComplexGELU(nn.Module):
    """
    Complex-valued GELU for interleaved-real tensors [..., 2].
    Applies GELU to the real and imaginary components independently,
    allowing cross-talk via expert routing weights and preserving holomorphism
    for stable Wirtinger calculus gradients.
    """
    def __init__(self, hidden_dim: int, device=None, dtype=torch.float32):
        super().__init__()
        # Keep bias for drop-in compatibility with previous Triton kernels if needed
        self.bias = nn.Parameter(torch.zeros(hidden_dim, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: interleaved-real tensor [..., hidden_dim, 2]
        returns: interleaved-real tensor of same shape
        """
        real = torch.nn.functional.gelu(x[..., 0] + self.bias)
        imag = torch.nn.functional.gelu(x[..., 1] + self.bias)
        return torch.stack([real, imag], dim=-1)

def complex_activation(x: torch.Tensor) -> torch.Tensor:
    """
    Stateless Complex GELU with zero bias.
    """
    real = torch.nn.functional.gelu(x[..., 0])
    imag = torch.nn.functional.gelu(x[..., 1])
    return torch.stack([real, imag], dim=-1)

try:
    import bitsandbytes as bnb
    import bitsandbytes.nn as bnb_nn
    _BNB_AVAILABLE = True
except ImportError:
    _BNB_AVAILABLE = False

try:
    from . import triton_kernels
    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False

class ExpertChoiceMoEMatcher(nn.Module):
    def __init__(self, hidden_dim: int, num_experts: int = 64, k_nodes: int = 2, 
                 device=None, dtype=torch.float32):
        super().__init__()
        self.num_experts = num_experts
        self.k_nodes_default = k_nodes
        self.hidden_dim = hidden_dim

        # 1. Routing Gate (Interleaved)
        self.gate_weights = nn.Parameter(
            torch.empty(hidden_dim * 2, num_experts, device=device, dtype=dtype)
        )
        nn.init.orthogonal_(self.gate_weights, gain=0.1)

        # 2. Experts (Interleaved Real: [E, D, D, 2])
        init_w = ComplexKaimingInitializer.initialize(
            (num_experts, hidden_dim, hidden_dim), 
            gain=1.0, device=device, dtype=dtype
        )
        
        # Training Stability Pivot: We use FP16 for experts. 
        # Sharding handles the VRAM load, and FP16 allows full gradient updates 
        # which NF4 (inference-only) does not support.
        self.experts_weight_real = nn.Parameter(init_w.half())
            
        self.activation = ComplexGELU(hidden_dim, device=device, dtype=dtype)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        """Handles loading into FP16 parameters."""
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)

    def cache_weights(self):
        """Pre-slice, align, and cast weights to FP32 to prevent allocation and quantization jitter in DEQ loops."""
        if self.experts_weight_real is not None:
            self._wr_f32 = self.experts_weight_real[..., 0].contiguous().float()
            self._wi_f32 = self.experts_weight_real[..., 1].contiguous().float()
        else:
            # NF4 dequantization for the search loop
            self._wr_f32 = self.expert_real.data.float()
            self._wi_f32 = self.expert_imag.data.float()
            
        self._wr_t_f32 = self._wr_f32.transpose(-2, -1).contiguous()
        self._wi_t_f32 = self._wi_f32.transpose(-2, -1).contiguous()

    def clear_cache(self):
        self._wr_f32 = None
        self._wi_f32 = None
        self._wr_t_f32 = None
        self._wi_t_f32 = None

    def get_indices(self, x, gate_bias=None):
        """Calculates routing indices and scores for MoE dispatch."""
        if x.dim() == 4:
            B, T, D, _ = x.shape
            B_T = B * T
        else:
            B_T, D, _ = x.shape
        k_nodes = self.k_nodes_default if self.k_nodes_default is not None else max(1, B_T // self.num_experts)

        x_gate_input = x.reshape(B_T, D * 2).to(dtype=self.gate_weights.dtype)
        scores = torch.matmul(x_gate_input, self.gate_weights)  # [B_T, num_experts]

        if gate_bias is not None:
            scores = scores + gate_bias.to(scores.dtype)

        topk_scores, topk_indices = torch.topk(scores, k_nodes, dim=0)  # [k_node, num_experts]
        return topk_indices.T.contiguous(), topk_scores.T.contiguous()

    def compute(self, x_batched, topk_indices, topk_scores, B_T):
        """Computes expert outputs from pre-gathered/pre-delayed token batches."""
        original_dtype = x_batched.dtype
        _, k_nodes, D, _ = x_batched.shape
        flat_indices = topk_indices.T.reshape(-1)

        # 3. Vectorized Complex BMM
        if hasattr(self, '_wr_f32') and self._wr_f32 is not None:
            xr, xi = x_batched[..., 0].float(), x_batched[..., 1].float()
            wr, wi = self._wr_f32, self._wi_f32
            yr = torch.matmul(xr, wr) - torch.matmul(xi, wi)
            yi = torch.matmul(xr, wi) + torch.matmul(xi, wr)
        else:
            xr_h, xi_h = x_batched[..., 0].half(), x_batched[..., 1].half()
            if self.experts_weight_real is not None:
                wr_h = self.experts_weight_real[..., 0].contiguous()
                wi_h = self.experts_weight_real[..., 1].contiguous()
            else:
                wr_h = self.expert_real
                wi_h = self.expert_imag
            yr = torch.matmul(xr_h, wr_h).float() - torch.matmul(xi_h, wi_h).float()
            yi = torch.matmul(xr_h, wi_h).float() + torch.matmul(xi_h, wr_h).float()
            
        y_all = torch.stack([yr, yi], dim=-1) # [E, k, D, 2]

        # 4. Score Weighting
        weights = topk_scores.T.reshape(self.num_experts, k_nodes, 1, 1).to(y_all.dtype)
        y_weighted = (y_all * weights).reshape(self.num_experts * k_nodes, D, 2)
        
        # 5. Fused Aggregation
        if _TRITON_AVAILABLE and x_batched.is_cuda:
            res = triton_kernels.fused_moe_aggregator(
                y_weighted, topk_indices, B_T, self.activation.bias.reshape(D, 1).repeat(1, 2)
            )
            counts_buf = torch.zeros((B_T, 1, 1), device=x_batched.device, dtype=torch.float32)
            counts_buf.index_add_(0, flat_indices, torch.ones((self.num_experts * k_nodes, 1, 1), device=x_batched.device))
        else:
            out_buf = torch.zeros((B_T, D, 2), device=x_batched.device, dtype=torch.float32)
            counts_buf = torch.zeros((B_T, 1, 1), device=x_batched.device, dtype=torch.float32)
            ones_buf = torch.ones((self.num_experts * k_nodes, 1, 1), device=x_batched.device, dtype=torch.float32)
            out_buf.index_add_(0, flat_indices, y_weighted.float())
            counts_buf.index_add_(0, flat_indices, ones_buf)
            res = self.activation(out_buf / counts_buf.clamp(min=1))

        return res.to(original_dtype), topk_indices, topk_scores, counts_buf

    def forward(self, x, gate_bias=None):
        B_T = x.shape[0]
        topk_indices, topk_scores = self.get_indices(x, gate_bias)
        
        # Standard Dispatch (if not using fused external dispatch)
        flat_indices = topk_indices.T.reshape(-1)
        x_batched = x[flat_indices].view(self.num_experts, topk_indices.shape[0], x.shape[1], 2)
        
        return self.compute(x_batched, topk_indices, topk_scores, B_T)

    def transpose_forward(self, residual, topk_indices, topk_scores, counts):
        original_dtype = residual.dtype
        B_T, D, _ = residual.shape
        k_nodes = topk_indices.shape[0]

        flat_indices = topk_indices.T.reshape(-1)
        res_batched = residual[flat_indices].view(self.num_experts, k_nodes, D, 2)

        if hasattr(self, '_wr_t_f32') and self._wr_t_f32 is not None:
            yr, yi = res_batched[..., 0].float(), res_batched[..., 1].float()
            wr_t, wi_t = self._wr_t_f32, self._wi_t_f32
            grad_r = torch.matmul(yr, wr_t) + torch.matmul(yi, wi_t)
            grad_i = torch.matmul(yi, wr_t) - torch.matmul(yr, wi_t)
        else:
            yr_h, yi_h = res_batched[..., 0].half(), res_batched[..., 1].half()
            if self.experts_weight_real is not None:
                wr_t = self.experts_weight_real[..., 0].transpose(-2, -1).contiguous()
                wi_t = self.experts_weight_real[..., 1].transpose(-2, -1).contiguous()
            else:
                wr_t = self.expert_real.transpose(-2, -1).contiguous()
                wi_t = self.expert_imag.transpose(-2, -1).contiguous()
            grad_r = torch.matmul(yr_h, wr_t).float() + torch.matmul(yi_h, wi_t).float()
            grad_i = torch.matmul(yi_h, wr_t).float() - torch.matmul(yr_h, wi_t).float()
            
        grad_all = torch.stack([grad_r, grad_i], dim=-1)
        weights = topk_scores.T.reshape(self.num_experts, k_nodes, 1, 1).to(grad_all.dtype)
        grad_weighted = (grad_all * weights).reshape(self.num_experts * k_nodes, D, 2)

        # 5. Fused Reduction (Reuse aggregator logic but without activation)
        if _TRITON_AVAILABLE and residual.is_cuda:
            # We use a zero-bias to skip activation logic or use a specialized kernel
            # For now, index_add is fine for transpose if we don't have a fused transpose aggregator
            out_buf = torch.zeros((B_T, D, 2), device=residual.device, dtype=torch.float32)
            out_buf.index_add_(0, flat_indices, grad_weighted.float())
            res = out_buf / counts.clamp(min=1)
        else:
            out_buf = torch.zeros((B_T, D, 2), device=residual.device, dtype=torch.float32)
            out_buf.index_add_(0, flat_indices, grad_weighted.float())
            res = out_buf / counts.clamp(min=1)

        return res.to(original_dtype)

class SpectralExpertGate(nn.Module):
    def __init__(self, hidden_dim: int, num_experts: int, device=None, dtype=torch.float32):
        super().__init__()
        self.num_experts = num_experts
        spectral_feat_dim = hidden_dim
        self.low_freq_proj  = nn.Linear(spectral_feat_dim, num_experts, bias=False, device=device, dtype=dtype)
        self.high_freq_proj = nn.Linear(spectral_feat_dim, num_experts, bias=False, device=device, dtype=dtype)
        self.spectral_blend = nn.Parameter(torch.tensor(0.0, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D, _ = x.shape
        # Optimized Magnitude (interleaved norm)
        x_mag = x.norm(dim=-1)
        X_fft = torch.fft.rfft(x_mag, dim=1, norm="ortho")
        
        # Fused Spectral Pooling (Hyper-Drive)
        if _TRITON_AVAILABLE and x.is_cuda:
            # Fused magnitude + pool
            # X_fft is complex; we pass it as [B, T_half, D, 2]
            X_fft_view = torch.view_as_real(X_fft) 
            pooled = triton_kernels.fused_spectral_gate(X_fft_view, B, X_fft.shape[1], D)
            low, high = pooled[..., 0], pooled[..., 1]
        else:
            X_mag = X_fft.abs()
            freq_bins = X_mag.shape[1]
            mid = freq_bins // 2
            low  = X_mag[:, :mid, :].mean(dim=1)
            high = X_mag[:, mid:, :].mean(dim=1)

        low_bias  = self.low_freq_proj(low).unsqueeze(1).expand(B, T, -1)
        high_bias = self.high_freq_proj(high).unsqueeze(1).expand(B, T, -1)
        return (self.spectral_blend * (low_bias + high_bias)).reshape(B * T, -1)


