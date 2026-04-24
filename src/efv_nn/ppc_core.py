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
        fan_in = shape[-2] if len(shape) > 1 else shape[-1] if len(shape) == 1 else 1

        scale = gain / math.sqrt(fan_in)

        # Pre-allocate the final interleaved-real tensor [..., 2]
        out = torch.empty((*shape, 2), device=device, dtype=dtype)

        # Use out[..., 1] as temporary storage for phase, then out[..., 0] for magnitude.
        # Must clone before overwriting because the final formula reads both simultaneously.
        phase = out[..., 1].uniform_(0, 2 * math.pi).clone()
        mag   = out[..., 0].normal_(0, 1).abs_().mul_(scale).clone()

        out[..., 0] = mag * torch.cos(phase)
        out[..., 1] = mag * torch.sin(phase)

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
        out = torch.empty_like(x)
        out[..., 0] = F.gelu(x[..., 0] + self.bias)
        out[..., 1] = F.gelu(x[..., 1] + self.bias)
        return out

def complex_activation(x: torch.Tensor) -> torch.Tensor:
    """
    Stateless Complex GELU with zero bias.
    """
    out = torch.empty_like(x)
    out[..., 0] = F.gelu(x[..., 0])
    out[..., 1] = F.gelu(x[..., 1])
    return out

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

        # Persistent scatter buffers — shape validated and re-allocated lazily on first use.
        # Avoids per-forward torch.zeros() allocations in the non-Triton path.
        self._out_buf: torch.Tensor | None = None
        self._counts_buf: torch.Tensor | None = None
        self._ones_buf: torch.Tensor | None = None
        self._scatter_B_T: int = 0
        self._scatter_n_slots: int = 0

    def cache_weights(self):
        """Pre-slice, align, and cast weights to FP32 to prevent allocation and quantization jitter in DEQ loops."""
        self._wr_f32 = self.experts_weight_real[..., 0].contiguous().float()
        self._wi_f32 = self.experts_weight_real[..., 1].contiguous().float()
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
        k_nodes = min(B_T, k_nodes) # Safety Clamp

        x_gate_input = x.reshape(B_T, D * 2).to(dtype=self.gate_weights.dtype)
        scores = torch.matmul(x_gate_input, self.gate_weights)  # [B_T, num_experts]

        if gate_bias is not None:
            scores = scores + gate_bias.to(scores.dtype)

        # Non-blocking NaN Siphon
        scores = torch.nan_to_num(scores, nan=0.0)

        topk_scores, topk_indices = torch.topk(scores, k_nodes, dim=0)  # [k_node, num_experts]
        return topk_indices.T.contiguous(), topk_scores.T.contiguous()

    def _ensure_scatter_bufs(self, B_T: int, D: int, n_slots: int, device, dtype):
        """Lazily allocate/resize persistent scatter buffers for the non-Triton path."""
        if (self._scatter_B_T != B_T or self._scatter_n_slots != n_slots
                or self._out_buf is None or self._out_buf.device != device):
            self._out_buf    = torch.zeros((B_T, D, 2), device=device, dtype=dtype)
            self._counts_buf = torch.zeros((B_T, 1, 1), device=device, dtype=dtype)
            self._ones_buf   = torch.ones((n_slots, 1, 1), device=device, dtype=dtype)
            self._scatter_B_T    = B_T
            self._scatter_n_slots = n_slots
        else:
            self._out_buf.zero_()
            self._counts_buf.zero_()

    def compute(self, x_batched, topk_indices, topk_scores, B_T):
        """Computes expert outputs from pre-gathered/pre-delayed token batches."""
        original_dtype = x_batched.dtype
        _, k_nodes, D, _ = x_batched.shape
        flat_indices = topk_indices.T.reshape(-1)

        # 3. Vectorized Complex BMM — use cached FP32 weights when available (normal DEQ path),
        #    fall back to a direct FP16 cast for inference calls outside the DEQ loop.
        if hasattr(self, '_wr_f32') and self._wr_f32 is not None:
            xr = x_batched[..., 0].float()
            xi = x_batched[..., 1].float()
            wr, wi = self._wr_f32, self._wi_f32
            yr = torch.matmul(xr, wr) - torch.matmul(xi, wi)
            yi = torch.matmul(xr, wi) + torch.matmul(xi, wr)
        else:
            xr_h = x_batched[..., 0].half()
            xi_h = x_batched[..., 1].half()
            wr_h = self.experts_weight_real[..., 0].contiguous()
            wi_h = self.experts_weight_real[..., 1].contiguous()
            yr = torch.matmul(xr_h, wr_h).float() - torch.matmul(xi_h, wi_h).float()
            yi = torch.matmul(xr_h, wi_h).float() + torch.matmul(xi_h, wr_h).float()

        # Write directly into a pre-allocated output tensor instead of torch.stack + new alloc
        y_all = torch.empty(yr.shape[0], yr.shape[1], D, 2, dtype=yr.dtype, device=yr.device)
        y_all[..., 0] = yr
        y_all[..., 1] = yi

        # Non-blocking NaN Siphon
        y_all = torch.nan_to_num(y_all, nan=0.0)

        # 4. Score Weighting
        weights = topk_scores.reshape(self.num_experts, k_nodes, 1, 1).to(y_all.dtype)
        y_weighted = (y_all * weights).reshape(self.num_experts * k_nodes, D, 2)

        # 5. Fused Aggregation
        if _TRITON_AVAILABLE and x_batched.is_cuda:
            res = triton_kernels.fused_moe_aggregator(
                y_weighted, topk_indices, B_T, self.activation.bias
            )
            counts_buf = torch.zeros((B_T, 1, 1), device=x_batched.device, dtype=torch.float32)
            counts_buf.index_add_(0, flat_indices, torch.ones((self.num_experts * k_nodes, 1, 1), device=x_batched.device))
        else:
            n_slots = self.num_experts * k_nodes
            if torch.is_grad_enabled():
                # Backward recomputation path: use fresh buffers so gradient flows
                # through y_weighted → BMM → expert params. Persistent buffers cannot
                # be used here because they would be zeroed in-place on the next forward
                # while the backward graph still holds a reference to them.
                out_buf = torch.zeros((B_T, D, 2), device=x_batched.device, dtype=torch.float32)
                counts_buf = torch.zeros((B_T, 1, 1), device=x_batched.device, dtype=torch.float32)
                ones = torch.ones((n_slots, 1, 1), device=x_batched.device, dtype=torch.float32)
                out_buf.index_add_(0, flat_indices, y_weighted.float())
                counts_buf.index_add_(0, flat_indices, ones)
                agg = out_buf / counts_buf.clamp(min=1)
            else:
                # No-grad forward loop path: reuse persistent buffers to avoid allocation churn
                self._ensure_scatter_bufs(B_T, D, n_slots, x_batched.device, torch.float32)
                self._out_buf.index_add_(0, flat_indices, y_weighted.float())
                self._counts_buf.index_add_(0, flat_indices, self._ones_buf)
                # Detach: the persistent buffer will be zeroed in-place on the next iteration;
                # no gradient needed in the no-grad forward path anyway.
                agg = (self._out_buf / self._counts_buf.clamp(min=1)).detach()
                counts_buf = self._counts_buf.detach()
            res = self.activation(agg)

        return res.to(original_dtype), topk_indices, topk_scores, counts_buf

    def forward(self, x, gate_bias=None):
        B_T = x.shape[0]
        topk_indices, topk_scores = self.get_indices(x, gate_bias)

        # Standard Dispatch (if not using fused external dispatch)
        flat_indices = topk_indices.reshape(-1)
        x_batched = x[flat_indices].view(self.num_experts, topk_indices.shape[1], x.shape[1], 2)

        return self.compute(x_batched, topk_indices, topk_scores, B_T)

    def transpose_forward(self, residual, topk_indices, topk_scores, counts):
        original_dtype = residual.dtype
        B_T, D, _ = residual.shape
        num_experts, k_nodes = topk_indices.shape

        flat_indices = topk_indices.reshape(-1)
        res_batched = residual[flat_indices].view(num_experts, k_nodes, D, 2)

        yr = res_batched[..., 0].float()
        yi = res_batched[..., 1].float()
        if hasattr(self, '_wr_t_f32') and self._wr_t_f32 is not None:
            wr_t, wi_t = self._wr_t_f32, self._wi_t_f32
        else:
            # transpose_forward is not on the critical path; cast directly from FP16
            wr_t = self.experts_weight_real[..., 0].transpose(-2, -1).contiguous().float()
            wi_t = self.experts_weight_real[..., 1].transpose(-2, -1).contiguous().float()
        grad_r = torch.matmul(yr, wr_t) + torch.matmul(yi, wi_t)
        grad_i = torch.matmul(yi, wr_t) - torch.matmul(yr, wi_t)

        grad_all = torch.empty_like(res_batched, dtype=grad_r.dtype)
        grad_all[..., 0] = grad_r
        grad_all[..., 1] = grad_i
        weights = topk_scores.T.reshape(self.num_experts, k_nodes, 1, 1).to(grad_all.dtype)
        grad_weighted = (grad_all * weights).reshape(self.num_experts * k_nodes, D, 2)

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
            if mid > 0:
                low  = X_mag[:, :mid, :].mean(dim=1)
                high = X_mag[:, mid:, :].mean(dim=1)
            else:
                # Fallback for T=1: use the only available bin (DC component) for both
                low  = X_mag.squeeze(1)
                high = X_mag.squeeze(1)

        # unsqueeze+expand deferred to after projection; reshape directly avoids materializing
        # the expanded tensor (expand is a view, but the subsequent reshape forces contiguous).
        low_bias  = self.low_freq_proj(low).unsqueeze(1).expand(B, T, -1)
        high_bias = self.high_freq_proj(high).unsqueeze(1).expand(B, T, -1)
        return (self.spectral_blend * (low_bias + high_bias)).reshape(B * T, -1)
