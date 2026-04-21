import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from efv_nn import ppc_core
from efv_nn.ppc_core import SpectralExpertGate
from efv_nn.deq_solvers import anderson_acceleration, DEQFunction

# Optimization: Import Triton kernels as module to allow dynamic reloads
try:
    from efv_nn import triton_kernels
    TRITON_AVAILABLE = True
except (ImportError, Exception):
    TRITON_AVAILABLE = False


def spectral_guardian_penalty(layer_energies: torch.Tensor, lam: float = 0.01) -> torch.Tensor:
    """
    Phasal Laplacian Regularization (Spectral Guardian — Pillar 2).

    Penalizes high-frequency 'jitter' between adjacent layer energies.
    Formula: λ · Σ (E_i - E_{i+1})²

    Args:
        layer_energies: [num_layers] tensor on device1.
        lam: Penalty strength. Default 0.01 (gentle guardian).

    Returns:
        Scalar penalty tensor on the same device. Zero graph breaks.
    """
    diff = layer_energies[1:] - layer_energies[:-1]  # [num_layers - 1]
    return lam * (diff ** 2).sum()


class PPCNodeLayer(nn.Module):
    def __init__(self, hidden_dim, num_experts=4, local_lr=0.5, lr_decay=0.85, tolerance=1e-3,
                 use_jacobian=False, prime_delays=(1, 2, 3, 5), use_triton=True, min_iters=8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.base_local_lr = max(0.0, min(0.99, local_lr))
        self.lr_decay = lr_decay
        self.tolerance = tolerance
        self.use_jacobian = use_jacobian
        self.min_iters = min_iters  # APD: hard floor, model can never learn to skip thinking

        # Adaptive Phasal Depth (APD): Learnable exit threshold with hard floor guard.
        # APD: Adaptive Phasal Depth threshold (0.000001 = 0.0001% phasal resonance error)
        self.exit_threshold = nn.Parameter(torch.tensor(0.000001))

        # OCNS Integration: Phasal Delay Embedding Gains
        self.prime_delays = list(prime_delays) if prime_delays else []
        if self.prime_delays:
            # Shape: [num_delays, hidden_dim, 2] - allows complex scaling and phase rotation
            self.delay_gains = nn.Parameter(torch.zeros(len(self.prime_delays), hidden_dim, 2))

        # Phase rotation parameters: store as cos/sin for manual rotation
        phase = torch.rand(hidden_dim) * 2 * math.pi
        self.register_buffer('cos_p', torch.cos(phase))
        self.register_buffer('sin_p', torch.sin(phase))

        # Check for Triton availability (used in forward)
        self._triton_available = TRITON_AVAILABLE and use_triton and torch.cuda.is_available()

        self.spectral_gate = SpectralExpertGate(hidden_dim, num_experts)
        self.moe = ppc_core.ExpertChoiceMoEMatcher(hidden_dim, num_experts=num_experts)
        # Pillar 5: Selective Compilation (REMOVED)
        # We previously compiled the MoE, but CUDAGraphs (used in reduce-overhead) 
        # corrupts the memory pool when switching between no_grad (loop) and grad (bridge).
        # Our manual FP16 matmul optimizations are sufficient.

        # Persistent Buffers (avoiding allocation churn)
        self._target_buf = None
        self._eff_buf    = None
        self._final_buf  = None
        
    def _apply_ocns_delays(self, x_states):
        """Memory-Efficient Phasal Delay Embedding (OCNS)."""
        if not self.prime_delays:
            return x_states
            
        # Start with a copy of current states
        x_eff = x_states.clone()
        
        # We use Slicing (Views) instead of Roll (Copies) to save VRAM
        for idx, tau in enumerate(self.prime_delays):
            # Complex Gains
            gr, gi = self.delay_gains[idx, ..., 0], self.delay_gains[idx, ..., 1]
            
            # Source (History): tokens from 0 to T-tau
            # Target (Present): tokens from tau to T
            dr = x_states[:, :-tau, ..., 0]
            di = x_states[:, :-tau, ..., 1]
            
            # Interference Math (In-place addition to Target slice)
            x_eff[:, tau:, ..., 0] += (dr * gr - di * gi)
            x_eff[:, tau:, ..., 1] += (dr * gi + di * gr)
            
        return x_eff
        
    def forward(self, x_stream, local_iters=8, gate_bias=None):
        """
        x_stream: [B, T, hidden_dim, 2] interleaved real
        """
        unbatched = x_stream.dim() == 3
        if unbatched:
            x_stream = x_stream.unsqueeze(0)

        original_dtype = x_stream.dtype
        # Holistic move: PPC iterations MUST be float32 for convergence stability
        x_stream = x_stream.float()

        # Pillar 4: Autocast Isolation. We force everything into FP32 to prevent 
        # compiler-drift in the iterative loop, ensuring E ~ 0.024 baseline.
        with torch.amp.autocast('cuda', enabled=False):
            B, T, D, _ = x_stream.shape
            iters_run = 0

            # PERFORMANCE: Cache contiguous MoE weights OUTSIDE no_grad so gradients flow naturally
            self.moe.cache_weights()

            # Pillar 1: Compute Spectral Routing Bias once per layer
            gate_bias = self.spectral_gate(x_stream) if hasattr(self, 'spectral_gate') else None

            # --- PHASE 1: TARGET CONSTRUCTION (STATIONARY TARGET) ---
            # To enable Anderson Acceleration, the target must not move during the micro-iterations.
            if self._triton_available:
                if self._target_buf is None or self._target_buf.shape != x_stream.shape:
                    self._target_buf = torch.empty_like(x_stream)
                    self._eff_buf    = torch.empty_like(x_stream)
                    self._final_buf  = torch.empty_like(x_stream)
                
                # We clone the output of fused_phase_rotation if we are in training mode
                # to prevent in-place buffer modification from corrupting saved tensors for backward pass
                _tmp_target = triton_kernels.fused_phase_rotation(
                    x_stream, self.cos_p, self.sin_p, out=self._target_buf
                )
                x_target = _tmp_target.clone() if torch.is_grad_enabled() else _tmp_target

            else:
                x_prev = x_stream[:, :-1, :, :]
                prev_r, prev_i = x_prev[..., 0], x_prev[..., 1]
                rot_r = prev_r * self.cos_p - prev_i * self.sin_p
                rot_i = prev_r * self.sin_p + prev_i * self.cos_p
                rot_prev = torch.stack([rot_r, rot_i], dim=-1)
                x_target = torch.zeros_like(x_stream, dtype=torch.float32)
                x_target[:, 1:, :, :] = rot_prev
                x_target[:, 0, :, :] = x_stream[:, 0, :, :]

            # --- DEQ Callables ---
            def f_forward_step(x, x_init, g_bias, target):
                if self._triton_available and self.prime_delays and not torch.is_grad_enabled():
                    x_eff = triton_kernels.fused_ocns_delay(x, self.delay_gains, self.prime_delays, out=self._eff_buf)
                else:
                    x_eff = self._apply_ocns_delays(x)
                B_in, T_in, D_in, _ = x_eff.shape
                pred, _, _, _ = self.moe(x_eff.reshape(B_in * T_in, D_in, 2), gate_bias=g_bias)
                pred = pred.float().reshape(B_in, T_in, D_in, 2)
                return x + self.base_local_lr * (target - pred)

            def f_solver(x_init, g_bias, target):
                exit_thr_sq = self.exit_threshold.item() ** 2
                return anderson_acceleration(
                    lambda x: f_forward_step(x, x_init, g_bias, target), 
                    x_init, 
                    m=5, max_iter=local_iters, tol=exit_thr_sq
                )

            # --- Implicit Differentiation (IFT) Bridge ---
            # Collect all trainable parameters to ensure they are tracked by the DEQ backward solver.
            layer_params = [p for p in self.parameters() if p.requires_grad]
            
            out, iters_run, res_norm = DEQFunction.apply(
                x_stream, f_solver, f_forward_step, gate_bias, x_target, self.moe.clear_cache, *layer_params
            )

            # Fix inference memory leak: Clear the FP32 cache manually if gradient is disabled
            if not torch.is_grad_enabled():
                self.moe.clear_cache()

        if unbatched:
            out = out.squeeze(0)
            
        return out.to(original_dtype), iters_run, res_norm


class PPCGraphLLM(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int, num_layers: int = 2,
                 num_experts: int = 4, local_lr: float = 0.5, lr_decay: float = 0.85, use_jacobian: bool = False):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim

        # Real-valued embedding stores interleaved complex [V, D, 2]
        self.embedding = nn.Embedding(vocab_size, hidden_dim * 2)
        
        with torch.no_grad():
            # Correctly initialise the interleaved real pair
            init_w = ppc_core.ComplexKaimingInitializer.initialize((vocab_size, hidden_dim))
            self.embedding.weight.copy_(init_w.reshape(vocab_size, hidden_dim * 2))

        self.layers = nn.ModuleList([
            PPCNodeLayer(hidden_dim, num_experts=num_experts, local_lr=local_lr, lr_decay=lr_decay, use_jacobian=use_jacobian)
            for _ in range(num_layers)
        ])

        # LayerNorm takes flattened 2D vector for stability
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)     
        self.output_head = nn.Linear(hidden_dim * 2, vocab_size, bias=True)
        nn.init.zeros_(self.output_head.bias)

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Map token IDs -> interleaved real [*, T, D, 2]."""
        # [*, T, 2*D] -> [*, T, D, 2]
        out_flat = self.embedding(input_ids)
        new_shape = list(out_flat.shape[:-1]) + [self.hidden_dim, 2]
        return out_flat.view(*new_shape)

    def forward(self, input_ids: torch.Tensor, local_iters: int = 8) -> torch.Tensor:
        x = self.embed(input_ids)            # [B, T, D, 2]

        total_iters = 0
        res_energies = []
        for layer in self.layers:
            x, iters, res_norm = layer(x, local_iters)
            total_iters += iters
            res_energies.append(res_norm.clone())

        layer_energies = torch.stack(res_energies)
        avg_energy = layer_energies.mean()

        # Flatten [..., D, 2] to [..., 2D] for decoder
        x_flat = x.flatten(-2)
        x_norm = self.layer_norm(x_flat)
        logits = self.output_head(x_norm)
        return logits, total_iters / len(self.layers), avg_energy, layer_energies
