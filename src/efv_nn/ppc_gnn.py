import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from efv_nn import ppc_core
from efv_nn.ppc_core import SpectralExpertGate, ExpertChoiceMoEMatcher
from efv_nn.deq_solvers import anderson_acceleration, DEQFunction
from efv_nn import diagnostics

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
    def __init__(self, hidden_dim: int, num_experts: int = 64, local_lr: float = 0.05,
                 lr_decay: float = 0.8, use_jacobian: bool = False,
                 prime_delays=(1, 2, 3, 5), use_triton: bool = True,
                 device=None, dtype=torch.float32, tolerance=1e-3, min_iters=8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.base_local_lr = max(0.0, min(0.99, local_lr))
        self.lr_decay = lr_decay
        self.tolerance = tolerance
        self.use_jacobian = use_jacobian
        self.min_iters = min_iters  # APD: hard floor, model can never learn to skip thinking

        # Adaptive Phasal Depth (APD): Learnable exit threshold with hard floor guard.
        # APD: Adaptive Phasal Depth threshold (0.000001 = 0.0001% phasal resonance error)
        self.exit_threshold = nn.Parameter(torch.tensor(0.000001, device=device, dtype=dtype))

        # OCNS Integration: Phasal Delay Embedding Gains
        self.prime_delays = list(prime_delays) if prime_delays else []
        if self.prime_delays:
            # Shape: [num_delays, hidden_dim, 2] - allows complex scaling and phase rotation
            self.delay_gains = nn.Parameter(torch.zeros(len(self.prime_delays), hidden_dim, 2, device=device, dtype=dtype))

        # Phase rotation parameters: store as cos/sin for manual rotation
        phase = torch.rand(hidden_dim, device=device) * 2 * math.pi
        self.register_buffer('cos_p', torch.cos(phase))
        self.register_buffer('sin_p', torch.sin(phase))

        # Check for Triton availability (used in forward)
        self._triton_available = TRITON_AVAILABLE and use_triton and torch.cuda.is_available()

        self.spectral_gate = SpectralExpertGate(hidden_dim, num_experts, device=device, dtype=dtype)
        # We determine the target device to avoid RAM spikes during initialization
        self.moe = ExpertChoiceMoEMatcher(
            hidden_dim, num_experts=num_experts, 
            device=device, dtype=dtype
        )
        # Pillar 5: Selective Compilation (REMOVED)
        # We previously compiled the MoE, but CUDAGraphs (used in reduce-overhead) 
        # corrupts the memory pool when switching between no_grad (loop) and grad (bridge).
        # Our manual FP16 matmul optimizations are sufficient.

        # Persistent Buffers (avoiding allocation churn)
        self._target_buf = None
        self._eff_buf    = None
        self._final_buf  = None
        
        # Adjoint Cache for Warm-Starting (IFT Pillar)
        self.register_buffer('_adjoint_cache', torch.zeros(1))
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
        
    def forward(self, x_stream, local_iters=8, gate_bias=None, rolling_energy=None):
        """
        x_stream: [B, T, hidden_dim, 2] interleaved real
        rolling_energy: Optional scalar representing the current phasal resonance energy.
        """
        unbatched = x_stream.dim() == 3
        if unbatched:
            x_stream = x_stream.unsqueeze(0)

        original_dtype = x_stream.dtype
        # Holistic move: PPC iterations MUST be float32 for convergence stability
        x_stream = torch.nan_to_num(x_stream.float(), nan=0.0)

        # Pillar 4: Autocast Isolation. We force everything into FP32 to prevent 
        # compiler-drift in the iterative loop, ensuring E ~ 0.024 baseline.
        with torch.amp.autocast('cuda', enabled=False):
            B, T, D, _ = x_stream.shape
            iters_run = 0

            # Pillar 1: Compute Spectral Routing Bias once per layer.
            # Honour externally provided gate_bias (e.g. from SpectralShardedPPCGraphLLM);
            # only compute internally when none was passed in.
            if gate_bias is None:
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
                
                diagnostics.debug_print_nan(x_target, "PPCNodeLayer.x_target")
                if gate_bias is not None:
                    diagnostics.debug_print_nan(gate_bias, "PPCNodeLayer.gate_bias")


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
            def f_forward_step(x, _x_init, g_bias, target):
                B, T, D, _ = x.shape
                # Hybrid Bridge: Use Triton for Forward Speed, PyTorch for Gradient Connectivity
                use_triton_now = self._triton_available and not torch.is_grad_enabled()

                if use_triton_now:
                    # Pillar 5: Dynamic Routing + Dispatch (Triton Path)
                    topk_indices, topk_scores = self.moe.get_indices(x, gate_bias=g_bias)
                    x_batched = triton_kernels.fused_moe_dispatch_delay(
                        x, self.delay_gains, self.prime_delays, topk_indices
                    )
                    pred, _, _, _ = self.moe.compute(x_batched, topk_indices, topk_scores, B * T)
                    pred = pred.reshape(B, T, D, 2)
                else:
                    # Pillar 5: Dynamic Routing + Dispatch (PyTorch Path - Reconnects Autograd)
                    if self.prime_delays:
                        x_eff = self._apply_ocns_delays(x)
                    else:
                        x_eff = x
                    B_in, T_in, D_in, _ = x_eff.shape
                    pred, _, _, _ = self.moe(x_eff.reshape(B_in * T_in, D_in, 2), gate_bias=g_bias)
                    pred = pred.float().reshape(B_in, T_in, D_in, 2)
                
                # Pillar 6: Contractive Residual. 
                # We clip the update to ensure the state remains within a stable basin.
                update = self.base_local_lr * (target - pred)
                update = torch.clamp(update, -10.0, 10.0) 
                
                return x + update

            def f_solver(x_init, g_bias, target):
                # APD Relaxation: Dynamic tolerance allows early exit during high-energy phases (Phase 0).
                # tol is compared directly against res_norm (L2 norm), not its square.
                base_tol = self.exit_threshold.item()
                if rolling_energy is not None:
                    # Invert relationship: high energy = volatile phase = need tighter convergence.
                    # Divide base_tol by (1 + scaled_energy) to tighten when energy is high.
                    dynamic_tol = base_tol / (1.0 + rolling_energy * 0.1)
                else:
                    dynamic_tol = base_tol

                return anderson_acceleration(
                    lambda x: f_forward_step(x, x_init, g_bias, target),
                    x_init,
                    m=5, max_iter=local_iters, tol=dynamic_tol
                )

            # Ensure Adjoint Cache matches the current stream shape for warm-starting
            if self._adjoint_cache.shape != x_stream.shape:
                # We use a non-parameter buffer to persist state across backward passes
                self._adjoint_cache = torch.zeros_like(x_stream)

            # --- Implicit Differentiation (IFT) Bridge ---
            # Collect all trainable parameters to ensure they are tracked by the DEQ backward solver.
            layer_params = [p for p in self.parameters() if p.requires_grad]
            
            out, iters_run, res_norm = DEQFunction.apply(
                x_stream, f_solver, f_forward_step, gate_bias, x_target, self.moe.cache_weights, self.moe.clear_cache, self._adjoint_cache, [self.moe], *layer_params
            )

        if unbatched:
            out = out.squeeze(0)
            
        return out.to(original_dtype), iters_run, res_norm


class PPCGraphLLM(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int, num_layers: int = 2,
                 num_experts: int = 4, local_lr: float = 0.05, lr_decay: float = 0.85, use_jacobian: bool = False):
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

    def forward(self, input_ids: torch.Tensor, local_iters: int = 8, rolling_energy: float = None) -> torch.Tensor:
        x = self.embed(input_ids)            # [B, T, D, 2]

        total_iters = 0
        layer_energies = torch.empty(len(self.layers), device=x.device, dtype=torch.float32)
        for i, layer in enumerate(self.layers):
            x, iters, res_norm = layer(x, local_iters, rolling_energy=rolling_energy)
            total_iters += iters
            layer_energies[i] = res_norm

        avg_energy = layer_energies.mean()

        # Flatten [..., D, 2] to [..., 2D] for decoder
        x_flat = x.flatten(-2)
        x_norm = self.layer_norm(x_flat)
        logits = self.output_head(x_norm)
        return logits, total_iters / len(self.layers), avg_energy, layer_energies
