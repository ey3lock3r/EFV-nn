import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from efv_nn import ppc_core

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
        # Initialized to a small value (0.01) so thinking happens by default.
        self.exit_threshold = nn.Parameter(torch.tensor(0.01))

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

            # 1. Local Convergence (Frozen fixed point search)
            with torch.no_grad():
                device = self.cos_p.device
                x_states = x_stream.clone().detach().float().to(device)

                # Buffer Management: Pre-allocate or reuse
                if self._triton_available:
                    if self._target_buf is None or self._target_buf.shape != x_states.shape:
                        self._target_buf = torch.empty_like(x_states)
                        self._eff_buf    = torch.empty_like(x_states)
                        self._final_buf  = torch.empty_like(x_states)

                # --- Phase Rotation + Target Construction ---
                if self._triton_available:
                    x_target_frozen = triton_kernels.fused_phase_rotation(
                        x_states, self.cos_p, self.sin_p, out=self._target_buf
                    )
                else:
                    x_prev = x_states[:, :-1, :, :]
                    prev_r, prev_i = x_prev[..., 0], x_prev[..., 1]
                    rot_r = prev_r * self.cos_p - prev_i * self.sin_p
                    rot_i = prev_r * self.sin_p + prev_i * self.cos_p
                    rot_prev = torch.stack([rot_r, rot_i], dim=-1)
                    x_target_frozen = torch.zeros_like(x_states, dtype=torch.float32)
                    x_target_frozen[:, 1:, :, :] = rot_prev
                    x_target_frozen[:, 0, :, :] = x_states[:, 0, :, :]

                # APD: get exit threshold value (clamped to positive, honoring min_iters floor)
                # We use squared threshold to avoid sqrt on GPU
                exit_thr_sq = self.exit_threshold.item() ** 2

                current_lr = self.base_local_lr
                for i in range(local_iters):
                    iters_run += 1

                    # --- OCNS INJECTION POINT 1 ---
                    if self._triton_available and self.prime_delays:
                        x_eff = triton_kernels.fused_ocns_delay(
                            x_states, self.delay_gains, self.prime_delays, out=self._eff_buf
                        )
                    else:
                        x_eff = self._apply_ocns_delays(x_states)

                    prediction, indices, scores, counts = self.moe(
                        x_eff.reshape(B * T, D, 2), gate_bias=gate_bias
                    )
                    prediction = prediction.float().reshape(B, T, D, 2)
                    residual = x_target_frozen - prediction

                    if self.use_jacobian:
                        step = self.moe.transpose_forward(
                            residual.reshape(B * T, D, 2), indices, scores, counts
                        ).float().reshape(B, T, D, 2)
                        # --- State Update ---
                        if self._triton_available:
                            triton_kernels.fused_state_update(x_states, step, current_lr)
                        else:
                            x_states.add_(torch.clamp(step, -10.0, 10.0), alpha=current_lr)
                    else:
                        if self._triton_available:
                            triton_kernels.fused_state_update(x_states, residual, current_lr)
                        else:
                            x_states.add_(torch.clamp(residual, -10.0, 10.0), alpha=current_lr)

                    current_lr *= self.lr_decay

                    # --- Adaptive Phasal Depth: Early Exit (Check every 8 iters to avoid sync overhead) ---
                    if (i + 1) >= self.min_iters and (i + 1) % 8 == 0:
                        # Optimization: Use squared norm (averaged) to avoid sync and sqrt
                        res_sq = torch.mean(residual * residual) * 2
                        if res_sq.item() < exit_thr_sq:
                            break
                        if torch.isnan(res_sq):
                            # Poisoning detected, but we keep thinking to see if Spectral Guardian can heal it
                            pass

                # --- Final Normalize + Activate ---
                # This step is critical for phasal resonance stability.
                if self._triton_available:
                    x_states = triton_kernels.fused_normalize_activate(
                        prediction, counts, self.moe.activation.bias, out=self._final_buf
                    )
                else:
                    x_states = self._normalize_activate(prediction, counts)
            
            # 2. DEQ Gradient Attachment
            # Everything here stays in float32 for the analytical bridge
            x_prev_grad = x_stream[:, :-1, :, :]
            pg_r, pg_i = x_prev_grad[..., 0], x_prev_grad[..., 1]
            rg_r = pg_r * self.cos_p - pg_i * self.sin_p
            rg_i = pg_r * self.sin_p + pg_i * self.cos_p
            
            x_target_grad = torch.zeros_like(x_stream, dtype=torch.float32)
            x_target_grad[:, 1:, :, 0] = rg_r
            x_target_grad[:, 1:, :, 1] = rg_i
            x_target_grad[:, 0, :, :] = x_stream[:, 0, :, :]
            
            # --- OCNS INJECTION POINT 2 (Gradient Path) ---
            x_eff_grad = self._apply_ocns_delays(x_states)
            
            prediction_grad, _, _, _ = self.moe(x_eff_grad.reshape(B*T, D, 2))
            prediction_grad = prediction_grad.float().reshape(B, T, D, 2)
            
            # Bridge uses un-decayed base_local_lr
            out = x_states + self.base_local_lr * (x_target_grad - prediction_grad)
            
            # Calculate final resonance energy (L2 norm of error) for swarm/diagnostics
            # Remove no_grad() to allow Spectral Guardian to penalise phasal jitter
            res_norm = torch.norm(x_target_frozen - prediction_grad, dim=-1).mean()
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
