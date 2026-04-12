import torch
import torch.nn as nn
import math
from efv_nn.ppc_core import ExpertChoiceMoEMatcher


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
    def __init__(self, hidden_dim, num_experts=4, local_lr=0.5, lr_decay=0.85, tolerance=1e-3, use_jacobian=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.base_local_lr = max(0.0, min(0.99, local_lr))
        self.lr_decay = lr_decay
        self.tolerance = tolerance
        self.use_jacobian = use_jacobian
        
        # Phase rotation parameters: store as cos/sin for manual rotation
        phase = torch.rand(hidden_dim) * 2 * math.pi
        self.register_buffer('cos_p', torch.cos(phase))
        self.register_buffer('sin_p', torch.sin(phase))
        
        self.moe = ExpertChoiceMoEMatcher(hidden_dim, num_experts)
        
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

        B, T, D, _ = x_stream.shape
        iters_run = 0
        
        # 1. Local Convergence (Frozen fixed point search)
        with torch.no_grad():
            x_states = x_stream.clone().detach()
            
            # Rotation buffers are typically float32, math should preserve it
            x_prev = x_states[:, :-1, :, :] # [B, T-1, D, 2]
            prev_r, prev_i = x_prev[..., 0], x_prev[..., 1]
            
            rot_r = prev_r * self.cos_p - prev_i * self.sin_p
            rot_i = prev_r * self.sin_p + prev_i * self.cos_p
            rot_prev = torch.stack([rot_r, rot_i], dim=-1)
            
            x_target_frozen = torch.zeros_like(x_states, dtype=torch.float32)
            x_target_frozen[:, 1:, :, :] = rot_prev
            x_target_frozen[:, 0, :, :] = x_states[:, 0, :, :]
            
            # Fixed-length loop for peak fusion: 16 small fused steps > 5 synced steps
            current_lr = self.base_local_lr
            for i in range(local_iters):
                iters_run += 1
                prediction, indices, scores, counts = self.moe(x_states.reshape(B*T, D, 2), gate_bias=gate_bias)
                prediction = prediction.float().reshape(B, T, D, 2)
                residual = x_target_frozen - prediction

                if self.use_jacobian:
                    step = self.moe.transpose_forward(
                        residual.reshape(B*T, D, 2), indices, scores, counts
                    ).float().reshape(B, T, D, 2)
                    # Safety Clamp: Prevent explosive updates in early random-weight phase
                    x_states.add_(torch.clamp(step, -10.0, 10.0), alpha=current_lr)
                else:
                    x_states.add_(torch.clamp(residual, -10.0, 10.0), alpha=current_lr)
                    
                current_lr *= self.lr_decay
        
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
        
        prediction_grad, _, _, _ = self.moe(x_states.reshape(B*T, D, 2))
        prediction_grad = prediction_grad.float().reshape(B, T, D, 2)
        
        # Bridge uses un-decayed base_local_lr
        out = x_states + self.base_local_lr * (x_target_grad - prediction_grad)
        
        # Calculate final resonance energy (L2 norm of error) for swarm/diagnostics
        with torch.no_grad():
            res_norm = torch.norm(x_target_frozen - prediction_grad, dim=-1).mean()

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
            from efv_nn.ppc_core import ComplexKaimingInitializer
            # Correctly initialise the interleaved real pair
            init_w = ComplexKaimingInitializer.initialize((vocab_size, hidden_dim))
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
            res_energies.append(res_norm)

        layer_energies = torch.stack(res_energies)
        avg_energy = layer_energies.mean()

        # Flatten [..., D, 2] to [..., 2D] for decoder
        x_flat = x.flatten(-2)
        x_norm = self.layer_norm(x_flat)
        logits = self.output_head(x_norm)
        return logits, total_iters / len(self.layers), avg_energy, layer_energies
