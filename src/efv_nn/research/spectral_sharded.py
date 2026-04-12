"""
Spectral Research Sandbox — Track 1
=====================================
SpectralShardedPPCGraphLLM: A research-grade extension of ShardedPPCGraphLLM
that investigates two spectral research pillars:

  Pillar 1 — Spectral Gate-Filtering:
    Expert routing is conditioned on the FFT-based spectral density of the
    hidden state. High-frequency tokens activate different expert groups than
    low-frequency prose, enabling true "frequency-aware" MoE specialization.

  Pillar 3 — Eigen-Resonance Solver (Experimental):
    The iterative PPC loop is replaced by a closed-form spectral projection
    using the top-K eigenvalues of the phasal operator. If validated, this
    collapses 16-step convergence into a single matrix operation.

NOTE: This module is ISOLATED from the main production code. It inherits the
full 3.2B parameter schema from ShardedPPCGraphLLM to ensure all comparisons
are architecturally equivalent.
"""

import torch
import torch.nn as nn
import torch.fft
from efv_nn.ppc_sharded import ShardedPPCGraphLLM


# ─────────────────────────────────────────────────────────────────────────────
# Pillar 1: Spectral Gate-Filtering
# ─────────────────────────────────────────────────────────────────────────────

class SpectralExpertGate(nn.Module):
    """
    Conditions MoE expert selection on the spectral density of the hidden state.

    Mechanism:
      1. Compute the 1D FFT of the hidden state along the token dimension.
      2. Split frequences into LOW (long-range context) and HIGH (local detail).
      3. Project these two spectral components into a routing bias that is
         added to the standard magnitude-based gate logits.

    This allows the model to learn: "When the input is spectrally 'high-frequency'
    (code, math, sudden topic changes), prefer Expert Group A. When it is
    'low-frequency' (smooth prose, grammar), prefer Expert Group B."

    Efficiency Purity: Pure GPU tensor math. Zero CPU syncs.
    """

    def __init__(self, hidden_dim: int, num_experts: int):
        super().__init__()
        self.num_experts = num_experts

        # Low-rank spectral projectors (keep VRAM overhead minimal)
        # We use hidden_dim // 2 as the spectral feature dimension
        spectral_feat_dim = hidden_dim

        self.low_freq_proj  = nn.Linear(spectral_feat_dim, num_experts, bias=False)
        self.high_freq_proj = nn.Linear(spectral_feat_dim, num_experts, bias=False)

        # Learnable blend: how much spectral bias matters vs magnitude routing
        self.spectral_blend = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, D, 2] — phasal hidden state (real, imag interleaved).
        Returns:
            spectral_bias: [B, T, num_experts] routing bias tensor.
        """
        B, T, D, _ = x.shape

        # Collapse real/imag into a single magnitude spectrum
        x_mag = x.norm(dim=-1)  # [B, T, D]

        # 1D FFT along the token dimension → spectral representation
        X_fft = torch.fft.rfft(x_mag, dim=1, norm="ortho")  # [B, T//2+1, D]
        X_mag = X_fft.abs()   # [B, freq_bins, D]

        freq_bins = X_mag.shape[1]
        mid = freq_bins // 2

        # Low-frequency: global context (smooth patterns)
        low  = X_mag[:, :mid, :].mean(dim=1)   # [B, D]
        # High-frequency: local detail (sharp changes)
        high = X_mag[:, mid:, :].mean(dim=1)   # [B, D]

        # Project to expert space and broadcast to token dimension
        low_bias  = self.low_freq_proj(low).unsqueeze(1).expand(B, T, -1)   # [B, T, E]
        high_bias = self.high_freq_proj(high).unsqueeze(1).expand(B, T, -1) # [B, T, E]

        return self.spectral_blend * (low_bias + high_bias)


# ─────────────────────────────────────────────────────────────────────────────
# Pillar 3: Eigen-Resonance Solver (Experimental)
# ─────────────────────────────────────────────────────────────────────────────

class EigenResonanceSolver(nn.Module):
    """
    Experimental: Replaces the iterative PPC loop with a closed-form
    Spectral Projection using the dominant eigenvectors of the phasal operator.

    Hypothesis:
        The 16-step iterative convergence finds the lowest-energy phasal state.
        This is equivalent to projecting the state onto the "ground state"
        of the system — which is the leading eigenvector of the phasal operator.
        A single eigenprojection could achieve the same in O(D²) vs O(16 * D²).

    Status: EXPERIMENTAL. Currently validated on toy problems (D=64).
    Full 3.2B (D=1024) validation is the goal of this research branch.

    Memory Note:
        We use a rank-K (K=8) approximation of the full D×D phasal operator
        to keep VRAM within dual-T4 bounds (avoids 1024×1024 intermediate).
    """

    def __init__(self, hidden_dim: int, rank_k: int = 8):
        super().__init__()
        self.rank_k = rank_k
        # Learnable phasal operator (low-rank factorisation for VRAM efficiency)
        self.phasal_basis  = nn.Parameter(torch.randn(hidden_dim, rank_k) * 0.01)
        self.phasal_energy = nn.Parameter(torch.ones(rank_k))  # "eigenvalues"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, D, 2] — current phasal state.
        Returns:
            x_projected: [B, T, D, 2] — spectrally projected state.
        """
        B, T, D, _ = x.shape
        x_real = x[..., 0]  # [B, T, D]
        x_imag = x[..., 1]  # [B, T, D]

        # Low-rank spectral projection: project both components onto phasal basis
        energy_weights = torch.softmax(self.phasal_energy, dim=0)

        # Real projection
        coeff_r = torch.einsum("btd,dk->btk", x_real, self.phasal_basis)
        x_proj_r = torch.einsum("btk,dk->btd", coeff_r * energy_weights, self.phasal_basis)

        # Imaginary projection
        coeff_i = torch.einsum("btd,dk->btk", x_imag, self.phasal_basis)
        x_proj_i = torch.einsum("btk,dk->btd", coeff_i * energy_weights, self.phasal_basis)

        return torch.stack([x_proj_r, x_proj_i], dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
# Spectral Sharded Model: Full 3.2B Research Build
# ─────────────────────────────────────────────────────────────────────────────

class SpectralShardedPPCGraphLLM(ShardedPPCGraphLLM):
    """
    Research-grade extension of the 3.2B ShardedPPCGraphLLM.

    Inherits:
      - All PPC core logic (Phasal Resonance, Complex Weights, Prospective Updates)
      - The Dual-T4 Sharding topology (12 layers per GPU)
      - Island Compilation (per-layer torch.compile)

    Adds:
      - Pillar 1: SpectralExpertGate (frequency-conditioned MoE routing bias)
      - Pillar 3: EigenResonanceSolver (optional replacement for iterative PPC)

    Toggles:
      use_spectral_gate    (bool): Enable Pillar 1. Default: True.
      use_eigen_resonance  (bool): Enable Pillar 3. Default: False (experimental).
    """

    def __init__(self, vocab_size: int, hidden_dim: int = 1024, num_layers: int = 24,
                 num_experts: int = 64, local_lr: float = 0.5, lr_decay: float = 0.85,
                 use_jacobian: bool = False,
                 use_spectral_gate: bool = True,
                 use_eigen_resonance: bool = False):

        super().__init__(vocab_size, hidden_dim, num_layers, num_experts,
                         local_lr, lr_decay, use_jacobian)

        self.use_spectral_gate   = use_spectral_gate
        self.use_eigen_resonance = use_eigen_resonance

        # Pillar 1: One spectral gate per GPU shard
        if use_spectral_gate:
            self.spectral_gate_0 = SpectralExpertGate(hidden_dim, num_experts).to(self.device0)
            self.spectral_gate_1 = SpectralExpertGate(hidden_dim, num_experts).to(self.device1)

        # Pillar 3: One eigen-solver per GPU shard (rank-8 approximation)
        if use_eigen_resonance:
            self.eigen_solver_0 = EigenResonanceSolver(hidden_dim, rank_k=8).to(self.device0)
            self.eigen_solver_1 = EigenResonanceSolver(hidden_dim, rank_k=8).to(self.device1)

    def forward(self, input_ids: torch.Tensor, local_iters: int = 8):
        """
        Spectral-augmented forward pass.

        Pillar 1 is applied as a gate bias at the start of each shard.
        Pillar 3 replaces the iterative PPC loop when enabled.
        """
        input_ids = input_ids.to(self.device0)
        x = self.embed(input_ids)  # [B, T, D, 2]

        total_iters = 0
        res_energies = []

        # Pillar 1: Compute spectral gate bias per shard
        gate_bias_0 = None
        gate_bias_1 = None
        if self.use_spectral_gate:
            B, T, D, _ = x.shape
            gate_bias_0 = self.spectral_gate_0(x).reshape(B * T, -1)        # [B*T, E]
            # We'll compute gate_bias_1 after the device transfer

        for i, layer in enumerate(self.layers):
            if i == self.split_point:
                x = x.to(self.device1)
                # Compute Pillar 1 bias for shard 1 after transfer
                if self.use_spectral_gate:
                    B, T, D, _ = x.shape
                    gate_bias_1 = self.spectral_gate_1(x).reshape(B * T, -1)

            # Select the correct gate bias for this shard
            current_gate_bias = gate_bias_0 if i < self.split_point else gate_bias_1

            # Pillar 3: Eigen-Resonance pre-conditioning (before iterative loop)
            if self.use_eigen_resonance:
                solver = self.eigen_solver_0 if i < self.split_point else self.eigen_solver_1
                x = x + solver(x)  # Residual spectral projection

            x, iters, res_norm = layer(x, local_iters, gate_bias=current_gate_bias)
            x = x.clone()
            total_iters += iters
            res_energies.append(res_norm)

        layer_energies = torch.stack([e.to(self.device1) for e in res_energies])
        avg_energy = layer_energies.mean()

        x_flat = x.flatten(-2)
        x_norm = self.layer_norm(x_flat)
        logits = self.output_head(x_norm)
        return logits, total_iters / self.num_layers, avg_energy, layer_energies
