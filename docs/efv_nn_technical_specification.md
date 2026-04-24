# EFV-nn Technical Specification (PPC-OCNS GNN V3)

## 1. System Overview
EFV-nn (Energy-Based Phasal Variational Neural Network) implements a state-of-the-art Parallel Phasal Computing (PPC) Graph Neural Network. The current architecture, V3, is a 3.2B parameter model optimized for sharded training on Dual-T4 hardware. The core design principle involves maintaining all internal token representations as Phasal Wavefronts in the complex domain.

## 2. Core Architectural Pillars

### 2.1 Complex Domain Representation
All internal token representations are structured as "interleaved real" pairs, effectively representing complex numbers with the shape `[..., hidden_dim, 2]`. This dual representation allows the network to model signal phases and magnitudes independently.
- **ComplexKaimingInitializer**: Provides memory-efficient, in-place initialization of interleaved-real parameters with non-negative magnitudes (normal distribution) and uniform phases `(-π, π]`.
- **ComplexGELU Activation**: Applies the GELU function to real and imaginary components independently. This allows structural cross-talk via expert routing weights while preserving holomorphism, ensuring stable Wirtinger calculus gradients for the implicit backward pass.

### 2.2 Deep Equilibrium (DEQ) Processing
Instead of unrolling standard sequential layers, PPC nodes iteratively update the hidden state until it converges to a fixed-point equilibrium (`x = f(x)`).
- **Anderson Acceleration**: A stabilized fixed-point solver utilizing history buffer recycling (`m=5`) to optimize convergence. It computes a mixing vector `alpha` to combine historical states, accelerating the path to equilibrium.
- **Implicit Function Theorem (IFT) Backward Pass**: Replaces standard autograd loop unrolling with an exact gradient flow calculation using the Adjoint Method. This decouples memory cost from the iteration depth of the forward pass.
- **Adjoint Warm-Starting**: The solver caches the adjoint state from the previous backward step to warm-start the current backward pass, reducing the required backward iterations by ~50%.
- **Adaptive Phasal Depth (APD)**: Micro-iterations exit early based on a dynamically calculated tolerance `res_norm < tol`. The base tolerance adapts based on the `rolling_energy`, allowing relaxed exits during high-energy initial phases (Phase 0).

### 2.3 Expert Choice MoE Matcher
Transformations are handled by a Mixture of Experts (MoE) layer configured for high throughput and expert utilization.
- **Dynamic Routing**: Tokens are routed to `num_experts` (default 64) using `ExpertChoiceMoEMatcher`. The routing determines the top-k experts per token.
- **Precision Isolation**: Expert weights are stored in **FP16** to allow full gradient updates while managing VRAM on Dual-T4s. However, to maintain phasal balance, accumulations, norms, and DEQ solvers strictly operate in **FP32**.
- **Ghost-Cache Accumulation Fix**: The MoE pre-casts and caches FP32 weights just before the active DEQ micro-iterations. `setup_fn` and `cleanup_fn` hooks in the `DEQFunction` strictly flush this cache immediately after the loop, preventing massive VRAM leakage across layers (the "25GB leak").

### 2.4 OCNS (Oscillatory Causal Neural States)
Temporal causal context is injected via phase-rotated history states.
- **Phasal Delay Embedding**: Historical tokens at specific delays (e.g., `prime_delays=[1, 2, 3, 5]`) are added directly into the current state via complex interference. The delays apply learnable complex gains.
- **Memory Hygiene**: Utilizes zero-copy tensor slicing (Views) instead of `torch.roll` inside the DEQ loops, preventing activation-buffer explosion.

### 2.5 Spectral Guardian & Expert Gate
- **Spectral Guardian**: A Laplacian regularization penalty (Pillar 2) applied to the layer energies (`λ · Σ (E_i - E_{i+1})²`). It softly penalizes high-frequency 'jitter' or divergence between adjacent layer energies to maintain phasal stability.
- **SpectralExpertGate**: Performs FFT on the input magnitude to extract low and high-frequency components. These components are projected to generate a spectral routing bias for the MoE, controlled by a learnable `spectral_blend`.

### 2.6 Sharded Architecture
The model is designed from the ground up for pipeline parallelism across Dual-T4 GPUs.
- **ShardedPPCGraphLLM**: Splits the model exactly at `num_layers // 2`. Embeddings and the first half of the layers reside on `cuda:0`, while the second half and the output head reside on `cuda:1`.
- **Zero-G Guard**: Optimized device object caching and strict device audits inside the forward pass to prevent implicit "CPU vs CUDA" overhead and CUDA Graph corruption.
- **Cognitive Swarm Inference**: For generation, the model expands the input into a `swarm_size`, injects tiny Gaussian noise into the phase components (Phase Perturbation), and evaluates parallel ghost-states. It selects the state with the deepest convergence (lowest phasal energy) for token decoding.

### 2.7 Hyper-Drive Triton Acceleration
To prevent compiler drift and `torch.compile` CUDAGraph memory corruption in the DEQ loops, core operations are offloaded to custom, in-place Triton kernels:
- `fused_phase_rotation`: Target state construction with complex phase rotation.
- `fused_ocns_delay`: Historical token state injection.
- `fused_state_update`: Performs the contractive residual step with strict clamping.
- `fused_normalize_activate`: Handles atomic counts and the `_gelu_fast` activation.
- `anderson_mixing`: Applies the Anderson history-weighted sum.
- `fused_moe_dispatch_delay` & `fused_moe_aggregator`: Fuses the MoE gather, compute, and scatter operations.
- `fused_spectral_gate`: Fused FFT magnitude calculation and frequency pooling.

## 3. Mathematical Stability Protocols
Due to the iterative and recursive nature of the DEQ solver, V3 implements strict numerical hardening:
- **Contractive Residual Clipping**: The state update step (`base_local_lr * (target - pred)`) is clamped strictly between `[-10.0, 10.0]` to prevent divergence spirals.
- **Anderson Alpha Clipping**: Alpha values generated by the Anderson solver are clamped to `[-1.0, 1.0]` to prevent extreme state-space jumps.
- **NaN-Siphon Principle**: Across Triton kernels and PyTorch solvers, any `NaN` values encountered during micro-iterations are immediately replaced with `0.0`. This isolates numerical failure, preventing a single diverging expert or token from poisoning the entire tensor.
- **Surgical Sigmoid Fix**: Triton activations use a stable polynomial approximation of Sigmoid/GELU to ensure consistency across Triton compiler versions.

## 4. Training Pipeline
- **Memory-Hardened Optimizers**: Employs `bitsandbytes` `PagedAdamW8bit` to allow the massive FP32 master weight states to spill to CPU RAM, avoiding OOM on 16GB T4s.
- **Automated Phasal Pilot**: Autonomous learning rate and phase progression (Phases 0 $\to$ 3) triggered dynamically by comparing the rolling layer `avg_energy` against targets.
- **Persistent Storage**: Utilizes `mmap=True` for memory-mapped NVMe checkpoints, loading the 6.4GB model directly via PCIe and bypassing system RAM spikes. Checkpoint rotations use `shutil.move` for atomic 2x disk limits.
