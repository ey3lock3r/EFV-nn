# 🚀 PPC-OCNS GNN V3 Optimization Roadmap

This document outlines the strategy for scaling the 3.2B parameter PPC-GNN architecture to achieve higher throughput on Dual-T4 hardware while maintaining phasal stability.

## 1. Immediate Throughput Wins (The "Low-Hanging Fruit")

### [ ] APD (Adaptive Phasal Depth) Relaxation
*   **Problem**: In early Phase 0, high energy prevents early exit, forcing the maximum number of iterations (e.g., 24) on every layer.
*   **Fix**: Implement "Dynamic Tolerance." Set the exit threshold to `max(1e-5, rolling_energy * 0.1)`. 
*   **Target**: Reduce average iterations from 24.0 to ~8.0 in Phase 0.
*   **Impact**: **~3x Speedup** in early training.

### [ ] Zero-G Guard Caching
*   **Problem**: `next(layer.parameters())` and string comparisons on every iteration add significant overhead.
*   **Fix**: Cache the target `torch.device` object and use integer index comparisons.
*   **Target**: Zero overhead for sharding safety checks.

## 2. Mathematical Efficiency (The Adjoint Bridge)

### [ ] Adjoint Solver Warm-Starting
*   **Problem**: The IFT backward pass starts the adjoint solver from scratch every step.
*   **Fix**: Cache the adjoint state `g` from step $t$ and use it as the initial guess for step $t+1$.
*   **Target**: Reduce backward iterations from 8 to 3-4.
*   **Impact**: **~20% reduction** in total step time.

### [ ] Half-Precision Adjoint Search
*   **Problem**: The adjoint solver currently runs in FP32 to ensure stability.
*   **Fix**: Run the Anderson search in FP16/BF16 and only cast to FP32 for the final Jacobian-Vector Product (JVP).

## 3. Triton & Kernel Fusion (The "Hyper-Drive" Layer)

### [ ] Fused MoE Dispatch & Delay
*   **Problem**: `ocns_delay` and MoE `gather` are separate memory passes.
*   **Fix**: Write a Triton kernel that fuses history lookups (delays) directly into the expert token gathering process.
*   **Impact**: Massive reduction in VRAM memory bandwidth usage.

### [ ] Fused Spectral Gate
*   **Problem**: FFT, mean, and linear projection are separate operations.
*   **Fix**: Fuse the post-FFT magnitude calculation and frequency bin averaging into a single Triton kernel.

## 4. Scaling Horizon

### [ ] NF4 Expert Quantization
*   **Goal**: Move expert weights to 4-bit NormalFloat (NF4).
*   **Why**: This will free up ~12GB of VRAM, allowing us to double the number of layers (to 48) or double the hidden dimension (to 2048) on the same hardware.

---
> [!NOTE]
> We will proceed with these optimizations once the current Phase 0 run has achieved a stable Phasal Lock ($E < 1.0$).
