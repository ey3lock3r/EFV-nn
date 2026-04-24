# EFV-nn: 3.2B PPC-GNN — Technical Specification

_Inferred from source as of 2026-04-22. Canonical references: `src/efv_nn/`._

---

## 0. System Overview

**EFV-nn** (*Energy-Based Phasal Variational Neural Network*) is a 3.2B-parameter research LLM built on the Parallel Phasal Computing (PPC) paradigm. All internal token representations are maintained as *Phasal Wavefronts* — complex-domain vectors in interleaved-real format `[..., D, 2]`. The core design principle is that language understanding emerges from the iterative convergence of these wavefronts to a fixed-point equilibrium (`x* = f(x*)`), rather than from a sequence of independent forward passes.

V3 replaces earlier `torch.compile`/Inductor-based fusion with a custom Triton kernel suite and an IFT gradient bridge, enabling stable training on Dual-T4 hardware within a 15.1 GB per-device VRAM budget.

**Design Pillars (V3):**

| Pillar | Name | Description |
|---|---|---|
| 1 | Spectral Expert Gate | FFT-based low/high-frequency routing bias for MoE |
| 2 | Spectral Guardian | Laplacian energy regularization between adjacent layers |
| 3 | Hybrid Autograd Bridge | Triton forward under `no_grad`; PyTorch re-entry under `enable_grad` for IFT |
| 4 | Autocast Isolation | DEQ loop forced to FP32; prevents numerical energy drift |
| 5 | Dynamic Expert Routing | Expert Choice MoE with OCNS delay fused into dispatch |
| 6 | Contractive Residual | State update clamped to `±10` to prevent divergence spirals |
| 7 | Hyper-Drive Triton | 8 custom kernels eliminating intermediate allocations and CUDAGraph hazards |

---

## 1. Scale & Parameter Budget

| Component | Shape | Parameters |
|---|---|---|
| Embedding | `[128256, 1024, 2]` | ~262M |
| Expert weights (per layer) | `[64, 1024, 1024, 2]` FP16 | ~134M |
| Expert weights (24 layers) | — | ~3.22B |
| Gate weights (per layer) | `[2048, 64]` | 131K |
| Spectral gate (per layer) | `2 × [1024, 64]` | 131K |
| OCNS delay gains (per layer) | `[8, 1024, 2]` | 16K |
| Phase buffers cos/sin (per layer) | `[1024]` × 2 (non-param) | — |
| Exit threshold (per layer) | scalar | 24 |
| Output head | `[2048, 128256]` | ~262M |
| Layer norm | `[2048]` | 4K |
| **Total (approx)** | | **~3.75B** |

---

## 2. Phasal Representation

All internal state is **interleaved real** format: `[..., D, 2]` where `[..., 0]` is the real component and `[..., 1]` is the imaginary component. Native `torch.complex` is avoided to prevent `autocast` incompatibilities and maintain full Wirtinger calculus gradient semantics.

**Initialization (`ComplexKaimingInitializer`):**
- Magnitude: `|N(0, 1/√fan_in)|`
- Phase: `Uniform(0, 2π)`
- Output: `(mag·cos(φ), mag·sin(φ))`

**Causal Target Construction:**  
Each token's fixed-point target is the previous token's state phase-rotated by a per-dimension angle `(cos_p, sin_p)`. This encodes autoregressive structure as a geometric constraint rather than attention:
```
target[t] = rotate(x[t-1], cos_p, sin_p)
target[0] = x[0]  (identity)
```

---

## 3. Precision Policy

| Component | Dtype | Enforcement |
|---|---|---|
| DEQ loop state | FP32 | `autocast(enabled=False)` guard in `PPCNodeLayer.forward` |
| Expert weights | FP16 | `init_w.half()`, stored as `nn.Parameter` |
| BMM accumulation | FP32 | Explicit `.float()` cast before `torch.matmul` |
| Gate weights | FP32 | Default `dtype=torch.float32` |
| Optimizer master states | FP32 spill to CPU RAM | `PagedAdamW8bit` |
| Checkpoint storage | FP16 | `v.cpu().half()` on every save |

---

## 4. DEQ Solver (Anderson Acceleration)

Solves the fixed-point equation `x* = f(x*)` via Anderson Acceleration.

**Solver parameters:**

| Parameter | Value |
|---|---|
| History buffer `m` | 5 |
| Regularization `λ` | `1e-2` for `k < 5`, `1e-4` thereafter |
| Alpha clipping | `±1.0` |
| State update clamp | `±10.0` |
| Linear solve backend | `torch.linalg.solve` in FP32 |
| Max iterations | Phase-controlled: 12 / 20 / 32 / 48 |
| Convergence tolerance | Learnable `exit_threshold` (init `1e-6`) |
| Dynamic tolerance floor | `max(exit_threshold², (rolling_E × 0.05)²)` |

**Fixed-point function `f`:**
```
x_{k+1} = x_k + clamp(local_lr × (target − MoE(OCNS(x_k))), ±10)
```

**Backward (IFT / Implicit Function Theorem):**  
`DEQFunction` custom autograd. Adjoint vector `g` is solved via a nested Anderson pass (m=5, max_iter=12) with dynamic tolerance `max(1e-5, ‖residual₀‖ × 0.1)`. Warm-started from persistent `_adjoint_cache` buffer, reducing backward iterations by ~50%.

```
grad_out → solve (I − J^T)g = grad_out → grad_params = ∂f/∂θ · g
```

---

## 4b. Ghost-Cache Accumulation Fix (The 25 GB Leak)

Expert weights are FP16 but the BMM must accumulate in FP32. Naively casting FP16→FP32 once per layer and caching across the DEQ micro-iterations causes a per-layer VRAM cost that compounds across all 24 layers:

```
12 layers × ~2.14 GB FP32 cache = ~25.7 GB — exceeds Dual-T4 budget entirely
```

**Fix:** `DEQFunction` accepts `setup_fn` / `cleanup_fn` callbacks. `ExpertChoiceMoEMatcher.cache_weights()` casts and caches FP32 weights; `clear_cache()` deallocates them. The `DEQFunction` calls `setup_fn` immediately before the Anderson loop and `cleanup_fn` immediately after, so the FP32 cache exists only for the active micro-iterations of one layer, not across the entire forward pass.

---

## 5. Mixture of Experts

**Routing type:** Expert Choice — each expert selects its top-k tokens (not token-to-expert routing).

- Scores: `x.reshape(B×T, 2D) @ gate_weights` → `[B×T, 64]`
- Selection: `topk(k=2, dim=0)` → each of 64 experts picks 2 tokens → 128 active pairs
- Plus `spectral_blend × (low_bias + high_bias)` additive routing bias

**Expert computation (vectorized complex BMM):**
```
yr = xr @ Wr − xi @ Wi
yi = xr @ Wi + xi @ Wr
```
Where `Wr, Wi` are the real and imaginary weight slices, computed in FP32 from cached FP16 params.

**Aggregation:** Atomic scatter-add → count normalization → ComplexGELU activation.

**ComplexGELU:** `GELU(x_real + bias)` and `GELU(x_imag + bias)` independently, preserving phasal structure while allowing non-linear mixing through shared routing weights.

---

## 5b. Spectral Guardian (Pillar 2)

A Laplacian regularization penalty applied to the per-layer DEQ residual energies:

```
L_spectral = λ · Σᵢ (Eᵢ − Eᵢ₊₁)²
```

- **Purpose:** Softly penalizes high-frequency "jitter" between adjacent layer convergence depths, encouraging smooth energy descent across the network.
- **Strength:** `λ = 0.01` (gentle — prevents divergence without over-constraining).
- **Implementation:** `spectral_guardian_penalty(layer_energies, lam=0.01)` in `ppc_gnn.py`. Added to the training loss alongside the CE loss.
- **Gradient path:** `layer_energies[i]` is the `res_norm` returned by each `DEQFunction.apply()`, which flows gradients back through the IFT bridge.

---

## 6. Spectral Expert Gate (Pillar 1)

Computes a per-token routing bias from global sequence spectral statistics:

1. Token magnitudes: `‖x‖₂` over last dim → `[B, T, D]`
2. `rfft` along T → `[B, T/2+1, D]` (complex)
3. Split at `T/4`: low-frequency mean → `[B, D]`, high-frequency mean → `[B, D]`
4. Project each to `[B, E]` via separate linear layers
5. Scale by learnable `spectral_blend` (init 0.0 = no influence at start)
6. Broadcast to `[B×T, E]` as additive routing bias

Fused in Triton (Kernel 8) on GPU: magnitude + spectral pooling in one pass.

---

## 7. OCNS (One Core Neuron System)

**Purpose:** Inject causal temporal memory by superimposing delayed token states onto the current token before expert dispatch.

**Delays:** 8 prime/Fibonacci taps — `[1, 2, 3, 5, 7, 11, 13, 17]`  
**Gains:** `[8, D, 2]` learnable parameters, init to **zero** (safe for resuming existing checkpoints)

**Operation per delay τᵢ:**
```
x_eff[t] += delay_gains[i] ⊗ x[t − τᵢ]   (complex multiplication)
= (gr·dr − gi·di,  gr·di + gi·dr)
```

**Implementation:**
- Forward (no_grad): Fused in Triton Kernels 2 and 6 (zero intermediate allocations)
- Forward (grad-enabled): PyTorch view slicing, no `.roll()` copy

---

## 7b. Surgical Sigmoid Fix

Triton compiler versions disagree on the behaviour of `tl.tanh`. To guarantee identical numerical output across compiler versions and avoid `NaN` from large inputs:

```python
# Non-portable:
_gelu_fast(x) = 0.5 * x * (1 + tl.tanh(0.7978845 * (x + 0.044715 * x³)))

# V3 portable form (Surgical Sigmoid Fix):
_gelu_fast(x) = 0.5 * x * (2 * tl.sigmoid(1.59576912 * (x + 0.044715 * x³)))
# Uses the identity: tanh(z) = 2·sigmoid(2z) − 1
```

This is applied in `_normalize_activate_kernel` (Kernel 4) and any other kernel requiring GELU.

---

## 8. Triton Kernel Suite

| Kernel | Name | Operation | Grid |
|---|---|---|---|
| 1 | `_phase_rotation_kernel` | Causal phase-rotation target | `[B×T]` |
| 2 | `_ocns_delay_kernel` | OCNS delay embed, ≤8 taps, statically unrolled | `[B×T]` |
| 3 | `_state_update_kernel` | In-place: `x += lr·clamp(step, ±10)` + NaN siphon | `[N/1024]` |
| 4 | `_normalize_activate_kernel` | Divide by count + ComplexGELU fused | `[B×T]` |
| 5 | `_anderson_mixing_kernel` | History-weighted Anderson mixing | `[B×ceil(N/1024)]` |
| 6 | `_moe_dispatch_delay_kernel` | Token gather + OCNS delay, fused | `[E×K]` |
| 7 | `_moe_aggregator_kernel` | Atomic scatter-add + normalize_activate | `[E×K]` |
| 8 | `_spectral_gate_pool_kernel` | FFT mag low/high split pool | `[B, D]` |

**Block size:** `triton.next_power_of_2(D)` = 1024 for D=1024.  
**Index arithmetic:** All pointer indices use `.to(tl.int64)` to prevent 32-bit overflow at 3.2B parameter scale.

---

## 9. Hardware Sharding (Dual T4)

```
cuda:0 │ embedding + layers[0..11]
       ├──────── pipeline transfer at split_point=12 ────────
cuda:1 │ layers[12..23] + layer_norm + output_head
```

- Device objects cached as `self.d0 / self.d1` (`torch.device`) — used in hot loop
- Zero-G Guard: `if x.device != target_device: x = x.to(target_device)` with device object comparison
- Activation isolation: `.clone()` after each layer prevents cross-GPU buffer aliasing

**VRAM budget per T4:**

| Item | Memory |
|---|---|
| Expert weights (12 layers, FP16) | ~3.22 GB |
| Activations `[2, 256, 1024, 2]` FP32 | ~4 MB |
| Anderson buffers `[B, m, N]` | ~40 MB |
| Adjoint cache | same as activation |
| Optimizer (spilled to CPU) | ~0 GB GPU |
| **Total** | **~4–5 GB active, <15.1 GB limit** |

---

## 10. Automated Training Pilot

### Phase Schedule

| Phase | Name | LR (base) | LR (spectral) | Max Iters | local_lr | Target E |
|---|---|---|---|---|---|---|
| 0 | Init | 2e-4 | 2e-3 | 12 | 0.05 | 100.0 |
| 1 | Accel | 1.5e-4 | 1.5e-3 | 20 | 0.15 | 10.0 |
| 2 | DeepConv | 1e-4 | 1e-3 | 32 | 0.25 | 1.0 |
| 3 | Crystallize | 5e-5 | 5e-4 | 48 | 0.10 | 0.1 |

**Promotion:** Rolling 50-step `avg_energy < phase.target_e`. Updates `local_lr` and per-group LR in-place on existing `param_groups` — no optimizer re-init (avoids Adam momentum wipe).

### Surgical Matrix (auto-applied at `len(rolling_e) > 50`)

| Scenario | Trigger | Action |
|---|---|---|
| Divergence Spiral | `e_trend > 1.5` OR `avg_e > 5000` after step 100 | LR × 0.5, reset rolling history |
| Noise Wall | `avg_e > 3000` AND `l_delta < 0.001` after step 1000 | LR × 1.5 |
| Syntactic Plateau | `l_delta < 0.005` AND `avg_e < 200` after step 500 | `spectral_blend += 0.05` per layer |

**`e_trend`:** `mean(last 10) / mean(steps -50 to -40)` — detects acceleration in energy.  
**`l_delta`:** `|mean(last 50) − mean(steps -100 to -50)|` — detects loss stagnation.

---

## 11. Memory Management Techniques

| Technique | Location | Effect |
|---|---|---|
| `mmap=True` checkpoint load | Cell 5 | Streams from NVMe via PCIe, bypasses RAM spike |
| `shutil.move` atomic rotation | `save_checkpoint` | 2× peak disk usage vs 3× for copy |
| `expandable_segments:True` | `PYTORCH_ALLOC_CONF` in `training.py` | Reduces fragmentation |
| MoE FP32 cache scoped to DEQ | `setup_fn/cleanup_fn` in `DEQFunction` | Prevents 25 GB cross-layer leak |
| Persistent MoE scatter buffers | `_ensure_scatter_bufs` | Eliminates per-forward `zeros()` calls |
| OCNS via view slicing | `_apply_ocns_delays` | No copy of activation tensor |
| Anderson pre-allocated buffers | `eye_m`, `F_ordered_buf`, `x_next_flat` | Eliminates per-iteration alloc |
| Pre-allocated layer_energies | `torch.empty(num_layers)` + index write | Replaces list + torch.stack |
| Pre-allocated generation buffer | `torch.empty(B, T0 + max_new_tokens)` + fill | Eliminates O(n²) `torch.cat` growth |
| Ghost-blind loader | `strip _orig_mod prefix` on load | Handles compiled checkpoint resume |

---

## 12. Gradient Flow Architecture

```
INPUT IDs
    ↓
Embedding [V, D×2] → reshape → [B, T, D, 2]
    ↓
┌──────────────────────────────────────────────────────────┐
│  PPCNodeLayer × 24                                       │
│                                                          │
│  Forward (no_grad, Triton):                              │
│    SpectralGate(x) → gate_bias [B×T, E]                 │
│    PhaseRotation(x) → x_target [B, T, D, 2]             │
│    DEQFunction.forward:                                  │
│      Anderson(f, x0, m=5) → x* (fixed point)            │
│      f(x) = x + clip(lr × (target − MoE(OCNS(x))), ±10) │
│                                                          │
│  Backward (enable_grad, PyTorch):                        │
│    Recompute z_next = f_forward(z*, x_in)               │
│    Adjoint Anderson(m=5, max_iter=12) → g               │
│    grad_params = autograd.grad(z_next, params, g)        │
└──────────────────────────────────────────────────────────┘
    ↓
LayerNorm([B, T, D×2])
    ↓
Linear(D×2 → V) → logits [B, T, V]
    ↓
CrossEntropyLoss
```

**Key invariant:** Triton kernels run exclusively under `torch.no_grad()`. The PyTorch path re-enters `enable_grad()` inside `DEQFunction.backward` for one-step Jacobian recomputation. This hybrid avoids CUDAGraph pool corruption while retaining full gradient connectivity through the IFT bridge.

**Why `torch.compile` was abandoned:** `torch.compile(mode="reduce-overhead")` forces CUDAGraphs, which assumes a static memory pool. The DEQ pattern switches between `torch.no_grad()` (Anderson iterations) and `torch.enable_grad()` (IFT backward recomputation), causing pool corruption (`curr_block->next == nullptr` crash). The Triton kernel suite replaces Inductor's fusion benefits without CUDAGraph constraints.

---

## 13. Inference Modes

### Standard `generate` (Autoregressive)
- Pre-allocated output buffer `[B, T0 + max_new_tokens]`
- Top-K multinomial sampling (no argmax)
- Temperature scaling: `logits / max(1e-6, temperature)`
- EOS token: 128001 (LLaMA-3)

### Cognitive Swarm Inference (`generate_swarm`)

*Cognitive Swarm Inference* explores the phasal equilibrium landscape by evaluating multiple perturbed ghost copies of the input in parallel, then selecting the most stable state for token decoding:

1. Expand input to `swarm_size` ghost copies via `repeat_interleave` → `[B×swarm_size, T, D, 2]`
2. Inject Phase Perturbation: `N(0, 1e-4)` noise added to imaginary components only (real components unchanged to preserve semantic structure)
3. Run all ghosts through the full 24-layer forward pass under `no_grad`
4. Select winner per batch item: `argmin(‖imag_component‖_F)` — lowest imaginary energy signals deepest phasal convergence and most stable equilibrium
5. Vectorized gather: `curr_x[arange(B), winner_indices]`

This differs from ensemble methods in that the selection criterion (phasal energy) is intrinsic to the model's internal representation rather than an external score.

---

## 14. Observable Health Signals

| Signal | Normal Range | Meaning |
|---|---|---|
| `avg_energy` (E) | Phase 0: ~10–100 → Phase 3: <0.1 | Residual of DEQ fixed-point |
| `avg_iters` (I) | ~15.0 at efficiency frontier | Average Anderson iterations per layer |
| `spectral_blend` (B) | 0.0 → grows slowly | Spectral gate influence; 8-decimal precision logged |
| `loss` | ~7.5 syntactic plateau → target <7.0 | Standard CE loss |
| Expert diversity | >80% = healthy | Fraction of unique tokens selected across experts |
| "15.0 iter signature" | exactly 15.0 avg | Indicates 8/16 layer phasal split — efficiency frontier |
