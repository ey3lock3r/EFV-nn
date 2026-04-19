# 🌌 EFV-nn: 3.2B PPC-GNN Project Memory

## 🏗️ Section 1: Core Architectural Axioms
- **Focus:** EFV & SOTA ML research.
- **Project Structure:** Sharded 3.2B PPC-GNN on Dual-T4 architectures.
- **Stack:** Python, `uv`, `bitsandbytes` (8-bit), `wandb`, `torch.compile`.
- **Complex Domain:** All internal representations are Phasal Wavefronts (`torch.view_as_complex`).

## ⚙️ Section 2: Engineering & Technical Standards
- **Coding Standards:** Modular, DRY, PEP 8, Type Hints, Robust Error Handling.
- **Source of Truth**: `src/` is canonical. Notebook imports source from GitHub clone.
- **Testing:** `pytest` (AAA). Prioritize phasal identity tests.
- **Precision Policy:** FP32 Accumulation MANDATORY for phasal balance; BF16/FP16 for expert storage only.
- **Convergence Policy:** Phase 5 requires 48+ local iterations for deep semantic resonance.
- **Hardware Guardian:** Total VRAM usage must remain < 15.1GB per T4 to prevent swap-latency.

## 🚀 Section 3: Master Execution Protocol (MANDATORY)
1.  **Phase A: Strategic Synchronization**: 
    -   **Contextual Alignment**: Visualize phasal impact/roadmap before edits. Analyze project memory + `task.md`.
    -   **Strategic Mirroring**: Every discovery (Axiom) must be verbalized and immediately codified in Section 4.
2.  **Phase B: Progressive Implementation**: 
    -   **Foundation-First**: Modify core library (`src/`) before patching notebooks.
    -   **Cell Integrity**: Use unique cell headers (e.g., `# Cell 4`) for programmatic edits. No blind patches.
    -   **No Duplication**: Pull from master/library; never duplicate logic or modules inside notebooks.
3.  **Phase C: Safety & Verification**: 
    -   **6-Point Pre-Flight Check**: Mandatory verification before every model execution:
        1. **Axiomatic Load-Order**: Load `state_dict` BEFORE `torch.compile`.
        2. **Hook Safety**: No live model hooks without `try/finally` cleanup.
        3. **Timer Integrity**: Reset `t0` post-disk I/O (saves).
        4. **Metric Flushing**: Mandatory `wandb.finish()`.
        5. **Sampling Policy**: No `argmax`; use Top-K/Multinomial.
        6. **Patch Boundary**: Verify cell headers for notebook edits.
    -   **Numerical Audit**: Verify "Positive Delta" (Loss/Energy/Expert Diversity) before Pushes to Master.
4.  **Phase D: Axiomatic Closure**: 
    -   **Post-Flight Encoding**: Use **High-Density Encoding** (Technical Shorthand) for Section 4 logs. Ensure **Zero Information Loss** while minimizing token bloat.
    -   **Environment Hygiene**: Immediate cleanup of `scratch/*.py` and temporary files.
    -   **Project Persistence**: Ensure `git commit` describes specific numerical breakthroughs.

---

## 4. Learnings & Mistakes Diary (High-Density)
- **[2026-04-18] The 20,000-Step Milestone (Phase 6):**
    - **Axiom: Syntactic Mastery**: At 10M tokens (Step 20k), 3.2B PPC-GNN achieves perfect syntax (punctuation/preposition clustering) without semantic coherence. Loss floor breached at `7.39`.
    - **Axiom: Phasal Annealing**: To break the final 7.5 plateau, LR must drop to micro-scale (`1e-6`) to force gradients out of the "wandering valley" and into absolute factual minima.
    - **Axiom: OCNS Memory Hygiene**: `torch.roll` inside iterative loops (48 iters) triggers OOM via activation-buffer explosion. **Fix**: Use zero-copy Slicing (Views) for temporal shifts to maintain dual-T4 feasibility.
    - **Axiom: Triton Dual-Path**: Keep Python reference logic alongside Triton kernels for parity testing.
    - **Axiom: APD Floor**: Adaptive Phasal Depth MUST have a hard `min_iters` floor (e.g., 8) to prevent model collapse.
    - **Axiom: Triton 3.6 Strictness**: `tl.where` on scalars is illegal. Use boolean math or broadcasted block masks.
    - **Axiom: The In-Place Copy Trap**: Calling `.float()` or `.contiguous()` inside a Triton wrapper creates a **copy**. If the kernel is in-place, it updates the copy and leaves the original stale. Always cast to FP32 *before* the loop.
    - **Axiom: NaN-Siphon Principle**: In iterative models, use `tl.where(tl.isnan(step), 0.0, step)` inside the update kernel to prevent a single bad expert from poisoning the entire state.
    - **Axiom: Zero-Product NaN**: In Triton, `0.0 * inf` equals `NaN`. Never use multiplication as a mask for potentially infinite values; use strict `tl.where` gating.
    - **Axiom: The Self-Binding Trap**: Assigning a pure function to `self.func` makes it a **method**. Calling `self.func()` will inject `self` as the first argument. Always call external kernels as imported functions, never as instance attributes.
    - **Axiom: Memory Churn (The 1152 Staller)**: Allocating tensors (`torch.empty_like`) inside high-depth iterative loops (e.g., 24 layers * 48 iters) creates massive driver overhead. **Fix**: Use Persistent Buffers pre-allocated once per layer and passed as 'out' arguments to kernels.
    - **Axiom: The Structured Ghost**: Never use string-matching auto-patching for `.ipynb` files. Treating a notebook as a text file leads to cell collisions and accidental deletions. **Protocol**: Always use structured JSON reconstruction (manual or scripted) to rebuild the cell array, never regex/string find-and-replace.
    - **Axiom: The Stale Reference Trap**: `from module import function` binds the function at import time. `importlib.reload(module)` will NOT update existing references in other modules. **Fix**: Always use `from package import module` and call `module.function()` to ensure reloads are globally effective.
    - **Axiom: The Shape Juggling Tax**: Mismatched dimensions between Model (4D) and Kernels (3D) lead to cascading `ValueError`. **Fix**: Standardize all Triton wrappers to accept native Model shapes (`[B, T, D, 2]`) and handle internal flattening/unflattening automatically.
    - **Axiom: CUDAGraphs Exorcism**: `torch.compile(mode="reduce-overhead")` forces CUDAGraphs, which assumes a static memory pool. Switching between `no_grad` (DEQ iterations) and `grad` (Gradient Bridge) corrupts this pool (`curr_block->next == nullptr` crash). **Fix**: Rely solely on custom Triton kernels and manual FP16 contiguity optimizations; avoid `torch.compile` in the loop.
- **[2026-04-19] Phase 6 Hard Stabilization (The Native Pivot):**
    -   **Axiom: Weight Cache (The 147GB Copy)**: Slicing interleaved real parameters `[..., 2]` inside a tight loop (48 iters) forces a `.contiguous()` allocation every single call. At 3.2B params, this moves **147 GB** per step. **Fix**: Implement `cache_weights()` outside the loop to pre-align FP16 views and eliminate allocation overhead.
    -   **Axiom: APD Mathematics**: Early Exit logic using `torch.sum(residual**2)` is numerically unreachable due to sequence-length scaling. **Fix**: Use `torch.mean(residual**2) * 2` to align with the normalized `res_norm` metric, allowing early exit at iteration 8.
    -   **Axiom: Atomic Move (The 20GB Ceiling)**: High-safety saves (`shutil.copy` + `.tmp`) consume 3x model size (22.5GB). **Fix**: Use `shutil.move` for atomic rotation to keep disk usage at 2x (15GB), fitting within Kaggle's 20GB limit.
    -   **Axiom: Ghost-Blind Loading**: Checkpoints from compiled sessions contain `_orig_mod` prefixes. **Fix**: Use a dynamic state-dict mapper to strip prefixes during loading, ensuring native models can resume from compiled weights.
    -   **Axiom: Ghost Persistence**: `importlib.reload` cannot flush classes hot-patched by Dynamo. A **Full Kernel Restart** is mandatory to clear stale optimized references from Python's RAM.
- **[2026-04-15] Infrastructure & Diagnostic Stability:**
    - **Axiom: Hook Safety**: Mandatory `try/finally` for hooks on live models. Prevents OOM/Perf-leaks.
    - **Axiom: Timer Integrity**: Reset `t0` post-disk I/O (7.5GB saves) to stop metric inflation.
    - **Axiom: Metric Flushing**: Mandatory `wandb.finish()`. Prevents dashboard "ghosting."
    - **Axiom: Quantized Resonance**: Pulse-pattern ($0.0 \leftrightarrow 0.33$) in Ph-5 ($i=48$). Stable phasal breathing.
    - **Axiom: Expert Saturation**: Vectorized Expert Choice maintains 100.0% Diversity Score regardless of LR depth (verified at `5e-6`).
    - **Axiom: The Knowledge Wall**: At current dataset scale (7M tokens), 3.2B params hit a "Syntactic Plateau" at Loss ~7.5.
- **[2026-04-13] Architectural Stability & Precision Standards:**
    - **Axiom: ph-Scaling**: Transition Ph-1 (16i) -> Ph-2 (24i) -> Ph-3 (32i). Deepens equilibrium without divergence.
    - **Axiom: Precision Isolation**: Isolate recursive loops from global `autocast`. Prevents "Numerical Energy Drift" (E spikes).
    - **Axiom: Driver Contiguity**: Mandatory `.contiguous()` on parameter views before matmul. Prevents `cudaErrorIllegalAddress`.
    - **Axiom: Regularizer Linkage**: Layer regularizers (Spectral Guardian) must use differentiable Bridge outputs for MoE-Routing gradients.
    - **Axiom: Inductor Hygiene**: Mandatory `.clone()` for list-appended metrics in `torch.compile` to prevent buffer aliasing.
    - **Axiom: Metric Coverage**: Generation fixes: Replaced Repetitive Greedy with Top-K/Multinomial Sampling.
- **[2026-04-12] Cognitive Swarm & Phasal Diagnostics:**
    - **Axiom: Swarm Optimization**: Use low-energy resonance (E) as ghost selection metric. Noise (1e-4) for ghost diversity.
    - **Axiom: Island Isolation**: Explicit `.clone()` for tensors crossing Sharded compiled layer boundaries.
    - **Axiom: Telemetry Standard**: Log `avg_energy` as primary internal health signal.
- **[2026-04-11] 3.2B MoE & Fusion Engineering:**
    - **Axiom: MoE Throughput**: Use Batch-BMM `[E, K, D]` for expert passes. No serial loops.
    - **Axiom: Zero-Break Fusion**: Static-length loops (`local_iters`) only. Guarantees 100% Inductor fusion.
    - **Axiom: Stability Precision**: Weights/Matmuls = FP16; Accumulation/Norms/index_add = FP32.
    - **Axiom: Static Loss-Scaling**: Use `loss/256` for sharded FP16 pipelines. `GradScaler` incompatible.
- **[2026-04-10] PPC Precision Logic:**
    - **Expert Precision**: FP16 experts w/ 1e-4 LR stagnated. Reverted to FP32 (pre-View Trick) for stability.
    - **Autocast**: Activation savings critical; expert parameters must remain isolated.
- **[2026-04-18] OCNS Integration & Phasal Resonance:**
    - **Axiom: Delay Embedding**: Implemented 4-tap Prime delay `[1, 2, 3, 5]` in `PPCNodeLayer`.
    - **Axiom: Phasal Resonance**: Used Complex multiplication for delay gains. Spectral Guardian mitigates Energy Drift.
    - **Axiom: Zero-Impact Injection**: Initialized `delay_gains` to `0.0`. Resumes existing 3.2B runs safely via `strict=False`.

---

## 5. Scaling & Optimization Roadmap
- **Adaptive Phasal Depth (APD)**: 
    - Implement Energy-based Early Exit. Tokens with $E < \epsilon$ exit the loop early.
    - Target: 3x-5x speedup in inference/training throughput.
- **Triton Kernel Fusion**:
    - Rewrite `PPCNodeLayer` iterative loop in raw Triton. 
    - Target: Eliminate `torch.compile` cold-start latency and fuse MoE/OCNS into a single SRAM kernel.
- **NF4 Quantized Experts**:
    - Move Experts to 4-bit NormalFloat (NF4) while keeping PPC core in FP32.
    - Target: Scale to 6.4B parameters on Dual-T4 hardware without OOM.

### Phase 6: Hard Stabilization & Phasal Resonance (3.2B)
- **Mistake: The "Frozen Target" Trap**: Initially, the phasal target was computed once before the DEQ loop. This allowed the model to "cheat" by hitting a static goal instantly.
- **Fix: Moving Target Logic**: Moving target construction **INSIDE** the loop forced the model to chase a dynamic fixed point, finally unlocking the hidden semantic precision.
- **Learning: Global Mean Dilution**: Using `mean(residual)` in APD allowed easy tokens (80%) to "cloak" hard tokens (20%).
- **Fix: "Weakest Link" Policy**: Switching to `max(token_error)` ensures the model iterates until the **single hardest token** in the batch is satisfied.
- **The Atomic Limit**: Found that `0.000005` (5 micro-resonance) is the Phase 6 "Sweet Spot." Tightening to `0.000001` leads to "Infinite Thinking" (48 iters) with diminishing returns.
- **Live-Tune Protocol**: Implementing hyperparameter injection into training signatures allows for real-time precision tuning without kernel restarts.
