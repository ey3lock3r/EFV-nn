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
- **Convergence Policy:** V3 uses an Automated Phasal Pilot (Phases 0 $\to$ 3). Phase transitions are dynamically triggered by rolling energy (`avg_e < target_e`), capping at 48 iterations (Phase 3: Crystallize).
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
- **[2026-04-21] PPC-GNN V3 Validation & System Hardening:**
    - **Axiom: ComplexGELU over ModReLU**: Activating real and imaginary parts independently (ComplexGELU) natively respects Wirtinger calculus, unlocking stable non-linear phase mixing without destroying the implicit backward pass.
    - **Axiom: DEQ Ghost-Cache Accumulation (The 25GB Leak)**: Caching MoE FP32 weights before a DEQ layer and clearing them in the backward pass causes massive VRAM leakage across layers during the forward pass (e.g., $12 \times 2.14GB = 25.7GB$). **Fix**: Pass `setup_fn` and `cleanup_fn` directly into `DEQFunction` to restrict MoE caching strictly to the active micro-iterations.
    - **Axiom: Triton Pythonic Type-Coercion Trap**: In Python 3, `pid / N` silently coerces to `float32`. Triton strictly throws `IncompatibleTypeErrorImpl` when this float touches memory pointers. **Fix**: Always use floor division `//` for Triton index grids.
    - **Axiom: PagedAdamW Multi-GPU Resurrection**: When loading a `bitsandbytes` optimizer from a CPU checkpoint (`map_location='cpu'`), native `load_state_dict()` leaves momentum tensors on the CPU. **Fix**: Explicitly loop through `opt.state` and map each tensor to its specific `param.device` to prevent multi-GPU crashes.
    - **Axiom: NVMe Memory-Mapped Checkpoints**: Loading a 6.4GB model via `torch.load()` causes a 6.4GB RAM spike. **Fix**: Using `mmap=True` streams the checkpoint directly from disk to GPU via PCIe, bypassing system RAM entirely.
    - **Mistake: Adam Momentum Disconnection**: Re-initializing the optimizer (`get_opt()`) during scheduled phase-transitions wipes all momentum/variance history, causing violent trajectory spikes. **Fix**: Traverse and update `pg['lr']` directly within the existing `param_groups`.
- **[2026-04-18] The 20,000-Step Milestone (Phase 6):**
    - **Axiom: Syntactic Mastery**: At 10M tokens (Step 20k), 3.2B PPC-GNN achieves perfect syntax (punctuation/preposition clustering) without semantic coherence. Loss floor breached at `7.39`.
    - **Axiom: Phasal Annealing**: To break the final 7.5 plateau, LR must drop to micro-scale (`1e-6`) to force gradients out of the "wandering valley" and into absolute factual minima.
    - **Axiom: OCNS Memory Hygiene**: `torch.roll` inside iterative loops (48 iters) triggers OOM via activation-buffer explosion. **Fix**: Use zero-copy Slicing (Views) for temporal shifts to maintain dual-T4 feasibility.
    - **Axiom: Triton Dual-Path**: Keep Python reference logic alongside Triton kernels for parity testing.
    - ~~**Axiom: APD Floor**: Adaptive Phasal Depth MUST have a hard `min_iters` floor~~ **[OUTDATED IN V3]**: Anderson Acceleration now mathematically guarantees stable convergence; early exits are exclusively dictated by exact adjoint tolerance (`res_norm < tol`), eliminating the need for arbitrary iteration floors.
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
- **[2026-04-20] Phase 6: Semantic Squeeze & Atomic Audits:**
    - **Axiom: The Semantic Squeeze**: 64 iters + 5e-6 LR "wakes up" experts, causing a linear energy climb (1.4e-5). Mandatory to break syntactic Word Salad.
    - **Axiom: Precision Cooling**: Dropping to 1e-6 LR + 0.0005 Thr allows the model to settle into semantic fixed points discovered during the Squeeze.
    - **Axiom: The "15.0 Iter" Signature**: An average of 15.0 iters (24-layer) signals a perfect 8/16 layer-wise phasal split; the Efficiency Frontier.
    - **Axiom: Pop & Purge Loader**: To load 7.5GB models in 30GB RAM, components must be "popped" and deleted from the state_dict immediately after extraction.
    - **Axiom: Atomic Audits**: Verification must use a tightened threshold (0.00001) and deeper floor (32 iters) to show the model's true phasal potential.
    - **Axiom: Inference "Safety Shield"**: Wrapping sandbox calls in `with torch.no_grad():` is critical to prevent gradient accumulation.
    - **Mistake: The Cell-Replace Trap**: Overwriting a notebook cell based on a `def func` match can delete other functions in that cell. **Protocol**: Verify cell boundaries before any Overwrite.
    - **Mistake: Indentation & Save Spikes**: Checkpoint saves must reset the timer (`t0`) inside the save-block to prevent disk-latency from leaking into performance metrics.
    - **Mistake: Missing Variable 'RESUME'**: Accidental deletion of cell-scope configuration variables during iterative updates is a high-frequency risk.
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
    - ~~**Axiom: Zero-Break Fusion**: Static-length loops (`local_iters`) only. Guarantees 100% Inductor fusion.~~ **[OUTDATED IN V3]**: We abandoned `torch.compile` (Inductor) due to CUDAGraphs crashing on the DEQ backward pass. Dynamic breaks via Anderson Acceleration (`res_norm < tol`) are now standard.
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
    - [DONE] Implement Energy-based Early Exit. Tokens with $E < \epsilon$ exit the loop early.
    - [DONE] Target: 15.0 iter average achieved (4x speedup).
- **Advanced Phasal Intelligence (Smashing the 7.0 Wall)**:
    - **Spectral Annealing**: Implement a phasal cooling schedule (e.g., $Thr: 0.0005 \to 0.00005$) over 100k steps to force semantic crystallization.
    - **Energy-Weighted Expert Choice**: Route tokens to specialized experts based on their Phasal Energy (e.g., "Stabilizer" experts for high-energy tokens).
    - **Adaptive Spectral Guardian**: Link the guardian penalty strength to the rolling loss average to balance phasal exploration vs. stability.
    - **OCNS Harmonic Priming**: Implement high-frequency learning rates for Phasal Gains to synchronize causal memory with logic wavefronts.
- **[DONE] Triton Kernel Fusion**:
    - Re-wrote Phase Rotation, OCNS Delays, and Anderson Mixing in raw Triton (`triton_kernels.py`) to permanently eliminate Inductor cold-start latency and CUDAGraphs memory corruption.
- **NF4 Quantized Experts**:
    - Move Experts to 4-bit NormalFloat (NF4) while keeping PPC core in FP32.
    - Target: Scale to 6.4B parameters on Dual-T4 hardware without OOM.

