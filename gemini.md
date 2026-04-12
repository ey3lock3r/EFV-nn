# EFV-nn: Gemini Protocol

## 1. Context & Tech
- **Focus:** EFV & SOTA ML research.
- **Stack:** Python, `uv`, `numpy`, `scikit-learn`, `matplotlib`.

## 2. Standards
- **Coding:** Modular, DRY, PEP 8, Type Hints, Robust Error Handling.
- **Design:** No hardware conflicts (T4 VRAM). Prioritize stable architecture over hacks.
- **Source of Truth:** `src/` is canonical. Notebook imports source from GitHub clone.
- **Testing:** `pytest` (AAA).
## 3. Protocol & Workflow
- **Loop:** `src/` (canonical) → **Pre-Flight Review** → **Execution** → **Log & Hygiene** → **Push** → **Notebook Pull**.
- **Pre-Flight:** Before any change, perform a **SINGLE** holistic check of math parity, logic, and engine constraints (`compile`/`autocast`).
- **High-Density Logging:** Log insights mid-flow using shorthand to maintain **Zero Information Loss** (ensure 100% architectural memory).
- **Hygiene:** Immediate cleanup of scratch/temp files. Maintain `task.md` as the living status.
- **No Duplication:** Never duplicate library code inside notebooks. Pull from master.

---

## 4. Learnings & Mistakes Diary
- **[2026-04-12] Advanced Cognitive Inference (Swarm):**
    - **Swarm Search (System 2):** Implemented parallel "Ghost State" exploration (N=8). Winners selected by lowest Phasal Resonance Energy (E). Fixed `IndexError` by aligning dimensional reduction in selection logic.
    - **CUDA Graph Overwrite (CRITICAL):** Identified `RuntimeError` during generation where Inductor's static CUDA Graph buffers were recycled before token-loop completion. **Fix:** Mandatory `.clone()` as state tensors exit each Island block.
    - **Signature Alignment:** Integrated `E (Energy)` return into all model methods (+forward/generate). Energy monitoring is now our primary "Internal Signal" for training stability.
    - **Global Model Pattern:** Transitioned notebook to decoupled architecture. 3.2B instantiation and checkpoint loading moved to a dedicated 'Setup' cell. Enables instant Train <-> Verify switching and live hot-patching without reloading VRAM.
    - **Inductor Optimization (Island only):** Removed redundant global `torch.compile(model)` in favor of class-internal per-layer compilation. Prevents cross-device graph breaks and reduces cold-start latency.
- **[2026-04-11] 3.2B + Inductor Optimization:**
    - **Vectorized MoE (Speed Fix):** Serial Python loop was the bottleneck. Migrated to **Batch-BMM** `[E, K, D]`.
    - **Zero-Break Fusion:** Identified **Early-Stopping Break** as a 384-sync/step bottleneck. Migrated to **Static-Length Loops** for 100% Inductor fusion. Result: Fused 16-step GPU burst >> 5-step synced sync. [Duration Delta: TBD ms/step]
    - **Precision Balancing (T4):** Enforced `f16` for BMM matmuls to engage **Tensor Cores** (8x speedup). Kept `f32` ONLY for `index_add_` (scatter stability).
    - **Calculus/NaN:** `GradScaler` incompatible with FP16 params. Switched to **Static Scaling** (`loss/256`). **ModReLU**: `safe_mag` ≥1e-8 for FP16 stability.
    - **Memory (View 2.0):** 64 experts fit T4 via Real-Pair `half` storage. VRAM-parity with complex-view but Inductor-optimized.
    - **Jacobian/Bridge:** `transpose_forward` weighted by `topk_scores`. DEQ Bridge used **Static `base_local_lr`** to solve vanishing signal ($10^{-28}$).
    - **Checkpointing:** Removed `autocast` from `checkpoint` (recompute fail). Autocast moved strictly INSIDE layers.
    - **MoE/Routing:** Gate uses `[real||imag]` concat (phase-aware). `k_nodes` dynamic ($B_T / E$).
    - **Kaggle Deployment:** Zero-install git-source protocol. manual `bitsandbytes`/`wandb` only.
- **[2026-04-10] PPC Optimization:**
    - **Mistake:** FP16 experts w/ 1e-4 LR stagnated. Reverted to FP32 (pre-View Trick).
    - **Learning:** `autocast` activation overhead savings are critical.
