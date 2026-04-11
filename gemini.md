# EFV-nn: Gemini Protocol

## 1. Context & Tech
- **Focus:** EFV & SOTA ML research.
- **Stack:** Python, `uv`, `numpy`, `scikit-learn`, `matplotlib`.

## 2. Standards
- **Coding:** Modular, DRY, PEP 8, Type Hints, Robust Error Handling.
- **Design:** No hardware conflicts (T4 VRAM). Prioritize stable architecture over hacks.
- **Source of Truth:** `src/` is canonical. Notebook imports source from GitHub clone.
- **Workflow:** Mod `src/` → Push → Notebook pulls. No inline duplication.
- **Testing:** `pytest` (AAA).

## 3. Directives
- **Workspace Hygiene:** Immediate temp/scratch cleanup. Update `task.md`.
- **Review:** Consult `gemini.md` & `task.md` PRIOR to architecture shifts.
- **Zero-Loss Logging:** Log insights **mid-flow** (during changes) using high-density shorthand (**Zero Information Loss**) to ensure 100% architectural memory.
- **Holistic Strategy:** Every fix MUST consider math parity, algorithm logic, and engine constraints (`compile`/`autocast`) holistically.
- **Iterative Precision (f32):** Non-linear/iterative loops (PPC) MUST enforce **f32** internally to block `autocast` erosion.

---

## 4. Learnings & Mistakes Diary
- **[2026-04-11] 3.2B + Inductor Optimization:**
    - **Vectorized MoE (Speed Fix):** Serial Python loop was the bottleneck (64-expert serial lag). Migrated to **Batch-BMM** `[E, K, D]`. Result: Final parallelization of experts, enabling full `reduce-overhead` fusion.
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
