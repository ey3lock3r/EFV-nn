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
- **Workspace Hygiene:** Immediate cleanup of temp/scratch. Update `task.md`.
- **Review:** Consult `gemini.md` & `task.md` before architecture shifts.
- **High-Density Logs:** Update **Diary** mid-flow using tech-shorthand. **Zero Information Loss.** No fluff.

---

## 4. Learnings & Mistakes Diary
- **[2026-04-11] 3.2B Architecture & Calculus Stabilization:**
    - **Calculus/NaN:** `GradScaler` fails on FP16 params. Replaced with **Manual Static Scaling** (`loss/256`, `clip/256`). Adam scale-invariance preserve updates. **ModReLU safety**: `safe_mag` must be ≥1e-8 (was 1e-12) to prevent FP16 div-by-zero.
    - **Jacobian Fix:** `transpose_forward` grad-pass now weights by `topk_scores` to reflect weighted forward gating importance.
    - **Vanishing Gradients:** DEQ bridge signal vanished ($10^{-28}$) over 24 layers due to decayed `current_lr`. Fixed via **Static `base_local_lr`** bridge.
    - **Memory (View Trick):** 64 expertos (3.2B) fit T4 via View Trick: `view_as_real(w).half()`. Saving: 6GB. Unpack JIT: `view_as_complex(w.float())` to avoid precision loss.
    - **Checkpointing:** Removed `autocast` from `checkpoint()` wrapper (does not propagate in `reentrant=False`). Autocast now strictly inside layers.
    - **MoE Routing:** Gate uses `[real||imag]` concat instead of `.abs()` to leverage phase info. `k_nodes` dynamic scaling: ($B_T / num\_experts$).
    - **Kaggle Deployment (Git-Source):** Zero-install protocol. `sys.path` injection from GitHub clone. Trusted system stack + manual missing packages (`bitsandbytes`, `wandb`).
- **[2026-04-10] PPC Optimization:**
    - **Mistake:** FP16 experts w/ 1e-4 LR stagnated. Reverted to FP32 (pre-View Trick).
    - **Learning:** `autocast` activation overhead savings are critical.
