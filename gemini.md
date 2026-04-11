# EFV-nn: Gemini Protocol

## 1. Context & Tech
- **Focus:** EFV & SOTA ML research.
- **Stack:** Python, `uv`, `numpy`, `scikit-learn`, `matplotlib`.

## 2. Standards
- **Coding:** Modular, DRY, PEP 8, Type Hints, Robust Error Handling.
- **Design:** No hardware conflicts (T4 VRAM). Prioritize stable architecture over hacks.
- **Source of Truth:** `src/` is canonical. Notebook installs via `uv` from GitHub.
- **Workflow:** Mod `src/` → Push → Notebook re-installs. No inline duplication.
- **Testing:** `pytest` (AAA).

## 3. Directives
- **Workspace Hygiene:** Immediate cleanup of temp/scratch. Update `task.md`.
- **Review:** Consult `gemini.md` & `task.md` before architecture shifts.
- **High-Density Logs:** Update **Diary** mid-flow using tech-shorthand. No fluff.

---

## 4. Learnings & Mistakes Diary
- **[2026-04-11] Holistic Architecture & Kaggle Stabilization:**
    - **VRAM/FP16:** 64 experts fit T4 via View Trick (`view_as_real(...).half()`). Unpack JIT: `view_as_complex(w.float())`. Saves 6GB.
    - **NaN/Scaling:** `GradScaler` incompatible w/ FP16 params. Solution: **Manual Static Scaling** (`loss/256`, `clip/256`). Adam scale-invariance ensures identical updates.
    - **Vanishing Gradients:** DEQ bridge used decayed `current_lr` (≈0.04) over 24 layers ($10^{-28}$ signal). Fix: Bridge uses static `base_local_lr`.
    - **Kaggle Setup:** Python req lowered to `>=3.10`. Bypass git-pip cache via `!uv pip install --system --force-reinstall`.
- **[2026-04-10] PPC Optimization:**
    - **Mistake:** FP16 experts w/ 1e-4 LR stagnated. Reverted to FP32 (pre-View Trick).
    - **Learning:** `autocast` activation overhead savings are critical.
