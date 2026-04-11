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
- **[2026-04-11] Model Architecture Stabilization:**
    - **Calculus/NaN:** Jacobian weighted by `topk_scores`. Gate uses `[real||imag]` concat. Bridge uses static `base_local_lr`. ModReLU `safe_mag` ≥ 1e-8. Manual static scaling (`loss/256`) preserves FP16.
    - **VRAM Optimization:** 64 experto (3.2B) T4 fit via View Trick (`view_as_real(...).half()`).
- **[2026-04-10] PPC Optimization:**
    - **Mistake:** FP16 experts w/ 1e-4 LR stagnated. Reverted to FP32 (pre-View Trick).
    - **Learning:** `autocast` activation overhead savings are critical.
