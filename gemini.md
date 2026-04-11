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
- **High-Density Logs:** Update **Diary** mid-flow using tech-shorthand. **Zero Information Loss.** No fluff.

---

## 4. Learnings & Mistakes Diary
- **[2026-04-11] Architecture & Kaggle Strategy (Stability Priority):**
    - **Calculus/NaN:** Jacobian weighted by `topk_scores`. Gate uses `[real||imag]` concat. Bridge uses static `base_local_lr`. ModReLU `safe_mag` ≥ 1e-8. Manual static scaling (`loss/256`) preserves FP16.
    - **Installer Loop Fix (Decoupled Setup):** Removed `bitsandbytes` from core `pyproject.toml` dependencies. This prevents `uv` from triggering a recursive re-install (and re-breaking CUDA links) during the GitHub library installation. BNB is now a manual prerequisite in Cell 1.
    - **Stability Fix (libnvJitLink):** Minimalist injection is too fragile on Kaggle. **Nuclear Option required**: Must `pip uninstall nvidia-*-cu12` and pin `bitsandbytes==0.42.0` to force fallback to system CUDA. 
    - **Dynamic Discovery**: Quiet `LD_LIBRARY_PATH` injection in Cell 1 is mandatory to help `bitsandbytes` find `libcudart.so`.
    - **Binary Compatibility:** Solved `ValueError: numpy.dtype size changed` by removing version pins (except BNB) and pinning `numpy<2.0.0`.
- **[2026-04-10] PPC Optimization:**
    - **Mistake:** FP16 experts w/ 1e-4 LR stagnated. Reverted to FP32 (pre-View Trick).
    - **Learning:** `autocast` activation overhead savings are critical.
