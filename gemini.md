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
- **[2026-04-11] Holistic Architecture & Kaggle Strategy:**
    - **Stability/NaN:** `GradScaler` fails on FP16 params. Replaced with **Manual Static Scaling** (`loss/256`, `clip/256`). Adam scale-invariance preserves updates. **ModReLU `safe_mag` must be ≥1e-8** (fixed from 1e-12) to prevent FP16 div-by-zero.
    - **VRAM/FP16:** 64 expertos (3.2B) fit T4 via View Trick (`view_as_real(...).half()`). Unpack JIT: `view_as_complex(w.float())`. Saves 6GB.
    - **Gradients:**
        - **DEQ Bridge:** Fixed vanishing signal ($10^{-28}$) by using static `base_local_lr` (previous used decayed `current_lr`).
        - **Jacobian:** `transpose_forward` now weights by `topk_scores` to match weighted forward pass.
        - **Checkpointing:** Removed `autocast` from `checkpoint()` wrapper (doesn't propagate in `reentrant=False`). Autocast now lives inside layers.
    - **MoE Routing:** Gate now uses `[real||imag]` concat instead of `x.abs()` to leverage phase info. `k_nodes` scales dynamically ($B_T / num\_experts$).
    - **Library/Notebook Sync:** Fixed `ShardedPPCGraphLLM` return signature mismatch (`(logits, avg_iters)`). Replaced all notebook inline code with `uv` GitHub install.
    - **Kaggle Setup:** Python req lowered to `>=3.10`. Bypass git-pip cache via `!uv pip install --system --force-reinstall`. Consolidated all installs into **Cell 1** for single-pass resolution.
    - **Versioning:** Removed imaginary high versions (sklearn 1.8) which caused `ImportError`. Reverted to stable 2024 versions in `pyproject.toml`.
- **[2026-04-10] PPC Optimization:**
    - **Mistake:** FP16 experts w/ 1e-4 LR stagnated. Reverted to FP32 (pre-View Trick).
    - **Learning:** `autocast` activation overhead savings are critical.
