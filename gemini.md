# EFV-nn: Gemini Protocol

## 1. Context & Tech
- **Focus:** EFV & SOTA ML research.
- **Stack:** Python, `uv`, `numpy`, `scikit-learn`, `matplotlib`.

## 2. Standards
- **Coding:** Modular, DRY, PEP 8, Type Hints, Robust Error Handling.
- **Design:** No hardware conflicts (T4 VRAM). Prioritize stable architecture over fragile hacks.
- **Sync:** Strict `src/` ↔ `.ipynb` alignment.
- **Testing:** `pytest` (AAA).

## 3. Directives
- **Workspace Hygiene:** Immediate cleanup of temp files/scratch/logs. Update `task.md`.
- **Review:** Consult `gemini.md` & `task.md` before architecture shifts.
- **High-Density Logs:** Proactively update **Diary** mid-flow using concise tech-shorthand. No fluff.

---

## 4. Learnings & Mistakes Diary
- **[2026-04-11] VRAM & FP16 View Trick:**
    - **Mistake:** Scaled 64 -> 32 experts due to false OOM from stripping `.half()`.
    - **Fix:** Use "View Trick": `experts_weight_real = nn.Parameter(torch.view_as_real(init_complex).half())`. Saves 50% VRAM (6GB vs 12GB). Unpack JIT: `view_as_complex(weight.float())`.
    - **Mistake (NaN):** Removing `GradScaler` at Step 30 caused FP16 overflow (`>65504`) -> `inf` -> `NaN` (Adam $m/\sqrt{v}$).
    - **Solution:** `GradScaler` fails on FP16 params (`ValueError`). Use **Manual Static Scaling**: `(loss/256).backward()` + `clip_grad_norm(1/256)`. Adam's scale-invariance ensures identical updates. **Never** disable `autocast` or `clipping`.
- **[2026-04-11] DEQ Bridge & Vanishing Gradients:**
    - **Mistake:** DEQ bridge used decayed `current_lr` (≈0.04). Applied over 24 layers: `0.04^24 ≈ 10^{-28}`. Signal vanished; loss flatlined (11.96 → 11.90).
    - **Fix:** Bridge must use un-decayed constant: `out = x_states + self.base_local_lr * (target - pred)`. 
    - **Rule:** Separate inner-loop LR (dynamic) from gradient bridge LR (static).
- **[2026-04-10] PPC Optimization:**
    - **Mistake:** FP16 experts w/ 1e-4 LR stagnated. Reverted to FP32 (pre-View Trick).
    - **Learning:** `autocast` activation overhead savings are critical.
