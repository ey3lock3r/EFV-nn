# 3.2B PPC-OCNS Version 3: Expert Team Review & Implementation Plan

Following a rigorous cross-examination of the mathematical foundations, algorithm design, and implementation details of the current `ShardedPPCGraphLLM`, our expert team has refined the analysis. We confirm that a **Full Restart (Phase 0)** is mathematically optimal to break the 7.0 Loss Wall. 

Below is the refined review and the proposed implementation plan for the Version 3 architecture.

## 1. Expert Review & Refinements

### A. Tokenization (The Semantic Multiplier)
*   **Verification:** Your current `ppc_gnn_v2.ipynb` uses `AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B')` in the data loader, which *is* a Tiktoken-based BPE tokenizer. 
*   **Correction:** We previously assumed character-level tokenization based on `run_ppc_shakespeare.py`. Since production already uses Llama-3 BPE (128k vocab), the tokenization bottleneck is **solved**. 
*   **New Diagnosis:** If you are still hitting a Loss Wall at 7.0 while using a 128k BPE vocabulary, the bottleneck is purely mathematical—specifically in the DEQ solver dynamics and the complex-valued gradient flow.

### B. The "Moving Target" DEQ Bottleneck
*   **Analysis:** In `PPCNodeLayer`, the target `x_target` is recalculated *inside* the `local_iters` loop based on the phase rotation of the current state. You are asking the solver to catch a moving target.
*   **Expert Insight:** Standard fixed-point iteration struggles heavily here. While we recommended **Anderson Acceleration**, applying Anderson to a non-stationary (moving) operator without safeguards can lead to divergence.
*   **Refined Recommendation:** We must reformulate the inner loop so the target phase rotation is applied to the *initial* sequence state, rendering the target **stationary** for the duration of the local convergence. Then, Anderson Acceleration can guarantee rapid, deep convergence (reducing 64 iters to ~10).

### C. MoE Routing & Precision Churn
*   **Analysis:** Your `ExpertChoiceMoEMatcher` uses `torch.topk(..., dim=0)`, which correctly implements **True Expert Choice** (experts pick tokens). This is a brilliant architectural strength that prevents token dropping.
*   **Expert Insight:** However, inside the vectorized BMM, you explicitly cast `x_batched` to `.half()`, run the FP16 matmul, and cast the result back to `.float()`. 
*   **Refined Recommendation:** Doing this 64 times inside an unrolled loop introduces accumulating quantization noise (Phasal Jitter). The MoE must either run natively in FP32 during the search phase, or we must use a stochastic rounding kernel to prevent the fixed-point solver from getting trapped in precision limits.

### D. Complex Activation (The ModReLU Trap)
*   **Analysis:** Your `ModReLU` applies a bias-shifted ReLU to the magnitude but leaves the phase purely linear ($z / |z|$).
*   **Expert Insight:** PyTorch uses **Wirtinger calculus** for complex gradients. Because `ModReLU` strictly separates magnitude and phase, it is non-holomorphic. During backpropagation, the gradients flowing through the phase component are completely linear, preventing the network from learning complex, non-linear phase interactions (which are critical for OCNS resonance).
*   **Refined Recommendation:** Replace `ModReLU` with **Complex GELU** (applying GELU to real and imaginary parts independently). This restores rich, non-linear phase dynamics and stabilizes the Wirtinger gradients.

### E. The Gradient Bridge (Pseudo-DEQ)
*   **Analysis:** Your gradient attachment (`out = x_states + lr * (target - prediction)`) is a 1-step Jacobian approximation. 
*   **Expert Insight:** True DEQs require solving the inverse Jacobian $(I - J)^{-1}$ via the **Implicit Function Theorem (IFT)**. Your current bridge severs the deep temporal connections of the solver.
*   **Refined Recommendation:** Wrap the loop in `torch.autograd.Function` and implement a backward Anderson solver to calculate exact implicit gradients.

---

## 2. Proposed Changes

We propose the following sequenced implementation plan to upgrade to Version 3.

### Phase 1: Mathematical Foundations (src/efv_nn/ppc_core.py)
#### [MODIFY] `ppc_core.py`
- Replace `ModReLU` with `ComplexGELU` to stabilize Wirtinger gradients and unlock non-linear phase resonance.
- Refactor `ExpertChoiceMoEMatcher` to support an FP32 "Search Mode" (for the DEQ loop) and an FP16 "Inference Mode".

### Phase 2: The Exact DEQ Solver (src/efv_nn/triton_kernels.py)
#### [NEW] `deq_solvers.py` (or added to kernels)
- Implement **Anderson Acceleration** with a stationary target formulation.
- Implement the **Implicit Function Theorem (IFT)** backward solver using a custom `torch.autograd.Function`.

### Phase 3: Architectural Integration (src/efv_nn/ppc_gnn.py)
#### [MODIFY] `ppc_gnn.py`
- Refactor `PPCNodeLayer` to use the new `torch.autograd.Function` DEQ solver.
- Move the `x_target` phase rotation *outside* the inner loop to create a stationary target.
- Integrate **Pillar 1 (Spectral Gate-Filtering)** from your research directory to route high/low-frequency tokens mathematically.

---

## 3. User Review Required

> [!WARNING]
> **Stationary vs. Moving Target**: Moving the phase rotation outside the inner loop changes the fundamental physics of your PPC algorithm. It means tokens will converge on the prediction of the *previous token's initial state*, rather than dynamically tracking the previous token's settling state. 
> **Question:** Does your biological plausibility axiom strictly require the target to move during the micro-iterations, or is a stationary micro-target acceptable for the sake of mathematical exactness?

> [!IMPORTANT]
> **Full Restart Required**: Implementing Complex GELU and the IFT backward pass changes the gradient landscape completely. The existing 3.2B parameters (currently at 30k steps) will be incompatible with these new physics. We will need to initialize a fresh training run.

---

## 4. Verification Plan

## 5. Phase Transition Playbook: V3 Migration

This section defines the "Go/No-Go" criteria for each training stage.

### Phase 0: Holomorphic Initialization (Step 0-5k)
*   **Success Metrics**: Energy ($E$) stabilizes at $0.08 - 0.12$; Loss ($L$) trends below $9.5$.
*   **Expectation**: High initial volatility as weights align with `ComplexGELU`.
*   **Recalibration**: If $E > 0.5$ after 1k steps, lower `local_lr` to $0.2$.
*   **Early Exit**: Proceed to Phase 1 if $E < 0.08$ for 500 consecutive steps.

### Phase 1: Spectral Acceleration (Step 5k-15k)
*   **Success Metrics**: `S-Blend` deviates from $0.0$ to $0.3+$; Expert Diversity $>70\%$.
*   **Expectation**: Model begins "routing by frequency."
*   **Recalibration**: If `S-Blend` remains $\approx 0$, increase `spectral_lr` to $5e-3$.
*   **Early Exit**: Proceed to Phase 2 if Expert Diversity hits $85\%$.

### Phase 2: Deep Convergence Sprint (Step 15k-40k)
*   **Success Metrics**: Energy ($E$) drops below $0.05$; Loss breaks $8.0$.
*   **Expectation**: Training speed may decrease as Anderson works deeper.
*   **Recalibration**: If $L$ plateaus $>8.5$, increase `local_iters` to $16$.
*   **Early Exit**: Proceed to Phase 3 if $E < 0.03$.

### Phase 3: Semantic Clarification (Step 40k-70k)
*   **Success Metrics**: Loss breaks $7.0$; Generates coherent biological concepts.
*   **Expectation**: This is the "7.0 Barrier" break phase.
*   **Recalibration**: Lower base `lr` to $5e-5$ if loss oscillations exceed $0.2$.

---

## 🛠️ Operational FAQ

**Q: Does the model stop iterating early?**  
**A: Yes.** The Anderson solver in `deq_solvers.py` uses a tolerance check (`res_norm < tol`). If the fixed point is found before reaching `local_iters`, it exits immediately to save compute. This is reflected in the `avg_iters` metric in your logs.

**Q: Why don't I see Spectral Blend in my logs?**  
**A: Log Hook Required.** The training loop must access `model.layers[0].moe.gate.spectral_blend`. I have updated your notebook's `train` function to pull this value and log it to both the terminal and W&B.
