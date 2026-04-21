"""
Triton Kernels for PPC-GNN (Triton 3.6.0 - Hyper-Drive Edition).

Optimizations:
  - Buffer-Aware: Functions now accept optional 'out' buffers to avoid churn.
  - Contextless: Removed device context from wrappers (handled at layer level).
"""

import torch
import triton
import triton.language as tl

print(f"🚀 [HYPER-DRIVE] Loading Triton Kernels from: {__file__}")
print(f"🧬 [HYPER-DRIVE] fused_phase_rotation signature verified with 'out' parameter.")


# ============================================================
# Kernel 1: Fused Phase Rotation + Target Construction
# ============================================================
@triton.jit
def _phase_rotation_kernel(
    x_states_ptr, cos_p_ptr, sin_p_ptr, x_target_ptr,
    T: tl.constexpr, D: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    t   = pid % T
    row = pid * D * 2

    d_off = tl.arange(0, BLOCK_D).to(tl.int64)
    mask  = d_off < D

    cur_r = tl.load(x_states_ptr + row + d_off * 2,     mask=mask, other=0.0)
    cur_i = tl.load(x_states_ptr + row + d_off * 2 + 1, mask=mask, other=0.0)

    has_prev = t > 0
    prev_row = (pid - has_prev.to(tl.int64)) * D * 2

    prev_r = tl.load(x_states_ptr + prev_row + d_off * 2,     mask=mask, other=0.0)
    prev_i = tl.load(x_states_ptr + prev_row + d_off * 2 + 1, mask=mask, other=0.0)
    cos_p  = tl.load(cos_p_ptr + d_off, mask=mask, other=0.0)
    sin_p  = tl.load(sin_p_ptr + d_off, mask=mask, other=0.0)

    rot_r = prev_r * cos_p - prev_i * sin_p
    rot_i = prev_r * sin_p + prev_i * cos_p

    cond = tl.broadcast_to(has_prev, (BLOCK_D,))
    out_r = tl.where(cond, rot_r, cur_r)
    out_i = tl.where(cond, rot_i, cur_i)

    tl.store(x_target_ptr + row + d_off * 2,     out_r, mask=mask)
    tl.store(x_target_ptr + row + d_off * 2 + 1, out_i, mask=mask)


def fused_phase_rotation(x_states, cos_p, sin_p, out=None):
    """Expects [B, T, D, 2]. Returns [B, T, D, 2]."""
    B, T, D, _ = x_states.shape
    if out is None:
        out = torch.empty_like(x_states)
    
    BLOCK_D = triton.next_power_of_2(D)
    # The kernel operates on flat (B*T) rows
    with torch.cuda.device(x_states.device):
        _phase_rotation_kernel[(B * T,)](
            x_states.contiguous(), 
            cos_p.contiguous(), 
            sin_p.contiguous(), 
            out,
            T=T, D=D, BLOCK_D=BLOCK_D,
        )
    return out


# ============================================================
# Kernel 2: Fused OCNS Delay Embedding
# ============================================================
@triton.jit
def _ocns_delay_kernel(
    x_states_ptr, delay_gains_ptr, x_eff_ptr,
    tau0: tl.constexpr, tau1: tl.constexpr, tau2: tl.constexpr, tau3: tl.constexpr,
    tau4: tl.constexpr, tau5: tl.constexpr, tau6: tl.constexpr, tau7: tl.constexpr,
    num_delays: tl.constexpr,
    T: tl.constexpr, D: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid   = tl.program_id(0).to(tl.int64)
    t     = pid % T
    row   = pid * D * 2

    d_off = tl.arange(0, BLOCK_D).to(tl.int64)
    mask  = d_off < D

    acc_r = tl.load(x_states_ptr + row + d_off * 2,     mask=mask, other=0.0)
    acc_i = tl.load(x_states_ptr + row + d_off * 2 + 1, mask=mask, other=0.0)

    for idx in tl.static_range(8):
        if idx < num_delays:
            if idx == 0: tau = tau0
            elif idx == 1: tau = tau1
            elif idx == 2: tau = tau2
            elif idx == 3: tau = tau3
            elif idx == 4: tau = tau4
            elif idx == 5: tau = tau5
            elif idx == 6: tau = tau6
            else: tau = tau7

            valid = t >= tau
            src_pid  = pid - (valid.to(tl.int64) * tau) 
            hist_row = src_pid * D * 2
            hist_mask = mask & valid

            dr = tl.load(x_states_ptr + hist_row + d_off * 2,     mask=hist_mask, other=0.0)
            di = tl.load(x_states_ptr + hist_row + d_off * 2 + 1, mask=hist_mask, other=0.0)

            gain_base = idx * D * 2
            gr = tl.load(delay_gains_ptr + gain_base + d_off * 2,     mask=mask, other=0.0)
            gi = tl.load(delay_gains_ptr + gain_base + d_off * 2 + 1, mask=mask, other=0.0)

            v_mask = tl.broadcast_to(valid, (BLOCK_D,))
            acc_r += tl.where(v_mask, dr * gr - di * gi, 0.0)
            acc_i += tl.where(v_mask, dr * gi + di * gr, 0.0)

    tl.store(x_eff_ptr + row + d_off * 2,     acc_r, mask=mask)
    tl.store(x_eff_ptr + row + d_off * 2 + 1, acc_i, mask=mask)


def fused_ocns_delay(x_states, delay_gains, prime_delays, out=None):
    """Expects [B, T, D, 2]. Returns [B, T, D, 2]."""
    B, T, D, _ = x_states.shape
    if out is None:
        out = torch.empty_like(x_states)
    
    padded  = list(prime_delays) + [0] * (8 - len(prime_delays))
    BLOCK_D = triton.next_power_of_2(D)
    with torch.cuda.device(x_states.device):
        _ocns_delay_kernel[(B * T,)](
            x_states.contiguous(), 
            delay_gains.contiguous(), 
            out,
            tau0=padded[0], tau1=padded[1], tau2=padded[2], tau3=padded[3],
            tau4=padded[4], tau5=padded[5], tau6=padded[6], tau7=padded[7],
            num_delays=len(prime_delays),
            T=T, D=D, BLOCK_D=BLOCK_D,
        )
    return out


# ============================================================
# Kernel 3: Fused State Update (Clamp + Scaled Add, In-Place)
# ============================================================
@triton.jit
def _state_update_kernel(
    x_states_ptr, step_ptr,
    lr:    tl.constexpr,
    numel: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid  = tl.program_id(0).to(tl.int64)
    off  = (pid * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
    mask = off < numel

    x = tl.load(x_states_ptr + off, mask=mask, other=0.0)
    s = tl.load(step_ptr      + off, mask=mask, other=0.0)

    s = tl.where(s != s, 0.0, s) # NaN siphon

    s_clamped = tl.minimum(tl.maximum(s, -10.0), 10.0)
    tl.store(x_states_ptr + off, x + lr * s_clamped, mask=mask)


def fused_state_update(x_states, step, current_lr):
    """In-place update. Accepts any shape as long as it is contiguous."""
    assert x_states.is_contiguous(), "State update must be in-place on contiguous tensor"
    numel = x_states.numel()
    BLOCK = 1024
    grid  = ((numel + BLOCK - 1) // BLOCK,)
    with torch.cuda.device(x_states.device):
        _state_update_kernel[grid](
            x_states, 
            step.contiguous(),
            lr=float(current_lr), numel=numel, BLOCK=BLOCK,
        )


# ============================================================
# Kernel 4: Fused Normalize + ComplexGELU Activation
# ============================================================
@triton.jit
def _gelu_fast(x):
    """Fast GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))"""
    return 0.5 * x * (1.0 + tl.tanh(0.79788456 * (x + 0.044715 * x * x * x)))

@triton.jit
def _normalize_activate_kernel(
    output_ptr, counts_ptr, bias_ptr, result_ptr,
    B_T: tl.constexpr, D: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid   = tl.program_id(0).to(tl.int64)
    d_off = tl.arange(0, BLOCK_D).to(tl.int64)
    mask  = d_off < D
    row   = pid * D * 2

    count = tl.load(counts_ptr + pid)
    count = tl.maximum(count, 1.0)

    # Load and normalize
    r = tl.load(output_ptr + row + d_off * 2,     mask=mask, other=0.0) / count
    i = tl.load(output_ptr + row + d_off * 2 + 1, mask=mask, other=0.0) / count

    # Apply ComplexGELU (PPC-OCNS v3 Pillar: Holomorphic Stability)
    # We apply GELU to real and imaginary parts independently.
    # Bias is added before activation.
    bias_r = tl.load(bias_ptr + d_off, mask=mask, other=0.0)
    bias_i = tl.load(bias_ptr + D + d_off, mask=mask, other=0.0) # Assume bias is [D, 2]

    # Nan/Inf Guard
    r = tl.where(tl.abs(r) > 1e18, 0.0, r)
    i = tl.where(tl.abs(i) > 1e18, 0.0, i)

    r_act = _gelu_fast(r + bias_r)
    i_act = _gelu_fast(i + bias_i)

    tl.store(result_ptr + row + d_off * 2,     r_act, mask=mask)
    tl.store(result_ptr + row + d_off * 2 + 1, i_act, mask=mask)


def fused_normalize_activate(output, counts, bias, out=None):
    """Expects [B, T, D, 2] or [B_T, D, 2]. Returns same shape as input."""
    if output.dim() == 4:
        B, T, D, _ = output.shape
        B_T = B * T
    else:
        B_T, D, _ = output.shape

    if out is None:
        out = torch.empty_like(output)
    
    BLOCK_D = triton.next_power_of_2(D)
    with torch.cuda.device(output.device):
        _normalize_activate_kernel[(B_T,)](
            output.contiguous(), 
            counts.contiguous().view(-1), 
            bias.contiguous(), 
            out,
            B_T=B_T, D=D, BLOCK_D=BLOCK_D,
        )
    return out


# ============================================================
# Kernel 5: Anderson Mixing (History-Weighted Sum)
# ============================================================
@triton.jit
def _anderson_mixing_kernel(
    f_x_ptr, F_hist_ptr, alpha_ptr, out_ptr,
    m_k: tl.constexpr, m_total: tl.constexpr,
    B: tl.constexpr, N: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    b   = pid // ((N + BLOCK_N - 1) // BLOCK_N)
    off = (pid % ((N + BLOCK_N - 1) // BLOCK_N)) * BLOCK_N
    
    n_off = off + tl.arange(0, BLOCK_N)
    mask  = n_off < N
    
    # Base offset for current batch in history buffers
    # F_hist is [B, m_total, N]
    
    f_x_base = b * N + n_off
    f_x = tl.load(f_x_ptr + f_x_base, mask=mask, other=0.0)
    
    acc = f_x
    
    for i in range(m_k):
        # alpha is [B, m_k]
        a = tl.load(alpha_ptr + b * m_k + i)
        
        hist_val = tl.load(F_hist_ptr + b * m_total * N + i * N + n_off, mask=mask, other=0.0)
        
        acc -= a * (f_x - hist_val)
        
    tl.store(out_ptr + f_x_base, acc, mask=mask)


def anderson_mixing(f_x, F_hist, alpha, m_k, out=None):
    """
    Computes: out = f_x - sum(alpha_i * (f_x - F_hist_i))
    f_x: [B, N]
    F_hist: [B, m_total, N]
    alpha: [B, m_k]
    """
    B, N = f_x.shape
    _, m_total, _ = F_hist.shape
    if out is None:
        out = torch.empty_like(f_x)
        
    BLOCK_N = 1024
    grid = (B * ((N + BLOCK_N - 1) // BLOCK_N),)
    
    with torch.cuda.device(f_x.device):
        _anderson_mixing_kernel[grid](
            f_x.contiguous(),
            F_hist.contiguous(),
            alpha.contiguous(),
            out,
            m_k=m_k, m_total=m_total, B=B, N=N, BLOCK_N=BLOCK_N
        )
    return out
