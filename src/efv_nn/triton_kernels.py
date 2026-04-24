import torch
import triton
import triton.language as tl

print(f"🚀 [HYPER-DRIVE] Loading Triton Kernels from: {__file__}")

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

    prev_r_f32 = prev_r.to(tl.float32)
    prev_i_f32 = prev_i.to(tl.float32)
    cos_p_f32  = cos_p.to(tl.float32)
    sin_p_f32  = sin_p.to(tl.float32)
    rot_r = prev_r_f32 * cos_p_f32 - prev_i_f32 * sin_p_f32
    rot_i = prev_r_f32 * sin_p_f32 + prev_i_f32 * cos_p_f32

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
            dr_f32 = dr.to(tl.float32)
            di_f32 = di.to(tl.float32)
            gr_f32 = gr.to(tl.float32)
            gi_f32 = gi.to(tl.float32)
            acc_r += tl.where(v_mask, dr_f32 * gr_f32 - di_f32 * gi_f32, 0.0)
            acc_i += tl.where(v_mask, dr_f32 * gi_f32 + di_f32 * gr_f32, 0.0)

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
    """In-place update."""
    assert x_states.is_contiguous()
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
    # Surgical Sigmoid Fix: tanh(x) = 2*sigmoid(2x) - 1
    # Consistent across all Triton versions.
    return 0.5 * x * (2.0 * tl.sigmoid(1.59576912 * (x + 0.044715 * x * x * x)))

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

    r = tl.load(output_ptr + row + d_off * 2,     mask=mask, other=0.0) / count
    i = tl.load(output_ptr + row + d_off * 2 + 1, mask=mask, other=0.0) / count

    bias_val = tl.load(bias_ptr + d_off, mask=mask, other=0.0)
    
    # Use tl.where for absolute value to maximize compatibility
    abs_r = tl.where(r < 0, -r, r)
    abs_i = tl.where(i < 0, -i, i)
    r = tl.where(abs_r > 1e18, 0.0, r)
    i = tl.where(abs_i > 1e18, 0.0, i)

    r_act = _gelu_fast(r + bias_val)
    i_act = _gelu_fast(i + bias_val)

    tl.store(result_ptr + row + d_off * 2,     r_act, mask=mask)
    tl.store(result_ptr + row + d_off * 2 + 1, i_act, mask=mask)


def fused_normalize_activate(output, counts, bias, out=None):
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
    
    f_x_base = b * N + n_off
    f_x = tl.load(f_x_ptr + f_x_base, mask=mask, other=0.0)
    acc = f_x.to(tl.float32)
    for i in range(m_k):
        a = tl.load(alpha_ptr + b * m_k + i).to(tl.float32)
        hist_val = tl.load(F_hist_ptr + b * m_total * N + i * N + n_off, mask=mask, other=0.0).to(tl.float32)
        acc -= a * (f_x.to(tl.float32) - hist_val)
    tl.store(out_ptr + f_x_base, acc, mask=mask)


def anderson_mixing(f_x, F_hist, alpha, m_k, out=None):
    B, N = f_x.shape
    _, m_total, _ = F_hist.shape
    if out is None:
        out = torch.empty_like(f_x)
    BLOCK_N = 1024
    grid = (B * ((N + BLOCK_N - 1) // BLOCK_N),)
    with torch.cuda.device(f_x.device):
        _anderson_mixing_kernel[grid](
            f_x.contiguous(), F_hist.contiguous(), alpha.contiguous(), out,
            m_k=m_k, m_total=m_total, B=B, N=N, BLOCK_N=BLOCK_N
        )
    return out


# ============================================================
# Kernel 6: Fused MoE Dispatch & Delay
# ============================================================
@triton.jit
def _moe_dispatch_delay_kernel(
    x_states_ptr, delay_gains_ptr, topk_indices_ptr, out_ptr,
    tau0: tl.constexpr, tau1: tl.constexpr, tau2: tl.constexpr, tau3: tl.constexpr,
    tau4: tl.constexpr, tau5: tl.constexpr, tau6: tl.constexpr, tau7: tl.constexpr,
    num_delays: tl.constexpr,
    T: tl.constexpr, D: tl.constexpr, E: tl.constexpr, K: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64) # expert_idx * K + node_idx
    token_idx = tl.load(topk_indices_ptr + pid)
    t     = token_idx % T
    b_idx = token_idx // T
    src_row = token_idx * D * 2
    d_off   = tl.arange(0, BLOCK_D).to(tl.int64)
    mask    = d_off < D
    acc_r = tl.load(x_states_ptr + src_row + d_off * 2,     mask=mask, other=0.0)
    acc_i = tl.load(x_states_ptr + src_row + d_off * 2 + 1, mask=mask, other=0.0)
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
            hist_token_idx = (b_idx * T) + (t - tau)
            hist_row = hist_token_idx * D * 2
            hist_mask = mask & valid
            dr = tl.load(x_states_ptr + hist_row + d_off * 2,     mask=hist_mask, other=0.0)
            di = tl.load(x_states_ptr + hist_row + d_off * 2 + 1, mask=hist_mask, other=0.0)
            gain_base = idx * D * 2
            gr = tl.load(delay_gains_ptr + gain_base + d_off * 2,     mask=mask, other=0.0)
            gi = tl.load(delay_gains_ptr + gain_base + d_off * 2 + 1, mask=mask, other=0.0)
            v_mask = tl.broadcast_to(valid, (BLOCK_D,))
            dr_f32 = dr.to(tl.float32)
            di_f32 = di.to(tl.float32)
            gr_f32 = gr.to(tl.float32)
            gi_f32 = gi.to(tl.float32)
            acc_r += tl.where(v_mask, dr_f32 * gr_f32 - di_f32 * gi_f32, 0.0)
            acc_i += tl.where(v_mask, dr_f32 * gi_f32 + di_f32 * gr_f32, 0.0)
    out_row = pid * D * 2
    tl.store(out_ptr + out_row + d_off * 2,     acc_r, mask=mask)
    tl.store(out_ptr + out_row + d_off * 2 + 1, acc_i, mask=mask)


def fused_moe_dispatch_delay(x_states, delay_gains, prime_delays, topk_indices, out=None):
    B, T, D, _ = x_states.shape
    E, K = topk_indices.shape
    if out is None:
        out = torch.empty((E, K, D, 2), device=x_states.device, dtype=x_states.dtype)
    padded  = list(prime_delays) + [0] * (8 - len(prime_delays))
    BLOCK_D = triton.next_power_of_2(D)
    with torch.cuda.device(x_states.device):
        _moe_dispatch_delay_kernel[(E * K,)](
            x_states.contiguous(), delay_gains.contiguous(), topk_indices.contiguous(), out,
            tau0=padded[0], tau1=padded[1], tau2=padded[2], tau3=padded[3],
            tau4=padded[4], tau5=padded[5], tau6=padded[6], tau7=padded[7],
            num_delays=len(prime_delays),
            T=T, D=D, E=E, K=K, BLOCK_D=BLOCK_D
        )
    return out


# ============================================================
# Kernel 7: Fused MoE Aggregator (Atomic Add + GELU)
# ============================================================
@triton.jit
def _moe_aggregator_kernel(
    y_weighted_ptr, topk_indices_ptr, bias_ptr, out_ptr, counts_ptr,
    E_K: tl.constexpr, D: tl.constexpr, B_T: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    token_idx = tl.load(topk_indices_ptr + pid)
    y_row     = pid * D * 2
    out_row   = token_idx * D * 2
    d_off = tl.arange(0, BLOCK_D).to(tl.int64)
    mask  = d_off < D
    yr = tl.load(y_weighted_ptr + y_row + d_off * 2,     mask=mask, other=0.0)
    yi = tl.load(y_weighted_ptr + y_row + d_off * 2 + 1, mask=mask, other=0.0)
    tl.atomic_add(out_ptr + out_row + d_off * 2,     yr, mask=mask)
    tl.atomic_add(out_ptr + out_row + d_off * 2 + 1, yi, mask=mask)
    tl.atomic_add(counts_ptr + token_idx, 1.0)


def fused_moe_aggregator(y_weighted, topk_indices, B_T, bias, out=None):
    device = y_weighted.device
    E_K, D, _ = y_weighted.shape
    if out is None:
        out = torch.zeros((B_T, D, 2), device=device, dtype=torch.float32)
    else:
        out.zero_()
    counts = torch.zeros((B_T,), device=device, dtype=torch.float32)
    BLOCK_D = triton.next_power_of_2(D)
    with torch.cuda.device(device):
        _moe_aggregator_kernel[(E_K,)](
            y_weighted.float().contiguous(), topk_indices.view(-1).contiguous(), bias.float().contiguous(), out, counts,
            E_K=E_K, D=D, B_T=B_T, BLOCK_D=BLOCK_D
        )
        fused_normalize_activate(out, counts, bias, out=out)
    return out


# ============================================================
# Kernel 8: Fused Spectral Gate (Mag + Pool)
# ============================================================
@triton.jit
def _spectral_gate_pool_kernel(
    x_fft_ptr, out_ptr,
    B: tl.constexpr, T_half: tl.constexpr, D: tl.constexpr,
    BLOCK_T: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid_b = tl.program_id(0).to(tl.int64)
    pid_d = tl.program_id(1).to(tl.int64)
    if pid_b >= B or pid_d >= D: return
    t_off = tl.arange(0, BLOCK_T).to(tl.int64)
    t_mask = t_off < T_half
    base = pid_b * T_half * D * 2 + pid_d * 2
    stride_t = D * 2
    fr = tl.load(x_fft_ptr + base + t_off * stride_t,     mask=t_mask, other=0.0)
    fi = tl.load(x_fft_ptr + base + t_off * stride_t + 1, mask=t_mask, other=0.0)
    # Using tl.sqrt for maximum compatibility
    mag = tl.sqrt(fr * fr + fi * fi)
    mid = T_half // 2
    low_mask = t_mask & (t_off < mid)
    high_mask = t_mask & (t_off >= mid)
    low_sum = tl.sum(tl.where(low_mask, mag, 0.0))
    high_sum = tl.sum(tl.where(high_mask, mag, 0.0))
    mid_safe = tl.maximum(mid.to(tl.float32), 1.0)
    span_safe = tl.maximum((T_half - mid).to(tl.float32), 1.0)
    low_mean = low_sum / mid_safe
    high_mean = high_sum / span_safe
    out_base = (pid_b * D + pid_d) * 2
    tl.store(out_ptr + out_base,     low_mean)
    tl.store(out_ptr + out_base + 1, high_mean)


def fused_spectral_gate(x_fft, B, T_half, D):
    device = x_fft.device
    out = torch.empty((B, D, 2), device=device, dtype=x_fft.dtype)
    BLOCK_T = triton.next_power_of_2(T_half)
    grid = (B, D)
    with torch.cuda.device(device):
        _spectral_gate_pool_kernel[grid](
            x_fft.contiguous(), out, B=B, T_half=T_half, D=D, BLOCK_T=BLOCK_T, BLOCK_D=1
        )
    return out
