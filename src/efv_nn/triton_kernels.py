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
    e_idx = pid // K
    k_idx = pid % K
    
    # Get the original token index from topk_indices [E, K]
    token_idx = tl.load(topk_indices_ptr + pid)
    
    t     = token_idx % T
    b_idx = token_idx // T
    
    # Source row in x_states [B, T, D, 2]
    src_row = token_idx * D * 2
    d_off   = tl.arange(0, BLOCK_D).to(tl.int64)
    mask    = d_off < D
    
    # Base state (no delay)
    acc_r = tl.load(x_states_ptr + src_row + d_off * 2,     mask=mask, other=0.0)
    acc_i = tl.load(x_states_ptr + src_row + d_off * 2 + 1, mask=mask, other=0.0)
    
    # Apply OCNS Delays
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
            # Find the history token in the same batch
            hist_token_idx = (b_idx * T) + (t - tau)
            hist_row = hist_token_idx * D * 2
            hist_mask = mask & valid

            dr = tl.load(x_states_ptr + hist_row + d_off * 2,     mask=hist_mask, other=0.0)
            di = tl.load(x_states_ptr + hist_row + d_off * 2 + 1, mask=hist_mask, other=0.0)

            gain_base = idx * D * 2
            gr = tl.load(delay_gains_ptr + gain_base + d_off * 2,     mask=mask, other=0.0)
            gi = tl.load(delay_gains_ptr + gain_base + d_off * 2 + 1, mask=mask, other=0.0)

            v_mask = tl.broadcast_to(valid, (BLOCK_D,))
            acc_r += tl.where(v_mask, dr * gr - di * gi, 0.0)
            acc_i += tl.where(v_mask, dr * gi + di * gr, 0.0)

    # Store into gathered buffer [E, K, D, 2]
    out_row = pid * D * 2
    tl.store(out_ptr + out_row + d_off * 2,     acc_r, mask=mask)
    tl.store(out_ptr + out_row + d_off * 2 + 1, acc_i, mask=mask)


def fused_moe_dispatch_delay(x_states, delay_gains, prime_delays, topk_indices, out=None):
    """
    Combines OCNS history lookups and MoE token gathering.
    x_states: [B, T, D, 2]
    topk_indices: [E, K] -> indices in flattened [B*T]
    out: [E, K, D, 2]
    """
    B, T, D, _ = x_states.shape
    E, K = topk_indices.shape
    if out is None:
        out = torch.empty((E, K, D, 2), device=x_states.device, dtype=x_states.dtype)
    
    padded  = list(prime_delays) + [0] * (8 - len(prime_delays))
    BLOCK_D = triton.next_power_of_2(D)
    
    with torch.cuda.device(x_states.device):
        _moe_dispatch_delay_kernel[(E * K,)](
            x_states.contiguous(),
            delay_gains.contiguous(),
            topk_indices.contiguous(),
            out,
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
    pid = tl.program_id(0).to(tl.int64) # index into [E*K]
    
    token_idx = tl.load(topk_indices_ptr + pid)
    y_row     = pid * D * 2
    out_row   = token_idx * D * 2
    
    d_off = tl.arange(0, BLOCK_D).to(tl.int64)
    mask  = d_off < D
    
    yr = tl.load(y_weighted_ptr + y_row + d_off * 2,     mask=mask, other=0.0)
    yi = tl.load(y_weighted_ptr + y_row + d_off * 2 + 1, mask=mask, other=0.0)
    
    # We use Atomic Add for the reduction part
    # Note: Triton 3.6.0 atomic_add only supports float32 for certain backends.
    # We ensure input is float32 at the wrapper level.
    tl.atomic_add(out_ptr + out_row + d_off * 2,     yr, mask=mask)
    tl.atomic_add(out_ptr + out_row + d_off * 2 + 1, yi, mask=mask)
    tl.atomic_add(counts_ptr + token_idx, 1.0)


def fused_moe_aggregator(y_weighted, topk_indices, B_T, bias, out=None):
    """
    Combines index_add and ComplexGELU.
    y_weighted: [E*K, D, 2]
    topk_indices: [E, K] -> flattened to [E*K]
    bias: [D, 2]
    out: [B_T, D, 2]
    """
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
            y_weighted.float().contiguous(),
            topk_indices.view(-1).contiguous(),
            bias.float().contiguous(),
            out,
            counts,
            E_K=E_K, D=D, B_T=B_T, BLOCK_D=BLOCK_D
        )
        
        # After atomic reduction, we apply normalization and activation
        # We reuse the existing normalize_activate kernel
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
    # Operating on [B, D] to calculate the mean across T_half
    pid_b = tl.program_id(0).to(tl.int64)
    pid_d = tl.program_id(1).to(tl.int64)
    
    if pid_b >= B or pid_d >= D:
        return
        
    # Offset for x_fft [B, T_half, D, 2]
    # Each program calculates mean(abs(fft)) for one (batch, dim) pair
    t_off = tl.arange(0, BLOCK_T).to(tl.int64)
    t_mask = t_off < T_half
    
    # Load all T bins for this (B, D)
    base = pid_b * T_half * D * 2 + pid_d * 2
    stride_t = D * 2
    
    # Load real and imag parts
    fr = tl.load(x_fft_ptr + base + t_off * stride_t,     mask=t_mask, other=0.0)
    fi = tl.load(x_fft_ptr + base + t_off * stride_t + 1, mask=t_mask, other=0.0)
    
    # Magnitude: sqrt(r^2 + i^2)
    mag = tl.sqrt(fr * fr + fi * fi)
    
    # Split pool: Low and High frequencies
    mid = T_half // 2
    low_mask = t_mask & (t_off < mid)
    high_mask = t_mask & (t_off >= mid)
    
    low_sum = tl.sum(tl.where(low_mask, mag, 0.0))
    high_sum = tl.sum(tl.where(high_mask, mag, 0.0))
    
    low_mean = low_sum / mid
    high_mean = high_sum / (T_half - mid)
    
    # Store results [B, D, 2] -> [low_mean, high_mean]
    out_base = (pid_b * D + pid_d) * 2
    tl.store(out_ptr + out_base,     low_mean)
    tl.store(out_ptr + out_base + 1, high_mean)


def fused_spectral_gate(x_fft, B, T_half, D):
    """
    Fuses magnitude calculation and mean pooling (low/high split).
    x_fft: [B, T_half, D, 2] (complex output from torch.fft.rfft)
    returns: [B, D, 2] containing [low_mean, high_mean]
    """
    device = x_fft.device
    out = torch.empty((B, D, 2), device=device, dtype=x_fft.dtype)
    
    BLOCK_T = triton.next_power_of_2(T_half)
    grid = (B, D)
    
    with torch.cuda.device(device):
        _spectral_gate_pool_kernel[grid](
            x_fft.contiguous(),
            out,
            B=B, T_half=T_half, D=D,
            BLOCK_T=BLOCK_T, BLOCK_D=1 # Not used but for signature
        )
    return out
