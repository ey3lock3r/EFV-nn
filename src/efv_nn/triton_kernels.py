"""
Triton Kernels for PPC-GNN (Triton 3.x compatible).

Key Triton 3.x rules applied:
  - No `if` on runtime tensors. Use tl.where() and masked loads.
  - All pointer offsets cast to tl.int64 to prevent 32-bit overflow.
  - Scalar float kernel args typed as tl.constexpr where needed.

All kernels operate on [B, T, D, 2] interleaved-real (complex) tensors stored
contiguously. The last dimension (2) holds [real, imaginary].
"""

import torch
import triton
import triton.language as tl


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

    # Always load current token (used when t == 0)
    cur_r = tl.load(x_states_ptr + row + d_off * 2,     mask=mask, other=0.0)
    cur_i = tl.load(x_states_ptr + row + d_off * 2 + 1, mask=mask, other=0.0)

    # Load prev token (safe even when t == 0 due to mask)
    has_prev  = t > 0
    prev_row  = tl.where(has_prev, pid - 1, pid) * D * 2
    prev_r    = tl.load(x_states_ptr + prev_row + d_off * 2,     mask=mask, other=0.0)
    prev_i    = tl.load(x_states_ptr + prev_row + d_off * 2 + 1, mask=mask, other=0.0)
    cos_p     = tl.load(cos_p_ptr + d_off, mask=mask, other=0.0)
    sin_p     = tl.load(sin_p_ptr + d_off, mask=mask, other=0.0)

    rot_r = prev_r * cos_p - prev_i * sin_p
    rot_i = prev_r * sin_p + prev_i * cos_p

    # t == 0 → copy current; t > 0 → write rotated prev
    out_r = tl.where(has_prev, rot_r, cur_r)
    out_i = tl.where(has_prev, rot_i, cur_i)

    tl.store(x_target_ptr + row + d_off * 2,     out_r, mask=mask)
    tl.store(x_target_ptr + row + d_off * 2 + 1, out_i, mask=mask)


def fused_phase_rotation(x_states, cos_p, sin_p):
    """
    Fused phase rotation + target construction.
    Args:
        x_states: [B, T, D, 2] float32 contiguous
        cos_p:    [D] float32
        sin_p:    [D] float32
    Returns:
        x_target: [B, T, D, 2] float32
    """
    B, T, D, _ = x_states.shape
    x_target    = torch.empty_like(x_states)
    BLOCK_D     = triton.next_power_of_2(D)
    _phase_rotation_kernel[(B * T,)](
        x_states.contiguous(), cos_p.contiguous(), sin_p.contiguous(), x_target,
        T=T, D=D, BLOCK_D=BLOCK_D,
    )
    return x_target


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

    # Start accumulator from current state
    acc_r = tl.load(x_states_ptr + row + d_off * 2,     mask=mask, other=0.0)
    acc_i = tl.load(x_states_ptr + row + d_off * 2 + 1, mask=mask, other=0.0)

    taus = (tau0, tau1, tau2, tau3, tau4, tau5, tau6, tau7)
    for idx in tl.static_range(num_delays):
        tau: tl.constexpr = taus[idx]

        # Only accumulate for tokens that have a valid history
        valid    = t >= tau
        src_pid  = tl.where(valid, pid - tau, pid)   # clamp to 0 when invalid
        hist_row = src_pid * D * 2
        hist_mask = mask & valid

        dr = tl.load(x_states_ptr + hist_row + d_off * 2,     mask=hist_mask, other=0.0)
        di = tl.load(x_states_ptr + hist_row + d_off * 2 + 1, mask=hist_mask, other=0.0)

        gain_base = (idx * D * 2).to(tl.int64)
        gr = tl.load(delay_gains_ptr + gain_base + d_off * 2,     mask=mask, other=0.0)
        gi = tl.load(delay_gains_ptr + gain_base + d_off * 2 + 1, mask=mask, other=0.0)

        acc_r += dr * gr - di * gi
        acc_i += dr * gi + di * gr

    tl.store(x_eff_ptr + row + d_off * 2,     acc_r, mask=mask)
    tl.store(x_eff_ptr + row + d_off * 2 + 1, acc_i, mask=mask)


def fused_ocns_delay(x_states, delay_gains, prime_delays):
    """
    Memory-efficient OCNS delay embedding. All delays fused into one kernel.
    Args:
        x_states:     [B, T, D, 2] float32 contiguous
        delay_gains:  [num_delays, D, 2] float32
        prime_delays: list[int] (max 8)
    Returns:
        x_eff: [B, T, D, 2] float32
    """
    B, T, D, _ = x_states.shape
    x_eff   = torch.empty_like(x_states)
    padded  = list(prime_delays) + [0] * (8 - len(prime_delays))
    BLOCK_D = triton.next_power_of_2(D)
    _ocns_delay_kernel[(B * T,)](
        x_states.contiguous(), delay_gains.contiguous(), x_eff,
        tau0=padded[0], tau1=padded[1], tau2=padded[2], tau3=padded[3],
        tau4=padded[4], tau5=padded[5], tau6=padded[6], tau7=padded[7],
        num_delays=len(prime_delays),
        T=T, D=D, BLOCK_D=BLOCK_D,
    )
    return x_eff


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

    # Fused clamp + scaled add
    s_clamped = tl.minimum(tl.maximum(s, -10.0), 10.0)
    tl.store(x_states_ptr + off, x + lr * s_clamped, mask=mask)


def fused_state_update(x_states, step, current_lr):
    """
    In-place: x_states += current_lr * clamp(step, -10, 10)
    Args:
        x_states:   any shape, float32, contiguous (modified in-place)
        step:       same shape, float32
        current_lr: float scalar
    """
    numel = x_states.numel()
    BLOCK = 1024
    grid  = ((numel + BLOCK - 1) // BLOCK,)
    _state_update_kernel[grid](
        x_states.contiguous(), step.contiguous(),
        lr=float(current_lr), numel=numel, BLOCK=BLOCK,
    )


# ============================================================
# Kernel 4: Fused Normalize + ModReLU Activation
# ============================================================
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

    # ModReLU: ReLU(|z| + bias) * (z / max(|z|, eps))
    mag          = tl.sqrt(r * r + i * i)
    safe_mag     = tl.maximum(mag, 1e-8)
    bias         = tl.load(bias_ptr + d_off, mask=mask, other=0.0)
    activated_mag = tl.maximum(mag + bias, 0.0)

    tl.store(result_ptr + row + d_off * 2,     (r / safe_mag) * activated_mag, mask=mask)
    tl.store(result_ptr + row + d_off * 2 + 1, (i / safe_mag) * activated_mag, mask=mask)


def fused_normalize_activate(output, counts, bias):
    """
    Fused: ModReLU(output / counts.clamp(min=1))
    Args:
        output: [B_T, D, 2] float32
        counts: [B_T, 1, 1] float32
        bias:   [D] float32
    Returns:
        result: [B_T, D, 2] float32
    """
    B_T, D, _ = output.shape
    result  = torch.empty_like(output)
    BLOCK_D = triton.next_power_of_2(D)
    _normalize_activate_kernel[(B_T,)](
        output.contiguous(), counts.contiguous().view(-1), bias.contiguous(), result,
        B_T=B_T, D=D, BLOCK_D=BLOCK_D,
    )
    return result
