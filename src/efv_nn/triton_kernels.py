"""
Triton Kernels for PPC-GNN (Prospective Predictive Coding Graph Neural Network).

These kernels fuse elementwise operations in the PPCNodeLayer's iterative
convergence loop, eliminating torch.compile cold-start and reducing memory.

All kernels operate on [B, T, D, 2] interleaved-real (complex) tensors stored
contiguously in memory. The last dimension (2) holds [real, imaginary].

Usage:
    from efv_nn.triton_kernels import (
        fused_phase_rotation,
        fused_ocns_delay,
        fused_state_update,
        fused_normalize_activate,
    )
"""

import torch
import triton
import triton.language as tl


# ============================================================
# Kernel 1: Fused Phase Rotation + Target Construction
# ============================================================
# Replaces: x_prev slicing, complex rotation, target construction, stack
# Python equivalent (6 ops):
#   x_prev = x_states[:, :-1]
#   rot_r = prev_r * cos_p - prev_i * sin_p
#   rot_i = prev_r * sin_p + prev_i * cos_p
#   x_target[:, 1:] = stack([rot_r, rot_i], -1)
#   x_target[:, 0] = x_states[:, 0]

@triton.jit
def _phase_rotation_kernel(
    x_states_ptr, cos_p_ptr, sin_p_ptr, x_target_ptr,
    T: tl.constexpr, D: tl.constexpr, BLOCK_D: tl.constexpr,
):
    # Each program handles one (b, t) position across all D complex pairs
    pid = tl.program_id(0)  # ∈ [0, B*T)
    t = pid % T
    row = pid * D * 2  # flat offset into [B*T, D*2]

    d_off = tl.arange(0, BLOCK_D)
    mask = d_off < D
    r_idx = row + d_off * 2       # real parts
    i_idx = row + d_off * 2 + 1   # imaginary parts

    if t == 0:
        # First token: copy directly
        r = tl.load(x_states_ptr + r_idx, mask=mask, other=0.0)
        i = tl.load(x_states_ptr + i_idx, mask=mask, other=0.0)
        tl.store(x_target_ptr + r_idx, r, mask=mask)
        tl.store(x_target_ptr + i_idx, i, mask=mask)
    else:
        # Rotate x_states[b, t-1, d] by (cos_p, sin_p)
        prev_row = (pid - 1) * D * 2
        prev_r = tl.load(x_states_ptr + prev_row + d_off * 2, mask=mask, other=0.0)
        prev_i = tl.load(x_states_ptr + prev_row + d_off * 2 + 1, mask=mask, other=0.0)
        cos_p = tl.load(cos_p_ptr + d_off, mask=mask, other=0.0)
        sin_p = tl.load(sin_p_ptr + d_off, mask=mask, other=0.0)

        rot_r = prev_r * cos_p - prev_i * sin_p
        rot_i = prev_r * sin_p + prev_i * cos_p
        tl.store(x_target_ptr + r_idx, rot_r, mask=mask)
        tl.store(x_target_ptr + i_idx, rot_i, mask=mask)


def fused_phase_rotation(x_states, cos_p, sin_p):
    """
    Fused phase rotation + target construction.

    Args:
        x_states: [B, T, D, 2] float32 contiguous
        cos_p: [D] float32
        sin_p: [D] float32

    Returns:
        x_target: [B, T, D, 2] float32
    """
    B, T, D, _ = x_states.shape
    x_target = torch.empty_like(x_states)
    grid = (B * T,)
    # Use next power of 2 for BLOCK_D
    BLOCK_D = triton.next_power_of_2(D)
    _phase_rotation_kernel[grid](
        x_states.contiguous().data_ptr(),
        cos_p.contiguous().data_ptr(),
        sin_p.contiguous().data_ptr(),
        x_target.data_ptr(),
        T=T, D=D, BLOCK_D=BLOCK_D,
    )
    return x_target


# ============================================================
# Kernel 2: Fused OCNS Delay Embedding
# ============================================================
# Replaces: clone + Python loop over delays + slicing + complex multiply + add
# Processes ALL delays in a single kernel launch.
# Max supported delays: 8 (covers Fibonacci [1,2,3,5,8,13,21,34])

@triton.jit
def _ocns_delay_kernel(
    x_states_ptr, delay_gains_ptr, x_eff_ptr,
    # Delay taps (passed as individual constexprs for compile-time unrolling)
    tau0: tl.constexpr, tau1: tl.constexpr, tau2: tl.constexpr, tau3: tl.constexpr,
    tau4: tl.constexpr, tau5: tl.constexpr, tau6: tl.constexpr, tau7: tl.constexpr,
    num_delays: tl.constexpr,
    T: tl.constexpr, D: tl.constexpr, BLOCK_D: tl.constexpr,
):
    # Each program handles one (b, t) position
    pid = tl.program_id(0)  # ∈ [0, B*T)
    t = pid % T
    row = pid * D * 2

    d_off = tl.arange(0, BLOCK_D)
    mask = d_off < D

    # Start with copy of x_states at this position
    x_r = tl.load(x_states_ptr + row + d_off * 2, mask=mask, other=0.0)
    x_i = tl.load(x_states_ptr + row + d_off * 2 + 1, mask=mask, other=0.0)

    # Accumulate interference from each delay
    taus = (tau0, tau1, tau2, tau3, tau4, tau5, tau6, tau7)
    for idx in tl.static_range(num_delays):
        tau = taus[idx]
        if t >= tau:
            # Load history: x_states[b, t - tau, d, :]
            hist_row = (pid - tau) * D * 2
            dr = tl.load(x_states_ptr + hist_row + d_off * 2, mask=mask, other=0.0)
            di = tl.load(x_states_ptr + hist_row + d_off * 2 + 1, mask=mask, other=0.0)

            # Load gains: delay_gains[idx, d, 0/1]
            gain_base = idx * D * 2
            gr = tl.load(delay_gains_ptr + gain_base + d_off * 2, mask=mask, other=0.0)
            gi = tl.load(delay_gains_ptr + gain_base + d_off * 2 + 1, mask=mask, other=0.0)

            # Complex multiply and accumulate
            x_r += dr * gr - di * gi
            x_i += dr * gi + di * gr

    tl.store(x_eff_ptr + row + d_off * 2, x_r, mask=mask)
    tl.store(x_eff_ptr + row + d_off * 2 + 1, x_i, mask=mask)


def fused_ocns_delay(x_states, delay_gains, prime_delays):
    """
    Memory-efficient OCNS delay embedding. All delays fused into one kernel.

    Args:
        x_states: [B, T, D, 2] float32 contiguous
        delay_gains: [num_delays, D, 2] float32
        prime_delays: list of int delay values (max 8)

    Returns:
        x_eff: [B, T, D, 2] float32
    """
    B, T, D, _ = x_states.shape
    x_eff = torch.empty_like(x_states)

    # Pad delays to 8 slots (unused slots = 0, won't be iterated due to num_delays)
    padded = list(prime_delays) + [0] * (8 - len(prime_delays))

    grid = (B * T,)
    BLOCK_D = triton.next_power_of_2(D)
    _ocns_delay_kernel[grid](
        x_states.contiguous().data_ptr(),
        delay_gains.contiguous().data_ptr(),
        x_eff.data_ptr(),
        tau0=padded[0], tau1=padded[1], tau2=padded[2], tau3=padded[3],
        tau4=padded[4], tau5=padded[5], tau6=padded[6], tau7=padded[7],
        num_delays=len(prime_delays),
        T=T, D=D, BLOCK_D=BLOCK_D,
    )
    return x_eff


# ============================================================
# Kernel 3: Fused State Update (Clamp + Scaled Add)
# ============================================================
# Replaces: torch.clamp(step, -10, 10) + x_states.add_(clamped, alpha=lr)
# Operates fully in-place on x_states.

@triton.jit
def _state_update_kernel(
    x_states_ptr, step_ptr, lr,
    numel: tl.constexpr, BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    mask = off < numel

    x = tl.load(x_states_ptr + off, mask=mask, other=0.0)
    s = tl.load(step_ptr + off, mask=mask, other=0.0)

    # Fused clamp + scaled add
    s_clamped = tl.minimum(tl.maximum(s, -10.0), 10.0)
    x += lr * s_clamped

    tl.store(x_states_ptr + off, x, mask=mask)


def fused_state_update(x_states, step, current_lr):
    """
    In-place: x_states += current_lr * clamp(step, -10, 10)

    Args:
        x_states: any shape, float32, contiguous (modified in-place)
        step: same shape as x_states, float32
        current_lr: float scalar
    """
    numel = x_states.numel()
    BLOCK = 1024
    grid = ((numel + BLOCK - 1) // BLOCK,)
    _state_update_kernel[grid](
        x_states.contiguous().data_ptr(),
        step.contiguous().data_ptr(),
        current_lr,
        numel=numel, BLOCK=BLOCK,
    )


# ============================================================
# Kernel 4: Fused Normalize + ModReLU Activation
# ============================================================
# Replaces: output / counts.clamp(min=1) + ModReLU forward
# ModReLU(z) = ReLU(|z| + bias) * (z / max(|z|, eps))

@triton.jit
def _normalize_activate_kernel(
    output_ptr, counts_ptr, bias_ptr, result_ptr,
    B_T: tl.constexpr, D: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)  # ∈ [0, B_T)
    d_off = tl.arange(0, BLOCK_D)
    mask = d_off < D

    row = pid * D * 2

    # Load count for this token (counts shape: [B_T, 1, 1])
    count = tl.load(counts_ptr + pid)
    count = tl.maximum(count, 1.0)

    # Load and normalize
    r = tl.load(output_ptr + row + d_off * 2, mask=mask, other=0.0) / count
    i = tl.load(output_ptr + row + d_off * 2 + 1, mask=mask, other=0.0) / count

    # ModReLU: magnitude, activation, rescale
    mag = tl.sqrt(r * r + i * i)
    safe_mag = tl.maximum(mag, 1e-8)
    bias = tl.load(bias_ptr + d_off, mask=mask, other=0.0)
    activated_mag = tl.maximum(mag + bias, 0.0)  # ReLU(|z| + b)

    # Unit phase * activated magnitude
    out_r = (r / safe_mag) * activated_mag
    out_i = (i / safe_mag) * activated_mag

    tl.store(result_ptr + row + d_off * 2, out_r, mask=mask)
    tl.store(result_ptr + row + d_off * 2 + 1, out_i, mask=mask)


def fused_normalize_activate(output, counts, bias):
    """
    Fused: ModReLU(output / counts.clamp(min=1))

    Args:
        output: [B_T, D, 2] float32
        counts: [B_T, 1, 1] float32
        bias: [D] float32 (ModReLU bias)

    Returns:
        result: [B_T, D, 2] float32
    """
    B_T, D, _ = output.shape
    result = torch.empty_like(output)
    BLOCK_D = triton.next_power_of_2(D)
    _normalize_activate_kernel[(B_T,)](
        output.contiguous().data_ptr(),
        counts.contiguous().view(-1).data_ptr(),
        bias.contiguous().data_ptr(),
        result.data_ptr(),
        B_T=B_T, D=D, BLOCK_D=BLOCK_D,
    )
    return result
