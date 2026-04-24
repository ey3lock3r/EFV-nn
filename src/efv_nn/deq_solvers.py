import torch
import torch.nn as nn
from typing import Callable, Tuple
from efv_nn import diagnostics

try:
    from . import triton_kernels
    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False

def anderson_acceleration(f: Callable, x0: torch.Tensor, m: int = 5, lam: float = 1e-4, max_iter: int = 50, tol: float = 1e-5) -> Tuple[torch.Tensor, int, torch.Tensor]:
    """
    Anderson Acceleration for Fixed Point Iteration.
    Solves x = f(x).
    """
    B, T, D, C = x0.shape
    assert C == 2, "Anderson expects interleaved complex tensors [..., 2]"
    N = T * D * 2
    
    x = x0.clone()
    X = torch.zeros(B, m, N, dtype=x.dtype, device=x.device)
    F = torch.zeros(B, m, N, dtype=x.dtype, device=x.device)
    
    x_flat = x.reshape(B, -1)
    f_x = f(x)
    f_x_flat = f_x.reshape(B, -1)
    # NaN Siphon: sanitize before storing in history to prevent cascade poisoning
    f_x_flat = torch.nan_to_num(f_x_flat, nan=0.0, posinf=10.0, neginf=-10.0)

    X[:, 0] = x_flat
    F[:, 0] = f_x_flat

    x = f_x_flat.reshape(B, T, D, 2)
    x_flat = x.reshape(B, -1)
    
    iters_run = 1
    res_norm = torch.norm(f_x_flat - X[:, 0], dim=-1).mean()
    
    dG = torch.zeros(B, N, m, dtype=x.dtype, device=x.device)
    # Pre-allocate F_ordered buffer for the Triton mixing path
    F_ordered_buf = torch.empty(B, m, N, dtype=x.dtype, device=x.device) if (_TRITON_AVAILABLE and x.is_cuda) else None
    x_next_flat = torch.empty_like(f_x_flat) if (_TRITON_AVAILABLE and x.is_cuda) else None

    best_res = float('inf')
    for k in range(1, max_iter):
        iters_run += 1
        f_x = f(x)
        f_x_flat = f_x.reshape(B, -1)

        # NaN Siphon: on poisoned steps, keep x unchanged and mirror previous history slot
        # to prevent gaps in the circular buffer that corrupt dG indexing.
        nan_step = torch.isnan(f_x_flat).any()
        if nan_step:
            buf_idx = k % m
            X[:, buf_idx] = x_flat
            F[:, buf_idx] = F[:, (k - 1) % m]  # repeat previous valid F entry
            continue

        res_k = f_x_flat - x_flat
        res_norm = torch.norm(res_k, dim=-1).mean()
        x_norm_val = torch.norm(x_flat, dim=-1).mean().clamp(min=1e-8)
        rel_res = res_norm / x_norm_val
        if res_norm < tol or rel_res < tol * 10:
            break

        m_k = min(k, m)
        # Update dG in-place for current history window
        for i in range(m_k):
            idx = (k - 1 - i) % m
            G_hist = F[:, idx] - X[:, idx]
            dG[:, :, i] = res_k - G_hist

        dG_curr = dG[:, :, :m_k]  # [B, N, m_k]
        try:
            # lstsq uses SVD — condition number is κ(dG) not κ(dGᵀdG). No manual regularisation needed.
            result = torch.linalg.lstsq(dG_curr.float(), res_k.float().unsqueeze(-1))
            alpha = result.solution.to(x.dtype).squeeze(-1)  # [B, m_k]
        except (torch._C._LinAlgError, RuntimeError):
            alpha = torch.zeros(B, m_k, dtype=x.dtype, device=x.device)

        diagnostics.debug_print_nan(alpha, f"anderson.alpha_{k}")
        if torch.isnan(alpha).any():
            alpha = torch.zeros(B, m_k, dtype=x.dtype, device=x.device)

        # Alpha Clipping: Prevent extreme jumps in the state space
        alpha = torch.clamp(alpha, -1.0, 1.0)

        diagnostics.debug_print_nan(res_norm, f"anderson.res_norm_{k}")

        # --- Update History (monotonic: only write F if this iterate improved residual) ---
        buf_idx = k % m
        current_res = res_norm.item()
        X[:, buf_idx] = x_flat
        f_sanitized = f_x_flat  # already confirmed non-NaN above
        if current_res <= best_res * 1.5:
            F[:, buf_idx] = f_sanitized
            best_res = min(best_res, current_res)

        # --- Next x (Triton Optimized) ---
        if _TRITON_AVAILABLE and x.is_cuda:
            # Re-order history into pre-allocated buffer to avoid torch.stack list allocation
            for i in range(m_k):
                F_ordered_buf[:, i] = F[:, (k - 1 - i) % m]
            triton_kernels.anderson_mixing(f_x_flat, F_ordered_buf[:, :m_k], alpha, m_k, out=x_next_flat)
            x_flat = torch.nan_to_num(x_next_flat, nan=0.0, posinf=10.0, neginf=-10.0)
        else:
            x_next_flat = f_x_flat.clone()
            for i in range(m_k):
                idx = (k - 1 - i) % m
                x_next_flat -= alpha[:, i:i+1] * (f_x_flat - F[:, idx])
            x_flat = torch.nan_to_num(x_next_flat, nan=0.0, posinf=10.0, neginf=-10.0)
        x = x_flat.reshape(B, T, D, 2)
        
    return x, iters_run, res_norm


class DEQFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_in, f_solver, f_forward, gate_bias, target, setup_fn, cleanup_fn, adjoint_cache, fp16_modules, *params):
        if setup_fn is not None: setup_fn()
        try:
            with torch.no_grad():
                z_star, iters, res_norm = f_solver(x_in, gate_bias, target)
        finally:
            if cleanup_fn is not None: cleanup_fn()

        ctx.save_for_backward(z_star.detach(), x_in, gate_bias, target, adjoint_cache, *params)
        ctx.f_forward = f_forward
        ctx.setup_fn = setup_fn
        ctx.cleanup_fn = cleanup_fn
        ctx.fp16_modules = fp16_modules  # list of modules with FP16 params and FP32 caches
        return z_star.detach(), torch.tensor(float(iters), device=x_in.device), res_norm

    @staticmethod
    def backward(ctx, grad_output, grad_iters, grad_res):
        z_star, x_in, gate_bias, target, adjoint_cache, *params = ctx.saved_tensors
        f_forward = ctx.f_forward
        fp16_modules = ctx.fp16_modules

        try:
            # 1. Solve (I - J^T) g = grad_output
            # setup_fn MUST run inside enable_grad so that _wr_f32/_wi_f32 are created
            # with requires_grad=True (backward engine runs under no_grad by default).
            with torch.enable_grad():
                if ctx.setup_fn is not None:
                    ctx.setup_fn()
                z_star_leaf = z_star.detach().requires_grad_(True)
                z_next = f_forward(z_star_leaf, x_in, gate_bias, target)

            def backward_f(g_curr):
                vjp = torch.autograd.grad(z_next, z_star_leaf, grad_outputs=g_curr, retain_graph=True)[0]

                # Adjoint Normalization: If the VJP explodes, we dampen it.
                vjp_norm = torch.linalg.norm(vjp)
                if vjp_norm > 100.0:
                    vjp = vjp * (100.0 / vjp_norm)

                return grad_output + vjp

            # --- ADJOINT WARM-STARTING ---
            # Blend 90% cache + 10% grad_output to maintain directional diversity.
            # Guard: only use cache if shape matches AND it's non-zero (not first backward pass).
            if adjoint_cache.shape == grad_output.shape and adjoint_cache.norm() > 0:
                g0 = 0.9 * adjoint_cache.detach() + 0.1 * grad_output
            else:
                g0 = grad_output

            # --- ADJOINT EARLY-EXIT (Dynamic Tolerance) ---
            # We relax the tolerance if the initial residual is high, allowing faster convergence
            # in early phasal training.
            with torch.no_grad():
                res0 = torch.norm(backward_f(g0) - g0).item()
                # 1e-3 is a safe target for adjoint stability
                dynamic_tol = max(1e-5, res0 * 0.1)

            g, _, _ = anderson_acceleration(backward_f, g0, m=5, max_iter=12, tol=dynamic_tol)

            # Update cache in-place for the next step
            if adjoint_cache.shape == g.shape:
                adjoint_cache.copy_(g.detach())

            # Adjoint Safety Check
            if torch.isnan(g).any():
                g = grad_output

            res = [None] * len(ctx.needs_input_grad)

            # Compute grad for inputs and FP32 parameters via autograd.grad.
            # FP16 parameters cannot be targets of autograd.grad inside a backward pass.
            # Instead, diff w.r.t. their FP32 proxy caches (_wr_f32/_wi_f32) and
            # backprop through the FP16→FP32 cast to recover the FP16 param grad.
            grad_targets = []
            target_indices = []
            fp32_proxy_targets = []  # (fp32_tensor, fp16_param) pairs
            # Non-leaf intermediates (x_in, gate_bias, target) come back from saved_tensors
            # with requires_grad=False — guard before adding to avoid "does not require grad".
            if ctx.needs_input_grad[0] and x_in.requires_grad:
                grad_targets.append(x_in); target_indices.append(0)
            if ctx.needs_input_grad[3] and gate_bias is not None and gate_bias.requires_grad:
                grad_targets.append(gate_bias); target_indices.append(3)
            if ctx.needs_input_grad[4] and target.requires_grad:
                grad_targets.append(target); target_indices.append(4)

            param_start_idx = 9  # shifted by 2: adjoint_cache + fp16_modules placeholder
            for i, p in enumerate(params):
                if ctx.needs_input_grad[param_start_idx + i] and p.requires_grad:
                    if p.dtype == torch.float16:
                        # Handle via FP32 proxy — collected separately below
                        pass
                    else:
                        grad_targets.append(p)
                        target_indices.append(param_start_idx + i)

            # Add FP32 proxies for FP16 params from fp16_modules.
            # Skip if already FP32 (e.g. after model.float()) — direct param path handles it.
            for mod in (fp16_modules or []):
                if (hasattr(mod, '_wr_f32') and mod._wr_f32 is not None
                        and mod.experts_weight_real.dtype == torch.float16):
                    fp32_proxy_targets.append((mod._wr_f32, mod.experts_weight_real, '[...,0]'))
                if (hasattr(mod, '_wi_f32') and mod._wi_f32 is not None
                        and mod.experts_weight_real.dtype == torch.float16):
                    fp32_proxy_targets.append((mod._wi_f32, mod.experts_weight_real, '[...,1]'))

            all_targets = grad_targets + [t for t, _, _ in fp32_proxy_targets]
            grads = torch.autograd.grad(z_next, all_targets, grad_outputs=g, retain_graph=True, allow_unused=True)

            for idx, g_out in zip(target_indices, grads[:len(grad_targets)]):
                res[idx] = g_out

            # Backprop FP32 proxy grads through FP16→FP32 cast to accumulate on FP16 param
            for (fp32_t, fp16_p, slice_str), g_fp32 in zip(fp32_proxy_targets, grads[len(grad_targets):]):
                if g_fp32 is None:
                    continue
                if fp16_p.grad is None:
                    fp16_p.grad = torch.zeros_like(fp16_p)
                _FP16_MAX = 60000.0
                g_fp32_safe = g_fp32.clamp(-_FP16_MAX, _FP16_MAX)
                if slice_str == '[...,0]':
                    fp16_p.grad[..., 0].add_(g_fp32_safe.half())
                else:
                    fp16_p.grad[..., 1].add_(g_fp32_safe.half())

        finally:
            if ctx.cleanup_fn is not None:
                ctx.cleanup_fn()

        return tuple(res)

