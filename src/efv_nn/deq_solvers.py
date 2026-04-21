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
    
    X[:, 0] = x_flat
    F[:, 0] = f_x_flat
    
    x = f_x
    x_flat = x.reshape(B, -1)
    
    iters_run = 1
    res_norm = torch.norm(f_x_flat - X[:, 0], dim=-1).mean()
    
    dG = torch.zeros(B, N, m, dtype=x.dtype, device=x.device)
    for k in range(1, max_iter):
        iters_run += 1
        f_x = f(x)
        f_x_flat = f_x.reshape(B, -1)
        
        res_k = f_x_flat - x_flat
        res_norm = torch.norm(res_k, dim=-1).mean()
        if res_norm < tol:
            break
            
        m_k = min(k, m)
        # Update dG in-place for current history window
        for i in range(m_k):
            idx = (k - 1 - i) % m
            G_hist = F[:, idx] - X[:, idx]
            dG[:, :, i] = res_k - G_hist
            
        # Regularization: 1e-3 is safer than 1e-4 for initial phasal resonance
        lam_stable = lam * 10 if k < 5 else lam
        dG_curr = dG[:, :, :m_k]
        dG_T = dG_curr.transpose(1, 2)
        A = torch.bmm(dG_T, dG_curr) + lam_stable * torch.eye(m_k, dtype=x.dtype, device=x.device).unsqueeze(0)
        b = torch.bmm(dG_T, res_k.unsqueeze(-1))
        
        try:
            # solve is more stable in FP32
            alpha = torch.linalg.solve(A.float(), b.float()).to(x.dtype).squeeze(-1) # [B, m_k]
        except torch._C._LinAlgError:
            alpha = torch.zeros(B, m_k, dtype=x.dtype, device=x.device)
            
        if diagnostics.debug_print_nan(alpha, f"anderson.alpha_{k}"):
            alpha = torch.zeros(B, m_k, dtype=x.dtype, device=x.device)
        
        # Alpha Clipping: Prevent extreme jumps in the state space
        alpha = torch.clamp(alpha, -1.0, 1.0)
            
        diagnostics.debug_print_nan(res_norm, f"anderson.res_norm_{k}")
            
        # --- Update History ---
        buf_idx = k % m
        X[:, buf_idx] = x_flat
        F[:, buf_idx] = f_x_flat

        # --- Next x (Triton Optimized) ---
        if _TRITON_AVAILABLE and x.is_cuda:
            # We need to pass the actual history slices. 
            # To avoid allocation, we pass the full F and let the kernel index it.
            # But alpha needs to correspond to the indices (k-1, k-2...).
            # Actually, let's just use the current order for simplicity or re-order alpha.
            # Simpler: Use Triton for the element-wise mixing part.
            x_next_flat = torch.empty_like(f_x_flat)
            # Re-order history to match alpha's expected indices for the fused kernel
            F_ordered = torch.stack([F[:, (k - 1 - i) % m] for i in range(m_k)], dim=1)
            triton_kernels.anderson_mixing(f_x_flat, F_ordered, alpha, m_k, out=x_next_flat)
        else:
            x_next_flat = f_x_flat.clone()
            for i in range(m_k):
                idx = (k - 1 - i) % m
                x_next_flat -= alpha[:, i:i+1] * (f_x_flat - F[:, idx])
        
        x_flat = x_next_flat
        x = x_flat.reshape(B, T, D, 2)
        
    return x, iters_run, res_norm


class DEQFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_in, f_solver, f_forward, gate_bias, target, setup_fn, cleanup_fn, *params):
        if setup_fn is not None: setup_fn()
        with torch.no_grad():
            z_star, iters, res_norm = f_solver(x_in, gate_bias, target)
        if cleanup_fn is not None: cleanup_fn()
        
        ctx.save_for_backward(z_star.detach(), x_in, gate_bias, target, *params)
        ctx.f_forward = f_forward
        ctx.setup_fn = setup_fn
        ctx.cleanup_fn = cleanup_fn
        return z_star.detach(), torch.tensor(float(iters), device=x_in.device), res_norm

    @staticmethod
    def backward(ctx, grad_output, grad_iters, grad_res):
        z_star, x_in, gate_bias, target, *params = ctx.saved_tensors
        f_forward = ctx.f_forward

        if hasattr(ctx, 'setup_fn') and ctx.setup_fn is not None:
            ctx.setup_fn()

        # 1. Solve (I - J^T) g = grad_output
        with torch.enable_grad():
            z_star_leaf = z_star.detach().requires_grad_(True)
            z_next = f_forward(z_star_leaf, x_in, gate_bias, target)

        def backward_f(g_curr):
            vjp = torch.autograd.grad(z_next, z_star_leaf, grad_outputs=g_curr, retain_graph=True)[0]
            
            # Adjoint Normalization: If the VJP explodes, we dampen it.
            vjp_norm = torch.linalg.norm(vjp)
            if vjp_norm > 100.0:
                vjp = vjp * (100.0 / vjp_norm)
                
            return grad_output + vjp

        g, _, _ = anderson_acceleration(backward_f, grad_output, m=5, max_iter=8, tol=1e-5)
        
        # Adjoint Safety Check
        if torch.isnan(g).any():
            g = grad_output

        grad_targets = []
        target_indices = []
        if ctx.needs_input_grad[0]: grad_targets.append(x_in); target_indices.append(0)
        if ctx.needs_input_grad[3]: grad_targets.append(gate_bias); target_indices.append(3)
        if ctx.needs_input_grad[4]: grad_targets.append(target); target_indices.append(4)
        
        param_start_idx = 7 # shifted by 1 because of setup_fn
        for i, p in enumerate(params):
            if ctx.needs_input_grad[param_start_idx + i]:
                grad_targets.append(p)
                target_indices.append(param_start_idx + i)
        
        grads = torch.autograd.grad(z_next, grad_targets, grad_outputs=g, retain_graph=True, allow_unused=True)
        
        if hasattr(ctx, 'cleanup_fn') and ctx.cleanup_fn is not None:
            ctx.cleanup_fn()

        res = [None] * len(ctx.needs_input_grad)
        for i, g_out in zip(target_indices, grads): 
            res[i] = g_out
            
        return tuple(res)

