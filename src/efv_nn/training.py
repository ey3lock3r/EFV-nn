import torch
import os
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'
import torch.nn.functional as F
import wandb
from tqdm import tqdm
import time

def restore_optimizer_state(optimizer, old_model, new_model):
    """
    Re-keys optimizer state from old model parameter objects to new model parameter objects,
    matched by parameter name. Moves tensors to the correct device.

    Call this after: new_model.load_state_dict(checkpoint['model']).
    """
    old_state_by_name = {
        name: optimizer.state[param]
        for name, param in old_model.named_parameters()
        if param in optimizer.state
    }
    for name, new_param in new_model.named_parameters():
        if name in old_state_by_name:
            state = old_state_by_name[name]
            optimizer.state[new_param] = {
                k: v.to(new_param.device) if isinstance(v, torch.Tensor) else v
                for k, v in state.items()
            }

def lr_floor_guard(state: dict, lr_init: float, max_halvings: int = 3, recovery_factor: float = 0.1) -> dict:
    """
    Prevents the FSM DIVERGING state from exponentially decaying LR into stall.

    Call this each time the FSM would halve the LR. If halving_count reaches
    max_halvings, resets LR to lr_init * recovery_factor and resets the counter.

    Args:
        state: dict with keys 'lr' and 'halving_count'.
        lr_init: The original starting LR for the current phase.
        max_halvings: Max consecutive halvings before recovery.
        recovery_factor: Recovery LR = lr_init * recovery_factor.

    Returns:
        Updated state dict.
    """
    state = dict(state)  # don't mutate caller's dict
    state['halving_count'] = state.get('halving_count', 0) + 1
    if state['halving_count'] > max_halvings:
        state['lr'] = lr_init * recovery_factor
        state['halving_count'] = 0
    return state


def train_ppc_sharded(model, dataloader, lr=1e-4, epochs=1, local_iterations=2):
    """
    Sharded Training Loop with GradScaler (AMP) for Dual T4 GPUs.
    Synchronized with ppc_gnn_v2.ipynb best practices.
    """
    import bitsandbytes as bnb
    import gc
    import torch.utils
    
    # 0. Nuclear Memory Reset
    gc.collect()
    torch.cuda.empty_cache()

    # 1. Setup Paged Optimizer (PagedAdamW8bit allows massive FP32 master states to spill to CPU RAM)
    optimizer = bnb.optim.PagedAdamW8bit([
        {'params': model.embedding.parameters(), 'lr': lr},
        {'params': model.layers.parameters(), 'lr': lr, 'weight_decay': 1e-2}, 
        {'params': model.output_head.parameters(), 'lr': lr},
        {'params': model.layer_norm.parameters(), 'lr': lr},
    ])
    
    # 2. Modern GradScaler for AMP stability (PyTorch 2.4+)
    scaler = torch.amp.GradScaler('cuda')

    device1 = model.device1
    model.train()
    
    step = 0
    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch+1} ---")
        pbar = tqdm(dataloader, desc="Training")
        
        for batch in pbar:
            step += 1
            start_time = time.time()

            # Input/Target management
            x = batch[:, :-1]
            y = batch[:, 1:].to(device1)

            optimizer.zero_grad()

            t_start_compute = time.time()
            # Forward with Modern AMP
            with torch.amp.autocast('cuda'):
                # V3 Model returns: logits, avg_iters, avg_energy, layer_energies, aux_loss, sg_penalty
                logits, avg_iters, avg_energy, _, aux_loss, sg_penalty = model(x, local_iters=local_iterations)

                B, T, V = logits.shape
                task_loss = F.cross_entropy(logits.reshape(B * T, V), y.reshape(B * T))
                loss = task_loss + 0.01 * aux_loss + sg_penalty
            t_compute = time.time() - t_start_compute

            # Scaled Backward & Optim
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scaler.update()

            if step % 50 == 0:
                torch.cuda.empty_cache()

            # Logging — batch all .item() / CPU syncs here, never in the hot path above
            if step % 10 == 0:
                loss_val   = loss.item()   # single GPU→CPU sync; triggers just once per 10 steps
                ppl        = torch.exp(loss).item()
                energy_val = avg_energy.item() if isinstance(avg_energy, torch.Tensor) else avg_energy
                elapsed_total = time.time() - start_time
                try:
                    grad_norm = sum(
                        p.grad.norm().item() ** 2
                        for p in model.parameters() if p.grad is not None
                    ) ** 0.5
                except Exception:
                    grad_norm = 0.0
                if wandb.run is not None:
                    wandb.log({
                        "train/loss": loss_val,
                        "train/ppl": ppl,
                        "train/tokens_per_sec_total": (B * T) / max(elapsed_total, 1e-6),
                        "train/tokens_per_sec_compute": (B * T) / max(t_compute, 1e-6),
                        "train/step": step,
                        "train/energy": energy_val,
                        "train/avg_iters": avg_iters,
                        "train/grad_norm": grad_norm,
                        "train/scaler_scale": scaler.get_scale(),
                    })
                pbar.set_postfix({"loss": f"{loss_val:.4f}", "E": f"{energy_val:.3f}"})

            # Reset timer AFTER logging/checkpoint work to exclude overhead from next-step measurement
            start_time = time.time()

    print("\nTraining Complete.")
