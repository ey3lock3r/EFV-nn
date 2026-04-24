import torch
import os
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'
import torch.nn.functional as F
import wandb
from tqdm import tqdm
import time

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
            
            # Forward with Modern AMP
            with torch.amp.autocast('cuda'):
                # V3 Model returns: logits, avg_iters, avg_energy, layer_energies, aux_loss, sg_penalty
                out = model(x, local_iters=local_iterations)
                if isinstance(out, tuple) and len(out) >= 5:
                    logits = out[0]
                    avg_iters = out[1]
                    avg_energy = out[2]
                    aux_loss = out[4] if len(out) > 4 else torch.tensor(0.0, device=out[0].device)
                    sg_penalty = out[5] if len(out) > 5 else torch.tensor(0.0, device=out[0].device)
                else:
                    logits = out
                    avg_iters, avg_energy = 0, 0.0
                    aux_loss = torch.tensor(0.0, device=logits.device)
                    sg_penalty = torch.tensor(0.0, device=logits.device)

                B, T, V = logits.shape
                loss = F.cross_entropy(logits.reshape(B * T, V), y.reshape(B * T)) + 0.01 * aux_loss + sg_penalty
            
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
                throughput = (B * T) / (time.time() - start_time)
                if wandb.run is not None:
                    wandb.log({
                        "train/loss": loss_val,
                        "train/ppl": ppl,
                        "train/tokens_per_sec": throughput,
                        "train/step": step,
                        "train/energy": energy_val,
                        "train/avg_iters": avg_iters,
                    })
                pbar.set_postfix({"loss": f"{loss_val:.4f}", "E": f"{energy_val:.3f}"})

    print("\nTraining Complete.")
