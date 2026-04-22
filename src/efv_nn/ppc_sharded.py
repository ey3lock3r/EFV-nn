import torch
import torch.nn as nn
import os
from efv_nn import diagnostics
from torch.utils.checkpoint import checkpoint
import bitsandbytes.nn as bnb_nn
from efv_nn import ppc_gnn, ppc_core

class ShardedPPCGraphLLM(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int = 1024, num_layers: int = 24,
                 num_experts: int = 64, local_lr: float = 0.05, lr_decay: float = 0.85,
                 use_jacobian: bool = False, prime_delays=(1, 2, 3, 5), use_triton: bool = True):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # GPU Assignment
        self.device0 = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device1 = "cuda:1" if torch.cuda.device_count() > 1 else self.device0
        self.split_point = num_layers // 2

        # Cache device objects to avoid string comparisons in the hot loop
        self.d0 = torch.device(self.device0)
        self.d1 = torch.device(self.device1)
        
        # 1. Sharded Embeddings (GPU 0) - stores interleaved real [V, D, 2]
        self.embedding = nn.Embedding(vocab_size, hidden_dim * 2).to(self.d0)

        with torch.no_grad():
            # Correctly initialise the interleaved real pair
            init_w = ppc_core.ComplexKaimingInitializer.initialize((vocab_size, hidden_dim))
            self.embedding.weight.copy_(init_w.reshape(vocab_size, hidden_dim * 2))

        # Pre-assign target devices for each layer to avoid conditional logic in loop
        self.layer_target_devices = [self.d0 if i < self.split_point else self.d1 for i in range(num_layers)]

        # 2. Sharded Layer Blocks
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            target_device = self.layer_target_devices[i]
            layer = ppc_gnn.PPCNodeLayer(
                hidden_dim, num_experts=num_experts, local_lr=local_lr,
                lr_decay=lr_decay, use_jacobian=use_jacobian,
                prime_delays=prime_delays, use_triton=use_triton,
                device=target_device
            ).to(target_device)
            self.layers.append(layer)

        # 3. Output Head (GPU 1)
        self.layer_norm = nn.LayerNorm(hidden_dim * 2).to(self.d1)
        self.output_head = nn.Linear(hidden_dim * 2, vocab_size, bias=True).to(self.d1)
        nn.init.zeros_(self.output_head.bias)

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Map token IDs -> interleaved-real embeddings [*, T, D, 2]."""
        out_flat = self.embedding(input_ids)
        new_shape = list(out_flat.shape[:-1]) + [self.hidden_dim, 2]
        return out_flat.view(*new_shape)

    def forward(self, input_ids: torch.Tensor, local_iters: int = 8, rolling_energy: float = None):
        input_ids = input_ids.to(self.d0)
        x = self.embed(input_ids) # [B, T, D, 2]

        total_iters = 0
        res_energies = []
        for i, layer in enumerate(self.layers):
            target_device = self.layer_target_devices[i]
            
            # Optimized Zero-G Guard: Comparison of device objects is much faster than strings
            if x.device != target_device:
                x = x.to(target_device)
            
            x, iters, res_norm = layer(x, local_iters, rolling_energy=rolling_energy)

            # Isolation: Prevent CUDA Graph buffer overwrite in loops
            # We use a non-blocking clone if possible, but standard clone is safe.
            x = x.clone() 
            total_iters += iters.item()
            res_energies.append(res_norm.clone())

        # Move all scalars to device1 before sum to avoid cross-device error
        layer_energies = torch.stack([e.to(self.d1) for e in res_energies])
        avg_energy = layer_energies.mean()
        
        diagnostics.debug_print_nan(avg_energy, "avg_energy")

        # Final decoding on device1
        x_flat = x.flatten(-2) # [..., 2D]
        x_norm = self.layer_norm(x_flat)
        logits = self.output_head(x_norm)
        return logits, total_iters / self.num_layers, avg_energy, layer_energies

    @torch.no_grad()
    def swarm_forward(self, input_ids: torch.Tensor, swarm_size: int = 8, local_iters: int = 16):
        """
        Execute parallel ghost-state convergence to find the lowest-energy resonance.
        """
        input_ids = input_ids.to(self.device0)
        # 1. Embed and Expand into Swarm
        x = self.embed(input_ids) # [B, T, D, 2]
        B, T, D, _ = x.shape
        
        # x_swarm: [B * S, T, D, 2]
        x_swarm = x.repeat_interleave(swarm_size, dim=0)
        
        # 2. Phase Perturbation: Add tiny noise to phase ghosts
        # Using 1e-4 noise to keep ghosts near the embedding but allow separate search paths
        x_swarm[..., 1] += torch.randn_like(x_swarm[..., 1]) * 1e-4

        total_iters = 0
        res_energies = torch.zeros(B * swarm_size, device=self.device1)
        
        curr_x = x_swarm
        for i, layer in enumerate(self.layers):
            if i == self.split_point:
                curr_x = curr_x.to(self.device1)
            
            curr_x, iters, res_norm = layer(curr_x, local_iters)
            curr_x = curr_x.clone() # Isolation: Prevent CUDA Graph buffer overwrite in loops
            # res_norm is usually scalar mean, but for swarm we need per-sample
            # In swarm mode, we'll re-calculate norm locally for selection
            total_iters += iters.item()

        # 3. Resonance Selection (Pick the ghost with the deepest convergence)
        # Reshape to [B, S, T, D, 2]
        curr_x = curr_x.reshape(B, swarm_size, T, D, 2)
        
        # Calculate Energy per Ghost [B, S]
        # We look at the final phasal stability of each ghost
        # curr_x[..., 1] is [B, S, T, D]
        final_energy = torch.norm(curr_x[..., 1], dim=(-2, -1)) # Result: [B, S]
        
        winner_indices = torch.argmin(final_energy, dim=1) # [B]
        
        # Gather Winners [B, T, D, 2]
        winners = torch.stack([curr_x[b, winner_indices[b]] for b in range(B)], dim=0)
        
        x_flat = winners.flatten(-2)
        logits = self.output_head(self.layer_norm(x_flat))
        
        # Return winner's mean energy as the 3rd value
        winner_energy = final_energy[0, winner_indices[0]].item() 
        
        return logits, total_iters / self.num_layers, winner_energy

    @torch.no_grad()
    def generate_swarm(self, input_ids: torch.Tensor, max_new_tokens: int = 50, swarm_size: int = 8, local_iters: int = 16, temperature: float = 1.0, top_k: int = 40):
        device = input_ids.device
        for _ in range(max_new_tokens):
            logits, _, _ = self.swarm_forward(input_ids, swarm_size=swarm_size, local_iters=local_iters)
            next_token_logits = logits[:, -1, :] / max(1e-6, temperature)
            
            # Top-K Sampling to prevent greedy repetition
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = -float('Inf')
            
            probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            input_ids = torch.cat([input_ids, next_token.to(device)], dim=1)
            
            if next_token.item() == 128001:
                break
        return input_ids

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 50, local_iters: int = 8, temperature: float = 1.0, top_k: int = 40):
        device = input_ids.device
        for _ in range(max_new_tokens):
            logits, _, _, _ = self.forward(input_ids, local_iters=local_iters)
            next_token_logits = logits[:, -1, :] / max(1e-6, temperature)
            
            # Top-K Sampling
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = -float('Inf')
                
            probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            input_ids = torch.cat([input_ids, next_token.to(device)], dim=1)
            
            if next_token.item() == 128001:
                break
        return input_ids

    @property
    def total_params(self):
        return sum(p.numel() for p in self.parameters())
