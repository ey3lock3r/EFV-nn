import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
import bitsandbytes.nn as bnb_nn
from efv_nn.ppc_gnn import PPCNodeLayer

class ShardedPPCGraphLLM(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int = 1024, num_layers: int = 24,
                 num_experts: int = 64, local_lr: float = 0.5, lr_decay: float = 0.85, use_jacobian: bool = False):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # GPU Assignment
        self.device0 = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device1 = "cuda:1" if torch.cuda.device_count() > 1 else self.device0
        
        self.split_point = num_layers // 2
        
        # 1. Sharded Embeddings (GPU 0) - stores interleaved real [V, D, 2]
        self.embedding = nn.Embedding(vocab_size, hidden_dim * 2).to(self.device0)
        
        with torch.no_grad():
            from efv_nn.ppc_core import ComplexKaimingInitializer
            init_w = ComplexKaimingInitializer.initialize((vocab_size, hidden_dim))
            self.embedding.weight.copy_(init_w.reshape(vocab_size, hidden_dim * 2))

        # 2. Sharded Layer Blocks (Independently Compiled Islands)
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            target_device = self.device0 if i < self.split_point else self.device1
            layer = PPCNodeLayer(hidden_dim, num_experts=num_experts, local_lr=local_lr, lr_decay=lr_decay, use_jacobian=use_jacobian).to(target_device)
            
            # Island Optimization: Compile each layer individually to avoid Multi-Device Graph Breaks
            compiled_layer = torch.compile(layer, mode="reduce-overhead")
            self.layers.append(compiled_layer)

        # 3. Output Head (GPU 1)
        self.layer_norm = nn.LayerNorm(hidden_dim * 2).to(self.device1)
        self.output_head = nn.Linear(hidden_dim * 2, vocab_size, bias=True).to(self.device1)
        nn.init.zeros_(self.output_head.bias)

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Map token IDs -> interleaved-real embeddings [*, T, D, 2]."""
        out_flat = self.embedding(input_ids)
        new_shape = list(out_flat.shape[:-1]) + [self.hidden_dim, 2]
        return out_flat.view(*new_shape)

    def forward(self, input_ids: torch.Tensor, local_iters: int = 8):
        input_ids = input_ids.to(self.device0)
        x = self.embed(input_ids) # [B, T, D, 2]

        total_iters = 0
        res_energies = []
        for i, layer in enumerate(self.layers):
            if i == self.split_point:
                x = x.to(self.device1)

            # Checkpointing is removed: We have 14GB free VRAM per T4. 
            x, iters, res_norm = layer(x, local_iters)
            x = x.clone() # Isolation: Prevent CUDA Graph buffer overwrite in loops
            total_iters += iters
            res_energies.append(res_norm)

        # One single sync point at the end of the entire graph
        avg_energy = (sum(res_energies) / len(res_energies)).item()

        # Final decoding on device1
        x_flat = x.flatten(-2) # [..., 2D]
        x_norm = self.layer_norm(x_flat)
        logits = self.output_head(x_norm)
        return logits, total_iters / self.num_layers, avg_energy

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
            total_iters += iters

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
    def generate_swarm(self, input_ids: torch.Tensor, max_new_tokens: int = 50, swarm_size: int = 8, local_iters: int = 16, temperature: float = 1.0):
        device = input_ids.device
        for _ in range(max_new_tokens):
            logits, _, _ = self.swarm_forward(input_ids, swarm_size=swarm_size, local_iters=local_iters)
            next_token_logits = logits[:, -1, :] / max(1e-6, temperature)
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token.to(device)], dim=1)
            
            if next_token.item() == 128001:
                break
        return input_ids

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 50, local_iters: int = 8, temperature: float = 1.0):
        device = input_ids.device
        for _ in range(max_new_tokens):
            logits, _, _ = self.forward(input_ids, local_iters=local_iters)
            next_token_logits = logits[:, -1, :] / max(1e-6, temperature)
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token.to(device)], dim=1)
            
            if next_token.item() == 128001:
                break
        return input_ids

    @property
    def total_params(self):
        return sum(p.numel() for p in self.parameters())
