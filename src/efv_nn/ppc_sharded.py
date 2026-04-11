import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
import bitsandbytes.nn as bnb_nn
from efv_nn.ppc_gnn import PPCNodeLayer

class ShardedPPCGraphLLM(nn.Module):
    """
    3.2B+ Parameter Prospective Predictive Coding LLM.
    
    Optimized for Dual T4 GPUs (Kaggle):
    - Shards layers across cuda:0 and cuda:1.
    - Uses Expert Choice MoE to scale parameters with sub-linear compute.
    - Employs Complex Latent Space and Gradient Checkpointing.
    """
    def __init__(self, vocab_size: int, hidden_dim: int = 1024, num_layers: int = 24,
                 num_experts: int = 64, local_lr: float = 0.5, lr_decay: float = 0.85, use_jacobian: bool = False):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # GPU Assignment
        self.device0 = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device1 = "cuda:1" if torch.cuda.device_count() > 1 else self.device0
        
        # Balanced split across 24 layers
        self.split_point = num_layers // 2
        
        # 1. Sharded Embeddings (GPU 0)
        self.embedding = nn.Embedding(vocab_size, hidden_dim * 2).to(self.device0)
        
        # Initialise with ComplexKaiming
        with torch.no_grad():
            from efv_nn.ppc_core import ComplexKaimingInitializer
            complex_w = ComplexKaimingInitializer.initialize((vocab_size, hidden_dim))
            real_w = torch.view_as_real(complex_w).reshape(vocab_size, hidden_dim * 2)
            self.embedding.weight.copy_(real_w)

        # 2. Sharded Layer Blocks
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            target_device = self.device0 if i < self.split_point else self.device1
            self.layers.append(
                PPCNodeLayer(hidden_dim, num_experts=num_experts, local_lr=local_lr, lr_decay=lr_decay, use_jacobian=use_jacobian).to(target_device)
            )

        # 3. Output Head (GPU 1)
        self.layer_norm = nn.LayerNorm(hidden_dim * 2).to(self.device1)
        self.output_head = nn.Linear(hidden_dim * 2, vocab_size, bias=True).to(self.device1)
        nn.init.zeros_(self.output_head.bias)

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Map token IDs -> complex embeddings on device0."""
        out_real_flat = self.embedding(input_ids)
        new_shape = list(out_real_flat.shape[:-1]) + [self.hidden_dim, 2]
        out_real_struct = out_real_flat.view(*new_shape)
        return torch.view_as_complex(out_real_struct)

    def forward(self, input_ids: torch.Tensor, local_iters: int = 8):
        # Move input to device0
        input_ids = input_ids.to(self.device0)
        x = self.embed(input_ids)

        # Issue 1 Fix: track total_iters to return avg_iters
        total_iters = 0
        for i, layer in enumerate(self.layers):
            if i == self.split_point:
                x = x.to(self.device1)

            # Issue 5 Fix: autocast must NOT wrap checkpoint — it doesn't propagate
            # into the recompute pass with use_reentrant=False, causing dtype mismatches.
            # autocast is applied INSIDE PPCNodeLayer / ExpertChoiceMoEMatcher instead.
            if self.training:
                x, iters = checkpoint(layer, x, local_iters, use_reentrant=False)
            else:
                x, iters = layer(x, local_iters)
            total_iters += iters

        # Final decoding on device1
        x_real_imag = torch.cat([x.real, x.imag], dim=-1)
        x_norm = self.layer_norm(x_real_imag)
        logits = self.output_head(x_norm)
        # Issue 1 Fix: return (logits, avg_iters) matching notebook training loop expectation
        return logits, total_iters / self.num_layers

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 50, local_iters: int = 8, temperature: float = 1.0):
        """
        Auto-regressive generation using the sharded architecture.
        """
        device = input_ids.device
        for _ in range(max_new_tokens):
            # 1. Get logits from the sharded forward pass
            logits, _ = self.forward(input_ids, local_iters=local_iters)
            
            # 2. Focus on the last token's logits
            next_token_logits = logits[:, -1, :] / max(1e-6, temperature)
            
            # 3. Greedy sampling (argmax)
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # 4. Append to input sequence
            input_ids = torch.cat([input_ids, next_token.to(device)], dim=1)
            
            # 5. Stop if we hit the EOS token (shorthand for 128001 in Llama-3 vocab)
            if next_token.item() == 128001:
                break
                
        return input_ids

    @property
    def total_params(self):
        return sum(p.numel() for p in self.parameters())
