import torch
import torch.nn as nn
import math
from efv_nn.ppc_core import ExpertChoiceMoEMatcher

class PPCNodeLayer(nn.Module):
    def __init__(self, hidden_dim, num_experts=4, local_lr=0.5, lr_decay=0.85, tolerance=1e-3, use_jacobian=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.base_local_lr = max(0.0, min(0.99, local_lr))
        self.lr_decay = lr_decay
        self.tolerance = tolerance
        self.use_jacobian = use_jacobian
        self.register_buffer('phase_offset', torch.rand(hidden_dim) * 2 * math.pi)
        self.moe = ExpertChoiceMoEMatcher(hidden_dim, num_experts)
        
    def forward(self, x_stream, local_iterations=8):
        """
        x_stream: [B, T, hidden_dim] batched sequence.
        """
        unbatched = x_stream.dim() == 2
        if unbatched:
            x_stream = x_stream.unsqueeze(0)

        B, T, D = x_stream.shape
        iters_run = 0
        
        # 1. Local Convergence (Frozen fixed point search)
        with torch.no_grad():
            x_states = x_stream.clone().detach()
            x_target_frozen = torch.zeros_like(x_states)
            x_target_frozen[:, 1:, :] = x_states[:, :-1, :] * torch.exp(1j * self.phase_offset)
            x_target_frozen[:, 0, :] = x_states[:, 0, :]
            
            current_lr = self.base_local_lr
            for _ in range(max(1, local_iterations - 1)):
                iters_run += 1
                prediction, indices, scores, counts = self.moe(x_states.reshape(B*T, D))
                prediction = prediction.reshape(B, T, D)
                residual = x_target_frozen - prediction

                if self.use_jacobian:
                    step = self.moe.transpose_forward(
                        residual.reshape(B*T, D), indices, scores, counts
                    ).reshape(B, T, D)
                else:
                    step = residual

                x_states = x_states + current_lr * step
                current_lr = current_lr * self.lr_decay # Decay learning rate
                
                # Dynamic Early Stopping
                if residual.abs().mean() < self.tolerance:
                    break
        
        # 2. DEQ Gradient Attachment (1nd-step analytical bridge)
        x_target_grad = torch.zeros_like(x_stream)
        x_target_grad[:, 1:, :] = x_stream[:, :-1, :] * torch.exp(1j * self.phase_offset)
        x_target_grad[:, 0, :] = x_stream[:, 0, :]
        
        prediction_grad, _, _, _ = self.moe(x_states.reshape(B*T, D))
        prediction_grad = prediction_grad.reshape(B, T, D)
        
        # Bridge uses the un-decayed base_local_lr to preserve gradient magnitude through all layers
        out = x_states + self.base_local_lr * (x_target_grad - prediction_grad)
        
        if unbatched:
            out = out.squeeze(0)
        return out, iters_run


class PPCGraphLLM(nn.Module):
    """
    Prospective Predictive Coding Graph Language Model.

    Architecture:
      - Complex-valued token embeddings (magnitude × e^{iφ})
      - Stacked PPCNodeLayer blocks (locally convergent, no global backprop through states)
      - Real + Imaginary concatenated decoder head (preserves phase information)

    Inputs:  [B, T] int64 token IDs  (or [T] for unbatched)
    Outputs: [B, T, vocab_size] real logits
    """
    def __init__(self, vocab_size: int, hidden_dim: int, num_layers: int = 2,
                 num_experts: int = 4, local_lr: float = 0.5, lr_decay: float = 0.85, use_jacobian: bool = False):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim

        # Issue 7 (Fix): PyTorch's nn.Embedding does not support complex autograd.
        # We use a real-valued embedding with 2*hidden_dim and view it as complex.
        self.embedding = nn.Embedding(vocab_size, hidden_dim * 2)
        
        # Initialize with ComplexKaimingInitializer values
        with torch.no_grad():
            from efv_nn.ppc_core import ComplexKaimingInitializer
            # Initialize a complex weight [V, D]
            complex_w = ComplexKaimingInitializer.initialize((vocab_size, hidden_dim))
            # View as real [V, D, 2] and flatten to [V, 2D] for the embedding table
            real_w = torch.view_as_real(complex_w).reshape(vocab_size, hidden_dim * 2)
            self.embedding.weight.copy_(real_w)

        self.layers = nn.ModuleList([
            PPCNodeLayer(hidden_dim, num_experts=num_experts, local_lr=local_lr, lr_decay=lr_decay, use_jacobian=use_jacobian)
            for _ in range(num_layers)
        ])

        # Issue 8: Using both real and imaginary parts in the decoder to preserve phase info
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)     
        self.output_head = nn.Linear(hidden_dim * 2, vocab_size, bias=True)
        nn.init.zeros_(self.output_head.bias)

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Map token IDs → complex embeddings [*, T, D]."""
        # [*, T, 2*D]
        out_real_flat = self.embedding(input_ids)
        # Reshape to [*, T, D, 2] so we can use view_as_complex
        new_shape = list(out_real_flat.shape[:-1]) + [self.hidden_dim, 2]
        out_real_struct = out_real_flat.view(*new_shape)
        # Convert to complex [*, T, D]
        return torch.view_as_complex(out_real_struct)

    def forward(self, input_ids: torch.Tensor, local_iterations: int = 2) -> torch.Tensor:
        """
        input_ids: [B, T] or [T]
        returns:   [B, T, vocab_size] or [T, vocab_size] real logits
        """
        x = self.embed(input_ids)            # complex [B, T, D] or [T, D]

        total_iters = 0
        for layer in self.layers:
            x, iters = layer(x, local_iterations)
            total_iters += iters

        # Issue 8: Preserve phase info by using both parts
        # If unbatched [T, D], x.real is [T, D], cat is [T, 2D]
        x_real_imag = torch.cat([x.real, x.imag], dim=-1)
        
        x_norm = self.layer_norm(x_real_imag) 
        return self.output_head(x_norm)      # [B, T, vocab_size]
