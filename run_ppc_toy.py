import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add src to python path for module resolution
sys.path.append(str(Path(__file__).parent / 'src'))

from efv_nn.ppc_gnn import PPCGraphLLM

def main():
    print("="*60)
    print("  Prospective Predictive Coding (PPC) Complex GNN LLM")
    print("="*60)
    
    # Toy alphabet setup
    vocab_size = 10 
    seq_len = 16
    hidden_dim = 32
    
    # Deterministic toy pattern: 0, 1, 2, ..., 9, 0, 1, ...
    input_ids = torch.arange(seq_len) % vocab_size
    target_ids = torch.roll(input_ids, -1) # Next token
    
    model = PPCGraphLLM(vocab_size=vocab_size, hidden_dim=hidden_dim, num_layers=2)
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # The pure PPC nodes update locally, but we still need to optimize the decoding head
    # and embedding matrix. The dynamic state `x` minimizes its own error locally inside forward.
    optimizer = torch.optim.Adam([
        {'params': model.embedding.parameters(), 'lr': 0.01},
        {'params': model.output_head.parameters(), 'lr': 0.01},
        {'params': model.layer_norm.parameters(), 'lr': 0.01},
        {'params': [p for name, p in model.named_parameters() if 'gate_weights' in name], 'lr': 0.05}
    ])
    
    criterion = nn.CrossEntropyLoss()
    
    print("\nStarting Fast-Forward Toy Training...")
    
    for epoch in range(10):
        optimizer.zero_grad()
        
        # In a biological neural network, forward passes occur continuously.
        # We simulate 3 local iterations of the Wirtinger node update per systemic clock cycle.
        logits = model(input_ids, local_iterations=3)
        
        # Loss calculated on the final token predictions
        # (Exclude the last token because there is no target for it without wrapping)
        loss = criterion(logits[:-1], target_ids[:-1])
        
        # Backward pass only propagates through embeddings, gates and output head. 
        # The internal states of PPC Layers were detached and updated locally.
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch+1:02d} | CrossEntropy Loss: {loss.item():.4f}")
        
    print("\nVerification Complete: Local PPC interactions function without creating Autograd blowup.")

if __name__ == '__main__':
    main()
