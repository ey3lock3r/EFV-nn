import torch
import torch.nn as nn
import sys
import os
from pathlib import Path

# Add src to python path for module resolution
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from efv_nn.ppc_gnn import PPCGraphLLM

def validate_v3():
    torch.autograd.set_detect_anomaly(True)
    print("="*60)
    print("  PPC-OCNS GNN V3 Validation Suite")
    print("="*60)
    
    # 1. Setup
    vocab_size = 100
    hidden_dim = 64
    seq_len = 16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    model = PPCGraphLLM(vocab_size=vocab_size, hidden_dim=hidden_dim, num_layers=1).to(device)
    model.float() # Convert to float32 to prevent PyTorch CPU FP16 Adam NaNs
    
    # 2. Test Forward Pass (Stationary Target + Anderson)
    print("\n[1/4] Testing Forward Pass...")
    input_ids = torch.randint(0, vocab_size, (1, seq_len)).to(device)
    
    # We run in float32 for stability as per Pillar 4
    logits, avg_iters, avg_energy, energies, aux_loss, sg_penalty = model(input_ids, local_iters=10)
    
    assert logits.shape == (1, seq_len, vocab_size)
    assert avg_iters > 0
    print(f"  Success: Avg Iters: {avg_iters:.2f}, Avg Energy: {avg_energy:.4f}")
    
    # 3. Test Backward Pass (IFT + Backward Anderson)
    print("\n[2/4] Testing Backward Pass (Implicit Differentiation)...")
    target_ids = torch.randint(0, vocab_size, (1, seq_len)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    for i in range(3):
        optimizer.zero_grad()
        logits, _, _, _, _, _ = model(input_ids, local_iters=5)
        loss = criterion(logits.view(-1, vocab_size), target_ids.view(-1))
        print(f"  Iteration {i}, Loss: {loss.item():.4f}")
        loss.backward()
        optimizer.step()
    
    # Check if gradients exist for key parameters
    assert model.embedding.weight.grad is not None
    assert model.layers[0].moe.gate_weights.grad is not None
    # Check if gradients are non-zero
    grad_norm = model.layers[0].moe.experts_weight_real.grad.norm().item()
    print(f"  Success: Expert Weight Grad Norm: {grad_norm:.6e}")
    assert grad_norm > 0
    
    # 4. Test Memory Management (MoE Cache Clearing)
    print("\n[3/4] Testing Memory Management (Inference Mode)...")
    
    # Ensure cache is empty initially
    model.layers[0].moe.clear_cache()
    assert not hasattr(model.layers[0].moe, '_wr_f32') or model.layers[0].moe._wr_f32 is None
    
    print("  Running inference (no_grad)...")
    with torch.no_grad():
        _ = model(input_ids, local_iters=5)
        
    # After inference, the cache SHOULD be cleared (as per my fix)
    has_cache = hasattr(model.layers[0].moe, '_wr_f32') and model.layers[0].moe._wr_f32 is not None
    print(f"  Cache present after inference? {has_cache}")
    assert not has_cache, "Memory leak: FP32 MoE cache not cleared after inference!"
    print("  Success: Cache cleared.")
    
    # 5. Test Complex Gradient Flow (Wirtinger Stability)
    print("\n[4/4] Testing Complex Activation (ComplexGELU)...")
    # This is implicitly tested by the backward pass not exploding, 
    # but we verify the output dtype and shape.
    layer0 = model.layers[0]
    test_input = torch.randn(1, seq_len, hidden_dim, 2).to(device)
    output, iters, energy = layer0(test_input)
    assert output.shape == test_input.shape
    assert output.dtype == test_input.dtype
    print("  Success: ComplexGELU and Phasal Rotation verified.")

    print("\n" + "="*60)
    print("  ALL V3 VALIDATION TESTS PASSED")
    print("="*60)

if __name__ == "__main__":
    try:
        validate_v3()
    except Exception as e:
        print(f"\n❌ VALIDATION FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
