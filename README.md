# 🌌 EFV-nn: Energy-Based Phasal Variational Neural Network

A 3.2B parameter PPC-GNN architecture optimized for sharded training on Dual-T4 hardware.

## 🏗️ Project Overview
EFV-nn implements a state-of-the-art Parallel Phasal Computing (PPC) Graph Neural Network. It features:
- **3.2B Parameters**: High-capacity sharded architecture.
- **DEQ Solvers**: Implicit Function Theorem (IFT) based backward passes.
- **Anderson Acceleration**: Stabilized convergence for phasal resonance.
- **Triton Kernels**: Custom-fused kernels for maximum throughput.

## 📁 Documentation & Artifacts
All important project artifacts have been organized in the `docs/` directory:

- **[Project Memory](docs/project_memory.md)**: Core architectural axioms, execution protocols, and learnings diary.
- **[Research Notebooks](docs/research/)**:
  - `ppc_gnn_v2.ipynb`: Main research and execution notebook.
  - `ppc_spectral_research.ipynb`: Spectral analysis and phasal stability research.
- **[Results](docs/results/)**:
  - `efv_comprehensive_results.png`: Visual performance metrics and convergence plots.
- **[Verification](docs/verification_state.pt)**: Saved state for system verification.

## 🚀 Quick Start
```bash
# Install dependencies
uv sync

# Run training pilot
python run_ppc_shakespeare.py
```

## 🛠️ Tech Stack
- **Core**: Python, PyTorch, Triton
- **Optimization**: `bitsandbytes` (8-bit), `torch.compile` (selective)
- **Monitoring**: Weights & Biases (W&B)
- **Precision**: FP32 Accumulation, FP16/BF16 Experts
