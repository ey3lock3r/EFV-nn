"""
run_ppc_shakespeare.py
======================
Tests PPCGraphLLM on Tiny Shakespeare — a canonical 1 MB character-level
language modelling benchmark used by Karpathy's nanoGPT, minGPT, etc.

Usage (CPU):
    uv run run_ppc_shakespeare.py

The script:
  1. Downloads Tiny Shakespeare (~1 MB) if not cached.
  2. Builds a character-level vocabulary (~65 tokens).
  3. Splits 90 / 10 into train / validation.
  4. Trains PPCGraphLLM with mini-batch gradient descent on the embedding +
     routing weights (PPC local iterations handle the hidden states).
  5. Reports per-epoch train loss, val loss, val perplexity, and accuracy.
  6. Compares against a simple Bigram baseline.
"""

import sys, math, time, urllib.request
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "src"))

import torch
import torch.nn as nn
import torch.nn.functional as F

from efv_nn.ppc_gnn import PPCGraphLLM

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
CACHE_PATH  = Path(".cache/shakespeare.txt")
DATA_URL    = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
SEQ_LEN          = 64       # Context window per training sample
BATCH_SIZE       = 64       # Mini-batch size
HIDDEN_DIM       = 128      # Model width
NUM_LAYERS       = 2        # PPC depth
NUM_EXPERTS      = 4        # Experts per node
LOCAL_LR         = 0.7      # PPC local contraction rate
LOCAL_ITERS      = 2        # PPC inner iterations per forward pass
EPOCHS           = 10
STEPS_PER_EPOCH  = 200      # Cap batches/epoch for CPU training (~12s/epoch)
LR               = 5e-3     # Adam learning rate for trainable weights
GRAD_CLIP        = 1.0
DEVICE           = "cpu"
SEED             = 42

torch.manual_seed(SEED)

# ──────────────────────────────────────────────
# 1. Data
# ──────────────────────────────────────────────
def download_shakespeare() -> str:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not CACHE_PATH.exists():
        print(f"Downloading Tiny Shakespeare → {CACHE_PATH} ...", flush=True)
        urllib.request.urlretrieve(DATA_URL, CACHE_PATH)
        print(f"  Downloaded {CACHE_PATH.stat().st_size / 1024:.0f} KB")
    return CACHE_PATH.read_text(encoding="utf-8")


class CharDataset:
    """Character-level dataset with sliding-window batching."""

    def __init__(self, text: str, seq_len: int, split: str = "train", split_ratio: float = 0.9):
        chars = sorted(set(text))
        self.vocab_size = len(chars)
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for c, i in self.stoi.items()}

        ids = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)
        n   = int(len(ids) * split_ratio)
        self.data = ids[:n] if split == "train" else ids[n:]
        self.seq_len = seq_len

    def __len__(self):
        return max(0, len(self.data) - self.seq_len - 1)

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + 1 : idx + self.seq_len + 1]
        return x, y

    def decode(self, ids) -> str:
        return "".join(self.itos[i] for i in ids)


def make_loader(dataset, batch_size, shuffle=True):
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True
    )


# ──────────────────────────────────────────────
# 2. Bigram Baseline
# ──────────────────────────────────────────────
class BigramLM(nn.Module):
    """Lookup-table bigram model — the simplest possible LM baseline."""
    def __init__(self, vocab_size):
        super().__init__()
        self.table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, x, **kwargs):
        return self.table(x)   # [B, T, V]


# ──────────────────────────────────────────────
# 3. Training helpers
# ──────────────────────────────────────────────
def run_epoch(model, loader, optimizer, device, train=True, step_limit=None):
    model.train(train)
    total_loss, total_correct, total_tokens = 0.0, 0, 0

    with torch.set_grad_enabled(train):
        for step, (x, y) in enumerate(loader):
            if step_limit and step >= step_limit:
                break
            x, y = x.to(device), y.to(device)
            logits = model(x, local_iterations=LOCAL_ITERS)                            # [B, T, V]
            B, T, V = logits.shape
            loss = F.cross_entropy(logits.reshape(B * T, V), y.reshape(B * T))

            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()

            total_loss   += loss.item() * B * T
            total_correct += (logits.argmax(-1) == y).sum().item()
            total_tokens  += B * T

    avg_loss  = total_loss  / total_tokens
    accuracy  = total_correct / total_tokens
    ppl       = math.exp(avg_loss)
    return avg_loss, ppl, accuracy


@torch.no_grad()
def sample(model, dataset, seed_text="ROMEO:", length=200, device="cpu"):
    model.eval()
    ids = torch.tensor([dataset.stoi.get(c, 0) for c in seed_text], dtype=torch.long).unsqueeze(0).to(device)
    generated = list(seed_text)
    for _ in range(length):
        logits = model(ids[:, -SEQ_LEN:])                   # [1, T, V]
        probs  = F.softmax(logits[:, -1, :] / 0.8, dim=-1) # temperature=0.8
        next_id = torch.multinomial(probs, 1)
        ids = torch.cat([ids, next_id], dim=1)
        generated.append(dataset.itos[next_id.item()])
    return "".join(generated)


# ──────────────────────────────────────────────
# 4. Main
# ──────────────────────────────────────────────
def main():
    print("=" * 65)
    print("  Prospective Predictive Coding GNN-LLM")
    print("  Dataset: Tiny Shakespeare (character-level)")
    print("=" * 65)

    text = download_shakespeare()
    print(f"\n  Corpus: {len(text):,} chars")

    train_ds = CharDataset(text, SEQ_LEN, split="train")
    val_ds   = CharDataset(text, SEQ_LEN, split="val")
    V        = train_ds.vocab_size

    print(f"  Vocab : {V} unique characters")
    print(f"  Train : {len(train_ds):,} sequences")
    print(f"  Val   : {len(val_ds):,} sequences\n")

    train_loader = make_loader(train_ds, BATCH_SIZE, shuffle=True)
    val_loader   = make_loader(val_ds,   BATCH_SIZE, shuffle=False)

    # ── PPC Model ──
    ppc_model = PPCGraphLLM(
        vocab_size   = V,
        hidden_dim   = HIDDEN_DIM,
        num_layers   = NUM_LAYERS,
        num_experts  = NUM_EXPERTS,
        local_lr     = LOCAL_LR,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in ppc_model.parameters())
    print(f"  PPC Model params : {n_params:,}")

    # ── Bigram Baseline ──
    bigram = BigramLM(V).to(DEVICE)
    bigram_opt = torch.optim.Adam(bigram.parameters(), lr=LR)
    print(f"  Bigram params    : {sum(p.numel() for p in bigram.parameters()):,}\n")

    # Train Bigram for same epochs (quick)
    print("─" * 65)
    print("  Training Bigram Baseline (1 epoch) ...")
    for _ in range(1):  # Single epoch is sufficient for a bigram table
        run_epoch(bigram, train_loader, bigram_opt, DEVICE, train=True,
                  step_limit=STEPS_PER_EPOCH)
    b_loss, b_ppl, b_acc = run_epoch(bigram, val_loader, None, DEVICE, train=False,
                                      step_limit=STEPS_PER_EPOCH)
    print(f"  Bigram  → val_loss={b_loss:.4f}  ppl={b_ppl:.2f}  acc={b_acc*100:.2f}%\n")

    # ── Train PPC ──
    optimizer = torch.optim.Adam(ppc_model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-4)

    print("─" * 65)
    fmt = "  {:>5s} | {:>10s} | {:>10s} | {:>8s} | {:>8s} | {:>7s}"
    print(fmt.format("Epoch", "Train Loss", "Val Loss", "Val PPL", "Val Acc", "Time"))
    print("  " + "─" * 60)

    best_val_loss = float("inf")
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        tr_loss, _, _           = run_epoch(ppc_model, train_loader, optimizer, DEVICE,
                                               train=True,  step_limit=STEPS_PER_EPOCH)
        val_loss, val_ppl, val_acc = run_epoch(ppc_model, val_loader,   None,      DEVICE,
                                               train=False, step_limit=STEPS_PER_EPOCH)
        scheduler.step()
        elapsed = time.time() - t0

        marker = " ◀ best" if val_loss < best_val_loss else ""
        best_val_loss = min(best_val_loss, val_loss)

        print(f"  {epoch:>5d} | {tr_loss:>10.4f} | {val_loss:>10.4f} | {val_ppl:>8.2f} | {val_acc*100:>7.2f}% | {elapsed:>6.1f}s{marker}")

    print("\n" + "─" * 65)
    print(f"  Final PPC  → val_loss={best_val_loss:.4f}  ppl={math.exp(best_val_loss):.2f}")
    print(f"  Bigram     → val_loss={b_loss:.4f}  ppl={b_ppl:.2f}")
    improvement = b_ppl - math.exp(best_val_loss)
    print(f"  PPL gain vs bigram: {improvement:+.2f}" if improvement > 0 else f"  PPL gap vs bigram: {improvement:.2f}")

    # ── Sample generation ──
    print("\n" + "─" * 65)
    print("  Generated text sample (PPC model, temp=0.8):")
    print("─" * 65)
    sample_text = sample(ppc_model, train_ds, seed_text="ROMEO:", length=300, device=DEVICE)
    print(sample_text)
    print("─" * 65)


if __name__ == "__main__":
    main()
