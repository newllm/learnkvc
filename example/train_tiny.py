"""Minimal PyTorch training loop for a tiny GPT.

Run:
    python3 example/train_tiny.py --steps 200 --device cpu

It learns a synthetic repeating token cycle (next-token prediction), so it
needs no dataset and finishes in well under a minute on CPU. The point is the
shape of one training step and the memory ledger, not model quality.

Training memory (per GPU, rough) is:
    weights     = params * bytes
    grads       = params * bytes
    optimizer   = params * (8 + 4)      AdamW m,v + fp32 master (mixed precision)
    activations = layers * batch * seq * hidden * 12 * bytes
                   (drop to ~2.5x with activation checkpointing)
"""

import argparse
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

D_MODEL = 64
N_HEADS = 4
N_LAYERS = 2
VOCAB = 32


class Attention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        b, t, c = x.shape
        q, k, v = self.qkv(x).split(c, dim=-1)
        q = q.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)

        # A fused attention kernel avoids materialising [b, h, t, t].
        # This small demo keeps the scores visible for teaching.
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = torch.tril(torch.ones(t, t, device=x.device, dtype=torch.bool))
        scores = scores.masked_fill(~mask, float("-inf"))
        probs = F.softmax(scores, dim=-1)
        y = (probs @ v).transpose(1, 2).contiguous().view(b, t, c)
        return self.proj(y)


class Block(nn.Module):
    def __init__(self, d_model, n_heads, mlp_ratio=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = Attention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_ratio * d_model),
            nn.GELU(),
            nn.Linear(mlp_ratio * d_model, d_model),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, vocab, seq_len, d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS):
        super().__init__()
        self.seq_len = seq_len
        self.tok = nn.Embedding(vocab, d_model)
        self.pos = nn.Embedding(seq_len, d_model)
        self.blocks = nn.ModuleList([Block(d_model, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)
        self.head.weight = self.tok.weight  # weight tying

    def forward(self, idx):
        b, t = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(t, device=idx.device))
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln_f(x))


def make_batch(data, batch, seq_len, generator, device):
    starts = torch.randint(0, len(data) - seq_len - 1, (batch,), generator=generator)
    x = torch.stack([data[i:i + seq_len] for i in starts]).to(device)
    y = torch.stack([data[i + 1:i + seq_len + 1] for i in starts]).to(device)
    return x, y


def ledger(model, batch, seq_len, weight_bytes=2):
    """Back-of-envelope training memory for one GPU."""
    params = sum(p.numel() for p in model.parameters())
    weights = params * weight_bytes
    grads = params * weight_bytes
    optimizer = params * 12  # AdamW fp32 m,v (8) + fp32 master (4)
    activations = N_LAYERS * batch * seq_len * D_MODEL * 12 * weight_bytes
    return {"params": params, "weights": weights, "grads": grads,
            "optimizer": optimizer, "activations": activations}


def mib(n):
    return f"{n / 2 ** 20:.1f} MiB"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq", type=int, default=16)
    ap.add_argument("--accum", type=int, default=4, help="gradient accumulation steps")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(0)
    data = torch.arange(VOCAB).repeat(300)  # repeating cycle, no dataset needed
    model = TinyGPT(VOCAB, args.seq).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    gen = torch.Generator().manual_seed(0)

    size = ledger(model, args.batch, args.seq)
    print(f"params                 {size['params']:,}")
    print(f"weights (bf16)         {mib(size['weights'])}")
    print(f"grads (bf16)           {mib(size['grads'])}")
    print(f"AdamW m,v + master     {mib(size['optimizer'])}")
    print(f"activations (1 micro)  {mib(size['activations'])}")
    print(f"global batch           {args.batch} x {args.accum} = {args.batch * args.accum}")

    t0 = time.time()
    for step in range(args.steps + 1):
        opt.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for _ in range(args.accum):
            x, y = make_batch(data, args.batch, args.seq, gen, args.device)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, VOCAB), y.reshape(-1))
            (loss / args.accum).backward()
            loss_sum += loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 50 == 0:
            print(f"step {step:4d}  loss {loss_sum / args.accum:.4f}")
    print(f"done in {time.time() - t0:.1f}s on {args.device}")


if __name__ == "__main__":
    main()
