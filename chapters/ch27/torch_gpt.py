#!/usr/bin/env python3
"""
torch_gpt.py — ~30M-param decoder-only GPT in PyTorch, trained on TinyStories.

Architecture: 6 layers, d_model=384, 6 heads, d_k=64, d_ff=1536, ctx=256.
Tokenizer:    tiktoken gpt2 (50257), char-level fallback if tiktoken absent.
Optim:        AdamW + linear warmup -> cosine decay, grad-clip 1.0.
Devices:      MPS > CUDA > CPU.

CLI:
    python3 torch_gpt.py --steps 50            # smoke (CPU OK)
    python3 torch_gpt.py --train --steps 3000  # full run (MPS/CUDA)
    python3 torch_gpt.py --sample              # generate from checkpoint
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------- config -----
@dataclass
class GPTConfig:
    vocab_size: int = 50257
    n_layer: int = 6
    n_head: int = 6
    d_model: int = 384
    d_ff: int = 1536
    ctx_len: int = 256
    dropout: float = 0.0


# -------------------------------------------------------------- tokenizer ----
class CharTokenizer:
    """Fallback tokenizer when tiktoken is unavailable."""

    def __init__(self, text: str):
        chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for i, c in enumerate(chars)}
        self.vocab_size = len(chars)

    def encode(self, s: str):
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, ids):
        return "".join(self.itos.get(int(i), "") for i in ids)


def build_tokenizer(text_for_fallback: str):
    try:
        import tiktoken

        enc = tiktoken.get_encoding("gpt2")

        class TT:
            vocab_size = enc.n_vocab

            def encode(self, s):
                return enc.encode(s)

            def decode(self, ids):
                return enc.decode([int(i) for i in ids])

        return TT(), "tiktoken-gpt2"
    except Exception:
        return CharTokenizer(text_for_fallback), "char-level"


# ---------------------------------------------------------------- model ------
class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.d_k = cfg.d_model // cfg.n_head
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.drop = nn.Dropout(cfg.dropout)
        mask = torch.tril(torch.ones(cfg.ctx_len, cfg.ctx_len)).view(
            1, 1, cfg.ctx_len, cfg.ctx_len
        )
        self.register_buffer("mask", mask)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.d_k).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.d_k).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_k).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.drop(att)
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ff = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.GELU(),
            nn.Linear(cfg.d_ff, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.ctx_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # weight tying
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def num_params(self):
        # PyTorch's .parameters() already de-dupes shared tensors, so the tied
        # head does not double-count. Just sum trainable params.
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.cfg.ctx_len, f"sequence {T} > ctx {self.cfg.ctx_len}"
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos)[None, :, :])
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new, temperature=0.7, top_k=40):
        for _ in range(max_new):
            cond = idx[:, -self.cfg.ctx_len :]
            logits, _ = self(cond)
            logits = logits[:, -1, :] / max(1e-6, temperature)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, nxt], dim=1)
        return idx


# ---------------------------------------------------------------- device -----
def pick_device():
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# --------------------------------------------------------------- dataset -----
FALLBACK_CORPUS = (
    "Once upon a time, a little fox lived in a green forest. "
    "Every morning the fox went to the river to drink water and watch the fish. "
    "One day the fox met a friendly rabbit who was lost. "
    "The fox said, 'Do not worry, I will help you find your way home.' "
    "Together they walked through the trees, past the tall oak and the small pond. "
    "Soon the rabbit saw his burrow and hopped inside, happy and safe. "
    "The fox smiled and went back to the river, glad to have made a new friend. "
    "The next day the rabbit brought carrots to share, and they sat in the warm sun. "
    "Birds sang in the branches and the wind moved softly through the leaves. "
    "From that day on, the fox and the rabbit were the best of friends in the green forest. "
) * 200


def load_corpus(max_chars: int = 2_000_000) -> str:
    try:
        from datasets import load_dataset

        ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
        buf = []
        n = 0
        for row in ds:
            t = row.get("text", "")
            buf.append(t)
            n += len(t)
            if n >= max_chars:
                break
        return "\n".join(buf)
    except Exception as e:
        print(f"[data] datasets unavailable ({e.__class__.__name__}); using fallback corpus")
        return FALLBACK_CORPUS


def make_data(tokens, ctx_len, batch_size, device):
    def get_batch():
        ix = torch.randint(0, len(tokens) - ctx_len - 1, (batch_size,))
        x = torch.stack([tokens[i : i + ctx_len] for i in ix])
        y = torch.stack([tokens[i + 1 : i + 1 + ctx_len] for i in ix])
        return x.to(device), y.to(device)

    return get_batch


# ---------------------------------------------------------------- schedule ---
def lr_at(step, warmup, total, lr_max, lr_min):
    if step < warmup:
        return lr_max * (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * min(p, 1.0)))


# ---------------------------------------------------------------- train ------
CKPT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "torch_gpt.pt")


def train(steps: int, full: bool):
    device = pick_device()
    print(f"[device] using {device}")

    text = load_corpus(max_chars=2_000_000 if full else 200_000)
    tok, tname = build_tokenizer(text)
    print(f"[tok] {tname}  vocab={tok.vocab_size}")

    ids = torch.tensor(tok.encode(text), dtype=torch.long)
    print(f"[data] {len(ids):,} tokens")

    cfg = GPTConfig(vocab_size=tok.vocab_size)
    model = GPT(cfg).to(device)
    n = model.num_params()
    print(f"[model] {n/1e6:.2f}M params  layers={cfg.n_layer} d={cfg.d_model} h={cfg.n_head} ctx={cfg.ctx_len}")

    batch = 16 if full else 4
    get_batch = make_data(ids, cfg.ctx_len, batch, device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)

    warmup = max(10, steps // 20)
    lr_max, lr_min = 3e-4, 3e-5
    log_every = max(1, steps // 20)
    losses = []
    t0 = time.time()
    model.train()
    for step in range(steps):
        x, y = get_batch()
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for g in opt.param_groups:
            g["lr"] = lr_at(step, warmup, steps, lr_max, lr_min)
        opt.step()
        losses.append(float(loss.item()))
        if step % log_every == 0 or step == steps - 1:
            print(f"  step {step:5d}  loss {losses[-1]:7.4f}  lr {opt.param_groups[0]['lr']:.5f}")
    elapsed = time.time() - t0
    print(f"[train] {steps} steps in {elapsed:.1f}s  ({steps/elapsed:.1f} step/s)")
    print(f"[train] init {losses[0]:.3f}  final {losses[-1]:.3f}")

    torch.save(
        {
            "model": model.state_dict(),
            "cfg": cfg.__dict__,
            "tok_name": tname,
            "vocab_size": tok.vocab_size,
            "losses": losses,
        },
        CKPT_PATH,
    )
    print(f"[ckpt] saved {CKPT_PATH}")
    return model, tok, device


def sample_from_ckpt(prompt: str = "Once upon a time", n_new: int = 100):
    if not os.path.exists(CKPT_PATH):
        print(f"[sample] no checkpoint at {CKPT_PATH}; run --train first")
        return
    device = pick_device()
    blob = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    cfg = GPTConfig(**blob["cfg"])
    model = GPT(cfg).to(device)
    model.load_state_dict(blob["model"])
    model.eval()
    text = load_corpus(max_chars=200_000)
    tok, _ = build_tokenizer(text)
    ids = torch.tensor([tok.encode(prompt)], dtype=torch.long, device=device)
    out = model.generate(ids, max_new=n_new, temperature=0.7, top_k=40)
    print(tok.decode(out[0].tolist()))


# ---------------------------------------------------------------- main -------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--train", action="store_true", help="run a full training session")
    ap.add_argument("--sample", action="store_true", help="generate from checkpoint")
    ap.add_argument("--prompt", type=str, default="Once upon a time")
    args = ap.parse_args()

    if args.sample and not args.train:
        sample_from_ckpt(args.prompt)
        return

    train(steps=args.steps, full=args.train)

    if args.train:
        sample_from_ckpt(args.prompt)


if __name__ == "__main__":
    main()
