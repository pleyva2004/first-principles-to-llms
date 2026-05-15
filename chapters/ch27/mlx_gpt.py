#!/usr/bin/env python3
"""
mlx_gpt.py — Apple-silicon-native ~30M-param GPT (MLX), TinyStories.

Same architecture as torch_gpt.py: 6L, d=384, 6h, d_ff=1536, ctx=256.

This file imports MLX guarded by try/except. On Linux/non-Mac it prints a
clear message and exits 0 — never errors at import time.

CLI:
    python3 mlx_gpt.py --steps 50
    python3 mlx_gpt.py --train --steps 3000
    python3 mlx_gpt.py --sample
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

# --- MLX guard --------------------------------------------------------------
try:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
except Exception:
    print(
        "MLX not available; this script must run on macOS with MLX installed: "
        "pip install mlx mlx-lm"
    )
    sys.exit(0)


# ---------------------------------------------------------------- config -----
class GPTConfig:
    def __init__(
        self,
        vocab_size=50257,
        n_layer=6,
        n_head=6,
        d_model=384,
        d_ff=1536,
        ctx_len=256,
    ):
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.d_model = d_model
        self.d_ff = d_ff
        self.ctx_len = ctx_len


# -------------------------------------------------------------- tokenizer ----
class CharTokenizer:
    def __init__(self, text: str):
        chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for i, c in enumerate(chars)}
        self.vocab_size = len(chars)

    def encode(self, s):
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, ids):
        return "".join(self.itos.get(int(i), "") for i in ids)


def build_tokenizer(text):
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
        return CharTokenizer(text), "char-level"


# ---------------------------------------------------------------- model ------
class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.d_k = cfg.d_model // cfg.n_head
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.ctx_len = cfg.ctx_len

    def __call__(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = mx.split(qkv, 3, axis=-1)
        q = q.reshape(B, T, self.n_head, self.d_k).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.n_head, self.d_k).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.n_head, self.d_k).transpose(0, 2, 1, 3)
        att = (q @ k.transpose(0, 1, 3, 2)) / math.sqrt(self.d_k)
        # causal mask
        mask = mx.triu(mx.ones((T, T)) * -1e9, k=1)
        att = att + mask
        att = mx.softmax(att, axis=-1)
        y = (att @ v).transpose(0, 2, 1, 3).reshape(B, T, C)
        return self.proj(y)


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.fc1 = nn.Linear(cfg.d_model, cfg.d_ff)
        self.fc2 = nn.Linear(cfg.d_ff, cfg.d_model)

    def __call__(self, x):
        x = x + self.attn(self.ln1(x))
        h = self.fc2(nn.gelu(self.fc1(self.ln2(x))))
        return x + h


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.ctx_len, cfg.d_model)
        self.blocks = [Block(cfg) for _ in range(cfg.n_layer)]
        self.ln_f = nn.LayerNorm(cfg.d_model)
        # weight-tied head: realised via tok_emb.weight in __call__

    def __call__(self, idx):
        B, T = idx.shape
        pos = mx.arange(T)
        x = self.tok_emb(idx) + self.pos_emb(pos)[None, :, :]
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        # weight tying: logits = x @ tok_emb.weight.T
        return x @ self.tok_emb.weight.T


def loss_fn(model, x, y):
    logits = model(x)
    B, T, V = logits.shape
    return nn.losses.cross_entropy(
        logits.reshape(B * T, V), y.reshape(B * T), reduction="mean"
    )


# ---------------------------------------------------------------- corpus -----
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


def load_corpus(max_chars=2_000_000):
    try:
        from datasets import load_dataset

        ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
        buf, n = [], 0
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


# ------------------------------------------------------------- schedule ------
def lr_at(step, warmup, total, lr_max, lr_min):
    if step < warmup:
        return lr_max * (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * min(p, 1.0)))


# ------------------------------------------------------------- training ------
CKPT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mlx_gpt.npz")


def train(steps: int, full: bool):
    text = load_corpus(max_chars=2_000_000 if full else 200_000)
    tok, tname = build_tokenizer(text)
    print(f"[tok] {tname}  vocab={tok.vocab_size}")

    ids = mx.array(tok.encode(text), dtype=mx.int32)
    print(f"[data] {ids.size:,} tokens")

    cfg = GPTConfig(vocab_size=tok.vocab_size)
    model = GPT(cfg)
    mx.eval(model.parameters())
    n_params = sum(v.size for _, v in tree_flatten(model.parameters()))
    print(f"[model] {n_params/1e6:.2f}M params  layers={cfg.n_layer} d={cfg.d_model} h={cfg.n_head} ctx={cfg.ctx_len}")

    batch = 16 if full else 4
    warmup = max(10, steps // 20)
    lr_max, lr_min = 3e-4, 3e-5
    opt = optim.AdamW(learning_rate=lr_max, betas=[0.9, 0.95], weight_decay=0.1)

    import numpy as _np

    def get_batch():
        ix = _np.random.randint(0, ids.size - cfg.ctx_len - 1, size=(batch,))
        x = mx.stack([ids[int(i) : int(i) + cfg.ctx_len] for i in ix])
        y = mx.stack([ids[int(i) + 1 : int(i) + 1 + cfg.ctx_len] for i in ix])
        return x, y

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    losses = []
    t0 = time.time()
    log_every = max(1, steps // 20)
    for step in range(steps):
        x, y = get_batch()
        loss, grads = loss_and_grad(model, x, y)
        # gradient clipping (global norm 1.0)
        grads = clip_grads(grads, 1.0)
        opt.learning_rate = lr_at(step, warmup, steps, lr_max, lr_min)
        opt.update(model, grads)
        mx.eval(model.parameters(), opt.state)
        losses.append(float(loss.item()))
        if step % log_every == 0 or step == steps - 1:
            print(f"  step {step:5d}  loss {losses[-1]:7.4f}  lr {opt.learning_rate:.5f}")
    print(f"[train] {steps} steps in {time.time()-t0:.1f}s")
    print(f"[train] init {losses[0]:.3f}  final {losses[-1]:.3f}")

    flat = dict(tree_flatten(model.parameters()))
    mx.savez(CKPT_PATH, **flat)
    print(f"[ckpt] saved {CKPT_PATH}")


def clip_grads(grads, max_norm):
    flat = tree_flatten(grads)
    sq = sum(mx.sum(g * g).item() for _, g in flat if g is not None)
    norm = math.sqrt(sq)
    scale = min(1.0, max_norm / (norm + 1e-12))
    if scale < 1.0:
        return tree_map(lambda g: g * scale, grads)
    return grads


def tree_flatten(d, prefix=""):
    out = []
    if isinstance(d, dict):
        for k, v in d.items():
            out.extend(tree_flatten(v, f"{prefix}.{k}" if prefix else k))
    elif isinstance(d, list):
        for i, v in enumerate(d):
            out.extend(tree_flatten(v, f"{prefix}.{i}"))
    else:
        out.append((prefix, d))
    return out


def tree_map(fn, d):
    if isinstance(d, dict):
        return {k: tree_map(fn, v) for k, v in d.items()}
    if isinstance(d, list):
        return [tree_map(fn, v) for v in d]
    return fn(d) if d is not None else d


# ---------------------------------------------------------------- main -------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--prompt", type=str, default="Once upon a time")
    args = ap.parse_args()

    if args.sample and not args.train:
        if not os.path.exists(CKPT_PATH):
            print(f"[sample] no checkpoint at {CKPT_PATH}; run --train first")
            return
        # for sampling: load weights, run greedy/top-k like torch version
        text = load_corpus(max_chars=200_000)
        tok, _ = build_tokenizer(text)
        cfg = GPTConfig(vocab_size=tok.vocab_size)
        model = GPT(cfg)
        weights = mx.load(CKPT_PATH)
        # restore (best-effort flat assignment)
        params = model.parameters()
        for k, v in weights.items():
            d = params
            keys = k.split(".")
            for kk in keys[:-1]:
                d = d[int(kk)] if kk.isdigit() else d[kk]
            last = keys[-1]
            if last.isdigit():
                d[int(last)] = v
            else:
                d[last] = v
        model.update(params)
        ids = mx.array([tok.encode(args.prompt)], dtype=mx.int32)
        for _ in range(100):
            cond = ids[:, -cfg.ctx_len :]
            logits = model(cond)[:, -1, :] / 0.7
            probs = mx.softmax(logits, axis=-1)
            # top-k
            k = 40
            sorted_idx = mx.argsort(-logits, axis=-1)
            keep = sorted_idx[:, :k]
            mask = mx.zeros_like(logits)
            for b in range(logits.shape[0]):
                mask[b, keep[b]] = 1
            probs = probs * mask
            probs = probs / probs.sum(axis=-1, keepdims=True)
            nxt = mx.random.categorical(mx.log(probs + 1e-12))
            ids = mx.concatenate([ids, nxt[:, None]], axis=1)
        print(tok.decode(ids[0].tolist()))
        return

    train(args.steps, args.train)


if __name__ == "__main__":
    main()
