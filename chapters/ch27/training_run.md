# Chapter 27 Training Run — Real ~30M-param GPT on TinyStories

> Template populated with **expected** values and methodology. After running on the
> Mac, replace `USER-REPLACE-AFTER-RUNNING` placeholders and the expected wall-clock
> with measured numbers from the actual run.

## Hardware

- **Machine:** MacBook Pro M4 Pro
- **Memory:** 48 GB unified
- **GPU:** 20 cores (Apple silicon)
- **OS:** macOS Sonoma+ with PyTorch MPS backend and MLX

## Hyperparameters

| Field           | Value                                               |
| --------------- | --------------------------------------------------- |
| Architecture    | decoder-only GPT, pre-norm, weight-tied head        |
| Layers          | 6                                                   |
| `d_model`       | 384                                                 |
| Heads           | 6 (`d_k` = 64)                                      |
| `d_ff`          | 1536                                                |
| Context length  | 256                                                 |
| Vocabulary      | tiktoken `gpt2` (50,257)                            |
| Total params    | ~30 M (weight-tied)                                 |
| Optimizer       | AdamW, $\beta_1=0.9$, $\beta_2=0.95$, wd $=0.1$     |
| LR schedule     | linear warmup ($5\%$) → cosine to $\eta_{\min}$     |
| $\eta_{\max}$   | $3 \times 10^{-4}$                                  |
| $\eta_{\min}$   | $3 \times 10^{-5}$                                  |
| Grad clip       | global-norm $= 1.0$                                 |
| Batch size      | 16                                                  |
| Steps           | 3000                                                |
| Tokens per step | $16 \times 256 = 4{,}096$                           |
| Total tokens    | $\approx 1.23 \times 10^{7}$                        |

## Wall-clock per backend

> **Methodology.** Run with `python3 torch_gpt.py --train --steps 3000` and
> `python3 mlx_gpt.py --train --steps 3000`. Walltime is reported by the script
> as `[train] {steps} steps in {elapsed:.1f}s`. Repeat each backend 2× and take
> the median; report wall-clock and tokens/sec separately.

| Backend              | Steps | Wall-clock (expected)    | Tokens/sec (expected) | Wall-clock (measured)        | Tokens/sec (measured)   |
| -------------------- | ----- | ------------------------ | --------------------- | ---------------------------- | ----------------------- |
| PyTorch + MPS (M4P)  | 3000  | ~45 min                  | ~4.5k                 | `USER-REPLACE-AFTER-RUNNING` | `USER-REPLACE`          |
| MLX (M4P)            | 3000  | ~25 min                  | ~8k                   | `USER-REPLACE-AFTER-RUNNING` | `USER-REPLACE`          |
| PyTorch + CPU (WSL2) | 5     | ~5 s (smoke test only)   | ~0.4k                 | `USER-REPLACE-AFTER-RUNNING` | `USER-REPLACE`          |

> Expected ratio MLX ≈ $1.8\text{--}2.5\times$ torch+MPS for this scale based on
> community benchmarks (tinier-GPT class with weight-tied embeddings). The
> advantage shrinks as the model gets larger and the Metal kernels become
> the bottleneck for both backends.

## Loss curve summary

> Replace this block with `losses[0]`, `losses[100]`, `losses[1000]`, `losses[2999]`
> from the actual run.

| Step | Expected CE loss | Measured CE loss             |
| ---- | ---------------- | ---------------------------- |
| 0    | ~10.8 (≈ $\log V$ for $V\!\approx\!50\text{k}$) | `USER-REPLACE` |
| 100  | ~7.5             | `USER-REPLACE`               |
| 500  | ~4.5             | `USER-REPLACE`               |
| 1000 | ~3.2             | `USER-REPLACE`               |
| 3000 | ~2.4             | `USER-REPLACE`               |

Final perplexity at step 3000 (expected): $\exp(2.4) \approx 11.0$, comparable to
nanoGPT-class baselines on TinyStories at this parameter count.

## Sample generations

> Run `python3 torch_gpt.py --sample --prompt "Once upon a time"` after training
> and paste the result here. Expected format:

```
Once upon a time, there was a little girl named Lily. She lived in a small house with her mother and father. One day, Lily went outside to play in the garden. She saw a butterfly with bright blue wings and tried to catch it. The butterfly flew away into the trees. Lily ran after it laughing happily until her mother called her in for dinner.
```

(USER-REPLACE-AFTER-RUNNING with actual model output. Quality at 30M params on
~12M training tokens should be a coherent 1–3 sentence opening with consistent
character, correct grammar in most positions, and recognisable TinyStories style;
do not expect long-range coherence.)

## Comparison: numpy 18K-param baseline vs torch 30M-param real run

| Metric           | numpy baseline (chapter cells)        | torch+MPS real run             |
| ---------------- | ------------------------------------- | ------------------------------ |
| Params           | $\sim 18{,}000$                       | $\sim 30{,}000{,}000$ (≈1700×) |
| Vocab            | 28 chars                              | 50,257 BPE tokens              |
| Context          | 16                                    | 256                            |
| Steps            | 600                                   | 3000                           |
| Final loss       | ~1.5 (28-char vocab, $\log V = 3.33$) | ~2.4 (BPE, $\log V = 10.83$)   |
| Wall-clock       | ~1 min on CPU                         | ~45 min on M4 Pro              |
| Sample quality   | character-level pastiche of 5 lines   | coherent 1–3 sentence stories  |
| Dependency       | numpy only                            | torch + tiktoken + datasets    |

**Why we keep both.** The numpy version is the *no-dependency baseline*: every
gradient is hand-derived, every line auditable, and the whole script runs on a
fresh Python install with `pip install numpy`. It cannot generate coherent text,
but it proves the pipeline works without hidden machinery. The torch + MLX
version is the *real model*: 1700× the parameters, BPE tokenizer, hardware
acceleration. Both run from the same chapter — the reader picks their level of
abstraction.
