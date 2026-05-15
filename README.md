# From First Principles to Modern LLMs

[![Render](https://github.com/pleyva2004/first-principles-to-llms/actions/workflows/render.yml/badge.svg)](https://github.com/pleyva2004/first-principles-to-llms/actions/workflows/render.yml)

An unbroken chain of derivation from set theory to modern large language models, then into the reinforcement-learning algorithms used to align them. Thirty-one chapters carry the reader from sets, functions, and proofs through real and multivariable analysis, linear algebra, probability, information theory, and convex/stochastic optimization, into neural networks, attention, the transformer block, pre-training, the post-training stack (SFT, RLHF, DPO), and finally the RL spine — MDPs, value-based methods, policy gradient, GRPO, and a runnable tiny-GPT alignment loop. Every step is presented with the same standard of rigor: definition, theorem, proof, code.

## Three forms

The same content lives in three synchronized forms. Pick the one that suits your reading mode.

| Form | File | Best for |
| --- | --- | --- |
| Markdown | [chain.md](chain.md) | Browsing on GitHub, quick search |
| PDF (typeset) | [chain.pdf](chain.pdf) | Linear reading, theorems, equations |
| Jupyter notebook | [chain.ipynb](chain.ipynb) | Running every code block live |

## Chapter list

### Block A — Foundations

| # | Chapter |
| --- | --- |
| 1 | Sets, functions, logic, proofs |
| 2 | Numbers, sequences, limits, completeness |
| 3 | Continuity, univariate differentiation, chain rule |
| 4 | Multivariate calculus: partials, gradients, Jacobians |
| 5 | Linear algebra I: vector spaces, basis, linear maps |
| 6 | Linear algebra II: inner products, norms, eigenvalues, SVD |
| 7 | Convexity and optimization; gradient descent convergence |

### Block B — Probability and Information

| # | Chapter |
| --- | --- |
| 8 | Probability foundations: sample spaces, sigma-algebras, Kolmogorov axioms |
| 9 | Random variables, distributions, CDF/PMF/PDF |
| 10 | Expectation, variance, covariance; Jensen's inequality |
| 11 | Information theory: self-information, entropy, cross-entropy, KL |
| 12 | Statistical inference: likelihood, MLE, ERM, bias-variance |

### Block C — Stochastic Optimization

| # | Chapter |
| --- | --- |
| 13 | SGD: stochastic-approximation theorem; mini-batching; convergence sketch |
| 14 | Momentum, RMSProp, AdamW: derivation and bias-correction proof |

### Block D — Neural Networks

| # | Chapter |
| --- | --- |
| 15 | MLPs as compositional functions; universal approximation |
| 16 | Activation functions: ReLU/GELU/softmax with derivatives |
| 17 | Loss functions: MSE, cross-entropy; gradients from first principles |
| 18 | Backpropagation: chain rule applied; reverse-mode AD as a graph algorithm |

### Block E — Sequence Models and Attention

| # | Chapter |
| --- | --- |
| 19 | Embeddings: token to vector; lookup as a linear map; weight tying |
| 20 | RNN intuition; vanishing-gradient proof; why we need attention |
| 21 | Scaled dot-product attention: derivation, softmax-temperature analysis |
| 22 | Multi-head attention: parallel heads as concat-then-project; complexity |
| 23 | Transformer block: residual + LayerNorm/RMSNorm + FFN + attention; gradient-flow argument |
| 24 | Positional encoding: sinusoidal derivation, RoPE construction |

### Block F — Pre-training

| # | Chapter |
| --- | --- |
| 25 | Causal masking; next-token prediction loss as MLE on the empirical distribution |
| 26 | Tokenization: BPE algorithm; greedy merge correctness |
| 27 | Pre-training pipeline: AdamW + warmup + cosine decay + gradient clipping; tiny-GPT training run |

### Block G — Post-training

| # | Chapter |
| --- | --- |
| 28 | SFT, RLHF (PPO/GRPO), and DPO; train + post-train a tiny GPT |

### Block H — Reinforcement Learning

| # | Chapter |
| --- | --- |
| 29 | MDP foundations: Bellman equations, value iteration, tabular Q-learning |
| 30 | Value-based deep RL: function approximation, DQN, max-entropy framework |
| 31 | Policy gradient, GRPO, and the RLHF/DPO bridge: tiny-GPT alignment loop |

## Build

The PDF and HTML forms are rendered automatically by GitHub Actions on every push. To build locally:

```bash
# PDF: run twice so the table of contents resolves
pdflatex chain.tex
pdflatex chain.tex

# Notebook (executes every code block end-to-end)
jupyter nbconvert --execute chain.ipynb
```

To refresh the auto-generated tables of contents and verify that every chapter exists in all three forms:

```bash
python3 generate.py
```

## Cross-links

Where a chapter cites a piece of mathematical machinery covered in more atomic detail elsewhere, it will link to the corresponding concept page in the companion repo [pleyva2004/math-foundations](https://github.com/pleyva2004/math-foundations). This repository is the *chain*; that repository is the *atlas*.

## License

MIT
