# Chapter 33: Speculative Decoding

## 1. Motivation

Autoregressive sampling from a transformer (Chapter 25) is *latency-bound*: each new token requires one full forward pass over $O(L)$ layers. Even with a KV cache (Chapter 32) eliminating recomputation over the prefix, the next token still costs one sequential pass through the target model $\pi$. On modern GPUs this is **memory-bandwidth bound** — the matmuls underutilize the compute units because we load the entire weight matrix to produce a single token's logits.

Speculative decoding (Leviathan et al. 2022; Chen et al. 2023) exploits a beautiful observation: a single forward pass of $\pi$ over $K+1$ tokens costs roughly the same wall-clock time as a pass over $1$ token, because the bottleneck is weight-loading, not arithmetic. So if we can *guess* the next $K$ tokens cheaply, we can verify all $K$ in parallel with one pass of $\pi$ — and provably get samples from $\pi$ exactly.

## 2. Definitions

**Draft model $\pi_d$.** A small, fast autoregressive model — typically $5\!-\!10\times$ smaller than the target. It generates $K$ candidate tokens $\tilde x_1, \ldots, \tilde x_K$ sequentially.

**Target model $\pi$.** The model whose distribution we *actually* want to sample from. After the draft proposes $\tilde x_{1:K}$, we run a single forward pass of $\pi$ on the prefix concatenated with $\tilde x_{1:K}$, producing target distributions $\pi(\cdot \mid x_{<i})$ at each position $i \in \{1,\ldots,K+1\}$ in parallel.

**Rejection-sampling acceptance rule.** For each draft token $\tilde x_i$, processed left-to-right:
$$
\text{accept } \tilde x_i \text{ with probability } a_i = \min\!\Big(1,\; \tfrac{\pi(\tilde x_i \mid x_{<i})}{\pi_d(\tilde x_i \mid x_{<i})}\Big).
$$
On the **first** rejection at position $i$, we discard $\tilde x_i, \tilde x_{i+1}, \ldots$ and resample from the **residual distribution**
$$
r_i(y) \;=\; \frac{\max\!\big(0,\; \pi(y \mid x_{<i}) - \pi_d(y \mid x_{<i})\big)}{Z_i}, \qquad Z_i = \sum_y \max(0, \pi(y) - \pi_d(y)).
$$
If *all* $K$ drafts are accepted, we use the target's free $(K+1)^{\text{th}}$ logit to sample one bonus token. Either way each speculative round produces between $1$ and $K+1$ accepted tokens at the cost of one target forward pass plus $K$ cheap draft passes.

**Speedup.** Let $\bar n \in [1, K+1]$ be the expected number of tokens accepted per round and let $\alpha = c_\pi / c_{\pi_d}$ be the cost ratio. Wall-clock speedup over plain decoding is approximately
$$
\text{speedup} \;\approx\; \frac{\bar n}{1 + K/\alpha}.
$$
For $\alpha \gg K$ (target much slower than draft), speedup $\to \bar n$, capped at $K+1$.

## 3. Theorems and Proofs

### Theorem 3.1 (Correctness — exact equivalence to sampling from $\pi$)

For any draft distribution $\pi_d$ with $\mathrm{supp}(\pi_d) \supseteq \mathrm{supp}(\pi)$, the speculative-decoding output at any single position is distributed *exactly* as $\pi$.

**Proof.** Fix a position $i$ and condition on $x_{<i}$; write $\pi(y) := \pi(y \mid x_{<i})$ and $\pi_d(y) := \pi_d(y \mid x_{<i})$. The output at this position equals some token $x$ via exactly one of two disjoint events:

*Case 1 — accept the draft.* The draft proposes $\tilde x_i = x$ (probability $\pi_d(x)$) and we accept (probability $\min(1, \pi(x)/\pi_d(x))$). Joint contribution:
$$
\mathbb{P}_1(x) \;=\; \pi_d(x) \cdot \min\!\Big(1, \tfrac{\pi(x)}{\pi_d(x)}\Big) \;=\; \min(\pi_d(x), \pi(x)).
$$

*Case 2 — reject and resample from $r$.* Total rejection probability over all proposals:
$$
P_{\text{rej}} \;=\; \sum_{\tilde x} \pi_d(\tilde x)\Big(1 - \min\!\big(1, \tfrac{\pi(\tilde x)}{\pi_d(\tilde x)}\big)\Big) \;=\; \sum_{\tilde x} \big(\pi_d(\tilde x) - \min(\pi_d(\tilde x), \pi(\tilde x))\big).
$$
Using $\pi_d(\tilde x) - \min(\pi_d(\tilde x), \pi(\tilde x)) = \max(0, \pi_d(\tilde x) - \pi(\tilde x))$ and the identity
$$
\sum_y \max(0, \pi_d(y) - \pi(y)) \;=\; \sum_y \max(0, \pi(y) - \pi_d(y)) \;=\; Z
$$
(both equal $\tfrac{1}{2}\|\pi - \pi_d\|_1$ since $\sum_y \pi = \sum_y \pi_d = 1$), we get $P_{\text{rej}} = Z$. After rejection we sample from $r(\cdot)$, contributing
$$
\mathbb{P}_2(x) \;=\; Z \cdot r(x) \;=\; Z \cdot \frac{\max(0, \pi(x) - \pi_d(x))}{Z} \;=\; \max(0, \pi(x) - \pi_d(x)).
$$

*Sum.* For any $x$,
$$
\mathbb{P}_1(x) + \mathbb{P}_2(x) \;=\; \min(\pi_d(x), \pi(x)) + \max(0, \pi(x) - \pi_d(x)) \;=\; \pi(x),
$$
because if $\pi(x) \le \pi_d(x)$ the first term is $\pi(x)$ and the second is $0$, while if $\pi(x) > \pi_d(x)$ the first term is $\pi_d(x)$ and the second is $\pi(x) - \pi_d(x)$. $\blacksquare$

The argument extends position-by-position: once $\tilde x_i$ is accepted, $x_i = \tilde x_i$ has the marginal $\pi(\cdot \mid x_{<i})$, and the chain rule gives joint sampling from $\pi$.

### Proposition 3.2 (Expected accepted prefix length)

Let $\alpha_j = \mathbb{E}_{\tilde x_j \sim \pi_d}[\min(1, \pi(\tilde x_j)/\pi_d(\tilde x_j))]$ be the marginal acceptance probability at position $j$ given a fresh draft. Then the expected number of *accepted draft tokens* in a $K$-token speculative round is
$$
\bar n_K \;=\; \sum_{i=1}^{K} \prod_{j=1}^{i} \alpha_j.
$$

**Sketch.** Let $A_j = \mathbb{1}[\text{first }j\text{ drafts all accepted}]$. By construction, acceptance at position $j$ is independent across positions conditional on the prefix, so $\mathbb{E}[A_j] = \prod_{j'\le j}\alpha_{j'}$. Then $\bar n_K = \mathbb{E}\!\big[\sum_j A_j\big] = \sum_j \prod_{j'\le j}\alpha_{j'}$. $\square$

A useful identity: $\alpha_j = 1 - \tfrac{1}{2}\|\pi - \pi_d\|_1$ at position $j$, so acceptance is governed by *total variation distance*.

### Corollary 3.3 (Optimal draft)

If $\pi_d = \pi$, then $\alpha_j = 1$ and $\bar n_K = K$, so all drafts accept — but we save no compute because the draft is as expensive as the target. The interesting regime is **cheap draft, slightly worse than target**: pay $K \cdot c_{\pi_d}$ to amortize one $c_\pi$ over $\bar n + 1$ tokens.

## 4. Code Sketch and Benchmarks

The companion notebook implements `speculative_decode(draft, target, prompt, K)` using toy categorical distributions over a small vocabulary (no GPT inference needed — Chapter 32 covers that). Key experiments:

1. **Empirical correctness** — draw $10{,}000$ tokens via the speculative protocol and compare to direct samples from $\pi$. A $\chi^2$ goodness-of-fit test fails to reject ($p > 0.05$).
2. **Speedup vs draft quality** — sweep a "noise" parameter $q$ where $\pi_d = (1-q)\,\pi + q\,\text{Uniform}$. Measure $\bar n$ and predicted wall-clock speedup at $K=4$, $\alpha = 7$.
3. **Breakeven** — find the $q$ at which speculative decoding *loses* to plain decoding because the draft is too poor.

## 5. Connection to LLMs

Production inference engines — **vLLM**, **llama.cpp**, **MLX-lm**, **TensorRT-LLM** — all ship speculative decoding. Reported wall-clock speedups: **1.5–3$\times$** for code/translation (high agreement between draft and target) and **1.1–1.5$\times$** for open-ended chat. Variants:

- **Self-speculation / Medusa heads**: instead of a separate draft model, attach extra prediction heads to the target itself.
- **Tree attention / SpecInfer**: verify *multiple* candidate continuations in one forward pass via causal-mask trees, raising $\bar n$.
- **EAGLE / Lookahead**: train a tiny draft head conditioned on the target's hidden state for higher acceptance.

The chapter's correctness proof transfers verbatim to all these schemes: as long as the verification step uses the rejection rule with the residual-distribution fallback, the output marginal is *exactly* $\pi$. No quality is sacrificed for the speedup — a rare free lunch in deep learning.
