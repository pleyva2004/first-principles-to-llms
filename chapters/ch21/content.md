## Motivation

Chapter 20 ended with a sharp negative result: recurrent networks propagate gradients through $|i-j|$ multiplicative steps, so information from token $j$ reaches token $i$ only after a chain of contractions or expansions. Either the gradient vanishes ($\rho < 1$) or it explodes ($\rho > 1$). Long-range dependencies are statistically unreachable.

We now build a primitive that escapes this: **scaled dot-product attention** (Vaswani et al., 2017). It connects every output position to every input position through a *single* matrix multiplication, so the gradient between any pair of tokens travels in $O(1)$ depth. Combined with the embedding layer of Chapter 19 and the linear maps of Chapter 5, it yields the building block of every modern transformer LM.

## Definitions

Let $X \in \mathbb{R}^{T \times d_{\mathrm{model}}}$ be a sequence of $T$ token vectors (e.g.\ embeddings from Chapter 19). Fix projection matrices
$$
W_Q, W_K \in \mathbb{R}^{d_{\mathrm{model}} \times d_k}, \qquad W_V \in \mathbb{R}^{d_{\mathrm{model}} \times d_v},
$$
and define **queries, keys, and values**
$$
Q = X W_Q, \quad K = X W_K, \quad V = X W_V.
$$

**Definition (Scaled dot-product attention).**
$$
\mathrm{Attn}(Q, K, V) = \mathrm{softmax}\!\left( \tfrac{Q K^\top}{\sqrt{d_k}} \right) V.
$$
The softmax is applied row-wise. We call $A := \mathrm{softmax}(QK^\top/\sqrt{d_k}) \in \mathbb{R}^{T \times T}$ the **attention weights**. Each row $A_{i,:}$ is a probability vector ($A_{ij} \geq 0$, $\sum_j A_{ij} = 1$). When $Q, K, V$ are all derived from the same $X$, this is **self-attention**; if instead $K, V$ come from a second sequence $Y$, it is **cross-attention**.

## Theorems

**Theorem 1 (Attention as a soft database lookup).** *For each query index $i$, the output row $\mathrm{Attn}(Q,K,V)_i$ is a convex combination of the value rows $V_j$.*

*Proof.* By construction $A_{i,:}$ is the softmax of a real vector, so $A_{ij} \in (0,1)$ and $\sum_j A_{ij} = 1$ (Chapter 16). Then $(A V)_i = \sum_j A_{ij} V_j \in \mathrm{conv}\{V_1, \dots, V_T\}$. $\square$

**Theorem 2 (Variance of dot products controls softmax temperature).** *Suppose $Q_{i,:}$ and $K_{j,:}$ have i.i.d.\ entries with mean $0$ and variance $1$, independent of each other. Then*
$$
\mathbb{E}[Q_i \cdot K_j] = 0, \qquad \mathrm{Var}(Q_i \cdot K_j) = d_k.
$$
*Hence $(QK^\top)_{ij}/\sqrt{d_k}$ has unit variance.*

*Proof.* Write $S = \sum_{k=1}^{d_k} Q_{ik} K_{jk}$. By linearity and independence,
$$
\mathbb{E}[S] = \sum_k \mathbb{E}[Q_{ik}]\,\mathbb{E}[K_{jk}] = 0.
$$
For the second moment, since terms with distinct $k \neq k'$ are mean-zero and independent (so their cross terms vanish in expectation),
$$
\mathbb{E}[S^2] = \sum_{k} \mathbb{E}[Q_{ik}^2 K_{jk}^2] + \sum_{k \neq k'} \mathbb{E}[Q_{ik} K_{jk}]\mathbb{E}[Q_{ik'} K_{jk'}] = \sum_{k} 1 \cdot 1 + 0 = d_k.
$$
Therefore $\mathrm{Var}(S) = d_k$, and dividing by $\sqrt{d_k}$ rescales to unit variance. $\square$

**Corollary 3 (Saturation without scaling).** *Without the $1/\sqrt{d_k}$ factor, the logits $S$ feeding softmax have standard deviation $\sqrt{d_k}$, growing without bound. As $d_k \to \infty$, the maximum logit dominates and softmax concentrates on a single index. The Jacobian $\partial \mathrm{softmax}_i / \partial s_j = \mathrm{softmax}_i (\delta_{ij} - \mathrm{softmax}_j)$ then collapses, vanishing the gradient w.r.t. $Q, K$.*

The factor $1/\sqrt{d_k}$ is therefore a *temperature normalization* keeping softmax in the well-conditioned regime where it has rank-$(T-1)$ Jacobian and informative gradients.

**Theorem 4 (Permutation equivariance).** *Let $\pi$ be a permutation of $\{1,\dots,T\}$ with permutation matrix $P_\pi$. Then*
$$
\mathrm{Attn}(P_\pi Q, P_\pi K, P_\pi V) = P_\pi \, \mathrm{Attn}(Q, K, V).
$$

*Proof.* The pre-softmax matrix becomes $(P_\pi Q)(P_\pi K)^\top / \sqrt{d_k} = P_\pi (Q K^\top) P_\pi^\top / \sqrt{d_k}$. Row-wise softmax commutes with row permutation: $\mathrm{softmax}(P_\pi M P_\pi^\top) = P_\pi \,\mathrm{softmax}(M P_\pi^\top \cdot)$, but more cleanly, applied row-by-row, $A' = P_\pi A P_\pi^\top$. Then $A' (P_\pi V) = P_\pi A P_\pi^\top P_\pi V = P_\pi A V$. $\square$

This symmetry is *too strong*: attention cannot tell "the cat sat" from "sat cat the". Chapter 24 will repair this with positional encodings.

**Theorem 5 (O(1) gradient depth).** *Fix output position $i$ and input position $j$. The Jacobian $\partial (AV)_i / \partial V_j = A_{ij} I_{d_v}$ has spectral norm $A_{ij} \in (0,1)$, regardless of $|i-j|$.*

*Proof.* $(AV)_i = \sum_\ell A_{i\ell} V_\ell$, so $\partial (AV)_i / \partial V_j = A_{ij} I$. $\square$

Compare with Chapter 20: an RNN has $\|\partial h_i/\partial h_j\| \lesssim \rho^{|i-j|}$. Attention is exponentially better in distance, paying instead $O(T^2)$ in compute for the $A$ matrix.

## Code sketch

In `cells.json` we (i) implement attention from scratch, (ii) empirically confirm $\mathrm{Var}(Q\cdot K) = d_k$, (iii) measure entropy collapse without scaling, (iv) verify permutation equivariance to floating-point, and (v) compute gradient magnitude across $T=50$ for both attention and an RNN.

## Connection to LLMs

Scaled dot-product attention is **the** primitive of every transformer. Chapter 22 stacks $h$ heads of it (multi-head attention); Chapter 23 wires attention together with MLPs and residual streams into the transformer block; Chapter 24 fixes Theorem 4 with positional encodings. Production-scale variants — Flash Attention (IO-aware tiling), multi-query and grouped-query attention (sharing $K, V$ across heads), sliding-window and linear attention — all preserve the scaled-softmax form derived here. Whenever you call an LLM, every token interaction inside it is an instance of $\mathrm{softmax}(QK^\top/\sqrt{d_k}) V$.
