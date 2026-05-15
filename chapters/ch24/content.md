## Motivation

In Chapter 21 we proved that scaled dot-product attention is **permutation-equivariant**: if $P$ is any permutation matrix and $X \in \mathbb{R}^{T \times d}$ a sequence of token embeddings, then
$$
\mathrm{Attention}(PX, PX, PX) = P \cdot \mathrm{Attention}(X, X, X).
$$
A direct corollary is catastrophic for language modeling: shuffling the tokens of a sentence and feeding them to attention produces the *same* set of output vectors, only re-indexed. Attention by itself sees a *bag of tokens*, not a sequence. To recover order, we must inject the position $t \in \{0, 1, \ldots, T-1\}$ into the representations.

Three approaches dominate practice: **sinusoidal** PEs (Vaswani et al. 2017, original Transformer), **learned** PEs (BERT, GPT-2), and **rotary position embedding** or **RoPE** (Su et al. 2021), used by Llama, Mistral, Claude, Gemini, and most modern LLMs. This chapter derives all three and proves the central RoPE relative-position theorem.

## Definitions

**Definition (Sinusoidal PE).** For position $\mathrm{pos} \in \mathbb{N}$ and embedding dimension $d$ (assumed even), define $\mathrm{PE}(\mathrm{pos}) \in \mathbb{R}^d$ by
$$
\mathrm{PE}(\mathrm{pos}, 2i) = \sin(\mathrm{pos} \cdot \theta_i), \quad
\mathrm{PE}(\mathrm{pos}, 2i+1) = \cos(\mathrm{pos} \cdot \theta_i),
\qquad \theta_i := 10000^{-2i/d},\ i = 0, \ldots, d/2 - 1.
$$
The position-aware token representation is $x_t = E[\mathrm{token}_t] + \mathrm{PE}(t)$.

**Definition (Learned PE).** Fix a maximum context length $T_{\max}$. Let $P \in \mathbb{R}^{T_{\max} \times d}$ be a parameter matrix. The position embedding at $t$ is the $t$-th row $P_t$, learned end-to-end. This is what BERT and GPT-2 use.

**Definition (RoPE).** Group the $d$ coordinates of a query or key vector into $d/2$ consecutive pairs. For pair $i$ at position $\mathrm{pos}$, define the $2 \times 2$ rotation
$$
R_{\mathrm{pos}, i} \;=\; \begin{pmatrix} \cos(\mathrm{pos}\,\theta_i) & -\sin(\mathrm{pos}\,\theta_i) \\ \sin(\mathrm{pos}\,\theta_i) & \cos(\mathrm{pos}\,\theta_i) \end{pmatrix},
\qquad \theta_i = 10000^{-2i/d}.
$$
Let $R_{\mathrm{pos}} \in \mathbb{R}^{d \times d}$ be the block-diagonal matrix with these rotations on the diagonal. Then, *unlike* sinusoidal PEs, RoPE is **not** added to embeddings; it is applied multiplicatively to the projected queries and keys:
$$
q_m^{\mathrm{rot}} = R_m\, q_m, \qquad k_n^{\mathrm{rot}} = R_n\, k_n.
$$

## Theorems and Proofs

**Theorem 24.1 (Sinusoidal PE encodes shift as rotation).** *For every offset $k \in \mathbb{Z}$ there exists a fixed block-diagonal matrix $M_k \in \mathbb{R}^{d \times d}$, independent of $\mathrm{pos}$, such that*
$$
\mathrm{PE}(\mathrm{pos} + k) \;=\; M_k \cdot \mathrm{PE}(\mathrm{pos}).
$$

*Proof.* Restrict to the $i$-th coordinate pair. By the trigonometric addition formulas,
$$
\sin\big((\mathrm{pos}+k)\theta_i\big) = \sin(\mathrm{pos}\,\theta_i)\cos(k\theta_i) + \cos(\mathrm{pos}\,\theta_i)\sin(k\theta_i),
$$
$$
\cos\big((\mathrm{pos}+k)\theta_i\big) = \cos(\mathrm{pos}\,\theta_i)\cos(k\theta_i) - \sin(\mathrm{pos}\,\theta_i)\sin(k\theta_i).
$$
In matrix form, with $c := \cos(k\theta_i),\ s := \sin(k\theta_i)$:
$$
\begin{pmatrix} \sin((\mathrm{pos}+k)\theta_i) \\ \cos((\mathrm{pos}+k)\theta_i) \end{pmatrix}
= \begin{pmatrix} c & s \\ -s & c \end{pmatrix}
\begin{pmatrix} \sin(\mathrm{pos}\,\theta_i) \\ \cos(\mathrm{pos}\,\theta_i) \end{pmatrix}.
$$
The $2 \times 2$ matrix depends only on $k$ and $\theta_i$, not on $\mathrm{pos}$. Stacking these blocks for $i = 0, \ldots, d/2 - 1$ gives the claimed $M_k$. $\square$

**Theorem 24.2 (RoPE relative-position property).** *For all $m, n \in \mathbb{N}$ and $q_m, k_n \in \mathbb{R}^d$,*
$$
\big\langle R_m q_m,\ R_n k_n \big\rangle \;=\; q_m^{\top}\, R_{n - m}\, k_n.
$$

*Proof.* We use two block-wise facts. First, each $R_{\mathrm{pos}, i}$ is a planar rotation, so
$$
R_{\mathrm{pos}, i}^{\top} = R_{-\mathrm{pos}, i}, \qquad R_{a, i}\, R_{b, i} = R_{a + b, i}, \tag{$\star$}
$$
the latter being the standard angle-addition identity for $\mathrm{SO}(2)$. Lifting to block-diagonal matrices preserves these: $R_m^{\top} = R_{-m}$ and $R_a R_b = R_{a+b}$.

Now compute:
$$
\langle R_m q_m,\ R_n k_n \rangle
= (R_m q_m)^{\top}(R_n k_n)
= q_m^{\top} R_m^{\top} R_n k_n
= q_m^{\top} R_{-m} R_n k_n
= q_m^{\top} R_{n - m} k_n. \quad\square
$$

**Corollary 24.3 (Translation invariance).** *For any constant $c \in \mathbb{Z}$, $\langle R_{m+c} q_m, R_{n+c} k_n \rangle = \langle R_m q_m, R_n k_n \rangle$.*

*Proof.* Apply Theorem 24.2 to both sides: the right-hand side is $q_m^{\top} R_{(n+c) - (m+c)} k_n = q_m^{\top} R_{n - m} k_n$, which equals $\langle R_m q_m, R_n k_n \rangle$. $\square$

**Proposition 24.4 (No learnable RoPE parameters).** *The map $(q, k, \mathrm{pos}) \mapsto R_{\mathrm{pos}} q, R_{\mathrm{pos}} k$ contains no trainable weights: the angles $\theta_i = 10000^{-2i/d}$ are fixed.*

*Proof.* By inspection of the definition. $\square$

This is operationally important: a model trained with context length $T_{\mathrm{train}}$ can be evaluated at $\mathrm{pos} > T_{\mathrm{train}}$ since $R_{\mathrm{pos}}$ is defined for all integers. Learned PEs cannot do this — there is simply no row $P_t$ for $t \geq T_{\max}$. RoPE thus offers a principled (though imperfect) path to length extrapolation, refined by NTK-aware scaling, YaRN, and Position Interpolation in Chapter 27.

## Code Sketch

In `cells.json` we (i) confirm permutation-invariance of vanilla attention numerically, (ii) implement and visualize the sinusoidal PE matrix, (iii) verify Theorem 24.1 by checking $\|\mathrm{PE}(\mathrm{pos}+k) - M_k \mathrm{PE}(\mathrm{pos})\| < 10^{-12}$, (iv) implement RoPE for $d=8, T=6$ and confirm Theorem 24.2 by comparing $\langle R_m q_m, R_n k_n \rangle$ against $q_m^{\top} R_{n-m} k_n$, and (v) verify Corollary 24.3 by shifting all positions by $c \in \{1, 5, 10\}$ and observing identical attention scores.

## Connection to LLMs

Vaswani et al.'s original Transformer used sinusoidal PEs. BERT and GPT-2 switched to learned PEs, trading off extrapolation for slight accuracy gains within the training context. From Llama (2023) onward, RoPE is the de facto standard: the relative-position property of Theorem 24.2 means attention scores depend only on token *separation*, matching the linguistic intuition that "the dog *across the street*" should attend the same way regardless of where the phrase appears. Combined with extrapolation tricks (Chapter 27), RoPE has enabled context windows from $2$K (Llama 1) to $1$M+ tokens (Claude, Gemini). Chapter 23 introduced multi-head attention; positional encoding is applied per-head before the dot product, so RoPE composes cleanly with everything that follows.
