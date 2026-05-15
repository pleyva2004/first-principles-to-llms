## Motivation

Every transformer layer is, between its bias terms and nonlinearities, a *linear map between finite-dimensional real vector spaces*. The "hidden dimension" $d$ is just $\dim \mathbb{R}^d$; the "weight matrix" $W \in \mathbb{R}^{m \times n}$ is just the matrix representation of a linear map $\mathbb{R}^n \to \mathbb{R}^m$ in the standard bases; "residual connections" are addition in a vector space; "low-rank adapters" are statements about the *image* of a linear map. Before we can speak of attention (Chapter 21) or embeddings (Chapter 19), we need the grammar of vector spaces, bases, dimension, kernels, images, and the rank--nullity theorem. We rely on Chapter 1 (sets, functions, logic) throughout.

## Definitions

A **field** $\mathbb{F}$ is a set with two binary operations $+, \cdot$ such that $(\mathbb{F}, +)$ is an abelian group with identity $0$, $(\mathbb{F} \setminus \{0\}, \cdot)$ is an abelian group with identity $1$, and multiplication distributes over addition. The canonical example is $\mathbb{F} = \mathbb{R}$.

A **vector space** $V$ over $\mathbb{F}$ is a set equipped with addition $+ : V \times V \to V$ and scalar multiplication $\cdot : \mathbb{F} \times V \to V$ satisfying the **eight axioms**, for all $u, v, w \in V$ and $a, b \in \mathbb{F}$:
1. $(u + v) + w = u + (v + w)$ (associativity);
2. $u + v = v + u$ (commutativity);
3. $\exists\, 0 \in V$ such that $v + 0 = v$ for all $v$ (zero);
4. $\forall v \in V,\ \exists (-v) \in V$ with $v + (-v) = 0$ (additive inverse);
5. $a \cdot (u + v) = a \cdot u + a \cdot v$ (distributivity over vectors);
6. $(a + b) \cdot v = a \cdot v + b \cdot v$ (distributivity over scalars);
7. $(ab) \cdot v = a \cdot (b \cdot v)$ (compatibility);
8. $1 \cdot v = v$ (scalar identity).

A **subspace** $U \subset V$ is a subset closed under $+$ and scalar multiplication that contains $0$. A **linear combination** of $v_1, \ldots, v_k \in V$ is any vector $\sum_{i=1}^k a_i v_i$ with $a_i \in \mathbb{F}$. The **span** of $S \subset V$ is $\mathrm{span}(S) := \{\sum a_i v_i : v_i \in S, a_i \in \mathbb{F}\}$, the smallest subspace containing $S$. A finite set $\{v_1, \ldots, v_k\}$ is **linearly independent** iff $\sum a_i v_i = 0 \Rightarrow a_1 = \cdots = a_k = 0$. A **basis** of $V$ is a linearly independent spanning set. The **dimension** $\dim V$ is the cardinality of any basis (well-defined by Theorem 3 below).

A **linear map** $T : V \to W$ between vector spaces over the same $\mathbb{F}$ satisfies $T(au + bv) = a T(u) + b T(v)$ for all $u, v \in V$, $a, b \in \mathbb{F}$. Its **kernel** is $\ker T := \{v \in V : T(v) = 0\}$ and its **image** is $\mathrm{im}\,T := \{T(v) : v \in V\}$; both are subspaces. Given bases $(e_1, \ldots, e_n)$ of $V = \mathbb{R}^n$ and $(f_1, \ldots, f_m)$ of $W = \mathbb{R}^m$, the **matrix representation** of $T$ has $j$-th column equal to the coordinate vector of $T(e_j)$ in the $f$-basis: $A \in \mathbb{R}^{m \times n}$ with $T(e_j) = \sum_i A_{ij} f_i$.

## Theorems and proofs

**Theorem 1 (Steinitz exchange lemma).** *Let $V$ be a vector space over $\mathbb{F}$. If $\{v_1, \ldots, v_m\}$ is linearly independent and $\{w_1, \ldots, w_n\}$ spans $V$, then $m \leq n$, and after reindexing the $w_j$ we may replace $m$ of them by $v_1, \ldots, v_m$ so that the resulting set still spans $V$.*

*Proof.* By induction on $m$. The case $m = 0$ is trivial. Assume the claim for $m - 1$: after reindexing, $\{v_1, \ldots, v_{m-1}, w_m, \ldots, w_n\}$ spans $V$ (so in particular $m - 1 \leq n$). Then $v_m \in V$ is a linear combination $v_m = \sum_{i < m} a_i v_i + \sum_{j \geq m} b_j w_j$. If all $b_j = 0$, then $v_m \in \mathrm{span}(v_1, \ldots, v_{m-1})$, contradicting linear independence of $\{v_1, \ldots, v_m\}$. So some $b_{j_0} \neq 0$; in particular such a $j_0 \in \{m, \ldots, n\}$ exists, forcing $n \geq m$. Reindex so $j_0 = m$. Solving for $w_m$,
$$w_m = b_m^{-1}\!\Big(v_m - \sum_{i<m} a_i v_i - \sum_{j>m} b_j w_j\Big) \in \mathrm{span}(v_1, \ldots, v_m, w_{m+1}, \ldots, w_n).$$
Hence $\mathrm{span}(v_1, \ldots, v_m, w_{m+1}, \ldots, w_n) \supset \{v_1, \ldots, v_{m-1}, w_m, \ldots, w_n\}$, which spans $V$. So $\{v_1, \ldots, v_m, w_{m+1}, \ldots, w_n\}$ spans $V$. $\blacksquare$

**Theorem 2 (Spanning sets contain bases; independent sets extend to bases).** *In a finite-dimensional $V$: (a) any finite spanning set contains a basis; (b) any linearly independent set extends to a basis.*

*Proof.* (a) Take a finite spanning set $S$. If linearly dependent, some $w \in S$ lies in $\mathrm{span}(S \setminus \{w\})$, so $S \setminus \{w\}$ still spans. Iterate until linearly independent; the result is a basis. (b) Take an independent set $L$ and any finite spanning set $S$. Apply Theorem 1: replace $|L|$ elements of $S$ by $L$ to obtain a spanning set containing $L$; then apply (a) by removing dependent vectors only from the $S$-side. $\blacksquare$

**Theorem 3 (Invariance of dimension).** *Any two bases of a finite-dimensional $V$ have the same cardinality.*

*Proof.* Let $\mathcal{B}_1, \mathcal{B}_2$ be bases with $|\mathcal{B}_1| = m$, $|\mathcal{B}_2| = n$. $\mathcal{B}_1$ is independent and $\mathcal{B}_2$ spans, so by Theorem 1, $m \leq n$. Swapping roles, $n \leq m$. Hence $m = n$. $\blacksquare$

**Theorem 4 (Rank--nullity).** *Let $T : V \to W$ be linear with $\dim V = n < \infty$. Then $\dim \ker T + \dim \mathrm{im}\, T = n$.*

*Proof.* Let $(u_1, \ldots, u_k)$ be a basis of $\ker T$. By Theorem 2(b), extend to a basis $(u_1, \ldots, u_k, v_1, \ldots, v_{n-k})$ of $V$. We claim $(T v_1, \ldots, T v_{n-k})$ is a basis of $\mathrm{im}\,T$.

*Spanning.* Any $w \in \mathrm{im}\,T$ has $w = T(\sum a_i u_i + \sum b_j v_j) = \sum b_j T v_j$ since $T u_i = 0$.

*Independence.* Suppose $\sum c_j T v_j = 0$. Then $T(\sum c_j v_j) = 0$, so $\sum c_j v_j \in \ker T = \mathrm{span}(u_i)$. Write $\sum c_j v_j = \sum d_i u_i$, i.e. $\sum c_j v_j - \sum d_i u_i = 0$. Linear independence of the full basis forces all $c_j = 0$ (and all $d_i = 0$).

Hence $\dim \mathrm{im}\,T = n - k = n - \dim \ker T$. $\blacksquare$

## Code sketch

We implement `is_linearly_independent` and `dim_span` via numpy's matrix rank (which equals the dimension of the column span). We then build a $4 \times 6$ random integer matrix, compute its rank, extract a basis of $\ker A$ from the right singular vectors with zero singular values, verify $A v = 0$ for each null-space basis vector, and confirm the rank--nullity identity numerically. Finally, we verify the change-of-basis formula $A v$ versus $P (P^{-1} A P) (P^{-1} v)$ for a random invertible $P$.

## Connection to LLMs

A transformer with hidden dimension $d$ operates on the vector space $\mathbb{R}^d$. Each linear projection ($Q, K, V$ in attention; the up-and-down projections in the MLP) is a linear map between $\mathbb{R}^d$ and $\mathbb{R}^{d_k}$ or $\mathbb{R}^{d_{\mathrm{ff}}}$ (Chapter 21). The token embedding map of Chapter 19 is a linear map $\mathbb{R}^{|\mathcal{V}|} \to \mathbb{R}^d$ applied to one-hot inputs; equivalently, a row-lookup. The "rank" of an attention matrix and the "intrinsic dimension" of activations are statements about $\dim \mathrm{im}$. LoRA fine-tuning constrains weight updates to lie in a low-dimensional subspace --- a direct application of $\dim \mathrm{im}\,T \leq \min(m, n)$. Rank--nullity will reappear when we count parameters and degrees of freedom.
