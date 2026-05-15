## Motivation

A linear map $x \mapsto Wx + b$ (Chapter 5) can only carve $\mathbb{R}^n$ into half-spaces and stretch them affinely. The continuous functions on a compact set $K \subset \mathbb{R}^n$ (Chapter 4) form a vastly richer space. To bridge the gap we interleave linear maps with a fixed pointwise nonlinearity. The resulting object — a *multilayer perceptron* (MLP) — turns out to be expressive enough to approximate **every** continuous function uniformly on $K$. This is the engine inside every transformer's feed-forward block.

## Definitions

**Definition (MLP).** Let $L \geq 1$ and widths $d_0, d_1, \dots, d_L \in \mathbb{N}$. A *multilayer perceptron of depth $L$* with activation $\sigma : \mathbb{R} \to \mathbb{R}$ (applied componentwise) is the function $f : \mathbb{R}^{d_0} \to \mathbb{R}^{d_L}$ defined by
$$
h^{(0)} = x, \qquad h^{(\ell)} = \sigma\!\big(W^{(\ell)} h^{(\ell-1)} + b^{(\ell)}\big) \text{ for } 1 \leq \ell \leq L-1,
$$
$$
f(x) = W^{(L)} h^{(L-1)} + b^{(L)},
$$
with $W^{(\ell)} \in \mathbb{R}^{d_\ell \times d_{\ell-1}}$ and $b^{(\ell)} \in \mathbb{R}^{d_\ell}$. The number $\max_\ell d_\ell$ is the *width*; $L$ is the *depth*.

**Definition (universal approximator).** A class $\mathcal{F} \subset C(K, \mathbb{R})$ is **dense** in $C(K)$ — i.e. a *universal approximator* — if for every $f \in C(K)$ and $\varepsilon > 0$ there exists $g \in \mathcal{F}$ with $\sup_{x \in K} |f(x) - g(x)| < \varepsilon$.

**Definition (sigmoidal / discriminatory).** A measurable $\sigma : \mathbb{R} \to \mathbb{R}$ is *sigmoidal* if $\sigma(t) \to 1$ as $t \to +\infty$ and $\sigma(t) \to 0$ as $t \to -\infty$. It is *discriminatory* if for every signed regular Borel measure $\mu$ on $K = [0,1]^n$,
$$
\int_K \sigma(w^\top x + b)\, d\mu(x) = 0 \text{ for all } w \in \mathbb{R}^n,\ b \in \mathbb{R} \;\Longrightarrow\; \mu = 0.
$$

## Theorems

### 1. Universal Approximation (Cybenko, 1989; Hornik, 1989)

**Theorem.** Let $\sigma$ be a continuous sigmoidal function and $K \subset \mathbb{R}^n$ compact. The class
$$
\mathcal{F}_\sigma = \Big\{ g(x) = \sum_{j=1}^{N} \alpha_j\, \sigma(w_j^\top x + b_j) \;\Big|\; N \in \mathbb{N},\ \alpha_j, b_j \in \mathbb{R},\ w_j \in \mathbb{R}^n \Big\}
$$
of single-hidden-layer MLPs is dense in $C(K)$.

*Proof sketch (Hahn–Banach + Riesz).* Suppose, for contradiction, $\mathcal{F}_\sigma$ is **not** dense. Then $S := \overline{\mathcal{F}_\sigma}$ is a proper closed subspace of the Banach space $C(K)$. Pick any $f_0 \in C(K) \setminus S$. By the **Hahn–Banach theorem** there exists a continuous linear functional $\Lambda \in C(K)^*$ with $\Lambda \not\equiv 0$ but $\Lambda|_S = 0$.

By the **Riesz representation theorem**, $\Lambda$ is given by integration against a finite signed regular Borel measure $\mu$ on $K$:
$$
\Lambda(g) = \int_K g(x)\, d\mu(x), \qquad g \in C(K).
$$
Since each ridge function $x \mapsto \sigma(w^\top x + b) \in \mathcal{F}_\sigma \subset S$,
$$
\int_K \sigma(w^\top x + b)\, d\mu(x) = 0 \quad \forall\, w, b.
$$
Cybenko shows any continuous sigmoidal $\sigma$ is **discriminatory** (proven via Fourier analysis: pushing $\sigma$ toward step functions and reading off vanishing characteristic functions of half-spaces forces $\mu \equiv 0$). Hence $\mu = 0$, so $\Lambda \equiv 0$, contradiction. $\square$

### 2. Linear-only networks collapse

**Proposition.** If $\sigma = \mathrm{id}$, then any depth-$L$ MLP is itself a single affine map.

*Proof.* By induction on $\ell$, $h^{(\ell)} = W^{(\ell)} h^{(\ell-1)} + b^{(\ell)}$. Unrolling,
$$
f(x) = W' x + b', \quad W' = \prod_{\ell=L}^{1} W^{(\ell)}, \quad b' = \sum_{\ell=1}^{L} \Big(\prod_{k=L}^{\ell+1} W^{(k)}\Big) b^{(\ell)}.
$$
Therefore $f$ is affine, regardless of $L$. Affine maps are not dense in $C(K)$ (e.g. cannot approximate $\sin$ on $[-1,1]$ to error $< 0.1$). $\square$

So the nonlinearity $\sigma$ is not cosmetic: without it, depth buys nothing.

### 3. Depth separation (Telgarsky, 2016)

**Theorem (informal).** For every $k \geq 1$ there is a function $f_k : [0,1] \to [0,1]$ realizable exactly by a ReLU network of depth $O(k^3)$ and width $O(1)$, such that **any** ReLU network of depth $\leq k$ approximating $f_k$ in $L^1$ to error $< 1/32$ requires width $\geq 2^k$.

*Sketch.* Take $f_k$ to be the $k$-fold composition $\Delta^{\circ k}$ of the triangular "sawtooth" $\Delta(x) = 2x$ on $[0, 1/2]$, $\Delta(x) = 2(1-x)$ on $[1/2, 1]$. Each composition doubles the number of monotone pieces, so $f_k$ has $2^k$ pieces. A shallow ReLU net of depth $\leq k$ realizes a piecewise linear function with at most $\mathrm{poly}(\text{width})^k$ pieces; matching $2^k$ pieces forces exponential width. Composition realizes the same function with width and depth growing linearly in $k$. $\square$

The moral: UAT says shallow nets *can* approximate everything, but composition (depth) is **exponentially more parameter-efficient** for naturally hierarchical targets.

## Code sketch

The companion `cells.json` notebook (i) hand-builds a numpy MLP with $\tanh$ activation, (ii) fits a 32-unit MLP to $\sin(2\pi x)$ on $[-1,1]$ via least-squares on hidden features and verifies $\|f - \hat f\|_\infty < 0.05$, (iii) compares width-32-depth-1 against width-8-depth-3 at fixed parameter budget and observes lower error from the deeper net, and (iv) numerically confirms the linear-only collapse $f(x) = W' x + b'$.

## Connection to LLMs

Every transformer block contains a position-wise feed-forward module
$$
\mathrm{FFN}(x) = W_2\, \sigma(W_1 x + b_1) + b_2,
$$
i.e. a **single-hidden-layer MLP** applied independently to each token. This is exactly the universal approximator from Theorem 1: attention shuffles information *across* positions, while the FFN performs nonlinear pointwise computation that — by Cybenko — can in principle realize any continuous transformation of the embedding. Modern LLMs make $W_1$ wide ($4d$ or $8d$ inner dim) precisely to exploit Theorem 1's expressive guarantee, while stacking $L$ such blocks exploits Theorem 3's exponential depth gain. We will assemble the full block in Chapter 23.
