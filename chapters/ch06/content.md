## Motivation

Chapter 5 built vector spaces and linear maps as raw algebraic objects. To do *geometry* — to talk about *length*, *angle*, *orthogonality*, and *best approximation* — we need additional structure: an **inner product**. From this single addition we will derive the Cauchy–Schwarz inequality, the triangle inequality, the spectral theorem, the singular value decomposition (SVD), and the Eckart–Young low-rank approximation theorem. These are the workhorses behind PCA, attention, and LoRA fine-tuning of LLMs.

## Definitions

**Definition (Inner product).** A function $\langle \cdot, \cdot\rangle : V \times V \to \mathbb{R}$ on a real vector space $V$ is an *inner product* if for all $\mathbf{x},\mathbf{y},\mathbf{z}\in V$ and $a,b\in\mathbb{R}$:

1. *Symmetry*: $\langle \mathbf{x},\mathbf{y}\rangle = \langle \mathbf{y},\mathbf{x}\rangle$.
2. *Linearity in the first argument*: $\langle a\mathbf{x}+b\mathbf{y},\mathbf{z}\rangle = a\langle \mathbf{x},\mathbf{z}\rangle + b\langle \mathbf{y},\mathbf{z}\rangle$.
3. *Positive-definiteness*: $\langle \mathbf{x},\mathbf{x}\rangle \geq 0$, with equality iff $\mathbf{x}=\mathbf{0}$.

On $\mathbb{R}^n$ the canonical example is the **dot product** $\langle \mathbf{x},\mathbf{y}\rangle = \sum_{i=1}^n x_i y_i = \mathbf{x}^T\mathbf{y}$.

**Definition (Induced norm).** $\|\mathbf{x}\| := \sqrt{\langle \mathbf{x},\mathbf{x}\rangle}$.

**Definition (Orthogonality).** Vectors $\mathbf{x},\mathbf{y}$ are *orthogonal* if $\langle \mathbf{x},\mathbf{y}\rangle = 0$. A set $\{\mathbf{q}_i\}$ is *orthonormal* if $\langle \mathbf{q}_i,\mathbf{q}_j\rangle = \delta_{ij}$. An *orthonormal basis* is an orthonormal spanning set.

**Definition (Eigenvalue, eigenvector).** For $A\in\mathbb{R}^{n\times n}$, a scalar $\lambda$ and nonzero $\mathbf{v}\in\mathbb{R}^n$ form an eigenpair if $A\mathbf{v}=\lambda\mathbf{v}$. The *characteristic polynomial* is $p_A(\lambda) := \det(A-\lambda I)$; its roots are exactly the eigenvalues.

**Definition (Symmetric, positive (semi)definite).** $A$ is *symmetric* if $A^T=A$; *positive semidefinite* (PSD) if symmetric with $\mathbf{x}^T A \mathbf{x}\geq 0$ for all $\mathbf{x}$; *positive definite* if strict inequality holds for $\mathbf{x}\ne 0$.

**Definition (SVD).** A *singular value decomposition* of $A\in\mathbb{R}^{m\times n}$ is a factorization $A = U\Sigma V^T$ where $U\in\mathbb{R}^{m\times m}$ and $V\in\mathbb{R}^{n\times n}$ are orthogonal ($U^T U=I$, $V^T V=I$) and $\Sigma\in\mathbb{R}^{m\times n}$ is "diagonal" with nonnegative entries $\sigma_1\geq\sigma_2\geq\cdots\geq 0$.

## Theorems and Proofs

**Theorem (Cauchy–Schwarz).** For all $\mathbf{x},\mathbf{y}\in V$, $|\langle \mathbf{x},\mathbf{y}\rangle| \leq \|\mathbf{x}\|\,\|\mathbf{y}\|$.

*Proof.* If $\mathbf{y}=\mathbf{0}$ both sides are $0$. Otherwise, for every $t\in\mathbb{R}$,
$$0 \leq \|\mathbf{x}-t\mathbf{y}\|^2 = \langle \mathbf{x}-t\mathbf{y},\mathbf{x}-t\mathbf{y}\rangle = \|\mathbf{x}\|^2 - 2t\langle \mathbf{x},\mathbf{y}\rangle + t^2\|\mathbf{y}\|^2.$$
This is a quadratic in $t$ that is nonnegative everywhere, so its discriminant is $\leq 0$:
$$(2\langle \mathbf{x},\mathbf{y}\rangle)^2 - 4\|\mathbf{x}\|^2\|\mathbf{y}\|^2 \leq 0,$$
i.e. $\langle \mathbf{x},\mathbf{y}\rangle^2 \leq \|\mathbf{x}\|^2\|\mathbf{y}\|^2$. Take square roots. $\square$

**Corollary (Triangle inequality).** $\|\mathbf{x}+\mathbf{y}\| \leq \|\mathbf{x}\| + \|\mathbf{y}\|$.

*Proof.* $\|\mathbf{x}+\mathbf{y}\|^2 = \|\mathbf{x}\|^2 + 2\langle \mathbf{x},\mathbf{y}\rangle + \|\mathbf{y}\|^2 \leq \|\mathbf{x}\|^2 + 2\|\mathbf{x}\|\|\mathbf{y}\| + \|\mathbf{y}\|^2 = (\|\mathbf{x}\|+\|\mathbf{y}\|)^2$, using Cauchy–Schwarz. $\square$

**Theorem (Spectral theorem, real symmetric case).** Every symmetric $A\in\mathbb{R}^{n\times n}$ admits a factorization $A = Q\Lambda Q^T$ with $Q$ orthogonal and $\Lambda$ real diagonal.

*Proof (induction on $n$).* For $n=1$ trivial. Assume the result for $n-1$. The Rayleigh quotient $R(\mathbf{x}) := \mathbf{x}^T A\mathbf{x}$ is continuous on the unit sphere $S^{n-1}=\{\mathbf{x}:\|\mathbf{x}\|=1\}$, which is compact, so $R$ attains a maximum at some $\mathbf{v}_1\in S^{n-1}$ with value $\lambda_1$. By Lagrange multipliers (or by directly differentiating $R(\mathbf{v}_1+t\mathbf{w})$ along any tangent $\mathbf{w}\perp \mathbf{v}_1$), $A\mathbf{v}_1 = \lambda_1\mathbf{v}_1$. The eigenvalue is automatically real because $\lambda_1 = \mathbf{v}_1^T A\mathbf{v}_1\in\mathbb{R}$ and $\mathbf{v}_1$ is real. (Equivalently, for symmetric $A$, $\langle A\mathbf{x},\mathbf{x}\rangle = \langle \mathbf{x},A\mathbf{x}\rangle$ forces complex eigenvalues to be real: if $A\mathbf{z}=\mu\mathbf{z}$ over $\mathbb{C}$, then $\mu \overline{\mathbf{z}}^T\mathbf{z} = \overline{\mathbf{z}}^T A\mathbf{z} = (A\overline{\mathbf{z}})^T\mathbf{z} = \overline{\mu}\overline{\mathbf{z}}^T\mathbf{z}$, so $\mu=\overline{\mu}$.)

Let $W = \{\mathbf{v}_1\}^\perp$, an $(n-1)$-dimensional subspace. For $\mathbf{w}\in W$, $\langle A\mathbf{w},\mathbf{v}_1\rangle = \langle \mathbf{w},A\mathbf{v}_1\rangle = \lambda_1\langle \mathbf{w},\mathbf{v}_1\rangle = 0$, so $A$ maps $W\to W$. The restriction $A|_W$ is symmetric with respect to the inherited inner product. By induction, choose an orthonormal eigenbasis $\mathbf{v}_2,\dots,\mathbf{v}_n$ of $W$. Then $\mathbf{v}_1,\dots,\mathbf{v}_n$ is an orthonormal eigenbasis of $A$; assemble into $Q=[\mathbf{v}_1\,\cdots\,\mathbf{v}_n]$ and $\Lambda=\mathrm{diag}(\lambda_1,\dots,\lambda_n)$. $\square$

**Theorem (Existence of SVD).** Every $A\in\mathbb{R}^{m\times n}$ has an SVD.

*Sketch.* The matrix $A^T A$ is symmetric and PSD: $\mathbf{x}^T A^T A\mathbf{x} = \|A\mathbf{x}\|^2\geq 0$. By the spectral theorem, $A^T A = V\Lambda V^T$ with $V$ orthogonal and $\Lambda=\mathrm{diag}(\lambda_1,\dots,\lambda_n)$, $\lambda_i\geq 0$. Set $\sigma_i := \sqrt{\lambda_i}$ in decreasing order. For each $i$ with $\sigma_i>0$ define $\mathbf{u}_i := A\mathbf{v}_i/\sigma_i$. Then $\langle \mathbf{u}_i,\mathbf{u}_j\rangle = (\sigma_i\sigma_j)^{-1}\mathbf{v}_i^T A^T A\mathbf{v}_j = \delta_{ij}$, so $\{\mathbf{u}_i\}$ is orthonormal in $\mathbb{R}^m$; extend to an orthonormal basis $U$. By construction $A V = U\Sigma$, hence $A = U\Sigma V^T$. $\square$

**Theorem (Eckart–Young).** Let $A=U\Sigma V^T$ with singular values $\sigma_1\geq\cdots\geq\sigma_r>0$. The truncated SVD $A_k := \sum_{i=1}^k \sigma_i\mathbf{u}_i\mathbf{v}_i^T$ minimises $\|A-B\|_F$ (and $\|A-B\|_2$) over all rank-$k$ matrices $B$, with $\|A-A_k\|_F^2 = \sum_{i>k}\sigma_i^2$.

*Sketch.* Frobenius norm is unitarily invariant: $\|A-B\|_F = \|U^T(A-B)V\|_F$. Reducing to diagonal $\Sigma$, the problem becomes: among rank-$k$ matrices, minimise $\|\Sigma-C\|_F^2 = \sum (\sigma_i - c_{ii})^2 + \text{(off-diagonal)}^2$. Optimum sets the top-$k$ diagonal of $C$ equal to the top-$k$ singular values and zeroes the rest, recovering $A_k$. The operator-norm version uses Courant–Fischer / minimax. $\square$

## Code Sketch

The accompanying notebook (`cells.json`) numerically (i) verifies Cauchy–Schwarz on 50 random pairs in $\mathbb{R}^{10}$, (ii) computes a symmetric eigendecomposition via `np.linalg.eigh` and checks $A\mathbf{v}_i=\lambda_i\mathbf{v}_i$ and orthonormality of eigenvectors, (iii) computes an SVD with `np.linalg.svd` and reconstructs $A$, and (iv) shows the Frobenius error of rank-1 and rank-2 approximations decreasing monotonically as predicted by Eckart–Young.

## Connection to LLMs

Inner products and SVD are not abstract decoration; they are the geometric backbone of every transformer.

- **Attention** (Chapters 21–22). The score matrix $QK^T$ is the matrix of pairwise inner products between query and key embeddings. Long-context efficiency research (linear attention, Performers, low-rank attention) hinges on the empirical observation that $QK^T$ is approximately low-rank — which by Eckart–Young is best captured by truncated SVD-style factorisations.
- **PCA / interpretability.** SVD of an embedding matrix produces principal components; the leading singular vectors expose semantic axes (e.g. sentiment, syntax) and underpin probing experiments.
- **LoRA fine-tuning.** A pretrained weight $W_0\in\mathbb{R}^{m\times n}$ is updated as $W_0 + BA$ with $B\in\mathbb{R}^{m\times r}$, $A\in\mathbb{R}^{r\times n}$, $r\ll\min(m,n)$. This is *literally* a rank-$r$ correction; Eckart–Young guarantees it is the best Frobenius-norm approximation of the ideal full-rank update at that rank.
- **Spectral norm regularisation.** Bounding $\sigma_1(W)$ controls Lipschitz constants and stabilises training (Chapter on optimisation).

Every subsequent geometric statement in this book — projections, least squares, gradient flow on weight matrices, contractive maps for stability — descends from the inequalities and decompositions proven here.
