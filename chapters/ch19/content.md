## Motivation

A transformer language model never sees the symbol *cat*. It sees an integer index $v \in \{0, 1, \ldots, V-1\}$ which is then mapped to a vector in $\mathbb{R}^d$. That vector is the only representation downstream layers ever touch: attention, MLPs, normalizations, residuals — everything is linear algebra on $\mathbb{R}^d$. The gateway between the discrete vocabulary $\mathcal{V}$ (Chapter 1) and the continuous geometry of $\mathbb{R}^d$ is the **embedding matrix**. This chapter shows that this gateway is not a fancy table lookup at all: it is the linear map of Chapter 5 applied to a one-hot vector. The trick of *weight tying* — sharing parameters between the input embedding and the output projection — then falls out as a very natural identification.

## Definitions

Let $\mathcal{V} = \{w_1, \ldots, w_V\}$ be a finite **vocabulary** (Chapter 1); we identify $\mathcal{V}$ with $\{0, 1, \ldots, V-1\}$ via tokenization order.

**Definition (One-hot encoding).** For $v \in \{0, \ldots, V-1\}$ the **one-hot vector** is $e_v \in \{0, 1\}^V$ with $e_v[i] = \mathbf{1}_{i = v}$. Equivalently, $e_v$ is the $v$-th standard basis vector of $\mathbb{R}^V$.

**Definition (Embedding matrix and lookup).** An **embedding matrix** is a parameter $E \in \mathbb{R}^{d \times V}$. Its columns $\{E_{:, 0}, \ldots, E_{:, V-1}\}$ are called **token embeddings**; the integer $d$ is the **embedding dimension** (also called $d_{\mathrm{model}}$). The **embedding lookup** is the function $\mathrm{emb} : \mathcal{V} \to \mathbb{R}^d$, $\mathrm{emb}(v) := E e_v$.

**Definition (Output projection / unembedding).** An **output projection** is a parameter $U \in \mathbb{R}^{V \times d}$ taking a hidden state $h \in \mathbb{R}^d$ to a logit vector $z := U h \in \mathbb{R}^V$. The probability over the vocabulary is $\hat p = \mathrm{softmax}(z)$ (Chapter 9, 17).

**Definition (Weight tying, Press & Wolf 2016; Inan et al. 2017).** A model with embedding matrix $E \in \mathbb{R}^{d \times V}$ and output projection $U \in \mathbb{R}^{V \times d}$ is **weight-tied** if $U = E^\top$. The two layers then share the same $dV$ scalar parameters.

## Theorems and proofs

**Theorem 1 (Lookup is a linear map).** *For every $v \in \{0, \ldots, V-1\}$, $E e_v = E_{:, v}$.*

*Proof.* By the definition of matrix–vector multiplication (Chapter 5), $(E e_v)_i = \sum_{j = 0}^{V-1} E_{ij}\, (e_v)_j = \sum_j E_{ij} \mathbf{1}_{j = v} = E_{iv}$ for every $i \in \{0, \ldots, d-1\}$. Stacking these scalars gives the $i$-th coordinate of the column $E_{:, v}$, hence $E e_v = E_{:, v}$. $\blacksquare$

So embedding-table lookup is a linear map $\mathbb{R}^V \to \mathbb{R}^d$ in disguise. Implementations skip the matmul and slice the column; the *meaning* is the matrix product.

**Theorem 2 (Weight-tying gradient identity).** *Let $E \in \mathbb{R}^{d \times V}$ and consider a model $f(v, E) = U h(E e_v) = E^\top h(E e_v)$ with tied $U = E^\top$, where $h : \mathbb{R}^d \to \mathbb{R}^d$ is some sub-network with no further dependence on $E$. Let $\mathcal{L}$ be a scalar loss applied to the logits $z = E^\top h \in \mathbb{R}^V$. Then*
$$
\nabla_E \mathcal{L} \;=\; \underbrace{h\, (\nabla_z \mathcal{L})^\top}_{\text{output-side}} \;+\; \underbrace{\big(J_h^\top E\, \nabla_z \mathcal{L}\big)\, e_v^\top}_{\text{input-side}},
$$
*where $J_h \in \mathbb{R}^{d \times d}$ is the Jacobian of $h$ at the point $E e_v$. The input-side contribution is a rank-one matrix whose only nonzero column is column $v$.*

*Proof.* Treat $E$ as occurring in two distinct roles: an output role $U = E^\top$ and an input role inside $h \circ (E \cdot e_v)$. By the multivariate chain rule (Chapter 18), the total gradient is the sum of the partial gradients with respect to each role.

*Output role.* The logit is $z = U h$ with $U = E^\top$, so $z_k = \sum_i E_{ik} h_i$ and $\partial z_k / \partial E_{ij} = \mathbf{1}_{k = j} h_i$. Hence
$$\frac{\partial \mathcal{L}}{\partial E_{ij}} \bigg|_{\text{out}} = \sum_k \frac{\partial \mathcal{L}}{\partial z_k} \frac{\partial z_k}{\partial E_{ij}} = h_i\, \frac{\partial \mathcal{L}}{\partial z_j},$$
so the output-side contribution is the outer product $h\, (\nabla_z \mathcal{L})^\top \in \mathbb{R}^{d \times V}$.

*Input role.* The hidden state is $h = h(x)$ with $x = E e_v$, so $x_i = E_{iv}$ and $\partial x_i / \partial E_{ab} = \mathbf{1}_{a = i}\mathbf{1}_{b = v}$. By the chain rule,
$$\frac{\partial \mathcal{L}}{\partial E_{ab}} \bigg|_{\text{in}} = \sum_i \frac{\partial \mathcal{L}}{\partial x_i} \frac{\partial x_i}{\partial E_{ab}} = \frac{\partial \mathcal{L}}{\partial x_a}\, \mathbf{1}_{b = v}.$$
The vector $\nabla_x \mathcal{L} = J_h^\top \nabla_h \mathcal{L} = J_h^\top (E\, \nabla_z \mathcal{L})$ since $\nabla_h \mathcal{L} = U^\top \nabla_z \mathcal{L} = E\, \nabla_z \mathcal{L}$. Hence the input-side contribution is the rank-one matrix $g\, e_v^\top$ with $g := J_h^\top E\, \nabla_z \mathcal{L} \in \mathbb{R}^d$, whose only nonzero column is $g$ in position $v$. Adding the two roles gives the claim. $\blacksquare$

**Remark (Sparsity of the input gradient).** The factor $e_v^\top$ kills every column except column $v$. This is why embedding tables are conventionally updated via a *sparse* gradient: each minibatch only touches the columns of the tokens it actually contains. The output-side term, by contrast, hits *all* $V$ columns through the dense outer product $h (\nabla_z \mathcal{L})^\top$.

**Heuristic (Distributional semantics; Mikolov et al. 2013).** During training, two tokens $u, v$ that appear in similar surrounding contexts produce similar gradients on their embeddings (the "context" enters through $\nabla_z \mathcal{L}$ and through $h$). Iterating, $E_{:, u}$ and $E_{:, v}$ drift toward similar locations in $\mathbb{R}^d$, so cosine similarity of embeddings tracks distributional similarity in the corpus. This is an empirical claim, not a theorem; we illustrate it in code.

**Theorem 3 (Dimensionality bound).** *If $V > d$ then no choice of $E \in \mathbb{R}^{d \times V}$ makes the columns $E_{:, 0}, \ldots, E_{:, V-1}$ pairwise orthogonal and nonzero.*

*Proof.* Pairwise orthogonal nonzero vectors in any inner-product space are linearly independent: from $\sum_v c_v E_{:, v} = 0$, taking the inner product with $E_{:, u}$ yields $c_u \|E_{:, u}\|^2 = 0$, so $c_u = 0$. Hence the columns would form a linearly independent set of size $V$ in $\mathbb{R}^d$. By Chapter 5 (invariance of dimension), $\dim \mathbb{R}^d = d$, so any independent set has size $\leq d$. Therefore $V \leq d$, contradicting $V > d$. $\blacksquare$

In LLMs we always have $V \gg d$ (e.g. $V \approx 5\cdot 10^4$ vs.\ $d \approx 4096$), so the embeddings cannot be mutually orthogonal — they live as an *overcomplete* set, and "near-orthogonality" is the best one can ask.

## Code sketch

We build a tiny embedding matrix and verify $E e_v = E_{:, v}$ exactly. We then construct a tied unembedding $U = E^\top$ and check that the logit for token $v$ equals $\langle E_{:, v}, h \rangle$. Next, a Mikolov-style skip-gram on a synthetic corpus shows that embeddings of co-occurring tokens develop higher cosine similarity than non-co-occurring ones after a few hundred SGD steps. Finally, we verify Theorem 3 by attempting to orthogonalize $V = 8$ vectors in $\mathbb{R}^3$ via Gram–Schmidt and observing dimension exhaustion.

## Connection to LLMs

Every transformer language model contains exactly the data structure of this chapter: an input embedding $E \in \mathbb{R}^{d \times V}$ and an output projection $U \in \mathbb{R}^{V \times d}$. The embedding dimension $d$ is the *model dimension* $d_{\mathrm{model}}$, which is what dominates parameter counts and FLOPs (Chapter 23–25). GPT-2 (Radford et al. 2019), Llama 1/2/3 (Touvron et al. 2023), and most open-weight LMs use weight tying $U = E^\top$, halving the embedding parameter count and empirically improving perplexity (Press & Wolf 2016; Inan et al. 2017). GPT-3/4 use an LM-head that is initialized from $E^\top$ but may be trained separately. Once a token is embedded, position information is added (Chapter 20), and the residual stream begins (Chapter 22). Every operation downstream — including the final logit projection $U h_{\text{final}}$ — is the same matrix $E$ that began the forward pass, viewed from a different side.
