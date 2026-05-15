## Motivation

Chapter 21 built a single attention head: a $T \times T$ soft-lookup that links every output position to every input position in $O(1)$ depth. A single head, however, must compress *every* relational pattern (syntactic, semantic, positional) into one softmax-weighted average over $V$. The Vaswani et al. (2017) fix is **multi-head attention** (MHA): run $H$ attention heads in parallel, each on its own low-dimensional subspace, then concatenate and re-project. The total parameter count stays $\Theta(d_{\mathrm{model}}^2)$ — the same as a single full-rank attention — yet the model gains $H$ independent "channels" of attention.

This chapter proves three things. First, MHA is *exactly* a single attention with block-diagonal projections. Second, the per-head split keeps the parameter count constant in $H$. Third, the compute scales as $O(T d^2 + T^2 d)$, with the $T^2 d$ term dominating long-context inference; this motivates the inference-time variants **MQA** and **GQA**, used in PaLM and Llama 2/3 respectively.

## Definitions

Throughout, $X \in \mathbb{R}^{T \times d}$ where $d = d_{\mathrm{model}}$. Fix a head count $H$ dividing $d$, set $d_k = d_v = d/H$, and write $\mathrm{Attn}(\cdot,\cdot,\cdot)$ for scaled dot-product attention (Chapter 21).

**Definition 1 (Multi-head attention, MHA).** For $h = 1, \dots, H$, fix
$$
W_Q^{(h)}, W_K^{(h)} \in \mathbb{R}^{d \times d_k}, \qquad W_V^{(h)} \in \mathbb{R}^{d \times d_v}, \qquad W_O \in \mathbb{R}^{H d_v \times d}.
$$
Define
$$
\mathrm{head}_h \;=\; \mathrm{Attn}\!\bigl(X W_Q^{(h)},\, X W_K^{(h)},\, X W_V^{(h)}\bigr) \;\in\; \mathbb{R}^{T \times d_v},
$$
$$
\mathrm{MHA}(X) \;=\; \mathrm{Concat}(\mathrm{head}_1, \dots, \mathrm{head}_H)\, W_O \;\in\; \mathbb{R}^{T \times d}.
$$

**Definition 2 (Multi-query attention, MQA).** All $H$ heads share a single $W_K, W_V$; only $W_Q^{(h)}$ varies per head. Hence the cached $K, V$ tensors are $H$ times smaller.

**Definition 3 (Grouped-query attention, GQA).** Partition the $H$ heads into $G$ groups, $G \mid H$. Heads within a group share $W_K, W_V$. MHA = $G = H$, MQA = $G = 1$. Llama 2/3 use $G \in \{4, 8\}$.

## Theorems

**Theorem 4 (MHA = block-structured single attention).** *Let $W_Q^{\mathrm{blk}} \in \mathbb{R}^{d \times H d_k}$ be the horizontal concatenation of the per-head $W_Q^{(h)}$, and similarly $W_K^{\mathrm{blk}}, W_V^{\mathrm{blk}}$. Set*
$$
Q^{\mathrm{blk}} = X W_Q^{\mathrm{blk}}, \quad K^{\mathrm{blk}} = X W_K^{\mathrm{blk}}, \quad V^{\mathrm{blk}} = X W_V^{\mathrm{blk}} \;\in\; \mathbb{R}^{T \times H d_k}.
$$
*Reshape each into $\mathbb{R}^{T \times H \times d_k}$ by splitting the last axis into $H$ blocks of width $d_k$. Then the per-head attention applied to slice $h$ of these reshaped tensors equals $\mathrm{head}_h$ exactly. Equivalently, MHA is a single linear projection followed by a per-head attention loop, with no information mixing across heads until $W_O$.*

*Proof.* The horizontal concatenation $W_Q^{\mathrm{blk}} = [W_Q^{(1)} \mid \cdots \mid W_Q^{(H)}]$ satisfies $X W_Q^{\mathrm{blk}} = [X W_Q^{(1)} \mid \cdots \mid X W_Q^{(H)}]$, so the $h$-th block of width $d_k$ is exactly $X W_Q^{(h)} = Q^{(h)}$. The same holds for $K$ and $V$. Reshape is a no-op on memory layout: slice $h$ of $Q^{\mathrm{blk}} \in \mathbb{R}^{T \times H \times d_k}$ is $Q^{(h)}$. Since $\mathrm{Attn}$ is computed independently per head and uses only that head's $Q^{(h)}, K^{(h)}, V^{(h)}$, the outputs match. $\square$

This is a *block-diagonal* picture in the following sense: if we wrote MHA as one giant attention with a $T \times T$ score matrix, that score matrix is the blockwise sum $\sum_h Q^{(h)} K^{(h)\top}/\sqrt{d_k}$ only after softmax-per-head, *not* a literal block-diagonal QKV — heads do not communicate inside the softmax. This is the key inductive bias: each head selects independently, and only $W_O$ blends the results.

**Theorem 5 (Parameter count is independent of $H$).** *Total parameters in $W_Q^{(\cdot)}, W_K^{(\cdot)}, W_V^{(\cdot)}, W_O$ equal $4 d^2$.*

*Proof.* Each per-head projection is $d \times d_k = d \times d/H$, with $d^2/H$ parameters. There are $H$ heads and three projections, giving $3 H \cdot d^2/H = 3 d^2$. The output matrix $W_O$ is $H d_v \times d = d \times d$, contributing $d^2$. Total: $4 d^2$. The $1/H$ shrinkage of each head exactly cancels the $H$-fold replication. $\square$

**Theorem 6 (Compute complexity of MHA).** *For input $X \in \mathbb{R}^{T \times d}$,*
$$
\mathrm{cost}(\mathrm{MHA}) \;=\; \Theta(T d^2 + T^2 d).
$$

*Proof.* We sum the FLOPs of each step.

1. *Q, K, V projections.* Each is a $T \times d$ times $d \times d$ matmul (after concatenating the $H$ blocks, by Theorem 4): $\Theta(T d^2)$ per projection, $\Theta(T d^2)$ in total over the three.
2. *Score matrix per head.* $Q^{(h)} K^{(h)\top}$ is $(T \times d_k)(d_k \times T) = \Theta(T^2 d_k)$. Summed over $H$ heads: $H \cdot \Theta(T^2 d_k) = \Theta(T^2 d)$ since $H d_k = d$.
3. *Softmax.* $\Theta(T^2)$ per head, $\Theta(H T^2)$ total — subdominant.
4. *Attention output $A V$.* Per head $(T \times T)(T \times d_v) = \Theta(T^2 d_v)$, summed to $\Theta(T^2 d)$.
5. *Output projection $W_O$.* $T \times (H d_v) \cdot (H d_v \times d) = \Theta(T d^2)$.

Adding: $\Theta(T d^2) + \Theta(T^2 d) + \Theta(H T^2) + \Theta(T^2 d) + \Theta(T d^2) = \Theta(T d^2 + T^2 d)$. $\square$

**Corollary 7 (Long-context regime).** *When $T \gg d$, the $T^2 d$ term dominates. Doubling the sequence quadruples the attention FLOPs but only doubles projection FLOPs.*

This is the structural reason transformer inference is bandwidth-bound on long contexts and motivates Chapter 27's efficient-attention literature.

**Theorem 8 (KV-cache memory).** *Autoregressive decoding caches $K, V$ for all past tokens. Per layer, MHA stores $2 T H d_k = 2 T d$ scalars; GQA with $G$ groups stores $2 T G d_k = 2 T d \cdot G/H$; MQA stores $2 T d_k = 2 T d / H$.*

*Proof.* Each head needs its own $K \in \mathbb{R}^{T \times d_k}$, $V \in \mathbb{R}^{T \times d_v}$, totalling $2 T d_k$ scalars. MHA replicates this $H$ times (one per head). GQA replicates $G$ times (one per group), MQA once. Multiply through. $\square$

A 70 B-parameter model with $d = 8192, H = 64$ at $T = 8192$ stores $2 \cdot 8192 \cdot 8192 = 134$ M scalars per layer for MHA, but only $134/8 \approx 17$ M for GQA with $G = 8$ — an 8$\times$ inference-memory reduction at minimal quality cost.

## Connection to LLMs

GPT-2 and GPT-3 use vanilla MHA. PaLM (2022) was the first major model to deploy MQA, motivated by inference throughput; Shazeer (2019) had earlier proposed it. Llama 2 (2023) introduced GQA as a quality/memory compromise; Llama 3 retains it. Mistral and most open-weights successors follow suit. The picture is now standard: training uses MHA-shaped compute, deployment uses GQA to fit the KV cache in HBM. Chapter 23 stacks these blocks into a transformer layer; Chapter 27 surveys the efficient-attention zoo (FlashAttention, sliding-window, linear attention) that targets the $T^2 d$ term proved above.
