## Motivation

Chapter 17 ended with a remarkably clean fact: for softmax + cross-entropy, the gradient of the loss with respect to the pre-softmax logits is just $\hat p - y$. That formula is exact and pleasant — but it tells us only the *output-layer* gradient. To train an actual network we need the gradient of the loss with respect to *every* parameter, including those buried under tens of intermediate nonlinearities. The naive approach — symbolic differentiation — produces expressions whose size explodes exponentially with depth. The naive numerical approach — finite differences over $P$ parameters — costs $O(P)$ forward passes, prohibitive when $P \approx 10^{11}$.

Backpropagation solves both problems at once. It is not a new mathematical idea: it is the multivariate chain rule (Chapter 4) applied along a directed acyclic graph in a particularly clever order. We make this precise.

## Computational graphs and AD

\textbf{Definition (Computational graph).} A *computational graph* is a DAG whose nodes are intermediate values $v_i \in \mathbb{R}^{d_i}$ and whose edges are elementary operations. Source nodes are inputs; sink nodes are outputs. A *forward pass* computes all $v_i$ in topological order given the inputs.

\textbf{Definition (JVP / forward-mode AD).} Given a tangent vector $\dot x$, forward-mode AD propagates $\dot v_i = \sum_{j \in \text{parents}(i)} \frac{\partial v_i}{\partial v_j} \dot v_j$ in topological order. One sweep computes $J \dot x$ for the full Jacobian $J = \partial f/\partial x$. Cost: one pass per *input* dimension to recover the full Jacobian.

\textbf{Definition (VJP / reverse-mode AD).} Given a cotangent $\bar y$ at the output, reverse-mode AD propagates *adjoints* $\bar v_i$ in *reverse* topological order:
$$
\bar v_j = \sum_{i \in \text{children}(j)} \left(\frac{\partial v_i}{\partial v_j}\right)^T \bar v_i.
$$
One sweep computes $J^T \bar y$. Cost: one pass per *output* dimension. When the output is a scalar loss $L$, a single backward pass yields $\nabla_x L$. The adjoint $\bar v_i := \partial L / \partial v_i$ is the central object of backprop.

## The backprop theorem

\textbf{Theorem (Backprop = reverse-mode AD on a feedforward net).} Let $L(x) = \ell(h^{(L)})$ where $h^{(\ell)} = f^{(\ell)}(h^{(\ell-1)})$, $h^{(0)} = x$, and each $f^{(\ell)}$ is differentiable with Jacobian $J^{(\ell)} := \partial f^{(\ell)} / \partial h^{(\ell-1)}$. Define adjoints $\bar h^{(\ell)} := \partial L / \partial h^{(\ell)}$ as row vectors. Then
$$
\bar h^{(L)} = \nabla \ell(h^{(L)}), \qquad \bar h^{(\ell-1)} = (J^{(\ell)})^T \bar h^{(\ell)}.
$$

\textbf{Proof.} By the multivariate chain rule (Ch. 4), for any $\ell$,
$\partial L / \partial h^{(\ell-1)} = (\partial L / \partial h^{(\ell)})(\partial h^{(\ell)} / \partial h^{(\ell-1)}) = \bar h^{(\ell)} J^{(\ell)}$,
which transposed gives the adjoint recurrence. The base case is the gradient of $\ell$ at $h^{(L)}$. Induction over $\ell = L, L-1, \ldots, 1$. $\square$

For Ch. 17's softmax + cross-entropy block, $\bar h^{(L)} = \hat p - y$ is the base case and the recurrence does the rest.

## VJPs for an MLP layer

\textbf{Proposition.} For a layer $z^{(\ell)} = W^{(\ell)} h^{(\ell-1)} + b^{(\ell)}$, $h^{(\ell)} = \sigma(z^{(\ell)})$ with elementwise $\sigma$, the VJPs are
$$
\bar z^{(\ell)} = \bar h^{(\ell)} \odot \sigma'(z^{(\ell)}), \quad \bar W^{(\ell)} = \bar z^{(\ell)} (h^{(\ell-1)})^T, \quad \bar b^{(\ell)} = \bar z^{(\ell)}, \quad \bar h^{(\ell-1)} = (W^{(\ell)})^T \bar z^{(\ell)}.
$$

\textbf{Derivation.} The chain rule gives $\bar z^{(\ell)}_i = \sum_k \bar h^{(\ell)}_k \, \partial h^{(\ell)}_k / \partial z^{(\ell)}_i = \bar h^{(\ell)}_i \sigma'(z^{(\ell)}_i)$ since $\sigma$ is elementwise. For $W$, since $z_i = \sum_j W_{ij} h^{(\ell-1)}_j$, $\partial z_i / \partial W_{kj} = \delta_{ik} h^{(\ell-1)}_j$, so $\bar W_{kj} = \bar z_k h^{(\ell-1)}_j$, i.e. an outer product. The bias is $\partial z_i / \partial b_j = \delta_{ij}$. For $h^{(\ell-1)}$, $\partial z_i / \partial h^{(\ell-1)}_j = W_{ij}$, giving $\bar h^{(\ell-1)} = W^T \bar z^{(\ell)}$. $\square$

## Cost and memory

\textbf{Theorem (Baur–Strassen).} For a function defined by $K$ elementary operations with bounded fan-out, the gradient can be computed by an algorithm whose arithmetic cost is at most $\sim 5 K$ — i.e., the backward pass costs only a small constant times the forward pass, *independent of the number of parameters*.

This is the magic. We pay a constant overhead, not a $P$-fold one, to differentiate with respect to all $P$ parameters simultaneously. Forward-mode AD does the opposite: $O(n)$ for $n$ inputs, $O(1)$ in outputs.

\textbf{Memory.} The recurrence $\bar h^{(\ell-1)} = (J^{(\ell)})^T \bar h^{(\ell)}$ requires evaluating $J^{(\ell)}$, which usually depends on $h^{(\ell-1)}$ (e.g. $\sigma'(z^{(\ell)})$). Hence reverse-mode AD must *cache* every intermediate activation produced during the forward pass — memory cost is $O(L)$ in depth times batch and width. This is exactly the constraint that motivates *gradient checkpointing* (Ch. 27): drop activations and recompute them on the backward pass, trading compute for memory.

## Connection to LLMs

A modern transformer has $L \in [12, 96]$ blocks, each consisting of attention + MLP sub-layers. Training one step is: (1) forward pass through all $L$ blocks, caching activations; (2) reverse-mode AD computing $\nabla_\theta L$ for every parameter (often $\theta \in \mathbb{R}^{10^{11}}$) in time $\le 5\times$ the forward pass — Baur–Strassen in action; (3) optimizer step. The per-token activation memory is the dominant constraint at scale and is precisely why frameworks ship gradient checkpointing, FlashAttention recomputation (Ch. 23), and ZeRO-style sharding by default. Every parameter update in a 175B model is the recurrence above, executed billions of times.
