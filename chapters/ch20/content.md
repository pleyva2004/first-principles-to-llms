## Motivation

So far we have built feedforward networks (Ch.\ 17--19): a fixed-depth pipeline of linear maps and nonlinearities. But language, audio, and code are *sequences*: the output at position $t$ depends on inputs $x_1, \dots, x_t$, and the sequence length $T$ varies. We need a model that consumes one token at a time, accumulates context, and emits one prediction per step. The Recurrent Neural Network (RNN) was the dominant answer from the late 1980s through 2017. In this chapter we derive its forward dynamics, prove the **vanishing/exploding gradient theorem** (Pascanu, Mikolov, Bengio 2013) using the SVD machinery of Chapter 6 and the backprop chain rule of Chapter 18, and show that this failure mode--together with a parallel *information bottleneck*--is precisely what attention (Ch.\ 21--24) was invented to fix.

## Definitions

**Sequence model.** A map $f : \mathcal{X}^T \to \mathcal{Y}^T$ that consumes $(x_1, \dots, x_T)$ and emits $(y_1, \dots, y_T)$, with the *causality* constraint $y_t = f_t(x_1, \dots, x_t)$.

**RNN cell.** Fix hidden dimension $d$ and parameters $W_h \in \mathbb{R}^{d \times d}$, $W_x \in \mathbb{R}^{d \times \dim x}$, $b \in \mathbb{R}^d$, and a pointwise nonlinearity $\sigma$ (typically $\tanh$). The RNN cell is the recursion
$$h_t = \sigma(W_h h_{t-1} + W_x x_t + b), \qquad h_0 = 0.$$
The output is $y_t = W_y h_t$.

**Backpropagation through time (BPTT).** Unroll the recursion into an acyclic computation graph of depth $T$, sharing the weights $(W_h, W_x, b)$ across every time step, then apply the backprop algorithm of Chapter 18 to that graph. The shared-weight gradient is the sum of the per-step gradients.

## Theorem 1 (Vanishing/Exploding Gradient, Pascanu et al.\ 2013)

*Linearized statement.* Drop the nonlinearity and biases: $h_t = W h_{t-1}$ with $W \in \mathbb{R}^{d \times d}$. Let $L$ be a scalar loss depending on $h_T$. Then for any $t < T$,
$$\frac{\partial L}{\partial h_t} = (W^\top)^{T-t} \frac{\partial L}{\partial h_T}.$$
Consequently, with $\sigma_{\max}(W)$ the largest singular value,
$$\Big\| \frac{\partial L}{\partial h_t} \Big\| \leq \sigma_{\max}(W)^{T-t} \Big\| \frac{\partial L}{\partial h_T} \Big\|,$$
with equality achievable. If $\sigma_{\max}(W) < 1$ the gradient *vanishes* exponentially in $T-t$; if $\sigma_{\max}(W) > 1$ it *explodes*.

**Proof.** By the multivariable chain rule (Ch.\ 18), $\partial h_t / \partial h_{t-1} = W$, so $\partial L / \partial h_{t-1} = W^\top \partial L / \partial h_t$. Iterating $T-t$ times gives the claimed product. By Chapter 6, $W = U \Sigma V^\top$, so $(W^\top)^{T-t} = V \Sigma^{T-t} U^\top$ (using $W^\top = V \Sigma U^\top$ and unitarity), and the operator-norm bound is just $\|\Sigma^{T-t}\|_2 = \sigma_{\max}(W)^{T-t}$. The bound is attained by aligning $\partial L / \partial h_T$ with the top left singular vector of $W^{T-t}$. $\blacksquare$

**Nonlinear extension.** With $\sigma$ such that $|\sigma'| \leq c$ pointwise (true for $\tanh$ with $c = 1$), the Jacobian of one step is $D_t W$ where $D_t = \mathrm{diag}(\sigma'(\cdot))$ has $\|D_t\|_2 \leq c$. Submultiplicativity of the operator norm gives
$$\Big\| \frac{\partial L}{\partial h_t} \Big\| \leq c^{T-t} \|W\|_2^{T-t} \Big\| \frac{\partial L}{\partial h_T} \Big\|.$$
For $\tanh$, $c = 1$ and saturated regions push $\|D_t\|$ much below $1$, *worsening* vanishing.

## Theorem 2 (Information Bottleneck of an RNN)

The hidden state $h_t \in \mathbb{R}^d$ is a deterministic function of $(x_1, \dots, x_t)$, so it carries at most $d$ real-valued degrees of freedom. By a counting / rate--distortion argument, if the inputs come from an alphabet of size $V$ and we want to losslessly recover $(x_1, \dots, x_t)$ from $h_t$, we need $d \geq \lceil t \log_2 V / B \rceil$ bits, where $B$ is the per-coordinate precision. Once $t \log_2 V \gg d B$, *some* information must be discarded; the per-token signal-to-noise of past tokens decays as $1/t$. *Sketch:* this is the data-processing inequality applied to the deterministic Markov chain $(x_1, \dots, x_t) \to h_t \to \hat x_s$ for $s \ll t$.

## Theorem 3 (Attention Dissolves Both Bottlenecks --- Sketch)

Let $h_T = \sum_{s=1}^T \alpha_{Ts} V_s$, where $V_s = V x_s$ are value vectors and $\alpha_{Ts}$ are scalar weights (in Ch.\ 21 we make $\alpha$ a learned softmax of query--key dot products). Then for *any* $s$,
$$\frac{\partial h_T}{\partial V_s} = \alpha_{Ts} I_d, \qquad \frac{\partial L}{\partial V_s} = \alpha_{Ts} \frac{\partial L}{\partial h_T}.$$
This is a *single* matrix-product step, independent of $T - s$. There is no exponentiation of any spectral quantity, so no vanishing/exploding regime in the depth direction. Furthermore, every past token has its own slot $V_s$ in working memory, so the bottleneck of Theorem 2 is replaced by an $O(T \cdot d)$-sized addressable cache. The full derivation, including queries, keys, and softmax, is Chapter 21.

## Connection to LLMs

The vanishing-gradient theorem is *the* historical reason transformer-based LLMs displaced RNNs. Bahdanau, Cho, and Bengio (2015) added attention to a seq2seq RNN to fix translation of long sentences; Vaswani et al.\ (2017, "Attention Is All You Need") removed recurrence entirely. Without an $O(1)$ gradient path between distant positions, scaling to $10^4$--$10^6$-token contexts is hopeless: a $\sigma_{\max} = 0.99$ RNN attenuates by $e^{-100}$ over $10\,000$ steps. We make this rigorous in Ch.\ 21 (single-head attention), generalize in Ch.\ 22 (multi-head), embed in the full Transformer block in Ch.\ 23, and scale in Ch.\ 24.
