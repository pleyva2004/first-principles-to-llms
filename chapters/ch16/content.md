## Motivation

A multi-layer perceptron (Chapter 15) without a nonlinear activation collapses into a single affine map: $W_2(W_1 x + b_1) + b_2 = (W_2 W_1) x + (W_2 b_1 + b_2)$. Universal approximation requires injecting a *pointwise* nonlinearity $\phi$ between affine layers. The choice of $\phi$ controls (i) the gradient signal that backpropagation (Chapter 3) is allowed to push through the network, (ii) the representational geometry of hidden activations, and (iii) numerical conditioning of the loss. This chapter derives the five activations that dominate modern deep learning: sigmoid, tanh, ReLU, GELU, and softmax. For each we compute the derivative or Jacobian from first principles, analyze saturation, and connect to the LLM stack (Chapters 21, 25).

## Definitions

**Sigmoid.** $\sigma:\mathbb{R}\to(0,1)$, $\sigma(x) = \frac{1}{1+e^{-x}}$.

**Hyperbolic tangent.** $\tanh:\mathbb{R}\to(-1,1)$, $\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} = 2\sigma(2x) - 1$.

**ReLU.** $\mathrm{ReLU}(x) = \max(0, x)$. Differentiable everywhere except $x=0$; the *Clarke subdifferential* at $0$ is $[0,1]$.

**GELU.** $\mathrm{GELU}(x) = x \Phi(x)$, where $\Phi(x) = \tfrac{1}{2}\bigl(1 + \mathrm{erf}(x/\sqrt 2)\bigr)$ is the standard-normal CDF. The Hendrycks–Gimpel tanh-approximation reads
$$\mathrm{GELU}(x) \approx \tfrac{1}{2}x\bigl(1 + \tanh(\sqrt{2/\pi}\,(x + 0.044715\,x^3))\bigr).$$

**Softmax.** $\mathrm{softmax}:\mathbb{R}^n \to \Delta^{n-1}$, $\mathrm{softmax}(z)_i = \frac{e^{z_i}}{\sum_{k=1}^n e^{z_k}}$, mapping logits to a probability simplex.

## Theorems and Proofs

**Theorem 16.1 (sigmoid derivative).** $\sigma'(x) = \sigma(x)(1-\sigma(x))$.

*Proof.* Write $\sigma(x) = (1+e^{-x})^{-1}$. By the chain rule, $\sigma'(x) = -(1+e^{-x})^{-2} \cdot (-e^{-x}) = \frac{e^{-x}}{(1+e^{-x})^2}$. Factor: $\frac{e^{-x}}{(1+e^{-x})^2} = \frac{1}{1+e^{-x}}\cdot \frac{e^{-x}}{1+e^{-x}} = \sigma(x)\bigl(1 - \sigma(x)\bigr)$, since $\frac{e^{-x}}{1+e^{-x}} = 1 - \sigma(x)$. $\square$

**Theorem 16.2 (tanh derivative).** $\tanh'(x) = 1 - \tanh^2(x)$.

*Proof.* From $\tanh = 2\sigma(2x) - 1$, the chain rule gives $\tanh'(x) = 4\sigma(2x)(1-\sigma(2x))$. Substitute $\sigma(2x) = (\tanh(x)+1)/2$: $4\cdot \tfrac{\tanh(x)+1}{2}\cdot \tfrac{1-\tanh(x)}{2} = (1+\tanh x)(1-\tanh x) = 1 - \tanh^2(x)$. $\square$

**Theorem 16.3 (ReLU derivative).** For $x\neq 0$, $\mathrm{ReLU}'(x) = \mathbf{1}_{x>0}$. At $x=0$, the Clarke subdifferential is $\partial\,\mathrm{ReLU}(0) = [0,1]$.

*Proof.* For $x>0$, $\mathrm{ReLU}(x)=x$ has derivative $1$. For $x<0$, derivative $0$. At $x=0$ the left derivative is $0$ and the right derivative is $1$; the convex hull of these limits, $[0,1]$, defines the subdifferential. Implementations conventionally pick $0$. $\square$

**Theorem 16.4 (GELU derivative).** $\mathrm{GELU}'(x) = \Phi(x) + x\phi(x)$, where $\phi(x)=\tfrac{1}{\sqrt{2\pi}}e^{-x^2/2}$.

*Proof.* Apply the product rule to $x\Phi(x)$: $\frac{d}{dx}[x\Phi(x)] = \Phi(x) + x\Phi'(x)$. By the fundamental theorem of calculus, $\Phi'(x) = \phi(x)$. $\square$

**Theorem 16.5 (softmax Jacobian).** Let $s = \mathrm{softmax}(z)$. Then $\frac{\partial s_i}{\partial z_j} = s_i(\delta_{ij} - s_j)$.

*Proof.* Let $S = \sum_k e^{z_k}$, so $s_i = e^{z_i}/S$. By the quotient rule,
$$\frac{\partial s_i}{\partial z_j} = \frac{(\partial e^{z_i}/\partial z_j)\,S - e^{z_i}\,(\partial S/\partial z_j)}{S^2} = \frac{\delta_{ij} e^{z_i} S - e^{z_i} e^{z_j}}{S^2} = \delta_{ij} s_i - s_i s_j = s_i(\delta_{ij} - s_j).\ \square$$

In matrix form, $J = \mathrm{diag}(s) - s s^\top$, a rank-$\le n$ symmetric PSD matrix with kernel spanned by $\mathbf{1}$ (consistent with softmax's translation invariance).

**Proposition 16.6 (saturation).** $\sigma'(x), \tanh'(x) \to 0$ as $|x|\to\infty$. Hence in deep MLPs, gradients $\prod_\ell \phi'(z_\ell)$ shrink geometrically — *vanishing gradients*. ReLU avoids saturation on the positive ray but suffers *dead neurons*: if a unit's pre-activation stays $\le 0$ across the whole training set, its gradient is identically zero. GELU is smooth and asymptotes to identity for $x \gg 0$ (since $\Phi(x)\to 1$ and $\phi(x)\to 0$, so $\mathrm{GELU}'(x)\to 1$) and to $0$ for $x\ll 0$ — combining ReLU-like sparsity with smoothness.

**Theorem 16.7 (softmax temperature limits).** Let $T>0$ and assume the maximizer $i^* = \arg\max_i z_i$ is unique. Then
$$\lim_{T\to 0^+} \mathrm{softmax}(z/T)_i = \mathbf{1}_{i = i^*},\qquad \lim_{T\to\infty} \mathrm{softmax}(z/T)_i = \tfrac{1}{n}.$$

*Proof.* Write $\mathrm{softmax}(z/T)_i = \frac{e^{(z_i - z_{i^*})/T}}{\sum_k e^{(z_k - z_{i^*})/T}}$. As $T\to 0^+$, every term $e^{(z_k - z_{i^*})/T}$ with $k\neq i^*$ tends to $0$ while the $k=i^*$ term equals $1$, giving $\mathbf{1}_{i=i^*}$. As $T\to\infty$, every $e^{(z_k - z_{i^*})/T}\to 1$, so the ratio tends to $1/n$. $\square$

**Proposition 16.8 (numerical stability).** $\mathrm{softmax}(z) = \mathrm{softmax}(z - c\mathbf{1})$ for any $c\in\mathbb{R}$. Choosing $c = \max_i z_i$ ensures every exponent is $\le 0$, so no overflow.

*Proof.* $\frac{e^{z_i - c}}{\sum_k e^{z_k - c}} = \frac{e^{-c} e^{z_i}}{e^{-c}\sum_k e^{z_k}} = \mathrm{softmax}(z)_i$. $\square$

**Corollary 16.9 (log-sum-exp).** $\log\sum_j e^{z_j} = \max_j z_j + \log\sum_j e^{z_j - \max_k z_k}$, also overflow-free.

## Code Sketch

The accompanying notebook (i) plots all five activations on $[-4,4]$ and tabulates them at $x\in\{-2,-1,0,1,2\}$; (ii) verifies analytic derivatives against centered finite differences; (iii) implements `softmax_jacobian(z) = diag(s) - np.outer(s, s)` and matches a numerical Jacobian to machine precision; (iv) sweeps temperature $T\in\{0.1, 1, 5, 100\}$ to visualize the one-hot/uniform limits of Theorem 16.7; (v) demonstrates that naive softmax of $(1000,1001,1002)$ overflows but the stabilized version recovers $(0.0900, 0.2447, 0.6652)$.

## Connection to LLMs

GELU is the default feedforward activation in GPT-2/3, BERT, and pre-Llama Transformers (Llama-family models switched to SwiGLU, a gated variant; see Chapter 17). Wherever Chapter 15's MLP block appears between attention layers, the inner nonlinearity is GELU acting elementwise on a $4d_{\text{model}}$-wide hidden state. Softmax appears in **two** distinct loci of an LLM:

1. **Attention** (Chapter 21): for query $q$ and keys $K$, attention weights are $\mathrm{softmax}(qK^\top/\sqrt{d_k})$ — a row-wise softmax converting compatibility scores into a probability distribution over keys.
2. **Output head** (Chapter 25): the final logits over the vocabulary are passed through softmax to produce the next-token distribution $p(x_{t+1}\mid x_{\le t})$. Sampling temperature (Theorem 16.7) is the standard knob exposed to users; $T\to 0$ recovers greedy decoding, $T\to\infty$ uniform random sampling.

Both invocations use the stabilized form of Proposition 16.8 in production kernels (FlashAttention, fused log-softmax + cross-entropy), making the seemingly trivial "subtract the max" identity load-bearing for trillion-parameter training.
