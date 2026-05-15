## Motivation

We have built a transformer block that mixes information across positions (Ch 23) and a positional encoding that injects order (Ch 24). To turn this into a *language model*, we must (a) decide what the network outputs, (b) decide what loss to minimize, and (c) reconcile the desire for **parallel** training with the requirement that the model must not peek at future tokens. All three are settled by one design choice: **causal masking + next-token prediction**.

The pre-training recipe behind GPT-1/2/3/4, Llama, Claude and Gemini is exactly this chapter — fit a categorical distribution over the next token by maximum likelihood on a corpus, with a triangular attention mask that makes the per-position losses backprop-independent.

## Definitions

**Definition (Causal language model).** A *causal* (decoder-only) language model with parameters $\theta$ assigns to a sequence $x_{1:T} = (x_1,\ldots,x_T)$ over vocabulary $\mathcal{V}$ the probability
$$ p_\theta(x_{1:T}) = \prod_{t=1}^{T} p_\theta(x_t \mid x_{<t}), \qquad x_{<t} := (x_1,\ldots,x_{t-1}). $$
Each conditional $p_\theta(\,\cdot\,\mid x_{<t})$ is a softmax over $\mathcal{V}$ produced by a transformer applied to the prefix $x_{<t}$.

**Definition (Causal attention mask).** Let $T \in \mathbb{N}$. The causal mask is the matrix $M \in \{0,-\infty\}^{T\times T}$ with
$$ M_{ij} = \begin{cases} 0 & j \le i,\\ -\infty & j > i.\end{cases} $$
Causal scaled dot-product attention replaces the score matrix $S = QK^\top/\sqrt{d_k}$ by $S + M$ before the row-wise softmax. Because $\exp(-\infty) = 0$, row $i$ of the softmax has zero mass on every column $j > i$.

**Definition (Teacher forcing).** During training, the input to predict $x_t$ is the ground-truth prefix $x_{<t}$ (not the model's own previous predictions). With the causal mask, all $T$ predictions are computed from a single forward pass on $x_{1:T}$.

## Theorems and proofs

**Theorem 25.1 (Chain-rule factorization).** For any joint distribution $p$ on $\mathcal{V}^T$,
$p(x_{1:T}) = \prod_{t=1}^T p(x_t\mid x_{<t})$.

*Proof.* Iterate the conditional-probability identity (Ch 8) $p(A,B)=p(A)\,p(B\mid A)$:
$p(x_{1:T}) = p(x_1)\,p(x_2\mid x_1)\,p(x_3\mid x_{1:2})\cdots p(x_T\mid x_{<T})$. $\square$

This is *not* an assumption on the model — it holds for every joint distribution. The modeling choice is to *parameterize* each conditional with a transformer.

**Theorem 25.2 (NTP loss = empirical NLL = cross-entropy with empirical distribution).** Let $\mathcal{D} = \{x^{(n)}_{1:T_n}\}_{n=1}^{N}$ be a corpus of i.i.d. documents drawn from a true distribution $p^\star$. The next-token-prediction loss
$$ \mathcal{L}(\theta) \;=\; -\sum_{n=1}^{N}\sum_{t=1}^{T_n} \log p_\theta\!\left(x^{(n)}_t \mid x^{(n)}_{<t}\right) $$
satisfies $\arg\min_\theta \mathcal{L}(\theta) = \arg\min_\theta H(\hat p_{\mathcal{D}},\, p_\theta)$, where $\hat p_{\mathcal{D}}$ is the empirical distribution and $H$ is cross-entropy.

*Proof.* By Theorem 25.1, $\log p_\theta(x_{1:T}) = \sum_t \log p_\theta(x_t\mid x_{<t})$, so $\mathcal{L}(\theta) = -\sum_n \log p_\theta(x^{(n)})$. The empirical distribution is $\hat p_{\mathcal{D}}(x) = \tfrac{1}{N}\sum_n \mathbb{1}[x = x^{(n)}]$. Then
$$ H(\hat p_{\mathcal{D}}, p_\theta) = -\!\!\sum_{x\in\mathcal{V}^*}\hat p_{\mathcal{D}}(x)\log p_\theta(x) = -\frac{1}{N}\sum_{n=1}^N \log p_\theta(x^{(n)}) = \frac{1}{N}\,\mathcal{L}(\theta). $$
Multiplying by $N>0$ does not change the argmin (Ch 12). $\square$

**Corollary 25.3 (Consistency).** If the family $\{p_\theta\}$ is correctly specified — i.e. $p^\star = p_{\theta^\star}$ for some $\theta^\star$ — and identifiable, then by the standard MLE consistency argument (Ch 12) the minimizer $\hat\theta_N \to \theta^\star$ as $N \to \infty$.

*Proof sketch.* By the law of large numbers, $\tfrac{1}{N}\mathcal{L}(\theta) \to -\mathbb{E}_{p^\star}[\log p_\theta(X)] = H(p^\star, p_\theta)$. Gibbs' inequality (Ch 12) states $H(p^\star, p_\theta) \ge H(p^\star)$ with equality iff $p_\theta = p^\star$. Identifiability + uniform convergence promote pointwise convergence of the minimum to $\theta^\star$. $\square$

**Theorem 25.4 (Causal mask preserves backprop locality).** With the causal mask, for every position pair $(t,s)$ with $s>t$ and every layer's input embedding $h_s^{(0)}$ at position $s$,
$$ \frac{\partial \mathcal{L}_t}{\partial h_s^{(0)}} = 0, $$
where $\mathcal{L}_t = -\log p_\theta(x_t \mid x_{<t})$.

*Proof.* Write the network as a composition of layers. By induction on layer depth $\ell$, the masked attention output at position $i$ is
$ y_i^{(\ell)} = \sum_{j \le i} \alpha_{ij}^{(\ell)} V_j^{(\ell)}, $
because $\alpha_{ij}^{(\ell)} = 0$ for $j > i$. Pointwise sublayers (residuals, MLP, LayerNorm) act per-position and do not introduce dependencies on $j > i$. Hence the position-$t$ output $y_t^{(L)}$ depends only on $\{h_j^{(0)}\}_{j\le t}$. The logit $z_t = W y_t^{(L)}$ inherits this support, so $\partial \mathcal{L}_t / \partial h_s^{(0)} = 0$ for $s>t$. $\square$

This is exactly what makes parallel training sound: stacking the per-position losses $\mathcal{L} = \sum_t \mathcal{L}_t$ and backpropagating through one forward pass gives the *same* gradient as $T$ separate forward passes on prefixes $x_{1:t}$.

**Theorem 25.5 (Train vs. inference asymmetry).** Training: feed $x_{1:T}$ once; with the mask, obtain all $T$ logits in $\Theta(T^2 d)$ time and update $\theta$ on $\sum_t \mathcal{L}_t$. Inference: starting from a prompt $x_{1:k}$, sample $\hat x_{k+1} \sim p_\theta(\cdot\mid x_{1:k})$, append, and repeat — autoregressive generation requires $T-k$ sequential forward passes because each sampled token feeds the next.

The asymmetry stems from teacher forcing: at train time we *know* $x_t$, so position $t$ can be computed in parallel with position $t+1$; at inference time $x_{t+1}$ is whatever the model just produced.

**Definition (Perplexity).** $\mathrm{PPL}(x_{1:T}) = \exp\!\big(\tfrac{1}{T}\sum_t -\log p_\theta(x_t\mid x_{<t})\big)$. It is the geometric mean of $1/p_\theta(x_t\mid x_{<t})$. A uniform model over $|\mathcal{V}|$ tokens achieves $\mathrm{PPL} = |\mathcal{V}|$; lower is better.

## Code sketch

The notebook builds (i) an $8\times 8$ causal mask, (ii) a single-head causal attention layer and verifies row $t$ has zero weight on $j>t$, (iii) a tiny logits "model" and checks NTP loss equals summed cross-entropy, (iv) a perturbation experiment showing that changing token $T$ leaves the logit at $T-1$ untouched (Theorem 25.4), and (v) perplexity of uniform vs trained model.

## Connection to LLMs

Every modern decoder-only LLM — GPT-1/2/3/4, Llama, Claude, Gemini — is a causal language model trained with exactly this loss. Pre-training scales the corpus $\mathcal{D}$ (trillions of tokens) and the parameter count (Ch 27 onwards), but the objective is the one derived above: minimize $-\sum_t \log p_\theta(x_t\mid x_{<t})$, with the causal mask of Definition 25.2 making all $T$ losses backprop-independent so that one forward pass yields $T$ supervisory signals. Perplexity remains the canonical eval metric on held-out text.
