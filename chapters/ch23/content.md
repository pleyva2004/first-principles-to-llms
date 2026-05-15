## Motivation

A single attention head (Chapter 22) is a beautiful linear-algebraic object, but a transformer is not one head — it is a *stack* of dozens to hundreds of nearly identical *blocks*. Two empirical facts dominate the design of those blocks. First, deep stacks of pure compositions $f_L \circ \cdots \circ f_1$ suffer from vanishing or exploding Jacobians (we proved a strict version of this for RNNs in Chapter 20). Second, the activations between layers drift in scale and mean unless we actively re-normalize. The transformer block solves both problems by wrapping every learned sublayer in a **residual connection** and a **normalization layer**, then alternating an **attention** sublayer with a position-wise **feedforward** MLP. This chapter pins down the algebra and proves the gradient-flow guarantee that makes 96-layer GPTs trainable end-to-end.

## Definitions

**Definition (Residual connection).** Given any map $F : \mathbb{R}^d \to \mathbb{R}^d$, the *residual* (or *skip*) wrapper is $\mathrm{Res}_F(x) = F(x) + x$.

**Definition (LayerNorm; Ba, Kiros, Hinton 2016).** For $x \in \mathbb{R}^d$ let $\mu = \tfrac1d \sum_i x_i$ and $\sigma^2 = \tfrac1d \sum_i (x_i - \mu)^2$. Then
$$\mathrm{LN}(x) = \gamma \odot \frac{x - \mu \mathbf{1}}{\sqrt{\sigma^2 + \varepsilon}} + \beta, \qquad \gamma, \beta \in \mathbb{R}^d.$$

**Definition (RMSNorm; Zhang & Sennrich 2019).** With $r(x)^2 = \tfrac1d \sum_i x_i^2$,
$$\mathrm{RMSN}(x) = \gamma \odot \frac{x}{\sqrt{r(x)^2 + \varepsilon}}.$$
RMSN drops the mean-subtraction step, removes the $\beta$ shift, and saves roughly 7% of the per-token FLOPs of LN.

**Definition (Position-wise FFN).** Two affine maps separated by a pointwise nonlinearity $\sigma$ (Chapter 16: usually GELU; in Llama-style models, SwiGLU):
$$\mathrm{FFN}(x) = W_2 \sigma(W_1 x + b_1) + b_2, \quad W_1 \in \mathbb{R}^{d_{\mathrm{ff}} \times d}, \ W_2 \in \mathbb{R}^{d \times d_{\mathrm{ff}}}, \ d_{\mathrm{ff}} = 4d.$$
This is exactly the one-hidden-layer MLP of Chapter 15, applied independently to every token position.

**Definition (Pre-norm transformer block).** Given a multi-head self-attention map $\mathrm{MHA}$ (Chapter 22), the modern (GPT-2 onward) block is
$$z = x + \mathrm{MHA}(\mathrm{LN}(x)), \qquad y = z + \mathrm{FFN}(\mathrm{LN}(z)).$$
The original Vaswani-2017 *post-norm* block applied LN *after* the residual sum: $z = \mathrm{LN}(x + \mathrm{MHA}(x))$. We motivate the switch in Theorem 3.

## Theorems

**Theorem 1 (Residual gradient identity).** *Let $y = F(x) + x$ with $F : \mathbb{R}^d \to \mathbb{R}^d$ differentiable. Then $\partial y / \partial x = J_F(x) + I$. Iterating across $L$ residual blocks $x_{\ell+1} = x_\ell + F^{(\ell)}(x_\ell)$, the end-to-end Jacobian is*
$$\frac{\partial x_L}{\partial x_0} \;=\; \prod_{\ell = L-1}^{0} \bigl(I + J_{F^{(\ell)}}(x_\ell)\bigr).$$

*Proof.* Direct from the chain rule (Chapter 18): $\partial x_{\ell+1}/\partial x_\ell = I + J_{F^{(\ell)}}(x_\ell)$, and Jacobians compose by left-multiplication. $\square$

**Corollary (Gradient flow).** Suppose $\|J_{F^{(\ell)}}\|_2 \le \kappa < 1$. Without residuals the backpropagated gradient norm is bounded by $\kappa^L$ — exponential decay, exactly the RNN catastrophe of Chapter 20. With residuals, expanding the product gives an identity-plus-perturbation:
$$\frac{\partial x_L}{\partial x_0} = I + \sum_\ell J_{F^{(\ell)}} + \sum_{\ell < m} J_{F^{(m)}} J_{F^{(\ell)}} + \cdots,$$
so $\|\partial x_L/\partial x_0\|_2 \ge 1 - L\kappa - \binom{L}{2}\kappa^2 - \cdots$, which stays bounded away from zero for any depth provided $\kappa$ is small enough. The identity term *guarantees* a non-vanishing path from output to input.

**Theorem 2 (LN invariances and Jacobian).** *For any $\alpha > 0$ and $\beta \in \mathbb{R}$, $\mathrm{LN}(\alpha x + \beta \mathbf{1}) = \mathrm{LN}(x)$ (we momentarily set $\gamma = 1, \beta_{\mathrm{LN}} = 0, \varepsilon = 0$ to isolate the normalizer).*

*Proof.* Let $x' = \alpha x + \beta \mathbf{1}$. Mean: $\mu' = \alpha \mu + \beta$. Variance: $\sigma'^2 = \tfrac1d \sum (\alpha x_i + \beta - \alpha \mu - \beta)^2 = \alpha^2 \sigma^2$. Hence
$$\frac{x' - \mu' \mathbf{1}}{\sigma'} = \frac{\alpha(x - \mu \mathbf{1})}{\alpha \sigma} = \frac{x - \mu \mathbf{1}}{\sigma}. \qquad \square$$

For the Jacobian, write $\hat x = (x - \mu \mathbf{1})/\sigma$. A short calculation (using $\partial \mu / \partial x_j = 1/d$ and $\partial \sigma / \partial x_j = (x_j - \mu)/(d\sigma)$) yields
$$\frac{\partial \hat x_i}{\partial x_j} = \frac{1}{\sigma} \left( \delta_{ij} - \frac{1}{d} - \frac{1}{d} \hat x_i \hat x_j \right).$$
The two subtracted terms project out the constant ($\mathbf{1}$) and $\hat x$ directions — exactly the directions LN is invariant to. The Jacobian is therefore rank $d - 2$, with kernel $\mathrm{span}\{\mathbf{1}, x - \mu \mathbf{1}\}$.

**Proposition 3 (RMSN $\approx$ LN on centered data).** *If $\mu(x) = 0$ then $\mathrm{LN}(x) = \mathrm{RMSN}(x)$ (with the LN $\beta$ absorbed into a downstream bias). For arbitrary $x$, $\mathrm{RMSN}(x) - \mathrm{LN}(x) = \gamma \odot \mu(x) \mathbf{1} / \sqrt{r(x)^2 + \varepsilon} + O(\mu/r)$.*

*Proof.* When $\mu = 0$, $r(x)^2 = \sigma(x)^2$ and $x - \mu \mathbf 1 = x$, so the two formulas coincide. The general expansion is one line of algebra. $\square$

The empirical observation behind RMSN (verified in our code cells) is that after a few transformer layers, residual streams in pre-LN networks already have near-zero mean, so dropping the centering step costs essentially no quality while saving compute and a parameter vector $\beta$.

**Remark (Pre-norm vs post-norm gradient scale).** With He/Xavier initialization (Chapter 17), assume each sublayer output has the same variance as its input. In a *post-norm* stack the residual sum $x + F(x)$ has variance $2 \mathrm{Var}(x)$, which LN immediately rescales to $\mathrm{Var}(x)$ — but the gradient picks up a factor of $1/\sqrt 2$ per block, and after $L$ blocks the gradient at the input scales like $2^{-L/2}$, requiring careful warmup. In a *pre-norm* stack the residual stream itself is never rescaled; only the *input to* each sublayer is normalized. The end-to-end Jacobian is Theorem 1's $\prod (I + J_F)$ with each $J_F$ acting on a normalized input, so the gradient norm at $x_0$ remains $\Theta(1)$ independent of $L$. This is why every model from GPT-2 onward (and Llama, Claude, Gemini, Mistral, Qwen, $\ldots$) uses pre-norm.

## Code sketch

The companion notebook (i) implements one full pre-norm block in numpy and verifies shape preservation; (ii) stacks 20 residual blocks vs 20 plain blocks and measures backpropagated gradient norms by finite differences; (iii) checks LN invariance under $x \mapsto \alpha x + \beta$ across many random $(\alpha, \beta)$; (iv) compares LN vs RMSN on centered and uncentered inputs.

## Connection to LLMs

Every layer of every modern decoder LLM is a tiny variation of the block above. GPT-2/3/4 use pre-LN with GELU FFN. Llama, Mistral, and Qwen swap LN for RMSN, GELU for SwiGLU (Chapter 16), and MHA for grouped-query attention (Chapter 24); Llama also uses RoPE positional encoding (Chapter 25) inside the attention sublayer. The block remains $z = x + \mathrm{Sublayer}_1(\mathrm{Norm}(x)),\ y = z + \mathrm{Sublayer}_2(\mathrm{Norm}(z))$. Stacking 32–120 such blocks, plus the embedding (Chapter 21), the unembedding tied to the embedding, and a final norm before the LM head, gives the entire forward pass of a frontier-scale language model. The training-stability argument of Theorem 1 is precisely what makes the 1.5T-parameter regime (Chapters 27, 28) reachable at all.
