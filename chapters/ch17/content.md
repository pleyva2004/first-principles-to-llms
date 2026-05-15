# Loss functions: MSE, cross-entropy; gradients from first principles

## Motivation

A neural network is a parametric map $f_\theta : \mathcal{X} \to \mathcal{Y}$. To *train* it we need a scalar
"badness" score whose gradient with respect to $\theta$ tells us how to nudge weights. That scalar is the **loss
function**. Two losses dominate modern deep learning:

- **Mean squared error (MSE)** for regression targets.
- **Categorical cross-entropy (CE)** for classification, including the next-token prediction at the heart of
  every language model.

Both are not arbitrary engineering choices: each is the negative log-likelihood of a generative model (Ch 12).
And in both cases the *gradient with respect to the network's pre-activation output* collapses to the same
elegant form, $\hat p - y$, which is what makes backpropagation through deep stacks numerically tractable.

## Definitions

**MSE loss.** For a single scalar prediction $\hat y$ against target $y$,
$$
\ell_{\mathrm{MSE}}(y, \hat y) \;=\; \tfrac{1}{2}(y - \hat y)^2.
$$
The factor $\tfrac{1}{2}$ is conventional; it cancels in the gradient.

**Categorical cross-entropy.** Let $y \in \{0,1\}^K$ be a one-hot label ($\sum_k y_k = 1$) and
$\hat p \in \Delta^{K-1}$ a predicted distribution over $K$ classes. Then
$$
\ell_{\mathrm{CE}}(y, \hat p) \;=\; -\sum_{k=1}^K y_k \log \hat p_k.
$$
This is the cross-entropy $H(y, \hat p)$ of Ch 11 evaluated on a single example.

**Cross-entropy with logits (softmax-CE).** In practice we never store $\hat p$; we store unnormalized logits
$z \in \mathbb{R}^K$ and apply softmax (Ch 16). Letting $c \in \{1,\dots,K\}$ be the true class index,
$$
\ell(z, c) \;=\; -\log \mathrm{softmax}(z)_c \;=\; -z_c + \log \sum_{j=1}^K e^{z_j}.
$$
The right-hand form is the **log-sum-exp** identity; it is the only numerically stable way to compute CE
from logits.

**Binary cross-entropy.** When $K = 2$ we represent the prediction by a single logit $z$ with
$\hat p = \sigma(z)$ and $y \in \{0, 1\}$:
$$
\ell(z, y) \;=\; -y \log \sigma(z) - (1-y)\log(1-\sigma(z)).
$$

## Theorems

### Theorem 1 (MSE gradient)

$\nabla_{\hat y} \ell_{\mathrm{MSE}} = \hat y - y$.

*Proof.* Differentiate $\tfrac{1}{2}(y-\hat y)^2$ in $\hat y$: $-\tfrac{1}{2}\cdot 2(y-\hat y) = \hat y - y$. $\square$

### Theorem 2 (MSE = MLE under Gaussian noise)

Suppose $y = f_\theta(x) + \varepsilon$ with $\varepsilon \sim \mathcal{N}(0, \sigma^2)$. Then maximum likelihood
estimation of $\theta$ is equivalent to minimizing MSE.

*Proof.* The Gaussian density gives
$$
-\log p_\theta(y \mid x) \;=\; -\log \frac{1}{\sqrt{2\pi}\,\sigma} + \frac{(y - f_\theta(x))^2}{2\sigma^2}
\;=\; \frac{1}{2\sigma^2}\,(y - f_\theta(x))^2 + C,
$$
where $C$ is independent of $\theta$. Summing over an i.i.d. dataset and dropping the additive constant and
positive multiplicative constant yields $\sum_n \tfrac{1}{2}(y_n - f_\theta(x_n))^2$, which is the MSE
objective. $\square$

### Theorem 3 (Categorical CE = MLE under softmax)

Let the model be $p_\theta(y = c \mid x) = \mathrm{softmax}(z_\theta(x))_c$. Then for a single sample,
$-\log p_\theta(y = c \mid x) = -\log \mathrm{softmax}(z)_c = \ell(z, c)$. Summed across i.i.d. samples,
MLE equals minimization of categorical cross-entropy. This is the single-sample case of the
$\arg\max$-of-likelihood = $\arg\min$-of-cross-entropy identity proved in Ch 12. $\square$

### Theorem 4 (Softmax + CE gradient)

For $\ell(z, c) = -z_c + \log \sum_j e^{z_j}$,
$$
\frac{\partial \ell}{\partial z_j} \;=\; \mathrm{softmax}(z)_j - \mathbf{1}_{[j=c]} \;=\; \hat p_j - y_j.
$$

*Proof.* $\partial(-z_c)/\partial z_j = -\mathbf{1}_{[j=c]}$. For the log-sum-exp,
$$
\frac{\partial}{\partial z_j} \log \sum_k e^{z_k} \;=\; \frac{e^{z_j}}{\sum_k e^{z_k}} \;=\; \mathrm{softmax}(z)_j.
$$
Adding the two gives $\hat p_j - \mathbf{1}_{[j=c]}$. The dramatic cancellation hinges on the appearance of
$\sum_k e^{z_k}$ in *both* the softmax denominator and the CE normalizer; the matrix-vector product implicit
in the softmax Jacobian (Ch 16) collapses to a vector subtraction. This is *the* identity that makes deep
classifiers fast: one subtraction per output unit replaces a $K\times K$ matrix multiply. $\square$

### Theorem 5 (Binary CE + sigmoid gradient)

For $\ell(z, y) = -y\log \sigma(z) - (1-y)\log(1-\sigma(z))$, $\partial \ell / \partial z = \sigma(z) - y$.

*Proof.* Use $\log \sigma(z) = -\log(1+e^{-z})$ and $\log(1-\sigma(z)) = -z - \log(1+e^{-z})$. Then
$\ell = (1-y)z + \log(1+e^{-z})$. Differentiating: $(1-y) - e^{-z}/(1+e^{-z}) = (1-y) - (1-\sigma(z)) =
\sigma(z) - y$. $\square$

### Theorem 6 (CE minimum at $\hat p = y$)

Treat $y, \hat p \in \Delta^{K-1}$ as full distributions. Then $H(y, \hat p) = -\sum_k y_k \log \hat p_k$ is
minimized over $\hat p$ at $\hat p = y$.

*Proof.* $H(y, \hat p) - H(y, y) = -\sum_k y_k \log(\hat p_k / y_k) = D_{\mathrm{KL}}(y \,\|\, \hat p) \ge 0$
by Gibbs' inequality (Ch 11), with equality iff $\hat p = y$. $\square$

## Code sketch

```python
def softmax_ce_with_logits(z, c):
    z = z - z.max()                      # log-sum-exp stabilization
    lse = np.log(np.exp(z).sum())
    loss = -z[c] + lse
    p = np.exp(z - lse)                  # softmax(z)
    grad = p.copy(); grad[c] -= 1.0      # p - y
    return loss, grad
```

The forward and backward share the cached softmax $\hat p$; the entire backward is a single
in-place subtraction.

## Connection to LLMs

A causal language model (Ch 25) at every position $t$ emits logits $z_t \in \mathbb{R}^V$ over a vocabulary of
size $V$. The training loss across a sequence $x_1,\dots,x_T$ is
$$
\mathcal{L}(\theta) \;=\; \sum_{t=1}^{T} -\log p_\theta(x_t \mid x_{<t}) \;=\; \sum_{t=1}^{T} \ell(z_t,\, x_t),
$$
i.e. *cross-entropy with logits, summed over positions*. By Theorem 4 the gradient at the output layer is
$\hat p_t - y_t$ at every position — a single subtraction per token, per vocabulary entry. This signal flows
backwards through the unembedding, the transformer blocks (Ch 27 forward), and finally into the token
embeddings. Without the softmax-CE collapse, gradient computation at $V \approx 10^5$ would be infeasible.
Every LLM ever trained leans on Theorem 4.
