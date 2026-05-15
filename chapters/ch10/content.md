## Motivation

Random variables (Chapter 9) describe uncertain outcomes. To compress an entire distribution into a single representative number, we use the **expectation**. Two further numbers — **variance** and **covariance** — describe spread and joint linear association. These three quantities, together with **Jensen's inequality**, are the structural backbone of every loss function and convergence proof in modern deep learning. Training an LLM is, formally, the minimization of an expectation $\mathbb{E}_{x \sim \mathcal{D}}[\ell(\theta; x)]$; SGD is a Monte Carlo estimator of that expectation; the variance of the estimator dictates convergence speed (we will revisit this in Chapter 13).

## Definitions

**Definition 10.1 (Expectation).** For a discrete random variable $X$ with pmf $p_X$,
$$\mathbb{E}[X] = \sum_x x\,p_X(x),$$
provided $\sum_x |x|\,p_X(x) < \infty$. For a continuous $X$ with density $f_X$,
$$\mathbb{E}[X] = \int_{\mathbb{R}} x\,f_X(x)\,dx,$$
provided the integral is absolutely convergent. In full generality, $\mathbb{E}[X] := \int_\Omega X\,d\mathbb{P}$, the Lebesgue integral of $X$ against the probability measure (we cite without proof; see Billingsley, *Probability and Measure*).

**Definition 10.2 (Variance).** $\mathrm{Var}(X) := \mathbb{E}[(X - \mathbb{E}[X])^2]$, when the expectation exists. The standard deviation is $\sigma_X := \sqrt{\mathrm{Var}(X)}$.

**Definition 10.3 (Covariance, correlation).** For $X,Y$ with finite variance,
$$\mathrm{Cov}(X,Y) := \mathbb{E}\big[(X - \mathbb{E}X)(Y - \mathbb{E}Y)\big], \qquad \rho_{X,Y} := \frac{\mathrm{Cov}(X,Y)}{\sigma_X \sigma_Y} \in [-1, 1].$$
The bound $|\rho| \le 1$ is the Cauchy–Schwarz inequality applied to the inner product $\langle U, V\rangle := \mathbb{E}[UV]$.

**Definition 10.4 (Conditional expectation).** For discrete $X, Y$ and any $y$ with $p_Y(y) > 0$,
$$\mathbb{E}[X \mid Y = y] = \sum_x x\,p_{X\mid Y}(x\mid y).$$
Viewed as $y$ varies, $\mathbb{E}[X\mid Y]$ is itself a random variable: a measurable function of $Y$. The general (Lebesgue) definition characterizes $\mathbb{E}[X\mid \mathcal{G}]$ as the unique (a.s.) $\mathcal{G}$-measurable random variable satisfying $\int_A \mathbb{E}[X\mid \mathcal{G}]\,d\mathbb{P} = \int_A X\,d\mathbb{P}$ for every $A \in \mathcal{G}$ (Radon–Nikodym).

## Theorems with proofs

**Theorem 10.5 (Linearity of expectation).** For random variables $X, Y$ with finite expectation and scalars $a, b \in \mathbb{R}$,
$$\mathbb{E}[aX + bY] = a\,\mathbb{E}[X] + b\,\mathbb{E}[Y].$$
*Independence is not required.*

*Proof.* In the discrete case, write the joint pmf $p_{X,Y}$. Then
$$\mathbb{E}[aX + bY] = \sum_{x,y} (ax + by)\,p_{X,Y}(x,y) = a\sum_{x,y} x\,p_{X,Y}(x,y) + b\sum_{x,y} y\,p_{X,Y}(x,y).$$
Marginalizing, $\sum_y p_{X,Y}(x,y) = p_X(x)$ and $\sum_x p_{X,Y}(x,y) = p_Y(y)$, giving $a\,\mathbb{E}[X] + b\,\mathbb{E}[Y]$. The continuous case is identical with sums replaced by integrals; in full generality, linearity is a property of the Lebesgue integral. $\blacksquare$

**Theorem 10.6 (Variance of a sum).**
$$\mathrm{Var}(X + Y) = \mathrm{Var}(X) + \mathrm{Var}(Y) + 2\,\mathrm{Cov}(X, Y).$$

*Proof.* Let $\mu_X = \mathbb{E}[X]$, $\mu_Y = \mathbb{E}[Y]$. By linearity, $\mathbb{E}[X+Y] = \mu_X + \mu_Y$. Then
\begin{align*}
\mathrm{Var}(X+Y) &= \mathbb{E}\big[((X-\mu_X) + (Y-\mu_Y))^2\big] \\
&= \mathbb{E}[(X-\mu_X)^2] + \mathbb{E}[(Y-\mu_Y)^2] + 2\,\mathbb{E}[(X-\mu_X)(Y-\mu_Y)] \\
&= \mathrm{Var}(X) + \mathrm{Var}(Y) + 2\,\mathrm{Cov}(X,Y). \qquad \blacksquare
\end{align*}

If $X \perp Y$, then $\mathrm{Cov}(X,Y) = 0$ and variances add. The converse is false in general.

**Theorem 10.7 (Jensen's inequality).** Let $\phi : \mathbb{R} \to \mathbb{R}$ be convex and $X$ a random variable with $\mathbb{E}|X| < \infty$ and $\mathbb{E}|\phi(X)| < \infty$. Then
$$\phi(\mathbb{E}[X]) \le \mathbb{E}[\phi(X)].$$

*Proof.* Recall (Chapter 7) that a convex function on $\mathbb{R}$ admits a *supporting line* at every interior point of its domain: for $x_0 = \mathbb{E}[X]$ there exists a subgradient $g \in \partial\phi(x_0)$ such that
$$\phi(x) \ge \phi(x_0) + g(x - x_0) \quad \text{for all } x.$$
Substitute $X$ and take expectations. By linearity,
$$\mathbb{E}[\phi(X)] \ge \phi(x_0) + g(\mathbb{E}[X] - x_0) = \phi(\mathbb{E}[X]),$$
since $\mathbb{E}[X] - x_0 = 0$. $\blacksquare$

**Theorem 10.8 (Markov's inequality).** Let $X \ge 0$ a.s. and $a > 0$. Then
$$\mathbb{P}(X \ge a) \le \frac{\mathbb{E}[X]}{a}.$$

*Proof.* Pointwise, $a\cdot \mathbf{1}_{\{X \ge a\}} \le X$ (when $X \ge a$, both sides equal $\le X$; when $X < a$, the left side is $0$ and $X \ge 0$). Take expectations: $a\,\mathbb{P}(X \ge a) \le \mathbb{E}[X]$. Divide by $a$. $\blacksquare$

**Theorem 10.9 (Chebyshev's inequality).** For $X$ with $\mathbb{E}[X] = \mu$ and finite $\sigma^2 = \mathrm{Var}(X)$, and $k > 0$,
$$\mathbb{P}(|X - \mu| \ge k\sigma) \le \frac{1}{k^2}.$$

*Proof.* Apply Markov to $Y := (X - \mu)^2 \ge 0$ with $a = k^2 \sigma^2$:
$$\mathbb{P}((X-\mu)^2 \ge k^2\sigma^2) \le \frac{\mathbb{E}[(X-\mu)^2]}{k^2 \sigma^2} = \frac{1}{k^2}.$$
The event $\{(X-\mu)^2 \ge k^2\sigma^2\}$ equals $\{|X-\mu| \ge k\sigma\}$. $\blacksquare$

**Theorem 10.10 (Weak law of large numbers).** Let $X_1, X_2, \dots$ be i.i.d. with $\mathbb{E}[X_i] = \mu$ and $\mathrm{Var}(X_i) = \sigma^2 < \infty$. Set $\bar{X}_n := \frac{1}{n}\sum_{i=1}^n X_i$. Then for every $\epsilon > 0$,
$$\mathbb{P}(|\bar{X}_n - \mu| \ge \epsilon) \xrightarrow{n\to\infty} 0.$$

*Proof.* By linearity, $\mathbb{E}[\bar{X}_n] = \mu$. By Theorem 10.6 and independence, $\mathrm{Var}(\bar{X}_n) = \sigma^2/n$. Chebyshev gives
$$\mathbb{P}(|\bar{X}_n - \mu| \ge \epsilon) \le \frac{\sigma^2}{n\epsilon^2} \to 0. \qquad \blacksquare$$

## Code sketch

The accompanying notebook (`cells.json`) verifies each theorem on small distributions: a 5-outcome categorical for $\mathbb{E}, \mathrm{Var}$; a perfectly dependent pair $(X, 2X+1)$ for linearity without independence; the convex $\phi(x)=e^x$ on Uniform $\{-1,+1\}$ for Jensen; and a Monte Carlo experiment showing $\bar X_n \to 1/2$ inside the Chebyshev band for $X_i \sim \mathrm{Unif}[0,1]$.

## Connection to LLMs

The training objective of every modern language model has the form
$$\mathcal{L}(\theta) = \mathbb{E}_{x \sim \mathcal{D}}[\ell(\theta; x)],$$
where $\mathcal{D}$ is the data distribution and $\ell$ is (typically) the next-token cross-entropy loss. The data distribution is intractable, so we replace the expectation by an empirical mean over a mini-batch of size $B$:
$$\hat{\mathcal{L}}(\theta) = \frac{1}{B}\sum_{i=1}^B \ell(\theta; x_i).$$
Linearity of expectation guarantees $\mathbb{E}[\nabla\hat{\mathcal{L}}] = \nabla \mathcal{L}$ (the SGD gradient is unbiased). The variance of $\nabla \hat{\mathcal{L}}$ scales as $1/B$ (Theorem 10.6 plus independence within a batch), which is exactly why larger batches give smoother training curves. The weak LLN tells us that as $B \to \infty$ we recover the true loss in probability — this is the formal content of "more data helps" (revisited rigorously in Chapter 13). Jensen's inequality, finally, justifies the variational lower bound underlying every modern likelihood-based generative model: $\log \mathbb{E}[Z] \ge \mathbb{E}[\log Z]$ for $Z > 0$, which we will use in the ELBO derivations of Chapter 22.
