## Motivation

Chapter 8 built probability spaces $(\Omega, \mathcal{F}, \mathbb{P})$ as the bedrock for modeling uncertainty. But raw $\Omega$ is rarely what we *measure*: we measure a temperature, a token id, a pixel intensity. A **random variable** is the bridge that turns abstract outcomes into numbers we can integrate, sum, optimize, and—most importantly for us—differentiate. Every loss function in deep learning is an expectation over a random variable; every sampler in a language model is a draw from a distribution; every softmax is a categorical PMF. This chapter formalizes the machinery.

## Definitions

**Definition (Random variable).** Let $(\Omega, \mathcal{F}, \mathbb{P})$ be a probability space. A function $X : \Omega \to \mathbb{R}$ is a *random variable* if for every $x \in \mathbb{R}$,
$$\{X \leq x\} := \{\omega \in \Omega : X(\omega) \leq x\} \in \mathcal{F}.$$
This *measurability* condition guarantees that probabilities of events defined through $X$ are well-defined.

**Definition (Distribution).** The *distribution* (or *law*) of $X$ is the pushforward measure $\mu_X$ on $(\mathbb{R}, \mathcal{B}(\mathbb{R}))$ defined by
$$\mu_X(B) = \mathbb{P}(X \in B), \qquad B \in \mathcal{B}(\mathbb{R}).$$

**Definition (CDF).** The *cumulative distribution function* of $X$ is $F_X : \mathbb{R} \to [0,1]$,
$$F_X(x) = \mathbb{P}(X \leq x) = \mu_X((-\infty, x]).$$

**Definition (Discrete RV / PMF).** $X$ is *discrete* if it takes values in a countable set $S \subset \mathbb{R}$. The *probability mass function* is $p_X(x) = \mathbb{P}(X = x)$ for $x \in S$.

**Definition (Continuous RV / PDF).** $X$ is *(absolutely) continuous* if there exists a non-negative measurable $f_X : \mathbb{R} \to [0,\infty)$, the *probability density function*, such that
$$\mathbb{P}(X \in A) = \int_A f_X(x)\, dx \quad \text{for every Borel } A.$$

**Standard families.** Bernoulli$(p)$: $p_X(1) = p$, $p_X(0) = 1-p$. Binomial$(n,p)$: $p_X(k) = \binom{n}{k} p^k (1-p)^{n-k}$. Categorical$(\pi_1,\dots,\pi_K)$: $p_X(k) = \pi_k$ with $\sum \pi_k = 1$. Geometric$(p)$: $p_X(k) = (1-p)^{k-1} p$, $k \geq 1$. Uniform$[a,b]$: $f_X(x) = \frac{1}{b-a} \mathbf{1}_{[a,b]}(x)$. Gaussian $\mathcal{N}(\mu,\sigma^2)$: $f_X(x) = \frac{1}{\sqrt{2\pi}\sigma} \exp(-(x-\mu)^2/(2\sigma^2))$.

**Joint, marginal, conditional.** For $(X, Y)$ on the same space, the *joint* distribution lives on $\mathbb{R}^2$. *Marginals* are obtained by integrating/summing out: $f_X(x) = \int f_{X,Y}(x,y)\, dy$. The *conditional* density is $f_{Y \mid X}(y \mid x) = f_{X,Y}(x,y) / f_X(x)$ when $f_X(x) > 0$.

## Theorems

**Theorem 9.1 (Properties of CDF).** $F_X$ is (i) non-decreasing, (ii) right-continuous, (iii) $\lim_{x \to -\infty} F_X(x) = 0$, (iv) $\lim_{x \to \infty} F_X(x) = 1$.

*Proof.* (i) If $x \leq y$, then $\{X \leq x\} \subseteq \{X \leq y\}$, so monotonicity of $\mathbb{P}$ gives $F_X(x) \leq F_X(y)$.

(ii) Fix $x$. Let $x_n \downarrow x$. Then $\{X \leq x_n\} \downarrow \{X \leq x\}$ (intersection over $n$). By the *continuity of measure from above* (Chapter 8, Theorem on monotone convergence of measures), since $\mathbb{P}(\{X \leq x_1\}) \leq 1 < \infty$,
$$F_X(x_n) = \mathbb{P}(X \leq x_n) \to \mathbb{P}(X \leq x) = F_X(x).$$

(iii) Take $x_n \downarrow -\infty$. Then $\{X \leq x_n\} \downarrow \emptyset$, so $F_X(x_n) \to \mathbb{P}(\emptyset) = 0$.

(iv) Take $x_n \uparrow \infty$. Then $\{X \leq x_n\} \uparrow \Omega$, so by continuity from below, $F_X(x_n) \to \mathbb{P}(\Omega) = 1$. $\blacksquare$

**Theorem 9.2 (Normalization).** $\sum_{x \in S} p_X(x) = 1$ (discrete) and $\int_{\mathbb{R}} f_X(x)\, dx = 1$ (continuous).

*Proof.* The events $\{X = x\}$, $x \in S$, are disjoint and their union is $\{X \in S\} = \Omega$ (since $X$ is $S$-valued). Countable additivity gives $1 = \mathbb{P}(\Omega) = \sum_{x \in S} \mathbb{P}(X = x) = \sum_x p_X(x)$. The continuous case: take $A = \mathbb{R}$ in the defining property: $1 = \mathbb{P}(X \in \mathbb{R}) = \int_\mathbb{R} f_X$. $\blacksquare$

**Theorem 9.3 (Change of variables, 1-D).** Let $X$ have density $f_X$, and let $g: \mathbb{R} \to \mathbb{R}$ be strictly monotone and $C^1$ on the support of $X$. Then $Y = g(X)$ has density
$$f_Y(y) = f_X(g^{-1}(y)) \,\bigl|(g^{-1})'(y)\bigr|.$$

*Proof.* Suppose $g$ is strictly increasing (decreasing case is symmetric). For any $y$,
$$F_Y(y) = \mathbb{P}(g(X) \leq y) = \mathbb{P}(X \leq g^{-1}(y)) = F_X(g^{-1}(y)).$$
Differentiating using the chain rule (Chapter 3):
$$f_Y(y) = \frac{d}{dy} F_X(g^{-1}(y)) = f_X(g^{-1}(y)) \cdot (g^{-1})'(y).$$
Since $g^{-1}$ is increasing, $(g^{-1})'(y) > 0$, so the absolute value is automatic. For decreasing $g$, $\{g(X) \leq y\} = \{X \geq g^{-1}(y)\}$, $F_Y(y) = 1 - F_X(g^{-1}(y))$, and differentiation yields $-f_X(g^{-1}(y)) (g^{-1})'(y)$, with $(g^{-1})' < 0$, again giving the absolute value. $\blacksquare$

**Theorem 9.4 (Inverse-CDF sampling).** Let $F$ be a CDF with generalized inverse $F^{-1}(u) = \inf\{x : F(x) \geq u\}$. If $U \sim \mathrm{Uniform}[0,1]$, then $X := F^{-1}(U)$ has CDF $F$.

*Proof.* It suffices to show $\{F^{-1}(U) \leq x\} = \{U \leq F(x)\}$ (up to a null set). If $F^{-1}(u) \leq x$, then by definition of infimum and right-continuity of $F$, $F(x) \geq u$. Conversely, if $u \leq F(x)$, then $x \in \{x' : F(x') \geq u\}$, so $F^{-1}(u) \leq x$. Therefore
$$\mathbb{P}(X \leq x) = \mathbb{P}(U \leq F(x)) = F(x),$$
using that $U$ is uniform on $[0,1]$. $\blacksquare$

## Code sketch

The notebook implements: (1) a 5-class categorical from softmax of fixed logits, with PMF/CDF and inverse-CDF sampling against `np.random.seed(0)`; (2) a numerical Gaussian density with Riemann-sum normalization and CDF; (3) the change-of-variable check for $Y = -\ln X$, $X \sim U[0,1]$, yielding $Y \sim \mathrm{Exp}(1)$; (4) a final inverse-CDF sampler convergence test.

## Connection to LLMs

A causal language model outputs logits $z_t \in \mathbb{R}^V$ at each step, which softmax maps to a categorical distribution
$$p_\theta(x_t \mid x_{<t}) = \mathrm{softmax}(z_t).$$
This is a discrete RV over the vocabulary; greedy decoding picks $\arg\max$, while temperature/nucleus sampling draws from this categorical. The standard implementation is *exactly* inverse-CDF sampling on the cumulative softmax (Theorem 9.4), or the equivalent **Gumbel-max trick** $\arg\max_k(z_k + G_k)$ with $G_k \sim \mathrm{Gumbel}(0,1)$. Cross-entropy loss (Chapter 17) is $-\log p_\theta(x_t \mid x_{<t})$, an expectation under the data distribution; the causal LM training objective (Chapter 25) is the joint log-likelihood factored by the chain rule for conditionals defined here.
