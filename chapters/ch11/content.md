## Motivation

Chapter 9 gave us random variables and their distributions; Chapter 10 gave us expectations and Jensen's inequality for convex $\phi$. We now ask a sharper question: how much *information* does an outcome carry, and how *different* are two distributions over the same space? The answers — Shannon entropy, cross-entropy, and the Kullback–Leibler divergence — are not optional flavor. They *are* the loss surface of every modern language model. Cross-entropy is the next-token training objective (Chapter 17, 25); KL is the trust-region regularizer in RLHF and DPO (Chapter 28); the entropy of the model's softmax is what people mean by "diversity" or "temperature." This chapter builds the machinery from the only axiom we need: $-\ln$ is convex.

We use the natural log $\ln$ throughout for clean derivatives. Switching to $\log_2$ multiplies every formula by $1/\ln 2$ and converts nats to bits; nothing else changes.

## Definitions

**Definition (Self-information).** For a discrete random variable $X$ with PMF $p$ and outcome $x$ in the support of $p$,
$$I(x) := -\ln p(x).$$
Rare outcomes carry more information; $I(x) \to \infty$ as $p(x) \to 0$.

**Definition (Shannon entropy).** The *entropy* of $p$ is the expected self-information:
$$H(p) := \mathbb{E}_{x \sim p}[I(x)] = -\sum_{x} p(x)\, \ln p(x),$$
with the convention $0 \ln 0 = 0$ (justified by $\lim_{t \downarrow 0} t \ln t = 0$).

**Definition (Joint and conditional entropy).** For a joint PMF $p_{X,Y}(x,y)$,
$$H(X, Y) = -\sum_{x,y} p_{X,Y}(x,y) \ln p_{X,Y}(x,y), \qquad H(Y \mid X) = -\sum_{x,y} p_{X,Y}(x,y)\, \ln p_{Y \mid X}(y \mid x).$$

**Definition (Cross-entropy).** For PMFs $p, q$ with $\mathrm{supp}(p) \subseteq \mathrm{supp}(q)$,
$$H(p, q) := -\sum_{x} p(x)\, \ln q(x) = \mathbb{E}_{x \sim p}[-\ln q(x)].$$

**Definition (Kullback–Leibler divergence).**
$$D_{\mathrm{KL}}(p \,\|\, q) := \sum_{x} p(x) \ln \frac{p(x)}{q(x)} = \mathbb{E}_{x \sim p}\!\left[\ln \frac{p(x)}{q(x)}\right].$$
Note: $D_{\mathrm{KL}}$ is *not* symmetric and *not* a metric.

**Definition (Mutual information).** For joint $p_{X,Y}$ with marginals $p_X, p_Y$,
$$I(X; Y) := D_{\mathrm{KL}}\!\left(p_{X,Y} \,\|\, p_X \otimes p_Y\right) = \sum_{x,y} p_{X,Y}(x,y) \ln \frac{p_{X,Y}(x,y)}{p_X(x) p_Y(y)}.$$

## Theorems and proofs

**Theorem 11.1 (Gibbs' inequality / KL nonnegativity).** For PMFs $p, q$ on the same finite support with $q(x) > 0$ wherever $p(x) > 0$,
$$D_{\mathrm{KL}}(p \,\|\, q) \geq 0,$$
with equality iff $p(x) = q(x)$ for all $x$ with $p(x) > 0$.

*Proof.* The function $\phi(t) = -\ln t$ is strictly convex on $(0, \infty)$ (its second derivative is $1/t^2 > 0$; convexity is the property used in Jensen's inequality, Chapter 10). Compute, treating the sum as $\mathbb{E}_{x \sim p}$:
$$-D_{\mathrm{KL}}(p \,\|\, q) = \sum_{x : p(x) > 0} p(x) \ln \frac{q(x)}{p(x)} = \mathbb{E}_{x \sim p}\!\left[-\phi\!\left(\tfrac{q(x)}{p(x)}\right)\right] = -\,\mathbb{E}_{x \sim p}\!\left[\phi\!\left(\tfrac{q(x)}{p(x)}\right)\right].$$
By Jensen (Chapter 10), $\mathbb{E}[\phi(Z)] \geq \phi(\mathbb{E}[Z])$, so
$$-D_{\mathrm{KL}}(p \,\|\, q) \leq -\phi\!\left(\mathbb{E}_{x \sim p}\!\left[\tfrac{q(x)}{p(x)}\right]\right) = \ln\!\left(\sum_{x : p(x) > 0} q(x)\right) \leq \ln 1 = 0,$$
where the last step uses $\sum_x q(x) = 1$. Hence $D_{\mathrm{KL}}(p \,\|\, q) \geq 0$. Strict convexity of $\phi$ makes Jensen an equality iff $q(x)/p(x)$ is constant $p$-a.s., and the constant must be $1$ (both sum to $1$); equivalently $p = q$. $\blacksquare$

**Theorem 11.2 (Cross-entropy decomposition).** $H(p, q) = H(p) + D_{\mathrm{KL}}(p \,\|\, q).$

*Proof.* Direct algebra:
$$H(p, q) = -\sum_x p(x) \ln q(x) = -\sum_x p(x)\bigl[\ln p(x) + \ln \tfrac{q(x)}{p(x)}\bigr] = H(p) + D_{\mathrm{KL}}(p \,\|\, q). \quad \blacksquare$$

**Corollary 11.3.** $H(p, q) \geq H(p)$ with equality iff $q = p$. So the cross-entropy minimum over $q$ is achieved at $q = p$, with value $H(p)$.

**Theorem 11.4 (Maximum entropy on a finite alphabet).** For any PMF $p$ supported on a set of size $K$,
$$H(p) \leq \ln K,$$
with equality iff $p$ is the uniform distribution $u(x) = 1/K$.

*Proof.* Let $u$ be uniform. Then
$$D_{\mathrm{KL}}(p \,\|\, u) = \sum_x p(x) \ln \frac{p(x)}{1/K} = -H(p) + \ln K \sum_x p(x) = \ln K - H(p).$$
By Theorem 11.1, this is $\geq 0$, so $H(p) \leq \ln K$, with equality iff $p = u$. $\blacksquare$

**Theorem 11.5 (Chain rule for entropy).** $H(X, Y) = H(X) + H(Y \mid X).$

*Proof.* Use $\ln p_{X,Y}(x,y) = \ln p_X(x) + \ln p_{Y \mid X}(y \mid x)$:
$$H(X, Y) = -\sum_{x,y} p_{X,Y}(x,y) \ln p_X(x) - \sum_{x,y} p_{X,Y}(x,y) \ln p_{Y \mid X}(y \mid x).$$
Marginalizing $y$ in the first sum gives $-\sum_x p_X(x) \ln p_X(x) = H(X)$; the second sum is $H(Y \mid X)$ by definition. $\blacksquare$

**Theorem 11.6 (Mutual information nonnegativity).** $I(X; Y) \geq 0$, with equality iff $X$ and $Y$ are independent.

*Proof.* By definition $I(X; Y) = D_{\mathrm{KL}}(p_{X,Y} \,\|\, p_X \otimes p_Y)$. Apply Theorem 11.1: this is $\geq 0$ with equality iff $p_{X,Y}(x,y) = p_X(x) p_Y(y)$ for all $x, y$ — exactly the definition of independence. Combined with the chain rule (Theorem 11.5), an equivalent identity is
$$I(X; Y) = H(X) + H(Y) - H(X, Y) = H(Y) - H(Y \mid X). \quad \blacksquare$$

## Code sketch

The notebook builds two categorical distributions $p, q$ over a 5-token alphabet, implements `entropy`, `cross_entropy`, and `kl` from their definitions, and verifies the decomposition $H(p, q) = H(p) + D_{\mathrm{KL}}(p \,\|\, q)$ to machine precision. It then samples 50 random Dirichlet pairs (seed 0) and confirms $D_{\mathrm{KL}} \geq 0$ for every pair, with $D_{\mathrm{KL}}(p \,\|\, p) = 0$. For maximum entropy, it computes $H(\text{uniform on 8})$ and checks it equals $\ln 8$, then sweeps 100 non-uniform draws and confirms $H < \ln 8$. Finally it builds a correlated joint on $\{0,1\}^2$, computes $I(X; Y)$ both as $H(X) + H(Y) - H(X, Y)$ and as $D_{\mathrm{KL}}(p_{X,Y} \,\|\, p_X \otimes p_Y)$, verifies they agree, and shows $I = 0$ for the product distribution.

## Connection to LLMs

A causal LM defines $p_\theta(x_t \mid x_{<t})$ via softmax (Chapter 9). The training objective on a corpus drawn from $p_{\mathrm{data}}$ is
$$\mathcal{L}(\theta) = \mathbb{E}_{x \sim p_{\mathrm{data}}}\!\left[-\ln p_\theta(x_t \mid x_{<t})\right] = H(p_{\mathrm{data}}, p_\theta) = H(p_{\mathrm{data}}) + D_{\mathrm{KL}}(p_{\mathrm{data}} \,\|\, p_\theta).$$
Theorem 11.2 explains exactly why minimizing cross-entropy *is* minimizing KL to the data distribution: the entropy term $H(p_{\mathrm{data}})$ does not depend on $\theta$. Theorem 11.4 sets the upper bound — a uniform LM over a 50k-token vocabulary has $\ln 50000 \approx 10.82$ nats of entropy, which is the floor every model improves on. In RLHF and DPO (Chapter 28), the policy update is regularized by $D_{\mathrm{KL}}(\pi_\theta \,\|\, \pi_{\mathrm{ref}})$ to keep the fine-tuned model close to the SFT initialization; Gibbs' inequality is what makes that penalty a meaningful "distance." And the entropy of the sampler's softmax — controlled by temperature — is the diversity knob (Chapter 25).
