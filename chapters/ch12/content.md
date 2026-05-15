## Motivation

Probability theory (Chapters 7–10) tells us how to reason from a known distribution to data. **Statistical inference** runs the arrow backwards: given data, recover the distribution. Two principles dominate modern practice and underwrite essentially all of deep learning: **maximum likelihood estimation** (MLE) and its generalization, **empirical risk minimization** (ERM). We will show that MLE is precisely the minimizer of cross-entropy between the empirical distribution and the model — connecting Chapter 11's information theory to Chapter 17's language-model objective.

The conceptual move is clean: a *model* picks out a smooth subset $\mathcal{P} \subset \Delta(\mathcal{X})$ of the simplex of all distributions on $\mathcal{X}$. The data picks out a single point $\hat p_n$ in that simplex (the empirical distribution). MLE then *projects* the empirical point onto the model manifold using the KL divergence as "distance". Every other notion in this chapter — ERM, bias-variance, the Gaussian-mean closed form, the consistency theorem — flows from this geometric picture, and so does the entire pre-training objective for modern language models.

## Definitions

A **statistical model** is a family $\mathcal{P} = \{p_\theta : \theta \in \Theta\}$ of probability densities (or mass functions) on a sample space $\mathcal{X}$, indexed by a parameter $\theta$ in a parameter space $\Theta \subseteq \mathbb{R}^d$. We assume the data $X_1,\dots,X_n$ are **iid** from some unknown $p_{\theta^*} \in \mathcal{P}$.

The **likelihood** is the joint density viewed as a function of $\theta$:
$$L(\theta;\mathbf{x}) \;=\; \prod_{i=1}^n p_\theta(x_i), \qquad \ell(\theta;\mathbf{x}) \;=\; \sum_{i=1}^n \log p_\theta(x_i).$$
The **maximum likelihood estimator (MLE)** is
$$\hat\theta_{\mathrm{MLE}} \;=\; \arg\max_{\theta\in\Theta} \ell(\theta;\mathbf{x}).$$

More generally, given a **loss** $\ell(\theta;x)$ (not necessarily $-\log p_\theta$), the **empirical risk minimizer** is
$$\hat\theta_{\mathrm{ERM}} \;=\; \arg\min_{\theta\in\Theta} \frac{1}{n}\sum_{i=1}^n \ell(\theta;x_i).$$
MLE is ERM with $\ell(\theta;x) = -\log p_\theta(x)$.

For an estimator $\hat\theta = \hat\theta(X_1,\dots,X_n)$ of a scalar $\theta^*$:
- **Bias**: $\mathrm{Bias}(\hat\theta) = \mathbb{E}[\hat\theta] - \theta^*$.
- **Variance**: $\mathrm{Var}(\hat\theta) = \mathbb{E}[(\hat\theta - \mathbb{E}\hat\theta)^2]$.
- **Mean squared error**: $\mathrm{MSE}(\hat\theta) = \mathbb{E}[(\hat\theta - \theta^*)^2]$.

## Theorems

### Theorem 12.1 (MLE = minimum cross-entropy)
Let $\hat p_n(x) = \frac{1}{n}\sum_{i=1}^n \mathbf{1}\{x_i = x\}$ be the empirical distribution (discrete case; the density case is analogous). Then
$$\arg\max_\theta \ell(\theta;\mathbf{x}) \;=\; \arg\min_\theta H(\hat p_n,\,p_\theta),$$
where $H(\hat p_n, p_\theta) = -\sum_x \hat p_n(x)\log p_\theta(x)$ is the cross-entropy from Chapter 11.

*Proof.* Direct algebra:
$$\tfrac{1}{n}\,\ell(\theta) \;=\; \tfrac{1}{n}\sum_{i=1}^n \log p_\theta(x_i) \;=\; \sum_x \hat p_n(x)\log p_\theta(x) \;=\; -H(\hat p_n,\,p_\theta).$$
Maximizing $\ell$ is equivalent to maximizing $\ell/n$, which is equivalent to minimizing $H(\hat p_n,p_\theta)$. $\square$

### Corollary 12.2 (MLE = minimum KL)
By the decomposition $H(\hat p_n,p_\theta) = H(\hat p_n) + D_{\mathrm{KL}}(\hat p_n \,\|\, p_\theta)$ from Chapter 11, and since $H(\hat p_n)$ does not depend on $\theta$,
$$\hat\theta_{\mathrm{MLE}} \;=\; \arg\min_\theta D_{\mathrm{KL}}(\hat p_n \,\|\, p_\theta).$$
Thus MLE projects the model family onto the empirical distribution in KL geometry.

### Theorem 12.3 (Bias–variance decomposition)
For any square-integrable estimator $\hat\theta$ of $\theta^* \in \mathbb{R}$,
$$\mathrm{MSE}(\hat\theta) \;=\; \mathrm{Bias}(\hat\theta)^2 + \mathrm{Var}(\hat\theta).$$

*Proof.* Let $\bar\theta = \mathbb{E}[\hat\theta]$. Add and subtract $\bar\theta$:
$$\mathbb{E}[(\hat\theta - \theta^*)^2] = \mathbb{E}\bigl[((\hat\theta-\bar\theta) + (\bar\theta - \theta^*))^2\bigr].$$
Expand the square:
$$= \mathbb{E}[(\hat\theta-\bar\theta)^2] + 2(\bar\theta-\theta^*)\,\mathbb{E}[\hat\theta-\bar\theta] + (\bar\theta-\theta^*)^2.$$
The cross term vanishes since $\mathbb{E}[\hat\theta-\bar\theta] = 0$, leaving $\mathrm{Var}(\hat\theta) + \mathrm{Bias}(\hat\theta)^2$. $\square$

### Theorem 12.4 (MLE for Gaussian mean, known variance)
Let $X_1,\dots,X_n \stackrel{\mathrm{iid}}{\sim} \mathcal{N}(\mu,\sigma^2)$ with $\sigma^2$ known. Then $\hat\mu_{\mathrm{MLE}} = \bar X_n := \tfrac1n\sum_i X_i$.

*Proof.* The log-likelihood is
$$\ell(\mu) = -\tfrac{n}{2}\log(2\pi\sigma^2) - \tfrac{1}{2\sigma^2}\sum_{i=1}^n (X_i - \mu)^2.$$
Differentiate: $\ell'(\mu) = \tfrac{1}{\sigma^2}\sum_i (X_i - \mu)$. Setting $\ell'(\mu) = 0$ yields $\sum_i X_i = n\mu$, i.e. $\hat\mu = \bar X_n$. The second derivative is $-n/\sigma^2 < 0$, so this is a maximum. $\square$

### Theorem 12.5 (Consistency of MLE — sketch)
Under regularity conditions (identifiability, compact $\Theta$, dominated $\log p_\theta$), $\hat\theta_n \xrightarrow{P} \theta^*$.

*Sketch.* By the LLN of Chapter 10, $\tfrac1n \ell_n(\theta) \xrightarrow{P} \mathbb{E}_{\theta^*}[\log p_\theta(X)] =: M(\theta)$. Gibbs' inequality (Chapter 11) gives $M(\theta) \le M(\theta^*)$ with equality iff $p_\theta = p_{\theta^*}$, so $\theta^*$ is the unique maximizer of the limit. A uniform LLN transfers the maximizer of $\ell_n/n$ to that of $M$. $\square$

## Code sketch

Discretize $\mu$, evaluate $\ell(\mu)$ on a grid for Gaussian samples, locate $\arg\max$, compare to $\bar X_n$. For the categorical case, compute $\hat p_n$ and verify cross-entropy is minimized exactly there. For bias-variance, simulate $T$ replicates and decompose the empirical MSE. Finally we solve linear regression as ERM with squared loss via the normal equation, recovering the slope from noisy data and confirming that ERM is *operationally identical* to MLE under the Gaussian-noise log-likelihood — a fact the reader should verify by hand using $-\log p_\theta(y\mid x) = \frac{(y-wx)^2}{2\sigma^2} + \mathrm{const}$.

## Connection to LLMs

Language-model **pre-training** is MLE on a corpus of token sequences. Given a sequence $x_{1:T}$, the autoregressive factorization $p_\theta(x_{1:T}) = \prod_t p_\theta(x_t \mid x_{<t})$ gives the per-corpus log-likelihood
$$\ell(\theta) = \sum_{\text{seq}}\sum_{t} \log p_\theta(x_t \mid x_{<t}),$$
and the standard training objective $-\ell(\theta)/N$ is exactly the **cross-entropy** $H(\hat p_n, p_\theta)$ where $\hat p_n$ is the empirical token distribution. Thus every gradient step a transformer takes (Chapter 25) is a step of MLE = ERM with log-loss = KL projection toward the empirical corpus distribution. Bias-variance reasoning then governs scaling laws and overfitting (Chapter 27).
