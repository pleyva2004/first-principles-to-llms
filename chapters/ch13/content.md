## Motivation

Chapter 7 proved that gradient descent on an $L$-smooth convex function reaches an $\epsilon$-stationary point in $O(1/\epsilon)$ iterations using the *full* gradient $\nabla F(\theta)$. For a transformer pre-trained on $10^{13}$ tokens, computing $\nabla F$ once means a forward/backward pass over the entire corpus. That is unaffordable: a single update would take weeks. The fix is to estimate the gradient from a random *mini-batch* of examples and accept a noisy step in exchange for a cheap one. This chapter justifies the trade. We define the stochastic objective, prove that mini-batching reduces variance as $1/B$, and combine the descent lemma of Chapter 7 with the linearity of expectation (Chapter 10) to derive the canonical $O(1/\sqrt{T})$ rate of stochastic gradient descent on smooth (possibly non-convex) losses. We sketch the strongly-convex $O(1/T)$ rate and state the Robbins–Monro conditions for almost-sure convergence under diminishing step sizes.

## Definitions

**Stochastic objective.** Let $\xi \sim \mathcal{D}$ be a random data point and $f(\cdot; \xi) : \mathbb{R}^n \to \mathbb{R}$ a loss. The *population risk* is $F(\theta) := \mathbb{E}_{\xi \sim \mathcal{D}}[f(\theta; \xi)]$, an expectation in the sense of Chapter 10. Training data are i.i.d. draws $\xi_1, \xi_2, \dots$ from $\mathcal{D}$.

**Stochastic gradient.** $\hat g(\theta; \xi) := \nabla_\theta f(\theta; \xi)$. Under mild regularity (interchange of $\nabla$ and $\mathbb{E}$; Chapter 10) it is *unbiased*: $\mathbb{E}_{\xi}[\hat g(\theta; \xi)] = \nabla F(\theta)$.

**SGD update.** Given a step size $\eta_t > 0$ and an independent sample $\xi_t \sim \mathcal{D}$,
$$\theta_{t+1} = \theta_t - \eta_t \hat g(\theta_t; \xi_t).$$

**Mini-batch SGD.** Draw $B$ i.i.d. samples $\xi_1, \dots, \xi_B$ and average:
$$\hat g_B(\theta) := \frac{1}{B} \sum_{j=1}^{B} \nabla_\theta f(\theta; \xi_j).$$

**Bounded variance.** We assume $\mathbb{E}_\xi \|\hat g(\theta; \xi) - \nabla F(\theta)\|^2 \leq \sigma^2$ for all $\theta$. This is the standard $\sigma^2$-noise model.

## Theorems and proofs

**Theorem 13.1 (Variance reduction by batching).** Under bounded variance and i.i.d. sampling,
$$\mathbb{E}\|\hat g_B(\theta) - \nabla F(\theta)\|^2 \leq \frac{\sigma^2}{B}.$$

*Proof.* Let $Z_j := \hat g(\theta; \xi_j) - \nabla F(\theta)$. The $Z_j$ are i.i.d., zero-mean, with $\mathbb{E}\|Z_j\|^2 \leq \sigma^2$. Then $\hat g_B - \nabla F = \frac{1}{B}\sum_j Z_j$, and
$$\mathbb{E}\Big\|\tfrac{1}{B}\sum_j Z_j\Big\|^2 = \tfrac{1}{B^2}\sum_{j,k} \mathbb{E}\langle Z_j, Z_k\rangle = \tfrac{1}{B^2}\sum_j \mathbb{E}\|Z_j\|^2 \leq \frac{\sigma^2}{B},$$
where cross terms vanish by independence and zero mean (Theorem 10.6, variance of a sum). $\square$

This is the LLN of Chapter 10 applied to the gradient: doubling the batch halves the noise variance.

**Theorem 13.2 (SGD on $L$-smooth, possibly non-convex $F$).** Suppose $F$ is $L$-smooth, bounded below by $F^\star$, and run SGD with constant step $\eta = \min(1/L, c/\sqrt{T})$ for $T$ iterations, with stochastic gradients of variance $\leq \sigma^2/B$. Then
$$\frac{1}{T}\sum_{t=0}^{T-1} \mathbb{E}\|\nabla F(\theta_t)\|^2 \leq \frac{2(F(\theta_0) - F^\star)}{\eta T} + \eta L \sigma^2 / B = O(1/\sqrt{T}).$$

*Proof.* Let $g_t := \nabla F(\theta_t)$ and write $\hat g_t = g_t + \zeta_t$ with $\mathbb{E}[\zeta_t \mid \theta_t] = 0$ and $\mathbb{E}\|\zeta_t\|^2 \leq \sigma^2/B$. Apply the descent lemma of Chapter 7 (which holds for any $L$-smooth $F$ and any displacement $\theta_{t+1} - \theta_t = -\eta \hat g_t$):
$$F(\theta_{t+1}) \leq F(\theta_t) - \eta \langle g_t, \hat g_t\rangle + \tfrac{L \eta^2}{2}\|\hat g_t\|^2.$$
Take conditional expectation given $\theta_t$. Using $\mathbb{E}[\hat g_t \mid \theta_t] = g_t$ and $\mathbb{E}\|\hat g_t\|^2 = \|g_t\|^2 + \mathbb{E}\|\zeta_t\|^2$,
$$\mathbb{E}[F(\theta_{t+1}) \mid \theta_t] \leq F(\theta_t) - \eta\|g_t\|^2 + \tfrac{L\eta^2}{2}\big(\|g_t\|^2 + \sigma^2/B\big).$$
With $\eta \leq 1/L$ we have $\tfrac{L\eta^2}{2} \leq \eta/2$, so the coefficient of $\|g_t\|^2$ is $-\eta + \eta/2 = -\eta/2$:
$$\mathbb{E}[F(\theta_{t+1}) \mid \theta_t] \leq F(\theta_t) - \tfrac{\eta}{2}\|g_t\|^2 + \tfrac{L \eta^2 \sigma^2}{2B}.$$
Take the full expectation, sum from $t = 0$ to $T-1$, and telescope (consecutive $\mathbb{E} F(\theta_t)$ cancel):
$$\tfrac{\eta}{2} \sum_{t=0}^{T-1} \mathbb{E}\|g_t\|^2 \leq F(\theta_0) - \mathbb{E} F(\theta_T) + \tfrac{L \eta^2 \sigma^2 T}{2B} \leq F(\theta_0) - F^\star + \tfrac{L \eta^2 \sigma^2 T}{2B}.$$
Divide by $\eta T / 2$ to obtain
$$\frac{1}{T}\sum_{t=0}^{T-1}\mathbb{E}\|g_t\|^2 \leq \frac{2(F(\theta_0) - F^\star)}{\eta T} + \frac{L \eta \sigma^2}{B}.$$
Choose $\eta = c/\sqrt{T}$ (subject to $\eta \leq 1/L$). Both right-hand terms become $O(1/\sqrt{T})$, with $c = \sqrt{2(F(\theta_0) - F^\star) B / (L \sigma^2)}$ minimizing the bound. $\square$

The takeaway: the average squared gradient norm decays like $1/\sqrt{T}$ and the noise floor is proportional to $\eta \sigma^2 / B$. Larger batches let us use a larger $\eta$ at the same noise level.

**Theorem 13.3 (SGD on $L$-smooth, $\mu$-strongly convex $F$, sketch).** With diminishing step $\eta_t = \tfrac{2}{\mu(t + t_0)}$ for $t_0$ large enough that $\eta_0 \leq 1/L$,
$$\mathbb{E}\|\theta_T - \theta^\star\|^2 \leq \frac{C}{T}.$$

*Sketch.* Let $r_t^2 := \mathbb{E}\|\theta_t - \theta^\star\|^2$. Expand $\|\theta_{t+1} - \theta^\star\|^2 = \|\theta_t - \theta^\star - \eta_t \hat g_t\|^2$ and take expectation. Strong convexity gives the contraction $\langle g_t, \theta_t - \theta^\star\rangle \geq \mu \|\theta_t - \theta^\star\|^2$ (Chapter 7), and the noise contributes $\eta_t^2 \sigma^2/B$:
$$r_{t+1}^2 \leq (1 - \eta_t \mu)\, r_t^2 + \eta_t^2 \sigma^2 / B.$$
With $\eta_t \mu = 2/(t + t_0)$, an induction $r_t^2 \leq C/(t + t_0)$ closes for $C \geq 4\sigma^2/(\mu^2 B)$, giving $r_T^2 = O(1/T)$. $\square$

Strong convexity buys a $1/T$ rate (vs $1/\sqrt{T}$); strong convexity is rarely available for deep nets, so the non-convex bound is the operative one in practice.

**Robbins–Monro (1951).** For the iteration $\theta_{t+1} = \theta_t - \eta_t \hat g_t$ to converge almost surely to a stationary point under standard regularity, the step sizes must satisfy
$$\sum_{t=0}^{\infty} \eta_t = \infty, \qquad \sum_{t=0}^{\infty} \eta_t^2 < \infty.$$
Intuition: $\sum \eta_t = \infty$ guarantees we *travel far enough* to escape any bounded region; $\sum \eta_t^2 < \infty$ guarantees the *cumulative noise* $\sum \eta_t \zeta_t$ is summable in second moment and hence converges (martingale-convergence). The schedule $\eta_t = c/(t+1)$ satisfies both; $\eta_t = c/\sqrt{t+1}$ does not (square-summability fails), and a constant step satisfies neither.

## Code sketch

The notebook for this chapter (`cells.json`) contains: (i) a side-by-side run of full-batch GD versus SGD on a 1-D least-squares loss with $n = 1000$ Gaussian targets; (ii) an empirical check that the variance of $\hat g_B(0)$ scales as $1/B$ for $B \in \{1, 8, 64, 256\}$; (iii) SGD on the smooth non-convex toy $F(\theta) = \theta^2/2 + 0.5 \sin(5\theta)$ with the running average of $\|\nabla F(\theta_t)\|^2$ tracking $T^{-1/2}$; and (iv) a strongly-convex quadratic where a *constant* step plateaus at a noise floor while the diminishing schedule $\eta_t = c/(t+1)$ converges to the optimum.

## Connection to LLMs

Pre-training a language model is mini-batch SGD on the cross-entropy loss (Chapter 11) of next-token prediction. The "batch size" reported in scaling-law papers is exactly the $B$ of Theorem 13.1; doubling it halves the per-step gradient variance but doubles the FLOPs per step. The $O(1/\sqrt{T})$ rate of Theorem 13.2 is why pre-training, even on huge models, must consume *many* tokens — there is no $1/T$ shortcut without strong convexity. In practice we use AdamW (Chapter 14), which adds momentum and per-coordinate adaptive scaling on top of the SGD skeleton derived here; the convergence intuition (variance reduction by batching, noise floor proportional to $\eta \sigma^2/B$, $1/\sqrt{T}$ asymptotics) carries over essentially unchanged.
