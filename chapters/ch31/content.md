## Motivation

Chapters 29 and 30 set up the MDP machinery and the maximum-entropy framework; Chapter 28 gave the closed-form DPO solution to the KL-regularized RLHF objective. What is still missing is the *online* alignment loop: an algorithm that takes a pre-trained language model (Chapter 27), a reward signal (human preference, RM, or a verifier), and produces a *better* policy by gradient ascent on $\mathbb{E}_{y \sim \pi_\theta}[r(x, y)]$. This chapter derives that loop from first principles. We start with the **policy gradient theorem** (Sutton 1999), reduce variance with **baselines** and **GAE**, enforce a trust region with **PPO**, drop the value function with **GRPO** (DeepSeek 2024), and end by running GRPO on the Chapter 27 tiny GPT — closing the 31-chapter loop with a measurable improvement in the reward of generated text. RLHF / DPO / GRPO are then revealed as three coordinates on the same KL-regularized policy-improvement landscape.

## Definitions

A **parameterized stochastic policy** is $\pi_\theta(a \mid s)$, a conditional distribution over actions whose parameters $\theta$ we will optimize. A **trajectory** is $\tau = (s_0, a_0, r_0, s_1, a_1, \ldots, s_{T-1}, a_{T-1}, r_{T-1}, s_T)$ with **return** $R(\tau) = \sum_{t=0}^{T-1} \gamma^t r_t$. Under the MDP $(S, A, P, R, p_0, \gamma)$ of Chapter 29, the **trajectory distribution** factorizes as
$$p_\theta(\tau) \;=\; p_0(s_0) \prod_{t=0}^{T-1} \pi_\theta(a_t \mid s_t)\, P(s_{t+1} \mid s_t, a_t).$$
The **objective** is $J(\theta) = \mathbb{E}_{\tau \sim p_\theta}[R(\tau)]$. The **state distribution** $d^{\pi}(s) = (1-\gamma) \sum_{t=0}^{\infty} \gamma^t \Pr_\pi[s_t = s]$ and **value functions** $V^\pi, Q^\pi$ are inherited from Chapter 29; the **advantage** is $A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)$.

The **score function** (a.k.a. likelihood-ratio identity) is the backbone of every algorithm in this chapter:
$$\nabla_\theta \log p_\theta(\tau) \;=\; \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t \mid s_t).$$
The dynamics $P$ and the initial distribution $p_0$ are $\theta$-independent, so their logs vanish under $\nabla_\theta$. This is the algebraic miracle that lets us optimize in *unknown* environments.

**Generalized Advantage Estimation** (GAE; Schulman 2016) interpolates between Monte Carlo and one-step TD with $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$:
$$\hat{A}_t^{\mathrm{GAE}(\gamma, \lambda)} \;=\; \sum_{l=0}^\infty (\gamma \lambda)^l \delta_{t+l}, \qquad \lambda \in [0, 1].$$
The **TRPO surrogate** (Schulman 2015) is $\mathcal{L}^{\mathrm{TRPO}}(\theta) = \mathbb{E}\!\left[\frac{\pi_\theta(a \mid s)}{\pi_{\mathrm{old}}(a \mid s)}\, \hat{A}^{\pi_{\mathrm{old}}}(s, a)\right]$ subject to a KL trust region $\mathbb{E}_s[D_{\mathrm{KL}}(\pi_{\mathrm{old}}(\cdot \mid s) \,\|\, \pi_\theta(\cdot \mid s))] \leq \delta$. The **PPO clipped surrogate** (Schulman 2017) replaces the constraint with a clipped objective using the per-step ratio $r_t(\theta) = \pi_\theta(a_t \mid s_t)/\pi_{\mathrm{old}}(a_t \mid s_t)$:
$$\mathcal{L}^{\mathrm{PPO}}(\theta) \;=\; \mathbb{E}\!\left[\min\bigl(r_t(\theta)\, \hat{A}_t,\; \mathrm{clip}(r_t(\theta), 1-\varepsilon, 1+\varepsilon)\, \hat{A}_t\bigr)\right].$$

**GRPO** (DeepSeek-Math, 2024) eliminates the value function. For each prompt $x$, sample a *group* of $G$ completions $\{y_1, \ldots, y_G\}$ from $\pi_{\mathrm{old}}(\cdot \mid x)$, score them with rewards $r_1, \ldots, r_G$, and use the **group-relative advantage**
$$\hat{A}_i \;=\; \frac{r_i - \mathrm{mean}(r_{1:G})}{\mathrm{std}(r_{1:G}) + \epsilon},$$
broadcast to every token of completion $i$. The PPO-clipped surrogate is then applied per token, with a KL penalty $\beta\, \mathbb{E}\,D_{\mathrm{KL}}(\pi_\theta \,\|\, \pi_{\mathrm{ref}})$ to prevent drift from the SFT reference (Chapter 28).

## Theorems

### Theorem (Policy Gradient Theorem; Sutton 1999)

Under regularity conditions making swap of $\nabla_\theta$ and $\int$ valid,
$$\nabla_\theta J(\theta) \;=\; \mathbb{E}_{\tau \sim p_\theta}\!\left[\sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t \mid s_t)\, Q^{\pi_\theta}(s_t, a_t)\right] \;=\; \frac{1}{1-\gamma}\, \mathbb{E}_{(s,a) \sim d^{\pi_\theta}}\!\left[\nabla_\theta \log \pi_\theta(a \mid s)\, Q^{\pi_\theta}(s, a)\right].$$

*Proof.* Start with $J(\theta) = \int p_\theta(\tau)\, R(\tau)\, d\tau$ and apply the log-derivative trick $\nabla p_\theta = p_\theta \nabla \log p_\theta$:
$$\nabla_\theta J(\theta) \;=\; \int \nabla_\theta p_\theta(\tau)\, R(\tau)\, d\tau \;=\; \int p_\theta(\tau)\, \nabla_\theta \log p_\theta(\tau)\, R(\tau)\, d\tau \;=\; \mathbb{E}_\tau[\nabla_\theta \log p_\theta(\tau)\, R(\tau)].$$
Substitute the factorized score: $\nabla_\theta \log p_\theta(\tau) = \sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t)$ since $p_0$ and $P$ do not depend on $\theta$. Now use the *causality* of rewards: under the trajectory distribution, $r_{t'}$ for $t' < t$ is uncorrelated with $a_t$ given the history, so $\mathbb{E}[\nabla \log \pi(a_t \mid s_t)\, r_{t'}] = 0$ for $t' < t$. Therefore
$$\nabla_\theta J(\theta) \;=\; \mathbb{E}\!\left[\sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t) \sum_{t' \geq t} \gamma^{t'} r_{t'}\right] \;=\; \mathbb{E}\!\left[\sum_t \gamma^t \nabla_\theta \log \pi_\theta(a_t \mid s_t)\, Q^\pi(s_t, a_t)\right],$$
where we recognized the discounted future return as $\gamma^t Q^\pi(s_t, a_t)$ in expectation. Folding $\gamma^t$ into $d^\pi$ gives the second form. $\blacksquare$

### Lemma (Baselines do not bias the gradient)

For any function $b : S \to \mathbb{R}$ depending only on the state,
$$\mathbb{E}_{a \sim \pi_\theta(\cdot \mid s)}\!\left[\nabla_\theta \log \pi_\theta(a \mid s)\, b(s)\right] \;=\; 0.$$

*Proof.* Pull $b(s)$ outside the action expectation and use the *log-trick in reverse*:
$$\sum_a \pi_\theta(a \mid s) \nabla_\theta \log \pi_\theta(a \mid s) \;=\; \sum_a \nabla_\theta \pi_\theta(a \mid s) \;=\; \nabla_\theta \sum_a \pi_\theta(a \mid s) \;=\; \nabla_\theta\, 1 \;=\; 0. \;\blacksquare$$

Subtracting $V^\pi(s)$ from $Q^\pi(s,a)$ (the optimal variance-reducing baseline up to a constant) yields the **advantage form** $\nabla_\theta J = \mathbb{E}[\nabla_\theta \log \pi_\theta(a \mid s)\, A^\pi(s, a)]$.

### Lemma (GAE recursion; bias–variance)

The estimator $\hat{A}_t^{\mathrm{GAE}(\gamma, \lambda)}$ satisfies $\hat{A}_t = \delta_t + \gamma \lambda\, \hat{A}_{t+1}$. With $\lambda = 0$, $\hat{A}_t = \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$, the one-step TD residual: low variance, biased by the bootstrap term $V(s_{t+1})$. With $\lambda = 1$, telescoping gives $\hat{A}_t = \sum_{l \geq 0} \gamma^l r_{t+l} - V(s_t)$, the Monte Carlo return minus a baseline: unbiased but high variance. The recursion is immediate from $\sum_{l \geq 0}(\gamma\lambda)^l \delta_{t+l} = \delta_t + \gamma\lambda \sum_{l \geq 0}(\gamma\lambda)^l \delta_{t+1+l}$.

### Theorem (TRPO surrogate-improvement inequality; Schulman 2015)

For any two policies $\pi, \pi'$,
$$J(\pi') \;\geq\; J(\pi) + \mathbb{E}_{(s,a) \sim d^\pi, \pi'(\cdot \mid s)}\!\left[A^\pi(s, a)\right] \;-\; \frac{2 \varepsilon \gamma}{(1 - \gamma)^2}\, D_{\mathrm{KL}}^{\max}(\pi, \pi'),$$
with $\varepsilon = \max_{s,a}|A^\pi(s,a)|$ and $D_{\mathrm{KL}}^{\max} = \max_s D_{\mathrm{KL}}(\pi(\cdot \mid s) \| \pi'(\cdot \mid s))$.

*Proof sketch.* The exact **performance-difference lemma** of Kakade–Langford reads $J(\pi') - J(\pi) = \frac{1}{1-\gamma}\,\mathbb{E}_{(s,a) \sim d^{\pi'}, \pi'}[A^\pi(s, a)]$. The first term on the RHS of the inequality replaces $d^{\pi'}$ with $d^\pi$ (the *surrogate*), introducing an error bounded by the total-variation distance between $d^\pi$ and $d^{\pi'}$. Pinsker's inequality (Chapter 11) bounds this TV distance by $\sqrt{\tfrac12 D_{\mathrm{KL}}}$, and a discounted-chain argument gives the $(1 - \gamma)^{-2}$ factor. The full proof is in Kakade & Langford (2002) and Schulman (2015). $\blacksquare$

This is *the* mathematical reason a KL leash on policy updates is principled, not merely heuristic.

### Proposition (PPO is a first-order trust region)

Around $\theta_{\mathrm{old}}$, $r_t(\theta) = 1 + \nabla_\theta \log \pi_\theta(a_t|s_t)\big|_{\theta_{\mathrm{old}}}^{\top}(\theta - \theta_{\mathrm{old}}) + O(\|\theta - \theta_{\mathrm{old}}\|^2)$. The clip $r_t \in [1-\varepsilon, 1+\varepsilon]$ therefore corresponds to a per-sample bound on the directional derivative — an *axis-aligned, first-order* surrogate for the global KL ball that TRPO enforces with a Fisher-vector product. Empirically the clip is loose enough to be cheap and tight enough to mimic the trust region.

### Theorem (GRPO group baseline does not bias the gradient, in the limit)

Sample $G$ i.i.d. completions $y_i \sim \pi_{\mathrm{old}}(\cdot \mid x)$, set $\bar r = \frac{1}{G}\sum_j r(x, y_j)$ and $\hat A_i = (r_i - \bar r) / (\hat\sigma + \epsilon)$. As $G \to \infty$, $\bar r \to \mu(x) := \mathbb{E}_{y \sim \pi_{\mathrm{old}}}[r(x, y)]$ and $\hat\sigma \to \sigma(x)$, both *independent of any single* $y_i$. Then $\mathbb{E}_{y_i}[\hat A_i \nabla \log \pi_\theta(y_i \mid x)] = \frac{1}{\sigma(x)}\bigl(\mathbb{E}[r_i \nabla \log \pi_\theta] - \mu(x)\,\mathbb{E}[\nabla \log \pi_\theta]\bigr)$. The second term is $\mu(x) \cdot 0 = 0$ by Lemma 2; the first equals $\sigma(x)^{-1}\nabla J(x)$. Hence GRPO is unbiased up to the (state-dependent) scaling $1/\sigma(x)$.

For finite $G$, $\bar r$ and $\hat\sigma$ depend on $y_i$ (via the $i$-th term), so a small $O(1/G)$ bias appears. DeepSeek (2024) verifies empirically that this bias is negligible for $G \geq 4$ and that the variance reduction more than compensates. The dramatic engineering payoff: **no value network**.

### Theorem (DPO–RLHF equivalence; recap of Chapter 28)

The KL-regularized RLHF objective $\max_\pi \mathbb{E}_{x, y \sim \pi(\cdot \mid x)}[r(x, y)] - \beta D_{\mathrm{KL}}(\pi \| \pi_{\mathrm{ref}})$ has closed-form maximizer $\pi^*(y \mid x) = \pi_{\mathrm{ref}}(y \mid x) \exp(r(x, y)/\beta) / Z(x)$. Solving for $r$ gives $r(x, y) = \beta \log\frac{\pi^*(y \mid x)}{\pi_{\mathrm{ref}}(y \mid x)} + \beta \log Z(x)$. Substituting into the Bradley–Terry preference likelihood, the partition function $Z(x)$ cancels in the difference $r(x, y_w) - r(x, y_l)$, yielding the DPO loss directly on $\pi_\theta$ with no reward model. (Full derivation in Chapter 28.)

## Code sketch

The notebook walks the algorithmic ladder: **REINFORCE** on a 3-state MDP (the bare policy gradient with Monte Carlo returns); **REINFORCE + baseline** showing the predicted variance drop; **GAE** sweeping $\lambda \in \{0, 0.5, 0.95, 1\}$ on a 5-state chain; a **PPO** loop with a 2-layer MLP policy on a CartPole-style env from Chapter 30; a **GRPO** warm-up on a synthetic 5-arm contextual bandit; and the climax — **GRPO on the Chapter 27 tiny GPT**, with a verifiable reward that fires when the generated continuation contains a target substring. The KL to $\pi_{\mathrm{ref}}$ is monitored each step to confirm the policy improves the reward without collapsing.

## Connection to LLMs

Modern post-training is a small constellation of three techniques sitting on the same KL-regularized landscape:

- **RLHF (PPO + RM).** Train an SFT model, train a Bradley–Terry reward model on preferences, then run PPO with the RM as the dense reward and a KL penalty to the SFT reference. Used by InstructGPT, GPT-3.5 / GPT-4, Claude, Llama-2-Chat. Pros: handles arbitrary reward signals, including learned ones. Cons: four models in memory (policy, value, reward, reference), reward hacking, brittle infra.
- **DPO.** Skip the RM entirely; use the closed-form solution of the KL-regularized objective to derive a direct preference loss on $\pi_\theta$ vs $\pi_{\mathrm{ref}}$. Used by Zephyr, Tülu, most open-weight chat models since late 2023. Pros: one model, one loss, no sampling at train time. Cons: limited to *pairwise* preferences; bounded by the diversity of the offline dataset.
- **GRPO.** PPO without the value network. Sample $G$ completions per prompt, baseline with the group mean, normalize by group std. Used by DeepSeek-Math and DeepSeek-R1 to push state-of-the-art reasoning. Pros: scales effortlessly to *verifiable* rewards (math correctness, code execution, format compliance) where an RM would itself be a bottleneck. Cons: needs $G$ samples per prompt at training time (compute trade-off vs the value network).

A fellowship-level mental model: all three are doing surrogate optimization of $J(\theta) - \beta D_{\mathrm{KL}}(\pi_\theta \| \pi_{\mathrm{ref}})$. RLHF estimates the gradient with PPO + a learned value function; DPO eliminates online sampling by inverting the closed-form solution; GRPO eliminates the value function by group-baselining. The pre-trained model from Chapter 27 is the floor — it sets *what is sayable* — and these chapters reshape only a small region around it. Pre-training is calorie intake; alignment is digestion. Everything in this 31-chapter chain composes to produce that final sentence.
