## Motivation

Chapter 29 solved the MDP problem under two unrealistic assumptions: the state space is finite and small enough to enumerate (so $Q$ is a table), and the agent learns *on-policy* from full sweeps over $\mathcal{S}\times\mathcal{A}$. Real problems — Atari from raw pixels, continuous control from joint angles, language models with $|\mathcal{V}|^T$ trajectories — break both. We need (a) a *function approximator* $Q_\theta(s,a)$ that generalizes across states, and (b) a learning rule that works *off-policy*, reusing data collected under earlier behavior policies.

The naive combination of (a), (b), and the bootstrapping target of TD-learning is unstable: this is the **deadly triad** (Sutton & Barto, Ch 11). The Deep Q-Network (DQN; Mnih et al. 2015) cleared the triad with two simple tricks — a *replay buffer* and a *target network* — and won Atari. We then move to the **maximum-entropy** reformulation, where adding a bonus $\alpha H(\pi)$ to the reward turns the hard $\max$ into a soft log-sum-exp, gives a closed-form Boltzmann policy $\pi^\* \propto \exp(Q/\alpha)$, and is the *exact same* derivation that produces the RLHF/DPO closed form $\pi^\* \propto \pi_{\mathrm{ref}}\exp(r/\beta)$ used in Chapter 28 and central to Chapter 31.

## Definitions

**Function approximator.** Parameterize $Q_\theta(s,a):\mathcal{S}\times\mathcal{A}\to\mathbb{R}$ by $\theta\in\mathbb{R}^d$. Linear: $Q_\theta(s,a)=\phi(s,a)^\top\theta$ for a feature map $\phi$. Neural: a multi-layer perceptron (Ch 15) with input $s$ and $|\mathcal{A}|$ outputs, trained by backprop (Ch 18).

**Mean-squared TD error (the DQN loss).** With a transition $(s,a,r,s')$,
$$\mathcal{L}(\theta)=\mathbb{E}_{(s,a,r,s')\sim\mathcal{D}}\Big[\big(\,r+\gamma\max_{a'}Q_{\theta^-}(s',a')\,-\,Q_\theta(s,a)\big)^2\Big],$$
where $\theta^-$ is the **target network** — a frozen copy of $\theta$ — and $\mathcal{D}$ is the **replay buffer**: a FIFO of past transitions sampled iid for the gradient step.

**Target network update.** Either *hard* refresh $\theta^-\leftarrow\theta$ every $K$ gradient steps, or *Polyak* averaging $\theta^-\leftarrow\tau\theta+(1-\tau)\theta^-$ with $\tau\ll 1$.

**Maximum-entropy RL objective.** For temperature $\alpha>0$,
$$J_\alpha(\pi)=\mathbb{E}_\pi\!\left[\sum_{t=0}^{\infty}\gamma^t\big(r_t+\alpha\,H(\pi(\cdot\mid s_t))\big)\right],\qquad H(p)=-\sum_a p(a)\log p(a).$$

**Soft value functions.** $V^\pi_{\mathrm{soft}}(s)=\mathbb{E}_\pi[r+\gamma V^\pi_{\mathrm{soft}}(s')+\alpha H(\pi(\cdot\mid s))]$ and $Q^\pi_{\mathrm{soft}}(s,a)=r(s,a)+\gamma\,\mathbb{E}_{s'}[V^\pi_{\mathrm{soft}}(s')]$.

**Soft Bellman optimality operator.**
$$(\mathcal{T}^\*_\alpha Q)(s,a)=r(s,a)+\gamma\,\mathbb{E}_{s'}\Big[\,\alpha\log\!\sum_{a'}\exp\!\big(Q(s',a')/\alpha\big)\Big].$$

## Theorems

### Theorem 1 (The deadly triad; Baird 1995).
*Off-policy learning + bootstrapping + linear function approximation can diverge.*

**Baird's 7-state counterexample.** Seven states $\{1,\dots,7\}$ and 8-dimensional features $\phi(s)\in\mathbb{R}^8$:
$$\phi(s) = 2 e_s + e_8 \ \ (s=1..6),\qquad \phi(7) = e_7 + 2 e_8.$$
Two actions: *solid* deterministically goes to state 7; *dashed* goes uniformly to one of states $1..6$. All rewards are $0$ and $\gamma=0.99$. The target policy always picks *solid* ($\pi(\text{solid}\mid s)=1$); the behavior policy picks *solid* with prob $1/7$. The off-policy importance ratio is $\rho=7$ on solid steps and $0$ on dashed steps. Initialize $w=(1,1,1,1,1,1,10,1)^\top$. The semi-gradient TD(0) update
$$w\leftarrow w+\alpha\,\rho\,\big(0+\gamma\,\phi(7)^\top w-\phi(s)^\top w\big)\,\phi(s)$$
satisfies $\|w_t\|\to\infty$.

*Why it diverges (sketch).* Under the behavior state distribution $d_b$, the expected semi-gradient update is $w_{t+1}=w_t+\alpha\,A\,w_t$ with
$$A = \mathbb{E}_{s\sim d_b}\big[\rho(s)\,\phi(s)\,(\gamma\phi(7)-\phi(s))^\top\big].$$
Stability requires every eigenvalue of $A$ to lie in the open left half-plane. In Baird's setup, only solid transitions contribute (dashed has $\rho=0$); at every state $s\in\{1..7\}$ visited by behavior, this contributes $\phi(s)(\gamma\phi(7)-\phi(s))^\top$ weighted by $1/7\cdot 7 = 1$. The eighth (shared) feature $e_8$ creates a constructive interference: $\gamma\phi(7)-\phi(s) = -2e_s+\gamma e_7+(2\gamma-1)e_8$, and $(2\gamma-1)>0$ for $\gamma>1/2$. The outer product across states accumulates a positive eigenvalue along the $e_8$ direction, so $\|w\|$ grows geometrically.

*The role of each leg.* Without **function approximation** (tabular: $\phi(s)=e_s$), $A$ collapses to a diagonal-dominant negative-definite matrix and TD(0) converges. Without **bootstrapping** (Monte Carlo: targets are full returns, no $\gamma\phi(7)^\top w$ term), the loss is convex regression and SGD converges. Without **off-policy** ($\rho\equiv 1$), Tsitsiklis & Van Roy (1997) show the on-policy projection $\Pi\mathcal{T}^\pi$ is a $\gamma$-contraction in the $d_\pi$-weighted norm, so TD(0) converges to the projected fixed point. Removing any one leg restores stability; keeping all three permits divergence. The numerical experiment in cell 4 reproduces the explosion: $\|w\|$ grows from $\approx 10$ to $>300$ in 1000 steps.

### Theorem 2 (DQN's two tricks rescue stability — informal).
**Replay buffer.** SGD's stochastic-approximation theorem (Ch 13) requires the gradients to be approximately iid (or at least a martingale-difference sequence). Sequential MDP samples $(s_t,a_t,r_t,s_{t+1})$ are highly correlated: $s_{t+1}$ is one step from $s_t$. The buffer $\mathcal{D}$ stores $\sim 10^5$–$10^6$ past transitions; sampling minibatches uniformly at random (a) decorrelates the gradient signal, (b) reuses each transition many times (sample efficiency), and (c) makes the data distribution close to stationary so the SGD analysis applies.

**Target network.** With a single network the bootstrap target $r+\gamma\max_{a'}Q_\theta(s',a')$ shifts every gradient step, which couples $Q_\theta(s,a)$ to itself and produces oscillation reminiscent of self-referential ODE instability. Freezing $\theta^-$ for $K$ steps decouples target from prediction; the loss $\mathcal{L}(\theta)$ is then a *standard* regression problem with stationary targets, which Adam/AdamW (Ch 14) handles. Mnih et al. 2015 show both tricks are necessary in ablations.

### Theorem 3 (Max-entropy RL = constrained optimization; closed-form policy).
*For fixed state $s$ and fixed $Q(s,\cdot)$, the maximizer of $\sum_a \pi(a)Q(s,a)+\alpha H(\pi)$ subject to $\sum_a\pi(a)=1$, $\pi(a)\ge 0$, is*
$$\pi^\*(a\mid s)=\frac{\exp(Q(s,a)/\alpha)}{\sum_{a'}\exp(Q(s,a')/\alpha)},\qquad \text{value }=\alpha\log\sum_a\exp(Q(s,a)/\alpha).$$

**Proof.** Lagrangian $\mathcal{L}=\sum_a\pi(a)Q(s,a)-\alpha\sum_a\pi(a)\log\pi(a)-\lambda(\sum_a\pi(a)-1)$. Stationarity: $\partial\mathcal{L}/\partial\pi(a)=Q(s,a)-\alpha(\log\pi(a)+1)-\lambda=0$, so $\pi(a)=\exp((Q(s,a)-\lambda-\alpha)/\alpha)$. Imposing $\sum_a\pi(a)=1$ pins $\lambda$ and yields the softmax. Substituting back, $\sum_a\pi^\*(a)Q(s,a)+\alpha H(\pi^\*)=\alpha\log\sum_a\exp(Q(s,a)/\alpha)$. The objective is strictly concave in $\pi$ (entropy is strictly concave; linear term is linear), so the KKT point is the unique global maximum; positivity is automatic from the exponential. $\square$

This is *exactly* the derivation that gives RLHF its closed-form solution (Ch 28): replace $H(\pi)$ by $-D_{\mathrm{KL}}(\pi\|\pi_{\mathrm{ref}})$ and the same Lagrangian gives $\pi^\*\propto\pi_{\mathrm{ref}}\exp(r/\beta)$.

### Theorem 4 (Soft Bellman is a $\gamma$-contraction).
*$\mathcal{T}^\*_\alpha$ is a $\gamma$-contraction on $(\mathbb{R}^{|\mathcal{S}|\times|\mathcal{A}|},\|\cdot\|_\infty)$, hence has a unique fixed point $Q^\*_{\mathrm{soft}}$ which value iteration $Q_{k+1}=\mathcal{T}^\*_\alpha Q_k$ reaches at rate $\gamma^k$.*

**Proof.** The soft-max $L_\alpha(x):=\alpha\log\sum_i\exp(x_i/\alpha)$ is 1-Lipschitz in $\|\cdot\|_\infty$. To see this, compute $\nabla L_\alpha(x) = \mathrm{softmax}(x/\alpha) =: p$, a probability vector (non-negative, sums to 1, $\|p\|_1=1$). By the mean-value theorem, $L_\alpha(x)-L_\alpha(y)=p_\xi^\top(x-y)$ for some $\xi$ on the segment between $x$ and $y$, and Hölder gives $|p_\xi^\top(x-y)|\le\|p_\xi\|_1\|x-y\|_\infty=\|x-y\|_\infty$. Now for any $Q_1,Q_2$ and any $(s,a)$,
$$|\mathcal{T}^\*_\alpha Q_1(s,a)-\mathcal{T}^\*_\alpha Q_2(s,a)| = \gamma\,\Big|\mathbb{E}_{s'}[L_\alpha(Q_1(s',\cdot))-L_\alpha(Q_2(s',\cdot))]\Big| \le \gamma\,\mathbb{E}_{s'}\|Q_1(s',\cdot)-Q_2(s',\cdot)\|_\infty\le\gamma\|Q_1-Q_2\|_\infty.$$
Taking sup over $(s,a)$ yields $\|\mathcal{T}^\*_\alpha Q_1-\mathcal{T}^\*_\alpha Q_2\|_\infty\le\gamma\|Q_1-Q_2\|_\infty$. Banach's fixed-point theorem gives existence, uniqueness of $Q^\*_{\mathrm{soft}}$, and geometric convergence $\|Q_k-Q^\*_{\mathrm{soft}}\|_\infty\le\gamma^k\|Q_0-Q^\*_{\mathrm{soft}}\|_\infty$. $\square$

### Theorem 5 (Soft $\to$ hard as $\alpha\to 0$).
*$\alpha\log\sum_a\exp(Q(s,a)/\alpha)\to\max_a Q(s,a)$ pointwise as $\alpha\to 0^+$, and the soft policy $\pi^\*_\alpha$ collapses to the greedy policy $\arg\max_a Q(s,a)$ (uniform over the argmax set).*

**Proof.** Let $M=\max_a Q(s,a)$, $\mathcal{A}^\*=\{a:Q(s,a)=M\}$, $k=|\mathcal{A}^\*|$. Write $\alpha\log\sum_a\exp(Q(s,a)/\alpha)=M+\alpha\log\!\big(k+\sum_{a\notin\mathcal{A}^\*}\exp((Q(s,a)-M)/\alpha)\big)$. Each suboptimal exponent is $\le 0$ with strict inequality, so its exponential vanishes as $\alpha\to 0$, leaving $M+\alpha\log k\to M$. The softmax mass on $\mathcal{A}^\*$ tends to $1/k$ each and to $0$ on suboptimal actions. This is the Ch 16 softmax-temperature limit applied to $Q/\alpha$. $\square$

## Code sketch

The notebook implements: (1) linear TD(0) on a 5-state chain, recovering the closed form; (2) Baird's counterexample, with $\|w\|$ exploding; (3) numpy DQN on a hand-coded CartPole, showing average episode length grows from $\sim 22$ to $> 100$; (4) soft Q-learning on a 4×4 grid for $\alpha\in\{0.01,0.5,5\}$, visualizing entropy growth and verifying $Q_{\mathrm{soft}}\to Q^\*$ as $\alpha\to 0$; (5) the soft Bellman fixed point on a 3-state MDP, with the residual zero, alongside a print-out of the analogous RLHF closed form.

## Connection to LLMs / RLHF

The max-entropy framework is the bridge to alignment. Compare the two per-state objectives side-by-side:

- **Soft RL (this chapter):** $\max_\pi \sum_a\pi(a)Q(s,a)+\alpha H(\pi)$ $\;\Rightarrow\;\pi^\*(a\mid s)\propto\exp(Q(s,a)/\alpha)$.
- **RLHF (Ch 28):** $\max_\pi \mathbb{E}_{y\sim\pi}[r(x,y)]-\beta D_{\mathrm{KL}}(\pi\|\pi_{\mathrm{ref}})$ $\;\Rightarrow\;\pi^\*(y\mid x)\propto\pi_{\mathrm{ref}}(y\mid x)\exp(r(x,y)/\beta)$.

The two derivations are *the same Lagrangian*: the only change is replacing the entropy bonus $\alpha H(\pi) = -\alpha\sum_a\pi(a)\log\pi(a)$ with the KL leash $-\beta D_{\mathrm{KL}}(\pi\|\pi_{\mathrm{ref}}) = -\beta\sum_a\pi(a)\log(\pi(a)/\pi_{\mathrm{ref}}(a))$, which adds a constant shift inside the log. Stationarity of the Lagrangian then yields $\pi(a)\propto\pi_{\mathrm{ref}}(a)\exp(Q(s,a)/\beta)$ — the RLHF closed form. DPO (Ch 28) inverts this expression, expresses the implicit reward in terms of $\log\pi_\theta/\pi_{\mathrm{ref}}$, and substitutes into the Bradley–Terry preference likelihood; the partition function $Z(x)$ cancels.

Architecturally, DQN's target network plays the same role as RLHF's $\pi_{\mathrm{ref}}$: a slowly-moving anchor that prevents the bootstrap (or the KL leash) from chasing its own tail. In DQN the anchor is refreshed every $K$ gradient steps; in RLHF it is held fixed for the entire fine-tuning run. In both cases the anchor turns a non-stationary self-referential objective into a sequence of stationary regression problems that AdamW (Ch 14) can reliably minimize. Chapter 31 closes the loop, fusing policy gradient (REINFORCE/GRPO) with the soft-RL view to derive the alignment loss actually shipped in modern LLMs.
