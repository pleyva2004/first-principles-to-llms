## Motivation

Pre-training (Chapter 27) produces a model $\pi_{\mathrm{base}}$ whose only objective is to assign high probability to next-token sequences drawn from the training corpus. That model is fluent but neither helpful nor safe: prompted with "Q: What is the boiling point of water?" it is just as likely to continue with another question as to answer. **Post-training** is the suite of procedures that turns $\pi_{\mathrm{base}}$ into a model that *follows instructions* and *reflects human preferences*. We study the three dominant techniques: supervised fine-tuning (SFT), reinforcement learning from human feedback (RLHF) with PPO/GRPO, and direct preference optimization (DPO). The DPO derivation — solving the KL-regularized RLHF objective in closed form and inverting the Bradley–Terry model — is the centerpiece.

## Definitions

**Supervised fine-tuning (SFT).** Given a curated dataset $\mathcal{D}_{\mathrm{SFT}} = \{(x_i, y_i)\}$ of (prompt, response) pairs, continue training $\pi_{\mathrm{base}}$ with the same next-token-prediction loss as Chapter 25, but mask the prompt tokens so the loss only fires on the response:
$$\mathcal{L}_{\mathrm{SFT}}(\theta) = -\mathbb{E}_{(x,y)\sim\mathcal{D}_{\mathrm{SFT}}}\sum_{t=1}^{|y|}\log\pi_\theta(y_t\mid x, y_{<t}).$$
This is MLE (Chapter 12) on the response distribution and yields the *reference model* $\pi_{\mathrm{ref}} := \pi_{\mathrm{SFT}}$.

**Reward model (RM).** A scalar function $r_\phi : (x, y) \mapsto \mathbb{R}$ trained on a preference dataset $\mathcal{D}_{\mathrm{pref}} = \{(x, y_w, y_l)\}$ where $y_w$ is the human-preferred ("winner") response and $y_l$ the rejected ("loser"). The **Bradley–Terry** (1952) model posits
$$\mathbb{P}(y_w \succ y_l \mid x) = \sigma\big(r_\phi(x, y_w) - r_\phi(x, y_l)\big),$$
giving the convex loss $\ell_{\mathrm{RM}} = -\log\sigma(r_\phi(x, y_w) - r_\phi(x, y_l))$.

**RLHF objective.** Christiano et al. (2017); Stiennon et al. (2020); Ouyang et al. (2022). Find $\pi_\theta$ maximizing
$$\mathcal{J}(\theta) = \mathbb{E}_{x\sim\mathcal{D}}\,\mathbb{E}_{y\sim\pi_\theta(\cdot\mid x)}\big[r_\phi(x, y)\big] - \beta\, D_{\mathrm{KL}}\!\big(\pi_\theta(\cdot\mid x)\,\|\,\pi_{\mathrm{ref}}(\cdot\mid x)\big),$$
where $\beta>0$ controls the KL leash to $\pi_{\mathrm{ref}}$ (Chapter 11).

**PPO** (Schulman et al. 2017). Clipped surrogate
$$\mathcal{L}^{\mathrm{PPO}}(\theta) = \mathbb{E}\Big[\min\!\big(\rho_t(\theta)\hat A_t,\ \mathrm{clip}(\rho_t(\theta), 1-\varepsilon, 1+\varepsilon)\hat A_t\big)\Big],\qquad \rho_t(\theta)=\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\mathrm{old}}(a_t\mid s_t)}.$$

**GRPO** (DeepSeek 2024). Drop the value network. Sample $G$ responses $\{y^{(1)},\dots,y^{(G)}\}$ per prompt $x$, score each with $r_\phi$, and form group-normalized advantages
$$\hat A^{(i)} = \frac{r^{(i)} - \mu_g}{\sigma_g + \epsilon},\qquad \mu_g = \tfrac{1}{G}\sum_j r^{(j)},\ \sigma_g^2 = \tfrac{1}{G}\sum_j(r^{(j)}-\mu_g)^2.$$

**DPO** (Rafailov et al. 2023). The closed-form minimizer of the RLHF objective, expressed *directly* as a preference loss on $\pi_\theta$ — no separate $r_\phi$, no RL:
$$\mathcal{L}_{\mathrm{DPO}}(\theta) = -\mathbb{E}_{(x,y_w,y_l)}\!\left[\log\sigma\!\left(\beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\mathrm{ref}}(y_w\mid x)} - \beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\mathrm{ref}}(y_l\mid x)}\right)\right].$$

## Theorems

### Theorem 1 (DPO derivation — central result).
*Fix a reward $r$ and a reference policy $\pi_{\mathrm{ref}}$ with full support. The unique maximizer of
$\mathcal{J}(\pi) = \mathbb{E}_{y\sim\pi}[r(x,y)] - \beta\,D_{\mathrm{KL}}(\pi\|\pi_{\mathrm{ref}})$
over distributions $\pi(\cdot\mid x)$ is*
$$\pi^*(y\mid x) = \frac{1}{Z(x)}\,\pi_{\mathrm{ref}}(y\mid x)\,\exp\!\big(r(x,y)/\beta\big),\qquad Z(x) = \sum_{y}\pi_{\mathrm{ref}}(y\mid x)\exp(r(x,y)/\beta).$$
*Inverting and substituting into the Bradley–Terry preference likelihood collapses the partition function and yields the DPO loss.*

**Proof.** Write the per-prompt objective as
$$\mathcal{J}_x(\pi) = \sum_y \pi(y)r(x,y) - \beta\sum_y \pi(y)\log\frac{\pi(y)}{\pi_{\mathrm{ref}}(y)} = -\beta\sum_y \pi(y)\log\frac{\pi(y)}{\pi_{\mathrm{ref}}(y)\exp(r(x,y)/\beta)}.$$
Define the (un-normalized) measure $q(y) := \pi_{\mathrm{ref}}(y)\exp(r(x,y)/\beta)$ and let $Z(x):=\sum_y q(y)$. Then
$$\mathcal{J}_x(\pi) = -\beta\sum_y \pi(y)\log\frac{\pi(y)}{Z(x)^{-1}q(y)} + \beta\log Z(x) \cdot \!\!\underbrace{\sum_y\pi(y)}_{=1} = -\beta\,D_{\mathrm{KL}}\!\big(\pi\,\|\,Z^{-1}q\big) + \beta\log Z(x).$$
Since $D_{\mathrm{KL}}\geq 0$ with equality iff $\pi = Z^{-1}q$ (Gibbs, Chapter 11), the unique maximizer is $\pi^*(y\mid x) = Z(x)^{-1}\pi_{\mathrm{ref}}(y\mid x)\exp(r(x,y)/\beta)$.

Solving for $r$: $r(x,y) = \beta\log\dfrac{\pi^*(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)} + \beta\log Z(x)$. Plug this into the Bradley–Terry log-likelihood for a preference triple $(x, y_w, y_l)$:
$$\log\sigma\big(r(x,y_w) - r(x,y_l)\big) = \log\sigma\!\left(\beta\log\frac{\pi^*(y_w|x)}{\pi_{\mathrm{ref}}(y_w|x)} - \beta\log\frac{\pi^*(y_l|x)}{\pi_{\mathrm{ref}}(y_l|x)} + \beta\log Z(x) - \beta\log Z(x)\right).$$
The two $\beta\log Z(x)$ terms cancel — this is the crucial cancellation that eliminates the intractable partition function. Replacing $\pi^*$ with the parametric $\pi_\theta$ and negating the expectation gives $\mathcal{L}_{\mathrm{DPO}}(\theta)$. $\square$

### Theorem 2 (PPO trust-region motivation).
*Let $\pi_{\mathrm{old}}$ be the policy before update and $\hat A$ an advantage estimator. The unclipped surrogate $L(\theta) = \mathbb{E}[\rho_t(\theta)\hat A_t]$ is a first-order approximation to the policy improvement $J(\pi_\theta) - J(\pi_{\mathrm{old}})$ in a neighbourhood of $\pi_{\mathrm{old}}$ (Kakade & Langford 2002). The clip operator caps $|\rho_t - 1|\le\varepsilon$ on every step where the unclipped objective would be made larger by an out-of-trust-region update, yielding a pessimistic lower bound on the surrogate.*

*Sketch.* When $\hat A_t > 0$ and $\rho_t > 1+\varepsilon$, the $\min$ selects the clipped term, removing the incentive to push $\rho_t$ further. Symmetrically when $\hat A_t < 0$ and $\rho_t < 1-\varepsilon$. Schulman et al. (2017) show empirically that this enforces the TRPO trust-region constraint without a second-order solve. A full proof (Achiam et al. 2017) bounds total-variation distance between successive policies. $\square$

### Theorem 3 (GRPO advantage is unbiased).
*Let $r^{(1)},\dots,r^{(G)}$ be i.i.d. rewards under $\pi_{\mathrm{old}}(\cdot\mid x)$ and $\hat A^{(i)} = (r^{(i)} - \mu_g)/(\sigma_g + \epsilon)$. Subtracting the empirical mean $\mu_g$ does not bias the policy gradient.*

**Proof.** The REINFORCE estimator is $\hat g = \sum_i (r^{(i)} - b)\nabla_\theta\log\pi_\theta(y^{(i)}\mid x)$ for any baseline $b$ that does not depend on $y^{(i)}$. The well-known baseline lemma states
$$\mathbb{E}_{y\sim\pi}[b\,\nabla_\theta\log\pi(y)] = b\sum_y \nabla_\theta\pi(y) = b\,\nabla_\theta\!\sum_y\pi(y) = b\,\nabla_\theta 1 = 0.$$
Hence subtracting any data-independent constant from each $r^{(i)}$ leaves $\mathbb{E}[\hat g]$ unchanged. The empirical mean $\mu_g$ is *not* independent of $\{y^{(i)}\}$; correctness of GRPO uses leave-one-out independence: conditional on $y^{(j)}$, the average of the other $G-1$ samples is independent of $y^{(j)}$. Standard control-variate arguments show the bias is $O(1/G)$ and vanishes as $G\to\infty$. Dividing by $\sigma_g$ rescales the loss by a (data-dependent) constant, which is absorbed into the learning rate. $\square$

## Code sketch

We continue the tiny GPT of Chapter 27. The pipeline: (1) re-train briefly on a toy corpus to fix $\pi_{\mathrm{base}}$; (2) SFT on a handful of (prompt, response) pairs to obtain $\pi_{\mathrm{ref}}$; (3) train a Bradley–Terry reward head on $\sim 5$ preference triples and verify it ranks $y_w > y_l$; (4) compute one PPO clipped surrogate batch as a sanity check; (5) run DPO for a few hundred steps and verify the model's preference ratio $\pi_\theta(y_w\mid x)/\pi_\theta(y_l\mid x)$ has *increased* relative to $\pi_{\mathrm{ref}}$.

## Connection to LLMs

InstructGPT (Ouyang 2022) introduced the SFT $\to$ RM $\to$ PPO recipe; GPT-3.5 / GPT-4, Claude 1/2/3, Llama-2-Chat, and Llama-3-Instruct all use a variant. DeepSeek-R1 (2024) replaced PPO with GRPO, dropping the critic. Zephyr, Tülu, and most open-weight chat models published after late 2023 use DPO (or its variants IPO, KTO, ORPO) because the closed-form derivation eliminates reward-model training, sampling, and credit assignment over long sequences. The KL leash to $\pi_{\mathrm{ref}}$ is what prevents post-training from destroying the capabilities laid down during pre-training (Chapter 27): post-training *re-shapes* a small region of the policy manifold around the SFT initialization rather than learning a new model from scratch.
