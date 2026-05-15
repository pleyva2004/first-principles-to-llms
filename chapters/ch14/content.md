## Motivation

Chapter 13 established that stochastic gradient descent (SGD) converges, but its rate is governed by the condition number $\kappa = L/\mu$ of the loss landscape. For deep networks --- and for transformer language models in particular --- $\kappa$ can be astronomical, with curvature varying by many orders of magnitude across coordinates (e.g., embedding rows touched once per epoch versus dense matrices updated every step). Plain SGD wastes most of its updates fighting oscillation in stiff directions while crawling along flat ones. Chapter 14 builds the optimizers that fix both pathologies: **momentum** smooths the trajectory, **RMSProp** rescales per coordinate, **Adam** combines both with a bias correction, and **AdamW** decouples weight decay so it survives Adam's rescaling. AdamW is the optimizer behind essentially every transformer pre-training run from GPT-2 onward (forward reference: Chapter 27).

## Definitions

\begin{definition}[Polyak's heavy-ball momentum]
Given gradient estimate $g_t = \nabla f(\theta_t; \xi_t)$ and momentum coefficient $\beta \in [0,1)$, the heavy-ball update is
$$v_{t+1} = \beta v_t + g_t, \qquad \theta_{t+1} = \theta_t - \eta\, v_{t+1},$$
with $v_0 = 0$. The state $v_t$ accumulates an exponentially weighted history of past gradients.
\end{definition}

\begin{definition}[Nesterov accelerated gradient]
NAG evaluates the gradient at the *look-ahead* point $\theta_t - \eta\beta v_t$ rather than at $\theta_t$. Empirically it improves the constant factor on convex problems; in deep learning it is usually a minor variation of heavy-ball.
\end{definition}

\begin{definition}[RMSProp]
With $\beta_2 \in [0,1)$ and small $\varepsilon > 0$,
$$v_t = \beta_2 v_{t-1} + (1-\beta_2)\, g_t^{\odot 2}, \qquad \theta_{t+1} = \theta_t - \eta\, g_t \oslash (\sqrt{v_t} + \varepsilon),$$
where $\odot$ and $\oslash$ are element-wise. RMSProp normalizes each coordinate by an EMA of its squared gradient.
\end{definition}

\begin{definition}[Adam]
Adam couples a first-moment EMA (momentum) with a second-moment EMA (RMSProp), correcting both for initialization bias:
\begin{align*}
m_t &= \beta_1 m_{t-1} + (1-\beta_1) g_t, \\
v_t &= \beta_2 v_{t-1} + (1-\beta_2) g_t^{\odot 2}, \\
\hat m_t &= m_t/(1-\beta_1^t), \quad \hat v_t = v_t/(1-\beta_2^t), \\
\theta_{t+1} &= \theta_t - \eta\, \hat m_t \oslash (\sqrt{\hat v_t} + \varepsilon).
\end{align*}
Defaults: $\beta_1 = 0.9, \beta_2 = 0.999, \varepsilon = 10^{-8}$.
\end{definition}

\begin{definition}[AdamW: decoupled weight decay]
$$\theta_{t+1} = \theta_t - \eta\!\left(\hat m_t \oslash (\sqrt{\hat v_t} + \varepsilon) + \lambda\, \theta_t\right).$$
The decay term $\lambda\theta_t$ is applied directly to the parameters and is *not* funneled through the $\hat v_t$ rescaling.
\end{definition}

## Theorems

\begin{lemma}[EMA closed form]
With $m_0 = 0$, $m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$ unrolls to
$$m_t = (1 - \beta_1)\sum_{i=1}^{t} \beta_1^{\,t-i}\, g_i.$$
\end{lemma}
\begin{proof}
Induction on $t$. Base $t=1$: $m_1 = (1-\beta_1)g_1$. Step: assume the formula for $t-1$; then
$m_t = \beta_1\bigl[(1-\beta_1)\sum_{i=1}^{t-1}\beta_1^{t-1-i}g_i\bigr] + (1-\beta_1)g_t = (1-\beta_1)\bigl[\sum_{i=1}^{t-1}\beta_1^{t-i}g_i + g_t\bigr]$, which matches.
\end{proof}

\begin{theorem}[Bias correction for Adam, first moment]
Assume $g_t$ is stationary with $\mathbb{E}[g_t] = g$. Then $\mathbb{E}[m_t] = (1 - \beta_1^t)\, g$, and consequently $\mathbb{E}[\hat m_t] = g$ for every $t \geq 1$.
\end{theorem}
\begin{proof}
By the lemma and linearity of expectation,
$\mathbb{E}[m_t] = (1-\beta_1)\sum_{i=1}^t \beta_1^{t-i}\, \mathbb{E}[g_i] = (1-\beta_1) g \sum_{j=0}^{t-1}\beta_1^j = (1-\beta_1) g\, \frac{1 - \beta_1^t}{1 - \beta_1} = (1 - \beta_1^t)\, g.$
Dividing by $1 - \beta_1^t$ gives $\mathbb{E}[\hat m_t] = g$.
\end{proof}

The identical argument with $g_t^{\odot 2}$ in place of $g_t$ yields $\mathbb{E}[\hat v_t] = \mathbb{E}[g_t^{\odot 2}]$ under stationarity. Without bias correction, $m_1 = (1-\beta_1)g_1 \approx 0.1\, g$ at the default $\beta_1 = 0.9$ --- a tenfold underestimate that would cripple early training.

\begin{proposition}[Momentum as a low-pass filter]
For heavy-ball with $v_0 = 0$, $v_t = \sum_{i=1}^t \beta^{\,t-i} g_i$. The effective averaging horizon is $\sum_{j=0}^\infty \beta^j = 1/(1-\beta)$; e.g. $\beta = 0.9 \Rightarrow$ horizon $\approx 10$ steps. High-frequency gradient noise is attenuated; the consistent low-frequency signal accumulates.
\end{proposition}

\begin{theorem}[Heavy-ball acceleration on convex quadratics, sketch]
For $f(\theta) = \tfrac{1}{2}\theta^\top A \theta$ with $\mu I \preceq A \preceq L I$ and condition number $\kappa = L/\mu$, plain GD requires $O(\kappa \log 1/\epsilon)$ steps to reach $\epsilon$-accuracy, while heavy-ball with optimally tuned $\eta, \beta$ achieves $O(\sqrt{\kappa}\log 1/\epsilon)$.
\end{theorem}
\begin{proof}[Proof sketch]
Diagonalize $A$ and decompose into eigen-coordinates $\theta_t = \sum_i c_t^{(i)} u_i$. In each eigen-direction with eigenvalue $\lambda \in [\mu, L]$ the recurrence is
$\binom{c_{t+1}}{c_t} = M_\lambda \binom{c_t}{c_{t-1}}$, $\quad M_\lambda = \begin{pmatrix} 1 + \beta - \eta\lambda & -\beta \\ 1 & 0 \end{pmatrix}.$
Convergence rate equals $\max_\lambda |\rho(M_\lambda)|$. Choosing $\beta = \bigl((\sqrt{L}-\sqrt{\mu})/(\sqrt{L}+\sqrt{\mu})\bigr)^2$ and $\eta = 4/(\sqrt{L}+\sqrt{\mu})^2$ makes the eigenvalues complex conjugate with modulus $\sqrt{\beta} = 1 - 2/(\sqrt{\kappa}+1) = 1 - O(1/\sqrt{\kappa})$, the claimed acceleration.
\end{proof}

\begin{proposition}[Why AdamW $\neq$ Adam $+$ $L_2$]
For SGD, adding $\frac{\lambda}{2}\|\theta\|^2$ to the loss adds $\lambda\theta$ to the gradient, recovering decay $\theta \mapsto (1-\eta\lambda)\theta$. For Adam, the $L_2$ term enters $g_t$ and is rescaled by $1/\sqrt{\hat v_t}$: large-gradient parameters are decayed less, small-gradient ones more. AdamW restores the intended uniform shrinkage by applying $\lambda\theta_t$ outside the adaptive rescaling.
\end{proposition}

## Connection to LLMs

AdamW is the universal pre-training optimizer for transformers (GPT-2/3/4 family, Llama, Mistral, Gemma). Two facts from the analysis above explain why. (i) Per-coordinate rescaling matters: embedding rows, LayerNorm scales, and dense projection weights see gradients that differ in magnitude by 3--6 orders of magnitude; only adaptive optimizers train them all at one global learning rate. (ii) Decoupled weight decay is essential for generalization at scale --- with vanilla L2-Adam, infrequently-updated parameters (rare-token embeddings) would be barely regularized while high-gradient parameters would be over-shrunk. Bias correction matters most in the first $\sim 10$--$1000$ steps; the $1/(1-\beta_1)$ effective horizon explains why warmup of a few hundred steps is standard practice. We will revisit AdamW configuration (decay $\lambda = 0.1$, $\beta_2 = 0.95$, gradient clipping) in the transformer training chapter (Chapter 27).
