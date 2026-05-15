# Chapter 7 — Convexity and optimization; gradient descent convergence

## Motivation

Chapters 3 and 4 built derivatives, the chain rule, and gradients; Chapter 6 equipped $\mathbb{R}^n$ with a norm and the Cauchy–Schwarz inequality. With those in hand we can finally ask the optimization question that drives every line of training code in this book: given $f : \mathbb{R}^n \to \mathbb{R}$, what does the iteration $x_{t+1} = x_t - \eta \nabla f(x_t)$ actually do, and when does it converge?

The honest answer for a transformer's loss is "we don't know." The loss is non-convex, the iterates are stochastic (Chapter 13), and rigorous global convergence is open. But the *convex* case admits complete proofs, and those proofs supply the only quantitative vocabulary we have for the non-convex one: smoothness, condition number, descent, contraction. Phrases like "smooth loss landscape," "warmup," and "$1/L$ step size" trace directly back to the two theorems below.

## Definitions

**Convex set.** $C \subseteq \mathbb{R}^n$ is *convex* iff for all $x, y \in C$ and $t \in [0,1]$, $tx + (1-t)y \in C$. The whole space $\mathbb{R}^n$ qualifies, and so does every Euclidean ball (Chapter 6).

**Convex function.** $f : \mathbb{R}^n \to \mathbb{R}$ is *convex* iff $f(tx + (1-t)y) \leq t f(x) + (1-t) f(y)$ for all $x, y \in \mathbb{R}^n$, $t \in [0,1]$. Equivalently, when $f$ is differentiable, $f(y) \geq f(x) + \langle \nabla f(x), y - x \rangle$ — the *first-order characterization*: every tangent hyperplane stays below the graph.

**$L$-smoothness.** $f$ is *$L$-smooth* iff $\nabla f$ is $L$-Lipschitz: $\|\nabla f(x) - \nabla f(y)\| \leq L \|x - y\|$ for all $x, y$, with the Euclidean norm of Chapter 6.

**$\mu$-strong convexity.** $f$ is *$\mu$-strongly convex* iff for all $x, y$,
$$f(y) \geq f(x) + \langle \nabla f(x), y - x \rangle + \tfrac{\mu}{2} \|y - x\|^2.$$
Equivalently, $f(x) - \tfrac{\mu}{2}\|x\|^2$ is convex. The ratio $\kappa = L/\mu \geq 1$ is the *condition number*.

**Gradient descent.** Fix $x_0 \in \mathbb{R}^n$ and a step size $\eta > 0$. Iterate
$$x_{t+1} = x_t - \eta \nabla f(x_t), \quad t = 0, 1, 2, \ldots$$

## Theorems and proofs

### Lemma (Descent lemma).
If $f$ is $L$-smooth then for all $x, y \in \mathbb{R}^n$,
$$f(y) \leq f(x) + \langle \nabla f(x), y - x \rangle + \tfrac{L}{2}\|y - x\|^2.$$

*Proof.* Let $g(t) = f(x + t(y - x))$ on $[0, 1]$. By the chain rule (Chapter 4), $g'(t) = \langle \nabla f(x + t(y - x)),\, y - x \rangle$. The fundamental theorem of calculus gives
$$f(y) - f(x) = g(1) - g(0) = \int_0^1 \langle \nabla f(x + t(y-x)),\, y - x \rangle \, dt.$$
Add and subtract $\langle \nabla f(x), y - x \rangle$:
$$f(y) - f(x) - \langle \nabla f(x), y - x \rangle = \int_0^1 \langle \nabla f(x + t(y-x)) - \nabla f(x),\, y - x \rangle \, dt.$$
By Cauchy–Schwarz (Chapter 6) and $L$-smoothness,
$$\langle \nabla f(x + t(y-x)) - \nabla f(x),\, y - x \rangle \leq \|\nabla f(x + t(y-x)) - \nabla f(x)\|\,\|y - x\| \leq L t \|y - x\|^2.$$
Integrating over $[0,1]$ yields $\int_0^1 L t \|y - x\|^2 \, dt = \tfrac{L}{2}\|y-x\|^2$. $\blacksquare$

### Theorem (GD on $L$-smooth convex). 
Let $f$ be convex and $L$-smooth with minimizer $x^*$ and minimum $f^* = f(x^*)$. With step size $\eta = 1/L$, the iterates of gradient descent satisfy
$$f(x_T) - f^* \;\leq\; \frac{L \|x_0 - x^*\|^2}{2T}, \qquad T \geq 1.$$

*Proof.* Apply the descent lemma with $y = x_{t+1} = x_t - \tfrac{1}{L}\nabla f(x_t)$ and $x = x_t$:
$$f(x_{t+1}) \leq f(x_t) + \langle \nabla f(x_t), -\tfrac{1}{L}\nabla f(x_t)\rangle + \tfrac{L}{2}\|\tfrac{1}{L}\nabla f(x_t)\|^2 = f(x_t) - \tfrac{1}{2L}\|\nabla f(x_t)\|^2. \qquad (\star)$$
So $f(x_t)$ is non-increasing. Now use convexity at $x_t$ against $x^*$: $f(x_t) - f^* \leq \langle \nabla f(x_t), x_t - x^*\rangle$. Combine with the gradient-step identity $x_{t+1} - x^* = (x_t - x^*) - \tfrac{1}{L}\nabla f(x_t)$:
$$\|x_{t+1} - x^*\|^2 = \|x_t - x^*\|^2 - \tfrac{2}{L}\langle \nabla f(x_t), x_t - x^*\rangle + \tfrac{1}{L^2}\|\nabla f(x_t)\|^2.$$
Hence $\tfrac{2}{L}\langle \nabla f(x_t), x_t - x^*\rangle = \|x_t - x^*\|^2 - \|x_{t+1} - x^*\|^2 + \tfrac{1}{L^2}\|\nabla f(x_t)\|^2$. Therefore
$$f(x_t) - f^* \leq \tfrac{L}{2}\bigl(\|x_t - x^*\|^2 - \|x_{t+1} - x^*\|^2\bigr) + \tfrac{1}{2L}\|\nabla f(x_t)\|^2.$$
Add $(\star)$ rewritten as $\tfrac{1}{2L}\|\nabla f(x_t)\|^2 \leq f(x_t) - f(x_{t+1})$:
$$f(x_{t+1}) - f^* \leq \tfrac{L}{2}\bigl(\|x_t - x^*\|^2 - \|x_{t+1} - x^*\|^2\bigr).$$
Telescope $t = 0, \ldots, T - 1$ and divide by $T$, using monotonicity of $f(x_t)$ to bound $f(x_T) - f^*$ by the average:
$$f(x_T) - f^* \leq \frac{1}{T}\sum_{t=0}^{T-1}\bigl(f(x_{t+1}) - f^*\bigr) \leq \frac{L \|x_0 - x^*\|^2}{2T}. \qquad \blacksquare$$

### Theorem (GD on $L$-smooth $\mu$-strongly convex). 
Let $f$ be $\mu$-strongly convex and $L$-smooth, $\eta = 1/L$. Then
$$\|x_t - x^*\|^2 \leq \bigl(1 - \mu/L\bigr)^t \|x_0 - x^*\|^2.$$

*Proof.* Strong convexity at $x_t$ versus $x^*$ gives $\langle \nabla f(x_t), x_t - x^*\rangle \geq f(x_t) - f^* + \tfrac{\mu}{2}\|x_t - x^*\|^2$, and from $(\star)$, $\tfrac{1}{2L}\|\nabla f(x_t)\|^2 \leq f(x_t) - f(x_{t+1}) \leq f(x_t) - f^*$. Plug both into the gradient-step expansion:
$$\|x_{t+1} - x^*\|^2 \leq \|x_t - x^*\|^2 - \tfrac{2}{L}\bigl(f(x_t) - f^* + \tfrac{\mu}{2}\|x_t - x^*\|^2\bigr) + \tfrac{2}{L}\bigl(f(x_t) - f^*\bigr) = \bigl(1 - \tfrac{\mu}{L}\bigr)\|x_t - x^*\|^2.$$
Iterate. $\blacksquare$

## Code sketch

We instantiate $f(x, y) = (x-1)^2 + 2(y+1)^2$, a strongly convex quadratic with Hessian $\mathrm{diag}(2, 4)$ so $\mu = 2$, $L = 4$. Gradient descent with $\eta = 1/L$ contracts $\|x_t - x^*\|^2$ by factor $1 - \mu/L = 1/2$ per step. We then take a degenerate $f(x) = \|Ax - b\|^2$ with $A \in \mathbb{R}^{20 \times 10}$ having a tiny smallest singular value (effectively non-strongly-convex on the slow direction); the rate degrades to the predicted $O(1/T)$, visible as slope $\approx -1$ on a log-log plot. Finally we compare $\eta = 1/L$ against the optimal $\eta = 2/(L + \mu)$, whose contraction factor $((\kappa - 1)/(\kappa + 1))^2$ is strictly smaller.

## Connection to LLMs

Transformer training losses are spectacularly non-convex: every permutation of attention heads, every sign flip in a layernorm, gives an equivalent minimum. None of the theorems above apply *globally*. They apply *locally*, and they supply the operating intuitions that survive the jump:

- **"Smooth loss landscape."** The descent lemma says $\eta < 2/L$ is a sufficient condition for monotone decrease. Empirically estimating local $L$ via the Hessian's top eigenvalue is exactly what learning-rate range tests and gradient-norm clipping approximate.
- **Warmup and LR scheduling (Chapter 27).** Early in training the local $L$ is large (sharp minima nearby); a small $\eta$ avoids the descent lemma's quadratic blow-up term. As the trajectory enters flatter regions, $L$ drops and $\eta$ can grow.
- **SGD and AdamW (Chapters 13, 14).** Stochastic gradients break the deterministic descent lemma; the convergence proofs replace $f(x_{t+1}) \leq f(x_t) - \tfrac{1}{2L}\|\nabla f(x_t)\|^2$ with an expectation inequality plus a variance term, but the algebraic skeleton — descent lemma, telescoping, contraction — is preserved.

The convex theory is thus a *load-bearing analogy*: the only place where the rates are honest, and the conceptual scaffold for every heuristic layered on top.
