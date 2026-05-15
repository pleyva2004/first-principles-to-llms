## Motivation

A neural network is a function $F : \mathbb{R}^n \to \mathbb{R}^m$ built by composing many simple maps. To train it we need the gradient of a scalar loss $\mathcal{L} : \mathbb{R}^p \to \mathbb{R}$ with respect to a high-dimensional parameter vector. Chapter 3 handled the one-dimensional case: existence of $f'(a)$, the mean value theorem (MVT), and the single-variable chain rule via the Carathéodory factorization $f(x) - f(a) = \varphi(x)(x-a)$ with $\varphi$ continuous at $a$. Here we lift those ideas to several variables. The two non-trivial issues are: (i) the *correct* notion of differentiability is **not** "all partials exist" but the stronger Fréchet condition, and (ii) the chain rule becomes a matrix product — exactly the operation that backpropagation iterates layer by layer (cf. Chapter 18).

## Definitions

Let $f : U \to \mathbb{R}^m$ with $U \subseteq \mathbb{R}^n$ open and $\mathbf{a} \in U$. Write $\mathbf{e}_i$ for the $i$-th standard basis vector and $\|\cdot\|$ for the Euclidean norm.

**Definition 4.1 (Partial derivative).** $\dfrac{\partial f}{\partial x_i}(\mathbf{a}) := \lim_{h \to 0} \dfrac{f(\mathbf{a} + h\mathbf{e}_i) - f(\mathbf{a})}{h}$, when the limit exists.

**Definition 4.2 (Directional derivative).** For a unit vector $\mathbf{v}$, $D_{\mathbf{v}} f(\mathbf{a}) := \lim_{t \to 0} \dfrac{f(\mathbf{a}+t\mathbf{v}) - f(\mathbf{a})}{t}$.

**Definition 4.3 (Total / Fréchet differentiability).** $f$ is *differentiable* at $\mathbf{a}$ if there exists a linear map $L : \mathbb{R}^n \to \mathbb{R}^m$ with
$$\lim_{\mathbf{h}\to\mathbf{0}} \frac{\|f(\mathbf{a}+\mathbf{h}) - f(\mathbf{a}) - L\mathbf{h}\|}{\|\mathbf{h}\|} = 0,$$
i.e. the remainder is $o(\|\mathbf{h}\|)$. The matrix of $L$ in the standard basis is the **Jacobian** $J_f(\mathbf{a}) \in \mathbb{R}^{m\times n}$.

**Definition 4.4 (Gradient).** When $m=1$, $\nabla f(\mathbf{a}) := J_f(\mathbf{a})^\top \in \mathbb{R}^n$, with components $\partial f / \partial x_i$.

Differentiability is strictly stronger than existence of partials: the textbook example $f(x,y)=xy/(x^2+y^2)$ ($f(0,0):=0$) has $\partial_x f(0,0)=\partial_y f(0,0)=0$ yet is discontinuous at the origin, hence not Fréchet differentiable.

## Theorems

**Theorem 4.5 (Differentiable $\Rightarrow$ partials exist; $L = J_f$).** If $f$ is differentiable at $\mathbf{a}$ with derivative $L$, then every partial exists and the $j$-th column of the matrix of $L$ is $\partial f / \partial x_j (\mathbf{a})$.

*Proof.* Take $\mathbf{h} = h\mathbf{e}_j$ with $h\in\mathbb{R}\setminus\{0\}$. Differentiability gives
$$f(\mathbf{a}+h\mathbf{e}_j) - f(\mathbf{a}) = h\, L\mathbf{e}_j + r(h), \quad \|r(h)\|/|h| \to 0.$$
Divide by $h$ and let $h\to 0$: the left side tends (by definition) to $\partial f/\partial x_j(\mathbf{a})$, the right side to $L\mathbf{e}_j$. Hence the partial exists and equals the $j$-th column of $L$. $\square$

The converse fails (partials alone are not enough), but a small extra hypothesis suffices.

**Theorem 4.6 (Continuous partials $\Rightarrow$ differentiable).** If all $\partial f/\partial x_j$ exist on a neighborhood of $\mathbf{a}$ and are continuous at $\mathbf{a}$, then $f$ is differentiable at $\mathbf{a}$.

*Sketch.* Reduce to $m=1$ componentwise. Telescope along axes:
$$f(\mathbf{a}+\mathbf{h}) - f(\mathbf{a}) = \sum_{j=1}^n \big[ f(\mathbf{a} + \mathbf{h}_{<j} + h_j\mathbf{e}_j) - f(\mathbf{a} + \mathbf{h}_{<j}) \big],$$
with $\mathbf{h}_{<j} := \sum_{i<j} h_i\mathbf{e}_i$. The single-variable MVT (Chapter 3) applied to each summand yields a $\xi_j$ between $0$ and $h_j$ with
$$f(\mathbf{a}+\mathbf{h}_{<j}+h_j\mathbf{e}_j) - f(\mathbf{a}+\mathbf{h}_{<j}) = h_j\, \partial_j f(\mathbf{a}+\mathbf{h}_{<j} + \xi_j \mathbf{e}_j).$$
Subtract $\sum_j h_j \partial_j f(\mathbf{a})$. By continuity of $\partial_j f$, each $|\partial_j f(\cdot) - \partial_j f(\mathbf{a})| \to 0$ as $\mathbf{h}\to\mathbf{0}$. Cauchy–Schwarz on the resulting sum gives a remainder bounded by $\|\mathbf{h}\| \cdot \varepsilon(\mathbf{h})$ with $\varepsilon \to 0$, i.e. $o(\|\mathbf{h}\|)$. (Spivak, *Calculus on Manifolds*, Thm. 2-8.) $\square$

**Theorem 4.7 (Multivariate chain rule).** Let $g : \mathbb{R}^k \to \mathbb{R}^n$ be differentiable at $\mathbf{a}$ and $f : \mathbb{R}^n \to \mathbb{R}^m$ differentiable at $\mathbf{b}=g(\mathbf{a})$. Then $f\circ g$ is differentiable at $\mathbf{a}$ and
$$J_{f\circ g}(\mathbf{a}) = J_f(\mathbf{b}) \, J_g(\mathbf{a}).$$

*Proof.* Differentiability gives Carathéodory-style remainders $\rho_g(\mathbf{h}) = g(\mathbf{a}+\mathbf{h}) - g(\mathbf{a}) - J_g(\mathbf{a})\mathbf{h}$ with $\rho_g(\mathbf{h}) = o(\|\mathbf{h}\|)$, and similarly $\rho_f(\mathbf{k}) = o(\|\mathbf{k}\|)$. Set $\mathbf{k}(\mathbf{h}) := g(\mathbf{a}+\mathbf{h}) - g(\mathbf{a})$. Then
$$f(g(\mathbf{a}+\mathbf{h})) - f(g(\mathbf{a})) = J_f(\mathbf{b})\,\mathbf{k}(\mathbf{h}) + \rho_f(\mathbf{k}(\mathbf{h})) = J_f(\mathbf{b})J_g(\mathbf{a})\,\mathbf{h} + J_f(\mathbf{b})\rho_g(\mathbf{h}) + \rho_f(\mathbf{k}(\mathbf{h})).$$
Bound the remainder. The first term is $\|J_f(\mathbf{b})\|_{\mathrm{op}} \cdot o(\|\mathbf{h}\|) = o(\|\mathbf{h}\|)$. For the second, $\|\mathbf{k}(\mathbf{h})\| \le (\|J_g(\mathbf{a})\|_{\mathrm{op}} + 1)\|\mathbf{h}\|$ for small $\|\mathbf{h}\|$, so $\rho_f(\mathbf{k}(\mathbf{h})) = o(\|\mathbf{k}\|) = o(\|\mathbf{h}\|)$. Thus the total remainder is $o(\|\mathbf{h}\|)$, proving differentiability with derivative $J_f(\mathbf{b})J_g(\mathbf{a})$. $\square$

**Theorem 4.8 (Schwarz / Clairaut: equality of mixed partials).** If $f : U \to \mathbb{R}$ is $C^2$ on an open $U$ containing $\mathbf{a}$, then $\partial^2 f / \partial x_i\,\partial x_j (\mathbf{a}) = \partial^2 f / \partial x_j\,\partial x_i (\mathbf{a})$.

*Proof.* Fix $i\neq j$ and write everything in the two-variable slice; suppress other coordinates. Define
$$\Delta(h,k) := f(\mathbf{a}+h\mathbf{e}_i+k\mathbf{e}_j) - f(\mathbf{a}+h\mathbf{e}_i) - f(\mathbf{a}+k\mathbf{e}_j) + f(\mathbf{a}).$$
Let $\varphi(s) := f(\mathbf{a}+s\mathbf{e}_i + k\mathbf{e}_j) - f(\mathbf{a}+s\mathbf{e}_i)$. Then $\Delta(h,k) = \varphi(h) - \varphi(0)$. Apply MVT: $\Delta = h\,\varphi'(\xi)$ for some $\xi\in(0,h)$, with $\varphi'(s) = \partial_i f(\mathbf{a}+s\mathbf{e}_i+k\mathbf{e}_j) - \partial_i f(\mathbf{a}+s\mathbf{e}_i)$. Apply MVT again in the $k$ variable: $\varphi'(\xi) = k\,\partial_j\partial_i f(\mathbf{a}+\xi\mathbf{e}_i+\eta\mathbf{e}_j)$ for some $\eta\in(0,k)$. Hence
$$\Delta(h,k) = hk\,\partial_j\partial_i f(\mathbf{a}+\xi\mathbf{e}_i+\eta\mathbf{e}_j).$$
Symmetrically, swapping the role of the two coordinates, $\Delta(h,k) = hk\,\partial_i\partial_j f(\mathbf{a}+\xi'\mathbf{e}_i+\eta'\mathbf{e}_j)$ with $\xi'\in(0,h)$, $\eta'\in(0,k)$. Divide by $hk$ and let $(h,k)\to(0,0)$. Continuity of the second partials (the $C^2$ hypothesis) makes both sides converge to $\partial_j\partial_i f(\mathbf{a})$ and $\partial_i\partial_j f(\mathbf{a})$ respectively. $\square$

## Code sketch

The accompanying notebook (`cells.json`) verifies each theorem numerically: gradient via central differences for $f(x,y)=x^2y+\sin(x+y)$; Jacobian column-by-column for $\mathbf{f}(x,y,z)=(xyz,\,x^2+\sin y+e^z)$; chain rule on $f\circ g$ with $g(t)=(\cos t,\sin t)$, $f(x,y)=x^2+y^2$ (the composite is constant, so the chain-rule product must vanish); Schwarz on $f(x,y)=x^3y^2+\sin(xy)$ via second-order finite differences.

## Connection to LLMs

A transformer is a composition $F = F_L \circ F_{L-1} \circ \cdots \circ F_1$ of differentiable layers. The scalar loss is $\mathcal{L} = \ell \circ F$. Theorem 4.7 gives
$$\nabla_{\theta_\ell} \mathcal{L} = \big( J_{\ell}(F(x))\, J_{F_L} \cdots J_{F_{\ell+1}} \big)\, \frac{\partial F_\ell}{\partial \theta_\ell}.$$
**Backpropagation never materializes any $J_{F_k}$.** It propagates the row vector $\mathbf{v}^\top := J_\ell\, J_{F_L} \cdots J_{F_{\ell+1}}$ right-to-left via vector–Jacobian products (VJPs): one VJP per layer, $O(\text{forward cost})$ each. Reverse-mode autodiff is exactly Theorem 4.7 applied with associativity exploited. We will derive this as an algorithm in Chapter 18.
