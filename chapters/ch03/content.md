## Motivation

Chapter 2 built the language of limits ($\varepsilon$–$N$ for sequences and $\varepsilon$–$\delta$ for functions). Continuity is the assertion that "small input perturbations produce small output perturbations"; differentiability strengthens this to "the perturbation is asymptotically linear." Both notions sit beneath every gradient that ever flows through a neural network. The chain rule, the central theorem of this chapter, is precisely the algebraic identity that **backpropagation** evaluates at scale (forward to Chapter 18).

## Definitions

**Definition 3.1 ($\varepsilon$–$\delta$ continuity).** Let $f:D\subseteq\mathbb{R}\to\mathbb{R}$ and $a\in D$. Then $f$ is *continuous at $a$* iff
$$\forall\,\varepsilon>0\;\exists\,\delta>0:\;|x-a|<\delta\text{ and }x\in D\;\Longrightarrow\;|f(x)-f(a)|<\varepsilon.$$
$f$ is *continuous on $S\subseteq D$* iff it is continuous at every $a\in S$.

**Proposition 3.2 (Sequential characterization).** $f$ is continuous at $a$ iff for every sequence $x_n\to a$ in $D$ we have $f(x_n)\to f(a)$.

*Proof.* ($\Rightarrow$) Given $\varepsilon>0$ pick $\delta$ from continuity; since $x_n\to a$, eventually $|x_n-a|<\delta$, so $|f(x_n)-f(a)|<\varepsilon$. ($\Leftarrow$) Suppose $f$ is *not* continuous at $a$: some $\varepsilon_0>0$ has no working $\delta$, so for each $n$ pick $x_n$ with $|x_n-a|<1/n$ but $|f(x_n)-f(a)|\ge\varepsilon_0$. Then $x_n\to a$ yet $f(x_n)\not\to f(a)$, contradiction. $\square$

**Definition 3.3 (Differentiability).** $f$ is *differentiable at $a$* (interior point of $D$) iff
$$f'(a)\;:=\;\lim_{h\to 0}\frac{f(a+h)-f(a)}{h}$$
exists in $\mathbb{R}$.

## Theorems with proofs

**Theorem 3.4 (Differentiable $\Rightarrow$ continuous).** If $f$ is differentiable at $a$, then $f$ is continuous at $a$.

*Proof.* For $h\ne 0$,
$$f(a+h)-f(a)\;=\;h\cdot\frac{f(a+h)-f(a)}{h}.$$
By limit algebra (Ch. 2) the right side tends to $0\cdot f'(a)=0$ as $h\to 0$. Hence $\lim_{h\to 0}f(a+h)=f(a)$, i.e. continuity at $a$. $\square$

**Theorem 3.5 (Sum, product, quotient rules).** Let $f,g$ be differentiable at $a$. Then $f+g$, $fg$, and (if $g(a)\ne 0$) $f/g$ are differentiable at $a$ with
$$(f+g)'(a)=f'(a)+g'(a),\quad (fg)'(a)=f'(a)g(a)+f(a)g'(a),$$
$$\left(\frac{f}{g}\right)'(a)=\frac{f'(a)g(a)-f(a)g'(a)}{g(a)^2}.$$

*Proof of the product rule (the others are analogous).* Use the "add and subtract" trick:
\begin{align*}
\frac{(fg)(a+h)-(fg)(a)}{h}
&=\frac{f(a+h)g(a+h)-f(a)g(a+h)+f(a)g(a+h)-f(a)g(a)}{h}\\
&=\frac{f(a+h)-f(a)}{h}\,g(a+h)\;+\;f(a)\,\frac{g(a+h)-g(a)}{h}.
\end{align*}
By Theorem 3.4, $g(a+h)\to g(a)$. The two difference quotients converge to $f'(a)$ and $g'(a)$. Limit algebra gives $f'(a)g(a)+f(a)g'(a)$. $\square$

**Theorem 3.6 (Chain rule, Carathéodory form).** Let $g$ be differentiable at $a$ and $f$ differentiable at $b:=g(a)$. Then $f\circ g$ is differentiable at $a$ and
$$(f\circ g)'(a)=f'(g(a))\cdot g'(a).$$

*Proof.* Define
$$\phi(y):=\begin{cases}\dfrac{f(y)-f(b)}{y-b},&y\ne b,\\ f'(b),&y=b.\end{cases}$$
By definition of $f'(b)$, $\phi$ is continuous at $b$. For *every* $y$ in a neighborhood of $b$,
$$f(y)-f(b)=\phi(y)\,(y-b),\tag{$\star$}$$
which holds at $y=b$ trivially and by construction otherwise. Apply ($\star$) with $y=g(x)$:
$$\frac{f(g(x))-f(g(a))}{x-a}=\phi(g(x))\cdot\frac{g(x)-g(a)}{x-a}.$$
As $x\to a$, $g(x)\to g(a)=b$ (Theorem 3.4) and $\phi$ is continuous at $b$, so $\phi(g(x))\to\phi(b)=f'(b)$. The second factor tends to $g'(a)$. Limit algebra closes the proof. $\square$

The Carathéodory device is preferred because it avoids the classical "$g(x)-g(a)$ might vanish" pitfall — $\phi$ is *defined everywhere*, so no division by zero ever occurs.

**Lemma 3.7 (Rolle).** If $f$ is continuous on $[a,b]$, differentiable on $(a,b)$, and $f(a)=f(b)$, then $\exists c\in(a,b)$ with $f'(c)=0$.

*Proof sketch.* By the extreme value theorem (proved in Ch. 4 from compactness; we cite it here), $f$ attains its max $M$ and min $m$ on $[a,b]$. If $M=m$ then $f$ is constant and any $c$ works. Otherwise an extremum is attained at some interior $c$; Fermat's interior-extremum lemma ($f'(c)=0$ at an interior extremum, immediate from one-sided difference quotients having opposite signs) finishes the job. $\square$

**Theorem 3.8 (Mean value theorem).** If $f$ is continuous on $[a,b]$ and differentiable on $(a,b)$, then $\exists c\in(a,b)$ with
$$f'(c)=\frac{f(b)-f(a)}{b-a}.$$

*Proof.* Define the auxiliary function
$$h(x):=f(x)-\frac{f(b)-f(a)}{b-a}(x-a).$$
Then $h$ is continuous on $[a,b]$, differentiable on $(a,b)$, and $h(a)=f(a)=h(b)$ (direct computation). By Rolle, $\exists c\in(a,b)$ with $h'(c)=0$, i.e. $f'(c)=(f(b)-f(a))/(b-a)$. $\square$

## Code sketch

The accompanying notebook (`cells.json`) numerically certifies $\varepsilon$–$\delta$ continuity for $f(x)=x^2$ at $a=2$ by binary-searching $\delta$, demonstrates the $O(h)$ error of forward differences for $\sin'$, verifies the chain rule on $\sin(x^2)$ at $x=1$ to $\sim 10^{-7}$ precision, and finds an MVT witness $c\in(0,2)$ for $f(x)=x^3-2x$ via bisection on $f'(x)-(f(2)-f(0))/2$.

## Connection to LLMs

A transformer is a composition $L_K\circ L_{K-1}\circ\cdots\circ L_1$ of differentiable layers. The gradient of the loss with respect to *any* parameter $\theta$ in layer $i$ is, by repeated application of Theorem 3.6,
$$\frac{\partial \mathcal{L}}{\partial \theta}=\frac{\partial \mathcal{L}}{\partial L_K}\cdot\frac{\partial L_K}{\partial L_{K-1}}\cdots\frac{\partial L_{i+1}}{\partial L_i}\cdot\frac{\partial L_i}{\partial \theta}.$$
**Backpropagation is precisely the right-to-left evaluation of this product** (Chapter 18). The MVT, in turn, underwrites convergence proofs for gradient descent (Chapter 14): it bounds $|f(x)-f(y)|$ by $\sup|f'|\cdot|x-y|$, which is the Lipschitz hypothesis under which descent provably decreases the loss. Without the chain rule, deep learning as we know it does not exist.
