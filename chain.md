# From First Principles to Modern Large Language Models

**Author:** Paul Leyva

## Abstract

An unbroken chain of derivation from set theory to modern large language models. Twenty-eight chapters carry the reader from sets, functions, and proofs through real and multivariable analysis, linear algebra, probability, information theory, and convex/stochastic optimization, into neural networks, attention, the transformer block, pre-training, and the modern post-training stack (SFT, RLHF/PPO/GRPO, DPO). Every chapter is presented in three synchronized forms: a Markdown reading copy, a typeset PDF (LaTeX), and an executable Jupyter notebook.

<!-- TOC START -->
## Table of contents

### A — Foundations
- [Chapter 1: Sets, functions, logic, proofs](#chapter-1-sets-functions-logic-proofs)
- [Chapter 2: Numbers, sequences, limits, completeness](#chapter-2-numbers-sequences-limits-completeness)
- [Chapter 3: Continuity, univariate differentiation, chain rule](#chapter-3-continuity-univariate-differentiation-chain-rule)
- [Chapter 4: Multivariate calculus: partials, gradients, Jacobians](#chapter-4-multivariate-calculus-partials-gradients-jacobians)
- [Chapter 5: Linear algebra I: vector spaces, basis, linear maps](#chapter-5-linear-algebra-i-vector-spaces-basis-linear-maps)
- [Chapter 6: Linear algebra II: inner products, norms, eigenvalues, SVD](#chapter-6-linear-algebra-ii-inner-products-norms-eigenvalues-svd)
- [Chapter 7: Convexity and optimization; gradient descent convergence](#chapter-7-convexity-and-optimization-gradient-descent-convergence)

### B — Probability and Information
- [Chapter 8: Probability foundations: sample spaces, sigma-algebras, Kolmogorov axioms](#chapter-8-probability-foundations-sample-spaces-sigma-algebras-kolmogorov-axioms)
- [Chapter 9: Random variables, distributions, CDF/PMF/PDF](#chapter-9-random-variables-distributions-cdfpmfpdf)
- [Chapter 10: Expectation, variance, covariance; Jensen's inequality](#chapter-10-expectation-variance-covariance-jensens-inequality)
- [Chapter 11: Information theory: self-information, entropy, cross-entropy, KL](#chapter-11-information-theory-self-information-entropy-cross-entropy-kl)
- [Chapter 12: Statistical inference: likelihood, MLE, ERM, bias-variance](#chapter-12-statistical-inference-likelihood-mle-erm-bias-variance)

### C — Stochastic Optimization
- [Chapter 13: SGD: stochastic-approximation theorem; mini-batching; convergence sketch](#chapter-13-sgd-stochastic-approximation-theorem-mini-batching-convergence-sketch)
- [Chapter 14: Momentum, RMSProp, AdamW: derivation and bias-correction proof](#chapter-14-momentum-rmsprop-adamw-derivation-and-bias-correction-proof)

### D — Neural Networks
- [Chapter 15: MLPs as compositional functions; universal approximation](#chapter-15-mlps-as-compositional-functions-universal-approximation)
- [Chapter 16: Activation functions: ReLU/GELU/softmax with derivatives](#chapter-16-activation-functions-relugelusoftmax-with-derivatives)
- [Chapter 17: Loss functions: MSE, cross-entropy; gradients from first principles](#chapter-17-loss-functions-mse-cross-entropy-gradients-from-first-principles)
- [Chapter 18: Backpropagation: chain rule applied; reverse-mode AD as a graph algorithm](#chapter-18-backpropagation-chain-rule-applied-reverse-mode-ad-as-a-graph-algorithm)

### E — Sequence Models and Attention
- [Chapter 19: Embeddings: token to vector; lookup as a linear map; weight tying](#chapter-19-embeddings-token-to-vector-lookup-as-a-linear-map-weight-tying)
- [Chapter 20: RNN intuition; vanishing-gradient proof; why we need attention](#chapter-20-rnn-intuition-vanishing-gradient-proof-why-we-need-attention)
- [Chapter 21: Scaled dot-product attention: derivation, softmax-temperature analysis](#chapter-21-scaled-dot-product-attention-derivation-softmax-temperature-analysis)
- [Chapter 22: Multi-head attention: parallel heads as concat-then-project; complexity](#chapter-22-multi-head-attention-parallel-heads-as-concat-then-project-complexity)
- [Chapter 23: Transformer block: residual + LayerNorm/RMSNorm + FFN + attention; gradient-flow argument](#chapter-23-transformer-block-residual-layernormrmsnorm-ffn-attention-gradient-flow-argument)
- [Chapter 24: Positional encoding: sinusoidal derivation, RoPE construction](#chapter-24-positional-encoding-sinusoidal-derivation-rope-construction)

### F — Pre-training
- [Chapter 25: Causal masking; next-token prediction loss as MLE on the empirical distribution](#chapter-25-causal-masking-next-token-prediction-loss-as-mle-on-the-empirical-distribution)
- [Chapter 26: Tokenization: BPE algorithm; greedy merge correctness](#chapter-26-tokenization-bpe-algorithm-greedy-merge-correctness)
- [Chapter 27: Pre-training pipeline: AdamW + warmup + cosine decay + gradient clipping; tiny-GPT training run](#chapter-27-pre-training-pipeline-adamw-warmup-cosine-decay-gradient-clipping-tiny-gpt-training-run)

### G — Post-training
- [Chapter 28: SFT, RLHF (PPO/GRPO), and DPO; train + post-train a tiny GPT](#chapter-28-sft-rlhf-ppogrpo-and-dpo-train-post-train-a-tiny-gpt)

<!-- TOC END -->


# Block A — Foundations

<!-- CHAPTER 1 START -->
<a id="chapter-1-sets-functions-logic-proofs"></a>
## Chapter 1: Sets, functions, logic, proofs

## Motivation

Every object in this book — a vector, a probability distribution, a neural network, a token — is ultimately a *set with structure*. Before we can build the embedding map of Chapter 19 or even define the real numbers in Chapter 5, we need a precise grammar for membership, functions, and proof. Chapter 8 (linear maps) will treat functions between vector spaces; Chapter 15 (probability) will treat measures on $\sigma$-algebras of sets. Both collapse without the vocabulary below. We therefore start, with no apology, from the bottom.

## Definitions

A **set** $S$ is a collection of distinct objects called *elements*. We write $x \in S$ when $x$ is an element of $S$, and $x \notin S$ otherwise. The **empty set** $\emptyset$ is the unique set with no elements. We say $A$ is a **subset** of $B$, written $A \subset B$, iff $\forall x\,(x \in A \Rightarrow x \in B)$. Two sets are **equal**, $A = B$, iff $A \subset B$ and $B \subset A$ (extensionality).

Given $A, B$ inside a universe $U$:
- **Union:** $A \cup B := \{x \in U : x \in A \lor x \in B\}$.
- **Intersection:** $A \cap B := \{x \in U : x \in A \land x \in B\}$.
- **Complement:** $A^c := \{x \in U : x \notin A\}$.
- **Power set:** $\mathcal{P}(S) := \{T : T \subset S\}$.

A **function** $f : A \to B$ is a rule assigning to each $a \in A$ exactly one $f(a) \in B$. The **image** is $f(A) := \{f(a) : a \in A\}$, and the **preimage** of $T \subset B$ is $f^{-1}(T) := \{a \in A : f(a) \in T\}$. We call $f$:
- **injective** iff $\forall a_1, a_2 \in A,\; f(a_1) = f(a_2) \Rightarrow a_1 = a_2$;
- **surjective** iff $\forall b \in B,\; \exists a \in A,\; f(a) = b$;
- **bijective** iff both.

**Logic.** The propositional connectives are $\land$ (and), $\lor$ (or), $\neg$ (not), $\Rightarrow$ (implies), $\Leftrightarrow$ (iff). The quantifiers are $\forall$ (for all) and $\exists$ (there exists). Standard proof techniques: **direct** (assume $P$, derive $Q$, conclude $P \Rightarrow Q$), **contrapositive** ($P \Rightarrow Q$ is logically equivalent to $\neg Q \Rightarrow \neg P$), **contradiction** (assume $\neg P$, derive $\bot$), and **induction** (proved below).

## Theorems and proofs

**Theorem 1 (De Morgan's laws).** *For sets $A, B \subset U$,*
$$(A \cup B)^c = A^c \cap B^c \qquad \text{and} \qquad (A \cap B)^c = A^c \cup B^c.$$

*Proof.* We prove the first; the second is symmetric. We show both inclusions, equivalently the biconditional $x \in (A \cup B)^c \Leftrightarrow x \in A^c \cap B^c$ for arbitrary $x \in U$.

$(\Rightarrow)$ Assume $x \in (A \cup B)^c$. By definition of complement, $x \notin A \cup B$, i.e. $\neg(x \in A \lor x \in B)$. By De Morgan's law of propositional logic this is $(\neg(x \in A)) \land (\neg(x \in B))$, i.e. $x \notin A$ and $x \notin B$. Hence $x \in A^c$ and $x \in B^c$, so $x \in A^c \cap B^c$.

$(\Leftarrow)$ Assume $x \in A^c \cap B^c$. Then $x \notin A$ and $x \notin B$, so $\neg(x \in A) \land \neg(x \in B)$, which by propositional De Morgan equals $\neg(x \in A \lor x \in B)$, i.e. $x \notin A \cup B$, i.e. $x \in (A \cup B)^c$.

For the second identity, the same argument with $\land$ and $\lor$ interchanged gives $x \in (A \cap B)^c \Leftrightarrow x \in A^c \cup B^c$. $\blacksquare$

**Theorem 2 (Composition of injections).** *If $f : A \to B$ and $g : B \to C$ are injective, then $g \circ f : A \to C$ is injective.*

*Proof.* Let $a_1, a_2 \in A$ with $(g \circ f)(a_1) = (g \circ f)(a_2)$, i.e. $g(f(a_1)) = g(f(a_2))$. Since $g$ is injective, $f(a_1) = f(a_2)$. Since $f$ is injective, $a_1 = a_2$. Therefore $g \circ f$ is injective. $\blacksquare$

**Theorem 3 (Principle of mathematical induction).** *Let $P(n)$ be a predicate on $\mathbb{N} = \{1, 2, 3, \ldots\}$. If (i) $P(1)$ holds and (ii) $\forall n \in \mathbb{N},\; P(n) \Rightarrow P(n+1)$, then $\forall n \in \mathbb{N},\; P(n)$.*

*Proof.* We assume the **well-ordering principle**: every nonempty subset of $\mathbb{N}$ has a least element. Let $S := \{n \in \mathbb{N} : \neg P(n)\}$. We show $S = \emptyset$ by contradiction. Suppose $S \neq \emptyset$. By well-ordering, $S$ has a least element $m$. By (i), $P(1)$, so $1 \notin S$, hence $m \geq 2$, so $m - 1 \in \mathbb{N}$. By minimality of $m$, $m - 1 \notin S$, i.e. $P(m-1)$ holds. By (ii) applied at $n = m - 1$, $P(m)$ holds, so $m \notin S$, contradicting $m \in S$. Hence $S = \emptyset$, i.e. $P(n)$ holds for all $n \in \mathbb{N}$. $\blacksquare$

## Code sketch

We will (a) enumerate $\mathcal{P}(\{a,b,c\})$ from scratch and check $|\mathcal{P}(S)| = 2^{|S|}$; (b) brute-force De Morgan on $U = \{1,\ldots,8\}$; (c) implement `is_injective`, `is_surjective`, `is_bijective` as predicates over finite functions; (d) verify $\sum_{k=1}^{n} k = n(n+1)/2$ both by direct sum and by an explicit induction step.

## Connection to LLMs

A language model's **vocabulary** $\mathcal{V}$ is a finite set of tokens (e.g. $|\mathcal{V}| \approx 50{,}000$). The tokenizer is a function $T : \text{strings} \to \mathcal{V}^*$. The **embedding map** $E : \mathcal{V} \to \mathbb{R}^d$ is a function from a finite set into a real vector space; the implementation as a lookup table $E[i]$ requires that the vocabulary index $\mathcal{V} \to \{0, 1, \ldots, |\mathcal{V}|-1\}$ be a **bijection**. Injectivity ensures distinct tokens get distinct rows; surjectivity ensures every row is reachable. We will revisit this in Chapter 19, where $E$ becomes the first learnable layer of the transformer. The rest of the book is, in a strict sense, a long story about which functions between which sets one is allowed to write down.

<!-- CHAPTER 1 END -->

<!-- CHAPTER 2 START -->
<a id="chapter-2-numbers-sequences-limits-completeness"></a>
## Chapter 2: Numbers, sequences, limits, completeness

## Motivation

In Chapter 1 we built logic, sets, and proof technique. We now build the arithmetic playground in which all of analysis, optimization, and ultimately neural network training takes place: the real numbers $\mathbb{R}$. The central question of this chapter is not "what are numbers?" in a metaphysical sense but rather: *what structural property of $\mathbb{R}$ makes it the right home for limits?* The answer is **completeness**, and it is what will let us, many chapters later, prove that gradient descent iterates $\theta_t \in \mathbb{R}^d$ actually converge.

## Definitions

We accept $\mathbb{N} = \{0, 1, 2, \ldots\}$ as constructed in Chapter 1 (von Neumann ordinals). The integers $\mathbb{Z}$ extend $\mathbb{N}$ by formal additive inverses; the rationals $\mathbb{Q}$ extend $\mathbb{Z}$ by formal multiplicative inverses of nonzero elements. We then take $\mathbb{R}$ to be a complete ordered field containing $\mathbb{Q}$; one explicit construction (Dedekind cuts or Cauchy completion of $\mathbb{Q}$) is sketched in the appendix. Thus
$$\mathbb{N} \subset \mathbb{Z} \subset \mathbb{Q} \subset \mathbb{R}.$$

**Definition 2.1 (Sequence).** A *sequence* in a set $X$ is a function $a : \mathbb{N} \to X$, written $(a_n)_{n \in \mathbb{N}}$.

**Definition 2.2 (Limit, $\varepsilon$–$N$).** A real sequence $(a_n)$ *converges* to $L \in \mathbb{R}$, written $a_n \to L$, iff
$$\forall \varepsilon > 0 \;\; \exists N \in \mathbb{N} \;\; \forall n \geq N : \;\; |a_n - L| < \varepsilon.$$

**Definition 2.3 (Cauchy).** $(a_n)$ is *Cauchy* iff $\forall \varepsilon > 0\; \exists N\; \forall m, n \geq N : |a_m - a_n| < \varepsilon$.

**Definition 2.4 (Bounded, monotone).** $(a_n)$ is *bounded above* iff $\exists M : a_n \leq M$ for all $n$; bounded below symmetrically; *bounded* iff both. It is *monotone increasing* iff $a_n \leq a_{n+1}$ for all $n$.

**Definition 2.5 (Supremum, infimum).** Let $S \subseteq \mathbb{R}$ be nonempty and bounded above. $u \in \mathbb{R}$ is an *upper bound* if $s \leq u$ for all $s \in S$. The *supremum* $\sup S$ is the least upper bound: an upper bound such that for every $\varepsilon > 0$ there exists $s \in S$ with $s > \sup S - \varepsilon$.

**Axiom 2.6 (Completeness of $\mathbb{R}$).** Every nonempty subset of $\mathbb{R}$ that is bounded above has a supremum in $\mathbb{R}$.

## Theorems and proofs

**Theorem 2.7 ($\sqrt{2} \notin \mathbb{Q}$).** There is no rational $q$ with $q^2 = 2$.

*Proof.* Suppose for contradiction (technique from Chapter 1) that $q = p/r$ with $p, r \in \mathbb{Z}$, $r \neq 0$, and the fraction in lowest terms (so $\gcd(p, r) = 1$), and $q^2 = 2$. Then $p^2 = 2 r^2$, so $p^2$ is even, hence $p$ is even (since the square of an odd integer is odd). Write $p = 2k$. Substituting, $4 k^2 = 2 r^2$, i.e. $r^2 = 2 k^2$, so $r^2$ is even and therefore $r$ is even. But then $2 \mid \gcd(p, r)$, contradicting $\gcd(p, r) = 1$. $\square$

This is the first concrete witness that $\mathbb{Q}$ is *incomplete*: the bounded-above set $S = \{q \in \mathbb{Q} : q^2 < 2\}$ has no supremum *in $\mathbb{Q}$*. Completeness (Axiom 2.6) is exactly the patch.

**Theorem 2.8 (Bounded monotone convergence).** Let $(a_n)$ be monotone increasing and bounded above in $\mathbb{R}$. Then $(a_n)$ converges, and $\lim a_n = \sup_n a_n$.

*Proof.* Let $S = \{a_n : n \in \mathbb{N}\}$. $S$ is nonempty (it contains $a_0$) and bounded above by hypothesis. By Axiom 2.6, $L := \sup S$ exists in $\mathbb{R}$. Fix $\varepsilon > 0$. By the supremum's least-upper-bound property, $L - \varepsilon$ is *not* an upper bound, so there exists $N \in \mathbb{N}$ with $a_N > L - \varepsilon$. For any $n \geq N$, monotonicity gives $a_n \geq a_N > L - \varepsilon$, while $a_n \leq L$ since $L$ is an upper bound. Hence $L - \varepsilon < a_n \leq L$, so $|a_n - L| < \varepsilon$. This is exactly Definition 2.2. $\square$

**Lemma 2.9 (Bolzano–Weierstrass for $\mathbb{R}$).** Every bounded real sequence has a convergent subsequence.

*Proof.* Let $(a_n) \subseteq [c_0, d_0]$ with $c_0 < d_0$. Bisect the interval at the midpoint $m_0 = (c_0 + d_0)/2$; at least one of $[c_0, m_0]$ or $[m_0, d_0]$ contains $a_n$ for infinitely many $n$ (else only finitely many terms, contradicting infinitude of $\mathbb{N}$). Call this half $[c_1, d_1]$ and pick $n_1$ with $a_{n_1} \in [c_1, d_1]$. Iterate: at stage $k$ we have nested $[c_k, d_k]$ of length $(d_0 - c_0) / 2^k$ each containing infinitely many terms, and pick $n_k > n_{k-1}$ with $a_{n_k} \in [c_k, d_k]$. The sequence $(c_k)$ is monotone increasing and bounded above by $d_0$, so by Theorem 2.8 it converges to some $L$; similarly $(d_k) \to L$ since $d_k - c_k \to 0$. By the squeeze (since $c_k \leq a_{n_k} \leq d_k$), $a_{n_k} \to L$. $\square$

**Theorem 2.10 (Cauchy completeness of $\mathbb{R}$).** Every Cauchy sequence in $\mathbb{R}$ converges.

*Proof.* Let $(a_n)$ be Cauchy. *Step 1: bounded.* Pick $\varepsilon = 1$ and $N$ from the Cauchy condition; for $n \geq N$, $|a_n| \leq |a_N| + 1$, so $(a_n)$ is bounded by $\max(|a_0|, \ldots, |a_{N-1}|, |a_N| + 1)$. *Step 2: convergent subsequence.* By Lemma 2.9, some subsequence $a_{n_k} \to L$. *Step 3: full sequence.* Fix $\varepsilon > 0$. Choose $N_1$ with $|a_m - a_n| < \varepsilon/2$ for $m, n \geq N_1$, and $K$ with $n_K \geq N_1$ and $|a_{n_K} - L| < \varepsilon/2$. Then for $n \geq N_1$,
$$|a_n - L| \leq |a_n - a_{n_K}| + |a_{n_K} - L| < \varepsilon/2 + \varepsilon/2 = \varepsilon. \quad \square$$

## Code sketch

Numerically the partial sums $S_n = \sum_{k=1}^n 1/k^2$ approach $\pi^2/6$. We will print $|S_n - \pi^2/6|$ and locate the smallest $N$ realizing each tolerance $\varepsilon \in \{10^{-2}, 10^{-4}, 10^{-6}\}$, and contrast a Cauchy sequence ($1/n$) with a non-Cauchy one ($(-1)^n$).

## Connection to LLMs

Training a transformer is a sequence $(\theta_t)_{t \in \mathbb{N}}$ in $\mathbb{R}^d$ produced by an optimizer (SGD, Adam). Statements like "the loss converges" or "the iterates converge to a stationary point" are *limit statements in the sense of Definition 2.2*. Convergence proofs for SGD (Chapter 13) and Adam (Chapter 14) reduce, ultimately, to bounded-monotone arguments on auxiliary scalar sequences (loss values, gradient norms) — and those arguments would simply be false over $\mathbb{Q}$. Completeness is not philosophical decoration; it is the load-bearing axiom under every convergence theorem in deep learning.

<!-- CHAPTER 2 END -->

<!-- CHAPTER 3 START -->
<a id="chapter-3-continuity-univariate-differentiation-chain-rule"></a>
## Chapter 3: Continuity, univariate differentiation, chain rule

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

<!-- CHAPTER 3 END -->

<!-- CHAPTER 4 START -->
<a id="chapter-4-multivariate-calculus-partials-gradients-jacobians"></a>
## Chapter 4: Multivariate calculus: partials, gradients, Jacobians

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

<!-- CHAPTER 4 END -->

<!-- CHAPTER 5 START -->
<a id="chapter-5-linear-algebra-i-vector-spaces-basis-linear-maps"></a>
## Chapter 5: Linear algebra I: vector spaces, basis, linear maps

## Motivation

Every transformer layer is, between its bias terms and nonlinearities, a *linear map between finite-dimensional real vector spaces*. The "hidden dimension" $d$ is just $\dim \mathbb{R}^d$; the "weight matrix" $W \in \mathbb{R}^{m \times n}$ is just the matrix representation of a linear map $\mathbb{R}^n \to \mathbb{R}^m$ in the standard bases; "residual connections" are addition in a vector space; "low-rank adapters" are statements about the *image* of a linear map. Before we can speak of attention (Chapter 21) or embeddings (Chapter 19), we need the grammar of vector spaces, bases, dimension, kernels, images, and the rank--nullity theorem. We rely on Chapter 1 (sets, functions, logic) throughout.

## Definitions

A **field** $\mathbb{F}$ is a set with two binary operations $+, \cdot$ such that $(\mathbb{F}, +)$ is an abelian group with identity $0$, $(\mathbb{F} \setminus \{0\}, \cdot)$ is an abelian group with identity $1$, and multiplication distributes over addition. The canonical example is $\mathbb{F} = \mathbb{R}$.

A **vector space** $V$ over $\mathbb{F}$ is a set equipped with addition $+ : V \times V \to V$ and scalar multiplication $\cdot : \mathbb{F} \times V \to V$ satisfying the **eight axioms**, for all $u, v, w \in V$ and $a, b \in \mathbb{F}$:
1. $(u + v) + w = u + (v + w)$ (associativity);
2. $u + v = v + u$ (commutativity);
3. $\exists\, 0 \in V$ such that $v + 0 = v$ for all $v$ (zero);
4. $\forall v \in V,\ \exists (-v) \in V$ with $v + (-v) = 0$ (additive inverse);
5. $a \cdot (u + v) = a \cdot u + a \cdot v$ (distributivity over vectors);
6. $(a + b) \cdot v = a \cdot v + b \cdot v$ (distributivity over scalars);
7. $(ab) \cdot v = a \cdot (b \cdot v)$ (compatibility);
8. $1 \cdot v = v$ (scalar identity).

A **subspace** $U \subset V$ is a subset closed under $+$ and scalar multiplication that contains $0$. A **linear combination** of $v_1, \ldots, v_k \in V$ is any vector $\sum_{i=1}^k a_i v_i$ with $a_i \in \mathbb{F}$. The **span** of $S \subset V$ is $\mathrm{span}(S) := \{\sum a_i v_i : v_i \in S, a_i \in \mathbb{F}\}$, the smallest subspace containing $S$. A finite set $\{v_1, \ldots, v_k\}$ is **linearly independent** iff $\sum a_i v_i = 0 \Rightarrow a_1 = \cdots = a_k = 0$. A **basis** of $V$ is a linearly independent spanning set. The **dimension** $\dim V$ is the cardinality of any basis (well-defined by Theorem 3 below).

A **linear map** $T : V \to W$ between vector spaces over the same $\mathbb{F}$ satisfies $T(au + bv) = a T(u) + b T(v)$ for all $u, v \in V$, $a, b \in \mathbb{F}$. Its **kernel** is $\ker T := \{v \in V : T(v) = 0\}$ and its **image** is $\mathrm{im}\,T := \{T(v) : v \in V\}$; both are subspaces. Given bases $(e_1, \ldots, e_n)$ of $V = \mathbb{R}^n$ and $(f_1, \ldots, f_m)$ of $W = \mathbb{R}^m$, the **matrix representation** of $T$ has $j$-th column equal to the coordinate vector of $T(e_j)$ in the $f$-basis: $A \in \mathbb{R}^{m \times n}$ with $T(e_j) = \sum_i A_{ij} f_i$.

## Theorems and proofs

**Theorem 1 (Steinitz exchange lemma).** *Let $V$ be a vector space over $\mathbb{F}$. If $\{v_1, \ldots, v_m\}$ is linearly independent and $\{w_1, \ldots, w_n\}$ spans $V$, then $m \leq n$, and after reindexing the $w_j$ we may replace $m$ of them by $v_1, \ldots, v_m$ so that the resulting set still spans $V$.*

*Proof.* By induction on $m$. The case $m = 0$ is trivial. Assume the claim for $m - 1$: after reindexing, $\{v_1, \ldots, v_{m-1}, w_m, \ldots, w_n\}$ spans $V$ (so in particular $m - 1 \leq n$). Then $v_m \in V$ is a linear combination $v_m = \sum_{i < m} a_i v_i + \sum_{j \geq m} b_j w_j$. If all $b_j = 0$, then $v_m \in \mathrm{span}(v_1, \ldots, v_{m-1})$, contradicting linear independence of $\{v_1, \ldots, v_m\}$. So some $b_{j_0} \neq 0$; in particular such a $j_0 \in \{m, \ldots, n\}$ exists, forcing $n \geq m$. Reindex so $j_0 = m$. Solving for $w_m$,
$$w_m = b_m^{-1}\!\Big(v_m - \sum_{i<m} a_i v_i - \sum_{j>m} b_j w_j\Big) \in \mathrm{span}(v_1, \ldots, v_m, w_{m+1}, \ldots, w_n).$$
Hence $\mathrm{span}(v_1, \ldots, v_m, w_{m+1}, \ldots, w_n) \supset \{v_1, \ldots, v_{m-1}, w_m, \ldots, w_n\}$, which spans $V$. So $\{v_1, \ldots, v_m, w_{m+1}, \ldots, w_n\}$ spans $V$. $\blacksquare$

**Theorem 2 (Spanning sets contain bases; independent sets extend to bases).** *In a finite-dimensional $V$: (a) any finite spanning set contains a basis; (b) any linearly independent set extends to a basis.*

*Proof.* (a) Take a finite spanning set $S$. If linearly dependent, some $w \in S$ lies in $\mathrm{span}(S \setminus \{w\})$, so $S \setminus \{w\}$ still spans. Iterate until linearly independent; the result is a basis. (b) Take an independent set $L$ and any finite spanning set $S$. Apply Theorem 1: replace $|L|$ elements of $S$ by $L$ to obtain a spanning set containing $L$; then apply (a) by removing dependent vectors only from the $S$-side. $\blacksquare$

**Theorem 3 (Invariance of dimension).** *Any two bases of a finite-dimensional $V$ have the same cardinality.*

*Proof.* Let $\mathcal{B}_1, \mathcal{B}_2$ be bases with $|\mathcal{B}_1| = m$, $|\mathcal{B}_2| = n$. $\mathcal{B}_1$ is independent and $\mathcal{B}_2$ spans, so by Theorem 1, $m \leq n$. Swapping roles, $n \leq m$. Hence $m = n$. $\blacksquare$

**Theorem 4 (Rank--nullity).** *Let $T : V \to W$ be linear with $\dim V = n < \infty$. Then $\dim \ker T + \dim \mathrm{im}\, T = n$.*

*Proof.* Let $(u_1, \ldots, u_k)$ be a basis of $\ker T$. By Theorem 2(b), extend to a basis $(u_1, \ldots, u_k, v_1, \ldots, v_{n-k})$ of $V$. We claim $(T v_1, \ldots, T v_{n-k})$ is a basis of $\mathrm{im}\,T$.

*Spanning.* Any $w \in \mathrm{im}\,T$ has $w = T(\sum a_i u_i + \sum b_j v_j) = \sum b_j T v_j$ since $T u_i = 0$.

*Independence.* Suppose $\sum c_j T v_j = 0$. Then $T(\sum c_j v_j) = 0$, so $\sum c_j v_j \in \ker T = \mathrm{span}(u_i)$. Write $\sum c_j v_j = \sum d_i u_i$, i.e. $\sum c_j v_j - \sum d_i u_i = 0$. Linear independence of the full basis forces all $c_j = 0$ (and all $d_i = 0$).

Hence $\dim \mathrm{im}\,T = n - k = n - \dim \ker T$. $\blacksquare$

## Code sketch

We implement `is_linearly_independent` and `dim_span` via numpy's matrix rank (which equals the dimension of the column span). We then build a $4 \times 6$ random integer matrix, compute its rank, extract a basis of $\ker A$ from the right singular vectors with zero singular values, verify $A v = 0$ for each null-space basis vector, and confirm the rank--nullity identity numerically. Finally, we verify the change-of-basis formula $A v$ versus $P (P^{-1} A P) (P^{-1} v)$ for a random invertible $P$.

## Connection to LLMs

A transformer with hidden dimension $d$ operates on the vector space $\mathbb{R}^d$. Each linear projection ($Q, K, V$ in attention; the up-and-down projections in the MLP) is a linear map between $\mathbb{R}^d$ and $\mathbb{R}^{d_k}$ or $\mathbb{R}^{d_{\mathrm{ff}}}$ (Chapter 21). The token embedding map of Chapter 19 is a linear map $\mathbb{R}^{|\mathcal{V}|} \to \mathbb{R}^d$ applied to one-hot inputs; equivalently, a row-lookup. The "rank" of an attention matrix and the "intrinsic dimension" of activations are statements about $\dim \mathrm{im}$. LoRA fine-tuning constrains weight updates to lie in a low-dimensional subspace --- a direct application of $\dim \mathrm{im}\,T \leq \min(m, n)$. Rank--nullity will reappear when we count parameters and degrees of freedom.

<!-- CHAPTER 5 END -->

<!-- CHAPTER 6 START -->
<a id="chapter-6-linear-algebra-ii-inner-products-norms-eigenvalues-svd"></a>
## Chapter 6: Linear algebra II: inner products, norms, eigenvalues, SVD

## Motivation

Chapter 5 built vector spaces and linear maps as raw algebraic objects. To do *geometry* — to talk about *length*, *angle*, *orthogonality*, and *best approximation* — we need additional structure: an **inner product**. From this single addition we will derive the Cauchy–Schwarz inequality, the triangle inequality, the spectral theorem, the singular value decomposition (SVD), and the Eckart–Young low-rank approximation theorem. These are the workhorses behind PCA, attention, and LoRA fine-tuning of LLMs.

## Definitions

**Definition (Inner product).** A function $\langle \cdot, \cdot\rangle : V \times V \to \mathbb{R}$ on a real vector space $V$ is an *inner product* if for all $\mathbf{x},\mathbf{y},\mathbf{z}\in V$ and $a,b\in\mathbb{R}$:

1. *Symmetry*: $\langle \mathbf{x},\mathbf{y}\rangle = \langle \mathbf{y},\mathbf{x}\rangle$.
2. *Linearity in the first argument*: $\langle a\mathbf{x}+b\mathbf{y},\mathbf{z}\rangle = a\langle \mathbf{x},\mathbf{z}\rangle + b\langle \mathbf{y},\mathbf{z}\rangle$.
3. *Positive-definiteness*: $\langle \mathbf{x},\mathbf{x}\rangle \geq 0$, with equality iff $\mathbf{x}=\mathbf{0}$.

On $\mathbb{R}^n$ the canonical example is the **dot product** $\langle \mathbf{x},\mathbf{y}\rangle = \sum_{i=1}^n x_i y_i = \mathbf{x}^T\mathbf{y}$.

**Definition (Induced norm).** $\|\mathbf{x}\| := \sqrt{\langle \mathbf{x},\mathbf{x}\rangle}$.

**Definition (Orthogonality).** Vectors $\mathbf{x},\mathbf{y}$ are *orthogonal* if $\langle \mathbf{x},\mathbf{y}\rangle = 0$. A set $\{\mathbf{q}_i\}$ is *orthonormal* if $\langle \mathbf{q}_i,\mathbf{q}_j\rangle = \delta_{ij}$. An *orthonormal basis* is an orthonormal spanning set.

**Definition (Eigenvalue, eigenvector).** For $A\in\mathbb{R}^{n\times n}$, a scalar $\lambda$ and nonzero $\mathbf{v}\in\mathbb{R}^n$ form an eigenpair if $A\mathbf{v}=\lambda\mathbf{v}$. The *characteristic polynomial* is $p_A(\lambda) := \det(A-\lambda I)$; its roots are exactly the eigenvalues.

**Definition (Symmetric, positive (semi)definite).** $A$ is *symmetric* if $A^T=A$; *positive semidefinite* (PSD) if symmetric with $\mathbf{x}^T A \mathbf{x}\geq 0$ for all $\mathbf{x}$; *positive definite* if strict inequality holds for $\mathbf{x}\ne 0$.

**Definition (SVD).** A *singular value decomposition* of $A\in\mathbb{R}^{m\times n}$ is a factorization $A = U\Sigma V^T$ where $U\in\mathbb{R}^{m\times m}$ and $V\in\mathbb{R}^{n\times n}$ are orthogonal ($U^T U=I$, $V^T V=I$) and $\Sigma\in\mathbb{R}^{m\times n}$ is "diagonal" with nonnegative entries $\sigma_1\geq\sigma_2\geq\cdots\geq 0$.

## Theorems and Proofs

**Theorem (Cauchy–Schwarz).** For all $\mathbf{x},\mathbf{y}\in V$, $|\langle \mathbf{x},\mathbf{y}\rangle| \leq \|\mathbf{x}\|\,\|\mathbf{y}\|$.

*Proof.* If $\mathbf{y}=\mathbf{0}$ both sides are $0$. Otherwise, for every $t\in\mathbb{R}$,
$$0 \leq \|\mathbf{x}-t\mathbf{y}\|^2 = \langle \mathbf{x}-t\mathbf{y},\mathbf{x}-t\mathbf{y}\rangle = \|\mathbf{x}\|^2 - 2t\langle \mathbf{x},\mathbf{y}\rangle + t^2\|\mathbf{y}\|^2.$$
This is a quadratic in $t$ that is nonnegative everywhere, so its discriminant is $\leq 0$:
$$(2\langle \mathbf{x},\mathbf{y}\rangle)^2 - 4\|\mathbf{x}\|^2\|\mathbf{y}\|^2 \leq 0,$$
i.e. $\langle \mathbf{x},\mathbf{y}\rangle^2 \leq \|\mathbf{x}\|^2\|\mathbf{y}\|^2$. Take square roots. $\square$

**Corollary (Triangle inequality).** $\|\mathbf{x}+\mathbf{y}\| \leq \|\mathbf{x}\| + \|\mathbf{y}\|$.

*Proof.* $\|\mathbf{x}+\mathbf{y}\|^2 = \|\mathbf{x}\|^2 + 2\langle \mathbf{x},\mathbf{y}\rangle + \|\mathbf{y}\|^2 \leq \|\mathbf{x}\|^2 + 2\|\mathbf{x}\|\|\mathbf{y}\| + \|\mathbf{y}\|^2 = (\|\mathbf{x}\|+\|\mathbf{y}\|)^2$, using Cauchy–Schwarz. $\square$

**Theorem (Spectral theorem, real symmetric case).** Every symmetric $A\in\mathbb{R}^{n\times n}$ admits a factorization $A = Q\Lambda Q^T$ with $Q$ orthogonal and $\Lambda$ real diagonal.

*Proof (induction on $n$).* For $n=1$ trivial. Assume the result for $n-1$. The Rayleigh quotient $R(\mathbf{x}) := \mathbf{x}^T A\mathbf{x}$ is continuous on the unit sphere $S^{n-1}=\{\mathbf{x}:\|\mathbf{x}\|=1\}$, which is compact, so $R$ attains a maximum at some $\mathbf{v}_1\in S^{n-1}$ with value $\lambda_1$. By Lagrange multipliers (or by directly differentiating $R(\mathbf{v}_1+t\mathbf{w})$ along any tangent $\mathbf{w}\perp \mathbf{v}_1$), $A\mathbf{v}_1 = \lambda_1\mathbf{v}_1$. The eigenvalue is automatically real because $\lambda_1 = \mathbf{v}_1^T A\mathbf{v}_1\in\mathbb{R}$ and $\mathbf{v}_1$ is real. (Equivalently, for symmetric $A$, $\langle A\mathbf{x},\mathbf{x}\rangle = \langle \mathbf{x},A\mathbf{x}\rangle$ forces complex eigenvalues to be real: if $A\mathbf{z}=\mu\mathbf{z}$ over $\mathbb{C}$, then $\mu \overline{\mathbf{z}}^T\mathbf{z} = \overline{\mathbf{z}}^T A\mathbf{z} = (A\overline{\mathbf{z}})^T\mathbf{z} = \overline{\mu}\overline{\mathbf{z}}^T\mathbf{z}$, so $\mu=\overline{\mu}$.)

Let $W = \{\mathbf{v}_1\}^\perp$, an $(n-1)$-dimensional subspace. For $\mathbf{w}\in W$, $\langle A\mathbf{w},\mathbf{v}_1\rangle = \langle \mathbf{w},A\mathbf{v}_1\rangle = \lambda_1\langle \mathbf{w},\mathbf{v}_1\rangle = 0$, so $A$ maps $W\to W$. The restriction $A|_W$ is symmetric with respect to the inherited inner product. By induction, choose an orthonormal eigenbasis $\mathbf{v}_2,\dots,\mathbf{v}_n$ of $W$. Then $\mathbf{v}_1,\dots,\mathbf{v}_n$ is an orthonormal eigenbasis of $A$; assemble into $Q=[\mathbf{v}_1\,\cdots\,\mathbf{v}_n]$ and $\Lambda=\mathrm{diag}(\lambda_1,\dots,\lambda_n)$. $\square$

**Theorem (Existence of SVD).** Every $A\in\mathbb{R}^{m\times n}$ has an SVD.

*Sketch.* The matrix $A^T A$ is symmetric and PSD: $\mathbf{x}^T A^T A\mathbf{x} = \|A\mathbf{x}\|^2\geq 0$. By the spectral theorem, $A^T A = V\Lambda V^T$ with $V$ orthogonal and $\Lambda=\mathrm{diag}(\lambda_1,\dots,\lambda_n)$, $\lambda_i\geq 0$. Set $\sigma_i := \sqrt{\lambda_i}$ in decreasing order. For each $i$ with $\sigma_i>0$ define $\mathbf{u}_i := A\mathbf{v}_i/\sigma_i$. Then $\langle \mathbf{u}_i,\mathbf{u}_j\rangle = (\sigma_i\sigma_j)^{-1}\mathbf{v}_i^T A^T A\mathbf{v}_j = \delta_{ij}$, so $\{\mathbf{u}_i\}$ is orthonormal in $\mathbb{R}^m$; extend to an orthonormal basis $U$. By construction $A V = U\Sigma$, hence $A = U\Sigma V^T$. $\square$

**Theorem (Eckart–Young).** Let $A=U\Sigma V^T$ with singular values $\sigma_1\geq\cdots\geq\sigma_r>0$. The truncated SVD $A_k := \sum_{i=1}^k \sigma_i\mathbf{u}_i\mathbf{v}_i^T$ minimises $\|A-B\|_F$ (and $\|A-B\|_2$) over all rank-$k$ matrices $B$, with $\|A-A_k\|_F^2 = \sum_{i>k}\sigma_i^2$.

*Sketch.* Frobenius norm is unitarily invariant: $\|A-B\|_F = \|U^T(A-B)V\|_F$. Reducing to diagonal $\Sigma$, the problem becomes: among rank-$k$ matrices, minimise $\|\Sigma-C\|_F^2 = \sum (\sigma_i - c_{ii})^2 + \text{(off-diagonal)}^2$. Optimum sets the top-$k$ diagonal of $C$ equal to the top-$k$ singular values and zeroes the rest, recovering $A_k$. The operator-norm version uses Courant–Fischer / minimax. $\square$

## Code Sketch

The accompanying notebook (`cells.json`) numerically (i) verifies Cauchy–Schwarz on 50 random pairs in $\mathbb{R}^{10}$, (ii) computes a symmetric eigendecomposition via `np.linalg.eigh` and checks $A\mathbf{v}_i=\lambda_i\mathbf{v}_i$ and orthonormality of eigenvectors, (iii) computes an SVD with `np.linalg.svd` and reconstructs $A$, and (iv) shows the Frobenius error of rank-1 and rank-2 approximations decreasing monotonically as predicted by Eckart–Young.

## Connection to LLMs

Inner products and SVD are not abstract decoration; they are the geometric backbone of every transformer.

- **Attention** (Chapters 21–22). The score matrix $QK^T$ is the matrix of pairwise inner products between query and key embeddings. Long-context efficiency research (linear attention, Performers, low-rank attention) hinges on the empirical observation that $QK^T$ is approximately low-rank — which by Eckart–Young is best captured by truncated SVD-style factorisations.
- **PCA / interpretability.** SVD of an embedding matrix produces principal components; the leading singular vectors expose semantic axes (e.g. sentiment, syntax) and underpin probing experiments.
- **LoRA fine-tuning.** A pretrained weight $W_0\in\mathbb{R}^{m\times n}$ is updated as $W_0 + BA$ with $B\in\mathbb{R}^{m\times r}$, $A\in\mathbb{R}^{r\times n}$, $r\ll\min(m,n)$. This is *literally* a rank-$r$ correction; Eckart–Young guarantees it is the best Frobenius-norm approximation of the ideal full-rank update at that rank.
- **Spectral norm regularisation.** Bounding $\sigma_1(W)$ controls Lipschitz constants and stabilises training (Chapter on optimisation).

Every subsequent geometric statement in this book — projections, least squares, gradient flow on weight matrices, contractive maps for stability — descends from the inequalities and decompositions proven here.

<!-- CHAPTER 6 END -->

<!-- CHAPTER 7 START -->
<a id="chapter-7-convexity-and-optimization-gradient-descent-convergence"></a>
## Chapter 7: Convexity and optimization; gradient descent convergence

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

<!-- CHAPTER 7 END -->


# Block B — Probability and Information

<!-- CHAPTER 8 START -->
<a id="chapter-8-probability-foundations-sample-spaces-sigma-algebras-kolmogorov-axioms"></a>
## Chapter 8: Probability foundations: sample spaces, sigma-algebras, Kolmogorov axioms

## Motivation

Every modern language model is, at its core, a probability distribution over sequences of tokens. To even *write down* the statement "the model assigns probability $p$ to token $t$ given context $c$," we need a rigorous notion of what a probability is, what it acts on, and what consistency rules it must obey. Naive frequentist intuition (probability is "long-run frequency") is suggestive but mathematically inadequate: it cannot, on its own, tell us why probabilities of disjoint events should add, why conditional probability is well-defined, or what it means to take limits of sequences of events. Kolmogorov's 1933 axiomatization solves this once and for all by reducing probability to *measure theory on a $\sigma$-algebra*. This chapter builds that machinery from set theory (Chapter 1) up through the workhorse identities — inclusion–exclusion, the union bound, Bayes' theorem, the law of total probability, and continuity of measure — that recur in every later chapter on training and inference.

## Definitions

**Sample space and outcomes.** A *sample space* $\Omega$ is a non-empty set; its elements $\omega \in \Omega$ are *outcomes*. An *event* is a subset $A \subseteq \Omega$. For two coin flips, $\Omega = \{HH, HT, TH, TT\}$ and "at least one head" is the event $\{HH, HT, TH\}$.

**$\sigma$-algebra.** A collection $\mathcal{F} \subseteq 2^\Omega$ is a *$\sigma$-algebra* on $\Omega$ if:

1. $\Omega \in \mathcal{F}$;
2. $A \in \mathcal{F} \implies A^c \in \mathcal{F}$ (closure under complements);
3. $A_1, A_2, \ldots \in \mathcal{F} \implies \bigcup_{n=1}^\infty A_n \in \mathcal{F}$ (closure under countable unions).

By De Morgan (Chapter 1), countable intersections are also in $\mathcal{F}$. The mirror to set algebra is exact: $\sigma$-algebras are precisely the set-algebraic structures stable under the "infinite" operations probability needs.

**Probability measure.** A function $\mathbb{P}: \mathcal{F} \to [0,1]$ is a *probability measure* if:

- (K1, non-negativity) $\mathbb{P}(A) \geq 0$ for all $A \in \mathcal{F}$;
- (K2, normalization) $\mathbb{P}(\Omega) = 1$;
- (K3, countable additivity) for pairwise disjoint $A_1, A_2, \ldots \in \mathcal{F}$, $\mathbb{P}\!\left(\bigsqcup_n A_n\right) = \sum_n \mathbb{P}(A_n)$.

The triple $(\Omega, \mathcal{F}, \mathbb{P})$ is a *probability space*.

**Conditional probability and independence.** For $B \in \mathcal{F}$ with $\mathbb{P}(B) > 0$,
$$\mathbb{P}(A \mid B) := \frac{\mathbb{P}(A \cap B)}{\mathbb{P}(B)}.$$
Events $A, B$ are *independent* iff $\mathbb{P}(A \cap B) = \mathbb{P}(A)\mathbb{P}(B)$. Equivalently, when $\mathbb{P}(B) > 0$, iff $\mathbb{P}(A \mid B) = \mathbb{P}(A)$ — conditioning on $B$ leaves our belief about $A$ unchanged.

## Theorems and Proofs

**Lemma (finite additivity, complement, monotonicity).** From (K2) and (K3) with $A_1 = \Omega$, $A_2 = A_3 = \cdots = \emptyset$ disjoint and $\Omega = \Omega \sqcup \emptyset \sqcup \emptyset \sqcup \cdots$, we get $1 = 1 + \sum_{n \geq 2}\mathbb{P}(\emptyset)$, forcing $\mathbb{P}(\emptyset) = 0$. Setting $A_3 = A_4 = \cdots = \emptyset$ in (K3) yields *finite additivity*: for disjoint $A, B$, $\mathbb{P}(A \sqcup B) = \mathbb{P}(A) + \mathbb{P}(B)$. Applying this to $\Omega = A \sqcup A^c$ gives $\mathbb{P}(A^c) = 1 - \mathbb{P}(A)$. If $A \subseteq B$, write $B = A \sqcup (B \setminus A)$; finite additivity and (K1) give $\mathbb{P}(A) \leq \mathbb{P}(B)$ (*monotonicity*).

**Theorem (inclusion–exclusion, two events).** $\mathbb{P}(A \cup B) = \mathbb{P}(A) + \mathbb{P}(B) - \mathbb{P}(A \cap B)$.

*Proof.* Decompose into three disjoint pieces: $A \cup B = (A \setminus B) \sqcup (B \setminus A) \sqcup (A \cap B)$, and $A = (A \setminus B) \sqcup (A \cap B)$, $B = (B \setminus A) \sqcup (A \cap B)$. Apply finite additivity to each: $\mathbb{P}(A) + \mathbb{P}(B) = \mathbb{P}(A \setminus B) + \mathbb{P}(B \setminus A) + 2\mathbb{P}(A \cap B) = \mathbb{P}(A \cup B) + \mathbb{P}(A \cap B)$. Rearrange. $\blacksquare$

**Theorem (union bound / Boole's inequality).** For any countable family $\{A_n\}_{n \geq 1} \subseteq \mathcal{F}$, $\mathbb{P}\!\left(\bigcup_n A_n\right) \leq \sum_n \mathbb{P}(A_n)$.

*Proof.* Disjointify: set $B_1 = A_1$ and $B_n = A_n \setminus \bigcup_{k<n} A_k$ for $n \geq 2$. Each $B_n \in \mathcal{F}$ (closure under complements and countable unions), the $B_n$ are pairwise disjoint, $B_n \subseteq A_n$, and $\bigsqcup_n B_n = \bigcup_n A_n$. By (K3) and monotonicity, $\mathbb{P}(\bigcup_n A_n) = \sum_n \mathbb{P}(B_n) \leq \sum_n \mathbb{P}(A_n)$. $\blacksquare$

**Theorem (Bayes).** If $\mathbb{P}(A), \mathbb{P}(B) > 0$, $\mathbb{P}(A \mid B) = \dfrac{\mathbb{P}(B \mid A)\, \mathbb{P}(A)}{\mathbb{P}(B)}$.

*Proof.* By definition, $\mathbb{P}(A \mid B)\mathbb{P}(B) = \mathbb{P}(A \cap B) = \mathbb{P}(B \cap A) = \mathbb{P}(B \mid A)\mathbb{P}(A)$. Divide by $\mathbb{P}(B)$. $\blacksquare$

**Theorem (law of total probability).** If $\{B_i\}_{i \in I}$ is a countable partition of $\Omega$ with each $\mathbb{P}(B_i) > 0$, then for every $A \in \mathcal{F}$, $\mathbb{P}(A) = \sum_i \mathbb{P}(A \mid B_i)\mathbb{P}(B_i)$.

*Proof.* $A = A \cap \Omega = A \cap \bigsqcup_i B_i = \bigsqcup_i (A \cap B_i)$, a disjoint union. By (K3), $\mathbb{P}(A) = \sum_i \mathbb{P}(A \cap B_i) = \sum_i \mathbb{P}(A \mid B_i)\mathbb{P}(B_i)$. $\blacksquare$

**Theorem (continuity of measure).** Let $A_n \in \mathcal{F}$.

- *(From below)* If $A_1 \subseteq A_2 \subseteq \cdots$ and $A = \bigcup_n A_n$, then $\mathbb{P}(A_n) \uparrow \mathbb{P}(A)$.
- *(From above)* If $A_1 \supseteq A_2 \supseteq \cdots$ and $A = \bigcap_n A_n$, then $\mathbb{P}(A_n) \downarrow \mathbb{P}(A)$.

*Proof (below).* Set $B_1 = A_1$ and $B_n = A_n \setminus A_{n-1}$ for $n \geq 2$; the $B_n$ are disjoint, $\bigsqcup_{k \leq n} B_k = A_n$, and $\bigsqcup_k B_k = A$. By (K3), $\mathbb{P}(A) = \sum_{k=1}^\infty \mathbb{P}(B_k) = \lim_n \sum_{k=1}^n \mathbb{P}(B_k) = \lim_n \mathbb{P}(A_n)$.

*(Above)* Apply the previous case to the increasing sequence $A_1 \setminus A_n$: it has union $A_1 \setminus A$, so $\mathbb{P}(A_1) - \mathbb{P}(A_n) \to \mathbb{P}(A_1) - \mathbb{P}(A)$, giving $\mathbb{P}(A_n) \to \mathbb{P}(A)$. (Here finiteness of $\mathbb{P}(A_1) \leq 1$ lets us subtract; this is why "from above" requires some $A_n$ of finite measure — automatic for probability measures.) $\blacksquare$

## Code sketch

We instantiate the smallest non-trivial probability space: two fair dice on $\Omega = \{1, \ldots, 6\}^2$, $\mathcal{F} = 2^\Omega$, $\mathbb{P}(A) = |A|/36$. The notebook verifies (K2), finite additivity on disjoint events, two-event inclusion–exclusion, the union bound on three events, and runs a Bayes computation on a disease-testing problem both analytically and via Monte Carlo.

## Connection to LLMs

A causal language model (Chapter 25) defines, for each context $c$, a probability measure on the discrete sample space $\Omega = \mathcal{V}$ (the vocabulary), with $\mathcal{F} = 2^{\mathcal{V}}$ and $\mathbb{P}(\{t\} \mid c) = p_\theta(t \mid c)$. The softmax output is exactly a Kolmogorov probability measure: non-negative (K1), normalized to one by construction (K2), and finitely additive on disjoint token sets (K3, vacuously countable since $\mathcal{V}$ is finite). Sampling, beam search, nucleus filtering, and importance-weighted training all reduce to operations on this measure. The chain rule $\mathbb{P}(t_1, \ldots, t_n) = \prod_i \mathbb{P}(t_i \mid t_{<i})$ is just iterated conditioning. Bayes' theorem reappears whenever we invert a generative model into a posterior over latents (alignment, RLHF reward modeling). Continuity of measure is what licenses limiting arguments — e.g., "the probability that the model ever emits a forbidden token in an infinite generation" — that we will need in safety analyses.

<!-- CHAPTER 8 END -->

<!-- CHAPTER 9 START -->
<a id="chapter-9-random-variables-distributions-cdfpmfpdf"></a>
## Chapter 9: Random variables, distributions, CDF/PMF/PDF

## Motivation

Chapter 8 built probability spaces $(\Omega, \mathcal{F}, \mathbb{P})$ as the bedrock for modeling uncertainty. But raw $\Omega$ is rarely what we *measure*: we measure a temperature, a token id, a pixel intensity. A **random variable** is the bridge that turns abstract outcomes into numbers we can integrate, sum, optimize, and—most importantly for us—differentiate. Every loss function in deep learning is an expectation over a random variable; every sampler in a language model is a draw from a distribution; every softmax is a categorical PMF. This chapter formalizes the machinery.

## Definitions

**Definition (Random variable).** Let $(\Omega, \mathcal{F}, \mathbb{P})$ be a probability space. A function $X : \Omega \to \mathbb{R}$ is a *random variable* if for every $x \in \mathbb{R}$,
$$\{X \leq x\} := \{\omega \in \Omega : X(\omega) \leq x\} \in \mathcal{F}.$$
This *measurability* condition guarantees that probabilities of events defined through $X$ are well-defined.

**Definition (Distribution).** The *distribution* (or *law*) of $X$ is the pushforward measure $\mu_X$ on $(\mathbb{R}, \mathcal{B}(\mathbb{R}))$ defined by
$$\mu_X(B) = \mathbb{P}(X \in B), \qquad B \in \mathcal{B}(\mathbb{R}).$$

**Definition (CDF).** The *cumulative distribution function* of $X$ is $F_X : \mathbb{R} \to [0,1]$,
$$F_X(x) = \mathbb{P}(X \leq x) = \mu_X((-\infty, x]).$$

**Definition (Discrete RV / PMF).** $X$ is *discrete* if it takes values in a countable set $S \subset \mathbb{R}$. The *probability mass function* is $p_X(x) = \mathbb{P}(X = x)$ for $x \in S$.

**Definition (Continuous RV / PDF).** $X$ is *(absolutely) continuous* if there exists a non-negative measurable $f_X : \mathbb{R} \to [0,\infty)$, the *probability density function*, such that
$$\mathbb{P}(X \in A) = \int_A f_X(x)\, dx \quad \text{for every Borel } A.$$

**Standard families.** Bernoulli$(p)$: $p_X(1) = p$, $p_X(0) = 1-p$. Binomial$(n,p)$: $p_X(k) = \binom{n}{k} p^k (1-p)^{n-k}$. Categorical$(\pi_1,\dots,\pi_K)$: $p_X(k) = \pi_k$ with $\sum \pi_k = 1$. Geometric$(p)$: $p_X(k) = (1-p)^{k-1} p$, $k \geq 1$. Uniform$[a,b]$: $f_X(x) = \frac{1}{b-a} \mathbf{1}_{[a,b]}(x)$. Gaussian $\mathcal{N}(\mu,\sigma^2)$: $f_X(x) = \frac{1}{\sqrt{2\pi}\sigma} \exp(-(x-\mu)^2/(2\sigma^2))$.

**Joint, marginal, conditional.** For $(X, Y)$ on the same space, the *joint* distribution lives on $\mathbb{R}^2$. *Marginals* are obtained by integrating/summing out: $f_X(x) = \int f_{X,Y}(x,y)\, dy$. The *conditional* density is $f_{Y \mid X}(y \mid x) = f_{X,Y}(x,y) / f_X(x)$ when $f_X(x) > 0$.

## Theorems

**Theorem 9.1 (Properties of CDF).** $F_X$ is (i) non-decreasing, (ii) right-continuous, (iii) $\lim_{x \to -\infty} F_X(x) = 0$, (iv) $\lim_{x \to \infty} F_X(x) = 1$.

*Proof.* (i) If $x \leq y$, then $\{X \leq x\} \subseteq \{X \leq y\}$, so monotonicity of $\mathbb{P}$ gives $F_X(x) \leq F_X(y)$.

(ii) Fix $x$. Let $x_n \downarrow x$. Then $\{X \leq x_n\} \downarrow \{X \leq x\}$ (intersection over $n$). By the *continuity of measure from above* (Chapter 8, Theorem on monotone convergence of measures), since $\mathbb{P}(\{X \leq x_1\}) \leq 1 < \infty$,
$$F_X(x_n) = \mathbb{P}(X \leq x_n) \to \mathbb{P}(X \leq x) = F_X(x).$$

(iii) Take $x_n \downarrow -\infty$. Then $\{X \leq x_n\} \downarrow \emptyset$, so $F_X(x_n) \to \mathbb{P}(\emptyset) = 0$.

(iv) Take $x_n \uparrow \infty$. Then $\{X \leq x_n\} \uparrow \Omega$, so by continuity from below, $F_X(x_n) \to \mathbb{P}(\Omega) = 1$. $\blacksquare$

**Theorem 9.2 (Normalization).** $\sum_{x \in S} p_X(x) = 1$ (discrete) and $\int_{\mathbb{R}} f_X(x)\, dx = 1$ (continuous).

*Proof.* The events $\{X = x\}$, $x \in S$, are disjoint and their union is $\{X \in S\} = \Omega$ (since $X$ is $S$-valued). Countable additivity gives $1 = \mathbb{P}(\Omega) = \sum_{x \in S} \mathbb{P}(X = x) = \sum_x p_X(x)$. The continuous case: take $A = \mathbb{R}$ in the defining property: $1 = \mathbb{P}(X \in \mathbb{R}) = \int_\mathbb{R} f_X$. $\blacksquare$

**Theorem 9.3 (Change of variables, 1-D).** Let $X$ have density $f_X$, and let $g: \mathbb{R} \to \mathbb{R}$ be strictly monotone and $C^1$ on the support of $X$. Then $Y = g(X)$ has density
$$f_Y(y) = f_X(g^{-1}(y)) \,\bigl|(g^{-1})'(y)\bigr|.$$

*Proof.* Suppose $g$ is strictly increasing (decreasing case is symmetric). For any $y$,
$$F_Y(y) = \mathbb{P}(g(X) \leq y) = \mathbb{P}(X \leq g^{-1}(y)) = F_X(g^{-1}(y)).$$
Differentiating using the chain rule (Chapter 3):
$$f_Y(y) = \frac{d}{dy} F_X(g^{-1}(y)) = f_X(g^{-1}(y)) \cdot (g^{-1})'(y).$$
Since $g^{-1}$ is increasing, $(g^{-1})'(y) > 0$, so the absolute value is automatic. For decreasing $g$, $\{g(X) \leq y\} = \{X \geq g^{-1}(y)\}$, $F_Y(y) = 1 - F_X(g^{-1}(y))$, and differentiation yields $-f_X(g^{-1}(y)) (g^{-1})'(y)$, with $(g^{-1})' < 0$, again giving the absolute value. $\blacksquare$

**Theorem 9.4 (Inverse-CDF sampling).** Let $F$ be a CDF with generalized inverse $F^{-1}(u) = \inf\{x : F(x) \geq u\}$. If $U \sim \mathrm{Uniform}[0,1]$, then $X := F^{-1}(U)$ has CDF $F$.

*Proof.* It suffices to show $\{F^{-1}(U) \leq x\} = \{U \leq F(x)\}$ (up to a null set). If $F^{-1}(u) \leq x$, then by definition of infimum and right-continuity of $F$, $F(x) \geq u$. Conversely, if $u \leq F(x)$, then $x \in \{x' : F(x') \geq u\}$, so $F^{-1}(u) \leq x$. Therefore
$$\mathbb{P}(X \leq x) = \mathbb{P}(U \leq F(x)) = F(x),$$
using that $U$ is uniform on $[0,1]$. $\blacksquare$

## Code sketch

The notebook implements: (1) a 5-class categorical from softmax of fixed logits, with PMF/CDF and inverse-CDF sampling against `np.random.seed(0)`; (2) a numerical Gaussian density with Riemann-sum normalization and CDF; (3) the change-of-variable check for $Y = -\ln X$, $X \sim U[0,1]$, yielding $Y \sim \mathrm{Exp}(1)$; (4) a final inverse-CDF sampler convergence test.

## Connection to LLMs

A causal language model outputs logits $z_t \in \mathbb{R}^V$ at each step, which softmax maps to a categorical distribution
$$p_\theta(x_t \mid x_{<t}) = \mathrm{softmax}(z_t).$$
This is a discrete RV over the vocabulary; greedy decoding picks $\arg\max$, while temperature/nucleus sampling draws from this categorical. The standard implementation is *exactly* inverse-CDF sampling on the cumulative softmax (Theorem 9.4), or the equivalent **Gumbel-max trick** $\arg\max_k(z_k + G_k)$ with $G_k \sim \mathrm{Gumbel}(0,1)$. Cross-entropy loss (Chapter 17) is $-\log p_\theta(x_t \mid x_{<t})$, an expectation under the data distribution; the causal LM training objective (Chapter 25) is the joint log-likelihood factored by the chain rule for conditionals defined here.

<!-- CHAPTER 9 END -->

<!-- CHAPTER 10 START -->
<a id="chapter-10-expectation-variance-covariance-jensens-inequality"></a>
## Chapter 10: Expectation, variance, covariance; Jensen's inequality

## Motivation

Random variables (Chapter 9) describe uncertain outcomes. To compress an entire distribution into a single representative number, we use the **expectation**. Two further numbers — **variance** and **covariance** — describe spread and joint linear association. These three quantities, together with **Jensen's inequality**, are the structural backbone of every loss function and convergence proof in modern deep learning. Training an LLM is, formally, the minimization of an expectation $\mathbb{E}_{x \sim \mathcal{D}}[\ell(\theta; x)]$; SGD is a Monte Carlo estimator of that expectation; the variance of the estimator dictates convergence speed (we will revisit this in Chapter 13).

## Definitions

**Definition 10.1 (Expectation).** For a discrete random variable $X$ with pmf $p_X$,
$$\mathbb{E}[X] = \sum_x x\,p_X(x),$$
provided $\sum_x |x|\,p_X(x) < \infty$. For a continuous $X$ with density $f_X$,
$$\mathbb{E}[X] = \int_{\mathbb{R}} x\,f_X(x)\,dx,$$
provided the integral is absolutely convergent. In full generality, $\mathbb{E}[X] := \int_\Omega X\,d\mathbb{P}$, the Lebesgue integral of $X$ against the probability measure (we cite without proof; see Billingsley, *Probability and Measure*).

**Definition 10.2 (Variance).** $\mathrm{Var}(X) := \mathbb{E}[(X - \mathbb{E}[X])^2]$, when the expectation exists. The standard deviation is $\sigma_X := \sqrt{\mathrm{Var}(X)}$.

**Definition 10.3 (Covariance, correlation).** For $X,Y$ with finite variance,
$$\mathrm{Cov}(X,Y) := \mathbb{E}\big[(X - \mathbb{E}X)(Y - \mathbb{E}Y)\big], \qquad \rho_{X,Y} := \frac{\mathrm{Cov}(X,Y)}{\sigma_X \sigma_Y} \in [-1, 1].$$
The bound $|\rho| \le 1$ is the Cauchy–Schwarz inequality applied to the inner product $\langle U, V\rangle := \mathbb{E}[UV]$.

**Definition 10.4 (Conditional expectation).** For discrete $X, Y$ and any $y$ with $p_Y(y) > 0$,
$$\mathbb{E}[X \mid Y = y] = \sum_x x\,p_{X\mid Y}(x\mid y).$$
Viewed as $y$ varies, $\mathbb{E}[X\mid Y]$ is itself a random variable: a measurable function of $Y$. The general (Lebesgue) definition characterizes $\mathbb{E}[X\mid \mathcal{G}]$ as the unique (a.s.) $\mathcal{G}$-measurable random variable satisfying $\int_A \mathbb{E}[X\mid \mathcal{G}]\,d\mathbb{P} = \int_A X\,d\mathbb{P}$ for every $A \in \mathcal{G}$ (Radon–Nikodym).

## Theorems with proofs

**Theorem 10.5 (Linearity of expectation).** For random variables $X, Y$ with finite expectation and scalars $a, b \in \mathbb{R}$,
$$\mathbb{E}[aX + bY] = a\,\mathbb{E}[X] + b\,\mathbb{E}[Y].$$
*Independence is not required.*

*Proof.* In the discrete case, write the joint pmf $p_{X,Y}$. Then
$$\mathbb{E}[aX + bY] = \sum_{x,y} (ax + by)\,p_{X,Y}(x,y) = a\sum_{x,y} x\,p_{X,Y}(x,y) + b\sum_{x,y} y\,p_{X,Y}(x,y).$$
Marginalizing, $\sum_y p_{X,Y}(x,y) = p_X(x)$ and $\sum_x p_{X,Y}(x,y) = p_Y(y)$, giving $a\,\mathbb{E}[X] + b\,\mathbb{E}[Y]$. The continuous case is identical with sums replaced by integrals; in full generality, linearity is a property of the Lebesgue integral. $\blacksquare$

**Theorem 10.6 (Variance of a sum).**
$$\mathrm{Var}(X + Y) = \mathrm{Var}(X) + \mathrm{Var}(Y) + 2\,\mathrm{Cov}(X, Y).$$

*Proof.* Let $\mu_X = \mathbb{E}[X]$, $\mu_Y = \mathbb{E}[Y]$. By linearity, $\mathbb{E}[X+Y] = \mu_X + \mu_Y$. Then
\begin{align*}
\mathrm{Var}(X+Y) &= \mathbb{E}\big[((X-\mu_X) + (Y-\mu_Y))^2\big] \\
&= \mathbb{E}[(X-\mu_X)^2] + \mathbb{E}[(Y-\mu_Y)^2] + 2\,\mathbb{E}[(X-\mu_X)(Y-\mu_Y)] \\
&= \mathrm{Var}(X) + \mathrm{Var}(Y) + 2\,\mathrm{Cov}(X,Y). \qquad \blacksquare
\end{align*}

If $X \perp Y$, then $\mathrm{Cov}(X,Y) = 0$ and variances add. The converse is false in general.

**Theorem 10.7 (Jensen's inequality).** Let $\phi : \mathbb{R} \to \mathbb{R}$ be convex and $X$ a random variable with $\mathbb{E}|X| < \infty$ and $\mathbb{E}|\phi(X)| < \infty$. Then
$$\phi(\mathbb{E}[X]) \le \mathbb{E}[\phi(X)].$$

*Proof.* Recall (Chapter 7) that a convex function on $\mathbb{R}$ admits a *supporting line* at every interior point of its domain: for $x_0 = \mathbb{E}[X]$ there exists a subgradient $g \in \partial\phi(x_0)$ such that
$$\phi(x) \ge \phi(x_0) + g(x - x_0) \quad \text{for all } x.$$
Substitute $X$ and take expectations. By linearity,
$$\mathbb{E}[\phi(X)] \ge \phi(x_0) + g(\mathbb{E}[X] - x_0) = \phi(\mathbb{E}[X]),$$
since $\mathbb{E}[X] - x_0 = 0$. $\blacksquare$

**Theorem 10.8 (Markov's inequality).** Let $X \ge 0$ a.s. and $a > 0$. Then
$$\mathbb{P}(X \ge a) \le \frac{\mathbb{E}[X]}{a}.$$

*Proof.* Pointwise, $a\cdot \mathbf{1}_{\{X \ge a\}} \le X$ (when $X \ge a$, both sides equal $\le X$; when $X < a$, the left side is $0$ and $X \ge 0$). Take expectations: $a\,\mathbb{P}(X \ge a) \le \mathbb{E}[X]$. Divide by $a$. $\blacksquare$

**Theorem 10.9 (Chebyshev's inequality).** For $X$ with $\mathbb{E}[X] = \mu$ and finite $\sigma^2 = \mathrm{Var}(X)$, and $k > 0$,
$$\mathbb{P}(|X - \mu| \ge k\sigma) \le \frac{1}{k^2}.$$

*Proof.* Apply Markov to $Y := (X - \mu)^2 \ge 0$ with $a = k^2 \sigma^2$:
$$\mathbb{P}((X-\mu)^2 \ge k^2\sigma^2) \le \frac{\mathbb{E}[(X-\mu)^2]}{k^2 \sigma^2} = \frac{1}{k^2}.$$
The event $\{(X-\mu)^2 \ge k^2\sigma^2\}$ equals $\{|X-\mu| \ge k\sigma\}$. $\blacksquare$

**Theorem 10.10 (Weak law of large numbers).** Let $X_1, X_2, \dots$ be i.i.d. with $\mathbb{E}[X_i] = \mu$ and $\mathrm{Var}(X_i) = \sigma^2 < \infty$. Set $\bar{X}_n := \frac{1}{n}\sum_{i=1}^n X_i$. Then for every $\epsilon > 0$,
$$\mathbb{P}(|\bar{X}_n - \mu| \ge \epsilon) \xrightarrow{n\to\infty} 0.$$

*Proof.* By linearity, $\mathbb{E}[\bar{X}_n] = \mu$. By Theorem 10.6 and independence, $\mathrm{Var}(\bar{X}_n) = \sigma^2/n$. Chebyshev gives
$$\mathbb{P}(|\bar{X}_n - \mu| \ge \epsilon) \le \frac{\sigma^2}{n\epsilon^2} \to 0. \qquad \blacksquare$$

## Code sketch

The accompanying notebook (`cells.json`) verifies each theorem on small distributions: a 5-outcome categorical for $\mathbb{E}, \mathrm{Var}$; a perfectly dependent pair $(X, 2X+1)$ for linearity without independence; the convex $\phi(x)=e^x$ on Uniform $\{-1,+1\}$ for Jensen; and a Monte Carlo experiment showing $\bar X_n \to 1/2$ inside the Chebyshev band for $X_i \sim \mathrm{Unif}[0,1]$.

## Connection to LLMs

The training objective of every modern language model has the form
$$\mathcal{L}(\theta) = \mathbb{E}_{x \sim \mathcal{D}}[\ell(\theta; x)],$$
where $\mathcal{D}$ is the data distribution and $\ell$ is (typically) the next-token cross-entropy loss. The data distribution is intractable, so we replace the expectation by an empirical mean over a mini-batch of size $B$:
$$\hat{\mathcal{L}}(\theta) = \frac{1}{B}\sum_{i=1}^B \ell(\theta; x_i).$$
Linearity of expectation guarantees $\mathbb{E}[\nabla\hat{\mathcal{L}}] = \nabla \mathcal{L}$ (the SGD gradient is unbiased). The variance of $\nabla \hat{\mathcal{L}}$ scales as $1/B$ (Theorem 10.6 plus independence within a batch), which is exactly why larger batches give smoother training curves. The weak LLN tells us that as $B \to \infty$ we recover the true loss in probability — this is the formal content of "more data helps" (revisited rigorously in Chapter 13). Jensen's inequality, finally, justifies the variational lower bound underlying every modern likelihood-based generative model: $\log \mathbb{E}[Z] \ge \mathbb{E}[\log Z]$ for $Z > 0$, which we will use in the ELBO derivations of Chapter 22.

<!-- CHAPTER 10 END -->

<!-- CHAPTER 11 START -->
<a id="chapter-11-information-theory-self-information-entropy-cross-entropy-kl"></a>
## Chapter 11: Information theory: self-information, entropy, cross-entropy, KL

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

<!-- CHAPTER 11 END -->

<!-- CHAPTER 12 START -->
<a id="chapter-12-statistical-inference-likelihood-mle-erm-bias-variance"></a>
## Chapter 12: Statistical inference: likelihood, MLE, ERM, bias-variance

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

<!-- CHAPTER 12 END -->


# Block C — Stochastic Optimization

<!-- CHAPTER 13 START -->
<a id="chapter-13-sgd-stochastic-approximation-theorem-mini-batching-convergence-sketch"></a>
## Chapter 13: SGD: stochastic-approximation theorem; mini-batching; convergence sketch

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

<!-- CHAPTER 13 END -->

<!-- CHAPTER 14 START -->
<a id="chapter-14-momentum-rmsprop-adamw-derivation-and-bias-correction-proof"></a>
## Chapter 14: Momentum, RMSProp, AdamW: derivation and bias-correction proof

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

<!-- CHAPTER 14 END -->


# Block D — Neural Networks

<!-- CHAPTER 15 START -->
## Chapter 15: MLPs as compositional functions; universal approximation

_(This chapter is currently a stub. It will contain Motivation, Definitions, Theorems and proofs, and Code and demonstration sections.)_

<!-- CHAPTER 15 END -->

<!-- CHAPTER 16 START -->
## Chapter 16: Activation functions: ReLU/GELU/softmax with derivatives

_(This chapter is currently a stub. It will contain Motivation, Definitions, Theorems and proofs, and Code and demonstration sections.)_

<!-- CHAPTER 16 END -->

<!-- CHAPTER 17 START -->
## Chapter 17: Loss functions: MSE, cross-entropy; gradients from first principles

_(This chapter is currently a stub. It will contain Motivation, Definitions, Theorems and proofs, and Code and demonstration sections.)_

<!-- CHAPTER 17 END -->

<!-- CHAPTER 18 START -->
## Chapter 18: Backpropagation: chain rule applied; reverse-mode AD as a graph algorithm

_(This chapter is currently a stub. It will contain Motivation, Definitions, Theorems and proofs, and Code and demonstration sections.)_

<!-- CHAPTER 18 END -->


# Block E — Sequence Models and Attention

<!-- CHAPTER 19 START -->
## Chapter 19: Embeddings: token to vector; lookup as a linear map; weight tying

_(This chapter is currently a stub. It will contain Motivation, Definitions, Theorems and proofs, and Code and demonstration sections.)_

<!-- CHAPTER 19 END -->

<!-- CHAPTER 20 START -->
## Chapter 20: RNN intuition; vanishing-gradient proof; why we need attention

_(This chapter is currently a stub. It will contain Motivation, Definitions, Theorems and proofs, and Code and demonstration sections.)_

<!-- CHAPTER 20 END -->

<!-- CHAPTER 21 START -->
## Chapter 21: Scaled dot-product attention: derivation, softmax-temperature analysis

_(This chapter is currently a stub. It will contain Motivation, Definitions, Theorems and proofs, and Code and demonstration sections.)_

<!-- CHAPTER 21 END -->

<!-- CHAPTER 22 START -->
## Chapter 22: Multi-head attention: parallel heads as concat-then-project; complexity

_(This chapter is currently a stub. It will contain Motivation, Definitions, Theorems and proofs, and Code and demonstration sections.)_

<!-- CHAPTER 22 END -->

<!-- CHAPTER 23 START -->
## Chapter 23: Transformer block: residual + LayerNorm/RMSNorm + FFN + attention; gradient-flow argument

_(This chapter is currently a stub. It will contain Motivation, Definitions, Theorems and proofs, and Code and demonstration sections.)_

<!-- CHAPTER 23 END -->

<!-- CHAPTER 24 START -->
## Chapter 24: Positional encoding: sinusoidal derivation, RoPE construction

_(This chapter is currently a stub. It will contain Motivation, Definitions, Theorems and proofs, and Code and demonstration sections.)_

<!-- CHAPTER 24 END -->


# Block F — Pre-training

<!-- CHAPTER 25 START -->
## Chapter 25: Causal masking; next-token prediction loss as MLE on the empirical distribution

_(This chapter is currently a stub. It will contain Motivation, Definitions, Theorems and proofs, and Code and demonstration sections.)_

<!-- CHAPTER 25 END -->

<!-- CHAPTER 26 START -->
## Chapter 26: Tokenization: BPE algorithm; greedy merge correctness

_(This chapter is currently a stub. It will contain Motivation, Definitions, Theorems and proofs, and Code and demonstration sections.)_

<!-- CHAPTER 26 END -->

<!-- CHAPTER 27 START -->
## Chapter 27: Pre-training pipeline: AdamW + warmup + cosine decay + gradient clipping; tiny-GPT training run

_(This chapter is currently a stub. It will contain Motivation, Definitions, Theorems and proofs, and Code and demonstration sections.)_

<!-- CHAPTER 27 END -->


# Block G — Post-training

<!-- CHAPTER 28 START -->
## Chapter 28: SFT, RLHF (PPO/GRPO), and DPO; train + post-train a tiny GPT

_(This chapter is currently a stub. It will contain Motivation, Definitions, Theorems and proofs, and Code and demonstration sections.)_

<!-- CHAPTER 28 END -->
