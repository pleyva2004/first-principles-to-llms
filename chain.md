# From First Principles to Modern Large Language Models

**Author:** Pablo Leyva

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

### H — Reinforcement Learning
- [Chapter 29: MDP foundations: Bellman equations, value iteration, tabular Q-learning](#chapter-29-mdp-foundations-bellman-equations-value-iteration-tabular-q-learning)
- [Chapter 30: Value-based deep RL: function approximation, DQN, max-entropy framework](#chapter-30-value-based-deep-rl-function-approximation-dqn-max-entropy-framework)
- [Chapter 31: Policy gradient, GRPO, and the RLHF/DPO bridge: tiny-GPT alignment loop](#chapter-31-policy-gradient-grpo-and-the-rlhfdpo-bridge-tiny-gpt-alignment-loop)

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
<a id="chapter-15-mlps-as-compositional-functions-universal-approximation"></a>
## Chapter 15: MLPs as compositional functions; universal approximation

## Motivation

A linear map $x \mapsto Wx + b$ (Chapter 5) can only carve $\mathbb{R}^n$ into half-spaces and stretch them affinely. The continuous functions on a compact set $K \subset \mathbb{R}^n$ (Chapter 4) form a vastly richer space. To bridge the gap we interleave linear maps with a fixed pointwise nonlinearity. The resulting object — a *multilayer perceptron* (MLP) — turns out to be expressive enough to approximate **every** continuous function uniformly on $K$. This is the engine inside every transformer's feed-forward block.

## Definitions

**Definition (MLP).** Let $L \geq 1$ and widths $d_0, d_1, \dots, d_L \in \mathbb{N}$. A *multilayer perceptron of depth $L$* with activation $\sigma : \mathbb{R} \to \mathbb{R}$ (applied componentwise) is the function $f : \mathbb{R}^{d_0} \to \mathbb{R}^{d_L}$ defined by
$$
h^{(0)} = x, \qquad h^{(\ell)} = \sigma\!\big(W^{(\ell)} h^{(\ell-1)} + b^{(\ell)}\big) \text{ for } 1 \leq \ell \leq L-1,
$$
$$
f(x) = W^{(L)} h^{(L-1)} + b^{(L)},
$$
with $W^{(\ell)} \in \mathbb{R}^{d_\ell \times d_{\ell-1}}$ and $b^{(\ell)} \in \mathbb{R}^{d_\ell}$. The number $\max_\ell d_\ell$ is the *width*; $L$ is the *depth*.

**Definition (universal approximator).** A class $\mathcal{F} \subset C(K, \mathbb{R})$ is **dense** in $C(K)$ — i.e. a *universal approximator* — if for every $f \in C(K)$ and $\varepsilon > 0$ there exists $g \in \mathcal{F}$ with $\sup_{x \in K} |f(x) - g(x)| < \varepsilon$.

**Definition (sigmoidal / discriminatory).** A measurable $\sigma : \mathbb{R} \to \mathbb{R}$ is *sigmoidal* if $\sigma(t) \to 1$ as $t \to +\infty$ and $\sigma(t) \to 0$ as $t \to -\infty$. It is *discriminatory* if for every signed regular Borel measure $\mu$ on $K = [0,1]^n$,
$$
\int_K \sigma(w^\top x + b)\, d\mu(x) = 0 \text{ for all } w \in \mathbb{R}^n,\ b \in \mathbb{R} \;\Longrightarrow\; \mu = 0.
$$

## Theorems

### 1. Universal Approximation (Cybenko, 1989; Hornik, 1989)

**Theorem.** Let $\sigma$ be a continuous sigmoidal function and $K \subset \mathbb{R}^n$ compact. The class
$$
\mathcal{F}_\sigma = \Big\{ g(x) = \sum_{j=1}^{N} \alpha_j\, \sigma(w_j^\top x + b_j) \;\Big|\; N \in \mathbb{N},\ \alpha_j, b_j \in \mathbb{R},\ w_j \in \mathbb{R}^n \Big\}
$$
of single-hidden-layer MLPs is dense in $C(K)$.

*Proof sketch (Hahn–Banach + Riesz).* Suppose, for contradiction, $\mathcal{F}_\sigma$ is **not** dense. Then $S := \overline{\mathcal{F}_\sigma}$ is a proper closed subspace of the Banach space $C(K)$. Pick any $f_0 \in C(K) \setminus S$. By the **Hahn–Banach theorem** there exists a continuous linear functional $\Lambda \in C(K)^*$ with $\Lambda \not\equiv 0$ but $\Lambda|_S = 0$.

By the **Riesz representation theorem**, $\Lambda$ is given by integration against a finite signed regular Borel measure $\mu$ on $K$:
$$
\Lambda(g) = \int_K g(x)\, d\mu(x), \qquad g \in C(K).
$$
Since each ridge function $x \mapsto \sigma(w^\top x + b) \in \mathcal{F}_\sigma \subset S$,
$$
\int_K \sigma(w^\top x + b)\, d\mu(x) = 0 \quad \forall\, w, b.
$$
Cybenko shows any continuous sigmoidal $\sigma$ is **discriminatory** (proven via Fourier analysis: pushing $\sigma$ toward step functions and reading off vanishing characteristic functions of half-spaces forces $\mu \equiv 0$). Hence $\mu = 0$, so $\Lambda \equiv 0$, contradiction. $\square$

### 2. Linear-only networks collapse

**Proposition.** If $\sigma = \mathrm{id}$, then any depth-$L$ MLP is itself a single affine map.

*Proof.* By induction on $\ell$, $h^{(\ell)} = W^{(\ell)} h^{(\ell-1)} + b^{(\ell)}$. Unrolling,
$$
f(x) = W' x + b', \quad W' = \prod_{\ell=L}^{1} W^{(\ell)}, \quad b' = \sum_{\ell=1}^{L} \Big(\prod_{k=L}^{\ell+1} W^{(k)}\Big) b^{(\ell)}.
$$
Therefore $f$ is affine, regardless of $L$. Affine maps are not dense in $C(K)$ (e.g. cannot approximate $\sin$ on $[-1,1]$ to error $< 0.1$). $\square$

So the nonlinearity $\sigma$ is not cosmetic: without it, depth buys nothing.

### 3. Depth separation (Telgarsky, 2016)

**Theorem (informal).** For every $k \geq 1$ there is a function $f_k : [0,1] \to [0,1]$ realizable exactly by a ReLU network of depth $O(k^3)$ and width $O(1)$, such that **any** ReLU network of depth $\leq k$ approximating $f_k$ in $L^1$ to error $< 1/32$ requires width $\geq 2^k$.

*Sketch.* Take $f_k$ to be the $k$-fold composition $\Delta^{\circ k}$ of the triangular "sawtooth" $\Delta(x) = 2x$ on $[0, 1/2]$, $\Delta(x) = 2(1-x)$ on $[1/2, 1]$. Each composition doubles the number of monotone pieces, so $f_k$ has $2^k$ pieces. A shallow ReLU net of depth $\leq k$ realizes a piecewise linear function with at most $\mathrm{poly}(\text{width})^k$ pieces; matching $2^k$ pieces forces exponential width. Composition realizes the same function with width and depth growing linearly in $k$. $\square$

The moral: UAT says shallow nets *can* approximate everything, but composition (depth) is **exponentially more parameter-efficient** for naturally hierarchical targets.

## Code sketch

The companion `cells.json` notebook (i) hand-builds a numpy MLP with $\tanh$ activation, (ii) fits a 32-unit MLP to $\sin(2\pi x)$ on $[-1,1]$ via least-squares on hidden features and verifies $\|f - \hat f\|_\infty < 0.05$, (iii) compares width-32-depth-1 against width-8-depth-3 at fixed parameter budget and observes lower error from the deeper net, and (iv) numerically confirms the linear-only collapse $f(x) = W' x + b'$.

## Connection to LLMs

Every transformer block contains a position-wise feed-forward module
$$
\mathrm{FFN}(x) = W_2\, \sigma(W_1 x + b_1) + b_2,
$$
i.e. a **single-hidden-layer MLP** applied independently to each token. This is exactly the universal approximator from Theorem 1: attention shuffles information *across* positions, while the FFN performs nonlinear pointwise computation that — by Cybenko — can in principle realize any continuous transformation of the embedding. Modern LLMs make $W_1$ wide ($4d$ or $8d$ inner dim) precisely to exploit Theorem 1's expressive guarantee, while stacking $L$ such blocks exploits Theorem 3's exponential depth gain. We will assemble the full block in Chapter 23.

<!-- CHAPTER 15 END -->

<!-- CHAPTER 16 START -->
<a id="chapter-16-activation-functions-relugelusoftmax-with-derivatives"></a>
## Chapter 16: Activation functions: ReLU/GELU/softmax with derivatives

## Motivation

A multi-layer perceptron (Chapter 15) without a nonlinear activation collapses into a single affine map: $W_2(W_1 x + b_1) + b_2 = (W_2 W_1) x + (W_2 b_1 + b_2)$. Universal approximation requires injecting a *pointwise* nonlinearity $\phi$ between affine layers. The choice of $\phi$ controls (i) the gradient signal that backpropagation (Chapter 3) is allowed to push through the network, (ii) the representational geometry of hidden activations, and (iii) numerical conditioning of the loss. This chapter derives the five activations that dominate modern deep learning: sigmoid, tanh, ReLU, GELU, and softmax. For each we compute the derivative or Jacobian from first principles, analyze saturation, and connect to the LLM stack (Chapters 21, 25).

## Definitions

**Sigmoid.** $\sigma:\mathbb{R}\to(0,1)$, $\sigma(x) = \frac{1}{1+e^{-x}}$.

**Hyperbolic tangent.** $\tanh:\mathbb{R}\to(-1,1)$, $\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} = 2\sigma(2x) - 1$.

**ReLU.** $\mathrm{ReLU}(x) = \max(0, x)$. Differentiable everywhere except $x=0$; the *Clarke subdifferential* at $0$ is $[0,1]$.

**GELU.** $\mathrm{GELU}(x) = x \Phi(x)$, where $\Phi(x) = \tfrac{1}{2}\bigl(1 + \mathrm{erf}(x/\sqrt 2)\bigr)$ is the standard-normal CDF. The Hendrycks–Gimpel tanh-approximation reads
$$\mathrm{GELU}(x) \approx \tfrac{1}{2}x\bigl(1 + \tanh(\sqrt{2/\pi}\,(x + 0.044715\,x^3))\bigr).$$

**Softmax.** $\mathrm{softmax}:\mathbb{R}^n \to \Delta^{n-1}$, $\mathrm{softmax}(z)_i = \frac{e^{z_i}}{\sum_{k=1}^n e^{z_k}}$, mapping logits to a probability simplex.

## Theorems and Proofs

**Theorem 16.1 (sigmoid derivative).** $\sigma'(x) = \sigma(x)(1-\sigma(x))$.

*Proof.* Write $\sigma(x) = (1+e^{-x})^{-1}$. By the chain rule, $\sigma'(x) = -(1+e^{-x})^{-2} \cdot (-e^{-x}) = \frac{e^{-x}}{(1+e^{-x})^2}$. Factor: $\frac{e^{-x}}{(1+e^{-x})^2} = \frac{1}{1+e^{-x}}\cdot \frac{e^{-x}}{1+e^{-x}} = \sigma(x)\bigl(1 - \sigma(x)\bigr)$, since $\frac{e^{-x}}{1+e^{-x}} = 1 - \sigma(x)$. $\square$

**Theorem 16.2 (tanh derivative).** $\tanh'(x) = 1 - \tanh^2(x)$.

*Proof.* From $\tanh = 2\sigma(2x) - 1$, the chain rule gives $\tanh'(x) = 4\sigma(2x)(1-\sigma(2x))$. Substitute $\sigma(2x) = (\tanh(x)+1)/2$: $4\cdot \tfrac{\tanh(x)+1}{2}\cdot \tfrac{1-\tanh(x)}{2} = (1+\tanh x)(1-\tanh x) = 1 - \tanh^2(x)$. $\square$

**Theorem 16.3 (ReLU derivative).** For $x\neq 0$, $\mathrm{ReLU}'(x) = \mathbf{1}_{x>0}$. At $x=0$, the Clarke subdifferential is $\partial\,\mathrm{ReLU}(0) = [0,1]$.

*Proof.* For $x>0$, $\mathrm{ReLU}(x)=x$ has derivative $1$. For $x<0$, derivative $0$. At $x=0$ the left derivative is $0$ and the right derivative is $1$; the convex hull of these limits, $[0,1]$, defines the subdifferential. Implementations conventionally pick $0$. $\square$

**Theorem 16.4 (GELU derivative).** $\mathrm{GELU}'(x) = \Phi(x) + x\phi(x)$, where $\phi(x)=\tfrac{1}{\sqrt{2\pi}}e^{-x^2/2}$.

*Proof.* Apply the product rule to $x\Phi(x)$: $\frac{d}{dx}[x\Phi(x)] = \Phi(x) + x\Phi'(x)$. By the fundamental theorem of calculus, $\Phi'(x) = \phi(x)$. $\square$

**Theorem 16.5 (softmax Jacobian).** Let $s = \mathrm{softmax}(z)$. Then $\frac{\partial s_i}{\partial z_j} = s_i(\delta_{ij} - s_j)$.

*Proof.* Let $S = \sum_k e^{z_k}$, so $s_i = e^{z_i}/S$. By the quotient rule,
$$\frac{\partial s_i}{\partial z_j} = \frac{(\partial e^{z_i}/\partial z_j)\,S - e^{z_i}\,(\partial S/\partial z_j)}{S^2} = \frac{\delta_{ij} e^{z_i} S - e^{z_i} e^{z_j}}{S^2} = \delta_{ij} s_i - s_i s_j = s_i(\delta_{ij} - s_j).\ \square$$

In matrix form, $J = \mathrm{diag}(s) - s s^\top$, a rank-$\le n$ symmetric PSD matrix with kernel spanned by $\mathbf{1}$ (consistent with softmax's translation invariance).

**Proposition 16.6 (saturation).** $\sigma'(x), \tanh'(x) \to 0$ as $|x|\to\infty$. Hence in deep MLPs, gradients $\prod_\ell \phi'(z_\ell)$ shrink geometrically — *vanishing gradients*. ReLU avoids saturation on the positive ray but suffers *dead neurons*: if a unit's pre-activation stays $\le 0$ across the whole training set, its gradient is identically zero. GELU is smooth and asymptotes to identity for $x \gg 0$ (since $\Phi(x)\to 1$ and $\phi(x)\to 0$, so $\mathrm{GELU}'(x)\to 1$) and to $0$ for $x\ll 0$ — combining ReLU-like sparsity with smoothness.

**Theorem 16.7 (softmax temperature limits).** Let $T>0$ and assume the maximizer $i^* = \arg\max_i z_i$ is unique. Then
$$\lim_{T\to 0^+} \mathrm{softmax}(z/T)_i = \mathbf{1}_{i = i^*},\qquad \lim_{T\to\infty} \mathrm{softmax}(z/T)_i = \tfrac{1}{n}.$$

*Proof.* Write $\mathrm{softmax}(z/T)_i = \frac{e^{(z_i - z_{i^*})/T}}{\sum_k e^{(z_k - z_{i^*})/T}}$. As $T\to 0^+$, every term $e^{(z_k - z_{i^*})/T}$ with $k\neq i^*$ tends to $0$ while the $k=i^*$ term equals $1$, giving $\mathbf{1}_{i=i^*}$. As $T\to\infty$, every $e^{(z_k - z_{i^*})/T}\to 1$, so the ratio tends to $1/n$. $\square$

**Proposition 16.8 (numerical stability).** $\mathrm{softmax}(z) = \mathrm{softmax}(z - c\mathbf{1})$ for any $c\in\mathbb{R}$. Choosing $c = \max_i z_i$ ensures every exponent is $\le 0$, so no overflow.

*Proof.* $\frac{e^{z_i - c}}{\sum_k e^{z_k - c}} = \frac{e^{-c} e^{z_i}}{e^{-c}\sum_k e^{z_k}} = \mathrm{softmax}(z)_i$. $\square$

**Corollary 16.9 (log-sum-exp).** $\log\sum_j e^{z_j} = \max_j z_j + \log\sum_j e^{z_j - \max_k z_k}$, also overflow-free.

## Code Sketch

The accompanying notebook (i) plots all five activations on $[-4,4]$ and tabulates them at $x\in\{-2,-1,0,1,2\}$; (ii) verifies analytic derivatives against centered finite differences; (iii) implements `softmax_jacobian(z) = diag(s) - np.outer(s, s)` and matches a numerical Jacobian to machine precision; (iv) sweeps temperature $T\in\{0.1, 1, 5, 100\}$ to visualize the one-hot/uniform limits of Theorem 16.7; (v) demonstrates that naive softmax of $(1000,1001,1002)$ overflows but the stabilized version recovers $(0.0900, 0.2447, 0.6652)$.

## Connection to LLMs

GELU is the default feedforward activation in GPT-2/3, BERT, and pre-Llama Transformers (Llama-family models switched to SwiGLU, a gated variant; see Chapter 17). Wherever Chapter 15's MLP block appears between attention layers, the inner nonlinearity is GELU acting elementwise on a $4d_{\text{model}}$-wide hidden state. Softmax appears in **two** distinct loci of an LLM:

1. **Attention** (Chapter 21): for query $q$ and keys $K$, attention weights are $\mathrm{softmax}(qK^\top/\sqrt{d_k})$ — a row-wise softmax converting compatibility scores into a probability distribution over keys.
2. **Output head** (Chapter 25): the final logits over the vocabulary are passed through softmax to produce the next-token distribution $p(x_{t+1}\mid x_{\le t})$. Sampling temperature (Theorem 16.7) is the standard knob exposed to users; $T\to 0$ recovers greedy decoding, $T\to\infty$ uniform random sampling.

Both invocations use the stabilized form of Proposition 16.8 in production kernels (FlashAttention, fused log-softmax + cross-entropy), making the seemingly trivial "subtract the max" identity load-bearing for trillion-parameter training.

<!-- CHAPTER 16 END -->

<!-- CHAPTER 17 START -->
<a id="chapter-17-loss-functions-mse-cross-entropy-gradients-from-first-principles"></a>
## Chapter 17: Loss functions: MSE, cross-entropy; gradients from first principles

# Loss functions: MSE, cross-entropy; gradients from first principles

## Motivation

A neural network is a parametric map $f_\theta : \mathcal{X} \to \mathcal{Y}$. To *train* it we need a scalar
"badness" score whose gradient with respect to $\theta$ tells us how to nudge weights. That scalar is the **loss
function**. Two losses dominate modern deep learning:

- **Mean squared error (MSE)** for regression targets.
- **Categorical cross-entropy (CE)** for classification, including the next-token prediction at the heart of
  every language model.

Both are not arbitrary engineering choices: each is the negative log-likelihood of a generative model (Ch 12).
And in both cases the *gradient with respect to the network's pre-activation output* collapses to the same
elegant form, $\hat p - y$, which is what makes backpropagation through deep stacks numerically tractable.

## Definitions

**MSE loss.** For a single scalar prediction $\hat y$ against target $y$,
$$
\ell_{\mathrm{MSE}}(y, \hat y) \;=\; \tfrac{1}{2}(y - \hat y)^2.
$$
The factor $\tfrac{1}{2}$ is conventional; it cancels in the gradient.

**Categorical cross-entropy.** Let $y \in \{0,1\}^K$ be a one-hot label ($\sum_k y_k = 1$) and
$\hat p \in \Delta^{K-1}$ a predicted distribution over $K$ classes. Then
$$
\ell_{\mathrm{CE}}(y, \hat p) \;=\; -\sum_{k=1}^K y_k \log \hat p_k.
$$
This is the cross-entropy $H(y, \hat p)$ of Ch 11 evaluated on a single example.

**Cross-entropy with logits (softmax-CE).** In practice we never store $\hat p$; we store unnormalized logits
$z \in \mathbb{R}^K$ and apply softmax (Ch 16). Letting $c \in \{1,\dots,K\}$ be the true class index,
$$
\ell(z, c) \;=\; -\log \mathrm{softmax}(z)_c \;=\; -z_c + \log \sum_{j=1}^K e^{z_j}.
$$
The right-hand form is the **log-sum-exp** identity; it is the only numerically stable way to compute CE
from logits.

**Binary cross-entropy.** When $K = 2$ we represent the prediction by a single logit $z$ with
$\hat p = \sigma(z)$ and $y \in \{0, 1\}$:
$$
\ell(z, y) \;=\; -y \log \sigma(z) - (1-y)\log(1-\sigma(z)).
$$

## Theorems

### Theorem 1 (MSE gradient)

$\nabla_{\hat y} \ell_{\mathrm{MSE}} = \hat y - y$.

*Proof.* Differentiate $\tfrac{1}{2}(y-\hat y)^2$ in $\hat y$: $-\tfrac{1}{2}\cdot 2(y-\hat y) = \hat y - y$. $\square$

### Theorem 2 (MSE = MLE under Gaussian noise)

Suppose $y = f_\theta(x) + \varepsilon$ with $\varepsilon \sim \mathcal{N}(0, \sigma^2)$. Then maximum likelihood
estimation of $\theta$ is equivalent to minimizing MSE.

*Proof.* The Gaussian density gives
$$
-\log p_\theta(y \mid x) \;=\; -\log \frac{1}{\sqrt{2\pi}\,\sigma} + \frac{(y - f_\theta(x))^2}{2\sigma^2}
\;=\; \frac{1}{2\sigma^2}\,(y - f_\theta(x))^2 + C,
$$
where $C$ is independent of $\theta$. Summing over an i.i.d. dataset and dropping the additive constant and
positive multiplicative constant yields $\sum_n \tfrac{1}{2}(y_n - f_\theta(x_n))^2$, which is the MSE
objective. $\square$

### Theorem 3 (Categorical CE = MLE under softmax)

Let the model be $p_\theta(y = c \mid x) = \mathrm{softmax}(z_\theta(x))_c$. Then for a single sample,
$-\log p_\theta(y = c \mid x) = -\log \mathrm{softmax}(z)_c = \ell(z, c)$. Summed across i.i.d. samples,
MLE equals minimization of categorical cross-entropy. This is the single-sample case of the
$\arg\max$-of-likelihood = $\arg\min$-of-cross-entropy identity proved in Ch 12. $\square$

### Theorem 4 (Softmax + CE gradient)

For $\ell(z, c) = -z_c + \log \sum_j e^{z_j}$,
$$
\frac{\partial \ell}{\partial z_j} \;=\; \mathrm{softmax}(z)_j - \mathbf{1}_{[j=c]} \;=\; \hat p_j - y_j.
$$

*Proof.* $\partial(-z_c)/\partial z_j = -\mathbf{1}_{[j=c]}$. For the log-sum-exp,
$$
\frac{\partial}{\partial z_j} \log \sum_k e^{z_k} \;=\; \frac{e^{z_j}}{\sum_k e^{z_k}} \;=\; \mathrm{softmax}(z)_j.
$$
Adding the two gives $\hat p_j - \mathbf{1}_{[j=c]}$. The dramatic cancellation hinges on the appearance of
$\sum_k e^{z_k}$ in *both* the softmax denominator and the CE normalizer; the matrix-vector product implicit
in the softmax Jacobian (Ch 16) collapses to a vector subtraction. This is *the* identity that makes deep
classifiers fast: one subtraction per output unit replaces a $K\times K$ matrix multiply. $\square$

### Theorem 5 (Binary CE + sigmoid gradient)

For $\ell(z, y) = -y\log \sigma(z) - (1-y)\log(1-\sigma(z))$, $\partial \ell / \partial z = \sigma(z) - y$.

*Proof.* Use $\log \sigma(z) = -\log(1+e^{-z})$ and $\log(1-\sigma(z)) = -z - \log(1+e^{-z})$. Then
$\ell = (1-y)z + \log(1+e^{-z})$. Differentiating: $(1-y) - e^{-z}/(1+e^{-z}) = (1-y) - (1-\sigma(z)) =
\sigma(z) - y$. $\square$

### Theorem 6 (CE minimum at $\hat p = y$)

Treat $y, \hat p \in \Delta^{K-1}$ as full distributions. Then $H(y, \hat p) = -\sum_k y_k \log \hat p_k$ is
minimized over $\hat p$ at $\hat p = y$.

*Proof.* $H(y, \hat p) - H(y, y) = -\sum_k y_k \log(\hat p_k / y_k) = D_{\mathrm{KL}}(y \,\|\, \hat p) \ge 0$
by Gibbs' inequality (Ch 11), with equality iff $\hat p = y$. $\square$

## Code sketch

```python
def softmax_ce_with_logits(z, c):
    z = z - z.max()                      # log-sum-exp stabilization
    lse = np.log(np.exp(z).sum())
    loss = -z[c] + lse
    p = np.exp(z - lse)                  # softmax(z)
    grad = p.copy(); grad[c] -= 1.0      # p - y
    return loss, grad
```

The forward and backward share the cached softmax $\hat p$; the entire backward is a single
in-place subtraction.

## Connection to LLMs

A causal language model (Ch 25) at every position $t$ emits logits $z_t \in \mathbb{R}^V$ over a vocabulary of
size $V$. The training loss across a sequence $x_1,\dots,x_T$ is
$$
\mathcal{L}(\theta) \;=\; \sum_{t=1}^{T} -\log p_\theta(x_t \mid x_{<t}) \;=\; \sum_{t=1}^{T} \ell(z_t,\, x_t),
$$
i.e. *cross-entropy with logits, summed over positions*. By Theorem 4 the gradient at the output layer is
$\hat p_t - y_t$ at every position — a single subtraction per token, per vocabulary entry. This signal flows
backwards through the unembedding, the transformer blocks (Ch 27 forward), and finally into the token
embeddings. Without the softmax-CE collapse, gradient computation at $V \approx 10^5$ would be infeasible.
Every LLM ever trained leans on Theorem 4.

<!-- CHAPTER 17 END -->

<!-- CHAPTER 18 START -->
<a id="chapter-18-backpropagation-chain-rule-applied-reverse-mode-ad-as-a-graph-algorithm"></a>
## Chapter 18: Backpropagation: chain rule applied; reverse-mode AD as a graph algorithm

## Motivation

Chapter 17 ended with a remarkably clean fact: for softmax + cross-entropy, the gradient of the loss with respect to the pre-softmax logits is just $\hat p - y$. That formula is exact and pleasant — but it tells us only the *output-layer* gradient. To train an actual network we need the gradient of the loss with respect to *every* parameter, including those buried under tens of intermediate nonlinearities. The naive approach — symbolic differentiation — produces expressions whose size explodes exponentially with depth. The naive numerical approach — finite differences over $P$ parameters — costs $O(P)$ forward passes, prohibitive when $P \approx 10^{11}$.

Backpropagation solves both problems at once. It is not a new mathematical idea: it is the multivariate chain rule (Chapter 4) applied along a directed acyclic graph in a particularly clever order. We make this precise.

## Computational graphs and AD

\textbf{Definition (Computational graph).} A *computational graph* is a DAG whose nodes are intermediate values $v_i \in \mathbb{R}^{d_i}$ and whose edges are elementary operations. Source nodes are inputs; sink nodes are outputs. A *forward pass* computes all $v_i$ in topological order given the inputs.

\textbf{Definition (JVP / forward-mode AD).} Given a tangent vector $\dot x$, forward-mode AD propagates $\dot v_i = \sum_{j \in \text{parents}(i)} \frac{\partial v_i}{\partial v_j} \dot v_j$ in topological order. One sweep computes $J \dot x$ for the full Jacobian $J = \partial f/\partial x$. Cost: one pass per *input* dimension to recover the full Jacobian.

\textbf{Definition (VJP / reverse-mode AD).} Given a cotangent $\bar y$ at the output, reverse-mode AD propagates *adjoints* $\bar v_i$ in *reverse* topological order:
$$
\bar v_j = \sum_{i \in \text{children}(j)} \left(\frac{\partial v_i}{\partial v_j}\right)^T \bar v_i.
$$
One sweep computes $J^T \bar y$. Cost: one pass per *output* dimension. When the output is a scalar loss $L$, a single backward pass yields $\nabla_x L$. The adjoint $\bar v_i := \partial L / \partial v_i$ is the central object of backprop.

## The backprop theorem

\textbf{Theorem (Backprop = reverse-mode AD on a feedforward net).} Let $L(x) = \ell(h^{(L)})$ where $h^{(\ell)} = f^{(\ell)}(h^{(\ell-1)})$, $h^{(0)} = x$, and each $f^{(\ell)}$ is differentiable with Jacobian $J^{(\ell)} := \partial f^{(\ell)} / \partial h^{(\ell-1)}$. Define adjoints $\bar h^{(\ell)} := \partial L / \partial h^{(\ell)}$ as row vectors. Then
$$
\bar h^{(L)} = \nabla \ell(h^{(L)}), \qquad \bar h^{(\ell-1)} = (J^{(\ell)})^T \bar h^{(\ell)}.
$$

\textbf{Proof.} By the multivariate chain rule (Ch. 4), for any $\ell$,
$\partial L / \partial h^{(\ell-1)} = (\partial L / \partial h^{(\ell)})(\partial h^{(\ell)} / \partial h^{(\ell-1)}) = \bar h^{(\ell)} J^{(\ell)}$,
which transposed gives the adjoint recurrence. The base case is the gradient of $\ell$ at $h^{(L)}$. Induction over $\ell = L, L-1, \ldots, 1$. $\square$

For Ch. 17's softmax + cross-entropy block, $\bar h^{(L)} = \hat p - y$ is the base case and the recurrence does the rest.

## VJPs for an MLP layer

\textbf{Proposition.} For a layer $z^{(\ell)} = W^{(\ell)} h^{(\ell-1)} + b^{(\ell)}$, $h^{(\ell)} = \sigma(z^{(\ell)})$ with elementwise $\sigma$, the VJPs are
$$
\bar z^{(\ell)} = \bar h^{(\ell)} \odot \sigma'(z^{(\ell)}), \quad \bar W^{(\ell)} = \bar z^{(\ell)} (h^{(\ell-1)})^T, \quad \bar b^{(\ell)} = \bar z^{(\ell)}, \quad \bar h^{(\ell-1)} = (W^{(\ell)})^T \bar z^{(\ell)}.
$$

\textbf{Derivation.} The chain rule gives $\bar z^{(\ell)}_i = \sum_k \bar h^{(\ell)}_k \, \partial h^{(\ell)}_k / \partial z^{(\ell)}_i = \bar h^{(\ell)}_i \sigma'(z^{(\ell)}_i)$ since $\sigma$ is elementwise. For $W$, since $z_i = \sum_j W_{ij} h^{(\ell-1)}_j$, $\partial z_i / \partial W_{kj} = \delta_{ik} h^{(\ell-1)}_j$, so $\bar W_{kj} = \bar z_k h^{(\ell-1)}_j$, i.e. an outer product. The bias is $\partial z_i / \partial b_j = \delta_{ij}$. For $h^{(\ell-1)}$, $\partial z_i / \partial h^{(\ell-1)}_j = W_{ij}$, giving $\bar h^{(\ell-1)} = W^T \bar z^{(\ell)}$. $\square$

## Cost and memory

\textbf{Theorem (Baur–Strassen).} For a function defined by $K$ elementary operations with bounded fan-out, the gradient can be computed by an algorithm whose arithmetic cost is at most $\sim 5 K$ — i.e., the backward pass costs only a small constant times the forward pass, *independent of the number of parameters*.

This is the magic. We pay a constant overhead, not a $P$-fold one, to differentiate with respect to all $P$ parameters simultaneously. Forward-mode AD does the opposite: $O(n)$ for $n$ inputs, $O(1)$ in outputs.

\textbf{Memory.} The recurrence $\bar h^{(\ell-1)} = (J^{(\ell)})^T \bar h^{(\ell)}$ requires evaluating $J^{(\ell)}$, which usually depends on $h^{(\ell-1)}$ (e.g. $\sigma'(z^{(\ell)})$). Hence reverse-mode AD must *cache* every intermediate activation produced during the forward pass — memory cost is $O(L)$ in depth times batch and width. This is exactly the constraint that motivates *gradient checkpointing* (Ch. 27): drop activations and recompute them on the backward pass, trading compute for memory.

## Connection to LLMs

A modern transformer has $L \in [12, 96]$ blocks, each consisting of attention + MLP sub-layers. Training one step is: (1) forward pass through all $L$ blocks, caching activations; (2) reverse-mode AD computing $\nabla_\theta L$ for every parameter (often $\theta \in \mathbb{R}^{10^{11}}$) in time $\le 5\times$ the forward pass — Baur–Strassen in action; (3) optimizer step. The per-token activation memory is the dominant constraint at scale and is precisely why frameworks ship gradient checkpointing, FlashAttention recomputation (Ch. 23), and ZeRO-style sharding by default. Every parameter update in a 175B model is the recurrence above, executed billions of times.

<!-- CHAPTER 18 END -->


# Block E — Sequence Models and Attention

<!-- CHAPTER 19 START -->
<a id="chapter-19-embeddings-token-to-vector-lookup-as-a-linear-map-weight-tying"></a>
## Chapter 19: Embeddings: token to vector; lookup as a linear map; weight tying

## Motivation

A transformer language model never sees the symbol *cat*. It sees an integer index $v \in \{0, 1, \ldots, V-1\}$ which is then mapped to a vector in $\mathbb{R}^d$. That vector is the only representation downstream layers ever touch: attention, MLPs, normalizations, residuals — everything is linear algebra on $\mathbb{R}^d$. The gateway between the discrete vocabulary $\mathcal{V}$ (Chapter 1) and the continuous geometry of $\mathbb{R}^d$ is the **embedding matrix**. This chapter shows that this gateway is not a fancy table lookup at all: it is the linear map of Chapter 5 applied to a one-hot vector. The trick of *weight tying* — sharing parameters between the input embedding and the output projection — then falls out as a very natural identification.

## Definitions

Let $\mathcal{V} = \{w_1, \ldots, w_V\}$ be a finite **vocabulary** (Chapter 1); we identify $\mathcal{V}$ with $\{0, 1, \ldots, V-1\}$ via tokenization order.

**Definition (One-hot encoding).** For $v \in \{0, \ldots, V-1\}$ the **one-hot vector** is $e_v \in \{0, 1\}^V$ with $e_v[i] = \mathbf{1}_{i = v}$. Equivalently, $e_v$ is the $v$-th standard basis vector of $\mathbb{R}^V$.

**Definition (Embedding matrix and lookup).** An **embedding matrix** is a parameter $E \in \mathbb{R}^{d \times V}$. Its columns $\{E_{:, 0}, \ldots, E_{:, V-1}\}$ are called **token embeddings**; the integer $d$ is the **embedding dimension** (also called $d_{\mathrm{model}}$). The **embedding lookup** is the function $\mathrm{emb} : \mathcal{V} \to \mathbb{R}^d$, $\mathrm{emb}(v) := E e_v$.

**Definition (Output projection / unembedding).** An **output projection** is a parameter $U \in \mathbb{R}^{V \times d}$ taking a hidden state $h \in \mathbb{R}^d$ to a logit vector $z := U h \in \mathbb{R}^V$. The probability over the vocabulary is $\hat p = \mathrm{softmax}(z)$ (Chapter 9, 17).

**Definition (Weight tying, Press & Wolf 2016; Inan et al. 2017).** A model with embedding matrix $E \in \mathbb{R}^{d \times V}$ and output projection $U \in \mathbb{R}^{V \times d}$ is **weight-tied** if $U = E^\top$. The two layers then share the same $dV$ scalar parameters.

## Theorems and proofs

**Theorem 1 (Lookup is a linear map).** *For every $v \in \{0, \ldots, V-1\}$, $E e_v = E_{:, v}$.*

*Proof.* By the definition of matrix–vector multiplication (Chapter 5), $(E e_v)_i = \sum_{j = 0}^{V-1} E_{ij}\, (e_v)_j = \sum_j E_{ij} \mathbf{1}_{j = v} = E_{iv}$ for every $i \in \{0, \ldots, d-1\}$. Stacking these scalars gives the $i$-th coordinate of the column $E_{:, v}$, hence $E e_v = E_{:, v}$. $\blacksquare$

So embedding-table lookup is a linear map $\mathbb{R}^V \to \mathbb{R}^d$ in disguise. Implementations skip the matmul and slice the column; the *meaning* is the matrix product.

**Theorem 2 (Weight-tying gradient identity).** *Let $E \in \mathbb{R}^{d \times V}$ and consider a model $f(v, E) = U h(E e_v) = E^\top h(E e_v)$ with tied $U = E^\top$, where $h : \mathbb{R}^d \to \mathbb{R}^d$ is some sub-network with no further dependence on $E$. Let $\mathcal{L}$ be a scalar loss applied to the logits $z = E^\top h \in \mathbb{R}^V$. Then*
$$
\nabla_E \mathcal{L} \;=\; \underbrace{h\, (\nabla_z \mathcal{L})^\top}_{\text{output-side}} \;+\; \underbrace{\big(J_h^\top E\, \nabla_z \mathcal{L}\big)\, e_v^\top}_{\text{input-side}},
$$
*where $J_h \in \mathbb{R}^{d \times d}$ is the Jacobian of $h$ at the point $E e_v$. The input-side contribution is a rank-one matrix whose only nonzero column is column $v$.*

*Proof.* Treat $E$ as occurring in two distinct roles: an output role $U = E^\top$ and an input role inside $h \circ (E \cdot e_v)$. By the multivariate chain rule (Chapter 18), the total gradient is the sum of the partial gradients with respect to each role.

*Output role.* The logit is $z = U h$ with $U = E^\top$, so $z_k = \sum_i E_{ik} h_i$ and $\partial z_k / \partial E_{ij} = \mathbf{1}_{k = j} h_i$. Hence
$$\frac{\partial \mathcal{L}}{\partial E_{ij}} \bigg|_{\text{out}} = \sum_k \frac{\partial \mathcal{L}}{\partial z_k} \frac{\partial z_k}{\partial E_{ij}} = h_i\, \frac{\partial \mathcal{L}}{\partial z_j},$$
so the output-side contribution is the outer product $h\, (\nabla_z \mathcal{L})^\top \in \mathbb{R}^{d \times V}$.

*Input role.* The hidden state is $h = h(x)$ with $x = E e_v$, so $x_i = E_{iv}$ and $\partial x_i / \partial E_{ab} = \mathbf{1}_{a = i}\mathbf{1}_{b = v}$. By the chain rule,
$$\frac{\partial \mathcal{L}}{\partial E_{ab}} \bigg|_{\text{in}} = \sum_i \frac{\partial \mathcal{L}}{\partial x_i} \frac{\partial x_i}{\partial E_{ab}} = \frac{\partial \mathcal{L}}{\partial x_a}\, \mathbf{1}_{b = v}.$$
The vector $\nabla_x \mathcal{L} = J_h^\top \nabla_h \mathcal{L} = J_h^\top (E\, \nabla_z \mathcal{L})$ since $\nabla_h \mathcal{L} = U^\top \nabla_z \mathcal{L} = E\, \nabla_z \mathcal{L}$. Hence the input-side contribution is the rank-one matrix $g\, e_v^\top$ with $g := J_h^\top E\, \nabla_z \mathcal{L} \in \mathbb{R}^d$, whose only nonzero column is $g$ in position $v$. Adding the two roles gives the claim. $\blacksquare$

**Remark (Sparsity of the input gradient).** The factor $e_v^\top$ kills every column except column $v$. This is why embedding tables are conventionally updated via a *sparse* gradient: each minibatch only touches the columns of the tokens it actually contains. The output-side term, by contrast, hits *all* $V$ columns through the dense outer product $h (\nabla_z \mathcal{L})^\top$.

**Heuristic (Distributional semantics; Mikolov et al. 2013).** During training, two tokens $u, v$ that appear in similar surrounding contexts produce similar gradients on their embeddings (the "context" enters through $\nabla_z \mathcal{L}$ and through $h$). Iterating, $E_{:, u}$ and $E_{:, v}$ drift toward similar locations in $\mathbb{R}^d$, so cosine similarity of embeddings tracks distributional similarity in the corpus. This is an empirical claim, not a theorem; we illustrate it in code.

**Theorem 3 (Dimensionality bound).** *If $V > d$ then no choice of $E \in \mathbb{R}^{d \times V}$ makes the columns $E_{:, 0}, \ldots, E_{:, V-1}$ pairwise orthogonal and nonzero.*

*Proof.* Pairwise orthogonal nonzero vectors in any inner-product space are linearly independent: from $\sum_v c_v E_{:, v} = 0$, taking the inner product with $E_{:, u}$ yields $c_u \|E_{:, u}\|^2 = 0$, so $c_u = 0$. Hence the columns would form a linearly independent set of size $V$ in $\mathbb{R}^d$. By Chapter 5 (invariance of dimension), $\dim \mathbb{R}^d = d$, so any independent set has size $\leq d$. Therefore $V \leq d$, contradicting $V > d$. $\blacksquare$

In LLMs we always have $V \gg d$ (e.g. $V \approx 5\cdot 10^4$ vs.\ $d \approx 4096$), so the embeddings cannot be mutually orthogonal — they live as an *overcomplete* set, and "near-orthogonality" is the best one can ask.

## Code sketch

We build a tiny embedding matrix and verify $E e_v = E_{:, v}$ exactly. We then construct a tied unembedding $U = E^\top$ and check that the logit for token $v$ equals $\langle E_{:, v}, h \rangle$. Next, a Mikolov-style skip-gram on a synthetic corpus shows that embeddings of co-occurring tokens develop higher cosine similarity than non-co-occurring ones after a few hundred SGD steps. Finally, we verify Theorem 3 by attempting to orthogonalize $V = 8$ vectors in $\mathbb{R}^3$ via Gram–Schmidt and observing dimension exhaustion.

## Connection to LLMs

Every transformer language model contains exactly the data structure of this chapter: an input embedding $E \in \mathbb{R}^{d \times V}$ and an output projection $U \in \mathbb{R}^{V \times d}$. The embedding dimension $d$ is the *model dimension* $d_{\mathrm{model}}$, which is what dominates parameter counts and FLOPs (Chapter 23–25). GPT-2 (Radford et al. 2019), Llama 1/2/3 (Touvron et al. 2023), and most open-weight LMs use weight tying $U = E^\top$, halving the embedding parameter count and empirically improving perplexity (Press & Wolf 2016; Inan et al. 2017). GPT-3/4 use an LM-head that is initialized from $E^\top$ but may be trained separately. Once a token is embedded, position information is added (Chapter 20), and the residual stream begins (Chapter 22). Every operation downstream — including the final logit projection $U h_{\text{final}}$ — is the same matrix $E$ that began the forward pass, viewed from a different side.

<!-- CHAPTER 19 END -->

<!-- CHAPTER 20 START -->
<a id="chapter-20-rnn-intuition-vanishing-gradient-proof-why-we-need-attention"></a>
## Chapter 20: RNN intuition; vanishing-gradient proof; why we need attention

## Motivation

So far we have built feedforward networks (Ch.\ 17--19): a fixed-depth pipeline of linear maps and nonlinearities. But language, audio, and code are *sequences*: the output at position $t$ depends on inputs $x_1, \dots, x_t$, and the sequence length $T$ varies. We need a model that consumes one token at a time, accumulates context, and emits one prediction per step. The Recurrent Neural Network (RNN) was the dominant answer from the late 1980s through 2017. In this chapter we derive its forward dynamics, prove the **vanishing/exploding gradient theorem** (Pascanu, Mikolov, Bengio 2013) using the SVD machinery of Chapter 6 and the backprop chain rule of Chapter 18, and show that this failure mode--together with a parallel *information bottleneck*--is precisely what attention (Ch.\ 21--24) was invented to fix.

## Definitions

**Sequence model.** A map $f : \mathcal{X}^T \to \mathcal{Y}^T$ that consumes $(x_1, \dots, x_T)$ and emits $(y_1, \dots, y_T)$, with the *causality* constraint $y_t = f_t(x_1, \dots, x_t)$.

**RNN cell.** Fix hidden dimension $d$ and parameters $W_h \in \mathbb{R}^{d \times d}$, $W_x \in \mathbb{R}^{d \times \dim x}$, $b \in \mathbb{R}^d$, and a pointwise nonlinearity $\sigma$ (typically $\tanh$). The RNN cell is the recursion
$$h_t = \sigma(W_h h_{t-1} + W_x x_t + b), \qquad h_0 = 0.$$
The output is $y_t = W_y h_t$.

**Backpropagation through time (BPTT).** Unroll the recursion into an acyclic computation graph of depth $T$, sharing the weights $(W_h, W_x, b)$ across every time step, then apply the backprop algorithm of Chapter 18 to that graph. The shared-weight gradient is the sum of the per-step gradients.

## Theorem 1 (Vanishing/Exploding Gradient, Pascanu et al.\ 2013)

*Linearized statement.* Drop the nonlinearity and biases: $h_t = W h_{t-1}$ with $W \in \mathbb{R}^{d \times d}$. Let $L$ be a scalar loss depending on $h_T$. Then for any $t < T$,
$$\frac{\partial L}{\partial h_t} = (W^\top)^{T-t} \frac{\partial L}{\partial h_T}.$$
Consequently, with $\sigma_{\max}(W)$ the largest singular value,
$$\Big\| \frac{\partial L}{\partial h_t} \Big\| \leq \sigma_{\max}(W)^{T-t} \Big\| \frac{\partial L}{\partial h_T} \Big\|,$$
with equality achievable. If $\sigma_{\max}(W) < 1$ the gradient *vanishes* exponentially in $T-t$; if $\sigma_{\max}(W) > 1$ it *explodes*.

**Proof.** By the multivariable chain rule (Ch.\ 18), $\partial h_t / \partial h_{t-1} = W$, so $\partial L / \partial h_{t-1} = W^\top \partial L / \partial h_t$. Iterating $T-t$ times gives the claimed product. By Chapter 6, $W = U \Sigma V^\top$, so $(W^\top)^{T-t} = V \Sigma^{T-t} U^\top$ (using $W^\top = V \Sigma U^\top$ and unitarity), and the operator-norm bound is just $\|\Sigma^{T-t}\|_2 = \sigma_{\max}(W)^{T-t}$. The bound is attained by aligning $\partial L / \partial h_T$ with the top left singular vector of $W^{T-t}$. $\blacksquare$

**Nonlinear extension.** With $\sigma$ such that $|\sigma'| \leq c$ pointwise (true for $\tanh$ with $c = 1$), the Jacobian of one step is $D_t W$ where $D_t = \mathrm{diag}(\sigma'(\cdot))$ has $\|D_t\|_2 \leq c$. Submultiplicativity of the operator norm gives
$$\Big\| \frac{\partial L}{\partial h_t} \Big\| \leq c^{T-t} \|W\|_2^{T-t} \Big\| \frac{\partial L}{\partial h_T} \Big\|.$$
For $\tanh$, $c = 1$ and saturated regions push $\|D_t\|$ much below $1$, *worsening* vanishing.

## Theorem 2 (Information Bottleneck of an RNN)

The hidden state $h_t \in \mathbb{R}^d$ is a deterministic function of $(x_1, \dots, x_t)$, so it carries at most $d$ real-valued degrees of freedom. By a counting / rate--distortion argument, if the inputs come from an alphabet of size $V$ and we want to losslessly recover $(x_1, \dots, x_t)$ from $h_t$, we need $d \geq \lceil t \log_2 V / B \rceil$ bits, where $B$ is the per-coordinate precision. Once $t \log_2 V \gg d B$, *some* information must be discarded; the per-token signal-to-noise of past tokens decays as $1/t$. *Sketch:* this is the data-processing inequality applied to the deterministic Markov chain $(x_1, \dots, x_t) \to h_t \to \hat x_s$ for $s \ll t$.

## Theorem 3 (Attention Dissolves Both Bottlenecks --- Sketch)

Let $h_T = \sum_{s=1}^T \alpha_{Ts} V_s$, where $V_s = V x_s$ are value vectors and $\alpha_{Ts}$ are scalar weights (in Ch.\ 21 we make $\alpha$ a learned softmax of query--key dot products). Then for *any* $s$,
$$\frac{\partial h_T}{\partial V_s} = \alpha_{Ts} I_d, \qquad \frac{\partial L}{\partial V_s} = \alpha_{Ts} \frac{\partial L}{\partial h_T}.$$
This is a *single* matrix-product step, independent of $T - s$. There is no exponentiation of any spectral quantity, so no vanishing/exploding regime in the depth direction. Furthermore, every past token has its own slot $V_s$ in working memory, so the bottleneck of Theorem 2 is replaced by an $O(T \cdot d)$-sized addressable cache. The full derivation, including queries, keys, and softmax, is Chapter 21.

## Connection to LLMs

The vanishing-gradient theorem is *the* historical reason transformer-based LLMs displaced RNNs. Bahdanau, Cho, and Bengio (2015) added attention to a seq2seq RNN to fix translation of long sentences; Vaswani et al.\ (2017, "Attention Is All You Need") removed recurrence entirely. Without an $O(1)$ gradient path between distant positions, scaling to $10^4$--$10^6$-token contexts is hopeless: a $\sigma_{\max} = 0.99$ RNN attenuates by $e^{-100}$ over $10\,000$ steps. We make this rigorous in Ch.\ 21 (single-head attention), generalize in Ch.\ 22 (multi-head), embed in the full Transformer block in Ch.\ 23, and scale in Ch.\ 24.

<!-- CHAPTER 20 END -->

<!-- CHAPTER 21 START -->
<a id="chapter-21-scaled-dot-product-attention-derivation-softmax-temperature-analysis"></a>
## Chapter 21: Scaled dot-product attention: derivation, softmax-temperature analysis

## Motivation

Chapter 20 ended with a sharp negative result: recurrent networks propagate gradients through $|i-j|$ multiplicative steps, so information from token $j$ reaches token $i$ only after a chain of contractions or expansions. Either the gradient vanishes ($\rho < 1$) or it explodes ($\rho > 1$). Long-range dependencies are statistically unreachable.

We now build a primitive that escapes this: **scaled dot-product attention** (Vaswani et al., 2017). It connects every output position to every input position through a *single* matrix multiplication, so the gradient between any pair of tokens travels in $O(1)$ depth. Combined with the embedding layer of Chapter 19 and the linear maps of Chapter 5, it yields the building block of every modern transformer LM.

## Definitions

Let $X \in \mathbb{R}^{T \times d_{\mathrm{model}}}$ be a sequence of $T$ token vectors (e.g.\ embeddings from Chapter 19). Fix projection matrices
$$
W_Q, W_K \in \mathbb{R}^{d_{\mathrm{model}} \times d_k}, \qquad W_V \in \mathbb{R}^{d_{\mathrm{model}} \times d_v},
$$
and define **queries, keys, and values**
$$
Q = X W_Q, \quad K = X W_K, \quad V = X W_V.
$$

**Definition (Scaled dot-product attention).**
$$
\mathrm{Attn}(Q, K, V) = \mathrm{softmax}\!\left( \tfrac{Q K^\top}{\sqrt{d_k}} \right) V.
$$
The softmax is applied row-wise. We call $A := \mathrm{softmax}(QK^\top/\sqrt{d_k}) \in \mathbb{R}^{T \times T}$ the **attention weights**. Each row $A_{i,:}$ is a probability vector ($A_{ij} \geq 0$, $\sum_j A_{ij} = 1$). When $Q, K, V$ are all derived from the same $X$, this is **self-attention**; if instead $K, V$ come from a second sequence $Y$, it is **cross-attention**.

## Theorems

**Theorem 1 (Attention as a soft database lookup).** *For each query index $i$, the output row $\mathrm{Attn}(Q,K,V)_i$ is a convex combination of the value rows $V_j$.*

*Proof.* By construction $A_{i,:}$ is the softmax of a real vector, so $A_{ij} \in (0,1)$ and $\sum_j A_{ij} = 1$ (Chapter 16). Then $(A V)_i = \sum_j A_{ij} V_j \in \mathrm{conv}\{V_1, \dots, V_T\}$. $\square$

**Theorem 2 (Variance of dot products controls softmax temperature).** *Suppose $Q_{i,:}$ and $K_{j,:}$ have i.i.d.\ entries with mean $0$ and variance $1$, independent of each other. Then*
$$
\mathbb{E}[Q_i \cdot K_j] = 0, \qquad \mathrm{Var}(Q_i \cdot K_j) = d_k.
$$
*Hence $(QK^\top)_{ij}/\sqrt{d_k}$ has unit variance.*

*Proof.* Write $S = \sum_{k=1}^{d_k} Q_{ik} K_{jk}$. By linearity and independence,
$$
\mathbb{E}[S] = \sum_k \mathbb{E}[Q_{ik}]\,\mathbb{E}[K_{jk}] = 0.
$$
For the second moment, since terms with distinct $k \neq k'$ are mean-zero and independent (so their cross terms vanish in expectation),
$$
\mathbb{E}[S^2] = \sum_{k} \mathbb{E}[Q_{ik}^2 K_{jk}^2] + \sum_{k \neq k'} \mathbb{E}[Q_{ik} K_{jk}]\mathbb{E}[Q_{ik'} K_{jk'}] = \sum_{k} 1 \cdot 1 + 0 = d_k.
$$
Therefore $\mathrm{Var}(S) = d_k$, and dividing by $\sqrt{d_k}$ rescales to unit variance. $\square$

**Corollary 3 (Saturation without scaling).** *Without the $1/\sqrt{d_k}$ factor, the logits $S$ feeding softmax have standard deviation $\sqrt{d_k}$, growing without bound. As $d_k \to \infty$, the maximum logit dominates and softmax concentrates on a single index. The Jacobian $\partial \mathrm{softmax}_i / \partial s_j = \mathrm{softmax}_i (\delta_{ij} - \mathrm{softmax}_j)$ then collapses, vanishing the gradient w.r.t. $Q, K$.*

The factor $1/\sqrt{d_k}$ is therefore a *temperature normalization* keeping softmax in the well-conditioned regime where it has rank-$(T-1)$ Jacobian and informative gradients.

**Theorem 4 (Permutation equivariance).** *Let $\pi$ be a permutation of $\{1,\dots,T\}$ with permutation matrix $P_\pi$. Then*
$$
\mathrm{Attn}(P_\pi Q, P_\pi K, P_\pi V) = P_\pi \, \mathrm{Attn}(Q, K, V).
$$

*Proof.* The pre-softmax matrix becomes $(P_\pi Q)(P_\pi K)^\top / \sqrt{d_k} = P_\pi (Q K^\top) P_\pi^\top / \sqrt{d_k}$. Row-wise softmax commutes with row permutation: $\mathrm{softmax}(P_\pi M P_\pi^\top) = P_\pi \,\mathrm{softmax}(M P_\pi^\top \cdot)$, but more cleanly, applied row-by-row, $A' = P_\pi A P_\pi^\top$. Then $A' (P_\pi V) = P_\pi A P_\pi^\top P_\pi V = P_\pi A V$. $\square$

This symmetry is *too strong*: attention cannot tell "the cat sat" from "sat cat the". Chapter 24 will repair this with positional encodings.

**Theorem 5 (O(1) gradient depth).** *Fix output position $i$ and input position $j$. The Jacobian $\partial (AV)_i / \partial V_j = A_{ij} I_{d_v}$ has spectral norm $A_{ij} \in (0,1)$, regardless of $|i-j|$.*

*Proof.* $(AV)_i = \sum_\ell A_{i\ell} V_\ell$, so $\partial (AV)_i / \partial V_j = A_{ij} I$. $\square$

Compare with Chapter 20: an RNN has $\|\partial h_i/\partial h_j\| \lesssim \rho^{|i-j|}$. Attention is exponentially better in distance, paying instead $O(T^2)$ in compute for the $A$ matrix.

## Code sketch

In `cells.json` we (i) implement attention from scratch, (ii) empirically confirm $\mathrm{Var}(Q\cdot K) = d_k$, (iii) measure entropy collapse without scaling, (iv) verify permutation equivariance to floating-point, and (v) compute gradient magnitude across $T=50$ for both attention and an RNN.

## Connection to LLMs

Scaled dot-product attention is **the** primitive of every transformer. Chapter 22 stacks $h$ heads of it (multi-head attention); Chapter 23 wires attention together with MLPs and residual streams into the transformer block; Chapter 24 fixes Theorem 4 with positional encodings. Production-scale variants — Flash Attention (IO-aware tiling), multi-query and grouped-query attention (sharing $K, V$ across heads), sliding-window and linear attention — all preserve the scaled-softmax form derived here. Whenever you call an LLM, every token interaction inside it is an instance of $\mathrm{softmax}(QK^\top/\sqrt{d_k}) V$.

<!-- CHAPTER 21 END -->

<!-- CHAPTER 22 START -->
<a id="chapter-22-multi-head-attention-parallel-heads-as-concat-then-project-complexity"></a>
## Chapter 22: Multi-head attention: parallel heads as concat-then-project; complexity

## Motivation

Chapter 21 built a single attention head: a $T \times T$ soft-lookup that links every output position to every input position in $O(1)$ depth. A single head, however, must compress *every* relational pattern (syntactic, semantic, positional) into one softmax-weighted average over $V$. The Vaswani et al. (2017) fix is **multi-head attention** (MHA): run $H$ attention heads in parallel, each on its own low-dimensional subspace, then concatenate and re-project. The total parameter count stays $\Theta(d_{\mathrm{model}}^2)$ — the same as a single full-rank attention — yet the model gains $H$ independent "channels" of attention.

This chapter proves three things. First, MHA is *exactly* a single attention with block-diagonal projections. Second, the per-head split keeps the parameter count constant in $H$. Third, the compute scales as $O(T d^2 + T^2 d)$, with the $T^2 d$ term dominating long-context inference; this motivates the inference-time variants **MQA** and **GQA**, used in PaLM and Llama 2/3 respectively.

## Definitions

Throughout, $X \in \mathbb{R}^{T \times d}$ where $d = d_{\mathrm{model}}$. Fix a head count $H$ dividing $d$, set $d_k = d_v = d/H$, and write $\mathrm{Attn}(\cdot,\cdot,\cdot)$ for scaled dot-product attention (Chapter 21).

**Definition 1 (Multi-head attention, MHA).** For $h = 1, \dots, H$, fix
$$
W_Q^{(h)}, W_K^{(h)} \in \mathbb{R}^{d \times d_k}, \qquad W_V^{(h)} \in \mathbb{R}^{d \times d_v}, \qquad W_O \in \mathbb{R}^{H d_v \times d}.
$$
Define
$$
\mathrm{head}_h \;=\; \mathrm{Attn}\!\bigl(X W_Q^{(h)},\, X W_K^{(h)},\, X W_V^{(h)}\bigr) \;\in\; \mathbb{R}^{T \times d_v},
$$
$$
\mathrm{MHA}(X) \;=\; \mathrm{Concat}(\mathrm{head}_1, \dots, \mathrm{head}_H)\, W_O \;\in\; \mathbb{R}^{T \times d}.
$$

**Definition 2 (Multi-query attention, MQA).** All $H$ heads share a single $W_K, W_V$; only $W_Q^{(h)}$ varies per head. Hence the cached $K, V$ tensors are $H$ times smaller.

**Definition 3 (Grouped-query attention, GQA).** Partition the $H$ heads into $G$ groups, $G \mid H$. Heads within a group share $W_K, W_V$. MHA = $G = H$, MQA = $G = 1$. Llama 2/3 use $G \in \{4, 8\}$.

## Theorems

**Theorem 4 (MHA = block-structured single attention).** *Let $W_Q^{\mathrm{blk}} \in \mathbb{R}^{d \times H d_k}$ be the horizontal concatenation of the per-head $W_Q^{(h)}$, and similarly $W_K^{\mathrm{blk}}, W_V^{\mathrm{blk}}$. Set*
$$
Q^{\mathrm{blk}} = X W_Q^{\mathrm{blk}}, \quad K^{\mathrm{blk}} = X W_K^{\mathrm{blk}}, \quad V^{\mathrm{blk}} = X W_V^{\mathrm{blk}} \;\in\; \mathbb{R}^{T \times H d_k}.
$$
*Reshape each into $\mathbb{R}^{T \times H \times d_k}$ by splitting the last axis into $H$ blocks of width $d_k$. Then the per-head attention applied to slice $h$ of these reshaped tensors equals $\mathrm{head}_h$ exactly. Equivalently, MHA is a single linear projection followed by a per-head attention loop, with no information mixing across heads until $W_O$.*

*Proof.* The horizontal concatenation $W_Q^{\mathrm{blk}} = [W_Q^{(1)} \mid \cdots \mid W_Q^{(H)}]$ satisfies $X W_Q^{\mathrm{blk}} = [X W_Q^{(1)} \mid \cdots \mid X W_Q^{(H)}]$, so the $h$-th block of width $d_k$ is exactly $X W_Q^{(h)} = Q^{(h)}$. The same holds for $K$ and $V$. Reshape is a no-op on memory layout: slice $h$ of $Q^{\mathrm{blk}} \in \mathbb{R}^{T \times H \times d_k}$ is $Q^{(h)}$. Since $\mathrm{Attn}$ is computed independently per head and uses only that head's $Q^{(h)}, K^{(h)}, V^{(h)}$, the outputs match. $\square$

This is a *block-diagonal* picture in the following sense: if we wrote MHA as one giant attention with a $T \times T$ score matrix, that score matrix is the blockwise sum $\sum_h Q^{(h)} K^{(h)\top}/\sqrt{d_k}$ only after softmax-per-head, *not* a literal block-diagonal QKV — heads do not communicate inside the softmax. This is the key inductive bias: each head selects independently, and only $W_O$ blends the results.

**Theorem 5 (Parameter count is independent of $H$).** *Total parameters in $W_Q^{(\cdot)}, W_K^{(\cdot)}, W_V^{(\cdot)}, W_O$ equal $4 d^2$.*

*Proof.* Each per-head projection is $d \times d_k = d \times d/H$, with $d^2/H$ parameters. There are $H$ heads and three projections, giving $3 H \cdot d^2/H = 3 d^2$. The output matrix $W_O$ is $H d_v \times d = d \times d$, contributing $d^2$. Total: $4 d^2$. The $1/H$ shrinkage of each head exactly cancels the $H$-fold replication. $\square$

**Theorem 6 (Compute complexity of MHA).** *For input $X \in \mathbb{R}^{T \times d}$,*
$$
\mathrm{cost}(\mathrm{MHA}) \;=\; \Theta(T d^2 + T^2 d).
$$

*Proof.* We sum the FLOPs of each step.

1. *Q, K, V projections.* Each is a $T \times d$ times $d \times d$ matmul (after concatenating the $H$ blocks, by Theorem 4): $\Theta(T d^2)$ per projection, $\Theta(T d^2)$ in total over the three.
2. *Score matrix per head.* $Q^{(h)} K^{(h)\top}$ is $(T \times d_k)(d_k \times T) = \Theta(T^2 d_k)$. Summed over $H$ heads: $H \cdot \Theta(T^2 d_k) = \Theta(T^2 d)$ since $H d_k = d$.
3. *Softmax.* $\Theta(T^2)$ per head, $\Theta(H T^2)$ total — subdominant.
4. *Attention output $A V$.* Per head $(T \times T)(T \times d_v) = \Theta(T^2 d_v)$, summed to $\Theta(T^2 d)$.
5. *Output projection $W_O$.* $T \times (H d_v) \cdot (H d_v \times d) = \Theta(T d^2)$.

Adding: $\Theta(T d^2) + \Theta(T^2 d) + \Theta(H T^2) + \Theta(T^2 d) + \Theta(T d^2) = \Theta(T d^2 + T^2 d)$. $\square$

**Corollary 7 (Long-context regime).** *When $T \gg d$, the $T^2 d$ term dominates. Doubling the sequence quadruples the attention FLOPs but only doubles projection FLOPs.*

This is the structural reason transformer inference is bandwidth-bound on long contexts and motivates Chapter 27's efficient-attention literature.

**Theorem 8 (KV-cache memory).** *Autoregressive decoding caches $K, V$ for all past tokens. Per layer, MHA stores $2 T H d_k = 2 T d$ scalars; GQA with $G$ groups stores $2 T G d_k = 2 T d \cdot G/H$; MQA stores $2 T d_k = 2 T d / H$.*

*Proof.* Each head needs its own $K \in \mathbb{R}^{T \times d_k}$, $V \in \mathbb{R}^{T \times d_v}$, totalling $2 T d_k$ scalars. MHA replicates this $H$ times (one per head). GQA replicates $G$ times (one per group), MQA once. Multiply through. $\square$

A 70 B-parameter model with $d = 8192, H = 64$ at $T = 8192$ stores $2 \cdot 8192 \cdot 8192 = 134$ M scalars per layer for MHA, but only $134/8 \approx 17$ M for GQA with $G = 8$ — an 8$\times$ inference-memory reduction at minimal quality cost.

## Connection to LLMs

GPT-2 and GPT-3 use vanilla MHA. PaLM (2022) was the first major model to deploy MQA, motivated by inference throughput; Shazeer (2019) had earlier proposed it. Llama 2 (2023) introduced GQA as a quality/memory compromise; Llama 3 retains it. Mistral and most open-weights successors follow suit. The picture is now standard: training uses MHA-shaped compute, deployment uses GQA to fit the KV cache in HBM. Chapter 23 stacks these blocks into a transformer layer; Chapter 27 surveys the efficient-attention zoo (FlashAttention, sliding-window, linear attention) that targets the $T^2 d$ term proved above.

<!-- CHAPTER 22 END -->

<!-- CHAPTER 23 START -->
<a id="chapter-23-transformer-block-residual-layernormrmsnorm-ffn-attention-gradient-flow-argument"></a>
## Chapter 23: Transformer block: residual + LayerNorm/RMSNorm + FFN + attention; gradient-flow argument

## Motivation

A single attention head (Chapter 22) is a beautiful linear-algebraic object, but a transformer is not one head — it is a *stack* of dozens to hundreds of nearly identical *blocks*. Two empirical facts dominate the design of those blocks. First, deep stacks of pure compositions $f_L \circ \cdots \circ f_1$ suffer from vanishing or exploding Jacobians (we proved a strict version of this for RNNs in Chapter 20). Second, the activations between layers drift in scale and mean unless we actively re-normalize. The transformer block solves both problems by wrapping every learned sublayer in a **residual connection** and a **normalization layer**, then alternating an **attention** sublayer with a position-wise **feedforward** MLP. This chapter pins down the algebra and proves the gradient-flow guarantee that makes 96-layer GPTs trainable end-to-end.

## Definitions

**Definition (Residual connection).** Given any map $F : \mathbb{R}^d \to \mathbb{R}^d$, the *residual* (or *skip*) wrapper is $\mathrm{Res}_F(x) = F(x) + x$.

**Definition (LayerNorm; Ba, Kiros, Hinton 2016).** For $x \in \mathbb{R}^d$ let $\mu = \tfrac1d \sum_i x_i$ and $\sigma^2 = \tfrac1d \sum_i (x_i - \mu)^2$. Then
$$\mathrm{LN}(x) = \gamma \odot \frac{x - \mu \mathbf{1}}{\sqrt{\sigma^2 + \varepsilon}} + \beta, \qquad \gamma, \beta \in \mathbb{R}^d.$$

**Definition (RMSNorm; Zhang & Sennrich 2019).** With $r(x)^2 = \tfrac1d \sum_i x_i^2$,
$$\mathrm{RMSN}(x) = \gamma \odot \frac{x}{\sqrt{r(x)^2 + \varepsilon}}.$$
RMSN drops the mean-subtraction step, removes the $\beta$ shift, and saves roughly 7% of the per-token FLOPs of LN.

**Definition (Position-wise FFN).** Two affine maps separated by a pointwise nonlinearity $\sigma$ (Chapter 16: usually GELU; in Llama-style models, SwiGLU):
$$\mathrm{FFN}(x) = W_2 \sigma(W_1 x + b_1) + b_2, \quad W_1 \in \mathbb{R}^{d_{\mathrm{ff}} \times d}, \ W_2 \in \mathbb{R}^{d \times d_{\mathrm{ff}}}, \ d_{\mathrm{ff}} = 4d.$$
This is exactly the one-hidden-layer MLP of Chapter 15, applied independently to every token position.

**Definition (Pre-norm transformer block).** Given a multi-head self-attention map $\mathrm{MHA}$ (Chapter 22), the modern (GPT-2 onward) block is
$$z = x + \mathrm{MHA}(\mathrm{LN}(x)), \qquad y = z + \mathrm{FFN}(\mathrm{LN}(z)).$$
The original Vaswani-2017 *post-norm* block applied LN *after* the residual sum: $z = \mathrm{LN}(x + \mathrm{MHA}(x))$. We motivate the switch in Theorem 3.

## Theorems

**Theorem 1 (Residual gradient identity).** *Let $y = F(x) + x$ with $F : \mathbb{R}^d \to \mathbb{R}^d$ differentiable. Then $\partial y / \partial x = J_F(x) + I$. Iterating across $L$ residual blocks $x_{\ell+1} = x_\ell + F^{(\ell)}(x_\ell)$, the end-to-end Jacobian is*
$$\frac{\partial x_L}{\partial x_0} \;=\; \prod_{\ell = L-1}^{0} \bigl(I + J_{F^{(\ell)}}(x_\ell)\bigr).$$

*Proof.* Direct from the chain rule (Chapter 18): $\partial x_{\ell+1}/\partial x_\ell = I + J_{F^{(\ell)}}(x_\ell)$, and Jacobians compose by left-multiplication. $\square$

**Corollary (Gradient flow).** Suppose $\|J_{F^{(\ell)}}\|_2 \le \kappa < 1$. Without residuals the backpropagated gradient norm is bounded by $\kappa^L$ — exponential decay, exactly the RNN catastrophe of Chapter 20. With residuals, expanding the product gives an identity-plus-perturbation:
$$\frac{\partial x_L}{\partial x_0} = I + \sum_\ell J_{F^{(\ell)}} + \sum_{\ell < m} J_{F^{(m)}} J_{F^{(\ell)}} + \cdots,$$
so $\|\partial x_L/\partial x_0\|_2 \ge 1 - L\kappa - \binom{L}{2}\kappa^2 - \cdots$, which stays bounded away from zero for any depth provided $\kappa$ is small enough. The identity term *guarantees* a non-vanishing path from output to input.

**Theorem 2 (LN invariances and Jacobian).** *For any $\alpha > 0$ and $\beta \in \mathbb{R}$, $\mathrm{LN}(\alpha x + \beta \mathbf{1}) = \mathrm{LN}(x)$ (we momentarily set $\gamma = 1, \beta_{\mathrm{LN}} = 0, \varepsilon = 0$ to isolate the normalizer).*

*Proof.* Let $x' = \alpha x + \beta \mathbf{1}$. Mean: $\mu' = \alpha \mu + \beta$. Variance: $\sigma'^2 = \tfrac1d \sum (\alpha x_i + \beta - \alpha \mu - \beta)^2 = \alpha^2 \sigma^2$. Hence
$$\frac{x' - \mu' \mathbf{1}}{\sigma'} = \frac{\alpha(x - \mu \mathbf{1})}{\alpha \sigma} = \frac{x - \mu \mathbf{1}}{\sigma}. \qquad \square$$

For the Jacobian, write $\hat x = (x - \mu \mathbf{1})/\sigma$. A short calculation (using $\partial \mu / \partial x_j = 1/d$ and $\partial \sigma / \partial x_j = (x_j - \mu)/(d\sigma)$) yields
$$\frac{\partial \hat x_i}{\partial x_j} = \frac{1}{\sigma} \left( \delta_{ij} - \frac{1}{d} - \frac{1}{d} \hat x_i \hat x_j \right).$$
The two subtracted terms project out the constant ($\mathbf{1}$) and $\hat x$ directions — exactly the directions LN is invariant to. The Jacobian is therefore rank $d - 2$, with kernel $\mathrm{span}\{\mathbf{1}, x - \mu \mathbf{1}\}$.

**Proposition 3 (RMSN $\approx$ LN on centered data).** *If $\mu(x) = 0$ then $\mathrm{LN}(x) = \mathrm{RMSN}(x)$ (with the LN $\beta$ absorbed into a downstream bias). For arbitrary $x$, $\mathrm{RMSN}(x) - \mathrm{LN}(x) = \gamma \odot \mu(x) \mathbf{1} / \sqrt{r(x)^2 + \varepsilon} + O(\mu/r)$.*

*Proof.* When $\mu = 0$, $r(x)^2 = \sigma(x)^2$ and $x - \mu \mathbf 1 = x$, so the two formulas coincide. The general expansion is one line of algebra. $\square$

The empirical observation behind RMSN (verified in our code cells) is that after a few transformer layers, residual streams in pre-LN networks already have near-zero mean, so dropping the centering step costs essentially no quality while saving compute and a parameter vector $\beta$.

**Remark (Pre-norm vs post-norm gradient scale).** With He/Xavier initialization (Chapter 17), assume each sublayer output has the same variance as its input. In a *post-norm* stack the residual sum $x + F(x)$ has variance $2 \mathrm{Var}(x)$, which LN immediately rescales to $\mathrm{Var}(x)$ — but the gradient picks up a factor of $1/\sqrt 2$ per block, and after $L$ blocks the gradient at the input scales like $2^{-L/2}$, requiring careful warmup. In a *pre-norm* stack the residual stream itself is never rescaled; only the *input to* each sublayer is normalized. The end-to-end Jacobian is Theorem 1's $\prod (I + J_F)$ with each $J_F$ acting on a normalized input, so the gradient norm at $x_0$ remains $\Theta(1)$ independent of $L$. This is why every model from GPT-2 onward (and Llama, Claude, Gemini, Mistral, Qwen, $\ldots$) uses pre-norm.

## Code sketch

The companion notebook (i) implements one full pre-norm block in numpy and verifies shape preservation; (ii) stacks 20 residual blocks vs 20 plain blocks and measures backpropagated gradient norms by finite differences; (iii) checks LN invariance under $x \mapsto \alpha x + \beta$ across many random $(\alpha, \beta)$; (iv) compares LN vs RMSN on centered and uncentered inputs.

## Connection to LLMs

Every layer of every modern decoder LLM is a tiny variation of the block above. GPT-2/3/4 use pre-LN with GELU FFN. Llama, Mistral, and Qwen swap LN for RMSN, GELU for SwiGLU (Chapter 16), and MHA for grouped-query attention (Chapter 24); Llama also uses RoPE positional encoding (Chapter 25) inside the attention sublayer. The block remains $z = x + \mathrm{Sublayer}_1(\mathrm{Norm}(x)),\ y = z + \mathrm{Sublayer}_2(\mathrm{Norm}(z))$. Stacking 32–120 such blocks, plus the embedding (Chapter 21), the unembedding tied to the embedding, and a final norm before the LM head, gives the entire forward pass of a frontier-scale language model. The training-stability argument of Theorem 1 is precisely what makes the 1.5T-parameter regime (Chapters 27, 28) reachable at all.

<!-- CHAPTER 23 END -->

<!-- CHAPTER 24 START -->
<a id="chapter-24-positional-encoding-sinusoidal-derivation-rope-construction"></a>
## Chapter 24: Positional encoding: sinusoidal derivation, RoPE construction

## Motivation

In Chapter 21 we proved that scaled dot-product attention is **permutation-equivariant**: if $P$ is any permutation matrix and $X \in \mathbb{R}^{T \times d}$ a sequence of token embeddings, then
$$
\mathrm{Attention}(PX, PX, PX) = P \cdot \mathrm{Attention}(X, X, X).
$$
A direct corollary is catastrophic for language modeling: shuffling the tokens of a sentence and feeding them to attention produces the *same* set of output vectors, only re-indexed. Attention by itself sees a *bag of tokens*, not a sequence. To recover order, we must inject the position $t \in \{0, 1, \ldots, T-1\}$ into the representations.

Three approaches dominate practice: **sinusoidal** PEs (Vaswani et al. 2017, original Transformer), **learned** PEs (BERT, GPT-2), and **rotary position embedding** or **RoPE** (Su et al. 2021), used by Llama, Mistral, Claude, Gemini, and most modern LLMs. This chapter derives all three and proves the central RoPE relative-position theorem.

## Definitions

**Definition (Sinusoidal PE).** For position $\mathrm{pos} \in \mathbb{N}$ and embedding dimension $d$ (assumed even), define $\mathrm{PE}(\mathrm{pos}) \in \mathbb{R}^d$ by
$$
\mathrm{PE}(\mathrm{pos}, 2i) = \sin(\mathrm{pos} \cdot \theta_i), \quad
\mathrm{PE}(\mathrm{pos}, 2i+1) = \cos(\mathrm{pos} \cdot \theta_i),
\qquad \theta_i := 10000^{-2i/d},\ i = 0, \ldots, d/2 - 1.
$$
The position-aware token representation is $x_t = E[\mathrm{token}_t] + \mathrm{PE}(t)$.

**Definition (Learned PE).** Fix a maximum context length $T_{\max}$. Let $P \in \mathbb{R}^{T_{\max} \times d}$ be a parameter matrix. The position embedding at $t$ is the $t$-th row $P_t$, learned end-to-end. This is what BERT and GPT-2 use.

**Definition (RoPE).** Group the $d$ coordinates of a query or key vector into $d/2$ consecutive pairs. For pair $i$ at position $\mathrm{pos}$, define the $2 \times 2$ rotation
$$
R_{\mathrm{pos}, i} \;=\; \begin{pmatrix} \cos(\mathrm{pos}\,\theta_i) & -\sin(\mathrm{pos}\,\theta_i) \\ \sin(\mathrm{pos}\,\theta_i) & \cos(\mathrm{pos}\,\theta_i) \end{pmatrix},
\qquad \theta_i = 10000^{-2i/d}.
$$
Let $R_{\mathrm{pos}} \in \mathbb{R}^{d \times d}$ be the block-diagonal matrix with these rotations on the diagonal. Then, *unlike* sinusoidal PEs, RoPE is **not** added to embeddings; it is applied multiplicatively to the projected queries and keys:
$$
q_m^{\mathrm{rot}} = R_m\, q_m, \qquad k_n^{\mathrm{rot}} = R_n\, k_n.
$$

## Theorems and Proofs

**Theorem 24.1 (Sinusoidal PE encodes shift as rotation).** *For every offset $k \in \mathbb{Z}$ there exists a fixed block-diagonal matrix $M_k \in \mathbb{R}^{d \times d}$, independent of $\mathrm{pos}$, such that*
$$
\mathrm{PE}(\mathrm{pos} + k) \;=\; M_k \cdot \mathrm{PE}(\mathrm{pos}).
$$

*Proof.* Restrict to the $i$-th coordinate pair. By the trigonometric addition formulas,
$$
\sin\big((\mathrm{pos}+k)\theta_i\big) = \sin(\mathrm{pos}\,\theta_i)\cos(k\theta_i) + \cos(\mathrm{pos}\,\theta_i)\sin(k\theta_i),
$$
$$
\cos\big((\mathrm{pos}+k)\theta_i\big) = \cos(\mathrm{pos}\,\theta_i)\cos(k\theta_i) - \sin(\mathrm{pos}\,\theta_i)\sin(k\theta_i).
$$
In matrix form, with $c := \cos(k\theta_i),\ s := \sin(k\theta_i)$:
$$
\begin{pmatrix} \sin((\mathrm{pos}+k)\theta_i) \\ \cos((\mathrm{pos}+k)\theta_i) \end{pmatrix}
= \begin{pmatrix} c & s \\ -s & c \end{pmatrix}
\begin{pmatrix} \sin(\mathrm{pos}\,\theta_i) \\ \cos(\mathrm{pos}\,\theta_i) \end{pmatrix}.
$$
The $2 \times 2$ matrix depends only on $k$ and $\theta_i$, not on $\mathrm{pos}$. Stacking these blocks for $i = 0, \ldots, d/2 - 1$ gives the claimed $M_k$. $\square$

**Theorem 24.2 (RoPE relative-position property).** *For all $m, n \in \mathbb{N}$ and $q_m, k_n \in \mathbb{R}^d$,*
$$
\big\langle R_m q_m,\ R_n k_n \big\rangle \;=\; q_m^{\top}\, R_{n - m}\, k_n.
$$

*Proof.* We use two block-wise facts. First, each $R_{\mathrm{pos}, i}$ is a planar rotation, so
$$
R_{\mathrm{pos}, i}^{\top} = R_{-\mathrm{pos}, i}, \qquad R_{a, i}\, R_{b, i} = R_{a + b, i}, \tag{$\star$}
$$
the latter being the standard angle-addition identity for $\mathrm{SO}(2)$. Lifting to block-diagonal matrices preserves these: $R_m^{\top} = R_{-m}$ and $R_a R_b = R_{a+b}$.

Now compute:
$$
\langle R_m q_m,\ R_n k_n \rangle
= (R_m q_m)^{\top}(R_n k_n)
= q_m^{\top} R_m^{\top} R_n k_n
= q_m^{\top} R_{-m} R_n k_n
= q_m^{\top} R_{n - m} k_n. \quad\square
$$

**Corollary 24.3 (Translation invariance).** *For any constant $c \in \mathbb{Z}$, $\langle R_{m+c} q_m, R_{n+c} k_n \rangle = \langle R_m q_m, R_n k_n \rangle$.*

*Proof.* Apply Theorem 24.2 to both sides: the right-hand side is $q_m^{\top} R_{(n+c) - (m+c)} k_n = q_m^{\top} R_{n - m} k_n$, which equals $\langle R_m q_m, R_n k_n \rangle$. $\square$

**Proposition 24.4 (No learnable RoPE parameters).** *The map $(q, k, \mathrm{pos}) \mapsto R_{\mathrm{pos}} q, R_{\mathrm{pos}} k$ contains no trainable weights: the angles $\theta_i = 10000^{-2i/d}$ are fixed.*

*Proof.* By inspection of the definition. $\square$

This is operationally important: a model trained with context length $T_{\mathrm{train}}$ can be evaluated at $\mathrm{pos} > T_{\mathrm{train}}$ since $R_{\mathrm{pos}}$ is defined for all integers. Learned PEs cannot do this — there is simply no row $P_t$ for $t \geq T_{\max}$. RoPE thus offers a principled (though imperfect) path to length extrapolation, refined by NTK-aware scaling, YaRN, and Position Interpolation in Chapter 27.

## Code Sketch

In `cells.json` we (i) confirm permutation-invariance of vanilla attention numerically, (ii) implement and visualize the sinusoidal PE matrix, (iii) verify Theorem 24.1 by checking $\|\mathrm{PE}(\mathrm{pos}+k) - M_k \mathrm{PE}(\mathrm{pos})\| < 10^{-12}$, (iv) implement RoPE for $d=8, T=6$ and confirm Theorem 24.2 by comparing $\langle R_m q_m, R_n k_n \rangle$ against $q_m^{\top} R_{n-m} k_n$, and (v) verify Corollary 24.3 by shifting all positions by $c \in \{1, 5, 10\}$ and observing identical attention scores.

## Connection to LLMs

Vaswani et al.'s original Transformer used sinusoidal PEs. BERT and GPT-2 switched to learned PEs, trading off extrapolation for slight accuracy gains within the training context. From Llama (2023) onward, RoPE is the de facto standard: the relative-position property of Theorem 24.2 means attention scores depend only on token *separation*, matching the linguistic intuition that "the dog *across the street*" should attend the same way regardless of where the phrase appears. Combined with extrapolation tricks (Chapter 27), RoPE has enabled context windows from $2$K (Llama 1) to $1$M+ tokens (Claude, Gemini). Chapter 23 introduced multi-head attention; positional encoding is applied per-head before the dot product, so RoPE composes cleanly with everything that follows.

<!-- CHAPTER 24 END -->


# Block F — Pre-training

<!-- CHAPTER 25 START -->
<a id="chapter-25-causal-masking-next-token-prediction-loss-as-mle-on-the-empirical-distribution"></a>
## Chapter 25: Causal masking; next-token prediction loss as MLE on the empirical distribution

## Motivation

We have built a transformer block that mixes information across positions (Ch 23) and a positional encoding that injects order (Ch 24). To turn this into a *language model*, we must (a) decide what the network outputs, (b) decide what loss to minimize, and (c) reconcile the desire for **parallel** training with the requirement that the model must not peek at future tokens. All three are settled by one design choice: **causal masking + next-token prediction**.

The pre-training recipe behind GPT-1/2/3/4, Llama, Claude and Gemini is exactly this chapter — fit a categorical distribution over the next token by maximum likelihood on a corpus, with a triangular attention mask that makes the per-position losses backprop-independent.

## Definitions

**Definition (Causal language model).** A *causal* (decoder-only) language model with parameters $\theta$ assigns to a sequence $x_{1:T} = (x_1,\ldots,x_T)$ over vocabulary $\mathcal{V}$ the probability
$$ p_\theta(x_{1:T}) = \prod_{t=1}^{T} p_\theta(x_t \mid x_{<t}), \qquad x_{<t} := (x_1,\ldots,x_{t-1}). $$
Each conditional $p_\theta(\,\cdot\,\mid x_{<t})$ is a softmax over $\mathcal{V}$ produced by a transformer applied to the prefix $x_{<t}$.

**Definition (Causal attention mask).** Let $T \in \mathbb{N}$. The causal mask is the matrix $M \in \{0,-\infty\}^{T\times T}$ with
$$ M_{ij} = \begin{cases} 0 & j \le i,\\ -\infty & j > i.\end{cases} $$
Causal scaled dot-product attention replaces the score matrix $S = QK^\top/\sqrt{d_k}$ by $S + M$ before the row-wise softmax. Because $\exp(-\infty) = 0$, row $i$ of the softmax has zero mass on every column $j > i$.

**Definition (Teacher forcing).** During training, the input to predict $x_t$ is the ground-truth prefix $x_{<t}$ (not the model's own previous predictions). With the causal mask, all $T$ predictions are computed from a single forward pass on $x_{1:T}$.

## Theorems and proofs

**Theorem 25.1 (Chain-rule factorization).** For any joint distribution $p$ on $\mathcal{V}^T$,
$p(x_{1:T}) = \prod_{t=1}^T p(x_t\mid x_{<t})$.

*Proof.* Iterate the conditional-probability identity (Ch 8) $p(A,B)=p(A)\,p(B\mid A)$:
$p(x_{1:T}) = p(x_1)\,p(x_2\mid x_1)\,p(x_3\mid x_{1:2})\cdots p(x_T\mid x_{<T})$. $\square$

This is *not* an assumption on the model — it holds for every joint distribution. The modeling choice is to *parameterize* each conditional with a transformer.

**Theorem 25.2 (NTP loss = empirical NLL = cross-entropy with empirical distribution).** Let $\mathcal{D} = \{x^{(n)}_{1:T_n}\}_{n=1}^{N}$ be a corpus of i.i.d. documents drawn from a true distribution $p^\star$. The next-token-prediction loss
$$ \mathcal{L}(\theta) \;=\; -\sum_{n=1}^{N}\sum_{t=1}^{T_n} \log p_\theta\!\left(x^{(n)}_t \mid x^{(n)}_{<t}\right) $$
satisfies $\arg\min_\theta \mathcal{L}(\theta) = \arg\min_\theta H(\hat p_{\mathcal{D}},\, p_\theta)$, where $\hat p_{\mathcal{D}}$ is the empirical distribution and $H$ is cross-entropy.

*Proof.* By Theorem 25.1, $\log p_\theta(x_{1:T}) = \sum_t \log p_\theta(x_t\mid x_{<t})$, so $\mathcal{L}(\theta) = -\sum_n \log p_\theta(x^{(n)})$. The empirical distribution is $\hat p_{\mathcal{D}}(x) = \tfrac{1}{N}\sum_n \mathbb{1}[x = x^{(n)}]$. Then
$$ H(\hat p_{\mathcal{D}}, p_\theta) = -\!\!\sum_{x\in\mathcal{V}^*}\hat p_{\mathcal{D}}(x)\log p_\theta(x) = -\frac{1}{N}\sum_{n=1}^N \log p_\theta(x^{(n)}) = \frac{1}{N}\,\mathcal{L}(\theta). $$
Multiplying by $N>0$ does not change the argmin (Ch 12). $\square$

**Corollary 25.3 (Consistency).** If the family $\{p_\theta\}$ is correctly specified — i.e. $p^\star = p_{\theta^\star}$ for some $\theta^\star$ — and identifiable, then by the standard MLE consistency argument (Ch 12) the minimizer $\hat\theta_N \to \theta^\star$ as $N \to \infty$.

*Proof sketch.* By the law of large numbers, $\tfrac{1}{N}\mathcal{L}(\theta) \to -\mathbb{E}_{p^\star}[\log p_\theta(X)] = H(p^\star, p_\theta)$. Gibbs' inequality (Ch 12) states $H(p^\star, p_\theta) \ge H(p^\star)$ with equality iff $p_\theta = p^\star$. Identifiability + uniform convergence promote pointwise convergence of the minimum to $\theta^\star$. $\square$

**Theorem 25.4 (Causal mask preserves backprop locality).** With the causal mask, for every position pair $(t,s)$ with $s>t$ and every layer's input embedding $h_s^{(0)}$ at position $s$,
$$ \frac{\partial \mathcal{L}_t}{\partial h_s^{(0)}} = 0, $$
where $\mathcal{L}_t = -\log p_\theta(x_t \mid x_{<t})$.

*Proof.* Write the network as a composition of layers. By induction on layer depth $\ell$, the masked attention output at position $i$ is
$ y_i^{(\ell)} = \sum_{j \le i} \alpha_{ij}^{(\ell)} V_j^{(\ell)}, $
because $\alpha_{ij}^{(\ell)} = 0$ for $j > i$. Pointwise sublayers (residuals, MLP, LayerNorm) act per-position and do not introduce dependencies on $j > i$. Hence the position-$t$ output $y_t^{(L)}$ depends only on $\{h_j^{(0)}\}_{j\le t}$. The logit $z_t = W y_t^{(L)}$ inherits this support, so $\partial \mathcal{L}_t / \partial h_s^{(0)} = 0$ for $s>t$. $\square$

This is exactly what makes parallel training sound: stacking the per-position losses $\mathcal{L} = \sum_t \mathcal{L}_t$ and backpropagating through one forward pass gives the *same* gradient as $T$ separate forward passes on prefixes $x_{1:t}$.

**Theorem 25.5 (Train vs. inference asymmetry).** Training: feed $x_{1:T}$ once; with the mask, obtain all $T$ logits in $\Theta(T^2 d)$ time and update $\theta$ on $\sum_t \mathcal{L}_t$. Inference: starting from a prompt $x_{1:k}$, sample $\hat x_{k+1} \sim p_\theta(\cdot\mid x_{1:k})$, append, and repeat — autoregressive generation requires $T-k$ sequential forward passes because each sampled token feeds the next.

The asymmetry stems from teacher forcing: at train time we *know* $x_t$, so position $t$ can be computed in parallel with position $t+1$; at inference time $x_{t+1}$ is whatever the model just produced.

**Definition (Perplexity).** $\mathrm{PPL}(x_{1:T}) = \exp\!\big(\tfrac{1}{T}\sum_t -\log p_\theta(x_t\mid x_{<t})\big)$. It is the geometric mean of $1/p_\theta(x_t\mid x_{<t})$. A uniform model over $|\mathcal{V}|$ tokens achieves $\mathrm{PPL} = |\mathcal{V}|$; lower is better.

## Code sketch

The notebook builds (i) an $8\times 8$ causal mask, (ii) a single-head causal attention layer and verifies row $t$ has zero weight on $j>t$, (iii) a tiny logits "model" and checks NTP loss equals summed cross-entropy, (iv) a perturbation experiment showing that changing token $T$ leaves the logit at $T-1$ untouched (Theorem 25.4), and (v) perplexity of uniform vs trained model.

## Connection to LLMs

Every modern decoder-only LLM — GPT-1/2/3/4, Llama, Claude, Gemini — is a causal language model trained with exactly this loss. Pre-training scales the corpus $\mathcal{D}$ (trillions of tokens) and the parameter count (Ch 27 onwards), but the objective is the one derived above: minimize $-\sum_t \log p_\theta(x_t\mid x_{<t})$, with the causal mask of Definition 25.2 making all $T$ losses backprop-independent so that one forward pass yields $T$ supervisory signals. Perplexity remains the canonical eval metric on held-out text.

<!-- CHAPTER 25 END -->

<!-- CHAPTER 26 START -->
<a id="chapter-26-tokenization-bpe-algorithm-greedy-merge-correctness"></a>
## Chapter 26: Tokenization: BPE algorithm; greedy merge correctness

## Motivation

A language model assigns probabilities over sequences drawn from a *finite* token vocabulary $\mathcal{V}$ (Chapter 1: $\mathcal{V}$ is a finite set, and the embedding matrix of Chapter 19 has exactly one row per element of $\mathcal{V}$). The choice of $\mathcal{V}$ is not innocuous. Two extremes:

- **Word-level vocabulary.** Treat each whitespace-delimited string as a token. Then $\mathcal{V}$ must grow without bound to cover the long tail of proper nouns, typos, code identifiers, URLs, and morphological variants. Any unseen word is mapped to a single `<unk>` symbol, destroying information at inference time. The OOV (out-of-vocabulary) problem is *unbounded*.
- **Character-level vocabulary.** Take $\mathcal{V}$ to be the set of Unicode codepoints (or, more robustly, the 256 raw bytes). Then $|\mathcal{V}|$ is small and OOV is impossible — every byte string is representable. But sequences become extremely long, and the quadratic cost of self-attention (Chapter 25) bites hard.

**Subword tokenization** is the middle ground: $\mathcal{V}$ is a learned set of byte strings, where common substrings (`the`, `ing`, `_function`) are atomic tokens and rare strings fall back to their constituent bytes. We study the dominant algorithm: **byte-pair encoding** (BPE), introduced as a compression heuristic by Gage (1994) and adapted to NMT by Sennrich, Haddow & Birch (2016).

## Definitions

\paragraph{Alphabet and corpus.} Fix a finite base alphabet $\Sigma$ — for byte-level BPE, $\Sigma = \{0, 1, \dots, 255\}$. A *corpus* is a finite multiset $\mathcal{C} \subset \Sigma^*$ of strings. Each string $s \in \mathcal{C}$ has a multiplicity $c(s) \in \mathbb{Z}_{>0}$.

\begin{definition}[Vocabulary and segmentation]
A *vocabulary* is a finite set $\mathcal{V} \subset \Sigma^+$ with $\Sigma \subseteq \mathcal{V}$. A *segmentation* of $s \in \Sigma^*$ under $\mathcal{V}$ is a tuple $(t_1, \dots, t_k) \in \mathcal{V}^k$ with $t_1 \cdots t_k = s$ (concatenation).
\end{definition}

\begin{definition}[Merge rule]
A *merge rule* is an ordered pair $(a, b) \in \mathcal{V} \times \mathcal{V}$. Applying it to a sequence $(t_1, \dots, t_k)$ replaces every adjacent occurrence of $(a, b)$ with the single token $ab$ (concatenation), scanning left to right and non-overlappingly.
\end{definition}

\begin{definition}[BPE training]
Given $\mathcal{C}$ and a budget $M \in \mathbb{Z}_{\geq 0}$:
\begin{enumerate}
\item Initialize $\mathcal{V}_0 = \Sigma$ and segment each $s \in \mathcal{C}$ as the sequence of its bytes.
\item For $m = 1, \dots, M$: count adjacent pair frequencies $f_m(a,b) = \sum_{s \in \mathcal{C}} c(s) \cdot \#\{\text{occurrences of }(a,b)\text{ in seg}_{m-1}(s)\}$. Pick $(a^\star, b^\star) = \arg\max f_m$ (ties broken lexicographically). Add the new token $a^\star b^\star$ to $\mathcal{V}_m = \mathcal{V}_{m-1} \cup \{a^\star b^\star\}$. Apply the merge rule everywhere in the corpus.
\end{enumerate}
The output is the pair $(\mathcal{V}_M, \,(r_1, \dots, r_M))$ where $r_m = (a^\star, b^\star)$ is the $m$-th merge rule.
\end{definition}

\begin{definition}[BPE encoding]
Given merge rules $(r_1, \dots, r_M)$, encode $s \in \Sigma^*$ by: start from the byte sequence of $s$; repeatedly find the smallest-rank applicable merge $r_i$ and apply it; halt when no rule applies. Decoding is concatenation: $(t_1, \dots, t_k) \mapsto t_1 \cdots t_k$.
\end{definition>

## Theorems

\begin{theorem}[Determinism of BPE]
Given a fixed merge list $(r_1, \dots, r_M)$ and a fixed tie-breaking rule, the encoder is a total function $\mathrm{enc}: \Sigma^* \to \mathcal{V}^*$.
\end{theorem}

\begin{proof}
Each merge replaces two adjacent tokens by one, strictly reducing sequence length by 1. Sequence length is a well-founded measure into $\mathbb{N}$, so the inner loop terminates after at most $|s|-1$ steps. At each step the smallest-rank applicable rule is unique (rules are an ordered list and tie-breaking is fixed), so the choice of merge to apply is deterministic.
\end{proof}

\begin{theorem}[Decoding inverts encoding]
For all $s \in \Sigma^*$, $\mathrm{dec}(\mathrm{enc}(s)) = s$.
\end{theorem}

\begin{proof}
Each token $t \in \mathcal{V}$ is, by induction on merges, a string in $\Sigma^+$. A merge replaces $(a,b)$ by $ab$ — concatenation — so the concatenation $t_1 \cdots t_k$ is invariant under any sequence of merges. Hence $\mathrm{dec}(\mathrm{enc}(s)) = $ concatenation of the bytes of $s$ $= s$.
\end{proof}

\begin{theorem}[Corpus-coverage monotonicity]
Let $L_m = \sum_{s \in \mathcal{C}} c(s) \cdot |\mathrm{seg}_m(s)|$ be the total token count of the corpus after $m$ training merges. Then $L_0 \geq L_1 \geq \cdots \geq L_M$, and $L_{m+1} < L_m$ whenever the chosen pair $r_{m+1}$ has positive count.
\end{theorem}

\begin{proof}
Applying the merge $r_{m+1} = (a^\star, b^\star)$ replaces each occurrence of the adjacent pair by a single token, decreasing the length of each affected segmentation by exactly the number of (non-overlapping) occurrences. So $L_{m+1} = L_m - f_{m+1}(a^\star, b^\star)$. Since $f_{m+1} \geq 0$, $L_{m+1} \leq L_m$, with strict decrease iff $f_{m+1}(a^\star, b^\star) > 0$.
\end{proof}

\begin{proposition}[Vocabulary size bound]
$|\mathcal{V}_M| \leq |\Sigma| + M$, with equality whenever every chosen pair is novel (which holds in the byte-level case since each new merge produces a string strictly longer than any of $\Sigma$, and pairs are chosen with positive count).
\end{proposition}

\begin{proof}
By induction on $m$: $|\mathcal{V}_0| = |\Sigma|$, and $|\mathcal{V}_{m+1}| \leq |\mathcal{V}_m| + 1$.
\end{proof}

\begin{remark}[Greedy vs.\ globally optimal]
Among all vocabularies of size $|\Sigma| + M$, finding the one that minimizes corpus token count $L$ is NP-hard (a reduction from \textsc{Set Cover}: each candidate substring is a "set" of corpus positions it can cover; choosing $M$ substrings to minimize residual length is equivalent to a weighted set-cover variant). BPE is the *greedy* choice: at each step, take the merge that gives the largest immediate compression. There is no guarantee of global optimality, but in practice BPE matches or beats more expensive alternatives on downstream perplexity.
\end{remark}

## Code sketch

```python
from collections import Counter
def train_bpe(corpus, M):
    seqs = [list(s.encode("utf-8")) for s in corpus]
    merges = []
    for _ in range(M):
        pairs = Counter()
        for seq in seqs:
            for a, b in zip(seq, seq[1:]):
                pairs[(a, b)] += 1
        if not pairs: break
        (a, b), _ = max(pairs.items(), key=lambda kv: (kv[1], kv[0]))
        new = (a, b) if isinstance(a, tuple) else (a,) + ((b,) if not isinstance(b, tuple) else b)
        merges.append((a, b))
        seqs = [merge_seq(seq, a, b) for seq in seqs]
    return merges
```

(Full runnable version in `cells.json`.)

## Connection to LLMs

The tokenizer is the *interface* between raw bytes and the model. GPT-2/3/4 use **byte-level BPE** with $|\mathcal{V}| \approx 50{,}257$ (gpt2) up to $\approx 100{,}000$ (cl100k_base). Llama uses **SentencePiece** BPE on Unicode with a $\sim$32k vocab. Claude uses a custom BPE-like scheme. In every case, the chosen $\mathcal{V}$ fixes the row count of the embedding matrix $E \in \mathbb{R}^{|\mathcal{V}| \times d}$ (Chapter 19), and shapes everything downstream: the softmax cost in the LM head, the effective sequence length, even what *concepts* the model can express atomically. Chapter 27 will explore practical tokenization pitfalls (numeric tokenization, multilinguality, the "SolidGoldMagikarp" effect) that follow directly from BPE's greedy, frequency-driven design.

<!-- CHAPTER 26 END -->

<!-- CHAPTER 27 START -->
<a id="chapter-27-pre-training-pipeline-adamw-warmup-cosine-decay-gradient-clipping-tiny-gpt-training-run"></a>
## Chapter 27: Pre-training pipeline: AdamW + warmup + cosine decay + gradient clipping; tiny-GPT training run

## Motivation

Chapters 14, 23, 25, and 26 gave us all the pieces in isolation: AdamW, the transformer block, next-token-prediction (NTP) loss with a causal mask, and a tokenizer. This chapter assembles them into the *pre-training pipeline* used to train every modern decoder-only language model — GPT-2/3/4, Llama, Mistral. The recipe is short enough to fit on a postcard:

$$\text{corpus} \xrightarrow{\text{tokenize}} \text{ids} \xrightarrow{\text{batch}} (B, T) \xrightarrow{\text{transformer}} \text{logits} \xrightarrow{\text{CE}} L \xrightarrow{\text{backprop}} g \xrightarrow{\text{clip+AdamW}} \theta_{t+1}.$$

Three engineering details — *learning-rate warmup*, *cosine decay*, and *gradient clipping* — separate "diverges in 200 steps" from "trains stably for $10^{12}$ tokens." We define each precisely, give first-principles justifications, and end with a runnable tiny-GPT training run.

## Definitions

\textbf{Definition 27.1 (Pre-training pipeline).} Given a corpus $\mathcal{C}$, a tokenizer $\tau: \mathcal{C} \to \{1, \dots, V\}^*$, and a parameterized causal LM $p_\theta(x_t \mid x_{<t})$, *pre-training* is the procedure

1. encode $\tau(\mathcal{C})$ into a long token stream $x_1, x_2, \dots, x_N$,
2. for each step $t = 1, \dots, T_{\text{train}}$: sample a batch of $B$ context windows of length $T$, compute the NTP loss $L_t = -\frac{1}{BT} \sum_{b,i} \log p_\theta(x^{(b)}_{i+1} \mid x^{(b)}_{\le i})$,
3. backpropagate, clip the global gradient norm, and take an AdamW step under the schedule $\eta_t$.

\textbf{Definition 27.2 (Linear warmup).} For warmup horizon $W \in \mathbb{N}$ and peak rate $\eta_{\max}$,
$$\eta_t^{\text{warm}} = \eta_{\max} \cdot \min\!\left(1, \frac{t}{W}\right).$$

\textbf{Definition 27.3 (Cosine decay).} For total horizon $T$, floor $\eta_{\min}$, and warmup $W$,
$$\eta_t = \eta_{\min} + \tfrac{1}{2}(\eta_{\max} - \eta_{\min})\!\left(1 + \cos\!\left(\pi \cdot \frac{t - W}{T - W}\right)\right), \qquad t \in [W, T].$$
The combined schedule ramps linearly to $\eta_{\max}$ on $[0, W]$ and then cosines down to $\eta_{\min}$ on $[W, T]$.

\textbf{Definition 27.4 (Global-norm gradient clipping).} Given the flattened gradient $g = \nabla_\theta L \in \mathbb{R}^P$ and a clip threshold $c > 0$,
$$\widetilde{g} = g \cdot \min\!\left(1, \frac{c}{\|g\|_2}\right).$$
Equivalently: rescale $g$ to lie in the ball of radius $c$.

\textbf{Definition 27.5 (Mixed precision).} Forward and backward computation is performed in bf16 (8-bit exponent, 7-bit mantissa); the AdamW *master* parameters and moments are stored in fp32. This roughly halves activation memory and exploits tensor-core throughput.

## Theorems

\textbf{Theorem 27.6 (Why warmup helps stability).} Let $L$ be $\beta$-smooth. The descent lemma (Ch.\ 7) gives
$$L(\theta_{t+1}) \le L(\theta_t) - \eta_t \|\nabla L_t\|^2 + \tfrac{\beta}{2} \eta_t^2 \|\nabla L_t\|^2.$$
At initialization, $\beta$ is effectively unbounded along certain directions because pre-norm activations have not yet equilibrated, so the second-order term dominates whenever $\eta_t > 2/\beta$. Linearly ramping $\eta_t$ from $0$ keeps the second-order penalty controlled for the early phase during which $\beta$ is large; once normalization layers and Adam's second moment $v_t$ stabilize, $\beta$ shrinks and a larger $\eta_{\max}$ becomes safe. RAdam (Liu et al., 2020) makes a closely related observation: Adam's variance estimate $\hat{v}_t$ has high variance for small $t$, and warmup is essentially a poor man's variance correction.

\textbf{Theorem 27.7 (Why cosine decay).} Smith (2017, 1cycle) and Loshchilov \& Hutter (SGDR, 2017) showed empirically that cosine decay outperforms step- and exponential-decay across vision and language tasks. A first-principles argument: near a minimum $\theta^\star$, a quadratic approximation $L(\theta) \approx \tfrac{1}{2}(\theta - \theta^\star)^\top H (\theta - \theta^\star)$ implies the iterate variance under SGD is $\Theta(\eta^2 \sigma^2 / \eta) = \Theta(\eta \sigma^2)$ (Ch.\ 14 noise-ball). Decaying $\eta_t \to \eta_{\min}$ shrinks the noise ball and lets $\theta_t$ resolve a sharper local minimum. Cosine in particular keeps $\eta$ near $\eta_{\max}$ for most of training (when exploration is valuable) and only collapses near the end.

\textbf{Theorem 27.8 (Clipping preserves descent).} Suppose $L$ is $\beta$-smooth and we clip the gradient to norm $\le c$. Then the AdamW update with rate $\eta$ satisfies
$$\mathbb{E}[L(\theta_{t+1})] \le L(\theta_t) - \eta \cdot \mathbb{E}\!\left[\frac{\widetilde{g}_t^\top \nabla L(\theta_t)}{\sqrt{\hat{v}_t} + \epsilon}\right] + \frac{\beta \eta^2 c^2}{2}.$$
\textit{Proof sketch.} Apply the smoothness bound to $\theta_{t+1} = \theta_t - \eta m_t / (\sqrt{\hat{v}_t} + \epsilon)$; the second-order term is $O(\eta^2 \|m_t\|^2)$ and clipping bounds $\|m_t\| \le c$. Crucially, the bound holds *uniformly* over batches, so a single rare bad gradient can no longer destroy the run. $\square$

\textbf{Remark 27.9 (Chinchilla scaling, Hoffmann et al.\ 2022).} For a fixed compute budget $C \approx 6 N D$ FLOPs (where $N$ is parameters and $D$ tokens), the loss-minimizing allocation satisfies $N^\star \propto C^{1/2}$ and $D^\star \propto C^{1/2}$, i.e.\ scale parameters and data *proportionally*. GPT-3 (175B params, 300B tokens) was severely under-trained by this metric; Chinchilla (70B, 1.4T tokens) used the same compute and matched/beat it. This is *not* a theorem we prove, but it is the rule that determines $T_{\text{train}}$ once $N$ and the compute budget are fixed.

## Code sketch

```python
def lr_schedule(t, W, T, eta_max, eta_min):
    if t < W:                          # linear warmup
        return eta_max * t / W
    progress = (t - W) / (T - W)       # cosine decay
    return eta_min + 0.5*(eta_max - eta_min)*(1 + math.cos(math.pi*progress))

def clip_global_norm(grads, c=1.0):
    flat = np.concatenate([g.ravel() for g in grads])
    norm = np.linalg.norm(flat)
    scale = min(1.0, c / (norm + 1e-12))
    return [g * scale for g in grads]
```

The notebook implements the full forward/backward of a 2-layer tiny GPT in numpy, drives it with this schedule + clip, and trains it to a clearly sub-trivial loss in $<5$ minutes on a CPU.

## Connection to LLMs

This chapter is the recipe. GPT-2 used exactly this pipeline (warmup $\approx$ 2K steps, cosine to $0.1\eta_{\max}$, clip $= 1.0$, AdamW $\beta_2 = 0.95$). Llama-2 used the same pipeline with a larger $W$, longer $T$, and bf16 mixed precision. What changes from "tiny GPT in this notebook" to "GPT-4 trained on a supercluster" is purely *scale*: $V$ from 30 to 100K, $T_{\text{ctx}}$ from 16 to 8K–128K, $d$ from 32 to 12K, $N$ from $10^4$ to $10^{12}$, and $D$ from $10^3$ tokens to $10^{13}$ tokens. The control flow of `for step in range(T_train): batch → forward → loss → backward → clip → adamw_step` is *byte-identical*.

## Scaling up: a real torch / MLX GPT on TinyStories

The pure-numpy implementation above is pedagogically clear but stops at 18K parameters and 28-character vocabulary --- coherent text generation requires more capacity. This subsection scales up by ~3 orders of magnitude using PyTorch (canonical) and MLX (Apple-native parallel cell).

### Model

A 6-layer pre-norm decoder with $d_{\text{model}}=384$, 6 heads ($d_k=64$), $d_{\text{ff}}=1536$, context length 256. With the GPT-2 vocabulary (50,257) and weight-tied embeddings/head, total parameter count lands at $\approx 30$M --- GPT-2-small class but tinier, in the same ballpark as `nanoGPT`. Architecture is identical across the PyTorch and MLX implementations; only the framework changes.

### Tokenizer

We use `tiktoken`'s `gpt2` BPE encoding (50,257 merges). The fallback path is character-level when `tiktoken` is unavailable, so the training script always runs.

### Training

AdamW with $\beta_1=0.9$, $\beta_2=0.95$, weight decay $0.1$. Linear warmup (5\% of steps) then cosine decay from $\eta_{\max}=3\!\times\!10^{-4}$ to $\eta_{\min}=3\!\times\!10^{-5}$. Global gradient norm is clipped at $1.0$. Batch size 16, ~3000 steps, ~12M training tokens. Wall-clock target on M4 Pro: torch+MPS $\approx 45$ min, MLX $\approx 25$ min.

### Validation

After training, the model generates ~100-token coherent stories at temperature 0.7 / top-k 40. See `training_run.md` for sample generations and wall-clock numbers from the actual Mac run.

### Cross-link to MLX

For Apple-silicon-native execution at ~2--3$\times$ torch+MPS throughput, see `mlx_gpt.py`. Same architecture, same hyperparameters.

<!-- CHAPTER 27 END -->


# Block G — Post-training

<!-- CHAPTER 28 START -->
<a id="chapter-28-sft-rlhf-ppogrpo-and-dpo-train-post-train-a-tiny-gpt"></a>
## Chapter 28: SFT, RLHF (PPO/GRPO), and DPO; train + post-train a tiny GPT

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

<!-- CHAPTER 28 END -->

# Block H — Reinforcement Learning

<!-- CHAPTER 29 START -->
<a id="chapter-29-mdp-foundations-bellman-equations-value-iteration-tabular-q-learning"></a>
## Chapter 29: MDP foundations: Bellman equations, value iteration, tabular Q-learning

## Motivation

Chapter 28 framed RLHF as KL-regularized reward maximization over the entire response distribution and derived DPO and PPO at the *loss* level, treating each completion $y$ as a single black-box action. That was a useful abstraction, but it skipped the structural fact that an LLM emits one token at a time, that each emission is a decision, that those decisions compound into a long-horizon credit-assignment problem, and that the very gradient estimators used by PPO, GRPO, REINFORCE, and the actor-critic baselines make sense only inside a richer model where states, actions, transitions, and discounted returns are formal objects. To derive *why* those algorithms converge — and to make sense of value baselines, advantages, the policy gradient, and the contraction-style proofs that justify trust-region methods — we need the formal object that sits underneath every reinforcement-learning method: the **Markov Decision Process** (MDP).

This chapter builds MDPs from scratch and proves the four pillars of the field. First, we derive the Bellman expectation and optimality equations as direct consequences of the recursive definition $G_t = r_t + \gamma G_{t+1}$ together with linearity of expectation (Chapter 10) and the tower property of conditional expectation (Chapter 8). Second, we prove that both the expectation operator $\mathcal{T}^\pi$ and the optimality operator $\mathcal{T}^*$ are $\gamma$-contractions in sup norm. Third, we invoke the Banach fixed-point theorem of Chapter 7 to conclude that **value iteration converges geometrically** with rate exactly $\gamma$ — the cleanest application of contraction mappings in the entire book. Fourth, we sketch finite-step convergence of policy iteration on finite MDPs and quote the Watkins–Tsitsiklis theorem connecting tabular Q-learning to the Robbins–Monro stochastic-approximation theory of Chapter 13. Token-by-token generation by an LLM *is* an MDP — state = prefix, action = next token, reward = terminal preference signal — so every theorem proved here applies, modulo combinatorial blow-up of $|\mathcal{S}|$, to the policies trained in Chapters 30 and 31.

## Definitions

**Markov Decision Process.** A tuple $(\mathcal{S}, \mathcal{A}, P, r, \gamma)$ where $\mathcal{S}$ is a finite state space, $\mathcal{A}$ a finite action space, $P : \mathcal{S} \times \mathcal{A} \to \Delta(\mathcal{S})$ a transition kernel ($P(s'|s,a)$ is the conditional probability of $s'$ given $(s,a)$, in the sense of Chapter 8), $r : \mathcal{S} \times \mathcal{A} \to \mathbb{R}$ a bounded reward, and $\gamma \in [0, 1)$ a discount factor.

**Trajectory.** A sequence $\tau = (s_0, a_0, r_0, s_1, a_1, r_1, \ldots)$ generated by sampling $a_t \sim \pi(\cdot|s_t)$, $s_{t+1} \sim P(\cdot|s_t, a_t)$, and setting $r_t = r(s_t, a_t)$.

**Return.** $G_t := \sum_{k=0}^\infty \gamma^k r_{t+k}$. Since $|r| \leq R_{\max}$ and $\gamma < 1$, $|G_t| \leq R_{\max}/(1-\gamma)$, so the series converges absolutely.

**Stationary policy.** A map $\pi : \mathcal{S} \to \Delta(\mathcal{A})$. Write $\pi(a|s)$ for the action probability.

**Value functions.** $V^\pi(s) := \mathbb{E}_\pi[G_t \mid s_t = s]$ and $Q^\pi(s, a) := \mathbb{E}_\pi[G_t \mid s_t = s, a_t = a]$. Optimal values: $V^*(s) := \sup_\pi V^\pi(s)$, $Q^*(s, a) := \sup_\pi Q^\pi(s, a)$.

**Bellman expectation operator.** For $V \in \mathbb{R}^{|\mathcal{S}|}$,
$$(\mathcal{T}^\pi V)(s) := \sum_a \pi(a|s) \big[r(s, a) + \gamma \sum_{s'} P(s'|s,a) V(s')\big].$$

**Bellman optimality operator.**
$$(\mathcal{T}^* V)(s) := \max_a \big[r(s, a) + \gamma \sum_{s'} P(s'|s,a) V(s')\big].$$

## Theorems and proofs

**Theorem 1 (Bellman expectation equation).** $V^\pi = \mathcal{T}^\pi V^\pi$.

*Proof.* By definition $G_t = r_t + \gamma G_{t+1}$. Take $\mathbb{E}_\pi[\,\cdot\,|s_t = s]$ and apply linearity of expectation (Chapter 10):
$$V^\pi(s) = \mathbb{E}_\pi[r_t \mid s_t = s] + \gamma\, \mathbb{E}_\pi[G_{t+1} \mid s_t = s].$$
By the tower property, $\mathbb{E}_\pi[r_t | s_t = s] = \sum_a \pi(a|s) r(s,a)$ and $\mathbb{E}_\pi[G_{t+1} | s_t = s] = \sum_{a, s'} \pi(a|s) P(s'|s,a)\, \mathbb{E}_\pi[G_{t+1} | s_{t+1} = s'] = \sum_{a, s'} \pi(a|s) P(s'|s,a) V^\pi(s')$ — the inner equality uses the Markov property of $P$. Combining yields $V^\pi(s) = (\mathcal{T}^\pi V^\pi)(s)$. $\square$

**Theorem 2 (Bellman optimality equation).** $V^* = \mathcal{T}^* V^*$.

*Proof.* Same calculation, but at the optimal policy the inner sum over $a$ is replaced by a max — for any state $s$, the Bellman principle says picking the locally-greedy action with respect to $r + \gamma P V^*$ is optimal, so $V^*(s) = \max_a [r(s,a) + \gamma \sum_{s'} P(s'|s,a) V^*(s')]$. The full proof reduces to showing this max is achieved by some deterministic policy, which holds because $\mathcal{A}$ is finite. $\square$

**Theorem 3 (contraction).** Both $\mathcal{T}^\pi$ and $\mathcal{T}^*$ are $\gamma$-contractions in sup norm $\|V\|_\infty := \max_s |V(s)|$.

*Proof.* Fix $V, V'$. Then for every $s$,
$$|(\mathcal{T}^\pi V)(s) - (\mathcal{T}^\pi V')(s)| = \gamma \big| \sum_a \pi(a|s) \sum_{s'} P(s'|s,a) [V(s') - V'(s')] \big| \leq \gamma \sum_{a, s'} \pi(a|s) P(s'|s,a) \|V - V'\|_\infty = \gamma \|V - V'\|_\infty,$$
since $\sum_a \pi(a|s) = 1$ and $\sum_{s'} P(s'|s,a) = 1$. Taking the max over $s$ gives $\|\mathcal{T}^\pi V - \mathcal{T}^\pi V'\|_\infty \leq \gamma \|V - V'\|_\infty$. For $\mathcal{T}^*$ use $|\max_a f(a) - \max_a g(a)| \leq \max_a |f(a) - g(a)|$ and repeat. $\square$

**Corollary 4 (value iteration).** Let $V_{k+1} := \mathcal{T}^* V_k$ from any $V_0$. By Banach's fixed-point theorem (Chapter 7) on the complete metric space $(\mathbb{R}^{|\mathcal{S}|}, \|\cdot\|_\infty)$, $V_k \to V^*$ and
$$\|V_k - V^*\|_\infty \leq \gamma^k \|V_0 - V^*\|_\infty.$$
Convergence is geometric with rate $\gamma$. The slope of $\log \|V_k - V^*\|_\infty$ versus $k$ equals $\log \gamma$.

**Theorem 5 (policy iteration; finite termination).** Alternate (i) *policy evaluation* $V^\pi = (I - \gamma P^\pi)^{-1} r^\pi$ — the matrix is invertible because $\rho(\gamma P^\pi) \leq \gamma < 1$ — with (ii) *greedy improvement* $\pi'(s) := \arg\max_a [r(s,a) + \gamma \sum_{s'} P(s'|s,a) V^\pi(s')]$. Then $V^{\pi'}(s) \geq V^\pi(s)$ pointwise (policy improvement theorem, by repeated application of $\mathcal{T}^{\pi'}$ on $V^\pi$ and contraction). Since the deterministic policy space has size $|\mathcal{A}|^{|\mathcal{S}|} < \infty$ and the value sequence is monotone non-decreasing, the algorithm terminates after finitely many iterations at a fixed point of $\mathcal{T}^*$, hence at $\pi^*$. $\square$

**Theorem 6 (Q-learning, Watkins 1989; Tsitsiklis 1994).** The update
$$Q_{t+1}(s_t, a_t) = (1 - \alpha_t) Q_t(s_t, a_t) + \alpha_t \big[r_t + \gamma \max_{a'} Q_t(s_{t+1}, a')\big]$$
satisfies $Q_t \to Q^*$ almost surely provided every $(s, a)$ is visited infinitely often and the per-pair step sizes satisfy the **Robbins–Monro conditions** of Chapter 13: $\sum_t \alpha_t = \infty$ and $\sum_t \alpha_t^2 < \infty$. *Sketch.* The update is exactly the Robbins–Monro recursion for the fixed point of $\mathcal{T}^*$, which is a contraction; the noise (replacing the expectation $\sum_{s'} P(s'|s,a)$ by a single sample $s_{t+1}$) is a martingale-difference with bounded variance, so Tsitsiklis' generalization of stochastic approximation to contraction operators applies. $\square$

## Code sketch

We build a $4{\times}4$ GridWorld, run value iteration, verify the geometric rate $\log \gamma$ on a log-y plot of $\|V_k - V^*\|_\infty$, extract the greedy policy from $Q^*$, then re-derive $V^*$ via policy iteration (in fewer iterations because each evaluation step is exact), then run tabular Q-learning for 5000 episodes with $\alpha_t = 1/(1+\mathrm{visits})$ and $\varepsilon$-greedy exploration to confirm $Q_t \to Q^*$. We close with REINFORCE on a 3-state MDP, previewing Chapter 30.

## Connection to LLMs / RLHF

A language model is an MDP whose state is the prompt-plus-prefix $s_t = (x, y_{<t})$, action is the next token $a_t = y_t \in \mathcal{V}$, transition is deterministic ($s_{t+1} = s_t \oplus a_t$), and the reward is sparse and terminal: $r_t = 0$ for $t < T$, $r_T = r_\phi(x, y)$ from a learned reward model. Under this MDP, the KL-regularized objective of Chapter 28 *is* expected discounted return with the per-step reward shaped by $-\beta \log \frac{\pi_\theta(a_t|s_t)}{\pi_\mathrm{ref}(a_t|s_t)}$ — the per-token KL acts as a dense intrinsic reward that prevents the post-training policy from drifting far from the SFT initialization. The Bellman optimality equation gives the closed-form $\pi^*(y|x) \propto \pi_\mathrm{ref}(y|x) \exp(r(x,y)/\beta)$ (Ch. 28, Theorem 1) as a one-step Bellman equation in this terminal-reward MDP: there is only one decision (the entire completion $y$), so $V^*$ collapses to $\beta \log Z(x)$ with the partition function from Chapter 28.

Value iteration and policy iteration are intractable here because $|\mathcal{S}|$ is exponential in context length and $|\mathcal{A}| = |\mathcal{V}| \approx 10^5$, so even one Bellman backup is impossible. This forces us into *parametric* methods — function approximation plus policy gradients — the subject of Chapter 30 (REINFORCE, score-function estimators, baselines, advantage estimation, generalized advantage estimation) and Chapter 31 (actor-critic, the PPO clipped-surrogate objective, GRPO's group-relative advantage, and the full RLHF stack used in InstructGPT, Claude, and Llama-3-Instruct). Every gradient estimator in those chapters is justified by an MDP identity proved here; every convergence claim is some non-tabular relaxation of the contraction and stochastic-approximation theorems above.

<!-- CHAPTER 29 END -->

<!-- CHAPTER 30 START -->
<a id="chapter-30-value-based-deep-rl-function-approximation-dqn-max-entropy-framework"></a>
## Chapter 30: Value-based deep RL: function approximation, DQN, max-entropy framework

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

<!-- CHAPTER 30 END -->

<!-- CHAPTER 31 START -->
<a id="chapter-31-policy-gradient-grpo-and-the-rlhfdpo-bridge-tiny-gpt-alignment-loop"></a>
## Chapter 31: Policy gradient, GRPO, and the RLHF/DPO bridge: tiny-GPT alignment loop

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

A unifying mental model: all three are doing surrogate optimization of $J(\theta) - \beta D_{\mathrm{KL}}(\pi_\theta \| \pi_{\mathrm{ref}})$. RLHF estimates the gradient with PPO + a learned value function; DPO eliminates online sampling by inverting the closed-form solution; GRPO eliminates the value function by group-baselining. The pre-trained model from Chapter 27 is the floor — it sets *what is sayable* — and these chapters reshape only a small region around it. Pre-training is calorie intake; alignment is digestion. Everything in this 31-chapter chain composes to produce that final sentence.

<!-- CHAPTER 31 END -->
