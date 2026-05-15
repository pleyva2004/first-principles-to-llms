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
