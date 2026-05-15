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
