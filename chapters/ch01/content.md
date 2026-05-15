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
