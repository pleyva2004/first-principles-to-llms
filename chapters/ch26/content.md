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
