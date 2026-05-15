## Quantization: int8 / int4 post-training quantization with measured perplexity & throughput tradeoffs

### Motivation

Llama-3 70B in `bfloat16` is 140 GB of weights -- larger than any consumer GPU's
VRAM and most workstation RAM. The same model in `int4` is ~40 GB and fits on
a single 48 GB card with room for KV-cache. The accuracy cost? For
well-calibrated `int8` quantization, perplexity typically rises by less than
0.5%; for `int4`, by 1--3%. **Quantization is the single technique most
responsible for moving frontier-class models out of the data center and onto
laptops.** This chapter develops it from first principles.

We restrict attention to *post-training quantization* (PTQ): take an already-trained
fp16/bf16 checkpoint and replace its weights with low-precision approximations
without further gradient updates. The mathematical object of study is a map
$Q : \mathbb{R}^{m\times n} \to \mathbb{Z}^{m\times n}$ together with a
dequantizer $D$ that approximates the identity: $D(Q(W)) \approx W$. Two
questions follow: (i) what is the worst-case reconstruction error $\|W - D(Q(W))\|$
(Ch. 6 norm theory), and (ii) how does that error propagate to the cross-entropy
loss and hence the perplexity $e^{\mathrm{CE}}$ (Ch. 17)?

### Definitions

**Definition 34.1 (Affine quantizer).** Fix bit-width $b\in\{8,4,2\}$ and let
$N = 2^b$. An *affine* quantizer with scale $s>0$ and zero-point
$z\in\mathbb{Z}$ is the function
$$Q_{s,z}(w) = \mathrm{clip}\!\big(\mathrm{round}(w/s) - z,\ q_{\min},\ q_{\max}\big),$$
where the integer range is $[q_{\min}, q_{\max}] = [-N/2, N/2-1]$ (signed)
or $[0, N-1]$ (unsigned). The dequantizer is $D_{s,z}(q) = (q + z)\cdot s$.
The map is *symmetric* when $z=0$.

**Definition 34.2 (Calibration).** Given a *calibration set*
$\{x_1,\dots,x_K\}\subset\mathbb{R}^n$ and a layer with weight $W$, the scale
$s$ is chosen to cover the dynamic range observed during a forward pass --
typically $s = \max_{ij}|w_{ij}|/(N/2-1)$ (max-abs) or the 99.99-percentile
absolute value (percentile clipping, robust to outliers).

**Definition 34.3 (Granularity).**
*Per-tensor* quantization uses one scalar $s$ for the entire matrix.
*Per-row* (per-output-channel) gives each row $W_{i,:}$ its own $s_i$.
*Per-channel-group* breaks each row into groups of 128 contiguous elements,
each with its own scale. The metadata cost in bits per weight is, respectively,
$O(1/mn)$, $O(1/n)$, $O(1/g)$, traded against tighter dynamic range.

**Definition 34.4 (Weight-only quantization, W$b$A16).** Only the weight
tensors are stored in `int`$b$; activations stay in fp16/bf16. The matmul
dequantizes weights on-the-fly. This avoids the catastrophic effect of
*activation outliers* identified by Dettmers et al. (2022, LLM.int8()).

**Definition 34.5 (Layer-local objective).** For a linear layer $y = Wx$ and
calibration matrix $X\in\mathbb{R}^{n\times K}$, the *layer-local quantization
problem* is
$$\widehat W \in \arg\min_{\widehat W \in \mathcal{Q}}\ \big\|WX - \widehat W X\big\|_F^2,$$
where $\mathcal{Q}$ is the (discrete) set of representable quantized matrices.
This is the objective minimized by **GPTQ** (Frantar et al., 2023).

### Theorems

**Theorem 34.6 (Round-to-nearest error bound).** Let $Q_s$ be the symmetric
affine quantizer with scale $s$ and let $\widehat w = D_s(Q_s(w)) = s\cdot
\mathrm{round}(w/s)$. For every $w$ in the representable range,
$$|\,w - \widehat w\,| \le s/2.$$

*Proof.* By definition of round-to-nearest, for any real $u$,
$|u - \mathrm{round}(u)| \le 1/2$. Set $u = w/s$:
$|w/s - \mathrm{round}(w/s)| \le 1/2$. Multiply by $s>0$:
$|w - s\cdot\mathrm{round}(w/s)| = |w - \widehat w| \le s/2$. $\square$

**Corollary 34.7 (Matrix Frobenius bound).** For an $m\times n$ matrix $W$
quantized symmetrically per-tensor with scale $s$,
$\|W - \widehat W\|_F \le (s/2)\sqrt{mn}$.

**Theorem 34.8 (Per-row beats per-tensor on heterogeneous matrices).** Let
$W\in\mathbb{R}^{m\times n}$ have rows with $\ell_\infty$ norms
$r_i = \|W_{i,:}\|_\infty$. With $b$ bits and $N=2^b$:
- *Per-tensor* scale $s_\star = (\max_i r_i)/(N/2-1)$ gives per-entry error
  $\le s_\star/2$, so
  $$\|W-\widehat W\|_F^2 \le \frac{mn}{4}\cdot\frac{(\max_i r_i)^2}{(N/2-1)^2}.$$
- *Per-row* scale $s_i = r_i/(N/2-1)$ gives
  $$\|W-\widehat W\|_F^2 \le \frac{n}{4(N/2-1)^2}\cdot\sum_{i=1}^m r_i^2.$$

The ratio is
$$\frac{\text{per-row bound}}{\text{per-tensor bound}}\;=\;\frac{1}{m}\cdot\frac{\sum_i r_i^2}{(\max_i r_i)^2}\;\le\; 1,$$
with equality iff every row has the same $\ell_\infty$ norm. The gap grows
with the variance of $\{r_i\}$.

*Proof.* Both inequalities follow from Theorem 34.6 applied entrywise; the
ratio identity is algebra. The bound is tight when one row dominates
(`max` $\gg$ mean), wasting most of the integer grid on small rows. $\square$

**Theorem 34.9 (GPTQ objective and OBQ update).** For a single layer with
weight $W\in\mathbb{R}^{m\times n}$ and calibration Hessian
$H = XX^\top \in \mathbb{R}^{n\times n}$ (positive semidefinite by
construction), the layer-local objective decouples row-wise:
$$\big\|WX-\widehat W X\big\|_F^2 \;=\; \sum_{i=1}^m (W_{i,:} - \widehat W_{i,:})\, H\, (W_{i,:} - \widehat W_{i,:})^\top.$$

*Proof.* Expand $\|WX - \widehat W X\|_F^2 = \mathrm{tr}((W-\widehat W)XX^\top
(W-\widehat W)^\top) = \sum_i \mathrm{row}_i (W - \widehat W)\, H\, \mathrm{row}_i(W-\widehat W)^\top$. $\square$

**Algorithm (GPTQ, sketch).** Process columns $j=1,\dots,n$ in order. At step
$j$, for each row $i$ quantize $w_{ij}$ to $\widehat w_{ij}$, compute the
residual $\delta = w_{ij}-\widehat w_{ij}$, and update the *unquantized*
columns $k>j$ by the Hessian-correcting term
$w_{ik} \leftarrow w_{ik} - \delta\cdot H^{-1}_{jk}/H^{-1}_{jj}$. The Cholesky
factor of $H^{-1}$ allows this in $O(n^3)$ per layer. The correction
absorbs the rounding error of column $j$ into the *not-yet-quantized*
columns, so subsequent quantizations partially cancel it out. The naïve
round-to-nearest (RTN) baseline is the special case where the correction is
omitted.

### Code sketch and benchmarks

The notebook (`cells.json`) contains six experiments:

1. **Memory accounting.** A small bytes-math table: GPT-2 small (124M),
   Llama-7B, Llama-70B in fp16 vs int8 vs int4.
2. **Round-to-nearest implementation.** Symmetric per-tensor `quantize(w, bits)`;
   measure $\|W-\widehat W\|_F$ on a Gaussian matrix.
3. **Per-tensor vs per-row.** A $64\times 128$ matrix with row norms growing
   geometrically as $10^{i/64}$; per-row gives roughly an order-of-magnitude
   smaller reconstruction error at int4.
4. **GPTQ on a tiny linear layer.** $d_{\text{in}}=16, d_{\text{out}}=8$,
   100-sample calibration set. The Hessian-corrected method reduces the
   layer-output error vs RTN by a clear margin.
5. **End-to-end perplexity.** A miniature Ch 27-style numpy GPT
   (2-layer character LM) is trained for a few hundred steps, then weights
   are quantized to int8 and int4. Perplexity on the training corpus is
   reported before and after.

### Connection to LLMs

Quantization closes the *deployment gap*. Training requires fp32 master
weights (Ch 14 AdamW state, Ch 27 mixed-precision); inference needs only
the forward pass, where weight precision dominates memory bandwidth on
modern GPUs. The empirical rule of thumb -- $<0.5\%$ perplexity loss for
int8, $1$--$3\%$ for int4 with GPTQ -- means a `Llama 3 70B` user trades
roughly 2% perplexity for a $3.5\times$ memory reduction and $2\times$
inference throughput. AWQ, SmoothQuant, and HQQ extend this picture:
all minimize variants of the layer-local objective from Theorem 34.9,
differing in how they reweight by activation statistics.
