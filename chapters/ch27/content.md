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
