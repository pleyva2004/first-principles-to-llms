## Motivation

A trained transformer is a function; an *inference engine* is a schedule for evaluating that function token by token. The schedule chosen by every modern LLM serving stack — vLLM, TensorRT-LLM, llama.cpp, MLX, transformers — pivots on one data structure: the **KV cache**. It is the difference between $O(T^2)$ generation and $O(T^3)$ generation, between 200 tokens/sec and 5 tokens/sec on the same hardware, and between fitting a 32k-context Llama on one A100 and not. This chapter derives the KV cache from the autoregressive factorization, proves it preserves the model's distribution exactly, characterizes its memory cost, and benchmarks the speedup on the tiny GPT we trained in Chapter 27.

Throughout, $T$ denotes current sequence length, $d$ the model dimension, $N_\text{layers}$ the number of decoder layers, $N_\text{heads}$ the number of attention heads, and $d_k = d / N_\text{heads}$.

## Definitions

**Autoregressive generation.** A decoder-only transformer parameterizes $p_\theta(x_t \mid x_{<t})$ for each position $t$. To sample a continuation of length $L$ given a prompt of length $T_0$, one repeats: compute $p_\theta(\cdot \mid x_{<t})$, draw $x_t$, append, advance $t$. The naive implementation runs the full forward pass over all of $x_{<t}$ at every step.

**Self-attention recap (Ch. 21–22).** A single head at layer $\ell$ maps the input $H^{(\ell-1)} \in \mathbb{R}^{T \times d}$ to
$$Q = H^{(\ell-1)} W_Q, \quad K = H^{(\ell-1)} W_K, \quad V = H^{(\ell-1)} W_V,$$
and outputs
$$\text{Attn}(Q, K, V) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}} + M_{\text{causal}}\right) V.$$
Multi-head attention concatenates $N_\text{heads}$ such heads (Ch. 22).

**KV cache.** For each layer $\ell$, head $h$, and previously processed token position $j$, store the row vectors
$$K_j^{(\ell, h)} \in \mathbb{R}^{d_k}, \qquad V_j^{(\ell, h)} \in \mathbb{R}^{d_v}.$$
At step $t+1$, given the new token embedding $h_t^{(\ell-1)}$, compute only
$$Q_t^{(\ell,h)} = h_t^{(\ell-1)} W_Q, \quad K_t^{(\ell,h)} = h_t^{(\ell-1)} W_K, \quad V_t^{(\ell,h)} = h_t^{(\ell-1)} W_V,$$
append $K_t, V_t$ to the cache, and compute attention as
$$\text{Attn}_t = \mathrm{softmax}\!\left(\frac{Q_t [K_{0:t+1}]^\top}{\sqrt{d_k}}\right) V_{0:t+1}.$$
There is no causal mask term because the cache contains only past keys by construction.

**Cache size in bytes.** Summing over layers and heads,
$$\boxed{\;\text{bytes}(T) = 2 \cdot N_\text{layers} \cdot T \cdot N_\text{heads} \cdot d_k \cdot b\;}$$
where $b$ is the byte width of the storage dtype ($b=2$ for fp16/bf16, $b=4$ for fp32) and the leading 2 accounts for storing both $K$ and $V$.

## Theorems

**Theorem 1 (Functional equivalence).** *Let $f_\theta$ denote the deterministic forward map of a causally-masked transformer. For any prompt $x_{0:T_0}$ and sampling seed $s$, the token sequence produced by cached generation equals the sequence produced by uncached generation.*

*Proof.* By induction on $t$. **Base** ($t = T_0$): no cache exists; both implementations run the same forward pass on $x_{0:T_0}$ and produce identical logits, hence (with shared seed $s$) identical $x_{T_0}$. **Inductive step.** Assume identical token sequences and identical layerwise activations $H^{(\ell)}_{0:t}$ through step $t$. At step $t+1$ the uncached path computes
$$Q_t^{(\ell)} = h_t^{(\ell-1)} W_Q,\quad K_{0:t+1}^{(\ell)} = H^{(\ell-1)}_{0:t+1} W_K,\quad V_{0:t+1}^{(\ell)} = H^{(\ell-1)}_{0:t+1} W_V,$$
then $\text{Attn}(Q_t, K_{0:t+1}, V_{0:t+1})$. The cached path stored $K_{0:t}^{(\ell)}, V_{0:t}^{(\ell)}$ from prior steps; by the inductive hypothesis these stored values are exactly the rows that the uncached path recomputes (each $K_j^{(\ell)}$ depends only on $h_j^{(\ell-1)}$, which the inductive hypothesis fixes). The cached path computes $K_t, V_t$ from the same $h_t^{(\ell-1)}$, concatenates, and applies the same softmax–matmul. Composition of deterministic operations on equal inputs yields equal outputs. The shared sampler seed $s$ then yields identical $x_{t+1}$. $\square$

The proof relies critically on the *causal mask*: $K_j, V_j$ for $j \leq t$ never need to be re-derived from later activations, so caching them is lossless. Without causality (e.g., bidirectional encoders), no analogous KV cache exists.

**Theorem 2 (Compute saved per generation step).** *Generating $L$ tokens after a prompt of length $T_0$ costs $\Theta((T_0 + L)^2 \, d)$ FLOPs with KV cache and $\Theta((T_0 + L)^3 \, d)$ without.*

*Proof.* At step $t$ (current length $T_0 + t$), the per-layer attention cost decomposes into the $QK^\top$ matmul and the $\mathrm{softmax} \cdot V$ matmul. **Cached**: $Q_t \in \mathbb{R}^{1 \times d_k}$, $K \in \mathbb{R}^{(T_0+t) \times d_k}$, so $Q K^\top$ is $\Theta((T_0+t) d_k)$ per head, $\Theta((T_0+t) d)$ across heads, and $\Theta((T_0+t) d)$ for the $V$ multiply, giving $\Theta((T_0 + t) d)$ per layer. **Uncached**: the model recomputes all $T_0 + t$ rows of $K, V$ and the full $(T_0+t) \times (T_0+t)$ attention matrix, giving $\Theta((T_0 + t)^2 d)$ per layer. Multiplying by $N_\text{layers}$ (a constant) and summing $t = 1, \ldots, L$:
$$\sum_{t=1}^{L}(T_0+t)\,d = \Theta((T_0+L)^2 d), \qquad \sum_{t=1}^{L}(T_0+t)^2 d = \Theta((T_0+L)^3 d). \;\square$$

**Theorem 3 (Memory cost).** *The KV cache for a fixed model grows linearly in context length $T$, with slope $2 \, N_\text{layers} \, N_\text{heads} \, d_k \, b$ bytes per token.*

*Proof.* Direct from the bytes formula; differentiate with respect to $T$. $\square$

**Corollary (Llama-7B at 8k).** Take $N_\text{layers} = 32$, $N_\text{heads} = 32$, $d_k = 128$, $b = 2$ (fp16), $T = 8192$:
$$\text{bytes} = 2 \cdot 32 \cdot 8192 \cdot 32 \cdot 128 \cdot 2 = 4.29 \times 10^9 \approx 4.0\,\text{GiB}.$$
This exceeds the size of the model's $W_K, W_V$ projection weights themselves, and motivates the architectural responses we cover in Ch. 22 (GQA, MQA) and the system-level response of PagedAttention.

## Code sketch and benchmarks

The notebook implements a $\sim 50$-line PyTorch decoder with an optional `kv_cache` argument. Each block holds a `(K, V)` pair of shape `(B, n_heads, T_cache, d_k)` that is `torch.cat`-extended at every step. Device auto-detection picks `mps` on the user's M4 Pro, falling back to CUDA or CPU. The benchmark loop generates 256 tokens for prompt lengths $T_0 \in \{32, 64, 128, 256\}$ and reports the wall-clock ratio. With cache, throughput is roughly constant in $T_0$; without cache, it falls as $1/T_0^2$, giving observed speedups of $\sim 5\times$ at $T_0 = 32$ and $\sim 40\times$ at $T_0 = 256$ — consistent with Theorem 2. A second cell tabulates analytical cache size for the Ch. 27 tiny GPT, GPT-2 small, Llama-7B, and Llama-70B (with 8-way GQA), demonstrating how grouping queries (Ch. 22) shrinks the cache by the GQA group factor.

## Connection to LLMs

Every production inference stack is, fundamentally, KV-cache management. **PagedAttention** (vLLM) treats the cache as virtual memory with block-level paging, eliminating fragmentation. **GQA / MQA** (Ch. 22) shrink $N_\text{heads}$ in the cache while keeping it for queries. **Continuous batching** packs many requests' caches side by side and schedules them at token granularity. **Speculative decoding** amortizes a verifier's cache across draft tokens. **Quantization** to int8 or int4 halves or quarters the per-token slope. The lesson of this chapter is that all of these techniques are reactions to two facts proved here: cached generation is exact (Theorem 1) and memory is linear in $T$ (Theorem 3). Once both hold, the engineering question is just how to fit the line under the GPU's HBM ceiling.
