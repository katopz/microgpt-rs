# mini-dllm: Core Architecture

## Overview
The transformer is a from-scratch GPT-2 style implementation. No frameworks — weights are `Vec<f32>`, ops are hand-written matmul/softmax/rmsnorm. Supports multi-layer, grouped-query attention (GQA), and zero-allocation inference.

## Config (`types.rs`)
```rust
pub struct Config {
    pub vocab_size: usize,
    pub block_size: usize,     // max sequence length
    pub n_embd: usize,         // embedding dimension
    pub n_head: usize,         // number of attention Q heads
    pub head_dim: usize,       // dimension per head (n_embd / n_head)
    pub mlp_hidden: usize,     // MLP intermediate size
    pub n_layer: usize,        // number of transformer layers
    pub n_kv_head: usize,      // number of K/V heads (≤ n_head for GQA)
    pub bos_token: usize,
    pub temperature: f32,
    pub draft_lookahead: usize,
    pub tree_budget: usize,
    pub parallel_threshold: usize,  // skip rayon if n_embd ≤ this
}
```
- All configs constructed via `Config::micro()`, `Config::draft()`, `Config::bpe()`, etc.
- Validation: `n_head % n_kv_head == 0`, `n_embd == n_head * head_dim`
- `kv_dim()` helper returns `n_kv_head * head_dim`

## TransformerWeights (`transformer.rs`)
```rust
pub struct TransformerWeights {
    pub wte: Vec<f32>,              // [vocab_size, n_embd] — token embedding
    pub wpe: Vec<f32>,              // [block_size, n_embd] — position embedding
    pub lm_head: Vec<f32>,          // [vocab_size, n_embd] — output projection
    pub layers: Vec<LayerWeights>,  // per-layer weights (n_layer entries)
}

pub struct LayerWeights {
    pub attn_wq: Vec<f32>,   // [n_embd, n_embd]
    pub attn_wk: Vec<f32>,   // [n_embd, kv_dim]
    pub attn_wv: Vec<f32>,   // [n_embd, kv_dim]
    pub attn_wo: Vec<f32>,   // [n_embd, n_embd]
    pub mlp_w1: Vec<f32>,    // [mlp_hidden, n_embd]
    pub mlp_w2: Vec<f32>,    // [n_embd, mlp_hidden]
}
```
- Weight init: Kaiming-style `rng.normal() * sqrt(2 / (n_embd * n_layer))`
- Embedding init: `sqrt(2 / n_embd)`
- `TransformerWeights::new(config, rng)` creates all layers

## ForwardContext (`transformer.rs`)
Pre-allocated scratch buffers for zero-allocation forward passes:
```rust
pub struct ForwardContext {
    x: Vec<f32>,              // [n_embd] — hidden state (mutated in-place)
    q: Vec<f32>,              // [n_embd]
    k: Vec<f32>,              // [kv_dim]
    v: Vec<f32>,              // [kv_dim]
    attn_out: Vec<f32>,       // [n_embd]
    hidden: Vec<f32>,         // [mlp_hidden]
    xr: Vec<f32>,             // [n_embd] — residual buffer 1
    xr2: Vec<f32>,            // [n_embd] — residual buffer 2
    scores: Vec<f32>,         // [block_size] — attention scores
    logits: Vec<f32>,         // [vocab_size]
    pub hidden_state: Vec<f32>, // [n_embd] — snapshot before lm_head (for REST/cLoRA)
}
```
- Created once, reused across calls via `cache.reset()`
- `hidden_state` is copied from `x` before lm_head projection — "free embedding" for vector search

## MultiLayerKVCache (`transformer.rs`)
```rust
pub struct MultiLayerKVCache {
    pub layers: Vec<KVCache>,
}
pub struct KVCache {
    pub key: Vec<f32>,    // [block_size, kv_dim]
    pub value: Vec<f32>,  // [block_size, kv_dim]
}
```
- One KVCache per layer
- `kv_dim = n_kv_head * head_dim` (may be < n_embd with GQA)
- `reset()` clears all layers
- `snapshot(pos, config)` → `KVSnapshot` (copies only filled slots `[0..pos*kv_dim]`)
- `restore(snapshot, config)` — rollback to earlier state

## Forward Pass (`transformer.rs`)
```rust
pub fn forward(
    ctx: &mut ForwardContext,
    weights: &TransformerWeights,
    cache: &mut MultiLayerKVCache,
    token: usize,
    pos: usize,
    config: &Config,
) -> &mut [f32]  // logits
```
Pipeline:
1. **Embedding**: `x = wte[token] + wpe[pos]`
2. **Layer loop** (n_layer iterations):
   a. RMSNorm → QKV projection (GQA: K/V use kv_group)
   b. Store K/V in per-layer cache at position `pos`
   c. Multi-head attention (fused: score → softmax → weighted value)
   d. Output projection + residual add
   e. RMSNorm → MLP (matmul_relu + matmul) + residual add
3. **Snapshot**: `hidden_state = x` (before lm_head)
4. **LM Head**: `logits = lm_head @ x`

### GQA (Grouped-Query Attention)
When `n_kv_head < n_head`, K/V heads are shared:
- `kv_group = q_head * n_kv_head / n_head`
- K/V projection outputs `kv_dim` instead of `n_embd`
- 4× KV cache reduction for `n_head=8, n_kv_head=2`

## Math Kernels (`types.rs`)
All hot-path kernels are `#[inline(always)]` with `unsafe get_unchecked`:
- `matmul(out, w, x, rows, cols)` — out = W @ x
- `matmul_relu(out, w, x, rows, cols)` — fused matmul + ReLU
- `softmax(x)` — in-place, one-pass exp+sum, uses `inv_sum` multiply
- `rmsnorm(x)` — in-place, two-pass with `inv_rms` multiply
- `attention_head(...)` — fused: score → softmax → weighted value (avoids separate softmax write)
- `sample_token(logits, rng)` — categorical sampling

## Generate (`transformer.rs`)
```rust
pub fn generate(ctx, cache, weights, config, rng, token, n_tokens) -> Vec<usize>
pub fn generate_into(ctx, cache, weights, config, rng, tokens, n_tokens)  // zero-alloc variant
```
- Autoregressive: sample → feed back → repeat
- `generate_into` reuses pre-allocated buffers

## PagedKVCache (implemented, DDTree integration pending)
```rust
pub struct PagedKVCache {
    pages: Vec<Vec<f32>>,                    // pool of fixed-size pages
    layer_page_tables: Vec<Vec<Vec<usize>>>, // per-layer, per-sequence page indices
    free_pages: Vec<usize>,                  // reuse pool
    kv_dim: usize,
}
```
- Fixed `PAGE_SIZE = 16` tokens per page
- `fork(seq_idx, fork_at_pos)` — copy-on-write branch (shares prefix pages)
- Designed for DDTree branch exploration (each branch = one sequence)
- Deferred integration: currently DDTree uses flat `snapshot()/restore()` instead

## KVSnapshot
```rust
pub struct KVSnapshot {
    pub layers: Vec<KVLayerSnapshot>,
    pub pos: usize,
}
pub struct KVLayerSnapshot {
    pub key: Vec<f32>,    // [pos, kv_dim]
    pub value: Vec<f32>,  // [pos, kv_dim]
}
```
- Cheap: copies only filled slots `[0..pos*kv_dim]` per layer
- Used in speculative rollback: snapshot before verify, restore on reject