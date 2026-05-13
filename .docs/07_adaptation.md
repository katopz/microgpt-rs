# microgpt-rs: Model Adaptation Techniques

Four production techniques that adapt the transformer to different tasks and domains **without modifying base weights**. All are feature-gated, zero-copy, and backward-compatible.

| # | Technique | Plan | Feature Flag | What It Does |
|---|-----------|------|-------------|--------------|
| 1 | Bidirectional Prefill | 025 | `router` | Prompt tokens attend to ALL others during prefill |
| 2 | Modality LoRA Switching | 025 | `router` | reader→writer LoRA swap at prefill→decode boundary |
| 3 | Sparse MLP (TwELL) | 022 | `sparse_mlp` | Skip dead ReLU neurons, O(alive) FLOPs |
| 4 | Domain Latent Injection | 038 | `domain_latent` | Mid-layer K/V conditioning per domain |

## Adaptation Pipeline

```
Prompt tokens
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│                    BIDIRECTIONAL PREFILL                     │
│  Phase A: K/V projections for all positions → cache         │
│  Phase B: Each position attends to K/V[0..prompt_len]       │
│           (no causal mask — code is non-linear)             │
│           reader_lora active                                 │
│           domain_latent injected at layer L/2               │
└─────────────────────┬───────────────────────────────────────┘
                      │ KV cache populated
                      │ first generated token
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                      CAUSAL DECODE                           │
│  Standard autoregressive: attend to K/V[0..pos+1]           │
│  writer_lora active (reference swap, zero data movement)    │
│  sparse_mlp: skip dead neurons in w2 @ hidden               │
│  domain_latent still conditioned from prefill               │
└─────────────────────────────────────────────────────────────┘
```

## Technique 1: Bidirectional Prefill

### Problem
Causal attention during prefill means each prompt token only sees preceding tokens. For code, this is wrong — a function body references a struct defined 3,000 tokens earlier. The model needs the whole file at once.

### Solution
Two-phase per-layer processing:

```
Layer L:
  Phase A: for p in 0..prompt_len { K[p], V[p] → cache }     // fill KV
  Phase B: for p in 0..prompt_len {                            // attend to ALL
    Q[p] → attend(Q[p], K[0..prompt_len], V[0..prompt_len])
    → output projection → MLP → hidden state
  }
```

The existing `attention_head` already accepts `t_n: usize` (number of KV positions). Prefill passes `prompt_len`; decode passes `pos + 1`. No API change.

### Implementation

```rust
// transformer.rs — PrefillContext (Plan 025)
pub struct PrefillContext {
    hidden: Vec<f32>,       // [max_prompt_len × n_embd] — multi-layer hidden states
    lora_buf: Vec<f32>,     // [rank] — pre-allocated LoRA intermediate
    max_prompt_len: usize,
}

pub fn forward_prefill(
    ctx: &mut ForwardContext,
    prefill: &mut PrefillContext,
    weights: &TransformerWeights,
    cache: &mut MultiLayerKVCache,
    tokens: &[usize],
    config: &Config,
    lora: Option<&LoraAdapter>,
    domain_latent: Option<&DomainLatent>,  // cfg(feature = "domain_latent")
) -> &mut [f32]
```

### Buffer Strategy (Zero Alloc)

| Buffer | Size | Allocation | Reuse |
|--------|------|------------|-------|
| `ForwardContext::x, q, k, v, attn_out, hidden, scores, logits` | Existing | `ForwardContext::new()` (once) | Per-position |
| `PrefillContext::hidden` | `prompt_len × n_embd` | `PrefillContext::new()` (once) | Between layers |
| `PrefillContext::lora_buf` | `[rank]` | `PrefillContext::new()` (once) | Per LoRA application |
| `MultiLayerKVCache` | Existing | Already pre-allocated | K/V storage |

**Single-layer optimization**: `PrefillContext::hidden` unused. Embeddings computed on-the-fly from `wte`/`wpe`. Zero extra memory.

### Performance

| Metric | Value |
|--------|-------|
| Prefill overhead vs causal | ~2× (two passes per layer) |
| Decode throughput impact | Zero (untouched code path) |
| Memory overhead (single-layer) | Zero extra beyond `lora_buf` |
| Memory overhead (multi-layer) | `prompt_len × n_embd × 4` bytes |

Prefill runs once per request. For 5K prompt → 500 generated tokens, prefill is 1 call, decode is 500. The 2× prefill overhead amortizes to near-zero.

## Technique 2: Modality LoRA Switching

### Problem
Different phases of a task need different behavior. During prefill, the model reads Python; during decode, it writes Rust. One LoRA can't optimize for both.

### Solution
Load two LoRA adapters per domain — `reader_lora` (active during prefill) and `writer_lora` (active during decode). The switch is a reference swap at the prefill→decode boundary.

```rust
// types.rs — LoRA pair (Plan 025)
pub struct LoraPair {
    /// Active during bidirectional prefill (e.g., Python Reader).
    pub reader: Option<LoraAdapter>,
    /// Active during causal decode (e.g., Rust Writer).
    pub writer: Option<LoraAdapter>,
}
```

### LoRA Application — In-Place Delta

```rust
// types.rs
pub struct LoraAdapter {
    pub a: Vec<f32>,     // [in_dim × rank]
    pub b: Vec<f32>,     // [rank × out_dim]
    pub rank: usize,
    pub alpha: f32,
    pub in_dim: usize,
    pub out_dim: usize,
}

/// Apply LoRA delta in-place: output += (α/r) × B @ (A @ input)
/// `lora_buf` is pre-allocated [rank] intermediate, zero alloc in hot path.
fn lora_apply(output: &mut [f32], lora: &LoraAdapter, input: &[f32], lora_buf: &mut [f32])
```

Applied after each Q/K/V/O/MLP projection when a LoRA is active. The delta is fused into the `matmul` output — no separate accumulation buffer.

### Switch Point

```rust
// transformer.rs — generate_with_prefill (Plan 025)

// 1. Bidirectional prefill with reader LoRA
let logits = forward_prefill(ctx, prefill, weights, cache, prompt_tokens, config,
    lora_pair.reader.as_ref(), domain_latent);

// 2. Switch to writer LoRA for decode
// ... reference swap, zero data movement ...

// 3. Causal decode with writer LoRA
let logits = forward_base(ctx, weights, cache, token, pos, config,
    lora_pair.writer.as_ref(), domain_latent);
```

### Performance

| Metric | Value |
|--------|-------|
| LoRA switch cost | Zero (reference swap) |
| LoRA apply overhead | 2 × rank × dim FLOPs per projection |
| Decode throughput impact | Negligible (small rank, fused into matmul) |

## Technique 3: Sparse MLP (TwELL-Inspired)

### Problem
ReLU zeros out ~50% of MLP neurons by definition. With L1 regularization during training, sparsity reaches 90-99%. Dense matmul wastes FLOPs on dead neurons.

### Solution
CPU index-packing sparse matmul for the MLP's second weight matrix (`w2 @ hidden`). Skip dead neurons to reduce FLOPs.

```rust
// types.rs — sparse_matmul (Plan 022)
/// Pack alive neurons (input[c] > 0.0) and multiply only those.
/// Returns alive count for diagnostics.
pub fn sparse_matmul(
    output: &mut [f32],
    weight: &[f32],
    input: &[f32],
    rows: usize,
    cols: usize,
    active_indices: &mut [usize],   // pre-allocated [mlp_hidden]
    active_values: &mut [f32],      // pre-allocated [mlp_hidden]
) -> usize
```

### Runtime Auto-Detection

Even with `sparse_mlp` feature enabled, the actual sparsity is checked at runtime:

```rust
// transformer.rs — forward_base MLP section
#[cfg(feature = "sparse_mlp")]
{
    let alive = types::sparse_matmul(
        &mut ctx.x, &layer.w2, &ctx.hidden, n, mlp_hidden,
        &mut ctx.active_indices, &mut ctx.active_values,
    );
    let alive_ratio = alive as f32 / mlp_hidden as f32;
    // Fallback to dense if not sparse enough
    if alive_ratio > (1.0 - config.sparse_threshold) {
        matmul(&mut ctx.x, &layer.w2, &ctx.hidden, n, mlp_hidden);
    }
}
```

- `sparse_threshold = 0.8` (default): use sparse when >80% of neurons are dead
- `0.0`: always use sparse
- `1.0`: never use sparse (always dense)

### Config

```rust
// types.rs
pub struct Config {
    pub sparse_threshold: f32,  // default: 0.8
    // ...
}
```

Feature flags:

| Flag | Description |
|------|-------------|
| `sparse_mlp` | TwELL-inspired sparse MLP matmul |
| `game_domain` | implies `domain_latent` |
| `full` | includes `sparse_mlp`, `ppot`, `domain_latent` |

### Pre-Allocated Buffers

```rust
// transformer.rs — ForwardContext (Plan 022)
#[cfg(feature = "sparse_mlp")]
active_indices: Vec<usize>,   // [mlp_hidden] — allocated once
#[cfg(feature = "sparse_mlp")]
active_values: Vec<f32>,     // [mlp_hidden] — allocated once
```

No `Vec::push` in hot loop. Buffers allocated in `ForwardContext::new()`, reused every forward pass.

### Design Decisions

1. **CPU-Only**: GPU stays dense. Unstructured sparsity causes warp divergence. Structured N:M sparsity (2:4, 4:8) is a separate plan.
2. **Feature-Gated**: Small models (mlp_hidden=64) won't benefit — packing overhead > savings. Users benchmark before enabling.
3. **w2 Only**: `w1 @ x` has dense input (no ReLU yet). `w2 @ hidden` has ReLU'd input → sparse.

### When It Helps

| Config | mlp_hidden | Benefit |
|--------|-----------|---------|
| micro | 64 | ❌ Packing overhead > savings |
| bpe | 128 | ❌ Marginal |
| small_target | 256 | ⚠️ Moderate (needs >80% sparsity) |
| large (real LLM) | 16384 | ✅ Significant at >50% sparsity |

## Technique 4: Domain Latent Injection

### Problem
LoRA adapts weights per domain, but has no mechanism for injecting an explicit domain signal. The model "knows" the domain implicitly through weight deltas, not through a direct conditioning vector.

### Solution
Distill the Free Transformer's mid-layer latent injection into a LoRA-compatible mechanism. Inject a learned domain embedding at layer `L/2` via K/V modulation.

```rust
// types.rs — DomainLatent (Plan 038)
pub struct DomainLatent {
    pub embedding: Vec<f32>,  // [kv_dim]
}

impl DomainLatent {
    pub fn load(path: &Path) -> Result<Self>;    // binary format with BLAKE3 checksum
    pub fn save(&self, path: &Path) -> Result<()>;
    pub fn zeros(kv_dim: usize) -> Self;
    pub fn from_vec(embedding: Vec<f32>) -> Self;
}
```

Binary format: `[MAGIC: "DLAT" 4B][VERSION: 1B][KV_DIM: 4B LE][EMBEDDING: kv_dim × f32 LE][BLAKE3: 32B]`

### Forward Pass Modification

At `layer_idx == n_layer / 2`, after K/V projections + LoRA, before cache write:

```rust
// transformer.rs — forward_base (Plan 038)
#[cfg(feature = "domain_latent")]
if layer_idx == config.n_layer / 2
    && let Some(dl) = domain_latent
{
    for i in 0..kvd {
        ctx.k[i] += dl.embedding[i];
        ctx.v[i] += dl.embedding[i];
    }
}
```

Cost: 2 × kv_dim additions at one layer. Zero allocations, zero RNG calls.

### Data Flow

```
Prompt tokens
     │
     ▼
┌─────────────┐
│ Layers 0..  │  Standard causal Transformer (no changes)
│   L/2 - 1   │
└─────┬───────┘
      │ X_{L/2}  [n_embd]
      │
      ├──► K/V projections ──► cache_k, cache_v
      │
      │    domain_embedding [kv_dim]  ◄── DomainLatent.embedding
      │         │
      │         ▼
      │    cache_k += domain_embedding
      │    cache_v += domain_embedding
      │
      ▼
┌─────────────┐
│ Layers L/2  │  Standard Transformer (conditioned on domain)
│   .. L-1    │
└─────┬───────┘
      │
      ▼
   Logits
```

### Why This Design

| Aspect | Free Transformer (Paper) | Our Domain Latent |
|--------|-------------------------|-------------------|
| Z source | VAE encoder (unsupervised) | Domain label (supervised) |
| Z dimension | 65536 (one-hot, H=16 bits) | kv_dim (continuous) |
| Training | From scratch + VAE loss | LoRA fine-tune + embedding |
| Inference | Uniform random Z sampling | Deterministic per domain |
| Requires new base model | Yes | No |

### Works with Bidirectional Prefill

Domain latent is injected in both `forward_base` (decode) and `forward_prefill` (prefill):

```rust
// transformer.rs — forward_prefill (Plan 038)
#[cfg(feature = "domain_latent")]
if layer_idx == config.n_layer / 2
    && let Some(dl) = domain_latent
{
    for i in 0..kvd {
        ctx.k[i] += dl.embedding[i];
        ctx.v[i] += dl.embedding[i];
    }
}
```

Both reader_lora and domain_latent condition the prefill phase. The second half of the model processes domain-informed K/V representations.

### GPU Training Support

`riir-gpu` provides training infrastructure:

```rust
// riir-gpu/src/domain_latent.rs
pub struct GpuDomainLatent {
    // GPU buffers for trainable domain latent (params, grads, m, v)
}

pub fn export_domain_latent(gpu_latent: &GpuDomainLatent, kv_dim: usize) -> DomainLatent;
// Downloads from GPU, saves as .dlat binary
```

`train_bomber.rs` trains LoRA + domain latent together, exporting both.

### Performance

| Metric | Value |
|--------|-------|
| Inference overhead | 2 × kv_dim additions at one layer (< 0.01% FLOPs) |
| Memory overhead | kv_dim × 4 bytes per domain (negligible) |
| Training overhead | One additional embedding vector (negligible vs LoRA) |

## Interaction Matrix

The four techniques compose without conflicts:

| Technique | Affects Prefill | Affects Decode | Feature Flag |
|-----------|:-:|:-:|-------------|
| Bidirectional Prefill | ✅ full attention | — | `router` |
| LoRA Switching | ✅ reader_lora | ✅ writer_lora | `router` |
| Sparse MLP | ✅ (if enabled) | ✅ (if enabled) | `sparse_mlp` |
| Domain Latent | ✅ K/V at L/2 | ✅ K/V at L/2 | `domain_latent` |

All are additive and backward-compatible. Standard `forward()` with no features works exactly as before.

## Key References

- [ZAYA1-VL-8B Technical Report](https://arxiv.org/abs/2504.02268) — Bidirectional prefix attention, token-specific LoRAs (Plan 025)
- [Sakana TwELL](https://sakana.ai/twell/) — Tile-wise ELLPACK sparse format (Plan 022 inspiration, GPU-specific; we use CPU index-packing)
- [The Free Transformer](https://arxiv.org/abs/2503.23153) — Mid-layer latent injection via K/V modulation (Plan 038)