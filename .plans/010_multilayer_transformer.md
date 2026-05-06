# Plan 010: Multi-Layer Transformer — n_layer > 1 Support

## Objective

Extend the transformer from single-layer to multi-layer, enabling cLoRA-scale configs (4-8 layers) and making the architecture match real GPT-2/GPT-3 designs. This is a prerequisite for Plan 008 (wgpu LoRA training) at scale.

## The Problem

Current `TransformerWeights` stores per-layer weights as flat `Vec<f32>`:

```rust
pub struct TransformerWeights {
    pub attn_wq: Vec<f32>,  // single layer: [n_embd, n_embd]
    pub attn_wk: Vec<f32>,
    pub attn_wv: Vec<f32>,
    pub attn_wo: Vec<f32>,
    pub mlp_w1: Vec<f32>,   // single layer: [mlp_hidden, n_embd]
    pub mlp_w2: Vec<f32>,
    // wte, wpe, lm_head are shared across layers (no change needed)
}
```

And `forward()` runs the attention + MLP block exactly once — no layer loop:

```rust
// 3. QKV projections (single layer)
matmul(&mut ctx.q, &weights.attn_wq, &ctx.x, n, n);
// ... attention ...
// 8. MLP (single layer)
matmul_relu(&mut ctx.hidden, &weights.mlp_w1, &ctx.x, mlp_hidden, n);
```

For multi-layer, we need:
1. `Config.n_layer` field
2. Per-layer weight arrays: `Vec<Vec<f32>>` for qkv/wo/mlp weights
3. Per-layer KV caches
4. A layer loop in `forward()`

## Architecture

### Config

```rust
// types.rs — add n_layer

pub struct Config {
    pub vocab_size: usize,
    pub block_size: usize,
    pub n_embd: usize,
    pub n_head: usize,
    pub head_dim: usize,
    pub mlp_hidden: usize,
    pub n_layer: usize,        // NEW: number of transformer layers
    pub bos_token: usize,
    pub temperature: f32,
    pub draft_lookahead: usize,
    pub tree_budget: usize,
}

impl Config {
    pub fn micro() -> Self {
        Self {
            // ... existing fields ...
            n_layer: 1,  // backward compatible
        }
    }
    
    pub fn draft() -> Self {
        Self {
            // ... existing fields ...
            n_layer: 1,
        }
    }
    
    /// Multi-layer target model for cLoRA scale.
    /// 4 layers, embd=64, mlp=256 — ~550KB total.
    pub fn small_target() -> Self {
        Self {
            vocab_size: 4096,
            block_size: 256,
            n_embd: 64,
            n_head: 4,
            head_dim: 16,
            mlp_hidden: 256,
            n_layer: 4,
            bos_token: 0,
            temperature: 0.8,
            draft_lookahead: 5,
            tree_budget: 32,
        }
    }
}
```

### TransformerWeights

```rust
// transformer.rs — multi-layer weights

pub struct TransformerWeights {
    // Shared across layers (unchanged)
    pub wte: Vec<f32>,       // [vocab_size, n_embd]
    pub wpe: Vec<f32>,       // [block_size, n_embd]
    pub lm_head: Vec<f32>,   // [vocab_size, n_embd]
    
    // Per-layer weights (NEW: Vec of layers)
    pub layers: Vec<LayerWeights>,
}

pub struct LayerWeights {
    pub attn_wq: Vec<f32>,   // [n_embd, n_embd]
    pub attn_wk: Vec<f32>,
    pub attn_wv: Vec<f32>,
    pub attn_wo: Vec<f32>,
    pub mlp_w1: Vec<f32>,    // [mlp_hidden, n_embd]
    pub mlp_w2: Vec<f32>,    // [n_embd, mlp_hidden]
}

impl TransformerWeights {
    pub fn new(config: &Config, rng: &mut Rng) -> Self {
        let n = config.n_embd;
        let scale = (2.0 / (n * config.n_layer) as f32).sqrt(); // scale by depth
        
        let mut init = |len: usize| -> Vec<f32> {
            (0..len).map(|_| rng.normal() * scale).collect()
        };
        
        let layers: Vec<LayerWeights> = (0..config.n_layer)
            .map(|_| LayerWeights {
                attn_wq: init(n * n),
                attn_wk: init(n * n),
                attn_wv: init(n * n),
                attn_wo: init(n * n),
                mlp_w1: init(config.mlp_hidden * n),
                mlp_w2: init(n * config.mlp_hidden),
            })
            .collect();
        
        let emb_scale = (2.0 / n as f32).sqrt();
        let mut emb_init = |len: usize| -> Vec<f32> {
            (0..len).map(|_| rng.normal() * emb_scale).collect())
        };
        
        Self {
            wte: emb_init(config.vocab_size * n),
            wpe: emb_init(config.block_size * n),
            lm_head: emb_init(config.vocab_size * n),
            layers,
        }
    }
}
```

### KV Cache

```rust
// transformer.rs — per-layer KV cache

pub struct KVCache {
    pub key: Vec<f32>,    // [block_size, n_embd]
    pub value: Vec<f32>,
}

pub struct MultiLayerKVCache {
    pub layers: Vec<KVCache>,
}

impl MultiLayerKVCache {
    pub fn new(config: &Config) -> Self {
        Self {
            layers: (0..config.n_layer)
                .map(|_| KVCache::new(config))
                .collect(),
        }
    }
    
    pub fn reset(&mut self) {
        for layer in &mut self.layers {
            layer.reset();
        }
    }
}
```

### Forward Pass

```rust
// transformer.rs — multi-layer forward

pub fn forward<'a>(
    ctx: &'a mut ForwardContext,
    weights: &TransformerWeights,
    cache: &mut MultiLayerKVCache,   // changed from KVCache
    token: usize,
    pos: usize,
    config: &Config,
) -> &'a mut [f32] {
    let n = config.n_embd;
    let hd = config.head_dim;
    
    // 1. Embedding: x = wte[token] + wpe[pos]
    for i in 0..n {
        ctx.x[i] = weights.wte[token * n + i] + weights.wpe[pos * n + i];
    }
    
    // 2. Layer loop (was single pass, now iterates n_layer times)
    for (layer_idx, layer_weights) in weights.layers.iter().enumerate() {
        let layer_cache = &mut cache.layers[layer_idx];
        
        // RMSNorm → attention → residual
        rmsnorm(&mut ctx.x);
        ctx.xr[..n].copy_from_slice(&ctx.x[..n]);
        
        matmul(&mut ctx.q, &layer_weights.attn_wq, &ctx.x, n, n);
        matmul(&mut ctx.k, &layer_weights.attn_wk, &ctx.x, n, n);
        matmul(&mut ctx.v, &layer_weights.attn_wv, &ctx.x, n, n);
        
        // Store K, V in layer cache
        let pos_off = pos * n;
        unsafe {
            std::ptr::copy_nonoverlapping(ctx.k.as_ptr(), layer_cache.key.as_mut_ptr().add(pos_off), n);
            std::ptr::copy_nonoverlapping(ctx.v.as_ptr(), layer_cache.value.as_mut_ptr().add(pos_off), n);
        }
        
        // Multi-head attention
        let scale = 1.0 / (hd as f32).sqrt();
        ctx.attn_out[..n].fill(0.0);
        let t_n = pos + 1;
        for h in 0..config.n_head {
            unsafe {
                attention_head(&ctx.q, &layer_cache.key, &layer_cache.value,
                    &mut ctx.attn_out, &mut ctx.scores, h * hd, n, hd, t_n, scale);
            }
        }
        
        matmul(&mut ctx.x, &layer_weights.attn_wo, &ctx.attn_out, n, n);
        for i in 0..n { ctx.x[i] += ctx.xr[i]; }
        
        // RMSNorm → MLP → residual
        ctx.xr2[..n].copy_from_slice(&ctx.x[..n]);
        rmsnorm(&mut ctx.x);
        matmul_relu(&mut ctx.hidden, &layer_weights.mlp_w1, &ctx.x, config.mlp_hidden, n);
        matmul(&mut ctx.x, &layer_weights.mlp_w2, &ctx.hidden, n, config.mlp_hidden);
        for i in 0..n { ctx.x[i] += ctx.xr2[i]; }
    }
    
    // 3. Snapshot hidden state (Plan 009)
    ctx.hidden_state[..n].copy_from_slice(&ctx.x[..n]);
    
    // 4. LM Head
    matmul(&mut ctx.logits, &weights.lm_head, &ctx.x, config.vocab_size, n);
    
    &mut ctx.logits
}
```

### Memory Estimates

| Config | n_layer | n_embd | MLP | Total Weights |
|--------|---------|--------|-----|---------------|
| `micro` | 1 | 16 | 64 | ~50 KB |
| `draft` | 1 | 4 | 16 | ~3 KB |
| `bpe` | 1 | 32 | 128 | ~1.1 MB |
| `small_target` | 4 | 64 | 256 | ~550 KB |
| cLoRA target | 4 | 256 | 1024 | ~33 MB |
| cLoRA large | 8 | 512 | 2048 | ~266 MB |

## Migration Strategy

**Backward compatible**: `n_layer: 1` produces identical behavior to the current single-layer code. All existing tests pass without modification.

**Breaking changes** (all internal, not API):
1. `TransformerWeights` fields restructured — `attn_wq/k/v/o`, `mlp_w1/w2` → `layers: Vec<LayerWeights>`
2. `KVCache` → `MultiLayerKVCache` with per-layer caches
3. `forward()` signature: `cache: &mut KVCache` → `cache: &mut MultiLayerKVCache`
4. All callers of `forward()`, `generate()`, `KVCache::new()` must update

**Affected files**:
- `transformer.rs` — weights struct, cache struct, forward(), generate()
- `speculative/verifier.rs` — `LeviathanVerifier` creates `KVCache`
- `speculative/dflash.rs` — `dflash_predict*` creates `KVCache`
- `speculative/step.rs` — `speculative_step*` creates `KVCache`
- `benchmark.rs` — creates `KVCache`
- `main.rs` — creates `KVCache`

## Tasks

### Phase 1: Add n_layer to Config
- [x] 1.1 Add `n_layer: usize` to `Config` in `types.rs`
- [x] 1.2 Add `n_layer: 1` to `micro()`, `draft()`, `bpe()`, `bpe_draft()`
- [x] 1.3 Add `Config::small_target()` with `n_layer: 4`
- [x] 1.4 Run `cargo test` — all pass (n_layer unused, backward compat)

### Phase 2: Multi-Layer Weights
- [x] 2.1 Create `LayerWeights` struct in `transformer.rs`
- [x] 2.2 Change `TransformerWeights` to hold `layers: Vec<LayerWeights>`
- [x] 2.3 Update `TransformerWeights::new()` to create `n_layer` layers
- [x] 2.4 Fix weight init scaling: divide by `sqrt(n * n_layer)`

### Phase 3: Multi-Layer KV Cache
- [x] 3.1 Create `MultiLayerKVCache` with `Vec<KVCache>`
- [x] 3.2 Add `reset()` that resets all layers
- [x] 3.3 Keep `KVCache` as-is (used per-layer internally)

### Phase 4: Forward Pass Layer Loop
- [x] 4.1 Change `forward()` signature: `cache: &mut MultiLayerKVCache`
- [x] 4.2 Add layer loop: `for (idx, layer) in weights.layers.iter().enumerate()`
- [x] 4.3 Move attention + MLP inside the loop
- [x] 4.4 Access per-layer cache: `cache.layers[idx]`

### Phase 5: Update All Callers
- [x] 5.1 Update `generate()` — use `MultiLayerKVCache`
- [x] 5.2 Update `dflash_predict*` — use `MultiLayerKVCache`
- [x] 5.3 Update `LeviathanVerifier` — use `MultiLayerKVCache`
- [x] 5.4 Update `SimulatedVerifier` — no change (doesn't call forward directly)
- [x] 5.5 Update `benchmark.rs` — use `MultiLayerKVCache`
- [x] 5.6 Update `main.rs` — use `MultiLayerKVCache`

### Phase 6: Validation
- [x] 6.1 Add test: `forward_output_size` with `n_layer: 2`
- [x] 6.2 Add test: `forward_logits_finite` with `n_layer: 4`
- [x] 6.3 Add test: `generate_deterministic` with `Config::small_target()`
- [x] 6.4 Add test: `n_layer_1_matches_current_behavior` — regression check
- [x] 6.5 Add benchmark: single-layer vs multi-layer throughput
- [x] 6.6 Run `cargo test --all-features` — all tests pass
- [x] 6.7 Run `cargo clippy --all-features` — zero warnings
- [x] 6.8 Run `cargo run --release` — benchmark unchanged for micro config

## Key Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Performance regression for `n_layer: 1` | Same speed or faster | Layer loop compiles to same code when unrolled; benchmark to verify |
| ForwardContext buffer sizes | Activation buffers (q, k, v) still single [n_embd] | Correct — we reuse the same buffers per layer (zero-alloc) |
| GPU (Plan 008) buffer layout changes | GpuWeightBuffers must mirror LayerWeights | Plan 008 will adapt; this plan defines the CPU reference |
| Weight file format change | Can't load old single-layer weights | Single-layer weights trivially map to `layers: Vec<LayerWeights>` with one element |

## Expected Outcomes

1. `Config.n_layer` — configurable layer count
2. `LayerWeights` — per-layer weight struct
3. `MultiLayerKVCache` — per-layer KV cache
4. `forward()` — layer loop with zero extra allocations
5. `Config::small_target()` — 4-layer config for cLoRA scale
6. Full backward compatibility: `n_layer: 1` = current behavior

## Files to Create/Modify

| File | Action | Phase |
|------|--------|-------|
| `src/types.rs` | Add `n_layer` to Config, add `small_target()` | 1 |
| `src/transformer.rs` | `LayerWeights`, multi-layer `TransformerWeights`, layer loop | 2-4 |
| `src/speculative/dflash.rs` | Use `MultiLayerKVCache` | 5 |
| `src/speculative/verifier.rs` | Use `MultiLayerKVCache` | 5 |
| `src/speculative/step.rs` | Update signatures | 5 |
| `src/benchmark.rs` | Use `MultiLayerKVCache`, add multi-layer bench | 5-6 |
| `src/main.rs` | Use `MultiLayerKVCache` | 5 |

## References

- `.plans/007_compiler_in_the_loop_clora.md` — Config::bpe(), cLoRA scale
- `.plans/008_wgpu_lora_training.md` — GpuWeightBuffers, LoRA per-layer adapters
- `.research/01_Advanced Neuro-Symbolic Rust Translation.md` — §Foundation Engine