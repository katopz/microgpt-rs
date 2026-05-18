# microgpt-rs: Core Architecture

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
    pub lora_rank: usize,           // LoRA adapter rank (Plan 008)
    pub lora_alpha: f32,            // LoRA scaling factor
    pub lora_dropout: f32,          // LoRA dropout probability
    pub lora_targets: Vec<String>,  // which projections to apply LoRA
    pub screening_threshold: f32,   // hard-trim cutoff for ScreeningPruner (Plan 021)
    pub sparse_threshold: f32,      // use sparse_mlp when alive ratio ≤ this (Plan 022)
    pub early_exit_patience: usize, // AutoTTS early exit patience (Plan 026)
    pub early_exit_gap: f32,        // AutoTTS early exit confidence gap
    // MTP Drafter thresholds (Plan 055: Gemma 4 MTP)
    pub mtp_activation_threshold: usize,    // enable MTP when n_embd >= this
    pub mtp_cluster_vocab_threshold: usize, // enable cluster LM head when vocab_size >= this
    pub mtp_shared_kv_prompt_threshold: usize, // enable shared KV for prompt when pos >= this
    pub mtp_cluster_size: usize,            // cluster size for round-robin vocab mapping
    // HLA Attention (Plan 057: Higher-order Linear Attention)
    pub hla_mode: HlaMode,                  // Standard, Hla, Ahla
    pub hla_normalize: bool,                // normalize HLA output
    pub hla_decay: f32,                     // decay factor for HLA state
    // D2F Discrete Diffusion Forcing (Plan 066)
    pub mask_token: usize,                  // mask token ID for dLLM
    pub attention_mode: AttentionMode,      // Causal, Bidirectional, BlockCausal, SpKv
    // SP-KV self-pruned KV attention (Plan 070)
    pub sp_kv_window: usize,               // sliding window size for SP-KV
    pub sp_kv_threshold: f32,              // gate threshold for SP-KV utility predictor
    pub sp_kv_predictor_hidden: usize,     // hidden dim for utility predictor MLP
    pub sp_kv_predictor_lr_mult: f32,      // learning rate multiplier for predictor
}
```
- All configs constructed via factory methods: `Config::micro()`, `Config::micro_lora()`, `Config::draft()`, `Config::game()`, `Config::bpe()`, `Config::bpe_draft()`, `Config::small_target()`, `Config::gqa_draft()`
- Validation: `n_head % n_kv_head == 0`, `n_embd == n_head * head_dim`
- `kv_dim()` helper returns `n_kv_head * head_dim`

### Key Enums (`types.rs`)

```rust
#[repr(u8)]
pub enum HlaMode {
    Standard,  // SDPA with KV cache (default)
    Hla,       // Symmetric second-order linear attention — O(1) per-token memory
    Ahla,      // Asymmetric second-order linear attention — lower state cost
}

#[repr(u8)]
pub enum AttentionMode {
    Causal,       // Standard autoregressive (default)
    Bidirectional, // Attend to ALL positions — dLLM masked prediction
    BlockCausal,  // Bidirectional within block, causal across blocks — D2F student
    SpKv,         // Self-pruned key-value attention with learned utility (Plan 070)
}
```

### InferenceOverrides (`types.rs`)

Runtime override fields that can be applied per-inference call without modifying the base `Config`:

```rust
pub struct InferenceOverrides {
    pub tree_budget: Option<usize>,
    pub temperature: Option<f32>,
    pub draft_lookahead: Option<usize>,
    // ...
}
```

Overrides are merged onto a base `Config` at inference time, allowing per-request parameter tuning without cloning or mutating the shared config.

## TransformerWeights (`transformer.rs`)
```rust
pub struct TransformerWeights {
    pub wte: Vec<f32>,              // [vocab_size, n_embd] — token embedding
    pub wpe: Vec<f32>,              // [block_size, n_embd] — position embedding
    pub lm_head: Vec<f32>,          // [vocab_size, n_embd] — output projection
    pub layers: Vec<LayerWeights>,  // per-layer weights (n_layer entries)
    pub mtp_activation_proj: Option<Vec<f32>>,  // MTP target activation projection (Plan 055)
    pub mtp_cluster_classifier: Option<Vec<f32>>, // MTP cluster classifier (Plan 055)
    pub mtp_cluster_map: Option<Vec<usize>>,     // MTP vocab cluster mapping (Plan 055)
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
    pub hidden_state: Vec<f32>, // [n_embd] — snapshot before lm_head (for REST/Validator)
    // Feature-gated buffers (allocated once, zero runtime cost when unused):
    lora_buf: Vec<f32>,       // [rank] — LoRA intermediate (always allocated)
    // #[cfg(feature = "sparse_mlp")]
    active_indices: Vec<usize>, // [mlp_hidden] — alive neuron indices (Plan 022)
    // #[cfg(feature = "sparse_mlp")]
    active_values: Vec<f32>,    // [mlp_hidden] — alive neuron values (Plan 022)
    // MTP Drafter buffers (Plan 055)
    mtp_context_buf: Vec<f32>,    // MTP projection intermediate buffer
    // TurboQuant buffers
    tq_dequant_pos: Vec<f32>,     // dequantized KV for current position
}
```
- Created once, reused across calls via `ctx.reset()`
- `hidden_state` is copied from `x` before lm_head projection — "free embedding" for vector search
- `lora_buf` avoids per-projection LoRA allocation; fused into `lora_apply()` in-place
- Sparse MLP buffers pack alive ReLU neurons for `sparse_matmul()` — only used when `alive_ratio ≤ sparse_threshold`

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

`forward()` is the **public API** — it delegates to internal `forward_base()` with feature-appropriate parameters:

```rust
// Public API — handles domain_latent feature gating internally
pub fn forward(
    ctx: &mut ForwardContext,
    weights: &TransformerWeights,
    cache: &mut MultiLayerKVCache,
    token: usize,
    pos: usize,
    config: &Config,
) -> &mut [f32]  // logits

// Internal — called by forward(), forward_prefill(), and generate_with_prefill()
// Accepts optional LoRA adapter and domain latent
fn forward_base(
    ctx, weights, cache, token, pos, config,
    lora: Option<&LoraAdapter>,        // cfg: always available
    domain_latent: Option<&DomainLatent>,  // cfg(feature = "domain_latent")
) -> &mut [f32]
```

Pipeline (inside `forward_base`):
1. **Embedding**: `x = wte[token] + wpe[pos]`
2. **Layer loop** (n_layer iterations):
   a. RMSNorm → QKV projection (GQA: K/V use kv_group)
   b. Store K/V in per-layer cache at position `pos`
   c. Multi-head attention (fused: score → softmax → weighted value)
   d. Output projection + residual add
   e. RMSNorm → MLP (matmul_relu + matmul) + residual add
   f. *(domain_latent)* At layer `n_layer / 2`: inject `DomainLatent` into K/V
3. **Snapshot**: `hidden_state = x` (before lm_head)
4. **LM Head**: `logits = lm_head @ x`

### GQA (Grouped-Query Attention)
When `n_kv_head < n_head`, K/V heads are shared:
- `kv_group = q_head * n_kv_head / n_head`
- K/V projection outputs `kv_dim` instead of `n_embd`
- 4× KV cache reduction for `n_head=8, n_kv_head=2`

## Math Kernels (`types.rs`)
All hot-path kernels are `#[inline(always)]` with `unsafe get_unchecked`:
- `matmul(out, w, x, rows, cols)` — out = W @ x — SIMD-accelerated via `simd_dot_f32` (Plan 060)
- `matmul_relu(out, w, x, rows, cols)` — fused matmul + ReLU — SIMD-accelerated with fused ReLU zero-clamp (Plan 060)
- `sparse_matmul(out, w, x, rows, cols, active_indices, active_values)` — skip dead ReLU neurons (Plan 022)
- `softmax(x)` — in-place, one-pass exp+sum, uses `inv_sum` multiply
- `softmax_scaled(x, scale)` — scaled softmax for attention (divides by sqrt(head_dim) before exp)
- `rmsnorm(x)` — in-place, two-pass with `inv_rms` multiply
- `attention_head(...)` — fused: score → softmax → weighted value (avoids separate softmax write)
- `sample_token(logits, rng)` — categorical sampling
- `lora_apply(output, lora, input, lora_buf)` — in-place LoRA delta: `output += (α/r) × B @ (A @ input)`

## SIMD Kernels (`simd.rs`, Plan 060)

Runtime SIMD detection and dispatch for hot-path operations:
- `SimdLevel` enum: `Scalar`, `Neon` (ARM), `Avx2` (x86_64)
- `simd_level()` — runtime detection of available SIMD level
- `simd_dot_f32(a, b, len)` — NEON `vfmaq_f32` / AVX2 `_mm256_mul_ps` dot product
- `simd_outer_product_acc(acc, a, b, m, n)` — rank-1 update for HLA SK, CQV, PKV
- `simd_matmul_rows(out, w, x, rows, cols)` — row-major matmul via SIMD dot
- `simd_matmul_relu_rows(out, w, x, rows, cols)` — SIMD matmul + fused ReLU clamp
- No dependencies — pure `core::arch::{aarch64, x86_64}` intrinsics
- Zero-cost dispatch: compile-time `#[cfg(target_arch)]` + runtime level check

## Additional Forward Variants (`transformer.rs`)

| Function | Description |
|----------|-------------|
| `forward_prefill(ctx, prefill, weights, cache, tokens, config, lora, domain_latent)` | Bidirectional prefill — all prompt tokens attend to all others (Plan 025) |
| `forward_paged(ctx, weights, paged_cache, token, pos, config, seq_idx)` | Paged KV cache forward — copy-on-write branch isolation |
| `forward_raven(ctx, weights, raven_cache, token, pos, config)` | Raven RSM forward — slot-based O(1) routing attention |
| `forward_turboquant(ctx, weights, tq_cache, token, pos, config)` | TurboQuant forward — bit-packed KV cache with dequantize-on-read |
| `forward_hla(ctx, weights, hla_cache, token, pos, config)` | Symmetric second-order HLA — O(d²) constant-state attention, SIMD-accelerated (Plan 057/060, `hla_attention`) |
| `forward_ahla(ctx, weights, ahla_cache, token, pos, config)` | Asymmetric AHLA — O(d·dv) constant-state attention, SIMD-accelerated (Plan 057/060, `hla_attention`) |
| `forward_with_domain_latent(ctx, weights, cache, token, pos, config, dl)` | Convenience wrapper — `forward_base` with domain latent only (no LoRA) |
| `forward_sp_kv(ctx, weights, sp_kv_cache, token, pos, config, predictors, bias)` | SP-KV self-pruned KV forward — utility-gated attention with learned predictor MLP (Plan 070, `sp_kv`) |

> **Plan 059 Note**: HLA is inference-only — SDPA→HLA distillation via LoRA shows KL divergence does NOT converge. HLA provides streaming O(1) attention for inference but cannot be trained to approximate SDPA outputs. Use DeltaMemoryState for facts/retrieval.

## MTP Projection (`transformer.rs`, Plan 055)

Multi-Token Prediction projection weights for draft model acceleration:
- `MtpProjection` — Projection weights for target-activation-based MTP drafting
- `project_target_activation()` — Projects hidden state to draft token logits
- `cluster_map_round_robin()` — Round-robin vocab cluster assignment
- `cluster_map_from_embeddings()` — Embedding-similarity-based cluster assignment
- Threshold-gated: features activate only when config thresholds are met (see `13_mtp_threshold_guide.md`)

## Generate (`transformer.rs`)
```rust
pub fn generate(ctx, cache, weights, config, rng, token, n_tokens) -> Vec<usize>
pub fn generate_into(ctx, cache, weights, config, rng, tokens, n_tokens)  // zero-alloc variant
pub fn generate_batch(ctx, cache, weights, config, rng, token, n_tokens, n_samples) -> Vec<Vec<usize>>
pub fn generate_with_prefill(
    ctx, prefill, cache, weights, config, rng,
    prompt_tokens, n_tokens,
    // Optional per-call overrides:
    lora_pair: Option<&LoraPair>,          // reader→writer LoRA switching
    domain_latent: Option<&DomainLatent>,  // mid-layer domain conditioning
) -> Vec<usize>
```
- Autoregressive: sample → feed back → repeat
- `generate_into` reuses pre-allocated buffers (zero-alloc hot path)
- `generate_batch` uses Rayon `par_iter` with per-worker contexts
- `generate_with_prefill` runs bidirectional prefill (reader LoRA) then switches to causal decode (writer LoRA), with optional domain latent injection
- `tokens_to_string(tokens, config)` — converts token IDs back to string via `id_to_vocab` lookup

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

## ScreeningPruner: Absolute Relevance (Plan 021)

Distilled from ["Screening Is Enough"](https://arxiv.org/abs/2604.01178) — upgrades binary pruning to **graded relevance**:

```rust
pub trait ScreeningPruner: Send + Sync {
    fn relevance(&self, depth: usize, token_idx: usize, parent_tokens: &[usize]) -> f32;
}
```

Score formula: `blended = parent_score + ln(P_llm) + ln(R)`

| Relevance R | ln(R) | Effect |
|---|---|---|
| 1.0 | 0.0 | No penalty — perfect match |
| 0.5 | -0.69 | Soft penalty — mediocre match |
| 0.0 | -∞ | **Hard trim** — branch killed |

`ConstraintPruner` adapts via `BinaryScreeningPruner(pruner)` (R ∈ {0.0, 1.0}). `WasmPruner` implements `ScreeningPruner` natively — loads optional WASM `relevance` export (Q16.16 fixed-point), falls back to binary `is_valid` if missing.

`config.screening_threshold` (default `0.0`) controls hard-trim cutoff. Set `> 0.0` to aggressively trim low-relevance branches.

## SpeculativeVerifier (Strategy Pattern)

Based on [Algorithm 1 from Leviathan et al. 2022](https://arxiv.org/pdf/2211.17192) — the verification strategy is swappable via trait:

```rust
pub trait SpeculativeVerifier: Send + Sync {
    fn speculate(&mut self, draft_weights, draft_config, token, pos, rng) -> Vec<usize>;
}
```

| Verifier | Availability | What it does |
|----------|--------------|--------------|
| `SimulatedVerifier` | always compiled | DFlash/AR draft → DDTree → simulated acceptance cap → bonus token from last marginal |
| `LeviathanVerifier` | always compiled | AR draft → target model p/q scoring → rejection sampling → residual distribution → bonus from target p(x). Proves Algorithm 1 works end-to-end. |

`SimulatedVerifier` is fast (no target model). `LeviathanVerifier` is the full Algorithm 1 — mathematically proven distribution-preserving, but needs large model asymmetry to be faster than pure AR.

## PPoT: Logit-Parameterized CPU Resampling (Plan 026 + 027)

After DFlash produces marginals and DDTree rejects all paths, PPoT identifies high-entropy positions in the saved marginals and resamples variant token sequences using **only CPU** — no additional GPU forward passes. Resampled paths are screened through `ScreeningPruner` for verification. This activates only on failure (zero overhead on success path).

Plan 027 extends baseline with TRT-inspired adaptive rescue: rejection memory (ring buffer of "don't" insights), per-sample strategy cycling across `TokenRule` variants, and self-consistency ranking for multi-valid variant selection. Knowledge accumulates within a generation session, biasing future resampling toward historically successful positions and rules.

```rust
pub enum TokenRule {
    Digit,      // prefer digit tokens
    Compare,    // prefer comparison operators
    Arithmetic, // prefer arithmetic operators
    Augment,    // prefer augmented assignment
    All,        // no preference
}
```

## Prompt Router: Batch-Level Domain Routing (Plan 023)

Inspired by [EMO: Pretraining Mixture of Experts for Emergent Modularity](https://arxiv.org/abs/2406.08732) — document-level routing constraints force experts to learn high-level semantic domains instead of syntax.

1. **Classify once** — `KeywordRouter` scores the prompt against domain keywords (V1, ~80% accuracy; embedding-based V2 via anyrag is planned)
2. **Select expert** — `ExpertRegistry` returns a `Box<dyn ScreeningPruner>` + optional LoRA path for the matched domain
3. **Lock for generation** — the selected `ScreeningPruner` is passed to `build_dd_tree_screened()`, preventing domain drift

```rust
let router = KeywordRouter::new(config.domain.clone());
let registry = ExpertRegistry::from_config(&config, pruner_dir);

let decision = router.route("solve this sudoku puzzle");
let expert = registry.get_expert(&decision.domain);
// expert.pruner is locked for the entire DDTree generation
```

Domains are defined in `domains.toml` — platform manages expert bundles via Web UI or MCP agent.

## Embedding Router: KV Cache Priming (Plan 024)

Extends keyword routing with **semantic embedding retrieval** from anyrag. When a user edits a known file, the system retrieves the most relevant document embedding, projects it to the draft model's hidden dimension, and injects it as KV cache priming context via `dflash_predict_conditioned_with`.

**Three-tier fallback** (graceful degradation when anyrag is unavailable):

```
1. Embedding search (POST /search/embedding)  ~200ms
   ↓ on failure
2. Domain classify (POST /classify/domain)     ~100ms
   ↓ on failure
3. KeywordRouter (local, no network)            <1ms
```

```rust
let router = EmbeddingRouter::new(
    embedding_config, domains, Box::new(TruncatePadProjector),
);

// Sync: delegates to KeywordRouter (no network)
let decision = router.route("fn validate_token(");

// Async: tries anyrag embedding search, falls back to keyword
let decision = router.route_async("fn validate_token(").await;

if let Some(embedding) = &decision.embedding {
    let projected = router.project_embedding(embedding, draft_config.n_embd);
    speculative_step_embedding_conditioned(&weights, &config, token, pos, &projected, &mut rng);
}
```

**Separation from target model conditioning:** `speculative_step_conditioned_with` uses the target model's hidden state (syntactic alignment). `speculative_step_embedding_conditioned` uses a retrieved embedding (semantic alignment). These are complementary signals.

## Bidirectional Prefill + Modality LoRA Switching (Plan 025)

Distilled from [ZAYA1-VL-8B Technical Report](https://arxiv.org/abs/2504.02268) — two production techniques adapted for the Python→Rust translation pipeline:

### 1. Bidirectional Prefill

During prefill, prompt tokens (Python code + anyRAG docs) attend to ALL other prompt tokens — no causal mask. Code is non-linear; a function body references a struct 3,000 tokens earlier. Generation tokens still use causal attention. Zero overhead on the decode hot path — prefill runs once per request.

### 2. Modality LoRA Switching

Load two LoRA adapters per domain — a `reader_lora` (active during prefill) and a `writer_lora` (active during decode). The switch is a reference swap at the prefill→decode boundary. Zero data movement.

```
  tokens[0..prompt_len]                    tokens[prompt_len..]
        │                                         │
   ┌────┴────┐                              ┌─────┴─────┐
   │ PREFILL │  bidirectional attention     │  DECODE   │  causal attention
   │         │  reader_lora active          │           │  writer_lora active
   └────┬────┘                              └─────┴─────┘
        │ KV cache populated                      │ generates tokens
        └──────────── shared KV cache ────────────┘
```

### LoraPair & PrefillContext

```rust
pub struct LoraPair {
    pub reader: Option<LoraAdapter>,  // active during bidirectional prefill
    pub writer: Option<LoraAdapter>,  // active during causal decode
}

pub struct PrefillContext {
    pub hidden: Vec<f32>,  // [prompt_len × n_embd] — pre-allocated once
}
```

Two-phase per layer (zero-copy):

| Phase | What | Buffers |
|-------|------|---------|
| A: KV Fill | Compute K/V for all positions → store in cache | Reuses `ForwardContext` per-position |
| B: Bidirectional Attend | Q attends to K/V[0..prompt_len] via `attention_head(t_n=prompt_len)` | `attention_head` unchanged — caller controls range |

```rust
let mut prefill = PrefillContext::new(&config);

// Bidirectional prefill with reader LoRA + optional domain latent
let logits = forward_prefill(&mut ctx, &mut prefill, &weights, &mut cache,
    &prompt_tokens, &config, lora_pair.reader.as_ref(), domain_latent);

// Causal decode — forward() delegates to forward_base(writer LoRA + domain latent)
let logits = forward(&mut ctx, &weights, &mut cache, token, pos, &config);
// Note: for explicit LoRA control during decode, use generate_with_prefill()
// which handles the reader→writer swap internally.
```

Domain config in `domains.toml`:
```toml
[[domain]]
name = "py2rs"
keywords = ["python", "rewrite", "translate"]
pruner = "syn_validator.wasm"
reader_lora = "python_reader.bin"   # active during bidirectional prefill
writer_lora = "rust_writer.bin"     # active during causal decode
```

LoRA application is fused in-place after each projection: `output += (α/r) × B @ (A @ input)`. Zero intermediate buffers — the delta accumulates directly into the output.