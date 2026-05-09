# Plan 025: Bidirectional Prefill + Modality LoRA Switching

**Branch:** `develop/feature/025_bidirectional_prefill_lora`
**Depends on:** `router` feature (for dual LoRA loading), existing `ForwardContext` / `MultiLayerKVCache`
**Research:** ZAYA1-VL-8B Technical Report (bidirectional prefix attention, token-specific LoRAs)

---

## Overview

Two production techniques distilled from ZAYA, adapted for the Python→Rust translation pipeline:

1. **Bidirectional Prefill**: During prefill, prompt tokens (Python code + anyRAG docs) attend to ALL other prompt tokens — no causal mask. Code is non-linear; a function body references a struct 3,000 tokens earlier. The model sees the whole file at once. Generation tokens still use causal attention. Zero overhead on the decode hot path — prefill runs once per request.

2. **Modality LoRA Switching**: Load two LoRA adapters per domain — a `reader_lora` (active during prefill) and a `writer_lora` (active during decode). When the router selects "py2rs", both adapters load. The switch is a reference swap at the prefill→decode boundary. Zero data movement.

Both are zero-copy: all buffers pre-allocated at startup, no `Vec::new()` in the request path, reference passing throughout.

**Caveat**: LoRA adapters require trained weights from Plan 008's GPU training pipeline. The infrastructure (loading, switching, application) ships in this plan. Actual trained `.bin` files come from training runs on Python↔Rust corpora.

---

## Architecture

### Data Flow

```
  tokens[0..prompt_len]                    tokens[prompt_len..]
        │                                         │
   ┌────┴────┐                              ┌─────┴─────┐
   │ PREFILL │  bidirectional attention     │  DECODE   │  causal attention
   │         │  reader_lora active          │           │  writer_lora active
   └────┬────┘                              └─────┬─────┘
        │ KV cache populated                      │ generates tokens
        │ logits[prompt_len-1] → first gen token  │
        └──────────── shared KV cache ────────────┘
```

### Bidirectional Prefill — Two-Phase Per Layer

For each transformer layer, the prefill splits into two phases:

**Phase A (KV Fill)**: Compute K/V projections for all prompt positions, store in `MultiLayerKVCache`. No attention computed yet. For single-layer models, input is token+position embeddings (computed on-the-fly, no buffer). For multi-layer, input comes from `PrefillContext::hidden` buffer.

**Phase B (Bidirectional Attend)**: For each position, compute Q, attend to K/V[0..prompt_len] via existing `attention_head` with `t_n = prompt_len`. Compute output projection + MLP. Update hidden state.

```
Layer L:
  Phase A: for p in 0..prompt_len { K[p], V[p] → cache }
  Phase B: for p in 0..prompt_len { Q[p] → attend(Q[p], K[0..prompt_len], V[0..prompt_len]) → MLP }
```

### Zero-Copy Buffer Strategy

| Buffer | Size | Allocation | Reuse |
|--------|------|------------|-------|
| `ForwardContext::x, q, k, v, attn_out, hidden, scores, logits` | Existing | `ForwardContext::new()` (once) | Per-position within Phase A/B |
| `PrefillContext::hidden` | `prompt_len × n_embd` | `PrefillContext::new()` (once) | Between layers (multi-layer) |
| `PrefillContext::lora_buf` | `[rank]` | `PrefillContext::new()` (once) | Per LoRA application |
| `MultiLayerKVCache` | Existing | Already pre-allocated | K/V storage across prefill+decode |
| `LoraAdapter::a, b` | Per adapter | `LoraAdapter::load()` (once per domain) | Every forward pass |

**Single-layer optimization**: `PrefillContext::hidden` is unused. Embeddings computed on-the-fly from `wte`/`wpe`. Zero extra memory.

**Multi-layer**: `PrefillContext::hidden` carries hidden states between layers. Allocated once to `max_prompt_len × n_embd`, reused across all requests.

### LoRA Application — In-Place Delta

```rust
/// Apply LoRA delta in-place: output += (α/r) × B @ (A @ input)
/// `lora_buf` is pre-allocated [rank] intermediate, zero alloc in hot path.
fn lora_apply(
    output: &mut [f32],
    lora: &LoraAdapter,
    input: &[f32],
    lora_buf: &mut [f32],  // pre-allocated [rank]
)
```

The LoRA delta is fused into the existing `matmul` output — no separate accumulation buffer. Applied after each Q/K/V/O/MLP projection when a LoRA is active.

### Attention Head — No API Change

The existing `attention_head` already accepts `t_n: usize` (number of KV positions to attend to). For prefill, pass `prompt_len`. For decode, pass `pos + 1`. The function is unchanged — the caller controls the attention range.

---

## Tasks

- [x] **Task 1: Add `LoraAdapter` CPU struct** (`src/types.rs`)
  - Production-grade CPU-side LoRA adapter for CPU inference path.
  - Mirrors `GpuLoraAdapter` fields but uses `Vec<f32>` instead of GPU buffers.
  - Loads from the same `.bin` format as `gpu/lora.rs::export_lora` (blake3 checksum, LORA magic).
  ```rust
  /// CPU-side LoRA adapter for CPU inference path.
  /// Loads from the same binary format as GpuLoraAdapter (Plan 008).
  /// Zero-copy: loaded once per domain, reference-passed during inference.
  pub struct LoraAdapter {
      pub a: Vec<f32>,       // [rank × in_dim]
      pub b: Vec<f32>,       // [out_dim × rank]
      pub rank: usize,
      pub alpha: f32,
      pub in_dim: usize,
      pub out_dim: usize,
  }

  impl LoraAdapter {
      /// Load from binary file (same format as gpu/lora.rs::export_lora).
      /// Format: [LORA(4) | version(4) | blake3(32) | payload...]
      pub fn load(path: &Path) -> Result<Self, String> { ... }
  }

  /// Apply LoRA delta in-place: output += (alpha/rank) × B @ (A @ input).
  /// `lora_buf` is a pre-allocated [rank] intermediate — zero alloc in hot path.
  pub fn lora_apply(
      output: &mut [f32],
      lora: &LoraAdapter,
      input: &[f32],
      lora_buf: &mut [f32],
  ) {
      let scale = lora.alpha / lora.rank as f32;
      // 1. hidden = A @ input  (rank × in_dim @ [in_dim] → [rank])
      matmul(lora_buf, &lora.a, input, lora.rank, lora.in_dim);
      // 2. delta = B @ hidden   (out_dim × rank @ [rank] → [out_dim])
      // 3. output += scale × delta  (fused into single loop)
      let mut delta_buf = [0.0f32; 0]; // stack for small dims, else reuse
      // Actually: fuse B@hidden + scale + add into output directly
      for r in 0..lora.out_dim {
          let row_off = r * lora.rank;
          let mut sum = 0.0f32;
          for k in 0..lora.rank {
              sum += unsafe { *lora.b.get_unchecked(row_off + k) }
                   * unsafe { *lora_buf.get_unchecked(k) };
          }
          unsafe { *output.get_unchecked_mut(r) += scale * sum; }
      }
  }
  ```
  - Reuses existing `matmul` for the A projection.
  - B projection fused directly into output (no intermediate delta buffer).
  - `lora_buf` passed in from `PrefillContext` or `ForwardContext`.

- [x] **Task 2: Add `PrefillContext` struct** (`src/transformer.rs`)
  - Pre-allocated buffers for bidirectional prefill. Created once, reused across requests.
  ```rust
  /// Pre-allocated context for bidirectional prefill phase.
  /// Created once at startup, reused across all requests. Zero alloc in request path.
  pub struct PrefillContext {
      /// Hidden states for all prompt positions, carried between layers.
      /// Size: [max_prompt_len × n_embd]. Only used when n_layer > 1.
      hidden: Vec<f32>,
      /// LoRA intermediate buffer. Size: [max_lora_rank].
      /// Reused for every LoRA application across all projections.
      lora_buf: Vec<f32>,
      /// Max prompt length this context supports.
      max_prompt_len: usize,
  }

  impl PrefillContext {
      pub fn new(config: &Config) -> Self {
          let max_prompt_len = config.block_size; // prefill can't exceed block_size
          Self {
              hidden: vec![0.0; max_prompt_len * config.n_embd],
              lora_buf: vec![0.0; config.lora_rank],
              max_prompt_len,
          }
      }
  }
  ```

- [x] **Task 3: Implement `forward_prefill`** (`src/transformer.rs`)
  - Bidirectional prefill: processes all prompt tokens, populates KV cache, returns logits for last prompt position.
  - Two-phase per layer: KV fill → bidirectional attention.
  - Zero-copy: reuses `ForwardContext` buffers per-position, `PrefillContext::hidden` between layers.
  ```rust
  /// Bidirectional prefill: process prompt tokens with full mutual attention.
  ///
  /// For each layer:
  ///   Phase A: Compute K/V for all positions → store in cache
  ///   Phase B: For each position, attend to ALL prompt K/V (bidirectional)
  ///
  /// Returns logits for the last prompt position (used to sample first gen token).
  /// KV cache is populated as a side effect, shared with subsequent decode calls.
  ///
  /// Zero-copy: no allocations. Uses ForwardContext buffers per-position,
  /// PrefillContext::hidden for multi-layer inter-layer state.
  pub fn forward_prefill<'a>(
      ctx: &'a mut ForwardContext,
      prefill: &mut PrefillContext,
      weights: &TransformerWeights,
      cache: &mut MultiLayerKVCache,
      tokens: &[usize],           // prompt tokens (borrowed, zero-copy)
      config: &Config,
      lora: Option<&LoraAdapter>, // reader LoRA (active during prefill)
  ) -> &'a mut [f32] {
      let prompt_len = tokens.len();
      let n = config.n_embd;
      let kvd = types::kv_dim(config);

      // Initialize hidden states from embeddings (only needed for multi-layer)
      // For n_layer == 1, we compute embeddings on-the-fly in Phase B.
      if config.n_layer > 1 {
          for (p, &token) in tokens.iter().enumerate() {
              let tok_off = token * n;
              let pos_off = p * n;
              for i in 0..n {
                  unsafe {
                      *prefill.hidden.get_unchecked_mut(p * n + i) =
                          *weights.wte.get_unchecked(tok_off + i)
                          + *weights.wpe.get_unchecked(pos_off + i);
                  }
              }
          }
      }

      for (layer_idx, layer_weights) in weights.layers.iter().enumerate() {
          // ── Phase A: Compute K/V for all positions → store in cache ──
          for (p, &token) in tokens.iter().enumerate() {
              // Get hidden state: from prefill.hidden (multi-layer) or compute embedding (single-layer)
              if config.n_layer > 1 {
                  ctx.x[..n].copy_from_slice(&prefill.hidden[p * n..(p + 1) * n]);
              } else {
                  let tok_off = token * n;
                  let pos_off = p * n;
                  for i in 0..n {
                      unsafe {
                          *ctx.x.get_unchecked_mut(i) = *weights.wte.get_unchecked(tok_off + i)
                              + *weights.wpe.get_unchecked(pos_off + i);
                      }
                  }
              }

              // Pre-attention norm (matches forward() exactly)
              rmsnorm(&mut ctx.x);
              if layer_idx == 0 && config.n_layer == 1 {
                  // Single-layer: skip first rmsnorm residual save (matches existing behavior)
              }
              rmsnorm(&mut ctx.x);

              // K/V projections
              matmul(&mut ctx.k, &layer_weights.attn_wk, &ctx.x, kvd, n);
              matmul(&mut ctx.v, &layer_weights.attn_wv, &ctx.x, kvd, n);

              // Apply reader LoRA to K and V projections (if active)
              if let Some(lora) = lora {
                  lora_apply(&mut ctx.k, lora, &ctx.x, &mut prefill.lora_buf);
                  lora_apply(&mut ctx.v, lora, &ctx.x, &mut prefill.lora_buf);
              }

              // Store K/V in cache
              let pos_off = p * kvd;
              let layer_cache = &mut cache.layers[layer_idx];
              unsafe {
                  std::ptr::copy_nonoverlapping(
                      ctx.k.as_ptr(), layer_cache.key.as_mut_ptr().add(pos_off), kvd,
                  );
                  std::ptr::copy_nonoverlapping(
                      ctx.v.as_ptr(), layer_cache.value.as_mut_ptr().add(pos_off), kvd,
                  );
              }
          }

          // ── Phase B: Bidirectional attention for all positions ──
          for (p, &token) in tokens.iter().enumerate() {
              // Get hidden state again (same as Phase A)
              if config.n_layer > 1 {
                  ctx.x[..n].copy_from_slice(&prefill.hidden[p * n..(p + 1) * n]);
              } else {
                  let tok_off = token * n;
                  let pos_off = p * n;
                  for i in 0..n {
                      unsafe {
                          *ctx.x.get_unchecked_mut(i) = *weights.wte.get_unchecked(tok_off + i)
                              + *weights.wpe.get_unchecked(pos_off + i);
                      }
                  }
              }

              // Pre-attention norm
              rmsnorm(&mut ctx.x);
              ctx.xr[..n].copy_from_slice(&ctx.x[..n]); // save residual
              rmsnorm(&mut ctx.x);

              // Q projection
              matmul(&mut ctx.q, &layer_weights.attn_wq, &ctx.x, n, n);
              if let Some(lora) = lora {
                  lora_apply(&mut ctx.q, lora, &ctx.x, &mut prefill.lora_buf);
              }

              // Bidirectional attention: attend to ALL prompt positions (t_n = prompt_len)
              let scale = 1.0 / (config.head_dim as f32).sqrt();
              ctx.attn_out[..n].fill(0.0);
              for h in 0..config.n_head {
                  let kv_group = h * config.n_kv_head / config.n_head;
                  unsafe {
                      attention_head(
                          &ctx.q, &cache.layers[layer_idx].key, &cache.layers[layer_idx].value,
                          &mut ctx.attn_out, &mut ctx.scores,
                          h * config.head_dim, kv_group * config.head_dim,
                          kvd, config.head_dim,
                          prompt_len, // ← BIDIRECTIONAL: full prompt range, not pos+1
                          scale,
                      );
                  }
              }

              // Output projection + residual
              matmul(&mut ctx.x, &layer_weights.attn_wo, &ctx.attn_out, n, n);
              if let Some(lora) = lora {
                  lora_apply(&mut ctx.x, lora, &ctx.attn_out, &mut prefill.lora_buf);
              }
              for i in 0..n {
                  unsafe { *ctx.x.get_unchecked_mut(i) += *ctx.xr.get_unchecked(i); }
              }

              // MLP: residual → RMSNorm → MLP → residual
              ctx.xr2[..n].copy_from_slice(&ctx.x[..n]);
              rmsnorm(&mut ctx.x);
              matmul_relu(&mut ctx.hidden, &layer_weights.mlp_w1, &ctx.x, config.mlp_hidden, n);
              if let Some(lora) = lora {
                  lora_apply(&mut ctx.hidden, lora, &ctx.x, &mut prefill.lora_buf);
              }
              // MLP w2
              #[cfg(feature = "sparse_mlp")]
              {
                  let alive = types::sparse_matmul(
                      &mut ctx.x, &layer_weights.mlp_w2, &ctx.hidden,
                      n, config.mlp_hidden, &mut ctx.active_indices, &mut ctx.active_values,
                  );
                  if (alive as f32 / config.mlp_hidden as f32) > (1.0 - config.sparse_threshold) {
                      matmul(&mut ctx.x, &layer_weights.mlp_w2, &ctx.hidden, n, config.mlp_hidden);
                  }
              }
              #[cfg(not(feature = "sparse_mlp"))]
              matmul(&mut ctx.x, &layer_weights.mlp_w2, &ctx.hidden, n, config.mlp_hidden);

              if let Some(lora) = lora {
                  lora_apply(&mut ctx.x, lora, &ctx.hidden, &mut prefill.lora_buf);
              }
              for i in 0..n {
                  unsafe { *ctx.x.get_unchecked_mut(i) += *ctx.xr2.get_unchecked(i); }
              }

              // Store output hidden state for next layer (multi-layer only)
              if config.n_layer > 1 {
                  prefill.hidden[p * n..(p + 1) * n].copy_from_slice(&ctx.x[..n]);
              }
          }
      }

      // LM Head (using last position's hidden state in ctx.x)
      ctx.hidden_state[..n].copy_from_slice(&ctx.x[..n]);
      matmul(&mut ctx.logits, &weights.lm_head, &ctx.x, config.vocab_size, n);

      &mut ctx.logits
  }
  ```

- [x] **Task 4: Add LoRA to decode `forward`** (`src/transformer.rs`)
  - Add optional `lora: Option<&LoraAdapter>` parameter to `forward`.
  - Apply LoRA delta after each projection (Q, K, V, O, MLP w1, MLP w2).
  - Zero-copy: `lora_buf` added to `ForwardContext` (pre-allocated, reused).
  - Update `ForwardContext::new` to allocate `lora_buf: vec![0.0; config.lora_rank]`.
  ```rust
  /// Updated forward signature (backward-compatible with default None)
  pub fn forward<'a>(
      ctx: &'a mut ForwardContext,
      weights: &TransformerWeights,
      cache: &mut MultiLayerKVCache,
      token: usize,
      pos: usize,
      config: &Config,
      lora: Option<&LoraAdapter>,  // writer LoRA (active during decode)
  ) -> &'a mut [f32]
  ```
  - All existing call sites pass `None` (no LoRA) — zero breaking change.
  - New call sites (with LoRA) pass `Some(&writer_lora)`.

- [x] **Task 5: Update call sites for `forward` signature** (`src/transformer.rs`, `src/speculative/step.rs`)
  - `generate_into`, `generate`, `generate_batch`: pass `None`.
  - `forward_paged`: add `lora` param, pass `None` in existing call sites.
  - `forward_raven`: add `lora` param, pass `None`.
  - `speculative_step_rollback`, `speculative_step_conditioned`, etc.: pass `None`.
  - All existing callers are non-breaking (new param has default `None` via wrapper).
  - Strategy: create `forward_base` with lora param, `forward` wrapper defaults to `None`.

- [x] **Task 6: Add dual LoRA to `DomainConfig` and `ExpertBundle`** (`src/router/types.rs`)
  ```toml
  # domains.toml
  [[domain]]
  name = "py2rs"
  keywords = ["python", "rewrite", "fastapi", "flask", "translate"]
  pruner = "syn_validator.wasm"
  reader_lora = "python_reader.bin"   # active during prefill
  writer_lora = "rust_writer.bin"     # active during decode
  ```
  ```rust
  // DomainConfig additions
  pub struct DomainConfig {
      // ... existing fields ...
      /// Path to reader LoRA adapter (active during bidirectional prefill).
      #[serde(default)]
      pub reader_lora: Option<String>,
      /// Path to writer LoRA adapter (active during causal decode).
      #[serde(default)]
      pub writer_lora: Option<String>,
  }

  /// A loaded LoRA pair for modality-specific inference.
  /// Reader is active during prefill, writer during decode.
  /// Switching is a reference swap — zero data movement.
  pub struct LoraPair {
      pub reader: Option<LoraAdapter>,
      pub writer: Option<LoraAdapter>,
  }

  // ExpertBundle additions
  pub struct ExpertBundle {
      pub domain: String,
      pub pruner: Box<dyn ScreeningPruner>,
      /// Legacy single LoRA path (backward compat, maps to writer_lora).
      pub lora_path: Option<PathBuf>,
      /// Loaded LoRA pair for modality switching.
      pub lora_pair: LoraPair,
  }
  ```
  - Keep `lora_path` for backward compat. If only `lora` is specified, it becomes `writer_lora` (decode-only).
  - If both `reader_lora` and `writer_lora` are specified, full ZAYA mode.

- [x] **Task 7: Load dual LoRAs in `ExpertRegistry`** (`src/router/registry.rs`)
  ```rust
  impl ExpertRegistry {
      fn resolve_lora_pair(domain: &DomainConfig, pruner_dir: &Path) -> LoraPair {
          let reader = domain.reader_lora.as_ref()
              .map(|p| {
                  let path = pruner_dir.join(p);
                  LoraAdapter::load(&path)
                      .unwrap_or_else(|e| {
                          eprintln!("[router] failed to load reader LoRA '{}': {e}", path.display());
                          // Graceful degradation: proceed without reader LoRA
                          panic!("LoRA load failed"); // or return None
                      })
                      .ok()
              })
              .flatten();
          // ... similar for writer ...
          LoraPair { reader, writer }
      }
  }
  ```
  - Graceful degradation: if a LoRA file fails to load, log warning, proceed without it.
  - The inference path checks `lora.is_some()` before applying — no penalty when absent.

- [x] **Task 8: Add `generate_with_prefill`** (`src/transformer.rs`)
  - End-to-end generation function that does prefill → decode with LoRA switching.
  ```rust
  /// Full generation pipeline: bidirectional prefill → causal decode.
  /// Switches from reader_lora to writer_lora at the prefill→decode boundary.
  /// Zero-copy: all buffers pre-allocated, no allocations in request path.
  pub fn generate_with_prefill(
      ctx: &mut ForwardContext,
      prefill: &mut PrefillContext,
      weights: &TransformerWeights,
      cache: &mut MultiLayerKVCache,
      config: &Config,
      rng: &mut Rng,
      prompt_tokens: &[usize],
      max_gen_tokens: usize,
      lora_pair: &LoraPair,
  ) -> Vec<usize> {
      // 1. Bidirectional prefill with reader LoRA
      let logits = forward_prefill(
          ctx, prefill, weights, cache,
          prompt_tokens, config,
          lora_pair.reader.as_ref(),
      );

      // 2. Sample first generation token from prefill output
      let mut p_dist = logits.to_vec();
      for p in p_dist.iter_mut() { *p /= config.temperature; }
      softmax(&mut p_dist);
      let mut token = sample_token(&p_dist, rng);

      let mut generated = vec![token];
      let mut pos = prompt_tokens.len();

      // 3. Causal decode with writer LoRA
      for _ in 1..max_gen_tokens {
          if pos >= config.block_size { break; }

          let logits = forward(
              ctx, weights, cache, token, pos, config,
              lora_pair.writer.as_ref(),  // ← WRITER LoRA
          );
          for logit in logits.iter_mut() { *logit /= config.temperature; }
          softmax(logits);

          token = sample_token(logits, rng);
          generated.push(token);
          pos += 1;

          if token == config.bos_token { break; }
      }

      generated
  }
  ```

- [x] **Task 9: Unit tests** (`src/transformer.rs` tests module)
  - `test_forward_prefill_logits_finite`: prefill produces finite logits
  - `test_forward_prefill_populates_cache`: KV cache has data for all prompt positions
  - `test_forward_prefill_logits_shape`: output shape matches vocab_size
  - `test_bidirectional_sees_all_positions`: verify attention scores cover full prompt range
  - `test_single_layer_no_hidden_buffer`: n_layer=1 doesn't use prefill.hidden
  - `test_multi_layer_hidden_carried`: n_layer>1 carries hidden states between layers
  - `test_lora_apply_delta`: LoRA application changes output (not zero)
  - `test_lora_load_roundtrip`: load a LoRA file, verify weights
  - `test_prefill_then_decode_shared_cache`: prefill + decode share same KV cache
  - `test_no_lora_matches_existing_forward`: forward with `None` matches old behavior
  - `test_generate_with_prefill_produces_tokens`: end-to-end generates valid tokens

- [x] **Task 10: Benchmark** (`examples/bidirectional_prefill_demo.rs`)
  - Benchmark `forward_prefill` vs sequential `forward` for same prompt length
  - Measure: time per prefill, cache population correctness
  - Compare: causal prefill (N calls to `forward`) vs bidirectional prefill (one `forward_prefill`)
  - Report: overhead ratio, memory usage

- [x] **Task 11: Update `domains.toml`** (`domains.toml`)
  - Add `reader_lora` / `writer_lora` fields to `py2rs` domain
  - Document dual LoRA config in comments
  ```toml
  [[domain]]
  name = "py2rs"
  keywords = ["python", "rewrite", "fastapi", "flask", "translate"]
  pruner = "syn_validator.wasm"
  reader_lora = "python_reader.bin"
  writer_lora = "rust_writer.bin"
  ```

- [x] **Task 12: Update README** (`README.md`)
  - Add "Bidirectional Prefill (Plan 025)" section under Architecture
  - Add "Modality LoRA Switching" subsection
  - Update Feature Flags
  - Update Project Structure

---

## File Change Summary

| File | Change |
|------|--------|
| `src/types.rs` | Add `LoraAdapter` struct, `lora_apply` function, `LoraAdapter::load` |
| `src/transformer.rs` | Add `PrefillContext`, `forward_prefill`, `forward_base`, update `forward` signature, add `generate_with_prefill`, update `ForwardContext` with `lora_buf` |
| `src/router/types.rs` | Add `reader_lora`/`writer_lora` to `DomainConfig`, add `LoraPair`, update `ExpertBundle` |
| `src/router/registry.rs` | Add `resolve_lora_pair`, load dual LoRAs |
| `domains.toml` | Add `reader_lora`/`writer_lora` fields |
| `README.md` | Add Plan 025 architecture section |

---

## Design Decisions

### 1. Two-Phase Per Layer (Not Batch)

Batch prefill (process all positions simultaneously) requires Q/K/V buffers of size `prompt_len × n_embd` — 80MB+ for a 5K-token prompt with 4096-dim embeddings. Two-phase per layer reuses existing single-token buffers, requiring only `prompt_len × n_embd` for multi-layer hidden states (and zero extra for single-layer).

### 2. `attention_head` Unchanged

The existing `attention_head` already accepts `t_n` as the number of KV positions. Bidirectional prefill passes `prompt_len`; causal decode passes `pos + 1`. No API change, no new branches in the hot loop.

### 3. LoRA Application After Each Projection

LoRA deltas are applied after `matmul` for Q, K, V, O, MLP-w1, MLP-w2. This matches the standard LoRA formulation: `output = W_base @ input + (α/r) × B @ A @ input`. The delta is fused directly into the output buffer — no intermediate accumulation.

### 4. Feature-Gated LoRA Loading

LoRA loading requires the `router` feature (for domain config parsing). `lora_apply` is always available (zero-cost when `lora` is `None`). No new feature flag needed.

### 5. Backward Compatible

- `forward` gains an optional `lora` parameter with default `None`.
- `DomainConfig` adds optional `reader_lora`/`writer_lora` fields.
- Existing configs, call sites, and tests work unchanged.

---

## Out of Scope

- GPU inference LoRA application (deferred to GPU forward pass integration)
- LoRA training / weight generation (Plan 008)
- Paged KV cache bidirectional prefill (follow-up: extend `forward_paged`)
- Raven KV cache bidirectional prefill (Raven uses O(1) slot memory, not standard cache)
- Multi-query attention (MHA → MQA conversion)
- Flash Attention (requires GPU, addresses same O(T²) problem differently)

---

## Performance Expectations

| Metric | Expected |
|--------|----------|
| Prefill overhead vs causal | ~2× (two passes per layer instead of one) |
| Decode throughput impact | Zero (untouched code path) |
| Memory overhead (single-layer) | Zero extra beyond `lora_buf` [rank] |
| Memory overhead (multi-layer) | `prompt_len × n_embd × 4` bytes |
| LoRA apply overhead | 2 × rank × dim FLOPs per projection |
| LoRA switch cost | Zero (reference swap) |

Prefill is a one-time cost per request. For a 5K-token prompt that generates 500 tokens, prefill is 1 call, decode is 500 calls. The 2× prefill overhead is amortized to near-zero.