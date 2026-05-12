# Plan 038: Free Transformer — Domain Latent Mid-Layer Injection

**Branch:** `feature/038_free_tf_domain_latent`
**Depends on:** Plan 025 (Bidirectional Prefill + LoRA), Plan 023 (Expert Registry)
**Research:** `.research/18_The_Free_Transformer_Latent_Injection.md`

---

## Overview

Distill the Free Transformer's mid-layer latent injection pattern into a **LoRA-compatible** domain conditioning mechanism. Instead of the paper's full VAE with binary mapper (requires training from scratch), inject a **learned domain embedding** at the middle layer of an existing model, fine-tuned via LoRA.

The Free Transformer paper proves that:
1. Injecting a latent signal at the middle layer (L/2+1) via K/V modulation is architecturally sound
2. Even 1/2 bit of latent information per token yields +5-11% on reasoning benchmarks
3. The injection point must be learned — random noise on an untrained model degrades quality

Our adaptation: replace the paper's unsupervised Z (65536-dim one-hot from VAE encoder) with a supervised domain embedding (small, explicit, LoRA-trainable). This trades the paper's "discover structure unsupervised" for "inject known structure explicitly" — which works with existing models and our LoRA pipeline.

---

## Architecture

### Data Flow

```
Prompt tokens
     │
     ▼
┌─────────────┐
│ Layers 0..  │  Standard causal Transformer
│   L/2 - 1   │  (no changes)
└─────┬───────┘
      │ X_{L/2}  [n_embd]
      │
      ├──► K/V projections ──► cache_k, cache_v
      │
      │    domain_embedding [kv_dim]  ◄── DomainConfig.domain_latent
      │         │
      │         ▼
      │    cache_k += domain_embedding
      │    cache_v += domain_embedding
      │
      ▼
┌─────────────┐
│ Layers L/2  │  Standard causal Transformer
│   .. L-1    │  (conditioned on domain embedding)
└─────┬───────┘
      │
      ▼
   Logits
```

### Weight Addition

```rust
/// Domain latent embedding for mid-layer conditioning.
/// Shape: [kv_dim] — one per domain, added to K and V at layer L/2.
/// Trained as part of LoRA fine-tuning (riir-burner).
pub struct DomainLatent {
    pub embedding: Vec<f32>,  // [kv_dim]
}
```

### Forward Pass Modification

In `forward_base`, at the mid-layer, before cache write:

```rust
// At layer_idx == n_layer / 2, after K/V projections:
if let Some(domain_latent) = domain_latent {
    for i in 0..kvd {
        ctx.k[i] += domain_latent.embedding[i];
        ctx.v[i] += domain_latent.embedding[i];
    }
}
```

Cost: 2 × kv_dim additions. Zero allocations, zero RNG calls.

### Why Not Full Free Transformer?

| Aspect | Free Transformer (Paper) | Our Domain Latent |
|--------|-------------------------|-------------------|
| Z source | VAE encoder (unsupervised) | Domain label (supervised) |
| Z dimension | 65536 (one-hot, H=16 bits) | kv_dim (continuous) |
| Training | From scratch + VAE loss | LoRA fine-tune + embedding |
| Inference | Uniform random Z sampling | Deterministic per domain |
| Requires new base model | Yes | No |
| Discoverable structure | Yes (unsupervised) | No (explicit) |

---

## Tasks

- [x] **Task 1: DomainLatent type** (`src/types.rs`) ✅
  - `pub struct DomainLatent { pub embedding: Vec<f32> }` — shape `[kv_dim]`
  - `pub fn load(path: &Path) -> Result<Self>` — load from binary file
  - `pub fn save(&self, path: &Path) -> Result<()>` — save to binary file
  - `pub fn zeros(kv_dim: usize) -> Self` — zero-initialized constructor
  - `pub fn from_vec(embedding: Vec<f32>) -> Self` — from raw vector
  - Binary format: `[MAGIC: "DLAT" 4B][VERSION: 1B][KV_DIM: 4B LE][EMBEDDING: kv_dim × f32 LE][BLAKE3: 32B]`
  - Unit tests: roundtrip, invalid magic, checksum mismatch, file too small, zeros

- [x] **Task 2: Mid-layer injection in forward_base** (`src/transformer.rs`) ✅
  - Added `#[cfg(feature = "domain_latent")] domain_latent: Option<&DomainLatent>` parameter to `forward_base`
  - At `layer_idx == config.n_layer / 2`, after K/V projections + LoRA, add domain_latent to `ctx.k` and `ctx.v` before cache write
  - Gate behind `#[cfg(feature = "domain_latent")]` feature flag
  - Updated `forward()` wrapper to dispatch with cfg-gated args
  - Added `forward_with_domain_latent()` public wrapper (feature-gated)
  - Unit test: `test_domain_latent_changes_logits` — non-zero embedding changes output
  - Unit test: `test_domain_latent_zero_embedding_same_logits` — zero embedding is identity
  - Unit test: `test_forward_with_domain_latent_wrapper` — public API works

- [ ] **Task 3: DomainLatent in Config** (`src/types.rs`) — partial
  - ✅ `DomainLatent` type exists with `load()`, `save()`, `zeros()`, `from_vec()`
  - ⏳ `domain_latent_path: Option<PathBuf>` in Config skipped — deferred to runtime config
  - ⏳ Lazy loading alongside `LoraAdapter` deferred — no runtime config system yet
  - ⏳ Integration test with lora + domain_latent deferred

- [x] **Task 4: Prefill integration** (`src/transformer.rs`) ✅
  - `forward_prefill` gained `#[cfg(feature = "domain_latent")] domain_latent` parameter
  - Injection at layer L/2 Phase A (K/V computation), same pattern as `forward_base`
  - Bidirectional prefill + domain_latent conditioning work together
  - Unit test: `test_domain_latent_prefill_changes_logits` — prefill output differs with latent
  - Unit test: `test_domain_latent_prefill_then_decode` — prefill→decode pipeline works

- [x] **Task 5a: riir-gpu training support (game domain)** (`riir-ai/crates/riir-gpu`) ✅
  - `GpuDomainLatent` — GPU buffers for trainable domain latent (params, grads, m, v)
  - `export_domain_latent()` — download from GPU, save as `.dlat` binary (DLAT format)
  - `DomainLatentAdamWStep` + `adamw_step_cpu()` — CPU AdamW step for domain latent
  - `AdamWOptimizer::step_domain_latent()` — GPU AdamW dispatch for domain latent
  - `train_bomber.rs` updated to train LoRA + domain latent together, export both
  - CPU fallback when no GPU available
  - 4 tests: zeros init, from_vec roundtrip, export format, AdamW convergence

- [ ] **Task 5b: riir-burner training support (language domain)** — deferred
  - For larger language models (4B+ params) that need Python training pipeline
  - Deferred until language model LoRA training pipeline matures

- [ ] **Task 6: Expert Registry integration** (`src/router/registry.rs`) — deferred
  - No `src/router/registry.rs` exists yet — Expert Registry (Plan 023) not implemented
  - `ExpertBundle` gains optional `domain_latent: Option<DomainLatent>`
  - When router resolves a domain, load the corresponding domain_latent
  - Pass to `forward()` and `forward_prefill()` via new parameter
  - Deferred until Expert Registry is built

---

## File Change Summary

| File | Change | Status |
|------|--------|--------|
| `src/types.rs` | `DomainLatent` struct, `load()`, `save()`, binary format, 5 tests | ✅ Done |
| `src/transformer.rs` | `forward_base` + `forward_prefill`: mid-layer injection, 5 tests | ✅ Done |
| `Cargo.toml` | `domain_latent` feature flag + added to `full` | ✅ Done |
| `src/router/registry.rs` | `ExpertBundle` includes `DomainLatent` | ⏳ Deferred |
| `riir-gpu/src/domain_latent.rs` | `GpuDomainLatent`, export, CPU AdamW, 4 tests | ✅ Done |
| `riir-gpu/src/optimizer.rs` | `step_domain_latent()` method | ✅ Done |
| `riir-gpu/examples/train_bomber.rs` | Train LoRA + domain latent, export both | ✅ Done |
| `riir-burner/train_lora.py` | Language model training (future) | ⏳ Deferred |

**Tests:** 260 pass (with `domain_latent`), 255 pass (without). 5 new domain_latent tests.

---

## Design Decisions

### 1. Deterministic (Not Random) Z

The paper uses random Z sampling to enable diverse generation. We use deterministic domain embeddings because:
- Our routing already decides the domain — no need to "discover" it via Z
- Deterministic Z means reproducible outputs for the same domain
- If we want diversity, we sample multiple domain latents (cf. Plan 030 Bandit)

### 2. Mid-Layer (Not Input-Layer) Injection

The paper proves mid-layer is the right point: too early starves the encoder, too late starves the decoder. Our bidirectional prefill (Plan 025) already processes the full prompt at all layers — the domain latent at mid-layer provides an additional structural signal that the second half of the model can leverage.

### 3. Feature-Gated

Like `sparse_mlp` and `ppot`, domain_latent is behind a feature flag. Models without trained domain latents work exactly as before. No performance regression on the standard path.

### 4. kv_dim (Not n_embd)

We inject into K and V, not into the residual stream. K/V dimension is `kv_dim = n_kv_head * head_dim`, which may differ from `n_embd` with GQA. The domain latent must match kv_dim to be added to K/V.

---

## Performance Expectations

- **Inference overhead:** 2 × kv_dim additions at one layer. For n_embd=384, kv_dim=96: 192 additions. < 0.01% of total FLOPs.
- **Memory overhead:** kv_dim × 4 bytes per domain. For kv_dim=96: 384 bytes. Negligible.
- **Training overhead:** One additional embedding vector to train. Negligible compared to LoRA matrices.
- **Expected quality gain:** Unclear without experiment. The paper shows +5-11% with unsupervised Z. Supervised domain Z should be at least as informative per bit (we know what the domain is). Realistic expectation: +2-5% on domain-specific benchmarks (code gen, translation).

---

## Out of Scope

- Full VAE training with KL divergence loss (requires training from scratch)
- Binary mapper (H=16 bits → 65536-dim one-hot) — overkill for supervised domain labels
- Random Z sampling at inference (useful only with VAE-trained models)
- Z-resampling in PPoT (violates CPU-only constraint, requires new forward passes)
- Multi-Z inference with DDTree merge (interesting but needs Free Transformer base model)

---

## Open Questions

1. **Should domain_latent be per-layer or single-vector?** The paper injects Z at one layer. We could inject at every layer in the second half (L/2..L). More expressive but more parameters to train.
2. **Should we add to Q as well?** The paper only adds to K/V. Adding to Q would let the model "query for" domain-specific features. Unexplored territory.
3. **Can we distill a domain_latent from existing LoRA weights?** If LoRA captures domain-specific adjustments, maybe the "average LoRA delta" at mid-layer approximates a domain_latent. This would avoid retraining.