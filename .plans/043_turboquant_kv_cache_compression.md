# Plan 043: TurboQuant KV Cache Compression — Near-Optimal Vector Quantization for Inference

> **Research:** `microgpt-rs/.research/20_TurboQuant_Online_Vector_Quantization.md`
> **Raw reference:** `.raw/turboquant/` (Python implementation with 35 passing tests)
> **Related:** Plan 020 (Raven RSM), Plan 042 (TTT Feedback Loop), riir-gpu `attention_score.wgsl`
> **Branch:** `develop/feature/043_turboquant_kv_cache`

---

## Tasks

- [x] **Task 1: Benchmark baseline KV cache memory and attention fidelity**
- [x] **Task 2: Implement `TurboQuantCodebook` (Lloyd-Max scalar quantizer)**
- [x] **Task 3: Implement `RandomRotation` (QR-based orthogonal matrix)**
- [x] **Task 4: Implement `TurboQuantKVCache` struct**
- [x] **Task 5: Implement `forward_turboquant()` attention path**
- [x] **Task 6: Add `attention_score_tq.wgsl` GPU kernel in riir-gpu**
- [x] **Task 7: Add benchmarks and quality validation**
- [x] **Task 8: Commit with conventional message**

---

## Overview

TurboQuant compresses KV cache entries from f32 (32 bits) to 2–4 bits per coordinate with near-optimal distortion (Theorem 1: within 2.7× of Shannon lower bound). For our draft model (head_dim=4) and target models (head_dim=64–256), this gives 8–16× KV cache compression with provable quality preservation.

The method is **data-oblivious** (no calibration), **online** (per-token, no preprocessing), and **composable** with existing Raven RSM (Plan 020) — Raven compresses the sequence dimension, TurboQuant compresses the embedding dimension.

### Why This Matters

1. **KV cache is the memory bottleneck** for long-context inference. `MultiLayerKVCache` stores f32 keys+values growing linearly with sequence length.
2. **TurboQuant is proven.** Theorems 1-3 give formal bounds. The `.raw/turboquant/` reference validates empirically: 3.5 bits = quality neutral on Llama-3.1-8B.
3. **Orthogonal to existing compression.** Raven compresses tokens→slots (sequence). Percepta compresses 2D points→hull (spatial). TurboQuant compresses f32→bits (precision). All compose.
4. **Unbiased attention scores.** Algorithm 2 guarantees E[estimated <Q,K>] = true <Q,K>. Attention patterns preserved on average.

### Compression Target

| Component | Current | After TQ (3-bit) | Ratio |
|-----------|---------|-------------------|-------|
| Key cache per token | head_dim × 4 bytes | head_dim × 3 bits + 4 bytes (norm) | 8.5× |
| Value cache per token | head_dim × 4 bytes | head_dim × 3 bits + 4 bytes (norm) | 8.5× |
| Combined KV per token | 2 × head_dim × 4 bytes | 2 × (head_dim × 3 bits + 4 bytes) | 8.5× |
| Codebook per layer | — | 2³ × 4 bytes = 32 bytes | negligible |
| Rotation matrix per layer | — | head_dim² × 4 bytes | 64KB (hd=128) |
| QJL matrix per layer | — | head_dim² × 4 bytes | 64KB (hd=128) |

For head_dim=128, 32K context, 32 layers: 32K × 128 × 4 × 2 × 32 = **1 GB** → **~120 MB** at 3-bit.

---

## Architecture

### Data Flow

```
Token t arrives:
  1. Compute K[t], V[t] as f32 (existing forward pass)
  2. ||k|| = K[t].norm(),  k_unit = K[t] / ||k||
  3. y = Π · k_unit                          (random rotation)
  4. idx[j] = searchsorted(boundaries, y[j])  (quantize per coord)
  5. packed = bit_pack(idx, bits=3)           (3 bits per coord → u8 array)
  6. Store: packed_indices, ||k||             (instead of f32 array)

Attention scoring:
  Option A (CPU, dequantize):
    1. Unpack indices → centroid lookup → ŷ
    2. k̂ = Πᵀ · ŷ · ||k||                  (dequantize)
    3. score = <Q, k̂> · scale               (standard attention)

  Option B (GPU, score in quantized space):
    1. Compute <Q, centroid[indices]>         (codebook gather + dot)
    2. Compute QJL residual contribution      (sign bits + projection)
    3. score = mse_contrib + qjl_contrib      (unbiased estimate)
```

### New Types

```rust
/// Precomputed codebook for a specific (dimension, bit-width) pair.
pub struct TurboQuantCodebook {
    /// Sorted centroids (2^b values).
    pub centroids: Vec<f32>,
    /// Decision boundaries (2^b + 1 values, includes -1.0 and 1.0).
    pub boundaries: Vec<f32>,
    /// MSE per coordinate (from Lloyd-Max optimization).
    pub mse_per_coord: f64,
    /// Dimension this codebook was computed for.
    pub dim: usize,
    /// Bit-width (2, 3, or 4).
    pub bits: u8,
}

/// Per-layer quantization state.
pub struct TurboQuantLayer {
    /// Random rotation matrix Π (dim × dim).
    pub rotation: Vec<f32>,
    /// QJL projection matrix S (dim × dim).
    pub qjl_matrix: Vec<f32>,
    /// Codebook for keys.
    pub key_codebook: TurboQuantCodebook,
    /// Codebook for values.
    pub val_codebook: TurboQuantCodebook,
}

/// Compressed KV cache using TurboQuant.
pub struct TurboQuantKVCache {
    /// Per-layer quantization state (rotation, codebook).
    pub layers: Vec<TurboQuantLayer>,
    /// Per-layer bit-packed key indices: [layer][token_block] = u8 packed array.
    pub key_indices: Vec<Vec<Vec<u8>>>,
    /// Per-layer key norms: [layer][token] = f32.
    pub key_norms: Vec<Vec<f32>>,
    /// Per-layer bit-packed value indices.
    pub val_indices: Vec<Vec<Vec<u8>>>,
    /// Per-layer value norms.
    pub val_norms: Vec<Vec<f32>>,
    /// Current position (number of cached tokens).
    pub pos: usize,
    /// Number of layers.
    pub n_layers: usize,
    /// KV dimension (n_kv_head × head_dim).
    pub kv_dim: usize,
    /// Bits per coordinate for keys.
    pub key_bits: u8,
    /// Bits per coordinate for values.
    pub val_bits: u8,
}
```

---

## Tasks

### Task 1: Baseline Benchmark

Before any changes, capture current KV cache metrics:

```bash
cd microgpt-rs && cargo bench --quiet 2>&1 | tee .plans/043_baseline.txt
```

| Metric | How to Extract | Target for Comparison |
|--------|---------------|----------------------|
| `forward (flat)` throughput | From bench | Baseline |
| `forward (flat)` memory/step | KV size growth | Baseline |
| `forward_raven` throughput | From bench | Baseline |
| Attention score fidelity | cos_sim(original, reconstructed) | Will measure after Task 5 |
| KV bytes per token | `kv_dim × 4 × 2` (K+V) | Will compare to packed size |

### Task 2: Implement `TurboQuantCodebook`

Port `.raw/turboquant/turboquant/codebook.py` to Rust.

**File:** `microgpt-rs/src/turboquant/codebook.rs`

```rust
/// Compute Lloyd-Max optimal codebook for Beta distribution on [-1, 1].
///
/// After random rotation of d-dimensional unit vectors, each coordinate
/// follows f(x) = Γ(d/2) / (√π · Γ((d-1)/2)) · (1-x²)^((d-3)/2)
/// which converges to N(0, 1/d) for large d.
pub fn compute_codebook(dim: usize, bits: u8) -> TurboQuantCodebook
```

Dependencies:
- `statrs` for `gamma::ln_gamma` (Gamma function)
- Numeric integration via trapezoidal rule (no external crate needed)
- Lloyd-Max iteration: initialize at quantile midpoints, iterate centroid/boundary updates

Test plan:
- Codebook for (d=128, b=2): centroids should be ≈ ±0.453/√d, ±1.51/√d
- Codebook for (d=128, b=3): 8 centroids, MSE per coord ≈ 0.03/d
- Precompute for common (dim, bits) pairs, cache to disk as JSON

~200 lines.

### Task 3: Implement `RandomRotation`

Port `.raw/turboquant/turboquant/rotation.py` to Rust.

**File:** `microgpt-rs/src/turboquant/rotation.rs`

```rust
/// Generate random orthogonal matrix via QR decomposition.
/// Deterministic from seed for reproducibility.
pub fn generate_rotation_matrix(dim: usize, seed: u64) -> Vec<f32>

/// Generate QJL projection matrix (i.i.d. N(0,1) entries).
pub fn generate_qjl_matrix(dim: usize, seed: u64) -> Vec<f32>
```

Dependencies:
- `nalgebra` for `QR::new()` decomposition
- `rand::rngs::StdRng` with `SeedableRng` for deterministic random
- `rand_distr::Normal` for Gaussian matrix entries

Test plan:
- Rotation matrix is orthogonal: `Π · Πᵀ ≈ I`
- Deterministic: same seed → same matrix
- QJL matrix has correct variance: E[S_{ij}²] ≈ 1.0

~80 lines.

### Task 4: Implement `TurboQuantKVCache`

**File:** `microgpt-rs/src/turboquant/kv_cache.rs`

```rust
impl TurboQuantKVCache {
    pub fn new(config: &Config, key_bits: u8, val_bits: u8) -> Self

    /// Quantize and store a key vector at given position.
    pub fn store_key(&mut self, layer: usize, key: &[f32])

    /// Quantize and store a value vector at given position.
    pub fn store_value(&mut self, layer: usize, value: &[f32])

    /// Dequantize key at position (for CPU attention scoring).
    pub fn dequantize_key(&self, layer: usize, pos: usize) -> Vec<f32>

    /// Dequantize value at position.
    pub fn dequantize_value(&self, layer: usize, pos: usize) -> Vec<f32>

    /// Reset cache for new sequence.
    pub fn reset(&mut self)

    /// Bytes stored per token (for benchmarking).
    pub fn bytes_per_token(&self) -> usize

    /// Compression ratio vs f32 KV cache.
    pub fn compression_ratio(&self) -> f64
}
```

Bit-packing:
- 2-bit: 4 values per u8
- 3-bit: stored as 4-bit (2 per u8) for simplicity (wastes 1 bit)
- 4-bit: 2 values per u8

Test plan:
- Round-trip: quantize → dequantize → cos_sim > 0.99 at 4-bit
- MSE: matches codebook prediction for (dim, bits)
- Memory: `bytes_per_token()` matches expected packed size
- Reset clears all data

~250 lines.

### Task 5: Implement `forward_turboquant()` Attention Path

**File:** `microgpt-rs/src/turboquant/forward.rs`

```rust
/// Forward pass using TurboQuantKVCache.
/// Same logic as forward() but stores to / reads from compressed cache.
pub fn forward_turboquant(
    ctx: &mut ForwardContext,
    weights: &TransformerWeights,
    cache: &mut TurboQuantKVCache,
    token: usize,
    pos: usize,
    config: &Config,
) -> &mut [f32]
```

This mirrors the existing `forward()` but:
1. After computing K[t], V[t], calls `cache.store_key()` and `cache.store_value()`
2. During attention, calls `cache.dequantize_key()` and `cache.dequantize_value()` for each position
3. Or: uses quantized attention scoring (Task 6 for GPU path)

For CPU path (this task): dequantize then standard dot product.
For GPU path (Task 6): score directly in quantized space.

~150 lines (largely copied from existing `forward()` with cache I/O changes).

### Task 6: Add `attention_score_tq.wgsl` GPU Kernel

**File:** `riir-ai/crates/riir-gpu/src/kernels/attention_score_tq.wgsl`

New compute shader implementing Algorithm 2's inner product estimator:

```wgsl
// 1. Load bit-packed indices for position t
// 2. Gather centroids: y_hat[d] = codebook[indices[t][d]]
// 3. Rotate back: k_hat = y_hat · Pi_T (or pre-rotated codebook)
// 4. MSE contribution: dot(query, k_hat) · scale
// 5. QJL contribution: dot(query · S_T, sign_bits) · qjl_scale · residual_norm
// 6. score = mse + qjl (unbiased)
```

Plus Rust bindings in `riir-ai/crates/riir-gpu/src/forward.rs`:
- New `dispatch_attention_tq()` method
- Upload codebook, rotation, QJL as storage buffers
- Upload bit-packed indices + norms as storage buffers

~100 lines WGSL + ~200 lines Rust binding.

### Task 7: Benchmarks and Quality Validation

Following the `bench_raven_vs_flat_cache` pattern:

```rust
/// Benchmark: flat f32 KV cache vs TurboQuant compressed cache.
pub fn bench_turboquant_vs_flat_cache(config: &Config) -> BenchResult

/// Quality: cos_sim between original and reconstructed KV entries.
pub fn bench_turboquant_quality(config: &Config) -> QualityResult

/// Attention fidelity: correlation between f32 and TQ attention scores.
pub fn bench_turboquant_attention_fidelity(config: &Config) -> FidelityResult
```

Metrics to capture:

| Metric | Target | Method |
|--------|--------|--------|
| Throughput (tok/s) | Within 10% of flat | Time N forward passes |
| cos_sim (key reconstruction) | > 0.99 at 4-bit | Quantize→dequantize→compare |
| cos_sim (value reconstruction) | > 0.95 at 3-bit | Same |
| Attention score correlation | > 0.98 at 3-bit | Compare f32 vs TQ scores |
| Compression ratio | > 8× at 3-bit | bytes_per_token / (kv_dim×4) |
| Memory (bytes/token) | < 50% of flat | Measure actual allocation |

### Task 8: Commit

```bash
git add -A
git commit -m "feat(turboquant): add KV cache compression via near-optimal vector quantization

Implements TurboQuant (arXiv:2504.19874) for KV cache compression:
- Lloyd-Max codebook for Beta distribution (Algorithm 1)
- Random rotation via QR decomposition
- Bit-packed KV cache (2-4 bits per coordinate)
- CPU forward path with dequantized attention
- Quality: cos_sim > 0.99 at 4-bit, > 0.95 at 3-bit
- Compression: 8.5× at 3-bit vs f32 flat cache

Refs: .research/20_TurboQuant_Online_Vector_Quantization.md"
```

---

## File Change Summary

### New files

| File | Lines | Purpose |
|------|-------|---------|
| `microgpt-rs/src/turboquant/mod.rs` | ~15 | Module index |
| `microgpt-rs/src/turboquant/types.rs` | ~60 | Structs: TurboQuantCodebook, TurboQuantLayer, TurboQuantKVCache |
| `microgpt-rs/src/turboquant/codebook.rs` | ~200 | Lloyd-Max scalar quantizer for Beta distribution |
| `microgpt-rs/src/turboquant/rotation.rs` | ~80 | Random rotation + QJL matrix generation |
| `microgpt-rs/src/turboquant/kv_cache.rs` | ~250 | Compressed KV cache: quantize, store, dequantize |
| `microgpt-rs/src/turboquant/forward.rs` | ~150 | forward_turboquant() attention path |
| `riir-ai/crates/riir-gpu/src/kernels/attention_score_tq.wgsl` | ~100 | GPU kernel for quantized attention scoring |

### Modified files

| File | Change |
|------|--------|
| `microgpt-rs/src/lib.rs` | Add `pub mod turboquant;` |
| `microgpt-rs/src/benchmark.rs` | Add `bench_turboquant_vs_flat_cache`, `bench_turboquant_quality` |
| `microgpt-rs/Cargo.toml` | Add `statrs`, `nalgebra` dependencies |
| `riir-ai/crates/riir-gpu/src/lib.rs` | Export TQ attention |
| `riir-ai/crates/riir-gpu/src/kernels/mod.rs` | Register `attention_score_tq` pipeline |
| `riir-ai/crates/riir-gpu/src/forward.rs` | Add `dispatch_attention_tq()` |
| `riir-ai/crates/riir-gpu/Cargo.toml` | Add `statrs` dependency |

---

## Design Decisions

### 1. Separate module, not extension of existing KV cache

`TurboQuantKVCache` is a new struct alongside `MultiLayerKVCache`, `PagedKVCache`, `RavenKVCache`.
It does NOT modify existing caches. This keeps the change additive and risk-free.

### 2. CPU path first, GPU path second

Task 5 (CPU forward with dequantize) ships first. Task 6 (GPU quantized scoring) is a follow-up.
The CPU path is slower (dequantize before scoring) but proves correctness and measures quality.
The GPU path (score without dequantize) is the performance win.

### 3. 3-bit as default bit-width

The paper shows 3.5 bits = quality neutral for Llama-3.1-8B. For our smaller draft model,
3 bits is a safe default. Configurable via `key_bits` and `val_bits` parameters.

### 4. Deterministic rotation matrices

Rotation and QJL matrices are generated from a fixed seed. Same model + same seed = same
quantization behavior. This is critical for reproducibility and testing.

### 5. No entropy encoding

The paper mentions entropy encoding of codebook indices for ~5% bit-width reduction. Skip
for simplicity. The gain is marginal and adds complexity to bit-packing.

### 6. Feature-gated

Everything behind `#[cfg(feature = "turboquant")]` in microgpt-rs. Zero overhead when disabled.

---

## Priority Order

| Priority | Task | Why | Effort |
|----------|------|-----|--------|
| P0 | Task 1: Baseline benchmark | Must measure before changing | Small |
| P0 | Task 2: Codebook | Foundation for all TQ work | Medium |
| P0 | Task 3: Rotation matrices | Required by quantize/dequantize | Small |
| P1 | Task 4: TurboQuantKVCache | Core new data structure | Medium |
| P1 | Task 5: forward_turboquant() | Makes cache usable | Medium |
| P2 | Task 7: Benchmarks | Validate quality claims | Small |
| P2 | Task 8: Commit | Ship it | Small |
| P3 | Task 6: GPU kernel | Performance win, follow-up | Large |

---

## Connection to Existing Plans & Research

| Item | Relationship |
|------|-------------|
| **Research 20 (TurboQuant)** | This plan IS the implementation. Theorem-driven KV cache compression. |
| **Plan 020 (Raven RSM)** | Composable: Raven compresses sequence (N→S slots), TQ compresses precision (f32→bits). Combined ~21,000×. |
| **Plan 042 (TTT Feedback Loop)** | Complementary: Plan 042 improves models (training data), this plan reduces inference cost (KV compression). |
| **riir-gpu `attention_score.wgsl`** | f32 attention kernel. Task 6 adds quantized variant alongside it. |
| **Research 18 (Free Transformer)** | Latent injection at mid-layer. TQ could compress latent vectors too (future). |
| **anyrag embedding search** | TQ outperforms PQ for NN search (paper Section 4.4). Could replace PQ in anyrag vector DB (future). |

---

## Expected Outcomes

1. **8–16× KV cache compression** at 2–4 bits per coordinate
2. **cos_sim > 0.99** between original and reconstructed KV entries at 4-bit
3. **Attention score correlation > 0.98** at 3-bit
4. **Throughput within 10%** of flat f32 cache (CPU path, dequantize overhead)
5. **GPU path**: throughput improvement from reduced memory bandwidth (Task 6)
6. **Composable with Raven**: fixed-slot + compressed precision = extreme compression
7. **New benchmark** in the suite alongside flat/paged/raven comparisons

---

## Risks and Mitigations

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Draft model head_dim=4 too small for Beta distribution | Medium | The Beta distribution converges for d≥3. d=4 works but is on the edge. If quality is poor, apply TQ only to target model (head_dim≥64). |
| Dequantize overhead on CPU slows attention | High | CPU path (Task 5) is correctness-first. GPU path (Task 6) avoids dequantize entirely. |
| Rotation matrix storage (d²×4 bytes) negates savings at small context | Low | For d=128: 64KB per layer. Breaks even at ~100 tokens. Always wins for any realistic context length. |
| Value quantization degrades quality | Medium | Adversarial audit shows value cos_sim=0.94 at 2-bit. Default to 3-bit for values. Allow config override. |
| `statrs` / `nalgebra` dependency bloat | Low | Both are well-maintained, widely used crates. Feature-gate behind `turboquant` feature. |

---

## Research Citation

```bibtex
@article{zandieh2025turboquant,
  title   = {TurboQuant: Online Vector Quantization with Near-Optimal Distortion Rate},
  author  = {Zandieh, Amir and Daliri, Majid and Hadian, Majid and Mirrokni, Vahab},
  journal = {arXiv preprint arXiv:2504.19874},
  year    = {2025}
}
```
