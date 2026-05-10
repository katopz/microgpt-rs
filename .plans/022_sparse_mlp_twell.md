# Plan 022: Sparse MLP — TwELL-Inspired Unstructured Sparsity

**Branch:** `develop/feature/022_sparse_mlp_twell`
**Depends on:** None (feature-gated, backward compatible)
**Research:** `.research/08_Sakana_TwELL_Sparse_MLP.md`

---

## Overview

Add a CPU sparse matmul path for the MLP's second weight matrix (`w2 @ hidden`), exploiting the natural sparsity of ReLU activations. ReLU zeros out ~50% of neurons by definition; with L1 regularization during training, sparsity can reach 90-99%. This skips dead neurons to reduce FLOPs. Feature-gated behind `sparse_mlp` with runtime auto-detection for safe fallback.

**Caveat**: This is CPU index-packing sparse matmul, not the paper's TwELL (Tile-wise ELLPACK) which is a GPU-specific tiled format. Our current models use random weights, so actual speedup is unproven. Targets real LLMs with `mlp_hidden >= 1024`; small configs (micro: 64, bpe: 128) likely won't benefit due to packing overhead exceeding savings.

---

## Tasks

- [x] **Task 1: Add `sparse_matmul` to `types.rs`**
  - New function: `sparse_matmul(output, weight, input, rows, cols, active_indices, active_values) -> usize`
  - Phase 1: Pack alive neurons (input[c] > 0.0) into pre-allocated buffers
  - Phase 2: Sparse multiply — only iterate alive indices
  - Return alive count for diagnostics
  - Keep existing `matmul` and `matmul_relu` untouched

- [x] **Task 2: Add `sparse_mlp` feature to `Cargo.toml`**
  - New feature: `sparse_mlp = []`
  - NOT in default features (opt-in)
  - NOT in `full` feature initially (until benchmarked)
  - Add to full after benchmarks prove it

- [x] **Task 3: Add `active_indices` and `active_values` to `ForwardContext`**
  - Two new buffers: `Vec<usize>` and `Vec<f32>`, both sized to `mlp_hidden`
  - Allocated once in `ForwardContext::new()`, reused every forward pass
  - Only compiled when `sparse_mlp` feature is enabled

- [x] **Task 4: Add `sparse_threshold` to `Config`**
  - `pub sparse_threshold: f32` defaulting to `0.8` (auto-sparse if >80% sparse)
  - `0.0` = always use sparse (even at low sparsity)
  - `1.0` = never use sparse (always dense)
  - When sparsity < threshold, fall back to dense `matmul`

- [x] **Task 5: Implement sparse MLP in `forward()`** (`src/transformer.rs`)
  - After `matmul_relu(hidden, w1, x)`:
    - If `sparse_mlp` feature enabled: call `sparse_matmul(x, w2, hidden, n, mlp_hidden, active_indices, active_values)`
    - Check alive_ratio = alive_count / mlp_hidden; if > (1 - sparse_threshold), fall back to dense
    - If feature disabled: existing `matmul(x, w2, hidden, n, mlp_hidden)` unchanged
  - Same pattern for `forward_paged()` and `forward_raven()`

- [x] **Task 6: Add benchmark** (`src/benchmark.rs`)
  - Benchmark: `matmul` vs `sparse_matmul` at 0%, 50%, 90%, 95%, 99% sparsity
  - Config sizes: micro (mlp_hidden=64), bpe (128), small_target (256), large (16384)
  - Include break-even analysis: at what sparsity does sparse win?
  - Use `std::time::Instant` like existing benchmarks

- [x] **Task 7: Unit tests**
  - Test: `sparse_matmul` produces identical output to `matmul` at 0% sparsity
  - Test: `sparse_matmul` produces identical output at 95% sparsity
  - Test: `sparse_matmul` produces identical output at 100% sparsity (all zeros → output all zeros)
  - Test: feature-gated — `sparse_mlp` off compiles and runs correctly
  - Test: feature-gated — `sparse_mlp` on compiles and runs correctly
  - Test: `ForwardContext` buffers are correct size
  - Test: fallback to dense when sparsity below threshold

- [x] **Task 8: Update GPU path docs** (`src/gpu/forward.rs`)
  - Add comment in `dispatch_layer()` MLP section explaining why GPU stays dense
  - Reference this plan and research doc

- [x] **Task 9: Update README**
  - Add "TwELL Sparse MLP (Plan 022)" section to Architecture
  - Update Feature Flags section with `sparse_mlp`
  - Add benchmark results after Task 6

---

## File Change Summary

| File | Change |
|------|--------|
| `microgpt-rs/src/types.rs` | Add `sparse_matmul()` function |
| `microgpt-rs/Cargo.toml` | Add `sparse_mlp` feature |
| `microgpt-rs/src/transformer.rs` | Add buffers to `ForwardContext`, sparse path in forward functions |
| `microgpt-rs/src/benchmark.rs` | Add sparse vs dense benchmark |
| `microgpt-rs/src/gpu/forward.rs` | Add docs comment for GPU sparse rationale |
| `microgpt-rs/README.md` | Add TwELL Sparse MLP section |

---

## Design Decisions

### 1. Feature-Gated, Not Default
`sparse_mlp` is opt-in because:
- Small models (micro: mlp_hidden=64) won't benefit — packing overhead > savings
- Sparsity depends on training (L1 regularization needed)
- Users should benchmark their specific model before enabling

### 2. Pre-Allocated Buffers in ForwardContext
Not `Vec::push` in hot loop. The `active_indices` and `active_values` buffers are allocated once and reused. The packing phase just writes into them without any allocation.

### 3. Runtime Auto-Detection
Even with feature enabled, check actual sparsity and fall back to dense if too many neurons are alive. This prevents regression when running models without L1 training.

### 4. CPU-Only Optimization
GPU stays dense. Unstructured sparsity causes warp divergence. If GPU sparse is needed later, use structured N:M sparsity (separate plan).

### 5. Config-Driven Threshold
`sparse_threshold: f32` in Config lets users tune the auto-detection. Default 0.8 means "use sparse when >80% of neurons are dead".

---

## Out of Scope

- GPU sparse matmul (requires structured sparsity, separate plan)
- Structured N:M sparsity (2:4, 4:8 patterns — hardware-specific)
- Training with L1 regularization (Candle/Unsloth side, not microgpt-rs)
- Sparse w1 matmul (input isn't sparse, no benefit)
- Quantization-aware sparse (future research)