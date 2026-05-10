# Handover 016: TwELL Sparse MLP Integration (Plan 022)

## What Happened

Implemented the TwELL-inspired sparse MLP matmul for the CPU path, completing the "Trinity" architecture alongside Raven (O(1) memory) and Screening (absolute relevance pruning). The MLP's second weight matrix (`w2 @ hidden`) now skips dead neurons when the `sparse_mlp` feature is enabled, exploiting the natural 95-99% sparsity of ReLU activations.

**All 9 tasks in Plan 022 completed.**

- Without feature: 242 tests pass (zero regressions)
- With `sparse_mlp` feature: 248 tests pass (6 new gated tests)
- Both paths compile clean with zero clippy warnings

## What Changed

### 1. `sparse_matmul` in `types.rs` (Task 1)
New function gated behind `#[cfg(feature = "sparse_mlp")]`. Two-phase execution:
- Phase 1: Pack alive neurons (input[c] > 0.0) into pre-allocated index/value buffers
- Phase 2: Sparse multiply — only iterate alive column indices
Returns alive count for diagnostics and runtime threshold checks.

### 2. Feature flag in `Cargo.toml` (Task 2)
`sparse_mlp = []` added as opt-in feature. NOT in `full` until benchmarked on real model weights.

### 3. ForwardContext buffers in `transformer.rs` (Task 3)
`active_indices: Vec<usize>` and `active_values: Vec<f32>` added behind `#[cfg(feature = "sparse_mlp")]`. Sized to `mlp_hidden`, allocated once in `ForwardContext::new()`, zero alloc in hot loop.

### 4. `sparse_threshold` in `Config` (Task 4)
`pub sparse_threshold: f32` defaulting to `0.8` across all 6 config constructors (`micro`, `draft`, `small_target`, `gqa_draft`, `bpe`, `bpe_draft`). Runtime auto-detection: if alive ratio exceeds `(1 - sparse_threshold)`, falls back to dense matmul.

### 5. Sparse path in 3 forward functions (Task 5)
`forward()`, `forward_paged()`, `forward_raven()` all have identical sparse MLP pattern:
- Call `sparse_matmul` first
- If alive ratio too high (not enough sparsity), fall back to dense `matmul`
- `#[cfg(not(feature = "sparse_mlp"))]` keeps original dense path untouched

### 6. Benchmark in `benchmark.rs` (Task 6)
`bench_sparse_mlp()` tests dense vs sparse at 0%, 50%, 90%, 95%, 99% sparsity across 4 config sizes (micro, bpe, small_target, large). Includes correctness verification against dense output.

### 7. Unit tests in `transformer.rs` (Task 7)
6 tests, all gated behind `#[cfg(feature = "sparse_mlp")]`:
- `test_sparse_matmul_0_percent_sparsity` — matches dense at 0% sparsity
- `test_sparse_matmul_95_percent_sparsity` — matches dense at 95% sparsity
- `test_sparse_matmul_100_percent_sparsity` — all-zero input → all-zero output, 0 alive
- `test_forward_context_sparse_buffers` — buffer sizes match config.mlp_hidden
- `test_forward_with_sparse_mlp` — end-to-end forward produces finite logits
- `test_sparse_matmul_negative_input` — correctly skips negative/zero values

### 8. GPU docs in `gpu/forward.rs` (Task 8)
Comment added before MLP w2 dispatch explaining why GPU stays dense (unstructured sparsity causes warp divergence). References Plan 022 and Research 08.

### 9. README updates (Task 9)
- New "⚡ TwELL Sparse MLP" section after Raven, before Percepta
- Feature flag table updated with `sparse_mlp`
- Project structure updated with `∘` symbol for sparse_mlp-gated items
- References section updated with Sakana paper citation

## Where Is the Plan/Code/Test

- **Plan**: `.plans/022_sparse_mlp_twell.md` — all 9 tasks checked `[x]`
- **Research**: `.research/08_Sakana_TwELL_Sparse_MLP.md`
- **Code changes**:
  - `src/types.rs` — `sparse_matmul` function + `sparse_threshold` in Config
  - `Cargo.toml` — `sparse_mlp` feature flag
  - `src/transformer.rs` — ForwardContext buffers + sparse path in 3 forward functions + 6 tests
  - `src/benchmark.rs` — `bench_sparse_mlp` function
  - `src/gpu/forward.rs` — docs comment for GPU sparse rationale
  - `README.md` — TwELL section, feature flag, project structure, references
- **Tests**: `cargo test --features sparse_mlp -- sparse` — 6 pass

## Reflection: Struggling / Solved

**Struggled with**: The edit tool duplicated the sparse MLP block in the first `forward()` function, creating a double `#[cfg(feature = "sparse_mlp")]` / `#[cfg(not(feature = "sparse_mlp"))` pair. This was caused by an ambiguous line range match.

**Solved by**: `git checkout -- src/transformer.rs` to get clean state, then re-applying changes with precise line targeting. Used `grep` to find exact line numbers for all 3 MLP sites before editing.

**Minor**: Format string in benchmark assertion used `output_dense[i]` inside `format!()` which Rust doesn't support — fixed by binding to local variables first.

## Remain Work

### Immediate (this plan)
- None — all 9 tasks complete.

### Future (out of scope)
- **Add `sparse_mlp` to `full` feature** after benchmarking on real model weights
- **GPU structured sparse** — N:M sparsity (2:4, 4:8) for GPU path (separate plan)
- **Training with L1 regularization** — on Candle/Unsloth side, not microgpt-rs
- **Wire `bench_sparse_mlp` into `run_all`** — currently standalone function, not called from main benchmark entry point
- **Sparse w1 matmul** — no benefit (input isn't sparse), not planned

### Known Limitations (documented in README)
- Sparsity depends on training (L1 regularization needed to reach 99%)
- Small models (micro: mlp_hidden=64) won't benefit — packing overhead > savings
- No GPU benefit — unstructured sparsity causes warp divergence
- Speedup claims must be validated on real model weights

## How to Dev/Test

```bash
# Build without feature (default, zero overhead)
cargo check --quiet
cargo test --quiet --lib

# Build with sparse MLP feature
cargo check --quiet --features sparse_mlp
cargo clippy --quiet --features sparse_mlp

# Run sparse-specific tests
cargo test --quiet --lib --features sparse_mlp -- sparse

# Run full suite with feature
cargo test --quiet --features sparse_mlp

# Run benchmark (currently standalone, not in run_all)
# Requires calling bench_sparse_mlp() from a main/binary
```

## Issues Ref

- Plan 022: `.plans/022_sparse_mlp_twell.md`
- Research 08: `.research/08_Sakana_TwELL_Sparse_MLP.md`
- Paper: arXiv:2603.23198 — "Sparser, Faster, Lighter Transformer Language Models" (Sakana AI & NVIDIA)
- Commit: `feat: add TwELL-inspired sparse MLP matmul (Plan 022)`
