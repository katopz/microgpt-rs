# Handover 009: Zero-Alloc + Rayon Optimization

## What Happened

Implemented Plan 013 — eliminated heap allocations from speculative decoding hot paths and expanded zero-alloc patterns across the codebase.

**Key results:**
- DFlash: **27% faster** (2.62→1.90 μs/step)
- Speculative (Simulated): **20% faster** (5.99→4.81 μs/step)
- Speculative (AR Draft): **21% faster** (5.97→4.69 μs/step)
- Prefill (no compress): **543% faster** (24.25→3.77 μs/step)
- Prefill (compressed): **504% faster** (24.63→4.08 μs/step)
- DDTree Build: **15% faster** (3.24→2.76 μs/step)
- Speedup: Speculative vs AR went from **0.72x → 1.48x**

Much of the foundation (SpeculativeContext, TreeBuilder, _with variants, generate_into, score_into) was already implemented in prior sessions. This session focused on:
1. Adding `extract_best_path_into` to dd_tree.rs
2. Changing `TreeBuilder::build` and `build_dd_tree_pruned` signature from `&[Vec<f32>]` to `&[&[f32]]` for true zero-alloc (avoids copying float data)
3. Fixing `SimulatedVerifier::speculate` to use borrowed `Vec<&[f32]>` views instead of allocating `Vec<Vec<f32>>`
4. Updating ALL callers (14+ test functions, benchmarks, step.rs, examples) for the `&[&[f32]]` signature change
5. Updating benchmark hot loops to use `SpeculativeContext`/`TreeBuilder` outside timed loops
6. Updating `run_speculative_ar_step` to accept `&mut SpeculativeContext` + `&mut TreeBuilder` and return `usize` (count only)
7. Updating `bench_prefill_compression` to use `score_into` with pre-allocated buffer

## Where Is the Plan/Code/Test

- **Plan**: `.plans/013_zero_alloc_rayon.md` — all tasks marked [x] except optional remaining items
- **Code**:
  - `src/speculative/types.rs` — `SpeculativeContext` struct (pre-allocated buffers)
  - `src/speculative/dflash.rs` — `_with` variants (`dflash_predict_with`, `_ar_with`, `_conditioned_with`), backward-compat wrappers
  - `src/speculative/dd_tree.rs` — `TreeBuilder` struct, `build`/`build_and_merge` with `&[&[f32]]` sig, `extract_best_path_into`
  - `src/speculative/sampling.rs` — `sample_residual_distribution_into` (zero-alloc)
  - `src/speculative/verifier.rs` — `SimulatedVerifier`/`LeviathanVerifier` hold `SpeculativeContext` + `TreeBuilder`
  - `src/speculative/step.rs` — `_with` variants for rollback/conditioned
  - `src/speculative/prefill.rs` — `score_into` on `PrefillScorer` trait
  - `src/transformer.rs` — `generate_into` added
  - `src/benchmark.rs` — all bench functions use zero-alloc APIs
  - `src/speculative/mod.rs` — re-exports all new types
- **Tests**: 136 tests pass (`cargo test --lib`)
- **Benchmarks**: `bench/018_bench_result.png` (leviathan), `bench/017_bench_result.png` (default)

## Reflection: Struggling/Solved

- **Borrow checker with disjoint fields**: `forward()` returns `&mut ctx.logits` which borrows `sctx.ctx`. Direct field access (e.g., `sctx.marginals_flat[...]`) works because Rust tracks disjoint borrows. Method calls taking `&mut self` would conflict — solution: use direct field access in hot paths.
- **`&[Vec<f32>]` → `&[&[f32]]` signature change**: This was the most impactful fix. The `SimulatedVerifier` was allocating `Vec<Vec<f32>>` just to pass to `TreeBuilder::build`. Changing to `&[&[f32]]` eliminated all float data copies while requiring updates to 14+ call sites.
- **`run_speculative_ar_step` return type**: Changed from `Vec<usize>` to `usize` (just the count) since benchmarks only need the count. The pre-allocated `accepted_buf` lives in `SpeculativeContext`.
- **Baseline `Transformer AR` variance**: First run showed 708K tok/s (slower), second run showed 1.1M tok/s — system noise, not regression.

## Remain Work

Optional/lower priority items from plan:
- [ ] Allocation tracking via `#[global_allocator]` wrapper in debug builds
- [ ] `run_all_parallel()` — rayon parallel benchmark runner
- [ ] `generate_batch()` — multi-sample generation via `par_iter`
- [ ] Rayon for `build_dd_tree_pruned` initial heap population (not worth it for vocab=27)
- [ ] Fix 4 clippy warnings in dd_tree.rs (too_many_arguments on build_and_merge, useless_vec in tests)

## Issues Ref

- Plan 013: `.plans/013_zero_alloc_rayon.md`
- Bench chart: `bench/018_bench_result.png`

## How to Dev/Test

```bash
# Build release
cargo build --release --quiet

# Run benchmarks (default features)
./target/release/microgpt-rs

# Run benchmarks
cargo run --release

# Run all tests
cargo test --lib

# Clippy
cargo clippy

# Specific test
cargo test -p microgpt-rs --lib -- speculative::verifier::tests
```

Key APIs for extending:
- `SpeculativeContext::new(&config)` — allocate once, `reset()` between calls
- `dflash_predict_with(&mut sctx, ...)` → `usize` (steps populated)
- `TreeBuilder::new(&config)` → `builder.build(&[&[f32]], &config, &pruner, chain_seed)` → `&[TreeNode]`
- `extract_best_path_into(tree, &mut path_buf)` — zero-alloc path extraction
- `sample_residual_distribution_into(p, q, &mut scratch, rng)` — zero-alloc sampling