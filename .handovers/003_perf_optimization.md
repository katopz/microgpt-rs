# Handover 003: Performance Optimization

## What Happened
Optimized the transformer forward pass hotpath for performance. Baseline was established (bench/008), then applied a series of optimizations targeting the most called code paths: `matmul`, `softmax`, `rmsnorm`, and the multi-head attention loop.

## Where Is the Plan/Code/Test
- **Plan**: `.plans/003_perf_optimization.md`
- **Code changed**: `src/types.rs` (inline hints, fused matmul_relu, optimized softmax/rmsnorm), `src/transformer.rs` (fused attention_head kernel, unsafe indexing, copy_nonoverlapping)
- **Tests**: All 157 existing tests pass (77 unit + 80 integration), no new tests added (optimizations are behavior-preserving)
- **Benchmark**: `bench/008_bench_result.png` (baseline), `bench/009_bench_result.png` (optimized)

## Baseline vs Optimized

| Method               | Baseline         | Optimized          | Change     |
|----------------------|------------------|--------------------|------------|
| Transformer AR       | 831,094 tok/s (1.20μs) | 1,120,827 tok/s (0.89μs) | **+34.9%** |
| DFlash               | 2,941,131 tok/s (2.72μs) | 3,217,815 tok/s (2.49μs) | **+9.4%**  |
| DDTree Build         | 317,079 trees/s (3.15μs) | 298,133 trees/s (3.35μs) | -6.0%      |
| Speculative Decoding | 669,303 tok/s (5.98μs) | 687,450 tok/s (5.82μs)   | **+2.7%**  |

Note: Transformer AR varies ±10% between runs (OS noise). DFlash is more stable. DDTree build is pure heap ops, not affected by transformer optimizations.

## Reflection — Struggling / Solved

### Solved
1. **Edition 2024 `unsafe` blocks**: `unsafe fn` requires explicit `unsafe {}` blocks for each `get_unchecked` call inside the function body. Added throughout `attention_head`.
2. **Clippy `too_many_arguments`**: `attention_head` has 10 params; suppressed with `#[allow(clippy::too_many_arguments)]`.
3. **Rayon not beneficial at tiny sizes**: At n_embd=16, mlp_hidden=64, rayon thread pool overhead dominates. Confirmed by testing.
4. **`std::simd` / `portable_simd`**: Nightly-only. The compiler's auto-vectorization on aarch64 NEON already handles small f32 loops well.

### Key Insights
- The biggest single win was **fused attention_head** (+12% alone) — combining score/softmax/weighted_value avoids writing back normalized scores and reduces function call overhead.
- **`matmul_relu`** fusion saves one full buffer scan of the 64-element hidden layer — small but measurable.
- **`#[inline(always)]`** across all hot kernels eliminates function call overhead in the tight inner loops.
- **`get_unchecked`** eliminates bounds checks in the innermost matmul loops where indices are provably in-bounds.

## Optimizations Applied

### `src/types.rs`
- `#[inline(always)]` on: `matmul`, `softmax`, `rmsnorm`, `sample_token`, `Rng::next`, `Rng::uniform`
- New `matmul_relu()`: fused matmul + ReLU in single pass (used for MLP hidden)
- Optimized `softmax`: manual max-finding loop, fuse exp+sum, `inv_sum` multiply instead of divide
- Optimized `rmsnorm`: manual sum_sq loop, `inv_rms` multiply instead of divide
- `matmul`: switched from `chunks_exact` iterator to index-based loop with `get_unchecked`

### `src/transformer.rs`
- `#[inline(always)]` on `forward`
- New `attention_head()`: fused Q·K scoring + inline softmax + weighted V accumulation in one function
- Embedding: pre-computed offsets, `get_unchecked` indexing
- KV cache store: `copy_nonoverlapping` instead of `copy_from_slice`
- Residual adds: `get_unchecked` indexing instead of zip iterator
- MLP: uses `matmul_relu` (no separate ReLU loop)

## What We Did NOT Do (and why)
- **Rayon parallel matmul**: Thread pool overhead dominates at 16×64 sizes
- **`std::simd`**: Nightly-only, auto-vectorization sufficient
- **Cache tiling for attention**: block_size=16 already fits L1, no benefit
- **Micro-benchmarks with criterion**: Not worth the dependency for this size project

## Remain Work
1. **Free Embedding Bridge** — Project pre-LM-head hidden states to 2D to query `KVCache2D` with actual transformer data
2. **Scale to actual LLM tokens** — Map Sudoku digits (1–9) to real vocabulary indices via tokenizer
3. **Streaming with print flush** — Switch from `format_events()` batch to callback-based real-time output
4. **Integration test coverage** — Path-aware tests are in unit tests; could add integration tests to `tests/integration.rs`

## Issues Ref
- No new issues created

## How to Dev/Test
```bash
# Run all tests
cargo test --quiet --all

# Run benchmark (release mode)
cargo run --quiet --release

# Clippy
cargo clippy --quiet

# Specific test
cargo test --quiet --lib -- test_forward_output_size
```

## Plan Status
| Plan | Status | Tasks |
|------|--------|-------|
| Plan 001: Sudoku 9×9 Example | ✅ Complete | 7/7 tasks |
| Plan 002: Dynamic Depth-Aware Pruning | ✅ Complete | 7/7 tasks |
| Plan 003: Perf Optimization | ✅ Complete | 9/9 tasks |