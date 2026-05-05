# Plan 003: Performance Optimization — Inline, Fused Ops, Unsafe Indexing

## Objective
Benchmark baseline → optimize hotpaths → verify perf gain with all tests passing.

## Baseline (pre-optimization, release build)
| Method               | Throughput      | μs/step |
|----------------------|-----------------|---------|
| Transformer AR       | 831,094 tok/s   | 1.20    |
| DFlash               | 2,941,131 tok/s | 2.72    |
| DDTree Build         | 317,079 trees/s | 3.15    |
| Speculative Decoding | 669,303 tok/s   | 5.98    |

## Optimized (post-optimization, release build)
| Method               | Throughput        | μs/step | Change   |
|----------------------|-------------------|---------|----------|
| Transformer AR       | 1,120,827 tok/s   | 0.89    | **+34.9%** |
| DFlash               | 3,217,815 tok/s   | 2.49    | **+9.4%**  |
| DDTree Build         | 298,133 trees/s   | 3.35    | -6.0%    |
| Speculative Decoding | 687,450 tok/s     | 5.82    | **+2.7%**  |

## Tasks

- [x] 1. **`#[inline(always)]` on hotpath kernels** — `matmul`, `softmax`, `rmsnorm`, `forward`, `sample_token`, `Rng::next`, `Rng::uniform`
- [x] 2. **Fused `matmul_relu`** — single pass for MLP hidden layer, avoids extra scan of `hidden` buffer
- [x] 3. **Fused `attention_head` kernel** — score → softmax → weighted value in one function, avoids separate `softmax()` call and write-back of normalized scores
- [x] 4. **Unsafe indexing** — `get_unchecked` / `get_unchecked_mut` in inner loops (matmul, attention, embedding, residual) to eliminate bounds checks
- [x] 5. **Optimized softmax** — fuse exp+sum in one pass, use `inv_sum = 1.0/sum` with multiply instead of divide
- [x] 6. **Optimized rmsnorm** — manual two-pass with `inv_rms` multiply instead of divide
- [x] 7. **`copy_nonoverlapping` for KV cache store** — faster than `copy_from_slice` for known-size copies
- [x] 8. **Run final benchmark** — all 157 tests pass (77 unit + 80 integration), zero clippy warnings, chart saved as `bench/009_bench_result.png`
- [x] 9. **Commit** with message `perf: fused matmul_relu, fused attention_head, inline+unsafe for transformer hotpath`

## Architecture Notes
- `types.rs` contains `matmul`, `softmax`, `rmsnorm` — these are the hot kernels
- `transformer.rs` contains `forward()` — the main hotpath calling all kernels
- `rayon` already in `Cargo.toml` but only used in `dflash_predict_parallel`
- `std::simd` / `portable_simd` is nightly-only; used stable-compatible `#[inline(always)]` + `get_unchecked` instead
- Micro config: n_embd=16, mlp_hidden=64 — matmul is (64×16) and (16×64), attention is (16×16)
- At these tiny sizes (head_dim=4, n_embd=16), rayon overhead dominates — parallelism not beneficial
- Edition 2024 requires explicit `unsafe {}` blocks inside `unsafe fn` — added throughout

## Files Modified
| File | Changes |
|------|---------|
| `src/types.rs` | `#[inline(always)]` on matmul/softmax/rmsnorm/sample_token; `matmul_relu` fused kernel; optimized softmax with `inv_sum`; optimized rmsnorm with `inv_rms`; unsafe indexing in matmul loops |
| `src/transformer.rs` | `#[inline(always)]` on forward; fused `attention_head` kernel; unsafe indexing in embedding/residual/KV-store; `matmul_relu` for MLP; `copy_nonoverlapping` for cache store |

## What We Did NOT Do (and why)
- **Rayon parallel matmul**: At n_embd=16, mlp_hidden=64, the overhead of rayon thread pool dominates. Benchmarked and confirmed no gain at these sizes.
- **`std::simd` / `portable_simd`**: Nightly-only feature. The compiler's auto-vectorization on aarch64 NEON already handles our small f32 loops well.
- **Cache tiling for attention**: block_size=16 already fits in L1 cache. Tiling adds complexity with no benefit.

## Outcome
- **Transformer AR: +34.9% throughput** (831K → 1121K tok/s)
- **DFlash: +9.4% throughput** (2.94M → 3.22M tok/s)
- All 157 tests passing (77 unit + 80 integration)
- Zero clippy warnings
- Clean `--release` build on stable Rust 1.93.0 (aarch64-apple-darwin)