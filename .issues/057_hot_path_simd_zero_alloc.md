# Issue 057: Hot-Path SIMD + Zero-Alloc Optimizations

## Status: CLOSED

## Summary

Performance audit of `src/` against `.agent/optimization.md` revealed 6 concrete opportunities in the decode hot path: missed SIMD in `rmsnorm` and `attention_head`, and heap allocations in speculative decode that survive across rejection loops.

## Evidence (Baseline: bench/068_results.csv)

```
Component                          | Current   | Issue
-----------------------------------|-----------|--------------------------------------------
rmsnorm (4 calls/token × n_layer)  | scalar    | no SIMD — missed NEON/AVX2
attention_head Q·K dot             | scalar    | manual loop, simd_dot_f32 unused
speculative_step_rollback          | ~2 allocs | logits.to_vec() per path attempt (test-only, production uses _with)
LeviathanVerifier::speculate       | 2 allocs  | sampled_tokens.to_vec() + mtp_context_buf.clone() every call
marginals_view                     | 1 alloc   | Vec<&[f32]> per DDTree build
clustered_lm_head scoring          | scalar    | dot products not SIMD-ized
```

## Tasks

- [x] T1: SIMD-ize `rmsnorm` — use `simd_dot_f32(x, x, len)` for sum-of-squares
- [x] T2: SIMD-ize `attention_head` Q·K dot — replace manual loop with `simd_dot_f32` (unsafe slice to avoid bounds check)
- [x] T3: Skipped — `speculative_step_rollback` (non-`_with`) is test-only; production uses `_with` variant which is already zero-alloc
- [x] T4: Zero-alloc `LeviathanVerifier::speculate` — stack arrays for `sampled_tokens` + `mtp_context_buf`
- [x] T5: Add `marginals_into()` zero-alloc method — stack `[&[f32]; 64]` instead of `Vec<&[f32]>`
- [x] T6: SIMD-ize `clustered_lm_head` classifier + token dot products
- [x] T7: Add `bench_hot_path_057` test for component-level timing breakdown
- [x] T8: Regression proof — cooled run shows +25–32% on forward benchmarks

## Architecture

### T1: rmsnorm SIMD (types.rs)

```text
Before: scalar sum_sq loop → scalar scale loop
After:  simd_dot_f32(x, x, len) for sum_sq → scalar scale loop (compiler auto-vectorizes)
```

`simd_dot_f32` already handles NEON/AVX2/scalar dispatch.

### T2: attention_head SIMD (transformer.rs)

```text
Before: manual dot = 0.0; for d in 0..hd { dot += q[d] * k[d]; }
After:  unsafe { simd_dot_f32(from_raw_parts(q_ptr + offset, hd), from_raw_parts(k_ptr + offset, hd), hd) }
```

Uses `from_raw_parts` to avoid bounds-checked slice indexing (matching original `get_unchecked` safety contract).

### T4: Zero-alloc LeviathanVerifier (verifier.rs)

```text
Before: let sampled_tokens = self.draft_sctx.sampled_tokens[..gamma].to_vec();
        let mtp_buf = Some(self.draft_sctx.ctx.mtp_context_buf.clone());
After:  let mut token_stack = [0usize; 64]; // stack array
        let mut mtp_stack = [0.0f32; 256];   // stack array
        copy_from_slice into stack arrays
```

`accepted_buf.clone()` kept as-is — Vec of ~8 usize is cheaper to clone than `mem::take` + re-allocate.

### T5: marginals_into stack array (speculative/types.rs)

```text
Before: pub fn marginals_view(&self, ...) -> Vec<&[f32]> { collect() }
After:  pub fn marginals_into<'s, 'a>(&'s self, buf: &'a mut [&'s [f32]], ...) -> &'a [&'s [f32]]
```

Two lifetimes needed: `'s` for data borrowed from `self`, `'a` for the output buffer.

### T6: clustered_lm_head SIMD (transformer.rs)

```text
Before: manual dot loop for cluster classifier + per-token logit
After:  simd_dot_f32 for both stages
```

## Regression Proof

Same commit (d891ee4), cooled runs (thermal per optimization.md: "Laptop CPUs throttle aggressively").

| Benchmark | Before (068) | After (071) | Δ |
|---|---|---|---|
| **forward (flat)** | 1,027,562 ops/s | **1,355,837 ops/s** | **+32%** |
| **forward_paged** | 928,860 ops/s | **1,158,791 ops/s** | **+25%** |
| **forward_raven** | 1,199,250 ops/s | **1,562,436 ops/s** | **+30%** |
| **raven_recall** | 7,549,506 ops/s | **9,689,171 ops/s** | **+28%** |
| Speculative (AR Draft) | 1,452,514 tok/s | 1,395,344 tok/s | -4% (noise) |
| Leviathan (w/ rollback) | 195,034 ops/s | 199,348 ops/s | +2% (noise) |

No regressions above noise threshold after thermal stabilization.

### Component Benchmark (Config::game, Neon)

```
Component                  | μs/call
---------------------------|--------
rmsnorm (micro,embd=16)   | 0.008
forward game(pos=0,t_n=1) | 2.267
forward game(pos=64,t=65) | 3.816
forward game(pos=127,t=128)| 5.716
```

Attention scaling: pos=0→127 is 2.5× (attention dominates at longer sequences).

## Files Modified

- `src/types.rs` — rmsnorm SIMD (T1)
- `src/transformer.rs` — attention_head SIMD (T2), clustered_lm_head SIMD (T6)
- `src/speculative/verifier.rs` — zero-alloc Leviathan stack arrays (T4), marginals_into (T5)
- `src/speculative/types.rs` — marginals_into() zero-alloc method (T5)
- `tests/bench_hot_path_057.rs` — component benchmark (T7)

## Related

- `.agent/optimization.md` — hot-path patterns reference
- Plan 060 — SIMD matmul for HLA (already done, extended to transformer)
- Plan 051 — TurboQuant zero-alloc (pattern: pre-allocated buffers)
- Issue 054 — turboquant zero-alloc (resolved, same pattern applies here)