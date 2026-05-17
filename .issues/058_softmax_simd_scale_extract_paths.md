# Issue 058: Hot-Path SIMD Scale + Zero-Alloc Audit (Issue 057 Follow-up)

## Status: CLOSED

## Summary

Deep audit of `src/` against `.agent/optimization.md` after Issue 057. Found 10 concrete optimization opportunities across 4 categories: SIMD scale, SIMD matmul, allocation elimination, and algorithmic redundancy.

## Evidence

### Category A: Scalar `*= scale` Loops (simd_scale_inplace)

**A1. softmax / softmax_scaled pass 3** — every forward call

```text
// src/types.rs L570-573 — softmax pass 3 (also softmax_scaled L613-616)
let inv_sum = 1.0 / sum;
for val in x.iter_mut() {
    *val *= inv_sum;  // ← scalar loop, vocab_size iterations
}
```

Called `n_head × n_layer` times per token (attention) + 1 final logit softmax = ~5-10 calls/token.
For `vocab_size=256`: 256 scalar multiplies → could be 64 NEON / 32 AVX2 ops.

**A2. rmsnorm pass 2** — 2× per layer

```text
// src/types.rs L631-634
let inv_rms = 1.0 / (sum_sq / x.len() as f32 + 1e-5).sqrt();
for val in x.iter_mut() {
    *val *= inv_rms;  // ← scalar loop, n_embd iterations
}
```

Pass 1 (sum-of-squares) already uses `simd_dot_f32` (Issue 057). Pass 2 still scalar.
Called 3 × n_layer times/token (pre-attention, post-attention, MLP).
For `n_embd=64, n_layer=2` → 384 scalar multiplies/token.

**A3. HLA kernel decay loops** — `hla/kernel.rs`

```text
// src/hla/kernel.rs L99-113 — hla_state_update decay (γ < 1.0)
for x in sk.iter_mut()          { *x *= gamma; }  // hd² elements!
for x in q_head.cqv.iter_mut()  { *x *= gamma; }  // hd² elements!
for x in q_head.mq.iter_mut()   { *x *= gamma; }  // hd elements
for x in q_head.g.iter_mut()    { *x *= gamma; }  // hd² elements!
for x in q_head.h.iter_mut()    { *x *= gamma; }  // hd elements
```

Same pattern in `hla_per_head_update` (L163-176), `ahla_step` (L362-373), `ahla_per_head_step` (L565-572), `hla_layer_update` (L465-468), `ahla_layer_step` (L648-655).
For `hd=16`: 5 × (256+256+16+256+16) = 1600 scalar multiplies per update.
HLA replaces standard attention entirely — this IS the hot path.

**A4. TurboQuant store_key/store_value normalize** — `turboquant/kv_cache.rs`

```text
// src/turboquant/kv_cache.rs L159-163 (store_key)
let inv_norm = 1.0 / norm;
for (i, &v) in key.iter().enumerate() {
    unsafe { *self.scratch_normalized.get_unchecked_mut(i) = v * inv_norm; }
}
```

Same in `store_value` (L203-207) and `dequantize_key_into`/`dequantize_value_into` final scale loop (L295-299, L340-344).
Called 2 × n_layer per token (key + value). For `kv_dim=64`: 128 scalar multiplies/token.

### Category B: Scalar Matmul (use existing simd_matmul_rows / simd_matvec)

**B1. TurboQuant mat_vec_into / mat_vec_t_into** — `turboquant/kv_cache.rs`

```text
// src/turboquant/kv_cache.rs L402-415 — scalar matmul!
fn mat_vec_into(m: &[f32], v: &[f32], out: &mut [f32]) {
    for (i, out_val) in out.iter_mut().enumerate() {
        let mut sum = 0.0f32;
        let row_off = i * dim;
        for j in 0..dim {
            sum += *m.get_unchecked(row_off + j) * *v.get_unchecked(j);  // scalar dot!
        }
        *out_val = sum;
    }
}
```

`simd_matmul_rows` already exists in `simd.rs` and does exactly this with NEON/AVX2.
`mat_vec_t_into` is transpose matmul — needs `simd_matvec_t` or column-wise access.
Called in `store_key`, `store_value` (rotation), `dequantize_key_into`, `dequantize_value_into` (inverse rotation).
4 calls/token × dim² scalar multiplies each.

**B2. HLA readout matvec** — `hla/kernel.rs`

```text
// src/hla/kernel.rs L238-247 — scalar qᵀ · SK matvec
tmp_u[..hd].fill(0.0);
for i in 0..hd {
    let qi = unsafe { *q.get_unchecked(i) };
    let sk_row = &sk[i * hd..i * hd + hd];
    for j in 0..hd {
        *tmp_u.get_unchecked_mut(j) += qi * *sk_row.get_unchecked(j);  // scalar!
    }
}
```

`simd_matvec` already exists in `simd.rs`. Same pattern in `hla_denom` (L279-288), `ahla_step` (L380-389), `ahla_per_head_step` (L574-583).
Called per head per token during HLA inference.

**B3. cosine_similarity** — `turboquant/forward.rs`

```text
// src/turboquant/forward.rs L128-134 — scalar dot + norms
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
```

`simd_dot_f32` already exists. Used in quality validation, not decode hot path — lower priority.

### Category C: Redundant Computation

**C1. forward_prefill double embedding** — `transformer.rs`

```text
// Phase A: compute K/V for all positions (L1074-1093)
for (p, &token) in tokens.iter().enumerate().take(prompt_len) {
    // Load hidden state → rmsnorm → K/V projections → store cache
}

// Phase B: bidirectional attention for all positions (L1096-1113)
for (p, &token) in tokens.iter().enumerate().take(prompt_len) {
    // Load hidden state AGAIN (same source!) → rmsnorm → Q projection → attention
}
```

For single-layer (n_layer=1), embedding is recomputed from scratch in both phases.
For multi-layer, hidden state is stored in `prefill.hidden` but still requires embedding + first-layer forward in Phase A.
The rmsnorm+QKV projection could be done in a single pass, avoiding the double load.

**Impact**: For prompt_len=P, saves P × n_embd embedding lookups + 2 × P × n_embd rmsnorm calls.

### Category D: Allocation in Speculative Hot Path

**D1. speculative_step_rollback / _paged** — `speculative/step.rs`

```text
// src/speculative/step.rs L109, L130, L162, L214, L457, L478, L525
let mut p_dist = logits.to_vec();  // ← allocates vocab_size f32s!
```

The `_with` variants (used by LeviathanVerifier production path) already use `probs_buf.copy_from_slice(logits)`.
But `speculative_step_rollback` and `speculative_step_rollback_paged` still allocate via `.to_vec()`.
These are called from benchmarks (`bench_leviathan`, `bench_snapshot_rollback`).

**D2. extract_ddtree_paths** — `speculative/step.rs`

```text
// src/speculative/step.rs L616-657 — O(roots × depths × tree_size)
fn extract_ddtree_paths(tree: &[TreeNode]) -> Vec<Vec<usize>> {
    let mut roots: Vec<_> = tree.iter().filter(|n| n.depth == 0).collect();  // O(N)
    roots.sort_by(|a, b| b.score.partial_cmp(&a.score)...);
    for root in &roots {
        for depth in 1..=max_depth {
            let child = tree.iter().filter(|n| n.depth == depth && ...).max_by_key(...);  // O(N) per depth!
        }
    }
}
```

Called in every speculative step. For `tree_budget=64, draft_lookahead=5`: 3 roots × 5 depths × 64 nodes = 960 iterations with branch mispredictions.

Same pattern in `extract_best_path` (`dd_tree.rs`) — O(D × N) per depth.

## Tasks

### T1: Add `simd_scale_inplace` (Category A foundation)
- Add `simd_scale_inplace(x: &mut [f32], scale: f32)` to `src/simd.rs`
- NEON (`vmulq_f32`) + AVX2 (`_mm256_mul_ps`) + scalar fallback
- Place before `horizontal_sum_256` (~L549)
- Add unit tests: aligned len, non-aligned, empty, single-element, large

### T2: Wire `simd_scale_inplace` into softmax/rmsnorm (A1, A2)
- `softmax` pass 3: `src/types.rs` ~L570 — replace scalar loop
- `softmax_scaled` pass 3: `src/types.rs` ~L613 — replace scalar loop
- `rmsnorm` pass 2: `src/types.rs` ~L631 — replace scalar loop

### T3: Wire `simd_scale_inplace` into HLA decay (A3)
- `hla_state_update`: `src/hla/kernel.rs` ~L99-113
- `hla_per_head_update`: `src/hla/kernel.rs` ~L163-176
- `ahla_step`: `src/hla/kernel.rs` ~L362-373
- `ahla_per_head_step`: `src/hla/kernel.rs` ~L565-572
- `hla_layer_update`: `src/hla/kernel.rs` ~L465-468
- `ahla_layer_step`: `src/hla/kernel.rs` ~L648-655

### T4: Wire `simd_scale_inplace` into TurboQuant (A4)
- `store_key` normalize: `src/turboquant/kv_cache.rs` ~L159-163
- `store_value` normalize: `src/turboquant/kv_cache.rs` ~L203-207
- `dequantize_key_into` scale: `src/turboquant/kv_cache.rs` ~L295-299
- `dequantize_value_into` scale: `src/turboquant/kv_cache.rs` ~L340-344

### T5: Replace TurboQuant scalar matmul with `simd_matmul_rows` (B1)
- `mat_vec_into`: `src/turboquant/kv_cache.rs` ~L402-415 → delegate to `simd_matmul_rows`
- `mat_vec_t_into`: `src/turboquant/kv_cache.rs` ~L426-438 → new `simd_matvec_transpose` or column-wise simd_dot
- Add `simd_matvec_transpose` to `simd.rs` if needed

### T6: Replace HLA readout scalar matvec with `simd_matvec` (B2)
- `hla_readout` qᵀ·SK: `src/hla/kernel.rs` ~L238-247
- `hla_denom` qᵀ·SK: `src/hla/kernel.rs` ~L279-288
- `ahla_step` qᵀ·PKV: `src/hla/kernel.rs` ~L380-389
- `ahla_per_head_step` qᵀ·PKV: `src/hla/kernel.rs` ~L574-583

### T7: Use `simd_dot_f32` in cosine_similarity (B3)
- `src/turboquant/forward.rs` ~L128-134 — replace scalar `a.iter().zip(b).map(|(x,y)| x*y).sum()`
- Also replace scalar norm computation with `simd_dot_f32(a, a, len).sqrt()`

### T8: Fuse forward_prefill Phase A+B embedding (C1) ✅
- For single-layer: compute embedding once, reuse across phases
- For multi-layer: store pre-rmsnorm hidden state to avoid recomputation
- Measure impact on prefill latency with `bench_prefill_compression`
- **Done**: Fused K/V/Q projections into single Phase A pass, storing Q + xr in `PrefillContext::queries`/`PrefillContext::residuals`. Phase B loads pre-computed values, eliminating redundant hidden load + double rmsnorm + Q matmul per position. Bit-identical output verified (547 tests pass).

### T9: Optimize `extract_ddtree_paths` + `extract_best_path` (D2)
- Pre-index tree nodes by depth: `[Vec<&TreeNode>; MAX_DEPTH]` built in O(N)
- Replace O(D × N) `.iter().filter()` scans with O(1) depth lookups
- Same for `extract_best_path` in `dd_tree.rs`

### T10: Document `speculative_step_rollback` / `_paged` as deprecated (D1)
- Add `#[deprecated(note = "Use speculative_step_rollback_with for zero-alloc production path")]`
- These are benchmark-only now; production uses `_with` variants via `LeviathanVerifier`

## Benchmark Plan

### B1: `bench_simd_scale` (T1-T4 validation)

```rust
// In benchmark.rs or tests/bench_simd_scale.rs
#[test]
fn bench_simd_scale() {
    let warmup = 100;
    let iters = 10_000;
    let sizes = [16, 64, 128, 256, 512];

    for &size in &sizes {
        let mut x = vec![0.5f32; size];

        for _ in 0..warmup {
            crate::simd::simd_scale_inplace(&mut x, 0.5);
            std::hint::black_box(&x);
            for val in &mut x { *val = 0.5; }
        }

        let start = std::time::Instant::now();
        for _ in 0..iters {
            crate::simd::simd_scale_inplace(&mut x, 0.5);
            std::hint::black_box(&x);
            for val in &mut x { *val = 0.5; }
        }
        let elapsed = start.elapsed();
        let ns_per = elapsed.as_secs_f64() * 1e9 / iters as f64;
        println!("  simd_scale_inplace(len={size}): {ns_per:.1} ns/call");
    }
}
```

### B2: `bench_softmax_scale` (T2 validation)

```rust
#[test]
fn bench_softmax_scale() {
    let warmup = 100;
    let iters = 10_000;
    let sizes = [64, 128, 256, 512];

    for &size in &sizes {
        let mut x = vec![0.5f32; size];

        for _ in 0..warmup {
            let mut tmp = x.clone();
            crate::types::softmax(&mut tmp);
            std::hint::black_box(&mut tmp);
        }

        let start = std::time::Instant::now();
        for _ in 0..iters {
            let mut tmp = x.clone();
            crate::types::softmax(&mut tmp);
            std::hint::black_box(&mut tmp);
        }
        let elapsed = start.elapsed();
        let us_per = elapsed.as_secs_f64() * 1e6 / iters as f64;
        println!("  softmax(vocab={size}): {us_per:.2} μs/call");
    }
}
```

### B3: `bench_hla_decay` (T3 validation)

```rust
#[test]
fn bench_hla_decay() {
    use crate::hla::types::*;
    let hd = 16;
    let gamma = 0.99;
    let iters = 10_000;
    let mut sk = vec![0.5f32; hd * hd];
    let mut cqv = vec![0.5f32; hd * hd];
    let mut mq = vec![0.5f32; hd];
    let mut g = vec![0.5f32; hd * hd];
    let mut h = vec![0.5f32; hd];

    let start = std::time::Instant::now();
    for _ in 0..iters {
        crate::simd::simd_scale_inplace(&mut sk, gamma);
        crate::simd::simd_scale_inplace(&mut cqv, gamma);
        crate::simd::simd_scale_inplace(&mut mq, gamma);
        crate::simd::simd_scale_inplace(&mut g, gamma);
        crate::simd::simd_scale_inplace(&mut h, gamma);
        std::hint::black_box((&sk, &cqv, &mq, &g, &h));
    }
    let elapsed = start.elapsed();
    let ns_per = elapsed.as_secs_f64() * 1e9 / iters as f64;
    println!("  HLA decay (hd={hd}, gamma={gamma}): {ns_per:.1} ns/update");
}
```

### B4: `bench_extract_paths` (T9 validation)

```rust
#[test]
fn bench_extract_paths() {
    let config = crate::types::Config::draft();
    let mut rng = crate::types::Rng::new(42);
    let weights = crate::transformer::TransformerWeights::new(&config, &mut rng);
    let marginals = crate::speculative::dflash::dflash_predict(&weights, &config, 0, 0);
    let mv: Vec<&[f32]> = marginals.iter().map(|s| s.as_slice()).collect();
    let tree = crate::speculative::dd_tree::build_dd_tree(&mv, &config);

    let warmup = 100;
    let iters = 10_000;

    for _ in 0..warmup {
        std::hint::black_box(crate::speculative::step::extract_ddtree_paths(&tree));
    }

    let start = std::time::Instant::now();
    for _ in 0..iters {
        std::hint::black_box(crate::speculative::step::extract_ddtree_paths(&tree));
    }
    let elapsed = start.elapsed();
    let us_per = elapsed.as_secs_f64() * 1e6 / iters as f64;
    println!("  extract_ddtree_paths(budget={}, nodes={}): {:.2} μs/call",
        config.tree_budget, tree.len(), us_per);
}
```

### B5: `bench_turboquant_matvec` (T5 validation)

```rust
#[test]
fn bench_turboquant_matvec() {
    let dim = 64;
    let iters = 10_000;
    let m: Vec<f32> = (0..dim*dim).map(|i| (i as f32 * 0.01).sin()).collect();
    let v: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.02).cos()).collect();
    let mut out = vec![0.0f32; dim];

    let start = std::time::Instant::now();
    for _ in 0..iters {
        crate::simd::simd_matmul_rows(&mut out, &m, &v, dim, dim);
        std::hint::black_box(&out);
    }
    let elapsed = start.elapsed();
    let us_per = elapsed.as_secs_f64() * 1e6 / iters as f64;
    println!("  simd_matmul_rows(dim={dim}): {us_per:.2} μs/call");
}
```

## Test Plan

### T1 tests: `simd_scale_inplace`
- `test_scale_aligned_len_8` — verify output matches scalar
- `test_scale_non_aligned_len_13` — verify remainder handling
- `test_scale_empty` — no panic
- `test_scale_single_element` — edge case
- `test_scale_zero` — all become 0.0
- `test_scale_matches_scalar` — fuzz-ish: random len, random values

### T5 tests: mat_vec_into SIMD
- `test_mat_vec_into_matches_scalar` — compare old scalar vs new SIMD output
- `test_mat_vec_t_into_matches_scalar` — same for transpose
- Existing TurboQuant roundtrip tests should still pass

### T9 tests: depth-indexed extraction
- `test_extract_ddtree_paths_matches_old` — same output as old O(D×N) version
- `test_extract_best_path_matches_old` — same
- `test_extract_paths_empty_tree` — edge case
- Existing `test_extract_ddtree_paths` should still pass

## Expected Improvement

| Component | Before | After | Method |
|-----------|--------|-------|--------|
| softmax normalize (vocab=256) | scalar 256× mul | NEON 64 / AVX2 32 ops | `simd_scale_inplace` |
| rmsnorm scale (embd=64) | scalar 64× mul | NEON 16 / AVX2 8 ops | `simd_scale_inplace` |
| HLA decay (hd=16) | scalar 1600× mul/update | NEON 400 / AVX2 200 ops | `simd_scale_inplace` |
| TQ store_key normalize | scalar kv_dim× mul | NEON/AVX2 | `simd_scale_inplace` |
| TQ mat_vec_into | scalar dim² dot | SIMD matmul | `simd_matmul_rows` |
| HLA readout qᵀ·SK | scalar hd² matvec | SIMD matvec | `simd_matvec` |
| cosine_similarity | scalar dot+norm | SIMD dot | `simd_dot_f32` |
| extract_ddtree_paths | O(3×5×64)=960 scans | O(64) index + O(15) lookup | depth-indexed array |
| forward_prefill | 2× embedding per token | 1× + reuse | fused Phase A+B ✅ |

## Priority Order

1. **T1-T4** (simd_scale_inplace) — highest impact, simplest, general utility
2. **T9** (extract_ddtree_paths) — algorithmic win, easy to verify
3. **T5-T6** (SIMD matmul) — moderate impact, more code to change
4. ~~**T8** (prefill fusion) — done: fused Phase A+B, Q + xr cached~~
5. **T7** (cosine_similarity) — trivial, low priority (validation-only)
6. **T10** (deprecation) — documentation only

## Notes

- Issue 057 already added `simd_dot_f32` for `rmsnorm` pass 1 and `attention_head` Q·K.
- `simd_scale_inplace` is a general utility — benefits softmax, rmsnorm, HLA, TurboQuant, and any future code doing bulk `*= scale`.
- `mat_vec_into` in TurboQuant could delegate to existing `simd_matmul_rows` directly (same row-major layout).
- `mat_vec_t_into` (transpose matmul) doesn't have an existing SIMD implementation — may need new `simd_matvec_transpose`.
- The HLA readout `qᵀ · SK` is a left-multiply (row vector × matrix), while `simd_matvec` does `matrix × column vector`. Need to verify calling convention matches.