# Plan 069: Hot-Path SIMD Scale + Zero-Alloc Audit

> **Status**: Active
> **Depends On**: Plan 060 (SIMD Matmul HLA), Issue 057 (simd_dot_f32 foundation)
> **Issue**: 058_softmax_simd_scale_extract_paths.md

## Objective

Deep audit of `src/` against `.agent/optimization.md` after Issue 057. Implement 10 concrete optimizations across 4 categories: SIMD scale, SIMD matmul, allocation elimination, and algorithmic redundancy.

## Tasks

- [x] T1: Add `simd_scale_inplace` (Category A foundation)
- [x] T2: Wire `simd_scale_inplace` into softmax/rmsnorm (A1, A2)
- [x] T3: Wire `simd_scale_inplace` into HLA decay (A3)
- [x] T4: Wire `simd_scale_inplace` into TurboQuant (A4)
- [x] T5: Replace TurboQuant scalar matmul with `simd_matmul_rows` (B1)
- [x] T6: Replace HLA readout scalar matvec with `simd_matvec` (B2)
- [x] T7: Use `simd_dot_f32` in cosine_similarity (B3)
- [x] T8: Fuse forward_prefill Phase A+B embedding (C1)
- [x] T9: Optimize `extract_ddtree_paths` + `extract_best_path` (D2)
- [x] T10: Document `speculative_step_rollback` / `_paged` as deprecated (D1)

## T1: Add `simd_scale_inplace` (Category A foundation)

**File**: `src/simd.rs` — place before `horizontal_sum_256` (~L549)

Add `simd_scale_inplace(x: &mut [f32], scale: f32)` with:
- NEON (`vmulq_f32`) — 4× f32 per op
- AVX2 (`_mm256_mul_ps`) — 8× f32 per op
- Scalar fallback for remainder elements

Pattern follows existing `neon_dot_f32` / `avx2_dot_f32` style:

```rust
pub fn simd_scale_inplace(x: &mut [f32], scale: f32) {
    #[cfg(target_arch = "aarch64")]
    { unsafe { neon_scale_inplace(x, scale) } }
    #[cfg(target_arch = "x86_64")]
    { if is_avx2_fma_available() { unsafe { avx2_scale_inplace(x, scale) } } else { scalar_scale_inplace(x, scale) } }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    { scalar_scale_inplace(x, scale) }
}
```

## T2: Wire `simd_scale_inplace` into softmax/rmsnorm (A1, A2)

**File**: `src/types.rs`

- `softmax` pass 3 (~L570): replace `for val in x.iter_mut() { *val *= inv_sum; }` → `crate::simd::simd_scale_inplace(x, inv_sum)`
- `softmax_scaled` pass 3 (~L613): same replacement
- `rmsnorm` pass 2 (~L631): replace `for val in x.iter_mut() { *val *= inv_rms; }` → `crate::simd::simd_scale_inplace(x, inv_rms)`

## T3: Wire `simd_scale_inplace` into HLA decay (A3)

**File**: `src/hla/kernel.rs`

Replace all `for x in slice.iter_mut() { *x *= gamma; }` patterns:
- `hla_state_update` (~L99-113): 5 decay loops → 5 `simd_scale_inplace` calls
- `hla_per_head_update` (~L163-176): 5 decay loops
- `ahla_step` (~L362-373): 2 decay loops
- `ahla_per_head_step` (~L565-572): 2 decay loops
- `hla_layer_update` (~L465-468): 2 decay loops
- `ahla_layer_step` (~L648-655): 2 decay loops

## T4: Wire `simd_scale_inplace` into TurboQuant (A4)

**File**: `src/turboquant/kv_cache.rs`

- `store_key` normalize (~L159-163): replace scalar normalize loop
- `store_value` normalize (~L203-207): same
- `dequantize_key_into` scale (~L295-299): replace final scale loop
- `dequantize_value_into` scale (~L340-344): same

## T5: Replace TurboQuant scalar matmul with `simd_matmul_rows` (B1)

**File**: `src/turboquant/kv_cache.rs`

- `mat_vec_into` (~L402-415): delegate to `simd_matmul_rows`
- `mat_vec_t_into` (~L426-438): add new `simd_matvec_transpose` to `src/simd.rs` if needed, or use column-wise `simd_dot_f32`

## T6: Replace HLA readout scalar matvec with `simd_matvec` (B2)

**File**: `src/hla/kernel.rs`

- `hla_readout` qᵀ·SK (~L238-247): use `simd_matvec`
- `hla_denom` qᵀ·SK (~L279-288): use `simd_matvec`
- `ahla_step` qᵀ·PKV (~L380-389): use `simd_matvec`
- `ahla_per_head_step` qᵀ·PKV (~L574-583): use `simd_matvec`

Note: HLA readout is `qᵀ · SK` (row vector × matrix), verify calling convention matches `simd_matvec` (matrix × column vector). May need transpose-aware dispatch.

## T7: Use `simd_dot_f32` in cosine_similarity (B3)

**File**: `src/turboquant/forward.rs` (~L128-134)

Replace scalar dot + norms:
```rust
let dot = simd_dot_f32(a, b, a.len());
let na = simd_dot_f32(a, a, a.len()).sqrt();
let nb = simd_dot_f32(b, b, b.len()).sqrt();
```

## T8: Fuse forward_prefill Phase A+B embedding (C1)

**File**: `src/transformer.rs`

- For single-layer: compute embedding once, reuse across phases
- For multi-layer: store pre-rmsnorm hidden state to avoid recomputation
- Measure impact on prefill latency

## T9: Optimize `extract_ddtree_paths` + `extract_best_path` (D2)

**File**: `src/speculative/step.rs`, `src/speculative/dd_tree.rs`

- Pre-index tree nodes by depth: `[Vec<&TreeNode>; MAX_DEPTH]` built in O(N)
- Replace O(D × N) `.iter().filter()` scans with O(1) depth lookups

## T10: Document `speculative_step_rollback` / `_paged` as deprecated (D1)

**File**: `src/speculative/step.rs`

Add `#[deprecated(note = "Use speculative_step_rollback_with for zero-alloc production path")]` to benchmark-only functions.

## Priority Order

1. **T1-T4** (`simd_scale_inplace`) — highest impact, simplest, general utility
2. **T9** (`extract_ddtree_paths`) — algorithmic win, easy to verify
3. **T5-T6** (SIMD matmul) — moderate impact, more code to change
4. **T8** (prefill fusion) — algorithmic win, but only affects prefill path
5. **T7** (`cosine_similarity`) — trivial, low priority (validation-only)
6. **T10** (deprecation) — documentation only

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
| forward_prefill | 2× embedding per token | 1× + reuse | fused Phase A+B |