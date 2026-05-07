# mini-dllm: Performance Engineering

## Benchmark Results (release build, 50K iterations, Apple Silicon)

| Method | Throughput | μs/step | Avg Accept Len |
|--------|-----------|---------|---------------|
| Transformer AR | 1,102,650 tok/s | 0.91 | 1.00 |
| DFlash (sequential) | 4,205,854 tok/s | 1.90 | 8.00 |
| DDTree Build | 362,181 trees/s | 2.76 | 16.00 |
| Speculative (Simulated) | 1,039,176 tok/s | 4.81 | 5.00 |
| Speculative (AR Draft) | 1,490,570 tok/s | 4.69 | 7.00 |
| Leviathan (Algorithm 1) | 108,885 tok/s | 10.83 | 1.18 |
| Leviathan (w/ rollback) | 161,324 tok/s | 7.28 | 1.18 |
| Spec (conditioned) | 972,163 tok/s | 6.94 | 6.74 |
| Prefill (no compress) | 16,962,509 tok/s | 3.77 | 64.00 |
| Prefill (compressed) | 1,714,061 tok/s | 4.08 | 7.00 |
| DDTree (no chain) | 364,458 trees/s | 2.74 | 16.00 |
| DDTree (chain-seed) | 385,957 trees/s | 2.59 | 16.00 |

Speedup: Speculative vs AR went from **0.72×** → **1.48×** after zero-alloc optimization.

## Hot Path Optimizations

### Inline + Unsafe
- `#[inline(always)]` on all hot kernels: `matmul`, `softmax`, `rmsnorm`, `forward`, `sample_token`
- `get_unchecked` / `get_unchecked_mut` in inner matmul loops — eliminates bounds checks
- `copy_nonoverlapping` for KV cache store — faster than `copy_from_slice` for known sizes
- Edition 2024: explicit `unsafe {}` blocks inside `unsafe fn`

### Fused Kernels
- **`matmul_relu`**: single-pass MLP hidden layer (avoids extra scan of hidden buffer)
- **`attention_head`**: fused score → softmax → weighted value (avoids separate softmax write-back)
- **Optimized softmax**: one-pass exp+sum, `inv_sum = 1.0/sum` multiply instead of divide
- **Optimized rmsnorm**: two-pass with `inv_rms` multiply instead of divide

### Impact
| Method | Before | After | Change |
|--------|--------|-------|--------|
| Transformer AR | 831K tok/s | 1,121K tok/s | **+34.9%** |
| DFlash | 2,941K tok/s | 3,218K tok/s | **+9.4%** |

## Zero-Allocation Strategy

### Problem
Every `dflash_predict`, `build_dd_tree`, and speculative step was allocating `Vec`s per call:
- `ForwardContext::new()` per DFlash call
- `MultiLayerKVCache::new()` per step
- `Vec::with_capacity()` for marginals, tree nodes, heap
- `logits.to_vec()` per forward step

### Solution: Pre-allocated Contexts

**SpeculativeContext** (`speculative/types.rs`):
- Holds `ForwardContext`, `MultiLayerKVCache`, flat marginals buffer, probs buffer, sampled tokens, accepted tokens, path buffer, residual buffer, p_distributions buffer
- `new(config)` allocates once, `reset()` clears for reuse
- All `_with()` function variants accept `&mut SpeculativeContext`

**TreeBuilder** (`speculative/dd_tree.rs`):
- Holds pre-allocated `BinaryHeap<TreeNode>`, `Vec<TreeNode>`, chain buffers
- `build()` clears and reuses internal buffers
- Returns `&[TreeNode]` (borrowed slice)

**ForwardContext** (`transformer.rs`):
- Already pre-allocated since Plan 003
- All forward pass intermediates (x, q, k, v, attn_out, hidden, logits) reused in-place

### Impact
| Method | Before | After | Change |
|--------|--------|-------|--------|
| DFlash | 3,058K tok/s | 4,206K tok/s | **+27%** |
| DDTree Build | 309K trees/s | 362K trees/s | **+15%** |
| Speculative (Simulated) | 834K tok/s | 1,039K tok/s | **+20%** |
| Speculative (AR Draft) | 1,172K tok/s | 1,491K tok/s | **+21%** |
| Prefill (no compress) | 2,639K tok/s | 16,963K tok/s | **+543%** |
| Prefill (compressed) | 284K tok/s | 1,714K tok/s | **+504%** |

## Rayon Parallelism

### Current Usage
- `dflash_predict_parallel` — `into_par_iter` + `map_init` with per-worker `ForwardContext`/`MultiLayerKVCache`
- `generate_batch` — multi-sample generation via `par_iter` with per-worker contexts

### Parallel Threshold
- `config.parallel_threshold` (default 128) — skip `par_iter` when `n_embd ≤ threshold`
- At micro scale (n_embd=16), Rayon overhead (~1-5μs) dominates sequential cost (~0.3μs)
- Parallelism only beneficial when individual forward passes are > 5μs

### What NOT to Parallelize
- DDTree initial heap population — vocab=27, not worth overhead for small vocab
- Benchmark runner — reduces wall time only, not throughput; already fast enough

## PagedKVCache Design

### Problem
DDTree branches clone entire `MultiLayerKVCache` → most data is shared prefix:
- small_target config: 32 × 131 KB = 4.2 MB of near-identical copies

### Solution
```rust
pub struct PagedKVCache {
    pages: Vec<Vec<f32>>,                     // pool of [PAGE_SIZE * kv_dim] pages
    layer_page_tables: Vec<Vec<Vec<usize>>>,  // [layer][seq] → page indices
    free_pages: Vec<usize>,
}
```
- PAGE_SIZE = 16 tokens (power of 2)
- `fork(seq_idx, fork_at_pos)` — copy-on-write: shares prefix pages, new pages only after fork
- `alloc_page()` — reuse from free list or grow pool
- Memory: O(tree_budget × pos_used) instead of O(tree_budget × block_size)

### Status
- Struct implemented and tested (Plan 011)
- DDTree integration pending (Plan 014)
- Currently DDTree uses flat `snapshot()/restore()` which works but copies more data

## What We Don't Do (and why)

| Technique | Reason |
|-----------|--------|
| Rayon parallel matmul | n_embd=16, mlp=64 — thread pool overhead dominates |
| `std::simd` / `portable_simd` | Nightly-only; aarch64 NEON auto-vectorization is sufficient |
| Cache tiling for attention | block_size=16 already fits L1 |
| SIMD intrinsics | Stable Rust lacks `std::simd`; revisit when n_embd ≥ 256 |
| f16/bf16 weights | Would halve memory bandwidth but requires `half` crate; sketched for future |
| GPU compute in inference | CPU-only for inference; GPU reserved for LoRA training (wgpu) |