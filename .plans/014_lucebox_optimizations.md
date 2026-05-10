# Plan 014: Lucebox Optimizations — PagedKVCache Integration + Rayon Threshold + f16 Prep

## Goal
Wire the existing `PagedKVCache` into the DDTree exploration loop to enable memory-efficient
branch sharing, add a Rayon threshold to avoid parallelism overhead on micro models, and
prepare the weight storage layer for future f16/bf16 half-precision support.

## Context

Plan 013 achieved zero-alloc hot paths. Plan 011 implemented `PagedKVCache` with `fork()`,
but it remains an isolated island — only tested, never integrated. The DDTree exploration
loop currently snapshots the entire flat `MultiLayerKVCache` for each branch, copying data
that's identical across all branches (the shared prefix).

### Current DDTree Branch Cost
- Each DDTree branch clones `MultiLayerKVCache` via `snapshot()` → `restore()`
- For micro config: `n_layer=1, block_size=16, kv_dim=16` → 512 bytes per clone
- For small_target config: `n_layer=4, block_size=256, kv_dim=64` → 131 KB per clone
- With `tree_budget=32`: 32 × 131 KB = **4.2 MB** of near-identical copies
- PagedKVCache `fork()` shares prefix pages → only new pages allocate after fork point

### Rayon Threshold
`dflash_predict_parallel` spawns Rayon workers unconditionally. For micro models
(`n_embd=4, head_dim=2`), the sequential forward pass takes ~0.3µs — Rayon thread
synchronization overhead (~1-5µs) dominates. Plan 013 showed sequential DFlash at
1.90µs vs the overhead of waking workers.

### f16 Preparation
Current `TransformerWeights` stores `Vec<f32>`. Real LLM weights are distributed in
`f16`/`bf16` (GGUF, safetensors). No code changes now — just ensure the architecture
doesn't prevent a future `half::f16` storage layer that casts to `f32` for computation.

## Baseline Benchmarks (from Plan 013, release build, 50K iters)

| Method | Throughput | μs/step | Avg Accept Len |
|---|---|---|---|
| Transformer AR | 1,102,650 tok/s | 0.91 | 1.00 |
| DFlash (sequential) | 4,205,854 tok/s | 1.90 | 8.00 |
| DDTree Build | 362,181 trees/s | 2.76 | 16.00 |
| Speculative (Simulated) | 1,039,176 tok/s | 4.81 | 5.00 |
| Speculative (AR Draft) | 1,490,570 tok/s | 4.69 | 7.00 |
| DDTree (no chain) | 364,458 trees/s | 2.74 | 16.00 |
| DDTree (chain-seed) | 385,957 trees/s | 2.59 | 16.00 |

## Tasks

- [x] 1. **Benchmark baseline** — run `cargo bench` on current `develop` branch, record results as `bench/023_bench_result.png`
- [x] 2. **Add Rayon threshold to `dflash_predict_parallel`** — skip `par_iter` when `config.n_embd <= 128`, fall through to sequential `dflash_predict`. Add config field `parallel_threshold: usize` defaulting to 128. Test both paths produce identical marginals.
- [x] 3. **Create `PagedKVCache::forward_paged()` adapter** — new function in `transformer.rs` that accepts `PagedKVCache` + `seq_idx` instead of `MultiLayerKVCache`. Reuses the same `ForwardContext` and `attention_head` kernel, but reads/writes KV via `paged_cache.read_kv()` / `paged_cache.write_kv()`.
- [x] 4. **Create `DDTreeBranchCache` struct** — wraps `PagedKVCache` with a branch allocator. Tracks `active_branches: Vec<usize>` (seq indices). Provides `fork_branch(from, at_pos) -> new_seq_idx` and `forward_branch(seq_idx, token, pos) -> logits`.
- [x] 5. **Wire `DDTreeBranchCache` into speculative step** — added `speculative_step_rollback_paged` in `speculative/step.rs` using `DDTreeBranchCache` for draft model KV exploration with copy-on-write fork semantics. Target model verification still uses `MultiLayerKVCache` snapshot/restore as fallback.
- [x] 6. **Add paged branch benchmarks** — benchmark DDTree build + speculative step with paged cache vs flat cache. Record as `bench/024_bench_result.png`.
- [x] 7. **Verify f16 compatibility** — add a `WeightStorage` enum or trait sketch in `types.rs` that shows how `f16` storage would work without changing hot-path signatures. Document in code comments only — no implementation. Ensure `TransformerWeights` fields could be replaced with `Vec<half::f16>` without breaking `forward()`.
- [x] 8. **Run all tests** — `cargo test --quiet`, `cargo clippy --fix --allow-dirty`, verify zero warnings
- [x] 9. **Final benchmark** — run full benchmark suite, record as `bench/025_bench_result.png`, compare vs baseline
- [x] 10. **Commit** with message `feat: wire PagedKVCache into DDTree branches, add Rayon threshold`

## Architecture

### PagedKVCache Forward Adapter

```
forward_paged(ctx, weights, paged_cache, seq_idx, token, pos, config)
  ├── embedding: same as forward() — wte[token] + wpe[pos]
  ├── per-layer:
  │   ├── rmsnorm + QKV projection (same)
  │   ├── paged_cache.write_kv(layer, seq, pos, k, v)  ← NEW
  │   ├── attention_head: read from paged_cache pages    ← NEW
  │   ├── output projection + residual (same)
  │   └── MLP + residual (same)
  └── LM head (same)
```

### DDTreeBranchCache

```
DDTreeBranchCache {
    paged: PagedKVCache,
    branch_count: usize,
    max_branches: usize,   // = tree_budget
}
```

### Rayon Threshold Logic

```
dflash_predict_parallel:
  if config.n_embd <= config.parallel_threshold {
    return dflash_predict(...)   // sequential fallback
  }
  // else: rayon par_iter as before
```

### f16 Sketch (documentation only)

```
// Future: TransformerWeights could store f16 weights
// pub wte: Vec<f16>,  // halved memory bandwidth
// matmul would cast: f32_weight = f16_weight.to_f32()
// KV cache stays f32 for accumulation precision
```

## Files to Modify

| File | Changes |
|------|---------|
| `src/transformer.rs` | Add `forward_paged()` function, `attention_head_paged()` helper |
| `src/speculative/dd_tree.rs` | No changes — tree builder is cache-agnostic |
| `src/speculative/step.rs` | Add paged cache branch in `speculative_step*` functions |
| `src/speculative/types.rs` | Add `DDTreeBranchCache` struct + impl |
| `src/speculative/dflash.rs` | Add Rayon threshold guard in `dflash_predict_parallel` |
| `src/types.rs` | Add `parallel_threshold` to `Config`, document f16 path in comments |
| `src/benchmark.rs` | Add `bench_paged_speculative` benchmark variant |

## What We Will NOT Do

- **SIMD intrinsics**: Stable Rust doesn't have `std::simd`. LLVM auto-vectorizes our small loops adequately. Revisit only when `n_embd >= 256`.
- **f16 implementation**: Requires `half` crate dependency and changes to every weight access. Sketch only for now.
- **GPU compute**: Out of scope — this is a CPU inference engine.
- **Real model weight loading**: Out of scope for this plan — the paged cache integration is independent of weight provenance.

## Success Criteria

- [x] All existing tests pass (zero regressions)
- [x] New `forward_paged()` passes same correctness tests as `forward()`
- [x] Paged branch exploration uses less memory than flat snapshot/restore
- [x] Rayon threshold prevents parallelism overhead on micro configs
- [x] Benchmark comparison shows paged cache overhead is < 10% vs flat cache
- [x] Zero clippy warnings
- [x] Clean commit on `develop` branch