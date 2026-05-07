# Plan 013: Zero-Alloc + Rayon Optimization

## Goal
Eliminate heap allocations from hot paths (speculative decoding, forward pass, sampling)
and expand rayon parallelism to all embarrassingly-parallel operations.

## Context

### Already Zero-Alloc
- `ForwardContext` (in `transformer.rs:153`) provides pre-allocated buffers for forward passes
- `bench_ar` already creates `ForwardContext` + `MultiLayerKVCache` outside the loop, calls `cache.reset()` inside
- `LeviathanVerifier` already holds pre-allocated `target_ctx: ForwardContext` + `target_cache: MultiLayerKVCache`
- `dflash_predict_parallel` already uses `map_init` with per-worker `ForwardContext` + `MultiLayerKVCache`

### Current Allocation Sites (hot paths that allocate per call)

| Function | File | What allocates |
|---|---|---|
| `dflash_predict` | `dflash.rs:12-25` | `ForwardContext`, `MultiLayerKVCache` per call; **new `MultiLayerKVCache` per step** inside loop; `Vec::with_capacity` marginals; `logits.to_vec()` per step |
| `dflash_predict_ar` | `dflash.rs:49-72` | `ForwardContext`, `MultiLayerKVCache` per call; `Vec::with_capacity` marginals + sampled_tokens; `logits.to_vec()` per step |
| `dflash_predict_conditioned` | `dflash.rs:84-132` | Same as `_ar` — `ForwardContext`, `MultiLayerKVCache`, marginals, sampled_tokens, `logits.to_vec()` per step |
| `build_dd_tree_pruned` | `dd_tree.rs:48-216` | `Vec::with_capacity(tree_budget)` tree + `BinaryHeap` + `chain_nodes: Vec` + `chain_parent_tokens: Vec` per call |
| `sample_residual_distribution` | `sampling.rs:27-44` | `Vec<f32>` residual (vocab_size) per call |
| `SimulatedVerifier::speculate` | `verifier.rs:44-74` | Calls `dflash_predict` (all its allocs) + `build_dd_tree` (all its allocs) + `extract_best_path` + `accepted: Vec` + bonus `result: Vec` |
| `LeviathanVerifier::speculate` | `verifier.rs:112-187` | Calls `dflash_predict_ar` (all its allocs) + `p_distributions: Vec<Vec<f32>>` + `accepted: Vec` per call |
| `speculative_step` | `step.rs:19-23` | Creates new `SimulatedVerifier` per call |
| `generate` | `transformer.rs:382-422` | `ForwardContext`, `MultiLayerKVCache`, `Vec::with_capacity(n_tokens)` tokens per call |
| `bench_dflash` | `benchmark.rs:92-109` | `dflash_predict` alloc inside loop (context not reused) |
| `bench_ddtree` | `benchmark.rs:111-129` | `build_dd_tree` alloc inside loop (heap not reused) |
| `bench_speculative` | `benchmark.rs:131-156` | `SimulatedVerifier` created outside loop ✓, but internal allocs per iter |
| `bench_speculative_ar` | `benchmark.rs:168-189` | `run_speculative_ar_step` alloc inside loop (full pipeline allocs) |
| `run_speculative_ar_step` | `benchmark.rs:196-232` | `dflash_predict_ar` + `build_dd_tree` + `path: Vec` + `accepted: Vec` + `result: Vec` per call |
| `bench_ddtree_chain_seed` | `benchmark.rs:249-291` | `build_dd_tree_pruned` alloc inside both loops |
| `bench_ddtree_budget_sweep` | `benchmark.rs:301-350` | `build_dd_tree_pruned` alloc inside all loops |
| `PrefillScorer::score` (impls) | `prefill.rs` | `ForwardContext`, `MultiLayerKVCache` per call |

### Rayon Usage (current)
- Only `dflash_predict_parallel` uses rayon (`into_par_iter` + `map_init`)
- All benchmarks run sequentially
- `generate()` runs sequentially
- Tree building runs sequentially

## Tasks

### 1. Pre-allocated SpeculativeContext
- [ ] Create `SpeculativeContext` struct in `src/speculative/types.rs`
  - `ctx: ForwardContext` — pre-allocated forward pass buffers
  - `cache: MultiLayerKVCache` — pre-allocated KV cache
  - `marginals_flat: Vec<f32>` — `[draft_lookahead * vocab_size]` flat buffer, slice per step
  - `marginals: Vec<&[f32]>` — views into `marginals_flat` (or use stride math)
  - `probs_buf: Vec<f32>` — `[vocab_size]` temp for logits→softmax (replaces `logits.to_vec()`)
  - `sampled_tokens: Vec<usize>` — `[draft_lookahead]` pre-allocated
  - `accepted_buf: Vec<usize>` — `[draft_lookahead + 1]` pre-allocated
  - `path_buf: Vec<usize>` — `[draft_lookahead + 1]` pre-allocated
  - `residual_buf: Vec<f32>` — `[vocab_size]` for `sample_residual_distribution`
- [ ] `SpeculativeContext::new(config: &Config)` — allocate all buffers from config dims
- [ ] `SpeculativeContext::reset()` — clear lengths to 0, zero-fill as needed

### 2. Zero-alloc DFlash predict
- [ ] Add `dflash_predict_with(ctx: &mut SpeculativeContext, draft_weights, draft_config, token, pos) -> &[&[f32]]`
  - Reuse `ctx.ctx`, `ctx.cache`, `ctx.marginals_flat`, `ctx.probs_buf`
  - Note: current sequential `dflash_predict` creates **new `MultiLayerKVCache` per step** (independent marginals) — with context, `cache.reset()` per step instead
  - Write softmax output directly into `marginals_flat` slices instead of `logits.to_vec()`
- [ ] Add `dflash_predict_ar_with(ctx: &mut SpeculativeContext, draft_weights, draft_config, token, pos, rng) -> (&[&[f32]], &[usize])`
  - Single cache (no reset per step — autoregressive), reuse `ctx.sampled_tokens`
- [ ] Update `dflash_predict_conditioned` to accept `&mut SpeculativeContext` similarly
  - Uses `ctx.cache` seeded with target hidden state
- [ ] Keep old `dflash_predict`, `dflash_predict_ar`, `dflash_predict_conditioned` as thin wrappers (create context, call `_with`, return owned Vecs) for backward compat
- [ ] Update `dflash_predict_parallel` — each worker already has own `ForwardContext`/`KVCache` via `map_init`; replace `logits.to_vec()` with in-place reuse

### 3. Zero-alloc DDTree build
- [ ] Create `TreeBuilder` struct in `src/speculative/dd_tree.rs`
  - `heap: BinaryHeap<TreeNode>` — pre-allocated, cleared via `clear()` (reuses capacity)
  - `tree: Vec<TreeNode>` — pre-allocated `[tree_budget]`, cleared via `clear()`
  - `chain_nodes: Vec<TreeNode>` — `[draft_lookahead]` for chain-seed phase
  - `chain_parent_tokens: Vec<usize>` — `[draft_lookahead]` for pruner path
  - `scratch: Vec<TreeNode>` — for `merge_retrieved_branches` sorting
- [ ] `TreeBuilder::new(config: &Config)` — allocate from config dims
- [ ] `TreeBuilder::build(&mut self, marginals, config, pruner, chain_seed) -> &[TreeNode]`
  - Clear + reuse `heap`, `tree`, `chain_nodes`, `chain_parent_tokens`
  - Return `&self.tree` (borrowed slice)
- [ ] `TreeBuilder::build_and_merge(&mut self, marginals, config, pruner, chain_seed, retrieved, rest_weight) -> &[TreeNode]`
  - For REST feature: build + merge in one call with shared scratch

### 4. Zero-alloc sampling
- [ ] Add `sample_residual_distribution_into(p, q, scratch: &mut [f32], rng) -> usize`
  - `scratch` must be `>= p.len()`; uses it for residual computation
  - Reuses `SpeculativeContext::residual_buf`
- [ ] Keep old `sample_residual_distribution` as wrapper (allocs Vec, calls `_into`)
- [ ] `sample_from_distribution` is already zero-alloc (iterates in-place) ✓

### 5. Zero-alloc SpeculativeVerifier
- [ ] Refactor `SimulatedVerifier` to hold `SpeculativeContext` + `TreeBuilder`
  - `SimulatedVerifier::new(acceptance_rate, draft_config)` — pre-alloc internal buffers
  - `speculate()` reuses all internal buffers, returns `Vec<usize>` (one alloc for result, or borrow from context)
  - Consider: `speculate_into(&mut self, ..., out: &mut Vec<usize>)` for true zero-alloc
- [ ] Refactor `LeviathanVerifier` similarly (feature-gated)
  - Already holds `target_ctx` + `target_cache` ✓
  - Add `SpeculativeContext` for draft-side (replaces per-call allocations in `dflash_predict_ar`)
  - Add `p_distributions_flat: Vec<f32>` — `[(draft_lookahead + 1) * vocab_size]` flat buffer
  - Add `residual_buf: Vec<f32>` — `[vocab_size]` for `sample_residual_distribution_into`
  - Add `accepted_buf: Vec<usize>` — `[draft_lookahead + 1]` pre-allocated

### 6. Zero-alloc benchmark hot loop
- [ ] `bench_ar`: already optimal ✓ (`ForwardContext` + `KVCache` outside loop, `cache.reset()` inside)
- [ ] `bench_dflash`: create `SpeculativeContext` outside loop, call `dflash_predict_with` + `ctx.reset()` inside
- [ ] `bench_ddtree`: create `TreeBuilder` outside loop, call `builder.build()` + reuse inside
- [ ] `bench_speculative`: `SimulatedVerifier` already outside loop ✓; ensure internal context is reused (depends on Task 5)
- [ ] `bench_speculative_ar`: refactor `run_speculative_ar_step` to accept `&mut SpeculativeContext` + `&mut TreeBuilder`
- [ ] `bench_leviathan`: `LeviathanVerifier` already outside loop ✓; ensure internal context reused
- [ ] `bench_ddtree_chain_seed`: create one `TreeBuilder` outside both loops, reuse for both chain/no-chain
- [ ] `bench_ddtree_budget_sweep`: create one `TreeBuilder` outside outer loop, reuse for all budgets

### 7. Expand Rayon parallelism
- [ ] `dflash_predict_parallel`: verify it avoids `logits.to_vec()` after Task 2 (use in-place probs buffer per worker)
- [ ] Add rayon parallel benchmark runner: `run_all_parallel()` — run independent bench methods via `par_iter`
  - `bench_ar`, `bench_dflash`, `bench_ddtree` are independent (different weights/configs)
  - `bench_speculative`, `bench_speculative_ar` use draft weights (can parallel with `bench_ar`)
  - `bench_leviathan` uses both weights
- [ ] Add `generate_batch()` for multi-sample generation — `n_samples` independent generate calls via `par_iter`, each with own `ForwardContext` + `KVCache` via `map_init`
- [ ] Consider rayon for `build_dd_tree_pruned` initial heap population (iterating `marginals[0]`) — likely not worth overhead for small vocab (27), skip unless vocab > 256
- [ ] Note: parallel forward pass for batch inference is blocked by mutable borrow on `ForwardContext`; would need one context per batch item (see `generate_batch` approach)

### 8. Zero-alloc generate()
- [ ] Add `generate_into(ctx: &mut ForwardContext, cache: &mut MultiLayerKVCache, weights, config, rng, tokens: &mut Vec<usize>, n_tokens)`
  - `ctx` and `cache` created by caller, reused across calls
  - `tokens` cleared and filled by callee
  - Uses existing `forward()` which already borrows `ctx` mutably
- [ ] Keep `generate()` as thin wrapper (creates context, calls `generate_into`, returns tokens)
- [ ] Update `speculative_step_rest` (REST feature) to accept pre-allocated buffers

### 9. Benchmarks: before vs after
- [ ] Run current benchmarks, record baseline (`cargo run --quiet` with `RUST_LOG=info`)
- [ ] Implement all tasks
- [ ] Run optimized benchmarks, compare throughput/time_per_step_us
- [ ] Add allocation tracking: `#[cfg(debug_assertions)]` count `Vec::new()` / `Vec::with_capacity()` via `#[global_allocator]` wrapper
- [ ] Update bench chart/plot with "Zero-Alloc" label

### 10. Tests
- [ ] Test `SpeculativeContext` lifecycle: `new(draft_config)` → `dflash_predict_with` → `reset()` → `dflash_predict_ar_with` → `reset()` → reuse
- [ ] Test `TreeBuilder` reuse: `new(config)` → `build()` → `build()` → verify results identical to fresh builds
- [ ] Test `dflash_predict_with` produces identical marginals to `dflash_predict` (bit-exact for same seed)
- [ ] Test `dflash_predict_ar_with` produces identical results to `dflash_predict_ar`
- [ ] Test `sample_residual_distribution_into` matches `sample_residual_distribution` for same inputs
- [ ] Test `generate_into` produces identical tokens to `generate` for same seed
- [ ] Verify all existing tests still pass with new APIs (backward compat wrappers)
- [ ] Test rayon parallel paths (`dflash_predict_parallel` after fix) produce same count as sequential
- [ ] Test `SimulatedVerifier` with internal context produces same accepted tokens as before

## File Changes Summary
- `src/speculative/types.rs` — add `SpeculativeContext` struct + impl
- `src/speculative/dflash.rs` — add `_with` variants, fix parallel `logits.to_vec()`, update `_conditioned`
- `src/speculative/dd_tree.rs` — add `TreeBuilder` struct + impl
- `src/speculative/sampling.rs` — add `sample_residual_distribution_into`
- `src/speculative/verifier.rs` — zero-alloc `SimulatedVerifier` + `LeviathanVerifier` internals
- `src/speculative/step.rs` — add `speculative_step_with` variants, update REST step
- `src/speculative/mod.rs` — new re-exports (`SpeculativeContext`, `TreeBuilder`, `_with` fns)
- `src/benchmark.rs` — zero-alloc bench loops, optional `run_all_parallel()`
- `src/transformer.rs` — add `generate_into`, keep `generate` as wrapper
- `src/speculative/prefill.rs` — update scorers to accept `&mut SpeculativeContext` (optional, lower priority)

## Dependency Order
1. `SpeculativeContext` (Task 1) — foundation for everything else
2. `sample_residual_distribution_into` (Task 4) — simple, no deps
3. `dflash_predict_with` (Task 2) — depends on Task 1
4. `TreeBuilder` (Task 3) — independent of Tasks 2-3
5. Verifier refactor (Task 5) — depends on Tasks 1-4
6. Benchmark updates (Task 6) — depends on Tasks 1-5
7. `generate_into` (Task 8) — independent, can parallel with Tasks 2-5
8. Rayon expansion (Task 7) — depends on Tasks 5-6
9. Before/after benchmarks (Task 9) — depends on all above
10. Tests (Task 10) — continuous, add as each task completes

## Performance Targets
- Zero allocations per forward pass ✓ (already done)
- Zero allocations per `dflash_predict` call (new — currently ~3+ allocs per step × steps)
- Zero allocations per DDTree build (new — currently 3-4 allocs per call)
- Zero allocations per speculative step (new — currently ~10+ allocs cascading)
- Zero allocations per `sample_residual_distribution` (new — currently 1 alloc per call)
- Measurable throughput improvement in `bench_dflash` from eliminated per-step `MultiLayerKVCache` alloc
- Measurable throughput improvement in `bench_speculative` from eliminated cascading malloc/free