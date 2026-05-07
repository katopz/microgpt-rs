# Handover 008: Plan 012 — Lucebox-Hub Distill (Chain-Seed DDTree + Speculative Prefill + KV Rollback)

## What Happened

Implemented Phases 1–5 of Plan 012 (Lucebox-Hub technique distillation). All core algorithmic changes are done and passing diagnostics with zero warnings/errors.

### Completed

| Phase | Description | Status |
|-------|-------------|--------|
| Phase 1 | Chain-Seed DDTree | ✅ All 8 tasks done |
| Phase 2 | DDTree Budget Sweep | ✅ 5 of 7 tasks done (2.3 multi-config sweep, 2.6 record results pending) |
| Phase 3 | KV-Cache Snapshot & Rollback | ✅ 8 of 9 tasks done (3.9 deferred to plan 011) |
| Phase 4 | Speculative Prefill (PFlash) | ✅ 6 of 9 tasks done (4.7 NIAH test, 4.8 bench, 4.9 REST bridge pending) |
| Phase 5 | Target-Conditioned Draft | ✅ All 7 tasks done |
| Phase 6 | Benchmark After & Documentation | ❌ Not started |

## Where Is the Plan/Code/Test

- **Plan**: `.plans/012_lucebox_distill.md` — checkboxes updated
- **Code changes**:
  - `src/speculative/dd_tree.rs` — Added `chain_seed: bool` to `build_dd_tree_pruned()`, 5 new tests
  - `src/transformer.rs` — Added `KVSnapshot`, `KVLayerSnapshot`, `snapshot()`, `restore()`, 5 new tests
  - `src/speculative/prefill.rs` — **NEW**: `PrefillScorer` trait, `AttentionScorer`, `RandomScorer`, `UniformScorer`, `compress_prompt()`, `speculative_prefill()`, 9 new tests
  - `src/speculative/dflash.rs` — Added `dflash_predict_conditioned()`, 4 new tests
  - `src/speculative/step.rs` — Added `speculative_step_rollback()` (Task 3.4), `speculative_step_conditioned()` (Task 5.7), `extract_ddtree_paths()`, 6 new tests behind `leviathan` feature
  - `src/speculative/mod.rs` — Added `pub mod prefill`, re-exports for all new public items including `speculative_step_rollback` and `speculative_step_conditioned`
  - `src/benchmark.rs` — Added `bench_ddtree_chain_seed()`, `bench_ddtree_budget_sweep()`, `bench_snapshot_rollback()` (Task 3.8), `bench_conditioned_vs_unconditioned()` (Task 5.4), updated `run_all()`
  - `src/main.rs` — Added budget sweep output section
  - `src/types.rs` — Added `#[derive(Clone)]` to `Config`
  - `examples/sudoku_speculative.rs`, `examples/sudoku_tui.rs`, `src/speculative/sudoku_pruner.rs` — Updated `build_dd_tree_pruned()` call sites to pass `chain_seed=false`

## Reflection: Struggling / Solved

- **Chain-seed cumulative scores**: Plan sketch used per-depth `marginals[depth][token].ln()` but existing tree uses cumulative log-prob. Fixed implementation to use cumulative scores for consistency. Siblings compute score as `parent_chain_score + log(prob)`.
- **AttentionScorer**: `ForwardContext.scores` is only `[block_size]` (overwritten per head), not `[n_head * block_size]`. Adapted to use last head's softmax'd self-attention weight at `ctx.scores[pos]` as importance proxy.
- **Config Clone**: Budget sweep needed `config.clone()` — added `#[derive(Clone)]` to `Config`.
- **ForwardContext.scores visibility**: Made `scores` field `pub` so `AttentionScorer` can read attention weights from outside the module.
- **Borrow checker in speculative_step_conditioned**: `forward()` mutably borrows `target_ctx`, then reading `target_ctx.hidden_state` conflicts. Fixed by chaining `.to_vec()` on the `forward()` return to release the borrow before cloning `hidden_state`.
- **Move-after-use in return tuples**: `return (result, result.len())` moves `result` then tries to borrow it. Fixed by computing `let len = result.len()` before the return.
- **extract_ddtree_paths**: Needed to extract top-3 root branches from DDTree for multi-path rollback verification. Each branch follows best children at subsequent depths, filtering by parent_path bitfield continuity.

## Remain Work

### Medium Priority
- [ ] **Task 2.3**: Run budget sweep for all 4 configs (micro, draft, small_target, gqa_draft)
- [ ] **Task 2.6**: Record optimal budgets in the plan table
- [ ] **Task 4.7**: NIAH-style needle-in-haystack test after compression
- [ ] **Task 4.8**: Prefill compression bench
- [ ] **Task 4.9**: Bridge prefill → REST speculative step integration test

### Low Priority (Phase 6)
- [ ] **Tasks 6.1–6.5**: Run full benchmarks, record results, verify all tests pass
- [ ] **Tasks 6.6–6.10**: Update README with Lucebox-Hub references, features, benchmarks
- [ ] **Task 6.11**: Commit with conventional message

## Issues Ref

- No issues filed for this plan.

## How to Dev/Test

```bash
# Build check
cargo check

# Run all lib tests (211 total: 131 lib + 80 example)
cargo test --quiet

# Run with leviathan feature (includes rollback + conditioned tests)
cargo test --quiet --features leviathan

# Run specific test modules
cargo test --lib -- dd_tree::tests
cargo test --lib -- transformer::tests::test_snapshot
cargo test --lib -- prefill::tests
cargo test --lib -- dflash::tests::test_dflash_conditioned
cargo test --lib --features leviathan -- step::tests::test_speculative_step_rollback
cargo test --lib --features leviathan -- step::tests::test_speculative_step_conditioned

# Run with all features
cargo test --quiet --all-features

# Clippy (zero warnings)
cargo clippy --all-targets --all-features --quiet

# Run benchmarks (includes chain-seed + budget sweep + rollback + conditioned)
cargo run --release
cargo run --release --features leviathan
```

## Key Architecture Decisions

1. **Chain-seed is additive** — `chain_seed=false` preserves original behavior exactly
2. **KV snapshot copies only `[0..pos * kv_dim]`** — cheap at our model scale (2KB–128KB)
3. **Target conditioning via KV seed (Option C)** — simplest, no weight changes
4. **PrefillScorer trait** — swappable scoring for ablation (Attention/Random/Uniform)
5. **dflash_predict_conditioned re-exported** from speculative module for external use
6. **speculative_step_rollback** — new function (not modifying speculative_step_verifier) behind `leviathan` feature; extracts top-3 DDTree paths, verifies each with p/q rejection + KV snapshot/rollback
7. **speculative_step_conditioned** — new function behind `leviathan` feature; target forward → hidden state → conditioned draft → DDTree → simulated acceptance
8. **All integration functions are separate** from the existing `speculative_step_verifier` trait pattern — no breaking changes to existing code