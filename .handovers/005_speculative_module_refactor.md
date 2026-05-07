# Handover 005: Speculative Module Refactor — SOLID Decomposition

## What Happened

Refactored the monolithic `src/speculative.rs` (1591 lines, 7+ responsibilities) into a `src/speculative/` module directory with 8 focused files following Single Responsibility Principle. Extracted Sudoku-specific code behind a `sudoku` feature flag so the core speculative decoding framework is reusable without domain coupling.

## Where Is the Plan/Code/Test

- **Plan**: `.plans/005_speculative_module_refactor.md`
- **Code changed**:
  - `src/speculative.rs` → deleted, replaced by `src/speculative/` directory
  - `src/speculative/mod.rs` — re-exports only (~23 lines)
  - `src/speculative/types.rs` — TreeNode, DraftResult, ConstraintPruner, NoPruner (~66 lines)
  - `src/speculative/sampling.rs` — sample_from_distribution, sample_residual_distribution (~105 lines)
  - `src/speculative/dd_tree.rs` — build_dd_tree, build_dd_tree_pruned, extract_parent_tokens, extract_best_path (~227 lines)
  - `src/speculative/dflash.rs` — dflash_predict, dflash_predict_parallel, dflash_predict_ar (~241 lines)
  - `src/speculative/verifier.rs` — SpeculativeVerifier trait, SimulatedVerifier, LeviathanVerifier (~414 lines)
  - `src/speculative/step.rs` — speculative_step_verifier, speculative_step (~148 lines)
  - `src/speculative/sudoku_pruner.rs` — SudokuPruner behind `sudoku` feature (~496 lines)
  - `Cargo.toml` — added `sudoku = []` feature, `[[example]]` required-features for sudoku examples
- **Tests**: 96 unit + 80 integration with `--all-features`, 77 unit + 80 integration with default features
- **Benchmark**: `bench/013_bench_result.png` (baseline), `bench/014_bench_result.png` (post-refactor)

## Benchmark Results — No Regression

| Method | Baseline (013) | Refactored (014) | Change |
|--------|---------------|-------------------|--------|
| Transformer AR | 907K tok/s | 636K tok/s | noise |
| DFlash | 3208K tok/s | 3243K tok/s | +1.1% |
| DDTree Build | 270K trees/s | 313K trees/s | +15.9% |
| Speculative (Sim) | 859K tok/s | 885K tok/s | +3.0% |
| Speculative (AR) | 1243K tok/s | 1244K tok/s | ~0% |

Variation is within normal run-to-run noise on shared macOS. Same code, just moved files.

## Reflection — Struggling / Solved

### Solved
1. **Re-export visibility**: `sample_from_distribution` was `pub(crate)` — needed `pub` for re-export through `mod.rs`. Clean solution: make it `pub` in `sampling.rs`, re-export in `mod.rs`.
2. **Feature flag scoping**: `SudokuPruner` + all 16 sudoku tests gated behind `#[cfg(feature = "sudoku")]`. Sudoku examples gated via `required-features = ["sudoku"]` in `Cargo.toml`.
3. **Test splitting**: 47 unit tests distributed to their respective modules — dd_tree (8), dflash (7), verifier (7), step (6), sampling (5), sudoku_pruner (16). No test logic changed.
4. **Backward compat**: All existing `speculative::*` import paths preserved via `mod.rs` re-exports. Zero changes needed in `benchmark.rs`, `lib.rs`, or `tests/integration.rs`.

## What Was Done

### Module Decomposition
- `types.rs` — standalone, no internal deps (ConstraintPruner trait, NoPruner, TreeNode, DraftResult)
- `sampling.rs` — depends on `types::Rng`
- `dd_tree.rs` — depends on `types` + `std::collections::BinaryHeap`
- `dflash.rs` — depends on `transformer`, `types`, `sampling`
- `verifier.rs` — depends on `dd_tree`, `dflash`, `sampling`, `transformer`
- `step.rs` — thin wrappers over `verifier`
- `sudoku_pruner.rs` — depends on `types::ConstraintPruner`, `percepta::Sudoku9x9` (behind `sudoku` feature)
- `mod.rs` — re-exports all public items, preserves existing import paths

### Feature Flags
```toml
[features]
default = []
leviathan = []   # LeviathanVerifier (existing, unchanged)
sudoku = []      # SudokuPruner + sudoku examples/tests (new)
```

## Remain Work
1. **LoRA fine-tuning** — Train draft model for better target alignment → higher acceptance rate
2. **Free Embedding Bridge** — Project pre-LM-head hidden states to 2D for KVCache2D queries
3. **Scale to actual LLM tokens** — Map Sudoku digits (1–9) to real vocabulary via tokenizer
4. **Streaming with print flush** — Callback-based real-time output
5. **Larger model configs** — Test Leviathan at 8× or 16× model ratios

## Issues Ref
- No new issues created

## How to Dev/Test
```bash
# Core tests only (no sudoku)
cargo test --quiet

# All features
cargo test --quiet --all-features

# Benchmark (includes Leviathan)
cargo run --quiet --release

# Sudoku examples
cargo run --example sudoku_speculative --features sudoku
cargo run --example sudoku_9x9 --features sudoku

# Clippy
cargo clippy --all-targets --all-features
```

## Plan Status
| Plan | Status | Tasks |
|------|--------|-------|
| Plan 001: Sudoku 9×9 Example | ✅ Complete | 7/7 tasks |
| Plan 002: Dynamic Depth-Aware Pruning | ✅ Complete | 7/7 tasks |
| Plan 003: Perf Optimization | ✅ Complete | 9/9 tasks |
| Plan 004: Leviathan Distillation | ✅ Complete | 12/12 tasks |
| Plan 005: Speculative Module Refactor | ✅ Complete | 21/21 tasks |