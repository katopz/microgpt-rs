# Plan 005: Speculative Module Refactor — SOLID Decomposition

## Objective
Decompose the monolithic `src/speculative.rs` (1591 lines) into a `src/speculative/` module directory following Single Responsibility Principle. Extract Sudoku-specific code behind a `sudoku` feature flag so the core speculative decoding framework is reusable without domain coupling.

## Current Problem
- `speculative.rs` imports `percepta::Sudoku9x9` → any consumer pulls in Sudoku domain
- 1591 lines mixing: core types, sampling math, DDTree, DFlash, verifiers, Sudoku pruner, 900+ lines of tests
- Not SOLID: one file owns 7+ responsibilities
- Not reusable for C++→Rust rewrites or other domains

## Proposed Module Layout

```
src/speculative/
├── mod.rs              # Re-exports only (~40 lines)
├── types.rs            # TreeNode, DraftResult, ConstraintPruner trait, NoPruner
├── sampling.rs         # sample_from_distribution, sample_residual_distribution
├── dd_tree.rs          # build_dd_tree, build_dd_tree_pruned, extract_parent_tokens, extract_best_path
├── dflash.rs           # dflash_predict, dflash_predict_parallel, dflash_predict_ar
├── verifier.rs         # SpeculativeVerifier trait, SimulatedVerifier, LeviathanVerifier
├── step.rs             # speculative_step_verifier, speculative_step (thin wrappers)
└── sudoku_pruner.rs    # SudokuPruner (behind "sudoku" feature flag)
```

## Feature Flags (Cargo.toml)

```toml
[features]
default = []
leviathan = []
sudoku = []
```

- `leviathan` — gates `LeviathanVerifier` (existing, unchanged)
- `sudoku` — gates `SudokuPruner`, sudoku examples, sudoku-specific tests

## Dependency Graph

```
types.rs         ← standalone (no internal deps)
sampling.rs      ← depends on types::Rng
dd_tree.rs       ← depends on types, sampling
dflash.rs        ← depends on types, transformer, sampling
verifier.rs      ← depends on types, dd_tree, dflash, sampling, transformer
step.rs          ← depends on verifier
sudoku_pruner.rs ← depends on types, percepta::Sudoku9x9 (behind sudoku feature)
mod.rs           ← re-exports everything
```

## Tasks

- [x] 1. **Create `src/speculative/` directory structure** — Created 8 files: mod.rs, types.rs, sampling.rs, dd_tree.rs, dflash.rs, verifier.rs, step.rs, sudoku_pruner.rs
- [x] 2. **Extract `types.rs`** — Moved `TreeNode`, `DraftResult`, `ConstraintPruner` trait, `NoPruner`, `Eq`/`Ord` impls. No internal imports needed.
- [x] 3. **Extract `sampling.rs`** — Moved `sample_from_distribution`, `sample_residual_distribution`. Changed `pub(crate)` → `pub` for re-export.
- [x] 4. **Extract `dd_tree.rs`** — Moved `build_dd_tree`, `build_dd_tree_pruned`, `extract_parent_tokens`, `extract_best_path` + 8 unit tests.
- [x] 5. **Extract `dflash.rs`** — Moved `dflash_predict`, `dflash_predict_parallel`, `dflash_predict_ar` + 7 unit tests.
- [x] 6. **Extract `verifier.rs`** — Moved `SpeculativeVerifier` trait, `SimulatedVerifier`, `LeviathanVerifier` + 7 tests (4 core, 3 leviathan-gated).
- [x] 7. **Extract `step.rs`** — Moved `speculative_step_verifier`, `speculative_step` + 6 tests.
- [x] 8. **Extract `sudoku_pruner.rs`** — Moved `SudokuPruner` behind `#[cfg(feature = "sudoku")]` + 16 tests.
- [x] 9. **Write `mod.rs`** — Re-exports all public items from submodules. Gates `sudoku_pruner` behind `#[cfg(feature = "sudoku")]`.
- [x] 10. **Update `Cargo.toml`** — Added `sudoku = []` feature + `[[example]]` required-features for sudoku examples.
- [x] 11. **Update `src/lib.rs`** — No change needed (already `pub mod speculative;`).
- [x] 12. **Update `src/benchmark.rs`** — No change needed (works via `speculative::*` re-exports).
- [x] 13. **Update examples** — Added `required-features = ["sudoku"]` in Cargo.toml for both sudoku examples.
- [x] 14. **Move tests to each submodule** — Split 47 unit tests to matching files. 16 sudoku tests behind `#[cfg(feature = "sudoku")]`.
- [x] 15. **Update `tests/integration.rs`** — No change needed (all `speculative::` imports resolve via re-exports).
- [x] 16. **Run `cargo check --all-features`** — Zero errors.
- [x] 17. **Run `cargo clippy --all-features`** — Zero warnings.
- [x] 18. **Run `cargo test --all-features`** — 96 unit + 80 integration = 176 tests pass, no regression.
- [x] 19. **Run `cargo test`** (default features) — 77 unit + 80 integration = 157 tests pass.
- [x] 20. **Run `cargo run --release`** — Benchmark runs, captured as `bench/014_bench_result.png`. No regression vs baseline 013.
- [x] 21. **Commit** with message `refactor: decompose speculative module into SOLID submodules`.

## File-by-File Breakdown

### `types.rs` (~60 lines)
```rust
// TreeNode, DraftResult, ConstraintPruner, NoPruner
// No internal crate imports — only std
```

### `sampling.rs` (~40 lines)
```rust
// sample_from_distribution, sample_residual_distribution
// Uses: crate::speculative::types::Rng (via super or pub use)
```

### `dd_tree.rs` (~80 lines)
```rust
// build_dd_tree, build_dd_tree_pruned, extract_parent_tokens, extract_best_path
// Uses: types::{TreeNode, ConstraintPruner, NoPruner}
```

### `dflash.rs` (~110 lines)
```rust
// dflash_predict, dflash_predict_parallel, dflash_predict_ar
// Uses: transformer, types, sampling
```

### `verifier.rs` (~200 lines)
```rust
// SpeculativeVerifier, SimulatedVerifier, LeviathanVerifier
// Uses: transformer, types, dd_tree, dflash, sampling
```

### `step.rs` (~25 lines)
```rust
// speculative_step_verifier, speculative_step
// Uses: verifier, transformer, types
```

### `sudoku_pruner.rs` (~90 lines)
```rust
// SudokuPruner — behind #[cfg(feature = "sudoku")]
// Uses: types::ConstraintPruner, crate::percepta::Sudoku9x9
```

### `mod.rs` (~40 lines)
```rust
// pub mod types; pub use types::*;
// pub mod sampling; pub use sampling::*;
// ... etc, re-export everything
// #[cfg(feature = "sudoku")] pub mod sudoku_pruner;
```

## What We Are NOT Doing

| Action | Reason |
|--------|--------|
| Moving percepta.rs | Already separate, no change needed |
| Changing public API | All re-exports preserve existing import paths |
| Moving integration tests | They use `speculative::` which still works |
| Renaming anything | Pure structural refactor, no renaming |

## Expected Outcome

- `speculative.rs` (1591 lines) → 8 focused files, largest ~200 lines
- Core speculative decoding has **zero** domain dependencies
- `SudokuPruner` behind `sudoku` feature — can be excluded
- All existing code compiles without import changes (re-exports)
- All tests pass with same counts (80 integration, unit tests split per module)
- Benchmark numbers identical (no code logic changed)

## Files to Modify

| File | Action |
|------|--------|
| `src/speculative.rs` | Delete (replaced by directory) |
| `src/speculative/mod.rs` | New — re-exports |
| `src/speculative/types.rs` | New — TreeNode, DraftResult, ConstraintPruner, NoPruner |
| `src/speculative/sampling.rs` | New — sampling functions |
| `src/speculative/dd_tree.rs` | New — DDTree build + path extraction |
| `src/speculative/dflash.rs` | New — DFlash predict functions |
| `src/speculative/verifier.rs` | New — SpeculativeVerifier trait + impls |
| `src/speculative/step.rs` | New — speculative_step wrappers |
| `src/speculative/sudoku_pruner.rs` | New — SudokuPruner (sudoku feature) |
| `Cargo.toml` | Add `sudoku = []` feature |
| `src/benchmark.rs` | Possibly update imports |
| `examples/sudoku_speculative.rs` | Possibly add feature gate |