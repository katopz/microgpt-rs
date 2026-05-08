# 016: Blue Bear Tactical Puzzle TUI Example

## Overview
Create an interactive TUI example that demonstrates using Speculative Decoding (DDTree) with ConstraintPruner as a state-space solver for the Blue Bear tactical puzzle. The example features emoji-based map rendering and step-by-step navigation with Next/Back controls.

Additionally, refactor `sudoku_pruner` from `speculative/` into the new `pruners/` module for consistency, and create a non-TUI test/benchmark example.

## Tasks
- [x] Create `src/pruners/mod.rs` with module exports
- [x] Create `src/pruners/blue_bear_pruner.rs` with GameState, BlueBearPruner, and ConstraintPruner impl
- [x] Move `src/speculative/sudoku_pruner.rs` → `src/pruners/sudoku_pruner.rs` (refactor for consistency)
- [x] Update `src/speculative/mod.rs` — remove module, keep backward-compatible re-export via `crate::pruners`
- [x] Update example imports (`sudoku_speculative.rs`, `sudoku_tui.rs`) to use `microgpt_rs::pruners::SudokuPruner`
- [x] Update `src/lib.rs` to include `pub mod pruners;`
- [x] Create `examples/blue_bear.rs` — non-TUI solver/benchmark with emoji output
- [x] Create `examples/blue_bear_tui.rs` with:
  - [x] Emoji map rendering (🐻=Bear, 👹=Monster, 💎=Treasure, 🚪=Goal, 🧱=Wall, ⬜=Floor, 🔑=Item)
  - [x] TUI with ratatui + crossterm
  - [x] Step-by-step solution navigation with Next/Back buttons
  - [x] State info panel (inventory, killed monsters, collected treasures)
  - [x] Action legend (↑↓←→ + Attack)
- [x] Update `Cargo.toml` with example entries (`blue_bear`, `blue_bear_tui`)
- [x] Fix `draft_lookahead` overflow — DDTree packs 16 bits/token into u128, max = 8
- [x] Redesign map to 7-step variant that fits within lookahead=8
- [x] Test all examples and verify all 267 tests pass

## Architecture
```
src/pruners/
  mod.rs                    # pub mod blue_bear_pruner; + #[cfg(sudoku)] pub mod sudoku_pruner
  blue_bear_pruner.rs       # GameState, BlueBearPruner, impl ConstraintPruner
  sudoku_pruner.rs          # SudokuPruner (moved from speculative/)

examples/
  blue_bear.rs              # Non-TUI solver/benchmark with emoji output
  blue_bear_tui.rs          # Interactive TUI with emoji map + step navigation
```

## Map Design (7 steps, fits u128/16=8 token limit)
```
B X T      B=Bear, X=Monster+Treasure, T=Treasure
# M G      #=Wall, M=Monster, G=Goal
```
Solution: → ⚔ ↓ ⚔ ↑ → ↓

## Key Constraint: DDTree Token Packing
DDTree packs tokens at 16 bits each into `u128` → max `draft_lookahead = 8`.
The original 3x3 map needed 10 steps which overflowed. The 2×3 map fits in 7 steps.

## Verification
- ✅ `cargo build` — clean
- ✅ `cargo clippy` — zero warnings
- ✅ All 267 tests pass (including 15 sudoku_pruner tests in new location)
- ✅ `cargo run --example blue_bear` — solves in 7 steps, 269 nodes, ~1.88ms
- ✅ `cargo run --example blue_bear_tui` — interactive TUI works
- ✅ `cargo build --features sudoku` — sudoku examples still work
- ✅ Backward-compatible: `speculative::SudokuPruner` re-exported from `crate::pruners`
