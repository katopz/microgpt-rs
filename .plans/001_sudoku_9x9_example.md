# Plan 001: 9×9 Sudoku Example with Streaming Thinking

## Goal
Create a runnable `examples/sudoku_9x9.rs` that demonstrates the Computable LoRA concept
by solving a 9×9 Sudoku puzzle with streaming "thinking" output — matching the web demo
experience shown in the Percepta blog.

## Background
- We already have a 4×4 Sudoku solver in `src/percepta.rs` (test-only, L1193-1389)
- We have `KVCache2D` with O(log N) attention for execution trace retrieval
- We have `speculative.rs` with DDTree for branch drafting
- The Gemini PoC shows the concept but is a toy — we need a real implementation

## Tasks

- [x] **T1: Add 9×9 Sudoku solver to `src/percepta.rs`**
  - Create `Sudoku9x9` struct with `grid: [[u8; 9]; 9]`
  - Implement `is_valid_move(row, col, digit) -> bool` (row/col/3x3 box checks)
  - Implement `solve(&mut self, cache: &mut KVCache2D, step: &mut usize) -> bool`
  - Implement `is_solved() -> bool`
  - Implement `display() -> String` for pretty-printing
  - Make it `pub` so examples can use it
  - Add unit tests in `tests/integration.rs`

- [x] **T2: Add Computable LoRA intercept struct**
  - `ComputableLora` struct that wraps constraint validation
  - Method `prune_drafts(state, row, col, logits) -> Vec<(u8, f32)>`
  - Returns only valid (digit, prob) pairs — invalid ones removed
  - This is the bridge between LLM drafting and deterministic rules

- [x] **T3: Create `examples/sudoku_9x9.rs`**
  - Load the Arto Inkala puzzle (21 clues)
  - Show initial board with clue count
  - Stream "thinking" output during solve (concise ~25 lines)
  - Show final solved board
  - Show hull compression stats (hull size vs total trace)
  - Show O(log N) attention retrieval of final state

- [x] **T4: Add `StreamingSolver` with event callbacks**
  - Struct that wraps the solver and emits events:
    - `Try { row, col, digit, depth }`
    - `Accepted { row, col, digit, filled }`
    - `Contradiction { row, col, digit, depth }`
    - `Backtrack { row, col, depth }`
    - `Solved { steps, hull_size, total_trace }`
  - `format_events()` produces concise web-demo-style output
  - Shows first 4 placements, evenly spaced middle, last 5

- [x] **T5: Verify example and tests**
  - `cargo test --quiet` passes all 138 tests (58 unit + 80 integration)
  - `cargo run --example sudoku_9x9` produces streaming output
  - `cargo clippy --quiet` clean
  - Commit: `097fd48 feat: 9x9 sudoku example with streaming thinking output`

- [x] **T6: Wire `ComputableLora` into `speculative.rs` DDTree**
  - Add `ConstraintPruner` trait to `speculative.rs` (`Send + Sync`)
  - Implement `NoPruner` (identity) and `SudokuPruner` (row/col/box rules)
  - `SudokuPruner` maps DDTree depth → (row, col), validates digits 1-9
  - Add `build_dd_tree_pruned(marginals, config, pruner)` function
  - Before adding children to heap, filter through `pruner.is_valid(depth, token)`
  - Invalid branches never enter the tree → saves verification budget
  - Refactored `build_dd_tree` to delegate to `build_dd_tree_pruned` with `NoPruner`
  - 10 new tests: NoPruner, SudokuPruner, pruned tree size, valid-only guarantee

- [x] **T7: End-to-end Sudoku speculative decoding example**
  - Create `examples/sudoku_speculative.rs`
  - Simulated draft model marginals (uniform over valid digits)
  - DDTree comparison: without vs with Computable LoRA pruning
  - Results: **52% valid unpruned → 100% valid pruned** (48 invalid branches eliminated)
  - Token distribution table shows exactly which digits were pruned per depth
  - Commit: `feat: constraint pruner for dd-tree speculative decoding`

## Results (so far)
- Arto Inkala puzzle solved: 49,559 steps, 7 hull vertices, 7079.9x compression
- O(49559) → O(log 7) ≈ O(3) attention speedup
- Linear and fast attention scores match perfectly
- Streaming output matches web demo style
- **DDTree pruning: 52% valid → 100% valid branches** (48 invalid eliminated)
- Computable LoRA guarantees mathematically valid placements
- 148 tests passing (68 unit + 80 integration), zero clippy warnings
</newtext>

## Constraints
- Keep solver logic in `src/percepta.rs` (pub) — examples call into it
- No external dependencies — pure Rust
- Streaming output uses concise summary (not every single step)
- The Arto Inkala puzzle (21 clues) is our reference hard puzzle

## Reference Puzzle (Arto Inkala)
```
8 . . . . . . . .
. . 3 6 . . . . .
. 7 . . 9 . 2 . .
. 5 . . . 7 . . .
. . . . 4 5 7 . .
. . . 1 . . . 3 .
. . 1 . . . . 6 8
. . 8 5 . . . 1 .
. 9 . . . . 4 . .
```
