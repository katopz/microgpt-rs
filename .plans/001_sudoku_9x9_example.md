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

- [x] **T5: Verify and commit**
  - `cargo test --quiet` passes all 138 tests (58 unit + 80 integration)
  - `cargo run --example sudoku_9x9` produces streaming output
  - `cargo clippy --quiet` clean
  - Commit with message: `feat: 9x9 sudoku example with streaming thinking output`

## Results
- Arto Inkala puzzle solved: 49,559 steps, 7 hull vertices, 7079.9x compression
- O(49559) → O(log 7) ≈ O(3) attention speedup
- Linear and fast attention scores match perfectly

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
