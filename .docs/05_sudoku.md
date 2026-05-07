# mini-dllm: Sudoku Domain — Solver, Constraint Pruning, TUI

## Overview
Sudoku serves as the proof-of-concept domain for Computable LoRA — demonstrating that a deterministic rules engine can validate LLM-drafted tokens inside the speculative decoding loop. The 9×9 solver with streaming output matches the web-demo experience from the Percepta blog.

## Sudoku9x9 (`percepta.rs`)

```rust
pub struct Sudoku9x9 {
    pub grid: [[u8; 9]; 9],
}
```

### Key Methods
- `is_valid_move(row, col, digit) -> bool` — row/col/3×3 box checks
- `solve(&mut self, cache: &mut KVCache2D, step: &mut usize) -> bool` — backtracking solver
- `is_solved() -> bool`
- `display() -> String` — pretty-printed grid with box separators

### Reference Puzzle (Arto Inkala, 21 clues)
```
8 . . | . . . | . . .
. . 3 | 6 . . | . . .
. 7 . | . 9 . | 2 . .
------+-------+------
. 5 . | . . 7 | . . .
. . . | . 4 5 | 7 . .
. . . | 1 . . | . 3 .
------+-------+------
. . 1 | . . . | . 6 8
. . 8 | 5 . . | . 1 .
. 9 . | . . . | 4 . .
```

### Results
- 49,559 backtracking steps to solve
- 7 hull vertices → 7079.9× compression via convex hull
- O(49559) → O(log 7) ≈ O(3) attention speedup

## StreamingSolver (`percepta.rs`)

Emits `SolveEvent`s during solve for real-time visualization:

```rust
pub enum SolveEvent {
    Try { row: usize, col: usize, digit: u8, depth: usize },
    Accepted { row: usize, col: usize, digit: u8, filled: usize },
    Contradiction { row: usize, col: usize, digit: u8, depth: usize },
    Backtrack { row: usize, col: usize, depth: usize },
    Solved { steps: usize, hull_size: usize, total_trace: usize },
}
```

- `format_events()` produces concise web-demo-style output
- Shows first 4 placements, evenly spaced middle, last 5

## SudokuPruner (`speculative/sudoku_pruner.rs`, behind "sudoku" feature)

Implements `ConstraintPruner` for DDTree branch validation:

### Static Pruning (v1)
- Checks each token against the **initial** board state
- Maps DDTree depth → (row, col) position, validates digit against row/col/box
- Result: 52% of branches valid → prunes 48% invalid

### Path-Aware Pruning (v2)
- Extended signature: `is_valid(depth, token_idx, parent_tokens: &[usize])`
- Checks digit against initial board AND all parent tokens in the path
- Catches cross-depth conflicts: depth 0 places `4` at (0,1), depth 1 tries `4` at (0,2) → pruned
- Result: **100% valid branches** (all cross-depth conflicts eliminated)

```rust
pub struct SudokuPruner {
    initial_grid: [[u8; 9]; 9],
    cell_order: Vec<(usize, usize)>,  // depth → (row, col) mapping
}
```

- `conflicts_with_parent(depth, digit, parent_tokens) -> bool`
- Incremental: O(parent_tokens.len()) per check = O(lookahead) — negligible

## DDTree Integration

`build_dd_tree_pruned(marginals, config, pruner)` in `dd_tree.rs`:
- Before adding children to heap, filters through `pruner.is_valid(depth, token, parent_tokens)`
- Invalid branches never enter the tree → saves verification budget
- Works with both chain-seed and REST merge

### Pruning Results
| Metric | No Pruner | Static Pruner | Path-Aware Pruner |
|--------|-----------|---------------|-------------------|
| Valid branches | 52% | ~80% | **100%** |
| Invalid pruned | 0 | ~20% | **48%** |
| Cross-depth conflicts | Not caught | Not caught | **All caught** |

## Examples

### `sudoku_9x9.rs`
- Solves Arto Inkala puzzle with streaming "thinking" output
- Shows hull compression stats and O(log N) attention
- No feature flag required (Sudoku9x9 is always public)

### `sudoku_speculative.rs` (behind "sudoku" feature)
- Simulated draft model marginals (uniform over valid digits)
- Compares DDTree: without vs with Computable LoRA pruning
- Token distribution table shows which digits pruned per depth

### `sudoku_tui.rs` (behind "sudoku" feature)
- Ratatui-based TUI with real-time solver visualization
- Color-coded cells: Green (clue), Cyan (accepted), Yellow (trying), Red (contradiction)
- Two tabs: 9×9 solver vs Speculative mode
- Channel-based: `mpsc` for streaming events from solver thread
- Stats bar: tok/s, tokens, lines/s

## Design Lessons
1. **ConstraintPruner is domain-agnostic** — same trait serves Sudoku and Rust AST (SynPruner)
2. **Path context is essential** — static pruning misses cross-depth conflicts
3. **Incremental validation is fast** — O(lookahead) per check, no board copies needed
4. **Computable LoRA promise delivered** — LLM drafts, deterministic rules validate, 100% valid outputs