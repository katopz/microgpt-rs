# Plan 006: Sudoku TUI Example

## Overview
Create a Ratatui-based TUI example that visualizes the Sudoku solver in real-time,
showing the grid filling step-by-step as the solver explores and backtracks.

## Layout
```
─────────────────────────────────────────────────
│ [Tab:9x9] [Tab:Speculative] [R:Restart] [Q:Quit]│
─────────────────────────────────────────────────
│  8 . . | . . . | . . .                        │
│  . . 3 | 6 . . | . . .                        │
│  . 7 . | . 9 . | 2 . .                        │
│  ------+-------+------                         │
│  . 5 . | . . 7 | . . .                        │
│  . . . | . 4 5 | 7 . .                        │
│  . . . | 1 . . | . 3 .                        │
│  ------+-------+------                         │
│  . . 1 | . . . | . 6 8                        │
│  . . 8 | 5 . . | . 1 .                        │
│  . 9 . | . . . | 4 . .                        │
─────────────────────────────────────────────────
│ 32,852 tok/s | 3,072,958 tokens | 7,236 l/s   │
─────────────────────────────────────────────────
│            │                                    │
│  Panel A   │   Panel B                          │
│  (steps)   │   (trace)                          │
│            │                                    │
─────────────────────────────────────────────────
```

## Architecture
- Use `std::sync::mpsc` channels to stream `SolveEvent`s from solver thread to TUI
- Solver runs in `std::thread::spawn`, sends events with small `thread::sleep` delay
- TUI main loop: receive events → update state → render frame
- Auto-scroll both panels to bottom on each new line

## Dependencies (add to Cargo.toml under `[dev-dependencies]`)
- `ratatui` — TUI framework
- `crossterm` — terminal backend (ratatui dependency)

## Files to Create
- `examples/sudoku_tui.rs` — single-file TUI example (~500 lines)

## Tasks
- [x] T1: Add `ratatui` + `crossterm` to `[dev-dependencies]` in Cargo.toml
- [x] T2: Create `examples/sudoku_tui.rs` with TUI scaffold
- [x] T3: Implement channel-based solver thread for 9x9 mode
- [x] T4: Implement Sudoku grid widget with color-coded cells
  - Green: clue (given)
  - Cyan: solver-placed (accepted)
  - Yellow: currently trying
  - Red: contradiction (briefly)
- [x] T5: Implement Panel A — human-readable step messages
- [x] T6: Implement Panel B — raw trace data (commit lines)
- [x] T7: Implement stats bar (tok/s, tokens, lines/s)
- [x] T8: Implement tabs (9x9 vs Speculative) with R to restart
- [x] T9: Implement speculative mode visualization
- [x] T10: Add example to Cargo.toml with `required-features = ["sudoku"]`
- [x] T11: Test and verify both modes work

## Key Design Decisions
1. **Single file** — keep it self-contained like other examples
2. **Channel-based** — `mpsc` for streaming events, no async runtime needed
3. **Color coding** — visual feedback for solver state changes
4. **Tab switching** — re-runs solver on tab switch with restart
5. **Configurable speed** — solver delay controllable for visual clarity

## Integration Notes
- Reuses `SolveEvent` enum from `percepta.rs`
- Reuses `Sudoku9x9`, `StreamingSolver` from `percepta.rs`
- Speculative mode reuses `SudokuPruner`, `build_dd_tree_pruned` from `speculative`
- No changes to library code — purely a new example
