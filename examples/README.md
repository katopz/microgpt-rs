# Examples

## sudoku_9x9

Streaming "Thinking" Sudoku solver demonstrating the Computable LoRA concept:
- Deterministic rules engine prunes LLM hallucinations
- O(log N) attention retrieves execution state via convex hull
- Streaming output shows step-by-step constraint satisfaction

```bash
cargo run --example sudoku_9x9 --features sudoku
```

## sudoku_speculative

DDTree + Computable LoRA pruning with 3-level comparison:
- **Unpruned**: Draft model proposes all high-probability tokens
- **Static-Only**: Prunes against initial board, ignores cross-depth conflicts
- **Path-Aware**: Prunes against initial board AND parent tokens in same path

Shows that path-aware pruning catches cross-depth row/col/box conflicts that static-only pruning misses.

```bash
cargo run --example sudoku_speculative --features sudoku
```

## sudoku_tui

Ratatui TUI visualization of the Sudoku solver with real-time grid display and speculative mode comparison:
- Color-coded Sudoku grid showing constraint satisfaction in real time
- Step/trace panels for live solver progress
- Speculative mode comparison side-by-side
- Uses `ratatui` + `crossterm` for terminal rendering

```bash
cargo run --example sudoku_tui --features sudoku
```

## validator_demo

Constraint validator pipeline demonstrating syntax-aware token pruning:
- BPE tokenize Rust source code
- Draft model proposes tokens
- `SynPruner` validates partial Rust syntax
- Only syntactically valid branches are explored
- Uses `syn` for real Rust parsing

```bash
cargo run --example validator_demo --features validator
```

## Feature Flags

| Flag | Gates |
|------|-------|
| `sudoku` | `SudokuPruner`, sudoku examples, sudoku-specific tests |
| `leviathan` | `LeviathanVerifier`, real p/q rejection sampling (Algorithm 1) |
| `validator` | `SynPruner`, `syn`-based syntax validation, validator examples |
| `rest` | REST API client via `reqwest` + `tokio` runtime |
| `gpu` | GPU compute via `wgpu`, `safetensors` model loading |
| `full` | All of the above |

```bash
# Run with all features
cargo run --example sudoku_9x9 --features "sudoku,leviathan"