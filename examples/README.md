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

## Feature Flags

| Flag | Gates |
|------|-------|
| `sudoku` | `SudokuPruner`, sudoku examples, sudoku-specific tests |
| `leviathan` | `LeviathanVerifier`, real p/q rejection sampling (Algorithm 1) |

```bash
# Run with all features
cargo run --example sudoku_9x9 --features "sudoku,leviathan"