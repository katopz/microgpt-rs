# Plan 002: Dynamic Depth-Aware Pruning

## Goal
Fix the critical gap in `SudokuPruner`: currently it validates each depth independently
against the **initial** board. It doesn't know what tokens were placed at earlier depths
in the same path, so cross-depth conflicts can slip through.

Example of the bug:
- Depth 0 places digit `4` at cell (0,1) — valid against initial board
- Depth 1 tries digit `4` at cell (0,2) — also valid against initial board
- But row 0 now has TWO `4`s → invalid accumulated state → pruner should reject

## Background
- `ConstraintPruner::is_valid(depth, token_idx)` has no path context
- `TreeNode::parent_path: u64` encodes the ancestor path (5 bits per depth)
- `build_dd_tree_pruned` has access to parent nodes when expanding children
- Sudoku validation only needs to check row/col/box against parent tokens, not full board

## Design: Incremental Path Validation

Instead of creating a full board copy per validation call, we do O(lookahead) checks:

```
is_valid(depth=2, token_idx=4, parent_tokens=[1, 3]):
  1. Check digit 4 against initial board at position[2] ← already works
  2. Check digit 4 conflicts with parent_tokens[0] at position[0]? (same row/col/box?)
  3. Check digit 4 conflicts with parent_tokens[1] at position[1]? (same row/col/box?)
  → Only valid if no conflict with initial board AND no conflict with parent tokens
```

This is O(parent_tokens.len()) per check, which is O(lookahead) — negligible.

## Tasks

- [x] **T1: Extend `ConstraintPruner` trait with parent path context**
  - Change signature: `is_valid(&self, depth, token_idx, parent_tokens: &[usize]) -> bool`
  - `parent_tokens[i]` = token placed at depth `i` in this path
  - `NoPruner` ignores parent_tokens (always true)
  - Update all existing call sites in `build_dd_tree_pruned`

- [x] **T2: Implement path-aware validation in `SudokuPruner`**
  - Add method: `conflicts_with_parent(depth, digit, parent_tokens) -> bool`
  - For each parent token, check if it's in the same row/col/box as current position
  - If any parent token with same digit shares row/col/box → conflict → prune
  - `is_valid` returns `false` if initial board check fails OR parent conflict found

- [x] **T3: Extract parent tokens from DDTree `parent_path` bitfield**
  - Add helper: `extract_parent_path(parent_path: u64, depth: usize) -> Vec<usize>`
  - `parent_path` uses 5 bits per depth: `(path << 5) | token_idx`
  - Extract tokens for depths 0..depth by shifting and masking
  - Max depths: 64/5 = 12 (sufficient for lookahead of 5-8)

- [x] **T4: Update `build_dd_tree_pruned` to pass parent tokens**
  - When expanding children at `next_depth`, extract parent path from `best.parent_path`
  - Pass `parent_tokens` to `pruner.is_valid(next_depth, i, &parent_tokens)`
  - Root level (depth 0): `parent_tokens` is empty slice

- [x] **T5: Add path-aware tests — prove cross-depth conflicts are caught**
  - Test: depth 0 places digit X, depth 1 same digit same row → pruned
  - Test: depth 0 places digit X, depth 1 same digit different box → NOT pruned
  - Test: multi-level path with conflict at depth 3 → pruned
  - Test: tree with path-aware pruner is smaller than tree with static pruner
  - Test: all nodes in path-aware tree are valid against accumulated state

- [x] **T6: Update `examples/sudoku_speculative.rs` with path-aware comparison**
  - Add 3rd column: Static Pruner vs Path-Aware Pruner
  - Show additional branches pruned by path awareness
  - Demonstrate that path-aware pruner catches cross-depth conflicts

- [x] **T7: Verify, benchmark, and commit**
  - All tests pass (existing + new)
  - Clippy clean
  - Run both examples, verify output
  - Commit: `feat: path-aware constraint pruning for dd-tree`

## Constraints
- `ConstraintPruner` must remain `Send + Sync` (used in rayon contexts)
- No full board copy per validation — incremental checks only
- `parent_path` bitfield supports max 12 depths (64 bits / 5 bits per depth)
- Backwards compatible: `NoPruner` ignores parent_tokens

## Expected Results
- Path-aware pruner catches cross-depth row/col/box conflicts
- Tree with path-aware pruning ≤ tree with static pruning
- 100% valid placements against accumulated state (not just initial board)
- Demonstrates the full Deterministic Validator promise: LLM drafts, rules engine validates