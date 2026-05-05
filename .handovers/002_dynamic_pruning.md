# Handover 002: Dynamic Depth-Aware Pruning

## What Happened

Implemented path-aware constraint pruning for the DDTree speculative decoding pipeline. The critical gap was that `SudokuPruner` validated each depth independently against the **initial** board — it didn't know what tokens were placed at earlier depths in the same path. Cross-depth row/col/box conflicts could slip through.

**Before**: Static-only pruning caught 48% invalid branches (against initial board), but missed 16% cross-depth conflicts (same digit in same row/col/box as a parent token in the path).

**After**: Path-aware pruning catches ALL conflicts — 100% accumulated validity guaranteed.

## Where Is the Plan/Code/Test

- **Plan**: `.plans/002_dynamic_pruning.md` — All 7 tasks completed ✅
- **Code**: `src/speculative.rs` — Trait extended, path-aware validation, `extract_parent_tokens` helper
- **Example**: `examples/sudoku_speculative.rs` — 3-column comparison (Unpruned / Static-Only / Path-Aware)
- **Tests**: `src/speculative.rs` `mod tests` — 9 new path-aware tests

### Key Changes

1. **`ConstraintPruner::is_valid` signature extended**:
   - Before: `is_valid(&self, depth, token_idx) -> bool`
   - After: `is_valid(&self, depth, token_idx, parent_tokens: &[usize]) -> bool`
   - `NoPruner` ignores `parent_tokens` (backwards compatible)

2. **`SudokuPruner::is_valid` now checks cross-depth conflicts**:
   - For each parent token with the SAME digit, checks row/col/box overlap
   - Incremental O(parent_tokens.len()) check — no board copy needed

3. **`extract_parent_tokens(parent_path: u64, num_tokens: usize) -> Vec<usize>`**:
   - Decodes the `parent_path` bitfield (5 bits per depth, most-recent in lowest bits)
   - Returns tokens ordered by depth: `result[k]` = token at depth `k`

4. **`build_dd_tree_pruned` extracts and passes parent tokens**:
   - At depth 0: `parent_tokens = &[]`
   - At deeper depths: extracted from `best.parent_path` bitfield

5. **`SudokuPruner::board()` getter added** for external accumulated-board checks

### Example Output (Arto Inkala, 8 depths, budget=100)

```
Unpruned:    100 nodes, 46 accumulated-valid (46.0%)
Static-Only: 100 nodes, 84 accumulated-valid (84.0%)
Path-Aware:  100 nodes, 100 accumulated-valid (100.0%)

Path awareness caught 16 cross-depth conflicts that static-only missed
Example: depth 1 places digit 6 at (1,3), but depth 0 already placed digit 6 at (1,2)
→ Same row 1!
```

## Reflection: Struggling / Solved

1. **`extract_parent_tokens` bit order was wrong** — The `parent_path` bitfield packs most-recent token in lowest bits (because we shift left then OR). Initial implementation extracted from low bits first, yielding reversed order. Fixed by using `(num_tokens - 1 - k) * 5` shift instead of `k * 5`.

2. **`count_invalid_accumulated` off-by-one** — `parent_path` includes the node's OWN token. To get parent-only tokens, must extract `depth + 1` tokens then take `&[..depth]`.

3. **Test digit selection** — Some test digits were invalid against the initial board (e.g., digit 1 at position (0,2) — column 2 already has 1 from the initial board). Fixed by choosing digits that are valid at both positions individually.

4. **Private field access** — `SudokuPruner::board` was private. Added `board()` getter method instead of making the field public.

## Remain Work

- **Free Embedding Bridge** — Project pre-LM-head hidden states to 2D for `KVCache2D` queries
- **Scale to actual LLM tokens** — Map Sudoku digits (1–9) to real vocabulary indices via tokenizer
- **Streaming with print flush** — Switch from `format_events()` batch to callback-based real-time output
- **Integration test coverage** — Path-aware tests are currently in `src/speculative.rs` unit tests; could add integration tests to `tests/integration.rs`

## Issues Ref

- Plan 001 established the static pruning that Plan 002 improves upon
- No standalone issues created; the gap was identified during Plan 001's T7 analysis

## How to Dev/Test

```bash
# Run all tests (157 total: 77 unit + 80 integration)
cargo test --quiet --workspace

# Run path-aware tests specifically
cargo test --quiet --lib -- speculative::tests::test_sudoku_pruner_path_aware
cargo test --quiet --lib -- speculative::tests::test_ddtree_path_aware
cargo test --quiet --lib -- speculative::tests::test_extract_parent_tokens

# Run speculative example (3-column comparison)
cargo run --example sudoku_speculative

# Run solver example
cargo run --example sudoku_9x9

# Clippy check
cargo clippy --quiet