# Handover 001: 9×9 Sudoku Example with Streaming Thinking

## What Happened

Implemented a complete 9×9 Sudoku solver with Percepta-style "streaming thinking" output, demonstrating the **Computable LoRA** concept from the Gemini PoC. The work distills the Gemini proposal into our existing `KVCache2D` architecture and creates a runnable example that matches the web demo experience.

The Gemini PoC showed a mock `SudokuState` + `SpeculativeSudokuDrafter` with hardcoded logits. We aligned it with our real implementation:
- Our `Sudoku9x9` replaces Gemini's `SudokuState` (production-quality, 9×9 instead of mock)
- Our `ComputableLora::prune_drafts` replaces Gemini's inline validation loop
- Our `StreamingSolver` + `SolveEvent` enum provides the "LLM thinking" output
- Our `KVCache2D::fast_attention` provides the O(log N) state retrieval (Gemini didn't have this)

## Where is the Plan/Code/Test

- **Plan**: `.plans/001_sudoku_9x9_example.md` — 7 tasks, all complete
- **Code**:
  - `src/percepta.rs` — Added `Sudoku9x9`, `ComputableLora`, `SolveEvent`, `StreamingSolver` (public API, ~380 lines)
  - `src/speculative.rs` — Added `ConstraintPruner` trait, `NoPruner`, `SudokuPruner`, `build_dd_tree_pruned` (~200 lines)
  - `examples/sudoku_9x9.rs` — Runnable example with Computable LoRA demo + streaming solve
  - `examples/sudoku_speculative.rs` — End-to-end DDTree pruning comparison demo
- **Tests**: 
  - `src/speculative.rs` — 10 new unit tests (pruner behavior, tree size, valid-only guarantee)
  - `tests/integration.rs` — 9 integration tests (solver, display, computable lora, streaming)
- **Commits**: `097fd48`, `5a1116a` on `main`

## Reflection: Struggling / Solved

1. **Display format mismatch**: The `test_sudoku9x9_display_format` test expected `"8 . . . . . . . ."` but the actual format has `| ` separators at column boundaries. Fixed by updating assertion to `"8 . . | . . . | . . "`.

2. **Streaming output verbosity**: Initial implementation showed every single event — the Arto Inkala puzzle produces ~49,559 trace entries with thousands of accepted/contradiction events. The output was 16,000+ bytes of noise. Solved by switching from "filter individual events" to "select key moments": first 4 placements, evenly spaced middle (~11), last 5. This produces a clean ~25-line summary that matches the web demo feel.

3. **Type mismatch in tuple**: `accepted_events` tuple used `(usize, usize, usize, u8, usize)` but the destructure expected `filled` as `usize` (it's `usize` from `SolveEvent::Accepted`). Fixed by correcting tuple type to `(usize, usize, u8, usize, usize)`.

4. **Unused variable warnings**: `total_accepted` and `backtrack_events` were collected but never used in `format_events`. Removed them to keep clippy clean.

5. **DDTree pruning test saturation**: `test_ddtree_pruned_sudoku_reduces_tree_size` initially failed because both pruned and unpruned trees hit the budget limit (100, then 500 nodes). The issue: with uniform marginals and 5 depths × 9 branches = huge candidate space, even 4-valid-per-depth × 5 = 1024 > 500. Fixed by using single-depth marginals (1 depth × 9 branches) with budget=20, so unpruned=9, pruned=4.

6. **Trait method resolution**: `SudokuPruner::is_valid` requires `ConstraintPruner` trait in scope. The example needed `use microgpt_rs::speculative::ConstraintPruner` import.

## Results

### Example 1: `cargo run --example sudoku_9x9`
- Computable LoRA intercept demo (LLM proposes 5 digits, rules engine prunes to 1)
- Streaming "thinking" output with ~25 key moments
- Arto Inkala puzzle solved: **49,559 steps, 7 hull vertices, 7,079.9x compression**
- O(49,559) → O(log 7) ≈ O(3) attention speedup
- Linear and fast attention scores match perfectly

### Example 2: `cargo run --example sudoku_speculative`
- DDTree comparison: without vs with Computable LoRA pruning
- **52% valid unpruned → 100% valid pruned** (48 invalid branches eliminated)
- Token distribution shows exactly which digits were pruned per depth
- Cell (1,2): pruned [3,5,7,8,9], kept [1,2,4,6]
- Cell (1,3): pruned [1,3,7,8], kept [2,4,5,6,9]

## Remain Work

1. **Free Embedding Bridge**: Project pre-LM-head hidden states to 2D to query the `KVCache2D` using actual transformer data. Currently the example uses `Vec2::new(1.0, 0.0)` as a query.

2. **Scale to actual LLM tokens**: The current example maps Sudoku digits (1-9) to tokens. For a real LLM, we'd need a tokenizer that maps digit tokens to vocabulary indices.

3. **Streaming with actual print flush**: The current `format_events()` collects all events first then formats. For real-time streaming, we'd want a callback-based approach that prints + flushes as events occur.

4. **Dynamic pruning across depths**: Current `SudokuPruner` checks each depth independently. For full speculative decoding, the pruner would need to track placements from parent path and validate against accumulated state.

## Issues Ref

No issues created. All tasks completed cleanly.

## How to Dev/Test

```bash
# Run the example
cargo run --example sudoku_9x9

# Run all tests
cargo test --quiet

# Run only 9x9 tests
cargo test --quiet sudoku9x9

# Run only computable_lora tests
cargo test --quiet computable_lora

# Run only streaming tests
cargo test --quiet streaming

# Clippy check
cargo clippy --quiet