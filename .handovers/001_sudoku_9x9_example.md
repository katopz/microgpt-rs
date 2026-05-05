# Handover 001: 9×9 Sudoku Example with Streaming Thinking

## What Happened

Implemented a complete 9×9 Sudoku solver with Percepta-style "streaming thinking" output, demonstrating the **Computable LoRA** concept from the Gemini PoC. The work distills the Gemini proposal into our existing `KVCache2D` architecture and creates a runnable example that matches the web demo experience.

The Gemini PoC showed a mock `SudokuState` + `SpeculativeSudokuDrafter` with hardcoded logits. We aligned it with our real implementation:
- Our `Sudoku9x9` replaces Gemini's `SudokuState` (production-quality, 9×9 instead of mock)
- Our `ComputableLora::prune_drafts` replaces Gemini's inline validation loop
- Our `StreamingSolver` + `SolveEvent` enum provides the "LLM thinking" output
- Our `KVCache2D::fast_attention` provides the O(log N) state retrieval (Gemini didn't have this)

## Where is the Plan/Code/Test

- **Plan**: `.plans/001_sudoku_9x9_example.md` — 5 tasks, all complete
- **Code**:
  - `src/percepta.rs` — Added `Sudoku9x9`, `ComputableLora`, `SolveEvent`, `StreamingSolver` (public API, ~380 lines)
  - `examples/sudoku_9x9.rs` — Runnable example with Computable LoRA demo + streaming solve
- **Tests**: `tests/integration.rs` — 9 new integration tests:
  - `test_sudoku9x9_arto_inkala_clues`
  - `test_sudoku9x9_is_valid_move`
  - `test_sudoku9x9_display_format`
  - `test_sudoku9x9_solve_arto_inkala`
  - `test_sudoku9x9_solve_hull_compression`
  - `test_computable_lora_prune_drafts`
  - `test_computable_lora_prune_all_invalid`
  - `test_streaming_solver_arto_inkala`
  - `test_sudoku9x9_next_empty`
- **Commit**: `097fd48` on `main`

## Reflection: Struggling / Solved

1. **Display format mismatch**: The `test_sudoku9x9_display_format` test expected `"8 . . . . . . . ."` but the actual format has `| ` separators at column boundaries. Fixed by updating assertion to `"8 . . | . . . | . . "`.

2. **Streaming output verbosity**: Initial implementation showed every single event — the Arto Inkala puzzle produces ~49,559 trace entries with thousands of accepted/contradiction events. The output was 16,000+ bytes of noise. Solved by switching from "filter individual events" to "select key moments": first 4 placements, evenly spaced middle (~11), last 5. This produces a clean ~25-line summary that matches the web demo feel.

3. **Type mismatch in tuple**: `accepted_events` tuple used `(usize, usize, usize, u8, usize)` but the destructure expected `filled` as `usize` (it's `usize` from `SolveEvent::Accepted`). Fixed by correcting tuple type to `(usize, usize, u8, usize, usize)`.

4. **Unused variable warnings**: `total_accepted` and `backtrack_events` were collected but never used in `format_events`. Removed them to keep clippy clean.

## Results

Running `cargo run --example sudoku_9x9` produces:
- Computable LoRA intercept demo (LLM proposes 5 digits, rules engine prunes to 1)
- Streaming "thinking" output with ~25 key moments
- Arto Inkala puzzle solved: **49,559 steps, 7 hull vertices, 7,079.9x compression**
- O(49,559) → O(log 7) ≈ O(3) attention speedup
- Linear and fast attention scores match perfectly

## Remain Work

1. **Wire into `speculative.rs`**: The `ComputableLora::prune_drafts` is currently standalone. Next step is to integrate it into `build_dd_tree` as a branch pruning step — the DDTree drafts branches, and the rules engine prunes invalid ones before target verification.

2. **Free Embedding Bridge**: Project pre-LM-head hidden states to 2D to query the `KVCache2D` using actual transformer data. Currently the example uses `Vec2::new(1.0, 0.0)` as a query.

3. **Scale to actual LLM tokens**: The current example maps Sudoku digits (1-9) to tokens. For a real LLM, we'd need a tokenizer that maps digit tokens to vocabulary indices.

4. **Streaming with actual print flush**: The current `format_events()` collects all events first then formats. For real-time streaming, we'd want a callback-based approach that prints + flushes as events occur.

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