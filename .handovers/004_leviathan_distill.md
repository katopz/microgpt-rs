# Handover 004: Leviathan Distillation — SpeculativeVerifier Trait + AR Drafting + Algorithm 1

## What Happened

Distilled the performance-positive parts of "Fast Inference from Transformers via Speculative Decoding" (Leviathan et al., 2022, Algorithm 1) into the existing speculative decoding pipeline. Added a `SpeculativeVerifier` trait for swappable verification strategies, autoregressive DFlash drafting, bonus token, and a full Leviathan Algorithm 1 implementation behind a `leviathan` feature flag.

## Where Is the Plan/Code/Test

- **Plan**: `.plans/004_leviathan_distill.md`
- **Code changed**:
  - `src/speculative.rs` — `SpeculativeVerifier` trait, `SimulatedVerifier`, `LeviathanVerifier` (full impl), `dflash_predict_ar()`, `sample_from_distribution()`, `sample_residual_distribution()`, `extract_best_path()`, updated `speculative_step()`, new `speculative_step_verifier()`
  - `src/benchmark.rs` — `bench_speculative` (uses SimulatedVerifier), `bench_speculative_ar`, `bench_leviathan` (behind feature flag), removed old `run_speculative_step`
  - `Cargo.toml` — added `[features] leviathan = []`
- **Tests**: 173 total with `--all-features` (93 unit + 80 integration). Without leviathan: 169 tests (89 unit + 80 integration)
- **Benchmark**: `bench/010_bench_result.png` (pre-fix), `bench/011_bench_result.png` (5 methods), `bench/012_bench_result.png` (6 methods with Leviathan)

## Benchmark Results

### Default (5 benchmarks)

| Method | Throughput | μs/step | Avg Accept |
|--------|-----------|---------|------------|
| Transformer AR | 813,714 tok/s | 1.23 | 1.00 |
| DFlash | 3,196,001 tok/s | 2.50 | 8.00 |
| DDTree Build | 321,060 trees/s | 3.11 | — |
| Speculative (Simulated) | 876,517 tok/s | 5.70 | 5.00 |
| Speculative (AR Draft) | 1,250,138 tok/s | 5.60 | 7.00 |

### With `--features leviathan` (6 benchmarks, adds)

| Method | Throughput | μs/step | Avg Accept |
|--------|-----------|---------|------------|
| Leviathan (Algorithm 1) | 107,157 tok/s | 11.00 | 1.18 |

Key finding: Leviathan is 8× slower than simulated at 4× model ratio. Low acceptance rate (1.18/8 = 15%) because random weights produce poorly aligned draft/target distributions.

## Reflection — Struggling / Solved

### Solved
1. **Trait design**: `SpeculativeVerifier::speculate()` handles the full pipeline (draft + verify), not just verification. This is because `SimulatedVerifier` uses DFlash+DDTree while `LeviathanVerifier` uses AR drafting + target model. Different draft strategies require different pipelines.
2. **Bonus token logic**: `SimulatedVerifier` adds bonus from last marginal when all accepted. `LeviathanVerifier` adds bonus from target p(x) at γ. Both return 1 to γ+1 tokens.
3. **Backward compat**: `speculative_step()` kept as wrapper calling `speculative_step_verifier()` with `SimulatedVerifier::new(0.75)`. Existing callers unchanged.
4. **Clippy needless_range_loop**: LeviathanVerifier target scoring loop restructured to avoid indexing — split into initial forward + enumerated loop over draft tokens.
5. **Feature flag gating**: `LeviathanVerifier`, its tests, and benchmark behind `#[cfg(feature = "leviathan")]`. Math helpers (`sample_from_distribution`, `sample_residual_distribution`) always compiled and tested.

### Key Insight
The paper's Algorithm 1 is mathematically elegant but requires large model asymmetry (>8× cost ratio) to be a net win. At our 4× ratio, the target model verification cost dominates. After LoRA fine-tuning the draft model for better alignment with the target, acceptance rates should improve and real verification becomes viable.

## What Was Done

### `src/speculative.rs`
- `SpeculativeVerifier` trait with `speculate()` method
- `SimulatedVerifier` — DFlash + DDTree + simulated acceptance cap + bonus token
- `LeviathanVerifier` (behind `leviathan` feature) — AR draft + target p/q scoring + rejection sampling + residual distribution + bonus from target
- `dflash_predict_ar()` — autoregressive variant that samples and feeds back tokens (returns `DraftResult { marginals, sampled_tokens }`)
- `sample_from_distribution()` — CDF-based sampling
- `sample_residual_distribution()` — Equation 3 from paper: normalize(max(0, p−q))
- `extract_best_path()` — extracts highest-scored token at each DDTree depth
- `speculative_step_verifier()` — takes `&mut dyn SpeculativeVerifier`
- `speculative_step()` — backward-compat wrapper with SimulatedVerifier(0.75)
- 12 new tests (8 always, 4 behind leviathan feature)

### `src/benchmark.rs`
- `bench_speculative` — uses SimulatedVerifier via `speculative_step_verifier`
- `bench_speculative_ar` — AR draft + DDTree + simulated acceptance + bonus token
- `bench_leviathan` (behind feature flag) — full Algorithm 1
- Removed old `run_speculative_step` (logic moved into SimulatedVerifier)

### `Cargo.toml`
- Added `[features] default = []` and `leviathan = []`

### `README.md`
- Updated benchmark table with 6 methods
- Added "Speculative Decoding: Distilled from Leviathan et al. 2022" section explaining Algorithm 1
- Added verifier comparison table
- Added cost breakdown explaining why Leviathan is slow at 4× ratio
- Added SpeculativeVerifier architecture section
- Updated references with direct arxiv link
- Updated build commands and test counts

## Remain Work
1. **LoRA fine-tuning** — Train draft model to improve alignment with target → higher acceptance rate → Leviathan becomes viable
2. **Free Embedding Bridge** — Project pre-LM-head hidden states to 2D to query `KVCache2D` with actual transformer data
3. **Scale to actual LLM tokens** — Map Sudoku digits (1–9) to real vocabulary indices via tokenizer
4. **Streaming with print flush** — Switch from `format_events()` batch to callback-based real-time output
5. **Larger model configs** — Test Leviathan at 8× or 16× model ratios to validate the paper's claims

## Issues Ref
- No new issues created

## How to Dev/Test
```bash
# Run all tests (173 with leviathan, 169 without)
cargo test --quiet --workspace --all-features

# Run benchmark (5 methods, default)
cargo run --quiet --release

# Run benchmark with Leviathan Algorithm 1 (6 methods)
cargo run --quiet --release --features leviathan

# Clippy
cargo clippy --all-targets --all-features

# Specific test
cargo test --quiet --lib -- test_leviathan_verifier_returns_at_least_one --features leviathan
```

## Plan Status
| Plan | Status | Tasks |
|------|--------|-------|
| Plan 001: Sudoku 9×9 Example | ✅ Complete | 7/7 tasks |
| Plan 002: Dynamic Depth-Aware Pruning | ✅ Complete | 7/7 tasks |
| Plan 003: Perf Optimization | ✅ Complete | 9/9 tasks |
| Plan 004: Leviathan Distillation | ✅ Complete | 12/12 tasks |