# Plan 032: Heuristic Learning Infrastructure

**Branch:** `develop/feature/032_heuristic_learning_infrastructure`
**Depends on:** Plan 030 (bandit feature in microgpt-rs), Plan 021 (ScreeningPruner)
**Research:** `.research/14_Learning_Beyond_Gradients.md`
**Goal:** Add the missing infrastructure for Heuristic Learning (HL) — trial persistence, absorb-compress cycle, hot-swap pruner, and regression suite. This enables the Bomberman arena (Plan 033) and future coding-agent-driven validator evolution.

---

## Overview

The bandit (Plan 030) does online learning without gradients. The screening pruner (Plan 021) provides domain constraints. WASM validators (riir-validator-sdk) provide sandboxed heuristic rules. What's missing:

1. **TrialLog** — persistent episode history (the `trials.jsonl` from the HL article)
2. **Absorb-Compress** — auto-promote stable bandit knowledge into hard constraints
3. **HotSwapPruner** — runtime `.wasm` reload without process restart
4. **RegressionSuite** — replay golden episodes to detect regressions

These four pieces turn the existing bandit + pruner architecture into a complete Heuristic System per the HL paradigm.

### The HL Loop (After This Plan)

```
Episode N:   BanditPruner selects arm → environment runs → reward → TrialLog.append()
             ...
Episode N+k: AbsorbCompress checks: arm 3 has Q=0.02 over 500 visits → promote to hard block
             → BanditPruner delegates to BlockedArmPruner for arm 3
             ...
Round N+m:   Agent writes new validator.rs → compile .wasm → HotSwapPruner.reload()
             → RegressionSuite.replay_golden() → all pass → keep new .wasm
```

---

## Tasks

- [x] **Task 1: TrialLog** (`src/pruners/trial_log.rs`)
  - Struct `TrialRecord { episode, arm, reward, q_value, cumulative_reward, cumulative_regret, config: String, note: String }`
  - `TrialLog::new(path)` — create/append to JSONL file
  - `TrialLog::append(&mut self, record: &TrialRecord)` — serialize and write one line
  - `TrialLog::flush(&mut self)` — ensure buffered writes hit disk
  - `TrialLog::load(path) -> Vec<TrialRecord>` — deserialize JSONL for analysis
  - `TrialLog::summary(&self) -> TrialSummary` — aggregate stats (total episodes, best arm, avg reward, avg regret)
  - Tests: roundtrip write+read, summary aggregation, empty log edge case
  - ~150 lines

- [x] **Task 2: AbsorbCompress** (`src/pruners/absorb_compress.rs`)
  - Trait `AbsorbCompress: ScreeningPruner`
  - `fn absorb(&mut self, arm: usize, reward: f32)` — feed new observation
  - `fn compress(&mut self) -> Vec<usize>` — promote stable low-Q arms to hard blocks, returns promoted arm indices
  - `fn compressed_arms(&self) -> &[usize]` — list of arms promoted to hard constraints
  - `fn should_compress(&self) -> bool` — check if compression threshold met
  - Struct `CompressConfig { min_visits: usize, q_threshold: f32, promote_count: usize }` — tunable thresholds
  - Default: `min_visits=200, q_threshold=0.05, promote_count=3` (arm must have 200+ visits, Q < 0.05, checked every 100 episodes)
  - Tests: no compress under threshold, compress fires at threshold, compressed arms blocked, double-compress idempotent
  - ~200 lines

- [x] **Task 3: HotSwapPruner** (`src/pruners/hot_swap.rs`)
  - Struct `HotSwapPruner { current: WasmPruner, wasm_path: PathBuf, version: u64 }`
  - `HotSwapPruner::new(wasm_path: &Path) -> Result<Self>` — load initial .wasm
  - `HotSwapPruner::reload(&mut self) -> Result<bool>` — reload .wasm from disk, returns true if changed (blake3 hash comparison)
  - `HotSwapPruner::version(&self) -> u64` — current version counter (increments on reload)
  - Implements `ConstraintPruner` and `ScreeningPruner` by delegating to current `WasmPruner`
  - Thread-safe: uses `RwLock` for the inner pruner (read-heavy, write only on reload)
  - Tests: reload same file = no version bump, reload changed file = version bump, pruner works after reload
  - ~180 lines
  - Note: requires `wasm` feature flag (same as WasmPruner)

- [x] **Task 4: RegressionSuite** (`src/pruners/regression.rs`)
  - Struct `GoldenTrace { label: String, actions: Vec<usize>, expected_reward: f32, expected_survival: bool }`
  - Struct `RegressionSuite { traces: Vec<GoldenTrace>, tolerance: f32 }`
  - `RegressionSuite::from_trials(path: &Path, top_n: usize) -> Result<Self>` — extract top-N episodes from TrialLog as golden traces
  - `RegressionSuite::run<F>(&self, pruner_factory: F) -> RegressionResult` — replay all traces through a fresh pruner, check reward ≥ expected
  - `RegressionResult { passed: usize, failed: usize, failures: Vec<RegressionFailure> }`
  - `RegressionFailure { trace_label, expected_reward, actual_reward, delta }`
  - Tests: all-pass suite, tolerance boundary, empty suite
  - ~150 lines

- [x] **Task 5: Integration — BanditPruner + TrialLog + AbsorbCompress** (`src/pruners/bandit.rs` extension)
  - Add `BanditPruner::run_with_trial_log()` method that wraps `BanditSession::run()` but also appends to `TrialLog`
  - Add `BanditPruner::absorb_compress_cycle()` that checks `should_compress()` and calls `compress()` after each episode batch
  - Wire `AbsorbCompress` as a trait bound option for `BanditPruner<P>` where `P: ScreeningPruner + AbsorbCompress`
  - Tests: trial log has correct episode count after run, compress triggers at threshold, compressed arms reflected in Q-values
  - ~100 lines added to existing bandit.rs

- [x] **Task 6: HL Demo** (`examples/hl_01_trial_log.rs`)
  - Uses `BernoulliEnv` (5 arms, one optimal) with `BanditSession`
  - Runs 1000 episodes with `TrialLog` persisting to `/tmp/hl_trial_log.jsonl`
  - After every 100 episodes, runs `AbsorbCompress::compress()`
  - Prints: initial Q-values, compressed arms, final Q-values, trial summary
  - Proves: trial persistence works, absorb-compress promotes bad arms to hard blocks
  - ~200 lines

- [x] **Task 7: HotSwap Demo** (`examples/hl_02_hotswap.rs`)
  - Loads a WASM validator via `HotSwapPruner`
  - Runs 100 episodes with `BanditPruner<HotSwapPruner>` + `TrialLog`
  - Simulates "agent writes new .wasm": copies a different validator to the path
  - Calls `hot_swap.reload()` mid-run, shows version bump
  - Runs `RegressionSuite` against golden traces from first 100 episodes
  - Prints: before/after Q-values, regression results
  - ~250 lines
  - Requires `wasm` feature

- [x] **Task 8: Benchmark — Absorb-Compress Overhead** (`tests/bench_absorb_compress.rs`)
  - Benchmark `BanditPruner::relevance()` with and without `AbsorbCompress`
  - Benchmark `TrialLog::append()` throughput (writes per second)
  - Benchmark `HotSwapPruner::reload()` latency (blake3 hash + wasm load)
  - Target: absorb-compress adds <5% overhead to relevance(), trial log sustains >100K writes/sec, hotswap reload <10ms
  - ~150 lines

- [x] **Task 9: Update docs**
  - Update `microgpt-rs/README.md` with HL section
  - Update `.docs/09_heuristic_learning.md` (Plan 033 dependency)
  - Update `src/pruners/mod.rs` with new module exports

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    HL Infrastructure                      │
│                                                          │
│  ┌─────────────┐    ┌──────────────┐    ┌────────────┐  │
│  │  TrialLog   │    │ AbsorbCompress│   │ Regression │  │
│  │  (JSONL)    │    │ (Q→hard block)│   │ Suite      │  │
│  │             │    │              │    │ (golden)   │  │
│  │ append()    │◄───│ absorb()     │    │ replay()   │  │
│  │ summary()   │    │ compress()   │    │ from_trials│  │
│  └──────┬──────┘    └──────┬───────┘    └─────┬──────┘  │
│         │                  │                   │          │
│         ▼                  ▼                   ▼          │
│  ┌──────────────────────────────────────────────────┐    │
│  │              BanditPruner<P>                      │    │
│  │  P = ScreeningPruner + AbsorbCompress             │    │
│  │  relevance() = domain_score × bandit_bonus        │    │
│  │  update() → TrialLog.append()                     │    │
│  │  compress() → promote low-Q to hard blocks        │    │
│  └──────────────────────┬───────────────────────────┘    │
│                          │                                │
│                          ▼                                │
│  ┌──────────────────────────────────────────────────┐    │
│  │           HotSwapPruner (optional)                │    │
│  │  Wraps WasmPruner with runtime .wasm reload       │    │
│  │  reload() → blake3 check → new WasmPruner         │    │
│  └──────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────┘
```

---

## Module Structure

```text
src/pruners/
├── mod.rs              (add new module exports)
├── bandit.rs           (extend with trial_log + absorb_compress)
├── trial_log.rs        (NEW — TrialRecord, TrialLog, TrialSummary)
├── absorb_compress.rs  (NEW — AbsorbCompress trait, CompressConfig)
├── hot_swap.rs         (NEW — HotSwapPruner, requires wasm feature)
├── regression.rs       (NEW — GoldenTrace, RegressionSuite)
├── ...existing modules...
```

---

## File Locations

| File | Lines | Status |
|------|-------|--------|
| `src/pruners/trial_log.rs` | ~150 | Pending |
| `src/pruners/absorb_compress.rs` | ~200 | Pending |
| `src/pruners/hot_swap.rs` | ~180 | Pending |
| `src/pruners/regression.rs` | ~150 | Pending |
| `src/pruners/bandit.rs` (extension) | ~100 added | Pending |
| `src/pruners/mod.rs` (exports) | ~10 added | Pending |
| `examples/hl_01_trial_log.rs` | ~200 | Pending |
| `examples/hl_02_hotswap.rs` | ~250 | Pending |
| `tests/bench_absorb_compress.rs` | ~150 | Pending |

---

## Out of Scope

- [ ] Coding agent integration (LLM writing new validators — future work)
- [ ] Multi-agent HL (multiple bandit pruners coordinating — Bomberman Plan 033)
- [ ] Contextual bandits (feature vectors per arm)
- [ ] WASM validator auto-generation from failure traces

---

## References

- Plan 030: Multi-Armed Bandit implementation
- Plan 021: ScreeningPruner (absolute relevance)
- Research 14: "Learning Beyond Gradients" (`.research/14_Learning_Beyond_Gradients.md`)
- riir-validator-sdk: WASM validator SDK