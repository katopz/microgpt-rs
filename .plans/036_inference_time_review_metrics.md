# Plan 036: Inference-Time Review Metrics (Reinforced Agent Distillation)

**Branch:** `develop/feature/036_inference_time_review_metrics`
**Depends on:** Plan 030 (bandit), Plan 032 (HL infrastructure), Plan 021 (ScreeningPruner)
**Research:** `.research/15_Reinforced_Agent_Inference_Time_Feedback.md` — arXiv:2604.27233
**Goal:** Add Helpfulness-Harmfulness metrics and structured review loops to the existing pruner/bandit pipeline, enabling data-driven decisions about when reviewer intervention is net-positive.

---

## Overview

The Reinforced Agent paper proves that inference-time feedback (review before execute) improves tool-calling accuracy by +5.5% on irrelevance detection and +7.1% on multi-turn tasks. The key insight is not the reviewer itself — our `ScreeningPruner` already does that — but the **measurement framework**:

1. **Helpfulness**: % of cases where base agent was WRONG and reviewer FIXED it
2. **Harmfulness**: % of cases where base agent was RIGHT and reviewer BROKE it
3. **Benefit-to-Risk Ratio**: Helpfulness ÷ Harmfulness (paper found 3.1:1 for reasoning reviewers)

Without these metrics, you cannot tell if your `BanditPruner` or `ScreeningPruner` is helping or hurting. The paper also introduces structured review loops (`rN` — progressive feedback with max N iterations) which map to a config-driven outer loop around our existing PPoT rescue.

### What We Already Have (No Action Needed)

| Paper Concept | Existing Equivalent |
|---|---|
| Reviewer Agent evaluates before execute | `ScreeningPruner::relevance()` / `ConstraintPruner::is_valid()` |
| Best-of-N Selection | `build_dd_tree_screened()` — tree search with scored branches |
| Over-skepticism mitigation | Plan 029 Task 7 — explicit ownership boundary |
| GEPA / prompt optimization | `AbsorbCompress` + `HotSwapPruner` (code-level, more robust) |
| Distilled reviewer | `WasmPruner` — compiled, deterministic, ultra-fast |

### What's Genuinely New

1. **ReviewMetrics** — track helpful/harmful/both_correct/both_wrong per pruner
2. **Benefit-Risk Ratio** — computed metric informing whether to keep reviewing
3. **ReviewLoopConfig** — structured outer loop with rejection-feedback injection (paper's `rN`)
4. **Distillation Gate** — only `AbsorbCompress::compress()` when benefit-risk ratio exceeds threshold

---

## Tasks

- [x] **Task 1: ReviewMetrics struct** (`src/pruners/review_metrics.rs`)
  - Struct `ReviewMetrics` with atomic counters:
    - `helpful: AtomicU64` — base was wrong, reviewer fixed it
    - `harmful: AtomicU64` — base was right, reviewer broke it
    - `both_correct: AtomicU64` — both agreed correct
    - `both_wrong: AtomicU64` — both agreed wrong
  - `ReviewMetrics::new()` — zero-initialized
  - `ReviewMetrics::record(base_correct: bool, reviewed_correct: bool)` — classifies and increments the right counter
  - `ReviewMetrics::helpfulness() -> f64` — `helpful / (helpful + both_wrong)` as percentage
  - `ReviewMetrics::harmfulness() -> f64` — `harmful / (harmful + both_correct)` as percentage
  - `ReviewMetrics::benefit_ratio() -> f64` — `helpfulness / harmfulness`, returns `f64::INFINITY` if harmfulness is 0
  - `ReviewMetrics::total() -> u64` — sum of all counters
  - `ReviewMetrics::reset()` — zero all counters
  - `ReviewMetrics::summary() -> ReviewSummary` — struct with all computed values for display/logging
  - `impl Display for ReviewMetrics` — human-readable one-liner: `"helpful=36.8% harmful=11.7% ratio=3.1:1 n=1000"`
  - Thread-safe: all operations use `Ordering::Relaxed` (statistics, not synchronization)
  - Tests: record classifications, ratio calculation, zero-harmful edge case, display format
  - ~120 lines

- [x] **Task 2: Integrate ReviewMetrics into TrialLog** (`src/pruners/trial_log.rs` extension)
  - Add `review_metrics: Option<ReviewMetrics>` field to `TrialLog`
  - `TrialLog::with_metrics(self) -> Self` — enable metrics tracking (builder pattern)
  - Extend `TrialRecord` with optional fields:
    - `base_correct: Option<bool>` — was the base pruner (without review) correct?
    - `reviewed_correct: Option<bool>` — was the reviewed decision correct?
  - `TrialLog::append_with_review(&mut self, record: &TrialRecord, base_correct: bool, reviewed_correct: bool)` — appends record AND updates metrics
  - `TrialLog::metrics(&self) -> Option<&ReviewMetrics>` — read-only access
  - `TrialLog::metrics_summary(&self) -> Option<ReviewSummary>` — convenience
  - Backwards compatible: existing `TrialLog::append()` still works, metrics remain `None`
  - Tests: append_with_review updates counters, metrics None when not enabled, summary matches manual calculation
  - ~60 lines added

- [x] **Task 3: ReviewStrategy enum** (`src/pruners/review_metrics.rs`)
  - Enum mirroring the paper's three mechanisms:
    ```rust
    pub enum ReviewStrategy {
        /// Progressive feedback: iteratively review and inject rejection feedback.
        /// Paper's rN — up to N review loops.
        ProgressiveFeedback { max_loops: usize },
        /// Best-of-N selection: generate N candidates, reviewer picks best.
        /// Maps to DDTree with budget N.
        BestOfNSelection { candidates: usize },
        /// Best-of-N grading: score each candidate 0.0-1.0, pick highest.
        /// Maps to DDTree with ScreeningPruner relevance scoring.
        BestOfNGrading { candidates: usize },
    }
    ```
  - `ReviewStrategy::default()` — `ProgressiveFeedback { max_loops: 2 }` (paper's best performer)
  - `impl Display for ReviewStrategy` — `"r2"`, `"s5"`, `"g5"` notation matching paper
  - Tests: default, display format
  - ~40 lines

- [x] **Task 4: BenefitRatioGate for AbsorbCompress** (`src/pruners/absorb_compress.rs` extension)
  - Add `min_benefit_ratio: f64` field to `CompressConfig` (default: `2.0` — conservative)
  - Extend `AbsorbCompress::should_compress()` to accept `&ReviewMetrics`:
    - Returns `false` if metrics show `benefit_ratio() < min_benefit_ratio`
    - Logic: "don't harden a reviewer's decisions if the reviewer is net-negative"
  - Add `AbsorbCompress::should_compress_blind(&self) -> bool` — original logic without metrics (backwards compat)
  - Update `compress()` to log the benefit ratio when gating
  - Tests: compress blocked when ratio below threshold, compress proceeds when ratio above, blind version unaffected
  - ~40 lines added

- [x] **Task 5: ReviewLoopConfig for PPoT** (`src/speculative/ppot/types.rs` extension)
  - Add fields to `PpotConfig`:
    ```rust
    /// Maximum review iterations before giving up (paper's rN).
    /// 0 = disabled, 1 = single rescue attempt, 2+ = structured review loop.
    pub max_review_loops: usize,
    /// Whether to carry rejection reason between review loops.
    pub inject_rejection_feedback: bool,
    /// Minimum benefit-risk ratio to continue reviewing.
    /// Below this threshold, stop reviewing (reviewer is net-negative).
    pub min_review_benefit_ratio: f64,
    ```
  - Defaults: `max_review_loops: 0` (disabled), `inject_rejection_feedback: false`, `min_review_benefit_ratio: 2.0`
  - Update `PpotConfig::default()` and `PpotConfig::for_math()` presets
  - Tests: default values, custom config
  - ~30 lines added

- [x] **Task 6: Structured review loop in PPoT rescue** (`src/speculative/ppot/resample.rs` extension)
  - New function `ppot_rescue_reviewed()`:
    ```rust
    pub fn ppot_rescue_reviewed(
        marginals: &[f32],
        config: &Config,
        pruner: &dyn ScreeningPruner,
        ppot_config: &PpotConfig,
        metrics: Option<&ReviewMetrics>,
        rng: &mut Rng,
    ) -> Option<Vec<usize>>
    ```
  - Logic:
    1. Check `metrics.benefit_ratio() >= ppot_config.min_review_benefit_ratio` — skip if reviewer is net-negative
    2. For `loop_idx in 0..ppot_config.max_review_loops`:
       a. Call `ppot_rescue()` to get candidate path
       b. If valid → return path
       c. If `inject_rejection_feedback` → update entropy threshold based on `RejectionInsight` from `SessionKnowledge`
       d. If metrics show diminishing returns (ratio dropping) → break early
    3. Return `None` if all loops fail
  - Delegates to existing `ppot_rescue()` internally — zero duplication
  - Tests: returns path on first success, loops on failure, breaks when ratio below threshold, disabled when max_loops=0
  - ~80 lines

- [x] **Task 7: Wire ReviewMetrics into BanditSession** (`src/pruners/bandit.rs` extension)
  - Add `review_metrics: Option<ReviewMetrics>` field to `BanditSession`
  - `BanditSession::with_metrics(self, metrics: ReviewMetrics) -> Self` — builder
  - In `BanditSession::run()`:
    - After each episode, compare bandit's chosen arm vs the true optimal arm
    - Record `base_correct` (was non-bandit random pick correct?) and `reviewed_correct` (was bandit pick correct?)
    - Call `metrics.record(base_correct, reviewed_correct)`
  - Add `benefit_ratio()` method to `BanditSessionResult` — convenience accessor
  - Tests: metrics populated after run, ratio matches manual calculation
  - ~60 lines added

- [x] **Task 8: Demo — ReviewMetrics with Bandit** (`examples/review_01_metrics.rs`)
  - Uses `BernoulliEnv` (5 arms) with `BanditSession`
  - Enables `ReviewMetrics` via `BanditSession::with_metrics()`
  - Runs 1000 episodes with `TrialLog` + metrics
  - Prints:
    - Episode-by-episode Q-value convergence
    - Final `ReviewMetrics::summary()` — helpful%, harmful%, ratio
    - Whether `AbsorbCompress` would be gated (ratio vs threshold comparison)
  - Demonstrates: "the bandit reviewer fixed X% of random errors, broke Y% of correct picks"
  - ~150 lines

- [x] **Task 9: Benchmark — ReviewMetrics overhead** (`tests/bench_review_metrics.rs`)
  - Benchmark `ReviewMetrics::record()` throughput (atomic increments)
  - Benchmark `BanditPruner::relevance()` with and without metrics enabled
  - Benchmark `ppot_rescue_reviewed()` vs `ppot_rescue()` — measure loop overhead
  - Target: `record()` adds <1ns (single atomic increment path), relevance() overhead <2%, rescue_reviewed() overhead proportional to max_loops
  - ~100 lines

- [x] **Task 10: Update docs**
  - Update `src/pruners/mod.rs` with `pub mod review_metrics;`
  - Update `README.md` with "Inference-Time Review Metrics" section referencing arXiv:2604.27233
  - Update `.docs/09_heuristic_learning.md` with benefit-risk ratio guidance
  - Update `src/speculative/ppot/mod.rs` re-exports for `ppot_rescue_reviewed`

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│              Inference-Time Review Metrics                       │
│                                                                  │
│  ┌─────────────────┐                                            │
│  │  ReviewMetrics   │  helpful / harmful / both_correct /       │
│  │  (atomic counts) │  both_wrong → benefit_ratio()             │
│  └────────┬────────┘                                            │
│           │                                                      │
│     ┌─────┼──────────────────────────────┐                      │
│     ▼     ▼                              ▼                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  TrialLog    │  │ BanditSession│  │  PPoT Rescue │          │
│  │  (persist)   │  │ (per-arm)    │  │ (per-loop)   │          │
│  │              │  │              │  │              │          │
│  │ append_with  │  │ run() tracks │  │ rescue_      │          │
│  │ _review()    │  │ base vs      │  │ reviewed()   │          │
│  │              │  │ reviewed     │  │ checks ratio │          │
│  └──────────────┘  └──────────────┘  │ before loop  │          │
│                                      └──────┬───────┘          │
│                                             │                   │
│                                             ▼                   │
│                                    ┌──────────────────┐        │
│                                    │ AbsorbCompress   │        │
│                                    │ Gate: only       │        │
│                                    │ compress when    │        │
│                                    │ ratio > threshold│        │
│                                    └──────────────────┘        │
│                                                                │
│  ┌──────────────────────────────────────────────────┐          │
│  │  ReviewStrategy enum                             │          │
│  │  ProgressiveFeedback { max_loops: N }  ← paper rN│          │
│  │  BestOfNSelection { candidates: N }    ← paper sN│          │
│  │  BestOfNGrading { candidates: N }      ← paper gN│          │
│  └──────────────────────────────────────────────────┘          │
└─────────────────────────────────────────────────────────────────┘
```

---

## Module Structure

```text
src/pruners/
├── mod.rs                    (add: pub mod review_metrics)
├── review_metrics.rs         (NEW — ReviewMetrics, ReviewSummary, ReviewStrategy)
├── trial_log.rs              (MODIFY — add review fields, metrics integration)
├── absorb_compress.rs        (MODIFY — add benefit_ratio gate)
├── bandit.rs                 (MODIFY — wire metrics into BanditSession)
├── ...

src/speculative/ppot/
├── types.rs                  (MODIFY — add review loop config fields)
├── resample.rs               (MODIFY — add ppot_rescue_reviewed)
├── mod.rs                    (MODIFY — add re-export)
├── ...

examples/
├── review_01_metrics.rs      (NEW — demo)

tests/
├── bench_review_metrics.rs   (NEW — benchmarks)
```

---

## Key Design Decisions

### 1. Atomic counters, not mutex

`ReviewMetrics` uses `AtomicU64` with `Ordering::Relaxed`. This is a statistics counter, not a synchronization primitive. No lock contention on the hot path.

### 2. Opt-in via builder pattern

`TrialLog::with_metrics()` and `BanditSession::with_metrics()` — metrics are opt-in. Existing code paths are untouched. Zero overhead when disabled.

### 3. Gate, not block

`AbsorbCompress` doesn't refuse to compress — it gates based on metrics. If metrics are unavailable (None), it falls back to the original behavior. The gate is advisory, not mandatory.

### 4. Paper's 3.1:1 ratio as default threshold

The paper found o3-mini achieves 3.1:1 benefit-to-risk ratio. We default `min_benefit_ratio` to 2.0 (conservative — allow slightly worse reviewers). Users can tighten to 3.0 if they want paper-quality gates.

### 5. ReviewLoopConfig lives in PpotConfig, not a new struct

The review loop is a PPoT concern (rescue with structured feedback). Adding it to existing `PpotConfig` avoids introducing a new config struct for 3 fields.

---

## File Locations

| File | Lines | Change Type |
|------|-------|-------------|
| `src/pruners/review_metrics.rs` | ~160 | NEW |
| `src/pruners/trial_log.rs` | ~60 | MODIFY |
| `src/pruners/absorb_compress.rs` | ~40 | MODIFY |
| `src/pruners/bandit.rs` | ~60 | MODIFY |
| `src/pruners/mod.rs` | ~2 | MODIFY |
| `src/speculative/ppot/types.rs` | ~30 | MODIFY |
| `src/speculative/ppot/resample.rs` | ~80 | MODIFY |
| `src/speculative/ppot/mod.rs` | ~3 | MODIFY |
| `examples/review_01_metrics.rs` | ~150 | NEW |
| `tests/bench_review_metrics.rs` | ~100 | NEW |

---

## Out of Scope

- [ ] LLM-based reviewer agent (the paper uses o3-mini; our WASM validators are the distilled equivalent)
- [ ] GEPA implementation for automated prompt optimization (AbsorbCompress + HotSwapPruner serve this role at code level)
- [ ] RAG query reviewer gate for riir-rest (the query is deterministic via `embedding_to_query`; a gate at `EmbeddingRouter` level in riir-router is more appropriate but out of scope for this plan)
- [ ] Cross-domain metric transfer (metrics are per-pruner, not shared across domains)
- [ ] Best-of-N Selection/Grading implementations (DDTree already supersedes these; paper shows Progressive Feedback outperforms both)

---

## References

- arXiv:2604.27233 — "Reinforced Agent: Inference-Time Feedback for Tool-Calling Agents"
- Plan 030 — Multi-Armed Bandit (`src/pruners/bandit.rs`)
- Plan 032 — Heuristic Learning Infrastructure (`TrialLog`, `AbsorbCompress`, `RegressionSuite`)
- Plan 021 — ScreeningPruner (`speculative/types.rs`)
- Plan 026/027 — PPoT rescue (`speculative/ppot/`)