# Plan 027: Adaptive PPoT Rescue with Rejection Memory

**Branch:** `develop/feature/027_adaptive_ppot_rescue`
**Depends on:** Plan 026 (PPoT Logit Resampling) — must be complete and benchmarked first
**Research:** `.research/12_TRT_Test-time_Recursive_Thinking.md`
**Status:** ✅ Mostly Complete (Task 7 benchmarks pending)

---

## Overview

Distill "Test-time Recursive Thinking" (arXiv:2602.03094) into microgpt-rs on top of PPoT rescue (Plan 026). The core idea: when PPoT resamples token variants and the `ConstraintPruner` rejects them, record structured "don't" insights that bias future resampling within the same generation session. This makes PPoT adaptive rather than random.

TRT proves three things we can apply at token level:
1. **"Don'ts" beat "dos"** — recording failure patterns outperforms recording successes (Figure 8)
2. **Knowledge is compact** — stays under 1.5% of context after 64 rounds (Figure 7)
3. **Depth beats breadth** — iterative refinement with accumulated knowledge outperforms parallel random sampling (Figure 6)

**Approach:** Bench Plan 026 first (PPoT rescue without adaptation). Then apply 027 on top. The delta between 027 and 026 isolates the TRT contribution.

**Expected result:** +2-4% additional acceptance rate on top of PPoT baseline, at zero additional GPU cost.

---

## Prerequisites

- [ ] Plan 026 fully implemented and merged
- [ ] PPoT rescue benchmarks recorded as baseline (Task 7 from Plan 026)
- [ ] `ppot` feature flag working in `Cargo.toml`
- [ ] All Plan 026 tests passing

---

## Tasks

- [x] **Task 1: RejectionInsight struct and SessionKnowledge** (`src/speculative/ppot/knowledge.rs`)
  - Define `RejectionInsight` struct:
    ```rust
    pub struct RejectionInsight {
        pub position: usize,
        pub rule: TokenRule,
        pub original_token: usize,
        pub attempted_token: usize,
        pub error_kind: Option<ErrorKind>,
        pub entropy: f32,
        pub accepted: bool,
    }
    ```
  - Define `SessionKnowledge` struct with bounded ring buffer:
    ```rust
    pub struct SessionKnowledge {
        insights: Vec<RejectionInsight>,
        max_insights: usize,           // default: 64
        success_count_by_rule: [usize; 5], // per TokenRule variant
        fail_count_by_rule: [usize; 5],
    }
    ```
  - `SessionKnowledge::record(insight: RejectionInsight)` — append, evict oldest if full
  - `SessionKnowledge::success_rate(rule: TokenRule) -> f32` — per-rule success rate
  - `SessionKnowledge::position_affinity(position: usize) -> f32` — how often resampling this position succeeds
  - `SessionKnowledge::should_skip(position: usize, rule: TokenRule) -> bool` — true if all attempts with this rule at nearby positions failed
  - `SessionKnowledge::preferred_rules(position: usize) -> Vec<TokenRule>` — rules sorted by success rate
  - Unit tests: ring buffer eviction, success rate calculation, skip logic

- [x] **Task 2: Adaptive position selection** (`src/speculative/ppot/entropy.rs` extension)
  - `fn identify_positions_adaptive(marginals, threshold, knowledge: &SessionKnowledge) -> Vec<usize>`
    - Starts from high-entropy positions (existing logic)
    - Reorders by `position_affinity` — positions with historical success get priority
    - Filters out `should_skip` positions — don't waste CPU on known-bad positions
    - Falls back to standard entropy-only if `knowledge` is empty (cold start)
  - `fn identify_positions_adaptive_into(marginals, threshold, knowledge, &mut Vec<usize>)` — zero-alloc variant
  - Unit tests: cold start matches entropy-only, knowledge biases ordering, skip filters work

- [x] **Task 3: Per-sample strategy cycling** (`src/speculative/ppot/resample.rs` extension)
  - `fn ppot_resample_multi_strategy(marginals, positions, strategies: &[TokenRule], rng) -> Vec<Vec<usize>>`
    - Generates m samples, each using a different strategy from the cycle
    - Strategy order: `[Digit, Arithmetic, Compare, Augment, All]` repeated for m samples
    - If `SessionKnowledge` has preferred rules for a position, use those first
  - `fn ppot_rescue_adaptive(sctx, knowledge, pruner, rng) -> Option<Vec<usize>>`
    - Replaces `ppot_rescue()` from Plan 026
    - Uses `identify_positions_adaptive` instead of `identify_high_entropy_positions`
    - Uses `ppot_resample_multi_strategy` instead of single-rule resample
    - Records `RejectionInsight` for each attempted variant (accepted or rejected)
    - Returns best variant ranked by self-consistency (if multiple valid)
  - Integration test: adaptive rescue produces different variants than random rescue after warm-up

- [x] **Task 4: Self-consistency ranking** (`src/speculative/ppot/rank.rs`)
  - `fn rank_by_consistency(variants: &[Vec<usize>]) -> Vec<(usize, usize)>`
    - Returns `(variant_index, agreement_count)` sorted descending
    - Agreement: two variants match on tokens outside resampled positions
    - O(m² × lookahead) — negligible for m=10, lookahead=5-8
  - `fn select_best_variant(variants: &[Vec<usize>], pruner: &dyn ScreeningPruner) -> Option<Vec<usize>>`
    - Filter to valid variants (pass pruner)
    - If single valid variant → return it
    - If multiple valid → rank by consistency, return highest agreement
    - If none valid → None
  - Unit tests: single variant returned, tie-breaking by consistency, all-rejected returns None

- [x] **Task 5: Adaptive entropy threshold** (`src/speculative/ppot/types.rs` extension)
  - Extend `PpotConfig`:
    ```rust
    pub struct PpotConfig {
        // ... existing fields from Plan 026 ...
        pub adaptive_threshold: bool,        // default: true
        pub entropy_threshold_min: f32,      // default: 0.3
        pub entropy_threshold_max: f32,      // default: 1.0
        pub threshold_lower_on_fail: f32,    // default: 0.1 (subtract on fail)
        pub threshold_raise_on_success: f32, // default: 0.05 (add on success)
    }
    ```
  - `fn adaptive_threshold(config: &PpotConfig, knowledge: &SessionKnowledge) -> f32`
    - If last rescue succeeded: threshold = min(max, current + raise)
    - If last rescue failed: threshold = max(min, current - lower)
    - If no history: use `entropy_threshold` from config
  - This captures TRT's finding that models switch strategy more after failure (82%) than success (74%)
  - Unit tests: threshold adjusts correctly, bounded by min/max

- [x] **Task 6: Wire adaptive rescue into speculative step** (`src/speculative/step.rs`)
  - Add `SessionKnowledge` to `SpeculativeContext` (or thread parameter, TBD)
  - Replace `ppot_rescue()` call with `ppot_rescue_adaptive()`
  - After each rescue attempt (success or fail), record insight into knowledge
  - Feature-gated behind `ppot` feature flag (same gate as Plan 026)
  - Add `adaptive_ppot: bool` config flag (default: true when ppot enabled)
  - Integration test: full speculative step with adaptive rescue

- [ ] **Task 7: Benchmarks — before/after comparison** (`src/benchmark.rs`) — PENDING
  - **Must run AFTER Plan 026 benchmarks are recorded**
  - Benchmark: adaptive rescue vs random rescue acceptance rate
  - Benchmark: adaptive rescue overhead (should match PPoT baseline ±2%)
  - Benchmark: knowledge accumulation memory (<1KB per full generation)
  - Benchmark: cold start (first 10 steps) vs warm (steps 50+) acceptance rate
  - Record results in `benchmarks/027_adaptive_ppot/` with comparison to Plan 026 baseline
  - Report: acceptance rate delta, wall-clock delta, memory overhead

- [x] **Task 8: Update docs**
  - Update `.research/12_TRT_Test_time_Recursive_Thinking.md` with implementation notes
  - Add `Adaptive PPoT Rescue (Plan 027)` section to README architecture
  - Update `src/speculative/ppot/` module docs
  - Reference Plan 026 baseline benchmarks

---

## File Change Summary

| File | Change |
|------|--------|
| `src/speculative/ppot/knowledge.rs` | ✅ **New:** `RejectionInsight`, `SessionKnowledge` with ring buffer |
| `src/speculative/ppot/rank.rs` | ✅ **New:** self-consistency ranking, best variant selection |
| `src/speculative/ppot/entropy.rs` | ✅ **Extend:** add `identify_positions_adaptive()` |
| `src/speculative/ppot/resample.rs` | ✅ **Extend:** add `ppot_resample_multi_strategy()`, `ppot_rescue_adaptive()` |
| `src/speculative/ppot/types.rs` | ✅ **Extend:** adaptive threshold fields in `PpotConfig` |
| `src/speculative/ppot/mod.rs` | ✅ **Update:** re-export new types, wire adaptive API |
| `src/speculative/step.rs` | ✅ API available, integration point wired via `ppot_rescue_adaptive()` |
| `src/benchmark.rs` | ⬜ **Pending:** add Plan 027 benchmarks with 026 comparison (Task 7) |
| `README.md` | ✅ **Update:** add Adaptive PPoT section |

---

## Architecture

```
Plan 026 (PPoT baseline):
  DDTree rejects all
    → identify H-positions (entropy only)
    → resample m variants (single TokenRule, random)
    → screen via ConstraintPruner
    → return first valid
    → DONE (no learning)

Plan 027 (Adaptive PPoT):
  DDTree rejects all
    → adaptive_threshold(knowledge)         // lower after fail, raise after success
    → identify_positions_adaptive(marginals, threshold, knowledge)
        → start from high-entropy positions
        → reorder by position_affinity (past success)
        → filter should_skip (known-dead positions)
    → ppot_resample_multi_strategy(positions, strategies)
        → sample 1: TokenRule::Digit        (try different constants)
        → sample 2: TokenRule::Arithmetic   (try different operators)
        → sample 3: TokenRule::Compare      (try different comparisons)
        → sample 4: TokenRule::Augment      (try different assignments)
        → sample 5: TokenRule::All          (unrestricted)
        → sample 6-10: repeat or use preferred_rules()
    → screen each via ConstraintPruner
    → rank valid variants by self-consistency
    → return best (highest agreement)
    → record RejectionInsight for each attempt
    → knowledge biases next rescue ← ─ ─ ─ ┘
```

### Knowledge Accumulation Flow

```
Step 1:  Rescue fails → record 10 insights (all rejected)
         Knowledge: "Digit@pos3 failed ×3, Arithmetic@pos3 failed ×2, ..."

Step 2:  Rescue fails → record 10 insights
         Knowledge: "pos3 consistently bad, pos7 has 1 success with Compare"

Step 3:  Rescue succeeds → Compare@pos7 accepted
         Knowledge biases: prefer Compare rule, prefer pos7-like positions

Step 4+: Adaptive: skip pos3, prioritize pos7, use Compare first
         Threshold raised (success) → fewer positions explored, higher quality
```

### Ring Buffer Sizing

```rust
const MAX_INSIGHTS: usize = 64;
// Each insight: ~48 bytes (position, rule, tokens, error, entropy, flag)
// Total: 64 × 48 = 3KB per session
// TRT proves <1.5% of context — for us it's <0.01% of generation memory
```

---

## Performance Targets

| Metric | Plan 026 Baseline | Plan 027 Target | Rationale |
|---|---|---|---|
| Rescue acceptance rate | X% (bench 026) | X + 2-4% | TRT's adaptive exploration gain |
| Wall-clock overhead | Y μs (bench 026) | Y + 0-5% | Knowledge lookup is O(1), ranking is O(m²×L) |
| Memory per session | 0 KB | < 4 KB | 64 × 48B ring buffer |
| Cold start acceptance | Same as 026 | Same as 026 | No knowledge = falls back to random |
| Warm acceptance (step 50+) | Same as 026 | X + 3-5% | Knowledge-informed bias kicks in |
| Strategy diversity | 1 rule per rescue | 5 rules per rescue | Prevents redundant exploration |

---

## Regression Watch

Plan 027 must NOT regress Plan 026 baselines:

| Metric | Regression Limit |
|---|---|
| PPoT rescue wall-clock | +5% max |
| Memory per decode step | +1 KB max |
| Cold start acceptance rate | -0% (must match 026) |
| DDTree build time | 0% (untouched) |
| DFlash marginal quality | 0% (untouched) |

If any regression exceeds limits, the adaptive logic should auto-disable and fall back to Plan 026 random rescue.

---

## Test Results

14 knowledge tests + 25 rank tests + adaptive entropy/resample tests = **39 Plan 027-specific tests** passing.
Combined with Plan 026 tests: **78 total PPoT tests**, all 242 project tests pass with zero regressions.

---

## Out of Scope

- Cross-generation knowledge persistence (knowledge dies with session)
- Knowledge sharing across parallel decode streams
- LLM-generated strategy prompts (static TokenRule enums only)
- Test execution for variant selection (WasmPruner is sufficient)
- Knowledge pruning / staleness eviction (sessions are short, ring buffer is sufficient)
- anyrag integration for document-level TRT (future consideration)
- riir-validator-sdk changes (all host-side)

---

## References

- "Test-time Recursive Thinking" (arXiv:2602.03094) — Zhuang et al.
- Research: `.research/12_TRT_Test_time_Recursive_Thinking.md`
- PPoT Plan: `.plans/026_ppot_logit_resampling.md`
- PPoT Research: `.research/11_PPoT_Probabilistic_Programs_of_Thought.md`
- Self-Consistency (Wang et al. 2022): arXiv:2203.11171
- Screening Pruner: `.plans/021_screening_pruner.md`
