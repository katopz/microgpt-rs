# Plan 042: TTT-Inspired E2E Feedback Loop — Inference Results → Training Data

> **Research:** `microgpt-rs/.research/19_TTT_Discover_Test_Time_Training.md`
> **Related:** Plan 041 (game training), anyrag Plan 003 (self-improving cycle), Plan 008 (inference budget)
> **Branch:** `develop/feature/042_ttt_feedback_loop`

---

## Overview

Close the feedback loop: high-reward inference results flow back as training data for LoRA adapters.
TTT-Discover (Research 19) validated this architecture at research scale ($500/problem).
We build it at production scale: every inference produces a reward signal, top results export as JSONL,
riir-burner trains LoRA from that JSONL, riir-ai deploys the adapter.

### The Loop

```
┌──────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│  ┌──────────┐   classify    ┌─────────┐   inference   ┌──────────────┐  │
│  │  anyrag  │──────────────▶│ Domain  │──────────────▶│  microgpt-rs │  │
│  │          │               │ Config  │               │  DDTree      │  │
│  └──────────┘               └─────────┘               └──────┬───────┘  │
│       ▲                                                      │          │
│       │              reward = relevance score                │          │
│       │                       │                              │          │
│       │                       ▼                              │          │
│       │               ┌──────────────┐                       │          │
│       │               │ WasmPruner   │                       │          │
│       │               │ .relevance() │                       │          │
│       │               └──────┬───────┘                       │          │
│       │                      │                               │          │
│  ┌────┴───────┐    top-K     ▼                               │          │
│  │  Solution  │◀──── results with reward > threshold         │          │
│  │  Cache     │                                              │          │
│  └────┬───────┘                                              │          │
│       │                                                      │          │
│       │  export JSONL                                        │          │
│       ▼                                                      │          │
│  ┌────────────┐    train LoRA    ┌───────────┐   deploy     │          │
│  │riir-burner │─────────────────▶│ adapter   │─────────────▶│          │
│  │            │                  │ .bin      │  (next       │          │
│  └────────────┘                  └───────────┘   request)   │          │
│                                                              │          │
└──────────────────────────────────────────────────────────────────────────┘
```

### Problem

1. **No reward feedback.** WasmPruner scores every token path via `relevance()`, but the score is discarded after inference. High-quality paths don't feed back into training.
2. **No solution cache.** Each inference is stateless. Similar queries re-solve from scratch. TTT-Discover's PUCT-inspired buffer shows reuse saves compute.
3. **LoRA training is offline-only.** riir-burner trains on curated JSONL corpora. The pipeline from "good inference result" → "training sample" doesn't exist for language tasks.
4. **Plan 041 is game-specific.** Bomberman training pipeline exists. Language/code domain needs the same loop.

### Scope

**In scope:**
- Reward signal format (`InferenceResult` with reward, domain, context, output)
- Solution cache in anyrag (PUCT-inspired scoring, top-K retention)
- JSONL export from cache → riir-burner-compatible format
- Wire the loop for at least one language domain (e.g., `py2rs`)

**Deferred:**
- Per-request budget adaptation (needs cache first)
- Entropic advantage estimation (no RL training during inference)
- Test-time LoRA updates (not viable for production)
- Multi-query batch rollouts (research-scale only)

---

## Architecture

### Reward Signal

Every DDTree inference already computes `relevance()` via ScreeningPruner. We capture the result:

```rust
/// Output of a single inference pass, with reward signal.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceResult {
    /// Domain that handled this inference.
    pub domain: String,
    /// Best-path reward (max relevance score from WasmPruner).
    pub reward: f32,
    /// Number of nodes explored in DDTree.
    pub tree_budget_used: usize,
    /// Inference budget that was applied.
    pub budget: InferenceBudget,
    /// Input prompt hash (for dedup, not stored).
    pub prompt_hash: u64,
    /// Generated output text.
    pub output: String,
    /// Timestamp (Uuid v7 prefix).
    pub timestamp: i64,
    /// Was this result screened out (reward below threshold)?
    pub screened: bool,
}
```

### Solution Cache

PUCT-inspired selection for reusing past high-reward solutions:

```rust
/// Cached solution with PUCT-style scoring metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedSolution {
    pub query_hash: u64,
    pub domain: String,
    pub output: String,
    pub reward: f32,
    pub reuse_count: usize,
    pub last_used: i64,
    /// Children spawned from reusing this solution.
    pub child_max_reward: f32,
}

impl CachedSolution {
    /// PUCT-inspired score: exploit (reward) + explore (under-visited).
    /// Simplified from TTT-Discover: no tree structure, flat rank-based.
    fn puct_score(&self, total_visits: usize, reward_scale: f32, c: f32) -> f32 {
        let q = self.child_max_reward.max(self.reward);
        let exploration = c * reward_scale * (1.0 + total_visits as f32).sqrt()
            / (1.0 + self.reuse_count as f32);
        q + exploration
    }
}
```

### Export Format

JSONL compatible with riir-burner's existing `--corpus` input:

```jsonl
{"instruction":"translate to rust: async def fetch(url):","output":"async fn fetch(url: &str) -> Result<String, reqwest::Error>","reward":0.95,"domain":"py2rs"}
{"instruction":"translate to rust: def fibonacci(n):","output":"fn fibonacci(n: u64) -> u64","reward":0.88,"domain":"py2rs"}
```

riir-burner already reads JSONL. The only addition: optional `reward` field for quality-weighted sampling.

---

## Tasks

- [ ] **Task 1: `InferenceResult` type in microgpt-rs**
  - Add `InferenceResult` struct to `microgpt-rs/src/types.rs`
  - Fields: domain, reward, tree_budget_used, budget, prompt_hash, output, timestamp, screened
  - Derive `Serialize, Deserialize, Clone`
  - `InferenceBudget` already exists in anyrag — use same type or mirror
  - Test: serde roundtrip, default construction
  - ~30 lines

- [ ] **Task 2: Reward capture in DDTree**
  - After `extract_best_path_into()`, compute `InferenceResult` from:
    - `domain`: from config/router context (passed through)
    - `reward`: max score from best path (already computed in heap)
    - `tree_budget_used`: `self.tree.len()`
    - `budget`: the `Config` used for this inference
    - `output`: the generated text
    - `screened`: reward < screening_threshold
  - Return `InferenceResult` alongside the generated text
  - Minimal change to DDTree API: wrap return in `(String, InferenceResult)`
  - ~20 lines in `microgpt-rs/src/ddtree.rs`

- [ ] **Task 3: Solution cache in anyrag**
  - New module: `crates/lib/src/cache/mod.rs` + `crates/lib/src/cache/solution_cache.rs`
  - `SolutionCache` struct with:
    - `entries: papaya::HashMap<u64, CachedSolution>` (lock-free, per user's lib preference)
    - `max_entries: usize` (configurable, default 1000)
    - `reward_threshold: f32` (minimum reward to cache, default 0.7)
  - Methods:
    - `insert(result: &InferenceResult) -> bool` — only if reward > threshold
    - `lookup(query_hash: u64, domain: &str) -> Option<&CachedSolution>`
    - `select_for_reuse(domain: &str) -> Option<&CachedSolution>` — PUCT scoring
    - `export_jsonl(domain: &str) -> Vec<TrainingSample>` — riir-burner format
    - `prune()` — keep top-K by reward, always keep initial seeds
  - Feature-gated behind `#[cfg(feature = "solution-cache")]`
  - ~120 lines
  - Test: insert/filter/lookup/prune/export cycle

- [ ] **Task 4: Cache API endpoint in anyrag server**
  - `GET /cache/stats` — cache hit rate, entry count, top domains
  - `POST /cache/export` — export domain's cache as JSONL (for riir-burner)
  - `DELETE /cache/prune` — manual prune trigger
  - Behind `#[cfg(feature = "solution-cache")]` feature flag
  - ~60 lines in `crates/server/src/handlers/cache.rs`

- [ ] **Task 5: JSONL export bridge**
  - Export format: `{"instruction": prompt, "output": output, "reward": f32, "domain": str}`
  - Compatible with riir-burner's existing `--corpus input/train.jsonl`
  - riir-burner: add optional `--reward-weight` flag to sample proportionally to reward
  - If `--reward-weight` is set, sample training examples weighted by `reward` field
  - Without flag, uniform sampling (backward compatible)
  - ~40 lines in `riir-burner/src/pipeline.rs`

- [ ] **Task 6: Wire feedback in microgpt-rs**
  - After inference, if `solution-cache` feature is enabled:
    1. Build `InferenceResult` from DDTree output
    2. POST to anyrag `/cache/ingest` (or direct call if embedded)
  - Make this opt-in via config: `feedback_url: Option<String>` in `Config`
  - If `feedback_url` is None, skip (no behavior change)
  - If set, fire-and-forget POST (don't block inference on cache write)
  - ~30 lines in `microgpt-rs/src/feedback.rs` (new file)

- [ ] **Task 7: E2E validation**
  - Run anyrag with `solution-cache` feature
  - Run microgpt-rs with `feedback_url` pointing to anyrag
  - Execute 10+ inference requests in a domain (e.g., py2rs code translation)
  - Verify cache entries appear in `/cache/stats`
  - Export JSONL via `/cache/export`
  - Run riir-burner with exported JSONL as corpus
  - Verify adapter trains (loss decreases)
  - Document results in `.docs/12_ttt_feedback_loop_results.md`

---

## File Change Summary

### New files

| File | Lines | Purpose | Repo |
|------|-------|---------|------|
| `anyrag/crates/lib/src/cache/mod.rs` | ~5 | Module index | anyrag |
| `anyrag/crates/lib/src/cache/solution_cache.rs` | ~120 | Cache with PUCT scoring | anyrag |
| `anyrag/crates/server/src/handlers/cache.rs` | ~60 | Cache API endpoints | anyrag |
| `microgpt-rs/src/feedback.rs` | ~30 | Fire-and-forget cache write | microgpt-rs |
| `microgpt-rs/.docs/12_ttt_feedback_loop_results.md` | ~50 | E2E validation results | microgpt-rs |

### Modified files

| File | Change | Repo |
|------|--------|------|
| `microgpt-rs/src/types.rs` | Add `InferenceResult` struct (~30 lines) | microgpt-rs |
| `microgpt-rs/src/ddtree.rs` | Return `InferenceResult` alongside output (~20 lines) | microgpt-rs |
| `microgpt-rs/src/lib.rs` | `pub mod feedback;` | microgpt-rs |
| `anyrag/crates/lib/src/lib.rs` | `pub mod cache;` + feature gate | anyrag |
| `anyrag/crates/lib/Cargo.toml` | Add `solution-cache` feature | anyrag |
| `anyrag/crates/server/Cargo.toml` | Add `solution-cache` feature | anyrag |
| `anyrag/crates/server/src/handlers/mod.rs` | Add cache module | anyrag |
| `riir-burner/src/pipeline.rs` | Add `--reward-weight` sampling (~40 lines) | riir-burner |

---

## Design Decisions

### 1. Fire-and-forget feedback (not blocking)

TTT-Discover does synchronous evaluate → train. Production can't block inference on cache writes.
The feedback POST is fire-and-forget: if it fails, inference still succeeds. The cache is a best-effort
side channel, not a critical path.

### 2. papaya over RwLock<HashMap>

Per user's lib preference: `papaya` for lock-free concurrent access. The cache is read-heavy (lookup)
with occasional writes (insert). Papaya's optimistic locking fits this pattern.

### 3. Flat cache (not tree-structured PUCT)

TTT-Discover maintains a tree of states with parent-child relationships. Our cache is flat:
(query_hash → solution). We don't generate 512 rollouts per query. The PUCT scoring is simplified
to `exploit (reward) + explore (1/sqrt(visits))` without tree ancestry.

### 4. Feature-gated

Everything behind `solution-cache` feature flag. Without it, zero overhead. This is a new subsystem
that should prove value before becoming default.

### 5. Reward threshold for quality control

Only cache results with `reward > 0.7` (configurable). TTT-Discover's entropic objective favors
max-reward solutions. We approximate by only caching the top fraction. This prevents the cache
from filling with mediocre results.

### 6. JSONL as the bridge format

riir-burner already reads JSONL. Adding an optional `reward` field is backward-compatible.
This avoids creating a new binary format or protocol between the systems.

---

## Priority Order

| Priority | Task | Why | Effort |
|----------|------|-----|--------|
| P0 | Task 1: InferenceResult type | Foundation for everything | Small |
| P0 | Task 2: Reward capture | Data must flow | Small |
| P1 | Task 3: Solution cache | Core new capability | Medium |
| P1 | Task 5: JSONL export bridge | Close the loop | Small |
| P2 | Task 4: Cache API endpoints | Operational visibility | Small |
| P2 | Task 6: Wire feedback | Production integration | Small |
| P3 | Task 7: E2E validation | Prove it works | Small |

---

## Connection to Existing Plans & Research

| Item | Relationship |
|------|-------------|
| **Research 19 (TTT-Discover)** | This plan IS the distillation. Validate → Train loop at production scale. |
| **Research 16 (AutoTTS)** | `InferenceBudget::from_beta()` already implemented. Budget flows through this loop. |
| **Research 14 (Heuristic Learning)** | Complementary: R14 edits code (Bomberman), this plan trains weights (language). |
| **Plan 041 (Game Training)** | Game-specific version of this loop. This plan generalizes to language tasks. |
| **anyrag Plan 003 (Self-Improving)** | Original self-improving vision. This plan implements the feedback half. |
| **anyrag Plan 008 (Inference Budget)** | Budget is part of the reward signal. Already implemented. |
| **riir-burner** | Training endpoint. Already reads JSONL. Minor addition for reward-weighted sampling. |

---

## Expected Outcomes

1. Every DDTree inference produces a reward signal (no behavior change by default)
2. High-reward results accumulate in solution cache (behind feature flag)
3. Cached solutions can be exported as riir-burner-compatible JSONL
4. riir-burner trains LoRA from exported data (reward-weighted)
5. E2E demo: 10 inferences → cache → export → train → adapter
6. Measured: cache hit rate, reward distribution, training loss improvement

---

## Research Citation

```bibtex
@article{yuksekgonul2026tttdiscover,
  title   = {Learning to Discover at Test Time},
  author  = {Yuksekgonul, Mert and Koceja, Daniel and Li, Xinhao and
             Bianchi, Federico and McCaleb, Jed and Wang, Xiaolong and
             Kautz, Jan and Choi, Yejin and Zou, James and Guestrin, Carlos and Sun, Yu},
  journal = {arXiv preprint arXiv:2601.16175},
  year    = {2026}
}
```
