# Research: AutoTTS — Dynamic Test-Time Compute Scaling (16)

> Source: [AutoTTS: LLMs Improving LLMs — Agentic Discovery for Test-Time Scaling Strategies](https://zhengkid.github.io/AutoTTS-web/) by Tong Zheng, Haolin Liu, Chengsong Huang, et al. (UMD · UVA · WUSTL · UNC · Google · Meta)
> Date: 2026, distilled 2025-06

## Summary

AutoTTS reframes test-time scaling (TTS) strategy design from hand-crafted heuristics to **environment-driven automatic search**. Humans build an offline replay environment (states, actions, feedback, objectives), and a coding agent iteratively proposes and refines **code-defined controllers** within it — code edits, no gradient updates. The discovered controller (CMC — Confidence Momentum Controller) saves ~69.5% tokens vs uniform SC@64 at β ≈ 0.5 while matching held-out accuracy across four backbone scales.

Key architectural insight: **β parameterization** — every internal hyperparameter is a deterministic, monotonic function of a single scalar β ∈ [0, 1]. The outer search collapses to sweeping β, eliminating brittle per-threshold tuning.

Key measurement insight: **0 LLM calls during evaluation** — the replay store freezes all reasoning traces upfront, so candidate controllers are simulated through table lookups only.

---

## Core Concepts

### MDP Formulation

The problem is framed as allocating a finite budget over branches in fixed-length intervals.

**State** at step `t`: `s_t = (q, m_t, I_t, ℓ_t, Ω_t)`
- `q`: question
- `m_t`: number of instantiated branches
- `I_t`: active branch set
- `ℓ_t`: depth vector
- `Ω_t`: revealed probe triples

**Admissible actions** `A(s_t)`:
- `BRANCH` — open a new branch through first interval
- `CONTINUE(i)` — advance branch `i` by one interval
- `PROBE(i)` — reveal probe response without advancing depth
- `PRUNE(i)` — deactivate branch `i`
- `ANSWER` — terminate and apply terminal aggregator

**Cost**: `Cost(s_t) = Σ_i ℓ_{t,i} + κ_probe · |Ω_t|`

**Objective**: `max_{π, β} E_{q,y}[ 1{ŷ = y} − γ · C_{π,β}(q) ]`

### Offline Replay Environment

Built once per (model, benchmark) before discovery starts:

1. **Specify the interface.** Fix states, actions, cost, objective.
2. **Offline trajectory collection.** Draw N parallel independent reasoning traces per query. Partition into fixed-length segments. Enumerate branch prefixes with probe responses.
3. **Materialize the replay store.** Every transition consults the archived table — no new decoding.
4. **Hand off to discovery.** Candidate controllers simulated exclusively through `observe`/`step`.

**Cost: $39.9 and 160 minutes** for one full discovery run.

### β Parameterization

Each controller exports a single scalar β ∈ [0, 1] plus a deterministic, monotonic map from β to every internal knob:

- `n_init`: 2 + 6β (branches to open initially)
- `max_branch_use`: 4 + 60β (maximum branches)
- `warm_up`: 2 + 8β (rounds before gating activates)
- `abandon_patience`: 3 + 9β (rounds before pruning deviant branches)
- `ema_alpha`: 0.70 − 0.40β (EMA inertia — lower = more smoothing at high β)
- `conf_thresh`: 0.85 + 0.12β (harder to stop at high β)
- `delta_slack`: 0.04 − 0.03β (tolerance for momentum gate)

Monotonicity constraints:
- Budget-use parameters (n_init, max_branch_use, burst_aligned, widen_burst, warm_up, abandon_patience) are **NON-DECREASING** in β
- `conf_thresh` is **NON-DECREASING** (harder to stop → more budget)
- `trend_thresh` is **NON-INCREASING** (easier to widen → more budget)
- `ema_alpha` is **NON-INCREASING** (lower α = slower EMA = more inertia → more budget)

### Confidence Momentum Controller (CMC)

The discovered controller has four mechanisms:

1. **Trend-based stopping.** EMA of pool confidence; gate fires only when level is high AND trend is non-negative. Prevents stopping on transient confidence spikes.

2. **Coupled width–depth control.** EMA delta links widening and deepening: strong confidence gains suppress new branch spawning, stagnation triggers widening.

3. **Alignment-aware depth allocation.** Branches matching the pool winner get extra probe steps (burst_aligned multiplier). Concentrates compute on consensus.

4. **Conservative branch abandonment.** Branches abandoned only after persistent deviation for ≥ abandon_patience rounds, with minimum 2 branches preserved.

### Discovery Process

Coding agent (Claude/CodeX) iteratively:
1. Proposes a `method.py` (controller code)
2. Replay environment evaluates it (0 LLM calls, pure table lookup)
3. Agent receives accuracy/cost feedback + full execution traces
4. Agent rewrites code, next round

History augmentation: alongside β-sweep results, full action-by-action trajectories are archived. Traces give the explorer fine-grained behavioral evidence.

### Key Results

| Metric | Value |
|---|---|
| Token savings vs SC@64 at β=0.5 | **~69.5%** |
| Held-out accuracy | Matches SC@64 across four backbone scales |
| Discovery cost | **$39.9** per run |
| Discovery wall-clock | **160 minutes** |
| LLM calls during evaluation | **0** (replay only) |

---

## What Maps to microgpt-rs

### What Actually Applies

#### 1. Dynamic Budget Per Domain (High Value, Clean Fit)

AutoTTS allocates more compute to harder problems. Our `Config` has `tree_budget` and `draft_lookahead` but they're set once at construction. The natural home for dynamic budget is **per-domain config**, not inside DDTree.

Current architecture:
```
Prompt → anyrag /classify/domain → RouteDecision { domain } → ExpertBundle { pruner, lora } → Config::draft() → DDTree
```

AutoTTS-inspired change:
```
Prompt → anyrag /classify/domain → RouteDecision { domain } → ExpertBundle { pruner, lora, inference_budget } → Config::with_overrides(budget) → DDTree
```

This requires:
- `InferenceBudget` struct in `DomainConfig` (optional fields: `tree_budget`, `draft_lookahead`, `screening_threshold`)
- `Config::with_overrides()` method that clones and applies non-None overrides
- Wire through `ExpertBundle` or `ExpertRegistry`

**Implementation: ~80 lines of real code. No new traits, no new modules.**

#### 2. β Parameterization (Medium Value, Good Design Pattern)

The idea of a single scalar controlling all budget parameters is a clean config pattern. Instead of tuning `tree_budget`, `lookahead`, and `threshold` independently, a domain could specify `complexity: 0.8` and derive everything:

```rust
impl InferenceBudget {
    pub fn from_beta(beta: f32) -> Self {
        Self {
            tree_budget: Some((16.0 + 4984.0 * beta).round() as usize),
            draft_lookahead: Some((3.0 + 12.0 * beta).round() as usize),
            screening_threshold: Some(0.0 + 0.3 * beta),
        }
    }
}
```

This is optional sugar — domains can specify exact values OR a β scalar. The β mapping would be our own design, not a copy of CMC's parameters (which are for reasoning chains, not token trees).

#### 3. Confidence-Gap Early Exit in DDTree (Medium Value)

CMC's momentum gate stops when confidence is high AND stable. An analogous check in our DDTree Phase C:

```rust
// Phase C: Best-first expansion with early exit
while self.tree.len() < config.tree_budget {
    let Some(best) = self.heap.pop() else { break };
    self.tree.push(best);

    // Confidence-gap early exit: if best is dominant for N iterations, stop
    if self.tree.len() > 3 {
        let gap = best.score - second_best_score;
        if gap > dominance_threshold && consecutive_dominant >= patience {
            break; // Early exit: best branch is clearly winning
        }
    }
    // ... expand children
}
```

This is a ~15-line addition to the existing Phase C loop. The heap-empty check already exists (`heap.pop()` returns None), so this is an additional stopping condition.

### What Does NOT Map

| AutoTTS Concept | Why It Doesn't Apply |
|---|---|
| **BRANCH / PROBE actions** | Their branches are independent LLM reasoning chains. Our DDTree branches are token-level alternatives within a single forward pass — fundamentally different granularity |
| **Offline replay environment** | They pre-cache full reasoning traces. We don't have "cached token traces" — our marginals come from live transformer forward passes |
| **Coding agent search** | Their controller is Python rewritten by an LLM. Our controller is Rust compiled ahead-of-time. The search loop doesn't transfer |
| **Majority voting** | They aggregate answers across branches. We select the single best path via `extract_best_path_into` |
| **Answer pool** | They track a pool of completed answers. Our tree is a search over token sequences — there's no "completed answer" concept at the token layer |
| **EMA momentum gate (as-is)** | Their EMA operates on completion confidence across reasoning branches. We'd need a different signal (score gap in DDTree) |
| **Probe-age priority scheduling** | They allocate probe steps based on branch investment. Our DDTree uses best-first expansion — the heap already prioritizes the most promising branches |

### Why the Naive POC Is Wrong

A proposed POC suggested `prompt_context.contains("async")` to set `tree_budget`. This fails because:

1. **We don't have prompt strings at this layer.** DDTree operates on `&[&[f32]]` marginals, not text.
2. **The WasmPruner already IS the verifier.** `ScreeningPruner::relevance() == 0.0` already provides "early exit" — invalid branches never enter the heap.
3. **The heap-empty check already stops expansion.** When all branches are pruned, expansion terminates.
4. **String matching is the wrong abstraction.** Domain classification (anyrag Plan 005) provides a proper semantic signal, not keyword hacks.

---

## Application to microgpt-rs

### Direct Mappings

| Paper Concept | microgpt-rs Equivalent | Status |
|---|---|---|
| **Dynamic budget per query** | `DomainConfig.inference_budget` → `Config::with_overrides()` | ❌ Missing |
| **β parameterization** | `InferenceBudget::from_beta()` — single scalar → budget/lookahead/threshold | ❌ Missing |
| **Early exit (momentum gate)** | Confidence-gap early exit in DDTree Phase C | ❌ Missing |
| **Adaptive width (branch spawning)** | `tree_budget` already controls max nodes — equivalent to `max_branch_use` | ✅ Exists |
| **Branch pruning** | `ScreeningPruner::relevance() <= threshold` → branch dropped | ✅ Exists |
| **Best-first expansion** | DDTree Phase C heap-based expansion | ✅ Exists |
| **Terminal aggregation** | `extract_best_path_into()` — single best path | ✅ Exists |
| **Replay environment** | Bandit demos use deterministic environments for evaluation | ✅ Partial (demos only) |

### What Our System Already Does Better

1. **Token-level granularity.** AutoTTS operates at reasoning-chain level (coarse). We operate at token level (fine) via `ScreeningPruner`. Bad reasoning = sequence of bad tokens.

2. **Deterministic verification.** AutoTTS uses LLM confidence for gating (probabilistic). Our `WasmPruner` is compiled WASM — deterministic, sub-microsecond. Their paper's conclusion ("distill the reviewer") is what we already built.

3. **Built-in budget control.** `tree_budget` caps DDTree expansion. `screening_threshold` controls pruning aggressiveness. These are the two knobs AutoTTS would want.

4. **Trait-based separation.** `ConstraintPruner` (binary) and `ScreeningPruner` (graded) provide the same separation AutoTTS stumbled into (binary prune vs confidence-weighted continue).

### What to Build (Gap Analysis)

1. **`InferenceBudget` struct**: Optional overrides for `tree_budget`, `draft_lookahead`, `screening_threshold`, `temperature`. Lives in `DomainConfig` behind `#[serde(default)]`.

2. **`Config::with_overrides()`**: Clones self, applies non-None overrides. ~10 lines in `microgpt-rs/src/types.rs`.

3. **`InferenceBudget::from_beta()`**: Optional convenience — derive all params from single scalar. Our own β mapping, not CMC's.

4. **Confidence-gap early exit**: Add dominance check to DDTree Phase C. Track best vs second-best score gap. Stop when gap exceeds threshold for N consecutive iterations.

5. **Wire through ExpertBundle**: When router selects a domain, pass inference overrides alongside pruner and LoRA.

### Connection to Existing Plans

| Plan | Relationship |
|---|---|
| `riir-ai` Plan 023 (Prompt Router) | **Infrastructure consumer.** Router selects domain → we need budget overrides to flow with it |
| `anyrag` Plan 005 (Domain Classifier) | **Signal source.** `/classify/domain` determines which domain → which budget |
| `anyrag` Plan 007 (Catalog-Driven Shaping) | **Config host.** Already proposes `[domain.truncation]` and `[domain.reasoning]` — `[domain.inference]` is the natural addition |
| `microgpt-rs` Plan 021 (ScreeningPruner) | **Verifier.** Already provides graded relevance — the "reviewer" in AutoTTS terms |
| `microgpt-rs` Plan 030 (Multi-Armed Bandit) | **Adaptive learning.** BanditPruner learns across episodes — could learn optimal budget over time |

### System Architecture Mapping

```
AutoTTS Architecture:
  Query → Controller(β) → BRANCH/CONTINUE/PROBE/PRUNE/ANSWER → Pool → Majority Vote
         ↑                                                            ↓
         └──── Replay Environment (cached traces) ←──────────────────┘

Our Architecture:
  Prompt → anyrag /classify/domain → RouteDecision { domain }
         → ExpertRegistry.get_expert(domain) → ExpertBundle { pruner, lora, inference_budget }
         → Config::with_overrides(inference_budget)
         → DDTree Phase A (chain seed with screening)
         → DDTree Phase B (sibling + child seeding)
         → DDTree Phase C (best-first with early exit + screening)
         → extract_best_path_into()
         → Target model verify
```

The key difference: AutoTTS's "controller" is a Python function that makes branching decisions during LLM reasoning. Our "controller" is a `Config` struct that parameterizes DDTree search before it starts. Both achieve adaptive compute, but at different levels of the stack.

### The Latency / Cost Lesson

AutoTTS's discovery process costs $39.9 per run with 0 LLM calls during evaluation. This validates our architecture:

```
AutoTTS:  Full reasoning traces (expensive) → Replay (cheap) → Controller evaluation
Us:       Transformer forward pass (expensive) → DDTree search (cheap) → WasmPruner verify (cheap)
```

Both architectures separate the expensive operation (LLM inference) from the cheap adaptive decision (controller/search). Our WasmPruner is even cheaper than their replay — sub-microsecond vs table lookup.

---

## Key Takeaways

1. **Spend compute proportional to difficulty.** Easy problems (linear logic) need `tree_budget=16`. Hard problems (async/lifetimes) need `tree_budget=5000`. Domain classification provides the difficulty signal.

2. **β parameterization is elegant config design.** One scalar → all knobs via deterministic monotonic functions. Prevents inconsistent parameter combinations.

3. **Replay-based evaluation enables cheap search.** Separate the expensive operation (inference) from the cheap evaluation (lookup/verification). Our WasmPruner already does this.

4. **Momentum beats instantaneous.** CMC's key insight: don't stop on a single confidence spike. For DDTree, the analogous principle is: don't early-exit on a single dominant branch — wait for sustained dominance.

5. **Not all of AutoTTS transfers.** Their MDP formulation, coding agent search, and majority voting solve problems we don't have at the token level. The distillation is: **per-domain Config overrides + confidence-gap early exit**. Everything else is their domain, not ours.

---

## Citation

```bibtex
@article{zheng2026autotts,
  title  = {LLMs Improving LLMs: Agentic Discovery for Test-Time Scaling},
  author = {Zheng, Tong and Liu, Haolin and Huang, Chengsong and Bao, Huiwen and
            Zhang, Sheng and Liu, Rui and Dai, Runpeng and Chen, Ruibo and
            Liu, Chenxi and Xiong, Tianyi and Wu, Xidong and Zhang, Hongming and
            Huang, Heng},
  journal = {arXiv preprint},
  year    = {2026}
}