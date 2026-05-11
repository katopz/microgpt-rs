# Research: Reinforced Agent — Inference-Time Feedback for Tool-Calling Agents (15)

> Source: [Reinforced Agent: Inference-Time Feedback for Tool-Calling Agents](https://arxiv.org/pdf/2604.27233) by Anh Ta, Junjie Zhu, Shahin Shayandeh (Apple)
> Date: 2026-05-01 (published), distilled 2025-06

## Summary

Apple demonstrated that a specialized reviewer agent evaluating provisional tool calls **before execution** improves accuracy without retraining the base agent. The key architectural insight is **separation of concerns**: one agent executes, another reviews. The key measurement insight is **Helpfulness-Harmfulness metrics**: without quantifying how often the reviewer helps vs hurts, you cannot tell if intervention is net-positive.

Best results: +5.5% irrelevance detection (BFCL), +7.1% multi-turn (τ²-Bench). Reasoning models (o3-mini) achieve 3.1:1 benefit-to-risk ratio vs 2.1:1 for standard models (GPT-4o). Automated prompt optimization (GEPA) adds +1.5–2.8%. Progressive Feedback (iterative review loops) outperforms Best-of-N Selection/Grading by +3–8%.

---

## Core Concepts

### Inference-Time Feedback

Move evaluation from **post-hoc** (after execution) to **proactive** (before execution). A reviewer agent evaluates the base agent's provisional tool calls and either approves or provides feedback for revision. This avoids the **state recovery problem**: once a destructive action executes (e.g., deleting an alarm instead of updating it), self-correction requires maintaining previous state in context — prohibitively expensive in multi-turn scenarios.

### Three Collaboration Mechanisms

| Mechanism | Notation | How It Works |
|---|---|---|
| **Progressive Feedback** | `rN` (e.g., `r2`) | Iterative review loops. Reviewer evaluates → injects feedback as system message → base agent revises → repeat up to N loops or approval. |
| **Best-of-N Selection** | `sN` (e.g., `s5`) | Base agent generates N candidates at varying temperatures (0.3–1.0). Reviewer picks the best one. Single-shot. |
| **Best-of-N Grading** | `gN` (e.g., `g5`) | Same as selection, but reviewer assigns explicit scores (0.0–1.0) with rationales. Highest-scored candidate wins. |

Progressive Feedback substantially outperforms both Best-of-N approaches (+3–8% on average). Selection actually underperforms baseline on some domains.

### Helpfulness-Harmfulness Metrics

| Metric | Definition |
|---|---|
| **Helpfulness** | % of test cases where base agent is WRONG and reviewer CORRECTS it |
| **Harmfulness** | % of test cases where base agent is RIGHT and reviewer INTRODUCES error |
| **Benefit-to-Risk Ratio** | Helpfulness ÷ Harmfulness |

These metrics reveal whether feedback provides net positive value. The paper found:
- o3-mini reviewer: 36.8% helpful, 11.7% harmful → **3.1:1 ratio**
- GPT-4o reviewer: 34.9% helpful, 12.9% harmful → **2.7:1 ratio**
- After GEPA optimization with GPT-5 mini: further improved

### Over-Skepticism: The Primary Reviewer Error Mode

The reviewer incorrectly flags valid tool calls as "incomplete," expecting execution results or user-facing dialogue. But the reviewer evaluates **provisional** calls (before execution), so no results exist yet.

**Root cause:** Reviewer expects user-facing dialogue, but BFCL only evaluates tool call accuracy.

**Fix:** Explicit guideline:
> [CRITICAL] Tool-only responses are complete. Do not mark tool-only responses as incomplete for lacking user-facing answers, follow-up explanations, or final results.

This reduced redundant review loops from **23% → 8%**.

### Automated Prompt Optimization (GEPA)

GEPA (Genetic-Pareto Prompt Evolution) iteratively improves reviewer prompts:
1. Start with manually-engineered prompts (v2)
2. Collect failure cases where reviewer made incorrect judgments
3. Use reasoning model (GPT-5 mini) to reflect and propose improvements
4. Iterate until convergence

Result: v3-gepa prompts are **4.5× longer** (1,599 vs 358 tokens) with detailed error criteria, edge case handling, and error checklists. Gains: +1.5% relevance, +2.8% irrelevance.

Key: **only the reviewer's prompt is optimized**. The base agent's prompt remains unchanged. This is the modular advantage of separation of concerns.

### Reasoning Models as Reviewers

Reasoning models (o3-mini, GPT-5 mini) outperform standard models as reviewers in two key areas:
1. **Irrelevance detection**: o3-mini outperforms GPT-4o by +0.6% (91.0% vs 90.4%) — requires determining whether any tool can address the request
2. **Lower harmfulness**: o3-mini introduces fewer errors (11.7% vs 12.9%) — systematic verification reduces over-correction

### Latency Trade-offs

| Benchmark | Baseline | With Reviewer | Multiplier |
|---|---|---|---|
| BFCL (single-turn) | 1.27s | 7.87s | **6.2×** |
| τ²-Bench (multi-turn) | 158.7s | 384.3s | **2.4×** |

Multi-turn overhead is amortized across ~40 turns per episode. Reviewer call patterns: BFCL averages 1.33 reviewer calls/item; τ²-Bench averages 0.96 reviewer calls/turn (lower due to stateful nature reducing ambiguity in later turns).

### Error Distribution Analysis (τ²-Bench)

| Error Type | Baseline | With Reviewer | Δ |
|---|---|---|---|
| Policy Constraint Violation | 31% | 18% | **-13%** |
| Missing Context Awareness | 24% | 15% | **-9%** |
| Incorrect Tool Selection | 19% | 22% | +3% |
| Argument Errors | 16% | 18% | +2% |
| Over-verbalization | 10% | 27% | **+17%** |

Reviewer catches policy violations effectively but introduces over-verbalization errors (the over-skepticism problem recurring in multi-turn contexts).

---

## Experimental Results

### BFCL (Single-Turn, Stateless)

| Configuration | Simple | Multiple | Parallel | Par_Mult | Irrel. | Rel. Suite |
|---|---|---|---|---|---|---|
| 4o baseline | 92.4 | 92.8 | 93.0 | 85.2 | 84.9 | 90.9 |
| 4o-r5-4o-v1 | 92.8 | 93.0 | 92.5 | 87.5 | 89.6 | 91.4 |
| 4o-r2-5-mini-v2 | 92.4 | 93.8 | 92.2 | 85.5 | 87.6 | 91.0 |
| 4o-r2-5-mini-v3-gepa | **95.3** | **94.3** | **93.0** | 87.3 | **90.4** | **92.5** |

### τ²-Bench (Multi-Turn, Stateful)

| Configuration | Airline | Retail | Telecom | Average |
|---|---|---|---|---|
| 4o baseline | 42.0 | 62.9 | 41.2 | 48.7 |
| 4o-r5-4o-v1 | 40.7 | 62.6 | **64.0** | **55.8** |
| 4o-g5-4o-v2-tau | 46.7 | **65.8** | 48.8 | 53.8 |

Key finding: Progressive Feedback (r5) achieves highest average, but v1 prompts sometimes outperform v2 domain-specific prompts — manual tuning doesn't generalize across tasks.

### Helpfulness vs Harmfulness (BFCL Non-Live)

| Reviewer + Prompt | Rel. | Irrel. | Helpful | Harmful | Ratio |
|---|---|---|---|---|---|
| GPT-4o v1 | 89.5% | 90.0% | 30.2% | 14.2% | 2.1:1 |
| GPT-4o v2 | 91.0% | 90.4% | 34.9% | 12.9% | 2.7:1 |
| o3-mini v2 | **91.8%** | **91.0%** | **36.8%** | **11.7%** | **3.1:1** |

---

## Application to microgpt-rs

### Direct Mappings

| Paper Concept | microgpt-rs Equivalent | Status |
|---|---|---|
| Reviewer Agent (evaluate before execute) | `ScreeningPruner::relevance()` — graded [0.0, 1.0] per token | ✅ Strong |
| Hard rejection (binary) | `ConstraintPruner::is_valid()` — binary accept/reject | ✅ Strong |
| Progressive Feedback (rN) | `speculative_step()` → verify → reject → re-draft loop | ✅ Strong |
| Best-of-N Selection (sN) | `build_dd_tree_screened()` — tree search, scored branches | ✅ Strong |
| Over-skepticism mitigation | Plan 029 Task 7 — ownership boundary between pruners | ✅ Done |
| Distilled reviewer (small/fast) | `WasmPruner` — compiled deterministic validator | ✅ Strong |
| Runtime reviewer update | `HotSwapPruner` — blake3-checked .wasm reload | ✅ Done |
| Absorb + Compress | `AbsorbCompress` — promote stable low-Q to hard blocks | ✅ Done |
| Trial persistence | `TrialLog` — JSONL episode history | ✅ Done |
| **Helpfulness-Harmfulness metrics** | **Missing** — no benefit-vs-harm tracking | ❌ Gap |
| **Benefit-to-risk ratio** | **Missing** — no ratio computation | ❌ Gap |
| **Structured review loop with feedback injection** | **Missing** — PPoT rescues but doesn't carry rejection context between attempts | ❌ Gap |
| **Ratio-gated compression** | **Missing** — AbsorbCompress doesn't check reviewer quality | ❌ Gap |

### What Our System Does Better

1. **Token-level granularity**: The paper reviews at tool-call level (coarse). We review at token level (fine) via `ScreeningPruner`. This is a superset — a bad tool call is a sequence of bad tokens.

2. **Deterministic reviewers**: The paper uses LLMs as reviewers (o3-mini, GPT-4o) — probabilistic, expensive, slow. Our `WasmPruner` is compiled WASM — deterministic, sub-microsecond, zero-latency overhead. This is the "distilled reviewer" the paper says to aim for.

3. **Explicit ownership boundaries**: The paper discovered over-skepticism through failure analysis. We preempted this via Plan 029 Task 7 — `ConstraintPruner` owns syntax, `ScreeningPruner` owns semantics, never compete.

### What to Build (Gap Analysis)

1. **ReviewMetrics**: Atomic counters tracking helpful/harmful/both_correct/both_wrong per pruner session
2. **benefit_ratio()**: Computed metric = helpfulness ÷ harmfulness, returns `f64::INFINITY` if harmfulness is zero
3. **ReviewLoopConfig**: Structured outer loop around PPoT rescue — max N loops, carry rejection reason between attempts, break if ratio below threshold
4. **Ratio-gated AbsorbCompress**: Only promote stable low-Q arms to hard blocks when reviewer's benefit ratio exceeds configurable threshold (default 2.0)
5. **ReviewStrategy enum**: Paper notation (`rN`/`sN`/`gN`) as config enum for future review mode switching

### The Latency Lesson

The paper's 6.2× overhead on single-turn comes from adding an LLM call as reviewer. Our architecture avoids this entirely:

```
Paper:  Tool call → LLM reviewer (7.87s) → Execute
Us:     Token     → ScreeningPruner (<1µs) → Accept/Reject
```

The paper's conclusion ("distill the reviewer into a smaller, faster model") is exactly what `WasmPruner` already is. The 2.4× multi-turn overhead is acceptable because it amortizes — our token-level overhead amortizes across hundreds of tokens per step.

### System Architecture Mapping

```
Paper's Architecture:
  Base Agent (GPT-4o) → Reviewer Agent (o3-mini) → Execute
                      ← Feedback injected ←

Our Architecture:
  DDTree Draft → ScreeningPruner (WasmPruner) → Target Model Verify
               ← RejectionReason ←
               → PPoT Rescue (CPU resample) →
               ← SessionKnowledge (rejection memory) ←
```

The key difference: our feedback loop is deterministic (WASM → constraint check → accept/reject), not probabilistic (LLM → "I think this is wrong" → regenerate). Deterministic feedback is faster, more reproducible, and doesn't introduce the over-skepticism error mode.

### Connection to Plan 033 (Bomberman Arena)

The paper's multi-agent separation (execution vs review) maps to the Bomberman arena's agent architecture:

```
Bomberman Agent = BanditPruner<SlotScreeningPruner>
  SlotScreeningPruner = "reviewer" (domain relevance for slot symbols)
  BanditPruner = "base agent" (adaptive exploration/exploitation)
  
  ReviewMetrics tracks: did bandit's adaptation help or hurt vs raw screening?
```

The helpfulness-harmfulness metrics answer: "is the bandit layer actually improving on top of the domain screener, or just adding noise?"

---

## Key Takeaways

1. **Measure before trusting.** A reviewer that helps 36% of the time but hurts 11% is net-positive (3:1). One that helps 15% and hurts 20% is net-negative. Without metrics, you can't tell.

2. **Progressive Feedback beats Best-of-N.** Iterative review with targeted feedback outperforms generating multiple candidates and picking the best. Our PPoT rescue should carry rejection context between loops.

3. **Reasoning about reasoning helps.** Models that verify systematically (o3-mini) make better reviewers than models that react quickly (GPT-4o). Our compiled validators are the extreme of this: verification is exact, not approximate.

4. **Over-skepticism is the default failure mode.** Reviewers tend to reject valid outputs. The fix is explicit scoping rules. Our pruner ownership boundary (Plan 029) prevents this class of error.

5. **Optimize the reviewer, not the base.** The paper improved results by improving only the reviewer's prompt (GEPA). Our equivalent: improve the `.wasm` validator without touching the inference engine.

---

## Citation

```bibtex
@misc{ta2026reinforced_agent,
  title = {Reinforced Agent: Inference-Time Feedback for Tool-Calling Agents},
  author = {Ta, Anh and Zhu, Junjie and Shayandeh, Shahin},
  year = {2026},
  month = apr,
  eprint = {2604.27233},
  archiveprefix = {arXiv},
  primaryclass = {cs.AI},
  note = {Apple}
}