# Research: Test-time Recursive Thinking (TRT)

**Date:** 2025-06
**Status:** Research → Verdict
**Context:** microgpt-rs speculative decoding + PPoT (Plan 026) + ConstraintPruner architecture
**Paper:** "Test-time Recursive Thinking: Self-Improvement without External Feedback" (arXiv:2602.03094) — Zhuang, Singh, Liu, Shen, Zhang, Shang, Gao, Chen (Microsoft Research / UC San Diego)

---

## TL;DR

LLMs can self-improve at test time by iterating three stages — **Generate** (with strategies + knowledge), **Select** (self-rank), **Reflect** (extract failure insights) — without external feedback. Open-source models reach 100% on AIME-25/24; o3 improves +14.8pp on LiveCodeBench hard problems. The key mechanisms are: (1) per-rollout strategy conditioning for diverse exploration, (2) compressed "don'ts" knowledge accumulation (<1.5% of context after 64 rounds), and (3) self-verification via mutual exclusivity (math) or test execution (code).

Applied to microgpt-rs after PPoT: the highest-value distillation is **rejection knowledge accumulation** — when PPoT rescue resamples and the `ConstraintPruner` rejects variants, record structured "don't" patterns that bias future resampling within the same generation session. This makes PPoT adaptive rather than random. Secondary: cycle `TokenRule` strategies across PPoT's m samples so each explores a distinct hypothesis.

---

## The Problem: Exploration vs Verification

### The Dilemma

Parallel sampling methods (self-consistency, best-of-N) generate independent traces. Each trace is unaware of insights from other attempts. Models repeat mistakes and fail to build on partial successes.

Exploration without verification → noise. Verification without exploration → stagnation.

TRT formalizes this as: **strategic exploration** (expand solution space) + **self-guided verification** (select without ground truth). Neither suffices alone.

### Current Paradigm (Parallel Sampling)

```
for j in 1..k:
    rollout_j = LLM(prompt)              # independent, no shared knowledge
    result_j = execute(rollout_j)

answer = majority_vote(results)           # or best-of-N with reward model
```

Each rollout is independent. If rollout 3 discovers "off-by-one in the loop bound", rollouts 4-k don't benefit.

### The Key Insight

Knowledge accumulates across iterations. After comparing failed vs successful rollouts, the model extracts compressed failure patterns that guide subsequent attempts:

```
K = []                                    # knowledge list (compressed "don'ts")
for t in 1..T:
    for k in 1..K:
        strategy_k = design_strategy(K)   # unique per rollout
        rollout_k = LLM(prompt, K, strategy_k)
    best = select(rollouts)               # self-judgment
    for each non-best rollout:
        K.append(extract_insight(rollout, best))  # "don't do X"
```

---

## The Core Mechanisms

### 1. Knowledge Representation: "Don'ts"

Knowledge entries are **negative constraints** — what not to do. The paper shows (Figure 8) that recording failures ("don'ts") yields higher accuracy than recording successes ("dos"):

```
Don't: "The naive greedy ordering fails when interleaving neutral flips matters"
Don't: "Don't use int division // for non-integer expected results"
Don't: "Avoid sorting only by cost — track running prefix sums for cumulative effects"
```

**Why "don'ts" beat "dos":** Successes overfit to specific solution steps. Failures abstract across problem instances. A single "don't" can prevent an entire class of mistakes.

**Compactness:** Knowledge list stays under 1.5% of context after 64 rounds for math, under 0.35% after 8 rounds for code (Figure 7). This is because only distilled insights are stored, not full traces.

### 2. Per-Rollout Strategy Design

Each rollout receives a **unique strategy** that guides exploration:

| Domain | Strategy Examples |
|---|---|
| Code | "Use dynamic programming" vs "Use greedy" vs "Use divide-and-conquer" |
| Code | "Optimize for memory" vs "Optimize for speed" |
| Math | "Algebraic manipulation" vs "Geometric intuition" |
| Math | "Work backwards" vs "Case analysis" |

Strategies are designed by the model itself, conditioned on accumulated knowledge. This ensures rollouts explore **complementary** regions of the solution space rather than redundant attempts.

**Key finding:** Strategy switches occur more frequently after failure (82%) than success (74%) (Figure 5b). Models naturally adapt exploration when they sense they're on the wrong track.

### 3. Selection Without Ground Truth

TRT lacks ground truth. Selection must exploit domain structure:

**Math — Mutual Exclusivity:** For problems with a single correct integer answer (AIME), correct reasoners converge on the same answer while incorrect ones disperse. If 8 of 10 rollouts agree on answer 42, that's strong signal.

**Code — Execution-Based Self-Verification:** The model generates unit tests from its understanding of the problem. Each candidate solution is executed against these tests. Solutions passing more tests rank higher.

**Ablation (Table 1):** Test execution contributes 7.4pp improvement for o4-mini. Strategy contributes 3.0pp. Both are necessary.

### 4. The Algorithm

```
Algorithm 1: Test-time Recursive Thinking
Input: Problem P, rounds T, rollouts per round K
Output: Selected solution r*

1. K ← ∅                              # knowledge list
2. S ← ∅                              # solution pool
3. for t = 1 to T:
4.   for k = 1 to K:
5.     s_k ← design_strategy(K)       # unique strategy per rollout
6.     r_k ← LLM(P, K, s_k)          # generate with knowledge + strategy
7.     S ← S ∪ {r_k}
8.   r* ← SELECT(S)                   # self-judgment ranking
9.   for each r in S where r ≠ r*:
10.    K ← K ∪ {insights from comparing r to r*}  # extract "don'ts"
11. return r*
```

---

## Experimental Results

### Mathematical Reasoning (AIME-25/24)

| Model | Method | Accuracy |
|---|---|---|
| gpt-oss-120b | Majority@64 | 96.7% |
| gpt-oss-120b | TRT 64 rounds | **100%** |
| Qwen3-235B | Majority@64 | 93.3% |
| Qwen3-235B | TRT 64 rounds | **100%** |

First time open-source models achieve 100% on AIME. Monotonic improvement across rounds confirms knowledge accumulation works.

### Code Generation (LiveCodeBench v6 Hard)

| Model | Method | Accuracy | Δ |
|---|---|---|---|
| o4-mini (High) | Baseline (pass@1) | 63.5% | — |
| o4-mini (High) | RSA 8 rounds | 70.4% | +6.9 |
| o4-mini (High) | **TRT 8 rounds** | **73.9%** | **+10.4** |
| o3 (High) | Baseline (pass@1) | 57.1% | — |
| o3 (High) | RSA 8 rounds | 69.7% | +12.6 |
| o3 (High) | **TRT 8 rounds** | **71.9%** | **+14.8** |

TRT outperforms RSA by 2.2-3.5pp at equivalent compute.

### Exploration Efficiency (Figure 4)

At equivalent sample counts, TRT's strategic planning improves pass@k by **2-7pp** over independent sampling across both models. Accumulated knowledge + per-rollout strategy genuinely improves solution space coverage.

### Depth vs Breadth (Figure 6)

K=2, K=4, K=8 all converge to similar cumulative best accuracy (78-82%). **Iterative knowledge accumulation across rounds matters more than parallel exploration within rounds.** Depth beats breadth.

### Knowledge Categories (Figure 9)

For code generation, accumulated knowledge breaks down as:

| Category | Share |
|---|---|
| Performance | 23% |
| Edge Cases | 19% |
| Indexing | 18% |
| Other | 14% |
| Bug Fixes | 10% |
| Algorithm | 6% |
| I/O Format | 7% |
| Numerical | 3% |

Higher-level execution insights (performance, edge cases, indexing) dominate over low-level syntax corrections.

---

## Mapping to microgpt-rs

### What Already Exists (after PPoT Plan 026)

| TRT Component | microgpt-rs Equivalent | Status |
|---|---|---|
| Multiple rollout generation | PPoT m=10 CPU resampled variants | ✅ Plan 026 |
| Strategy conditioning | `TokenRule` enum (Digit, Compare, Arithmetic, Augment, All) | ✅ Plan 026 |
| Constraint verification | `ConstraintPruner` / `ScreeningPruner` / `WasmPruner` | ✅ Done |
| Rejection signals | `CompilerFeedback` with `ErrorKind` + `suggestion` | ✅ Done |
| Entropy-based position selection | `identify_high_entropy_positions()` | ✅ Plan 026 |
| Resampling core | `ppot_resample()` / `ppot_resample_different_value()` | ✅ Plan 026 |

### What's Missing

| TRT Component | Gap | Value |
|---|---|---|
| Compressed rejection knowledge accumulation | No mechanism to record "resampling position X with rule Y failed because Z" | **High** |
| Knowledge-informed position biasing | PPoT identifies high-entropy positions but doesn't learn which positions are worth resampling | **High** |
| Per-sample strategy cycling | PPoT uses single `TokenRule` for all m samples | **Medium** |
| Self-consistency ranking | PPoT returns first valid variant, doesn't rank by agreement | **Medium** |
| Knowledge pruning (evict stale entries) | No bounded knowledge buffer | **Low** |

### Natural Integration Point (After Plan 026)

```
Current (Plan 026):
  DDTree rejects all → PPoT Rescue → identify H-positions → resample m variants
                  → screen via ConstraintPruner → return first valid

With TRT (Plan 027):
  DDTree rejects all → PPoT Rescue → identify H-positions (biased by knowledge)
                  → resample m variants (each with different TokenRule strategy)
                  → screen via ConstraintPruner
                  → rank by self-consistency (agreement count)
                  → return best variant
                  → record rejection insights into session knowledge
                  → knowledge biases next rescue's position selection
```

---

## What to Adopt

### 1. Rejection Knowledge Accumulation (HIGH VALUE)

When PPoT rescue runs and variants get rejected by `ConstraintPruner`, record structured insights:

```rust
struct RejectionInsight {
    position: usize,           // which token position was resampled
    rule: TokenRule,           // what strategy was used
    original_token: usize,     // what was there before
    attempted_tokens: Vec<usize>, // what was tried
    error_kind: ErrorKind,     // why it was rejected
    entropy_at_position: f32,  // how uncertain the model was
}
```

Accumulated across decoding steps in a session, this forms a compact "don'ts" list. The next time PPoT rescue activates, it can:
- **Skip** positions where all `TokenRule` variants were rejected (don't waste CPU)
- **Prioritize** positions where past resampling succeeded (bias toward known-good perturbation points)
- **Deprioritize** `ErrorKind` patterns that never lead to valid paths

TRT shows this knowledge stays tiny (<1.5% of context). For token-level decoding, it's even smaller — maybe 10-50 insights per full generation.

### 2. Per-Sample Strategy Cycling (MEDIUM VALUE)

Instead of PPoT using a single `TokenRule` for all m=10 samples, cycle strategies:

```
Sample 1: TokenRule::Digit        (try different constants)
Sample 2: TokenRule::Arithmetic   (try different operators)
Sample 3: TokenRule::Compare      (try different comparisons)
Sample 4: TokenRule::All          (unrestricted)
Sample 5-10: repeat or mix
```

TRT shows this improves pass@k by 2-7pp (Figure 4) because each sample explores a distinct failure hypothesis. Simple implementation: `strategies.cycle()` over m samples.

### 3. Self-Consistency Ranking (MEDIUM VALUE)

When multiple PPoT variants pass the `ConstraintPruner`, rank by **agreement**:

```
If 3 variants produce the same token sequence (post-resample convergence):
  → high confidence, select this path
  
If all 10 variants produce different sequences:
  → low confidence, fall back to greedy
```

This is TRT's mutual exclusivity insight applied at token level. Cost: O(m² × lookahead) comparison — negligible since m=10 and lookahead=5-8.

### 4. Knowledge-Informed Entropy Threshold (LOW VALUE)

TRT's models adapt strategy after failure (82% switch vs 74% on success). Approximate this by adjusting the entropy threshold dynamically:

```
If last rescue attempt succeeded: threshold stays high (only resample very uncertain positions)
If last rescue attempt failed: threshold drops (try more positions, explore wider)
```

This is a 5-line change to `PpotConfig` that captures TRT's adaptive exploration spirit.

---

## What NOT to Adopt

| TRT Concept | Why Skip |
|---|---|
| Multi-round LLM calls (T=64) | microgpt-rs is a single-pass token decoder, not a chat loop. No LLM API per round. |
| Model-generated strategy prompts | We don't have a "prompt the model to design strategies" loop. Static `TokenRule` enums are sufficient for token-level rescue. |
| Test execution for selection | `WasmPruner` already provides deterministic verification. Generated tests would be slower and less reliable. |
| Cross-problem knowledge | TRT accumulates per-problem. We accumulate per-generation-session. Different scope, same principle. |
| Sequential editing vs regeneration | TRT Appendix A10 shows editing beats rewriting. Irrelevant for token-level speculative decoding where we resample individual tokens. |
| Knowledge pruning (remove stale entries) | Our sessions are short (one generation). Context overflow isn't a concern. |

---

## Risks & Caveats

1. **Scope mismatch:** TRT operates at the problem level (full programs). We operate at the token level (individual tokens in a sequence). "Don'ts" about token resampling may be less informative than "don'ts" about algorithm choice.

2. **Cold start:** The first rescue attempt has no accumulated knowledge. Benefits only appear after several failed rescues within the same generation. Short generations may not accumulate enough.

3. **Overfitting risk:** If knowledge is too specific ("position 5 with token 1234 failed"), it won't generalize to other positions. Insights must be abstracted to the rule level ("Digit positions near operators are bad resampling candidates when bracket pruner rejects").

4. **Measurement difficulty:** The improvement is on top of PPoT's improvement. We need clean A/B: bench 026 first, then 027 on top, to isolate the TRT contribution.

5. **Independence assumption persists:** TRT's "don'ts" work because full-program resampling can test correctness end-to-end. Token-level "don'ts" only tell us about local constraint violations, not global correctness.

---

## Verdict: Adopt (Plan 027, After 026 Baseline)

TRT's **"don'ts" knowledge accumulation** is the highest-value distillation. The mechanism is simple (record rejection patterns), compact (<1.5% context proven), and effective ("don'ts" > "dos" proven). It makes PPoT adaptive rather than random within a generation session.

**Execution plan:** Implement Plan 026 first. Benchmark PPoT rescue acceptance rate. Then implement Plan 027 (adaptive rescue with rejection memory). Bench again. The delta between 027 and 026 isolates the TRT contribution.

**Deferred:** Per-sample strategy cycling is nice-to-have (add to 027 if benchmarks justify). Self-consistency ranking is low priority (most rescues produce 0-2 valid variants, not enough for meaningful voting).

---

## References

- "Test-time Recursive Thinking" (arXiv:2602.03094) — Zhuang et al.
- TRT Code: https://github.com/microsoft/TRT
- PPoT Research: `.research/11_PPoT_Probabilistic_Programs_of_Thought.md`
- PPoT Plan: `.plans/026_ppot_logit_resampling.md`
- Self-Consistency (Wang et al. 2022): arXiv:2203.11171
- Recursive Self-Aggregation (Venkatraman et al. 2025): arXiv:2509.26626
- Screening Pruner Research: `.research/07_Screening_Absolute_Relevance.md`
