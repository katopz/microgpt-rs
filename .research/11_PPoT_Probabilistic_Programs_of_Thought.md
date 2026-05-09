# Research: Probabilistic Programs of Thought (PPoT)

**Date:** 2025-06
**Status:** Research → Verdict
**Context:** microgpt-rs speculative decoding + DDTree + ConstraintPruner architecture
**Paper:** "Probabilistic Programs of Thought" (arXiv:2604.17290) — Poorva Garg et al. (UCLA / Allen AI)

---

## TL;DR

After an LLM generates a program (1 GPU pass), the **next-token logits are already available but discarded**. PPoT identifies high-entropy "key tokens" (digits, operators, brackets), converts them into random variables parameterized by those saved logits, and samples exponentially many variant programs using **only CPU** — no additional GPU forward passes. Result: **2-7% accuracy gain with m=5 CPU-only samples, at near-zero compute cost**.

Applied to microgpt-rs: the DFlash marginals in `SpeculativeContext` already capture the expensive part. The missing piece is a ~200-300 line module that identifies high-entropy positions and resamples from saved distributions on CPU, verified through existing `ConstraintPruner` / `ScreeningPruner`.

---

## The Problem: GPU Compute Waste

### Current Paradigm (Programs-of-Thought / Best-of-N)

To get k samples from an LLM, you need k full GPU forward passes:

```
for j in 1..k:
    for i in 1..n:
        x_i = sample(P_M(· | x_{<i}, prompt))   # GPU forward pass
    program = parse(x_1, ..., x_n)
    result = execute(program)
```

Each generation discards the full next-token distribution after sampling a single token. For 20 samples on a 7B model, that's 20 × ~100ms = 2 seconds of GPU time.

### The Key Insight

The LLM's next-token distribution already encodes uncertainty. When the model outputs `"total = 2{0}0"` with entropy concentrated on that single digit, the full distribution over `{0-9}` is available in the logits. PPoT reuses this information:

```
# After one LLM generation:
logits[i]  ← saved at position i  (already computed, usually discarded)
entropy[i] = -Σ p(x) * log(p(x))  # identify uncertain positions

# For each of m cheap samples (CPU only):
for each high-entropy position i:
    resample x_i from saved logits[i]  # Categorical sample, no GPU
assemble new program from resampled tokens
```

---

## The Mathematical Core

### LLM as a Probabilistic Program

The LLM induces a distribution over programs:

```
P_code(X_1, ..., X_n, Prog, Exc | prompt) = ∏_{i=1}^{n} P_M(X_i | X_{<i}, prompt)
```

Each token X_i is a categorical random variable parameterized by the next-token distribution. Standard decoding samples concretely; PPoT keeps the distributions.

### PPoT Factorization

Let L be the set of token indices treated as random variables (high-entropy positions). The key equation:

```
P_code = ∏_{i∉L} P_M(X_i | X_{<i}, prompt)    ← sampled from LM (GPU, done once)
        × ∏_{j∈L} P_M(X_j | X_{<j}, prompt)    ← probabilistic program (CPU, cheap)
```

The first product is fixed after the initial generation. The second product is the "probabilistic program" — a factored distribution that can be sampled thousands of times on CPU.

### Soundness (Assumption 1)

For PPoT samples to converge to the true LLM distribution, resampled tokens must be independent of succeeding tokens given preceding tokens:

```
∀j ∈ L, j < i ≤ n: P_M(X_i | X_{<i}) = P_M(X_i | X_{j}=x'_j, X_{<j}, X_{j+1..i-1})
```

This is approximately true for digits and operators in isolated expressions. It's violated when resampling changes the parse tree (e.g., changing a `(` to `)` cascades through all subsequent tokens). The paper shows empirical convergence despite this.

### Different-Value Constraint

To avoid resampling the original LLM output, PPoT conditions on not reproducing seen samples:

```
P_sample(C_i | not-seen) = [P(C_i) - P(C_i, seen)] / [1 - P(seen)]
```

This is tractable because the factored distribution admits trivial sampling without replacement.

---

## What PPoT Actually Does

### Algorithm 2 (Core)

```
Input: prompt t, sequence length n, LLM samples k, PPoT samples m
Output: k × m total samples

1. for j in 1..k:                        # k GPU generations
2.   for i in 1..n:
3.     x_i = sample(P_M(· | x_{<i}, t))  # GPU forward pass
4.     probs[i] = P_M(· | x_{<i}, t)     # SAVE the distribution
5.   prog = parse(x_1, ..., x_n)
6.   L = token_analysis(prog)             # identify high-entropy positions
7.   PP = compile(prog, probs[L])         # build probabilistic program
8.   for i' in 1..m:                      # m CPU-only samples
9.     prog_new = sample(PP)              # cheap categorical resampling
10.    exc = execute(prog_new)
11.    samples.append((prog_new, exc))
12. return samples
```

### Token Analysis Rules

PPoT defines domain-specific support sets for random variables:

| Rule | Support Set | Example |
|---|---|---|
| `digit` | `{0, 1, 2, ..., 9}` | `total = 2{0}0` → `{1}00` |
| `compare` | `{==, >, <, !=, <=, >=}` | `if x {>} 5` → `if x >= 5` |
| `arithmetic` | `{+, -, *, /, //, **}` | `y = x {-} 3` → `y = x + 3` |
| `augment` | `{+=, -=, *=, /=, //=}` | `x {+=} 1` → `x -= 1` |

Each rule narrows the support to semantically meaningful alternatives, making resampling more efficient than naive full-vocab resampling.

### Entropy-Based Position Identification

The paper uses regex rules on decoded token strings. The entropy of each position determines uncertainty:

```
H(i) = -Σ_{x ∈ vocab} P(x) × log(P(x))
```

High entropy → candidate random variable. Low entropy → keep fixed.

### Subset Sampling (for CRUXEval)

For structured output generation (lists, strings), PPoT uses suffix-masked sequential resampling:

```
At position i, only tokens appearing in target_tokens[i:] are allowed.
If sampled token matches one later in sequence, delete intermediate tokens.
Example: [1, 2, 3] → at position 1, comma could be ] → [1, 2] (valid shorter list)
```

---

## Experimental Results

### GSM8k (Math Reasoning)

| Model | k LLM samples | m PPoT samples | Accuracy | Δ |
|---|---|---|---|---|
| Qwen2.5 0.5B | 20 | 0 | ~72% | baseline |
| Qwen2.5 0.5B | 20 | 5 | ~77% | +5% |
| Qwen2.5 0.5B | 8 | 20 | ~76% | matches 20 LLM-only |
| Qwen2.5 3B | 20 | 5 | ~89% | +4% |
| Qwen2.5 7B | 20 | 5 | ~93% | +2% |

**Key result**: With m=5 PPoT samples, 8 LLM samples match 20 LLM-only samples' accuracy.

### Compute Efficiency

- PPoT compilation + sampling: **<1% of total wall-clock time**
- Runtime curves for m=0 and m=20 are nearly indistinguishable
- Effective surplus: m=20 PPoT ≈ 39 additional LLM samples (for k=20)
- Even m=1 PPoT ≈ 8 additional LLM samples

### Scaling Law

Error follows a power law: `1 - acc(k; m) = a_m × k^{-b_m}`

PPoT increases the exponent b_m (not just intercept), meaning error decreases faster as sampling scales. With m=20: b increases from 0.58 to 0.74.

---

## Qualitative Examples

### Digit Fix (GSM8k)
```
LLM:   total_feed_needed = 20 * 3 * 3     # wrong: 180, answer should be 20
PPoT:  total_feed_needed = 20 * 3 * 1     # resampled 3→1, correct: 60
```

### Operator Fix (GSM8k)
```
LLM:   amy_age = corey_age + 2            # wrong direction
PPoT:  amy_age = corey_age - 2            # resampled +→-, correct
```

### Integer Division Fix (GSM8k)
```
LLM:   sheets = pages / 2                 # wrong for 32 pages → 16.0
PPoT:  sheets = pages // 4                # resampled /→// and 2→4, correct: 8.0
```

### Array Length Fix (Plot2Code)
```
LLM:   data1 = np.random.randn(5, 50)     # wrong dimension → runtime error
PPoT:  data1 = np.random.randn(6, 50)     # resampled 5→6, renders correctly
```

---

## Mapping to microgpt-rs

### What Already Exists

| PPoT Component | microgpt-rs Equivalent | Status |
|---|---|---|
| Next-token logit capture | `DFlash` marginals in `SpeculativeContext::marginals_flat` | ✅ Done |
| Categorical sampling | `sample_from_distribution()` in `src/speculative/sampling.rs` | ✅ Done |
| Residual distribution | `sample_residual_distribution()` | ✅ Done |
| Constraint verification | `ConstraintPruner` / `ScreeningPruner` / `WasmPruner` | ✅ Done |
| Tree-based path exploration | `DDTree` with best-first search | ✅ Done |
| Path-aware token tracking | `TreeNode::parent_path` bitfield + `extract_parent_tokens()` | ✅ Done |

### What's Missing

| PPoT Component | Gap | Integration Point |
|---|---|---|
| Entropy calculation per position | Not implemented | New function after DFlash |
| High-entropy position identification | Not implemented | New `identify_rv_positions()` |
| Logit-parameterized CPU resampling | Not implemented | New `ppot_resample()` |
| Token type rule categories (digit, op, etc.) | Not implemented | New `TokenRule` enum |
| Different-value constraint sampling | Partially done via residual sampling | Leverage existing |
| Subset resampling (CRUXEval-style) | Not implemented | Future, for structured output |

### Natural Integration Point

```
Current:  DFlash → DDTree → Verify → Accept/Reject
With PPoT: DFlash → DDTree → Verify → Accept/Reject
                ↓
          Calculate per-position entropy
                ↓
          Identify high-entropy positions (TokenRule-aware)
                ↓
          Resample m=5-20 CPU variants from saved marginals
                ↓
          Feed each through ConstraintPruner / ScreeningPruner
                ↓
          Merge valid variants into DDTree or return as bonus paths
```

Or simpler — **post-verification rescue**: when verifier rejects all DDTree paths, use PPoT resampling on highest-scoring rejected path to find cheap alternatives instead of falling back to greedy.

---

## Key Design Decisions for Integration

### 1. Where to Compute Entropy
After `DFlash` populates `sctx.marginals_flat`, entropy is trivial:
```rust
fn token_entropy(probs: &[f32]) -> f32 {
    probs.iter().filter(|&&p| p > 0.0).map(|&p| -p * p.ln()).sum()
}
```
Cost: O(vocab_size × lookahead_steps) — negligible vs GPU forward pass.

### 2. Token Type Rules as ConstraintPruner
The `digit`, `arithmetic`, `compare`, `augment` support sets can be a new `ScreeningPruner` that returns `relevance = 0.0` for tokens outside the rule's support. This reuses the entire existing pruning infrastructure.

### 3. Different-Value via Residual Sampling
`sample_residual_distribution(p, q, ...)` already samples from `max(0, p - q)`. Setting `q` to a Kronecker delta at the original token's position implements the different-value constraint naturally.

### 4. No Gumbel-Max Needed
PPoT uses Gumbel-max sampling because PyTorch makes it convenient. Rust's `sample_from_distribution()` with CDF-based sampling is equivalent and already battle-tested in microgpt-rs.

### 5. Zero-Alloc Compatibility
All resampling can use the existing `SpeculativeContext` scratch buffers (`residual_buf`, `probs_buf`). No new allocations in the hot path.

---

## Risks & Caveats

1. **Independence assumption violation**: Resampling a token without updating succeeding tokens breaks autoregressive dependencies. Works for isolated digits/operators, fails for structural changes (brackets, keywords that reshape parse tree).
2. **Tokenizer dependence**: PPoT's regex-based position identification assumes subword tokenization where digits/operators are separate tokens. BPE tokenizers may merge `"200"` into a single token, making per-digit resampling impossible.
3. **Support set curation**: The `digit`, `arithmetic`, etc. rule sets are manually defined per domain. Generalizing requires either heuristics or learning.
4. **Marginal quality**: DFlash produces independent marginals (not autoregressive). PPoT resampling on top of marginals compounds the independence assumption.
5. **Diminishing returns with good models**: Larger models (7B+) have lower entropy at key positions, so PPoT's advantage shrinks. The paper shows +2% for 7B vs +7% for 0.5B.

---

## Verdict: Adopt (Targeted)

PPoT's core insight — **reusing discarded next-token distributions for cheap CPU resampling** — is architecturally clean and fits naturally into microgpt-rs's DFlash → DDTree pipeline. The implementation cost is low (~300 lines) and the existing `ConstraintPruner` / `ScreeningPruner` infrastructure handles verification.

**Adopt for the "post-DDTree rescue" use case**: when speculative decoding fails (all paths rejected), try PPoT resampling before falling back to greedy. This is the highest-ROI integration point because:
1. It only activates when needed (no overhead on success path)
2. The marginals are already computed
3. CPU resampling is essentially free
4. Any valid path found is a pure win over greedy fallback

**Defer full integration** (PPoT as primary sampling strategy) until benchmarks show the rescue path is insufficient. The independence assumption is a real concern for autoregressive models, and the paper's GSM8k results rely on Python code generation where token independence is more plausible.

---

## References

- "Probabilistic Programs of Thought" (arXiv:2604.17290) — Garg, Geh, Israel, Millstein, Richardson, Van den Broeck
- PPoT Reference Implementation: `raw/PPoT/ppot/` in this repo
- microgpt-rs DFlash: `src/speculative/dflash.rs`
- microgpt-rs DDTree: `src/speculative/dd_tree.rs`
- microgpt-rs Sampling: `src/speculative/sampling.rs`
- Screening Pruner Research: `.research/07_Screening_Absolute_Relevance.md`
- Leviathan Speculative Decoding: `.research/02_Fast_Inference_from_Transformers_via_Speculative_Decoding.md`
