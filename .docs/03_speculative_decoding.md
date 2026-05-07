# mini-dllm: Speculative Decoding Engine

## Pipeline Overview

```
Token + Position
       │
       ▼
  DFlash Predict ──→ Marginal Distributions [depth][vocab]
       │
       ▼
  DDTree Build ──→ Candidate Token Tree (best-first search)
       │                    │
       │              ConstraintPruner (optional)
       │              filters invalid branches
       ▼                    ▼
  SpeculativeVerifier
  ├─ SimulatedVerifier: DDTree path + acceptance rate
  └─ LeviathanVerifier: real p/q rejection + residual + bonus token
       │
       ▼
  Accepted Tokens (1 to draft_lookahead+1)
```

---

## DDTree (`speculative/dd_tree.rs`)

### TreeNode

```rust
pub struct TreeNode {
    pub score: f32,         // log-probability
    pub depth: usize,       // position in lookahead
    pub token_idx: usize,   // token ID
    pub parent_path: u128,  // 16 bits per depth, LSB-first
}
```

**Path encoding:** 16-bit slots per depth level → max token ID 65535, max depth 8. The `parent_path` field encodes the entire ancestry in a single `u128`, enabling zero-alloc path reconstruction.

- `extract_parent_tokens(path, n) → Vec<usize>` — decodes the path into a vector of token IDs.
- `extract_parent_tokens_into(path, n, buf)` — zero-alloc variant that writes into a pre-allocated buffer.

### Tree Building

| Function | Description |
|----------|-------------|
| `build_dd_tree(marginals, config) → Vec<TreeNode>` | Standard best-first tree construction from marginal distributions. |
| `build_dd_tree_pruned(marginals, config, pruner, chain_seed) → Vec<TreeNode>` | Best-first with constraint pruning and optional chain-seed backbone. |
| `merge_retrieved_branches(tree, marginals, config, retrieved, rest_weight)` | Inject REST-retrieved token sequences into the tree with blended scores. |

**Chain-seed mode:**
1. **Phase A** — builds a greedy argmax backbone (chain) by always selecting the highest-probability token at each depth.
2. **Phase B** — expands branch nodes from every chain node using the remaining budget, allowing exploration of alternative continuations.

**Budget:** `config.tree_budget` caps the total number of nodes in the tree, ensuring bounded memory and compute.

### TreeBuilder (zero-alloc)

```rust
pub struct TreeBuilder {
    heap: BinaryHeap<TreeNode>,
    tree: Vec<TreeNode>,
    chain_nodes: Vec<TreeNode>,
    chain_parent_tokens: Vec<usize>,
}
```

- Pre-allocated once, cleared via `clear()` (reuses existing capacity across calls).
- `build(&mut self, marginals: &[&[f32]], config, pruner, chain_seed) -> &[TreeNode]` — main entry point, returns a slice into the internal buffer.
- `build_and_merge(...)` — performs tree build + REST merge in a single call.

---

## DFlash (`speculative/dflash.rs`)

Draft-Flash produces marginal distributions over future tokens. It runs the draft model in a single forward pass per depth level, avoiding autoregressive serialization.

### Functions

| Function | Description |
|----------|-------------|
| `dflash_predict(weights, config, token, pos) → Vec<Vec<f32>>` | Independent marginals per depth (same token/pos fed each step). |
| `dflash_predict_ar(weights, config, token, pos, rng)` | Autoregressive: sample a token → feed it back → repeat for each depth. |
| `dflash_predict_parallel(weights, config, token, pos)` | Rayon parallel marginals across depths (skips parallelism if `n_embd ≤ threshold`). |
| `dflash_predict_conditioned(weights, config, token, pos, hidden_state)` | Target-conditioned: seeds the draft KV cache with the target model's hidden state for better alignment. |

### Zero-alloc Variants

All `_with` variants accept a `&mut SpeculativeContext` and write results into pre-allocated buffers:

| Function | Returns |
|----------|---------|
| `dflash_predict_with(sctx, weights, config, token, pos) -> usize` | Count of depth levels produced; marginals stored in `sctx`. |
| `dflash_predict_ar_with(sctx, weights, config, token, pos, rng) -> usize` | Same contract, autoregressive mode. |
| `dflash_predict_conditioned_with(sctx, weights, config, token, pos, hidden_state) -> usize` | Same contract, target-conditioned mode. |

---

## SpeculativeVerifier (`speculative/verifier.rs`)

### Trait

```rust
pub trait SpeculativeVerifier: Send + Sync {
    fn speculate(&self, draft_weights, draft_config, token, pos, rng) -> Vec<usize>;
}
```

### SimulatedVerifier

A lightweight verifier that estimates acceptance without running the target model. Useful for benchmarking draft quality independently of target throughput.

**Pipeline:**
1. DFlash produces marginals.
2. DDTree builds the candidate tree.
3. Extract the best path from the tree.
4. Apply a simulated acceptance rate (default 0.75) to cap accepted token count.
5. **Bonus token:** on full acceptance (all draft tokens accepted), sample +1 token from the last marginal — this is "free" since the marginal was already computed.

**Configuration:**
- `acceptance_rate: f32` — probability of accepting each draft token (default 0.75).

### LeviathanVerifier (behind `"leviathan"` feature)

Implements the full Algorithm 1 from Leviathan et al. 2022 ("Fast Inference from Transformers via Speculative Decoding").

**Internal state:**
- Pre-allocated target `ForwardContext` + `MultiLayerKVCache`
- Pre-allocated draft `SpeculativeContext`

**Pipeline:**
1. DFlash draft → save draft distributions q(x) at each depth.
2. Target model forward pass on the full draft token sequence → get target distributions p(x).
3. For each draft token, accept with probability min(1, p/q).
4. On rejection: sample from the residual distribution max(0, p − q), normalized.
5. On full acceptance of all draft tokens: sample a bonus token from p(x) at position γ (the last draft position).

**Key properties:**
- Guarantees exact target distribution (no approximation drift).
- Real p/q rejection sampling + residual distribution + bonus token.
- Currently slow at 4× model ratio — requires training/distillation for viable throughput in production.

---

## Sampling (`speculative/sampling.rs`)

Low-level sampling primitives used by verifiers and tree builders.

| Function | Description |
|----------|-------------|
| `sample_from_distribution(probs, rng) -> usize` | Categorical sampling. Zero-alloc, operates in-place on the provided slice. |
| `sample_residual_distribution(p, q, rng) -> usize` | Sample from max(0, p − q) normalized. Used by Leviathan verifier on rejection. |
| `sample_residual_distribution_into(p, q, scratch, rng) -> usize` | Zero-alloc variant that writes the residual into a pre-allocated scratch buffer. |

---

## Step Functions (`speculative/step.rs`)

High-level entry points that compose the full speculative decoding pipeline.

| Function | Description |
|----------|-------------|
| `speculative_step(draft_weights, draft_config, token, pos, rng)` | Default: uses `SimulatedVerifier` internally. |
| `speculative_step_verifier(..., verifier)` | Plugs in a custom `SpeculativeVerifier` implementation. |
| `speculative_step_rest(..., rest_client)` | REST-augmented speculative decoding (behind `"rest"` feature). |
| `speculative_step_rollback(..., verifier)` | KV snapshot/rollback per branch — enables speculative branching without corrupting the main KV cache. |
| `speculative_step_conditioned(...)` | Target-conditioned draft seeding + Leviathan verification for highest-fidelity output. |

---

## REST Bridge (`rest/`) (behind `"rest"` feature)

Augments speculative decoding with historically successful token sequences retrieved from an external vector store.

### REST Architecture

```
Target Model Forward Pass
    │
    ├─ ctx.x (hidden state) ──► copy to embedding buffer ──► POST /search/vector
    │                                                          │
    │                                                          ▼
    │                                                    anyrag returns
    │                                                    historical continuations
    │                                                    (token sequences from
    │                                                     past successful compilations)
    │                                                          │
    ▼                                                          ▼
DDTree Build ◄──────── merge retrieved sequences as candidate branches
    │
    ▼
ConstraintPruner validates
    │
    ▼
Target Verification
```

### RestClient

```rust
pub struct RestClient {
    base_url: String,
    client: reqwest::Client,
}
```

- `retrieve(embedding: &[f32], top_k: usize) -> RetrievalResult` — queries an anyrag `/search/vector` endpoint with the current hidden state embedding. Returns historical token continuations ranked by cosine similarity.

### RetrievalResult

```rust
pub struct RetrievalResult {
    pub token_sequences: Vec<Vec<usize>>,
    pub scores: Vec<f32>,
}
```

### DDTree Merge

`merge_retrieved_branches(tree, marginals, config, retrieved, rest_weight)` blends REST results into the draft tree:

- **Score blending:** `(1 − w) × log(draft_prob) + w × log(retrieval_score)` where `w = rest_weight`.
- Merged branches coexist with DDTree candidates; both respect `tree_budget`.
- Retrieval branches that duplicate draft branches are merged (scores summed in log-space).

---

## SpeculativeContext (`speculative/types.rs`)

Pre-allocated buffer struct for zero-alloc speculative decoding across all pipeline stages.

```rust
pub struct SpeculativeContext {
    pub ctx: ForwardContext,
    pub cache: MultiLayerKVCache,
    pub marginals_flat: Vec<f32>,       // [draft_lookahead × vocab_size]
    pub probs_buf: Vec<f32>,            // [vocab_size] temp for softmax
    pub sampled_tokens: Vec<usize>,     // [draft_lookahead]
    pub accepted_buf: Vec<usize>,       // [draft_lookahead + 1]
    pub path_buf: Vec<usize>,           // [draft_lookahead + 1]
    pub residual_buf: Vec<f32>,         // [vocab_size]
    pub p_distributions_flat: Vec<f32>, // [(draft_lookahead + 1) × vocab_size] for Leviathan
    pub parent_tokens_buf: Vec<usize>,  // [draft_lookahead + 1] for pruner
}
```

- `new(config)` — allocates all buffers based on `draft_lookahead` and `vocab_size`.
- `reset()` — clears lengths to zero; capacity is reused.
- Used by all `_with` variants and both verifier implementations.

---

## ConstraintPruner (`speculative/types.rs`)

Trait for filtering invalid token branches during tree construction.

```rust
pub trait ConstraintPruner: Send + Sync {
    fn is_valid(&self, depth: usize, token_idx: usize, parent_tokens: &[usize]) -> bool;
}
```

### Implementations

| Pruner | Feature | Description |
|--------|---------|-------------|
| `NoPruner` | (always available) | Always returns `true`. No-op pass-through. |
| `SudokuPruner` | `"sudoku"` | Row/column/box validation with path-aware cross-depth checks. Ensures generated token sequences satisfy Sudoku constraints. |
| `SynPruner` | `"validator"` | Bracket balance + syntax parse validation. Ensures generated code/structured output remains syntactically valid. |

Pruners are called during `build_dd_tree_pruned` for every candidate node before it enters the priority heap. Invalid branches are discarded immediately, saving budget for valid explorations.

---

## Speculative Prefill (`speculative/prefill.rs`)

PFlash-inspired prompt compression for reducing Time-To-First-Token (TTFT) on long prompts.

### PrefillScorer Trait

```rust
pub trait PrefillScorer: Send + Sync {
    fn score(&self, tokens: &[usize], weights, config) -> Vec<f32>;
}
```

| Scorer | Description |
|--------|-------------|
| `AttentionScorer` | Uses Q·K dot-product attention magnitudes to estimate token importance. Tokens with high attention scores are retained. |
| `RandomScorer` | Random importance scores (baseline/ablation). |
| `UniformScorer` | Uniform scores (baseline/ablation). |

### Compression

`compress_prompt(tokens, scores, keep_ratio) -> Vec<usize>` — retains the first token, the last token, and the top-scoring spans in between, proportional to `keep_ratio`.

### Full Prefill Pipeline

`speculative_prefill(draft_weights, draft_config, target_weights, target_config, prompt, scorer, keep_ratio)`

1. Score prompt tokens using the draft model (fast).
2. Compress the prompt based on scores and `keep_ratio`.
3. Run target prefill on the compressed prompt (slow, but ~10× shorter).
4. Result: ~10× TTFT reduction with minimal quality loss for long prompts.

---

## Feature Flags Summary

| Flag | Enables |
|------|---------|
| (default) | SimulatedVerifier, basic speculative decoding |
| `"leviathan"` | `LeviathanVerifier` — full p/q rejection sampling |
| `"rest"` | `RestClient`, REST-augmented tree merge |
| `"sudoku"` | `SudokuPruner` — constrained decoding for Sudoku |
| `"validator"` | `SynPruner` — syntax-aware constrained decoding |

---

## Key Design Principles

1. **Zero-allocation hot path:** All `_with` variants and `TreeBuilder` reuse pre-allocated buffers. No `Vec::push` in the inner loop — only `clear()` + index writes.
2. **Separable pipeline stages:** DFlash → DDTree → Verifier are independent. Each can be swapped, benchmarked, or disabled independently.
3. **Config-driven behavior:** `SpeculativeConfig` controls lookahead depth, tree budget, acceptance rates, and parallelism thresholds. No runtime branching on magic numbers.
4. **Feature-gated complexity:** Leviathan verifier, REST bridge, and constraint pruners are behind feature flags. The default build stays lean.