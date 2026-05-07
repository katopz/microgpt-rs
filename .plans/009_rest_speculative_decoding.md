# Plan 009: REST Speculative Decoding — anyrag Hidden-State Bridge

> **Rename Note**: The `clora` module was renamed to `validator` because it contains
> deterministic syntax validation code (SynPruner, PartialParser), not neural LoRA weights.
> Feature flag: `clora` → `validator`. Module path: `src/clora/` → `src/validator/`.
> The actual LoRA adapter (`lora.bin`) lives in the `gpu` feature (Plan 008).

## Objective

Connect the mini-dllm inference engine to anyrag for Retrieval-Based Speculative Decoding (REST). Extract the "free embedding" (last hidden state before lm_head) during inference, query anyrag's `/search/vector` endpoint, and inject retrieved token continuations into the DDTree as additional candidate branches.

## The Problem

Current speculative decoding only uses the draft model's marginal distributions to populate the DDTree. The research (00 §Free Embedding, 01 §Stage 2) shows that the hidden state is already computed during target model inference — it's "free" to use as a vector embedding for querying historical token continuations from anyrag's Turso database.

Currently:
1. `forward()` computes `ctx.x` (hidden state) then immediately projects it through `lm_head` into logits
2. `ctx.x` is **private** to `ForwardContext` — no external access
3. Nothing calls anyrag during inference

## Architecture

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
SynPruner (Plan 007) validates
    │
    ▼
Target Verification
```

### Hidden State Extraction

```rust
// transformer.rs — extended ForwardContext

pub struct ForwardContext {
    // ... existing fields (unchanged) ...
    x: Vec<f32>,          // [n_embd] — still private, mutated in-place
    
    // NEW: pre-allocated buffer for hidden state snapshot
    pub hidden_state: Vec<f32>,  // [n_embd] — copied before lm_head
}

impl ForwardContext {
    pub fn new(config: &Config) -> Self {
        Self {
            // ... existing ...
            hidden_state: vec![0.0; config.n_embd],
        }
    }
}
```

In `forward()`, add one line before the lm_head matmul:

```rust
// 9.5. Snapshot hidden state (before lm_head destroys ctx.x)
ctx.hidden_state[..n].copy_from_slice(&ctx.x[..n]);

// 10. LM Head
matmul(&mut ctx.logits, &weights.lm_head, &ctx.x, config.vocab_size, n);
```

### REST Client

```rust
// src/rest/mod.rs (new module, feature-gated behind "rest" feature)

pub struct RestClient {
    base_url: String,     // e.g., "http://localhost:9090"
    client: reqwest::Client,
}

pub struct RetrievalResult {
    pub token_sequences: Vec<Vec<usize>>,  // retrieved token ID sequences
    pub scores: Vec<f32>,                  // similarity scores
}

impl RestClient {
    /// Query anyrag /search/vector with hidden state embedding.
    /// Returns historical token continuations ranked by similarity.
    pub async fn retrieve(
        &self,
        embedding: &[f32],
        top_k: usize,
    ) -> Result<RetrievalResult, RestError> {
        // POST /search/vector with embedding as query vector
        // Parse response into token sequences
        // Requires anyrag to store token sequences as metadata alongside embeddings
    }
}
```

### DDTree Merge

```rust
// speculative/dd_tree.rs — add retrieved branches to tree

/// Inject retrieved token sequences into the DDTree as candidate branches.
/// Each retrieved sequence becomes a path in the tree with score = similarity * weight.
pub fn merge_retrieved_branches(
    tree: &mut Vec<TreeNode>,
    marginals: &[Vec<f32>],
    config: &Config,
    retrieved: &RetrievalResult,
    rest_weight: f32,  // blend factor (0.0 = ignore REST, 1.0 = trust REST fully)
) {
    for (seq_idx, seq) in retrieved.token_sequences.iter().enumerate() {
        let similarity = retrieved.scores.get(seq_idx).copied().unwrap_or(0.0);
        for (depth, &token_idx) in seq.iter().enumerate() {
            if depth >= marginals.len() { break; }
            let base_prob = marginals[depth].get(token_idx).copied().unwrap_or(0.0);
            let score = (base_prob.ln() * (1.0 - rest_weight))
                      + (similarity.ln() * rest_weight);
            
            let parent_path = if depth == 0 {
                token_idx as u128
            } else {
                // Reconstruct path from sequence prefix
                let mut path = 0u128;
                for (d, &t) in seq[..depth].iter().enumerate() {
                    path |= (t as u128) << (d * 16);
                }
                path
            };
            
            tree.push(TreeNode { score, depth, token_idx, parent_path });
        }
    }
    // Re-sort by score descending
    tree.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    tree.truncate(config.tree_budget);
}
```

### Integration Point

```rust
// speculative/step.rs — extended speculative step with REST

#[cfg(feature = "rest")]
pub async fn speculative_step_rest(
    draft_weights: &TransformerWeights,
    draft_config: &Config,
    target_weights: &TransformerWeights,
    target_config: &Config,
    token: usize,
    pos: usize,
    rng: &mut Rng,
    rest_client: &RestClient,
) -> Vec<usize> {
    // 1. Draft marginals via DFlash
    let marginals = dflash_predict(draft_weights, draft_config, token, pos);
    
    // 2. Build initial DDTree
    let mut tree = build_dd_tree(&marginals, draft_config);
    
    // 3. Run target model forward to get hidden state
    let mut target_ctx = ForwardContext::new(target_config);
    let mut target_cache = KVCache::new(target_config);
    let logits = forward(&mut target_ctx, target_weights, &mut target_cache, token, pos, target_config);
    
    // 4. Query anyrag with hidden state embedding
    let retrieved = rest_client.retrieve(&target_ctx.hidden_state, 5).await
        .unwrap_or(RetrievalResult::default());
    
    // 5. Merge retrieved branches into DDTree
    merge_retrieved_branches(&mut tree, &marginals, draft_config, &retrieved, 0.3);
    
    // 6. Verify with target model (existing flow)
    // ...
    extract_best_path(&tree)
}
```

## Dependency Additions

```toml
[dependencies]
# ... existing ...
reqwest = { version = "0.12", features = ["json"], optional = true }
tokio = { version = "1", features = ["rt"], optional = true }

[features]
# ... existing ...
rest = ["reqwest", "tokio"]   # Retrieval-based speculative decoding
```

## Tasks

### Phase 1: Hidden State Extraction
- [x] 1.1 Add `pub hidden_state: Vec<f32>` field to `ForwardContext` in `transformer.rs`
- [x] 1.2 Initialize in `ForwardContext::new()` with `vec![0.0; config.n_embd]`
- [x] 1.3 Add `ctx.hidden_state[..n].copy_from_slice(&ctx.x[..n])` before lm_head matmul in `forward()`
- [x] 1.4 Verify existing tests still pass (zero-alloc guarantee unchanged)
- [x] 1.5 Add test: `hidden_state` differs from `logits` (different dimensionality)
- [x] 1.6 Add test: `hidden_state` at same pos/token is deterministic

### Phase 2: REST Client Module
- [x] 2.1 Add `reqwest` and `tokio` to `Cargo.toml` behind `rest` feature
- [x] 2.2 Create `src/rest/mod.rs` with feature gate
- [x] 2.3 Create `src/rest/client.rs` — `RestClient`, `RetrievalResult`
- [x] 2.4 Create `src/rest/types.rs` — request/response types matching anyrag API
- [x] 2.5 Add `pub mod rest;` to `src/lib.rs` behind `#[cfg(feature = "rest")]`
- [x] 2.6 Add tests: mock REST response parsing

### Phase 3: DDTree Merge
- [x] 3.1 Add `merge_retrieved_branches()` to `speculative/dd_tree.rs`
- [x] 3.2 Implement score blending: `(1-w) * log(draft_prob) + w * log(retrieval_score)`
- [x] 3.3 Implement path reconstruction from retrieved sequences
- [x] 3.4 Add test: merge preserves tree_budget
- [x] 3.5 Add test: merge sorts by blended score
- [x] 3.6 Add test: merge with empty retrieval is no-op

### Phase 4: Integration
- [x] 4.1 Create `speculative_step_rest()` in `speculative/step.rs`
- [x] 4.2 Wire: DFlash → DDTree → target forward → REST query → merge → verify
- [x] 4.3 Add benchmark: `Speculative (REST)` vs `Speculative (Simulated)` acceptance rate
- [x] 4.4 Add example: `examples/rest_demo.rs` (behind `rest` feature)
- [x] 4.5 Run full benchmark suite

## Feature Flags

```toml
[features]
default = []
leviathan = []
sudoku = []
clora = ["syn", "proc-macro2"]
rest = ["reqwest", "tokio"]
gpu = ["wgpu", "bytemuck", "pollster", "safetensors"]
full = ["leviathan", "sudoku", "clora", "rest", "training", "gpu"]
```

## Key Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|-----------|
| REST latency >> draft latency | Speculative step slower, not faster | Async query; cache results; only query every N steps |
| Hidden state not a good embedding for retrieval | Poor retrieval quality | Normalize hidden state; experiment with projection layer |
| anyrag doesn't store token sequences as metadata | Can't retrieve continuations | Store BPE-encoded sequences in document metadata during ingestion |
| Feature flag explosion | Complex Cargo.toml | Each feature is independent; `full` enables everything |

## Expected Outcomes

1. `ForwardContext.hidden_state` — free embedding extraction with zero extra compute
2. `RestClient` — async bridge to anyrag vector search
3. `merge_retrieved_branches()` — inject retrieved continuations into DDTree
4. Higher acceptance rate: retrieved sequences are historically verified → fewer rejections

## Files to Create/Modify

| File | Action | Phase |
|------|--------|-------|
| `src/transformer.rs` | Add `hidden_state` field + copy | 1 |
| `Cargo.toml` | Add `reqwest`, `tokio`, `rest` feature | 2 |
| `src/rest/mod.rs` | New | 2 |
| `src/rest/client.rs` | New | 2 |
| `src/rest/types.rs` | New | 2 |
| `src/lib.rs` | Add `mod rest` behind feature gate | 2 |
| `src/speculative/dd_tree.rs` | Add `merge_retrieved_branches` | 3 |
| `src/speculative/step.rs` | Add `speculative_step_rest` | 4 |
| `src/benchmark.rs` | Add REST benchmark | 4 |
| `examples/rest_demo.rs` | New | 4 |

## References

- `.research/00_Neuro-Symbolic LLM Architecture.md` — §Part 2: "Free Embedding", REST
- `.research/01_Advanced Neuro-Symbolic Rust Translation.md` — §Stage 2: Dual-Mode Speculative Drafting
- `anyrag/README.md` — `/search/vector` endpoint, embedding API