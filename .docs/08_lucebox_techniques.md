# microgpt-rs: Advanced Techniques (Lucebox-Hub Distillation)

## Source
Techniques distilled from [Luce-Org/lucebox-hub](https://github.com/Luce-Org/lucebox-hub/) — open LLM inference optimized per-chip. We take the algorithmic ideas (chain-seed DDTree, importance scoring, rollback) and implement them on CPU without CUDA.

## Plan Dependency Map

```
Plan 009 (REST)              Plan 010 (Multi-Layer)        Plan 011 (GQA + Paged)
     │                              │                            │
     │ hidden_state                 │ MultiLayerKVCache          │ n_kv_head, kv_dim()
     │ RestClient                   │ LayerWeights               │ PagedKVCache (pending)
     │ merge_retrieved_branches()   │ Config::small_target()     │ Config::gqa_draft()
     │                              │ Config.n_layer             │
     ▼                              ▼                            ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     Lucebox Techniques (This Doc)                       │
│                                                                         │
│  Chain-Seed DDTree ────── uses build_dd_tree_pruned() +                │
│                           coexists with merge_retrieved_branches        │
│                                                                         │
│  Budget Sweep ─────────── uses Config.tree_budget,                     │
│                           sweeps micro/draft/small_target/gqa           │
│                                                                         │
│  KV Snapshot/Rollback ─── uses MultiLayerKVCache.layers,               │
│                           kv_dim(), forward() per layer                 │
│                           future: PagedKVCache.fork()                   │
│                                                                         │
│  Speculative Prefill ──── uses hidden_state for scoring,                │
│                           draft model + MultiLayerKVCache,              │
│                           bridge to speculative_step_rest()             │
│                                                                         │
│  Target-Conditioned ───── uses hidden_state + MultiLayerKVCache         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Technique 1: Chain-Seed DDTree

### Problem
Pure best-first DDTree gives acceptance length (AL) ~4 on quantized targets. The tree lacks a high-confidence "spine" to branch from.

### Solution
Two-phase tree construction:
1. **Phase A (Chain)**: Greedy argmax over marginals for each depth → build backbone of highest-probability tokens
2. **Phase B (Branch)**: Best-first expansion from ALL chain nodes (not just root) → branch alternatives

```rust
// dd_tree.rs — chain_seed parameter
build_dd_tree_pruned(marginals, config, pruner, chain_seed: bool)
```

- Chain nodes consume budget: if chain length = L, remaining = tree_budget - L
- Chain broken by constraint → fall through to standard best-first
- Coexists with `merge_retrieved_branches()` (REST merge adds branches AFTER chain-seed build)

### Results
| Config | DDTree (no chain) | DDTree (chain-seed) |
|--------|:-:|:-:|
| micro | 364,458 trees/s | 385,957 trees/s |
| Draft sweep AL | baseline | marginal improvement at draft scale |

Lucebox found AL recovered from ~4 to ~9 at 27B scale. Benefit grows with model size.

## Technique 2: DDTree Budget Sweep

### Problem
Tree budget was hardcoded (16 or 32). Optimal budget depends on model size and target ratio.

### Solution
Sweep budgets empirically: `[4, 8, 12, 16, 20, 22, 24, 32, 48, 64]`
- Per budget: measure tree build time, node count, simulated acceptance length
- Lucebox found budget=22 sweet spot for RTX 3090 + 27B Q4_K_M

### Results (draft config, 75% simulated acceptance)
| Budget | Throughput | AL |
|--------|-----------|-----|
| 4 | fastest | low |
| **8** | **585K trees/s (optimal)** | good |
| 16 | baseline | good |
| 32+ | diminishing returns | marginal |

Optimal: budget=8 for draft config (throughput tradeoff). Budget scaling is model-dependent.

## Technique 3: KV-Cache Snapshot & Rollback

### Problem
DDTree branch verification writes to shared KV cache. On reject, stale data corrupts subsequent branches.

### Solution
```rust
// transformer.rs
pub struct KVSnapshot {
    pub layers: Vec<KVLayerSnapshot>,
    pub pos: usize,
}

impl MultiLayerKVCache {
    pub fn snapshot(&self, pos: usize, config: &Config) -> KVSnapshot {
        // Copies only filled slots [0..pos * kv_dim] per layer
    }
    pub fn restore(&mut self, snapshot: &KVSnapshot, config: &Config) {
        // Writes snapshot back + zeros stale positions
    }
}
```

- Cheap: copies only `[0..pos * kv_dim]` per layer, not entire `[block_size * kv_dim]`
- Micro config: ~2 KB per snapshot
- small_target (4 layers, kv_dim=64): ~128 KB per snapshot

### Integration
`speculative_step_rollback()` in `step.rs`:
1. Snapshot KV cache before verifying each DDTree branch
2. Run forward passes for branch tokens
3. On reject at position k: restore snapshot, try next branch
4. Extracts top-3 candidate paths, verifies each with rollback

### Results
| Method | Throughput | Notes |
|--------|-----------|-------|
| Leviathan (no rollback) | 108,827 tok/s | Corrupts cache on reject |
| **Leviathan (w/ rollback)** | **161,324 tok/s** | **+49% per accepted token** |

### Future: PagedKVCache fork-based rollback
- `PagedKVCache.fork()` shares prefix pages (copy-on-write)
- Only new pages allocated after fork point
- Deferred to Plan 014 — currently uses flat snapshot/restore

## Technique 4: Speculative Prefill (PFlash-Inspired)

### Problem
Long prompts require expensive target model prefill over every token. 128K tokens → slow TTFT.

### Solution
Use draft model's attention scores to identify important tokens, compress prompt before target prefill.

```rust
// speculative/prefill.rs
pub trait PrefillScorer: Send + Sync {
    fn score(&self, draft_weights, draft_config, prompt_tokens) -> Vec<f32>;
}
pub struct AttentionScorer;  // Q·K attention importance (PFlash-inspired)
pub struct RandomScorer;     // Baseline
pub struct UniformScorer;    // Baseline: keep every Nth token
```

### Pipeline
1. `score_token_importance()` — run draft model forward per token, extract Q·K attention scores
2. `compress_prompt(tokens, scores, keep_ratio)` — always keep first/last N, select top middle spans
3. `speculative_prefill()` — target model forward on compressed prompt → filled KV cache

### Results
| Method | Throughput | Effective Tokens | Notes |
|--------|-----------|:---:|-------|
| Prefill (no compress) | 2,691K tok/s | 64 | Full prompt |
| **Prefill (compressed)** | **1,714K tok/s** | **7** | ~10.9% keep ratio |

Compression trades throughput for compute savings: 128K → 2.6K tokens would give ~10.4× TTFT reduction.

### Bridge to REST
After prefill compression, `speculative_step_rest()` continues decode with REST retrieval.

## Technique 5: Target-Conditioned Draft

### Problem
DFlash produces independent marginals (same token/pos each step). Every position conditions on the same input, not on real target features.

### Solution
Seed draft model's KV cache with target hidden state:
```rust
// dflash.rs
pub fn dflash_predict_conditioned(
    weights, config, token, pos, hidden_state: &[f32]
) -> Vec<Vec<f32>>
```
- Projects target `hidden_state` to draft `kv_dim`
- Seeds draft KV cache with projected hidden state
- Draft model conditions on real target features, not its own noisy predictions

### Integration
`speculative_step_conditioned()` — target forward → hidden state → conditioned draft → DDTree → simulated acceptance

### Results
| Method | Throughput | Accept Len |
|--------|-----------|:---:|
| Spec (unconditioned) | 842,657 tok/s | 5.00 |
| **Spec (conditioned)** | **972,163 tok/s** | **6.74** |

+15% acceptance length improvement from target conditioning.

## Architecture Decisions

1. **Chain-seed is additive** — `build_dd_tree()` works as before (chain_seed=false)
2. **Prefill is a new module** — `speculative/prefill.rs`, no feature flag needed
3. **KV snapshot copies only filled slots** — cheap at our scale, uses `kv_dim()` for GQA
4. **Target conditioning via KV seed** — simplest option, no weight changes
5. **Flat cache first** — PagedKVCache integration deferred to Plan 014
6. **No new model weights** — reuses draft model attention + target hidden_state

## Key References
- [Luce-Org/lucebox-hub](https://github.com/Luce-Org/lucebox-hub/) — Open LLM Inference, Rewritten by Hand for One Specific Chip at a Time
- [DFlash: Block-Diffusion Speculative Decoding](https://arxiv.org/abs/2602.06036) — Wang et al., 2026
- [DDTree: Block Diffusion Draft Trees](https://arxiv.org/abs/2604.12989) — Ringel & Romano, 2026
- [Cross-Family Speculative Prefill](https://arxiv.org/abs/2603.02631) — Liu et al., ICLR 2026
- [FlashPrefill](https://arxiv.org/abs/2603.06199) — Fan et al., 2026