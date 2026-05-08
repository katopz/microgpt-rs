# Research: Raven — Routing Slot Memories for O(1) Perfect Recall

**Date:** 2025-06
**Status:** Research → Verdict
**Context:** microgpt-rs + anyrag neuro-symbolic architecture
**Paper:** "Raven: High-Recall Sequence Modeling with Sparse Memory Routing" (Afzal, Bick, Xing, Cevher, Gu — 2025)
**Source:** https://github.com/goombalab/raven

---

## TL;DR

Raven replaces the growing KV cache (Transformer) and dense-overwrite SSM state (Mamba) with a **fixed-size slot memory** updated via **sparse Top-K routing**. Each token selects only a few slots to write; unselected slots are **completely frozen**. The result: $O(1)$ per-token compute, fixed memory, and near-perfect recall even at 16× the training context length.

---

## The Problem Space

| Architecture | Memory | Per-Token Write | Recall Quality | Failure Mode |
|---|---|---|---|---|
| Standard Attention | O(N) growing KV | All tokens written | Perfect (within context) | OOM on long sequences |
| SSM (Mamba/GLA) | O(1) fixed state | Dense — all slots updated | Degrades — old info blurred | Dense overwrite destroys old facts |
| SWA (Sliding Window) | O(1) fixed window | FIFO eviction | Recent only | Hard eviction, old facts gone forever |
| **Raven RSM** | **O(1) fixed slots** | **Sparse Top-K** | **Near-perfect** | **Learned routing can misroute** |

The key insight is in the write pattern:

```
SSM:     Every token updates  ALL  slots → old info decays uniformly
SWA:     Every token evicts   OLDEST token → old info gone permanently
Raven:   Every token updates   FEW  slots → unselected slots FROZEN perfectly
```

---

## Core Mathematics

### Equation 1: Sparse Router

```
route_scores = TopK( sigmoid(W_route × x_t) )
```

For each token `x_t`, a linear projection produces logits over all slots. Sigmoid converts to [0,1] scores. Only Top-K are kept; the rest are zeroed.

### Equation 2: Gated Memory Update

```
decay = exp(route_scores × f_t)           // f_t = forget gate (Mamba2-style)
H_new = H_old × decay + (1 - decay) × new // per-slot gated write
```

Where `route_scores[slot] == 0`:
- `decay = exp(0) = 1.0` → `H_new = H_old × 1.0 = H_old` → **perfectly preserved**

Where `route_scores[slot] > 0`:
- `decay < 1.0` → old content partially overwritten with new content

### Equation 3: Readout

```
o_t = softmax(Q_t × K_slots) × V_slots    // standard attention over fixed slots
```

Readout is $O(\text{slots})$ — constant regardless of sequence length.

---

## Benchmarks (From Paper)

### In-Context Recall (NIAH — Needle In A Haystack)

| Context | 1K | 2K | 4K | 8K | 16K | 32K |
|---------|-----|-----|-----|-----|------|------|
| Transformer+RoPE | 100 | 100 | 0 | 0 | 0 | 0 |
| Mamba-2 | 99.2 | 95.6 | 52.2 | 12.8 | 5.4 | 2.8 |
| GDN | 99.2 | 100 | **99.8** | 92.0 | 41.8 | 22.1 |
| **Raven** | **99.8** | **100** | **99.8** | **99.8** | **99.4** | **91.4** |

Raven is the only architecture that maintains >99% recall from 1K through 16K tokens. At 32K it still hits 91.4% — where Transformers and SSMs collapse to near-zero.

---

## Verdict

### What's Real

1. **The recall numbers are extraordinary.** 99.4% at 16K context vs Mamba-2's 5.4%. This is a 18× improvement.
2. **The $O(1)$ per-token claim is legitimate.** Fixed slots + Top-K routing = constant compute regardless of sequence length.
3. **No auxiliary loss needed.** Raven explicitly removes load balancing (unlike MoE). The model naturally learns imbalanced routing — 90% of slots for syntax, 10% for critical facts. This is a feature, not a bug.
4. **It's built on Flash Linear Attention (FLA).** Production-grade Triton kernels, not a toy implementation. Hybrid models (Raven + standard attention layers) are supported.

### What's Risky

1. **Training cost.** Raven uses Triton kernels requiring CUDA GPUs. The 340M model was trained on 4 GPUs for 30K steps. No CPU/fallback training path exists.
2. **Routing is learned, not guaranteed.** If the router never learns to protect certain slots, those facts get overwritten. There's no hard invariant — it's a soft optimization target.
3. **Slot count is a hyperparameter.** 256 slots worked for 340M params. We don't know the scaling laws for larger models or different domains.
4. **Hybrid is better than pure.** Table 4 shows Hybrid-Raven (with some standard attention layers) significantly outperforms pure Raven on multi-needle retrieval. Pure Raven drops to 0 on N3-32K.

### What We Distill

| Concept | Raven Paper | Our Distillation | Target Project |
|---------|-------------|------------------|----------------|
| Fixed slot memory | Neural hidden state H ∈ R^(slots × d_v) | `RavenKVCache` struct | `microgpt-rs` |
| Sparse Top-K router | Linear projection + sigmoid | `compute_router()` | `microgpt-rs` |
| Gated update | `exp(r_t × f_t)` decay | `update()` with per-slot decay | `microgpt-rs` |
| Imbalanced routing | No load-balancing loss | Allow 90/10 slot specialization | `microgpt-rs` |
| Slot-based sharding | N/A — neural only | `RoutedRagDB` with named slots | `anyrag` |
| Selective decay | Per-slot forget gate | Per-slot confidence decay in Turso | `anyrag` |
| Routing-guided retrieval | N/A — single model | LLM `r_t` → RAG slot selection | `anyrag` |

---

## Integration Plan: microgpt-rs

### Current State

The draft model in `transformer.rs` uses standard MHA with growing KV cache:

```microgpt-rs/src/transformer.rs#L88-94
pub struct KVCache {
    pub key: Vec<f32>,   // [block_size, kv_dim]
    pub value: Vec<f32>, // [block_size, kv_dim]
}
```

For long inputs (e.g., a 5000-line Python file being translated to Rust), this grows linearly and becomes the bottleneck.

### Proposed: RavenKVCache

Replace the draft model's `KVCache` with a fixed-slot variant. The target model keeps standard attention (it needs full precision for verification).

```rust
/// Raven Routing Slot Memory — O(1) KV replacement for the draft model.
///
/// Replaces the growing [block_size, kv_dim] cache with a fixed
/// [num_slots, kv_dim] memory updated via sparse Top-K routing.
/// Unselected slots are completely frozen — perfect for preserving
/// struct definitions and imports while churning through token soup.
pub struct RavenKVCache {
    num_slots: usize,
    kv_dim: usize,
    /// Key memory: [num_slots, kv_dim]
    keys: Vec<f32>,
    /// Value memory: [num_slots, kv_dim]
    values: Vec<f32>,
}
```

### Integration Points

| File | What Changes | Why |
|------|-------------|-----|
| `src/transformer.rs` | Add `RavenKVCache` alongside existing `KVCache` | Draft model uses Raven, target keeps standard |
| `src/speculative/step.rs` | Draft step uses `RavenKVCache.readout()` | $O(1)$ per draft token |
| `src/speculative/prefill.rs` | Prefill populates slots via sparse routing | Initial context → slot memory |
| `src/percepta.rs` | Raven replaces the 2D hull for adversarial cases | Hull fails on V-shapes; Raven doesn't |

### Why NOT Replace Percepta Entirely

Percepta's 2D convex hull attention is $O(\log N)$ for the *common case* — when keys form a clean hull. It's elegant and fast. Raven's $O(\text{slots})$ is also constant but requires more computation per step (router projection + Top-K).

The right play is **adaptive**:

```
if hull_is_valid(keys):
    use Percepta O(log N)    // fast path for well-behaved sequences
else:
    use Raven RSM O(slots)   // fallback for adversarial inputs
```

This mirrors the Hybrid-Raven approach from Table 4 — use the fast path when possible, fall back to robust path when needed.

### PoC: RavenKVCache Implementation

```rust
// microgpt-rs/src/transformer.rs (addition, not replacement)

/// Sparse router: computes Top-K routing vector from raw logits.
///
/// Implements: r_t = Normalize(TopK(Sigmoid(W_route × x_t)))
/// Unselected slots get 0.0 → completely frozen during update.
fn compute_router(raw_logits: &[f32], top_k: usize) -> Vec<f32> {
    let num_slots = raw_logits.len();

    // Sigmoid + enumerate
    let mut scored: Vec<(usize, f32)> = raw_logits
        .iter()
        .enumerate()
        .map(|(i, &x)| (i, 1.0 / (1.0 + (-x).exp())))
        .collect();

    // Partial sort: find Top-K by descending score
    scored.select_nth_unstable_by(num_slots - top_k, |a, b| {
        a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut r_t = vec![0.0f32; num_slots];
    let mut sum = 0.0f32;

    // Keep only Top-K
    for (idx, score) in scored.iter().rev().take(top_k) {
        r_t[*idx] = *score;
        sum += *score;
    }

    // Normalize so selected slots sum to 1.0
    if sum > 0.0 {
        for v in r_t.iter_mut() {
            *v /= sum;
        }
    }

    r_t
}

/// Gated memory update: Raven Equation 18.
///
/// For each slot:
///   decay = exp(a_t × r_t[slot])
///   H_new = decay × H_old + (1 - decay) × new_content
///
/// When r_t[slot] == 0: decay = exp(0) = 1.0 → H_new = H_old (FROZEN)
/// When r_t[slot] > 0: decay < 1.0 → old content decays, new writes in
fn raven_update(
    keys: &mut [f32],
    values: &mut [f32],
    new_key: &[f32],
    new_value: &[f32],
    r_t: &[f32],
    forget_rate: f32,
    num_slots: usize,
    kv_dim: usize,
) {
    for slot in 0..num_slots {
        let decay = (forget_rate * r_t[slot]).exp();
        let write = 1.0 - decay;
        let offset = slot * kv_dim;

        for d in 0..kv_dim {
            keys[offset + d] = decay * keys[offset + d] + write * new_key[d];
            values[offset + d] = decay * values[offset + d] + write * new_value[d];
        }
    }
}

/// Readout: attention over fixed slot memory.
/// O(num_slots × kv_dim) — constant regardless of sequence length.
fn raven_readout(
    query: &[f32],
    keys: &[f32],
    values: &[f32],
    num_slots: usize,
    kv_dim: usize,
) -> Vec<f32> {
    let mut scores = vec![0.0f32; num_slots];
    let mut max_score = f32::NEG_INFINITY;

    // Q · K^T
    for slot in 0..num_slots {
        let off = slot * kv_dim;
        let dot: f32 = (0..kv_dim).map(|d| query[d] * keys[off + d]).sum();
        scores[slot] = dot;
        max_score = max_score.max(dot);
    }

    // Softmax
    let sum_exp: f32 = scores.iter().map(|s| (s - max_score).exp()).sum();
    let mut output = vec![0.0f32; kv_dim];
    for slot in 0..num_slots {
        let weight = (scores[slot] - max_score).exp() / sum_exp;
        let off = slot * kv_dim;
        for d in 0..kv_dim {
            output[d] += weight * values[off + d];
        }
    }

    output
}
```

### What This Gives the Draft Model

| Property | Before (Standard KV) | After (Raven RSM) |
|----------|---------------------|-------------------|
| Memory per layer | `block_size × kv_dim` (grows) | `256 × kv_dim` (fixed) |
| Per-token compute | $O(N)$ attention scan | $O(256)$ slot attention |
| Long-file handling | Slows down linearly | Constant speed |
| Recall of file header | Perfect (within window) | Near-perfect (frozen slots) |
| Beyond context window | Impossible | 99.4% at 16× training length |

---

## Integration Plan: anyrag

### Current State

anyrag ingests documents into a unified vector space via Turso/libsql. The `Ingestor` trait handles different document types:

```rust
// anyrag crate structure
pub trait Ingestor {
    async fn ingest(&self, content: &str) -> Result<Vec<IngestedArticle>>;
}
```

The problem: as the knowledge base grows to 500K+ documents, nearest-neighbor retrieval hits the **Neighborhood Density** problem — relevant but distinct concepts crowd each other out in embedding space.

### Proposed: RoutedRagDB

Map Raven's slot architecture to Turso sharding. Instead of one massive `embeddings` table, use named slots:

```sql
-- Before: monolithic
CREATE TABLE embeddings (
    id TEXT PRIMARY KEY,
    content TEXT,
    embedding BLOB,
    metadata JSON
);

-- After: slot-routed
CREATE TABLE rag_slots (
    slot_name TEXT PRIMARY KEY,
    description TEXT,
    routing_keywords JSON,  -- used by router to assign docs
    confidence REAL DEFAULT 1.0,
    created_at TEXT,
    updated_at TEXT
);

CREATE TABLE slot_documents (
    id TEXT PRIMARY KEY,
    slot_name TEXT REFERENCES rag_slots(slot_name),
    content TEXT,
    embedding BLOB,
    confidence REAL DEFAULT 1.0,
    ingested_at TEXT
);
```

### The Routing Layer

A router (initially keyword-based, eventually WASM-based) assigns incoming documents to slots:

```rust
/// Slot assignment result from the router.
pub struct SlotAssignment {
    pub slot_name: String,
    pub confidence: f32,
    pub decay_existing: bool,
}

/// Simple keyword-based router (Phase 1).
/// Maps document content keywords to pre-defined slots.
pub fn route_document(content: &str, slots: &[RagSlot]) -> Vec<SlotAssignment> {
    let lower = content.to_lowercase();
    let mut assignments = Vec::new();

    for slot in slots {
        let keywords: Vec<&str> = serde_json::from_str(&slot.routing_keywords)
            .unwrap_or_default();
        let match_count = keywords.iter()
            .filter(|kw| lower.contains(*kw))
            .count();

        if match_count > 0 {
            let score = match_count as f32 / keywords.len().max(1) as f32;
            assignments.push(SlotAssignment {
                slot_name: slot.slot_name.clone(),
                confidence: score,
                decay_existing: score > 0.8, // strong match → update slot
            });
        }
    }

    assignments
}
```

### Selective Decay (Raven Equation 18 → SQL)

When a new document is routed to a slot, old documents in **that slot only** decay. Other slots are untouched:

```rust
/// Apply Raven-style selective decay to a specific slot.
/// Other slots experience ZERO decay — perfectly preserved.
pub async fn decay_slot(
    db: &libsql::Database,
    slot_name: &str,
    decay_factor: f32, // 0.0 = total replace, 1.0 = no change
) -> Result<()> {
    let conn = db.connect()?;

    // Only update documents in the targeted slot
    conn.execute(
        "UPDATE slot_documents
         SET confidence = confidence * ?1
         WHERE slot_name = ?2 AND confidence > 0.01",
        &[&decay_factor.to_string(), &slot_name],
    ).await?;

    // Remove fully decayed documents
    conn.execute(
        "DELETE FROM slot_documents WHERE confidence < 0.01",
        &[],
    ).await?;

    Ok(())
}
```

### Default Slot Schema for Code RAG

| Slot Name | Content | Decay Policy |
|-----------|---------|-------------|
| `architecture` | Project structure, module layout, design docs | **Never decay** (frozen) |
| `types` | Struct/enum definitions, type aliases | Decay on override |
| `apis` | Function signatures, trait impls | Decay on deprecation |
| `dependencies` | Cargo.toml, package versions | Decay on version bump |
| `tests` | Test files, test patterns | Moderate decay |
| `chatter` | Comments, README prose, Slack logs | Aggressive decay |

### What This Gives anyrag

| Property | Before (Monolithic) | After (Routed Slots) |
|----------|--------------------|--------------------|
| Retrieval accuracy | Degrades with scale | Stable — per-slot search |
| Architecture docs | Buried in noise | Frozen slot, always crisp |
| Old version docs | Confuse retrieval | Decayed in dependency slot |
| Curator control | None | Define custom slots + routing |
| SQL complexity | Single table scan | Indexed per-slot queries |

---

## The Grand Unification: Routed Speculation

The most powerful distillation connects both systems through the DDTree speculative decoding loop.

### Current Flow

```
Draft Model → DDTree branches → Target Model verify → Accept/Reject
```

### Routed Speculation Flow

```
Draft Model (RavenKVCache)
    ↓ generates r_t (routing vector)
    ↓
DDTree branches + r_t
    ↓
Target Model verify
    ↓ (on reject, need context)
    ↓
anyrag RoutedRagDB
    ← r_t tells WHICH slots to search
    ← Only retrieve from high-r_t slots
    ↓
Context injection → retry draft
```

The draft model's internal routing state `r_t` becomes the **query plan** for RAG retrieval. Instead of blindly searching all 500K documents, the LLM tells the RAG system *which conceptual slots* it's currently reasoning about.

This is not a metaphor — it's the same math:

```
Raven:  r_t[slot] > 0 → update that slot    (write path)
RAG:    r_t[slot] > 0 → search that slot     (read path)
```

### Implementation Sketch

```rust
/// Routed RAG query: use LLM routing state to select search slots.
pub async fn routed_search(
    db: &libsql::Database,
    query_embedding: &[f32],
    routing_vector: &[f32], // r_t from draft model
    slot_names: &[String],
    top_k: usize,
) -> Result<Vec<RetrievedDoc>> {
    let conn = db.connect()?;

    // Only search slots where r_t > threshold
    let active_slots: Vec<&str> = slot_names.iter()
        .enumerate()
        .filter(|(i, _)| routing_vector.get(*i).copied().unwrap_or(0.0) > 0.1)
        .map(|(_, name)| name.as_str())
        .collect();

    if active_slots.is_empty() {
        // Fallback: search all slots with uniform weight
        return uniform_search(db, query_embedding, top_k).await;
    }

    // Targeted search: only active slots, weighted by r_t
    let mut results = Vec::new();
    for slot in active_slots {
        let slot_docs = search_slot(db, slot, query_embedding, top_k).await?;
        results.extend(slot_docs);
    }

    // Sort by combined score (embedding similarity × routing weight)
    results.sort_by(|a, b| b.combined_score.partial_cmp(&a.combined_score).unwrap());
    results.truncate(top_k);

    Ok(results)
}
```

---

## What NOT To Do

1. **Don't replace the target model's KV cache with Raven.** The target model needs full precision for verification. Raven is for the *draft* model where speed matters more than precision.

2. **Don't remove Percepta.** The 2D hull is faster than Raven for well-behaved sequences. Use Raven as the adversarial fallback, not the default.

3. **Don't implement load balancing on slots.** Raven's key insight is that imbalanced routing is correct behavior. Let 90% of traffic hit 10% of slots. The quiet slots preserve rare-but-critical information.

4. **Don't make the RAG router neural in Phase 1.** Start with keyword matching. Upgrade to WASM-based routing (the existing `WasmPruner` pattern) only after the slot schema stabilizes.

5. **Don't store embeddings in the slot table.** Keep `rag_slots` as metadata. Store embeddings in `slot_documents`. Slots route; documents embed.

---

## Phased Implementation

### Phase 1: RavenKVCache in microgpt-rs (Draft Model Only)

- [ ] Add `RavenKVCache` struct to `transformer.rs`
- [ ] Add `compute_router`, `raven_update`, `raven_readout` functions
- [ ] Wire into draft model path in `speculative/step.rs`
- [ ] Benchmark: draft speed on 5K-token input vs standard KV
- [ ] Test: recall of first 100 tokens after processing 5000 tokens

### Phase 2: Routed Slot Schema in anyrag

- [ ] Add `rag_slots` and `slot_documents` tables to migration
- [ ] Implement keyword-based `route_document()` function
- [ ] Implement `decay_slot()` selective decay
- [ ] Define default slots: architecture, types, apis, dependencies, tests, chatter
- [ ] Benchmark: retrieval accuracy at 100K docs vs monolithic

### Phase 3: Routed Speculation (Connect Both)

- [ ] Export `r_t` from draft model's RavenKVCache
- [ ] Add `routed_search()` to anyrag's search API
- [ ] Wire DDTree rejection → routed RAG retrieval → context injection
- [ ] End-to-end benchmark: Python→Rust translation with routed context

---

## References

- Paper: "Raven: High-Recall Sequence Modeling with Sparse Memory Routing" (Afzal, Bick, Xing, Cevher, Gu — 2025)
- Code: https://github.com/goombalab/raven (built on Flash Linear Attention)
- Related: GDN (Gated Delta Network), GLA (Gated Linear Attention), Mamba-2
- Our related research: `.research/05_Artifact_Definition.md` (Deterministic Validator pattern), `.research/02_Fast Inference from Transformers via Speculative Decoding.md` (DDTree)