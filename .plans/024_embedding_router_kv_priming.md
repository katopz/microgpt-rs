# Plan 024: Embedding Router + KV Cache Priming Pipeline

**Branch:** `develop/feature/024_embedding_router_kv_priming`
**Depends on:** Plan 023 (`router` feature), anyrag Plan 005 (`/classify/domain`)
**Research:** `.research/10_ColaDLM_Continuous_Latent_Diffusion.md` (Section 5.1)

---

## Overview

Wire anyrag's embedding search into microgpt-rs's draft model as **KV cache priming context**. When a user edits a known file (e.g., `auth.rs`), the system retrieves the most relevant document embedding from anyrag, projects it to the draft model's hidden dimension, and injects it via the existing `dflash_predict_conditioned_with` mechanism. The draft model produces higher-quality speculative tokens because it "sees" semantic context from related code.

This is NOT Cola DLM's per-position latent diffusion (one vector, all layers, position 0). It is **retrieval-conditioned speculative decoding** — a pragmatic approximation of "global semantic planning" using existing infrastructure.

### Concrete Use Case — Hot Function Continuation

1. User opens `auth.rs` in IDE → file already ingested in anyrag
2. User types `fn validate_token(` → partial function signature
3. System flow:
   - `EmbeddingRouter` calls anyrag `/search/embedding` with prompt + file context
   - anyrag returns top embedding vector from matching documents
   - `EmbeddingProjector` maps embedding dim (e.g., 768) → draft model `n_embd` (e.g., 64)
   - Projected vector injected as `target_hidden_state` via `dflash_predict_conditioned_with`
   - Draft model generates tokens with semantic bias toward `validate_token`'s known patterns
4. Higher acceptance rate against target LLM → fewer verification steps → lower latency

### Key Insight

We already have `speculative_step_conditioned_with` which takes the **target model's** hidden state. Plan 024 adds a **second conditioning source**: the **retrieved document's** embedding. Both can coexist — target hidden state for syntactic alignment, retrieved embedding for semantic alignment.

---

## Architecture

```text
┌─────────────────────────────────────────────────────────────────┐
│                      EmbeddingRouter                             │
│  (PromptRouter impl — replaces KeywordRouter for V2)            │
│                                                                  │
│  route(prompt, context) ──► anyrag POST /classify/domain        │
│                               + POST /search/embedding           │
│                                    │                             │
│                          RouteDecision {                         │
│                              domain,                             │
│                              confidence,                         │
│                              embedding: Option<Vec<f32>>,  ◄── new
│                          }                                       │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                   EmbeddingProjector                             │
│                                                                  │
│   project(embedding: &[f32], target_dim: usize) → Vec<f32>      │
│                                                                  │
│   Strategies (feature-gated):                                    │
│   - TruncatePad: truncate or zero-pad (default, zero-cost)      │
│   - LinearProjection: trained W projection matrix (future)       │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│            dflash_predict_conditioned_with (existing)            │
│                                                                  │
│   target_hidden_state = [target_model_hidden]  ← existing path  │
│   OR                                                             │
│   target_hidden_state = [projected_embedding]  ← NEW path       │
│   OR                                                             │
│   target_hidden_state = [combined]             ← future         │
│                                                                  │
│   Broadcasts to all KV cache layers at position 0               │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow

```text
IDE Context ("editing auth.rs, typing fn validate_token(")
    │
    ▼
EmbeddingRouter::route(prompt, Some(file_context))
    │
    ├─► HTTP POST anyrag /search/embedding
    │   { "query": "fn validate_token", "context_files": ["auth.rs"], "limit": 1 }
    │   Response: { "embedding": [0.12, -0.34, ...], "score": 0.87, "source": "auth.rs#L45" }
    │
    ├─► (fallback) HTTP POST anyrag /classify/domain
    │   { "prompt": "fn validate_token(" }
    │   Response: { "domain": "rust_code", "confidence": 0.91 }
    │
    └─► (fallback) KeywordRouter::route(prompt)  ← local, no network
        Returns: RouteDecision { domain: "rust_code", confidence: 0.6, embedding: None }
    │
    ▼
RouteDecision { domain, confidence, embedding: Some(vec![...]) }
    │
    ▼
EmbeddingProjector::project(&embedding, draft_config.n_embd)
    │
    ▼
projected: Vec<f32>  // dim = n_embd
    │
    ▼
speculative_step_embedding_conditioned(draft, target, token, pos, &projected, rng)
    │
    ▼
dflash_predict_conditioned_with(sctx, draft_weights, draft_config, token, pos, &projected, rng)
    │
    ▼
Draft tokens with semantic bias toward retrieved code patterns
```

---

## Tasks

- [ ] **Task 1: Add `EmbeddingRouteDecision` to `src/router/types.rs`**
  - New type extending `RouteDecision` with optional embedding:
    ```rust
    /// Result of routing with optional retrieved embedding for KV cache priming.
    #[derive(Debug, Clone)]
    pub struct EmbeddingRouteDecision {
        /// Base routing decision (domain, confidence, paths).
        pub route: RouteDecision,
        /// Retrieved embedding vector from anyrag, if available.
        /// Used to prime the draft model's KV cache for context-aware drafting.
        pub embedding: Option<Vec<f32>>,
        /// Source document that produced the embedding (for diagnostics).
        pub embedding_source: Option<String>,
    }

    /// Configuration for the embedding router.
    #[derive(Debug, Clone, Deserialize)]
    pub struct EmbeddingRouterConfig {
        /// anyrag server URL (e.g., "http://localhost:9090").
        pub anyrag_url: String,
        /// Timeout in milliseconds for anyrag calls.
        #[serde(default = "default_timeout")]
        pub timeout_ms: u64,
        /// Whether to also classify domain (hybrid: embedding + domain).
        #[serde(default = "default_true")]
        pub classify_domain: bool,
        /// JWT bearer token for anyrag auth (optional if auth disabled).
        pub auth_token: Option<String>,
    }

    fn default_timeout() -> u64 { 200 }
    fn default_true() -> bool { true }
    ```
  - Keep `RouteDecision` unchanged for backward compat
  - `EmbeddingRouteDecision` wraps it, adding embedding data
  - `EmbeddingRouterConfig` loaded from `domains.toml` `[embedding_router]` section

- [ ] **Task 2: Add `EmbeddingProjector` to `src/router/projector.rs`**
  - New module with dimension projection strategies:
    ```rust
    /// Projects an embedding vector to the draft model's hidden dimension.
    pub trait EmbeddingProjector: Send + Sync {
        fn project(&self, embedding: &[f32], target_dim: usize) -> Vec<f32>;
    }

    /// Strategy 1: Truncate or zero-pad. Zero-cost, no training needed.
    /// If embedding dim > target_dim, take the first `target_dim` elements.
    /// If embedding dim < target_dim, zero-pad to `target_dim`.
    pub struct TruncatePadProjector;

    /// Strategy 2: Learned linear projection (future, requires training).
    /// W: [target_dim, embedding_dim], b: [target_dim]
    /// output = W * embedding + b
    /// NOT implemented in this plan — placeholder for future.
    pub struct LinearProjector {
        // weights: Vec<f32>,  // [target_dim, embedding_dim]
        // bias: Vec<f32>,     // [target_dim]
    }
    ```
  - `TruncatePadProjector` is the default — works immediately with any model
  - Unit tests: project dim 768 → 64 (truncate), 32 → 64 (pad), 64 → 64 (identity)
  - Unit test: all-zeros embedding produces all-zeros output
  - `LinearProjector` is a future extension point (needs trained weights)

- [ ] **Task 3: Add anyrag `/search/embedding` response types to `src/router/types.rs`**
  - Types for parsing anyrag embedding search response:
    ```rust
    /// Response from anyrag `/search/embedding` endpoint.
    #[derive(Debug, Deserialize)]
    pub struct EmbeddingSearchResponse {
        pub result: EmbeddingSearchResult,
    }

    #[derive(Debug, Deserialize)]
    pub struct EmbeddingSearchResult {
        /// Raw embedding vector from the top matching document.
        pub embedding: Vec<f32>,
        /// Cosine similarity score [0.0, 1.0].
        pub score: f32,
        /// Source file/chunk that produced this embedding.
        pub source: String,
    }

    /// Request body for anyrag `/search/embedding`.
    #[derive(Debug, Serialize)]
    pub struct EmbeddingSearchRequest {
        pub query: String,
        /// Optional file context to bias retrieval.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub context_files: Option<Vec<String>>,
        pub limit: u32,
    }
    ```
  - These types exist in microgpt-rs, not anyrag — they're the client's view of anyrag's API
  - anyrag may need a new endpoint or extension to existing search (see Cross-Project)

- [ ] **Task 4: Implement `EmbeddingRouter` in `src/router/embedding.rs`**
  - `PromptRouter` impl with three-tier fallback:
    ```rust
    pub struct EmbeddingRouter {
        config: EmbeddingRouterConfig,
        keyword_router: KeywordRouter,
        projector: Box<dyn EmbeddingProjector>,
        client: reqwest::Client,
    }

    impl EmbeddingRouter {
        pub fn new(
            config: EmbeddingRouterConfig,
            domains: Vec<DomainConfig>,
            projector: Box<dyn EmbeddingProjector>,
        ) -> Self { ... }
    }

    impl PromptRouter for EmbeddingRouter {
        fn route(&self, prompt: &str) -> RouteDecision {
            // Simplified: returns RouteDecision without embedding.
            // Full embedding retrieval requires async, handled by route_async().
        }
    }

    impl EmbeddingRouter {
        /// Async routing with embedding retrieval for KV cache priming.
        /// Falls back through: embedding search → domain classify → keyword.
        pub async fn route_async(&self, prompt: &str) -> EmbeddingRouteDecision {
            // 1. Try embedding search (POST /search/embedding)
            // 2. Try domain classify (POST /classify/domain)
            // 3. Fall back to KeywordRouter (local)
            // Any failure → graceful degradation to next tier
        }
    }
    ```
  - **Key design:** `PromptRouter::route()` is sync (trait requirement), so it delegates to `KeywordRouter`. The new `route_async()` returns `EmbeddingRouteDecision` with optional embedding. Callers that can await use `route_async()`, others use `route()` (keyword-only).
  - `reqwest::Client` reused across calls (connection pooling)
  - Timeout: 200ms default (configurable) — fail fast, fall back to keyword
  - Auth: Optional Bearer token from config

- [ ] **Task 5: Add `speculative_step_embedding_conditioned` to `src/speculative/step.rs`**
  - New speculative step that uses retrieved embedding (not target model hidden state):
    ```rust
    /// Speculative step with embedding-conditioned draft.
    ///
    /// Unlike `speculative_step_conditioned_with` which uses the target model's
    /// hidden state, this uses a retrieved embedding vector projected to the
    /// draft model's dimension. Useful when target model hasn't run yet (first
    /// token) or when semantic context from RAG is more valuable than syntactic
    /// alignment with the target.
    pub fn speculative_step_embedding_conditioned(
        draft_weights: &TransformerWeights,
        draft_config: &Config,
        token: usize,
        pos: usize,
        projected_embedding: &[f32],  // Already projected to n_embd
        rng: &mut Rng,
    ) -> (Vec<usize>, usize) {
        let mut sctx = SpeculativeContext::new(draft_config);
        let num_steps = dflash_predict_conditioned_with(
            &mut sctx,
            draft_weights,
            draft_config,
            token,
            pos,
            projected_embedding,
            rng,
        );
        // ... build DDTree from marginals, same as existing steps
    }
    ```
  - Plus zero-alloc `_with` variant reusing `SpeculativeContext`
  - Reuses existing `dflash_predict_conditioned_with` — no changes to dflash.rs

- [ ] **Task 6: Add `embedding_router` feature to `Cargo.toml`**
  - New feature: `embedding_router = ["router", "reqwest", "tokio"]`
  - Depends on `router` (for types/registry), `reqwest` (for HTTP), `tokio` (for async runtime)
  - Add to `full` feature: `full = [..., "embedding_router"]`
  - NOT in default features (requires running anyrag server)

- [ ] **Task 7: Module wiring (`src/router/mod.rs`, `src/lib.rs`)**
  - Add to `src/router/mod.rs`:
    ```rust
    #[cfg(feature = "embedding_router")]
    pub mod embedding;
    #[cfg(feature = "embedding_router")]
    pub mod projector;

    #[cfg(feature = "embedding_router")]
    pub use embedding::EmbeddingRouter;
    #[cfg(feature = "embedding_router")]
    pub use projector::{EmbeddingProjector, TruncatePadProjector};
    ```
  - Export `EmbeddingRouteDecision` and `EmbeddingRouterConfig` from `types.rs`
  - No changes to `src/lib.rs` (already has `#[cfg(feature = "router")] pub mod router`)

- [ ] **Task 8: Add `EmbeddingPruner` adapter to `src/router/types.rs`**
  - Wrapper that combines domain routing with embedding retrieval:
    ```rust
    /// A screening pruner that also carries an optional embedding for
    /// KV cache priming. The pruner delegates to the domain's registered
    /// pruner; the embedding is used separately by the speculative step.
    pub struct EmbeddingExpertBundle {
        /// The domain's screening pruner (from ExpertRegistry).
        pub pruner: Box<dyn ScreeningPruner>,
        /// Retrieved embedding projected to draft model dim, if available.
        pub projected_embedding: Option<Vec<f32>>,
        /// Source of the embedding (for diagnostics).
        pub embedding_source: Option<String>,
        /// LoRA adapter path from domain config.
        pub lora_path: Option<PathBuf>,
    }
    ```
  - This bundles everything the speculative step needs: pruner + embedding + lora
  - The speculative step checks `projected_embedding.is_some()` to decide
    between `speculative_step_conditioned_with` (target hidden state),
    `speculative_step_embedding_conditioned` (retrieved embedding), or
    `speculative_step_with` (no conditioning)

- [ ] **Task 9: Integration example (`examples/embedding_router_demo.rs`)**
  - Demonstrates the full pipeline:
    1. Load config → build `EmbeddingRouter` with `TruncatePadProjector`
    2. Call `route_async("fn validate_token(")` with anyrag running
    3. Show fallback behavior when anyrag is down
    4. Project embedding to draft model dim
    5. Run `speculative_step_embedding_conditioned` with projected embedding
    6. Compare marginals with/without embedding conditioning
  - Feature-gated: `required-features = ["embedding_router"]`

- [ ] **Task 10: Unit tests**
  - `TruncatePadProjector`: truncate 768 → 64 (first 64 elements kept)
  - `TruncatePadProjector`: pad 32 → 64 (32 zeros appended)
  - `TruncatePadProjector`: identity 64 → 64 (unchanged)
  - `TruncatePadProjector`: empty input → all zeros
  - `EmbeddingRouter::route()`: falls back to KeywordRouter when async not used
  - `EmbeddingSearchRequest`: serialization round-trip
  - `EmbeddingRouteDecision`: default with no embedding (keyword-only path)
  - `speculative_step_embedding_conditioned`: produces valid marginals
  - `speculative_step_embedding_conditioned`: differs from unconditioned
  - `speculative_step_embedding_conditioned`: empty embedding = unconditioned
  - Feature-gate: compiles with and without `embedding_router`

- [ ] **Task 11: Update README**
  - Add "Embedding Router (Plan 024)" section to Architecture
  - Update Feature Flags section with `embedding_router`
  - Add data flow diagram
  - Note anyrag prerequisite and LM Studio API requirement

---

## File Change Summary

| File | Change |
|------|--------|
| `microgpt-rs/src/router/types.rs` | Add `EmbeddingRouteDecision`, `EmbeddingRouterConfig`, `EmbeddingExpertBundle`, anyrag request/response types |
| `microgpt-rs/src/router/projector.rs` | New: `EmbeddingProjector` trait + `TruncatePadProjector` impl |
| `microgpt-rs/src/router/embedding.rs` | New: `EmbeddingRouter` with async `route_async()` + three-tier fallback |
| `microgpt-rs/src/router/mod.rs` | Add feature-gated exports for `embedding` and `projector` |
| `microgpt-rs/src/speculative/step.rs` | Add `speculative_step_embedding_conditioned` + `_with` variant |
| `microgpt-rs/Cargo.toml` | Add `embedding_router` feature |
| `microgpt-rs/examples/embedding_router_demo.rs` | New: integration example |
| `microgpt-rs/README.md` | Add Embedding Router architecture section |

---

## Cross-Project Dependencies

### anyrag Changes Required (NOT in this plan, separate anyrag plan)

| Change | Why | Complexity |
|--------|-----|------------|
| `POST /search/embedding` endpoint | Return raw embedding vectors alongside search results. Current `/search/vector` returns `SearchResult` (text + score) but NOT the raw embedding vector. | Low — extend existing vector search to optionally include the stored embedding in the response |
| `SearchResult.embedding` field (optional) | Add `Option<Vec<f32>>` to `SearchResult` so existing endpoints can optionally include embeddings | Low — backward compatible with `#[serde(skip_serializing_if = "Option::is_none")]` |
| `context_files` parameter | Allow search to bias toward documents from specific files (e.g., "auth.rs") | Medium — requires metadata filtering in vector search |

**Workaround if anyrag endpoint not ready:** `EmbeddingRouter` can call LM Studio's embedding API directly (bypass anyrag search) using the same `reqwest::Client`. Less semantic (no document matching), but validates the pipeline end-to-end.

---

## Design Decisions

### 1. `TruncatePadProjector` as Default

No training, no weights, no complexity. The first N dimensions of an embedding often carry the most information (PCA-like). If this proves insufficient, `LinearProjector` can be trained later using paired (embedding, hidden_state) data from the target model.

**Why not just pad to larger dim?** The draft model's `n_embd` is small (16-64 for micro configs). A 768-dim embedding truncated to 16 dims is lossy but preserves the principal components. If the draft model had `n_embd = 768`, we'd keep the full embedding.

### 2. Sync Trait + Async Extension

`PromptRouter::route()` is sync (existing trait). `EmbeddingRouter` implements it by delegating to `KeywordRouter` (local, no network). The async `route_async()` method is the real entry point for embedding retrieval. This avoids breaking the trait API while adding async capability.

### 3. Three-Tier Fallback

```
Embedding search → Domain classify → Keyword-only
     (200ms)          (100ms)         (<1ms)
```

Each tier degrades gracefully. The system always works — just with less semantic context when anyrag is unavailable.

### 4. Feature-Gated Behind `embedding_router`

Requires `reqwest` + `tokio` + `router`. Not in default features because:
- Needs a running anyrag server
- Needs LM Studio API for embeddings
- Adds HTTP client dependency
- Pure offline use should use `KeywordRouter` (Plan 023)

### 5. Separation from Target Model Conditioning

`speculative_step_conditioned_with` uses the **target model's hidden state** (syntactic alignment).
`speculative_step_embedding_conditioned` uses a **retrieved embedding** (semantic alignment).
These are separate, complementary signals. Future: combine both (target hidden + retrieved embedding) for maximum conditioning.

---

## Out of Scope

- **Multimodal (image understanding):** No vision encoder in our stack. Cola DLM's Image VAE requires GPU. Anyrag-github image extraction (Layer 1 — ingestion-only) is a separate anyrag improvement, not a microgpt-rs concern. See `.research/10_ColaDLM_Continuous_Latent_Diffusion.md` Section 5.2.
- **Multimodal (inference-time):** Shared image+text latent space via MMDiT prior. Destroys our CPU/sub-ms performance profile. The paper calls this "preliminary qualitative evidence." DO NOT pursue until GPU inference with real models exists.
- **Per-layer/per-position conditioning:** Cola DLM uses position-specific, layer-specific VAE latents. We broadcast one vector to all layers. Per-layer conditioning would require modifying `dflash_predict_conditioned_with` to accept `Vec<Vec<f32>>` (one per layer) — significant architecture change with unproven benefit for short completions.
- **LinearProjection training:** Requires paired (embedding, hidden_state) data from the target model. No training infrastructure exists yet. `TruncatePadProjector` is sufficient to validate the pipeline.
- **anyrag `/search/embedding` endpoint implementation:** This is an anyrag change, not a microgpt-rs change. Tracked as cross-project dependency. `EmbeddingRouter` can work with direct LM Studio API calls as a workaround.
- **IDE integration:** The "hot context" protocol (IDE tells microgpt-rs which file is being edited) requires LSP/extension work. This plan focuses on the inference pipeline; IDE integration is a future step.

---

## Performance Considerations

| Operation | Expected Latency | Notes |
|-----------|-----------------|-------|
| KeywordRouter::route | <1ms | Local, no network |
| anyrag `/classify/domain` | ~50-100ms | Local LM Studio embedding + turso query |
| anyrag `/search/embedding` | ~100-200ms | Local LM Studio embedding + vector search |
| TruncatePadProjector::project | <1μs | Memory copy / zero-fill |
| `dflash_predict_conditioned_with` | Same as existing | No perf change to hot path |

The embedding retrieval adds ~200ms to the **first token** latency (one-time per request). This is acceptable for interactive use (IDE autocomplete) where the user has already spent seconds typing. For streaming/batch use, the router falls back to keyword-only.

---

## Benchmark Plan

- [ ] Compare speculative acceptance rates: keyword-conditioned vs embedding-conditioned vs unconditioned
- [ ] Measure first-token latency with and without anyrag call
- [ ] Profile truncation quality: does dim 768→16 preserve enough signal?
- [ ] A/B test: same prompt, with/without embedding, measure target model acceptance rate

---

## SOLID Principles Applied

- **S:** `EmbeddingRouter` only routes + retrieves. `EmbeddingProjector` only projects dimensions. `speculative_step_embedding_conditioned` only runs the draft.
- **O:** New projection strategies via `EmbeddingProjector` trait. New routing strategies via `PromptRouter` trait. No code changes to add new methods.
- **L:** `EmbeddingRouter` is a `PromptRouter` (via `KeywordRouter` delegation). `TruncatePadProjector` is an `EmbeddingProjector`. Swappable.
- **I:** `EmbeddingProjector` has one method: `project()`. `PromptRouter` has one method: `route()`. Thin traits.
- **D:** Speculative step depends on `&[f32]` (projected embedding), not on `EmbeddingRouter` or `anyrag`. The pipeline produces the data; the step consumes it.