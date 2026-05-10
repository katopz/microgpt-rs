//! Embedding router — KV cache priming via anyrag embedding retrieval (Plan 024).
//!
//! Extends the keyword-based routing (Plan 023) with semantic embedding retrieval
//! from anyrag. The retrieved embedding is projected to the draft model's hidden
//! dimension and injected as KV cache priming context for context-aware drafting.
//!
//! # Three-Tier Fallback
//!
//! ```text
//! 1. Embedding search (POST /search/embedding)  ~200ms
//!    ↓ on failure
//! 2. Domain classify (POST /classify/domain)     ~100ms
//!    ↓ on failure
//! 3. KeywordRouter (local, no network)            <1ms
//! ```
//!
//! # Sync vs Async
//!
//! [`PromptRouter::route()`] is sync (trait requirement), so it delegates to
//! [`KeywordRouter`]. The new [`route_async()`] method is the real entry point
//! for embedding retrieval. Callers that can await should use `route_async()`.
//!
//! # Feature Flag
//!
//! Gated behind `embedding_router` feature (requires `router` + `reqwest` + `tokio`).

use reqwest::Client;
use serde::Deserialize;

use super::keyword::KeywordRouter;
use super::projector::EmbeddingProjector;
use super::prompt_router::PromptRouter;
use super::types::{
    EmbeddingRouteDecision, EmbeddingRouterConfig, EmbeddingSearchRequest, EmbeddingSearchResponse,
    RouteDecision,
};

// ---------------------------------------------------------------------------
// Domain classify response (from anyrag)
// ---------------------------------------------------------------------------

/// Response from anyrag `/classify/domain` endpoint.
#[derive(Debug, Deserialize)]
struct ClassifyDomainResponse {
    domain: String,
    confidence: f32,
}

// ---------------------------------------------------------------------------
// EmbeddingRouter
// ---------------------------------------------------------------------------

/// A prompt router that retrieves semantic embeddings from anyrag for
/// KV cache priming, falling back to keyword-based routing when unavailable.
///
/// # Architecture
///
/// ```text
/// EmbeddingRouter
///   ├── config: EmbeddingRouterConfig    (anyrag URL, timeout, auth)
///   ├── keyword_router: KeywordRouter    (local fallback)
///   ├── projector: Box<dyn EmbeddingProjector>  (dim projection)
///   └── client: reqwest::Client          (HTTP connection pooling)
/// ```
pub struct EmbeddingRouter {
    config: EmbeddingRouterConfig,
    keyword_router: KeywordRouter,
    projector: Box<dyn EmbeddingProjector>,
    client: Client,
}

impl EmbeddingRouter {
    /// Build a new embedding router.
    ///
    /// - `config`: Connection settings for the anyrag server.
    /// - `domains`: Domain definitions for the internal `KeywordRouter` fallback.
    /// - `projector`: Strategy for projecting embeddings to the draft model's dim.
    pub fn new(
        config: EmbeddingRouterConfig,
        domains: Vec<super::types::DomainConfig>,
        projector: Box<dyn EmbeddingProjector>,
    ) -> Self {
        let client = Client::builder()
            .timeout(std::time::Duration::from_millis(config.timeout_ms))
            .build()
            .expect("reqwest client construction should not fail");

        Self {
            config,
            keyword_router: KeywordRouter::new(domains),
            projector,
            client,
        }
    }

    /// Async routing with embedding retrieval for KV cache priming.
    ///
    /// Three-tier fallback:
    /// 1. `POST /search/embedding` → embedding + domain
    /// 2. `POST /classify/domain`  → domain only
    /// 3. `KeywordRouter::route()` → local keyword match
    ///
    /// Any failure degrades gracefully to the next tier.
    pub async fn route_async(&self, prompt: &str) -> EmbeddingRouteDecision {
        // Tier 1: Try embedding search
        match self.try_embedding_search(prompt).await {
            Some(decision) => return decision,
            None => {
                eprintln!(
                    "[embedding_router] embedding search failed, \
                     falling back to domain classify"
                );
            }
        }

        // Tier 2: Try domain classify (if enabled)
        if self.config.classify_domain {
            match self.try_domain_classify(prompt).await {
                Some(decision) => return decision,
                None => {
                    eprintln!(
                        "[embedding_router] domain classify failed, \
                         falling back to keyword router"
                    );
                }
            }
        }

        // Tier 3: Keyword fallback (always succeeds)
        let route = self.keyword_router.route(prompt);
        EmbeddingRouteDecision {
            route,
            embedding: None,
            embedding_source: None,
        }
    }

    /// Project an embedding to the draft model's hidden dimension.
    ///
    /// Convenience method that delegates to the configured projector.
    pub fn project_embedding(&self, embedding: &[f32], target_dim: usize) -> Vec<f32> {
        self.projector.project(embedding, target_dim)
    }

    // -----------------------------------------------------------------------
    // Internal: Tier 1 — Embedding search
    // -----------------------------------------------------------------------

    async fn try_embedding_search(&self, prompt: &str) -> Option<EmbeddingRouteDecision> {
        let url = format!("{}/search/embedding", self.config.anyrag_url);
        let body = EmbeddingSearchRequest {
            query: prompt.to_string(),
            context_files: None,
            limit: 1,
        };

        let request = self.client.post(&url).json(&body);
        let request = match &self.config.auth_token {
            Some(token) => request.bearer_auth(token),
            None => request,
        };

        let response = request.send().await.ok()?;
        let search_result: EmbeddingSearchResponse = response.json().await.ok()?;

        // Use keyword router for the domain decision (embedding search doesn't
        // return a domain; we derive it from the same keyword logic).
        let route = self.keyword_router.route(prompt);

        Some(EmbeddingRouteDecision {
            route,
            embedding: Some(search_result.result.embedding),
            embedding_source: Some(search_result.result.source),
        })
    }

    // -----------------------------------------------------------------------
    // Internal: Tier 2 — Domain classify
    // -----------------------------------------------------------------------

    async fn try_domain_classify(&self, prompt: &str) -> Option<EmbeddingRouteDecision> {
        let url = format!("{}/classify/domain", self.config.anyrag_url);

        #[derive(serde::Serialize)]
        struct ClassifyRequest {
            prompt: String,
        }

        let body = ClassifyRequest {
            prompt: prompt.to_string(),
        };

        let request = self.client.post(&url).json(&body);
        let request = match &self.config.auth_token {
            Some(token) => request.bearer_auth(token),
            None => request,
        };

        let response = request.send().await.ok()?;
        let result: ClassifyDomainResponse = response.json().await.ok()?;

        // Build a RouteDecision from the classified domain.
        // We use the keyword router's paths for the classified domain
        // (LoRA, pruner) if they match a known domain.
        let keyword_decision = self.keyword_router.route(prompt);

        // Override domain and confidence with the anyrag result if it
        // provides higher confidence.
        let route = if result.confidence > keyword_decision.confidence {
            RouteDecision {
                domain: result.domain,
                confidence: result.confidence,
                lora_path: keyword_decision.lora_path,
                pruner_path: keyword_decision.pruner_path,
            }
        } else {
            keyword_decision
        };

        Some(EmbeddingRouteDecision {
            route,
            embedding: None,
            embedding_source: None,
        })
    }
}

// ---------------------------------------------------------------------------
// PromptRouter impl (sync — delegates to KeywordRouter)
// ---------------------------------------------------------------------------

impl PromptRouter for EmbeddingRouter {
    /// Sync routing via keyword fallback.
    ///
    /// Embedding retrieval requires async; callers that can await should
    /// use [`route_async()`](Self::route_async) instead.
    fn route(&self, prompt: &str) -> RouteDecision {
        self.keyword_router.route(prompt)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::router::projector::TruncatePadProjector;
    use crate::router::types::DomainConfig;

    fn make_domains() -> Vec<DomainConfig> {
        vec![
            DomainConfig {
                name: "sudoku".into(),
                keywords: vec![
                    "sudoku".into(),
                    "puzzle".into(),
                    "grid".into(),
                    "9x9".into(),
                    "digit".into(),
                ],
                pruner: None,
                lora: None,
                reader_lora: None,
                writer_lora: None,
                native_pruner: Some("sudoku".into()),
                truncation: None,
                reasoning_retention: None,
            },
            DomainConfig {
                name: "rust_code".into(),
                keywords: vec![
                    "rust".into(),
                    "cargo".into(),
                    "axum".into(),
                    "tokio".into(),
                    "trait".into(),
                    "impl".into(),
                    "compile".into(),
                ],
                pruner: Some("syn_validator.wasm".into()),
                lora: None,
                reader_lora: None,
                writer_lora: None,
                native_pruner: None,
                truncation: None,
                reasoning_retention: None,
            },
            DomainConfig {
                name: "general".into(),
                keywords: vec![],
                pruner: None,
                lora: None,
                reader_lora: None,
                writer_lora: None,
                native_pruner: Some("no_pruner".into()),
                truncation: None,
                reasoning_retention: None,
            },
        ]
    }

    fn make_config() -> EmbeddingRouterConfig {
        EmbeddingRouterConfig {
            anyrag_url: "http://localhost:9090".into(),
            timeout_ms: 50,
            classify_domain: true,
            auth_token: None,
        }
    }

    #[test]
    fn test_route_delegates_to_keyword_router() {
        let router = EmbeddingRouter::new(
            make_config(),
            make_domains(),
            Box::new(TruncatePadProjector),
        );

        let decision = router.route("solve this sudoku puzzle");
        assert_eq!(decision.domain, "sudoku");
        assert!(decision.confidence > 0.0);
    }

    #[test]
    fn test_route_falls_back_to_general() {
        let router = EmbeddingRouter::new(
            make_config(),
            make_domains(),
            Box::new(TruncatePadProjector),
        );

        let decision = router.route("what is the meaning of life?");
        assert_eq!(decision.domain, "general");
    }

    #[test]
    fn test_project_embedding_delegates_to_projector() {
        let router = EmbeddingRouter::new(
            make_config(),
            make_domains(),
            Box::new(TruncatePadProjector),
        );

        let embedding = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let projected = router.project_embedding(&embedding, 3);
        assert_eq!(projected, vec![1.0, 2.0, 3.0]);
    }

    #[tokio::test]
    async fn test_route_async_falls_back_to_keyword_when_server_down() {
        let config = EmbeddingRouterConfig {
            anyrag_url: "http://localhost:19999".into(), // unreachable port
            timeout_ms: 50,
            classify_domain: true,
            auth_token: None,
        };

        let router = EmbeddingRouter::new(config, make_domains(), Box::new(TruncatePadProjector));

        let decision = router.route_async("solve this sudoku puzzle").await;
        // Should fall back to keyword routing
        assert_eq!(decision.route.domain, "sudoku");
        assert!(decision.embedding.is_none());
        assert!(decision.embedding_source.is_none());
    }

    #[tokio::test]
    async fn test_route_async_no_embedding_when_server_down() {
        let config = EmbeddingRouterConfig {
            anyrag_url: "http://localhost:19999".into(),
            timeout_ms: 50,
            classify_domain: false, // skip domain classify tier
            auth_token: None,
        };

        let router = EmbeddingRouter::new(config, make_domains(), Box::new(TruncatePadProjector));

        let decision = router.route_async("write Rust code with axum").await;
        assert_eq!(decision.route.domain, "rust_code");
        assert!(decision.embedding.is_none());
    }

    #[test]
    fn test_embedding_search_request_serialization() {
        let request = EmbeddingSearchRequest {
            query: "fn validate_token(".into(),
            context_files: Some(vec!["auth.rs".into()]),
            limit: 1,
        };

        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("fn validate_token("));
        assert!(json.contains("auth.rs"));
        assert!(json.contains("1"));
    }

    #[test]
    fn test_embedding_search_request_skips_none_context() {
        let request = EmbeddingSearchRequest {
            query: "test query".into(),
            context_files: None,
            limit: 5,
        };

        let json = serde_json::to_string(&request).unwrap();
        assert!(!json.contains("context_files"));
    }
}
