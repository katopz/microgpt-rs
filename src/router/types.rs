//! Router types for config-driven domain routing.
//!
//! Defines the data structures used by the prompt router system:
//! - [`RouteDecision`] — output of classifying a prompt
//! - [`ExpertBundle`] — a loadable pruner + optional LoRA adapter pair
//! - [`DomainConfig`] — a domain definition loaded from `domains.toml`
//! - [`RouterConfig`] — top-level config wrapping all domains
//! - [`EmbeddingRouteDecision`] — routing decision with optional embedding (Plan 024)
//! - [`EmbeddingRouterConfig`] — config for the embedding router (Plan 024)
//! - [`EmbeddingExpertBundle`] — expert bundle with projected embedding (Plan 024)

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::speculative::types::ScreeningPruner;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Result of routing a prompt to a domain.
///
/// Produced by [`crate::router::router::PromptRouter::route`]. The `domain`
/// string is the key used to look up an [`ExpertBundle`] in the registry.
#[derive(Debug, Clone)]
pub struct RouteDecision {
    /// The matched domain name (e.g., `"sudoku"`, `"rust_code"`, `"general"`).
    pub domain: String,
    /// Heuristic confidence in `[0.0, 1.0]`. Higher is better.
    pub confidence: f32,
    /// Optional LoRA adapter path associated with the domain.
    pub lora_path: Option<PathBuf>,
    /// Optional WASM pruner path associated with the domain.
    pub pruner_path: Option<PathBuf>,
}

// ---------------------------------------------------------------------------
// ExpertBundle — what the registry serves
// ---------------------------------------------------------------------------

/// A loadable expert bundle: a [`ScreeningPruner`] + optional LoRA adapter path.
///
/// The registry maps domain names to these bundles. When the router classifies
/// a prompt, the caller fetches the matching bundle and uses its pruner for
/// DDTree construction.
pub struct ExpertBundle {
    /// Domain name this bundle belongs to.
    pub domain: String,
    /// The screening pruner used to score token relevance during DDTree.
    pub pruner: Box<dyn ScreeningPruner>,
    /// Path to a LoRA adapter file (loading deferred to a future plan).
    pub lora_path: Option<PathBuf>,
}

// ---------------------------------------------------------------------------
// Config types (loaded from TOML)
// ---------------------------------------------------------------------------

/// A single domain definition loaded from `domains.toml`.
///
/// ```toml
/// [[domain]]
/// name = "sudoku"
/// keywords = ["sudoku", "puzzle", "grid", "9x9", "digit"]
/// native_pruner = "sudoku"
/// ```
///
/// ```toml
/// [[domain]]
/// name = "rust_code"
/// keywords = ["rust", "cargo", "axum", "tokio", "trait", "impl", "compile"]
/// pruner = "syn_validator.wasm"
/// lora = "rust_code_lora.bin"
/// ```
#[derive(Debug, Clone, Deserialize)]
pub struct DomainConfig {
    /// Unique domain name (used as registry key).
    pub name: String,
    /// Keywords used by [`crate::router::keyword::KeywordRouter`] for scoring.
    #[serde(default)]
    pub keywords: Vec<String>,
    /// Path to a WASM pruner file (relative to pruner directory).
    #[serde(default)]
    pub pruner: Option<String>,
    /// Path to a LoRA adapter file (relative to pruner directory).
    #[serde(default)]
    pub lora: Option<String>,
    /// Name of a built-in native pruner: `"sudoku"`, `"tactical"`, `"no_pruner"`.
    #[serde(default)]
    pub native_pruner: Option<String>,
}

/// Top-level router configuration loaded from `domains.toml`.
///
/// ```toml
/// [[domain]]
/// name = "sudoku"
/// keywords = ["sudoku", "puzzle"]
/// native_pruner = "sudoku"
///
/// [[domain]]
/// name = "general"
/// keywords = []
/// native_pruner = "no_pruner"
/// ```
#[derive(Debug, Clone, Deserialize)]
pub struct RouterConfig {
    /// All domain definitions.
    #[serde(default)]
    pub domain: Vec<DomainConfig>,
}

// ---------------------------------------------------------------------------
// Embedding Router types (Plan 024)
// ---------------------------------------------------------------------------

/// Result of routing with optional retrieved embedding for KV cache priming.
///
/// Wraps a [`RouteDecision`] with an optional embedding vector retrieved from
/// anyrag. The embedding is projected to the draft model's hidden dimension
/// and injected via `dflash_predict_conditioned_with` for semantic context.
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
///
/// Loaded from `domains.toml` under the `[embedding_router]` section.
/// Controls how the router connects to anyrag for embedding retrieval.
///
/// ```toml
/// [embedding_router]
/// anyrag_url = "http://localhost:9090"
/// timeout_ms = 200
/// classify_domain = true
/// auth_token = "optional-jwt-token"
/// ```
#[derive(Debug, Clone, Deserialize)]
pub struct EmbeddingRouterConfig {
    /// anyrag server URL (e.g., `"http://localhost:9090"`).
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

fn default_timeout() -> u64 {
    200
}

fn default_true() -> bool {
    true
}

// ---------------------------------------------------------------------------
// anyrag API types (Plan 024)
// ---------------------------------------------------------------------------

/// Response from anyrag `/search/embedding` endpoint.
#[derive(Debug, Deserialize)]
pub struct EmbeddingSearchResponse {
    pub result: EmbeddingSearchResult,
}

/// A single embedding search result with vector, score, and source.
#[derive(Debug, Deserialize)]
pub struct EmbeddingSearchResult {
    /// Raw embedding vector from the top matching document.
    pub embedding: Vec<f32>,
    /// Cosine similarity score `[0.0, 1.0]`.
    pub score: f32,
    /// Source file/chunk that produced this embedding.
    pub source: String,
}

/// Request body for anyrag `/search/embedding`.
#[derive(Debug, Serialize)]
pub struct EmbeddingSearchRequest {
    /// The query text to search for.
    pub query: String,
    /// Optional file context to bias retrieval toward specific files.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_files: Option<Vec<String>>,
    /// Maximum number of results to return.
    pub limit: u32,
}

// ---------------------------------------------------------------------------
// Embedding Expert Bundle (Plan 024)
// ---------------------------------------------------------------------------

/// A screening pruner combined with an optional projected embedding for
/// KV cache priming. Bundles everything the speculative step needs:
/// pruner + embedding + LoRA adapter path.
///
/// The speculative step checks `projected_embedding` to decide between:
/// - `speculative_step_conditioned_with` (target hidden state)
/// - `speculative_step_embedding_conditioned` (retrieved embedding)
/// - `speculative_step_with` (no conditioning)
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
