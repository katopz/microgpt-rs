//! Router types for config-driven domain routing.
//!
//! Defines the data structures used by the prompt router system:
//! - [`RouteDecision`] — output of classifying a prompt
//! - [`ExpertBundle`] — a loadable pruner + optional LoRA adapter pair
//! - [`DomainConfig`] — a domain definition loaded from `domains.toml`
//! - [`RouterConfig`] — top-level config wrapping all domains

use std::path::PathBuf;

use serde::Deserialize;

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
