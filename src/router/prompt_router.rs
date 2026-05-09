//! Prompt router trait — classify a prompt into a domain.
//!
//! The [`PromptRouter`] trait has a single method that maps a prompt string to
//! a [`RouteDecision`]. It is called **once per request** (batch-level), never
//! inside the DDTree hot path, so latency is not critical.
//!
//! # Implementations
//!
//! - [`crate::router::keyword::KeywordRouter`] — keyword-count scoring (V1, ~80% accuracy)
//! - Future: embedding-based router via anyrag (V2, ~95% accuracy)

use super::types::RouteDecision;

/// Classifies a prompt into a domain, returning a routing decision.
///
/// Implementations must be `Send + Sync` so the router can be shared across
/// threads (e.g., stored in an axum `State` behind an `Arc`).
pub trait PromptRouter: Send + Sync {
    /// Route a prompt to the best-matching domain.
    ///
    /// Returns a [`RouteDecision`] containing the domain name, confidence score,
    /// and optional LoRA / pruner paths associated with that domain.
    fn route(&self, prompt: &str) -> RouteDecision;
}
