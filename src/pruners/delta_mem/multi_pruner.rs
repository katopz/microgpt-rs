//! ScreeningPruner with per-domain memory states (MSW variant).
//!
//! Paper finding (Table 2): MSW helps smaller models most.
//!   SmolLM3-3B: 26.08 → 36.96 (+10.88) with MSW
//!   Qwen3-8B:   47.20 → 50.86 (+3.66) with SSW
//!
//! Our equivalent: domains with less WASM validator coverage
//! benefit most from per-domain memory states.

use std::collections::HashMap;

use crate::speculative::types::ScreeningPruner;

use super::pruner::{CorrectionMode, MemorySteeredPruner, WriteGranularity};
use super::state::DeltaMemoryConfig;

/// ScreeningPruner with per-domain memory states.
///
/// Routes DDTree build to the appropriate domain's pruner.
/// Each domain gets its own `MemorySteeredPruner` with independent memory.
pub struct MultiDomainMemoryPruner<P: ScreeningPruner> {
    /// Per-domain pruners (each wraps inner + memory).
    pruners: HashMap<String, MemorySteeredPruner<P>>,
    /// Current domain (set by caller before DDTree build).
    current_domain: Option<String>,
    /// Default config for new domain pruners.
    default_config: DeltaMemoryConfig,
    /// Default alpha for new pruners.
    default_alpha: f32,
    /// Default correction mode.
    default_mode: CorrectionMode,
    /// Default write granularity.
    default_granularity: WriteGranularity,
    /// Factory for creating inner pruners.
    inner_factory: fn() -> P,
}

impl<P: ScreeningPruner> MultiDomainMemoryPruner<P> {
    /// Create a new multi-domain memory pruner.
    ///
    /// The `factory` closure produces a fresh inner pruner for each new domain.
    /// Use `|| NoScreeningPruner` or similar for unit-struct pruners.
    pub fn new(
        config: DeltaMemoryConfig,
        alpha: f32,
        mode: CorrectionMode,
        granularity: WriteGranularity,
        factory: fn() -> P,
    ) -> Self {
        Self {
            pruners: HashMap::new(),
            current_domain: None,
            default_config: config,
            default_alpha: alpha,
            default_mode: mode,
            default_granularity: granularity,
            inner_factory: factory,
        }
    }

    /// Set the current domain for the next DDTree build.
    pub fn set_domain(&mut self, domain: &str) {
        self.current_domain = Some(domain.to_string());
        self.ensure_domain(domain);
    }

    /// Get or create a domain's pruner.
    fn ensure_domain(&mut self, domain: &str) {
        if !self.pruners.contains_key(domain) {
            let inner = (self.inner_factory)();
            let pruner = MemorySteeredPruner::new(
                inner,
                self.default_config.clone(),
                self.default_alpha,
                self.default_mode,
                self.default_granularity,
            );
            self.pruners.insert(domain.to_string(), pruner);
        }
    }

    /// Get the current domain's pruner (mutable).
    pub fn current_pruner(&mut self) -> Option<&mut MemorySteeredPruner<P>> {
        let domain = self.current_domain.as_deref()?;
        self.pruners.get_mut(domain)
    }

    /// Get a specific domain's pruner (mutable).
    pub fn domain_pruner(&mut self, domain: &str) -> Option<&mut MemorySteeredPruner<P>> {
        self.pruners.get_mut(domain)
    }

    /// Reset all domain pruners.
    pub fn reset_all(&mut self) {
        for pruner in self.pruners.values_mut() {
            pruner.reset();
        }
    }

    /// Number of domains.
    pub fn domain_count(&self) -> usize {
        self.pruners.len()
    }

    /// Get domain names.
    pub fn domains(&self) -> Vec<&str> {
        self.pruners.keys().map(|s| s.as_str()).collect()
    }

    /// Get current domain name.
    pub fn current_domain(&self) -> Option<&str> {
        self.current_domain.as_deref()
    }
}

impl<P: ScreeningPruner> ScreeningPruner for MultiDomainMemoryPruner<P> {
    fn relevance(&self, depth: usize, token_idx: usize, parent_token: &[usize]) -> f32 {
        match self.current_domain.as_deref() {
            Some(domain) => {
                if let Some(pruner) = self.pruners.get(domain) {
                    pruner.relevance(depth, token_idx, parent_token)
                } else {
                    1.0 // Default: no pruning for unknown domain
                }
            }
            None => 1.0, // No domain set: no pruning
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::speculative::types::NoScreeningPruner;

    fn make_multi() -> MultiDomainMemoryPruner<NoScreeningPruner> {
        MultiDomainMemoryPruner::new(
            DeltaMemoryConfig {
                rank: 4,
                ..Default::default()
            },
            2.0,
            CorrectionMode::OutputSide,
            WriteGranularity::Token,
            || NoScreeningPruner,
        )
    }

    #[test]
    fn test_no_domain_returns_one() {
        let pruner = make_multi();
        let rel = pruner.relevance(0, 0, &[]);
        assert!((rel - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_set_domain_creates_pruner() {
        let mut pruner = make_multi();
        pruner.set_domain("coding");
        assert_eq!(pruner.domain_count(), 1);
        assert_eq!(pruner.current_domain(), Some("coding"));
    }

    #[test]
    fn test_domain_pruner_routes_correctly() {
        let mut pruner = make_multi();
        pruner.set_domain("coding");

        // Should route to coding pruner
        let rel = pruner.relevance(0, 0, &[]);
        assert!((rel - 1.0).abs() < 1e-6); // Fresh memory, no correction
    }

    #[test]
    fn test_multiple_domains_isolated() {
        let mut pruner = make_multi();

        // Set domain A and observe
        pruner.set_domain("coding");
        let p = pruner.current_pruner().unwrap();
        assert_eq!(p.memory().update_count(), 0);

        // Switch to domain B
        pruner.set_domain("math");
        assert_eq!(pruner.domain_count(), 2);

        // Domain B should be independent
        let rel = pruner.relevance(0, 0, &[]);
        assert!((rel - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_reset_all() {
        let mut pruner = make_multi();
        pruner.set_domain("a");
        pruner.set_domain("b");
        pruner.reset_all();
        // Should not panic, all states zeroed
        assert_eq!(pruner.domain_count(), 2);
    }

    #[test]
    fn test_unknown_domain_returns_one() {
        let pruner = MultiDomainMemoryPruner::<NoScreeningPruner> {
            pruners: HashMap::new(),
            current_domain: Some("nonexistent".to_string()),
            default_config: DeltaMemoryConfig::default(),
            default_alpha: 2.0,
            default_mode: CorrectionMode::OutputSide,
            default_granularity: WriteGranularity::Token,
            inner_factory: || NoScreeningPruner,
        };
        let rel = pruner.relevance(0, 0, &[]);
        assert!((rel - 1.0).abs() < 1e-6);
    }
}
