//! Keyword-based prompt router — V1 domain classifier.
//!
//! Scores each domain by counting how many of its keywords appear in the
//! prompt (case-insensitive). The domain with the highest keyword-count wins.
//! Falls back to `"general"` when no keywords match.
//!
//! Accuracy is ~80% for obvious domains. Embedding-based routing (anyrag
//! integration, Plan 005) is the V2 upgrade path.

use super::prompt_router::PromptRouter;
use super::types::{DomainConfig, RouteDecision};

// ---------------------------------------------------------------------------
// KeywordRouter
// ---------------------------------------------------------------------------

/// A prompt router that scores domains by keyword-count overlap.
///
/// # Scoring
///
/// For each domain, count how many of its `keywords` appear (case-insensitive
/// substring match) in the prompt. The domain with the highest count wins.
///
/// # Confidence
///
/// `confidence = best_score / total_keywords_across_all_domains`
///
/// This is a simple heuristic — not calibrated, but sufficient for V1.
pub struct KeywordRouter {
    domains: Vec<DomainConfig>,
    /// Total number of keywords across all domains (pre-computed for confidence).
    total_keywords: usize,
}

impl KeywordRouter {
    /// Build a new keyword router from domain configs.
    ///
    /// The first domain named `"general"` (if present) is used as the fallback
    /// when no keywords match any domain.
    pub fn new(domains: Vec<DomainConfig>) -> Self {
        let total_keywords: usize = domains.iter().map(|d| d.keywords.len()).sum();
        Self {
            domains,
            total_keywords,
        }
    }

    /// Score a single domain against the prompt.
    ///
    /// Returns the number of keyword matches (case-insensitive substring).
    fn score_domain(&self, prompt_lower: &str, domain: &DomainConfig) -> usize {
        domain
            .keywords
            .iter()
            .filter(|kw| prompt_lower.contains(&kw.to_lowercase()))
            .count()
    }

    /// Find the best-matching domain and its score.
    ///
    /// Returns `None` if there are no domains configured.
    fn best_match(&self, prompt: &str) -> Option<(&DomainConfig, usize)> {
        let prompt_lower = prompt.to_lowercase();

        let mut best: Option<(&DomainConfig, usize)> = None;

        for domain in &self.domains {
            let score = self.score_domain(&prompt_lower, domain);
            match best {
                Some((_, best_score)) if score <= best_score => {}
                _ => best = Some((domain, score)),
            }
        }

        best
    }

    /// Find the `"general"` fallback domain.
    fn general_domain(&self) -> Option<&DomainConfig> {
        self.domains.iter().find(|d| d.name == "general")
    }
}

impl PromptRouter for KeywordRouter {
    fn route(&self, prompt: &str) -> RouteDecision {
        let total_kw = match self.total_keywords {
            0 => 1, // avoid division by zero
            n => n,
        };

        match self.best_match(prompt) {
            // No domains configured at all — return a synthetic "general".
            None => RouteDecision {
                domain: "general".to_string(),
                confidence: 0.0,
                lora_path: None,
                pruner_path: None,
            },

            // Best match found with score > 0 — use it.
            Some((domain, score)) if score > 0 => RouteDecision {
                domain: domain.name.clone(),
                confidence: score as f32 / total_kw as f32,
                lora_path: domain.lora.clone().map(Into::into),
                pruner_path: domain.pruner.clone().map(Into::into),
            },

            // Best match has score == 0 (no keywords matched) — fall back to general.
            Some((domain, 0)) => match self.general_domain() {
                Some(general) => RouteDecision {
                    domain: general.name.clone(),
                    confidence: 0.0,
                    lora_path: general.lora.clone().map(Into::into),
                    pruner_path: general.pruner.clone().map(Into::into),
                },
                None => RouteDecision {
                    domain: domain.name.clone(),
                    confidence: 0.0,
                    lora_path: domain.lora.clone().map(Into::into),
                    pruner_path: domain.pruner.clone().map(Into::into),
                },
            },

            // Should be unreachable (score is usize, can't be negative).
            Some((domain, _)) => RouteDecision {
                domain: domain.name.clone(),
                confidence: 0.0,
                lora_path: domain.lora.clone().map(Into::into),
                pruner_path: domain.pruner.clone().map(Into::into),
            },
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

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
                native_pruner: Some("sudoku".into()),
            },
            DomainConfig {
                name: "pathfinding".into(),
                keywords: vec![
                    "path".into(),
                    "maze".into(),
                    "bear".into(),
                    "terrain".into(),
                    "tactical".into(),
                    "grid".into(),
                ],
                pruner: None,
                lora: None,
                native_pruner: Some("tactical".into()),
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
                native_pruner: None,
            },
            DomainConfig {
                name: "py2rs".into(),
                keywords: vec![
                    "python".into(),
                    "rewrite".into(),
                    "fastapi".into(),
                    "flask".into(),
                    "translate".into(),
                ],
                pruner: Some("syn_validator.wasm".into()),
                lora: Some("py2rs_lora.bin".into()),
                native_pruner: None,
            },
            DomainConfig {
                name: "general".into(),
                keywords: vec![],
                pruner: None,
                lora: None,
                native_pruner: Some("no_pruner".into()),
            },
        ]
    }

    #[test]
    fn test_route_sudoku() {
        let router = KeywordRouter::new(make_domains());
        let decision = router.route("solve this sudoku puzzle for me");
        assert_eq!(decision.domain, "sudoku");
        assert!(decision.confidence > 0.0);
    }

    #[test]
    fn test_route_rust_code() {
        let router = KeywordRouter::new(make_domains());
        let decision = router.route("write Rust code for an HTTP server using axum and tokio");
        assert_eq!(decision.domain, "rust_code");
        assert_eq!(decision.pruner_path, Some("syn_validator.wasm".into()));
    }

    #[test]
    fn test_route_falls_back_to_general() {
        let router = KeywordRouter::new(make_domains());
        let decision = router.route("what is the meaning of life?");
        assert_eq!(decision.domain, "general");
        assert_eq!(decision.confidence, 0.0);
    }

    #[test]
    fn test_route_py2rs() {
        let router = KeywordRouter::new(make_domains());
        let decision = router.route("translate this FastAPI python code to rust");
        // "python", "fastapi", "translate", "rust" → py2rs should score higher
        assert_eq!(decision.domain, "py2rs");
        assert_eq!(decision.lora_path, Some("py2rs_lora.bin".into()));
    }

    #[test]
    fn test_multi_keyword_higher_confidence() {
        let router = KeywordRouter::new(make_domains());

        let one_kw = router.route("sudoku");
        let three_kw = router.route("solve this sudoku puzzle with a 9x9 grid");

        // More keyword matches → higher confidence.
        assert!(three_kw.confidence > one_kw.confidence);
    }

    #[test]
    fn test_empty_router_returns_general() {
        let router = KeywordRouter::new(vec![]);
        let decision = router.route("anything");
        assert_eq!(decision.domain, "general");
        assert_eq!(decision.confidence, 0.0);
    }

    #[test]
    fn test_case_insensitive() {
        let router = KeywordRouter::new(make_domains());
        let decision = router.route("SUDOKU puzzle GRID");
        assert_eq!(decision.domain, "sudoku");
        assert!(decision.confidence > 0.0);
    }
}
