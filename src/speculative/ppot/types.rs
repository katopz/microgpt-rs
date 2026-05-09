//! PPoT types: TokenRule enum, PpotConfig struct.
//!
//! Distilled from "Probabilistic Programs of Thought" (arXiv:2604.17290).
//! TokenRule defines support sets for constrained resampling;
//! PpotConfig holds all tunable parameters for PPoT rescue.

// ── Token Rule ─────────────────────────────────────────────────

/// Token rule defining a constrained support set for PPoT resampling.
///
/// Each variant maps to a subset of the vocabulary relevant to a particular
/// domain (digits, operators, comparisons). [`TokenRule::All`] resamples
/// from the full vocabulary (unrestricted).
///
/// Support sets are heuristic defaults for character-level tokenizers.
/// BPE tokenizers should override via [`PpotConfig::custom_support`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TokenRule {
    /// Digits `0-9`: token IDs 0–9 (character-level convention).
    Digit,
    /// Comparison operators: `<`, `>`, `=` (token IDs by convention).
    Compare,
    /// Arithmetic operators: `+`, `-`, `*`, `/` (token IDs by convention).
    Arithmetic,
    /// Augmented assignment: combines arithmetic + `=` variants.
    Augment,
    /// Full vocabulary: unrestricted resampling at all positions.
    All,
}

impl TokenRule {
    /// Returns the number of defined rules (excluding `All`).
    pub const fn rule_count() -> usize {
        4 // Digit, Compare, Arithmetic, Augment
    }

    /// All strategy rules in canonical cycle order.
    pub const STRATEGIES: [TokenRule; 5] = [
        TokenRule::Digit,
        TokenRule::Arithmetic,
        TokenRule::Compare,
        TokenRule::Augment,
        TokenRule::All,
    ];

    /// Returns the support set (allowed token IDs) for this rule.
    ///
    /// For character-level tokenizers, these are direct mappings.
    /// For BPE tokenizers, use [`PpotConfig::custom_support`] to override.
    pub fn support(&self, vocab_size: usize) -> Vec<usize> {
        match self {
            TokenRule::Digit => (0..10.min(vocab_size)).collect(),
            TokenRule::Compare => {
                // '<' = 60, '>' = 62, '=' = 61, '!' = 33 in ASCII
                // For character-level (vocab < 256): use ASCII codes
                // For BPE (vocab >= 256): fall back to small range
                if vocab_size < 256 {
                    vec![33, 60, 61, 62]
                        .into_iter()
                        .filter(|&t| t < vocab_size)
                        .collect()
                } else {
                    // BPE fallback: can't reliably identify operator tokens
                    (0..vocab_size).collect()
                }
            }
            TokenRule::Arithmetic => {
                // '+' = 43, '-' = 45, '*' = 42, '/' = 47
                if vocab_size < 256 {
                    vec![42, 43, 45, 47]
                        .into_iter()
                        .filter(|&t| t < vocab_size)
                        .collect()
                } else {
                    (0..vocab_size).collect()
                }
            }
            TokenRule::Augment => {
                // Augmented assignment operators: same as arithmetic + '='
                if vocab_size < 256 {
                    vec![42, 43, 45, 47, 61]
                        .into_iter()
                        .filter(|&t| t < vocab_size)
                        .collect()
                } else {
                    (0..vocab_size).collect()
                }
            }
            TokenRule::All => (0..vocab_size).collect(),
        }
    }

    /// Returns the support as a bitmask-compatible slice for fast membership checks.
    /// For hot paths, prefer pre-computing support once via [`PpotConfig`].
    pub fn index(&self) -> usize {
        match self {
            TokenRule::Digit => 0,
            TokenRule::Compare => 1,
            TokenRule::Arithmetic => 2,
            TokenRule::Augment => 3,
            TokenRule::All => 4,
        }
    }

    /// Parse from string (for config file compatibility).
    pub fn from_str_ignore_case(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "digit" => Some(TokenRule::Digit),
            "compare" => Some(TokenRule::Compare),
            "arithmetic" => Some(TokenRule::Arithmetic),
            "augment" => Some(TokenRule::Augment),
            "all" => Some(TokenRule::All),
            _ => None,
        }
    }
}

impl std::fmt::Display for TokenRule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TokenRule::Digit => write!(f, "digit"),
            TokenRule::Compare => write!(f, "compare"),
            TokenRule::Arithmetic => write!(f, "arithmetic"),
            TokenRule::Augment => write!(f, "augment"),
            TokenRule::All => write!(f, "all"),
        }
    }
}

// ── PPoT Config ────────────────────────────────────────────────

/// Configuration for PPoT (Probabilistic Programs of Thought) rescue.
///
/// Combines Plan 026 (logit-parameterized CPU resampling) and
/// Plan 027 (adaptive rescue with rejection memory) parameters.
///
/// # Defaults
///
/// All fields have sensible defaults. PPoT is **opt-in** via [`PpotConfig::enabled`].
/// When disabled, zero overhead is incurred on the speculative decoding path.
///
/// # Example
///
/// ```ignore
/// use speculative::ppot::PpotConfig;
///
/// let config = PpotConfig {
///     enabled: true,
///     num_samples: 10,
///     ..PpotConfig::default()
/// };
/// ```
#[derive(Clone, Debug)]
pub struct PpotConfig {
    // ── Plan 026: Core PPoT ──
    /// Whether PPoT rescue is enabled (opt-in).
    pub enabled: bool,
    /// Shannon entropy threshold for identifying high-entropy positions.
    /// Positions with `H(i) > threshold` are candidates for resampling.
    pub entropy_threshold: f32,
    /// Number of variant paths to resample per rescue attempt.
    pub num_samples: usize,
    /// Default token rule for resampling.
    pub rule: TokenRule,
    /// Whether to enforce different-value constraint (avoid reproducing original).
    pub different_constraint: bool,

    // ── Plan 027: Adaptive PPoT ──
    /// Whether to use adaptive threshold adjustment (TRT-inspired).
    /// When true, entropy threshold is raised after success and lowered after failure.
    pub adaptive_threshold: bool,
    /// Minimum entropy threshold (adaptive lower bound).
    pub entropy_threshold_min: f32,
    /// Maximum entropy threshold (adaptive upper bound).
    pub entropy_threshold_max: f32,
    /// Amount to lower threshold on rescue failure.
    pub threshold_lower_on_fail: f32,
    /// Amount to raise threshold on rescue success.
    pub threshold_raise_on_success: f32,
    /// Maximum number of rejection insights to retain (ring buffer size).
    pub max_insights: usize,
    /// Whether adaptive rescue is enabled (requires PPoT enabled).
    pub adaptive_enabled: bool,
}

impl Default for PpotConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            entropy_threshold: 0.5,
            num_samples: 10,
            rule: TokenRule::All,
            different_constraint: true,

            adaptive_threshold: true,
            entropy_threshold_min: 0.3,
            entropy_threshold_max: 1.0,
            threshold_lower_on_fail: 0.1,
            threshold_raise_on_success: 0.05,
            max_insights: 64,
            adaptive_enabled: true,
        }
    }
}

impl PpotConfig {
    /// PPoT config with sensible defaults for character-level models.
    pub fn for_char_level() -> Self {
        Self {
            enabled: true,
            rule: TokenRule::All,
            ..Self::default()
        }
    }

    /// PPoT config optimized for math/expression tasks.
    pub fn for_math() -> Self {
        Self {
            enabled: true,
            rule: TokenRule::Digit,
            entropy_threshold: 0.3,
            ..Self::default()
        }
    }

    /// Clamp the adaptive threshold to `[min, max]` bounds.
    #[inline]
    pub fn clamp_threshold(&self, threshold: f32) -> f32 {
        threshold.clamp(self.entropy_threshold_min, self.entropy_threshold_max)
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token_rule_support_digit() {
        let rule = TokenRule::Digit;
        let support = rule.support(27);
        assert_eq!(support, vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    }

    #[test]
    fn test_token_rule_support_digit_small_vocab() {
        let rule = TokenRule::Digit;
        let support = rule.support(5);
        assert_eq!(support, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn test_token_rule_support_all() {
        let rule = TokenRule::All;
        let support = rule.support(10);
        assert_eq!(support, vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    }

    #[test]
    fn test_token_rule_from_str() {
        assert_eq!(
            TokenRule::from_str_ignore_case("digit"),
            Some(TokenRule::Digit)
        );
        assert_eq!(
            TokenRule::from_str_ignore_case("COMPARE"),
            Some(TokenRule::Compare)
        );
        assert_eq!(
            TokenRule::from_str_ignore_case("Arithmetic"),
            Some(TokenRule::Arithmetic)
        );
        assert_eq!(
            TokenRule::from_str_ignore_case("augment"),
            Some(TokenRule::Augment)
        );
        assert_eq!(TokenRule::from_str_ignore_case("all"), Some(TokenRule::All));
        assert_eq!(TokenRule::from_str_ignore_case("unknown"), None);
    }

    #[test]
    fn test_token_rule_display() {
        let digit = TokenRule::Digit;
        let all = TokenRule::All;
        assert_eq!(format!("{digit}"), "digit");
        assert_eq!(format!("{all}"), "all");
    }

    #[test]
    fn test_token_rule_strategies_order() {
        assert_eq!(TokenRule::STRATEGIES[0], TokenRule::Digit);
        assert_eq!(TokenRule::STRATEGIES[1], TokenRule::Arithmetic);
        assert_eq!(TokenRule::STRATEGIES[2], TokenRule::Compare);
        assert_eq!(TokenRule::STRATEGIES[3], TokenRule::Augment);
        assert_eq!(TokenRule::STRATEGIES[4], TokenRule::All);
    }

    #[test]
    fn test_ppot_config_default_disabled() {
        let config = PpotConfig::default();
        assert!(!config.enabled);
        assert!(!config.adaptive_enabled || !config.enabled);
    }

    #[test]
    fn test_ppot_config_clamp_threshold() {
        let config = PpotConfig::default();
        assert_eq!(config.clamp_threshold(0.2), 0.3); // min
        assert_eq!(config.clamp_threshold(1.5), 1.0); // max
        assert_eq!(config.clamp_threshold(0.7), 0.7); // in range
    }

    #[test]
    fn test_token_rule_index() {
        assert_eq!(TokenRule::Digit.index(), 0);
        assert_eq!(TokenRule::Compare.index(), 1);
        assert_eq!(TokenRule::Arithmetic.index(), 2);
        assert_eq!(TokenRule::Augment.index(), 3);
        assert_eq!(TokenRule::All.index(), 4);
    }

    #[test]
    fn test_rule_count() {
        assert_eq!(TokenRule::rule_count(), 4);
    }
}
