//! PPoT entropy: Shannon entropy calculation and high-entropy position identification.
//!
//! Distilled from "Probabilistic Programs of Thought" (arXiv:2604.17290).
//! After DFlash produces marginals, this module identifies positions where
//! the model is uncertain (high Shannon entropy) — candidates for CPU resampling.
//!
//! Plan 027 extension: adaptive position selection using `SessionKnowledge`
//! from TRT (arXiv:2602.03094). Historical success/failure patterns bias
//! which positions are explored, avoiding known-bad regions.

use super::types::TokenRule;

#[cfg(feature = "ppot")]
use super::knowledge::SessionKnowledge;

// ── Shannon Entropy ────────────────────────────────────────────

/// Compute Shannon entropy of a probability distribution: `H = -Σ p·ln(p)`.
///
/// Returns 0.0 for deterministic distributions (single peak) and
/// `ln(vocab_size)` for uniform distributions (maximum uncertainty).
///
/// # Examples
///
/// ```ignore
/// let probs = &[0.0, 0.0, 1.0, 0.0];
/// assert_eq!(token_entropy(probs), 0.0); // deterministic
///
/// let uniform = &[0.25, 0.25, 0.25, 0.25];
/// let h = token_entropy(uniform);
/// assert!(h > 1.0); // high entropy
/// ```
#[inline]
pub fn token_entropy(probs: &[f32]) -> f32 {
    let mut entropy = 0.0f32;
    for &p in probs {
        if p > 0.0 {
            entropy -= p * p.ln();
        }
    }
    entropy
}

// ── Position Identification (Plan 026) ─────────────────────────

/// Identify positions where Shannon entropy exceeds threshold.
///
/// Returns indices into `marginals` where `H(i) > threshold`.
/// Each entry in `marginals` is a probability slice for one decoding step.
///
/// This is the core "random variable" identification from PPoT:
/// high-entropy positions are where the model is uncertain, and
/// CPU resampling can explore alternative tokens.
pub fn identify_high_entropy_positions(marginals: &[&[f32]], threshold: f32) -> Vec<usize> {
    let mut positions = Vec::with_capacity(marginals.len());
    identify_high_entropy_positions_into(marginals, threshold, &mut positions);
    positions
}

/// Zero-alloc variant of [`identify_high_entropy_positions`].
///
/// Clears `buf` and writes high-entropy position indices into it.
/// Reuses pre-allocated buffer capacity across calls.
#[inline]
pub fn identify_high_entropy_positions_into(
    marginals: &[&[f32]],
    threshold: f32,
    buf: &mut Vec<usize>,
) {
    buf.clear();
    for (i, &probs) in marginals.iter().enumerate() {
        let h = token_entropy(probs);
        if h > threshold {
            buf.push(i);
        }
    }
}

/// Identify positions filtered by both entropy and token rule support.
///
/// Only positions where:
/// 1. `H(i) > threshold` (model is uncertain), AND
/// 2. At least one token in the rule's support has nonzero probability
///
/// are included. This avoids resampling positions where the rule's
/// domain doesn't apply (e.g., no digit tokens in a comparison position).
pub fn identify_positions_by_rule(
    marginals: &[&[f32]],
    rule: &TokenRule,
    threshold: f32,
) -> Vec<usize> {
    let mut positions = Vec::with_capacity(marginals.len());
    identify_positions_by_rule_into(marginals, rule, threshold, &mut positions);
    positions
}

/// Zero-alloc variant of [`identify_positions_by_rule`].
#[inline]
pub fn identify_positions_by_rule_into(
    marginals: &[&[f32]],
    rule: &TokenRule,
    threshold: f32,
    buf: &mut Vec<usize>,
) {
    buf.clear();
    let vocab_size = marginals.first().map(|m| m.len()).unwrap_or(0);
    if vocab_size == 0 {
        return;
    }

    let support = rule.support(vocab_size);

    for (i, &probs) in marginals.iter().enumerate() {
        let h = token_entropy(probs);
        if h <= threshold {
            continue;
        }

        // Check if any support token has nonzero probability
        let has_support_mass = support
            .iter()
            .any(|&tok| probs.get(tok).copied().unwrap_or(0.0) > 0.0);

        if has_support_mass {
            buf.push(i);
        }
    }
}

// ── Adaptive Position Selection (Plan 027) ─────────────────────

/// Identify positions using adaptive strategy from TRT.
///
/// When `knowledge` is `None` or empty (cold start), falls back to
/// standard entropy-only selection via [`identify_high_entropy_positions`].
///
/// When knowledge is available:
/// 1. Starts from high-entropy positions (same as baseline)
/// 2. Filters out `should_skip` positions (known-dead from history)
/// 3. Reorders by `position_affinity` (historically successful positions first)
///
/// This captures TRT's finding that models switch strategy more after
/// failure (82%) than success (74%) — positions that consistently fail
/// are deprioritized.
#[cfg(feature = "ppot")]
pub fn identify_positions_adaptive(
    marginals: &[&[f32]],
    threshold: f32,
    knowledge: Option<&SessionKnowledge>,
) -> Vec<usize> {
    let mut positions = Vec::with_capacity(marginals.len());
    identify_positions_adaptive_into(marginals, threshold, knowledge, &mut positions);
    positions
}

/// Zero-alloc variant of [`identify_positions_adaptive`].
#[cfg(feature = "ppot")]
#[inline]
pub fn identify_positions_adaptive_into(
    marginals: &[&[f32]],
    threshold: f32,
    knowledge: Option<&SessionKnowledge>,
    buf: &mut Vec<usize>,
) {
    // Start with entropy-based positions
    identify_high_entropy_positions_into(marginals, threshold, buf);

    match knowledge {
        Some(k) if k.has_insights() => {
            // Filter out positions that should be skipped (known-dead)
            buf.retain(|&pos| !k.should_skip_position(pos));

            // Sort by position affinity (highest success rate first)
            // Stable sort preserves entropy order for equal affinity
            buf.sort_by(|&a, &b| {
                let affinity_a = k.position_affinity(a);
                let affinity_b = k.position_affinity(b);
                affinity_b
                    .partial_cmp(&affinity_a)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }
        _ => {
            // Cold start: no knowledge, use entropy-only ordering
        }
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token_entropy_deterministic() {
        // Single peak: entropy should be 0
        let probs = &[0.0, 0.0, 1.0, 0.0];
        let h = token_entropy(probs);
        assert!(
            h < 0.001,
            "deterministic distribution should have ~0 entropy, got {h}"
        );
    }

    #[test]
    fn test_token_entropy_uniform() {
        // Uniform over 4 tokens: H = ln(4) ≈ 1.386
        let probs = &[0.25f32, 0.25, 0.25, 0.25];
        let h = token_entropy(probs);
        let expected = 4.0f32.ln();
        assert!(
            (h - expected).abs() < 0.01,
            "uniform entropy should be ~{expected}, got {h}"
        );
    }

    #[test]
    fn test_token_entropy_uniform_large_vocab() {
        // Uniform over 10 tokens: H = ln(10) ≈ 2.302
        let probs: Vec<f32> = vec![0.1; 10];
        let h = token_entropy(&probs);
        let expected = 10.0f32.ln();
        assert!(
            (h - expected).abs() < 0.01,
            "uniform entropy should be ~{expected}, got {h}"
        );
    }

    #[test]
    fn test_token_entropy_partial() {
        // Mixed distribution: [0.5, 0.25, 0.125, 0.125]
        let probs = &[0.5f32, 0.25, 0.125, 0.125];
        let h = token_entropy(probs);
        // H = -(0.5*ln(0.5) + 0.25*ln(0.25) + 0.125*ln(0.125) + 0.125*ln(0.125))
        let expected = -(0.5f32.ln() * 0.5 + 0.25f32.ln() * 0.25 + 0.125f32.ln() * 0.25);
        assert!(
            (h - expected).abs() < 0.01,
            "partial entropy should be ~{expected}, got {h}"
        );
    }

    #[test]
    fn test_identify_high_entropy_positions_basic() {
        // Position 0: deterministic (H=0)
        // Position 1: uniform (H≈1.386)
        // Position 2: near-deterministic (H≈0)
        // Position 3: uniform (H≈1.386)
        let marginals: Vec<&[f32]> = vec![
            &[0.0, 0.0, 1.0, 0.0],
            &[0.25, 0.25, 0.25, 0.25],
            &[0.97, 0.01, 0.01, 0.01],
            &[0.25, 0.25, 0.25, 0.25],
        ];

        let positions = identify_high_entropy_positions(&marginals, 0.5);
        assert_eq!(
            positions,
            vec![1, 3],
            "should identify high-entropy positions"
        );
    }

    #[test]
    fn test_identify_high_entropy_positions_none_above_threshold() {
        let marginals: Vec<&[f32]> = vec![&[0.9, 0.05, 0.05], &[0.95, 0.025, 0.025]];

        let positions = identify_high_entropy_positions(&marginals, 2.0);
        assert!(
            positions.is_empty(),
            "no positions should exceed high threshold"
        );
    }

    #[test]
    fn test_identify_high_entropy_positions_all_above() {
        let marginals: Vec<&[f32]> = vec![&[0.25, 0.25, 0.25, 0.25], &[0.25, 0.25, 0.25, 0.25]];

        let positions = identify_high_entropy_positions(&marginals, 0.1);
        assert_eq!(
            positions,
            vec![0, 1],
            "all positions should exceed low threshold"
        );
    }

    #[test]
    fn test_identify_positions_into_reuses_buffer() {
        let marginals: Vec<&[f32]> = vec![&[0.25, 0.25, 0.25, 0.25]];

        let mut buf = Vec::new();
        identify_high_entropy_positions_into(&marginals, 0.1, &mut buf);
        assert_eq!(buf, &[0]);

        // Second call should clear and reuse
        let marginals2: Vec<&[f32]> = vec![&[1.0, 0.0, 0.0, 0.0], &[0.25, 0.25, 0.25, 0.25]];
        identify_high_entropy_positions_into(&marginals2, 0.1, &mut buf);
        assert_eq!(buf, &[1]);
    }

    #[test]
    fn test_identify_positions_by_rule_digit() {
        // Position 0: high entropy, digits have mass
        // Position 1: high entropy, no digit mass (all on token 15)
        let marginals: Vec<&[f32]> = vec![
            &[
                0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
            &[
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
            ],
        ];

        let positions = identify_positions_by_rule(&marginals, &TokenRule::Digit, 0.5);
        assert_eq!(positions, vec![0], "only position 0 has digit support mass");
    }

    #[test]
    fn test_identify_positions_by_rule_all() {
        // All rule: should include any high-entropy position
        let marginals: Vec<&[f32]> = vec![&[0.25, 0.25, 0.25, 0.25], &[1.0, 0.0, 0.0, 0.0]];

        let positions = identify_positions_by_rule(&marginals, &TokenRule::All, 0.1);
        assert_eq!(positions, vec![0]);
    }

    #[test]
    fn test_identify_positions_by_rule_no_support_mass() {
        // High entropy but no mass on digits (all on token 20)
        let marginals: Vec<&[f32]> = vec![&[
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ]];

        let positions = identify_positions_by_rule(&marginals, &TokenRule::Digit, 0.1);
        assert!(positions.is_empty(), "no digit support mass");
    }

    #[test]
    fn test_token_entropy_empty_slice() {
        let probs: &[f32] = &[];
        let h = token_entropy(probs);
        assert_eq!(h, 0.0, "empty slice should have zero entropy");
    }

    #[test]
    fn test_entropy_threshold_boundary() {
        // Exactly at threshold should NOT be included (> not >=)
        let probs = &[0.5f32, 0.5]; // H = ln(2) ≈ 0.693
        let h = token_entropy(probs);
        let marginals: Vec<&[f32]> = vec![probs];

        let positions = identify_high_entropy_positions(&marginals, h);
        assert!(
            positions.is_empty(),
            "entropy exactly at threshold should not be included"
        );

        let positions = identify_high_entropy_positions(&marginals, h - 0.01);
        assert_eq!(
            positions,
            vec![0],
            "entropy just above threshold should be included"
        );
    }
}
