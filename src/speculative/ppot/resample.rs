//! PPoT resample: CPU resampling core.
//!
//! Distilled from "Probabilistic Programs of Thought" (arXiv:2604.17290).
//! After DFlash produces marginals and high-entropy positions are identified,
//! this module resamples variant token sequences using **only CPU** — no GPU
//! forward passes.
//!
//! Three resampling modes:
//! 1. **Basic** — resample from full vocabulary at specified positions
//! 2. **Support-constrained** — resample within a [`TokenRule`]'s support set
//! 3. **Different-value** — conditioned on not reproducing the original token
//!
//! Plan 027 extension: multi-strategy cycling and adaptive rescue with
//! rejection memory from TRT (arXiv:2602.03094).

use crate::speculative::sampling::sample_from_distribution;
use crate::speculative::types::ScreeningPruner;
use crate::types::Rng;

use super::entropy::identify_high_entropy_positions;
use super::types::{PpotConfig, TokenRule};

// Plan 027 imports
use super::knowledge::{RejectionInsight, SessionKnowledge};
use super::rank::rank_by_consistency;

// ── Low-level Sampling Helpers ─────────────────────────────────

/// Sample a token from a probability distribution restricted to a support set.
///
/// Builds a restricted distribution over `support` tokens, normalizes, and
/// samples. Falls back to uniform over support if all probabilities are zero.
///
/// `scratch` must be `>= support.len()`; written to but not meaningful after return.
#[inline]
fn sample_from_support(
    probs: &[f32],
    support: &[usize],
    scratch: &mut [f32],
    rng: &mut Rng,
) -> usize {
    let len = support.len().min(scratch.len());
    if len == 0 {
        return 0;
    }

    // Build restricted distribution
    // i is bounded by len = support.len().min(scratch.len()), so support[i] is safe.
    // Only probs needs a bounds check (support tokens may exceed vocab).
    let mut sum = 0.0f32;
    for (i, slot) in scratch.iter_mut().enumerate().take(len) {
        let tok = support[i];
        // SAFETY: support tokens are generated from TokenRule::support(vocab_size)
        // which guarantees tok < vocab_size == probs.len()
        debug_assert!(
            tok < probs.len(),
            "support token {tok} exceeds vocab size {}",
            probs.len()
        );
        let p = unsafe { *probs.get_unchecked(tok) };
        *slot = p;
        sum += p;
    }

    if sum > 0.0 {
        let inv = 1.0 / sum;
        for val in &mut scratch[..len] {
            *val *= inv;
        }
        let idx = sample_from_distribution(&scratch[..len], rng);
        support[idx.min(len - 1)]
    } else {
        // Degenerate: uniform over support
        support[(rng.next() as usize) % len]
    }
}

/// Sample a token different from `original_token` using the different-value constraint.
///
/// Masks out the original token's probability, renormalizes, and samples.
/// This is the PPoT different-value constraint:
/// `P_sample(x | x ≠ x_orig) = normalize(max(0, P(x) - δ(x, x_orig)))`
///
/// `scratch` must be `>= probs.len()`; written to but not meaningful after return.
#[inline]
fn sample_different_value(
    probs: &[f32],
    original_token: usize,
    scratch: &mut [f32],
    rng: &mut Rng,
) -> usize {
    let len = probs.len().min(scratch.len());
    if len == 0 {
        return 0;
    }

    // Copy and mask original token
    scratch[..len].copy_from_slice(&probs[..len]);
    if original_token < len {
        scratch[original_token] = 0.0;
    }

    // Renormalize
    let sum: f32 = scratch[..len].iter().sum();
    if sum > 0.0 {
        let inv = 1.0 / sum;
        for val in &mut scratch[..len] {
            *val *= inv;
        }
        sample_from_distribution(&scratch[..len], rng)
    } else {
        // All mass was on original — sample uniformly from non-original tokens
        let fallback = (rng.next() as usize) % len;
        if fallback == original_token && len > 1 {
            (fallback + 1) % len
        } else {
            fallback
        }
    }
}

// ── Core Resampling (Plan 026) ──────────────────────────────────

/// Resample tokens at specified positions from marginals.
///
/// For each position in `positions`, draws a new token from the marginal
/// distribution at that position. Tokens at other positions are kept from
/// `base_path`. This is unrestricted resampling (full vocabulary).
///
/// Returns a new token sequence with resampled positions.
pub fn ppot_resample(
    base_path: &[usize],
    marginals: &[&[f32]],
    positions: &[usize],
    rng: &mut Rng,
) -> Vec<usize> {
    let mut result = base_path.to_vec();
    for &pos in positions {
        if pos < marginals.len() && pos < result.len() {
            result[pos] = sample_from_distribution(marginals[pos], rng);
        }
    }
    result
}

/// Resample tokens at specified positions, constrained to a support set.
///
/// Only tokens in `support` are considered at resampled positions.
/// Probabilities are renormalized within the support set.
/// For [`TokenRule::All`], falls back to unrestricted resampling.
///
/// `scratch` must be `>= max(support.len())`; reused across positions.
pub fn ppot_resample_with_support(
    base_path: &[usize],
    marginals: &[&[f32]],
    positions: &[usize],
    support: &[usize],
    scratch: &mut [f32],
    rng: &mut Rng,
) -> Vec<usize> {
    let mut result = base_path.to_vec();
    for &pos in positions {
        if pos < marginals.len() && pos < result.len() {
            result[pos] = sample_from_support(marginals[pos], support, scratch, rng);
        }
    }
    result
}

/// Resample tokens at specified positions, conditioned on producing different values.
///
/// At each resampled position, the original token's probability is masked to zero
/// and the distribution is renormalized. This guarantees at least one position
/// differs from the base path (the PPoT different-value constraint).
///
/// `scratch` must be `>= vocab_size`; reused across positions.
pub fn ppot_resample_different_value(
    base_path: &[usize],
    marginals: &[&[f32]],
    positions: &[usize],
    scratch: &mut [f32],
    rng: &mut Rng,
) -> Vec<usize> {
    let mut result = base_path.to_vec();
    for &pos in positions {
        if pos < marginals.len() && pos < result.len() {
            result[pos] = sample_different_value(marginals[pos], base_path[pos], scratch, rng);
        }
    }
    result
}

// ── Multi-Strategy Resampling (Plan 027) ────────────────────────

/// Generate multiple variant paths, each using a different [`TokenRule`] strategy.
///
/// Cycles through [`TokenRule::STRATEGIES`] for `count` samples:
/// - Sample 0: `Digit` (try different constants)
/// - Sample 1: `Arithmetic` (try different operators)
/// - Sample 2: `Compare` (try different comparisons)
/// - Sample 3: `Augment` (try different assignments)
/// - Sample 4: `All` (unrestricted)
/// - Sample 5+: repeat cycle
///
/// If `preferred` is non-empty, those rules are tried first before cycling.
/// Each variant uses the different-value constraint to avoid reproducing the base path.
///
/// `config` provides pre-cached support sets via [`PpotConfig::with_cached_support`].
/// `scratch` must be `>= vocab_size`.
#[allow(clippy::too_many_arguments)]
pub fn ppot_resample_multi_strategy(
    base_path: &[usize],
    marginals: &[&[f32]],
    positions: &[usize],
    count: usize,
    preferred: &[TokenRule],
    config: &PpotConfig,
    scratch: &mut [f32],
    rng: &mut Rng,
) -> Vec<Vec<usize>> {
    // Pre-allocate all variants as copies of base_path (single bulk allocation).
    // Resampling writes directly into each variant — no per-iteration to_vec().
    let mut variants = Vec::with_capacity(count);
    for _ in 0..count {
        variants.push(base_path.to_vec());
    }

    // Build strategy sequence: preferred first, then cycle through STRATEGIES
    let mut strategy_iter = preferred
        .iter()
        .chain(TokenRule::STRATEGIES.iter().cycle())
        .take(count);

    for variant in variants.iter_mut() {
        let &rule = strategy_iter.next().unwrap_or(&TokenRule::All);

        if matches!(rule, TokenRule::All) {
            // Unrestricted: different-value constraint, resample in-place
            for &pos in positions {
                if pos < marginals.len() && pos < variant.len() {
                    variant[pos] =
                        sample_different_value(marginals[pos], base_path[pos], scratch, rng);
                }
            }
        } else {
            // Constrained: resample within rule's cached support, different value
            let support = config.support_for(rule);
            for &pos in positions {
                if pos < marginals.len() && pos < variant.len() {
                    let probs = marginals[pos];
                    let original = base_path[pos];
                    // Reuse outer scratch (>= vocab_size >= support.len())
                    let candidate = sample_from_support(probs, support, scratch, rng);
                    // If same as original, try once more (simple retry)
                    variant[pos] = if candidate == original && support.len() > 1 {
                        sample_from_support(probs, support, scratch, rng)
                    } else {
                        candidate
                    };
                }
            }
        }
    }

    variants
}

// ── Path Validation ────────────────────────────────────────────

/// Check if a token path passes the screening pruner.
///
/// Returns `true` if every token in the path has positive relevance
/// (no hard rejection). Uses `ScreeningPruner::relevance()` at each depth.
#[inline]
fn is_path_valid<P: ScreeningPruner>(path: &[usize], pruner: &P) -> bool {
    for (depth, &token) in path.iter().enumerate() {
        let relevance = pruner.relevance(depth, token, &path[..depth]);
        if relevance <= 0.0 {
            return false;
        }
    }
    true
}

// ── Rescue Entry Point (Plan 026 Baseline) ──────────────────────

/// PPoT rescue: attempt to find a valid path after DDTree rejection.
///
/// Pipeline:
/// 1. Identify high-entropy positions from marginals
/// 2. Resample `config.num_samples` variants with different-value constraint
/// 3. Screen each through the pruner
/// 4. Return first valid variant
///
/// Returns `None` if no valid variant is found (caller should fall back to greedy).
///
/// This is the Plan 026 baseline — random resampling without adaptation.
/// Zero overhead on the success path (only called when DDTree rejects all paths).
///
/// # Arguments
///
/// * `marginals` — probability distributions from DFlash, one per position
/// * `base_path` — originally sampled tokens (resampling base)
/// * `pruner` — screening pruner for validation
/// * `config` — PPoT configuration (threshold, num_samples, rule, etc.)
/// * `scratch` — temporary buffer, must be `>= vocab_size`
/// * `rng` — deterministic random number generator
pub fn ppot_rescue<P: ScreeningPruner>(
    marginals: &[&[f32]],
    base_path: &[usize],
    pruner: &P,
    config: &PpotConfig,
    scratch: &mut [f32],
    rng: &mut Rng,
) -> Option<Vec<usize>> {
    if marginals.is_empty() || base_path.is_empty() {
        return None;
    }

    // 1. Identify high-entropy positions
    let positions = identify_high_entropy_positions(marginals, config.entropy_threshold);
    if positions.is_empty() {
        return None;
    }

    // 2. Resample m variants
    // When rule is not All, use cached support for constrained resampling.
    // Requires `config.with_cached_support(vocab_size)` to have been called.
    let use_cached_support = config.has_cached_support() && !matches!(config.rule, TokenRule::All);

    for _ in 0..config.num_samples {
        let variant = if config.different_constraint {
            ppot_resample_different_value(base_path, marginals, &positions, scratch, rng)
        } else if use_cached_support {
            let support = config.support_for(config.rule);
            ppot_resample_with_support(base_path, marginals, &positions, support, scratch, rng)
        } else {
            ppot_resample(base_path, marginals, &positions, rng)
        };

        // 3. Screen through pruner
        if is_path_valid(&variant, pruner) {
            return Some(variant);
        }
    }

    // 4. All samples rejected
    None
}

// ── Adaptive Rescue (Plan 027) ──────────────────────────────────

/// Adaptive PPoT rescue with rejection memory (TRT-inspired).
///
/// Extends baseline rescue with:
/// 1. **Adaptive threshold** — lowers after failure, raises after success
/// 2. **Knowledge-biased positions** — prioritizes historically successful positions
/// 3. **Strategy cycling** — each sample uses a different `TokenRule`
/// 4. **Self-consistency ranking** — if multiple valid variants, pick best agreement
/// 5. **Insight recording** — every attempt feeds back into knowledge
///
/// Falls back to baseline behavior on cold start (no knowledge).
///
/// # Arguments
///
/// * `knowledge` — session-level rejection memory (persists across rescue attempts)
///
/// Other arguments same as [`ppot_rescue`].
pub fn ppot_rescue_adaptive<P: ScreeningPruner>(
    marginals: &[&[f32]],
    base_path: &[usize],
    pruner: &P,
    config: &PpotConfig,
    knowledge: &mut SessionKnowledge,
    scratch: &mut [f32],
    rng: &mut Rng,
) -> Option<Vec<usize>> {
    if marginals.is_empty() || base_path.is_empty() {
        return None;
    }

    // 1. Adaptive threshold
    let threshold = if config.adaptive_threshold {
        knowledge.adaptive_threshold(config)
    } else {
        config.entropy_threshold
    };

    // 2. Identify positions (adaptive or entropy-only)
    let positions = if knowledge.has_insights() {
        super::entropy::identify_positions_adaptive(marginals, threshold, Some(knowledge))
    } else {
        identify_high_entropy_positions(marginals, threshold)
    };

    if positions.is_empty() {
        return None;
    }

    // 3. Get preferred rules from knowledge (if any)
    let preferred: Vec<TokenRule> = positions
        .first()
        .map(|&pos| {
            knowledge
                .preferred_rules(pos)
                .into_iter()
                .flatten()
                .collect()
        })
        .unwrap_or_default();

    // 4. Generate multi-strategy variants
    let variants = ppot_resample_multi_strategy(
        base_path,
        marginals,
        &positions,
        config.num_samples,
        &preferred,
        config,
        scratch,
        rng,
    );

    // 5. Validate and collect results
    let mut valid_variants = Vec::new();
    let mut valid_indices = Vec::new();

    // Pre-compute entropy per position (once, not per-sample-per-position)
    let entropy_cache: Vec<f32> = positions
        .iter()
        .map(|&pos| super::entropy::token_entropy(marginals.get(pos).copied().unwrap_or(&[])))
        .collect();

    for (idx, variant) in variants.iter().enumerate() {
        let accepted = is_path_valid(variant, pruner);

        let rule = TokenRule::STRATEGIES
            .get(idx % TokenRule::STRATEGIES.len())
            .copied()
            .unwrap_or(TokenRule::All);

        for (pos_idx, &pos) in positions.iter().enumerate() {
            if pos < variant.len() {
                knowledge.record(RejectionInsight {
                    position: pos,
                    rule,
                    original_token: base_path.get(pos).copied().unwrap_or(0),
                    attempted_token: variant[pos],
                    error_kind: None,
                    entropy: entropy_cache[pos_idx],
                    accepted,
                });
            }
        }

        if accepted {
            valid_variants.push(variant.clone());
            valid_indices.push(idx);
        }
    }

    // 6. Select best variant
    match valid_variants.len() {
        0 => None,
        1 => Some(valid_variants.into_iter().next().unwrap()),
        _ => {
            // Multiple valid: rank by self-consistency
            let ranked = rank_by_consistency(&valid_variants);
            ranked
                .into_iter()
                .next()
                .map(|(idx, _)| valid_variants[idx].clone())
        }
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::speculative::types::NoScreeningPruner;

    // ── sample_from_support tests ──

    #[test]
    fn test_sample_from_support_basic() {
        let mut rng = Rng::new(42);
        let probs = &[0.0, 0.0, 0.5, 0.5]; // only tokens 2,3 have mass
        let support = &[1, 2, 3]; // restrict to tokens 1,2,3
        let mut scratch = vec![0.0f32; 4];

        // Should only pick token 2 or 3 (only ones with mass in support)
        for _ in 0..50 {
            let tok = sample_from_support(probs, support, &mut scratch, &mut rng);
            assert!(tok == 2 || tok == 3, "should pick token 2 or 3, got {tok}");
        }
    }

    #[test]
    fn test_sample_from_support_empty() {
        let mut rng = Rng::new(42);
        let probs = &[0.5, 0.5];
        let support: &[usize] = &[];
        let mut scratch = vec![0.0f32; 4];

        let tok = sample_from_support(probs, support, &mut scratch, &mut rng);
        assert_eq!(tok, 0, "empty support should return 0");
    }

    #[test]
    fn test_sample_from_support_degenerate() {
        let mut rng = Rng::new(42);
        let probs = &[0.0, 0.0, 0.0, 0.0]; // all zero
        let support = &[0, 1, 2];
        let mut scratch = vec![0.0f32; 4];

        // Should fall back to uniform over support
        let tok = sample_from_support(probs, support, &mut scratch, &mut rng);
        assert!(tok < 3, "degenerate should pick from support, got {tok}");
    }

    // ── sample_different_value tests ──

    #[test]
    fn test_sample_different_value_basic() {
        let mut rng = Rng::new(42);
        let probs = &[0.25, 0.25, 0.25, 0.25];
        let mut scratch = vec![0.0f32; 4];

        // Should never pick token 1 (original)
        for _ in 0..50 {
            let tok = sample_different_value(probs, 1, &mut scratch, &mut rng);
            assert_ne!(tok, 1, "should not pick original token");
            assert!(tok < 4, "token should be valid");
        }
    }

    #[test]
    fn test_sample_different_value_all_mass_on_original() {
        let mut rng = Rng::new(42);
        let probs = &[0.0, 1.0, 0.0, 0.0]; // all mass on token 1
        let mut scratch = vec![0.0f32; 4];

        // Should fall back to non-original token
        let tok = sample_different_value(probs, 1, &mut scratch, &mut rng);
        assert_ne!(tok, 1, "should not pick original even with all mass");
    }

    #[test]
    fn test_sample_different_value_deterministic_distribution() {
        let mut rng = Rng::new(42);
        let probs = &[0.0, 0.0, 1.0, 0.0]; // all on token 2
        let mut scratch = vec![0.0f32; 4];

        // Masking token 2 → all zero → fallback
        let tok = sample_different_value(probs, 2, &mut scratch, &mut rng);
        assert_ne!(tok, 2, "should not pick original");
    }

    // ── ppot_resample tests ──

    #[test]
    fn test_ppot_resample_basic() {
        let mut rng = Rng::new(42);
        let base_path = vec![0, 1, 2, 3];
        let marginals: Vec<&[f32]> = vec![
            &[1.0, 0.0, 0.0, 0.0],     // deterministic at pos 0
            &[0.0, 0.0, 0.0, 1.0],     // deterministic at pos 1
            &[0.25, 0.25, 0.25, 0.25], // uniform at pos 2
            &[0.25, 0.25, 0.25, 0.25], // uniform at pos 3
        ];

        // Only resample position 2
        let result = ppot_resample(&base_path, &marginals, &[2], &mut rng);
        assert_eq!(result[0], 0, "position 0 should be unchanged");
        assert_eq!(result[1], 1, "position 1 should be unchanged");
        // position 2 is resampled (could be anything)
        assert_eq!(result[3], 3, "position 3 should be unchanged");
    }

    #[test]
    fn test_ppot_resample_out_of_range_position() {
        let mut rng = Rng::new(42);
        let base_path = vec![0, 1];
        let marginals: Vec<&[f32]> = vec![&[1.0, 0.0], &[0.0, 1.0]];

        // Position 5 is out of range — should be ignored
        let result = ppot_resample(&base_path, &marginals, &[5], &mut rng);
        assert_eq!(result, base_path, "out-of-range position should be ignored");
    }

    #[test]
    fn test_ppot_resample_no_positions() {
        let mut rng = Rng::new(42);
        let base_path = vec![0, 1, 2];
        let marginals: Vec<&[f32]> = vec![&[1.0, 0.0], &[0.0, 1.0], &[1.0, 0.0]];

        let result = ppot_resample(&base_path, &marginals, &[], &mut rng);
        assert_eq!(result, base_path, "no positions should return base path");
    }

    // ── ppot_resample_with_support tests ──

    #[test]
    fn test_ppot_resample_with_support_constrained() {
        let mut rng = Rng::new(42);
        let base_path = vec![5, 5, 5];
        let marginals: Vec<&[f32]> = vec![
            &[0.1, 0.1, 0.1, 0.1, 0.1, 0.5], // mass on token 5
            &[0.1, 0.1, 0.1, 0.1, 0.1, 0.5],
            &[0.1, 0.1, 0.1, 0.1, 0.1, 0.5],
        ];
        let support = &[0, 1, 2]; // restrict to tokens 0-2
        let mut scratch = vec![0.0f32; 6];

        let result = ppot_resample_with_support(
            &base_path,
            &marginals,
            &[0, 1, 2],
            support,
            &mut scratch,
            &mut rng,
        );

        for &tok in &result {
            assert!(tok <= 2, "resampled token should be in support, got {tok}");
        }
    }

    // ── ppot_resample_different_value tests ──

    #[test]
    fn test_ppot_resample_different_value_produces_different() {
        let mut rng = Rng::new(42);
        let base_path = vec![0, 1, 2];
        let marginals: Vec<&[f32]> = vec![
            &[0.25, 0.25, 0.25, 0.25],
            &[0.25, 0.25, 0.25, 0.25],
            &[0.25, 0.25, 0.25, 0.25],
        ];
        let mut scratch = vec![0.0f32; 4];

        // With uniform marginals, should produce different values with high probability
        let mut any_different = false;
        for _ in 0..20 {
            let result = ppot_resample_different_value(
                &base_path,
                &marginals,
                &[0, 1, 2],
                &mut scratch,
                &mut rng,
            );
            if result != base_path {
                any_different = true;
                break;
            }
        }
        assert!(any_different, "should produce at least one different path");
    }

    // ── ppot_rescue tests ──

    /// Pruner that rejects token 0 at any position.
    struct RejectZeroPruner;

    impl ScreeningPruner for RejectZeroPruner {
        fn relevance(&self, _depth: usize, token_idx: usize, _parent_tokens: &[usize]) -> f32 {
            if token_idx == 0 { 0.0 } else { 1.0 }
        }
    }

    #[test]
    fn test_ppot_rescue_finds_valid_path() {
        let mut rng = Rng::new(42);
        // Base path has token 0 at position 0 (rejected by pruner)
        let base_path = vec![0, 1, 2];
        let marginals: Vec<&[f32]> = vec![
            &[0.3, 0.3, 0.2, 0.2],     // position 0: can resample to non-zero
            &[0.1, 0.3, 0.3, 0.3],     // position 1
            &[0.25, 0.25, 0.25, 0.25], // position 2
        ];

        let mut config = PpotConfig::default();
        config.enabled = true;
        config.entropy_threshold = 0.1; // low threshold → all positions are candidates
        config.num_samples = 20;
        config.different_constraint = true;

        let mut scratch = vec![0.0f32; 4];
        let result = ppot_rescue(
            &marginals,
            &base_path,
            &RejectZeroPruner,
            &config,
            &mut scratch,
            &mut rng,
        );

        assert!(result.is_some(), "rescue should find a valid path");
        let path = result.unwrap();
        assert_ne!(path[0], 0, "rescued path should not start with token 0");
    }

    #[test]
    fn test_ppot_rescue_no_high_entropy_positions() {
        let mut rng = Rng::new(42);
        let base_path = vec![0, 1];
        let marginals: Vec<&[f32]> = vec![
            &[0.99, 0.01], // near-deterministic
            &[0.01, 0.99], // near-deterministic
        ];

        let mut config = PpotConfig::default();
        config.enabled = true;
        config.entropy_threshold = 2.0; // very high → no positions qualify

        let mut scratch = vec![0.0f32; 4];
        let result = ppot_rescue(
            &marginals,
            &base_path,
            &NoScreeningPruner,
            &config,
            &mut scratch,
            &mut rng,
        );

        assert!(result.is_none(), "no high-entropy positions → None");
    }

    #[test]
    fn test_ppot_rescue_empty_inputs() {
        let mut rng = Rng::new(42);
        let config = PpotConfig::default();
        let mut scratch = vec![0.0f32; 4];

        assert!(
            ppot_rescue(
                &[],
                &[],
                &NoScreeningPruner,
                &config,
                &mut scratch,
                &mut rng
            )
            .is_none()
        );
        assert!(
            ppot_rescue(
                &[&[0.5, 0.5]],
                &[],
                &NoScreeningPruner,
                &config,
                &mut scratch,
                &mut rng
            )
            .is_none()
        );
    }

    #[test]
    fn test_ppot_rescue_all_rejected() {
        let mut rng = Rng::new(42);
        // All paths will have at least one token, and RejectAllPruner rejects everything
        struct RejectAllPruner;
        impl ScreeningPruner for RejectAllPruner {
            fn relevance(&self, _: usize, _: usize, _: &[usize]) -> f32 {
                0.0
            }
        }

        let base_path = vec![0, 1, 2];
        let marginals: Vec<&[f32]> = vec![
            &[0.25, 0.25, 0.25, 0.25],
            &[0.25, 0.25, 0.25, 0.25],
            &[0.25, 0.25, 0.25, 0.25],
        ];

        let mut config = PpotConfig::default();
        config.enabled = true;
        config.entropy_threshold = 0.1;
        config.num_samples = 5;

        let mut scratch = vec![0.0f32; 4];
        let result = ppot_rescue(
            &marginals,
            &base_path,
            &RejectAllPruner,
            &config,
            &mut scratch,
            &mut rng,
        );

        assert!(result.is_none(), "all-rejecting pruner should return None");
    }

    // ── is_path_valid tests ──

    #[test]
    fn test_is_path_valid_no_pruner() {
        let path = vec![0, 1, 2, 3];
        assert!(is_path_valid(&path, &NoScreeningPruner));
    }

    #[test]
    fn test_is_path_valid_reject_zero() {
        let valid_path = vec![1, 2, 3];
        let invalid_path = vec![0, 1, 2];

        assert!(is_path_valid(&valid_path, &RejectZeroPruner));
        assert!(!is_path_valid(&invalid_path, &RejectZeroPruner));
    }

    // ── ppot_resample_multi_strategy tests ──

    #[test]
    fn test_ppot_resample_multi_strategy_count() {
        let mut rng = Rng::new(42);
        let base_path = vec![0, 1, 2];
        let marginals: Vec<&[f32]> = vec![
            &[0.25, 0.25, 0.25, 0.25],
            &[0.25, 0.25, 0.25, 0.25],
            &[0.25, 0.25, 0.25, 0.25],
        ];
        let mut scratch = vec![0.0f32; 4];
        let config = PpotConfig::default().with_cached_support(4);

        let variants = ppot_resample_multi_strategy(
            &base_path,
            &marginals,
            &[0, 1, 2],
            5,
            &[],
            &config,
            &mut scratch,
            &mut rng,
        );

        assert_eq!(variants.len(), 5, "should produce 5 variants");
        for variant in &variants {
            assert_eq!(variant.len(), 3, "each variant should have 3 tokens");
        }
    }

    #[test]
    fn test_ppot_resample_multi_strategy_with_preferred() {
        let mut rng = Rng::new(42);
        let base_path = vec![0, 1, 2];
        let marginals: Vec<&[f32]> = vec![
            &[0.25, 0.25, 0.25, 0.25],
            &[0.25, 0.25, 0.25, 0.25],
            &[0.25, 0.25, 0.25, 0.25],
        ];
        let mut scratch = vec![0.0f32; 4];
        let config = PpotConfig::default().with_cached_support(4);

        let preferred = vec![TokenRule::Digit, TokenRule::Arithmetic];
        let variants = ppot_resample_multi_strategy(
            &base_path,
            &marginals,
            &[0, 1, 2],
            3,
            &preferred,
            &config,
            &mut scratch,
            &mut rng,
        );

        assert_eq!(variants.len(), 3, "should produce 3 variants");
    }
}
