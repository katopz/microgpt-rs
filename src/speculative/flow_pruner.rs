//! FlowPruner — GFlowNet-inspired stop-probability regularization.
//!
//! The GFlowNet paper proves that minimizing expected trajectory length
//! forces the policy to concentrate on shortest paths. The flow regularization
//! is `λ * exp(logsumexp(-log_pf_stop))`.
//!
//! In our model, `P_F(s_f | s)` is the LoRA model's probability of the EOS token
//! at depth d. The flow `F(s) = 1/P_stop(s)` — high when the model thinks the
//! solution continues. Low P_stop = high flow = boost exploration there.
//!
//! FlowPruner is a wrapper — it composes with any ScreeningPruner (NoScreeningPruner,
//! BanditPruner, AbsorbCompress, etc.) and adds a multiplicative flow bonus.
//!
//! # Feature Gate
//!
//! All code behind `#[cfg(feature = "bandit")]`.

use super::types::ScreeningPruner;

// ── FlowPruner ──────────────────────────────────────────────────

/// ScreeningPruner wrapper that adds GFlowNet flow-based exploration bonus.
///
/// High stop_prob = model wants to stop = low flow = no bonus needed.
/// Low stop_prob = model wants to continue = high flow = boost exploration.
///
/// # Formula
///
/// ```text
/// relevance_flow(depth, token, path) = inner_relevance * (1 + λ × (1 - stop_prob(depth)))
/// ```
///
/// Where:
/// - `inner_relevance` = the wrapped pruner's relevance score
/// - `λ` = flow regularization strength (paper: reg_coef)
/// - `stop_prob(depth)` = P(EOS | depth) from marginals
///
/// # Design
///
/// - Zero-alloc: just a multiplication on top of inner relevance
/// - Additive: wraps any ScreeningPruner without modifying it
/// - Revertible: set `lambda = 0.0` to disable
pub struct FlowPruner<P: ScreeningPruner> {
    /// Inner pruner to wrap.
    inner: P,
    /// Flow regularization strength (paper: reg_coef).
    /// Default: 0.3. Set to 0.0 to disable flow bonus.
    lambda: f32,
    /// Per-depth EOS probability from marginals.
    /// `stop_probs[depth]` = probability model assigns to stopping at that depth.
    stop_probs: Vec<f32>,
}

impl<P: ScreeningPruner> FlowPruner<P> {
    /// Create a new FlowPruner wrapping an inner pruner.
    ///
    /// # Arguments
    ///
    /// * `inner` — The ScreeningPruner to wrap
    /// * `lambda` — Flow regularization strength (default: 0.3)
    /// * `stop_probs` — Per-depth EOS probability from marginals.
    ///   Extract from `marginals[depth][eos_token_idx]`.
    ///   If no EOS token in vocab, use entropy as proxy:
    ///   `stop_probs[depth] = 1.0 - entropy(marginals[depth]) / max_entropy`
    pub fn new(inner: P, lambda: f32, stop_probs: Vec<f32>) -> Self {
        Self {
            inner,
            lambda,
            stop_probs,
        }
    }

    /// Create with default lambda (0.3) and empty stop probs.
    /// Call `set_stop_probs()` before use.
    pub fn with_inner(inner: P) -> Self {
        Self {
            inner,
            lambda: 0.3,
            stop_probs: Vec::new(),
        }
    }

    /// Set flow regularization strength.
    pub fn with_lambda(mut self, lambda: f32) -> Self {
        self.lambda = lambda;
        self
    }

    /// Update stop probabilities from marginals.
    ///
    /// Call this before each DDTree build to refresh stop probs.
    /// For each depth, extract the EOS token probability or use entropy proxy.
    pub fn set_stop_probs(&mut self, stop_probs: Vec<f32>) {
        self.stop_probs = stop_probs;
    }

    /// Update stop probabilities from marginals using EOS token index.
    ///
    /// For each depth's marginal distribution, extracts the EOS token's probability.
    /// If `eos_token_idx >= marginal.len()`, uses entropy as proxy.
    pub fn set_stop_probs_from_marginals(&mut self, marginals: &[&[f32]], eos_token_idx: usize) {
        self.stop_probs = marginals
            .iter()
            .map(|marginal| {
                if eos_token_idx < marginal.len() {
                    // Direct: use EOS token probability
                    marginal[eos_token_idx]
                } else {
                    // Proxy: use max probability as "confidence of stopping"
                    // High max prob = model is sure about something = likely to stop
                    marginal.iter().copied().fold(0.0f32, f32::max)
                }
            })
            .collect();
    }

    /// Update stop probabilities using entropy as proxy.
    ///
    /// High entropy = model unsure = should continue = low stop_prob.
    /// Low entropy = model confident = should stop = high stop_prob.
    pub fn set_stop_probs_from_entropy(&mut self, marginals: &[&[f32]]) {
        self.stop_probs = marginals
            .iter()
            .map(|marginal| {
                let sum: f32 = marginal.iter().copied().sum();
                if sum <= 0.0 {
                    return 0.5; // Unknown
                }
                // Compute entropy
                let entropy: f32 = marginal
                    .iter()
                    .filter(|&&p| p > 0.0)
                    .map(|&p| {
                        let pn = p / sum;
                        -pn * pn.ln()
                    })
                    .sum();
                // Max entropy for uniform distribution over vocab
                let n = marginal.len().max(1) as f32;
                let max_entropy = n.ln();
                // Normalize: low entropy = high stop_prob
                if max_entropy > 0.0 {
                    1.0 - (entropy / max_entropy).min(1.0)
                } else {
                    0.5
                }
            })
            .collect();
    }

    /// Get the stop probability at a given depth.
    /// Returns 0.5 (neutral) if depth is out of range.
    #[inline]
    pub fn stop_prob(&self, depth: usize) -> f32 {
        self.stop_probs.get(depth).copied().unwrap_or(0.5)
    }

    /// Get the flow bonus at a given depth.
    /// `flow_bonus = 1 + λ × (1 - stop_prob(depth))`
    #[inline]
    pub fn flow_bonus(&self, depth: usize) -> f32 {
        1.0 + self.lambda * (1.0 - self.stop_prob(depth))
    }

    /// Access the inner pruner.
    pub fn inner(&self) -> &P {
        &self.inner
    }

    /// Mutable access to the inner pruner.
    pub fn inner_mut(&mut self) -> &mut P {
        &mut self.inner
    }

    /// Get lambda value.
    pub fn lambda(&self) -> f32 {
        self.lambda
    }

    /// Get stop probabilities slice.
    pub fn stop_probs(&self) -> &[f32] {
        &self.stop_probs
    }
}

impl<P: ScreeningPruner> ScreeningPruner for FlowPruner<P> {
    #[inline]
    fn relevance(&self, depth: usize, token_idx: usize, parent_tokens: &[usize]) -> f32 {
        let inner = self.inner.relevance(depth, token_idx, parent_tokens);
        if inner <= 0.0 {
            return 0.0;
        }

        // Flow bonus: F(s) = 1/P_stop(s)
        // High stop_prob = model wants to stop = low flow = no bonus
        // Low stop_prob = model wants to continue = high flow = boost exploration
        let flow_bonus = self.flow_bonus(depth);
        (inner * flow_bonus).clamp(0.0, 1.0)
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::speculative::types::NoScreeningPruner;

    #[test]
    fn test_flow_pruner_no_stop_probs_passes_through() {
        let pruner = FlowPruner::new(NoScreeningPruner, 0.3, vec![]);
        // No stop probs → default 0.5 → bonus = 1 + 0.3 * 0.5 = 1.15
        let rel = pruner.relevance(0, 0, &[]);
        assert!((rel - 1.0).abs() < 1e-6, "Clamped to 1.0: {rel}");
    }

    #[test]
    fn test_flow_pruner_low_stop_prob_boosts() {
        // Low stop prob (0.1) → high flow → boost
        let pruner = FlowPruner::new(NoScreeningPruner, 0.3, vec![0.1]);
        // bonus = 1 + 0.3 * (1 - 0.1) = 1 + 0.27 = 1.27
        // inner = 1.0, result = 1.27 → clamped to 1.0
        let rel = pruner.relevance(0, 0, &[]);
        assert!((rel - 1.0).abs() < 1e-6, "Clamped to 1.0: {rel}");
    }

    #[test]
    fn test_flow_pruner_high_stop_prob_no_boost() {
        // High stop prob (0.9) → low flow → minimal bonus
        let pruner = FlowPruner::new(NoScreeningPruner, 0.3, vec![0.9]);
        // bonus = 1 + 0.3 * (1 - 0.9) = 1 + 0.03 = 1.03
        // inner = 1.0, result = 1.03 → clamped to 1.0
        let rel = pruner.relevance(0, 0, &[]);
        assert!((rel - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_flow_pruner_with_half_relevance() {
        struct HalfPruner;
        impl ScreeningPruner for HalfPruner {
            fn relevance(&self, _depth: usize, _token_idx: usize, _parent_tokens: &[usize]) -> f32 {
                0.5
            }
        }

        // Low stop prob → boost
        let pruner = FlowPruner::new(HalfPruner, 0.3, vec![0.1]);
        // bonus = 1 + 0.3 * 0.9 = 1.27
        // inner = 0.5, result = 0.5 * 1.27 = 0.635
        let rel = pruner.relevance(0, 0, &[]);
        assert!((rel - 0.635).abs() < 1e-5, "Expected 0.635, got {rel}");

        // High stop prob → minimal boost
        let pruner = FlowPruner::new(HalfPruner, 0.3, vec![0.9]);
        // bonus = 1 + 0.3 * 0.1 = 1.03
        // inner = 0.5, result = 0.5 * 1.03 = 0.515
        let rel = pruner.relevance(0, 0, &[]);
        assert!((rel - 0.515).abs() < 1e-5, "Expected 0.515, got {rel}");
    }

    #[test]
    fn test_flow_pruner_zero_inner_is_zero() {
        struct ZeroPruner;
        impl ScreeningPruner for ZeroPruner {
            fn relevance(&self, _depth: usize, _token_idx: usize, _parent_tokens: &[usize]) -> f32 {
                0.0
            }
        }

        let pruner = FlowPruner::new(ZeroPruner, 0.3, vec![0.1]);
        let rel = pruner.relevance(0, 0, &[]);
        assert!((rel - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_flow_pruner_lambda_zero_no_bonus() {
        let pruner = FlowPruner::new(NoScreeningPruner, 0.0, vec![0.1]);
        let rel = pruner.relevance(0, 0, &[]);
        assert!((rel - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_set_stop_probs_from_marginals() {
        let mut pruner = FlowPruner::with_inner(NoScreeningPruner);

        let m1: Vec<f32> = vec![0.1, 0.7, 0.1, 0.1]; // EOS at index 1 = 0.7
        let m2: Vec<f32> = vec![0.3, 0.3, 0.3, 0.1]; // EOS at index 1 = 0.3

        pruner.set_stop_probs_from_marginals(&[&m1, &m2], 1);
        assert!((pruner.stop_prob(0) - 0.7).abs() < 1e-6);
        assert!((pruner.stop_prob(1) - 0.3).abs() < 1e-6);
    }

    #[test]
    fn test_set_stop_probs_from_entropy() {
        let mut pruner = FlowPruner::with_inner(NoScreeningPruner);

        // Concentrated distribution → low entropy → high stop_prob
        let concentrated: Vec<f32> = vec![0.9, 0.05, 0.03, 0.02];
        // Uniform distribution → high entropy → low stop_prob
        let uniform: Vec<f32> = vec![0.25, 0.25, 0.25, 0.25];

        pruner.set_stop_probs_from_entropy(&[&concentrated, &uniform]);

        let sp_concentrated = pruner.stop_prob(0);
        let sp_uniform = pruner.stop_prob(1);

        assert!(
            sp_concentrated > sp_uniform,
            "Concentrated should have higher stop_prob: {sp_concentrated} vs {sp_uniform}"
        );
    }

    #[test]
    fn test_out_of_range_depth_defaults_neutral() {
        let pruner = FlowPruner::new(NoScreeningPruner, 0.3, vec![0.2]);
        // depth 5 is out of range → default 0.5
        let bonus = pruner.flow_bonus(5);
        let expected = 1.0 + 0.3 * (1.0 - 0.5);
        assert!((bonus - expected).abs() < 1e-6);
    }

    #[test]
    fn test_with_lambda_builder() {
        let pruner = FlowPruner::with_inner(NoScreeningPruner).with_lambda(0.5);
        assert!((pruner.lambda() - 0.5).abs() < 1e-6);
    }
}
