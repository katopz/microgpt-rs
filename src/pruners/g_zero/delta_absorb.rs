//! Absorb-compress cycle gated by Hint-δ — promote heuristics only for blind spots.
//!
//! Replaces the raw-reward absorb gate with a δ-derived signal:
//! - Current: absorbs based on raw reward (did the environment say "good"?)
//! - New: absorbs based on δ (did the hint reveal a blind spot?)
//! - Blind spots = high-δ = the model doesn't already know this → promote to constraint
//!
//! # Why δ-gating is smarter
//!
//! Raw reward is sparse and delayed (episode completion). δ is dense and immediate
//! (every token scored). High δ means the hint carried information the Solver lacks —
//! exactly the arms worth promoting to hard constraints.
//!
//! # Usage
//!
//! ```rust,ignore
//! let layer = DeltaGatedAbsorbCompress::new(
//!     AbsorbCompressLayer::new(NoScreeningPruner, 5, CompressConfig::default()),
//!     5,
//!     DeltaGatedConfig::default(),
//! );
//!
//! // Feed δ observation — only absorbs if δ >= threshold
//! layer.observe_delta(0, 0.15);  // absorbed
//! layer.observe_delta(1, 0.01);  // NOT absorbed (below threshold)
//! ```

use std::cmp::Ordering;

use crate::pruners::absorb_compress::{AbsorbCompress, AbsorbCompressLayer};
use crate::pruners::review_metrics::ReviewMetrics;
use crate::speculative::types::ScreeningPruner;

use super::types::HintDelta;

// ── Config ──────────────────────────────────────────────────────

/// Tunable thresholds for δ-gated absorb-compress.
///
/// An arm is absorbed (eligible for promotion) when:
/// 1. Its accumulated δ exceeds `delta_threshold`
/// 2. The inner `AbsorbCompressLayer` visit/Q-value criteria are met
///
/// Default δ threshold: 0.02 (minimum δ to consider a hint "meaningful").
/// The G-Zero paper retains the lower half of the δ distribution,
/// but for gating individual arms we use a fixed threshold.
#[derive(Clone, Debug)]
pub struct DeltaGatedConfig {
    /// Minimum δ to absorb an arm (default: 0.02).
    ///
    /// Arms with δ below this are ignored — the hint didn't reveal a blind spot.
    pub delta_threshold: f32,
    /// Minimum benefit-to-risk ratio for compression (Plan 036 compatibility).
    ///
    /// Set to 0.0 to disable review metrics gating.
    pub min_benefit_ratio: f64,
}

impl Default for DeltaGatedConfig {
    fn default() -> Self {
        Self {
            delta_threshold: 0.02,
            min_benefit_ratio: 2.0,
        }
    }
}

impl DeltaGatedConfig {
    /// Create config with custom δ threshold.
    pub fn new(delta_threshold: f32) -> Self {
        Self {
            delta_threshold,
            ..Self::default()
        }
    }

    /// Create config with custom benefit-ratio threshold.
    pub fn with_benefit_ratio(mut self, min_benefit_ratio: f64) -> Self {
        self.min_benefit_ratio = min_benefit_ratio;
        self
    }
}

// ── DeltaGatedAbsorbCompress ────────────────────────────────────

/// Absorb-compress layer gated by Hint-δ instead of raw reward.
///
/// Wraps [`AbsorbCompressLayer`] and only promotes arms where the hint
/// made a meaningful difference (δ above threshold). This targets the
/// model's blind spots rather than just low-reward arms.
///
/// # Architecture
///
/// ```text
/// DeltaGatedAbsorbCompress<P>
///   ├── inner: AbsorbCompressLayer<P>  (existing absorb-compress logic)
///   ├── delta_history: Vec<f32>         (per-arm accumulated δ)
///   └── delta_threshold: f32            (minimum δ to absorb)
/// ```
///
/// δ observations are fed via [`observe_delta`](Self::observe_delta).
/// Regular reward observations go through [`AbsorbCompress::absorb`]
/// and are gated by the arm's accumulated δ.
pub struct DeltaGatedAbsorbCompress<P: ScreeningPruner> {
    /// Inner absorb-compress layer (delegates actual promotion logic).
    inner: AbsorbCompressLayer<P>,
    /// Per-arm accumulated δ values.
    delta_history: Vec<f32>,
    /// Per-arm δ observation count (for mean δ tracking).
    delta_counts: Vec<usize>,
    /// Configuration thresholds.
    config: DeltaGatedConfig,
}

impl<P: ScreeningPruner> DeltaGatedAbsorbCompress<P> {
    /// Create a new δ-gated absorb-compress layer.
    ///
    /// Wraps an existing `AbsorbCompressLayer` with δ-based gating.
    pub fn new(inner: AbsorbCompressLayer<P>, num_arms: usize, config: DeltaGatedConfig) -> Self {
        Self {
            inner,
            delta_history: vec![0.0; num_arms],
            delta_counts: vec![0; num_arms],
            config,
        }
    }

    /// Feed a δ observation for an arm.
    ///
    /// Accumulates δ per arm. If the arm's mean δ exceeds the threshold,
    /// the reward is forwarded to the inner absorb-compress layer.
    ///
    /// Only positive δ is absorbed — negative δ means the hint hurt,
    /// which doesn't indicate a blind spot worth promoting.
    pub fn observe_delta(&mut self, arm: usize, delta: f32, reward: f32) {
        if arm >= self.delta_history.len() {
            return;
        }

        // Accumulate δ for this arm
        self.delta_history[arm] += delta.max(0.0);
        self.delta_counts[arm] += 1;

        // Only absorb if mean δ exceeds threshold
        let mean_delta = self.delta_history[arm] / self.delta_counts[arm] as f32;
        if mean_delta >= self.config.delta_threshold {
            self.inner.absorb(arm, reward);
        }
    }

    /// Feed a [`HintDelta`] directly — convenience wrapper for [`observe_delta`](Self::observe_delta).
    ///
    /// Uses `delta.value` as the δ signal and `delta.value.max(0.0)` as the reward.
    pub fn observe_hint_delta(&mut self, arm: usize, delta: &HintDelta) {
        let reward = delta.value.max(0.0);
        self.observe_delta(arm, delta.value, reward);
    }

    /// Mean δ for a specific arm.
    ///
    /// Returns 0.0 for unobserved arms.
    pub fn mean_delta(&self, arm: usize) -> f32 {
        if arm >= self.delta_counts.len() || self.delta_counts[arm] == 0 {
            return 0.0;
        }
        self.delta_history[arm] / self.delta_counts[arm] as f32
    }

    /// Accumulated δ for a specific arm.
    pub fn total_delta(&self, arm: usize) -> f32 {
        self.delta_history.get(arm).copied().unwrap_or(0.0)
    }

    /// Number of δ observations for a specific arm.
    pub fn delta_observation_count(&self, arm: usize) -> usize {
        self.delta_counts.get(arm).copied().unwrap_or(0)
    }

    /// Which arms have the highest accumulated δ (top-K blind spots)?
    ///
    /// Useful for targeting [`super::template_proposer::TemplateProposer`]
    /// toward the model's weakest areas.
    pub fn blind_spot_arms(&self, top_k: usize) -> Vec<usize> {
        let mut indexed: Vec<(usize, f32)> = self
            .delta_history
            .iter()
            .copied()
            .enumerate()
            .filter(|(_, d)| *d > 0.0)
            .collect();

        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        indexed.into_iter().take(top_k).map(|(i, _)| i).collect()
    }

    /// Access the inner absorb-compress layer.
    pub fn inner(&self) -> &AbsorbCompressLayer<P> {
        &self.inner
    }

    /// Mutable access to the inner absorb-compress layer.
    pub fn inner_mut(&mut self) -> &mut AbsorbCompressLayer<P> {
        &mut self.inner
    }

    /// Number of arms tracked.
    pub fn num_arms(&self) -> usize {
        self.delta_history.len()
    }

    /// δ threshold configuration.
    pub fn config(&self) -> &DeltaGatedConfig {
        &self.config
    }
}

impl<P: ScreeningPruner> ScreeningPruner for DeltaGatedAbsorbCompress<P> {
    fn relevance(&self, depth: usize, token_idx: usize, parent_tokens: &[usize]) -> f32 {
        // Delegate to inner layer (which handles compressed arm blocking)
        self.inner.relevance(depth, token_idx, parent_tokens)
    }
}

impl<P: ScreeningPruner> AbsorbCompress for DeltaGatedAbsorbCompress<P> {
    /// Absorb a reward observation — gated by this arm's accumulated δ.
    ///
    /// If the arm hasn't been observed via `observe_delta` yet, the absorb
    /// is skipped (no δ evidence = don't promote).
    fn absorb(&mut self, arm: usize, reward: f32) {
        // Gate: only absorb if we have δ evidence above threshold
        let mean_delta = self.mean_delta(arm);
        if mean_delta >= self.config.delta_threshold {
            self.inner.absorb(arm, reward);
        }
    }

    fn compress(&mut self) -> Vec<usize> {
        self.inner.compress()
    }

    fn compressed_arms(&self) -> &[usize] {
        self.inner.compressed_arms()
    }

    fn should_compress(&self) -> bool {
        self.inner.should_compress()
    }

    /// Dual gate: δ must be meaningful AND reviewer must be net-positive.
    ///
    /// Returns `true` when:
    /// 1. At least one arm has δ above threshold (δ evidence exists)
    /// 2. Inner layer is ready to compress (visit/Q-value criteria)
    /// 3. Review metrics (if provided) show net-positive benefit ratio
    fn should_compress_gated(&self, metrics: Option<&ReviewMetrics>) -> bool {
        // Gate 1: must have δ evidence
        let has_delta_evidence =
            self.delta_history
                .iter()
                .zip(self.delta_counts.iter())
                .any(|(&total, &count)| {
                    count > 0 && (total / count as f32) >= self.config.delta_threshold
                });

        if !has_delta_evidence {
            return false;
        }

        // Gate 2: inner layer ready
        if !self.inner.should_compress() {
            return false;
        }

        // Gate 3: review metrics (if provided)
        let Some(metrics) = metrics else {
            return true;
        };
        let ratio = metrics.benefit_ratio();
        if ratio < self.config.min_benefit_ratio {
            eprintln!(
                "delta_absorb: compression gated — benefit ratio {ratio:.2} < threshold {:.2}",
                self.config.min_benefit_ratio
            );
            return false;
        }

        true
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pruners::absorb_compress::CompressConfig;

    /// No-op screener that allows everything.
    struct AllowAll;

    impl ScreeningPruner for AllowAll {
        fn relevance(&self, _depth: usize, _token_idx: usize, _parent_tokens: &[usize]) -> f32 {
            1.0
        }
    }

    fn make_layer(
        num_arms: usize,
        delta_config: DeltaGatedConfig,
        compress_config: CompressConfig,
    ) -> DeltaGatedAbsorbCompress<AllowAll> {
        let inner = AbsorbCompressLayer::new(AllowAll, num_arms, compress_config);
        DeltaGatedAbsorbCompress::new(inner, num_arms, delta_config)
    }

    #[test]
    fn test_absorb_gated_by_delta() {
        let config = DeltaGatedConfig::new(0.02);
        let compress_config = CompressConfig::new(1, 0.05, 3, 1);
        let mut layer = make_layer(3, config, compress_config);

        // Arm 0: high δ — should absorb
        layer.observe_delta(0, 0.15, 0.5);
        // Arm 1: low δ — should NOT absorb
        layer.observe_delta(1, 0.005, 0.5);

        assert!(layer.mean_delta(0) > 0.02);
        assert!(layer.mean_delta(1) < 0.02);
    }

    #[test]
    fn test_negative_delta_ignored() {
        let config = DeltaGatedConfig::new(0.02);
        let compress_config = CompressConfig::new(1, 0.05, 3, 1);
        let mut layer = make_layer(2, config, compress_config);

        // Negative δ: hint hurt, should not count as blind spot
        layer.observe_delta(0, -0.5, 0.0);

        // delta_history should be 0 (negative clamped)
        assert!((layer.total_delta(0)).abs() < 1e-6);
        assert_eq!(layer.delta_observation_count(0), 1);
        assert!((layer.mean_delta(0)).abs() < 1e-6);
    }

    #[test]
    fn test_blind_spot_arms_ranked() {
        let config = DeltaGatedConfig::new(0.0);
        let compress_config = CompressConfig::new(1, 0.05, 3, 1);
        let mut layer = make_layer(4, config, compress_config);

        layer.observe_delta(0, 0.1, 0.1);
        layer.observe_delta(1, 0.5, 0.5);
        layer.observe_delta(2, 0.3, 0.3);
        // Arm 3: no observations

        let blind = layer.blind_spot_arms(2);
        assert_eq!(blind, vec![1, 2]); // Highest accumulated δ
    }

    #[test]
    fn test_should_compress_gated_requires_delta_evidence() {
        let config = DeltaGatedConfig::new(0.02);
        let compress_config = CompressConfig::new(1, 0.05, 3, 1);
        let mut layer = make_layer(2, config, compress_config);

        // No δ observations at all
        assert!(!layer.should_compress_gated(None));

        // Add absorb to trigger should_compress interval
        layer.inner.absorb(0, 0.01);
        // Still no δ evidence
        assert!(!layer.should_compress_gated(None));
    }

    #[test]
    fn test_should_compress_gated_with_review_metrics() {
        let config = DeltaGatedConfig::new(0.02);
        let compress_config = CompressConfig::new(1, 0.05, 3, 1);
        let mut layer = make_layer(2, config, compress_config);

        // Add δ evidence
        layer.observe_delta(0, 0.15, 0.5);
        // Add absorb to trigger should_compress interval
        layer.inner.absorb(0, 0.01);

        // Net-positive review metrics: helpful = base WRONG, reviewer FIXED
        // benefit_ratio = helpfulness / harmfulness — need > 2.0
        let metrics = ReviewMetrics::new();
        metrics.record(false, true); // helpful: base wrong, reviewer fixed
        metrics.record(false, true); // helpful
        metrics.record(false, true); // helpful (3 helpful, 0 harmful → ratio = ∞)

        assert!(layer.should_compress_gated(Some(&metrics)));
    }

    #[test]
    fn test_should_compress_gated_blocks_negative_reviewer() {
        let config = DeltaGatedConfig::new(0.02).with_benefit_ratio(2.0);
        let compress_config = CompressConfig::new(1, 0.05, 3, 1);
        let mut layer = make_layer(2, config, compress_config);

        // Add δ evidence
        layer.observe_delta(0, 0.15, 0.5);
        // Add absorb to trigger should_compress interval
        layer.inner.absorb(0, 0.01);

        // Net-negative review metrics (more harmful than helpful)
        let metrics = ReviewMetrics::new();
        metrics.record(true, false); // harmful: base correct, reviewed wrong
        metrics.record(true, false); // harmful
        metrics.record(true, true); // helpful (1)

        // benefit_ratio = helpfulness / harmfulness = 0.5 / 1.0 = 0.5 < 2.0
        assert!(metrics.benefit_ratio() < 2.0);
        assert!(!layer.should_compress_gated(Some(&metrics)));
    }

    #[test]
    fn test_absorb_trait_gated_by_delta() {
        let config = DeltaGatedConfig::new(0.02);
        let compress_config = CompressConfig::new(1, 0.05, 3, 1);
        let mut layer = make_layer(2, config, compress_config);

        // No δ evidence yet — absorb should be skipped
        layer.absorb(0, 0.5);
        assert_eq!(layer.inner().total_absorbed(), 0);

        // Add δ evidence — observe_delta also absorbs internally (δ >= threshold)
        layer.observe_delta(0, 0.15, 0.5);
        assert_eq!(layer.inner().total_absorbed(), 1); // internal absorb from observe_delta

        // Explicit absorb should also go through (δ evidence already recorded)
        layer.absorb(0, 0.5);
        assert_eq!(layer.inner().total_absorbed(), 2);
    }

    #[test]
    fn test_compress_delegates_to_inner() {
        let config = DeltaGatedConfig::new(0.0);
        let compress_config = CompressConfig::new(10, 0.05, 3, 100);
        let mut layer = make_layer(3, config, compress_config);

        // Feed enough δ + absorb to compress arm 0
        for _ in 0..20 {
            layer.observe_delta(0, 0.1, 0.01);
        }

        let promoted = layer.compress();
        assert_eq!(promoted, vec![0]);

        // Compressed arm gets relevance 0.0
        assert_eq!(layer.relevance(0, 0, &[]), 0.0);
        assert_eq!(layer.relevance(0, 1, &[]), 1.0);
    }

    #[test]
    fn test_observe_hint_delta_convenience() {
        let config = DeltaGatedConfig::new(0.02);
        let compress_config = CompressConfig::new(1, 0.05, 3, 1);
        let mut layer = make_layer(2, config, compress_config);

        let delta = HintDelta {
            value: 0.15,
            query: "q".into(),
            hint: "h".into(),
            a_hard: "a".into(),
            a_assisted: "b".into(),
            logp_q: -2.0,
            logp_qh: -2.15,
        };

        layer.observe_hint_delta(0, &delta);

        assert!((layer.total_delta(0) - 0.15).abs() < 1e-6);
        assert_eq!(layer.delta_observation_count(0), 1);
    }

    #[test]
    fn test_out_of_bounds_arm_is_noop() {
        let config = DeltaGatedConfig::default();
        let compress_config = CompressConfig::default();
        let mut layer = make_layer(2, config, compress_config);

        layer.observe_delta(99, 0.5, 0.5);
        layer.absorb(99, 0.5);

        assert_eq!(layer.inner().total_absorbed(), 0);
    }

    #[test]
    fn test_mean_delta_for_unobserved_arm() {
        let config = DeltaGatedConfig::default();
        let compress_config = CompressConfig::default();
        let layer = make_layer(3, config, compress_config);

        assert!((layer.mean_delta(2)).abs() < 1e-6);
        assert_eq!(layer.delta_observation_count(2), 0);
    }
}
