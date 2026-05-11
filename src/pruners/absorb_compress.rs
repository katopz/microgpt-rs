//! Absorb-compress cycle for Heuristic Learning — promote stable bandit knowledge to hard constraints.
//!
//! The absorb-compress cycle is the core of HL's "compression" operation:
//! low-Q arms that have been visited enough times get promoted to hard blocks,
//! removing them from the bandit's exploration space entirely.
//!
//! # Usage
//!
//! ```rust,ignore
//! let layer = AbsorbCompressLayer::new(
//!     NoScreeningPruner,
//!     5,  // 5 arms
//!     CompressConfig::default(),
//! );
//!
//! // Feed observations
//! layer.absorb(0, 0.01);
//! layer.absorb(0, 0.02);
//! // ... many episodes later ...
//!
//! if layer.should_compress() {
//!     let promoted = layer.compress();
//!     println!("Hard-blocked arms: {promoted:?}");
//! }
//! ```
//!
//! # Benefit-Ratio Gate (Plan 036)
//!
//! When review metrics are available, compression can be gated by the
//! benefit-to-risk ratio. If the reviewer is net-negative (ratio below
//! threshold), compressing its decisions into hard blocks would be harmful.
//! Use [`AbsorbCompressLayer::should_compress_gated`] to check.

use std::collections::HashSet;

use crate::speculative::types::ScreeningPruner;

use super::review_metrics::ReviewMetrics;

// ── Config ──────────────────────────────────────────────────────

/// Tunable thresholds for the absorb-compress cycle.
///
/// An arm is promoted to a hard block when:
/// 1. It has been visited at least `min_visits` times
/// 2. Its average reward (Q-value) is below `q_threshold`
/// 3. `compress()` is called and it's among the worst `promote_count` arms
///
/// # Benefit-Ratio Gate (Plan 036)
///
/// When `min_benefit_ratio` is set, compression is only allowed when the
/// reviewer's benefit-to-risk ratio exceeds this threshold. This prevents
/// hardening a net-negative reviewer's decisions into permanent blocks.
/// Paper default: 2.0 (conservative — allows slightly worse reviewers).
#[derive(Clone, Debug)]
pub struct CompressConfig {
    /// Minimum visits before an arm is eligible for compression.
    pub min_visits: usize,
    /// Q-value threshold — arms with average reward below this are candidates.
    pub q_threshold: f32,
    /// Maximum arms to promote per `compress()` call.
    pub promote_count: usize,
    /// Check `should_compress()` every N `absorb()` calls.
    pub check_interval: usize,
    /// Minimum benefit-to-risk ratio to allow compression (Plan 036).
    ///
    /// When review metrics show `benefit_ratio() < min_benefit_ratio`,
    /// compression is blocked — the reviewer is net-negative and
    /// hardening its decisions would be harmful.
    ///
    /// Default: 2.0 (conservative). Paper found 3.1:1 for o3-mini.
    /// Set to 0.0 to disable the gate.
    pub min_benefit_ratio: f64,
}

impl Default for CompressConfig {
    fn default() -> Self {
        Self {
            min_visits: 200,
            q_threshold: 0.05,
            promote_count: 3,
            check_interval: 100,
            min_benefit_ratio: 2.0,
        }
    }
}

impl CompressConfig {
    /// Create config with custom thresholds for testing or tuning.
    pub fn new(
        min_visits: usize,
        q_threshold: f32,
        promote_count: usize,
        check_interval: usize,
    ) -> Self {
        Self {
            min_visits,
            q_threshold,
            promote_count,
            check_interval,
            min_benefit_ratio: 2.0,
        }
    }

    /// Create config with custom benefit-ratio threshold.
    pub fn with_benefit_ratio(mut self, min_benefit_ratio: f64) -> Self {
        self.min_benefit_ratio = min_benefit_ratio;
        self
    }
}

// ── Trait ───────────────────────────────────────────────────────

/// Trait for the absorb-compress cycle in Heuristic Learning.
///
/// Extends [`ScreeningPruner`] — compressed arms get relevance 0.0
/// (hard block), overriding the inner pruner's score.
///
/// This trait enables `BanditPruner<P>` to delegate compression to
/// any inner pruner `P` that implements `ScreeningPruner + AbsorbCompress`.
pub trait AbsorbCompress: ScreeningPruner {
    /// Feed a new (arm, reward) observation for compression tracking.
    fn absorb(&mut self, arm: usize, reward: f32);

    /// Promote stable low-Q arms to hard blocks.
    ///
    /// Returns indices of newly promoted arms (may be empty).
    /// Idempotent: already-compressed arms are skipped.
    fn compress(&mut self) -> Vec<usize>;

    /// Arms already promoted to hard constraints.
    fn compressed_arms(&self) -> &[usize];

    /// Whether enough observations have been absorbed to trigger compression.
    fn should_compress(&self) -> bool;

    /// Whether compression is allowed given review metrics (Plan 036).
    ///
    /// Returns `true` when:
    /// - `metrics` is `None` (no gate, fall through), OR
    /// - `metrics.benefit_ratio() >= min_benefit_ratio` (reviewer is net-positive)
    ///
    /// Returns `false` when the reviewer is net-negative (ratio below threshold).
    fn should_compress_gated(&self, metrics: Option<&ReviewMetrics>) -> bool;
}

// ── Layer ───────────────────────────────────────────────────────

/// Concrete absorb-compress layer wrapping any [`ScreeningPruner`].
///
/// Tracks per-arm reward statistics and promotes low-performing arms
/// to hard blocks when compression thresholds are met.
///
/// Compressed arms get `relevance() = 0.0` regardless of the inner pruner.
pub struct AbsorbCompressLayer<P: ScreeningPruner> {
    inner: P,
    arm_reward_sums: Vec<f32>,
    arm_visits: Vec<usize>,
    compressed: Vec<usize>,
    /// O(1) lookup for compressed arms (mirrors `compressed` vec).
    compressed_set: HashSet<usize>,
    config: CompressConfig,
    total_absorbed: usize,
}

impl<P: ScreeningPruner> AbsorbCompressLayer<P> {
    /// Create a new absorb-compress layer wrapping `inner` with `num_arms` tracking slots.
    pub fn new(inner: P, num_arms: usize, config: CompressConfig) -> Self {
        Self {
            inner,
            arm_reward_sums: vec![0.0; num_arms],
            arm_visits: vec![0; num_arms],
            compressed: Vec::new(),
            compressed_set: HashSet::new(),
            config,
            total_absorbed: 0,
        }
    }

    /// Access the inner pruner.
    pub fn inner(&self) -> &P {
        &self.inner
    }

    /// Mutable access to the inner pruner.
    pub fn inner_mut(&mut self) -> &mut P {
        &mut self.inner
    }

    /// Number of arms tracked.
    pub fn num_arms(&self) -> usize {
        self.arm_visits.len()
    }

    /// Total observations absorbed so far.
    pub fn total_absorbed(&self) -> usize {
        self.total_absorbed
    }

    /// Average reward (Q-value) for a specific arm from absorbed data.
    pub fn arm_q_value(&self, arm: usize) -> f32 {
        if arm >= self.arm_visits.len() || self.arm_visits[arm] == 0 {
            return 0.0;
        }
        self.arm_reward_sums[arm] / self.arm_visits[arm] as f32
    }

    /// Visit count for a specific arm from absorbed data.
    pub fn arm_visits(&self, arm: usize) -> usize {
        self.arm_visits.get(arm).copied().unwrap_or(0)
    }
}

impl<P: ScreeningPruner> ScreeningPruner for AbsorbCompressLayer<P> {
    fn relevance(&self, depth: usize, token_idx: usize, parent_tokens: &[usize]) -> f32 {
        // Hard block compressed arms — O(1) via HashSet
        if self.compressed_set.contains(&token_idx) {
            return 0.0;
        }
        // Delegate to inner pruner
        self.inner.relevance(depth, token_idx, parent_tokens)
    }
}

impl<P: ScreeningPruner> AbsorbCompress for AbsorbCompressLayer<P> {
    fn absorb(&mut self, arm: usize, reward: f32) {
        if arm >= self.arm_visits.len() {
            return;
        }
        self.arm_reward_sums[arm] += reward;
        self.arm_visits[arm] += 1;
        self.total_absorbed += 1;
    }

    fn compress(&mut self) -> Vec<usize> {
        // Find candidate arms: visited enough, low Q, not already compressed
        let mut candidates: Vec<(usize, f32)> = (0..self.arm_visits.len())
            .filter(|&arm| {
                self.arm_visits[arm] >= self.config.min_visits && !self.compressed.contains(&arm)
            })
            .map(|arm| (arm, self.arm_q_value(arm)))
            .filter(|(_, q)| *q < self.config.q_threshold)
            .collect();

        // Sort by Q-value ascending (worst first)
        candidates.sort_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // Take top promote_count worst arms
        let promoted: Vec<usize> = candidates
            .into_iter()
            .take(self.config.promote_count)
            .map(|(arm, _)| arm)
            .collect();

        self.compressed_set.extend(promoted.iter().copied());
        self.compressed.extend_from_slice(&promoted);
        promoted
    }

    fn compressed_arms(&self) -> &[usize] {
        &self.compressed
    }

    fn should_compress(&self) -> bool {
        self.total_absorbed > 0
            && self
                .total_absorbed
                .is_multiple_of(self.config.check_interval)
    }

    fn should_compress_gated(&self, metrics: Option<&ReviewMetrics>) -> bool {
        if !self.should_compress() {
            return false;
        }
        // No metrics → no gate, fall through to original behavior
        let Some(metrics) = metrics else {
            return true;
        };
        // Gate: only compress when reviewer is net-positive
        let ratio = metrics.benefit_ratio();
        if ratio < self.config.min_benefit_ratio {
            eprintln!(
                "absorb_compress: compression gated — benefit ratio {ratio:.2} < threshold {:.2}",
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

    /// No-op screener that allows everything.
    struct AllowAll;

    impl ScreeningPruner for AllowAll {
        fn relevance(&self, _depth: usize, _token_idx: usize, _parent_tokens: &[usize]) -> f32 {
            1.0
        }
    }

    fn make_layer(num_arms: usize, config: CompressConfig) -> AbsorbCompressLayer<AllowAll> {
        AbsorbCompressLayer::new(AllowAll, num_arms, config)
    }

    #[test]
    fn test_no_compress_under_threshold() {
        let config = CompressConfig::new(200, 0.05, 3, 100);
        let mut layer = make_layer(5, config);

        // Only 50 visits — below min_visits=200
        for _ in 0..50 {
            layer.absorb(0, 0.01);
        }

        assert!(!layer.should_compress());
        let promoted = layer.compress();
        assert!(promoted.is_empty());
        assert!(layer.compressed_arms().is_empty());
    }

    #[test]
    fn test_compress_fires_at_threshold() {
        let config = CompressConfig::new(10, 0.05, 3, 100);
        let mut layer = make_layer(5, config);

        // Arm 0: low reward, many visits → should be compressed
        for _ in 0..20 {
            layer.absorb(0, 0.01);
        }
        // Arm 1: high reward, many visits → should NOT be compressed
        for _ in 0..20 {
            layer.absorb(1, 0.9);
        }

        // Manually trigger compress (bypass should_compress interval check)
        let promoted = layer.compress();

        assert_eq!(promoted, vec![0]); // Only arm 0 compressed
        assert!(layer.compressed_arms().contains(&0));
        assert!(!layer.compressed_arms().contains(&1));
    }

    #[test]
    fn test_compressed_arms_blocked() {
        let config = CompressConfig::new(10, 0.05, 3, 100);
        let mut layer = make_layer(5, config);

        // Arm 0 gets compressed
        for _ in 0..20 {
            layer.absorb(0, 0.01);
        }
        layer.compress();

        // Verify relevance is 0.0 for compressed arm
        assert_eq!(layer.relevance(0, 0, &[]), 0.0);
        // Non-compressed arm still passes through
        assert_eq!(layer.relevance(0, 1, &[]), 1.0);
    }

    #[test]
    fn test_double_compress_idempotent() {
        let config = CompressConfig::new(10, 0.05, 3, 100);
        let mut layer = make_layer(5, config);

        for _ in 0..20 {
            layer.absorb(0, 0.01);
        }

        let first = layer.compress();
        let second = layer.compress();

        assert_eq!(first, vec![0]);
        assert!(second.is_empty()); // Already compressed
        assert_eq!(layer.compressed_arms().len(), 1);
    }

    #[test]
    fn test_should_compress_interval() {
        let config = CompressConfig::new(10, 0.05, 3, 5);
        let mut layer = make_layer(3, config);

        // 0 absorbs: should_compress = false
        assert!(!layer.should_compress());

        // 5 absorbs: should_compress = true (5 % 5 == 0)
        for i in 0..5 {
            layer.absorb(i % 3, 0.5);
        }
        assert!(layer.should_compress());

        // 6 absorbs: should_compress = false
        layer.absorb(0, 0.5);
        assert!(!layer.should_compress());

        // 10 absorbs: should_compress = true again
        for i in 0..4 {
            layer.absorb(i % 3, 0.5);
        }
        assert!(layer.should_compress());
    }

    #[test]
    fn test_arm_q_value_tracking() {
        let config = CompressConfig::default();
        let mut layer = make_layer(3, config);

        layer.absorb(0, 1.0);
        layer.absorb(0, 0.0);
        layer.absorb(1, 0.8);

        assert!((layer.arm_q_value(0) - 0.5).abs() < 0.01);
        assert!((layer.arm_q_value(1) - 0.8).abs() < 0.01);
        assert_eq!(layer.arm_q_value(2), 0.0); // Never visited
        assert_eq!(layer.arm_visits(0), 2);
        assert_eq!(layer.arm_visits(1), 1);
        assert_eq!(layer.total_absorbed(), 3);
    }

    #[test]
    fn test_out_of_bounds_arm_absorb_is_noop() {
        let config = CompressConfig::default();
        let mut layer = make_layer(2, config);

        layer.absorb(99, 0.5); // Out of bounds — should be ignored
        assert_eq!(layer.total_absorbed(), 0);
    }
}
