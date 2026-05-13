//! Bandit pruner using Hint-δ as dense reward signal.
//!
//! Instead of waiting for environment reward (compile success, game score),
//! use the model's own predictive shift (δ) as an immediate, dense reward.
//!
//! # Why δ-reward is better than environment reward
//!
//! - **Dense:** Every token scored, not just episode outcome
//! - **Immediate:** No need to wait for episode completion
//! - **Intrinsic:** Derived from model's own distribution, no external oracle
//! - **Targeted:** High δ = blind spot = exactly where the model needs exploration
//!
//! # Usage
//!
//! ```rust,ignore
//! let pruner = DeltaBanditPruner::new(
//!     BanditPruner::new(domain_screener, BanditStrategy::Ucb1, 10),
//!     10,
//! );
//!
//! // Feed δ signal — replaces traditional reward observation
//! pruner.observe_delta(3, 0.42);
//!
//! // Find blind spots for targeted exploration
//! let blind = pruner.blind_spot_arms(3);
//! ```

use std::cmp::Ordering;

use crate::pruners::bandit::BanditPruner;
use crate::speculative::types::ScreeningPruner;

use super::types::HintDelta;

// ── DeltaBanditPruner ────────────────────────────────────────────

/// Bandit pruner using Hint-δ as the reward signal for arm selection.
///
/// Wraps [`BanditPruner`] and feeds δ observations as rewards.
/// δ is dense and immediate — every scored token contributes to arm selection.
///
/// # Architecture
///
/// ```text
/// DeltaBanditPruner<P>
///   ├── inner: BanditPruner<P>       (existing bandit logic)
///   ├── delta_weights: Vec<f32>       (per-arm accumulated δ)
///   └── delta_floor: f32              (minimum δ to count as reward)
/// ```
///
/// Maps directly to `ScreeningPruner::relevance()` philosophy —
/// δ replaces raw environment reward as the arm quality signal.
pub struct DeltaBanditPruner<P: ScreeningPruner> {
    /// Inner bandit pruner (delegates arm selection logic).
    inner: BanditPruner<P>,
    /// Per-arm accumulated δ (total blind-spot density).
    delta_weights: Vec<f32>,
    /// Per-arm δ observation count (for mean tracking).
    delta_counts: Vec<usize>,
    /// Minimum δ to count as reward (default: 0.0).
    ///
    /// Negative δ means hint hurt — ignore those observations.
    delta_floor: f32,
}

impl<P: ScreeningPruner> DeltaBanditPruner<P> {
    /// Create a new δ-driven bandit pruner.
    ///
    /// Wraps an existing `BanditPruner` with δ-based reward feeding.
    pub fn new(inner: BanditPruner<P>, num_arms: usize) -> Self {
        Self {
            inner,
            delta_weights: vec![0.0; num_arms],
            delta_counts: vec![0; num_arms],
            delta_floor: 0.0,
        }
    }

    /// Create with custom δ floor (minimum δ to count as reward).
    pub fn with_delta_floor(mut self, floor: f32) -> Self {
        self.delta_floor = floor;
        self
    }

    /// Feed δ signal as reward to the bandit.
    ///
    /// δ is clamped to `[delta_floor, ∞)` before feeding as reward.
    /// Negative δ (hint hurt) is ignored — not a blind spot indicator.
    ///
    /// The bandit's Q-value for this arm is updated incrementally:
    /// `Q(arm) += (reward - Q(arm)) / visits(arm)`
    pub fn observe_delta(&mut self, arm: usize, delta: f32) {
        if arm >= self.delta_weights.len() {
            return;
        }

        // Accumulate δ (only positive counts as blind spot signal)
        let effective_delta = delta.max(self.delta_floor);
        self.delta_weights[arm] += effective_delta;
        self.delta_counts[arm] += 1;

        // Feed as reward to inner bandit
        self.inner.update(arm, effective_delta);
    }

    /// Feed a [`HintDelta`] directly — convenience wrapper for [`observe_delta`](Self::observe_delta).
    pub fn observe_hint_delta(&mut self, arm: usize, delta: &HintDelta) {
        self.observe_delta(arm, delta.value);
    }

    /// Which arms have the highest accumulated blind-spot density (top-K)?
    ///
    /// Returns arms sorted by total accumulated δ, descending.
    /// Only arms with at least one δ observation are included.
    ///
    /// Useful for targeting [`super::template_proposer::TemplateProposer`]
    /// toward the model's weakest areas.
    pub fn blind_spot_arms(&self, top_k: usize) -> Vec<usize> {
        let mut indexed: Vec<(usize, f32)> = self
            .delta_weights
            .iter()
            .copied()
            .enumerate()
            .filter(|(_, d)| *d > 0.0)
            .collect();

        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        indexed.into_iter().take(top_k).map(|(i, _)| i).collect()
    }

    /// Mean δ for a specific arm.
    ///
    /// Returns 0.0 for unobserved arms.
    pub fn mean_delta(&self, arm: usize) -> f32 {
        if arm >= self.delta_counts.len() || self.delta_counts[arm] == 0 {
            return 0.0;
        }
        self.delta_weights[arm] / self.delta_counts[arm] as f32
    }

    /// Accumulated δ for a specific arm.
    pub fn total_delta(&self, arm: usize) -> f32 {
        self.delta_weights.get(arm).copied().unwrap_or(0.0)
    }

    /// Number of δ observations for a specific arm.
    pub fn delta_observation_count(&self, arm: usize) -> usize {
        self.delta_counts.get(arm).copied().unwrap_or(0)
    }

    /// Access the inner bandit pruner.
    pub fn inner(&self) -> &BanditPruner<P> {
        &self.inner
    }

    /// Mutable access to the inner bandit pruner.
    pub fn inner_mut(&mut self) -> &mut BanditPruner<P> {
        &mut self.inner
    }

    /// Number of arms tracked.
    pub fn num_arms(&self) -> usize {
        self.delta_weights.len()
    }

    /// δ floor configuration.
    pub fn delta_floor(&self) -> f32 {
        self.delta_floor
    }
}

// Delegate ScreeningPruner to inner BanditPruner
impl<P: ScreeningPruner> ScreeningPruner for DeltaBanditPruner<P> {
    fn relevance(&self, depth: usize, token_idx: usize, parent_tokens: &[usize]) -> f32 {
        self.inner.relevance(depth, token_idx, parent_tokens)
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pruners::BanditStrategy;
    use crate::speculative::types::NoScreeningPruner;

    fn make_pruner(num_arms: usize) -> DeltaBanditPruner<NoScreeningPruner> {
        let inner = BanditPruner::new(NoScreeningPruner, BanditStrategy::Ucb1, num_arms);
        DeltaBanditPruner::new(inner, num_arms)
    }

    #[test]
    fn test_observe_delta_feeds_reward() {
        let mut pruner = make_pruner(3);

        // Feed δ for arm 2
        pruner.observe_delta(2, 0.5);

        // δ should be accumulated
        assert!((pruner.total_delta(2) - 0.5).abs() < 1e-6);
        assert_eq!(pruner.delta_observation_count(2), 1);

        // Q-value should be updated (delta clamped to floor, which is 0.0)
        let q = pruner.inner().q_values()[2];
        assert!(q > 0.0, "Q-value should be positive after δ observation");
    }

    #[test]
    fn test_negative_delta_clamped_to_floor() {
        let mut pruner = make_pruner(3);

        // Negative δ: hint hurt, should be clamped to floor (0.0)
        pruner.observe_delta(0, -0.5);

        // delta_weights should be 0 (negative clamped to floor)
        assert!((pruner.total_delta(0)).abs() < 1e-6);
        assert_eq!(pruner.delta_observation_count(0), 1);

        // Q-value should be 0 (reward was 0.0)
        let q = pruner.inner().q_values()[0];
        assert!((q).abs() < 1e-6);
    }

    #[test]
    fn test_custom_delta_floor() {
        let mut pruner = make_pruner(3).with_delta_floor(0.1);

        // δ below floor should be clamped
        pruner.observe_delta(0, 0.05);
        assert!((pruner.total_delta(0) - 0.1).abs() < 1e-6);

        // δ above floor passes through
        pruner.observe_delta(1, 0.5);
        assert!((pruner.total_delta(1) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_blind_spot_arms_ranked() {
        let mut pruner = make_pruner(5);

        pruner.observe_delta(0, 0.1);
        pruner.observe_delta(1, 0.5);
        pruner.observe_delta(2, 0.3);
        pruner.observe_delta(3, 0.8);
        // Arm 4: no observations

        let blind = pruner.blind_spot_arms(3);
        assert_eq!(blind, vec![3, 1, 2], "Should be sorted by δ descending");
    }

    #[test]
    fn test_blind_spot_arms_excludes_unobserved() {
        let pruner = make_pruner(3);
        let blind = pruner.blind_spot_arms(5);
        assert!(blind.is_empty(), "No observations = no blind spots");
    }

    #[test]
    fn test_blind_spot_arms_top_k_limits() {
        let mut pruner = make_pruner(5);

        for i in 0..5 {
            pruner.observe_delta(i, i as f32 * 0.1);
        }

        let blind = pruner.blind_spot_arms(2);
        assert_eq!(blind.len(), 2);
        assert_eq!(blind[0], 4); // Highest δ
        assert_eq!(blind[1], 3); // Second highest
    }

    #[test]
    fn test_mean_delta() {
        let mut pruner = make_pruner(3);

        pruner.observe_delta(0, 0.3);
        pruner.observe_delta(0, 0.5);

        let mean = pruner.mean_delta(0);
        assert!((mean - 0.4).abs() < 1e-6);
    }

    #[test]
    fn test_mean_delta_unobserved() {
        let pruner = make_pruner(3);
        assert!((pruner.mean_delta(2)).abs() < 1e-6);
    }

    #[test]
    fn test_observe_hint_delta() {
        let mut pruner = make_pruner(3);

        let delta = HintDelta {
            value: 0.42,
            query: "q".into(),
            hint: "h".into(),
            a_hard: "a".into(),
            a_assisted: "b".into(),
            logp_q: -2.0,
            logp_qh: -2.42,
        };

        pruner.observe_hint_delta(1, &delta);

        assert!((pruner.total_delta(1) - 0.42).abs() < 1e-6);
        assert_eq!(pruner.delta_observation_count(1), 1);
    }

    #[test]
    fn test_out_of_bounds_arm_is_noop() {
        let mut pruner = make_pruner(2);
        pruner.observe_delta(99, 0.5);

        // No panic, no changes
        assert_eq!(pruner.num_arms(), 2);
        assert!((pruner.total_delta(0)).abs() < 1e-6);
    }

    #[test]
    fn test_delegates_relevance_to_inner() {
        let pruner = make_pruner(3);

        // NoScreeningPruner always returns 1.0
        let rel = pruner.relevance(0, 0, &[]);
        assert!((rel - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_multiple_deltas_accumulate() {
        let mut pruner = make_pruner(2);

        for _ in 0..10 {
            pruner.observe_delta(0, 0.15);
        }

        assert!((pruner.total_delta(0) - 1.5).abs() < 1e-6);
        assert_eq!(pruner.delta_observation_count(0), 10);
        assert!((pruner.mean_delta(0) - 0.15).abs() < 1e-6);
    }
}
