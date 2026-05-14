//! Intra-trajectory reward shaping distilled from StepCodeReasoner (arXiv 2605.11922).
//!
//! Paper Eq. 11: Â_intra(i,g) = r_{i,g} × (1 + (1/(n-i)) × Σ_{j=i+1}^{n} r_{j,g})
//!
//! Our adaptation: after DDTree verification, scan the accepted path and compute
//! shaped rewards for each arm based on how many subsequent arms were also correct.
//!
//! Properties preserved from paper:
//! 1. Only correct arms get non-zero shaped reward
//! 2. Arms enabling more correct future arms get boosted
//! 3. No discount factor or value function needed
//!
//! λ = 0.3 (paper default). λ = 0.0 reverts to flat binary rewards.
//!
//! # Feature Gate
//!
//! All types behind `#[cfg(feature = "stepcode")]` which requires `bandit`.

use crate::speculative::types::ScreeningPruner;

use super::absorb_compress::{AbsorbCompress, AbsorbCompressLayer};
use super::bandit::BanditPruner;

#[cfg(feature = "g_zero")]
use super::g_zero::DeltaBanditPruner;

// ── PathStep ────────────────────────────────────────────────────

/// A single step in a verified DDTree path.
#[derive(Clone, Debug)]
pub struct PathStep {
    /// Arm (token index) selected at this depth.
    pub arm: usize,
    /// Depth in the DDTree.
    pub depth: usize,
    /// Binary reward: 1.0 if accepted/verified, 0.0 if rejected.
    pub reward: f32,
}

// ── ShapedPath ──────────────────────────────────────────────────

/// Result of shaping a verification path.
///
/// Each step's reward is boosted proportionally to how many subsequent
/// steps were also correct — rewarding arms that "enable" downstream success.
#[derive(Clone, Debug)]
pub struct ShapedPath {
    /// Original steps.
    pub steps: Vec<PathStep>,
    /// Shaped rewards (same length as steps).
    pub shaped_rewards: Vec<f32>,
    /// Shaping coefficient λ (0.0 = flat, 0.3 = paper default).
    pub lambda: f32,
    /// Fraction of steps that were correct (path consistency).
    pub consistency: f32,
}

impl ShapedPath {
    /// Compute shaped rewards for a verified path.
    ///
    /// # Formula (paper Eq. 11)
    ///
    /// ```text
    /// shaped_reward[i] = reward[i] × (1 + λ × future_accuracy[i])
    /// future_accuracy[i] = count_correct[i+1..n] / (n - i)
    /// ```
    ///
    /// # Complexity
    ///
    /// O(n) with suffix-sum precomputation (n = path length ≤ block_size = 16).
    ///
    /// # Arguments
    ///
    /// * `steps` — verified path from DDTree (accepted + rejected arms)
    /// * `lambda` — shaping coefficient (0.0 = flat, 0.3 = paper default)
    pub fn shape(steps: Vec<PathStep>, lambda: f32) -> Self {
        let n = steps.len();
        let mut shaped_rewards = vec![0.0; n];

        if n == 0 {
            return Self {
                steps,
                shaped_rewards,
                lambda,
                consistency: 0.0,
            };
        }

        // Suffix sum of rewards: suffix_correct[i] = sum of rewards[i+1..n]
        let mut suffix_correct = vec![0.0f32; n];
        for i in (0..n.saturating_sub(1)).rev() {
            suffix_correct[i] = suffix_correct[i + 1] + steps[i + 1].reward;
        }

        // Compute shaped rewards
        for i in 0..n {
            let remaining = (n - i - 1) as f32;
            let future_accuracy = if remaining > 0.0 {
                suffix_correct[i] / remaining
            } else {
                0.0 // terminal step: no future to shape
            };
            shaped_rewards[i] = steps[i].reward * (1.0 + lambda * future_accuracy);
        }

        // Path consistency = fraction of correct steps
        let correct_count = steps.iter().map(|s| s.reward).sum::<f32>();
        let consistency = correct_count / n as f32;

        Self {
            steps,
            shaped_rewards,
            lambda,
            consistency,
        }
    }

    /// Feed shaped rewards into a BanditPruner.
    ///
    /// Calls `BanditPruner::update(arm, shaped_reward)` for each step.
    /// Steps with reward = 0.0 are skipped (no information gain).
    pub fn apply_to_bandit<P: ScreeningPruner>(&self, bandit: &mut BanditPruner<P>) {
        for (step, shaped) in self.steps.iter().zip(self.shaped_rewards.iter()) {
            if *shaped > 0.0 {
                bandit.update(step.arm, *shaped);
            }
        }
    }

    /// Feed shaped rewards into a DeltaBanditPruner (G-Zero).
    ///
    /// Uses shaped reward as the dense reward signal for δ-gated arms.
    /// The shaped reward is fed via `observe_delta` as a δ-like signal
    /// where arms enabling downstream success get proportionally more credit.
    #[cfg(feature = "g_zero")]
    pub fn apply_to_delta_bandit<P: ScreeningPruner>(&self, bandit: &mut DeltaBanditPruner<P>) {
        for (step, shaped) in self.steps.iter().zip(self.shaped_rewards.iter()) {
            if *shaped > 0.0 {
                bandit.observe_delta(step.arm, *shaped);
            }
        }
    }

    /// Feed shaped rewards into AbsorbCompress layer.
    ///
    /// Promotes arms that consistently enable downstream success.
    /// Positive shaped rewards reinforce; zero rewards are still absorbed
    /// for accurate Q-value tracking.
    pub fn apply_to_absorb<P: ScreeningPruner>(&self, layer: &mut AbsorbCompressLayer<P>) {
        for (step, shaped) in self.steps.iter().zip(self.shaped_rewards.iter()) {
            layer.absorb(step.arm, *shaped);
        }
    }
}

// ── Convenience Functions ───────────────────────────────────────

/// Convenience: shape a flat `(arm, reward)` path with given λ.
///
/// Use this when you don't need depth tracking.
pub fn shape_path(path: &[(usize, f32)], lambda: f32) -> Vec<(usize, f32)> {
    let steps: Vec<PathStep> = path
        .iter()
        .enumerate()
        .map(|(i, (arm, reward))| PathStep {
            arm: *arm,
            depth: i,
            reward: *reward,
        })
        .collect();
    let shaped = ShapedPath::shape(steps, lambda);
    shaped
        .steps
        .iter()
        .zip(shaped.shaped_rewards.iter())
        .map(|(s, r)| (s.arm, *r))
        .collect()
}

/// Convenience: compute path consistency from a flat reward path.
///
/// Returns fraction of correct steps (0.0 to 1.0).
pub fn path_consistency(rewards: &[f32]) -> f32 {
    if rewards.is_empty() {
        return 0.0;
    }
    let correct = rewards.iter().filter(|&&r| r > 0.0).count();
    correct as f32 / rewards.len() as f32
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shape_all_correct() {
        let steps: Vec<PathStep> = (0..5)
            .map(|i| PathStep {
                arm: i,
                depth: i,
                reward: 1.0,
            })
            .collect();
        let shaped = ShapedPath::shape(steps, 0.3);

        // All steps correct → consistency = 1.0
        assert!((shaped.consistency - 1.0).abs() < 1e-6);

        // Terminal step gets no future shaping: 1.0 × (1 + 0.3 × 0) = 1.0
        let last = shaped.shaped_rewards.last().unwrap();
        assert!((last - 1.0).abs() < 1e-6);

        // First step gets full future shaping: 1.0 × (1 + 0.3 × 1.0) = 1.3
        let first = shaped.shaped_rewards.first().unwrap();
        assert!((first - 1.3).abs() < 1e-6);

        // Rewards should be monotonically decreasing (earlier → more future)
        for w in shaped.shaped_rewards.windows(2) {
            assert!(w[0] >= w[1]);
        }
    }

    #[test]
    fn test_shape_all_wrong() {
        let steps: Vec<PathStep> = (0..5)
            .map(|i| PathStep {
                arm: i,
                depth: i,
                reward: 0.0,
            })
            .collect();
        let shaped = ShapedPath::shape(steps, 0.3);

        // All wrong → all shaped = 0.0 (property 1: r_i=0 → shaped=0)
        assert!(shaped.shaped_rewards.iter().all(|&r| r == 0.0));
        assert!((shaped.consistency - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_shape_terminal_flat() {
        let steps = vec![
            PathStep {
                arm: 0,
                depth: 0,
                reward: 1.0,
            },
            PathStep {
                arm: 1,
                depth: 1,
                reward: 1.0,
            },
        ];
        let shaped = ShapedPath::shape(steps, 0.5);

        // Terminal step (depth=1) gets no future: 1.0 × (1 + 0.5 × 0) = 1.0
        assert!((shaped.shaped_rewards[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_shape_lambda_zero() {
        let steps: Vec<PathStep> = (0..3)
            .map(|i| PathStep {
                arm: i,
                depth: i,
                reward: 1.0,
            })
            .collect();
        let shaped = ShapedPath::shape(steps, 0.0);

        // λ=0 → all shaped rewards = raw reward (flat binary)
        for r in &shaped.shaped_rewards {
            assert!((r - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_shape_enables_downstream() {
        // Arm 0 is correct, arm 1 is correct → arm 0 "enabled" downstream success
        // Arm 2 is wrong → terminal doesn't matter
        let steps = vec![
            PathStep {
                arm: 0,
                depth: 0,
                reward: 1.0,
            },
            PathStep {
                arm: 1,
                depth: 1,
                reward: 1.0,
            },
            PathStep {
                arm: 2,
                depth: 2,
                reward: 0.0,
            },
        ];
        let shaped = ShapedPath::shape(steps, 0.3);

        // Arm 0: 1.0 × (1 + 0.3 × (1+0)/2) = 1.0 × 1.15 = 1.15
        assert!((shaped.shaped_rewards[0] - 1.15).abs() < 1e-5);

        // Arm 1: 1.0 × (1 + 0.3 × 0/1) = 1.0
        assert!((shaped.shaped_rewards[1] - 1.0).abs() < 1e-5);

        // Arm 2: 0.0 × anything = 0.0
        assert!((shaped.shaped_rewards[2] - 0.0).abs() < 1e-6);

        // Arm 0 gets higher reward than arm 1 (enabled more future success)
        assert!(shaped.shaped_rewards[0] > shaped.shaped_rewards[1]);
    }

    #[test]
    fn test_shape_empty_path() {
        let shaped = ShapedPath::shape(vec![], 0.3);
        assert!(shaped.shaped_rewards.is_empty());
        assert!((shaped.consistency - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_path_consistency_full() {
        let rewards = vec![1.0, 1.0, 1.0, 1.0];
        assert!((path_consistency(&rewards) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_path_consistency_mixed() {
        let rewards = vec![1.0, 0.0, 1.0, 0.0, 1.0];
        // 3/5 correct = 0.6
        assert!((path_consistency(&rewards) - 0.6).abs() < 1e-6);
    }
}
