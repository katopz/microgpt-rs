//! Regression suite for Heuristic Learning — replay golden episodes to detect regressions.
//!
//! Extracts top-performing episodes from a [`TrialLog`] as golden traces,
//! then replays them through a fresh pruner to verify no regression occurred.
//!
//! # Usage
//!
//! ```rust,ignore
//! // Extract top 10 episodes from trial log
//! let suite = RegressionSuite::from_trials(Path::new("/tmp/trials.jsonl"), 10)?;
//!
//! // Replay through a fresh pruner
//! let result = suite.run(|_trace| {
//!     // Create fresh pruner for each trace
//!     BanditPruner::new(domain_screener, BanditStrategy::Ucb1, 5)
//! });
//!
//! println!("Passed: {}/{}", result.passed, result.total());
//! ```

use std::io::Result;
use std::path::Path;

use serde::{Deserialize, Serialize};

use super::trial_log::TrialLog;

// ── Golden Trace ────────────────────────────────────────────────

/// A recorded episode trajectory used for regression testing.
///
/// Represents a "golden" episode that the system should be able to reproduce
/// with at least equivalent performance. Actions are the sequence of arms pulled.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GoldenTrace {
    /// Human-readable label (e.g., "ep_042_best").
    pub label: String,
    /// Sequence of arm selections in the episode.
    pub actions: Vec<usize>,
    /// Expected cumulative reward for this trace.
    pub expected_reward: f32,
    /// Whether the agent was expected to survive (reward > threshold).
    pub expected_survival: bool,
}

// ── Regression Failure ──────────────────────────────────────────

/// Details of a failed regression test.
#[derive(Clone, Debug)]
pub struct RegressionFailure {
    /// Label of the failing trace.
    pub trace_label: String,
    /// Expected cumulative reward.
    pub expected_reward: f32,
    /// Actual cumulative reward from replay.
    pub actual_reward: f32,
    /// Difference: expected - actual (positive = regression).
    pub delta: f32,
}

// ── Regression Result ───────────────────────────────────────────

/// Outcome of running a regression suite.
#[derive(Clone, Debug)]
pub struct RegressionResult {
    /// Number of traces that passed.
    pub passed: usize,
    /// Number of traces that failed.
    pub failed: usize,
    /// Details of each failure.
    pub failures: Vec<RegressionFailure>,
}

impl RegressionResult {
    /// Total traces tested.
    pub fn total(&self) -> usize {
        self.passed + self.failed
    }

    /// Whether all traces passed.
    pub fn all_passed(&self) -> bool {
        self.failures.is_empty()
    }
}

// ── Regression Suite ────────────────────────────────────────────

/// Collection of golden traces with tolerance for regression testing.
///
/// Replays each trace through a fresh pruner and checks that actual reward
/// is within `tolerance` of expected. Traces that fall below are flagged
/// as regressions.
pub struct RegressionSuite {
    /// Golden traces to replay.
    pub traces: Vec<GoldenTrace>,
    /// Acceptable deviation from expected reward (e.g., 0.1 = ±10%).
    pub tolerance: f32,
}

impl RegressionSuite {
    /// Create a suite from pre-built golden traces.
    pub fn new(traces: Vec<GoldenTrace>, tolerance: f32) -> Self {
        Self { traces, tolerance }
    }

    /// Extract the top-N best episodes from a trial log as golden traces.
    ///
    /// Reads the JSONL file, sorts episodes by reward (descending),
    /// and takes the top `top_n`. Each trace captures the arm as a
    /// single-action sequence with the observed reward.
    pub fn from_trials(path: &Path, top_n: usize) -> Result<Self> {
        let records = TrialLog::load(path)?;

        let mut ranked: Vec<_> = records.into_iter().collect();
        ranked.sort_by(|a, b| {
            b.reward
                .partial_cmp(&a.reward)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let traces: Vec<GoldenTrace> = ranked
            .into_iter()
            .take(top_n)
            .map(|record| GoldenTrace {
                label: format!(
                    "ep_{episode:04}_arm{arm}",
                    episode = record.episode,
                    arm = record.arm
                ),
                actions: vec![record.arm],
                expected_reward: record.reward,
                expected_survival: record.reward > 0.0,
            })
            .collect();

        Ok(Self {
            traces,
            tolerance: 0.1,
        })
    }

    /// Replay all traces through fresh pruners created by `pruner_factory`.
    ///
    /// The factory receives each trace and returns a pruner + an optional
    /// reward function. The replay pulls each action in the trace and
    /// sums observed rewards, then compares against expected.
    ///
    /// For simple single-arm traces, the reward is the factory's returned value.
    pub fn run<F, R>(&self, pruner_factory: F) -> RegressionResult
    where
        F: Fn(&GoldenTrace) -> R,
        R: ReplayReward,
    {
        let mut passed = 0;
        let mut failures = Vec::new();

        for trace in &self.traces {
            let mut runner = pruner_factory(trace);
            let actual_reward = runner.replay_reward(trace);

            let delta = trace.expected_reward - actual_reward;
            if delta.abs() <= self.tolerance
                || actual_reward >= trace.expected_reward - self.tolerance
            {
                passed += 1;
            } else {
                failures.push(RegressionFailure {
                    trace_label: trace.label.clone(),
                    expected_reward: trace.expected_reward,
                    actual_reward,
                    delta,
                });
            }
        }

        let failed = failures.len();
        RegressionResult {
            passed,
            failed,
            failures,
        }
    }

    /// Number of golden traces in the suite.
    pub fn len(&self) -> usize {
        self.traces.len()
    }

    /// Whether the suite has no traces.
    pub fn is_empty(&self) -> bool {
        self.traces.is_empty()
    }
}

// ── Replay Reward Trait ─────────────────────────────────────────

/// Trait for computing reward during trace replay.
///
/// Implemented by any type that can simulate pulling arms and
/// accumulating rewards. This decouples the regression suite from
/// specific pruner or environment types.
pub trait ReplayReward {
    /// Compute the total reward for replaying the given trace's actions.
    ///
    /// Returns the cumulative reward (sum of per-action rewards).
    fn replay_reward(&mut self, trace: &GoldenTrace) -> f32;
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pruners::trial_log::TrialRecord;

    fn temp_path(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "microgpt_test_regression_{name}_{pid}.jsonl",
            pid = std::process::id()
        ))
    }

    /// Simple replay that returns a fixed reward per action.
    struct FixedReward {
        reward_per_action: f32,
    }

    impl ReplayReward for FixedReward {
        fn replay_reward(&mut self, trace: &GoldenTrace) -> f32 {
            self.reward_per_action * trace.actions.len() as f32
        }
    }

    fn sample_trace(label: &str, arm: usize, reward: f32) -> GoldenTrace {
        GoldenTrace {
            label: label.to_string(),
            actions: vec![arm],
            expected_reward: reward,
            expected_survival: reward > 0.0,
        }
    }

    #[test]
    fn test_all_pass_suite() {
        let suite = RegressionSuite::new(
            vec![
                sample_trace("t1", 0, 0.8),
                sample_trace("t2", 1, 0.9),
                sample_trace("t3", 2, 0.7),
            ],
            0.1,
        );

        let result = suite.run(|_| FixedReward {
            reward_per_action: 0.8,
        });

        // tolerance=0.1: all within ±0.1 of expected
        // t1: 0.8 vs 0.8 → delta 0.0 ≤ 0.1 → pass
        // t2: 0.9 vs 0.8 → delta 0.1 ≤ 0.1 → pass (boundary)
        // t3: 0.7 vs 0.8 → delta 0.1 ≤ 0.1 → pass (boundary)
        assert_eq!(result.passed, 3);
        assert_eq!(result.failed, 0);
        assert!(result.all_passed());
    }

    #[test]
    fn test_tolerance_boundary() {
        let suite = RegressionSuite::new(
            vec![sample_trace("edge", 0, 1.0)],
            0.2, // 20% tolerance
        );

        // Exactly at boundary
        let result = suite.run(|_| FixedReward {
            reward_per_action: 0.8,
        });
        // delta = 1.0 - 0.8 = 0.2 == tolerance, should pass
        assert_eq!(result.passed, 1);
        assert_eq!(result.failed, 0);
        assert!(result.all_passed());
    }

    #[test]
    fn test_empty_suite() {
        let suite = RegressionSuite::new(vec![], 0.1);

        let result = suite.run(|_| FixedReward {
            reward_per_action: 1.0,
        });

        assert_eq!(result.passed, 0);
        assert_eq!(result.failed, 0);
        assert_eq!(result.total(), 0);
        assert!(result.all_passed());
        assert!(suite.is_empty());
    }

    #[test]
    fn test_suite_len() {
        let suite = RegressionSuite::new(
            vec![sample_trace("a", 0, 0.5), sample_trace("b", 1, 0.6)],
            0.1,
        );
        assert_eq!(suite.len(), 2);
        assert!(!suite.is_empty());
    }

    #[test]
    fn test_from_trials_extracts_top_n() {
        let path = temp_path("from_trials");
        let _ = std::fs::remove_file(&path);

        // Write records with varying rewards
        {
            let mut log = crate::pruners::trial_log::TrialLog::new(&path).unwrap();
            for i in 0..10 {
                let reward = (i as f32) * 0.1; // 0.0, 0.1, ..., 0.9
                log.append(&TrialRecord {
                    episode: i,
                    arm: i % 3,
                    reward,
                    q_value: reward,
                    cumulative_reward: reward,
                    cumulative_regret: 1.0 - reward,
                    config: String::new(),
                    note: String::new(),
                    base_correct: None,
                    reviewed_correct: None,
                })
                .unwrap();
            }
            log.flush().unwrap();
        }

        // Extract top 3
        let suite = RegressionSuite::from_trials(&path, 3).unwrap();
        assert_eq!(suite.len(), 3);

        // Top 3 should be rewards 0.9, 0.8, 0.7 (sorted descending)
        let rewards: Vec<f32> = suite.traces.iter().map(|t| t.expected_reward).collect();
        assert!((rewards[0] - 0.9).abs() < 0.01);
        assert!((rewards[1] - 0.8).abs() < 0.01);
        assert!((rewards[2] - 0.7).abs() < 0.01);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_from_trials_empty_file() {
        let path = temp_path("empty_trials");
        let _ = std::fs::remove_file(&path);

        {
            let mut log = crate::pruners::trial_log::TrialLog::new(&path).unwrap();
            log.flush().unwrap();
        }

        let suite = RegressionSuite::from_trials(&path, 5).unwrap();
        assert!(suite.is_empty());

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_failure_details_populated() {
        let suite = RegressionSuite::new(vec![sample_trace("fail_trace", 0, 1.0)], 0.05);

        let result = suite.run(|_| FixedReward {
            reward_per_action: 0.5,
        });

        assert_eq!(result.failed, 1);
        assert_eq!(result.failures.len(), 1);
        assert_eq!(result.failures[0].trace_label, "fail_trace");
        assert!((result.failures[0].expected_reward - 1.0).abs() < 0.01);
        assert!((result.failures[0].actual_reward - 0.5).abs() < 0.01);
        assert!((result.failures[0].delta - 0.5).abs() < 0.01);
    }
}
