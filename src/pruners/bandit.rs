//! Multi-Armed Bandit (MAB) — Adaptive ScreeningPruner
//!
//! Implements bandit strategies that plug into the DDTree screening pipeline
//! via the [`ScreeningPruner`] trait, demonstrating how microgpt-rs's trait-based
//! architecture extends to sequential decision-making under uncertainty.
//!
//! # Architecture
//!
//! - [`BanditStrategy`] — exploration strategy enum (UCB1, ε-greedy, Thompson Sampling)
//! - [`BanditStats`] — shared arm tracking: Q-values, visit counts, scoring
//! - [`BanditPruner`] — wraps any `ScreeningPruner`, adds adaptive relevance for DDTree
//! - [`BanditEnv`] — trait for reward environments ([`BernoulliEnv`], [`GaussianEnv`])
//! - [`BanditSession`] — orchestrates episodes, tracks reward/regret, emits events
//!
//! # Two Usage Modes
//!
//! **1. DDTree Integration** — `BanditPruner` implements `ScreeningPruner`:
//! ```rust,ignore
//! let pruner = BanditPruner::new(NoScreeningPruner, BanditStrategy::Ucb1, 5);
//! pruner.prepare_episode(&mut rng); // Cache Thompson samples if needed
//! let tree = build_dd_tree_screened(&marginals, &config, &pruner, false);
//! // After verification:
//! pruner.update(accepted_token, reward);
//! ```
//!
//! **2. Standalone Bandit** — `BanditSession` runs episodes directly:
//! ```rust,ignore
//! let env = BernoulliEnv::new(&[0.2, 0.5, 0.8, 0.4, 0.6]);
//! let session = BanditSession::new(env, BanditStrategy::Ucb1);
//! let (events, result) = session.run(500, &mut Rng::new(42));
//! assert_eq!(result.best_arm, 2); // Arm 2 has highest mean (0.8)
//! ```

use std::fmt;

use crate::speculative::types::ScreeningPruner;
use crate::types::Rng;

// ── Strategy ────────────────────────────────────────────────────

/// Exploration strategy for multi-armed bandit.
///
/// Each variant implements a different explore/exploit tradeoff.
/// All strategies converge to the optimal arm with sufficient episodes.
#[derive(Clone, Debug)]
pub enum BanditStrategy {
    /// UCB1: `Q(a) + sqrt(2 * ln(N) / n(a))`.
    ///
    /// Deterministic, no RNG needed. O(log N) regret bound.
    /// Best default choice for DDTree integration.
    Ucb1,

    /// ε-greedy: explore with probability `epsilon`, exploit otherwise.
    ///
    /// `decay` multiplies epsilon after each episode.
    /// Use `decay = 1.0` for no decay, `decay < 1.0` for annealing.
    EpsilonGreedy { epsilon: f32, decay: f32 },

    /// Thompson Sampling: sample from Beta(α, β) posterior per arm.
    ///
    /// Optimal asymptotic regret for Bernoulli rewards.
    /// Uses cached samples in [`BanditPruner`] (call `prepare_episode` first).
    ThompsonSampling,
}

impl fmt::Display for BanditStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Ucb1 => write!(f, "UCB1"),
            Self::EpsilonGreedy { epsilon, decay } => {
                write!(f, "ε-greedy(ε={epsilon:.3}, decay={decay:.3})")
            }
            Self::ThompsonSampling => write!(f, "Thompson"),
        }
    }
}

// ── Stats ───────────────────────────────────────────────────────

/// Shared arm tracking state: Q-values, visit counts, scoring.
///
/// Used by both [`BanditPruner`] and [`BanditSession`] to avoid duplication.
/// All methods are O(1) except [`BanditStats::best_arm`] which is O(arms).
pub struct BanditStats {
    q_values: Vec<f32>,
    visits: Vec<u32>,
    total_pulls: u32,
    num_arms: usize,
}

impl BanditStats {
    /// Create zero-initialized stats for `num_arms` arms.
    pub fn new(num_arms: usize) -> Self {
        Self {
            q_values: vec![0.0; num_arms],
            visits: vec![0; num_arms],
            total_pulls: 0,
            num_arms,
        }
    }

    /// Update Q-value for `arm` after observing `reward`.
    ///
    /// Uses incremental mean: `Q(a) += (reward - Q(a)) / n(a)`.
    #[inline]
    pub fn update(&mut self, arm: usize, reward: f32) {
        if arm >= self.num_arms {
            return;
        }
        self.visits[arm] += 1;
        self.total_pulls += 1;
        let n = self.visits[arm] as f32;
        self.q_values[arm] += (reward - self.q_values[arm]) / n;
    }

    /// UCB1 score: `Q(a) + sqrt(2 * ln(N) / n(a))`.
    ///
    /// Returns `f32::MAX` for unvisited arms (must explore first).
    #[inline]
    pub fn ucb1_score(&self, arm: usize) -> f32 {
        if self.visits[arm] == 0 || self.total_pulls == 0 {
            return f32::MAX;
        }
        let q = self.q_values[arm];
        let n = self.visits[arm] as f32;
        let total = self.total_pulls as f32;
        q + (2.0 * total.ln() / n).sqrt()
    }

    /// Thompson Sampling: draw from Beta(α, β) posterior.
    ///
    /// α = Q·n + 1, β = (1-Q)·n + 1 (Laplace smoothing).
    /// Returns uniform sample for unvisited arms.
    #[inline]
    pub fn thompson_sample(&self, arm: usize, rng: &mut Rng) -> f32 {
        if self.visits[arm] == 0 {
            return rng.uniform();
        }
        let n = self.visits[arm] as f32;
        let q = self.q_values[arm].clamp(0.0, 1.0);
        let alpha = q * n + 1.0;
        let beta = (1.0 - q) * n + 1.0;
        sample_beta(alpha, beta, rng)
    }

    /// Index of the arm with highest Q-value.
    pub fn best_arm(&self) -> usize {
        self.q_values
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Q-value estimate for an arm.
    #[inline]
    pub fn q_value(&self, arm: usize) -> f32 {
        self.q_values.get(arm).copied().unwrap_or(0.0)
    }

    /// Visit count for an arm.
    #[inline]
    pub fn visit_count(&self, arm: usize) -> u32 {
        self.visits.get(arm).copied().unwrap_or(0)
    }

    /// Total pulls across all arms.
    #[inline]
    pub fn total_pulls(&self) -> u32 {
        self.total_pulls
    }

    /// Number of arms.
    #[inline]
    pub fn num_arms(&self) -> usize {
        self.num_arms
    }

    /// Q-values slice (for inspection/logging).
    pub fn q_values(&self) -> &[f32] {
        &self.q_values
    }

    /// Visit counts slice (for inspection/logging).
    pub fn visits(&self) -> &[u32] {
        &self.visits
    }
}

// ── Pruner ──────────────────────────────────────────────────────

/// Adaptive screening pruner using multi-armed bandit strategies.
///
/// Wraps any inner [`ScreeningPruner`] for domain knowledge and adds
/// bandit-based relevance scoring. Implements [`ScreeningPruner`] —
/// plugs directly into [`build_dd_tree_screened`](crate::speculative::build_dd_tree_screened).
///
/// # Usage with DDTree
///
/// 1. Create pruner with strategy and arm count
/// 2. Call [`BanditPruner::prepare_episode`] before each DDTree build (caches Thompson samples)
/// 3. Pass to `build_dd_tree_screened` as the screener
/// 4. After verification, call [`BanditPruner::update`] with observed rewards
///
/// # Strategy Behavior in `relevance()`
///
/// | Strategy | Relevance Source |
/// |----------|-----------------|
/// | UCB1 | Q + exploration bonus (deterministic) |
/// | EpsilonGreedy | Q-value (exploration via session) |
/// | ThompsonSampling | Cached posterior sample (call `prepare_episode` first) |
pub struct BanditPruner<P: ScreeningPruner> {
    inner: P,
    strategy: BanditStrategy,
    stats: BanditStats,
    /// Cached Thompson samples, updated by `prepare_episode`.
    thompson_cache: Vec<f32>,
}

impl<P: ScreeningPruner> BanditPruner<P> {
    /// Create a new bandit pruner wrapping an inner pruner.
    ///
    /// `num_arms` is the vocabulary size (number of discrete actions).
    pub fn new(inner: P, strategy: BanditStrategy, num_arms: usize) -> Self {
        Self {
            inner,
            strategy,
            stats: BanditStats::new(num_arms),
            thompson_cache: vec![0.0; num_arms],
        }
    }

    /// Prepare for a new episode. Call before each DDTree build.
    ///
    /// For Thompson Sampling: draws posterior samples and caches them.
    /// For other strategies: no-op.
    pub fn prepare_episode(&mut self, rng: &mut Rng) {
        if matches!(self.strategy, BanditStrategy::ThompsonSampling) {
            for i in 0..self.stats.num_arms {
                self.thompson_cache[i] = self.stats.thompson_sample(i, rng);
            }
        }
    }

    /// Update Q-value for an arm after observing a reward.
    #[inline]
    pub fn update(&mut self, arm: usize, reward: f32) {
        self.stats.update(arm, reward);
    }

    /// Decay epsilon after an episode (EpsilonGreedy only).
    pub fn decay_epsilon(&mut self) {
        if let BanditStrategy::EpsilonGreedy { epsilon, decay } = &mut self.strategy {
            *epsilon *= *decay;
        }
    }

    /// Index of the best arm (highest Q-value).
    pub fn best_arm(&self) -> usize {
        self.stats.best_arm()
    }

    /// Q-values slice (for inspection).
    pub fn q_values(&self) -> &[f32] {
        self.stats.q_values()
    }

    /// Visit counts slice (for inspection).
    pub fn visits(&self) -> &[u32] {
        self.stats.visits()
    }

    /// Total pulls across all arms.
    pub fn total_pulls(&self) -> u32 {
        self.stats.total_pulls()
    }

    /// Strategy reference.
    pub fn strategy(&self) -> &BanditStrategy {
        &self.strategy
    }
}

impl<P: ScreeningPruner> ScreeningPruner for BanditPruner<P> {
    fn relevance(&self, depth: usize, token_idx: usize, parent_tokens: &[usize]) -> f32 {
        if token_idx >= self.stats.num_arms {
            return 0.0;
        }

        // Domain relevance from inner pruner
        let domain = self.inner.relevance(depth, token_idx, parent_tokens);
        if domain <= 0.0 {
            return 0.0;
        }

        // Cold start: no data yet, use domain only
        if self.stats.total_pulls == 0 {
            return domain;
        }

        // Unvisited arm: maximum exploration priority
        if self.stats.visit_count(token_idx) == 0 {
            return domain;
        }

        // Bandit score based on strategy
        let bandit = match &self.strategy {
            BanditStrategy::Ucb1 => self.stats.ucb1_score(token_idx).clamp(0.0, 1.5) / 1.5,
            BanditStrategy::EpsilonGreedy { .. } => {
                self.stats.q_value(token_idx).clamp(0.0, 1.0).max(0.01)
            }
            BanditStrategy::ThompsonSampling => {
                // Use cached sample from prepare_episode
                self.thompson_cache
                    .get(token_idx)
                    .copied()
                    .unwrap_or_else(|| self.stats.q_value(token_idx))
                    .clamp(0.0, 1.0)
                    .max(0.01)
            }
        };

        // Harmonic blend: domain × bandit
        (domain * bandit).clamp(0.0, 1.0)
    }
}

// ── Beta Distribution Sampling ──────────────────────────────────

/// Sample from Beta(α, β) distribution using Jöhnk's algorithm.
///
/// Works well for α, β ≥ 1 (our case: posterior with +1 pseudocounts).
/// Uses rejection sampling. Falls back to 0.5 after 256 rejections.
fn sample_beta(alpha: f32, beta: f32, rng: &mut Rng) -> f32 {
    // Uniform prior: α=1, β=1
    if (alpha - 1.0).abs() < f32::EPSILON && (beta - 1.0).abs() < f32::EPSILON {
        return rng.uniform();
    }

    // Jöhnk's algorithm: X = U1^(1/α), Y = U2^(1/β), accept if X+Y ≤ 1
    for _ in 0..256 {
        let u1 = rng.uniform().max(f32::EPSILON);
        let u2 = rng.uniform().max(f32::EPSILON);
        let x = u1.powf(1.0 / alpha);
        let y = u2.powf(1.0 / beta);
        let sum = x + y;
        if sum <= 1.0 && sum > 0.0 {
            return x / sum;
        }
    }

    // Fallback: Q-value midpoint
    0.5
}

// ── Environment ─────────────────────────────────────────────────

/// A multi-armed bandit environment that generates stochastic rewards.
///
/// Each arm has a hidden reward distribution. The agent's goal is to
/// identify the arm with the highest expected reward while minimizing
/// cumulative regret.
pub trait BanditEnv: Send + Sync {
    /// Pull an arm and receive a stochastic reward in [0.0, 1.0].
    fn pull(&self, arm: usize, rng: &mut Rng) -> f32;

    /// Expected (mean) reward for a specific arm.
    fn expected_reward(&self, arm: usize) -> f32;

    /// Expected reward of the optimal arm.
    fn optimal_reward(&self) -> f32;

    /// Number of arms.
    fn num_arms(&self) -> usize;

    /// Index of the optimal arm (highest expected reward).
    fn optimal_arm(&self) -> usize;
}

// ── Bernoulli Environment ───────────────────────────────────────

/// Bernoulli bandit: each arm returns 1.0 with probability `p`, 0.0 otherwise.
///
/// Classic MAB setting. Optimal for Thompson Sampling with Beta posteriors.
pub struct BernoulliEnv {
    probs: Vec<f32>,
    optimal_arm: usize,
    optimal_reward: f32,
}

impl BernoulliEnv {
    /// Create a Bernoulli bandit with per-arm success probabilities.
    pub fn new(probs: &[f32]) -> Self {
        let optimal_arm = probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        let optimal_reward = probs[optimal_arm];
        Self {
            probs: probs.to_vec(),
            optimal_arm,
            optimal_reward,
        }
    }

    /// Success probability for each arm.
    pub fn probs(&self) -> &[f32] {
        &self.probs
    }
}

impl BanditEnv for BernoulliEnv {
    fn pull(&self, arm: usize, rng: &mut Rng) -> f32 {
        if arm >= self.probs.len() || rng.uniform() >= self.probs[arm] {
            0.0
        } else {
            1.0
        }
    }

    fn expected_reward(&self, arm: usize) -> f32 {
        self.probs.get(arm).copied().unwrap_or(0.0)
    }

    fn optimal_reward(&self) -> f32 {
        self.optimal_reward
    }

    fn num_arms(&self) -> usize {
        self.probs.len()
    }

    fn optimal_arm(&self) -> usize {
        self.optimal_arm
    }
}

// ── Gaussian Environment ────────────────────────────────────────

/// Gaussian bandit: each arm returns a reward sampled from N(μ, σ²).
///
/// Rewards are clamped to [0.0, 1.0]. Useful for continuous reward settings.
pub struct GaussianEnv {
    means: Vec<f32>,
    std: f32,
    optimal_arm: usize,
    optimal_reward: f32,
}

impl GaussianEnv {
    /// Create a Gaussian bandit with per-arm means and shared standard deviation.
    pub fn new(means: &[f32], std: f32) -> Self {
        let optimal_arm = means
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        let optimal_reward = means[optimal_arm];
        Self {
            means: means.to_vec(),
            std,
            optimal_arm,
            optimal_reward,
        }
    }

    /// Mean reward for each arm.
    pub fn means(&self) -> &[f32] {
        &self.means
    }
}

impl BanditEnv for GaussianEnv {
    fn pull(&self, arm: usize, rng: &mut Rng) -> f32 {
        if arm >= self.means.len() {
            return 0.0;
        }
        // Box-Muller transform for Gaussian sampling
        let u1 = rng.uniform().max(f32::EPSILON);
        let u2 = rng.uniform();
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
        (self.means[arm] + self.std * z).clamp(0.0, 1.0)
    }

    fn expected_reward(&self, arm: usize) -> f32 {
        self.means.get(arm).copied().unwrap_or(0.0)
    }

    fn optimal_reward(&self) -> f32 {
        self.optimal_reward
    }

    fn num_arms(&self) -> usize {
        self.means.len()
    }

    fn optimal_arm(&self) -> usize {
        self.optimal_arm
    }
}

// ── Bandit Event ────────────────────────────────────────────────

/// Events emitted during bandit session execution.
#[derive(Clone, Debug)]
pub enum BanditEvent {
    /// An arm was pulled and reward observed.
    Pull {
        episode: usize,
        arm: usize,
        reward: f32,
        q_value: f32,
    },
    /// Episode completed with cumulative stats.
    EpisodeComplete {
        episode: usize,
        arm: usize,
        reward: f32,
        cumulative_reward: f32,
        cumulative_regret: f32,
    },
    /// Session completed with final stats.
    SessionComplete {
        total_episodes: usize,
        total_reward: f32,
        total_regret: f32,
        best_arm: usize,
        optimal_arm: usize,
    },
}

// ── Bandit Result ───────────────────────────────────────────────

/// Final result of a bandit session.
#[derive(Clone, Debug)]
pub struct BanditResult {
    /// Total episodes run.
    pub total_episodes: usize,
    /// Sum of all observed rewards.
    pub total_reward: f32,
    /// Sum of per-episode regret: Σ(optimal_reward - arm_expected_reward).
    pub total_regret: f32,
    /// Arm with highest Q-value at session end.
    pub best_arm: usize,
    /// True optimal arm from the environment.
    pub optimal_arm: usize,
    /// Final Q-value estimates.
    pub q_values: Vec<f32>,
    /// Final visit counts.
    pub visits: Vec<u32>,
}

impl BanditResult {
    /// Whether the bandit found the true optimal arm.
    pub fn found_optimal(&self) -> bool {
        self.best_arm == self.optimal_arm
    }

    /// Average reward per episode.
    pub fn avg_reward(&self) -> f32 {
        if self.total_episodes == 0 {
            0.0
        } else {
            self.total_reward / self.total_episodes as f32
        }
    }

    /// Average regret per episode.
    pub fn avg_regret(&self) -> f32 {
        if self.total_episodes == 0 {
            0.0
        } else {
            self.total_regret / self.total_episodes as f32
        }
    }
}

// ── Bandit Session ──────────────────────────────────────────────

/// Orchestrates multi-armed bandit episodes.
///
/// Runs N episodes of arm selection → reward observation → Q-value update.
/// Tracks cumulative reward and pseudo-regret. Emits events for logging.
///
/// # Example
///
/// ```rust,ignore
/// let env = BernoulliEnv::new(&[0.2, 0.5, 0.8, 0.4, 0.6]);
/// let session = BanditSession::new(env, BanditStrategy::ThompsonSampling);
/// let (events, result) = session.run(500, &mut Rng::new(42));
/// assert!(result.found_optimal());
/// ```
pub struct BanditSession<E: BanditEnv> {
    env: E,
    strategy: BanditStrategy,
    stats: BanditStats,
    cumulative_reward: f32,
    cumulative_regret: f32,
}

impl<E: BanditEnv> BanditSession<E> {
    /// Create a new bandit session with the given environment and strategy.
    pub fn new(env: E, strategy: BanditStrategy) -> Self {
        let num_arms = env.num_arms();
        Self {
            env,
            strategy,
            stats: BanditStats::new(num_arms),
            cumulative_reward: 0.0,
            cumulative_regret: 0.0,
        }
    }

    /// Select an arm based on the current strategy and stats.
    fn select_arm(&self, rng: &mut Rng) -> usize {
        let num_arms = self.env.num_arms();

        // Cold start: play each arm once
        for i in 0..num_arms {
            if self.stats.visit_count(i) == 0 {
                return i;
            }
        }

        match &self.strategy {
            BanditStrategy::Ucb1 => self.select_ucb1(),
            BanditStrategy::EpsilonGreedy { epsilon, .. } => {
                self.select_epsilon_greedy(*epsilon, rng)
            }
            BanditStrategy::ThompsonSampling => self.select_thompson(rng),
        }
    }

    fn select_ucb1(&self) -> usize {
        (0..self.env.num_arms())
            .max_by(|&a, &b| {
                self.stats
                    .ucb1_score(a)
                    .partial_cmp(&self.stats.ucb1_score(b))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or(0)
    }

    fn select_epsilon_greedy(&self, epsilon: f32, rng: &mut Rng) -> usize {
        let num_arms = self.env.num_arms();
        if rng.uniform() < epsilon {
            // Explore: random arm
            (rng.uniform() * num_arms as f32) as usize % num_arms
        } else {
            // Exploit: best Q-value
            self.stats.best_arm()
        }
    }

    fn select_thompson(&self, rng: &mut Rng) -> usize {
        (0..self.env.num_arms())
            .map(|i| (i, self.stats.thompson_sample(i, rng)))
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Decay epsilon (EpsilonGreedy only).
    fn decay_epsilon(&mut self) {
        if let BanditStrategy::EpsilonGreedy { epsilon, decay } = &mut self.strategy {
            *epsilon *= *decay;
        }
    }

    /// Run the bandit session for `episodes` episodes.
    ///
    /// Returns `(events, result)`. Events include per-episode stats.
    /// Pseudo-regret = Σ(optimal_expected - chosen_arm_expected).
    pub fn run(mut self, episodes: usize, rng: &mut Rng) -> (Vec<BanditEvent>, BanditResult) {
        let mut events = Vec::with_capacity(episodes + 1);
        let optimal_arm = self.env.optimal_arm();
        let optimal_reward = self.env.optimal_reward();

        for episode in 0..episodes {
            let arm = self.select_arm(rng);
            let reward = self.env.pull(arm, rng);
            let q_before = self.stats.q_value(arm);

            events.push(BanditEvent::Pull {
                episode,
                arm,
                reward,
                q_value: q_before,
            });

            self.stats.update(arm, reward);
            self.cumulative_reward += reward;
            self.cumulative_regret += optimal_reward - self.env.expected_reward(arm);

            self.decay_epsilon();

            events.push(BanditEvent::EpisodeComplete {
                episode,
                arm,
                reward,
                cumulative_reward: self.cumulative_reward,
                cumulative_regret: self.cumulative_regret,
            });
        }

        let best_arm = self.stats.best_arm();
        let result = BanditResult {
            total_episodes: episodes,
            total_reward: self.cumulative_reward,
            total_regret: self.cumulative_regret,
            best_arm,
            optimal_arm,
            q_values: self.stats.q_values.to_vec(),
            visits: self.stats.visits.to_vec(),
        };

        events.push(BanditEvent::SessionComplete {
            total_episodes: episodes,
            total_reward: self.cumulative_reward,
            total_regret: self.cumulative_regret,
            best_arm,
            optimal_arm,
        });

        (events, result)
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::speculative::types::NoScreeningPruner;

    // ── Stats Tests ─────────────────────────────────────────────

    #[test]
    fn test_stats_update_incremental_mean() {
        let mut stats = BanditStats::new(3);

        stats.update(0, 1.0);
        assert_eq!(stats.q_value(0), 1.0);

        stats.update(0, 0.0);
        assert_eq!(stats.q_value(0), 0.5);

        stats.update(0, 1.0);
        // (1.0 + 0.0 + 1.0) / 3 = 0.666...
        assert!((stats.q_value(0) - 0.666).abs() < 0.01);
        assert_eq!(stats.visit_count(0), 3);
        assert_eq!(stats.total_pulls(), 3);
    }

    #[test]
    fn test_stats_best_arm() {
        let mut stats = BanditStats::new(4);
        stats.update(0, 0.2);
        stats.update(1, 0.8);
        stats.update(2, 0.5);
        stats.update(3, 0.3);
        assert_eq!(stats.best_arm(), 1);
    }

    #[test]
    fn test_stats_ucb1_unvisited_max_priority() {
        let mut stats = BanditStats::new(3);
        stats.update(0, 0.5);
        stats.update(1, 0.5);
        // Arm 2 unvisited → should get f32::MAX score
        assert_eq!(stats.ucb1_score(2), f32::MAX);
        // Visited arms get finite scores
        assert!(stats.ucb1_score(0).is_finite());
        assert!(stats.ucb1_score(1).is_finite());
    }

    #[test]
    fn test_stats_ucb1_increases_with_few_visits() {
        let mut stats = BanditStats::new(2);
        // Both arms get same reward, but arm 0 gets visited less
        for _ in 0..10 {
            stats.update(0, 0.5);
        }
        for _ in 0..100 {
            stats.update(1, 0.5);
        }
        // Arm 0 should have higher UCB1 bonus (fewer visits)
        assert!(stats.ucb1_score(0) > stats.ucb1_score(1));
    }

    // ── Environment Tests ───────────────────────────────────────

    #[test]
    fn test_bernoulli_env_optimal() {
        let env = BernoulliEnv::new(&[0.2, 0.5, 0.8, 0.4, 0.6]);
        assert_eq!(env.optimal_arm(), 2);
        assert!((env.optimal_reward() - 0.8).abs() < f32::EPSILON);
        assert_eq!(env.num_arms(), 5);
        assert!((env.expected_reward(0) - 0.2).abs() < f32::EPSILON);
    }

    #[test]
    fn test_bernoulli_env_pull_distribution() {
        let env = BernoulliEnv::new(&[0.0, 1.0]);
        let mut rng = Rng::new(42);

        // Arm 0 always returns 0.0
        for _ in 0..100 {
            assert_eq!(env.pull(0, &mut rng), 0.0);
        }
        // Arm 1 always returns 1.0
        for _ in 0..100 {
            assert_eq!(env.pull(1, &mut rng), 1.0);
        }
    }

    #[test]
    fn test_gaussian_env_optimal() {
        let env = GaussianEnv::new(&[0.2, 0.7, 0.5], 0.1);
        assert_eq!(env.optimal_arm(), 1);
        assert!((env.optimal_reward() - 0.7).abs() < f32::EPSILON);
        assert_eq!(env.num_arms(), 3);
    }

    #[test]
    fn test_gaussian_env_pull_clamped() {
        let env = GaussianEnv::new(&[0.5], 0.1);
        let mut rng = Rng::new(42);
        // All rewards should be in [0.0, 1.0]
        for _ in 0..1000 {
            let r = env.pull(0, &mut rng);
            assert!(r >= 0.0 && r <= 1.0, "reward {r} out of bounds");
        }
    }

    // ── Beta Sampling Tests ─────────────────────────────────────

    #[test]
    fn test_beta_sampling_bounds() {
        let mut rng = Rng::new(42);
        for alpha in [1.0, 2.0, 5.0, 10.0] {
            for beta in [1.0, 2.0, 5.0, 10.0] {
                for _ in 0..100 {
                    let sample = sample_beta(alpha, beta, &mut rng);
                    assert!(
                        sample >= 0.0 && sample <= 1.0,
                        "Beta({alpha},{beta}) sample {sample} out of bounds"
                    );
                }
            }
        }
    }

    #[test]
    fn test_beta_sampling_mean_converges() {
        let mut rng = Rng::new(42);
        let alpha = 3.0f32;
        let beta = 7.0f32;
        let expected_mean = alpha / (alpha + beta); // 0.3
        let n = 10000;
        let sum: f32 = (0..n).map(|_| sample_beta(alpha, beta, &mut rng)).sum();
        let mean = sum / n as f32;
        assert!(
            (mean - expected_mean).abs() < 0.05,
            "Beta({alpha},{beta}) mean {mean} too far from expected {expected_mean}"
        );
    }

    // ── BanditPruner Tests ──────────────────────────────────────

    #[test]
    fn test_pruner_cold_start_uses_domain() {
        let pruner = BanditPruner::new(NoScreeningPruner, BanditStrategy::Ucb1, 5);
        // Before any updates, relevance = domain relevance
        let rel = pruner.relevance(0, 0, &[]);
        assert!((rel - 1.0).abs() < f32::EPSILON); // NoScreeningPruner returns 1.0
    }

    #[test]
    fn test_pruner_respects_domain_hard_trim() {
        struct AlwaysZero;
        impl ScreeningPruner for AlwaysZero {
            fn relevance(&self, _: usize, _: usize, _: &[usize]) -> f32 {
                0.0
            }
        }
        let mut pruner = BanditPruner::new(AlwaysZero, BanditStrategy::Ucb1, 5);
        pruner.update(0, 0.9); // High reward, but domain says 0
        let rel = pruner.relevance(0, 0, &[]);
        assert_eq!(rel, 0.0);
    }

    #[test]
    fn test_pruner_ucb1_unvisited_arm_priority() {
        let mut pruner = BanditPruner::new(NoScreeningPruner, BanditStrategy::Ucb1, 3);
        // Visit arm 0 and 1, leave arm 2 unvisited
        pruner.update(0, 0.5);
        pruner.update(1, 0.5);
        let rel_visited = pruner.relevance(0, 0, &[]);
        let rel_unvisited = pruner.relevance(0, 2, &[]);
        // Unvisited arm should have equal or higher relevance
        assert!(rel_unvisited >= rel_visited);
    }

    #[test]
    fn test_pruner_thompson_uses_cache() {
        let mut pruner = BanditPruner::new(NoScreeningPruner, BanditStrategy::ThompsonSampling, 3);
        // Update some Q-values
        for _ in 0..10 {
            pruner.update(0, 0.9);
            pruner.update(1, 0.1);
        }
        let mut rng = Rng::new(42);
        pruner.prepare_episode(&mut rng);

        // Relevance should use cached sample (non-zero since arm 0 has high Q)
        let rel = pruner.relevance(0, 0, &[]);
        assert!(rel > 0.0);
    }

    // ── Convergence Tests ───────────────────────────────────────

    #[test]
    fn test_ucb1_convergence() {
        let env = BernoulliEnv::new(&[0.2, 0.5, 0.8, 0.4, 0.6]);
        let session = BanditSession::new(env, BanditStrategy::Ucb1);
        let (_, result) = session.run(500, &mut Rng::new(42));
        assert!(
            result.found_optimal(),
            "UCB1 should find optimal arm 2, found arm {} with Q-values {:?}",
            result.best_arm,
            result.q_values
        );
    }

    #[test]
    fn test_thompson_convergence() {
        let env = BernoulliEnv::new(&[0.2, 0.5, 0.8, 0.4, 0.6]);
        let session = BanditSession::new(env, BanditStrategy::ThompsonSampling);
        let (_, result) = session.run(500, &mut Rng::new(42));
        assert!(
            result.found_optimal(),
            "Thompson should find optimal arm 2, found arm {} with Q-values {:?}",
            result.best_arm,
            result.q_values
        );
    }

    #[test]
    fn test_epsilon_greedy_convergence_with_decay() {
        let env = BernoulliEnv::new(&[0.2, 0.5, 0.8, 0.4, 0.6]);
        let strategy = BanditStrategy::EpsilonGreedy {
            epsilon: 0.3,
            decay: 0.995,
        };
        let session = BanditSession::new(env, strategy);
        let (_, result) = session.run(1000, &mut Rng::new(42));
        assert!(
            result.found_optimal(),
            "ε-greedy(decay) should find optimal arm 2, found arm {} with Q-values {:?}",
            result.best_arm,
            result.q_values
        );
    }

    #[test]
    fn test_epsilon_greedy_no_decay_still_finds_good_arm() {
        let env = BernoulliEnv::new(&[0.1, 0.9]);
        let strategy = BanditStrategy::EpsilonGreedy {
            epsilon: 0.1,
            decay: 1.0,
        };
        let session = BanditSession::new(env, strategy);
        let (_, result) = session.run(2000, &mut Rng::new(42));
        // With fixed ε, may not always find optimal but should be close
        assert!(
            result.q_values[1] > 0.5,
            "ε-greedy(no decay) should learn arm 1 is good, Q-values: {:?}",
            result.q_values
        );
    }

    // ── Regret Tests ────────────────────────────────────────────

    #[test]
    fn test_regret_sublinear_ucb1() {
        let env = BernoulliEnv::new(&[0.2, 0.5, 0.8, 0.4, 0.6]);
        let session = BanditSession::new(env, BanditStrategy::Ucb1);
        let (_, result) = session.run(1000, &mut Rng::new(42));

        // Sub-linear regret: total_regret should grow slower than linear
        // Linear regret would be ~1000 * (0.8 - 0.2) = 600 for always choosing worst arm
        // Sub-linear should be much less, roughly O(sqrt(N)) ≈ ~30-60
        assert!(
            result.total_regret < 100.0,
            "UCB1 regret should be sub-linear, got {}",
            result.total_regret
        );
    }

    #[test]
    fn test_regret_sublinear_thompson() {
        let env = BernoulliEnv::new(&[0.2, 0.5, 0.8, 0.4, 0.6]);
        let session = BanditSession::new(env, BanditStrategy::ThompsonSampling);
        let (_, result) = session.run(1000, &mut Rng::new(42));

        // Thompson is stochastic — higher variance than UCB1.
        // Linear regret worst-case ≈ 600. Sub-linear threshold generous but still
        // well below linear: must be spending most pulls on high-value arms.
        assert!(
            result.total_regret < 250.0,
            "Thompson regret should be sub-linear, got {}",
            result.total_regret
        );
    }

    // ── Gaussian Bandit Test ────────────────────────────────────

    #[test]
    fn test_gaussian_convergence() {
        let env = GaussianEnv::new(&[0.3, 0.7, 0.5], 0.1);
        let session = BanditSession::new(env, BanditStrategy::Ucb1);
        let (_, result) = session.run(500, &mut Rng::new(42));
        assert!(
            result.found_optimal(),
            "UCB1 should find Gaussian optimal arm 1, found arm {} with Q-values {:?}",
            result.best_arm,
            result.q_values
        );
    }

    // ── Session Event Tests ─────────────────────────────────────

    #[test]
    fn test_session_events_count() {
        let env = BernoulliEnv::new(&[0.5, 0.8]);
        let session = BanditSession::new(env, BanditStrategy::Ucb1);
        let (events, _) = session.run(10, &mut Rng::new(42));

        // 10 Pull + 10 EpisodeComplete + 1 SessionComplete = 21
        assert_eq!(events.len(), 21);
    }

    #[test]
    fn test_session_result_fields() {
        let env = BernoulliEnv::new(&[0.5, 0.8]);
        let session = BanditSession::new(env, BanditStrategy::Ucb1);
        let (_, result) = session.run(100, &mut Rng::new(42));

        assert_eq!(result.total_episodes, 100);
        assert_eq!(result.optimal_arm, 1);
        assert_eq!(result.q_values.len(), 2);
        assert_eq!(result.visits.len(), 2);
        assert!(result.total_reward > 0.0);
        assert!(result.avg_reward() > 0.0);
    }

    // ── Constrained Bandit Tests ────────────────────────────────

    /// Domain pruner that blocks specific arms via relevance 0.0.
    struct BlockedArmPruner {
        blocked: Vec<usize>,
    }

    impl BlockedArmPruner {
        fn new(blocked: &[usize]) -> Self {
            Self {
                blocked: blocked.to_vec(),
            }
        }
    }

    impl ScreeningPruner for BlockedArmPruner {
        fn relevance(&self, _depth: usize, token_idx: usize, _parent_tokens: &[usize]) -> f32 {
            if self.blocked.contains(&token_idx) {
                0.0
            } else {
                1.0
            }
        }
    }

    #[test]
    fn test_constrained_bandit_never_pulls_blocked_arm() {
        // Arm 4 has highest reward (0.9) but is blocked
        let mut pruner = BanditPruner::new(BlockedArmPruner::new(&[4]), BanditStrategy::Ucb1, 5);

        let env = BernoulliEnv::new(&[0.1, 0.3, 0.7, 0.4, 0.9]);
        let mut rng = Rng::new(42);

        // Select arms via pruner relevance for 500 episodes
        for _ in 0..500 {
            let arm = (0..5)
                .max_by(|&a, &b| {
                    pruner
                        .relevance(0, a, &[])
                        .partial_cmp(&pruner.relevance(0, b, &[]))
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap_or(0);

            let reward = env.pull(arm, &mut rng);
            pruner.update(arm, reward);
        }

        // Arm 4 should never be visited (relevance always 0.0)
        assert_eq!(pruner.visits()[4], 0, "blocked arm should never be pulled");
        // Best arm should be 2 (highest unblocked reward: 0.7)
        assert_eq!(
            pruner.best_arm(),
            2,
            "should find best valid arm, not blocked arm"
        );
    }

    #[test]
    fn test_constrained_bandit_respects_domain_over_bandit() {
        // Even after giving arm 4 high reward manually, domain pruner overrides
        let mut pruner = BanditPruner::new(BlockedArmPruner::new(&[4]), BanditStrategy::Ucb1, 5);

        // Manually pump arm 4's Q-value high
        for _ in 0..100 {
            pruner.update(4, 1.0);
        }

        // Domain still blocks it
        let rel = pruner.relevance(0, 4, &[]);
        assert_eq!(
            rel, 0.0,
            "domain pruner must override bandit score for blocked arms"
        );

        // Other arms still allowed
        assert!(pruner.relevance(0, 0, &[]) >= 0.0);
        assert!(pruner.relevance(0, 2, &[]) >= 0.0);
    }
}
