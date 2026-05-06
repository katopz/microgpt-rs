use crate::speculative::dd_tree::{build_dd_tree, extract_best_path};
use crate::speculative::dflash::dflash_predict;
use crate::speculative::sampling::sample_from_distribution;
use crate::transformer::TransformerWeights;
use crate::types::{Config, Rng};

#[cfg(feature = "leviathan")]
use crate::speculative::dflash::dflash_predict_ar;
#[cfg(feature = "leviathan")]
use crate::speculative::sampling::sample_residual_distribution;
#[cfg(feature = "leviathan")]
use crate::transformer::{ForwardContext, KVCache, forward};
#[cfg(feature = "leviathan")]
use crate::types::softmax;

// ── Speculative Verifier: Strategy Pattern ──────────────────

/// Strategy for verifying drafted tokens against a target distribution.
///
/// Same pattern as `ConstraintPruner` — trait-based swap point.
/// - `SimulatedVerifier`: fast, no target model needed (default).
/// - `LeviathanVerifier`: real p/q rejection sampling with target model
///   (behind `leviathan` feature flag).
pub trait SpeculativeVerifier: Send + Sync {
    /// Run one speculative decoding step end-to-end.
    /// Returns accepted tokens (always ≥ 1, up to γ + 1 with bonus).
    fn speculate(
        &mut self,
        draft_weights: &TransformerWeights,
        draft_config: &Config,
        token: usize,
        pos: usize,
        rng: &mut Rng,
    ) -> Vec<usize>;
}

/// Simulated verification: DDTree path + acceptance cap + bonus token.
/// No target model needed — fast, used by default.
pub struct SimulatedVerifier {
    pub acceptance_rate: f32,
}

impl SimulatedVerifier {
    pub fn new(acceptance_rate: f32) -> Self {
        Self {
            acceptance_rate: acceptance_rate.clamp(0.0, 1.0),
        }
    }
}

impl SpeculativeVerifier for SimulatedVerifier {
    fn speculate(
        &mut self,
        draft_weights: &TransformerWeights,
        draft_config: &Config,
        token: usize,
        pos: usize,
        rng: &mut Rng,
    ) -> Vec<usize> {
        // 1. Sequential DFlash draft (avoids rayon overhead for tiny model)
        let marginals = dflash_predict(draft_weights, draft_config, token, pos);

        // 2. DDTree build
        let tree = build_dd_tree(&marginals, draft_config);

        // 3. Extract best path (highest-scored token at each depth)
        let path = extract_best_path(&tree);

        if path.is_empty() {
            return vec![sample_from_distribution(
                marginals.first().map(|m| m.as_slice()).unwrap_or(&[1.0]),
                rng,
            )];
        }

        // 4. Simulate acceptance: cap at rate
        let max_accept = ((path.len() as f32) * self.acceptance_rate).ceil() as usize;
        let accepted: Vec<usize> = path.into_iter().take(max_accept.max(1)).collect();

        // 5. Bonus token: if all accepted, sample +1 from last marginal
        if accepted.len() == max_accept && !marginals.is_empty() {
            let last_marginal = marginals.last().unwrap();
            let bonus = sample_from_distribution(last_marginal, rng);
            let mut result = accepted;
            result.push(bonus);
            return result;
        }

        accepted
    }
}

// ── LeviathanVerifier: Real p/q rejection sampling (Algorithm 1) ──

#[cfg(feature = "leviathan")]
pub struct LeviathanVerifier<'a> {
    pub target_weights: &'a TransformerWeights,
    pub target_config: &'a Config,
    target_ctx: ForwardContext,
    target_cache: KVCache,
}

#[cfg(feature = "leviathan")]
impl<'a> LeviathanVerifier<'a> {
    pub fn new(target_weights: &'a TransformerWeights, target_config: &'a Config) -> Self {
        Self {
            target_weights,
            target_config,
            target_ctx: ForwardContext::new(target_config),
            target_cache: KVCache::new(target_config),
        }
    }
}

#[cfg(feature = "leviathan")]
impl SpeculativeVerifier for LeviathanVerifier<'_> {
    fn speculate(
        &mut self,
        draft_weights: &TransformerWeights,
        draft_config: &Config,
        token: usize,
        pos: usize,
        rng: &mut Rng,
    ) -> Vec<usize> {
        // Phase 1: Autoregressive draft (Algorithm 1, line 2–5)
        let draft_result = dflash_predict_ar(draft_weights, draft_config, token, pos, rng);
        let draft_tokens = &draft_result.sampled_tokens;
        let q_dists = &draft_result.marginals;
        let gamma = draft_tokens.len();

        if gamma == 0 {
            // No draft tokens — run target once, return 1 token
            self.target_cache.reset();
            let logits = forward(
                &mut self.target_ctx,
                self.target_weights,
                &mut self.target_cache,
                token,
                pos,
                self.target_config,
            );
            for logit in logits.iter_mut() {
                *logit /= self.target_config.temperature;
            }
            softmax(logits);
            return vec![sample_from_distribution(logits, rng)];
        }

        // Phase 2: Target scoring (Algorithm 1, line 7–8)
        self.target_cache.reset();
        let mut p_distributions: Vec<Vec<f32>> = Vec::with_capacity(gamma + 1);

        // Score the initial token → p(x) at position 0
        let logits = forward(
            &mut self.target_ctx,
            self.target_weights,
            &mut self.target_cache,
            token,
            pos,
            self.target_config,
        );
        for logit in logits.iter_mut() {
            *logit /= self.target_config.temperature;
        }
        softmax(logits);
        p_distributions.push(logits.to_vec());

        // Score each drafted token → p(x) at positions 1..=gamma
        for (i, &draft_tok) in draft_tokens.iter().enumerate() {
            let logits = forward(
                &mut self.target_ctx,
                self.target_weights,
                &mut self.target_cache,
                draft_tok,
                pos + 1 + i,
                self.target_config,
            );
            for logit in logits.iter_mut() {
                *logit /= self.target_config.temperature;
            }
            softmax(logits);
            p_distributions.push(logits.to_vec());
        }

        // Phase 3: Rejection sampling (Algorithm 1, line 10–16)
        let mut accepted = Vec::with_capacity(gamma + 1);
        let mut all_accepted = true;

        for i in 0..gamma {
            let p_dist = &p_distributions[i];
            let q_dist = &q_dists[i];
            let drafted_token = draft_tokens[i];

            let p_i = p_dist[drafted_token];
            let q_i = q_dist[drafted_token];

            // Accept with prob min(1, p/q)
            let acceptance_prob = if q_i > 0.0 {
                (p_i / q_i).min(1.0)
            } else {
                1.0 // q=0 means draft didn't propose this; accept if target likes it
            };
            let r = rng.uniform();

            if r <= acceptance_prob {
                accepted.push(drafted_token);
            } else {
                // Reject: sample replacement from residual max(0, p - q)
                let replacement = sample_residual_distribution(p_dist, q_dist, rng);
                accepted.push(replacement);
                all_accepted = false;
                break;
            }
        }

        // Phase 4: Bonus token (Algorithm 1, line 18–19)
        if all_accepted && p_distributions.len() > gamma {
            let bonus = sample_from_distribution(&p_distributions[gamma], rng);
            accepted.push(bonus);
        }

        // Safety: always return at least 1 token
        if accepted.is_empty() {
            accepted.push(sample_from_distribution(&p_distributions[0], rng));
        }

        accepted
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transformer::TransformerWeights;
    use crate::types::{Config, Rng};

    fn make_draft() -> (TransformerWeights, Config) {
        let config = Config::draft();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        (weights, config)
    }

    // ── SimulatedVerifier Tests ───────────────────────────────────

    #[test]
    fn test_simulated_verifier_returns_at_least_one() {
        let (weights, config) = make_draft();
        let mut verifier = SimulatedVerifier::new(0.75);
        let mut rng = Rng::new(42);
        let accepted = verifier.speculate(&weights, &config, 0, 0, &mut rng);
        assert!(!accepted.is_empty(), "should return at least 1 token");
        assert!(!accepted.is_empty());
        for &t in &accepted {
            assert!(t < config.vocab_size, "token {t} out of range");
        }
    }

    #[test]
    fn test_simulated_verifier_deterministic() {
        let (weights, config) = make_draft();

        let a1 = {
            let mut verifier = SimulatedVerifier::new(0.75);
            verifier.speculate(&weights, &config, 0, 0, &mut Rng::new(77))
        };
        let a2 = {
            let mut verifier = SimulatedVerifier::new(0.75);
            verifier.speculate(&weights, &config, 0, 0, &mut Rng::new(77))
        };

        assert_eq!(a1, a2, "same seed should produce same accepted tokens");
    }

    #[test]
    fn test_simulated_verifier_bonus_token() {
        let (weights, config) = make_draft();
        let mut saw_bonus = false;
        for seed in 0..200u64 {
            let mut verifier = SimulatedVerifier::new(0.95);
            let accepted = verifier.speculate(&weights, &config, 0, 0, &mut Rng::new(seed));
            if accepted.len() > 1 {
                saw_bonus = true;
                break;
            }
        }
        assert!(
            saw_bonus,
            "should see bonus token at least once with high acceptance rate"
        );
    }

    // ── LeviathanVerifier Tests (feature-gated) ───────────────

    #[cfg(feature = "leviathan")]
    #[test]
    fn test_leviathan_verifier_returns_at_least_one() {
        let config = Config::micro();
        let draft_config = Config::draft();
        let mut rng = Rng::new(42);
        let target_weights = TransformerWeights::new(&config, &mut rng);
        let mut draft_rng = Rng::new(99);
        let draft_weights = TransformerWeights::new(&draft_config, &mut draft_rng);

        let mut verifier = LeviathanVerifier::new(&target_weights, &config);
        let mut rng = Rng::new(100);
        let accepted =
            verifier.speculate(&draft_weights, &draft_config, config.bos_token, 0, &mut rng);

        assert!(!accepted.is_empty(), "should return at least 1 token");
        assert!(
            accepted.len() <= draft_config.draft_lookahead + 1,
            "should return at most gamma+1"
        );
        for &t in &accepted {
            assert!(t < config.vocab_size, "token {t} out of range");
        }
    }

    #[cfg(feature = "leviathan")]
    #[test]
    fn test_leviathan_verifier_deterministic() {
        let config = Config::micro();
        let draft_config = Config::draft();
        let mut rng = Rng::new(42);
        let target_weights = TransformerWeights::new(&config, &mut rng);
        let mut draft_rng = Rng::new(99);
        let draft_weights = TransformerWeights::new(&draft_config, &mut draft_rng);

        let r1 = {
            let mut verifier = LeviathanVerifier::new(&target_weights, &config);
            verifier.speculate(
                &draft_weights,
                &draft_config,
                config.bos_token,
                0,
                &mut Rng::new(100),
            )
        };
        let r2 = {
            let mut verifier = LeviathanVerifier::new(&target_weights, &config);
            verifier.speculate(
                &draft_weights,
                &draft_config,
                config.bos_token,
                0,
                &mut Rng::new(100),
            )
        };

        assert_eq!(r1, r2, "same seed should produce same results");
    }

    #[cfg(feature = "leviathan")]
    #[test]
    fn test_leviathan_verifier_bonus_token() {
        let config = Config::micro();
        let draft_config = Config::draft();
        let mut rng = Rng::new(42);
        let target_weights = TransformerWeights::new(&config, &mut rng);
        let mut draft_rng = Rng::new(99);
        let draft_weights = TransformerWeights::new(&draft_config, &mut draft_rng);

        let mut saw_bonus = false;
        for seed in 0..200u64 {
            let mut verifier = LeviathanVerifier::new(&target_weights, &config);
            let accepted = verifier.speculate(
                &draft_weights,
                &draft_config,
                config.bos_token,
                0,
                &mut Rng::new(seed),
            );
            // gamma=1 → max 2 tokens with bonus
            if accepted.len() >= 2 {
                saw_bonus = true;
                break;
            }
        }
        assert!(
            saw_bonus,
            "should see bonus token (gamma+1) at least once in 200 tries"
        );
    }

    #[cfg(feature = "leviathan")]
    #[test]
    fn test_leviathan_verifier_acceptance_decreases_with_gamma() {
        let config = Config::micro();
        let draft_config = Config::draft();
        let mut rng = Rng::new(42);
        let target_weights = TransformerWeights::new(&config, &mut rng);
        let mut draft_rng = Rng::new(99);
        let draft_weights = TransformerWeights::new(&draft_config, &mut draft_rng);

        let avg_for_gamma = |gamma: usize| -> f64 {
            let mut total = 0usize;
            let iters = 100;
            let mut gc = Config::draft();
            gc.draft_lookahead = gamma;
            for seed in 0..iters as u64 {
                let mut verifier = LeviathanVerifier::new(&target_weights, &config);
                let accepted = verifier.speculate(
                    &draft_weights,
                    &gc,
                    config.bos_token,
                    0,
                    &mut Rng::new(seed),
                );
                total += accepted.len();
            }
            total as f64 / iters as f64
        };

        let avg_1 = avg_for_gamma(1);
        let avg_4 = avg_for_gamma(4);
        let avg_8 = avg_for_gamma(8);

        assert!(avg_1 >= 1.0, "gamma=1 should give >=1 token, got {avg_1}");
        assert!(avg_4 >= 1.0, "gamma=4 should give >=1 token, got {avg_4}");
        assert!(avg_8 >= 1.0, "gamma=8 should give >=1 token, got {avg_8}");
    }
}
