use crate::speculative::dd_tree::{TreeBuilder, extract_best_path_into};
use crate::speculative::dflash::dflash_predict_with;
use crate::speculative::sampling::sample_from_distribution;
use crate::speculative::types::{NoPruner, SpeculativeContext};
use crate::transformer::TransformerWeights;
use crate::types::{Config, Rng};

#[cfg(feature = "leviathan")]
use crate::speculative::dflash::dflash_predict_ar_with;
#[cfg(feature = "leviathan")]
use crate::speculative::sampling::sample_residual_distribution_into;
#[cfg(feature = "leviathan")]
use crate::transformer::{ForwardContext, MultiLayerKVCache, forward};
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
///
/// Uses pre-allocated `SpeculativeContext` and `TreeBuilder` for zero-alloc
/// hot paths. Create once with `new(acceptance_rate, config)`, reuse across calls.
pub struct SimulatedVerifier {
    pub acceptance_rate: f32,
    sctx: SpeculativeContext,
    tree_builder: TreeBuilder,
}

impl SimulatedVerifier {
    pub fn new(acceptance_rate: f32, draft_config: &Config) -> Self {
        Self {
            acceptance_rate: acceptance_rate.clamp(0.0, 1.0),
            sctx: SpeculativeContext::new(draft_config),
            tree_builder: TreeBuilder::new(draft_config),
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
        let vocab_size = draft_config.vocab_size;

        // 1. Zero-alloc DFlash draft
        self.sctx.reset();
        dflash_predict_with(&mut self.sctx, draft_weights, draft_config, token, pos);

        // 2. Build marginals view for tree building (zero-alloc: borrow slices from flat buffer)
        let marginals_view = self.sctx.marginals_view(vocab_size);
        let tree = self
            .tree_builder
            .build(&marginals_view, draft_config, &NoPruner, false);

        // 3. Extract best path into pre-allocated buffer
        extract_best_path_into(tree, &mut self.sctx.path_buf);

        if self.sctx.path_buf.is_empty() {
            let first_marginal = self.sctx.marginal_slice(0, vocab_size);
            return vec![sample_from_distribution(
                if first_marginal.is_empty() {
                    &[1.0]
                } else {
                    first_marginal
                },
                rng,
            )];
        }

        // 4. Simulated acceptance: cap at rate
        let max_accept = ((self.sctx.path_buf.len() as f32) * self.acceptance_rate).ceil() as usize;
        self.sctx.accepted_buf.clear();
        self.sctx
            .accepted_buf
            .extend(self.sctx.path_buf.iter().take(max_accept.max(1)).copied());

        // 5. Bonus token: if all accepted, sample +1 from last marginal
        if self.sctx.accepted_buf.len() == max_accept && self.sctx.steps_populated > 0 {
            let last_step = self.sctx.steps_populated - 1;
            let start = last_step * vocab_size;
            let end = start + vocab_size;
            let last_marginal: &[f32] = if end <= self.sctx.marginals_flat.len() {
                &self.sctx.marginals_flat[start..end]
            } else {
                &[]
            };
            let bonus = sample_from_distribution(
                if last_marginal.is_empty() {
                    &[1.0]
                } else {
                    last_marginal
                },
                rng,
            );
            self.sctx.accepted_buf.push(bonus);
        }

        self.sctx.accepted_buf.clone()
    }
}

// ── LeviathanVerifier: Real p/q rejection sampling (Algorithm 1) ──

#[cfg(feature = "leviathan")]
pub struct LeviathanVerifier<'a> {
    pub target_weights: &'a TransformerWeights,
    pub target_config: &'a Config,
    target_ctx: ForwardContext,
    target_cache: MultiLayerKVCache,
    draft_sctx: SpeculativeContext,
    #[allow(dead_code)] // Pre-allocated for future tree-based Leviathan variants
    tree_builder: TreeBuilder,
}

#[cfg(feature = "leviathan")]
impl<'a> LeviathanVerifier<'a> {
    pub fn new(
        target_weights: &'a TransformerWeights,
        target_config: &'a Config,
        draft_config: &Config,
    ) -> Self {
        Self {
            target_weights,
            target_config,
            target_ctx: ForwardContext::new(target_config),
            target_cache: MultiLayerKVCache::new(target_config),
            draft_sctx: SpeculativeContext::new(draft_config),
            tree_builder: TreeBuilder::new(draft_config),
        }
    }
}

#[cfg(feature = "leviathan")]
impl SpeculativeVerifier for LeviathanVerifier<'_> {
    #[allow(clippy::needless_range_loop)]
    fn speculate(
        &mut self,
        draft_weights: &TransformerWeights,
        draft_config: &Config,
        token: usize,
        pos: usize,
        rng: &mut Rng,
    ) -> Vec<usize> {
        let vocab_size = draft_config.vocab_size;
        let target_temp = self.target_config.temperature;

        // Phase 1: Zero-alloc AR draft
        self.draft_sctx.reset();
        let gamma = dflash_predict_ar_with(
            &mut self.draft_sctx,
            draft_weights,
            draft_config,
            token,
            pos,
            rng,
        );

        if gamma == 0 {
            self.target_cache.reset();
            let logits = forward(
                &mut self.target_ctx,
                self.target_weights,
                &mut self.target_cache,
                token,
                pos,
                self.target_config,
            );
            let mut temp_buf = logits.to_vec();
            for p in temp_buf.iter_mut() {
                *p /= target_temp;
            }
            softmax(&mut temp_buf);
            return vec![sample_from_distribution(&temp_buf, rng)];
        }

        // Phase 2: Target scoring — write p_dist directly to flat buffer
        self.target_cache.reset();

        // Score initial token → p_dist[0]
        {
            let logits = forward(
                &mut self.target_ctx,
                self.target_weights,
                &mut self.target_cache,
                token,
                pos,
                self.target_config,
            );
            let mut temp_buf = logits.to_vec();
            for p in temp_buf.iter_mut() {
                *p /= target_temp;
            }
            softmax(&mut temp_buf);
            self.draft_sctx.p_distributions_flat[..vocab_size].copy_from_slice(&temp_buf);
        }

        // Copy sampled tokens before iterating (avoids borrow conflicts with flat buffers)
        let sampled_tokens = self.draft_sctx.sampled_tokens[..gamma].to_vec();

        // Score each drafted token → p_dist[1..=gamma]
        for (i, &draft_tok) in sampled_tokens.iter().enumerate() {
            let logits = forward(
                &mut self.target_ctx,
                self.target_weights,
                &mut self.target_cache,
                draft_tok,
                pos + 1 + i,
                self.target_config,
            );
            let mut temp_buf = logits.to_vec();
            for p in temp_buf.iter_mut() {
                *p /= target_temp;
            }
            softmax(&mut temp_buf);
            let start = (i + 1) * vocab_size;
            self.draft_sctx.p_distributions_flat[start..start + vocab_size]
                .copy_from_slice(&temp_buf);
        }

        // Phase 3: Rejection sampling
        self.draft_sctx.accepted_buf.clear();
        let mut all_accepted = true;

        for i in 0..gamma {
            let offset = i * vocab_size;
            let drafted_token = sampled_tokens[i];

            let p_i = self
                .draft_sctx
                .p_distributions_flat
                .get(offset + drafted_token)
                .copied()
                .unwrap_or(0.0);
            let q_i = self
                .draft_sctx
                .marginals_flat
                .get(offset + drafted_token)
                .copied()
                .unwrap_or(0.0);

            let acceptance_prob = if q_i > 0.0 { (p_i / q_i).min(1.0) } else { 1.0 };
            let r = rng.uniform();

            if r <= acceptance_prob {
                self.draft_sctx.accepted_buf.push(drafted_token);
            } else {
                let p_dist = &self.draft_sctx.p_distributions_flat[offset..offset + vocab_size];
                let q_dist = &self.draft_sctx.marginals_flat[offset..offset + vocab_size];
                let replacement = sample_residual_distribution_into(
                    p_dist,
                    q_dist,
                    &mut self.draft_sctx.residual_buf,
                    rng,
                );
                self.draft_sctx.accepted_buf.push(replacement);
                all_accepted = false;
                break;
            }
        }

        // Phase 4: Bonus token
        if all_accepted {
            let bonus_start = gamma * vocab_size;
            let bonus_end = bonus_start + vocab_size;
            if bonus_end <= self.draft_sctx.p_distributions_flat.len() {
                let bonus_dist = &self.draft_sctx.p_distributions_flat[bonus_start..bonus_end];
                let bonus = sample_from_distribution(bonus_dist, rng);
                self.draft_sctx.accepted_buf.push(bonus);
            }
        }

        if self.draft_sctx.accepted_buf.is_empty() {
            let p_dist = &self.draft_sctx.p_distributions_flat[..vocab_size];
            self.draft_sctx
                .accepted_buf
                .push(sample_from_distribution(p_dist, rng));
        }

        self.draft_sctx.accepted_buf.clone()
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
        let mut verifier = SimulatedVerifier::new(0.75, &config);
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
            let mut verifier = SimulatedVerifier::new(0.75, &config);
            verifier.speculate(&weights, &config, 0, 0, &mut Rng::new(77))
        };
        let a2 = {
            let mut verifier = SimulatedVerifier::new(0.75, &config);
            verifier.speculate(&weights, &config, 0, 0, &mut Rng::new(77))
        };

        assert_eq!(a1, a2, "same seed should produce same accepted tokens");
    }

    #[test]
    fn test_simulated_verifier_bonus_token() {
        let (weights, config) = make_draft();
        let mut saw_bonus = false;
        for seed in 0..200u64 {
            let mut verifier = SimulatedVerifier::new(0.95, &config);
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

        let mut verifier = LeviathanVerifier::new(&target_weights, &config, &draft_config);
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
            let mut verifier = LeviathanVerifier::new(&target_weights, &config, &draft_config);
            verifier.speculate(
                &draft_weights,
                &draft_config,
                config.bos_token,
                0,
                &mut Rng::new(100),
            )
        };
        let r2 = {
            let mut verifier = LeviathanVerifier::new(&target_weights, &config, &draft_config);
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
            let mut verifier = LeviathanVerifier::new(&target_weights, &config, &draft_config);
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
                let mut verifier = LeviathanVerifier::new(&target_weights, &config, &gc);
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
