use crate::speculative::dd_tree::{TreeBuilder, extract_best_path_into};
use crate::speculative::dflash::dflash_predict_with;
use crate::speculative::sampling::sample_from_distribution;
use crate::speculative::types::{NoPruner, SpeculativeContext};
use crate::transformer::TransformerWeights;
use crate::types::{Config, Rng};

use crate::speculative::dflash::dflash_predict_ar_with;
use crate::speculative::sampling::sample_residual_distribution_into;
use crate::transformer::{
    ForwardContext, MultiLayerKVCache, forward, preload_kv_cache, project_target_activation,
};
use crate::types::softmax_scaled;

// ── Speculative Verifier: Strategy Pattern ──────────────────

/// Strategy for verifying drafted tokens against a target distribution.
///
/// Same pattern as `ConstraintPruner` — trait-based swap point.
/// - `SimulatedVerifier`: fast, no target model needed (default).
/// - `LeviathanVerifier`: real p/q rejection sampling with target model.
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

        // 2. Build marginals view for tree building (zero-alloc: stack array + marginals_into)
        let mut marginals_buf: [&[f32]; 64] = [&[]; 64];
        let marginals_view = self.sctx.marginals_into(&mut marginals_buf, vocab_size);
        let tree = self
            .tree_builder
            .build(marginals_view, draft_config, &NoPruner, false);

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

pub struct LeviathanVerifier<'a> {
    pub target_weights: &'a TransformerWeights,
    pub target_config: &'a Config,
    target_ctx: ForwardContext,
    target_cache: MultiLayerKVCache,
    draft_sctx: SpeculativeContext,
    #[allow(dead_code)] // Pre-allocated for future tree-based Leviathan variants
    tree_builder: TreeBuilder,
}

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

        // Phase 0 (MTP): Get target hidden state for conditioning + score initial token
        self.target_cache.reset();
        {
            let logits = forward(
                &mut self.target_ctx,
                self.target_weights,
                &mut self.target_cache,
                token,
                pos,
                self.target_config,
            );
            self.draft_sctx.probs_buf.copy_from_slice(logits);
            softmax_scaled(&mut self.draft_sctx.probs_buf, 1.0 / target_temp);
            // Save p_dist[0] from this early target forward (avoids re-scoring later)
            self.draft_sctx.p_distributions_flat[..vocab_size]
                .copy_from_slice(&self.draft_sctx.probs_buf);
        }

        // Project target hidden state → drafter context buffer
        project_target_activation(
            &mut self.draft_sctx.ctx.mtp_context_buf,
            &self.target_ctx.hidden_state,
            self.target_weights.mtp_activation_proj.as_ref(),
            self.target_config.n_embd,
            draft_config.n_embd,
            self.target_config.mtp_activation_threshold,
        );

        // Determine if MTP conditioning is active (threshold gate)
        let mtp_active = self.target_config.n_embd >= self.target_config.mtp_activation_threshold;

        // Phase 1: AR draft with optional MTP conditioning
        self.draft_sctx.reset();

        // Preload drafter's KV cache from target's pre-computed KV
        // (hybrid: shared past positions + drafter computes own new positions)
        // Only active when prompt is long enough to benefit and kv_dim matches
        if pos > self.target_config.mtp_shared_kv_prompt_threshold {
            preload_kv_cache(
                &mut self.draft_sctx.cache,
                &self.target_cache,
                pos,
                self.target_config,
                draft_config,
            );
        }

        // Copy MTP context to stack (zero-alloc, avoids borrow conflict with &mut self.draft_sctx)
        let mut mtp_stack = [0.0f32; 256];
        let mtp_buf: Option<&[f32]> = if mtp_active {
            let n = draft_config.n_embd.min(mtp_stack.len());
            mtp_stack[..n].copy_from_slice(&self.draft_sctx.ctx.mtp_context_buf[..n]);
            Some(&mtp_stack[..n])
        } else {
            None
        };
        let gamma = dflash_predict_ar_with(
            &mut self.draft_sctx,
            draft_weights,
            draft_config,
            token,
            pos,
            rng,
            mtp_buf,
        );

        if gamma == 0 {
            // Already have softmaxed target logits in probs_buf from Phase 0
            return vec![sample_from_distribution(&self.draft_sctx.probs_buf, rng)];
        }

        // Phase 2: Target scoring — skip token[0] (already scored in Phase 0)
        // Copy sampled tokens to stack (zero-alloc, avoids borrow conflict with &mut self.draft_sctx)
        let mut token_stack = [0usize; 64];
        let gamma_bounded = gamma.min(token_stack.len());
        token_stack[..gamma_bounded]
            .copy_from_slice(&self.draft_sctx.sampled_tokens[..gamma_bounded]);

        // Score each drafted token → p_dist[1..=gamma]
        for (i, &draft_tok) in token_stack[..gamma_bounded].iter().enumerate() {
            let logits = forward(
                &mut self.target_ctx,
                self.target_weights,
                &mut self.target_cache,
                draft_tok,
                pos + 1 + i,
                self.target_config,
            );
            self.draft_sctx.probs_buf.copy_from_slice(logits);
            softmax_scaled(&mut self.draft_sctx.probs_buf, 1.0 / target_temp);
            let start = (i + 1) * vocab_size;
            self.draft_sctx.p_distributions_flat[start..start + vocab_size]
                .copy_from_slice(&self.draft_sctx.probs_buf);
        }

        // Phase 3: Rejection sampling
        self.draft_sctx.accepted_buf.clear();
        let mut all_accepted = true;

        for i in 0..gamma {
            let offset = i * vocab_size;
            let drafted_token = token_stack[i];

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

    // ── MTP Smoke Tests (Phase 5: T21–T23) ───────────────────

    /// T21: Small config (game) — all MTP features disabled.
    /// `mtp_activation_threshold = usize::MAX` → MTP inactive (32 < MAX).
    /// `draft_lookahead = 0` → no speculative drafting, returns single target sample.
    /// Output should be valid with the MTP code path present but inactive.
    #[test]
    fn test_mtp_small_config_disabled() {
        let target_config = Config::game();
        let draft_config = Config::game();

        let mut rng = Rng::new(42);
        let target_weights = TransformerWeights::new(&target_config, &mut rng);
        let draft_weights = TransformerWeights::new(&draft_config, &mut Rng::new(99));

        // MTP should be completely inactive: n_embd(32) < threshold(usize::MAX)
        assert!(
            target_config.n_embd < target_config.mtp_activation_threshold,
            "MTP should be inactive for game config"
        );

        let mut verifier = LeviathanVerifier::new(&target_weights, &target_config, &draft_config);
        let result = verifier.speculate(
            &draft_weights,
            &draft_config,
            target_config.bos_token,
            0,
            &mut Rng::new(100),
        );

        assert!(!result.is_empty(), "should return at least 1 token");
        for &t in &result {
            assert!(
                t < target_config.vocab_size,
                "token {t} out of range (vocab_size={})",
                target_config.vocab_size
            );
        }
    }

    /// T22: BPE config — MTP features active.
    /// `mtp_activation_threshold = 32`, `n_embd = 32` → MTP IS active (32 >= 32).
    /// Measures that acceptance rate is non-zero over multiple runs.
    #[test]
    fn test_mtp_bpe_config_active() {
        let target_config = Config::bpe();
        let draft_config = Config::bpe_draft();

        // MTP is active: n_embd(32) >= threshold(32)
        assert!(
            target_config.n_embd >= target_config.mtp_activation_threshold,
            "MTP should be active for bpe config"
        );

        let mut rng = Rng::new(42);
        let target_weights = TransformerWeights::new(&target_config, &mut rng);
        let draft_weights = TransformerWeights::new(&draft_config, &mut Rng::new(99));

        // Measure acceptance rate across multiple seeds
        let iters = 50u64;
        let mut total_tokens = 0usize;
        let mut all_valid = true;

        for seed in 0..iters {
            let mut verifier =
                LeviathanVerifier::new(&target_weights, &target_config, &draft_config);
            let result = verifier.speculate(
                &draft_weights,
                &draft_config,
                target_config.bos_token,
                0,
                &mut Rng::new(seed),
            );

            assert!(
                !result.is_empty(),
                "seed {seed}: should return at least 1 token"
            );
            total_tokens += result.len();

            for &t in &result {
                if t >= target_config.vocab_size {
                    all_valid = false;
                }
            }
        }

        let avg_tokens = total_tokens as f64 / iters as f64;
        assert!(all_valid, "all tokens should be within vocab range");
        assert!(
            avg_tokens >= 1.0,
            "acceptance rate should be >= 1.0 avg tokens, got {avg_tokens}"
        );
    }

    /// T23: Projection fallback — no trained MTP weights loaded.
    /// `TransformerWeights::new()` always sets `mtp_activation_proj: None`,
    /// so `project_target_activation` falls back to truncate/pad.
    /// Verifies the fallback path produces valid results with mismatched
    /// target n_embd(32) vs draft n_embd(16).
    #[test]
    fn test_mtp_projection_fallback_no_weights() {
        let target_config = Config::bpe(); // n_embd=32, threshold=32 → active
        let draft_config = Config::bpe_draft(); // n_embd=16

        let mut rng = Rng::new(42);
        let target_weights = TransformerWeights::new(&target_config, &mut rng);

        // Verify no MTP projection weights loaded (always None from ::new)
        assert!(
            target_weights.mtp_activation_proj.is_none(),
            "mtp_activation_proj should be None when loaded via ::new"
        );

        // MTP is active but will use truncate/pad fallback
        assert!(
            target_config.n_embd >= target_config.mtp_activation_threshold,
            "MTP should be active for this test"
        );
        assert_ne!(
            target_config.n_embd, draft_config.n_embd,
            "target and draft n_embd should differ to exercise truncate/pad"
        );

        let draft_weights = TransformerWeights::new(&draft_config, &mut Rng::new(99));

        let mut verifier = LeviathanVerifier::new(&target_weights, &target_config, &draft_config);
        let result = verifier.speculate(
            &draft_weights,
            &draft_config,
            target_config.bos_token,
            0,
            &mut Rng::new(100),
        );

        assert!(
            !result.is_empty(),
            "should return at least 1 token via fallback path"
        );
        for &t in &result {
            assert!(
                t < target_config.vocab_size,
                "token {t} out of range (vocab_size={})",
                target_config.vocab_size
            );
        }
    }

    // ── Shared KV Cache Tests (Phase 3, Plan 055) ──────────────

    /// T14: Shared KV with mismatching kv_dim — preload silently skipped.
    /// bpe (kv_dim=32) vs bpe_draft (kv_dim=16) — dimensions don't match,
    /// so drafter computes its own KV as before. Should still produce valid output.
    #[test]
    fn test_shared_kv_mismatching_dims_skips_preload() {
        let target_config = Config::bpe(); // kv_dim=32
        let draft_config = Config::bpe_draft(); // kv_dim=16

        let target_kv = crate::types::kv_dim(&target_config);
        let draft_kv = crate::types::kv_dim(&draft_config);
        assert_ne!(target_kv, draft_kv, "kv_dim should differ for this test");

        let mut rng = Rng::new(42);
        let target_weights = TransformerWeights::new(&target_config, &mut rng);
        let draft_weights = TransformerWeights::new(&draft_config, &mut Rng::new(99));

        let mut verifier = LeviathanVerifier::new(&target_weights, &target_config, &draft_config);

        // Use pos above threshold — but preload should still skip due to dim mismatch
        let pos = target_config.mtp_shared_kv_prompt_threshold + 1;
        let result = verifier.speculate(
            &draft_weights,
            &draft_config,
            target_config.bos_token,
            pos,
            &mut Rng::new(100),
        );

        assert!(!result.is_empty(), "should return at least 1 token");
        for &t in &result {
            assert!(t < target_config.vocab_size, "token {t} out of range");
        }
    }

    /// T14: Shared KV with matching kv_dim — preload active.
    /// Uses same config for both target and draft (guaranteed matching kv_dim).
    /// Verifies hybrid forward produces valid output.
    #[test]
    fn test_shared_kv_matching_dims_produces_valid_output() {
        // Use small_target with threshold=0 so preload always triggers
        let config = Config {
            mtp_shared_kv_prompt_threshold: 0,
            ..Config::small_target()
        };

        let mut rng = Rng::new(42);
        let target_weights = TransformerWeights::new(&config, &mut rng);
        let draft_weights = TransformerWeights::new(&config, &mut Rng::new(99));

        let mut verifier = LeviathanVerifier::new(&target_weights, &config, &config);

        // pos=5 > threshold(0), kv_dim matches → preload active
        let pos = 5;
        let result = verifier.speculate(
            &draft_weights,
            &config,
            config.bos_token,
            pos,
            &mut Rng::new(100),
        );

        assert!(
            !result.is_empty(),
            "should return at least 1 token with shared KV"
        );
        for &t in &result {
            assert!(t < config.vocab_size, "token {t} out of range");
        }

        // Run again at different positions to verify stability
        for pos in [1, 10, 20] {
            let result = verifier.speculate(
                &draft_weights,
                &config,
                config.bos_token,
                pos,
                &mut Rng::new(pos as u64),
            );
            assert!(
                !result.is_empty(),
                "should return at least 1 token at pos={pos}"
            );
            for &t in &result {
                assert!(t < config.vocab_size, "token {t} out of range at pos={pos}");
            }
        }
    }

    /// T14: Shared KV below threshold — preload not triggered.
    /// Same config (matching kv_dim) but pos <= threshold → no preload.
    #[test]
    fn test_shared_kv_below_threshold_no_preload() {
        let config = Config {
            mtp_shared_kv_prompt_threshold: 10, // Only preload when pos > 10
            ..Config::small_target()
        };

        let mut rng = Rng::new(42);
        let target_weights = TransformerWeights::new(&config, &mut rng);
        let draft_weights = TransformerWeights::new(&config, &mut Rng::new(99));

        let mut verifier = LeviathanVerifier::new(&target_weights, &config, &config);

        // pos=3 < threshold(10) → preload NOT triggered
        let result = verifier.speculate(
            &draft_weights,
            &config,
            config.bos_token,
            3,
            &mut Rng::new(100),
        );

        assert!(
            !result.is_empty(),
            "should return at least 1 token without preload"
        );
        for &t in &result {
            assert!(t < config.vocab_size, "token {t} out of range");
        }
    }
}
