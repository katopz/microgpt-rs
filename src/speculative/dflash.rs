use crate::speculative::sampling::sample_from_distribution;
use crate::speculative::types::DraftResult;
use crate::transformer::{ForwardContext, MultiLayerKVCache, TransformerWeights, forward};
use crate::types::{Config, Rng, softmax};
use rayon::prelude::*;

/// Sequential DFlash: Predict marginal distributions using draft model.
/// Uses pre-allocated ForwardContext for zero-alloc per step.
pub fn dflash_predict(
    draft_weights: &TransformerWeights,
    draft_config: &Config,
    token: usize,
    pos: usize,
) -> Vec<Vec<f32>> {
    let mut ctx = ForwardContext::new(draft_config);
    let max_steps = draft_config
        .draft_lookahead
        .min(draft_config.block_size.saturating_sub(pos));

    let mut marginals = Vec::with_capacity(max_steps);
    for step in 0..max_steps {
        let mut cache = MultiLayerKVCache::new(draft_config);
        let draft_pos = pos + step;
        let logits = forward(
            &mut ctx,
            draft_weights,
            &mut cache,
            token,
            draft_pos,
            draft_config,
        );
        let mut probs = logits.to_vec();
        for p in probs.iter_mut() {
            *p /= draft_config.temperature;
        }
        softmax(&mut probs);
        marginals.push(probs);
    }
    marginals
}

/// Parallel DFlash: Predict marginals using rayon.
/// One ForwardContext per rayon worker thread — no contention, zero waste.
pub fn dflash_predict_parallel(
    draft_weights: &TransformerWeights,
    draft_config: &Config,
    token: usize,
    pos: usize,
) -> Vec<Vec<f32>> {
    let max_steps = draft_config
        .draft_lookahead
        .min(draft_config.block_size.saturating_sub(pos));

    if max_steps == 0 {
        return Vec::new();
    }

    (0..max_steps)
        .into_par_iter()
        .map_init(
            || {
                (
                    ForwardContext::new(draft_config),
                    MultiLayerKVCache::new(draft_config),
                )
            },
            |(ctx, cache), step| {
                let draft_pos = pos + step;
                let logits = forward(ctx, draft_weights, cache, token, draft_pos, draft_config);
                let mut probs = logits.to_vec();
                for p in probs.iter_mut() {
                    *p /= draft_config.temperature;
                }
                softmax(&mut probs);
                probs
            },
        )
        .collect()
}

/// Autoregressive DFlash: Predict marginals by sampling and feeding back tokens.
///
/// Unlike `dflash_predict` (which feeds the same token/pos to every step),
/// this samples a token at each step and feeds it back as input for the next.
/// Produces conditional q(x|x_{<i}) distributions instead of independent marginals.
pub fn dflash_predict_ar(
    draft_weights: &TransformerWeights,
    draft_config: &Config,
    token: usize,
    pos: usize,
    rng: &mut Rng,
) -> DraftResult {
    let mut ctx = ForwardContext::new(draft_config);
    let mut cache = MultiLayerKVCache::new(draft_config);
    let max_steps = draft_config
        .draft_lookahead
        .min(draft_config.block_size.saturating_sub(pos));

    let mut marginals = Vec::with_capacity(max_steps);
    let mut sampled_tokens = Vec::with_capacity(max_steps);
    let mut cur_token = token;

    for step in 0..max_steps {
        let logits = forward(
            &mut ctx,
            draft_weights,
            &mut cache,
            cur_token,
            pos + step,
            draft_config,
        );
        let mut probs = logits.to_vec();
        for p in probs.iter_mut() {
            *p /= draft_config.temperature;
        }
        softmax(&mut probs);

        // Sample next token and feed back
        let next_token = sample_from_distribution(&probs, rng);
        marginals.push(probs);
        sampled_tokens.push(next_token);
        cur_token = next_token;
    }

    DraftResult {
        marginals,
        sampled_tokens,
    }
}

/// Target-conditioned DFlash: Predict marginals using draft model
/// conditioned on the target model's hidden state.
///
/// Uses Option C from plan 012: seed draft KV cache with target hidden state.
/// The target's hidden state (from `ForwardContext.hidden_state`) is projected
/// to the draft model's KV dimension and used as the initial KV cache entry.
/// This gives the draft model access to the target's representation without
/// any weight matrix changes.
///
/// Returns `DraftResult` with marginals and sampled tokens.
pub fn dflash_predict_conditioned(
    draft_weights: &TransformerWeights,
    draft_config: &Config,
    token: usize,
    pos: usize,
    target_hidden_state: &[f32],
    rng: &mut Rng,
) -> DraftResult {
    let mut ctx = ForwardContext::new(draft_config);
    let mut cache = MultiLayerKVCache::new(draft_config);
    let max_steps = draft_config.draft_lookahead.min(
        draft_config
            .block_size
            .saturating_sub(pos)
            .saturating_sub(1),
    );

    // Seed draft KV cache with target hidden state (Option C)
    // Project target hidden_state (target n_embd) to draft kv_dim
    // KV cache is [block_size * kv_dim] flat, so position 0 occupies [0..kv_dim]
    let draft_kv_dim = crate::types::kv_dim(draft_config);
    if !target_hidden_state.is_empty() && draft_kv_dim > 0 {
        let target_dim = target_hidden_state.len().min(draft_kv_dim);
        for layer in &mut cache.layers {
            layer.key[..target_dim].copy_from_slice(&target_hidden_state[..target_dim]);
            layer.key[target_dim..draft_kv_dim].fill(0.0);
            layer.value[..target_dim].copy_from_slice(&target_hidden_state[..target_dim]);
            layer.value[target_dim..draft_kv_dim].fill(0.0);
        }
    }

    let mut marginals = Vec::with_capacity(max_steps);
    let mut sampled_tokens = Vec::with_capacity(max_steps);
    let mut cur_token = token;

    for step in 0..max_steps {
        let logits = forward(
            &mut ctx,
            draft_weights,
            &mut cache,
            cur_token,
            pos + step + 1, // +1 because KV slot 0 is occupied by seed
            draft_config,
        );
        let mut probs = logits.to_vec();
        for p in probs.iter_mut() {
            *p /= draft_config.temperature;
        }
        softmax(&mut probs);

        let next_token = sample_from_distribution(&probs, rng);
        marginals.push(probs);
        sampled_tokens.push(next_token);
        cur_token = next_token;
    }

    DraftResult {
        marginals,
        sampled_tokens,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::speculative::dd_tree::{build_dd_tree, extract_best_path};
    use crate::transformer::TransformerWeights;
    use crate::types::{Config, Rng};

    fn make_draft() -> (TransformerWeights, Config) {
        let config = Config::draft();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        (weights, config)
    }

    #[test]
    fn test_dflash_produces_marginals() {
        let (weights, config) = make_draft();
        let marginals = dflash_predict(&weights, &config, 0, 0);
        assert!(!marginals.is_empty());
        assert!(marginals.len() <= config.draft_lookahead);

        for (i, row) in marginals.iter().enumerate() {
            assert_eq!(row.len(), config.vocab_size, "row {i} wrong size");
            let sum: f32 = row.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-4,
                "row {i} sum = {sum}, expected 1.0"
            );
        }
    }

    #[test]
    fn test_dflash_parallel_matches_count() {
        let (weights, config) = make_draft();
        let seq = dflash_predict(&weights, &config, 0, 0);
        let par = dflash_predict_parallel(&weights, &config, 0, 0);
        assert_eq!(seq.len(), par.len(), "parallel should produce same count");
    }

    #[test]
    fn test_dflash_positions_differ() {
        let (weights, config) = make_draft();
        let m0 = dflash_predict(&weights, &config, 0, 0);
        let m1 = dflash_predict(&weights, &config, 0, 1);
        assert_ne!(
            m0[0], m1[0],
            "marginals at different positions should differ"
        );
    }

    #[test]
    fn test_dflash_ar_produces_marginals() {
        let (weights, config) = make_draft();
        let result = dflash_predict_ar(&weights, &config, 0, 0, &mut Rng::new(42));
        assert!(!result.marginals.is_empty(), "should produce marginals");
        assert!(
            !result.sampled_tokens.is_empty(),
            "should produce sampled tokens"
        );
        assert_eq!(result.marginals.len(), result.sampled_tokens.len());
        for probs in &result.marginals {
            assert_eq!(probs.len(), config.vocab_size);
            let sum: f32 = probs.iter().sum();
            assert!(
                (sum - 1.0).abs() < 0.01,
                "probs should sum to ~1.0, got {sum}"
            );
        }
    }

    #[test]
    fn test_dflash_ar_is_autoregressive() {
        let (weights, config) = make_draft();
        let r1 = dflash_predict_ar(&weights, &config, 0, 0, &mut Rng::new(1));
        let r2 = dflash_predict_ar(&weights, &config, 0, 0, &mut Rng::new(2));
        assert_ne!(
            r1.sampled_tokens, r2.sampled_tokens,
            "different seeds should produce different AR tokens"
        );
    }

    #[test]
    fn test_dflash_ar_deterministic() {
        let (weights, config) = make_draft();
        let r1 = dflash_predict_ar(&weights, &config, 0, 0, &mut Rng::new(42));
        let r2 = dflash_predict_ar(&weights, &config, 0, 0, &mut Rng::new(42));
        assert_eq!(
            r1.sampled_tokens, r2.sampled_tokens,
            "same seed should produce same tokens"
        );
        for (a, b) in r1.marginals.iter().zip(r2.marginals.iter()) {
            for (pa, pb) in a.iter().zip(b.iter()) {
                assert!((pa - pb).abs() < 1e-6, "marginals should be identical");
            }
        }
    }

    #[test]
    fn test_extract_best_path() {
        let (weights, config) = make_draft();
        let marginals = dflash_predict(&weights, &config, 0, 0);
        let tree = build_dd_tree(&marginals, &config);
        let path = extract_best_path(&tree);
        if !tree.is_empty() {
            assert!(!path.is_empty(), "non-empty tree should produce a path");
            for &t in &path {
                assert!(t < config.vocab_size, "token {t} out of range");
            }
        }
    }

    #[test]
    fn test_dflash_conditioned_produces_marginals() {
        let (weights, config) = make_draft();
        let target_config = Config::micro();
        let mut rng = Rng::new(42);
        let target_weights = TransformerWeights::new(&target_config, &mut rng);

        // Get target hidden state
        let mut target_ctx = ForwardContext::new(&target_config);
        let mut target_cache = MultiLayerKVCache::new(&target_config);
        let _ = forward(
            &mut target_ctx,
            &target_weights,
            &mut target_cache,
            0,
            0,
            &target_config,
        );
        let hidden = target_ctx.hidden_state.clone();

        let result =
            dflash_predict_conditioned(&weights, &config, 0, 0, &hidden, &mut Rng::new(42));
        assert!(!result.marginals.is_empty());
        assert_eq!(result.marginals.len(), result.sampled_tokens.len());
        for probs in &result.marginals {
            assert_eq!(probs.len(), config.vocab_size);
            let sum: f32 = probs.iter().sum();
            assert!(
                (sum - 1.0).abs() < 0.01,
                "probs should sum to ~1.0, got {sum}"
            );
        }
    }

    #[test]
    fn test_dflash_conditioned_differs_from_unconditioned() {
        let (weights, config) = make_draft();
        let target_config = Config::micro();
        let mut rng = Rng::new(42);
        let target_weights = TransformerWeights::new(&target_config, &mut rng);

        let mut target_ctx = ForwardContext::new(&target_config);
        let mut target_cache = MultiLayerKVCache::new(&target_config);
        let _ = forward(
            &mut target_ctx,
            &target_weights,
            &mut target_cache,
            0,
            0,
            &target_config,
        );
        let hidden = target_ctx.hidden_state.clone();

        let uncond = dflash_predict_ar(&weights, &config, 0, 0, &mut Rng::new(42));
        let cond = dflash_predict_conditioned(&weights, &config, 0, 0, &hidden, &mut Rng::new(42));

        // Conditioned should differ from unconditioned (different KV cache seed)
        assert_ne!(
            cond.sampled_tokens, uncond.sampled_tokens,
            "conditioned marginals should differ from unconditioned"
        );
    }

    #[test]
    fn test_dflash_conditioned_valid_probs() {
        let (weights, config) = make_draft();
        let hidden = vec![0.5; config.n_embd]; // fake hidden state
        let result =
            dflash_predict_conditioned(&weights, &config, 0, 0, &hidden, &mut Rng::new(42));
        for probs in &result.marginals {
            for &p in probs {
                assert!(p.is_finite(), "prob should be finite");
                assert!(p >= 0.0, "prob should be non-negative");
            }
        }
    }

    #[test]
    fn test_dflash_conditioned_empty_hidden() {
        let (weights, config) = make_draft();
        let result = dflash_predict_conditioned(&weights, &config, 0, 0, &[], &mut Rng::new(42));
        // Empty hidden state should still produce valid output (no seeding)
        assert!(!result.marginals.is_empty());
    }
}
