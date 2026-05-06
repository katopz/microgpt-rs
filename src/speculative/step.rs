use crate::speculative::verifier::{SimulatedVerifier, SpeculativeVerifier};
use crate::transformer::TransformerWeights;
use crate::types::{Config, Rng};

#[cfg(feature = "rest")]
use crate::rest::{RestClient, RetrievalResult};
#[cfg(feature = "rest")]
use crate::speculative::dd_tree::{build_dd_tree, extract_best_path, merge_retrieved_branches};
#[cfg(feature = "rest")]
use crate::speculative::dflash::dflash_predict;
#[cfg(feature = "rest")]
use crate::speculative::sampling::sample_from_distribution;
#[cfg(feature = "rest")]
use crate::transformer::{ForwardContext, MultiLayerKVCache, forward};

/// Speculative decoding step with a custom verifier.
/// Pass any `SpeculativeVerifier` to control how drafts are verified.
pub fn speculative_step_verifier(
    draft_weights: &TransformerWeights,
    draft_config: &Config,
    token: usize,
    pos: usize,
    rng: &mut Rng,
    verifier: &mut dyn SpeculativeVerifier,
) -> (Vec<usize>, usize) {
    let accepted = verifier.speculate(draft_weights, draft_config, token, pos, rng);
    let len = accepted.len();
    (accepted, len)
}

/// Speculative decoding step with simulated verification (backward compat).
/// Uses `SimulatedVerifier` with 75% acceptance rate + DDTree.
pub fn speculative_step(
    draft_weights: &TransformerWeights,
    draft_config: &Config,
    token: usize,
    pos: usize,
    rng: &mut Rng,
) -> (Vec<usize>, usize) {
    let mut verifier = SimulatedVerifier::new(0.75);
    speculative_step_verifier(draft_weights, draft_config, token, pos, rng, &mut verifier)
}

// ── REST Speculative Step ─────────────────────────────────────

/// Speculative decoding step with REST retrieval augmentation.
///
/// Pipeline: DFlash → DDTree → target forward → REST query → merge → verify.
///
/// The hidden state from the target model forward pass is sent to anyrag,
/// which returns historical token continuations. These are merged into the
/// DDTree with blended scores, potentially improving acceptance rate.
#[cfg(feature = "rest")]
#[allow(clippy::too_many_arguments)]
pub async fn speculative_step_rest(
    draft_weights: &TransformerWeights,
    draft_config: &Config,
    target_weights: &TransformerWeights,
    target_config: &Config,
    token: usize,
    pos: usize,
    rng: &mut Rng,
    rest_client: &RestClient,
    rest_weight: f32,
) -> Vec<usize> {
    // 1. Draft marginals via DFlash
    let marginals = dflash_predict(draft_weights, draft_config, token, pos);

    // 2. Build initial DDTree
    let mut tree = build_dd_tree(&marginals, draft_config);

    // 3. Run target model forward to get hidden state
    let mut target_ctx = ForwardContext::new(target_config);
    let mut target_cache = MultiLayerKVCache::new(target_config);
    let _logits = forward(
        &mut target_ctx,
        target_weights,
        &mut target_cache,
        token,
        pos,
        target_config,
    );

    // 4. Query anyrag with hidden state embedding
    let retrieved = rest_client
        .retrieve(&target_ctx.hidden_state, 5)
        .await
        .unwrap_or(RetrievalResult::default());

    // 5. Merge retrieved branches into DDTree
    merge_retrieved_branches(
        &mut tree,
        &marginals,
        draft_config,
        &retrieved.token_sequences,
        &retrieved.scores,
        rest_weight,
    );

    // 6. Extract best path
    let path = extract_best_path(&tree);
    if path.is_empty() {
        return vec![sample_from_distribution(
            marginals.first().map(|m| m.as_slice()).unwrap_or(&[1.0]),
            rng,
        )];
    }

    // 7. Simulated acceptance (same as SimulatedVerifier)
    let acceptance_rate = 0.75;
    let max_accept = ((path.len() as f32) * acceptance_rate).ceil() as usize;
    let accepted: Vec<usize> = path.into_iter().take(max_accept.max(1)).collect();

    if accepted.len() == max_accept && !marginals.is_empty() {
        let last_marginal = marginals.last().unwrap();
        let bonus = sample_from_distribution(last_marginal, rng);
        let mut result = accepted;
        result.push(bonus);
        return result;
    }

    accepted
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

    #[test]
    fn test_speculative_step_accepts_at_least_one() {
        let (weights, config) = make_draft();
        for seed in [0, 42, 100, 999] {
            let mut rng = Rng::new(seed);
            let (accepted, accept_len) = speculative_step(&weights, &config, 0, 0, &mut rng);
            assert!(
                !accepted.is_empty(),
                "seed {seed}: should accept at least 1 token"
            );
            assert!(accept_len >= 1, "seed {seed}: accept_len should be >= 1");
            for &t in &accepted {
                assert!(t < config.vocab_size, "seed {seed}: token {t} out of range");
            }
        }
    }

    #[test]
    fn test_speculative_step_consistent_for_same_seed() {
        let (weights, config) = make_draft();

        let mut rng1 = Rng::new(77);
        let (a1, l1) = speculative_step(&weights, &config, 0, 0, &mut rng1);

        let mut rng2 = Rng::new(77);
        let (a2, l2) = speculative_step(&weights, &config, 0, 0, &mut rng2);

        assert_eq!(a1, a2, "same seed should produce same accepted tokens");
        assert_eq!(l1, l2, "same seed should produce same acceptance length");
    }

    #[test]
    fn test_simulated_verifier_returns_at_least_one() {
        use crate::speculative::verifier::SimulatedVerifier;

        let (weights, config) = make_draft();
        let mut verifier = SimulatedVerifier::new(0.75);
        let mut rng = Rng::new(42);
        let (accepted, len) =
            speculative_step_verifier(&weights, &config, 0, 0, &mut rng, &mut verifier);
        assert!(!accepted.is_empty(), "should return at least 1 token");
        assert!(len >= 1);
        for &t in &accepted {
            assert!(t < config.vocab_size, "token {t} out of range");
        }
    }

    #[test]
    fn test_simulated_verifier_deterministic() {
        use crate::speculative::verifier::SimulatedVerifier;

        let (weights, config) = make_draft();

        let (a1, l1) = {
            let mut verifier = SimulatedVerifier::new(0.75);
            speculative_step_verifier(&weights, &config, 0, 0, &mut Rng::new(77), &mut verifier)
        };
        let (a2, l2) = {
            let mut verifier = SimulatedVerifier::new(0.75);
            speculative_step_verifier(&weights, &config, 0, 0, &mut Rng::new(77), &mut verifier)
        };

        assert_eq!(a1, a2, "same seed should produce same accepted tokens");
        assert_eq!(l1, l2, "same seed should produce same acceptance length");
    }

    #[test]
    fn test_simulated_verifier_bonus_token() {
        use crate::speculative::verifier::SimulatedVerifier;

        let (weights, config) = make_draft();
        let mut saw_bonus = false;
        for seed in 0..200u64 {
            let mut verifier = SimulatedVerifier::new(0.95);
            let (accepted, _) = speculative_step_verifier(
                &weights,
                &config,
                0,
                0,
                &mut Rng::new(seed),
                &mut verifier,
            );
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

    #[test]
    fn test_no_pruner_allows_all() {
        use crate::speculative::types::{ConstraintPruner, NoPruner};

        let pruner = NoPruner;
        assert!(pruner.is_valid(0, 0, &[]));
        assert!(pruner.is_valid(0, 26, &[]));
        assert!(pruner.is_valid(100, 999, &[]));
    }
}
