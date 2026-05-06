use crate::speculative::verifier::{SimulatedVerifier, SpeculativeVerifier};
use crate::transformer::TransformerWeights;
use crate::types::{Config, Rng};

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
