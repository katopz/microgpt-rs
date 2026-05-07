//! Speculative prefill: PFlash-inspired prompt compression via importance scoring.
//!
//! Uses the draft model's attention scores to estimate per-token importance,
//! then compresses the prompt to top-`keep_ratio` spans before target prefill.
//! Inspired by [Cross-Family Speculative Prefill](https://arxiv.org/abs/2603.02631).

use crate::transformer::{ForwardContext, MultiLayerKVCache, TransformerWeights, forward};
use crate::types::Config;

// ── Scoring Trait ──────────────────────────────────────────────

/// Strategy for scoring per-token importance over a prompt.
pub trait PrefillScorer: Send + Sync {
    /// Returns per-token importance scores, same length as `prompt_tokens`.
    fn score(
        &self,
        draft_weights: &TransformerWeights,
        draft_config: &Config,
        prompt_tokens: &[usize],
    ) -> Vec<f32>;
}

// ── Scorer Implementations ─────────────────────────────────────

/// Attention-based importance scorer (PFlash-inspired).
/// Uses softmax'd self-attention weights from draft model forward pass.
///
/// After each `forward()` call, `ctx.scores[0..=pos]` contains the last
/// attention head's normalized attention weights. The weight at index `pos`
/// (the self-attention weight) serves as a proxy for per-token importance.
pub struct AttentionScorer;

impl PrefillScorer for AttentionScorer {
    fn score(
        &self,
        draft_weights: &TransformerWeights,
        draft_config: &Config,
        prompt_tokens: &[usize],
    ) -> Vec<f32> {
        if prompt_tokens.is_empty() {
            return Vec::new();
        }

        let mut ctx = ForwardContext::new(draft_config);
        let mut cache = MultiLayerKVCache::new(draft_config);

        let mut scores = vec![0.0f32; prompt_tokens.len()];

        for (pos, &token) in prompt_tokens.iter().enumerate() {
            let _logits = forward(
                &mut ctx,
                draft_weights,
                &mut cache,
                token,
                pos,
                draft_config,
            );

            // After forward(), ctx.scores[0..=pos] holds the last head's softmax'd
            // attention weights. The self-attention weight at [pos] indicates how
            // strongly this position attends to itself relative to all prior positions.
            if pos < draft_config.block_size {
                scores[pos] = ctx.scores[pos];
            }
        }

        // Normalize scores to [0, 1] range
        let max_score = scores.iter().cloned().fold(0.0f32, f32::max);
        if max_score > 0.0 {
            for s in scores.iter_mut() {
                *s /= max_score;
            }
        }

        scores
    }
}

/// Random importance scorer (baseline).
/// Assigns random importance for ablation studies.
pub struct RandomScorer {
    pub seed: u64,
}

impl PrefillScorer for RandomScorer {
    fn score(
        &self,
        _draft_weights: &TransformerWeights,
        _draft_config: &Config,
        prompt_tokens: &[usize],
    ) -> Vec<f32> {
        let mut rng = crate::types::Rng::new(self.seed);
        prompt_tokens.iter().map(|_| rng.uniform()).collect()
    }
}

/// Uniform importance scorer (baseline).
/// All tokens get equal importance — equivalent to stride-based subsampling.
pub struct UniformScorer;

impl PrefillScorer for UniformScorer {
    fn score(
        &self,
        _draft_weights: &TransformerWeights,
        _draft_config: &Config,
        prompt_tokens: &[usize],
    ) -> Vec<f32> {
        vec![1.0; prompt_tokens.len()]
    }
}

// ── Compression ────────────────────────────────────────────────

/// Compress a prompt by selecting top-importance tokens.
///
/// Always keeps:
/// - First `prefix_len` tokens (system prompt / instruction prefix)
/// - Last `suffix_len` tokens (immediate context)
/// - Top-scoring tokens from the middle by importance
///
/// Returns indices of selected tokens in original order.
pub fn compress_prompt(
    importance_scores: &[f32],
    keep_ratio: f32,
    prefix_len: usize,
    suffix_len: usize,
) -> Vec<usize> {
    let total = importance_scores.len();
    if total == 0 {
        return Vec::new();
    }

    // Always keep prefix and suffix
    let prefix_len = prefix_len.min(total);
    let suffix_len = suffix_len.min(total.saturating_sub(prefix_len));

    let mandatory = prefix_len + suffix_len;
    let budget = ((total as f32 * keep_ratio).ceil() as usize).max(mandatory);

    if budget >= total {
        // Keep everything
        return (0..total).collect();
    }

    // Middle tokens: score-sorted selection
    let middle_start = prefix_len;
    let middle_end = total.saturating_sub(suffix_len);
    let middle_budget = budget.saturating_sub(mandatory);

    let mut middle_indices: Vec<(usize, f32)> = (middle_start..middle_end)
        .map(|i| (i, importance_scores[i]))
        .collect();

    // Sort by score descending, take top middle_budget
    middle_indices.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    middle_indices.truncate(middle_budget);

    // Collect all selected indices
    let mut selected: Vec<usize> = Vec::with_capacity(budget);

    // Prefix
    selected.extend(0..prefix_len);

    // Middle (sorted by original position)
    let mut middle_selected: Vec<usize> = middle_indices.into_iter().map(|(i, _)| i).collect();
    middle_selected.sort();
    selected.extend(middle_selected);

    // Suffix
    selected.extend(total.saturating_sub(suffix_len)..total);

    selected
}

// ── Top-level API ──────────────────────────────────────────────

/// Run speculative prefill: score tokens, compress, return selected indices.
///
/// This is the main entry point. After compression, the caller runs the
/// target model forward on only the selected tokens.
pub fn speculative_prefill(
    scorer: &dyn PrefillScorer,
    draft_weights: &TransformerWeights,
    draft_config: &Config,
    prompt_tokens: &[usize],
    keep_ratio: f32,
    prefix_len: usize,
    suffix_len: usize,
) -> Vec<usize> {
    if prompt_tokens.is_empty() {
        return Vec::new();
    }

    let scores = scorer.score(draft_weights, draft_config, prompt_tokens);
    compress_prompt(&scores, keep_ratio, prefix_len, suffix_len)
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transformer::TransformerWeights;
    use crate::types::Rng;

    fn make_draft() -> (TransformerWeights, Config) {
        let config = Config::draft();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        (weights, config)
    }

    #[test]
    fn test_compress_preserves_prefix_and_suffix() {
        let scores = vec![0.1; 20];
        let selected = compress_prompt(&scores, 0.5, 3, 2);

        // First 3 must be present
        assert!(selected.contains(&0));
        assert!(selected.contains(&1));
        assert!(selected.contains(&2));
        // Last 2 must be present
        assert!(selected.contains(&18));
        assert!(selected.contains(&19));
    }

    #[test]
    fn test_compress_ratio_approximate() {
        let scores = vec![0.5; 100];
        let selected = compress_prompt(&scores, 0.1, 2, 2);
        // 100 * 0.1 = 10, plus mandatory prefix(2) + suffix(2) = 4
        // Budget = max(10, 4) = 10, middle_budget = 10 - 4 = 6
        assert!(
            selected.len() <= 14,
            "should keep roughly keep_ratio tokens"
        );
    }

    #[test]
    fn test_compress_empty_prompt() {
        let selected = compress_prompt(&[], 0.5, 2, 2);
        assert!(selected.is_empty());
    }

    #[test]
    fn test_compress_single_token() {
        let scores = vec![0.5];
        let selected = compress_prompt(&scores, 0.5, 0, 0);
        assert_eq!(selected, vec![0]);
    }

    #[test]
    fn test_compress_keeps_all_when_budget_exceeds() {
        let scores = vec![0.5; 5];
        let selected = compress_prompt(&scores, 1.0, 0, 0);
        assert_eq!(selected.len(), 5);
    }

    #[test]
    fn test_uniform_scorer_all_equal() {
        let (weights, config) = make_draft();
        let tokens: Vec<usize> = vec![0; 8];
        let scorer = UniformScorer;
        let scores = scorer.score(&weights, &config, &tokens);
        assert!(scores.iter().all(|&s| (s - 1.0).abs() < 1e-6));
    }

    #[test]
    fn test_random_scorer_varies() {
        let (weights, config) = make_draft();
        let tokens: Vec<usize> = vec![0; 8];
        let s1 = RandomScorer { seed: 1 };
        let s2 = RandomScorer { seed: 2 };
        let scores1 = s1.score(&weights, &config, &tokens);
        let scores2 = s2.score(&weights, &config, &tokens);
        assert_ne!(scores1, scores2, "different seeds should differ");
    }

    #[test]
    fn test_attention_scorer_produces_scores() {
        let (weights, config) = make_draft();
        let tokens: Vec<usize> = (0..config.vocab_size.min(8)).collect();
        let scorer = AttentionScorer;
        let scores = scorer.score(&weights, &config, &tokens);
        assert_eq!(scores.len(), tokens.len());
        // All scores should be finite
        for &s in &scores {
            assert!(s.is_finite(), "score should be finite, got {s}");
        }
    }

    #[test]
    fn test_speculative_prefill_end_to_end() {
        let (weights, config) = make_draft();
        let tokens: Vec<usize> = (0..8).collect();
        let selected = speculative_prefill(&UniformScorer, &weights, &config, &tokens, 0.5, 1, 1);
        // Should keep first, last, and some middle
        assert!(selected.contains(&0), "should keep first token");
        assert!(selected.contains(&7), "should keep last token");
        assert!(selected.len() < tokens.len(), "should compress");
    }

    /// NIAH-style test: needle-in-haystack retrieval after compression.
    /// Generates [hay×N] + [needle_marker, secret] + [hay×N], then verifies
    /// the needle tokens survive compression when importance scores are high.
    #[test]
    fn test_niah_needle_preserved_after_compression() {
        // Prompt layout: [hay×5] + [needle_marker, secret] + [hay×5] = 12 tokens
        // vocab_size=27: a-z (0-25) + bos (26). Use 'a'=0 as hay, 'z'=25 as marker, 'y'=24 as secret.
        let hay_token: usize = 0; // 'a'
        let needle_marker: usize = 25; // 'z'
        let secret: usize = 24; // 'y'

        let prompt_tokens: Vec<usize> = [
            vec![hay_token; 5],
            vec![needle_marker, secret],
            vec![hay_token; 5],
        ]
        .concat();
        assert_eq!(prompt_tokens.len(), 12);

        // Craft importance scores: needle and secret get high scores, hay gets low
        let mut scores = vec![0.1f32; 12];
        scores[5] = 1.0; // needle marker position — highest importance
        scores[6] = 0.9; // secret position — second highest

        // Case 1: keep_ratio=0.3 → budget = ceil(12*0.3) = 4 tokens
        // Needle (score=1.0) and secret (score=0.9) should both survive
        let selected_30 = compress_prompt(&scores, 0.3, 1, 1);
        assert!(
            selected_30.contains(&5),
            "needle marker (pos 5) should survive at keep_ratio=0.3"
        );
        assert!(
            selected_30.contains(&6),
            "secret (pos 6) should survive at keep_ratio=0.3"
        );

        // Case 2: keep_ratio=0.1 → budget = ceil(12*0.1) = 2 tokens
        // Only top-2 by score survive: needle (1.0) and secret (0.9)
        // After mandatory prefix(1) + suffix(1) = 2, middle_budget = 0
        // So needle/secret must win via prefix/suffix OR score selection
        let selected_10 = compress_prompt(&scores, 0.1, 1, 1);
        // With prefix_len=1 and suffix_len=1, mandatory=2, budget=2
        // Middle budget = 0, so only prefix[0] and suffix[11] are kept.
        // Needle at pos 5 and secret at pos 6 are in the middle and won't be kept.
        assert!(
            selected_10.contains(&0),
            "prefix token should survive at keep_ratio=0.1"
        );
        assert!(
            selected_10.contains(&11),
            "suffix token should survive at keep_ratio=0.1"
        );
        // Middle tokens (including needle) are expected to be dropped at this budget
        assert_eq!(
            selected_10.len(),
            2,
            "only prefix+suffix survive when middle_budget=0"
        );

        // Instead, test with no prefix/suffix mandatory so needle wins by score
        let selected_10_nofix = compress_prompt(&scores, 0.1, 0, 0);
        // budget = 2, middle_budget = 2, top-2 scores: pos 5 (1.0) and pos 6 (0.9)
        assert!(
            selected_10_nofix.contains(&5),
            "needle marker should survive when no mandatory prefix/suffix"
        );
        assert!(
            selected_10_nofix.contains(&6),
            "secret should survive when no mandatory prefix/suffix"
        );

        // Case 3: uniform scores (no importance discrimination) — needle is lost
        let uniform_scores = vec![0.5f32; 12];
        let selected_uniform = compress_prompt(&uniform_scores, 0.1, 0, 0);
        // budget = 2, all scores equal, so arbitrary 2 tokens selected
        // Needle is NOT guaranteed to survive without importance discrimination
        assert_eq!(
            selected_uniform.len(),
            2,
            "uniform scoring should select exactly budget tokens"
        );
        // Demonstrate that importance scoring is what preserves the needle:
        // With uniform scores, needle may or may not be in the result
        let needle_survives_uniform =
            selected_uniform.contains(&5) && selected_uniform.contains(&6);
        // This is expected to usually be false — proving why scoring matters
        // We don't assert it's false (could be coincidentally true), but log the result
        match needle_survives_uniform {
            true => eprintln!("  [NIAH] Needle coincidentally survived with uniform scores"),
            false => {
                eprintln!("  [NIAH] Needle lost with uniform scores (expected — scoring matters)")
            }
        }
    }

    /// Bridge test: prefill compression → KV cache fill → speculative decode.
    /// Validates the data flow from prompt compression through to speculative
    /// step, simulating what speculative_step_rest() would do with a real REST target.
    #[cfg(feature = "rest")]
    #[test]
    fn test_bridge_prefill_to_speculative_decode() {
        use crate::speculative::{SimulatedVerifier, speculative_step_verifier};
        use crate::transformer::{ForwardContext, MultiLayerKVCache, forward};

        // Config with block_size large enough for 32 tokens
        let config = Config {
            block_size: 64,
            ..Config::draft()
        };
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);

        // 1. Create a 32-token prompt (vocab_size=27, so cycle tokens)
        let prompt_tokens: Vec<usize> = (0..32).map(|i| i % config.vocab_size).collect();
        assert_eq!(prompt_tokens.len(), 32);

        // 2. Run speculative prefill to compress
        let compressed_indices =
            speculative_prefill(&UniformScorer, &weights, &config, &prompt_tokens, 0.5, 2, 2);

        // Compression should reduce the prompt
        assert!(!compressed_indices.is_empty(), "should select some tokens");
        let compressed_len = compressed_indices.len();
        assert!(
            compressed_len < prompt_tokens.len(),
            "should compress from 32 to {compressed_len}"
        );

        // Indices should be in ascending order (compress_prompt guarantees this)
        for window in compressed_indices.windows(2) {
            assert!(window[0] < window[1], "indices should be sorted ascending");
        }

        // 3. Build target KV cache and fill with compressed tokens
        let mut target_cache = MultiLayerKVCache::new(&config);
        let mut target_ctx = ForwardContext::new(&config);

        for (pos, &idx) in compressed_indices.iter().enumerate() {
            let token = prompt_tokens[idx];
            let _logits = forward(
                &mut target_ctx,
                &weights,
                &mut target_cache,
                token,
                pos,
                &config,
            );
        }

        // 4. Verify target KV cache is populated (has non-zero values)
        let filled_positions = compressed_indices.len();
        let kv_dim = crate::types::kv_dim(&config);
        for layer in &target_cache.layers {
            let end = filled_positions * kv_dim;
            let key_nonzero = layer.key[..end].iter().any(|&v| v != 0.0);
            assert!(
                key_nonzero,
                "target KV cache key should have non-zero values"
            );
            let val_nonzero = layer.value[..end].iter().any(|&v| v != 0.0);
            assert!(
                val_nonzero,
                "target KV cache value should have non-zero values"
            );
        }

        // 5. Run speculative step with verifier using the filled cache state
        //    (simulates what speculative_step_rest does after prefill)
        let next_pos = filled_positions;
        let last_token = prompt_tokens[*compressed_indices.last().unwrap()];
        let mut verifier = SimulatedVerifier::new(0.75);
        let mut step_rng = Rng::new(123);
        let (accepted, accept_len) = speculative_step_verifier(
            &weights,
            &config,
            last_token,
            next_pos,
            &mut step_rng,
            &mut verifier,
        );

        // Should always accept at least one token
        assert!(
            !accepted.is_empty(),
            "should accept at least 1 token from speculative step"
        );
        assert!(accept_len >= 1, "accept_len should be >= 1");
        for &t in &accepted {
            assert!(t < config.vocab_size, "token {t} out of vocab range");
        }
    }
}
