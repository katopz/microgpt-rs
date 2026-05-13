//! Speculative prefill: PFlash-inspired prompt compression via importance scoring.
//!
//! Uses the draft model's attention scores to estimate per-token importance,
//! then compresses the prompt to top-`keep_ratio` spans before target prefill.
//! Inspired by [Cross-Family Speculative Prefill](https://arxiv.org/abs/2603.02631).

use crate::speculative::types::{FlashPrefillConfig, PrefillMode, SpeculativeContext};
use crate::transformer::{TransformerWeights, forward};
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

    /// Zero-alloc variant: writes scores into pre-allocated buffer.
    /// `scores` must be `>= prompt_tokens.len()`. Written up to `prompt_tokens.len()`.
    fn score_into(
        &self,
        draft_weights: &TransformerWeights,
        draft_config: &Config,
        prompt_tokens: &[usize],
        scores: &mut [f32],
    ) {
        // Default: allocate and copy
        let result = self.score(draft_weights, draft_config, prompt_tokens);
        let len = result.len().min(scores.len());
        scores[..len].copy_from_slice(&result[..len]);
    }
}

// ── Scorer Implementations ─────────────────────────────────────

/// Attention-based importance scorer (PFlash-inspired).
/// Uses softmax'd self-attention weights from draft model forward pass.
///
/// After each `forward()` call, `ctx.scores[0..=pos]` contains the last
/// attention head's normalized attention weights. The weight at index `pos`
/// (the self-attention weight) serves as a proxy for per-token importance.
pub struct AttentionScorer;

impl AttentionScorer {
    /// Zero-alloc scoring using pre-allocated context.
    /// `scores` must be `>= prompt_tokens.len()`.
    pub fn score_with(
        &self,
        sctx: &mut SpeculativeContext,
        draft_weights: &TransformerWeights,
        draft_config: &Config,
        prompt_tokens: &[usize],
        scores: &mut [f32],
    ) {
        if prompt_tokens.is_empty() {
            return;
        }

        sctx.cache.reset();

        let len = prompt_tokens.len().min(scores.len());
        scores[..len].fill(0.0f32);

        for (pos, &token) in prompt_tokens.iter().enumerate() {
            if pos >= draft_config.block_size {
                break;
            }
            let _logits = forward(
                &mut sctx.ctx,
                draft_weights,
                &mut sctx.cache,
                token,
                pos,
                draft_config,
            );
            scores[pos] = sctx.ctx.scores[pos];
        }

        // Normalize scores to [0, 1] range
        let max_score = scores[..len].iter().cloned().fold(0.0f32, f32::max);
        if max_score > 0.0 {
            for s in scores[..len].iter_mut() {
                *s /= max_score;
            }
        }
    }
}

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

        let mut sctx = SpeculativeContext::new(draft_config);
        let mut scores = vec![0.0f32; prompt_tokens.len()];
        self.score_with(
            &mut sctx,
            draft_weights,
            draft_config,
            prompt_tokens,
            &mut scores,
        );
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

// ── PFlash Block-Sparse Selection (Plan 044) ──────────────────

/// Block selection: turns per-block scores into selected block indices.
///
/// Rules (from FlashPrefill / PFlash):
///   - sink:       k_block < attention_sink -> always include
///   - window:     |q_block - k_block| < window -> include (recent context)
///   - last_full:  q_block >= num_blocks - last_n_full -> include all blocks
///   - alpha:      score >= max_score * alpha -> include (importance threshold)
///   - causal:     k_block <= q_block
///
/// For prefill scoring, q_block = last block (scoring from generation-start position).
pub fn block_select(block_scores: &[f32], cfg: &FlashPrefillConfig) -> Vec<usize> {
    let num_blocks = block_scores.len();
    if num_blocks == 0 {
        return Vec::new();
    }

    let q_block = num_blocks - 1;
    let max_score = block_scores.iter().cloned().fold(0.0f32, f32::max);
    let threshold = max_score * cfg.alpha;

    let mut selected: Vec<usize> = Vec::with_capacity(num_blocks);

    for (k_block, &score) in block_scores.iter().enumerate() {
        if k_block > q_block {
            continue;
        }

        let keep = k_block < cfg.attention_sink
            || q_block.abs_diff(k_block) < cfg.window
            || q_block >= num_blocks.saturating_sub(cfg.last_n_full)
            || score >= threshold;

        if keep {
            selected.push(k_block);
        }
    }

    selected.sort();
    selected.dedup();
    selected
}

/// Full block-selection with per-(q_block, k_block, head) score grid.
///
/// `score`: [M][N][H] row-major (M=q_blocks, N=k_blocks, H=heads).
/// Returns selected indices per (q_block, head) and counts.
pub fn block_select_grid(
    score: &[f32],
    num_q_blocks: usize,
    num_k_blocks: usize,
    num_heads: usize,
    cfg: &FlashPrefillConfig,
) -> (Vec<i32>, Vec<i32>) {
    let m = num_q_blocks;
    let n = num_k_blocks;
    let h = num_heads;

    let mut idx_out = vec![-1i32; m * n * h];
    let mut cnt_out = vec![0i32; m * h];

    for q in 0..m {
        let last_full = q >= m.saturating_sub(cfg.last_n_full);

        for head in 0..h {
            let mut max_score: f32 = -f32::INFINITY;
            for k in 0..=q.min(n - 1) {
                let v = score[q * n * h + k * h + head];
                if v > max_score {
                    max_score = v;
                }
            }
            let thresh = max_score * cfg.alpha;

            let mut selected = Vec::with_capacity(n);
            for k in 0..=q.min(n - 1) {
                let keep = k < cfg.attention_sink
                    || q.abs_diff(k) < cfg.window
                    || last_full
                    || score[q * n * h + k * h + head] >= thresh;

                if keep {
                    selected.push(k as i32);
                }
            }

            selected.sort();

            let idx_row = &mut idx_out[q * n * h + head..];
            for (i, &sel) in selected.iter().enumerate() {
                idx_row[i * h] = sel;
            }
            cnt_out[q * h + head] = selected.len() as i32;
        }
    }

    (idx_out, cnt_out)
}

/// Block-sparse attention scorer (CPU fallback for PFlash).
///
/// Aggregates token-level attention scores into block-level importance,
/// then upsamples back to per-token scores for compression.
pub struct BlockAttentionScorer {
    pub config: FlashPrefillConfig,
}

impl BlockAttentionScorer {
    /// Zero-alloc scoring using pre-allocated context.
    pub fn score_with(
        &self,
        sctx: &mut SpeculativeContext,
        draft_weights: &TransformerWeights,
        draft_config: &Config,
        prompt_tokens: &[usize],
        scores: &mut [f32],
    ) {
        let block_size = self.config.block_size;
        let seq_len = prompt_tokens.len();
        let num_blocks = seq_len.div_ceil(block_size);

        if seq_len == 0 {
            return;
        }

        sctx.cache.reset();

        let filled = seq_len.min(draft_config.block_size);
        for (pos, &token) in prompt_tokens.iter().enumerate().take(filled) {
            let _logits = forward(
                &mut sctx.ctx,
                draft_weights,
                &mut sctx.cache,
                token,
                pos,
                draft_config,
            );
        }

        let mut block_scores = vec![0.0f32; num_blocks];
        let mut block_counts = vec![0usize; num_blocks];

        let tail_start = seq_len.saturating_sub(self.config.tail_window * block_size);
        for pos in tail_start..filled {
            let score = sctx.ctx.scores[pos];
            let block_idx = pos / block_size;
            if block_idx < num_blocks {
                block_scores[block_idx] += score;
                block_counts[block_idx] += 1;
            }
        }

        for i in 0..num_blocks {
            if block_counts[i] > 0 {
                block_scores[i] /= block_counts[i] as f32;
            }
        }

        let max_block = block_scores.iter().cloned().fold(0.0f32, f32::max);
        if max_block > 0.0 {
            for s in &mut block_scores {
                *s /= max_block;
            }
        }

        for (pos, slot) in scores.iter_mut().enumerate().take(seq_len) {
            *slot = block_scores[pos / block_size];
        }
    }
}

impl PrefillScorer for BlockAttentionScorer {
    fn score(
        &self,
        draft_weights: &TransformerWeights,
        draft_config: &Config,
        prompt_tokens: &[usize],
    ) -> Vec<f32> {
        let mut sctx = SpeculativeContext::new(draft_config);
        let mut scores = vec![0.0f32; prompt_tokens.len()];
        self.score_with(
            &mut sctx,
            draft_weights,
            draft_config,
            prompt_tokens,
            &mut scores,
        );
        scores
    }
}

/// Compress prompt using block-sparse selection (PFlash algorithm).
///
/// 1. Aggregate per-token scores to per-block scores (max of block)
/// 2. Select blocks via `block_select` rules
/// 3. Flatten selected blocks to token indices
/// 4. Always include prefix and suffix tokens
pub fn compress_prompt_blocks(
    importance_scores: &[f32],
    cfg: &FlashPrefillConfig,
    prefix_len: usize,
    suffix_len: usize,
) -> Vec<usize> {
    let total = importance_scores.len();
    if total == 0 {
        return Vec::new();
    }

    let block_size = cfg.block_size;
    let num_blocks = total.div_ceil(block_size);

    let mut block_scores = vec![0.0f32; num_blocks];
    for (i, &score) in importance_scores.iter().enumerate() {
        let block_idx = i / block_size;
        block_scores[block_idx] = block_scores[block_idx].max(score);
    }

    let selected_blocks = block_select(&block_scores, cfg);

    let mut selected_tokens: Vec<usize> = Vec::new();
    let prefix_end = prefix_len.min(total);
    selected_tokens.extend(0..prefix_end);

    for &block_idx in &selected_blocks {
        let block_start = block_idx * block_size;
        let block_end = ((block_idx + 1) * block_size).min(total);
        for token_idx in block_start..block_end {
            if token_idx >= prefix_end && token_idx < total.saturating_sub(suffix_len) {
                selected_tokens.push(token_idx);
            }
        }
    }

    let suffix_start = total.saturating_sub(suffix_len);
    selected_tokens.extend(suffix_start..total);

    selected_tokens.sort();
    selected_tokens.dedup();

    selected_tokens
}

/// Whether to apply compression for the given prompt length and mode.
pub fn should_compress(mode: PrefillMode, prompt_len: usize, threshold: usize) -> bool {
    match mode {
        PrefillMode::Off => false,
        PrefillMode::Always => true,
        PrefillMode::Auto => prompt_len >= threshold,
    }
}

/// PFlash compression — CPU path (fallback when no GPU).
pub fn speculative_prefill_block(
    scorer: &dyn PrefillScorer,
    draft_weights: &TransformerWeights,
    draft_config: &Config,
    prompt_tokens: &[usize],
    cfg: &FlashPrefillConfig,
    prefix_len: usize,
    suffix_len: usize,
) -> Vec<usize> {
    if prompt_tokens.is_empty() {
        return Vec::new();
    }
    let scores = scorer.score(draft_weights, draft_config, prompt_tokens);
    compress_prompt_blocks(&scores, cfg, prefix_len, suffix_len)
}

/// PFlash compression with adaptive threshold.
/// Picks CPU path (GPU path available via feature flag).
#[allow(clippy::too_many_arguments)]
pub fn speculative_prefill_adaptive(
    scorer: &dyn PrefillScorer,
    draft_weights: &TransformerWeights,
    draft_config: &Config,
    prompt_tokens: &[usize],
    mode: PrefillMode,
    threshold: usize,
    cfg: &FlashPrefillConfig,
    prefix_len: usize,
    suffix_len: usize,
) -> Vec<usize> {
    if !should_compress(mode, prompt_tokens.len(), threshold) {
        return (0..prompt_tokens.len()).collect();
    }
    speculative_prefill_block(
        scorer,
        draft_weights,
        draft_config,
        prompt_tokens,
        cfg,
        prefix_len,
        suffix_len,
    )
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::speculative::types::{FlashPrefillConfig, PrefillMode};
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
        let mut verifier = SimulatedVerifier::new(0.75, &config);
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

    // ── Plan 044: PFlash Block-Sparse Prefill Tests ─────────────

    #[test]
    fn test_block_select_sink_rule() {
        let cfg = FlashPrefillConfig::default();
        let scores = vec![0.0, 0.5, 0.8, 0.3, 0.6]; // 5 blocks
        let selected = block_select(&scores, &cfg);
        // Block 0 should always be selected (sink rule)
        assert!(selected.contains(&0));
    }

    #[test]
    fn test_block_select_window_rule() {
        let cfg = FlashPrefillConfig::default();
        let scores = vec![0.0, 0.0, 0.0, 0.0, 0.0]; // 5 blocks
        let selected = block_select(&scores, &cfg);
        // Last 2 blocks should be selected (window around q_block=4)
        assert!(selected.contains(&4));
        assert!(selected.contains(&3));
    }

    #[test]
    fn test_block_select_alpha_rule() {
        let cfg = FlashPrefillConfig {
            alpha: 0.5,
            attention_sink: 0,
            window: 0,
            last_n_full: 0,
            ..Default::default()
        };
        let scores = vec![0.1, 0.9, 0.2, 0.8, 1.0]; // 5 blocks, q=4
        let selected = block_select(&scores, &cfg);
        // Should select blocks with score >= 0.5 (0.5 * max=1.0)
        assert!(selected.contains(&1)); // 0.9 >= 0.5
        assert!(selected.contains(&3)); // 0.8 >= 0.5
        assert!(selected.contains(&4)); // 1.0 >= 0.5 (q_block)
        assert!(!selected.contains(&0)); // 0.1 < 0.5
    }

    #[test]
    fn test_block_select_empty() {
        let cfg = FlashPrefillConfig::default();
        let selected = block_select(&[], &cfg);
        assert!(selected.is_empty());
    }

    #[test]
    fn test_compress_prompt_blocks_preserves_prefix_suffix() {
        let cfg = FlashPrefillConfig::default();
        let scores = vec![0.5; 100];
        let selected = compress_prompt_blocks(&scores, &cfg, 5, 3);
        // First 5 tokens always included
        assert!(selected.contains(&0));
        assert!(selected.contains(&4));
        // Last 3 tokens always included
        assert!(selected.contains(&97));
        assert!(selected.contains(&99));
    }

    #[test]
    fn test_should_compress_modes() {
        assert!(!should_compress(PrefillMode::Off, 1000, 100));
        assert!(should_compress(PrefillMode::Always, 10, 100));
        assert!(should_compress(PrefillMode::Auto, 200, 100));
        assert!(!should_compress(PrefillMode::Auto, 50, 100));
    }
}
