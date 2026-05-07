use crate::speculative::verifier::{SimulatedVerifier, SpeculativeVerifier};
use crate::transformer::TransformerWeights;
use crate::types::{Config, Rng};

#[cfg(feature = "rest")]
use crate::rest::{RestClient, RetrievalResult};

// Shared imports for rest and leviathan features
#[cfg(feature = "rest")]
use crate::speculative::dd_tree::merge_retrieved_branches;
#[cfg(any(feature = "rest", feature = "leviathan"))]
use crate::speculative::dd_tree::{build_dd_tree, extract_best_path};
#[cfg(any(feature = "rest", feature = "leviathan"))]
use crate::speculative::dflash::dflash_predict;
#[cfg(feature = "leviathan")]
use crate::speculative::dflash::dflash_predict_conditioned;
#[cfg(any(feature = "rest", feature = "leviathan"))]
use crate::speculative::sampling::sample_from_distribution;
#[cfg(feature = "leviathan")]
use crate::speculative::sampling::sample_residual_distribution;
#[cfg(any(feature = "rest", feature = "leviathan"))]
use crate::transformer::{ForwardContext, MultiLayerKVCache, forward};
#[cfg(feature = "leviathan")]
use crate::types::softmax;

// Zero-alloc _with imports
#[cfg(feature = "leviathan")]
use crate::speculative::dd_tree::TreeBuilder;
#[cfg(feature = "leviathan")]
use crate::speculative::dflash::{dflash_predict_conditioned_with, dflash_predict_with};
#[cfg(feature = "leviathan")]
use crate::speculative::sampling::sample_residual_distribution_into;
#[cfg(feature = "leviathan")]
use crate::speculative::types::{DDTreeBranchCache, NoPruner, SpeculativeContext};

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
    let mut verifier = SimulatedVerifier::new(0.75, draft_config);
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
    let mv: Vec<&[f32]> = marginals.iter().map(|s| s.as_slice()).collect();

    // 2. Build initial DDTree
    let mut tree = build_dd_tree(&mv, draft_config);

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
        &mv,
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

// ── Leviathan: KV Rollback + Conditioned Draft ────────────────

/// Speculative step with KV-Cache snapshot/rollback for tree verification.
///
/// Builds a DDTree from draft marginals, then verifies multiple candidate
/// paths against the target model using p/q rejection sampling. Before each
/// path attempt, snapshots the target KV cache. On rejection, rolls back to
/// the snapshot and tries the next branch — avoids re-running target from scratch.
///
/// Snapshot cost: O(n_layer × pos × kv_dim) — cheap at our model scale.
///
/// Requires `--features leviathan` (target model forward pass needed).
#[cfg(feature = "leviathan")]
#[allow(clippy::too_many_arguments)]
pub fn speculative_step_rollback(
    draft_weights: &TransformerWeights,
    draft_config: &Config,
    target_weights: &TransformerWeights,
    target_config: &Config,
    target_ctx: &mut ForwardContext,
    target_cache: &mut MultiLayerKVCache,
    token: usize,
    pos: usize,
    rng: &mut Rng,
) -> (Vec<usize>, usize) {
    // 1. Draft marginals via DFlash
    let marginals = dflash_predict(draft_weights, draft_config, token, pos);
    let mv: Vec<&[f32]> = marginals.iter().map(|s| s.as_slice()).collect();

    // 2. Build DDTree
    let tree = build_dd_tree(&mv, draft_config);

    // 3. Extract candidate paths (top-3 root branches)
    let paths = extract_ddtree_paths(&tree);

    if paths.is_empty() {
        let fallback = sample_from_distribution(
            marginals.first().map(|m| m.as_slice()).unwrap_or(&[1.0]),
            rng,
        );
        return (vec![fallback], 1);
    }

    // 4. Snapshot target KV cache at current position
    let snapshot = target_cache.snapshot(pos, target_config);

    // 5. Try each candidate path with rollback on rejection
    for path in &paths {
        target_cache.restore(&snapshot, target_config);

        let mut accepted = Vec::with_capacity(path.len());
        let mut all_accepted = true;

        // Score initial token with target
        let logits = forward(
            target_ctx,
            target_weights,
            target_cache,
            token,
            pos,
            target_config,
        );
        let mut p_dist = logits.to_vec();
        for p in p_dist.iter_mut() {
            *p /= target_config.temperature;
        }
        softmax(&mut p_dist);

        for (i, &draft_tok) in path.iter().enumerate() {
            let q_dist = marginals.get(i).map(|m| m.as_slice()).unwrap_or(&[]);
            let q_i = q_dist.get(draft_tok).copied().unwrap_or(0.0);
            let p_i = p_dist.get(draft_tok).copied().unwrap_or(0.0);

            let acceptance_prob = if q_i > 0.0 { (p_i / q_i).min(1.0) } else { 1.0 };

            if rng.uniform() <= acceptance_prob {
                accepted.push(draft_tok);
                if i + 1 < path.len() {
                    let logits = forward(
                        target_ctx,
                        target_weights,
                        target_cache,
                        draft_tok,
                        pos + 1 + i,
                        target_config,
                    );
                    p_dist = logits.to_vec();
                    for p in p_dist.iter_mut() {
                        *p /= target_config.temperature;
                    }
                    softmax(&mut p_dist);
                }
            } else {
                let replacement = sample_residual_distribution(&p_dist, q_dist, rng);
                accepted.push(replacement);
                all_accepted = false;
                break;
            }
        }

        if all_accepted && !p_dist.is_empty() {
            let bonus = sample_from_distribution(&p_dist, rng);
            accepted.push(bonus);
        }

        if !accepted.is_empty() {
            let len = accepted.len();
            return (accepted, len);
        }
    }

    // All paths exhausted: restore and sample from target
    target_cache.restore(&snapshot, target_config);
    let logits = forward(
        target_ctx,
        target_weights,
        target_cache,
        token,
        pos,
        target_config,
    );
    let mut p_dist = logits.to_vec();
    for p in p_dist.iter_mut() {
        *p /= target_config.temperature;
    }
    softmax(&mut p_dist);
    let fallback = sample_from_distribution(&p_dist, rng);
    (vec![fallback], 1)
}

/// Speculative step with target-conditioned draft (DFlash-inspired).
///
/// Runs target model forward to capture hidden state, then conditions the
/// draft model's KV cache via `dflash_predict_conditioned`. The draft sees
/// target features, producing higher-quality marginals. Uses simulated
/// acceptance (no real p/q verification).
///
/// Requires `--features leviathan` (target model forward pass needed).
#[cfg(feature = "leviathan")]
#[allow(clippy::too_many_arguments)]
pub fn speculative_step_conditioned(
    draft_weights: &TransformerWeights,
    draft_config: &Config,
    target_weights: &TransformerWeights,
    target_config: &Config,
    target_ctx: &mut ForwardContext,
    target_cache: &mut MultiLayerKVCache,
    token: usize,
    pos: usize,
    rng: &mut Rng,
) -> (Vec<usize>, usize) {
    // 1. Run target forward to get hidden state
    let logits = forward(
        target_ctx,
        target_weights,
        target_cache,
        token,
        pos,
        target_config,
    )
    .to_vec(); // clone to release mutable borrow on target_ctx
    let hidden = target_ctx.hidden_state.clone();

    // 2. Conditioned draft using target hidden state
    let draft_result =
        dflash_predict_conditioned(draft_weights, draft_config, token, pos, &hidden, rng);

    // 3. Build DDTree from conditioned marginals
    let mv: Vec<&[f32]> = draft_result
        .marginals
        .iter()
        .map(|s| s.as_slice())
        .collect();
    let tree = build_dd_tree(&mv, draft_config);
    let path = extract_best_path(&tree);

    if path.is_empty() {
        let mut p_dist = logits.to_vec();
        for p in p_dist.iter_mut() {
            *p /= target_config.temperature;
        }
        softmax(&mut p_dist);
        let fallback = sample_from_distribution(&p_dist, rng);
        return (vec![fallback], 1);
    }

    // 4. Simulated acceptance (75% cap)
    let acceptance_rate = 0.75;
    let max_accept = ((path.len() as f32) * acceptance_rate).ceil() as usize;
    let accepted: Vec<usize> = path.into_iter().take(max_accept.max(1)).collect();

    // 5. Bonus token if all accepted
    if accepted.len() == max_accept && !draft_result.marginals.is_empty() {
        let last_marginal = draft_result.marginals.last().unwrap();
        let bonus = sample_from_distribution(last_marginal, rng);
        let mut result = accepted;
        result.push(bonus);
        let len = result.len();
        return (result, len);
    }

    let len = accepted.len();
    (accepted, len)
}

/// Zero-alloc variant of [`speculative_step_rollback`].
///
/// Reuses pre-allocated buffers from `SpeculativeContext`, `TreeBuilder`,
/// `probs_buf`, and `residual_buf` to minimize allocations in the hot path.
#[cfg(feature = "leviathan")]
#[allow(clippy::too_many_arguments)]
pub fn speculative_step_rollback_with(
    draft_sctx: &mut SpeculativeContext,
    tree_builder: &mut TreeBuilder,
    draft_weights: &TransformerWeights,
    draft_config: &Config,
    target_weights: &TransformerWeights,
    target_config: &Config,
    target_ctx: &mut ForwardContext,
    target_cache: &mut MultiLayerKVCache,
    probs_buf: &mut [f32],
    residual_buf: &mut [f32],
    token: usize,
    pos: usize,
    rng: &mut Rng,
) -> (Vec<usize>, usize) {
    // 1. Draft marginals via DFlash (zero-alloc into draft_sctx flat buffer)
    let num_steps = dflash_predict_with(draft_sctx, draft_weights, draft_config, token, pos);
    let vocab_size = draft_config.vocab_size;

    // Convert flat marginals to Vec<&[f32]> for tree builder (borrowed slices, no alloc)
    let marginals: Vec<&[f32]> = (0..num_steps)
        .map(|step| draft_sctx.marginal_slice(step, vocab_size))
        .collect();

    // 2. Build DDTree (reuses pre-allocated heap/tree buffers)
    let tree = tree_builder.build(&marginals, draft_config, &NoPruner, false);

    // 3. Extract candidate paths (top-3 root branches)
    let paths = extract_ddtree_paths(tree);

    if paths.is_empty() {
        let fallback = sample_from_distribution(marginals.first().copied().unwrap_or(&[1.0]), rng);
        return (vec![fallback], 1);
    }

    // 4. Snapshot target KV cache at current position
    let snapshot = target_cache.snapshot(pos, target_config);

    // 5. Try each candidate path with rollback on rejection
    for path in &paths {
        target_cache.restore(&snapshot, target_config);

        let mut accepted = Vec::with_capacity(path.len());
        let mut all_accepted = true;

        // Score initial token with target (zero-alloc: reuse probs_buf)
        let logits = forward(
            target_ctx,
            target_weights,
            target_cache,
            token,
            pos,
            target_config,
        );
        probs_buf.copy_from_slice(logits);
        for p in probs_buf.iter_mut() {
            *p /= target_config.temperature;
        }
        softmax(probs_buf);

        for (i, &draft_tok) in path.iter().enumerate() {
            let q_dist = marginals.get(i).copied().unwrap_or(&[]);
            let q_i = q_dist.get(draft_tok).copied().unwrap_or(0.0);
            let p_i = probs_buf.get(draft_tok).copied().unwrap_or(0.0);

            let acceptance_prob = if q_i > 0.0 { (p_i / q_i).min(1.0) } else { 1.0 };

            if rng.uniform() <= acceptance_prob {
                accepted.push(draft_tok);
                if i + 1 < path.len() {
                    let logits = forward(
                        target_ctx,
                        target_weights,
                        target_cache,
                        draft_tok,
                        pos + 1 + i,
                        target_config,
                    );
                    probs_buf.copy_from_slice(logits);
                    for p in probs_buf.iter_mut() {
                        *p /= target_config.temperature;
                    }
                    softmax(probs_buf);
                }
            } else {
                let replacement =
                    sample_residual_distribution_into(probs_buf, q_dist, residual_buf, rng);
                accepted.push(replacement);
                all_accepted = false;
                break;
            }
        }

        if all_accepted && !probs_buf.is_empty() {
            let bonus = sample_from_distribution(probs_buf, rng);
            accepted.push(bonus);
        }

        if !accepted.is_empty() {
            let len = accepted.len();
            return (accepted, len);
        }
    }

    // All paths exhausted: restore and sample from target
    target_cache.restore(&snapshot, target_config);
    let logits = forward(
        target_ctx,
        target_weights,
        target_cache,
        token,
        pos,
        target_config,
    );
    probs_buf.copy_from_slice(logits);
    for p in probs_buf.iter_mut() {
        *p /= target_config.temperature;
    }
    softmax(probs_buf);
    let fallback = sample_from_distribution(probs_buf, rng);
    (vec![fallback], 1)
}

/// Paged KV-cache variant of [`speculative_step_rollback`].
///
/// Uses `DDTreeBranchCache` (copy-on-write `PagedKVCache`) for the draft model's
/// KV exploration instead of snapshot/restore on `MultiLayerKVCache`. Shared prefix
/// pages are not copied — only new pages allocate after each fork point.
///
/// The target model still uses `MultiLayerKVCache` with snapshot/restore for
/// verification rollback (only a few candidate paths are verified).
///
/// Falls back to `speculative_step_rollback` behavior when branch budget is exhausted.
#[cfg(feature = "leviathan")]
#[allow(clippy::too_many_arguments)]
pub fn speculative_step_rollback_paged(
    draft_weights: &TransformerWeights,
    draft_config: &Config,
    target_weights: &TransformerWeights,
    target_config: &Config,
    target_ctx: &mut ForwardContext,
    target_cache: &mut MultiLayerKVCache,
    branch_cache: &mut DDTreeBranchCache,
    draft_ctx: &mut ForwardContext,
    token: usize,
    pos: usize,
    rng: &mut Rng,
) -> (Vec<usize>, usize) {
    // 1. Draft marginals via DFlash
    let marginals = dflash_predict(draft_weights, draft_config, token, pos);
    let mv: Vec<&[f32]> = marginals.iter().map(|s| s.as_slice()).collect();

    // 2. Build DDTree
    let tree = build_dd_tree(&mv, draft_config);

    // 3. Extract candidate paths (top-3 root branches)
    let paths = extract_ddtree_paths(&tree);

    if paths.is_empty() {
        let fallback = sample_from_distribution(
            marginals.first().map(|m| m.as_slice()).unwrap_or(&[1.0]),
            rng,
        );
        return (vec![fallback], 1);
    }

    // 4. Explore branches using paged KV cache
    // Run trunk forward for the prompt token to populate shared prefix
    branch_cache.reset();
    let _ = branch_cache.forward_branch(draft_ctx, draft_weights, 0, token, pos, draft_config);

    // Fork branches from the trunk at current position
    for (path_idx, path) in paths.iter().enumerate() {
        if path_idx == 0 {
            // First path continues on trunk (seq 0)
            for (depth, &tok) in path.iter().enumerate() {
                let _ = branch_cache.forward_branch(
                    draft_ctx,
                    draft_weights,
                    0,
                    tok,
                    pos + 1 + depth,
                    draft_config,
                );
            }
        } else {
            // Subsequent paths fork from trunk
            let branch_seq = branch_cache.fork_branch(0, pos + 1);
            for (depth, &tok) in path.iter().enumerate() {
                let _ = branch_cache.forward_branch(
                    draft_ctx,
                    draft_weights,
                    branch_seq,
                    tok,
                    pos + 1 + depth,
                    draft_config,
                );
            }
        }
    }

    // 5. Snapshot target KV cache at current position for verification rollback
    let snapshot = target_cache.snapshot(pos, target_config);

    // 6. Verify candidate paths against target model (same as speculative_step_rollback)
    for path in &paths {
        target_cache.restore(&snapshot, target_config);

        let mut accepted = Vec::with_capacity(path.len());
        let mut all_accepted = true;

        // Score initial token with target
        let logits = forward(
            target_ctx,
            target_weights,
            target_cache,
            token,
            pos,
            target_config,
        );
        let mut p_dist = logits.to_vec();
        for p in p_dist.iter_mut() {
            *p /= target_config.temperature;
        }
        softmax(&mut p_dist);

        for (i, &draft_tok) in path.iter().enumerate() {
            let q_dist = marginals.get(i).map(|m| m.as_slice()).unwrap_or(&[]);
            let q_i = q_dist.get(draft_tok).copied().unwrap_or(0.0);
            let p_i = p_dist.get(draft_tok).copied().unwrap_or(0.0);

            let acceptance_prob = if q_i > 0.0 { (p_i / q_i).min(1.0) } else { 1.0 };

            if rng.uniform() <= acceptance_prob {
                accepted.push(draft_tok);
                if i + 1 < path.len() {
                    let logits = forward(
                        target_ctx,
                        target_weights,
                        target_cache,
                        draft_tok,
                        pos + 1 + i,
                        target_config,
                    );
                    p_dist = logits.to_vec();
                    for p in p_dist.iter_mut() {
                        *p /= target_config.temperature;
                    }
                    softmax(&mut p_dist);
                }
            } else {
                let replacement = sample_residual_distribution(&p_dist, q_dist, rng);
                accepted.push(replacement);
                all_accepted = false;
                break;
            }
        }

        if all_accepted && !p_dist.is_empty() {
            let bonus = sample_from_distribution(&p_dist, rng);
            accepted.push(bonus);
        }

        if !accepted.is_empty() {
            let len = accepted.len();
            return (accepted, len);
        }
    }

    // All paths exhausted: restore and sample from target
    target_cache.restore(&snapshot, target_config);
    let logits = forward(
        target_ctx,
        target_weights,
        target_cache,
        token,
        pos,
        target_config,
    );
    let mut p_dist = logits.to_vec();
    for p in p_dist.iter_mut() {
        *p /= target_config.temperature;
    }
    softmax(&mut p_dist);
    let fallback = sample_from_distribution(&p_dist, rng);
    (vec![fallback], 1)
}

/// Zero-alloc variant of [`speculative_step_conditioned`].
///
/// Reuses pre-allocated buffers from `SpeculativeContext`, `TreeBuilder`,
/// and `probs_buf` to minimize allocations in the hot path.
#[cfg(feature = "leviathan")]
#[allow(clippy::too_many_arguments)]
pub fn speculative_step_conditioned_with(
    draft_sctx: &mut SpeculativeContext,
    tree_builder: &mut TreeBuilder,
    draft_weights: &TransformerWeights,
    draft_config: &Config,
    target_weights: &TransformerWeights,
    target_config: &Config,
    target_ctx: &mut ForwardContext,
    target_cache: &mut MultiLayerKVCache,
    probs_buf: &mut [f32],
    token: usize,
    pos: usize,
    rng: &mut Rng,
) -> (Vec<usize>, usize) {
    // 1. Run target forward to get logits (zero-alloc: copy into probs_buf)
    let logits = forward(
        target_ctx,
        target_weights,
        target_cache,
        token,
        pos,
        target_config,
    );
    probs_buf.copy_from_slice(logits);
    for p in probs_buf.iter_mut() {
        *p /= target_config.temperature;
    }
    softmax(probs_buf);

    // 2. Conditioned draft using target hidden state (no clone — borrow directly)
    let hidden = &target_ctx.hidden_state;
    let num_steps = dflash_predict_conditioned_with(
        draft_sctx,
        draft_weights,
        draft_config,
        token,
        pos,
        hidden,
        rng,
    );
    let vocab_size = draft_config.vocab_size;

    // Convert flat marginals to Vec<&[f32]> for tree builder (borrowed slices, no alloc)
    let marginals: Vec<&[f32]> = (0..num_steps)
        .map(|step| draft_sctx.marginal_slice(step, vocab_size))
        .collect();

    // Pre-extract marginals info before releasing immutable borrow on draft_sctx
    let has_marginals = !marginals.is_empty();
    let last_marginal = marginals.last().copied().unwrap_or(&[]);

    // 3. Build DDTree (reuses pre-allocated heap/tree buffers)
    let tree = tree_builder.build(&marginals, draft_config, &NoPruner, false);

    // Extract best path (small Vec alloc acceptable — avoids borrow conflict with marginals)
    let path = extract_best_path(tree);

    if path.is_empty() {
        let fallback = sample_from_distribution(probs_buf, rng);
        return (vec![fallback], 1);
    }

    // 4. Simulated acceptance (75% cap)
    let acceptance_rate = 0.75;
    let max_accept = ((path.len() as f32) * acceptance_rate).ceil() as usize;
    let accepted: Vec<usize> = path.into_iter().take(max_accept.max(1)).collect();

    // 5. Bonus token if all accepted
    if accepted.len() == max_accept && has_marginals {
        let bonus = sample_from_distribution(last_marginal, rng);
        let mut result = accepted;
        result.push(bonus);
        let len = result.len();
        return (result, len);
    }

    let len = accepted.len();
    (accepted, len)
}

/// Extract candidate verification paths from DDTree (top-3 root branches).
/// Each branch follows the best child at subsequent depths.
#[cfg(feature = "leviathan")]
fn extract_ddtree_paths(tree: &[crate::speculative::types::TreeNode]) -> Vec<Vec<usize>> {
    if tree.is_empty() {
        return Vec::new();
    }

    let max_depth = tree.iter().map(|n| n.depth).max().unwrap_or(0);

    // Collect root nodes (depth 0), sorted by score descending
    let mut roots: Vec<_> = tree.iter().filter(|n| n.depth == 0).collect();
    roots.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    roots.truncate(3);

    let mut paths = Vec::with_capacity(roots.len());

    for root in roots {
        let mut path = vec![root.token_idx];
        let mut current_path = root.parent_path;

        for depth in 1..=max_depth {
            let child = tree
                .iter()
                .filter(|n| n.depth == depth && n.parent_path >> 16 == current_path)
                .max_by_key(|n| (n.score * 1e6) as i64);

            match child {
                Some(node) => {
                    path.push(node.token_idx);
                    current_path = node.parent_path;
                }
                None => break,
            }
        }

        paths.push(path);
    }

    paths
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
        let mut verifier = SimulatedVerifier::new(0.75, &config);
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
            let mut verifier = SimulatedVerifier::new(0.75, &config);
            speculative_step_verifier(&weights, &config, 0, 0, &mut Rng::new(77), &mut verifier)
        };
        let (a2, l2) = {
            let mut verifier = SimulatedVerifier::new(0.75, &config);
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
            let mut verifier = SimulatedVerifier::new(0.95, &config);
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

    // ── Leviathan: Rollback + Conditioned Draft Tests ─────────────

    #[cfg(feature = "leviathan")]
    #[test]
    fn test_speculative_step_rollback_returns_at_least_one() {
        let target_config = Config::micro();
        let draft_config = Config::draft();
        let mut rng = Rng::new(42);
        let target_weights = TransformerWeights::new(&target_config, &mut rng);
        let mut draft_rng = Rng::new(99);
        let draft_weights = TransformerWeights::new(&draft_config, &mut draft_rng);

        let mut target_ctx = ForwardContext::new(&target_config);
        let mut target_cache = MultiLayerKVCache::new(&target_config);

        let (accepted, len) = speculative_step_rollback(
            &draft_weights,
            &draft_config,
            &target_weights,
            &target_config,
            &mut target_ctx,
            &mut target_cache,
            0,
            0,
            &mut Rng::new(100),
        );

        assert!(!accepted.is_empty(), "should return at least 1 token");
        assert!(len >= 1);
        for &t in &accepted {
            assert!(t < target_config.vocab_size, "token {t} out of range");
        }
    }

    #[cfg(feature = "leviathan")]
    #[test]
    fn test_speculative_step_rollback_deterministic() {
        let target_config = Config::micro();
        let draft_config = Config::draft();
        let mut rng = Rng::new(42);
        let target_weights = TransformerWeights::new(&target_config, &mut rng);
        let mut draft_rng = Rng::new(99);
        let draft_weights = TransformerWeights::new(&draft_config, &mut draft_rng);

        let (a1, l1) = {
            let mut target_ctx = ForwardContext::new(&target_config);
            let mut target_cache = MultiLayerKVCache::new(&target_config);
            speculative_step_rollback(
                &draft_weights,
                &draft_config,
                &target_weights,
                &target_config,
                &mut target_ctx,
                &mut target_cache,
                0,
                0,
                &mut Rng::new(100),
            )
        };

        let (a2, l2) = {
            let mut target_ctx = ForwardContext::new(&target_config);
            let mut target_cache = MultiLayerKVCache::new(&target_config);
            speculative_step_rollback(
                &draft_weights,
                &draft_config,
                &target_weights,
                &target_config,
                &mut target_ctx,
                &mut target_cache,
                0,
                0,
                &mut Rng::new(100),
            )
        };

        assert_eq!(a1, a2, "same seed should produce same results");
        assert_eq!(l1, l2);
    }

    #[cfg(feature = "leviathan")]
    #[test]
    fn test_speculative_step_conditioned_returns_at_least_one() {
        let target_config = Config::micro();
        let draft_config = Config::draft();
        let mut rng = Rng::new(42);
        let target_weights = TransformerWeights::new(&target_config, &mut rng);
        let mut draft_rng = Rng::new(99);
        let draft_weights = TransformerWeights::new(&draft_config, &mut draft_rng);

        let mut target_ctx = ForwardContext::new(&target_config);
        let mut target_cache = MultiLayerKVCache::new(&target_config);

        let (accepted, len) = speculative_step_conditioned(
            &draft_weights,
            &draft_config,
            &target_weights,
            &target_config,
            &mut target_ctx,
            &mut target_cache,
            0,
            0,
            &mut Rng::new(100),
        );

        assert!(!accepted.is_empty(), "should return at least 1 token");
        assert!(len >= 1);
        for &t in &accepted {
            assert!(t < target_config.vocab_size, "token {t} out of range");
        }
    }

    #[cfg(feature = "leviathan")]
    #[test]
    fn test_speculative_step_conditioned_deterministic() {
        let target_config = Config::micro();
        let draft_config = Config::draft();
        let mut rng = Rng::new(42);
        let target_weights = TransformerWeights::new(&target_config, &mut rng);
        let mut draft_rng = Rng::new(99);
        let draft_weights = TransformerWeights::new(&draft_config, &mut draft_rng);

        let (a1, l1) = {
            let mut target_ctx = ForwardContext::new(&target_config);
            let mut target_cache = MultiLayerKVCache::new(&target_config);
            speculative_step_conditioned(
                &draft_weights,
                &draft_config,
                &target_weights,
                &target_config,
                &mut target_ctx,
                &mut target_cache,
                0,
                0,
                &mut Rng::new(100),
            )
        };

        let (a2, l2) = {
            let mut target_ctx = ForwardContext::new(&target_config);
            let mut target_cache = MultiLayerKVCache::new(&target_config);
            speculative_step_conditioned(
                &draft_weights,
                &draft_config,
                &target_weights,
                &target_config,
                &mut target_ctx,
                &mut target_cache,
                0,
                0,
                &mut Rng::new(100),
            )
        };

        assert_eq!(a1, a2, "same seed should produce same results");
        assert_eq!(l1, l2);
    }

    #[cfg(feature = "leviathan")]
    #[test]
    fn test_speculative_step_conditioned_differs_from_unconditioned() {
        let target_config = Config::micro();
        let draft_config = Config::draft();
        let mut rng = Rng::new(42);
        let target_weights = TransformerWeights::new(&target_config, &mut rng);
        let mut draft_rng = Rng::new(99);
        let draft_weights = TransformerWeights::new(&draft_config, &mut draft_rng);

        let (cond_accepted, _) = {
            let mut target_ctx = ForwardContext::new(&target_config);
            let mut target_cache = MultiLayerKVCache::new(&target_config);
            speculative_step_conditioned(
                &draft_weights,
                &draft_config,
                &target_weights,
                &target_config,
                &mut target_ctx,
                &mut target_cache,
                0,
                0,
                &mut Rng::new(100),
            )
        };

        let (uncond_accepted, _) = {
            let mut verifier = SimulatedVerifier::new(0.75, &draft_config);
            speculative_step_verifier(
                &draft_weights,
                &draft_config,
                0,
                0,
                &mut Rng::new(100),
                &mut verifier,
            )
        };

        assert_ne!(
            cond_accepted, uncond_accepted,
            "conditioned draft should differ from unconditioned"
        );
    }

    #[cfg(feature = "leviathan")]
    #[test]
    fn test_extract_ddtree_paths() {
        let (weights, config) = make_draft();
        let marginals = dflash_predict(&weights, &config, 0, 0);
        let mv: Vec<&[f32]> = marginals.iter().map(|s| s.as_slice()).collect();
        let tree = build_dd_tree(&mv, &config);
        let paths = extract_ddtree_paths(&tree);

        if !tree.is_empty() {
            assert!(!paths.is_empty(), "non-empty tree should produce paths");
            for path in &paths {
                assert!(!path.is_empty(), "each path should have at least one token");
                for &t in path {
                    assert!(t < config.vocab_size, "token {t} out of range");
                }
            }
        }
    }

    #[cfg(feature = "leviathan")]
    #[test]
    fn test_speculative_step_rollback_paged_returns_at_least_one() {
        let target_config = Config::micro();
        let draft_config = Config::draft();
        let mut rng = Rng::new(42);
        let target_weights = TransformerWeights::new(&target_config, &mut rng);
        let mut draft_rng = Rng::new(99);
        let draft_weights = TransformerWeights::new(&draft_config, &mut draft_rng);

        let mut target_ctx = ForwardContext::new(&target_config);
        let mut target_cache = MultiLayerKVCache::new(&target_config);
        let mut branch_cache = DDTreeBranchCache::new(&draft_config, 8);
        let mut draft_ctx = ForwardContext::new(&draft_config);

        let (accepted, len) = speculative_step_rollback_paged(
            &draft_weights,
            &draft_config,
            &target_weights,
            &target_config,
            &mut target_ctx,
            &mut target_cache,
            &mut branch_cache,
            &mut draft_ctx,
            0,
            0,
            &mut Rng::new(100),
        );

        assert!(!accepted.is_empty(), "should return at least 1 token");
        assert!(len >= 1);
        for &t in &accepted {
            assert!(t < target_config.vocab_size, "token {t} out of range");
        }
    }

    #[cfg(feature = "leviathan")]
    #[test]
    fn test_speculative_step_rollback_paged_deterministic() {
        let target_config = Config::micro();
        let draft_config = Config::draft();
        let mut rng = Rng::new(42);
        let target_weights = TransformerWeights::new(&target_config, &mut rng);
        let mut draft_rng = Rng::new(99);
        let draft_weights = TransformerWeights::new(&draft_config, &mut draft_rng);

        let (a1, l1) = {
            let mut target_ctx = ForwardContext::new(&target_config);
            let mut target_cache = MultiLayerKVCache::new(&target_config);
            let mut branch_cache = DDTreeBranchCache::new(&draft_config, 8);
            let mut draft_ctx = ForwardContext::new(&draft_config);
            speculative_step_rollback_paged(
                &draft_weights,
                &draft_config,
                &target_weights,
                &target_config,
                &mut target_ctx,
                &mut target_cache,
                &mut branch_cache,
                &mut draft_ctx,
                0,
                0,
                &mut Rng::new(100),
            )
        };

        let (a2, l2) = {
            let mut target_ctx = ForwardContext::new(&target_config);
            let mut target_cache = MultiLayerKVCache::new(&target_config);
            let mut branch_cache = DDTreeBranchCache::new(&draft_config, 8);
            let mut draft_ctx = ForwardContext::new(&draft_config);
            speculative_step_rollback_paged(
                &draft_weights,
                &draft_config,
                &target_weights,
                &target_config,
                &mut target_ctx,
                &mut target_cache,
                &mut branch_cache,
                &mut draft_ctx,
                0,
                0,
                &mut Rng::new(100),
            )
        };

        assert_eq!(a1, a2, "same seed should produce same results");
        assert_eq!(l1, l2);
    }

    #[cfg(feature = "leviathan")]
    #[test]
    fn test_speculative_step_rollback_paged_matches_flat_results() {
        // Both paged and flat should produce at least 1 valid token
        let target_config = Config::micro();
        let draft_config = Config::draft();
        let mut rng = Rng::new(42);
        let target_weights = TransformerWeights::new(&target_config, &mut rng);
        let mut draft_rng = Rng::new(99);
        let draft_weights = TransformerWeights::new(&draft_config, &mut draft_rng);

        for seed in [0u64, 42, 100, 999] {
            // Paged variant
            let (paged_accepted, paged_len) = {
                let mut target_ctx = ForwardContext::new(&target_config);
                let mut target_cache = MultiLayerKVCache::new(&target_config);
                let mut branch_cache = DDTreeBranchCache::new(&draft_config, 8);
                let mut draft_ctx = ForwardContext::new(&draft_config);
                speculative_step_rollback_paged(
                    &draft_weights,
                    &draft_config,
                    &target_weights,
                    &target_config,
                    &mut target_ctx,
                    &mut target_cache,
                    &mut branch_cache,
                    &mut draft_ctx,
                    0,
                    0,
                    &mut Rng::new(seed),
                )
            };

            assert!(
                !paged_accepted.is_empty(),
                "seed {seed}: paged should return at least 1 token"
            );
            assert!(paged_len >= 1, "seed {seed}: paged len should be >= 1");
            for &t in &paged_accepted {
                assert!(
                    t < target_config.vocab_size,
                    "seed {seed}: paged token {t} out of range"
                );
            }
        }
    }
}
