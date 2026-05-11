use std::collections::BinaryHeap;

#[cfg(test)]
use super::types::BinaryScreeningPruner;
#[cfg(test)]
use super::types::NoScreeningPruner;
use super::types::{ConstraintPruner, NoPruner, ScreeningPruner, TreeNode};
use rayon::prelude::*;

/// Extract tokens from `parent_path` bitfield for path-aware pruning.
///
/// `parent_path` uses 5 bits per depth, packed LSB-first:
/// - Depth 0 token: bits 0–4
/// - Depth 1 token: bits 5–9
/// - ...
/// - Depth k token: bits (k*5) to (k*5+4)
///
/// Returns `Vec<usize>` where `result[k]` = token at depth `k`.
/// Max depths: 64/5 = 12 (sufficient for lookahead of 5–8).
pub fn extract_parent_tokens(parent_path: u128, num_tokens: usize) -> Vec<usize> {
    // parent_path packs tokens with most-recent in lowest bits:
    //   depth 0 token → bits (num_tokens-1)*16 .. (num_tokens-1)*16+15
    //   depth k token → bits (num_tokens-1-k)*16 .. (num_tokens-1-k)*16+15
    (0..num_tokens)
        .map(|k| ((parent_path >> ((num_tokens - 1 - k) * 16)) & 0xFFFF) as usize)
        .collect()
}

/// Zero-alloc variant of [`extract_parent_tokens`].
/// Writes `num_tokens` parent tokens into `buf`, which must be large enough.
/// Returns the slice `&buf[..num_tokens]`.
#[inline]
pub fn extract_parent_tokens_into(
    parent_path: u128,
    num_tokens: usize,
    buf: &mut [usize],
) -> &[usize] {
    for (k, slot) in buf.iter_mut().enumerate().take(num_tokens) {
        *slot = ((parent_path >> ((num_tokens - 1 - k) * 16)) & 0xFFFF) as usize;
    }
    &buf[..num_tokens]
}

/// DDTree: Build verification tree from marginals using Best-First Search.
/// Returns tree nodes ordered by score (best first).
///
/// Equivalent to `build_dd_tree_pruned` with `NoPruner` and `chain_seed=false`.
///
/// # Branch Ordering Preserves Reasoning Sequence (Plan 029)
///
/// Each DDTree branch stores tokens in `parent_path` as an **ordered sequence**,
/// preserving the exact order the draft model produced them. This is critical for
/// agentic inference where reasoning and tool calls must remain interleaved:
///
/// ```text
/// CORRECT (DDTree preserves this):
///   reasoning_0 → tool_call_0 → reasoning_1 → tool_call_1
///
/// WRONG (would lose sequence meaning):
///   reasoning_0 → reasoning_1 → tool_call_0 → tool_call_1
/// ```
///
/// NVIDIA Dynamo found that grouping reasoning separate from tool calls increased
/// TTFT 1.9× (322ms vs 167ms on B200) because the target model couldn't associate
/// each tool call with its preceding reasoning. Our `extract_parent_tokens()` and
/// `extract_parent_tokens_into()` maintain this ordering per branch.
pub fn build_dd_tree(marginals: &[&[f32]], config: &crate::types::Config) -> Vec<TreeNode> {
    build_dd_tree_pruned(marginals, config, &NoPruner, false)
}

/// DDTree with Constraint Pruner: Build verification tree from marginals,
/// filtering branches through a deterministic rules engine.
///
/// When `chain_seed=true`, builds a greedy chain backbone first (argmax at
/// each depth with cumulative log-prob scores), then seeds the best-first
/// heap with siblings at each chain depth and children of the last chain
/// node. Standard best-first expansion fills the remaining budget.
///
/// When `chain_seed=false`, uses the original best-first algorithm.
///
/// The pruner is called for every candidate token at every depth.
/// Invalid tokens are never added to the heap — they don't waste tree budget.
///
/// This is the **Symbolic Validator intercept**: the draft model proposes
/// logits (semantic probability), the pruner enforces constraints
/// (mathematical validity), and only valid branches reach verification.
pub fn build_dd_tree_pruned(
    marginals: &[&[f32]],
    config: &crate::types::Config,
    pruner: &dyn ConstraintPruner,
    chain_seed: bool,
) -> Vec<TreeNode> {
    let mut builder = TreeBuilder::new(config);
    builder.build(marginals, config, pruner, chain_seed);
    std::mem::take(&mut builder.tree)
}

/// DDTree with Screening Pruner: Build verification tree from marginals,
/// blending LLM log-probabilities with absolute relevance scores.
///
/// This is the upgraded version of [`build_dd_tree_pruned`]. Instead of
/// binary valid/invalid, the [`ScreeningPruner`] returns `R ∈ [0.0, 1.0]`:
/// - `R = 1.0` → no penalty (`ln(1.0) = 0.0`)
/// - `0.0 < R < 1.0` → soft penalty (`ln(R)` added to score)
/// - `R ≤ threshold` → hard trim (branch killed, never added to heap)
///
/// Score formula: `blended = parent_score + ln(P_llm) + ln(R)`
///
/// The `screening_threshold` is read from `config.screening_threshold`.
/// When threshold is `0.0`, only `R == 0.0` triggers hard trim (pure softmask).
pub fn build_dd_tree_screened(
    marginals: &[&[f32]],
    config: &crate::types::Config,
    screener: &dyn ScreeningPruner,
    chain_seed: bool,
) -> Vec<TreeNode> {
    let mut builder = TreeBuilder::new(config);
    builder.build_screened(marginals, config, screener, chain_seed);
    std::mem::take(&mut builder.tree)
}

/// Zero-alloc variant of `extract_best_path`.
/// Writes best-scored token at each depth into `path` (cleared first).
pub fn extract_best_path_into(tree: &[TreeNode], path: &mut Vec<usize>) {
    path.clear();
    if tree.is_empty() {
        return;
    }
    let max_depth = tree.iter().map(|n| n.depth).max().unwrap_or(0);
    for depth in 0..=max_depth {
        let best = tree
            .iter()
            .filter(|n| n.depth == depth)
            .max_by_key(|n| (n.score * 1e6) as i64);
        match best {
            Some(node) => path.push(node.token_idx),
            None => break,
        }
    }
}

/// Extract best-scored token at each depth from a DDTree.
pub fn extract_best_path(tree: &[TreeNode]) -> Vec<usize> {
    if tree.is_empty() {
        return Vec::new();
    }
    let max_depth = tree.iter().map(|n| n.depth).max().unwrap_or(0);
    let mut path = Vec::with_capacity(max_depth + 1);
    for depth in 0..=max_depth {
        let best = tree
            .iter()
            .filter(|n| n.depth == depth)
            .max_by_key(|n| (n.score * 1e6) as i64);
        match best {
            Some(node) => path.push(node.token_idx),
            None => break,
        }
    }
    path
}

/// Inject retrieved token sequences into the DDTree as candidate branches.
///
/// Each retrieved sequence becomes a path with blended score.
/// Score blending: `(1-w) * log(draft_prob) + w * log(similarity)`
///
/// This is a pure computation function — no feature gating needed.
/// The REST feature provides the data; this function processes it.
pub fn merge_retrieved_branches(
    tree: &mut Vec<TreeNode>,
    marginals: &[&[f32]],
    config: &crate::types::Config,
    token_sequences: &[Vec<usize>],
    scores: &[f32],
    rest_weight: f32,
) {
    if token_sequences.is_empty() || rest_weight <= 0.0 {
        return;
    }

    let inv_weight = 1.0 - rest_weight;

    for (seq_idx, seq) in token_sequences.iter().enumerate() {
        let similarity = scores.get(seq_idx).copied().unwrap_or(0.0);
        if similarity <= 0.0 {
            continue;
        }

        for (depth, &token_idx) in seq.iter().enumerate() {
            if depth >= marginals.len() {
                break;
            }
            if token_idx >= config.vocab_size {
                break;
            }

            let base_prob = marginals[depth].get(token_idx).copied().unwrap_or(0.0);
            if base_prob <= 0.0 {
                continue;
            }

            let blended = (base_prob.ln() * inv_weight) + (similarity.ln() * rest_weight);

            // Reconstruct parent_path from sequence prefix up to current depth
            let parent_path = seq[..=depth]
                .iter()
                .enumerate()
                .fold(0u128, |acc, (d, &t)| {
                    if d == 0 {
                        t as u128
                    } else {
                        (acc << 16) | (t as u128)
                    }
                });

            tree.push(TreeNode {
                score: blended,
                depth,
                token_idx,
                parent_path,
            });
        }
    }

    // Re-sort by score descending
    tree.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    tree.truncate(config.tree_budget);
}

/// Pre-allocated buffers for zero-alloc DDTree building.
///
/// Create once with `TreeBuilder::new(config)`, reuse across calls.
/// `build()` clears and reuses internal buffers — no allocation on steady state.
pub struct TreeBuilder {
    heap: BinaryHeap<TreeNode>,
    tree: Vec<TreeNode>,
    chain_nodes: Vec<TreeNode>,
    chain_parent_tokens: Vec<usize>,
    parent_tokens_buf: Vec<usize>,
}

impl TreeBuilder {
    /// Allocate all buffers from config dimensions.
    pub fn new(config: &crate::types::Config) -> Self {
        Self {
            heap: BinaryHeap::new(),
            tree: Vec::with_capacity(config.tree_budget),
            chain_nodes: Vec::with_capacity(config.draft_lookahead),
            chain_parent_tokens: Vec::with_capacity(config.draft_lookahead),
            parent_tokens_buf: vec![0usize; config.draft_lookahead + 1],
        }
    }

    /// Build DDTree from marginals, reusing pre-allocated buffers.
    ///
    /// Clears and reuses `heap`, `tree`, `chain_nodes`, `chain_parent_tokens`.
    /// Returns a borrowed slice valid until the next `build()` call.
    pub fn build(
        &mut self,
        marginals: &[&[f32]],
        config: &crate::types::Config,
        pruner: &dyn ConstraintPruner,
        chain_seed: bool,
    ) -> &[TreeNode] {
        self.heap.clear();
        self.tree.clear();
        self.chain_nodes.clear();
        self.chain_parent_tokens.clear();

        if marginals.is_empty() {
            return &self.tree;
        }

        if chain_seed {
            // ── Phase A: Build greedy chain backbone ──────────────
            let mut cumulative_score: f32 = 0.0;
            let mut parent_path: u128 = 0;

            for (depth, marginal) in marginals.iter().enumerate() {
                if self.tree.len() >= config.tree_budget {
                    break;
                }

                // Find argmax token at this depth
                let best_token = marginal
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i);

                let Some(token_idx) = best_token else {
                    break;
                };
                let prob = marginal[token_idx];

                // Chain breaks if argmax has zero prob or is pruned
                if prob <= 0.0 || !pruner.is_valid(depth, token_idx, &self.chain_parent_tokens) {
                    break;
                }

                cumulative_score += prob.ln();
                let node_path = if depth == 0 {
                    token_idx as u128
                } else {
                    (parent_path << 16) | (token_idx as u128)
                };

                let node = TreeNode {
                    score: cumulative_score,
                    depth,
                    token_idx,
                    parent_path: node_path,
                };

                self.tree.push(node);
                self.chain_nodes.push(node);
                parent_path = node_path;
                self.chain_parent_tokens.push(token_idx);
            }

            // ── Phase B: Seed heap with siblings + last chain children ──
            if self.chain_nodes.is_empty() {
                // No chain built — fall back to original root seeding
                if config.vocab_size > 256 {
                    let nodes: Vec<TreeNode> = marginals[0]
                        .par_iter()
                        .enumerate()
                        .filter_map(|(i, &prob)| {
                            if prob > 0.0 && pruner.is_valid(0, i, &[]) {
                                Some(TreeNode {
                                    score: prob.ln(),
                                    depth: 0,
                                    token_idx: i,
                                    parent_path: i as u128,
                                })
                            } else {
                                None
                            }
                        })
                        .collect();
                    self.heap.extend(nodes);
                } else {
                    for (i, &prob) in marginals[0].iter().enumerate() {
                        if prob > 0.0 && pruner.is_valid(0, i, &[]) {
                            self.heap.push(TreeNode {
                                score: prob.ln(),
                                depth: 0,
                                token_idx: i,
                                parent_path: i as u128,
                            });
                        }
                    }
                }
            } else {
                // Seed siblings at each chain depth
                for chain_node in &self.chain_nodes {
                    let depth = chain_node.depth;
                    let parent_chain_score = if depth == 0 {
                        0.0f32
                    } else {
                        self.chain_nodes[depth - 1].score
                    };

                    // Parent tokens for pruning: chain tokens at depths 0..depth-1
                    let sibling_parent_tokens = extract_parent_tokens_into(
                        chain_node.parent_path >> 16,
                        depth,
                        &mut self.parent_tokens_buf,
                    );

                    for (i, &prob) in marginals[depth].iter().enumerate() {
                        if i == chain_node.token_idx {
                            continue;
                        }
                        if prob > 0.0 && pruner.is_valid(depth, i, sibling_parent_tokens) {
                            let sibling_path = if depth == 0 {
                                i as u128
                            } else {
                                let ancestor_path = chain_node.parent_path >> 16;
                                (ancestor_path << 16) | (i as u128)
                            };

                            self.heap.push(TreeNode {
                                score: parent_chain_score + prob.ln(),
                                depth,
                                token_idx: i,
                                parent_path: sibling_path,
                            });
                        }
                    }
                }

                // Seed children of the last chain node
                let last = self.chain_nodes.last().unwrap();
                if last.depth + 1 < marginals.len() {
                    let next_depth = last.depth + 1;
                    let parent_tokens = extract_parent_tokens_into(
                        last.parent_path,
                        last.depth + 1,
                        &mut self.parent_tokens_buf,
                    );
                    for (i, &prob) in marginals[next_depth].iter().enumerate() {
                        if prob > 0.0 && pruner.is_valid(next_depth, i, parent_tokens) {
                            self.heap.push(TreeNode {
                                score: last.score + prob.ln(),
                                depth: next_depth,
                                token_idx: i,
                                parent_path: (last.parent_path << 16) | (i as u128),
                            });
                        }
                    }
                }
            }
        } else {
            // Original behavior: seed heap with root's children, filtered by pruner
            if config.vocab_size > 256 {
                let nodes: Vec<TreeNode> = marginals[0]
                    .par_iter()
                    .enumerate()
                    .filter_map(|(i, &prob)| {
                        if prob > 0.0 && pruner.is_valid(0, i, &[]) {
                            Some(TreeNode {
                                score: prob.ln(),
                                depth: 0,
                                token_idx: i,
                                parent_path: i as u128,
                            })
                        } else {
                            None
                        }
                    })
                    .collect();
                self.heap.extend(nodes);
            } else {
                for (i, &prob) in marginals[0].iter().enumerate() {
                    if prob > 0.0 && pruner.is_valid(0, i, &[]) {
                        self.heap.push(TreeNode {
                            score: prob.ln(),
                            depth: 0,
                            token_idx: i,
                            parent_path: i as u128,
                        });
                    }
                }
            }
        }

        // ── Phase C: Standard best-first expansion ────────────────
        let mut best_score: Option<f32> = None;
        let mut second_best_score: Option<f32> = None;
        let mut consecutive_dominant: usize = 0;
        while self.tree.len() < config.tree_budget {
            let Some(best) = self.heap.pop() else {
                break;
            };
            self.tree.push(best);

            // Confidence-gap early exit (Plan 026: AutoTTS)
            let score = best.score;
            match best_score {
                None => {
                    best_score = Some(score);
                }
                Some(bs) if score > bs => {
                    second_best_score = Some(bs);
                    best_score = Some(score);
                    consecutive_dominant = 1;
                }
                Some(bs) => {
                    // Not a new best — track running second best (degrades with heap)
                    second_best_score = Some(score);
                    if bs - score > config.early_exit_gap {
                        consecutive_dominant += 1;
                    } else {
                        consecutive_dominant = 0;
                    }
                }
            }
            if config.early_exit_patience > 0
                && config.early_exit_gap > 0.0
                && consecutive_dominant >= config.early_exit_patience
                && best_score.unwrap_or(0.0) - second_best_score.unwrap_or(0.0)
                    > config.early_exit_gap
            {
                break;
            }

            if best.depth + 1 < marginals.len() {
                let next_depth = best.depth + 1;
                // Extract parent tokens from path bitfield for path-aware pruning
                let parent_tokens = extract_parent_tokens_into(
                    best.parent_path,
                    best.depth + 1,
                    &mut self.parent_tokens_buf,
                );
                for (i, &prob) in marginals[next_depth].iter().enumerate() {
                    // NEURO-SYMBOLIC INTERCEPT: prune before adding to heap
                    if prob > 0.0 && pruner.is_valid(next_depth, i, parent_tokens) {
                        self.heap.push(TreeNode {
                            score: best.score + prob.ln(),
                            depth: next_depth,
                            token_idx: i,
                            parent_path: (best.parent_path << 16) | (i as u128),
                        });
                    }
                }
            }
        }

        &self.tree
    }

    /// Build tree and merge retrieved branches in one call.
    ///
    /// For REST feature: builds the DDTree, then calls `merge_retrieved_branches`
    /// on the internal tree buffer. Returns a borrowed slice valid until the
    /// next `build()` or `build_and_merge()` call.
    #[allow(clippy::too_many_arguments)]
    pub fn build_and_merge(
        &mut self,
        marginals: &[&[f32]],
        config: &crate::types::Config,
        pruner: &dyn ConstraintPruner,
        chain_seed: bool,
        token_sequences: &[Vec<usize>],
        scores: &[f32],
        rest_weight: f32,
    ) -> &[TreeNode] {
        self.build(marginals, config, pruner, chain_seed);
        merge_retrieved_branches(
            &mut self.tree,
            marginals,
            config,
            token_sequences,
            scores,
            rest_weight,
        );
        &self.tree
    }

    /// Consume the builder and return the tree as an owned `Vec`.
    pub fn into_tree(self) -> Vec<TreeNode> {
        self.tree
    }

    /// Build DDTree with graded relevance screening (Plan 021).
    ///
    /// Like [`build()`] but uses [`ScreeningPruner`] for continuous relevance
    /// instead of binary [`ConstraintPruner`]. The relevance score `R ∈ [0.0, 1.0]`
    /// is blended into log-prob space: `score += ln(P_llm) + ln(R)`.
    ///
    /// Branches with `relevance <= config.screening_threshold` are hard-trimmed.
    pub fn build_screened(
        &mut self,
        marginals: &[&[f32]],
        config: &crate::types::Config,
        screener: &dyn ScreeningPruner,
        chain_seed: bool,
    ) -> &[TreeNode] {
        let threshold = config.screening_threshold;
        self.heap.clear();
        self.tree.clear();
        self.chain_nodes.clear();
        self.chain_parent_tokens.clear();

        if marginals.is_empty() {
            return &self.tree;
        }

        if chain_seed {
            // ── Phase A: Build greedy chain backbone with screening ──
            let mut cumulative_score: f32 = 0.0;
            let mut parent_path: u128 = 0;

            for (depth, marginal) in marginals.iter().enumerate() {
                if self.tree.len() >= config.tree_budget {
                    break;
                }

                let best_token = marginal
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i);

                let Some(token_idx) = best_token else {
                    break;
                };
                let prob = marginal[token_idx];

                if prob <= 0.0 {
                    break;
                }

                let relevance = screener.relevance(depth, token_idx, &self.chain_parent_tokens);
                if relevance <= threshold {
                    break;
                }

                // Blended score: ln(P_llm) + ln(R)
                cumulative_score += prob.ln() + relevance.ln();
                let node_path = if depth == 0 {
                    token_idx as u128
                } else {
                    (parent_path << 16) | (token_idx as u128)
                };

                let node = TreeNode {
                    score: cumulative_score,
                    depth,
                    token_idx,
                    parent_path: node_path,
                };

                self.tree.push(node);
                self.chain_nodes.push(node);
                parent_path = node_path;
                self.chain_parent_tokens.push(token_idx);
            }

            // ── Phase B: Seed heap with siblings + last chain children ──
            if self.chain_nodes.is_empty() {
                if config.vocab_size > 256 {
                    let nodes: Vec<TreeNode> = marginals[0]
                        .par_iter()
                        .enumerate()
                        .filter_map(|(i, &prob)| {
                            if prob <= 0.0 {
                                return None;
                            }
                            let relevance = screener.relevance(0, i, &[]);
                            if relevance <= threshold {
                                return None;
                            }
                            Some(TreeNode {
                                score: prob.ln() + relevance.ln(),
                                depth: 0,
                                token_idx: i,
                                parent_path: i as u128,
                            })
                        })
                        .collect();
                    self.heap.extend(nodes);
                } else {
                    for (i, &prob) in marginals[0].iter().enumerate() {
                        if prob <= 0.0 {
                            continue;
                        }
                        let relevance = screener.relevance(0, i, &[]);
                        if relevance <= threshold {
                            continue;
                        }
                        self.heap.push(TreeNode {
                            score: prob.ln() + relevance.ln(),
                            depth: 0,
                            token_idx: i,
                            parent_path: i as u128,
                        });
                    }
                }
            } else {
                for chain_node in &self.chain_nodes {
                    let depth = chain_node.depth;
                    let parent_chain_score = if depth == 0 {
                        0.0f32
                    } else {
                        self.chain_nodes[depth - 1].score
                    };

                    let sibling_parent_tokens = extract_parent_tokens_into(
                        chain_node.parent_path >> 16,
                        depth,
                        &mut self.parent_tokens_buf,
                    );

                    for (i, &prob) in marginals[depth].iter().enumerate() {
                        if i == chain_node.token_idx {
                            continue;
                        }
                        if prob <= 0.0 {
                            continue;
                        }
                        let relevance = screener.relevance(depth, i, sibling_parent_tokens);
                        if relevance <= threshold {
                            continue;
                        }
                        let sibling_path = if depth == 0 {
                            i as u128
                        } else {
                            let ancestor_path = chain_node.parent_path >> 16;
                            (ancestor_path << 16) | (i as u128)
                        };

                        self.heap.push(TreeNode {
                            score: parent_chain_score + prob.ln() + relevance.ln(),
                            depth,
                            token_idx: i,
                            parent_path: sibling_path,
                        });
                    }
                }

                let last = self.chain_nodes.last().unwrap();
                if last.depth + 1 < marginals.len() {
                    let next_depth = last.depth + 1;
                    let parent_tokens = extract_parent_tokens_into(
                        last.parent_path,
                        last.depth + 1,
                        &mut self.parent_tokens_buf,
                    );
                    for (i, &prob) in marginals[next_depth].iter().enumerate() {
                        if prob <= 0.0 {
                            continue;
                        }
                        let relevance = screener.relevance(next_depth, i, parent_tokens);
                        if relevance <= threshold {
                            continue;
                        }
                        self.heap.push(TreeNode {
                            score: last.score + prob.ln() + relevance.ln(),
                            depth: next_depth,
                            token_idx: i,
                            parent_path: (last.parent_path << 16) | (i as u128),
                        });
                    }
                }
            }
        } else {
            // Original seeding with screening
            if config.vocab_size > 256 {
                let nodes: Vec<TreeNode> = marginals[0]
                    .par_iter()
                    .enumerate()
                    .filter_map(|(i, &prob)| {
                        if prob <= 0.0 {
                            return None;
                        }
                        let relevance = screener.relevance(0, i, &[]);
                        if relevance <= threshold {
                            return None;
                        }
                        Some(TreeNode {
                            score: prob.ln() + relevance.ln(),
                            depth: 0,
                            token_idx: i,
                            parent_path: i as u128,
                        })
                    })
                    .collect();
                self.heap.extend(nodes);
            } else {
                for (i, &prob) in marginals[0].iter().enumerate() {
                    if prob <= 0.0 {
                        continue;
                    }
                    let relevance = screener.relevance(0, i, &[]);
                    if relevance <= threshold {
                        continue;
                    }
                    self.heap.push(TreeNode {
                        score: prob.ln() + relevance.ln(),
                        depth: 0,
                        token_idx: i,
                        parent_path: i as u128,
                    });
                }
            }
        }

        // ── Phase C: Best-first expansion with screening ─────────
        let mut best_score: Option<f32> = None;
        let mut second_best_score: Option<f32> = None;
        let mut consecutive_dominant: usize = 0;
        while self.tree.len() < config.tree_budget {
            let Some(best) = self.heap.pop() else {
                break;
            };
            self.tree.push(best);

            // Confidence-gap early exit (Plan 026: AutoTTS)
            let score = best.score;
            match best_score {
                None => {
                    best_score = Some(score);
                }
                Some(bs) if score > bs => {
                    second_best_score = Some(bs);
                    best_score = Some(score);
                    consecutive_dominant = 1;
                }
                Some(bs) => {
                    // Not a new best — track running second best (degrades with heap)
                    second_best_score = Some(score);
                    if bs - score > config.early_exit_gap {
                        consecutive_dominant += 1;
                    } else {
                        consecutive_dominant = 0;
                    }
                }
            }
            if config.early_exit_patience > 0
                && config.early_exit_gap > 0.0
                && consecutive_dominant >= config.early_exit_patience
                && best_score.unwrap_or(0.0) - second_best_score.unwrap_or(0.0)
                    > config.early_exit_gap
            {
                break;
            }

            if best.depth + 1 < marginals.len() {
                let next_depth = best.depth + 1;
                let parent_tokens = extract_parent_tokens_into(
                    best.parent_path,
                    best.depth + 1,
                    &mut self.parent_tokens_buf,
                );
                for (i, &prob) in marginals[next_depth].iter().enumerate() {
                    if prob <= 0.0 {
                        continue;
                    }
                    let relevance = screener.relevance(next_depth, i, parent_tokens);
                    if relevance <= threshold {
                        continue;
                    }
                    // SCREENING: ln(P_llm) + ln(R) blended score
                    self.heap.push(TreeNode {
                        score: best.score + prob.ln() + relevance.ln(),
                        depth: next_depth,
                        token_idx: i,
                        parent_path: (best.parent_path << 16) | (i as u128),
                    });
                }
            }
        }

        &self.tree
    }

    /// Build tree with screening and merge retrieved branches in one call.
    #[allow(clippy::too_many_arguments)]
    pub fn build_and_merge_screened(
        &mut self,
        marginals: &[&[f32]],
        config: &crate::types::Config,
        screener: &dyn ScreeningPruner,
        chain_seed: bool,
        token_sequences: &[Vec<usize>],
        scores: &[f32],
        rest_weight: f32,
    ) -> &[TreeNode] {
        self.build_screened(marginals, config, screener, chain_seed);
        merge_retrieved_branches(
            &mut self.tree,
            marginals,
            config,
            token_sequences,
            scores,
            rest_weight,
        );
        &self.tree
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::speculative::dflash::dflash_predict;
    use crate::transformer::TransformerWeights;
    use crate::types::{Config, Rng};

    fn make_draft() -> (TransformerWeights, Config) {
        let config = Config::draft();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        (weights, config)
    }

    // ── Original DDTree Tests ─────────────────────────────────

    #[test]
    fn test_extract_parent_tokens_roundtrip() {
        let path_d0 = 3u128;
        let path_d1 = (path_d0 << 16) | 7;
        let path_d2 = (path_d1 << 16) | 1;

        assert_eq!(extract_parent_tokens(path_d0, 1), vec![3]);
        assert_eq!(extract_parent_tokens(path_d1, 2), vec![3, 7]);
        assert_eq!(extract_parent_tokens(path_d2, 3), vec![3, 7, 1]);
        let empty: Vec<usize> = extract_parent_tokens(0, 0);
        assert!(empty.is_empty());
    }

    #[test]
    fn test_ddtree_respects_budget() {
        let (weights, config) = make_draft();
        let marginals = dflash_predict(&weights, &config, 0, 0);
        let mv: Vec<&[f32]> = marginals.iter().map(|s| s.as_slice()).collect();
        let tree = build_dd_tree(&mv, &config);
        assert!(
            tree.len() <= config.tree_budget,
            "tree size {} exceeds budget {}",
            tree.len(),
            config.tree_budget
        );
        assert!(!tree.is_empty(), "tree should have at least one node");
    }

    #[test]
    fn test_ddtree_scores_descending() {
        let (weights, config) = make_draft();
        let marginals = dflash_predict(&weights, &config, 0, 0);
        let mv: Vec<&[f32]> = marginals.iter().map(|s| s.as_slice()).collect();
        let tree = build_dd_tree(&mv, &config);
        for window in tree.windows(2) {
            assert!(
                window[0].score >= window[1].score,
                "scores not descending: {} >= {}",
                window[0].score,
                window[1].score
            );
        }
    }

    #[test]
    fn test_ddtree_depth_within_lookahead() {
        let (weights, config) = make_draft();
        let marginals = dflash_predict(&weights, &config, 0, 0);
        let mv: Vec<&[f32]> = marginals.iter().map(|s| s.as_slice()).collect();
        let tree = build_dd_tree(&mv, &config);
        for node in &tree {
            assert!(
                node.depth < config.draft_lookahead,
                "depth {} should be < lookahead {}",
                node.depth,
                config.draft_lookahead
            );
        }
    }

    #[test]
    fn test_ddtree_valid_token_indices() {
        let (weights, config) = make_draft();
        let marginals = dflash_predict(&weights, &config, 0, 0);
        let mv: Vec<&[f32]> = marginals.iter().map(|s| s.as_slice()).collect();
        let tree = build_dd_tree(&mv, &config);
        for node in &tree {
            assert!(
                node.token_idx < config.vocab_size,
                "token_idx {} out of range",
                node.token_idx
            );
        }
    }

    #[test]
    fn test_ddtree_empty_marginals() {
        let config = Config::draft();
        let tree = build_dd_tree(&[], &config);
        assert!(tree.is_empty(), "empty marginals should produce empty tree");
    }

    #[test]
    fn test_ddtree_pruned_same_as_unpruned_with_no_pruner() {
        let (weights, config) = make_draft();
        let marginals = dflash_predict(&weights, &config, 0, 0);
        let mv: Vec<&[f32]> = marginals.iter().map(|s| s.as_slice()).collect();

        let tree_unpruned = build_dd_tree(&mv, &config);
        let tree_pruned = build_dd_tree_pruned(&mv, &config, &NoPruner, false);

        assert_eq!(
            tree_unpruned.len(),
            tree_pruned.len(),
            "NoPruner should produce identical tree"
        );
        for (a, b) in tree_unpruned.iter().zip(tree_pruned.iter()) {
            assert_eq!(a.score, b.score, "scores should match");
            assert_eq!(a.token_idx, b.token_idx, "tokens should match");
        }
    }

    #[test]
    fn test_ddtree_pruned_empty_marginals() {
        let config = Config::draft();
        let pruner = NoPruner;
        let tree = build_dd_tree_pruned(&[], &config, &pruner, false);
        assert!(tree.is_empty(), "empty marginals should produce empty tree");
    }

    // ── merge_retrieved_branches Tests ─────────────────────────

    #[test]
    fn test_merge_empty_retrieval_noop() {
        let config = Config::draft();
        let marginals = [vec![0.5; config.vocab_size]];
        let mv: Vec<&[f32]> = marginals.iter().map(|s| s.as_slice()).collect();
        let mut tree = vec![TreeNode {
            score: 1.0,
            depth: 0,
            token_idx: 0,
            parent_path: 0,
        }];
        let original_len = tree.len();

        merge_retrieved_branches(&mut tree, &mv, &config, &[], &[], 0.5);

        assert_eq!(
            tree.len(),
            original_len,
            "empty retrieval should not change tree"
        );
    }

    #[test]
    fn test_merge_preserves_budget() {
        let config = Config::draft();
        let marginals = vec![vec![0.1; config.vocab_size]; 4];
        let mv: Vec<&[f32]> = marginals.iter().map(|s| s.as_slice()).collect();
        let mut tree = build_dd_tree(&mv, &config);

        // Create many sequences that would exceed budget
        let sequences: Vec<Vec<usize>> = (0..100)
            .map(|i| vec![i % config.vocab_size, (i + 1) % config.vocab_size])
            .collect();
        let scores: Vec<f32> = (0..100).map(|_| 0.9).collect();

        merge_retrieved_branches(&mut tree, &mv, &config, &sequences, &scores, 0.3);

        assert!(
            tree.len() <= config.tree_budget,
            "tree should not exceed budget, got {}",
            tree.len()
        );
    }

    #[test]
    fn test_merge_sorts_by_score() {
        let config = Config::draft();
        let marginals = vec![vec![0.1; config.vocab_size]; 2];
        let mv: Vec<&[f32]> = marginals.iter().map(|s| s.as_slice()).collect();
        let mut tree = Vec::new();

        let sequences = vec![vec![0, 1], vec![2, 3]];
        let scores = vec![0.5, 0.9];

        merge_retrieved_branches(&mut tree, &mv, &config, &sequences, &scores, 0.5);

        for window in tree.windows(2) {
            assert!(
                window[0].score >= window[1].score,
                "tree should be sorted by score descending"
            );
        }
    }

    #[test]
    fn test_merge_with_empty_tree_adds_nodes() {
        let config = Config::draft();
        // Marginals with non-zero prob at specific tokens
        let mut m0 = vec![0.0; config.vocab_size];
        m0[5] = 0.8;
        let mut m1 = vec![0.0; config.vocab_size];
        m1[10] = 0.6;
        let marginals = [m0, m1];
        let mv: Vec<&[f32]> = marginals.iter().map(|s| s.as_slice()).collect();
        let mut tree = Vec::new();

        let sequences = vec![vec![5, 10]];
        let scores = vec![0.7];

        merge_retrieved_branches(&mut tree, &mv, &config, &sequences, &scores, 0.3);

        assert_eq!(tree.len(), 2, "should add 2 nodes for 2-depth sequence");
        assert_eq!(tree[0].token_idx, 5, "first node should be token 5");
    }

    #[test]
    fn test_merge_zero_weight_is_noop() {
        let config = Config::draft();
        let marginals = [vec![0.5; config.vocab_size]];
        let mv: Vec<&[f32]> = marginals.iter().map(|s| s.as_slice()).collect();
        let mut tree = Vec::new();

        let sequences = vec![vec![0]];
        let scores = vec![0.9];

        merge_retrieved_branches(&mut tree, &mv, &config, &sequences, &scores, 0.0);

        assert!(tree.is_empty(), "zero rest_weight should be no-op");
    }

    // ── Chain-Seed DDTree Tests ───────────────────────────────

    /// Create marginals with known argmax at each depth for deterministic testing.
    fn make_chain_marginals(config: &Config) -> Vec<Vec<f32>> {
        let mut m0 = vec![0.01; config.vocab_size];
        m0[5] = 0.9;
        let mut m1 = vec![0.01; config.vocab_size];
        m1[10] = 0.85;
        let mut m2 = vec![0.01; config.vocab_size];
        m2[3] = 0.8;
        vec![m0, m1, m2]
    }

    #[test]
    fn test_chain_seed_produces_chain_path() {
        let config = Config::draft();
        let marginals = make_chain_marginals(&config);
        let mv: Vec<&[f32]> = marginals.iter().map(|s| s.as_slice()).collect();

        let tree = build_dd_tree_pruned(&mv, &config, &NoPruner, true);

        // Chain nodes are the first 3 entries (depths 0, 1, 2)
        assert!(
            tree.len() >= 3,
            "tree should have at least 3 chain nodes, got {}",
            tree.len()
        );

        // Verify chain nodes form contiguous path with argmax tokens
        assert_eq!(tree[0].depth, 0, "first chain node at depth 0");
        assert_eq!(tree[0].token_idx, 5, "chain node depth 0 = argmax token 5");

        assert_eq!(tree[1].depth, 1, "second chain node at depth 1");
        assert_eq!(
            tree[1].token_idx, 10,
            "chain node depth 1 = argmax token 10"
        );

        assert_eq!(tree[2].depth, 2, "third chain node at depth 2");
        assert_eq!(tree[2].token_idx, 3, "chain node depth 2 = argmax token 3");

        // Verify chain node parent_paths form contiguous path
        assert_eq!(tree[0].parent_path, 5, "depth 0 parent_path = token 5");
        assert_eq!(
            tree[1].parent_path,
            (5u128 << 16) | 10,
            "depth 1 parent_path = [5, 10]"
        );
        assert_eq!(
            tree[2].parent_path,
            ((5u128 << 16) | 10) << 16 | 3,
            "depth 2 parent_path = [5, 10, 3]"
        );

        // Verify cumulative scores
        let expected_d0 = marginals[0][5].ln();
        let expected_d1 = expected_d0 + marginals[1][10].ln();
        let expected_d2 = expected_d1 + marginals[2][3].ln();

        assert!(
            (tree[0].score - expected_d0).abs() < 1e-5,
            "depth 0 score: expected {expected_d0}, got {}",
            tree[0].score
        );
        assert!(
            (tree[1].score - expected_d1).abs() < 1e-5,
            "depth 1 score: expected {expected_d1}, got {}",
            tree[1].score
        );
        assert!(
            (tree[2].score - expected_d2).abs() < 1e-5,
            "depth 2 score: expected {expected_d2}, got {}",
            tree[2].score
        );
    }

    #[test]
    fn test_chain_seed_false_matches_original() {
        let (weights, config) = make_draft();
        let marginals = dflash_predict(&weights, &config, 0, 0);
        let mv: Vec<&[f32]> = marginals.iter().map(|s| s.as_slice()).collect();

        // build_dd_tree calls build_dd_tree_pruned with chain_seed=false
        let tree_via_wrapper = build_dd_tree(&mv, &config);
        let tree_via_pruned = build_dd_tree_pruned(&mv, &config, &NoPruner, false);

        assert_eq!(
            tree_via_wrapper.len(),
            tree_via_pruned.len(),
            "both should produce same number of nodes"
        );
        for (a, b) in tree_via_wrapper.iter().zip(tree_via_pruned.iter()) {
            assert_eq!(a.score, b.score, "scores should match");
            assert_eq!(a.token_idx, b.token_idx, "tokens should match");
            assert_eq!(a.depth, b.depth, "depths should match");
            assert_eq!(a.parent_path, b.parent_path, "parent_paths should match");
        }
    }

    #[test]
    fn test_chain_seed_respects_budget() {
        let (weights, config) = make_draft();
        let marginals = dflash_predict(&weights, &config, 0, 0);
        let mv: Vec<&[f32]> = marginals.iter().map(|s| s.as_slice()).collect();

        let tree = build_dd_tree_pruned(&mv, &config, &NoPruner, true);

        assert!(
            tree.len() <= config.tree_budget,
            "chain-seed tree size {} exceeds budget {}",
            tree.len(),
            config.tree_budget
        );
        assert!(!tree.is_empty(), "tree should have at least one node");
    }

    /// Pruner that blocks a specific token at a specific depth.
    struct BlockTokenPruner {
        depth: usize,
        token: usize,
    }

    impl ConstraintPruner for BlockTokenPruner {
        fn is_valid(&self, depth: usize, token_idx: usize, _parent_tokens: &[usize]) -> bool {
            !(depth == self.depth && token_idx == self.token)
        }
    }

    #[test]
    fn test_chain_seed_with_pruner() {
        let config = Config::draft();
        let marginals = make_chain_marginals(&config);

        // Block token 10 at depth 1 (the argmax) — chain should break there
        let pruner = BlockTokenPruner {
            depth: 1,
            token: 10,
        };
        let mv: Vec<&[f32]> = marginals.iter().map(|s| s.as_slice()).collect();
        let tree = build_dd_tree_pruned(&mv, &config, &pruner, true);

        // Chain should have only depth 0 (broke at depth 1)
        assert!(
            !tree.is_empty(),
            "tree should have at least the depth 0 chain node"
        );
        assert_eq!(
            tree[0].token_idx, 5,
            "depth 0 chain node should be argmax token 5"
        );
        assert_eq!(tree[0].depth, 0);

        // No node at depth 1 should have token 10 (blocked)
        for node in &tree {
            if node.depth == 1 {
                assert_ne!(
                    node.token_idx, 10,
                    "blocked token 10 should not appear at depth 1"
                );
            }
        }

        // Verify tree still contains valid nodes (siblings and best-first)
        assert!(
            tree.len() > 1,
            "tree should have more than just the chain node (siblings/best-first)"
        );
    }

    #[test]
    fn test_chain_seed_empty_marginals() {
        let config = Config::draft();
        let tree = build_dd_tree_pruned(&[], &config, &NoPruner, true);
        assert!(
            tree.is_empty(),
            "empty marginals should produce empty tree with chain_seed=true"
        );
    }

    // ── ScreeningPruner Tests (Plan 021) ──────────────────────

    /// Screener that returns fixed relevance per token index.
    struct FixedRelevanceScreener {
        relevances: Vec<f32>,
    }

    impl ScreeningPruner for FixedRelevanceScreener {
        fn relevance(&self, _depth: usize, token_idx: usize, _parent_tokens: &[usize]) -> f32 {
            self.relevances.get(token_idx).copied().unwrap_or(0.1)
        }
    }

    #[test]
    fn test_screened_no_screener_matches_unpruned() {
        // NoScreeningPruner returns 1.0 everywhere → ln(1.0)=0.0 → same as unpruned
        let (weights, config) = make_draft();
        let marginals = dflash_predict(&weights, &config, 0, 0);
        let mv: Vec<&[f32]> = marginals.iter().map(|s| s.as_slice()).collect();

        let tree_unpruned = build_dd_tree(&mv, &config);
        let tree_screened = build_dd_tree_screened(&mv, &config, &NoScreeningPruner, false);

        assert_eq!(
            tree_unpruned.len(),
            tree_screened.len(),
            "NoScreeningPruner should produce identical tree size"
        );
        for (a, b) in tree_unpruned.iter().zip(tree_screened.iter()) {
            assert!(
                (a.score - b.score).abs() < 1e-5,
                "scores should match: {} vs {}",
                a.score,
                b.score
            );
            assert_eq!(a.token_idx, b.token_idx, "tokens should match");
        }
    }

    #[test]
    fn test_screened_binary_compat_via_adapter() {
        // BinaryScreeningPruner adapter: ConstraintPruner → ScreeningPruner with R∈{0.0,1.0}
        let (weights, config) = make_draft();
        let marginals = dflash_predict(&weights, &config, 0, 0);
        let mv: Vec<&[f32]> = marginals.iter().map(|s| s.as_slice()).collect();

        let tree_pruned = build_dd_tree_pruned(&mv, &config, &NoPruner, false);
        // NoPruner wrapped in adapter: is_valid=true → relevance=1.0 → ln(1.0)=0.0
        let tree_screened =
            build_dd_tree_screened(&mv, &config, &BinaryScreeningPruner(NoPruner), false);

        assert_eq!(
            tree_pruned.len(),
            tree_screened.len(),
            "binary compat: same tree size via adapter"
        );
        for (a, b) in tree_pruned.iter().zip(tree_screened.iter()) {
            assert!(
                (a.score - b.score).abs() < 1e-5,
                "binary compat: scores should match"
            );
        }
    }

    #[test]
    fn test_screened_relevance_zero_hard_trims() {
        let mut config = Config::draft();
        config.tree_budget = 64;

        // 3 tokens: index 0 has high prob but R=0.0, index 1 has lower prob but R=1.0
        let mut m0 = vec![0.01; config.vocab_size];
        m0[0] = 0.9; // high LLM prob
        m0[1] = 0.05; // lower LLM prob
        let marginals = [m0];
        let mv: Vec<&[f32]> = marginals.iter().map(|s| s.as_slice()).collect();

        let screener = FixedRelevanceScreener {
            relevances: vec![0.0, 1.0], // token 0 trimmed, token 1 passes
        };

        let tree = build_dd_tree_screened(&mv, &config, &screener, false);

        // Token 0 should be completely absent (hard trim)
        for node in &tree {
            assert_ne!(
                node.token_idx, 0,
                "token 0 with relevance 0.0 should be hard-trimmed"
            );
        }
    }

    #[test]
    fn test_screened_relevance_half_applies_penalty() {
        let mut config = Config::draft();
        config.tree_budget = 64;

        // Two tokens with same LLM prob but different relevance
        let mut m0 = vec![0.01; config.vocab_size];
        m0[0] = 0.5;
        m0[1] = 0.5;
        let marginals = [m0];
        let mv: Vec<&[f32]> = marginals.iter().map(|s| s.as_slice()).collect();

        let screener = FixedRelevanceScreener {
            relevances: vec![1.0, 0.5], // token 1 gets -0.69 penalty
        };

        let tree = build_dd_tree_screened(&mv, &config, &screener, false);

        let node_0 = tree.iter().find(|n| n.token_idx == 0);
        let node_1 = tree.iter().find(|n| n.token_idx == 1);

        assert!(node_0.is_some(), "token 0 should be in tree");
        assert!(node_1.is_some(), "token 1 should be in tree");

        let score_0 = node_0.unwrap().score;
        let score_1 = node_1.unwrap().score;

        // Token 0: ln(0.5) + ln(1.0) = ln(0.5) + 0
        // Token 1: ln(0.5) + ln(0.5) = ln(0.5) - 0.693...
        let expected_diff = 0.5f32.ln().abs(); // ≈ 0.693
        let actual_diff = score_0 - score_1;

        assert!(
            (actual_diff - expected_diff).abs() < 1e-4,
            "penalty should be ln(0.5) ≈ -0.693, got diff={actual_diff:.4}, expected={expected_diff:.4}"
        );
    }

    #[test]
    fn test_screened_relevance_one_no_penalty() {
        let mut config = Config::draft();
        config.tree_budget = 64;

        let mut m0 = vec![0.01; config.vocab_size];
        m0[0] = 0.8;
        let marginals = [m0];
        let mv: Vec<&[f32]> = marginals.iter().map(|s| s.as_slice()).collect();

        let screener = FixedRelevanceScreener {
            relevances: vec![1.0],
        };

        let tree = build_dd_tree_screened(&mv, &config, &screener, false);

        let node = tree.iter().find(|n| n.token_idx == 0);
        assert!(node.is_some(), "token 0 should be in tree");

        let expected_score = 0.8f32.ln(); // ln(P) + ln(1.0) = ln(P) + 0
        assert!(
            (node.unwrap().score - expected_score).abs() < 1e-5,
            "relevance 1.0 should not modify score"
        );
    }

    #[test]
    fn test_screened_threshold_trims_mediocre() {
        let mut config = Config::draft();
        config.tree_budget = 64;
        config.screening_threshold = 0.4; // trim anything ≤ 0.4

        let mut m0 = vec![0.01; config.vocab_size];
        m0[0] = 0.5;
        m0[1] = 0.5;
        m0[2] = 0.5;
        let marginals = [m0];
        let mv: Vec<&[f32]> = marginals.iter().map(|s| s.as_slice()).collect();

        let screener = FixedRelevanceScreener {
            relevances: vec![0.3, 0.5, 0.8], // token 0 trimmed (≤0.4), 1 and 2 pass
        };

        let tree = build_dd_tree_screened(&mv, &config, &screener, false);

        // Token 0 (R=0.3 ≤ threshold 0.4) should be absent
        for node in &tree {
            assert_ne!(
                node.token_idx, 0,
                "token 0 with R=0.3 should be trimmed by threshold 0.4"
            );
        }
        // Token 1 (R=0.5 > threshold) and token 2 (R=0.8 > threshold) should be present
        assert!(
            tree.iter().any(|n| n.token_idx == 1),
            "token 1 with R=0.5 should survive threshold 0.4"
        );
        assert!(
            tree.iter().any(|n| n.token_idx == 2),
            "token 2 with R=0.8 should survive threshold 0.4"
        );
    }

    #[test]
    fn test_screened_empty_marginals() {
        let config = Config::draft();
        let tree = build_dd_tree_screened(&[], &config, &NoScreeningPruner, false);
        assert!(tree.is_empty(), "empty marginals should produce empty tree");
    }

    #[test]
    fn test_screened_chain_seed_with_relevance() {
        let mut config = Config::draft();
        config.tree_budget = 64;

        let mut m0 = vec![0.01; config.vocab_size];
        m0[5] = 0.9;
        let mut m1 = vec![0.01; config.vocab_size];
        m1[10] = 0.85;
        let marginals = [m0, m1];
        let mv: Vec<&[f32]> = marginals.iter().map(|s| s.as_slice()).collect();

        // Give token 5 at depth 0 a relevance of 0.6
        let mut relevances = vec![0.1; config.vocab_size];
        relevances[5] = 0.6;
        relevances[10] = 1.0;
        let screener = FixedRelevanceScreener { relevances };

        let tree = build_dd_tree_screened(&mv, &config, &screener, true);

        // Chain should build: token 5 (R=0.6), token 10 (R=1.0)
        assert!(
            tree.len() >= 2,
            "chain should have at least 2 nodes, got {}",
            tree.len()
        );

        // Score for token 5 should include ln(0.6) penalty
        let chain_d0 = tree.iter().find(|n| n.depth == 0 && n.token_idx == 5);
        assert!(chain_d0.is_some(), "chain node at depth 0 should exist");
        let expected_d0 = 0.9f32.ln() + 0.6f32.ln();
        assert!(
            (chain_d0.unwrap().score - expected_d0).abs() < 1e-4,
            "chain d0 score should include ln(0.6) penalty"
        );
    }

    // ── Early Exit Tests (Plan 026: AutoTTS) ──────────────────

    #[test]
    fn test_ddtree_early_exit_triggers_on_clear_winner() {
        // Create marginals where one path dominates massively
        let config = Config {
            tree_budget: 1000,
            early_exit_patience: 3,
            early_exit_gap: 1.0,
            ..Config::draft()
        };
        // One dominant token per depth
        let mut marginals = Vec::new();
        for _ in 0..config.draft_lookahead {
            let mut probs = vec![0.001_f32; config.vocab_size];
            probs[0] = 0.99; // token 0 dominates
            marginals.push(probs);
        }
        let mv: Vec<&[f32]> = marginals.iter().map(|s| s.as_slice()).collect();
        let tree = build_dd_tree(&mv, &config);
        // Should exit well before budget of 1000
        assert!(
            tree.len() < 1000,
            "early exit should trigger, got {} nodes vs budget 1000",
            tree.len()
        );
    }

    #[test]
    fn test_ddtree_early_exit_disabled_when_patience_zero() {
        let config = Config {
            tree_budget: 100,
            early_exit_patience: 0,
            early_exit_gap: 100.0,
            ..Config::draft()
        };
        let (weights, _) = make_draft();
        let marginals = dflash_predict(&weights, &Config::draft(), 0, 0);
        let mv: Vec<&[f32]> = marginals.iter().map(|s| s.as_slice()).collect();
        let tree = build_dd_tree(&mv, &config);
        // patience=0 disables early exit — tree should reach natural termination
        assert!(
            tree.len() <= config.tree_budget,
            "tree size {} exceeds budget {}",
            tree.len(),
            config.tree_budget
        );
    }

    #[test]
    fn test_ddtree_early_exit_no_false_exit_on_tight_gap() {
        // Uniform marginals — no clear winner, gap stays small
        let config = Config {
            tree_budget: 50,
            early_exit_patience: 5,
            early_exit_gap: 50.0, // very high gap requirement
            ..Config::draft()
        };
        let (weights, _) = make_draft();
        let marginals = dflash_predict(&weights, &Config::draft(), 0, 0);
        let mv: Vec<&[f32]> = marginals.iter().map(|s| s.as_slice()).collect();
        let tree = build_dd_tree(&mv, &config);
        // Gap too high to ever trigger — tree should fill normally
        assert!(!tree.is_empty());
    }
}
