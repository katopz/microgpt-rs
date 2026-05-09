//! PPoT rank: Self-consistency ranking and best variant selection.
//!
//! Distilled from "Test-time Recursive Thinking" (arXiv:2602.03094) and
//! "Self-Consistency" (Wang et al. 2022, arXiv:2203.11171).
//!
//! When PPoT produces multiple valid variants (all pass the screening pruner),
//! self-consistency ranking selects the one with highest agreement with other
//! valid variants. The intuition: correct solutions tend to converge on the
//! same answer, while incorrect ones disperse.
//!
//! Complexity: O(m² × L) for m variants of length L — negligible for m=10,
//! L=5–8 typical speculative decoding parameters.
//!
//! ## Performance (Plan 027 optimized)
//!
//! - `count_agreements`: chunked 4-at-a-time comparison for auto-vectorization
//! - `rank_by_consistency`: serial O(m²×L) — rayon overhead dominates for m≤16
//! - `select_best_variant`: clones only the winning variant once

use crate::speculative::types::ScreeningPruner;

// ── Agreement Counting ──────────────────────────────────────────

/// Count the number of token-level agreements between two variants.
///
/// Two tokens "agree" when they are identical at the same position.
/// Positions outside the resampled region should always agree (same base path).
///
/// Returns the raw count of matching positions.
///
/// ## Implementation
///
/// Uses chunked 4-at-a-time comparison via `u64` casts on 64-bit platforms,
/// enabling the compiler to auto-vectorize the equality checks.
#[inline]
fn count_agreements(a: &[usize], b: &[usize]) -> usize {
    let min_len = a.len().min(b.len());
    if min_len == 0 {
        return 0;
    }

    // Safety: usize and u64 have identical layout on 64-bit platforms.
    let a_u64 = unsafe { std::slice::from_raw_parts(a.as_ptr() as *const u64, min_len) };
    let b_u64 = unsafe { std::slice::from_raw_parts(b.as_ptr() as *const u64, min_len) };

    // Process 4 at a time — SIMD/auto-vectorization friendly
    let chunks = min_len / 4;
    let mut count = 0usize;

    for i in 0..chunks {
        let base = i * 4;
        count += (a_u64[base] == b_u64[base]) as usize;
        count += (a_u64[base + 1] == b_u64[base + 1]) as usize;
        count += (a_u64[base + 2] == b_u64[base + 2]) as usize;
        count += (a_u64[base + 3] == b_u64[base + 3]) as usize;
    }

    // Remainder (0–3 elements)
    for i in (chunks * 4)..min_len {
        count += (a_u64[i] == b_u64[i]) as usize;
    }

    count
}

/// Count agreements only at positions where the variants differ from the base path.
///
/// This focuses ranking on the *resampled* region. If both variants agree at
/// a resampled position, that's a stronger signal than agreeing at an unchanged
/// base path position.
#[inline]
fn count_resampled_agreements(a: &[usize], b: &[usize], base: &[usize]) -> usize {
    let min_len = a.len().min(b.len()).min(base.len());
    let mut agreements = 0;
    for i in 0..min_len {
        // Only count agreement at positions where at least one variant differs from base
        let a_changed = a.get(i).copied() != base.get(i).copied();
        let b_changed = b.get(i).copied() != base.get(i).copied();
        if (a_changed || b_changed) && a.get(i) == b.get(i) {
            agreements += 1;
        }
    }
    agreements
}

// ── Internal: compute agreement row for one variant ─────────────

/// Compute total agreement score for variant at index `i` against all others.
#[inline]
fn compute_agreement_score(variants: &[Vec<usize>], i: usize, m: usize) -> (usize, usize) {
    let mut agreements = variants[i].len(); // self-agreement: all positions match
    for j in 0..m {
        if i != j {
            agreements += count_agreements(&variants[i], &variants[j]);
        }
    }
    (i, agreements)
}

/// Compute weighted agreement score for variant at index `i` against all others.
#[inline]
fn compute_weighted_agreement_score(
    variants: &[Vec<usize>],
    base_path: &[usize],
    i: usize,
    m: usize,
) -> (usize, usize) {
    // Self: count resampled positions where variant differs from base
    let self_agreements = variants[i]
        .iter()
        .zip(base_path.iter())
        .filter(|(v, b)| v != b)
        .count()
        .max(1); // at least 1 for self
    let mut agreements = self_agreements;
    for j in 0..m {
        if i != j {
            agreements += count_resampled_agreements(&variants[i], &variants[j], base_path);
        }
    }
    (i, agreements)
}

/// Sort agreement counts: descending by count, ascending by index for ties.
#[inline]
fn sort_by_agreement(agreement_counts: &mut [(usize, usize)]) {
    agreement_counts.sort_by(|a, b| match b.1.cmp(&a.1) {
        std::cmp::Ordering::Equal => a.0.cmp(&b.0),
        other => other,
    });
}

// ── Consistency Ranking ─────────────────────────────────────────

/// Rank variants by self-consistency (pairwise agreement).
///
/// For each variant, counts how many other variants it agrees with at each
/// position. Variants with higher agreement are more "central" in the
/// solution space — they represent the consensus answer.
///
/// Returns `(variant_index, agreement_count)` sorted by agreement descending.
/// Ties are broken by variant index (lower index first, representing earlier
/// strategy in the cycle).
///
/// # Complexity
///
/// O(m² × L) where m = variants.len() and L = variant length.
/// For m=10, L=8: 10 × 10 × 8 = 800 comparisons — negligible.
///
/// # Parallelism
///
/// Uses rayon parallel iteration when m ≥ 8 to distribute the O(m²) pairwise
/// work across available cores.
///
/// # Example
///
/// ```ignore
/// let variants = vec![
///     vec![1, 2, 3], // variant 0
///     vec![1, 2, 4], // variant 1
///     vec![1, 2, 3], // variant 2 (same as 0)
/// ];
///
/// let ranked = rank_by_consistency(&variants);
/// // variant 0 agrees with 2 → 2 agreements (itself + variant 2)
/// // variant 1 agrees with none → 1 agreement (only itself)
/// // variant 2 agrees with 0 → 2 agreements (itself + variant 0)
/// assert_eq!(ranked[0].0, 0); // variant 0 is most consistent
/// ```
pub fn rank_by_consistency(variants: &[Vec<usize>]) -> Vec<(usize, usize)> {
    let m = variants.len();
    if m == 0 {
        return Vec::new();
    }

    // Serial O(m²×L) — rayon thread-pool overhead (~5μs) dominates for m≤16
    // where each row computation is ~0.1μs. Parallel wins only at m≥64.
    let mut agreement_counts: Vec<(usize, usize)> = (0..m)
        .map(|i| compute_agreement_score(variants, i, m))
        .collect();

    sort_by_agreement(&mut agreement_counts);
    agreement_counts
}

/// Rank variants by self-consistency, weighted toward resampled positions.
///
/// Like [`rank_by_consistency`] but only counts agreements at positions
/// where at least one variant differs from the base path. This focuses
/// the ranking on the quality of resampling choices rather than the
/// unchanged base path.
///
/// Use this when the base path is long and the resampled region is small
/// (typical PPoT scenario: resample 2–4 positions out of 5–8 total).
///
/// Serial evaluation — rayon overhead dominates for typical m≤16.
pub fn rank_by_consistency_weighted(
    variants: &[Vec<usize>],
    base_path: &[usize],
) -> Vec<(usize, usize)> {
    let m = variants.len();
    if m == 0 {
        return Vec::new();
    }

    let mut agreement_counts: Vec<(usize, usize)> = (0..m)
        .map(|i| compute_weighted_agreement_score(variants, base_path, i, m))
        .collect();

    sort_by_agreement(&mut agreement_counts);
    agreement_counts
}

// ── Best Variant Selection ──────────────────────────────────────

/// Select the best valid variant from a set of candidates.
///
/// Pipeline:
/// 1. Filter to variants that pass the screening pruner
/// 2. If single valid variant → return it
/// 3. If multiple valid → rank by self-consistency, return highest agreement
/// 4. If none valid → return `None`
///
/// This implements TRT's selection without ground truth: when multiple
/// candidates pass validation, the one most consistent with others is
/// preferred (mutual exclusivity principle from Wang et al. 2022).
///
/// Only clones the winning variant once at the end.
pub fn select_best_variant(
    variants: &[Vec<usize>],
    pruner: &dyn ScreeningPruner,
) -> Option<Vec<usize>> {
    if variants.is_empty() {
        return None;
    }

    // Validate each variant — collect indices of valid ones
    let valid_indices: Vec<usize> = variants
        .iter()
        .enumerate()
        .filter(|(_, variant)| is_path_valid(variant, pruner))
        .map(|(idx, _)| idx)
        .collect();

    match valid_indices.len() {
        0 => None,
        1 => Some(variants[valid_indices[0]].clone()),
        _ => {
            // Multiple valid: rank by consistency within the valid subset
            let valid_variants: Vec<&Vec<usize>> =
                valid_indices.iter().map(|&i| &variants[i]).collect();

            let ranked = rank_by_consistency_subset(&valid_variants);

            // Clone only the winner — map subset index back to original index
            ranked
                .into_iter()
                .next()
                .map(|(sub_idx, _)| variants[valid_indices[sub_idx]].clone())
        }
    }
}

/// Select the best valid variant with base-path-weighted consistency.
///
/// Like [`select_best_variant`] but uses [`rank_by_consistency_weighted`]
/// to focus ranking on resampled positions.
///
/// Only clones the winning variant once at the end.
pub fn select_best_variant_weighted(
    variants: &[Vec<usize>],
    base_path: &[usize],
    pruner: &dyn ScreeningPruner,
) -> Option<Vec<usize>> {
    if variants.is_empty() {
        return None;
    }

    let valid_indices: Vec<usize> = variants
        .iter()
        .enumerate()
        .filter(|(_, variant)| is_path_valid(variant, pruner))
        .map(|(idx, _)| idx)
        .collect();

    match valid_indices.len() {
        0 => None,
        1 => Some(variants[valid_indices[0]].clone()),
        _ => {
            let valid_variants: Vec<&Vec<usize>> =
                valid_indices.iter().map(|&i| &variants[i]).collect();

            let ranked = rank_by_consistency_weighted_subset(&valid_variants, base_path);

            // Clone only the winner — map subset index back to original index
            ranked
                .into_iter()
                .next()
                .map(|(sub_idx, _)| variants[valid_indices[sub_idx]].clone())
        }
    }
}

// ── Internal Helpers ────────────────────────────────────────────

/// Check if a token path passes the screening pruner.
/// Returns `true` if every token has positive relevance (no hard rejection).
#[inline]
fn is_path_valid(path: &[usize], pruner: &dyn ScreeningPruner) -> bool {
    for (depth, &token) in path.iter().enumerate() {
        let relevance = pruner.relevance(depth, token, &path[..depth]);
        if relevance <= 0.0 {
            return false;
        }
    }
    true
}

/// Rank a subset of variants by consistency (avoids re-indexing).
/// Used internally by `select_best_variant` on already-filtered valid variants.
fn rank_by_consistency_subset(variants: &[&Vec<usize>]) -> Vec<(usize, usize)> {
    let m = variants.len();
    if m == 0 {
        return Vec::new();
    }

    let mut agreement_counts: Vec<(usize, usize)> = (0..m)
        .map(|i| {
            let mut agreements = variants[i].len(); // self-agreement
            for j in 0..m {
                if i != j {
                    agreements += count_agreements(variants[i], variants[j]);
                }
            }
            (i, agreements)
        })
        .collect();

    sort_by_agreement(&mut agreement_counts);
    agreement_counts
}

/// Rank a subset of variants by weighted consistency.
/// Used internally by `select_best_variant_weighted` on already-filtered valid variants.
fn rank_by_consistency_weighted_subset(
    variants: &[&Vec<usize>],
    base_path: &[usize],
) -> Vec<(usize, usize)> {
    let m = variants.len();
    if m == 0 {
        return Vec::new();
    }

    let mut agreement_counts: Vec<(usize, usize)> = (0..m)
        .map(|i| {
            let self_agreements = variants[i]
                .iter()
                .zip(base_path.iter())
                .filter(|(v, b)| v != b)
                .count()
                .max(1);
            let mut agreements = self_agreements;
            for j in 0..m {
                if i != j {
                    agreements += count_resampled_agreements(variants[i], variants[j], base_path);
                }
            }
            (i, agreements)
        })
        .collect();

    sort_by_agreement(&mut agreement_counts);
    agreement_counts
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::speculative::types::NoScreeningPruner;

    // ── count_agreements tests ──

    #[test]
    fn test_count_agreements_identical() {
        let a = vec![1, 2, 3, 4];
        let b = vec![1, 2, 3, 4];
        assert_eq!(count_agreements(&a, &b), 4);
    }

    #[test]
    fn test_count_agreements_completely_different() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        assert_eq!(count_agreements(&a, &b), 0);
    }

    #[test]
    fn test_count_agreements_partial() {
        let a = vec![1, 2, 3, 4];
        let b = vec![1, 6, 3, 8];
        assert_eq!(count_agreements(&a, &b), 2); // positions 0 and 2
    }

    #[test]
    fn test_count_agreements_different_lengths() {
        let a = vec![1, 2, 3];
        let b = vec![1, 2];
        assert_eq!(count_agreements(&a, &b), 2); // only compare up to min length
    }

    #[test]
    fn test_count_agreements_empty() {
        let a: Vec<usize> = vec![];
        let b: Vec<usize> = vec![];
        assert_eq!(count_agreements(&a, &b), 0);
    }

    // ── count_resampled_agreements tests ──

    #[test]
    fn test_count_resampled_agreements_basic() {
        let base = vec![1, 2, 3, 4];
        let a = vec![1, 5, 3, 6]; // changed positions 1, 3
        let b = vec![1, 5, 3, 7]; // changed positions 1, 3

        // Both changed position 1 to same value (5) → 1 agreement at resampled pos
        // Both changed position 3 but to different values → no agreement
        assert_eq!(count_resampled_agreements(&a, &b, &base), 1);
    }

    #[test]
    fn test_count_resampled_agreements_no_resampled() {
        let base = vec![1, 2, 3];
        let a = vec![1, 2, 3]; // same as base
        let b = vec![1, 2, 3]; // same as base

        // No positions changed → no resampled agreements
        assert_eq!(count_resampled_agreements(&a, &b, &base), 0);
    }

    #[test]
    fn test_count_resampled_agreements_one_changed() {
        let base = vec![1, 2, 3];
        let a = vec![1, 5, 3]; // changed position 1
        let b = vec![1, 2, 3]; // same as base

        // Position 1: a changed, b didn't → no agreement
        assert_eq!(count_resampled_agreements(&a, &b, &base), 0);
    }

    // ── rank_by_consistency tests ──

    #[test]
    fn test_rank_by_consistency_single() {
        let variants = vec![vec![1, 2, 3]];
        let ranked = rank_by_consistency(&variants);
        assert_eq!(ranked, vec![(0, 3)]); // self-agreement = variant length (3)
    }

    #[test]
    fn test_rank_by_consistency_duplicate_variants() {
        let variants = vec![
            vec![1, 2, 3], // 0
            vec![1, 2, 3], // 1 (same as 0)
            vec![4, 5, 6], // 2 (completely different)
        ];

        let ranked = rank_by_consistency(&variants);

        // Variant 0: self(3) + agree with 1(3) + agree with 2(0) = 6
        // Variant 1: self(3) + agree with 0(3) + agree with 2(0) = 6
        // Variant 2: self(3) + agree with 0(0) + agree with 1(0) = 3
        assert_eq!(ranked.len(), 3);
        assert_eq!(ranked[0].0, 0); // tied at 6, lower index first
        assert_eq!(ranked[0].1, 6);
        assert_eq!(ranked[1].0, 1);
        assert_eq!(ranked[1].1, 6);
        assert_eq!(ranked[2].0, 2);
        assert_eq!(ranked[2].1, 3);
    }

    #[test]
    fn test_rank_by_consistency_partial_agreement() {
        let variants = vec![
            vec![1, 2, 3], // 0
            vec![1, 2, 4], // 1 (agrees on positions 0,1)
            vec![1, 5, 3], // 2 (agrees on positions 0,2)
        ];

        let ranked = rank_by_consistency(&variants);

        // Variant 0: self(3) + agree1(2) + agree2(2) = 7
        // Variant 1: self(3) + agree0(2) + agree2(1) = 6
        // Variant 2: self(3) + agree0(2) + agree1(1) = 6
        assert_eq!(ranked[0].0, 0);
        assert_eq!(ranked[0].1, 7);
        assert!(ranked[1].0 == 1 || ranked[1].0 == 2);
        assert_eq!(ranked[1].1, 6);
    }

    #[test]
    fn test_rank_by_consistency_empty() {
        let variants: Vec<Vec<usize>> = vec![];
        let ranked = rank_by_consistency(&variants);
        assert!(ranked.is_empty());
    }

    // ── select_best_variant tests ──

    /// Pruner that rejects token 0 at any position.
    struct RejectZeroPruner;

    impl ScreeningPruner for RejectZeroPruner {
        fn relevance(&self, _depth: usize, token_idx: usize, _parent_tokens: &[usize]) -> f32 {
            if token_idx == 0 { 0.0 } else { 1.0 }
        }
    }

    #[test]
    fn test_select_best_variant_single_valid() {
        let variants = vec![
            vec![0, 1, 2], // rejected (has token 0)
            vec![1, 2, 3], // valid
        ];

        let result = select_best_variant(&variants, &RejectZeroPruner);
        assert_eq!(result, Some(vec![1, 2, 3]));
    }

    #[test]
    fn test_select_best_variant_multiple_valid() {
        let variants = vec![
            vec![1, 2, 3], // valid
            vec![1, 2, 3], // valid (duplicate of 0)
            vec![4, 5, 6], // valid
        ];

        let result = select_best_variant(&variants, &NoScreeningPruner);
        assert!(result.is_some());
        // Should prefer the most consistent variant (0 or 1, tied)
        let path = result.unwrap();
        assert!(path == vec![1, 2, 3] || path == vec![1, 2, 3]);
    }

    #[test]
    fn test_select_best_variant_none_valid() {
        let variants = vec![
            vec![0, 1, 2], // rejected
            vec![3, 0, 4], // rejected
        ];

        let result = select_best_variant(&variants, &RejectZeroPruner);
        assert!(result.is_none());
    }

    #[test]
    fn test_select_best_variant_empty() {
        let variants: Vec<Vec<usize>> = vec![];
        let result = select_best_variant(&variants, &NoScreeningPruner);
        assert!(result.is_none());
    }

    #[test]
    fn test_select_best_variant_no_pruner() {
        let variants = vec![vec![1, 2, 3], vec![4, 5, 6]];

        let result = select_best_variant(&variants, &NoScreeningPruner);
        assert!(result.is_some());
    }

    // ── select_best_variant_weighted tests ──

    #[test]
    fn test_select_best_variant_weighted_basic() {
        let base = vec![1, 2, 3, 4];
        let variants = vec![
            vec![1, 5, 3, 6], // changed positions 1, 3
            vec![1, 5, 3, 6], // same as 0 (high agreement at resampled positions)
            vec![1, 7, 3, 8], // different resampled values
        ];

        let result = select_best_variant_weighted(&variants, &base, &NoScreeningPruner);
        assert!(result.is_some());
        let path = result.unwrap();
        // Should prefer variant 0 or 1 (same, highest agreement at resampled positions)
        assert!(path == vec![1, 5, 3, 6]);
    }

    // ── is_path_valid tests ──

    #[test]
    fn test_is_path_valid_no_pruner() {
        assert!(is_path_valid(&[0, 1, 2], &NoScreeningPruner));
        assert!(is_path_valid(&[], &NoScreeningPruner));
    }

    #[test]
    fn test_is_path_valid_reject_zero() {
        assert!(is_path_valid(&[1, 2, 3], &RejectZeroPruner));
        assert!(!is_path_valid(&[0, 1, 2], &RejectZeroPruner));
        assert!(!is_path_valid(&[1, 0, 2], &RejectZeroPruner));
    }

    // ── Consensus degree helper ──

    #[test]
    fn test_consensus_perfect() {
        // All 5 variants identical → perfect consensus
        let variants: Vec<Vec<usize>> = (0..5).map(|_| vec![1, 2, 3]).collect();
        let ranked = rank_by_consistency(&variants);

        // Each variant agrees with all 5 (including self) at all 3 positions
        // total agreements = 5 × 3 = 15
        assert_eq!(ranked[0].1, 15);
    }

    #[test]
    fn test_no_consensus() {
        // All variants completely different from each other
        let variants = vec![vec![1, 1, 1], vec![2, 2, 2], vec![3, 3, 3]];

        let ranked = rank_by_consistency(&variants);

        // Each only agrees with itself: 3 agreements
        assert_eq!(ranked[0].1, 3);
    }
}
