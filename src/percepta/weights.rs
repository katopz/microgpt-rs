// SPDX-License-Identifier: Apache-2.0
// Distilled from Percepta's `transformer-vm` (Apache-2.0 © Percepta).

//! Analytical weight construction: graph + schedule → transformer weight tensors.
//!
//! This module bridges the abstract computation graph ([`ProgramGraph`]) and its
//! schedule (from MILP) to produce concrete weight matrices for transformer
//! inference. Each expression in the graph is mapped to a dense weight vector
//! in slot space, and attention/FFN weights are constructed to implement the
//! computation graph's semantics.
//!
//! # Key concepts
//!
//! - **Slot space**: Each dimension is assigned a slot (position in the d_model vector)
//!   by the scheduler's interval coloring. [`expr_to_vector`] maps expressions to dense
//!   vectors by placing coefficients at their dimension's slot positions.
//! - **Attention heads**: Each LookUp operation becomes one or more attention heads.
//!   Query/key weights use parabolic encoding with HARD_K scaling for hardmax behavior.
//! - **FFN neurons**: Each ReGLU dimension becomes one FFN neuron with gate (b_expr)
//!   and value (a_expr) weights. ReGLU computes `relu(b) * a`.
//! - **Passthrough**: Persist expressions that reference non-produced dims need
//!   passthrough heads/neurons to copy values from source to destination slots.
//! - **Erase**: When a slot is reused (dying dim → born dim), the persist projection
//!   subtracts 1.0 from the diagonal to zero out the old value.

use std::collections::{HashMap, HashSet};

use crate::percepta::graph::types::{DimId, DimensionKind, Expression, LookupId, ProgramGraph};
use crate::percepta::scheduler::{Schedule, StdLayer};
use crate::percepta::types::TieBreak;

// ── Constants ──────────────────────────────────────────────────

/// Softmax temperature scaling to approximate hardmax (argmax) attention.
///
/// Must match `HARD_K` in [`crate::percepta::types::HARD_K`].
const HARD_K: f64 = 1e10;

/// Square root of 2, used for attention head scaling.
const SQRT_2: f64 = std::f64::consts::SQRT_2;

// ── Output Types ───────────────────────────────────────────────

/// Complete set of transformer weight matrices ready for inference.
///
/// All matrices use `Vec<Vec<f64>>` (row-major): `matrix[row][col]`.
#[derive(Clone, Debug)]
pub struct TransformerWeights {
    /// Token embedding matrix: `[vocab_size, d_model]`.
    ///
    /// Row `i` is the d_model embedding for input token `i`.
    pub embedding: Vec<Vec<f64>>,
    /// Unembedding matrix: `[vocab_size, d_model]`.
    ///
    /// Row `i` is the d_model scoring vector for output token `i`.
    pub unembedding: Vec<Vec<f64>>,
    /// Per-layer weights (attention + FFN).
    pub layers: Vec<LayerWeights>,
    /// Per-layer per-head tie-break flags.
    ///
    /// `tiebreak[layer][head]` is `true` for Latest tie-break, `false` for Average.
    pub head_tiebreak: Vec<Vec<bool>>,
    /// Per-layer erased slots at persist1 boundary (attention half-layer).
    pub attn_erase: Vec<Vec<usize>>,
    /// Per-layer erased slots at persist2 boundary (FFN half-layer).
    pub ffn_erase: Vec<Vec<usize>>,
    /// Model dimension (`d_model`).
    pub d_model: usize,
    /// Number of attention heads per layer (`d_model / 2`).
    pub n_heads: usize,
    /// FFN hidden dimension per layer.
    pub d_ffn: usize,
    /// Number of transformer layers.
    pub n_layers: usize,
    /// Vocabulary size (`max(input_tokens.len(), output_tokens.len())`).
    pub vocab_size: usize,
}

/// Per-layer weight matrices.
#[derive(Clone, Debug)]
pub struct LayerWeights {
    /// Attention weights for this layer.
    pub attention: AttentionWeights,
    /// FFN weights for this layer.
    pub ffn: FfnWeights,
}

/// Attention weight matrices: `[3 * d_model, d_model]` for in_proj,
/// `[d_model, d_model]` for out_proj.
///
/// Layout follows PyTorch's `nn.MultiheadAttention` convention:
/// - `in_proj[0..d_model]` = query weights (n_heads × 2 rows, each head has qx, qy)
/// - `in_proj[d_model..2*d_model]` = key weights
/// - `in_proj[2*d_model..3*d_model]` = value weights
/// - `out_proj[d_model, d_model]` = output projection
#[derive(Clone, Debug)]
pub struct AttentionWeights {
    /// Input projection: `[3 * d_model, d_model]`.
    ///
    /// Concatenation of [query; key; value] weight matrices.
    pub in_proj: Vec<Vec<f64>>,
    /// Output projection: `[d_model, d_model]`.
    pub out_proj: Vec<Vec<f64>>,
}

/// FFN weight matrices with ReGLU activation.
///
/// Layout: `ff_in[2 * d_ffn, d_model]`, `ff_out[d_model, d_ffn]`.
///
/// For each ReGLU neuron `j`:
/// - `ff_in[j]` = gate weights (from b_expr)
/// - `ff_in[d_ffn + j]` = value weights (from a_expr)
/// - Output: `relu(ff_in[j] · x) * (ff_in[d_ffn+j] · x)`
#[derive(Clone, Debug)]
pub struct FfnWeights {
    /// FFN input projection: `[2 * d_ffn, d_model]`.
    ///
    /// First `d_ffn` rows are gate weights, next `d_ffn` rows are value weights.
    pub ff_in: Vec<Vec<f64>>,
    /// FFN output projection: `[d_model, d_ffn]`.
    pub ff_out: Vec<Vec<f64>>,
}

/// Metadata for an attention head during weight construction.
#[derive(Clone, Debug)]
pub enum HeadInfo {
    /// A lookup head implementing hard attention retrieval.
    Lookup {
        lookup_id: LookupId,
        value_names: Vec<String>,
        tie_break: TieBreak,
    },
    /// A passthrough head copying slot values through attention.
    Passthrough {
        /// Source slots being passed through.
        src_slots: Vec<usize>,
    },
}

// ── Core: expr_to_vector ───────────────────────────────────────

/// Convert an [`Expression`] to a dense weight vector in slot space.
///
/// Creates a zero vector of length `width`, then for each `(dim, coeff)` term
/// in the expression, places `coeff` at position `slot_of[dim]` (if the dim
/// has an assigned slot).
///
/// Dims without slot assignments (internal dims produced and consumed within
/// a single phase) are silently skipped — their coefficients are dropped.
///
/// # Arguments
/// * `expr` — The expression to convert.
/// * `slot_of` — Maps DimId → slot index (from schedule).
/// * `width` — The model dimension (`d_model`).
///
/// # Returns
/// Dense vector of length `width`.
pub fn expr_to_vector(
    expr: &Expression,
    slot_of: &HashMap<DimId, usize>,
    width: usize,
) -> Vec<f64> {
    let mut w = vec![0.0; width];
    for (&dim, &coeff) in &expr.terms {
        if let Some(&slot) = slot_of.get(&dim)
            && slot < width
        {
            w[slot] += coeff;
        }
    }
    w
}

/// Copy `src` into `dst[row][..len]`, multiplying each element by `scale`.
///
/// Avoids clippy::manual_memcpy by using iterator-based copy.
#[inline]
fn copy_scaled(dst: &mut [f64], src: &[f64], scale: f64) {
    for (d, s) in dst.iter_mut().zip(src.iter()) {
        *d = s * scale;
    }
}

/// Copy `src` into `dst[row][..len]` element-wise (no scaling).
#[inline]
fn copy_row(dst: &mut [f64], src: &[f64]) {
    dst[..src.len()].copy_from_slice(&src[..src.len()]);
}

// ── Internal helpers ───────────────────────────────────────────

/// Compute the set of "internal" dimensions — dims produced and consumed
/// within a single phase, never appearing in any alive-after set, output
/// tokens, or input dims.
///
/// These dims don't get slot assignments and are excluded from weight
/// construction. In practice, checking `slot_of.contains_key(&dim)` is
/// sufficient, but this function provides the full set for diagnostics.
fn compute_internal_dims(pg: &ProgramGraph, schedule: &Schedule) -> HashSet<DimId> {
    let mut non_internal: HashSet<DimId> = HashSet::new();

    // Dims alive at any persist boundary
    for dims in schedule.alive_after.values() {
        non_internal.extend(dims.iter());
    }

    // Input dims
    for (&id, dim) in &pg.all_dims {
        if matches!(dim.kind, DimensionKind::Input) {
            non_internal.insert(id);
        }
    }

    // Dims referenced by output token expressions
    for expr in &pg.output_tokens {
        non_internal.extend(expr.terms.keys());
    }

    // Dims referenced by input token expressions
    for expr in &pg.input_tokens {
        non_internal.extend(expr.terms.keys());
    }

    // Dims in slot_of are non-internal by definition
    non_internal.extend(schedule.slot_of.keys());

    // Internal = everything else
    pg.all_dims
        .keys()
        .filter(|d| !non_internal.contains(d))
        .copied()
        .collect()
}

/// Compute erased slots at a persist boundary.
///
/// Erased slots are those where a dying dimension's slot is reused by a
/// newly born dimension. The persist projection subtracts 1.0 from the
/// diagonal for these slots, effectively zeroing out the old value.
fn compute_erased_slots(
    alive_before: &HashSet<DimId>,
    alive_after: &HashSet<DimId>,
    slot_of: &HashMap<DimId, usize>,
    protected: &HashSet<usize>,
) -> Vec<usize> {
    let dying: HashSet<DimId> = alive_before - alive_after;
    let born: HashSet<DimId> = alive_after - alive_before;

    let freed_slots: HashSet<usize> = dying
        .iter()
        .filter_map(|d| slot_of.get(d).copied())
        .filter(|s| !protected.contains(s))
        .collect();

    let mut erased: Vec<usize> = born
        .iter()
        .filter_map(|d| slot_of.get(d).copied())
        .filter(|s| freed_slots.contains(s))
        .collect();

    erased.sort();
    erased
}

/// Count per-layer attention heads and FFN neurons for resource sizing.
///
/// Tracks alive sets through the schedule to correctly account for erased slots
/// that need passthrough neurons. Returns `(per_layer_heads, per_layer_ffn)` vectors.
fn count_layer_resources(
    std_layers: &[StdLayer],
    pg: &ProgramGraph,
    schedule: &Schedule,
    internal_dims: &HashSet<DimId>,
    protected: &HashSet<usize>,
) -> (Vec<usize>, Vec<usize>) {
    let n_layers = std_layers.len();
    let mut per_layer_heads = Vec::with_capacity(n_layers);
    let mut per_layer_ffn = Vec::with_capacity(n_layers);

    // Track alive sets through the schedule
    let input_dims: HashSet<DimId> = pg
        .all_dims
        .iter()
        .filter(|(_, d)| matches!(d.kind, DimensionKind::Input))
        .map(|(&id, _)| id)
        .collect();
    let mut cur_alive: HashSet<DimId> = input_dims;

    for (layer_idx, layer) in std_layers.iter().enumerate() {
        // ── Attention half-layer ──
        let c1 = (4 * layer_idx + 1) as i32;
        let nxt1 = schedule
            .alive_after
            .get(&c1)
            .cloned()
            .unwrap_or_else(|| cur_alive.clone());
        let erased1 = compute_erased_slots(&cur_alive, &nxt1, &schedule.slot_of, protected);

        // Lookup heads
        let mut lu_heads = 0;
        let mut lookup_dims: HashSet<DimId> = HashSet::new();

        for &lu_id in &layer.attention {
            let Some(lu) = pg.all_lookups.get(&lu_id) else {
                continue;
            };
            lu_heads += lu.value_exprs.len().div_ceil(2);
            lookup_dims.extend(lu.dim_ids.iter().copied());
        }

        // Passthrough sources: erased slots + non-lookup dims from persist1
        let mut attn_pt_sources: HashSet<usize> = erased1.into_iter().collect();
        for &pd_id in &layer.persist1 {
            let Some(pd) = pg.all_dims.get(&pd_id) else {
                continue;
            };
            let Some(_dst_slot) = schedule.slot_of.get(&pd_id) else {
                continue;
            };
            let DimensionKind::Persist { expr } = &pd.kind else {
                continue;
            };
            for &d in expr.terms.keys() {
                if !lookup_dims.contains(&d)
                    && !internal_dims.contains(&d)
                    && let Some(&src_slot) = schedule.slot_of.get(&d)
                {
                    attn_pt_sources.insert(src_slot);
                }
            }
        }

        let total_heads = lu_heads + attn_pt_sources.len().div_ceil(2);
        per_layer_heads.push(total_heads);
        cur_alive = nxt1;

        // ── FFN half-layer ──
        let c3 = (4 * layer_idx + 3) as i32;
        let nxt3 = schedule
            .alive_after
            .get(&c3)
            .cloned()
            .unwrap_or_else(|| cur_alive.clone());
        let erased3 = compute_erased_slots(&cur_alive, &nxt3, &schedule.slot_of, protected);

        // Passthrough sources: erased slots + non-reglu dims from persist2
        let reglu_set: HashSet<DimId> = layer.ffn.iter().copied().collect();
        let mut ffn_pt_sources: HashSet<usize> = erased3.into_iter().collect();

        for &pd_id in &layer.persist2 {
            let Some(pd) = pg.all_dims.get(&pd_id) else {
                continue;
            };
            let DimensionKind::Persist { expr } = &pd.kind else {
                continue;
            };
            for &d in expr.terms.keys() {
                if !reglu_set.contains(&d)
                    && !internal_dims.contains(&d)
                    && let Some(&src_slot) = schedule.slot_of.get(&d)
                {
                    ffn_pt_sources.insert(src_slot);
                }
            }
        }

        let total_ffn = layer.ffn.len() + ffn_pt_sources.len();
        per_layer_ffn.push(total_ffn.max(1));
        cur_alive = nxt3;
    }

    (per_layer_heads, per_layer_ffn)
}

// ── Main: build_weights ────────────────────────────────────────

/// Build all transformer weight matrices from a computation graph and schedule.
///
/// This is the main entry point for weight construction. It produces a complete
/// set of weight matrices that implement the computation graph's semantics
/// when used with a standard transformer forward pass with ReGLU FFN and
/// 2D hard attention.
///
/// # Arguments
/// * `pg` — The computation graph with input/output token expressions.
/// * `schedule` — The MILP schedule with phase assignments and slot allocations.
///
/// # Returns
/// [`TransformerWeights`] with all weight matrices populated.
///
/// # Panics
/// Panics if the schedule is inconsistent with the graph (e.g., missing dims).
pub fn build_weights(pg: &ProgramGraph, schedule: &Schedule) -> TransformerWeights {
    let n_layers = schedule.num_layers;
    let slot_of = &schedule.slot_of;

    // Compute internal dims (no slot assignment)
    let internal_dims = compute_internal_dims(pg, schedule);

    // Protected slots (position, inv_log_pos, position_sq) — never erased
    let protected: HashSet<usize> = [
        slot_of.get(&pg.position).copied(),
        slot_of.get(&pg.inv_log_pos).copied(),
        slot_of.get(&pg.position_sq).copied(),
    ]
    .into_iter()
    .flatten()
    .collect();

    // ── Resource sizing ─────────────────────────────────────
    let (per_layer_heads, per_layer_ffn) = count_layer_resources(
        &schedule.std_layers,
        pg,
        schedule,
        &internal_dims,
        &protected,
    );

    let max_heads = per_layer_heads.iter().copied().max().unwrap_or(0);
    let max_ffn = per_layer_ffn.iter().copied().max().unwrap_or(1);

    // Model dimension: must accommodate both slots and heads
    let d_model = schedule.width.max(2 * max_heads);
    let d_model = d_model + d_model % 2; // round up to even
    let n_heads = d_model / 2;
    let d_ffn = max_ffn;

    // Vocabulary: max of input/output token counts
    let vocab_size = pg.input_tokens.len().max(pg.output_tokens.len()).max(1);

    // ── Build embedding ─────────────────────────────────────
    let mut embedding = vec![vec![0.0; d_model]; vocab_size];
    for (idx, expr) in pg.input_tokens.iter().enumerate() {
        embedding[idx] = expr_to_vector(expr, slot_of, d_model);
        // Zero out protected slots (position channels)
        if let Some(&s) = slot_of.get(&pg.position) {
            embedding[idx][s] = 0.0;
        }
        if let Some(&s) = slot_of.get(&pg.inv_log_pos) {
            embedding[idx][s] = 0.0;
        }
        if let Some(&s) = slot_of.get(&pg.position_sq) {
            embedding[idx][s] = 0.0;
        }
    }

    // ── Build unembedding ───────────────────────────────────
    let mut unembedding = vec![vec![0.0; d_model]; vocab_size];
    for (idx, expr) in pg.output_tokens.iter().enumerate() {
        unembedding[idx] = expr_to_vector(expr, slot_of, d_model);
    }

    // ── Build per-layer weights ─────────────────────────────
    let mut layers = Vec::with_capacity(n_layers);
    let mut head_tiebreak_all = Vec::with_capacity(n_layers);
    let mut attn_erase_all = Vec::with_capacity(n_layers);
    let mut ffn_erase_all = Vec::with_capacity(n_layers);

    // Track alive dims through the schedule
    let mut cur_alive: HashSet<DimId> = pg
        .all_dims
        .iter()
        .filter(|(_, d)| matches!(d.kind, DimensionKind::Input))
        .map(|(&id, _)| id)
        .collect();

    for layer_idx in 0..n_layers {
        let layer = &schedule.std_layers[layer_idx];

        // ── ATTENTION HALF-LAYER ──────────────────────────
        let c1 = (4 * layer_idx + 1) as i32;
        let nxt1 = schedule
            .alive_after
            .get(&c1)
            .cloned()
            .unwrap_or_else(|| cur_alive.clone());

        let erased1 = compute_erased_slots(&cur_alive, &nxt1, slot_of, &protected);

        let attn_weights = build_attention_layer_weights(
            layer,
            pg,
            slot_of,
            d_model,
            n_heads,
            &internal_dims,
            &erased1,
        );

        // Pad tiebreak to n_heads
        let mut tb = attn_weights.tiebreak;
        while tb.len() < n_heads {
            tb.push(false);
        }
        head_tiebreak_all.push(tb);
        attn_erase_all.push(erased1);

        cur_alive = nxt1;

        // ── FFN HALF-LAYER ────────────────────────────────
        let c3 = (4 * layer_idx + 3) as i32;
        let nxt3 = schedule
            .alive_after
            .get(&c3)
            .cloned()
            .unwrap_or_else(|| cur_alive.clone());

        let erased3 = compute_erased_slots(&cur_alive, &nxt3, slot_of, &protected);

        let ffn_weights =
            build_ffn_layer_weights(layer, pg, slot_of, d_model, d_ffn, &internal_dims, &erased3);

        ffn_erase_all.push(erased3);
        cur_alive = nxt3;

        layers.push(LayerWeights {
            attention: attn_weights.weights,
            ffn: ffn_weights,
        });
    }

    TransformerWeights {
        embedding,
        unembedding,
        layers,
        head_tiebreak: head_tiebreak_all,
        attn_erase: attn_erase_all,
        ffn_erase: ffn_erase_all,
        d_model,
        n_heads,
        d_ffn,
        n_layers,
        vocab_size,
    }
}

// ── Attention layer weight construction ────────────────────────

/// Result of building attention layer weights.
struct AttnBuildResult {
    weights: AttentionWeights,
    tiebreak: Vec<bool>,
}

/// Build attention weights for one layer.
///
/// Constructs the in_proj (Q, K, V) and out_proj weight matrices for the
/// attention half-layer. This includes:
/// 1. LookUp heads (hard attention retrieval with parabolic encoding)
/// 2. Passthrough heads (copy non-lookup dims through attention)
/// 3. Erase adjustments (subtract identity for reused slots)
#[allow(clippy::too_many_arguments)]
fn build_attention_layer_weights(
    layer: &StdLayer,
    pg: &ProgramGraph,
    slot_of: &HashMap<DimId, usize>,
    d_model: usize,
    n_heads: usize,
    internal_dims: &HashSet<DimId>,
    erased_slots: &[usize],
) -> AttnBuildResult {
    let mut in_proj = vec![vec![0.0; d_model]; 3 * d_model];
    let mut out_proj = vec![vec![0.0; d_model]; d_model];
    let mut tiebreak = Vec::new();
    let mut head_idx: usize = 0;

    // ── 1) LookUp heads ─────────────────────────────────────
    let mut lookup_dim_to_head: HashMap<DimId, (usize, usize)> = HashMap::new();

    for &lu_id in &layer.attention {
        let Some(lu) = pg.all_lookups.get(&lu_id) else {
            continue;
        };

        let nv = lu.value_exprs.len();
        let n_pairs = nv.div_ceil(2);

        for p in 0..n_pairs {
            let h = head_idx;
            head_idx += 1;

            // Tie-break flag
            let is_latest = matches!(lu.tie_break, TieBreak::Latest);
            tiebreak.push(is_latest);

            // Query weights (HARD_K scaled for hardmax)
            let scale = HARD_K * SQRT_2;
            let q0 = expr_to_vector(&lu.query_exprs_2d[0], slot_of, d_model);
            let q1 = expr_to_vector(&lu.query_exprs_2d[1], slot_of, d_model);
            copy_scaled(&mut in_proj[h * 2], &q0, scale);
            copy_scaled(&mut in_proj[h * 2 + 1], &q1, scale);

            // Key weights (no scaling — parabolic encoding is in the expressions)
            let k0 = expr_to_vector(&lu.key_exprs_2d[0], slot_of, d_model);
            let k1 = expr_to_vector(&lu.key_exprs_2d[1], slot_of, d_model);
            copy_row(&mut in_proj[d_model + h * 2], &k0);
            copy_row(&mut in_proj[d_model + h * 2 + 1], &k1);

            // Value weights
            let v0 = expr_to_vector(&lu.value_exprs[p * 2], slot_of, d_model);
            copy_row(&mut in_proj[2 * d_model + h * 2], &v0);
            if p * 2 + 1 < nv {
                let v1 = expr_to_vector(&lu.value_exprs[p * 2 + 1], slot_of, d_model);
                copy_row(&mut in_proj[2 * d_model + h * 2 + 1], &v1);
            }

            // Output projection: wire lookup dim → slot
            let d0 = lu.dim_ids[p * 2];
            if !internal_dims.contains(&d0)
                && let Some(&slot) = slot_of.get(&d0)
            {
                lookup_dim_to_head.insert(d0, (h, 0));
                out_proj[slot][h * 2] = 1.0;
            }
            if p * 2 + 1 < nv {
                let d1 = lu.dim_ids[p * 2 + 1];
                if !internal_dims.contains(&d1)
                    && let Some(&slot) = slot_of.get(&d1)
                {
                    lookup_dim_to_head.insert(d1, (h, 1));
                    out_proj[slot][h * 2 + 1] = 1.0;
                }
            }
        }
    }

    // ── 2) Passthrough contributions ────────────────────────
    // Build: src_slot → {dst_slot: coeff}
    let mut pt: HashMap<usize, HashMap<usize, f64>> = HashMap::new();

    for &pd_id in &layer.persist1 {
        let Some(pd) = pg.all_dims.get(&pd_id) else {
            continue;
        };
        let Some(&dst_slot) = slot_of.get(&pd_id) else {
            continue;
        };
        let DimensionKind::Persist { expr } = &pd.kind else {
            continue;
        };

        for (&d, &c) in &expr.terms {
            if let Some(&(h, comp)) = lookup_dim_to_head.get(&d) {
                // Wire through attention output projection
                out_proj[dst_slot][h * 2 + comp] += c;
            } else if let Some(&src_slot) = slot_of.get(&d)
                && !internal_dims.contains(&d)
            {
                pt.entry(src_slot)
                    .or_default()
                    .entry(dst_slot)
                    .and_modify(|v| *v += c)
                    .or_insert(c);
            }
        }
    }

    // Add erase contributions: subtract identity for erased slots
    for &s in erased_slots {
        pt.entry(s)
            .or_default()
            .entry(s)
            .and_modify(|v| *v -= 1.0)
            .or_insert(-1.0);
    }

    // ── 3) Pack passthroughs 2 per head ─────────────────────
    // Each passthrough head reads one source slot per component,
    // using position-based identity attention.
    let erase_q2d = [
        Expression::from_dim(pg.position),
        Expression::from_dim(pg.one),
    ];
    let erase_k2d = [
        Expression::from_dim(pg.position) * 2.0,
        Expression::from_dim(pg.one),
    ];

    let mut pt_items: Vec<(usize, HashMap<usize, f64>)> = pt.into_iter().collect();
    pt_items.sort_by_key(|(src, _)| *src);

    for pair_idx in (0..pt_items.len()).step_by(2) {
        let h = head_idx;
        head_idx += 1;

        // Query: position-based identity (scaled for hardmax)
        let scale = HARD_K * SQRT_2;
        let eq0 = expr_to_vector(&erase_q2d[0], slot_of, d_model);
        let eq1 = expr_to_vector(&erase_q2d[1], slot_of, d_model);
        copy_scaled(&mut in_proj[h * 2], &eq0, scale);
        copy_scaled(&mut in_proj[h * 2 + 1], &eq1, scale);

        // Key: position-based identity
        let ek0 = expr_to_vector(&erase_k2d[0], slot_of, d_model);
        let ek1 = expr_to_vector(&erase_k2d[1], slot_of, d_model);
        copy_row(&mut in_proj[d_model + h * 2], &ek0);
        copy_row(&mut in_proj[d_model + h * 2 + 1], &ek1);

        // Value: read from source slots
        let (src1, dsts1) = &pt_items[pair_idx];
        in_proj[2 * d_model + h * 2][*src1] = 1.0;
        for (&dst, &coeff) in dsts1 {
            out_proj[dst][h * 2] += coeff;
        }

        // Second component of the pair (if exists)
        if pair_idx + 1 < pt_items.len() {
            let (src2, dsts2) = &pt_items[pair_idx + 1];
            in_proj[2 * d_model + h * 2 + 1][*src2] = 1.0;
            for (&dst, &coeff) in dsts2 {
                out_proj[dst][h * 2 + 1] += coeff;
            }
        }

        tiebreak.push(false); // Passthrough heads use Average tie-break
    }

    debug_assert!(
        head_idx <= n_heads,
        "Layer attention: {head_idx} heads > {n_heads}"
    );

    AttnBuildResult {
        weights: AttentionWeights { in_proj, out_proj },
        tiebreak,
    }
}

// ── FFN layer weight construction ──────────────────────────────

/// Build FFN weights for one layer.
///
/// Constructs the ff_in (gate + value) and ff_out weight matrices for the
/// FFN half-layer. This includes:
/// 1. ReGLU neurons: `relu(b) * a` gating
/// 2. Passthrough neurons: copy non-reglu dims through FFN
/// 3. Erase adjustments: subtract identity for reused slots
#[allow(clippy::too_many_arguments)]
fn build_ffn_layer_weights(
    layer: &StdLayer,
    pg: &ProgramGraph,
    slot_of: &HashMap<DimId, usize>,
    d_model: usize,
    d_ffn: usize,
    internal_dims: &HashSet<DimId>,
    erased_slots: &[usize],
) -> FfnWeights {
    let mut ff_in = vec![vec![0.0; d_model]; 2 * d_ffn];
    let mut ff_out = vec![vec![0.0; d_ffn]; d_model];

    let mut j: usize = 0;

    // ── 1) ReGLU neurons ────────────────────────────────────
    let mut reglu_to_gate: HashMap<DimId, usize> = HashMap::new();
    let same_rg: HashSet<DimId> = layer.ffn.iter().copied().collect();

    for &rg_id in &layer.ffn {
        let Some(rg) = pg.all_dims.get(&rg_id) else {
            continue;
        };
        let (a_expr, b_expr) = match &rg.kind {
            DimensionKind::ReGLU { a_expr, b_expr } => (a_expr, b_expr),
            _ => continue,
        };

        // Gate weights (b_expr → neuron j)
        let gate_w = expr_to_vector(b_expr, slot_of, d_model);
        copy_row(&mut ff_in[j], &gate_w);

        // Value weights (a_expr → neuron d_ffn + j)
        let val_w = expr_to_vector(a_expr, slot_of, d_model);
        copy_row(&mut ff_in[d_ffn + j], &val_w);

        reglu_to_gate.insert(rg_id, j);

        // Output projection: wire reglu dim → slot
        if !internal_dims.contains(&rg_id)
            && let Some(&slot) = slot_of.get(&rg_id)
        {
            ff_out[slot][j] = 1.0;
        }

        j += 1;
    }

    // ── 2) Passthrough contributions ────────────────────────
    let mut pt_ffn: HashMap<usize, HashMap<usize, f64>> = HashMap::new();

    for &pd_id in &layer.persist2 {
        let Some(pd) = pg.all_dims.get(&pd_id) else {
            continue;
        };
        let Some(&dst_slot) = slot_of.get(&pd_id) else {
            continue;
        };
        let DimensionKind::Persist { expr } = &pd.kind else {
            continue;
        };

        for (&d, &c) in &expr.terms {
            if let Some(&gate_idx) = reglu_to_gate.get(&d) {
                // Wire through FFN output projection
                ff_out[dst_slot][gate_idx] += c;
            } else if let Some(&src_slot) = slot_of.get(&d)
                && !same_rg.contains(&d)
                && !internal_dims.contains(&d)
            {
                pt_ffn
                    .entry(src_slot)
                    .or_default()
                    .entry(dst_slot)
                    .and_modify(|v| *v += c)
                    .or_insert(c);
            }
        }
    }

    // Add erase contributions
    for &s in erased_slots {
        pt_ffn
            .entry(s)
            .or_default()
            .entry(s)
            .and_modify(|v| *v -= 1.0)
            .or_insert(-1.0);
    }

    // ── 3) One neuron per passthrough source slot ───────────
    let one_expr = Expression::from_dim(pg.one);

    let mut pt_items: Vec<(usize, HashMap<usize, f64>)> = pt_ffn.into_iter().collect();
    pt_items.sort_by_key(|(src, _)| *src);

    for (src, dsts) in &pt_items {
        debug_assert!(j <= d_ffn, "FFN neurons overflow: {j} > {d_ffn}");

        // Gate: identity (from one) — always fires
        let gate_w = expr_to_vector(&one_expr, slot_of, d_model);
        copy_row(&mut ff_in[j], &gate_w);

        // Value: read from source slot
        ff_in[d_ffn + j][*src] = 1.0;

        // Output: write to all destination slots
        for (&dst, &coeff) in dsts {
            ff_out[dst][j] += coeff;
        }

        j += 1;
    }

    debug_assert!(j <= d_ffn, "FFN neurons overflow: {j} > {d_ffn}");

    FfnWeights { ff_in, ff_out }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::percepta::graph::types::GraphBuilder;
    use crate::percepta::scheduler::milp_schedule;

    // ── expr_to_vector tests ─────────────────────────────────

    #[test]
    fn test_expr_to_vector_basic() {
        let builder = GraphBuilder::new();
        let one = builder.one;
        let pos = builder.position;

        // expr = 3*one + 2*pos
        let expr = Expression::from_dim(one) * 3.0 + Expression::from_dim(pos) * 2.0;

        let mut slot_of = HashMap::new();
        slot_of.insert(one, 0);
        slot_of.insert(pos, 1);

        let v = expr_to_vector(&expr, &slot_of, 4);
        assert_eq!(v.len(), 4);
        assert!(
            (v[0] - 3.0).abs() < 1e-12,
            "slot 0 should be 3.0, got {v0}",
            v0 = v[0]
        );
        assert!(
            (v[1] - 2.0).abs() < 1e-12,
            "slot 1 should be 2.0, got {v1}",
            v1 = v[1]
        );
        assert!(
            (v[2]).abs() < 1e-12,
            "slot 2 should be 0.0, got {v2}",
            v2 = v[2]
        );
        assert!(
            (v[3]).abs() < 1e-12,
            "slot 3 should be 0.0, got {v3}",
            v3 = v[3]
        );
    }

    #[test]
    fn test_expr_to_vector_zero_expression() {
        let expr = Expression::zero();
        let slot_of = HashMap::new();
        let v = expr_to_vector(&expr, &slot_of, 4);
        assert_eq!(v, vec![0.0; 4]);
    }

    #[test]
    fn test_expr_to_vector_missing_slot() {
        let builder = GraphBuilder::new();
        let one = builder.one;
        let pos = builder.position;

        // expr references pos, but only one has a slot
        let expr = Expression::from_dim(one) + Expression::from_dim(pos);

        let mut slot_of = HashMap::new();
        slot_of.insert(one, 0);
        // pos has no slot → coefficient dropped

        let v = expr_to_vector(&expr, &slot_of, 4);
        assert!((v[0] - 1.0).abs() < 1e-12, "slot 0 should be 1.0");
        assert!(v[1].abs() < 1e-12, "slot 1 should be 0.0 (no slot for pos)");
    }

    #[test]
    fn test_expr_to_vector_accumulates() {
        let builder = GraphBuilder::new();
        let one = builder.one;

        // expr = one + one = 2*one (terms get combined in HashMap)
        let expr = Expression::from_dim(one) + Expression::from_dim(one);

        let mut slot_of = HashMap::new();
        slot_of.insert(one, 5);

        let v = expr_to_vector(&expr, &slot_of, 10);
        assert!((v[5] - 2.0).abs() < 1e-12, "slot 5 should be 2.0");
    }

    // ── Internal helpers tests ───────────────────────────────

    #[test]
    fn test_internal_dims_empty_for_simple_graph() {
        let builder = GraphBuilder::new();
        let one = builder.one;

        let input_tokens = vec![Expression::from_dim(one)];
        let output_tokens = vec![Expression::from_dim(one)];

        let pg = builder.build(input_tokens, output_tokens);
        let schedule = milp_schedule(&pg, None).expect("schedule should succeed");

        let internal = compute_internal_dims(&pg, &schedule);
        // Input dims and output-referenced dims should not be internal
        assert!(!internal.contains(&pg.one), "one should not be internal");
        assert!(
            !internal.contains(&pg.position),
            "position should not be internal"
        );
    }

    #[test]
    fn test_erased_slots_no_overlap() {
        let slot_of: HashMap<DimId, usize> =
            [(0u32, 0), (1, 1), (2, 2), (3, 3)].into_iter().collect();
        let protected: HashSet<usize> = [0].into_iter().collect();

        // Before: dims 0,1,2 alive. After: dims 0,3 alive.
        // Dying: {1,2}, Born: {3}
        // Slot 1 freed, slot 2 freed, dim 3 gets slot 3 (new, not reused)
        let alive_before: HashSet<DimId> = [0, 1, 2].into_iter().collect();
        let alive_after: HashSet<DimId> = [0, 3].into_iter().collect();

        let erased = compute_erased_slots(&alive_before, &alive_after, &slot_of, &protected);
        assert!(erased.is_empty(), "no slot reuse → no erase");
    }

    #[test]
    fn test_erased_slots_with_reuse() {
        let slot_of: HashMap<DimId, usize> =
            [(0u32, 0), (1, 1), (2, 2), (3, 1)].into_iter().collect();
        let protected: HashSet<usize> = [0].into_iter().collect();

        // Before: dims 0,1,2 alive. After: dims 0,3 alive.
        // Dying: {1,2}, Born: {3}
        // Dim 3 gets slot 1 (reused from dying dim 1)
        let alive_before: HashSet<DimId> = [0, 1, 2].into_iter().collect();
        let alive_after: HashSet<DimId> = [0, 3].into_iter().collect();

        let erased = compute_erased_slots(&alive_before, &alive_after, &slot_of, &protected);
        assert_eq!(erased, vec![1], "slot 1 reused → erase");
    }

    #[test]
    fn test_erased_slots_protected() {
        let slot_of: HashMap<DimId, usize> = [(0u32, 0), (1, 1), (2, 1)].into_iter().collect();
        let protected: HashSet<usize> = [1].into_iter().collect();

        // Before: dims 0,1 alive. After: dims 0,2 alive.
        // Dying: {1}, Born: {2}
        // Dim 2 gets slot 1, but it's protected → no erase
        let alive_before: HashSet<DimId> = [0, 1].into_iter().collect();
        let alive_after: HashSet<DimId> = [0, 2].into_iter().collect();

        let erased = compute_erased_slots(&alive_before, &alive_after, &slot_of, &protected);
        assert!(erased.is_empty(), "protected slot → no erase");
    }

    // ── build_weights integration tests ──────────────────────

    #[test]
    fn test_build_weights_simple_graph() {
        // Build a simple graph: input = [one], output = [one]
        let builder = GraphBuilder::new();
        let one = builder.one;

        let input_tokens = vec![Expression::from_dim(one)];
        let output_tokens = vec![Expression::from_dim(one)];

        let pg = builder.build(input_tokens, output_tokens);
        let schedule = milp_schedule(&pg, None).expect("schedule should succeed");

        let weights = build_weights(&pg, &schedule);

        // Basic dimensional checks
        assert_eq!(weights.d_model, schedule.width); // at least schedule.width
        assert_eq!(weights.d_model % 2, 0, "d_model must be even");
        assert_eq!(weights.n_layers, schedule.num_layers);
        assert_eq!(weights.vocab_size, 1);
        assert_eq!(weights.embedding.len(), 1);
        assert_eq!(weights.unembedding.len(), 1);
        assert_eq!(weights.embedding[0].len(), weights.d_model);
        assert_eq!(weights.layers.len(), schedule.num_layers);
    }

    #[test]
    fn test_build_weights_with_reglu() {
        // Build a graph with ReGLU: output = relu(one) * position
        let mut builder = GraphBuilder::new();
        let one = builder.one;
        let pos = builder.position;

        let rg = builder.reglu(Expression::from_dim(pos), Expression::from_dim(one));

        let input_tokens = vec![Expression::from_dim(one)];
        let output_tokens = vec![rg.clone()];

        let pg = builder.build(input_tokens, output_tokens);
        let schedule = milp_schedule(&pg, None).expect("schedule should succeed");

        let weights = build_weights(&pg, &schedule);

        // Should have at least one layer with FFN
        assert!(weights.n_layers >= 1, "need at least 1 layer for ReGLU");
        assert!(weights.d_ffn >= 1, "need at least 1 FFN neuron");

        // FFN weights should be non-zero
        let has_nonzero_ffn = weights
            .layers
            .iter()
            .any(|l| l.ffn.ff_in.iter().any(|row| row.iter().any(|&v| v != 0.0)));
        assert!(
            has_nonzero_ffn,
            "FFN weights should be non-zero for ReGLU graph"
        );
    }

    #[test]
    fn test_build_weights_with_persist() {
        // Build a graph with persist: materialize expression into slot
        let mut builder = GraphBuilder::new();
        let one = builder.one;
        let pos = builder.position;

        // persist(2*one + 3*pos)
        let expr = Expression::from_dim(one) * 2.0 + Expression::from_dim(pos) * 3.0;
        let pd = builder.persist(expr);

        let input_tokens = vec![Expression::from_dim(one)];
        let output_tokens = vec![pd.clone()];

        let pg = builder.build(input_tokens, output_tokens);
        let schedule = milp_schedule(&pg, None).expect("schedule should succeed");

        let weights = build_weights(&pg, &schedule);

        // Should have at least one layer
        assert!(weights.n_layers >= 1, "need at least 1 layer for persist");
    }

    #[test]
    fn test_weight_dimensions_consistency() {
        // Build a non-trivial graph and verify all weight dimensions
        let mut builder = GraphBuilder::new();
        let one = builder.one;
        let pos = builder.position;

        let rg = builder.reglu(Expression::from_dim(pos), Expression::from_dim(one));

        let input_tokens = vec![Expression::from_dim(one), Expression::from_dim(pos)];
        let output_tokens = vec![rg.clone()];

        let pg = builder.build(input_tokens, output_tokens);
        let schedule = milp_schedule(&pg, None).expect("schedule should succeed");

        let weights = build_weights(&pg, &schedule);
        let d = weights.d_model;
        let nh = weights.n_heads;
        let df = weights.d_ffn;

        // Embedding: [vocab, d_model]
        assert_eq!(weights.embedding.len(), weights.vocab_size);
        for row in &weights.embedding {
            assert_eq!(row.len(), d);
        }

        // Unembedding: [vocab, d_model]
        assert_eq!(weights.unembedding.len(), weights.vocab_size);
        for row in &weights.unembedding {
            assert_eq!(row.len(), d);
        }

        // Per-layer checks
        for layer in &weights.layers {
            // in_proj: [3*d_model, d_model]
            assert_eq!(layer.attention.in_proj.len(), 3 * d);
            for row in &layer.attention.in_proj {
                assert_eq!(row.len(), d);
            }

            // out_proj: [d_model, d_model]
            assert_eq!(layer.attention.out_proj.len(), d);
            for row in &layer.attention.out_proj {
                assert_eq!(row.len(), d);
            }

            // ff_in: [2*d_ffn, d_model]
            assert_eq!(layer.ffn.ff_in.len(), 2 * df);
            for row in &layer.ffn.ff_in {
                assert_eq!(row.len(), d);
            }

            // ff_out: [d_model, d_ffn]
            assert_eq!(layer.ffn.ff_out.len(), d);
            for row in &layer.ffn.ff_out {
                assert_eq!(row.len(), df);
            }
        }

        // Tiebreak: [n_layers][n_heads]
        assert_eq!(weights.head_tiebreak.len(), weights.n_layers);
        for tb in &weights.head_tiebreak {
            assert_eq!(tb.len(), nh);
        }

        // Erase: [n_layers]
        assert_eq!(weights.attn_erase.len(), weights.n_layers);
        assert_eq!(weights.ffn_erase.len(), weights.n_layers);
    }

    #[test]
    fn test_embedding_zeros_protected_slots() {
        // Verify that embedding zeros out position/inv_log_pos/position_sq slots
        let builder = GraphBuilder::new();
        let one = builder.one;
        let pos = builder.position;

        // Input token that includes position
        let input_tokens = vec![Expression::from_dim(one) + Expression::from_dim(pos)];
        let output_tokens = vec![Expression::from_dim(one)];

        let pg = builder.build(input_tokens, output_tokens);
        let schedule = milp_schedule(&pg, None).expect("schedule should succeed");

        let weights = build_weights(&pg, &schedule);

        // Embedding row 0 should have position slot zeroed
        let pos_slot = schedule.slot_of.get(&pg.position).copied();
        if let Some(s) = pos_slot {
            assert!(
                weights.embedding[0][s].abs() < 1e-12,
                "position slot should be zeroed in embedding, got {v}",
                v = weights.embedding[0][s]
            );
        }
    }

    #[test]
    fn test_reglu_ffn_gate_and_value_weights() {
        // Verify ReGLU FFN weights: gate = b_expr, value = a_expr
        let mut builder = GraphBuilder::new();
        let one = builder.one;
        let pos = builder.position;

        // reglu(a=pos, b=one) → relu(one) * pos
        let rg = builder.reglu(Expression::from_dim(pos), Expression::from_dim(one));
        let rg_id = *rg
            .terms
            .keys()
            .next()
            .expect("reglu expression should have one dim");

        let input_tokens = vec![Expression::from_dim(one)];
        let output_tokens = vec![rg.clone()];

        let pg = builder.build(input_tokens, output_tokens);
        let schedule = milp_schedule(&pg, None).expect("schedule should succeed");

        let weights = build_weights(&pg, &schedule);

        // Find the layer with FFN
        let layer = weights
            .layers
            .iter()
            .find(|l| l.ffn.ff_in.iter().any(|r| r.iter().any(|&v| v != 0.0)))
            .expect("should have a layer with non-zero FFN");

        let _d = weights.d_model;
        let df = weights.d_ffn;

        // Gate row 0 should be b_expr = one (coefficient 1 at one's slot)
        let one_slot = schedule.slot_of[&pg.one];
        assert!(
            (layer.ffn.ff_in[0][one_slot] - 1.0).abs() < 1e-12,
            "gate weight should have 1.0 at one slot, got {v}",
            v = layer.ffn.ff_in[0][one_slot]
        );

        // Value row d_ffn+0 should be a_expr = pos (coefficient 1 at pos's slot)
        let pos_slot = schedule.slot_of[&pg.position];
        assert!(
            (layer.ffn.ff_in[df][pos_slot] - 1.0).abs() < 1e-12,
            "value weight should have 1.0 at pos slot, got {v}",
            v = layer.ffn.ff_in[df][pos_slot]
        );

        // Output projection: rg dim → slot, neuron 0
        let rg_slot = schedule.slot_of[&rg_id];
        assert!(
            (layer.ffn.ff_out[rg_slot][0] - 1.0).abs() < 1e-12,
            "output projection should wire rg dim to slot, got {v}",
            v = layer.ffn.ff_out[rg_slot][0]
        );
    }

    #[test]
    fn test_unembedding_matches_output_tokens() {
        let builder = GraphBuilder::new();
        let one = builder.one;
        let pos = builder.position;

        // Output = 5*one + 3*pos
        let out_expr = Expression::from_dim(one) * 5.0 + Expression::from_dim(pos) * 3.0;

        let input_tokens = vec![Expression::from_dim(one)];
        let output_tokens = vec![out_expr.clone()];

        let pg = builder.build(input_tokens, output_tokens);
        let schedule = milp_schedule(&pg, None).expect("schedule should succeed");

        let weights = build_weights(&pg, &schedule);

        // Unembedding row 0 should match out_expr
        let one_slot = schedule.slot_of[&pg.one];
        let pos_slot = schedule.slot_of[&pg.position];

        assert!(
            (weights.unembedding[0][one_slot] - 5.0).abs() < 1e-12,
            "unembedding one coeff should be 5.0"
        );
        assert!(
            (weights.unembedding[0][pos_slot] - 3.0).abs() < 1e-12,
            "unembedding pos coeff should be 3.0"
        );
    }

    #[test]
    fn test_d_model_even() {
        // d_model should always be even (for 2D attention heads)
        let mut builder = GraphBuilder::new();
        let one = builder.one;

        let rg1 = builder.reglu(Expression::from_dim(one), Expression::from_dim(one));
        let rg2 = builder.reglu(Expression::from_dim(one), Expression::from_dim(one));

        let input_tokens = vec![Expression::from_dim(one)];
        let output_tokens = vec![rg1.clone(), rg2.clone()];

        let pg = builder.build(input_tokens, output_tokens);
        let schedule = milp_schedule(&pg, None).expect("schedule should succeed");

        let weights = build_weights(&pg, &schedule);

        assert_eq!(weights.d_model % 2, 0, "d_model must be even");
        assert_eq!(weights.n_heads, weights.d_model / 2);
    }
}
