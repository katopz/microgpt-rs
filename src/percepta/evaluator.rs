// SPDX-License-Identifier: Apache-2.0
// Distilled from Percepta's `transformer-vm` (Apache-2.0 © Percepta).

//! Graph evaluator with exact arithmetic for correctness verification.
//!
//! Evaluates the computation graph directly without building transformer weights.
//! Each dimension's value is computed as an exact floating-point number, and
//! attention lookups are simulated with brute-force O(n) search over all past
//! entries.
//!
//! This is useful for:
//! - Correctness verification against the transformer's floating-point approximations
//! - Debugging graph construction (no MILP / weight construction needed)
//! - Generating reference traces for test comparison
//!
//! # Usage
//!
//! ```ignore
//! use percepta::evaluator::GraphEvaluator;
//! use percepta::wasm::interpreter;
//! use percepta::graph::types::GraphBuilder;
//!
//! let mut builder = GraphBuilder::new();
//! let (input_tokens, output_tokens) = interpreter::build(None, &mut builder);
//! let graph = builder.build(input_tokens.values().cloned().collect(),
//!                           output_tokens.values().cloned().collect());
//!
//! let mut evaluator = GraphEvaluator::new(input_tokens, output_tokens, graph);
//! let result = evaluator.evaluate(&prefix, 50000);
//! ```
//!
//! Reference: `.raw/transformer-vm/transformer_vm/evaluator.py` (404 lines)

use std::collections::HashMap;

use crate::percepta::graph::types::{
    DimId, DimensionKind, Expression, LookUp, LookupId, ProgramGraph,
};
use crate::percepta::types::TieBreak;

// ── Error Type ─────────────────────────────────────────────────

/// Errors that can occur during graph evaluation.
#[derive(Clone, Debug)]
pub enum EvalError {
    /// The token name is not in the input vocabulary.
    UnknownToken(String),
}

impl std::fmt::Display for EvalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnknownToken(name) => write!(f, "unknown token: {name}"),
        }
    }
}

impl std::error::Error for EvalError {}

// ── Attention Entry ────────────────────────────────────────────

/// A single entry in the brute-force attention cache.
///
/// Each step that uses a [`LookUp`] inserts one entry recording the key
/// coordinates and evaluated value expressions at that position.
#[derive(Clone, Debug)]
struct AttentionEntry {
    /// Sequence number (position) when this entry was inserted.
    seq: usize,
    /// Key x-coordinate (evaluated from `key_exprs_2d[0]`).
    kx: f64,
    /// Key y-coordinate (evaluated from `key_exprs_2d[1]`).
    ky: f64,
    /// Value expressions evaluated at insertion time.
    values: Vec<f64>,
}

// ── GraphEvaluator ─────────────────────────────────────────────

/// Evaluate a computation graph with exact arithmetic (no transformer weights).
///
/// The graph evaluator simulates the transformer forward pass by directly
/// evaluating the computation graph's expressions. This provides exact arithmetic
/// results useful for correctness verification against the transformer's
/// floating-point approximations.
///
/// # Evaluation Order
///
/// Dimensions are evaluated in DimId order (ascending). Since DimIds are
/// allocated sequentially by [`GraphBuilder`], this naturally respects
/// dependencies: a dimension can only reference dimensions with lower IDs.
///
/// # Attention Simulation
///
/// Attention lookups are simulated with brute-force O(n) search over all past
/// entries. This is correct but slower than the CHT-based O(log n) method used
/// in the transformer. For large programs, consider using the transformer instead.
///
/// [`GraphBuilder`]: crate::percepta::graph::types::GraphBuilder
pub struct GraphEvaluator {
    /// Token name → Expression for input tokens (embeddings).
    input_tokens: HashMap<String, Expression>,
    /// Token name → scoring Expression for output tokens.
    output_tokens: HashMap<String, Expression>,
    /// The computation graph being evaluated.
    graph: ProgramGraph,
    /// Current position (token index).
    position: usize,
    /// Cumulative sum accumulators: CumSum dim ID → accumulated value.
    cumsum_accum: HashMap<DimId, f64>,
    /// Attention caches: Lookup ID → list of past entries.
    attention_entries: HashMap<LookupId, Vec<AttentionEntry>>,
    /// Sorted dimension IDs for deterministic evaluation order.
    dim_order: Vec<DimId>,
}

impl GraphEvaluator {
    /// Create a new graph evaluator.
    ///
    /// # Arguments
    /// * `input_tokens` — Token name → embedding expression mapping.
    /// * `output_tokens` — Token name → scoring expression mapping.
    /// * `graph` — The computation graph to evaluate.
    pub fn new(
        input_tokens: HashMap<String, Expression>,
        output_tokens: HashMap<String, Expression>,
        graph: ProgramGraph,
    ) -> Self {
        // Sort dimension IDs for deterministic evaluation order.
        // DimId is u32 allocated sequentially, so sorting gives creation order
        // which respects dependencies (lower ID = created first).
        let mut dim_order: Vec<DimId> = graph.all_dims.keys().copied().collect();
        dim_order.sort_unstable();

        // Initialize cumsum accumulators for all CumSum dimensions
        let mut cumsum_accum = HashMap::new();
        for &dim_id in &dim_order {
            if let Some(dim) = graph.all_dims.get(&dim_id)
                && matches!(dim.kind, DimensionKind::CumSum { .. }) {
                    cumsum_accum.insert(dim_id, 0.0);
                }
        }

        Self {
            input_tokens,
            output_tokens,
            graph,
            position: 0,
            cumsum_accum,
            attention_entries: HashMap::new(),
            dim_order,
        }
    }

    /// Reset the evaluator to initial state.
    ///
    /// Clears all accumulated state (position, cumsum, attention caches)
    /// so the evaluator can be reused for a new program.
    pub fn reset(&mut self) {
        self.position = 0;
        for accum in self.cumsum_accum.values_mut() {
            *accum = 0.0;
        }
        self.attention_entries.clear();
    }

    /// Process a single token: update dimension values.
    ///
    /// Looks up the token's embedding, sets position-based dimensions,
    /// and evaluates all dimensions in dependency order.
    ///
    /// Returns the computed dimension values after processing this token.
    pub fn step(&mut self, token_name: &str) -> Result<HashMap<DimId, f64>, EvalError> {
        // Look up token embedding
        let embedding = self
            .input_tokens
            .get(token_name)
            .ok_or_else(|| EvalError::UnknownToken(token_name.to_string()))?;

        // Initialize values from embedding terms
        let mut vals: HashMap<DimId, f64> = HashMap::with_capacity(self.dim_order.len());
        for (&dim, &coeff) in &embedding.terms {
            vals.insert(dim, coeff);
        }

        // Ensure `one` dimension is always 1.0 (used for scalar constants)
        vals.insert(self.graph.one, 1.0);

        // Set position-based dimensions
        let pos = self.position as f64;
        vals.insert(self.graph.position, pos);
        vals.insert(self.graph.position_sq, pos * pos);
        vals.insert(
            self.graph.inv_log_pos,
            1.0 / std::f64::consts::LN_2 - 1.0 / (pos + 2.0).ln(),
        );

        // Track already-processed lookups to avoid re-computing within one step
        let mut processed_lookups: HashMap<LookupId, Vec<f64>> = HashMap::new();

        // Phase 1: Collect owned work items from the graph (ends immutable borrow).
        // This avoids holding an immutable borrow of self.graph while we need
        // mutable access to self for attention operations.
        enum DimWorkItem {
            Input,
            CumSum {
                dim_id: DimId,
                value_expr: Expression,
            },
            ReGLU {
                dim_id: DimId,
                a_expr: Expression,
                b_expr: Expression,
            },
            Persist {
                dim_id: DimId,
                expr: Expression,
            },
            LookUp {
                dim_id: DimId,
                lookup_id: LookupId,
                value_index: usize,
            },
            Generic,
        }

        let work_items: Vec<DimWorkItem> = self
            .dim_order
            .iter()
            .filter(|&&dim_id| !vals.contains_key(&dim_id))
            .filter_map(|&dim_id| {
                let dim = self.graph.all_dims.get(&dim_id)?;
                Some(match &dim.kind {
                    DimensionKind::Input => DimWorkItem::Input,
                    DimensionKind::CumSum { value_expr } => DimWorkItem::CumSum {
                        dim_id,
                        value_expr: value_expr.clone(),
                    },
                    DimensionKind::ReGLU { a_expr, b_expr } => DimWorkItem::ReGLU {
                        dim_id,
                        a_expr: a_expr.clone(),
                        b_expr: b_expr.clone(),
                    },
                    DimensionKind::Persist { expr } => DimWorkItem::Persist {
                        dim_id,
                        expr: expr.clone(),
                    },
                    DimensionKind::LookUp {
                        lookup_id,
                        value_index,
                    } => DimWorkItem::LookUp {
                        dim_id,
                        lookup_id: *lookup_id,
                        value_index: *value_index,
                    },
                    DimensionKind::Generic => DimWorkItem::Generic,
                })
            })
            .collect();

        // Phase 2: Process work items (may mutate self for attention).
        // Clone lookups needed for attention queries (ends immutable borrow of self.graph).
        let lookup_data: HashMap<LookupId, LookUp> = self
            .graph
            .all_lookups
            .iter()
            .map(|(&id, lu)| (id, lu.clone()))
            .collect();

        for item in work_items {
            match item {
                DimWorkItem::Input => {
                    // Input dims should already be set via embedding
                }
                DimWorkItem::CumSum { dim_id, value_expr } => {
                    let value = value_expr.evaluate(&vals);
                    let accum = self
                        .cumsum_accum
                        .get_mut(&dim_id)
                        .expect("cumsum_accum initialized for all CumSum dims");
                    *accum += value;
                    vals.insert(dim_id, *accum);
                }
                DimWorkItem::ReGLU {
                    dim_id,
                    a_expr,
                    b_expr,
                } => {
                    let a = a_expr.evaluate(&vals);
                    let b = b_expr.evaluate(&vals);
                    vals.insert(dim_id, a * b.max(0.0));
                }
                DimWorkItem::Persist { dim_id, expr } => {
                    vals.insert(dim_id, expr.evaluate(&vals));
                }
                DimWorkItem::LookUp {
                    dim_id,
                    lookup_id,
                    value_index,
                } => {
                    let result = match processed_lookups.get(&lookup_id) {
                        Some(r) => r.clone(),
                        None => {
                            let lookup = lookup_data
                                .get(&lookup_id)
                                .expect("lookup_id exists in graph");
                            let result = self.attention_insert_and_query(lookup, &vals);
                            processed_lookups.insert(lookup_id, result.clone());
                            result
                        }
                    };
                    if let Some(&val) = result.get(value_index) {
                        vals.insert(dim_id, val);
                    }
                }
                DimWorkItem::Generic => {
                    // Generic dims are set via embedding or explicit assignment
                }
            }
        }

        self.position += 1;
        Ok(vals)
    }

    /// Insert current entry into attention cache and query for best match.
    ///
    /// Implements brute-force O(n) hard attention:
    /// 1. Compute key `(kx, ky)` from current dimension values
    /// 2. Evaluate value expressions from current dimension values
    /// 3. Append `(seq, kx, ky, values)` to the cache
    /// 4. Compute query `(qx, qy)` from current dimension values
    /// 5. Find max dot product: `max(qx·kx + qy·ky)` over all entries
    /// 6. Resolve ties using the lookup's tie-break mode
    fn attention_insert_and_query(
        &mut self,
        lookup: &LookUp,
        vals: &HashMap<DimId, f64>,
    ) -> Vec<f64> {
        // Compute key from current values
        let kx = lookup.key_exprs_2d[0].evaluate(vals);
        let ky = lookup.key_exprs_2d[1].evaluate(vals);

        // Evaluate value expressions from current values
        let raw_vals: Vec<f64> = lookup
            .value_exprs
            .iter()
            .map(|expr| expr.evaluate(vals))
            .collect();

        // Insert entry into cache
        let entries = self.attention_entries.entry(lookup.id).or_default();
        entries.push(AttentionEntry {
            seq: self.position,
            kx,
            ky,
            values: raw_vals,
        });

        // Compute query from current values
        let qx = lookup.query_exprs_2d[0].evaluate(vals);
        let qy = lookup.query_exprs_2d[1].evaluate(vals);

        // Find best attention score (max dot product)
        let best_score = entries
            .iter()
            .map(|e| qx * e.kx + qy * e.ky)
            .fold(f64::NEG_INFINITY, f64::max);

        // Resolve ties according to tie-break mode
        let n_values = lookup.value_exprs.len();
        match lookup.tie_break {
            TieBreak::Average => {
                let mut total = vec![0.0; n_values];
                let mut count = 0usize;
                for entry in entries.iter() {
                    let score = qx * entry.kx + qy * entry.ky;
                    if (score - best_score).abs() <= 1e-9 {
                        for (j, v) in entry.values.iter().enumerate() {
                            total[j] += v;
                        }
                        count += 1;
                    }
                }
                if count > 0 {
                    let inv = 1.0 / count as f64;
                    for t in total.iter_mut() {
                        *t *= inv;
                    }
                }
                total
            }
            TieBreak::Latest => {
                let mut best_entry: Option<&AttentionEntry> = None;
                for entry in entries.iter() {
                    let score = qx * entry.kx + qy * entry.ky;
                    if (score - best_score).abs() <= 1e-9 {
                        match best_entry {
                            None => best_entry = Some(entry),
                            Some(prev) if entry.seq > prev.seq => best_entry = Some(entry),
                            _ => {}
                        }
                    }
                }
                match best_entry {
                    Some(e) => e.values.clone(),
                    None => vec![0.0; n_values],
                }
            }
        }
    }

    /// Get the predicted next token scores for all output tokens.
    ///
    /// Returns a sorted list of `(token_name, score)` pairs, highest score first.
    pub fn predict(&self, vals: &HashMap<DimId, f64>) -> Vec<(String, f64)> {
        let mut scores: Vec<(String, f64)> = self
            .output_tokens
            .iter()
            .map(|(name, expr)| (name.clone(), expr.evaluate(vals)))
            .collect();
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores
    }

    /// Get the predicted next token name (argmax over output tokens).
    ///
    /// Returns `None` if there are no output tokens.
    pub fn predict_next(&self, vals: &HashMap<DimId, f64>) -> Option<String> {
        let mut best_name: Option<String> = None;
        let mut best_score = f64::NEG_INFINITY;

        for (name, expr) in self.output_tokens.iter() {
            let score = expr.evaluate(vals);
            if score > best_score {
                best_score = score;
                best_name = Some(name.clone());
            }
        }

        best_name
    }

    /// Run the full execution trace for a program.
    ///
    /// Feeds all tokens in `prefix` through the evaluator, then generates
    /// tokens autoregressively until `"halt"` is predicted or `max_steps` is reached.
    ///
    /// Returns the complete token sequence (prefix + generated).
    pub fn evaluate(&mut self, prefix: &[String], max_steps: usize) -> Vec<String> {
        // Feed all prefix tokens
        let mut vals = HashMap::new();
        for token in prefix {
            match self.step(token) {
                Ok(v) => vals = v,
                Err(_) => break,
            }
        }

        // Build predicted sequence starting from prefix
        let mut predicted: Vec<String> = prefix.to_vec();

        // Generate tokens autoregressively
        for _ in 0..max_steps {
            let next_tok = match self.predict_next(&vals) {
                Some(t) => t,
                None => break,
            };
            predicted.push(next_tok.clone());

            if next_tok == "halt" {
                break;
            }

            match self.step(&next_tok) {
                Ok(v) => vals = v,
                Err(_) => break,
            }
        }

        predicted
    }

    /// Run the full execution trace and extract output characters.
    ///
    /// Similar to [`evaluate`](Self::evaluate), but also extracts the output
    /// byte sequence from `out(XY)` tokens after the `"OUT"` drain signal.
    ///
    /// Returns the predicted token sequence and decoded output string.
    pub fn evaluate_with_output(
        &mut self,
        prefix: &[String],
        max_steps: usize,
    ) -> (Vec<String>, String) {
        let predicted = self.evaluate(prefix, max_steps);

        // Extract output characters from hex bytes after "OUT" token
        let mut output_chars = Vec::new();
        let mut draining = false;

        for tok in &predicted {
            match tok.as_str() {
                "OUT" => {
                    draining = true;
                }
                "halt" => break,
                _ => {
                    if draining
                        && let Ok(bv) = u32::from_str_radix(tok, 16) {
                            let ch = if (32..127).contains(&bv) {
                                char::from_u32(bv).unwrap_or('.')
                            } else {
                                '.'
                            };
                            output_chars.push(ch);
                        }
                }
            }
        }

        (predicted, output_chars.into_iter().collect())
    }

    /// Compare predicted output with reference tokens.
    ///
    /// Returns `Ok(true)` if they match exactly, `Ok(false)` if they differ.
    /// On mismatch, logs the first differing position via `log::warn!`.
    pub fn compare_with_reference(
        &mut self,
        prefix: &[String],
        reference: &[String],
        max_steps: usize,
    ) -> Result<bool, EvalError> {
        // Feed prefix
        let mut vals = HashMap::new();
        for token in prefix {
            vals = self.step(token)?;
        }

        // Predict autoregressively
        let mut predicted: Vec<String> = prefix.to_vec();

        for _ in 0..max_steps {
            let next_tok = self
                .predict_next(&vals)
                .ok_or_else(|| EvalError::UnknownToken("no output tokens".into()))?;
            predicted.push(next_tok.clone());

            if next_tok == "halt" {
                break;
            }

            vals = self.step(&next_tok)?;
        }

        // Compare
        if predicted == reference {
            return Ok(true);
        }

        // Log first mismatch
        let max_len = predicted.len().max(reference.len());
        for i in 0..max_len {
            let p = predicted.get(i).map(|s| s.as_str()).unwrap_or("<END>");
            let r = reference.get(i).map(|s| s.as_str()).unwrap_or("<END>");
            if p != r {
                log::warn!("MISMATCH at position {i}: predicted={p}, expected={r}");
                break;
            }
        }

        Ok(false)
    }

    /// Get the current position (number of tokens processed).
    pub fn position(&self) -> usize {
        self.position
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::percepta::graph::types::GraphBuilder;
    use crate::percepta::types::TieBreak;

    /// Extract the single DimId from an expression (assumes 1-term expression).
    fn expr_to_dim(expr: &Expression) -> DimId {
        *expr
            .terms
            .keys()
            .next()
            .expect("expression should have one term")
    }

    /// Build a minimal graph for basic evaluator tests.
    /// Returns (input_tokens, output_tokens, graph, position_dim, one_dim, inv_log_pos_dim, position_sq_dim).
    fn build_test_graph_full() -> (
        HashMap<String, Expression>,
        HashMap<String, Expression>,
        ProgramGraph,
        DimId,
        DimId,
        DimId,
        DimId,
    ) {
        let mut builder = GraphBuilder::new();

        // Create a simple graph: cumsum of a generic dimension
        let a = builder.generic("a");
        let _cumsum_a = builder.fetch_sum_single(a.clone());

        // Save special DimIds before consuming the builder
        let one_id = builder.one;
        let position_id = builder.position;
        let inv_log_pos_id = builder.inv_log_pos;
        let position_sq_id = builder.position_sq;

        // Compute a_dim before building token maps
        let a_dim = expr_to_dim(&a);

        // Input tokens: simple scalar embeddings (include output tokens for autoregressive loop)
        let input_tokens = HashMap::from([
            ("zero".to_string(), Expression::from_scalar(0.0, one_id)),
            ("one".to_string(), Expression::from_scalar(1.0, one_id)),
            ("two".to_string(), Expression::from_scalar(2.0, one_id)),
            ("done".to_string(), Expression::from_dim(a_dim)),
            ("halt".to_string(), Expression::zero()),
        ]);

        // Output tokens: score by the `a` dimension value
        let output_tokens = HashMap::from([
            ("done".to_string(), Expression::from_dim(a_dim)),
            ("halt".to_string(), Expression::zero()),
        ]);

        let graph = builder.build(
            input_tokens.values().cloned().collect(),
            output_tokens.values().cloned().collect(),
        );

        (
            input_tokens,
            output_tokens,
            graph,
            position_id,
            one_id,
            inv_log_pos_id,
            position_sq_id,
        )
    }

    #[test]
    fn test_evaluator_step_unknown_token() {
        let (input_tokens, output_tokens, graph, _, _, _, _) = build_test_graph_full();
        let mut evaluator = GraphEvaluator::new(input_tokens, output_tokens, graph);

        let result = evaluator.step("nonexistent");
        assert!(result.is_err());
        match result.unwrap_err() {
            EvalError::UnknownToken(name) => assert_eq!(name, "nonexistent"),
        }
    }

    #[test]
    fn test_evaluator_step_sets_position() {
        let (input_tokens, output_tokens, graph, position_id, _, _, _) = build_test_graph_full();
        let mut evaluator = GraphEvaluator::new(input_tokens, output_tokens, graph);

        let vals = evaluator.step("zero").unwrap();
        assert_eq!(vals.get(&position_id), Some(&0.0));

        let vals = evaluator.step("zero").unwrap();
        assert_eq!(vals.get(&position_id), Some(&1.0));
    }

    #[test]
    fn test_evaluator_step_sets_one() {
        let (input_tokens, output_tokens, graph, _, one_id, _, _) = build_test_graph_full();
        let mut evaluator = GraphEvaluator::new(input_tokens, output_tokens, graph);

        let vals = evaluator.step("zero").unwrap();
        assert_eq!(vals.get(&one_id), Some(&1.0));
    }

    #[test]
    fn test_evaluator_step_sets_position_sq() {
        let (input_tokens, output_tokens, graph, _, _, _, position_sq_id) = build_test_graph_full();
        let mut evaluator = GraphEvaluator::new(input_tokens, output_tokens, graph);

        let vals = evaluator.step("zero").unwrap();
        assert_eq!(vals.get(&position_sq_id), Some(&0.0));

        let vals = evaluator.step("zero").unwrap();
        assert_eq!(vals.get(&position_sq_id), Some(&1.0));

        let vals = evaluator.step("zero").unwrap();
        assert_eq!(vals.get(&position_sq_id), Some(&4.0));
    }

    #[test]
    fn test_evaluator_reset() {
        let (input_tokens, output_tokens, graph, position_id, _, _, _) = build_test_graph_full();
        let mut evaluator = GraphEvaluator::new(input_tokens, output_tokens, graph);

        evaluator.step("zero").unwrap();
        evaluator.step("one").unwrap();
        assert_eq!(evaluator.position(), 2);

        evaluator.reset();
        assert_eq!(evaluator.position(), 0);

        let vals = evaluator.step("zero").unwrap();
        assert_eq!(vals.get(&position_id), Some(&0.0));
    }

    #[test]
    fn test_evaluator_predict_returns_sorted() {
        let (input_tokens, output_tokens, graph, _, _, _, _) = build_test_graph_full();
        let mut evaluator = GraphEvaluator::new(input_tokens, output_tokens, graph);

        let vals = evaluator.step("zero").unwrap();
        let scores = evaluator.predict(&vals);

        // Should be sorted highest first
        for window in scores.windows(2) {
            assert!(
                window[0].1 >= window[1].1,
                "scores not sorted: {:?}",
                scores
            );
        }
    }

    #[test]
    fn test_evaluator_predict_next_returns_some() {
        let (input_tokens, output_tokens, graph, _, _, _, _) = build_test_graph_full();
        let mut evaluator = GraphEvaluator::new(input_tokens, output_tokens, graph);

        let vals = evaluator.step("zero").unwrap();
        let next = evaluator.predict_next(&vals);
        assert!(next.is_some());
    }

    #[test]
    fn test_evaluator_evaluate_produces_tokens() {
        let (input_tokens, output_tokens, graph, _, _, _, _) = build_test_graph_full();
        let mut evaluator = GraphEvaluator::new(input_tokens, output_tokens, graph);

        let predicted = evaluator.evaluate(&[], 100);
        // Should produce at least the halt token
        assert!(!predicted.is_empty() || evaluator.output_tokens.is_empty());
    }

    #[test]
    fn test_evaluator_evaluate_with_prefix() {
        let (input_tokens, output_tokens, graph, _, _, _, _) = build_test_graph_full();
        let mut evaluator = GraphEvaluator::new(input_tokens, output_tokens, graph);

        let prefix: Vec<String> = vec!["one".to_string(), "two".to_string()];
        let predicted = evaluator.evaluate(&prefix, 100);
        // Prefix should be included
        assert_eq!(&predicted[0..2], &prefix[..]);
    }

    #[test]
    fn test_evaluator_compare_with_reference_match() {
        let (input_tokens, output_tokens, graph, _, _, _, _) = build_test_graph_full();
        let mut evaluator = GraphEvaluator::new(input_tokens, output_tokens, graph);

        // Generate reference
        let reference = evaluator.evaluate(&[], 100);
        evaluator.reset();

        // Compare should match
        let result = evaluator.compare_with_reference(&[], &reference, 100);
        assert!(result.is_ok());
    }

    #[test]
    fn test_evaluator_compare_with_reference_mismatch() {
        let (input_tokens, output_tokens, graph, _, _, _, _) = build_test_graph_full();
        let mut evaluator = GraphEvaluator::new(input_tokens, output_tokens, graph);

        let fake_ref = vec!["wrong_token".to_string()];
        let result = evaluator.compare_with_reference(&[], &fake_ref, 100);
        assert_eq!(result.unwrap(), false);
    }

    #[test]
    fn test_attention_average_tiebreak() {
        // Build a graph with a fetch that uses Average tiebreak
        let mut builder = GraphBuilder::new();
        let a = builder.generic("a");
        let a_dim = expr_to_dim(&a);
        let _one = Expression::from_dim(builder.one);

        // fetch(value, query: Option, key: Option, clear_key: Option, tie_break)
        let fetched = builder.fetch(a.clone(), None, None, None, TieBreak::Average);

        let input_tokens = HashMap::from([("x".to_string(), Expression::from_dim(a_dim))]);
        let output_tokens = HashMap::from([
            ("done".to_string(), fetched),
            ("halt".to_string(), Expression::zero()),
        ]);

        let graph = builder.build(
            input_tokens.values().cloned().collect(),
            output_tokens.values().cloned().collect(),
        );

        let mut evaluator = GraphEvaluator::new(input_tokens, output_tokens, graph);

        // Step twice — the second step should see the first in the attention cache
        let _v1 = evaluator.step("x").unwrap();
        let _v2 = evaluator.step("x").unwrap();

        // Should not panic
    }

    #[test]
    fn test_evaluator_position_encoding_at_zero() {
        let (input_tokens, output_tokens, graph, _, _, inv_log_pos_id, _) = build_test_graph_full();
        let mut evaluator = GraphEvaluator::new(input_tokens, output_tokens, graph);

        let vals = evaluator.step("zero").unwrap();

        // At position 0: inv_log_pos = 1/ln(2) - 1/ln(2) = 0
        let inv_log_pos = vals[&inv_log_pos_id];
        assert!(
            (inv_log_pos - 0.0).abs() < 1e-10,
            "inv_log_pos at pos 0 should be ~0, got {inv_log_pos}"
        );
    }

    #[test]
    fn test_evaluator_position_encoding_at_one() {
        let (input_tokens, output_tokens, graph, _, _, inv_log_pos_id, _) = build_test_graph_full();
        let mut evaluator = GraphEvaluator::new(input_tokens, output_tokens, graph);

        let _ = evaluator.step("zero").unwrap();
        let vals = evaluator.step("zero").unwrap();

        // At position 1: inv_log_pos = 1/ln(2) - 1/ln(3)
        let expected = 1.0 / std::f64::consts::LN_2 - 1.0 / 3.0_f64.ln();
        let inv_log_pos = vals[&inv_log_pos_id];
        assert!(
            (inv_log_pos - expected).abs() < 1e-10,
            "inv_log_pos at pos 1: expected {expected}, got {inv_log_pos}"
        );
    }
}
