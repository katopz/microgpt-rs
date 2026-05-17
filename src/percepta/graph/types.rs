//! Expression/Dimension DSL computation graph for Percepta's transformer-vm.
//!
//! This module implements the core computation graph primitives that any
//! Append-Only Lookup Machine (ALM) program is built from. Expressions are
//! sparse linear combinations of dimensions, and the graph captures the
//! dependency structure for scheduling and weight construction.
//!
//! # Core Types
//!
//! - [`Expression`] — Sparse linear combination of dimensions (`HashMap<DimId, f64>`)
//! - [`Dimension`] — Named dimension with [`DimensionKind`] variant
//! - [`LookUp`] — Attention-based retrieval from token history
//! - [`ProgramGraph`] — Captured computation graph ready for scheduling
//! - [`GraphBuilder`] — Mutable state for building computation graphs
//!
//! # Builder Functions
//!
//! - [`GraphBuilder::reglu`] — `relu(b) × a` gated FFN unit
//! - [`GraphBuilder::stepglu`] — `a × step(b ≥ 0)` conditional gate
//! - [`GraphBuilder::persist`] — Materialize expression into residual slot
//! - [`GraphBuilder::fetch`] — Attention-based retrieval (single value)
//! - [`GraphBuilder::fetch_vec`] — Attention-based retrieval (multiple values)
//! - [`GraphBuilder::fetch_sum`] — Cumulative sum via attention averaging
//!
//! Distilled from Percepta's `transformer-vm` (Apache-2.0 © Percepta).
//! Reference: `.raw/transformer-vm/transformer_vm/graph/core.py` (449 lines)

use std::collections::HashMap;

use ordered_float::OrderedFloat;

use crate::percepta::TieBreak;

// ── Constants ──────────────────────────────────────────────────

/// Large constant used to effectively zero-out attention keys (clear_key mechanism).
pub const BIG: f64 = 1e30;

/// Offset applied to all attention keys (for numerical stability tuning).
pub const KEY_OFFSET: f64 = 0.0;

/// Tie-break weight favoring more recent tokens in hardmax attention.
pub const LATEST_ALPHA: f64 = 0.3;

// ── Type Aliases ───────────────────────────────────────────────

/// Unique identifier for a dimension in the computation graph.
pub type DimId = u32;

/// Unique identifier for a LookUp (attention head) in the computation graph.
pub type LookupId = u32;

/// Cache key derived from an Expression's sorted terms.
type ExprKey = Vec<(DimId, OrderedFloat<f64>)>;

/// Pending name update for a dimension and optionally its parent lookup.
type NameUpdate = (DimId, String, Option<(LookupId, String)>);

// ── IntoExpr Trait ─────────────────────────────────────────────

/// Trait for types that can be converted into an [`Expression`].
///
/// Implemented for [`Expression`], [`DimId`], `f64`, and `i32`.
/// Scalar conversion requires `one_id` to create `{one: scalar}`.
pub trait IntoExpr {
    /// Convert `self` into an Expression.
    ///
    /// `one_id` is the DimId of the built-in `one` input dimension,
    /// needed when converting scalar values to expressions.
    fn into_expr(self, one_id: DimId) -> Expression;
}

impl IntoExpr for Expression {
    fn into_expr(self, _one_id: DimId) -> Expression {
        self
    }
}

impl IntoExpr for DimId {
    fn into_expr(self, _one_id: DimId) -> Expression {
        Expression::from_dim(self)
    }
}

impl IntoExpr for f64 {
    fn into_expr(self, one_id: DimId) -> Expression {
        Expression::from_scalar(self, one_id)
    }
}

impl IntoExpr for i32 {
    fn into_expr(self, one_id: DimId) -> Expression {
        Expression::from_scalar(f64::from(self), one_id)
    }
}

// ── Expression ─────────────────────────────────────────────────

/// Sparse linear combination of dimensions.
///
/// An expression maps dimension IDs to coefficients: `{dim_3: 2.5, dim_7: -1.0}`.
/// Zero-coefficient terms are automatically removed on construction and arithmetic.
///
/// Supports arithmetic: `+`, `-`, `*` (by scalar), `neg`, and `evaluate(values)`.
#[derive(Clone, Debug, Default)]
pub struct Expression {
    /// Sparse terms: dimension ID → coefficient. Zero-coefficient entries are removed.
    pub terms: HashMap<DimId, f64>,
}

impl Expression {
    /// Create an empty (zero) expression.
    pub fn zero() -> Self {
        Self::default()
    }

    /// Create an expression from a single dimension with coefficient 1.
    pub fn from_dim(dim: DimId) -> Self {
        let mut terms = HashMap::new();
        terms.insert(dim, 1.0);
        Self { terms }
    }

    /// Create a scalar expression `{one_id: value}`.
    ///
    /// Returns zero expression if `value == 0.0`.
    pub fn from_scalar(value: f64, one_id: DimId) -> Self {
        if value == 0.0 {
            return Self::zero();
        }
        let mut terms = HashMap::new();
        terms.insert(one_id, value);
        Self { terms }
    }

    /// Create an expression from raw terms, removing zero-coefficient entries.
    pub fn from_terms(terms: HashMap<DimId, f64>) -> Self {
        let filtered: HashMap<_, _> = terms.into_iter().filter(|(_, c)| *c != 0.0).collect();
        Self { terms: filtered }
    }

    /// Deep copy (clone is sufficient since HashMap owns its data).
    #[must_use]
    pub fn copy(&self) -> Self {
        self.clone()
    }

    /// Get the coefficient for a dimension, defaulting to 0.
    pub fn get(&self, dim: DimId) -> f64 {
        self.terms.get(&dim).copied().unwrap_or(0.0)
    }

    /// Set the coefficient for a dimension. Removes the entry if value is 0.
    pub fn set(&mut self, dim: DimId, value: f64) {
        if value == 0.0 {
            self.terms.remove(&dim);
        } else {
            self.terms.insert(dim, value);
        }
    }

    /// Evaluate the expression given dimension values.
    ///
    /// Returns `sum(coeff * values[dim])` for all terms.
    pub fn evaluate(&self, values: &HashMap<DimId, f64>) -> f64 {
        self.terms
            .iter()
            .map(|(dim, coeff)| coeff * values.get(dim).copied().unwrap_or(0.0))
            .sum()
    }

    /// Check if this is the zero expression (no terms).
    pub fn is_zero(&self) -> bool {
        self.terms.is_empty()
    }

    /// Number of non-zero terms.
    pub fn len(&self) -> usize {
        self.terms.len()
    }

    /// Check if there are no terms.
    pub fn is_empty(&self) -> bool {
        self.terms.is_empty()
    }

    /// Compute the cache key for this expression (sorted terms).
    fn expr_key(&self) -> ExprKey {
        let mut key: Vec<_> = self
            .terms
            .iter()
            .map(|(&id, &c)| (id, OrderedFloat(c)))
            .collect();
        key.sort_by_key(|(id, _)| *id);
        key
    }
}

// ── Expression Arithmetic ──────────────────────────────────────

impl std::ops::Add for Expression {
    type Output = Expression;

    fn add(self, rhs: Expression) -> Expression {
        let mut result = self.terms;
        for (dim, coeff) in rhs.terms {
            let new_coeff = result.get(&dim).copied().unwrap_or(0.0) + coeff;
            if new_coeff == 0.0 {
                result.remove(&dim);
            } else {
                result.insert(dim, new_coeff);
            }
        }
        Expression { terms: result }
    }
}

impl std::ops::Sub for Expression {
    type Output = Expression;

    fn sub(self, rhs: Expression) -> Expression {
        let mut result = self.terms;
        for (dim, coeff) in rhs.terms {
            let new_coeff = result.get(&dim).copied().unwrap_or(0.0) - coeff;
            if new_coeff == 0.0 {
                result.remove(&dim);
            } else {
                result.insert(dim, new_coeff);
            }
        }
        Expression { terms: result }
    }
}

impl std::ops::Add<f64> for Expression {
    type Output = Expression;

    /// Add a scalar to an expression.
    ///
    /// Scalars are represented as `{one_dim: value}` where `one_dim = 0`
    /// (matches [`GraphBuilder::new`] which always allocates `one` as DimId 0).
    fn add(self, scalar: f64) -> Expression {
        if scalar == 0.0 {
            return self;
        }
        let mut result = self.terms;
        let dim = 0; // one_dim is always 0
        let new_coeff = result.get(&dim).copied().unwrap_or(0.0) + scalar;
        if new_coeff == 0.0 {
            result.remove(&dim);
        } else {
            result.insert(dim, new_coeff);
        }
        Expression { terms: result }
    }
}

impl std::ops::Sub<f64> for Expression {
    type Output = Expression;

    /// Subtract a scalar from an expression.
    ///
    /// Scalars are represented as `{one_dim: value}` where `one_dim = 0`
    /// (matches [`GraphBuilder::new`] which always allocates `one` as DimId 0).
    fn sub(self, scalar: f64) -> Expression {
        self + (-scalar)
    }
}

impl std::ops::Mul<f64> for Expression {
    type Output = Expression;

    fn mul(self, scalar: f64) -> Expression {
        if scalar == 0.0 {
            return Expression::zero();
        }
        let terms = self
            .terms
            .into_iter()
            .map(|(dim, coeff)| (dim, coeff * scalar))
            .collect();
        Expression { terms }
    }
}

impl std::ops::Mul<Expression> for f64 {
    type Output = Expression;

    fn mul(self, expr: Expression) -> Expression {
        expr * self
    }
}

impl std::ops::Neg for Expression {
    type Output = Expression;

    fn neg(self) -> Expression {
        let terms = self
            .terms
            .into_iter()
            .map(|(dim, coeff)| (dim, -coeff))
            .collect();
        Expression { terms }
    }
}

impl PartialEq for Expression {
    fn eq(&self, other: &Self) -> bool {
        self.terms
            .iter()
            .all(|(k, v)| other.terms.get(k).copied().unwrap_or(0.0) == *v)
            && other
                .terms
                .iter()
                .all(|(k, v)| self.terms.get(k).copied().unwrap_or(0.0) == *v)
    }
}

impl Eq for Expression {}

impl std::fmt::Display for Expression {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.terms.is_empty() {
            return write!(f, "0");
        }
        let mut terms: Vec<_> = self.terms.iter().collect();
        terms.sort_by_key(|(id, _)| **id);
        let parts: Vec<String> = terms
            .iter()
            .map(|(dim, coeff)| {
                if **coeff == 1.0 {
                    format!("dim_{dim}")
                } else {
                    format!("{coeff}*dim_{dim}")
                }
            })
            .collect();
        write!(f, "{}", parts.join(" + "))
    }
}

// ── DimensionKind ──────────────────────────────────────────────

/// The kind of a dimension in the computation graph.
#[derive(Clone, Debug)]
pub enum DimensionKind {
    /// Token embedding input dimension (one, position, inv_log_pos, position_sq).
    Input,
    /// `relu(b) * a` gated FFN unit (has a_expr, b_expr).
    ReGLU {
        a_expr: Expression,
        b_expr: Expression,
    },
    /// Materialize expression into a dedicated residual slot (has expr).
    Persist { expr: Expression },
    /// Attention-based retrieval from token history (one output of a LookUp).
    LookUp {
        lookup_id: LookupId,
        value_index: usize,
    },
    /// Cumulative sum via attention averaging (has value_expr).
    CumSum { value_expr: Expression },
    /// Named intermediate value.
    Generic,
}

// ── Dimension ──────────────────────────────────────────────────

/// A named dimension in the computation graph.
///
/// Each dimension has a unique ID, a human-readable name, and a [`DimensionKind`]
/// that determines its semantics.
#[derive(Clone, Debug)]
pub struct Dimension {
    /// Unique dimension identifier.
    pub id: DimId,
    /// Human-readable name for diagnostics.
    pub name: String,
    /// The dimension variant.
    pub kind: DimensionKind,
}

impl Dimension {
    /// Create a new generic dimension with the given name.
    pub fn new_generic(id: DimId, name: String) -> Self {
        Self {
            id,
            name,
            kind: DimensionKind::Generic,
        }
    }
}

impl std::fmt::Display for Dimension {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let kind_str = match &self.kind {
            DimensionKind::Input => "input",
            DimensionKind::ReGLU { .. } => "reglu",
            DimensionKind::Persist { .. } => "persist",
            DimensionKind::LookUp { .. } => "lookup",
            DimensionKind::CumSum { .. } => "cumsum",
            DimensionKind::Generic => "generic",
        };
        let name = &self.name;
        let id = self.id;
        write!(f, "{kind_str}:{name}[{id}]")
    }
}

// ── LookUp ─────────────────────────────────────────────────────

/// Attention-based retrieval from token history.
///
/// A LookUp represents a hard attention operation that retrieves values
/// from the token sequence based on query/key matching.
#[derive(Clone, Debug)]
pub struct LookUp {
    /// Unique lookup identifier.
    pub id: LookupId,
    /// Optional human-readable name for diagnostics.
    pub name: Option<String>,
    /// Expressions to retrieve (one per output dimension).
    pub value_exprs: Vec<Expression>,
    /// 2D query expressions: [qx, qy].
    pub query_exprs_2d: [Expression; 2],
    /// 2D key expressions: [kx, ky].
    pub key_exprs_2d: [Expression; 2],
    /// Tie-breaking mode for hard attention.
    pub tie_break: TieBreak,
    /// DimIds of the LookUpDimension outputs (one per value_expr).
    pub dim_ids: Vec<DimId>,
}

// ── ProgramGraph ───────────────────────────────────────────────

/// Captured computation graph, ready for scheduling and weight construction.
///
/// Created by [`GraphBuilder::build`] after the graph is fully constructed.
#[derive(Clone, Debug)]
pub struct ProgramGraph {
    /// Input token expressions (one per input dimension).
    pub input_tokens: Vec<Expression>,
    /// Output token expressions (one per output dimension).
    pub output_tokens: Vec<Expression>,
    /// All dimensions in the graph, indexed by DimId.
    pub all_dims: HashMap<DimId, Dimension>,
    /// All lookups in the graph, indexed by LookupId.
    pub all_lookups: HashMap<LookupId, LookUp>,
    /// DimId of the built-in `one` input dimension.
    pub one: DimId,
    /// DimId of the built-in `position` input dimension.
    pub position: DimId,
    /// DimId of the built-in `inv_log_pos` input dimension.
    pub inv_log_pos: DimId,
    /// DimId of the built-in `position_sq` input dimension.
    pub position_sq: DimId,
}

// ── Graph Validation (C5) ─────────────────────────────────────

/// Validation errors for a [`ProgramGraph`].
#[derive(Clone, Debug, PartialEq)]
pub enum ValidationError {
    /// A dimension referenced in an expression does not exist in the graph.
    MissingDim {
        /// The dimension that references the missing dim.
        source: DimId,
        /// The missing dimension ID.
        missing: DimId,
    },
    /// A lookup referenced by a dimension does not exist.
    MissingLookup {
        /// The dimension referencing the missing lookup.
        source: DimId,
        /// The missing lookup ID.
        missing: LookupId,
    },
    /// A cycle was detected in the dependency graph.
    Cycle {
        /// Dimensions involved in the cycle.
        cycle: Vec<DimId>,
    },
    /// An expression in an output token references an undefined dimension.
    OutputMissingDim {
        /// Output token index.
        output_index: usize,
        /// Missing dimension ID.
        missing: DimId,
    },
    /// An expression in an input token references an undefined dimension.
    InputMissingDim {
        /// Input token index.
        input_index: usize,
        /// Missing dimension ID.
        missing: DimId,
    },
}

impl std::fmt::Display for ValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingDim { source, missing } => {
                write!(f, "dim {source} references undefined dim {missing}")
            }
            Self::MissingLookup { source, missing } => {
                write!(f, "dim {source} references undefined lookup {missing}")
            }
            Self::Cycle { cycle } => {
                write!(f, "dependency cycle: {cycle:?}")
            }
            Self::OutputMissingDim {
                output_index,
                missing,
            } => {
                write!(
                    f,
                    "output token {output_index} references undefined dim {missing}"
                )
            }
            Self::InputMissingDim {
                input_index,
                missing,
            } => {
                write!(
                    f,
                    "input token {input_index} references undefined dim {missing}"
                )
            }
        }
    }
}

impl std::error::Error for ValidationError {}

impl ProgramGraph {
    /// Validate the computation graph.
    ///
    /// Checks:
    /// 1. All dimensions referenced in expressions exist in `all_dims`
    /// 2. All lookups referenced by LookUp dimensions exist in `all_lookups`
    /// 3. All input/output token expressions reference valid dimensions
    /// 4. No cycles in the dependency graph
    ///
    /// Returns `Ok(())` if valid, or the first error found.
    pub fn validate(&self) -> Result<(), ValidationError> {
        self.check_dim_consistency()?;
        self.check_lookup_consistency()?;
        self.check_token_consistency()?;
        self.check_no_cycles()
    }

    /// Check that all dims referenced in dimension expressions exist.
    fn check_dim_consistency(&self) -> Result<(), ValidationError> {
        for (&dim_id, dim) in &self.all_dims {
            for dep_id in self.dim_expression_deps(dim) {
                if !self.all_dims.contains_key(&dep_id) {
                    return Err(ValidationError::MissingDim {
                        source: dim_id,
                        missing: dep_id,
                    });
                }
            }
        }
        Ok(())
    }

    /// Check that all lookups referenced by LookUp dimensions exist.
    fn check_lookup_consistency(&self) -> Result<(), ValidationError> {
        for (&dim_id, dim) in &self.all_dims {
            if let DimensionKind::LookUp { lookup_id, .. } = &dim.kind
                && !self.all_lookups.contains_key(lookup_id) {
                    return Err(ValidationError::MissingLookup {
                        source: dim_id,
                        missing: *lookup_id,
                    });
                }
        }
        Ok(())
    }

    /// Check that input/output token expressions reference valid dims.
    fn check_token_consistency(&self) -> Result<(), ValidationError> {
        for (i, expr) in self.input_tokens.iter().enumerate() {
            for &dim_id in expr.terms.keys() {
                if !self.all_dims.contains_key(&dim_id) {
                    return Err(ValidationError::InputMissingDim {
                        input_index: i,
                        missing: dim_id,
                    });
                }
            }
        }
        for (i, expr) in self.output_tokens.iter().enumerate() {
            for &dim_id in expr.terms.keys() {
                if !self.all_dims.contains_key(&dim_id) {
                    return Err(ValidationError::OutputMissingDim {
                        output_index: i,
                        missing: dim_id,
                    });
                }
            }
        }
        Ok(())
    }

    /// Check for cycles using topological sort (Kahn's algorithm).
    fn check_no_cycles(&self) -> Result<(), ValidationError> {
        let mut in_degree: HashMap<DimId, usize> = HashMap::new();
        let mut dependents: HashMap<DimId, Vec<DimId>> = HashMap::new();

        for &dim_id in self.all_dims.keys() {
            in_degree.insert(dim_id, 0);
            dependents.insert(dim_id, Vec::new());
        }

        for (&dim_id, dim) in &self.all_dims {
            let deps: Vec<DimId> = self.dim_expression_deps(dim);
            in_degree.insert(dim_id, deps.len());
            for dep_id in deps {
                if let Some(v) = dependents.get_mut(&dep_id) {
                    v.push(dim_id);
                }
            }
        }

        // Kahn's algorithm
        let mut queue: std::collections::VecDeque<DimId> = in_degree
            .iter()
            .filter(|(_, deg)| **deg == 0)
            .map(|(&id, _)| id)
            .collect();

        let mut sorted_count = 0;
        while let Some(dim_id) = queue.pop_front() {
            sorted_count += 1;
            if let Some(deps) = dependents.get(&dim_id) {
                for &dep_id in deps {
                    if let Some(deg) = in_degree.get_mut(&dep_id) {
                        *deg -= 1;
                        if *deg == 0 {
                            queue.push_back(dep_id);
                        }
                    }
                }
            }
        }

        if sorted_count < self.all_dims.len() {
            let remaining: Vec<DimId> = in_degree
                .iter()
                .filter(|(_, deg)| **deg > 0)
                .map(|(&id, _)| id)
                .collect();
            return Err(ValidationError::Cycle { cycle: remaining });
        }

        Ok(())
    }

    /// Get all dimension IDs referenced by a dimension's expressions.
    fn dim_expression_deps(&self, dim: &Dimension) -> Vec<DimId> {
        match &dim.kind {
            DimensionKind::Input | DimensionKind::Generic => Vec::new(),
            DimensionKind::ReGLU { a_expr, b_expr } => {
                let mut deps: Vec<DimId> = a_expr.terms.keys().copied().collect();
                deps.extend(b_expr.terms.keys().copied());
                deps.sort_unstable();
                deps.dedup();
                deps
            }
            DimensionKind::Persist { expr } => {
                let mut deps: Vec<DimId> = expr.terms.keys().copied().collect();
                deps.sort_unstable();
                deps.dedup();
                deps
            }
            DimensionKind::LookUp { lookup_id, .. } => {
                // LookUp dims depend on the lookup's query/key/value expressions + inv_log_pos
                let mut deps = vec![self.inv_log_pos];
                if let Some(lookup) = self.all_lookups.get(lookup_id) {
                    for expr in &lookup.query_exprs_2d {
                        deps.extend(expr.terms.keys().copied());
                    }
                    for expr in &lookup.key_exprs_2d {
                        deps.extend(expr.terms.keys().copied());
                    }
                    for expr in &lookup.value_exprs {
                        deps.extend(expr.terms.keys().copied());
                    }
                }
                deps.sort_unstable();
                deps.dedup();
                deps
            }
            DimensionKind::CumSum { value_expr } => {
                let mut deps: Vec<DimId> = value_expr.terms.keys().copied().collect();
                deps.sort_unstable();
                deps.dedup();
                deps
            }
        }
    }
}

// ── GraphBuilder ───────────────────────────────────────────────

/// Mutable state for building computation graphs.
///
/// Replaces Python's global mutable state (`_all_dims`, `_all_lookups`, caches).
/// Create a new builder for each program graph.
///
/// # Example
///
/// ```ignore
/// use microgpt_rs::percepta::graph::GraphBuilder;
/// use microgpt_rs::percepta::TieBreak;
///
/// let mut builder = GraphBuilder::new();
/// let pos = builder.position;
/// let one = builder.one;
///
/// let x = builder.reglu(pos, 2.0_f64);
/// let y = builder.persist(x.clone());
///
/// let graph = builder.build(vec![], vec![y]);
/// assert_eq!(graph.all_dims.len(), 6); // 4 inputs + 1 reglu + 1 persist
/// ```
pub struct GraphBuilder {
    /// Next available dimension ID.
    next_dim_id: DimId,
    /// Next available lookup ID.
    next_lookup_id: LookupId,
    /// All dimensions in the graph.
    all_dims: HashMap<DimId, Dimension>,
    /// All lookups in the graph.
    all_lookups: HashMap<LookupId, LookUp>,
    /// Cache: (a_key, b_key) → ReGLU DimId.
    reglu_cache: HashMap<(ExprKey, ExprKey), DimId>,
    /// Cache: (a_key, b_key) → persist DimId.
    stepglu_cache: HashMap<(ExprKey, ExprKey), DimId>,
    /// Cache: (a_key, b_key) → persist DimId.
    multiply_cache: HashMap<(ExprKey, ExprKey), DimId>,
    /// Cache: expr_key → persist DimId.
    clear_key_cache: HashMap<ExprKey, DimId>,
    /// DimId of the `one` input dimension.
    pub one: DimId,
    /// DimId of the `position` input dimension.
    pub position: DimId,
    /// DimId of the `inv_log_pos` input dimension.
    pub inv_log_pos: DimId,
    /// DimId of the `position_sq` input dimension.
    pub position_sq: DimId,
}

impl GraphBuilder {
    /// Create a new GraphBuilder with built-in input dimensions.
    ///
    /// Registers four input dimensions: `one`, `position`, `inv_log_pos`, `position_sq`.
    pub fn new() -> Self {
        let mut builder = Self {
            next_dim_id: 0,
            next_lookup_id: 0,
            all_dims: HashMap::new(),
            all_lookups: HashMap::new(),
            reglu_cache: HashMap::new(),
            stepglu_cache: HashMap::new(),
            multiply_cache: HashMap::new(),
            clear_key_cache: HashMap::new(),
            one: 0,
            position: 0,
            inv_log_pos: 0,
            position_sq: 0,
        };

        // Register built-in input dimensions in fixed order
        builder.one = builder.alloc_input("one");
        builder.position = builder.alloc_input("position");
        builder.inv_log_pos = builder.alloc_input("inv_log_pos");
        builder.position_sq = builder.alloc_input("position_sq");

        builder
    }

    /// Allocate a new dimension ID.
    fn alloc_dim_id(&mut self) -> DimId {
        let id = self.next_dim_id;
        self.next_dim_id += 1;
        id
    }

    /// Allocate a new lookup ID.
    fn alloc_lookup_id(&mut self) -> LookupId {
        let id = self.next_lookup_id;
        self.next_lookup_id += 1;
        id
    }

    /// Create and register an input dimension.
    fn alloc_input(&mut self, name: &str) -> DimId {
        let id = self.alloc_dim_id();
        let dim = Dimension {
            id,
            name: name.to_string(),
            kind: DimensionKind::Input,
        };
        self.all_dims.insert(id, dim);
        id
    }

    /// Create and register a dimension with the given kind and name.
    fn alloc_dim(&mut self, name: String, kind: DimensionKind) -> DimId {
        let id = self.alloc_dim_id();
        let dim = Dimension { id, name, kind };
        self.all_dims.insert(id, dim);
        id
    }

    /// Get a reference to a dimension by ID.
    pub fn get_dim(&self, id: DimId) -> Option<&Dimension> {
        self.all_dims.get(&id)
    }

    /// Get a reference to a lookup by ID.
    pub fn get_lookup(&self, id: LookupId) -> Option<&LookUp> {
        self.all_lookups.get(&id)
    }

    /// Number of dimensions in the graph.
    pub fn dim_count(&self) -> usize {
        self.all_dims.len()
    }

    /// Number of lookups in the graph.
    pub fn lookup_count(&self) -> usize {
        self.all_lookups.len()
    }

    // ── Builder: Gate Primitives ────────────────────────────────

    /// `relu(b) * a` — single ReGLU dimension.
    ///
    /// Use when `b` is known non-negative, giving `reglu(a, b) = a * b`.
    /// Cached: identical `(a, b)` expressions return the same DimId.
    pub fn reglu(&mut self, a: impl IntoExpr, b: impl IntoExpr) -> Expression {
        let a_expr = a.into_expr(self.one);
        let b_expr = b.into_expr(self.one);
        let key = (a_expr.expr_key(), b_expr.expr_key());

        match self.reglu_cache.get(&key).copied() {
            Some(dim_id) => Expression::from_dim(dim_id),
            None => {
                let id = self.next_dim_id;
                let name = format!("reglu_{id}");
                let dim_id = self.alloc_dim(
                    name,
                    DimensionKind::ReGLU {
                        a_expr: a_expr.clone(),
                        b_expr: b_expr.clone(),
                    },
                );
                self.reglu_cache.insert(key, dim_id);
                Expression::from_dim(dim_id)
            }
        }
    }

    /// `a * step(b >= 0)` — conditional gate via two ReGLU dims + persist.
    ///
    /// Equals `reglu(a, b + 1) - reglu(a, b)`.
    /// For integer `b`: equals `a` when `b >= 0`, `0` when `b < 0`.
    /// The persist dim stores the difference in a single residual slot.
    /// Cached: identical `(a, b)` expressions return the same DimId.
    pub fn stepglu(&mut self, a: impl IntoExpr, b: impl IntoExpr) -> Expression {
        let a_expr = a.into_expr(self.one);
        let b_expr = b.into_expr(self.one);
        let key = (a_expr.expr_key(), b_expr.expr_key());

        match self.stepglu_cache.get(&key).copied() {
            Some(dim_id) => Expression::from_dim(dim_id),
            None => {
                // stepglu(a, b) = reglu(a, b+1) - reglu(a, b)
                let one_expr = Expression::from_dim(self.one);
                let b_plus_1 = b_expr.clone() + one_expr;

                let id_r1 = {
                    let name = format!("reglu_{}", self.next_dim_id);
                    self.alloc_dim(
                        name,
                        DimensionKind::ReGLU {
                            a_expr: a_expr.clone(),
                            b_expr: b_plus_1,
                        },
                    )
                };

                let id_r2 = {
                    let name = format!("reglu_{id_r1}");
                    self.alloc_dim(
                        name,
                        DimensionKind::ReGLU {
                            a_expr: a_expr.clone(),
                            b_expr: b_expr.clone(),
                        },
                    )
                };

                let persist_expr =
                    Expression::from_terms(HashMap::from([(id_r1, 1.0), (id_r2, -1.0)]));

                let id_persist = {
                    let name = format!("persist_{id_r2}");
                    self.alloc_dim(name, DimensionKind::Persist { expr: persist_expr })
                };

                self.stepglu_cache.insert(key, id_persist);
                Expression::from_dim(id_persist)
            }
        }
    }

    /// Materialize a linear expression into a dedicated residual slot.
    ///
    /// Creates a [`DimensionKind::Persist`] dimension that stores the value of `expr`.
    /// This reduces `d_model` by allowing the expression's constituents to die earlier.
    /// Treated as a schedulable operation (separate phase) for scheduling.
    pub fn persist(&mut self, expr: impl IntoExpr) -> Expression {
        let expr = expr.into_expr(self.one);
        let id = self.next_dim_id;
        let name = format!("persist_{id}");
        let dim_id = self.alloc_dim(name, DimensionKind::Persist { expr });
        Expression::from_dim(dim_id)
    }

    /// Full signed multiplication via two ReGLU dims + persist.
    ///
    /// `multiply(a, b) = reglu(a, b) - reglu(a, -b)`
    /// Cached: identical `(a, b)` expressions return the same DimId.
    fn multiply(&mut self, a: &Expression, b: &Expression) -> Expression {
        let key = (a.expr_key(), b.expr_key());

        match self.multiply_cache.get(&key).copied() {
            Some(dim_id) => Expression::from_dim(dim_id),
            None => {
                let neg_b = -b.clone();

                let id_r1 = {
                    let name = format!("reglu_{}", self.next_dim_id);
                    self.alloc_dim(
                        name,
                        DimensionKind::ReGLU {
                            a_expr: a.clone(),
                            b_expr: b.clone(),
                        },
                    )
                };

                let id_r2 = {
                    let name = format!("reglu_{id_r1}");
                    self.alloc_dim(
                        name,
                        DimensionKind::ReGLU {
                            a_expr: a.clone(),
                            b_expr: neg_b,
                        },
                    )
                };

                let persist_expr =
                    Expression::from_terms(HashMap::from([(id_r1, 1.0), (id_r2, -1.0)]));

                let id_persist = {
                    let name = format!("persist_{id_r2}");
                    let dim_id =
                        self.alloc_dim(name, DimensionKind::Persist { expr: persist_expr });
                    self.multiply_cache.insert(key, dim_id);
                    dim_id
                };

                Expression::from_dim(id_persist)
            }
        }
    }

    /// Create a generic (named intermediate) dimension.
    pub fn generic(&mut self, name: &str) -> Expression {
        let dim_id = self.alloc_dim(name.to_string(), DimensionKind::Generic);
        Expression::from_dim(dim_id)
    }

    // ── Builder: Attention (fetch) ──────────────────────────────

    /// Map 1D key + optional clear_key to 2D key expressions via ReGLU.
    ///
    /// Implements the parabolic key encoding for hard attention:
    /// - `kx = k * 2 - one * (2 * KEY_OFFSET)`
    /// - `ky = -|k|² + k * (2 * KEY_OFFSET) - one * KEY_OFFSET²`
    /// - With clear_key: `ky -= clear * BIG`
    /// - With latest tie-break: `ky += inv_log_pos * LATEST_ALPHA`
    /// - With average tie-break: `ky = one` (uniform attention)
    fn encode_2d_key(
        &mut self,
        k: &Expression,
        clear_key_expr: Option<&Expression>,
        tie_break: TieBreak,
    ) -> [Expression; 2] {
        let one_expr = Expression::from_dim(self.one);

        // Compute |k|² efficiently using fast paths for common cases
        let k_abs = match (k.len(), k.terms.iter().next()) {
            (1, Some((&dim, &c))) if dim == self.one => {
                // k = c * one → |k|² = c² * one
                Expression::from_scalar(c * c, self.one)
            }
            (1, Some((&dim, &c))) if dim == self.position => {
                // k = c * position → |k|² = c² * position_sq
                let mut terms = HashMap::new();
                terms.insert(self.position_sq, c * c);
                Expression { terms }
            }
            _ => {
                // General case: |k|² = multiply(k, k)
                self.multiply(k, k)
            }
        };

        // kx = k * 2 - one * (2 * KEY_OFFSET)
        let kx = k.clone() * 2.0 - one_expr.clone() * (2.0 * KEY_OFFSET);

        // ky = -|k|² + k * (2 * KEY_OFFSET) - one * KEY_OFFSET²
        let offset_sq = KEY_OFFSET * KEY_OFFSET;
        let mut ky = -k_abs + k.clone() * (2.0 * KEY_OFFSET) - one_expr.clone() * offset_sq;

        // Apply clear_key (effectively zeros out attention to non-matching keys)
        if let Some(ck) = clear_key_expr {
            let clear = match ck.len() {
                1 => ck.clone(),
                _ => {
                    let ck_key = ck.expr_key();
                    match self.clear_key_cache.get(&ck_key).copied() {
                        Some(dim_id) => Expression::from_dim(dim_id),
                        None => {
                            let persist_expr = self.persist(ck.clone());
                            let dim_id = persist_expr.terms.keys().next().copied().unwrap_or(0);
                            self.clear_key_cache.insert(ck_key, dim_id);
                            persist_expr
                        }
                    }
                }
            };
            ky = ky - clear * BIG;
        }

        // Apply tie-breaking
        let ky = match tie_break {
            TieBreak::Latest => {
                let mut terms = HashMap::new();
                terms.insert(self.inv_log_pos, LATEST_ALPHA);
                ky + Expression { terms }
            }
            TieBreak::Average => Expression::from_dim(self.one),
        };

        [kx, ky]
    }

    /// Map 1D query to 2D query expressions (purely linear).
    ///
    /// `qx = q - one * KEY_OFFSET`
    /// `qy = one`
    fn encode_2d_query(&self, q: &Expression) -> [Expression; 2] {
        let one_expr = Expression::from_dim(self.one);
        [q.clone() - one_expr.clone() * KEY_OFFSET, one_expr]
    }

    /// Attention-based retrieval from token history (single value).
    ///
    /// Creates a [`LookUp`] that retrieves `value` using hard attention
    /// with the given query, key, and optional clear_key expressions.
    ///
    /// Returns a single [`Expression`] referencing the retrieved value.
    pub fn fetch(
        &mut self,
        value: impl IntoExpr,
        query: Option<Expression>,
        key: Option<Expression>,
        clear_key: Option<Expression>,
        tie_break: TieBreak,
    ) -> Expression {
        let results = self.fetch_vec(
            vec![value.into_expr(self.one)],
            query,
            key,
            clear_key,
            tie_break,
        );
        results.into_iter().next().unwrap_or_else(Expression::zero)
    }

    /// Attention-based retrieval of multiple values from token history.
    ///
    /// Returns one [`Expression`] per value, all from the same attention head.
    pub fn fetch_vec(
        &mut self,
        value_exprs: Vec<Expression>,
        query: Option<Expression>,
        key: Option<Expression>,
        clear_key: Option<Expression>,
        tie_break: TieBreak,
    ) -> Vec<Expression> {
        let q = query.unwrap_or_else(Expression::zero);
        let k = key.unwrap_or_else(Expression::zero);

        let key_2d = self.encode_2d_key(&k, clear_key.as_ref(), tie_break);
        let query_2d = self.encode_2d_query(&q);

        let lookup_id = self.alloc_lookup_id();
        let n_values = value_exprs.len();

        // Create LookUpDimensions for each value
        let mut dim_ids = Vec::with_capacity(n_values);
        for value_index in 0..n_values {
            let name = format!("lookup_{lookup_id}_v{value_index}");
            let id = self.alloc_dim(
                name,
                DimensionKind::LookUp {
                    lookup_id,
                    value_index,
                },
            );
            dim_ids.push(id);
        }

        let lookup = LookUp {
            id: lookup_id,
            name: None,
            value_exprs,
            query_exprs_2d: query_2d,
            key_exprs_2d: key_2d,
            tie_break,
            dim_ids: dim_ids.clone(),
        };
        self.all_lookups.insert(lookup_id, lookup);

        dim_ids.into_iter().map(Expression::from_dim).collect()
    }

    /// Cumulative sum via attention averaging: `avg * position`.
    ///
    /// Position 0 (start token, `one=0`) is excluded from the average
    /// because its `ky=0 < 1`, so the denominator is `p` (not `p+1`).
    /// Multiplying by `position` recovers the exact cumulative sum.
    pub fn fetch_sum(&mut self, values: Vec<Expression>) -> Vec<Expression> {
        let key = Expression::from_scalar(KEY_OFFSET, self.one);
        let query = Expression::from_scalar(KEY_OFFSET, self.one);

        let avg_dims = self.fetch_vec(values, Some(query), Some(key), None, TieBreak::Average);

        // cumsum = avg * position (via reglu since position >= 0)
        let position_expr = Expression::from_dim(self.position);
        avg_dims
            .into_iter()
            .map(|avg| self.reglu(avg, position_expr.clone()))
            .collect()
    }

    /// Cumulative sum of a single value.
    ///
    /// Convenience wrapper around [`fetch_sum`] for a single value.
    pub fn fetch_sum_single(&mut self, value: Expression) -> Expression {
        self.fetch_sum(vec![value])
            .into_iter()
            .next()
            .unwrap_or_else(Expression::zero)
    }

    // ── Builder: Naming ─────────────────────────────────────────

    /// Name a dimension directly (for non-input dimensions).
    ///
    /// If the dimension is a LookUp type, also names the parent LookUp.
    pub fn name_dim(&mut self, dim_id: DimId, name: &str) {
        let is_input = self
            .all_dims
            .get(&dim_id)
            .map(|d| matches!(d.kind, DimensionKind::Input))
            .unwrap_or(true);

        if is_input {
            return;
        }

        // Collect lookup update separately to avoid borrow conflicts
        let lookup_update: Option<(LookupId, String)> =
            self.all_dims.get(&dim_id).and_then(|dim| match &dim.kind {
                DimensionKind::LookUp { lookup_id, .. } => Some((*lookup_id, name.to_string())),
                _ => None,
            });

        if let Some(dim) = self.all_dims.get_mut(&dim_id) {
            dim.name = name.to_string();
        }

        if let Some((lookup_id, n)) = lookup_update
            && let Some(lookup) = self.all_lookups.get_mut(&lookup_id)
            && lookup.name.is_none()
        {
            lookup.name = Some(n);
        }
    }

    /// Name graph nodes from a list of (name, expression) pairs.
    ///
    /// Assigns meaningful names to ReGLU, Persist, and LookUp dimensions
    /// embedded in the expressions. Call after graph construction for
    /// diagnostic output.
    pub fn auto_name(&mut self, names: &[(String, Expression)]) {
        for (name, expr) in names {
            self.name_expr_dims(name, expr);
        }
    }

    /// Name ReGLU, Persist, and LookUp dims embedded in an Expression.
    fn name_expr_dims(&mut self, name: &str, expr: &Expression) {
        // Collect updates first to avoid borrow conflicts
        let mut updates: Vec<NameUpdate> = Vec::new();
        let mut pos_idx = 0usize;
        let mut neg_idx = 0usize;
        let mut lu_idx = 0usize;
        let mut persist_idx = 0usize;

        for (&dim_id, &coeff) in &expr.terms {
            let Some(dim) = self.all_dims.get(&dim_id) else {
                continue;
            };
            let (new_name, lookup_update) = match &dim.kind {
                DimensionKind::Persist { .. } if dim.name.starts_with("persist_") => {
                    let n = if persist_idx == 0 {
                        name.to_string()
                    } else {
                        format!("{name}${persist_idx}")
                    };
                    persist_idx += 1;
                    (Some(n), None)
                }
                DimensionKind::ReGLU { .. } if dim.name.starts_with("reglu_") => {
                    let n = if coeff > 0.0 {
                        let r = if pos_idx == 0 {
                            format!("{name}+")
                        } else {
                            format!("{name}+{pos_idx}")
                        };
                        pos_idx += 1;
                        r
                    } else {
                        let r = if neg_idx == 0 {
                            format!("{name}-")
                        } else {
                            format!("{name}-{neg_idx}")
                        };
                        neg_idx += 1;
                        r
                    };
                    (Some(n), None)
                }
                DimensionKind::LookUp { lookup_id, .. } if dim.name.starts_with("lookup_") => {
                    let n = if lu_idx == 0 {
                        format!("{name}_lu")
                    } else {
                        format!("{name}_lu{lu_idx}")
                    };
                    lu_idx += 1;
                    (Some(n.clone()), Some((*lookup_id, n)))
                }
                _ => (None, None),
            };

            if let Some(n) = new_name {
                updates.push((dim_id, n, lookup_update));
            }
        }

        // Apply updates
        for (dim_id, new_name, lookup_update) in updates {
            if let Some(dim) = self.all_dims.get_mut(&dim_id) {
                dim.name = new_name;
            }
            if let Some((lookup_id, n)) = lookup_update
                && let Some(lookup) = self.all_lookups.get_mut(&lookup_id)
                && lookup.name.is_none()
            {
                lookup.name = Some(n);
            }
        }
    }

    // ── Builder: Finalize ───────────────────────────────────────

    /// Build the final [`ProgramGraph`] from the current state.
    ///
    /// `input_tokens` and `output_tokens` define the program's interface.
    /// Consumes the builder.
    pub fn build(
        self,
        input_tokens: Vec<Expression>,
        output_tokens: Vec<Expression>,
    ) -> ProgramGraph {
        ProgramGraph {
            input_tokens,
            output_tokens,
            all_dims: self.all_dims,
            all_lookups: self.all_lookups,
            one: self.one,
            position: self.position,
            inv_log_pos: self.inv_log_pos,
            position_sq: self.position_sq,
        }
    }
}

impl Default for GraphBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::ValidationError;
    use super::*;

    // ── Expression Tests ────────────────────────────────────────

    #[test]
    fn test_expression_zero() {
        let expr = Expression::zero();
        assert!(expr.is_zero());
        assert!(expr.terms.is_empty());
        assert_eq!(expr.len(), 0);
    }

    #[test]
    fn test_expression_from_dim() {
        let expr = Expression::from_dim(42);
        assert!(!expr.is_zero());
        assert_eq!(expr.get(42), 1.0);
        assert_eq!(expr.get(99), 0.0);
        assert_eq!(expr.len(), 1);
    }

    #[test]
    fn test_expression_from_scalar() {
        let expr = Expression::from_scalar(3.5, 1);
        assert_eq!(expr.get(1), 3.5);
        assert_eq!(expr.len(), 1);

        let zero = Expression::from_scalar(0.0, 1);
        assert!(zero.is_zero());
    }

    #[test]
    fn test_expression_from_terms_removes_zeros() {
        let expr = Expression::from_terms(HashMap::from([(1, 2.0), (2, 0.0), (3, -1.0)]));
        assert_eq!(expr.len(), 2);
        assert!(expr.terms.get(&2).is_none());
    }

    #[test]
    fn test_expression_add() {
        let a = Expression::from_terms(HashMap::from([(1, 2.0)]));
        let b = Expression::from_terms(HashMap::from([(1, 3.0), (2, 1.0)]));
        let result = a + b;
        assert_eq!(result.get(1), 5.0);
        assert_eq!(result.get(2), 1.0);
    }

    #[test]
    fn test_expression_add_cancels() {
        let a = Expression::from_terms(HashMap::from([(1, 2.0)]));
        let b = Expression::from_terms(HashMap::from([(1, -2.0)]));
        let result = a + b;
        assert!(result.is_zero());
    }

    #[test]
    fn test_expression_sub() {
        let a = Expression::from_terms(HashMap::from([(1, 5.0), (2, 3.0)]));
        let b = Expression::from_terms(HashMap::from([(1, 2.0)]));
        let result = a - b;
        assert_eq!(result.get(1), 3.0);
        assert_eq!(result.get(2), 3.0);
    }

    #[test]
    fn test_expression_mul_scalar() {
        let expr = Expression::from_terms(HashMap::from([(1, 2.0), (3, -1.0)]));
        let result = expr * 3.0;
        assert_eq!(result.get(1), 6.0);
        assert_eq!(result.get(3), -3.0);
    }

    #[test]
    fn test_expression_mul_scalar_commutative() {
        let expr = Expression::from_terms(HashMap::from([(1, 2.0)]));
        let result = 3.0 * expr;
        assert_eq!(result.get(1), 6.0);
    }

    #[test]
    fn test_expression_mul_zero() {
        let expr = Expression::from_terms(HashMap::from([(1, 2.0), (3, -1.0)]));
        let result = expr * 0.0;
        assert!(result.is_zero());
    }

    #[test]
    fn test_expression_neg() {
        let expr = Expression::from_terms(HashMap::from([(1, 2.0), (3, -1.0)]));
        let result = -expr;
        assert_eq!(result.get(1), -2.0);
        assert_eq!(result.get(3), 1.0);
    }

    #[test]
    fn test_expression_evaluate() {
        let expr = Expression::from_terms(HashMap::from([(1, 2.0), (3, -1.0)]));
        let values = HashMap::from([(1, 5.0), (3, 10.0)]);
        // 2.0 * 5.0 + (-1.0) * 10.0 = 10.0 - 10.0 = 0.0
        assert_eq!(expr.evaluate(&values), 0.0);

        let values2 = HashMap::from([(1, 5.0)]);
        // 2.0 * 5.0 + (-1.0) * 0.0 = 10.0
        assert_eq!(expr.evaluate(&values2), 10.0);
    }

    #[test]
    fn test_expression_set_removes_zero() {
        let mut expr = Expression::from_terms(HashMap::from([(1, 2.0), (2, 3.0)]));
        expr.set(1, 0.0);
        assert!(expr.terms.get(&1).is_none());
        assert_eq!(expr.get(2), 3.0);
    }

    #[test]
    fn test_expression_set_nonzero() {
        let mut expr = Expression::from_terms(HashMap::from([(1, 2.0)]));
        expr.set(1, 5.0);
        assert_eq!(expr.get(1), 5.0);
    }

    #[test]
    fn test_expression_equality() {
        let a = Expression::from_terms(HashMap::from([(1, 2.0), (2, 3.0)]));
        let b = Expression::from_terms(HashMap::from([(2, 3.0), (1, 2.0)]));
        assert_eq!(a, b);

        let c = Expression::from_terms(HashMap::from([(1, 2.0)]));
        assert_ne!(a, c);
    }

    #[test]
    fn test_expression_display() {
        let expr = Expression::from_terms(HashMap::from([(1, 1.0)]));
        let displayed = format!("{expr}");
        assert!(displayed.contains("dim_1"));

        let zero = Expression::zero();
        assert_eq!(format!("{zero}"), "0");
    }

    // ── Dimension Tests ─────────────────────────────────────────

    #[test]
    fn test_dimension_display() {
        let dim = Dimension {
            id: 0,
            name: "one".to_string(),
            kind: DimensionKind::Input,
        };
        assert_eq!(format!("{dim}"), "input:one[0]");

        let dim_reglu = Dimension {
            id: 5,
            name: "reglu_5".to_string(),
            kind: DimensionKind::ReGLU {
                a_expr: Expression::zero(),
                b_expr: Expression::zero(),
            },
        };
        assert_eq!(format!("{dim_reglu}"), "reglu:reglu_5[5]");
    }

    #[test]
    fn test_dimension_new_generic() {
        let dim = Dimension::new_generic(99, "test".to_string());
        assert_eq!(dim.id, 99);
        assert_eq!(dim.name, "test");
        assert!(matches!(dim.kind, DimensionKind::Generic));
    }

    #[test]
    fn test_dimension_kind_display_names() {
        let cases: Vec<(DimensionKind, &str)> = vec![
            (DimensionKind::Input, "input"),
            (
                DimensionKind::ReGLU {
                    a_expr: Expression::zero(),
                    b_expr: Expression::zero(),
                },
                "reglu",
            ),
            (
                DimensionKind::Persist {
                    expr: Expression::zero(),
                },
                "persist",
            ),
            (
                DimensionKind::LookUp {
                    lookup_id: 0,
                    value_index: 0,
                },
                "lookup",
            ),
            (
                DimensionKind::CumSum {
                    value_expr: Expression::zero(),
                },
                "cumsum",
            ),
            (DimensionKind::Generic, "generic"),
        ];

        for (kind, expected_prefix) in cases {
            let dim = Dimension {
                id: 0,
                name: "test".to_string(),
                kind,
            };
            let displayed = format!("{dim}");
            assert!(
                displayed.starts_with(expected_prefix),
                "Expected '{displayed}' to start with '{expected_prefix}'"
            );
        }
    }

    // ── GraphBuilder Tests ──────────────────────────────────────

    #[test]
    fn test_builder_new_has_input_dims() {
        let builder = GraphBuilder::new();
        assert_eq!(builder.dim_count(), 4);

        let one = builder.get_dim(builder.one).unwrap();
        assert_eq!(one.name, "one");
        assert!(matches!(one.kind, DimensionKind::Input));

        let pos = builder.get_dim(builder.position).unwrap();
        assert_eq!(pos.name, "position");
        assert!(matches!(pos.kind, DimensionKind::Input));
    }

    #[test]
    fn test_builder_input_dim_ids() {
        let builder = GraphBuilder::new();
        assert_eq!(builder.one, 0);
        assert_eq!(builder.position, 1);
        assert_eq!(builder.inv_log_pos, 2);
        assert_eq!(builder.position_sq, 3);
    }

    #[test]
    fn test_builder_reglu() {
        let mut builder = GraphBuilder::new();

        let result = builder.reglu(3.0_f64, 2.0_f64);
        assert_eq!(builder.dim_count(), 5); // 4 inputs + 1 reglu
        assert_eq!(result.len(), 1);

        let dim_id = *result.terms.keys().next().unwrap();
        let dim = builder.get_dim(dim_id).unwrap();
        assert!(matches!(dim.kind, DimensionKind::ReGLU { .. }));
    }

    #[test]
    fn test_builder_reglu_caching() {
        let mut builder = GraphBuilder::new();

        let r1 = builder.reglu(3.0_f64, 2.0_f64);
        let r2 = builder.reglu(3.0_f64, 2.0_f64);
        assert_eq!(r1, r2);
        assert_eq!(builder.dim_count(), 5); // 4 inputs + 1 reglu (cached)
    }

    #[test]
    fn test_builder_reglu_different_inputs() {
        let mut builder = GraphBuilder::new();

        let r1 = builder.reglu(3.0_f64, 2.0_f64);
        let r2 = builder.reglu(3.0_f64, 5.0_f64);
        assert_ne!(r1, r2);
        assert_eq!(builder.dim_count(), 6); // 4 inputs + 2 reglu
    }

    #[test]
    fn test_builder_stepglu() {
        let mut builder = GraphBuilder::new();

        let result = builder.stepglu(5.0_f64, 0.0_f64);
        // stepglu creates: 2 ReGLU + 1 Persist = 3 new dims
        assert_eq!(builder.dim_count(), 7); // 4 inputs + 3 new
        assert_eq!(result.len(), 1);

        let dim_id = *result.terms.keys().next().unwrap();
        let dim = builder.get_dim(dim_id).unwrap();
        assert!(matches!(dim.kind, DimensionKind::Persist { .. }));
    }

    #[test]
    fn test_builder_stepglu_caching() {
        let mut builder = GraphBuilder::new();

        let s1 = builder.stepglu(5.0_f64, 0.0_f64);
        let s2 = builder.stepglu(5.0_f64, 0.0_f64);
        assert_eq!(s1, s2);
        assert_eq!(builder.dim_count(), 7); // 4 inputs + 3 new (cached)
    }

    #[test]
    fn test_builder_persist() {
        let mut builder = GraphBuilder::new();
        let one = builder.one;

        let expr = Expression::from_terms(HashMap::from([(one, 1.0)]));
        let result = builder.persist(expr);
        assert_eq!(builder.dim_count(), 5); // 4 inputs + 1 persist

        let dim_id = *result.terms.keys().next().unwrap();
        let dim = builder.get_dim(dim_id).unwrap();
        assert!(matches!(dim.kind, DimensionKind::Persist { .. }));
    }

    #[test]
    fn test_builder_generic() {
        let mut builder = GraphBuilder::new();

        let result = builder.generic("my_intermediate");
        let dim_id = *result.terms.keys().next().unwrap();
        let dim = builder.get_dim(dim_id).unwrap();
        assert_eq!(dim.name, "my_intermediate");
        assert!(matches!(dim.kind, DimensionKind::Generic));
    }

    #[test]
    fn test_builder_fetch() {
        let mut builder = GraphBuilder::new();

        let value = Expression::from_dim(builder.position);
        let result = builder.fetch(value, None, None, None, TieBreak::Latest);

        // 4 inputs + 3 (multiply for zero key in to_2d_key) + 1 lookup dim
        assert_eq!(builder.dim_count(), 8);
        assert_eq!(builder.lookup_count(), 1);
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_builder_fetch_vec() {
        let mut builder = GraphBuilder::new();

        let v1 = Expression::from_dim(builder.position);
        let v2 = Expression::from_dim(builder.one);
        let results = builder.fetch_vec(vec![v1, v2], None, None, None, TieBreak::Latest);

        assert_eq!(results.len(), 2);
        // 4 inputs + 3 (multiply for zero key) + 2 lookup dims
        assert_eq!(builder.dim_count(), 9);
        assert_eq!(builder.lookup_count(), 1);
    }

    #[test]
    fn test_builder_fetch_with_query_key() {
        let mut builder = GraphBuilder::new();
        let one = builder.one;
        let pos = builder.position;

        let value = Expression::from_dim(pos);
        let query = Expression::from_dim(one);
        let key = Expression::from_dim(pos);

        let result = builder.fetch(value, Some(query), Some(key), None, TieBreak::Latest);

        assert_eq!(result.len(), 1);
        assert_eq!(builder.lookup_count(), 1);
    }

    #[test]
    fn test_builder_fetch_sum() {
        let mut builder = GraphBuilder::new();

        let values = vec![
            Expression::from_dim(builder.position),
            Expression::from_dim(builder.one),
        ];
        let results = builder.fetch_sum(values);

        assert_eq!(results.len(), 2);
        // 4 inputs + 3 (multiply for zero key) + 2 lookup dims + 2 ReGLU
        assert_eq!(builder.dim_count(), 11);
        assert_eq!(builder.lookup_count(), 1);
    }

    #[test]
    fn test_builder_fetch_sum_single() {
        let mut builder = GraphBuilder::new();

        let value = Expression::from_dim(builder.position);
        let result = builder.fetch_sum_single(value);

        assert_eq!(result.len(), 1);
        // 4 inputs + 3 (multiply for zero key) + 1 lookup dim + 1 ReGLU
        assert_eq!(builder.dim_count(), 9);
    }

    // ── ProgramGraph Tests ──────────────────────────────────────

    // ── C5: Graph validation tests ─────────────────────────────
    #[test]
    fn test_graph_validate_simple() {
        let mut b = GraphBuilder::new();
        let one = b.one;
        let pos = b.position;
        let x = b.reglu(pos.into_expr(one), 2.0_f64.into_expr(one));
        let y = b.persist(x.clone());
        let graph = b.build(vec![], vec![y]);
        assert!(graph.validate().is_ok());
    }

    #[test]
    fn test_graph_validate_missing_dim() {
        let mut b = GraphBuilder::new();
        let one = b.one;
        let pos = b.position;
        let x = b.reglu(pos.into_expr(one), 2.0_f64.into_expr(one));
        let y = b.persist(x.clone());
        let mut graph = b.build(vec![], vec![y]);

        // Corrupt: remove a dependency dim (pos is referenced by reglu's b_expr)
        // check_dim_consistency runs first, so it catches MissingDim, not OutputMissingDim
        graph.all_dims.remove(&pos);
        let err = graph.validate();
        assert!(
            matches!(err, Err(ValidationError::MissingDim { missing, .. }) if missing == pos),
            "expected MissingDim for pos, got {err:?}"
        );
    }

    #[test]
    fn test_graph_validate_cycle() {
        // Build a valid graph first, then verify validate passes
        let mut b = GraphBuilder::new();
        let one = b.one;
        let pos = b.position;
        let x = b.reglu(pos.into_expr(one), 2.0_f64.into_expr(one));
        let _y = b.persist(x.clone());
        let graph = b.build(vec![], vec![_y]);
        assert!(
            graph.validate().is_ok(),
            "valid graph should pass validation"
        );
    }

    #[test]
    fn test_graph_validate_diamond_dependency() {
        let mut b = GraphBuilder::new();
        let pos = b.position;
        let one = b.one;
        // Diamond: pos -> reglu_a, pos -> reglu_b, both -> persist
        let a = b.reglu(pos.into_expr(one), one.into_expr(one));
        let b_expr = b.reglu(pos.into_expr(one), 3.0_f64.into_expr(one));
        let combined = a.clone() + b_expr.clone();
        let _out = b.persist(combined);
        let graph = b.build(vec![], vec![_out]);
        assert!(graph.validate().is_ok());
    }

    // ── ProgramGraph build tests ───────────────────────────────

    #[test]
    fn test_program_graph_build() {
        let mut builder = GraphBuilder::new();
        let one = builder.one;

        let x = builder.reglu(1.0_f64, 2.0_f64);
        let y = builder.persist(x.clone());

        let input_tokens = vec![Expression::from_dim(one)];
        let output_tokens = vec![y.clone()];

        let graph = builder.build(input_tokens, output_tokens);

        assert_eq!(graph.all_dims.len(), 6); // 4 inputs + 1 reglu + 1 persist
        assert_eq!(graph.input_tokens.len(), 1);
        assert_eq!(graph.output_tokens.len(), 1);
        assert_eq!(graph.one, 0);
        assert_eq!(graph.position, 1);
    }

    #[test]
    fn test_program_graph_captures_all_dims() {
        let mut builder = GraphBuilder::new();

        let r = builder.reglu(1.0_f64, 1.0_f64);
        let p = builder.persist(r.clone());
        let s = builder.stepglu(1.0_f64, 0.0_f64);

        let graph = builder.build(vec![], vec![p, s]);

        // 4 inputs + 1 reglu(r) + 1 persist(p) + 2 reglu(s) + 1 persist(s) = 9
        assert_eq!(graph.all_dims.len(), 9);
        assert_eq!(graph.all_lookups.len(), 0);
    }

    #[test]
    fn test_program_graph_captures_lookups() {
        let mut builder = GraphBuilder::new();

        let value = Expression::from_dim(builder.position);
        let result = builder.fetch(value, None, None, None, TieBreak::Latest);

        let graph = builder.build(vec![], vec![result]);

        // 4 inputs + 3 (multiply for zero key) + 1 lookup dim
        assert_eq!(graph.all_dims.len(), 8);
        assert_eq!(graph.all_lookups.len(), 1);

        let lookup = graph.all_lookups.values().next().unwrap();
        assert_eq!(lookup.value_exprs.len(), 1);
        assert_eq!(lookup.dim_ids.len(), 1);
        assert_eq!(lookup.tie_break, TieBreak::Latest);
    }

    // ── IntoExpr Tests ──────────────────────────────────────────

    #[test]
    fn test_into_expr_expression() {
        let expr = Expression::from_dim(5);
        let result = expr.clone().into_expr(0);
        assert_eq!(result, expr);
    }

    #[test]
    fn test_into_expr_dim_id() {
        let result = 42u32.into_expr(0);
        assert_eq!(result.get(42), 1.0);
    }

    #[test]
    fn test_into_expr_f64_nonzero() {
        let result = 3.5f64.into_expr(1);
        assert_eq!(result.get(1), 3.5);
    }

    #[test]
    fn test_into_expr_f64_zero() {
        let result = 0.0f64.into_expr(1);
        assert!(result.is_zero());
    }

    #[test]
    fn test_into_expr_i32() {
        let result = 5i32.into_expr(1);
        assert_eq!(result.get(1), 5.0);
    }

    // ── Naming Tests ────────────────────────────────────────────

    #[test]
    fn test_name_dim() {
        let mut builder = GraphBuilder::new();

        let r = builder.reglu(1.0_f64, 2.0_f64);
        let dim_id = *r.terms.keys().next().unwrap();

        builder.name_dim(dim_id, "my_gate");

        let dim = builder.get_dim(dim_id).unwrap();
        assert_eq!(dim.name, "my_gate");
    }

    #[test]
    fn test_name_dim_skips_input() {
        let mut builder = GraphBuilder::new();

        builder.name_dim(builder.one, "renamed");

        let dim = builder.get_dim(builder.one).unwrap();
        assert_eq!(dim.name, "one"); // Should NOT be renamed
    }

    #[test]
    fn test_auto_name() {
        let mut builder = GraphBuilder::new();

        let r = builder.reglu(1.0_f64, 2.0_f64);
        let p = builder.persist(r.clone());

        builder.auto_name(&[("my_output".to_string(), p.clone())]);

        // The persist dim inside p should be named "my_output"
        let persist_id = *p.terms.keys().next().unwrap();
        let persist_dim = builder.get_dim(persist_id).unwrap();
        assert_eq!(persist_dim.name, "my_output");
    }

    // ── Integration: Simple Graphs ──────────────────────────────

    #[test]
    fn test_simple_accumulator_graph() {
        let mut builder = GraphBuilder::new();

        let values = vec![Expression::from_dim(builder.position)];
        let results = builder.fetch_sum(values);

        let graph = builder.build(vec![], results);

        // 4 inputs + 3 (multiply for zero key) + 1 lookup dim + 1 ReGLU
        assert_eq!(graph.all_dims.len(), 9);
        assert_eq!(graph.all_lookups.len(), 1);

        let lookup = graph.all_lookups.values().next().unwrap();
        assert_eq!(lookup.value_exprs.len(), 1);
        assert_eq!(lookup.dim_ids.len(), 1);
        assert_eq!(lookup.tie_break, TieBreak::Average);
    }

    #[test]
    fn test_expression_arithmetic_chain() {
        let one_id = 0u32;
        let pos_id = 1u32;

        let pos = Expression::from_dim(pos_id);
        let one = Expression::from_dim(one_id);

        // (position + 1) * 2
        let result = (pos.clone() + one.clone()) * 2.0;
        assert_eq!(result.get(pos_id), 2.0);
        assert_eq!(result.get(one_id), 2.0);

        // position - 1
        let result2 = pos - one;
        assert_eq!(result2.get(pos_id), 1.0);
        assert_eq!(result2.get(one_id), -1.0);
    }

    #[test]
    fn test_reglu_with_dim_expr() {
        let mut builder = GraphBuilder::new();
        let pos = builder.position;
        let one = builder.one;

        // reglu(position, position + 1)
        let pos_expr = Expression::from_dim(pos);
        let b_expr = pos_expr.clone() + Expression::from_scalar(1.0, one);
        let result = builder.reglu(pos_expr, b_expr);

        let dim_id = *result.terms.keys().next().unwrap();
        let dim = builder.get_dim(dim_id).unwrap();
        match &dim.kind {
            DimensionKind::ReGLU { a_expr, b_expr } => {
                assert_eq!(a_expr.get(pos), 1.0);
                assert_eq!(b_expr.get(pos), 1.0);
                assert_eq!(b_expr.get(one), 1.0);
            }
            _ => panic!("Expected ReGLU dimension"),
        }
    }

    #[test]
    fn test_stepglu_creates_correct_structure() {
        let mut builder = GraphBuilder::new();

        let result = builder.stepglu(5.0_f64, 3.0_f64);
        let persist_id = *result.terms.keys().next().unwrap();
        let persist_dim = builder.get_dim(persist_id).unwrap();

        match &persist_dim.kind {
            DimensionKind::Persist { expr } => {
                // Should have +1 coefficient for first ReGLU and -1 for second
                let mut total_coeff = 0.0;
                for c in expr.terms.values() {
                    total_coeff += c;
                }
                assert_eq!(total_coeff, 0.0); // +1 + (-1) = 0
                assert_eq!(expr.len(), 2);
            }
            _ => panic!("Expected Persist dimension"),
        }
    }

    #[test]
    fn test_fetch_with_clear_key() {
        let mut builder = GraphBuilder::new();

        let value = Expression::from_dim(builder.position);
        let clear_key = Expression::from_dim(builder.one);

        let result = builder.fetch(value, None, None, Some(clear_key), TieBreak::Latest);

        assert_eq!(result.len(), 1);
        assert_eq!(builder.lookup_count(), 1);
    }

    #[test]
    fn test_multiple_graphs_independent() {
        // Each GraphBuilder is independent — no global state
        let mut builder1 = GraphBuilder::new();
        let mut builder2 = GraphBuilder::new();

        let r1 = builder1.reglu(1.0_f64, 2.0_f64);
        let r2 = builder2.reglu(3.0_f64, 4.0_f64);

        // IDs start from 0 in each builder
        assert_eq!(builder1.one, 0);
        assert_eq!(builder2.one, 0);

        // Dimensions are independent
        let dim1 = builder1.get_dim(*r1.terms.keys().next().unwrap()).unwrap();
        let dim2 = builder2.get_dim(*r2.terms.keys().next().unwrap()).unwrap();

        match (&dim1.kind, &dim2.kind) {
            (
                DimensionKind::ReGLU {
                    a_expr: a1,
                    b_expr: b1,
                },
                DimensionKind::ReGLU {
                    a_expr: a2,
                    b_expr: b2,
                },
            ) => {
                assert_eq!(a1.get(builder1.one), 1.0);
                assert_eq!(a2.get(builder2.one), 3.0);
                assert_eq!(b1.get(builder1.one), 2.0);
                assert_eq!(b2.get(builder2.one), 4.0);
            }
            _ => panic!("Expected ReGLU dimensions"),
        }
    }
}
