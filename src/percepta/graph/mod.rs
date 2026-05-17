//! Expression/Dimension DSL computation graph for Percepta's transformer-vm.
//!
//! This module implements the computation graph primitives that any
//! Append-Only Lookup Machine (ALM) program is built from. The graph captures
//! the dependency structure between dimensions for scheduling and weight
//! construction.
//!
//! # Core Types
//!
//! - [`Expression`] — Sparse linear combination of dimensions (`HashMap<DimId, f64>`)
//! - [`Dimension`] — Named dimension with [`DimensionKind`] variant
//! - [`DimensionKind`] — 6 variants: Input, ReGLU, Persist, LookUp, CumSum, Generic
//! - [`LookUp`] — Attention-based retrieval from token history
//! - [`ProgramGraph`] — Captured computation graph ready for scheduling
//! - [`GraphBuilder`] — Mutable state for building computation graphs (no globals)
//!
//! # Builder Functions
//!
//! All builder functions are methods on [`GraphBuilder`]:
//!
//! - [`GraphBuilder::reglu`] — `relu(b) × a` gated FFN unit
//! - [`GraphBuilder::stepglu`] — `a × step(b ≥ 0)` conditional gate
//! - [`GraphBuilder::persist`] — Materialize expression into residual slot
//! - [`GraphBuilder::fetch`] — Attention-based retrieval (single value)
//! - [`GraphBuilder::fetch_vec`] — Attention-based retrieval (multiple values)
//! - [`GraphBuilder::fetch_sum`] — Cumulative sum via attention averaging
//!
//! # Design Differences from Python
//!
//! - Uses `u32` IDs for dimensions instead of Python's object identity
//! - Uses `HashMap<DimId, f64>` for Expression terms (sparse)
//! - No global mutable state — uses a `GraphBuilder` struct
//! - Expression arithmetic implements `Add`, `Sub`, `Mul`, `Neg` traits
//!
//! Distilled from Percepta's `transformer-vm` (Apache-2.0 © Percepta).
//! Reference: `.raw/transformer-vm/transformer_vm/graph/core.py` (449 lines)

/// Core types: Expression, Dimension, DimensionKind, LookUp, ProgramGraph, GraphBuilder.
pub mod types;

// ── Re-exports ──────────────────────────────────────────────────

pub use types::{
    BIG, DimId, Dimension, DimensionKind, Expression, GraphBuilder, IntoExpr, KEY_OFFSET,
    LATEST_ALPHA, LookUp, LookupId, ProgramGraph, ValidationError,
};
