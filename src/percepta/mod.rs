//! Percepta-style O(log N) 2D Attention via Convex Hull KV Cache.
//!
//! Standard transformer attention computes Q·K for all N past keys → O(N) per step.
//! Percepta restricts attention heads to d=2, making the dot product a 2D geometric
//! projection. When keys form a convex hull, finding the maximum attention score
//! becomes ternary search over a unimodal (bitonic) sequence → O(log N).
//!
//! Integration points with microgpt-rs:
//! - DDTree branch pruning: validate drafted tokens before target verification
//! - Deterministic Validator: encode state-machine rules as 2D key embeddings
//! - "Free embedding" bridge: project hidden states to 2D for fast retrieval
//!
//! # Module layout
//!
//! - [`legacy`] — Original KVCache2D (Graham Scan + ternary search), Sudoku, StreamingSolver
//! - [`types`]  — Shared types for CHT hull vertices (`HullMeta`, `TieBreak`, `Vec2` f64)
//! - [`cht`]    — Dynamic Convex Hull Trick (line container, upper envelope)
//! - [`hull`]     — HullHalf, HardAttentionHead, BruteAttentionHead (O(log N) 2D hard attention)
//! - [`encoding`] — Parabolic key encoding helpers for 2D attention
//! - [`cumsum`]  — Cumulative sum via uniform attention (fetch_sum equivalent)
//! - [`transformer`] — VanillaTransformer with ReGLU FFN, autoregressive generation

// ── Submodules ─────────────────────────────────────────────────

/// Legacy KVCache2D (Graham Scan + ternary search), Sudoku9x9, StreamingSolver.
/// Always compiled — no optional dependencies.
pub mod legacy;

/// Shared types for the CHT Hull KV Cache: `HullMeta`, `TieBreak`, `Vec2` (f64).
#[cfg(feature = "percepta")]
pub mod types;

/// Dynamic Convex Hull Trick / LineContainer for O(log h) max-envelope queries.
#[cfg(feature = "percepta")]
pub mod cht;

/// CHT Hull KV Cache — `HullHalf`, `HardAttentionHead`, `BruteAttentionHead`.
#[cfg(feature = "percepta")]
pub mod hull;

/// Parabolic key encoding helpers for 2D attention.
#[cfg(feature = "percepta")]
pub mod encoding;

/// Cumulative sum via uniform attention (fetch_sum equivalent).
#[cfg(feature = "percepta")]
pub mod cumsum;

/// Standard O(n) softmax attention KV cache for correctness verification.
#[cfg(feature = "percepta")]
pub mod standard_cache;

/// ReGLU, stepglu, multiply, persist gate primitives: `GateKind`, `PersistSlot`.
#[cfg(feature = "percepta_gates")]
pub mod gates;

/// Expression/Dimension DSL computation graph: `Expression`, `Dimension`, `GraphBuilder`, `ProgramGraph`.
#[cfg(feature = "percepta_graph")]
pub mod graph;

/// WASM decoder + lowering passes for transformer-vm compilation.
#[cfg(feature = "percepta_wasm")]
pub mod wasm;

/// MILP scheduler: 4-phase layer assignment minimizing `d_model`.
#[cfg(all(feature = "percepta_graph", feature = "good_lp"))]
pub mod scheduler;

/// Analytical weight construction: graph + schedule → transformer weight tensors.
#[cfg(feature = "percepta_compile")]
pub mod weights;

/// Vanilla transformer with ReGLU FFN and CHT hull KV cache: `VanillaTransformer`, autoregressive generation.
#[cfg(feature = "percepta_compile")]
pub mod transformer;

/// First Futamura projection: specialize universal WASM interpreter for a specific program.
#[cfg(feature = "percepta_compile")]
pub mod specialize;

/// Graph evaluator with exact arithmetic for correctness verification (no transformer weights needed).
#[cfg(feature = "percepta_compile")]
pub mod evaluator;

/// Pipeline runner: compile → build → run → evaluate orchestration.
#[cfg(feature = "percepta_compile")]
pub mod runner;

/// C → WASM → dispatch table → token prefix compile pipeline.
#[cfg(feature = "percepta_compile")]
pub mod compile;

// ── Re-exports from legacy (always available) ──────────────────

pub use legacy::{KVCache2D, SolveEvent, StreamingSolver, Sudoku9x9, SymbolicValidator, Vec2};

// ── Re-exports from hull (feature-gated) ──────────────────────

#[cfg(feature = "percepta")]
pub use hull::{AttentionResult, BruteAttentionHead, HardAttentionHead, HullHalf};

#[cfg(feature = "percepta")]
pub use types::TieBreak;

#[cfg(feature = "percepta")]
pub use encoding::{clear_key, encode_key, encode_query, hard_scale, hard_scale_query};

#[cfg(feature = "percepta")]
pub use cumsum::CumSum;

#[cfg(feature = "percepta")]
pub use standard_cache::StandardCache;

// ── Re-exports from gates (feature-gated) ──────────────────────

#[cfg(feature = "percepta_gates")]
pub use gates::{GateKind, PersistSlot, multiply, reglu, stepglu};

// ── Re-exports from graph (feature-gated) ──────────────────────

#[cfg(feature = "percepta_graph")]
pub use graph::{
    BIG, DimId, Dimension, DimensionKind, Expression, GraphBuilder, IntoExpr, KEY_OFFSET,
    LATEST_ALPHA, LookUp, LookupId, ProgramGraph, ValidationError,
};

// ── Re-exports from scheduler (feature-gated) ──────────────────

#[cfg(all(feature = "percepta_graph", feature = "good_lp"))]
pub use scheduler::{
    DepGraph, OpKey, Phase, Schedule, ScheduleError, StdLayer, build_dep_graph, interval_coloring,
    milp_schedule, min_layers,
};

// ── Re-exports from weights (feature-gated) ────────────────────

#[cfg(feature = "percepta_compile")]
pub use weights::{
    AttentionWeights, FfnWeights, HeadInfo, LayerWeights, TransformerWeights, build_weights,
    expr_to_vector,
};

// ── Re-exports from transformer (feature-gated) ───────────────

#[cfg(feature = "percepta_compile")]
pub use transformer::{
    GenerationResult, TransformerConfig, TransformerVocab, VanillaTransformer, decode_trace,
    encode_output_byte, parse_output_byte,
};

// ── Re-exports from specialize (feature-gated) ────────────────

#[cfg(feature = "percepta_compile")]
pub use specialize::{
    SpecializationError, SpecializationReduction, SpecializedModel, UniversalModel,
    build_universal, specialize,
};

// ── Re-exports from evaluator (feature-gated) ─────────────────

#[cfg(feature = "percepta_compile")]
pub use evaluator::{EvalError, GraphEvaluator};

// ── Re-exports from runner (feature-gated) ─────────────────────

#[cfg(feature = "percepta_compile")]
pub use runner::{BuildResult, Runner, RunnerError};

// ── Re-exports from compile (feature-gated) ───────────────────

#[cfg(feature = "percepta_compile")]
pub use compile::{
    CompileError, CompiledProgram, RUNTIME_H, compile_c_to_wasm, compile_program,
    compile_wasm_to_prefix, find_clang, format_input_section, format_prefix, format_spec_input,
    int_to_bytes, write_runtime_h,
};
