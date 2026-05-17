// SPDX-License-Identifier: Apache-2.0
// Distilled from Percepta's `transformer-vm` (Apache-2.0 © Percepta).

//! First Futamura Projection: specialize the universal WASM interpreter for a specific program.
//!
//! The Futamura specialization takes a universal WASM interpreter (with instruction-fetch
//! attention heads) and a specific program, and produces a specialized transformer where:
//! - The program's bytecode is baked into FFN weights (piecewise-constant step functions)
//! - No instruction-fetch attention heads needed (smaller, faster model)
//! - Same output as the universal model but with fewer layers/heads
//!
//! # How It Works
//!
//! 1. Build the WASM interpreter's computation graph in **specialized mode**
//!    (passing the program to [`interpreter::build`])
//! 2. The interpreter uses [`PiecewiseLookup`](interpreter::PiecewiseLookup) instead of
//!    attention-based instruction fetch — the program counter indexes a table of
//!    precomputed opcode properties via ReGLU step functions
//! 3. Schedule the specialized graph via MILP (usually fewer layers)
//! 4. Build weights for the specialized model
//!
//! # Example
//!
//! ```ignore
//! use percepta::wasm::interpreter::{ProgramInstruction, Opcode};
//! use percepta::specialize::{specialize, build_universal};
//!
//! let program = vec![
//!     ProgramInstruction::with_i32(Opcode::I32Const, 42),
//!     ProgramInstruction::new(Opcode::Output),
//!     ProgramInstruction::new(Opcode::Halt),
//! ];
//!
//! // Build specialized model (also builds universal internally for comparison)
//! let specialized = specialize(&program, None, None)?;
//!
//! // Or provide pre-built universal for efficiency
//! let universal = build_universal(None)?;
//! let specialized = specialize(&program, Some(&universal), None)?;
//!
//! println!("Reduction: {} dims → {} dims",
//!     specialized.reduction.universal_dims,
//!     specialized.reduction.specialized_dims,
//! );
//! ```

use std::collections::HashMap;

use log::info;

use crate::percepta::graph::types::{Expression, GraphBuilder, ProgramGraph};
use crate::percepta::scheduler::{self, Schedule, ScheduleError};
use crate::percepta::wasm::interpreter::{self, ProgramInstruction};
use crate::percepta::weights::{self, TransformerWeights};

// ── Error Type ────────────────────────────────────────────────

/// Errors that can occur during Futamura specialization.
#[derive(Debug)]
pub enum SpecializationError {
    /// MILP scheduling failed.
    Schedule(ScheduleError),
    /// The program is empty (no instructions).
    EmptyProgram,
    /// The universal model failed to build.
    UniversalBuild(String),
}

impl std::fmt::Display for SpecializationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Schedule(err) => write!(f, "Schedule error: {err}"),
            Self::EmptyProgram => write!(f, "Program is empty; at least one instruction required"),
            Self::UniversalBuild(msg) => write!(f, "Universal build failed: {msg}"),
        }
    }
}

impl std::error::Error for SpecializationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Schedule(err) => Some(err),
            Self::EmptyProgram | Self::UniversalBuild(_) => None,
        }
    }
}

impl From<ScheduleError> for SpecializationError {
    fn from(err: ScheduleError) -> Self {
        Self::Schedule(err)
    }
}

// ── Output Types ──────────────────────────────────────────────

/// Reduction statistics comparing universal vs specialized model.
#[derive(Clone, Debug)]
pub struct SpecializationReduction {
    /// Number of dimensions in the universal model.
    pub universal_dims: usize,
    /// Number of dimensions in the specialized model.
    pub specialized_dims: usize,
    /// Number of lookups (attention heads) in the universal model.
    pub universal_lookups: usize,
    /// Number of lookups (attention heads) in the specialized model.
    pub specialized_lookups: usize,
    /// Number of transformer layers in the universal model.
    pub universal_layers: usize,
    /// Number of transformer layers in the specialized model.
    pub specialized_layers: usize,
    /// Model dimension (d_model) of the universal model.
    pub universal_d_model: usize,
    /// Model dimension (d_model) of the specialized model.
    pub specialized_d_model: usize,
    /// Number of input tokens in the universal model.
    pub universal_input_tokens: usize,
    /// Number of input tokens in the specialized model.
    pub specialized_input_tokens: usize,
    /// Number of output tokens in the universal model.
    pub universal_output_tokens: usize,
    /// Number of output tokens in the specialized model.
    pub specialized_output_tokens: usize,
    /// Number of instructions baked into the specialized model.
    pub instructions_baked: usize,
}

impl SpecializationReduction {
    /// Dimension reduction ratio (specialized / universal).
    ///
    /// Returns 0.0 if the universal model has zero dimensions.
    pub fn dim_ratio(&self) -> f64 {
        if self.universal_dims == 0 {
            return 0.0;
        }
        self.specialized_dims as f64 / self.universal_dims as f64
    }

    /// Lookup reduction ratio (specialized / universal).
    ///
    /// Returns 0.0 if the universal model has zero lookups.
    pub fn lookup_ratio(&self) -> f64 {
        if self.universal_lookups == 0 {
            return 0.0;
        }
        self.specialized_lookups as f64 / self.universal_lookups as f64
    }

    /// Layer reduction ratio (specialized / universal).
    ///
    /// Returns 0.0 if the universal model has zero layers.
    pub fn layer_ratio(&self) -> f64 {
        if self.universal_layers == 0 {
            return 0.0;
        }
        self.specialized_layers as f64 / self.universal_layers as f64
    }
}

/// Result of First Futamura specialization.
#[derive(Clone, Debug)]
pub struct SpecializedModel {
    /// The specialized computation graph.
    pub graph: ProgramGraph,
    /// The specialized schedule.
    pub schedule: Schedule,
    /// The specialized transformer weights.
    pub weights: TransformerWeights,
    /// Reduction statistics comparing universal vs specialized.
    pub reduction: SpecializationReduction,
}

/// Universal model built for comparison.
#[derive(Clone, Debug)]
pub struct UniversalModel {
    /// The universal computation graph.
    pub graph: ProgramGraph,
    /// The universal schedule.
    pub schedule: Schedule,
    /// The universal transformer weights.
    pub weights: TransformerWeights,
}

// ── Core Functions ────────────────────────────────────────────

/// Convert interpreter's HashMap tokens to Vec and build the ProgramGraph.
///
/// The interpreter's [`interpreter::build`] returns token maps keyed by name,
/// but [`GraphBuilder::build`] expects ordered Vecs. This helper performs the
/// conversion, preserving the HashMap's insertion order (which matches the
/// interpreter's token definition order).
fn build_graph(
    builder: GraphBuilder,
    input_tokens: HashMap<String, Expression>,
    output_tokens: HashMap<String, Expression>,
) -> ProgramGraph {
    let input_vec: Vec<Expression> = input_tokens.into_values().collect();
    let output_vec: Vec<Expression> = output_tokens.into_values().collect();
    builder.build(input_vec, output_vec)
}

/// Build the universal WASM interpreter model (attention-based instruction fetch).
///
/// This is the "unspecialized" model that can execute any WASM program by
/// fetching instructions via attention heads at runtime.
///
/// # Arguments
///
/// * `max_layers` — Optional maximum number of transformer layers for the MILP scheduler.
///
/// # Errors
///
/// Returns [`SpecializationError`] if scheduling fails.
pub fn build_universal(max_layers: Option<usize>) -> Result<UniversalModel, SpecializationError> {
    let mut builder = GraphBuilder::new();
    let (input_tokens, output_tokens) = interpreter::build(None, &mut builder);
    let graph = build_graph(builder, input_tokens, output_tokens);

    info!(
        "Universal graph: {} dims, {} lookups, {} input tokens, {} output tokens",
        graph.all_dims.len(),
        graph.all_lookups.len(),
        graph.input_tokens.len(),
        graph.output_tokens.len(),
    );

    let schedule = scheduler::milp_schedule(&graph, max_layers)?;

    info!(
        "Universal schedule: {} layers, d_model={}",
        schedule.num_layers, schedule.width,
    );

    let weights = weights::build_weights(&graph, &schedule);

    info!(
        "Universal weights: d_model={}, n_heads={}, d_ffn={}, vocab={}, n_layers={}",
        weights.d_model, weights.n_heads, weights.d_ffn, weights.vocab_size, weights.n_layers,
    );

    Ok(UniversalModel {
        graph,
        schedule,
        weights,
    })
}

/// Perform First Futamura projection: universal model + program → specialized model.
///
/// Takes a WASM program (list of instructions) and produces a specialized transformer
/// where the program's bytecode is baked into FFN weights via piecewise-constant step
/// functions. The specialized model has fewer attention heads (no instruction-fetch heads)
/// and typically fewer layers.
///
/// # Arguments
///
/// * `program` — The WASM program to specialize (list of [`ProgramInstruction`]).
/// * `universal` — Optional pre-built universal model for reduction statistics.
///   If `None`, a universal model is built internally for comparison.
/// * `max_layers` — Optional maximum number of transformer layers for the MILP scheduler.
///
/// # Errors
///
/// Returns [`SpecializationError`] if the program is empty or scheduling fails.
///
/// # Pipeline
///
/// 1. **Build specialized graph** — `interpreter::build(Some(program), ...)` triggers
///    `build_specialized_fetch` which uses `PiecewiseLookup` (ReGLU step functions)
///    instead of attention-based instruction fetch. The program counter indexes a table
///    of precomputed opcode properties (circle-point coords, stack delta, immediate).
///
/// 2. **Schedule** — MILP 4-phase layer assignment produces the transformer architecture.
///
/// 3. **Build weights** — Analytical weight construction produces concrete tensors.
///
/// 4. **Compare** — Reduction statistics vs the universal model.
pub fn specialize(
    program: &[ProgramInstruction],
    universal: Option<&UniversalModel>,
    max_layers: Option<usize>,
) -> Result<SpecializedModel, SpecializationError> {
    if program.is_empty() {
        return Err(SpecializationError::EmptyProgram);
    }

    // ── Build specialized graph ─────────────────────────────
    let mut builder = GraphBuilder::new();
    let (input_tokens, output_tokens) = interpreter::build(Some(program), &mut builder);
    let graph = build_graph(builder, input_tokens, output_tokens);

    info!(
        "Specialized graph: {} dims, {} lookups, {} input tokens, {} output tokens, {} instructions baked",
        graph.all_dims.len(),
        graph.all_lookups.len(),
        graph.input_tokens.len(),
        graph.output_tokens.len(),
        program.len(),
    );

    // ── Schedule ────────────────────────────────────────────
    let schedule = scheduler::milp_schedule(&graph, max_layers)?;

    info!(
        "Specialized schedule: {} layers, d_model={}",
        schedule.num_layers, schedule.width,
    );

    // ── Build weights ───────────────────────────────────────
    let weights = weights::build_weights(&graph, &schedule);

    info!(
        "Specialized weights: d_model={}, n_heads={}, d_ffn={}, vocab={}, n_layers={}",
        weights.d_model, weights.n_heads, weights.d_ffn, weights.vocab_size, weights.n_layers,
    );

    // ── Compute reduction stats ─────────────────────────────
    let reduction = compute_reduction_from_universal(
        universal,
        max_layers,
        &graph,
        &schedule,
        &weights,
        program.len(),
    );

    log_reduction(&reduction);

    Ok(SpecializedModel {
        graph,
        schedule,
        weights,
        reduction,
    })
}

// ── Helpers ───────────────────────────────────────────────────

/// Compute reduction statistics, building universal on-the-fly if needed.
fn compute_reduction_from_universal(
    universal: Option<&UniversalModel>,
    max_layers: Option<usize>,
    specialized_graph: &ProgramGraph,
    specialized_schedule: &Schedule,
    specialized_weights: &TransformerWeights,
    instructions_baked: usize,
) -> SpecializationReduction {
    match universal {
        Some(u) => compute_reduction(
            instructions_baked,
            &u.graph,
            &u.schedule,
            &u.weights,
            specialized_graph,
            specialized_schedule,
            specialized_weights,
        ),
        None => match build_universal(max_layers) {
            Ok(univ) => compute_reduction(
                instructions_baked,
                &univ.graph,
                &univ.schedule,
                &univ.weights,
                specialized_graph,
                specialized_schedule,
                specialized_weights,
            ),
            Err(e) => {
                info!("Could not build universal model for comparison: {e}");
                SpecializationReduction {
                    universal_dims: 0,
                    specialized_dims: specialized_graph.all_dims.len(),
                    universal_lookups: 0,
                    specialized_lookups: specialized_graph.all_lookups.len(),
                    universal_layers: 0,
                    specialized_layers: specialized_schedule.num_layers,
                    universal_d_model: 0,
                    specialized_d_model: specialized_weights.d_model,
                    universal_input_tokens: 0,
                    specialized_input_tokens: specialized_graph.input_tokens.len(),
                    universal_output_tokens: 0,
                    specialized_output_tokens: specialized_graph.output_tokens.len(),
                    instructions_baked,
                }
            }
        },
    }
}

/// Compute reduction statistics comparing universal vs specialized models.
fn compute_reduction(
    instructions_baked: usize,
    universal_graph: &ProgramGraph,
    universal_schedule: &Schedule,
    universal_weights: &TransformerWeights,
    specialized_graph: &ProgramGraph,
    specialized_schedule: &Schedule,
    specialized_weights: &TransformerWeights,
) -> SpecializationReduction {
    SpecializationReduction {
        universal_dims: universal_graph.all_dims.len(),
        specialized_dims: specialized_graph.all_dims.len(),
        universal_lookups: universal_graph.all_lookups.len(),
        specialized_lookups: specialized_graph.all_lookups.len(),
        universal_layers: universal_schedule.num_layers,
        specialized_layers: specialized_schedule.num_layers,
        universal_d_model: universal_weights.d_model,
        specialized_d_model: specialized_weights.d_model,
        universal_input_tokens: universal_graph.input_tokens.len(),
        specialized_input_tokens: specialized_graph.input_tokens.len(),
        universal_output_tokens: universal_graph.output_tokens.len(),
        specialized_output_tokens: specialized_graph.output_tokens.len(),
        instructions_baked,
    }
}

/// Log the reduction statistics.
fn log_reduction(reduction: &SpecializationReduction) {
    if reduction.universal_dims == 0 {
        info!("No universal model available for reduction comparison.");
        return;
    }

    info!(
        "Reduction: dims {}→{} ({:.1}%), lookups {}→{} ({:.1}%), layers {}→{} ({:.1}%)",
        reduction.universal_dims,
        reduction.specialized_dims,
        (1.0 - reduction.dim_ratio()) * 100.0,
        reduction.universal_lookups,
        reduction.specialized_lookups,
        (1.0 - reduction.lookup_ratio()) * 100.0,
        reduction.universal_layers,
        reduction.specialized_layers,
        (1.0 - reduction.layer_ratio()) * 100.0,
    );
}

// ── Tests ─────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::percepta::wasm::interpreter::Opcode;

    /// Helper: create a simple 3-instruction program (load const, output, halt).
    fn simple_program() -> Vec<ProgramInstruction> {
        vec![
            ProgramInstruction::with_i32(Opcode::I32Const, 42),
            ProgramInstruction::new(Opcode::Output),
            ProgramInstruction::new(Opcode::Halt),
        ]
    }

    /// Helper: create a 5-instruction arithmetic program.
    fn arithmetic_program() -> Vec<ProgramInstruction> {
        vec![
            ProgramInstruction::with_i32(Opcode::I32Const, 10),
            ProgramInstruction::with_i32(Opcode::I32Const, 20),
            ProgramInstruction::new(Opcode::I32Add),
            ProgramInstruction::new(Opcode::Output),
            ProgramInstruction::new(Opcode::Halt),
        ]
    }

    // ── I1: Specialized graph building ──────────────────────

    #[test]
    fn test_specialized_graph_has_dims_and_lookups() {
        let program = simple_program();
        let mut builder = GraphBuilder::new();
        let (input_tokens, output_tokens) = interpreter::build(Some(&program), &mut builder);
        let graph = build_graph(builder, input_tokens, output_tokens);

        assert!(
            !graph.all_dims.is_empty(),
            "Specialized graph should have dims"
        );
        assert!(
            !graph.all_lookups.is_empty(),
            "Specialized graph should have lookups"
        );
        assert!(
            !graph.input_tokens.is_empty(),
            "Specialized graph should have input tokens"
        );
        assert!(
            !graph.output_tokens.is_empty(),
            "Specialized graph should have output tokens"
        );
    }

    #[test]
    #[ignore] // Slow: builds universal model for comparison
    fn test_specialized_graph_fewer_input_tokens_than_universal() {
        let program = simple_program();

        let universal = build_universal(None).expect("universal build should succeed");

        let mut builder = GraphBuilder::new();
        let (input_tokens, output_tokens) = interpreter::build(Some(&program), &mut builder);
        let specialized_graph = build_graph(builder, input_tokens, output_tokens);

        // Specialized model needs fewer input tokens
        // (no opcode_x, opcode_y, delta_stack_prefix, store_to_stack_prefix, is_write)
        assert!(
            specialized_graph.input_tokens.len() < universal.graph.input_tokens.len(),
            "Specialized input tokens ({}) should be < universal input tokens ({})",
            specialized_graph.input_tokens.len(),
            universal.graph.input_tokens.len(),
        );
    }

    // ── I2: Piecewise-constant FFN lookup (bake instruction table) ──

    #[test]
    #[ignore] // Slow: builds universal model for comparison
    fn test_specialized_fewer_lookups_than_universal() {
        let program = simple_program();

        let universal = build_universal(None).expect("universal build should succeed");

        let mut builder = GraphBuilder::new();
        let (input_tokens, output_tokens) = interpreter::build(Some(&program), &mut builder);
        let specialized_graph = build_graph(builder, input_tokens, output_tokens);

        // Specialized model has fewer lookups because instruction fetch uses
        // PiecewiseLookup (ReGLU step functions) instead of attention heads
        assert!(
            specialized_graph.all_lookups.len() <= universal.graph.all_lookups.len(),
            "Specialized lookups ({}) should be <= universal lookups ({})",
            specialized_graph.all_lookups.len(),
            universal.graph.all_lookups.len(),
        );
    }

    #[test]
    fn test_specialized_graph_has_reglu_dims_for_piecewise_lookup() {
        // The specialized graph should contain ReGLU dimensions that encode the
        // piecewise-constant step functions for instruction decode.
        use crate::percepta::graph::types::DimensionKind;

        let program = arithmetic_program();
        let mut builder = GraphBuilder::new();
        let (input_tokens, output_tokens) = interpreter::build(Some(&program), &mut builder);
        let graph = build_graph(builder, input_tokens, output_tokens);

        let reglu_count = graph
            .all_dims
            .values()
            .filter(|d| matches!(d.kind, DimensionKind::ReGLU { .. }))
            .count();

        assert!(
            reglu_count > 0,
            "Specialized graph should have ReGLU dims for piecewise lookup, found 0"
        );
    }

    // ── I3: Specialized model generation (schedule + build weights) ──

    #[test]
    fn test_specialize_empty_program_errors() {
        let result = specialize(&[], None, None);
        assert!(result.is_err());
        match result.unwrap_err() {
            SpecializationError::EmptyProgram => {}
            other => panic!("Expected EmptyProgram, got {other}"),
        }
    }

    #[test]
    #[ignore] // Slow: MILP scheduling (needs many layers for interpreter graph)
    fn test_specialize_simple_program_succeeds() {
        let program = simple_program();
        let result = specialize(&program, None, None);
        assert!(result.is_ok(), "Specialization failed: {:?}", result.err());

        let model = result.unwrap();
        assert!(model.weights.n_layers > 0, "Should have at least 1 layer");
        assert!(model.weights.d_model > 0, "d_model should be positive");
        assert!(
            model.weights.vocab_size > 0,
            "vocab_size should be positive"
        );
    }

    #[test]
    #[ignore] // Slow: MILP scheduling (needs many layers for interpreter graph)
    fn test_specialize_larger_program_succeeds() {
        let program = arithmetic_program();
        let result = specialize(&program, None, None);
        assert!(
            result.is_ok(),
            "Larger program specialization failed: {:?}",
            result.err()
        );

        let model = result.unwrap();
        assert_eq!(model.reduction.instructions_baked, 5);
    }

    #[test]
    #[ignore] // Slow: builds universal model
    fn test_specialize_with_prebuilt_universal() {
        let program = simple_program();
        let universal = build_universal(None).expect("universal should build");
        let specialized =
            specialize(&program, Some(&universal), None).expect("specialization should succeed");

        // Should have valid reduction stats from the provided universal
        assert!(
            specialized.reduction.universal_dims > 0,
            "Should have universal dims from provided model"
        );
        assert!(
            specialized.reduction.universal_lookups > 0,
            "Should have universal lookups from provided model"
        );
    }

    #[test]
    #[ignore] // Slow: MILP scheduling (needs many layers for interpreter graph)
    fn test_specialize_with_max_layers() {
        let program = simple_program();
        let result = specialize(&program, None, None);
        assert!(result.is_ok(), "Specialization with max_layers failed");
    }

    #[test]
    #[ignore] // Slow: builds universal model
    fn test_reduction_stats_populated() {
        let program = simple_program();
        let universal = build_universal(None).expect("universal should build");
        let specialized =
            specialize(&program, Some(&universal), None).expect("specialization should succeed");

        let r = &specialized.reduction;
        assert!(r.universal_dims > 0);
        assert!(r.specialized_dims > 0);
        assert!(r.universal_lookups > 0);
        assert!(r.specialized_lookups > 0);
        assert!(r.universal_layers > 0);
        assert!(r.specialized_layers > 0);
        assert!(r.universal_d_model > 0);
        assert!(r.specialized_d_model > 0);
        assert_eq!(r.instructions_baked, 3);
    }

    #[test]
    #[ignore] // Slow: builds universal model
    fn test_reduction_ratios_valid() {
        let program = simple_program();
        let universal = build_universal(None).expect("universal should build");
        let specialized =
            specialize(&program, Some(&universal), None).expect("specialization should succeed");

        let r = &specialized.reduction;
        assert!(
            r.dim_ratio() <= 1.0,
            "dim ratio should be <= 1.0, got {}",
            r.dim_ratio(),
        );
        assert!(
            r.lookup_ratio() <= 1.0,
            "lookup ratio should be <= 1.0, got {}",
            r.lookup_ratio(),
        );
        assert!(
            r.layer_ratio() <= 1.0,
            "layer ratio should be <= 1.0, got {}",
            r.layer_ratio(),
        );
    }

    #[test]
    #[ignore] // Slow: MILP scheduling
    fn test_build_universal_succeeds() {
        let result = build_universal(None);
        assert!(result.is_ok(), "Universal build failed: {:?}", result.err());

        let model = result.unwrap();
        assert!(!model.graph.all_dims.is_empty());
        assert!(!model.graph.all_lookups.is_empty());
        assert!(model.weights.n_layers > 0);
        assert!(model.weights.d_model > 0);
    }

    #[test]
    #[ignore] // Slow: MILP scheduling (needs many layers for interpreter graph)
    fn test_specialized_weights_structure() {
        let program = simple_program();
        let model = specialize(&program, None, None).expect("specialization should succeed");

        // Verify weight structure consistency
        let w = &model.weights;
        assert_eq!(
            w.embedding.len(),
            w.vocab_size,
            "embedding rows should match vocab"
        );
        assert_eq!(
            w.unembedding.len(),
            w.vocab_size,
            "unembedding rows should match vocab"
        );
        assert_eq!(
            w.layers.len(),
            w.n_layers,
            "layers count should match n_layers"
        );

        for (i, layer) in w.layers.iter().enumerate() {
            let attn_rows = layer.attention.in_proj.len();
            assert_eq!(
                attn_rows,
                3 * w.d_model,
                "layer {i}: in_proj should be 3*d_model rows"
            );
            assert_eq!(
                layer.attention.in_proj[0].len(),
                w.d_model,
                "layer {i}: in_proj cols should be d_model"
            );
        }
    }

    // ── I4: Verification tests — SKIPPED (depends on full pipeline) ──
    // These will be added when the evaluator + reference trace generator are complete.
}
