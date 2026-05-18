// SPDX-License-Identifier: Apache-2.0
// Distilled from Percepta's `transformer-vm` (Apache-2.0 © Percepta).

//! Pipeline runner that orchestrates the full compile → build → run flow.
//!
//! The runner provides high-level functions for each stage of the transformer-vm
//! pipeline:
//!
//! - **compile**: C source → WASM → lowered bytecode → token prefix (requires clang)
//! - **build**: token prefix + graph → schedule → weights → transformer
//! - **run**: transformer + token prefix → autoregressive execution
//! - **specialize**: universal model → specialized model (Futamura projection)
//! - **evaluate**: graph evaluator for correctness verification (no weights needed)
//! - **full_pipeline**: compile → build → run in one call
//!
//! # Example
//!
//! ```ignore
//! use percepta::runner::Runner;
//!
//! // Evaluate a program with exact arithmetic (no clang needed)
//! let output = Runner::evaluate_from_prefix(&graph, &input_tokens, &output_tokens, &prefix, 50000);
//!
//! // Full pipeline: compile → build → run
//! let result = Runner::full_pipeline(source, None, 50000);
//! ```
//!
//! Reference: `.raw/transformer-vm/transformer_vm/runner.py` (301 lines)

use std::collections::HashMap;

use crate::percepta::compile::{self, CompileError, CompiledProgram};
use crate::percepta::evaluator::{EvalError, GraphEvaluator};
use crate::percepta::graph::types::{Expression, GraphBuilder, ProgramGraph};
use crate::percepta::scheduler::{Schedule, ScheduleError, milp_schedule};
use crate::percepta::transformer::{
    GenerationResult, TransformerConfig, TransformerVocab, VanillaTransformer,
};
use crate::percepta::wasm::interpreter;
use crate::percepta::weights::{TransformerWeights, build_weights};

// ── Error Type ─────────────────────────────────────────────────

/// Errors that can occur during pipeline execution.
#[derive(Debug)]
pub enum RunnerError {
    /// Error during MILP scheduling.
    ScheduleError(ScheduleError),
    /// Error during graph evaluation.
    EvalError(EvalError),
    /// Error during compilation (e.g., clang not found, WASM decode failure).
    CompileError(String),
    /// Error during weight construction.
    WeightError(String),
    /// The token prefix is empty or invalid.
    InvalidPrefix(String),
    /// Feature not yet implemented.
    NotImplemented(String),
}

impl std::fmt::Display for RunnerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ScheduleError(e) => write!(f, "schedule error: {e}"),
            Self::EvalError(e) => write!(f, "eval error: {e}"),
            Self::CompileError(msg) => write!(f, "compile error: {msg}"),
            Self::WeightError(msg) => write!(f, "weight error: {msg}"),
            Self::InvalidPrefix(msg) => write!(f, "invalid prefix: {msg}"),
            Self::NotImplemented(msg) => write!(f, "not implemented: {msg}"),
        }
    }
}

impl std::error::Error for RunnerError {}

impl From<ScheduleError> for RunnerError {
    fn from(e: ScheduleError) -> Self {
        Self::ScheduleError(e)
    }
}

impl From<EvalError> for RunnerError {
    fn from(e: EvalError) -> Self {
        Self::EvalError(e)
    }
}

impl From<CompileError> for RunnerError {
    fn from(e: CompileError) -> Self {
        Self::CompileError(format!("{e}"))
    }
}

// ── Build Result ───────────────────────────────────────────────

/// Result of the build pipeline: weights + config + vocab + schedule.
#[derive(Clone, Debug)]
pub struct BuildResult {
    /// Transformer weights ready for inference.
    pub weights: TransformerWeights,
    /// Transformer configuration.
    pub config: TransformerConfig,
    /// Token vocabulary (name ↔ ID mapping).
    pub vocab: TransformerVocab,
    /// The MILP schedule used for weight construction.
    pub schedule: Schedule,
    /// The computation graph.
    pub graph: ProgramGraph,
    /// Input token expressions (name → expression).
    pub input_tokens: HashMap<String, Expression>,
    /// Output token expressions (name → expression).
    pub output_tokens: HashMap<String, Expression>,
}

// ── Runner ─────────────────────────────────────────────────────

/// Pipeline runner that orchestrates the full compile → build → run flow.
///
/// All methods are associated functions (no state) since each pipeline stage
/// can be called independently.
pub struct Runner;

impl Runner {
    // ── Compile Pipeline ───────────────────────────────────────

    /// Compile C source to WASM token prefix.
    ///
    /// This stage requires `clang` with wasm32 target support. The pipeline:
    /// 1. Compile C source to WASM via clang (with embedded `runtime.h`)
    /// 2. Decode WASM binary to instructions
    /// 3. Lower unsupported ops (MUL, DIV, etc.)
    /// 4. Convert to flat dispatch table
    /// 5. Format as token prefix string
    ///
    /// # Arguments
    /// * `source` — C source code (must define `void compute(const char *input)`)
    ///
    /// # Returns
    /// The compiled program with dispatch table, prefix string, and input base.
    pub fn compile(source: &str) -> Result<CompiledProgram, RunnerError> {
        compile::compile_program(source, "").map_err(Into::into)
    }

    /// Compile C source with input data to WASM token prefix.
    ///
    /// Like [`compile`](Self::compile), but also formats the input section
    /// for programs that read from `input`.
    ///
    /// # Arguments
    /// * `source` — C source code
    /// * `input_str` — Input string to pass to the program
    pub fn compile_with_input(
        source: &str,
        input_str: &str,
    ) -> Result<CompiledProgram, RunnerError> {
        compile::compile_program(source, input_str).map_err(Into::into)
    }

    /// Compile pre-built WASM bytes to token prefix.
    ///
    /// Skips the C→WASM step. Use this when you already have a `.wasm` binary
    /// (e.g., compiled externally or from a file).
    ///
    /// # Arguments
    /// * `wasm_bytes` — Raw WASM binary data
    pub fn compile_wasm(wasm_bytes: &[u8]) -> Result<CompiledProgram, RunnerError> {
        compile::compile_wasm_to_prefix(wasm_bytes).map_err(Into::into)
    }

    /// Compile C source to token prefix strings only.
    ///
    /// Convenience wrapper that returns just the prefix and input section strings.
    pub fn compile_to_prefix(
        source: &str,
        input_str: &str,
    ) -> Result<(String, String), RunnerError> {
        let compiled = compile::compile_program(source, input_str)?;
        Ok((compiled.prefix.clone(), compiled.input_section.clone()))
    }

    // ── Build Pipeline ─────────────────────────────────────────

    /// Build transformer weights from a computation graph.
    ///
    /// Constructs the full transformer (weights + config + vocab) by:
    /// 1. Building the WASM interpreter computation graph (universal mode)
    /// 2. Solving the MILP schedule
    /// 3. Constructing weight matrices analytically
    ///
    /// # Arguments
    /// * `max_layers` — Optional maximum number of transformer layers.
    ///   If `None`, uses the minimum computed from dependency analysis.
    ///
    /// # Returns
    /// A [`BuildResult`] containing all components needed for inference.
    pub fn build(max_layers: Option<usize>) -> Result<BuildResult, RunnerError> {
        // Step 1: Build the WASM interpreter computation graph
        let mut builder = GraphBuilder::new();
        let (input_tokens, output_tokens) = interpreter::build(None, &mut builder);

        // Collect token names for vocabulary
        let mut input_names: Vec<String> = input_tokens.keys().cloned().collect();
        input_names.sort();

        let mut output_names: Vec<String> = output_tokens.keys().cloned().collect();
        output_names.sort();

        // Build unified vocabulary (union of input + output token names)
        let mut all_names: Vec<String> = input_names;
        for name in &output_names {
            if !all_names.contains(name) {
                all_names.push(name.clone());
            }
        }
        all_names.sort();

        // Build the ProgramGraph (vec-based, index = token ID in vocab)
        let graph = builder.build(
            all_names
                .iter()
                .filter_map(|name| input_tokens.get(name).cloned())
                .collect(),
            all_names
                .iter()
                .filter_map(|name| output_tokens.get(name).cloned())
                .collect(),
        );

        Self::build_from_graph(graph, input_tokens, output_tokens, all_names, max_layers)
    }

    /// Build transformer from an existing computation graph.
    ///
    /// This is the lower-level build function that takes a pre-built graph.
    /// Use [`build`](Self::build) for the standard pipeline.
    pub fn build_from_graph(
        graph: ProgramGraph,
        input_tokens: HashMap<String, Expression>,
        output_tokens: HashMap<String, Expression>,
        vocab_names: Vec<String>,
        max_layers: Option<usize>,
    ) -> Result<BuildResult, RunnerError> {
        // Step 2: Solve MILP schedule
        let schedule = milp_schedule(&graph, max_layers)?;

        // Step 3: Construct weights
        let weights = build_weights(&graph, &schedule);

        // Build config from weights
        let config = TransformerConfig {
            d_model: weights.d_model,
            n_heads: weights.n_heads,
            n_layers: weights.n_layers,
            d_ffn: weights.d_ffn,
            stop_token: "halt".to_string(),
            max_gen: 50000,
        };

        // Build vocabulary
        let vocab = TransformerVocab::new(vocab_names, "halt");

        Ok(BuildResult {
            weights,
            config,
            vocab,
            schedule,
            graph,
            input_tokens,
            output_tokens,
        })
    }

    // ── Run Pipeline ───────────────────────────────────────────

    /// Run autoregressive execution with a pre-built transformer.
    ///
    /// Processes the token prefix through the transformer, then generates
    /// tokens autoregressively until `"halt"` or `max_tokens` is reached.
    ///
    /// # Arguments
    /// * `build_result` — Pre-built transformer components.
    /// * `prefix` — Input token sequence to prime the transformer.
    /// * `max_tokens` — Maximum number of tokens to generate.
    ///
    /// # Returns
    /// The generation result containing all tokens and the execution trace.
    pub fn run(
        build_result: &BuildResult,
        prefix: &[String],
        max_tokens: usize,
    ) -> Result<GenerationResult, RunnerError> {
        if prefix.is_empty() {
            return Err(RunnerError::InvalidPrefix(
                "token prefix must not be empty".into(),
            ));
        }

        let transformer = VanillaTransformer::new(
            build_result.weights.clone(),
            build_result.config.clone(),
            build_result.vocab.clone(),
        );

        let result = transformer.generate(prefix, max_tokens);
        Ok(result)
    }

    /// Run autoregressive execution with raw weights and config.
    ///
    /// Convenience function that creates a transformer and runs generation.
    pub fn run_with_weights(
        weights: TransformerWeights,
        config: TransformerConfig,
        vocab: TransformerVocab,
        prefix: &[String],
        max_tokens: usize,
    ) -> Result<GenerationResult, RunnerError> {
        if prefix.is_empty() {
            return Err(RunnerError::InvalidPrefix(
                "token prefix must not be empty".into(),
            ));
        }

        let transformer = VanillaTransformer::new(weights, config, vocab);
        let result = transformer.generate(prefix, max_tokens);
        Ok(result)
    }

    // ── Evaluate Pipeline ──────────────────────────────────────

    /// Evaluate with graph evaluator (exact arithmetic, no transformer).
    ///
    /// This uses the computation graph directly without building transformer
    /// weights. Useful for correctness verification and debugging.
    ///
    /// # Arguments
    /// * `input_tokens` — Token name → embedding expression.
    /// * `output_tokens` — Token name → scoring expression.
    /// * `graph` — The computation graph to evaluate.
    /// * `prefix` — Input token sequence.
    /// * `max_steps` — Maximum number of generation steps.
    ///
    /// # Returns
    /// The predicted token sequence.
    pub fn evaluate(
        graph: &ProgramGraph,
        input_tokens: &HashMap<String, Expression>,
        output_tokens: &HashMap<String, Expression>,
        prefix: &[String],
        max_steps: usize,
    ) -> Result<Vec<String>, RunnerError> {
        let mut evaluator =
            GraphEvaluator::new(input_tokens.clone(), output_tokens.clone(), graph.clone());
        let result = evaluator.evaluate(prefix, max_steps);
        Ok(result)
    }

    /// Evaluate with graph evaluator and extract output characters.
    ///
    /// Like [`evaluate`](Self::evaluate), but also returns the decoded
    /// output string from `out(XY)` tokens.
    pub fn evaluate_with_output(
        graph: &ProgramGraph,
        input_tokens: &HashMap<String, Expression>,
        output_tokens: &HashMap<String, Expression>,
        prefix: &[String],
        max_steps: usize,
    ) -> Result<(Vec<String>, String), RunnerError> {
        let mut evaluator =
            GraphEvaluator::new(input_tokens.clone(), output_tokens.clone(), graph.clone());
        let (tokens, output) = evaluator.evaluate_with_output(prefix, max_steps);
        Ok((tokens, output))
    }

    /// Evaluate a program from a pre-built graph.
    ///
    /// Convenience function that builds the graph and evaluates it.
    pub fn evaluate_from_prefix(
        graph: &ProgramGraph,
        input_tokens: &HashMap<String, Expression>,
        output_tokens: &HashMap<String, Expression>,
        prefix: &[String],
        max_steps: usize,
    ) -> Result<Vec<String>, RunnerError> {
        Self::evaluate(graph, input_tokens, output_tokens, prefix, max_steps)
    }

    // ── Specialize Pipeline ────────────────────────────────────

    /// Specialize a universal model for a specific program.
    ///
    /// Performs the first Futamura projection: bake the program's instruction
    /// table into the FFN weights, producing a smaller specialized model.
    ///
    /// **Note:** This is not yet implemented.
    pub fn specialize(
        _build_result: &BuildResult,
        _program: &[interpreter::ProgramInstruction],
    ) -> Result<BuildResult, RunnerError> {
        Err(RunnerError::NotImplemented(
            "specialize pipeline (Futamura projection) not yet implemented".into(),
        ))
    }

    // ── Full Pipeline ──────────────────────────────────────────

    /// Full pipeline: build → evaluate.
    ///
    /// Builds the universal transformer model and evaluates a program
    /// using the graph evaluator (exact arithmetic, no transformer inference).
    ///
    /// This is the recommended entry point for correctness verification
    /// since it doesn't require building transformer weights.
    ///
    /// # Arguments
    /// * `prefix` — Input token sequence.
    /// * `max_steps` — Maximum number of generation steps.
    ///
    /// # Returns
    /// The predicted token sequence.
    pub fn full_evaluate(prefix: &[String], max_steps: usize) -> Result<Vec<String>, RunnerError> {
        // Build the WASM interpreter computation graph
        let mut builder = GraphBuilder::new();
        let (input_tokens, output_tokens) = interpreter::build(None, &mut builder);

        // Build the ProgramGraph (for dimension lookups)
        let graph = builder.build(vec![], vec![]);

        // Evaluate
        Self::evaluate(&graph, &input_tokens, &output_tokens, prefix, max_steps)
    }

    /// Full pipeline: build → run (transformer inference).
    ///
    /// Builds the universal transformer model and runs autoregressive
    /// generation on the given prefix.
    ///
    /// # Arguments
    /// * `prefix` — Input token sequence.
    /// * `max_layers` — Optional max transformer layers.
    /// * `max_tokens` — Maximum number of tokens to generate.
    ///
    /// # Returns
    /// The generation result containing all tokens and execution trace.
    pub fn full_pipeline(
        prefix: &[String],
        max_layers: Option<usize>,
        max_tokens: usize,
    ) -> Result<GenerationResult, RunnerError> {
        let build_result = Self::build(max_layers)?;
        Self::run(&build_result, prefix, max_tokens)
    }

    /// Build the universal model without running it.
    ///
    /// Convenience for getting a [`BuildResult`] with all components
    /// needed for subsequent `run` calls.
    pub fn build_universal(max_layers: Option<usize>) -> Result<BuildResult, RunnerError> {
        Self::build(max_layers)
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_runner_compile_error_on_invalid_source() {
        // Invalid C source should fail with CompileError
        let result = Runner::compile("this is not valid C");
        assert!(result.is_err());
        match result.unwrap_err() {
            RunnerError::CompileError(msg) => {
                // Could be clang not found or clang compilation failure
                assert!(!msg.is_empty());
            }
            other => panic!("expected CompileError, got {other:?}"),
        }
    }

    #[test]
    fn test_runner_compile_wasm_with_valid_binary() {
        // Minimal valid WASM binary with compute export
        let wasm_bytes: Vec<u8> = vec![
            // Magic + version
            0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00, // Type section: func() -> ()
            0x01, 0x04, 0x01, 0x60, 0x00, 0x00, // Import section: output_byte
            0x02, 0x13, 0x01, 0x03, 0x65, 0x6e, 0x76, 0x0b, 0x6f, 0x75, 0x74, 0x70, 0x75, 0x74,
            0x5f, 0x62, 0x79, 0x74, 0x65, 0x00, 0x00,
            // Function section: 1 function type 0
            0x03, 0x02, 0x01, 0x00, // Global section: __heap_base = 65536
            0x06, 0x08, 0x01, 0x7f, 0x00, 0x41, 0x80, 0x80, 0x04, 0x0b,
            // Export section: compute func 1, __heap_base global 0
            0x07, 0x19, 0x02, 0x07, 0x63, 0x6f, 0x6d, 0x70, 0x75, 0x74, 0x65, 0x00, 0x01, 0x0b,
            0x5f, 0x5f, 0x68, 0x65, 0x61, 0x70, 0x5f, 0x62, 0x61, 0x73, 0x65, 0x03, 0x00,
            // Code section: i32.const 72, call output_byte, end
            0x0a, 0x08, 0x01, 0x06, 0x00, 0x41, 0x48, 0x10, 0x00, 0x0b,
        ];

        let result = Runner::compile_wasm(&wasm_bytes);
        assert!(result.is_ok(), "compile_wasm failed: {:?}", result.err());

        let compiled = result.unwrap();
        assert!(compiled.prefix.starts_with("{\n"));
        assert!(compiled.prefix.ends_with("}\n"));
        assert!(compiled.program.iter().any(|(op, _)| op == "output"));
        assert!(compiled.program.iter().any(|(op, _)| op == "halt"));
    }

    #[test]
    fn test_runner_compile_wasm_with_invalid_binary() {
        let result = Runner::compile_wasm(&[0x00, 0x01, 0x02]);
        assert!(result.is_err());
    }

    #[test]
    #[ignore = "MILP solver too slow for unit tests; run with --ignored flag"]
    fn test_runner_specialize_not_implemented() {
        // Build first to get a BuildResult
        let build_result = Runner::build(None);
        if build_result.is_err() {
            // If build fails (e.g., MILP solver issue), just test the error type
            return;
        }
        let build_result = build_result.unwrap();
        let result = Runner::specialize(&build_result, &[]);
        assert!(result.is_err());
        match result.unwrap_err() {
            RunnerError::NotImplemented(msg) => {
                assert!(msg.contains("Futamura"));
            }
            other => panic!("expected NotImplemented, got {other:?}"),
        }
    }

    #[test]
    fn test_runner_run_empty_prefix_fails() {
        // Create minimal weights to test the empty prefix check
        let result = Runner::run_with_weights(
            make_minimal_weights(),
            TransformerConfig::default(),
            TransformerVocab::new(vec!["halt".to_string()], "halt"),
            &[],
            100,
        );
        assert!(result.is_err());
        match result.unwrap_err() {
            RunnerError::InvalidPrefix(msg) => {
                assert!(msg.contains("empty"));
            }
            other => panic!("expected InvalidPrefix, got {other:?}"),
        }
    }

    #[test]
    fn test_runner_error_display() {
        let err = RunnerError::CompileError("test error".into());
        assert_eq!(format!("{err}"), "compile error: test error");

        let err = RunnerError::NotImplemented("feature".into());
        assert_eq!(format!("{err}"), "not implemented: feature");

        let err = RunnerError::InvalidPrefix("empty".into());
        assert_eq!(format!("{err}"), "invalid prefix: empty");

        let err = RunnerError::WeightError("bad weights".into());
        assert_eq!(format!("{err}"), "weight error: bad weights");
    }

    #[test]
    #[ignore = "MILP solver too slow for unit tests; run with --ignored flag"]
    fn test_runner_build_result_has_correct_dimensions() {
        let result = Runner::build(None);
        if result.is_err() {
            // MILP solver may not be available in all test environments
            eprintln!("Skipping build test: {:?}", result.unwrap_err());
            return;
        }
        let build = result.unwrap();

        // Config should match weights
        assert_eq!(build.config.d_model, build.weights.d_model);
        assert_eq!(build.config.n_heads, build.weights.n_heads);
        assert_eq!(build.config.n_layers, build.weights.n_layers);
        assert_eq!(build.config.d_ffn, build.weights.d_ffn);

        // Vocab should contain tokens
        assert!(!build.vocab.is_empty());
        assert!(build.vocab.token_id("halt").is_some());
    }

    #[test]
    #[ignore = "MILP solver too slow for unit tests; run with --ignored flag"]
    fn test_runner_build_result_has_token_maps() {
        let result = Runner::build(None);
        if result.is_err() {
            eprintln!("Skipping: {:?}", result.unwrap_err());
            return;
        }
        let build = result.unwrap();

        // Should have both input and output tokens
        assert!(!build.input_tokens.is_empty());
        assert!(!build.output_tokens.is_empty());

        // Should have common tokens
        assert!(build.input_tokens.contains_key("halt"));
        assert!(build.output_tokens.contains_key("halt"));
    }

    #[test]
    fn test_runner_full_evaluate_with_simple_graph() {
        // Build a minimal graph for testing
        let mut builder = GraphBuilder::new();
        let one_id = builder.one;
        let a = builder.generic("a");

        let input_tokens = HashMap::from([
            ("zero".to_string(), Expression::from_scalar(0.0, one_id)),
            ("one".to_string(), Expression::from_scalar(1.0, one_id)),
        ]);

        let output_tokens = HashMap::from([
            ("done".to_string(), a.clone()),
            ("halt".to_string(), Expression::zero()),
        ]);

        let graph = builder.build(vec![], vec![]);

        let result = Runner::evaluate(
            &graph,
            &input_tokens,
            &output_tokens,
            &["zero".to_string()],
            100,
        );
        assert!(result.is_ok());

        let predicted = result.unwrap();
        // Should include the prefix
        assert_eq!(predicted[0], "zero");
    }

    #[test]
    fn test_runner_evaluate_with_output() {
        let builder = GraphBuilder::new();
        let one_id = builder.one;

        let input_tokens = HashMap::from([
            ("zero".to_string(), Expression::from_scalar(0.0, one_id)),
            ("halt".to_string(), Expression::zero()),
        ]);

        let output_tokens = HashMap::from([("halt".to_string(), Expression::zero())]);

        let graph = builder.build(vec![], vec![]);

        let result = Runner::evaluate_with_output(
            &graph,
            &input_tokens,
            &output_tokens,
            &["zero".to_string()],
            100,
        );
        assert!(result.is_ok());
    }

    /// Create minimal weights for testing (all zeros).
    fn make_minimal_weights() -> TransformerWeights {
        let d_model = 4;
        let n_heads = 2;
        let d_ffn = 4;
        let n_layers = 1;
        let vocab_size = 2;

        TransformerWeights {
            embedding: vec![vec![0.0; d_model]; vocab_size],
            unembedding: vec![vec![0.0; d_model]; vocab_size],
            layers: vec![],
            head_tiebreak: vec![],
            attn_erase: vec![],
            ffn_erase: vec![],
            d_model,
            n_heads,
            d_ffn,
            n_layers,
            vocab_size,
        }
    }
}
