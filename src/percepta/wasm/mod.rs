// SPDX-License-Identifier: Apache-2.0
// Distilled from Percepta's `transformer-vm` (Apache-2.0 © Percepta).

//! WASM decoder + lowering passes + interpreter computation graph for transformer-vm.
//!
//! This module provides a pure-Rust WASM MVP binary decoder, a lowering
//! pipeline that replaces hard-to-simulate instructions (MUL, DIV, AND, OR,
//! XOR, SHL, SHR, ROTL, ROTR, CLZ, CTZ, POPCNT, EXTEND) with sequences of
//! basic instructions (ADD, SUB, comparisons, branches, local ops, LOAD/STORE)
//! that the transformer-vm can execute natively, and a WASM interpreter
//! expressed as a computation graph.
//!
//! # Module layout
//!
//! - [`decoder`]     — WASM MVP binary decoder (opcode + immediate parsing)
//! - [`lower`]       — Lower unsupported ops to supported sequences
//! - [`interpreter`] — WASM interpreter as computation graph (circle-point dispatch, byte-serial ALU)
//!
//! # Usage
//!
//! ```ignore
//! use percepta::wasm::{decode, lower_hard_ops, check_basic_only};
//!
//! let module = decode(&wasm_bytes)?;
//! for func in &module.functions {
//!     let lowered = lower_hard_ops(func, num_params);
//!     let bad = check_basic_only(&lowered);
//!     assert!(bad.is_empty(), "unsupported ops remain: {bad:?}");
//! }
//! ```
//!
//! # Interpreter (computation graph)
//!
//! ```ignore
//! use percepta::wasm::interpreter;
//! use percepta::graph::types::GraphBuilder;
//!
//! let mut builder = GraphBuilder::new();
//!
//! // Universal mode: generic interpreter with attention-based instruction fetch
//! let (input_tokens, output_tokens) = interpreter::build(None, &mut builder);
//!
//! // Specialized mode (Futamura projection): bake program into FFN weights
//! let program = vec![
//!     interpreter::ProgramInstruction::with_i32(interpreter::Opcode::I32Const, 42),
//!     interpreter::ProgramInstruction::new(interpreter::Opcode::Output),
//!     interpreter::ProgramInstruction::new(interpreter::Opcode::Halt),
//! ];
//! let (input_tokens, output_tokens) = interpreter::build(Some(&program), &mut builder);
//! ```

pub mod decoder;
pub mod interpreter;
pub mod lower;

// ── Re-exports from decoder ──────────────────────────────────

pub use decoder::{
    DataSegment,
    DecodeError,
    Export,
    ExportKind,
    FuncBody,
    FuncType,
    Global,
    Import,
    ImportKind,
    // Opcode constants
    OP_BLOCK,
    OP_BR,
    OP_BR_IF,
    OP_BR_TABLE,
    OP_CALL,
    OP_CALL_INDIRECT,
    OP_DROP,
    OP_ELSE,
    OP_END,
    OP_GLOBAL_GET,
    OP_GLOBAL_SET,
    OP_I32_ADD,
    OP_I32_AND,
    OP_I32_CLZ,
    OP_I32_CONST,
    OP_I32_CTZ,
    OP_I32_DIV_S,
    OP_I32_DIV_U,
    OP_I32_EQ,
    OP_I32_EQZ,
    OP_I32_EXTEND8_S,
    OP_I32_EXTEND16_S,
    OP_I32_GE_S,
    OP_I32_GE_U,
    OP_I32_GT_S,
    OP_I32_GT_U,
    OP_I32_LE_S,
    OP_I32_LE_U,
    OP_I32_LOAD,
    OP_I32_LOAD8_S,
    OP_I32_LOAD8_U,
    OP_I32_LOAD16_S,
    OP_I32_LOAD16_U,
    OP_I32_LT_S,
    OP_I32_LT_U,
    OP_I32_MUL,
    OP_I32_NE,
    OP_I32_OR,
    OP_I32_POPCNT,
    OP_I32_REM_S,
    OP_I32_REM_U,
    OP_I32_ROTL,
    OP_I32_ROTR,
    OP_I32_SHL,
    OP_I32_SHR_S,
    OP_I32_SHR_U,
    OP_I32_STORE,
    OP_I32_STORE8,
    OP_I32_STORE16,
    OP_I32_SUB,
    OP_I32_XOR,
    OP_IF,
    OP_LOCAL_GET,
    OP_LOCAL_SET,
    OP_LOCAL_TEE,
    OP_LOOP,
    OP_MEMORY_SIZE,
    OP_NOP,
    OP_RETURN,
    OP_SELECT,
    OP_UNREACHABLE,
    // Value type constants
    VALTYPE_I32,
    WASM_OP_NAMES,
    WasmInstr,
    WasmModule,
    decode,
    decode_instruction,
    read_signed_leb128,
    read_unsigned_leb128,
};

// ── Re-exports from lower ────────────────────────────────────

pub use lower::{
    BASIC_OPS, LOWERABLE_BINOPS, LOWERABLE_UNARY, SCRATCH_ADDR, check_basic_only, lower_hard_ops,
};

// ── Re-exports from interpreter ──────────────────────────────

pub use interpreter::{
    ByteAlu, CIRCLE_POINTS, ComparisonGates, EMIT_SCALE, EmitGates, FetchedState, InputDims,
    LOCAL_STRIDE, OPCODE_COUNT, Opcode, OpcodeDispatch, POINTS_R2, ProgramInstruction,
    build_input_tokens, build_output_tokens, get_byte_value, unique_stack_pairs,
};
