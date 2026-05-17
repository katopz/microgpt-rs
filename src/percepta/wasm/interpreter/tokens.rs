// SPDX-License-Identifier: Apache-2.0
// Distilled from Percepta's `transformer-vm` (Apache-2.0 © Percepta).

//! Input and output token vocabulary construction for the WASM interpreter.
//!
//! # Input Tokens
//!
//! Each input token maps to a linear expression over interpreter input dimensions.
//! The transformer selects one token per step; its embedding activates the
//! corresponding dimensions.
//!
//! Token categories:
//! - **Byte tokens** (`"00"`–`"ff"`, `"00'"`–`"ff'"`): raw byte values + carry
//! - **Commit tokens** (`"commit(+1,sts=1,bt=0)"`): state transition metadata
//! - **Output tokens** (`"out(A)"`, `"out(ff)"`): character output channel
//! - **Control tokens** (`"branch_taken"`, `"call_commit"`, `"return_commit"`)
//! - **Opcode tokens** (universal mode): embed instruction + dispatch info
//! - **Structural tokens** (`"{"`/`"}"` universal, `"start"` specialized)
//!
//! # Output Tokens
//!
//! Each output token maps to a scoring expression. The transformer predicts
//! the next token via argmax over these scores.
//!
//! Scoring uses a quadratic form: `(2·v)·signal − v²` which peaks at `v = signal`.
//! A large constant `H = 1e5` scales category-level gates to dominate.

use std::collections::HashMap;

use crate::percepta::graph::types::{DimId, Expression};

use super::dispatch::{Opcode, unique_stack_pairs};

// ── Scoring constant ───────────────────────────────────────────

/// Large scaling factor for category-level emit gates.
///
/// Must dominate the quadratic scoring terms so that the correct
/// token category is always selected first.
pub const EMIT_SCALE: f64 = 1e5;

// ── InputDims — input dimension expressions ────────────────────

/// Holds all input dimension expressions for token construction.
///
/// Created in the main `build()` function and passed to [`build_input_tokens`].
/// Universal-mode dimensions are `Some`; specialized-mode dimensions are `None`.
pub struct InputDims {
    /// Byte value channel: coefficient = (byte_value + 1).
    pub byte_number: Expression,
    /// Carry flag channel: coefficient = 0 or 1.
    pub carry: Expression,
    /// Cursor movement channel.
    pub delta_cursor: Expression,
    /// Stack depth change channel.
    pub delta_stack: Expression,
    /// Jump/branch indicator channel.
    pub is_jump: Expression,
    /// Store-to-stack indicator channel.
    pub store_to_stack: Expression,
    /// Branch-taken flag channel.
    pub is_branch_taken: Expression,
    /// Call depth change channel.
    pub delta_call_depth: Expression,
    /// Return commit flag channel.
    pub is_return_commit: Expression,
    /// Scalar unit dimension.
    pub one: Expression,

    // ── Universal-mode only ──
    /// Stack delta prefix channel (universal mode).
    pub delta_stack_prefix: Option<Expression>,
    /// Store-to-stack prefix channel (universal mode).
    pub store_to_stack_prefix: Option<Expression>,
    /// Opcode x-coordinate channel (universal mode).
    pub opcode_x: Option<Expression>,
    /// Opcode y-coordinate channel (universal mode).
    pub opcode_y: Option<Expression>,
    /// Write-operation indicator channel (universal mode).
    pub is_write: Option<Expression>,
}

// ── EmitGates — emit gate expressions ──────────────────────────

/// Holds all emit gate expressions for output token scoring.
///
/// Each gate is approximately 1 when that token category should be emitted
/// and approximately 0 otherwise.
pub struct EmitGates {
    /// Emit the `"halt"` token.
    pub emit_halt: Expression,
    /// Emit the `"branch_taken"` token.
    pub emit_branch_taken: Expression,
    /// Emit the `"call_commit"` token.
    pub emit_call_commit: Expression,
    /// Emit the `"return_commit"` token.
    pub emit_return_commit: Expression,
    /// Emit an `"out(...)"` token.
    pub emit_out: Expression,
    /// Emit a byte token (`"00"`–`"ff'"`).
    pub emit_byte: Expression,
    /// Emit a `"commit(...)"` token.
    pub emit_commit: Expression,
    /// Emit branch-taken byte (for br/br_if completion).
    pub emit_bt: Expression,
}

// ── FetchedState — fetched instruction properties ──────────────

/// Holds fetched instruction state for output token scoring.
pub struct FetchedState {
    /// Fetched stack delta from the current instruction.
    pub fetched_stack_delta: Expression,
    /// Fetched store-to-stack flag from the current instruction.
    pub fetched_store_to_stack: Expression,
}

// ── build_input_tokens ─────────────────────────────────────────

/// Build all input tokens for the WASM interpreter.
///
/// Returns a `HashMap<String, Expression>` mapping token names to their
/// embedding expressions.
///
/// # Arguments
///
/// * `dims` — input dimension expressions
/// * `specialized` — `true` for Futamura mode, `false` for universal
/// * `one_id` — the `DimId` of the `one` dimension (for adding `one=1`)
pub fn build_input_tokens(
    dims: &InputDims,
    specialized: bool,
    one_id: DimId,
) -> HashMap<String, Expression> {
    let mut tokens = HashMap::new();

    // ── Byte tokens: "00"–"ff" (carry=0) and "00'"–"ff'" (carry=1) ──
    for bv in 0u32..256 {
        for c in 0u32..2 {
            let name = if c == 1 {
                format!("{bv:02x}'")
            } else {
                format!("{bv:02x}")
            };
            let expr =
                dims.byte_number.clone() * ((bv as f64) + 1.0) + dims.carry.clone() * (c as f64);
            tokens.insert(name, expr);
        }
    }

    // ── Commit tokens: unique (stack_delta, is_sts) × branch_taken ──
    let pairs = unique_stack_pairs();
    for (sd, sts) in &pairs {
        for bt in 0u32..2 {
            let name = format!("commit({sd:+},sts={sts},bt={bt})");
            let expr = dims.delta_cursor.clone()
                + dims.delta_stack.clone() * (*sd as f64)
                + dims.store_to_stack.clone() * (*sts as f64)
                + dims.is_jump.clone() * (bt as f64);
            tokens.insert(name, expr);
        }
    }

    // ── Output tokens: "out(A)" for printable ASCII, "out(ff)" otherwise ──
    for bv in 0u32..256 {
        let name = match bv {
            0x21..=0x7E => format!("out({})", char::from_u32(bv).unwrap()),
            _ => format!("out({bv:02x})"),
        };
        // out tokens contribute delta_cursor (trigger cursor advance)
        tokens.insert(name, dims.delta_cursor.clone());
    }

    // ── Special control tokens ──
    tokens.insert("branch_taken".to_string(), dims.is_branch_taken.clone());
    tokens.insert(
        "call_commit".to_string(),
        dims.delta_cursor.clone() + dims.delta_call_depth.clone() + dims.is_jump.clone(),
    );
    tokens.insert(
        "return_commit".to_string(),
        dims.delta_cursor.clone() - dims.delta_call_depth.clone()
            + dims.is_return_commit.clone()
            + dims.is_jump.clone(),
    );

    // ── Mode-specific tokens ──
    if !specialized {
        // Universal mode: structural + opcode embedding tokens
        tokens.insert("{".to_string(), Expression::zero());
        tokens.insert("}".to_string(), dims.delta_stack.clone() * 3.0);

        let opcode_x = dims
            .opcode_x
            .as_ref()
            .expect("opcode_x required for universal mode");
        let opcode_y = dims
            .opcode_y
            .as_ref()
            .expect("opcode_y required for universal mode");
        let dsp = dims
            .delta_stack_prefix
            .as_ref()
            .expect("delta_stack_prefix required for universal mode");
        let stsp = dims
            .store_to_stack_prefix
            .as_ref()
            .expect("store_to_stack_prefix required for universal mode");
        let is_write = dims
            .is_write
            .as_ref()
            .expect("is_write required for universal mode");

        for op in Opcode::ALL {
            let (px, py) = op.circle_point();
            let sd = op.stack_delta();
            let sts: i32 = if op.is_sts() { 1 } else { 0 };
            let iw: i32 = if op.is_write() { 1 } else { 0 };

            let embedding = opcode_x.clone() * (px as f64)
                + opcode_y.clone() * (py as f64)
                + dsp.clone() * (sd as f64)
                + stsp.clone() * (sts as f64)
                + is_write.clone() * (iw as f64);

            tokens.insert(op.as_str().to_string(), embedding);
        }
    } else {
        // Specialized (Futamura) mode: only start token
        tokens.insert("start".to_string(), dims.delta_stack.clone() * 3.0);
    }

    // ── Printable ASCII aliases: 'A' -> same as '41', etc. ──
    for bv in 0x21u32..0x7F {
        let ch = char::from_u32(bv).unwrap();
        let ch_str = ch.to_string();
        if tokens.contains_key(&ch_str) {
            continue;
        }
        let hex_name = format!("{bv:02x}");
        if let Some(hex_expr) = tokens.get(&hex_name) {
            tokens.insert(ch_str, hex_expr.clone());
        }
    }

    // ── Add one=1 to all tokens except the start token ──
    // The start token has zero embedding (no previous context).
    let start_token = if specialized { "start" } else { "{" };
    for (name, expr) in tokens.iter_mut() {
        if name.as_str() != start_token {
            expr.set(one_id, 1.0);
        }
    }

    tokens
}

// ── build_output_tokens ────────────────────────────────────────

/// Build all output tokens (scoring expressions for argmax prediction).
///
/// Each output token maps to a scoring expression. The transformer predicts
/// the next token by computing all scores and selecting the argmax.
///
/// # Scoring formula
///
/// The quadratic scoring form `(2·v)·signal − v²` peaks at `v = signal`:
/// - For byte tokens: `score = H·emit_byte + (2·bv)·result_byte − bv² + (2·c)·result_carry − c²`
/// - For commit tokens: `score = H·emit_commit + (2·sd)·fetched_sd − sd² + ...`
/// - For output tokens: `score = H·emit_out + (2·bv)·top_byte − bv²`
///
/// # Arguments
///
/// * `gates` — emit gate expressions
/// * `state` — fetched instruction properties
/// * `result_byte` — multiplexed ALU result byte (gated by opcode)
/// * `result_carry` — multiplexed ALU result carry (gated by opcode)
/// * `top_byte` — current byte of top stack value (for output scoring)
pub fn build_output_tokens(
    gates: &EmitGates,
    state: &FetchedState,
    result_byte: &Expression,
    result_carry: &Expression,
    top_byte: &Expression,
) -> HashMap<String, Expression> {
    let h = EMIT_SCALE;
    let mut tokens = HashMap::new();

    // ── Control flow tokens ──
    tokens.insert("halt".to_string(), gates.emit_halt.clone() * h);
    tokens.insert(
        "branch_taken".to_string(),
        gates.emit_branch_taken.clone() * h,
    );
    tokens.insert(
        "call_commit".to_string(),
        gates.emit_call_commit.clone() * h,
    );
    tokens.insert(
        "return_commit".to_string(),
        gates.emit_return_commit.clone() * h,
    );

    // ── Output tokens: score by matching top_byte ──
    for bv in 0u32..256 {
        let name = match bv {
            0x21..=0x7E => format!("out({})", char::from_u32(bv).unwrap()),
            _ => format!("out({bv:02x})"),
        };
        let bvf = bv as f64;
        let score = gates.emit_out.clone() * h + top_byte.clone() * (2.0 * bvf) - bvf * bvf;
        tokens.insert(name, score);
    }

    // ── Commit tokens: score by matching fetched instruction properties ──
    let pairs = unique_stack_pairs();
    for (sd, sts) in &pairs {
        for bt in 0u32..2 {
            let name = format!("commit({sd:+},sts={sts},bt={bt})");
            let sdf = *sd as f64;
            let stsf = *sts as f64;
            let btf = bt as f64;
            let score = gates.emit_commit.clone() * h
                + state.fetched_stack_delta.clone() * (2.0 * sdf)
                - sdf * sdf
                + state.fetched_store_to_stack.clone() * (2.0 * stsf)
                - stsf * stsf
                + gates.emit_bt.clone() * (2.0 * btf)
                - btf * btf;
            tokens.insert(name, score);
        }
    }

    // ── Byte tokens with carry: score by matching result_byte and result_carry ──
    for bv in 0u32..256 {
        let bvf = bv as f64;
        let bv_base = gates.emit_byte.clone() * h + result_byte.clone() * (2.0 * bvf) - bvf * bvf;
        for c in 0u32..2 {
            let name = if c == 1 {
                format!("{bv:02x}'")
            } else {
                format!("{bv:02x}")
            };
            let cf = c as f64;
            let score = bv_base.clone() + result_carry.clone() * (2.0 * cf) - cf * cf;
            tokens.insert(name, score);
        }
    }

    tokens
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::percepta::graph::types::GraphBuilder;

    /// Helper: create InputDims for universal mode testing.
    fn make_universal_dims(builder: &mut GraphBuilder) -> (InputDims, DimId) {
        let one_id = builder.one;
        let one_expr = Expression::from_dim(one_id);

        let dims = InputDims {
            byte_number: builder.generic("byte_number"),
            carry: builder.generic("carry"),
            delta_cursor: builder.generic("delta_cursor"),
            delta_stack: builder.generic("delta_stack"),
            is_jump: builder.generic("is_jump"),
            store_to_stack: builder.generic("store_to_stack"),
            is_branch_taken: builder.generic("is_branch_taken"),
            delta_call_depth: builder.generic("delta_call_depth"),
            is_return_commit: builder.generic("is_return_commit"),
            one: one_expr,
            delta_stack_prefix: Some(builder.generic("delta_stack_prefix")),
            store_to_stack_prefix: Some(builder.generic("store_to_stack_prefix")),
            opcode_x: Some(builder.generic("opcode_x")),
            opcode_y: Some(builder.generic("opcode_y")),
            is_write: Some(builder.generic("is_write")),
        };

        (dims, one_id)
    }

    /// Helper: create InputDims for specialized mode testing.
    fn make_specialized_dims(builder: &mut GraphBuilder) -> (InputDims, DimId) {
        let one_id = builder.one;
        let one_expr = Expression::from_dim(one_id);

        let dims = InputDims {
            byte_number: builder.generic("byte_number"),
            carry: builder.generic("carry"),
            delta_cursor: builder.generic("delta_cursor"),
            delta_stack: builder.generic("delta_stack"),
            is_jump: builder.generic("is_jump"),
            store_to_stack: builder.generic("store_to_stack"),
            is_branch_taken: builder.generic("is_branch_taken"),
            delta_call_depth: builder.generic("delta_call_depth"),
            is_return_commit: builder.generic("is_return_commit"),
            one: one_expr,
            delta_stack_prefix: None,
            store_to_stack_prefix: None,
            opcode_x: None,
            opcode_y: None,
            is_write: None,
        };

        (dims, one_id)
    }

    #[test]
    fn test_input_tokens_universal_has_all_byte_tokens() {
        let mut builder = GraphBuilder::new();
        let (dims, one_id) = make_universal_dims(&mut builder);

        let tokens = build_input_tokens(&dims, false, one_id);

        // Should have byte tokens for all 256 values × 2 carry states
        for bv in 0u32..256 {
            assert!(
                tokens.contains_key(&format!("{bv:02x}")),
                "missing {bv:02x}"
            );
            assert!(
                tokens.contains_key(&format!("{bv:02x}'")),
                "missing {bv:02x}'"
            );
        }
    }

    #[test]
    fn test_input_tokens_universal_has_commit_tokens() {
        let mut builder = GraphBuilder::new();
        let (dims, one_id) = make_universal_dims(&mut builder);

        let tokens = build_input_tokens(&dims, false, one_id);

        // Check some specific commit tokens
        assert!(tokens.contains_key("commit(+1,sts=1,bt=0)"));
        assert!(tokens.contains_key("commit(+1,sts=1,bt=1)"));
        assert!(tokens.contains_key("commit(-1,sts=0,bt=0)"));
        assert!(tokens.contains_key("commit(-2,sts=0,bt=0)"));
    }

    #[test]
    fn test_input_tokens_universal_has_opcode_tokens() {
        let mut builder = GraphBuilder::new();
        let (dims, one_id) = make_universal_dims(&mut builder);

        let tokens = build_input_tokens(&dims, false, one_id);

        // Should have all opcode tokens
        for op in Opcode::ALL {
            assert!(
                tokens.contains_key(op.as_str()),
                "missing opcode {}",
                op.as_str()
            );
        }

        // Structural tokens
        assert!(tokens.contains_key("{"));
        assert!(tokens.contains_key("}"));
    }

    #[test]
    fn test_input_tokens_specialized_has_start() {
        let mut builder = GraphBuilder::new();
        let (dims, one_id) = make_specialized_dims(&mut builder);

        let tokens = build_input_tokens(&dims, true, one_id);

        assert!(tokens.contains_key("start"));
        // "{" and "}" exist as printable ASCII aliases (0x7B, 0x7D) for byte tokens,
        // but NOT as structural tokens with special zero/3*delta_stack embeddings.
        assert!(tokens.contains_key("{"));
        assert!(tokens.contains_key("}"));
    }

    #[test]
    fn test_input_tokens_universal_start_has_no_one() {
        let mut builder = GraphBuilder::new();
        let (dims, one_id) = make_universal_dims(&mut builder);

        let tokens = build_input_tokens(&dims, false, one_id);

        // The "{" start token should NOT have one=1
        let brace = tokens.get("{").expect("missing { token");
        assert_eq!(
            brace.get(one_id),
            0.0,
            "start token {{ should not have one=1"
        );

        // Non-start tokens should have one=1
        let byte_00 = tokens.get("00").expect("missing 00 token");
        assert_eq!(byte_00.get(one_id), 1.0, "00 should have one=1");
    }

    #[test]
    fn test_input_tokens_specialized_start_has_no_one() {
        let mut builder = GraphBuilder::new();
        let (dims, one_id) = make_specialized_dims(&mut builder);

        let tokens = build_input_tokens(&dims, true, one_id);

        let start = tokens.get("start").expect("missing start token");
        assert_eq!(start.get(one_id), 0.0, "start token should not have one=1");

        let byte_00 = tokens.get("00").expect("missing 00 token");
        assert_eq!(byte_00.get(one_id), 1.0, "00 should have one=1");
    }

    #[test]
    fn test_input_tokens_printable_ascii_aliases() {
        let mut builder = GraphBuilder::new();
        let (dims, one_id) = make_universal_dims(&mut builder);

        let tokens = build_input_tokens(&dims, false, one_id);

        // 'A' (0x41) should be aliased to the same embedding as "41"
        assert!(tokens.contains_key("A"));
        let a_expr = tokens.get("A").expect("missing A alias");
        let hex_expr = tokens.get("41").expect("missing 41");
        assert_eq!(a_expr, hex_expr, "A should alias 41");
    }

    #[test]
    fn test_output_tokens_has_all_categories() {
        let mut builder = GraphBuilder::new();
        let one_expr = Expression::from_dim(builder.one);

        let gates = EmitGates {
            emit_halt: one_expr.clone(),
            emit_branch_taken: one_expr.clone(),
            emit_call_commit: one_expr.clone(),
            emit_return_commit: one_expr.clone(),
            emit_out: one_expr.clone(),
            emit_byte: one_expr.clone(),
            emit_commit: one_expr.clone(),
            emit_bt: one_expr.clone(),
        };

        let state = FetchedState {
            fetched_stack_delta: one_expr.clone(),
            fetched_store_to_stack: one_expr.clone(),
        };

        let result_byte = builder.generic("result_byte");
        let result_carry = builder.generic("result_carry");
        let top_byte = builder.generic("top_byte");

        let tokens = build_output_tokens(&gates, &state, &result_byte, &result_carry, &top_byte);

        // Control tokens
        assert!(tokens.contains_key("halt"));
        assert!(tokens.contains_key("branch_taken"));
        assert!(tokens.contains_key("call_commit"));
        assert!(tokens.contains_key("return_commit"));

        // Byte tokens
        assert!(tokens.contains_key("00"));
        assert!(tokens.contains_key("ff"));
        assert!(tokens.contains_key("00'"));
        assert!(tokens.contains_key("ff'"));

        // Output tokens
        assert!(tokens.contains_key("out(00)"));
        assert!(tokens.contains_key("out(A)"));

        // Commit tokens
        assert!(tokens.contains_key("commit(+1,sts=1,bt=0)"));
    }

    #[test]
    fn test_output_tokens_byte_scoring_is_nonzero() {
        let mut builder = GraphBuilder::new();
        let one_expr = Expression::from_dim(builder.one);

        let gates = EmitGates {
            emit_halt: one_expr.clone(),
            emit_branch_taken: one_expr.clone(),
            emit_call_commit: one_expr.clone(),
            emit_return_commit: one_expr.clone(),
            emit_out: one_expr.clone(),
            emit_byte: one_expr.clone(),
            emit_commit: one_expr.clone(),
            emit_bt: one_expr.clone(),
        };

        let state = FetchedState {
            fetched_stack_delta: one_expr.clone(),
            fetched_store_to_stack: one_expr.clone(),
        };

        let result_byte = builder.generic("result_byte");
        let result_carry = builder.generic("result_carry");
        let top_byte = builder.generic("top_byte");

        let tokens = build_output_tokens(&gates, &state, &result_byte, &result_carry, &top_byte);

        // All byte tokens should have non-zero scoring
        for bv in 0u32..256 {
            let name = format!("{bv:02x}");
            let score = tokens
                .get(&name)
                .unwrap_or_else(|| panic!("missing {name}"));
            assert!(
                !score.is_zero(),
                "byte token {name} should have non-zero score"
            );
        }
    }

    #[test]
    fn test_emit_scale_constant() {
        // Verify the scoring constant is what we expect
        assert!((EMIT_SCALE - 1e5).abs() < 1e-10);
    }
}
