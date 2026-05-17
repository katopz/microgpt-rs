// SPDX-License-Identifier: Apache-2.0
// Distilled from Percepta's `transformer-vm` (Apache-2.0 © Percepta).

//! WASM interpreter as a computation graph.
//!
//! Constructs a computation graph where each "step" of the transformer
//! executes one instruction of a WASM program. Uses circle-point opcode
//! dispatch, byte-serial arithmetic, and attention-based state tracking.
//!
//! # Modes
//!
//! - **Universal** (`program = None`): Generic interpreter with attention-based instruction fetch.
//! - **Specialized** (`program = Some`): Futamura projection — bake program into FFN weights.

pub mod arithmetic;
pub mod dispatch;
pub mod tokens;

use std::collections::HashMap;

use crate::percepta::TieBreak;
use crate::percepta::graph::types::{Expression, GraphBuilder};

// ── Re-exports (also serve as private imports within this module) ──

pub use arithmetic::{ByteArithmetic as ByteAlu, Comparisons as ComparisonGates, get_byte_value};
pub use dispatch::{
    CIRCLE_POINTS, LOCAL_STRIDE, OPCODE_COUNT, Opcode, OpcodeDispatch, POINTS_R2,
    ProgramInstruction, unique_stack_pairs,
};
pub use tokens::{
    EMIT_SCALE, EmitGates, FetchedState, InputDims, build_input_tokens, build_output_tokens,
};

// ── DispatchCache ──────────────────────────────────────────────

/// Pre-computed opcode dispatch values to avoid double mutable borrows.
///
/// All `op_dot` and `is_op` values are computed eagerly before any
/// `builder` calls that use them.
struct DispatchCache {
    op_dot: Vec<Expression>,
    is_op: Vec<Expression>,
}

impl DispatchCache {
    fn build(dispatch: &mut OpcodeDispatch, builder: &mut GraphBuilder) -> Self {
        let n = Opcode::ALL.len();
        let mut op_dot = Vec::with_capacity(n);
        let mut is_op = Vec::with_capacity(n);

        for op in Opcode::ALL {
            op_dot.push(dispatch.op_dot(op));
            is_op.push(dispatch.is_op(builder, op));
        }

        Self { op_dot, is_op }
    }

    #[inline]
    fn dot(&self, op: Opcode) -> &Expression {
        &self.op_dot[op as usize]
    }

    #[inline]
    fn gate(&self, op: Opcode) -> &Expression {
        &self.is_op[op as usize]
    }
}

// ── FetchResult ────────────────────────────────────────────────

struct FetchResult {
    fetched_opcode_x: Expression,
    fetched_opcode_y: Expression,
    fetched_stack_delta: Expression,
    fetched_store_to_stack: Expression,
    fetched_is_write: Expression,
    immediate: Expression,
    const_byte: Expression,
}

// ── PiecewiseLookup ────────────────────────────────────────────

/// Piecewise-constant FFN lookup for specialized (Futamura) mode.
struct PiecewiseLookup {
    r_pos: Vec<Expression>,
    r_neg: Vec<Expression>,
    one_expr: Expression,
}

impl PiecewiseLookup {
    fn new(
        builder: &mut GraphBuilder,
        n_instr: usize,
        cursor: &Expression,
        one_expr: &Expression,
    ) -> Self {
        let capacity = n_instr.saturating_sub(1);
        let mut r_pos = Vec::with_capacity(capacity);
        let mut r_neg = Vec::with_capacity(capacity);
        for i in 1..n_instr {
            let fi = i as f64;
            r_pos.push(builder.reglu(one_expr.clone(), cursor.clone() - (fi - 1.0)));
            r_neg.push(builder.reglu(one_expr.clone(), cursor.clone() - fi));
        }
        Self {
            r_pos,
            r_neg,
            one_expr: one_expr.clone(),
        }
    }

    fn lookup(&self, builder: &mut GraphBuilder, values: &[f64]) -> Expression {
        let mut expr = self.one_expr.clone() * values[0];
        for i in 1..values.len() {
            let diff = values[i] - values[i - 1];
            if diff == 0.0 {
                continue;
            }
            expr = expr + self.r_pos[i - 1].clone() * diff - self.r_neg[i - 1].clone() * diff;
        }
        builder.persist(expr)
    }
}

// ── build_universal_fetch ──────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn build_universal_fetch(
    builder: &mut GraphBuilder,
    cursor: &Expression,
    position: &Expression,
    byte_number_m1: &Expression,
    byte_index: &Expression,
    opcode_x: &Expression,
    opcode_y: &Expression,
    delta_stack_prefix: &Expression,
    store_to_stack_prefix: &Expression,
    is_write: &Expression,
) -> FetchResult {
    let one_expr = Expression::from_dim(builder.one);
    let instruction_position = cursor.clone() * 5.0 + one_expr;

    let fetched = builder.fetch_vec(
        vec![
            opcode_x.clone(),
            opcode_y.clone(),
            delta_stack_prefix.clone(),
            store_to_stack_prefix.clone(),
            is_write.clone(),
        ],
        Some(instruction_position.clone()),
        Some(position.clone()),
        None,
        TieBreak::Latest,
    );

    let mut immediate = Expression::zero();
    for i in 1..=4usize {
        let query = instruction_position.clone() + (i as f64);
        let byte = builder.fetch(
            byte_number_m1.clone(),
            Some(query),
            Some(position.clone()),
            None,
            TieBreak::Latest,
        );
        immediate = immediate + byte * ((1u64 << (8 * (i - 1))) as f64);
    }
    let immediate = builder.persist(immediate);

    let cb_query = instruction_position + byte_index.clone() + 1.0;
    let const_byte = builder.fetch(
        byte_number_m1.clone(),
        Some(cb_query),
        Some(position.clone()),
        None,
        TieBreak::Latest,
    );

    FetchResult {
        fetched_opcode_x: fetched[0].clone(),
        fetched_opcode_y: fetched[1].clone(),
        fetched_stack_delta: fetched[2].clone(),
        fetched_store_to_stack: fetched[3].clone(),
        fetched_is_write: fetched[4].clone(),
        immediate,
        const_byte,
    }
}

// ── build_specialized_fetch ────────────────────────────────────

fn build_specialized_fetch(
    builder: &mut GraphBuilder,
    program: &[ProgramInstruction],
    cursor: &Expression,
    byte_index: &Expression,
) -> FetchResult {
    let one_expr = Expression::from_dim(builder.one);
    let n_instr = program.len();
    let lookup = PiecewiseLookup::new(builder, n_instr, cursor, &one_expr);

    let opx: Vec<f64> = program
        .iter()
        .map(|ins| ins.opcode.circle_point().0 as f64)
        .collect();
    let opy: Vec<f64> = program
        .iter()
        .map(|ins| ins.opcode.circle_point().1 as f64)
        .collect();
    let sd: Vec<f64> = program
        .iter()
        .map(|ins| ins.opcode.stack_delta() as f64)
        .collect();
    let sts: Vec<f64> = program
        .iter()
        .map(|ins| if ins.opcode.is_sts() { 1.0 } else { 0.0 })
        .collect();
    let iw: Vec<f64> = program
        .iter()
        .map(|ins| if ins.opcode.is_write() { 1.0 } else { 0.0 })
        .collect();
    let imm: Vec<f64> = program
        .iter()
        .map(|ins| ins.immediate_u32() as f64)
        .collect();

    let fetched_opcode_x = lookup.lookup(builder, &opx);
    let fetched_opcode_y = lookup.lookup(builder, &opy);
    let fetched_stack_delta = lookup.lookup(builder, &sd);
    let fetched_store_to_stack = lookup.lookup(builder, &sts);
    let fetched_is_write = lookup.lookup(builder, &iw);
    let immediate = lookup.lookup(builder, &imm);

    let cb: Vec<Expression> = (0..4)
        .map(|b| {
            let bvals: Vec<f64> = program.iter().map(|ins| ins.bytes[b] as f64).collect();
            lookup.lookup(builder, &bvals)
        })
        .collect();

    let mut const_byte = cb[0].clone();
    for b in 1..4 {
        let diff = cb[b].clone() - cb[b - 1].clone();
        let sel = builder.stepglu(diff, byte_index.clone() - (b as f64));
        const_byte = const_byte + sel;
    }

    FetchResult {
        fetched_opcode_x,
        fetched_opcode_y,
        fetched_stack_delta,
        fetched_store_to_stack,
        fetched_is_write,
        immediate,
        const_byte,
    }
}

// ════════════════════════════════════════════════════════════════
//  Main build function
// ════════════════════════════════════════════════════════════════

/// Build the WASM interpreter computation graph.
///
/// Returns `(input_tokens, output_tokens)` mapping token names to
/// their embedding/scoring expressions.
pub fn build(
    program: Option<&[ProgramInstruction]>,
    builder: &mut GraphBuilder,
) -> (HashMap<String, Expression>, HashMap<String, Expression>) {
    let specialized = program.is_some();
    let one = Expression::from_dim(builder.one);
    let position = Expression::from_dim(builder.position);
    let mut names: Vec<(String, Expression)> = Vec::new();

    // ════════════════════════════════════════════════════════════
    // §1. Input dimensions
    // ════════════════════════════════════════════════════════════

    let byte_number = builder.generic("byte_number");
    let carry = builder.generic("carry");
    let delta_cursor = builder.generic("delta_cursor");
    let delta_stack = builder.generic("delta_stack");
    let is_jump = builder.generic("is_jump");
    let store_to_stack = builder.generic("store_to_stack");
    let is_branch_taken = builder.generic("is_branch_taken");
    let delta_call_depth = builder.generic("delta_call_depth");
    let is_return_commit = builder.generic("is_return_commit");

    names.extend([
        ("byte_number".into(), byte_number.clone()),
        ("carry".into(), carry.clone()),
        ("delta_cursor".into(), delta_cursor.clone()),
        ("delta_stack".into(), delta_stack.clone()),
        ("is_jump".into(), is_jump.clone()),
        ("store_to_stack".into(), store_to_stack.clone()),
        ("is_branch_taken".into(), is_branch_taken.clone()),
        ("delta_call_depth".into(), delta_call_depth.clone()),
        ("is_return_commit".into(), is_return_commit.clone()),
    ]);

    let (dsp, stsp, ox, oy, iw) = if !specialized {
        let dsp = builder.generic("delta_stack_prefix");
        let stsp = builder.generic("store_to_stack_prefix");
        let ox = builder.generic("opcode_x");
        let oy = builder.generic("opcode_y");
        let iw = builder.generic("is_write");
        names.extend([
            ("delta_stack_prefix".into(), dsp.clone()),
            ("store_to_stack_prefix".into(), stsp.clone()),
            ("opcode_x".into(), ox.clone()),
            ("opcode_y".into(), oy.clone()),
            ("is_write".into(), iw.clone()),
        ]);
        (Some(dsp), Some(stsp), Some(ox), Some(oy), Some(iw))
    } else {
        (None, None, None, None, None)
    };

    // ════════════════════════════════════════════════════════════
    // §2. Input tokens
    // ════════════════════════════════════════════════════════════

    let input_dims = InputDims {
        byte_number: byte_number.clone(),
        carry: carry.clone(),
        delta_cursor: delta_cursor.clone(),
        delta_stack: delta_stack.clone(),
        is_jump: is_jump.clone(),
        store_to_stack: store_to_stack.clone(),
        is_branch_taken: is_branch_taken.clone(),
        delta_call_depth: delta_call_depth.clone(),
        is_return_commit: is_return_commit.clone(),
        one: one.clone(),
        delta_stack_prefix: dsp.clone(),
        store_to_stack_prefix: stsp.clone(),
        opcode_x: ox.clone(),
        opcode_y: oy.clone(),
        is_write: iw.clone(),
    };

    let input_tokens = build_input_tokens(&input_dims, specialized, builder.one);

    // ════════════════════════════════════════════════════════════
    // §3. Store value (4-byte reconstruction)
    // ════════════════════════════════════════════════════════════

    let byte_number_m1 = byte_number - 1.0;

    let store_bytes: Vec<Expression> = (1..=4i32)
        .map(|i| {
            builder.fetch(
                byte_number_m1.clone(),
                Some(position.clone() - (i as f64)),
                Some(position.clone()),
                None,
                TieBreak::Latest,
            )
        })
        .collect();

    let mut store_value_expr = Expression::zero();
    for i in 1..=4usize {
        let shift = 8 * (4 - i);
        store_value_expr = store_value_expr + store_bytes[i - 1].clone() * ((1u64 << shift) as f64);
    }
    let store_value = builder.persist(store_value_expr);
    let msb = store_bytes[0].clone();

    names.push(("store_value".into(), store_value.clone()));
    names.push(("msb".into(), msb.clone()));

    // ════════════════════════════════════════════════════════════
    // §4. Branch offset computation
    // ════════════════════════════════════════════════════════════

    let unsigned_branch = builder.reglu(store_value.clone(), is_jump.clone());
    let jump_sign = builder.stepglu(one.clone(), msb + is_jump.clone() * 128.0 - 256.0);
    let delta_cursor_expr =
        delta_cursor + unsigned_branch.clone() - jump_sign.clone() * ((1u64 << 32) as f64);

    names.push(("unsigned_branch".into(), unsigned_branch));
    names.push(("jump_sign".into(), jump_sign));
    names.push(("delta_cursor_expr".into(), delta_cursor_expr.clone()));

    // ════════════════════════════════════════════════════════════
    // §5. Byte index and boundary detection
    // ════════════════════════════════════════════════════════════

    let start_offset = builder.fetch(
        position.clone(),
        Some(one.clone()),
        Some(one.clone()),
        Some(byte_number_m1.clone()),
        TieBreak::Latest,
    );
    let byte_index = position - start_offset;
    let is_boundary = builder.stepglu(one.clone(), -byte_number_m1.clone());

    names.push(("byte_index".into(), byte_index.clone()));
    names.push(("is_boundary".into(), is_boundary.clone()));

    // ════════════════════════════════════════════════════════════
    // §6. Cumulative state (stack_depth, cursor, call_depth)
    // ════════════════════════════════════════════════════════════

    let cum_state = builder.fetch_sum(vec![
        delta_stack.clone(),
        delta_cursor_expr,
        delta_call_depth.clone(),
    ]);
    let stack_depth = cum_state[0].clone();
    let cursor = cum_state[1].clone();
    let call_depth = cum_state[2].clone();

    names.push(("stack_depth".into(), stack_depth.clone()));
    names.push(("cursor".into(), cursor.clone()));
    names.push(("call_depth".into(), call_depth.clone()));

    // ════════════════════════════════════════════════════════════
    // §7. Instruction fetch
    // ════════════════════════════════════════════════════════════

    let fetch_result = match program {
        Some(prog) => build_specialized_fetch(builder, prog, &cursor, &byte_index),
        None => build_universal_fetch(
            builder,
            &cursor,
            &Expression::from_dim(builder.position),
            &byte_number_m1,
            &byte_index,
            ox.as_ref().unwrap(),
            oy.as_ref().unwrap(),
            dsp.as_ref().unwrap(),
            stsp.as_ref().unwrap(),
            iw.as_ref().unwrap(),
        ),
    };

    names.extend([
        (
            "fetched_opcode_x".into(),
            fetch_result.fetched_opcode_x.clone(),
        ),
        (
            "fetched_opcode_y".into(),
            fetch_result.fetched_opcode_y.clone(),
        ),
        (
            "fetched_stack_delta".into(),
            fetch_result.fetched_stack_delta.clone(),
        ),
        (
            "fetched_store_to_stack".into(),
            fetch_result.fetched_store_to_stack.clone(),
        ),
        (
            "fetched_is_write".into(),
            fetch_result.fetched_is_write.clone(),
        ),
        ("immediate".into(), fetch_result.immediate.clone()),
        ("const_byte".into(), fetch_result.const_byte.clone()),
    ]);

    // ════════════════════════════════════════════════════════════
    // §8. Opcode dispatch — pre-compute all values
    // ════════════════════════════════════════════════════════════

    let mut dispatch = OpcodeDispatch::new(
        fetch_result.fetched_opcode_x,
        fetch_result.fetched_opcode_y,
        one.clone(),
        specialized,
    );

    let dc = DispatchCache::build(&mut dispatch, builder);

    // ════════════════════════════════════════════════════════════
    // §9. Derived instruction properties
    // ════════════════════════════════════════════════════════════

    let is_output = dc.gate(Opcode::Output).clone();

    let is_store_sum = dc.gate(Opcode::I32Store).clone()
        + dc.gate(Opcode::I32Store8).clone()
        + dc.gate(Opcode::I32Store16).clone()
        + dc.gate(Opcode::InputBase).clone();
    let memory_write_gate = builder.persist(is_store_sum);

    let uses_top_byte = fetch_result.fetched_is_write.clone() + is_output;
    let is_producing_bytes =
        fetch_result.fetched_store_to_stack.clone() + fetch_result.fetched_is_write.clone();

    names.push(("memory_write_gate".into(), memory_write_gate.clone()));
    names.push(("uses_top_byte".into(), uses_top_byte.clone()));
    names.push(("is_producing_bytes".into(), is_producing_bytes.clone()));

    // ════════════════════════════════════════════════════════════
    // §10. Local variables
    // ════════════════════════════════════════════════════════════

    let local_write_key_dim = call_depth.clone() * (LOCAL_STRIDE as f64)
        + fetch_result.immediate.clone() * 4.0
        + byte_index.clone();

    let not_local_write =
        one.clone() - dc.gate(Opcode::LocalSet).clone() - dc.gate(Opcode::LocalTee).clone()
            + is_boundary.clone();

    let local_byte = builder.fetch(
        byte_number_m1.clone(),
        Some(local_write_key_dim.clone() + 1.0),
        Some(local_write_key_dim),
        Some(not_local_write),
        TieBreak::Latest,
    );

    names.push(("local_byte".into(), local_byte.clone()));

    // ════════════════════════════════════════════════════════════
    // §11. Stack access
    // ════════════════════════════════════════════════════════════

    let not_sts = one.clone() - store_to_stack.clone();
    let pos_m4 = Expression::from_dim(builder.position) - 4.0;

    let stack_top = builder.fetch_vec(
        vec![store_value.clone(), pos_m4.clone()],
        Some(stack_depth.clone()),
        Some(stack_depth.clone()),
        Some(not_sts.clone()),
        TieBreak::Latest,
    );
    let stack_top_value = stack_top[0].clone();
    let stack_top_position = stack_top[1].clone();

    let stack_second = builder.fetch_vec(
        vec![store_value.clone(), pos_m4.clone()],
        Some(stack_depth.clone() - 1.0),
        Some(stack_depth.clone()),
        Some(not_sts.clone()),
        TieBreak::Latest,
    );
    let stack_second_value = stack_second[0].clone();
    let stack_second_position = stack_second[1].clone();

    let stack_third_position = builder.fetch(
        pos_m4,
        Some(stack_depth.clone() - 2.0),
        Some(stack_depth),
        Some(not_sts),
        TieBreak::Latest,
    );

    let pos_expr = Expression::from_dim(builder.position);
    let top_byte = builder.fetch(
        byte_number_m1.clone(),
        Some(stack_top_position.clone() + byte_index.clone()),
        Some(pos_expr.clone()),
        None,
        TieBreak::Latest,
    );
    let second_byte = builder.fetch(
        byte_number_m1.clone(),
        Some(stack_second_position.clone() + byte_index.clone()),
        Some(pos_expr.clone()),
        None,
        TieBreak::Latest,
    );
    let third_byte = builder.fetch(
        byte_number_m1.clone(),
        Some(stack_third_position.clone() + byte_index.clone()),
        Some(pos_expr),
        None,
        TieBreak::Latest,
    );

    names.extend([
        ("stack_top_value".into(), stack_top_value.clone()),
        ("stack_top_position".into(), stack_top_position),
        ("stack_second_value".into(), stack_second_value.clone()),
        ("stack_second_position".into(), stack_second_position),
        ("stack_third_position".into(), stack_third_position),
        ("top_byte".into(), top_byte.clone()),
        ("second_byte".into(), second_byte.clone()),
        ("third_byte".into(), third_byte.clone()),
    ]);

    // ════════════════════════════════════════════════════════════
    // §12. Memory
    // ════════════════════════════════════════════════════════════

    let memory_read_address =
        stack_top_value.clone() + fetch_result.immediate.clone() + byte_index.clone();
    let memory_write_address =
        stack_second_value.clone() + fetch_result.immediate.clone() + byte_index.clone() - 1.0;

    let not_memory_write_byte = one.clone() + is_boundary.clone() - memory_write_gate.clone();

    let memory_dirty = builder.fetch_vec(
        vec![byte_number_m1.clone(), memory_write_address.clone()],
        Some(memory_read_address.clone()),
        Some(memory_write_address),
        Some(not_memory_write_byte),
        TieBreak::Latest,
    );
    let memory_byte_dirty = memory_dirty[0].clone();
    let memory_byte_dirty_pos = memory_dirty[1].clone();

    // 3-point interpolation: exact address match at diff = 0
    let mem_diff = memory_byte_dirty_pos - memory_read_address;
    let memory_byte = builder.reglu(memory_byte_dirty.clone(), mem_diff.clone() + 1.0)
        - builder.reglu(memory_byte_dirty.clone(), mem_diff.clone()) * 2.0
        + builder.reglu(memory_byte_dirty, mem_diff - 1.0);

    names.push(("memory_byte".into(), memory_byte.clone()));

    // ════════════════════════════════════════════════════════════
    // §13. Byte-serial arithmetic
    // ════════════════════════════════════════════════════════════

    let carry_late = builder.persist(carry);

    // Addition: sum = second + top + carry_in
    let add_value = second_byte.clone() + top_byte.clone() + carry_late.clone();
    let add_carry = builder.stepglu(one.clone(), add_value.clone() - 256.0 * one.clone());
    let add_byte = add_value - 256.0 * add_carry.clone();

    // Subtraction: diff = second − top − carry_in
    let sub_value = second_byte.clone() - top_byte.clone() - carry_late.clone();
    let sub_step = builder.stepglu(one.clone(), sub_value.clone());
    let sub_borrow = one.clone() - sub_step;
    let sub_byte = sub_value + 256.0 * sub_borrow.clone();

    // Memory sign bit
    let memory_sign = builder.stepglu(one.clone(), memory_byte.clone() - 128.0);

    names.extend([
        ("carry_late".into(), carry_late.clone()),
        ("add_carry".into(), add_carry.clone()),
        ("add_byte".into(), add_byte.clone()),
        ("sub_borrow".into(), sub_borrow.clone()),
        ("sub_byte".into(), sub_byte.clone()),
        ("memory_sign".into(), memory_sign.clone()),
    ]);

    // ════════════════════════════════════════════════════════════
    // §14. Comparisons
    // ════════════════════════════════════════════════════════════

    let cmp = ComparisonGates::build(builder, stack_top_value, stack_second_value);

    names.extend([
        ("a_gt_b_u".into(), cmp.a_gt_b_u.clone()),
        ("a_lt_b_u".into(), cmp.a_lt_b_u.clone()),
        ("a_eq_b".into(), cmp.a_eq_b.clone()),
        ("a_gt_b_s".into(), cmp.a_gt_b_s.clone()),
        ("a_lt_b_s".into(), cmp.a_lt_b_s.clone()),
        ("cond_nonzero".into(), cmp.cond_nonzero.clone()),
    ]);

    // ════════════════════════════════════════════════════════════
    // §15. Call stack
    // ════════════════════════════════════════════════════════════

    let call_write_key = call_depth.clone() * 4.0 + byte_index.clone() - 1.0;
    let call_read_key = (call_depth - 1.0) * 4.0 + byte_index.clone();

    let call_dot = dc.dot(Opcode::Call).clone();
    let not_call_byte = if specialized {
        let inner = one.clone() - is_boundary.clone();
        one.clone() - builder.stepglu(inner, call_dot)
    } else {
        let inner = one.clone() - is_boundary.clone();
        one.clone() - builder.reglu(inner, call_dot)
    };

    let call_stack_byte = builder.fetch(
        byte_number_m1,
        Some(call_read_key),
        Some(call_write_key),
        Some(not_call_byte),
        TieBreak::Latest,
    );

    names.push(("call_stack_byte".into(), call_stack_byte.clone()));

    // ════════════════════════════════════════════════════════════
    // §16. Result byte multiplexer
    // ════════════════════════════════════════════════════════════

    // Branch/return byte subtraction
    let ret_dot = dc.dot(Opcode::Return).clone();
    let csb = builder.reglu(call_stack_byte, ret_dot.clone());
    let cc = builder.reglu(carry_late.clone(), ret_dot);
    let branch_sub_val = builder.persist(fetch_result.const_byte - csb - cc);
    let branch_sub_step = builder.stepglu(one.clone(), branch_sub_val.clone());
    let branch_sub_borrow = one.clone() - branch_sub_step;
    let branch_byte = branch_sub_val.clone() + 256.0 * branch_sub_borrow.clone();
    let branch_carry = branch_sub_borrow;

    let byte_at_2 = builder.stepglu(one.clone(), byte_index.clone() - 2.0);

    // Early result bytes
    let dot_local_get = dc.dot(Opcode::LocalGet).clone();
    let dot_select = dc.dot(Opcode::Select).clone();
    let cond_nonzero = cmp.cond_nonzero.clone();

    let rbe_0 = builder.reglu(local_byte, dot_local_get);
    let rbe_1 = builder.reglu(top_byte.clone(), uses_top_byte);
    let rbe_2 = builder.reglu(
        third_byte,
        dot_select.clone() + cond_nonzero.clone() - one.clone(),
    );
    let rbe_3 = builder.reglu(second_byte, dot_select - cond_nonzero.clone());
    let result_byte_early = builder.persist(rbe_0 + rbe_1 + rbe_2 + rbe_3);

    // Part 1: byte-value results (arithmetic, memory, branch)
    let boundary_m1 = is_boundary.clone() - one.clone();

    let rp1_0 = builder.reglu(branch_sub_val.clone(), dc.dot(Opcode::I32Const).clone());
    let rp1_1 = builder.reglu(add_byte, dc.dot(Opcode::I32Add).clone());
    let rp1_2 = builder.reglu(sub_byte, dc.dot(Opcode::I32Sub).clone());
    let rp1_3 = builder.reglu(memory_byte.clone(), dc.dot(Opcode::I32Load).clone());
    let rp1_4 = builder.reglu(
        memory_byte.clone(),
        dc.dot(Opcode::I32Load8U).clone() + boundary_m1.clone(),
    );
    let rp1_5 = builder.reglu(
        memory_byte.clone(),
        dc.dot(Opcode::I32Load8S).clone() + boundary_m1.clone(),
    );
    let rp1_6 = builder.reglu(
        carry_late.clone(),
        dc.dot(Opcode::I32Load8S).clone() - is_boundary.clone(),
    );
    let rp1_7 = builder.reglu(
        memory_byte.clone(),
        dc.dot(Opcode::I32Load16U).clone() - byte_at_2.clone(),
    );
    let rp1_8 = builder.reglu(
        memory_byte.clone(),
        dc.dot(Opcode::I32Load16S).clone() - byte_at_2.clone(),
    );
    let rp1_9 = builder.reglu(
        carry_late.clone(),
        dc.dot(Opcode::I32Load16S).clone() + byte_at_2.clone() - one.clone(),
    );
    let rp1_10 = builder.reglu(branch_sub_val.clone(), dc.dot(Opcode::Br).clone());
    let rp1_11 = builder.reglu(branch_sub_val.clone(), dc.dot(Opcode::BrIf).clone());
    let rp1_12 = builder.reglu(branch_sub_val, dc.dot(Opcode::Call).clone());
    let rp1_13 = builder.reglu(branch_byte, dc.dot(Opcode::Return).clone());
    let result_byte_p1 = builder.persist(
        rp1_0
            + rp1_1
            + rp1_2
            + result_byte_early
            + rp1_3
            + rp1_4
            + rp1_5
            + rp1_6 * 255.0
            + rp1_7
            + rp1_8
            + rp1_9 * 255.0
            + rp1_10
            + rp1_11
            + rp1_12
            + rp1_13,
    );

    // Part 2: comparison results (only at boundary byte 0)
    let rp2_0 = builder.reglu(
        cmp.a_eq_b.clone(),
        dc.dot(Opcode::I32Eq).clone() + boundary_m1.clone(),
    );
    let rp2_1 = builder.reglu(
        one.clone() - cmp.a_eq_b,
        dc.dot(Opcode::I32Ne).clone() + boundary_m1.clone(),
    );
    let rp2_2 = builder.reglu(
        cmp.a_gt_b_u.clone(),
        dc.dot(Opcode::I32GtU).clone() + boundary_m1.clone(),
    );
    let rp2_3 = builder.reglu(
        one.clone() - cmp.a_gt_b_u,
        dc.dot(Opcode::I32LeU).clone() + boundary_m1.clone(),
    );
    let rp2_4 = builder.reglu(
        cmp.a_lt_b_u.clone(),
        dc.dot(Opcode::I32LtU).clone() + boundary_m1.clone(),
    );
    let rp2_5 = builder.reglu(
        one.clone() - cmp.a_lt_b_u,
        dc.dot(Opcode::I32GeU).clone() + boundary_m1.clone(),
    );
    let rp2_6 = builder.reglu(
        cmp.a_gt_b_s.clone(),
        dc.dot(Opcode::I32GtS).clone() + boundary_m1.clone(),
    );
    let rp2_7 = builder.reglu(
        one.clone() - cmp.a_gt_b_s,
        dc.dot(Opcode::I32LeS).clone() + boundary_m1.clone(),
    );
    let rp2_8 = builder.reglu(
        cmp.a_lt_b_s.clone(),
        dc.dot(Opcode::I32LtS).clone() + boundary_m1.clone(),
    );
    let rp2_9 = builder.reglu(
        one.clone() - cmp.a_lt_b_s,
        dc.dot(Opcode::I32GeS).clone() + boundary_m1.clone(),
    );
    let rp2_10 = builder.reglu(
        one.clone() - cond_nonzero,
        dc.dot(Opcode::I32Eqz).clone() + boundary_m1,
    );
    let result_byte_p2 = builder.persist(
        rp2_0 + rp2_1 + rp2_2 + rp2_3 + rp2_4 + rp2_5 + rp2_6 + rp2_7 + rp2_8 + rp2_9 + rp2_10,
    );

    let result_byte = result_byte_p1 + result_byte_p2;

    // Result carry multiplexer
    let rc_0 = builder.reglu(add_carry, dc.dot(Opcode::I32Add).clone());
    let rc_1 = builder.reglu(sub_borrow, dc.dot(Opcode::I32Sub).clone());
    let rc_2 = builder.reglu(
        memory_sign.clone(),
        dc.dot(Opcode::I32Load8S).clone() + is_boundary.clone() - one.clone(),
    );
    let rc_3 = builder.reglu(
        carry_late.clone(),
        dc.dot(Opcode::I32Load8S).clone() - is_boundary.clone(),
    );
    let rc_4 = builder.reglu(
        memory_sign,
        dc.dot(Opcode::I32Load16S).clone() - byte_at_2.clone(),
    );
    let rc_5 = builder.reglu(
        carry_late - memory_byte,
        dc.dot(Opcode::I32Load16S).clone() + byte_at_2.clone() - one.clone(),
    );
    let rc_6 = builder.reglu(branch_carry, dc.dot(Opcode::Return).clone());
    let result_carry = builder.persist(rc_0 + rc_1 + rc_2 + rc_3 + rc_4 + rc_5 + rc_6);

    names.extend([
        ("result_byte".into(), result_byte.clone()),
        ("result_carry".into(), result_carry.clone()),
    ]);

    // ════════════════════════════════════════════════════════════
    // §17. Emit gates
    // ════════════════════════════════════════════════════════════

    let byte_index_4 = builder.stepglu(one.clone(), byte_index - 4.0);

    let ed_0 = builder.reglu(
        one.clone() - is_boundary.clone(),
        dc.dot(Opcode::I32Store8).clone(),
    );
    let ed_1 = builder.reglu(byte_at_2.clone(), dc.dot(Opcode::I32Store16).clone());
    let early_done = builder.persist(ed_0 + ed_1);

    let byte_done = byte_index_4.clone() + early_done;
    let is_byte_seq = one.clone() - is_boundary.clone() - byte_done.clone();

    let emit_halt = builder.reglu(is_boundary.clone(), dc.dot(Opcode::Halt).clone());

    let ebt_0 = builder.reglu(
        is_boundary.clone(),
        dc.dot(Opcode::Br).clone() - is_branch_taken.clone(),
    );
    let ebt_1 = builder.reglu(
        is_boundary.clone(),
        cmp.cond_nonzero + dc.dot(Opcode::BrIf).clone() - is_branch_taken.clone() - one.clone(),
    );
    let ebt_2 = builder.reglu(
        is_boundary.clone(),
        dc.dot(Opcode::Return).clone() - is_branch_taken.clone(),
    );
    let ebt_3 = builder.reglu(
        is_boundary.clone(),
        dc.dot(Opcode::Call).clone() - is_branch_taken.clone(),
    );
    let emit_branch_taken = ebt_0 + ebt_1 + ebt_2 + ebt_3;

    let emit_return_commit = builder.reglu(byte_index_4.clone(), dc.dot(Opcode::Return).clone());
    let emit_out = builder.reglu(is_boundary.clone(), dc.dot(Opcode::Output).clone());
    let emit_byte_start = builder.reglu(is_producing_bytes.clone(), is_boundary.clone());
    let emit_byte = emit_byte_start.clone() + is_byte_seq + is_branch_taken.clone();
    let emit_call_commit = builder.reglu(byte_index_4, dc.dot(Opcode::Call).clone());
    let emit_bt = builder.reglu(byte_done.clone(), dc.dot(Opcode::Br).clone())
        + builder.reglu(byte_done.clone(), dc.dot(Opcode::BrIf).clone());

    let emit_commit = byte_done + is_boundary
        - emit_halt.clone()
        - emit_branch_taken.clone()
        - emit_return_commit.clone()
        - emit_out.clone()
        - emit_byte_start
        - is_branch_taken
        - emit_call_commit.clone();

    names.extend([
        ("emit_halt".into(), emit_halt.clone()),
        ("emit_branch_taken".into(), emit_branch_taken.clone()),
        ("emit_return_commit".into(), emit_return_commit.clone()),
        ("emit_out".into(), emit_out.clone()),
        ("emit_byte".into(), emit_byte.clone()),
        ("emit_call_commit".into(), emit_call_commit.clone()),
        ("emit_commit".into(), emit_commit.clone()),
    ]);

    // ════════════════════════════════════════════════════════════
    // §18. Output tokens
    // ════════════════════════════════════════════════════════════

    let emit_gates = EmitGates {
        emit_halt,
        emit_branch_taken,
        emit_call_commit,
        emit_return_commit,
        emit_out,
        emit_byte,
        emit_commit,
        emit_bt,
    };

    let fetched_state = FetchedState {
        fetched_stack_delta: fetch_result.fetched_stack_delta,
        fetched_store_to_stack: fetch_result.fetched_store_to_stack,
    };

    let output_tokens = build_output_tokens(
        &emit_gates,
        &fetched_state,
        &result_byte,
        &result_carry,
        &top_byte,
    );

    // ════════════════════════════════════════════════════════════
    // §19. Auto-name graph dimensions
    // ════════════════════════════════════════════════════════════

    builder.auto_name(&names);

    (input_tokens, output_tokens)
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_universal_creates_nonempty_tokens() {
        let mut builder = GraphBuilder::new();
        let (input_tokens, output_tokens) = build(None, &mut builder);

        assert!(!input_tokens.is_empty(), "input tokens should not be empty");
        assert!(
            !output_tokens.is_empty(),
            "output tokens should not be empty"
        );

        assert!(input_tokens.contains_key("00"), "missing byte token 00");
        assert!(input_tokens.contains_key("ff"), "missing byte token ff");
        assert!(input_tokens.contains_key("00'"), "missing carry token 00'");
        assert!(
            input_tokens.contains_key("commit(+1,sts=1,bt=0)"),
            "missing commit token"
        );
        assert!(
            input_tokens.contains_key("i32.add"),
            "missing opcode token i32.add"
        );
        assert!(
            input_tokens.contains_key("local.get"),
            "missing opcode token local.get"
        );
        assert!(input_tokens.contains_key("{"), "missing {{ token");
        assert!(input_tokens.contains_key("}"), "missing }} token");

        assert!(
            output_tokens.contains_key("halt"),
            "missing halt output token"
        );
        assert!(
            output_tokens.contains_key("00"),
            "missing byte output token"
        );
        assert!(
            output_tokens.contains_key("ff'"),
            "missing carry output token"
        );
    }

    #[test]
    fn test_build_universal_has_correct_token_count() {
        let mut builder = GraphBuilder::new();
        let (input_tokens, output_tokens) = build(None, &mut builder);

        assert!(
            input_tokens.len() > 700,
            "expected >700 input tokens, got {}",
            input_tokens.len()
        );
        assert!(
            output_tokens.len() > 700,
            "expected >700 output tokens, got {}",
            output_tokens.len()
        );
    }

    #[test]
    fn test_build_universal_creates_graph_dimensions() {
        let mut builder = GraphBuilder::new();
        let dim_count_before = builder.dim_count();

        let _ = build(None, &mut builder);

        let dim_count_after = builder.dim_count();
        assert!(
            dim_count_after > dim_count_before + 100,
            "expected significant dimension growth, got {} -> {}",
            dim_count_before,
            dim_count_after
        );
    }

    #[test]
    fn test_build_specialized_creates_nonempty_tokens() {
        let mut builder = GraphBuilder::new();

        let program = vec![
            ProgramInstruction::with_i32(Opcode::I32Const, 42),
            ProgramInstruction::new(Opcode::Output),
            ProgramInstruction::new(Opcode::Halt),
        ];

        let (input_tokens, output_tokens) = build(Some(&program), &mut builder);

        assert!(!input_tokens.is_empty(), "input tokens should not be empty");
        assert!(
            !output_tokens.is_empty(),
            "output tokens should not be empty"
        );

        assert!(
            !input_tokens.contains_key("i32.add"),
            "specialized should not have opcode tokens"
        );
        assert!(
            input_tokens.contains_key("start"),
            "specialized should have start token"
        );
        // "{" exists as a printable ASCII alias (0x7B) for byte token "7b" in both modes,
        // but the structural "{" token (zero embedding) is only in universal mode.
        // Check that opcode tokens are absent instead.
        assert!(
            !input_tokens.contains_key("local.get"),
            "specialized should not have opcode tokens"
        );
    }

    #[test]
    fn test_build_idempotent_with_same_builder() {
        let mut builder1 = GraphBuilder::new();
        let _ = build(None, &mut builder1);

        let mut builder2 = GraphBuilder::new();
        let _ = build(None, &mut builder2);

        assert_eq!(
            builder1.dim_count(),
            builder2.dim_count(),
            "identical builds should have same dimension count"
        );
        assert_eq!(
            builder1.lookup_count(),
            builder2.lookup_count(),
            "identical builds should have same lookup count"
        );
    }

    #[test]
    fn test_build_universal_creates_lookups() {
        let mut builder = GraphBuilder::new();
        let _ = build(None, &mut builder);

        // Should have created attention lookups for instruction fetch,
        // stack access, memory, cumsum, etc.
        assert!(
            builder.lookup_count() > 10,
            "expected >10 lookups, got {}",
            builder.lookup_count()
        );
    }
}
