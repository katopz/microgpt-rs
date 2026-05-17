// SPDX-License-Identifier: Apache-2.0
// Distilled from Percepta's `transformer-vm` (Apache-2.0 © Percepta).

//! Byte-serial arithmetic for the WASM interpreter computation graph.
//!
//! All arithmetic operations work byte-by-byte with carry propagation,
//! like hardware adders. Each step of the transformer processes one byte
//! (byte_index 0..4) and propagates carry to the next step.
//!
//! # Byte-serial addition
//!
//! ```text
//! sum     = second_byte + top_byte + carry_in
//! carry   = step(sum >= 256)
//! result  = sum − 256·carry
//! ```
//!
//! # Byte-serial subtraction
//!
//! ```text
//! diff    = second_byte − top_byte − carry_in
//! borrow  = 1 − step(diff >= 0)
//! result  = diff + 256·borrow
//! ```

use crate::percepta::graph::types::{Expression, GraphBuilder};

// ── Byte value helper ──────────────────────────────────────────

/// Compute the value contribution of a single byte in a multi-byte integer.
///
/// For byte `bv` at position `i` (little-endian):
/// - Unsigned: `bv * (1 << (8 * i))`
/// - Signed (when `bv >= 128`): `(bv - 256) * (1 << (8 * i))`
///
/// This matches the Python `get_byte_value(bv, i, signed)`.
pub fn get_byte_value(bv: u8, i: u32, signed: bool) -> i64 {
    let raw = if signed && bv >= 128 {
        bv as i64 - 256
    } else {
        bv as i64
    };
    raw << (8 * i)
}

// ── ByteArithmetic — result of byte-serial arithmetic build ────

/// Holds all computed arithmetic expressions from the byte-serial ALU.
///
/// Built by [`ByteArithmetic::build`], these expressions are used by
/// the interpreter's result-byte multiplexer.
pub struct ByteArithmetic {
    /// Result byte of addition: `sum − 256·carry_out`.
    pub add_byte: Expression,
    /// Carry-out of addition: `step(sum >= 256)`.
    pub add_carry: Expression,
    /// Result byte of subtraction: `diff + 256·borrow`.
    pub sub_byte: Expression,
    /// Borrow-out of subtraction: `1 − step(diff >= 0)`.
    pub sub_borrow: Expression,
    /// Sign bit extracted from memory byte (for sign-extension).
    pub memory_sign: Expression,
}

impl ByteArithmetic {
    /// Build the byte-serial ALU using graph primitives.
    ///
    /// # Arguments
    ///
    /// * `builder` — the computation graph builder
    /// * `second_byte` — byte from the second stack value (byte_index position)
    /// * `top_byte` — byte from the top stack value (byte_index position)
    /// * `memory_byte` — byte from memory (for memory load sign bits)
    /// * `carry` — the carry input expression (from `InputDimension("carry")`)
    ///
    /// # Returns
    ///
    /// All computed arithmetic result expressions.
    pub fn build(
        builder: &mut GraphBuilder,
        second_byte: Expression,
        top_byte: Expression,
        memory_byte: Expression,
        carry: Expression,
    ) -> Self {
        let one_expr = Expression::from_dim(builder.one);

        // Persist carry so it survives across steps
        let carry_late = builder.persist(carry);

        // ── Addition: sum = second + top + carry ──
        let add_value = second_byte.clone() + top_byte.clone() + carry_late.clone();
        let add_carry = builder.stepglu(
            one_expr.clone(),
            add_value.clone() - 256.0_f64 * one_expr.clone(),
        );
        let add_byte = add_value - 256.0_f64 * add_carry.clone();

        // ── Subtraction: diff = second − top − carry ──
        let sub_value = second_byte - top_byte - carry_late;
        let sub_borrow_expr = builder.stepglu(one_expr.clone(), sub_value.clone());
        // borrow = 1 − step(diff >= 0)
        let sub_borrow = one_expr.clone() - sub_borrow_expr;
        let sub_byte = sub_value + 256.0_f64 * sub_borrow.clone();

        // ── Memory sign bit (for sign-extending loads) ──
        let memory_sign = builder.stepglu(one_expr, memory_byte - 128.0_f64);

        Self {
            add_byte,
            add_carry,
            sub_byte,
            sub_borrow,
            memory_sign,
        }
    }
}

// ── Comparisons — unsigned and signed ──────────────────────────

/// Holds comparison result expressions.
///
/// Built by [`Comparisons::build`], these gates indicate the result of
/// comparing the top two stack values.
pub struct Comparisons {
    /// `step(second > top − 1)` — 1 if second >= top (unsigned).
    pub a_gt_b_u: Expression,
    /// `step(top > second − 1)` — 1 if top >= second (unsigned).
    pub a_lt_b_u: Expression,
    /// `one − gt_u − lt_u` — 1 if second == top.
    pub a_eq_b: Expression,
    /// Signed greater-than: combines sign difference with unsigned comparison.
    pub a_gt_b_s: Expression,
    /// Signed less-than: combines sign difference with unsigned comparison.
    pub a_lt_b_s: Expression,
    /// `step(top >= 1)` — 1 if top value is nonzero.
    pub cond_nonzero: Expression,
}

impl Comparisons {
    /// Build comparison gates for the top two stack values.
    ///
    /// # Arguments
    ///
    /// * `builder` — the computation graph builder
    /// * `stack_top_value` — full 32-bit top stack value expression
    /// * `stack_second_value` — full 32-bit second stack value expression
    pub fn build(
        builder: &mut GraphBuilder,
        stack_top_value: Expression,
        stack_second_value: Expression,
    ) -> Self {
        let one_expr = Expression::from_dim(builder.one);

        // ── Unsigned comparisons via stepglu ──
        let a_gt_b_u = builder.stepglu(
            one_expr.clone(),
            stack_second_value.clone() - stack_top_value.clone() - one_expr.clone(),
        );
        let a_lt_b_u = builder.stepglu(
            one_expr.clone(),
            stack_top_value.clone() - stack_second_value.clone() - one_expr.clone(),
        );

        // Equality: if neither is strictly greater, they must be equal
        let a_eq_b = one_expr.clone() - a_gt_b_u.clone() - a_lt_b_u.clone();

        // ── Sign extraction for signed comparison ──
        // sign_diff = sign(top) − sign(second)
        // where sign(x) = reglu(1, x − 2³¹ + 1) − reglu(1, x − 2³¹)
        let i31 = 1_i64 << 31;
        let sign_top = builder.reglu(
            one_expr.clone(),
            stack_top_value.clone() - (i31 as f64) + one_expr.clone(),
        ) - builder.reglu(one_expr.clone(), stack_top_value.clone() - (i31 as f64));
        let sign_second = builder.reglu(
            one_expr.clone(),
            stack_second_value.clone() - (i31 as f64) + one_expr.clone(),
        ) - builder
            .reglu(one_expr.clone(), stack_second_value.clone() - (i31 as f64));

        let sign_diff = builder.persist(sign_top - sign_second);

        // Signed greater-than: positive sign_diff wins, otherwise use unsigned
        let a_gt_b_s = builder.stepglu(
            one_expr.clone(),
            sign_diff.clone() + a_gt_b_u.clone() - one_expr.clone(),
        );
        // Signed less-than: negative sign_diff wins, otherwise use unsigned
        let a_lt_b_s = builder.stepglu(
            one_expr.clone(),
            -sign_diff + a_lt_b_u.clone() - one_expr.clone(),
        );

        // ── Nonzero check (for br_if, eqz, select) ──
        let cond_nonzero = builder.stepglu(one_expr, stack_top_value - 1.0_f64);

        Self {
            a_gt_b_u,
            a_lt_b_u,
            a_eq_b,
            a_gt_b_s,
            a_lt_b_s,
            cond_nonzero,
        }
    }
}

// ── Branch subtraction ─────────────────────────────────────────

/// Build the branch/return byte subtraction with carry.
///
/// Used for computing the target byte of branch (br, br_if, call, return)
/// instructions. The branch byte is `const_byte − call_stack_byte − carry_in`,
/// with borrow propagation.
///
/// # Returns
///
/// `(branch_byte, branch_carry)` where:
/// - `branch_byte` = `diff + 256·borrow`
/// - `branch_carry` = `borrow`
pub fn build_branch_sub(
    builder: &mut GraphBuilder,
    const_byte: Expression,
    call_stack_byte: Expression,
    carry: Expression,
    dispatch: &mut crate::percepta::wasm::interpreter::dispatch::OpcodeDispatch,
) -> (Expression, Expression) {
    let one_expr = Expression::from_dim(builder.one);

    // Mask call_stack_byte and carry to only the return opcode
    let csb = builder.reglu(
        call_stack_byte,
        dispatch.op_dot(crate::percepta::wasm::interpreter::dispatch::Opcode::Return),
    );
    let cc = builder.reglu(
        carry,
        dispatch.op_dot(crate::percepta::wasm::interpreter::dispatch::Opcode::Return),
    );

    // branch_sub_val = const_byte − call_stack_byte − carry
    let branch_sub_val = builder.persist(const_byte - csb - cc);

    // borrow = 1 − step(branch_sub_val >= 0)
    let sub_borrow_expr = builder.stepglu(one_expr.clone(), branch_sub_val.clone());
    let branch_sub_borrow = one_expr - sub_borrow_expr;

    // result = diff + 256·borrow
    let branch_byte = branch_sub_val.clone() + 256.0_f64 * branch_sub_borrow.clone();
    let branch_carry = branch_sub_borrow;

    (branch_byte, branch_carry)
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_byte_value_unsigned() {
        // Byte 0x42 at position 0 → 0x42
        assert_eq!(get_byte_value(0x42, 0, false), 0x42);
        // Byte 0xFF at position 0 → 255 (unsigned)
        assert_eq!(get_byte_value(0xFF, 0, false), 255);
        // Byte 0x01 at position 1 → 256
        assert_eq!(get_byte_value(0x01, 1, false), 256);
        // Byte 0x01 at position 2 → 65536
        assert_eq!(get_byte_value(0x01, 2, false), 65536);
    }

    #[test]
    fn test_get_byte_value_signed() {
        // Byte 0x42 at position 0 (signed) → 0x42 (below 128, same as unsigned)
        assert_eq!(get_byte_value(0x42, 0, true), 0x42);
        // Byte 0xFF at position 0 (signed) → -1 (0xFF - 256)
        assert_eq!(get_byte_value(0xFF, 0, true), -1);
        // Byte 0x80 at position 0 (signed) → -128 (0x80 - 256)
        assert_eq!(get_byte_value(0x80, 0, true), -128);
        // Byte 0xFE at position 1 (signed) → (-2) * 256 = -512
        assert_eq!(get_byte_value(0xFE, 1, true), -512);
    }

    #[test]
    fn test_byte_arithmetic_build_creates_expressions() {
        let mut builder = GraphBuilder::new();
        let second_byte = builder.generic("second_byte");
        let top_byte = builder.generic("top_byte");
        let memory_byte = builder.generic("memory_byte");
        let carry = Expression::from_dim(builder.one); // use one as carry

        let dim_count_before = builder.dim_count();
        let arith = ByteArithmetic::build(&mut builder, second_byte, top_byte, memory_byte, carry);

        // All results should be non-zero expressions
        assert!(!arith.add_byte.is_zero());
        assert!(!arith.add_carry.is_zero());
        assert!(!arith.sub_byte.is_zero());
        assert!(!arith.sub_borrow.is_zero());
        assert!(!arith.memory_sign.is_zero());

        // Should have created new dimensions (persist, stepglu, etc.)
        assert!(builder.dim_count() > dim_count_before);
    }

    #[test]
    fn test_comparisons_build_creates_expressions() {
        let mut builder = GraphBuilder::new();
        let stack_top = builder.generic("stack_top");
        let stack_second = builder.generic("stack_second");

        let dim_count_before = builder.dim_count();
        let cmp = Comparisons::build(&mut builder, stack_top, stack_second);

        assert!(!cmp.a_gt_b_u.is_zero());
        assert!(!cmp.a_lt_b_u.is_zero());
        assert!(!cmp.a_eq_b.is_zero());
        assert!(!cmp.a_gt_b_s.is_zero());
        assert!(!cmp.a_lt_b_s.is_zero());
        assert!(!cmp.cond_nonzero.is_zero());

        assert!(builder.dim_count() > dim_count_before);
    }

    #[test]
    fn test_branch_sub_build_creates_expressions() {
        use crate::percepta::wasm::interpreter::dispatch::OpcodeDispatch;

        let mut builder = GraphBuilder::new();
        let one_expr = Expression::from_dim(builder.one);
        let fetched_x = builder.generic("fx");
        let fetched_y = builder.generic("fy");

        let mut dispatch = OpcodeDispatch::new(fetched_x, fetched_y, one_expr, false);

        let const_byte = builder.generic("const_byte");
        let call_stack_byte = builder.generic("call_stack_byte");
        let carry = builder.generic("carry");

        let (branch_byte, branch_carry) = build_branch_sub(
            &mut builder,
            const_byte,
            call_stack_byte,
            carry,
            &mut dispatch,
        );

        assert!(!branch_byte.is_zero());
        assert!(!branch_carry.is_zero());
    }
}
