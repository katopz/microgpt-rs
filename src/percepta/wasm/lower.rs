// SPDX-License-Identifier: Apache-2.0
// Distilled from Percepta's `transformer-vm` (Apache-2.0 © Percepta).

//! WASM instruction lowering pass.
//!
//! Replaces hard-to-simulate WASM instructions (MUL, DIV_U, DIV_S, REM_U, REM_S,
//! AND, OR, XOR, SHL, SHR_U, SHR_S, ROTL, ROTR, CLZ, CTZ, POPCNT, EXTEND8_S,
//! EXTEND16_S) with sequences of basic instructions that the transformer can
//! execute: ADD, SUB, comparisons, branches, local ops, LOAD/STORE.
//!
//! The lowering applies when the hard op is preceded by `i32.const C`
//! (so the second operand is a known constant), or falls back to runtime
//! loop-based expansion for variable operands.

use std::collections::HashMap;

use super::decoder::*;

/// Scratch memory address (byte 0 of linear memory used for byte extraction).
pub const SCRATCH_ADDR: i64 = 0;

/// Number of temporary i32 locals allocated for lowering.
const NUM_TEMPS: u32 = 4;

/// The set of "basic" opcodes that the transformer-vm can execute natively.
pub const BASIC_OPS: &[u8] = &[
    OP_UNREACHABLE,
    OP_NOP,
    OP_BLOCK,
    OP_LOOP,
    OP_IF,
    OP_ELSE,
    OP_END,
    OP_BR,
    OP_BR_IF,
    OP_BR_TABLE,
    OP_RETURN,
    OP_CALL,
    OP_MEMORY_SIZE,
    OP_DROP,
    OP_SELECT,
    OP_LOCAL_GET,
    OP_LOCAL_SET,
    OP_LOCAL_TEE,
    OP_GLOBAL_GET,
    OP_GLOBAL_SET,
    OP_I32_LOAD,
    OP_I32_LOAD8_S,
    OP_I32_LOAD8_U,
    OP_I32_LOAD16_S,
    OP_I32_LOAD16_U,
    OP_I32_STORE,
    OP_I32_STORE8,
    OP_I32_STORE16,
    OP_I32_CONST,
    OP_I32_EQZ,
    OP_I32_EQ,
    OP_I32_NE,
    OP_I32_LT_S,
    OP_I32_LT_U,
    OP_I32_GT_S,
    OP_I32_GT_U,
    OP_I32_LE_S,
    OP_I32_LE_U,
    OP_I32_GE_S,
    OP_I32_GE_U,
    OP_I32_ADD,
    OP_I32_SUB,
];

/// Binary opcodes that can be lowered when preceded by `i32.const`.
pub const LOWERABLE_BINOPS: &[u8] = &[
    OP_I32_MUL,
    OP_I32_DIV_S,
    OP_I32_DIV_U,
    OP_I32_REM_U,
    OP_I32_REM_S,
    OP_I32_AND,
    OP_I32_OR,
    OP_I32_SHL,
    OP_I32_SHR_U,
    OP_I32_SHR_S,
    OP_I32_XOR,
    OP_I32_ROTL,
    OP_I32_ROTR,
];

/// Unary opcodes that can be lowered.
pub const LOWERABLE_UNARY: &[u8] = &[
    OP_I32_EXTEND8_S,
    OP_I32_EXTEND16_S,
    OP_I32_CLZ,
    OP_I32_CTZ,
    OP_I32_POPCNT,
];

// ===================================================================== //
//  Bitwise operation kind                                                //
// ===================================================================== //

/// Kind of bitwise operation for byte-level expansion.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BitOp {
    And,
    Or,
    Xor,
}

// ===================================================================== //
//  Immediates conversion trait                                           //
// ===================================================================== //

/// Trait for types that can be converted into a `Vec<i64>` of immediates.
pub trait IntoImms {
    fn into_imms(self) -> Vec<i64>;
}

impl IntoImms for i64 {
    fn into_imms(self) -> Vec<i64> {
        vec![self]
    }
}

impl IntoImms for Vec<i64> {
    fn into_imms(self) -> Vec<i64> {
        self
    }
}

impl<const N: usize> IntoImms for [i64; N] {
    fn into_imms(self) -> Vec<i64> {
        self.to_vec()
    }
}

// ===================================================================== //
//  Helpers                                                               //
// ===================================================================== //

/// Shorthand for a [`WasmInstr`] with no immediates.
fn ni(op: u8) -> WasmInstr {
    WasmInstr::new(op)
}

/// Shorthand to create a [`WasmInstr`] with immediates.
fn instr(op: u8, imm: impl IntoImms) -> WasmInstr {
    WasmInstr::with_imms(op, imm.into_imms())
}

/// Check if an opcode is in the given slice.
fn opcode_in(opcode: u8, set: &[u8]) -> bool {
    set.contains(&opcode)
}

/// Pre-scan instructions to find locals always set to the same `i32.const`.
///
/// Returns a map of local index → constant value for locals that are only
/// ever assigned from `i32.const` with a single consistent value.
fn find_const_locals(instrs: &[WasmInstr]) -> HashMap<u32, i64> {
    let mut candidates: HashMap<u32, Option<i64>> = HashMap::new();
    for (i, ins) in instrs.iter().enumerate() {
        if ins.opcode == OP_LOCAL_SET || ins.opcode == OP_LOCAL_TEE {
            let local_idx = ins.immediates[0] as u32;
            if i > 0 && instrs[i - 1].opcode == OP_I32_CONST {
                let val = instrs[i - 1].immediates[0];
                candidates
                    .entry(local_idx)
                    .and_modify(|e| {
                        if let Some(prev) = e
                            && *prev != val {
                                *e = None;
                            }
                    })
                    .or_insert(Some(val));
            } else {
                candidates.insert(local_idx, None);
            }
        }
    }
    candidates
        .into_iter()
        .filter_map(|(k, v)| v.map(|val| (k, val)))
        .collect()
}

// ===================================================================== //
//  Constant-operand expansion functions                                  //
// ===================================================================== //

/// Expand `x * C` using additions (binary method).
/// Assumes `x` is in `local_a`. Returns instructions that leave result on stack.
fn expand_mul(c: i64, local_a: u32) -> Vec<WasmInstr> {
    let c = c & 0xFFFFFFFF;
    match c {
        0 => vec![instr(OP_I32_CONST, 0)],
        1 => vec![instr(OP_LOCAL_GET, local_a as i64)],
        0xFFFFFFFF => vec![
            instr(OP_I32_CONST, 0),
            instr(OP_LOCAL_GET, local_a as i64),
            ni(OP_I32_SUB),
        ],
        _ => {
            let mut bits = Vec::new();
            let mut v = c as u64;
            while v > 0 {
                bits.push(v & 1 != 0);
                v >>= 1;
            }
            let tmp = (local_a + 1) as i64;
            let mut out = vec![
                instr(OP_LOCAL_GET, local_a as i64),
                instr(OP_LOCAL_SET, tmp),
            ];
            for i in (0..bits.len().saturating_sub(1)).rev() {
                out.push(instr(OP_LOCAL_GET, tmp));
                out.push(instr(OP_LOCAL_GET, tmp));
                out.push(ni(OP_I32_ADD));
                out.push(instr(OP_LOCAL_SET, tmp));
                if bits[i] {
                    out.push(instr(OP_LOCAL_GET, tmp));
                    out.push(instr(OP_LOCAL_GET, local_a as i64));
                    out.push(ni(OP_I32_ADD));
                    out.push(instr(OP_LOCAL_SET, tmp));
                }
            }
            out.push(instr(OP_LOCAL_GET, tmp));
            out
        }
    }
}

/// Expand unsigned `x / C` using subtraction loop.
/// Assumes `x` is on stack (not in local yet).
fn expand_div_u(c: i64, local_a: u32) -> Vec<WasmInstr> {
    let n = local_a as i64;
    let q = (local_a + 1) as i64;
    vec![
        instr(OP_LOCAL_SET, n),
        instr(OP_I32_CONST, 0),
        instr(OP_LOCAL_SET, q),
        instr(OP_BLOCK, 0x40),
        instr(OP_LOOP, 0x40),
        instr(OP_LOCAL_GET, n),
        instr(OP_I32_CONST, c),
        ni(OP_I32_LT_U),
        instr(OP_BR_IF, 1),
        instr(OP_LOCAL_GET, n),
        instr(OP_I32_CONST, c),
        ni(OP_I32_SUB),
        instr(OP_LOCAL_SET, n),
        instr(OP_LOCAL_GET, q),
        instr(OP_I32_CONST, 1),
        ni(OP_I32_ADD),
        instr(OP_LOCAL_SET, q),
        instr(OP_BR, 0),
        ni(OP_END),
        ni(OP_END),
        instr(OP_LOCAL_GET, q),
    ]
}

/// Expand unsigned `x % C` using subtraction loop.
fn expand_rem_u(c: i64, local_a: u32) -> Vec<WasmInstr> {
    let n = local_a as i64;
    vec![
        instr(OP_LOCAL_SET, n),
        instr(OP_BLOCK, 0x40),
        instr(OP_LOOP, 0x40),
        instr(OP_LOCAL_GET, n),
        instr(OP_I32_CONST, c),
        ni(OP_I32_LT_U),
        instr(OP_BR_IF, 1),
        instr(OP_LOCAL_GET, n),
        instr(OP_I32_CONST, c),
        ni(OP_I32_SUB),
        instr(OP_LOCAL_SET, n),
        instr(OP_BR, 0),
        ni(OP_END),
        ni(OP_END),
        instr(OP_LOCAL_GET, n),
    ]
}

/// Expand signed `x / C` (truncates toward zero).
fn expand_div_s(c: i64, local_a: u32) -> Vec<WasmInstr> {
    let c = c & 0xFFFFFFFF;
    let c_signed = if c >= (1 << 31) { c - (1i64 << 32) } else { c };
    let abs_c = c_signed.unsigned_abs() as i64 & 0xFFFFFFFF;
    let c_negative = c_signed < 0;

    let x = local_a as i64;
    let q = (local_a + 1) as i64;
    let neg = (local_a + 2) as i64;

    let mut out = vec![
        instr(OP_LOCAL_SET, x),
        instr(OP_LOCAL_GET, x),
        instr(OP_I32_CONST, 0),
        ni(OP_I32_LT_S),
        instr(OP_LOCAL_SET, neg),
        instr(OP_BLOCK, 0x40),
        instr(OP_LOCAL_GET, neg),
        ni(OP_I32_EQZ),
        instr(OP_BR_IF, 0),
        instr(OP_I32_CONST, 0),
        instr(OP_LOCAL_GET, x),
        ni(OP_I32_SUB),
        instr(OP_LOCAL_SET, x),
        ni(OP_END),
    ];
    if c_negative {
        out.push(instr(OP_LOCAL_GET, neg));
        out.push(ni(OP_I32_EQZ));
        out.push(instr(OP_LOCAL_SET, neg));
    }
    out.extend_from_slice(&[
        instr(OP_I32_CONST, 0),
        instr(OP_LOCAL_SET, q),
        instr(OP_BLOCK, 0x40),
        instr(OP_LOOP, 0x40),
        instr(OP_LOCAL_GET, x),
        instr(OP_I32_CONST, abs_c),
        ni(OP_I32_LT_U),
        instr(OP_BR_IF, 1),
        instr(OP_LOCAL_GET, x),
        instr(OP_I32_CONST, abs_c),
        ni(OP_I32_SUB),
        instr(OP_LOCAL_SET, x),
        instr(OP_LOCAL_GET, q),
        instr(OP_I32_CONST, 1),
        ni(OP_I32_ADD),
        instr(OP_LOCAL_SET, q),
        instr(OP_BR, 0),
        ni(OP_END),
        ni(OP_END),
        instr(OP_BLOCK, 0x40),
        instr(OP_LOCAL_GET, neg),
        ni(OP_I32_EQZ),
        instr(OP_BR_IF, 0),
        instr(OP_I32_CONST, 0),
        instr(OP_LOCAL_GET, q),
        ni(OP_I32_SUB),
        instr(OP_LOCAL_SET, q),
        ni(OP_END),
        instr(OP_LOCAL_GET, q),
    ]);
    out
}

/// Expand `i32.clz` (count leading zeros). Assumes `x` is on the stack.
fn expand_clz(local_a: u32) -> Vec<WasmInstr> {
    let x = local_a as i64;
    let count = (local_a + 1) as i64;
    vec![
        instr(OP_LOCAL_SET, x),
        instr(OP_I32_CONST, 0),
        instr(OP_LOCAL_SET, count),
        instr(OP_BLOCK, 0x40),
        instr(OP_BLOCK, 0x40),
        instr(OP_LOCAL_GET, x),
        instr(OP_BR_IF, 0),
        instr(OP_I32_CONST, 32),
        instr(OP_LOCAL_SET, count),
        instr(OP_BR, 1),
        ni(OP_END),
        instr(OP_LOOP, 0x40),
        instr(OP_LOCAL_GET, x),
        instr(OP_I32_CONST, 0),
        ni(OP_I32_LT_S),
        instr(OP_BR_IF, 1),
        instr(OP_LOCAL_GET, x),
        instr(OP_LOCAL_GET, x),
        ni(OP_I32_ADD),
        instr(OP_LOCAL_SET, x),
        instr(OP_LOCAL_GET, count),
        instr(OP_I32_CONST, 1),
        ni(OP_I32_ADD),
        instr(OP_LOCAL_SET, count),
        instr(OP_BR, 0),
        ni(OP_END),
        ni(OP_END),
        instr(OP_LOCAL_GET, count),
    ]
}

/// Expand `i32.ctz` (count trailing zeros). Uses scratch memory for byte extraction.
fn expand_ctz(local_a: u32) -> Vec<WasmInstr> {
    let x = local_a as i64;
    let count = (local_a + 1) as i64;
    let byte = (local_a + 2) as i64;
    let q = (local_a + 3) as i64;
    vec![
        instr(OP_LOCAL_SET, x),
        instr(OP_I32_CONST, 0),
        instr(OP_LOCAL_SET, count),
        instr(OP_BLOCK, 0x40),
        instr(OP_BLOCK, 0x40),
        instr(OP_LOCAL_GET, x),
        instr(OP_BR_IF, 0),
        instr(OP_I32_CONST, 32),
        instr(OP_LOCAL_SET, count),
        instr(OP_BR, 1),
        ni(OP_END),
        instr(OP_LOOP, 0x40),
        instr(OP_I32_CONST, SCRATCH_ADDR),
        instr(OP_LOCAL_GET, x),
        instr(OP_I32_STORE8, [0, 0]),
        instr(OP_I32_CONST, SCRATCH_ADDR),
        instr(OP_I32_LOAD8_U, [0, 0]),
        instr(OP_LOCAL_SET, byte),
        instr(OP_BLOCK, 0x40),
        instr(OP_LOOP, 0x40),
        instr(OP_LOCAL_GET, byte),
        instr(OP_I32_CONST, 2),
        ni(OP_I32_LT_U),
        instr(OP_BR_IF, 1),
        instr(OP_LOCAL_GET, byte),
        instr(OP_I32_CONST, 2),
        ni(OP_I32_SUB),
        instr(OP_LOCAL_SET, byte),
        instr(OP_BR, 0),
        ni(OP_END),
        ni(OP_END),
        instr(OP_LOCAL_GET, byte),
        instr(OP_BR_IF, 1),
        instr(OP_I32_CONST, 0),
        instr(OP_LOCAL_SET, q),
        instr(OP_BLOCK, 0x40),
        instr(OP_LOOP, 0x40),
        instr(OP_LOCAL_GET, x),
        instr(OP_I32_CONST, 2),
        ni(OP_I32_LT_U),
        instr(OP_BR_IF, 1),
        instr(OP_LOCAL_GET, x),
        instr(OP_I32_CONST, 2),
        ni(OP_I32_SUB),
        instr(OP_LOCAL_SET, x),
        instr(OP_LOCAL_GET, q),
        instr(OP_I32_CONST, 1),
        ni(OP_I32_ADD),
        instr(OP_LOCAL_SET, q),
        instr(OP_BR, 0),
        ni(OP_END),
        ni(OP_END),
        instr(OP_LOCAL_GET, q),
        instr(OP_LOCAL_SET, x),
        instr(OP_LOCAL_GET, count),
        instr(OP_I32_CONST, 1),
        ni(OP_I32_ADD),
        instr(OP_LOCAL_SET, count),
        instr(OP_BR, 0),
        ni(OP_END),
        ni(OP_END),
        instr(OP_LOCAL_GET, count),
    ]
}

/// Expand `i32.popcnt` (population count). Uses scratch memory for bit extraction.
fn expand_popcnt(local_a: u32) -> Vec<WasmInstr> {
    let x = local_a as i64;
    let count = (local_a + 1) as i64;
    let byte = (local_a + 2) as i64;
    let q = (local_a + 3) as i64;
    vec![
        instr(OP_LOCAL_SET, x),
        instr(OP_I32_CONST, 0),
        instr(OP_LOCAL_SET, count),
        instr(OP_BLOCK, 0x40),
        instr(OP_LOOP, 0x40),
        instr(OP_LOCAL_GET, x),
        ni(OP_I32_EQZ),
        instr(OP_BR_IF, 1),
        instr(OP_I32_CONST, SCRATCH_ADDR),
        instr(OP_LOCAL_GET, x),
        instr(OP_I32_STORE8, [0, 0]),
        instr(OP_I32_CONST, SCRATCH_ADDR),
        instr(OP_I32_LOAD8_U, [0, 0]),
        instr(OP_LOCAL_SET, byte),
        instr(OP_BLOCK, 0x40),
        instr(OP_LOOP, 0x40),
        instr(OP_LOCAL_GET, byte),
        instr(OP_I32_CONST, 2),
        ni(OP_I32_LT_U),
        instr(OP_BR_IF, 1),
        instr(OP_LOCAL_GET, byte),
        instr(OP_I32_CONST, 2),
        ni(OP_I32_SUB),
        instr(OP_LOCAL_SET, byte),
        instr(OP_BR, 0),
        ni(OP_END),
        ni(OP_END),
        instr(OP_LOCAL_GET, count),
        instr(OP_LOCAL_GET, byte),
        ni(OP_I32_ADD),
        instr(OP_LOCAL_SET, count),
        instr(OP_I32_CONST, 0),
        instr(OP_LOCAL_SET, q),
        instr(OP_BLOCK, 0x40),
        instr(OP_LOOP, 0x40),
        instr(OP_LOCAL_GET, x),
        instr(OP_I32_CONST, 2),
        ni(OP_I32_LT_U),
        instr(OP_BR_IF, 1),
        instr(OP_LOCAL_GET, x),
        instr(OP_I32_CONST, 2),
        ni(OP_I32_SUB),
        instr(OP_LOCAL_SET, x),
        instr(OP_LOCAL_GET, q),
        instr(OP_I32_CONST, 1),
        ni(OP_I32_ADD),
        instr(OP_LOCAL_SET, q),
        instr(OP_BR, 0),
        ni(OP_END),
        ni(OP_END),
        instr(OP_LOCAL_GET, q),
        instr(OP_LOCAL_SET, x),
        instr(OP_BR, 0),
        ni(OP_END),
        ni(OP_END),
        instr(OP_LOCAL_GET, count),
    ]
}

/// Expand `x rotl C` where C is a compile-time constant.
fn expand_rotl_const(c: i64, local_a: u32) -> Vec<WasmInstr> {
    let c = c & 31;
    if c == 0 {
        return vec![];
    }
    let saved_x = (local_a + 2) as i64;
    let left = (local_a + 3) as i64;
    let mut out = vec![
        instr(OP_LOCAL_SET, local_a as i64),
        instr(OP_LOCAL_GET, local_a as i64),
        instr(OP_LOCAL_SET, saved_x),
    ];
    out.extend(expand_shl(c, local_a));
    out.push(instr(OP_LOCAL_SET, left));
    out.push(instr(OP_LOCAL_GET, saved_x));
    out.extend(expand_shr_u(32 - c, local_a));
    out.push(instr(OP_LOCAL_GET, left));
    out.push(ni(OP_I32_ADD));
    out
}

/// Expand `x rotr C` where C is a compile-time constant.
fn expand_rotr_const(c: i64, local_a: u32) -> Vec<WasmInstr> {
    let c = c & 31;
    if c == 0 {
        return vec![];
    }
    let saved_x = (local_a + 2) as i64;
    let right = (local_a + 3) as i64;
    let mut out = vec![
        instr(OP_LOCAL_SET, local_a as i64),
        instr(OP_LOCAL_GET, local_a as i64),
        instr(OP_LOCAL_SET, saved_x),
    ];
    out.push(instr(OP_LOCAL_GET, local_a as i64));
    out.extend(expand_shr_u(c, local_a));
    out.push(instr(OP_LOCAL_SET, right));
    out.push(instr(OP_LOCAL_GET, saved_x));
    out.push(instr(OP_LOCAL_SET, local_a as i64));
    out.extend(expand_shl(32 - c, local_a));
    out.push(instr(OP_LOCAL_GET, right));
    out.push(ni(OP_I32_ADD));
    out
}

/// Expand `x & 255` using store8 + load8_u at scratch address.
fn expand_and_255(local_a: u32) -> Vec<WasmInstr> {
    vec![
        instr(OP_LOCAL_SET, local_a as i64),
        instr(OP_I32_CONST, SCRATCH_ADDR),
        instr(OP_LOCAL_GET, local_a as i64),
        instr(OP_I32_STORE8, [0, 0]),
        instr(OP_I32_CONST, SCRATCH_ADDR),
        instr(OP_I32_LOAD8_U, [0, 0]),
    ]
}

/// Expand `i32.extend8_s` using store8 + load8_u + sign extension.
fn expand_extend8_s(local_a: u32) -> Vec<WasmInstr> {
    let a = local_a as i64;
    vec![
        instr(OP_LOCAL_SET, a),
        instr(OP_I32_CONST, SCRATCH_ADDR),
        instr(OP_LOCAL_GET, a),
        instr(OP_I32_STORE8, [0, 0]),
        instr(OP_I32_CONST, SCRATCH_ADDR),
        instr(OP_I32_LOAD8_U, [0, 0]),
        instr(OP_LOCAL_SET, a),
        instr(OP_BLOCK, 0x40),
        instr(OP_LOCAL_GET, a),
        instr(OP_I32_CONST, 128),
        ni(OP_I32_LT_U),
        instr(OP_BR_IF, 0),
        instr(OP_LOCAL_GET, a),
        instr(OP_I32_CONST, 256),
        ni(OP_I32_SUB),
        instr(OP_LOCAL_SET, a),
        ni(OP_END),
        instr(OP_LOCAL_GET, a),
    ]
}

/// Expand `i32.extend16_s` using store16 + load16_s at scratch address.
fn expand_extend16_s(local_a: u32) -> Vec<WasmInstr> {
    vec![
        instr(OP_LOCAL_SET, local_a as i64),
        instr(OP_I32_CONST, SCRATCH_ADDR),
        instr(OP_LOCAL_GET, local_a as i64),
        instr(OP_I32_STORE16, [0, 0]),
        instr(OP_I32_CONST, SCRATCH_ADDR),
        instr(OP_I32_LOAD16_S, [0, 0]),
    ]
}

/// Expand `x << C`. Uses memory for byte-aligned part, doubling for rest.
/// Precondition: x is in local `local_a`. Postcondition: result on stack.
fn expand_shl(c: i64, local_a: u32) -> Vec<WasmInstr> {
    let a = local_a as i64;
    if c == 0 {
        return vec![instr(OP_LOCAL_GET, a)];
    }
    if c >= 32 {
        return vec![instr(OP_I32_CONST, 0)];
    }
    let q = c / 8;
    let r = c % 8;
    if q > 0 {
        let mut out = vec![
            instr(OP_I32_CONST, SCRATCH_ADDR),
            instr(OP_I32_CONST, 0),
            instr(OP_I32_STORE, [0, 0]),
            instr(OP_I32_CONST, SCRATCH_ADDR),
            instr(OP_LOCAL_GET, a),
            instr(OP_I32_STORE, [0, q]),
            instr(OP_I32_CONST, SCRATCH_ADDR),
            instr(OP_I32_LOAD, [0, 0]),
        ];
        for _ in 0..r {
            out.push(instr(OP_LOCAL_TEE, a));
            out.push(instr(OP_LOCAL_GET, a));
            out.push(ni(OP_I32_ADD));
        }
        return out;
    }
    let mut out = vec![instr(OP_LOCAL_GET, a)];
    for _ in 0..c {
        out.push(instr(OP_LOCAL_TEE, a));
        out.push(instr(OP_LOCAL_GET, a));
        out.push(ni(OP_I32_ADD));
    }
    out
}

/// Expand `x << C` where x is on the stack (not yet in a local).
fn expand_shl_from_stack(c: i64, local_a: u32) -> Vec<WasmInstr> {
    if c == 0 {
        return vec![];
    }
    if c >= 32 {
        return vec![ni(OP_DROP), instr(OP_I32_CONST, 0)];
    }
    let q = c / 8;
    if q > 0 {
        let mut out = vec![instr(OP_LOCAL_SET, local_a as i64)];
        out.extend(expand_shl(c, local_a));
        return out;
    }
    let a = local_a as i64;
    let mut out = Vec::new();
    for _ in 0..c {
        out.push(instr(OP_LOCAL_TEE, a));
        out.push(instr(OP_LOCAL_GET, a));
        out.push(ni(OP_I32_ADD));
    }
    out
}

/// Expand unsigned `x >> C`. Uses memory for byte-aligned part.
/// Precondition: x is on the stack. Postcondition: result on stack.
fn expand_shr_u(c: i64, local_a: u32) -> Vec<WasmInstr> {
    if c >= 32 {
        return vec![ni(OP_DROP), instr(OP_I32_CONST, 0)];
    }
    let q = c / 8;
    let r = c % 8;
    let a = local_a as i64;
    if q > 0 {
        let mut out = vec![
            instr(OP_LOCAL_SET, a),
            instr(OP_I32_CONST, SCRATCH_ADDR),
            instr(OP_LOCAL_GET, a),
            instr(OP_I32_STORE, [0, 0]),
            instr(OP_I32_CONST, SCRATCH_ADDR),
            instr(OP_I32_CONST, 0),
            instr(OP_I32_STORE, [0, 4]),
            instr(OP_I32_CONST, SCRATCH_ADDR),
            instr(OP_I32_LOAD, [0, q]),
        ];
        if r > 0 {
            out.extend(expand_div_u(1 << r, local_a));
        }
        return out;
    }
    expand_div_u(1 << c, local_a)
}

/// Emit instructions for byte-level bitwise op with a compile-time mask.
fn emit_byte_bitop(
    op: BitOp,
    mask_byte: i64,
    local_byte: u32,
    local_result: u32,
) -> Vec<WasmInstr> {
    let lb = local_byte as i64;
    let lr = local_result as i64;
    let mut out = vec![instr(OP_I32_CONST, 0), instr(OP_LOCAL_SET, lr)];

    for i in (0..8).rev() {
        let mb = (mask_byte >> i) & 1;
        let add_if_set = matches!(
            op,
            BitOp::And if mb == 1
        ) || matches!(op, BitOp::Or if mb == 0)
            || matches!(op, BitOp::Xor if mb == 0);
        let always_add = matches!(op, BitOp::Or if mb == 1);
        let add_if_clear = matches!(op, BitOp::Xor if mb == 1);
        let val = 1i64 << i;

        if add_if_clear {
            out.extend_from_slice(&[
                instr(OP_LOCAL_GET, lb),
                instr(OP_I32_CONST, val),
                ni(OP_I32_GE_U),
                instr(OP_IF, 0x40),
                instr(OP_LOCAL_GET, lb),
                instr(OP_I32_CONST, val),
                ni(OP_I32_SUB),
                instr(OP_LOCAL_SET, lb),
                ni(OP_ELSE),
                instr(OP_LOCAL_GET, lr),
                instr(OP_I32_CONST, val),
                ni(OP_I32_ADD),
                instr(OP_LOCAL_SET, lr),
                ni(OP_END),
            ]);
        } else {
            out.extend_from_slice(&[
                instr(OP_BLOCK, 0x40),
                instr(OP_LOCAL_GET, lb),
                instr(OP_I32_CONST, val),
                ni(OP_I32_LT_U),
                instr(OP_BR_IF, 0),
                instr(OP_LOCAL_GET, lb),
                instr(OP_I32_CONST, val),
                ni(OP_I32_SUB),
                instr(OP_LOCAL_SET, lb),
            ]);
            if add_if_set {
                out.extend_from_slice(&[
                    instr(OP_LOCAL_GET, lr),
                    instr(OP_I32_CONST, val),
                    ni(OP_I32_ADD),
                    instr(OP_LOCAL_SET, lr),
                ]);
            }
            out.push(ni(OP_END));
            if always_add {
                out.extend_from_slice(&[
                    instr(OP_LOCAL_GET, lr),
                    instr(OP_I32_CONST, val),
                    ni(OP_I32_ADD),
                    instr(OP_LOCAL_SET, lr),
                ]);
            }
        }
    }
    out
}

/// General 32-bit bitwise op with a compile-time constant (AND/OR/XOR).
fn expand_bitop_general(op: BitOp, c: i64, local_a: u32) -> Vec<WasmInstr> {
    let c = c & 0xFFFFFFFF;
    let c_bytes = [c as u8, (c >> 8) as u8, (c >> 16) as u8, (c >> 24) as u8];
    let x = local_a as i64;
    let lb = (local_a + 1) as i64;
    let lr = (local_a + 2) as i64;

    let mut out = vec![
        instr(OP_LOCAL_SET, x),
        instr(OP_I32_CONST, SCRATCH_ADDR),
        instr(OP_LOCAL_GET, x),
        instr(OP_I32_STORE, [0, 0]),
    ];

    for b in 0..4u32 {
        let cb = c_bytes[b as usize];
        match op {
            BitOp::And => {
                if cb == 0xFF {
                    continue;
                }
                if cb == 0x00 {
                    out.extend_from_slice(&[
                        instr(OP_I32_CONST, SCRATCH_ADDR),
                        instr(OP_I32_CONST, 0),
                        instr(OP_I32_STORE8, [0, b as i64]),
                    ]);
                    continue;
                }
            }
            BitOp::Or => {
                if cb == 0x00 {
                    continue;
                }
                if cb == 0xFF {
                    out.extend_from_slice(&[
                        instr(OP_I32_CONST, SCRATCH_ADDR),
                        instr(OP_I32_CONST, 0xFF),
                        instr(OP_I32_STORE8, [0, b as i64]),
                    ]);
                    continue;
                }
            }
            BitOp::Xor => {
                if cb == 0x00 {
                    continue;
                }
                if cb == 0xFF {
                    out.extend_from_slice(&[
                        instr(OP_I32_CONST, SCRATCH_ADDR),
                        instr(OP_I32_LOAD8_U, [0, b as i64]),
                        instr(OP_LOCAL_SET, lb),
                        instr(OP_I32_CONST, SCRATCH_ADDR),
                        instr(OP_I32_CONST, 255),
                        instr(OP_LOCAL_GET, lb),
                        ni(OP_I32_SUB),
                        instr(OP_I32_STORE8, [0, b as i64]),
                    ]);
                    continue;
                }
            }
        }
        out.extend_from_slice(&[
            instr(OP_I32_CONST, SCRATCH_ADDR),
            instr(OP_I32_LOAD8_U, [0, b as i64]),
            instr(OP_LOCAL_SET, lb),
        ]);
        out.extend(emit_byte_bitop(op, cb as i64, local_a + 1, local_a + 2));
        out.extend_from_slice(&[
            instr(OP_I32_CONST, SCRATCH_ADDR),
            instr(OP_LOCAL_GET, lr),
            instr(OP_I32_STORE8, [0, b as i64]),
        ]);
    }
    out.extend_from_slice(&[
        instr(OP_I32_CONST, SCRATCH_ADDR),
        instr(OP_I32_LOAD, [0, 0]),
    ]);
    out
}

/// Expand `x & 1` = x mod 2 (extract least significant bit).
fn expand_and_1(local_a: u32) -> Vec<WasmInstr> {
    let x = local_a as i64;
    let byte = (local_a + 1) as i64;
    vec![
        instr(OP_LOCAL_SET, x),
        instr(OP_I32_CONST, SCRATCH_ADDR),
        instr(OP_LOCAL_GET, x),
        instr(OP_I32_STORE8, [0, 0]),
        instr(OP_I32_CONST, SCRATCH_ADDR),
        instr(OP_I32_LOAD8_U, [0, 0]),
        instr(OP_LOCAL_SET, byte),
        instr(OP_BLOCK, 0x40),
        instr(OP_LOOP, 0x40),
        instr(OP_LOCAL_GET, byte),
        instr(OP_I32_CONST, 2),
        ni(OP_I32_LT_U),
        instr(OP_BR_IF, 1),
        instr(OP_LOCAL_GET, byte),
        instr(OP_I32_CONST, 2),
        ni(OP_I32_SUB),
        instr(OP_LOCAL_SET, byte),
        instr(OP_BR, 0),
        ni(OP_END),
        ni(OP_END),
        instr(OP_LOCAL_GET, byte),
    ]
}

/// Expand `x & 0xFFFFFFFE` = x with lowest bit cleared = x - (x mod 2).
fn expand_and_fffffffe(local_a: u32) -> Vec<WasmInstr> {
    let x = local_a as i64;
    let byte = (local_a + 1) as i64;
    vec![
        instr(OP_LOCAL_SET, x),
        instr(OP_I32_CONST, SCRATCH_ADDR),
        instr(OP_LOCAL_GET, x),
        instr(OP_I32_STORE8, [0, 0]),
        instr(OP_I32_CONST, SCRATCH_ADDR),
        instr(OP_I32_LOAD8_U, [0, 0]),
        instr(OP_LOCAL_SET, byte),
        instr(OP_BLOCK, 0x40),
        instr(OP_LOOP, 0x40),
        instr(OP_LOCAL_GET, byte),
        instr(OP_I32_CONST, 2),
        ni(OP_I32_LT_U),
        instr(OP_BR_IF, 1),
        instr(OP_LOCAL_GET, byte),
        instr(OP_I32_CONST, 2),
        ni(OP_I32_SUB),
        instr(OP_LOCAL_SET, byte),
        instr(OP_BR, 0),
        ni(OP_END),
        ni(OP_END),
        instr(OP_LOCAL_GET, x),
        instr(OP_LOCAL_GET, byte),
        ni(OP_I32_SUB),
    ]
}

/// Expand `x & 0x7FFFFFFE` using conditional subtraction.
fn expand_and_7ffffffe_v2(local_a: u32) -> Vec<WasmInstr> {
    let x = local_a as i64;
    let byte = (local_a + 1) as i64;
    let mut out = vec![instr(OP_LOCAL_SET, x)];
    // Clear bit 0
    out.extend_from_slice(&[
        instr(OP_I32_CONST, SCRATCH_ADDR),
        instr(OP_LOCAL_GET, x),
        instr(OP_I32_STORE8, [0, 0]),
        instr(OP_I32_CONST, SCRATCH_ADDR),
        instr(OP_I32_LOAD8_U, [0, 0]),
        instr(OP_LOCAL_SET, byte),
        instr(OP_BLOCK, 0x40),
        instr(OP_LOOP, 0x40),
        instr(OP_LOCAL_GET, byte),
        instr(OP_I32_CONST, 2),
        ni(OP_I32_LT_U),
        instr(OP_BR_IF, 1),
        instr(OP_LOCAL_GET, byte),
        instr(OP_I32_CONST, 2),
        ni(OP_I32_SUB),
        instr(OP_LOCAL_SET, byte),
        instr(OP_BR, 0),
        ni(OP_END),
        ni(OP_END),
        instr(OP_LOCAL_GET, x),
        instr(OP_LOCAL_GET, byte),
        ni(OP_I32_SUB),
        instr(OP_LOCAL_SET, x),
    ]);
    // Clear bit 31
    out.extend_from_slice(&[
        instr(OP_BLOCK, 0x40),
        instr(OP_LOCAL_GET, x),
        instr(OP_I32_CONST, -2147483648_i64),
        ni(OP_I32_LT_U),
        instr(OP_BR_IF, 0),
        instr(OP_LOCAL_GET, x),
        instr(OP_I32_CONST, -2147483648_i64),
        ni(OP_I32_SUB),
        instr(OP_LOCAL_SET, x),
        ni(OP_END),
    ]);
    out.push(instr(OP_LOCAL_GET, x));
    out
}

/// Expand `x & C` for arbitrary C.
fn expand_and_general(c: i64, local_a: u32) -> Vec<WasmInstr> {
    let c = c & 0xFFFFFFFF;
    match c {
        0xFFFFFFFE => expand_and_fffffffe(local_a),
        1 => expand_and_1(local_a),
        _ => expand_bitop_general(BitOp::And, c, local_a),
    }
}

/// Expand `x ^ C`.
fn expand_xor(c: i64, local_a: u32) -> Vec<WasmInstr> {
    let c = c & 0xFFFFFFFF;
    if c == 0xFFFFFFFF {
        return vec![
            instr(OP_LOCAL_SET, local_a as i64),
            instr(OP_I32_CONST, -1),
            instr(OP_LOCAL_GET, local_a as i64),
            ni(OP_I32_SUB),
        ];
    }
    if c == 1 {
        let x = local_a as i64;
        let bit = (local_a + 1) as i64;
        let mut out = vec![instr(OP_LOCAL_TEE, x)];
        out.extend(expand_and_1(local_a));
        out.extend_from_slice(&[
            instr(OP_LOCAL_SET, bit),
            instr(OP_LOCAL_GET, x),
            instr(OP_I32_CONST, 1),
            ni(OP_I32_ADD),
            instr(OP_LOCAL_GET, bit),
            instr(OP_LOCAL_GET, bit),
            ni(OP_I32_ADD),
            ni(OP_I32_SUB),
        ]);
        return out;
    }
    expand_bitop_general(BitOp::Xor, c, local_a)
}

/// Expand `x | C`.
fn expand_or(c: i64, local_a: u32) -> Vec<WasmInstr> {
    expand_bitop_general(BitOp::Or, c, local_a)
}

/// Expand signed `x >> C` using memory byte extraction.
fn expand_shr_s(c: i64, local_a: u32) -> Vec<WasmInstr> {
    let a = local_a as i64;
    if c == 0 {
        return vec![];
    }
    if c >= 32 {
        return vec![
            instr(OP_LOCAL_SET, a),
            instr(OP_I32_CONST, -1),
            instr(OP_I32_CONST, 0),
            instr(OP_LOCAL_GET, a),
            ni(OP_I32_LT_S),
            ni(OP_SELECT),
        ];
    }
    let q = c / 8;
    let r = c % 8;

    if r == 0 {
        return match q {
            3 => vec![
                instr(OP_LOCAL_SET, a),
                instr(OP_I32_CONST, SCRATCH_ADDR),
                instr(OP_LOCAL_GET, a),
                instr(OP_I32_STORE, [0, 0]),
                instr(OP_I32_CONST, SCRATCH_ADDR),
                instr(OP_I32_LOAD8_S, [0, 3]),
            ],
            2 => vec![
                instr(OP_LOCAL_SET, a),
                instr(OP_I32_CONST, SCRATCH_ADDR),
                instr(OP_LOCAL_GET, a),
                instr(OP_I32_STORE, [0, 0]),
                instr(OP_I32_CONST, SCRATCH_ADDR),
                instr(OP_I32_LOAD16_S, [0, 2]),
            ],
            1 => {
                let tmp = (local_a + 1) as i64;
                vec![
                    instr(OP_LOCAL_SET, a),
                    instr(OP_I32_CONST, SCRATCH_ADDR),
                    instr(OP_LOCAL_GET, a),
                    instr(OP_I32_STORE, [0, 0]),
                    instr(OP_I32_CONST, SCRATCH_ADDR),
                    instr(OP_I32_CONST, 0),
                    instr(OP_I32_STORE, [0, 4]),
                    instr(OP_I32_CONST, SCRATCH_ADDR),
                    instr(OP_I32_LOAD, [0, 1]),
                    instr(OP_LOCAL_SET, tmp),
                    instr(OP_BLOCK, 0x40),
                    instr(OP_LOCAL_GET, tmp),
                    instr(OP_I32_CONST, 0x800000),
                    ni(OP_I32_LT_U),
                    instr(OP_BR_IF, 0),
                    instr(OP_LOCAL_GET, tmp),
                    instr(OP_I32_CONST, 0x1000000),
                    ni(OP_I32_SUB),
                    instr(OP_LOCAL_SET, tmp),
                    ni(OP_END),
                    instr(OP_LOCAL_GET, tmp),
                ]
            }
            _ => vec![],
        };
    }

    if q > 0 {
        let mut out = vec![
            instr(OP_LOCAL_SET, a),
            instr(OP_I32_CONST, SCRATCH_ADDR),
            instr(OP_LOCAL_GET, a),
            instr(OP_I32_STORE, [0, 0]),
            instr(OP_I32_CONST, SCRATCH_ADDR),
            instr(OP_I32_CONST, 0),
            instr(OP_I32_STORE, [0, 4]),
            instr(OP_I32_CONST, SCRATCH_ADDR),
            instr(OP_I32_LOAD, [0, q]),
        ];
        out.extend(expand_div_u(1 << r, local_a));
        return out;
    }
    expand_div_u(1 << c, local_a)
}

// ===================================================================== //
//  Runtime (variable-operand) expansion functions                        //
// ===================================================================== //

/// Runtime SHL: loop `b` times, doubling `a` each iteration.
fn expand_runtime_shl(temp_base: u32) -> Vec<WasmInstr> {
    let a = temp_base as i64;
    let b = (temp_base + 1) as i64;
    vec![
        instr(OP_LOCAL_SET, b),
        instr(OP_LOCAL_SET, a),
        instr(OP_BLOCK, 0x40),
        instr(OP_LOOP, 0x40),
        instr(OP_LOCAL_GET, b),
        ni(OP_I32_EQZ),
        instr(OP_BR_IF, 1),
        instr(OP_LOCAL_GET, a),
        instr(OP_LOCAL_GET, a),
        ni(OP_I32_ADD),
        instr(OP_LOCAL_SET, a),
        instr(OP_LOCAL_GET, b),
        instr(OP_I32_CONST, 1),
        ni(OP_I32_SUB),
        instr(OP_LOCAL_SET, b),
        instr(OP_BR, 0),
        ni(OP_END),
        ni(OP_END),
        instr(OP_LOCAL_GET, a),
    ]
}

/// Runtime SHR_U: loop `b` times, halving `a` via div-by-2 each iteration.
fn expand_runtime_shr_u(temp_base: u32) -> Vec<WasmInstr> {
    let a = temp_base as i64;
    let b = (temp_base + 1) as i64;
    let q = (temp_base + 2) as i64;
    vec![
        instr(OP_LOCAL_SET, b),
        instr(OP_LOCAL_SET, a),
        instr(OP_BLOCK, 0x40),
        instr(OP_LOOP, 0x40),
        instr(OP_LOCAL_GET, b),
        ni(OP_I32_EQZ),
        instr(OP_BR_IF, 1),
        instr(OP_I32_CONST, 0),
        instr(OP_LOCAL_SET, q),
        instr(OP_BLOCK, 0x40),
        instr(OP_LOOP, 0x40),
        instr(OP_LOCAL_GET, a),
        instr(OP_I32_CONST, 2),
        ni(OP_I32_LT_U),
        instr(OP_BR_IF, 1),
        instr(OP_LOCAL_GET, a),
        instr(OP_I32_CONST, 2),
        ni(OP_I32_SUB),
        instr(OP_LOCAL_SET, a),
        instr(OP_LOCAL_GET, q),
        instr(OP_I32_CONST, 1),
        ni(OP_I32_ADD),
        instr(OP_LOCAL_SET, q),
        instr(OP_BR, 0),
        ni(OP_END),
        ni(OP_END),
        instr(OP_LOCAL_GET, q),
        instr(OP_LOCAL_SET, a),
        instr(OP_LOCAL_GET, b),
        instr(OP_I32_CONST, 1),
        ni(OP_I32_SUB),
        instr(OP_LOCAL_SET, b),
        instr(OP_BR, 0),
        ni(OP_END),
        ni(OP_END),
        instr(OP_LOCAL_GET, a),
    ]
}

/// Runtime SHR_S: treat as SHR_U (correct for non-negative values).
fn expand_runtime_shr_s(temp_base: u32) -> Vec<WasmInstr> {
    expand_runtime_shr_u(temp_base)
}

/// Runtime MUL: `result = 0; while b > 0: result += a; b--`.
fn expand_runtime_mul(temp_base: u32) -> Vec<WasmInstr> {
    let a = temp_base as i64;
    let b = (temp_base + 1) as i64;
    let r = (temp_base + 2) as i64;
    vec![
        instr(OP_LOCAL_SET, b),
        instr(OP_LOCAL_SET, a),
        instr(OP_I32_CONST, 0),
        instr(OP_LOCAL_SET, r),
        instr(OP_BLOCK, 0x40),
        instr(OP_LOOP, 0x40),
        instr(OP_LOCAL_GET, b),
        ni(OP_I32_EQZ),
        instr(OP_BR_IF, 1),
        instr(OP_LOCAL_GET, r),
        instr(OP_LOCAL_GET, a),
        ni(OP_I32_ADD),
        instr(OP_LOCAL_SET, r),
        instr(OP_LOCAL_GET, b),
        instr(OP_I32_CONST, 1),
        ni(OP_I32_SUB),
        instr(OP_LOCAL_SET, b),
        instr(OP_BR, 0),
        ni(OP_END),
        ni(OP_END),
        instr(OP_LOCAL_GET, r),
    ]
}

/// Runtime DIV_U: subtraction loop.
fn expand_runtime_div_u(temp_base: u32) -> Vec<WasmInstr> {
    let a = temp_base as i64;
    let b = (temp_base + 1) as i64;
    let q = (temp_base + 2) as i64;
    vec![
        instr(OP_LOCAL_SET, b),
        instr(OP_LOCAL_SET, a),
        instr(OP_I32_CONST, 0),
        instr(OP_LOCAL_SET, q),
        instr(OP_BLOCK, 0x40),
        instr(OP_LOOP, 0x40),
        instr(OP_LOCAL_GET, a),
        instr(OP_LOCAL_GET, b),
        ni(OP_I32_LT_U),
        instr(OP_BR_IF, 1),
        instr(OP_LOCAL_GET, a),
        instr(OP_LOCAL_GET, b),
        ni(OP_I32_SUB),
        instr(OP_LOCAL_SET, a),
        instr(OP_LOCAL_GET, q),
        instr(OP_I32_CONST, 1),
        ni(OP_I32_ADD),
        instr(OP_LOCAL_SET, q),
        instr(OP_BR, 0),
        ni(OP_END),
        ni(OP_END),
        instr(OP_LOCAL_GET, q),
    ]
}

/// Runtime REM_U/REM_S: subtraction loop returning remainder.
fn expand_runtime_rem(temp_base: u32) -> Vec<WasmInstr> {
    let a = temp_base as i64;
    let b = (temp_base + 1) as i64;
    vec![
        instr(OP_LOCAL_SET, b),
        instr(OP_LOCAL_SET, a),
        instr(OP_BLOCK, 0x40),
        instr(OP_LOOP, 0x40),
        instr(OP_LOCAL_GET, a),
        instr(OP_LOCAL_GET, b),
        ni(OP_I32_LT_U),
        instr(OP_BR_IF, 1),
        instr(OP_LOCAL_GET, a),
        instr(OP_LOCAL_GET, b),
        ni(OP_I32_SUB),
        instr(OP_LOCAL_SET, a),
        instr(OP_BR, 0),
        ni(OP_END),
        ni(OP_END),
        instr(OP_LOCAL_GET, a),
    ]
}

/// Runtime DIV_S: signed division via abs + unsigned div.
fn expand_runtime_div_s(temp_base: u32) -> Vec<WasmInstr> {
    let a = temp_base as i64;
    let b = (temp_base + 1) as i64;
    let q = (temp_base + 2) as i64;
    let neg = (temp_base + 3) as i64;
    vec![
        instr(OP_LOCAL_SET, b),
        instr(OP_LOCAL_SET, a),
        instr(OP_LOCAL_GET, a),
        instr(OP_I32_CONST, 0),
        ni(OP_I32_LT_S),
        instr(OP_LOCAL_GET, b),
        instr(OP_I32_CONST, 0),
        ni(OP_I32_LT_S),
        ni(OP_I32_NE),
        instr(OP_LOCAL_SET, neg),
        instr(OP_BLOCK, 0x40),
        instr(OP_LOCAL_GET, a),
        instr(OP_I32_CONST, 0),
        ni(OP_I32_GE_S),
        instr(OP_BR_IF, 0),
        instr(OP_I32_CONST, 0),
        instr(OP_LOCAL_GET, a),
        ni(OP_I32_SUB),
        instr(OP_LOCAL_SET, a),
        ni(OP_END),
        instr(OP_BLOCK, 0x40),
        instr(OP_LOCAL_GET, b),
        instr(OP_I32_CONST, 0),
        ni(OP_I32_GE_S),
        instr(OP_BR_IF, 0),
        instr(OP_I32_CONST, 0),
        instr(OP_LOCAL_GET, b),
        ni(OP_I32_SUB),
        instr(OP_LOCAL_SET, b),
        ni(OP_END),
        instr(OP_I32_CONST, 0),
        instr(OP_LOCAL_SET, q),
        instr(OP_BLOCK, 0x40),
        instr(OP_LOOP, 0x40),
        instr(OP_LOCAL_GET, a),
        instr(OP_LOCAL_GET, b),
        ni(OP_I32_LT_U),
        instr(OP_BR_IF, 1),
        instr(OP_LOCAL_GET, a),
        instr(OP_LOCAL_GET, b),
        ni(OP_I32_SUB),
        instr(OP_LOCAL_SET, a),
        instr(OP_LOCAL_GET, q),
        instr(OP_I32_CONST, 1),
        ni(OP_I32_ADD),
        instr(OP_LOCAL_SET, q),
        instr(OP_BR, 0),
        ni(OP_END),
        ni(OP_END),
        instr(OP_BLOCK, 0x40),
        instr(OP_LOCAL_GET, neg),
        ni(OP_I32_EQZ),
        instr(OP_BR_IF, 0),
        instr(OP_I32_CONST, 0),
        instr(OP_LOCAL_GET, q),
        ni(OP_I32_SUB),
        instr(OP_LOCAL_SET, q),
        ni(OP_END),
        instr(OP_LOCAL_GET, q),
    ]
}

/// Runtime XOR: approximate as NE (0 or 1).
fn expand_runtime_xor() -> Vec<WasmInstr> {
    vec![ni(OP_I32_NE)]
}

/// Runtime AND: `select(a, 0, b)`.
fn expand_runtime_and(temp_base: u32) -> Vec<WasmInstr> {
    let a = temp_base as i64;
    vec![
        instr(OP_LOCAL_SET, a),
        instr(OP_I32_CONST, 0),
        instr(OP_LOCAL_GET, a),
        ni(OP_SELECT),
    ]
}

/// Runtime OR: `select(1, a, b)`.
fn expand_runtime_or(temp_base: u32) -> Vec<WasmInstr> {
    let a = temp_base as i64;
    vec![
        instr(OP_LOCAL_SET, a),
        instr(OP_I32_CONST, 1),
        instr(OP_LOCAL_GET, a),
        ni(OP_SELECT),
    ]
}

/// Runtime ROTL: loop `b` times rotating left by 1.
fn expand_runtime_rotl(temp_base: u32) -> Vec<WasmInstr> {
    let a = temp_base as i64;
    let b = (temp_base + 1) as i64;
    let bit = (temp_base + 2) as i64;
    vec![
        instr(OP_LOCAL_SET, b),
        instr(OP_LOCAL_SET, a),
        instr(OP_BLOCK, 0x40),
        instr(OP_LOOP, 0x40),
        instr(OP_LOCAL_GET, b),
        instr(OP_I32_CONST, 32),
        ni(OP_I32_LT_U),
        instr(OP_BR_IF, 1),
        instr(OP_LOCAL_GET, b),
        instr(OP_I32_CONST, 32),
        ni(OP_I32_SUB),
        instr(OP_LOCAL_SET, b),
        instr(OP_BR, 0),
        ni(OP_END),
        ni(OP_END),
        instr(OP_BLOCK, 0x40),
        instr(OP_LOOP, 0x40),
        instr(OP_LOCAL_GET, b),
        ni(OP_I32_EQZ),
        instr(OP_BR_IF, 1),
        instr(OP_LOCAL_GET, a),
        instr(OP_I32_CONST, 0),
        ni(OP_I32_LT_S),
        instr(OP_LOCAL_SET, bit),
        instr(OP_LOCAL_GET, a),
        instr(OP_LOCAL_GET, a),
        ni(OP_I32_ADD),
        instr(OP_LOCAL_GET, bit),
        ni(OP_I32_ADD),
        instr(OP_LOCAL_SET, a),
        instr(OP_LOCAL_GET, b),
        instr(OP_I32_CONST, 1),
        ni(OP_I32_SUB),
        instr(OP_LOCAL_SET, b),
        instr(OP_BR, 0),
        ni(OP_END),
        ni(OP_END),
        instr(OP_LOCAL_GET, a),
    ]
}

/// Runtime ROTR: convert to ROTL by `(32 - b%32)%32`.
fn expand_runtime_rotr(temp_base: u32) -> Vec<WasmInstr> {
    let a = temp_base as i64;
    let b = (temp_base + 1) as i64;
    let bit = (temp_base + 2) as i64;
    vec![
        instr(OP_LOCAL_SET, b),
        instr(OP_LOCAL_SET, a),
        instr(OP_BLOCK, 0x40),
        instr(OP_LOOP, 0x40),
        instr(OP_LOCAL_GET, b),
        instr(OP_I32_CONST, 32),
        ni(OP_I32_LT_U),
        instr(OP_BR_IF, 1),
        instr(OP_LOCAL_GET, b),
        instr(OP_I32_CONST, 32),
        ni(OP_I32_SUB),
        instr(OP_LOCAL_SET, b),
        instr(OP_BR, 0),
        ni(OP_END),
        ni(OP_END),
        instr(OP_I32_CONST, 32),
        instr(OP_LOCAL_GET, b),
        ni(OP_I32_SUB),
        instr(OP_LOCAL_SET, b),
        instr(OP_BLOCK, 0x40),
        instr(OP_LOOP, 0x40),
        instr(OP_LOCAL_GET, b),
        instr(OP_I32_CONST, 32),
        ni(OP_I32_LT_U),
        instr(OP_BR_IF, 1),
        instr(OP_LOCAL_GET, b),
        instr(OP_I32_CONST, 32),
        ni(OP_I32_SUB),
        instr(OP_LOCAL_SET, b),
        instr(OP_BR, 0),
        ni(OP_END),
        ni(OP_END),
        instr(OP_BLOCK, 0x40),
        instr(OP_LOOP, 0x40),
        instr(OP_LOCAL_GET, b),
        ni(OP_I32_EQZ),
        instr(OP_BR_IF, 1),
        instr(OP_LOCAL_GET, a),
        instr(OP_I32_CONST, 0),
        ni(OP_I32_LT_S),
        instr(OP_LOCAL_SET, bit),
        instr(OP_LOCAL_GET, a),
        instr(OP_LOCAL_GET, a),
        ni(OP_I32_ADD),
        instr(OP_LOCAL_GET, bit),
        ni(OP_I32_ADD),
        instr(OP_LOCAL_SET, a),
        instr(OP_LOCAL_GET, b),
        instr(OP_I32_CONST, 1),
        ni(OP_I32_SUB),
        instr(OP_LOCAL_SET, b),
        instr(OP_BR, 0),
        ni(OP_END),
        ni(OP_END),
        instr(OP_LOCAL_GET, a),
    ]
}

// ===================================================================== //
//  Main lowering entry point                                             //
// ===================================================================== //

/// Lower hard-to-simulate instructions in a function body.
///
/// Returns a new `FuncBody` with hard ops replaced by basic instruction sequences.
/// Adds temporary locals as needed.
///
/// # Arguments
/// * `func` - The original function body.
/// * `num_params` - Number of function parameters (local indices start after these).
pub fn lower_hard_ops(func: &FuncBody, num_params: u32) -> FuncBody {
    let instrs = &func.instructions;

    // Check if any lowering is needed
    let needs_lowering = instrs.iter().any(|ins| {
        opcode_in(ins.opcode, LOWERABLE_BINOPS) || opcode_in(ins.opcode, LOWERABLE_UNARY)
    });
    if !needs_lowering {
        return func.clone();
    }

    let const_locals = find_const_locals(instrs);

    let total_existing = num_params + func.num_locals;
    let temp_base = total_existing;

    let new_locals = {
        let mut l = func.locals.clone();
        l.push((NUM_TEMPS, VALTYPE_I32));
        l
    };
    let new_num_locals = func.num_locals + NUM_TEMPS;

    let mut new_instrs = Vec::with_capacity(instrs.len() * 4);
    let mut i = 0;
    let mut lowered_count: usize = 0;

    while i < instrs.len() {
        let ins = &instrs[i];

        // Pattern 1: i32.const C + binop
        // Pattern 2: local.get X (known const) + binop
        let mut const_val: Option<i64> = None;
        let mut binop_idx: Option<usize> = None;

        if ins.opcode == OP_I32_CONST
            && i + 1 < instrs.len()
            && opcode_in(instrs[i + 1].opcode, LOWERABLE_BINOPS)
        {
            const_val = Some(ins.immediates[0] & 0xFFFFFFFF);
            binop_idx = Some(i + 1);
        } else if ins.opcode == OP_LOCAL_GET {
            let local_idx = ins.immediates[0] as u32;
            if let Some(&cv) = const_locals.get(&local_idx)
                && i + 1 < instrs.len() && opcode_in(instrs[i + 1].opcode, LOWERABLE_BINOPS) {
                    const_val = Some(cv & 0xFFFFFFFF);
                    binop_idx = Some(i + 1);
                }
        }

        if let (Some(cv), Some(bi)) = (const_val, binop_idx) {
            let op = instrs[bi].opcode;
            let local_a = temp_base;
            let expansion: Option<Vec<WasmInstr>> = match op {
                OP_I32_AND => {
                    let c = cv & 0xFFFFFFFF;
                    match c {
                        255 => Some(expand_and_255(local_a)),
                        0xFFFFFFFE => Some(expand_and_fffffffe(local_a)),
                        0x7FFFFFFE => Some(expand_and_7ffffffe_v2(local_a)),
                        _ => Some(expand_and_general(cv, local_a)),
                    }
                }
                OP_I32_MUL => {
                    let mut exp = vec![instr(OP_LOCAL_SET, local_a as i64)];
                    exp.extend(expand_mul(cv, local_a));
                    Some(exp)
                }
                OP_I32_DIV_U => Some(expand_div_u(cv, local_a)),
                OP_I32_REM_U | OP_I32_REM_S => Some(expand_rem_u(cv, local_a)),
                OP_I32_DIV_S => Some(expand_div_s(cv, local_a)),
                OP_I32_SHL => Some(expand_shl_from_stack(cv, local_a)),
                OP_I32_SHR_U => Some(expand_shr_u(cv, local_a)),
                OP_I32_SHR_S => Some(expand_shr_s(cv, local_a)),
                OP_I32_XOR => Some(expand_xor(cv, local_a)),
                OP_I32_OR => Some(expand_or(cv, local_a)),
                OP_I32_ROTL => Some(expand_rotl_const(cv, local_a)),
                OP_I32_ROTR => Some(expand_rotr_const(cv, local_a)),
                _ => None,
            };

            if let Some(exp) = expansion {
                new_instrs.extend(exp);
                i = bi + 1;
                lowered_count += 1;
                continue;
            }
        }

        // Runtime lowering (no preceding const)
        let runtime_exp: Option<Vec<WasmInstr>> = match ins.opcode {
            OP_I32_SHL => Some(expand_runtime_shl(temp_base)),
            OP_I32_SHR_U => Some(expand_runtime_shr_u(temp_base)),
            OP_I32_SHR_S => Some(expand_runtime_shr_s(temp_base)),
            OP_I32_MUL => Some(expand_runtime_mul(temp_base)),
            OP_I32_DIV_U => Some(expand_runtime_div_u(temp_base)),
            OP_I32_REM_U | OP_I32_REM_S => Some(expand_runtime_rem(temp_base)),
            OP_I32_DIV_S => Some(expand_runtime_div_s(temp_base)),
            OP_I32_XOR => Some(expand_runtime_xor()),
            OP_I32_AND => Some(expand_runtime_and(temp_base)),
            OP_I32_OR => Some(expand_runtime_or(temp_base)),
            OP_I32_ROTL => Some(expand_runtime_rotl(temp_base)),
            OP_I32_ROTR => Some(expand_runtime_rotr(temp_base)),
            OP_I32_CLZ => Some(expand_clz(temp_base)),
            OP_I32_CTZ => Some(expand_ctz(temp_base)),
            OP_I32_POPCNT => Some(expand_popcnt(temp_base)),
            OP_I32_EXTEND8_S => Some(expand_extend8_s(temp_base)),
            OP_I32_EXTEND16_S => Some(expand_extend16_s(temp_base)),
            _ => None,
        };

        if let Some(exp) = runtime_exp {
            new_instrs.extend(exp);
            i += 1;
            lowered_count += 1;
            continue;
        }

        new_instrs.push(ins.clone());
        i += 1;
    }

    if lowered_count > 0 {
        log::debug!("  Lowered {lowered_count} hard ops");
    }

    FuncBody {
        locals: new_locals,
        num_locals: new_num_locals,
        instructions: new_instrs,
    }
}

// ===================================================================== //
//  Verification                                                          //
// ===================================================================== //

/// Check that a function body uses only basic ops.
///
/// Returns a map from unsupported op names to their occurrence count.
/// Empty map means all instructions are basic.
pub fn check_basic_only(func: &FuncBody) -> HashMap<String, usize> {
    let mut bad: HashMap<String, usize> = HashMap::new();
    for ins in &func.instructions {
        if !opcode_in(ins.opcode, BASIC_OPS) {
            let name = WASM_OP_NAMES
                .get(&ins.opcode)
                .map(|s| (*s).to_owned())
                .unwrap_or_else(|| format!("0x{:02x}", ins.opcode));
            *bad.entry(name).or_insert(0) += 1;
        }
    }
    bad
}

// ===================================================================== //
//  Tests                                                                 //
// ===================================================================== //

#[cfg(test)]
mod tests {
    use super::*;

    fn make_func(instrs: Vec<WasmInstr>) -> FuncBody {
        FuncBody {
            locals: vec![],
            num_locals: 0,
            instructions: instrs,
        }
    }

    fn make_func_with_locals(locals: Vec<(u32, u8)>, instrs: Vec<WasmInstr>) -> FuncBody {
        let num_locals = locals.iter().map(|(c, _)| *c).sum();
        FuncBody {
            locals,
            num_locals,
            instructions: instrs,
        }
    }

    #[test]
    fn test_lower_mul_const() {
        // i32.const 3; i32.mul → should be lowered
        let func = make_func(vec![
            instr(OP_LOCAL_GET, 0), // x
            instr(OP_I32_CONST, 3), // C=3
            ni(OP_I32_MUL),
            ni(OP_END),
        ]);
        let lowered = lower_hard_ops(&func, 1);
        // Should not contain MUL
        assert!(lowered.instructions.iter().all(|i| i.opcode != OP_I32_MUL));
        // Should contain ADD (from multiplication expansion)
        assert!(lowered.instructions.iter().any(|i| i.opcode == OP_I32_ADD));
        // Should have extra temp locals
        assert_eq!(lowered.num_locals, NUM_TEMPS);
    }

    #[test]
    fn test_lower_div_u_const() {
        let func = make_func(vec![
            instr(OP_LOCAL_GET, 0),
            instr(OP_I32_CONST, 4),
            ni(OP_I32_DIV_U),
            ni(OP_END),
        ]);
        let lowered = lower_hard_ops(&func, 1);
        assert!(
            lowered
                .instructions
                .iter()
                .all(|i| i.opcode != OP_I32_DIV_U)
        );
        assert!(lowered.instructions.iter().any(|i| i.opcode == OP_I32_SUB));
    }

    #[test]
    fn test_lower_runtime_mul() {
        // MUL without preceding const → runtime loop
        let func = make_func(vec![
            instr(OP_LOCAL_GET, 0),
            instr(OP_LOCAL_GET, 1),
            ni(OP_I32_MUL),
            ni(OP_END),
        ]);
        let lowered = lower_hard_ops(&func, 2);
        assert!(lowered.instructions.iter().all(|i| i.opcode != OP_I32_MUL));
        assert!(lowered.instructions.iter().any(|i| i.opcode == OP_I32_ADD));
    }

    #[test]
    fn test_lower_and_255() {
        let func = make_func(vec![
            instr(OP_LOCAL_GET, 0),
            instr(OP_I32_CONST, 255),
            ni(OP_I32_AND),
            ni(OP_END),
        ]);
        let lowered = lower_hard_ops(&func, 1);
        assert!(lowered.instructions.iter().all(|i| i.opcode != OP_I32_AND));
        assert!(
            lowered
                .instructions
                .iter()
                .any(|i| i.opcode == OP_I32_STORE8)
        );
        assert!(
            lowered
                .instructions
                .iter()
                .any(|i| i.opcode == OP_I32_LOAD8_U)
        );
    }

    #[test]
    fn test_lower_unary_clz() {
        let func = make_func(vec![instr(OP_LOCAL_GET, 0), ni(OP_I32_CLZ), ni(OP_END)]);
        let lowered = lower_hard_ops(&func, 1);
        assert!(lowered.instructions.iter().all(|i| i.opcode != OP_I32_CLZ));
    }

    #[test]
    fn test_no_lowering_needed() {
        let func = make_func(vec![
            instr(OP_LOCAL_GET, 0),
            instr(OP_LOCAL_GET, 1),
            ni(OP_I32_ADD),
            ni(OP_END),
        ]);
        let lowered = lower_hard_ops(&func, 2);
        // Should be unchanged (cloned)
        assert_eq!(lowered.instructions.len(), func.instructions.len());
        assert_eq!(lowered.num_locals, 0);
    }

    #[test]
    fn test_check_basic_only_pass() {
        let func = make_func(vec![
            instr(OP_LOCAL_GET, 0),
            instr(OP_I32_CONST, 1),
            ni(OP_I32_ADD),
            ni(OP_END),
        ]);
        let bad = check_basic_only(&func);
        assert!(bad.is_empty());
    }

    #[test]
    fn test_check_basic_only_fail() {
        let func = make_func(vec![
            instr(OP_LOCAL_GET, 0),
            instr(OP_LOCAL_GET, 1),
            ni(OP_I32_MUL),
            ni(OP_END),
        ]);
        let bad = check_basic_only(&func);
        assert_eq!(bad.get("i32.mul"), Some(&1));
    }

    #[test]
    fn test_check_basic_only_after_lowering() {
        let func = make_func(vec![
            instr(OP_LOCAL_GET, 0),
            instr(OP_I32_CONST, 5),
            ni(OP_I32_MUL),
            ni(OP_END),
        ]);
        let lowered = lower_hard_ops(&func, 1);
        let bad = check_basic_only(&lowered);
        assert!(bad.is_empty(), "unexpected hard ops: {bad:?}");
    }

    #[test]
    fn test_find_const_locals() {
        let instrs = vec![
            instr(OP_I32_CONST, 42),
            instr(OP_LOCAL_SET, 1),
            instr(OP_I32_CONST, 42),
            instr(OP_LOCAL_SET, 1),
            instr(OP_LOCAL_GET, 0),
            instr(OP_LOCAL_SET, 2),
        ];
        let cl = find_const_locals(&instrs);
        assert_eq!(cl.get(&1), Some(&42));
        assert!(!cl.contains_key(&2));
    }

    #[test]
    fn test_lower_const_via_local_get() {
        // local.get 2 (which is always const 3) + i32.mul
        let func = make_func_with_locals(
            vec![(1, VALTYPE_I32)],
            vec![
                instr(OP_I32_CONST, 3),
                instr(OP_LOCAL_SET, 2),
                instr(OP_LOCAL_GET, 0),
                instr(OP_LOCAL_GET, 2),
                ni(OP_I32_MUL),
                ni(OP_END),
            ],
        );
        let lowered = lower_hard_ops(&func, 1);
        assert!(lowered.instructions.iter().all(|i| i.opcode != OP_I32_MUL));
    }

    #[test]
    fn test_lower_shl_const() {
        let func = make_func(vec![
            instr(OP_LOCAL_GET, 0),
            instr(OP_I32_CONST, 2),
            ni(OP_I32_SHL),
            ni(OP_END),
        ]);
        let lowered = lower_hard_ops(&func, 1);
        assert!(lowered.instructions.iter().all(|i| i.opcode != OP_I32_SHL));
    }

    #[test]
    fn test_lower_extend8_s() {
        let func = make_func(vec![
            instr(OP_LOCAL_GET, 0),
            ni(OP_I32_EXTEND8_S),
            ni(OP_END),
        ]);
        let lowered = lower_hard_ops(&func, 1);
        assert!(
            lowered
                .instructions
                .iter()
                .all(|i| i.opcode != OP_I32_EXTEND8_S)
        );
        let bad = check_basic_only(&lowered);
        assert!(bad.is_empty(), "unexpected hard ops: {bad:?}");
    }
}
