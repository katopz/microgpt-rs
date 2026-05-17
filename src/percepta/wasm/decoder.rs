// SPDX-License-Identifier: Apache-2.0
// Distilled from Percepta's `transformer-vm` (Apache-2.0 © Percepta).

//! WASM MVP binary decoder.
//!
//! Parses a `.wasm` binary into structured sections and decoded instruction lists.
//! Only the MVP subset is supported (no extensions).
//!
//! Reference: <https://webassembly.github.io/spec/core/binary/>

use std::fmt;

// ===================================================================== //
//  LEB128 helpers                                                        //
// ===================================================================== //

/// Decode an unsigned LEB128 integer. Returns `(value, new_pos)`.
pub fn read_unsigned_leb128(data: &[u8], pos: usize) -> Result<(u64, usize), DecodeError> {
    let mut result: u64 = 0;
    let mut shift: u32 = 0;
    let mut p = pos;
    loop {
        if p >= data.len() {
            return Err(DecodeError::UnexpectedEof("unsigned leb128"));
        }
        let b = data[p];
        p += 1;
        result |= u64::from(b & 0x7F) << shift;
        shift += 7;
        if b & 0x80 == 0 {
            break;
        }
    }
    Ok((result, p))
}

/// Decode a signed LEB128 integer. Returns `(value, new_pos)`.
/// `bits` is the target bit width (typically 32 or 64).
pub fn read_signed_leb128(data: &[u8], pos: usize, bits: u32) -> Result<(i64, usize), DecodeError> {
    let mut result: i64 = 0;
    let mut shift: u32 = 0;
    let mut last_byte: u8;
    let mut p = pos;
    loop {
        if p >= data.len() {
            return Err(DecodeError::UnexpectedEof("signed leb128"));
        }
        let b = data[p];
        p += 1;
        last_byte = b;
        result |= i64::from(b & 0x7F) << shift;
        shift += 7;
        if b & 0x80 == 0 {
            break;
        }
    }
    // Sign extend
    if shift < bits && (last_byte & 0x40) != 0 {
        result |= -(1i64 << shift);
    }
    // Truncate to the specified bit width (two's complement)
    let mask = (1i64 << bits) - 1;
    result &= mask;
    // Re-sign-extend from the truncated width so i64 carries the correct sign
    if bits < 64 {
        let sign_bit = 1i64 << (bits - 1);
        if (result & sign_bit) != 0 {
            result |= !mask;
        }
    }
    Ok((result, p))
}

// ===================================================================== //
//  WASM value types                                                      //
// ===================================================================== //

/// WASM value type discriminator.
pub const VALTYPE_I32: u8 = 0x7F;
pub const VALTYPE_I64: u8 = 0x7E;
pub const VALTYPE_F32: u8 = 0x7D;
pub const VALTYPE_F64: u8 = 0x7C;

// ===================================================================== //
//  Section IDs                                                           //
// ===================================================================== //

pub const SEC_CUSTOM: u8 = 0;
pub const SEC_TYPE: u8 = 1;
pub const SEC_IMPORT: u8 = 2;
pub const SEC_FUNCTION: u8 = 3;
pub const SEC_TABLE: u8 = 4;
pub const SEC_MEMORY: u8 = 5;
pub const SEC_GLOBAL: u8 = 6;
pub const SEC_EXPORT: u8 = 7;
pub const SEC_START: u8 = 8;
pub const SEC_ELEMENT: u8 = 9;
pub const SEC_CODE: u8 = 10;
pub const SEC_DATA: u8 = 11;
pub const SEC_DATACOUNT: u8 = 12;

// ===================================================================== //
//  Wasm instruction representation                                       //
// ===================================================================== //

/// A single decoded wasm instruction.
#[derive(Clone, Debug, PartialEq)]
pub struct WasmInstr {
    /// Opcode byte.
    pub opcode: u8,
    /// Immediate operands (varies by opcode).
    pub immediates: Vec<i64>,
}

impl WasmInstr {
    /// Create a new instruction with no immediates.
    pub fn new(opcode: u8) -> Self {
        Self {
            opcode,
            immediates: Vec::new(),
        }
    }

    /// Create a new instruction with immediates.
    pub fn with_imms(opcode: u8, immediates: impl Into<Vec<i64>>) -> Self {
        Self {
            opcode,
            immediates: immediates.into(),
        }
    }
}

impl fmt::Display for WasmInstr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = WASM_OP_NAMES.get(&self.opcode).unwrap_or("???");
        if self.immediates.is_empty() {
            write!(f, "{name}")
        } else {
            let args: Vec<String> = self.immediates.iter().map(|a| format!("{a}")).collect();
            write!(f, "{name}({})", args.join(", "))
        }
    }
}

// ===================================================================== //
//  Wasm MVP opcode constants                                             //
// ===================================================================== //

// Control flow
pub const OP_UNREACHABLE: u8 = 0x00;
pub const OP_NOP: u8 = 0x01;
pub const OP_BLOCK: u8 = 0x02;
pub const OP_LOOP: u8 = 0x03;
pub const OP_IF: u8 = 0x04;
pub const OP_ELSE: u8 = 0x05;
pub const OP_END: u8 = 0x0B;
pub const OP_BR: u8 = 0x0C;
pub const OP_BR_IF: u8 = 0x0D;
pub const OP_BR_TABLE: u8 = 0x0E;
pub const OP_RETURN: u8 = 0x0F;
pub const OP_CALL: u8 = 0x10;
pub const OP_CALL_INDIRECT: u8 = 0x11;
pub const OP_DROP: u8 = 0x1A;
pub const OP_SELECT: u8 = 0x1B;

// Variable access
pub const OP_LOCAL_GET: u8 = 0x20;
pub const OP_LOCAL_SET: u8 = 0x21;
pub const OP_LOCAL_TEE: u8 = 0x22;
pub const OP_GLOBAL_GET: u8 = 0x23;
pub const OP_GLOBAL_SET: u8 = 0x24;

// Memory instructions
pub const OP_I32_LOAD: u8 = 0x28;
pub const OP_I64_LOAD: u8 = 0x29;
pub const OP_F32_LOAD: u8 = 0x2A;
pub const OP_F64_LOAD: u8 = 0x2B;
pub const OP_I32_LOAD8_S: u8 = 0x2C;
pub const OP_I32_LOAD8_U: u8 = 0x2D;
pub const OP_I32_LOAD16_S: u8 = 0x2E;
pub const OP_I32_LOAD16_U: u8 = 0x2F;
pub const OP_I64_LOAD8_S: u8 = 0x30;
pub const OP_I64_LOAD8_U: u8 = 0x31;
pub const OP_I64_LOAD16_S: u8 = 0x32;
pub const OP_I64_LOAD16_U: u8 = 0x33;
pub const OP_I64_LOAD32_S: u8 = 0x34;
pub const OP_I64_LOAD32_U: u8 = 0x35;
pub const OP_I32_STORE: u8 = 0x36;
pub const OP_I64_STORE: u8 = 0x37;
pub const OP_F32_STORE: u8 = 0x38;
pub const OP_F64_STORE: u8 = 0x39;
pub const OP_I32_STORE8: u8 = 0x3A;
pub const OP_I32_STORE16: u8 = 0x3B;
pub const OP_I64_STORE8: u8 = 0x3C;
pub const OP_I64_STORE16: u8 = 0x3D;
pub const OP_I64_STORE32: u8 = 0x3E;
pub const OP_MEMORY_SIZE: u8 = 0x3F;
pub const OP_MEMORY_GROW: u8 = 0x40;

// Constants
pub const OP_I32_CONST: u8 = 0x41;
pub const OP_I64_CONST: u8 = 0x42;
pub const OP_F32_CONST: u8 = 0x43;
pub const OP_F64_CONST: u8 = 0x44;

// i32 comparison
pub const OP_I32_EQZ: u8 = 0x45;
pub const OP_I32_EQ: u8 = 0x46;
pub const OP_I32_NE: u8 = 0x47;
pub const OP_I32_LT_S: u8 = 0x48;
pub const OP_I32_LT_U: u8 = 0x49;
pub const OP_I32_GT_S: u8 = 0x4A;
pub const OP_I32_GT_U: u8 = 0x4B;
pub const OP_I32_LE_S: u8 = 0x4C;
pub const OP_I32_LE_U: u8 = 0x4D;
pub const OP_I32_GE_S: u8 = 0x4E;
pub const OP_I32_GE_U: u8 = 0x4F;

// i32 arithmetic / bitwise
pub const OP_I32_CLZ: u8 = 0x67;
pub const OP_I32_CTZ: u8 = 0x68;
pub const OP_I32_POPCNT: u8 = 0x69;
pub const OP_I32_ADD: u8 = 0x6A;
pub const OP_I32_SUB: u8 = 0x6B;
pub const OP_I32_MUL: u8 = 0x6C;
pub const OP_I32_DIV_S: u8 = 0x6D;
pub const OP_I32_DIV_U: u8 = 0x6E;
pub const OP_I32_REM_S: u8 = 0x6F;
pub const OP_I32_REM_U: u8 = 0x70;
pub const OP_I32_AND: u8 = 0x71;
pub const OP_I32_OR: u8 = 0x72;
pub const OP_I32_XOR: u8 = 0x73;
pub const OP_I32_SHL: u8 = 0x74;
pub const OP_I32_SHR_S: u8 = 0x75;
pub const OP_I32_SHR_U: u8 = 0x76;
pub const OP_I32_ROTL: u8 = 0x77;
pub const OP_I32_ROTR: u8 = 0x78;

// Sign-extension (post-MVP but very common)
pub const OP_I32_EXTEND8_S: u8 = 0xC0;
pub const OP_I32_EXTEND16_S: u8 = 0xC1;

/// Zero-sized type providing opcode → human-readable name lookup.
///
/// Use via the [`WASM_OP_NAMES`] static, which offers a `.get(&opcode)` API
/// compatible with `HashMap`-style access.
pub struct WasmOpNames;

impl WasmOpNames {
    /// Look up the human-readable name for an opcode.
    pub fn get(&self, opcode: &u8) -> Option<&'static str> {
        match *opcode {
            OP_UNREACHABLE => Some("unreachable"),
            OP_NOP => Some("nop"),
            OP_BLOCK => Some("block"),
            OP_LOOP => Some("loop"),
            OP_IF => Some("if"),
            OP_ELSE => Some("else"),
            OP_END => Some("end"),
            OP_BR => Some("br"),
            OP_BR_IF => Some("br_if"),
            OP_BR_TABLE => Some("br_table"),
            OP_RETURN => Some("return"),
            OP_CALL => Some("call"),
            OP_CALL_INDIRECT => Some("call_indirect"),
            OP_DROP => Some("drop"),
            OP_SELECT => Some("select"),
            OP_LOCAL_GET => Some("local.get"),
            OP_LOCAL_SET => Some("local.set"),
            OP_LOCAL_TEE => Some("local.tee"),
            OP_GLOBAL_GET => Some("global.get"),
            OP_GLOBAL_SET => Some("global.set"),
            OP_I32_LOAD => Some("i32.load"),
            OP_I32_LOAD8_S => Some("i32.load8_s"),
            OP_I32_LOAD8_U => Some("i32.load8_u"),
            OP_I32_LOAD16_S => Some("i32.load16_s"),
            OP_I32_LOAD16_U => Some("i32.load16_u"),
            OP_I32_STORE => Some("i32.store"),
            OP_I32_STORE8 => Some("i32.store8"),
            OP_I32_STORE16 => Some("i32.store16"),
            OP_MEMORY_SIZE => Some("memory.size"),
            OP_MEMORY_GROW => Some("memory.grow"),
            OP_I32_CONST => Some("i32.const"),
            OP_I32_EQZ => Some("i32.eqz"),
            OP_I32_EQ => Some("i32.eq"),
            OP_I32_NE => Some("i32.ne"),
            OP_I32_LT_S => Some("i32.lt_s"),
            OP_I32_LT_U => Some("i32.lt_u"),
            OP_I32_GT_S => Some("i32.gt_s"),
            OP_I32_GT_U => Some("i32.gt_u"),
            OP_I32_LE_S => Some("i32.le_s"),
            OP_I32_LE_U => Some("i32.le_u"),
            OP_I32_GE_S => Some("i32.ge_s"),
            OP_I32_GE_U => Some("i32.ge_u"),
            OP_I32_CLZ => Some("i32.clz"),
            OP_I32_CTZ => Some("i32.ctz"),
            OP_I32_POPCNT => Some("i32.popcnt"),
            OP_I32_ADD => Some("i32.add"),
            OP_I32_SUB => Some("i32.sub"),
            OP_I32_MUL => Some("i32.mul"),
            OP_I32_DIV_S => Some("i32.div_s"),
            OP_I32_DIV_U => Some("i32.div_u"),
            OP_I32_REM_S => Some("i32.rem_s"),
            OP_I32_REM_U => Some("i32.rem_u"),
            OP_I32_AND => Some("i32.and"),
            OP_I32_OR => Some("i32.or"),
            OP_I32_XOR => Some("i32.xor"),
            OP_I32_SHL => Some("i32.shl"),
            OP_I32_SHR_S => Some("i32.shr_s"),
            OP_I32_SHR_U => Some("i32.shr_u"),
            OP_I32_ROTL => Some("i32.rotl"),
            OP_I32_ROTR => Some("i32.rotr"),
            OP_I32_EXTEND8_S => Some("i32.extend8_s"),
            OP_I32_EXTEND16_S => Some("i32.extend16_s"),
            _ => None,
        }
    }
}

/// Opcode → human-readable name mapping (zero-sized, match-based lookup).
pub static WASM_OP_NAMES: WasmOpNames = WasmOpNames;

// ===================================================================== //
//  Module representation                                                 //
// ===================================================================== //

/// WASM function type: params → results.
#[derive(Clone, Debug, Default)]
pub struct FuncType {
    /// Parameter value types (list of `VALTYPE_*`).
    pub params: Vec<u8>,
    /// Result value types (list of `VALTYPE_*`).
    pub results: Vec<u8>,
}

/// Import kind discriminator.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImportKind {
    /// Function import (type index).
    Func,
    /// Table import.
    Table,
    /// Memory import.
    Memory,
    /// Global import.
    Global,
}

/// WASM import entry.
#[derive(Clone, Debug)]
pub struct Import {
    /// Module name.
    pub module: String,
    /// Field name.
    pub name: String,
    /// Import kind.
    pub kind: ImportKind,
    /// Type index for function imports.
    pub index: u32,
}

/// Export kind discriminator.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExportKind {
    /// Function export.
    Func,
    /// Table export.
    Table,
    /// Memory export.
    Memory,
    /// Global export.
    Global,
}

/// WASM export entry.
#[derive(Clone, Debug)]
pub struct Export {
    /// Export name.
    pub name: String,
    /// Export kind.
    pub kind: ExportKind,
    /// Index within the kind's index space.
    pub index: u32,
}

/// Decoded function body.
#[derive(Clone, Debug)]
pub struct FuncBody {
    /// Local declarations: `(count, valtype)` pairs.
    pub locals: Vec<(u32, u8)>,
    /// Total local count (sum of counts).
    pub num_locals: u32,
    /// Decoded instructions.
    pub instructions: Vec<WasmInstr>,
}

/// WASM data segment (active, memory 0).
#[derive(Clone, Debug)]
pub struct DataSegment {
    /// Constant offset expression value.
    pub offset: i32,
    /// Raw data bytes.
    pub data: Vec<u8>,
}

/// Global variable descriptor.
#[derive(Clone, Debug)]
pub struct Global {
    /// Value type (`VALTYPE_*`).
    pub valtype: u8,
    /// Mutability (0 = const, 1 = mutable).
    pub mutable: u8,
    /// Initial value from init expression.
    pub init: i32,
}

/// Parsed wasm module.
#[derive(Clone, Debug, Default)]
pub struct WasmModule {
    /// Function type signatures.
    pub types: Vec<FuncType>,
    /// Import entries.
    pub imports: Vec<Import>,
    /// Type indices for declared functions.
    pub func_type_indices: Vec<u32>,
    /// Export entries.
    pub exports: Vec<Export>,
    /// Function bodies (code section).
    pub functions: Vec<FuncBody>,
    /// Data segments.
    pub data_segments: Vec<DataSegment>,
    /// Global variables.
    pub globals: Vec<Global>,
}

impl WasmModule {
    /// Number of imported functions.
    pub fn num_imported_funcs(&self) -> usize {
        self.imports
            .iter()
            .filter(|imp| imp.kind == ImportKind::Func)
            .count()
    }
}

// ===================================================================== //
//  Error type                                                            //
// ===================================================================== //

/// Errors that can occur during WASM binary decoding.
#[derive(Clone, Debug)]
pub enum DecodeError {
    /// File too short or unexpected end of data.
    UnexpectedEof(&'static str),
    /// Bad magic number.
    BadMagic([u8; 4]),
    /// Unsupported WASM version.
    BadVersion(u32),
    /// Expected functype marker 0x60.
    ExpectedFuncType(u8),
    /// Expected i32.const in data segment offset.
    ExpectedConstInDataOffset(u8),
    /// Unsupported data segment flags.
    UnsupportedDataFlags(u32),
    /// Unsupported opcode encountered.
    UnsupportedOpcode { opcode: u8, pos: usize },
    /// Generic validation error.
    Other(String),
}

impl fmt::Display for DecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnexpectedEof(ctx) => write!(f, "unexpected EOF while reading {ctx}"),
            Self::BadMagic(magic) => write!(f, "bad magic: {magic:02x?}"),
            Self::BadVersion(v) => write!(f, "unsupported wasm version: {v}"),
            Self::ExpectedFuncType(b) => write!(f, "expected functype 0x60, got 0x{b:02x}"),
            Self::ExpectedConstInDataOffset(op) => {
                write!(f, "expected i32.const in data offset, got 0x{op:02x}")
            }
            Self::UnsupportedDataFlags(flags) => {
                write!(f, "unsupported data segment flags: {flags}")
            }
            Self::UnsupportedOpcode { opcode, pos } => {
                write!(
                    f,
                    "unsupported wasm opcode: 0x{opcode:02x} at position {pos}"
                )
            }
            Self::Other(msg) => write!(f, "{msg}"),
        }
    }
}

impl std::error::Error for DecodeError {}

// ===================================================================== //
//  Opcodes with no immediates                                            //
// ===================================================================== //

/// Opcodes that have no immediate operands.
const NO_IMM_OPS: &[u8] = &[
    OP_UNREACHABLE,
    OP_NOP,
    OP_END,
    OP_ELSE,
    OP_RETURN,
    OP_DROP,
    OP_SELECT,
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
    OP_I32_CLZ,
    OP_I32_CTZ,
    OP_I32_POPCNT,
    OP_I32_ADD,
    OP_I32_SUB,
    OP_I32_MUL,
    OP_I32_DIV_S,
    OP_I32_DIV_U,
    OP_I32_REM_S,
    OP_I32_REM_U,
    OP_I32_AND,
    OP_I32_OR,
    OP_I32_XOR,
    OP_I32_SHL,
    OP_I32_SHR_S,
    OP_I32_SHR_U,
    OP_I32_ROTL,
    OP_I32_ROTR,
    OP_I32_EXTEND8_S,
    OP_I32_EXTEND16_S,
];

/// Opcodes with block-type immediate (1 byte).
const BLOCK_TYPE_OPS: &[u8] = &[OP_BLOCK, OP_LOOP, OP_IF];

/// Opcodes with a single local/global index (unsigned LEB128).
const VAR_INDEX_OPS: &[u8] = &[
    OP_LOCAL_GET,
    OP_LOCAL_SET,
    OP_LOCAL_TEE,
    OP_GLOBAL_GET,
    OP_GLOBAL_SET,
];

/// Opcodes with alignment + offset (two unsigned LEB128).
const MEM_INSTR_OPS: &[u8] = &[
    OP_I32_LOAD,
    OP_I64_LOAD,
    OP_F32_LOAD,
    OP_F64_LOAD,
    OP_I32_LOAD8_S,
    OP_I32_LOAD8_U,
    OP_I32_LOAD16_S,
    OP_I32_LOAD16_U,
    OP_I64_LOAD8_S,
    OP_I64_LOAD8_U,
    OP_I64_LOAD16_S,
    OP_I64_LOAD16_U,
    OP_I64_LOAD32_S,
    OP_I64_LOAD32_U,
    OP_I32_STORE,
    OP_I64_STORE,
    OP_F32_STORE,
    OP_F64_STORE,
    OP_I32_STORE8,
    OP_I32_STORE16,
    OP_I64_STORE8,
    OP_I64_STORE16,
    OP_I64_STORE32,
];

// ===================================================================== //
//  Decoder                                                               //
// ===================================================================== //

/// Decode a WASM binary into a [`WasmModule`].
pub fn decode(data: &[u8]) -> Result<WasmModule, DecodeError> {
    if data.len() < 8 {
        return Err(DecodeError::Other(
            "file too short to be a wasm module".into(),
        ));
    }

    let magic: [u8; 4] = [data[0], data[1], data[2], data[3]];
    if magic != [0x00, 0x61, 0x73, 0x6D] {
        return Err(DecodeError::BadMagic(magic));
    }

    let version = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
    if version != 1 {
        return Err(DecodeError::BadVersion(version));
    }

    let mut module = WasmModule::default();
    let mut pos: usize = 8;

    while pos < data.len() {
        let section_id = data[pos];
        pos += 1;
        let (section_size, new_pos) = read_unsigned_leb128(data, pos)?;
        pos = new_pos;
        let section_end = pos + section_size as usize;

        match section_id {
            SEC_TYPE => decode_type_section(data, pos, section_end, &mut module)?,
            SEC_IMPORT => decode_import_section(data, pos, section_end, &mut module)?,
            SEC_FUNCTION => decode_function_section(data, pos, section_end, &mut module)?,
            SEC_EXPORT => decode_export_section(data, pos, section_end, &mut module)?,
            SEC_CODE => decode_code_section(data, pos, section_end, &mut module)?,
            SEC_DATA => decode_data_section(data, pos, section_end, &mut module)?,
            SEC_GLOBAL => decode_global_section(data, pos, section_end, &mut module)?,
            // Skip other sections (custom, table, memory, start, element, datacount)
            _ => {}
        }

        pos = section_end;
    }

    Ok(module)
}

// ===================================================================== //
//  Section decoders                                                      //
// ===================================================================== //

fn decode_type_section(
    data: &[u8],
    mut pos: usize,
    end: usize,
    module: &mut WasmModule,
) -> Result<(), DecodeError> {
    let (count, new_pos) = read_unsigned_leb128(data, pos)?;
    pos = new_pos;
    for _ in 0..count {
        let form = data[pos];
        pos += 1;
        if form != 0x60 {
            return Err(DecodeError::ExpectedFuncType(form));
        }
        // Params
        let (num_params, new_pos) = read_unsigned_leb128(data, pos)?;
        pos = new_pos;
        let mut params = Vec::with_capacity(num_params as usize);
        for _ in 0..num_params {
            params.push(data[pos]);
            pos += 1;
        }
        // Results
        let (num_results, new_pos) = read_unsigned_leb128(data, pos)?;
        pos = new_pos;
        let mut results = Vec::with_capacity(num_results as usize);
        for _ in 0..num_results {
            results.push(data[pos]);
            pos += 1;
        }
        module.types.push(FuncType { params, results });
    }
    let _ = end;
    Ok(())
}

fn decode_import_section(
    data: &[u8],
    mut pos: usize,
    _end: usize,
    module: &mut WasmModule,
) -> Result<(), DecodeError> {
    let (count, new_pos) = read_unsigned_leb128(data, pos)?;
    pos = new_pos;
    for _ in 0..count {
        let (mod_len, new_pos) = read_unsigned_leb128(data, pos)?;
        pos = new_pos;
        let module_name = String::from_utf8_lossy(&data[pos..pos + mod_len as usize]).into_owned();
        pos += mod_len as usize;
        let (name_len, new_pos) = read_unsigned_leb128(data, pos)?;
        pos = new_pos;
        let field_name = String::from_utf8_lossy(&data[pos..pos + name_len as usize]).into_owned();
        pos += name_len as usize;
        let kind = data[pos];
        pos += 1;
        match kind {
            0 => {
                // Function
                let (type_idx, new_pos) = read_unsigned_leb128(data, pos)?;
                pos = new_pos;
                module.imports.push(Import {
                    module: module_name,
                    name: field_name,
                    kind: ImportKind::Func,
                    index: type_idx as u32,
                });
            }
            1 => {
                // Table: elem_type + limits
                pos += 1; // elem_type
                let flags = data[pos];
                pos += 1;
                let (_, new_pos) = read_unsigned_leb128(data, pos)?;
                pos = new_pos; // min
                if flags & 1 != 0 {
                    let (_, new_pos) = read_unsigned_leb128(data, pos)?;
                    pos = new_pos; // max
                }
                module.imports.push(Import {
                    module: module_name,
                    name: field_name,
                    kind: ImportKind::Table,
                    index: 0,
                });
            }
            2 => {
                // Memory: limits
                let flags = data[pos];
                pos += 1;
                let (_, new_pos) = read_unsigned_leb128(data, pos)?;
                pos = new_pos; // min
                if flags & 1 != 0 {
                    let (_, new_pos) = read_unsigned_leb128(data, pos)?;
                    pos = new_pos; // max
                }
                module.imports.push(Import {
                    module: module_name,
                    name: field_name,
                    kind: ImportKind::Memory,
                    index: 0,
                });
            }
            3 => {
                // Global: valtype + mutability
                pos += 1; // valtype
                pos += 1; // mutability
                module.imports.push(Import {
                    module: module_name,
                    name: field_name,
                    kind: ImportKind::Global,
                    index: 0,
                });
            }
            _ => {
                return Err(DecodeError::Other(format!("unknown import kind: {kind}")));
            }
        }
    }
    Ok(())
}

fn decode_function_section(
    data: &[u8],
    mut pos: usize,
    _end: usize,
    module: &mut WasmModule,
) -> Result<(), DecodeError> {
    let (count, new_pos) = read_unsigned_leb128(data, pos)?;
    pos = new_pos;
    for _ in 0..count {
        let (type_idx, new_pos) = read_unsigned_leb128(data, pos)?;
        pos = new_pos;
        module.func_type_indices.push(type_idx as u32);
    }
    Ok(())
}

fn decode_export_section(
    data: &[u8],
    mut pos: usize,
    _end: usize,
    module: &mut WasmModule,
) -> Result<(), DecodeError> {
    let (count, new_pos) = read_unsigned_leb128(data, pos)?;
    pos = new_pos;
    for _ in 0..count {
        let (name_len, new_pos) = read_unsigned_leb128(data, pos)?;
        pos = new_pos;
        let name = String::from_utf8_lossy(&data[pos..pos + name_len as usize]).into_owned();
        pos += name_len as usize;
        let kind = data[pos];
        pos += 1;
        let (index, new_pos) = read_unsigned_leb128(data, pos)?;
        pos = new_pos;
        let export_kind = match kind {
            0 => ExportKind::Func,
            1 => ExportKind::Table,
            2 => ExportKind::Memory,
            3 => ExportKind::Global,
            _ => return Err(DecodeError::Other(format!("unknown export kind: {kind}"))),
        };
        module.exports.push(Export {
            name,
            kind: export_kind,
            index: index as u32,
        });
    }
    Ok(())
}

fn decode_global_section(
    data: &[u8],
    mut pos: usize,
    _end: usize,
    module: &mut WasmModule,
) -> Result<(), DecodeError> {
    let (count, new_pos) = read_unsigned_leb128(data, pos)?;
    pos = new_pos;
    for _ in 0..count {
        let valtype = data[pos];
        pos += 1;
        let mutable = data[pos];
        pos += 1;
        // Decode init expression (simplified: expect i32.const + end)
        let mut init_val: i32 = 0;
        let op = data[pos];
        pos += 1;
        if op == OP_I32_CONST {
            let (val, new_pos) = read_signed_leb128(data, pos, 32)?;
            pos = new_pos;
            init_val = val as i32;
        }
        // Skip until END
        while data[pos] != OP_END {
            pos += 1;
        }
        pos += 1; // skip END
        module.globals.push(Global {
            valtype,
            mutable,
            init: init_val,
        });
    }
    Ok(())
}

fn decode_code_section(
    data: &[u8],
    mut pos: usize,
    _end: usize,
    module: &mut WasmModule,
) -> Result<(), DecodeError> {
    let (count, new_pos) = read_unsigned_leb128(data, pos)?;
    pos = new_pos;
    for _ in 0..count {
        let (body_size, new_pos) = read_unsigned_leb128(data, pos)?;
        pos = new_pos;
        let body_end = pos + body_size as usize;

        // Locals
        let (num_local_decls, new_pos) = read_unsigned_leb128(data, pos)?;
        pos = new_pos;
        let mut locals_list = Vec::with_capacity(num_local_decls as usize);
        let mut total_locals: u32 = 0;
        for _ in 0..num_local_decls {
            let (lcount, new_pos) = read_unsigned_leb128(data, pos)?;
            pos = new_pos;
            let ltype = data[pos];
            pos += 1;
            locals_list.push((lcount as u32, ltype));
            total_locals += lcount as u32;
        }

        // Instructions
        let mut instructions = Vec::new();
        while pos < body_end {
            let (instr, new_pos) = decode_instruction(data, pos)?;
            pos = new_pos;
            instructions.push(instr);
        }

        module.functions.push(FuncBody {
            locals: locals_list,
            num_locals: total_locals,
            instructions,
        });
    }
    Ok(())
}

fn decode_data_section(
    data: &[u8],
    mut pos: usize,
    _end: usize,
    module: &mut WasmModule,
) -> Result<(), DecodeError> {
    let (count, new_pos) = read_unsigned_leb128(data, pos)?;
    pos = new_pos;
    for _ in 0..count {
        let (seg_flags, new_pos) = read_unsigned_leb128(data, pos)?;
        pos = new_pos;
        if seg_flags == 0 {
            // Active segment, memory 0, with offset expression
            let op = data[pos];
            pos += 1;
            if op != OP_I32_CONST {
                return Err(DecodeError::ExpectedConstInDataOffset(op));
            }
            let (offset_val, new_pos) = read_signed_leb128(data, pos, 32)?;
            pos = new_pos;
            let end_op = data[pos];
            pos += 1;
            if end_op != OP_END {
                return Err(DecodeError::Other(
                    "expected END after data segment offset".into(),
                ));
            }
            let (byte_count, new_pos) = read_unsigned_leb128(data, pos)?;
            pos = new_pos;
            let seg_data = data[pos..pos + byte_count as usize].to_vec();
            pos += byte_count as usize;
            module.data_segments.push(DataSegment {
                offset: offset_val as i32,
                data: seg_data,
            });
        } else {
            return Err(DecodeError::UnsupportedDataFlags(seg_flags as u32));
        }
    }
    Ok(())
}

// ===================================================================== //
//  Instruction decoder                                                   //
// ===================================================================== ///

/// Decode a single WASM instruction. Returns `(WasmInstr, new_pos)`.
pub fn decode_instruction(data: &[u8], pos: usize) -> Result<(WasmInstr, usize), DecodeError> {
    let opcode = data[pos];
    let mut p = pos + 1;

    // No immediates
    if NO_IMM_OPS.contains(&opcode) {
        return Ok((WasmInstr::new(opcode), p));
    }

    // Block type (1 byte)
    if BLOCK_TYPE_OPS.contains(&opcode) {
        let block_type = data[p] as i64;
        p += 1;
        return Ok((WasmInstr::with_imms(opcode, [block_type]), p));
    }

    // Branch (single unsigned LEB128 label index)
    if opcode == OP_BR || opcode == OP_BR_IF {
        let (label_idx, new_pos) = read_unsigned_leb128(data, p)?;
        p = new_pos;
        return Ok((WasmInstr::with_imms(opcode, [label_idx as i64]), p));
    }

    // br_table: count targets + default
    if opcode == OP_BR_TABLE {
        let (num_targets, new_pos) = read_unsigned_leb128(data, p)?;
        p = new_pos;
        let mut targets = Vec::with_capacity(num_targets as usize);
        for _ in 0..num_targets {
            let (t, new_pos) = read_unsigned_leb128(data, p)?;
            p = new_pos;
            targets.push(t as i64);
        }
        let (default, new_pos) = read_unsigned_leb128(data, p)?;
        p = new_pos;
        let mut immediates = targets;
        immediates.push(default as i64);
        return Ok((WasmInstr::with_imms(opcode, immediates), p));
    }

    // Call (unsigned LEB128 function index)
    if opcode == OP_CALL {
        let (func_idx, new_pos) = read_unsigned_leb128(data, p)?;
        p = new_pos;
        return Ok((WasmInstr::with_imms(opcode, [func_idx as i64]), p));
    }

    // call_indirect (type_idx + reserved table byte)
    if opcode == OP_CALL_INDIRECT {
        let (type_idx, new_pos) = read_unsigned_leb128(data, p)?;
        p = new_pos;
        let table_idx = data[p] as i64;
        p += 1;
        return Ok((
            WasmInstr::with_imms(opcode, [type_idx as i64, table_idx]),
            p,
        ));
    }

    // Local/global variable access (unsigned LEB128 index)
    if VAR_INDEX_OPS.contains(&opcode) {
        let (idx, new_pos) = read_unsigned_leb128(data, p)?;
        p = new_pos;
        return Ok((WasmInstr::with_imms(opcode, [idx as i64]), p));
    }

    // Memory instructions: alignment + offset (two unsigned LEB128)
    if MEM_INSTR_OPS.contains(&opcode) {
        let (align, new_pos) = read_unsigned_leb128(data, p)?;
        p = new_pos;
        let (offset, new_pos) = read_unsigned_leb128(data, p)?;
        p = new_pos;
        return Ok((
            WasmInstr::with_imms(opcode, [align as i64, offset as i64]),
            p,
        ));
    }

    // memory.size / memory.grow: 1-byte reserved index
    if opcode == OP_MEMORY_SIZE || opcode == OP_MEMORY_GROW {
        let reserved = data[p] as i64;
        p += 1;
        return Ok((WasmInstr::with_imms(opcode, [reserved]), p));
    }

    // i32.const (signed LEB128)
    if opcode == OP_I32_CONST {
        let (val, new_pos) = read_signed_leb128(data, p, 32)?;
        p = new_pos;
        return Ok((WasmInstr::with_imms(opcode, [val]), p));
    }

    // i64.const (signed LEB128, 64-bit)
    if opcode == OP_I64_CONST {
        let (val, new_pos) = read_signed_leb128(data, p, 64)?;
        p = new_pos;
        return Ok((WasmInstr::with_imms(opcode, [val]), p));
    }

    // f32.const: 4-byte IEEE 754
    if opcode == OP_F32_CONST {
        if p + 4 > data.len() {
            return Err(DecodeError::UnexpectedEof("f32.const"));
        }
        let bytes: [u8; 4] = [data[p], data[p + 1], data[p + 2], data[p + 3]];
        let bits = u32::from_le_bytes(bytes);
        p += 4;
        return Ok((WasmInstr::with_imms(opcode, [bits as i64]), p));
    }

    // f64.const: 8-byte IEEE 754
    if opcode == OP_F64_CONST {
        if p + 8 > data.len() {
            return Err(DecodeError::UnexpectedEof("f64.const"));
        }
        let bytes: [u8; 8] = [
            data[p],
            data[p + 1],
            data[p + 2],
            data[p + 3],
            data[p + 4],
            data[p + 5],
            data[p + 6],
            data[p + 7],
        ];
        let bits = u64::from_le_bytes(bytes);
        p += 8;
        return Ok((WasmInstr::with_imms(opcode, [bits as i64]), p));
    }

    Err(DecodeError::UnsupportedOpcode { opcode, pos })
}

// ===================================================================== //
//  Tests                                                                 //
// ===================================================================== //

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal valid WASM module with one empty function.
    fn minimal_wasm_module() -> Vec<u8> {
        let mut bytes = Vec::new();
        // Magic + version
        bytes.extend_from_slice(&[0x00, 0x61, 0x73, 0x6D]); // \0asm
        bytes.extend_from_slice(&[0x01, 0x00, 0x00, 0x00]); // version 1

        // Type section: one function type () -> ()
        bytes.push(SEC_TYPE);
        bytes.push(0x04); // section size
        bytes.push(0x01); // count
        bytes.push(0x60); // functype
        bytes.push(0x00); // 0 params
        bytes.push(0x00); // 0 results

        // Function section: one function, type index 0
        bytes.push(SEC_FUNCTION);
        bytes.push(0x02); // section size
        bytes.push(0x01); // count
        bytes.push(0x00); // type index 0

        // Export section: export "main" as function 0
        bytes.push(SEC_EXPORT);
        bytes.push(0x08); // section size
        bytes.push(0x01); // count
        bytes.push(0x04); // name length
        bytes.extend_from_slice(b"main");
        bytes.push(0x00); // func export
        bytes.push(0x00); // index 0

        // Code section: one function body (empty)
        bytes.push(SEC_CODE);
        bytes.push(0x04); // section size
        bytes.push(0x01); // count
        bytes.push(0x02); // body size
        bytes.push(0x00); // 0 local declarations
        bytes.push(OP_END);

        bytes
    }

    #[test]
    fn test_read_unsigned_leb128() {
        // Single byte
        let data = [0x05];
        let (val, pos) = read_unsigned_leb128(&data, 0).unwrap();
        assert_eq!(val, 5);
        assert_eq!(pos, 1);

        // Multi-byte: 128 → [0x80, 0x01]
        let data = [0x80, 0x01];
        let (val, pos) = read_unsigned_leb128(&data, 0).unwrap();
        assert_eq!(val, 128);
        assert_eq!(pos, 2);

        // 624485 → [0xE5, 0x8E, 0x26]
        let data = [0xE5, 0x8E, 0x26];
        let (val, pos) = read_unsigned_leb128(&data, 0).unwrap();
        assert_eq!(val, 624485);
        assert_eq!(pos, 3);
    }

    #[test]
    fn test_read_signed_leb128() {
        // Positive: 42 → [0x2A]
        let data = [0x2A];
        let (val, pos) = read_signed_leb128(&data, 0, 32).unwrap();
        assert_eq!(val, 42);
        assert_eq!(pos, 1);

        // Negative: -1 → [0x7F]
        let data = [0x7F];
        let (val, pos) = read_signed_leb128(&data, 0, 32).unwrap();
        assert_eq!(val, -1);
        assert_eq!(pos, 1);

        // Negative: -123456 → [0xC0, 0xBB, 0x78]
        let data = [0xC0, 0xBB, 0x78];
        let (val, pos) = read_signed_leb128(&data, 0, 32).unwrap();
        assert_eq!(val, -123456);
        assert_eq!(pos, 3);
    }

    #[test]
    fn test_decode_minimal_module() {
        let bytes = minimal_wasm_module();
        let module = decode(&bytes).unwrap();

        assert_eq!(module.types.len(), 1);
        assert_eq!(module.types[0].params.len(), 0);
        assert_eq!(module.types[0].results.len(), 0);

        assert_eq!(module.func_type_indices.len(), 1);
        assert_eq!(module.func_type_indices[0], 0);

        assert_eq!(module.exports.len(), 1);
        assert_eq!(module.exports[0].name, "main");
        assert_eq!(module.exports[0].kind, ExportKind::Func);
        assert_eq!(module.exports[0].index, 0);

        assert_eq!(module.functions.len(), 1);
        assert_eq!(module.functions[0].num_locals, 0);
        assert_eq!(module.functions[0].instructions.len(), 1);
        assert_eq!(module.functions[0].instructions[0].opcode, OP_END);
    }

    #[test]
    fn test_decode_bad_magic() {
        let bytes = [0xDE, 0xAD, 0xBE, 0xEF, 0x01, 0x00, 0x00, 0x00];
        let err = decode(&bytes).unwrap_err();
        assert!(matches!(err, DecodeError::BadMagic(_)));
    }

    #[test]
    fn test_decode_bad_version() {
        let bytes = [0x00, 0x61, 0x73, 0x6D, 0x02, 0x00, 0x00, 0x00];
        let err = decode(&bytes).unwrap_err();
        assert!(matches!(err, DecodeError::BadVersion(2)));
    }

    #[test]
    fn test_decode_instructions() {
        // i32.const 42
        let data = [OP_I32_CONST, 0x2A];
        let (instr, pos) = decode_instruction(&data, 0).unwrap();
        assert_eq!(instr.opcode, OP_I32_CONST);
        assert_eq!(instr.immediates, [42]);
        assert_eq!(pos, 2);

        // local.get 3
        let data = [OP_LOCAL_GET, 0x03];
        let (instr, pos) = decode_instruction(&data, 0).unwrap();
        assert_eq!(instr.opcode, OP_LOCAL_GET);
        assert_eq!(instr.immediates, [3]);
        assert_eq!(pos, 2);

        // i32.add (no immediates)
        let data = [OP_I32_ADD];
        let (instr, pos) = decode_instruction(&data, 0).unwrap();
        assert_eq!(instr.opcode, OP_I32_ADD);
        assert!(instr.immediates.is_empty());
        assert_eq!(pos, 1);

        // block void
        let data = [OP_BLOCK, 0x40];
        let (instr, pos) = decode_instruction(&data, 0).unwrap();
        assert_eq!(instr.opcode, OP_BLOCK);
        assert_eq!(instr.immediates, [0x40]);
        assert_eq!(pos, 2);
    }

    #[test]
    fn test_wasm_module_num_imported_funcs() {
        let mut module = WasmModule::default();
        module.imports.push(Import {
            module: "env".into(),
            name: "print".into(),
            kind: ImportKind::Func,
            index: 0,
        });
        module.imports.push(Import {
            module: "env".into(),
            name: "memory".into(),
            kind: ImportKind::Memory,
            index: 0,
        });
        assert_eq!(module.num_imported_funcs(), 1);
    }

    #[test]
    fn test_wasm_instr_display() {
        let instr = WasmInstr::with_imms(OP_I32_CONST, [42]);
        assert_eq!(format!("{instr}"), "i32.const(42)");

        let instr = WasmInstr::new(OP_I32_ADD);
        assert_eq!(format!("{instr}"), "i32.add");
    }

    #[test]
    fn test_decode_module_with_const_add() {
        let mut bytes = Vec::new();
        // Magic + version
        bytes.extend_from_slice(&[0x00, 0x61, 0x73, 0x6D, 0x01, 0x00, 0x00, 0x00]);

        // Type section: () -> (i32)
        bytes.push(SEC_TYPE);
        bytes.push(0x05);
        bytes.push(0x01); // count
        bytes.push(0x60); // functype
        bytes.push(0x00); // 0 params
        bytes.push(0x01); // 1 result
        bytes.push(VALTYPE_I32);

        // Function section
        bytes.push(SEC_FUNCTION);
        bytes.push(0x02);
        bytes.push(0x01);
        bytes.push(0x00);

        // Code section: 0 locals, i32.const 3, i32.const 5, i32.add, end
        let code = [
            0x00,
            OP_I32_CONST,
            0x03,
            OP_I32_CONST,
            0x05,
            OP_I32_ADD,
            OP_END,
        ];
        bytes.push(SEC_CODE);
        bytes.push(1 + 1 + code.len() as u8); // section size: count(1) + bodysize(1) + body
        bytes.push(0x01); // count
        bytes.push(code.len() as u8); // body size
        bytes.extend_from_slice(&code);

        let module = decode(&bytes).unwrap();
        let func = &module.functions[0];
        assert_eq!(func.instructions.len(), 4);
        assert_eq!(func.instructions[0].opcode, OP_I32_CONST);
        assert_eq!(func.instructions[0].immediates[0], 3);
        assert_eq!(func.instructions[1].opcode, OP_I32_CONST);
        assert_eq!(func.instructions[1].immediates[0], 5);
        assert_eq!(func.instructions[2].opcode, OP_I32_ADD);
        assert_eq!(func.instructions[3].opcode, OP_END);
    }
}
