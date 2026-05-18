// SPDX-License-Identifier: Apache-2.0
// Distilled from Percepta's `transformer-vm` (Apache-2.0 © Percepta).

//! Compile pipeline: C source → WASM → lowered bytecode → token prefix.
//!
//! This module implements the full compilation pipeline that converts C source code
//! into the token prefix format used by the transformer-vm:
//!
//! 1. **C → WASM**: Invoke clang with wasm32 target
//! 2. **WASM → Dispatch table**: Decode, lower hard ops, compile to flat dispatch table
//! 3. **Dispatch table → Prefix**: Format as token prefix string
//!
//! # Usage
//!
//! ```ignore
//! use percepta::compile::{compile_program, CompiledProgram};
//!
//! let result = compile_program(
//!     "void compute(const char *input) { putchar('H'); putchar('i'); }",
//!     "world",
//! )?;
//! println!("Prefix:\n{}", result.prefix);
//! ```

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::percepta::wasm::decoder::{
    self, DecodeError, ExportKind, FuncBody, ImportKind, OP_BLOCK, OP_BR, OP_BR_IF, OP_CALL,
    OP_DROP, OP_ELSE, OP_END, OP_GLOBAL_GET, OP_GLOBAL_SET, OP_I32_ADD, OP_I32_CONST, OP_I32_EQ,
    OP_I32_EQZ, OP_I32_GE_S, OP_I32_GE_U, OP_I32_GT_S, OP_I32_GT_U, OP_I32_LE_S, OP_I32_LE_U,
    OP_I32_LOAD, OP_I32_LOAD8_S, OP_I32_LOAD8_U, OP_I32_LOAD16_S, OP_I32_LOAD16_U, OP_I32_LT_S,
    OP_I32_LT_U, OP_I32_NE, OP_I32_STORE, OP_I32_STORE8, OP_I32_STORE16, OP_I32_SUB, OP_IF,
    OP_LOCAL_GET, OP_LOCAL_SET, OP_LOCAL_TEE, OP_LOOP, OP_NOP, OP_RETURN, OP_SELECT,
    OP_UNREACHABLE, WASM_OP_NAMES, WasmModule,
};
use crate::percepta::wasm::lower::lower_hard_ops;

// ── Constants ──────────────────────────────────────────────────

/// Mask for 32-bit unsigned values.
const MASK32: i64 = 0xFFFFFFFF;

/// Memory base address for WASM globals.
const GLOBAL_BASE: i64 = 8;

/// Placeholder for unresolved branch targets.
const PLACEHOLDER: i32 = 0xDEAD;

/// Embedded runtime.h for WASM compilation.
pub const RUNTIME_H: &str =
    include_str!("../../.raw/transformer-vm/transformer_vm/compilation/runtime.h");

// ── Error Type ─────────────────────────────────────────────────

/// Errors that can occur during compilation.
#[derive(Debug)]
pub enum CompileError {
    /// No clang with wasm32 support found.
    ClangNotFound(String),
    /// clang invocation failed.
    ClangFailed(String),
    /// WASM decode error.
    DecodeError(DecodeError),
    /// Unsupported WASM opcode encountered.
    UnsupportedOpcode(String),
    /// I/O error during compilation.
    IoError(String),
    /// Other error.
    Other(String),
}

impl fmt::Display for CompileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ClangNotFound(msg) => write!(f, "clang not found: {msg}"),
            Self::ClangFailed(msg) => write!(f, "clang failed: {msg}"),
            Self::DecodeError(err) => write!(f, "WASM decode error: {err}"),
            Self::UnsupportedOpcode(msg) => write!(f, "unsupported opcode: {msg}"),
            Self::IoError(msg) => write!(f, "I/O error: {msg}"),
            Self::Other(msg) => write!(f, "{msg}"),
        }
    }
}

impl std::error::Error for CompileError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::DecodeError(err) => Some(err),
            _ => None,
        }
    }
}

impl From<DecodeError> for CompileError {
    fn from(err: DecodeError) -> Self {
        Self::DecodeError(err)
    }
}

// ── Types ──────────────────────────────────────────────────────

/// A dispatch table entry: (opcode_name, immediate_value).
pub type DispatchEntry = (String, i32);

/// Result of compilation.
#[derive(Clone, Debug)]
pub struct CompiledProgram {
    /// The dispatch table entries.
    pub program: Vec<DispatchEntry>,
    /// The formatted token prefix string.
    pub prefix: String,
    /// The input base address (0 if no input).
    pub input_base: i32,
    /// The formatted input section (empty if no input).
    pub input_section: String,
}

// ── Label Frame ────────────────────────────────────────────────

/// Kind of structured control flow block.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LabelKind {
    Block,
    Loop,
    If,
}

/// Tracks label stack for branch resolution during compilation.
#[derive(Debug)]
struct LabelFrame {
    kind: LabelKind,
    /// Position in entries where this block starts (LOOP back-edge target).
    start_pc: usize,
    /// Entry indices that need patching when END is reached.
    patches: Vec<usize>,
    /// For IF: the br_if entry index to patch on ELSE.
    if_entry: Option<usize>,
}

// ── Helpers ────────────────────────────────────────────────────

/// Convert a 32-bit value to 4 little-endian bytes.
pub fn int_to_bytes(v: i64) -> [u8; 4] {
    let v = (v & MASK32) as u32;
    v.to_le_bytes()
}

/// Map WASM opcode to dispatch name. Returns `None` for unmapped opcodes.
fn wasm_opcode_to_name(opcode: u8) -> Option<&'static str> {
    match opcode {
        OP_I32_CONST => Some("i32.const"),
        OP_LOCAL_GET => Some("local.get"),
        OP_LOCAL_SET => Some("local.set"),
        OP_LOCAL_TEE => Some("local.tee"),
        OP_DROP => Some("drop"),
        OP_SELECT => Some("select"),
        OP_I32_ADD => Some("i32.add"),
        OP_I32_SUB => Some("i32.sub"),
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
        OP_I32_EQZ => Some("i32.eqz"),
        OP_I32_LOAD => Some("i32.load"),
        OP_I32_LOAD8_S => Some("i32.load8_s"),
        OP_I32_LOAD8_U => Some("i32.load8_u"),
        OP_I32_LOAD16_S => Some("i32.load16_s"),
        OP_I32_LOAD16_U => Some("i32.load16_u"),
        OP_I32_STORE => Some("i32.store"),
        OP_I32_STORE8 => Some("i32.store8"),
        OP_I32_STORE16 => Some("i32.store16"),
        OP_BR => Some("br"),
        OP_BR_IF => Some("br_if"),
        OP_UNREACHABLE => Some("halt"),
        OP_RETURN => Some("halt"),
        _ => None,
    }
}

// ── C → WASM ──────────────────────────────────────────────────

/// Find a clang binary with wasm32 target support.
///
/// Resolution order:
/// 1. `CLANG_PATH` environment variable
/// 2. `clang` on `$PATH` (via `which`)
/// 3. Platform-specific fallbacks (Homebrew on macOS, common Linux paths)
pub fn find_clang() -> Result<PathBuf, CompileError> {
    let mut candidates: Vec<PathBuf> = Vec::new();

    // 1. CLANG_PATH env var
    if let Ok(path) = std::env::var("CLANG_PATH") {
        candidates.push(PathBuf::from(path));
    }

    // 2. which clang
    if let Ok(output) = Command::new("which").arg("clang").output()
        && output.status.success()
    {
        let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
        if !path.is_empty() {
            candidates.push(PathBuf::from(path));
        }
    }

    // 3. Platform-specific fallbacks
    candidates.extend([
        // macOS (Homebrew)
        PathBuf::from("/opt/homebrew/opt/llvm/bin/clang"),
        PathBuf::from("/usr/local/opt/llvm/bin/clang"),
        // Linux
        PathBuf::from("/usr/lib/llvm-18/bin/clang"),
        PathBuf::from("/usr/lib/llvm-17/bin/clang"),
        PathBuf::from("/usr/lib/llvm-16/bin/clang"),
        PathBuf::from("/usr/bin/clang"),
    ]);

    for cc in &candidates {
        match Command::new(cc).arg("--print-targets").output() {
            Ok(output) => {
                if output.status.success() {
                    let stdout = String::from_utf8_lossy(&output.stdout);
                    if stdout.contains("wasm32") {
                        return Ok(cc.clone());
                    }
                }
            }
            Err(_) => continue,
        }
    }

    Err(CompileError::ClangNotFound(
        "No clang with wasm32 target found. Install LLVM with wasm32 support or set CLANG_PATH."
            .to_string(),
    ))
}

/// Write the embedded runtime.h to a temp directory and return its path.
pub fn write_runtime_h(dir: &Path) -> Result<PathBuf, CompileError> {
    let path = dir.join("runtime.h");
    fs::write(&path, RUNTIME_H).map_err(|e| CompileError::IoError(format!("{e}")))?;
    Ok(path)
}

/// Compile C source code to WASM bytes.
///
/// The `runtime_h_path` should point to the runtime.h header file
/// (use [`write_runtime_h`] to create one from the embedded content).
pub fn compile_c_to_wasm(c_source: &str, runtime_h_path: &Path) -> Result<Vec<u8>, CompileError> {
    let temp_dir = std::env::temp_dir().join(format!("microgpt-compile-{}", std::process::id()));
    fs::create_dir_all(&temp_dir).map_err(|e| CompileError::IoError(format!("{e}")))?;

    let c_path = temp_dir.join("input.c");
    let wasm_path = temp_dir.join("output.wasm");

    fs::write(&c_path, c_source).map_err(|e| CompileError::IoError(format!("{e}")))?;

    let cc = find_clang()?;

    let runtime_include = format!("-include{}", runtime_h_path.display());

    let result = Command::new(&cc)
        .args([
            "--target=wasm32",
            "-nostdlib",
            "-O2",
            "-fno-builtin",
            "-fno-jump-tables",
            "-mllvm",
            "--combiner-store-merging=false",
            "-Wl,--no-entry",
            "-Wl,--export=compute",
            "-Wl,--export=__heap_base",
            "-Wl,-z,stack-size=4096",
            "-Wl,--initial-memory=10485760",
        ])
        .arg(&runtime_include)
        .arg("-o")
        .arg(&wasm_path)
        .arg(&c_path)
        .output()
        .map_err(|e| CompileError::ClangFailed(format!("Failed to execute clang: {e}")))?;

    if !result.status.success() {
        let stderr = String::from_utf8_lossy(&result.stderr);
        return Err(CompileError::ClangFailed(format!(
            "exit {}: {}",
            result.status.code().unwrap_or(-1),
            stderr.trim()
        )));
    }

    fs::read(&wasm_path).map_err(|e| CompileError::IoError(format!("{e}")))
}

// ── WASM → Dispatch Table ─────────────────────────────────────

/// Compile a single WASM function to a flat dispatch table.
///
/// Converts WASM opcodes to dispatch opcode names, handles control flow
/// (BLOCK/LOOP/IF/ELSE/END) with branch resolution via label stack.
///
/// # Arguments
/// * `func` - The WASM function body
/// * `module` - The parent WASM module (for import resolution)
/// * `local_func_idx` - Index of this function in `module.functions`
/// * `is_main` - Whether this is the main `compute` function (affects RETURN → halt)
/// * `global_temp_local` - Temp local index for global.set lowering, if needed
fn compile_function(
    func: &FuncBody,
    module: &WasmModule,
    _local_func_idx: usize,
    is_main: bool,
    global_temp_local: Option<i32>,
) -> Result<Vec<DispatchEntry>, CompileError> {
    // Build import map: func_index → import name
    let mut func_import_idx = 0usize;
    let mut import_map: HashMap<usize, &str> = HashMap::new();
    for imp in &module.imports {
        if imp.kind == ImportKind::Func {
            import_map.insert(func_import_idx, &imp.name);
            func_import_idx += 1;
        }
    }
    let num_imports = func_import_idx;

    let mut entries: Vec<DispatchEntry> = Vec::new();
    let mut label_stack: Vec<LabelFrame> = Vec::new();

    for instr in &func.instructions {
        let op = instr.opcode;

        if op == OP_NOP {
            continue;
        }

        // ── Control flow ────────────────────────────────────

        if op == OP_BLOCK {
            label_stack.push(LabelFrame {
                kind: LabelKind::Block,
                start_pc: entries.len(),
                patches: Vec::new(),
                if_entry: None,
            });
            continue;
        }

        if op == OP_LOOP {
            label_stack.push(LabelFrame {
                kind: LabelKind::Loop,
                start_pc: entries.len(),
                patches: Vec::new(),
                if_entry: None,
            });
            continue;
        }

        if op == OP_IF {
            entries.push(("i32.eqz".to_string(), 0));
            let br_idx = entries.len();
            entries.push(("br_if".to_string(), PLACEHOLDER));
            label_stack.push(LabelFrame {
                kind: LabelKind::If,
                start_pc: entries.len() - 2,
                patches: vec![br_idx],
                if_entry: Some(br_idx),
            });
            continue;
        }

        if op == OP_ELSE {
            let frame = label_stack
                .last_mut()
                .ok_or_else(|| CompileError::Other("ELSE without matching IF".to_string()))?;
            if frame.kind != LabelKind::If {
                return Err(CompileError::Other("ELSE without matching IF".to_string()));
            }
            let else_br_idx = entries.len();
            entries.push(("br".to_string(), PLACEHOLDER));
            let else_pc = entries.len();
            frame.patches = vec![else_br_idx];
            if let Some(if_entry) = frame.if_entry {
                entries[if_entry].1 = else_pc as i32;
            }
            continue;
        }

        if op == OP_END {
            if label_stack.is_empty() {
                entries.push((if is_main { "halt" } else { "return" }.to_string(), 0));
                continue;
            }
            let frame = label_stack.pop().unwrap();
            let end_pc = entries.len();
            for idx in &frame.patches {
                entries[*idx].1 = end_pc as i32;
            }
            continue;
        }

        // ── Branches ────────────────────────────────────────

        if op == OP_BR {
            let label_idx = instr.immediates[0] as usize;
            let target_frame_idx =
                label_stack
                    .len()
                    .checked_sub(label_idx + 1)
                    .ok_or_else(|| {
                        CompileError::Other(format!("BR label index {label_idx} out of range"))
                    })?;
            let target_frame = &label_stack[target_frame_idx];
            if target_frame.kind == LabelKind::Loop {
                entries.push(("br".to_string(), target_frame.start_pc as i32));
            } else {
                let idx = entries.len();
                entries.push(("br".to_string(), PLACEHOLDER));
                label_stack[target_frame_idx].patches.push(idx);
            }
            continue;
        }

        if op == OP_BR_IF {
            let label_idx = instr.immediates[0] as usize;
            let target_frame_idx =
                label_stack
                    .len()
                    .checked_sub(label_idx + 1)
                    .ok_or_else(|| {
                        CompileError::Other(format!("BR_IF label index {label_idx} out of range"))
                    })?;
            let target_frame = &label_stack[target_frame_idx];
            if target_frame.kind == LabelKind::Loop {
                entries.push(("br_if".to_string(), target_frame.start_pc as i32));
            } else {
                let idx = entries.len();
                entries.push(("br_if".to_string(), PLACEHOLDER));
                label_stack[target_frame_idx].patches.push(idx);
            }
            continue;
        }

        if op == OP_RETURN {
            entries.push((if is_main { "halt" } else { "return" }.to_string(), 0));
            continue;
        }

        if op == OP_UNREACHABLE {
            entries.push(("halt".to_string(), 0));
            continue;
        }

        // ── Calls ───────────────────────────────────────────

        if op == OP_CALL {
            let fi = instr.immediates[0] as usize;
            if fi < num_imports {
                match import_map.get(&fi) {
                    Some(&"output_byte") => {
                        entries.push(("output".to_string(), 0));
                    }
                    _ => {
                        return Err(CompileError::UnsupportedOpcode(format!(
                            "CALL to import {fi}"
                        )));
                    }
                }
            } else {
                entries.push(("call".to_string(), fi as i32));
            }
            continue;
        }

        // ── Globals ─────────────────────────────────────────

        if op == OP_GLOBAL_GET {
            let gidx = instr.immediates[0];
            entries.push(("i32.const".to_string(), (GLOBAL_BASE + 4 * gidx) as i32));
            entries.push(("i32.load".to_string(), 0));
            continue;
        }

        if op == OP_GLOBAL_SET {
            let gidx = instr.immediates[0];
            let temp = global_temp_local.ok_or_else(|| {
                CompileError::Other("global.set requires a temp local".to_string())
            })?;
            entries.push(("local.set".to_string(), temp));
            entries.push(("i32.const".to_string(), (GLOBAL_BASE + 4 * gidx) as i32));
            entries.push(("local.get".to_string(), temp));
            entries.push(("i32.store".to_string(), 0));
            continue;
        }

        // ── Locals ──────────────────────────────────────────

        if op == OP_LOCAL_GET || op == OP_LOCAL_SET || op == OP_LOCAL_TEE {
            let name = wasm_opcode_to_name(op).unwrap_or("???");
            entries.push((name.to_string(), instr.immediates[0] as i32));
            continue;
        }

        // ── Constants ───────────────────────────────────────

        if op == OP_I32_CONST {
            entries.push((
                "i32.const".to_string(),
                (instr.immediates[0] & MASK32) as i32,
            ));
            continue;
        }

        // ── Memory ops ──────────────────────────────────────

        if matches!(
            op,
            OP_I32_LOAD
                | OP_I32_LOAD8_S
                | OP_I32_LOAD8_U
                | OP_I32_LOAD16_S
                | OP_I32_LOAD16_U
                | OP_I32_STORE
                | OP_I32_STORE8
                | OP_I32_STORE16
        ) {
            let name = wasm_opcode_to_name(op).unwrap_or("???");
            let offset = instr.immediates[1] as i32;
            entries.push((name.to_string(), offset));
            continue;
        }

        // ── Simple ops (no immediates) ──────────────────────

        if let Some(name) = wasm_opcode_to_name(op) {
            entries.push((name.to_string(), 0));
            continue;
        }

        // ── Unsupported ─────────────────────────────────────
        let op_name = WASM_OP_NAMES.get(&op).unwrap_or("???");
        return Err(CompileError::UnsupportedOpcode(format!(
            "unsupported wasm opcode: {op_name}"
        )));
    }

    Ok(entries)
}

// ── Build Program Helpers ─────────────────────────────────────

/// Return the temp local index for global.set, or None if not needed.
fn func_global_temp(func: &FuncBody, module: &WasmModule, local_func_idx: usize) -> Option<i32> {
    let uses_global_set = func
        .instructions
        .iter()
        .any(|ins| ins.opcode == OP_GLOBAL_SET);
    if !uses_global_set {
        return None;
    }
    let type_idx = module.func_type_indices[local_func_idx] as usize;
    let param_count = module.types[type_idx].params.len() as i32;
    Some(param_count + func.num_locals as i32)
}

/// Compute input base address from `__heap_base` global export.
fn compute_input_base(module: &WasmModule) -> Result<i32, CompileError> {
    for exp in &module.exports {
        if exp.name == "__heap_base" && exp.kind == ExportKind::Global {
            let init = module.globals[exp.index as usize].init;
            return Ok((init + 15) & !15);
        }
    }
    Err(CompileError::Other(
        "__heap_base not exported — add -Wl,--export=__heap_base to linker flags".to_string(),
    ))
}

/// Adjust branch targets in a compiled function body by adding an offset.
fn adjust_branches(body: &[DispatchEntry], offset: usize) -> Vec<DispatchEntry> {
    body.iter()
        .map(|(op, imm)| {
            if op == "br" || op == "br_if" {
                (op.clone(), imm + offset as i32)
            } else {
                (op.clone(), *imm)
            }
        })
        .collect()
}

// ── Build Program ─────────────────────────────────────────────

/// Build the full dispatch table including prologue, main function, and helpers.
///
/// Returns `(program, input_base)`.
fn build_program(module: &WasmModule) -> Result<(Vec<DispatchEntry>, i32), CompileError> {
    let num_imports = module.num_imported_funcs();

    // Find compute export → main function
    let mut main_local_idx = 0usize;
    for exp in &module.exports {
        if exp.kind == ExportKind::Func && exp.name == "compute" {
            main_local_idx = exp.index as usize - num_imports;
            break;
        }
    }

    let func = &module.functions[main_local_idx];
    let type_idx = module.func_type_indices[main_local_idx] as usize;
    let param_count = module.types[type_idx].params.len();
    let num_locals = param_count + func.num_locals as usize;

    let input_base = if param_count > 0 {
        compute_input_base(module)?
    } else {
        0
    };
    let entry_args: Vec<i32> = if param_count > 0 {
        vec![input_base]
    } else {
        Vec::new()
    };

    let mut prologue: Vec<DispatchEntry> = Vec::new();

    // Part 0: input_base instruction
    if param_count > 0 {
        prologue.push(("input_base".to_string(), input_base));
    }

    // Part 1: local variable initialization (reverse order)
    for k in (0..num_locals).rev() {
        let init_val = if k < entry_args.len() {
            entry_args[k]
        } else {
            0
        };
        prologue.push(("i32.const".to_string(), init_val));
        prologue.push(("local.set".to_string(), k as i32));
    }

    // Part 2: memory initialization (globals + data segments, skip zeros)
    let mut initial_memory: BTreeMap<i32, u8> = BTreeMap::new();

    // Find which globals are actually used
    let mut used_globals: HashSet<usize> = HashSet::new();
    for fn_body in &module.functions {
        for ins in &fn_body.instructions {
            if ins.opcode == OP_GLOBAL_GET || ins.opcode == OP_GLOBAL_SET {
                used_globals.insert(ins.immediates[0] as usize);
            }
        }
    }

    // Initialize used globals
    for gidx in 0..module.globals.len() {
        if !used_globals.contains(&gidx) {
            continue;
        }
        let gval = module.globals[gidx].init as i64 & MASK32;
        let addr = GLOBAL_BASE + 4 * gidx as i64;
        for b in 0..4u32 {
            initial_memory.insert(addr as i32 + b as i32, ((gval >> (8 * b)) & 0xFF) as u8);
        }
    }

    // Initialize data segments
    for seg in &module.data_segments {
        for (i, &byte_val) in seg.data.iter().enumerate() {
            initial_memory.insert(seg.offset + i as i32, byte_val);
        }
    }

    // Emit non-zero memory bytes
    for (&addr, &byte_val) in &initial_memory {
        if byte_val == 0 {
            continue;
        }
        prologue.push(("i32.const".to_string(), addr));
        prologue.push(("i32.const".to_string(), byte_val as i32));
        prologue.push(("i32.store8".to_string(), 0));
    }

    // Part 3: compile main function body
    let gt = func_global_temp(func, module, main_local_idx);
    let body0 = compile_function(func, module, main_local_idx, true, gt)?;
    let mut program = prologue;
    let body_adjusted = adjust_branches(&body0, program.len());
    program.extend(body_adjusted);

    // Part 4: compile helper functions with parameter prologues
    let mut func_addresses: HashMap<usize, usize> = HashMap::new();
    for fi in 0..module.functions.len() {
        if fi == main_local_idx {
            continue;
        }
        let func_fi = &module.functions[fi];
        let ti = module.func_type_indices[fi] as usize;
        let n_params = module.types[ti].params.len();

        let func_start = program.len();
        func_addresses.insert(num_imports + fi, func_start);

        // Parameter prologue: set locals from stack (reverse order)
        for k in (0..n_params).rev() {
            program.push(("local.set".to_string(), k as i32));
        }

        let gt_fi = func_global_temp(func_fi, module, fi);
        let body_fi = compile_function(func_fi, module, fi, false, gt_fi)?;
        let body_fi_adjusted = adjust_branches(&body_fi, func_start + n_params);
        program.extend(body_fi_adjusted);

        // Patch return instructions with distance from func_start
        for (j, entry) in program.iter_mut().enumerate().skip(func_start) {
            if entry.0 == "return" {
                let d_local = (j - func_start) as i32;
                entry.1 = !d_local;
            }
        }
    }

    // Part 5: resolve CALL targets (func_idx → absolute cursor position)
    for entry in program.iter_mut() {
        if entry.0 == "call" {
            let fi = entry.1 as usize;
            let target = *func_addresses.get(&fi).ok_or_else(|| {
                CompileError::Other(format!("call to unknown function index {fi}"))
            })?;
            entry.1 = target as i32;
        }
    }

    // Part 6: convert absolute branch/call targets to relative offsets
    for (i, entry) in program.iter_mut().enumerate() {
        if matches!(entry.0.as_str(), "br" | "br_if" | "call") {
            let target = entry.1;
            entry.1 = target - i as i32 - 1;
        }
    }

    Ok((program, input_base))
}

// ── Formatting ────────────────────────────────────────────────

/// Convert dispatch table to token prefix string with `{` `}` delimiters.
pub fn format_prefix(program: &[DispatchEntry]) -> String {
    let mut lines = vec!["{".to_string()];
    for (op, imm) in program {
        let bytes = int_to_bytes(*imm as i64);
        let hex_bytes: Vec<String> = bytes.iter().map(|b| format!("{b:02x}")).collect();
        lines.push(format!("{op} {}", hex_bytes.join(" ")));
    }
    lines.push("}".to_string());
    lines.join("\n") + "\n"
}

/// Format input bytes + commit token for appending after the program.
pub fn format_input_section(input_str: &str) -> String {
    let data = input_str.as_bytes();
    let mut tokens: Vec<String> = Vec::with_capacity(data.len() + 2);

    for &b in data {
        if (0x20..0x7F).contains(&b) && b != b'{' && b != b'}' {
            tokens.push(String::from(b as char));
        } else {
            tokens.push(format!("{b:02x}"));
        }
    }
    // Null terminator
    tokens.push("00".to_string());
    tokens.push("commit(+0,sts=0,bt=0)".to_string());
    tokens.join(" ") + "\n"
}

/// Format the specialized model input (start + optional input tokens).
pub fn format_spec_input(input_str: &str) -> String {
    let mut tokens = vec!["start".to_string()];
    if !input_str.is_empty() {
        let data = input_str.as_bytes();
        for &b in data {
            if (0x20..0x7F).contains(&b) && b != b'{' && b != b'}' {
                tokens.push(String::from(b as char));
            } else {
                tokens.push(format!("{b:02x}"));
            }
        }
        tokens.push("00".to_string());
        tokens.push("commit(+0,sts=0,bt=0)".to_string());
    }
    tokens.join(" ") + "\n"
}

// ── Pipeline ──────────────────────────────────────────────────

/// Full pipeline: WASM bytes → decode → lower → dispatch table → prefix.
pub fn compile_wasm_to_prefix(wasm_bytes: &[u8]) -> Result<CompiledProgram, CompileError> {
    let mut module = decoder::decode(wasm_bytes)?;

    // Lower hard ops for all functions
    for fi in 0..module.functions.len() {
        let type_idx = module.func_type_indices[fi] as usize;
        let num_params = module.types[type_idx].params.len() as u32;
        module.functions[fi] = lower_hard_ops(&module.functions[fi], num_params);
    }

    let (program, input_base) = build_program(&module)?;
    let prefix = format_prefix(&program);

    Ok(CompiledProgram {
        program,
        prefix,
        input_base,
        input_section: String::new(),
    })
}

/// End-to-end compilation: C source + input → compiled program with prefix.
pub fn compile_program(c_source: &str, input_str: &str) -> Result<CompiledProgram, CompileError> {
    // Write runtime.h to temp dir
    let temp_dir = std::env::temp_dir().join(format!("microgpt-compile-{}", std::process::id()));
    fs::create_dir_all(&temp_dir).map_err(|e| CompileError::IoError(format!("{e}")))?;

    let runtime_h_path = write_runtime_h(&temp_dir)?;

    // Compile C → WASM
    let wasm_bytes = compile_c_to_wasm(c_source, &runtime_h_path)?;

    // WASM → prefix
    let mut result = compile_wasm_to_prefix(&wasm_bytes)?;

    // Format input section
    if !input_str.is_empty() && result.input_base > 0 {
        result.input_section = format_input_section(input_str);
    }

    Ok(result)
}

// ── Tests ─────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::percepta::wasm::decoder::{Export, FuncType, Global, Import, WasmInstr};

    #[test]
    fn test_int_to_bytes() {
        // Zero
        assert_eq!(int_to_bytes(0), [0x00, 0x00, 0x00, 0x00]);

        // Small value
        assert_eq!(int_to_bytes(1), [0x01, 0x00, 0x00, 0x00]);

        // 256
        assert_eq!(int_to_bytes(256), [0x00, 0x01, 0x00, 0x00]);

        // 0x48 = 'H'
        assert_eq!(int_to_bytes(0x48), [0x48, 0x00, 0x00, 0x00]);

        // 65536 (heap base)
        assert_eq!(int_to_bytes(65536), [0x00, 0x00, 0x01, 0x00]);

        // Large value
        assert_eq!(int_to_bytes(0x12345678), [0x78, 0x56, 0x34, 0x12]);

        // Masked negative: -1 as i64 → 0xFFFFFFFF
        assert_eq!(int_to_bytes(-1), [0xFF, 0xFF, 0xFF, 0xFF]);

        // Value exceeding 32 bits gets masked
        assert_eq!(int_to_bytes(0x1_0000_0000), [0x00, 0x00, 0x00, 0x00]);
    }

    #[test]
    fn test_find_clang() {
        // Just verify it doesn't panic and returns a reasonable result.
        // It may succeed or fail depending on the system.
        match find_clang() {
            Ok(path) => {
                assert!(path.exists(), "clang path should exist: {path:?}");
            }
            Err(CompileError::ClangNotFound(_)) => {
                // Expected on systems without wasm32-capable clang
            }
            Err(e) => panic!("Unexpected error: {e}"),
        }
    }

    #[test]
    fn test_compile_function_simple() {
        // Create a minimal module with one function import
        let module = WasmModule {
            types: vec![FuncType {
                params: vec![decoder::VALTYPE_I32],
                results: vec![],
            }],
            imports: vec![Import {
                module: "env".to_string(),
                name: "output_byte".to_string(),
                kind: ImportKind::Func,
                index: 0,
            }],
            func_type_indices: vec![0],
            exports: vec![],
            functions: vec![FuncBody {
                locals: vec![],
                num_locals: 0,
                instructions: vec![
                    WasmInstr::with_imms(OP_I32_CONST, vec![72]), // 'H'
                    WasmInstr::with_imms(OP_CALL, vec![0]),       // output_byte
                    WasmInstr::new(OP_END),
                ],
            }],
            data_segments: vec![],
            globals: vec![],
        };

        let result = compile_function(&module.functions[0], &module, 0, true, None).unwrap();

        // Should produce: i32.const 72, output 0, halt 0
        assert_eq!(result.len(), 3);
        assert_eq!(result[0], ("i32.const".to_string(), 72));
        assert_eq!(result[1], ("output".to_string(), 0));
        assert_eq!(result[2], ("halt".to_string(), 0));
    }

    #[test]
    fn test_compile_function_with_branch() {
        // Create a function with a simple loop
        let module = WasmModule {
            types: vec![FuncType {
                params: vec![],
                results: vec![],
            }],
            imports: vec![],
            func_type_indices: vec![0],
            exports: vec![],
            functions: vec![FuncBody {
                locals: vec![(1, decoder::VALTYPE_I32)],
                num_locals: 1,
                instructions: vec![
                    // loop
                    WasmInstr::new(OP_LOOP),
                    // local.get 0
                    WasmInstr::with_imms(OP_LOCAL_GET, vec![0]),
                    // i32.const 1
                    WasmInstr::with_imms(OP_I32_CONST, vec![1]),
                    // i32.add
                    WasmInstr::new(OP_I32_ADD),
                    // local.set 0
                    WasmInstr::with_imms(OP_LOCAL_SET, vec![0]),
                    // br 0 (back to loop)
                    WasmInstr::with_imms(OP_BR, vec![0]),
                    // end
                    WasmInstr::new(OP_END),
                ],
            }],
            data_segments: vec![],
            globals: vec![],
        };

        let result = compile_function(&module.functions[0], &module, 0, true, None).unwrap();

        // Expected:
        // 0: local.get 0
        // 1: i32.const 1
        // 2: i32.add
        // 3: local.set 0
        // 4: br 0 (→ start_pc = 0)
        // (no halt because loop is infinite, but END would emit halt)
        assert!(result.len() >= 5);
        assert_eq!(result[0].0, "local.get");
        assert_eq!(result[1].0, "i32.const");
        assert_eq!(result[2].0, "i32.add");
        assert_eq!(result[3].0, "local.set");
        assert_eq!(result[4].0, "br");
        assert_eq!(result[4].1, 0); // branch to start_pc = 0
    }

    #[test]
    fn test_compile_function_if_else() {
        let module = WasmModule {
            types: vec![FuncType {
                params: vec![decoder::VALTYPE_I32],
                results: vec![],
            }],
            imports: vec![Import {
                module: "env".to_string(),
                name: "output_byte".to_string(),
                kind: ImportKind::Func,
                index: 0,
            }],
            func_type_indices: vec![0],
            exports: vec![],
            functions: vec![FuncBody {
                locals: vec![],
                num_locals: 0,
                instructions: vec![
                    // if (param0 != 0)
                    WasmInstr::with_imms(OP_LOCAL_GET, vec![0]),
                    WasmInstr::new(OP_IF),
                    // then: output 'Y'
                    WasmInstr::with_imms(OP_I32_CONST, vec![b'Y' as i64]),
                    WasmInstr::with_imms(OP_CALL, vec![0]),
                    // else
                    WasmInstr::new(OP_ELSE),
                    // else: output 'N'
                    WasmInstr::with_imms(OP_I32_CONST, vec![b'N' as i64]),
                    WasmInstr::with_imms(OP_CALL, vec![0]),
                    // end if
                    WasmInstr::new(OP_END),
                    // end function
                    WasmInstr::new(OP_END),
                ],
            }],
            data_segments: vec![],
            globals: vec![],
        };

        let result = compile_function(&module.functions[0], &module, 0, true, None).unwrap();

        // Expected structure:
        // 0: local.get 0          (push condition)
        // 1: i32.eqz 0            (IF inverts condition)
        // 2: br_if → else_start   (skip then if false)
        // 3: i32.const 'Y'
        // 4: output 0
        // 5: br → end_if          (skip else after then)
        // 6: i32.const 'N'        (else_start)
        // 7: output 0
        // 8: halt 0               (end_if, resolved from patches)

        assert_eq!(result[1].0, "i32.eqz");
        assert_eq!(result[2].0, "br_if");
        assert_eq!(result[2].1, 6); // br_if targets else_start
        assert_eq!(result[5].0, "br");
        assert_eq!(result[5].1, 8); // br targets end_if
    }

    #[test]
    fn test_format_prefix() {
        let program: Vec<DispatchEntry> = vec![
            ("i32.const".to_string(), 72),
            ("output".to_string(), 0),
            ("halt".to_string(), 0),
        ];

        let prefix = format_prefix(&program);

        assert!(prefix.starts_with("{\n"));
        assert!(prefix.ends_with("}\n"));
        assert!(prefix.contains("i32.const 48 00 00 00"));
        assert!(prefix.contains("output 00 00 00 00"));
        assert!(prefix.contains("halt 00 00 00 00"));
    }

    #[test]
    fn test_format_input_section() {
        let result = format_input_section("Hello");

        // 'H', 'e', 'l', 'l', 'o' are printable and not {/}
        // Plus null terminator and commit
        assert!(result.contains("H"));
        assert!(result.contains("e"));
        assert!(result.contains("l"));
        assert!(result.contains("o"));
        assert!(result.contains("00")); // null terminator
        assert!(result.contains("commit(+0,sts=0,bt=0)"));
        assert!(result.ends_with('\n'));
    }

    #[test]
    fn test_format_input_section_with_special_chars() {
        let result = format_input_section("\x00\x01{");

        // Null byte → "00"
        assert!(result.contains("00"));
        // 0x01 → "01"
        assert!(result.contains("01"));
        // '{' → "7b"
        assert!(result.contains("7b"));
        assert!(result.contains("commit(+0,sts=0,bt=0)"));
    }

    #[test]
    fn test_format_spec_input() {
        let result = format_spec_input("test");
        assert!(result.starts_with("start "));
        assert!(result.contains("commit(+0,sts=0,bt=0)"));

        let empty = format_spec_input("");
        assert!(empty.starts_with("start\n"));
        assert!(!empty.contains("commit"));
    }

    #[test]
    fn test_wasm_to_name_mapping() {
        // Test all mapped opcodes
        assert_eq!(wasm_opcode_to_name(OP_I32_CONST), Some("i32.const"));
        assert_eq!(wasm_opcode_to_name(OP_LOCAL_GET), Some("local.get"));
        assert_eq!(wasm_opcode_to_name(OP_LOCAL_SET), Some("local.set"));
        assert_eq!(wasm_opcode_to_name(OP_LOCAL_TEE), Some("local.tee"));
        assert_eq!(wasm_opcode_to_name(OP_DROP), Some("drop"));
        assert_eq!(wasm_opcode_to_name(OP_SELECT), Some("select"));
        assert_eq!(wasm_opcode_to_name(OP_I32_ADD), Some("i32.add"));
        assert_eq!(wasm_opcode_to_name(OP_I32_SUB), Some("i32.sub"));
        assert_eq!(wasm_opcode_to_name(OP_I32_EQ), Some("i32.eq"));
        assert_eq!(wasm_opcode_to_name(OP_I32_NE), Some("i32.ne"));
        assert_eq!(wasm_opcode_to_name(OP_I32_LT_S), Some("i32.lt_s"));
        assert_eq!(wasm_opcode_to_name(OP_I32_LT_U), Some("i32.lt_u"));
        assert_eq!(wasm_opcode_to_name(OP_I32_GT_S), Some("i32.gt_s"));
        assert_eq!(wasm_opcode_to_name(OP_I32_GT_U), Some("i32.gt_u"));
        assert_eq!(wasm_opcode_to_name(OP_I32_LE_S), Some("i32.le_s"));
        assert_eq!(wasm_opcode_to_name(OP_I32_LE_U), Some("i32.le_u"));
        assert_eq!(wasm_opcode_to_name(OP_I32_GE_S), Some("i32.ge_s"));
        assert_eq!(wasm_opcode_to_name(OP_I32_GE_U), Some("i32.ge_u"));
        assert_eq!(wasm_opcode_to_name(OP_I32_EQZ), Some("i32.eqz"));
        assert_eq!(wasm_opcode_to_name(OP_I32_LOAD), Some("i32.load"));
        assert_eq!(wasm_opcode_to_name(OP_I32_LOAD8_S), Some("i32.load8_s"));
        assert_eq!(wasm_opcode_to_name(OP_I32_LOAD8_U), Some("i32.load8_u"));
        assert_eq!(wasm_opcode_to_name(OP_I32_LOAD16_S), Some("i32.load16_s"));
        assert_eq!(wasm_opcode_to_name(OP_I32_LOAD16_U), Some("i32.load16_u"));
        assert_eq!(wasm_opcode_to_name(OP_I32_STORE), Some("i32.store"));
        assert_eq!(wasm_opcode_to_name(OP_I32_STORE8), Some("i32.store8"));
        assert_eq!(wasm_opcode_to_name(OP_I32_STORE16), Some("i32.store16"));
        assert_eq!(wasm_opcode_to_name(OP_BR), Some("br"));
        assert_eq!(wasm_opcode_to_name(OP_BR_IF), Some("br_if"));
        assert_eq!(wasm_opcode_to_name(OP_UNREACHABLE), Some("halt"));
        assert_eq!(wasm_opcode_to_name(OP_RETURN), Some("halt"));

        // Unmapped opcodes
        assert_eq!(wasm_opcode_to_name(OP_BLOCK), None);
        assert_eq!(wasm_opcode_to_name(OP_LOOP), None);
        assert_eq!(wasm_opcode_to_name(OP_IF), None);
        assert_eq!(wasm_opcode_to_name(OP_ELSE), None);
        assert_eq!(wasm_opcode_to_name(OP_END), None);
        assert_eq!(wasm_opcode_to_name(OP_CALL), None);
        assert_eq!(wasm_opcode_to_name(OP_NOP), None);
    }

    /// Build a minimal WASM binary that exports `compute` and `__heap_base`.
    /// The compute function outputs 'H' (72) via output_byte and returns.
    fn minimal_hello_wasm() -> Vec<u8> {
        vec![
            // Magic + version
            0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
            // Type section (id=1): 1 type — func(i32) -> ()
            0x01, 0x05, 0x01, 0x60, 0x01, 0x7f, 0x00,
            // Import section (id=2): 1 import — "env"."output_byte" func 0
            0x02, 0x13, 0x01, 0x03, 0x65, 0x6e, 0x76, 0x0b, 0x6f, 0x75, 0x74, 0x70, 0x75, 0x74,
            0x5f, 0x62, 0x79, 0x74, 0x65, 0x00, 0x00,
            // Function section (id=3): 1 function — type 0
            0x03, 0x02, 0x01, 0x00,
            // Global section (id=6): 1 global — i32 immutable, init=65536
            0x06, 0x08, 0x01, 0x7f, 0x00, 0x41, 0x80, 0x80, 0x04, 0x0b,
            // Export section (id=7): 2 exports
            0x07, 0x19, 0x02, // Export "compute" func 1
            0x07, 0x63, 0x6f, 0x6d, 0x70, 0x75, 0x74, 0x65, 0x00, 0x01,
            // Export "__heap_base" global 0
            0x0b, 0x5f, 0x5f, 0x68, 0x65, 0x61, 0x70, 0x5f, 0x62, 0x61, 0x73, 0x65, 0x03, 0x00,
            // Code section (id=10): 1 body
            0x0a, 0x08, 0x01, 0x06, 0x00, // local.get 0 (input param)
            0x20, 0x00, // call 0 (output_byte)
            0x10, 0x00, // end
            0x0b,
        ]
    }

    #[test]
    fn test_compile_wasm_to_prefix() {
        let wasm_bytes = minimal_hello_wasm();
        let result = compile_wasm_to_prefix(&wasm_bytes).unwrap();

        // Should have a valid prefix
        assert!(result.prefix.starts_with("{\n"));
        assert!(result.prefix.ends_with("}\n"));

        // Should contain input_base instruction (heap_base is 65536, aligned to 65536)
        assert_eq!(result.input_base, 65536);

        // Should contain output instruction
        assert!(result.program.iter().any(|(op, _)| op == "output"));

        // Should contain halt instruction
        assert!(result.program.iter().any(|(op, _)| op == "halt"));

        // Count non-empty lines in prefix
        let lines: Vec<&str> = result
            .prefix
            .lines()
            .filter(|l| !l.is_empty() && *l != "{" && *l != "}")
            .collect();
        assert!(!lines.is_empty(), "prefix should have instructions");
    }

    #[test]
    fn test_compute_input_base_alignment() {
        // heap_base = 65536 → aligned to 65536 (already aligned)
        let module = WasmModule {
            types: vec![],
            imports: vec![],
            func_type_indices: vec![],
            exports: vec![Export {
                name: "__heap_base".to_string(),
                kind: ExportKind::Global,
                index: 0,
            }],
            functions: vec![],
            data_segments: vec![],
            globals: vec![Global {
                valtype: decoder::VALTYPE_I32,
                mutable: 0,
                init: 65536,
            }],
        };

        assert_eq!(compute_input_base(&module).unwrap(), 65536);

        // heap_base = 65537 → aligned to 65552 ((65537 + 15) & !15)
        let module2 = WasmModule {
            globals: vec![Global {
                valtype: decoder::VALTYPE_I32,
                mutable: 0,
                init: 65537,
            }],
            exports: vec![Export {
                name: "__heap_base".to_string(),
                kind: ExportKind::Global,
                index: 0,
            }],
            ..module
        };

        assert_eq!(compute_input_base(&module2).unwrap(), 65552);
    }

    #[test]
    fn test_compile_function_global_access() {
        let module = WasmModule {
            types: vec![FuncType {
                params: vec![],
                results: vec![],
            }],
            imports: vec![Import {
                module: "env".to_string(),
                name: "output_byte".to_string(),
                kind: ImportKind::Func,
                index: 0,
            }],
            func_type_indices: vec![0],
            exports: vec![],
            functions: vec![FuncBody {
                locals: vec![(1, decoder::VALTYPE_I32)],
                num_locals: 1,
                instructions: vec![
                    // global.get 0
                    WasmInstr::with_imms(OP_GLOBAL_GET, vec![0]),
                    // call output_byte
                    WasmInstr::with_imms(OP_CALL, vec![0]),
                    // end
                    WasmInstr::new(OP_END),
                ],
            }],
            data_segments: vec![],
            globals: vec![Global {
                valtype: decoder::VALTYPE_I32,
                mutable: 0,
                init: 42,
            }],
        };

        // param_count=0, num_locals=1, so temp local is at index 1
        let result = compile_function(
            &module.functions[0],
            &module,
            0,
            true,
            Some(1), // global_temp_local
        )
        .unwrap();

        // GLOBAL_GET expands to: i32.const (addr) + i32.load 0
        assert!(result.iter().any(|(op, _)| op == "i32.load"));
        // Should contain the global address: GLOBAL_BASE + 4*0 = 8
        assert!(
            result
                .iter()
                .any(|(op, imm)| op == "i32.const" && *imm == GLOBAL_BASE as i32)
        );
    }

    #[test]
    fn test_adjust_branches() {
        let body: Vec<DispatchEntry> = vec![
            ("i32.const".to_string(), 1),
            ("br".to_string(), 5),
            ("br_if".to_string(), 3),
            ("halt".to_string(), 0),
        ];

        let adjusted = adjust_branches(&body, 10);

        assert_eq!(adjusted[0], ("i32.const".to_string(), 1)); // unchanged
        assert_eq!(adjusted[1], ("br".to_string(), 15)); // 5 + 10
        assert_eq!(adjusted[2], ("br_if".to_string(), 13)); // 3 + 10
        assert_eq!(adjusted[3], ("halt".to_string(), 0)); // unchanged
    }

    #[test]
    fn test_func_global_temp() {
        let module = WasmModule {
            types: vec![FuncType {
                params: vec![decoder::VALTYPE_I32],
                results: vec![],
            }],
            imports: vec![],
            func_type_indices: vec![0],
            exports: vec![],
            functions: vec![FuncBody {
                locals: vec![(2, decoder::VALTYPE_I32)],
                num_locals: 2,
                instructions: vec![WasmInstr::with_imms(OP_GLOBAL_SET, vec![0])],
            }],
            data_segments: vec![],
            globals: vec![],
        };

        // param_count=1, num_locals=2, temp = 1+2 = 3
        assert_eq!(func_global_temp(&module.functions[0], &module, 0), Some(3));

        // Function without global.set
        let module2 = WasmModule {
            functions: vec![FuncBody {
                locals: vec![],
                num_locals: 0,
                instructions: vec![WasmInstr::new(OP_NOP)],
            }],
            ..module
        };
        assert_eq!(func_global_temp(&module2.functions[0], &module2, 0), None);
    }

    #[test]
    fn test_relative_branch_conversion() {
        // Build a minimal module and verify branches are relative
        let wasm_bytes = vec![
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
            // Code section: simple infinite loop outputting 'A'
            // loop
            //   i32.const 65
            //   call 0 (output_byte)
            //   br 0 (back to loop)
            // end
            0x0a, 0x0c, 0x01, 0x0a, 0x00, 0x03, 0x40, 0x41, 0x41, 0x10, 0x00, 0x0c, 0x00, 0x0b,
        ];

        let result = compile_wasm_to_prefix(&wasm_bytes).unwrap();

        // Find the br instruction and verify it has a negative relative offset
        // (loop back-edge)
        let br_entries: Vec<_> = result
            .program
            .iter()
            .enumerate()
            .filter(|(_, (op, _))| op == "br")
            .collect();

        assert!(!br_entries.is_empty(), "should have br instructions");
        for (idx, (_, imm)) in &br_entries {
            // Loop back-edge should have negative offset
            assert!(
                *imm < 0,
                "loop br at {idx} should have negative offset, got {imm}"
            );
        }
    }

    #[test]
    fn test_runtime_h_is_valid() {
        // Verify the embedded runtime.h is non-empty and looks like C
        assert!(!RUNTIME_H.is_empty());
        assert!(RUNTIME_H.contains("putchar"));
        assert!(RUNTIME_H.contains("compute"));
        assert!(RUNTIME_H.contains("#ifndef"));
    }

    #[test]
    fn test_write_runtime_h() {
        let temp_dir = std::env::temp_dir().join("microgpt-test-runtime");
        let _ = fs::create_dir_all(&temp_dir);
        let path = write_runtime_h(&temp_dir).unwrap();

        assert!(path.exists());
        let content = fs::read_to_string(&path).unwrap();
        assert_eq!(content, RUNTIME_H);

        // Cleanup
        let _ = fs::remove_dir_all(&temp_dir);
    }

    // ── End-to-end tests (require clang with wasm32) ──────────

    /// Helper: skip if clang with wasm32 is not available.
    fn skip_without_clang() -> Option<PathBuf> {
        match find_clang() {
            Ok(cc) => Some(cc),
            Err(CompileError::ClangNotFound(_)) => None,
            Err(e) => panic!("unexpected clang error: {e}"),
        }
    }

    #[test]
    fn test_e2e_compile_hello_c() {
        let _cc = match skip_without_clang() {
            Some(cc) => cc,
            None => {
                eprintln!("skipping: no clang with wasm32");
                return;
            }
        };

        let hello_c = r#"
void compute(const char *input) {
    print_str("Hello ");
    print_str(input);
    print_str("!\n");
}
"#;

        let result = compile_program(hello_c, "World");
        assert!(result.is_ok(), "compile failed: {:?}", result.err());

        let compiled = result.unwrap();

        // Prefix must be valid
        assert!(compiled.prefix.starts_with("{\n"));
        assert!(compiled.prefix.ends_with("}\n"));

        // Must have input_base > 0 (takes const char* input)
        assert!(
            compiled.input_base > 0,
            "input_base should be > 0, got {}",
            compiled.input_base
        );

        // Input section should contain "World"
        assert!(
            compiled.input_section.contains("W"),
            "input section should contain 'W': {}",
            compiled.input_section
        );

        // Must contain output instruction (print_str → putchar → output_byte)
        assert!(
            compiled.program.iter().any(|(op, _)| op == "output"),
            "program should contain output instruction"
        );

        // Must end with halt
        assert!(
            compiled
                .program
                .last()
                .map_or(false, |(op, _)| op == "halt"),
            "program should end with halt"
        );

        // Input section should contain commit token
        assert!(
            compiled.input_section.contains("commit(+0,sts=0,bt=0)"),
            "input section should contain commit token"
        );

        eprintln!(
            "hello.c: {} instructions, input_base={}",
            compiled.program.len(),
            compiled.input_base
        );
    }

    #[test]
    fn test_e2e_compile_collatz_c() {
        let _cc = match skip_without_clang() {
            Some(cc) => cc,
            None => {
                eprintln!("skipping: no clang with wasm32");
                return;
            }
        };

        let collatz_c = r#"
void compute(const char *input) {
    int n;
    sscanf(input, "%d", &n);
    if (n <= 0) { printf("need n>0\n"); return; }

    printf("%d", n);
    while (n != 1) {
        if (n % 2 == 0) {
            n = n / 2;
        } else {
            n = 3 * n + 1;
        }
        printf(" %d", n);
    }
    printf("\n");
}
"#;

        let result = compile_program(collatz_c, "7");
        assert!(result.is_ok(), "compile failed: {:?}", result.err());

        let compiled = result.unwrap();

        assert!(compiled.prefix.starts_with("{\n"));
        assert!(compiled.prefix.ends_with("}\n"));
        assert!(compiled.input_base > 0);

        // Collatz has loops, so should have br instructions
        assert!(
            compiled
                .program
                .iter()
                .any(|(op, _)| op == "br" || op == "br_if"),
            "collatz should have branch instructions (loop)"
        );

        // Should have output instructions
        assert!(
            compiled.program.iter().any(|(op, _)| op == "output"),
            "collatz should have output instructions (printf → putchar)"
        );

        eprintln!(
            "collatz.c: {} instructions, input_base={}",
            compiled.program.len(),
            compiled.input_base
        );
    }

    #[test]
    fn test_e2e_compile_c_to_wasm_only() {
        let _cc = match skip_without_clang() {
            Some(cc) => cc,
            None => {
                eprintln!("skipping: no clang with wasm32");
                return;
            }
        };

        let simple_c = r#"
void compute(const char *input) {
    putchar('A');
    putchar('B');
}
"#;

        let temp_dir = std::env::temp_dir().join(format!("microgpt-e2e-{}", std::process::id()));
        let _ = fs::create_dir_all(&temp_dir);
        let runtime_h = write_runtime_h(&temp_dir).unwrap();

        // Step 1: C → WASM bytes
        let wasm_bytes = compile_c_to_wasm(simple_c, &runtime_h);
        assert!(
            wasm_bytes.is_ok(),
            "compile_c_to_wasm failed: {:?}",
            wasm_bytes.err()
        );
        let wasm_bytes = wasm_bytes.unwrap();

        // Verify WASM magic
        assert_eq!(
            &wasm_bytes[0..4],
            &[0x00, 0x61, 0x73, 0x6d],
            "should have WASM magic"
        );

        // Step 2: WASM → prefix
        let result = compile_wasm_to_prefix(&wasm_bytes);
        assert!(
            result.is_ok(),
            "compile_wasm_to_prefix failed: {:?}",
            result.err()
        );

        let compiled = result.unwrap();
        assert!(compiled.program.iter().any(|(op, _)| op == "output"));
        assert!(compiled.program.iter().any(|(op, _)| op == "halt"));

        // Cleanup
        let _ = fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_e2e_compile_no_input_program() {
        let _cc = match skip_without_clang() {
            Some(cc) => cc,
            None => {
                eprintln!("skipping: no clang with wasm32");
                return;
            }
        };

        // A program that doesn't take input
        let no_input_c = r#"
void compute(void) {
    putchar('O');
    putchar('K');
}
"#;

        // compile_program should still work — but no input_base
        let result = compile_c_to_wasm_only(no_input_c);
        match result {
            Ok(_wasm) => {
                // Some clang versions may still produce a valid binary
                // even without __heap_base for no-arg compute
            }
            Err(CompileError::Other(msg)) if msg.contains("__heap_base") => {
                // Expected: no-arg compute may not export __heap_base
                // which is fine — this program doesn't need input
            }
            Err(e) => panic!("unexpected error: {e}"),
        }
    }

    fn compile_c_to_wasm_only(c_source: &str) -> Result<Vec<u8>, CompileError> {
        let temp_dir =
            std::env::temp_dir().join(format!("microgpt-e2e-noinput-{}", std::process::id()));
        let _ = fs::create_dir_all(&temp_dir);
        let runtime_h = write_runtime_h(&temp_dir)?;
        let result = compile_c_to_wasm(c_source, &runtime_h);
        let _ = fs::remove_dir_all(&temp_dir);
        result
    }
}
