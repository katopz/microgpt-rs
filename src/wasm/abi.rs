//! WASM ABI constants and memory layout helpers.
//!
//! Defines the memory layout convention for validator WASM modules and provides
//! safe helper functions for reading/writing data to WASM linear memory.
//!
//! # Memory Layout
//!
//! ```text
//! WASM Linear Memory:
//!   0x000000 - 0x0000FF  Validator State (256 bytes)
//!   0x000100 - 0x0001FF  Validator Name (max 256 bytes, null-terminated)
//!   0x000200 - 0x001FFF  Scratch Buffer (7.5 KB for parent_tokens + strings)
//!   0x002000+            Validator Heap
//! ```

use wasmtime::{Memory, Store};

use super::state::ValidatorState;

// ── Memory Offsets ───────────────────────────────────────────────

/// Offset for validator state region in WASM linear memory.
pub const VALIDATOR_STATE_OFFSET: u32 = 0x000000;

/// Offset for validator name in WASM linear memory (null-terminated, max 256 bytes).
pub const VALIDATOR_NAME_OFFSET: u32 = 0x000100;

/// Offset for scratch buffer in WASM linear memory.
/// Used for passing parent tokens and strings between host and WASM.
pub const SCRATCH_BUFFER_OFFSET: u32 = 0x000200;

/// Size of the scratch buffer in bytes (7.5 KB).
pub const SCRATCH_BUFFER_SIZE: u32 = 0x001E00;

/// Maximum number of parent tokens that can be passed to `is_valid`.
/// Effective limit may be lower due to scratch buffer size (7680 / 4 = 1920).
pub const MAX_PARENT_TOKENS: usize = 2048;

/// Maximum WASM memory pages (64 pages × 64 KB = 4 MB).
pub const MAX_MEMORY_PAGES: u64 = 64;

/// Fuel budget per call (≈100μs of execution).
pub const FUEL_PER_CALL: u64 = 100_000;

// ── Return Values ────────────────────────────────────────────────

/// Token validation result: valid.
pub const VALID: u32 = 1;

/// Token validation result: invalid.
pub const INVALID: u32 = 0;

// ── Export Names ─────────────────────────────────────────────────

/// Required export: `is_valid(depth, token_idx, parent_ptr, parent_len) -> i32`.
pub const EXPORT_IS_VALID: &str = "is_valid";

/// Optional export: `relevance(depth, token_idx, parent_ptr, parent_len) -> u32`.
/// Returns Q16.16 fixed-point relevance score (0x00000000=0.0, 0x00010000=1.0).
/// If missing, host falls back to `is_valid` with binary 0/1 → 0.0/1.0.
pub const EXPORT_RELEVANCE: &str = "relevance";

/// Optional export: `validate_string(ptr, len) -> i32`.
pub const EXPORT_VALIDATE_STRING: &str = "validate_string";

/// Required export: `name() -> i32` (returns pointer to null-terminated name).
pub const EXPORT_NAME: &str = "name";

/// Required export: `version() -> i32` (returns packed u32: major<<16 | minor<<8 | patch).
pub const EXPORT_VERSION: &str = "version";

/// Required export: linear memory.
pub const EXPORT_MEMORY: &str = "memory";

// ── Memory Helpers ───────────────────────────────────────────────

/// Write parent tokens to WASM scratch buffer as a packed u32 array.
///
/// Each `usize` token is truncated to `u32` and written in little-endian format
/// starting at [`SCRATCH_BUFFER_OFFSET`].
///
/// # Returns
///
/// `(ptr, len)` where `ptr` is the WASM memory offset and `len` is the token count.
///
/// # Errors
///
/// Returns an error if the token count exceeds [`MAX_PARENT_TOKENS`] or the
/// scratch buffer size.
pub fn write_parent_tokens(
    memory: &Memory,
    store: &mut Store<ValidatorState>,
    tokens: &[usize],
) -> Result<(u32, u32), String> {
    if tokens.len() > MAX_PARENT_TOKENS {
        return Err(format!(
            "too many parent tokens: {} > {MAX_PARENT_TOKENS}",
            tokens.len()
        ));
    }

    let byte_len = tokens.len() * size_of::<u32>();
    if byte_len as u32 > SCRATCH_BUFFER_SIZE {
        return Err(format!(
            "parent tokens exceed scratch buffer: {byte_len} > {SCRATCH_BUFFER_SIZE}"
        ));
    }

    let offset = SCRATCH_BUFFER_OFFSET as usize;
    let mem = memory.data_mut(store);

    if offset + byte_len > mem.len() {
        return Err(format!(
            "WASM memory too small: need {byte_len} bytes at offset {offset}, have {}",
            mem.len()
        ));
    }

    for (i, &token) in tokens.iter().enumerate() {
        let start = offset + i * size_of::<u32>();
        let bytes = (token as u32).to_le_bytes();
        mem[start..start + size_of::<u32>()].copy_from_slice(&bytes);
    }

    Ok((SCRATCH_BUFFER_OFFSET, tokens.len() as u32))
}

/// Write a string to WASM scratch buffer.
///
/// The string bytes are written starting at [`SCRATCH_BUFFER_OFFSET`].
/// Note: parent tokens and strings share the same scratch buffer region,
/// so they should not be used simultaneously in the same call.
///
/// # Returns
///
/// `(ptr, len)` where `ptr` is the WASM memory offset and `len` is the byte count.
///
/// # Errors
///
/// Returns an error if the string exceeds [`SCRATCH_BUFFER_SIZE`].
pub fn write_string(
    memory: &Memory,
    store: &mut Store<ValidatorState>,
    s: &str,
) -> Result<(u32, u32), String> {
    let bytes = s.as_bytes();
    let byte_len = bytes.len();

    if byte_len as u32 > SCRATCH_BUFFER_SIZE {
        return Err(format!(
            "string exceeds scratch buffer: {byte_len} > {SCRATCH_BUFFER_SIZE}"
        ));
    }

    let offset = SCRATCH_BUFFER_OFFSET as usize;
    let mem = memory.data_mut(store);

    if offset + byte_len > mem.len() {
        return Err(format!(
            "WASM memory too small: need {byte_len} bytes at offset {offset}, have {}",
            mem.len()
        ));
    }

    mem[offset..offset + byte_len].copy_from_slice(bytes);

    Ok((SCRATCH_BUFFER_OFFSET, byte_len as u32))
}

/// Read a null-terminated C string from WASM linear memory.
///
/// Reads bytes starting at `ptr` until a null byte (0x00) is found or `max_len`
/// bytes have been read, whichever comes first.
///
/// # Errors
///
/// Returns an error if `ptr` is out of bounds, no null terminator is found
/// within `max_len` bytes, or the bytes are not valid UTF-8.
pub fn read_cstring(
    memory: &Memory,
    store: &Store<ValidatorState>,
    ptr: u32,
    max_len: usize,
) -> Result<String, String> {
    let ptr = ptr as usize;
    let mem = memory.data(store);

    if ptr >= mem.len() {
        return Err(format!(
            "string pointer out of bounds: {ptr} >= {}",
            mem.len()
        ));
    }

    let available = mem.len() - ptr;
    let search_len = max_len.min(available);

    match mem[ptr..ptr + search_len].iter().position(|&b| b == 0) {
        Some(null_pos) => String::from_utf8(mem[ptr..ptr + null_pos].to_vec())
            .map_err(|e| format!("invalid UTF-8 in string at ptr {ptr}: {e}")),
        None => Err(format!(
            "no null terminator found within {max_len} bytes at ptr {ptr}"
        )),
    }
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Create a minimal WASM instance with exported memory for testing.
    fn create_test_store_and_memory() -> (wasmtime::Store<ValidatorState>, Memory) {
        let engine = wasmtime::Engine::default();
        let wat_str = "(module (memory (export \"memory\") 4))";
        let module = wasmtime::Module::new(&engine, wat_str).expect("test WAT should be valid");
        let mut store = wasmtime::Store::new(&engine, ValidatorState::placeholder());
        let linker = wasmtime::Linker::new(&engine);
        let instance = linker
            .instantiate(&mut store, &module)
            .expect("test instantiation should succeed");
        let memory = instance
            .get_memory(&mut store, EXPORT_MEMORY)
            .expect("test module should export memory");
        (store, memory)
    }

    #[test]
    fn test_constants_are_consistent() {
        assert_eq!(SCRATCH_BUFFER_SIZE, 0x001E00);
        assert_eq!(SCRATCH_BUFFER_OFFSET, 0x000200);
        assert_eq!(VALIDATOR_NAME_OFFSET, 0x000100);
        assert_eq!(VALIDATOR_STATE_OFFSET, 0x000000);
        assert_eq!(VALID, 1);
        assert_eq!(INVALID, 0);
        assert_eq!(FUEL_PER_CALL, 100_000);
        assert_eq!(MAX_MEMORY_PAGES, 64);
    }

    #[test]
    fn test_scratch_buffer_fits_in_memory_gap() {
        let scratch_end = SCRATCH_BUFFER_OFFSET + SCRATCH_BUFFER_SIZE;
        // Scratch buffer should end at 0x002000 (heap start)
        assert_eq!(scratch_end, 0x002000);
    }

    #[test]
    fn test_write_parent_tokens_empty() {
        let (mut store, memory) = create_test_store_and_memory();
        let (ptr, len) =
            write_parent_tokens(&memory, &mut store, &[]).expect("empty tokens should work");
        assert_eq!(ptr, SCRATCH_BUFFER_OFFSET);
        assert_eq!(len, 0);
    }

    #[test]
    fn test_write_parent_tokens_single() {
        let (mut store, memory) = create_test_store_and_memory();
        let tokens = [42usize];
        let (ptr, len) =
            write_parent_tokens(&memory, &mut store, &tokens).expect("single token should work");
        assert_eq!(ptr, SCRATCH_BUFFER_OFFSET);
        assert_eq!(len, 1);

        // Verify the token was written correctly as u32 LE
        let mem = memory.data(&store);
        let offset = SCRATCH_BUFFER_OFFSET as usize;
        let written = u32::from_le_bytes(mem[offset..offset + 4].try_into().expect("4 bytes"));
        assert_eq!(written, 42u32);
    }

    #[test]
    fn test_write_parent_tokens_multiple() {
        let (mut store, memory) = create_test_store_and_memory();
        let tokens = [10usize, 20, 30, 40, 50];
        let (ptr, len) =
            write_parent_tokens(&memory, &mut store, &tokens).expect("multiple tokens should work");
        assert_eq!(ptr, SCRATCH_BUFFER_OFFSET);
        assert_eq!(len, 5);

        let mem = memory.data(&store);
        let offset = SCRATCH_BUFFER_OFFSET as usize;
        for (i, &expected) in tokens.iter().enumerate() {
            let start = offset + i * 4;
            let written = u32::from_le_bytes(mem[start..start + 4].try_into().expect("4 bytes"));
            assert_eq!(written, expected as u32, "mismatch at index {i}");
        }
    }

    #[test]
    fn test_write_parent_tokens_overflow_truncation() {
        let (mut store, memory) = create_test_store_and_memory();
        // usize value that truncates to u32
        let tokens = [0x1_0000_0042usize]; // truncates to 0x42
        let (_, _) = write_parent_tokens(&memory, &mut store, &tokens)
            .expect("should succeed with truncation");

        let mem = memory.data(&store);
        let offset = SCRATCH_BUFFER_OFFSET as usize;
        let written = u32::from_le_bytes(mem[offset..offset + 4].try_into().expect("4 bytes"));
        assert_eq!(written, 0x42u32);
    }

    #[test]
    fn test_write_string_empty() {
        let (mut store, memory) = create_test_store_and_memory();
        let (ptr, len) = write_string(&memory, &mut store, "").expect("empty string should work");
        assert_eq!(ptr, SCRATCH_BUFFER_OFFSET);
        assert_eq!(len, 0);
    }

    #[test]
    fn test_write_string_hello() {
        let (mut store, memory) = create_test_store_and_memory();
        let (ptr, len) =
            write_string(&memory, &mut store, "hello").expect("simple string should work");
        assert_eq!(ptr, SCRATCH_BUFFER_OFFSET);
        assert_eq!(len, 5);

        let mem = memory.data(&store);
        let offset = SCRATCH_BUFFER_OFFSET as usize;
        assert_eq!(&mem[offset..offset + 5], b"hello");
    }

    #[test]
    fn test_write_string_unicode() {
        let (mut store, memory) = create_test_store_and_memory();
        let s = "日本語テスト";
        let (ptr, len) = write_string(&memory, &mut store, s).expect("unicode string should work");
        assert_eq!(ptr, SCRATCH_BUFFER_OFFSET);
        assert_eq!(len, s.len() as u32);

        let mem = memory.data(&store);
        let offset = SCRATCH_BUFFER_OFFSET as usize;
        assert_eq!(&mem[offset..offset + s.len()], s.as_bytes());
    }

    #[test]
    fn test_write_string_too_large() {
        let (mut store, memory) = create_test_store_and_memory();
        let large_string = "X".repeat(SCRATCH_BUFFER_SIZE as usize + 1);
        let result = write_string(&memory, &mut store, &large_string);
        assert!(result.is_err());
        match result {
            Err(msg) => assert!(msg.contains("exceeds scratch buffer")),
            Ok(_) => panic!("should have failed"),
        }
    }

    #[test]
    fn test_read_cstring_basic() {
        let (mut store, memory) = create_test_store_and_memory();
        // Write "test\0" at offset 0
        let msg = b"test\0";
        let mem = memory.data_mut(&mut store);
        mem[0..5].copy_from_slice(msg);

        let result = read_cstring(&memory, &store, 0, 256).expect("should read cstring");
        assert_eq!(result, "test");
    }

    #[test]
    fn test_read_cstring_at_name_offset() {
        let (mut store, memory) = create_test_store_and_memory();
        let name = b"my_validator\0";
        let offset = VALIDATOR_NAME_OFFSET as usize;
        let mem = memory.data_mut(&mut store);
        mem[offset..offset + name.len()].copy_from_slice(name);

        let result =
            read_cstring(&memory, &store, VALIDATOR_NAME_OFFSET, 256).expect("should read name");
        assert_eq!(result, "my_validator");
    }

    #[test]
    fn test_read_cstring_empty() {
        let (mut store, memory) = create_test_store_and_memory();
        let mem = memory.data_mut(&mut store);
        mem[0] = 0; // null byte immediately

        let result = read_cstring(&memory, &store, 0, 256).expect("should read empty string");
        assert_eq!(result, "");
    }

    #[test]
    fn test_read_cstring_out_of_bounds() {
        let (store, memory) = create_test_store_and_memory();
        let result = read_cstring(&memory, &store, 0xFFFFFFFF, 256);
        assert!(result.is_err());
        match result {
            Err(msg) => assert!(msg.contains("out of bounds")),
            Ok(_) => panic!("should have failed"),
        }
    }

    #[test]
    fn test_read_cstring_no_null_terminator() {
        let (mut store, memory) = create_test_store_and_memory();
        // Fill with non-null bytes
        let mem = memory.data_mut(&mut store);
        mem[0..10].fill(0x41); // 'A'

        let result = read_cstring(&memory, &store, 0, 10);
        assert!(result.is_err());
        match result {
            Err(msg) => assert!(msg.contains("no null terminator")),
            Ok(_) => panic!("should have failed"),
        }
    }

    #[test]
    fn test_read_cstring_null_within_max_len() {
        let (mut store, memory) = create_test_store_and_memory();
        let msg = b"hi\0world\0";
        let mem = memory.data_mut(&mut store);
        mem[0..msg.len()].copy_from_slice(msg);

        // Should stop at first null
        let result = read_cstring(&memory, &store, 0, 256).expect("should read until first null");
        assert_eq!(result, "hi");
    }

    #[test]
    fn test_read_cstring_max_len_limits_search() {
        let (mut store, memory) = create_test_store_and_memory();
        // Place null at position 5
        let msg = b"AAAAA\0";
        let mem = memory.data_mut(&mut store);
        mem[0..msg.len()].copy_from_slice(msg);

        // With max_len=3, we won't reach the null
        let result = read_cstring(&memory, &store, 0, 3);
        assert!(result.is_err());
    }
}
