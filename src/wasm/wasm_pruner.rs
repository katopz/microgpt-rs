//! WASM-based constraint pruner implementing [`ConstraintPruner`].
//!
//! Loads sandboxed WASM validator modules with fuel-based execution limits.
//! No WASI access — validators run in complete isolation.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
//! │ DDTree      │────▶│ WasmPruner   │────▶│ WASM Module     │
//! │ (speculative│     │ (Mutex-wrap) │     │ (sandboxed,     │
//! │  decoding)  │◀────│              │◀────│  fuel-limited)  │
//! └─────────────┘     └──────────────┘     └─────────────────┘
//!       is_valid()       FFI boundary          is_valid()
//! ```
//!
//! # Safety
//!
//! WASM modules run in a sandboxed environment with no access to:
//! - Filesystem
//! - Network
//! - Environment variables
//! - System time
//!
//! Each call is fuel-limited to [`FUEL_PER_CALL`] to prevent infinite loops.

use std::sync::Mutex;

use wasmtime::{Config, Engine, Linker, Memory, Module, Store, TypedFunc};

use crate::speculative::types::ConstraintPruner;

use super::abi;
use super::state::ValidatorState;

// ── Inner State (requires mutable access) ────────────────────────

/// Mutable WASM components wrapped behind a [`Mutex`].
///
/// All wasmtime operations require `&mut Store`, so we wrap everything
/// that needs mutation in a single lock. The lock is uncontended in
/// practice (single-threaded DDTree building per pruner instance).
struct WasmInner {
    store: Store<ValidatorState>,
    is_valid_fn: TypedFunc<(u32, u32, u32, u32), u32>,
    validate_string_fn: Option<TypedFunc<(u32, u32), u32>>,
    memory: Memory,
}

impl WasmInner {
    /// Call `is_valid` in the WASM module. Owns `&mut self` so field borrows don't conflict.
    fn call_is_valid(&mut self, depth: usize, token_idx: usize, parent_tokens: &[usize]) -> bool {
        // Set fuel budget for this call
        if self.store.set_fuel(abi::FUEL_PER_CALL).is_err() {
            return false;
        }

        // Write parent tokens to WASM linear memory
        let (ptr, len) =
            match abi::write_parent_tokens(&self.memory, &mut self.store, parent_tokens) {
                Ok(result) => result,
                Err(_) => return false,
            };

        // Call is_valid(depth, token_idx, ptr, len)
        match self
            .is_valid_fn
            .call(&mut self.store, (depth as u32, token_idx as u32, ptr, len))
        {
            Ok(result) => result == abi::VALID,
            Err(_) => false,
        }
    }

    /// Call `validate_string` in the WASM module. Owns `&mut self` so field borrows don't conflict.
    fn call_validate_string(&mut self, code: &str) -> bool {
        let Some(validate_fn) = self.validate_string_fn.as_ref() else {
            return false;
        };

        if self.store.set_fuel(abi::FUEL_PER_CALL).is_err() {
            return false;
        }

        let (ptr, len) = match abi::write_string(&self.memory, &mut self.store, code) {
            Ok(result) => result,
            Err(_) => return false,
        };

        match validate_fn.call(&mut self.store, (ptr, len)) {
            Ok(result) => result == abi::VALID,
            Err(_) => false,
        }
    }
}

// ── WasmPruner ───────────────────────────────────────────────────

/// WASM-based constraint pruner implementing [`ConstraintPruner`].
///
/// Loads a sandboxed WASM module and delegates token validation to it.
/// The WASM module has no access to filesystem, network, or environment
/// (no WASI), and is fuel-limited to prevent infinite loops.
///
/// # ABI Contract
///
/// The WASM module must export:
/// - `memory`: Linear memory (at least 1 page)
/// - `is_valid(depth, token_idx, parent_ptr, parent_len) -> i32`: Required. Returns 1 for valid, 0 for invalid
/// - `name() -> i32`: Required. Returns pointer to null-terminated name string
/// - `version() -> i32`: Required. Returns packed version (major\<\<16 | minor\<\<8 | patch)
///
/// Optional exports:
/// - `validate_string(ptr, len) -> i32`: Returns 1 for valid, 0 for invalid
///
/// # Thread Safety
///
/// [`WasmPruner`] implements [`Send`] + [`Sync`] via internal [`Mutex`].
/// The mutex is uncontended in typical usage (single-threaded DDTree building).
pub struct WasmPruner {
    inner: Mutex<WasmInner>,
}

impl WasmPruner {
    /// Load a WASM validator from raw bytes.
    ///
    /// Creates a sandboxed wasmtime instance with fuel consumption enabled.
    /// Extracts required exports (`is_valid`, `memory`, `name`, `version`)
    /// and optional export (`validate_string`).
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - WASM bytes fail to compile
    /// - Required exports are missing
    /// - Export `name()` or `version()` call fails
    pub fn load(wasm_bytes: &[u8]) -> Result<Self, String> {
        // 1. Create engine with fuel enabled
        let mut config = Config::new();
        config.consume_fuel(true);
        let engine =
            Engine::new(&config).map_err(|e| format!("failed to create wasmtime engine: {e}"))?;

        // 2. Load module from bytes
        let module = Module::new(&engine, wasm_bytes)
            .map_err(|e| format!("failed to compile WASM module: {e}"))?;

        // 3. Create linker (no WASI — fully sandboxed)
        let linker = Linker::new(&engine);

        // 4. Create store with placeholder state
        let mut store = Store::new(&engine, ValidatorState::placeholder());

        // 5. Instantiate module
        let instance = linker
            .instantiate(&mut store, &module)
            .map_err(|e| format!("failed to instantiate WASM module: {e}"))?;

        // 6. Extract memory export (required)
        let memory = instance
            .get_memory(&mut store, abi::EXPORT_MEMORY)
            .ok_or_else(|| format!("missing required export: '{}'", abi::EXPORT_MEMORY))?;

        // 7. Extract is_valid export (required)
        let is_valid_fn: TypedFunc<(u32, u32, u32, u32), u32> = instance
            .get_typed_func(&mut store, abi::EXPORT_IS_VALID)
            .map_err(|e| format!("missing required export '{}': {e}", abi::EXPORT_IS_VALID))?;

        // 8. Extract validate_string export (optional)
        let validate_string_fn = instance
            .get_typed_func::<(u32, u32), u32>(&mut store, abi::EXPORT_VALIDATE_STRING)
            .ok();

        // 9. Extract name and version function exports
        let name_fn: TypedFunc<(), u32> = instance
            .get_typed_func(&mut store, abi::EXPORT_NAME)
            .map_err(|e| format!("missing required export '{}': {e}", abi::EXPORT_NAME))?;

        let version_fn: TypedFunc<(), u32> = instance
            .get_typed_func(&mut store, abi::EXPORT_VERSION)
            .map_err(|e| format!("missing required export '{}': {e}", abi::EXPORT_VERSION))?;

        // 10. Call name() to get validator name pointer
        store
            .set_fuel(abi::FUEL_PER_CALL)
            .map_err(|e| format!("failed to set fuel for name(): {e}"))?;
        let name_ptr = name_fn
            .call(&mut store, ())
            .map_err(|e| format!("failed to call name(): {e}"))?;
        let name = abi::read_cstring(&memory, &store, name_ptr, 256)
            .map_err(|e| format!("failed to read validator name: {e}"))?;

        // 11. Call version() to get packed version
        store
            .set_fuel(abi::FUEL_PER_CALL)
            .map_err(|e| format!("failed to set fuel for version(): {e}"))?;
        let packed = version_fn
            .call(&mut store, ())
            .map_err(|e| format!("failed to call version(): {e}"))?;
        let version = (
            ((packed >> 16) & 0xFF) as u8,
            ((packed >> 8) & 0xFF) as u8,
            (packed & 0xFF) as u8,
        );

        // 12. Update store state with extracted metadata
        *store.data_mut() = ValidatorState::new(name.clone(), version);

        Ok(Self {
            inner: Mutex::new(WasmInner {
                store,
                is_valid_fn,
                validate_string_fn,
                memory,
            }),
        })
    }

    /// Load a WASM validator from a file path.
    pub fn load_from_file(path: &str) -> Result<Self, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("failed to read '{path}': {e}"))?;
        Self::load(&bytes)
    }

    /// Get the validator name (extracted from WASM `name()` export).
    pub fn name(&self) -> String {
        let inner = self.inner.lock().expect("WasmPruner mutex poisoned");
        inner.store.data().name.clone()
    }

    /// Get the validator version tuple.
    pub fn version(&self) -> (u8, u8, u8) {
        let inner = self.inner.lock().expect("WasmPruner mutex poisoned");
        inner.store.data().version
    }

    /// Validate a string via the WASM validator's `validate_string` export.
    ///
    /// Returns `false` if:
    /// - The export is not available
    /// - Fuel cannot be added
    /// - The string cannot be written to WASM memory
    /// - The WASM function traps
    /// - The function returns 0 (invalid)
    pub fn validate_string(&self, code: &str) -> bool {
        let mut inner = match self.inner.lock() {
            Ok(guard) => guard,
            Err(_) => return false,
        };
        inner.call_validate_string(code)
    }
}

// ── ConstraintPruner Implementation ──────────────────────────────

impl ConstraintPruner for WasmPruner {
    fn is_valid(&self, depth: usize, token_idx: usize, parent_tokens: &[usize]) -> bool {
        let mut inner = match self.inner.lock() {
            Ok(guard) => guard,
            Err(_) => return false,
        };
        inner.call_is_valid(depth, token_idx, parent_tokens)
    }
}

// ── Compile-Time Assertions ──────────────────────────────────────

const _: () = {
    // WasmPruner must be Send + Sync (required by ConstraintPruner)
    const fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<WasmPruner>();
};

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Test WAT Modules ───────────────────────────────────────

    /// Accept-all validator: every token and string is valid.
    fn accept_all_wat() -> &'static str {
        r#"
        (module
          (memory (export "memory") 1)
          (data (i32.const 0) "accept_all\00")
          (func (export "is_valid") (param i32 i32 i32 i32) (result i32)
            i32.const 1)
          (func (export "validate_string") (param i32 i32) (result i32)
            i32.const 1)
          (func (export "name") (result i32)
            i32.const 0)
          (func (export "version") (result i32)
            i32.const 0x000100))
        "#
    }

    /// Reject-all validator: every token and string is invalid.
    fn reject_all_wat() -> &'static str {
        r#"
        (module
          (memory (export "memory") 1)
          (data (i32.const 0) "reject_all\00")
          (func (export "is_valid") (param i32 i32 i32 i32) (result i32)
            i32.const 0)
          (func (export "validate_string") (param i32 i32) (result i32)
            i32.const 0)
          (func (export "name") (result i32)
            i32.const 0)
          (func (export "version") (result i32)
            i32.const 0x000100))
        "#
    }

    /// Reject-token-zero validator: accepts tokens where token_idx > 0.
    fn reject_zero_wat() -> &'static str {
        r#"
        (module
          (memory (export "memory") 1)
          (data (i32.const 0) "reject_zero\00")
          (func (export "is_valid") (param i32 i32 i32 i32) (result i32)
            local.get 1
            i32.const 0
            i32.gt_u)
          (func (export "validate_string") (param i32 i32) (result i32)
            i32.const 1)
          (func (export "name") (result i32)
            i32.const 0)
          (func (export "version") (result i32)
            i32.const 0x000100))
        "#
    }

    /// Module without validate_string export.
    fn no_validate_string_wat() -> &'static str {
        r#"
        (module
          (memory (export "memory") 1)
          (data (i32.const 0) "no_vs\00")
          (func (export "is_valid") (param i32 i32 i32 i32) (result i32)
            i32.const 1)
          (func (export "name") (result i32)
            i32.const 0)
          (func (export "version") (result i32)
            i32.const 0x000100))
        "#
    }

    /// Module without memory export.
    fn no_memory_wat() -> &'static str {
        r#"
        (module
          (func (export "is_valid") (param i32 i32 i32 i32) (result i32)
            i32.const 1)
          (func (export "name") (result i32)
            i32.const 0)
          (func (export "version") (result i32)
            i32.const 1))
        "#
    }

    /// Module without is_valid export.
    fn no_is_valid_wat() -> &'static str {
        r#"
        (module
          (memory (export "memory") 1)
          (func (export "name") (result i32)
            i32.const 0)
          (func (export "version") (result i32)
            i32.const 1))
        "#
    }

    /// Module without name export.
    fn no_name_wat() -> &'static str {
        r#"
        (module
          (memory (export "memory") 1)
          (func (export "is_valid") (param i32 i32 i32 i32) (result i32)
            i32.const 1)
          (func (export "version") (result i32)
            i32.const 1))
        "#
    }

    /// Module without version export.
    fn no_version_wat() -> &'static str {
        r#"
        (module
          (memory (export "memory") 1)
          (func (export "is_valid") (param i32 i32 i32 i32) (result i32)
            i32.const 1)
          (func (export "name") (result i32)
            i32.const 0))
        "#
    }

    /// Parse WAT string to WASM binary bytes.
    fn parse_wat(wat: &str) -> Vec<u8> {
        wat::parse_str(wat).expect("test WAT should be valid")
    }

    // ── Load Tests ─────────────────────────────────────────────

    #[test]
    fn test_load_accept_all() {
        let pruner =
            WasmPruner::load(&parse_wat(accept_all_wat())).expect("accept_all should load");
        assert_eq!(pruner.name(), "accept_all");
        assert_eq!(pruner.version(), (0, 1, 0));
    }

    #[test]
    fn test_load_reject_all() {
        let pruner =
            WasmPruner::load(&parse_wat(reject_all_wat())).expect("reject_all should load");
        assert_eq!(pruner.name(), "reject_all");
        assert_eq!(pruner.version(), (0, 1, 0));
    }

    #[test]
    fn test_load_reject_zero() {
        let pruner =
            WasmPruner::load(&parse_wat(reject_zero_wat())).expect("reject_zero should load");
        assert_eq!(pruner.name(), "reject_zero");
    }

    #[test]
    fn test_load_from_file() {
        let bytes = parse_wat(accept_all_wat());
        let dir = std::env::temp_dir().join("wasm_pruner_test_load.wasm");
        std::fs::write(&dir, &bytes).expect("should write temp file");

        let pruner = WasmPruner::load_from_file(dir.to_str().expect("path should be valid utf-8"))
            .expect("should load from file");
        assert_eq!(pruner.name(), "accept_all");

        let _ = std::fs::remove_file(&dir);
    }

    #[test]
    fn test_load_from_file_not_found() {
        let result = WasmPruner::load_from_file("/nonexistent/path/test.wasm");
        assert!(result.is_err());
        match result {
            Err(msg) => assert!(msg.contains("failed to read")),
            Ok(_) => panic!("should have failed for missing file"),
        }
    }

    #[test]
    fn test_load_invalid_wasm_bytes() {
        let result = WasmPruner::load(&[0xFF, 0xFE, 0xFD, 0xFC]);
        assert!(result.is_err());
        match result {
            Err(msg) => assert!(msg.contains("failed to compile")),
            Ok(_) => panic!("should have failed for invalid bytes"),
        }
    }

    #[test]
    fn test_load_missing_memory_export() {
        let result = WasmPruner::load(&parse_wat(no_memory_wat()));
        assert!(result.is_err());
        match result {
            Err(msg) => assert!(msg.contains("missing required export: 'memory'")),
            Ok(_) => panic!("should have failed without memory export"),
        }
    }

    #[test]
    fn test_load_missing_is_valid_export() {
        let result = WasmPruner::load(&parse_wat(no_is_valid_wat()));
        assert!(result.is_err());
        match result {
            Err(msg) => assert!(msg.contains("is_valid")),
            Ok(_) => panic!("should have failed without is_valid export"),
        }
    }

    #[test]
    fn test_load_missing_name_export() {
        let result = WasmPruner::load(&parse_wat(no_name_wat()));
        assert!(result.is_err());
        match result {
            Err(msg) => assert!(msg.contains("name")),
            Ok(_) => panic!("should have failed without name export"),
        }
    }

    #[test]
    fn test_load_missing_version_export() {
        let result = WasmPruner::load(&parse_wat(no_version_wat()));
        assert!(result.is_err());
        match result {
            Err(msg) => assert!(msg.contains("version")),
            Ok(_) => panic!("should have failed without version export"),
        }
    }

    // ── is_valid Tests ─────────────────────────────────────────

    #[test]
    fn test_is_valid_accept_all_always_true() {
        let pruner = WasmPruner::load(&parse_wat(accept_all_wat())).expect("should load");

        assert!(pruner.is_valid(0, 0, &[]));
        assert!(pruner.is_valid(0, 42, &[]));
        assert!(pruner.is_valid(1, 100, &[50]));
        assert!(pruner.is_valid(5, 999, &[1, 2, 3, 4, 5]));
    }

    #[test]
    fn test_is_valid_reject_all_always_false() {
        let pruner = WasmPruner::load(&parse_wat(reject_all_wat())).expect("should load");

        assert!(!pruner.is_valid(0, 0, &[]));
        assert!(!pruner.is_valid(0, 42, &[]));
        assert!(!pruner.is_valid(1, 100, &[50]));
    }

    #[test]
    fn test_is_valid_reject_zero_conditional() {
        let pruner = WasmPruner::load(&parse_wat(reject_zero_wat())).expect("should load");

        // token_idx == 0 should be rejected
        assert!(!pruner.is_valid(0, 0, &[]));

        // token_idx > 0 should be accepted
        assert!(pruner.is_valid(0, 1, &[]));
        assert!(pruner.is_valid(0, 42, &[]));
        assert!(pruner.is_valid(5, 100, &[1, 2, 3, 4, 5]));
    }

    #[test]
    fn test_is_valid_with_parent_tokens() {
        let pruner = WasmPruner::load(&parse_wat(accept_all_wat())).expect("should load");

        let parents: Vec<usize> = (0..100).collect();
        assert!(pruner.is_valid(100, 42, &parents));
    }

    #[test]
    fn test_is_valid_empty_parent_tokens() {
        let pruner = WasmPruner::load(&parse_wat(accept_all_wat())).expect("should load");

        assert!(pruner.is_valid(0, 0, &[]));
    }

    // ── validate_string Tests ──────────────────────────────────

    #[test]
    fn test_validate_string_accept_all() {
        let pruner = WasmPruner::load(&parse_wat(accept_all_wat())).expect("should load");

        assert!(pruner.validate_string("hello world"));
        assert!(pruner.validate_string(""));
        assert!(pruner.validate_string("fn main() {}"));
    }

    #[test]
    fn test_validate_string_reject_all() {
        let pruner = WasmPruner::load(&parse_wat(reject_all_wat())).expect("should load");

        assert!(!pruner.validate_string("hello"));
        assert!(!pruner.validate_string(""));
    }

    #[test]
    fn test_validate_string_missing_export() {
        let pruner = WasmPruner::load(&parse_wat(no_validate_string_wat())).expect("should load");

        // Should return false when export is missing
        assert!(!pruner.validate_string("anything"));
    }

    // ── Trait Object Tests ─────────────────────────────────────

    #[test]
    fn test_as_trait_object() {
        let pruner: Box<dyn ConstraintPruner> =
            Box::new(WasmPruner::load(&parse_wat(accept_all_wat())).expect("should load"));

        assert!(pruner.is_valid(0, 42, &[]));
        assert!(pruner.is_valid(5, 100, &[1, 2, 3, 4, 5]));
    }

    #[test]
    fn test_multiple_pruners() {
        let accept = WasmPruner::load(&parse_wat(accept_all_wat())).expect("should load");
        let reject = WasmPruner::load(&parse_wat(reject_all_wat())).expect("should load");
        let conditional = WasmPruner::load(&parse_wat(reject_zero_wat())).expect("should load");

        assert!(accept.is_valid(0, 0, &[]));
        assert!(!reject.is_valid(0, 0, &[]));
        assert!(!conditional.is_valid(0, 0, &[]));
        assert!(conditional.is_valid(0, 1, &[]));
    }

    // ── Version Parsing Tests ──────────────────────────────────

    #[test]
    fn test_version_parsing_0_1_0() {
        let pruner = WasmPruner::load(&parse_wat(accept_all_wat())).expect("should load");
        assert_eq!(pruner.version(), (0, 1, 0));
    }

    #[test]
    fn test_version_packing() {
        // version 0x010203 = (1, 2, 3)
        let wat = r#"
        (module
          (memory (export "memory") 1)
          (data (i32.const 0) "versioned\00")
          (func (export "is_valid") (param i32 i32 i32 i32) (result i32)
            i32.const 1)
          (func (export "name") (result i32)
            i32.const 0)
          (func (export "version") (result i32)
            i32.const 0x010203))
        "#;
        let pruner = WasmPruner::load(&parse_wat(wat)).expect("should load");
        assert_eq!(pruner.version(), (1, 2, 3));
    }

    // ── Edge Case Tests ────────────────────────────────────────

    #[test]
    fn test_name_with_unicode() {
        let wat = r#"
        (module
          (memory (export "memory") 1)
          (data (i32.const 0) "\e6\97\a5\e6\9c\ac\e8\aa\9e\00")
          (func (export "is_valid") (param i32 i32 i32 i32) (result i32)
            i32.const 1)
          (func (export "name") (result i32)
            i32.const 0)
          (func (export "version") (result i32)
            i32.const 0x000100))
        "#;
        let pruner = WasmPruner::load(&parse_wat(wat)).expect("should load");
        assert_eq!(pruner.name(), "日本語");
    }

    #[test]
    fn test_deep_depth_with_many_parents() {
        let pruner = WasmPruner::load(&parse_wat(accept_all_wat())).expect("should load");

        // Build a long parent path
        let parents: Vec<usize> = (0..500).map(|i| i * 2).collect();
        assert!(pruner.is_valid(500, 1000, &parents));
    }

    #[test]
    fn test_repeated_calls_same_pruner() {
        let pruner = WasmPruner::load(&parse_wat(accept_all_wat())).expect("should load");

        // Should handle many calls without state corruption
        for i in 0..50 {
            assert!(pruner.is_valid(i, i + 1, &[]), "failed at iteration {i}");
        }
    }
}
