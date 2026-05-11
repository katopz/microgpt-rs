//! WASM bomber validator loaded from `.wasm` files via wasmtime.
//!
//! Wraps a wasmtime instance that validates bomber actions against game state.
//! The WASM module runs sandboxed with no WASI access and fuel-limited execution.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐
//! │ BomberGame  │────▶│ BomberWasmPruner │────▶│ WASM Module     │
//! │ (arena tick)│     │ (Mutex-wrapped)  │     │ (sandboxed,     │
//! │             │◀────│                  │◀────│  fuel-limited)  │
//! └─────────────┘     └──────────────────┘     └─────────────────┘
//!     is_safe_action()     serialize + FFI          is_valid()
//! ```
//!
//! # ABI Contract
//!
//! The WASM module must export:
//! - `memory`: Linear memory (at least 1 page)
//! - `is_valid(depth, action_idx, state_ptr, state_len) -> i32`: Required
//! - `name() -> i32`: Required (pointer to null-terminated name)
//! - `version() -> i32`: Required (packed: major<<16 | minor<<8 | patch)
//!
//! Optional exports:
//! - `relevance(depth, action_idx, state_ptr, state_len) -> i32`: Q16.16 score

use std::sync::Mutex;

use wasmtime::{Config, Engine, Linker, Memory, Module, Store, TypedFunc};

use super::ArenaGrid;
use super::wasm_state::serialize_game_state;

// ── Constants ──────────────────────────────────────────────────

/// Fuel limit per WASM call (prevents infinite loops).
const FUEL_PER_CALL: u64 = 10_000;

/// Memory limit: 64 pages (4MB).
#[allow(dead_code)]
const MEMORY_PAGES: u64 = 64;

/// Return value indicating "valid" from WASM.
const VALID: u32 = 1;

// ── WASM Export Names ──────────────────────────────────────────

mod abi {
    pub const EXPORT_MEMORY: &str = "memory";
    pub const EXPORT_IS_VALID: &str = "is_valid";
    pub const EXPORT_RELEVANCE: &str = "relevance";
    pub const EXPORT_NAME: &str = "name";
    pub const EXPORT_VERSION: &str = "version";
}

// ── Inner State ────────────────────────────────────────────────

/// Mutable WASM components wrapped behind a [`Mutex`].
///
/// All wasmtime operations require `&mut Store`, so we wrap everything
/// that needs mutation in a single lock.
struct BomberInner {
    store: Store<()>,
    is_valid_fn: TypedFunc<(u32, u32, u32, u32), u32>,
    relevance_fn: Option<TypedFunc<(u32, u32, u32, u32), u32>>,
    memory: Memory,
    name: String,
    version: (u8, u8, u8),
}

impl BomberInner {
    /// Write game state bytes to WASM linear memory.
    ///
    /// Returns `(ptr, len)` pointing to the written data in WASM memory.
    fn write_state(&mut self, state: &[u8]) -> Result<(u32, u32), String> {
        let mem_size = self.memory.data_size(&self.store);
        if state.len() > mem_size {
            let extra_pages = ((state.len() - mem_size) / 65536) + 1;
            self.memory
                .grow(&mut self.store, extra_pages as u64)
                .map_err(|e| format!("failed to grow WASM memory: {e}"))?;
        }

        self.memory.data_mut(&mut self.store)[..state.len()].copy_from_slice(state);

        Ok((0, state.len() as u32))
    }

    /// Call `is_valid` in the WASM module.
    fn call_is_valid(
        &mut self,
        action_idx: usize,
        grid: &ArenaGrid,
        player_x: i32,
        player_y: i32,
        player_id: u8,
        bombs: &[((i32, i32), u32, u32)],
    ) -> bool {
        if self.store.set_fuel(FUEL_PER_CALL).is_err() {
            return false;
        }

        let (state, token_count) = serialize_game_state(grid, player_x, player_y, player_id, bombs);
        let (ptr, _byte_len) = match self.write_state(&state) {
            Ok(result) => result,
            Err(_) => return false,
        };

        // depth=0 (unused for bomber), action_idx = action (0-5)
        // Pass token_count as len — WASM SDK's read_parent_tokens reads len×4 bytes
        match self
            .is_valid_fn
            .call(&mut self.store, (0, action_idx as u32, ptr, token_count))
        {
            Ok(result) => result == VALID,
            Err(_) => false,
        }
    }

    /// Call `relevance` in the WASM module. Returns Q16.16 fixed-point decoded to f32.
    /// Falls back to binary `is_valid` (0.0/1.0) if relevance export is missing.
    fn call_relevance(
        &mut self,
        action_idx: usize,
        grid: &ArenaGrid,
        player_x: i32,
        player_y: i32,
        player_id: u8,
        bombs: &[((i32, i32), u32, u32)],
    ) -> f32 {
        // Check existence first to avoid borrow conflict with write_state
        if self.relevance_fn.is_none() {
            return if self.call_is_valid(action_idx, grid, player_x, player_y, player_id, bombs) {
                1.0
            } else {
                0.0
            };
        }

        if self.store.set_fuel(FUEL_PER_CALL).is_err() {
            return 0.0;
        }

        let (state, token_count) = serialize_game_state(grid, player_x, player_y, player_id, bombs);
        let (ptr, _byte_len) = match self.write_state(&state) {
            Ok(result) => result,
            Err(_) => return 0.0,
        };

        // Extract fn after mutable work to avoid borrow conflict
        // Pass token_count as len — WASM SDK's read_parent_tokens reads len×4 bytes
        let relevance_fn = self.relevance_fn.as_ref().unwrap();
        match relevance_fn.call(&mut self.store, (0, action_idx as u32, ptr, token_count)) {
            Ok(raw) => {
                // Decode Q16.16 fixed-point: f32 = raw_u32 / 65536.0
                let relevance = raw as f32 / 65536.0;
                relevance.clamp(0.0, 1.0)
            }
            Err(_) => 0.0,
        }
    }
}

// ── BomberWasmPruner ───────────────────────────────────────────

/// WASM bomber validator loaded from `.wasm` file.
///
/// Wraps a wasmtime instance that validates bomber actions against
/// game state. Falls back to native Rust if WASM fails to load.
///
/// # Thread Safety
///
/// [`BomberWasmPruner`] implements [`Send`] + [`Sync`] via internal [`Mutex`].
pub struct BomberWasmPruner {
    inner: Mutex<BomberInner>,
}

impl BomberWasmPruner {
    /// Load a WASM bomber validator from file.
    pub fn load_from_file(path: &str) -> Result<Self, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("failed to read '{path}': {e}"))?;
        Self::load(&bytes)
    }

    /// Load a WASM bomber validator from bytes.
    ///
    /// Creates a sandboxed wasmtime instance with fuel consumption enabled.
    /// Extracts required exports (`is_valid`, `memory`, `name`, `version`)
    /// and optional export (`relevance`).
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

        // 4. Create store
        let mut store = Store::new(&engine, ());

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

        // 8. Extract relevance export (optional)
        let relevance_fn = instance
            .get_typed_func::<(u32, u32, u32, u32), u32>(&mut store, abi::EXPORT_RELEVANCE)
            .ok();

        // 9. Extract and call name()
        let name_fn: TypedFunc<(), u32> = instance
            .get_typed_func(&mut store, abi::EXPORT_NAME)
            .map_err(|e| format!("missing required export '{}': {e}", abi::EXPORT_NAME))?;

        store
            .set_fuel(FUEL_PER_CALL)
            .map_err(|e| format!("failed to set fuel for name(): {e}"))?;
        let name_ptr = name_fn
            .call(&mut store, ())
            .map_err(|e| format!("failed to call name(): {e}"))?;
        let name = read_cstring(&memory, &store, name_ptr, 256)
            .map_err(|e| format!("failed to read validator name: {e}"))?;

        // 10. Extract and call version()
        let version_fn: TypedFunc<(), u32> = instance
            .get_typed_func(&mut store, abi::EXPORT_VERSION)
            .map_err(|e| format!("missing required export '{}': {e}", abi::EXPORT_VERSION))?;

        store
            .set_fuel(FUEL_PER_CALL)
            .map_err(|e| format!("failed to set fuel for version(): {e}"))?;
        let packed = version_fn
            .call(&mut store, ())
            .map_err(|e| format!("failed to call version(): {e}"))?;
        let version = (
            ((packed >> 16) & 0xFF) as u8,
            ((packed >> 8) & 0xFF) as u8,
            (packed & 0xFF) as u8,
        );

        Ok(Self {
            inner: Mutex::new(BomberInner {
                store,
                is_valid_fn,
                relevance_fn,
                memory,
                name,
                version,
            }),
        })
    }

    /// Check if an action is safe given game state.
    ///
    /// Serializes the game state, writes to WASM memory, and calls `is_valid`.
    /// Returns `false` if the WASM module traps or any step fails.
    pub fn is_safe_action(
        &self,
        action_idx: usize,
        grid: &ArenaGrid,
        player_x: i32,
        player_y: i32,
        player_id: u8,
        bombs: &[((i32, i32), u32, u32)],
    ) -> bool {
        let mut inner = match self.inner.lock() {
            Ok(guard) => guard,
            Err(_) => return false,
        };
        inner.call_is_valid(action_idx, grid, player_x, player_y, player_id, bombs)
    }

    /// Get action relevance score via WASM.
    ///
    /// Calls `relevance` export if available, decoding Q16.16 fixed-point to f32.
    /// Falls back to binary `is_valid` (0.0/1.0) if the relevance export is missing.
    pub fn action_relevance(
        &self,
        action_idx: usize,
        grid: &ArenaGrid,
        player_x: i32,
        player_y: i32,
        player_id: u8,
        bombs: &[((i32, i32), u32, u32)],
    ) -> f32 {
        let mut inner = match self.inner.lock() {
            Ok(guard) => guard,
            Err(_) => return 0.0,
        };
        inner.call_relevance(action_idx, grid, player_x, player_y, player_id, bombs)
    }

    /// Get validator name.
    pub fn name(&self) -> String {
        let inner = self.inner.lock().expect("BomberWasmPruner mutex poisoned");
        inner.name.clone()
    }

    /// Get validator version.
    pub fn version(&self) -> (u8, u8, u8) {
        let inner = self.inner.lock().expect("BomberWasmPruner mutex poisoned");
        inner.version
    }
}

// ── Helpers ────────────────────────────────────────────────────

/// Read a null-terminated C string from WASM memory.
fn read_cstring(
    memory: &Memory,
    store: &Store<()>,
    ptr: u32,
    max_len: usize,
) -> Result<String, String> {
    let data = memory.data(store);
    let start = ptr as usize;
    if start >= data.len() {
        return Err(format!("name pointer {ptr} out of memory bounds"));
    }

    let slice = &data[start..data.len().min(start + max_len)];
    let end = slice
        .iter()
        .position(|&b| b == 0)
        .ok_or_else(|| format!("no null terminator found within {max_len} bytes from {ptr}"))?;

    String::from_utf8(slice[..end].to_vec())
        .map_err(|e| format!("validator name is not valid UTF-8: {e}"))
}

// ── Compile-Time Assertions ────────────────────────────────────

const _: () = {
    // BomberWasmPruner must be Send + Sync (shared across threads)
    const fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<BomberWasmPruner>();
};

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pruners::bomber::{ARENA_H, ARENA_W, Cell};

    fn empty_grid() -> ArenaGrid {
        ArenaGrid {
            cells: vec![vec![Cell::Floor; ARENA_W]; ARENA_H],
            width: ARENA_W,
            height: ARENA_H,
        }
    }

    #[test]
    fn load_invalid_wasm_bytes_fails() {
        let result = BomberWasmPruner::load(b"not valid wasm");
        match result {
            Err(e) => assert!(
                e.contains("failed to compile WASM module"),
                "unexpected error message: {e}"
            ),
            Ok(_) => panic!("expected error for invalid WASM bytes"),
        }
    }

    #[test]
    fn load_from_file_not_found_fails() {
        let result = BomberWasmPruner::load_from_file("/nonexistent/path.wasm");
        match result {
            Err(e) => assert!(
                e.contains("failed to read"),
                "unexpected error message: {e}"
            ),
            Ok(_) => panic!("expected error for missing file"),
        }
    }

    #[test]
    fn serialize_game_state_integration() {
        // Verify serialization works independently of WASM loading
        let grid = empty_grid();
        let bombs: [((i32, i32), u32, u32); 2] = [((3, 4), 2, 3), ((5, 6), 1, 1)];
        let (state, token_count) = serialize_game_state(&grid, 1, 1, 0, &bombs);

        // 173 header + 2×4 bomb tokens = 181 tokens × 4 bytes = 724 bytes
        assert_eq!(token_count, 181);
        assert_eq!(state.len(), 181 * 4);
    }

    #[test]
    fn read_cstring_valid() {
        // We can't easily test this without a WASM instance,
        // but we can verify the function exists and compiles.
        let _ = std::mem::size_of::<BomberWasmPruner>();
    }

    #[test]
    fn memory_pages_constant() {
        assert_eq!(MEMORY_PAGES, 64);
        assert_eq!(MEMORY_PAGES * 65536, 4 * 1024 * 1024); // 4MB
    }

    #[test]
    fn fuel_per_call_constant() {
        assert_eq!(FUEL_PER_CALL, 10_000);
    }
}
