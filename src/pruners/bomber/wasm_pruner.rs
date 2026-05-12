//! WASM bomber validator with batch API, zero-copy, and papaya instance pool.
//!
//! Wraps a wasmtime instance that validates bomber actions against game state.
//! The WASM module runs sandboxed with no WASI access and fuel-limited execution.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐
//! │ BomberGame  │────▶│ BomberWasmPruner │────▶│ WASM Module     │
//! │ (arena tick)│     │ (papaya pool)    │     │ (sandboxed,     │
//! │             │◀────│                  │◀────│  fuel-limited)  │
//! └─────────────┘     └──────────────────┘     └─────────────────┘
//!     batch_validate()  serialize once   batch_is_valid() × N×M
//! ```
//!
//! # Improvements over v1 (Plan 034)
//!
//! - **Batch API**: Serialize grid once per tick, validate all player×action
//!   pairs in one FFI call (~14× faster per tick)
//! - **Zero-copy**: Write directly to WASM memory via reusable stack buffer
//!   (no Vec allocation per call)
//! - **Papaya pool**: Lock-free per-thread WASM instances via
//!   `papaya::HashMap<ThreadId, Mutex<BomberInner>>` (no global Mutex contention)
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
//! - `batch_is_valid(state_ptr, state_len, players_ptr, player_count,
//!    actions_ptr, action_count, results_ptr) -> i32`: Batch validation
//! - `batch_relevance(...)`: Batch relevance scoring

use std::sync::{Arc, Mutex};
use std::thread::ThreadId;

use papaya::HashMap;
use wasmtime::{Config, Engine, Linker, Memory, Module, Store, TypedFunc};

use super::ArenaGrid;
use super::wasm_state::ZeroCopyStateBuffer;

// ── Constants ──────────────────────────────────────────────────

/// Fuel limit per individual WASM call (prevents infinite loops).
///
/// Must be sufficient for BFS escape route analysis with up to 16 bombs
/// on a 13×13 grid. Each BFS step checks blast zones against all bombs ×
/// 4 directions × range. With 169 cells, 4 neighbours, and 17 total bombs
/// (16 existing + 1 new), the instruction count can reach ~40K WASM ops.
/// 50K provides headroom for worst-case scenarios.
const FUEL_PER_CALL: u64 = 50_000;

/// Fuel multiplier for batch calls.
/// Batch does N×M individual checks internally, each needing full fuel.
/// With 4 players × 6 actions = 24 checks, the WASM loop overhead is
/// modest since grid is parsed once. 2× is sufficient.
const FUEL_BATCH_MULTIPLIER: u64 = 2;

/// Return value indicating "valid" from WASM.
const VALID: u32 = 1;

/// Maximum players per batch call.
const MAX_PLAYERS: usize = 4;

/// Actions per player (Up, Down, Left, Right, Bomb, Wait).
const ACTION_COUNT: usize = 6;

/// Pre-computed actions array: [0,1,2,3,4,5] as u32 LE bytes.
const ACTIONS_BYTES: [u8; ACTION_COUNT * 4] = [
    0, 0, 0, 0, // Up
    1, 0, 0, 0, // Down
    2, 0, 0, 0, // Left
    3, 0, 0, 0, // Right
    4, 0, 0, 0, // Bomb
    5, 0, 0, 0, // Wait
];

// ── WASM Export Names ──────────────────────────────────────────

mod abi {
    pub const MEMORY: &str = "memory";
    pub const IS_VALID: &str = "is_valid";
    pub const RELEVANCE: &str = "relevance";
    pub const NAME: &str = "name";
    pub const VERSION: &str = "version";
    pub const BATCH_IS_VALID: &str = "batch_is_valid";
    pub const BATCH_RELEVANCE: &str = "batch_relevance";
}

// ── Helpers ────────────────────────────────────────────────────

/// Align `offset` up to the next 8-byte boundary.
#[inline]
const fn align8(offset: usize) -> usize {
    (offset + 7) & !7
}

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

// ── BomberInner (per-thread WASM instance) ─────────────────────

/// Mutable WASM components for a single thread.
///
/// Each thread gets its own `BomberInner` stored in the papaya pool,
/// so the wrapping `Mutex` is never contended. The `Store` requires
/// `&mut` for all operations, which is why we still need interior
/// mutability — but the lock is per-thread and uncontended.
#[expect(
    clippy::type_complexity,
    reason = "wasmtime TypedFunc API requires many params"
)]
struct BomberInner {
    store: Store<()>,
    is_valid_fn: TypedFunc<(u32, u32, u32, u32), u32>,
    relevance_fn: Option<TypedFunc<(u32, u32, u32, u32), u32>>,
    batch_is_valid_fn: Option<TypedFunc<(u32, u32, u32, u32, u32, u32, u32), u32>>,
    batch_relevance_fn: Option<TypedFunc<(u32, u32, u32, u32, u32, u32, u32), u32>>,
    memory: Memory,
    /// Reusable zero-copy buffer — avoids Vec allocation per call.
    state_buf: ZeroCopyStateBuffer,
}

impl BomberInner {
    /// Create a new WASM instance from shared engine and module.
    fn new(engine: &Engine, module: &Module) -> Result<Self, String> {
        let linker = Linker::new(engine);
        let mut store = Store::new(engine, ());

        let instance = linker
            .instantiate(&mut store, module)
            .map_err(|e| format!("failed to instantiate WASM module: {e}"))?;

        // Required exports
        let memory = instance
            .get_memory(&mut store, abi::MEMORY)
            .ok_or_else(|| format!("missing required export: '{}'", abi::MEMORY))?;

        let is_valid_fn: TypedFunc<(u32, u32, u32, u32), u32> = instance
            .get_typed_func(&mut store, abi::IS_VALID)
            .map_err(|e| format!("missing required export '{}': {e}", abi::IS_VALID))?;

        // Optional exports
        let relevance_fn = instance
            .get_typed_func::<(u32, u32, u32, u32), u32>(&mut store, abi::RELEVANCE)
            .ok();

        let batch_is_valid_fn = instance
            .get_typed_func::<(u32, u32, u32, u32, u32, u32, u32), u32>(
                &mut store,
                abi::BATCH_IS_VALID,
            )
            .ok();

        let batch_relevance_fn = instance
            .get_typed_func::<(u32, u32, u32, u32, u32, u32, u32), u32>(
                &mut store,
                abi::BATCH_RELEVANCE,
            )
            .ok();

        Ok(Self {
            store,
            is_valid_fn,
            relevance_fn,
            batch_is_valid_fn,
            batch_relevance_fn,
            memory,
            state_buf: ZeroCopyStateBuffer::new(),
        })
    }

    /// Whether this instance supports batch operations.
    fn has_batch(&self) -> bool {
        self.batch_is_valid_fn.is_some()
    }

    /// Write bytes to WASM linear memory at the given offset.
    fn write_memory(&mut self, offset: usize, data: &[u8]) -> Result<(), String> {
        let end = offset + data.len();
        let mem_size = self.memory.data_size(&self.store);
        if end > mem_size {
            let extra_pages = ((end - mem_size) / 65536) + 1;
            self.memory
                .grow(&mut self.store, extra_pages as u64)
                .map_err(|e| format!("failed to grow WASM memory: {e}"))?;
        }
        self.memory.data_mut(&mut self.store)[offset..end].copy_from_slice(data);
        Ok(())
    }

    /// Read u32 LE results from WASM memory.
    fn read_u32_results(&self, offset: usize, count: usize) -> Vec<u32> {
        let data = &self.memory.data(&self.store)[offset..offset + count * 4];
        data.chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    // ── Individual calls (zero-copy) ────────────────────────────

    /// Check if a single action is safe for one player.
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

        // Serialize into reusable stack buffer, then copy to release borrow
        let (bytes_written, token_count) = self
            .state_buf
            .serialize(grid, player_x, player_y, player_id, bombs);

        let mut tmp = [0u8; 1024];
        tmp[..bytes_written].copy_from_slice(self.state_buf.as_bytes(bytes_written));

        if self.write_memory(0, &tmp[..bytes_written]).is_err() {
            return false;
        }

        match self
            .is_valid_fn
            .call(&mut self.store, (0, action_idx as u32, 0, token_count))
        {
            Ok(result) => result == VALID,
            Err(_) => false,
        }
    }

    /// Compute relevance score for a single action (Q16.16 decoded to f32).
    fn call_relevance(
        &mut self,
        action_idx: usize,
        grid: &ArenaGrid,
        player_x: i32,
        player_y: i32,
        player_id: u8,
        bombs: &[((i32, i32), u32, u32)],
    ) -> f32 {
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

        let (bytes_written, token_count) = self
            .state_buf
            .serialize(grid, player_x, player_y, player_id, bombs);

        let mut tmp = [0u8; 1024];
        tmp[..bytes_written].copy_from_slice(self.state_buf.as_bytes(bytes_written));

        if self.write_memory(0, &tmp[..bytes_written]).is_err() {
            return 0.0;
        }

        let relevance_fn = self.relevance_fn.as_ref().unwrap();
        match relevance_fn.call(&mut self.store, (0, action_idx as u32, 0, token_count)) {
            Ok(raw) => (raw as f32 / 65536.0).clamp(0.0, 1.0),
            Err(_) => 0.0,
        }
    }

    // ── Batch calls ─────────────────────────────────────────────

    /// Batch validate all player×action pairs in one FFI call.
    ///
    /// Memory layout written to WASM:
    /// ```text
    /// [0..state_end)           grid + bombs (batch layout)
    /// [players_off..+N×12)     player array: N × (id, x, y) u32 LE
    /// [actions_off..+M×4)      action indices u32 LE
    /// [results_off..+N×M×4)    output: u32 LE results (0/1)
    /// ```
    fn call_batch_validate(
        &mut self,
        grid: &ArenaGrid,
        players: &[(u8, i32, i32)],
        bombs: &[((i32, i32), u32, u32)],
    ) -> Option<BatchResult> {
        // Clone TypedFunc to release borrow on self before mutable operations.
        // TypedFunc wraps a Func handle (cheap index copy).
        let batch_fn = self.batch_is_valid_fn.as_ref()?.clone();

        let n = players.len().min(MAX_PLAYERS);
        if n == 0 {
            return None;
        }
        let m = ACTION_COUNT;

        if self
            .store
            .set_fuel(FUEL_PER_CALL * FUEL_BATCH_MULTIPLIER)
            .is_err()
        {
            return None;
        }

        // 1. Serialize grid + bombs (batch layout: no player data)
        let (state_bytes, state_tokens) = self.state_buf.serialize_grid(grid, bombs);

        // Stack copy to release state_buf borrow before mutable self access
        let mut tmp = [0u8; 1024];
        tmp[..state_bytes].copy_from_slice(self.state_buf.as_bytes(state_bytes));

        // 2. Compute aligned offsets
        let players_off = align8(state_bytes);
        let players_bytes = n * 3 * 4; // N × (id, x, y)
        let actions_off = players_off + players_bytes;
        let actions_bytes = m * 4;
        let results_off = actions_off + actions_bytes;

        // 3. Write state
        self.write_memory(0, &tmp[..state_bytes]).ok()?;

        // 4. Write players array: N × (id: u32, x: u32, y: u32)
        let mut players_buf = [0u8; MAX_PLAYERS * 12];
        for (i, &(id, x, y)) in players.iter().take(n).enumerate() {
            let off = i * 12;
            players_buf[off..off + 4].copy_from_slice(&(id as u32).to_le_bytes());
            players_buf[off + 4..off + 8].copy_from_slice(&(x as u32).to_le_bytes());
            players_buf[off + 8..off + 12].copy_from_slice(&(y as u32).to_le_bytes());
        }
        self.write_memory(players_off, &players_buf[..n * 12])
            .ok()?;

        // 5. Write actions array (always [0,1,2,3,4,5])
        self.write_memory(actions_off, &ACTIONS_BYTES).ok()?;

        // 6. Call batch_is_valid
        let result = batch_fn.call(
            &mut self.store,
            (
                0,                  // state_ptr
                state_tokens,       // state_len
                players_off as u32, // players_ptr
                n as u32,           // player_count
                actions_off as u32, // actions_ptr
                m as u32,           // action_count
                results_off as u32, // results_ptr
            ),
        );

        if !matches!(result, Ok(v) if v == VALID) {
            return None;
        }

        // 7. Read results
        let raw = self.read_u32_results(results_off, n * m);

        Some(BatchResult {
            results: raw,
            player_count: n,
            action_count: m,
        })
    }

    /// Batch compute relevance scores for all player×action pairs.
    fn call_batch_relevance(
        &mut self,
        grid: &ArenaGrid,
        players: &[(u8, i32, i32)],
        bombs: &[((i32, i32), u32, u32)],
    ) -> Option<BatchRelevanceResult> {
        // Clone TypedFunc to release borrow on self before mutable operations.
        let batch_fn = self.batch_relevance_fn.as_ref()?.clone();

        let n = players.len().min(MAX_PLAYERS);
        if n == 0 {
            return None;
        }
        let m = ACTION_COUNT;

        if self
            .store
            .set_fuel(FUEL_PER_CALL * FUEL_BATCH_MULTIPLIER)
            .is_err()
        {
            return None;
        }

        let (state_bytes, state_tokens) = self.state_buf.serialize_grid(grid, bombs);

        let mut tmp = [0u8; 1024];
        tmp[..state_bytes].copy_from_slice(self.state_buf.as_bytes(state_bytes));

        let players_off = align8(state_bytes);
        let players_bytes = n * 3 * 4;
        let actions_off = players_off + players_bytes;
        let actions_bytes = m * 4;
        let results_off = actions_off + actions_bytes;

        self.write_memory(0, &tmp[..state_bytes]).ok()?;

        let mut players_buf = [0u8; MAX_PLAYERS * 12];
        for (i, &(id, x, y)) in players.iter().take(n).enumerate() {
            let off = i * 12;
            players_buf[off..off + 4].copy_from_slice(&(id as u32).to_le_bytes());
            players_buf[off + 4..off + 8].copy_from_slice(&(x as u32).to_le_bytes());
            players_buf[off + 8..off + 12].copy_from_slice(&(y as u32).to_le_bytes());
        }
        self.write_memory(players_off, &players_buf[..n * 12])
            .ok()?;

        self.write_memory(actions_off, &ACTIONS_BYTES).ok()?;

        let result = batch_fn.call(
            &mut self.store,
            (
                0,
                state_tokens,
                players_off as u32,
                n as u32,
                actions_off as u32,
                m as u32,
                results_off as u32,
            ),
        );

        if !matches!(result, Ok(v) if v == VALID) {
            return None;
        }

        let raw = self.read_u32_results(results_off, n * m);
        let scores: Vec<f32> = raw
            .iter()
            .map(|&v| (v as f32 / 65536.0).clamp(0.0, 1.0))
            .collect();

        Some(BatchRelevanceResult {
            scores,
            player_count: n,
            action_count: m,
        })
    }

    /// Fallback: emulate batch by calling individual `is_valid` N×M times.
    fn call_batch_fallback(
        &mut self,
        grid: &ArenaGrid,
        players: &[(u8, i32, i32)],
        bombs: &[((i32, i32), u32, u32)],
    ) -> BatchResult {
        let n = players.len().min(MAX_PLAYERS);
        let m = ACTION_COUNT;
        let mut results = Vec::with_capacity(n * m);

        for &(id, x, y) in players.iter().take(n) {
            for action_idx in 0..m {
                let valid = self.call_is_valid(action_idx, grid, x, y, id, bombs);
                results.push(if valid { 1 } else { 0 });
            }
        }

        BatchResult {
            results,
            player_count: n,
            action_count: m,
        }
    }
}

// ── Batch Result Types ─────────────────────────────────────────

/// Result of batch validation: `results[player_idx * action_count + action_idx]`.
///
/// Use [`is_valid`](BatchResult::is_valid) to check specific player×action pairs.
pub struct BatchResult {
    results: Vec<u32>,
    player_count: usize,
    action_count: usize,
}

impl BatchResult {
    /// Create an empty batch result (no players).
    pub fn empty() -> Self {
        Self {
            results: Vec::new(),
            player_count: 0,
            action_count: ACTION_COUNT,
        }
    }

    /// Number of players in this batch.
    pub fn player_count(&self) -> usize {
        self.player_count
    }

    /// Number of actions per player.
    pub fn action_count(&self) -> usize {
        self.action_count
    }

    /// Check if a specific player×action pair is valid.
    ///
    /// Returns `false` for out-of-bounds indices.
    pub fn is_valid(&self, player_idx: usize, action_idx: usize) -> bool {
        if player_idx >= self.player_count || action_idx >= self.action_count {
            return false;
        }
        self.results[player_idx * self.action_count + action_idx] == VALID
    }

    /// Get all valid actions for a specific player.
    ///
    /// Returns a fixed-size array of 6 booleans (one per action).
    pub fn valid_actions(&self, player_idx: usize) -> [bool; ACTION_COUNT] {
        let mut arr = [false; ACTION_COUNT];
        if player_idx < self.player_count {
            let base = player_idx * self.action_count;
            for (ai, slot) in arr
                .iter_mut()
                .enumerate()
                .take(self.action_count.min(ACTION_COUNT))
            {
                *slot = self.results[base + ai] == VALID;
            }
        }
        arr
    }

    /// Total number of valid actions across all players.
    pub fn total_valid(&self) -> usize {
        self.results.iter().filter(|&&v| v == VALID).count()
    }
}

/// Result of batch relevance scoring: `scores[player_idx * action_count + action_idx]`.
pub struct BatchRelevanceResult {
    scores: Vec<f32>,
    player_count: usize,
    action_count: usize,
}

impl BatchRelevanceResult {
    /// Get the relevance score for a specific player×action pair.
    ///
    /// Returns `0.0` for out-of-bounds indices.
    pub fn score(&self, player_idx: usize, action_idx: usize) -> f32 {
        if player_idx >= self.player_count || action_idx >= self.action_count {
            return 0.0;
        }
        self.scores[player_idx * self.action_count + action_idx]
    }

    /// Number of players in this batch.
    pub fn player_count(&self) -> usize {
        self.player_count
    }
}

// ── BomberWasmPruner ───────────────────────────────────────────

/// WASM bomber validator with papaya instance pool.
///
/// Uses a lock-free `papaya::HashMap` to store per-thread WASM instances,
/// eliminating global Mutex contention. Each thread gets its own `BomberInner`
/// on first access, with an uncontended `Mutex` wrapping the `wasmtime::Store`
/// (required because `Store` needs `&mut` for all operations).
///
/// # Thread Safety
///
/// - `Engine` and `Module` are `Arc`'d and immutable after construction
/// - Per-thread `BomberInner` stored in papaya HashMap (lock-free reads)
/// - Each thread's `Mutex<BomberInner>` is never contended
/// - `BomberWasmPruner` implements `Send + Sync`
pub struct BomberWasmPruner {
    /// Shared WASM engine (immutable after init).
    engine: Arc<Engine>,
    /// Shared compiled module (immutable after init).
    module: Arc<Module>,
    /// Validator name (extracted once at load time).
    name: String,
    /// Validator version (extracted once at load time).
    version: (u8, u8, u8),
    /// Lock-free per-thread instance pool.
    pool: HashMap<ThreadId, Mutex<BomberInner>>,
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
    /// Extracts required exports and optional batch exports.
    pub fn load(wasm_bytes: &[u8]) -> Result<Self, String> {
        // 1. Engine with fuel
        let mut config = Config::new();
        config.consume_fuel(true);
        let engine =
            Arc::new(Engine::new(&config).map_err(|e| format!("engine creation failed: {e}"))?);

        // 2. Compile module
        let module = Arc::new(
            Module::new(&engine, wasm_bytes)
                .map_err(|e| format!("WASM compilation failed: {e}"))?,
        );

        // 3. Create a temporary instance to extract name + version metadata
        let linker = Linker::new(&engine);
        let mut store = Store::new(&engine, ());
        let instance = linker
            .instantiate(&mut store, &module)
            .map_err(|e| format!("metadata instantiation failed: {e}"))?;
        let memory = instance
            .get_memory(&mut store, abi::MEMORY)
            .ok_or_else(|| format!("missing export: '{}'", abi::MEMORY))?;

        // Extract name
        let name_fn: TypedFunc<(), u32> = instance
            .get_typed_func(&mut store, abi::NAME)
            .map_err(|e| format!("missing export '{}': {e}", abi::NAME))?;
        store
            .set_fuel(FUEL_PER_CALL)
            .map_err(|e| format!("fuel setup failed: {e}"))?;
        let name_ptr = name_fn
            .call(&mut store, ())
            .map_err(|e| format!("name() call failed: {e}"))?;
        let name = read_cstring(&memory, &store, name_ptr, 256)?;

        // Extract version
        let version_fn: TypedFunc<(), u32> = instance
            .get_typed_func(&mut store, abi::VERSION)
            .map_err(|e| format!("missing export '{}': {e}", abi::VERSION))?;
        store
            .set_fuel(FUEL_PER_CALL)
            .map_err(|e| format!("fuel setup failed: {e}"))?;
        let packed = version_fn
            .call(&mut store, ())
            .map_err(|e| format!("version() call failed: {e}"))?;
        let version = (
            ((packed >> 16) & 0xFF) as u8,
            ((packed >> 8) & 0xFF) as u8,
            (packed & 0xFF) as u8,
        );

        // Drop the temporary store — real instances created per-thread via pool
        drop(store);

        Ok(Self {
            engine,
            module,
            name,
            version,
            pool: HashMap::new(),
        })
    }

    /// Execute a closure with this thread's `BomberInner`.
    ///
    /// Lazily creates a new WASM instance for the current thread on first
    /// access. The papaya HashMap provides lock-free reads for existing
    /// entries; the per-thread `Mutex` is uncontended.
    fn with_inner<R>(&self, f: impl FnOnce(&mut BomberInner) -> R) -> Option<R> {
        let id = std::thread::current().id();
        let guard = self.pool.pin();

        // Lock-free read: check if this thread already has an instance
        if guard.get(&id).is_none() {
            // First call for this thread — create instance
            let inner = match BomberInner::new(&self.engine, &self.module) {
                Ok(i) => Mutex::new(i),
                Err(_) => return None,
            };
            guard.insert(id, inner);
        }

        let mutex = guard.get(&id)?;
        let mut inner = match mutex.lock() {
            Ok(g) => g,
            Err(_) => return None,
        };

        Some(f(&mut inner))
    }

    /// Whether the loaded WASM module supports batch operations.
    pub fn has_batch(&self) -> bool {
        self.with_inner(|inner| inner.has_batch()).unwrap_or(false)
    }

    /// Check if an action is safe given game state (individual call).
    ///
    /// Uses zero-copy serialization (no Vec allocation).
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
        self.with_inner(|inner| {
            inner.call_is_valid(action_idx, grid, player_x, player_y, player_id, bombs)
        })
        .unwrap_or(false)
    }

    /// Get action relevance score via WASM (individual call).
    ///
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
        self.with_inner(|inner| {
            inner.call_relevance(action_idx, grid, player_x, player_y, player_id, bombs)
        })
        .unwrap_or(0.0)
    }

    /// Batch validate all player×action pairs in one FFI call.
    ///
    /// Serializes the grid+bombs once, then validates all N players × M actions
    /// in a single WASM call. Falls back to individual calls if the WASM module
    /// doesn't export `batch_is_valid`.
    ///
    /// # Arguments
    ///
    /// - `grid`: The arena grid (shared for all players)
    /// - `players`: Slice of `(player_id, x, y)` tuples (max 4)
    /// - `bombs`: Active bombs on the grid
    ///
    /// # Performance
    ///
    /// - With batch export: 1 serialization + 1 FFI call (vs 24 individual)
    /// - Without batch: falls back to 24 individual calls (still zero-copy)
    pub fn batch_validate(
        &self,
        grid: &ArenaGrid,
        players: &[(u8, i32, i32)],
        bombs: &[((i32, i32), u32, u32)],
    ) -> BatchResult {
        self.with_inner(|inner| {
            if inner.has_batch() {
                inner
                    .call_batch_validate(grid, players, bombs)
                    .unwrap_or_else(|| inner.call_batch_fallback(grid, players, bombs))
            } else {
                inner.call_batch_fallback(grid, players, bombs)
            }
        })
        .unwrap_or_else(BatchResult::empty)
    }

    /// Batch compute relevance scores for all player×action pairs.
    ///
    /// Falls back to individual `action_relevance` calls if the WASM module
    /// doesn't export `batch_relevance`.
    pub fn batch_relevance(
        &self,
        grid: &ArenaGrid,
        players: &[(u8, i32, i32)],
        bombs: &[((i32, i32), u32, u32)],
    ) -> Option<BatchRelevanceResult> {
        self.with_inner(|inner| inner.call_batch_relevance(grid, players, bombs))
            .flatten()
    }

    /// Get validator name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get validator version.
    pub fn version(&self) -> (u8, u8, u8) {
        self.version
    }
}

// ── Compile-Time Assertions ────────────────────────────────────

const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<BomberWasmPruner>();
    assert_send_sync::<BomberInner>();
    assert_send_sync::<BatchResult>();
    assert_send_sync::<BatchRelevanceResult>();
};

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_invalid_wasm_bytes_fails() {
        let result = BomberWasmPruner::load(b"not valid wasm");
        match result {
            Err(e) => assert!(
                e.contains("WASM compilation failed"),
                "unexpected error: {e}"
            ),
            Ok(_) => panic!("expected error for invalid WASM bytes"),
        }
    }

    #[test]
    fn load_from_file_not_found_fails() {
        let result = BomberWasmPruner::load_from_file("/nonexistent/path.wasm");
        match result {
            Err(e) => assert!(e.contains("failed to read"), "unexpected error: {e}"),
            Ok(_) => panic!("expected error for missing file"),
        }
    }

    #[test]
    fn fuel_constants_reasonable() {
        assert_eq!(FUEL_PER_CALL, 50_000);
        assert!(FUEL_BATCH_MULTIPLIER >= 2);
        // Batch fuel: 50K × 2 = 100K, sufficient for 24 individual checks
        assert!(FUEL_PER_CALL * FUEL_BATCH_MULTIPLIER >= 80_000);
    }

    #[test]
    fn actions_bytes_layout() {
        // Verify pre-computed actions array
        for i in 0..ACTION_COUNT {
            let off = i * 4;
            let val = u32::from_le_bytes([
                ACTIONS_BYTES[off],
                ACTIONS_BYTES[off + 1],
                ACTIONS_BYTES[off + 2],
                ACTIONS_BYTES[off + 3],
            ]);
            assert_eq!(val, i as u32, "action {i} mismatch");
        }
    }

    #[test]
    fn align8_rounds_up() {
        assert_eq!(align8(0), 0);
        assert_eq!(align8(1), 8);
        assert_eq!(align8(7), 8);
        assert_eq!(align8(8), 8);
        assert_eq!(align8(9), 16);
        assert_eq!(align8(680), 680); // already aligned
        assert_eq!(align8(681), 688);
    }

    #[test]
    fn batch_result_empty() {
        let result = BatchResult::empty();
        assert_eq!(result.player_count(), 0);
        assert!(!result.is_valid(0, 0));
    }

    #[test]
    fn batch_result_is_valid() {
        let result = BatchResult {
            results: vec![1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0],
            player_count: 2,
            action_count: 6,
        };
        assert!(result.is_valid(0, 0)); // player 0, action 0 = valid
        assert!(!result.is_valid(0, 1)); // player 0, action 1 = invalid
        assert!(result.is_valid(0, 2)); // player 0, action 2 = valid
        assert!(!result.is_valid(1, 0)); // player 1, action 0 = invalid
        assert!(result.is_valid(1, 1)); // player 1, action 1 = valid
        assert!(!result.is_valid(5, 0)); // out of bounds = false
        assert_eq!(result.total_valid(), 5);
    }

    #[test]
    fn batch_result_valid_actions() {
        let result = BatchResult {
            results: vec![1, 1, 0, 0, 0, 1],
            player_count: 1,
            action_count: 6,
        };
        let actions = result.valid_actions(0);
        assert!(actions[0]); // Up
        assert!(actions[1]); // Down
        assert!(!actions[2]); // Left
        assert!(!actions[3]); // Right
        assert!(!actions[4]); // Bomb
        assert!(actions[5]); // Wait
    }

    #[test]
    fn batch_relevance_result_score() {
        let result = BatchRelevanceResult {
            scores: vec![0.5, 0.8, 0.0, 1.0, 0.2, 0.3],
            player_count: 1,
            action_count: 6,
        };
        assert!((result.score(0, 0) - 0.5).abs() < 0.001);
        assert!((result.score(0, 1) - 0.8).abs() < 0.001);
        assert!((result.score(0, 2) - 0.0).abs() < 0.001);
        assert!((result.score(0, 3) - 1.0).abs() < 0.001);
        assert_eq!(result.score(1, 0), 0.0); // out of bounds
    }

    #[test]
    fn pruner_has_send_sync() {
        fn check<T: Send + Sync>() {}
        check::<BomberWasmPruner>();
        check::<BatchResult>();
        check::<BatchRelevanceResult>();
    }
}
