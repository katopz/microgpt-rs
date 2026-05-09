//! Thread-safe cache for compiled [`WasmPruner`] instances.
//!
//! First call to [`WasmPrunerCache::get_or_load`] compiles the `.wasm` file.
//! Subsequent calls return the cached [`Arc<WasmPruner>`]. Multiple domains
//! that reference the same `.wasm` file share a single compiled instance.
//!
//! # Thread Safety
//!
//! Uses a lock-free [`papaya::HashMap`] for concurrent access. WASM
//! compilation happens outside any critical section. Atomic operations
//! ensure correct behaviour even when multiple threads compile the same
//! file simultaneously.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::wasm::WasmPruner;

// ---------------------------------------------------------------------------
// WasmPrunerCache
// ---------------------------------------------------------------------------

/// Caches compiled [`WasmPruner`] instances keyed by resolved file path.
///
/// Relative `.wasm` paths from `domains.toml` are resolved against the
/// `pruner_dir` supplied at construction time.
///
/// ```ignore
/// use microgpt_rs::router::WasmPrunerCache;
/// use std::path::{Path, PathBuf};
///
/// let cache = WasmPrunerCache::new(PathBuf::from("./pruners"));
/// // Returns None if file missing or invalid WASM
/// let pruner = cache.get_or_load(Path::new("syn_validator.wasm"));
/// ```
pub struct WasmPrunerCache {
    cache: papaya::HashMap<PathBuf, Arc<WasmPruner>>,
    pruner_dir: PathBuf,
}

impl WasmPrunerCache {
    /// Create a new cache rooted at `pruner_dir`.
    ///
    /// Relative `.wasm` paths are resolved against this directory.
    pub fn new(pruner_dir: PathBuf) -> Self {
        Self {
            cache: papaya::HashMap::new(),
            pruner_dir,
        }
    }

    /// Get a compiled [`WasmPruner`], loading and caching on first access.
    ///
    /// Returns `None` if the file does not exist or contains invalid WASM.
    /// On success, subsequent calls for the same path return the cached
    /// [`Arc`] without recompiling.
    pub fn get_or_load(&self, relative_path: &Path) -> Option<Arc<WasmPruner>> {
        let full_path = self.pruner_dir.join(relative_path);

        // Fast path: already cached.
        {
            let cache = self.cache.pin();
            if let Some(cached) = cache.get(&full_path) {
                return Some(Arc::clone(cached));
            }
        }

        // Slow path: compile, then insert.
        let path_str = full_path.to_str()?;
        let pruner = WasmPruner::load_from_file(path_str).ok()?;
        let arc = Arc::new(pruner);

        let cache = self.cache.pin();
        // Another thread may have inserted while we compiled — prefer existing.
        Some(cache.get_or_insert(full_path, arc).clone())
    }

    /// Returns the number of cached pruners.
    #[cfg(test)]
    fn len(&self) -> usize {
        self.cache.pin().len()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal valid WAT that exports the required ABI.
    fn accept_all_wat() -> &'static str {
        r#"
(module
  (memory (export "memory") 1)
  (data (i32.const 0) "test_cache\00")
  (func (export "is_valid") (param i32 i32 i32 i32) (result i32)
    i32.const 1)
  (func (export "relevance") (param i32 i32 i32 i32) (result i32)
    i32.const 1065353216)
  (func (export "name") (result i32) i32.const 0)
  (func (export "version") (result i32) i32.const 0x000100))
"#
    }

    fn compile_wat() -> Vec<u8> {
        wat::parse_str(accept_all_wat()).expect("valid WAT")
    }

    #[test]
    fn test_cache_returns_none_for_missing_file() {
        let dir = std::env::temp_dir().join("microgpt_test_cache_missing");
        let cache = WasmPrunerCache::new(dir);
        let result = cache.get_or_load(Path::new("nonexistent.wasm"));
        assert!(result.is_none());
    }

    #[test]
    fn test_cache_loads_and_caches() {
        let dir = std::env::temp_dir().join("microgpt_test_cache_load");
        std::fs::create_dir_all(&dir).ok();

        let wasm_path = dir.join("test.wasm");
        std::fs::write(&wasm_path, compile_wat()).expect("write wasm");

        let cache = WasmPrunerCache::new(dir.clone());

        // First load — compiles.
        let first = cache.get_or_load(Path::new("test.wasm"));
        assert!(first.is_some(), "first load should succeed");
        assert_eq!(cache.len(), 1);

        // Second load — returns cached Arc (same pointer).
        let second = cache.get_or_load(Path::new("test.wasm"));
        assert!(second.is_some(), "second load should succeed");
        assert_eq!(cache.len(), 1, "should not duplicate entry");

        // Verify they point to the same underlying WasmPruner.
        assert!(
            Arc::ptr_eq(first.as_ref().unwrap(), second.as_ref().unwrap()),
            "cached Arcs should be identical"
        );

        // Clean up.
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_cache_returns_none_for_invalid_wasm() {
        let dir = std::env::temp_dir().join("microgpt_test_cache_invalid");
        std::fs::create_dir_all(&dir).ok();

        let bad_path = dir.join("bad.wasm");
        std::fs::write(&bad_path, b"not valid wasm bytes").expect("write bad wasm");

        let cache = WasmPrunerCache::new(dir.clone());
        let result = cache.get_or_load(Path::new("bad.wasm"));
        assert!(result.is_none(), "invalid WASM should return None");
        assert_eq!(cache.len(), 0, "invalid WASM should not be cached");

        std::fs::remove_dir_all(&dir).ok();
    }
}
