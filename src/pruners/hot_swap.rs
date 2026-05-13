//! Hot-swap pruner for runtime reload of pruner from disk.
//!
//! Wraps any [`ScreeningPruner`] with runtime reload capability.
//! Uses blake3 hash comparison to detect file changes — only reloads
//! when the file actually changed.
//!
//! # Usage
//!
//! ```rust,ignore
//! let pruner = HotSwapPruner::new(
//!     Path::new("validator.wasm"),
//!     Box::new(|path| WasmPruner::load_from_file(path.to_str().unwrap())),
//! )?;
//!
//! // Later, file changes on disk...
//! let changed = pruner.reload()?;
//! if changed {
//!     println!("Reloaded! Version: {}", pruner.version());
//! }
//! ```
//!
//! # Thread Safety
//!
//! Uses `RwLock` for the inner pruner — read-heavy (every `relevance()` call),
//! write only on [`reload`](Self::reload). Safe to share across threads.

use std::fs;
use std::io::Result;
use std::path::{Path, PathBuf};
use std::sync::RwLock;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::speculative::types::ScreeningPruner;

// ── Type Aliases ────────────────────────────────────────────────

/// Factory closure type for loading a pruner from a file path.
type PrunerFactory<P> = Box<dyn Fn(&Path) -> Result<P> + Send + Sync>;

// ── Inner State ─────────────────────────────────────────────────

/// Mutable inner state behind `RwLock`.
struct HotSwapInner<P> {
    pruner: P,
    hash: [u8; 32],
}

// ── HotSwapPruner ────────────────────────────────────────────────

/// Runtime-reloadable wrapper for any [`ScreeningPruner`].
///
/// Watches a file on disk. When [`reload`](Self::reload) is called:
/// 1. Reads the file and computes blake3 hash
/// 2. If hash matches current, returns `Ok(false)` (no change)
/// 3. If hash differs, calls the factory to create a new pruner
/// 4. Replaces the inner pruner and increments version counter
///
/// The factory closure is stored and reused on every reload.
pub struct HotSwapPruner<P: ScreeningPruner> {
    inner: RwLock<HotSwapInner<P>>,
    path: PathBuf,
    factory: PrunerFactory<P>,
    version: AtomicU64,
}

impl<P: ScreeningPruner> HotSwapPruner<P> {
    /// Create a new hot-swap pruner.
    ///
    /// Loads the initial pruner from `path` using `factory`.
    /// The `factory` closure is stored for future [`reload`](Self::reload) calls.
    pub fn new(path: &Path, factory: PrunerFactory<P>) -> Result<Self> {
        let bytes = fs::read(path)?;
        let hash: [u8; 32] = blake3::hash(&bytes).into();
        let pruner = factory(path)?;

        Ok(Self {
            inner: RwLock::new(HotSwapInner { pruner, hash }),
            path: path.to_path_buf(),
            factory,
            version: AtomicU64::new(1),
        })
    }

    /// Reload the pruner from disk if the file has changed.
    ///
    /// Returns `Ok(true)` if reloaded (file changed), `Ok(false)` if unchanged.
    /// Increments version counter on successful reload.
    pub fn reload(&self) -> Result<bool> {
        let bytes = fs::read(&self.path)?;
        let new_hash: [u8; 32] = blake3::hash(&bytes).into();

        // Fast path: read lock to check hash
        {
            let inner = self.inner.read().unwrap();
            if new_hash == inner.hash {
                return Ok(false);
            }
        }

        // Slow path: file changed — create new pruner under write lock
        let new_pruner = (self.factory)(&self.path)?;

        {
            let mut inner = self.inner.write().unwrap();
            inner.pruner = new_pruner;
            inner.hash = new_hash;
        }

        self.version.fetch_add(1, Ordering::Relaxed);
        Ok(true)
    }

    /// Current version counter (starts at 1, increments on each reload).
    pub fn version(&self) -> u64 {
        self.version.load(Ordering::Relaxed)
    }

    /// Path being watched.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Access the inner pruner with a read lock.
    ///
    /// Use for inspection (e.g., reading name/version from a WASM pruner).
    pub fn with_inner<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&P) -> R,
    {
        let inner = self.inner.read().unwrap();
        f(&inner.pruner)
    }
}

impl<P: ScreeningPruner> ScreeningPruner for HotSwapPruner<P> {
    fn relevance(&self, depth: usize, token_idx: usize, parent_tokens: &[usize]) -> f32 {
        let inner = self.inner.read().unwrap();
        inner.pruner.relevance(depth, token_idx, parent_tokens)
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Simple pruner whose relevance is read from a file.
    struct FileValuePruner {
        value: f32,
    }

    impl FileValuePruner {
        fn load(path: &Path) -> Result<Self> {
            let content = fs::read_to_string(path)?;
            let value = content.trim().parse::<f32>().unwrap_or(1.0);
            Ok(Self { value })
        }
    }

    impl ScreeningPruner for FileValuePruner {
        fn relevance(&self, _depth: usize, _token_idx: usize, _parent_tokens: &[usize]) -> f32 {
            self.value
        }
    }

    fn temp_path(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "microgpt_test_hotswap_{name}_{pid}.txt",
            pid = std::process::id()
        ))
    }

    fn write_file(path: &Path, content: &str) {
        fs::write(path, content).unwrap();
    }

    fn make_pruner(path: &Path) -> Result<HotSwapPruner<FileValuePruner>> {
        HotSwapPruner::new(path, Box::new(FileValuePruner::load))
    }

    #[test]
    fn test_reload_same_file_no_version_bump() {
        let path = temp_path("same");
        write_file(&path, "0.8");

        let pruner = make_pruner(&path).unwrap();
        assert_eq!(pruner.version(), 1);

        let changed = pruner.reload().unwrap();
        assert!(!changed);
        assert_eq!(pruner.version(), 1);

        let _ = fs::remove_file(&path);
    }

    #[test]
    fn test_reload_changed_file_version_bump() {
        let path = temp_path("changed");
        write_file(&path, "0.5");

        let pruner = make_pruner(&path).unwrap();
        assert_eq!(pruner.version(), 1);
        assert!((pruner.relevance(0, 0, &[]) - 0.5).abs() < 0.01);

        // Change file content
        write_file(&path, "0.9");

        let changed = pruner.reload().unwrap();
        assert!(changed);
        assert_eq!(pruner.version(), 2);
        assert!((pruner.relevance(0, 0, &[]) - 0.9).abs() < 0.01);

        let _ = fs::remove_file(&path);
    }

    #[test]
    fn test_pruner_works_after_reload() {
        let path = temp_path("after");
        write_file(&path, "0.1");

        let pruner = make_pruner(&path).unwrap();
        assert!((pruner.relevance(0, 0, &[]) - 0.1).abs() < 0.01);

        // Reload with different value
        write_file(&path, "0.7");
        pruner.reload().unwrap();

        // Verify new value is used
        assert!((pruner.relevance(0, 0, &[]) - 0.7).abs() < 0.01);
        assert!((pruner.relevance(5, 42, &[1, 2, 3]) - 0.7).abs() < 0.01);

        let _ = fs::remove_file(&path);
    }

    #[test]
    fn test_multiple_reloads() {
        let path = temp_path("multi");
        write_file(&path, "0.1");

        let pruner = make_pruner(&path).unwrap();
        assert_eq!(pruner.version(), 1);

        write_file(&path, "0.2");
        assert!(pruner.reload().unwrap());
        assert_eq!(pruner.version(), 2);

        write_file(&path, "0.3");
        assert!(pruner.reload().unwrap());
        assert_eq!(pruner.version(), 3);

        // No change
        assert!(!pruner.reload().unwrap());
        assert_eq!(pruner.version(), 3);

        let _ = fs::remove_file(&path);
    }

    #[test]
    fn test_with_inner_inspection() {
        let path = temp_path("inspect");
        write_file(&path, "0.42");

        let pruner = make_pruner(&path).unwrap();
        let value = pruner.with_inner(|p| p.value);
        assert!((value - 0.42).abs() < 0.01);

        let _ = fs::remove_file(&path);
    }
}
