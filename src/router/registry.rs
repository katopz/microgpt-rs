//! Expert registry — config-driven domain-to-pruner mapping.
//!
//! Loads domain definitions from [`RouterConfig`], resolves native and WASM
//! pruners, and serves [`ExpertBundle`] instances by domain name. Unknown
//! domains fall back to the default bundle (`"general"`).
//!
//! # Pruner Resolution Priority
//!
//! 1. `native_pruner` field — built-in pruner by name (`"no_pruner"`,
//!    `"sudoku"`, `"tactical"`)
//! 2. `pruner` field — WASM file path, loaded via [`WasmPruner`]
//! 3. [`NoScreeningPruner`] fallback — no pruning applied

use std::collections::HashMap;
use std::path::Path;

use crate::speculative::types::{NoScreeningPruner, ScreeningPruner};
use crate::types::{LoraAdapter, LoraPair};
use crate::wasm::WasmPruner;

use super::types::{DomainConfig, ExpertBundle, RouterConfig};

// ---------------------------------------------------------------------------
// ExpertRegistry
// ---------------------------------------------------------------------------

/// Registry mapping domain names to [`ExpertBundle`] instances.
///
/// Constructed once at startup from a [`RouterConfig`]. Each domain's pruner
/// is resolved eagerly — if a WASM file fails to load, the domain silently
/// falls back to [`NoScreeningPruner`] so the system degrades gracefully.
pub struct ExpertRegistry {
    bundles: HashMap<String, ExpertBundle>,
    default_domain: String,
}

impl ExpertRegistry {
    /// Build the registry from a [`RouterConfig`] and a pruner directory.
    ///
    /// Relative WASM and LoRA paths in the config are resolved against
    /// `pruner_dir`. A default bundle named `"general"` is always present
    /// (auto-created if the config doesn't define one).
    pub fn from_config(config: &RouterConfig, pruner_dir: &Path) -> Self {
        let mut bundles = HashMap::new();
        let mut default_domain = String::from("general");

        for domain_config in &config.domain {
            let pruner = Self::resolve_pruner(domain_config, pruner_dir);
            let lora_path = domain_config.lora.as_ref().map(|l| pruner_dir.join(l));
            let lora_pair = Self::resolve_lora_pair(domain_config, pruner_dir);

            let bundle = ExpertBundle {
                domain: domain_config.name.clone(),
                pruner,
                lora_path,
                lora_pair,
            };

            if domain_config.name == "general" {
                default_domain = domain_config.name.clone();
            }

            bundles.insert(domain_config.name.clone(), bundle);
        }

        // Ensure the default bundle exists.
        if !bundles.contains_key("general") {
            bundles.insert(
                "general".into(),
                ExpertBundle {
                    domain: "general".into(),
                    pruner: Box::new(NoScreeningPruner),
                    lora_path: None,
                    lora_pair: LoraPair::none(),
                },
            );
        }

        Self {
            bundles,
            default_domain,
        }
    }

    /// Get the [`ExpertBundle`] for a domain name.
    ///
    /// Falls back to the default bundle (`"general"`) when the domain is
    /// not registered. The default bundle is guaranteed to exist.
    pub fn get_expert(&self, domain: &str) -> &ExpertBundle {
        match self.bundles.get(domain) {
            Some(bundle) => bundle,
            None => self
                .bundles
                .get(&self.default_domain)
                .expect("default bundle always exists"),
        }
    }

    /// Returns the number of registered bundles (for testing).
    #[cfg(test)]
    pub fn len(&self) -> usize {
        self.bundles.len()
    }

    /// Returns `true` if no bundles are registered (for testing).
    #[cfg(test)]
    pub fn is_empty(&self) -> bool {
        self.bundles.is_empty()
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    /// Resolve a domain config to a concrete [`ScreeningPruner`].
    fn resolve_pruner(domain: &DomainConfig, pruner_dir: &Path) -> Box<dyn ScreeningPruner> {
        // 1. Native pruner by name.
        if let Some(ref native) = domain.native_pruner {
            return Self::resolve_native(native, &domain.name);
        }

        // 2. WASM pruner from file.
        if let Some(ref wasm_file) = domain.pruner {
            let full_path = pruner_dir.join(wasm_file);
            let path_str = match full_path.to_str() {
                Some(s) => s,
                None => {
                    eprintln!(
                        "[router] non-UTF-8 path '{}'; \
                         falling back to NoScreeningPruner for domain '{}'",
                        full_path.display(),
                        domain.name,
                    );
                    return Box::new(NoScreeningPruner);
                }
            };
            match WasmPruner::load_from_file(path_str) {
                Ok(pruner) => return Box::new(pruner),
                Err(e) => {
                    eprintln!(
                        "[router] failed to load WASM pruner '{}': {e}; \
                         falling back to NoScreeningPruner for domain '{}'",
                        full_path.display(),
                        domain.name,
                    );
                }
            }
        }

        // 3. No pruning.
        Box::new(NoScreeningPruner)
    }

    /// Resolve a `native_pruner` name to a boxed [`ScreeningPruner`].
    fn resolve_native(name: &str, domain: &str) -> Box<dyn ScreeningPruner> {
        match name {
            "no_pruner" => Box::new(NoScreeningPruner),

            "sudoku" => {
                // SudokuPruner requires a Sudoku9x9 board provided at runtime,
                // not at config-load time. Fall back to NoScreeningPruner;
                // the caller should provide a SudokuPruner directly when a
                // board is available.
                eprintln!(
                    "[router] native_pruner 'sudoku' requires a runtime board; \
                     falling back to NoScreeningPruner for domain '{domain}'"
                );
                Box::new(NoScreeningPruner)
            }

            "tactical" => {
                // TacticalPruner requires a map string provided at runtime,
                // not at config-load time. Fall back to NoScreeningPruner;
                // the caller should provide a TacticalPruner directly when
                // a map is available.
                eprintln!(
                    "[router] native_pruner 'tactical' requires a runtime map; \
                     falling back to NoScreeningPruner for domain '{domain}'"
                );
                Box::new(NoScreeningPruner)
            }

            other => {
                eprintln!(
                    "[router] unknown native_pruner '{other}'; \
                     falling back to NoScreeningPruner for domain '{domain}'"
                );
                Box::new(NoScreeningPruner)
            }
        }
    }

    /// Resolve dual LoRA adapters from domain config.
    /// Priority: reader_lora/writer_lora fields > legacy lora field.
    /// Graceful degradation: failed loads log warning, proceed without LoRA.
    fn resolve_lora_pair(domain: &DomainConfig, pruner_dir: &Path) -> LoraPair {
        let reader = domain.reader_lora.as_ref().and_then(|p| {
            let path = pruner_dir.join(p);
            match LoraAdapter::load(&path) {
                Ok(adapter) => Some(adapter),
                Err(e) => {
                    eprintln!(
                        "[router] failed to load reader LoRA '{}': {e}; proceeding without",
                        path.display(),
                    );
                    None
                }
            }
        });

        // Writer: writer_lora > legacy lora field
        let writer_path = domain.writer_lora.as_ref().or(domain.lora.as_ref());
        let writer = writer_path.and_then(|p| {
            let path = pruner_dir.join(p);
            match LoraAdapter::load(&path) {
                Ok(adapter) => Some(adapter),
                Err(e) => {
                    eprintln!(
                        "[router] failed to load writer LoRA '{}': {e}; proceeding without",
                        path.display(),
                    );
                    None
                }
            }
        });

        LoraPair { reader, writer }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn make_config(domains: Vec<DomainConfig>) -> RouterConfig {
        RouterConfig { domain: domains }
    }

    fn no_pruner_domain(name: &str) -> DomainConfig {
        DomainConfig {
            name: name.into(),
            keywords: vec![],
            pruner: None,
            lora: None,
            reader_lora: None,
            writer_lora: None,
            native_pruner: Some("no_pruner".into()),
        }
    }

    #[test]
    fn test_empty_config_creates_default_general() {
        let config = make_config(vec![]);
        let registry = ExpertRegistry::from_config(&config, Path::new("."));

        assert_eq!(registry.len(), 1);
        let bundle = registry.get_expert("general");
        assert_eq!(bundle.domain, "general");
        assert!(bundle.lora_path.is_none());
    }

    #[test]
    fn test_unknown_domain_falls_back_to_default() {
        let config = make_config(vec![no_pruner_domain("general")]);
        let registry = ExpertRegistry::from_config(&config, Path::new("."));

        let bundle = registry.get_expert("nonexistent_domain");
        assert_eq!(bundle.domain, "general");
    }

    #[test]
    fn test_no_pruner_domain() {
        let config = make_config(vec![
            no_pruner_domain("general"),
            DomainConfig {
                name: "test_domain".into(),
                keywords: vec!["test".into()],
                pruner: None,
                lora: None,
                reader_lora: None,
                writer_lora: None,
                native_pruner: Some("no_pruner".into()),
            },
        ]);
        let registry = ExpertRegistry::from_config(&config, Path::new("."));

        let bundle = registry.get_expert("test_domain");
        assert_eq!(bundle.domain, "test_domain");
        assert!(bundle.lora_path.is_none());
    }

    #[test]
    fn test_lora_path_resolved() {
        let config = make_config(vec![
            no_pruner_domain("general"),
            DomainConfig {
                name: "py2rs".into(),
                keywords: vec!["python".into()],
                pruner: None,
                lora: Some("py2rs_lora.bin".into()),
                reader_lora: None,
                writer_lora: None,
                native_pruner: Some("no_pruner".into()),
            },
        ]);
        let registry = ExpertRegistry::from_config(&config, Path::new("/pruners"));

        let bundle = registry.get_expert("py2rs");
        assert_eq!(
            bundle.lora_path,
            Some(PathBuf::from("/pruners/py2rs_lora.bin"))
        );
    }

    #[test]
    fn test_wasm_pruner_missing_file_falls_back() {
        let config = make_config(vec![
            no_pruner_domain("general"),
            DomainConfig {
                name: "rust_code".into(),
                keywords: vec!["rust".into()],
                pruner: Some("nonexistent.wasm".into()),
                lora: None,
                reader_lora: None,
                writer_lora: None,
                native_pruner: None,
            },
        ]);
        let registry = ExpertRegistry::from_config(&config, Path::new("/no/such/dir"));

        // Should not panic; falls back to NoScreeningPruner.
        let bundle = registry.get_expert("rust_code");
        assert_eq!(bundle.domain, "rust_code");
    }

    #[test]
    fn test_multiple_domains() {
        let config = make_config(vec![
            no_pruner_domain("general"),
            no_pruner_domain("sudoku"),
            no_pruner_domain("pathfinding"),
        ]);
        let registry = ExpertRegistry::from_config(&config, Path::new("."));

        assert_eq!(registry.len(), 3);
        assert_eq!(registry.get_expert("general").domain, "general");
        assert_eq!(registry.get_expert("sudoku").domain, "sudoku");
        assert_eq!(registry.get_expert("pathfinding").domain, "pathfinding");
    }

    #[test]
    fn test_no_native_pruner_and_no_wasm_falls_back() {
        let config = make_config(vec![
            no_pruner_domain("general"),
            DomainConfig {
                name: "bare_domain".into(),
                keywords: vec!["bare".into()],
                pruner: None,
                lora: None,
                reader_lora: None,
                writer_lora: None,
                native_pruner: None,
            },
        ]);
        let registry = ExpertRegistry::from_config(&config, Path::new("."));

        let bundle = registry.get_expert("bare_domain");
        assert_eq!(bundle.domain, "bare_domain");
    }
}
