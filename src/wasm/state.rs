//! WASM validator state — stored in wasmtime Store for tracking validator metadata.

/// State carried inside the wasmtime [`Store`] for a loaded WASM validator.
///
/// Contains metadata extracted from the validator module via exported
/// `name` and `version` functions during [`WasmPruner::load`].
///
/// [`Store`]: wasmtime::Store
/// [`WasmPruner::load`]: super::WasmPruner::load
#[derive(Debug, Clone)]
pub struct ValidatorState {
    /// Human-readable validator name (e.g., "sudoku", "json", "python").
    pub name: String,
    /// Semantic version triple (major, minor, patch).
    pub version: (u8, u8, u8),
}

impl ValidatorState {
    /// Create a default state with placeholder values.
    /// Used temporarily during module instantiation before exports are queried.
    pub fn placeholder() -> Self {
        Self {
            name: String::from("unknown"),
            version: (0, 0, 0),
        }
    }

    /// Create a new state with the given name and version.
    pub fn new(name: String, version: (u8, u8, u8)) -> Self {
        Self { name, version }
    }
}

impl Default for ValidatorState {
    fn default() -> Self {
        Self::placeholder()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_placeholder_state() {
        let state = ValidatorState::placeholder();
        assert_eq!(state.name, "unknown");
        assert_eq!(state.version, (0, 0, 0));
    }

    #[test]
    fn test_new_state() {
        let state = ValidatorState::new(String::from("test_validator"), (1, 2, 3));
        assert_eq!(state.name, "test_validator");
        assert_eq!(state.version, (1, 2, 3));
    }

    #[test]
    fn test_default_is_placeholder() {
        let default = ValidatorState::default();
        let placeholder = ValidatorState::placeholder();
        assert_eq!(default.name, placeholder.name);
        assert_eq!(default.version, placeholder.version);
    }
}
