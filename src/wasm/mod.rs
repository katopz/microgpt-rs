//! WASM Validator Pipeline — sandboxed constraint pruning via WASM modules.
//!
//! Provides [`WasmPruner`] which implements [`ConstraintPruner`] by delegating
//! token validation to a sandboxed WASM module. No WASI access, fuel-limited
//! execution.
//!
//! # Feature Flag
//!
//! This module is gated behind the `wasm` feature:
//!
//! ```toml
//! [dependencies]
//! microgpt-rs = { features = ["wasm"] }
//! ```
//!
//! # Example
//!
//! ```ignore
//! use microgpt_rs::wasm::WasmPruner;
//! use microgpt_rs::speculative::types::ConstraintPruner;
//!
//! let wasm_bytes = wat::parse_str(r#"
//!     (module
//!       (memory (export "memory") 1)
//!       (data (i32.const 0) "my_validator\00")
//!       (func (export "is_valid") (param i32 i32 i32 i32) (result i32)
//!         i32.const 1)
//!       (func (export "name") (result i32) i32.const 0)
//!       (func (export "version") (result i32) i32.const 0x000100))
//! "#)?;
//!
//! let pruner = WasmPruner::load(&wasm_bytes)?;
//! assert!(pruner.is_valid(0, 42, &[]));
//! ```

mod abi;
mod state;
mod wasm_pruner;

pub use abi::{
    EXPORT_IS_VALID, EXPORT_MEMORY, EXPORT_NAME, EXPORT_VALIDATE_STRING, EXPORT_VERSION,
    FUEL_PER_CALL, INVALID, MAX_MEMORY_PAGES, MAX_PARENT_TOKENS, SCRATCH_BUFFER_OFFSET,
    SCRATCH_BUFFER_SIZE, VALID, VALIDATOR_NAME_OFFSET, VALIDATOR_STATE_OFFSET,
};
pub use state::ValidatorState;
pub use wasm_pruner::WasmPruner;
