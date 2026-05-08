pub mod benchmark;
pub mod percepta;
pub mod plot;
pub mod speculative;
pub mod tokenizer;
pub mod transformer;
pub mod types;

#[cfg(debug_assertions)]
pub mod alloc;

/// Debug-only global allocator that tracks allocation count and bytes.
#[cfg(debug_assertions)]
#[global_allocator]
static GLOBAL_ALLOC: alloc::TrackingAllocator = alloc::TrackingAllocator;

#[cfg(feature = "rest")]
pub mod rest;

#[cfg(feature = "validator")]
pub mod validator;

#[cfg(feature = "gpu")]
pub mod gpu;

#[cfg(feature = "wasm")]
pub mod wasm;
