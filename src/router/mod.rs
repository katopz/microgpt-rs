//! Prompt Router — config-driven domain routing for MoE-style expert selection.
//!
//! Classifies a user prompt once per request into a semantic domain (e.g.,
//! `"sudoku"`, `"rust_code"`), then selects the appropriate [`ScreeningPruner`]
//! and optional LoRA adapter from a registry. The selected expert bundle is
//! locked for the entire DDTree generation, preventing domain drift.
//!
//! # Architecture
//!
//! ```text
//! Prompt → PromptRouter::route() → RouteDecision { domain, ... }
//!                                        ↓
//!                               ExpertRegistry::get_expert(domain)
//!                                        ↓
//!                               ExpertBundle { pruner, lora_path, lora_pair }
//!                                        ↓
//!                          build_dd_tree_screened(marginals, config, &bundle.pruner)
//! ```
//!
//! # Feature Flag
//!
//! This module is gated behind the `router` feature:
//!
//! ```toml
//! [dependencies]
//! microgpt-rs = { features = ["router"] }
//! ```
//!
//! # Quick Start
//!
//! ```ignore
//! use microgpt_rs::router::{KeywordRouter, ExpertRegistry, PromptRouter};
//! use microgpt_rs::router::types::RouterConfig;
//!
//! let config: RouterConfig = toml::from_str(&toml_str)?;
//! let router = KeywordRouter::new(config.domain.clone());
//! let registry = ExpertRegistry::from_config(&config, pruner_dir.as_path());
//!
//! let decision = router.route("solve this sudoku puzzle");
//! let expert = registry.get_expert(&decision.domain);
//! // expert.pruner is a Box<dyn ScreeningPruner> ready for DDTree
//! ```

pub mod keyword;
pub mod prompt_router;
pub mod registry;
pub mod types;
pub mod wasm_cache;

#[cfg(feature = "embedding_router")]
pub mod embedding;
#[cfg(feature = "embedding_router")]
pub mod projector;

pub use keyword::KeywordRouter;
pub use prompt_router::PromptRouter;
pub use registry::ExpertRegistry;
pub use types::{DomainConfig, ExpertBundle, RouteDecision, RouterConfig};
pub use wasm_cache::WasmPrunerCache;

#[cfg(feature = "embedding_router")]
pub use embedding::EmbeddingRouter;
#[cfg(feature = "embedding_router")]
pub use projector::{EmbeddingProjector, TruncatePadProjector};
#[cfg(feature = "embedding_router")]
pub use types::{EmbeddingExpertBundle, EmbeddingRouteDecision, EmbeddingRouterConfig};
