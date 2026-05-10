//! REST-based Speculative Decoding (REST) module.
//!
//! Connects microgpt-rs to anyrag for Retrieval-Based Speculative Decoding.
//! Queries anyrag's `/search/vector` endpoint with hidden state embeddings,
//! then injects retrieved token continuations into the DDTree.
//!
//! # Feature Flag
//!
//! This module is only available when the `rest` feature is enabled:
//!
//! ```toml
//! [dependencies]
//! microgpt-rs = { features = ["rest"] }
//! ```

pub mod client;
pub mod types;

pub use client::{RestClient, RestError, RetrievalResult};
pub use types::{
    AgentHints, DetokenizeRequest, DetokenizeResponse, SearchRequest, SearchResponse,
    SearchResultItem, TokenizeRequest, TokenizeResponse,
};
