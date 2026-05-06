# Handover 006: REST Speculative Decoding

## What Happened

Implemented Plan 009: REST Speculative Decoding — connecting mini-dllm to anyrag for Retrieval-Based Speculative Decoding. The hidden state extraction was already done in Plan 010 (`ForwardContext.hidden_state`). This plan adds:

1. **REST client module** (`src/rest/`) — async HTTP client querying anyrag's `/search/vector` endpoint
2. **DDTree merge** (`merge_retrieved_branches`) — injects retrieved token continuations into the speculative decoding tree with blended scores
3. **Integration function** (`speculative_step_rest`) — full pipeline: DFlash → DDTree → target forward → REST query → merge → verify

All REST code is behind the `rest` feature flag. The `merge_retrieved_branches` function is NOT feature-gated (pure computation).

## Where Is the Plan/Code/Test

- **Plan**: `.plans/009_rest_speculative_decoding.md`
- **Code**:
  - `Cargo.toml` — added `reqwest`, `tokio`, `serde`, `serde_json` behind `rest` feature
  - `src/rest/mod.rs` — module index with re-exports
  - `src/rest/client.rs` — `RestClient`, `RetrievalResult`, `RestError`, `embedding_to_query`
  - `src/rest/types.rs` — `SearchRequest`, `SearchResponse`, `SearchResultItem` with `extract_token_sequence()`
  - `src/lib.rs` — `#[cfg(feature = "rest")] pub mod rest;`
  - `src/speculative/dd_tree.rs` — `merge_retrieved_branches()` function + 5 tests
  - `src/speculative/mod.rs` — re-exports `merge_retrieved_branches` and `speculative_step_rest`
  - `src/speculative/step.rs` — `speculative_step_rest()` async function
- **Tests**:
  - `src/rest/types.rs` — 6 tests for token sequence extraction from descriptions
  - `src/rest/client.rs` — 5 tests for client construction, embedding conversion, errors
  - `src/speculative/dd_tree.rs` — 5 tests for merge (noop, budget, sorting, empty tree, zero weight)

## Reflection Struggling/Solved

- **anyrag API mismatch**: Plan originally assumed `/search/vector` accepts raw vector embeddings. Actual API takes text `query` and generates embeddings server-side. Solved by adding `embedding_to_query()` that quantizes hidden state into searchable text tokens. Token sequences are parsed from document descriptions in `"tokens:1,2,3,4"` format.
- **serde_json PartialEq ambiguity**: `assert_eq!(extract_parent_tokens(0, 0), vec![])` failed with `--all-features` because serde_json adds `impl PartialEq<Value> for usize`. Fixed by using typed variable + `assert!(empty.is_empty())`.
- **TreeNode parent_path uses u64**: Plan spec used `u128` but existing codebase uses `u64` with 5-bit packing. Adapted `merge_retrieved_branches` to use the existing `u64` path format with `fold`.

## Remain Work

- [ ] Benchmark: `Speculative (REST)` vs `Speculative (Simulated)` acceptance rate (Plan Phase 4.3)
- [ ] Example: `examples/rest_demo.rs` behind `rest` feature (Plan Phase 4.4)
- [ ] End-to-end test with live anyrag instance (requires running server)
- [ ] Token sequence storage in anyrag documents during ingestion (anyrag side)
- [ ] Experiment with hidden state projection layer for better retrieval quality

## Issues Ref

- No `.issues/` files created for this plan.

## How to Dev/Test

```bash
# Check REST feature compiles
cargo check --features rest

# Run all tests including REST
cargo test --all-features

# Run just REST module tests
cargo test --features rest -- rest

# Run just merge tests (no feature needed)
cargo test -- merge

# Clippy all features
cargo clippy --all-features

# Note: speculative_step_rest is async, needs tokio runtime for live testing
# RestClient::retrieve() requires anyrag server at configured base_url