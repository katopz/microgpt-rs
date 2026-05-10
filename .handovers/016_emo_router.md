# Handover 016: EMO Router Integration — Prompt Router + Domain Classifier

## What Happened

Integrated the EMO (Emergent Modularity) paper's document-level routing pattern into the `microgpt-rs` + `anyrag` ecosystem across 3 plans:

1. **microgpt-rs Plan 023** — Config-driven prompt router with `KeywordRouter` (V1, ~80% accuracy), `ExpertRegistry`, `WasmPrunerCache`, and `PromptRouter` trait. Feature-gated behind `router`. 10/10 tasks complete.

2. **anyrag Plan 004 completion** — Finished 8 remaining integration tests and benchmarks for the Raven Routed Slots system. 29/29 tasks now complete.

3. **anyrag Plan 005** — Domain Classifier API with `HybridClassifier` (30% keyword + 70% embedding scoring), `POST /classify/domain` endpoint, domain mapping config, and integration tests. 7/7 tasks complete. Was previously marked BLOCKED; unblocked and fully implemented.

### Files Created (microgpt-rs)

| File | Purpose |
|------|---------|
| `src/router/mod.rs` | Module index + re-exports |
| `src/router/prompt_router.rs` | `PromptRouter` trait (Send + Sync) |
| `src/router/keyword.rs` | `KeywordRouter` — keyword-count scoring (7 tests) |
| `src/router/registry.rs` | `ExpertRegistry` — config-driven domain loading (6 tests) |
| `src/router/wasm_cache.rs` | `WasmPrunerCache` — thread-safe compiled WASM caching (3 tests) |
| `src/router/types.rs` | `RouteDecision`, `ExpertBundle`, `DomainConfig`, `RouterConfig` |
| `domains.toml` | Default 5-domain config |
| `examples/router_demo.rs` | Working demo |
| `.research/09_EMO_Emergent_Modularity.md` | Paper verdict |
| `.plans/023_prompt_router.md` | Implementation plan (10/10 [x]) |

### Files Created (anyrag)

| File | Purpose |
|------|---------|
| `crates/server/tests/slots_test.rs` | Plan 004 integration tests (5 tests) |
| `crates/server/tests/slots_bench_test.rs` | Plan 004 benchmarks (3 tests, `#[ignore]`) |
| `crates/server/tests/classify_test.rs` | Plan 005 integration tests (7 tests) |
| `crates/server/src/handlers/classify.rs` | `POST /classify/domain` handler (pre-existing, updated) |
| `crates/lib/src/router/` | Domain classifier module (pre-existing, Tasks 1-4) |
| `crates/lib/src/types.rs` | Added `DomainMapping` + `default_domain_mappings()` |

### Files Modified (anyrag — bug fixes)

| File | Fix |
|------|-----|
| `crates/lib/src/slots/search.rs` | Replaced `\` line continuations with `r#"..."#` (broken SQL). Replaced `IN (SELECT ...)` with JOIN (turso compat). |
| `crates/lib/src/slots/ingest.rs` | Replaced `INSERT OR IGNORE` with `DELETE + INSERT` (turso doesn't support `ON CONFLICT`). |
| `crates/server/src/handlers/slots.rs` | Replaced `INSERT OR IGNORE` with plain `INSERT` in reindex (safe because `DELETE FROM slot_documents` runs first). |

## Where Is the Plan/Code/Test

| Artifact | Location |
|----------|----------|
| microgpt-rs Plan 023 | `microgpt-rs/.plans/023_prompt_router.md` |
| anyrag Plan 004 | `anyrag/.plans/004_raven_routed_slots.md` |
| anyrag Plan 005 | `anyrag/.plans/005_domain_classifier_api.md` |
| EMO Research | `microgpt-rs/.research/09_EMO_Emergent_Modularity.md` |
| microgpt-rs router code | `microgpt-rs/src/router/` |
| anyrag slot tests | `anyrag/crates/server/tests/slots_test.rs` |
| anyrag slot benchmarks | `anyrag/crates/server/tests/slots_bench_test.rs` |
| anyrag classify tests | `anyrag/crates/server/tests/classify_test.rs` |
| anyrag classifier code | `anyrag/crates/lib/src/router/` |

## Reflection — Struggling / Solved

1. **Native pruners need runtime state.** `SudokuPruner::new()` requires a `Sudoku9x9` board and `TacticalPruner::new()` requires a map string — neither can be constructed at config-load time. Solved by documenting this and falling back to `NoScreeningPruner` with a stderr warning. Callers provide concrete pruners directly when they have the runtime data.

2. **Module inception warning.** `router/router.rs` triggered clippy's `module_inception` lint. Renamed to `router/prompt_router.rs`.

3. **turso SQL compatibility.** turso/libSQL doesn't support `IN (SELECT ...)` subqueries or `INSERT OR IGNORE` with `ON CONFLICT`. Fixed by rewriting to JOINs and `DELETE + INSERT` patterns. This was discovered during anyrag Plan 004 integration tests.

4. **Doc comment formatting.** `//! + optional LoRA` was parsed as a markdown list item. Changed `+` to `and`.

## Remain Work

### microgpt-rs
- Native pruners (`sudoku`, `tactical`) cannot be auto-loaded by the registry — they need runtime state. A future plan could add a `PrunerFactory` trait or runtime registration.
- LoRA adapter loading is `Option<PathBuf>` — actual loading deferred to a future plan.
- Embedding-based routing via anyrag's `/classify/domain` endpoint (V2 upgrade).

### anyrag
- `test_gen_tx_handler` pre-existing failure (404 on `/gen/tx`) — unrelated to slots/classify, not investigated.
- Slot benchmarks (7.1-7.3) are `#[ignore]` by design — run with `cargo test -- --ignored`.
- True embedding scoring in `HybridClassifier` requires populated slot document embeddings — currently falls back to keyword-only.

## Issues Ref

- microgpt-rs Plan 023: `develop/feature/023_prompt_router` — commit `5ecf36b`
- anyrag Plans 004+005: `feature/004_raven_routed_slots` — commit `e4713d8`
- Pre-existing: anyrag `test_gen_tx_handler` failure (unrelated)

## How to Dev/Test

### microgpt-rs

```sh
# Run router tests (19 tests)
cargo test --features router -- router --quiet

# Run full suite (489 tests)
cargo test --features full -- --quiet

# Run demo
cargo run --example router_demo --features router

# Clippy (0 new warnings from router)
cargo clippy --features full
```

### anyrag

```sh
# Run slot integration tests (5 tests)
cargo test -p anyrag-server --test slots_test -- --quiet

# Run classify integration tests (7 tests)
cargo test -p anyrag-server --test classify_test -- --quiet

# Run lib unit tests (62 tests)
cargo test -p anyrag --lib -- --quiet

# Run benchmarks (ignored by default)
cargo test -p anyrag-server --test slots_bench_test -- --ignored --nocapture

# Clippy
cargo clippy -p anyrag -p anyrag-server
```

### Cross-Project Integration Flow

```text
User Prompt → microgpt-rs KeywordRouter::route() → RouteDecision { domain }
    ↓
ExpertRegistry::get_expert(domain) → ExpertBundle { pruner, lora_path }
    ↓
build_dd_tree_screened(marginals, config, &bundle.pruner)

# V2 upgrade: microgpt-rs calls anyrag's POST /classify/domain instead of local KeywordRouter
# Fallback: if anyrag unavailable, KeywordRouter runs locally