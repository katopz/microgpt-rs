# Plan 023: Prompt Router + Expert Registry — Batch-Level Domain Routing

**Branch:** `develop/feature/023_prompt_router`
**Depends on:** `wasm` feature (for dynamic WASM pruner loading)
**Research:** `.research/03_Commercial_Open_Source_Strategy_Verdict.md` (Curator Marketplace)

---

## Overview

Add a config-driven prompt router that classifies the user prompt once per request, then selects the appropriate `ConstraintPruner` (native or WASM) + LoRA adapter pair from a registry. Routing is batch-level (once per request), not per-token — zero overhead in the DDTree hot path.

This implements the "Mixture of Experts" pattern without MoE architecture changes. The "experts" are `Box<dyn ScreeningPruner>` + optional LoRA path pairs, loaded from a TOML config. Curators add new experts by uploading `.wasm` + `.bin` file pairs — no code changes, no recompilation.

**Caveat**: The keyword-based router is ~80% accurate for obvious domains. Embedding-based routing (anyrag integration) is a future upgrade. LoRA adapter loading does not exist yet — the registry stores `Option<PathBuf>` and actual loading is deferred to a future plan.

---

## Architecture

```text
                    ┌─────────────────────────┐
                    │   PromptRouter trait     │
                    │  (classify → RouteDecision)│
                    └────────┬────────────────┘
                             │ route(prompt)
                             ▼
┌──────────────────────────────────────────────────────┐
│                  ExpertRegistry                       │
│  HashMap<String, ExpertBundle>                       │
│  HashMap<PathBuf, WasmPruner>  ← compiled cache      │
│                                                       │
│  "sudoku"       → NativePruner(SudokuPruner)         │
│  "pathfinding"  → NativePruner(TacticalPruner)       │
│  "rust_code"    → WasmPruner(syn.wasm) + rust.bin    │
│  "py2rs"        → WasmPruner(syn.wasm) + py2rs.bin   │
│  "general"      → NoPruner                           │
└──────────┬───────────────────────────────────────────┘
           │ get_expert(domain) → &ExpertBundle
           ▼  (locked for entire generation)
   build_dd_tree_screened(marginals, config, &bundle.pruner, ...)
```

### SOLID Principles Applied

- **S (Single Responsibility):** `PromptRouter` only classifies. `ExpertRegistry` only manages bundles. `WasmPrunerCache` only caches compiled WASM.
- **O (Open/Closed):** New domains added via `domains.toml`, no code changes. New routing strategies via `PromptRouter` trait impls. New pruner types via `ScreeningPruner` trait.
- **L (Liskov):** Any `PromptRouter` impl is swappable (keyword → embedding → neural). Any `ScreeningPruner` impl works (native, WASM, NoPruner).
- **I (Interface Segregation):** `PromptRouter` has one method: `fn route(&self, prompt: &str) -> RouteDecision`. `ExpertRegistry` has one method: `fn get_expert(&self, domain: &str) -> Option<&ExpertBundle>`. Thin traits.
- **D (Dependency Inversion):** DDTree depends on `ScreeningPruner` trait. Router depends on `PromptRouter` trait. Neither depends on concrete implementations.

### Key Design Decisions

1. **Config-driven, not enum-driven.** Domains are defined in `domains.toml`. Curators add new domains without recompiling `microgpt-rs`. This is critical for the marketplace.

2. **`ScreeningPruner` over `ConstraintPruner`.** The registry returns `Box<dyn ScreeningPruner>` (from Plan 021), which subsumes `ConstraintPruner`. WASM validators without `relevance` export fall back to binary via `BinaryScreeningPruner` adapter. Future-proof.

3. **WASM compilation cached.** `WasmPrunerCache` wraps a `HashMap<PathBuf, WasmPruner>`. First call compiles (~ms), subsequent calls hit cache. One `WasmPruner` per `.wasm` file, shared across domains that use the same validator.

4. **LoRA as `Option<PathBuf>`.** LoRA loading doesn't exist yet. The registry stores the path; actual loading is a future plan. This lets the config and registry be built now without blocking on LoRA infrastructure.

5. **Feature-gated behind `router`.** Opt-in, backward compatible. No impact on existing code paths when disabled.

---

## Tasks

- [x] **Task 1: Create `src/router/` module with types** (`src/router/types.rs`)
  - New types:
    ```rust
    /// Result of routing a prompt to a domain.
    pub struct RouteDecision {
        pub domain: String,
        pub confidence: f32,
        pub lora_path: Option<PathBuf>,
        pub pruner_path: Option<PathBuf>,
    }

    /// A loadable expert bundle: pruner + optional LoRA adapter.
    pub struct ExpertBundle {
        pub domain: String,
        pub pruner: Box<dyn ScreeningPruner>,
        pub lora_path: Option<PathBuf>,
    }

    /// Domain definition loaded from config.
    #[derive(Deserialize)]
    pub struct DomainConfig {
        pub name: String,
        pub keywords: Vec<String>,
        pub lora: Option<String>,
        pub pruner: Option<String>,
        pub native_pruner: Option<String>,  // "sudoku", "tactical", "no_pruner"
    }
    ```
  - `RouteDecision`: output of routing, no redundant fields (`Option` handles defaults)
  - `ExpertBundle`: what the registry serves up
  - `DomainConfig`: what comes from TOML
  - All types in `types.rs` per project convention

- [x] **Task 2: Define `PromptRouter` trait** (`src/router/prompt_router.rs`)
  ```rust
  /// Classifies a prompt into a domain, returning a routing decision.
  /// Called once per request (batch-level), never in the DDTree hot path.
  pub trait PromptRouter: Send + Sync {
      fn route(&self, prompt: &str) -> RouteDecision;
  }
  ```
  - One method. Thin trait. Easy to implement alternative strategies later.
  - `Send + Sync` for use across threads (REST server context).

- [x] **Task 3: Implement `KeywordRouter`** (`src/router/keyword.rs`)
  ```rust
  pub struct KeywordRouter {
      domains: Vec<DomainConfig>,
  }

  impl KeywordRouter {
      pub fn new(domains: Vec<DomainConfig>) -> Self { ... }
  }

  impl PromptRouter for KeywordRouter {
      fn route(&self, prompt: &str) -> RouteDecision {
          let prompt_lower = prompt.to_lowercase();
          let mut best_domain = "general";
          let mut best_score = 0usize;

          for domain in &self.domains {
              let score = domain.keywords.iter()
                  .filter(|kw| prompt_lower.contains(&kw.to_lowercase()))
                  .count();
              if score > best_score {
                  best_score = score;
                  best_domain = &domain.name;
              }
          }

          // ... build RouteDecision from best_domain
      }
  }
  ```
  - Keyword count scoring (not just boolean match)
  - Falls back to "general" domain if no keywords match
  - Confidence = best_score / total_keywords (simple heuristic)

- [x] **Task 4: Implement `WasmPrunerCache`** (`src/router/wasm_cache.rs`)
  ```rust
  /// Caches compiled WasmPruner instances. Shared across domains that use
  /// the same .wasm file (e.g., "rust_code" and "py2rs" both use syn.wasm).
  pub struct WasmPrunerCache {
      cache: Mutex<HashMap<PathBuf, WasmPruner>>,
      pruner_dir: PathBuf,
  }

  impl WasmPrunerCache {
      pub fn new(pruner_dir: PathBuf) -> Self { ... }

      /// Get or compile a WasmPruner. First call compiles (~ms), subsequent
      /// calls return cached instance. Returns None on load failure.
      pub fn get_or_load(&self, path: &Path) -> Option<Arc<WasmPruner>> { ... }
  }
  ```
  - `Mutex<HashMap>` for thread-safe caching
  - Graceful degradation: load failure returns `None`, caller falls back to `NoPruner`
  - Log warnings on load failures (corrupt .wasm, wrong ABI version)
  - Only compiled when `wasm` feature is also enabled

- [x] **Task 5: Implement `ExpertRegistry`** (`src/router/registry.rs`)
  ```rust
  pub struct ExpertRegistry {
      bundles: HashMap<String, ExpertBundle>,
      wasm_cache: WasmPrunerCache,
      default_bundle: ExpertBundle,
  }

  impl ExpertRegistry {
      pub fn from_config(config: &RouterConfig, pruner_dir: &Path) -> Self { ... }

      /// Get expert bundle for a domain. Falls back to default on miss.
      pub fn get_expert(&self, domain: &str) -> &ExpertBundle { ... }
  }
  ```
  - Loads domains from `RouterConfig`, resolves native vs WASM pruners
  - Native pruners: `"sudoku"` → SudokuPruner (feature-gated), `"tactical"` → TacticalPruner, `"no_pruner"` → NoPruner
  - WASM pruners: resolved via `WasmPrunerCache`
  - Default bundle always has `NoPruner` + no LoRA
  - Shared WASM pruners: "rust_code" and "py2rs" pointing to same .wasm share one cached instance

- [x] **Task 6: Add `router` feature and TOML config** (`Cargo.toml`)
  - New feature: `router = ["wasm"]` (depends on WASM for dynamic pruner loading)
  - NOT in default features (opt-in)
  - Add to `full` feature
  - Default `domains.toml`:
    ```toml
    [[domain]]
    name = "sudoku"
    keywords = ["sudoku", "puzzle", "grid", "9x9", "digit"]
    native_pruner = "sudoku"

    [[domain]]
    name = "pathfinding"
    keywords = ["path", "maze", "bear", "terrain", "tactical", "grid"]
    native_pruner = "tactical"

    [[domain]]
    name = "rust_code"
    keywords = ["rust", "cargo", "axum", "tokio", "trait", "impl", "compile"]
    pruner = "syn_validator.wasm"

    [[domain]]
    name = "py2rs"
    keywords = ["python", "rewrite", "fastapi", "flask", "translate"]
    pruner = "syn_validator.wasm"
    lora = "py2rs_lora.bin"

    [[domain]]
    name = "general"
    keywords = []
    native_pruner = "no_pruner"
    ```

- [x] **Task 7: Module wiring** (`src/router/mod.rs`)
  - `mod.rs` for index only (project convention)
  - Re-export public types
  - Feature-gate the entire module
  - Update `src/lib.rs` to include `#[cfg(feature = "router")] pub mod router;`

- [x] **Task 8: Integration example** (`examples/router_demo.rs`)
  - Demonstrates: load config → build registry → route prompt → get expert → run DDTree
  - Show routing for: "solve this sudoku puzzle", "write Rust code for HTTP server", "find path through maze"
  - Print routing decision + expert selection

- [x] **Task 9: Unit tests**
  - Test: `KeywordRouter` correctly identifies sudoku domain
  - Test: `KeywordRouter` correctly identifies rust_code domain
  - Test: `KeywordRouter` falls back to general for unknown prompts
  - Test: `KeywordRouter` multi-keyword scoring (more matches = higher confidence)
  - Test: `ExpertRegistry` returns default bundle for unknown domain
  - Test: `ExpertRegistry` returns correct bundle for known domain
  - Test: `WasmPrunerCache` compiles and caches .wasm (integration test, feature-gated)
  - Test: `WasmPrunerCache` returns None for missing .wasm
  - Test: `RouteDecision` has no redundant fields (compile-time, by design)
  - Test: TOML config parsing with valid and invalid configs

- [x] **Task 10: Update README**
  - Add "Prompt Router (Plan 023)" section to Architecture
  - Update Feature Flags section with `router`
  - Update Project Structure section

---

## File Change Summary

| File | Change |
|------|--------|
| `microgpt-rs/src/router/types.rs` | New: `RouteDecision`, `ExpertBundle`, `DomainConfig`, `RouterConfig` |
| `microgpt-rs/src/router/prompt_router.rs` | New: `PromptRouter` trait |
| `microgpt-rs/src/router/keyword.rs` | New: `KeywordRouter` implementation |
| `microgpt-rs/src/router/wasm_cache.rs` | New: `WasmPrunerCache` for compiled WASM caching (standalone, not used in registry yet) |
| `microgpt-rs/src/router/registry.rs` | New: `ExpertRegistry` with config-driven domain loading |
| `microgpt-rs/src/router/mod.rs` | New: module index + re-exports |
| `microgpt-rs/src/router/prompt_router.rs` | New: `PromptRouter` trait (renamed from `router.rs` to avoid inception warning) |
| `microgpt-rs/Cargo.toml` | Add `router` feature, update `full` |
| `microgpt-rs/src/lib.rs` | Add `#[cfg(feature = "router")] pub mod router;` |
| `microgpt-rs/domains.toml` | New: default domain config file |
| `microgpt-rs/examples/router_demo.rs` | New: integration example |
| `microgpt-rs/README.md` | Add Prompt Router architecture section |

---

## Cross-Project References

| Project | Plan | Relationship |
|---------|------|-------------|
| `microgpt-rs` | This plan (023) | Core routing infrastructure |
| `anyrag` | `.plans/005_domain_classifier_api.md` | V2 embedding-based router backend (future) |
| `riir-validator-sdk` | N/A | Curators build `.wasm` validators that the registry loads |
| `microgpt-rs` | Plan 021 (ScreeningPruner) | Registry returns `Box<dyn ScreeningPruner>` |
| `microgpt-rs` | Plan 015 (WASM Pipeline) | `WasmPruner` is the loadable pruner unit |

---

## Backward Compatibility

- Feature-gated behind `router` — zero impact when disabled
- All existing code paths unchanged
- `KeywordRouter` is the default impl; swapping requires only a different `PromptRouter` impl
- `ExpertRegistry` falls back to `NoPruner` for unknown domains or load failures
- `domains.toml` is optional — hardcoded defaults if file not found

---

## Out of Scope

- LoRA adapter loading (separate plan after training infrastructure exists)
- Embedding-based routing via anyrag (see anyrag Plan 005)
- Neural router (small classifier model — future research)
- REST endpoint for routing (follows after this plan)
- Multi-domain routing (one prompt → multiple experts — future research)
- Runtime domain registration (Curator uploads during runtime — marketplace phase)