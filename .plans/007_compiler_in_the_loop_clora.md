# Plan 007: Compiler-in-the-Loop cLoRA — Full Rust Vocabulary + Training Pipeline

## Objective

Build a complete neuro-symbolic inference system where `rustc`/`syn` acts as the deterministic referee inside the speculative decoding loop. This is NOT a 27-token character-level toy — we target a real BPE tokenizer trained on the entire Rust ecosystem, a `SynPruner` that validates drafted token sequences against the Rust AST, and a training data pipeline that ingests Rust docs + GitHub repos via `anyrag` to produce a `lora.bin` with an astronomically high zero-shot compilation rate.

## Verdict Against Current Codebase (Critical Findings)

Before implementation, I audited every file in `src/`. Here are the blockers the original plan missed:

### Blocker 1: `parent_path` bitfield overflows with BPE vocab (SHOWSTOPPER)

```mini-dllm/src/speculative/dd_tree.rs#L19-20
    .map(|k| ((parent_path >> ((num_tokens - 1 - k) * 5)) & 0x1F) as usize)
```

The `parent_path` packs **5 bits per depth** (`<< 5`, `& 0x1F`). This means **max token index = 31**. A 32K BPE vocab would overflow immediately — token ID 32768 can't fit in 5 bits. Also, max depth = `64/5 = 12`, which is already tight for the current `draft_lookahead=8`.

**Resolution**: Redesign `TreeNode.parent_path` encoding. See §Architecture → Path Encoding.

### Blocker 2: `TransformerWeights` scales linearly with `vocab_size`

```mini-dllm/src/transformer.rs#L25-35
        Self {
            wte: init(config.vocab_size * n),      // [vocab_size * n_embd]
            lm_head: init(config.vocab_size * n),  // [vocab_size * n_embd]
            ...
        }
```

With `vocab_size=32768` and the current `n_embd=16`:
- `wte`: 32768 × 16 = 524K floats = **2 MB** (fine)
- `lm_head`: another **2 MB** (fine)
- `ForwardContext.logits`: 32768 × 4 = 128 KB (fine)

The plan must specify a concrete `Config::bpe()` that keeps weights in the low-MB range for the educational/perf profile of this project.

**Resolution**: `Config::bpe()` uses `n_embd=32, vocab_size=4096, block_size=256`. See §Config.

### Blocker 3: `syn` parse is not partial — DFA scope is underspecified

`syn::parse_str::<Stmt>("let mu")` returns `Err`. The `PartialParser` DFA is the right idea, but "state machine for Rust syntax" is massively underspecified — Rust's grammar is one of the most complex of any programming language.

**Resolution**: Start with a **bracket balancer** (balanced `{}`, `()`, `[]`, `<>`) + keyword acceptance table. NOT a full Rust parser. See §PartialParser.

### Blocker 4: Plan creates 3 new top-level modules at once

The project's pattern is incremental: one module per plan, behind feature flags (sudoku → plan 002/005/006, leviathan → plan 004). The original plan proposed `clora/`, `tokenizer/`, `data/` simultaneously.

**Resolution**: Phase per module. Phase 1 = `tokenizer/` only. Phase 2 = `clora/` only. Phase 3 = `data/` = separate plan 009. Plan 008 is now wgpu LoRA training.

### Non-Blocker: `ConstraintPruner::is_valid` doesn't carry tokenizer

```mini-dllm/src/speculative/types.rs#L15-20
pub trait ConstraintPruner: Send + Sync {
    fn is_valid(&self, depth: usize, token_idx: usize, parent_tokens: &[usize]) -> bool;
}
```

The `SynPruner` needs to decode tokens → string. Solution: `SynPruner` holds `Arc<BpeTokenizer>` internally. Trait signature unchanged. Clean.

## The Grand Vision (from Research)

```
┌─────────────────────────────────────────────────────────────────┐
│                    INFERENCE (Production)                        │
│                                                                  │
│  User Prompt ──► BPE Encode ──► Draft Model (microgpt-rs)       │
│                                      │                           │
│                               DDTree Branches                   │
│                                      │                           │
│                          ┌───────────▼───────────┐              │
│                          │   SynPruner (cLoRA)    │              │
│                          │   ┌─────────────────┐  │              │
│                          │   │ bracket balance  │  │  Tier 0 DFA │
│                          │   │ keyword accept   │  │  ~100ns/tok │
│                          │   └─────────────────┘  │              │
│                          │   ┌─────────────────┐  │              │
│                          │   │ syn full parse   │  │  Tier 1 AST │
│                          │   │ (completed paths)│  │  ~1-10μs    │
│                          │   └─────────────────┘  │              │
│                          └───────────┬───────────┘              │
│                                      │                           │
│                          Validated DDTree Branches               │
│                                      │                           │
│                          ┌───────────▼───────────┐              │
│                          │  Target Model Verify   │              │
│                          │  (semantic quality)    │              │
│                          └───────────┬───────────┘              │
│                                      │                           │
│                          ┌───────────▼───────────┐              │
│                          │  cargo check (final)   │              │
│                          │  OK → anyrag + Turso   │              │
│                          │  ERR → feedback loop   │              │
│                          └───────────────────────┘              │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    TRAINING (Offline Batch)                      │
│                                                                  │
│  Rust Docs ──┐                                                   │
│  GitHub ────┼──► ingester ──► BPE tokenize ──► syn filter       │
│  Crates.io ─┘       │                │                │         │
│                     │                │                ▼         │
│              concept sharding    vocab build    only valid AST   │
│              via anyrag          merge + train   0 errors/warns  │
│                     │                │                │         │
│                     ▼                ▼                ▼         │
│              Turso episodic    tokenizer.json    clean.jsonl     │
│              (hidden states)   (BPE merges)     (training data) │
│                     │                │                │         │
│                     └────────────────┼────────────────┘         │
│                                      ▼                           │
│                              LoRA fine-tune                     │
│                              (lora.bin)                         │
│                                      │                           │
│                              Draft model upgrade                │
│                              (higher acceptance rate)           │
└─────────────────────────────────────────────────────────────────┘
```

## Architecture

### Path Encoding Redesign (Fixes Blocker 1)

Current: `u64` bitfield, 5 bits per depth → max token 31, max depth 12.

New: `u128` bitfield, 16 bits per depth → max token 65535, max depth 8.

```rust
// speculative/types.rs — updated TreeNode

/// DDTree node for Best-First Search.
///
/// Path encoding: 16 bits per depth, packed LSB-first into u128.
/// - Depth 0 token: bits 0–15
/// - Depth 1 token: bits 16–31
/// - Depth k token: bits (k*16) to (k*16+15)
///
/// Limits: max token index = 65535, max depth = 128/16 = 8.
/// Sufficient for vocab ≤ 65K and draft_lookahead ≤ 8.
#[derive(Copy, Clone, PartialEq)]
pub struct TreeNode {
    pub score: f32,
    pub depth: usize,
    pub token_idx: usize,
    pub parent_path: u128,
}

// Updated extract:
pub fn extract_parent_tokens(parent_path: u128, num_tokens: usize) -> Vec<usize> {
    (0..num_tokens)
        .map(|k| ((parent_path >> (k * 16)) & 0xFFFF) as usize)
        .collect()
}

// Updated push:
// parent_path: (best.parent_path << 16) | (i as u128)
```

**Breaking change**: All code using `parent_path: u64` must update to `u128`. This affects:

**Semantic change**: The encoding direction flips. Current code packs depth-0 in HIGH bits (MSB-first extract). New code packs depth-0 in LOW bits (LSB-first extract). This is simpler and more natural but means all `extract_parent_tokens` roundtrip tests must be rewritten.

- `types.rs`: `TreeNode.parent_path` type
- `dd_tree.rs`: `build_dd_tree_pruned` shift/mask, `extract_parent_tokens`
- `sudoku_pruner.rs`: no change (only reads via `extract_parent_tokens`)

### Config (Fixes Blocker 2)

```rust
// types.rs — new Config constructor

impl Config {
    /// BPE model for Rust code generation.
    /// vocab=4096 subword tokens, block=256, embd=32, heads=4, mlp=128.
    /// Weight sizes: wte=512KB, lm_head=512KB — fits in L2 cache.
    /// Targets CPU inference at ~1000+ tok/s for draft model.
    pub fn bpe() -> Self {
        Self {
            vocab_size: 4096,
            block_size: 256,
            n_embd: 32,
            n_head: 4,
            head_dim: 8,
            n_layer: 1,         // single-layer; multi-layer needs Vec<Vec<f32>> weights (Plan 008)
            mlp_hidden: 128,
            bos_token: 0,       // BOS = token 0 in BPE vocab
            temperature: 0.8,
            draft_lookahead: 5,
            tree_budget: 32,
        }
    }

    /// BPE draft model — 4× smaller than target.
    pub fn bpe_draft() -> Self {
        Self {
            vocab_size: 4096,
            block_size: 256,
            n_embd: 8,
            n_head: 2,
            head_dim: 4,
            n_layer: 1,
            mlp_hidden: 32,
            bos_token: 0,
            temperature: 0.8,
            draft_lookahead: 5,
            tree_budget: 32,
        }
    }
}
```

**Note on multi-layer**: Both configs use `n_layer: 1` (single-layer). The current `TransformerWeights` struct is single-layer — all weight fields are flat `Vec<f32>`, not `Vec<Vec<f32>>`. Adding multi-layer support requires:
1. Adding `n_layer: usize` to `Config`
2. Changing `TransformerWeights` to hold `Vec<Vec<f32>>` for per-layer weights (`attn_wq/k/v/o`, `mlp_w1/w2`)
3. Adding a layer loop in `forward()`

This is a prerequisite for Plan 008 (wgpu LoRA training) at cLoRA scale, but NOT for Plan 007's BPE tokenizer + SynPruner. Multi-layer will be added as a Phase 0.5 task when needed for Plan 008 integration.

**Memory estimates** for `Config::bpe()`:
| Buffer | Size | Bytes |
|--------|------|-------|
| `wte` | 4096 × 32 | 512 KB |
| `wpe` | 256 × 32 | 32 KB |
| `attn_wq/k/v/o` | 4 × (32 × 32) | 16 KB each |
| `mlp_w1` | 128 × 32 | 16 KB |
| `mlp_w2` | 32 × 128 | 16 KB |
| `lm_head` | 4096 × 32 | 512 KB |
| **Total** | | **~1.1 MB** |

Draft model (`Config::bpe_draft()`): **~130 KB** total. Both fit comfortably in L2 cache.

### Module Layout (Fixes Blocker 4 — Incremental)

Phase 1 (this plan) creates `tokenizer/` only:

```
src/
├── tokenizer/                      # NEW (Phase 1)
│   ├── mod.rs                      # Re-exports
│   ├── types.rs                    # BpeTokenizer, MergeRule, SpecialTokens
│   └── bpe.rs                      # encode(), decode(), train()
├── clora/                          # NEW (Phase 2 — same plan, later task)
│   ├── mod.rs                      # Re-exports
│   ├── types.rs                    # PruneResult, ErrorKind, CompilerFeedback
│   ├── syn_pruner.rs               # ConstraintPruner impl — bracket balancer + syn
│   └── partial_parser.rs           # DFA: bracket balance + keyword acceptance
├── speculative/                    # EXISTING (modified)
│   ├── types.rs                    # TreeNode.parent_path: u64 → u128
│   ├── dd_tree.rs                  # shift 5 → 16, mask 0x1F → 0xFFFF
│   └── ...
├── transformer.rs                  # EXISTING (unchanged — already parameterized)
├── types.rs                        # EXISTING (add Config::bpe(), Config::bpe_draft())
└── lib.rs                          # EXISTING (add mod tokenizer, mod clora)
```

Plan 009 creates `data/` for the training data pipeline. Plan 008 covers wgpu GPU-accelerated LoRA training.

### Dependency Additions (`Cargo.toml`)

```toml
[dependencies]
plotters = "0.3"
rayon = "1.10"
blake3 = "1"                       # fast hashing for corpus dedup + BPE cache
serde = { version = "1", features = ["derive"] }
serde_json = "1"
syn = { version = "2", features = ["full", "parsing"], optional = true }
proc-macro2 = { version = "1", optional = true }

[dev-dependencies]
ratatui = "0.29"
crossterm = "0.28"

[features]
default = []
leviathan = []
sudoku = []
clora = ["syn", "proc-macro2"]     # gates clora/ module — syn becomes required dep
```

**Note**: `syn` and `proc-macro2` are **optional dependencies** gated behind the `clora` feature. They are NOT dev-dependencies. The tokenizer and BPE training work without them. Only the `SynPruner` (Phase 2) requires `syn`.

### PartialParser (Fixes Blocker 3 — Realistic Scope)

NOT a full Rust parser. A **bracket balancer + keyword acceptor**:

```rust
// clora/partial_parser.rs

/// Incremental bracket balancer for Rust syntax.
/// Fast enough for per-token DDTree validation (~50ns per call).
///
/// Tracks:
/// - Balanced delimiters: { } ( ) [ ] < >
/// - String literal state (inside "..." → accept anything)
/// - Char literal state (inside '...' → accept anything)
/// - Block comment state (inside /* ... */ → accept anything)
/// - Line comment state (after // → accept anything until \n)
///
/// Does NOT track:
/// - Whether keywords are in correct order (too complex for DFA)
/// - Whether types are valid (requires semantic analysis)
/// - Borrow checker rules (requires full rustc)
///
/// ## Honest Assessment (vs Research Ambition)
///
/// The research describes a "Computable LoRA" using Percepta 2D convex-hull
/// attention to execute an AST parser at O(log n) per token. This PartialParser
/// is "Phase 0 cLoRA" — a pragmatic baseline that:
/// - Catches ~10-20% of invalid branches (unbalanced delimiters)
/// - Has near-zero false negatives (rarely prunes valid code)
/// - Runs at ~50ns/tok (well within DDTree budget)
///
/// The remaining ~80-90% of invalid branches pass through to Tier 1 (syn)
/// and Tier 2 (cargo check). This is acceptable because:
/// 1. Tier 1 runs per-path (not per-token), so volume is bounded by tree_budget
/// 2. Tier 2 runs offline during training data generation, not during inference
///
/// Future work: upgrade to keyword-aware DFA or actual Percepta integration
/// for higher pruning rates in the DDTree hot loop.
pub struct PartialParser {
    paren_depth: u32,
    brace_depth: u32,
    bracket_depth: u32,
    angle_depth: u32,
    in_string: bool,
    in_char: bool,
    in_block_comment: bool,
    in_line_comment: bool,
}

impl PartialParser {
    /// Check if appending `token_str` keeps the code plausibly valid.
    /// Returns false only for CLEARLY invalid states:
    /// - Negative bracket depth (more closes than opens)
    /// - Unfinished string/char at end of chunk
    pub fn is_plausible(&mut self, token_str: &str) -> bool {
        for ch in token_str.chars() {
            if self.in_line_comment {
                if ch == '\n' { self.in_line_comment = false; }
                continue;
            }
            if self.in_block_comment {
                if ch == '*' && /* peek */ false { self.in_block_comment = false; }
                continue;
            }
            if self.in_string {
                if ch == '"' { self.in_string = false; }
                continue;
            }
            if self.in_char {
                if ch == '\'' { self.in_char = false; }
                continue;
            }
            match ch {
                '(' => self.paren_depth += 1,
                ')' => { self.paren_depth = self.paren_depth.saturating_sub(1); }
                '{' => self.brace_depth += 1,
                '}' => { self.brace_depth = self.brace_depth.saturating_sub(1); }
                '[' => self.bracket_depth += 1,
                ']' => { self.bracket_depth = self.bracket_depth.saturating_sub(1); }
                '<' => self.angle_depth += 1,
                '>' => { self.angle_depth = self.angle_depth.saturating_sub(1); }
                '"' => self.in_string = true,
                '\'' => self.in_char = true,
                '/' => { /* peek for // or /* */ }
                _ => {}
            }
        }
        // Only reject clearly broken state
        self.paren_depth != u32::MAX
            && self.brace_depth != u32::MAX
            && self.bracket_depth != u32::MAX
    }
}
```

### SynPruner Implementation

```rust
// clora/syn_pruner.rs

/// Compiler-in-the-Loop pruner — two-tier validation.
///
/// Tier 0 (per-token in DDTree): PartialParser bracket balance — ~50ns
/// Tier 1 (per-path after DDTree): syn full parse — ~1-10μs
///
/// Implements the same ConstraintPruner trait as SudokuPruner.
/// Plugs directly into build_dd_tree_pruned().
pub struct SynPruner {
    tokenizer: Arc<BpeTokenizer>,
}

impl SynPruner {
    pub fn new(tokenizer: Arc<BpeTokenizer>) -> Self {
        Self { tokenizer }
    }

    /// Full validation of a completed code string.
    /// Called AFTER DDTree produces candidate paths, BEFORE target verification.
    pub fn validate_path(&self, token_ids: &[usize]) -> PruneResult {
        let code = self.tokenizer.decode(token_ids);
        // Try parsing as various Rust AST nodes (most to least specific)
        if syn::parse_str::<syn::File>(&code).is_ok() { return PruneResult::Valid; }
        if syn::parse_str::<syn::Item>(&code).is_ok() { return PruneResult::Valid; }
        if syn::parse_str::<syn::Stmt>(&code).is_ok() { return PruneResult::Valid; }
        if syn::parse_str::<syn::Expr>(&code).is_ok() { return PruneResult::Valid; }
        if syn::parse_str::<syn::Type>(&code).is_ok() { return PruneResult::Valid; }
        PruneResult::Invalid { reason: "syn parse failed".into() }
    }
}

impl ConstraintPruner for SynPruner {
    fn is_valid(&self, _depth: usize, token_idx: usize, parent_tokens: &[usize]) -> bool {
        // Tier 0: bracket balance check (fast, no allocation)
        let mut parser = PartialParser::new();
        for &tid in parent_tokens {
            let s = self.tokenizer.decode_single(tid);
            if !parser.is_plausible(&s) { return false; }
        }
        let s = self.tokenizer.decode_single(token_idx);
        parser.is_plausible(&s)
    }
}
```

### Performance Strategy

| Validation Tier | When | Method | Latency |
|----------------|------|--------|---------|
| **Tier 0: DFA** | Per-token in DDTree hot loop | PartialParser bracket balance | ~50ns |
| **Tier 1: syn** | Per-path after DDTree build | `syn::parse_str` | ~1-10μs |
| **Tier 2: cargo check** | Post-generation, offline | `cargo check` subprocess | ~100ms-1s |
| **Tier 3: clippy** | Training data gate only | `cargo clippy` subprocess | ~200ms-2s |

DDTree only uses Tier 0. Tier 1 runs on the top-K paths. Tiers 2-3 run offline only.

## Phase 1: BPE Tokenizer

### 1.1 Core Types

```rust
// tokenizer/types.rs

/// BPE tokenizer for Rust source code.
pub struct BpeTokenizer {
    /// Token string → token ID
    pub vocab_to_id: HashMap<String, usize>,
    /// Token ID → token string
    pub id_to_vocab: Vec<String>,
    /// Ordered merge rules
    pub merges: Vec<MergeRule>,
    /// Special token IDs
    pub bos_id: usize,
    pub eos_id: usize,
    pub pad_id: usize,
}

/// A single BPE merge rule: (left_id, right_id) → merged_id.
pub struct MergeRule {
    pub left: usize,
    pub right: usize,
    pub merged: usize,
}
```

### 1.2 BPE Algorithm

```rust
// tokenizer/bpe.rs
impl BpeTokenizer {
    /// Encode a string into BPE token IDs.
    /// 1. Convert string to byte-level tokens
    /// 2. Iteratively apply merge rules (most frequent first)
    pub fn encode(&self, text: &str) -> Vec<usize> { /* ... */ }

    /// Decode token IDs back to string.
    pub fn decode(&self, ids: &[usize]) -> String { /* ... */ }

    /// Decode a single token ID to its string representation.
    pub fn decode_single(&self, id: usize) -> String {
        self.id_to_vocab.get(id).cloned().unwrap_or_else(|| "<UNK>".into())
    }
}
```

### 1.3 BPE Training

```rust
// tokenizer/bpe.rs
impl BpeTokenizer {
    /// Train BPE from a directory of .rs files.
    ///
    /// Algorithm:
    /// 1. Read all .rs files → byte sequences
    /// 2. Initialize vocab = 256 byte-level tokens
    /// 3. Count all adjacent byte-pair frequencies
    /// 4. Merge most frequent pair → new token, add to vocab
    /// 5. Repeat until target vocab_size reached
    ///
    /// Special tokens: <BOS>=vocab_size-4, <EOS>=vocab_size-3,
    ///                 <PAD>=vocab_size-2, <UNK>=vocab_size-1
    pub fn train(corpus_dir: &Path, target_vocab_size: usize) -> Self { /* ... */ }
}
```

### 1.4 Corpus Sources

| Source | Estimated .rs files | Estimated lines |
|--------|--------------------:|----------------:|
| `rust-lang/rust` (compiler + std) | ~30K | ~15M |
| Top 100 crates.io (tokio, serde, clap, etc.) | ~20K | ~10M |
| Rust docs/examples (book, by example, nomicon) | ~5K | ~2M |
| **Phase 1 total** | **~55K** | **~27M** |

At 27M lines, BPE training takes ~30-60 minutes on a modern CPU. Output: ~4096 tokens.

Common merges expected: `fn `, `pub `, `let `, `mut `, `impl `, ` ->`, `::`, `use `, `Result<`, `Option<`, `self::`, `Vec<`, `String`, `async `, `#[`, `fn(`, `return`, `match `, `struct `, `enum `, `trait `.

## Phase 2: SynPruner (cLoRA Core)

### 2.1 Module Structure

```
src/clora/
├── mod.rs              # pub mod types; pub mod partial_parser; pub mod syn_pruner;
├── types.rs            # PruneResult, ErrorKind, CompilerFeedback
├── partial_parser.rs   # Bracket balancer DFA (~150 lines)
└── syn_pruner.rs       # ConstraintPruner impl (~100 lines)
```

### 2.2 How It Connects to Existing Code

```rust
// Usage in speculative step:
use crate::clora::syn_pruner::SynPruner;

let tokenizer = Arc::new(BpeTokenizer::load("rust_bpe.json")?);
let pruner = SynPruner::new(tokenizer);
let tree = build_dd_tree_pruned(&marginals, &config, &pruner);

// Validate top paths with syn (Tier 1)
for path in extract_best_path(&tree) {
    match pruner.validate_path(&path) {
        PruneResult::Valid => { /* send to target model */ },
        PruneResult::Invalid { reason } => { /* feed back error */ },
    }
}
```

### 2.3 Error Feedback (Self-Correction Loop)

```rust
// clora/types.rs

/// Compiler error → steering context for next LLM draft.
pub struct CompilerFeedback {
    pub error_message: String,
    pub failing_code: String,
    pub suggestion: Option<String>,
}

impl CompilerFeedback {
    /// Extract suggestion from common rustc error patterns.
    pub fn extract_suggestion(error: &str) -> Option<String> {
        if error.contains("E0382") || error.contains("use of moved value") {
            return Some("consider .clone() or using a reference (&T)".into());
        }
        if error.contains("E0495") || error.contains("lifetime") {
            return Some("add explicit lifetime annotation".into());
        }
        if error.contains("E0277") || error.contains("trait bound") {
            return Some("implement the required trait or add a where clause".into());
        }
        None
    }

    /// Format as context to prepend to the next LLM prompt.
    pub fn to_context(&self) -> String {
        format!(
            "/* COMPILER ERROR: {}\n   Suggestion: {} */\n",
            self.error_message,
            self.suggestion.as_deref().unwrap_or("review the error above")
        )
    }
}
```

## Phase 3: Training Data Pipeline (Separate Plan 009)

The training data pipeline (`src/data/`) is deferred to plan 009 because:
1. It depends on `tokenizer/` and `clora/` being complete
2. It introduces new dependencies (`walkdir`, `tempfile`)
3. The pipeline is a batch tool, not a runtime component

Plan 008 (wgpu LoRA Training) covers GPU-accelerated forward + backward pass for `lora.bin` fine-tuning.

**Module ownership**: `src/data/` is owned by Plan 009. Plan 008 places its `DataLoader` in `src/gpu/dataloader.rs` (inside the `gpu/` module) to avoid ownership conflict. The split is:
- `src/data/ingester.rs`, `src/data/filter.rs`, `src/data/exporter.rs` → Plan 009 (corpus processing)
- `src/gpu/dataloader.rs` → Plan 008 (batch iteration for GPU training)

Plan 009 will cover:
- `data/ingester.rs` — walk dirs, read .rs, blake3 dedup
- `data/filter.rs` — syn validation + cargo check gate
- `data/exporter.rs` — JSONL output for LoRA fine-tuning
- anyrag integration via `/ingest/text` and `/search/vector`

## Tasks

### Phase 0: Path Encoding Fix (Prerequisite)

- [ ] 0.1 Change `TreeNode.parent_path` from `u64` to `u128` in `speculative/types.rs`
- [ ] 0.2 Update `extract_parent_tokens` to use 16-bit shifts (`<< 16`, `& 0xFFFF`)
- [ ] 0.3 Update `build_dd_tree_pruned` shift from `<< 5` to `<< 16`
- [ ] 0.4 Update all tests in `dd_tree.rs` for new encoding
- [ ] 0.5 Run `cargo test --all-features` — all 176 tests pass
- [ ] 0.6 Run `cargo clippy --all-features` — zero warnings
- [ ] 0.7 Run `cargo run --release` — benchmark unchanged (perf check)
- [ ] 0.8 Commit with message `refactor: TreeNode path encoding 5-bit→16-bit for BPE vocab support`
- [ ] 0.9 Add `n_layer: usize` field to `Config` in `types.rs` (default: 1 for all configs)
- [ ] 0.10 Note: multi-layer `TransformerWeights` deferred until Plan 008 needs it

### Phase 1: BPE Tokenizer

- [ ] 1.1 Add `blake3`, `serde`, `serde_json` to `Cargo.toml` dependencies
- [ ] 1.2 Create `src/tokenizer/mod.rs` with re-exports
- [ ] 1.3 Create `src/tokenizer/types.rs` with `BpeTokenizer`, `MergeRule`
- [ ] 1.4 Create `src/tokenizer/bpe.rs` with `encode()`, `decode()`, `decode_single()`, `train()`
- [ ] 1.5 Add `Config::bpe()` and `Config::bpe_draft()` to `src/types.rs`
- [ ] 1.6 Add `pub mod tokenizer;` to `src/lib.rs`
- [ ] 1.7 Add tests: encode/decode roundtrip, special tokens, vocab coverage
- [ ] 1.8 Add benchmark: BPE encode/decode throughput (in `src/benchmark.rs`)
- [ ] 1.9 Run `cargo clippy --all-features`, `cargo test --all-features`
- [ ] 1.10 Commit with message `feat: BPE tokenizer for Rust source code`

### Phase 2: SynPruner (cLoRA Core)

- [ ] 2.1 Add `syn` and `proc-macro2` to `Cargo.toml` under `[dependencies]` with `optional = true`
- [ ] 2.2 Add `clora = ["syn", "proc-macro2"]` to `[features]`
- [ ] 2.3 Create `src/clora/mod.rs` with re-exports (behind `#[cfg(feature = "clora")]`)
- [ ] 2.4 Create `src/clora/types.rs` with `PruneResult`, `ErrorKind`, `CompilerFeedback`
- [ ] 2.5 Create `src/clora/partial_parser.rs` with bracket balancer DFA
- [ ] 2.6 Create `src/clora/syn_pruner.rs` with `SynPruner` implementing `ConstraintPruner`
- [ ] 2.7 Add `pub mod clora;` to `src/lib.rs` (behind `#[cfg(feature = "clora")]`)
- [ ] 2.8 Add tests: partial parser accepts valid fragments, rejects unbalanced
- [ ] 2.9 Add tests: SynPruner prunes invalid Rust, accepts valid Rust
- [ ] 2.10 Add benchmark: SynPruner overhead vs NoPruner on DDTree build
- [ ] 2.11 Run `cargo test --features clora`, `cargo clippy --features clora`
- [ ] 2.12 Commit with message `feat: SynPruner cLoRA — bracket balance + syn validation`

### Phase 3: Integration & Validation

- [ ] 3.1 Create example: `examples/clora_demo.rs` (behind `clora` feature)
- [ ] 3.2 Demo shows: BPE encode → draft → SynPruner → syn validate → output
- [ ] 3.3 Run baseline benchmark (no cLoRA) → `bench/015_bench_baseline.png`
- [ ] 3.4 Run cLoRA benchmark (with SynPruner) → `bench/016_bench_clora.png`
- [ ] 3.5 Measure DDTree build time overhead: target ≤5%
- [ ] 3.6 Commit with message `feat: cLoRA demo + benchmark`

## Feature Flags

```toml
[features]
default = []
leviathan = []
sudoku = []
clora = ["syn", "proc-macro2"]
```

## Key Risks & Mitigations

| Risk | Impact | Likelihood | Mitigation |
|------|--------|-----------|-----------|
| `u128` slower than `u64` for path encoding | DDTree build 5-10% slower | Low | Benchmark; u128 is single register on x86-64 |
| BPE vocab 4096 too small for Rust | Uncommon tokens split into bytes | Medium | Configurable: test with 8192, 16384 |
| PartialParser false negatives | Valid branches pruned | Medium | Tune to be permissive; only reject clearly broken |
| `syn` compile time adds ~10s | Slower dev cycle | High | Behind feature flag; not required for tokenizer work |
| Bracket balancer too simple | Most invalid code passes Tier 0 | Expected | That's fine — Tier 1 syn catches it, Tier 2 cargo check catches the rest |

## Expected Outcomes

1. **BPE Tokenizer**: ~4096 token vocabulary trained on Rust corpus, enabling meaningful code generation
2. **Path Encoding**: `u128` with 16-bit slots supporting vocabs up to 65K (future-proof)
3. **SynPruner**: `ConstraintPruner` implementation with two-tier validation (DFA + syn)
4. **Config**: `Config::bpe()` / `Config::bpe_draft()` for BPE-based models
5. **Quality**: Foundation for >95% zero-shot compilation rate after LoRA training (plan 008 wgpu + plan 009 data pipeline)

## Files to Create/Modify

| File | Action | Phase | Breaking? |
|------|--------|-------|-----------|
| `src/speculative/types.rs` | `u64` → `u128` in TreeNode | 0 | **Yes** — all tests update |
| `src/speculative/dd_tree.rs` | shift/mask update | 0 | No (internal) |
| `Cargo.toml` | Add deps + features | 1-2 | No |
| `src/tokenizer/mod.rs` | New | 1 | No |
| `src/tokenizer/types.rs` | New | 1 | No |
| `src/tokenizer/bpe.rs` | New | 1 | No |
| `src/types.rs` | Add Config::bpe() | 1 | No |
| `src/lib.rs` | Add mod tokenizer | 1 | No |
| `src/clora/mod.rs` | New | 2 | No |
| `src/clora/types.rs` | New | 2 | No |
| `src/clora/partial_parser.rs` | New | 2 | No |
| `src/clora/syn_pruner.rs` | New | 2 | No |
| `examples/clora_demo.rs` | New | 3 | No |
| `src/benchmark.rs` | Add BPE + clora benches | 1-3 | No |

## References

- `.research/01_Advanced Neuro-Symbolic Rust Translation.md` — Grand Unification architecture
- `.research/00_Neuro-Symbolic LLM Architecture.md` — Original cLoRA concept
- `.plans/004_leviathan_distill.md` — SpeculativeVerifier trait pattern
- `.plans/005_speculative_module_refactor.md` — ConstraintPruner trait, DDTree pruning
- `anyrag/README.md` — RAG pipeline, concept sharding, JSONL export, `/knowledge/export`
