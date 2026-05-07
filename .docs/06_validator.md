# mini-dllm: Constraint Validator

## What is Validator?
Deterministic validation — a neuro-symbolic inference system where `rustc`/`syn` acts as the deterministic referee inside the speculative decoding loop. The LLM drafts token sequences; a rules engine validates them against the Rust AST before target model verification.

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
│                          │   SynPruner (Validator)│              │
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

## Status
| Component | Status | Notes |
|-----------|--------|-------|
| Path encoding (u128) | ❌ Planned | Current u64 5-bit overflows with BPE vocab > 31 |
| BPE Tokenizer | ❌ Planned | Module structure sketched |
| PartialParser | ❌ Planned | Bracket balancer DFA designed |
| SynPruner | ❌ Planned | Two-tier validation designed |
| Training data pipeline | ❌ Deferred to Plan 009 | Depends on tokenizer + validator |

## Path Encoding Redesign

### Current Problem
```rust
// dd_tree.rs — current encoding
parent_path: u64  // 5 bits per depth → max token 31, max depth 12
```
BPE vocab=4096 → token ID 4096 needs 13 bits. Current 5-bit slots overflow immediately.

### Proposed Fix
```rust
// speculative/types.rs — redesigned
parent_path: u128  // 16 bits per depth → max token 65535, max depth 8
```
- Extract: `(path >> (depth * 16)) & 0xFFFF`
- Push: `(path << 16) | token_idx as u128`
- Zero-alloc variant: `extract_parent_tokens_into(path, n, buf)`

### Breaking Change
- All code using `parent_path: u64` → `u128`
- Encoding direction flips: MSB-first → LSB-first
- All roundtrip tests must be rewritten

## BPE Tokenizer (`tokenizer/`) (planned)

### Module Layout
```
src/tokenizer/
├── mod.rs       # Re-exports
├── types.rs     # BpeTokenizer, MergeRule, SpecialTokens
└── bpe.rs       # encode(), decode(), train()
```

### Core Types
```rust
pub struct BpeTokenizer {
    pub vocab_to_id: HashMap<String, usize>,
    pub id_to_vocab: Vec<String>,
    pub merges: Vec<MergeRule>,
    pub bos_id: usize,
    pub eos_id: usize,
    pub pad_id: usize,
}

pub struct MergeRule {
    pub left: usize,
    pub right: usize,
    pub merged: usize,
}
```

### Training Algorithm
1. Read .rs files → byte sequences
2. Initialize vocab = 256 byte-level tokens
3. Count adjacent byte-pair frequencies
4. Merge most frequent pair → new token
5. Repeat until target vocab_size (4096) reached

### Corpus Sources
- `rust-lang/rust` (~30K files, ~15M lines)
- Top 100 crates.io crates (~20K files, ~10M lines)
- Rust docs/examples (~5K files, ~2M lines)

### Expected Common Merges
`fn `, `pub `, `let `, `mut `, `impl `, ` ->`, `::`, `use `, `Result<`, `Option<`, `Vec<`, `String`, `async `, `#[`, `match `, `struct `

### Config
```rust
// types.rs — new constructors
impl Config {
    pub fn bpe() -> Self {
        Self { vocab_size: 4096, block_size: 256, n_embd: 32, n_head: 4,
               head_dim: 8, n_layer: 1, mlp_hidden: 128, ... }
    }
    pub fn bpe_draft() -> Self {
        Self { vocab_size: 4096, n_embd: 8, n_head: 2, head_dim: 4,
               n_layer: 1, mlp_hidden: 32, ... }
    }
}
```
- Total weights: ~1.1 MB (target), ~130 KB (draft)

## PartialParser (`validator/partial_parser.rs`) (planned)

### Design: Bracket Balancer + Keyword Acceptor
NOT a full Rust parser. A pragmatic DFA:

```rust
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
```

- `is_plausible(token_str) -> bool` — incremental state update
- Rejects: negative bracket depth, unfinished strings
- Accepts: everything else (permissive by design)
- Latency: ~50ns per token call

### Honest Assessment
This is "Phase 0 Validator":
- Catches ~10-20% of invalid branches (unbalanced delimiters)
- Near-zero false negatives (rarely prunes valid code)
- The remaining ~80-90% pass to Tier 1 (syn) and Tier 2 (cargo check)

## SynPruner (`validator/syn_pruner.rs`) (planned)

### Two-Tier Validation

| Tier | When | Method | Latency |
|------|------|--------|---------|
| Tier 0: DFA | Per-token in DDTree | PartialParser bracket balance | ~50ns |
| Tier 1: syn | Per-path after DDTree | `syn::parse_str` as various AST nodes | ~1-10μs |
| Tier 2: cargo check | Post-generation, offline | `cargo check` subprocess | ~100ms-1s |

```rust
pub struct SynPruner {
    tokenizer: Arc<BpeTokenizer>,
}

impl ConstraintPruner for SynPruner {
    fn is_valid(&self, depth, token_idx, parent_tokens) -> bool {
        // Tier 0: bracket balance (no allocation)
        let mut parser = PartialParser::new();
        for &tid in parent_tokens {
            if !parser.is_plausible(&self.tokenizer.decode_single(tid)) { return false; }
        }
        parser.is_plausible(&self.tokenizer.decode_single(token_idx))
    }
}
```

Tier 1 validation on completed paths:
```rust
pub fn validate_path(&self, token_ids: &[usize]) -> PruneResult {
    let code = self.tokenizer.decode(token_ids);
    if syn::parse_str::<syn::File>(&code).is_ok() { return PruneResult::Valid; }
    if syn::parse_str::<syn::Item>(&code).is_ok() { return PruneResult::Valid; }
    if syn::parse_str::<syn::Stmt>(&code).is_ok() { return PruneResult::Valid; }
    if syn::parse_str::<syn::Expr>(&code).is_ok() { return PruneResult::Valid; }
    PruneResult::Invalid { reason: "syn parse failed" }
}
```

## CompilerFeedback (`validator/types.rs`) (planned)
```rust
pub struct CompilerFeedback {
    pub error_message: String,
    pub failing_code: String,
    pub suggestion: Option<String>,
}
```
- `extract_suggestion(error)` — pattern-matches common rustc errors (E0382, E0495, E0277)
- `to_context()` — formats as `/* COMPILER ERROR: ... Suggestion: ... */` for next LLM prompt

## Feature Flag
```toml
validator = ["syn", "proc-macro2"]
```
- `syn` is an optional dependency, NOT required for core functionality
- BPE tokenizer works without syn
- Only SynPruner requires syn

## Dependency Chain
```
Phase 0: Path encoding fix (u128) — prerequisite for BPE vocab > 31
Phase 1: BPE Tokenizer — encode/decode/train
Phase 2: SynPruner — PartialParser + ConstraintPruner impl
Phase 3: Integration — examples, benchmarks
```

## Training Data Vision (Plan 009)
```
Rust Docs ──┐
GitHub ────┼──► CorpusIngester ──► TrainingFilter ──► JSONL
Crates.io ─┘    (walk+dedup)       (syn+cargo check)    (training data)
                                                          │
                                                    GPU LoRA Training (Plan 008 wgpu)
                                                          │
                                                    lora.bin
                                                          │
                                                    Draft model upgrade
```

## Key References
- `.research/00_Neuro-Symbolic LLM Architecture.md` — Original Validator concept
- `.research/01_Advanced Neuro-Symbolic Rust Translation.md` — Grand Unification architecture
- `syn` crate — Rust AST parser