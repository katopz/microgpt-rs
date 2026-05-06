# Plan 007: Compiler-in-the-Loop cLoRA — Full Rust Vocabulary + Training Pipeline

## Objective

Build a complete neuro-symbolic inference system where `rustc`/`syn` acts as the deterministic referee inside the speculative decoding loop. This is NOT a 27-token character-level toy — we target a real BPE tokenizer trained on the entire Rust ecosystem, a `SynPruner` that validates drafted token sequences against the Rust AST, and a training data pipeline that ingests Rust docs + GitHub repos via `anyrag` to produce a `lora.bin` with an astronomically high zero-shot compilation rate.

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
│                          │   │ syn AST parse    │  │              │
│                          │   │ partial tokenize │  │              │
│                          │   │ borrow-check DFA │  │              │
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
│  GitHub ────┼──► anyrag ingest ──► BPE tokenize ──► syn filter  │
│  Crates.io ─┘       │                │                │         │
│                     │                │                ▼         │
│              concept sharding    vocab build    only valid AST   │
│              embeddings store    merge + train   0 errors/warns  │
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

## Why Not 27 Tokens

The current `Config::micro()` uses `vocab_size=27` (a-z + bos). This is fine for proving the DDTree/pruning architecture works. But for real Rust code generation, we need:

| Aspect | Current (27-token) | Target (BPE) |
|--------|-------------------|--------------|
| Vocabulary | a-z characters | ~32K BPE tokens from Rust corpus |
| Token meaning | single letter | subword: `fn `, `mut `, `::`, `Result<`, `impl` |
| Block size | 16 tokens | 256-1024 tokens |
| What model sees | `"f" "n" " "` | `"fn "` (one token) |
| SynPruner input | useless single char | meaningful subword sequences |
| Training data | random chars | actual Rust source code |

**The key insight**: BPE tokenization compresses common Rust patterns (`pub fn `, `let mut `, `impl `, `Result<`, `Option<`, `async fn `, `#[derive(`) into single tokens. This means the draft model proposes *syntactically meaningful chunks*, and `syn` can validate *partial sequences* at subword boundaries.

## Architecture Overview

### New Modules

```
src/
├── clora/                          # NEW: Compiler-in-the-Loop cLoRA
│   ├── mod.rs                      # Re-exports
│   ├── types.rs                    # SynPruner, PruneResult, CompilerFeedback
│   ├── syn_pruner.rs               # ConstraintPruner impl using syn
│   ├── partial_parser.rs           # Incremental/partial AST validation
│   ├── error_feedback.rs           # rustc error → steering context
│   └── training_filter.rs          # cargo check gate for training data
├── tokenizer/                      # NEW: BPE tokenizer for Rust
│   ├── mod.rs                      # Re-exports
│   ├── types.rs                    # Tokenizer struct, Vocab, Merge
│   ├── bpe.rs                      # BPE encode/decode algorithm
│   ├── trainer.rs                  # Train BPE from Rust corpus
│   └── rust_vocab.rs               # Pre-trained vocab + special tokens
├── data/                           # NEW: Training data pipeline
│   ├── mod.rs                      # Re-exports
│   ├── types.rs                    # CorpusEntry, TrainingSample, QualityReport
│   ├── ingester.rs                 # Walk Rust repos/docs, extract .rs files
│   ├── filter.rs                   # cargo check + syn validation gate
│   └── exporter.rs                 # JSONL export for LoRA fine-tuning
├── speculative/                    # EXISTING (extended)
│   ├── ...
│   └── syn_pruner.rs              # NEW: implements ConstraintPruner via clora
├── transformer.rs                  # EXISTING (extended for BPE vocab)
├── types.rs                        # EXISTING (Config gains tokenizer fields)
└── lib.rs                          # EXISTING (add mod clora, tokenizer, data)
```

### Dependency Additions (`Cargo.toml`)

```toml
[dependencies]
plotters = "0.3"
rayon = "1.10"
syn = { version = "2", features = ["full", "parsing", "extra-traits"] }
proc-macro2 = "1"
blake3 = "1"           # fast hashing for corpus dedup
serde = { version = "1", features = ["derive"] }
serde_json = "1"        # JSONL export
walkdir = "2"           # recursive dir walking for corpus
rayon = "1.10"          # parallel corpus processing (already present)

[features]
default = []
leviathan = []
sudoku = []
clora = ["syn"]         # gate syn-dependent code
```

## Phase 1: BPE Tokenizer for Rust

### 1.1 Tokenizer Core

Train a Byte-Pair Encoding tokenizer on a Rust corpus. The tokenizer:

- Starts from byte-level (256 base tokens)
- Learns merges from Rust source code
- Produces ~32K tokens optimized for Rust syntax
- Includes special tokens: `<BOS>`, `<EOS>`, `<PAD>`, `<MASK>`

```rust
// tokenizer/types.rs
pub struct BpeTokenizer {
    /// Token → string mapping
    pub vocab: Vec<String>,
    /// String → token ID mapping
    pub token_to_id: HashMap<String, usize>,
    /// Ordered merge rules (pair → new token)
    pub merges: Vec<MergeRule>,
    /// Special token IDs
    pub bos_id: usize,
    pub eos_id: usize,
    pub pad_id: usize,
    pub mask_id: usize,
}

pub struct MergeRule {
    pub left: usize,
    pub right: usize,
    pub result: usize,
}
```

### 1.2 Training the BPE

```rust
// tokenizer/trainer.rs
impl BpeTrainer {
    /// Train BPE from a directory of .rs files.
    /// 1. Read all .rs files → byte sequences
    /// 2. Count byte-pair frequencies
    /// 3. Merge most frequent pair → new token
    /// 4. Repeat until vocab_size reached
    pub fn train(corpus_dir: &Path, vocab_size: usize, num_merges: usize) -> BpeTokenizer {
        // ...
    }
}
```

**Corpus sources** (ranked by quality):
1. `rust-lang/rust` — compiler + std library (canonical Rust)
2. `tokio-rs/tokio` — async runtime patterns
3. `serde-rs/serde` — derive macro patterns
4. `hyperium/hyper` — HTTP/networking patterns
5. `bevyengine/bevy` — ECS/game patterns
6. All crates in `crates.io` top-1000 by downloads
7. The Rust Reference (`rust-lang/reference`)
8. Rust by Example (`rust-lang/rust-by-example`)
9. The Rustonomicon (`rust-lang/nomicon`)

### 1.3 Pre-trained Vocabulary

Ship a pre-trained `rust_vocab.rs` so users don't need to run BPE training:

```rust
// tokenizer/rust_vocab.rs
/// Pre-trained BPE vocabulary from ~50M lines of Rust source.
/// 32,768 tokens covering Rust syntax patterns.
pub const RUST_VOCAB: &[&str] = &[
    // Byte-level (0-255)
    "\0", "\x01", /* ... */, "\xff",
    // Common Rust merges (256+)
    " ", "  ", "\n", "fn ", "pub ", "let ", "mut ",
    "impl ", "struct ", "enum ", "trait ", "type ",
    "use ", "mod ", "crate", "self", "super",
    "Result<", "Option<", "Vec<", "Box<", "Arc<",
    "String", "&str", "bool", "usize", "i32", "u8",
    "async ", "await", "fn(", " -> ", "impl<",
    "#[derive(", "#[cfg(", "#[test]", "macro_rules!",
    "::", "::std", "std::", "core::", "alloc::",
    "unsafe ", "extern ", "where ", "for<",
    // ... thousands more from BPE training
];
```

### 1.4 Config Update

```rust
// types.rs — extended Config
pub struct Config {
    // ... existing fields ...
    
    // Tokenizer fields
    pub vocab_size: usize,        // was 27, now 32768
    pub tokenizer_vocab: Option<Vec<String>>,  // BPE vocab
    
    // cLoRA fields
    pub syn_prune_enabled: bool,  // enable syn-based pruning
    pub cargo_check_enabled: bool, // enable cargo check gate
}
```

## Phase 2: SynPruner (cLoRA for DDTree)

### 2.1 The Core Problem: Partial Validation

The `ConstraintPruner` trait operates at token-level during DDTree construction:

```rust
pub trait ConstraintPruner: Send + Sync {
    fn is_valid(&self, depth: usize, token_idx: usize, parent_tokens: &[usize]) -> bool;
}
```

With BPE tokens, each token is a subword like `fn `, `mut `, `Result<`. We can:

1. **Decode parent_tokens → string** using the BPE tokenizer
2. **Append the candidate token** to get the partial code string
3. **Validate incrementally** using a partial parser

### 2.2 Partial Parsing Strategy

`syn` requires complete syntax. For partial validation during DDTree:

```rust
// clora/partial_parser.rs

/// Incremental Rust syntax validator.
/// Maintains a state machine for partial Rust code.
/// Much faster than full syn parse for per-token validation.
pub struct PartialParser {
    /// Accumulated code buffer (decoded from BPE tokens)
    buffer: String,
    /// Current parser state (what we expect next)
    state: ParseState,
    /// Brace/bracket/paren depth tracking
    depth: DepthTracker,
}

#[derive(Clone, Debug)]
pub enum ParseState {
    /// Start of file — expect item or attribute
    TopLevel,
    /// After `fn` keyword — expect identifier or generics
    FnSignature,
    /// Inside function body — expect statement or expression
    FnBody,
    /// After `let` — expect pattern
    LetBinding,
    /// After `impl` — expect trait or type
    ImplBlock,
    /// Inside type annotation — expect type tokens
    TypeAnnotation,
    /// Inside expression — expect operands/operators
    Expression,
    /// Inside string literal — expect chars or closing quote
    StringLiteral,
    /// Inside block comment
    BlockComment,
    /// After `#` — expect attribute
    Attribute,
    /// Error state — cannot recover
    Invalid,
}

#[derive(Clone, Debug, Default)]
pub struct DepthTracker {
    pub paren: u32,    // ( )
    pub brace: u32,    // { }
    pub bracket: u32,  // [ ]
    pub angle: u32,    // < > (approximation — Rust's <> is context-dependent)
}

impl PartialParser {
    /// Validate whether appending `token_str` to the current buffer is syntactically plausible.
    /// Returns true if the resulting partial code could be valid Rust.
    /// 
    /// This is a FAST heuristic check — not a full AST parse.
    /// False positives are allowed (target model catches them).
    /// False negatives are minimized (we don't want to prune valid branches).
    pub fn is_plausible(&self, token_str: &str) -> bool {
        // ... state machine logic
    }
    
    /// Full validation using syn — for completed paths only.
    /// Called after DDTree produces candidate paths, before target verification.
    pub fn full_validate(code: &str) -> Result<(), SynError> {
        // Try parsing as various Rust AST nodes
        if parse_str::<syn::File>(code).is_ok() { return Ok(()); }
        if parse_str::<syn::Item>(code).is_ok() { return Ok(()); }
        if parse_str::<syn::Stmt>(code).is_ok() { return Ok(()); }
        if parse_str::<syn::Expr>(code).is_ok() { return Ok(()); }
        parse_str::<syn::Type>(code).map(|_| ()).map_err(|e| e)
    }
}
```

### 2.3 SynPruner Implementation

```rust
// clora/syn_pruner.rs

/// Compiler-in-the-Loop pruner using syn for Rust syntax validation.
/// 
/// Two-tier validation:
/// 1. FAST (per-token): PartialParser state machine — microsecond-scale
/// 2. SLOW (per-path): syn full parse — after DDTree produces candidate paths
///
/// This pruner goes into the DDTree hot loop via ConstraintPruner trait.
pub struct SynPruner {
    tokenizer: Arc<BpeTokenizer>,
    /// Per-path parsers — cloned from template for each DDTree branch
    parser_template: PartialParser,
    /// Enable full syn validation on completed paths
    full_validation: bool,
}

impl SynPruner {
    /// Create a new SynPruner with the given tokenizer.
    pub fn new(tokenizer: Arc<BpeTokenizer>) -> Self {
        Self {
            tokenizer,
            parser_template: PartialParser::new(),
            full_validation: true,
        }
    }
    
    /// Full validation of a completed code string.
    /// Used after DDTree produces candidate paths.
    pub fn validate_path(&self, token_ids: &[usize]) -> PruneResult {
        let code = self.tokenizer.decode(token_ids);
        match PartialParser::full_validate(&code) {
            Ok(()) => PruneResult::Valid,
            Err(e) => PruneResult::Invalid {
                error: e.to_string(),
                error_kind: classify_error(&e),
            },
        }
    }
}

impl ConstraintPruner for SynPruner {
    fn is_valid(&self, depth: usize, token_idx: usize, parent_tokens: &[usize]) -> bool {
        // Decode parent tokens + candidate into string
        let mut all_tokens = parent_tokens.to_vec();
        all_tokens.push(token_idx);
        let partial_code = self.tokenizer.decode(&all_tokens);
        
        // Fast partial validation — state machine check
        let mut parser = self.parser_template.clone();
        for token_id in &all_tokens {
            let token_str = self.tokenizer.decode(&[*token_id]);
            if !parser.is_plausible(&token_str) {
                return false;
            }
        }
        true
    }
}

#[derive(Debug)]
pub enum PruneResult {
    Valid,
    Invalid { error: String, error_kind: ErrorKind },
}

#[derive(Debug)]
pub enum ErrorKind {
    SyntaxError,
    UnexpectedToken,
    UnclosedDelimiter,
    InvalidExpression,
    Other,
}

fn classify_error(error: &syn::Error) -> ErrorKind {
    let msg = error.to_string();
    if msg.contains("unexpected token") { ErrorKind::UnexpectedToken }
    else if msg.contains("expected") { ErrorKind::SyntaxError }
    else if msg.contains("unclosed") { ErrorKind::UnclosedDelimiter }
    else { ErrorKind::Other }
}
```

### 2.4 Performance Strategy

| Validation Tier | When | Method | Latency |
|----------------|------|--------|---------|
| **Tier 0: DFA** | Per-token in DDTree hot loop | PartialParser state machine | ~100ns |
| **Tier 1: syn** | Per-path after DDTree build | `syn::parse_str` | ~1-10μs |
| **Tier 2: cargo check** | Post-generation, batch | `cargo check` subprocess | ~100ms-1s |
| **Tier 3: clippy** | Training data gate | `cargo clippy` subprocess | ~200ms-2s |

The DDTree only uses Tier 0 (fast). Tier 1 runs on the top-K paths. Tier 2-3 run offline.

## Phase 3: Training Data Pipeline

### 3.1 Data Flow

```
┌──────────────────────────────────────────────────────────┐
│                   DATA SOURCES                           │
│                                                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────┐ │
│  │ Rust Doc │  │ GitHub   │  │ Crates   │  │ Rust    │ │
│  │ (std)    │  │ repos    │  │ .io top  │  │ Book    │ │
│  │          │  │          │  │ 1000     │  │ Ref     │ │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬────┘ │
│       │              │              │              │      │
└───────┼──────────────┼──────────────┼──────────────┼──────┘
        │              │              │              │
        ▼              ▼              ▼              ▼
┌──────────────────────────────────────────────────────────┐
│              INGESTER (data/ingester.rs)                  │
│                                                          │
│  1. Walk directories, find .rs files                     │
│  2. Read + normalize (strip comments, normalize ws)      │
│  3. Deduplicate via blake3 hash                          │
│  4. Split into training chunks (256-1024 tokens)         │
│  5. BPE encode each chunk                                │
│                                                          │
│  Output: Vec<CorpusEntry>                                │
│    { file_path, blake3_hash, bpe_tokens, raw_source }    │
└──────────────────────┬───────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────┐
│              FILTER (data/filter.rs)                      │
│                                                          │
│  Tier 1: syn validation                                  │
│    parse_str::<File>(source) → must succeed              │
│                                                          │
│  Tier 2: cargo check (optional, slow)                    │
│    Create temp crate, write source, cargo check          │
│    Must pass with 0 errors, 0 warnings                   │
│                                                          │
│  Tier 3: Quality heuristics                              │
│    - Min 10 lines, max 500 lines per chunk               │
│    - Must contain at least 1 function definition         │
│    - No TODO/FIXME/HACK comments                         │
│    - No unsafe blocks (unless explicitly opted in)       │
│    - blake3 dedup (skip identical chunks)                │
│                                                          │
│  Output: Vec<TrainingSample>                              │
│    { bpe_tokens, source, quality_score, syn_valid }      │
└──────────────────────┬───────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────┐
│              EXPORTER (data/exporter.rs)                  │
│                                                          │
│  Format 1: JSONL for LoRA fine-tuning                    │
│    {"input": "<BOS> pub fn hello() { ...",               │
│     "output": "pub fn hello() -> String { ..."}          │
│                                                          │
│  Format 2: Binary for microgpt-rs direct training        │
│    [token_ids: u32] packed                               │
│                                                          │
│  Format 3: anyrag/Turso ingestion payload                │
│    POST /ingest/text with structured chunks               │
│                                                          │
│  Output: training.jsonl, tokens.bin, anyrag_payload.json │
└──────────────────────────────────────────────────────────┘
```

### 3.2 CorpusEntry & TrainingSample Types

```rust
// data/types.rs

/// A single entry from the Rust corpus (pre-filter).
pub struct CorpusEntry {
    /// Source file path (relative to corpus root)
    pub file_path: String,
    /// blake3 hash for deduplication
    pub content_hash: [u8; 32],
    /// Raw source text
    pub source: String,
    /// BPE-encoded token IDs
    pub bpe_tokens: Vec<usize>,
    /// Source repository/package
    pub origin: String,
    /// Line count
    pub line_count: usize,
}

/// A training sample that passed quality filters.
pub struct TrainingSample {
    /// BPE token IDs (input sequence)
    pub tokens: Vec<usize>,
    /// Original source (for debugging)
    pub source: String,
    /// Quality score (0.0-1.0, based on complexity + idiomacy)
    pub quality_score: f32,
    /// Whether syn full-parse succeeded
    pub syn_valid: bool,
    /// Whether cargo check succeeded (if run)
    pub cargo_check_valid: Option<bool>,
    /// Classification for concept sharding
    pub concepts: Vec<RustConcept>,
}

/// Concept tags for anyrag concept sharding.
#[derive(Clone, Debug, PartialEq)]
pub enum RustConcept {
    AsyncAwait,
    Traits,
    Lifetimes,
    ErrorHandling,
    SmartPointers,
    Concurrency,
    Unsafe,
    Macros,
    FFi,
    Generics,
    Pattern,
    Iterators,
    Closures,
    BuildSystem,
    Testing,
    Serialization,
    Networking,
    Collections,
}

/// Quality report from the filter pipeline.
pub struct QualityReport {
    pub total_entries: usize,
    pub syn_valid: usize,
    pub cargo_check_valid: usize,
    pub deduplicated: usize,
    pub final_samples: usize,
    pub rejection_reasons: HashMap<String, usize>,
}
```

### 3.3 Cargo Check Gate

```rust
// data/filter.rs

impl TrainingFilter {
    /// Run cargo check on a source snippet.
    /// Creates a temporary crate, writes the source, runs cargo check.
    /// Returns stdout/stderr for error feedback.
    pub fn cargo_check(source: &str) -> CargoCheckResult {
        let temp_dir = TempDir::new("clora_check").unwrap();
        
        // Write Cargo.toml
        let cargo_toml = r#"
[package]
name = "check"
version = "0.1.0"
edition = "2024"
"#;
        fs::write(temp_dir.path().join("Cargo.toml"), cargo_toml)?;
        
        // Write source as src/lib.rs
        fs::create_dir_all(temp_dir.path().join("src"))?;
        fs::write(temp_dir.path().join("src/lib.rs"), source)?;
        
        // Run cargo check
        let output = Command::new("cargo")
            .args(["check", "--quiet", "--message-format=short"])
            .current_dir(temp_dir.path())
            .output()?;
        
        CargoCheckResult {
            success: output.status.success(),
            stderr: String::from_utf8_lossy(&output.stderr).to_string(),
        }
    }
}
```

### 3.4 Scale Estimates

| Source | Estimated .rs files | Estimated lines | After dedup |
|--------|--------------------:|----------------:|------------:|
| `rust-lang/rust` | ~30K | ~15M | ~10M |
| Top 1000 crates.io | ~200K | ~100M | ~60M |
| GitHub Rust repos (top 10K) | ~2M | ~500M | ~200M |
| Rust docs/examples | ~5K | ~2M | ~1.5M |
| **Total** | **~2.2M** | **~617M** | **~271M** |

At 256 tokens per training sample (avg ~100 lines), this yields:
- **~2.7M training samples** from 271M lines
- After filtering (estimated 70% pass rate): **~1.9M high-quality samples**
- JSONL size: ~10-20 GB

## Phase 4: Error Feedback Loop (Self-Correction)

### 4.1 Compiler Errors as Steering Signals

When `syn` or `cargo check` rejects a drafted path, the error message becomes context for the next draft iteration:

```rust
// clora/error_feedback.rs

/// Converts compiler errors into steering context for the LLM.
pub struct ErrorFeedback {
    /// The error message from syn or cargo check
    pub error_message: String,
    /// The code that caused the error
    pub failing_code: String,
    /// The error kind (for classification)
    pub error_kind: ErrorKind,
    /// Suggested fix (if extractable from error message)
    pub suggestion: Option<String>,
}

impl ErrorFeedback {
    /// Format as a context string to prepend to the next LLM prompt.
    pub fn to_context(&self) -> String {
        format!(
            "/* COMPILER ERROR: {}\n   In code: {}\n   Suggestion: {} */\n",
            self.error_message,
            self.failing_code.lines().next().unwrap_or(""),
            self.suggestion.as_deref().unwrap_or("review the error above")
        )
    }
    
    /// Extract suggestion from common rustc error patterns.
    pub fn extract_suggestion(error: &str) -> Option<String> {
        // E0382: use of moved value → "consider cloning or using a reference"
        if error.contains("E0382") || error.contains("use of moved value") {
            return Some("consider .clone() or using a reference (&T)".to_string());
        }
        // E0495: lifetime may not live long enough → "add explicit lifetime"
        if error.contains("E0495") || error.contains("lifetime") {
            return Some("add explicit lifetime annotation".to_string());
        }
        // E0277: the trait bound is not satisfied → "implement the trait"
        if error.contains("E0277") || error.contains("trait bound") {
            return Some("implement the required trait or add a where clause".to_string());
        }
        None
    }
}
```

### 4.2 Feedback Integration with DDTree

```rust
// In speculative step — when all branches are pruned:

if valid_branches.is_empty() {
    // Take the best pruned branch's error and feed it back
    let feedback = pruned_branches
        .iter()
        .max_by(|a, b| a.log_prob.partial_cmp(&b.log_prob).unwrap())
        .and_then(|b| b.compiler_feedback.clone());
    
    if let Some(error) = feedback {
        let steering = ErrorFeedback {
            error_message: error,
            failing_code: current_context.to_string(),
            error_kind: ErrorKind::Other,
            suggestion: ErrorFeedback::extract_suggestion(&error),
        };
        // Inject steering context into the next draft iteration
        context.push_str(&steering.to_context());
    }
}
```

## Phase 5: anyrag Integration (Self-Improving Loop)

### 5.1 The 32-Day Cycle (from Research)

```
Day 1-29: RAG Operation
  ├── User submits Python → Rust translation request
  ├── anyrag retrieves relevant examples from Turso
  ├── microgpt-rs drafts code (DDTree + SynPruner)
  ├── Target model verifies semantics
  ├── cargo check validates final output
  └── IF valid: INSERT hidden_state + code INTO Turso episodic

Day 30: Synthesis
  ├── anyrag batch analyzes episodic database
  ├── Groups similar translation patterns
  ├── Generates structured Q&A pairs
  └── Exports to JSONL

Day 31: LoRA Fine-Tuning
  ├── Train LoRA adapter on accumulated JSONL
  ├── Validate zero-shot compilation rate on held-out test set
  └── Save as lora.bin

Day 32: Base Upgrade
  ├── Load new lora.bin into draft model
  ├── Higher acceptance rate → faster inference
  ├── Wipe episodic memory (patterns internalized)
  └── Begin new cycle collecting edge cases
```

### 5.2 anyrag API Integration Points

```rust
// In the inference pipeline:

/// After successful cargo check, save to anyrag for RAG.
pub fn save_successful_compile(
    anyrag_url: &str,
    hidden_state: &[f32],
    source_code: &str,
    concepts: &[RustConcept],
) -> Result<(), anyhow::Error> {
    let client = reqwest::Client::new();
    
    // POST /ingest/text — save as episodic memory
    client.post(format!("{anyrag_url}/ingest/text"))
        .json(&serde_json::json!({
            "text": source_code,
            "owner_id": "clora-pipeline",
            "metadata": {
                "type": "validated_rust",
                "concepts": concepts.iter().map(|c| format!("{c:?}")).collect::<Vec<_>>(),
                "hidden_state_dim": hidden_state.len(),
            }
        }))
        .send()
        .await?;
    
    Ok(())
}

/// Before drafting, query anyrag for similar patterns.
pub fn retrieve_similar_patterns(
    anyrag_url: &str,
    hidden_state: &[f32],
    top_k: usize,
) -> Result<Vec<String>, anyhow::Error> {
    let client = reqwest::Client::new();
    
    // POST /search/vector — retrieve similar compiled code
    let response = client.post(format!("{anyrag_url}/search/vector"))
        .json(&serde_json::json!({
            "query_vector": hidden_state,
            "top_k": top_k,
            "filter": {"type": "validated_rust"}
        }))
        .send()
        .await?;
    
    // Extract source code from results
    let results: Vec<String> = response.json().await?;
    Ok(results)
}
```

### 5.3 Concept Sharding for Rust

Map Rust concepts to anyrag's concept shards for sub-millisecond retrieval:

| Shard | Rust Concepts | Example Patterns |
|-------|--------------|-----------------|
| `async` | AsyncAwait, Concurrency | `async fn`, `.await`, `tokio::spawn` |
| `types` | Traits, Lifetimes, Generics | `impl Trait`, `where T:`, `<'a>` |
| `memory` | SmartPointers, Unsafe | `Arc<Mutex<>>`, `Box::new`, `unsafe` |
| `errors` | ErrorHandling | `Result<T,E>`, `?`, `anyhow` |
| `meta` | Macros, BuildSystem | `macro_rules!`, `#[derive(`, `build.rs` |
| `data` | Collections, Iterators | `Vec::new`, `.map()`, `.collect()` |
| `ffi` | FFi, Serialization | `extern "C"`, `#[no_mangle]`, `serde` |

## Phase 6: Benchmarking Strategy

### 6.1 Benchmarks Before/After

| Benchmark | What It Measures | Baseline (no cLoRA) | Target (with cLoRA) |
|-----------|-----------------|--------------------|--------------------|
| `bench_ddtree_build` | DDTree construction speed | Current speed | ≤5% slower (pruner overhead) |
| `bench_syn_prune` | SynPruner per-token speed | N/A | <1μs/token |
| `bench_full_validate` | syn full-parse per-path | N/A | <10μs/path |
| `bench_inference` | End-to-end tok/s | X tok/s | ≥0.9X tok/s (pruning saves target verify) |
| `bench_compilation_rate` | % outputs passing cargo check | ~60-70% (LLM baseline) | **>95%** (cLoRA goal) |
| `bench_acceptance_rate` | Draft acceptance rate | ~75% | **>85%** (after LoRA) |

### 6.2 Quality Metrics

```
Zero-Shot Compilation Rate:
  ┌──────────────────────────────────────────────────┐
  │ Without cLoRA:  ~60-70% of LLM outputs compile   │
  │ With SynPruner: ~80-85% (syntax-only filter)     │
  │ With cargo check: ~95%+ (full validation)        │
  │ After LoRA:     ~98%+ (internalized patterns)    │
  └──────────────────────────────────────────────────┘
```

## Tasks

### Phase 1: BPE Tokenizer

- [ ] 1.1 Create `src/tokenizer/` module with `mod.rs`, `types.rs`, `bpe.rs`, `trainer.rs`, `rust_vocab.rs`
- [ ] 1.2 Implement `BpeTokenizer` struct with encode/decode
- [ ] 1.3 Implement `BpeTrainer` that trains from a directory of .rs files
- [ ] 1.4 Create `src/data/ingester.rs` that walks directories and extracts .rs files
- [ ] 1.5 Run BPE training on initial corpus (rust-lang/rust + top 100 crates)
- [ ] 1.6 Generate `rust_vocab.rs` with pre-trained vocabulary
- [ ] 1.7 Update `Config` with tokenizer fields and new defaults
- [ ] 1.8 Add tests: encode/decode roundtrip, vocab coverage, special tokens
- [ ] 1.9 Benchmark: BPE encode/decode throughput

### Phase 2: SynPruner (cLoRA Core)

- [ ] 2.1 Create `src/clora/` module with `mod.rs`, `types.rs`, `syn_pruner.rs`, `partial_parser.rs`
- [ ] 2.2 Implement `PartialParser` state machine for Rust syntax
- [ ] 2.3 Implement `SynPruner` that implements `ConstraintPruner` trait
- [ ] 2.4 Add `syn` and `proc-macro2` dependencies behind `clora` feature flag
- [ ] 2.5 Implement `PruneResult` and `ErrorKind` classification
- [ ] 2.6 Integrate SynPruner into DDTree via `build_dd_tree_pruned`
- [ ] 2.7 Implement `error_feedback.rs` — extract suggestions from rustc errors
- [ ] 2.8 Add tests: partial parser accepts valid fragments, rejects invalid
- [ ] 2.9 Add tests: SynPruner prunes invalid Rust, accepts valid Rust
- [ ] 2.10 Benchmark: SynPruner overhead vs NoPruner on DDTree build

### Phase 3: Training Data Pipeline

- [ ] 3.1 Create `src/data/` module with `mod.rs`, `types.rs`, `ingester.rs`, `filter.rs`, `exporter.rs`
- [ ] 3.2 Implement `CorpusIngester` — walk dirs, read .rs, hash with blake3, dedup
- [ ] 3.3 Implement `TrainingFilter` — syn validation, quality heuristics
- [ ] 3.4 Implement `cargo_check` gate (temp crate creation + subprocess)
- [ ] 3.5 Implement `ConceptClassifier` — tag samples with RustConcept
- [ ] 3.6 Implement `TrainingExporter` — JSONL output, binary token output
- [ ] 3.7 Add `walker` and `blake3` dependencies
- [ ] 3.8 Add `serde` + `serde_json` dependencies for JSONL
- [ ] 3.9 Add tests: corpus ingestion, dedup, filter, export
- [ ] 3.10 Add CLI command: `cargo run --bin clora-ingest -- /path/to/corpus`

### Phase 4: Error Feedback Loop

- [ ] 4.1 Implement `ErrorFeedback` struct with `to_context()` and `extract_suggestion()`
- [ ] 4.2 Add feedback injection into speculative step (when all branches pruned)
- [ ] 4.3 Add feedback context to DDTree re-draft cycle
- [ ] 4.4 Add tests: error classification, suggestion extraction
- [ ] 4.5 Benchmark: acceptance rate improvement with feedback loop

### Phase 5: anyrag Integration

- [ ] 5.1 Add `reqwest` dependency (behind `anyrag` feature flag)
- [ ] 5.2 Implement `save_successful_compile()` — POST to anyrag /ingest/text
- [ ] 5.3 Implement `retrieve_similar_patterns()` — POST to anyrag /search/vector
- [ ] 5.4 Implement concept shard mapping (RustConcept → anyrag metadata)
- [ ] 5.5 Implement `/knowledge/export` JSONL → LoRA training data conversion
- [ ] 5.6 Add integration tests with mock anyrag server
- [ ] 5.7 Document anyrag setup for cLoRA pipeline

### Phase 6: Benchmarking & Validation

- [ ] 6.1 Add `bench_syn_prune` to benchmark suite
- [ ] 6.2 Add `bench_full_validate` to benchmark suite
- [ ] 6.3 Add `bench_compilation_rate` — generate N samples, cargo check each
- [ ] 6.4 Run baseline benchmarks (no cLoRA) → `bench/015_bench_baseline.png`
- [ ] 6.5 Run cLoRA benchmarks (with SynPruner) → `bench/016_bench_clora.png`
- [ ] 6.6 Measure compilation rate improvement
- [ ] 6.7 Run on larger model (BPE vocab, bigger draft) → `bench/017_bench_bpe.png`

## Feature Flags

```toml
[features]
default = []
leviathan = []           # Real p/q verification with target model
sudoku = []              # Sudoku constraint pruner
clora = ["syn"]          # Compiler-in-the-loop pruning
training = ["serde", "serde_json", "walkdir"]  # Training data pipeline
anyrag-integration = ["reqwest"]  # anyrag REST API client
full = ["leviathan", "sudoku", "clora", "training"]
```

## Key Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|-----------|
| `syn` too slow for per-token pruning | DDTree build time 10x | PartialParser DFA for hot loop, syn only for paths |
| BPE vocab doesn't align with syn tokens | Pruner rejects valid code | Test with partial sequences; allow false positives |
| Training corpus too noisy | Low-quality lora.bin | Multi-tier filtering: syn → cargo check → clippy |
| Memory overhead of large vocab | OOM on small devices | Configurable vocab size; 4K/8K/16K/32K options |
| Partial parser false negatives | Valid branches pruned | Tune DFA to be permissive; only prune obvious errors |
| cargo check latency | Training pipeline too slow | Parallel temp crate checking with rayon |

## Expected Outcomes

1. **SynPruner**: A `ConstraintPruner` implementation that uses incremental Rust syntax validation to prune invalid DDTree branches before target verification
2. **BPE Tokenizer**: A ~32K token vocabulary trained on the Rust ecosystem, enabling meaningful code generation
3. **Training Pipeline**: A data pipeline that ingests Rust docs + GitHub repos → filters through syn + cargo check → exports clean JSONL
4. **Error Feedback Loop**: Self-correction mechanism that feeds compiler errors back into the drafting context
5. **anyrag Integration**: Episodic memory of successful compiles → concept-sharded retrieval → LoRA fine-tuning cycle
6. **Quality Metrics**: >95% zero-shot compilation rate (up from ~60-70% without cLoRA)

## Files to Create/Modify

| File | Action | Phase |
|------|--------|-------|
| `Cargo.toml` | Add deps + feature flags | 1-5 |
| `src/tokenizer/mod.rs` | New | 1 |
| `src/tokenizer/types.rs` | New | 1 |
| `src/tokenizer/bpe.rs` | New | 1 |
| `src/tokenizer/trainer.rs` | New | 1 |
| `src/tokenizer/rust_vocab.rs` | New (generated) | 1 |
| `src/clora/mod.rs` | New | 2 |
| `src/clora/types.rs` | New | 2 |
| `src/clora/syn_pruner.rs` | New | 2 |
| `src/clora/partial_parser.rs` | New | 2 |
| `src/clora/error_feedback.rs` | New | 4 |
| `src/clora/training_filter.rs` | New | 3 |
| `src/data/mod.rs` | New | 3 |
| `src/data/types.rs` | New | 3 |
| `src/data/ingester.rs` | New | 3 |
| `src/data/filter.rs` | New | 3 |
| `src/data/exporter.rs` | New | 3 |
| `src/types.rs` | Extend Config | 1 |
| `src/lib.rs` | Add mod clora, tokenizer, data | 1-3 |
| `src/speculative/mod.rs` | Add syn_pruner re-export | 2 |
| `src/benchmark.rs` | Add clora benchmarks | 6 |

## References

- `.research/01_Advanced Neuro-Symbolic Rust Translation.md` — Grand Unification architecture
- `.research/00_Neuro-Symbolic LLM Architecture.md` — Original cLoRA concept
- `.plans/004_leviathan_distill.md` — SpeculativeVerifier trait pattern
- `.plans/005_speculative_module_refactor.md` — ConstraintPruner trait, DDTree pruning
- `anyrag/README.md` — RAG pipeline, concept sharding, JSONL export