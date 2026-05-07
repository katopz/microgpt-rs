//! Validator Demo: BPE Tokenizer → DDTree → SynPruner Validation
//!
//! Demonstrates the compiler-in-the-loop pipeline:
//! 1. Train a BPE tokenizer on Rust source code
//! 2. Encode sample code into token IDs
//! 3. Build a DDTree with and without SynPruner
//! 4. Compare pruned vs unpruned tree sizes
//! 5. Validate code fragments through two-tier syntax checking
//!
//! Run: cargo run --example validator_demo --features validator

use std::sync::Arc;

use microgpt_rs::speculative::{NoPruner, build_dd_tree, build_dd_tree_pruned};
use microgpt_rs::tokenizer::{BpeTokenizerImpl, BpeTrainer};
use microgpt_rs::types::Config;
use microgpt_rs::validator::{CompilerFeedback, ErrorKind, PartialParser, SynPruner};

/// Sample Rust code corpus for BPE training.
const RUST_CORPUS: &str = r#"
fn main() {
    let x = 42;
    let mut y = 0;
    if x > 10 {
        y = x + 1;
    } else {
        y = x - 1;
    }
    println!("{}", y);
}

fn add(a: i32, b: i32) -> i32 {
    a + b
}

struct Point {
    x: f64,
    y: f64,
}

impl Point {
    fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }
    fn distance(&self, other: &Point) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        (dx * dx + dy * dy).sqrt()
    }
}

enum Result<T, E> {
    Ok(T),
    Err(E),
}

fn factorial(n: u64) -> u64 {
    match n {
        0 => 1,
        _ => n * factorial(n - 1),
    }
}

fn fibonacci(n: u32) -> u64 {
    let mut a = 0u64;
    let mut b = 1u64;
    for _ in 0..n {
        let temp = b;
        b = a + b;
        a = temp;
    }
    a
}

trait Summary {
    fn summarize(&self) -> String;
}

fn longest<'a>(x: &'a str, y: &'a str) -> &'a str {
    if x.len() > y.len() { x } else { y }
}
"#;

/// Code fragments for validation testing.
const VALID_FRAGMENTS: &[&str] = &[
    "fn main() { }",
    "let x = 42;",
    "struct Foo { x: i32 }",
    "if true { } else { }",
    "match n { 0 => 1, _ => n }",
    "fn add(a: i32, b: i32) -> i32 { a + b }",
    "let s = \"hello\";",
    "arr.iter().map(|x| x * 2).collect()",
];

const INVALID_FRAGMENTS: &[&str] = &[
    "fn main() {",      // unmatched brace
    "let x = ;",        // missing expression
    "if true { } else", // missing else block
    "foo(((bar)",       // unbalanced parens
    "struct { }",       // missing name
    "match { 0 => 1 }", // missing expression
];

fn main() {
    println!("🧠 Validator Demo: Compiler-in-the-Loop Token Pruning");
    println!("{}", "═".repeat(60));

    // ── Phase 1: Train BPE Tokenizer ─────────────────────────────
    println!("\n📖 Phase 1: BPE Tokenizer Training");
    println!("{}", "─".repeat(40));

    let vocab_size = 256;
    let tokenizer = BpeTrainer::train(RUST_CORPUS, vocab_size);
    println!(
        "  Vocabulary size: {} tokens (target: {vocab_size})",
        tokenizer.id_to_vocab.len()
    );
    println!("  Merge rules: {}", tokenizer.merges.len());
    println!(
        "  Special tokens: <pad>={}, <bos>={}, <eos>={}, <unk>={}",
        tokenizer.pad_id,
        tokenizer.bos_id,
        tokenizer.eos_id,
        tokenizer.unk_id()
    );

    // ── Phase 2: Encode/Decode Roundtrip ─────────────────────────
    println!("\n🔤 Phase 2: Encode/Decode Roundtrip");
    println!("{}", "─".repeat(40));

    let samples = ["fn main() { }", "let x = 42;", "a + b"];
    for sample in &samples {
        let ids = BpeTokenizerImpl::encode(&tokenizer, sample);
        let decoded = BpeTokenizerImpl::decode(&tokenizer, &ids);
        let roundtrip_ok = decoded == *sample;
        println!(
            "  '{}' → {:?} → '{}' {}",
            sample,
            ids,
            decoded,
            if roundtrip_ok { "✓" } else { "✗ MISMATCH" }
        );
    }

    // ── Phase 3: Two-Tier Syntax Validation ──────────────────────
    println!("\n🔍 Phase 3: Two-Tier Syntax Validation");
    println!("{}", "─".repeat(40));

    let tokenizer_arc = Arc::new(tokenizer.clone());
    let mut syn_pruner = SynPruner::new(Arc::clone(&tokenizer_arc));

    println!("\n  Valid fragments:");
    for code in VALID_FRAGMENTS {
        let result = syn_pruner.validate(code);
        let tier = match &result.error_kind {
            ErrorKind::None => "✓ syn OK",
            ErrorKind::UnbalancedBrackets => "✗ brackets",
            ErrorKind::SynError(msg) => {
                let suggestion = CompilerFeedback::extract_suggestion(msg)
                    .map(|s| format!(" ({s})"))
                    .unwrap_or_default();
                &format!("✗ syn: {msg}{suggestion}")
            }
        };
        println!("    {tier:40} | {code}");
    }

    println!("\n  Invalid fragments:");
    for code in INVALID_FRAGMENTS {
        let result = syn_pruner.validate(code);
        let tier = match &result.error_kind {
            ErrorKind::None => "✓ syn OK (unexpected!)",
            ErrorKind::UnbalancedBrackets => "✗ Tier 0: brackets",
            ErrorKind::SynError(msg) => {
                let short: String = msg.chars().take(40).collect();
                &format!("✗ Tier 1: {short}")
            }
        };
        println!("    {tier:50} | {code}");
    }

    // ── Phase 4: Tier 0 Bracket Balancer Stats ───────────────────
    println!("\n⚙️  Phase 4: Bracket Balancer (PartialParser)");
    println!("{}", "─".repeat(40));

    let mut parser = PartialParser::new();
    let balance_tests: &[(&str, bool)] = &[
        ("{[()]}", true),
        ("fn() { x + y }", true),
        ("(((", false),
        ("})]", false),
        ("let s = \"{not a bracket}\";", true),
        ("// comment with { bracket\nlet x = 1;", true),
        ("/* block { comment } */ let y = 2;", true),
        ("'{'", true),
    ];

    for (code, expected) in balance_tests {
        let result = parser.is_valid(code);
        let status = if result == *expected {
            "✓"
        } else {
            "✗ WRONG"
        };
        println!("  {status} {code}");
    }

    // ── Phase 5: DDTree with SynPruner vs NoPruner ───────────────
    println!("\n🌳 Phase 5: DDTree Pruning Comparison");
    println!("{}", "─".repeat(40));

    let config = Config::bpe_draft();
    let vocab = config.vocab_size;

    // Create synthetic marginals (uniform distribution over a subset)
    let num_valid_tokens = 10usize;
    let marginals: Vec<Vec<f32>> = (0..config.draft_lookahead)
        .map(|_| {
            let mut probs = vec![0.0f32; vocab];
            for prob in probs.iter_mut().take(num_valid_tokens.min(vocab)) {
                *prob = 1.0 / num_valid_tokens as f32;
            }
            probs
        })
        .collect();
    let mv: Vec<&[f32]> = marginals.iter().map(|s| s.as_slice()).collect();

    // Unpruned tree
    let tree_unpruned = build_dd_tree(&mv, &config);

    // NoPruner (should match unpruned)
    let tree_no_prune = build_dd_tree_pruned(&mv, &config, &NoPruner, false);

    // SynPruner (should reduce tree)
    let tree_syn = build_dd_tree_pruned(&mv, &config, &syn_pruner, false);

    println!(
        "  Config: vocab={vocab}, lookahead={}",
        config.draft_lookahead
    );
    println!("  Tree budget: {}", config.tree_budget);
    println!("  Unpruned:      {:3} nodes", tree_unpruned.len());
    println!(
        "  NoPruner:      {:3} nodes (baseline)",
        tree_no_prune.len()
    );
    println!("  SynPruner:     {:3} nodes", tree_syn.len());

    let reduction = if !tree_no_prune.is_empty() {
        let pct = 100.0 * (1.0 - tree_syn.len() as f64 / tree_no_prune.len() as f64);
        format!("{pct:.1}%")
    } else {
        "N/A".to_string()
    };
    println!("  Reduction:     {reduction}");

    // ── Phase 6: Compiler Feedback Self-Correction ───────────────
    println!("\n🔄 Phase 6: Compiler Feedback Self-Correction");
    println!("{}", "─".repeat(40));

    let error_cases = ["let = ;", "fn main() {", "struct { }"];
    for code in &error_cases {
        let result = syn_pruner.validate(code);
        if !result.is_valid {
            let feedback = CompilerFeedback {
                error_message: match &result.error_kind {
                    ErrorKind::SynError(msg) => msg.clone(),
                    ErrorKind::UnbalancedBrackets => "Unbalanced brackets".to_string(),
                    ErrorKind::None => String::new(),
                },
                failing_code: (*code).to_string(),
                suggestion: match &result.error_kind {
                    ErrorKind::SynError(msg) => CompilerFeedback::extract_suggestion(msg),
                    _ => None,
                },
            };
            println!("  Code:    {code}");
            println!("  Error:   {}", feedback.error_message);
            if let Some(suggestion) = &feedback.suggestion {
                println!("  Hint:    {suggestion}");
            }
            println!("  Context: {}", feedback.to_context().replace('\n', " | "));
            println!();
        }
    }

    // ── Summary ──────────────────────────────────────────────────
    println!("{}", "═".repeat(60));
    println!("✅ Validator pipeline complete");
    println!();
    println!(
        "  BPE tokenizer:   {} vocab, {} merges",
        tokenizer.id_to_vocab.len(),
        tokenizer.merges.len()
    );
    println!("  Tier 0 (bracket): O(n) DFA, rejects unbalanced code");
    println!("  Tier 1 (syn):    Full Rust parse, accurate but expensive");
    println!("  DDTree pruning:  SynPruner reduces invalid branches before verification");
    println!("  Self-correction:  Compiler feedback extracts suggestions for re-prompting");
}
