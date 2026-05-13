//! hello_py2rs — Python → Rust Transpilation Demo (Plan 025)
//!
//! Demonstrates ZAYA-inspired modality LoRA switching with REAL code I/O:
//!
//!   1. Train BPE tokenizer on a Python + Rust corpus
//!   2. Encode real Python code into token IDs
//!   3. Bidirectional prefill with Python-reader LoRA
//!   4. Causal decode with Rust-writer LoRA
//!   5. Decode generated tokens back to real text
//!
//! Honest note: with random weights, the output is gibberish.
//! The *pipeline* is real — trained weights would produce real transpilation.
//!
//! Run: cargo run --example hello_py2rs

use microgpt_rs::tokenizer::{BpeTokenizerImpl, BpeTrainer};
use microgpt_rs::transformer::{
    ForwardContext, MultiLayerKVCache, PrefillContext, TransformerWeights, forward,
    forward_prefill, generate_with_prefill,
};
use microgpt_rs::types::{Config, LoraAdapter, LoraPair, Rng};

// ---------------------------------------------------------------------------
// Training corpus: Python + Rust code
// ---------------------------------------------------------------------------

/// Python source examples for BPE training + prompt.
const PYTHON_CORPUS: &str = r#"
def greet():
    print("hello")

def add(a, b):
    return a + b

def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)

def fibonacci(n):
    a, b = 0, 1
    for i in range(n):
        a, b = b, a + b
    return a

def square(x):
    return x * x

def max_of(a, b):
    if a > b:
        return a
    return b

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def distance(self, other):
        dx = self.x - other.x
        dy = self.y - other.y
        return (dx * dx + dy * dy) ** 0.5
"#;

/// Rust source examples for BPE training (target modality).
const RUST_CORPUS: &str = r#"
fn greet() {
    println!("hello");
}

fn add(a: i32, b: i32) -> i32 {
    a + b
}

fn factorial(n: u64) -> u64 {
    if n <= 1 {
        return 1;
    }
    n * factorial(n - 1)
}

fn fibonacci(n: u32) -> u64 {
    let mut a: u64 = 0;
    let mut b: u64 = 1;
    for _ in 0..n {
        let temp = b;
        b = a + b;
        a = temp;
    }
    a
}

fn square(x: f64) -> f64 {
    x * x
}

fn max_of(a: i32, b: i32) -> i32 {
    if a > b { a } else { b }
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
"#;

// ---------------------------------------------------------------------------
// LoRA helpers
// ---------------------------------------------------------------------------

/// Create a deterministic LoRA adapter for a given modality seed.
fn make_lora(config: &Config, seed: u32) -> LoraAdapter {
    let rank = config.lora_rank;
    let dim = config.n_embd;

    let a: Vec<f32> = (0..rank * dim)
        .map(|i| {
            let v = ((seed as u64).wrapping_mul((i + 1) as u64)) as u32;
            ((v as f32 / u32::MAX as f32) - 0.5) * 0.1
        })
        .collect();

    let b: Vec<f32> = (0..dim * rank)
        .map(|i| {
            let v = ((seed as u64).wrapping_mul((i + 100) as u64)) as u32;
            ((v as f32 / u32::MAX as f32) - 0.5) * 0.1
        })
        .collect();

    LoraAdapter {
        a,
        b,
        rank,
        alpha: config.lora_alpha,
        in_dim: dim,
        out_dim: dim,
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    println!("╔═══════════════════════════════════════════════════════════════════╗");
    println!("║  hello_py2rs: Python → Rust Transpilation Demo (Plan 025)       ║");
    println!("╚═══════════════════════════════════════════════════════════════════╝");
    println!();

    // ── Phase 1: Train BPE Tokenizer ──────────────────────────────
    // Train on combined Python + Rust corpus so the vocabulary covers
    // both "modalities" — Python keywords (def, class, return) and
    // Rust keywords (fn, let, mut, impl).

    let full_corpus = format!("{PYTHON_CORPUS}\n{RUST_CORPUS}");
    let target_vocab = 256;
    let tokenizer = BpeTrainer::train(&full_corpus, target_vocab);
    let actual_vocab = tokenizer.id_to_vocab.len();

    println!("━━━ Phase 1: BPE Tokenizer ━━━");
    println!("  Training corpus:  {} chars", full_corpus.len());
    println!("  Target vocab:     {target_vocab}");
    println!("  Actual vocab:     {actual_vocab} tokens");
    println!("  Merge rules:      {}", tokenizer.merges.len());
    println!();

    // ── Phase 2: Encode Python Prompt ─────────────────────────────

    let python_input = r#"def add(a, b):
    return a + b"#;

    let prompt_tokens = BpeTokenizerImpl::encode(&tokenizer, python_input);
    let roundtrip = BpeTokenizerImpl::decode(&tokenizer, &prompt_tokens);

    println!("━━━ Phase 2: Encode Python Input ━━━");
    println!("  Python source:");
    for line in python_input.lines() {
        println!("    {line}");
    }
    println!("  Tokens:    {} ids", prompt_tokens.len());
    println!(
        "  Roundtrip: \"{roundtrip}\" {}",
        if roundtrip == python_input {
            "✓"
        } else {
            "≠ (BPE lossy)"
        }
    );
    println!();

    // ── Phase 3: Config ───────────────────────────────────────────
    // Use Config::bpe() as base but override vocab to match tokenizer.

    let config = Config {
        vocab_size: actual_vocab,
        block_size: 128,
        n_layer: 2,
        bos_token: tokenizer.bos_id,
        n_embd: 32,
        n_head: 4,
        head_dim: 8,
        mlp_hidden: 128,
        n_kv_head: 4,
        temperature: 0.8,
        lora_rank: 4,
        lora_alpha: 8.0,
        ..Config::default()
    };
    config.validate().unwrap();

    println!("━━━ Phase 3: Model Config ━━━");
    println!(
        "  vocab={}, embd={}, heads={}, layers={}, lora_rank={}",
        config.vocab_size, config.n_embd, config.n_head, config.n_layer, config.lora_rank
    );
    println!();

    let mut rng = Rng::new(42);
    let weights = TransformerWeights::new(&config, &mut rng);

    // ── Phase 4: LoRA Pair ────────────────────────────────────────
    // Reader LoRA: activated during bidirectional prefill (Python context)
    // Writer LoRA: activated during causal decode (Rust generation)

    let lora_pair = LoraPair {
        reader: Some(make_lora(&config, 100)),
        writer: Some(make_lora(&config, 200)),
    };

    println!("━━━ Phase 4: Modality LoRA Pair ━━━");
    println!("  Reader (Python): seed=100, rank={}", config.lora_rank);
    println!("  Writer (Rust):   seed=200, rank={}", config.lora_rank);
    println!();

    // ── Phase 5: Bidirectional Prefill + Causal Decode ────────────

    println!("━━━ Phase 5: Pipeline Execution ━━━");
    println!();
    println!("  Step 1: Bidirectional Prefill");
    println!("    Input:   {} tokens of Python code", prompt_tokens.len());
    println!("    LoRA:    Reader (Python, seed=100)");
    println!("    Attn:    ◀ BIDIRECTIONAL — all Python tokens see each other ▶");

    let max_gen = 32;
    let mut ctx = ForwardContext::new(&config);
    let mut prefill = PrefillContext::new(&config);
    let mut cache = MultiLayerKVCache::new(&config);
    let mut gen_rng = Rng::new(123);

    let generated = generate_with_prefill(
        &mut ctx,
        &mut prefill,
        &weights,
        &mut cache,
        &config,
        &mut gen_rng,
        &prompt_tokens,
        max_gen,
        &lora_pair,
    );

    let rust_output = BpeTokenizerImpl::decode(&tokenizer, &generated);

    println!();
    println!("  Step 2: Causal Decode");
    println!("    LoRA:    Writer (Rust, seed=200) ← SWITCHED!");
    println!("    Attn:    LEFT → RIGHT causal");
    println!("    Output:  {} tokens", generated.len());
    println!();
    println!("  ┌─────────────────────────────────────────┐");
    println!("  │  Python Input:                           │");
    for line in python_input.lines() {
        println!("  │    {line:<37} │",);
    }
    println!("  │                                          │");
    println!("  │  Rust Output (random weights):            │");
    println!(
        "  │    {output:<37} │",
        output = rust_output.chars().take(37).collect::<String>()
    );
    if rust_output.len() > 37 {
        println!(
            "  │    {output:<37} │",
            output = rust_output.chars().skip(37).take(37).collect::<String>()
        );
    }
    println!("  └─────────────────────────────────────────┘");
    println!();

    // ── Phase 6: Proof — Bidirectional ≠ Causal ──────────────────

    println!("━━━ Proof 1: Bidirectional Prefill ≠ Causal ━━━");
    println!();

    // Recreate fresh weights for fair comparison
    let mut rng_c = Rng::new(42);
    let weights_c = TransformerWeights::new(&config, &mut rng_c);

    // Causal: process Python tokens one at a time
    let mut ctx_c = ForwardContext::new(&config);
    let mut cache_c = MultiLayerKVCache::new(&config);
    let mut causal_logits = vec![0.0f32; config.vocab_size];

    for (i, &tok) in prompt_tokens.iter().enumerate() {
        let logits = forward(&mut ctx_c, &weights_c, &mut cache_c, tok, i, &config);
        if i == prompt_tokens.len() - 1 {
            causal_logits.copy_from_slice(logits);
        }
    }

    // Bidirectional: all Python tokens see each other
    let mut rng_b = Rng::new(42);
    let weights_b = TransformerWeights::new(&config, &mut rng_b);
    let mut ctx_b = ForwardContext::new(&config);
    let mut pf_b = PrefillContext::new(&config);
    let mut cache_b = MultiLayerKVCache::new(&config);

    let bi_logits = forward_prefill(
        &mut ctx_b,
        &mut pf_b,
        &weights_b,
        &mut cache_b,
        &prompt_tokens,
        &config,
        None,
        None,
    );

    let max_diff: f32 = causal_logits
        .iter()
        .zip(bi_logits.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    let mean_diff: f32 = causal_logits
        .iter()
        .zip(bi_logits.iter())
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>()
        / causal_logits.len() as f32;

    println!(
        "  Causal logits[0..5]:       {:.4} {:.4} {:.4} {:.4} {:.4}",
        causal_logits[0], causal_logits[1], causal_logits[2], causal_logits[3], causal_logits[4]
    );
    println!(
        "  Bidirectional logits[0..5]: {:.4} {:.4} {:.4} {:.4} {:.4}",
        bi_logits[0], bi_logits[1], bi_logits[2], bi_logits[3], bi_logits[4]
    );
    println!("  Max logit diff:  {max_diff:.4}");
    println!("  Mean logit diff: {mean_diff:.4}");

    if max_diff > 0.01 {
        println!("  ✅ PROVEN: Bidirectional prefill diverges from causal");
        println!("     → Python code processed holistically, not left-to-right");
    }
    println!();

    // ── Phase 7: Proof — LoRA Switch Changes Output ──────────────

    println!("━━━ Proof 2: LoRA Switch Changes Output ━━━");
    println!();

    // Without switch: reader→reader (same LoRA for both phases)
    let lora_no_switch = LoraPair {
        reader: Some(make_lora(&config, 100)),
        writer: Some(make_lora(&config, 100)), // same!
    };

    let mut ctx_ns = ForwardContext::new(&config);
    let mut pf_ns = PrefillContext::new(&config);
    let mut cache_ns = MultiLayerKVCache::new(&config);
    let mut rng_ns = Rng::new(123);

    let generated_ns = generate_with_prefill(
        &mut ctx_ns,
        &mut pf_ns,
        &weights,
        &mut cache_ns,
        &config,
        &mut rng_ns,
        &prompt_tokens,
        max_gen,
        &lora_no_switch,
    );

    let output_ns = BpeTokenizerImpl::decode(&tokenizer, &generated_ns);

    // Compare token-by-token
    let compare_len = generated.len().min(generated_ns.len()).min(16);
    let matching: usize = generated
        .iter()
        .zip(generated_ns.iter())
        .take(compare_len)
        .filter(|(a, b)| a == b)
        .count();

    println!(
        "  With switch    (reader→writer): \"{}\"",
        rust_output.chars().take(50).collect::<String>()
    );
    println!(
        "  Without switch (reader→reader): \"{}\"",
        output_ns.chars().take(50).collect::<String>()
    );
    println!(
        "  Token match:   {matching}/{compare_len} ({:.0}%)",
        if compare_len > 0 {
            matching as f32 / compare_len as f32 * 100.0
        } else {
            0.0
        }
    );

    if matching < compare_len {
        println!("  ✅ PROVEN: LoRA switch changes generation output");
    } else {
        println!("  ⚠️  Outputs happen to match (random weights, low LoRA rank)");
    }
    println!();

    // ── Phase 8: Proof — BPE Roundtrip ────────────────────────────

    println!("━━━ Proof 3: BPE Tokenizer Roundtrip ━━━");
    println!();

    let test_snippets = [
        ("Python", "def add(a, b):\n    return a + b"),
        ("Rust", "fn add(a: i32, b: i32) -> i32 {\n    a + b\n}"),
        ("Rust", "let x: i64 = 42;"),
        ("Python", "for i in range(n):"),
    ];

    for (lang, code) in &test_snippets {
        let ids = BpeTokenizerImpl::encode(&tokenizer, code);
        let decoded = BpeTokenizerImpl::decode(&tokenizer, &ids);
        let ok = decoded == *code;
        println!(
            "  [{lang:6}] {} tokens  {}  \"{code}\"",
            ids.len(),
            if ok { "✓ roundtrip" } else { "≠ lossy" }
        );
        if !ok {
            println!("           decoded: \"{decoded}\"");
        }
    }
    println!();

    // ── Summary ───────────────────────────────────────────────────

    println!("╔═══════════════════════════════════════════════════════════════════╗");
    println!("║  Summary                                                         ║");
    println!("║                                                                   ║");
    println!("║  ✅ BPE tokenizer trained on Python + Rust ({actual_vocab} tokens)      ║");
    println!(
        "║  ✅ Real Python code → {:<3} BPE tokens                          ",
        prompt_tokens.len()
    );
    println!("║  ✅ Bidirectional prefill (reader LoRA) → causal decode (writer)  ║");
    println!("║  ✅ Generated tokens decoded back to real text                    ║");
    println!("║  ✅ Bidirectional ≠ causal (max diff {max_diff:.4})                     ║",);
    println!("║                                                                   ║");
    println!("║  Honest note: output is gibberish because weights are random.    ║");
    println!("║  With trained weights, this pipeline transpiles Python → Rust.   ║");
    println!("║                                                                   ║");
    println!("║  To get real transpilation:                                       ║");
    println!("║  1. Train LoRA weights on (Python, Rust) pairs                   ║");
    println!("║  2. Use a 7B+ parameter model                                    ║");
    println!("║  3. Inject anyRAG API docs into prompt                            ║");
    println!("╚═══════════════════════════════════════════════════════════════════╝");
}
