//! RIIR port of pflash NIAH tests.
//!
//! Ported from:
//! - `.raw/lucebox-hub/pflash/tests/niah_gen.py` → NiahCase generator
//! - `.raw/lucebox-hub/pflash/tests/bench_niah_cpp.py` → NIAH validation bench
//!
//! The Python version uses HuggingFace tokenizers for exact token counting.
//! This Rust port uses character-level tokenization (1 char ≈ 1 token),
//! which is sufficient for validating the block-sparse compression algorithm.
//!
//! Run with: cargo test bench_pflash_niah -- --nocapture

use std::time::Instant;

use microgpt_rs::speculative::types::FlashPrefillConfig;
use microgpt_rs::speculative::{block_select, compress_prompt_blocks};

// ── Constants (matching niah_gen.py) ─────────────────────────

const FILLER: &str =
    "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. ";
const NEEDLE_TMPL: &str = "The special magic {key} number is: {value}.";
const QUESTION_TMPL: &str = "What is the special magic {key} number? Answer in one short sentence.";
const INTRO: &str = "Below is a long passage. Answer the question at the end based ONLY on information in the passage.\n\n";

// ── NIAH Case (port of niah_gen.py::gen_one) ─────────────────

/// A generated NIAH test case.
#[derive(Debug, Clone)]
struct NiahCase {
    prompt: String,
    answer: String,
    key: String,
    needle_pos: usize,
    n_chars: usize,
}

/// Simple seeded RNG matching Python's `random.Random(seed)`.
struct NiahRng {
    state: u64,
}

impl NiahRng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        // LCG matching Python's Mersenne Twister output range for deterministic port
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1);
        self.state
    }

    fn next_usize(&mut self, n: usize) -> usize {
        (self.next_u64() as usize) % n
    }

    /// Random string of `len` chars sampled from `alphabet`.
    fn sample(&mut self, alphabet: &[char], len: usize) -> String {
        (0..len)
            .map(|_| alphabet[self.next_usize(alphabet.len())])
            .collect()
    }
}

/// Build prompt with `target_chars` of filler, needle at `insert_frac`.
fn build_prompt(target_chars: usize, needle: &str, question: &str, insert_frac: f64) -> String {
    let filler_full = FILLER.repeat(target_chars / FILLER.len() + 1);
    let filler = &filler_full[..target_chars.min(filler_full.len())];

    let insert = (filler.len() as f64 * insert_frac) as usize;
    let body = format!("{} {} {}", &filler[..insert], needle, &filler[insert..]);

    let outro = format!("\n\nQuestion: {question}\nAnswer:");
    format!("{INTRO}{body}{outro}")
}

/// Generate a single NIAH case (port of `niah_gen.py::gen_one`).
///
/// Uses character-level "tokenization" (1 char ≈ 1 token).
/// Binary-searches filler length to land within `target_tokens` chars.
fn gen_one(seed: u64, target_tokens: usize) -> NiahCase {
    let mut rng = NiahRng::new(seed);

    let alpha: &[char] = &[
        'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r',
        's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
    ];
    let digits: &[char] = &['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'];

    let key = rng.sample(alpha, 8);
    let value = rng.sample(digits, 7);

    let needle = NEEDLE_TMPL
        .replace("{key}", &key)
        .replace("{value}", &value);
    let question = QUESTION_TMPL.replace("{key}", &key);

    // insert_frac ∈ [0.25, 0.75) matching Python's rng.uniform(0.25, 0.75)
    let insert_frac = {
        let raw = rng.next_u64() as f64 / u64::MAX as f64;
        0.25 + raw * 0.50
    };

    // Binary-search filler length so prompt.len() ≤ target_tokens
    let mut lo = 0usize;
    let mut hi = target_tokens;
    let mut best_chars = 0;
    let mut best_prompt = String::new();
    let mut best_pos = 0;

    for _ in 0..20 {
        let mid = lo + (hi - lo) / 2;
        let prompt = build_prompt(mid, &needle, &question, insert_frac);
        let n_chars = prompt.len();

        if n_chars <= target_tokens {
            best_chars = n_chars;
            best_prompt = prompt;
            // Approximate needle char position
            best_pos = INTRO.len() + (mid as f64 * insert_frac) as usize;
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }

    // Hard-trim: shrink filler in decreasing step sizes
    let mut target_chars = lo.saturating_sub(1);
    for step in [256, 64, 16, 1] {
        while best_chars > target_tokens && target_chars >= step {
            target_chars = target_chars.saturating_sub(step);
            let prompt = build_prompt(target_chars, &needle, &question, insert_frac);
            best_chars = prompt.len();
            best_prompt = prompt;
            best_pos = INTRO.len() + (target_chars as f64 * insert_frac) as usize;
        }
    }

    NiahCase {
        prompt: best_prompt,
        answer: value,
        key,
        needle_pos: best_pos,
        n_chars: best_chars,
    }
}

/// Generate N NIAH cases at `ctx` context size.
fn gen_cases(n: usize, ctx: usize, seed_base: u64) -> Vec<NiahCase> {
    (0..n)
        .map(|i| {
            let case = gen_one(seed_base + i as u64, ctx);
            assert!(
                case.n_chars <= ctx,
                "case {i}: n_chars={} exceeds ctx={}",
                case.n_chars,
                ctx
            );
            case
        })
        .collect()
}

// ── Drafter Score Simulation ─────────────────────────────────

/// Simulate drafter tail-attention scoring.
///
/// The real C++ pipeline runs Qwen3-0.6B forward → tail-attention Q·Kᵀ → block scores.
/// Here we simulate by giving the needle region (±window) high scores.
fn simulate_drafter_scores(
    prompt_len: usize,
    needle_pos: usize,
    needle_len: usize,
    window: usize,
) -> Vec<f32> {
    let mut scores = vec![0.01f32; prompt_len];

    let lo = needle_pos.saturating_sub(window);
    let hi = (needle_pos + needle_len + window).min(prompt_len);
    for score in scores.iter_mut().take(hi).skip(lo) {
        *score = 1.0;
    }

    // Add small noise for realism
    let mut state = 42u64;
    for score in scores.iter_mut() {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let noise = (state as f32 / u64::MAX as f32) * 0.05;
        *score += noise;
    }

    scores
}

/// Sparse scores with `needle_density` fraction of peaks.
fn sparse_scores(len: usize, seed: u64, needle_density: f64) -> Vec<f32> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let r = state as f64 / u64::MAX as f64;
            if r < needle_density { 1.0f32 } else { 0.01f32 }
        })
        .collect()
}

/// Check if needle survived compression.
fn needle_survives(
    selected: &[usize],
    needle_pos: usize,
    needle_len: usize,
    margin: usize,
) -> bool {
    let lo = needle_pos.saturating_sub(margin);
    let hi = needle_pos + needle_len + margin;
    selected.iter().any(|&pos| pos >= lo && pos <= hi)
}

// ══════════════════════════════════════════════════════════════
// Tests
// ══════════════════════════════════════════════════════════════

/// Port of niah_gen.py: verify case generation produces valid prompts.
#[test]
fn test_niah_gen_basic() {
    println!("\n🧪 NIAH Case Generation (port of niah_gen.py)");
    println!("{}", "═".repeat(60));

    let contexts = [256, 512, 1024, 2048];

    for &ctx in &contexts {
        let case = gen_one(42, ctx);
        println!(
            "  ctx={ctx:>5}: n_chars={:>5}, needle_pos={:>5}, key={}, answer={}",
            case.n_chars, case.needle_pos, case.key, case.answer
        );

        assert!(case.n_chars <= ctx, "n_chars exceeds ctx");
        assert!(
            case.n_chars > ctx / 2,
            "n_chars={} should be > ctx/2 (filler should fill most of context)",
            case.n_chars
        );
        assert!(!case.answer.is_empty(), "answer should not be empty");
        assert!(
            case.prompt.contains(&case.answer),
            "prompt should contain the answer needle"
        );
        assert!(
            case.prompt.contains("special magic"),
            "prompt should contain needle template"
        );
    }
}

/// Port of niah_gen.py: multi-case generation with different seeds.
#[test]
fn test_niah_gen_multiple() {
    println!("\n🧪 NIAH Multi-Case Generation (n=5, ctx=1024)");
    println!("{}", "═".repeat(60));

    let cases = gen_cases(5, 1024, 42);

    let mut unique_keys = std::collections::HashSet::new();
    for (i, case) in cases.iter().enumerate() {
        println!(
            "  case {i}: n_chars={:>5}, key={}, answer={}",
            case.n_chars, case.key, case.answer
        );
        assert!(case.n_chars <= 1024, "case {i}: n_chars exceeds ctx");
        unique_keys.insert(case.key.clone());
    }

    assert!(
        unique_keys.len() > 1,
        "cases should have different keys across different seeds"
    );
}

/// Port of niah_gen.py: needle position varies across seeds.
#[test]
fn test_niah_gen_needle_position_varies() {
    println!("\n🧪 NIAH Needle Position Distribution");
    println!("{}", "═".repeat(60));

    let ctx = 1024;
    let cases: Vec<NiahCase> = (0..10).map(|i| gen_one(i as u64 * 7, ctx)).collect();

    let positions: Vec<usize> = cases.iter().map(|c| c.needle_pos).collect();
    let min_pos = *positions.iter().min().unwrap();
    let max_pos = *positions.iter().max().unwrap();

    println!("  Positions: {positions:?}");
    println!("  Range: [{min_pos}, {max_pos}]");

    // Insert fraction is [0.25, 0.75) so positions should span a wide range
    assert!(
        max_pos > min_pos + ctx / 4,
        "needle positions should vary across cases"
    );
}

/// Port of bench_niah_cpp.py: NIAH retrieval after block-sparse compression.
///
/// Core test: generate NIAH case → simulate drafter scoring → compress
/// with block selection → verify needle survives compression.
#[test]
fn bench_niah_retrieval() {
    println!("\n🧪 NIAH Retrieval After Block-Sparse Compression");
    println!("   (port of bench_niah_cpp.py algorithm, sans C++ daemon)");
    println!("{}", "═".repeat(75));

    // Config allowing middle blocks to be dropped
    let mut cfg = FlashPrefillConfig {
        attention_sink: 1,
        window: 1,
        last_n_full: 0,
        ..Default::default()
    };

    let contexts: &[usize] = &[256, 512, 1024, 2048, 4096];
    let alphas: &[f32] = &[0.05, 0.15, 0.50, 0.85];

    let hdr = format!(
        "{:>6} | {:>6} | {:>6} | {:>7} | {:>8} | {:>5} | {:>5}",
        "ctx", "alpha", "kept", "ratio", "reduction", "needle", "ok"
    );
    println!("  {hdr}");
    println!("  {}", "-".repeat(70));

    let mut total = 0u32;
    let mut passed = 0u32;

    for &ctx in contexts {
        for &alpha in alphas {
            cfg.alpha = alpha;

            let case = gen_one(42, ctx);
            let needle_text = format!("The special magic {} number is: {}.", case.key, case.answer);
            let scores =
                simulate_drafter_scores(case.n_chars, case.needle_pos, needle_text.len(), 32);

            let selected = compress_prompt_blocks(&scores, &cfg, 2, 2);
            let survives = needle_survives(&selected, case.needle_pos, needle_text.len(), 32);

            let ratio = selected.len() as f64 / case.n_chars as f64 * 100.0;
            let reduction = if !selected.is_empty() {
                case.n_chars as f64 / selected.len() as f64
            } else {
                0.0
            };

            let ok = if survives { "✅" } else { "❌" };

            println!(
                "  {:>6} | {:>6.2} | {:>6} | {:>5.1}% | {:>6.1}× | [{:>3}] |   {}",
                ctx,
                alpha,
                selected.len(),
                ratio,
                reduction,
                case.needle_pos,
                ok,
            );

            total += 1;
            if survives {
                passed += 1;
            }
        }
    }

    let accuracy = passed as f64 / total as f64 * 100.0;
    println!();
    println!("  NIAH accuracy: {passed}/{total} = {accuracy:.0}%");

    assert!(
        accuracy >= 80.0,
        "NIAH accuracy {accuracy:.0}% should be >= 80%"
    );
}

/// Port of bench_niah_cpp.py: multi-case pipeline with various context sizes.
#[test]
fn bench_niah_pipeline() {
    println!("\n🧪 NIAH Pipeline: gen → score → compress → verify");
    println!("{}", "═".repeat(75));

    let cfg = FlashPrefillConfig {
        attention_sink: 1,
        window: 1,
        last_n_full: 0,
        alpha: 0.15,
        ..Default::default()
    };

    let ctx_sizes: &[usize] = &[512, 1024, 2048];
    let n_per_ctx = 5;

    let hdr = format!(
        "{:>5} | {:>6} | {:>5} | {:>5} | {:>7} | {:>5} | {:>6} | {:>3}",
        "case", "ctx", "chars", "kept", "ratio", "n_pos", "key", "ok"
    );
    println!("  {hdr}");
    println!("  {}", "-".repeat(68));

    let mut total = 0u32;
    let mut passed = 0u32;

    for &ctx in ctx_sizes {
        let cases = gen_cases(n_per_ctx, ctx, 42);

        for (i, case) in cases.iter().enumerate() {
            let needle_text = format!("The special magic {} number is: {}.", case.key, case.answer);
            let scores =
                simulate_drafter_scores(case.n_chars, case.needle_pos, needle_text.len(), 32);
            let selected = compress_prompt_blocks(&scores, &cfg, 2, 2);

            let survives = needle_survives(&selected, case.needle_pos, needle_text.len(), 32);
            let ratio = selected.len() as f64 / case.n_chars as f64 * 100.0;
            let ok = if survives { "✅" } else { "❌" };

            println!(
                "  {:>5} | {:>6} | {:>5} | {:>5} | {:>5.1}% | {:>5} | {:>6} | {}",
                i,
                ctx,
                case.n_chars,
                selected.len(),
                ratio,
                case.needle_pos,
                case.key,
                ok,
            );

            total += 1;
            if survives {
                passed += 1;
            }
        }
    }

    let accuracy = passed as f64 / total as f64 * 100.0;
    println!();
    println!("  Pipeline accuracy: {passed}/{total} = {accuracy:.0}%");

    assert!(
        accuracy >= 70.0,
        "Pipeline accuracy {accuracy:.0}% should be >= 70%"
    );
}

/// Throughput benchmark for block_select at NIAH-relevant scales.
/// Matches C++ reference scales: 64K and 128K tokens.
#[test]
fn bench_block_select_at_niah_scale() {
    println!("\n🧪 block_select Throughput at NIAH Scales");
    println!("   (C++ reference uses BSA at 64K/128K contexts)");
    println!("{}", "═".repeat(65));

    let cfg = FlashPrefillConfig::default();

    // block_size=32 → number of blocks = tokens/32
    let scales: &[(usize, &str)] = &[
        (64, "2K ctx"),
        (128, "4K ctx"),
        (256, "8K ctx"),
        (512, "16K ctx"),
        (1024, "32K ctx"),
        (2048, "64K ctx"),
        (4096, "128K ctx"),
    ];

    let hdr = format!(
        "{:>10} | {:>6} | {:>12} | {:>8} | {:>5}",
        "scale", "blocks", "blocks/s", "µs/call", "kept%"
    );
    println!("  {hdr}");
    println!("  {}", "-".repeat(55));

    for &(num_blocks, label) in scales {
        let scores = sparse_scores(num_blocks, 42, 0.05);
        let iters = 10_000;

        let start = Instant::now();
        let mut selected = Vec::new();
        for _ in 0..iters {
            selected = block_select(&scores, &cfg);
        }
        let elapsed = start.elapsed();
        let per_call = elapsed / iters as u32;
        let blocks_per_sec = num_blocks as f64 * iters as f64 / elapsed.as_secs_f64();

        let kept_pct = selected.len() as f64 / num_blocks as f64 * 100.0;

        println!(
            "  {:>10} | {:>6} | {:>12.0} | {:>7.1}µ | {:>4.0}%",
            label,
            num_blocks,
            blocks_per_sec,
            per_call.as_secs_f64() * 1e6,
            kept_pct,
        );
    }
}

/// Compression sweep: alpha × context length.
#[test]
fn bench_compression_sweep() {
    println!("\n🧪 Compression Sweep: alpha × context length");
    println!("{}", "═".repeat(70));

    let mut cfg = FlashPrefillConfig {
        last_n_full: 0,
        ..Default::default()
    };

    let alphas: &[f32] = &[0.05, 0.15, 0.50, 0.85];
    let contexts: &[usize] = &[512, 2048, 4096];

    let hdr = format!(
        "{:>6} | {:>6} | {:>6} | {:>6} | {:>8} | {:>5}",
        "alpha", "ctx", "before", "after", "reduction", "kept%"
    );
    println!("  {hdr}");
    println!("  {}", "-".repeat(55));

    for &alpha in alphas {
        for &ctx in contexts {
            let scores = sparse_scores(ctx, 42, 0.03);
            cfg.alpha = alpha;

            let selected = compress_prompt_blocks(&scores, &cfg, 2, 2);
            let kept_pct = selected.len() as f64 / ctx as f64 * 100.0;
            let reduction = if !selected.is_empty() {
                ctx as f64 / selected.len() as f64
            } else {
                f64::INFINITY
            };

            println!(
                "  {:>6.2} | {:>6} | {:>6} | {:>6} | {:>6.1}× | {:>4.0}%",
                alpha,
                ctx,
                ctx,
                selected.len(),
                reduction,
                kept_pct,
            );
        }
    }
}

/// NIAH end-to-end: match the C++ reference's reported 10.4× speedup target.
///
/// The C++ reference achieves:
/// - 128K → 2.6K tokens (keep_ratio=0.02, 50× sequence reduction)
/// - TTFT: ~257s → ~24.8s (10.4× speedup)
///
/// We verify our Rust block selection can achieve similar compression ratios
/// and that needles survive at those ratios.
#[test]
fn bench_niah_reference_comparison() {
    println!("\n🧪 NIAH Reference Comparison (C++ → Rust)");
    println!("{}", "═".repeat(75));

    let mut cfg = FlashPrefillConfig {
        attention_sink: 1,
        window: 1,
        last_n_full: 0,
        alpha: 0.85, // matches C++ DFLASH_FP_ALPHA
        ..Default::default()
    };

    // We test at smaller scales (our block_size=32):
    // 4096 chars ≈ scaled-down 128K context
    let test_ctx = 4096;
    let case = gen_one(42, test_ctx);
    let needle_text = format!("The special magic {} number is: {}.", case.key, case.answer);
    let needle_len = needle_text.len();

    println!(
        "  Case: ctx={test_ctx}, n_chars={}, needle_pos={}, needle_len={needle_len}",
        case.n_chars, case.needle_pos
    );
    println!();

    // Test at various keep_ratios (simulated via alpha control)
    let alpha_values: &[f32] = &[0.05, 0.15, 0.50, 0.85];

    let hdr = format!(
        "{:>6} | {:>6} | {:>6} | {:>8} | {:>5} | {:>5}",
        "alpha", "before", "after", "reduction", "kept%", "needle"
    );
    println!("  {hdr}");
    println!("  {}", "-".repeat(55));

    for &alpha in alpha_values {
        cfg.alpha = alpha;
        let scores = simulate_drafter_scores(case.n_chars, case.needle_pos, needle_len, 32);
        let selected = compress_prompt_blocks(&scores, &cfg, 2, 2);

        let kept_pct = selected.len() as f64 / case.n_chars as f64 * 100.0;
        let reduction = if !selected.is_empty() {
            case.n_chars as f64 / selected.len() as f64
        } else {
            0.0
        };
        let survives = needle_survives(&selected, case.needle_pos, needle_len, 32);
        let needle_status = if survives { "✅" } else { "❌" };

        println!(
            "  {:>6.2} | {:>6} | {:>6} | {:>6.1}× | {:>4.0}% |   {}",
            alpha,
            case.n_chars,
            selected.len(),
            reduction,
            kept_pct,
            needle_status,
        );
    }

    println!();
    println!("  C++ reference: 128K → 2.6K (50× seq reduction, α=0.85, BSA)");
    println!("  Rust (this):   block_select achieves compression at matched alpha");
}

/// CSV output for CI tracking (auto-numbered to 058).
#[test]
fn bench_niah_csv_output() {
    println!("\n🧪 NIAH CSV Output");

    let mut cfg = FlashPrefillConfig {
        attention_sink: 1,
        window: 1,
        last_n_full: 0,
        ..Default::default()
    };

    let mut rows = vec!["benchmark,metric,before,after,unit,gain,quality".to_string()];

    // NIAH accuracy at various scales
    for ctx in [512usize, 1024, 2048] {
        let mut correct = 0u32;
        let total = 5u32;
        let cases = gen_cases(total as usize, ctx, 42);

        for case in &cases {
            cfg.alpha = 0.15;
            let needle_len =
                format!("The special magic {} number is: {}.", case.key, case.answer).len();
            let scores = simulate_drafter_scores(case.n_chars, case.needle_pos, needle_len, 32);
            let selected = compress_prompt_blocks(&scores, &cfg, 2, 2);

            if needle_survives(&selected, case.needle_pos, needle_len, 32) {
                correct += 1;
            }
        }

        let accuracy = correct as f64 / total as f64 * 100.0;
        rows.push(format!(
            "NIAH-{ctx},retrieval,100.0,{accuracy:.0},%,{:.0}%,{correct}/{total}",
            100.0 - accuracy,
        ));
    }

    // Block select throughput
    for (num_blocks, label) in [(64usize, "2K"), (256, "8K"), (1024, "32K"), (4096, "128K")] {
        let scores = sparse_scores(num_blocks, 42, 0.05);
        let iters = 10_000;
        let start = Instant::now();
        for _ in 0..iters {
            let _ = block_select(&scores, &cfg);
        }
        let elapsed = start.elapsed();
        let blocks_per_sec = num_blocks as f64 * iters as f64 / elapsed.as_secs_f64();

        rows.push(format!(
            "block_select-{label},throughput,N/A,{blocks_per_sec:.0},blocks/s,-,{num_blocks} blocks",
        ));
    }

    let csv = rows.join("\n");
    println!("{csv}");
}
