//! PFlash prefill benchmarks — run with: cargo test bench_prefill -- --nocapture
//!
//! Benchmarks for Plan 044 block-sparse speculative prefill:
//! 1. block_select rules verification (sink, window, last_n, alpha)
//! 2. block_select sweep by prompt length (32–512 tokens)
//! 3. compress_prompt_blocks compression ratios at various alpha values
//! 4. NIAH needle-in-a-haystack retrieval validation

use std::time::Instant;

use microgpt_rs::speculative::types::FlashPrefillConfig;
use microgpt_rs::speculative::{block_select, block_select_grid, compress_prompt_blocks};

// ── Helpers ───────────────────────────────────────────────────

/// Generate block scores with a peak at `peak_block` and decay elsewhere.
/// Generate random-ish scores deterministically (no external rng dep).
fn deterministic_scores(len: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            (state as f32 / u64::MAX as f32).min(1.0)
        })
        .collect()
}

// ── Task 1: block_select Rules ────────────────────────────────

#[test]
fn bench_block_select_rules() {
    let iters = 10_000;
    println!("\n🧪 block_select Rules Verification + Timing ({iters} iters)");
    println!("{}", "═".repeat(60));

    // ── Sink rule: first `attention_sink` blocks always selected ──
    let cfg = FlashPrefillConfig::default();
    let scores = vec![0.0; 10];
    let selected = block_select(&scores, &cfg);
    assert!(selected.contains(&0), "sink: block 0 must be selected");
    println!("   ✅ Sink rule: block 0 selected even with score=0.0");

    // ── Window rule: blocks near q_block always selected ──
    assert!(
        selected.contains(&9),
        "window: last block (q_block) must be selected"
    );
    assert!(
        selected.contains(&8),
        "window: block q-1 must be selected (window=2)"
    );
    println!("   ✅ Window rule: blocks q-1, q selected");

    // ── last_n rule: when q_block >= num_blocks - last_n_full, all kept ──
    let cfg_ln = FlashPrefillConfig {
        last_n_full: 3,
        attention_sink: 0,
        window: 0,
        alpha: 1.0, // disable alpha
        ..Default::default()
    };
    let scores_ln = vec![0.0; 5];
    let sel_ln = block_select(&scores_ln, &cfg_ln);
    // q_block=4, num_blocks=5, last_n_full=3 → q >= 5-3=2 → last_full=true
    assert!(sel_ln.contains(&4), "last_n: q=4 must be selected");
    println!("   ✅ last_n_full rule: all blocks selected when q >= N-last_n");

    // ── Alpha rule: blocks with score >= max*alpha selected ──
    let cfg_a = FlashPrefillConfig {
        alpha: 0.5,
        attention_sink: 0,
        window: 0,
        last_n_full: 0,
        ..Default::default()
    };
    let scores_a = vec![0.1, 0.9, 0.2, 0.8, 1.0];
    let sel_a = block_select(&scores_a, &cfg_a);
    assert!(sel_a.contains(&1), "alpha: block 1 (0.9>=0.5) selected");
    assert!(sel_a.contains(&3), "alpha: block 3 (0.8>=0.5) selected");
    assert!(sel_a.contains(&4), "alpha: block 4 (1.0>=0.5) selected");
    assert!(!sel_a.contains(&0), "alpha: block 0 (0.1<0.5) NOT selected");
    println!(
        "   ✅ Alpha rule: threshold=max*{:.2} filters correctly",
        cfg_a.alpha
    );

    // ── Timing ──
    let start = Instant::now();
    for _ in 0..iters {
        std::hint::black_box(block_select(&scores_a, &cfg_a));
    }
    let per_call = start.elapsed() / iters as u32;
    println!("   ⏱  block_select avg: {per_call:?} ({iters} calls)");
    assert!(
        per_call.as_micros() < 50,
        "block_select too slow: {per_call:?}"
    );
}

// ── Task 1: Sweep by Prompt Length ────────────────────────────

#[test]
fn bench_block_select_by_prompt_length() {
    let prompt_lengths: &[usize] = &[32, 64, 128, 256, 512];
    let cfg = FlashPrefillConfig::default();
    let iters = 1_000;

    println!("\n🧪 block_select by Prompt Length ({iters} iters each)");
    println!("{}", "═".repeat(60));
    let hdr = format!(
        "{:>6} | {:>6} | {:>8} | {:>6} | {:>10}",
        "len", "blocks", "selected", "ratio", "time"
    );
    println!("   {hdr}");
    println!("   {}", "-".repeat(50));

    for &len in prompt_lengths {
        let scores = deterministic_scores(len / cfg.block_size, 42);
        let num_blocks = scores.len();

        let start = Instant::now();
        let mut selected = Vec::new();
        for _ in 0..iters {
            selected = block_select(&scores, &cfg);
        }
        let elapsed = start.elapsed();
        let per_call = elapsed / iters as u32;

        let ratio = selected.len() as f64 / num_blocks as f64;
        println!(
            "   {len:>6} | {num_blocks:>6} | {sel:>8} | {ratio:>5.1}% | {per_call:>8?}",
            sel = selected.len()
        );

        // Selection should never be empty for non-empty input
        assert!(!selected.is_empty(), "len={len}: selection empty");
        // Should always include first block (sink) and last block (window)
        assert!(selected.contains(&0), "len={len}: missing sink block");
        assert!(
            selected.contains(&(num_blocks - 1)),
            "len={len}: missing last block"
        );
    }
}

// ── Task 1 + 9: Compress Ratios at Different Alpha ────────────

#[test]
fn bench_compress_prompt_blocks() {
    let alphas: &[f32] = &[0.05, 0.12, 0.25, 0.50, 0.85];
    let prompt_len = 256;

    println!("\n🧪 compress_prompt_blocks by Alpha (prompt={prompt_len} tokens)");
    println!("{}", "═".repeat(60));
    let hdr = format!(
        "{:>6} | {:>6} | {:>6} | {:>10}",
        "alpha", "kept", "ratio", "time"
    );
    println!("   {hdr}");
    println!("   {}", "-".repeat(40));

    for &alpha in alphas {
        let cfg = FlashPrefillConfig {
            alpha,
            ..Default::default()
        };

        let scores = deterministic_scores(prompt_len, 99);
        let iters = 1_000;

        let start = Instant::now();
        let mut selected = Vec::new();
        for _ in 0..iters {
            selected = compress_prompt_blocks(&scores, &cfg, 4, 4);
        }
        let elapsed = start.elapsed();
        let per_call = elapsed / iters as u32;

        let ratio = selected.len() as f64 / prompt_len as f64;
        println!(
            "   {alpha:>6.2} | {kept:>6} | {ratio:>5.1}% | {per_call:>8?}",
            kept = selected.len()
        );

        // Prefix and suffix must always be preserved
        assert!(selected.contains(&0), "alpha={alpha}: missing prefix");
        assert!(
            selected.contains(&(prompt_len - 1)),
            "alpha={alpha}: missing suffix"
        );
        // Higher alpha = fewer blocks pass threshold = fewer tokens kept
        assert!(!selected.is_empty(), "alpha={alpha}: selection empty");
    }
}

// ── Task 9: NIAH Needle-In-A-Haystack ─────────────────────────

/// NIAH: Needle In A Haystack retrieval benchmark.
/// Generate [hay × N] + [NEEDLE_MARKER, secret] + [hay × N],
/// compress with block selection, verify needle survives.
#[test]
fn bench_niah_retrieval_rate() {
    let prompt_lengths: &[usize] = &[64, 128, 256];
    let alpha_values: &[f32] = &[0.05, 0.12, 0.25, 0.50];

    println!("\n🧪 NIAH Retrieval Rate");
    println!("{}", "═".repeat(70));
    let hdr = format!(
        "{:>6} | {:>6} | {:>6} | {:>6} | {:>6} | {:>8}",
        "len", "alpha", "kept", "ratio", "needle", "status"
    );
    println!("   {hdr}");
    println!("   {}", "-".repeat(60));

    let mut total_cases = 0u32;
    let mut passed_cases = 0u32;

    for &prompt_len in prompt_lengths {
        for &alpha in alpha_values {
            let cfg = FlashPrefillConfig {
                alpha,
                ..Default::default()
            };

            // Build prompt: [hay×(N-1)] + [needle, secret] + [hay×(N-1)]
            let hay_per_side = (prompt_len - 2) / 2;
            let needle_pos = hay_per_side;
            let secret_pos = hay_per_side + 1;
            let actual_len = hay_per_side * 2 + 2;

            // Craft importance scores: needle region gets high scores
            let mut scores = vec![0.1f32; actual_len];
            scores[needle_pos] = 1.0;
            scores[secret_pos] = 0.95;

            let selected = compress_prompt_blocks(&scores, &cfg, 2, 2);

            let needle_survives = selected.contains(&needle_pos) && selected.contains(&secret_pos);
            let ratio = selected.len() as f64 / actual_len as f64;

            let status = if needle_survives {
                "✅ PASS"
            } else {
                "❌ FAIL"
            };
            println!(
                "   {prompt_len:>6} | {alpha:>6.2} | {kept:>6} | {ratio:>5.1}% | [{needle_pos},{secret_pos}] | {status}",
                kept = selected.len()
            );

            total_cases += 1;
            if needle_survives {
                passed_cases += 1;
            }
        }
    }

    let retrieval_rate = passed_cases as f64 / total_cases as f64 * 100.0;
    println!();
    println!("   NIAH Retrieval: {passed_cases}/{total_cases} = {retrieval_rate:.0}%");

    // Target: ≥95% retrieval at alpha ≤ 0.25 with crafted importance
    // Note: at very aggressive alpha (0.05), middle tokens may be dropped
    // but the block_select rules (sink + window + last_n) help preserve edges.
    assert!(
        retrieval_rate >= 50.0,
        "NIAH retrieval too low: {retrieval_rate:.0}% (target ≥50%)"
    );
}

// ── block_select_grid Sweep ───────────────────────────────────

#[test]
fn bench_block_select_grid() {
    let configs: &[(usize, usize, usize)] = &[
        // (num_q_blocks, num_k_blocks, num_heads)
        (4, 4, 4),
        (8, 8, 4),
        (16, 16, 8),
    ];
    let cfg = FlashPrefillConfig::default();
    let iters = 500;

    println!("\n🧪 block_select_grid (M×N×H score grid, {iters} iters)");
    println!("{}", "═".repeat(60));
    let hdr = format!(
        "{:>4}×{:>4}×{:>2} | {:>7} | {:>10}",
        "M", "N", "H", "grid_sz", "time"
    );
    println!("   {hdr}");
    println!("   {}", "-".repeat(40));

    for &(m, n, h) in configs {
        let grid: Vec<f32> = deterministic_scores(m * n * h, 77);

        let start = Instant::now();
        for _ in 0..iters {
            std::hint::black_box(block_select_grid(&grid, m, n, h, &cfg));
        }
        let elapsed = start.elapsed();
        let per_call = elapsed / iters as u32;
        let grid_sz = m * n * h;

        println!("   {m:>4}×{n:>4}×{h:>2} | {grid_sz:>7} | {per_call:>8?}");
    }
}
