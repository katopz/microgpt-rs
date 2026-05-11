//! Absorb-compress overhead benchmark — run with: cargo test --features bandit bench_absorb_compress -- --nocapture
//!
//! Benchmarks HL infrastructure components:
//! 1. AbsorbCompress overhead on relevance() calls
//! 2. TrialLog::append() throughput (writes/sec)
//! 3. HotSwapPruner::reload() latency (blake3 hash + load)
//!
//! Targets:
//! - absorb-compress adds <5% overhead to relevance()
//! - trial log sustains >100K writes/sec
//! - hot-swap reload <10ms

#[cfg(feature = "bandit")]
#[test]
fn bench_absorb_compress_overhead() {
    use std::time::Instant;

    use microgpt_rs::pruners::{AbsorbCompress, AbsorbCompressLayer, BanditStats, CompressConfig};
    use microgpt_rs::speculative::types::{NoScreeningPruner, ScreeningPruner};
    use microgpt_rs::types::Rng;

    let num_arms = 100;
    let warmup = 1000;
    let iters = 100_000;

    println!("\n🧪 Absorb-Compress Overhead Benchmark ({iters} iters, {warmup} warmup)");
    println!("{}", "═".repeat(70));

    // ── Baseline: BanditStats (no absorb-compress) ────────────────

    let mut stats = BanditStats::new(num_arms);
    let mut rng = Rng::new(42);

    // Warmup
    for i in 0..warmup {
        let arm = i % num_arms;
        stats.update(arm, rng.uniform());
    }

    let mut rng = Rng::new(42);
    let start = Instant::now();
    for i in 0..iters {
        let arm = i % num_arms;
        stats.update(arm, rng.uniform());
    }
    let baseline_update = start.elapsed();

    // ── With AbsorbCompress ───────────────────────────────────────

    let config = CompressConfig::new(50, 0.1, 5, 1000);
    let mut absorb = AbsorbCompressLayer::new(NoScreeningPruner, num_arms, config.clone());
    let mut rng = Rng::new(42);

    // Warmup
    for i in 0..warmup {
        let arm = i % num_arms;
        absorb.absorb(arm, rng.uniform());
    }

    let mut rng = Rng::new(42);
    let start = Instant::now();
    for i in 0..iters {
        let arm = i % num_arms;
        absorb.absorb(arm, rng.uniform());
    }
    let absorb_update = start.elapsed();

    let overhead_pct =
        ((absorb_update.as_nanos() as f64 / baseline_update.as_nanos() as f64) - 1.0) * 100.0;

    println!("   update() only:");
    println!("     Baseline (BanditStats):  {baseline_update:>8?}");
    println!("     With AbsorbCompress:     {absorb_update:>8?}");
    println!("     Overhead:                {overhead_pct:+.1}%");

    // ── relevance() overhead ──────────────────────────────────────

    let baseline_pruner = NoScreeningPruner;
    let absorb_pruner = AbsorbCompressLayer::new(NoScreeningPruner, num_arms, config.clone());

    // Warmup
    for i in 0..warmup {
        let _ = baseline_pruner.relevance(0, i % num_arms, &[]);
        let _ = absorb_pruner.relevance(0, i % num_arms, &[]);
    }

    let start = Instant::now();
    for i in 0..iters {
        let _ = baseline_pruner.relevance(0, i % num_arms, &[]);
    }
    let baseline_relevance = start.elapsed();

    let start = Instant::now();
    for i in 0..iters {
        let _ = absorb_pruner.relevance(0, i % num_arms, &[]);
    }
    let absorb_relevance = start.elapsed();

    let relevance_overhead_pct =
        ((absorb_relevance.as_nanos() as f64 / baseline_relevance.as_nanos() as f64) - 1.0) * 100.0;

    println!();
    println!("   relevance() call:");
    println!("     Baseline (NoScreening):  {baseline_relevance:>8?}");
    println!("     With AbsorbCompress:     {absorb_relevance:>8?}");
    println!("     Overhead:                {relevance_overhead_pct:+.1}%");

    // ── compress() call ───────────────────────────────────────────

    let mut absorb_for_compress = AbsorbCompressLayer::new(NoScreeningPruner, num_arms, config);
    // Feed enough data to trigger compress
    for arm in 0..num_arms {
        for _ in 0..60 {
            absorb_for_compress.absorb(arm, 0.05);
        }
    }

    let compress_iters = 1000;
    let start = Instant::now();
    for _ in 0..compress_iters {
        let _ = absorb_for_compress.compress();
    }
    let compress_time = start.elapsed();
    let compress_per_call = compress_time / compress_iters as u32;

    println!();
    println!("   compress() call:");
    println!("     {compress_iters} calls in {compress_time:?}");
    println!("     Per call: {compress_per_call:?}");

    println!();
    println!("   Target: absorb overhead < 5%");
    if overhead_pct < 5.0 {
        println!("   ✅ PASS: absorb overhead is {overhead_pct:.1}%");
    } else {
        println!("   ⚠️  FAIL: absorb overhead is {overhead_pct:.1}% (target <5%)");
    }
}

#[cfg(feature = "bandit")]
#[test]
fn bench_trial_log_throughput() {
    use std::time::Instant;

    use microgpt_rs::pruners::{TrialLog, TrialRecord};

    let iters = 100_000;
    let path = std::env::temp_dir().join(format!(
        "microgpt_bench_trial_{pid}.jsonl",
        pid = std::process::id()
    ));
    let _ = std::fs::remove_file(&path);

    println!("\n🧪 TrialLog Throughput Benchmark ({iters} writes)");
    println!("{}", "═".repeat(70));

    let mut log = TrialLog::new(&path).expect("Failed to create trial log");

    let sample = TrialRecord {
        episode: 0,
        arm: 2,
        reward: 0.8,
        q_value: 0.75,
        cumulative_reward: 42.0,
        cumulative_regret: 10.0,
        config: "bench".to_string(),
        note: "test".to_string(),
        base_correct: None,
        reviewed_correct: None,
    };

    // Warmup
    for i in 0..1000 {
        let mut rec = sample.clone();
        rec.episode = i;
        log.append(&rec).unwrap();
    }
    log.flush().unwrap();

    let start = Instant::now();
    for i in 0..iters {
        let mut rec = sample.clone();
        rec.episode = i;
        log.append(&rec).unwrap();
    }
    log.flush().unwrap();
    let elapsed = start.elapsed();

    let writes_per_sec = iters as f64 / elapsed.as_secs_f64();

    println!("   {iters} writes in {elapsed:?}");
    println!("   Throughput: {writes_per_sec:.0} writes/sec");
    println!("   Per write:  {:?}", elapsed / iters as u32);

    println!();
    println!("   Target: >100K writes/sec");
    if writes_per_sec > 100_000.0 {
        println!("   ✅ PASS: {writes_per_sec:.0} writes/sec");
    } else {
        println!("   ⚠️  FAIL: {writes_per_sec:.0} writes/sec (target >100K)");
    }

    // Cleanup
    let _ = std::fs::remove_file(&path);
}

#[cfg(feature = "bandit")]
#[test]
fn bench_hot_swap_reload() {
    use std::fs;
    use std::path::Path;
    use std::time::Instant;

    use microgpt_rs::pruners::HotSwapPruner;
    use microgpt_rs::speculative::types::ScreeningPruner;

    let iters = 100;
    let path = std::env::temp_dir().join(format!(
        "microgpt_bench_hotswap_{pid}.txt",
        pid = std::process::id()
    ));

    println!("\n🧪 HotSwapPruner Reload Benchmark ({iters} reloads)");
    println!("{}", "═".repeat(70));

    /// Minimal pruner for benchmarking reload cost.
    struct BenchPruner {
        value: f32,
    }

    impl BenchPruner {
        fn load(path: &Path) -> std::io::Result<Self> {
            let content = fs::read_to_string(path)?;
            let value = content.trim().parse::<f32>().unwrap_or(1.0);
            Ok(Self { value })
        }
    }

    impl ScreeningPruner for BenchPruner {
        fn relevance(&self, _depth: usize, _token_idx: usize, _parent_tokens: &[usize]) -> f32 {
            self.value
        }
    }

    // Create initial file
    fs::write(&path, "0.5").expect("Failed to write pruner file");

    let hs = HotSwapPruner::new(&path, Box::new(|p| BenchPruner::load(p)))
        .expect("Failed to create HotSwapPruner");

    // ── Reload with same file (no change) ────────────────────────

    let start = Instant::now();
    for _ in 0..iters {
        let _ = hs.reload().expect("Reload failed");
    }
    let same_file_time = start.elapsed();
    let same_per_call = same_file_time / iters as u32;

    println!("   Reload (same file, {iters}x):");
    println!("     Total: {same_file_time:?}");
    println!("     Per call: {same_per_call:?}");

    // ── Reload with changed file ─────────────────────────────────

    let start = Instant::now();
    for i in 0..iters {
        // Change file each time to force actual reload
        let value = format!("{:.3}", 0.1 + (i as f32 * 0.001));
        fs::write(&path, &value).expect("Failed to write");
        let _ = hs.reload().expect("Reload failed");
    }
    let changed_file_time = start.elapsed();
    let changed_per_call = changed_file_time / iters as u32;

    println!();
    println!("   Reload (changed file, {iters}x):");
    println!("     Total: {changed_file_time:?}");
    println!("     Per call: {changed_per_call:?}");

    // ── Blake3 hash cost (just hashing, no reload) ────────────────

    let test_data = vec![0u8; 4096]; // ~4KB WASM file size
    let hash_iters = 10_000;

    let start = Instant::now();
    for _ in 0..hash_iters {
        let _ = blake3::hash(&test_data);
    }
    let hash_time = start.elapsed();
    let hash_per_call = hash_time / hash_iters as u32;

    println!();
    println!("   Blake3 hash (4KB, {hash_iters}x):");
    println!("     Per call: {hash_per_call:?}");

    println!();
    println!("   Target: reload < 10ms");
    if changed_per_call.as_millis() < 10 {
        println!("   ✅ PASS: reload latency is {changed_per_call:?}");
    } else {
        println!("   ⚠️  FAIL: reload latency is {changed_per_call:?} (target <10ms)");
    }

    // Cleanup
    let _ = fs::remove_file(&path);
}
