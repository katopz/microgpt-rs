//! HL Demo: Hot-Swap Pruner with Trial Log and Regression Suite
//!
//! Demonstrates runtime pruner reload without process restart:
//! 1. Creates a simple text-file-based pruner (value read from file)
//! 2. Runs 100 episodes with manual UCB1 + HotSwapPruner + TrialLog
//! 3. Simulates "agent writes new pruner": changes file content
//! 4. Calls hot_swap.reload() mid-run, shows version bump
//! 5. Runs RegressionSuite against golden traces from first phase
//! 6. Prints before/after Q-values, reload timeline, regression results
//!
//! Run: `cargo run --example hl_02_hotswap --features bandit`

use std::fs;
use std::path::Path;

use microgpt_rs::pruners::{
    AbsorbCompress, AbsorbCompressLayer, BanditEnv, BanditStats, BernoulliEnv, CompressConfig,
    GoldenTrace, HotSwapPruner, RegressionSuite, ReplayReward, TrialLog, TrialRecord,
};
use microgpt_rs::speculative::types::ScreeningPruner;
use microgpt_rs::types::Rng;

// ── Config ──────────────────────────────────────────────────────

const SEED: u64 = 42;
const EPISODES: usize = 200;
const PRUNER_PATH: &str = "/tmp/hl_hotswap_pruner.txt";
const TRIAL_PATH: &str = "/tmp/hl_hotswap_trial.jsonl";

const PROBS: [f32; 5] = [0.1, 0.3, 0.8, 0.4, 0.2];

// ── File-Based Pruner ───────────────────────────────────────────

/// Simple pruner that reads relevance from a text file.
/// Simulates a WASM validator without the WASM dependency.
struct FilePruner {
    value: f32,
}

impl FilePruner {
    fn load(path: &Path) -> std::io::Result<Self> {
        let content = fs::read_to_string(path)?;
        let value = content.trim().parse::<f32>().unwrap_or(1.0);
        Ok(Self { value })
    }
}

impl ScreeningPruner for FilePruner {
    fn relevance(&self, _depth: usize, _token_idx: usize, _parent_tokens: &[usize]) -> f32 {
        self.value
    }
}

// ── Replay Runner ───────────────────────────────────────────────

/// Replay runner that uses a BernoulliEnv to simulate arm pulls.
struct ReplayRunner {
    env: BernoulliEnv,
    rng: Rng,
}

impl ReplayRunner {
    fn new(env: BernoulliEnv, seed: u64) -> Self {
        Self {
            env,
            rng: Rng::new(seed),
        }
    }
}

impl ReplayReward for ReplayRunner {
    fn replay_reward(&mut self, trace: &GoldenTrace) -> f32 {
        trace
            .actions
            .iter()
            .map(|&arm| self.env.pull(arm, &mut self.rng))
            .sum()
    }
}

// ── Arm Selection ───────────────────────────────────────────────

fn select_arm_ucb1(stats: &BanditStats, compressed: &[usize], num_arms: usize) -> usize {
    for i in 0..num_arms {
        if stats.visit_count(i) == 0 && !compressed.contains(&i) {
            return i;
        }
    }

    (0..num_arms)
        .filter(|arm| !compressed.contains(arm))
        .max_by(|&a, &b| {
            stats
                .ucb1_score(a)
                .partial_cmp(&stats.ucb1_score(b))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap_or(0)
}

// ── Display Helpers ─────────────────────────────────────────────

fn print_header() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║     Heuristic Learning — Hot-Swap Pruner + Regression       ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();
}

fn print_env() {
    println!("🎯 Bernoulli Environment:");
    println!("   Arms:     {}", PROBS.len());
    println!(
        "   Probs:    [{}]",
        PROBS
            .iter()
            .map(|p| format!("{p:.1}"))
            .collect::<Vec<_>>()
            .join(", ")
    );
    println!(
        "   Optimal:  Arm {} (p={:.1})",
        PROBS
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap(),
        PROBS.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
    );
    println!("   Strategy: UCB1");
    println!("   Pruner:   {PRUNER_PATH} (file-based)");
    println!("   TrialLog: {TRIAL_PATH}");
    println!();
}

fn print_q_values(label: &str, stats: &BanditStats, compressed: &[usize]) {
    let q: Vec<String> = (0..stats.num_arms())
        .map(|i| {
            let block = if compressed.contains(&i) { " ✗" } else { "" };
            format!("{:.3}{block}", stats.q_value(i))
        })
        .collect();
    let v: Vec<String> = (0..stats.num_arms())
        .map(|i| format!("{}", stats.visit_count(i)))
        .collect();
    println!("   {label} Q-values: [{}]", q.join(", "));
    println!("   {label} Visits:   [{}]", v.join(", "));
}

// ── Main ────────────────────────────────────────────────────────

fn main() {
    print_header();
    print_env();

    let env = BernoulliEnv::new(&PROBS);
    let mut rng = Rng::new(SEED);
    let mut stats = BanditStats::new(env.num_arms());

    // Cleanup from previous runs
    let _ = fs::remove_file(PRUNER_PATH);
    let _ = fs::remove_file(TRIAL_PATH);

    // Absorb-compress layer
    let compress_config = CompressConfig::new(15, 0.2, 2, 50);

    // ── Phase 1: Initial Pruner ─────────────────────────────────

    println!("📋 Phase 1: Initial Pruner (relevance=0.5)");
    fs::write(PRUNER_PATH, "0.5").expect("Failed to write pruner file");

    let hot_swap = HotSwapPruner::new(Path::new(PRUNER_PATH), Box::new(FilePruner::load))
        .expect("Failed to create HotSwapPruner");

    let mut absorb = AbsorbCompressLayer::new(hot_swap, env.num_arms(), compress_config);
    let mut trial_log = TrialLog::new(Path::new(TRIAL_PATH)).expect("Failed to create trial log");

    println!("   HotSwap version: {}", absorb.inner().version());
    println!(
        "   Pruner relevance: {:.1}",
        absorb.inner().relevance(0, 0, &[])
    );
    println!();

    // Run Phase 1 episodes
    let mut cumulative_reward = 0.0f32;
    let mut cumulative_regret = 0.0f32;
    let mut reload_log: Vec<(usize, bool, u64)> = Vec::new();

    println!("🔄 Running {EPISODES} episodes with mid-run reload...");
    for episode in 0..EPISODES {
        let compressed = absorb.compressed_arms().to_vec();
        let arm = select_arm_ucb1(&stats, &compressed, env.num_arms());
        let reward = env.pull(arm, &mut rng);

        stats.update(arm, reward);
        absorb.absorb(arm, reward);

        cumulative_reward += reward;
        cumulative_regret += env.optimal_reward() - env.expected_reward(arm);

        // Persist to trial log
        let note = format!("version={}", absorb.inner().version());
        let record = TrialRecord {
            episode,
            arm,
            reward,
            q_value: stats.q_value(arm),
            cumulative_reward,
            cumulative_regret,
            config: "UCB1".to_string(),
            note,
            base_correct: None,
            reviewed_correct: None,
            anchors: None,
        };
        if let Err(e) = trial_log.append(&record) {
            eprintln!("Trial log error: {e}");
        }

        // Simulate agent writing new pruner at episode 100
        if episode == 99 {
            println!();
            println!("🔄 Episode 100: Agent writes new pruner (relevance 0.5 → 0.9)...");
            fs::write(PRUNER_PATH, "0.9").expect("Failed to update pruner file");

            let changed = absorb.inner_mut().reload().expect("HotSwap reload failed");
            let new_version = absorb.inner().version();

            reload_log.push((episode + 1, changed, new_version));
            println!("   Reloaded: changed={changed} version={new_version}");
            println!(
                "   New relevance: {:.1}",
                absorb.inner().relevance(0, 0, &[])
            );
        }

        // Compress check every 50 episodes
        if (episode + 1) % 50 == 0 {
            if absorb.should_compress() {
                let promoted = absorb.compress();
                if !promoted.is_empty() {
                    println!(
                        "   📦 Episode {:4}: Compressed arms {promoted:?}",
                        episode + 1
                    );
                }
            }

            let avg = cumulative_reward / (episode + 1) as f32;
            println!(
                "   ✓ Episode {:4}: avg_reward={avg:.3}  compressed={:?}  version={}",
                episode + 1,
                absorb.compressed_arms(),
                absorb.inner().version(),
            );
        }
    }
    trial_log.flush().expect("Failed to flush");
    println!();

    // ── Final State ─────────────────────────────────────────────

    let compressed = absorb.compressed_arms();
    println!("📊 Final State:");
    print_q_values("Final", &stats, compressed);
    println!(
        "   Best arm: {} (optimal: {})",
        stats.best_arm(),
        env.optimal_arm()
    );
    println!("   Total reward: {cumulative_reward:.1}");
    println!(
        "   Avg reward:   {:.3}",
        cumulative_reward / EPISODES as f32
    );
    println!("   Total regret: {cumulative_regret:.1}");
    println!(
        "   Avg regret:   {:.3}",
        cumulative_regret / EPISODES as f32
    );
    println!("   Compressed:   {:?}", compressed);
    println!();

    // ── Reload Timeline ─────────────────────────────────────────

    println!("🔄 Reload Timeline:");
    for (ep, changed, ver) in &reload_log {
        println!("   Episode {ep}: changed={changed} version={ver}");
    }
    println!();

    // ── Hot-Swap Demo (standalone) ──────────────────────────────

    println!("🔄 Hot-Swap Reload Demo (standalone):");
    fs::write(PRUNER_PATH, "0.7").expect("Failed to write pruner v3");

    let hs = HotSwapPruner::new(Path::new(PRUNER_PATH), Box::new(FilePruner::load))
        .expect("Failed to create HotSwapPruner");

    println!("   Created: version={}", hs.version());

    // Reload same file → no change
    let changed = hs.reload().expect("Reload failed");
    println!(
        "   Reload (same file): changed={changed} version={}",
        hs.version()
    );

    // Change file → reload detects change
    fs::write(PRUNER_PATH, "1.0").expect("Failed to write pruner v4");
    let changed = hs.reload().expect("Reload failed");
    println!(
        "   Reload (changed): changed={changed} version={}",
        hs.version()
    );

    // Reload same again
    let changed = hs.reload().expect("Reload failed");
    println!(
        "   Reload (same file): changed={changed} version={}",
        hs.version()
    );

    // Another change
    fs::write(PRUNER_PATH, "0.3").expect("Failed to write pruner v5");
    let changed = hs.reload().expect("Reload failed");
    println!(
        "   Reload (changed): changed={changed} version={}",
        hs.version()
    );
    println!();

    // ── Regression Suite ────────────────────────────────────────

    println!("🧪 Regression Suite (golden traces from trial log)");
    let suite = RegressionSuite::from_trials(Path::new(TRIAL_PATH), 10)
        .expect("Failed to build regression suite");

    println!("   Golden traces: {}", suite.len());
    for trace in &suite.traces {
        println!(
            "     {} → expected_reward={:.3}",
            trace.label, trace.expected_reward
        );
    }

    let result = suite.run(|_trace| ReplayRunner::new(BernoulliEnv::new(&PROBS), SEED));

    println!("   Passed: {}", result.passed);
    println!("   Failed: {}", result.failed);

    if !result.failures.is_empty() {
        println!("   ⚠️  Failures:");
        for f in &result.failures {
            println!(
                "      {}: expected={:.3} actual={:.3} delta={:.3}",
                f.trace_label, f.expected_reward, f.actual_reward, f.delta
            );
        }
    } else {
        println!("   ✅ All traces passed!");
    }
    println!();

    // ── Trial Log Summary ───────────────────────────────────────

    let records = TrialLog::load(Path::new(TRIAL_PATH)).expect("Failed to load trial log");
    let summary = TrialLog::summary(&records);

    println!("📝 Trial Log Summary ({TRIAL_PATH}):");
    println!("   Records written:  {}", trial_log.count());
    println!("   Records loaded:   {}", records.len());
    println!("   Best arm:         {}", summary.best_arm);
    println!("   Avg reward:       {:.3}", summary.avg_reward);
    println!("   Avg regret:       {:.3}", summary.avg_regret);
    println!();

    // Cleanup temp files
    let _ = fs::remove_file(PRUNER_PATH);
    let _ = fs::remove_file(TRIAL_PATH);

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║   HL Infrastructure: HotSwap ✓  TrialLog ✓  Regression ✓   ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
}
