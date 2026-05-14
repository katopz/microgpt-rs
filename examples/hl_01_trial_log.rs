//! HL Demo: Trial Log + Absorb-Compress with BernoulliEnv
//!
//! Demonstrates the Heuristic Learning infrastructure:
//! 1. Runs 1000 bandit episodes with UCB1 strategy
//! 2. Persists every episode to a JSONL trial log
//! 3. After every 100 episodes, checks absorb-compress thresholds
//! 4. Low-Q arms get promoted to hard blocks (compressed)
//! 5. Prints initial/final Q-values, compressed arms, and trial summary
//!
//! Run: `cargo run --example hl_01_trial_log --features bandit`

use std::path::Path;

use microgpt_rs::pruners::{
    AbsorbCompress, AbsorbCompressLayer, BanditEnv, BanditStats, BernoulliEnv, CompressConfig,
    TrialLog, TrialRecord,
};
use microgpt_rs::speculative::types::NoScreeningPruner;
use microgpt_rs::types::Rng;

// ── Config ──────────────────────────────────────────────────────

const EPISODES: usize = 1000;
const SEED: u64 = 42;
const TRIAL_PATH: &str = "/tmp/hl_trial_log.jsonl";

/// Arm probabilities: arm 2 (p=0.8) is optimal, arms 0 and 4 are bad.
const PROBS: [f32; 5] = [0.1, 0.3, 0.8, 0.4, 0.2];

// ── Arm Selection ───────────────────────────────────────────────

/// UCB1 arm selection, skipping compressed arms.
fn select_arm(stats: &BanditStats, compressed: &[usize], num_arms: usize) -> usize {
    // Cold start: play each non-compressed arm once
    for i in 0..num_arms {
        if stats.visit_count(i) == 0 && !compressed.contains(&i) {
            return i;
        }
    }

    // UCB1: pick highest score among non-compressed arms
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
    println!("║     Heuristic Learning — Trial Log + Absorb-Compress        ║");
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
    println!("   Episodes: {EPISODES}");
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

    // Setup environment
    let env = BernoulliEnv::new(&PROBS);
    let mut rng = Rng::new(SEED);
    let mut stats = BanditStats::new(env.num_arms());

    // Absorb-compress layer with demo-friendly thresholds
    let config = CompressConfig::new(
        15,  // min_visits — low enough for UCB1 to trigger
        0.2, // q_threshold — catches arms 0 (p=0.1) and 4 (p=0.2)
        2,   // promote_count
        100, // check_interval
    );
    let mut absorb = AbsorbCompressLayer::new(NoScreeningPruner, env.num_arms(), config);

    // Trial log
    let trial_path = Path::new(TRIAL_PATH);
    let _ = std::fs::remove_file(trial_path);
    let mut trial_log = TrialLog::new(trial_path).expect("Failed to create trial log");

    // Print initial state
    println!("📊 Initial State:");
    print_q_values("Init", &stats, &[]);
    println!();

    // Run episodes
    let mut cumulative_reward = 0.0f32;
    let mut cumulative_regret = 0.0f32;
    let mut compressed_at_episode: Vec<(usize, Vec<usize>)> = Vec::new();

    println!("🔄 Running {EPISODES} episodes...");
    for episode in 0..EPISODES {
        let compressed = absorb.compressed_arms().to_vec();
        let arm = select_arm(&stats, &compressed, env.num_arms());
        let reward = env.pull(arm, &mut rng);

        stats.update(arm, reward);
        absorb.absorb(arm, reward);

        cumulative_reward += reward;
        cumulative_regret += env.optimal_reward() - env.expected_reward(arm);

        // Persist to trial log
        let record = TrialRecord {
            episode,
            arm,
            reward,
            q_value: stats.q_value(arm),
            cumulative_reward,
            cumulative_regret,
            config: "UCB1".to_string(),
            note: String::new(),
            base_correct: None,
            reviewed_correct: None,
            anchors: None,
        };
        if let Err(e) = trial_log.append(&record) {
            eprintln!("Trial log write error at episode {episode}: {e}");
        }

        // Compress check every 100 episodes
        if (episode + 1) % 100 == 0 {
            if absorb.should_compress() {
                let promoted = absorb.compress();
                if !promoted.is_empty() {
                    compressed_at_episode.push((episode + 1, promoted.clone()));
                    println!(
                        "   📦 Episode {:4}: Compressed arms {promoted:?} (hard-blocked)",
                        episode + 1
                    );
                }
            }

            // Progress
            let avg = cumulative_reward / (episode + 1) as f32;
            println!(
                "   ✓ Episode {:4}: avg_reward={avg:.3}  compressed={:?}",
                episode + 1,
                absorb.compressed_arms(),
            );
        }
    }
    trial_log.flush().expect("Failed to flush trial log");
    println!();

    // Final state
    let compressed = absorb.compressed_arms();
    println!("📊 Final State:");
    print_q_values("Final", &stats, compressed);
    println!(
        "   Best arm: {} (optimal: {})",
        stats.best_arm(),
        env.optimal_arm()
    );
    println!("   Total reward:    {cumulative_reward:.1}",);
    println!(
        "   Avg reward:      {:.3}",
        cumulative_reward / EPISODES as f32
    );
    println!("   Total regret:    {cumulative_regret:.1}",);
    println!(
        "   Avg regret:      {:.3}",
        cumulative_regret / EPISODES as f32
    );
    println!("   Compressed arms: {:?}", compressed,);
    println!();

    // Compression timeline
    if !compressed_at_episode.is_empty() {
        println!("📦 Compression Timeline:");
        for (ep, arms) in &compressed_at_episode {
            let arm_labels: Vec<String> = arms
                .iter()
                .map(|&a| format!("Arm{a}(p={:.1})", PROBS[a]))
                .collect();
            println!("   Episode {ep}: {arm_labels:?}");
        }
        println!();
    }

    // Trial log summary
    let records = TrialLog::load(trial_path).expect("Failed to load trial log");
    let summary = TrialLog::summary(&records);
    println!("📝 Trial Log Summary ({TRIAL_PATH}):");
    println!("   Records written:  {}", trial_log.count());
    println!("   Records loaded:   {}", records.len());
    println!("   Best arm (by avg reward): {}", summary.best_arm);
    println!("   Avg reward:       {:.3}", summary.avg_reward);
    println!("   Avg regret:       {:.3}", summary.avg_regret);
    println!();

    // Verify persistence
    assert_eq!(trial_log.count(), EPISODES, "Trial log count mismatch");
    assert_eq!(records.len(), EPISODES, "Loaded records mismatch");

    // Verify compression happened
    if !compressed.is_empty() {
        println!("✅ Absorb-compress successfully promoted low-Q arms to hard blocks!");
        for &arm in compressed {
            println!(
                "   Arm {arm}: Q={:.3} (expected≈{:.1}) → blocked",
                stats.q_value(arm),
                PROBS[arm]
            );
        }
    } else {
        println!("⚠️  No arms were compressed — thresholds may need adjustment");
    }

    println!();
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║   HL Infrastructure: TrialLog ✓  AbsorbCompress ✓          ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
}
