//! Game Resolver Demo — Domain Validator + Bandit + DDTree Endgame
//!
//! Proves the endgame architecture: domain knowledge (game action screener) +
//! adaptive learning (bandit) + DDTree speculative search.
//!
//! The "resolver" = validator (domain constraint) + bandit (adaptive learning).
//! Unlike `bandit_02_ddtree.rs` which uses `NoScreeningPruner`, this demo
//! uses a real domain screener that enforces game action syntax.
//!
//! Run: `cargo run --example bandit_06_resolver --features bandit`
//!
//! # Architecture
//!
//! ```text
//! GameActionScreener (domain knowledge)
//!     ↓ relevance()
//! BanditPruner<GameActionScreener> (resolver = validator + learning)
//!     ↓ blended = ln(P_draft) + ln(R_domain × R_bandit)
//! build_dd_tree_screened() → DDTree
//!     ↓ extract_best_path()
//! Simulated verification → reward
//!     ↓
//! bandit.update(token, reward) → learn for next episode
//! ```
//!
//! # Comparison
//!
//! | Mode | Inner Pruner | Bandit | What It Tests |
//! |------|-------------|--------|---------------|
//! | Constrained | GameActionScreener | ✅ | Domain + adaptive learning |
//! | Unconstrained | NoScreeningPruner | ✅ | Bandit alone, no domain |

use std::time::Instant;

use microgpt_rs::pruners::{BanditPruner, BanditStrategy};
use microgpt_rs::speculative::{
    NoScreeningPruner, ScreeningPruner, build_dd_tree_screened, extract_best_path_into,
};
use microgpt_rs::types::{Config, Rng};

// ── Game Action Token Vocabulary (vocab_size=27) ──────────────

const PADDING: usize = 0;

// Commands (depth 0)
const CMD_MOVE: usize = 1;
const CMD_ATTACK: usize = 2;
const CMD_CAST: usize = 3;
const CMD_IDLE: usize = 4;

// Directions (depth 1 after MOVE/ATTACK)
const DIR_NORTH: usize = 5;
const DIR_SOUTH: usize = 6;
const DIR_EAST: usize = 7;
const DIR_WEST: usize = 8;

// Spells (depth 1 after CAST)
const SPELL_FIRE: usize = 9;
const SPELL_ICE: usize = 10;
const SPELL_HEAL: usize = 11;

// Numeric parameters (depth 2)
const NUM_START: usize = 12;
const NUM_END: usize = 20; // 12..=20 = 9 values

// Separators/terminators
const SEP_SPACE: usize = 21;
const TERM_CMD: usize = 22;
const TERM_SEQ: usize = 23;

// ── GameActionScreener ────────────────────────────────────────

/// Domain screener for game action syntax.
///
/// Enforces game command structure at the token level via `relevance()`:
/// - Depth 0: command prefix only (MOVE/ATTACK/CAST/IDLE)
/// - Depth 1: context-dependent params (directions, spells, separators)
/// - Depth 2: numeric or separator
/// - Depth 3+: separator only
///
/// Invalid tokens get `relevance = 0.0` → DDTree hard trims them.
/// Valid tokens get graded scores (ATTACK > CAST > MOVE > IDLE).
struct GameActionScreener;

impl GameActionScreener {
    fn is_command(token: usize) -> bool {
        matches!(token, CMD_MOVE | CMD_ATTACK | CMD_CAST | CMD_IDLE)
    }

    fn is_direction(token: usize) -> bool {
        matches!(token, DIR_NORTH | DIR_SOUTH | DIR_EAST | DIR_WEST)
    }

    fn is_spell(token: usize) -> bool {
        matches!(token, SPELL_FIRE | SPELL_ICE | SPELL_HEAL)
    }

    fn is_numeric(token: usize) -> bool {
        (NUM_START..=NUM_END).contains(&token)
    }

    fn is_separator(token: usize) -> bool {
        matches!(token, SEP_SPACE | TERM_CMD | TERM_SEQ)
    }

    fn is_valid_at(depth: usize, token: usize, parent: Option<usize>) -> bool {
        match depth {
            0 => Self::is_command(token),
            1 => match parent {
                Some(CMD_MOVE | CMD_ATTACK) => Self::is_direction(token),
                Some(CMD_CAST) => Self::is_spell(token),
                Some(CMD_IDLE) => Self::is_separator(token),
                _ => false,
            },
            2 => Self::is_numeric(token) || Self::is_separator(token),
            _ => Self::is_separator(token),
        }
    }
}

impl ScreeningPruner for GameActionScreener {
    fn relevance(&self, depth: usize, token_idx: usize, parent_tokens: &[usize]) -> f32 {
        if token_idx == PADDING {
            return 0.0;
        }

        let parent = parent_tokens.first().copied();

        if !Self::is_valid_at(depth, token_idx, parent) {
            return 0.0;
        }

        // Graded relevance by command type at depth 0
        match depth {
            0 => match token_idx {
                CMD_ATTACK => 0.9,
                CMD_CAST => 0.8,
                CMD_MOVE => 0.7,
                CMD_IDLE => 0.3,
                _ => 0.5,
            },
            1 => match parent {
                Some(CMD_ATTACK) => 0.9,
                Some(CMD_CAST) => 0.8,
                Some(CMD_MOVE) => 0.7,
                _ => 0.5,
            },
            _ => 0.5,
        }
    }
}

// ── Episode Metrics ───────────────────────────────────────────

struct EpisodeResult {
    reward: f32,
    regret: f32,
    tree_nodes: usize,
    accepted: usize,
    total: usize,
    time_us: u64,
}

// ── Marginal Generation ───────────────────────────────────────

/// Game-aware marginals: concentrated on valid game tokens.
/// Simulates a draft model that knows game syntax.
fn game_marginals(vocab_size: usize, lookahead: usize, rng: &mut Rng) -> Vec<Vec<f32>> {
    // Valid command tokens and their weights
    let commands = [CMD_MOVE, CMD_ATTACK, CMD_CAST, CMD_IDLE];
    let cmd_weights = [0.25, 0.35, 0.25, 0.15];

    let mut marginals = Vec::with_capacity(lookahead);
    for step in 0..lookahead {
        let mut probs = vec![f32::MIN_POSITIVE; vocab_size];

        if step % 3 == 0 {
            // Command position: weighted commands
            for (&cmd, &w) in commands.iter().zip(&cmd_weights) {
                probs[cmd] = w;
            }
        } else if step % 3 == 1 {
            // Parameter position: directions and spells
            let dir_weight = 0.08;
            for &d in &[DIR_NORTH, DIR_SOUTH, DIR_EAST, DIR_WEST] {
                probs[d] = dir_weight;
            }
            for &s in &[SPELL_FIRE, SPELL_ICE, SPELL_HEAL] {
                probs[s] = 0.07;
            }
        } else {
            // Numeric/separator position
            #[allow(clippy::needless_range_loop)]
            for t in NUM_START..=NUM_END {
                probs[t] = 0.04;
            }
            probs[SEP_SPACE] = 0.1;
            probs[TERM_SEQ] = 0.1;
        }

        // Add small noise from rng for variety
        for p in probs.iter_mut() {
            *p += rng.uniform() * 0.01;
        }

        // Normalize
        let sum: f32 = probs.iter().sum();
        for p in probs.iter_mut() {
            *p = (*p / sum).max(f32::MIN_POSITIVE);
        }

        marginals.push(probs);
    }
    marginals
}

/// Uniform marginals: all tokens equal probability.
fn uniform_marginals(vocab_size: usize, lookahead: usize) -> Vec<Vec<f32>> {
    let u = 1.0 / vocab_size as f32;
    (0..lookahead).map(|_| vec![u; vocab_size]).collect()
}

// ── Simulated Verification ────────────────────────────────────

fn simulate_verify(path: &[usize], rng: &mut Rng, base_rate: f32) -> (Vec<f32>, f32) {
    let mut rewards = Vec::with_capacity(path.len());
    let mut cum = 0.0f32;
    for &token in path {
        // Valid game tokens get higher acceptance
        let validity_bonus = if GameActionScreener::is_command(token)
            || GameActionScreener::is_direction(token)
            || GameActionScreener::is_spell(token)
        {
            0.15
        } else {
            0.0
        };
        let rate = (base_rate + validity_bonus).min(1.0);
        let r = if rng.uniform() < rate { 1.0 } else { 0.0 };
        rewards.push(r);
        cum += r;
    }
    (rewards, cum)
}

// ── Episode Runner (Constrained) ──────────────────────────────

fn run_constrained(
    pruner: &mut BanditPruner<GameActionScreener>,
    config: &Config,
    rng: &mut Rng,
    optimal: f32,
) -> EpisodeResult {
    let start = Instant::now();
    pruner.prepare_episode(rng);

    let marginals = game_marginals(config.vocab_size, config.draft_lookahead, rng);
    let slices: Vec<&[f32]> = marginals.iter().map(|m| m.as_slice()).collect();
    let tree = build_dd_tree_screened(&slices, config, pruner, true);

    let mut path = Vec::new();
    extract_best_path_into(&tree, &mut path);

    let (rewards, cum_reward) = simulate_verify(&path, rng, 0.65);
    for (&tok, &r) in path.iter().zip(&rewards) {
        pruner.update(tok, r);
    }
    pruner.decay_epsilon();

    EpisodeResult {
        reward: cum_reward,
        regret: (optimal - cum_reward).max(0.0),
        tree_nodes: tree.len(),
        accepted: rewards.iter().filter(|&&r| r > 0.5).count(),
        total: path.len(),
        time_us: start.elapsed().as_micros() as u64,
    }
}

// ── Episode Runner (Unconstrained) ────────────────────────────

fn run_unconstrained(
    pruner: &mut BanditPruner<NoScreeningPruner>,
    config: &Config,
    rng: &mut Rng,
    optimal: f32,
) -> EpisodeResult {
    let start = Instant::now();
    pruner.prepare_episode(rng);

    let marginals = uniform_marginals(config.vocab_size, config.draft_lookahead);
    let slices: Vec<&[f32]> = marginals.iter().map(|m| m.as_slice()).collect();
    let tree = build_dd_tree_screened(&slices, config, pruner, true);

    let mut path = Vec::new();
    extract_best_path_into(&tree, &mut path);

    let (rewards, cum_reward) = simulate_verify(&path, rng, 0.35);
    for (&tok, &r) in path.iter().zip(&rewards) {
        pruner.update(tok, r);
    }
    pruner.decay_epsilon();

    EpisodeResult {
        reward: cum_reward,
        regret: (optimal - cum_reward).max(0.0),
        tree_nodes: tree.len(),
        accepted: rewards.iter().filter(|&&r| r > 0.5).count(),
        total: path.len(),
        time_us: start.elapsed().as_micros() as u64,
    }
}

// ── Results Printer ───────────────────────────────────────────

fn print_comparison(c: &[EpisodeResult], u: &[EpisodeResult], episodes: usize) {
    println!("Game Resolver: Constrained vs Unconstrained ({episodes} episodes)");
    println!("═══════════════════════════════════════════════════════════════");
    println!(
        "{:<25} {:>14} {:>14} {:>10}",
        "Metric", "Constrained", "Unconstrained", "Δ"
    );
    println!("───────────────────────────────────────────────────────────────");

    let cr: f32 = c.iter().map(|r| r.reward).sum();
    let ur: f32 = u.iter().map(|r| r.reward).sum();
    let delta = if ur.abs() > f32::EPSILON {
        format!("{:+.1}%", (cr - ur) / ur.abs() * 100.0)
    } else {
        "n/a".to_string()
    };
    println!(
        "{:<25} {:>14.2} {:>14.2} {:>10}",
        "Cumulative Reward", cr, ur, delta
    );

    let creg: f32 = c.iter().map(|r| r.regret).sum();
    let ureg: f32 = u.iter().map(|r| r.regret).sum();
    let delta = if ureg.abs() > f32::EPSILON {
        format!("{:+.1}%", (creg - ureg) / ureg.abs() * 100.0)
    } else {
        "n/a".to_string()
    };
    println!(
        "{:<25} {:>14.2} {:>14.2} {:>10}",
        "Cumulative Regret", creg, ureg, delta
    );

    let ca: usize = c.iter().map(|r| r.accepted).sum();
    let ct: usize = c.iter().map(|r| r.total).sum();
    let ua: usize = u.iter().map(|r| r.accepted).sum();
    let ut: usize = u.iter().map(|r| r.total).sum();
    let c_rate = if ct > 0 {
        ca as f32 / ct as f32 * 100.0
    } else {
        0.0
    };
    let u_rate = if ut > 0 {
        ua as f32 / ut as f32 * 100.0
    } else {
        0.0
    };
    println!(
        "{:<25} {:>13.1}% {:>13.1}% {:>+9.1}%",
        "Accept Rate (%)",
        c_rate,
        u_rate,
        c_rate - u_rate
    );

    let cn: f32 = c.iter().map(|r| r.tree_nodes as f32).sum::<f32>() / episodes as f32;
    let un: f32 = u.iter().map(|r| r.tree_nodes as f32).sum::<f32>() / episodes as f32;
    println!(
        "{:<25} {:>14.1} {:>14.1} {:>10}",
        "Avg Tree Nodes", cn, un, ""
    );

    let ct_us: f64 = c.iter().map(|r| r.time_us as f64).sum::<f64>() / episodes as f64;
    let ut_us: f64 = u.iter().map(|r| r.time_us as f64).sum::<f64>() / episodes as f64;
    println!(
        "{:<25} {:>11.1} µs {:>11.1} µs {:>10}",
        "Avg Time/Episode", ct_us, ut_us, ""
    );
    println!();
}

// ── Main ──────────────────────────────────────────────────────

fn main() {
    println!("Game Resolver Demo — Domain Validator + Bandit + DDTree\n");

    let config = Config::draft();
    let episodes = 1000;
    let seed = 42;

    println!(
        "Config: vocab_size={}, draft_lookahead={}, tree_budget={}",
        config.vocab_size, config.draft_lookahead, config.tree_budget
    );
    println!("Strategy: EpsilonGreedy {{ epsilon: 0.3, decay: 0.995 }}");
    println!("Episodes: {episodes}\n");

    let strategy = BanditStrategy::EpsilonGreedy {
        epsilon: 0.3,
        decay: 0.995,
    };
    let optimal = config.draft_lookahead as f32;

    // Constrained: BanditPruner<GameActionScreener>
    println!("Running constrained (domain + bandit)...");
    let mut c_pruner = BanditPruner::new(GameActionScreener, strategy.clone(), config.vocab_size);
    let mut c_rng = Rng::new(seed);
    let constrained: Vec<EpisodeResult> = (0..episodes)
        .map(|_| run_constrained(&mut c_pruner, &config, &mut c_rng, optimal))
        .collect();

    // Unconstrained: BanditPruner<NoScreeningPruner>
    println!("Running unconstrained (bandit only)...");
    let mut u_pruner = BanditPruner::new(NoScreeningPruner, strategy, config.vocab_size);
    let mut u_rng = Rng::new(seed + 1);
    let unconstrained: Vec<EpisodeResult> = (0..episodes)
        .map(|_| run_unconstrained(&mut u_pruner, &config, &mut u_rng, optimal))
        .collect();

    println!();
    print_comparison(&constrained, &unconstrained, episodes);

    // Final bandit state comparison
    println!("Bandit Q-values (top 5 arms):");
    println!("  Constrained:");
    let mut c_arms: Vec<(usize, f32, u32)> = (0..config.vocab_size)
        .map(|i| (i, c_pruner.q_values()[i], c_pruner.visits()[i]))
        .filter(|(_, _, v)| *v > 0)
        .collect();
    c_arms.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    for (arm, q, visits) in c_arms.iter().take(5) {
        let name = token_name(*arm);
        println!("    arm={arm:>2} ({name:<8}) Q={q:.3} visits={visits}");
    }

    println!("  Unconstrained:");
    let mut u_arms: Vec<(usize, f32, u32)> = (0..config.vocab_size)
        .map(|i| (i, u_pruner.q_values()[i], u_pruner.visits()[i]))
        .filter(|(_, _, v)| *v > 0)
        .collect();
    u_arms.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    for (arm, q, visits) in u_arms.iter().take(5) {
        let name = token_name(*arm);
        println!("    arm={arm:>2} ({name:<8}) Q={q:.3} visits={visits}");
    }
    println!();
}

fn token_name(token: usize) -> &'static str {
    match token {
        PADDING => "PAD",
        CMD_MOVE => "MOVE",
        CMD_ATTACK => "ATTACK",
        CMD_CAST => "CAST",
        CMD_IDLE => "IDLE",
        DIR_NORTH => "NORTH",
        DIR_SOUTH => "SOUTH",
        DIR_EAST => "EAST",
        DIR_WEST => "WEST",
        SPELL_FIRE => "FIRE",
        SPELL_ICE => "ICE",
        SPELL_HEAL => "HEAL",
        SEP_SPACE => "SPACE",
        TERM_CMD => "DOT",
        TERM_SEQ => "SEMI",
        n if (NUM_START..=NUM_END).contains(&n) => "NUM",
        _ => "???",
    }
}
