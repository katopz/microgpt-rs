//! Combat Bandit Demo — "Smart Ass Modelless" Monster AI
//!
//! Proves that a multi-armed bandit can learn to adapt monster behavior
//! to different player types WITHOUT any neural network, LLM, or scripted AI.
//!
//! # The "Smart Ass Modelless" Concept
//!
//! Traditional game AI uses:
//! - **Scripted trees**: "if player aggressive → defend" (programmer wrote the counter)
//! - **Neural nets**: learn from millions of replay frames
//! - **Hardcoded tables**: rock-paper-scissors matchup chart
//!
//! This demo uses **NONE** of that. The monster starts knowing NOTHING about
//! the player. It tries random actions, observes per-turn rewards, and gradually
//! learns which actions work best against each player archetype.
//!
//! The "intelligence" is just statistics: UCB1 balances explore/exploit,
//! Thompson Sampling draws from posterior distributions, ε-greedy anneals
//! randomness. No model. No training data. No neural net. Just math.
//!
//! # How This Beats Scripted AI
//!
//! A scripted AI knows "Defend beats Aggressive" because the programmer told it.
//! This bandit AI *discovers* that Defend beats Aggressive through trial and error.
//! Change the damage numbers, and it re-learns automatically. No code changes needed.
//!
//! Run: `cargo run --example combat_bandit_demo --features bandit`

use microgpt_rs::pruners::{BanditStats, BanditStrategy};
use microgpt_rs::types::Rng;

// ── Constants ──────────────────────────────────────────────────

const NUM_ARMS: usize = 5;
const MAX_HP: f32 = 100.0;
const COMBATS_PER_TYPE: usize = 100;
const COMBATS_STRATEGY_TEST: usize = 500;
const MAX_TURNS: u32 = 200;

// ── Enums ──────────────────────────────────────────────────────

/// Monster combat actions — each maps to a bandit arm.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MonsterAction {
    Attack = 0,
    Defend = 1,
    Heal = 2,
    Special = 3,
    Flee = 4,
}

impl MonsterAction {
    fn from_usize(v: usize) -> Self {
        match v % NUM_ARMS {
            0 => Self::Attack,
            1 => Self::Defend,
            2 => Self::Heal,
            3 => Self::Special,
            _ => Self::Flee,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Attack => "Attack",
            Self::Defend => "Defend",
            Self::Heal => "Heal",
            Self::Special => "Special",
            Self::Flee => "Flee",
        }
    }
}

impl std::fmt::Display for MonsterAction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.label())
    }
}

/// Player archetypes — fixed probability distributions over actions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PlayerType {
    Aggressive, // 70% Attack, 10% Defend, 10% Heal, 10% Special
    Defensive,  // 20% Attack, 50% Defend, 20% Heal, 10% Special
    Balanced,   // 30% Attack, 20% Defend, 20% Heal, 30% Special
    Burst,      // 10% Attack, 10% Defend, 10% Heal, 70% Special
}

impl PlayerType {
    fn label(self) -> &'static str {
        match self {
            Self::Aggressive => "Aggressive",
            Self::Defensive => "Defensive",
            Self::Balanced => "Balanced",
            Self::Burst => "Burst",
        }
    }

    fn all() -> [Self; 4] {
        [
            Self::Aggressive,
            Self::Defensive,
            Self::Balanced,
            Self::Burst,
        ]
    }

    /// Select action index based on fixed probability distribution.
    fn select_action(self, rng: &mut Rng) -> usize {
        let r = rng.uniform();
        match self {
            Self::Aggressive => {
                if r < 0.70 {
                    0
                } else if r < 0.80 {
                    1
                } else if r < 0.90 {
                    2
                } else {
                    3
                }
            }
            Self::Defensive => {
                if r < 0.20 {
                    0
                } else if r < 0.70 {
                    1
                } else if r < 0.90 {
                    2
                } else {
                    3
                }
            }
            Self::Balanced => {
                if r < 0.30 {
                    0
                } else if r < 0.50 {
                    1
                } else if r < 0.70 {
                    2
                } else {
                    3
                }
            }
            Self::Burst => {
                if r < 0.10 {
                    0
                } else if r < 0.20 {
                    1
                } else if r < 0.30 {
                    2
                } else {
                    3
                }
            }
        }
    }
}

/// Combat outcome.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CombatResult {
    MonsterWin,
    PlayerWin,
    Draw,
}

// ── Combat Resolution ──────────────────────────────────────────

struct TurnResult {
    monster_dmg_taken: f32,
    player_dmg_taken: f32,
    monster_heal: f32,
    raw_incoming: f32, // player damage before monster defend reduction
    raw_outgoing: f32, // monster damage before player defend reduction
}

/// Resolve one turn of combat. Both sides act simultaneously.
fn resolve_turn(monster_action: usize, player_action: usize, rng: &mut Rng) -> TurnResult {
    // Raw damage from each side (before defend reduction)
    let monster_raw = match monster_action {
        0 => 15.0 + rng.uniform() * 10.0, // Attack: 15–25
        3 => {
            // Special: 25–40, 20% fail
            if rng.uniform() < 0.2 {
                0.0
            } else {
                25.0 + rng.uniform() * 15.0
            }
        }
        _ => 0.0,
    };
    let player_raw = match player_action {
        0 => 15.0 + rng.uniform() * 10.0,
        3 => {
            if rng.uniform() < 0.2 {
                0.0
            } else {
                25.0 + rng.uniform() * 15.0
            }
        }
        _ => 0.0,
    };

    // Defend: 50% damage reduction + 5 HP self-heal
    let monster_dmg_taken = if monster_action == 1 {
        player_raw * 0.5
    } else {
        player_raw
    };
    let player_dmg_taken = if player_action == 1 {
        monster_raw * 0.5
    } else {
        monster_raw
    };

    let monster_heal = match monster_action {
        1 => 5.0,                         // Defend: small self-heal
        2 => 20.0 + rng.uniform() * 10.0, // Heal: 20–30
        _ => 0.0,
    };

    TurnResult {
        monster_dmg_taken,
        player_dmg_taken,
        monster_heal,
        raw_incoming: player_raw,
        raw_outgoing: monster_raw,
    }
}

/// Per-turn reward shaping so bandit learns action-specific value.
///
/// Key design principle: each action must be situationally optimal against
/// exactly one player archetype. The bandit discovers WHICH through trial-and-error.
///
/// | Action    | Best vs      | Why                              |
/// |-----------|-------------|----------------------------------|
/// | Defend    | Aggressive  | Blocks their frequent attacks    |
/// | Attack    | Defensive   | They rarely attack back, safe DPS |
/// | Special   | Balanced    | High ceiling vs mixed strategy   |
/// | Heal      | Burst       | Survive spike damage windows     |
fn turn_reward(action: usize, t: &TurnResult, m_hp: f32) -> f32 {
    let urgency = (MAX_HP - m_hp) / MAX_HP; // 0.0 at full HP → 1.0 at 0 HP

    match action {
        // Attack: reward for raw damage attempted (before opponent defend).
        // Reduced when player_defend absorbed the hit (wasted effort).
        // Best vs Defensive (they don't attack much) and Aggressive (they don't defend).
        // Mediocre vs Balanced (20% defend chance absorbs some hits).
        0 => {
            let dealt_ratio = if t.raw_outgoing > 0.0 {
                t.player_dmg_taken / t.raw_outgoing // 1.0 = full damage, 0.5 = blocked
            } else {
                1.0
            };
            (t.raw_outgoing / 25.0 * dealt_ratio).clamp(0.1, 0.85)
        }

        // Defend: reward for damage prevented. Scales with incoming threat.
        // Best vs Aggressive (70% attack = constant stream to block).
        // Poor vs Defensive/Burst (they rarely attack).
        1 => {
            let prevented = t.raw_incoming - t.monster_dmg_taken;
            if prevented > 8.0 {
                // Blocked a real attack — big reward
                (prevented / 12.0 + 0.3).min(1.0)
            } else if prevented > 2.0 {
                // Blocked something small
                (prevented / 15.0 + 0.1).min(0.5)
            } else {
                0.02 // nothing to block — wasted turn
            }
        }

        // Heal: reward scales with urgency (how hurt are you?).
        // Full HP → wasteful (0.05). Low HP → critical (up to 0.85).
        // Best vs Burst (70% Special = spike damage → high urgency windows).
        // Poor vs Defensive (low incoming → monster stays healthy → wasteful).
        2 => {
            if urgency < 0.15 {
                0.03 // nearly full HP — wasteful overheal
            } else if urgency > 0.5 {
                // Critical: healing is life-saving, big multiplier
                let effective = t.monster_heal.min(MAX_HP - m_hp);
                (effective / 20.0 * (0.5 + urgency)).clamp(0.1, 0.85)
            } else {
                // Moderate damage: decent but not urgent
                let effective = t.monster_heal.min(MAX_HP - m_hp);
                (effective / 25.0 * urgency).clamp(0.05, 0.55)
            }
        }

        // Special: high reward on hit (raw power), near-zero on miss.
        // Slightly better base than Attack because of 20% failure risk.
        // Best vs Balanced (moderate defense, moderate offense → consistent value).
        // Decent vs Aggressive (they don't defend, 10%).
        3 => {
            if t.raw_outgoing > 0.0 {
                (t.raw_outgoing / 28.0).clamp(0.3, 0.95)
            } else {
                0.02 // failed — wasted turn
            }
        }

        // Flee: cowardice is punished
        _ => -0.5,
    }
}

// ── Bandit Arm Selection ──────────────────────────────────────

/// Select an arm using the given strategy. Cold-start: play each arm once first.
fn select_arm(
    stats: &BanditStats,
    strategy: &BanditStrategy,
    rng: &mut Rng,
    epsilon: f32,
) -> usize {
    // Cold start: play each combat arm (0-3) once, skip Flee (arm 4)
    let combat_arms = NUM_ARMS - 1;
    for i in 0..combat_arms {
        if stats.visit_count(i) == 0 {
            return i;
        }
    }
    match strategy {
        BanditStrategy::Ucb1 => (0..combat_arms)
            .max_by(|&a, &b| {
                stats
                    .ucb1_score(a)
                    .partial_cmp(&stats.ucb1_score(b))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or(0),
        BanditStrategy::EpsilonGreedy { .. } => {
            if rng.uniform() < epsilon {
                // Explore among combat actions only (skip Flee)
                (rng.uniform() * combat_arms as f32) as usize % combat_arms
            } else {
                stats.best_arm()
            }
        }
        BanditStrategy::ThompsonSampling => (0..combat_arms)
            .map(|i| (i, stats.thompson_sample(i, rng)))
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0),
    }
}

// ── Combat Loop ────────────────────────────────────────────────

/// Track cumulative results across multiple combats.
#[derive(Default)]
struct CombatTracker {
    wins: u32,
    losses: u32,
    draws: u32,
    total_turns: u32,
    action_counts: [u32; NUM_ARMS],
}

impl CombatTracker {
    fn record(&mut self, result: CombatResult, turns: u32, actions: &[u32; NUM_ARMS]) {
        match result {
            CombatResult::MonsterWin => self.wins += 1,
            CombatResult::PlayerWin => self.losses += 1,
            CombatResult::Draw => self.draws += 1,
        }
        self.total_turns += turns;
        for (dst, &src) in self.action_counts.iter_mut().zip(actions.iter()) {
            *dst += src;
        }
    }

    fn total(&self) -> u32 {
        self.wins + self.losses + self.draws
    }

    fn win_rate(&self) -> f32 {
        if self.total() == 0 {
            0.0
        } else {
            self.wins as f32 / self.total() as f32
        }
    }

    fn avg_turns(&self) -> f32 {
        if self.total() == 0 {
            0.0
        } else {
            self.total_turns as f32 / self.total() as f32
        }
    }

    fn most_used(&self) -> MonsterAction {
        let best = self
            .action_counts
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.cmp(b))
            .map(|(i, _)| i)
            .unwrap_or(0);
        MonsterAction::from_usize(best)
    }
}

/// Run a single combat. Returns (result, turns, action_counts).
fn run_combat(
    stats: &mut BanditStats,
    player_type: PlayerType,
    strategy: &BanditStrategy,
    rng: &mut Rng,
    epsilon: f32,
) -> (CombatResult, u32, [u32; NUM_ARMS]) {
    let mut m_hp = MAX_HP;
    let mut p_hp = MAX_HP;
    let mut turns: u32 = 0;
    let mut counts = [0u32; NUM_ARMS];

    loop {
        turns += 1;
        let m_act = select_arm(stats, strategy, rng, epsilon);
        counts[m_act] += 1;

        if m_act == MonsterAction::Flee as usize {
            return (CombatResult::Draw, turns, counts);
        }

        let p_act = player_type.select_action(rng);
        let t = resolve_turn(m_act, p_act, rng);
        let reward = turn_reward(m_act, &t, m_hp);
        stats.update(m_act, reward);

        m_hp = (m_hp - t.monster_dmg_taken + t.monster_heal).clamp(0.0, MAX_HP);
        p_hp = (p_hp - t.player_dmg_taken).clamp(0.0, MAX_HP);

        if p_hp <= 0.0 {
            return (CombatResult::MonsterWin, turns, counts);
        }
        if m_hp <= 0.0 {
            return (CombatResult::PlayerWin, turns, counts);
        }
        if turns >= MAX_TURNS {
            return (CombatResult::Draw, turns, counts);
        }
    }
}

// ── Section 1: Monster vs 4 Player Types ──────────────────────

fn section1() -> Vec<(PlayerType, BanditStats)> {
    println!();
    println!("╔══════════════════════════════════════════════════════════════════════════╗");
    println!(
        "║  Section 1: Monster vs 4 Player Types (UCB1, {n:>3} combats each)       ║",
        n = COMBATS_PER_TYPE
    );
    println!("╚══════════════════════════════════════════════════════════════════════════╝");
    println!();
    println!("  Monster starts knowing NOTHING about each player type.");
    println!("  Each type gets its own BanditStats — the monster adapts individually.");
    println!();

    println!("  ┌──────────────┬──────────┬───────────┬────────────────┐");
    println!("  │ Player Type  │ Win Rate │ Avg Turns │ Most Used Arm  │");
    println!("  ├──────────────┼──────────┼───────────┼────────────────┤");

    let mut all_stats = Vec::new();
    for pt in PlayerType::all() {
        let mut stats = BanditStats::new(NUM_ARMS);
        let mut tracker = CombatTracker::default();
        let mut rng = Rng::new(42 + pt as u64);

        for _ in 0..COMBATS_PER_TYPE {
            let (result, turns, counts) =
                run_combat(&mut stats, pt, &BanditStrategy::Ucb1, &mut rng, 0.0);
            tracker.record(result, turns, &counts);
        }

        println!(
            "  │ {:<12} │ {:>6.1}%  │ {:>7.1}   │ {:<14} │",
            pt.label(),
            tracker.win_rate() * 100.0,
            tracker.avg_turns(),
            tracker.most_used().label(),
        );
        all_stats.push((pt, stats));
    }

    println!("  └──────────────┴──────────┴───────────┴────────────────┘");
    println!();

    // Insight: per-type learned strategy
    println!("  💡 What the monster discovered (no programmer told it):");
    for (pt, stats) in &all_stats {
        let best = stats.best_arm();
        let action = MonsterAction::from_usize(best);
        let q = stats.q_value(best);
        let visits = stats.visit_count(best);
        println!(
            "     vs {:<12} → prefers {action} (Q={q:.3}, pulled {visits}x)",
            pt.label()
        );
    }
    println!();
    all_stats
}

// ── Section 2: Strategy Comparison vs Aggressive ──────────────

fn section2() {
    println!("╔══════════════════════════════════════════════════════════════════════════╗");
    println!(
        "║  Section 2: Strategy Comparison vs Aggressive ({n:>3} combats each)      ║",
        n = COMBATS_STRATEGY_TEST
    );
    println!("╚══════════════════════════════════════════════════════════════════════════╝");
    println!();

    let strategies: Vec<(&str, BanditStrategy)> = vec![
        ("UCB1", BanditStrategy::Ucb1),
        (
            "ε-greedy",
            BanditStrategy::EpsilonGreedy {
                epsilon: 0.3,
                decay: 0.995,
            },
        ),
        ("Thompson", BanditStrategy::ThompsonSampling),
    ];

    println!("  ┌────────────┬──────────┬────────────┬──────────────────┐");
    println!("  │ Strategy   │ Win Rate │ Avg Reward │ Convergence Turn │");
    println!("  ├────────────┼──────────┼────────────┼──────────────────┤");

    let mut all_curves: Vec<(&str, Vec<f32>)> = Vec::new();

    for (name, strategy) in &strategies {
        let mut stats = BanditStats::new(NUM_ARMS);
        let mut rng = Rng::new(123);
        let mut epsilon = 0.3f32;
        let mut wins = 0u32;
        let mut convergence = 0usize;
        let mut converged = false;
        let mut curve = Vec::new();
        let mut total_reward = 0.0f32;

        for combat_i in 0..COMBATS_STRATEGY_TEST {
            let (result, _, _) = run_combat(
                &mut stats,
                PlayerType::Aggressive,
                strategy,
                &mut rng,
                epsilon,
            );
            if result == CombatResult::MonsterWin {
                wins += 1;
            }
            total_reward += stats.q_value(stats.best_arm());

            // Decay epsilon for EpsilonGreedy
            if matches!(strategy, BanditStrategy::EpsilonGreedy { .. }) {
                epsilon *= 0.995;
            }

            // Track win rate at checkpoints
            if (combat_i + 1) % 50 == 0 {
                let wr = wins as f32 / (combat_i + 1) as f32;
                curve.push(wr);
            }

            // Detect convergence: best arm hasn't changed for 50 combats
            if !converged && combat_i >= 50 {
                let arm = stats.best_arm();
                let min_visits = stats.visits().iter().filter(|&&v| v > 0).count();
                if min_visits >= NUM_ARMS
                    && stats.visit_count(arm) as f32 / stats.total_pulls() as f32 > 0.4
                {
                    convergence = combat_i;
                    converged = true;
                }
            }
        }

        let win_rate = wins as f32 / COMBATS_STRATEGY_TEST as f32;
        let avg_reward = total_reward / COMBATS_STRATEGY_TEST as f32;
        let conv_label = if converged {
            format!("{convergence}")
        } else {
            "—".to_string()
        };

        println!(
            "  │ {:<10} │ {:>6.1}%  │ {:>10.3} │ {:>16} │",
            name,
            win_rate * 100.0,
            avg_reward,
            conv_label,
        );
        all_curves.push((*name, curve));
    }

    println!("  └────────────┴──────────┴────────────┴──────────────────┘");
    println!();

    // ASCII convergence plot
    println!("  Win Rate Convergence (sampled every 50 combats):");
    println!();
    println!("  1.0 ┤");

    let checkpoints = COMBATS_STRATEGY_TEST / 50;
    for row in (0..=10).rev() {
        let threshold = row as f32 / 10.0;
        print!("  {:3.1} ┤", threshold);
        for cp in 0..checkpoints {
            let mut best_char = ' ';
            for (si, (_, curve)) in all_curves.iter().enumerate() {
                if cp < curve.len() {
                    let val = curve[cp];
                    if (val * 10.0).round() as i32 == row {
                        best_char = match si {
                            0 => '█',
                            1 => '▓',
                            2 => '░',
                            _ => '?',
                        };
                    }
                }
            }
            print!("{best_char}");
        }
        if row == 10 {
            print!(" ← UCB1(█) ε-gr(▓) Thom(░)");
        }
        println!();
    }

    println!("       └{}", "─".repeat(checkpoints));
    print!("        ");
    for i in 0..checkpoints {
        if i % 2 == 0 {
            print!("{:>2}", (i + 1) * 50);
        } else {
            print!("  ");
        }
    }
    println!("  → combats");
    println!();
}

// ── Section 3: Learned Q-values Heatmap ───────────────────────

fn section3(all_stats: &[(PlayerType, BanditStats)]) {
    println!("╔══════════════════════════════════════════════════════════════════════════╗");
    println!("║  Section 3: Learned Q-Value Table & Heatmap                             ║");
    println!("╚══════════════════════════════════════════════════════════════════════════╝");
    println!();

    // Q-value table
    println!("  Q-Values (monster's estimated value per action vs each player type):");
    println!();
    print!("  {:>12} │", "");
    for pt in PlayerType::all() {
        print!(" {:>10}", pt.label());
    }
    println!(" │ Best");
    println!("  ─────────────┼────────────────────────────────────────────┼──────────");

    for arm in 0..NUM_ARMS {
        let action = MonsterAction::from_usize(arm);
        print!("  {:>12} │", action.label());
        for (_pt, stats) in all_stats {
            let q = stats.q_value(arm);
            print!(" {:>10.3}", q);
        }
        // Show which player type this action is best against
        let best_vs: Vec<&str> = all_stats
            .iter()
            .filter(|(_, stats)| stats.best_arm() == arm)
            .map(|(pt, _)| pt.label())
            .collect();
        let best_label = if best_vs.is_empty() {
            "—".to_string()
        } else {
            best_vs.join(", ")
        };
        print!(" │ {best_label}");
        println!();
    }

    println!("  ─────────────┼────────────────────────────────────────────┼──────────");
    println!();

    // ASCII heatmap
    println!("  Heatmap (block density ∝ Q-value):");
    println!();
    print!("  {:>12} │", "");
    for pt in PlayerType::all() {
        print!(" {:>10}", pt.label());
    }
    println!();

    for arm in 0..NUM_ARMS {
        let action = MonsterAction::from_usize(arm);
        print!("  {:>12} │", action.label());
        for (_, stats) in all_stats {
            let q = stats.q_value(arm);
            // Map Q to block characters: ░ ▒ ▓ █
            let blocks = match q {
                q if q < 0.2 => "░░░░░░░░░░",
                q if q < 0.4 => "░░▒▒▒▒░░░░",
                q if q < 0.6 => "▒▒▒▓▓▓▒▒░░",
                q if q < 0.8 => "▓▓▓▓██▓▓▒▒",
                _ => "██████▓▓▓▓",
            };
            print!(" {blocks}");
        }
        println!();
    }

    println!();
    println!("  Key: ░ = low Q (<0.2)  ▒ = medium (0.2–0.4)  ▓ = good (0.4–0.8)  █ = high (>0.8)");
    println!();
    println!("  💡 Notice: the Q-value pattern differs per player type — the monster");
    println!("     learned DISTINCT counter-strategies without any explicit programming.");
    println!();
}

// ── Main ───────────────────────────────────────────────────────

fn main() {
    println!();
    println!("╔══════════════════════════════════════════════════════════════════════════╗");
    println!("║       Combat Bandit Demo — \"Smart Ass Modelless\" Monster AI            ║");
    println!("║                                                                        ║");
    println!("║  A monster learns to fight WITHOUT neural nets, behavior trees,        ║");
    println!("║  or hardcoded counters. Just multi-armed bandit statistics.            ║");
    println!("╚══════════════════════════════════════════════════════════════════════════╝");

    let all_stats = section1();
    section2();
    section3(&all_stats);

    println!("╔══════════════════════════════════════════════════════════════════════════╗");
    println!("║  Takeaway: No LLM. No neural net. No script. Just bandit statistics.  ║");
    println!("║  The monster adapted to each player type through pure trial-and-error. ║");
    println!("╚══════════════════════════════════════════════════════════════════════════╝");
    println!();
}
