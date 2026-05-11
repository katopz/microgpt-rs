# microgpt-rs: Heuristic Learning

> **Status (Plan 036):** ReviewMetrics, ReviewStrategy, and benefit-ratio gating are implemented behind `--features bandit`. AbsorbCompress gates compression by benefit-risk ratio. `ppot_rescue_reviewed` provides structured review loops behind `--features bandit,ppot`. See example `review_01_metrics`.
>
> **Status (Plan 032):** TrialLog, AbsorbCompress, HotSwapPruner, and RegressionSuite are implemented behind `--features bandit`. See examples `hl_01_trial_log` and `hl_02_hotswap`.

## What is Heuristic Learning?

Heuristic Learning (HL) is a paradigm where **software systems evolve through code updates** rather than neural network weight updates. A coding agent reads feedback (test failures, environment rewards, logs, replays) and directly edits policies, validators, tests, and memory — no backpropagation required.

> Source: [Learning Beyond Gradients](https://trinkle23897.github.io/learning-beyond-gradients/) by Jiayi Weng

### The Core Idea

```
Traditional ML:  data → gradient → update weights → better model
Heuristic Learning: feedback → agent edits code → better rules → better system
```

The coding agent changes the **maintenance cost curve** for heuristics. Rules that were once "too expensive to own" become viable long-term code when an agent can maintain them.

---

## HL in microgpt-rs

microgpt-rs is uniquely positioned for HL because of its **trait-based pruner architecture** and **WASM sandbox**:

| HL Concept | microgpt-rs Component |
|---|---|
| Heuristic Policy | `ConstraintPruner::is_valid()` — masks invalid tokens/actions |
| Relevance Scoring | `ScreeningPruner::relevance()` — prioritizes good actions |
| Gradient-free Learning | `BanditPruner` — Q-value updates without backprop |
| Sandboxed Heuristics | `WasmPruner` — compiled validators in WASM sandbox |
| Trial History | `TrialLog` — persistent JSONL episode records |
| Rule Compression | `AbsorbCompress` — promote stable Q-values to hard constraints |
| Hot-reload | `HotSwapPruner` — runtime .wasm reload |
| Regression Safety | `RegressionSuite` — replay golden episodes |

---

## The Two Operations

A healthy Heuristic System needs two operations (from the HL research):

### 1. Absorb

Feed new observations back into the system:

```
Episode N:   BanditPruner selects arm → environment runs → reward
             TrialLog.append(episode, arm, reward, q_value, note)
             AbsorbCompress.absorb(arm, reward)
```

### 2. Compress

Fold accumulated knowledge into simpler, more maintainable rules:

```
After N episodes:
  arm 3 (Wait) has Q=0.02 over 500 visits → promote to hard block
  arm 0 (Attack near enemy) has Q=0.89 → boost relevance weight
  → AbsorbCompress.compress() returns [3]
  → BanditPruner delegates arm 3 to BlockedArmPruner
```

> An HS that only grows and never compresses becomes a big ball of mud.

---

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                  Heuristic System                     │
│                                                       │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────┐ │
│  │ ConstraintPruner│ │ScreeningPruner│ │ BanditPruner│ │
│  │ is_valid()    │  │ relevance()  │  │ relevance() │ │
│  │ hard block    │  │ soft score   │  │ adaptive    │ │
│  └──────┬───────┘  └──────┬───────┘  └──────┬─────┘ │
│         │                 │                  │        │
│         ▼                 ▼                  ▼        │
│  ┌─────────────────────────────────────────────────┐  │
│  │              WASM Validator (.wasm)              │  │
│  │  Sandbox: no I/O, no floating point, 4MB max    │  │
│  └────────────────────┬────────────────────────────┘  │
│                        │                               │
│         ┌──────────────┼──────────────┐                │
│         ▼              ▼              ▼                │
│  ┌────────────┐ ┌────────────┐ ┌──────────────┐      │
│  │  TrialLog  │ │AbsorbCompress│ │RegressionSuite│    │
│  │  (JSONL)   │ │ (Q→blocks)  │ │ (golden)     │      │
│  └────────────┘ └────────────┘ └──────────────┘      │
└──────────────────────────────────────────────────────┘
```

---

## The HL Feedback Loop

### During Episodes (Online Learning)

```
1. DDTree proposes branches (tokens/actions)
2. BanditPruner.relevance() scores each branch
3. Best branch selected → environment executes
4. Reward observed → BanditPruner.update(arm, reward)
5. TrialLog.append(record) for persistence
6. AbsorbCompress.absorb(arm, reward) for compression check
```

### Between Episodes (Compression)

```
1. AbsorbCompress.should_compress() → true (threshold met)
2. AbsorbCompress.compress() → identify arms to promote
3. Low-Q arms → BlockedArmPruner (hard constraints)
4. High-Q arms → boost relevance weight
5. RegressionSuite.replay_golden() → verify no regression
```

### Between Rounds (Evolution)

```
1. Agent reads TrialLog → identifies failure patterns
2. Agent writes new validator.rs → compile to .wasm
3. HotSwapPruner.reload() → load new .wasm
4. RegressionSuite.replay_golden() → verify improvement
5. Keep or revert based on regression results
```

---

## System 1 / System 2

The HL paradigm suggests a split between fast intuition and slow deliberation:

### System 1 (Fast, ~100µs)

The inference hot path:

```
LoRA Draft Model → DDTree Branches → BanditPruner scores → WasmPruner validates
                                                     ↓
                                              Select best valid action
```

- **LoRA model**: "intuition" about good actions (marginals)
- **BanditPruner**: adaptive scoring based on past experience (Q-values)
- **WasmPruner**: hard safety constraints (validation rules)

### System 2 (Slow, seconds)

The evolution loop:

```
TrialLog → Agent reads failures → Writes new validator → compile .wasm → HotSwap → Regression test
```

- **TrialLog**: persistent memory of what worked and what didn't
- **AbsorbCompress**: automatic rule promotion from experience
- **RegressionSuite**: safety net against regressions
- **Coding agent**: writes new validators based on failure analysis

---

## Bomberman HL Arena (Proof of Concept)

4 AI players compete in a Bomberman arena built with `bevy_ecs` (standalone) + ratatui emoji TUI. Game logic patterns adapted from `raw/bomby/` (Fish Folk: Bomby) — same ECS components, resources, and systems, but tick-based instead of real-time.

### Architecture

```
raw/bomby/ (reference)              →  microgpt-rs bomberman (ours)
──────────────────────────────────────────────────────────────────
bevy (full engine)                  →  bevy_ecs (standalone ECS only)
bevy_ecs_ldtk (LDtk level loading)  →  ProceduralArena (grid generator)
bevy Sprite / TextureAtlas          →  ratatui emoji TUI
bevy_kira_audio                     →  (none — silent)
bevy Time (real delta)              →  discrete tick counter
leafwing-input-manager              →  BomberPlayer trait (AI selects)
bevy Commands / EventWriter         →  bevy_ecs Commands / EventWriter ✅
bevy Query / Resource / Plugin      →  bevy_ecs (same patterns) ✅
```

### Players (Technology Ladder)

| Player | Emoji | Tech Stack | What It Proves |
|---|---|---|---|
| P1: Random | 🐰 | `fastrand::random()` | Baseline |
| P2: Model | 🐱 | LoRA marginals | Model > random |
| P3: Validator | 🐶 | LoRA + WASM pruner | Validator > model alone |
| P4: Full HL | 🐵 | LoRA + WASM + Bandit + TrialLog + AbsorbCompress | HL > static rules |

The P3 vs P4 comparison is the key proof: both use the same model and validator, but P4 adapts through bandit learning while P3 uses static rules.

### ECS Components (from bomby patterns)

```rust
#[derive(Component)] struct Player { id: u8 }
#[derive(Component)] struct GridPos { x: i32, y: i32 }
#[derive(Component)] struct BombFuse { owner: Entity, ticks_remaining: u32 }
#[derive(Component)] struct BombRange { cells: u32 }
#[derive(Component)] struct BombCount { max: u8, active: u8 }
#[derive(Component)] struct Speed { cells_per_tick: u8 }
#[derive(Component)] struct Alive;
```

### TUI Grid (emoji rendering)

| Cell | Emoji | Cell | Emoji |
|------|-------|------|-------|
| Floor | `··` | Fixed wall | `🧱` |
| Destructible | `📦` | Player 1-4 | `🐰🐱🐶🐵` |
| Bomb (fresh) | `💣` | Bomb (low fuse) | `🧨` |
| Blast | `💥` | PowerUp | `🔥💥👟` |

See [Plan 033](/.plans/033_bomberman_arena.md) for full implementation details.

---

## Quick Start

```rust
use microgpt_rs::pruners::{
    BanditPruner, BanditStrategy, AbsorbCompress, TrialLog, CompressConfig,
};

// Create a bandit pruner with absorb-compress
let mut bandit = BanditPruner::new(
    domain_screener,
    BanditStrategy::Ucb1,
    6, // 6 arms (actions)
);

// Create trial log
let mut trial_log = TrialLog::new("/tmp/hl_trials.jsonl")?;

// Run episodes
for episode in 0..1000 {
    let arm = bandit.best_arm();  // select via strategy
    let reward = env.pull(arm);   // environment feedback
    
    bandit.update(arm, reward);
    trial_log.append(TrialRecord {
        episode, arm, reward,
        q_value: bandit.q_value(arm),
        cumulative_reward: bandit.total_pulls() as f32 * reward,
        cumulative_regret: 0.0,
        config: String::new(),
        note: String::new(),
    });
    
    // Absorb-compress check every 100 episodes
    if episode % 100 == 0 && bandit.should_compress() {
        let promoted = bandit.compress();
        println!("Compressed arms: {promoted:?}");
    }
}
```

---

## Slot Machine Bandit: Rules-Based Speculative Decoding (Plan 031)

A slot machine that closes the full speculative decoding loop with **no real transformer needed**:

```
Reel weights → DDTree → Payline rules → Reward → Bandit learns → Repeat
```

Unlike `bandit_demo.rs` (coin flips, disclaimer required) and `bandit_ddtree_demo.rs` (random marginals, random verification), this demo uses **structured reel weights** as marginals and **deterministic payline rules** for verification — proving actual value, not just mechanical compatibility.

### Slot ↔ Speculative Decoding Analogy

| Speculative Decoding | Slot Machine |
|---------------------|--------------|
| Draft model marginals P(token\|context) | Reel weights P(symbol\|reel) |
| Target model verification | Payline rules (combo valid?) |
| Accept → 1.0, Reject → 0.0 | Payout table (graded 0.0–1.0) |
| BanditPruner screens branches | Bandit learns which symbols pay |

### Slot Machine Configuration

6 symbols (vocab_size=6), 3 reels (lookahead=3):

| Symbol | Reel 0 | Reel 1 | Reel 2 | Payout (Triple) |
|--------|--------|--------|--------|-----------------|
| 🍒 Cherry | 30% | 25% | 20% | 0.5 |
| 🍋 Lemon | 25% | 20% | 20% | 0.5 |
| 🍊 Orange | 20% | 20% | 20% | 0.5 |
| 🔔 Bell | 15% | 15% | 15% | 0.6 |
| 💎 Diamond | 7% | 10% | 15% | 0.8 |
| 7️⃣ Seven | 3% | 10% | 10% | 1.0 (JACKPOT) |

### Results (500 episodes, seed=42)

| Strategy | Total Reward | Avg Reward | Best Combo | Triples | vs Random |
|----------|-------------|------------|------------|---------|-----------|
| UCB1 | 82.40 | 0.1648 | 🍒🍒🍒 | 6 | +60.9% |
| ε-greedy | 250.10 | 0.5002 | 🔔🔔🔔 | 500 | +388.5% |
| Thompson | 247.30 | 0.4946 | 🔔🔔🔔 | 490 | +383.0% |
| Random | 51.20 | 0.1024 | 💎💎💎 | 17 | baseline |

All bandit strategies significantly outperform random. ε-greedy and Thompson converge to Bell triples (reliable 0.6 reward) while random occasionally hits Diamond triples by luck.

Run: `cargo run --example bandit_03_slot --features bandit`

---

## Model vs Modelless Bandit: Proven Results (Plan 025)

Two demos prove whether model-based speculative decoding with bandit is worth the cost vs modelless bandit-only.

### bandit_ddtree_demo.rs — Model-Based vs Modelless

Uses simulated marginals (concentrated vs uniform) flowing through real `build_dd_tree_screened()` + `BanditPruner`.

| Metric | Model-based | Modelless | Δ |
|--------|-------------|-----------|---|
| Cumulative Reward | 7880.00 | 7027.00 | **+12.1%** |
| Cumulative Regret | 120.00 | 973.00 | **-87.7%** |
| Accept Rate | 98.5% | 87.8% | **+10.7%** |
| Avg Time/Episode | 70.8 µs | 63.9 µs | +10.8% |

### game_resolver_demo.rs — Domain Validator + Bandit

Uses `GameActionScreener` (native Rust game action validator) as inner pruner for `BanditPruner<GameActionScreener>`.

| Metric | Constrained (domain+bandit) | Unconstrained (bandit only) | Δ |
|--------|----------------------------|-----------------------------|---|
| Cumulative Reward | 2275.00 | 2929.00 | -22.3% |
| Cumulative Regret | 5725.00 | 5071.00 | +12.9% |
| Accept Rate | 75.8% | 36.6% | **+39.2%** |
| Avg Time/Episode | 39.6 µs | 62.5 µs | **-36.6%** |

### Key Findings

1. **Model-based wins on quality**: +12.1% reward, -87.7% regret, +10.7% accept rate
2. **Domain screener dramatically improves accept rate**: +39.2% over bandit alone
3. **Domain screener is faster**: -36.6% latency — pruning invalid branches early reduces DDTree work
4. **Bandit learns meaningful arms**: Constrained converges on game-relevant tokens; Unconstrained spreads visits thinly
5. **Modelless still functional**: 87.8% accept rate proves bandit can learn without model priors, just slower

Run: `cargo run --example bandit_02_ddtree --features bandit`
Run: `cargo run --example bandit_06_resolver --features bandit`

---

## Inference-Time Review Metrics (Plan 036)

Based on arXiv:2604.27233 — "Reinforced Agent: Inference-Time Feedback for Tool-Calling Agents". The paper proves that inference-time review improves tool-calling accuracy by +5.5% on irrelevance detection and +7.1% on multi-turn tasks. The key insight is the **measurement framework**, not the reviewer itself.

### Classification Matrix

Each (base_correct, reviewed_correct) pair is classified into one of four categories:

| Base Correct | Reviewed Correct | Classification | Meaning |
|:---:|:---:|:---:|:---|
| false | true | **Helpful** | Reviewer fixed a wrong answer |
| true | false | **Harmful** | Reviewer broke a correct answer |
| true | true | Both Correct | Both agreed (no effect) |
| false | false | Both Wrong | Both failed (no effect) |

### Benefit-Risk Ratio

```
Benefit-Risk Ratio = Helpfulness ÷ Harmfulness
```

- **Helpfulness** = `helpful / (helpful + both_wrong)` — % of base-wrong cases the reviewer fixed
- **Harmfulness** = `harmful / (harmful + both_correct)` — % of base-correct cases the reviewer broke

The paper found **3.1:1** for o3-mini. Our default threshold is **2.0:1** (conservative — allows slightly worse reviewers).

### How It Connects to Existing Systems

| Component | Review Metrics Integration |
|---|---|
| `ReviewMetrics` | Atomic counters tracking helpful/harmful/both_correct/both_wrong |
| `BanditSession::with_metrics()` | Records whether bandit pick vs random pick was optimal |
| `AbsorbCompress::should_compress_gated()` | Blocks compression when ratio < threshold |
| `PpotConfig::with_review_loop(N)` | Structured review loop (paper's rN) |
| `ppot_rescue_reviewed()` | PPoT rescue with benefit-ratio gate |
| `TrialLog::append_with_review()` | Persist episode with review classification |

### Benefit-Ratio Guidance

| Ratio | Interpretation | Action |
|:---:|:---|:---|
| **> 3.0** | Excellent reviewer (paper quality) | Aggressively compress, trust reviewer |
| **2.0–3.0** | Acceptable reviewer | Compress with normal caution |
| **1.0–2.0** | Marginal reviewer | Gate compression, investigate failures |
| **< 1.0** | Net-negative reviewer | Stop reviewing, revert to base |
| **∞** | Perfect (never broke correct) | Trust fully, but monitor for overfitting |

### Quick Start

```rust,ignore
use std::sync::Arc;
use microgpt_rs::pruners::{BanditSession, BanditStrategy, BernoulliEnv, ReviewMetrics};

let metrics = Arc::new(ReviewMetrics::new());

let session = BanditSession::new(env, BanditStrategy::Ucb1)
    .with_metrics(Arc::clone(&metrics));

let (events, result) = session.run(1000, &mut rng);

// Print review metrics
println!("{metrics}"); // "helpful=83.5% harmful=20.5% ratio=4.1:1 n=1000"

// Check if compression is safe
let ratio = metrics.benefit_ratio();
if ratio >= 2.0 {
    absorb.compress(); // Safe to harden reviewer decisions
}
```

Run: `cargo run --example review_01_metrics --features bandit`

---

## References

- [Learning Beyond Gradients](https://trinkle23897.github.io/learning-beyond-gradients/) — Jiayi Weng, 2026
- [Reinforced Agent: Inference-Time Feedback](https://arxiv.org/abs/2604.27233) — arXiv:2604.27233
- Plan 025: Model vs Modelless Bandit
- Plan 030: Multi-Armed Bandit
- Plan 031: Slot Machine Bandit
- Plan 032: HL Infrastructure
- Plan 033: Bomberman Arena
- Plan 036: Inference-Time Review Metrics
- Research 14: HL Distillation