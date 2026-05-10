# microgpt-rs: Heuristic Learning

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

## References

- [Learning Beyond Gradients](https://trinkle23897.github.io/learning-beyond-gradients/) — Jiayi Weng, 2026
- Plan 030: Multi-Armed Bandit
- Plan 032: HL Infrastructure
- Plan 033: Bomberman Arena
- Research 14: HL Distillation