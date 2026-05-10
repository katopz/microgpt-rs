# Plan 033: Bomberman HL Arena — Implementation Summary

**Branch:** `develop/feature/033_bomberman_arena`
**Commit:** `423281edf`
**Status:** Complete (10/10 tasks)

---

## Architecture

### bevy_ecs World-Based Tick Loop

The arena uses `bevy_ecs` **standalone** (not the full Bevy engine) for a deterministic, tick-based game loop. All systems operate on `&mut World` directly — no ECS schedule, no real-time delta, no plugins.

```
init_world(seed)
  ├─ ArenaGrid::generate(seed)     → 13×13 procedural grid
  ├─ GameRng, TickCounter, ScoreBoard → resources
  └─ Events<GameEvent>             → event bus

spawn_players(world)
  └─ 4 entities at corner spawns with Player, GridPos, BombCount, BombRange, Speed, Alive

run_tick(world, actions) → bool   // returns false when round ends
  ├─ tick_bomb_fuses()            → countdown, collect expired
  ├─ process_explosions()         → blast propagation (cardinal, stops at walls)
  ├─ apply_movement()             → move players, wall collision
  ├─ place_bombs()                → spawn bomb entities if action=Bomb
  ├─ collect_powerups()           → walk over hidden power-ups
  └─ cleanup_and_check()          → kill players in blast, check round end
```

### Grid Layout (13×13)

Standard Bomberman layout generated from seed:
- **Border walls** — fixed perimeter
- **Interior pillars** — fixed walls at even (x, y) intersections
- **Destructible walls** — ~40% fill with hidden power-ups (`BombUp`, `FireUp`, `SpeedUp`)
- **Spawn zones** — 3×3 corners kept clear at (1,1), (11,1), (1,11), (11,11)

### ECS Components & Resources

| Component | Purpose |
|-----------|---------|
| `Player { id }` | Player identity (0–3) |
| `GridPos { x, y }` | Position on grid |
| `Bomb`, `BombFuse`, `BombRange` | Bomb entity with countdown and blast radius |
| `BombCount { max, active }` | Per-player bomb limit tracking |
| `Speed { cells_per_tick }` | Movement speed |
| `Alive` | Marker component (removed on death) |
| `DestructibleWall` | Destructible wall entity |
| `PowerUp { kind }` | Collectible power-up |
| `Blast` | Visual blast marker (1-tick lifetime) |

| Resource | Purpose |
|----------|---------|
| `ArenaGrid` | 13×13 grid of `Cell` enum |
| `GameRng` | Deterministic seed |
| `TickCounter` | Current tick number |
| `ScoreBoard` | Per-player scores |
| `PlayerEntities` | 4 player `Entity` ids |

### Events

```rust
enum GameEvent {
    PlayerMoved { player, from, to },
    BombPlaced { player, pos },
    BombExploded { pos, range },
    PlayerKilled { victim, killer },
    PowerUpCollected { player, kind },
    WallDestroyed { pos },
    RoundEnd { survivors },
}
```

---

## Player Types (4 HL Tech Levels)

### P1 🐰 RandomPlayer — Baseline

- **Tech:** None. Uniform random from 6 actions.
- **Constraint:** Wall collision only (ECS rejects invalid moves).
- **No learning, no memory, no model.** Pure baseline for comparison.

### P2 🐱 GreedyPlayer — Heuristic (simulates LoRA)

- **Tech:** Heuristic scoring simulating LoRA draft model marginals.
- **Selection:** Scores each action by proximity to opponents, power-ups, destructible walls.
- **Safety:** Penalizes walking into walls. No blast avoidance.
- **Simulates:** What a LoRA model would produce — better than random, but no safety rules.

### P3 🐶 ValidatorPlayer — Heuristic + Safety Rules (simulates LoRA + WASM)

- **Tech:** Same heuristic as P2, plus hard safety validation (simulating WASM `ScreeningPruner`).
- **Validation rules:**
  - Reject walking into blast zones (tracks known bomb positions + ranges)
  - Reject placing bomb with no escape route (BFS checks reachable safe cell)
  - Boost escape actions when in danger zone
  - Boost power-up collection when safe
- **Result:** Near-perfect survival — the validator prevents virtually all suicides.

### P4 🐵 HLPlayer — Full HL (Bandit + AbsorbCompress)

- **Tech:** P3 base + bandit Q-values over 6 actions + absorb-compress cycle.
- **Blended scoring:** `60% heuristic + 40% bandit Q-value + safety penalty`
- **ε-greedy:** 10% explore, 90% exploit (explore random non-compressed actions)
- **Absorb-Compress:** Every 100 rounds, arms with `visits ≥ 20 && Q < 0.1` get hard-blocked (compressed).
- **Round memory:** Tracks all actions taken in a round; distributes reward proportionally.
- **Reward shaping:** `+1.0 survive, -1.0 die, +0.5 kill, +0.2/powerup`
- **Persists across rounds:** Q-values, visits, and compressed arms carry forward.

---

## Key Files

### New Files (8)

| File | Lines | Purpose |
|------|-------|---------|
| `src/pruners/bomber/mod.rs` | 304 | Module index: enums, components, resources, events, constants |
| `src/pruners/bomber/arena.rs` | 195 | Procedural 13×13 grid generation with `ArenaGrid::generate(seed)` |
| `src/pruners/bomber/systems.rs` | 530 | World-based ECS systems: `init_world`, `spawn_players`, `run_tick` |
| `src/pruners/bomber/players.rs` | ~820 | `BomberPlayer` trait + 4 implementations |
| `examples/bomber_01_arena.rs` | 232 | Headless 10-round tournament runner |
| `examples/bomber_02_tui.rs` | 506 | Animated ratatui TUI replay with emoji rendering |
| `examples/bomber_03_hl_proof.rs` | 457 | 1000-round HL proof experiment with golden traces |
| `tests/bench_bomber_arena.rs` | ~100 | 4 benchmark tests |

### Modified Files (3)

| File | Changes |
|------|---------|
| `Cargo.toml` | `bevy_ecs = "0.15"` optional dep, `bomber` feature, 3 example entries |
| `src/pruners/mod.rs` | `pub mod bomber` + 13 re-exports |
| `.plans/033_bomberman_arena.md` | All 10 tasks marked complete |

### Public API (from `src/pruners/bomber/mod.rs`)

```rust
// Enums
pub enum BomberAction { Up, Down, Left, Right, Bomb, Wait }
pub enum PowerUpKind { BombUp, FireUp, SpeedUp }
pub enum Cell { Floor, FixedWall, DestructibleWall, PowerUpHidden(PowerUpKind) }
pub enum GameEvent { PlayerMoved, BombPlaced, BombExploded, PlayerKilled, PowerUpCollected, WallDestroyed, RoundEnd }

// Components
pub struct Player { pub id: u8 }
pub struct GridPos { pub x: i32, pub y: i32 }
pub struct BombFuse { pub owner: Entity, pub ticks_remaining: u32 }
// ... BombRange, BombCount, Speed, Alive, etc.

// Resources
pub struct ArenaGrid { pub cells: Vec<Vec<Cell>>, pub width: usize, pub height: usize }
pub struct GameRng { pub seed: u64 }
pub struct TickCounter(pub u32)
pub struct ScoreBoard { pub scores: [i32; 4] }
pub struct PlayerEntities { pub entities: [Entity; 4] }

// Systems
pub fn init_world(seed: u64) -> World
pub fn spawn_players(world: &mut World) -> [Entity; 4]
pub fn run_tick(world: &mut World, actions: [Option<BomberAction>; 4]) -> bool

// Players
pub trait BomberPlayer { fn select_action(...); fn name(); fn emoji(); fn reset(); }
pub struct RandomPlayer
pub struct GreedyPlayer
pub struct ValidatorPlayer
pub struct HLPlayer  // with update_outcome(), compress_cycle(), compress_report()
```

---

## Benchmark Results

All benchmarks run with `cargo test --features bomber bench_bomber_arena -- --nocapture`.

| Component | Target | Actual | Status |
|-----------|--------|--------|--------|
| Arena generation | <100µs | **~12µs** | ✅ 8× under target |
| Single tick (4 players) | <50µs | **~30µs** | ✅ |
| Full game (200 ticks) | <10ms | **~5.6ms** | ✅ |
| P4 HL decision | <200µs | **~849ns** | ✅ 235× under target |

---

## HL Experiment Results (1000 rounds)

Run with `cargo run --example bomber_03_hl_proof --features bomber`.

| Player | Emoji | Tech | Survival Rate | Kill Rate |
|--------|-------|------|---------------|-----------|
| P1 | 🐰 | Random | 0.0% | — |
| P2 | 🐱 | Greedy | 0.3% | — |
| P3 | 🐶 | Validator | **99.9%** | Low |
| P4 | 🐵 | Full HL | **90.4%** | Moderate |

### Key Observations

1. **P3 (Validator) achieves 99.9% survival** — safety rules are extremely effective at preventing self-destruction.
2. **P4 (HL) at 90.4%** — the bandit's 10% exploration (ε-greedy) causes occasional deaths, trading safety for learning.
3. **P4's bandit adapts** — after compress cycles, low-Q arms get hard-blocked, reducing future mistakes.
4. **The P4 > P3 proof is NOT yet achieved** — P3's static safety outperforms P4's adaptive learning in this configuration.

### Why P3 > P4 (Unexpected Result)

The current implementation reveals an important insight: **pure survival is not enough**. P3's hard safety rules are near-optimal for staying alive, but they're passive — P3 rarely kills opponents. P4 explores more aggressively (bandit-driven), which leads to more kills but also more deaths.

To prove P4 > P3, the reward shaping and/or competitive metric needs adjustment (see Future Improvements).

---

## How to Run

```bash
# Headless 10-round tournament
cargo run --example bomber_01_arena --features bomber

# Animated TUI replay (keyboard controls: ←/→/Space/Q)
cargo run --example bomber_02_tui --features bomber

# 1000-round HL proof experiment with stats
cargo run --example bomber_03_hl_proof --features bomber

# Benchmarks
cargo test --features bomber bench_bomber_arena -- --nocapture

# Tests (20 bomber tests)
cargo test --features bomber
```

---

## Future Improvements (P4 > P3 Proof)

The current results show P3 (static validator) > P4 (HL) on survival. To prove the HL thesis that **adaptive > static**, the following improvements are needed:

### 1. Reward Shaping for Kills, Not Just Survival

- P3's 99.9% survival is "safe but passive" — it rarely places bombs near opponents.
- P4's bandit needs positive reward for opponent kills to learn aggressive-yet-safe play.
- Proposed: `+2.0 per kill, +1.0 survive, -1.0 die` weighting.

### 2. Contextual Bandit (State-Dependent Q-Values)

- Current bandit is flat: one Q-value per action regardless of board state.
- Need per-state features (e.g., "am I in blast zone?", "is opponent adjacent?") with separate Q-tables.
- This would let P4 learn "bomb is safe HERE but suicide THERE."

### 3. Multi-Step Credit Assignment

- Current reward distributes equally across all round actions.
- Need temporal difference (TD) learning: actions closer to outcome get more credit.
- Proposed: discounted reward `γ^steps_from_end * reward`.

### 4. Opponent Modeling

- P4 currently treats all opponents identically.
- Could track opponent patterns (e.g., "P3 always runs from bombs") and exploit them.
- This is where HL truly separates from static validators.

### 5. Real LoRA Integration

- P2 (Greedy) simulates LoRA with heuristics; replace with actual `lora.bin` from `riir-burner` trained on bomberman game traces.
- This would make the P1→P2→P3→P4 ladder a true model→validator→HL pipeline.

### 6. Tournament Metric: Score-Based, Not Survival-Based

- Current metric is survival rate, which favors passive play.
- Switch to composite score: `score = kills * 5 + survival * 2 + powerups`.
- This rewards active, intelligent play over hiding in corners.

### 7. Decreasing Exploration Over Time

- P4's ε=0.1 is constant; should decay as rounds increase.
- Proposed: `ε = max(0.02, 0.1 * 0.995^round)` — explores early, exploits late.
- After enough learning, P4 should match P3's safety while retaining learned aggression.

---

## References

- **Plan:** `.plans/033_bomberman_arena.md`
- **Reference impl:** `raw/bomby/` — Fish Folk: Bomby (Apache-2.0 / MIT)
- **Prior plans:** Plan 032 (HL Infrastructure), Plan 030 (Bandit), Plan 021 (ScreeningPruner)
- **Research:** `.research/14_Learning_Beyond_Gradients.md`
- [bevy_ecs standalone docs](https://docs.rs/bevy_ecs)