# Plan 033: Bomberman HL Arena — 4-Player Heuristic Learning Proof

**Branch:** `develop/feature/033_bomberman_arena`
**Depends on:** Plan 032 (HL Infrastructure), Plan 030 (Bandit), Plan 021 (ScreeningPruner)
**Research:** `.research/14_Learning_Beyond_Gradients.md`
**Reference:** `raw/bomby/` — Fish Folk: Bomby (Bevy ECS + LDtk, Apache-2.0 / MIT)
**Goal:** Build a 4-player Bomberman arena using `bevy_ecs` (standalone) + ratatui TUI where each player uses progressively more HL technology. The arena proves the value of each layer: model > random, validator > model, HL > static validator.

---

## Overview

4 AI players compete in a Bomberman arena. Each player represents a rung on the HL technology ladder:

```
P1: Modelless (random)           — baseline
P2: Model-based (lora.bin)       — proves model is worth it
P3: Model + Validator (wasm)     — proves validator is worth it
P4: Full HL (lora + wasm + bandit + triallog + absorb/compress) — proves HL is worth it
```

The competitive format makes the proof self-contained: no external baselines needed, just "who survives."

### Architecture: bevy_ecs (Standalone) + ratatui TUI

We extract game logic patterns from `raw/bomby/` but use **only `bevy_ecs`** (the standalone ECS crate), not the full Bevy engine. Rendering uses ratatui with emoji/ASCII art — same pattern as `dungeon_01_tui.rs`.

```
raw/bomby/ (reference)              →  microgpt-rs bomberman (ours)
──────────────────────────────────────────────────────────────────
bevy (full engine)                  →  bevy_ecs (standalone ECS only)
bevy_ecs_ldtk (LDtk level loading)  →  ProceduralArena (grid generator)
bevy Sprite / TextureAtlas          →  ratatui emoji TUI
bevy_kira_audio                     →  (none — silent)
bevy Time (real delta)              →  discrete tick counter
bevy InputManagerBundle             →  BomberPlayer trait (AI selects)
leafwing-input-manager              →  (none — AI only, no human input)
bevy Commands / EventWriter         →  bevy_ecs Commands / EventWriter ✅
bevy Query<(&Component, &mut ...)>  →  bevy_ecs Query (same pattern) ✅
bevy Resource                       →  bevy_ecs Resource (same pattern) ✅
bevy Plugin / App                   →  bevy_ecs Plugin / App (same pattern) ✅
```

### What We Keep from bomby

| bomby Pattern | Our Usage |
|---|---|
| ECS Components: `Bomb { spawner, timer }`, `CountBombs`, `Player`, `Velocity` | Same components, tick-based instead of float time |
| ECS Resources: `GameRng(SmallRng)` | Same, `GameRng` as bevy_ecs Resource |
| ECS Systems: `spawn_bombs`, `update_bombs`, `movement_input`, `player_collisions` | Same system structure, tick-based logic |
| Grid math: `to_grid()`, `grid_normalised()` | Same coordinate conversion, no LDtk dependency |
| Blast logic: orthogonal range, stops at walls, chain explosions | Same, configurable blast range |
| Collision: wall + bomb blocking, bounds per player | Same, simplified for grid-only movement |

### What We Drop from bomby

| bomby Module | Reason |
|---|---|
| `audio.rs` | No audio in TUI |
| `camera.rs` | No camera in TUI (no 2D world space) |
| `debug.rs` | Bevy inspector, not needed |
| `z_sort.rs` | Z-sorting for sprite overlap, not needed in TUI |
| `ui.rs` | Bevy UI buttons, replaced by ratatui |
| LDtk asset loading | Replaced by procedural grid generation |
| Sprite/texture loading | Replaced by emoji rendering |

### What Each Comparison Proves

| Matchup | Proves | Plan 025 Precedent |
|---|---|---|
| P1 vs P2 | LoRA model > random | +12.1% reward, -87.7% regret |
| P2 vs P3 | Validator adds value on top of model | +39.2% accept rate, -36.6% latency |
| P3 vs P4 | **HL evolution > static rules** | **NOT YET PROVEN — this is the new proof** |

---

## Bomberman Game Rules

### Arena

- 13×13 grid (standard Bomberman layout, same as bomby's LDtk level)
- Fixed walls (indestructible, checkerboard pattern on odd rows/cols)
- Destructible walls (random placement, ~40% fill, same as bomby's "Bombable" layer)
- Open spaces for movement
- 4 corner spawns (players start safe in 3×3 corners, same as bomby)

### Grid Layout (13×13, standard)

```text
C . . # . . . # . . . # . . C    C = corner spawn (player start)
. █ . # . █ . # . █ . # . █ .    █ = destructible wall
. . . # . . . # . . . # . . .    # = fixed wall (indestructible)
# # # # # # # # # # # # # # #    . = floor
. █ . # . █ . # . █ . # . █ .
. . . # . . . # . . . # . . .
# # # # # # # # # # # # # # #
. █ . # . █ . # . █ . # . █ .
. . . # . . . # . . . # . . .
C . . # . . . # . . . # . . C
```

### Actions (6 = vocab_size)

| Action | Token | Emoji | Description |
|--------|-------|-------|-------------|
| Up | 0 | ↑ | Move up one cell |
| Down | 1 | ↓ | Move down one cell |
| Left | 2 | ← | Move left one cell |
| Right | 3 | → | Move right one cell |
| Bomb | 4 | 💣 | Place bomb at current position |
| Wait | 5 | ⏸ | Do nothing this tick |

### Bombs (adapted from bomby's `bomb.rs`)

- Fuse: 4 ticks (bomby uses 1.5s real-time; we use discrete ticks)
- Blast range: starts at 2 cells in each cardinal direction (bomby uses 1; we increase for strategic depth)
- Blast stops at: fixed walls (blocked), destructible walls (destroyed + stopped)
- Chain explosions: blast hitting another bomb triggers it immediately (same as bomby)
- Player can be killed by own bomb (same as bomby)
- Max bombs per player: starts at 1 (bomby uses 2; we start lower for strategy)

### Power-ups (hidden in destructible walls, same as standard Bomberman)

| Power-up | Effect | Emoji |
|----------|--------|-------|
| BombUp | +1 max simultaneous bombs | 💥 |
| FireUp | +1 blast range | 🔥 |
| SpeedUp | +1 moves per tick (default 1, max 2) | 👟 |

### Scoring (per round)

| Event | Score |
|-------|-------|
| Last player standing | +5 |
| Each kill (opponent dies from your blast) | +3 |
| Each power-up collected | +1 |
| Death (any cause) | -3 |
| Suicide (own bomb kills you) | -5 |
| Draw (all die same tick) | 0 for all |

### Win Condition

- Round ends when 0 or 1 players alive, OR tick limit reached (200 ticks)
- If timeout: surviving players get +3 (not +5, since they didn't "win")
- Tournament = N rounds, cumulative score determines ranking

---

## Player Architectures

### P1: Modelless (Random)

```
Arms: {Up, Down, Left, Right, Bomb, Wait}
Selection: uniform random
Constraint: wall collision only (via ECS player_collisions system)
No learning. No memory. No model. Pure baseline.
```

### P2: Model-based (LoRA)

```
Arms: {Up, Down, Left, Right, Bomb, Wait}
Selection: marginals from LoRA draft model
  - model encodes board state → predicts action probabilities
  - sample from model output distribution
Constraint: wall collision only
No bandit. No validator. The model IS the policy.
```

Uses trained LoRA adapter (`lora.bin`) from riir-burner on bomberman game traces.

### P3: Model + Validator (LoRA + WASM)

```
Arms: {Up, Down, Left, Right, Bomb, Wait}
Selection: LoRA marginals → WASM ScreeningPruner
  - model proposes, validator disposes
  - is_valid() → hard reject:
    • don't walk into blast zone
    • don't walk into walls
    • don't place bomb with no escape route
  - relevance() → soft score:
    • boost escape when in danger zone
    • boost powerup collection when safe
    • boost bomb placement near opponents
Constraint: WASM validator (bomber_validator.wasm)
No bandit. Static rules. Expert system.
```

### P4: Full HL (LoRA + WASM + Bandit + TrialLog + Absorb/Compress)

```
Arms: {Up, Down, Left, Right, Bomb, Wait}
Selection: LoRA marginals → WASM pruner → BanditPruner residual → final choice
  - BanditPruner<HotSwapPruner<WasmPruner>>
  - Bandit learns: "in this board state, which actions actually pay off?"
Constraint: same WASM base as P3, PLUS bandit-adapted relevance
Memory: TrialLog (JSONL) — every episode, every death, every kill
Absorb: new failure patterns → bandit Q-value updates
Compress: after N episodes, stable low-Q arms → promoted to hard constraints
HotSwap: between tournament rounds, reload new .wasm

The difference from P3:
  P3's validator is STATIC (written once, never changes)
  P4's validator EVOLVES (bandit adapts, compresses into rules between games)
```

---

## ECS Architecture (bevy_ecs standalone)

### Components (from bomby patterns)

```rust
// ── Marker ──
#[derive(Component)]
struct Player { id: u8 }

#[derive(Component)]
struct Bomb;

#[derive(Component)]
struct PowerUp { kind: PowerUpKind }

#[derive(Component)]
struct Blast;

#[derive(Component)]
struct DestructibleWall;

// ── Data ──
#[derive(Component)]
struct GridPos { x: i32, y: i32 }

#[derive(Component)]
struct BombFuse { owner: Entity, ticks_remaining: u32 }

#[derive(Component)]
struct BombRange { cells: u32 }  // blast range in each direction

#[derive(Component)]
struct BombCount { max: u8, active: u8 }

#[derive(Component)]
struct Speed { cells_per_tick: u8 }

#[derive(Component, Default)]
struct Alive;
```

### Resources

```rust
#[derive(Resource)]
struct ArenaGrid { cells: Vec<Vec<Cell>> }  // 13×13 grid state

#[derive(Resource)]
struct GameRng(SmallRng);  // from bomby

#[derive(Resource)]
struct TickCounter(u32);

#[derive(Resource)]
struct ScoreBoard { scores: [i32; 4] };

#[derive(Resource, Deref, DerefMut)]
struct PlayerEntities([Entity; 4]);
```

### Events (from bomby patterns)

```rust
#[derive(Event)]
enum GameEvent {
    PlayerMoved { player: u8, from: (i32, i32), to: (i32, i32) },
    BombPlaced { player: u8, pos: (i32, i32) },
    BombExploded { pos: (i32, i32), range: u32 },
    PlayerKilled { victim: u8, killer: Option<u8> },
    PowerUpCollected { player: u8, kind: PowerUpKind },
    WallDestroyed { pos: (i32, i32) },
    RoundEnd { survivors: Vec<u8> },
}
```

### Systems (from bomby patterns, tick-based)

```rust
// ── Schedule (same ordering as bomby) ──
// PreUpdate:  bomb_fuse_system (tick down, explode if 0)
//             blast_propagation_system (chain explosions, damage)
// Update:     player_action_system (BomberPlayer selects action)
//             movement_system (grid movement, wall collision — from bomby player_collisions)
//             bomb_place_system (check max bombs, no double-place — from bomby spawn_bombs)
//             powerup_system (check collection on move)
// PostUpdate: cleanup_system (despawn blasts, check round end)
```

### App Setup

```rust
fn bomber_app() -> App {
    let mut app = App::new();
    app.insert_resource(ArenaGrid::generate(seed, 13, 13))
        .insert_resource(GameRng(SmallRng::seed_from_u64(seed)))
        .insert_resource(TickCounter(0))
        .insert_resource(ScoreBoard::default())
        .add_event::<GameEvent>()
        .add_systems(PreUpdate, (
            bomb_fuse_system,
            blast_propagation_system,
        ).chain())
        .add_systems(Update, (
            player_action_system,
            movement_system,
            bomb_place_system,
            powerup_system,
        ).chain())
        .add_systems(PostUpdate, cleanup_system);
    app
}
```

---

## TUI Rendering (ratatui + emoji)

Same pattern as `dungeon_01_tui.rs` — crossterm backend, ratatui layout.

### Grid Emoji Map

| Cell | Emoji | Description |
|------|-------|-------------|
| Floor | `··` | Open space |
| Fixed wall | `🧱` | Indestructible |
| Destructible wall | `📦` | Breakable |
| Player 1 | `🐰` | Random (baseline) |
| Player 2 | `🐱` | Model (LoRA) |
| Player 3 | `🐶` | Validator (LoRA + WASM) |
| Player 4 | `🐵` | Full HL (smartest) |
| Dead player | `💀` | Died this round |
| Bomb (fuse 3+) | `💣` | Fresh bomb |
| Bomb (fuse 1-2) | `🧨` | About to explode |
| Blast | `💥` | Active explosion |
| BombUp | `💥` | Power-up |
| FireUp | `🔥` | Power-up |
| SpeedUp | `👟` | Power-up |

### TUI Layout

```text
┌─── Bomberman HL Arena ──────────────────────────────────┐
│ ┌─── Arena (13×13) ───┐  ┌─── Scoreboard ─────────────┐ │
│ │ 🧱··#··📦#📦··📦··🐰│  │ P1 🐰 Random:     0 pts  │ │
│ │ ·📦·#·📦·#··📦·#·📦·│  │ P2 🐱 Model:      0 pts  │ │
│ │ ····#····#····#····  │  │ P3 🐶 Validator:  0 pts  │ │
│ │ ## ## ## ## ## ## ## │  │ P4 🐵 Full HL:    0 pts  │ │
│ │ ·📦·#·📦·#··📦·#·📦·│  │                            │ │
│ │ ····#····#····#····  │  │ Round: 42/1000            │ │
│ │ ## ## ## ## ## ## ## │  │ Tick:  87/200             │ │
│ │ ·📦·#··📦·#·📦·#·📦·│  │                            │ │
│ │ ····#····#····#····  │  │ P4 Bandit Q-values:       │ │ │
│ │ ## ## ## ## ## ## ## │  │ ↑ 0.45  ↓ 0.38           │ │
│ │ ·📦·#·📦·#··📦·#·📦·│  │ ← 0.41  → 0.39           │ │
│ │ ····#····#····#····  │  │ 💣 0.62  ⏸ 0.05 (blocked)│ │
│ │ 🧱··#····#····#··🐵 │  │                            │ │
│ └─────────────────────┘  │ TrialLog: /tmp/hl/42.jsonl │ │
│                           └────────────────────────────┘ │
│ [←/→] Step  [Space] Auto  [Q] Quit  [R] New Round       │
└──────────────────────────────────────────────────────────┘
```

### Controls (same as dungeon_01_tui.rs)

| Key | Action |
|-----|--------|
| `←` / Backspace | Previous tick |
| `→` / Enter | Next tick |
| Space | Toggle auto-play |
| Home/End | Jump to start/end of replay |
| Q / Esc | Quit |
| R | New round (re-generate arena) |

---

## Tasks

- [ ] **Task 1: ECS Components & Resources** (`src/pruners/bomber/mod.rs`)
  - Define all bevy_ecs components: `Player`, `GridPos`, `Bomb`, `BombFuse`, `BombRange`, `BombCount`, `Speed`, `Alive`, `DestructibleWall`, `PowerUp`, `Blast`
  - Define resources: `ArenaGrid`, `GameRng`, `TickCounter`, `ScoreBoard`, `PlayerEntities`
  - Define events: `GameEvent` enum
  - Define `Cell` enum for grid: `Floor`, `FixedWall`, `DestructibleWall`, `PowerUpHidden(PowerUpKind)`
  - Define `PowerUpKind` enum: `BombUp`, `FireUp`, `SpeedUp`
  - Define `BomberAction` enum: `Up`, `Down`, `Left`, `Right`, `Bomb`, `Wait` (6 arms)
  - ~150 lines

- [ ] **Task 2: Arena Generation** (`src/pruners/bomber/arena.rs`)
  - `ArenaGrid::generate(seed, width, height) -> Self` — procedural 13×13 grid
  - Fixed walls at odd row/col intersections (standard Bomberman pattern)
  - Destructible walls: seeded random ~40% fill, exclude 3×3 corners (spawn safety)
  - Corner spawns for 4 players
  - Adapted from bomby's LDtk layout but without LDtk dependency
  - Tests: grid dimensions correct, corners clear, fixed walls at correct positions, seed reproducibility
  - ~120 lines

- [ ] **Task 3: ECS Systems — Core Logic** (`src/pruners/bomber/systems.rs`)
  - `spawn_players_system` — create 4 player entities at corner spawns (adapted from bomby's `spawn_players`)
  - `bomb_fuse_system` — tick down `BombFuse`, mark for explosion at 0 (adapted from bomby's `update_bombs`)
  - `blast_propagation_system` — propagate blast in 4 directions, stop at walls, chain-explode other bombs, kill players, destroy walls, spawn powerups (adapted from bomby's blast logic in `update_bombs`)
  - `movement_system` — grid-based movement with wall/bomb collision (adapted from bomby's `player_collisions` + `update_position`)
  - `bomb_place_system` — check max count, no double-place (adapted from bomby's `spawn_bombs`)
  - `powerup_system` — detect player on powerup cell, apply effect
  - `cleanup_system` — despawn blasts, check round end condition
  - All systems tick-based (no `Res<Time>`, just increment `TickCounter`)
  - Tests: bomb explodes after 4 ticks, blast stops at walls, chain explosion, wall destruction, player death, powerup collection
  - ~400 lines

- [ ] **Task 4: BomberPlayer Trait & Implementations** (`src/pruners/bomber/players.rs`)
  - Trait `BomberPlayer { fn select_action(&mut self, grid: &ArenaGrid, pos: GridPos, events: &[GameEvent]) -> BomberAction; fn name(&self) -> &str; fn reset(&mut self); }`
  - `RandomPlayer` — uniform random (P1), fastrand-based
  - `ModelPlayer` — LoRA marginals sampling (P2), uses `TransformerWeights` with LoRA adapter
  - `ValidatedPlayer` — LoRA + WASM ScreeningPruner (P3), `is_valid` + `relevance` filtering
  - `HLPlayer` — LoRA + `BanditPruner<HotSwapPruner<WasmPruner>>` + TrialLog + AbsorbCompress (P4)
  - `player_action_system` — queries each player entity, calls corresponding `BomberPlayer::select_action`
  - Players stored as `Box<dyn BomberPlayer>` in a Resource to allow different types
  - Tests: random player produces valid actions, validated player rejects unsafe moves
  - ~300 lines

- [ ] **Task 5: Bomberman Validator** (`riir-validator-sdk/examples/bomber_validator.rs`)
  - Implement `Validator` trait for Bomberman safety rules
  - `is_valid()`: reject walking into walls, reject walking into active blast zones, reject bomb with no escape route
  - `relevance()`: score actions by safety (dodge blast high, collect powerup medium, bomb near enemy medium, random low)
  - `validate_string()`: validate action sequence string (e.g., "UUURRBB" = up up up right right bomb bomb)
  - Escape route check: BFS from bomb position, must have path to safe cell within blast range + 1
  - Compile to `bomber_validator.wasm`
  - ~300 lines

- [ ] **Task 6: Arena Tournament Runner** (`examples/bomber_01_arena.rs`)
  - Build bevy_ecs `App` with all systems
  - `BomberPlayer` implementations for each slot
  - Run N rounds: reset arena → spawn players → run tick loop → score
  - Print per-round results and cumulative standings
  - Output: `trials.jsonl` with per-round scores for all players
  - Headless mode (no TUI), configurable: rounds, seed, arena size, tick limit
  - ~250 lines

- [ ] **Task 7: TUI Replay** (`examples/bomber_02_tui.rs`)
  - Animated TUI replay of tournament rounds (same pattern as `dungeon_01_tui.rs`)
  - `ratatui` + `crossterm` backend
  - Two-panel layout: arena grid (emoji) + scoreboard/info panel
  - Show all 4 players moving simultaneously per tick
  - Show bomb fuse countdown (💣→🧨), blast animation (💥), powerup collection
  - Scoreboard: cumulative scores, current round, tick, P4 Q-values, trial log path
  - Controls: ←/→ step, space autoplay, home/end jump, q quit, r new round
  - Record tick states for replay (Vec of grid snapshots)
  - ~400 lines

- [ ] **Task 8: HL Experiment — P3 vs P4 Proof** (`examples/bomber_03_hl_proof.rs`)
  - Run 1000-round tournament
  - P1 (random) + P2 (model) + P3 (model+validator) + P4 (full HL)
  - After every 100 rounds: P4 runs absorb-compress cycle
  - Print comparison table at end: survival rate, kill count, avg score, powerup efficiency
  - Golden trace extraction: save top 10 P4 episodes as regression suite
  - Expected result: P4 > P3 > P2 > P1 in survival rate
  - ~200 lines

- [ ] **Task 9: Benchmark — Arena Performance** (`tests/bench_bomber_arena.rs`)
  - Benchmark `App::update()` (single tick, 4 players + bombs)
  - Benchmark full game (200 ticks, 4 players)
  - Benchmark per-player `select_action()` time (P1-P4)
  - Benchmark arena generation
  - Target: tick <50µs, full game <10ms, P4 decision <200µs, arena gen <100µs
  - ~100 lines

- [ ] **Task 10: Update docs & module index**
  - Update `src/pruners/mod.rs` with `pub mod bomber` and re-exports
  - Update `Cargo.toml` with `bevy_ecs` dependency and `[[example]]` entries
  - Update `microgpt-rs/README.md` with Bomberman HL Arena section
  - Update `.docs/09_heuristic_learning.md` with arena results
  - Add bomber_validator to riir-validator-sdk examples table
  - Add feature flag `bomber` for bevy_ecs dependency

---

## Cargo.toml Changes

```toml
[dependencies]
# ... existing ...
bevy_ecs = { version = "0.15", optional = true }  # Standalone ECS (Plan 033)

[features]
# ... existing ...
bomber = ["bevy_ecs", "bandit"]  # Bomberman HL Arena (Plan 033)

[[example]]
name = "bomber_01_arena"
required-features = ["bomber"]

[[example]]
name = "bomber_02_tui"
required-features = ["bomber"]

[[example]]
name = "bomber_03_hl_proof"
required-features = ["bomber"]
```

---

## Module Structure

```text
src/pruners/bomber/
├── mod.rs        # Module index, component/resource/event definitions, re-exports
├── arena.rs      # ArenaGrid generation (procedural, no LDtk)
├── systems.rs    # bevy_ecs systems (fuse, blast, move, bomb, powerup, cleanup)
└── players.rs    # BomberPlayer trait + RandomPlayer, ModelPlayer, ValidatedPlayer, HLPlayer
```

---

## File Locations

| File | Lines | Status |
|------|-------|--------|
| `src/pruners/bomber/mod.rs` | ~150 | Pending |
| `src/pruners/bomber/arena.rs` | ~120 | Pending |
| `src/pruners/bomber/systems.rs` | ~400 | Pending |
| `src/pruners/bomber/players.rs` | ~300 | Pending |
| `riir-validator-sdk/examples/bomber_validator.rs` | ~300 | Pending |
| `examples/bomber_01_arena.rs` | ~250 | Pending |
| `examples/bomber_02_tui.rs` | ~400 | Pending |
| `examples/bomber_03_hl_proof.rs` | ~200 | Pending |
| `tests/bench_bomber_arena.rs` | ~100 | Pending |

---

## Expected Results

### Survival Rate (1000 rounds)

| Player | Tech Stack | Expected Survival Rate | Reasoning |
|---|---|---|---|
| P1 🐰 Random | ~10-15% | Symmetric baseline, 4-way FFA |
| P2 🐱 Model | ~20-25% | Model knows bombs>walls, blast>dodge |
| P3 🐶 Validator | ~30-35% | Validator prevents suicides, enforces safety |
| P4 🐵 Full HL | ~40-50% | Adapts to opponents' patterns, compresses rules |

### The Key Proof: P3 vs P4

P3 (🐶) and P4 (🐵) use the same LoRA model and same WASM validator. The **only** difference:

- P3: static relevance scores (hand-coded)
- P4: bandit-adapted relevance (learned from experience)

If P4 beats P3 by a significant margin, it proves the HL thesis: **the bandit's ability to adapt relevance based on observed outcomes makes the validator more valuable than static expert rules alone.**

### Compression Evidence

After 1000 rounds, P4's absorb-compress should show:
- "Bomb in corner" arm promoted to hard block (Q < 0.05, usually suicide)
- "Wait when no threat" arm promoted to hard block (opportunity cost too high)
- "Move toward powerup when safe" arm boosted (Q > 0.8)

---

## Out of Scope

- [ ] Real-time multiplayer (human keyboard input — future: add bevy_input)
- [ ] Network play
- [ ] Complex bomb types (remote, landmine, piercing)
- [ ] Custom maps (fixed arena for reproducibility)
- [ ] Coding agent writing validators (future: after HL infrastructure proven)
- [ ] Full Bevy renderer (staying with ratatui TUI)
- [ ] bevy_audio (staying silent)

---

## References

- `raw/bomby/` — Fish Folk: Bomby (reference implementation, Apache-2.0 / MIT)
- Plan 032: HL Infrastructure (TrialLog, AbsorbCompress, HotSwapPruner, RegressionSuite)
- Plan 030: Multi-Armed Bandit
- Plan 021: ScreeningPruner
- Plan 025: Model vs Modelless Bandit (precedent for P1 vs P2 vs P3)
- Research 14: "Learning Beyond Gradients"
- [bevy_ecs standalone docs](https://docs.rs/bevy_ecs)
- Classic Bomberman (Hudson Soft, 1983)