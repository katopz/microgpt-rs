# Plan 019: Terrain Costs, Multi-Layer Dungeons, Procedural Generation & Parallel Solving

**Goal**: Extend the tactical AI system to handle real-world game scenarios — terrain variety, multi-floor dungeons, procedural content, and parallel batch verification.

**Why**: The current system proves hierarchical DDTree + A* works on hand-crafted single-floor maps. Real games have terrain costs (sand/water), multiple floors connected by stairs, and need procedural validation at scale.

## Tasks

- [x] Phase 1: Terrain Cost Example
- [x] Phase 2: Multi-Layer Dungeon Architecture
- [x] Phase 3: Procedural Map Generator
- [x] Phase 4: Parallel Batch Solving
- [x] Phase 5: Benchmark & Verify
- [x] Update `.handovers`
- [x] Commit and clean up

---

## Phase 1: Terrain Cost Example

**Status**: Infrastructure exists (`terrain_cost()`, `~` sand, `w` water), but no example exercises it.

### Tasks

- [x] Create `examples/tactical_terrain.rs`
- [x] Define terrain maps:
  - [x] **Desert crossing**: sand (`~`) shortcut vs grass (`.`) detour
  - [x] **River crossing**: water (`w`) expensive but direct, bridge (`.`) circuitous
  - [x] **Mixed terrain**: sand + water + grass maze (8×8)
- [x] Solve each with strategic DDTree + A*
- [x] Print cost comparison: step count vs total_cost (terrain-weighted)
- [x] Show A* prefers cheaper terrain routes when available
- [x] Verify all maps solvable, assertions pass
- [x] Add `[[example]]` to Cargo.toml

### Key Insight

Current bench measures steps. Terrain maps show **cost-optimal** paths matter — A* with weighted g-score avoids water (cost 3) even if it means more steps.

---

## Phase 2: Multi-Layer Dungeon Architecture

**Status**: New architecture required. Current system is single-floor only.

### Design

```
DungeonMap
├── floors: Vec<FloorGrid>          // Each floor is a 2D grid
├── stairs: Vec<StairConnection>    // Links between floors
├── start: (floor, r, c)
├── goal: (floor, r, c)
├── monsters: Vec<(floor, r, c)>
└── treasures: Vec<(floor, r, c)>

StairConnection
├── from: (floor_a, r, c)          // Stairs down position
└── to: (floor_b, r, c)            // Stairs up destination

DungeonState (extends GameState)
├── floor: usize                    // Current floor index
├── r, c, inventory, killed, collected, total_cost
```

### Actions

| Action | Code | Description |
|--------|------|-------------|
| Up | 0 | Move up on current floor |
| Down | 1 | Move down on current floor |
| Left | 2 | Move left on current floor |
| Right | 3 | Move right on current floor |
| Attack | 4 | Attack monster on current tile |
| Use Stairs | 5 | Enter stairs (floor transition) |

### Cross-Floor A*

A* needs to work across floors. Approach:

1. **Floor-local path**: Standard A* within a floor (blocked = walls + live monsters + locked goal)
2. **Cross-floor path**: Chain floor-local paths through stair connections
3. **Multi-floor A***: Build a graph where stairs are portal edges, run A* on this abstract graph
4. Simplest: find path to nearest stairs → transit → find path on next floor → repeat

Implementation:

```rust
fn find_path_multifloor(
    dungeon: &DungeonMap,
    from: (usize, usize, usize),  // (floor, r, c)
    to: (usize, usize, usize),
    blocked: &MultiFloorBlocked,
) -> Option<Vec<DungeonAction>>
```

### Multi-Floor TUI Layout

Based on `blue_bear_tui.rs` layout, restructured for multi-floor:

```
┌───────────────────────────────────────────────────────────────────────────────┐
│ 🏰 Dungeon  Floor 2/3 · 45 steps · Cost 67 · 120ms          ← → · Space    │
└───────────────────────────────────────────────────────────────────────────────┘
┌ 🏢 Floors ─┐┌ 🗺 Floor 2 ──────────────────┐┌ 📊 State ─────────────────────┐
│ 3F 🚪      ││ 🧱 🧱 🧱 🧱 🧱 🧱 🧱 🧱 🧱   ││  Position:  F2 (3, 5)         │
│  ↑🪜(1,2)  ││ 🧱 ⬜ ⬜ ⬜ ⬜ ⬜ ⬜ ⬜ 🧱   ││  Floor:      2/3               │
│ 2F 🐻 ◀   ││ 🧱 ⬜ 👹 ⬜ ⬜ ⬜ ⬜ ⬜ 🧱   ││  Cost:       67                │
│  ↓🪜(5,7)  ││ 🧱 ⬜ ⬜ ⬜ ⬜ ⬜ ⬜ ⬜ 🧱   ││  Inventory:  🔑                │
│ 1F ⬜      ││ 🧱 ⬜ ⬜ 💎 ⬜ ⬜ ⬜ ⬜ 🧱   ││  Monsters:   1/3 killed        │
│            ││ 🧱 ⬜ ⬜ ⬜ ⬜ ⬜ ⬜ ⬜ 🧱   ││  Treasures:  1/3 collected     │
│ Stairs:    ││ 🧱 ⬜ ⬜ ⬜ ⬜ 🪜⬇ ⬜ 🧱   ││  Stairs:     ↑(1,2) ↓(5,7)     │
│ ↑ to F3    ││ 🧱 🧱 🧱 🧱 🧱 🧱 🧱 🧱 🧱   │└────────────────────────────────┘
│ ↓ to F1    ││                                 │
└────────────┘│                                 │┌ Legend ───────────────────────┐
              └─────────────────────────────────┘│ 🐻 You     👹 Monster         │
                                               │ 💎 Treasure 🚪 Exit            │
                                               │ 🧱 Wall    ⬜ Floor            │
                                               │ 🪜⬇ Down   🪜⬆ Up             │
                                               └──────────────────────────────┘
┌───────────────────────────────────────────────────────────────────────────────┐
│  1. → Right  (+1)  F1    2. ⬇ Stairs F1→F2   3. ↑ Up (+1) F2               │
│  4. ⚔ Attack  F2    ...  45. → Right F3 (goal)                              │
└───────────────────────────────────────────────────────────────────────────────┘
```

#### TUI Design Decisions

| Element | Design | Rationale |
|---------|--------|-----------|
| **Floors sidebar** | Leftmost narrow panel: floor list with `◀` on current, stair positions labeled | Quick overview of dungeon structure without switching views |
| **Map** | Shows current floor only, stairs rendered as `🪜⬇`/`🪜⬆` | Single floor map stays readable at 16×16 |
| **Floor transitions** | Solution list shows `⬇ Stairs F1→F2` as an action step | Clear when bear changes floors in replay |
| **Legend** | Moved to right panel below State | Frees left side for floors sidebar |
| **Solution nav** | Bottom bar, current floor indicated per step | Know which floor each step belongs to |
| **Ghost overlay** | `PageUp`/`PageDown` to view other floors (dimmed, bear not shown) | Debug cross-floor paths without losing current position |
| **Floor peek** | `[/]` keys to cycle visible floor in map panel | Inspect other floors while paused |

#### Stairs on Map

- Stairs down tile: `🪜⬇` (bear can use action 5 here to go down)
- Stairs up tile: `🪜⬆` (bear can use action 5 here to go up)
- When bear is on stairs tile, bears renders as `🐻` (stairs icon hidden underneath)

#### Floor Sidebar Details

```
┌ 🏢 Floors ──┐
│ 3F 🚪       │  ← Goal on F3 (emoji shows what's there)
│  ↑🪜(1,2)   │  ← Stairs UP at (1,2) on F2, leads to F3
│ 2F 🐻 ◀    │  ← Current floor (bear here, ◀ indicator)
│  ↓🪜(5,7)   │  ← Stairs DOWN at (5,7) on F2, leads to F1
│ 1F ⬜       │  ← Start floor (cleared/empty indicator)
│             │
│ Stairs:     │  ← Summary section
│ ↑ leads F3  │
│ ↓ leads F1  │
└─────────────┘
```

Each floor row shows:
- Floor number (1F, 2F, 3F)
- Key emoji if notable item exists (🚪 goal, 💎 uncollected treasure, 👹 live monster)
- `◀` on current floor
- Stairs rows show position and direction between floor rows

### Tasks

- [x] Create `src/pruners/dungeon_pruner.rs`:
  - [x] `DungeonMap` struct (floors, stairs, targets)
  - [x] `DungeonState` struct (adds `floor` field)
  - [x] `DungeonPruner` (multi-floor movement, combat, stairs)
  - [x] `DungeonPruner::apply_action` (actions 0-5)
  - [x] `DungeonPruner::terrain_cost` (per-floor terrain)
- [x] Create `src/pruners/dungeon_pathfinder.rs`:
  - [x] `find_path_multifloor` (cross-floor A*)
  - [x] `find_path_on_floor` (delegates to existing `find_path`)
  - [x] `StairConnection` struct
- [x] Create `examples/dungeon_multifloor.rs`:
  - [x] 2-floor dungeon: B1 (monsters + treasures) → B2 (more monsters + goal)
  - [x] 3-floor dungeon: F1 (start) → F2 (monsters) → F3 (goal)
  - [x] Strategic DDtree with multi-floor targets
  - [x] Print floor-by-floor solution
  - [x] Verify all assertions
- [x] Create `examples/dungeon_tui.rs`:
  - [x] Floors sidebar (leftmost panel, floor list with ◀ indicator)
  - [x] Current floor map (stairs rendered as 🪜⬇/🪜⬆)
  - [x] State panel (add floor number, stairs positions)
  - [x] Legend panel (add stairs icons)
  - [x] Solution nav (floor label per step, stairs transitions)
  - [x] Ghost floor view (`PageUp`/`PageDown` to peek other floors)
  - [x] Reuse animation system from `blue_bear_tui.rs`
  - [x] Floor transition animation (instant step transitions with auto-play)
- [x] Add `mod dungeon_pruner` and `mod dungeon_pathfinder` to `src/pruners/mod.rs`
- [x] Add `[[example]]` entries to Cargo.toml
- [x] Add tests to `tests/` folder (tests embedded in modules)

### Key Insight

Multi-floor extends the hierarchical approach naturally:
- **DDTree**: still operates on target tokens (now tagged by floor)
- **A***: chains floor-local paths through stair portals
- **Scaling**: number of floors doesn't affect DDtree complexity — only target count matters

---

## Phase 3: Procedural Map Generator

**Status**: New code. Generates random solvable maps for batch testing.

### Design

```rust
struct MapGenerator {
    rng: fastrand::Rng,  // or use rand
    width: usize,
    height: usize,
    num_monsters: usize,
    num_treasures: usize,
    wall_density: f32,     // 0.0 = open, 0.3 = maze-like
    terrain_mix: bool,     // add sand/water tiles?
}

impl MapGenerator {
    fn generate(&mut self) -> Option<GeneratedMap>
    fn is_solvable(&self, map: &GeneratedMap) -> bool
}
```

### Generation Strategy

1. Start with open grid (all floor tiles)
2. Place walls using random walk (ensures connectivity)
3. Place start, goal at opposite corners
4. Place monsters/treasures on reachable floor tiles
5. Verify solvability with strategic solver
6. If unsolvable, regenerate (max N attempts)
7. Optionally add terrain (sand/water patches)

### Tasks

- [x] Add `fastrand` dependency to Cargo.toml (lightweight RNG)
- [x] Create `src/pruners/map_generator.rs`:
  - [x] `MapGenerator` struct with configuration
  - [x] `generate_single_floor` (random solvable map)
  - [x] `generate_multi_floor` (random multi-layer dungeon)
  - [x] `ensure_connectivity` (BFS check: all targets reachable from start)
  - [x] `add_terrain_patches` (sand/water clusters)
- [x] Add `mod map_generator` to `src/pruners/mod.rs`
- [x] Create `examples/tactical_procedural.rs`:
  - [x] Generate 10 random single-floor maps
  - [x] Generate 5 random multi-floor dungeons
  - [x] Solve each, print solvability rate + stats
  - [x] Show example generated maps (ASCII art)
- [x] Add `[[example]]` to Cargo.toml

### Key Insight

Procedural generation validates the solver at scale. If it can solve 100 random maps, it's robust — not just tuned for hand-crafted examples.

---

## Phase 4: Parallel Batch Solving

**Status**: `rayon` is already a dependency. Just needs an example.

### Design

Use `rayon::par_iter` to solve multiple maps concurrently:

```rust
let results: Vec<SolveResult> = maps
    .par_iter()
    .map(|map| solve_strategic(map))
    .collect();
```

### Tasks

- [x] Create `examples/tactical_parallel.rs`:
  - [x] Generate N maps procedurally (Phase 3)
  - [x] Solve sequentially: measure total time
  - [x] Solve in parallel with rayon: measure total time
  - [x] Print speedup factor
  - [x] Verify all solutions match
- [x] Add `[[example]]` to Cargo.toml

### Key Insight

Embarrassingly parallel — each map solve is independent. Linear speedup expected with core count. On 8-core Mac, expect ~6-7x speedup (overhead from rayon setup).

---

## Phase 5: Benchmark & Verify

### Tasks

- [x] Run all new examples, collect timing data
- [x] Save benchmark results to `.benchmarks/002_terrain_multilayer.md`
- [x] Run `cargo test` — all 234 lib tests + 80 integration tests pass
- [x] Run `cargo clippy --examples` — zero warnings
- [x] Verify multi-floor maps solve correctly (2-floor 25 steps, 3-floor 27 steps)
- [x] Verify procedural maps have high solvability rate (100% for seeds 1-10)
- [x] Verify parallel results match sequential results (15/15 match)
- [x] Update plan with results table

### Expected Results

| Example | Maps | Avg Time | Notes |
|---------|------|----------|-------|
| `tactical_terrain` | 3 | ~50-200ms | Cost-optimal paths through sand/water |
| `dungeon_multifloor` | 2 | ~100-300ms | 2-3 floor dungeons, cross-floor A* |
| `dungeon_tui` | 1 | ~100-200ms | Multi-floor TUI with floors sidebar + stair navigation |
| `tactical_procedural` | 15 | ~100-500ms each | Random maps, ~80% solvable |
| `tactical_parallel` | 15 | ~100ms total | Rayon batch, ~6x speedup vs sequential |

---

## File Structure (New)

```
src/pruners/
├── mod.rs              (add dungeon_pruner, dungeon_pathfinder, map_generator)
├── tactical_pruner.rs  (existing, unchanged)
├── pathfinder.rs       (existing, unchanged)
├── dungeon_pruner.rs   (NEW — multi-floor rules engine)
├── dungeon_pathfinder.rs (NEW — cross-floor A*)
└── map_generator.rs    (NEW — procedural generation)

examples/
├── tactical_terrain.rs     (NEW — terrain cost demo)
├── dungeon_multifloor.rs   (NEW — multi-layer dungeon)
├── dungeon_tui.rs          (NEW — multi-floor dungeon TUI with floors sidebar)
├── tactical_procedural.rs  (NEW — procedural generation + solve)
├── tactical_parallel.rs    (NEW — rayon batch solving)
├── tactical_bench.rs       (existing, unchanged)
├── tactical_ai.rs          (existing, unchanged)
└── tactical_ai_tui.rs      (existing, unchanged)

.benchmarks/
├── 001_tactical_bench.md   (existing)
└── 002_terrain_multilayer.md (NEW)
```

## Dependency Changes

```toml
[dependencies]
fastrand = "2"  # Lightweight RNG for procedural generation
# rayon already exists
```

## Design Principles

1. **Extend, don't modify** — new types in new files, existing code unchanged
2. **Single-floor still works** — `TacticalPruner` untouched, multi-floor is opt-in
3. **Separation of concerns** — pathfinder doesn't know about DDtree, pruner doesn't know about A*
4. **Real-world focus** — every example demonstrates a practical game scenario
5. **Small examples matter** — 2×3 map shows the concept, 16×16 shows it scales, procedural shows it's robust

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Multi-floor A* too complex | Start with chained single-floor A*, optimize later |
| Procedural maps unsolvable | Regenerate with max retries, lower wall density |
| Rayon overhead > benefit for small batches | Batch size ≥ 10 maps, measure and report |
| Cross-floor path invalidation | Re-validate after each floor transition |
| Stairs blocked by monster | Clear stairs tile in generator, unblock in pruner |

## Relationship to Previous Plans

| Plan | Contribution |
|------|-------------|
| 016 (Blue Bear TUI) | Animation primitives, terrain emoji |
| 017 (Hierarchical AI) | Strategic DDtree + A* architecture |
| 018 (Animated TUI) | Cost model, weighted A* g-score |
| 019 (This plan) | Terrain examples, multi-layer, procedural, parallel |