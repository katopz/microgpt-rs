# 017: Hierarchical Tactical AI — DDTree + A* + State Machine

## Overview
Design and implement a **two-level AI architecture** for grid-based tactical puzzles
that scales to real game maps (16×16+). The DDTree becomes the **strategic brain**
(choosing *what* targets to visit in what order) while A* becomes the **tactical legs**
(choosing *how* to get there). A state machine handles execution, and a re-evaluation
loop handles dynamic situations.

This is the same pattern used by Into the Breach, Final Fantasy Tactics, and similar
tactical games.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Strategic Layer (DDTree)                      │
│  Tokens = target indices, NOT movement directions               │
│  Pruner validates: inventory, goal-lock, reachability           │
│  Output: ordered list of targets to visit                       │
│  Constraint: 8 targets × 16 bits = 128 bits = fits u128 ✓      │
└──────────────┬──────────────────────────────────────────────────┘
               │ target sequence: [Monster0, Monster1, Treasure0, Goal]
               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Tactical Layer (A* Pathfinder)                │
│  Input: current position, target position, current game state   │
│  Output: list of movement actions (↑↓←→) to reach target       │
│  Considers: walls, live monsters (obstacles), grid bounds       │
└──────────────┬──────────────────────────────────────────────────┘
               │ path: [→, →, ↓, ↓]
               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Execution Layer (State Machine)               │
│  States: Idle → Moving → Arrived → Acting → Done → Idle        │
│  Follows A* rally points step-by-step                           │
│  Performs action at target (Attack / Collect)                   │
│  Reports completion back to strategic layer                     │
└──────────────┬──────────────────────────────────────────────────┘
               │ "Target Monster0 completed, inventory=1"
               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Re-evaluation Loop                            │
│  After each target completion:                                   │
│  1. Update game state                                            │
│  2. Re-compute A* distances to remaining targets                │
│  3. Re-run DDTree with updated state                            │
│  4. Feed new plan to execution layer                            │
│  Handles: blocked paths, new threats, changed priorities        │
└─────────────────────────────────────────────────────────────────┘
```

## Key Insight: Tokens as Targets

The DDTree operates on **target indices**, not movement directions:

```
vocab_size = num_monsters + num_treasures + 1 (goal)

Token 0      = "Go kill Monster 0"     → A* → {→,→,↓,↓} → Attack
Token 1      = "Go kill Monster 1"     → A* → {←,←,↓}    → Attack
Token 2      = "Go collect Treasure 0" → A* → {→,↑,→}
Token 3      = "Go to Goal"            → A* → {↓,↓,→,→}

DDTree lookahead = num_targets ≤ 8
Each token expands to 10-20 step A* path
Total solution: 50-100+ steps, but DDTree only sees 8 strategic decisions
```

## Target System

```rust
enum Target {
    Monster(usize),    // "Go kill monster i"
    Treasure(usize),   // "Go collect treasure j"  
    Goal,              // "Go to exit"
}

// Derived from map at parse time:
// targets[0] = Monster(0) at (2, 3)
// targets[1] = Monster(1) at (7, 5)
// targets[2] = Treasure(0) at (1, 7)
// targets[3] = Treasure(1) at (9, 3)
// targets[4] = Goal at (14, 14)
```

## StrategicPruner (ConstraintPruner for target selection)

Validates strategic-level decisions:

```
is_valid(depth, target_idx, parent_targets):
  target = targets[target_idx]
  
  match target:
    Monster(i):
      - Not already killed ✓
      - Reachable from current position ✓ (A* check)
      
    Treasure(j):
      - Not already collected ✓
      - Has inventory item ✓ (from killing a monster earlier)
      - No live monster on same tile ✓
      - Reachable ✓
      
    Goal:
      - All treasures collected ✓
      - Reachable ✓
  
  // Also check: target not already visited in parent_targets
```

## Cost Model (for marginals / scoring)

Each target gets a cost/weight for the DDTree marginals:

```rust
fn target_cost(from: Pos, target: Target, state: &GameState) -> Cost {
    let distance = astar_distance(from, target.pos, state);
    let action_cost = match target {
        Monster(_) => distance + 1,  // +1 for attack action
        Treasure(_) => distance,     // auto-collect on arrival
        Goal => distance,
    };
    let urgency = match target {
        Monster(_) if inventory == 0 => HIGH,    // need item!
        Treasure(_) if inventory > 0 => HIGH,     // use item before waste
        Goal if all_collected => CRITICAL,
        _ => NORMAL,
    };
    Cost { steps: action_cost, urgency }
}

// Marginals: closer + more urgent targets get higher probability
// DDTree explores best strategic sequences first
```

## A* Pathfinder

```rust
fn find_path(
    grid: &[Vec<char>],
    from: (usize, usize),
    to: (usize, usize),
    blocked: &HashSet<(usize, usize)>,  // live monster positions
) -> Option<Vec<usize>>  // action sequence (0=Up, 1=Down, 2=Left, 3=Right)
```

- Standard A* with Manhattan distance heuristic
- Considers walls (#) and live monster positions as obstacles
- Returns `None` if unreachable
- State-independent: only depends on grid layout and blocking set

## State Machine

```
States:
  Idle       → waiting for command
  Moving     → following A* path, one step per tick
  Arrived    → reached target position
  Attacking  → performing attack action on monster
  Collecting → performing collect action on treasure
  Evaluating → re-running strategic layer

Transitions:
  Idle       → Moving      (new target assigned)
  Moving     → Moving      (next step along path)
  Moving     → Arrived     (reached target pos)
  Arrived    → Attacking   (target is monster)
  Arrived    → Collecting  (target is treasure with item)
  Arrived    → Idle        (target is goal, done!)
  Attacking  → Evaluating  (monster killed, re-plan)
  Collecting → Evaluating  (treasure collected, re-plan)
  Evaluating → Idle        (new plan ready)
```

## Re-evaluation Loop

After completing each target:
1. Update `GameState` (position, inventory, killed, collected, dropped)
2. Re-compute A* distances to remaining targets (some may now be reachable/unreachable)
3. Build new marginals based on updated costs
4. Re-run DDTree with StrategicPruner using updated state
5. Feed next target to state machine

This handles:
- Monster killed → its tile no longer blocks A*
- Item picked up → treasures now unlockable
- Dynamic priorities → closer targets may become better choices

## File Structure

```
src/pruners/
  mod.rs                 # exports
  tactical_pruner.rs     # Game rules engine (unchanged from 016)
  pathfinder.rs          # NEW: A* pathfinding on grid
  
examples/
  blue_bear.rs           # Small map, direct DDTree (unchanged, 016)
  blue_bear_tui.rs       # Small map TUI (unchanged, 016)
  tactical_ai.rs         # NEW: 16×16 map, hierarchical AI demo
  tactical_ai_tui.rs     # NEW: 16×16 map, hierarchical AI TUI
```

## Tasks

### Phase 1: Core Infrastructure
- [x] Add `Hash` derive to `GameState` (needed for A* visited set)
- [x] Create `src/pruners/pathfinder.rs` with A* implementation
- [x] Add `find_path` function: grid A* considering walls and blocked tiles
- [x] Add `find_distance` function: A* distance only (faster, no path reconstruction)
- [x] Add `reachable_positions` function: BFS flood fill for cost evaluation

### Phase 2: Target System & StrategicPruner
- [x] Define `Target` enum in `pathfinder.rs` or new module
- [x] Add `enumerate_targets` function — enumerate all targets from map data
- [x] Create `StrategicPruner` struct wrapping `TacticalPruner`
- [x] Implement `ConstraintPruner` for `StrategicPruner`
  - [x] Validate target not already visited
  - [x] Validate monster not already killed
  - [x] Validate treasure: have item + no live monster on tile
  - [x] Validate goal: all treasures collected
  - [x] Check reachability via A*
- [x] Unit tests for pathfinder (7 tests in `pathfinder.rs::tests`)

### Phase 3: Hierarchical Solver
- [x] Create `examples/tactical_ai.rs`
- [x] Design 17×16 dungeon map with 3 monsters, 3 treasures, walls, corridors
- [x] Implement target enumeration from map
- [x] Build marginals (uniform BFS — pruner does all the work)
- [x] Run DDTree with StrategicPruner → get target sequence
- [x] Expand target sequence into full action sequence via A*
- [x] Print step-by-step with emoji grid (condensed: first 5 + last 3 steps)
- [x] Assert solution correctness

### Phase 4: TUI Visualization
- [ ] Create `examples/tactical_ai_tui.rs`
- [ ] Show strategic plan (target order) in sidebar
- [ ] Highlight current target on map
- [ ] Show A* path overlay (breadcrumbs)
- [ ] Show state machine state (Moving/Attacking/etc.)
- [ ] Step navigation through combined micro-path

### Phase 5: Polish
- [ ] Benchmark: time strategic solve vs. brute-force BFS
- [ ] Verify 16×16 map solvability with different layouts
- [ ] Add cost/stamina model (stretch goal)
- [ ] Update `.plans/016` to reference this plan
- [ ] Update `.handovers`

## 17×16 Dungeon Map (Actual)

```
################
#B.....#.......#
#.####.#.####..#
#....#.#.#..T..#
#.M..#.#.#.###.#
####.#.#.#.....#
#....#...#.....#
#.########.###.#
#.#.......#...G#
#.#.###.###.##.#
#T..#.#.M.#..#.#
###.#.#.#..##..#
#...#.#.#.##...#
#.###.#.#....#.#
#.....#.####.#.#
#....M.....#.T.#
################
```

- Monsters (M): (4,2), (10,8), (15,5) = 3
- Treasures (T): (3,11), (10,2), (15,13) = 3
- Goal (G): (8,14)
- Bear (B): (1,1)
- Strategic tokens: 3M + 3T + 1G = 7 targets
- DDTree lookahead = 7 → fits u128/16 ✓

## Actual Results

- **125 action steps** solved in **~68ms**
- Strategic layer: DDTree explores 7 target tokens → finds valid visit order
- Tactical layer: A* expands each target into 15-30 step paths
- All 3 monsters killed, all 3 treasures collected, bear at goal
- All assertions pass ✅
- 259 tests pass (252 existing + 7 new pathfinder tests)

## Design Decisions

1. **Keep TacticalPruner unchanged** — it's the game rules engine, works at micro level
2. **StrategicPruner wraps TacticalPruner** — reuses game state logic at macro level
3. **A* is stateless function** — takes grid + blockers, returns path. No mutation.
4. **Re-evaluation is lightweight** — only re-solve remaining targets, not full plan
5. **State machine is simple** — just tracks execution phase, no complex logic
6. **Examples are self-contained** — new files, don't break existing 016 examples

## Why This Architecture Works for Real Games

1. **Scalable**: 100×100 map? A* handles it. 20 targets? DDTree handles it.
2. **Dynamic**: Re-evaluation handles changing situations (new enemies, blocked paths)
3. **Extensible**: Add new target types (NPC, door, switch) without changing architecture
4. **Testable**: Each layer tested independently (A*, StrategicPruner, State Machine)
5. **Debuggable**: Can inspect strategic plan, A* paths, and execution separately
6. **Performant**: DDTree only sees 8 strategic tokens, not 100+ movement steps

## Precedent: How This Maps to Existing Sudoku System

| Sudoku | Tactical AI |
|--------|-------------|
| Token = digit (1-9) | Token = target index (0..N) |
| Depth = empty cell position | Depth = visit order position |
| Pruner = row/col/box conflict | Pruner = inventory/goal/reachability |
| Uniform marginals = BFS | Distance-weighted marginals = A*-guided |
| One cell at a time | One target at a time |
| Solution = filled grid | Solution = target visit order + paths |