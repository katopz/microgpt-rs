# 018: Animated TUI with Movement Costs

## Overview
Enhance `blue_bear_tui` with two major features:
1. **Animated movement** — Gap-based interpolation so the bear visually walks between cells with terrain-cost-based speed
2. **Movement cost model** — Terrain types with different traversal costs (grass=1, sand=2, water=3) displayed in the TUI

## Tasks
- [x] Add terrain cost model to `TacticalPruner`
  - [x] Add `terrain_cost(grid, r, c) -> u32` function (grass=1, sand=2, water=3)
  - [x] Update `GameState` to track `total_cost: u32`
  - [x] Update `apply_action` to accumulate cost on move (attack is free)
  - [x] Add terrain char support to map parsing: `.` = grass(1), `~` = sand(2), `w` = water(3)
- [x] Update A* `pathfinder.rs` to use terrain costs instead of uniform cost=1
  - [x] Add `terrain_cost()` function
  - [x] `g = current.g + terrain_cost(grid, nr, nc)` instead of `g + 1`
  - [x] Update `find_path` and `find_distance`
  - [x] Add test `test_terrain_cost_affects_path`
- [x] Redesign `blue_bear_tui.rs` map rendering
  - [x] Gap-based cell spacing: each cell has a 2-char gap for bear animation
  - [x] Bear position interpolated as `f32` for smooth movement between cells
  - [x] Animation tick via `event::poll()` with timeout (no blocking)
  - [x] Cost overlay: digit shown on tiles with cost > 1
  - [x] Terrain emoji: ⬜=grass, 🟨=sand, 🟦=water, 🧱=wall
- [x] Add animation state machine to `App`
  - [x] `AnimState { from, to, action, start, duration_ms }`
  - [x] Animation speed: 150ms per cost unit (grass fast, water slow), 200ms for attack
  - [x] On animation complete: advance `current` step index
- [x] Update map to showcase terrain variety
  - [x] New 3×4 map: `B . ~ T / . M # . / . . w G`
  - [x] Solvable within DDTree lookahead=8
  - [x] Display total cost in state panel
- [x] Update state panel to show cost
  - [x] Add "Cost:" field showing accumulated `total_cost`
  - [x] Add terrain legend with cost per type
  - [x] Cost delta (+N) shown in solution step list
  - [x] Terrain info in navigation bar
- [x] Update key bindings
  - [x] `→` / `n` / Enter — start animation to next step
  - [x] `←` / `p` / Backspace — instant jump back (no reverse animation)
  - [x] `Space` — toggle auto-play (animate all steps sequentially)
  - [x] `.` — skip animation (instant jump to next)
- [x] Fix all clippy warnings
- [x] All 260 tests pass (180 + 80)

## Architecture

### Gap-Based Tile Rendering
```
Each cell = "{emoji}{cost}" followed by 2-char gap
Cost indicator: " " for cost=1, "2" for sand, "3" for water

Example row:  🐻  ⬜2 🟨  👹  ⬜  🚪
              ^cost hidden  ^cost shown

During animation, bear slides through the gap:
Step N:   🐻  ⬜2 👹  ...
Anim:     ⬜  🐻 👹  ...     ← bear in gap, sliding right
Step N+1: ⬜2  🐻  ...     ← bear arrived at next cell
```

### Animation State Machine
```
┌──────┐  →/n     ┌──────────┐  complete  ┌──────┐
│ Idle │────────→│ Animating│──────────→│ Idle │
│      │←────────│          │           │ step+1│
└──────┘  . skip  └──────────┘           └──────┘
              ↑                                  │
              └────── Space toggles auto-play ───┘
```

### Cost Model
```
Terrain  Char  Cost  Emoji    Animation Speed
Grass    .     1     ⬜       150ms (fast)
Sand     ~     2     🟨       300ms (medium)
Water    w     3     🟦       450ms (slow)
Wall     #     ∞     🧱       impassable
```

### New Map (3×4 with terrain)
```
B . ~ T    Bear(0,0), Sand(0,2) cost 2, Treasure(0,3)
. M # .    Monster(1,1), Wall(1,2)
. . w G    Water(2,2) cost 3, Goal(2,3)
```
Solution: 8 steps, total cost = 8 (varies by terrain path chosen)

## Files Modified
- `src/pruners/tactical_pruner.rs` — `total_cost` in GameState, `terrain_cost()`, cost accumulation in `apply_action`
- `src/pruners/pathfinder.rs` — `terrain_cost()`, weighted A* g-score, new test
- `examples/blue_bear_tui.rs` — animated rendering, cost display, new map, auto-play, gap-based animation

## Verification
- ✅ `cargo clippy --examples` — zero warnings
- ✅ 260 tests pass (180 + 80)
- ✅ `cargo run --example blue_bear` — non-TUI solver still works
- ✅ `cargo run --example tactical_ai` — hierarchical AI still works
- ✅ TUI launches and renders correctly (verified via timeout test)