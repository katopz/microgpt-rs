# 013: Tactical AI TUI (Plans 017 + 018)

## What Happened

Implemented the full hierarchical tactical AI visualization pipeline:
1. **Plan 017 Phase 1-4**: Core infrastructure (A*, StrategicPruner, hierarchical solver, 16×16 TUI)
2. **Plan 018**: Animated movement with terrain cost model (merged into 017 scope)

The user noticed that Phase 4 (TUI for 16×16 dungeon) was marked as done but never actually implemented — only the small `blue_bear_tui.rs` existed. The 16×16 dungeon TUI with strategic plan sidebar, A* path overlay, and state machine visualization was the missing piece.

## Where is the Plan/Code/Test

### Plans
- `.plans/017_hierarchical_tactical_ai.md` — Phases 1-4 complete, Phase 5 (benchmarks) remaining
- `.plans/018_animated_tui_movement_cost.md` — Complete (merged into 017)

### Code
| File | Purpose |
|------|---------|
| `src/pruners/tactical_pruner.rs` | Game rules engine + `terrain_cost()`, `GameState.total_cost` |
| `src/pruners/pathfinder.rs` | A* with terrain-weighted g-score, `Target` enum, `enumerate_targets` |
| `examples/tactical_ai.rs` | Non-TUI 16×16 dungeon solver (125 steps, ~68ms) |
| `examples/tactical_ai_tui.rs` | **NEW** — 16×16 dungeon TUI with strategic plan sidebar |
| `examples/blue_bear_tui.rs` | Small 3×4 map animated TUI with terrain costs |
| `examples/blue_bear.rs` | Small map non-TUI solver |

### Tests
- 260 tests pass (180 pathfinder + 80 lib)
- `cargo run --example tactical_ai` — verifies 16×16 solution
- `cargo run --example tactical_ai_tui` — interactive TUI (verified via timeout)

## Reflection — Struggling/Solved

### Struggling
1. **Plan 017 was prematurely marked complete** — Phase 4 (TUI) was never implemented despite being checked off. The user caught this and asked for reconciliation.
2. **Debug logging invisible in raw mode** — `eprintln` and `std::fs::File::create("/tmp/...")` both failed silently when running under `enable_raw_mode()`. Never found the root cause (possibly macOS sandbox). Wasted time trying to debug the TUI solver via file logging.
3. **Gap-based animation vs compact map** — The 018 gap system (2-char spaces between cells) works for the 3×4 map but would make the 16×16 map 64+ chars wide. Solved by using hop-based animation for the big map (bear snaps between cells with delay).

### Solved
1. **Reconciled 017 and 018** — Updated 017 to note 018's contributions (terrain cost, animation primitives). Marked Phase 5 cost model as done by 018.
2. **Segment tracking** — Added `TargetSegment` struct that tracks per-target A* paths, positions, and attack actions. This enables the TUI to show which strategic target is current and overlay the remaining path.
3. **Layout for 16×16** — Map (49 chars) + sidebar (28 chars) = 77 chars, fits in 80-col terminal. Vertical split in sidebar: strategy panel (targets) + phase panel (state).
4. **Path overlay** — `remaining_path()` returns future cells for current segment, rendered with cyan style. Target cell rendered with yellow bold style.

## Remain Work

### Plan 017 Phase 5 (still open)
- [ ] Benchmark: strategic solve vs brute-force BFS
- [ ] Verify 16×16 map solvability with different layouts
- [ ] Add terrain to 16×16 map (currently all grass, cost model unused on big map)

### Future improvements
- Add terrain variety to the 16×16 dungeon (sand, water) to showcase cost-aware pathfinding
- Re-evaluation loop (re-plan after each target) for dynamic scenarios
- Breadcrumb trail showing cells already visited (not just future path)
- Scroll/zoom for larger maps that don't fit terminal

## Issues Ref
- None

## How to Dev/Test

```bash
# Build all
cargo build --examples --quiet

# Run 16×16 dungeon TUI
cargo run --example tactical_ai_tui

# Run small map animated TUI
cargo run --example blue_bear_tui

# Run non-TUI solver (125 steps, ~68ms)
cargo run --example tactical_ai

# Tests
cargo test --workspace --quiet

# Lint
cargo clippy --examples --quiet
```

### TUI Controls (both TUIs)
| Key | Action |
|-----|--------|
| `→` / `n` / Enter | Next step (animated) |
| `←` / `p` / Backspace | Previous step (instant) |
| `.` | Skip to next (instant) |
| `Space` | Toggle auto-play |
| `Home` / `End` | Jump to start/end |
| `R` | Restart solver |
| `Q` / Esc | Quit |