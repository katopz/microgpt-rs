# Benchmark 002: Terrain Costs, Multi-Layer Dungeons, Procedural Generation & Parallel Solving

**Date**: 2025-07-XX
**Plan**: 019
**Examples**: `tactical_terrain`, `dungeon_multifloor`, `dungeon_tui`, `tactical_procedural`, `tactical_parallel`

## Setup

- **Terrain A***: Weighted g-score by terrain cost (grass=1, sand=2, water=3)
- **Multi-floor**: DDTree on target tokens + cross-floor A* via stair portals
- **Procedural**: `fastrand` seeded RNG, random walk walls, BFS connectivity check
- **Parallel**: `rayon::par_iter` batch solving, 15 procedurally generated maps
- **Hardware**: macOS (dev build, unoptimized)

## Phase 1: Terrain Cost Results

| Map | Size | Terrain | Steps | Cost | Surcharge | Time | Route |
|-----|------|---------|-------|------|-----------|------|-------|
| Desert | 5×7 | sand shortcut | 8 | 9 | +1 | 349µs | sand forced by wall gap |
| River | 5×7 | water river | 9 | 9 | +0 | 355µs | bridge bypass avoids water |
| Mixed | 8×8 | sand+water+walls | 12 | 13 | +1 | 422µs | minimizes expensive terrain |

**Key insight**: A* with terrain-weighted g-score avoids water (cost 3) even if it means more steps. Desert map forces sand traversal (+1 surcharge) due to wall layout.

## Phase 2: Multi-Layer Dungeon Results

| Dungeon | Floors | Monsters | Treasures | Steps | Cost | Time |
|---------|--------|----------|-----------|-------|------|------|
| B1→B2 | 2 | 4 | 2 | 25 | 23 | 1.68s |
| F1→F2→F3 | 3 | 2 | 2 | 27 | 25 | 17.15ms |

**Floor breakdown (Dungeon 1)**:
- Floor 0: 12 actions, cost +11 (kill monster, collect treasure, use stairs)
- Floor 1: 13 actions, cost +12 (kill 2 monsters, collect treasure, reach goal)

**Floor breakdown (Dungeon 2)**:
- Floor 0: 7 actions, cost +7 (navigate to stairs)
- Floor 1: 11 actions, cost +10 (kill monster, collect treasure, use stairs)
- Floor 2: 9 actions, cost +8 (kill monster, collect treasure, reach goal)

**Key insight**: DDTree complexity depends on target count (4-6 targets), NOT floor count. Adding floors doesn't increase branching factor.

## Phase 3: Procedural Generation Results

### Single-Floor Maps (Seeds 1-10, 8×8, 2M/2T, 15% walls)

| Seed | Solvable | Steps | Cost | Time | Nodes |
|------|----------|-------|------|------|-------|
| 1 | ✅ | 21 | 19 | 3.25ms | 30 |
| 2 | ✅ | 27 | 25 | 7.36ms | 32 |
| 3 | ✅ | 33 | 31 | 2.92ms | 17 |
| 4 | ✅ | 33 | 31 | 6.05ms | 28 |
| 5 | ✅ | 23 | 21 | 2.44ms | 13 |
| 6 | ✅ | 25 | 23 | 3.85ms | 16 |
| 7 | ✅ | 28 | 26 | 8.49ms | 32 |
| 8 | ✅ | 35 | 33 | 8.89ms | 32 |
| 9 | ✅ | 28 | 26 | 9.73ms | 32 |
| 10 | ✅ | 27 | 25 | 4.76ms | 32 |

**Solvability rate**: 10/10 (100%)

### Multi-Floor Dungeons (Seeds 101-105, 6×6, 1M/1T per floor)

| Seed | Floors | Monsters | Treasures | Wall% | Status |
|------|--------|----------|-----------|-------|--------|
| 101 | 3 | 3 | 3 | 5.6% | ✅ |
| 102 | 2 | 2 | 2 | 5.6% | ✅ |
| 103 | 3 | 3 | 3 | 5.6% | ✅ |
| 104 | 2 | 2 | 2 | 5.6% | ✅ |
| 105 | 3 | 3 | 3 | 5.6% | ✅ |

**Multi-floor success**: 5/5 (100%)

**Key insight**: 100% solvability at 15% wall density validates the generator. Random walk wall placement + BFS connectivity carve ensures reachable targets.

## Phase 4: Parallel Batch Solving Results

| Metric | Value |
|--------|-------|
| Maps generated | 15 (seeds 1-15, 10×10, 3M/2T) |
| Maps solvable | 15/15 (100%) |
| Sequential total | 1.15s |
| Parallel total | 171.17ms |
| **Speedup** | **6.72x** |

### Per-Map Timing

| Map | Seed | Sequential | Parallel | Steps |
|-----|------|------------|----------|-------|
| 1 | 1 | 67.32ms | 80.99ms | 35 |
| 2 | 2 | 92.99ms | 108.94ms | 44 |
| 3 | 3 | 61.31ms | 83.80ms | 38 |
| 4 | 4 | 65.15ms | 77.26ms | 45 |
| 5 | 5 | 68.45ms | 97.33ms | 40 |
| 6 | 6 | 67.77ms | 75.34ms | 44 |
| 7 | 7 | 43.83ms | 61.99ms | 30 |
| 8 | 8 | 25.71ms | 38.07ms | 25 |
| 9 | 9 | 101.13ms | 113.80ms | 29 |
| 10 | 10 | 20.00ms | 22.43ms | 31 |
| 11 | 11 | 68.95ms | 87.00ms | 25 |
| 12 | 12 | 39.25ms | 47.88ms | 35 |
| 13 | 13 | 117.38ms | 137.81ms | 35 |
| 14 | 14 | 41.98ms | 64.15ms | 29 |
| 15 | 15 | 60.36ms | 79.19ms | 35 |

**Key insight**: Near-linear speedup (6.72x on ~8 cores). Embarrassingly parallel — each map solve is independent. Individual per-map times include rayon scheduling overhead.

## Summary

| Example | Maps | Avg Time | Key Result |
|---------|------|----------|------------|
| `tactical_terrain` | 3 | ~375µs | Cost-optimal paths through sand/water |
| `dungeon_multifloor` | 2 | ~850ms | 2-3 floor dungeons, cross-floor A* |
| `dungeon_tui` | 1 | N/A | Multi-floor TUI with floor sidebar + stair nav |
| `tactical_procedural` | 15 | ~5.8ms | Random maps, 100% solvable |
| `tactical_parallel` | 15 | ~171ms total | Rayon batch, 6.72x speedup |

## Test Results

- **234 lib tests**: All pass ✅
- **80 integration tests**: All pass ✅
- **Clippy**: Zero warnings ✅