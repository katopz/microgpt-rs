# Benchmark 001: Tactical AI — Strategic vs Brute-Force DDTree

**Date**: 2025-01-XX
**Plan**: 017 (Phase 5)
**Example**: `cargo run --example tactical_bench`

## Setup

- **Brute-force**: DDTree on micro-actions (vocab=5: Up/Down/Left/Right/Attack)
- **Strategic**: DDTree on target tokens (vocab=N targets) + A* path expansion
- **Hardware**: macOS (dev build, unoptimized)

## Maps Tested

| Map | Size | Monsters | Treasures | Goal | Description |
|-----|------|----------|-----------|------|-------------|
| Small | 2×3 | 2 | 2 | (1,2) | Blue Bear original (BXT/SMG) |
| Original | 17×16 | 3 | 3 | (8,14) | Complex dungeon with walls |
| Arena | 16×16 | 3 | 3 | (14,14) | Open arena, few walls |
| Corridor | 16×16 | 3 | 3 | (14,14) | Horizontal walls with gap corridors |

## Results

| Map | Approach | Nodes | Time | Steps | Solved |
|-----|----------|-------|------|-------|--------|
| Small | Brute-force | 269 | 2.1ms | 7 | ✅ |
| Small | Strategic | 4 | 160µs | 7 | ✅ |
| Original | Strategic | 63 | 69.7ms | 125 | ✅ |
| Original | Brute-force | — | N/A | — | ❌ infeasible |
| Arena | Strategic | 662 | 670.5ms | 67 | ✅ |
| Corridor | Strategic | 613 | 416.3ms | 59 | ✅ |

## Analysis

### Why Brute-Force Fails at Scale

- **DDTree constraint**: max lookahead = 8 (u128/16 bits per token)
- **Brute-force state space**: 5^8 = 390,625 nodes max
- **16×16 puzzles require**: 59-125 action steps
- **Conclusion**: Brute-force only works for ≤8 step puzzles

### Why Strategic Scales

- **DDTree operates on targets** (7 tokens for 3M+3T+1G)
- **State space**: 7! = 5,040 permutations max
- **A* handles movement**: expands each target into 10-30 step paths
- **Result**: solves 125-step puzzles in ~70ms, 67-step puzzles in ~670ms

### Node Count Insight

- Small map strategic: 4 nodes (direct path, no backtracking)
- Original dungeon: 63 nodes (7 targets, moderate backtracking)
- Arena/Corridor: 600+ nodes (more backtracking due to open spaces)

The higher node count on Arena/Corridor maps suggests the solver explores more
permutations before finding a valid visit order. This is expected — open maps
have more valid paths, so the pruner has less constraint power.

## Verification

All maps verified through:
1. DDTree finds target visit order
2. A* expands targets into action sequence
3. Actions replayed through TacticalPruner
4. Assertions: bear at goal, all treasures collected, all monsters killed

```
✅ All assertions passed. All maps verified solvable.
```
