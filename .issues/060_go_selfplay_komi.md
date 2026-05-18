# Issue 060: Go Self-Play Komi Imbalance — 98.6% Black Wins

## Severity

Medium — self-play learning signal is broken for GZero

## Summary

GZero self-play (`go_04_gzero`) produces 98.6% Black wins across 500 episodes on 9×9 with komi=7.5. The first-move advantage overwhelms template-based learning, making outcome-based delta-gating meaningless.

## Current Behavior

```
GZero Self-Play Results (500 episodes, 9×9, komi=7.5):
  Black Wins:  493 (98.6%)
  White Wins:    7 (1.4%)
  Draws:         0

Template δ Ranking:
  Capture:    +0.0000 (only neutral template)
  CornerStar: -7.25
  Tenuku:     -9.50
  Defend:     -50.22
```

## Root Cause Analysis

Komi IS applied correctly — `GoState::new()` uses `DEFAULT_KOMI = 7.5`, and `score()` adds komi to white_score before computing `black_score - white_score`. The problem is structural:

1. **Weak players amplify first-move advantage** — Both sides use identical templates (Capture, CornerStar, Tenuki, Defend). Black moves first, securing territory before White can respond. With no strategic depth, this compounds every turn.

2. **Delta-gating sees no signal** — Template deltas are computed from move quality, but the game outcome is predetermined by color assignment. All templates get negative δ because "playing any template as White = losing."

3. **7.5 komi is calibrated for strong players** — In professional Go, 7.5 komi produces ~50% Black wins. But pros extract maximum value from every move. Weak players leave so much on the board that the 7.5 point compensation is negligible.

## Proposed Solutions

### Option A: Adaptive Komi (Recommended)

Dynamically adjust komi based on observed win rates during self-play.

```text
if black_win_rate > 0.7:
    komi += delta_komi  (e.g., +2.0)
if white_win_rate > 0.7:
    komi -= delta_komi  (e.g., -2.0)
clamp komi to [0.0, 20.0]
re-evaluate every 100 episodes
```

### Option B: Swap Colors Mid-Training

Each agent plays both colors equally. Episode N: A=Black, B=White. Episode N+1: A=White, B=Black. Aggregate deltas across both colors per agent.

### Option C: Score-Based Rewards Instead of Win/Loss

Use margin of victory as reward signal instead of binary win/loss.

```text
reward = score / abs(score).max(1.0)  # normalized to [-1, 1]
```

This gives partial credit even to the losing side if they played well.

### Option D: Handicap Stone Placement

Instead of komi, give White 1-2 handicap stones (placed before game starts). More directly compensates for move-order advantage.

## Recommended Approach

Combine **Option A** (adaptive komi) + **Option C** (score-based rewards):

1. Start with komi=7.5
2. Every 100 episodes, check win balance
3. Adjust komi by ±2.0 toward 50/50 balance
4. Use score margin as reward signal (not binary win/loss)
5. This gives meaningful deltas even when the game is lopsided

## Files to Modify

| File | Change |
|------|--------|
| `src/pruners/go/g_zero_player.rs` | Add adaptive komi logic to `run_gzero_selfplay()`, change reward from binary to score-based |
| `src/pruners/go/state.rs` | Add `GoState::with_komi()` already exists, may need `set_komi()` |
| `examples/go_04_gzero.rs` | Report komi adjustments during training |
| `.docs/15_go_arena.md` | Update results after fix |

## Success Criteria

- [ ] Black win rate between 40-60% over 500 episodes
- [ ] Template deltas reflect move quality, not color assignment
- [ ] At least 2 templates promoted via delta-gating absorb-compress
- [ ] Updated docs with new results

## References

- `go_04_gzero` example output
- `.docs/15_go_arena.md` — full Go arena documentation
- Plan 065 — Go arena implementation