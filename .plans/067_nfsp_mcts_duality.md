# Plan 067: NFSP/MCTS Duality Unification

> **Status**: Draft
> **Feature Gate**: `bandit_mcts` (implies `bandit`, `game_state`, `bomber`)
> **Depends On**: Plan 056 (GameState Forward Model), Plan 049 (G-Zero)

## Context

The NFSP/MCTS duality: both find a better action at state `s` for a student policy to imitate.
They differ only in where the better action comes from:

| Teacher | Method | Direction | Our Component |
|---------|--------|-----------|---------------|
| A (NFSP) | Q-learning on observed trajectories | ← Backward from past | `BanditPruner` Q-values |
| B (MCTS) | UCT expansion via simulated rollouts | → Forward into futures | `mcts_search<S>()` |

**Critical finding**: Generic MCTS ≈ random (25% each) in Bomberman because it has no backward signal.
HL's BanditPruner carries Q-values across episodes and dominates (+177 vs +131 Greedy).

**Insight**: Wire Teacher A into Teacher B — bandit Q-values inform MCTS rollouts.
This is the AlphaZero pattern, but modelless (no neural net, just bandit Q-values).

## What We Have

- `BanditStats`: Q-values, visit counts, UCB1 scoring — the backward signal
- `mcts_search<S>()`: Generic MCTS with random rollouts — forward search, no memory
- `StateHeuristic<S>`: Pluggable leaf evaluation — already supports domain knowledge
- `ReplayBackwardWalker`: Retrospective analysis from winning replays — NFSP pattern
- `HintDelta`: Counterfactual signal (neither backward nor forward)
- `AbsorbCompress`: Student that absorbs stable heuristics into hard constraints
- DDTree + BanditPruner: Already embodies this duality at the token level

## What's Missing

1. MCTS uses random rollout policy — ignores accumulated bandit Q-values
2. No `RolloutPolicy` trait — rollout is hardcoded as `rng.usize(0..actions.len())`
3. No bridge between `BanditStats` and `StateHeuristic` — they live in separate worlds
4. No benchmark comparing MCTS vs BanditMCTS vs HL in game arenas

## Architecture

### New Trait: `RolloutPolicy`

```rust
/// Pluggable rollout policy for MCTS.
/// Replaces hardcoded random selection with informed action choice.
pub trait RolloutPolicy<S: GameState> {
    /// Select an action during MCTS rollout.
    /// `state`: current rollout state
    /// `actions`: available actions for player
    /// `player_id`: which player is acting
    /// `rng`: RNG for stochastic policies
    fn select(
        &mut self,
        state: &S,
        actions: &[S::Action],
        player_id: u8,
        rng: &mut Rng,
    ) -> usize;
}
```

### New Struct: `BanditRolloutPolicy`

Wraps `BanditStats` to provide informed rollout selection.
Uses ε-greedy: exploit bandit Q-values with probability (1-ε), explore randomly otherwise.

```rust
pub struct BanditRolloutPolicy<S: GameState> {
    stats: BanditStats,
    epsilon: f32,
    action_lookup: PhantomData<S>,
}
```

Key challenge: BanditStats tracks arms by index (0..N), but `GameState::Action` is a domain enum.
We need a stable mapping from `Action` → arm index. `BomberAction` has 6 variants (fixed), so
we can derive a simple `action_index()` method or use the existing `available_actions()` ordering.

### New Function: `mcts_search_informed<S>()`

Like `mcts_search` but accepts a `&mut dyn RolloutPolicy<S>` instead of hardcoded random.

### New Struct: `BanditBomberHeuristic`

Combines `BomberHeuristic` (domain knowledge) with `BanditStats` (backward signal):

```rust
pub struct BanditBomberHeuristic {
    domain: BomberHeuristic,
    bandit_weight: f32,  // λ: how much to trust bandit vs domain heuristic
}
```

Evaluates as: `domain.evaluate(s, pid) + λ * bandit_q_bonus(s, pid)`

### Data Flow

```
Episode 1..N:
  BanditPruner accumulates Q-values across episodes (Teacher A, backward)
       │
       ▼
  BanditRolloutPolicy wraps Q-values for MCTS rollouts
       │
       ▼
  mcts_search_informed() uses bandit-guided rollouts (Teacher B, forward + informed)
       │
       ▼
  Better action selected → environment reward → update bandit (close the loop)
       │
       ▼
  ReplayBackwardWalker extracts backward policy data from wins (optional)
       │
       ▼
  AbsorbCompress promotes stable heuristics to hard constraints (student)
```

## Tasks

- [ ] T1: Add `RolloutPolicy<S>` trait to `game_state/mod.rs`
- [ ] T2: Implement `RandomRolloutPolicy` (wraps existing random logic, for parity testing)
- [ ] T3: Implement `BanditRolloutPolicy<S>` in `game_state/mcts.rs`
- [ ] T4: Add `action_index()` to `BomberAction` (stable arm mapping)
- [ ] T5: Refactor `mcts_search` to accept `&mut dyn RolloutPolicy<S>` (backward-compatible)
- [ ] T6: Implement `BanditBomberHeuristic` combining domain + bandit signals
- [ ] T7: Add `mcts_search_informed()` function with pluggable policy + heuristic
- [ ] T8: Create benchmark test: `bench_bandit_mcts.rs`
- [ ] T9: Run benchmark: MCTS (random) vs BanditMCTS vs HL vs Random — 100+ rounds
- [ ] T10: Update README.md with results and duality documentation

## Benchmark Plan

```
Players (100-round tournament):
  P0: BanditMCTS (budget=200, depth=10, ε=0.2, bandit carries across rounds)
  P1: MCTS (budget=200, depth=10, random rollouts, no memory)
  P2: Random
  P3: Random

Hypothesis: BanditMCTS > MCTS ≈ Random > Random
  - BanditMCTS benefits from backward signal informing forward search
  - Plain MCTS ≈ random (already confirmed in Plan 056)

Run: cargo test --features "bandit_mcts" --test bench_bandit_mcts -- --nocapture
```

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Bomberman action space too small (6) for MCTS to help | Medium | Low | Also benchmark on Monopoly/FFT |
| Bandit Q-values too noisy early on | Medium | Low | Warmup bandit for N episodes before enabling MCTS |
| Action→index mapping fragile | Low | Medium | Use enum discriminant or explicit lookup table |
| No improvement over plain HL | Medium | None | Document finding, consider for larger action spaces |

## Success Criteria

- [ ] BanditMCTS win rate > plain MCTS win rate (≥10pp improvement)
- [ ] `RolloutPolicy` trait is generic over any `GameState`
- [ ] Feature-gated: `bandit_mcts` (not in `full` by default)
- [ ] No regressions in existing MCTS or BanditPruner tests

## The Bigger Picture

```
              Past                    Future
         ┌──────────────────┬──────────────────────┐
  Real   │ ReplayBackward  │  MCTS rollouts        │
         │ (NFSP Q-learn)  │  (UCT expansion)      │
         │ BanditPruner    │  mcts_search()        │
         ├──────────────────┼──────────────────────┤
  Counter│ Bandit Q-update  │  Hint-δ              │
 factual │ (what worked)   │  (what model doesn't  │
         │                  │   know)               │
         └──────────────────┴──────────────────────┘

  Student: AbsorbCompress (doesn't know which teacher spoke)
```

This plan unifies the top-right (MCTS) with the bottom-left (Bandit) by wiring backward
signals into forward search. The DDTree pipeline already does this at the token level —
we're extending the same pattern to game state search.