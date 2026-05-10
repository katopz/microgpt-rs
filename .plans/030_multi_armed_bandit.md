# Plan 030: Multi-Armed Bandit PoC

## Goal

Add a multi-armed bandit (MAB) module that implements `ScreeningPruner` with adaptive relevance,
demonstrating how microgpt-rs's trait-based architecture naturally extends to sequential
decision-making under uncertainty.

## Context

The `ScreeningPruner` trait returns `f32` relevance `[0.0, 1.0]` — this IS a reward signal.
The DDTree's best-first search with budget IS exploration. What's missing is **policy update
across episodes**. This PoC closes the loop:

```
Episode N: DDTree proposes arms (tokens) → BanditPruner scores via UCB1 → select → observe reward → update Q-values
Episode N+1: Updated Q-values bias relevance → better arm selection
```

This does NOT change core hot paths. It adds a new pruner that implements existing traits.

### Constrained Bandit (Action Masking)

`BanditPruner` wrapping a domain `ScreeningPruner` enables **constrained bandits** —
RL exploration gated by neuro-symbolic rules. Domain pruner returns `relevance = 0.0`
for invalid arms → bandit score overridden → DDTree never explores them, regardless
of Q-value. This bridges RL with the existing constraint pruning architecture.

```
┌─────────────────┐    ┌──────────────┐    ┌─────────────┐
│ Domain Pruner   │    │ BanditPruner │    │   DDTree    │
│ (action mask)   │───▶│ relevance()  │───▶│ screened    │
│ blocked → 0.0   │    │ domain × β   │    │ build       │
│ valid   → 1.0   │    │ 0.0 if blocked│    │             │
└─────────────────┘    └──────────────┘    └─────────────┘
```

## Architecture

### BanditStrategy Enum (OCP)
- `Ucb1` — Upper Confidence Bound 1 (deterministic, no RNG needed)
- `EpsilonGreedy { epsilon: f32, decay: f32 }` — ε-greedy with decay annealing
- `ThompsonSampling` — Beta distribution posterior sampling

### BanditPruner<P> (Generic wrapper, DRY)
- Wraps any inner `ScreeningPruner` for domain knowledge + action masking
- Maintains per-arm Q-values, visit counts, total pulls
- Implements `ScreeningPruner::relevance()` by blending inner relevance with strategy bonus
- Domain `relevance = 0.0` always overrides bandit score (hard trim)

### BanditStats (Shared state, DRY)
- Per-arm Q-values, visit counts, total pulls
- UCB1 scoring, Thompson posterior sampling — used by both `BanditPruner` and `BanditSession`

### BanditEnv Trait (SRP)
- `fn pull(&self, arm: usize, rng) -> f32` — stochastic reward for an arm
- `fn expected_reward(&self, arm: usize) -> f32` — per-arm mean
- `fn optimal_reward(&self) -> f32` — for regret calculation
- `fn num_arms(&self) -> usize`
- `fn optimal_arm(&self) -> usize`

### Built-in Environments
- `BernoulliEnv` — binary rewards with per-arm success probabilities (classic MAB)
- `GaussianEnv` — continuous rewards with per-arm mean/std (Box-Muller, clamped to [0, 1])

### BanditSession (Orchestration)
- Runs N episodes, tracking cumulative reward and pseudo-regret
- Emits `BanditEvent` stream for visualization/logging (Pull, EpisodeComplete, SessionComplete)
- Returns `BanditResult` with final Q-values, visits, best arm, convergence check

## Tasks

- [x] 1. Create `src/pruners/bandit.rs` with `BanditStrategy` enum
- [x] 2. Add `BanditPruner<P: ScreeningPruner>` implementing `ScreeningPruner`
- [x] 3. Add `BanditEnv` trait + `BernoulliEnv` + `GaussianEnv`
- [x] 4. Add `BanditSession` with episode loop, reward/regret tracking, `BanditEvent` stream
- [x] 5. Register module in `src/pruners/mod.rs` behind `#[cfg(feature = "bandit")]`
- [x] 6. Add `bandit = []` feature flag to `Cargo.toml`, add to `full`
- [x] 7. Add `[[example]] bandit_demo` gated by `bandit` feature
- [x] 8. Add unit tests for UCB1, ε-greedy, Thompson Sampling convergence
- [x] 9. Add integration test: 5-armed Bernoulli bandit with known optimal
- [x] 10. Update `README.md` with Bandit section + benchmark results
- [x] 11. Distill constrained bandit: `BanditPruner` wrapping domain `ScreeningPruner` with action masking + tests + example section

## File Changes

| File | Action | Description |
|------|--------|-------------|
| `src/pruners/bandit.rs` | Create | Core module: strategy, pruner, env, session |
| `src/pruners/mod.rs` | Edit | Add `bandit` module + re-exports |
| `src/lib.rs` | Edit | No changes needed (re-exported via pruners) |
| `Cargo.toml` | Edit | Add `bandit = []` feature, example entry |
| `examples/bandit_demo.rs` | Create | 5-armed Bernoulli demo with regret plot |
| `README.md` | Edit | Add Bandit section |

## Design Decisions

1. **BanditPruner wraps ScreeningPruner, not replaces it** — domain knowledge from inner
   pruner stays relevant; bandit layer adds exploration bonus on top. Domain `relevance = 0.0`
   always overrides bandit score — action masking for constrained bandits.

2. **BanditEnv is separate from ScreeningPruner** — environment owns reward generation,
   pruner owns action scoring. SRP: one trait, one job.

3. **BanditSession orchestrates episodes** — not embedded in DDTree. The session calls
   DDTree per episode, observes reward, updates pruner. Clean boundary.

4. **Feature-gated** — zero impact when disabled. No changes to `ForwardContext`,
   `SpeculativeContext`, or any hot path.

5. **No external dependencies** — Beta distribution for Thompson Sampling implemented
   via Jöhnk's algorithm with `fastrand` (already in deps). No `rand_distr` needed.

6. **Constrained bandit via action masking** — `BanditPruner` wrapping domain `ScreeningPruner`
   gates RL exploration with neuro-symbolic rules. Invalid arms get `relevance = 0.0`,
   DDTree never explores them, regardless of Q-value. Distilled from external AI suggestion;
   the self-contained PoC code was redundant (we already built all pieces), but the concept
   of "constrained bandit = ScreeningPruner as action mask" is the genuinely useful insight.

## Success Criteria

- [x] 5-armed Bernoulli bandit converges to optimal arm within 500 episodes
- [x] Cumulative regret grows sub-linearly (O(log N) for UCB1/Thompson)
- [x] ε-greedy with decay outperforms fixed ε
- [x] BanditPruner plugs into `build_dd_tree_screened()` without changes
- [x] All tests pass with `cargo test --features bandit` (273 passed: 25 bandit + 248 existing)
- [x] Zero regressions on existing tests (`cargo test` without features: 248 passed)
- [x] Constrained bandit: blocked arm never pulled, domain overrides bandit score, best valid arm found

## Out of Scope

- Contextual bandits (requires feature vectors per arm)
- Multi-agent bandits
- Non-stationary environments (adversarial bandits)
- Integration with real Transformer weights (this PoC uses simulated rewards)