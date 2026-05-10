# Research: Learning Beyond Gradients — Heuristic Learning (14)

> Source: [Learning Beyond Gradients](https://trinkle23897.github.io/learning-beyond-gradients/) by Jiayi Weng
> Date: 2026-05-10 (published), distilled 2025-06

## Summary

Weng discovered that LLM coding agents (Codex/gpt-5.4) can maintain and evolve **programmatic heuristic policies** that rival or exceed Deep RL baselines — without any neural network training. The key insight: coding agents change the **maintenance cost curve** for heuristics, making rules that were once "too expensive to own" into viable long-term code.

Breakout reached theoretical max (864). Ant reached 6000+ via CPG + residual MPC. HalfCheetah reached 11836.7. VizDoom D3 Battle reached mean=557.0 with pure cv2/NumPy. Atari57 median HNS matched PPO baselines at same step count. All without training a neural network.

---

## Core Concepts

### Heuristic Learning (HL)

- **HL** = learning loop where the object being updated is **software structure**, not neural network parameters
- **Heuristic System (HS)** = the maintained artifact: policy, state detectors, tests, replays, memory, update mechanism
- Feedback consumed by coding agent, not backpropagation
- Updates are direct code edits, not gradient steps

### HL vs Deep RL Comparison

| Axis | Deep RL | HL |
|---|---|---|
| Policy | NN parameters | Code: rules, state machines, controllers, MPC |
| State | Observations | Explicit variables, detectors, caches |
| Action | NN forward pass | Code execution |
| Feedback | Fixed reward | Tests, logs, replays, environment feedback |
| Update | Gradient descent | Direct code edits by agent |
| Memory | Replay buffer | Trials, summaries, replays, version diffs |

### Key Properties

1. **Explainability**: Code policies translate to plain language
2. **Sample Efficiency**: One code edit jumps to new policy (no learning rate tuning)
3. **Regression-testable**: Old capabilities become tests, replays, golden cases
4. **Constrained overfitting**: Simplification + multi-seed evaluation = engineering regularization
5. **Partial catastrophic forgetting avoidance**: Old capabilities written into rule sets and tests

### The Two Operations of a Healthy HS

1. **Absorb**: Write new failures, logs, rewards back into the system
2. **Compress**: Fold local patches into simpler, more maintainable representations

> An HS that only grows and never compresses becomes a big ball of mud.

### Coupling Complexity

Defined as: how many interdependent states, rules, tests, feedback signals, and historical constraints an update must account for simultaneously. Bounded by:
- **Code side**: module boundaries, interface stability, test coverage, observability, rollback cost, state reproducibility
- **Agent side**: model capability, context length, memory quality, tool quality, iteration speed

Hypotheses:
- Clearer feedback → higher maintainable coupling complexity
- Stronger models → handle higher coupling complexity
- Modularity/tests/replays move coupling into environment
- Only-growing-never-compressing → complexity exceeds maintenance capacity

### System 1 / System 2 Split

- **System 1 (fast)**: Specialized shallow NNs (perception) + HL (rules, tests, memory, safety)
- **System 2 (slow)**: LLM agent (feedback, data improvement, periodic NN updates)

### The "Montezuma Boundary"

Montezuma's Revenge exposed where reactive `if/else` fails: need macro-actions, recoverable search state, long-term memory. Some environments need new program forms before feedback can enter the system.

---

## Experimental Results (Selected)

### Breakout: 387 → 507 → 839 → 864 (theoretical max)

- 387: ball interception, no loop breaking
- 507: stuck-loop perturbation (offset on predicted landing point)
- 839: fast-low-ball lead correction
- 864: late-game offset release + paddle drift compensation
- RAM→RGB migration: 14,504 image-only steps after structure stable in RAM

### Ant: 2291 → 6146 (CPG + Residual MPC)

- 2291: four-leg phase oscillator + PD controller
- 3162: second/third harmonics
- 3635+: residual MPC on top of CPG base gait
- 6146: speed-adaptive phase + terminal velocity cost

### Key Pattern: Base + Residual

```
base_heuristic() + residual_correction(online_planning) = final_action
```

This is the same pattern as speculative decoding: draft model proposes, target model verifies and corrects.

---

## Application to microgpt-rs

### Direct Mappings

| HL Concept | microgpt-rs Equivalent | Status |
|---|---|---|
| Heuristic Policy | `ConstraintPruner::is_valid()` | ✅ Strong |
| Feedback signal | `SolveEvent` (Accept/Reject/Prune/Score) | ✅ Strong |
| Update without backprop | `BanditPruner` Q-value updates | ✅ Strong |
| Regression testing | `riir-validator-check` ABI compliance | ✅ Partial |
| Memory | `BanditStats`, `BanditEvent` | ✅ Partial |
| Modular decomposition | Tiered `SynPruner`, WASM validators | ✅ Strong |
| Absorb + Compress | Missing — bandit accumulates but never compresses | ❌ Gap |
| Trial persistence | Missing — `BanditEvent` is in-memory only | ❌ Gap |
| Coding agent in loop | Phase 3 cargo check (placeholder) | ❌ Gap |

### Reference Implementation: raw/bomby/

[Fish Folk: Bomby](https://github.com/fishfolk/bomby) lives at `raw/bomby/` — a Bevy ECS Bomberman with LDtk levels, sprite rendering, and audio. We extract game logic patterns (Components, Resources, Systems) into our arena using `bevy_ecs` standalone + ratatui emoji TUI. See Plan 033.

| bomby Module | We Extract | We Drop |
|---|---|---|
| `bomb.rs` | Bomb placement, fuse timer, blast range, chain explosions, player death | Bevy `Sprite`, `Transform`, `EventWriter` |
| `player.rs` | Grid movement, wall/bomb collision, 4-player spawn | Bevy sprites, `InputManagerBundle`, animation |
| `ldtk.rs` | Grid coordinate math (`to_grid`, `to_world`) | LDtk level loading, asset loading |
| `audio.rs`, `camera.rs`, `debug.rs`, `z_sort.rs`, `ui.rs` | Nothing | All — replaced by ratatui |

### What to Build (Gap Analysis)

1. **TrialLog**: Persist `BanditEvent` → JSONL with score/steps/config/note per episode
2. **AbsorbCompress**: Auto-promote stable low-Q arms to hard constraints in `BlockedArmPruner`
3. **HotSwapPruner**: Runtime `.wasm` reload without restarting the process
4. **RegressionSuite**: Replay golden episodes, verify score ≥ baseline
5. **Coupling metric**: Track active rules per validator, flag threshold exceeded
6. **Bomberman arena**: `bevy_ecs` standalone + ratatui TUI, 4-player HL proof (Plan 033)

### The "Residual MPC" = Speculative Decoding Analogy

The article's CPG + residual MPC pattern maps to draft model + target model verification:

```
Article:  base_gait(CPG) + residual(MPC rollout)     = final_action
microgpt: draft_model(marginals) + target_model(verify) = final_token
```

The `BanditPruner` wrapping a `ScreeningPruner` is the closest analog to residual MPC — domain pruner provides the base, bandit provides the adaptive residual.

### System 1/System 2 in Our Architecture

```
System 1 (fast, ~100µs/token):
  LoRA-adapted Transformer + WasmPruner + BanditPruner + ConstraintPruner

System 2 (slow, seconds):
  Phase 3 cargo check loop + Curator API + coding agent
  (reads failures, writes new validators, updates LoRA)
```

---

## Citation

```bibtex
@misc{weng2026learning_beyond_gradients,
  title = {Learning Beyond Gradients},
  author = {Weng, Jiayi},
  year = {2026},
  month = may,
  howpublished = {\url{https://trinkle23897.github.io/learning-beyond-gradients/}},
  note = {Blog post}
}
```
