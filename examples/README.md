# Examples

All examples run with `cargo run --example <name>`. Some require feature flags.

## Quick Reference

| # | Group | Examples | Feature |
|---|-------|----------|---------|
| 1 | Bandit (RL) | 7 examples | `bandit` |
| 2 | Heuristic Learning | 2 examples | `bandit` |
| 3 | Bomberman Arena | 3 examples | `bomber` |
| 4 | Monopoly FSM | 4 examples | `monopoly` |
| 5 | FFT Tactics Arena | 1 example | — |
| 6 | Blue Bear | 2 examples | — |
| 7 | Core | 4 examples | varies |
| 8 | Dungeon | 2 examples | — |
| 9 | Sudoku | 3 examples | `sudoku` |
| 10 | Tactical AI | 6 examples | — |
| 11 | Getting Started | 1 example | — |
| 12 | Stepwise Reward Shaping | 1 example | `stepcode` |

---

## 1. Bandit (RL / Game Theory)

Multi-armed bandit strategies for adaptive decision-making under uncertainty.

### bandit_01_basic

Basic bandit strategy comparison: UCB1, ε-greedy with decay, Thompson Sampling on a 5-armed Bernoulli bandit with ASCII regret plot.

```bash
cargo run --example bandit_01_basic --features bandit
```

### bandit_02_ddtree

DDTree-based speculative decoding with bandit arm selection for token pruning.

```bash
cargo run --example bandit_02_ddtree --features bandit
```

### bandit_03_slot

Slot machine simulation demonstrating bandit reward optimization.

```bash
cargo run --example bandit_03_slot --features bandit
```

### bandit_04_combat

Combat scenario with bandit-driven action selection and opponent modeling.

```bash
cargo run --example bandit_04_combat --features bandit
```

### bandit_05_rps

Rock-Paper-Scissors tournament with bandit strategy adaptation.

```bash
cargo run --example bandit_05_rps --features bandit
```

### bandit_06_resolver

Conflict resolver using constrained bandit with action masking — blocked arms get relevance 0.0, never explored even if highest reward.

```bash
cargo run --example bandit_06_resolver --features bandit
```

### bandit_07_director

Director pattern — meta-bandit that selects which sub-bandit strategy to activate per round.

```bash
cargo run --example bandit_07_director --features bandit
```

---

## 2. Heuristic Learning (HL)

Trial logging and hot-swapping for the HL (Heuristic Learning) infrastructure.

### hl_01_trial_log

Trial logging with structured outcome tracking — logs strategy selection, reward, and Q-value updates per trial.

```bash
cargo run --example hl_01_trial_log --features bandit
```

### hl_02_hotswap

Hot-swap demo — dynamically switches between pruner strategies at runtime based on performance feedback.

```bash
cargo run --example hl_02_hotswap --features bandit
```

---

## 3. Bomberman HL Arena

4-player Bomberman with `bevy_ecs` standalone. Tick-based priority FSM with 8 states and 4 AI tiers.

### bomber_01_arena

Headless 100-game tournament. Per-game results, cumulative standings, and HL thesis check.

```bash
cargo run --example bomber_01_arena --features bomber
```

### bomber_02_tui

Animated ratatui TUI replay with board rendering, player stats, and event log. Controls: Space/←/→/F/A/Q.

```bash
cargo run --example bomber_02_tui --features bomber
```

### bomber_03_hl_proof

1000-game HL proof experiment. Measures win rate, survival rate, and bandit learning Q-values.

```bash
cargo run --example bomber_03_hl_proof --features bomber
```

---

## 4. Monopoly FSM Arena

4-player Monopoly with `bevy_ecs` standalone. Turn-based event-driven FSM with 8 phases, 40-square board, and 4 AI tiers.

### monopoly_01_arena

Headless 100-game tournament with cumulative standings and HL thesis check.

```bash
cargo run --example monopoly_01_arena --features monopoly
```

### monopoly_02_tui

Animated ratatui TUI replay with colored property groups, player stats, and scrollable event log. Controls: Space/←/→/F/A/Home/End/Q.

```bash
cargo run --example monopoly_02_tui --features monopoly
```

### monopoly_03_hl_proof

1000-game HL proof experiment. Survival rates, win rates, bandit Q-values, and statistical significance check.

**Results:** HL 56.5% win rate, 93.7% survival, +41.3pp over Validator. ✅ HL Thesis PROVEN.

```bash
cargo run --example monopoly_03_hl_proof --features monopoly
```

### monopoly_04_bench

Performance benchmark — measures game throughput, per-turn latency, and latency distribution (p50/p90/p99).

**Performance:** 87 games/sec, 41µs/turn (25× under 1ms target).

```bash
cargo run --example monopoly_04_bench --features monopoly
```

---

## 5. FFT Tactics Arena

Final Fantasy Tactics-inspired 4v4 turn-based battle. Pure data-driven (no ECS), speed-based turn queue, 4 classes with HP/MP, and 4 AI tiers.

### fft_01_arena

Headless 100-round tournament. 8 units (4v4) with classes: Knight, Archer, Black Mage, White Mage. AI strategies: Random, Greedy, Validator, HL (bandit Q-learning). Outputs per-round kills, final standings, unit stats, and HL Q-value convergence.

```bash
cargo run --example fft_01_arena
```

---

## 6. Blue Bear

Experimental tools and TUI demos.

### bear_01_demo

Basic Blue Bear demonstration.

```bash
cargo run --example bear_01_demo
```

### bear_02_tui

Blue Bear ratatui TUI visualization.

```bash
cargo run --example bear_02_tui
```

---

## 7. Core

Core library features — validation, inference, sampling.

### core_01_validator

Syntax-aware token pruning with `SynPruner`. BPE tokenizes Rust source, validates partial syntax with `syn`, only explores syntactically valid branches.

```bash
cargo run --example core_01_validator --features validator
```

### core_02_raven

Raven inference demo.

```bash
cargo run --example core_02_raven
```

### core_03_ppot

PPoT logit-parameterized CPU resampling.

```bash
cargo run --example core_03_ppot --features ppot
```

### core_04_prefill

Prefill/prompt processing demo.

```bash
cargo run --example core_04_prefill
```

---

## 8. Dungeon

Roguelike dungeon generation and TUI.

### dungeon_01_tui

Ratatui TUI dungeon explorer with procedural generation.

```bash
cargo run --example dungeon_01_tui
```

### dungeon_02_multifloor

Multi-floor dungeon generation with stairs and floor transitions.

```bash
cargo run --example dungeon_02_multifloor
```

---

## 9. Sudoku

Streaming "Thinking" Sudoku solver with deterministic validation.

### sudoku_01_9x9

Standard 9×9 solver demonstrating deterministic rules engine, O(log N) attention, and streaming step-by-step constraint satisfaction.

```bash
cargo run --example sudoku_01_9x9 --features sudoku
```

### sudoku_02_speculative

DDTree + Deterministic Validator pruning with 3-level comparison: Unpruned vs Static-Only vs Path-Aware. Shows path-aware pruning catches cross-depth conflicts.

```bash
cargo run --example sudoku_02_speculative --features sudoku
```

### sudoku_03_tui

Ratatui TUI with color-coded grid, step/trace panels, and speculative mode comparison side-by-side.

```bash
cargo run --example sudoku_03_tui --features sudoku
```

---

## 10. Tactical AI

Grid-based tactical AI with terrain, procedural maps, and parallel simulation.

### tactical_01_ai

Basic tactical AI decision-making on a grid.

```bash
cargo run --example tactical_01_ai
```

### tactical_02_terrain

Terrain-aware pathfinding and movement cost calculation.

```bash
cargo run --example tactical_02_terrain
```

### tactical_03_procedural

Procedural map generation for tactical scenarios.

```bash
cargo run --example tactical_03_procedural
```

### tactical_04_parallel

Parallel tactical simulation with rayon — runs multiple scenarios concurrently.

```bash
cargo run --example tactical_04_parallel
```

### tactical_05_bench

Performance benchmark for tactical AI systems.

```bash
cargo run --example tactical_05_bench
```

### tactical_06_tui

Ratatui TUI tactical map viewer with unit positions and terrain rendering.

```bash
cargo run --example tactical_06_tui
```

---

## 11. Getting Started

### hello_py2rs

Python-to-Rust migration primer — demonstrates idiomatic Rust patterns for Python developers.

```bash
cargo run --example hello_py2rs
```

---

## 12. Stepwise Reward Shaping (StepCodeReasoner Plan 054)

Intra-trajectory reward shaping distilled from StepCodeReasoner (ICML 2026). Rewards bandit arms proportionally to how many downstream arms they enable.

### stepcode_01_shaped_bandit

Demonstrates flat vs shaped reward convergence, path consistency metrics, and the effect of λ (shaping coefficient).

```bash
cargo run --example stepcode_01_shaped_bandit --features stepcode
```

---

## Feature Flags

| Flag | Gates | Dependencies |
|------|-------|-------------|
| `sudoku` | SudokuPruner, sudoku examples | — |
| `validator` | SynPruner, syntax validation | `syn`, `proc-macro2` |
| `ppot` | PPoT resampling | — |
| `bandit` | BanditPruner, bandit/HL examples | — |
| `bomber` | Bomberman arena (Plan 033) | `bevy_ecs`, `bandit` |
| `monopoly` | Monopoly FSM arena (Plan 035) | `bevy_ecs`, `bandit` |
| — | FFT Tactics Arena (Plan 047) | `fastrand` |
| `rest` | REST API client | `reqwest`, `tokio` |
| `embedding_router` | Semantic embedding retrieval | — |
| `gpu` | GPU compute | `wgpu`, `safetensors` |
| `leviathan` | LeviathanVerifier rejection sampling | — |
| `full` | All of the above | — |

```bash
# Run with specific feature
cargo run --example monopoly_01_arena --features monopoly

# Run with all features
cargo run --example sudoku_01_9x9 --features full