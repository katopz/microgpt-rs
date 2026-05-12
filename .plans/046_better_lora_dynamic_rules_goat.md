# Plan 046: Better LoRA + Dynamic Rules GOAT Proof

**Branch:** `develop/feature/046_better_lora_dynamic_rules_goat`
**Depends on:** Plan 033 (Arena), Plan 034 (WASM Validator), Plan 045 (Tech Isolation A/B)
**Goal:** Fix training data quality, retrain LoRA to convergence, prove dynamic LoRA+WASM beats static HL.

---

## Problem Statement

Plan 045 showed HL (+475) >> LoRA+WASM (-15). The gap exists because:

1. **Training data is garbage-in-garbage-out:**
   - 7056 samples, ALL quality=1.0 (only winners survive, no negative examples)
   - No death states, no bad actions, no failure patterns
   - Model never learns "what NOT to do"

2. **LoRA barely converged:**
   - `final_loss: 17.03` (no improvement over training)
   - `action_accuracy: [0,0,0,0,0,0]` (literally zero)
   - Only 2 epochs, 20 steps — woefully undertrained

3. **No dynamic rules proof:**
   - HL uses static heuristics + bandit (proven good)
   - LoRA+WASM uses static model + static validator (proven mediocre)
   - Missing: dynamic model that adapts mid-game

---

## Hypothesis

> A properly trained LoRA on quality data (including failures) can match or beat heuristic HL.
> Dynamic rules (hot-swapped LoRA/WASM between rounds) prove the GOAT thesis:
> **adaptation > static** regardless of which layer adapts.

---

## Tasks

### Phase 1: Data Quality Fix (microgpt-rs)

- [x] **T1: `bomber_06_replay_gen_v2.rs` — Balanced replay generator**
  - Include ALL player types (Random, Greedy, Validator, HL)
  - Include losers AND winners (quality reflects outcome)
  - Add per-tick danger signal: `is_in_danger` bool
  - Add opponent distance as feature
  - Target: ~50K samples, quality spread 0.0–1.0
  - Filter: quality >= 0.3 (not just winners) for positive set
  - Filter: quality < 0.3 for negative set (what NOT to do)

- [x] **T2: Enhance `ReplaySample` with richer features**
  - Add `danger_level: u8` (0=safe, 1=adjacent blast, 2=in blast zone)
  - Add `nearest_opponent_dist: u8` (manhattan distance, 255=none visible)
  - Add `escape_routes: u8` (count of safe adjacent cells)
  - Keep backward compat (old JSONL still loads, new fields default)

- [x] **T3: Run replay gen v2, produce `output/replays_v2/`**
  - 2000 rounds (2x previous) for more data
  - Verify quality distribution: ~40% high, ~30% mid, ~30% low
  - Verify action distribution: reasonable spread, not dominated by moves
  - Target: 20K-50K samples

### Phase 2: Better LoRA Training (riir-ai)

- [x] **T4: Update `train_bomber.rs` — better training config**
  - Accept v2 replay format with new features
  - Increase epochs: 2 → 10 (with early stopping)
  - Increase learning rate warmup
  - Add validation split (80/20)
  - Add per-epoch action accuracy reporting
  - Target: `final_loss < 5.0`, `action_accuracy > 0.3` on at least 2 actions

- [x] **T5: Train new LoRA on v2 data**
  - Run training with new data
  - Compare loss curves: v1 (flat @17) vs v2 (converging)
  - Save as `game_lora_v2.bin`
  - Keep `game_lora.bin` as v1 baseline

- [x] **T6: Validate LoRA v2 quality**
  - Run inference on held-out samples
  - Check action distribution matches training data
  - Check dangerous states → model predicts safe actions
  - If action_accuracy still 0: investigate encoding, loss function

### Phase 3: Dynamic Rules Proof (microgpt-rs + riir-ai)

- [ ] **T7: `DynamicHLPlayer` in riir-ai — composition-based Full HL**
  - `riir-ai` defines `FullHLPlayer` struct that impls `BomberPlayer`
  - Composes: LoRA v2 proposals + WASM safety + Bandit adaptation
  - Hot-swap: between rounds, reload LoRA weights from updated file
  - Lives in `riir-ai` (uses private artifacts), no changes to `microgpt-rs` player types
  - Architecture:
    ```
    LoRA v2 → score all 6 actions
      ↓
    WASM validator → prune unsafe (safety filter)
      ↓
    Bandit Q-values → blend with LoRA scores (85% LoRA + 15% bandit)
      ↓
    ε-greedy explore (5% — less than HL's 10% because LoRA already explores)
      ↓
    Final action
    ```

- [ ] **T8: `bomber_dynamic_rules_demo.rs` in riir-ai — The GOAT proof**
  - 5-player tournament, 1000 rounds:
    - P0 🐰 Random — baseline
    - P1 🤖 LoRA v1 — old model (proves v2 > v1)
    - P2 🔮 LoRA v2 — new model only
    - P3 🛡️ LoRA v2 + WASM — new model + safety
    - P4 🐵 Static HL — heuristic + bandit (current champion)
    - P5 👑 Dynamic HL — LoRA v2 + WASM + Bandit + HotSwap (challenger)
  - Every 100 rounds: Dynamic HL hot-swaps retrained LoRA
  - Print comparison table every 200 rounds
  - Final verdict: Dynamic HL > Static HL proves dynamic rules GOAT

- [ ] **T9: Run GOAT tournament, analyze results**
  - Expected results:
    ```
    P5 Dynamic HL > P4 Static HL (dynamic > static)
    P2 LoRA v2 > P1 LoRA v1 (better training > garbage training)
    P3 LoRA v2+WASM > P2 LoRA v2 (safety adds value)
    ```
  - If Dynamic HL doesn't beat Static HL: investigate, don't fake results
  - Document actual findings honestly

### Phase 4: Cleanup & Documentation

- [ ] **T10: Update docs**
  - `.plans/033_bomberman_arena.md` — add P4 Full HL note about composition
  - `.docs/08_examples.md` — add bomber_dynamic_rules_demo
  - `README.md` — update results section
  - `.plans/046` — mark tasks complete

---

## Architecture: Composition Over Inheritance

```
microgpt-rs (MIT):
  BomberPlayer trait          ← interface
  LoraAdapter::load()         ← CPU LoRA loading
  lora_apply()                ← CPU LoRA inference
  BomberWasmPruner            ← WASM loading
  HLPlayer                    ← bandit + absorb/compress
  (no secrets, no private players)

riir-ai (Private):
  FullHLPlayer impl BomberPlayer  ← composition of secrets
    fields:
      lora: LoraAdapter           ← Secret A (game_lora_v2.bin)
      wasm: BomberWasmPruner      ← Secret A2 (bomber_validator.wasm)
      q_values: [f32; 6]          ← bandit memory (like HLPlayer)
      visits: [u32; 6]            ← bandit visits
      compressed: [bool; 6]       ← absorb/compress
      round_actions: Vec<u8>      ← trial log
    methods:
      select_action()             ← LoRA → WASM → Bandit blend
      update_outcome()            ← absorb rewards
      hot_swap_lora(path)         ← reload weights between rounds
      compress_cycle()            ← promote low-Q to hard blocks
```

The key insight: `FullHLPlayer` lives entirely in `riir-ai`. It uses the public `BomberPlayer` trait
and public `LoraAdapter`/`BomberWasmPruner` types from `microgpt-rs`, but the composition of all
three secrets (LoRA + WASM + Bandit) is the commercial product. MIT users get the pieces but not
the assembled product — "Ferrari, no gas, no driver."

---

## Data Quality Analysis

### Current v1 JSONL Problems

| Issue | Impact | Fix |
|-------|--------|-----|
| All quality=1.0 | Model never sees failures | Include losers |
| Only HL+Validator types | No diversity | Include all 4 types |
| No danger signal | Can't learn blast avoidance | Add `danger_level` |
| No opponent info | Can't learn hunting | Add `nearest_opponent_dist` |
| 7056 samples | Too few for 6-class classification | Target 20K-50K |
| 2 epochs, 20 steps | Severely undertrained | 10 epochs with early stop |

### Target v2 Quality Distribution

```
quality < 0.3  (bad/death):    ~30%  → "what NOT to do"
quality 0.3-0.7 (survived):   ~30%  → "mediocre play"
quality > 0.7  (won+kills):   ~40%  → "good play to imitate"
```

### Target v2 Training Metrics

| Metric | v1 (current) | v2 (target) |
|--------|--------------|-------------|
| final_loss | 17.03 | < 5.0 |
| action_accuracy | [0,0,0,0,0,0] | avg > 0.3 |
| epochs | 2 | 10 (early stop) |
| samples | 7056 | 20K-50K |

---

## Benchmark Plan

### Before (Plan 045 results, v1 LoRA)

```
  #1  🐵 HL                    +475    143 wins    (91% survival)
  #2  🔮 LoRA+WASM (v1)        -15     56 wins     (92% survival)
  #3  🤖 LoRA (v1)             -46     40 wins     (92% survival)
  #4  🛡️  Heuristic+WASM       -286    69 wins     (84% survival)
```

### After (Plan 046 expected, v2 LoRA + Dynamic HL)

```
  #1  👑 Dynamic HL (LoRA v2+WASM+Bandit+HotSwap)   > +475
  #2  🐵 Static HL (heuristic+bandit)               ~ +475
  #3  🔮 LoRA v2+WASM                               > -15
  #4  🤖 LoRA v2                                    > -46
  #5  🤖 LoRA v1                                    ~ -46
```

### Key Comparisons to Prove

| # | Comparison | Proves | Expected |
|---|-----------|--------|----------|
| 1 | LoRA v2 > LoRA v1 | Better data/training matters | ✅ |
| 2 | LoRA v2+WASM > LoRA v2 | Safety filter adds value | ✅ |
| 3 | Dynamic HL > Static HL | **Dynamic rules > static rules (GOAT)** | ✅ |

If comparison #3 fails: the LoRA signal is still too weak. Investigate:
- Is the encoding losing information?
- Is the model too small (18K params)?
- Is the bandit interfering with LoRA proposals?

Document honestly either way.

---

## File Changes

### microgpt-rs (MIT engine)

```
src/pruners/bomber/
  replay.rs              ← T2: add danger_level, opponent_dist, escape_routes fields
  mod.rs                 ← re-exports unchanged

examples/
  bomber_06_replay_gen_v2.rs  ← T1: balanced replay generator (NEW)
```

### riir-ai (Private)

```
crates/riir-gpu/
  examples/train_bomber.rs    ← T4: better training config, v2 format
  src/game/replay.rs          ← parse v2 ReplaySample format

crates/riir-examples/
  examples/
    bomber_dynamic_rules_demo.rs  ← T8: GOAT proof tournament (NEW)
  src/
    bomber_full_hl.rs             ← T7: FullHLPlayer impl (NEW)
    bomber_full_hl/
      mod.rs
      player.rs                   ← FullHLPlayer struct + BomberPlayer impl

output/
  game_lora_v2.bin            ← T5: new trained LoRA
  training_report_v2.json     ← T5: new training metrics
  replays_v2/                 ← T3: balanced replay data
```

---

## Risks & Mitigations

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| LoRA v2 still doesn't converge | Medium | Check encoding, increase model size to 2 layers |
| Dynamic HL < Static HL | Medium | LoRA noise hurts bandit; reduce blend ratio |
| GPU unavailable for training | Low | CPU fallback path exists in train_bomber.rs |
| JSONL v2 too large | Low | Cap at 50K samples, shuffle before training |

---

## Success Criteria

1. **Data:** v2 JSONL has quality spread 0.0–1.0, 20K+ samples
2. **Training:** `final_loss < 5.0`, at least 2 actions have accuracy > 0.25
3. **Tournament:** LoRA v2 > LoRA v1 (better data works)
4. **GOAT proof:** Dynamic HL > Static HL (dynamic adaptation wins)
5. **Honest:** Results documented regardless of outcome

If criteria 2-4 fail: document findings, analyze why, create follow-up plan. Don't fake results.