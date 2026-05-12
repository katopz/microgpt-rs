# Plan 039: Game Replay Training Data Pipeline

**Branch:** `develop/feature/039_game_replay_training_data`
**Depends on:** Plan 033 (Bomberman Arena), Plan 034 (Bomber WASM Validator — T2/T3 scaffolding)
**Research:** `.research/17_Fast_BLT_Byte_Level_Transformer.md` (action-level = byte-level for games)

---

## Overview

Generate meaningful training data from the existing Bomberman arena by serializing **tick-level board states** from high-quality games (HL/Validator wins). Feed these replays into `riir-gpu`'s wgpu trainer to produce `game_lora.bin` — a real trained policy adapter for `NNPlayer`.

**The gap:** Plan 034 T2/T3 (`replay.rs`, `train_bomber.rs`) has scaffolding but is blocked on training corpus. The arena produces `RoundTrace` (actions only) but `GameSample` needs `(board_state, action, quality)`. This plan fills that gap.

**Why "smaller burner" (riir-gpu, not riir-burner):**
- 6-action vocab (Up/Down/Left/Right/Bomb/Wait) — not 4000-token BPE
- ~6K parameters total — wgpu handles this in seconds
- No Python needed — pure Rust training pipeline
- Same `output/` directory, same `LoraAdapter` binary format

---

## Data Flow

```
bomber_03_hl_proof.rs (1000 rounds)
  │
  │  At each tick, for each alive player:
  │    serialize(board_state, action_taken, outcome_quality)
  │
  ▼
output/replays/bomber_replay_001.jsonl    (~50K-200K samples)
output/replays/bomber_replay_002.jsonl
  ...
  │
  │  Filter: only winning episodes (score > threshold)
  │  Filter: only HL (P4) and Validator (P3) players
  │
  ▼
riir-gpu/examples/train_bomber.rs         (riir-ai repo)
  │  Loads JSONL → GameSample → wgpu training
  │  3-5 epochs on ~100K samples
  │
  ▼
output/game_lora.bin                       (Secret A — real trained weights)
  │
  │  cp to microgpt-rs or load at runtime
  │
  ▼
NNPlayer → DDTree(marginals, config, bomber_validator.wasm)
```

---

## Replay JSONL Format

Each line is one `(board_state, action, quality)` sample:

```json
{
  "board": [0,0,1,2,0,...],
  "player_pos": [3, 5],
  "player_id": 3,
  "bombs": [[3,5,3,8],[7,2,3,4]],
  "powerups": [[10,10],[5,8]],
  "action": 4,
  "quality": 0.85,
  "tick": 42,
  "round": 7,
  "player_type": "HL"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `board` | `[u8; 169]` | 13×13 grid flattened. 0=Floor, 1=FixedWall, 2=DestructibleWall, 3=PowerUpHidden |
| `player_pos` | `[u8; 2]` | Player position (x, y) |
| `player_id` | `u8` | Player index (0-3) |
| `bombs` | `[[u8; 4]]` | Active bombs: (x, y, blast_range, fuse_ticks) |
| `powerups` | `[[u8; 2]]` | Active powerups: (x, y) |
| `action` | `u8` | Action taken: 0=Up, 1=Down, 2=Left, 3=Right, 4=Bomb, 5=Wait |
| `quality` | `f32` | Outcome quality: 0.0 (death) → 1.0 (win) |
| `tick` | `u32` | Tick number within the round |
| `round` | `u32` | Round number |
| `player_type` | `String` | Player type: "Random", "Greedy", "Validator", "HL" |

**Quality scoring:**
- Death: 0.0
- Survived (no win): 0.5
- Winner: 1.0
- Powerup collected: +0.05 bonus
- Kill scored: +0.1 bonus

---

## Tasks

- [ ] **Task 1: `ReplaySample` type** (`src/pruners/bomber/replay.rs` — NEW)
  - `struct ReplaySample { board: [u8; 169], player_pos: [u8; 2], ... }` matching JSONL format
  - `ReplaySample::from_game_state(grid, pos, bombs, powerups, action, player_id, player_type) -> Self`
  - `ReplaySample::to_json(&self) -> String` — serialize to JSON line
  - `ReplaySample::quality(survived, winner, powerups, kills) -> f32` — compute quality score
  - Unit tests: roundtrip serialization, quality computation

- [ ] **Task 2: `ReplayWriter`** (`src/pruners/bomber/replay.rs`)
  - `struct ReplayWriter { file: BufWriter<File>, round: u32, sample_count: u64 }`
  - `ReplayWriter::create(path: &Path) -> Result<Self>` — open JSONL file
  - `ReplayWriter::write_sample(&mut self, sample: &ReplaySample) -> Result<()>`
  - `ReplayWriter::flush(&mut self) -> Result<()>`
  - `ReplayWriter::sample_count(&self) -> u64`
  - Unit tests: write N samples, read back, verify format

- [ ] **Task 3: Game state serialization** (`src/pruners/bomber/replay.rs`)
  - `fn serialize_board(grid: &ArenaGrid) -> [u8; 169]` — extract grid cells
  - `fn serialize_bombs(world: &World) -> Vec<[u8; 4]>` — extract active bombs from ECS
  - `fn serialize_powerups(world: &World) -> Vec<[u8; 2]>` — extract powerup positions
  - These functions read from the existing ECS world — no game logic changes
  - Unit tests: verify against known grid states

- [ ] **Task 4: Modify arena to dump replays** (`examples/bomber_01_arena.rs`)
  - Add `--replay-dir <path>` CLI argument (default: none = no replay dump)
  - When set, create `ReplayWriter` for each round
  - Inside `run_round()`, after each player's `select_action()`:
    - Capture `(board_state, action_chosen, player_id)`
  - At round end, compute quality from `RoundResult` and backfill
  - Close writer, move to next round
  - Print replay stats at end: total samples, per-player breakdown

- [ ] **Task 5: Modify HL proof to dump filtered replays** (`examples/bomber_03_hl_proof.rs`)
  - Add `--replay-dir output/replays` (default: `output/replays`)
  - Only dump samples from P3 (Validator) and P4 (HL) — these produce quality play
  - Only dump winning episodes (score > threshold) or top-N by score
  - This is the primary data source for training — 1000 rounds, filtered quality

- [ ] **Task 6: Standalone replay generator** (`examples/bomber_04_replay_gen.rs` — NEW)
  - Dedicated example for generating training data
  - Runs 1000 rounds with default 4 players
  - Filters: only dump P3/P4 winning episodes
  - Output: `output/replays/bomber_replay_{timestamp}.jsonl`
  - Prints sample statistics: total, per-action distribution, avg quality
  - ~150 lines

- [ ] **Task 7: Wire `parse_replay()` in riir-ai** (`riir-ai/crates/riir-gpu/src/game/replay.rs`)
  - Replace stub `parse_replay()` with actual JSONL parsing
  - Read JSONL file → deserialize to `GameSample`
  - Map `ReplaySample` → `GameSample` (board, action, quality)
  - Filter by `player_type` and `quality` threshold
  - Unit tests: parse a small JSONL file, verify sample count and content

- [ ] **Task 8: Wire `train_bomber.rs`** (`riir-ai/crates/riir-gpu/examples/train_bomber.rs`)
  - Replace stub with actual training loop
  - Load JSONL from `output/replays/` via `parse_replay()`
  - Convert `GameSample` → training batches for wgpu
  - Train LoRA adapter using `GameConfig` (6 vocab, 32 embd)
  - Save to `output/game_lora.bin`
  - Print training report: loss curve, sample count, epochs
  - ~200 lines

- [ ] **Task 9: End-to-end validation**
  - Run `bomber_04_replay_gen` → produce JSONL
  - Run `train_bomber` → produce `game_lora.bin`
  - Verify `game_lora.bin` loads in `LoraAdapter::load_from_bin()`
  - Verify `NNPlayer` can use it for action selection
  - Compare NNPlayer win rate vs RandomPlayer (should be > 0%)

- [ ] **Task 10: Update docs**
  - Update `microgpt-rs/.docs/10_bomber_arena.md` with replay pipeline
  - Update `riir-ai/.docs/09_training_data_pipeline.md` with game training section
  - Update `riir-ai/.plans/034_bomber_wasm_validator.md` — unblock T3

---

## File Changes

### microgpt-rs (engine — replay generation)

| File | Action | Description |
|------|--------|-------------|
| `src/pruners/bomber/replay.rs` | NEW | `ReplaySample`, `ReplayWriter`, board/bomb/powerup serialization |
| `src/pruners/bomber/mod.rs` | Edit | Add `pub mod replay;` |
| `examples/bomber_01_arena.rs` | Edit | Add `--replay-dir` flag, optional replay dump |
| `examples/bomber_03_hl_proof.rs` | Edit | Add `--replay-dir`, filtered P3/P4 winning episodes |
| `examples/bomber_04_replay_gen.rs` | NEW | Dedicated replay generator, 1000 rounds, quality-filtered |

### riir-ai (secrets — training)

| File | Action | Description |
|------|--------|-------------|
| `crates/riir-gpu/src/game/replay.rs` | Edit | Replace `parse_replay()` stub with real JSONL parser |
| `crates/riir-gpu/examples/train_bomber.rs` | Edit | Replace stub with actual wgpu training loop |

---

## Why riir-gpu (Not riir-burner)

| Aspect | riir-gpu (wgpu) | riir-burner (Python/MLX) |
|--------|-----------------|--------------------------|
| Target model | GameConfig (6 vocab, 6K params) | Gemma 4 (4B params) |
| Training data | Game replays (~100K samples) | Python→Rust corpus (~millions) |
| Training time | Seconds on GPU | Hours on GPU |
| Python needed | No | Yes (unsloth-mlx) |
| Output format | Same `lora.bin` binary | Same `adapter.bin` binary |
| Purpose | Game AI policy | Language translation |

The game model is so small that Python overhead would dominate. wgpu trains it in seconds.

---

## Design Decisions

### 1. Quality-Based Filtering (Not All Games)

Random player data is noise — a random policy trained on random data stays random. Only dump episodes where:
- Player is P3 (Validator) or P4 (HL) — these use strategy
- Episode outcome is positive (survived or won)
- Quality > 0.5 threshold

This produces a dataset of "good play" that the policy learns to imitate.

### 2. Tick-Level (Not Round-Level)

Board state changes every tick. A round has ~200 ticks. Dumping at tick granularity gives ~200K samples per 1000 rounds — enough for a 6K-param model.

### 3. JSONL (Not Binary)

Human-readable, debuggable, works with existing serde. The file sizes are small (~50MB for 200K samples). If performance becomes an issue, we can switch to binary later.

### 4. Board-as-Input (Not Tokens)

The model sees the raw 13×13 grid as 169 "tokens" (one per cell). This maps to BLT's byte-level approach: each cell is a "byte" (0-5 values), the grid is a "byte sequence", and the model learns patterns over it. No BPE needed — the vocabulary is already tiny (6 actions, 6 cell types).

### 5. Backward-Compatible

The replay dump is opt-in (`--replay-dir` flag). Without it, the arena runs exactly as before. No performance impact on the non-replay path.

---

## Expected Outcomes

| Metric | Before (Plan 034) | After (This Plan) |
|--------|-------------------|-------------------|
| Training data | None (blocked) | ~100-200K samples from 1000 rounds |
| `game_lora.bin` | 2404 bytes (random weights) | Real trained policy from game replays |
| NNPlayer win rate | ~25% (random) | Target: >40% (trained on Validator/HL play) |
| Training time | N/A | <30 seconds on Apple Silicon GPU |
| Pipeline | Stub (prints warning) | End-to-end: arena → JSONL → wgpu → .bin |

---

## Success Criteria

- [ ] `bomber_04_replay_gen` produces JSONL with >50K samples from 1000 rounds
- [ ] JSONL contains only P3/P4 winning episodes, quality > 0.5
- [ ] `train_bomber` loads JSONL and trains for 3+ epochs with decreasing loss
- [ ] `output/game_lora.bin` loads successfully via `LoraAdapter::load_from_bin()`
- [ ] NNPlayer with trained adapter wins more rounds than RandomPlayer
- [ ] Arena without `--replay-dir` runs identically to before (zero regression)
- [ ] All existing tests pass (`cargo test --features bomber`)

---

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Dataset too small for meaningful learning | 1000 rounds × 200 ticks × 2 players = 400K samples; enough for 6K params |
| Validator/HL play is too conservative | Add P2 (Greedy) data with lower quality weight; blend aggressive + safe |
| JSONL too large for disk | ~50MB for 200K samples — negligible; add gzip if needed |
| wgpu training doesn't converge on game data | Start with high learning rate (1e-2), monitor loss; fall back to CPU AdamW |
| Board encoding loses information | 169 bytes captures full grid; add bomb/powerup metadata as separate fields |
| Overfitting to specific arena seeds | Random seeds per round, shuffle samples before training |

---

## Cross-Project Coordination

| Project | Plan | Relationship |
|---------|------|-------------|
| `microgpt-rs` | Plan 033 (Bomberman Arena) | ✅ Complete — arena runs games, produces events |
| `microgpt-rs` | Plan 034 (Bomber WASM Validator) | ✅ T1-T10 done — WASM validator ready, NNPlayer ready |
| `microgpt-rs` | Plan 038 (Free Transformer Domain Latent) | ✅ Done — `DomainLatent` type + mid-layer injection; `train_bomber` exports `.dlat` |
| `riir-ai` | Plan 038 (riir-gpu domain_latent) | ✅ Done — `GpuDomainLatent` + `export_domain_latent()` + CPU AdamW fallback |
| `riir-ai` | Plan 034 T2/T3 (replay.rs, train_bomber) | Scaffolding exists — this plan unblocks it |
| `riir-ai` | Plan 026 (AutoTTS) | ✅ Complete — early exit + dynamic budget for game inference |
| `riir-burner` | N/A | Not used — game model too small for Python pipeline |