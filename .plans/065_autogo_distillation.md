# Plan 065: AutoGo Distillation — Go GameState + API Bridge + G-Zero Self-Play

**Branch:** `develop/feature/065_autogo_distillation`
**Depends on:** Plan 056 (GameState Forward Model), Plan 049 (G-Zero Self-Play), Plan 030 (Bandit)
**Research:** `.research/33_autogo_distillation_strategy.md`
**Reference:** `microgpt-rs/.raw/autogo/` (local AutoGo codebase)
**Goal:** Implement Go (9×9) as a `GameState` trait impl, build REST API bridge to play head-to-head against AutoGo's agents, run G-Zero self-play, and benchmark research velocity. Prove our generic, Rust-based, bandit-driven system produces competitive game AI faster than a Go-specific, Python-based pipeline.

---

## Gap Analysis (Refinements from v1)

### G1: Feature Gate Chain Fix
**Problem:** Plan v1 said `go = ["game_state", ...]` but `game_state = ["bomber"]` pulls in `bevy_ecs` + entire bomber arena.
**Fix:** `go` depends on `["bandit"]` only. The `GameState` trait in `src/pruners/game_state/mod.rs` is **always compiled** (no `#[cfg]` on the trait itself). Only the `BomberState` impl is gated behind `game_state = ["bomber"]`. Go provides its own impl.

### G2: API Dual-Move Semantics
**Problem:** AutoGo's `POST /api/game/{id}/move` plays human move AND immediately triggers `_ai_make_move()`. Response contains BOTH moves applied.
**Fix:** Tournament loop: `new_game(color=X)` → read `legal_moves` from response → pick move → `make_move(row, col)` → response already has AI's response → read new `legal_moves` → repeat. **One HTTP call = two Go moves.**

### G3: GTPEngine is the Ground Truth
**Problem:** Plan assumed we validate against `FastGoBoard`. The API uses `GTPEngine` (wrapping GNU Go subprocess via GTP protocol).
**Fix:** Validation compares our `GoState::get_legal_moves()` against API's `legal_moves` field. The API is ground truth regardless of which engine backs it.

### G4: Missing Constructor
**Problem:** `GoState` struct defined but no `GoState::new(size)` constructor specified.
**Fix:** Added T10a task for constructor with neighbor cache pre-computation.

### G5: Missing Property Tests
**Problem:** Unit tests only. No fuzz/property tests comparing our impl against API.
**Fix:** Added T15a — fuzz test: play 100 random games on both sides, compare legal moves at each step.

### G6: Missing Replay Module
**Problem:** Architecture lists `replay.rs` but no task creates it. Bomber has `replay.rs` + `replay_backward.rs`.
**Fix:** Added T16a — `GoReplay` with `MoveRecord` for game recording and playback.

### G7: G-Zero Template Scope Reduction
**Problem:** 9 templates (CornerStar..Joseki) is premature for initial scope.
**Fix:** Start with 4 proven templates: `CornerStar`, `Capture`, `Defend`, `Tenuki`. Expand based on G-Zero results.

### G8: Blocking vs Async HTTP
**Problem:** No discussion of blocking vs async `reqwest`.
**Fix:** Use `reqwest::blocking` — tournament is sequential (one game at a time via API). Async adds complexity with no benefit for sequential play. Parallel games would need multiple game_ids but API state is per-game.

---

## Tasks

### Phase 0: API Bridge — Play Against AutoGo (Integration First)

Before writing any Go rules, validate the head-to-head benchmarking path by spinning up AutoGo and calling its REST API.

- [ ] T0: Plan creation
- [ ] T1: Create `src/pruners/go/` module directory with `mod.rs` (index + re-exports)
- [ ] T2: Add feature gate `go = ["bandit", "dep:reqwest", "dep:serde", "dep:serde_json"]` in `Cargo.toml`
  - **NOT** `game_state` (which pulls in `bevy_ecs` via bomber). The `GameState` trait is always compiled.
  - `bandit` is needed for HL player and G-Zero.
- [ ] T3: Create `src/pruners/go/autogo_client.rs` — REST API client for AutoGo's `play.py` server:
  ```rust
  /// AutoGo REST API client (calls play.py FastAPI server).
  ///
  /// ## API Dual-Move Semantics (G2)
  ///
  /// `make_move()` plays our move AND immediately triggers AutoGo's AI response.
  /// The returned `AutoGoGameState` has BOTH moves applied. This means:
  /// - One HTTP call = two Go moves (ours + theirs)
  /// - We only need to call `make_move()` on our turn
  /// - If we play White, `new_game(color=white)` triggers AI's first move automatically
  pub struct AutoGoClient {
      base_url: String,
      client: reqwest::blocking::Client,
  }

  /// Response from AutoGo's GameState model.
  ///
  /// Field names match the actual API response from `play.py:GameState`.
  /// Note: `human_color` (not `color`) — the API uses "human" to mean "the REST client".
  #[derive(Deserialize)]
  pub struct AutoGoGameState {
      pub game_id: String,
      pub board: Vec<Vec<i8>>,         // 0=empty, 1=black, 2=white
      pub size: usize,
      pub to_play: i8,                 // 1=BLACK, 2=WHITE
      pub last_move: Option<(usize, usize)>,
      pub is_over: bool,
      pub result: Option<String>,      // e.g. "W+2.5"
      pub legal_moves: Vec<(usize, usize)>,
      pub human_color: i8,             // OUR color (1=BLACK, 2=WHITE)
      pub message: String,
  }

  /// Move request matching AutoGo's `MoveRequest` model.
  #[derive(Serialize)]
  struct MoveRequest {
      row: Option<usize>,
      col: Option<usize>,
      pass_move: bool,
  }

  impl AutoGoClient {
      pub fn new(base_url: &str) -> Self;
      pub fn list_agents(&self) -> Result<Vec<String>>;
      pub fn new_game(&self, size: usize, color: &str, agent: &str) -> Result<AutoGoGameState>;
      pub fn get_game(&self, game_id: &str) -> Result<AutoGoGameState>;
      /// Play a stone move. Response includes BOTH our move AND AI's response (G2).
      pub fn make_move(&self, game_id: &str, row: usize, col: usize) -> Result<AutoGoGameState>;
      /// Pass turn. Uses same endpoint with `pass_move=true` (matches API).
      pub fn pass_move(&self, game_id: &str) -> Result<AutoGoGameState>;
  }
  ```
- [ ] T4: Add `reqwest` (blocking) + `serde` + `serde_json` as optional deps behind `go` feature
- [ ] T5: Write integration test `tests/go_integration.rs` — spin up AutoGo Docker, play 10 random games via API, validate game flow (new → move → response → is_over)
- [ ] T6: Write `scripts/autogo_server.sh` — helper script to build + run AutoGo Docker container on port 8000:
  ```bash
  #!/bin/bash
  # Spin up AutoGo container for head-to-head benchmarking
  cd "$(git rev-parse --show-toplevel)/.raw/autogo"
  docker build -f .devcontainer/Dockerfile -t autogo-dev . 2>/dev/null
  docker run -d --name autogo -p 8000:8000 autogo-dev \
    bash -c "uv run -m alpha_go.play --host 0.0.0.0 --port 8000"
  echo "AutoGo server: http://localhost:8000"
  echo "Agents: $(curl -s http://localhost:8000/api/agents | python3 -m json.tool)"
  ```
- [ ] T7: Create `examples/go_00_api_bridge.rs` — play 5 random games against AutoGo's random agent via API, print results
- [ ] T8: Measure API latency: time per full game (new + N moves), estimate max games/sec. Note: one `make_move()` call = two Go moves (G2), so effective moves/sec = 2 × API calls/sec.

### Phase 1: Go GameState (Port from AutoGo)

Port `FastGoBoard` (`go.py`) + `GoBoard` (`go_game.h`) to Rust, implementing our `GameState` trait. Reference both Python and C++ implementations.

- [ ] T9: Define `GoAction` enum and `GoCell` enum in `src/pruners/go/types.rs`:
  ```rust
  #[derive(Clone, Debug, PartialEq)]
  pub enum GoAction {
      Place(usize, usize),  // (row, col)
      Pass,
  }

  #[derive(Clone, Copy, Debug, PartialEq, Eq)]
  #[repr(i8)]
  pub enum GoCell {
      Empty = 0,
      Black = 1,
      White = 2,
  }

  impl GoCell {
      /// Opposite color. Panics on Empty.
      pub fn opponent(self) -> Self {
          match self {
              Self::Black => Self::White,
              Self::White => Self::Black,
              Self::Empty => panic!("GoCell::opponent() called on Empty"),
          }
      }
  }
  ```
- [ ] T10: Define `GoState` snapshot in `src/pruners/go/state.rs` — port from `FastGoBoard` (Python) with C++ optimizations:
  ```rust
  /// Lightweight Go state snapshot. Port from go.py:FastGoBoard + go_game.h:GoBoard.
  ///
  /// 9×9: 81 cells ≈ 500 bytes total. Clone < 100ns.
  /// 19×19: 361 cells ≈ 1.8KB total. Clone < 500ns.
  #[derive(Clone)]
  pub struct GoState {
      pub board: Vec<GoCell>,           // flat array: row * size + col
      pub size: usize,                  // 9 or 19
      pub to_play: GoCell,              // Black or White
      pub ko_point: Option<usize>,      // flat index of forbidden recapture (simple ko)
      pub consecutive_passes: u8,       // two passes = game end
      pub move_count: u32,
      pub komi: f32,                    // 7.5 (AI standard, matches go_game.h KOMI)
      pub captured_black: u32,          // stones black captured from white
      pub captured_white: u32,          // stones white captured from black
      // Pre-computed neighbor offsets (rebuilt on construction)
      neighbor_offsets: Vec<Vec<usize>>,
  }
  ```
- [ ] T10a: Implement `GoState::new(size)` constructor:
  - Initialize empty board `vec![GoCell::Empty; size * size]`
  - Pre-compute neighbor offsets for each position (skip edges)
  - Set `to_play = GoCell::Black`, `komi = 7.5`
  - Port from `FastGoBoard.__init__` neighbor cache pattern
- [ ] T11: Implement Go core logic (port from `go.py:FastGoBoard`):
  - `get_neighbors(idx)` — cached neighbor lookup (port from `_neighbor_cache`)
  - `get_group_and_liberties(idx)` — BFS flood fill (port from `_get_group_and_liberties`)
  - `would_be_suicide(idx, color)` — temp placement + liberty + capture check (port from `_would_be_suicide`)
  - `is_legal(row, col)` — occupied + ko + suicide check (port from `is_legal`)
  - `play(row, col)` — stone placement + capture resolution + ko update (port from `play`)
  - `pass()` — increment consecutive_passes, switch player
  - `get_legal_moves()` — iterate all cells, filter by `is_legal`
  - `resolve_captures(idx, color)` — remove opponent groups with zero liberties, track captures
- [ ] T12: Implement Tromp-Taylor scoring (port from `go.py:score()` + `go_game.h:score()`):
  - `score()` → f32 — black_score - white_score (with komi)
  - `flood_territory(idx)` — BFS empty region, determine ownership by border colors
  - `get_winner()` — `GoCell::Black` if score > 0, `GoCell::White` if score < 0
- [ ] T13: Implement `GameState` trait for `GoState`:
  ```rust
  impl GameState for GoState {
      type Action = GoAction;
      fn available_actions(&self, _player_id: u8) -> Vec<GoAction>;
      fn advance(&self, action: &GoAction, _player_id: u8) -> Self;
      fn is_terminal(&self) -> bool;  // consecutive_passes >= 2
      fn reward(&self, player_id: u8) -> f32;  // 1.0 win, 0.5 draw, 0.0 loss
      fn tick(&self) -> u32;  // move_count
  }
  ```
- [ ] T14: Implement `GoHeuristic` (`StateHeuristic<GoState>`):
  ```rust
  pub struct GoHeuristic;

  impl StateHeuristic<GoState> for GoHeuristic {
      fn evaluate(&self, state: &GoState, player_id: u8) -> f32 {
          let color = if player_id == 0 { GoCell::Black } else { GoCell::White };
          // Weighted sum of features
          let liberty_score = self.liberty_advantage(state, color);
          let capture_score = self.capture_delta(state, color);
          let influence_score = self.influence(state, color);
          let center_score = self.center_preference(state, color);
          liberty_score * 0.4 + capture_score * 0.3 + influence_score * 0.2 + center_score * 0.1
      }
  }
  ```
- [ ] T15: Write unit tests (port from `tests/test_go.py` + `tests/test_cpp_go.py`):
  - Capture: single stone capture, group capture, snapback
  - Ko: simple ko violation blocked, ko threat allowed after other move
  - Suicide: suicide blocked, capture-not-suicide allowed
  - Scoring: Tromp-Taylor matches known positions (simple territory, seki-ish, capture-heavy)
  - Terminal: two passes end game, pass + move + pass does NOT end game
  - `advance()`: produces valid successor, immutable self
- [ ] T15a: Write property-based fuzz tests (G5):
  - Generate random game: `GoState::new(9)` → random legal moves until terminal
  - Assert invariants at every step: `available_actions` only contains legal moves, `advance` produces valid board (stone counts consistent), no panics
  - Run 200 random games, 0 panics = pass
  - **API cross-validation**: If AutoGo Docker is running, play same random game via API, compare `legal_moves` at each step. Skip silently if Docker unavailable.
- [ ] T16: Create `examples/go_01_mcts.rs` — MCTS player vs Random on 9×9, 100 games, print win rates
- [ ] T16a: Create `src/pruners/go/replay.rs` — game recording and playback (G6):
  ```rust
  /// Single move record for replay.
  #[derive(Clone, Debug, Serialize, Deserialize)]
  pub struct MoveRecord {
      pub action: GoAction,
      pub player: GoCell,
      pub move_number: u32,
      pub legal_move_count: usize,  // branching factor at this point
  }

  /// Complete game replay.
  #[derive(Clone, Debug, Serialize, Deserialize)]
  pub struct GoReplay {
      pub size: usize,
      pub komi: f32,
      pub moves: Vec<MoveRecord>,
      pub winner: Option<GoCell>,
      pub final_score: f32,
  }

  impl GoReplay {
      pub fn new(size: usize, komi: f32) -> Self;
      pub fn record(&mut self, action: &GoAction, player: GoCell, legal_count: usize);
      pub fn finalize(&mut self, winner: Option<GoCell>, score: f32);
      /// Replay all moves, returning final state. Validates all moves are legal.
      pub fn replay(&self) -> Result<GoState>;
  }
  ```

### Phase 2: Go Player Strategies (Prove HL Thesis on Go)

- [ ] T17: Define `GoPlayer` trait in `src/pruners/go/players.rs` (adapted from `agents/base.py:Agent`):
  ```rust
  /// Go player strategy trait. Matches AutoGo's `agents/base.py:Agent` pattern.
  pub trait GoPlayer {
      fn select_move(&mut self, state: &GoState, legal_moves: &[(usize, usize)], rng: &mut impl Rng) -> GoAction;
      fn name(&self) -> &'static str;
  }
  ```
- [ ] T18: Implement `GoRandomPlayer` — random legal move (port from `agents/random.py`)
- [ ] T19: Implement `GoGreedyPlayer` — maximize immediate captures + liberty advantage
- [ ] T20: Implement `GoValidatorPlayer` — safety rules: no self-atari large groups, maintain 2+ eye potential, avoid filling own territory
- [ ] T21: Implement `GoHLPlayer` — bandit-driven with Go-specific features:
  - Opening moves bandit (corner star points → side → center)
  - Capture/defend heuristic scoring
  - Influence map-based territory estimation
  - Endgame pass timing (pass when all territory secured)
- [ ] T22: Implement `GoGZeroPlayer` — template proposer + delta bandit (adapt from `pruners/g_zero/`)
- [ ] T23: Implement `GoMctsPlayer` — wraps `mcts_search::<GoState>()` with configurable budget
- [ ] T24: Create `examples/go_02_tournament.rs` — all player types, configurable board size, 100 games, print win rates

### Phase 3: Head-to-Head Tournament via API (Prove Against AutoGo)

- [ ] T25: Create `src/pruners/go/tournament.rs` — tournament runner using `AutoGoClient`:
  ```rust
  pub struct GoTournamentConfig {
      pub board_size: usize,           // 9
      pub num_games: usize,            // 100+
      pub our_player: GoPlayerType,    // Random, Greedy, Validator, HL, GZero, MCTS
      pub their_agent: String,         // "random", "gnugo1"
      pub autogo_url: String,          // "http://localhost:8000"
      pub play_both_colors: bool,      // true = each player plays Black and White
  }

  pub struct GoTournamentResult {
      pub our_wins: usize,
      pub their_wins: usize,
      pub draws: usize,
      pub avg_score_delta: f32,
      pub games_per_sec: f32,
      pub total_moves: usize,
  }
  ```
- [ ] T26: Implement `run_tournament()` — plays N games via API, records results:
  - For each game: `new_game()` → our player picks from `legal_moves` → `make_move()` → **response has AI move baked in (G2)** → read new `legal_moves` → repeat until `is_over`
  - Swap colors each game if `play_both_colors`
  - Track per-move timing, game outcome, score delta
  - **Note:** API returns `result` as string like "W+2.5" — parse to determine winner
- [ ] T27: Implement `AutoGoProxyPlayer` — adapter that lets our tournament runner also play AS AutoGo's agent (for control experiments):
  ```rust
  /// Wraps AutoGo's REST API as a GoPlayer, so we can run pure-AutoGo games for baseline.
  pub struct AutoGoProxyPlayer<'a> {
      client: &'a AutoGoClient,
      game_id: String,
  }
  ```
- [ ] T28: Run baseline tournaments:
  - Our Random vs AutoGo Random (sanity: should be ~50/50)
  - Our Random vs AutoGo `gnugo1` (expect: we lose badly)
  - Our Greedy vs AutoGo `gnugo1` (expect: competitive)
- [ ] T29: Run HL tournaments:
  - Our HL vs AutoGo Random (expect: >70% win rate)
  - Our HL vs AutoGo `gnugo1` (target: >55% win rate)
- [ ] T30: Run G-Zero tournament:
  - Our GZero vs AutoGo `gnugo1` (stretch: >55% win rate)
- [ ] T31: Create `examples/go_03_head_to_head.rs` — run tournament against AutoGo, print results table

### Phase 4: Go G-Zero Self-Play (Prove Transfer)

- [ ] T32: Create `src/pruners/go/g_zero_player.rs` — full G-Zero self-play for Go
- [ ] T33: Implement `GoTemplateProposer` with **4 initial templates** (G7 — reduced from 9):
  ```rust
  /// Go strategy templates for G-Zero self-play.
  /// Start with 4 proven patterns, expand based on δ signal results.
  pub enum GoTemplate {
      CornerStar,     // Play on star point (4-4, 3-3) — strongest opening heuristic
      Capture,        // Atari group or capture stones — tactical reading
      Defend,         // Connect cutting point, make eye shape — defensive safety
      Tenuki,         // Play away (center or another corner) — strategic flexibility
  }
  ```
  Future expansion (post-plan): `SideApproach`, `Invasion`, `Attachment`, `Peep`, `Joseki`
- [ ] T34: Implement Go-specific HintDelta computation:
  - Query: board state as flat token array (0=empty, 1=black, 2=white)
  - Hint: template-suggested move + rationale string
  - δ = log-prob shift between hinted and unhinted generation
- [ ] T35: Implement `DeltaGatedAbsorbCompress` for Go — promote high-δ templates to hard constraints
- [ ] T36: Run G-Zero self-play: 500 episodes, 9×9, track δ evolution, template exploration, win rates over time
- [ ] T37: Create `examples/go_04_gzero.rs` — G-Zero self-play with per-round metrics + δ tracking

### Phase 5: AutoResearch Loop (Prove Velocity)

- [ ] T38: Create `src/pruners/go/autoresearch.rs` — automated hyperparameter search over Go
- [ ] T39: Define `GoResearchConfig`:
  ```rust
  pub struct GoResearchConfig {
      pub board_size: usize,            // 9
      pub mcts_budget: usize,           // 100-10000
      pub rollout_depth: usize,         // 5-50
      pub exploration_constant: f32,    // UCB1 C param (0.5-2.0)
      pub bandit_epsilon: f32,          // ε-greedy (0.05-0.5)
      pub template_count: usize,        // number of active templates (2-4, G7)
      pub heuristic_weights: [f32; 4],  // liberty, capture, influence, center
  }
  ```
- [ ] T40: Implement `AutoResearchLoop`:
  - Bandit over `GoResearchConfig` arms
  - Evaluate each arm: run N games against baseline (AutoGo Random), compute win rate
  - `TrialLog` records per-arm results
  - UCB1 selects next arm
  - Early stopping: drop arms below 25th percentile after 10 evaluations
- [ ] T41: Run AutoResearch: 30 arms × 50 games each, find config that maximizes win rate vs AutoGo `gnugo1`
- [ ] T42: Create `examples/go_05_autoresearch.rs` — automated research loop with progress reporting

### Phase 6: Benchmarks & Documentation

- [ ] T43: Add `bench_go_state()` — GoState::advance() ops/sec for 9×9 and 19×19
- [ ] T44: Add `bench_go_mcts()` — mcts_search::<GoState>() actions/sec with varying budgets
- [ ] T45: Add `bench_go_api()` — games/sec through AutoGo REST API bridge
- [ ] T46: Record scaling law data: win rate vs episodes for each player type (CSV timeseries)
- [ ] T47: Update `README.md` with Go section (GameState impl, tournament results, API bridge)
- [ ] T48: Update `.docs/01_overview.md` with Go module in feature flags and module structure
- [ ] T49: Update `.research/33_autogo_distillation_strategy.md` with actual benchmark results
- [ ] T50: Run `cargo clippy --fix --allow-dirty`, fix warnings
- [ ] T51: Commit with message `feat(go): gamestate trait impl + api bridge + head-to-head tournament`

---

## Architecture

```text
src/pruners/go/
├── mod.rs              — Module index + re-exports + GoPlayer trait
├── types.rs            — GoAction, GoCell enums (with GoCell::opponent())
├── state.rs            — GoState::new() + snapshot + GameState impl + Tromp-Taylor scoring
                          (port from .raw/autogo/src/alpha_go/go.py:FastGoBoard
                           + .raw/autogo/src/alpha_go/cpp/go/go_game.h:GoBoard)
├── heuristic.rs        — GoHeuristic (liberty + influence + capture + center/edge)
├── players.rs          — GoRandomPlayer, GoGreedyPlayer, GoValidatorPlayer,
                          GoHLPlayer, GoMctsPlayer, GoPlayer trait
├── g_zero_player.rs    — GoGZeroPlayer — template proposer + delta bandit
├── templates.rs        — GoTemplate (4 initial strategies), GoTemplateProposer
├── replay.rs           — GoReplay, MoveRecord — game recording and playback
├── autogo_client.rs    — AutoGoClient — REST API bridge to play.py FastAPI server
├── tournament.rs       — GoTournamentConfig, run_tournament(), GoTournamentResult
├── autoresearch.rs     — AutoResearchLoop — bandit over GoResearchConfig arms

examples/
├── go_00_api_bridge.rs     — Play random games against AutoGo via API
├── go_01_mcts.rs           — MCTS vs Random on Go
├── go_02_tournament.rs     — All player types tournament
├── go_03_head_to_head.rs   — Our players vs AutoGo agents via API
├── go_04_gzero.rs          — G-Zero self-play with metrics
└── go_05_autoresearch.rs   — Automated research loop

tests/
└── go_integration.rs       — Integration tests for Go GameState + API bridge

scripts/
└── autogo_server.sh        — Spin up AutoGo Docker container for benchmarking
```

---

## Key Design Decisions

### 1. API Bridge First (Phase 0 Before Phase 1)

Instead of implementing Go rules and hoping we can benchmark later, we start by proving the benchmarking infrastructure works. The API bridge lets us:
- Validate against AutoGo's GTPEngine (ground truth via GNU Go subprocess)
- Measure real win rates against real agents
- Run control experiments before writing any Go AI

### 2. Simple Ko, Not Superko

The C++ `GoBoard` implements full positional superko (Zobrist hashing). The Python `FastGoBoard` uses simple ko. For modelless play, simple ko is sufficient. We start with simple ko (matches Python impl), add superko later if needed for NN training.

### 3. Flat Array Board (Match C++ Layout)

Using `Vec<GoCell>` with flat indexing (`row * size + col`) matches the C++ `go_game.h` layout for cache efficiency. Pre-computed neighbor offsets avoid per-access computation.

### 4. Komi = 7.5 (AI Standard)

The C++ `GoBoard::KOMI = 7.5f` (comment: "7.5 is recommended for high level / AI"). The Python `FastGoBoard` uses 6.5 (via `alpha_go_cpp.GoBoard.KOMI`). We use 7.5 to match the C++ reference and AI convention.

### 5. Feature Gate `go` (G1 Fix)

```toml
[features]
go = ["bandit", "dep:reqwest", "dep:serde", "dep:serde_json"]
```

The `GameState` trait in `src/pruners/game_state/mod.rs` is **always compiled** — it's not behind `#[cfg(feature = "game_state")]`. Only the `BomberState` impl is gated. Go provides its own `GoState` impl behind `#[cfg(feature = "go")]`. Zero impact on existing features. Does NOT pull in `bevy_ecs`.

### 6. 9×9 First, 19×19 Stretch

9×9 branching factor ~80 (manageable for MCTS). 19×19 branching ~360 (very expensive). Prove the system on 9×9 first.

### 7. Blocking HTTP Client (G8)

Using `reqwest::blocking::Client` because:
- Tournament is sequential (one game at a time)
- API has per-game state (can't parallelize within a game)
- Blocking is simpler, no async runtime needed
- If we need parallelism later: multiple game_ids, each with own blocking client in a thread

---

## Expected Outcomes

### Success Criteria

| Criterion | Target | Measurement |
|-----------|--------|-------------|
| API bridge works | Play 10 games via REST | `go_00_api_bridge` example passes |
| GoState compiles with GameState trait | ✅ | `mcts_search::<GoState>()` returns valid `GoAction` |
| Go-specific logic correct | All unit tests pass | Captures, ko, suicide, scoring |
| Property fuzz tests pass | 200 random games, 0 panics | Invariant checks at every step |
| Validate against AutoGo API | Same legal moves + score | Play same moves, compare results |
| Our Random vs AutoGo Random | ~50% win rate | 100 games (sanity check) |
| Our HL vs AutoGo Random | >70% win rate | 100 games |
| Our HL vs AutoGo `gnugo1` | >55% win rate (stretch) | 100 games |
| GoState::advance() throughput | >50K ops/sec (9×9) | Benchmark |
| API bridge throughput | >5 games/sec | Benchmark (limited by HTTP overhead) |

### Tournament Predictions (9×9, 100 games, internal)

| Player | Predicted Win% | Rationale |
|--------|---------------|-----------|
| Random | ~10% | Baseline |
| Greedy | ~15% | Better but no reading |
| Validator | ~20% | Safety prevents blunders |
| MCTS (budget=1000) | ~25% | STRATEGA finding: generic search ≈ random without heuristics |
| HL (Bandit) | ~40% | Proven pattern from Bomber/FFT |
| GZero | ~50% | δ signal + template exploration > static bandit |

### Head-to-Head Predictions (vs AutoGo agents)

| Matchup | Prediction | Rationale |
|---------|-----------|-----------|
| Our Random vs their Random | ~50% | Both uniform random |
| Our Greedy vs their `gnugo1` | ~30% | GNU Go has stronger heuristics |
| Our HL vs their `gnugo1` | ~45% | Bandit may close the gap |
| Our GZero vs their `gnugo1` | ~50% (stretch) | Self-improvement could match |

---

## Reference Code Map

| What | AutoGo Source | Our Target |
|------|--------------|------------|
| Board rules | `.raw/autogo/src/alpha_go/go.py:FastGoBoard` | `state.rs` |
| C++ board | `.raw/autogo/src/alpha_go/cpp/go/go_game.h` | `state.rs` (layout reference) |
| MCTS state | `.raw/autogo/src/alpha_go/go.py:GoState` | `state.rs` (GameState impl) |
| Scoring | `.raw/autogo/src/alpha_go/go.py:FastGoBoard::score()` | `state.rs` |
| Capture logic | `.raw/autogo/src/alpha_go/go.py:FastGoBoard::play()` | `state.rs` |
| Ko rule | `.raw/autogo/src/alpha_go/cpp/go/go_game.h` (simple ko in Python, superko in C++) | `state.rs` |
| Agent base | `.raw/autogo/src/alpha_go/agents/base.py:Agent` | `players.rs` |
| Random agent | `.raw/autogo/src/alpha_go/agents/random.py` | `players.rs` |
| REST API | `.raw/autogo/src/alpha_go/play.py` | `autogo_client.rs` |
| GameState model | `.raw/autogo/src/alpha_go/play.py:GameState` | `autogo_client.rs` (note: `human_color` field) |
| MoveRequest model | `.raw/autogo/src/alpha_go/play.py:MoveRequest` | `autogo_client.rs` (note: `pass_move: bool` field) |
| Self-play | `.raw/autogo/src/alpha_go/self_play.py` | `g_zero_player.rs` |
| MCTS | `.raw/autogo/src/alpha_go/mcts.py` + `.raw/autogo/src/alpha_go/cpp/mcts/mcts.h` | `game_state/mcts.rs` (existing) |
| Model | `.raw/autogo/src/alpha_go/model.py` | Future plan (not this one) |
| Dataset | `.raw/autogo/src/alpha_go/dataset.py` | Future plan |
| Game replay → LoRA | `riir-gpu/src/game/trainer.rs` (Bomberman) | Future: Go token adapter (82-token seq) |
| Game policy config | `riir-gpu/src/game/policy.rs` | Future: Go vocab (82 actions) |
| GRPO loss | `riir-gpu/src/loss_grpo.rs` | Future: G-Zero Proposer training |
| DPO loss (GPU) | `riir-gpu/src/loss_dpo.rs` | Future: G-Zero Generator training |
| GZeroLoop | `riir-gpu/src/gzero_loop.rs` | Future: Model-based Go self-play |
| Fourier MCTS | `riir-engine/src/fourier/mcts.rs` | Future: Go spatial hash transposition |
| WASM Validator | `riir-validator-sdk/` + `riir-wasm/` | Future: `go_validator.wasm` |

---

## Relationship to Existing Work

| Plan | Relationship |
|------|-------------|
| Plan 056 (GameState) | Provides the `GameState` trait we implement for Go |
| Plan 049 (G-Zero) | Provides `DeltaBanditPruner`, `TemplateProposer`, HintDelta |
| Plan 030 (Bandit) | Provides `BanditPruner`, `TrialLog` for AutoResearch |
| Plan 033 (Bomber Arena) | First `GameState` impl — pattern to follow |
| Plan 053 (FFT Arena) | Second `GameState` impl — G-Zero transfer proven |
| Plan 067 (FFO Schur) | ✅ COMPLETE — Schur 1-shot exact solve for domain latent (100% lower loss vs AdamW). If we add policy/value network, `DomainLatentSchurAccumulator` ready for use |

---

## Risks

### 1. Docker/API Setup Complexity
AutoGo's Docker build requires C++ extension compilation (`libpython3.10.so`, cmake, pybind11). On macOS this may fail.

**Mitigation:** Fallback to running Python directly (`uv run -m alpha_go.play`). Or skip API bridge and benchmark internally only. The API bridge is nice-to-have, not blocking.

### 2. GNU Go Too Strong for Modelless
GNU Go level 1 may still be too strong for our heuristic players, making head-to-head results discouraging.

**Mitigation:** Start with AutoGo's `random` agent as baseline. GNU Go results are stretch goals. We measure research velocity (improvement rate), not absolute ELO.

### 3. Go Complexity Underestimated
Go has subtler tactics than Bomberman (ladders, nets, snapbacks, life/death). Our heuristic may not capture enough for meaningful play.

**Mitigation:** Start with 9×9 (simpler tactics). Accept that modelless Go is weak — the point is proving the infrastructure and measuring improvement velocity.

### 4. API Latency Too High
HTTP overhead may make tournament runs too slow for meaningful sample sizes.

**Mitigation:** Profile latency. If >500ms/game, run games in parallel (multiple game_ids). Internal benchmarks (no API) have no latency issue. Also: one API call = two Go moves (G2), effectively halving the round-trip count.

### 5. Scope Creep
Go implementation could balloon into a multi-month project.

**Mitigation:** Strict scope: 9×9 only. No neural networks. Simple ko only. Feature gate ensures zero impact. API bridge is Phase 0 — if it doesn't work, we skip and benchmark internally.

### 6. Feature Gate Pulls in bevy_ecs (G1)
If we accidentally depend on `game_state` feature, we pull in bomber + bevy_ecs.

**Mitigation:** `go = ["bandit"]` only. The `GameState` trait is always available. Verify with `cargo tree -f "{f}" --features go | grep bevy` — must be empty.

---

## Honest Assessment

### What This Plan Delivers

1. ✅ REST API bridge to play against AutoGo's agents head-to-head
2. ✅ Another `GameState` implementation proving the trait's generality on Go
3. ✅ Go-specific MCTS benchmark (comparable to AutoGo's approach)
4. ✅ G-Zero self-play on a fundamentally different game genre (territory, not combat)
5. ✅ AutoResearch loop proving automated hyperparameter search
6. ✅ Head-to-head tournament results: objective, reproducible benchmark numbers
7. ✅ Property-based fuzz tests comparing our impl against AutoGo API (G5)
8. ✅ Replay module for game recording and playback (G6)

### What This Plan Does NOT Deliver

1. ❌ Professional-strength Go AI
2. ❌ Policy/value neural network training
3. ❌ 19×19 board support (stretch only)
4. ❌ Positional superko
5. ❌ Distributed training across GPU fleet
6. ❌ More than 4 G-Zero templates (G7 — start small, expand based on results)

### Verdict

This plan is **research infrastructure + benchmarking**, not a Go product. The value is in proving:
- `GameState` trait works across 4 game genres (Bomber, FFT, Monopoly, Go)
- G-Zero transfers to territory games (not just combat games)
- Bandit-driven research automation matches LLM-driven approaches
- Rust provides iteration speed advantage over Python
- Head-to-head benchmarking via REST API is practical and reproducible

If Go results are weak, we still have 3 proven games + the API bridge infrastructure. If Go results are strong, we have a compelling benchmark paper with objective numbers from playing against AutoGo's own agents.

---

## Future Directions: Model-Based Go via `riir-gpu` (Feature-Gated)

Plan 065 implements `go` feature gate (GoState + MCTS + HL + API bridge). Each subsequent feature gate is unlocked by proving the previous one works via benchmarks. **No speculation — just run experiments.**

### 1. Feature-Gate Strategy

```toml
[features]
go = ["bandit"]                           # Phase 1: GoState + MCTS + HL (THIS PLAN)
go-training = ["go", "riir-gpu/training"] # Phase 2: LoRA training + GZeroLoop
go-wasm = ["go", "riir-validator-sdk"]    # Phase 2: go_validator.wasm
go-fourier = ["go", "riir-engine/fourier"] # Phase 2: Fourier spatial MCTS
go-mtp = ["go", "riir-router"]            # Phase 3: MTP projections for Go vocab
go-full = ["go-training", "go-wasm", "go-fourier", "go-mtp"]  # Everything
```

Same pattern as bomber (`bomber`, `bomber-agent`, `bomber-wasm`). New domain, same gates.

### 2. Proven Results (Not Theory)

These demos are running code with measured results — the pipeline works:

1. **`bandit_with_real_model_demo.rs`** — Loads real `rust_validator.wasm`, real `py2rs_lora.bin` (trained by riir-burner), runs real LeviathanVerifier p/q rejection sampling. Full pipeline: Draft → DDTree + BanditPruner\<WasmPruner\> → LeviathanVerifier → bandit.update(). **Swap validator + vocab = Go.**

2. **`bomber_tech_ab_demo.rs`** — 1000-round A/B: LoRA-only vs WASM-only vs LoRA+WASM vs Full HL (LoRA+WASM+Bandit+AbsorbCompress). Combined wins. **No component conflict.**

3. **`g_zero_04_player_ab_benchmark.rs`** — Isolated perf benchmark, 5 configs × 1000 rounds. Measures survival rate, score, kills, per-action latency. **Rust players are fast.**

4. **`g_zero_fft_01_arena.rs`** through `g_zero_fft_06_tft_benchmark.rs` — 6 FFT demos proving G-Zero transfers across game genres. **Go is the next transfer, not a leap.**

### 3. Board State → Token Encoding

`game/trainer.rs` encodes Bomberman into 170-token sequences. Same pipeline for Go:

| Component | Bomber (existing) | Go (adapt) |
|-----------|-------------------|------------|
| Board vocab | 4 tokens (Floor, Wall, Destructible, PowerUp) | 3 tokens (Empty, Black, White) |
| Action vocab | 6 tokens (Up/Down/Left/Right/Bomb/Wait) | 82 tokens (81 placements + Pass) |
| Sequence length | 170 (169 board + 1 action) | 82 (81 board + 1 action) |
| Board size | 13×13 = 169 | 9×9 = 81 |

### 4. Training Pipeline (Existing → Adapt, `go-training` gate)

```text
GoState::play_random_game()           (Plan 065 GoState)
  → GoReplay                          (Plan 065 T16a)
  → game/replay.rs → samples          (adapt GameAction enum)
  → game/trainer.rs → 82-token seq    (adapt encode_game_sample)
  → riir-gpu LoRA fine-tuning         (existing wgpu kernels)
  → Go LoRA adapter (.bin)            (tiny: ~4K params)
  → GoLoRAPlayer                      (new player type)
```

### 5. G-Zero Model-Based Loop (`go-training` gate)

GZeroLoop activates when Go replays + tokenization exist:

```text
GZeroLoop round (riir-gpu/src/gzero_loop.rs):
  1. GoTemplateProposer → query-hint pairs    (Plan 065 T33)
  2. Go LoRA Generator → move predictions     (adapted game/policy.rs)
  3. HintDelta → intrinsic reward             (log-prob shift)
  4. DeltaFilter → preference pairs           (6-stage, game-agnostic)
  5. GRPO → train Proposer                    (loss_grpo.rs)
  6. DPO → train Generator                    (loss_dpo.rs, WGSL kernels)
```

### 6. Fourier Spatial MCTS (`go-fourier` gate)

`riir-engine/src/fourier/mcts.rs` — position-invariant transposition table. For Go, implement `state_to_entities` callback:
- Stone positions relative to groups
- Liberty counts as spatial features
- Ko point as special entity

Same encoder, different features. Transposition recognizes identical tactical shapes across board locations.

### 7. WASM Go Validator (`go-wasm` gate)

`bomber_validator.wasm` exists and works. Same pattern for `go_validator.wasm`:
- Compile `GoState::is_legal` → WASM via `riir-validator-sdk`
- Load via `WasmPruner` → `relevance = 0.0` for illegal moves
- DDTree never explores invalid branches

### 8. MTP Projections (`go-mtp` gate, document for future)

`MtpProjectionCache` loads `.bin` files for Multi-Token Prediction. For Go:
- Train MTP projections on Go token sequences
- Inject into `TransformerWeights` at load time
- Dynamic compute scaling for critical mid-game fights

### 9. Research Questions (Answered by Benchmarks)

| Question | How to Answer | Feature Gate |
|----------|--------------|-------------|
| Does GoState produce correct legal moves? | Fuzz vs AutoGo API (T15a) | `go` |
| Does Fourier spatial hash help Go MCTS? | Benchmark: Fourier vs vanilla MCTS on Go | `go-fourier` |
| Does Go LoRA training converge? | Train on replay, measure loss curve | `go-training` |
| Does GZeroLoop improve Go win rate? | Self-play tournament, rounds vs win rate | `go-training` |
| Does `go_validator.wasm` prevent illegal moves? | A/B: LoRA vs LoRA+WASM on Go | `go-wasm` |
| Does MTP improve Go token prediction? | Benchmark: with/without MTP | `go-mtp` |
| Does our stack beat AutoGo? | Head-to-head via API (T25-T31) | `go` |

**Every question answered by a benchmark. No speculation.**