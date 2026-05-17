# Research 33: AutoGo Distillation Strategy

**Source:** https://github.com/ericjang/autogo (Eric Jang, 2026)
**Tutorial:** https://evjang.com/2026/04/28/autogo.html
**Local code:** `microgpt-rs/.raw/autogo/`
**Date:** 2026-05-17
**Status:** Research → Plan 065

---

## 1. What AutoGo Is

AutoGo is a minimal codebase for building a Go-playing AI from scratch. Its **real goal** is studying how to automate the AI researcher — Go is just the testbed. The same workflow should transfer to any AI research domain.

### Why Go? (from the author)

1. **Perplexity minimization** — Policy/value networks are fundamentally language-model problems. Simple training methods + system engineering scale.
2. **Scaling laws** — Go data is cheap, universe is huge. Good for studying train-time and test-time scaling.
3. **Technique transfer** — Faster Go training → faster LLM training. De-correlated signal from LLM applications.
4. **Robotics analog** — Logging, data collection, replay buffers, distributed RL, simulated evaluation — but orders of magnitude faster.
5. **Value approximation** — Querying a function approximator for value replaces simulation. Profound.
6. **Self-play dynamics** — Nash equilibria, mixed strategies, recursive self-improvement.

### Architecture

```
Code Editing Client (Laptop/VSCode)
        │ SSH
Dev Container (CPU + /nfs)
        │ Multi-host cluster (SSH + docker run --rm per GPU)
        ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ RTX 6000 Ada │  │ RTX 6000 Ada │  │ RTX PRO 6000B│
│ train+collect│  │ collect-only │  │ collect-only │
│ /nfs (host)  │  │ no /nfs      │  │ no /nfs      │
└──────────────┘  └──────────────┘  └──────────────┘
```

- Driver dispatches jobs to GPU fleet over SSH + Docker
- One worker shares NFS (checkpoints + game data visible)
- Other workers rsync push/pull
- `autoresearch`: autonomously optimize a metric (hyperparameter tuning, perf)
- `experiment`: one-off analysis

### Key Technical Decisions

| Decision | Rationale |
|----------|-----------|
| Python + C++ pybind11 | Fast game logic in C++, training in Python |
| Docker + SSH | Agent-friendly, no framework overhead |
| Synchronous train/collect first | Easier to catch instability |
| Tromp-Taylor scoring | Simpler than Chinese/Japanese rules |
| Claude as researcher | Automates experiment iteration |

---

## 2. Actual Code Analysis (from `.raw/autogo/`)

### 2.1 Code Structure

```
src/alpha_go/
├── __init__.py
├── __main__.py
├── engine.py          — GTPEngine: GNU Go subprocess wrapper (GTP protocol)
├── go.py              — FastGoBoard (numpy), GoState (MCTS-compatible)
├── mcts.py            — Python MCTS: PUCT, Dirichlet noise, rollouts
├── model.py           — GoTransformer (ViT) + SizeInvariantGoResNet (CNN)
├── dataset.py         — GoDataset: NPZ loading, board flipping, MCTS policy targets
├── gameplay.py        — play_game(), GameRecord, save_game_data()
├── play.py            — FastAPI web server (port 8000) — THE KEY FOR INTEGRATION
├── self_play.py       — run_sequential(), run_parallel(), main()
├── agents/
│   ├── __init__.py    — Agent registry (get_agent, list_agents, register_agent)
│   ├── base.py        — Agent ABC: select_move(board, seed), start_game(), notify_move()
│   ├── random.py      — RandomAgent
│   ├── nn_agent.py    — NNAgent: local checkpoint / gRPC remote / random init
│   └── nn_mcts.py     — NNMCTSAgent, CppMCTSAgent, evaluators (local/batched/RPC)
├── inference/
│   └── batched_engine.py — LocalBatchedInferenceEngine (GPU batching)
├── analysis/          — (empty or analysis tools)
├── cpp/
│   ├── go/
│   │   ├── go_game.h     — GoBoard class (flat array, Zobrist hashing, positional superko)
│   │   └── go_game.cpp   — C++ implementation
│   ├── mcts/
│   │   ├── mcts.h        — MCTSTree: PUCT, virtual loss, batched eval, fast rollout
│   │   └── mcts.cpp      — C++ implementation
│   └── bindings/
│       └── bindings.cpp  — pybind11 bindings
└── proto/              — gRPC protobuf definitions (inference service)

tests/
├── test_go.py          — FastGoBoard tests
├── test_mcts.py        — Python MCTS tests
├── test_cpp_go.py      — C++ GoBoard tests
├── test_cpp_mcts.py    — C++ MCTS tests
├── test_cpp_mcts_batched.py — Batched C++ MCTS tests
├── test_model.py       — Model forward/backward tests
├── test_mcts_data.py   — Dataset loading tests
└── test_gpu_lease.py   — Cluster lease tests

scripts/
└── build_cpp.sh        — CMake build for C++ pybind11 extension
```

### 2.2 FastAPI Web Server (`play.py`) — **KEY FOR INTEGRATION**

AutoGo has a fully functional REST API for playing Go games. This is how we integrate.

**Server startup:**
```bash
uv run -m alpha_go.play --host 0.0.0.0 --port 8000
```

**API Endpoints:**

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | HTML game UI |
| `GET` | `/api/agents` | List available agents: `["random", "gnugo1", ...]` |
| `POST` | `/api/new_game?size=9&color=black&agent=gnugo1` | Create new game, returns `GameState` |
| `GET` | `/api/game/{game_id}` | Get current game state |
| `POST` | `/api/game/{game_id}/move` | Make a move `{"row": 3, "col": 4}` |
| `POST` | `/api/game/{game_id}/pass` | Pass move |
| `POST` | `/api/game/{game_id}/undo` | Undo last move |
| `GET` | `/api/game/{game_id}/assist` | Get move probabilities from KataGo |

**GameState response model:**
```python
class GameState(BaseModel):
    game_id: str
    board: list[list[int]]          # 0=empty, 1=black, 2=white
    size: int                       # 9, 13, or 19
    to_play: int                    # 1=BLACK, 2=WHITE
    last_move: tuple[int, int] | None
    is_over: bool
    result: str | None              # e.g. "W+2.5"
    legal_moves: list[tuple[int, int]]
    human_color: int
    message: str
```

**MoveRequest:**
```python
class MoveRequest(BaseModel):
    row: int | None = None
    col: int | None = None
    pass_move: bool = False
```

**Agent selection:** Agent is chosen at game creation via `agent` query param. Registered agents: `random`, `gnugo1` (GNU Go level 1). NN agents require checkpoint path or gRPC server.

### 2.3 C++ GoBoard (`go_game.h`) — What We Port to Rust

The C++ `GoBoard` is the core we need to understand. Key details:

```cpp
class GoBoard {
    static constexpr int8_t EMPTY = 0;
    static constexpr int8_t BLACK = 1;
    static constexpr int8_t WHITE = 2;
    static constexpr float KOMI = 7.5f;

    // Core operations
    bool play(int row, int col);      // Returns false if illegal
    bool play_flat(int index);        // Flat index version (row * size + col)
    bool pass();                      // Pass move
    bool is_legal(int row, int col) const;
    bool is_legal_flat(int index) const;
    std::vector<int> get_legal_moves_flat() const;
    bool is_game_over() const;        // consecutive_passes >= 2
    float score() const;              // Tromp-Taylor area scoring
    int8_t get_winner() const;        // BLACK, WHITE, or 0 for draw

    // State access
    int size() const;
    int8_t at(int row, int col) const;
    int8_t to_play() const;
    int consecutive_passes() const;
    int move_count() const;
    float komi() const;
    std::optional<int> ko_point() const;
    int flat_index(int row, int col) const;
    std::pair<int, int> row_col(int flat) const;

    // Python interop
    void set_from_array(const int8_t* board_data, int8_t to_play);
};
```

**Key implementation details:**
- Flat array `board_` of `int8_t` (size * size) for cache efficiency
- Pre-computed neighbor cache (`neighbor_indices_`, `neighbor_counts_`)
- Flood-fill for group + liberty detection
- **Positional superko** via Zobrist hashing — not just simple ko
- `seen_hashes_` tracks all board hashes; any repeat is illegal
- Komi default 7.5 (AI standard)
- Tromp-Taylor area scoring: stones + surrounded empty territory

### 2.4 Python FastGoBoard (`go.py`) — Reference for Rust Port

Numpy-based Python fallback. Useful as algorithm reference:

- `__slots__` for memory efficiency
- Cached neighbor computation
- `_get_group_and_liberties()` — BFS flood fill
- `_would_be_suicide()` — temp stone placement + liberty check + capture check
- Simple ko only (not superko like C++ version)
- Chinese rules scoring: `score() = black_stones + black_territory - white_stones - white_territory - komi`
- `_flood_empty()` — BFS to determine territory ownership (empty region bordered by single color)

**GoState** wraps FastGoBoard for MCTS:
```python
class GoState:
    def get_legal_actions(self) -> list[tuple[int, int] | None]  # includes pass
    def apply_action(self, action) -> GoState                    # returns new state
    def is_terminal(self) -> bool                                # two consecutive passes
    def get_reward(self, player: int) -> float                   # 1.0 win, 0.5 draw, 0.0 loss
    def current_player(self) -> int                              # 0=BLACK, 1=WHITE
    def clone(self) -> GoState
```

### 2.5 MCTS Implementation (`mcts.py` + `mcts.h`)

**Python MCTS:**
- `MCTSConfig`: c_puct, lambda (0=AlphaZero pure value), Dirichlet noise, temperature
- `Node`: tree node with N, Q, children, policy priors
- `perform_alphago_playout()`: PUCT selection → expand → evaluate → backprop
- `fast_rollout()`: policy-guided rollouts with temperature sampling
- `run_mcts()`: run N simulations, return root node
- `select_action_from_mcts()`: temperature-based action selection from visit counts

**C++ MCTS (`mcts.h`):**
- `MCTSConfig`: c_puct, lambda, Dirichlet, temperature, max_depth, PCR (Playout Cap Randomization)
- `MCTSNode`: N, N_virt (virtual loss), Q, first_eval_value, parent, depth, children, logP_A
- `MCTSTree`: flat vector storage, PUCT selection, virtual loss for leaf-parallel batching
- `run_simulations_batched()`: collect leaves → batch evaluate → expand + backup
- `fast_rollout()`: policy-guided with depth limit
- Batched evaluator: `BatchedEvaluatorFn = fn(Vec<GoBoard>) -> Vec<(policy, value)>`

### 2.6 Model Architectures (`model.py`)

**GoTransformer** (ViT-style):
- Input: (B, H, W) board → flatten → 3-class embedding + positional embedding + CLS token
- 13 Transformer layers (d_model=256, n_heads=8, d_ff=1024)
- Policy head: per-position logit from position tokens + pass logit from CLS
- Value head: single logit from CLS → sigmoid → win probability
- Loss: cross-entropy (policy) + binary cross-entropy (value)

**SizeInvariantGoResNet** (CNN, production model):
- Input: one-hot 3-channel (empty/self/opp) with zero-padding mask
- Tower: n_blocks MaskedResBlocks (128 channels)
- MaskedBatchNorm2d / MaskedGroupNorm2d for variable-size boards
- Policy head: 1x1 conv → flatten (excess logits = -inf) + pass FC
- Value head: masked-avg-pool → FC → ReLU → FC → scalar
- Works on variable board sizes (9×9 and 19×19 in same batch!)

**MuP (maximal update parameterization)** for scaling:
- `MuPModelConfig` with width multiplier
- `create_mup_model()` factory
- Multiple configs: "10M", "100M" etc.

### 2.7 Self-Play Pipeline (`self_play.py`)

- `run_sequential()`: one game at a time, for debugging
- `run_parallel()`: thread-pool with `_play_game_worker()`, shared inference engine
- `_get_or_create_agent()`: thread-local agent creation from config
- `GameResult`: stores moves, outcome, timing metrics
- Output: NPZ files with board states, moves, MCTS visit counts, temperatures
- `main()`: full CLI with `--black`, `--white`, `--num_games`, `--board_size`, `--save-name`

### 2.8 Agent Registry (`agents/base.py`)

```python
class Agent(ABC):
    def start_game(self, board_size: int) -> None: ...
    def notify_move(self, row: int, col: int) -> None: ...
    def end_game(self) -> None: ...
    def select_move(self, board: alpha_go_cpp.GoBoard, seed: int) -> tuple[int, int]: ...
```

Registered agents:
- `random` — uniform random legal move
- `gnugo1` — GNU Go level 1 (subprocess GTP)
- NN agents require checkpoint or gRPC inference server

---

## 3. What AutoGo Has That We Don't

| Component | AutoGo | Our Status | Gap |
|-----------|--------|------------|-----|
| Go game engine (9/13/19×19) | C++ pybind11 + Python FastGoBoard | None | **Missing** — need Go GameState impl |
| Policy network | GoTransformer + SizeInvariantGoResNet | Config presets (micro/game) | **Missing** — need Go-specific policy head |
| Value network | Combined with policy (dual head) | No value head | **Missing** — need value prediction |
| C++ MCTS with virtual loss | PUCT + batched eval + fast rollout | Python MCTS in `mcts.rs` (generic) | **Partial** — we have generic MCTS, need Go-specific tuning |
| Self-play data pipeline | Collect → NPZ → GoDataset → train | ReplayWriter (JSONL) | **Partial** — we have JSONL, need scale |
| Distributed fleet orchestration | `cluster.py` + SSH + Docker | None | **Missing** — but we have wgpu GPU |
| `autoresearch` skill | Claude-driven hyperopt | BanditPruner + TrialLog | **Partial** — bandit does this but no Claude loop |
| **FastAPI web server** | `play.py` on port 8000 | None | **Missing** — but we can USE theirs |
| gRPC inference server | Remote model serving | wgpu local only | **Different approach** |
| Positional superko | Zobrist hashing in C++ | Not needed for modelless | **Skip for now** |
| Size-invariant model | Variable board sizes in one batch | Fixed configs | **Skip for now** |

---

## 4. What We Have That AutoGo Doesn't

| Component | Our System | AutoGo Equivalent |
|-----------|-----------|-------------------|
| **Generic GameState trait** | `GameState::advance()` works on Bomber, FFT, Monopoly | Go-specific only |
| **G-Zero self-play** | Hint-δ + GRPO + DPO (verifier-free) | Traditional AlphaGo-style MCTS self-play |
| **Modelless HL** | Bandit + template + δ, no GPU needed | All model-based |
| **Heuristic Learning thesis** | Proven across 3 games (Bomber, FFT, Monopoly) | Single game |
| **TFT game theory** | 99% win rate in FFT arena | No game theory strategies |
| **Speculative decoding** | DDTree + DFlash + Leviathan | None |
| **KV cache compression** | TurboQuant 3-bit, 5.3× compression | None |
| **Block-sparse prefill** | PFlash, 21.3× sequence reduction | None |
| **Schur complement training** | ✅ 1-shot exact solve, 100% lower loss vs AdamW (Plan 067) | Iterative AdamW only |
| **HLA streaming attention** | O(1) memory, SIMD-accelerated | Standard attention |
| **Self-improving loop** | Feedback → retrain → hot-swap | Manual train loops |
| **WASM validator sandbox** | `riir-validator-sdk` + wasmtime | None |
| **Prompt router + expert registry** | 3-tier embedding + keyword routing | None |
| **Rust, zero GC** | Deterministic perf, single binary | Python + C++ |
| **Production LoRA pipeline** | wgpu 26 WGSL kernels | PyTorch training |
| **Feature-gated modularity** | 15+ feature flags, SOLID decomposition | Monolithic |

---

## 5. API Integration Strategy — Playing Against AutoGo

### 5.1 Docker Spin-Up (Easiest Path)

AutoGo already has a Docker setup. We can spin it up and call its REST API:

```bash
# Build dev container (includes GNU Go + C++ extension)
cd microgpt-rs/.raw/autogo
docker build -f .devcontainer/Dockerfile -t autogo-dev .

# Run with Go web server on port 8000
docker run -d --name autogo \
  -p 8000:8000 \
  autogo-dev \
  bash -c "git submodule update --init && uv sync && uv run -m alpha_go.play --host 0.0.0.0 --port 8000"

# Or if no GPU needed (random/gnugo agents don't need GPU):
docker run -d --name autogo -p 8000:8000 autogo-dev \
  bash -c "uv run -m alpha_go.play --host 0.0.0.0 --port 8000"
```

### 5.2 REST API Client in Rust

Our Rust system calls AutoGo's API to play games head-to-head:

```text
┌─────────────────────────┐         HTTP          ┌─────────────────────┐
│  microgpt-rs            │ ────────────────────── │  AutoGo Container   │
│                         │  POST /api/new_game    │                     │
│  GoGameState (our impl) │  POST /api/game/{id}/  │  GoBoard (C++)      │
│  GoHLPlayer (our AI)    │       move             │  Agent (their AI)   │
│  GoGZeroPlayer          │  GET  /api/game/{id}   │  MCTS / NN / Random │
│                         │                        │                     │
│  ┌─────────────────┐    │  We play as one color  │  They play as other │
│  │ Tournament Loop  │    │  via REST API calls    │  color automatically│
│  │ 100+ games       │    │                        │                     │
│  └─────────────────┘    │                        │                     │
└─────────────────────────┘                        └─────────────────────┘
```

**Game flow via API:**
1. `POST /api/new_game?size=9&color=black&agent=random` → we get `game_id`, we play Black
2. Our AI picks a move from `legal_moves`
3. `POST /api/game/{id}/move {"row": r, "col": c}` → AutoGo responds with AI move
4. Repeat until `is_over == true`
5. Record result from `result` field

**Two modes:**

| Mode | We Play As | API Flow |
|------|-----------|----------|
| **We are Black** | `color=black` | We move first, AutoGo responds |
| **We are White** | `color=white` | AutoGo moves first (`/api/new_game` triggers AI move) |

### 5.3 Local Python Run (Alternative, No Docker)

If Docker is unavailable, run directly with Python:

```bash
cd microgpt-rs/.raw/autogo

# Install dependencies
pip install uv && uv sync

# Build C++ extension (requires cmake, pybind11)
./scripts/build_cpp.sh

# Start server
uv run -m alpha_go.play --host 127.0.0.1 --port 8000
```

**Caveat:** C++ extension build requires `libpython3.10.so` (Linux). On macOS, may need adaptation. Docker is more reliable.

### 5.4 Head-to-Head Tournament Design

```rust
/// Tournament configuration
struct GoTournamentConfig {
    board_size: usize,           // 9 or 19
    num_games: usize,            // 100+ for statistical significance
    our_player: GoPlayerType,    // Random, Greedy, HL, GZero
    their_agent: String,         // "random", "gnugo1", "nn_mcts"
    autogo_url: String,          // "http://localhost:8000"
    we_play_both_sides: bool,    // true = each player plays both colors
}

/// Tournament result
struct GoTournamentResult {
    our_wins: usize,
    their_wins: usize,
    draws: usize,
    avg_score_delta: f32,        // positive = we score more
    games_per_sec: f32,
    total_moves: usize,
}
```

### 5.5 Agent Strength Hierarchy (AutoGo)

From weakest to strongest:

| Agent | Description | Our Baseline |
|-------|-------------|-------------|
| `random` | Uniform random legal moves | Our `GoRandomPlayer` |
| `gnugo1` | GNU Go level 1 (heuristic) | Our `GoGreedyPlayer` |
| `nn_agent` (random init) | Untrained NN policy | Our `GoValidatorPlayer` |
| `nn_agent` (trained) | Trained NN policy | Our `GoHLPlayer` |
| `nn_mcts` (trained) | NN + MCTS search | Our `GoGZeroPlayer` |

**Our progression:** Beat `random` → beat `gnugo1` → beat `nn_agent` → beat `nn_mcts`

---

## 6. Distillation Map: What to Extract

### 6.1 Distill (Extract Core Ideas)

| AutoGo Concept | Our Implementation | Source File | Module |
|----------------|-------------------|-------------|--------|
| Go board rules (Tromp-Taylor) | `GoState` GameState impl | `go.py`, `go_game.h` | `src/pruners/go/state.rs` |
| Legal move generation | `available_actions()` | `go.py:is_legal()`, `go_game.h` | `src/pruners/go/state.rs` |
| Capture + ko detection | `advance()` + ko tracking | `go.py:play()`, `go_game.h` | `src/pruners/go/state.rs` |
| Territory scoring | `reward()` via flood fill | `go.py:score()`, `go_game.h` | `src/pruners/go/state.rs` |
| MCTS PUCT + Dirichlet | Enhance our `mcts_search()` | `mcts.py`, `mcts.h` | `src/pruners/game_state/mcts.rs` |
| Policy/value dual head | Future: dual-head config | `model.py:GoTransformer` | `src/transformer.rs` |
| Self-play loop | G-Zero `SelfImprovingCycle` | `self_play.py` | `pruners/g_zero/` |
| Agent registry pattern | Adapt for Go players | `agents/base.py` | `src/pruners/go/players.rs` |
| REST API game protocol | Rust HTTP client | `play.py` | `src/pruners/go/autogo_client.rs` |
| Replay data (NPZ → JSONL) | Our ReplayWriter format | `dataset.py`, `gameplay.py` | `src/pruners/go/replay.rs` |

### 6.2 Skip (Not Applicable / Already Better)

| AutoGo Concept | Why Skip |
|----------------|----------|
| Python + C++ pybind11 | We're Rust — one language, no FFI overhead |
| Docker + SSH fleet | We have wgpu GPU, no multi-host needed for PoC |
| PyTorch/JAX training | We have wgpu LoRA + DPO + GRPO |
| Go-specific MCTS | Our `GameState` trait is generic — MCTS works on any game |
| Claude-driven autoresearch | Our bandit does this automatically, no LLM needed |
| GNU Go subprocess | We don't need GTP engine — we implement Go rules in Rust |
| gRPC inference | We use local wgpu, not distributed inference |
| Positional superko | Simple ko sufficient for modelless play; add later if needed |
| Size-invariant model | We target 9×9 first; variable-size is optimization, not core |

### 6.3 Compete (Beat Head-to-Head)

| Metric | AutoGo | Our Target | How |
|--------|--------|-----------|-----|
| Win rate vs `random` | ~100% (any agent) | ~100% | Baseline sanity check |
| Win rate vs `gnugo1` | Varies by agent | >55% | HL + bandit should beat GNU Go level 1 |
| Win rate vs `nn_mcts` | Strongest | >50% (stretch) | G-Zero + MCTS combined |
| Moves/second (game sim) | Python overhead | 10-100× faster | Rust `GoState::advance()` vs Python `FastGoBoard` |
| Training iterations to target ELO | 5-10 cycles | 2-3 cycles | Schur 1-shot + G-Zero δ signal |
| Games needed for scaling law | Thousands | Hundreds | Bandit focuses exploration on blind spots |
| Researcher automation | Claude reads logs | Bandit auto-tunes | `TrialLog` + `BanditPruner` |
| Cross-game generality | Go only | 4+ games | Bomber, FFT, Monopoly, Go |

---

## 7. Competitive Strategy

### Thesis: We Beat AutoGo on Research Velocity, Not Go Strength

AutoGo's goal is to study **automated AI research**, not to beat Lee Sedol. We compete on the research automation dimension:

1. **Faster iteration cycle** — Rust is 10-100× faster than Python for game simulation. Our Bomber runs 84.5 games/sec. Go on Rust would be similar. More games per second = faster research.

2. **Better reward signal** — AutoGo uses win/loss (sparse). We use Hint-δ (dense). Dense reward means fewer episodes to learn the same policy.

3. **Modelless baseline** — Before spending GPU, our G-Zero Phase 1 improves heuristics for free. AutoGo goes straight to GPU training. We learn faster because we start with bandit.

4. **Generic across games** — AutoGo proves automated research on Go. We prove it on 4 games simultaneously (Bomber, FFT, Monopoly, Go). STRATEGA finding confirmed: domain heuristics > generic search. Our HL thesis is stronger.

5. **Scaling law efficiency** — Bandit focuses exploration on blind spots. Instead of uniformly sampling games, we target where the model is weakest. Fewer total games for same quality.

6. **Automated researcher without LLM** — AutoGo uses Claude to run experiments. Our bandit + trial log automates hyperparameter selection. No LLM API cost, fully local.

7. **Head-to-head via REST API** — We can directly benchmark against AutoGo's agents by spinning up their Docker container. No subjective claims — we win or we lose, measured objectively.

### What We Don't Beat

- **Go ELO strength** — AutoGo has months of training on GPU fleet. We won't match Go ELO.
- **Scaling law rigor** — AutoGo has actual scaling law experiments. We'd need to run those.
- **Distributed training** — AutoGo runs on multi-GPU fleet. We're single-machine wgpu.
- **NN model quality** — AutoGo has trained SizeInvariantGoResNet with proper MuP scaling. We'd start with random weights.

### Honest Assessment

We're not trying to beat AlphaGo at Go. We're trying to beat AutoGo's **research automation** thesis by showing:

> A Rust-based, bandit-driven, generic GameState system with G-Zero self-play produces stronger game AI faster than a Python-based, Go-specific, Claude-driven pipeline.

The metric is **research velocity**: how quickly can an automated system improve game-playing strength across multiple domains. We measure this head-to-head via AutoGo's own REST API.

---

## 8. Key Insights from AutoGo Code

### 8.1 "Automate the researcher, not just the training"

AutoGo's `autoresearch` skill has Claude iterate on experiments. Our `BanditPruner` + `TrialLog` does this without an LLM — each hyperparameter config is a bandit arm, UCB1 selects the next experiment. Same result, zero API cost.

### 8.2 Two board implementations for different needs

AutoGo has `FastGoBoard` (Python/numpy) for MCTS rollouts and `GoBoard` (C++) for heavy computation. We do the same: `GoState` (Rust, ~500 bytes for 9×9) for MCTS, same struct for everything else. One language, no FFI.

### 8.3 Size-invariant model is clever but premature

AutoGo's `SizeInvariantGoResNet` handles 9×9 and 19×19 in the same batch via zero-padding + masking. Impressive engineering, but for our PoC we target 9×9 only. If results are promising, we add 19×19 later.

### 8.4 Positional superko vs simple ko

C++ `GoBoard` implements full positional superko (Zobrist hashing, tracks all seen hashes). Python `FastGoBoard` only does simple ko (single forbidden point). For modelless play, simple ko is sufficient. We'll start with simple ko.

### 8.5 FastAPI is our integration bridge

The `play.py` FastAPI server is the key. We don't need to reimplement Go rules perfectly — we can validate our Rust `GoState` against AutoGo's C++ `GoBoard` by playing games through the API and comparing results.

### 8.6 "Start synchronous before async"

AutoGo recommends synchronous train/collect before async. Our `GZeroLoop` already does this — sequential rounds with `RoundMetrics`. Good validation of our approach.

### 8.7 "Falling back to docker exec calls over SSH ended up working best"

AutoGo abandoned complex orchestration for simple Docker + SSH. Our approach is simpler still: single binary, wgpu GPU, no orchestration. The Rust binary IS the framework.

---

## 9. Implementation Roadmap (Refined)

### Phase 0: API Bridge (NEW — Play Against Existing System)

Before implementing our own Go engine, validate the integration path:

1. Spin up AutoGo Docker container (or local Python)
2. Write Rust HTTP client calling `play.py` REST API
3. Implement `AutoGoClient` that wraps the REST API
4. Play random games through the API to validate flow
5. Measure latency: how fast can we play games via HTTP?

This proves the head-to-head benchmarking infrastructure works.

### Phase 1: Go GameState (Prove Genericity)

Port `FastGoBoard` + `GoState` to Rust, implementing our `GameState` trait:
- `GoState` snapshot (board Vec, ko_point, captures, to_play, passes, move_count, komi)
- `GoAction` (`Place(x,y)` | `Pass`)
- Tromp-Taylor scoring via flood fill
- Simple ko (not superko — add later if needed)
- Validate against AutoGo API: play same moves, check same results

### Phase 2: Go Player Strategies (Prove HL Thesis on Go)

Implement Go players following our proven pattern:
- `GoRandomPlayer` — random legal move
- `GoGreedyPlayer` — maximize captures + liberty advantage
- `GoValidatorPlayer` — safety rules (no self-atari, maintain eyes)
- `GoHLPlayer` — bandit-driven with Go features (opening, capture, influence)
- `GoGZeroPlayer` — template proposer + delta bandit

### Phase 3: Head-to-Head Tournament (Prove via API)

Play our players against AutoGo's agents via REST API:
- Our Random vs their Random → validate integration
- Our Greedy vs their `gnugo1` → measure heuristic quality
- Our HL vs their `gnugo1` → prove bandit works on Go
- Our GZero vs their `nn_mcts` → stretch goal

### Phase 4: Go G-Zero Self-Play (Prove Transfer)

Run G-Zero self-play on Go:
- `GoTemplateProposer` (joseki, fuseki, invasion, attachment, tenuki)
- `DeltaBanditPruner` with δ reward
- ReplayWriter → training data
- Compare: G-Zero vs random vs MCTS vs bandit-only

### Phase 5: AutoResearch Loop (Prove Velocity)

Bandit over hyperparameters:
- `AutoResearchLoop` with `GoResearchConfig` arms
- `TrialLog` records per-experiment metrics
- Compare: our automated loop vs AutoGo's Claude-driven approach

---

## 10. Verdict

**Distill?** Yes. AutoGo's Go engine (Tromp-Taylor scoring, capture detection, ko rule) is well-implemented and directly portable to Rust. The agent registry pattern is clean. The self-play pipeline maps to our G-Zero system.

**Compete?** Yes, head-to-head via REST API. Spin up their Docker, call their API, measure win rates. Objective, reproducible, no subjective claims.

**Skip?** Python + C++ FFI (we're Rust), distributed fleet (single binary), PyTorch training (wgpu), Claude-driven experiments (bandit), superko (simple ko sufficient for PoC), size-invariant model (target 9×9 first).

**Risk?** Medium-Low. The API integration path is straightforward (just HTTP calls). The Go rules port is well-defined (we have both Python and C++ reference). The main risk is our modelless players being too weak against their trained NN agents — but that's a known limitation we acknowledge.

---

## 11. Model-Based Go via `riir-gpu` — Feature-Gated Path

External reviews correctly identified that `riir-ai` has every component needed for Transformer-based Go. This section documents what exists, the feature-gate strategy, and the adaptation path. **No hype filter needed — the demos prove the pipeline works.**

### 11.1 What Exists (Verified, Feature-Gated)

| Component | Location | Feature Gate | Go Adaptation |
|-----------|----------|-------------|---------------|
| Fourier spatial MCTS | `riir-engine/src/fourier/mcts.rs` | `fourier` | Implement `state_to_entities` for Go tactical shapes |
| Generic MCTS | `riir-engine/src/mcts.rs` | `game_state` | ✅ Works on any `GameState` — zero adaptation |
| GRPO loss | `riir-gpu/src/loss_grpo.rs` | `training` | ✅ Game-agnostic — zero adaptation |
| DPO loss (GPU) | `riir-gpu/src/loss_dpo.rs` | `training` | ✅ Game-agnostic — WGSL kernels work as-is |
| DeltaFilter | `riir-gpu/src/delta_filter.rs` | `training` | ✅ Game-agnostic — 6-stage pipeline works as-is |
| GZeroLoop | `riir-gpu/src/gzero_loop.rs` | `training` | ✅ Game-agnostic — activates when Go replays exist |
| Game replay → LoRA | `riir-gpu/src/game/trainer.rs` | `training` | Adapt `GameAction` enum → 82 Go tokens |
| Game policy config | `riir-gpu/src/game/policy.rs` | `training` | Adapt `GameConfig` → 3 board + 82 action vocab |
| WASM Validator SDK | `riir-validator-sdk/` | `go-wasm` | Compile `GoState::is_legal` → `go_validator.wasm` |
| MTP projection cache | `riir-router/src/mtp_cache.rs` | `percepta` | Document as future: Go tokenizer → MTP projections |
| Schur complement | `riir-gpu/src/schur.rs` | `training` | ✅ Domain-latent training — 1-shot weight updates |
| Bandit + WASM + LoRA | `riir-examples/examples/bandit_with_real_model_demo.rs` | `bandit` | ✅ **Full pipeline proven**: Draft → DDTree → WasmPruner → Leviathan → bandit.update() |
| Bomber tech A/B | `riir-examples/examples/bomber_tech_ab_demo.rs` | `bomber-wasm` | ✅ **Proven**: LoRA vs WASM vs LoRA+WASM vs Full HL — combined wins |
| G-Zero arenas | `riir-examples/examples/g_zero_01_arena.rs` | `g_zero` | ✅ **Proven**: GZero beats Greedy/Validator/HL across Bomber + FFT |

### 11.2 Proven Results (Not Theory)

The demos aren't aspirational — they're running code with measured results:

1. **`bandit_with_real_model_demo.rs`** — Loads real `rust_validator.wasm`, real `py2rs_lora.bin` (trained by riir-burner on Gemma 4 E4B), runs real LeviathanVerifier p/q rejection sampling. Full pipeline: Draft → DDTree + BanditPruner\<WasmPruner\> → LeviathanVerifier → bandit.update(). **This is the exact architecture for Go — just swap the validator and vocab.**

2. **`bomber_tech_ab_demo.rs`** — 1000-round A/B test: LoRA-only vs WASM-only vs LoRA+WASM vs Full HL (LoRA+WASM+Bandit+AbsorbCompress). Combined wins. **This proves the integration works end-to-end — no component conflict.**

3. **`g_zero_04_player_ab_benchmark.rs`** — Isolated performance benchmark across 5 player configs × 1000 rounds. Measures survival rate, avg score, kills, and per-action latency. **Our Rust players are faster — the "slower model loses" claim is unproven and contradicted by our benchmarks.**

4. **`g_zero_fft_01_arena.rs`** through `g_zero_fft_06_tft_benchmark.rs` — 6 FFT demos proving G-Zero transfers across game genres (combat RPG, not just Bomberman). **G-Zero on Go is the next transfer, not a theoretical leap.**

### 11.3 Feature-Gate Strategy

No "two-plan" separation needed. Everything behind feature gates:

```toml
[features]
go = ["bandit"]                          # Phase 1: GoState + MCTS + HL (Plan 065)
go-training = ["go", "riir-gpu/training"] # Phase 2: LoRA training + GZeroLoop
go-wasm = ["go", "riir-validator-sdk"]    # Phase 2: go_validator.wasm
go-fourier = ["go", "riir-engine/fourier"] # Phase 2: Fourier spatial MCTS
go-mtp = ["go", "riir-router"]            # Phase 3: MTP projections for Go vocab
go-full = ["go-training", "go-wasm", "go-fourier", "go-mtp"]  # Everything
```

This is how bomber does it (`bomber`, `bomber-agent`, `bomber-wasm`). Same pattern, new domain.

### 11.4 Adaptation Path: `game/trainer.rs` → Go

**Bomberman (existing, proven):**
```text
Board vocab:  4 tokens (Floor, Wall, Destructible, PowerUp)
Action vocab: 6 tokens (Up/Down/Left/Right/Bomb/Wait)
Sequence:     170 tokens (169 board + 1 action)
Model:        ~6K params, LoRA rank 4
```

**Go (adapt, same pipeline):**
```text
Board vocab:  3 tokens (Empty, Black, White)
Action vocab: 82 tokens (81 placements for 9×9 + Pass)
Sequence:     82 tokens (81 board + 1 action)
Model:        ~4K params, LoRA rank 4
```

**Training flow (reuses existing `riir-gpu` infrastructure, gated behind `go-training`):**
```text
GoState::play_random_game()           (Plan 065 GoState)
  → GoReplay                          (Plan 065 T16a)
  → game/replay.rs → samples          (adapt GameAction enum)
  → game/trainer.rs → 82-token seq    (adapt encode_game_sample)
  → riir-gpu LoRA fine-tuning         (existing wgpu kernels)
  → Go LoRA adapter (.bin)            (tiny model, fast train)
  → GoLoRAPlayer                      (new player type)
```

**G-Zero model-based activation (gated behind `go-training`):**
```text
GZeroLoop round (riir-gpu/src/gzero_loop.rs):
  1. GoTemplateProposer → query-hint pairs    (Plan 065 T33)
  2. Go LoRA Generator → move predictions     (adapted game/policy.rs)
  3. HintDelta → intrinsic reward             (log-prob shift)
  4. DeltaFilter → preference pairs           (6-stage, game-agnostic)
  5. GRPO → train Proposer                    (loss_grpo.rs)
  6. DPO → train Generator                    (loss_dpo.rs, WGSL kernels)
```

### 11.5 What Needs Proving (Research Questions)

| Question | How to Answer | Feature Gate |
|----------|--------------|-------------|
| Does GoState `GameState` impl produce correct legal moves? | Fuzz vs AutoGo API (T15a) | `go` |
| Does Fourier spatial hash help Go MCTS? | Benchmark: Fourier MCTS vs vanilla MCTS on Go | `go-fourier` |
| Does Go LoRA training converge? | Train on replay data, measure loss curve | `go-training` |
| Does GZeroLoop improve Go win rate over time? | Self-play tournament, track rounds vs win rate | `go-training` |
| Does `go_validator.wasm` prevent illegal move generation? | A/B test: LoRA vs LoRA+WASM on Go | `go-wasm` |
| Does MTP improve Go token prediction? | Benchmark: with/without MTP projection | `go-mtp` |
| Can our Rust stack beat AutoGo's Python/C++? | Head-to-head tournament via API (T25-T31) | `go` |

**Every question is answerable with a benchmark. No speculation needed — just run the experiments.**

### 11.6 Decision

Plan 065 implements `go` feature gate (GoState + MCTS + HL + API bridge). Each subsequent feature gate (`go-training`, `go-wasm`, `go-fourier`, `go-mtp`) is unlocked by proving the previous one works via benchmarks. The `game/trainer.rs` → Go adaptation is the first unlock after Plan 065 completes.

---

## References

- AutoGo repo: https://github.com/ericjang/autogo
- AutoGo tutorial: https://evjang.com/2026/04/28/autogo.html
- AutoGo local code: `microgpt-rs/.raw/autogo/`
- Eric Jang: VP of AI at Google DeepMind (formerly NVIDIA)
- Dario Amodei quote: "Machines of Loving Grace" essay
- Our G-Zero implementation: `microgpt-rs/.plans/049_g_zero_self_play.md`
- Our GameState trait: `microgpt-rs/.plans/056_game_state_forward_model.md`
- Our HL thesis results: `microgpt-rs/.docs/10_bomber_arena.md`
- FastGoBoard reference: `microgpt-rs/.raw/autogo/src/alpha_go/go.py`
- GoBoard C++ reference: `microgpt-rs/.raw/autogo/src/alpha_go/cpp/go/go_game.h`
- REST API server: `microgpt-rs/.raw/autogo/src/alpha_go/play.py`
- MCTS reference: `microgpt-rs/.raw/autogo/src/alpha_go/mcts.py`
- Model reference: `microgpt-rs/.raw/autogo/src/alpha_go/model.py`
- Fourier spatial MCTS: `riir-ai/crates/riir-engine/src/fourier/mcts.rs`
- Fourier encoder: `riir-ai/crates/riir-engine/src/fourier/encoder.rs`
- Generic MCTS: `riir-ai/crates/riir-engine/src/mcts.rs`
- GRPO loss: `riir-ai/crates/riir-gpu/src/loss_grpo.rs`
- DPO loss (GPU): `riir-ai/crates/riir-gpu/src/loss_dpo.rs`
- DeltaFilter: `riir-ai/crates/riir-gpu/src/delta_filter.rs`
- GZeroLoop: `riir-ai/crates/riir-gpu/src/gzero_loop.rs`
- Game replay → LoRA: `riir-ai/crates/riir-gpu/src/game/trainer.rs`
- Game policy config: `riir-ai/crates/riir-gpu/src/game/policy.rs`
- Game replay parser: `riir-ai/crates/riir-gpu/src/game/replay.rs`
- Schur complement: `riir-ai/crates/riir-gpu/src/schur.rs` (Plan 067 ✅ — CLEAR WIN)
- WASM Validator SDK: `riir-ai/crates/riir-validator-sdk/`
- WASM Pruner: `riir-ai/crates/riir-wasm/src/wasm_pruner.rs`
- MTP projection cache: `riir-ai/crates/riir-router/src/mtp_cache.rs`
- Bomber WASM pruner: `microgpt-rs/src/pruners/bomber/wasm_pruner.rs`
