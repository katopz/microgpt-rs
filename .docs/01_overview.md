# microgpt-rs: Overview

## What It Is

A from-scratch Rust implementation of a GPT-2 style transformer with speculative decoding, designed as an educational/performance research vehicle. No ML frameworks — just `Vec<f32>`, matmul, and hand-tuned attention kernels.

## Project Goals

- CPU-first inference engine with zero-allocation hot paths
- Speculative decoding pipeline (DDTree + DFlash + Leviathan verification)
- Domain-specific constraint pruning (Sudoku, Rust AST via Validator)
- BPE tokenizer + SynPruner for Rust syntax validation
- Sub-millisecond inference on Apple Silicon
- Discrete Diffusion Forcing (dLLM) research with block-parallel denoising

## Current Capabilities

- Single-token autoregressive generation: ~900K tok/s (micro config)
- DFlash marginal prediction: ~4.2M tok/s
- DDTree build: ~431K trees/s
- Speculative decoding: ~1.64M tok/s (AR Draft)
- forward_raven (16 slots): ~1.6M trees/s
- raven_recall (1000 noise): ~9.3M tok/s
- SIMD-accelerated matmul/HLA kernels: 15.6M ops/s [16×16] NEON (Plan 060)
- forward_hla: ~939K tok/s (single-core, 30K CCU feasible)
- forward_ahla: ~1.2M tok/s (single-core)
- TurboQuant 3-bit KV cache: 5.3× compression, 0.99 attention correlation
- dLLM Discrete Diffusion Forcing: block-parallel denoising (behind `"dllm"` feature, Plan 066)
- SP-KV self-pruned KV attention: 3-10× KV reduction with utility prediction (behind `"sp_kv"` feature, Plan 070)
- PFlash block-sparse prefill: up to 21.3× sequence reduction, 100% NIAH retrieval
- 516 tests passing, zero clippy warnings

## Module Structure

```
src/
  lib.rs            Module index + debug tracking allocator
  main.rs           Entry point (proof → bench → Percepta bench → plot)
  types.rs          Config (micro/micro_lora/draft/game/bpe/bpe_draft/small_target/gqa_draft/dllm_micro/validate + with_overrides), InferenceOverrides, Rng, HlaMode, AttentionMode, softmax, rmsnorm, matmul, matmul_relu, sparse_matmul, sample_token, LoraAdapter, LoraPair, DomainLatent, InferenceResult, lora_apply, kv_dim
  simd.rs          SimdLevel (Scalar/Neon/Avx2), simd_level(), simd_dot_f32, simd_fma_row, simd_outer_product_acc, simd_matvec, simd_matmul_rows, simd_matmul_relu_rows, simd_sparse_dot_f32, simd_sparse_matmul_rows, simd_scale_inplace (Plan 060)
  transformer.rs    TransformerWeights (+ mtp projections), LayerWeights, KVCache, MultiLayerKVCache, KVSnapshot, PagedKVCache, RavenKVCache, ForwardContext (+ sparse buffers + lora_buf + mtp_context_buf + tq_dequant_pos), PrefillContext, forward, forward_with_domain_latent, forward_prefill, forward_paged, forward_raven, forward_turboquant, generate, generate_into, generate_batch, generate_with_prefill, tokens_to_string, project_target_activation, cluster_map_round_robin, cluster_map_from_embeddings, raven_compute_router, raven_update, raven_readout, preload_kv_cache
  feedback.rs       FeedbackConfig, send_feedback ⌁
  percepta/         Percepta 2D Convex Hull Attention + Computation Graph:
    mod.rs          Module declarations, re-exports (15+ submodules)
    types.rs        TieBreak, HullMeta, Vec2 (f64), constants (HARD_K, BIG, EPS)
    legacy.rs       Vec2 (f32), KVCache2D (Graham Scan), Sudoku9x9, SymbolicValidator, StreamingSolver, SolveEvent
    cht.rs          Line, CHT — dynamic convex hull trick / line container
    hull.rs         AttentionResult, HullHalf, HardAttentionHead (dual-hull O(log N)), BruteAttentionHead
    encoding.rs     encode_key, encode_query, clear_key, hard_scale, hard_scale_query
    cumsum.rs       CumSum — cumulative sum via uniform attention
    standard_cache.rs  StandardCache — O(n) softmax attention KV cache
    gates.rs        reglu, stepglu, multiply — gate primitives; PersistSlot, GateKind
    graph/          Computation Graph DSL:
      mod.rs        Module root, re-exports
      types.rs      Expression (sparse linear combo), DimensionKind, Dimension, LookUp, ProgramGraph, GraphBuilder, ValidationError
    weights.rs      TransformerWeights, LayerWeights, AttentionWeights, FfnWeights, HeadInfo, build_weights
    transformer.rs  TransformerConfig, TransformerVocab, GenerationResult, VanillaTransformer
    evaluator.rs    GraphEvaluator — step/predict/evaluate/compare_with_reference
    specialize.rs   SpecializationError, SpecializationReduction, SpecializedModel, UniversalModel
    scheduler.rs    OpKey, Phase, StdLayer, DepGraph, Schedule, build_dep_graph, milp_schedule
    runner.rs       RunnerError, BuildResult, Runner — compile/build/run/evaluate/specialize/full_pipeline
    compile.rs      compile_program, CompiledProgram — C source → WASM → lowered bytecode → token prefix (behind "percepta_compile")
    wasm/           WASM MVP decoder + lowering + interpreter (Futamura projection):
      mod.rs        Module root
      decoder.rs    WasmModule, FuncType, FuncBody, WasmInstr, decode
      lower.rs      lower_hard_ops, check_basic_only
      interpreter/  WASM interpreter as computation graph:
        mod.rs      Module root
        arithmetic.rs  Arithmetic ops dispatch
        dispatch.rs    Instruction dispatch table
        tokens.rs      Token mapping
  benchmark.rs      BenchCategory, BenchResult, run_all, run_all_parallel, save_results_csv, append_timeseries_csv, generate_batch
  plot.rs           plot_results → PNG, plot_timeseries

  speculative/      SOLID decomposition:
    mod.rs          Re-exports
    types.rs        TreeNode, DraftResult, ConstraintPruner trait, ScreeningPruner trait, NoPruner, NoScreeningPruner, BinaryScreeningPruner, SpeculativeContext, DDTreeBranchCache, RejectionReason, DraftEvent, PrefillMode, FlashPrefillConfig, BlockScores
    sampling.rs     sample_from_distribution, sample_residual_distribution, sample_residual_distribution_into
    dd_tree.rs      build_dd_tree, build_dd_tree_pruned, build_dd_tree_screened, build_dd_tree_balanced, TreeBuilder, extract_parent_tokens, extract_parent_tokens_into, extract_best_path, extract_best_path_into, build_inference_result, merge_retrieved_branches
    dflash.rs       dflash_predict, dflash_predict_with, dflash_predict_ar, dflash_predict_ar_with, dflash_predict_conditioned, dflash_predict_conditioned_with, dflash_predict_parallel
    verifier.rs     SpeculativeVerifier trait, SimulatedVerifier, LeviathanVerifier
    step.rs         speculative_step, speculative_step_verifier, speculative_step_rollback, speculative_step_rollback_with, speculative_step_conditioned, speculative_step_conditioned_with, speculative_step_rollback_paged
    prefill.rs      PrefillScorer trait, AttentionScorer, BlockAttentionScorer, compress_prompt, compress_prompt_blocks, block_select, block_select_grid, should_compress, speculative_prefill, speculative_prefill_block, speculative_prefill_adaptive
    flow_pruner.rs  FlowPruner<P> — GFlowNet-inspired stop-probability regularization ♭
    d2f.rs          D2fBlockState, D2fDecodeConfig, D2fBlockResult, D2fPipelineBlock, D2fPipeline, D2fPipelineResult, d2f_decode_block* (behind "dllm" feature)
    ppot/           PPoT (Plans 026 + 027) ○
      mod.rs        Module root, public API re-exports
      types.rs      TokenRule enum, PpotConfig
      entropy.rs    token_entropy, identify_high_entropy_positions, identify_positions_by_rule, identify_positions_adaptive
      resample.rs   ppot_resample, ppot_resample_with_support, ppot_resample_different_value, ppot_resample_multi_strategy, ppot_rescue, ppot_rescue_adaptive, ppot_rescue_reviewed
      knowledge.rs  RejectionInsight, ErrorKind, SessionKnowledge
      rank.rs       rank_by_consistency, rank_by_consistency_weighted, select_best_variant, select_best_variant_weighted

  pruners/          Domain-specific constraint pruners:
    mod.rs          Re-exports
    pathfinder.rs   Target, find_path, find_distance, reachable_positions, enumerate_targets, terrain_cost, manhattan
    tactical_pruner.rs  GameState, TacticalPruner (grid-based tactical puzzle)
    dungeon_pruner.rs   FloorGrid, StairConnection, DungeonMap, DungeonState, DungeonPruner (multi-floor)
    dungeon_pathfinder.rs  DungeonAction, MultiFloorTarget, find_path_on_floor, find_path_multifloor, enumerate_multifloor_targets
    map_generator.rs  GeneratedMap, GeneratedDungeon, MapGenerator (procedural generation)
    sudoku_pruner.rs  SudokuPruner *
    bandit.rs       BanditStrategy, BanditStats, BanditPruner<P>, BanditSession, BanditEvent, BanditResult, BanditEnv trait, BernoulliEnv, GaussianEnv, SharedBanditStats ♭
    trial_log.rs    TrialRecord, TrialSummary, TrialLog ♭
    absorb_compress.rs  CompressConfig, AbsorbCompress trait, AbsorbCompressLayer<P> ♭
    hot_swap.rs     HotSwapPruner<P> — blake3 hash comparison reload ♭
    regression.rs   GoldenTrace, RegressionFailure, RegressionResult, RegressionSuite, ReplayReward trait ♭
    review_metrics.rs  ReviewSummary, ReviewMetrics, ReviewStrategy, EntropyAnomalySummary ♭
    stepcode.rs     PathStep, ShapedPath, shape_path, path_consistency ≋
    game_state/     GameState forward model trait + generic MCTS (Plan 056) ⎗
      mcts_search   mcts_search — Monte Carlo Tree Search
                    StateHeuristic trait, ActionSpaceLog
    bomber/         Bomberman 4-player HL arena (bevy_ecs) ⍟
      mod.rs        BomberAction, PowerUpKind, Cell, ECS components/resources, GameEvent
      arena.rs      ArenaGrid — 13×13 grid generation + presets
      players.rs    BomberPlayer trait, RandomPlayer, GreedyPlayer, ValidatorPlayer, HLPlayer, LoraPlayer, LoraWasmPlayer, NNPlayer
      g_zero_player.rs  GZeroPlayer — G-Zero self-play with template proposer + delta bandit
      tft_player.rs  TftPlayer — Tit-for-Tat with provocation detection
      replay.rs     ReplaySample, ReplayWriter — JSONL replay persistence
      replay_backward.rs  BackwardSample, ReplayBackwardWalker — GFlowNet backward policy
      systems.rs    init_world, spawn_players, run_tick
      wasm_pruner.rs  BomberWasmPruner — WASM batch validation
      wasm_state.rs  serialize_game_state, ZeroCopyStateBuffer
    monopoly/       Monopoly board game engine (bevy_ecs) ✦
      mod.rs        PropertyGroup, SquareKind, TurnPhase, GameEvent (30+ variants), Player, Property, Board, etc.
      board.rs      build_board, shuffle_decks, group_squares
      players.rs    MonopolyPlayer trait, RandomPlayer, GreedyPlayer, ValidatorPlayer, HLPlayer, DecisionContext, Strategy
      systems.rs    init_world, spawn_players, execute_turn, run_game, calculate_rent, transfer_assets
    fft/            FFT Tactics Arena — ATB battle engine ✧
      mod.rs        Module root, re-exports
      types.rs      Class (6), Team, ActionType (9), Stats, Pos, Unit, Action, GameEvent
      battle.rs     BattleState, resolve_action, should_forgive
      status.rs     StatusEffect (9), ActiveEffect, apply_tick_effects, can_cast, can_act, ct_fill_rate
      players.rs    FftPlayer trait, GreedyFFTPlayer, ValidatorFFTPlayer, HLFFTPlayer
      g_zero_player.rs  GZeroFFTPlayer — G-Zero self-play for FFT
      tft_player.rs  TftFFTPlayer — Tit-for-Tat FFT player
    go/             Go GameState + AutoGo API bridge + tournament ⛩
      mod.rs        Module root, re-exports
      types.rs      GoAction (Place, Pass), GoCell (Empty, Black, White)
      state.rs      GoState — flat array board, simple ko, Tromp-Taylor scoring, GameState trait, GoHeuristic
      autogo_client.rs  AutoGoClient — REST API bridge to AutoGo play.py server
      replay.rs     GoReplay, MoveRecord — game recording + deterministic playback
      players.rs    GoPlayer trait, GoRandomPlayer, GoGreedyPlayer, GoValidatorPlayer, GoHLPlayer, GoGZeroPlayer, GoMctsPlayer
      tournament.rs GoTournamentConfig, GoTournamentResult, AutoGoProxyPlayer, run_tournament
      g_zero_player.rs  GoGZeroSelfPlay — HintDelta + absorb-compress self-play
      autoresearch.rs   AutoResearchLoop — UCB1 bandit over config arms, early stopping
    delta_mem/      δ-Mem modelless distillation — associative bandit memory ⌘
      mod.rs        Module root, re-exports
      state.rs      DeltaMemoryConfig, DeltaMemoryState, DeltaMemorySnapshot
      hash.rs       FeatureHasher, ContextFeatures, OutcomeFeatures
      pruner.rs     CorrectionMode, WriteGranularity, MemorySteeredPruner<P>
      multi.rs      AggregationStrategy, MultiDomainMemory
      multi_pruner.rs  MultiDomainMemoryPruner<P>
    g_zero/         G-Zero self-play distillation — verifier-free self-evolution ǂ
      mod.rs        Module root, re-exports
      types.rs      HintDelta, LogProbResult
      template_proposer.rs  QueryTemplate, GeneratedPair, TemplateProposer
      bomber_templates.rs  BomberTemplate (8 strategies), BomberTemplateProposer
      delta_bandit.rs  DeltaBanditPruner<P>
      delta_absorb.rs  DeltaGatedConfig, DeltaGatedAbsorbCompress<P>
      fft_templates.rs  FFTTemplate (10 strategies), FFTTemplateProposer

  tokenizer/        BPE tokenizer (encode/decode/train, Config::bpe())
    mod.rs          Re-exports: BpeTokenizerImpl, BpeTrainer, BpeTokenizer, MergeRule
    types.rs        BpeTokenizer, MergeRule
    bpe.rs          BpeTokenizerImpl (encode/decode), BpeTrainer (train)

  validator/        SynPruner + partial parser ‡
    mod.rs          Module root
    types.rs        PruneResult, ErrorKind, CompilerFeedback
    partial_parser.rs  PartialParser — bracket balance DFA (Tier 0)
    syn_pruner.rs   SynPruner — two-tier pruner (DFA + syn parse)

  turboquant/      TurboQuant KV cache compression (arXiv:2504.19874):
    mod.rs          Module root (re-exports)
    types.rs        TurboQuantCodebook, TurboQuantLayer, TurboQuantKVCacheConfig
    codebook.rs     Lloyd-Max codebook (compute_codebook, quantize, dequantize)
    rotation.rs     QR-based orthogonal rotation + QJL projection
    kv_cache.rs     TurboQuantKVCache (store_key, store_value, dequantize, bit-pack)
    forward.rs      attention_turboquant, dequantize_keys_flat/values_flat, cosine_similarity

  dllm.rs          NoiseSchedule, D2fContext, DenoiseConstraint trait, corrupt_block, forward_bidirectional_positions, forward_block_causal_positions, denoise_loop, denoising_accuracy ⌂
  hla/             Higher-order Linear Attention — O(1) inference cache (Plan 057, SIMD Plan 060) ⎔
    mod.rs          Module root
    types.rs        HlaQHeadState, HlaLayerState, MultiLayerHlaCache, AhlaQHeadState, AhlaLayerState, MultiLayerAhlaCache, HlaVariant
    kernel.rs       hla_state_update, hla_readout, hla_denom, ahla_step, ahla_denom, hla_layer_update, hla_layer_readout, ahla_layer_step
    forward.rs      forward_hla, forward_ahla, generate_hla_into, generate_ahla_into
  sp_kv/           Self-Pruned Key-Value Attention (Plan 070) §
    mod.rs          Module root
    types.rs        SpKvGateMode, SpKvConfig, SpKvLayerCache, SpKvCache, UtilityPredictorWeights, SpKvPredictors, GateBiasBuffer
    utility_predictor.rs  predict, predict_single_head, soft_gate_bias, hard_gate_bias, tahg_gate_bias, UtilityAggregation
    forward.rs      SpKvForwardContext, BiasProvider trait, attention_head_core, attention_head_gated, forward_sp_kv

  alloc.rs          Debug-only TrackingAllocator, reset_alloc_stats, get_alloc_stats (debug builds)

  * behind --features sudoku
  ∘ behind --features sparse_mlp    (default)
  ○ behind --features ppot           (default)
  ‡ behind --features validator
  ♭ behind --features bandit         (default)
  ⍟ behind --features bomber         (bevy_ecs + bandit)
  ✦ behind --features monopoly       (bevy_ecs + bandit)
  ✧ behind --features fft            (bandit)
  ⛩ behind --features go             (bandit + reqwest)
  ⌘ behind --features delta_mem      (bandit)
  ǂ behind --features g_zero         (bandit)
  ⌁ behind --features feedback
  ⎔ behind --features hla_attention
  § behind --features sp_kv
  ⌂ behind --features dllm
  ≋ behind --features stepcode
  ⎗ behind --features game_state
```

## Feature Flags

| Flag | Dependencies | Description |
|------|-------------|-------------|
| `sparse_mlp` | — | TwELL-inspired sparse MLP matmul (Plan 022) |
| `ppot` | — | PPoT logit-parameterized CPU resampling + adaptive rescue (Plans 026 + 027) |
| `domain_latent` | — | Free Transformer mid-layer domain conditioning (Plan 038) |
| `bandit` | — | Multi-armed bandit + HL infrastructure: TrialLog, AbsorbCompress, HotSwapPruner, RegressionSuite, ReviewMetrics (Plans 030–032) |
| `sudoku` | — | SudokuPruner constraint pruning + examples |
| `validator` | `syn`, `proc-macro2` | SynPruner + partial parser |
| `delta_mem` | `bandit` | δ-Mem modelless distillation — associative bandit memory (Plan 053) |
| `g_zero` | `bandit` | G-Zero self-play distillation — Hint-δ gated absorb + bandit (Plan 049) |
| `hla_attention` | — | HLA/AHLA streaming attention kernels (Plan 057, SIMD-accelerated in Plan 060) |
| `fft` | `bandit` | FFT Tactics Arena — ATB battle engine with status effects (Plan 053) |
| `bomber` | `bevy_ecs`, `bandit` | Bomberman HL arena (Plan 033) |
| `bomber-wasm` | `bomber`, `wasmtime`, `papaya` | WASM bomber validator loader + batch pool (Plans 034 + 037) |
| `monopoly` | `bevy_ecs`, `bandit` | Monopoly board game engine (Plan 035) |
| `feedback` | — | E2E feedback loop — sends inference results to REST endpoint (Plan 042) |
| `rest` | — | REST bridge test + merge stub (Plan 009, client lives in riir-ai/riir-rest) |
| `embedding_router` | — | Semantic embedding routing (Plan 024) |
| `game_domain` | `domain_latent` | Alias for domain_latent — game-specific Config presets (Plan 040) |
| `language_domain` | — | Language domain: BPE vocab, LLM models (Plan 040) |
| `gpu` | — | Placeholder — GPU training lives in riir-ai/riir-gpu |
| `go` | `bandit`, `reqwest` | Go GameState + AutoGo API bridge + tournament + G-Zero self-play + AutoResearch loop (Plan 065) |
| `sp_kv` | — | SP-KV self-pruned key-value attention (Plan 070) |
| `dllm` | — | D2F Discrete Diffusion Forcing — mini dLLM research (Plan 066) |
| `stepcode` | `bandit` | Path shaping + consistency scoring (Plan 054, infrastructure only, no perf gain) |
| `bomber-agent` | `bomber` | Coding agent validator loop (Issue 052) |
| `game_state` | `bomber` | GameState forward model trait + generic MCTS (Plan 056) |
| `bandit_mcts` | `game_state` | Bandit-guided MCTS rollout policy — NFSP/MCTS duality (Plan 067) |
| `percepta` | `ordered-float` | CHT hull cache: upper+lower, HullMeta, tie-break, cumsum |
| `percepta_gates` | `percepta` | + ReGLU, stepglu, multiply, persist primitives |
| `percepta_graph` | `percepta_gates` | + Expression/Dimension DSL, ProgramGraph |
| `percepta_wasm` | `percepta_graph` | + WASM decoder + lowering + interpreter (pure Rust) |
| `percepta_compile` | `percepta_wasm`, `good_lp` | + MILP scheduling + weights + transformer + Futamura |
| `full` | all above | Enable all features |

Default features: `sparse_mlp`, `domain_latent`, `ppot`, `bandit` (production best perf + accuracy, Plan 051).

## Quick Start

```bash
cargo test --quiet --workspace --all-features   # Run all 400+ tests
cargo run --release                             # Run benchmark suite (includes Leviathan verification)
cargo run --example hello_py2rs                                # BPE + bidirectional prefill demo
cargo run --example sudoku_01_9x9 --features sudoku           # Sudoku streaming solver
cargo run --example sudoku_02_speculative --features sudoku   # DDTree pruning demo
cargo run --example sudoku_03_tui --features sudoku           # TUI visualization
cargo run --example core_01_validator --features validator     # SynPruner + DDTree pipeline
cargo run --example core_02_raven                             # Raven RSM demo
cargo run --example core_03_ppot --features ppot              # PPoT resampling demo
cargo run --example core_04_prefill                           # PFlash prefill demo
cargo run --example bandit_01_basic --features bandit         # Bandit basics
cargo run --example bomber_01_arena --features bomber         # Bomberman arena
cargo run --example monopoly_01_arena --features monopoly     # Monopoly arena
cargo run --example fft_01_arena --features fft               # FFT Tactics arena
cargo run --example go_06_bench --features go --release       # Go benchmark suite
```

## Config Presets

| Config | vocab | embd | heads | layers | mlp | Purpose |
|--------|-------|------|-------|--------|-----|---------|
| `micro` | 27 | 16 | 4 | 1 | 64 | Default benchmark target |
| `micro_lora` | 27 | 16 | 4 | 1 | 64 | Micro + LoRA adapter support |
| `draft` | 27 | 4 | 2 | 1 | 16 | Tiny draft model |
| `game` | 27 | 16 | 4 | 1 | 64 | Game domain preset (domain_latent) |
| `bpe` | 4096 | 32 | 4 | 1 | 128 | BPE Rust code model |
| `bpe_draft` | 4096 | 8 | 2 | 1 | 32 | BPE draft model |
| `small_target` | 4096 | 64 | 4 | 4 | 256 | Multi-layer target |
| `gqa_draft` | 4096 | 64 | 8 | 4 | 256 | GQA draft (n_kv_head=2) |

## Key Design Principles

1. **Zero allocations on hot paths** — all buffers pre-allocated in `SpeculativeContext` and `ForwardContext`
2. **Feature-gated modularity** — domain code (sudoku, validator) never pollutes core
3. **Trait-based strategy** — `ConstraintPruner`, `SpeculativeVerifier`, `PrefillScorer`, `ScreeningPruner` for swappable behavior
4. **SOLID module decomposition** — each file < 1024 lines, single responsibility
5. **`mod.rs` for index only**, minimal `main.rs`/`lib.rs`
6. **Unsafe only in verified hot-path kernels** with `get_unchecked` + `#[inline(always)]` + SIMD intrinsics (`core::arch` NEON/AVX2)

## Related Documentation

| # | Document | Topic |
|---|----------|-------|
| 01 | `01_overview.md` | Overview & reference card (this file) |
| 02 | `02_architecture.md` | Architecture details (forward pass, routers, LoRA) |
| 03 | `03_speculative_decoding.md` | Speculative decoding deep-dive |
| 04 | `04_performance.md` | Performance engineering & benchmarks |
| 05 | `05_sudoku.md` | Sudoku solvers |
| 06 | `06_validator.md` | Constraint validator + SynPruner |
| 07 | `07_adaptation.md` | Model adaptation (bidirectional prefill, LoRA switching, sparse MLP, domain latent) |
| 08 | `08_lucebox_techniques.md` | LuceBox techniques |
| 09 | `09_heuristic-learning.md` | Heuristic learning, bandit, HL arena |
| 10 | `10_bomber_arena.md` | Bomberman HL arena (Plan 033) |
| 11 | `11_monopoly_fsm.md` | Monopoly FSM arena (Plan 035) |
| 12 | `12_fft_arena.md` | FFT Tactics Arena (Plan 053) |
| 13 | `13_mtp_threshold_guide.md` | MTP threshold guide (Plan 055) |
| 14 | `14_sp_kv_research.md` | SP-KV research note (Plan 070) |
| 15 | `15_go_arena.md` | Go Arena (Plan 065) |