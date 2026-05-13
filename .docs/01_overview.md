# microgpt-rs: Overview

## What It Is

A from-scratch Rust implementation of a GPT-2 style transformer with speculative decoding, designed as an educational/performance research vehicle. No ML frameworks — just `Vec<f32>`, matmul, and hand-tuned attention kernels.

## Project Goals

- CPU-first inference engine with zero-allocation hot paths
- Speculative decoding pipeline (DDTree + DFlash + Leviathan verification)
- Domain-specific constraint pruning (Sudoku, Rust AST via Validator)
- BPE tokenizer + SynPruner for Rust syntax validation
- Sub-millisecond inference on Apple Silicon

## Current Capabilities

- Single-token autoregressive generation: ~900K tok/s (micro config)
- DFlash marginal prediction: ~4.2M tok/s
- DDTree build: ~431K trees/s
- Speculative decoding: ~1.64M tok/s (AR Draft)
- forward_raven (16 slots): ~1.6M trees/s
- raven_recall (1000 noise): ~9.3M tok/s
- TurboQuant 3-bit KV cache: 5.3× compression, 0.99 attention correlation
- PFlash block-sparse prefill: up to 21.3× sequence reduction, 100% NIAH retrieval
- 295+ tests passing, zero clippy warnings

## Module Structure

```
src/
  lib.rs            Module index
  main.rs           Entry point (proof → bench → Percepta bench → plot)
  types.rs          Config (micro + draft, screening_threshold, sparse_threshold), Rng, softmax, rmsnorm, matmul, matmul_relu, sparse_matmul, sample_token, LoraAdapter, LoraPair, lora_apply
  transformer.rs    TransformerWeights, KVCache, PagedKVCache, RavenKVCache, ForwardContext (+ sparse buffers + lora_buf), PrefillContext, forward, forward_base, forward_prefill, forward_paged, forward_raven, generate, generate_into, generate_batch, generate_with_prefill
  speculative/      SOLID decomposition:
    mod.rs          Re-exports
    types.rs        TreeNode, DraftResult, ConstraintPruner trait, ScreeningPruner trait, NoPruner, NoScreeningPruner, BinaryScreeningPruner, SpeculativeContext, DDTreeBranchCache
    sampling.rs     sample_from_distribution, sample_residual_distribution, sample_residual_distribution_into
    dd_tree.rs      build_dd_tree, build_dd_tree_pruned, build_dd_tree_screened, TreeBuilder, extract_parent_tokens, extract_parent_tokens_into
    dflash.rs       dflash_predict, dflash_predict_with, dflash_predict_ar, dflash_predict_ar_with, dflash_predict_parallel
    verifier.rs     SpeculativeVerifier trait, SimulatedVerifier, LeviathanVerifier
    step.rs         speculative_step, speculative_step_verifier, speculative_step_rollback, speculative_step_conditioned
    prefill.rs      PrefillScorer trait, AttentionScorer, compress_prompt, speculative_prefill, score_with
    ppot/           PPoT (Plans 026 + 027)
      mod.rs        Module root, public API re-exports
      types.rs      TokenRule enum, PpotConfig
      entropy.rs    token_entropy, identify_high_entropy_positions, identify_positions_adaptive
      resample.rs   ppot_rescue, ppot_rescue_adaptive, ppot_resample_multi_strategy
      knowledge.rs  RejectionInsight, SessionKnowledge
      rank.rs       rank_by_consistency, select_best_variant
    sudoku_pruner.rs  SudokuPruner *
    bandit.rs         BanditPruner, BanditSession, BanditEnv, BernoulliEnv, GaussianEnv, BanditStrategy, BanditStats ♭
    trial_log.rs      TrialLog, TrialRecord, TrialSummary ♭
    absorb_compress.rs AbsorbCompress trait, AbsorbCompressLayer, CompressConfig ♭
    hot_swap.rs       HotSwapPruner ♭
    regression.rs     RegressionSuite, GoldenTrace, RegressionResult, ReplayReward ♭
  tokenizer/        BPE tokenizer (encode/decode/train, Config::bpe())
  validator/        SynPruner + partial parser ‡
  percepta.rs       Vec2, KVCache2D, Sudoku9x9, SymbolicValidator, StreamingSolver, SolveEvent
  turboquant/      TurboQuant KV cache compression:
    mod.rs          Module root (re-exports)
    types.rs        TurboQuantCodebook, TurboQuantLayer, TurboQuantKVCacheConfig
    codebook.rs     Lloyd-Max codebook (compute_codebook, quantize, dequantize)
    rotation.rs     QR-based orthogonal rotation + QJL projection
    kv_cache.rs     TurboQuantKVCache (store_key, store_value, dequantize, bit-pack)
    forward.rs      attention_turboquant, dequantize_keys_flat/values_flat, cosine_similarity
  benchmark.rs      BenchResult, run_all, save_results_csv
  plot.rs           plot_results → PNG

  * behind --features sudoku
  ∘ behind --features sparse_mlp
  ○ behind --features ppot
  ‡ behind --features validator
  ♭ behind --features bandit
```

## Feature Flags

| Flag | Dependencies | Description |
|------|-------------|-------------|
| `sudoku` | — | SudokuPruner constraint pruning + examples |
| `validator` | `syn`, `proc-macro2` | SynPruner + partial parser |
| `sparse_mlp` | — | TwELL-inspired sparse MLP matmul (Plan 022) |
| `ppot` | — | PPoT logit-parameterized CPU resampling + adaptive rescue (Plans 026 + 027) |
| `bandit` | — | Multi-armed bandit + HL infrastructure: TrialLog, AbsorbCompress, HotSwapPruner, RegressionSuite (Plans 030–032) |
| `bomber` | `bevy_ecs`, `bandit` | Bomberman HL arena (Plan 033) |
| `monopoly` | `bevy_ecs`, `bandit` | Monopoly FSM arena (Plan 035) |
| `full` | all above | Enable all features |

## Quick Start

```bash
cargo test --quiet --workspace --all-features   # Run all 295+ tests
cargo run --release                             # Run benchmark suite (includes Leviathan verification)
cargo run --example sudoku_01_9x9 --features sudoku           # Sudoku streaming solver
cargo run --example sudoku_02_speculative --features sudoku   # DDTree pruning demo
cargo run --example sudoku_03_tui --features sudoku           # TUI visualization
cargo run --example validator_demo --features validator        # SynPruner + DDTree pipeline
cargo run --example py2rs_hello                                 # BPE + bidirectional prefill demo
```

## Config Presets

| Config | vocab | embd | heads | layers | mlp | Purpose |
|--------|-------|------|-------|--------|-----|---------|
| `micro` | 27 | 16 | 4 | 1 | 64 | Default benchmark target |
| `draft` | 27 | 4 | 2 | 1 | 16 | Tiny draft model |
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
6. **Unsafe only in verified hot-path kernels** with `get_unchecked` + `#[inline(always)]`

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