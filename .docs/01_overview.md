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

- Single-token autoregressive generation: ~1.18M tok/s (micro config)
- DFlash marginal prediction: ~4.1M tok/s
- DDTree build: ~360K trees/s
- Speculative decoding: ~1.48M tok/s (AR Draft)
- 240+ tests passing, zero clippy warnings

## Module Structure

```
src/
├── lib.rs                    # Public API surface
├── main.rs                   # Benchmark runner
├── types.rs                  # Config, Rng, math kernels (matmul, softmax, rmsnorm)
├── transformer.rs            # ForwardContext, TransformerWeights, LayerWeights, forward(), generate()
├── percepta.rs               # Sudoku solvers (4x4, 9x9), StreamingSolver, KVCache2D
├── benchmark.rs              # All benchmark functions
├── plot.rs                   # Plotting utilities (plotters-based)
├── speculative/
│   ├── mod.rs                # Re-exports
│   ├── types.rs              # TreeNode, DraftResult, ConstraintPruner, SpeculativeContext
│   ├── sampling.rs           # sample_from_distribution, sample_residual_distribution
│   ├── dd_tree.rs            # DDTree build (best-first + chain-seed), TreeBuilder
│   ├── dflash.rs             # DFlash predict (marginal, AR, parallel, conditioned)
│   ├── verifier.rs           # SpeculativeVerifier trait, SimulatedVerifier, LeviathanVerifier
│   ├── step.rs               # High-level step functions (speculative_step, rollback, conditioned)
│   ├── prefill.rs            # Speculative prefill scoring + prompt compression
│   └── sudoku_pruner.rs      # SudokuPruner (behind "sudoku" feature)
├── tokenizer/                # BPE tokenizer (behind "validator" feature)
├── validator/                # SynPruner + PartialParser (behind "validator" feature)
└── ppot/                     # PPoT CPU resampling (behind "ppot" feature)
```

## Feature Flags

```toml
[features]
default = []
sudoku = []                         # SudokuPruner + sudoku examples
validator = ["syn", "proc-macro2"]  # BPE tokenizer + SynPruner
sparse_mlp = []                     # TwELL-inspired sparse MLP matmul
ppot = []                           # PPoT logit-parameterized CPU resampling
full = ["sudoku", "validator", "sparse_mlp", "ppot"]
```

## Quick Start

```bash
cargo test --quiet                           # Run all 240+ tests
cargo run --release                          # Run benchmark suite (includes Leviathan verification)
cargo run --example sudoku_9x9 --features sudoku               # Sudoku streaming solver
cargo run --example sudoku_speculative --features sudoku       # DDTree pruning demo
cargo run --example sudoku_tui --features sudoku               # TUI visualization
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
3. **Trait-based strategy** — `ConstraintPruner`, `SpeculativeVerifier`, `PrefillScorer` for swappable behavior
4. **SOLID module decomposition** — each file < 1024 lines, single responsibility
5. **`mod.rs` for index only**, minimal `main.rs`/`lib.rs`
6. **Unsafe only in verified hot-path kernels** with `get_unchecked` + `#[inline(always)]`

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────┐
│                     Benchmark Runner                     │
│                      (main.rs)                          │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌──────────────┐    ┌──────────────────────────────┐  │
│  │  Transformer  │    │      Speculative Pipeline     │  │
│  │  (forward,    │◄──►│  ┌─────┐ ┌───────┐ ┌──────┐  │  │
│  │   generate,   │    │  │DFlash│→│DDTree │→│Verify│  │  │
│  │   weights)    │    │  └─────┘ └───────┘ └──────┘  │  │
│  └──────────────┘    └──────────────────────────────┘  │
│         │                        │                      │
│         ▼                        ▼                      │
│  ┌──────────────┐    ┌──────────────────────────────┐  │
│  │  Types/Kernel │    │     Constraint Pruners        │  │
│  │  (matmul,     │    │  ┌────────┐ ┌──────────────┐ │  │
│  │   softmax,    │    │  │Sudoku  │ │Validator(plan)│ │  │
│  │   rmsnorm)    │    │  └────────┘ └──────────────┘ │  │
│  └──────────────┘    └──────────────────────────────┘  │
│                                                         │
│  ┌──────────────┐    ┌──────────────────────────────┐  │
│  │  Percepta     │    │       Validator / BPE          │  │
│  │  (Sudoku      │    │  (SynPruner, PartialParser,    │  │
│  │   solvers)    │    │   BPE tokenizer)               │  │
│  └──────────────┘    └──────────────────────────────┘  │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

## Related Documentation

| # | Document | Topic |
|---|----------|-------|
| 01 | `01_sudoku_9x9_example.md` | 9×9 Sudoku streaming solver |
| 02 | `02_dynamic_pruning.md` | Dynamic constraint pruning |
| 03 | `03_perf_optimization.md` | Performance optimization notes |
| 04 | `04_leviathan_distill.md` | Leviathan verification distillation |
| 05 | `05_speculative_module_refactor.md` | Speculative module design |
| 06 | `06_sudoku_tui.md` | TUI visualization |
| 07 | `07_compiler_in_the_loop_validator.md` | Validator compiler-in-the-loop |
| 08 | `08_lucebox_techniques.md` | LuceBox techniques |
| 09 | — | *(GPU training docs moved to riir-ai)* |
| 10 | `06_validator.md` | Constraint validator + SynPruner |
| 11 | `04_performance.md` | Performance engineering |
| 12 | `05_sudoku.md` | Sudoku solvers |
| 13 | `02_architecture.md` | Architecture details |
| 14 | `03_speculative_decoding.md` | Speculative decoding deep-dive |