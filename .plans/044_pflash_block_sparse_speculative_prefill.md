# Plan 044: PFlash Block-Sparse Speculative Prefill — Metal-Accelerated Long-Context TTFT Reduction

> **Raw reference:** `.raw/lucebox-hub/pflash/` + `.raw/lucebox-hub/dflash/src/flashprefill*`
> **GPU infra:** `riir-ai/crates/riir-gpu/` (wgpu → Metal on macOS, unified memory)
> **Related:** Plan 043 (TurboQuant KV), Plan 020 (Raven RSM), `.docs/08_lucebox_techniques.md`
> **Branch:** `develop/feature/044_pflash_block_sparse_prefill`

---

## Tasks

- [x] **Task 1: Benchmark baseline prefill scoring (CPU token-level `AttentionScorer` + GPU `attention_score.wgsl`)**
- [x] **Task 2: Add `FlashPrefillConfig` struct to `types.rs`**
- [x] **Task 3: Implement block selection logic in Rust (sink + window + last_n_full + alpha threshold)**
- [x] **Task 4: Write `flashprefill_mean_k.wgsl` — GPU kernel: per-block mean K-vector**
- [x] **Task 5: Write `flashprefill_block_score.wgsl` — GPU kernel: Q_tail × mean_K block scores**
- [x] **Task 6: Write `flashprefill_block_select.wgsl` — GPU kernel: threshold + rule-based selection**
- [x] **Task 7: Write `flashprefill_sparse_forward.wgsl` — GPU kernel: block-sparse attention forward**
- [x] **Task 8: Wire GPU PFlash pipeline in `riir-gpu/src/forward.rs` + CPU fallback in `prefill.rs`**
- [x] **Task 9: Add NIAH needle-retrieval quality benchmarks (CPU + GPU paths)**
- [x] **Task 10: Update `.docs/08_lucebox_techniques.md` and commit**

---

## Overview

PFlash (Speculative Prefill) compresses long prompts before target prefill by scoring token importance with a small drafter model. The lucebox-hub C++ implementation achieves **10.4× TTFT reduction** (128K → 2.6K tokens) on a single RTX 3090 with NIAH retrieval preserved.

**We have Metal.** The `riir-gpu` crate already provides:
- wgpu → Metal backend on macOS (Apple Silicon unified memory, zero-copy CPU↔GPU)
- WGSL compute shaders: matmul, softmax, layernorm, `attention_score.wgsl` with full KV cache
- `GpuForwardPass` with per-layer QKV projection, attention dispatch, LoRA integration
- `GpuWeightBuffers` / `GpuActivationBuffers` for GPU-resident model weights

This plan ports the C++ PFlash's 4-kernel pipeline (`mean_K → block_score → block_select → sparse_forward`) to WGSL compute shaders, giving us the same GPU-accelerated block-sparse scoring on Apple Silicon. The C++ uses CUDA WMMA/BSA on NVIDIA; we use wgpu compute shaders on Metal. Same algorithm, different metal.

### Why This Matters

1. **Prefill is the long-context bottleneck.** 128K tokens × 27B target → ~257s TTFT (llama.cpp). Compressing to 2.6K → ~24.8s.
2. **Metal unified memory = zero transfer overhead.** Unlike CUDA's PCIe copy, Apple Silicon's CPU and GPU share the same RAM. KV cache stays put; GPU kernels read it directly.
3. **GPU block scoring is O(S/block_size) dispatches, not O(S).** The C++ PFlash computes `mean_K` per block in one kernel, `Q_block @ mean_K` in another. On Metal, this is parallel across all blocks simultaneously.
4. **Composable with TurboQuant (Plan 043).** PFlash compresses the *sequence* dimension (fewer tokens). TurboQuant compresses the *precision* dimension (fewer bits per token). Both together → 128K context in ~15 MB.
5. **Quality-preserving by construction.** Block selection rules (sink + window + last_n_full) guarantee system prompt, recent context, and final positions are always kept regardless of score noise.

### Compression Target

| Metric | Before PFlash | After PFlash | Ratio |
|--------|---------------|--------------|-------|
| Prefill tokens (128K ctx) | 131,072 | ~2,621 (keep_ratio=0.02) | 50× |
| TTFT (27B Q4_K_M, RTX 3090) | ~257s | ~24.8s | 10.4× |
| Scoring GPU dispatches (128K) | N/A (no GPU path) | ~4 (mean_K + score + select + forward) | — |
| Draft model memory | — | ~1.2 GB (Qwen3-0.6B BF16) | one-time |
| CPU→GPU data transfer | N/A | 0 bytes (unified memory) | 0× |

For our draft model (head_dim=4, 2 layers): GPU block scoring at block_size=32 is 4096× parallel across blocks.

---

## Architecture

### GPU Data Flow (Metal via wgpu)

```
Prompt tokens [0..S) — GPU buffer (already resident in unified memory)
     │
     ▼
┌──────────────────────────────────────────────────────────────┐
│  GPU Kernel 1: flashprefill_mean_k.wgsl                     │
│   Input:  K cache [S, n_kv_head, head_dim]                  │
│   Output: mean_K [M, n_kv_head, head_dim]                   │
│   Where M = ceil(S / block_size)                             │
│   Per block: sum K vectors → divide by block_size            │
│   Parallel: one workgroup per (block, head)                  │
│   Cost:   1 dispatch, ~M × Hk workgroups                     │
└──────────────────────────────────────────────────────────────┘
     │
     ▼
┌──────────────────────────────────────────────────────────────┐
│  GPU Kernel 2: flashprefill_block_score.wgsl                │
│   Input:  Q cache [S, n_head, head_dim], mean_K [M, Hk, D]  │
│   Output: block_scores [M, M, n_head] (float)               │
│   Per (q_block, k_block, head): Q_mean @ mean_K^T × scale   │
│   Tail-window: average over last tail_window q_blocks        │
│   Parallel: one workgroup per (q_block, head)                │
│   Cost:   1 dispatch, ~M × H workgroups                      │
└──────────────────────────────────────────────────────────────┘
     │
     ▼
┌──────────────────────────────────────────────────────────────┐
│  GPU Kernel 3: flashprefill_block_select.wgsl               │
│   Input:  block_scores [M, M, n_head], FlashPrefillConfig   │
│   Output: selected_indices [M, M, H] (i32, -1 = pad)        │
│           selected_counts [M, H] (i32)                       │
│   Rules per (q_block, head):                                 │
│    - sink:     k_block < attention_sink → keep                │
│    - window:   |q - k| < window → keep                       │
│    - last_n:   q >= M - last_n_full → keep all               │
│    - alpha:    score >= max_score * alpha → keep              │
│   Parallel: one workgroup per (q_block, head)                │
│   Cost:   1 dispatch, ~M × H workgroups                      │
└──────────────────────────────────────────────────────────────┘
     │
     ▼
┌──────────────────────────────────────────────────────────────┐
│  GPU Kernel 4: flashprefill_sparse_forward.wgsl             │
│   Input:  Q, K, V caches + selected_indices + counts         │
│   Output: O [S, n_head, head_dim] (sparse attention output)  │
│   Per (q_block, head): attend only to selected k_blocks      │
│   Online softmax over sparse K blocks                        │
│   Parallel: one workgroup per (q_block, head)                │
│   Cost:   1 dispatch, ~M × H workgroups                      │
└──────────────────────────────────────────────────────────────┘
     │
     ▼
  compress_prompt_blocks (CPU) — flatten selected blocks → token indices
     │
     ▼
  Target model prefill on compressed tokens
```

### CPU Fallback Data Flow (no GPU available)

```
Prompt tokens [0..S)
     │
     ▼
┌──────────────────────────────────────────────────┐
│  BlockAttentionScorer (CPU, prefill.rs)          │
│   Single forward pass over all tokens            │
│   Aggregate per-block attention scores           │
│   Upsample to per-token scores                   │
└──────────────────────────────────────────────────┘
     │
     ▼
  block_select (CPU) — same selection rules as GPU
     │
     ▼
  compress_prompt_blocks (CPU) — flatten → token indices
     │
     ▼
  Target model prefill on compressed tokens
```

### New Types

```rust
// speculative/types.rs

/// Algorithmic parameters for block-sparse FlashPrefill scoring.
/// Ported from dflash/src/flashprefill.h FlashPrefillConfig.
/// Shared between CPU and GPU paths.
#[derive(Debug, Clone)]
pub struct FlashPrefillConfig {
    /// K/Q block stride in tokens.
    /// C++ default=128 (CUDA). Metal sweet spot: 64 (matches Apple GPU workgroup sizing).
    /// CPU sweet spot: 32 (finer granularity at shorter prompts).
    pub block_size: usize,
    /// First N k-blocks always selected (attention sinks).
    /// Preserves system prompt / instruction prefix.
    pub attention_sink: usize,
    /// Last `window` k-blocks before query always selected.
    /// Preserves recent context (rolling conversation tail).
    pub window: usize,
    /// Last N q-blocks attend to ALL selected blocks.
    /// Ensures final positions have full context for generation start.
    pub last_n_full: usize,
    /// Dynamic top-K threshold: score >= max_score * alpha → keep.
    /// Higher = stricter = fewer blocks. 0.12 = permissive, 0.85 = strict.
    pub alpha: f32,
    /// Number of tail query positions to average for scoring.
    /// More positions = smoother scores, less noise.
    pub tail_window: usize,
}

/// When to apply speculative prefill compression.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrefillMode {
    /// Never compress — standard full prefill.
    Off,
    /// Compress only when prompt length >= threshold tokens.
    Auto,
    /// Compress every prompt (testing / benchmarking).
    Always,
}

/// Block importance scores: one score per block of tokens.
#[derive(Debug, Clone)]
pub struct BlockScores {
    /// Number of blocks = ceil(seq_len / block_size).
    pub num_blocks: usize,
    /// Block size in tokens.
    pub block_size: usize,
    /// Per-block importance score [num_blocks].
    pub scores: Vec<f32>,
    /// Which blocks are selected (indices into [0..num_blocks)).
    pub selected: Vec<usize>,
}
```

```rust
// riir-ai/crates/riir-gpu/src/forward.rs (additions)

/// GPU-side FlashPrefill buffers.
/// Allocated once, reused across scoring calls.
pub struct GpuFlashPrefillBuffers {
    /// mean_K: [M, n_kv_head, head_dim] f32
    pub mean_k: Buffer,
    /// block_scores: [M, M, n_head] f32
    pub block_scores: Buffer,
    /// score_max: [M, M, n_head] f32 (max per row for normalization)
    pub score_max: Buffer,
    /// selected_indices: [M, M, n_head] i32 (-1 = padding)
    pub selected_indices: Buffer,
    /// selected_counts: [M, n_head] i32
    pub selected_counts: Buffer,
    /// sparse output: [S, n_head, head_dim] f32
    pub sparse_output: Buffer,
    /// uniform config buffer
    pub uniform_config: Buffer,
}

/// GPU dispatch for the 4-kernel PFlash pipeline.
pub struct GpuFlashPrefillPass {
    pub ctx: GpuContext,
    pub config: Config,
    pub buffers: GpuFlashPrefillBuffers,
    pipelines: FlashPrefillPipelines,
}

struct FlashPrefillPipelines {
    mean_k: GpuPipeline,
    block_score: GpuPipeline,
    block_select: GpuPipeline,
    sparse_forward: GpuPipeline,
}
```

### Default Config Values

| Parameter | C++ (CUDA, 27B target) | RS Metal (Apple Silicon) | RS CPU Fallback | Rationale |
|-----------|------------------------|--------------------------|-----------------|-----------|
| `block_size` | 128 | 64 | 32 | Metal workgroups prefer 64; CPU prefers 32 for granularity |
| `attention_sink` | 2 (256 tok) | 1 (64 tok) | 1 (32 tok) | Preserve first block as system prompt anchor |
| `window` | 4 (512 tok) | 2 (128 tok) | 2 (64 tok) | Recent context window |
| `last_n_full` | 2 | 1 | 1 | Last block gets full attention |
| `alpha` | 0.12 → 0.85 | 0.15 (default) | 0.15 (default) | Start permissive, tune via benchmark |
| `tail_window` | N/A (final Q) | 4 | 4 | Average last 4 query positions |

### Metal vs CUDA Comparison for PFlash

| Aspect | CUDA (C++ PFlash) | Metal (Our WGSL PFlash) |
|--------|-------------------|--------------------------|
| Backend | CUDA 12+, WMMA/BSA | wgpu → Metal 3, compute shaders |
| Memory model | Device memory, PCIe copy | Unified memory, zero-copy |
| Block scoring | Custom WMMA m16n16k16 | WGSL workgroup shared memory + SIMD |
| Block selection | GPU kernel or host fallback | GPU compute shader (parallel per q_block/head) |
| Sparse forward | BSA (FA-2 derived) or WMMA fallback | Custom WGSL sparse attention kernel |
| Data transfer | cudaMemcpy Device↔Host | None (unified memory) |
| Precision | BF16 weights/activations | f32 (can add f16 later) |
| Dispatch overhead | ~5-10μs per kernel launch | ~3-8μs per compute pass (Metal is lower overhead than CUDA) |

---

## Tasks

### Task 1: Benchmark Baseline Prefill Scoring

Add baseline benchmarks for both CPU and GPU paths:

```rust
// tests/bench_prefill.rs (new)

/// Benchmark: CPU AttentionScorer (token-level) time vs prompt length.
fn bench_cpu_attention_scorer_by_length() {
    // Sweep: 8, 32, 128, 512, 1024 tokens
    // Report: scoring time (μs), tokens scored
}

/// Benchmark: GPU attention_score.wgsl (existing kernel) time vs prompt length.
fn bench_gpu_attention_score_by_length() {
    // Same sweep, using GpuForwardPass dispatch_attention
    // Report: scoring time (μs), GPU→CPU download time (should be ~0 on unified mem)
}

/// Benchmark: NIAH retrieval rate vs keep_ratio.
fn bench_niah_retrieval_rate() {
    // Generate synthetic [hay × N] + [needle] + [hay × N]
    // Sweep: keep_ratio [0.02, 0.05, 0.10, 0.20]
    // Report: needle survival rate (%), compression ratio
}
```

**Acceptance:** Baseline numbers recorded for CPU and GPU paths at multiple prompt lengths.

### Task 2: Add `FlashPrefillConfig` Struct

Add `FlashPrefillConfig`, `PrefillMode`, and `BlockScores` to `speculative/types.rs`.

```rust
// speculative/types.rs

impl Default for FlashPrefillConfig {
    fn default() -> Self {
        Self {
            block_size: 32,
            attention_sink: 1,
            window: 2,
            last_n_full: 1,
            alpha: 0.15,
            tail_window: 4,
        }
    }
}

impl FlashPrefillConfig {
    /// Config for GPU path (Metal). Larger blocks for GPU parallelism.
    pub fn metal() -> Self {
        Self {
            block_size: 64,
            attention_sink: 1,
            window: 2,
            last_n_full: 1,
            alpha: 0.15,
            tail_window: 4,
        }
    }

    /// Config tuned for long-context compression (keep_ratio ≤ 0.05).
    pub fn long_context() -> Self {
        Self {
            block_size: 64,
            attention_sink: 1,
            window: 2,
            last_n_full: 1,
            alpha: 0.85,
            tail_window: 8,
        }
    }

    /// Config for short/medium prompts (keep_ratio 0.1–0.3).
    pub fn short_context() -> Self {
        Self {
            block_size: 32,
            attention_sink: 1,
            window: 2,
            last_n_full: 1,
            alpha: 0.12,
            tail_window: 2,
        }
    }
}
```

**Acceptance:** Structs compile, `Default` provides sensible values, tests for construction.

### Task 3: Implement Block Selection Logic in Rust

Port `block_select` from `dflash/src/flashprefill_select.cpp`. This runs on CPU even in the GPU path (selection output is small — ~1100×16 entries at 128K context).

```rust
// speculative/prefill.rs

/// Block selection: turns per-(q_block, k_block, head) scores into
/// selected block indices per (q_block, head).
///
/// Rules (from FlashPrefill / C++ block_select_host):
///   - sink:       k_block < attention_sink → always include
///   - window:     |q_block - k_block| < window → include (recent context)
///   - last_full:  q_block >= M - last_n_full → include all blocks
///   - alpha:      score >= max_score * alpha → include (importance threshold)
///   - causal:     k_block <= q_block
///   - sorted:     output indices in ascending order
///
/// For prefill scoring, we use the "last block as query" perspective:
/// q_block = last block, score all k_blocks. This gives us which source
/// blocks matter for the generation-start position.
///
/// Returns indices of selected blocks in [0..num_blocks).
pub fn block_select(
    block_scores: &[f32],
    cfg: &FlashPrefillConfig,
) -> Vec<usize> {
    let num_blocks = block_scores.len();
    if num_blocks == 0 {
        return Vec::new();
    }

    let q_block = num_blocks - 1;

    // Find max score
    let max_score = block_scores.iter().cloned().fold(0.0f32, f32::max);
    let threshold = max_score * cfg.alpha;

    let mut selected: Vec<usize> = Vec::with_capacity(num_blocks);

    for k_block in 0..num_blocks {
        // Causal: k <= q
        if k_block > q_block {
            continue;
        }

        let mut keep = false;

        // Rule 1: Attention sink — always keep first N blocks
        if k_block < cfg.attention_sink {
            keep = true;
        }

        // Rule 2: Local window — keep blocks near the query
        if q_block.abs_diff(k_block) < cfg.window {
            keep = true;
        }

        // Rule 3: Last N query blocks get full attention
        if q_block >= num_blocks.saturating_sub(cfg.last_n_full) {
            keep = true;
        }

        // Rule 4: Alpha threshold — keep high-importance blocks
        if block_scores[k_block] >= threshold {
            keep = true;
        }

        if keep {
            selected.push(k_block);
        }
    }

    selected.sort();
    selected.dedup();
    selected
}

/// Full block-selection with per-(q_block, k_block, head) score grid.
/// Ported from dflash/src/flashprefill_select.cpp block_select_host.
///
/// `score`: [M][N][H] row-major (M=q_blocks, N=k_blocks, H=heads).
/// Returns selected indices per (q_block, head) and counts.
pub fn block_select_grid(
    score: &[f32],
    num_q_blocks: usize,
    num_k_blocks: usize,
    num_heads: usize,
    cfg: &FlashPrefillConfig,
) -> (Vec<i32>, Vec<i32>) {
    let m = num_q_blocks;
    let n = num_k_blocks;
    let h = num_heads;

    let mut idx_out = vec![-1i32; m * n * h];
    let mut cnt_out = vec![0i32; m * h];

    for q in 0..m {
        let last_full = q >= m.saturating_sub(cfg.last_n_full);

        for head in 0..h {
            // Find max score for this (q, head) across k in [0, q]
            let mut max_score: f32 = -f32::INFINITY;
            for k in 0..=q.min(n - 1) {
                let v = score[q * n * h + k * h + head];
                if v > max_score {
                    max_score = v;
                }
            }
            let thresh = max_score * cfg.alpha;

            let mut selected = Vec::with_capacity(n);
            for k in 0..=q.min(n - 1) {
                let mut keep = false;

                if k < cfg.attention_sink {
                    keep = true;
                }
                if q.abs_diff(k) < cfg.window {
                    keep = true;
                }
                if last_full {
                    keep = true;
                }
                if !keep {
                    let v = score[q * n * h + k * h + head];
                    if v >= thresh {
                        keep = true;
                    }
                }
                if keep {
                    selected.push(k as i32);
                }
            }

            selected.sort();

            let idx_row = &mut idx_out[q * n * h + head..];
            for (i, &sel) in selected.iter().enumerate() {
                idx_row[i * h] = sel;
            }
            cnt_out[q * h + head] = selected.len() as i32;
        }
    }

    (idx_out, cnt_out)
}
```

Also add CPU fallback scorer:

```rust
// speculative/prefill.rs

/// Block-sparse attention scorer (CPU fallback).
///
/// Single forward pass over all tokens, then block-level aggregation.
/// Used when GPU is not available.
pub struct BlockAttentionScorer {
    pub config: FlashPrefillConfig,
}

impl BlockAttentionScorer {
    pub fn score_with(
        &self,
        sctx: &mut SpeculativeContext,
        draft_weights: &TransformerWeights,
        draft_config: &Config,
        prompt_tokens: &[usize],
        scores: &mut [f32],
    ) {
        let block_size = self.config.block_size;
        let seq_len = prompt_tokens.len();
        let num_blocks = (seq_len + block_size - 1) / block_size;

        if seq_len == 0 {
            return;
        }

        sctx.cache.reset();

        // Single forward pass to build KV cache
        let filled = seq_len.min(draft_config.block_size);
        for (pos, &token) in prompt_tokens.iter().enumerate().take(filled) {
            let _logits = forward(
                &mut sctx.ctx, draft_weights, &mut sctx.cache,
                token, pos, draft_config,
            );
        }

        // Aggregate attention scores per block
        let mut block_scores = vec![0.0f32; num_blocks];
        let mut block_counts = vec![0usize; num_blocks];

        let tail_start = seq_len.saturating_sub(self.config.tail_window * block_size);
        for pos in tail_start..filled {
            let score = sctx.ctx.scores[pos];
            let block_idx = pos / block_size;
            if block_idx < num_blocks {
                block_scores[block_idx] += score;
                block_counts[block_idx] += 1;
            }
        }

        // Normalize
        for i in 0..num_blocks {
            if block_counts[i] > 0 {
                block_scores[i] /= block_counts[i] as f32;
            }
        }

        let max_block = block_scores.iter().cloned().fold(0.0f32, f32::max);
        if max_block > 0.0 {
            for s in &mut block_scores {
                *s /= max_block;
            }
        }

        // Upsample to per-token
        for (pos, slot) in scores.iter_mut().enumerate().take(seq_len) {
            *slot = block_scores[pos / block_size];
        }
    }
}

impl PrefillScorer for BlockAttentionScorer {
    fn score(
        &self,
        draft_weights: &TransformerWeights,
        draft_config: &Config,
        prompt_tokens: &[usize],
    ) -> Vec<f32> {
        let mut sctx = SpeculativeContext::new(draft_config);
        let mut scores = vec![0.0f32; prompt_tokens.len()];
        self.score_with(&mut sctx, draft_weights, draft_config, prompt_tokens, &mut scores);
        scores
    }
}
```

**Acceptance:** `block_select` and `block_select_grid` pass all selection rule tests. `BlockAttentionScorer` produces per-token scores via block aggregation.

### Task 4: Write `flashprefill_mean_k.wgsl`

First GPU kernel: compute mean K-vector per block.

```wgsl
// riir-ai/crates/riir-gpu/src/kernels/flashprefill_mean_k.wgsl

// Kernel 1: Compute mean K-vector per block.
// Input:  K cache [seq_len, n_kv_head, head_dim]
// Output: mean_K [num_blocks, n_kv_head, head_dim]
// One workgroup per (block_idx, kv_head).

struct FlashPrefillConfig {
    block_size: u32,       // tokens per block
    attention_sink: u32,
    window: u32,
    last_n_full: u32,
    alpha: f32,
    tail_window: u32,
    seq_len: u32,          // total sequence length
    num_blocks: u32,       // ceil(seq_len / block_size)
    n_kv_head: u32,
    head_dim: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<storage, read>       k_cache: array<f32>;     // [seq_len * n_kv_head * head_dim]
@group(0) @binding(1) var<storage, read_write> mean_k: array<f32>;      // [num_blocks * n_kv_head * head_dim]
@group(0) @binding(2) var<uniform>             config: FlashPrefillConfig;

@compute @workgroup_size(64)
fn flashprefill_mean_k(@builtin(global_invocation_id) gid: vec3<u32>) {
    let block_idx = gid.x;
    let kv_head = gid.y;

    if (block_idx >= config.num_blocks || kv_head >= config.n_kv_head) {
        return;
    }

    let block_start = block_idx * config.block_size;
    let block_end = min(block_start + config.block_size, config.seq_len);
    let count = block_end - block_start;

    if (count == 0u) { return; }

    let kv_stride = config.n_kv_head * config.head_dim;

    for (var d = 0u; d < config.head_dim; d = d + 1u) {
        var sum: f32 = 0.0;
        for (var t = block_start; t < block_end; t = t + 1u) {
            let k_offset = t * kv_stride + kv_head * config.head_dim + d;
            sum = sum + k_cache[k_offset];
        }
        let out_offset = block_idx * config.n_kv_head * config.head_dim
                       + kv_head * config.head_dim + d;
        mean_k[out_offset] = sum / f32(count);
    }
}
```

**Acceptance:** Kernel compiles. Unit test: known K cache → expected mean_K output.

### Task 5: Write `flashprefill_block_score.wgsl`

Second GPU kernel: compute Q_tail × mean_K block importance scores.

```wgsl
// riir-ai/crates/riir-gpu/src/kernels/flashprefill_block_score.wgsl

// Kernel 2: Compute block importance scores.
// Input:  Q cache [seq_len, n_head, head_dim], mean_K [num_blocks, n_kv_head, head_dim]
// Output: block_scores [num_blocks, num_blocks, n_head]
//
// For each (q_block, k_block, head):
//   score = sum_d(Q_mean[q_block, head, d] * mean_K[k_block, kv_group, d]) * scale
//
// Then reduce: max over heads, mean over tail q_blocks → per-k_block score.
// For prefill selection we use tail_window q_blocks from the end.

struct FlashPrefillConfig {
    block_size: u32,
    attention_sink: u32,
    window: u32,
    last_n_full: u32,
    alpha: f32,
    tail_window: u32,
    seq_len: u32,
    num_blocks: u32,
    n_kv_head: u32,
    head_dim: u32,
    n_head: u32,
    scale: f32,           // 1.0 / sqrt(head_dim)
    _pad0: u32,
}

@group(0) @binding(0) var<storage, read>       q_cache: array<f32>;       // [seq_len * n_head * head_dim]
@group(0) @binding(1) var<storage, read>       mean_k: array<f32>;        // [num_blocks * n_kv_head * head_dim]
@group(0) @binding(2) var<storage, read_write> block_scores: array<f32>;  // [num_blocks * num_blocks * n_head]
@group(0) @binding(3) var<uniform>             config: FlashPrefillConfig;

@compute @workgroup_size(64)
fn flashprefill_block_score(@builtin(global_invocation_id) gid: vec3<u32>) {
    let q_block = gid.x;
    let head = gid.y;

    if (q_block >= config.num_blocks || head >= config.n_head) {
        return;
    }

    let kv_group = head * config.n_kv_head / config.n_head;
    let q_stride = config.n_head * config.head_dim;
    let block_start = q_block * config.block_size;
    let block_end = min(block_start + config.block_size, config.seq_len);
    let count = block_end - block_start;

    if (count == 0u) { return; }

    // Compute mean Q for this (q_block, head)
    var mean_q: array<f32, 256> = array<f32, 256>(/* zero init */);
    // Note: head_dim <= 256 for our models. For larger, use workgroup shared memory.
    for (var d = 0u; d < config.head_dim; d = d + 1u) {
        var sum: f32 = 0.0;
        for (var t = block_start; t < block_end; t = t + 1u) {
            sum = sum + q_cache[t * q_stride + head * config.head_dim + d];
        }
        mean_q[d] = sum / f32(count);
    }

    // Score against each k_block
    for (var k_block = 0u; k_block <= q_block; k_block = k_block + 1u) {
        var dot: f32 = 0.0;
        for (var d = 0u; d < config.head_dim; d = d + 1u) {
            let mk_offset = k_block * config.n_kv_head * config.head_dim
                          + kv_group * config.head_dim + d;
            dot = dot + mean_q[d] * mean_k[mk_offset];
        }
        let score = dot * config.scale;
        let out_idx = q_block * config.num_blocks * config.n_head
                    + k_block * config.n_head + head;
        block_scores[out_idx] = score;
    }
}
```

**Acceptance:** Kernel compiles. Unit test: known Q + mean_K → expected block_scores.

### Task 6: Write `flashprefill_block_select.wgsl`

Third GPU kernel: apply selection rules to block scores.

```wgsl
// riir-ai/crates/riir-gpu/src/kernels/flashprefill_block_select.wgsl

// Kernel 3: Block selection with heuristic rules.
// Input:  block_scores [M, M, H]
// Output: selected_indices [M, M, H] (i32, -1 = padding)
//         selected_counts [M, H] (i32)
//
// Rules per (q_block, head):
//   - sink:     k_block < attention_sink → keep
//   - window:   |q - k| < window → keep
//   - last_n:   q >= M - last_n_full → keep all
//   - alpha:    score >= max_score * alpha → keep
//   - causal:   k <= q

struct FlashPrefillConfig {
    block_size: u32,
    attention_sink: u32,
    window: u32,
    last_n_full: u32,
    alpha: f32,
    tail_window: u32,
    seq_len: u32,
    num_blocks: u32,
    n_kv_head: u32,
    head_dim: u32,
    n_head: u32,
    scale: f32,
    _pad0: u32,
}

@group(0) @binding(0) var<storage, read>       block_scores: array<f32>;       // [M * M * H]
@group(0) @binding(1) var<storage, read_write> selected_indices: array<i32>;   // [M * M * H]
@group(0) @binding(2) var<storage, read_write> selected_counts: array<i32>;    // [M * H]
@group(0) @binding(3) var<uniform>             config: FlashPrefillConfig;

@compute @workgroup_size(64)
fn flashprefill_block_select(@builtin(global_invocation_id) gid: vec3<u32>) {
    let q_block = gid.x;
    let head = gid.y;

    let m = config.num_blocks;
    let h = config.n_head;

    if (q_block >= m || head >= h) {
        return;
    }

    let last_full = q_block >= m - config.last_n_full;

    // Find max score for this (q_block, head)
    var max_score: f32 = -3.4e38; // -FLT_MAX
    for (var k = 0u; k <= q_block; k = k + 1u) {
        let score = block_scores[q_block * m * h + k * h + head];
        if (score > max_score) { max_score = score; }
    }

    let thresh = max_score * config.alpha;

    // Select blocks
    var count: u32 = 0u;
    for (var k = 0u; k <= q_block; k = k + 1u) {
        var keep = false;

        // Sink
        if (k < config.attention_sink) { keep = true; }
        // Window
        let diff = select(q_block - k, k - q_block, k > q_block);
        if (diff < config.window) { keep = true; }
        // Last full
        if (last_full) { keep = true; }
        // Alpha threshold
        if (!keep) {
            let score = block_scores[q_block * m * h + k * h + head];
            if (score >= thresh) { keep = true; }
        }

        let idx_out = q_block * m * h + count * h + head;
        if (keep && count < m) {
            selected_indices[idx_out] = i32(k);
            count = count + 1u;
        }
    }

    // Pad remaining with -1
    for (var i = count; i < m; i = i + 1u) {
        selected_indices[q_block * m * h + i * h + head] = -1;
    }

    selected_counts[q_block * h + head] = i32(count);
}
```

**Acceptance:** Kernel compiles. Unit test: known scores → correct selection with all 4 rules.

### Task 7: Write `flashprefill_sparse_forward.wgsl`

Fourth GPU kernel: sparse attention using only selected blocks.

```wgsl
// riir-ai/crates/riir-gpu/src/kernels/flashprefill_sparse_forward.wgsl

// Kernel 4: Sparse attention forward.
// Input:  Q, K, V caches + selected_indices + selected_counts
// Output: O [seq_len, n_head, head_dim]
//
// For each (q_block, head):
//   1. Load Q for this block's tokens
//   2. For each selected k_block: load K, compute attention scores
//   3. Online softmax over sparse K blocks
//   4. Weighted sum of V from selected blocks
//
// This is the core performance win: instead of attending to ALL k_blocks,
// we attend only to the selected few (typically 5-20% of total).

struct FlashPrefillConfig {
    block_size: u32,
    attention_sink: u32,
    window: u32,
    last_n_full: u32,
    alpha: f32,
    tail_window: u32,
    seq_len: u32,
    num_blocks: u32,
    n_kv_head: u32,
    head_dim: u32,
    n_head: u32,
    scale: f32,
    _pad0: u32,
}

@group(0) @binding(0) var<storage, read>       q_cache: array<f32>;            // [S * n_head * head_dim]
@group(0) @binding(1) var<storage, read>       k_cache: array<f32>;            // [S * n_kv_head * head_dim]
@group(0) @binding(2) var<storage, read>       v_cache: array<f32>;            // [S * n_kv_head * head_dim]
@group(0) @binding(3) var<storage, read>       selected_indices: array<i32>;   // [M * M * H]
@group(0) @binding(4) var<storage, read>       selected_counts: array<i32>;    // [M * H]
@group(0) @binding(5) var<storage, read_write> output: array<f32>;             // [S * n_head * head_dim]
@group(0) @binding(6) var<uniform>             config: FlashPrefillConfig;

var<workgroup> shared_scores: array<f32, 64>;  // max block_size = 64

@compute @workgroup_size(64)
fn flashprefill_sparse_forward(@builtin(global_invocation_id) gid: vec3<u32>) {
    let thread_idx = gid.x;  // token within block
    let q_block = gid.y;
    let head = gid.z;

    let m = config.num_blocks;
    let h = config.n_head;
    let hd = config.head_dim;
    let bs = config.block_size;

    if (q_block >= m || head >= h) {
        return;
    }

    let kv_group = head * config.n_kv_head / config.n_head;
    let q_stride = h * hd;
    let kv_stride = config.n_kv_head * hd;
    let n_selected = selected_counts[q_block * h + head];

    let q_global_start = q_block * bs;
    let q_global_end = min(q_global_start + bs, config.seq_len);

    // Process only if this thread maps to a valid token in the block
    if (thread_idx >= bs) { return; }
    let q_global = q_global_start + thread_idx;
    if (q_global >= config.seq_len) { return; }

    // Load Q for this token/head
    var my_q: array<f32, 256> = array<f32, 256>(/* zero */);
    for (var d = 0u; d < hd; d = d + 1u) {
        my_q[d] = q_cache[q_global * q_stride + head * hd + d];
    }

    // Online softmax: iterate over selected k_blocks
    var max_score: f32 = -3.4e38;
    var sum_exp: f32 = 0.0;
    var my_output: array<f32, 256> = array<f32, 256>(/* zero */);

    // Two-pass: first find max, then accumulate
    // Pass 1: find max score
    for (var si = 0u; si < u32(n_selected); si = si + 1u) {
        let k_block = selected_indices[q_block * m * h + si * h + head];
        if (k_block < 0) { continue; }

        let k_start = u32(k_block) * bs;
        let k_end = min(k_start + bs, config.seq_len);

        for (var t = k_start; t < k_end; t = t + 1u) {
            if (t > q_global) { break; }  // causal
            var dot: f32 = 0.0;
            for (var d = 0u; d < hd; d = d + 1u) {
                dot = dot + my_q[d] * k_cache[t * kv_stride + kv_group * hd + d];
            }
            let score = dot * config.scale;
            if (score > max_score) { max_score = score; }
        }
    }

    // Pass 2: softmax weights + weighted sum
    for (var si = 0u; si < u32(n_selected); si = si + 1u) {
        let k_block = selected_indices[q_block * m * h + si * h + head];
        if (k_block < 0) { continue; }

        let k_start = u32(k_block) * bs;
        let k_end = min(k_start + bs, config.seq_len);

        for (var t = k_start; t < k_end; t = t + 1u) {
            if (t > q_global) { break; }
            var dot: f32 = 0.0;
            for (var d = 0u; d < hd; d = d + 1u) {
                dot = dot + my_q[d] * k_cache[t * kv_stride + kv_group * hd + d];
            }
            let s = exp(dot * config.scale - max_score);
            sum_exp = sum_exp + s;

            for (var d = 0u; d < hd; d = d + 1u) {
                my_output[d] = my_output[d] + s * v_cache[t * kv_stride + kv_group * hd + d];
            }
        }
    }

    // Normalize and write output
    if (sum_exp > 0.0) {
        let inv_sum = 1.0 / sum_exp;
        for (var d = 0u; d < hd; d = d + 1u) {
            output[q_global * h * hd + head * hd + d] = my_output[d] * inv_sum;
        }
    }
}
```

**Acceptance:** Kernel compiles. Correctness test: sparse forward output matches dense forward for selected blocks.

### Task 8: Wire GPU PFlash Pipeline + CPU Fallback

Connect the 4 GPU kernels into a pipeline in `riir-gpu`, and add the CPU fallback in `prefill.rs`.

```rust
// riir-ai/crates/riir-gpu/src/forward.rs (additions)

impl GpuFlashPrefillPass {
    pub fn new(ctx: GpuContext, config: &Config, fp_config: &FlashPrefillConfig) -> Self {
        // Allocate buffers, compile pipelines for the 4 kernels
        // ...
    }

    /// Run the full PFlash 4-kernel pipeline on GPU.
    /// Returns selected block indices for CPU-side token flattening.
    pub fn score_and_select(
        &mut self,
        key_cache: &Buffer,
        value_cache: &Buffer,
        query_cache: &Buffer,
        seq_len: usize,
    ) -> Result<Vec<usize>, GpuError> {
        let m = (seq_len + self.fp_config.block_size - 1) / self.fp_config.block_size;

        // Dispatch kernel 1: mean_K
        self.dispatch_mean_k(key_cache, seq_len)?;

        // Dispatch kernel 2: block_score
        self.dispatch_block_score(query_cache, seq_len)?;

        // Dispatch kernel 3: block_select
        self.dispatch_block_select(m)?;

        // Download selected_counts (small: M × H × 4 bytes)
        let counts = download_f32(&self.ctx.device, &self.ctx.queue,
                                   &self.buffers.selected_counts,
                                   m * self.config.n_head)?;

        // Optional: dispatch kernel 4 (sparse_forward) for the drafter's own scoring
        // This is only needed if we want the drafter to use sparse attention for scoring.
        // For now, we just need the selection output.

        // Convert to block indices
        Ok(self.selected_blocks_from_counts(&counts, m))
    }
}
```

```rust
// speculative/prefill.rs — compress_prompt_blocks and top-level API

/// Compress prompt using block-sparse selection (PFlash algorithm).
///
/// 1. Score blocks via any PrefillScorer (CPU) or GPU pipeline
/// 2. Select blocks via `block_select` (sink + window + last_n + alpha rules)
/// 3. Flatten selected blocks to token indices
/// 4. Always include prefix_len tokens and suffix_len tokens
pub fn compress_prompt_blocks(
    importance_scores: &[f32],
    cfg: &FlashPrefillConfig,
    prefix_len: usize,
    suffix_len: usize,
) -> Vec<usize> {
    let total = importance_scores.len();
    if total == 0 {
        return Vec::new();
    }

    let block_size = cfg.block_size;
    let num_blocks = (total + block_size - 1) / block_size;

    // Aggregate per-token scores to per-block scores (max of block)
    let mut block_scores = vec![0.0f32; num_blocks];
    for (i, &score) in importance_scores.iter().enumerate() {
        let block_idx = i / block_size;
        block_scores[block_idx] = block_scores[block_idx].max(score);
    }

    // Select blocks
    let selected_blocks = block_select(&block_scores, cfg);

    // Flatten blocks to token indices
    let mut selected_tokens: Vec<usize> = Vec::new();

    // Always include prefix
    let prefix_end = prefix_len.min(total);
    selected_tokens.extend(0..prefix_end);

    // Flatten selected blocks (skip prefix/suffix overlap)
    for &block_idx in &selected_blocks {
        let block_start = block_idx * block_size;
        let block_end = ((block_idx + 1) * block_size).min(total);
        for token_idx in block_start..block_end {
            if token_idx >= prefix_end && token_idx < total.saturating_sub(suffix_len) {
                selected_tokens.push(token_idx);
            }
        }
    }

    // Always include suffix
    let suffix_start = total.saturating_sub(suffix_len);
    selected_tokens.extend(suffix_start..total);

    selected_tokens.sort();
    selected_tokens.dedup();

    selected_tokens
}

/// Whether to apply compression for the given prompt length and mode.
pub fn should_compress(mode: PrefillMode, prompt_len: usize, threshold: usize) -> bool {
    match mode {
        PrefillMode::Off => false,
        PrefillMode::Always => true,
        PrefillMode::Auto => prompt_len >= threshold,
    }
}

/// PFlash compression — CPU path (fallback when no GPU).
pub fn speculative_prefill_block(
    scorer: &dyn PrefillScorer,
    draft_weights: &TransformerWeights,
    draft_config: &Config,
    prompt_tokens: &[usize],
    cfg: &FlashPrefillConfig,
    prefix_len: usize,
    suffix_len: usize,
) -> Vec<usize> {
    if prompt_tokens.is_empty() {
        return Vec::new();
    }
    let scores = scorer.score(draft_weights, draft_config, prompt_tokens);
    compress_prompt_blocks(&scores, cfg, prefix_len, suffix_len)
}

/// PFlash compression — GPU path via riir-gpu Metal kernels.
/// Available when `gpu` feature is enabled.
#[cfg(feature = "gpu")]
pub fn speculative_prefill_gpu(
    gpu_pass: &mut riir_gpu::GpuFlashPrefillPass,
    draft_weights_gpu: &riir_gpu::GpuWeightBuffers,
    prompt_tokens: &[usize],
    cfg: &FlashPrefillConfig,
    prefix_len: usize,
    suffix_len: usize,
) -> Vec<usize> {
    if prompt_tokens.is_empty() {
        return Vec::new();
    }

    // GPU forward pass to populate KV cache
    let seq_len = prompt_tokens.len();
    gpu_pass.forward_draft(draft_weights_gpu, prompt_tokens)?;

    // GPU block selection pipeline
    let selected_blocks = gpu_pass.score_and_select(
        &draft_weights_gpu.key_cache,
        &draft_weights_gpu.value_cache,
        &draft_weights_gpu.query_cache,
        seq_len,
    )?;

    // Flatten to token indices (CPU, lightweight)
    let mut selected_tokens: Vec<usize> = Vec::new();
    let prefix_end = prefix_len.min(seq_len);

    selected_tokens.extend(0..prefix_end);

    for &block_idx in &selected_blocks {
        let block_start = block_idx * cfg.block_size;
        let block_end = ((block_idx + 1) * cfg.block_size).min(seq_len);
        for token_idx in block_start..block_end {
            if token_idx >= prefix_end && token_idx < seq_len.saturating_sub(suffix_len) {
                selected_tokens.push(token_idx);
            }
        }
    }

    let suffix_start = seq_len.saturating_sub(suffix_len);
    selected_tokens.extend(suffix_start..seq_len);

    selected_tokens.sort();
    selected_tokens.dedup();

    selected_tokens
}

/// PFlash compression with adaptive threshold.
/// Picks GPU path when available, CPU fallback otherwise.
pub fn speculative_prefill_adaptive(
    scorer: &dyn PrefillScorer,
    draft_weights: &TransformerWeights,
    draft_config: &Config,
    prompt_tokens: &[usize],
    mode: PrefillMode,
    threshold: usize,
    cfg: &FlashPrefillConfig,
    prefix_len: usize,
    suffix_len: usize,
) -> Vec<usize> {
    if !should_compress(mode, prompt_tokens.len(), threshold) {
        return (0..prompt_tokens.len()).collect();
    }
    speculative_prefill_block(scorer, draft_weights, draft_config, prompt_tokens, cfg, prefix_len, suffix_len)
}
```

**Acceptance:** Both CPU and GPU paths produce valid compressed indices. GPU path shows speedup over CPU at prompt_len ≥ 128.

### Task 9: Add NIAH Quality Benchmarks

Port the needle-in-a-haystack validation:

```rust
// tests/bench_prefill.rs

/// NIAH single-needle retrieval benchmark.
/// Sweeps: prompt_len × keep_ratio × scorer_type × (CPU/GPU).
fn bench_niah_retrieval_rate() {
    let prompt_lengths = [64, 128, 256, 512];
    let keep_ratios = [0.02, 0.05, 0.10, 0.20];
    let scorers = ["attention", "block_attention", "uniform"];

    for &prompt_len in &prompt_lengths {
        for &keep_ratio in &keep_ratios {
            for &scorer in &scorers {
                // Generate: [hay×(N-1)] + [needle_marker, secret] + [hay×(N-1)]
                // Compress with block selection
                // Check if needle survives
                // Report: PASS/FAIL + compression ratio + time
            }
        }
    }
}

/// GPU vs CPU scoring throughput comparison.
fn bench_gpu_vs_cpu_scoring() {
    // CPU: BlockAttentionScorer (token-level forward + block aggregation)
    // GPU: 4-kernel pipeline (mean_K → block_score → block_select)
    // At: 128, 256, 512, 1024, 2048 tokens
    // Report: μs per scoring pass, compression ratio, needle survival
}

/// Block selection rule validation.
fn bench_block_selection_guarantees() {
    // Verify sink blocks always selected (even with score=0)
    // Verify window blocks always selected
    // Verify last_n blocks select everything
    // Verify alpha threshold filters correctly
    // Sweep alpha: [0.05, 0.12, 0.25, 0.50, 0.85, 0.99]
}
```

**Acceptance:** Benchmark file runs. NIAH retrieval ≥ 95% at keep_ratio=0.05 with block selection. GPU path shows measurable speedup over CPU at prompt_len ≥ 256.

### Task 10: Update Docs and Commit

1. Update `.docs/08_lucebox_techniques.md` — add Technique 4a: Metal-Accelerated Block-Sparse PFlash
2. Update `.docs/01_overview.md` — module structure with new types and GPU kernels
3. Update `speculative/mod.rs` — re-export `FlashPrefillConfig`, `PrefillMode`, `BlockScores`, `BlockAttentionScorer`, `block_select`, `block_select_grid`, `compress_prompt_blocks`, `speculative_prefill_block`, `speculative_prefill_adaptive`, `should_compress`
4. Update `riir-ai/crates/riir-gpu/src/kernels/mod.rs` — register 4 new WGSL kernels
5. Update `riir-ai/crates/riir-gpu/src/lib.rs` — export `GpuFlashPrefillPass`, `GpuFlashPrefillBuffers`
6. Run `cargo clippy --fix --allow-dirty` on both `microgpt-rs` and `riir-gpu`
7. Run `cargo test --quiet --workspace --all-features` — all tests pass
8. Commit: `feat(speculative): Metal-accelerated PFlash block-sparse speculative prefill (Plan 044)`

**Acceptance:** All tests green, docs updated, clean clippy, committed.

---

## File Change Summary

### New files
- `microgpt-rs/tests/bench_prefill.rs` — baseline + NIAH + GPU benchmarks
- `riir-ai/crates/riir-gpu/src/kernels/flashprefill_mean_k.wgsl` — kernel 1
- `riir-ai/crates/riir-gpu/src/kernels/flashprefill_block_score.wgsl` — kernel 2
- `riir-ai/crates/riir-gpu/src/kernels/flashprefill_block_select.wgsl` — kernel 3
- `riir-ai/crates/riir-gpu/src/kernels/flashprefill_sparse_forward.wgsl` — kernel 4

### Modified files
- `microgpt-rs/src/speculative/types.rs` — add `FlashPrefillConfig`, `PrefillMode`, `BlockScores`
- `microgpt-rs/src/speculative/prefill.rs` — add `BlockAttentionScorer`, `block_select`, `block_select_grid`, `compress_prompt_blocks`, `speculative_prefill_block`, `speculative_prefill_adaptive`, `should_compress`
- `microgpt-rs/src/speculative/mod.rs` — re-export new types and functions
- `riir-ai/crates/riir-gpu/src/forward.rs` — add `GpuFlashPrefillPass`, `GpuFlashPrefillBuffers`, GPU pipeline
- `riir-ai/crates/riir-gpu/src/kernels/mod.rs` — register 4 new WGSL shaders
- `riir-ai/crates/riir-gpu/src/lib.rs` — export GPU PFlash types
- `microgpt-rs/.docs/08_lucebox_techniques.md` — document Metal PFlash
- `microgpt-rs/.docs/01_overview.md` — update module structure

---

## Design Decisions

### 1. wgpu → Metal, not raw Metal
We use wgpu's compute shader API which compiles WGSL to MSL (Metal Shading Language) via naga. This gives us:
- Portability (same code runs on Vulkan/DX12 if needed)
- Existing infrastructure (`GpuContext`, `GpuForwardPass`, buffer management)
- No need to learn MSL syntax
- Debug via Xcode GPU profiler (wgpu on macOS uses Metal backend)

The C++ PFlash uses raw CUDA WMMA. We use wgpu compute shaders. Same mathematical operations, different API.

### 2. Unified memory = zero data transfer
Apple Silicon's CPU and GPU share the same physical RAM. When we upload KV cache to a GPU buffer, the data stays in the same memory — no PCIe DMA, no cudaMemcpy. The GPU command buffer just points to the same address. This means:
- No upload/download latency for scoring
- KV cache can be shared between CPU forward pass and GPU block scoring
- The only data transfer is downloading `selected_counts` (~4KB at 128K context)

### 3. Block size = 64 for Metal, 32 for CPU
Apple GPU workgroups perform best at 64-256 threads. Block size 64 means each workgroup processes one (block, head) pair with 64 threads for the head_dim dimension. On CPU, smaller blocks (32) give finer granularity.

### 4. Block selection rules are the key innovation
The `attention_sink + window + last_n_full + alpha` rules are what make PFlash work in practice. Without them, naive top-K scoring drops the needle at keep_ratio < 0.1. With them, NIAH retrieval is preserved even at keep_ratio=0.02. These rules are pure logic — same on GPU and CPU.

### 5. CPU fallback always available
The GPU path is gated behind `#[cfg(feature = "gpu")]`. The CPU `BlockAttentionScorer` + `block_select` path works everywhere. Users get the same quality guarantees regardless of hardware.

### 6. No new feature flag for PFlash
The block-sparse selection is always compiled. `PrefillMode::Off` is the default — no compression unless explicitly requested. The GPU kernels are in `riir-gpu` which is already feature-gated.

---

## Priority Order

| Priority | Task | Impact | Effort |
|----------|------|--------|--------|
| P0 | Task 2: `FlashPrefillConfig` | Foundation | Low |
| P0 | Task 3: `block_select` (Rust) | Core algorithm | Low |
| P1 | Task 5: `flashprefill_mean_k.wgsl` | GPU scoring kernel 1 | Medium |
| P1 | Task 6: `flashprefill_block_score.wgsl` | GPU scoring kernel 2 | Medium |
| P1 | Task 7: `flashprefill_block_select.wgsl` | GPU selection kernel | Medium |
| P1 | Task 8: Wire GPU pipeline + CPU fallback | End-to-end | High |
| P2 | Task 4: `flashprefill_sparse_forward.wgsl` | Full GPU sparse attn | Medium |
| P2 | Task 9: NIAH benchmarks | Validation | Medium |
| P3 | Task 1: Baseline benchmarks | Measurement | Low |
| P3 | Task 10: Docs + commit | Hygiene | Low |

---

## Connection to Existing Plans & Research

| Plan/Doc | Relationship |
|----------|-------------|
| **Plan 043 (TurboQuant)** | Orthogonal: PFlash compresses sequence (fewer tokens), TurboQuant compresses precision (fewer bits). Composable: compressed prompt → target prefill → TQ KV cache. |
| **Plan 020 (Raven RSM)** | Orthogonal: Raven compresses sequence via reservoir sampling (fixed slots). PFlash compresses via importance scoring (variable slots). Different tradeoffs. |
| **`.docs/08_lucebox_techniques.md`** | Technique 4 (Speculative Prefill) upgraded to Technique 4a (Metal-Accelerated Block-Sparse PFlash). |
| **`riir-gpu/src/kernels/attention_score.wgsl`** | Existing GPU attention kernel. New PFlash kernels follow same pattern (bind groups, uniform params, workgroup dispatch). |
| **`dflash/src/flashprefill*.cpp`** | Algorithmic reference for block scoring. We port the logic to WGSL compute shaders. |
| **`pflash/README.md`** | Config reference: `DFLASH_FP_ALPHA`, `keep_ratio`, NIAH methodology. |

---

## Expected Outcomes

| Metric | Before (Plan 044) | After (Plan 044) |
|--------|-------------------|-------------------|
| Scoring path | CPU token-level only | GPU 4-kernel pipeline + CPU fallback |
| Selection rules | Top-K only | Sink + window + last_n + alpha |
| GPU dispatches for 128K scoring | 0 | 4 (mean_K + score + select + sparse_fwd) |
| CPU→GPU data transfer | N/A | 0 bytes (unified memory) |
| NIAH retrieval @ keep_ratio=0.05 | ~60% (token-level top-K) | ~95% (block selection rules) |
| Config flexibility | `keep_ratio` only | Full `FlashPrefillConfig` + `PrefillMode` |
| API surface | `speculative_prefill` | `speculative_prefill_block`, `speculative_prefill_adaptive`, `speculative_prefill_gpu` |

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| WGSL workgroup shared memory limits (32KB on Metal) | Kernel correctness | Use block_size=64, head_dim ≤ 256. Fall back to global memory if needed. |
| Metal shader compilation errors (naga WGSL→MSL) | Build failure | Test WGSL compilation early in Task 4. Use simple shader patterns naga supports. |
| GPU buffer allocation failure at long context | OOM | Check `M * M * H * 4` before allocation. At 128K, M=2048, H=16: ~256MB. Within 36GB unified memory. |
| CPU fallback quality worse than GPU on short prompts | Lower compression quality | Fall back to token-level when `prompt_len < threshold` (via `PrefillMode::Auto`) |
| Existing `AttentionScorer` behavior changes | Regression | No changes to existing functions. New API is additive. |
| GPU path slower than CPU for short prompts | Latency regression | `PrefillMode::Auto` with threshold (default: 128 tokens) — GPU only for longer prompts where parallelism wins. |

---

## Research Citation

```bibtex
@inproceedings{liu2026speculative_prefill,
  title     = {Cross-Family Speculative Prefill},
  author    = {Liu, Jingyu and others},
  booktitle = {ICLR},
  year      = {2026},
  note      = {Algorithm: small drafter scores per-token importance, selects spans for target prefill}
}

@article{fan2026flashprefill,
  title     = {FlashPrefill: Block-Sparse Attention for Long-Context Prefill},
  author    = {Fan, Qihui and others},
  year      = {2026},
  note      = {Kernels: mean\_K → block\_score → block\_select → sparse\_forward}
}

@software{luce_pflash_2026,
  title     = {Luce PFlash: Speculative Prefill Compression for Long-Context Spec Decode},
  author    = {Luce},
  url       = {https://github.com/Luce-Org/lucebox-hub/tree/main/pflash},
  year      = {2026},
  note      = {C++/CUDA implementation: 10.4× TTFT reduction at 128K context}
}