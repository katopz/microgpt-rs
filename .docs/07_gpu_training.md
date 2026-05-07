# mini-dllm: GPU LoRA Training (wgpu)

## Overview
A `wgpu`-based GPU training backend that produces `lora.bin` from training data JSONL. Targets WASM (WebGPU), Metal (Apple Silicon), Vulkan (Nvidia/AMD), and DX12 (Windows). The CPU path remains the zero-allocation reference; GPU accelerates forward + backward for LoRA parameter updates.

## LoRA Training Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    LoRA TRAINING PIPELINE                       │
│                                                                  │
│  training.jsonl ──► DataLoader ──► Batch ──► GPU Upload         │
│  (from cLoRA pipeline)                               │         │
│                                      ┌───────────────▼───────┐ │
│                                      │   wgpu Compute Pass   │ │
│                                      │                        │ │
│                                      │  1. Embed tokens       │ │
│                                      │  2. For each layer:    │ │
│                                      │     a. QKV projection  │ │
│                                      │        + LoRA A/B      │ │
│                                      │     b. Attention        │ │
│                                      │     c. Out projection  │ │
│                                      │        + LoRA A/B      │ │
│                                      │     d. LayerNorm        │ │
│                                      │     e. MLP             │ │
│                                      │        + LoRA A/B      │ │
│                                      │     f. LayerNorm        │ │
│                                      │  3. lm_head            │ │
│                                      │  4. Softmax + CE loss  │ │
│                                      │  5. Backward (LoRA)    │ │
│                                      │  6. Optimizer step     │ │
│                                      └───────────┬───────────┘ │
│                                                  │              │
│                                        ┌─────────▼─────────┐   │
│                                        │  Updated LoRA A/B  │   │
│                                        │  (lora.bin)        │   │
│                                        └───────────────────┘   │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    wgpu BACKEND LAYERS                          │
│                                                                  │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────────────┐  │
│  │ gpu/context  │  │ gpu/kernels/ │  │ gpu/training/          │  │
│  │              │  │              │  │                        │  │
│  │ device init  │  │ matmul.wgsl  │  │ forward.rs            │  │
│  │ queue setup  │  │ softmax.wgsl │  │ backward.rs           │  │
│  │ buffer alloc │  │ layernorm.wgsl│  │ loss.rs               │  │
│  │ shader load  │  │ attention.wgsl│  │ optimizer.rs          │  │
│  │              │  │ embedding.wgsl│  │ dataloader.rs         │  │
│  │              │  │ lora.wgsl     │  │ loop.rs               │  │
│  │              │  │ optimizer.wgsl│  │                        │  │
│  └─────────────┘  └──────────────┘  └───────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Status
All planned — not yet implemented.

## LoRA Architecture

### Concept
LoRA injects low-rank adapter matrices into the transformer. Base weights are **frozen**. Only A and B are trained:
```
Standard:  Y = W · x
LoRA:      Y = W · x + α · (B · A) · x
           A ∈ ℝ^(rank × n_embd), B ∈ ℝ^(n_embd × rank)
           A: Kaiming init, B: zero init → ΔW starts as zero
```

### Adapter Locations (per layer)
| Adapter | Base Shape | LoRA A Shape | LoRA B Shape |
|---------|-----------|-------------|-------------|
| `attn_wq` | [n_embd, n_embd] | [rank, n_embd] | [n_embd, rank] |
| `attn_wk` | [n_embd, n_embd] | [rank, n_embd] | [n_embd, rank] |
| `attn_wv` | [n_embd, n_embd] | [rank, n_embd] | [n_embd, rank] |
| `attn_wo` | [n_embd, n_embd] | [rank, n_embd] | [n_embd, rank] |
| `mlp_w1` | [mlp_hidden, n_embd] | [rank, n_embd] | [mlp_hidden, rank] |
| `mlp_w2` | [n_embd, mlp_hidden] | [rank, mlp_hidden] | [n_embd, rank] |
| **Total per layer** | | | **12 × rank × n_embd** |

### Parameter Estimates
| Config | n_embd | rank | n_layers | LoRA Params | Memory |
|--------|--------|------|----------|-------------|--------|
| `micro` | 16 | 4 | 1 | ~5K | ~20 KB |
| `draft` | 64 | 4 | 1 | ~20K | ~80 KB |
| cLoRA target | 256 | 16 | 4 | ~524K | ~2 MB |
| cLoRA large | 512 | 32 | 8 | ~4.2M | ~17 MB |

## Module Layout (planned)
```
src/gpu/
├── mod.rs              # Feature gate + re-exports
├── context.rs          # GpuContext: device, queue, adapter info
├── buffer.rs           # upload_f32, download_f32, create_buffer
├── kernels/            # WGSL compute shaders
│   ├── mod.rs
│   ├── matmul.wgsl     # Tiled 16×16 matrix multiply with shared memory
│   ├── elementwise.wgsl
│   ├── softmax.wgsl    # Stable online softmax
│   ├── layernorm.wgsl  # RMSNorm
│   ├── embedding.wgsl
│   ├── attention.wgsl
│   ├── lora.wgsl       # Two-dispatch: A·input → intermediate, then W·input + B·intermediate
│   ├── loss.wgsl       # Cross-entropy: per-sample + tree reduction
│   └── optimizer.wgsl  # AdamW with bias correction
├── forward.rs          # GpuForwardPass orchestration
├── backward.rs         # GpuBackwardPass (LoRA gradients only)
├── lora.rs             # Adapter init, merge, export
├── optimizer.rs        # AdamW state management
├── loss.rs             # CE loss dispatch
├── training_loop.rs    # Trainer: epoch loop, logging, checkpoints
└── dataloader.rs       # JSONL → batched tensors (shuffling, padding)
```

## GpuContext (`gpu/context.rs`)
```rust
pub struct GpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub adapter_info: wgpu::AdapterInfo,
    pub limits: wgpu::Limits,
}
```
- `new() -> Result<Self, GpuError>` — uses `pollster::block_on` for sync API
- WASM: needs `wasm-bindgen-futures` instead of `pollster`
- Power preference: `HighPerformance`
- Limits: `downlevel_defaults` (widest compatibility)

## GPU Forward Pass (`gpu/forward.rs`)

### Buffer Structs
```rust
pub struct GpuWeightBuffers {
    pub wte: wgpu::Buffer,
    pub wpe: wgpu::Buffer,
    pub lm_head: wgpu::Buffer,
    pub layers: Vec<GpuLayerBuffers>,  // mirrors TransformerWeights.layers
}
pub struct GpuLoraBuffers {
    pub adapters: Vec<GpuLoraAdapter>,  // 6 per layer
}
pub struct GpuLoraAdapter {
    pub a: wgpu::Buffer,       // [rank, in_dim]
    pub b: wgpu::Buffer,       // [out_dim, rank]
    pub grad_a: wgpu::Buffer,
    pub grad_b: wgpu::Buffer,
    pub m_a: wgpu::Buffer,     // AdamW first moment
    pub v_a: wgpu::Buffer,     // AdamW second moment
    pub m_b: wgpu::Buffer,
    pub v_b: wgpu::Buffer,
}
```

### Pipeline
```
1. Embedding lookup: hidden = wte[tokens] + wpe[positions]
2. For each layer:
   a. RMSNorm
   b. QKV projection + LoRA merge (W·x + α·B·A·x)
   c. Multi-head attention
   d. Output projection + LoRA merge + residual
   e. RMSNorm
   f. MLP + LoRA merge + residual
3. LM head projection
4. Softmax + cross-entropy loss
```

## GPU Backward Pass (`gpu/backward.rs`)
Only LoRA parameters have gradients. Base weights frozen.
```
Chain rule:
  grad_B = α · dL/d_output^T · (A · input)
  grad_A = α · B^T · dL/d_output · input^T
```

## WGSL Shader Highlights

### Tiled Matmul
- Workgroup size: 16×16 = 256 invocations
- Shared memory: two 16×16 tiles (A and B)
- Accumulates partial dot products across tiles

### LoRA Merge (two dispatches)
- Dispatch 1: `A[rank, n_embd] × input[n_embd] → intermediate[rank]`
- Dispatch 2: `W[out_dim, n_embd] × input + α × B[out_dim, rank] × intermediate → output`

### Cross-Entropy Loss
- Dispatch 1: per-sample softmax + loss (one invocation per sample)
- Dispatch 2: tree reduction to mean loss (workgroup_size=256)

### AdamW Optimizer
- Updates params in-place: momentum + velocity + bias correction + weight decay
- One invocation per parameter

## Training Loop (`gpu/training_loop.rs`)
```rust
pub struct TrainingConfig {
    pub epochs: usize,
    pub learning_rate: f32,
    pub weight_decay: f32,    // 0.01
    pub beta1: f32,           // 0.9
    pub beta2: f32,           // 0.999
    pub warmup_steps: usize,
    pub log_interval: usize,
    pub checkpoint_interval: usize,
}
```
Per step: forward → loss → backward (LoRA grads) → optimizer step → logging → checkpoint

## DataLoader (`gpu/dataloader.rs`)
- Reads JSONL from cLoRA pipeline (Plan 007)
- Batches samples with shuffling and padding
- Input: `[batch_size, seq_len]`, Target: shifted by 1 (next-token prediction)

## LoRA Export (`gpu/lora.rs`)
- `export_lora(adapters, config, path)` — download A/B from GPU → safetensors
- `load_lora(path, forward)` — read safetensors → upload to GPU buffers
- WASM fallback: simpler binary format with blake3 checksum (safetensors not WASM-compatible)

## Feature Flag
```toml
gpu = ["wgpu", "bytemuck", "pollster", "safetensors"]
```

## Key Risks
| Risk | Mitigation |
|------|-----------|
| WGSL shared memory limits (16KB) | Use 16×16 tiles (4KB each) |
| No cooperative groups in WebGPU | Chain dispatches in command encoder |
| GPU-CPU transfer latency | Keep data on GPU; only download for checkpoints |
| Numerical precision GPU vs CPU | Stable softmax (max subtraction); test within epsilon |
| `safetensors` not WASM-compatible | Binary format fallback with blake3 |

## Prerequisites
- Plan 007 (cLoRA): BPE Tokenizer must define vocabulary and Config dimensions
- Training data JSONL from Plan 007/009 pipeline
- Development/testing uses `Config::micro()` which already exists