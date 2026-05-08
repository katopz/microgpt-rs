# Plan 008: wgpu LoRA Training — GPU-Accelerated `lora.bin` Fine-Tuning

> **Rename Note**: The `clora` module was renamed to `validator` because it contains
> deterministic syntax validation code (SynPruner, PartialParser), not neural LoRA weights.
> Feature flag: `clora` → `validator` (previously `clora`). Module path: `src/clora/` → `src/validator/`.
> The actual LoRA adapter (`lora.bin`) lives in this plan's `gpu` feature.
> The concept "Computable LoRA" / "cLoRA" is now called "Deterministic Validator".

## Objective

Build a `wgpu`-based GPU training backend that produces `lora.bin` (neural LoRA weights) from the training data JSONL produced by plan 007 (Deterministic Validator, previously cLoRA). The CPU path remains the zero-allocation reference implementation. The GPU path accelerates the forward + backward pass for LoRA parameter updates, targeting WASM (WebGPU), Metal (Apple Silicon), Vulkan (Nvidia/AMD), and DX12 (Windows).

## The Problem

Plan 007 produces `training.jsonl` — millions of BPE-tokenized Rust code samples filtered through SynPruner + cargo check. To convert that data into a `lora.bin` (the "muscle memory" that makes the draft model natively better), we need:

```
training.jsonl → batch → forward pass → cross-entropy loss → backward pass → LoRA A/B update → lora.bin
```

The forward + backward pass is matmul-heavy. CPU works for the toy model (vocab=27, n_embd=16). But at Deterministic Validator scale (vocab=32K, n_embd=256+, multi-layer), GPU becomes essential for practical training times.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    LoRA TRAINING PIPELINE                       │
│                                                                  │
│  training.jsonl ──► DataLoader ──► Batch ──► GPU Upload         │
│  (from plan 007)                                      │         │
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

## Design Principle: WASM-First, wgpu-Native

This plan targets **WebGPU via wgpu** as the sole GPU backend. Rationale:

1. **WASM compatibility**: wgpu compiles to `wasm32-unknown-unknown` via WebGPU. This means the same training code runs in-browser.
2. **Cross-platform**: wgpu abstracts over Metal (macOS), Vulkan (Linux/Android), DX12 (Windows), and WebGPU (browser).
3. **No candle/burn dependency**: Those crates don't support WASM GPU compute. wgpu + WGSL is the only path for browser-based LoRA training.
4. **Micro-scale appropriate**: At `n_embd ≤ 256`, WebGPU compute shaders are sufficient. No need for CUDA-specific optimizations.
5. **safetensors may not compile on WASM** — for browser, use a simpler binary format (raw f32 slices with blake3 checksum). safetensors stays for native.

## Current Codebase State (as of Plan 019)

| What | Status | Impact on Plan 008 |
|------|--------|--------------------|
| Multi-layer `TransformerWeights` with `Vec<LayerWeights>` | ✅ Done (Plan 010) | GpuWeightBuffers must mirror this structure |
| `Config.n_layer` field | ✅ Done (Plan 010) | No Config changes needed for n_layer |
| GQA (`n_kv_head`) | ✅ Done (Plan 011) | GPU attention must handle GQA groups |
| `PagedKVCache` | ✅ Done (Plan 011) | GPU KV cache can use paged design |
| `ForwardContext` zero-alloc | ✅ Done (Plan 013) | GPU activation buffers follow same pattern |
| `hidden_state` extraction | ✅ Done (Plan 009) | GPU forward must also expose hidden state |
| BPE tokenizer | ✅ Done (Plan 007) | `Config::bpe()` / `Config::bpe_draft()` available |
| SynPruner / validator module | ✅ Done (Plan 007) | SynPruner validates training samples |
| `src/gpu/context.rs` + `buffer.rs` | ✅ Done (Phase 1) | GpuContext, upload/download/create_buffer with tests |
| 9× WGSL compute kernels | ✅ Done (Phase 2 partial) | matmul, elementwise, softmax, layernorm, embedding, attention, lora, loss, optimizer |
| `src/gpu/kernels/mod.rs` (pipeline helpers) | ❌ Missing | Needed before Phase 3 — shader loading & pipeline creation |
| GPU shader tests / benchmarks | ❌ Not started | 2.11, 2.12 still pending |

### Prerequisites Update
- ~~Plan 007 Phase 1 (BPE Tokenizer) must be complete~~ ✅ Plan 007 is fully complete
- Development and testing uses `Config::micro()` which already exists
- `Config::bpe()` / `Config::bpe_draft()` from Plan 007 define the production config dimensions
- Phase 1 (wgpu context + buffers) is done — Phase 2 kernels exist but need `kernels/mod.rs` + tests

## LoRA Architecture

LoRA injects low-rank adapter matrices into the transformer. Base weights are **frozen**. Only LoRA A and B are updated during training.

```
Standard:    Y = W·x
LoRA:        Y = W·x + (B·A)·x   where A ∈ ℝ^(rank×n_embd), B ∈ ℝ^(n_embd×rank)
                                   rank << n_embd (e.g., rank=4, n_embd=256)
                                   A initialized with Kaiming, B initialized with zeros
                                   → ΔW = B·A starts as zero (no behavior change at init)
```

### Adapter Locations (per layer)

| Adapter | Base Weight Shape | LoRA A Shape | LoRA B Shape | Params (rank=4) |
|---------|------------------|-------------|-------------|----------------|
| `attn_wq` | [n_embd, n_embd] | [rank, n_embd] | [n_embd, rank] | 2 × rank × n_embd |
| `attn_wk` | [n_embd, n_embd] | [rank, n_embd] | [n_embd, rank] | 2 × rank × n_embd |
| `attn_wv` | [n_embd, n_embd] | [rank, n_embd] | [n_embd, rank] | 2 × rank × n_embd |
| `attn_wo` | [n_embd, n_embd] | [rank, n_embd] | [n_embd, rank] | 2 × rank × n_embd |
| `mlp_w1` | [mlp_hidden, n_embd] | [rank, n_embd] | [mlp_hidden, rank] | 2 × rank × n_embd |
| `mlp_w2` | [n_embd, mlp_hidden] | [rank, mlp_hidden] | [n_embd, rank] | 2 × rank × mlp_hidden |
| **Total per layer** | | | | **12 × rank × n_embd** |

### Parameter Count Estimates

| Config | n_embd | mlp_hidden | rank | n_layers | LoRA Params | Memory (f32) |
|--------|--------|------------|------|----------|-------------|-------------|
| `micro` (toy) | 16 | 64 | 4 | 1 | ~5K | ~20 KB |
| `draft` (small) | 64 | 256 | 4 | 1 | ~20K | ~80 KB |
| validator target (previously cLoRA) | 256 | 1024 | 16 | 4 | ~524K | ~2 MB |
| validator large (previously cLoRA) | 512 | 2048 | 32 | 8 | ~4.2M | ~17 MB |

Even the largest config fits easily in GPU memory. The bottleneck is the **base model forward pass** (to compute activations), not the LoRA params.

## New Modules

```
src/
├── gpu/                               # NEW: wgpu backend (feature-gated)
│   ├── mod.rs                         # Re-exports, feature gate
│   ├── context.rs                     # GpuContext: device, queue, pipeline cache
│   ├── buffer.rs                      # Buffer management: alloc, upload, download
│   ├── kernels/                       # WGSL shader source
│   │   ├── mod.rs                     # Shader loading utilities
│   │   ├── matmul.wgsl                # Tiled matrix multiply
│   │   ├── elementwise.wgsl           # Add, mul, relu, sigmoid, scale
│   │   ├── softmax.wgsl               # Stable softmax (online algorithm)
│   │   ├── layernorm.wgsl             # RMSNorm / LayerNorm
│   │   ├── embedding.wgsl             # Token + position embedding lookup
│   │   ├── attention.wgsl             # Scaled dot-product attention
│   │   ├── lora.wgsl                  # LoRA merge: Y = Wx + BAx
│   │   ├── loss.wgsl                  # Cross-entropy loss
│   │   └── optimizer.wgsl             # AdamW parameter update
│   ├── forward.rs                     # GPU forward pass orchestration
│   ├── backward.rs                    # GPU backward pass (LoRA gradients only)
│   ├── lora.rs                        # LoRA adapter: init, merge, export
│   ├── optimizer.rs                   # AdamW optimizer state + step
│   ├── loss.rs                        # Cross-entropy loss computation
│   └── training_loop.rs               # Epoch loop, logging, checkpoint
├── gpu/                               # (continued from above)
│   └── dataloader.rs                  # NEW: batch iteration, shuffling, padding (NOT src/data/)
├── transformer.rs                     # EXISTING (unchanged — CPU fallback)
├── types.rs                           # EXISTING (extended for LoRA config)
└── lib.rs                             # EXISTING (add mod gpu behind feature flag)
```

## Dependency Additions (`Cargo.toml`)

```toml
[dependencies]
# ... existing ...
wgpu = { version = "24", optional = true }
bytemuck = { version = "1", features = ["derive"], optional = true }
pollster = { version = "0.4", optional = true }  # block_on for async wgpu init
safetensors = { version = "0.4", optional = true }   # Native only — not WASM compatible

[features]
default = []
leviathan = []
sudoku = []
validator = ["syn"]                 # previously clora
training = ["serde", "serde_json"]
gpu = ["wgpu", "bytemuck", "pollster", "safetensors"]             # GPU-accelerated training
full = ["leviathan", "sudoku", "validator", "training", "gpu"]    # previously clora
```

## Phase 1: wgpu Context & Buffer Management

### 1.1 GpuContext

```rust
// gpu/context.rs

/// Initialized wgpu device, queue, and adapter info.
/// Feature-gated behind `gpu` — only compiled when the feature is enabled.
pub struct GpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub adapter_info: wgpu::AdapterInfo,
    /// Limits from the adapter (max workgroup size, max buffer size, etc.)
    pub limits: wgpu::Limits,
}

impl GpuContext {
    /// Initialize wgpu device. Uses `pollster::block_on` for sync API.
    /// On WASM, this should be called from a wasm-bindgen async context.
    pub fn new() -> Result<Self, GpuError> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter = pollster::block_on(instance.request_adapter(
            &wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            },
        )).ok_or(GpuError::NoAdapter)?;
        
        let adapter_info = adapter.get_info();
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_defaults(),
                ..Default::default()
            },
            None,
        )).map_err(GpuError::DeviceError)?;
        
        Ok(Self {
            limits: device.limits(),
            adapter_info,
            device,
            queue,
        })
    }
}

#[derive(Debug)]
pub enum GpuError {
    NoAdapter,
    DeviceError(wgpu::RequestDeviceError),
    ShaderError(String),
    BufferError(String),
}
```

### 1.2 Buffer Utilities

```rust
// gpu/buffer.rs

/// Upload a `Vec<f32>` to a GPU storage buffer.
pub fn upload_f32(device: &wgpu::Device, data: &[f32], label: &str) -> wgpu::Buffer {
    let bytes: &[u8] = bytemuck::cast_slice(data);
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytes,
        usage: wgpu::BufferUsages::STORAGE
             | wgpu::BufferUsages::COPY_DST
             | wgpu::BufferUsages::COPY_SRC,
    })
}

/// Download a GPU buffer back to `Vec<f32>`.
pub fn download_f32(device: &wgpu::Device, queue: &wgpu::Queue, buffer: &wgpu::Buffer, len: usize) -> Vec<f32> {
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging"),
        size: (len * 4) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, (len * 4) as u64);
    queue.submit(std::iter::once(encoder.finish()));
    
    // Block until mapped (pollster for native, wasm-bindgen-futures for WASM)
    pollster::block_on(async {
        let slice = staging.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| ());
        device.poll(wgpu::Maintain::Wait).panic_on_timeout();
        let data = slice.get_mapped_range();
        bytemuck::cast_slice(&data).to_vec()
    })
}

/// Create an empty storage buffer of given size (f32 elements).
pub fn create_buffer(device: &wgpu::Device, len: usize, label: &str) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: (len * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE
             | wgpu::BufferUsages::COPY_DST
             | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}
```

## Phase 2: WGSL Compute Shaders

### 2.1 Tiled Matrix Multiply

The core operation. Tiled to leverage workgroup shared memory for SRAM bandwidth.

```wgSL
// gpu/kernels/matmul.wgsl

// Tiled matrix multiply: C[M,P] = A[M,N] * B[N,P]
// Each workgroup computes a tile of the output.
// Workgroup size: 16x16 = 256 invocations (within WebGPU limits).

var<workgroup> tile_a: array<f32, 256>;  // 16x16 tile of A
var<workgroup> tile_b: array<f32, 256>;  // 16x16 tile of B

@group(0) @binding(0) var<storage, read>        a_data: array<f32>;
@group(0) @binding(1) var<storage, read>        b_data: array<f32>;
@group(0) @binding(2) var<storage, read_write>  c_data: array<f32>;
@group(0) @binding(3) var<uniform>              params: MatmulParams;

struct MatmulParams {
    m: u32,  // rows of A
    n: u32,  // cols of A / rows of B
    p: u32,  // cols of B
}

@compute @workgroup_size(16, 16, 1)
fn matmul_tiled(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let row = gid.x;
    let col = gid.y;
    if (row >= params.m || col >= params.p) { return; }
    
    let local_row = lid.x;
    let local_col = lid.y;
    
    var sum: f32 = 0.0;
    
    // Tile loop: process N in chunks of 16
    let num_tiles = (params.n + 15u) / 16u;
    for (var t = 0u; t < num_tiles; t = t + 1u) {
        // Load tile of A into shared memory
        let a_col = t * 16u + local_col;
        if (row < params.m && a_col < params.n) {
            tile_a[local_row * 16u + local_col] = a_data[row * params.n + a_col];
        } else {
            tile_a[local_row * 16u + local_col] = 0.0;
        }
        
        // Load tile of B into shared memory
        let b_row = t * 16u + local_row;
        if (b_row < params.n && col < params.p) {
            tile_b[local_row * 16u + local_col] = b_data[b_row * params.p + col];
        } else {
            tile_b[local_row * 16u + local_col] = 0.0;
        }
        
        workgroupBarrier();
        
        // Accumulate partial dot product
        for (var k = 0u; k < 16u; k = k + 1u) {
            sum = sum + tile_a[local_row * 16u + k] * tile_b[k * 16u + local_col];
        }
        
        workgroupBarrier();
    }
    
    c_data[row * params.p + col] = sum;
}
```

### 2.2 LoRA Merge

Two-dispatch approach: Dispatch 1 computes `A * input → intermediate[rank]` in parallel. Dispatch 2 computes `W * input + B * intermediate → output[out_dim]` in parallel.

```wgsl
// gpu/kernels/lora.wgsl

// Dispatch 1: Compute A * input → intermediate[rank]
// One invocation per rank element. Fully parallel across rank dimension.

@group(0) @binding(0) var<storage, read>       lora_a: array<f32>;       // [rank, n_embd]
@group(0) @binding(1) var<storage, read>       input: array<f32>;        // [n_embd]
@group(0) @binding(2) var<storage, read_write> intermediate: array<f32>; // [rank]
@group(0) @binding(3) var<uniform>             params: LoraParamsA;

struct LoraParamsA {
    rank: u32,
    n_embd: u32,
}

@compute @workgroup_size(64, 1, 1)
fn lora_a_forward(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let r = gid.x;
    if (r >= params.rank) { return; }
    
    var sum: f32 = 0.0;
    for (var j = 0u; j < params.n_embd; j = j + 1u) {
        sum = sum + lora_a[r * params.n_embd + j] * input[j];
    }
    intermediate[r] = sum;
}

// Dispatch 2: Compute output[i] = W[i,:] * input + alpha * B[i,:] * intermediate
// One invocation per output row. Fully parallel across out_dim.

@group(0) @binding(0) var<storage, read>       base_weight: array<f32>;  // [out_dim, n_embd]
@group(0) @binding(1) var<storage, read>       lora_b: array<f32>;       // [out_dim, rank]
@group(0) @binding(2) var<storage, read>       input: array<f32>;        // [n_embd]
@group(0) @binding(3) var<storage, read>       intermediate: array<f32>; // [rank] from dispatch 1
@group(0) @binding(4) var<storage, read_write> output: array<f32>;       // [out_dim]
@group(0) @binding(5) var<uniform>             params: LoraParamsB;

struct LoraParamsB {
    out_dim: u32,
    n_embd: u32,
    rank: u32,
    alpha: f32,
}

@compute @workgroup_size(64, 1, 1)
fn lora_b_forward(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let i = gid.x;
    if (i >= params.out_dim) { return; }
    
    // Base weight: W[i,:] * input
    var base_sum: f32 = 0.0;
    for (var j = 0u; j < params.n_embd; j = j + 1u) {
        base_sum = base_sum + base_weight[i * params.n_embd + j] * input[j];
    }
    
    // LoRA: B[i,:] * intermediate (intermediate = A * input, already computed)
    var lora_sum: f32 = 0.0;
    for (var r = 0u; r < params.rank; r = r + 1u) {
        lora_sum = lora_sum + lora_b[i * params.rank + r] * intermediate[r];
    }
    
    output[i] = base_sum + params.alpha * lora_sum;
}
```

### 2.3 Cross-Entropy Loss

Two-dispatch approach: Dispatch 1 computes per-sample softmax + loss in parallel. Dispatch 2 reduces to mean loss.

```wgsl
// gpu/kernels/loss.wgsl

// Dispatch 1: Per-sample softmax + cross-entropy (one workgroup invocation per sample)
// Each invocation handles one position in [0, batch_seq) independently.

@group(0) @binding(0) var<storage, read>       logits: array<f32>;      // [batch * seq * vocab]
@group(0) @binding(1) var<storage, read>       targets: array<u32>;     // [batch * seq]
@group(0) @binding(2) var<storage, read_write> per_sample_loss: array<f32>;  // [batch_seq] output
@group(0) @binding(3) var<storage, read_write> log_probs: array<f32>;   // [batch * seq * vocab] for backward
@group(0) @binding(4) var<uniform>             params: LossParams;

struct LossParams {
    batch_seq: u32,
    vocab_size: u32,
    total_tokens: u32,
}

@compute @workgroup_size(64, 1, 1)
fn cross_entropy_per_sample(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let i = gid.x;
    if (i >= params.batch_seq) { return; }
    
    let offset = i * params.vocab_size;
    let target = targets[i];
    
    // Find max for numerical stability
    var max_logit: f32 = logits[offset];
    for (var v = 1u; v < params.vocab_size; v = v + 1u) {
        let val = logits[offset + v];
        if (val > max_logit) { max_logit = val; }
    }
    
    // Compute sum of exp(logit - max) + normalize
    var sum_exp: f32 = 0.0;
    for (var v = 0u; v < params.vocab_size; v = v + 1u) {
        let exp_val = exp(logits[offset + v] - max_logit);
        log_probs[offset + v] = exp_val;
        sum_exp = sum_exp + exp_val;
    }
    
    // Normalize + compute loss
    var target_prob: f32 = 0.0;
    for (var v = 0u; v < params.vocab_size; v = v + 1u) {
        log_probs[offset + v] = log_probs[offset + v] / sum_exp;
        if (v == target) {
            target_prob = log_probs[offset + v];
        }
    }
    
    per_sample_loss[i] = -log(target_prob + 1e-10);
}

// Dispatch 2: Tree reduction to compute mean loss from per-sample losses.
// Called with workgroup_size=256. For batch_seq <= 256, single dispatch.
// For larger, chain multiple dispatches (future optimization).

@group(0) @binding(0) var<storage, read>       per_sample_loss: array<f32>;
@group(0) @binding(1) var<storage, read_write> loss: array<f32>;        // [1] output
@group(0) @binding(2) var<uniform>             params: LossParams;

var<workgroup> shared_loss: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn cross_entropy_reduce(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let i = gid.x;
    var val: f32 = 0.0;
    if (i < params.batch_seq) {
        val = per_sample_loss[i];
    }
    shared_loss[lid.x] = val;
    workgroupBarrier();
    
    // Tree reduction
    var stride = 128u;
    while (stride > 0u) {
        if (lid.x < stride) {
            shared_loss[lid.x] = shared_loss[lid.x] + shared_loss[lid.x + stride];
        }
        stride = stride >> 1u;
        workgroupBarrier();
    }
    
    if (lid.x == 0u) {
        loss[0] = shared_loss[0] / f32(params.total_tokens);
    }
}
```

### 2.4 Optimizer (AdamW)

```wgsl
// gpu/kernels/optimizer.wgsl

// AdamW optimizer step for LoRA parameters.
// Updates params in-place using gradient, momentum (m), and velocity (v).

@group(0) @binding(0) var<storage, read_write> params: array<f32>;
@group(0) @binding(1) var<storage, read>       grads: array<f32>;
@group(0) @binding(2) var<storage, read_write> m: array<f32>;      // first moment
@group(0) @binding(3) var<storage, read_write> v: array<f32>;      // second moment
@group(0) @binding(4) var<uniform>             opts: AdamWParams;

struct AdamWParams {
    lr: f32,           // learning rate
    beta1: f32,        // 0.9
    beta2: f32,        // 0.999
    eps: f32,          // 1e-8
    weight_decay: f32, // 0.01
    step: u32,         // current training step (for bias correction)
    param_count: u32,
}

@compute @workgroup_size(256, 1, 1)
fn adamw_step(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let i = gid.x;
    if (i >= opts.param_count) { return; }
    
    let g = grads[i];
    let current_m = m[i];
    let current_v = v[i];
    
    // Update moments
    let new_m = opts.beta1 * current_m + (1.0 - opts.beta1) * g;
    let new_v = opts.beta2 * current_v + (1.0 - opts.beta2) * g * g;
    m[i] = new_m;
    v[i] = new_v;
    
    // Bias correction
    let step_f = f32(opts.step);
    let m_hat = new_m / (1.0 - pow(opts.beta1, step_f));
    let v_hat = new_v / (1.0 - pow(opts.beta2, step_f));
    
    // AdamW: weight decay is applied directly to params (not through gradient)
    let decayed = params[i] * (1.0 - opts.lr * opts.weight_decay);
    
    // Parameter update
    params[i] = decayed - opts.lr * m_hat / (sqrt(v_hat) + opts.eps);
}
```

## Phase 3: Forward Pass on GPU

### 3.1 Pipeline Orchestration

```rust
// gpu/forward.rs

/// GPU forward pass state. Holds all buffers for one forward pass.
pub struct GpuForwardPass {
    ctx: Arc<GpuContext>,
    config: Config,
    
    // Weight buffers (uploaded once, read-only during training)
    weights: GpuWeightBuffers,
    
    // LoRA adapter buffers (updated by optimizer)
    lora: GpuLoraBuffers,
    
    // Activation buffers (intermediate results, needed for backward)
    activations: GpuActivationBuffers,
    
    // Pipeline objects (compiled shaders)
    pipelines: GpuPipelines,
}

/// All base model weight buffers on GPU.
pub struct GpuWeightBuffers {
    pub wte: wgpu::Buffer,       // [vocab_size, n_embd]
    pub wpe: wgpu::Buffer,       // [block_size, n_embd]
    pub lm_head: wgpu::Buffer,   // [vocab_size, n_embd]
    pub layers: Vec<GpuLayerBuffers>,  // [n_layer] — matches TransformerWeights.layers
}

pub struct GpuLayerBuffers {
    pub attn_wq: wgpu::Buffer,
    pub attn_wk: wgpu::Buffer,
    pub attn_wv: wgpu::Buffer,
    pub attn_wo: wgpu::Buffer,
    pub mlp_w1: wgpu::Buffer,
    pub mlp_w2: wgpu::Buffer,
}

/// LoRA adapter buffers. 6 adapters per layer (Q, K, V, O, MLP1, MLP2).
pub struct GpuLoraBuffers {
    pub adapters: Vec<GpuLoraAdapter>,  // one per adapter location
    pub rank: usize,
    pub alpha: f32,
}

pub struct GpuLoraAdapter {
    pub a: wgpu::Buffer,   // [rank, in_dim]  — Kaiming init
    pub b: wgpu::Buffer,   // [out_dim, rank] — zero init
    pub grad_a: wgpu::Buffer,
    pub grad_b: wgpu::Buffer,
    pub m_a: wgpu::Buffer, // AdamW first moment
    pub v_a: wgpu::Buffer, // AdamW second moment
    pub m_b: wgpu::Buffer,
    pub v_b: wgpu::Buffer,
}

/// Saved activations for backward pass.
pub struct GpuActivationBuffers {
    pub token_embed: wgpu::Buffer,     // [seq_len, n_embd]
    pub pos_embed: wgpu::Buffer,       // [seq_len, n_embd]
    pub hidden: wgpu::Buffer,          // [seq_len, n_embd]
    pub q: wgpu::Buffer,              // [seq_len, n_embd]
    pub k: wgpu::Buffer,
    pub v: wgpu::Buffer,
    pub attn_out: wgpu::Buffer,        // [seq_len, n_embd]
    pub mlp_hidden: wgpu::Buffer,      // [seq_len, mlp_hidden]
    pub logits: wgpu::Buffer,          // [seq_len, vocab_size]
}
```

### 3.2 Forward Dispatch

```rust
// gpu/forward.rs

impl GpuForwardPass {
    /// Run the full forward pass, returning logits buffer.
    /// Saves all activations for the backward pass.
    pub fn forward(
        &mut self,
        token_ids: &[usize],  // input token sequence
    ) -> Result<&wgpu::Buffer, GpuError> {
        let mut encoder = self.ctx.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("forward") }
        );
        
        // 1. Embedding lookup: hidden = wte[tokens] + wpe[positions]
        self.dispatch_embedding(&mut encoder, token_ids);
        
        // 2. For each layer in self.weights.layers: attention + MLP with LoRA
        for layer_idx in 0..self.config.n_layer {
            self.dispatch_layer(&mut encoder, layer_idx)?;
        }
        
        // 3. Final lm_head projection
        self.dispatch_lm_head(&mut encoder)?;
        
        self.ctx.queue.submit(std::iter::once(encoder.finish()));
        Ok(&self.activations.logits)
    }
}
```

## Phase 4: Backward Pass (LoRA Only)

Only LoRA parameters have gradients. Base weights are frozen.

```rust
// gpu/backward.rs

/// Backward pass: compute gradients for LoRA A and B only.
/// Base weights are frozen — no gradients computed for them.
///
/// Chain rule through LoRA:
///   forward:  lora_out = base_out + alpha * B @ (A @ input)
///   grad_A:   dL/dA = alpha * B^T @ dL/d_lora_out @ input^T
///   grad_B:   dL/dB = alpha * dL/d_lora_out @ (A @ input)^T
pub struct GpuBackwardPass {
    ctx: Arc<GpuContext>,
}

impl GpuBackwardPass {
    /// Compute LoRA gradients for one adapter.
    ///
    /// Requires:
    /// - dL/d_output (gradient flowing back from the next layer)
    /// - A @ input (cached from forward pass, the "lora_pre" activation)
    /// - input (cached from forward pass)
    ///
    /// Produces:
    /// - grad_A: [rank, in_dim]
    /// - grad_B: [out_dim, rank]
    pub fn compute_lora_gradients(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        adapter: &GpuLoraAdapter,
        grad_output: &wgpu::Buffer,    // dL/d_output: [seq_len, out_dim]
        lora_pre: &wgpu::Buffer,       // A @ input: [seq_len, rank]
        input: &wgpu::Buffer,          // input: [seq_len, in_dim]
    ) -> Result<(), GpuError> {
        // grad_B = alpha * grad_output^T @ lora_pre
        //         = alpha * [out_dim, seq_len] @ [seq_len, rank]
        //         → [out_dim, rank]
        self.dispatch_matmul(
            encoder,
            /* a = */ grad_output,  /* transpose_a = */ true,
            /* b = */ lora_pre,     /* transpose_b = */ false,
            /* c = */ &adapter.grad_b,
            /* m = */ out_dim, /* n = */ seq_len, /* p = */ rank,
        )?;
        
        // grad_A = alpha * B^T @ grad_output @ input^T
        //        = alpha * (B^T @ grad_output) @ input^T
        // Step 1: temp = B^T @ grad_output → [rank, seq_len]
        // Step 2: grad_A = temp @ input^T → [rank, in_dim]
        self.dispatch_matmul(
            encoder,
            /* a = */ &adapter.b,      /* transpose_a = */ true,
            /* b = */ grad_output,      /* transpose_b = */ false,
            /* c = */ &temp_buffer,     /* [rank, seq_len] */
            /* m = */ rank, /* n = */ out_dim, /* p = */ seq_len,
        )?;
        self.dispatch_matmul(
            encoder,
            /* a = */ &temp_buffer,     /* transpose_a = */ false,
            /* b = */ input,            /* transpose_b = */ true,
            /* c = */ &adapter.grad_a,  /* [rank, in_dim] */
            /* m = */ rank, /* n = */ seq_len, /* p = */ in_dim,
        )?;
        
        Ok(())
    }
}
```

## Phase 5: Training Loop

### 5.1 DataLoader

```rust
// gpu/dataloader.rs

/// Batches training samples from JSONL for GPU consumption.
/// Handles shuffling, padding, and sequence length truncation.
pub struct DataLoader {
    samples: Vec<TrainingSample>,
    batch_size: usize,
    seq_len: usize,
    pad_id: usize,
    rng: Rng,
}

impl DataLoader {
    /// Create dataloader from a JSONL file (produced by plan 007's exporter).
    pub fn from_jsonl(path: &Path, batch_size: usize, seq_len: usize, pad_id: usize) -> Result<Self, DataLoaderError> {
        let file = std::fs::File::open(path)?;
        let samples: Vec<TrainingSample> = std::io::BufReader::new(file)
            .lines()
            .filter_map(|line| {
                let line = line.ok()?;
                serde_json::from_str(&line).ok()
            })
            .collect();
        
        Ok(Self {
            samples,
            batch_size,
            seq_len,
            pad_id,
            rng: Rng::new(42),
        })
    }
    
    /// Iterate over batches. Each batch is (input_ids, target_ids) as flat f32.
    /// input_ids: [batch_size, seq_len]  — the token sequence
    /// target_ids: [batch_size, seq_len] — shifted by 1 (next-token prediction)
    pub fn batches(&mut self) -> impl Iterator<Item = (Vec<u32>, Vec<u32>)> {
        // Shuffle samples
        self.shuffle();
        
        self.samples.chunks(self.batch_size).map(move |batch| {
            let mut input_ids = Vec::with_capacity(batch.len() * self.seq_len);
            let mut target_ids = Vec::with_capacity(batch.len() * self.seq_len);
            
            for sample in batch {
                let tokens = &sample.tokens;
                // Truncate or pad to seq_len
                for t in 0..self.seq_len {
                    if t + 1 < tokens.len() {
                        input_ids.push(tokens[t] as u32);
                        target_ids.push(tokens[t + 1] as u32);
                    } else {
                        input_ids.push(self.pad_id as u32);
                        target_ids.push(self.pad_id as u32);
                    }
                }
            }
            
            (input_ids, target_ids)
        })
    }
}
```

### 5.2 Training Loop

```rust
// gpu/training_loop.rs

/// Main training loop: iterate epochs, run forward+backward, update LoRA.
pub struct Trainer {
    forward: GpuForwardPass,
    backward: GpuBackwardPass,
    optimizer: AdamWOptimizer,
    dataloader: DataLoader,
    config: TrainingConfig,
}

pub struct TrainingConfig {
    pub epochs: usize,
    pub learning_rate: f32,
    pub weight_decay: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub warmup_steps: usize,
    pub log_interval: usize,
    pub checkpoint_interval: usize,
    pub checkpoint_dir: String,
}

impl Trainer {
    /// Run the full training loop.
    pub fn train(&mut self) -> Result<TrainingReport, GpuError> {
        let mut step = 0u32;
        let mut total_loss = 0.0;
        let mut best_loss = f32::MAX;
        
        for epoch in 0..self.config.epochs {
            for (input_ids, target_ids) in self.dataloader.batches() {
                // 1. Forward pass
                let logits = self.forward.forward(&input_ids)?;
                
                // 2. Compute loss
                let loss = self.compute_loss(logits, &target_ids)?;
                
                // 3. Backward pass (LoRA gradients only)
                self.backward.compute_all_gradients(
                    &self.forward.activations,
                    &self.forward.lora,
                )?;
                
                // 4. Optimizer step
                let lr = self.schedule_lr(step);
                self.optimizer.step(lr, step)?;
                
                // 5. Logging
                total_loss += loss;
                step += 1;
                
                if step % self.config.log_interval as u32 == 0 {
                    let avg_loss = total_loss / self.config.log_interval as f32;
                    println!("[step {step}] epoch={epoch} loss={avg_loss:.4}");
                    total_loss = 0.0;
                }
                
                // 6. Checkpoint
                if step % self.config.checkpoint_interval as u32 == 0 {
                    if loss < best_loss {
                        self.save_checkpoint(step, loss)?;
                        best_loss = loss;
                    }
                }
            }
        }
        
        Ok(TrainingReport { steps: step, best_loss })
    }
}
```

## Phase 6: LoRA Export (`lora.bin`)

After training, export LoRA weights to a portable format.

```rust
// gpu/lora.rs

/// Export trained LoRA adapters to `lora.bin` (safetensors format).
pub fn export_lora(adapters: &[GpuLoraAdapter], config: &Config, path: &Path) -> Result<(), GpuError> {
    let mut tensors = HashMap::new();
    
    for (i, adapter) in adapters.iter().enumerate() {
        // Download A and B matrices from GPU
        let a_data = download_f32(&adapter.a, adapter_rank * config.n_embd)?;
        let b_data = download_f32(&adapter.b, out_dim * adapter_rank)?;
        
        tensors.insert(format!("lora.{i}.a"), TensorData::F32(a_data));
        tensors.insert(format!("lora.{i}.b"), TensorData::F32(b_data));
    }
    
    // Write as safetensors
    safetensors::serialize_to_file(&tensors, &HashMap::new(), path)?;
    Ok(())
}

/// Load a `lora.bin` and inject into a GpuForwardPass.
pub fn load_lora(path: &Path, forward: &mut GpuForwardPass) -> Result<(), GpuError> {
    let data = std::fs::read(path)?;
    let tensors = safetensors::deserialize(&data)?;
    
    for (i, adapter) in forward.lora.adapters.iter_mut().enumerate() {
        let a = tensors.get(&format!("lora.{i}.a")).ok_or(GpuError::InvalidFormat)?;
        let b = tensors.get(&format!("lora.{i}.b")).ok_or(GpuError::InvalidFormat)?;
        
        upload_f32(&forward.ctx.device, a.data(), &format!("lora_{i}_a"))?;
        upload_f32(&forward.ctx.device, b.data(), &format!("lora_{i}_b"))?;
    }
    
    Ok(())
}
```

Note: safetensors may not compile on WASM targets. For WASM, use a simpler export format: `[blake3_hash(4B) | n_layers(4B) | rank(4B) | layer_data...]` where each layer_data is `[a_len(4B) | a_data | b_len(4B) | b_data]`. Gate safetensors behind a native-only path.

## Phase 7: Integration with Deterministic Validator Pipeline

### Data Flow from Plan 007 → Plan 008

```
Plan 007 (Deterministic Validator)  Plan 008 (wgpu LoRA Training)
─────────────────                   ──────────────────────────────
rust-lang/rust ─┐
top 1000 crates─┼─► CorpusIngester
rust docs ──────┘       │
                       ▼
                  TrainingFilter
                  (syn + cargo check)
                       │
                       ▼
                  TrainingExporter
                  → training.jsonl ──────────► DataLoader.from_jsonl()
                                                    │
                                                    ▼
                                              GPU Training Loop
                                              (forward → loss → backward → update)
                                                    │
                                                    ▼
                                              lora.bin (safetensors)
                                                    │
                                                    ▼
                                              Load into draft model
                                              → higher acceptance rate
```

### Config Compatibility

Plan 007 redefines `Config` with BPE dimensions. Plan 008's GPU buffers must match:

```rust
// types.rs — extended for LoRA + GPU

pub struct Config {
    // ... existing fields from plan 007 (vocab_size=4096, n_embd=32, n_layer=1) ...
    
    pub n_layer: usize,           // number of transformer layers (default: 1)
    
    // LoRA fields (new)
    pub lora_rank: usize,         // rank of LoRA adapters (default: 4)
    pub lora_alpha: f32,          // LoRA scaling factor (default: 8.0)
    pub lora_dropout: f32,        // dropout during training (default: 0.0)
    pub lora_targets: Vec<String>,// which weights get adapters (default: ["q","k","v","o","mlp1","mlp2"])
}

impl Config {
    /// Micro config with LoRA defaults.
    pub fn micro_lora() -> Self {
        let mut c = Self::micro();
        c.n_layer = 1;
        c.lora_rank = 4;
        c.lora_alpha = 8.0;
        c.lora_dropout = 0.0;
        c.lora_targets = vec![
            "q".into(), "k".into(), "v".into(), "o".into(),
            "mlp1".into(), "mlp2".into(),
        ];
        c
    }
}
```

**Note**: Plan 007 defines `vocab_size=4096` for `Config::bpe()`. Plan 008's GPU buffers must match this. The "32K vocab" mentioned in the parameter estimates table is for a future "validator large" config (previously "cLoRA large") that is NOT yet defined — it will be added when multi-layer support lands. All Plan 008 development and testing uses `vocab_size=4096` from Plan 007.

**Note on multi-layer**: Plan 010 already implemented multi-layer support. `TransformerWeights` uses `layers: Vec<LayerWeights>`, `Config` has `n_layer: usize`, and `forward()` has a layer loop. The GPU forward pass must iterate `layers` and allocate per-layer activation buffers.

## Phase 8: Benchmarking

### Benchmarks

| Benchmark | What It Measures | Target |
|-----------|-----------------|--------|
| `bench_gpu_matmul` | GPU matmul throughput vs CPU | ≥10× CPU for n_embd ≥ 128 |
| `bench_gpu_forward` | Full forward pass GPU vs CPU | ≥5× CPU for n_embd ≥ 128 |
| `bench_gpu_backward` | LoRA backward pass | ≤2× forward time |
| `bench_gpu_training_step` | Full step (fwd + bwd + opt) | End-to-end time per step |
| `bench_gpu_vs_cpu_loss` | Verify GPU and CPU produce same loss | <0.1% difference |
| `bench_lora_convergence` | Loss curve over 1000 steps on toy data | Monotonically decreasing |

### Validation

1. **Correctness**: Run forward pass on CPU and GPU with same weights → logits must match within float epsilon
2. **Convergence**: Train on toy dataset → loss must decrease over epochs
3. **Gradient check**: Numerical gradient vs analytical gradient for LoRA params → relative error < 1e-4
4. **Export/Import**: Train → export `lora.bin` → load → forward pass must produce same logits

## Tasks

### Phase 1: wgpu Context & Buffers
- [x] 1.1 Add `wgpu`, `bytemuck`, `pollster` to `Cargo.toml` behind `gpu` feature
- [x] 1.2 Create `src/gpu/mod.rs` with feature gate
- [x] 1.3 Create `src/gpu/context.rs` — `GpuContext::new()`, error types
- [x] 1.4 Create `src/gpu/buffer.rs` — `upload_f32`, `download_f32`, `create_buffer`
- [x] 1.5 Add tests: context init, buffer upload/download roundtrip
- [x] 1.5.1 Add `#[cfg(target_arch = "wasm32")]` conditional for async GPU init (use wasm-bindgen-futures instead of pollster::block_on)
- [x] 1.6 Verify compilation on WASM target (`cargo build --target wasm32-unknown-unknown --features gpu`)

### Phase 2: WGSL Compute Shaders
- [ ] 2.1 Create `src/gpu/kernels/mod.rs` — shader loading, pipeline creation helpers
- [x] 2.2 Write `matmul.wgsl` — tiled matrix multiply with shared memory
- [x] 2.3 Write `elementwise.wgsl` — add, multiply, ReLU, scale, copy
- [x] 2.4 Write `softmax.wgsl` — stable online softmax
- [x] 2.5 Write `layernorm.wgsl` — RMSNorm
- [x] 2.6 Write `embedding.wgsl` — token + position embedding lookup
- [x] 2.7 Write `attention.wgsl` — scaled dot-product attention
- [x] 2.8 Write `lora.wgsl` — LoRA merge: Y = Wx + alpha * BAx
- [x] 2.9 Write `loss.wgsl` — cross-entropy with softmax
- [x] 2.10 Write `optimizer.wgsl` — AdamW parameter update
- [ ] 2.11 Add tests: each shader against CPU reference (matmul correctness, softmax sum=1, etc.)
- [ ] 2.12 Benchmark: `bench_gpu_matmul` vs CPU matmul at various sizes

### Phase 3: Forward Pass on GPU
- [ ] 3.1 Create `src/gpu/forward.rs` — `GpuForwardPass` struct
- [ ] 3.2 Create `src/gpu/lora.rs` — `GpuLoraAdapter`, `GpuLoraBuffers`, init (Kaiming + zeros)
- [ ] 3.3 Implement weight upload: `TransformerWeights` → `GpuWeightBuffers`
- [ ] 3.4 Implement `dispatch_embedding` — token + position lookup
- [ ] 3.5 Implement `dispatch_layer` — attention with LoRA + MLP with LoRA
- [ ] 3.6 Implement `dispatch_lm_head` — final projection
- [ ] 3.7 Add test: GPU forward produces same logits as CPU forward (within epsilon)
- [ ] 3.8 Benchmark: `bench_gpu_forward` vs CPU forward

### Phase 4: Backward Pass (LoRA Only)
- [ ] 4.1 Create `src/gpu/backward.rs` — `GpuBackwardPass` struct
- [ ] 4.2 Implement `compute_lora_gradients` — grad_A, grad_B via matmul
- [ ] 4.3 Implement full backward pass through all layers (reverse order)
- [ ] 4.4 Add test: numerical gradient check vs analytical gradient (relative error < 1e-4)
- [ ] 4.5 Benchmark: `bench_gpu_backward` vs forward time

### Phase 5: Training Loop
- [ ] 5.1 Create `src/gpu/dataloader.rs` — JSONL loading, batching, shuffling (NOT src/data/ — that's Plan 009)
- [ ] 5.2 Create `src/gpu/loss.rs` — cross-entropy loss dispatch
- [ ] 5.3 Create `src/gpu/optimizer.rs` — AdamW state management, step dispatch
- [ ] 5.4 Create `src/gpu/training_loop.rs` — `Trainer`, epoch loop, logging
- [ ] 5.5 Add test: train on 10 toy samples → loss decreases over 100 steps
- [ ] 5.6 Add test: full training on toy model → `lora.bin` export → load → verify
- [ ] 5.7 Benchmark: `bench_lora_convergence` — loss curve over 1000 steps

### Phase 6: LoRA Export/Import
- [ ] 6.1 Implement `export_lora` — download A/B from GPU → safetensors file
- [ ] 6.2 Implement `load_lora` — read safetensors → upload to GPU buffers
- [ ] 6.3 Verify `safetensors` dependency resolves (already added in Phase 1 deps)
- [ ] 6.4 Add test: export → load → forward pass produces same logits
- [ ] 6.5 Add CLI command: `cargo run --features gpu -- train --data training.jsonl --output lora.bin`

### Phase 7: Deterministic Validator Integration
- [ ] 7.1 Update `Config` with LoRA fields (rank, alpha, dropout, targets) — note: `n_layer` already exists from Plan 010
- [ ] 7.2 Verify GPU buffer sizes match plan 007's BPE dimensions (vocab_size=4096, n_embd=32, n_layer=1)
- [ ] 7.2.1 Update `GpuWeightBuffers` to use `Vec<GpuLayerWeights>` matching the `Vec<LayerWeights>` structure from Plan 010
- [ ] 7.3 Add integration test: load plan 007's JSONL → train → export lora.bin
- [ ] 7.4 Document the data flow: plan 007 JSONL → plan 008 training → lora.bin

### Phase 8: Benchmarking & Validation
- [ ] 8.1 Add `bench_gpu_matmul` to benchmark suite
- [ ] 8.2 Add `bench_gpu_forward` to benchmark suite
- [ ] 8.3 Add `bench_gpu_training_step` to benchmark suite
- [ ] 8.4 Run correctness validation: GPU vs CPU loss comparison
- [ ] 8.5 Run convergence benchmark on toy data → `bench/018_gpu_lora_convergence.png`
- [ ] 8.6 Run scaling benchmark (vary n_embd: 16, 64, 256) → `bench/019_gpu_scaling.png`
- [ ] 8.7 Run WASM benchmark in browser (if applicable)

## Feature Flags

```toml
[features]
default = []
leviathan = []
sudoku = []
validator = ["syn"]                 # previously clora
training = ["serde", "serde_json", "walkdir"]
gpu = ["wgpu", "bytemuck", "pollster"]             # GPU training backend
full = ["leviathan", "sudoku", "validator", "training", "gpu"]    # previously clora
```

## Key Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|-----------|
| WGSL shared memory limits (16KB per workgroup) | Matmul tile size limited | Use 16×16 tiles (4KB each); fall back to 8×8 if needed |
| No cooperative groups in WebGPU | Can't do persistent megakernel | Chain dispatches in command encoder; minimize CPU round-trips |
| GPU-CPU transfer latency | Download/upload bottleneck | Keep data on GPU between steps; only download for checkpointing |
| WGSL doesn't support fp16/bf16 | Wider buffers, more bandwidth | f32 is fine for training (precision > speed at small scale) |
| WASM async limitations | Can't block on GPU in browser | Use `wasm-bindgen-futures` for async GPU ops on WASM |
| Numerical precision GPU vs CPU | Loss values differ slightly | Use stable softmax (max subtraction); test within epsilon |
| `naga` shader compilation errors | WGSL may not map to all backends | Test on Metal + Vulkan + WebGPU; avoid backend-specific ops |

## Expected Outcomes

1. **GPU Training Backend**: A `wgpu`-based training pipeline that runs forward + backward + optimizer step entirely on GPU
2. **WGSL Shader Library**: Reusable compute shaders for matmul, attention, softmax, layernorm, LoRA, loss, optimizer
3. **`lora.bin` Export**: Trained LoRA weights serialized as safetensors, loadable for inference
4. **Cross-Platform**: Same code runs on Metal (macOS), Vulkan (Linux), DX12 (Windows), and WebGPU (WASM/browser)
5. **Correctness**: GPU and CPU forward passes produce identical results within float epsilon
6. **Convergence**: Training loop produces monotonically decreasing loss on validation data

## Prerequisites

- **Plan 007 (Deterministic Validator)**: ✅ **Complete.** BPE Tokenizer, SynPruner, and all phases are done.
  - `Config::bpe()` / `Config::bpe_draft()` define vocabulary and dimensions
  - Phase 3 (Training Data Pipeline) was deferred to Plan 009 — `training.jsonl` not yet available
- The toy model (`Config::micro()`) can be used for development and testing without the validator feature
- **Next step**: Complete 2.1 (`kernels/mod.rs` pipeline helpers), then 2.11–2.12 (shader tests/benchmarks), then proceed to Phase 3

## Files to Create/Modify

| File | Action | Phase |
|------|--------|-------|
| `Cargo.toml` | Add wgpu, bytemuck, pollster, safetensors deps + gpu feature | 1 |
| `src/gpu/mod.rs` | New | 1 |
| `src/gpu/context.rs` | New | 1 |
| `src/gpu/buffer.rs` | New | 1 |
| `src/gpu/kernels/mod.rs` | New | 2 |
| `src/gpu/kernels/matmul.wgsl` | New | 2 |
| `src/gpu/kernels/elementwise.wgsl` | New | 2 |
| `src/gpu/kernels/softmax.wgsl` | New | 2 |
| `src/gpu/kernels/layernorm.wgsl` | New | 2 |
| `src/gpu/kernels/embedding.wgsl` | New | 2 |
| `src/gpu/kernels/attention.wgsl` | New | 2 |
| `src/gpu/kernels/lora.wgsl` | New | 2 |
| `src/gpu/kernels/loss.wgsl` | New | 2 |
| `src/gpu/kernels/optimizer.wgsl` | New | 2 |
| `src/gpu/forward.rs` | New | 3 |
| `src/gpu/lora.rs` | New | 3 |
| `src/gpu/backward.rs` | New | 4 |
| `src/gpu/dataloader.rs` | New | 5 |
| `src/gpu/loss.rs` | New | 5 |
| `src/gpu/optimizer.rs` | New | 5 |
| `src/gpu/training_loop.rs` | New | 5 |
| `src/types.rs` | Add LoRA config fields | 7 |
| `src/lib.rs` | Add `mod gpu` behind feature gate | 1 |
| `src/benchmark.rs` | Add GPU benchmarks | 8 |

## References

- `.plans/007_constraint_validator.md` — Produces training data JSONL, defines BPE dimensions (previously `007_compiler_in_the_loop_clora.md`)
- `.research/01_Advanced Neuro-Symbolic Rust Translation.md` — LoRA training loop in the 32-day cycle
- [wgpu repo](https://github.com/gfx-rs/wgpu) — Cross-platform GPU API for Rust
- [WebGPU Compute Shaders](https://webgpufundamentals.org/webgpu/lessons/webgpu-compute-shaders.html) — WGSL basics
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — Hu et al., 2021
- [Percepta](https://github.com/Percepta-Core/transformer-vm) — Analytical weight compilation reference