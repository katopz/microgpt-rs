// GPU forward pass: uploads weights, runs forward pass with LoRA, saves activations.
// Processes positions sequentially (autoregressive), matching CPU forward pass logic.

use std::sync::Arc;

use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, Buffer, BufferDescriptor,
    BufferUsages, CommandEncoder, ComputePassDescriptor,
};

use crate::gpu::buffer::{create_buffer, download_f32, upload_f32};
use crate::gpu::context::GpuError;
use crate::gpu::kernels::{GpuPipelines, dispatch_1d};
use crate::gpu::lora::{GpuLoraBuffers, LoraTarget};
use crate::transformer::TransformerWeights;
use crate::types::Config;

// ── Uniform param structs (16-byte aligned for WGSL) ───────────────

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MatmulParams {
    pub m: u32,
    pub n: u32,
    pub p: u32,
    pub _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ElementwiseParams {
    count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SoftmaxParams {
    rows: u32,
    cols: u32,
    _pad0: u32,
    _pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct LayernormParams {
    batch_seq: u32,
    dim: u32,
    _pad0: u32,
    _pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct EmbeddingParams {
    batch_seq: u32,
    n_embd: u32,
    vocab_size: u32,
    block_size: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct AttnScoreParams {
    head_dim: u32,
    pos: u32,
    scale: f32,
    kv_offset: u32,
    kv_stride: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct LoraParamsA {
    rank: u32,
    n_embd: u32,
    _pad0: u32,
    _pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct LoraParamsB {
    out_dim: u32,
    n_embd: u32,
    rank: u32,
    alpha: f32,
}

// ── Weight buffers ─────────────────────────────────────────────────

/// Per-layer weight buffers on GPU.
pub struct GpuLayerWeights {
    pub attn_wq: Buffer,
    pub attn_wk: Buffer,
    pub attn_wv: Buffer,
    pub attn_wo: Buffer,
    pub mlp_w1: Buffer,
    pub mlp_w2: Buffer,
}

/// All model weight buffers on GPU.
pub struct GpuWeightBuffers {
    pub wte: Buffer,
    pub wpe: Buffer,
    pub lm_head: Buffer,
    pub layers: Vec<GpuLayerWeights>,
}

impl GpuWeightBuffers {
    /// Upload CPU transformer weights to GPU buffers.
    pub fn from_weights(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        weights: &TransformerWeights,
    ) -> Self {
        let wte = upload_f32(device, queue, &weights.wte, "wte");
        let wpe = upload_f32(device, queue, &weights.wpe, "wpe");
        let lm_head = upload_f32(device, queue, &weights.lm_head, "lm_head");

        let layers = weights
            .layers
            .iter()
            .enumerate()
            .map(|(i, l)| GpuLayerWeights {
                attn_wq: upload_f32(device, queue, &l.attn_wq, &format!("l{i}_wq")),
                attn_wk: upload_f32(device, queue, &l.attn_wk, &format!("l{i}_wk")),
                attn_wv: upload_f32(device, queue, &l.attn_wv, &format!("l{i}_wv")),
                attn_wo: upload_f32(device, queue, &l.attn_wo, &format!("l{i}_wo")),
                mlp_w1: upload_f32(device, queue, &l.mlp_w1, &format!("l{i}_mlp1")),
                mlp_w2: upload_f32(device, queue, &l.mlp_w2, &format!("l{i}_mlp2")),
            })
            .collect();

        Self {
            wte,
            wpe,
            lm_head,
            layers,
        }
    }
}

// ── Activation buffers ─────────────────────────────────────────────

/// Saved activations for backward pass.
pub struct GpuActivationBuffers {
    // Per-position working buffers (reused each position)
    pub hidden: Buffer,     // [n_embd]
    pub residual: Buffer,   // [n_embd]
    pub residual2: Buffer,  // [n_embd]
    pub q: Buffer,          // [n_embd]
    pub k: Buffer,          // [n_embd]
    pub v: Buffer,          // [n_embd]
    pub attn_out: Buffer,   // [n_embd]
    pub mlp_hidden: Buffer, // [mlp_hidden]

    // Temp output buffer for ops where output aliases an input (WebGPU restriction).
    // Must be >= max(n_embd, mlp_hidden, kv_dim, vocab_size).
    pub temp_out: Buffer,

    // KV cache per layer: [block_size * kv_dim]
    pub key_cache: Vec<Buffer>,
    pub value_cache: Vec<Buffer>,

    // Output logits for all positions: [seq_len * vocab_size]
    pub logits: Buffer,

    // LoRA intermediates per adapter: A @ input [rank]
    pub lora_intermediates: Vec<Buffer>,

    // Saved inputs per adapter (for backward): the input to each LoRA projection
    pub lora_inputs: Vec<Buffer>,
}

impl GpuActivationBuffers {
    /// Create activation buffers for given config, seq_len, and lora.
    pub fn new(
        device: &wgpu::Device,
        config: &Config,
        seq_len: usize,
        lora: &GpuLoraBuffers,
    ) -> Self {
        let n = config.n_embd;
        let kvd = config.n_kv_head * config.head_dim;

        let hidden = create_buffer(device, n, "act_hidden");
        let residual = create_buffer(device, n, "act_residual");
        let residual2 = create_buffer(device, n, "act_residual2");
        let q = create_buffer(device, n, "act_q");
        let k = create_buffer(device, n, "act_k"); // n_embd >= kv_dim for micro config
        let v = create_buffer(device, n, "act_v");
        let attn_out = create_buffer(device, n, "act_attn_out");
        let mlp_hidden = create_buffer(device, config.mlp_hidden, "act_mlp_hidden");

        // Temp buffer for aliased output ops (must fit any single output)
        let temp_size = n.max(config.mlp_hidden).max(config.vocab_size);
        let temp_out = create_buffer(device, temp_size, "act_temp_out");

        // KV cache: one per layer
        let key_cache = (0..config.n_layer)
            .map(|i| create_buffer(device, config.block_size * kvd, &format!("key_cache_{i}")))
            .collect();
        let value_cache = (0..config.n_layer)
            .map(|i| create_buffer(device, config.block_size * kvd, &format!("val_cache_{i}")))
            .collect();

        // Logits: one row per position
        let logits = create_buffer(device, seq_len * config.vocab_size, "logits");

        // LoRA intermediates and inputs
        let lora_intermediates: Vec<Buffer> = lora
            .adapters
            .iter()
            .enumerate()
            .map(|(i, a)| create_buffer(device, a.rank, &format!("lora_inter_{i}")))
            .collect();
        let lora_inputs: Vec<Buffer> = lora
            .adapters
            .iter()
            .enumerate()
            .map(|(i, a)| create_buffer(device, a.in_dim, &format!("lora_input_{i}")))
            .collect();

        Self {
            hidden,
            residual,
            residual2,
            q,
            k,
            v,
            attn_out,
            mlp_hidden,
            temp_out,
            key_cache,
            value_cache,
            logits,
            lora_intermediates,
            lora_inputs,
        }
    }
}

// ── GPU Forward Pass ───────────────────────────────────────────────

/// Main GPU forward pass orchestrator.
pub struct GpuForwardPass {
    pub ctx: Arc<crate::gpu::GpuContext>,
    pub config: Config,
    pub weights: GpuWeightBuffers,
    pub lora: GpuLoraBuffers,
    pub activations: GpuActivationBuffers,
    pipelines: Arc<GpuPipelines>,

    // Reusable uniform buffers (one per shader type)
    uniform_matmul: Buffer,
    uniform_elem: Buffer,
    #[allow(dead_code)] // Reserved for future GPU-native softmax dispatch
    uniform_softmax: Buffer,
    uniform_layernorm: Buffer,
    #[allow(dead_code)] // Reserved for future GPU-native embedding dispatch
    uniform_embedding: Buffer,
    uniform_attn_score: Buffer,
    uniform_lora_a: Buffer,
    uniform_lora_b: Buffer,
}

impl GpuForwardPass {
    /// Create a new GPU forward pass.
    pub fn new(
        ctx: Arc<crate::gpu::GpuContext>,
        config: Config,
        weights: &TransformerWeights,
        lora: GpuLoraBuffers,
        seq_len: usize,
    ) -> Result<Self, GpuError> {
        let pipelines = Arc::new(GpuPipelines::new(&ctx.device));
        let gpu_weights = GpuWeightBuffers::from_weights(&ctx.device, &ctx.queue, weights);
        let activations = GpuActivationBuffers::new(&ctx.device, &config, seq_len, &lora);

        let uniform_matmul = Self::create_uniform(&ctx.device, "u_matmul");
        let uniform_elem = Self::create_uniform(&ctx.device, "u_elem");
        let uniform_softmax = Self::create_uniform(&ctx.device, "u_softmax");
        let uniform_layernorm = Self::create_uniform(&ctx.device, "u_layernorm");
        let uniform_embedding = Self::create_uniform(&ctx.device, "u_embedding");
        let uniform_attn_score = Self::create_uniform(&ctx.device, "u_attn_score");
        let uniform_lora_a = Self::create_uniform(&ctx.device, "u_lora_a");
        let uniform_lora_b = Self::create_uniform(&ctx.device, "u_lora_b");

        Ok(Self {
            ctx,
            config,
            weights: gpu_weights,
            lora,
            activations,
            pipelines,
            uniform_matmul,
            uniform_elem,
            uniform_softmax,
            uniform_layernorm,
            uniform_embedding,
            uniform_attn_score,
            uniform_lora_a,
            uniform_lora_b,
        })
    }

    fn create_uniform(device: &wgpu::Device, label: &str) -> Buffer {
        device.create_buffer(&BufferDescriptor {
            label: Some(label),
            size: 64, // enough for all param structs
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// Write uniform data to buffer.
    fn write_uniform<T: bytemuck::Pod>(&self, buffer: &Buffer, data: &T) {
        let bytes = bytemuck::cast_slice(std::slice::from_ref(data));
        self.ctx.queue.write_buffer(buffer, 0, bytes);
    }

    // ── Bind group helpers ──────────────────────────────────────────

    fn make_bg(
        &self,
        layout: &BindGroupLayout,
        entries: &[BindGroupEntry],
        label: &str,
    ) -> BindGroup {
        self.ctx.device.create_bind_group(&BindGroupDescriptor {
            label: Some(label),
            layout,
            entries,
        })
    }

    /// Check if two buffers are the same object by comparing their internal pointers.
    /// wgpu::Buffer wraps an Arc, so we compare the raw pointer identity.
    fn same_buffer(a: &Buffer, b: &Buffer) -> bool {
        std::ptr::eq(a as *const Buffer, b as *const Buffer)
    }

    fn entry(binding: u32, buffer: &Buffer) -> BindGroupEntry<'_> {
        BindGroupEntry {
            binding,
            resource: buffer.as_entire_binding(),
        }
    }

    // ── Forward pass ────────────────────────────────────────────────

    /// Run forward pass for a sequence of tokens.
    /// Returns the logits buffer [seq_len * vocab_size].
    pub fn forward(&self, token_ids: &[usize]) -> Result<&Buffer, GpuError> {
        // Upload token IDs as u32 buffer
        let tokens_u32: Vec<u32> = token_ids.iter().map(|&t| t as u32).collect();
        let _tokens_buf = upload_f32(
            &self.ctx.device,
            &self.ctx.queue,
            bytemuck::cast_slice(&tokens_u32),
            "tokens",
        );
        // We need a u32 storage buffer, but upload_f32 creates f32. Re-create properly.
        let tokens_bytes = bytemuck::cast_slice(&tokens_u32);
        let tokens_buf = self.ctx.device.create_buffer(&BufferDescriptor {
            label: Some("tokens"),
            size: tokens_bytes.len() as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.ctx.queue.write_buffer(&tokens_buf, 0, tokens_bytes);

        for (pos, &token_id) in token_ids.iter().enumerate() {
            let mut encoder =
                self.ctx
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some(&format!("forward_pos_{pos}")),
                    });

            // 1. Embedding: hidden = wte[token] + wpe[pos]
            self.dispatch_embedding(&mut encoder, &tokens_buf, token_id, pos)?;

            // 2. Layer loop
            for layer_idx in 0..self.config.n_layer {
                self.dispatch_layer(&mut encoder, layer_idx, pos)?;
            }

            // 3. LM head: logits[pos * vocab..(pos+1) * vocab] = lm_head @ hidden
            self.dispatch_lm_head(&mut encoder, pos)?;

            self.ctx.queue.submit(std::iter::once(encoder.finish()));
        }

        Ok(&self.activations.logits)
    }

    /// Embedding lookup: hidden = wte[token] + wpe[pos].
    /// Uses a simplified approach: dispatch embedding shader for single position.
    fn dispatch_embedding(
        &self,
        encoder: &mut CommandEncoder,
        _tokens_buf: &Buffer,
        token_id: usize,
        pos: usize,
    ) -> Result<(), GpuError> {
        let n = self.config.n_embd;

        // CPU embedding: wte[token_id*n..(token_id+1)*n] + wpe[pos*n..(pos+1)*n] → hidden
        // Use elementwise add: hidden = wte_row + wpe_row
        // First, copy wte row to hidden using copy shader, then add wpe row
        // Simpler: do it on CPU for single position, upload. For training, this is fast enough.
        let wte_data = download_f32(
            &self.ctx.device,
            &self.ctx.queue,
            &self.weights.wte,
            self.config.vocab_size * n,
        )
        .map_err(|e| GpuError::BufferError(format!("wte download: {e}")))?;
        let wpe_data = download_f32(
            &self.ctx.device,
            &self.ctx.queue,
            &self.weights.wpe,
            self.config.block_size * n,
        )
        .map_err(|e| GpuError::BufferError(format!("wpe download: {e}")))?;

        // wte indexed by token_id, wpe indexed by position
        let hidden_data: Vec<f32> = (0..n)
            .map(|i| wte_data[token_id * n + i] + wpe_data[pos * n + i])
            .collect();
        self.ctx.queue.write_buffer(
            &self.activations.hidden,
            0,
            bytemuck::cast_slice(&hidden_data),
        );

        let _ = encoder; // Acknowledge encoder for future GPU-native impl
        Ok(())
    }

    /// Process one transformer layer.
    fn dispatch_layer(
        &self,
        encoder: &mut CommandEncoder,
        layer_idx: usize,
        pos: usize,
    ) -> Result<(), GpuError> {
        let n = self.config.n_embd;
        let kvd = self.config.n_kv_head * self.config.head_dim;
        let mlp_hidden = self.config.mlp_hidden;
        let layer = &self.weights.layers[layer_idx];

        // 1. RMSNorm(hidden) → save to residual
        self.dispatch_rmsnorm(encoder, &self.activations.hidden, n)?;
        self.dispatch_copy(
            encoder,
            &self.activations.hidden,
            &self.activations.residual,
            n,
        )?;

        // 2. RMSNorm(hidden) again
        self.dispatch_rmsnorm(encoder, &self.activations.hidden, n)?;

        // 3. QKV projections with LoRA
        // (lora_inputs saved inside dispatch_lora_merge for each adapter)
        self.dispatch_lora_merge(
            encoder,
            &layer.attn_wq,
            layer_idx,
            LoraTarget::Q,
            &self.activations.hidden,
            &self.activations.q,
            n,
            n,
        )?;

        self.dispatch_lora_merge(
            encoder,
            &layer.attn_wk,
            layer_idx,
            LoraTarget::K,
            &self.activations.hidden,
            &self.activations.k,
            kvd,
            n,
        )?;

        self.dispatch_lora_merge(
            encoder,
            &layer.attn_wv,
            layer_idx,
            LoraTarget::V,
            &self.activations.hidden,
            &self.activations.v,
            kvd,
            n,
        )?;

        // 4. Store K, V in cache
        self.dispatch_copy_to_offset(
            encoder,
            &self.activations.k,
            &self.activations.key_cache[layer_idx],
            kvd,
            pos * kvd,
        )?;
        self.dispatch_copy_to_offset(
            encoder,
            &self.activations.v,
            &self.activations.value_cache[layer_idx],
            kvd,
            pos * kvd,
        )?;

        // 5. Multi-head attention
        self.dispatch_attention(encoder, pos)?;

        // 6. Output projection with LoRA + residual
        self.dispatch_lora_merge(
            encoder,
            &layer.attn_wo,
            layer_idx,
            LoraTarget::O,
            &self.activations.attn_out,
            &self.activations.hidden,
            n,
            n,
        )?;
        self.dispatch_add(
            encoder,
            &self.activations.hidden,
            &self.activations.residual,
            &self.activations.hidden,
            n,
        )?;

        // 7. MLP: save residual → RMSNorm → W1+ReLU → W2 → residual
        self.dispatch_copy(
            encoder,
            &self.activations.hidden,
            &self.activations.residual2,
            n,
        )?;
        self.dispatch_rmsnorm(encoder, &self.activations.hidden, n)?;

        self.dispatch_lora_merge(
            encoder,
            &layer.mlp_w1,
            layer_idx,
            LoraTarget::Mlp1,
            &self.activations.hidden,
            &self.activations.mlp_hidden,
            mlp_hidden,
            n,
        )?;
        self.dispatch_relu(encoder, &self.activations.mlp_hidden, mlp_hidden)?;

        self.dispatch_lora_merge(
            encoder,
            &layer.mlp_w2,
            layer_idx,
            LoraTarget::Mlp2,
            &self.activations.mlp_hidden,
            &self.activations.hidden,
            n,
            mlp_hidden,
        )?;
        self.dispatch_add(
            encoder,
            &self.activations.hidden,
            &self.activations.residual2,
            &self.activations.hidden,
            n,
        )?;

        Ok(())
    }

    /// LM head projection: logits[pos * vocab..] = lm_head @ hidden.
    fn dispatch_lm_head(&self, encoder: &mut CommandEncoder, pos: usize) -> Result<(), GpuError> {
        let n = self.config.n_embd;
        let v = self.config.vocab_size;

        // Use matmul: output[v] = lm_head[v, n] @ hidden[n]
        // For per-position, this is a matvec: M=v, N=n, P=1
        // We'll write directly to the correct offset in logits buffer
        // Need a temp buffer for the matmul output
        let temp_logits = create_buffer(&self.ctx.device, v, "temp_logits");

        self.write_uniform(
            &self.uniform_matmul,
            &MatmulParams {
                m: v as u32,
                n: n as u32,
                p: 1,
                _pad: 0,
            },
        );

        let layout = &self.pipelines.matmul.bind_group_layout;
        let bg = self.make_bg(
            layout,
            &[
                Self::entry(0, &self.weights.lm_head),
                Self::entry(1, &self.activations.hidden),
                Self::entry(2, &temp_logits),
                Self::entry(3, &self.uniform_matmul),
            ],
            "bg_lm_head",
        );

        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("lm_head"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines.matmul.pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(dispatch_1d(v, 16)[0], 1, 1);
        }

        // Copy temp_logits to correct position in logits buffer
        let offset_bytes = (pos * v * std::mem::size_of::<f32>()) as u64;
        encoder.copy_buffer_to_buffer(
            &temp_logits,
            0,
            &self.activations.logits,
            offset_bytes,
            (v * std::mem::size_of::<f32>()) as u64,
        );

        Ok(())
    }

    // ── Dispatch helpers ────────────────────────────────────────────

    /// Dispatch RMSNorm on a buffer (in-place).
    fn dispatch_rmsnorm(
        &self,
        encoder: &mut CommandEncoder,
        buffer: &Buffer,
        dim: usize,
    ) -> Result<(), GpuError> {
        self.write_uniform(
            &self.uniform_layernorm,
            &LayernormParams {
                batch_seq: 1,
                dim: dim as u32,
                _pad0: 0,
                _pad1: 0,
            },
        );

        let layout = &self.pipelines.rmsnorm.bind_group_layout;
        let bg = self.make_bg(
            layout,
            &[
                Self::entry(0, buffer),
                Self::entry(1, &self.uniform_layernorm),
            ],
            "bg_rmsnorm",
        );

        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("rmsnorm"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines.rmsnorm.pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(dispatch_1d(1, 64)[0], 1, 1);
        }

        Ok(())
    }

    /// Dispatch elementwise add: out = a + b.
    /// Uses temp buffer if `out` aliases `a` or `b` (WebGPU restriction).
    fn dispatch_add(
        &self,
        encoder: &mut CommandEncoder,
        a: &Buffer,
        b: &Buffer,
        out: &Buffer,
        count: usize,
    ) -> Result<(), GpuError> {
        self.write_uniform(
            &self.uniform_elem,
            &ElementwiseParams {
                count: count as u32,
                _pad0: 0,
                _pad1: 0,
                _pad2: 0,
            },
        );

        // Check if output aliases either input — WebGPU forbids same buffer
        // as both read and read-write in a single dispatch.
        let needs_temp = Self::same_buffer(a, out) || Self::same_buffer(b, out);
        let actual_out = if needs_temp {
            &self.activations.temp_out
        } else {
            out
        };

        let layout = &self.pipelines.add.bind_group_layout;
        let bg = self.make_bg(
            layout,
            &[
                Self::entry(0, a),
                Self::entry(1, b),
                Self::entry(2, actual_out),
                Self::entry(3, &self.uniform_elem),
            ],
            "bg_add",
        );

        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("add"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines.add.pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(dispatch_1d(count, 256)[0], 1, 1);
        }

        // Copy temp result to actual output if we used a temp buffer
        if needs_temp {
            let size_bytes = (count * std::mem::size_of::<f32>()) as u64;
            encoder.copy_buffer_to_buffer(&self.activations.temp_out, 0, out, 0, size_bytes);
        }

        Ok(())
    }

    /// Dispatch copy: dst = src (using elementwise copy shader).
    /// Note: copy entry point only uses bindings 0, 2, 3 (not 1).
    /// Uses temp buffer if src == dst (WebGPU forbids same buffer as both read and read-write).
    fn dispatch_copy(
        &self,
        encoder: &mut CommandEncoder,
        src: &Buffer,
        dst: &Buffer,
        count: usize,
    ) -> Result<(), GpuError> {
        self.write_uniform(
            &self.uniform_elem,
            &ElementwiseParams {
                count: count as u32,
                _pad0: 0,
                _pad1: 0,
                _pad2: 0,
            },
        );

        // Check if src aliases dst — WebGPU forbids same buffer as both read and read-write.
        let needs_temp = Self::same_buffer(src, dst);
        let actual_dst = if needs_temp {
            &self.activations.temp_out
        } else {
            dst
        };

        let layout = &self.pipelines.copy.bind_group_layout;
        let bg = self.make_bg(
            layout,
            &[
                Self::entry(0, src),
                Self::entry(2, actual_dst),
                Self::entry(3, &self.uniform_elem),
            ],
            "bg_copy",
        );

        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("copy"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines.copy.pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(dispatch_1d(count, 256)[0], 1, 1);
        }

        // Copy temp result to actual output if we used a temp buffer
        if needs_temp {
            let size_bytes = (count * std::mem::size_of::<f32>()) as u64;
            encoder.copy_buffer_to_buffer(&self.activations.temp_out, 0, dst, 0, size_bytes);
        }

        Ok(())
    }

    /// Dispatch ReLU: out = max(0, a).
    /// Note: relu entry point only uses bindings 0, 2, 3 (not 1).
    /// Uses temp buffer because WebGPU forbids same buffer as both read and read-write.
    fn dispatch_relu(
        &self,
        encoder: &mut CommandEncoder,
        buffer: &Buffer,
        count: usize,
    ) -> Result<(), GpuError> {
        self.write_uniform(
            &self.uniform_elem,
            &ElementwiseParams {
                count: count as u32,
                _pad0: 0,
                _pad1: 0,
                _pad2: 0,
            },
        );

        // ReLU always needs a temp buffer since it reads from `buffer` (binding 0)
        // and writes to `buffer` (binding 2) — same buffer, different access modes.
        let temp = &self.activations.temp_out;

        let layout = &self.pipelines.relu.bind_group_layout;
        let bg = self.make_bg(
            layout,
            &[
                Self::entry(0, buffer),
                Self::entry(2, temp),
                Self::entry(3, &self.uniform_elem),
            ],
            "bg_relu",
        );

        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("relu"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines.relu.pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(dispatch_1d(count, 256)[0], 1, 1);
        }

        // Copy temp result back to the actual buffer
        let size_bytes = (count * std::mem::size_of::<f32>()) as u64;
        encoder.copy_buffer_to_buffer(temp, 0, buffer, 0, size_bytes);

        Ok(())
    }

    /// Dispatch LoRA merge: output = W @ input + alpha * B @ (A @ input).
    /// Uses two dispatches: lora_a_forward then lora_b_forward.
    /// Uses temp buffer if input aliases output (WebGPU restriction).
    fn dispatch_lora_merge(
        &self,
        encoder: &mut CommandEncoder,
        base_weight: &Buffer,
        layer_idx: usize,
        target: LoraTarget,
        input: &Buffer,
        output: &Buffer,
        out_dim: usize,
        in_dim: usize,
    ) -> Result<(), GpuError> {
        let adapter_idx = GpuLoraBuffers::adapter_index(layer_idx, target);
        let adapter = &self.lora.adapters[adapter_idx];
        let intermediate = &self.activations.lora_intermediates[adapter_idx];
        let lora_input_buf = &self.activations.lora_inputs[adapter_idx];
        let rank = self.lora.rank;
        let alpha = self.lora.alpha;

        // Save input for LoRA backward pass (needed for gradient computation)
        self.dispatch_copy(encoder, input, lora_input_buf, in_dim)?;

        // Check if input aliases output — lora_b_forward binds input as read
        // and output as read-write; WebGPU forbids same buffer in both roles.
        let needs_temp = Self::same_buffer(input, output);
        let actual_output = if needs_temp {
            &self.activations.temp_out
        } else {
            output
        };

        // Dispatch 1: intermediate = A @ input [rank]
        self.write_uniform(
            &self.uniform_lora_a,
            &LoraParamsA {
                rank: rank as u32,
                n_embd: in_dim as u32,
                _pad0: 0,
                _pad1: 0,
            },
        );

        let layout_a = &self.pipelines.lora_a_forward.bind_group_layout;
        let bg_a = self.make_bg(
            layout_a,
            &[
                Self::entry(0, &adapter.a),
                Self::entry(1, input),
                Self::entry(2, intermediate),
                Self::entry(3, &self.uniform_lora_a),
            ],
            &format!("bg_lora_a_{layer_idx}_{:?}", target),
        );

        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("lora_a"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines.lora_a_forward.pipeline);
            pass.set_bind_group(0, &bg_a, &[]);
            pass.dispatch_workgroups(dispatch_1d(rank, 64)[0], 1, 1);
        }

        // Dispatch 2: output = W @ input + alpha * B @ intermediate [out_dim]
        self.write_uniform(
            &self.uniform_lora_b,
            &LoraParamsB {
                out_dim: out_dim as u32,
                n_embd: in_dim as u32,
                rank: rank as u32,
                alpha,
            },
        );

        let layout_b = &self.pipelines.lora_b_forward.bind_group_layout;
        let bg_b = self.make_bg(
            layout_b,
            &[
                Self::entry(0, base_weight),
                Self::entry(1, &adapter.b),
                Self::entry(2, input),
                Self::entry(3, intermediate),
                Self::entry(4, actual_output),
                Self::entry(5, &self.uniform_lora_b),
            ],
            &format!("bg_lora_b_{layer_idx}_{:?}", target),
        );

        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("lora_b"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines.lora_b_forward.pipeline);
            pass.set_bind_group(0, &bg_b, &[]);
            pass.dispatch_workgroups(dispatch_1d(out_dim, 64)[0], 1, 1);
        }

        // Copy temp result to actual output if we used a temp buffer
        if needs_temp {
            let size_bytes = (out_dim * std::mem::size_of::<f32>()) as u64;
            encoder.copy_buffer_to_buffer(&self.activations.temp_out, 0, output, 0, size_bytes);
        }

        Ok(())
    }

    /// Dispatch multi-head attention for one position.
    /// Uses the attention_score shader per head.
    fn dispatch_attention(&self, encoder: &mut CommandEncoder, pos: usize) -> Result<(), GpuError> {
        let hd = self.config.head_dim;
        let n_head = self.config.n_head;
        let n_kv = self.config.n_kv_head;
        let kvd = n_kv * hd;
        let scale = 1.0 / (hd as f32).sqrt();
        let _t_n = pos + 1; // number of positions to attend to

        // Initialize attn_out to zero
        let zero_data = vec![0.0f32; self.config.n_embd];
        self.ctx.queue.write_buffer(
            &self.activations.attn_out,
            0,
            bytemuck::cast_slice(&zero_data),
        );

        // For each head, compute attention scores and weighted sum
        for h in 0..n_head {
            let kv_group = h * n_kv / n_head;

            // Extract query for this head: q[h * hd .. (h+1) * hd]
            // Extract key cache for this KV group: key_cache[kv_group * hd ..]
            // The attention_score shader reads from full buffers but only accesses
            // the relevant slice. We need per-head temp buffers.
            let head_query = create_buffer(&self.ctx.device, hd, &format!("hq_{h}"));
            let head_output = create_buffer(&self.ctx.device, hd, &format!("ho_{h}"));

            // Copy query slice for this head
            let q_offset = (h * hd * std::mem::size_of::<f32>()) as u64;
            let _k_offset = (kv_group * hd * std::mem::size_of::<f32>()) as u64;
            let hd_bytes = (hd * std::mem::size_of::<f32>()) as u64;

            encoder.copy_buffer_to_buffer(&self.activations.q, q_offset, &head_query, 0, hd_bytes);

            self.write_uniform(
                &self.uniform_attn_score,
                &AttnScoreParams {
                    head_dim: hd as u32,
                    pos: pos as u32,
                    scale,
                    kv_offset: (kv_group * hd) as u32,
                    kv_stride: kvd as u32,
                    _pad0: 0,
                    _pad1: 0,
                    _pad2: 0,
                },
            );

            let layout = &self.pipelines.attention_score.bind_group_layout;
            let bg = self.make_bg(
                layout,
                &[
                    Self::entry(0, &head_query),
                    Self::entry(1, &self.activations.key_cache[0]), // simplified: single layer
                    Self::entry(2, &self.activations.value_cache[0]),
                    Self::entry(3, &head_output),
                    Self::entry(4, &self.uniform_attn_score),
                ],
                &format!("bg_attn_head_{h}"),
            );

            {
                let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                    label: Some(&format!("attn_head_{h}")),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipelines.attention_score.pipeline);
                pass.set_bind_group(0, &bg, &[]);
                pass.dispatch_workgroups(1, 1, 1);
            }

            // Copy head output to attn_out at correct offset
            let out_offset = (h * hd * std::mem::size_of::<f32>()) as u64;
            encoder.copy_buffer_to_buffer(
                &head_output,
                0,
                &self.activations.attn_out,
                out_offset,
                hd_bytes,
            );
        }

        Ok(())
    }

    /// Copy src to dst at byte offset in dst.
    fn dispatch_copy_to_offset(
        &self,
        encoder: &mut CommandEncoder,
        src: &Buffer,
        dst: &Buffer,
        count: usize,
        dst_offset_elements: usize,
    ) -> Result<(), GpuError> {
        let offset_bytes = (dst_offset_elements * std::mem::size_of::<f32>()) as u64;
        let size_bytes = (count * std::mem::size_of::<f32>()) as u64;
        encoder.copy_buffer_to_buffer(src, 0, dst, offset_bytes, size_bytes);
        Ok(())
    }

    /// Download logits as Vec<f32>.
    pub fn download_logits(&self, seq_len: usize) -> Result<Vec<f32>, GpuError> {
        download_f32(
            &self.ctx.device,
            &self.ctx.queue,
            &self.activations.logits,
            seq_len * self.config.vocab_size,
        )
    }
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::GpuContext;
    use crate::types::{Config, Rng};

    fn get_ctx() -> Option<Arc<GpuContext>> {
        GpuContext::new().ok().map(Arc::new)
    }

    #[test]
    fn test_gpu_weight_upload() {
        let Some(ctx) = get_ctx() else {
            println!("No GPU — skipping weight upload test");
            return;
        };
        let config = Config::micro();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);

        let gpu_weights = GpuWeightBuffers::from_weights(&ctx.device, &ctx.queue, &weights);

        // Verify sizes
        assert_eq!(
            gpu_weights.wte.size(),
            (config.vocab_size * config.n_embd * std::mem::size_of::<f32>()) as u64
        );
        assert_eq!(gpu_weights.layers.len(), config.n_layer);
    }

    #[test]
    fn test_gpu_forward_creates() {
        let Some(ctx) = get_ctx() else {
            println!("No GPU — skipping forward create test");
            return;
        };
        let config = Config::micro();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        let lora = GpuLoraBuffers::new(&ctx.device, &ctx.queue, &config, 4, 8.0, &mut rng);

        let seq_len = 4;
        let result = GpuForwardPass::new(ctx, config, &weights, lora, seq_len);
        assert!(
            result.is_ok(),
            "Forward pass creation failed: {:?}",
            result.err()
        );
    }
}
