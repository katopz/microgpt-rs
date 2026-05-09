// WGSL shader loading and compute pipeline creation helpers.
// All shaders are embedded at compile time via include_str!.

use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutEntry,
    BindingType, Buffer, BufferBindingType, BufferDescriptor, BufferUsages, ComputePipeline,
    ComputePipelineDescriptor, Device, ShaderModuleDescriptor, ShaderSource,
};

#[allow(unused_imports)]
use crate::gpu::context::GpuError;

// ── Embedded WGSL sources ──────────────────────────────────────────

pub const MATMUL: &str = include_str!("matmul.wgsl");
pub const ELEMENTWISE: &str = include_str!("elementwise.wgsl");
pub const SCALE: &str = include_str!("scale.wgsl");
pub const SOFTMAX: &str = include_str!("softmax.wgsl");
pub const LAYERNORM: &str = include_str!("layernorm.wgsl");
pub const EMBEDDING: &str = include_str!("embedding.wgsl");
pub const ATTENTION_QKV: &str = include_str!("attention_qkv.wgsl");
pub const ATTENTION_SCORE: &str = include_str!("attention_score.wgsl");
pub const LORA_A: &str = include_str!("lora_a.wgsl");
pub const LORA_B: &str = include_str!("lora_b.wgsl");
pub const LOSS_PER_SAMPLE: &str = include_str!("loss_per_sample.wgsl");
pub const LOSS_REDUCE: &str = include_str!("loss_reduce.wgsl");
pub const OPTIMIZER: &str = include_str!("optimizer.wgsl");

// ── Shader entry points ────────────────────────────────────────────

/// Entry point names for each compute shader dispatch.
pub mod entry {
    // matmul
    pub const MATMUL_TILED: &str = "matmul_tiled";
    // elementwise
    pub const ADD: &str = "add";
    pub const MULTIPLY: &str = "multiply";
    pub const RELU: &str = "relu";
    pub const COPY: &str = "copy";
    // scale (separate file to avoid binding conflicts)
    pub const SCALE: &str = "scale";
    // softmax
    pub const SOFTMAX: &str = "softmax";
    // layernorm
    pub const RMSNORM: &str = "rmsnorm";
    // embedding
    pub const EMBEDDING_LOOKUP: &str = "embedding_lookup";
    // attention
    pub const QKV_PROJECTION: &str = "qkv_projection";
    pub const ATTENTION_SCORE: &str = "attention_score";
    // lora
    pub const LORA_A_FORWARD: &str = "lora_a_forward";
    pub const LORA_B_FORWARD: &str = "lora_b_forward";
    // loss
    pub const CROSS_ENTROPY_PER_SAMPLE: &str = "cross_entropy_per_sample";
    pub const CROSS_ENTROPY_REDUCE: &str = "cross_entropy_reduce";
    // optimizer
    pub const ADAMW_STEP: &str = "adamw_step";
}

// ── Pipeline creation ──────────────────────────────────────────────

/// A compute pipeline bundled with its auto-derived bind group layout.
///
/// Created via `layout: None` so wgpu derives the bind group layout from
/// the WGSL shader bindings. The layout is then extracted and stored here
/// for callers to use when creating bind groups.
pub struct PipelineBundle {
    pub pipeline: ComputePipeline,
    pub bind_group_layout: BindGroupLayout,
}

/// Create a compute pipeline from WGSL source and entry point.
/// Uses `layout: None` to auto-derive bind group layouts from shader bindings.
pub fn create_pipeline(
    device: &Device,
    shader_source: &str,
    entry_point: &str,
    label: Option<&str>,
) -> PipelineBundle {
    let module = device.create_shader_module(ShaderModuleDescriptor {
        label,
        source: ShaderSource::Wgsl(shader_source.into()),
    });

    let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label,
        layout: None,
        module: &module,
        entry_point: Some(entry_point),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    let bind_group_layout = pipeline.get_bind_group_layout(0);

    PipelineBundle {
        pipeline,
        bind_group_layout,
    }
}

// ── Bind group helpers ─────────────────────────────────────────────

/// Standard visibility for compute shaders.
#[allow(dead_code)]
const COMPUTE_VIS: wgpu::ShaderStages = wgpu::ShaderStages::COMPUTE;

/// Create a storage buffer binding layout (read-only).
#[allow(dead_code)]
pub fn storage_read_binding(binding: u32) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: COMPUTE_VIS,
        ty: BindingType::Buffer {
            ty: BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

/// Create a storage buffer binding layout (read-write).
#[allow(dead_code)]
pub fn storage_rw_binding(binding: u32) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: COMPUTE_VIS,
        ty: BindingType::Buffer {
            ty: BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

/// Create a uniform buffer binding layout.
#[allow(dead_code)]
pub fn uniform_binding(binding: u32) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: COMPUTE_VIS,
        ty: BindingType::Buffer {
            ty: BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

/// Create a uniform buffer initialized with raw bytes.
#[allow(dead_code)]
pub fn create_uniform_buffer(device: &Device, data: &[u8], label: &str) -> Buffer {
    device.create_buffer(&BufferDescriptor {
        label: Some(label),
        size: data.len() as u64,
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

/// Write uniform data to an existing buffer via queue.
#[allow(dead_code)]
pub fn write_uniform<T: bytemuck::Pod>(queue: &wgpu::Queue, buffer: &Buffer, data: &T) {
    let bytes = bytemuck::cast_slice(std::slice::from_ref(data));
    queue.write_buffer(buffer, 0, bytes);
}

/// Create a simple bind group from a layout and buffer entries.
/// Each entry is (binding_index, buffer).
pub fn simple_bind_group(
    device: &Device,
    layout: &BindGroupLayout,
    entries: &[(u32, &Buffer)],
    label: Option<&str>,
) -> BindGroup {
    let bg_entries: Vec<BindGroupEntry> = entries
        .iter()
        .map(|(binding, buffer)| BindGroupEntry {
            binding: *binding,
            resource: buffer.as_entire_binding(),
        })
        .collect();

    device.create_bind_group(&BindGroupDescriptor {
        label,
        layout,
        entries: &bg_entries,
    })
}

// ── Shader-specific pipeline builders ──────────────────────────────

/// All compute pipelines needed for GPU training.
/// Each pipeline is bundled with its auto-derived bind group layout.
pub struct GpuPipelines {
    pub matmul: PipelineBundle,
    pub add: PipelineBundle,
    pub multiply: PipelineBundle,
    pub relu: PipelineBundle,
    pub copy: PipelineBundle,
    pub scale: PipelineBundle,
    pub softmax: PipelineBundle,
    pub rmsnorm: PipelineBundle,
    pub embedding_lookup: PipelineBundle,
    pub qkv_projection: PipelineBundle,
    pub attention_score: PipelineBundle,
    pub lora_a_forward: PipelineBundle,
    pub lora_b_forward: PipelineBundle,
    pub cross_entropy_per_sample: PipelineBundle,
    pub cross_entropy_reduce: PipelineBundle,
    pub adamw_step: PipelineBundle,
}

impl GpuPipelines {
    /// Create all compute pipelines with auto-derived bind group layouts.
    pub fn new(device: &Device) -> Self {
        let matmul = create_pipeline(device, MATMUL, entry::MATMUL_TILED, Some("matmul"));

        // Elementwise: all entry points are in the same shader module
        let add = create_pipeline(device, ELEMENTWISE, entry::ADD, Some("add"));
        let multiply = create_pipeline(device, ELEMENTWISE, entry::MULTIPLY, Some("multiply"));
        let relu = create_pipeline(device, ELEMENTWISE, entry::RELU, Some("relu"));
        let copy = create_pipeline(device, ELEMENTWISE, entry::COPY, Some("copy"));
        let scale = create_pipeline(device, SCALE, entry::SCALE, Some("scale"));

        let softmax = create_pipeline(device, SOFTMAX, entry::SOFTMAX, Some("softmax"));
        let rmsnorm = create_pipeline(device, LAYERNORM, entry::RMSNORM, Some("rmsnorm"));
        let embedding_lookup = create_pipeline(
            device,
            EMBEDDING,
            entry::EMBEDDING_LOOKUP,
            Some("embedding_lookup"),
        );

        let qkv_projection = create_pipeline(
            device,
            ATTENTION_QKV,
            entry::QKV_PROJECTION,
            Some("qkv_projection"),
        );
        let attention_score = create_pipeline(
            device,
            ATTENTION_SCORE,
            entry::ATTENTION_SCORE,
            Some("attention_score"),
        );

        let lora_a_forward = create_pipeline(
            device,
            LORA_A,
            entry::LORA_A_FORWARD,
            Some("lora_a_forward"),
        );
        let lora_b_forward = create_pipeline(
            device,
            LORA_B,
            entry::LORA_B_FORWARD,
            Some("lora_b_forward"),
        );

        let cross_entropy_per_sample = create_pipeline(
            device,
            LOSS_PER_SAMPLE,
            entry::CROSS_ENTROPY_PER_SAMPLE,
            Some("ce_per_sample"),
        );
        let cross_entropy_reduce = create_pipeline(
            device,
            LOSS_REDUCE,
            entry::CROSS_ENTROPY_REDUCE,
            Some("ce_reduce"),
        );

        let adamw_step = create_pipeline(device, OPTIMIZER, entry::ADAMW_STEP, Some("adamw"));

        Self {
            matmul,
            add,
            multiply,
            relu,
            copy,
            scale,
            softmax,
            rmsnorm,
            embedding_lookup,
            qkv_projection,
            attention_score,
            lora_a_forward,
            lora_b_forward,
            cross_entropy_per_sample,
            cross_entropy_reduce,
            adamw_step,
        }
    }
}

// ── Dispatch workgroup size helpers ────────────────────────────────

/// Calculate dispatch count: ceil(total / workgroup_size).
#[inline]
pub fn dispatch_count(total: usize, workgroup_size: usize) -> u32 {
    total.div_ceil(workgroup_size) as u32
}

/// Dispatch for 2D workgroups (e.g., matmul 16x16).
#[inline]
#[allow(dead_code)]
pub fn dispatch_2d(rows: usize, cols: usize, wg_x: usize, wg_y: usize) -> [u32; 3] {
    [dispatch_count(rows, wg_x), dispatch_count(cols, wg_y), 1]
}

/// Dispatch for 1D workgroups (e.g., 256).
#[inline]
pub fn dispatch_1d(total: usize, workgroup_size: usize) -> [u32; 3] {
    [dispatch_count(total, workgroup_size), 1, 1]
}
