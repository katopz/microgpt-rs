// Cross-entropy loss computation on GPU.
// Uses two dispatch passes:
//   1. Per-sample softmax + loss (one invocation per position)
//   2. Tree reduction to compute mean loss

use std::sync::Arc;

use wgpu::{Buffer, BufferDescriptor, BufferUsages, ComputePassDescriptor};

use crate::gpu::buffer::{create_buffer, download_f32};
use crate::gpu::context::GpuError;
use crate::gpu::kernels::{GpuPipelines, dispatch_1d, simple_bind_group};

// ── Uniform params ─────────────────────────────────────────────────

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct LossParams {
    batch_seq: u32,
    vocab_size: u32,
    total_tokens: u32,
    _pad: u32,
}

// ── GpuLoss ─────────────────────────────────────────────────────────

/// GPU cross-entropy loss computation.
/// Coordinates the two WGSL dispatch passes for per-sample loss + reduction.
pub struct GpuLoss {
    ctx: Arc<crate::gpu::GpuContext>,
    pipelines: Arc<GpuPipelines>,
    uniform_loss: Buffer,
}

impl GpuLoss {
    /// Create a new GPU loss computation context.
    pub fn new(
        ctx: Arc<crate::gpu::GpuContext>,
        pipelines: Arc<GpuPipelines>,
    ) -> Result<Self, GpuError> {
        let device = &ctx.device;

        let uniform_loss = device.create_buffer(&BufferDescriptor {
            label: Some("u_loss"),
            size: 64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Ok(Self {
            ctx,
            pipelines,
            uniform_loss,
        })
    }

    fn write_uniform<T: bytemuck::Pod>(&self, buffer: &Buffer, data: &T) {
        let bytes = bytemuck::cast_slice(std::slice::from_ref(data));
        self.ctx.queue.write_buffer(buffer, 0, bytes);
    }

    /// Compute cross-entropy loss on GPU.
    ///
    /// - `logits`: buffer of [batch_seq * vocab_size] f32 values
    /// - `targets`: slice of target token IDs
    /// - `vocab_size`: vocabulary dimension
    /// - `log_probs_buf`: output buffer for softmax probabilities [batch_seq * vocab_size]
    ///   (needed for backward pass)
    ///
    /// Returns the scalar mean loss value.
    pub fn compute_loss(
        &self,
        logits: &Buffer,
        targets: &[usize],
        vocab_size: usize,
        log_probs_buf: &Buffer,
    ) -> Result<f32, GpuError> {
        let batch_seq = targets.len();
        if batch_seq == 0 {
            return Ok(0.0);
        }

        let device = &self.ctx.device;
        let queue = &self.ctx.queue;

        // Upload targets as u32 buffer
        let targets_u32: Vec<u32> = targets.iter().map(|&t| t as u32).collect();
        let targets_bytes = bytemuck::cast_slice(&targets_u32);
        let targets_buf = device.create_buffer(&BufferDescriptor {
            label: Some("loss_targets"),
            size: targets_bytes.len() as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&targets_buf, 0, targets_bytes);

        // Per-sample loss output buffer
        let per_sample_loss = create_buffer(device, batch_seq, "per_sample_loss");

        // Loss output buffer (single f32)
        let loss_buf = create_buffer(device, 1, "loss_output");

        // Dispatch 1: per-sample softmax + loss
        self.write_uniform(
            &self.uniform_loss,
            &LossParams {
                batch_seq: batch_seq as u32,
                vocab_size: vocab_size as u32,
                total_tokens: batch_seq as u32,
                _pad: 0,
            },
        );

        let bg_per_sample = simple_bind_group(
            device,
            &self.pipelines.cross_entropy_per_sample.bind_group_layout,
            &[
                (0, logits),
                (1, &targets_buf),
                (2, &per_sample_loss),
                (3, log_probs_buf),
                (4, &self.uniform_loss),
            ],
            Some("bg_loss_per_sample"),
        );

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("loss_per_sample"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("ce_per_sample"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines.cross_entropy_per_sample.pipeline);
            pass.set_bind_group(0, &bg_per_sample, &[]);
            pass.dispatch_workgroups(dispatch_1d(batch_seq, 64)[0], 1, 1);
        }

        // Dispatch 2: tree reduction
        let bg_reduce = simple_bind_group(
            device,
            &self.pipelines.cross_entropy_reduce.bind_group_layout,
            &[
                (0, &per_sample_loss),
                (1, &loss_buf),
                (2, &self.uniform_loss),
            ],
            Some("bg_loss_reduce"),
        );

        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("ce_reduce"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines.cross_entropy_reduce.pipeline);
            pass.set_bind_group(0, &bg_reduce, &[]);
            // Workgroup size 256 for tree reduction
            pass.dispatch_workgroups(1, 1, 1);
        }

        queue.submit(std::iter::once(encoder.finish()));

        // Download the scalar loss
        let loss_data = download_f32(device, queue, &loss_buf, 1)?;
        Ok(loss_data[0])
    }

    /// CPU fallback: compute cross-entropy loss on CPU.
    /// Useful for verification and when GPU is unavailable.
    pub fn compute_loss_cpu(logits: &[f32], targets: &[usize], vocab_size: usize) -> f32 {
        let batch_seq = targets.len();
        if batch_seq == 0 {
            return 0.0;
        }

        let mut total_loss = 0.0f32;

        for i in 0..batch_seq {
            let offset = i * vocab_size;
            let target = targets[i];

            // Find max for numerical stability
            let mut max_logit = f32::NEG_INFINITY;
            for v in 0..vocab_size {
                if offset + v < logits.len() {
                    max_logit = max_logit.max(logits[offset + v]);
                }
            }

            // Compute log-softmax
            let mut sum_exp = 0.0f32;
            for v in 0..vocab_size {
                if offset + v < logits.len() {
                    sum_exp += (logits[offset + v] - max_logit).exp();
                }
            }

            let log_sum_exp = sum_exp.ln();

            // Loss = -log_softmax(target)
            if target < vocab_size && offset + target < logits.len() {
                let target_logit = logits[offset + target];
                let loss = -(target_logit - max_logit) + log_sum_exp;
                total_loss += loss;
            }
        }

        total_loss / batch_seq as f32
    }
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::GpuContext;
    use crate::gpu::buffer::upload_f32;
    use crate::types::Config;

    fn get_ctx() -> Option<Arc<GpuContext>> {
        GpuContext::new().ok().map(Arc::new)
    }

    #[test]
    fn test_cpu_loss_basic() {
        // Logits: [[10.0, 0.0, 0.0]] → softmax ≈ [1.0, 0.0, 0.0]
        // Target: 0 → loss ≈ 0
        let logits = vec![10.0, 0.0, 0.0];
        let loss = GpuLoss::compute_loss_cpu(&logits, &[0], 3);
        assert!(
            loss < 0.01,
            "Loss should be near zero for correct prediction: {loss}"
        );
    }

    #[test]
    fn test_cpu_loss_wrong_prediction() {
        // Logits: [[0.0, 0.0, 10.0]] → softmax ≈ [0.0, 0.0, 1.0]
        // Target: 0 → loss ≈ 10
        let logits = vec![0.0, 0.0, 10.0];
        let loss = GpuLoss::compute_loss_cpu(&logits, &[0], 3);
        assert!(
            loss > 5.0,
            "Loss should be large for wrong prediction: {loss}"
        );
    }

    #[test]
    fn test_cpu_loss_uniform() {
        // Uniform logits → loss = ln(vocab_size)
        let logits = vec![1.0, 1.0, 1.0, 1.0];
        let loss = GpuLoss::compute_loss_cpu(&logits, &[0], 4);
        let expected = 4.0f32.ln();
        assert!(
            (loss - expected).abs() < 0.01,
            "Uniform logits: expected {expected}, got {loss}"
        );
    }

    #[test]
    fn test_cpu_loss_batch() {
        // Two samples: one correct, one wrong
        let logits = vec![
            10.0, 0.0, 0.0, // correct: target=0
            0.0, 0.0, 10.0, // wrong: target=0
        ];
        let loss = GpuLoss::compute_loss_cpu(&logits, &[0, 0], 3);
        // Average of ~0 and ~10 → ~5
        assert!(
            loss > 2.0 && loss < 8.0,
            "Batch loss should be moderate: {loss}"
        );
    }

    #[test]
    fn test_gpu_loss_matches_cpu() {
        let Some(ctx) = get_ctx() else {
            println!("No GPU — skipping GPU loss test");
            return;
        };

        let pipelines = Arc::new(GpuPipelines::new(&ctx.device));
        let gpu_loss = GpuLoss::new(ctx.clone(), pipelines).expect("create loss");

        let vocab_size = 27;
        let logits: Vec<f32> = (0..vocab_size).map(|i| i as f32 * 0.1).collect();
        let targets = vec![5usize];

        // CPU loss
        let cpu_loss = GpuLoss::compute_loss_cpu(&logits, &targets, vocab_size);

        // GPU loss
        let logits_buf = upload_f32(&ctx.device, &ctx.queue, &logits, "test_logits");
        let log_probs_buf = create_buffer(&ctx.device, vocab_size, "test_log_probs");

        let gpu_loss_val = gpu_loss
            .compute_loss(&logits_buf, &targets, vocab_size, &log_probs_buf)
            .expect("gpu loss");

        let rel_error = (cpu_loss - gpu_loss_val).abs() / cpu_loss.abs().max(1e-10);
        assert!(
            rel_error < 0.05,
            "GPU loss ({gpu_loss_val:.4}) should match CPU ({cpu_loss:.4}), rel_error={rel_error:.4}"
        );
    }

    #[test]
    fn test_gpu_loss_batch_matches_cpu() {
        let Some(ctx) = get_ctx() else {
            println!("No GPU — skipping GPU batch loss test");
            return;
        };

        let pipelines = Arc::new(GpuPipelines::new(&ctx.device));
        let gpu_loss = GpuLoss::new(ctx.clone(), pipelines).expect("create loss");

        let vocab_size = 16;
        let batch_seq = 4;

        // Create random-ish logits
        let logits: Vec<f32> = (0..batch_seq * vocab_size)
            .map(|i| ((i as f32 * 0.37) % 2.0) - 1.0)
            .collect();
        let targets = vec![3, 7, 1, 12];

        let cpu_loss = GpuLoss::compute_loss_cpu(&logits, &targets, vocab_size);

        let logits_buf = upload_f32(&ctx.device, &ctx.queue, &logits, "batch_logits");
        let log_probs_buf = create_buffer(&ctx.device, batch_seq * vocab_size, "batch_log_probs");

        let gpu_loss_val = gpu_loss
            .compute_loss(&logits_buf, &targets, vocab_size, &log_probs_buf)
            .expect("gpu loss batch");

        let rel_error = (cpu_loss - gpu_loss_val).abs() / cpu_loss.abs().max(1e-10);
        assert!(
            rel_error < 0.05,
            "GPU batch loss ({gpu_loss_val:.4}) should match CPU ({cpu_loss:.4}), rel_error={rel_error:.4}"
        );
    }
}
