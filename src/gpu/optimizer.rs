// AdamW optimizer for LoRA parameters on GPU.
// Dispatches the adamw_step WGSL shader to update parameters in-place.
// Supports learning rate scheduling with linear warmup.

use std::sync::Arc;

use wgpu::{Buffer, BufferDescriptor, BufferUsages, CommandEncoder, ComputePassDescriptor};

use crate::gpu::context::GpuError;
use crate::gpu::kernels::{GpuPipelines, dispatch_1d, simple_bind_group};
use crate::gpu::lora::GpuLoraBuffers;

// ── Uniform params for AdamW ───────────────────────────────────────

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct AdamWParams {
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    step: u32,
    param_count: u32,
    _pad: u32,
}

// ── AdamW Optimizer ─────────────────────────────────────────────────

/// AdamW optimizer configuration.
#[derive(Clone, Debug)]
pub struct AdamWConfig {
    pub learning_rate: f32,
    pub weight_decay: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub warmup_steps: usize,
}

impl Default for AdamWConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1e-3,
            weight_decay: 0.01,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            warmup_steps: 100,
        }
    }
}

/// GPU-based AdamW optimizer for LoRA parameters.
/// Dispatches the `adamw_step` compute shader for each parameter group.
pub struct AdamWOptimizer {
    ctx: Arc<crate::gpu::GpuContext>,
    config: AdamWConfig,
    pipelines: Arc<GpuPipelines>,
    uniform_adamw: Buffer,
    step: u32,
}

impl AdamWOptimizer {
    /// Create a new AdamW optimizer.
    pub fn new(
        ctx: Arc<crate::gpu::GpuContext>,
        config: AdamWConfig,
        pipelines: Arc<GpuPipelines>,
    ) -> Self {
        let device = &ctx.device;

        let uniform_adamw = device.create_buffer(&BufferDescriptor {
            label: Some("u_adamw"),
            size: 64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            ctx,
            config,
            pipelines,
            uniform_adamw,
            step: 0,
        }
    }

    /// Get the current step count.
    pub fn current_step(&self) -> u32 {
        self.step
    }

    /// Compute the learning rate with linear warmup.
    /// During warmup: lr = base_lr * (step / warmup_steps)
    /// After warmup: lr = base_lr
    pub fn current_lr(&self) -> f32 {
        if self.config.warmup_steps == 0 {
            return self.config.learning_rate;
        }

        if self.step as usize <= self.config.warmup_steps {
            self.config.learning_rate * (self.step as f32 / self.config.warmup_steps as f32)
        } else {
            self.config.learning_rate
        }
    }

    /// Compute learning rate with cosine decay schedule.
    /// After warmup: lr = base_lr * 0.5 * (1 + cos(pi * progress))
    pub fn current_lr_cosine(&self, total_steps: usize) -> f32 {
        let base_lr = self.config.learning_rate;

        if self.config.warmup_steps > 0 && self.step as usize <= self.config.warmup_steps {
            return base_lr * (self.step as f32 / self.config.warmup_steps as f32);
        }

        if total_steps == 0 {
            return base_lr;
        }

        let progress = ((self.step as usize).saturating_sub(self.config.warmup_steps) as f32)
            / (total_steps.saturating_sub(self.config.warmup_steps) as f32);
        let cosine_decay = 0.5 * (1.0 + (std::f32::consts::PI * progress).cos());
        base_lr * cosine_decay
    }

    fn write_uniform<T: bytemuck::Pod>(&self, buffer: &Buffer, data: &T) {
        let bytes = bytemuck::cast_slice(std::slice::from_ref(data));
        self.ctx.queue.write_buffer(buffer, 0, bytes);
    }

    /// Perform one optimizer step: update all LoRA parameters.
    /// Dispatches the adamw_step shader for each A and B matrix.
    pub fn step(&mut self, lora: &GpuLoraBuffers) -> Result<(), GpuError> {
        self.step += 1;
        let lr = self.current_lr();

        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("adamw_step"),
            });

        // Update all adapters
        for (i, adapter) in lora.adapters.iter().enumerate() {
            // Update A matrix
            let a_count = adapter.rank * adapter.in_dim;
            self.dispatch_adamw(
                &mut encoder,
                &adapter.a,
                &adapter.grad_a,
                &adapter.m_a,
                &adapter.v_a,
                a_count,
                lr,
                &format!("adamw_a_{i}"),
            );

            // Update B matrix
            let b_count = adapter.out_dim * adapter.rank;
            self.dispatch_adamw(
                &mut encoder,
                &adapter.b,
                &adapter.grad_b,
                &adapter.m_b,
                &adapter.v_b,
                b_count,
                lr,
                &format!("adamw_b_{i}"),
            );
        }

        self.ctx.queue.submit(std::iter::once(encoder.finish()));

        Ok(())
    }

    /// Dispatch adamw_step for one parameter group.
    fn dispatch_adamw(
        &self,
        encoder: &mut CommandEncoder,
        params: &Buffer,
        grads: &Buffer,
        m: &Buffer,
        v: &Buffer,
        param_count: usize,
        lr: f32,
        label: &str,
    ) {
        self.write_uniform(
            &self.uniform_adamw,
            &AdamWParams {
                lr,
                beta1: self.config.beta1,
                beta2: self.config.beta2,
                eps: self.config.eps,
                weight_decay: self.config.weight_decay,
                step: self.step,
                param_count: param_count as u32,
                _pad: 0,
            },
        );

        let bg = simple_bind_group(
            &self.ctx.device,
            &self.pipelines.adamw_step.bind_group_layout,
            &[
                (0, params),
                (1, grads),
                (2, m),
                (3, v),
                (4, &self.uniform_adamw),
            ],
            Some(label),
        );

        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some(label),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines.adamw_step.pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(dispatch_1d(param_count, 256)[0], 1, 1);
        }
    }

    /// Reset optimizer state (for re-training).
    pub fn reset(&mut self) {
        self.step = 0;
    }

    /// CPU-based AdamW step for verification.
    /// Updates params in-place using the same formula as the GPU shader.
    pub fn step_cpu(
        params: &mut [f32],
        grads: &[f32],
        m: &mut [f32],
        v: &mut [f32],
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
        step: u32,
    ) {
        let step_f = step as f32;
        let bias_correction1 = 1.0 - beta1.powf(step_f);
        let bias_correction2 = 1.0 - beta2.powf(step_f);

        for i in 0..params.len() {
            let g = grads[i];

            // Update moments
            m[i] = beta1 * m[i] + (1.0 - beta1) * g;
            v[i] = beta2 * v[i] + (1.0 - beta2) * g * g;

            // Bias correction
            let m_hat = m[i] / bias_correction1;
            let v_hat = v[i] / bias_correction2;

            // AdamW: weight decay applied directly to params
            let decayed = params[i] * (1.0 - lr * weight_decay);

            // Parameter update
            params[i] = decayed - lr * m_hat / (v_hat.sqrt() + eps);
        }
    }
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::GpuContext;
    use crate::gpu::buffer::{download_f32, upload_f32};
    use crate::gpu::lora::GpuLoraBuffers;
    use crate::types::{Config, Rng};

    fn get_ctx() -> Option<Arc<GpuContext>> {
        GpuContext::new().ok().map(Arc::new)
    }

    #[test]
    fn test_adamw_config_default() {
        let config = AdamWConfig::default();
        assert_eq!(config.learning_rate, 1e-3);
        assert_eq!(config.weight_decay, 0.01);
        assert_eq!(config.beta1, 0.9);
        assert_eq!(config.beta2, 0.999);
        assert_eq!(config.eps, 1e-8);
        assert_eq!(config.warmup_steps, 100);
    }

    #[test]
    fn test_warmup_lr() {
        let config = AdamWConfig {
            learning_rate: 1e-3,
            warmup_steps: 100,
            ..Default::default()
        };
        let ctx = GpuContext::new().ok().map(Arc::new);
        let Some(ctx) = ctx else {
            println!("No GPU — skipping warmup lr test");
            return;
        };
        let pipelines = Arc::new(GpuPipelines::new(&ctx.device));
        let mut optimizer = AdamWOptimizer::new(ctx, config, pipelines);

        // At step 0, lr should be 0
        assert!((optimizer.current_lr()).abs() < 1e-10);

        // Step 1 manually
        optimizer.step = 50;
        let lr = optimizer.current_lr();
        assert!(
            (lr - 5e-4).abs() < 1e-6,
            "Warmup at 50%: expected ~5e-4, got {lr}"
        );

        // After warmup
        optimizer.step = 200;
        let lr = optimizer.current_lr();
        assert!(
            (lr - 1e-3).abs() < 1e-10,
            "After warmup: expected 1e-3, got {lr}"
        );
    }

    #[test]
    fn test_cosine_decay_lr() {
        let config = AdamWConfig {
            learning_rate: 1e-3,
            warmup_steps: 0,
            ..Default::default()
        };
        let ctx = GpuContext::new().ok().map(Arc::new);
        let Some(ctx) = ctx else {
            println!("No GPU — skipping cosine decay test");
            return;
        };
        let pipelines = Arc::new(GpuPipelines::new(&ctx.device));
        let mut optimizer = AdamWOptimizer::new(ctx, config, pipelines);

        // At step 0, cosine lr = base_lr * 0.5 * (1 + cos(0)) = base_lr * 1.0
        let lr = optimizer.current_lr_cosine(1000);
        assert!(
            (lr - 1e-3).abs() < 1e-10,
            "Cosine at start: expected 1e-3, got {lr}"
        );

        // At step 500 (50%), cosine lr ≈ base_lr * 0.5
        optimizer.step = 500;
        let lr = optimizer.current_lr_cosine(1000);
        assert!(
            (lr - 5e-4).abs() < 1e-5,
            "Cosine at 50%: expected ~5e-4, got {lr}"
        );

        // At step 1000 (100%), cosine lr ≈ 0
        optimizer.step = 1000;
        let lr = optimizer.current_lr_cosine(1000);
        assert!(lr < 1e-5, "Cosine at end: expected ~0, got {lr}");
    }

    #[test]
    fn test_optimizer_step_updates_params() {
        let Some(ctx) = get_ctx() else {
            println!("No GPU — skipping optimizer step test");
            return;
        };

        let config = Config::micro();
        let mut rng = Rng::new(42);
        let lora = GpuLoraBuffers::new(&ctx.device, &ctx.queue, &config, 2, 4.0, &mut rng);

        let adamw_config = AdamWConfig {
            learning_rate: 1e-2,
            warmup_steps: 0,
            ..Default::default()
        };
        let pipelines = Arc::new(GpuPipelines::new(&ctx.device));
        let mut optimizer = AdamWOptimizer::new(ctx.clone(), adamw_config, pipelines);

        // Set some non-zero gradients
        let adapter = &lora.adapters[0];
        let grad_a_data: Vec<f32> = (0..adapter.rank * adapter.in_dim)
            .map(|i| (i as f32 * 0.1 - 0.5))
            .collect();
        let grad_b_data: Vec<f32> = (0..adapter.out_dim * adapter.rank)
            .map(|i| (i as f32 * 0.1 - 0.3))
            .collect();

        ctx.queue
            .write_buffer(&adapter.grad_a, 0, bytemuck::cast_slice(&grad_a_data));
        ctx.queue
            .write_buffer(&adapter.grad_b, 0, bytemuck::cast_slice(&grad_b_data));

        // Download params before
        let a_before = download_f32(
            &ctx.device,
            &ctx.queue,
            &adapter.a,
            adapter.rank * adapter.in_dim,
        )
        .expect("download a before");

        // Run optimizer step
        optimizer.step(&lora).expect("optimizer step");

        // Download params after
        let a_after = download_f32(
            &ctx.device,
            &ctx.queue,
            &adapter.a,
            adapter.rank * adapter.in_dim,
        )
        .expect("download a after");

        // Params should have changed
        let mut changed = false;
        for (before, after) in a_before.iter().zip(a_after.iter()) {
            if (before - after).abs() > 1e-8 {
                changed = true;
                break;
            }
        }
        assert!(changed, "Parameters should change after optimizer step");
    }

    #[test]
    fn test_cpu_adamw_step() {
        let mut params = vec![1.0, 2.0, 3.0];
        let grads = vec![0.1, 0.2, 0.3];
        let mut m = vec![0.0; 3];
        let mut v = vec![0.0; 3];

        let lr = 1e-3;
        let beta1 = 0.9;
        let beta2 = 0.999;
        let eps = 1e-8;
        let weight_decay = 0.01;

        // Step 1
        AdamWOptimizer::step_cpu(
            &mut params,
            &grads,
            &mut m,
            &mut v,
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
            1,
        );

        // After step 1, params should have decreased (gradient descent)
        assert!(params[0] < 1.0, "Param should decrease after step");
        assert!(params[1] < 2.0, "Param should decrease after step");
        assert!(params[2] < 3.0, "Param should decrease after step");

        // Moments should be non-zero
        assert!(m[0].abs() > 0.0, "First moment should be non-zero");
        assert!(v[0].abs() > 0.0, "Second moment should be non-zero");
    }

    #[test]
    fn test_cpu_adamw_convergence() {
        // Simple test: minimize f(x) = x^2 starting from x=10
        let mut params = vec![10.0];
        let mut m = vec![0.0];
        let mut v = vec![0.0];

        let lr = 0.1;

        for step in 1..=5000 {
            // Gradient of x^2 is 2x
            let grads = vec![2.0 * params[0]];
            AdamWOptimizer::step_cpu(
                &mut params,
                &grads,
                &mut m,
                &mut v,
                lr,
                0.9,
                0.999,
                1e-8,
                0.0,
                step,
            );
        }

        // After 5000 steps, x should be close to 0
        assert!(
            params[0].abs() < 0.1,
            "AdamW should converge to ~0, got {}",
            params[0]
        );
    }
}
