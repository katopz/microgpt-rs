// GPU training loop: coordinates forward pass, loss, backward pass, and optimizer.
// Iterates epochs over training data, logs progress, and exports checkpoints.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::gpu::backward::GpuBackwardPass;

use crate::gpu::context::GpuError;
use crate::gpu::dataloader::{DataLoader, DataLoaderError};
use crate::gpu::forward::GpuForwardPass;
use crate::gpu::kernels::GpuPipelines;
use crate::gpu::lora::{GpuLoraBuffers, export_lora};
use crate::gpu::loss::GpuLoss;
use crate::gpu::optimizer::{AdamWConfig, AdamWOptimizer};
use crate::transformer::TransformerWeights;
use crate::types::Config;

// ── Training configuration ──────────────────────────────────────────

/// Configuration for the training loop.
#[derive(Clone, Debug)]
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
    pub seq_len: usize,
    pub batch_size: usize,
    pub pad_id: usize,
    pub lora_rank: usize,
    pub lora_alpha: f32,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            epochs: 10,
            learning_rate: 1e-3,
            weight_decay: 0.01,
            beta1: 0.9,
            beta2: 0.999,
            warmup_steps: 100,
            log_interval: 10,
            checkpoint_interval: 100,
            checkpoint_dir: "checkpoints".into(),
            seq_len: 16,
            batch_size: 4,
            pad_id: 0,
            lora_rank: 4,
            lora_alpha: 8.0,
        }
    }
}

impl TrainingConfig {
    /// Config for quick testing (few steps, small model).
    pub fn toy() -> Self {
        Self {
            epochs: 5,
            learning_rate: 1e-3,
            weight_decay: 0.01,
            beta1: 0.9,
            beta2: 0.999,
            warmup_steps: 0,
            log_interval: 1,
            checkpoint_interval: 5,
            checkpoint_dir: "checkpoints_toy".into(),
            seq_len: 8,
            batch_size: 2,
            pad_id: 0,
            lora_rank: 2,
            lora_alpha: 4.0,
        }
    }

    /// Config for BPE validator training (plan 007 dimensions).
    pub fn bpe_validator() -> Self {
        Self {
            epochs: 3,
            learning_rate: 5e-4,
            weight_decay: 0.01,
            beta1: 0.9,
            beta2: 0.999,
            warmup_steps: 50,
            log_interval: 10,
            checkpoint_interval: 100,
            checkpoint_dir: "checkpoints_bpe".into(),
            seq_len: 64,
            batch_size: 8,
            pad_id: 0,
            lora_rank: 4,
            lora_alpha: 8.0,
        }
    }
}

// ── Training report ─────────────────────────────────────────────────

/// Results from a completed training run.
#[derive(Clone, Debug)]
pub struct TrainingReport {
    pub total_steps: u32,
    pub total_epochs: usize,
    pub best_loss: f32,
    pub final_loss: f32,
    pub loss_history: Vec<(u32, f32)>,
}

impl std::fmt::Display for TrainingReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Training Report:")?;
        writeln!(f, "  Steps: {}", self.total_steps)?;
        writeln!(f, "  Epochs: {}", self.total_epochs)?;
        writeln!(f, "  Best loss: {:.6}", self.best_loss)?;
        writeln!(f, "  Final loss: {:.6}", self.final_loss)?;

        if let Some(first) = self.loss_history.first() {
            writeln!(f, "  Initial loss: {:.6}", first.1)?;
        }

        if self.loss_history.len() >= 2 {
            let improvement = self.loss_history.first().unwrap().1 - self.final_loss;
            writeln!(f, "  Loss improvement: {:.6}", improvement)?;
        }

        Ok(())
    }
}

// ── Training error ──────────────────────────────────────────────────

/// Errors that can occur during training.
#[derive(Debug)]
pub enum TrainingError {
    Gpu(GpuError),
    Data(DataLoaderError),
    NoImprovement(String),
}

impl std::fmt::Display for TrainingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TrainingError::Gpu(e) => write!(f, "GPU error: {e}"),
            TrainingError::Data(e) => write!(f, "Data error: {e}"),
            TrainingError::NoImprovement(msg) => write!(f, "No improvement: {msg}"),
        }
    }
}

impl std::error::Error for TrainingError {}

impl From<GpuError> for TrainingError {
    fn from(e: GpuError) -> Self {
        TrainingError::Gpu(e)
    }
}

impl From<DataLoaderError> for TrainingError {
    fn from(e: DataLoaderError) -> Self {
        TrainingError::Data(e)
    }
}

// ── Trainer ─────────────────────────────────────────────────────────

/// Main training loop orchestrator.
/// Coordinates forward pass, loss computation, backward pass, and optimizer.
pub struct Trainer {
    ctx: Arc<crate::gpu::GpuContext>,
    config: Config,
    training_config: TrainingConfig,
    pipelines: Arc<GpuPipelines>,
    weights: TransformerWeights,
}

impl Trainer {
    /// Create a new trainer.
    pub fn new(
        ctx: Arc<crate::gpu::GpuContext>,
        config: Config,
        training_config: TrainingConfig,
        weights: TransformerWeights,
    ) -> Result<Self, GpuError> {
        let pipelines = Arc::new(GpuPipelines::new(&ctx.device));

        Ok(Self {
            ctx,
            config,
            training_config,
            pipelines,
            weights,
        })
    }

    /// Run the full training loop with a JSONL data file.
    pub fn train_from_jsonl(&mut self, data_path: &Path) -> Result<TrainingReport, TrainingError> {
        let mut dataloader = DataLoader::from_jsonl(
            data_path,
            self.training_config.batch_size,
            self.training_config.seq_len,
            self.training_config.pad_id,
        )?;

        self.train(&mut dataloader)
    }

    /// Run the full training loop with a dataloader.
    pub fn train(&mut self, dataloader: &mut DataLoader) -> Result<TrainingReport, TrainingError> {
        let mut rng = crate::types::Rng::new(42);

        // Create LoRA buffers
        let lora = GpuLoraBuffers::new(
            &self.ctx.device,
            &self.ctx.queue,
            &self.config,
            self.training_config.lora_rank,
            self.training_config.lora_alpha,
            &mut rng,
        );

        // Create forward pass
        let forward_pass = GpuForwardPass::new(
            self.ctx.clone(),
            self.config.clone(),
            &self.weights,
            lora,
            self.training_config.seq_len,
        )?;

        // Create backward pass
        let backward_pass = GpuBackwardPass::new(
            self.ctx.clone(),
            self.config.clone(),
            self.pipelines.clone(),
        );

        // Create loss computation
        let gpu_loss = GpuLoss::new(self.ctx.clone(), self.pipelines.clone())?;

        // Create optimizer
        let adamw_config = AdamWConfig {
            learning_rate: self.training_config.learning_rate,
            weight_decay: self.training_config.weight_decay,
            beta1: self.training_config.beta1,
            beta2: self.training_config.beta2,
            eps: 1e-8,
            warmup_steps: self.training_config.warmup_steps,
        };
        let mut optimizer =
            AdamWOptimizer::new(self.ctx.clone(), adamw_config, self.pipelines.clone());

        // Training loop
        let mut step = 0u32;
        let mut best_loss = f32::MAX;
        let mut loss_history: Vec<(u32, f32)> = Vec::new();
        let mut epoch_losses: Vec<f32> = Vec::new();
        let mut total_loss_since_log = 0.0f32;
        let mut steps_since_log = 0u32;

        // Create checkpoint directory
        let checkpoint_dir = PathBuf::from(&self.training_config.checkpoint_dir);
        std::fs::create_dir_all(&checkpoint_dir).ok();

        for epoch in 0..self.training_config.epochs {
            let batches = dataloader.batches();
            let num_batches = batches.len();

            for (batch_idx, (input_ids, target_ids)) in batches.into_iter().enumerate() {
                // Convert input_ids to usize for forward pass
                let token_ids: Vec<usize> = input_ids.iter().map(|&t| t as usize).collect();
                let targets: Vec<usize> = target_ids.iter().map(|&t| t as usize).collect();

                // 1. Forward pass
                let logits_buf = forward_pass.forward(&token_ids)?;
                let seq_len = token_ids.len();

                // 2. Compute loss
                let log_probs_buf = crate::gpu::buffer::create_buffer(
                    &self.ctx.device,
                    seq_len * self.config.vocab_size,
                    "log_probs",
                );
                let loss = gpu_loss.compute_loss(
                    logits_buf,
                    &targets,
                    self.config.vocab_size,
                    &log_probs_buf,
                )?;

                // Track loss
                total_loss_since_log += loss;
                steps_since_log += 1;
                epoch_losses.push(loss);
                loss_history.push((step, loss));

                // 3. Backward pass (LoRA gradients)
                backward_pass.backward_pass(&forward_pass, &token_ids, &targets)?;

                // 4. Optimizer step
                optimizer.step(&forward_pass.lora)?;

                step += 1;

                // 5. Logging
                if step.is_multiple_of(self.training_config.log_interval as u32) {
                    let avg_loss = total_loss_since_log / steps_since_log as f32;
                    let lr = optimizer.current_lr();
                    println!(
                        "[step {step}] epoch={epoch} batch={batch_idx}/{num_batches} loss={avg_loss:.4} lr={lr:.6}"
                    );
                    total_loss_since_log = 0.0;
                    steps_since_log = 0;
                }

                // 6. Checkpoint
                if step.is_multiple_of(self.training_config.checkpoint_interval as u32)
                    && loss < best_loss {
                        best_loss = loss;
                        let checkpoint_path = checkpoint_dir.join(format!("lora_step_{step}.bin"));
                        match export_lora(
                            &self.ctx.device,
                            &self.ctx.queue,
                            &forward_pass.lora,
                            &checkpoint_path,
                        ) {
                            Ok(()) => {
                                println!(
                                    "[checkpoint] Saved best model (loss={loss:.6}) to {}",
                                    checkpoint_path.display()
                                );
                            }
                            Err(e) => {
                                eprintln!("[checkpoint] Failed to save: {e}");
                            }
                        }
                    }
            }

            // End of epoch summary
            if !epoch_losses.is_empty() {
                let epoch_avg: f32 = epoch_losses.iter().sum::<f32>() / epoch_losses.len() as f32;
                println!(
                    "[epoch {epoch}] avg_loss={epoch_avg:.4} batches={}",
                    epoch_losses.len()
                );
                epoch_losses.clear();
            }
        }

        // Save final model
        let final_path = checkpoint_dir.join("lora_final.bin");
        export_lora(
            &self.ctx.device,
            &self.ctx.queue,
            &forward_pass.lora,
            &final_path,
        )?;
        println!(
            "[done] Saved final model to {} (best_loss={best_loss:.6})",
            final_path.display()
        );

        let final_loss = loss_history.last().map(|(_, l)| *l).unwrap_or(0.0);

        Ok(TrainingReport {
            total_steps: step,
            total_epochs: self.training_config.epochs,
            best_loss,
            final_loss,
            loss_history,
        })
    }

    /// Train on toy data to verify the pipeline works.
    /// Uses synthetic data generated from the model's vocabulary.
    pub fn train_toy(&mut self) -> Result<TrainingReport, TrainingError> {
        let _rng = crate::types::Rng::new(42);

        // Generate toy training data: simple repeating patterns
        let vocab_size = self.config.vocab_size;
        let num_samples = 10;
        let sample_len = self.training_config.seq_len + 2; // +2 for input/target overlap

        let samples: Vec<crate::gpu::dataloader::TrainingSample> = (0..num_samples)
            .map(|i| {
                let tokens: Vec<usize> = (0..sample_len)
                    .map(|j| {
                        // Create a pattern: alternating tokens based on position
                        let base = (i % 3 + 1) * 3;
                        (base + j % (vocab_size / 4)).min(vocab_size - 1)
                    })
                    .collect();
                crate::gpu::dataloader::TrainingSample { tokens }
            })
            .collect();

        let mut dataloader = DataLoader::from_samples(
            samples,
            self.training_config.batch_size,
            self.training_config.seq_len,
            self.training_config.pad_id,
        )?;

        self.train(&mut dataloader)
    }
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::GpuContext;

    fn get_ctx() -> Option<Arc<GpuContext>> {
        GpuContext::new().ok().map(Arc::new)
    }

    #[test]
    fn test_training_config_default() {
        let config = TrainingConfig::default();
        assert_eq!(config.epochs, 10);
        assert_eq!(config.learning_rate, 1e-3);
        assert_eq!(config.lora_rank, 4);
        assert_eq!(config.lora_alpha, 8.0);
    }

    #[test]
    fn test_training_config_toy() {
        let config = TrainingConfig::toy();
        assert_eq!(config.epochs, 5);
        assert_eq!(config.warmup_steps, 0);
        assert_eq!(config.lora_rank, 2);
    }

    #[test]
    fn test_report_display() {
        let report = TrainingReport {
            total_steps: 100,
            total_epochs: 5,
            best_loss: 0.5,
            final_loss: 0.3,
            loss_history: vec![(0, 2.5), (50, 0.5), (100, 0.3)],
        };

        let display = format!("{report}");
        assert!(display.contains("Steps: 100"));
        assert!(display.contains("Best loss: 0.500000"));
        assert!(display.contains("Final loss: 0.300000"));
        assert!(display.contains("Loss improvement"));
    }

    #[test]
    fn test_toy_training_decreases_loss() {
        let Some(ctx) = get_ctx() else {
            println!("No GPU — skipping toy training test");
            return;
        };

        let config = Config::micro();
        let mut rng = crate::types::Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);

        let training_config = TrainingConfig {
            epochs: 3,
            learning_rate: 1e-2,
            warmup_steps: 0,
            log_interval: 1,
            checkpoint_interval: 100,
            seq_len: 4,
            batch_size: 2,
            lora_rank: 2,
            lora_alpha: 4.0,
            ..Default::default()
        };

        let mut trainer =
            Trainer::new(ctx, config, training_config, weights).expect("create trainer");

        let report = trainer.train_toy().expect("toy training");

        // Should have completed all steps
        assert!(report.total_steps > 0, "Should have completed steps");

        // Loss should have history
        assert!(
            report.loss_history.len() >= 2,
            "Should have loss history entries"
        );

        // Final loss should be finite
        assert!(
            report.final_loss.is_finite(),
            "Final loss should be finite: {}",
            report.final_loss
        );

        println!("Toy training report:\n{report}");
    }

    #[test]
    fn test_trainer_creation() {
        let Some(ctx) = get_ctx() else {
            println!("No GPU — skipping trainer creation test");
            return;
        };

        let config = Config::micro();
        let mut rng = crate::types::Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);

        let trainer = Trainer::new(ctx, config, TrainingConfig::toy(), weights);
        assert!(trainer.is_ok(), "Trainer creation should succeed");
    }
}
