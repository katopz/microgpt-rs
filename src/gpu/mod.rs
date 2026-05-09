#[cfg(feature = "gpu")]
mod backward;
#[cfg(feature = "gpu")]
mod buffer;
#[cfg(feature = "gpu")]
mod context;
#[cfg(feature = "gpu")]
mod dataloader;
#[cfg(feature = "gpu")]
mod forward;
#[cfg(feature = "gpu")]
mod kernels;
#[cfg(feature = "gpu")]
mod lora;
#[cfg(feature = "gpu")]
mod loss;
#[cfg(feature = "gpu")]
mod optimizer;
#[cfg(feature = "gpu")]
mod training_loop;

#[cfg(feature = "gpu")]
pub use backward::GpuBackwardPass;
#[cfg(feature = "gpu")]
pub use buffer::{create_buffer, download_f32, upload_f32};
#[cfg(feature = "gpu")]
pub use context::{GpuContext, GpuError};
#[cfg(feature = "gpu")]
pub use dataloader::{DataLoader, DataLoaderError, TrainingSample};
#[cfg(feature = "gpu")]
pub use forward::{GpuActivationBuffers, GpuForwardPass, GpuWeightBuffers};
#[cfg(feature = "gpu")]
pub use kernels::GpuPipelines;
#[cfg(feature = "gpu")]
pub use lora::{
    GpuLoraBuffers, export_lora, load_lora, load_lora_from_safetensors, load_lora_from_wasm_binary,
};
#[cfg(feature = "gpu")]
pub use loss::GpuLoss;
#[cfg(feature = "gpu")]
pub use optimizer::{AdamWConfig, AdamWOptimizer};
#[cfg(feature = "gpu")]
pub use training_loop::{Trainer, TrainingConfig, TrainingReport};
