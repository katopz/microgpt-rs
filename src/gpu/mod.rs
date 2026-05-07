#[cfg(feature = "gpu")]
mod buffer;
#[cfg(feature = "gpu")]
mod context;

#[cfg(feature = "gpu")]
pub use buffer::{create_buffer, download_f32, upload_f32};
#[cfg(feature = "gpu")]
pub use context::{GpuContext, GpuError};
