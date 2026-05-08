use std::sync::Arc;

use wgpu::{
    AdapterInfo, Backends, Device, DeviceDescriptor, Features, Limits, PowerPreference, Queue,
    RequestAdapterOptions,
};

/// GPU context holding device and queue.
pub struct GpuContext {
    pub device: Arc<Device>,
    pub queue: Arc<Queue>,
    pub adapter_info: AdapterInfo,
    pub limits: Limits,
}

impl GpuContext {
    /// Create a new GPU context, selecting the best available adapter.
    /// Uses `pollster::block_on` for synchronous initialization.
    pub fn new() -> Result<Self, GpuError> {
        pollster::block_on(Self::new_async())
    }

    /// Async GPU context initialization.
    async fn new_async() -> Result<Self, GpuError> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: Backends::all(),
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&RequestAdapterOptions {
                power_preference: PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or(GpuError::NoAdapter)?;

        let adapter_info = adapter.get_info();

        // Use the adapter's actual limits (not downlevel defaults) to support
        // shaders with many storage bindings (e.g., qkv_projection has 7+).
        let adapter_limits = adapter.limits();

        let (device, queue) = adapter
            .request_device(
                &DeviceDescriptor {
                    label: Some("microgpt-rs gpu device"),
                    required_features: Features::empty(),
                    required_limits: adapter_limits,
                    ..Default::default()
                },
                None,
            )
            .await
            .map_err(|e| GpuError::DeviceError(e.to_string()))?;

        let limits = device.limits();

        Ok(Self {
            device: Arc::new(device),
            queue: Arc::new(queue),
            adapter_info,
            limits,
        })
    }
}

#[derive(Debug)]
pub enum GpuError {
    NoAdapter,
    DeviceError(String),
    ShaderError(String),
    BufferError(String),
}

impl std::fmt::Display for GpuError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GpuError::NoAdapter => write!(f, "No suitable GPU adapter found"),
            GpuError::DeviceError(msg) => write!(f, "Device error: {msg}"),
            GpuError::ShaderError(msg) => write!(f, "Shader error: {msg}"),
            GpuError::BufferError(msg) => write!(f, "Buffer error: {msg}"),
        }
    }
}

impl std::error::Error for GpuError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_context_init() {
        match GpuContext::new() {
            Ok(ctx) => {
                println!("GPU adapter: {}", ctx.adapter_info.name);
                assert!(!ctx.adapter_info.name.is_empty());
            }
            Err(GpuError::NoAdapter) => {
                println!("No GPU adapter available — skipping GPU tests");
            }
            Err(e) => panic!("Unexpected GPU error: {e}"),
        }
    }
}
