use std::sync::{Arc, Mutex};

use bytemuck;
use wgpu::util::DeviceExt;
use wgpu::{Buffer, BufferDescriptor, BufferUsages};

use super::context::GpuError;

/// Upload f32 data to a GPU buffer.
pub fn upload_f32(
    device: &wgpu::Device,
    _queue: &wgpu::Queue,
    data: &[f32],
    label: &str,
) -> Buffer {
    let bytes = bytemuck::cast_slice(data);
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytes,
        usage: BufferUsages::COPY_DST | BufferUsages::STORAGE | BufferUsages::COPY_SRC,
    })
}

/// Download f32 data from a GPU buffer (blocking).
pub fn download_f32(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    buffer: &Buffer,
    count: usize,
) -> Result<Vec<f32>, GpuError> {
    let bytes_needed = count * std::mem::size_of::<f32>();

    let staging_buffer = device.create_buffer(&BufferDescriptor {
        label: Some("staging download buffer"),
        size: bytes_needed as u64,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("download encoder"),
    });

    encoder.copy_buffer_to_buffer(buffer, 0, &staging_buffer, 0, bytes_needed as u64);
    queue.submit(std::iter::once(encoder.finish()));

    let result: Arc<Mutex<Option<Result<(), wgpu::BufferAsyncError>>>> = Arc::new(Mutex::new(None));
    let result_clone = result.clone();

    let buffer_slice = staging_buffer.slice(..);
    buffer_slice.map_async(wgpu::MapMode::Read, move |res| {
        *result_clone.lock().unwrap() = Some(res);
    });

    device.poll(wgpu::Maintain::Wait);

    let map_result = result
        .lock()
        .unwrap()
        .take()
        .ok_or_else(|| GpuError::BufferError("Map callback not called".into()))?;
    map_result.map_err(|e| GpuError::BufferError(e.to_string()))?;

    let data = buffer_slice.get_mapped_range();
    let output: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    staging_buffer.unmap();

    Ok(output)
}

/// Create an empty GPU buffer with the specified size (in f32 elements).
pub fn create_buffer(device: &wgpu::Device, count: usize, label: &str) -> Buffer {
    let size = (count * std::mem::size_of::<f32>()) as u64;
    device.create_buffer(&BufferDescriptor {
        label: Some(label),
        size,
        usage: BufferUsages::COPY_DST | BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::GpuContext;

    #[test]
    fn test_buffer_upload_download_roundtrip() {
        let ctx = match GpuContext::new() {
            Ok(ctx) => ctx,
            Err(_) => {
                println!("No GPU — skipping buffer test");
                return;
            }
        };

        let original: Vec<f32> = (0..16).map(|i| i as f32 * 0.5).collect();
        let buffer = upload_f32(&ctx.device, &ctx.queue, &original, "test buffer");

        let downloaded =
            download_f32(&ctx.device, &ctx.queue, &buffer, 16).expect("download should succeed");

        assert_eq!(original.len(), downloaded.len());
        for (a, b) in original.iter().zip(downloaded.iter()) {
            assert!((a - b).abs() < 1e-6, "Mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn test_create_empty_buffer() {
        let ctx = match GpuContext::new() {
            Ok(ctx) => ctx,
            Err(_) => {
                println!("No GPU — skipping empty buffer test");
                return;
            }
        };

        let buffer = create_buffer(&ctx.device, 1024, "empty test buffer");
        assert_eq!(buffer.size(), 1024 * std::mem::size_of::<f32>() as u64);
    }
}
