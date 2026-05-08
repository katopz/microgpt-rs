// LoRA adapter buffers, initialization, export/import.
// 6 adapters per layer: Q, K, V, O, MLP1, MLP2.

use std::path::Path;

use wgpu::{Buffer, Device, Queue};

use crate::gpu::buffer::{create_buffer, download_f32, upload_f32};
use crate::gpu::context::GpuError;
use crate::types::{Config, Rng};

// ── Adapter targets ────────────────────────────────────────────────

/// Which weight matrices get LoRA adapters.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LoraTarget {
    Q = 0,
    K = 1,
    V = 2,
    O = 3,
    Mlp1 = 4,
    Mlp2 = 5,
}

impl LoraTarget {
    pub const COUNT: usize = 6;

    pub fn all() -> &'static [LoraTarget] {
        &[
            LoraTarget::Q,
            LoraTarget::K,
            LoraTarget::V,
            LoraTarget::O,
            LoraTarget::Mlp1,
            LoraTarget::Mlp2,
        ]
    }

    /// Get (in_dim, out_dim) for this adapter target given a config.
    pub fn dims(&self, config: &Config) -> (usize, usize) {
        let kv_dim = config.n_kv_head * config.head_dim;
        match self {
            LoraTarget::Q => (config.n_embd, config.n_embd),
            LoraTarget::K => (config.n_embd, kv_dim),
            LoraTarget::V => (config.n_embd, kv_dim),
            LoraTarget::O => (config.n_embd, config.n_embd),
            LoraTarget::Mlp1 => (config.n_embd, config.mlp_hidden),
            LoraTarget::Mlp2 => (config.mlp_hidden, config.n_embd),
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            LoraTarget::Q => "q",
            LoraTarget::K => "k",
            LoraTarget::V => "v",
            LoraTarget::O => "o",
            LoraTarget::Mlp1 => "mlp1",
            LoraTarget::Mlp2 => "mlp2",
        }
    }
}

// ── Single adapter ─────────────────────────────────────────────────

/// Single LoRA adapter: Y = Wx + alpha * BAx.
/// A: [rank, in_dim] Kaiming init, B: [out_dim, rank] zero init.
pub struct GpuLoraAdapter {
    pub a: Buffer, // [rank, in_dim]
    pub b: Buffer, // [out_dim, rank]
    pub grad_a: Buffer,
    pub grad_b: Buffer,
    pub m_a: Buffer, // AdamW first moment for A
    pub v_a: Buffer, // AdamW second moment for A
    pub m_b: Buffer, // AdamW first moment for B
    pub v_b: Buffer, // AdamW second moment for B
    pub in_dim: usize,
    pub out_dim: usize,
    pub rank: usize,
}

impl GpuLoraAdapter {
    /// Create a new LoRA adapter with Kaiming init for A, zeros for B.
    pub fn new(
        device: &Device,
        queue: &Queue,
        rank: usize,
        in_dim: usize,
        out_dim: usize,
        rng: &mut Rng,
        label: &str,
    ) -> Self {
        let std_a = (2.0 / in_dim as f32).sqrt();
        let a_data: Vec<f32> = (0..rank * in_dim).map(|_| rng.normal() * std_a).collect();
        let b_data = vec![0.0f32; out_dim * rank];

        let a = upload_f32(device, queue, &a_data, &format!("{label}_a"));
        let b = upload_f32(device, queue, &b_data, &format!("{label}_b"));
        let grad_a = create_buffer(device, rank * in_dim, &format!("{label}_grad_a"));
        let grad_b = create_buffer(device, out_dim * rank, &format!("{label}_grad_b"));
        let m_a = create_buffer(device, rank * in_dim, &format!("{label}_m_a"));
        let v_a = create_buffer(device, rank * in_dim, &format!("{label}_v_a"));
        let m_b = create_buffer(device, out_dim * rank, &format!("{label}_m_b"));
        let v_b = create_buffer(device, out_dim * rank, &format!("{label}_v_b"));

        Self {
            a,
            b,
            grad_a,
            grad_b,
            m_a,
            v_a,
            m_b,
            v_b,
            in_dim,
            out_dim,
            rank,
        }
    }

    /// Total trainable parameter count (A + B).
    pub fn param_count(&self) -> usize {
        self.rank * self.in_dim + self.out_dim * self.rank
    }
}

// ── All adapters ───────────────────────────────────────────────────

/// All LoRA adapters for the model. 6 adapters per layer.
pub struct GpuLoraBuffers {
    pub adapters: Vec<GpuLoraAdapter>,
    pub rank: usize,
    pub alpha: f32,
}

impl GpuLoraBuffers {
    /// Create LoRA adapters for all layers.
    pub fn new(
        device: &Device,
        queue: &Queue,
        config: &Config,
        rank: usize,
        alpha: f32,
        rng: &mut Rng,
    ) -> Self {
        let mut adapters = Vec::with_capacity(config.n_layer * LoraTarget::COUNT);

        for layer_idx in 0..config.n_layer {
            for target in LoraTarget::all() {
                let (in_dim, out_dim) = target.dims(config);
                let label = format!("lora_l{layer_idx}_{}", target.name());
                let adapter =
                    GpuLoraAdapter::new(device, queue, rank, in_dim, out_dim, rng, &label);
                adapters.push(adapter);
            }
        }

        Self {
            adapters,
            rank,
            alpha,
        }
    }

    /// Get adapter index for layer and target.
    #[inline]
    pub fn adapter_index(layer_idx: usize, target: LoraTarget) -> usize {
        layer_idx * LoraTarget::COUNT + target as usize
    }

    /// Get adapter for layer and target.
    #[inline]
    pub fn get_adapter(&self, layer_idx: usize, target: LoraTarget) -> &GpuLoraAdapter {
        &self.adapters[Self::adapter_index(layer_idx, target)]
    }

    /// Get mutable adapter for layer and target.
    #[inline]
    pub fn get_adapter_mut(&mut self, layer_idx: usize, target: LoraTarget) -> &mut GpuLoraAdapter {
        &mut self.adapters[Self::adapter_index(layer_idx, target)]
    }

    /// Total parameter count across all adapters.
    pub fn total_param_count(&self) -> usize {
        self.adapters.iter().map(|a| a.param_count()).sum()
    }
}

// ── Export / Import (custom binary with blake3) ────────────────────

/// Magic bytes for lora.bin format.
const LORA_MAGIC: &[u8; 4] = b"LORA";
const LORA_VERSION: u32 = 1;

/// Export LoRA adapters to a binary file with blake3 checksum.
///
/// Format:
/// ```text
/// [magic(4) | version(4) | checksum(32) | payload...]
/// payload: [n_adapters(4) | rank(4) | alpha(4) | adapter_data...]
/// adapter_data: [in_dim(4) | out_dim(4) | a_f32s | b_f32s]
/// ```
pub fn export_lora(
    device: &Device,
    queue: &Queue,
    lora: &GpuLoraBuffers,
    path: &Path,
) -> Result<(), GpuError> {
    let mut payload = Vec::new();

    // Header
    payload.extend_from_slice(&(lora.adapters.len() as u32).to_le_bytes());
    payload.extend_from_slice(&(lora.rank as u32).to_le_bytes());
    payload.extend_from_slice(&lora.alpha.to_le_bytes());

    // Adapter data
    for (i, adapter) in lora.adapters.iter().enumerate() {
        payload.extend_from_slice(&(adapter.in_dim as u32).to_le_bytes());
        payload.extend_from_slice(&(adapter.out_dim as u32).to_le_bytes());

        let a_count = adapter.rank * adapter.in_dim;
        let b_count = adapter.out_dim * adapter.rank;

        // Download A from GPU
        let a_data = download_f32(device, queue, &adapter.a, a_count)
            .map_err(|e| GpuError::BufferError(format!("Failed to download adapter {i} A: {e}")))?;
        // Download B from GPU
        let b_data = download_f32(device, queue, &adapter.b, b_count)
            .map_err(|e| GpuError::BufferError(format!("Failed to download adapter {i} B: {e}")))?;

        // Write A and B as f32 LE
        for val in &a_data {
            payload.extend_from_slice(&val.to_le_bytes());
        }
        for val in &b_data {
            payload.extend_from_slice(&val.to_le_bytes());
        }
    }

    // Compute blake3 checksum of payload
    let checksum = blake3::hash(&payload);

    // Assemble file: magic + version + checksum + payload
    let mut file_data = Vec::with_capacity(4 + 4 + 32 + payload.len());
    file_data.extend_from_slice(LORA_MAGIC);
    file_data.extend_from_slice(&LORA_VERSION.to_le_bytes());
    file_data.extend_from_slice(checksum.as_bytes());
    file_data.extend_from_slice(&payload);

    std::fs::write(path, &file_data)
        .map_err(|e| GpuError::BufferError(format!("Failed to write lora file: {e}")))?;

    Ok(())
}

/// Load LoRA adapters from a binary file and upload to GPU.
pub fn load_lora(
    device: &Device,
    queue: &Queue,
    path: &Path,
    alpha: f32,
) -> Result<GpuLoraBuffers, GpuError> {
    let file_data = std::fs::read(path)
        .map_err(|e| GpuError::BufferError(format!("Failed to read lora file: {e}")))?;

    // Validate header
    if file_data.len() < 44 {
        return Err(GpuError::BufferError(
            "File too small for lora header".into(),
        ));
    }

    let magic = &file_data[0..4];
    if magic != LORA_MAGIC {
        return Err(GpuError::BufferError("Invalid lora magic bytes".into()));
    }

    let version = u32::from_le_bytes(file_data[4..8].try_into().map_err(
        |e: std::array::TryFromSliceError| {
            GpuError::BufferError(format!("Version parse error: {e}"))
        },
    )?);
    if version != LORA_VERSION {
        return Err(GpuError::BufferError(format!(
            "Unsupported lora version: {version}"
        )));
    }

    let stored_checksum = &file_data[8..40];
    let payload = &file_data[40..];

    // Verify blake3 checksum
    let computed = blake3::hash(payload);
    if computed.as_bytes() != stored_checksum {
        return Err(GpuError::BufferError("LoRA file checksum mismatch".into()));
    }

    // Parse payload
    let mut offset = 0usize;

    let n_adapters = read_u32_le(payload, &mut offset)? as usize;
    let rank = read_u32_le(payload, &mut offset)? as usize;
    let file_alpha = read_f32_le(payload, &mut offset)?;
    let effective_alpha = if alpha != 0.0 { alpha } else { file_alpha };

    let mut adapters = Vec::with_capacity(n_adapters);

    for i in 0..n_adapters {
        let in_dim = read_u32_le(payload, &mut offset)? as usize;
        let out_dim = read_u32_le(payload, &mut offset)? as usize;

        let a_count = rank * in_dim;
        let b_count = out_dim * rank;

        let a_bytes = a_count * std::mem::size_of::<f32>();
        let b_bytes = b_count * std::mem::size_of::<f32>();

        if offset + a_bytes + b_bytes > payload.len() {
            return Err(GpuError::BufferError(format!(
                "Truncated data for adapter {i}"
            )));
        }

        // Read A
        let a_data: Vec<f32> = payload[offset..offset + a_bytes]
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("chunk is 4 bytes")))
            .collect();
        offset += a_bytes;

        // Read B
        let b_data: Vec<f32> = payload[offset..offset + b_bytes]
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("chunk is 4 bytes")))
            .collect();
        offset += b_bytes;

        // Upload to GPU
        let label = format!("lora_{i}");
        let a = upload_f32(device, queue, &a_data, &format!("{label}_a"));
        let b = upload_f32(device, queue, &b_data, &format!("{label}_b"));

        // Zero-initialize gradient and optimizer state
        let grad_a = create_buffer(device, a_count, &format!("{label}_grad_a"));
        let grad_b = create_buffer(device, b_count, &format!("{label}_grad_b"));
        let m_a = create_buffer(device, a_count, &format!("{label}_m_a"));
        let v_a = create_buffer(device, a_count, &format!("{label}_v_a"));
        let m_b = create_buffer(device, b_count, &format!("{label}_m_b"));
        let v_b = create_buffer(device, b_count, &format!("{label}_v_b"));

        adapters.push(GpuLoraAdapter {
            a,
            b,
            grad_a,
            grad_b,
            m_a,
            v_a,
            m_b,
            v_b,
            in_dim,
            out_dim,
            rank,
        });
    }

    Ok(GpuLoraBuffers {
        adapters,
        rank,
        alpha: effective_alpha,
    })
}

// ── Helpers ────────────────────────────────────────────────────────

fn read_u32_le(data: &[u8], offset: &mut usize) -> Result<u32, GpuError> {
    if *offset + 4 > data.len() {
        return Err(GpuError::BufferError("Unexpected end of data".into()));
    }
    let val = u32::from_le_bytes(
        data[*offset..*offset + 4]
            .try_into()
            .expect("slice is 4 bytes"),
    );
    *offset += 4;
    Ok(val)
}

fn read_f32_le(data: &[u8], offset: &mut usize) -> Result<f32, GpuError> {
    if *offset + 4 > data.len() {
        return Err(GpuError::BufferError("Unexpected end of data".into()));
    }
    let val = f32::from_le_bytes(
        data[*offset..*offset + 4]
            .try_into()
            .expect("slice is 4 bytes"),
    );
    *offset += 4;
    Ok(val)
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::GpuContext;
    use std::path::PathBuf;

    fn get_ctx() -> Option<GpuContext> {
        GpuContext::new().ok()
    }

    /// Create a temporary file path that auto-cleans on drop.
    struct TempFile {
        path: PathBuf,
    }

    impl TempFile {
        fn new(prefix: &str) -> Self {
            let dir = std::env::temp_dir();
            let id = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos();
            let path = dir.join(format!("{prefix}_{id}.bin"));
            Self { path }
        }

        fn path(&self) -> &Path {
            &self.path
        }
    }

    impl Drop for TempFile {
        fn drop(&mut self) {
            std::fs::remove_file(&self.path).ok();
        }
    }

    #[test]
    fn test_lora_target_dims() {
        let config = Config::micro();
        let (q_in, q_out) = LoraTarget::Q.dims(&config);
        assert_eq!(q_in, config.n_embd);
        assert_eq!(q_out, config.n_embd);

        let (k_in, k_out) = LoraTarget::K.dims(&config);
        assert_eq!(k_in, config.n_embd);
        let kv_dim = config.n_kv_head * config.head_dim;
        assert_eq!(k_out, kv_dim);

        let (mlp1_in, mlp1_out) = LoraTarget::Mlp1.dims(&config);
        assert_eq!(mlp1_in, config.n_embd);
        assert_eq!(mlp1_out, config.mlp_hidden);
    }

    #[test]
    fn test_adapter_param_count() {
        let Some(ctx) = get_ctx() else {
            println!("No GPU — skipping adapter param count test");
            return;
        };
        let mut rng = Rng::new(42);
        let rank = 4;
        let in_dim = 16;
        let out_dim = 16;

        let adapter = GpuLoraAdapter::new(
            &ctx.device,
            &ctx.queue,
            rank,
            in_dim,
            out_dim,
            &mut rng,
            "test",
        );

        // A: rank * in_dim = 4 * 16 = 64
        // B: out_dim * rank = 16 * 4 = 64
        assert_eq!(adapter.param_count(), 128);
    }

    #[test]
    fn test_lora_buffers_creation() {
        let Some(ctx) = get_ctx() else {
            println!("No GPU — skipping buffers creation test");
            return;
        };
        let config = Config::micro();
        let mut rng = Rng::new(42);
        let rank = 4;
        let alpha = 8.0;

        let lora = GpuLoraBuffers::new(&ctx.device, &ctx.queue, &config, rank, alpha, &mut rng);

        // 1 layer * 6 targets = 6 adapters
        assert_eq!(lora.adapters.len(), config.n_layer * LoraTarget::COUNT);
        assert_eq!(lora.rank, rank);
        assert_eq!(lora.alpha, alpha);
        assert!(lora.total_param_count() > 0);
    }

    #[test]
    fn test_export_import_roundtrip() {
        let Some(ctx) = get_ctx() else {
            println!("No GPU — skipping export/import test");
            return;
        };
        let config = Config::micro();
        let mut rng = Rng::new(42);
        let rank = 4;
        let alpha = 8.0;

        let lora = GpuLoraBuffers::new(&ctx.device, &ctx.queue, &config, rank, alpha, &mut rng);

        // Download original A/B for comparison
        let orig_a: Vec<Vec<f32>> = lora
            .adapters
            .iter()
            .map(|a| download_f32(&ctx.device, &ctx.queue, &a.a, a.rank * a.in_dim).unwrap())
            .collect();
        let orig_b: Vec<Vec<f32>> = lora
            .adapters
            .iter()
            .map(|a| download_f32(&ctx.device, &ctx.queue, &a.b, a.out_dim * a.rank).unwrap())
            .collect();

        // Export
        let tmp = TempFile::new("test_lora_export");
        let path = tmp.path().to_path_buf();
        export_lora(&ctx.device, &ctx.queue, &lora, &path).expect("export");

        // Import
        let loaded = load_lora(&ctx.device, &ctx.queue, &path, alpha).expect("load");

        assert_eq!(loaded.adapters.len(), lora.adapters.len());
        assert_eq!(loaded.rank, lora.rank);

        // Verify A/B data matches
        for (i, adapter) in loaded.adapters.iter().enumerate() {
            let a_data = download_f32(
                &ctx.device,
                &ctx.queue,
                &adapter.a,
                adapter.rank * adapter.in_dim,
            )
            .unwrap();
            let b_data = download_f32(
                &ctx.device,
                &ctx.queue,
                &adapter.b,
                adapter.out_dim * adapter.rank,
            )
            .unwrap();

            for (j, (orig, loaded)) in orig_a[i].iter().zip(a_data.iter()).enumerate() {
                assert!(
                    (orig - loaded).abs() < 1e-6,
                    "Adapter {i} A[{j}] mismatch: {orig} vs {loaded}"
                );
            }
            for (j, (orig, loaded)) in orig_b[i].iter().zip(b_data.iter()).enumerate() {
                assert!(
                    (orig - loaded).abs() < 1e-6,
                    "Adapter {i} B[{j}] mismatch: {orig} vs {loaded}"
                );
            }
        }
    }

    #[test]
    fn test_adapter_index() {
        assert_eq!(GpuLoraBuffers::adapter_index(0, LoraTarget::Q), 0);
        assert_eq!(GpuLoraBuffers::adapter_index(0, LoraTarget::Mlp2), 5);
        assert_eq!(GpuLoraBuffers::adapter_index(1, LoraTarget::Q), 6);
        assert_eq!(GpuLoraBuffers::adapter_index(2, LoraTarget::K), 13);
    }
}
