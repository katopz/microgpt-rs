// RMSNorm: x = x * rsqrt(mean(x^2) + eps)
// One invocation per vector (processes all elements of one vector).

@group(0) @binding(0) var<storage, read_write> data: array<f32>;
@group(0) @binding(1) var<uniform>             params: LayernormParams;

struct LayernormParams {
    batch_seq: u32,  // number of vectors
    dim: u32,        // elements per vector
}

@compute @workgroup_size(64, 1, 1)
fn rmsnorm(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.batch_seq) { return; }

    let offset = idx * params.dim;

    // Pass 1: sum of squares
    var sum_sq = 0.0;
    for (var d = 0u; d < params.dim; d = d + 1u) {
        let v = data[offset + d];
        sum_sq = sum_sq + v * v;
    }

    // Pass 2: scale by inverse RMS
    let inv_rms = 1.0 / sqrt(sum_sq / f32(params.dim) + 1e-5);
    for (var d = 0u; d < params.dim; d = d + 1u) {
        data[offset + d] = data[offset + d] * inv_rms;
    }
}
