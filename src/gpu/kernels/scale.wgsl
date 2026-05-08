// Scale: out = a * scale
// Single-input elementwise operation with a scalar parameter.
// Separated from elementwise.wgsl to avoid binding layout conflicts.

@group(0) @binding(0) var<storage, read>       a: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform>             params: ScaleParams;

struct ScaleParams {
    count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    scale: f32,
}

@compute @workgroup_size(256, 1, 1)
fn scale(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.count) { return; }
    out[i] = a[i] * params.scale;
}
