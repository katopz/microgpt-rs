// Elementwise operations on 1D arrays.
// Each operation is a separate compute entry point.

@group(0) @binding(0) var<storage, read>       a: array<f32>;
@group(0) @binding(1) var<storage, read>       b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform>             params: ElementwiseParams;

struct ElementwiseParams {
    count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

// Also need a params struct for single-input ops
struct ScaleParams {
    count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    scale: f32,
}

@compute @workgroup_size(256, 1, 1)
fn add(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.count) { return; }
    out[i] = a[i] + b[i];
}

@compute @workgroup_size(256, 1, 1)
fn multiply(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.count) { return; }
    out[i] = a[i] * b[i];
}

@compute @workgroup_size(256, 1, 1)
fn relu(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.count) { return; }
    out[i] = max(0.0, a[i]);
}

@compute @workgroup_size(256, 1, 1)
fn copy(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.count) { return; }
    out[i] = a[i];
}

// Scale: out = a * scale (uses different bindings for single input)
@group(0) @binding(0) var<storage, read>       scale_a: array<f32>;
@group(0) @binding(1) var<storage, read_write> scale_out: array<f32>;
@group(0) @binding(2) var<uniform>             scale_params: ScaleParams;

@compute @workgroup_size(256, 1, 1)
fn scale(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= scale_params.count) { return; }
    scale_out[i] = scale_a[i] * scale_params.scale;
}
