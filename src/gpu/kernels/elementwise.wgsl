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

// ScaleParams moved to scale.wgsl.

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

// Scale operation moved to scale.wgsl to avoid binding layout conflicts.
