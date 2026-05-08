// QKV projection for scaled dot-product attention.
// Computes Q, K, V projections for one position.
// One workgroup handles one attention head.

@group(0) @binding(0) var<storage, read>       wq: array<f32>;       // [n_head, head_dim, n_embd]
@group(0) @binding(1) var<storage, read>       wk: array<f32>;       // [n_kv_head, head_dim, n_embd]
@group(0) @binding(2) var<storage, read>       wv: array<f32>;       // [n_kv_head, head_dim, n_embd]
@group(0) @binding(3) var<storage, read>       x: array<f32>;        // [n_embd] input
@group(0) @binding(4) var<storage, read_write> q_out: array<f32>;    // [n_head * head_dim]
@group(0) @binding(5) var<storage, read_write> k_out: array<f32>;    // [n_kv_head * head_dim]
@group(0) @binding(6) var<storage, read_write> v_out: array<f32>;    // [n_kv_head * head_dim]
@group(0) @binding(7) var<uniform>             params: AttnParams;

struct AttnParams {
    n_embd: u32,
    n_head: u32,
    n_kv_head: u32,
    head_dim: u32,
    pos: u32,
    kv_dim: u32,
}

@compute @workgroup_size(64, 1, 1)
fn qkv_projection(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total_q = params.n_head * params.head_dim;
    let total_kv = params.n_kv_head * params.head_dim;

    if (idx < total_q) {
        // Q projection
        let head = idx / params.head_dim;
        let d = idx % params.head_dim;
        var sum: f32 = 0.0;
        for (var j = 0u; j < params.n_embd; j = j + 1u) {
            sum = sum + wq[head * params.head_dim * params.n_embd + d * params.n_embd + j] * x[j];
        }
        q_out[idx] = sum;
    }

    if (idx < total_kv) {
        // K projection
        let head = idx / params.head_dim;
        let d = idx % params.head_dim;
        var k_sum: f32 = 0.0;
        for (var j = 0u; j < params.n_embd; j = j + 1u) {
            k_sum = k_sum + wk[head * params.head_dim * params.n_embd + d * params.n_embd + j] * x[j];
        }
        k_out[idx] = k_sum;

        // V projection
        var v_sum: f32 = 0.0;
        for (var j = 0u; j < params.n_embd; j = j + 1u) {
            v_sum = v_sum + wv[head * params.head_dim * params.n_embd + d * params.n_embd + j] * x[j];
        }
        v_out[idx] = v_sum;
    }
}
