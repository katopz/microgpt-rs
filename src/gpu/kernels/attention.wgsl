// Scaled dot-product attention for one head of one position.
// This is a simplified version for the micro model where we process
// one query vector at a time (inference mode, not training batch).
//
// Computes: Q = Wq @ x, K = Wk @ x, V = Wv @ x
//           attn = softmax(Q @ K^T / sqrt(d_k))
//           out = attn @ V
//           final = Wo @ out (in lora.wgsl or separately)

// Dispatch 1: Compute Q, K, V projections for one position
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

// Dispatch 2: Compute attention scores and weighted sum for one head
// Reads from stored KV cache (not handled here — uses external buffers).

@group(0) @binding(0) var<storage, read>       query: array<f32>;       // [head_dim] for this head
@group(0) @binding(1) var<storage, read>       keys: array<f32>;        // [pos+1, head_dim] all K up to current pos
@group(0) @binding(2) var<storage, read>       values: array<f32>;      // [pos+1, head_dim] all V up to current pos
@group(0) @binding(3) var<storage, read_write> attn_out: array<f32>;    // [head_dim] output for this head
@group(0) @binding(4) var<uniform>             params: AttnScoreParams;

struct AttnScoreParams {
    head_dim: u32,
    pos: u32,        // current position (0-indexed)
    scale: f32,      // 1.0 / sqrt(head_dim)
}

@compute @workgroup_size(1, 1, 1)
fn attention_score(@builtin(global_invocation_id) _gid: vec3<u32>) {
    // Compute attention scores: score[t] = Q . K[t] * scale
    let n_positions = params.pos + 1u;

    // Find max score for numerical stability
    var max_score: f32 = -1e30;
    for (var t = 0u; t < n_positions; t = t + 1u) {
        var dot: f32 = 0.0;
        for (var d = 0u; d < params.head_dim; d = d + 1u) {
            dot = dot + query[d] * keys[t * params.head_dim + d];
        }
        let score = dot * params.scale;
        if (score > max_score) { max_score = score; }
    }

    // Softmax
    var sum_exp: f32 = 0.0;
    var scores: array<f32, 256>;  // max block_size = 256
    for (var t = 0u; t < n_positions; t = t + 1u) {
        var dot: f32 = 0.0;
        for (var d = 0u; d < params.head_dim; d = d + 1u) {
            dot = dot + query[d] * keys[t * params.head_dim + d];
        }
        let s = exp(dot * params.scale - max_score);
        scores[t] = s;
        sum_exp = sum_exp + s;
    }

    // Weighted sum of values
    for (var d = 0u; d < params.head_dim; d = d + 1u) {
        attn_out[d] = 0.0;
    }
    for (var t = 0u; t < n_positions; t = t + 1u) {
        let weight = scores[t] / sum_exp;
        for (var d = 0u; d < params.head_dim; d = d + 1u) {
            attn_out[d] = attn_out[d] + weight * values[t * params.head_dim + d];
        }
    }
}
