// Compute attention scores and weighted sum for one head.
// Reads from stored KV cache.

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
