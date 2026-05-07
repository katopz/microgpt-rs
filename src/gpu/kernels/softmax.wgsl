// Stable softmax: two-pass (max + exp/sum/normalize).
// One invocation handles one row of a 2D array.

@group(0) @binding(0) var<storage, read_write> data: array<f32>;
@group(0) @binding(1) var<uniform>             params: SoftmaxParams;

struct SoftmaxParams {
    rows: u32,
    cols: u32,
}

@compute @workgroup_size(64, 1, 1)
fn softmax(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if (row >= params.rows) { return; }

    let offset = row * params.cols;

    // Pass 1: find max
    var max_val = data[offset];
    for (var c = 1u; c < params.cols; c = c + 1u) {
        let val = data[offset + c];
        if (val > max_val) { max_val = val; }
    }

    // Pass 2: exp, sum, normalize
    var sum = 0.0;
    for (var c = 0u; c < params.cols; c = c + 1u) {
        let exp_val = exp(data[offset + c] - max_val);
        data[offset + c] = exp_val;
        sum = sum + exp_val;
    }

    let inv_sum = 1.0 / sum;
    for (var c = 0u; c < params.cols; c = c + 1u) {
        data[offset + c] = data[offset + c] * inv_sum;
    }
}
