// LoRA merge: two-dispatch approach.
// Dispatch 1: compute intermediate = A * input [rank elements]
// Dispatch 2: compute output = W * input + alpha * B * intermediate

// Dispatch 1: Compute A * input -> intermediate[rank]
@group(0) @binding(0) var<storage, read>       lora_a: array<f32>;       // [rank, n_embd]
@group(0) @binding(1) var<storage, read>       input: array<f32>;        // [n_embd]
@group(0) @binding(2) var<storage, read_write> intermediate: array<f32>; // [rank]
@group(0) @binding(3) var<uniform>             params: LoraParamsA;

struct LoraParamsA {
    rank: u32,
    n_embd: u32,
}

@compute @workgroup_size(64, 1, 1)
fn lora_a_forward(@builtin(global_invocation_id) gid: vec3<u32>) {
    let r = gid.x;
    if (r >= params.rank) { return; }

    var sum: f32 = 0.0;
    for (var j = 0u; j < params.n_embd; j = j + 1u) {
        sum = sum + lora_a[r * params.n_embd + j] * input[j];
    }
    intermediate[r] = sum;
}

// Dispatch 2: Compute output = W * input + alpha * B * intermediate
@group(0) @binding(0) var<storage, read>       base_weight: array<f32>;  // [out_dim, n_embd]
@group(0) @binding(1) var<storage, read>       lora_b: array<f32>;       // [out_dim, rank]
@group(0) @binding(2) var<storage, read>       input: array<f32>;        // [n_embd]
@group(0) @binding(3) var<storage, read>       intermediate: array<f32>; // [rank] from dispatch 1
@group(0) @binding(4) var<storage, read_write> output: array<f32>;       // [out_dim]
@group(0) @binding(5) var<uniform>             params: LoraParamsB;

struct LoraParamsB {
    out_dim: u32,
    n_embd: u32,
    rank: u32,
    alpha: f32,
}

@compute @workgroup_size(64, 1, 1)
fn lora_b_forward(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.out_dim) { return; }

    // Base weight: W[i,:] * input
    var base_sum: f32 = 0.0;
    for (var j = 0u; j < params.n_embd; j = j + 1u) {
        base_sum = base_sum + base_weight[i * params.n_embd + j] * input[j];
    }

    // LoRA: B[i,:] * intermediate
    var lora_sum: f32 = 0.0;
    for (var r = 0u; r < params.rank; r = r + 1u) {
        lora_sum = lora_sum + lora_b[i * params.rank + r] * intermediate[r];
    }

    output[i] = base_sum + params.alpha * lora_sum;
}
