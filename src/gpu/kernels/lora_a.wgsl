// LoRA A matrix forward: compute intermediate = A * input [rank elements]
// Extracted from lora.wgsl to avoid binding layout conflicts with lora_b_forward.

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
