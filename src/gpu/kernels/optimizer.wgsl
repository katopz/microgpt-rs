// AdamW optimizer step for LoRA parameters.
// Updates params in-place using gradient, momentum (m), and velocity (v).

@group(0) @binding(0) var<storage, read_write> params: array<f32>;
@group(0) @binding(1) var<storage, read>       grads: array<f32>;
@group(0) @binding(2) var<storage, read_write> m: array<f32>;      // first moment
@group(0) @binding(3) var<storage, read_write> v: array<f32>;      // second moment
@group(0) @binding(4) var<uniform>             opts: AdamWParams;

struct AdamWParams {
    lr: f32,           // learning rate
    beta1: f32,        // 0.9
    beta2: f32,        // 0.999
    eps: f32,          // 1e-8
    weight_decay: f32, // 0.01
    step: u32,         // current training step (for bias correction)
    param_count: u32,
}

@compute @workgroup_size(256, 1, 1)
fn adamw_step(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= opts.param_count) { return; }

    let g = grads[i];
    let current_m = m[i];
    let current_v = v[i];

    // Update moments
    let new_m = opts.beta1 * current_m + (1.0 - opts.beta1) * g;
    let new_v = opts.beta2 * current_v + (1.0 - opts.beta2) * g * g;
    m[i] = new_m;
    v[i] = new_v;

    // Bias correction
    let step_f = f32(opts.step);
    let m_hat = new_m / (1.0 - pow(opts.beta1, step_f));
    let v_hat = new_v / (1.0 - pow(opts.beta2, step_f));

    // AdamW: weight decay applied directly to params (not through gradient)
    let decayed = params[i] * (1.0 - opts.lr * opts.weight_decay);

    // Parameter update
    params[i] = decayed - opts.lr * m_hat / (sqrt(v_hat) + opts.eps);
}
