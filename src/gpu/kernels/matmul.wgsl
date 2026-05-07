// Tiled matrix multiply: C[M,P] = A[M,N] * B[N,P]
// Each workgroup computes a tile of the output.
// Workgroup size: 16x16 = 256 invocations.

var<workgroup> tile_a: array<f32, 256>;  // 16x16 tile of A
var<workgroup> tile_b: array<f32, 256>;  // 16x16 tile of B

@group(0) @binding(0) var<storage, read>        a_data: array<f32>;
@group(0) @binding(1) var<storage, read>        b_data: array<f32>;
@group(0) @binding(2) var<storage, read_write>  c_data: array<f32>;
@group(0) @binding(3) var<uniform>              params: MatmulParams;

struct MatmulParams {
    m: u32,  // rows of A
    n: u32,  // cols of A / rows of B
    p: u32,  // cols of B
}

@compute @workgroup_size(16, 16, 1)
fn matmul_tiled(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let row = gid.x;
    let col = gid.y;
    if (row >= params.m || col >= params.p) { return; }

    let local_row = lid.x;
    let local_col = lid.y;

    var sum: f32 = 0.0;

    // Tile loop: process N in chunks of 16
    let num_tiles = (params.n + 15u) / 16u;
    for (var t = 0u; t < num_tiles; t = t + 1u) {
        // Load tile of A into shared memory
        let a_col = t * 16u + local_col;
        if (row < params.m && a_col < params.n) {
            tile_a[local_row * 16u + local_col] = a_data[row * params.n + a_col];
        } else {
            tile_a[local_row * 16u + local_col] = 0.0;
        }

        // Load tile of B into shared memory
        let b_row = t * 16u + local_row;
        if (b_row < params.n && col < params.p) {
            tile_b[local_row * 16u + local_col] = b_data[b_row * params.p + col];
        } else {
            tile_b[local_row * 16u + local_col] = 0.0;
        }

        workgroupBarrier();

        // Accumulate partial dot product
        for (var k = 0u; k < 16u; k = k + 1u) {
            sum = sum + tile_a[local_row * 16u + k] * tile_b[k * 16u + local_col];
        }

        workgroupBarrier();
    }

    c_data[row * params.p + col] = sum;
}
