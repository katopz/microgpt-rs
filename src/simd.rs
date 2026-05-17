//! SIMD-accelerated linear algebra kernels for inference.
//!
//! Provides NEON (aarch64) and AVX2 (x86_64) backends for hot-path operations:
//! - Dot products (matmul inner loop)
//! - Outer product accumulation (HLA state updates)
//! - Matrix-vector multiply (HLA readout)
//!
//! Runtime detection selects the best available backend.
//! Falls back to scalar when SIMD is unavailable.
//!
//! # Stability
//!
//! Uses `core::arch` intrinsics directly — stable on both `aarch64` and `x86_64`.
//! No nightly features, no external SIMD crates.

/// SIMD capability level detected at runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimdLevel {
    /// No SIMD — scalar fallback.
    Scalar,
    /// ARM NEON (4× f32 per operation).
    Neon,
    /// x86 AVX2+FMA (8× f32 per operation).
    Avx2,
}

/// Detect the best available SIMD level for the current CPU.
///
/// On `aarch64`: always returns [`SimdLevel::Neon`] (mandatory on ARMv8+).
/// On `x86_64`: returns [`SimdLevel::Avx2`] if CPU supports AVX2+FMA, else [`SimdLevel::Scalar`].
/// On other architectures: returns [`SimdLevel::Scalar`].
#[inline]
pub fn simd_level() -> SimdLevel {
    #[cfg(target_arch = "aarch64")]
    {
        SimdLevel::Neon
    }
    #[cfg(target_arch = "x86_64")]
    {
        if is_avx2_fma_available() {
            SimdLevel::Avx2
        } else {
            SimdLevel::Scalar
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        SimdLevel::Scalar
    }
}

// ── x86_64 Runtime Detection ─────────────────────────────────

#[cfg(target_arch = "x86_64")]
fn is_avx2_fma_available() -> bool {
    #[cfg(target_feature = "avx2")]
    {
        true
    }
    #[cfg(not(target_feature = "avx2"))]
    {
        // Runtime detection via cpuid
        let cpuid1 = unsafe { core::arch::x86_64::__cpuid(1) };
        let has_avx = (cpuid1.ecx & (1 << 28)) != 0;
        let has_fma = (cpuid1.ecx & (1 << 12)) != 0;

        let cpuid7 = unsafe { core::arch::x86_64::__cpuid(7) };
        let has_avx2 = (cpuid7.ebx & (1 << 5)) != 0;

        has_avx && has_fma && has_avx2
    }
}

// ── Dot Product ───────────────────────────────────────────────

/// SIMD-accelerated dot product: `Σ a[i] * b[i]` for `len` elements.
///
/// Dispatches to NEON, AVX2, or scalar based on compile-time target and
/// runtime CPU feature detection.
#[inline]
pub fn simd_dot_f32(a: &[f32], b: &[f32], len: usize) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { neon_dot_f32(a, b, len) }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if is_avx2_fma_available() {
            unsafe { avx2_dot_f32(a, b, len) }
        } else {
            scalar_dot_f32(a, b, len)
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        scalar_dot_f32(a, b, len)
    }
}

/// Single-row FMA: `Σ weight_row[i] * input[i]` — alias for [`simd_dot_f32`].
/// Named for clarity in matmul context.
#[inline(always)]
pub fn simd_fma_row(weight_row: &[f32], input: &[f32], len: usize) -> f32 {
    simd_dot_f32(weight_row, input, len)
}

#[inline]
#[allow(dead_code)]
fn scalar_dot_f32(a: &[f32], b: &[f32], len: usize) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..len {
        unsafe {
            sum += *a.get_unchecked(i) * *b.get_unchecked(i);
        }
    }
    sum
}

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn neon_dot_f32(a: &[f32], b: &[f32], len: usize) -> f32 {
    use core::arch::aarch64::{vaddvq_f32, vdupq_n_f32, vfmaq_f32, vld1q_f32};

    unsafe {
        let mut acc = vdupq_n_f32(0.0);
        let mut i = 0;
        let chunks = len / 4;

        for _ in 0..chunks {
            let va = vld1q_f32(a.as_ptr().add(i));
            let vb = vld1q_f32(b.as_ptr().add(i));
            acc = vfmaq_f32(acc, va, vb);
            i += 4;
        }

        let mut sum = vaddvq_f32(acc);
        while i < len {
            sum += *a.get_unchecked(i) * *b.get_unchecked(i);
            i += 1;
        }

        sum
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn avx2_dot_f32(a: &[f32], b: &[f32], len: usize) -> f32 {
    use core::arch::x86_64::{_mm256_add_ps, _mm256_loadu_ps, _mm256_mul_ps, _mm256_setzero_ps};

    unsafe {
        let mut acc = _mm256_setzero_ps();
        let mut i = 0;
        let chunks = len / 8;

        for _ in 0..chunks {
            let va = _mm256_loadu_ps(a.as_ptr().add(i));
            let vb = _mm256_loadu_ps(b.as_ptr().add(i));
            let prod = _mm256_mul_ps(va, vb);
            acc = _mm256_add_ps(acc, prod);
            i += 8;
        }

        let mut sum = horizontal_sum_256(acc);

        while i < len {
            sum += *a.get_unchecked(i) * *b.get_unchecked(i);
            i += 1;
        }

        sum
    }
}

// ── Outer Product Accumulation ────────────────────────────────

/// SIMD-accelerated outer product accumulation: `acc[i*n + j] += a[i] * b[j]`.
///
/// Used for HLA rank-1 updates (SK += kkᵀ, CQV += qvᵀ, PKV += kvᵀ).
/// `acc` is `[m × n]` row-major, `a` is `[m]`, `b` is `[n]`.
#[inline]
pub fn simd_outer_product_acc(acc: &mut [f32], a: &[f32], b: &[f32], m: usize, n: usize) {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { neon_outer_product_acc(acc, a, b, m, n) }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if is_avx2_fma_available() {
            unsafe { avx2_outer_product_acc(acc, a, b, m, n) }
        } else {
            scalar_outer_product_acc(acc, a, b, m, n)
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        scalar_outer_product_acc(acc, a, b, m, n)
    }
}

#[inline]
#[allow(dead_code)]
fn scalar_outer_product_acc(acc: &mut [f32], a: &[f32], b: &[f32], m: usize, n: usize) {
    for i in 0..m {
        let ai = unsafe { *a.get_unchecked(i) };
        let row = &mut acc[i * n..i * n + n];
        for j in 0..n {
            unsafe {
                *row.get_unchecked_mut(j) += ai * *b.get_unchecked(j);
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn neon_outer_product_acc(acc: &mut [f32], a: &[f32], b: &[f32], m: usize, n: usize) {
    use core::arch::aarch64::{vfmaq_f32, vld1q_dup_f32, vld1q_f32, vst1q_f32};

    unsafe {
        let n_chunks = n / 4;

        for i in 0..m {
            let ai = *a.get_unchecked(i);
            let va = vld1q_dup_f32(&ai);
            let row = &mut acc[i * n..i * n + n];

            let mut j = 0;
            for _ in 0..n_chunks {
                let vacc = vld1q_f32(row.as_ptr().add(j));
                let vb = vld1q_f32(b.as_ptr().add(j));
                let vresult = vfmaq_f32(vacc, va, vb);
                vst1q_f32(row.as_mut_ptr().add(j), vresult);
                j += 4;
            }

            while j < n {
                *row.get_unchecked_mut(j) += ai * *b.get_unchecked(j);
                j += 1;
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn avx2_outer_product_acc(acc: &mut [f32], a: &[f32], b: &[f32], m: usize, n: usize) {
    use core::arch::x86_64::{
        _mm256_add_ps, _mm256_broadcast_ss, _mm256_loadu_ps, _mm256_mul_ps, _mm256_storeu_ps,
    };

    unsafe {
        let n_chunks8 = n / 8;

        for i in 0..m {
            let ai = *a.get_unchecked(i);
            let va = _mm256_broadcast_ss(&ai);
            let row = &mut acc[i * n..i * n + n];

            let mut j = 0;
            for _ in 0..n_chunks8 {
                let vacc = _mm256_loadu_ps(row.as_ptr().add(j));
                let vb = _mm256_loadu_ps(b.as_ptr().add(j));
                let prod = _mm256_mul_ps(va, vb);
                let vresult = _mm256_add_ps(vacc, prod);
                _mm256_storeu_ps(row.as_mut_ptr().add(j), vresult);
                j += 8;
            }

            while j < n {
                *row.get_unchecked_mut(j) += ai * *b.get_unchecked(j);
                j += 1;
            }
        }
    }
}

// ── Matrix-Vector Multiply ────────────────────────────────────

/// SIMD-accelerated matvec: `acc[i] = Σ mat[i*cols + j] * vec[j]` for each row.
///
/// Used for HLA readout (qᵀ·SK, qᵀ·PKV, etc.).
/// `mat` is `[rows × cols]` row-major, `vec` is `[cols]`, `acc` is `[rows]`.
#[inline]
pub fn simd_matvec(acc: &mut [f32], mat: &[f32], vec: &[f32], rows: usize, cols: usize) {
    for r in 0..rows {
        let row = &mat[r * cols..r * cols + cols];
        unsafe {
            *acc.get_unchecked_mut(r) = simd_dot_f32(row, vec, cols);
        }
    }
}

// ── Matmul Row Dispatch ───────────────────────────────────────

/// SIMD-accelerated matmul dispatch: `output[r] = dot(weight_row_r, input)`.
///
/// Replaces the inner loop of `matmul()` in `types.rs`.
#[inline(always)]
pub fn simd_matmul_rows(
    output: &mut [f32],
    weight: &[f32],
    input: &[f32],
    rows: usize,
    cols: usize,
) {
    for r in 0..rows {
        let row_off = r * cols;
        unsafe {
            *output.get_unchecked_mut(r) =
                simd_dot_f32(&weight[row_off..row_off + cols], input, cols);
        }
    }
}

/// SIMD-accelerated matmul + ReLU: `output[r] = max(0, dot(weight_row_r, input))`.
///
/// Replaces the inner loop of `matmul_relu()` in `types.rs`.
#[inline(always)]
pub fn simd_matmul_relu_rows(
    output: &mut [f32],
    weight: &[f32],
    input: &[f32],
    rows: usize,
    cols: usize,
) {
    for r in 0..rows {
        let row_off = r * cols;
        let sum = simd_dot_f32(&weight[row_off..row_off + cols], input, cols);
        unsafe {
            *output.get_unchecked_mut(r) = sum.max(0.0);
        }
    }
}

// ── Sparse Dot Product (Scattered Gather) ────────────────────

/// SIMD sparse dot: `Σ weight[row_off + active_indices[i]] * active_values[i]` for `i in 0..alive`.
///
/// Gathers weight values at scattered positions and multiplies with contiguous
/// `active_values`. Used for sparse MLP matmul where only alive (post-ReLU)
/// neurons contribute.
///
/// Scalar fallback for alive ≤ 4 (gather overhead not worth it).
/// NEON/AVX2 processes 4/8 elements per iteration for larger counts.
#[inline]
pub fn simd_sparse_dot_f32(
    weight: &[f32],
    row_off: usize,
    active_indices: &[usize],
    active_values: &[f32],
    alive: usize,
) -> f32 {
    // Scalar fallback for very sparse cases — gather setup overhead exceeds benefit.
    if alive <= 4 {
        let mut sum = 0.0f32;
        for i in 0..alive {
            unsafe {
                let c = *active_indices.get_unchecked(i);
                sum += *weight.get_unchecked(row_off + c) * *active_values.get_unchecked(i);
            }
        }
        return sum;
    }

    #[cfg(target_arch = "aarch64")]
    {
        unsafe { neon_sparse_dot_f32(weight, row_off, active_indices, active_values, alive) }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if is_avx2_fma_available() {
            unsafe { avx2_sparse_dot_f32(weight, row_off, active_indices, active_values, alive) }
        } else {
            scalar_sparse_dot_f32(weight, row_off, active_indices, active_values, alive)
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        scalar_sparse_dot_f32(weight, row_off, active_indices, active_values, alive)
    }
}

#[allow(dead_code)]
fn scalar_sparse_dot_f32(
    weight: &[f32],
    row_off: usize,
    active_indices: &[usize],
    active_values: &[f32],
    alive: usize,
) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..alive {
        unsafe {
            let c = *active_indices.get_unchecked(i);
            sum += *weight.get_unchecked(row_off + c) * *active_values.get_unchecked(i);
        }
    }
    sum
}

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn neon_sparse_dot_f32(
    weight: &[f32],
    row_off: usize,
    active_indices: &[usize],
    active_values: &[f32],
    alive: usize,
) -> f32 {
    use core::arch::aarch64::{vaddvq_f32, vdupq_n_f32, vfmaq_f32, vld1q_f32, vsetq_lane_f32};

    unsafe {
        let mut acc = vdupq_n_f32(0.0);
        let mut i = 0;
        let chunks = alive / 4;

        for _ in 0..chunks {
            // Gather 4 weight values from scattered indices into NEON register
            let mut ww = vdupq_n_f32(0.0);
            ww = vsetq_lane_f32(
                *weight.get_unchecked(row_off + *active_indices.get_unchecked(i)),
                ww,
                0,
            );
            ww = vsetq_lane_f32(
                *weight.get_unchecked(row_off + *active_indices.get_unchecked(i + 1)),
                ww,
                1,
            );
            ww = vsetq_lane_f32(
                *weight.get_unchecked(row_off + *active_indices.get_unchecked(i + 2)),
                ww,
                2,
            );
            ww = vsetq_lane_f32(
                *weight.get_unchecked(row_off + *active_indices.get_unchecked(i + 3)),
                ww,
                3,
            );

            // Load 4 contiguous active values
            let vv = vld1q_f32(active_values.as_ptr().add(i));

            // FMA: acc += ww * vv (4 multiply-accumulates in one instruction)
            acc = vfmaq_f32(acc, ww, vv);
            i += 4;
        }

        let mut sum = vaddvq_f32(acc);
        // Remainder tail (0..3 elements)
        while i < alive {
            let c = *active_indices.get_unchecked(i);
            sum += *weight.get_unchecked(row_off + c) * *active_values.get_unchecked(i);
            i += 1;
        }

        sum
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn avx2_sparse_dot_f32(
    weight: &[f32],
    row_off: usize,
    active_indices: &[usize],
    active_values: &[f32],
    alive: usize,
) -> f32 {
    use core::arch::x86_64::{
        _mm256_fmadd_ps, _mm256_i32gather_ps, _mm256_loadu_ps, _mm256_set_epi32, _mm256_setzero_ps,
    };

    unsafe {
        let mut acc = _mm256_setzero_ps();
        let mut i = 0;
        let chunks = alive / 8;

        for _ in 0..chunks {
            // Load 8 indices into __m256i for hardware gather
            // Note: _mm256_set_epi32 parameter order is (e7, e6, ..., e0)
            let idx = _mm256_set_epi32(
                *active_indices.get_unchecked(i + 7) as i32,
                *active_indices.get_unchecked(i + 6) as i32,
                *active_indices.get_unchecked(i + 5) as i32,
                *active_indices.get_unchecked(i + 4) as i32,
                *active_indices.get_unchecked(i + 3) as i32,
                *active_indices.get_unchecked(i + 2) as i32,
                *active_indices.get_unchecked(i + 1) as i32,
                *active_indices.get_unchecked(i) as i32,
            );

            // Hardware gather: weight[row_off + active_indices[j]] for each lane j
            let ww = _mm256_i32gather_ps(weight.as_ptr().add(row_off), idx, 4);

            // Load 8 contiguous active values
            let vv = _mm256_loadu_ps(active_values.as_ptr().add(i));

            // FMA: acc += ww * vv (8 multiply-accumulates in one instruction)
            acc = _mm256_fmadd_ps(ww, vv, acc);
            i += 8;
        }

        let mut sum = horizontal_sum_256(acc);
        // Remainder tail (0..7 elements)
        while i < alive {
            let c = *active_indices.get_unchecked(i);
            sum += *weight.get_unchecked(row_off + c) * *active_values.get_unchecked(i);
            i += 1;
        }

        sum
    }
}

// ── Sparse Matmul Row Dispatch ────────────────────────────────

/// SIMD-accelerated sparse matmul: `output[r] = sparse_dot(weight_row_r, active)`.
///
/// Replaces the inner loop of `sparse_matmul()` in `types.rs`.
/// `alive` is the count of active (non-zero) input elements after ReLU.
///
/// For each output row, computes the dot product using only the `alive`
/// elements at `active_indices` positions from the weight row.
#[inline(always)]
pub fn simd_sparse_matmul_rows(
    output: &mut [f32],
    weight: &[f32],
    active_indices: &[usize],
    active_values: &[f32],
    rows: usize,
    cols: usize,
    alive: usize,
) {
    for r in 0..rows {
        let row_off = r * cols;
        unsafe {
            *output.get_unchecked_mut(r) =
                simd_sparse_dot_f32(weight, row_off, active_indices, active_values, alive);
        }
    }
}

// ── Scale Inplace ─────────────────────────────────────────────

/// SIMD-accelerated in-place scale: `x[i] *= scale` for all `i`.
///
/// General utility for softmax normalize, rmsnorm scale, HLA decay,
/// TurboQuant normalize, and any bulk `*= scale` pattern.
///
/// NEON: 4× f32 per op. AVX2: 8× f32 per op. Scalar fallback for remainder.
#[inline]
pub fn simd_scale_inplace(x: &mut [f32], scale: f32) {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { neon_scale_inplace(x, scale) }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if is_avx2_fma_available() {
            unsafe { avx2_scale_inplace(x, scale) }
        } else {
            scalar_scale_inplace(x, scale)
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        scalar_scale_inplace(x, scale)
    }
}

#[inline]
#[allow(dead_code)]
fn scalar_scale_inplace(x: &mut [f32], scale: f32) {
    for val in x.iter_mut() {
        *val *= scale;
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn neon_scale_inplace(x: &mut [f32], scale: f32) {
    use core::arch::aarch64::{vdupq_n_f32, vld1q_f32, vmulq_f32, vst1q_f32};

    unsafe {
        let vs = vdupq_n_f32(scale);
        let mut i = 0;
        let chunks = x.len() / 4;

        for _ in 0..chunks {
            let vx = vld1q_f32(x.as_ptr().add(i));
            let result = vmulq_f32(vx, vs);
            vst1q_f32(x.as_mut_ptr().add(i), result);
            i += 4;
        }

        while i < x.len() {
            *x.get_unchecked_mut(i) *= scale;
            i += 1;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn avx2_scale_inplace(x: &mut [f32], scale: f32) {
    use core::arch::x86_64::{_mm256_loadu_ps, _mm256_mul_ps, _mm256_set1_ps, _mm256_storeu_ps};

    unsafe {
        let vs = _mm256_set1_ps(scale);
        let mut i = 0;
        let chunks = x.len() / 8;

        for _ in 0..chunks {
            let vx = _mm256_loadu_ps(x.as_ptr().add(i));
            let result = _mm256_mul_ps(vx, vs);
            _mm256_storeu_ps(x.as_mut_ptr().add(i), result);
            i += 8;
        }

        while i < x.len() {
            *x.get_unchecked_mut(i) *= scale;
            i += 1;
        }
    }
}

// ── x86_64 Horizontal Sum Helpers ─────────────────────────────

#[cfg(target_arch = "x86_64")]
#[inline]
fn horizontal_sum_256(v: core::arch::x86_64::__m256) -> f32 {
    use core::arch::x86_64::{_mm_add_ps, _mm256_castps256_ps128, _mm256_extractf128_ps};
    unsafe {
        let hi = _mm256_extractf128_ps(v, 1);
        let lo = _mm256_castps256_ps128(v);
        let sum128 = _mm_add_ps(lo, hi);
        horizontal_sum_128(sum128)
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
fn horizontal_sum_128(v: core::arch::x86_64::__m128) -> f32 {
    use core::arch::x86_64::{_mm_add_ps, _mm_add_ss, _mm_cvtss_f32, _mm_shuffle_ps};
    unsafe {
        let shuf = _mm_shuffle_ps(v, v, 0xB1);
        let sums = _mm_add_ps(v, shuf);
        let shuf2 = _mm_shuffle_ps(sums, sums, 0x2A);
        let result = _mm_add_ss(sums, shuf2);
        _mm_cvtss_f32(result)
    }
}

// ── Tests ─────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simd_level_matches_platform() {
        let level = simd_level();
        #[cfg(target_arch = "aarch64")]
        assert_eq!(level, SimdLevel::Neon);
        #[cfg(target_arch = "x86_64")]
        assert!(matches!(level, SimdLevel::Avx2 | SimdLevel::Scalar));
        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        assert_eq!(level, SimdLevel::Scalar);
    }

    #[test]
    fn dot_product_aligned_len_8() {
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = [0.5f32, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0];

        let scalar = scalar_dot_f32(&a, &b, 8);
        let simd = simd_dot_f32(&a, &b, 8);

        assert!((scalar - simd).abs() < 1e-4, "scalar={scalar}, simd={simd}");
        // Expected: 0.5+2+4.5+8+12.5+18+24.5+32 = 102
        assert!((simd - 102.0).abs() < 1e-4, "simd={simd}");
    }

    #[test]
    fn dot_product_non_aligned_len() {
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let b = [1.0f32, 1.0, 1.0, 1.0, 1.0];

        let scalar = scalar_dot_f32(&a, &b, 5);
        let simd = simd_dot_f32(&a, &b, 5);

        assert!((scalar - simd).abs() < 1e-4, "scalar={scalar}, simd={simd}");
        assert!((simd - 15.0).abs() < 1e-4);
    }

    #[test]
    fn dot_product_len_4() {
        let a = [1.0f32, 2.0, 3.0, 4.0];
        let b = [1.0f32, 0.5, 0.25, 0.125];

        let expected = 1.0 + 1.0 + 0.75 + 0.5;
        let simd = simd_dot_f32(&a, &b, 4);

        assert!((simd - expected).abs() < 1e-4);
    }

    #[test]
    fn dot_product_len_32() {
        // Game config n_embd=32
        let a: Vec<f32> = (0..32).map(|i| (i as f32 + 1.0) * 0.1).collect();
        let b: Vec<f32> = (0..32).map(|i| (i as f32 + 1.0) * 0.05).collect();

        let scalar = scalar_dot_f32(&a, &b, 32);
        let simd = simd_dot_f32(&a, &b, 32);

        assert!((scalar - simd).abs() < 1e-3, "scalar={scalar}, simd={simd}");
    }

    #[test]
    fn dot_product_zero_length() {
        let simd = simd_dot_f32(&[], &[], 0);
        assert!((simd - 0.0).abs() < 1e-6);
    }

    #[test]
    fn outer_product_4x4_matches_scalar() {
        let m = 4;
        let n = 4;
        let a = [1.0f32, 2.0, 3.0, 4.0];
        let b = [0.5f32, 1.0, 1.5, 2.0];

        let mut acc_scalar = vec![0.0f32; m * n];
        let mut acc_simd = vec![0.0f32; m * n];

        scalar_outer_product_acc(&mut acc_scalar, &a, &b, m, n);
        simd_outer_product_acc(&mut acc_simd, &a, &b, m, n);

        for i in 0..m * n {
            assert!(
                (acc_scalar[i] - acc_simd[i]).abs() < 1e-4,
                "mismatch at {i}: scalar={}, simd={}",
                acc_scalar[i],
                acc_simd[i]
            );
        }
    }

    #[test]
    fn outer_product_8x8_matches_scalar() {
        // Game config: hd=8
        let m = 8;
        let n = 8;
        let a: Vec<f32> = (0..m).map(|i| (i + 1) as f32 * 0.1).collect();
        let b: Vec<f32> = (0..n).map(|j| (j + 1) as f32 * 0.2).collect();

        let mut acc_scalar = vec![0.0f32; m * n];
        let mut acc_simd = vec![0.0f32; m * n];

        scalar_outer_product_acc(&mut acc_scalar, &a, &b, m, n);
        simd_outer_product_acc(&mut acc_simd, &a, &b, m, n);

        for i in 0..m * n {
            assert!(
                (acc_scalar[i] - acc_simd[i]).abs() < 1e-4,
                "mismatch at {i}: scalar={}, simd={}",
                acc_scalar[i],
                acc_simd[i]
            );
        }
    }

    #[test]
    fn outer_product_accumulates() {
        let m = 4;
        let n = 4;
        let a = [1.0f32, 0.0, 0.0, 0.0];
        let b = [0.0f32, 0.0, 0.0, 1.0];

        let mut acc = vec![0.0f32; m * n];
        simd_outer_product_acc(&mut acc, &a, &b, m, n);

        // Only acc[0*4 + 3] = 1.0 * 1.0 = 1.0 should be non-zero
        assert!((acc[3] - 1.0).abs() < 1e-5);
        for (i, &val) in acc.iter().enumerate() {
            if i != 3 {
                assert!(val.abs() < 1e-6, "acc[{i}] should be 0, got {val}");
            }
        }
    }

    #[test]
    fn matvec_matches_scalar() {
        let rows = 3;
        let cols = 4;
        let mat = [
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0f32,
        ];
        let vec = [1.0, 0.0, 1.0, 0.0f32];

        let mut acc_scalar = vec![0.0f32; rows];
        let mut acc_simd = vec![0.0f32; rows];

        for r in 0..rows {
            let mut sum = 0.0f32;
            for c in 0..cols {
                sum += mat[r * cols + c] * vec[c];
            }
            acc_scalar[r] = sum;
        }

        simd_matvec(&mut acc_simd, &mat, &vec, rows, cols);

        for r in 0..rows {
            assert!(
                (acc_scalar[r] - acc_simd[r]).abs() < 1e-4,
                "mismatch at row {r}: scalar={}, simd={}",
                acc_scalar[r],
                acc_simd[r]
            );
        }
    }

    #[test]
    fn matmul_rows_identity() {
        let rows = 4;
        let cols = 4;
        let weight = [
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];
        let input = [1.0, 2.0, 3.0, 4.0f32];

        let mut output = vec![0.0f32; rows];
        simd_matmul_rows(&mut output, &weight, &input, rows, cols);

        assert!((output[0] - 1.0).abs() < 1e-5);
        assert!((output[1] - 2.0).abs() < 1e-5);
        assert!((output[2] - 3.0).abs() < 1e-5);
        assert!((output[3] - 4.0).abs() < 1e-5);
    }

    #[test]
    fn matmul_relu_clamps_negative() {
        let rows = 2;
        let cols = 2;
        let weight = [-1.0, 0.0, 1.0, 1.0];
        let input = [1.0, 1.0];

        let mut output = vec![0.0f32; rows];
        simd_matmul_relu_rows(&mut output, &weight, &input, rows, cols);

        assert!((output[0]).abs() < 1e-5, "negative should clamp to 0");
        assert!((output[1] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn fma_row_matches_dot() {
        let a = [1.0f32, 2.0, 3.0, 4.0];
        let b = [0.5f32, 1.0, 1.5, 2.0];

        let dot = simd_dot_f32(&a, &b, 4);
        let fma = simd_fma_row(&a, &b, 4);

        assert!((dot - fma).abs() < 1e-6);
    }

    // ── Sparse SIMD Tests ────────────────────────────────────

    #[test]
    fn sparse_dot_matches_scalar_dense() {
        // 8 elements, all alive (indices 0..7) — should match dense dot
        let weight = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let indices: Vec<usize> = (0..8).collect();
        let values = [0.5f32, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0];

        let sparse = simd_sparse_dot_f32(&weight, 0, &indices, &values, 8);
        let dense = simd_dot_f32(&weight, &values, 8);

        assert!(
            (sparse - dense).abs() < 1e-4,
            "sparse={sparse}, dense={dense}"
        );
    }

    #[test]
    fn sparse_dot_matches_scalar_sparse() {
        // 13 elements alive out of 64 (typical micro config: 20% of mlp_hidden=64)
        let mut weight = vec![0.0f32; 64];
        for (i, w) in weight.iter_mut().enumerate() {
            *w = (i as f32 + 1.0) * 0.01;
        }
        let indices: Vec<usize> = vec![0, 3, 7, 12, 15, 20, 25, 31, 38, 45, 50, 56, 63];
        let values: Vec<f32> = indices.iter().map(|&i| weight[i] * 2.0).collect();

        let simd_result = simd_sparse_dot_f32(&weight, 0, &indices, &values, 13);
        let scalar_result = scalar_sparse_dot_f32(&weight, 0, &indices, &values, 13);

        assert!(
            (simd_result - scalar_result).abs() < 1e-4,
            "simd={simd_result}, scalar={scalar_result}"
        );
    }

    #[test]
    fn sparse_dot_small_alive_uses_scalar() {
        // alive=3 — should use inline scalar fallback (≤4)
        let weight = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let indices = vec![0usize, 3, 7];
        let values = [0.5f32, 1.0, 1.5];

        let result = simd_sparse_dot_f32(&weight, 0, &indices, &values, 3);
        let expected = 1.0 * 0.5 + 4.0 * 1.0 + 8.0 * 1.5; // 0.5 + 4.0 + 12.0 = 16.5

        assert!(
            (result - expected).abs() < 1e-4,
            "result={result}, expected={expected}"
        );
    }

    #[test]
    fn sparse_dot_zero_alive() {
        let weight = [1.0f32, 2.0, 3.0, 4.0];
        let indices: Vec<usize> = vec![];
        let values: Vec<f32> = vec![];

        let result = simd_sparse_dot_f32(&weight, 0, &indices, &values, 0);
        assert!(result.abs() < 1e-6, "expected 0.0, got {result}");
    }

    #[test]
    fn sparse_dot_with_row_offset() {
        // 8-element weight row at offset 4 in a 12-element weight matrix
        let mut weight = [0.0f32; 12]; // first 4 are padding
        weight[4] = 1.0;
        weight[5] = 2.0;
        weight[6] = 3.0;
        weight[7] = 4.0;
        weight[8] = 5.0;
        weight[9] = 6.0;
        weight[10] = 7.0;
        weight[11] = 8.0;
        // Need mutable for construction
        let weight = weight;

        let indices: Vec<usize> = (0..8).collect();
        let values = [1.0f32; 8];

        let result = simd_sparse_dot_f32(&weight, 4, &indices, &values, 8);
        // Expected: 1+2+3+4+5+6+7+8 = 36
        assert!((result - 36.0).abs() < 1e-4, "result={result}");
    }

    #[test]
    fn sparse_dot_alive_5_triggers_simd() {
        // alive=5 — just above scalar fallback threshold (4)
        let weight = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let indices: Vec<usize> = (0..8).collect();
        let values = [1.0f32, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];

        let simd_result = simd_sparse_dot_f32(&weight, 0, &indices, &values, 5);
        let expected = 1.0 + 2.0 + 3.0 + 4.0 + 5.0; // first 5 only

        assert!(
            (simd_result - expected).abs() < 1e-4,
            "simd={simd_result}, expected={expected}"
        );
    }

    #[test]
    fn sparse_matmul_rows_matches_scalar() {
        let rows = 4;
        let cols = 8;
        // Identity-like weight: row r has weight[r*cols + r] = 1.0, rest = 0.1
        let weight: Vec<f32> = (0..rows * cols)
            .map(|i| {
                let r = i / cols;
                let c = i % cols;
                if r == c { 1.0 } else { 0.1 }
            })
            .collect();

        // Only indices 1, 3, 5 are alive with values
        let indices = vec![1usize, 3, 5];
        let values = vec![2.0f32, 3.0, 4.0];

        let mut output_scalar = vec![0.0f32; rows];
        let mut output_simd = vec![0.0f32; rows];

        // Scalar
        for (r, out) in output_scalar.iter_mut().enumerate() {
            *out = scalar_sparse_dot_f32(&weight, r * cols, &indices, &values, 3);
        }

        // SIMD
        simd_sparse_matmul_rows(&mut output_simd, &weight, &indices, &values, rows, cols, 3);

        for (r, (scalar, simd)) in output_scalar.iter().zip(output_simd.iter()).enumerate() {
            assert!(
                (scalar - simd).abs() < 1e-4,
                "row {r}: scalar={scalar}, simd={simd}"
            );
        }
    }

    #[test]
    fn sparse_matmul_rows_game_config() {
        // Game config: n_embd=32 rows, mlp_hidden=128 cols, ~20% alive = 26 elements
        let rows = 32;
        let cols = 128;
        let weight: Vec<f32> = (0..rows * cols).map(|i| (i % 100) as f32 * 0.01).collect();

        // Simulate 26 alive neurons (20% of 128)
        let alive = 26;
        let indices: Vec<usize> = (0..alive).map(|i| i * (cols / alive)).collect();
        let values: Vec<f32> = (0..alive).map(|i| (i as f32 + 1.0) * 0.1).collect();

        let mut output_scalar = vec![0.0f32; rows];
        let mut output_simd = vec![0.0f32; rows];

        for (r, out) in output_scalar.iter_mut().enumerate() {
            *out = scalar_sparse_dot_f32(&weight, r * cols, &indices, &values, alive);
        }
        simd_sparse_matmul_rows(
            &mut output_simd,
            &weight,
            &indices,
            &values,
            rows,
            cols,
            alive,
        );

        for r in 0..rows {
            assert!(
                (output_scalar[r] - output_simd[r]).abs() < 1e-3,
                "row {r}: scalar={}, simd={}",
                output_scalar[r],
                output_simd[r]
            );
        }
    }

    // ── simd_scale_inplace tests ──────────────────────────────

    #[test]
    fn scale_aligned_len_8() {
        let mut x = [2.0f32, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0];
        simd_scale_inplace(&mut x, 0.5);
        let expected = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        for i in 0..8 {
            assert!((x[i] - expected[i]).abs() < 1e-6, "x[{i}]={}", x[i]);
        }
    }

    #[test]
    fn scale_non_aligned_len_13() {
        let mut x = [1.0f32; 13];
        simd_scale_inplace(&mut x, 3.0);
        for i in 0..13 {
            assert!((x[i] - 3.0).abs() < 1e-6, "x[{i}]={}", x[i]);
        }
    }

    #[test]
    fn scale_empty() {
        let mut x: [f32; 0] = [];
        simd_scale_inplace(&mut x, 2.0); // should not panic
    }

    #[test]
    fn scale_single_element() {
        let mut x = [5.0f32];
        simd_scale_inplace(&mut x, 0.2);
        assert!((x[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn scale_zero() {
        let mut x = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        simd_scale_inplace(&mut x, 0.0);
        for val in &x {
            assert!(*val == 0.0, "expected 0.0, got {val}");
        }
    }

    #[test]
    fn scale_matches_scalar() {
        let mut x_simd: Vec<f32> = (0..97).map(|i| (i as f32 * 0.1).sin()).collect();
        let mut x_scalar = x_simd.clone();
        let scale = 0.42f32;

        simd_scale_inplace(&mut x_simd, scale);
        scalar_scale_inplace(&mut x_scalar, scale);

        for i in 0..x_simd.len() {
            assert!(
                (x_simd[i] - x_scalar[i]).abs() < 1e-6,
                "x[{i}]: simd={}, scalar={}",
                x_simd[i],
                x_scalar[i]
            );
        }
    }
}
