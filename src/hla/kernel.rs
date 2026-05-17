//! Higher-order Linear Attention streaming kernels.
//!
//! Implements zero-alloc streaming recurrence for symmetric second-order HLA
//! and asymmetric AHLA. These kernels replace the O(N·d) attention loop with
//! O(d² + d·dv) (symmetric) or O(d·dv) (asymmetric) constant-time operations.
//!
//! # Update Order (Critical Correctness Requirement)
//!
//! Cross-terms G,h MUST be computed using OLD CQV,mQ BEFORE updating them:
//!
//! ```text
//! 1. Compute kᵀ·CQV_{t-1} → G_t += k · (kᵀ·CQV_{t-1})
//! 2. Compute kᵀ·mQ_{t-1}  → h_t += k · (kᵀ·mQ_{t-1})
//! 3. THEN: SK_t += kkᵀ, CQV_t += qvᵀ, mQ_t += q
//! ```
//!
//! Violating this order introduces look-ahead bias (using future information).
//!
//! Reference: Zhang et al. (2026), "Higher-order Linear Attention," §3.

use crate::hla::types::{AhlaLayerState, AhlaQHeadState, HlaLayerState, HlaQHeadState};
use crate::simd;

// ── Symmetric Second-Order HLA Kernels ─────────────────────────

/// Update symmetric HLA state with new (q_t, k_t, v_t) for one Q head.
///
/// Streaming recurrence with correct update ordering:
/// 1. Cross-terms G, h using OLD CQV, mQ
/// 2. Main accumulators SK, CQV, mQ
///
/// # Arguments
/// * `sk` - Key second moment for this KV group [hd × hd] (mutated in-place)
/// * `q_head` - Per-Q-head state (mutated in-place)
/// * `q` - Query slice for this head [hd]
/// * `k` - Key slice for this KV group [hd]
/// * `v` - Value slice for this head [hd] (same as kv_dim in practice)
/// * `hd` - Head dimension
/// * `gamma` - Exponential decay (1.0 = no decay)
/// * `tmp_k_cqv` - Pre-allocated temp buffer [hd] (avoided heap alloc)
/// * `tmp_q_g` - Pre-allocated temp buffer [hd] (avoided heap alloc)
#[allow(clippy::too_many_arguments)]
#[inline]
pub fn hla_state_update(
    sk: &mut [f32],
    q_head: &mut HlaQHeadState,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    hd: usize,
    gamma: f32,
    tmp_k_cqv: &mut [f32],
    _tmp_q_g: &mut [f32],
) {
    debug_assert_eq!(sk.len(), hd * hd);
    debug_assert_eq!(q_head.cqv.len(), hd * hd);
    debug_assert_eq!(q_head.mq.len(), hd);
    debug_assert_eq!(q_head.g.len(), hd * hd);
    debug_assert_eq!(q_head.h.len(), hd);
    debug_assert_eq!(q.len(), hd);
    debug_assert_eq!(k.len(), hd);
    debug_assert_eq!(v.len(), hd);
    debug_assert!(tmp_k_cqv.len() >= hd);
    debug_assert!(_tmp_q_g.len() >= hd);

    // ── Step 1: Cross-terms using OLD state ──

    // kᵀ · CQV_{t-1} → [hd] (1×hd vector)
    // tmp_k_cqv[j] = Σ_i k[i] * CQV[i*hd + j]
    tmp_k_cqv[..hd].fill(0.0);
    for i in 0..hd {
        let ki = unsafe { *k.get_unchecked(i) };
        let cqv_row = &q_head.cqv[i * hd..i * hd + hd];
        for j in 0..hd {
            unsafe {
                *tmp_k_cqv.get_unchecked_mut(j) += ki * *cqv_row.get_unchecked(j);
            }
        }
    }

    // G_t += k_t · (kᵀ · CQV_{t-1})
    simd::simd_outer_product_acc(&mut q_head.g, k, &tmp_k_cqv[..hd], hd, hd);

    // kᵀ · mQ_{t-1} → scalar
    let k_mq = simd::simd_dot_f32(k, &q_head.mq, hd);

    // h_t += k_t · (kᵀ · mQ_{t-1})
    for i in 0..hd {
        unsafe {
            *q_head.h.get_unchecked_mut(i) += *k.get_unchecked(i) * k_mq;
        }
    }

    // ── Step 2: Main accumulators ──

    // Apply decay if needed (γ < 1.0)
    if gamma < 1.0 {
        crate::simd::simd_scale_inplace(sk, gamma);
        crate::simd::simd_scale_inplace(&mut q_head.cqv, gamma);
        crate::simd::simd_scale_inplace(&mut q_head.mq, gamma);
        crate::simd::simd_scale_inplace(&mut q_head.g, gamma);
        crate::simd::simd_scale_inplace(&mut q_head.h, gamma);
    }

    // SK_t += k_t · k_tᵀ (rank-1 update)
    simd::simd_outer_product_acc(sk, k, k, hd, hd);

    // CQV_t += q_t · v_tᵀ (rank-1 update)
    simd::simd_outer_product_acc(&mut q_head.cqv, q, v, hd, hd);

    // mQ_t += q_t
    for i in 0..hd {
        unsafe {
            *q_head.mq.get_unchecked_mut(i) += *q.get_unchecked(i);
        }
    }
}

/// Per-Q-head update for symmetric HLA (no shared state mutation).
///
/// Computes cross-terms using OLD per-head state, then updates per-head
/// accumulators. Shared SK must be updated separately (once per KV group)
/// by [`hla_layer_update`].
#[allow(clippy::too_many_arguments)]
#[inline]
fn hla_per_head_update(
    q_head: &mut HlaQHeadState,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    hd: usize,
    gamma: f32,
    tmp_k_cqv: &mut [f32],
    _tmp_q_g: &mut [f32],
) {
    // Step 1: Cross-terms using OLD CQV, mQ (before decay/accumulation)

    // kᵀ · CQV_{t-1} → [hd]
    tmp_k_cqv[..hd].fill(0.0);
    for i in 0..hd {
        let ki = unsafe { *k.get_unchecked(i) };
        let cqv_row = &q_head.cqv[i * hd..i * hd + hd];
        for j in 0..hd {
            unsafe {
                *tmp_k_cqv.get_unchecked_mut(j) += ki * *cqv_row.get_unchecked(j);
            }
        }
    }

    // G_t += k_t · (kᵀ · CQV_{t-1})
    simd::simd_outer_product_acc(&mut q_head.g, k, &tmp_k_cqv[..hd], hd, hd);

    // kᵀ · mQ_{t-1} → scalar
    let k_mq = simd::simd_dot_f32(k, &q_head.mq, hd);

    // h_t += k_t · (kᵀ · mQ_{t-1})
    for i in 0..hd {
        unsafe {
            *q_head.h.get_unchecked_mut(i) += *k.get_unchecked(i) * k_mq;
        }
    }

    // Step 2: Decay per-head state
    if gamma < 1.0 {
        crate::simd::simd_scale_inplace(&mut q_head.cqv, gamma);
        crate::simd::simd_scale_inplace(&mut q_head.mq, gamma);
        crate::simd::simd_scale_inplace(&mut q_head.g, gamma);
        crate::simd::simd_scale_inplace(&mut q_head.h, gamma);
    }

    // Step 3: Accumulate per-head state
    // CQV_t += q_t · v_tᵀ
    simd::simd_outer_product_acc(&mut q_head.cqv, q, v, hd, hd);

    // mQ_t += q_t
    for i in 0..hd {
        unsafe {
            *q_head.mq.get_unchecked_mut(i) += *q.get_unchecked(i);
        }
    }
}

/// Symmetric HLA readout: compute attention output from current state.
///
/// ```text
/// o_t = q_tᵀ (SK_t · CQV_t − G_t)
/// ```
///
/// Two-stage computation to avoid materializing the full d×d product:
/// 1. `u = q_tᵀ · SK` (1×d vector)
/// 2. `out[j] = u · CQV[:,j] − q_tᵀ · G[:,j]`
///
/// # Arguments
/// * `q` - Query for this head [hd]
/// * `sk` - Key second moment [hd × hd]
/// * `q_head` - Per-Q-head state (CQV, G used; mQ, h not used here)
/// * `hd` - Head dimension
/// * `out` - Output buffer [hd]
/// * `tmp_u` - Pre-allocated temp buffer [hd]
#[inline]
pub fn hla_readout(
    q: &[f32],
    sk: &[f32],
    q_head: &HlaQHeadState,
    hd: usize,
    out: &mut [f32],
    tmp_u: &mut [f32],
) {
    debug_assert_eq!(sk.len(), hd * hd);
    debug_assert!(out.len() >= hd);
    debug_assert!(tmp_u.len() >= hd);

    // TODO(T6): u = q_tᵀ · SK is a transpose matvec (SK^T * q with row-major SK).
    // `simd_matvec` requires row-major * column-vector, so doesn't apply directly.
    // Could benefit from a `simd_matvec_transpose` kernel or `simd_scalar_fma`.
    // Kept scalar for now — only hd² ops (256 for hd=16).
    tmp_u[..hd].fill(0.0);
    for i in 0..hd {
        let qi = unsafe { *q.get_unchecked(i) };
        let sk_row = &sk[i * hd..i * hd + hd];
        for j in 0..hd {
            unsafe {
                *tmp_u.get_unchecked_mut(j) += qi * *sk_row.get_unchecked(j);
            }
        }
    }

    // out[j] = (u · CQV[:,j]) − (q · G[:,j])
    for j in 0..hd {
        let mut val = 0.0f32;
        for i in 0..hd {
            unsafe {
                val += *tmp_u.get_unchecked(i) * *q_head.cqv.get_unchecked(i * hd + j);
                val -= *q.get_unchecked(i) * *q_head.g.get_unchecked(i * hd + j);
            }
        }
        unsafe {
            *out.get_unchecked_mut(j) = val;
        }
    }
}

/// Symmetric HLA normalization denominator.
///
/// ```text
/// denom = q_tᵀ (SK_t · mQ_t − h_t) + ε
/// ```
///
/// Used for optional normalized output: `o_t / denom`.
#[inline]
pub fn hla_denom(
    q: &[f32],
    sk: &[f32],
    q_head: &HlaQHeadState,
    hd: usize,
    eps: f32,
    tmp_u: &mut [f32],
) -> f32 {
    // TODO(T6): transpose matvec — see hla_readout comment
    tmp_u[..hd].fill(0.0);
    for i in 0..hd {
        let qi = unsafe { *q.get_unchecked(i) };
        let sk_row = &sk[i * hd..i * hd + hd];
        for j in 0..hd {
            unsafe {
                *tmp_u.get_unchecked_mut(j) += qi * *sk_row.get_unchecked(j);
            }
        }
    }

    // denom = Σ_j u[j] * mQ[j] − Σ_j q[j] * h[j]
    let mut denom = eps;
    denom += simd::simd_dot_f32(&tmp_u[..hd], &q_head.mq, hd);
    denom -= simd::simd_dot_f32(q, &q_head.h, hd);
    denom
}

// ── Asymmetric AHLA Kernel ────────────────────────────────────

/// AHLA streaming update and readout for one Q head.
///
/// Combined update+readout: O(d·dv) per token, no d×d matrix operations.
///
/// ```text
/// PKV_t = PKV_{t-1} + k_t · v_tᵀ
/// mK_t  = mK_{t-1} + k_t
/// r_t   = q_tᵀ · PKV_t
/// E_t   = E_{t-1} + k_t · r_t
/// n_t   = n_{t-1} + k_t · (q_tᵀ · mK_t)
/// o_t   = q_tᵀ · E_t
/// ```
///
/// AHLA uses left-cascaded A·A·V instead of symmetric A·Aᵀ·V,
/// capturing second-order interactions at linear attention cost.
///
/// # Arguments
/// * `pkv` - Key-value prefix for this KV group [hd × hd] (mutated)
/// * `mk` - Key mass for this KV group [hd] (mutated)
/// * `q_head` - Per-Q-head AHLA state (mutated)
/// * `q` - Query slice for this head [hd]
/// * `k` - Key slice for this KV group [hd]
/// * `v` - Value slice [hd]
/// * `hd` - Head dimension
/// * `gamma` - Exponential decay (1.0 = no decay)
/// * `out` - Output buffer [hd]
/// * `tmp_r` - Pre-allocated temp buffer [hd]
#[allow(clippy::too_many_arguments)]
#[inline]
pub fn ahla_step(
    pkv: &mut [f32],
    mk: &mut [f32],
    q_head: &mut AhlaQHeadState,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    hd: usize,
    gamma: f32,
    out: &mut [f32],
    tmp_r: &mut [f32],
) {
    debug_assert_eq!(pkv.len(), hd * hd);
    debug_assert_eq!(mk.len(), hd);
    debug_assert_eq!(q_head.e.len(), hd * hd);
    debug_assert_eq!(q_head.n.len(), hd);
    debug_assert!(tmp_r.len() >= hd);
    debug_assert!(out.len() >= hd);

    // Apply decay if needed (before accumulation)
    if gamma < 1.0 {
        crate::simd::simd_scale_inplace(pkv, gamma);
        crate::simd::simd_scale_inplace(mk, gamma);
        crate::simd::simd_scale_inplace(&mut q_head.e, gamma);
        crate::simd::simd_scale_inplace(&mut q_head.n, gamma);
    }

    // PKV_t += k_t · v_tᵀ (rank-1 update)
    simd::simd_outer_product_acc(pkv, k, v, hd, hd);

    // TODO(T6): transpose matvec — see hla_readout comment
    tmp_r[..hd].fill(0.0);
    for i in 0..hd {
        let qi = unsafe { *q.get_unchecked(i) };
        let pkv_row = &pkv[i * hd..i * hd + hd];
        for j in 0..hd {
            unsafe {
                *tmp_r.get_unchecked_mut(j) += qi * *pkv_row.get_unchecked(j);
            }
        }
    }

    // mK_t += k_t
    for i in 0..hd {
        unsafe {
            *mk.get_unchecked_mut(i) += *k.get_unchecked(i);
        }
    }

    // E_t += k_t · r_t (outer product)
    simd::simd_outer_product_acc(&mut q_head.e, k, &tmp_r[..hd], hd, hd);

    // q_tᵀ · mK_t → scalar
    let q_mk = simd::simd_dot_f32(q, mk, hd);

    // n_t += k_t · (q_tᵀ · mK_t)
    for i in 0..hd {
        unsafe {
            *q_head.n.get_unchecked_mut(i) += *k.get_unchecked(i) * q_mk;
        }
    }

    // TODO(T6): transpose matvec (E^T * q with row-major E) — see hla_readout comment
    for j in 0..hd {
        let mut val = 0.0f32;
        for i in 0..hd {
            unsafe {
                val += *q.get_unchecked(i) * *q_head.e.get_unchecked(i * hd + j);
            }
        }
        unsafe {
            *out.get_unchecked_mut(j) = val;
        }
    }
}

/// AHLA normalization denominator.
///
/// ```text
/// denom = q_tᵀ · n_t + ε
/// ```
#[inline]
pub fn ahla_denom(q: &[f32], q_head: &AhlaQHeadState, hd: usize, eps: f32) -> f32 {
    let mut denom = eps;
    for i in 0..hd {
        unsafe {
            denom += *q.get_unchecked(i) * *q_head.n.get_unchecked(i);
        }
    }
    denom
}

// ── Full-Layer Helpers ────────────────────────────────────────

/// Update all heads in one layer for symmetric HLA.
///
/// Handles GQA mapping: each Q head shares the SK from its KV group.
///
/// # Arguments
/// * `layer` - Layer state with SK per KV group and per-Q-head state
/// * `q` - Full query tensor [n_head × hd]
/// * `k` - Full key tensor [n_kv_head × hd]
/// * `v` - Full value tensor [n_head × hd] (or kv_dim if shared)
/// * `config` - Model config for GQA mapping
/// * `gamma` - Exponential decay
/// * `tmp_k_cqv` - Pre-allocated temp [hd] (reused across heads)
/// * `tmp_q_g` - Pre-allocated temp [hd] (reused across heads)
#[allow(clippy::too_many_arguments)]
pub fn hla_layer_update(
    layer: &mut HlaLayerState,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    config: &crate::types::Config,
    gamma: f32,
    tmp_k_cqv: &mut [f32],
    tmp_q_g: &mut [f32],
) {
    let hd = config.head_dim;
    let n_kv = config.n_kv_head;

    // Phase 1: Update shared SK once per KV group (avoid double-accumulation with GQA)
    for g in 0..n_kv {
        // Decay
        if gamma < 1.0 {
            crate::simd::simd_scale_inplace(&mut layer.sk[g], gamma);
        }
        // SK[g] += k_g · k_gᵀ
        let k_slice = &k[g * hd..(g + 1) * hd];
        simd::simd_outer_product_acc(&mut layer.sk[g], k_slice, k_slice, hd, hd);
    }

    // Phase 2: Per-Q-head update (cross-terms + accumulators)
    for h in 0..config.n_head {
        let kv_group = h * n_kv / config.n_head;
        let q_slice = &q[h * hd..(h + 1) * hd];
        let k_slice = &k[kv_group * hd..(kv_group + 1) * hd];
        let v_slice = &v[kv_group * hd..(kv_group + 1) * hd];

        hla_per_head_update(
            &mut layer.heads[h],
            q_slice,
            k_slice,
            v_slice,
            hd,
            gamma,
            tmp_k_cqv,
            tmp_q_g,
        );
    }
}

/// Readout all heads in one layer for symmetric HLA.
///
/// Produces `attn_out[h*hd..(h+1)*hd]` for each head h.
///
/// # Arguments
/// * `layer` - Layer state (read-only)
/// * `q` - Full query tensor [n_head × hd]
/// * `config` - Model config for GQA mapping
/// * `normalize` - Whether to divide by denominator
/// * `eps` - Epsilon for normalization
/// * `attn_out` - Output buffer [n_head × hd]
/// * `tmp_u` - Pre-allocated temp [hd] (reused across heads)
#[allow(clippy::too_many_arguments)]
pub fn hla_layer_readout(
    layer: &HlaLayerState,
    q: &[f32],
    config: &crate::types::Config,
    normalize: bool,
    eps: f32,
    attn_out: &mut [f32],
    tmp_u: &mut [f32],
) {
    let hd = config.head_dim;
    let n_kv = config.n_kv_head;

    for h in 0..config.n_head {
        let kv_group = h * n_kv / config.n_head;
        let q_slice = &q[h * hd..(h + 1) * hd];
        let out_slice = &mut attn_out[h * hd..(h + 1) * hd];

        hla_readout(
            q_slice,
            &layer.sk[kv_group],
            &layer.heads[h],
            hd,
            out_slice,
            tmp_u,
        );

        if normalize {
            let denom = hla_denom(
                q_slice,
                &layer.sk[kv_group],
                &layer.heads[h],
                hd,
                eps,
                tmp_u,
            );
            if denom.abs() > 1e-8 {
                for x in out_slice.iter_mut() {
                    *x /= denom;
                }
            }
        }
    }
}

/// Per-Q-head update + readout for AHLA (no shared state mutation).
///
/// Reads shared PKV and mK (already updated for this KV group), updates
/// per-head E and n, then computes readout.
#[allow(clippy::too_many_arguments)]
#[inline]
fn ahla_per_head_step(
    pkv: &[f32],
    mk: &[f32],
    q_head: &mut AhlaQHeadState,
    q: &[f32],
    k: &[f32],
    hd: usize,
    gamma: f32,
    out: &mut [f32],
    tmp_r: &mut [f32],
) {
    // Step 1: Decay per-head state
    if gamma < 1.0 {
        crate::simd::simd_scale_inplace(&mut q_head.e, gamma);
        crate::simd::simd_scale_inplace(&mut q_head.n, gamma);
    }

    // TODO(T6): transpose matvec — see hla_readout comment
    tmp_r[..hd].fill(0.0);
    for i in 0..hd {
        let qi = unsafe { *q.get_unchecked(i) };
        let pkv_row = &pkv[i * hd..i * hd + hd];
        for j in 0..hd {
            unsafe {
                *tmp_r.get_unchecked_mut(j) += qi * *pkv_row.get_unchecked(j);
            }
        }
    }

    // Step 3: E_t += k_t · r_t
    simd::simd_outer_product_acc(&mut q_head.e, k, &tmp_r[..hd], hd, hd);

    // Step 4: q_tᵀ · mK_t → scalar (shared, already updated)
    let q_mk = simd::simd_dot_f32(q, mk, hd);

    // Step 5: n_t += k_t · (q_tᵀ · mK_t)
    for i in 0..hd {
        unsafe {
            *q_head.n.get_unchecked_mut(i) += *k.get_unchecked(i) * q_mk;
        }
    }

    // TODO(T6): transpose matvec (E^T * q with row-major E) — see hla_readout comment
    for j in 0..hd {
        let mut val = 0.0f32;
        for i in 0..hd {
            unsafe {
                val += *q.get_unchecked(i) * *q_head.e.get_unchecked(i * hd + j);
            }
        }
        unsafe {
            *out.get_unchecked_mut(j) = val;
        }
    }
}

/// Update + readout all heads in one layer for AHLA.
///
/// GQA-aware: shared PKV and mK are updated once per KV group,
/// then per-Q-head state (E, n) is updated separately.
///
/// # Arguments
/// * `layer` - AHLA layer state
/// * `q` - Full query tensor [n_head × hd]
/// * `k` - Full key tensor [n_kv_head × hd]
/// * `v` - Full value tensor [n_kv_head × hd]
/// * `config` - Model config for GQA mapping
/// * `gamma` - Exponential decay
/// * `normalize` - Whether to divide by denominator
/// * `eps` - Epsilon for normalization
/// * `attn_out` - Output buffer [n_head × hd]
/// * `tmp_r` - Pre-allocated temp [hd] (reused across heads)
#[allow(clippy::too_many_arguments)]
pub fn ahla_layer_step(
    layer: &mut AhlaLayerState,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    config: &crate::types::Config,
    gamma: f32,
    normalize: bool,
    eps: f32,
    attn_out: &mut [f32],
    tmp_r: &mut [f32],
) {
    let hd = config.head_dim;
    let n_kv = config.n_kv_head;

    // Phase 1: Update shared PKV, mK once per KV group (avoid double-accumulation with GQA)
    for g in 0..n_kv {
        // Decay
        if gamma < 1.0 {
            crate::simd::simd_scale_inplace(&mut layer.pkv[g], gamma);
            crate::simd::simd_scale_inplace(&mut layer.mk[g], gamma);
        }
        // PKV[g] += k_g · v_gᵀ
        let k_slice = &k[g * hd..(g + 1) * hd];
        let v_slice = &v[g * hd..(g + 1) * hd];
        simd::simd_outer_product_acc(&mut layer.pkv[g], k_slice, v_slice, hd, hd);
        // mK[g] += k_g
        for i in 0..hd {
            unsafe {
                *layer.mk[g].get_unchecked_mut(i) += *k_slice.get_unchecked(i);
            }
        }
    }

    // Phase 2: Per-Q-head update + readout
    for h in 0..config.n_head {
        let kv_group = h * n_kv / config.n_head;
        let q_slice = &q[h * hd..(h + 1) * hd];
        let k_slice = &k[kv_group * hd..(kv_group + 1) * hd];
        let out_slice = &mut attn_out[h * hd..(h + 1) * hd];

        ahla_per_head_step(
            &layer.pkv[kv_group],
            &layer.mk[kv_group],
            &mut layer.heads[h],
            q_slice,
            k_slice,
            hd,
            gamma,
            out_slice,
            tmp_r,
        );

        if normalize {
            let denom = ahla_denom(q_slice, &layer.heads[h], hd, eps);
            if denom.abs() > 1e-8 {
                for x in out_slice.iter_mut() {
                    *x /= denom;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify update ordering: cross-terms use OLD state.
    /// If we update CQV before computing G, the result would be wrong.
    #[test]
    fn symmetric_update_order_correctness() {
        let hd = 4;
        let mut sk = vec![0.0; hd * hd];
        let mut q_head = crate::hla::types::HlaQHeadState::new(hd);
        let mut tmp_k_cqv = vec![0.0; hd];
        let mut tmp_q_g = vec![0.0; hd];

        let q = [1.0, 0.0, 0.0, 0.0];
        let k = [0.0, 1.0, 0.0, 0.0];
        let v = [0.0, 0.0, 1.0, 0.0];

        // First token: G should be zero (no previous CQV)
        hla_state_update(
            &mut sk,
            &mut q_head,
            &q,
            &k,
            &v,
            hd,
            1.0,
            &mut tmp_k_cqv,
            &mut tmp_q_g,
        );

        // G should be zero because CQV was zero before this update
        assert!(
            q_head.g.iter().all(|&x| x == 0.0),
            "G should be zero on first token (no previous CQV)"
        );

        // After first update: CQV[0,2] = q[0]*v[2] = 1.0
        assert_eq!(q_head.cqv[2], 1.0);

        // Second token: different q, k, v
        let q2 = [0.0, 1.0, 0.0, 0.0];
        let k2 = [1.0, 0.0, 0.0, 0.0];
        let v2 = [0.0, 0.0, 0.0, 1.0];

        hla_state_update(
            &mut sk,
            &mut q_head,
            &q2,
            &k2,
            &v2,
            hd,
            1.0,
            &mut tmp_k_cqv,
            &mut tmp_q_g,
        );

        // G should now have k2 · (k2ᵀ · CQV_old)
        // k2ᵀ · CQV_old: row 0 of CQV = [0,0,1,0], dot with k2=[1,0,0,0] → [0,0,1,0]
        // Wait: k2ᵀ · CQV = Σ_i k2[i]*CQV[i,:], k2=[1,0,0,0] → CQV[0,:] = [0,0,1,0]
        // G += k2 · [0,0,1,0] = [1,0,0,0]ᵀ · [0,0,1,0]
        // G[0,0]=0, G[0,1]=0, G[0,2]=1, G[0,3]=0
        assert_eq!(q_head.g[2], 1.0, "G should capture k2 · (k2ᵀ·CQV_old)");
        assert_eq!(q_head.g[0], 0.0, "G[0,0] should be 0");
    }

    /// Verify symmetric readout: o = qᵀ(SK·CQV − G)
    #[test]
    fn symmetric_readout_basic() {
        let hd = 2;
        let mut sk = vec![0.0; hd * hd];
        let mut q_head = crate::hla::types::HlaQHeadState::new(hd);
        let mut tmp_k_cqv = vec![0.0; hd];
        let mut tmp_q_g = vec![0.0; hd];

        // Manually set state for predictable readout
        // SK = [[1, 0], [0, 1]] (identity)
        sk[0] = 1.0;
        sk[3] = 1.0;
        // CQV = [[2, 0], [0, 3]]
        q_head.cqv[0] = 2.0;
        q_head.cqv[3] = 3.0;
        // G = 0 (no causal correction)

        let q = [1.0, 1.0];
        let mut out = vec![0.0; hd];
        let mut tmp_u = vec![0.0; hd];

        hla_readout(&q, &sk, &q_head, hd, &mut out, &mut tmp_u);

        // u = qᵀ·SK = [1,1]
        // out[j] = u·CQV[:,j] = [1*2+1*0, 1*0+1*3] = [2, 3]
        assert!(
            (out[0] - 2.0).abs() < 1e-6,
            "out[0] should be 2.0, got {}",
            out[0]
        );
        assert!(
            (out[1] - 3.0).abs() < 1e-6,
            "out[1] should be 3.0, got {}",
            out[1]
        );
    }

    /// Verify AHLA step produces correct output.
    #[test]
    fn ahla_step_basic() {
        let hd = 2;
        let mut pkv = vec![0.0; hd * hd];
        let mut mk = vec![0.0; hd];
        let mut q_head = crate::hla::types::AhlaQHeadState::new(hd);
        let mut out = vec![0.0; hd];
        let mut tmp_r = vec![0.0; hd];

        let q = [1.0, 0.0];
        let k = [0.0, 1.0];
        let v = [2.0, 0.0];

        ahla_step(
            &mut pkv,
            &mut mk,
            &mut q_head,
            &q,
            &k,
            &v,
            hd,
            1.0,
            &mut out,
            &mut tmp_r,
        );

        // After first token: E should be zero (no previous PKV contribution to r)
        // Actually: PKV += k·vᵀ = [0,1]ᵀ·[2,0] = [[0,0],[2,0]]
        // r = qᵀ·PKV = [1,0]·[[0,0],[2,0]] = [0,0]  (row 0 of PKV = [0,0])
        // Wait: r[j] = Σ_i q[i]*PKV[i*hd+j], q=[1,0] → r[j] = PKV[0*hd+j] = PKV[j]
        // PKV = [[0,0],[2,0]] → PKV[0]=0, PKV[1]=0 → r = [0,0]
        // E += k·r = [0,1]ᵀ·[0,0] = 0
        // o = qᵀ·E = 0
        assert!(
            out.iter().all(|&x| x == 0.0),
            "First token output should be zero: {out:?}"
        );

        // Second token
        let q2 = [1.0, 0.0];
        let k2 = [0.0, 1.0];
        let v2 = [3.0, 0.0];

        ahla_step(
            &mut pkv,
            &mut mk,
            &mut q_head,
            &q2,
            &k2,
            &v2,
            hd,
            1.0,
            &mut out,
            &mut tmp_r,
        );

        // PKV now = [[0,0],[5,0]]
        // r = q2ᵀ·PKV = [1,0]·[[0,0],[5,0]] = PKV[0:2] = [0,0]
        // Still zero because PKV row 0 is zero
        // Let me check with a different config where PKV row 0 has values
    }

    /// Verify AHLA with values that produce non-zero output.
    #[test]
    fn ahla_step_nonzero_output() {
        let hd = 2;
        let mut pkv = vec![0.0; hd * hd];
        let mut mk = vec![0.0; hd];
        let mut q_head = crate::hla::types::AhlaQHeadState::new(hd);
        let mut out = vec![0.0; hd];
        let mut tmp_r = vec![0.0; hd];

        // Token 1: q hits PKV row that k fills
        let q = [1.0, 0.0];
        let k = [1.0, 0.0];
        let v = [5.0, 0.0];

        ahla_step(
            &mut pkv,
            &mut mk,
            &mut q_head,
            &q,
            &k,
            &v,
            hd,
            1.0,
            &mut out,
            &mut tmp_r,
        );

        // PKV += k·vᵀ = [1,0]ᵀ·[5,0] = [[5,0],[0,0]]
        // r = qᵀ·PKV = [1,0]·[[5,0],[0,0]] = [5,0]
        // E += k·r = [1,0]ᵀ·[5,0] = [[5,0],[0,0]]
        // o = qᵀ·E = [1,0]·[[5,0],[0,0]] = [5,0]
        assert!(
            (out[0] - 5.0).abs() < 1e-5,
            "out[0] should be 5.0, got {}",
            out[0]
        );
        assert!(
            (out[1]).abs() < 1e-5,
            "out[1] should be 0.0, got {}",
            out[1]
        );
    }

    /// Verify decay reduces state magnitudes.
    #[test]
    fn symmetric_decay_works() {
        let hd = 2;
        let mut sk = vec![0.0; hd * hd];
        let mut q_head = crate::hla::types::HlaQHeadState::new(hd);
        let mut tmp_k_cqv = vec![0.0; hd];
        let mut tmp_q_g = vec![0.0; hd];

        let q = [1.0, 1.0];
        let k = [1.0, 1.0];
        let v = [1.0, 1.0];

        // Update with gamma=1.0
        hla_state_update(
            &mut sk,
            &mut q_head,
            &q,
            &k,
            &v,
            hd,
            1.0,
            &mut tmp_k_cqv,
            &mut tmp_q_g,
        );
        let sk_no_decay = sk[0];

        // Reset and update with gamma=0.5
        sk.fill(0.0);
        q_head.reset();
        hla_state_update(
            &mut sk,
            &mut q_head,
            &q,
            &k,
            &v,
            hd,
            0.5,
            &mut tmp_k_cqv,
            &mut tmp_q_g,
        );
        // With gamma=0.5, state is decayed before adding. Since initial state is 0,
        // first update is same as no decay. Let's do a second update.
        let sk_after_first = sk[0];

        hla_state_update(
            &mut sk,
            &mut q_head,
            &q,
            &k,
            &v,
            hd,
            0.5,
            &mut tmp_k_cqv,
            &mut tmp_q_g,
        );
        let sk_after_second = sk[0];

        // sk[0] should be: 0.5 * sk_after_first + 1.0 (kkᵀ[0,0] = 1)
        assert!(
            (sk_after_second - (0.5 * sk_after_first + 1.0)).abs() < 1e-5,
            "Decay should reduce old state: {sk_after_second} vs {}",
            0.5 * sk_after_first + 1.0
        );
    }

    /// Verify normalization produces finite values.
    #[test]
    fn symmetric_normalization_finite() {
        let hd = 4;
        let mut sk = vec![0.0; hd * hd];
        let mut q_head = crate::hla::types::HlaQHeadState::new(hd);
        let mut tmp_k_cqv = vec![0.0; hd];
        let mut tmp_q_g = vec![0.0; hd];
        let mut tmp_u = vec![0.0; hd];
        let mut out = vec![0.0; hd];

        // Several updates with non-zero state
        for t in 0..10 {
            let q: Vec<f32> = (0..hd).map(|i| (t * hd + i) as f32 * 0.1).collect();
            let k: Vec<f32> = (0..hd).map(|i| (t * hd + i + 1) as f32 * 0.1).collect();
            let v: Vec<f32> = (0..hd).map(|i| (t * hd + i + 2) as f32 * 0.1).collect();

            hla_state_update(
                &mut sk,
                &mut q_head,
                &q,
                &k,
                &v,
                hd,
                1.0,
                &mut tmp_k_cqv,
                &mut tmp_q_g,
            );
        }

        let q_test = vec![0.5; hd];
        hla_readout(&q_test, &sk, &q_head, hd, &mut out, &mut tmp_u);

        let denom = hla_denom(&q_test, &sk, &q_head, hd, 1e-6, &mut tmp_u);

        assert!(denom.is_finite(), "Denominator should be finite: {denom}");
        assert!(
            denom > 0.0,
            "Denominator should be positive with ε: {denom}"
        );

        for &x in &out {
            assert!(x.is_finite(), "Output should be finite: {x}");
        }
    }
}
