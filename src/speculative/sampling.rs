use crate::types::Rng;

/// CDF-based sampling from a probability distribution.
pub fn sample_from_distribution(probs: &[f32], rng: &mut Rng) -> usize {
    let r = rng.uniform();
    let mut cdf = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        cdf += p;
        if r <= cdf {
            return i;
        }
    }
    probs.len().saturating_sub(1)
}

/// Residual distribution sampling into pre-allocated scratch buffer (zero-alloc).
///
/// `p'(x) = normalize(max(0, p(x) - q(x)))`
///
/// Samples from tokens the target model likes *more* than the draft model.
/// Falls back to `sample_from_distribution(p)` if distributions are identical.
///
/// `scratch` must be `>= p.len()`. Written to but contents not meaningful after return.
pub fn sample_residual_distribution_into(
    p: &[f32],
    q: &[f32],
    scratch: &mut [f32],
    rng: &mut Rng,
) -> usize {
    let len = p.len().min(scratch.len());
    for i in 0..len {
        scratch[i] = (p[i] - q[i]).max(0.0);
    }

    let sum: f32 = scratch[..len].iter().sum();

    if sum > 0.0 {
        let inv_sum = 1.0 / sum;
        for val in &mut scratch[..len] {
            *val *= inv_sum;
        }
        sample_from_distribution(&scratch[..len], rng)
    } else {
        // Distributions identical — fallback to target distribution
        sample_from_distribution(p, rng)
    }
}

/// Residual distribution sampling (Equation 3 from Leviathan et al. 2022).
///
/// Allocating wrapper around `sample_residual_distribution_into`.
/// Prefer `_into` variant with `SpeculativeContext::residual_buf` for hot paths.
pub fn sample_residual_distribution(p: &[f32], q: &[f32], rng: &mut Rng) -> usize {
    let mut scratch = vec![0.0f32; p.len()];
    sample_residual_distribution_into(p, q, &mut scratch, rng)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Rng;

    #[test]
    fn test_sample_from_distribution() {
        let mut rng = Rng::new(42);
        let probs = vec![0.1, 0.2, 0.5, 0.2];
        for _ in 0..100 {
            let t = sample_from_distribution(&probs, &mut rng);
            assert!(t < 4, "token should be 0-3, got {t}");
        }
    }

    #[test]
    fn test_sample_from_distribution_degenerate() {
        let mut rng = Rng::new(42);
        let probs = vec![0.0, 0.0, 1.0, 0.0];
        for _ in 0..50 {
            let t = sample_from_distribution(&probs, &mut rng);
            assert_eq!(t, 2, "should always sample token 2");
        }
    }

    #[test]
    fn test_residual_distribution_sums_to_one() {
        let mut rng = Rng::new(42);
        let p = vec![0.3, 0.5, 0.1, 0.1];
        let q = vec![0.1, 0.6, 0.2, 0.1];
        // residual = [0.2, 0.0, 0.0, 0.0] → normalized [1.0, 0.0, 0.0, 0.0]
        for _ in 0..50 {
            let token = sample_residual_distribution(&p, &q, &mut rng);
            assert_eq!(token, 0, "residual should only pick token 0");
        }
    }

    #[test]
    fn test_residual_distribution_fallback_on_identical() {
        let mut rng = Rng::new(42);
        let p = vec![0.25, 0.25, 0.25, 0.25];
        let q = vec![0.25, 0.25, 0.25, 0.25];
        let token = sample_residual_distribution(&p, &q, &mut rng);
        assert!(token < 4, "token should be valid, got {token}");
    }

    #[test]
    fn test_residual_distribution_multiple_positive() {
        let mut rng = Rng::new(42);
        let p = vec![0.5, 0.1, 0.3, 0.1];
        let q = vec![0.1, 0.5, 0.1, 0.3];
        // residual = [0.4, 0.0, 0.2, 0.0] → normalized [0.667, 0.0, 0.333, 0.0]
        let mut counts = [0usize; 4];
        for _ in 0..1000 {
            let token = sample_residual_distribution(&p, &q, &mut rng);
            counts[token] += 1;
        }
        assert!(counts[0] > counts[2], "token 0 should be more frequent");
        assert_eq!(counts[1], 0, "token 1 should never be picked");
        assert_eq!(counts[3], 0, "token 3 should never be picked");
    }
}
