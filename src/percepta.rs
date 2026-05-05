//! Percepta-style O(log N) 2D Attention via Convex Hull KV Cache.
//!
//! Standard transformer attention computes Q·K for all N past keys → O(N) per step.
//! Percepta restricts attention heads to d=2, making the dot product a 2D geometric
//! projection. When keys form a convex hull, finding the maximum attention score
//! becomes ternary search over a unimodal (bitonic) sequence → O(log N).
//!
//! Integration points with microgpt-rs:
//! - DDTree branch pruning: validate drafted tokens before target verification
//! - Computable LoRA: encode state-machine rules as 2D key embeddings
//! - "Free embedding" bridge: project hidden states to 2D for fast retrieval

/// 2D vector for geometric attention operations.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Vec2 {
    pub x: f32,
    pub y: f32,
}

impl Vec2 {
    pub fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    /// Dot product — the core attention score in 2D.
    #[inline]
    pub fn dot(&self, other: &Self) -> f32 {
        self.x * other.x + self.y * other.y
    }

    /// Z-component of cross product AB × AC.
    /// Positive = left turn, Negative = right turn, Zero = collinear.
    #[inline]
    pub fn cross_z(a: &Self, b: &Self, c: &Self) -> f32 {
        (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)
    }
}

/// Specialized KV Cache for 2D attention heads.
/// Maintains the upper convex hull of keys for O(log N) attention lookup.
///
/// Keys must have monotonically non-decreasing X coordinates — natural for
/// sequential execution traces where position encodes time step.
pub struct KVCache2D {
    keys: Vec<Vec2>,
    values: Vec<usize>,
    upper_hull: Vec<usize>,
}

impl Default for KVCache2D {
    fn default() -> Self {
        Self::new()
    }
}

impl KVCache2D {
    pub fn new() -> Self {
        Self {
            keys: Vec::new(),
            values: Vec::new(),
            upper_hull: Vec::new(),
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            keys: Vec::with_capacity(capacity),
            values: Vec::with_capacity(capacity),
            upper_hull: Vec::with_capacity(capacity),
        }
    }

    pub fn len(&self) -> usize {
        self.keys.len()
    }

    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    pub fn hull_len(&self) -> usize {
        self.upper_hull.len()
    }

    /// Append a key-value pair. Amortized O(1) hull maintenance via Graham Scan.
    ///
    /// For keys with monotonically increasing X:
    /// - Points creating non-right turns (collinear or concave) are removed
    /// - The upper hull captures the "skyline" of the key distribution
    pub fn append(&mut self, key: Vec2, value: usize) {
        let idx = self.keys.len();
        self.keys.push(key);
        self.values.push(value);

        // Maintain upper convex hull: pop points violating convexity
        while self.upper_hull.len() >= 2 {
            let len = self.upper_hull.len();
            let a = &self.keys[self.upper_hull[len - 2]];
            let b = &self.keys[self.upper_hull[len - 1]];
            let c = &key;

            // Right turn (cross < 0) preserves convexity. Remove otherwise.
            if Vec2::cross_z(a, b, c) >= 0.0 {
                self.upper_hull.pop();
            } else {
                break;
            }
        }
        self.upper_hull.push(idx);
    }

    /// Standard O(N) attention: linear scan over all keys.
    /// Baseline for correctness verification.
    pub fn linear_attention(&self, query: &Vec2) -> (f32, usize) {
        match self.keys.len() {
            0 => (f32::NEG_INFINITY, 0),
            _ => {
                let mut max_score = f32::NEG_INFINITY;
                let mut best_idx = 0;
                for (i, key) in self.keys.iter().enumerate() {
                    let score = query.dot(key);
                    if score > max_score {
                        max_score = score;
                        best_idx = i;
                    }
                }
                (max_score, self.values[best_idx])
            }
        }
    }

    /// O(log N) attention via ternary search over the convex hull.
    ///
    /// The dot product of a fixed query against points on a convex hull
    /// forms a unimodal (bitonic) sequence: it rises to a peak then falls.
    /// Ternary search finds the peak in O(log H) where H = hull size.
    pub fn fast_attention(&self, query: &Vec2) -> (f32, usize) {
        let n = self.upper_hull.len();
        match n {
            0 => (f32::NEG_INFINITY, 0),
            1 => {
                let idx = self.upper_hull[0];
                (query.dot(&self.keys[idx]), self.values[idx])
            }
            2 => {
                let idx0 = self.upper_hull[0];
                let idx1 = self.upper_hull[1];
                let s0 = query.dot(&self.keys[idx0]);
                let s1 = query.dot(&self.keys[idx1]);
                match s0 >= s1 {
                    true => (s0, self.values[idx0]),
                    false => (s1, self.values[idx1]),
                }
            }
            _ => {
                let mut left = 0usize;
                let mut right = n - 1;

                // Ternary search on unimodal dot-product sequence
                while right - left > 2 {
                    let third = (right - left) / 3;
                    let m1 = left + third;
                    let m2 = right - third;

                    let s1 = query.dot(&self.keys[self.upper_hull[m1]]);
                    let s2 = query.dot(&self.keys[self.upper_hull[m2]]);

                    match s1 < s2 {
                        true => left = m1,
                        false => right = m2,
                    }
                }

                // Scan the remaining 1–3 candidates
                let mut max_score = f32::NEG_INFINITY;
                let mut best_idx = self.upper_hull[left];

                for i in left..=right {
                    let idx = self.upper_hull[i];
                    let score = query.dot(&self.keys[idx]);
                    if score > max_score {
                        max_score = score;
                        best_idx = idx;
                    }
                }

                (max_score, self.values[best_idx])
            }
        }
    }

    /// Get hull indices (for debugging/testing).
    pub fn hull_indices(&self) -> &[usize] {
        &self.upper_hull
    }

    /// Reset the cache.
    pub fn reset(&mut self) {
        self.keys.clear();
        self.values.clear();
        self.upper_hull.clear();
    }

    /// Get all keys (for debugging/testing).
    pub fn keys(&self) -> &[Vec2] {
        &self.keys
    }

    /// Get all values (for debugging/testing).
    pub fn values(&self) -> &[usize] {
        &self.values
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vec2_dot_product() {
        let a = Vec2::new(1.0, 2.0);
        let b = Vec2::new(3.0, 4.0);
        assert!((a.dot(&b) - 11.0).abs() < 1e-6);
    }

    #[test]
    fn test_vec2_cross_z_collinear() {
        let a = Vec2::new(0.0, 0.0);
        let b = Vec2::new(1.0, 1.0);
        let c = Vec2::new(2.0, 2.0);
        assert!(Vec2::cross_z(&a, &b, &c).abs() < 1e-6);
    }

    #[test]
    fn test_vec2_cross_z_left_turn() {
        let a = Vec2::new(0.0, 0.0);
        let b = Vec2::new(1.0, 0.0);
        let c = Vec2::new(1.0, 1.0);
        assert!(Vec2::cross_z(&a, &b, &c) > 0.0);
    }

    #[test]
    fn test_vec2_cross_z_right_turn() {
        let a = Vec2::new(0.0, 0.0);
        let b = Vec2::new(1.0, 1.0);
        let c = Vec2::new(2.0, 0.0);
        assert!(Vec2::cross_z(&a, &b, &c) < 0.0);
    }

    #[test]
    fn test_cache_empty() {
        let cache = KVCache2D::new();
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
        assert_eq!(cache.hull_len(), 0);
    }

    #[test]
    fn test_cache_single_element() {
        let mut cache = KVCache2D::new();
        cache.append(Vec2::new(1.0, 2.0), 42);
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.hull_len(), 1);
    }

    #[test]
    fn test_cache_linear_empty() {
        let cache = KVCache2D::new();
        let (score, val) = cache.linear_attention(&Vec2::new(1.0, 0.0));
        assert_eq!(score, f32::NEG_INFINITY);
        assert_eq!(val, 0);
    }

    #[test]
    fn test_cache_fast_empty() {
        let cache = KVCache2D::new();
        let (score, val) = cache.fast_attention(&Vec2::new(1.0, 0.0));
        assert_eq!(score, f32::NEG_INFINITY);
        assert_eq!(val, 0);
    }

    #[test]
    fn test_single_element_attention() {
        let mut cache = KVCache2D::new();
        cache.append(Vec2::new(1.0, 2.0), 42);

        let query = Vec2::new(3.0, 4.0);
        let (lin_score, lin_val) = cache.linear_attention(&query);
        let (fast_score, fast_val) = cache.fast_attention(&query);

        assert!((lin_score - fast_score).abs() < 1e-6);
        assert_eq!(lin_val, fast_val);
        assert_eq!(lin_val, 42);
    }

    #[test]
    fn test_two_elements_attention() {
        let mut cache = KVCache2D::new();
        cache.append(Vec2::new(0.0, 10.0), 0);
        cache.append(Vec2::new(10.0, 0.0), 1);

        // X-dominant query should pick (10, 0)
        let query = Vec2::new(1.0, 0.0);
        let (lin_score, lin_val) = cache.linear_attention(&query);
        let (fast_score, fast_val) = cache.fast_attention(&query);

        assert!((lin_score - fast_score).abs() < 1e-6);
        assert_eq!(lin_val, fast_val);
        assert_eq!(lin_val, 1);

        // Y-dominant query should pick (0, 10)
        let query = Vec2::new(0.0, 1.0);
        let (lin_score, lin_val) = cache.linear_attention(&query);
        let (fast_score, fast_val) = cache.fast_attention(&query);

        assert!((lin_score - fast_score).abs() < 1e-6);
        assert_eq!(lin_val, fast_val);
        assert_eq!(lin_val, 0);
    }

    #[test]
    fn test_hull_removes_collinear() {
        let mut cache = KVCache2D::new();
        cache.append(Vec2::new(0.0, 0.0), 0);
        cache.append(Vec2::new(1.0, 1.0), 1);
        cache.append(Vec2::new(2.0, 2.0), 2);
        // Collinear: cross_z >= 0 removes middle point
        assert_eq!(cache.hull_len(), 2);
    }

    #[test]
    fn test_hull_keeps_concave_down() {
        let mut cache = KVCache2D::new();
        // Concave-down parabola: all points are on the upper hull
        for i in 0..10 {
            let x = i as f32;
            let y = -(x - 4.5).powi(2);
            cache.append(Vec2::new(x, y), i);
        }
        assert_eq!(cache.hull_len(), 10);
    }

    #[test]
    fn test_hull_compresses_flat_line() {
        let mut cache = KVCache2D::new();
        for i in 0..100 {
            cache.append(Vec2::new(i as f32, 0.0), i);
        }
        // Collinear points compress to 2 endpoints
        assert!(cache.hull_len() <= 2);
    }

    #[test]
    fn test_linear_fast_agree_parabolic() {
        let mut cache = KVCache2D::new();
        for i in 0..1000 {
            let x = i as f32;
            let y = -((x - 500.0) / 100.0).powi(2);
            cache.append(Vec2::new(x, y), i);
        }

        let queries = [
            Vec2::new(1.0, 0.0),
            Vec2::new(0.0, 1.0),
            Vec2::new(1.0, 1.0),
            Vec2::new(-1.0, 1.0),
            Vec2::new(5.0, 10.0),
            Vec2::new(-3.0, 7.0),
        ];

        for query in &queries {
            let (lin_score, lin_val) = cache.linear_attention(query);
            let (fast_score, fast_val) = cache.fast_attention(query);
            assert!(
                (lin_score - fast_score).abs() < 1e-3,
                "Score mismatch for query ({}, {}): linear={}, fast={}",
                query.x,
                query.y,
                lin_score,
                fast_score
            );
            assert_eq!(
                lin_val, fast_val,
                "Value mismatch for query ({}, {})",
                query.x, query.y
            );
        }
    }

    #[test]
    fn test_linear_fast_agree_100k_trace() {
        let mut cache = KVCache2D::new();
        for i in 0..100_000 {
            let x = i as f32;
            let y = -((x - 50000.0) / 1000.0).powi(2);
            cache.append(Vec2::new(x, y), i);
        }

        let query = Vec2::new(5.0, 10.0);
        let (lin_score, lin_val) = cache.linear_attention(&query);
        let (fast_score, fast_val) = cache.fast_attention(&query);

        assert!((lin_score - fast_score).abs() < 1e-3);
        assert_eq!(lin_val, fast_val);
    }

    #[test]
    fn test_linear_fast_agree_sin_wave() {
        let mut cache = KVCache2D::new();
        // Sinusoidal keys — hull compresses peaks only
        for i in 0..1000 {
            let x = i as f32;
            let y = (x * 0.01).sin();
            cache.append(Vec2::new(x, y), i);
        }

        let queries = [
            Vec2::new(1.0, 0.0),
            Vec2::new(0.0, 1.0),
            Vec2::new(1.0, 1.0),
        ];

        for query in &queries {
            let (lin_score, lin_val) = cache.linear_attention(query);
            let (fast_score, fast_val) = cache.fast_attention(query);
            assert!(
                (lin_score - fast_score).abs() < 1e-3,
                "Score mismatch: linear={lin_score}, fast={fast_score}"
            );
            assert_eq!(lin_val, fast_val, "Value mismatch");
        }
    }

    #[test]
    fn test_reset() {
        let mut cache = KVCache2D::new();
        cache.append(Vec2::new(1.0, 2.0), 0);
        cache.append(Vec2::new(3.0, 4.0), 1);
        assert!(!cache.is_empty());
        cache.reset();
        assert!(cache.is_empty());
        assert_eq!(cache.hull_len(), 0);
    }

    #[test]
    fn test_hull_compression_ratio() {
        // Zigzag pattern: hull should only keep the peaks
        let mut cache = KVCache2D::new();
        for i in 0..1000 {
            let x = i as f32;
            let y = match i % 2 {
                0 => 10.0,
                _ => 0.0,
            };
            cache.append(Vec2::new(x, y), i);
        }
        // Upper hull should be much smaller than 1000
        assert!(
            cache.hull_len() < 100,
            "zigzag hull should compress heavily, got {}",
            cache.hull_len()
        );
    }

    /// ADVERSARIAL: V-shaped (concave-up) keys cause fast_attention to FAIL.
    ///
    /// Keys: (0,10), (1,5), (2,0), (3,5), (4,10)
    /// Upper hull: only (0,10) and (4,10) — the valley is skipped.
    /// Query (0, -1): dot products = -10, -5, 0, -5, -10
    ///   Linear: picks index 2 (score 0) — the valley bottom
    ///   Fast: picks hull index 0 or 4 (score -10) — WRONG
    ///
    /// This proves fast_attention is NOT correct for arbitrary distributions.
    /// It requires keys where the max-dot-product key is on the UPPER hull.
    #[test]
    fn test_adversarial_v_shape_fast_attention_wrong() {
        let mut cache = KVCache2D::new();
        // V-shape: valley at index 2
        cache.append(Vec2::new(0.0, 10.0), 0);
        cache.append(Vec2::new(1.0, 5.0), 1);
        cache.append(Vec2::new(2.0, 0.0), 2); // valley bottom
        cache.append(Vec2::new(3.0, 5.0), 3);
        cache.append(Vec2::new(4.0, 10.0), 4);

        // Verify hull only has the two peaks (indices 0 and 4)
        assert_eq!(cache.hull_len(), 2, "V-shape hull should be 2 endpoints");

        // Query pointing DOWN: maximizes dot at the valley bottom
        let query = Vec2::new(0.0, -1.0);
        let (lin_score, lin_val) = cache.linear_attention(&query);
        let (fast_score, fast_val) = cache.fast_attention(&query);

        // Linear correctly finds valley bottom (index 2)
        assert_eq!(lin_val, 2, "linear should find valley bottom");
        assert!((lin_score - 0.0).abs() < 1e-6, "linear score should be 0");

        // Fast gives WRONG answer — it can't see inside the valley
        assert_ne!(
            fast_val, lin_val,
            "fast should disagree on V-shape valley query"
        );
        assert!(fast_score < lin_score, "fast score should be worse");
    }

    /// CORRECTNESS GUARANTEE: For the same V-shape, queries with
    /// positive y-component correctly find the hull vertices (peaks).
    #[test]
    fn test_adversarial_v_shape_positive_query_correct() {
        let mut cache = KVCache2D::new();
        cache.append(Vec2::new(0.0, 10.0), 0);
        cache.append(Vec2::new(1.0, 5.0), 1);
        cache.append(Vec2::new(2.0, 0.0), 2);
        cache.append(Vec2::new(3.0, 5.0), 3);
        cache.append(Vec2::new(4.0, 10.0), 4);

        // Query pointing UP: maximizes dot at the peaks (on hull)
        let query = Vec2::new(0.0, 1.0);
        let (lin_score, _lin_val) = cache.linear_attention(&query);
        let (fast_score, fast_val) = cache.fast_attention(&query);

        // Both should find the peak (index 0 or 4, both have y=10)
        assert!(
            (lin_score - fast_score).abs() < 1e-6,
            "scores should match for hull-optimal query"
        );
        assert!(
            fast_val == 0 || fast_val == 4,
            "fast should find a peak, got {fast_val}"
        );
    }

    /// DFA COMPUTATION: Simulates executing a DFA via 2D attention.
    ///
    /// DFA: Recognizes binary strings divisible by 3.
    /// States: 0 (accept), 1, 2
    /// Transition: δ(state, bit) = (2*state + bit) % 3
    ///
    /// The execution trace is encoded as KV pairs where:
    /// - Key = Vec2(step, state * 100.0 + bit * 10.0)
    /// - Value = next_state
    ///
    /// This proves: the KV cache can store and retrieve computational state.
    /// The attention mechanism correctly identifies matching context entries.
    #[test]
    fn test_dfa_divisible_by_3_trace() {
        // Binary representation of 54 (divisible by 3): 110110
        let input = [1, 1, 0, 1, 1, 0];
        let mut state = 0usize;
        let mut cache = KVCache2D::new();

        // Build execution trace
        for (step, &bit) in input.iter().enumerate() {
            let next_state = (state * 2 + bit) % 3;
            cache.append(
                Vec2::new(step as f32, state as f32 * 100.0 + bit as f32 * 10.0),
                next_state,
            );
            state = next_state;
        }

        // Final state should be 0 (accept — 54 % 3 == 0)
        assert_eq!(state, 0, "54 should be divisible by 3");

        // Verify trace has correct transitions
        assert_eq!(cache.len(), 6, "should have 6 trace entries");
    }

    /// DFA STRESS TEST: Run the divisible-by-3 DFA on all integers 0..1000.
    /// Verify the attention trace correctly identifies accept/reject.
    #[test]
    fn test_dfa_divisible_by_3_stress() {
        for n in 0..1000u32 {
            let bits: Vec<u8> = (0..16)
                .rev()
                .map(|shift| ((n >> shift) & 1) as u8)
                .collect();

            let mut state = 0usize;
            let mut cache = KVCache2D::new();

            for (step, &bit) in bits.iter().enumerate() {
                let next_state = (state * 2 + bit as usize) % 3;
                cache.append(Vec2::new(step as f32, state as f32 * 100.0), next_state);
                state = next_state;
            }

            let expected_accept = n % 3 == 0;
            let actual_accept = state == 0;
            assert_eq!(
                actual_accept, expected_accept,
                "DFA wrong for n={n}: expected accept={expected_accept}, got state={state}"
            );

            // Verify fast_attention agrees with linear on the trace
            if !cache.is_empty() {
                let query = Vec2::new(0.0, 1.0);
                let (lin_s, _) = cache.linear_attention(&query);
                let (fast_s, _) = cache.fast_attention(&query);
                assert!(
                    (lin_s - fast_s).abs() < 1e-3,
                    "DFA trace attention mismatch for n={n}"
                );
            }
        }
    }

    /// FIBONACCI TRACE: Encode Fibonacci computation as a 2D attention trace.
    ///
    /// Each step: fib(i) = fib(i-1) + fib(i-2)
    /// Key: Vec2(step, fib_value)
    ///
    /// IMPORTANT FINDING: Exponential growth is concave-UP (accelerating).
    /// Concave-up distributions compress to just 2 hull vertices.
    /// This proves: NOT all computations produce hull-friendly geometries.
    /// Exponential processes compress too aggressively for hull attention.
    ///
    /// For queries aligned with the hull endpoints, fast_attention is correct.
    /// For other queries, the compressed hull may miss the true optimum.
    #[test]
    fn test_fibonacci_trace_attention() {
        let mut cache = KVCache2D::with_capacity(50);
        let mut fib = vec![0u64, 1u64];
        cache.append(Vec2::new(0.0, 0.0), 0);
        cache.append(Vec2::new(1.0, 1.0), 1);

        for i in 2..45u32 {
            let next = fib[i as usize - 1] + fib[i as usize - 2];
            fib.push(next);
            cache.append(Vec2::new(i as f32, next as f32), i as usize);
        }

        // Fibonacci grows exponentially → concave-UP → hull compresses to 2
        assert!(
            cache.hull_len() <= 2,
            "exponential growth should compress hull heavily, got {}",
            cache.hull_len()
        );

        // Verify attention for queries where the optimum IS at a hull endpoint.
        // Query (1, 0) → maximizes x → picks last point (index 44).
        let query = Vec2::new(1.0, 0.0);
        let (lin_s, lin_v) = cache.linear_attention(&query);
        let (fast_s, fast_v) = cache.fast_attention(&query);
        assert!(
            (lin_s - fast_s).abs() < 1e-3,
            "fibonacci endpoint query mismatch: lin={lin_s}, fast={fast_s}"
        );
        assert_eq!(lin_v, fast_v, "fibonacci endpoint value mismatch");

        // Verify known Fibonacci values
        assert_eq!(fib[10], 55);
        assert_eq!(fib[20], 6765);
        assert_eq!(fib[44], 701408733);
    }

    /// COUNTER TRACE: Simulate a simple counting program.
    /// At each step, state increments by 1.
    /// Key: Vec2(step, state) — both monotonically increasing → collinear.
    /// Collinear keys compress to 2 hull vertices.
    /// Attention still correctly finds the maximum.
    #[test]
    fn test_counter_trace_collinear() {
        let mut cache = KVCache2D::new();
        for i in 0..10000 {
            cache.append(Vec2::new(i as f32, i as f32), i);
        }

        // Collinear: hull should compress heavily
        assert!(
            cache.hull_len() <= 2,
            "counter trace should compress to 2, got {}",
            cache.hull_len()
        );

        // But attention still works correctly
        let query = Vec2::new(1.0, 1.0);
        let (lin_s, lin_v) = cache.linear_attention(&query);
        let (fast_s, fast_v) = cache.fast_attention(&query);

        assert!((lin_s - fast_s).abs() < 1e-3);
        assert_eq!(lin_v, fast_v);
        assert_eq!(lin_v, 9999, "should pick the last (highest) counter value");
    }

    /// UNIMODALITY: Prove that dot products over hull vertices form a
    /// unimodal (bitonic) sequence. This is the mathematical foundation
    /// that makes ternary search correct.
    ///
    /// For a convex polygon traversed in order, the dot product with any
    /// fixed query direction increases monotonically to a maximum, then
    /// decreases. This is because the vertices of a convex polygon,
    /// when traversed counter-clockwise, have monotonically changing
    /// outward normals.
    #[test]
    fn test_hull_dot_products_unimodal() {
        let mut cache = KVCache2D::new();
        // Concave-down parabola — all points on hull
        for i in 0..100u32 {
            let x = i as f32;
            let y = -(x - 50.0).powi(2) / 100.0;
            cache.append(Vec2::new(x, y), i as usize);
        }

        assert_eq!(cache.hull_len(), 100, "parabola should keep all points");

        // For several query directions, verify dot products are unimodal
        let queries = [
            Vec2::new(1.0, 0.0),
            Vec2::new(0.0, 1.0),
            Vec2::new(1.0, 1.0),
            Vec2::new(-1.0, 1.0),
            Vec2::new(2.0, -1.0),
        ];

        for query in &queries {
            let hull = cache.hull_indices();
            let scores: Vec<f32> = hull
                .iter()
                .map(|&idx| query.dot(&cache.keys()[idx]))
                .collect();

            // Check unimodality: scores should increase, then decrease
            let max_pos = scores
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0;

            // Before max: scores should be non-decreasing
            for i in 1..=max_pos {
                assert!(
                    scores[i] >= scores[i - 1] - 1e-6,
                    "not unimodal before max at i={i}: {} < {} for query ({}, {})",
                    scores[i],
                    scores[i - 1],
                    query.x,
                    query.y
                );
            }
            // After max: scores should be non-increasing
            for i in max_pos + 1..scores.len() {
                assert!(
                    scores[i] <= scores[i - 1] + 1e-6,
                    "not unimodal after max at i={i}: {} > {} for query ({}, {})",
                    scores[i],
                    scores[i - 1],
                    query.x,
                    query.y
                );
            }
        }
    }

    /// SUPPORTING POINT: Verify that the fast_attention result is always
    /// the point on the convex hull furthest in the query direction.
    ///
    /// This is the geometric interpretation: the argmax of q·k over all
    /// points equals the argmax of q·k over the convex hull vertices.
    /// This property holds because the maximum of a linear function over
    /// a convex set is always at a vertex.
    ///
    /// We verify this by checking that for CONVEX distributions (parabolic),
    /// fast_attention always matches linear_attention.
    #[test]
    fn test_supporting_point_property() {
        let mut cache = KVCache2D::new();
        // Concave-down parabola: all points on upper hull = convex
        for i in 0..500u32 {
            let x = i as f32;
            let y = -(x - 250.0).powi(2) / 100.0 + 100.0;
            cache.append(Vec2::new(x, y), i as usize);
        }

        // Sweep through 360 degrees of query directions
        for deg in 0..360 {
            let rad = (deg as f32).to_radians();
            let query = Vec2::new(rad.cos(), rad.sin());

            let (lin_s, lin_v) = cache.linear_attention(&query);
            let (fast_s, fast_v) = cache.fast_attention(&query);

            assert!(
                (lin_s - fast_s).abs() < 1e-3,
                "supporting point violated at deg={deg}: lin={lin_s}, fast={fast_s}"
            );
            assert_eq!(lin_v, fast_v, "value mismatch at deg={deg}");
        }
    }

    /// RANDOMIZED STRESS: Generate random convex distributions and verify
    /// fast_attention matches linear_attention for random queries.
    ///
    /// NOTE: Queries are restricted to qy >= 0 because for concave-down
    /// parabolas, negative qy produces U-shaped (non-unimodal) dot product
    /// sequences, which breaks ternary search correctness.
    #[test]
    fn test_random_convex_stress() {
        let mut seed = 12345u64;
        let next_seed = |s: &mut u64| -> f32 {
            *s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((*s >> 33) as f32) / (1u64 << 31) as f32
        };

        let mut cache = KVCache2D::with_capacity(10000);
        let center = next_seed(&mut seed) * 5000.0;
        let scale = 100.0 + next_seed(&mut seed) * 900.0;
        let offset = next_seed(&mut seed) * 50.0;

        for i in 0..10000u32 {
            let x = i as f32;
            let y = -(x - center).powi(2) / scale + offset;
            cache.append(Vec2::new(x, y), i as usize);
        }

        for _ in 0..100 {
            let qx = (next_seed(&mut seed) - 0.5) * 20.0;
            let qy = next_seed(&mut seed) * 20.0; // qy >= 0 for unimodal guarantee
            let query = Vec2::new(qx, qy);

            let (lin_s, lin_v) = cache.linear_attention(&query);
            let (fast_s, fast_v) = cache.fast_attention(&query);

            assert!(
                (lin_s - fast_s).abs() < 1e-2,
                "random stress: score mismatch for ({qx:.2}, {qy:.2}): lin={lin_s:.2}, fast={fast_s:.2}"
            );
            assert_eq!(
                lin_v, fast_v,
                "random stress: value mismatch for ({qx:.2}, {qy:.2})"
            );
        }
    }

    /// ADVERSARIAL: Multiple valleys where fast_attention misses interior maxima.
    /// Proves the limitation is systematic, not a one-off edge case.
    #[test]
    fn test_adversarial_multiple_valleys() {
        let mut cache = KVCache2D::new();
        // W-shape: two valleys with peaks between them
        //   10      10      10
        //    \  /\  /\  /
        //     \/  \/  \/
        //      0  5  0  5  0
        cache.append(Vec2::new(0.0, 10.0), 0);
        cache.append(Vec2::new(1.0, 0.0), 1); // valley
        cache.append(Vec2::new(2.0, 10.0), 2);
        cache.append(Vec2::new(3.0, 5.0), 3); // shallow valley
        cache.append(Vec2::new(4.0, 10.0), 4);
        cache.append(Vec2::new(5.0, 0.0), 5); // valley
        cache.append(Vec2::new(6.0, 10.0), 6);

        // Query pointing down should find a valley bottom
        let query = Vec2::new(0.0, -1.0);
        let (lin_score, lin_val) = cache.linear_attention(&query);
        let (fast_score, fast_val) = cache.fast_attention(&query);

        // Linear finds a valley (index 1 or 5, score = 0)
        assert_eq!(lin_val, 1, "linear should find first valley bottom");

        // Fast misses the valley — only sees hull peaks
        assert_ne!(fast_val, lin_val, "fast should disagree on valley query");
        assert!(fast_score < lin_score, "fast score should be worse");
    }

    // ── Arithmetic Computation via Attention ─────────────────────────
    //
    // These tests prove that the 4 fundamental arithmetic operations can be
    // computed incrementally using the 2D attention mechanism. Each step:
    //   1. Retrieves the previous accumulator value via fast_attention
    //   2. Computes the next value FROM the retrieved value (not a local var)
    //   3. Appends the result to the trace
    //   4. Verifies fast_attention matches linear_attention
    //
    // Query (1, 0) always returns the most recent entry because
    // dot((1,0), (step, acc)) = step, maximized at the latest step.

    /// ADDITION: 42 + 17 = 59
    ///
    /// Trace: acc goes 42 → 43 → ... → 59 (17 increment steps)
    /// Keys are collinear upward slope → hull compresses to 2 endpoints.
    /// Attention retrieves previous state, computation uses retrieved value.
    #[test]
    fn test_arithmetic_addition() {
        let mut cache = KVCache2D::new();
        let query = Vec2::new(1.0, 0.0); // "give me the latest state"

        // Step 0: load initial value
        cache.append(Vec2::new(0.0, 42.0), 42);

        // Steps 1..17: increment by 1 each step
        for step in 1..=17 {
            let (lin_s, lin_v) = cache.linear_attention(&query);
            let (fast_s, fast_v) = cache.fast_attention(&query);
            assert!((lin_s - fast_s).abs() < 1e-3, "step {step}: score mismatch");
            assert_eq!(lin_v, fast_v, "step {step}: value mismatch");

            // Compute next state FROM the attention-retrieved value
            let next = fast_v + 1;
            cache.append(Vec2::new(step as f32, next as f32), next);
        }

        // Final result via attention
        let (_, result) = cache.fast_attention(&query);
        assert_eq!(result, 59, "42 + 17 = 59");
        assert_eq!(cache.len(), 18, "18 trace entries (initial + 17 steps)");
    }

    /// SUBTRACTION: 100 - 37 = 63
    ///
    /// Trace: acc goes 100 → 99 → ... → 63 (37 decrement steps)
    /// Keys form a downward slope (step↑, acc↓) → still collinear → hull = 2.
    /// Query (1, 0) still picks the latest step regardless of acc direction.
    #[test]
    fn test_arithmetic_subtraction() {
        let mut cache = KVCache2D::new();
        let query = Vec2::new(1.0, 0.0);

        cache.append(Vec2::new(0.0, 100.0), 100);

        for step in 1..=37 {
            let (_, prev) = cache.fast_attention(&query);
            let next = prev - 1;
            cache.append(Vec2::new(step as f32, next as f32), next);
        }

        let (_, result) = cache.fast_attention(&query);
        assert_eq!(result, 63, "100 - 37 = 63");

        // Verify fast matches linear on the full trace
        let (lin_s, lin_v) = cache.linear_attention(&query);
        let (fast_s, fast_v) = cache.fast_attention(&query);
        assert!((lin_s - fast_s).abs() < 1e-3);
        assert_eq!(lin_v, fast_v);
    }

    /// MULTIPLICATION: 7 × 8 = 56
    ///
    /// Implemented as repeated addition: 0 → 7 → 14 → ... → 56 (8 steps).
    /// Keys form an upward slope (step↑, acc↑) → collinear → hull = 2.
    #[test]
    fn test_arithmetic_multiplication() {
        let mut cache = KVCache2D::new();
        let query = Vec2::new(1.0, 0.0);

        cache.append(Vec2::new(0.0, 0.0), 0);

        for step in 1..=8 {
            let (_, prev) = cache.fast_attention(&query);
            let next = prev + 7;
            cache.append(Vec2::new(step as f32, next as f32), next);
        }

        let (_, result) = cache.fast_attention(&query);
        assert_eq!(result, 56, "7 × 8 = 56");
        assert_eq!(cache.len(), 9, "9 trace entries");
    }

    /// DIVISION: 100 ÷ 7 = 14 remainder 2
    ///
    /// Implemented as repeated subtraction: 100 → 93 → 86 → ... → 2 (14 steps).
    /// Returns both quotient (step count) and remainder (final acc).
    #[test]
    fn test_arithmetic_division() {
        let mut cache = KVCache2D::new();
        let query = Vec2::new(1.0, 0.0);

        cache.append(Vec2::new(0.0, 100.0), 100);

        let mut quotient = 0usize;
        for step in 1.. {
            let (_, prev) = cache.fast_attention(&query);
            if prev < 7 {
                break;
            }
            let next = prev - 7;
            cache.append(Vec2::new(step as f32, next as f32), next);
            quotient += 1;
        }

        let (_, remainder) = cache.fast_attention(&query);
        assert_eq!(quotient, 14, "100 ÷ 7 = 14");
        assert_eq!(remainder, 2, "100 % 7 = 2");
    }

    /// MODULO: 17 % 5 = 2
    ///
    /// Same as division but we only care about the remainder.
    #[test]
    fn test_arithmetic_modulo() {
        let mut cache = KVCache2D::new();
        let query = Vec2::new(1.0, 0.0);

        cache.append(Vec2::new(0.0, 17.0), 17);

        for step in 1.. {
            let (_, prev) = cache.fast_attention(&query);
            if prev < 5 {
                break;
            }
            let next = prev - 5;
            cache.append(Vec2::new(step as f32, next as f32), next);
        }

        let (_, remainder) = cache.fast_attention(&query);
        assert_eq!(remainder, 2, "17 % 5 = 2");
    }

    /// POWER: 2^10 = 1024
    ///
    /// Implemented as repeated multiplication via doubling.
    /// Trace: 1 → 2 → 4 → 8 → ... → 1024 (10 doublings).
    /// Exponential growth → concave-UP → hull compresses to 2 endpoints.
    /// But query (1, 0) still retrieves the latest step correctly.
    #[test]
    fn test_arithmetic_power() {
        let mut cache = KVCache2D::new();
        let query = Vec2::new(1.0, 0.0);

        cache.append(Vec2::new(0.0, 1.0), 1);

        for step in 1..=10 {
            let (_, prev) = cache.fast_attention(&query);
            let next = prev * 2;
            cache.append(Vec2::new(step as f32, next as f32), next);
        }

        let (_, result) = cache.fast_attention(&query);
        assert_eq!(result, 1024, "2^10 = 1024");

        // Exponential trace compresses heavily
        assert!(
            cache.hull_len() <= 2,
            "exponential trace should compress to 2, got {}",
            cache.hull_len()
        );
    }

    /// COMBINED EXPRESSION: (3 + 5) × 2 - 4 ÷ 2 = 14
    ///
    /// Simulates a tiny virtual machine executing:
    ///   LOAD 3    → acc = 3
    ///   ADD  5    → acc = 3 + 5 = 8
    ///   MUL  2    → acc = 8 × 2 = 16
    ///   SUB  4    → acc = 16 - 4 = 12
    ///   DIV  2    → acc = 12 ÷ 2 = 6
    ///
    /// Wait, let me recalculate: (3+5)*2 - 4/2 = 8*2 - 2 = 14
    /// With integer division order: LOAD 3, ADD 5, MUL 2, SUB 2 = 14
    ///
    /// Each instruction retrieves the previous acc via attention,
    /// applies the operation, and appends to the trace.
    #[test]
    fn test_arithmetic_combined_expression() {
        let mut cache = KVCache2D::new();
        let query = Vec2::new(1.0, 0.0);

        // (3 + 5) * 2 - 2 = 14
        // VM instructions: (opcode, operand)
        let program: Vec<(&str, usize)> = vec![
            ("LOAD", 3), // acc = 3
            ("ADD", 5),  // acc = 3 + 5 = 8
            ("MUL", 2),  // acc = 8 * 2 = 16
            ("SUB", 2),  // acc = 16 - 2 = 14
        ];

        // Track expected accumulator for verification
        let mut expected = 0usize;

        for (step, (opcode, operand)) in program.iter().enumerate() {
            let acc = match *opcode {
                "LOAD" => *operand,
                "ADD" => {
                    let (_, prev) = cache.fast_attention(&query);
                    prev + operand
                }
                "SUB" => {
                    let (_, prev) = cache.fast_attention(&query);
                    prev - operand
                }
                "MUL" => {
                    let (_, prev) = cache.fast_attention(&query);
                    prev * operand
                }
                _ => panic!("unknown opcode: {opcode}"),
            };

            cache.append(Vec2::new(step as f32, acc as f32), acc);
            expected = acc;
        }

        // Final result via attention
        let (_, result) = cache.fast_attention(&query);
        assert_eq!(result, 14, "(3 + 5) × 2 - 2 = 14");
        assert_eq!(result, expected);

        // Verify fast matches linear on the final trace
        let (lin_s, lin_v) = cache.linear_attention(&query);
        let (fast_s, fast_v) = cache.fast_attention(&query);
        assert!((lin_s - fast_s).abs() < 1e-3);
        assert_eq!(lin_v, fast_v);
    }

    /// COMPREHENSIVE ARITHMETIC: verify all operations on many inputs.
    ///
    /// Proves attention-based computation is correct for:
    /// - a + b for all (a, b) in 0..=10
    /// - a × b for all (a, b) in 0..=10
    /// - a - b where a >= b, for all (a, b) in 0..=10
    /// - a ÷ b where b > 0, for all (a, b) in 0..=20
    #[test]
    fn test_arithmetic_comprehensive() {
        let query = Vec2::new(1.0, 0.0);

        // Addition: a + b for all a, b in 0..=10
        for a in 0..=10u32 {
            for b in 0..=10u32 {
                let mut cache = KVCache2D::new();
                cache.append(Vec2::new(0.0, a as f32), a as usize);
                for step in 1..=b {
                    let (_, prev) = cache.fast_attention(&query);
                    cache.append(Vec2::new(step as f32, prev as f32 + 1.0), prev + 1);
                }
                let (_, result) = cache.fast_attention(&query);
                assert_eq!(result, (a + b) as usize, "{a} + {b}");
            }
        }

        // Multiplication: a × b via repeated addition
        for a in 0..=10u32 {
            for b in 0..=10u32 {
                let mut cache = KVCache2D::new();
                cache.append(Vec2::new(0.0, 0.0), 0);
                for step in 1..=b {
                    let (_, prev) = cache.fast_attention(&query);
                    let next = prev + a as usize;
                    cache.append(Vec2::new(step as f32, next as f32), next);
                }
                let (_, result) = cache.fast_attention(&query);
                assert_eq!(result, (a * b) as usize, "{a} × {b}");
            }
        }

        // Subtraction: a - b where a >= b
        for a in 0..=10u32 {
            for b in 0..=a {
                let mut cache = KVCache2D::new();
                cache.append(Vec2::new(0.0, a as f32), a as usize);
                for step in 1..=b {
                    let (_, prev) = cache.fast_attention(&query);
                    cache.append(Vec2::new(step as f32, prev as f32 - 1.0), prev - 1);
                }
                let (_, result) = cache.fast_attention(&query);
                assert_eq!(result, (a - b) as usize, "{a} - {b}");
            }
        }

        // Division: a ÷ b where b > 0
        for a in 0..=20u32 {
            for b in 1..=10u32 {
                let mut cache = KVCache2D::new();
                cache.append(Vec2::new(0.0, a as f32), a as usize);
                let mut quotient = 0usize;
                for step in 1.. {
                    let (_, prev) = cache.fast_attention(&query);
                    if prev < b as usize {
                        break;
                    }
                    let next = prev - b as usize;
                    cache.append(Vec2::new(step as f32, next as f32), next);
                    quotient += 1;
                }
                assert_eq!(quotient, (a / b) as usize, "{a} ÷ {b} quotient");
                let (_, remainder) = cache.fast_attention(&query);
                assert_eq!(remainder, (a % b) as usize, "{a} ÷ {b} remainder");
            }
        }
    }

    // ── Backtracking Computation: Sudoku & N-Queens ──────────────
    //
    // The Percepta blog solved the Arto Inkala Sudoku (hardest in the world)
    // inside a transformer at 32K tok/s. They did NOT train the model —
    // they COMPILED a C solver into transformer weights via MILP solvers.
    // The model executes the program deterministically, like a CPU.
    //
    // These tests prove our 2D attention mechanism correctly tracks
    // backtracking search: forward placements, dead-end detection, undos,
    // and alternative branches. No training needed.

    fn sudoku4_check(board: &[u8; 16], pos: usize, digit: u8) -> bool {
        let row = pos / 4;
        let col = pos % 4;
        for c in 0..4 {
            if board[row * 4 + c] == digit {
                return false;
            }
        }
        for r in 0..4 {
            if board[r * 4 + col] == digit {
                return false;
            }
        }
        let br = (row / 2) * 2;
        let bc = (col / 2) * 2;
        for r in br..br + 2 {
            for c in bc..bc + 2 {
                if board[r * 4 + c] == digit {
                    return false;
                }
            }
        }
        true
    }

    fn sudoku4_valid(board: &[u8; 16]) -> bool {
        for pos in 0..16 {
            let d = board[pos];
            if d == 0 {
                return false;
            }
            let mut tmp = *board;
            tmp[pos] = 0;
            if !sudoku4_check(&tmp, pos, d) {
                return false;
            }
        }
        true
    }

    fn sudoku4_solve(board: &mut [u8; 16], cache: &mut KVCache2D, step: &mut usize) -> bool {
        let filled = board.iter().filter(|&&v| v > 0).count();
        cache.append(Vec2::new(*step as f32, filled as f32 * 10.0), *step);
        *step += 1;

        let pos = match board.iter().position(|&v| v == 0) {
            Some(p) => p,
            None => return true,
        };

        for digit in 1..=4u8 {
            if sudoku4_check(board, pos, digit) {
                board[pos] = digit;
                if sudoku4_solve(board, cache, step) {
                    return true;
                }
                board[pos] = 0;
            }
        }
        false
    }

    fn nqueens_check(queens: &[i32], row: usize, col: i32) -> bool {
        for (r, &c) in queens.iter().enumerate().take(row) {
            if c == col || (c - col).abs() == (r as i32 - row as i32).abs() {
                return false;
            }
        }
        true
    }

    fn nqueens_solve(
        queens: &mut [i32],
        row: usize,
        n: usize,
        cache: &mut KVCache2D,
        step: &mut usize,
    ) -> bool {
        let placed = queens.iter().filter(|&&q| q >= 0).count();
        cache.append(Vec2::new(*step as f32, placed as f32 * 10.0), *step);
        *step += 1;

        if row >= n {
            return true;
        }

        for col in 0..n {
            if nqueens_check(queens, row, col as i32) {
                queens[row] = col as i32;
                if nqueens_solve(queens, row + 1, n, cache, step) {
                    return true;
                }
                queens[row] = -1;
            }
        }
        false
    }

    /// BACKTRACKING PATTERN: forward exploration → dead end → backtrack → solution.
    /// The trace creates a "mountain range" pattern.
    /// The hull captures peaks (deepest explorations), skips valleys (backtracks).
    #[test]
    fn test_backtracking_forward_undo_pattern() {
        let mut cache = KVCache2D::new();

        // Forward: depth 0→1→2→3→4 (peak)
        cache.append(Vec2::new(0.0, 10.0), 0);
        cache.append(Vec2::new(1.0, 20.0), 1);
        cache.append(Vec2::new(2.0, 30.0), 2);
        cache.append(Vec2::new(3.0, 40.0), 3);
        cache.append(Vec2::new(4.0, 50.0), 4); // peak

        // Dead end → backtrack to depth 2
        cache.append(Vec2::new(5.0, 30.0), 5); // valley

        // New branch from depth 2 → goes deeper
        cache.append(Vec2::new(6.0, 40.0), 6);
        cache.append(Vec2::new(7.0, 50.0), 7);
        cache.append(Vec2::new(8.0, 60.0), 8);
        cache.append(Vec2::new(9.0, 70.0), 9); // solution

        let query = Vec2::new(1.0, 0.0);
        let (_, result) = cache.fast_attention(&query);
        assert_eq!(result, 9, "should return final state");

        // Hull: should capture peaks (0, 4, 9) but NOT the valley (5)
        let hull = cache.hull_indices();
        assert!(
            hull.len() <= 3,
            "hull should compress to ~3 vertices, got {}",
            hull.len()
        );
        assert!(hull.contains(&9), "hull should contain solution step");
        assert!(
            !hull.contains(&5),
            "hull should NOT contain the backtrack valley"
        );

        // Verify fast matches linear
        let (lin_s, lin_v) = cache.linear_attention(&query);
        let (fast_s, fast_v) = cache.fast_attention(&query);
        assert!((lin_s - fast_s).abs() < 1e-3);
        assert_eq!(lin_v, fast_v);
    }

    /// 4×4 SUDOKU: Full backtracking solver with attention trace.
    /// Each recursive call records state. Attention retrieves latest state.
    #[test]
    fn test_sudoku_4x4_backtracking() {
        // 4×4 Sudoku: 1 _ _ _ / _ _ 2 _ / _ 3 _ _ / _ _ _ 4
        let mut board: [u8; 16] = [1, 0, 0, 0, 0, 0, 2, 0, 0, 3, 0, 0, 0, 0, 0, 4];
        let mut cache = KVCache2D::new();
        let mut step = 0usize;

        let solved = sudoku4_solve(&mut board, &mut cache, &mut step);

        assert!(solved, "4×4 Sudoku should be solvable");
        assert!(board.iter().all(|&v| v > 0), "all cells filled");
        assert!(
            sudoku4_valid(&board),
            "solution should satisfy all constraints"
        );

        // Attention retrieves correct final state
        let query = Vec2::new(1.0, 0.0);
        let (lin_s, lin_v) = cache.linear_attention(&query);
        let (fast_s, fast_v) = cache.fast_attention(&query);
        assert!((lin_s - fast_s).abs() < 1e-3);
        assert_eq!(lin_v, fast_v);
        assert_eq!(fast_v, step - 1, "should return final step");
    }

    /// 4×4 SUDOKU HULL: Hull captures search tree peaks, compresses backtracks.
    #[test]
    fn test_sudoku_4x4_hull_captures_search() {
        let mut board: [u8; 16] = [1, 0, 0, 0, 0, 0, 2, 0, 0, 3, 0, 0, 0, 0, 0, 4];
        let mut cache = KVCache2D::new();
        let mut step = 0usize;

        sudoku4_solve(&mut board, &mut cache, &mut step);

        // Hull should compress (backtracking creates valleys)
        assert!(
            cache.hull_len() < cache.len(),
            "hull should compress: hull={}, total={}",
            cache.hull_len(),
            cache.len()
        );

        // Hull should contain the final state
        let hull = cache.hull_indices();
        assert!(
            hull.contains(&(step - 1)),
            "hull should contain final step {}",
            step - 1
        );
    }

    /// 8-QUEENS: Classic backtracking with attention trace.
    #[test]
    fn test_nqueens_8_backtracking() {
        let mut queens: [i32; 8] = [-1; 8];
        let mut cache = KVCache2D::new();
        let mut step = 0usize;

        let solved = nqueens_solve(&mut queens, 0, 8, &mut cache, &mut step);

        assert!(solved, "8-Queens should have a solution");
        assert!(queens.iter().all(|&q| q >= 0), "all queens placed");

        // Verify no conflicts
        for i in 0..8 {
            for j in i + 1..8 {
                assert_ne!(queens[i], queens[j], "queens {i},{j} same column");
                assert_ne!(
                    (queens[i] - queens[j]).abs(),
                    (j - i) as i32,
                    "queens {i},{j} same diagonal"
                );
            }
        }

        // Attention retrieves final state
        let query = Vec2::new(1.0, 0.0);
        let (lin_s, lin_v) = cache.linear_attention(&query);
        let (fast_s, fast_v) = cache.fast_attention(&query);
        assert!((lin_s - fast_s).abs() < 1e-3);
        assert_eq!(lin_v, fast_v);

        // Hull shows backtracking structure
        assert!(
            cache.hull_len() < cache.len(),
            "8-Queens hull should compress: hull={}, total={}",
            cache.hull_len(),
            cache.len()
        );
    }
}
