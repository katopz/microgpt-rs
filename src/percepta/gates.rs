//! ReGLU, stepglu, multiply, persist gate primitives for Percepta computation graph.
//!
//! These gates are the fundamental building blocks for Percepta's transformer-vm.
//! Each gate maps to specific transformer weight patterns:
//!
//! - [`reglu`] — `relu(b) × a` → single FFN neuron
//! - [`stepglu`] — `a × step(b ≥ 0)` → Heaviside step via 2 ReGLU + persist
//! - [`multiply`] — `a × b` → full signed multiplication via 2 ReGLU + persist
//! - [`PersistSlot`] — materialize expression into a dedicated residual slot
//!
//! # Gate Neuron Counts
//!
//! | Gate | ReGLU Dims | Persist Dims | FFN Neurons |
//! |------|-----------|-------------|-------------|
//! | `reglu` | 1 | 0 | 1 |
//! | `stepglu` | 2 | 1 | 3 |
//! | `multiply` | 2 | 1 | 3 |
//!
//! Distilled from Percepta's `transformer-vm` (Apache-2.0 © Percepta).
//! Reference: `.raw/transformer-vm/transformer_vm/graph/core.py` (L244-290)

// ── Gate Primitives ─────────────────────────────────────────────

/// Gate primitive: `reglu(a, b) = relu(b) × a`.
///
/// This is the fundamental FFN gate. When `b ≥ 0`, returns `a × b`.
/// When `b < 0`, returns `0.0` (gated off).
///
/// Maps to a single ReGLU FFN neuron: `W_gate·x → relu → × (W_value·x)`.
///
/// # Examples
///
/// ```
/// use microgpt_rs::percepta::gates::reglu;
/// assert_eq!(reglu(3.0, 2.0), 6.0);   // a × b when b ≥ 0
/// assert_eq!(reglu(3.0, -1.0), 0.0);  // gated off when b < 0
/// assert_eq!(reglu(3.0, 0.0), 0.0);   // relu(0) = 0
/// ```
#[inline]
pub fn reglu(a: f64, b: f64) -> f64 {
    a * b.max(0.0)
}

/// Gate primitive: `stepglu(a, b) = a × step(b ≥ 0)`.
///
/// Heaviside step function via two ReGLU gates and a persist:
/// ```text
/// stepglu(a, b) = reglu(a, b + 1) - reglu(a, b)
/// ```
///
/// For **integer** `b`:
/// - `b ≥ 0` → returns `a` (gate passes through)
/// - `b < 0`  → returns `0.0` (gate blocks)
///
/// The step function enables conditional logic in the transformer:
/// "if condition ≥ 0, pass value; else zero".
///
/// # Examples
///
/// ```
/// use microgpt_rs::percepta::gates::stepglu;
/// // Integer b ≥ 0: passes a through
/// assert_eq!(stepglu(5.0, 0.0), 5.0);
/// assert_eq!(stepglu(5.0, 3.0), 5.0);
/// // Integer b < 0: blocks
/// assert_eq!(stepglu(5.0, -1.0), 0.0);
/// assert_eq!(stepglu(5.0, -5.0), 0.0);
/// ```
#[inline]
pub fn stepglu(a: f64, b: f64) -> f64 {
    reglu(a, b + 1.0) - reglu(a, b)
}

/// Gate primitive: `multiply(a, b) = a × b` (full signed multiplication).
///
/// Full signed multiplication via two ReGLU gates and a persist:
/// ```text
/// multiply(a, b) = reglu(a, b) - reglu(a, -b)
/// ```
///
/// This works because `relu(b) - relu(-b) = b` for all `b`:
/// - `b ≥ 0`: `relu(b) - relu(-b) = b - 0 = b` → `a × b`
/// - `b < 0`:  `relu(b) - relu(-b) = 0 - (-b) = b` → `a × b`
///
/// # Examples
///
/// ```
/// use microgpt_rs::percepta::gates::multiply;
/// assert_eq!(multiply(3.0, 4.0), 12.0);    // positive × positive
/// assert_eq!(multiply(3.0, -2.0), -6.0);   // positive × negative
/// assert_eq!(multiply(-3.0, 4.0), -12.0);  // negative × positive
/// assert_eq!(multiply(-3.0, -2.0), 6.0);   // negative × negative
/// assert_eq!(multiply(3.0, 0.0), 0.0);     // zero
/// ```
#[inline]
pub fn multiply(a: f64, b: f64) -> f64 {
    reglu(a, b) - reglu(a, -b)
}

// ── Persist Slot ────────────────────────────────────────────────

/// A named residual slot for persisting gate outputs.
///
/// In the transformer, `persist(expr)` materializes a linear expression
/// into a dedicated dimension, reducing `d_model` by allowing the
/// expression's constituent dimensions to be freed earlier.
///
/// At the primitive level, persist is identity — the actual slot
/// allocation happens during graph scheduling (TG-D).
///
/// Treated as a schedulable gate (own phase) for pathwidth/NL purposes.
#[derive(Clone, Debug, PartialEq)]
pub struct PersistSlot {
    /// Slot name for diagnostic output.
    pub name: String,
    /// The persisted value.
    pub value: f64,
}

impl PersistSlot {
    /// Create a new persist slot with the given name and value.
    pub fn new(name: impl Into<String>, value: f64) -> Self {
        Self {
            name: name.into(),
            value,
        }
    }

    /// Materialize a value into a named persist slot.
    ///
    /// This is the gate-level primitive. At the graph level (TG-C),
    /// `persist` operates on `Expression` values and creates
    /// `PersistDimension` nodes.
    pub fn persist(name: impl Into<String>, value: f64) -> Self {
        Self::new(name, value)
    }
}

// ── Gate Kind ───────────────────────────────────────────────────

/// Enum of gate types for computation graph node classification.
///
/// Each variant records the gate's inputs for later weight construction.
#[derive(Clone, Debug, PartialEq)]
pub enum GateKind {
    /// `relu(b) × a` — single ReGLU neuron.
    Reglu { a: f64, b: f64 },
    /// `a × step(b ≥ 0)` — Heaviside step (2 ReGLU + 1 persist).
    Stepglu { a: f64, b: f64 },
    /// `a × b` — full signed multiplication (2 ReGLU + 1 persist).
    Multiply { a: f64, b: f64 },
    /// Materialize value into a named residual slot.
    Persist { name: String, value: f64 },
}

impl GateKind {
    /// Evaluate this gate and return its output value.
    pub fn evaluate(&self) -> f64 {
        match self {
            Self::Reglu { a, b } => reglu(*a, *b),
            Self::Stepglu { a, b } => stepglu(*a, *b),
            Self::Multiply { a, b } => multiply(*a, *b),
            Self::Persist { value, .. } => *value,
        }
    }

    /// Number of FFN neurons this gate requires.
    ///
    /// Each ReGLU dimension maps to one FFN neuron.
    /// Persist dimensions map to one FFN output neuron each.
    pub fn neuron_count(&self) -> usize {
        match self {
            Self::Reglu { .. } => 1,
            // stepglu = 2 ReGLU + 1 persist = 3 neurons
            Self::Stepglu { .. } => 3,
            // multiply = 2 ReGLU + 1 persist = 3 neurons
            Self::Multiply { .. } => 3,
            // persist = 1 output neuron
            Self::Persist { .. } => 1,
        }
    }

    /// Number of ReGLU dimensions this gate creates.
    pub fn reglu_dim_count(&self) -> usize {
        match self {
            Self::Reglu { .. } => 1,
            Self::Stepglu { .. } => 2,
            Self::Multiply { .. } => 2,
            Self::Persist { .. } => 0,
        }
    }

    /// Number of persist dimensions this gate creates.
    pub fn persist_dim_count(&self) -> usize {
        match self {
            Self::Reglu { .. } => 0,
            Self::Stepglu { .. } => 1,
            Self::Multiply { .. } => 1,
            Self::Persist { .. } => 1,
        }
    }
}

impl std::fmt::Display for GateKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Reglu { a, b } => write!(f, "reglu({a}, {b}) = {}", self.evaluate()),
            Self::Stepglu { a, b } => write!(f, "stepglu({a}, {b}) = {}", self.evaluate()),
            Self::Multiply { a, b } => write!(f, "multiply({a}, {b}) = {}", self.evaluate()),
            Self::Persist { name, value } => write!(f, "persist({name}) = {value}"),
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use crate::percepta::types::EPS;

    use super::*;

    // ── B5: Unit tests — each gate produces correct output ──────

    #[test]
    fn test_reglu_positive_b() {
        // relu(b) = b when b > 0, so reglu(a, b) = a * b
        assert_eq!(reglu(3.0, 2.0), 6.0);
        assert_eq!(reglu(1.0, 5.0), 5.0);
        assert_eq!(reglu(-2.0, 3.0), -6.0);
    }

    #[test]
    fn test_reglu_negative_b() {
        // relu(b) = 0 when b < 0, so reglu(a, b) = 0
        assert_eq!(reglu(3.0, -1.0), 0.0);
        assert_eq!(reglu(100.0, -0.001), 0.0);
        assert_eq!(reglu(-5.0, -3.0), 0.0);
    }

    #[test]
    fn test_reglu_zero_b() {
        // relu(0) = 0
        assert_eq!(reglu(3.0, 0.0), 0.0);
        assert_eq!(reglu(0.0, 0.0), 0.0);
    }

    #[test]
    fn test_reglu_zero_a() {
        // 0 * relu(b) = 0 for any b
        assert_eq!(reglu(0.0, 5.0), 0.0);
        assert_eq!(reglu(0.0, -3.0), 0.0);
    }

    #[test]
    fn test_stepglu_integer_nonneg_b() {
        // For integer b >= 0: stepglu(a, b) = a
        assert_eq!(stepglu(5.0, 0.0), 5.0);
        assert_eq!(stepglu(5.0, 1.0), 5.0);
        assert_eq!(stepglu(5.0, 10.0), 5.0);
        assert_eq!(stepglu(-3.0, 2.0), -3.0);
    }

    #[test]
    fn test_stepglu_integer_neg_b() {
        // For integer b < 0: stepglu(a, b) = 0
        assert_eq!(stepglu(5.0, -1.0), 0.0);
        assert_eq!(stepglu(5.0, -2.0), 0.0);
        assert_eq!(stepglu(5.0, -10.0), 0.0);
    }

    #[test]
    fn test_stepglu_fractional_b() {
        // For fractional b, stepglu is not exact step function
        // but follows the reglu(a, b+1) - reglu(a, b) formula
        // b = 0.5: relu(1.5) - relu(0.5) = 1.5 - 0.5 = 1.0 → a × 1.0
        assert!((stepglu(5.0, 0.5) - 5.0).abs() < EPS);
        // b = -0.5: relu(0.5) - relu(-0.5) = 0.5 - 0 = 0.5 → a × 0.5
        assert!((stepglu(5.0, -0.5) - 2.5).abs() < EPS);
    }

    #[test]
    fn test_multiply_positive_positive() {
        assert_eq!(multiply(3.0, 4.0), 12.0);
        assert_eq!(multiply(1.0, 1.0), 1.0);
    }

    #[test]
    fn test_multiply_positive_negative() {
        assert_eq!(multiply(3.0, -2.0), -6.0);
        assert_eq!(multiply(1.0, -5.0), -5.0);
    }

    #[test]
    fn test_multiply_negative_positive() {
        assert_eq!(multiply(-3.0, 4.0), -12.0);
        assert_eq!(multiply(-2.0, 3.0), -6.0);
    }

    #[test]
    fn test_multiply_negative_negative() {
        assert_eq!(multiply(-3.0, -2.0), 6.0);
        assert_eq!(multiply(-1.0, -1.0), 1.0);
    }

    #[test]
    fn test_multiply_zero() {
        assert_eq!(multiply(3.0, 0.0), 0.0);
        assert_eq!(multiply(0.0, 5.0), 0.0);
        assert_eq!(multiply(0.0, 0.0), 0.0);
    }

    #[test]
    fn test_multiply_matches_native() {
        // Compare against native multiplication for a range of values
        let test_cases = [
            (2.5, 3.7),
            (-1.5, 4.2),
            (0.0, 100.0),
            (-3.0, -7.0),
            (0.001, 1000.0),
            (1e10, 1e-10),
            (-1e10, -1e-10),
        ];
        for (a, b) in test_cases {
            let expected = a * b;
            let actual = multiply(a, b);
            assert!(
                (actual - expected).abs() < expected.abs() * EPS + EPS,
                "multiply({a}, {b}) = {actual}, expected {expected}"
            );
        }
    }

    // ── PersistSlot tests ───────────────────────────────────────

    #[test]
    fn test_persist_slot_new() {
        let slot = PersistSlot::new("test_slot", 42.0);
        assert_eq!(slot.name, "test_slot");
        assert_eq!(slot.value, 42.0);
    }

    #[test]
    fn test_persist_slot_persist() {
        let slot = PersistSlot::persist("result", multiply(3.0, 4.0));
        assert_eq!(slot.name, "result");
        assert_eq!(slot.value, 12.0);
    }

    // ── GateKind tests ──────────────────────────────────────────

    #[test]
    fn test_gate_kind_evaluate() {
        let reglu_gate = GateKind::Reglu { a: 3.0, b: 2.0 };
        assert_eq!(reglu_gate.evaluate(), 6.0);

        let stepglu_gate = GateKind::Stepglu { a: 5.0, b: 0.0 };
        assert_eq!(stepglu_gate.evaluate(), 5.0);

        let mult_gate = GateKind::Multiply { a: 3.0, b: -2.0 };
        assert_eq!(mult_gate.evaluate(), -6.0);

        let persist_gate = GateKind::Persist {
            name: "x".to_string(),
            value: 42.0,
        };
        assert_eq!(persist_gate.evaluate(), 42.0);
    }

    #[test]
    fn test_gate_kind_neuron_counts() {
        assert_eq!(GateKind::Reglu { a: 0.0, b: 0.0 }.neuron_count(), 1);
        assert_eq!(GateKind::Stepglu { a: 0.0, b: 0.0 }.neuron_count(), 3);
        assert_eq!(GateKind::Multiply { a: 0.0, b: 0.0 }.neuron_count(), 3);
        assert_eq!(
            GateKind::Persist {
                name: String::new(),
                value: 0.0
            }
            .neuron_count(),
            1
        );
    }

    #[test]
    fn test_gate_kind_dim_counts() {
        let reglu = GateKind::Reglu { a: 0.0, b: 0.0 };
        assert_eq!(reglu.reglu_dim_count(), 1);
        assert_eq!(reglu.persist_dim_count(), 0);

        let stepglu = GateKind::Stepglu { a: 0.0, b: 0.0 };
        assert_eq!(stepglu.reglu_dim_count(), 2);
        assert_eq!(stepglu.persist_dim_count(), 1);

        let mult = GateKind::Multiply { a: 0.0, b: 0.0 };
        assert_eq!(mult.reglu_dim_count(), 2);
        assert_eq!(mult.persist_dim_count(), 1);

        let persist = GateKind::Persist {
            name: String::new(),
            value: 0.0,
        };
        assert_eq!(persist.reglu_dim_count(), 0);
        assert_eq!(persist.persist_dim_count(), 1);
    }

    // ── B6: Integration test — gates compose into conditional logic ─

    #[test]
    fn test_conditional_if_else_pattern() {
        // Build: if (b >= 0) { a } else { c }
        // Using stepglu: result = stepglu(a, b) + stepglu(c, -b - 1)
        // When b >= 0: stepglu(a, b) = a, stepglu(c, -b-1) = 0 → result = a
        // When b < 0:  stepglu(a, b) = 0, stepglu(c, -b-1) = c → result = c

        let a = 10.0;
        let c = 20.0;

        // Case 1: b >= 0 (condition true → pick a)
        let b_true = 1.0;
        let result_true = stepglu(a, b_true) + stepglu(c, -b_true - 1.0);
        assert_eq!(result_true, a, "b={b_true}: should pick a={a}");

        // Case 2: b < 0 (condition false → pick c)
        let b_false = -1.0;
        let result_false = stepglu(a, b_false) + stepglu(c, -b_false - 1.0);
        assert_eq!(result_false, c, "b={b_false}: should pick c={c}");

        // Case 3: b = 0 (boundary → pick a)
        let b_zero = 0.0;
        let result_zero = stepglu(a, b_zero) + stepglu(c, -b_zero - 1.0);
        assert_eq!(result_zero, a, "b={b_zero}: boundary should pick a={a}");
    }

    #[test]
    fn test_abs_via_multiply() {
        // |a| = sqrt(a²) = sqrt(multiply(a, a))
        // But more useful: sign extraction via stepglu
        // sign(a) = stepglu(1, a) - stepglu(1, -a) (for integer a != 0)
        let a = 5.0;
        let sign = stepglu(1.0, a) - stepglu(1.0, -a);
        assert_eq!(sign, 1.0, "sign of {a} should be 1");

        let a = -3.0;
        let sign = stepglu(1.0, a) - stepglu(1.0, -a);
        assert_eq!(sign, -1.0, "sign of {a} should be -1");
    }

    #[test]
    fn test_max_via_gates() {
        // max(a, b) = reglu(1.0, a - b) + b = relu(a-b) * 1.0 + b
        // = (a - b).max(0.0) + b: when a-b >= 0 → a-b+b = a; when a-b < 0 → 0+b = b
        let a = 7.0;
        let b = 3.0;
        let max_val = reglu(1.0, a - b) + b;
        assert_eq!(max_val, a, "max({a}, {b}) should be {a}");

        let a = 2.0;
        let b = 5.0;
        let max_val = reglu(1.0, a - b) + b;
        assert_eq!(max_val, b, "max({a}, {b}) should be {b}");
    }

    #[test]
    fn test_gate_composition_saturating_add() {
        // Saturating add: result = min(a + b, max_val)
        // Using: result = (a + b) - reglu(1.0, a + b - max_val)
        // reglu(1.0, x) = relu(x) * 1.0 = max(0, x)
        // When a+b <= max: relu(a+b-max) = 0, result = a+b
        // When a+b > max:  relu(a+b-max) = a+b-max, result = max
        let max_val = 10.0;

        let a = 3.0;
        let b = 4.0;
        let sum = a + b;
        let sat = sum - reglu(1.0, sum - max_val);
        assert_eq!(sat, 7.0, "3+4 should not saturate at 10");

        let a = 8.0;
        let b = 5.0;
        let sum = a + b;
        let sat = sum - reglu(1.0, sum - max_val);
        assert_eq!(sat, max_val, "8+5 should saturate at {max_val}");
    }
}
