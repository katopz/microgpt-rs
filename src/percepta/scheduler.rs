//! MILP scheduler: minimize d_model for delayed-reuse erase scheduling.
//!
//! Assigns each gate (LookUp, ReGLU, Persist) to a 4-phase layer:
//!   phase 0: Attention (LookUp)
//!   phase 1: Persist1
//!   phase 2: FFN (ReGLU)
//!   phase 3: Persist2
//!
//! Minimizes `d_model = 2 * D_half` where `D_half >= max` over all boundaries of
//! the effective width (dims alive and needing a slot at each persist boundary).
//!
//! Uses `good_lp` with the `microlp` pure-Rust backend. Suitable for small-to-medium
//! programs; switch to `highs` for larger schedules.
//!
//! Distilled from Percepta's `transformer-vm` (Apache-2.0 © Percepta).
//! Reference: `.raw/transformer-vm/transformer_vm/scheduler/milp.py` (814 lines)

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};

use good_lp::{
    Expression as LpAffine, ProblemVariables, Solution, SolverModel, Variable, default_solver,
    variable,
};
use log::info;

use crate::percepta::TieBreak;
use crate::percepta::graph::*;

// ── Helper ────────────────────────────────────────────────────

/// Wrap an LP [`Variable`] as an affine [`LpAffine`] for arithmetic.
#[inline]
fn e(v: Variable) -> LpAffine {
    LpAffine::from(v)
}

// ── Operation Key ─────────────────────────────────────────────

/// Identifies a scheduled operation in the computation graph.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum OpKey {
    /// Attention-based retrieval operation.
    LookUp(LookupId),
    /// ReGLU gated FFN operation, keyed by its output dimension.
    ReGLU(DimId),
    /// Persist (materialize) operation, keyed by its output dimension.
    Persist(DimId),
}

impl OpKey {
    /// Human-readable label for the operation kind.
    pub fn kind_label(self) -> &'static str {
        match self {
            Self::LookUp(_) => "lookup",
            Self::ReGLU(_) => "reglu",
            Self::Persist(_) => "persist",
        }
    }
}

// ── Phase ─────────────────────────────────────────────────────

/// Phase within a 4-phase transformer layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    /// Phase 0: Attention (LookUp) operations.
    Attention,
    /// Phase 1: First persist boundary.
    Persist1,
    /// Phase 2: FFN (ReGLU) operations.
    Ffn,
    /// Phase 3: Second persist boundary.
    Persist2,
}

impl Phase {
    /// Phase index within a layer (0, 1, 2, or 3).
    #[must_use]
    pub const fn index(self) -> u32 {
        match self {
            Self::Attention => 0,
            Self::Persist1 => 1,
            Self::Ffn => 2,
            Self::Persist2 => 3,
        }
    }

    /// All four phases in order.
    pub const ALL: [Phase; 4] = [Self::Attention, Self::Persist1, Self::Ffn, Self::Persist2];
}

// ── Standard Layer ────────────────────────────────────────────

/// One transformer layer's scheduled operations.
#[derive(Clone, Debug, Default)]
pub struct StdLayer {
    /// LookUp operations at phase 4L (attention).
    pub attention: Vec<LookupId>,
    /// Persist dimensions at phase 4L+1 (persist1).
    pub persist1: Vec<DimId>,
    /// ReGLU dimensions at phase 4L+2 (FFN).
    pub ffn: Vec<DimId>,
    /// Persist dimensions at phase 4L+3 (persist2).
    pub persist2: Vec<DimId>,
}

// ── Dependency Graph ──────────────────────────────────────────

/// Dependency graph extracted from a [`ProgramGraph`].
///
/// Maps each operation to its dependencies, produced dimensions,
/// and identifies tight constraints (persist must be same layer as avg lookup).
#[derive(Clone, Debug)]
pub struct DepGraph {
    /// All scheduled operations.
    pub ops: Vec<OpKey>,
    /// Input dimension IDs.
    pub inputs: Vec<DimId>,
    /// Dimensions produced by each operation.
    pub produced: HashMap<OpKey, HashSet<DimId>>,
    /// Raw dimension dependencies for each operation.
    pub deps_cache: HashMap<OpKey, HashSet<DimId>>,
    /// Operation that produces each dimension.
    pub dim_to_op: HashMap<DimId, OpKey>,
    /// Operation dependencies (edges in the DAG).
    pub op_deps: HashMap<OpKey, HashSet<OpKey>>,
    /// Children (reverse dependency edges).
    pub children: HashMap<OpKey, HashSet<OpKey>>,
    /// Dimensions consumed by each operation.
    pub consumers: HashMap<DimId, HashSet<OpKey>>,
    /// Tight constraints: op → set of average-tiebreak lookups that must be same layer.
    pub tight_to: HashMap<OpKey, HashSet<OpKey>>,
}

// ── Schedule Output ───────────────────────────────────────────

/// Complete MILP schedule output.
#[derive(Clone, Debug)]
pub struct Schedule {
    /// Phase assignment for each operation (0..4*N).
    pub phase_assign: HashMap<OpKey, i32>,
    /// Per-layer operation structure.
    pub std_layers: Vec<StdLayer>,
    /// Total number of transformer layers.
    pub num_layers: usize,
    /// Dimension birth phase (input dims have birth = -1).
    pub dim_birth: HashMap<DimId, i32>,
    /// Dimension death phase (last consumer phase).
    pub dim_death: HashMap<DimId, i32>,
    /// Dims alive after each persist boundary (phase → dim set).
    pub alive_after: HashMap<i32, HashSet<DimId>>,
    /// Model dimension (`d_model = 2 * max_alive`).
    pub width: usize,
    /// Slot assignment from interval coloring.
    pub slot_of: HashMap<DimId, usize>,
}

// ── Schedule Error ────────────────────────────────────────────

/// Errors that can occur during MILP scheduling.
#[derive(Debug)]
pub enum ScheduleError {
    /// No feasible schedule exists within the given layer count.
    Infeasible,
    /// The dependency graph contains a cycle.
    CycleDetected,
    /// The MILP solver returned an error.
    Solver(String),
}

impl std::fmt::Display for ScheduleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Infeasible => write!(f, "MILP infeasible; try more layers"),
            Self::CycleDetected => write!(f, "Cycle in operation dependencies"),
            Self::Solver(msg) => write!(f, "Solver error: {msg}"),
        }
    }
}

impl std::error::Error for ScheduleError {}

// ── Graph Building ────────────────────────────────────────────

/// Build the dependency graph from a [`ProgramGraph`].
///
/// Extracts operations (LookUp, ReGLU, Persist), their dependencies,
/// produced dimensions, and tight constraints.
pub fn build_dep_graph(pg: &ProgramGraph) -> DepGraph {
    let mut ops = Vec::new();
    let mut inputs = Vec::new();

    // Collect ops from dimensions
    for (&dim_id, dim) in &pg.all_dims {
        match &dim.kind {
            DimensionKind::Input => {
                inputs.push(dim_id);
            }
            DimensionKind::ReGLU { .. } => {
                ops.push(OpKey::ReGLU(dim_id));
            }
            DimensionKind::Persist { .. } => {
                ops.push(OpKey::Persist(dim_id));
            }
            DimensionKind::LookUp { .. }
            | DimensionKind::CumSum { .. }
            | DimensionKind::Generic => {}
        }
    }

    // Collect LookUp operations
    for &lookup_id in pg.all_lookups.keys() {
        ops.push(OpKey::LookUp(lookup_id));
    }

    // Produced dims
    let mut produced = HashMap::new();
    for &op in &ops {
        let dims: HashSet<DimId> = match op {
            OpKey::LookUp(id) => pg.all_lookups[&id].dim_ids.iter().copied().collect(),
            OpKey::ReGLU(dim_id) | OpKey::Persist(dim_id) => [dim_id].into_iter().collect(),
        };
        produced.insert(op, dims);
    }

    // Dependency cache: raw dimension deps for each op
    let mut deps_cache = HashMap::new();
    for &op in &ops {
        let deps: HashSet<DimId> = match op {
            OpKey::ReGLU(dim_id) => {
                let dim = &pg.all_dims[&dim_id];
                match &dim.kind {
                    DimensionKind::ReGLU { a_expr, b_expr } => a_expr
                        .terms
                        .keys()
                        .chain(b_expr.terms.keys())
                        .copied()
                        .collect(),
                    _ => HashSet::new(),
                }
            }
            OpKey::Persist(dim_id) => {
                let dim = &pg.all_dims[&dim_id];
                match &dim.kind {
                    DimensionKind::Persist { expr } => expr.terms.keys().copied().collect(),
                    _ => HashSet::new(),
                }
            }
            OpKey::LookUp(id) => {
                let lookup = &pg.all_lookups[&id];
                let mut d = HashSet::new();
                for expr in &lookup.query_exprs_2d {
                    d.extend(expr.terms.keys().copied());
                }
                for expr in &lookup.key_exprs_2d {
                    d.extend(expr.terms.keys().copied());
                }
                for expr in &lookup.value_exprs {
                    d.extend(expr.terms.keys().copied());
                }
                d.insert(pg.inv_log_pos);
                d
            }
        };
        deps_cache.insert(op, deps);
    }

    // dim_to_op: which operation produces each dim
    let mut dim_to_op = HashMap::new();
    for &op in &ops {
        for &dim in &produced[&op] {
            dim_to_op.insert(dim, op);
        }
    }

    // op_deps, children, consumers
    let mut op_deps = HashMap::new();
    let mut children: HashMap<OpKey, HashSet<OpKey>> = HashMap::new();
    let mut consumers: HashMap<DimId, HashSet<OpKey>> = HashMap::new();

    for &op in &ops {
        let mut deps = HashSet::new();
        for &dim in &deps_cache[&op] {
            consumers.entry(dim).or_default().insert(op);
            if let Some(&pred) = dim_to_op.get(&dim)
                && pred != op {
                    deps.insert(pred);
                    children.entry(pred).or_default().insert(op);
                }
        }
        op_deps.insert(op, deps);
    }

    // Tight constraints: persist/reglu → average-tiebreak lookups
    let avg_lookups: HashSet<OpKey> = ops
        .iter()
        .filter(|&&op| match op {
            OpKey::LookUp(id) => pg.all_lookups[&id].tie_break == TieBreak::Average,
            _ => false,
        })
        .copied()
        .collect();

    let mut tight_to = HashMap::new();
    for &op in &ops {
        match op {
            OpKey::ReGLU(_) | OpKey::Persist(_) => {
                let mut tight_lus = HashSet::new();
                for &dim in &deps_cache[&op] {
                    if let Some(&lu_op) = dim_to_op.get(&dim)
                        && avg_lookups.contains(&lu_op) {
                            tight_lus.insert(lu_op);
                        }
                }
                if !tight_lus.is_empty() {
                    tight_to.insert(op, tight_lus);
                }
            }
            OpKey::LookUp(_) => {}
        }
    }

    DepGraph {
        ops,
        inputs,
        produced,
        deps_cache,
        dim_to_op,
        op_deps,
        children,
        consumers,
        tight_to,
    }
}

/// Collect all dimensions (inputs + produced) in deterministic order.
fn all_result_dims(graph: &DepGraph) -> Vec<DimId> {
    let mut dims = Vec::new();
    let mut seen = HashSet::new();
    for &dim in &graph.inputs {
        dims.push(dim);
        seen.insert(dim);
    }
    for &op in &graph.ops {
        for &dim in &graph.produced[&op] {
            if seen.insert(dim) {
                dims.push(dim);
            }
        }
    }
    dims
}

// ── Min Layers (Critical Path) ────────────────────────────────

/// Compute minimum number of layers via ASAP critical-path analysis.
///
/// Phase parity constraints:
/// - LookUp: phase ≡ 0 (mod 4)
/// - ReGLU: phase ≡ 2 (mod 4)
/// - Persist: phase ≡ 1 or 3 (mod 4) (odd)
///
/// # Panics
///
/// Panics if the dependency graph contains a cycle.
pub fn min_layers(ops: &[OpKey], op_deps: &HashMap<OpKey, HashSet<OpKey>>) -> usize {
    if ops.is_empty() {
        return 0;
    }
    let mut phase: HashMap<OpKey, i32> = HashMap::new();
    let mut remaining: HashSet<OpKey> = ops.iter().copied().collect();

    while !remaining.is_empty() {
        let mut progress = false;
        for op in remaining.clone() {
            let deps = op_deps.get(&op).cloned().unwrap_or_default();
            if !deps.iter().all(|p| phase.contains_key(p)) {
                continue;
            }
            let lo: i32 = deps.iter().map(|p| phase[p]).max().unwrap_or(-1) + 1;
            let lo = match op {
                OpKey::LookUp(_) => lo + (-lo).rem_euclid(4),
                OpKey::ReGLU(_) => lo + (2 - lo.rem_euclid(4) + 4).rem_euclid(4),
                OpKey::Persist(_) => {
                    if lo % 2 == 1 {
                        lo
                    } else {
                        lo + 1
                    }
                }
            };
            phase.insert(op, lo);
            remaining.remove(&op);
            progress = true;
        }
        assert!(progress, "Cycle in dependencies");
    }

    let max_phase = phase.values().copied().max().unwrap_or(0);
    (max_phase / 4 + 1) as usize
}

// ── Phase Expression Helper ───────────────────────────────────

/// Compute the LP expression for the phase of an operation.
///
/// - LookUp: `4 * k[op]`
/// - ReGLU: `4 * k[op] + 2`
/// - Persist: `4 * k[op] + 1 + 2 * z[op]`
fn phase_of_expr(
    op: OpKey,
    k: &HashMap<OpKey, Variable>,
    z: &HashMap<DimId, Variable>,
) -> LpAffine {
    let k_var = k[&op];
    match op {
        OpKey::LookUp(_) => e(k_var) * 4.0,
        OpKey::ReGLU(_) => e(k_var) * 4.0 + 2.0,
        OpKey::Persist(dim_id) => e(k_var) * 4.0 + 1.0 + e(z[&dim_id]) * 2.0,
    }
}

// ── MILP Schedule ─────────────────────────────────────────────

/// Compute optimal MILP schedule minimizing `d_model`.
///
/// # Arguments
///
/// * `pg` — The computation graph to schedule.
/// * `max_layers` — Maximum number of transformer layers (defaults to [`min_layers`]).
///
/// # Errors
///
/// Returns [`ScheduleError`] if the problem is infeasible or the solver fails.
pub fn milp_schedule(
    pg: &ProgramGraph,
    max_layers: Option<usize>,
) -> Result<Schedule, ScheduleError> {
    let graph = build_dep_graph(pg);
    let dims_vec = all_result_dims(&graph);
    let n = max_layers.unwrap_or(min_layers(&graph.ops, &graph.op_deps));
    let p = 4 * n;

    if n == 0 || graph.ops.is_empty() {
        return Ok(trivial_schedule(pg, &dims_vec));
    }

    // Output and protected dims
    let output_dims: HashSet<DimId> = pg
        .output_tokens
        .iter()
        .flat_map(|expr| expr.terms.keys().copied())
        .collect();
    let protected: HashSet<DimId> = [pg.position, pg.inv_log_pos, pg.position_sq]
        .into_iter()
        .collect();

    info!(
        "MILP: {} ops, {} dims, {n} layers, {p} phases",
        graph.ops.len(),
        dims_vec.len()
    );

    let p_f = p as f64;
    let boundaries: Vec<i32> = (0..p as i32).filter(|c| c % 2 == 1).collect();

    // ── Phase 1: Create all variables ─────────────────────────
    let mut vars = ProblemVariables::new();

    // Objective: minimize D_half
    let d_half = vars.add(variable().min(0).integer());

    // k[op]: layer assignment (integer, 0..n-1)
    let mut k = HashMap::new();
    for &op in &graph.ops {
        k.insert(
            op,
            vars.add(variable().min(0).max((n - 1) as f64).integer()),
        );
    }

    // z[dim_id]: persist1 (z=0) vs persist2 (z=1) — binary
    let mut z = HashMap::new();
    for &op in &graph.ops {
        if let OpKey::Persist(dim_id) = op {
            z.insert(dim_id, vars.add(variable().binary()));
        }
    }

    // Death variables: one per non-output, non-protected dim with consumers
    let mut death = HashMap::new();
    for &d in &dims_vec {
        if output_dims.contains(&d) || protected.contains(&d) {
            continue;
        }
        let has_cons = graph
            .consumers
            .get(&d)
            .map(|s| s.iter().any(|op| k.contains_key(op)))
            .unwrap_or(false);
        if !has_cons && d != pg.position {
            continue;
        }
        death.insert(d, vars.add(variable().min(0).max((p - 1) as f64).integer()));
    }

    // Pre-compute dim categories for indicator variable creation
    let input_set: HashSet<DimId> = graph.inputs.iter().copied().collect();

    // Indicator variables: bb (birth≤c), eu (death≥c-1), ev (bb∧eu)
    let mut bb: HashMap<(usize, i32), Variable> = HashMap::new();
    let mut eu: HashMap<(usize, i32), Variable> = HashMap::new();
    let mut ev: HashMap<(usize, i32), Variable> = HashMap::new();

    for (di, &d) in dims_vec.iter().enumerate() {
        let is_out_or_prot = output_dims.contains(&d) || protected.contains(&d);
        let is_input = input_set.contains(&d);
        let has_death = death.contains_key(&d);
        let has_producer = graph
            .dim_to_op
            .get(&d)
            .map(|&op| k.contains_key(&op))
            .unwrap_or(false);

        for &c in &boundaries {
            if is_out_or_prot {
                if has_producer {
                    bb.insert((di, c), vars.add(variable().binary()));
                }
                continue;
            }
            if !has_death {
                continue;
            }
            if is_input {
                eu.insert((di, c), vars.add(variable().binary()));
            } else if has_producer {
                bb.insert((di, c), vars.add(variable().binary()));
                eu.insert((di, c), vars.add(variable().binary()));
                ev.insert((di, c), vars.add(variable().binary()));
            }
        }
    }

    // ── Phase 2: Create model + constraints ───────────────────
    let mut model = vars.minimise(e(d_half)).using(default_solver);

    // Dependency ordering: phase(op) >= phase(dep) + 1
    for &op in &graph.ops {
        if let Some(deps) = graph.op_deps.get(&op) {
            for &dep in deps {
                let phase_op = phase_of_expr(op, &k, &z);
                let phase_dep = phase_of_expr(dep, &k, &z);
                model = model.with(phase_op.geq(phase_dep + 1.0));
            }
        }
    }

    // Tight constraints: k[op] == k[lu] for average-tiebreak lookups
    for (&op, lus) in &graph.tight_to {
        for &lu in lus {
            if let (Some(k_op), Some(k_lu)) = (k.get(&op), k.get(&lu)) {
                model = model.with(e(*k_op).eq(e(*k_lu)));
            }
        }
    }

    // Death variable constraints: death[d] >= phase_of(c_op) for all consumers
    for (&d, &dv) in &death {
        if let Some(cons) = graph.consumers.get(&d) {
            for &c_op in cons {
                if k.contains_key(&c_op) {
                    let phase_c = phase_of_expr(c_op, &k, &z);
                    model = model.with(e(dv).geq(phase_c));
                }
            }
        }
    }

    // Birth indicator constraints: bb=1 iff phase_of(producer) ≤ c
    for (&(di, c), &bb_var) in &bb {
        let d = dims_vec[di];
        let is_out_or_prot = output_dims.contains(&d) || protected.contains(&d);

        if !is_out_or_prot {
            if let Some(&prod) = graph.dim_to_op.get(&d)
                && k.contains_key(&prod) {
                    let phase_prod = phase_of_expr(prod, &k, &z);
                    let c_f = c as f64;
                    // bb=1: phase ≤ c; bb=0: phase ≥ c+1
                    model = model.with((phase_prod.clone() + e(bb_var) * p_f).leq(c_f + p_f));
                    model = model.with((phase_prod + e(bb_var) * p_f).geq(c_f + 1.0));
                }
        } else if let Some(&prod) = graph.dim_to_op.get(&d) {
            // Output/protected dims produced by an op still need birth indicator
            if k.contains_key(&prod) {
                let phase_prod = phase_of_expr(prod, &k, &z);
                let c_f = c as f64;
                model = model.with((phase_prod.clone() + e(bb_var) * p_f).leq(c_f + p_f));
                model = model.with((phase_prod + e(bb_var) * p_f).geq(c_f + 1.0));
            }
        }
    }

    // Death indicator constraints: eu=1 iff death ≥ c-1
    for (&(di, c), &eu_var) in &eu {
        let d = dims_vec[di];
        if let Some(&dv) = death.get(&d) {
            let c_f = c as f64;
            // eu=1: death ≥ c-1; eu=0: death ≤ c-2
            model = model.with((e(dv) - e(eu_var) * p_f).geq(c_f - 1.0 - p_f));
            model = model.with((e(dv) - e(eu_var) * p_f).leq(c_f - 2.0));
        }
    }

    // Effective width indicator: ev = bb ∧ eu
    for (&(di, c), &ev_var) in &ev {
        if let (Some(&bb_var), Some(&eu_var)) = (bb.get(&(di, c)), eu.get(&(di, c))) {
            model = model.with(e(ev_var).leq(e(bb_var)));
            model = model.with(e(ev_var).leq(e(eu_var)));
            model = model.with((e(ev_var) - e(bb_var) - e(eu_var)).geq(-1.0));
        }
    }

    // Width constraint: 2 * D_half ≥ sum(effective_width) at each boundary
    for &c in &boundaries {
        let mut ew_sum = LpAffine::default();

        for (di, &d) in dims_vec.iter().enumerate() {
            let is_out_or_prot = output_dims.contains(&d) || protected.contains(&d);
            let is_input = input_set.contains(&d);

            if is_out_or_prot {
                if is_input {
                    ew_sum += 1.0;
                } else if let Some(&bb_var) = bb.get(&(di, c)) {
                    ew_sum += e(bb_var);
                }
                continue;
            }

            if let Some(&ev_var) = ev.get(&(di, c)) {
                ew_sum += e(ev_var);
            } else if let Some(&eu_var) = eu.get(&(di, c)) {
                ew_sum += e(eu_var);
            }
        }

        model = model.with((e(d_half) * 2.0).geq(ew_sum));
    }

    // ── Phase 3: Solve ────────────────────────────────────────
    info!("Solving MILP...");
    let solution = model
        .solve()
        .map_err(|err| ScheduleError::Solver(format!("{err}")))?;

    let opt_d_half = solution.value(d_half).round() as usize;
    let opt_d = 2 * opt_d_half;
    info!("MILP optimal d_model: {opt_d}");

    // ── Phase 4: Extract results ──────────────────────────────
    let pa = extract_phase_assignments(&graph, &k, &z, &solution);
    let num_layers = pa.values().copied().max().unwrap_or(0) / 4 + 1;
    let std_layers = build_std_layers(&pa, num_layers as usize);
    let (dim_birth, dim_death) =
        compute_birth_death(&dims_vec, &graph, &pa, &output_dims, &protected, num_layers);
    let alive_after = compute_alive_sets(&dims_vec, &dim_birth, &dim_death, num_layers as usize);
    let slot_of = interval_coloring(&dims_vec, &dim_birth, &dim_death, None);

    Ok(Schedule {
        phase_assign: pa,
        std_layers,
        num_layers: num_layers as usize,
        dim_birth,
        dim_death,
        alive_after,
        width: opt_d,
        slot_of,
    })
}

/// Build a trivial (empty) schedule for programs with no operations.
fn trivial_schedule(_pg: &ProgramGraph, dims_vec: &[DimId]) -> Schedule {
    let dim_birth: HashMap<DimId, i32> = dims_vec.iter().map(|&d| (d, -1)).collect();
    let dim_death: HashMap<DimId, i32> = dims_vec.iter().map(|&d| (d, 1)).collect();
    let alive_after = HashMap::new();
    let slot_of = interval_coloring(dims_vec, &dim_birth, &dim_death, None);

    Schedule {
        phase_assign: HashMap::new(),
        std_layers: Vec::new(),
        num_layers: 0,
        dim_birth,
        dim_death,
        alive_after,
        width: dims_vec.len(),
        slot_of,
    }
}

/// Extract phase assignments from the solved MILP.
fn extract_phase_assignments(
    graph: &DepGraph,
    k: &HashMap<OpKey, Variable>,
    z: &HashMap<DimId, Variable>,
    solution: &impl Solution,
) -> HashMap<OpKey, i32> {
    let mut pa = HashMap::new();
    for &op in &graph.ops {
        let layer = solution.value(k[&op]).round() as i32;
        let phase = match op {
            OpKey::LookUp(_) => 4 * layer,
            OpKey::ReGLU(_) => 4 * layer + 2,
            OpKey::Persist(dim_id) => {
                let is_p2 = solution.value(z[&dim_id]).round() as i32;
                4 * layer + 1 + 2 * is_p2
            }
        };
        pa.insert(op, phase);
    }
    pa
}

/// Build [`StdLayer`] list from phase assignments.
fn build_std_layers(pa: &HashMap<OpKey, i32>, num_layers: usize) -> Vec<StdLayer> {
    let mut layers = vec![StdLayer::default(); num_layers];
    for (&op, &phase) in pa {
        let l = (phase / 4) as usize;
        if l >= num_layers {
            continue;
        }
        match op {
            OpKey::LookUp(id) if phase % 4 == 0 => layers[l].attention.push(id),
            OpKey::Persist(dim) if phase % 4 == 1 => layers[l].persist1.push(dim),
            OpKey::ReGLU(dim) if phase % 4 == 2 => layers[l].ffn.push(dim),
            OpKey::Persist(dim) if phase % 4 == 3 => layers[l].persist2.push(dim),
            _ => {}
        }
    }
    layers
}

/// Compute dimension birth and death phases.
fn compute_birth_death(
    dims_vec: &[DimId],
    graph: &DepGraph,
    pa: &HashMap<OpKey, i32>,
    output_dims: &HashSet<DimId>,
    protected: &HashSet<DimId>,
    num_layers: i32,
) -> (HashMap<DimId, i32>, HashMap<DimId, i32>) {
    let input_set: HashSet<DimId> = graph.inputs.iter().copied().collect();
    let last_boundary = 4 * num_layers - 1;

    // Birth
    let mut dim_birth = HashMap::new();
    for &d in dims_vec {
        if input_set.contains(&d) {
            dim_birth.insert(d, -1);
        } else if let Some(&prod) = graph.dim_to_op.get(&d)
            && let Some(&phase) = pa.get(&prod) {
                dim_birth.insert(d, phase);
            }
    }

    // Death
    let mut dim_death = HashMap::new();
    for &d in dims_vec {
        if output_dims.contains(&d) || protected.contains(&d) {
            dim_death.insert(d, last_boundary + 1);
            continue;
        }
        let mut last = -1;
        if let Some(cons) = graph.consumers.get(&d) {
            for &c_op in cons {
                if let Some(&phase) = pa.get(&c_op) {
                    last = last.max(phase);
                }
            }
        }
        if last >= 0 {
            dim_death.insert(d, last);
        } else if let Some(&birth) = dim_birth.get(&d) {
            dim_death.insert(d, birth);
        }
    }

    (dim_birth, dim_death)
}

/// Compute alive sets at each persist boundary.
fn compute_alive_sets(
    dims_vec: &[DimId],
    dim_birth: &HashMap<DimId, i32>,
    dim_death: &HashMap<DimId, i32>,
    num_layers: usize,
) -> HashMap<i32, HashSet<DimId>> {
    let mut alive_after = HashMap::new();
    for l in 0..num_layers {
        for sp in [1, 3] {
            let c = 4 * l as i32 + sp;
            let alive: HashSet<DimId> = dims_vec
                .iter()
                .copied()
                .filter(|&d| {
                    let birth = dim_birth.get(&d).copied().unwrap_or(i32::MAX);
                    let death = dim_death.get(&d).copied().unwrap_or(i32::MIN);
                    birth <= c && death > c
                })
                .collect();
            alive_after.insert(c, alive);
        }
    }
    alive_after
}

// ── Interval Coloring ─────────────────────────────────────────

/// Greedy interval coloring: assign slots to dims with `[birth, death)` lifetimes.
///
/// Uses a min-heap to track freed slots, giving optimal (minimum) slot count
/// for interval-graph coloring.
///
/// # Arguments
///
/// * `all_dims` — All dimension IDs to assign.
/// * `dim_birth` — Birth phase for each dim (input dims have birth = -1).
/// * `dim_death` — Death phase for each dim.
/// * `fixed` — Optional pre-assigned slots `{dim: slot}`.
///
/// # Returns
///
/// Map from dim to slot index (0-based).
pub fn interval_coloring(
    all_dims: &[DimId],
    dim_birth: &HashMap<DimId, i32>,
    dim_death: &HashMap<DimId, i32>,
    fixed: Option<&HashMap<DimId, usize>>,
) -> HashMap<DimId, usize> {
    let empty_fixed: HashMap<DimId, usize> = HashMap::new();
    let fixed = fixed.unwrap_or(&empty_fixed);

    // Sort remaining dims by birth phase
    let remaining: Vec<DimId> = all_dims
        .iter()
        .copied()
        .filter(|d| !fixed.contains_key(d))
        .collect();

    let mut items: Vec<(i32, i32, DimId)> = remaining
        .into_iter()
        .filter_map(|d| {
            let birth = dim_birth.get(&d)?;
            let death = dim_death.get(&d)?;
            if *death > *birth {
                Some((*birth, *death, d))
            } else {
                None
            }
        })
        .collect();
    items.sort_by_key(|(birth, _, _)| *birth);

    let mut slot_of = fixed.clone();

    // Min-heap of (death_phase, slot) for freed slots
    let mut free: BinaryHeap<Reverse<(i32, usize)>> = BinaryHeap::new();
    let mut next_slot = fixed.values().max().map(|&m| m + 1).unwrap_or(0);

    // Seed free heap with fixed slots whose dims have died
    for (&d, &slot) in fixed {
        if let Some(&death) = dim_death.get(&d) {
            free.push(Reverse((death, slot)));
        }
    }

    for (birth, death, d) in items {
        // Collect all slots freed by this birth phase
        let mut available = Vec::new();
        while let Some(&Reverse((death_free, slot))) = free.peek() {
            if death_free <= birth {
                free.pop();
                available.push(slot);
            } else {
                break;
            }
        }

        let slot = match available.iter().min() {
            Some(&s) => {
                // Re-insert unused slots
                for &s2 in &available {
                    if s2 != s {
                        free.push(Reverse((death, s2)));
                    }
                }
                s
            }
            None => {
                let s = next_slot;
                next_slot += 1;
                s
            }
        };

        slot_of.insert(d, slot);
        free.push(Reverse((death, slot)));
    }

    slot_of
}

// ── Tests ─────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_simple_graph() -> ProgramGraph {
        let mut builder = GraphBuilder::new();
        let pos = builder.position;
        let value = Expression::from_dim(pos);
        let fetched = builder.fetch(value, None, None, None, TieBreak::Latest);
        let _persisted = builder.persist(fetched);
        builder.build(vec![], vec![])
    }

    fn make_chain_graph() -> ProgramGraph {
        let mut builder = GraphBuilder::new();
        let pos = builder.position;
        let _one = builder.one;

        // fetch → persist → fetch → persist (chain)
        let v1 = Expression::from_dim(pos);
        let fetched1 = builder.fetch(v1, None, None, None, TieBreak::Latest);
        let persist1 = builder.persist(fetched1);

        let v2 = persist1;
        let fetched2 = builder.fetch(v2, None, None, None, TieBreak::Latest);
        let persist2 = builder.persist(fetched2);

        builder.build(vec![], vec![persist2])
    }

    // ── DepGraph Tests ────────────────────────────────────────

    #[test]
    fn test_build_dep_graph_simple() {
        let pg = make_simple_graph();
        let graph = build_dep_graph(&pg);

        // Should have at least one LookUp and one Persist
        let has_lookup = graph.ops.iter().any(|op| matches!(op, OpKey::LookUp(_)));
        let has_persist = graph.ops.iter().any(|op| matches!(op, OpKey::Persist(_)));
        assert!(has_lookup, "Expected at least one LookUp operation");
        assert!(has_persist, "Expected at least one Persist operation");

        // Persist should have non-empty dependencies (ReGLU dims from fetch encoding, or LookUp)
        let persist_op = graph
            .ops
            .iter()
            .find(|op| matches!(op, OpKey::Persist(_)))
            .copied()
            .unwrap();
        let deps = graph.op_deps.get(&persist_op).cloned().unwrap_or_default();
        assert!(
            !deps.is_empty(),
            "Persist should depend on at least one operation, got no deps"
        );
    }

    #[test]
    fn test_build_dep_graph_inputs() {
        let pg = make_simple_graph();
        let graph = build_dep_graph(&pg);

        // Should have input dims
        assert!(!graph.inputs.is_empty(), "Expected input dimensions");

        // Each input should have a DimId in all_dims
        for &d in &graph.inputs {
            assert!(
                pg.all_dims.contains_key(&d),
                "Input dim {d} not in all_dims"
            );
        }
    }

    #[test]
    fn test_build_dep_graph_produced() {
        let pg = make_simple_graph();
        let graph = build_dep_graph(&pg);

        // Each op should produce at least one dim
        for &op in &graph.ops {
            let produced = graph.produced.get(&op).cloned().unwrap_or_default();
            assert!(
                !produced.is_empty(),
                "Op {:?} should produce at least one dim",
                op
            );
        }

        // dim_to_op should be consistent with produced
        for (&op, dims) in &graph.produced {
            for &d in dims {
                assert_eq!(
                    graph.dim_to_op.get(&d),
                    Some(&op),
                    "dim_to_op[{d}] should point to {op:?}"
                );
            }
        }
    }

    // ── Min Layers Tests ──────────────────────────────────────

    #[test]
    fn test_min_layers_empty() {
        let n = min_layers(&[], &HashMap::new());
        assert_eq!(n, 0, "Empty graph should need 0 layers");
    }

    #[test]
    fn test_min_layers_single_lookup() {
        let pg = make_simple_graph();
        let graph = build_dep_graph(&pg);

        let lookups: Vec<OpKey> = graph
            .ops
            .iter()
            .filter(|&&op| matches!(op, OpKey::LookUp(_)))
            .copied()
            .collect();

        // Single lookup with no deps → 1 layer
        let op_deps: HashMap<OpKey, HashSet<OpKey>> =
            lookups.iter().map(|&op| (op, HashSet::new())).collect();
        let n = min_layers(&lookups, &op_deps);
        assert!(n >= 1, "Single lookup should need at least 1 layer");
    }

    #[test]
    fn test_min_layers_chain() {
        let pg = make_chain_graph();
        let graph = build_dep_graph(&pg);

        let n = min_layers(&graph.ops, &graph.op_deps);
        // Chain: fetch1 → persist1 → fetch2 → persist2
        // fetch1 at L0, persist1 at L0, fetch2 depends on persist1
        // fetch2 at L0 or L1 depending on phase alignment
        assert!(n >= 1, "Chain should need at least 1 layer");
    }

    // ── MILP Schedule Tests ───────────────────────────────────

    #[test]
    fn test_milp_schedule_simple() {
        let pg = make_simple_graph();
        let schedule = milp_schedule(&pg, None).unwrap();

        assert!(
            schedule.num_layers >= 1,
            "Should have at least 1 layer, got {}",
            schedule.num_layers
        );
        assert!(
            schedule.width > 0,
            "d_model should be positive, got {}",
            schedule.width
        );

        // Verify dependency ordering
        for (&op, &phase) in &schedule.phase_assign {
            let graph = build_dep_graph(&pg);
            if let Some(deps) = graph.op_deps.get(&op) {
                for &dep in deps {
                    if let Some(&dep_phase) = schedule.phase_assign.get(&dep) {
                        assert!(
                            phase > dep_phase,
                            "Dependency violation: {:?} (phase {phase}) should be after {:?} (phase {dep_phase})",
                            op,
                            dep
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_milp_schedule_chain() {
        let pg = make_chain_graph();
        let schedule = milp_schedule(&pg, None).unwrap();

        assert!(
            schedule.num_layers >= 1,
            "Chain should have at least 1 layer"
        );

        // Should have birth/death for non-input dims
        assert!(!schedule.dim_birth.is_empty(), "Should have birth phases");
        assert!(!schedule.dim_death.is_empty(), "Should have death phases");
    }

    #[test]
    fn test_milp_schedule_phase_parity() {
        let pg = make_simple_graph();
        let schedule = milp_schedule(&pg, None).unwrap();

        for (&op, &phase) in &schedule.phase_assign {
            match op {
                OpKey::LookUp(_) => {
                    assert_eq!(
                        phase % 4,
                        0,
                        "LookUp phase should be ≡ 0 (mod 4), got {phase}"
                    );
                }
                OpKey::ReGLU(_) => {
                    assert_eq!(
                        phase % 4,
                        2,
                        "ReGLU phase should be ≡ 2 (mod 4), got {phase}"
                    );
                }
                OpKey::Persist(_) => {
                    assert_eq!(phase % 2, 1, "Persist phase should be odd, got {phase}");
                }
            }
        }
    }

    #[test]
    fn test_milp_schedule_std_layers() {
        let pg = make_simple_graph();
        let schedule = milp_schedule(&pg, None).unwrap();

        // Std layers should match num_layers
        assert_eq!(
            schedule.std_layers.len(),
            schedule.num_layers,
            "std_layers count should match num_layers"
        );

        // Each layer's attention should contain LookupIds from the schedule
        for (l, layer) in schedule.std_layers.iter().enumerate() {
            let l_i = l as i32;
            for &id in &layer.attention {
                let op = OpKey::LookUp(id);
                if let Some(&phase) = schedule.phase_assign.get(&op) {
                    assert_eq!(phase, 4 * l_i, "Lookup {id} should be at phase {}", 4 * l_i);
                }
            }
            for &dim in &layer.ffn {
                let op = OpKey::ReGLU(dim);
                if let Some(&phase) = schedule.phase_assign.get(&op) {
                    assert_eq!(
                        phase,
                        4 * l_i + 2,
                        "ReGLU {dim} should be at phase {}",
                        4 * l_i + 2
                    );
                }
            }
        }
    }

    // ── Alive Sets Tests ──────────────────────────────────────

    #[test]
    fn test_alive_sets_consistency() {
        let pg = make_simple_graph();
        let schedule = milp_schedule(&pg, None).unwrap();

        // Every dim in alive_after should be born before and die after the boundary
        for (&c, alive) in &schedule.alive_after {
            for &d in alive {
                let birth = schedule.dim_birth.get(&d).copied().unwrap_or(i32::MAX);
                let death = schedule.dim_death.get(&d).copied().unwrap_or(i32::MIN);
                assert!(
                    birth <= c && death > c,
                    "Dim {d} in alive_after[{c}] but birth={birth}, death={death}"
                );
            }
        }
    }

    // ── Interval Coloring Tests ───────────────────────────────

    #[test]
    fn test_interval_coloring_empty() {
        let slot_of = interval_coloring(&[], &HashMap::new(), &HashMap::new(), None);
        assert!(slot_of.is_empty());
    }

    #[test]
    fn test_interval_coloring_non_overlapping() {
        let dims = vec![0u32, 1, 2];
        let birth = HashMap::from([(0u32, 0i32), (1, 5), (2, 10)]);
        let death = HashMap::from([(0u32, 4i32), (1, 9), (2, 14)]);

        let slot_of = interval_coloring(&dims, &birth, &death, None);

        // Non-overlapping intervals should all get slot 0
        assert_eq!(slot_of[&0], 0);
        assert_eq!(slot_of[&1], 0);
        assert_eq!(slot_of[&2], 0);
    }

    #[test]
    fn test_interval_coloring_overlapping() {
        let dims = vec![0u32, 1];
        let birth = HashMap::from([(0u32, 0i32), (1, 0)]);
        let death = HashMap::from([(0u32, 5i32), (1, 5)]);

        let slot_of = interval_coloring(&dims, &birth, &death, None);

        // Overlapping intervals need different slots
        assert_ne!(slot_of[&0], slot_of[&1]);
        // Should use exactly 2 slots
        let max_slot = *slot_of.values().max().unwrap_or(&0);
        assert_eq!(max_slot, 1, "Should use exactly 2 slots (0 and 1)");
    }

    #[test]
    fn test_interval_coloring_partial_overlap() {
        // [0,3) and [2,5) overlap → 2 slots
        // [4,7) doesn't overlap with [0,3) → can reuse slot 0
        let dims = vec![0u32, 1, 2];
        let birth = HashMap::from([(0u32, 0i32), (1, 2), (2, 4)]);
        let death = HashMap::from([(0u32, 3i32), (1, 5), (2, 7)]);

        let slot_of = interval_coloring(&dims, &birth, &death, None);

        // Dims 0 and 1 overlap → different slots
        assert_ne!(slot_of[&0], slot_of[&1]);
        // Dim 2 starts at 4, dim 0 dies at 3 → can reuse
        assert_eq!(
            slot_of[&2], slot_of[&0],
            "Dim 2 should reuse slot from dim 0"
        );
    }

    #[test]
    fn test_interval_coloring_with_fixed() {
        let dims = vec![0u32, 1];
        let birth = HashMap::from([(0u32, 0i32), (1, 0)]);
        let death = HashMap::from([(0u32, 5i32), (1, 5)]);
        let fixed = HashMap::from([(0u32, 3usize)]);

        let slot_of = interval_coloring(&dims, &birth, &death, Some(&fixed));

        // Dim 0 is fixed to slot 3
        assert_eq!(slot_of[&0], 3);
        // Dim 1 overlaps with dim 0 → needs a different slot
        assert_ne!(slot_of[&1], 3);
    }

    #[test]
    fn test_interval_coloring_slot_reuse() {
        // Create a staircase pattern: each dim starts after the previous dies
        let dims: Vec<DimId> = (0..5u32).collect();
        let birth: HashMap<DimId, i32> = (0..5u32).map(|i| (i, i as i32 * 3)).collect();
        let death: HashMap<DimId, i32> = (0..5u32).map(|i| (i, i as i32 * 3 + 2)).collect();

        let slot_of = interval_coloring(&dims, &birth, &death, None);

        // All should get slot 0 since they don't overlap
        for i in 0..5u32 {
            assert_eq!(slot_of[&i], 0, "Non-overlapping dim {i} should get slot 0");
        }
    }

    // ── Integration Tests ─────────────────────────────────────

    #[test]
    fn test_schedule_with_max_layers() {
        let pg = make_chain_graph();

        // Chain graph has many internal ReGLU dims (from encode_2d_key in fetch),
        // so it needs more layers. Use auto-detect first, then verify a capped version works.
        let auto_schedule = milp_schedule(&pg, None).unwrap();
        let auto_layers = auto_schedule.num_layers;

        // Verify that explicitly setting max_layers=auto_layers gives the same result
        let schedule = milp_schedule(&pg, Some(auto_layers)).unwrap();
        assert!(
            schedule.num_layers <= auto_layers,
            "Should use at most {auto_layers} layers, got {}",
            schedule.num_layers
        );
    }

    #[test]
    fn test_schedule_alive_width_consistency() {
        let pg = make_simple_graph();
        let schedule = milp_schedule(&pg, None).unwrap();

        // Width (d_model = 2 * D_half) should be at least the max alive count.
        // Note: alive count is an upper bound on effective width (some dims don't need slots).
        let max_alive = schedule
            .alive_after
            .values()
            .map(|s| s.len())
            .max()
            .unwrap_or(0);
        assert!(
            schedule.width >= max_alive,
            "d_model ({}) should be >= max_alive ({})",
            schedule.width,
            max_alive
        );
    }

    #[test]
    fn test_schedule_slots_unique_per_layer() {
        let pg = make_simple_graph();
        let schedule = milp_schedule(&pg, None).unwrap();

        // Dims alive at the same boundary should have different slots
        for alive in schedule.alive_after.values() {
            let mut slots: Vec<usize> = alive
                .iter()
                .filter_map(|d| schedule.slot_of.get(d).copied())
                .collect();
            slots.sort();
            slots.dedup();
            // All slots should be unique (no duplicates after dedup means length unchanged)
            let original_count = alive.len();
            let unique_slots: HashSet<usize> = alive
                .iter()
                .filter_map(|d| schedule.slot_of.get(d))
                .copied()
                .collect();
            assert_eq!(
                unique_slots.len(),
                original_count,
                "Alive dims should have unique slots"
            );
        }
    }

    #[test]
    fn test_phase_enum_index() {
        assert_eq!(Phase::Attention.index(), 0);
        assert_eq!(Phase::Persist1.index(), 1);
        assert_eq!(Phase::Ffn.index(), 2);
        assert_eq!(Phase::Persist2.index(), 3);
    }

    #[test]
    fn test_opkey_kind_label() {
        assert_eq!(OpKey::LookUp(0).kind_label(), "lookup");
        assert_eq!(OpKey::ReGLU(1).kind_label(), "reglu");
        assert_eq!(OpKey::Persist(2).kind_label(), "persist");
    }

    #[test]
    fn test_schedule_error_display() {
        let err = ScheduleError::Infeasible;
        assert!(err.to_string().contains("infeasible"));

        let err = ScheduleError::CycleDetected;
        assert!(err.to_string().contains("Cycle"));

        let err = ScheduleError::Solver("test".to_string());
        assert!(err.to_string().contains("test"));
    }
}
