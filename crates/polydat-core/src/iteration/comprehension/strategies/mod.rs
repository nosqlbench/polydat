// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Strategy implementations — spec §3.6 + §10.2 R2 + §10.7.8.
//!
//! ## Selection, then lookup
//!
//! A strategy's order is a function of its input's shape alone: the
//! input's `IndexFn`, its tuple count, the truncation, and the seed.
//! [`Strategy::select`] computes that order as a [`Selection`] of
//! positions into the input without seeing a tuple, which is what
//! lets an index-addressed evaluator choose `order halton/100`'s
//! tuples from a large product and compute only those 100 (spec
//! §10.2 R2). [`Strategy::apply`] is the same selection looked up
//! against an [`EvaluatedInput`]'s materialized tuples.
//!
//! Per spec §10.7.8 this is the **strategy invocation
//! contract**: V4 fires at invocation time against the input's
//! `index_fn` — definitively, regardless of how the input source
//! was authored (literal, range, context-free generator, or
//! workload-param).
//!
//! Each strategy module holds a closed-form path over an `IndexFn`
//! that supports lookup and a fallback over a one-axis position
//! range; both produce positions.
//!
//! Strategies are selected by [`StrategyName`]; [`for_name`]
//! dispatches a strategy name to its boxed [`Strategy`] impl.

use super::metadata::IndexFn;
use super::strategy::StrategyName;

pub mod antidiagonal;
pub mod diagonal;
pub mod extrema;
pub mod halton;
pub mod lex;
pub mod lhs;
pub mod prng;
pub mod reverse_lex;
pub mod shells;
pub mod shuffle;
pub mod sobol;

/// A multi-coordinate index. Each component is the per-axis
/// position in the input's index space. Length equals the
/// input's dimensionality (1 for `Lockstep` / `Modular` /
/// `Concatenation`; N for `Lattice` / `Continuous` /
/// `Hybrid`).
///
/// `MultiIndex` is the indexed-form output type. The R2 IR
/// opcode emitted by the IR compiler consumes these and resolves
/// each through the input's `IndexFn` to dispense the actual
/// tuple.
pub type MultiIndex = Vec<u64>;

/// A named-tuple value. Subset of the polydat `Value` set that
/// is the strategy layer's currency; the runtime walker
/// converts `Value`s to it before `apply` and maps results
/// back. For the strategy module in isolation, this
/// lightweight type lets tests run without pulling in the
/// broader runtime.
#[derive(Debug, Clone, PartialEq)]
pub struct Tuple {
    /// The tuple's `(name, value)` pairs, in shape order.
    pub bindings: Vec<(String, TupleValue)>,
}

/// Subset of polydat's `Value` enum. `TupleValue` is the
/// strategy layer's currency; the runtime walker converts
/// `Value`s to it before `apply` and maps results back.
#[derive(Debug, Clone, PartialEq)]
pub enum TupleValue {
    /// An unsigned integer.
    U64(u64),
    /// A signed integer.
    I64(i64),
    /// A float.
    F64(f64),
    /// A string.
    Str(String),
    /// A boolean.
    Bool(bool),
}

impl Tuple {
    /// An empty tuple.
    pub fn new() -> Self {
        Self {
            bindings: Vec::new(),
        }
    }

    /// The tuple with one more binding.
    pub fn with<K: Into<String>>(mut self, key: K, value: TupleValue) -> Self {
        self.bindings.push((key.into(), value));
        self
    }
}

impl Default for Tuple {
    fn default() -> Self {
        Self::new()
    }
}

/// The materialized input to a strategy at invocation time
/// (spec §10.7.8).
///
/// `tuples` are the input stream's tuples in source order (the
/// natural enumeration of the upstream comprehension subtree).
/// `cardinality` matches `tuples.len() as u64`. `index_fn` is
/// the addressing scheme the input actually satisfies —
/// derived from observed shape for Generator /
/// WorkloadParamList leaves via the [`crate::iteration::comprehension::eval_source`]
/// layer, combined upward by the runtime walker per spec
/// §10.7.2 propagation rules.
pub struct EvaluatedInput {
    /// The input's tuples, in source order.
    pub tuples: Vec<Tuple>,
    /// How many tuples: `tuples.len()`.
    pub cardinality: u64,
    /// The addressing scheme the input satisfies.
    pub index_fn: IndexFn,
}

/// The positions a strategy emits, in emission order, as offsets
/// into its input's natural enumeration.
///
/// A prefix and a reversal are held as their bounds; every other
/// order is the list of positions it chose, one per emitted tuple.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Selection {
    /// Positions `0..n`.
    Prefix(u64),
    /// Positions `total - 1`, `total - 2`, …, `len` of them.
    Reverse {
        /// The input's tuple count.
        total: u64,
        /// How many positions are emitted.
        len: u64,
    },
    /// The chosen positions, each below the input's tuple count.
    Positions(Vec<u64>),
}

impl Selection {
    /// How many positions the selection emits.
    pub fn len(&self) -> u64 {
        match self {
            Selection::Prefix(n) => *n,
            Selection::Reverse { len, .. } => *len,
            Selection::Positions(p) => p.len() as u64,
        }
    }

    /// Whether the selection emits nothing.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The input position emitted at `i`, or `None` past the end.
    pub fn get(&self, i: u64) -> Option<u64> {
        match self {
            Selection::Prefix(n) => (i < *n).then_some(i),
            Selection::Reverse { total, len } => (i < *len).then(|| total - 1 - i),
            Selection::Positions(p) => usize::try_from(i).ok().and_then(|i| p.get(i).copied()),
        }
    }

    /// The emitted positions, in order.
    pub fn iter(&self) -> impl Iterator<Item = u64> + '_ {
        (0..self.len()).filter_map(|i| self.get(i))
    }

    /// The positions of `multi_indices` over `idx`, keeping those
    /// that land below `cardinality`.
    pub(crate) fn from_multi_indices(
        idx: &IndexFn,
        multi_indices: Vec<MultiIndex>,
        cardinality: u64,
    ) -> Self {
        Selection::Positions(
            multi_indices
                .into_iter()
                .filter_map(|mi| multi_index_to_flat(idx, &mi))
                .map(|flat| flat as u64)
                .filter(|p| *p < cardinality)
                .collect(),
        )
    }
}

/// The strategy invocation surface per spec §10.7.8.
///
/// Implementations are stateless — every call to
/// [`select`](Strategy::select) produces the same positions given the
/// same inputs (deterministic). PRNG-based strategies (`Shuffle`,
/// `Lhs`) derive their state from the authored seed, or a module
/// constant when none is authored, plus the input length; no
/// per-streamer seed is threaded.
pub trait Strategy {
    /// The strategy's name. Mirrors [`StrategyName`].
    fn name(&self) -> StrategyName;

    /// V4 input-shape check (spec §3.6). `None` represents an
    /// input with no closed-form index function; only `Lex`
    /// accepts that. Concrete `IndexFn` variants are accepted
    /// per the per-strategy rules in spec §3.6's table.
    fn accepts_input(&self, idx: Option<&IndexFn>) -> bool;

    /// R2 push-down eligibility (spec §10.2 R2). `true` if this
    /// strategy has a closed-form multi-index rule over the given
    /// input; otherwise [`select`](Strategy::select) orders the
    /// input's positions as one axis.
    fn has_closed_form_for(&self, idx: &IndexFn) -> bool;

    /// The positions this strategy emits over an input of
    /// `cardinality` tuples addressed by `index_fn`, cut to
    /// `truncation`, under the authored `seed` (comprehension_forms.md
    /// §3.6: a seeded strategy, `Shuffle` or `Lhs`, derives its state
    /// from the seed and the input's structural identity, and from its
    /// fixed default when `seed` is `None`; every other strategy
    /// ignores it).
    ///
    /// The selection reads no tuple, so a caller that can compute the
    /// tuple at a position computes only the selected ones. V4 is the
    /// caller's responsibility: call `accepts_input` first.
    fn select(
        &self,
        index_fn: &IndexFn,
        cardinality: u64,
        truncation: Option<u64>,
        seed: Option<u64>,
    ) -> Selection;

    /// Apply this strategy to the given input: its
    /// [`select`](Strategy::select)ion looked up against
    /// `input.tuples`.
    ///
    /// V4 is the caller's responsibility — call
    /// `accepts_input(Some(&input.index_fn))` before `apply`
    /// to fire V4 at strategy-invocation time per spec §10.7.8.
    fn apply(&self, input: &EvaluatedInput, truncation: Option<u64>) -> Vec<Tuple> {
        self.apply_seeded(input, truncation, None)
    }

    /// [`apply`](Strategy::apply) under an authored seed.
    fn apply_seeded(
        &self,
        input: &EvaluatedInput,
        truncation: Option<u64>,
        seed: Option<u64>,
    ) -> Vec<Tuple> {
        self.select(&input.index_fn, input.tuples.len() as u64, truncation, seed)
            .iter()
            .filter_map(|p| input.tuples.get(p as usize).cloned())
            .collect()
    }
}

/// `n` capped at `total`, or `total` when there is no cap.
pub(crate) fn capped(truncation: Option<u64>, total: u64) -> u64 {
    truncation.map_or(total, |t| t.min(total))
}

/// Dispatch a [`StrategyName`] to its concrete [`Strategy`]
/// implementation. The returned trait object is stateless;
/// callers can hold a single instance per strategy name for
/// the life of the process if desired.
pub fn for_name(name: StrategyName) -> Box<dyn Strategy + Send + Sync> {
    match name {
        StrategyName::Lex => Box::new(lex::Lex),
        StrategyName::ReverseLex => Box::new(reverse_lex::ReverseLex),
        StrategyName::Shuffle => Box::new(shuffle::Shuffle),
        StrategyName::Halton => Box::new(halton::Halton),
        StrategyName::Sobol => Box::new(sobol::Sobol),
        StrategyName::Lhs => Box::new(lhs::Lhs),
        StrategyName::Extrema => Box::new(extrema::Extrema),
        StrategyName::Shells => Box::new(shells::Shells),
        StrategyName::Diagonal => Box::new(diagonal::Diagonal),
        StrategyName::Antidiagonal => Box::new(antidiagonal::Antidiagonal),
    }
}

/// Resolve a [`MultiIndex`] to a flat position in the
/// input's tuple list, given the input's [`IndexFn`].
///
/// The flat position matches the natural enumeration order
/// the runtime walker produces:
///
/// - `Lattice { axis_sizes: [s0, s1, …, sN-1] }` — row-major
///   over the axes: `flat = i0 * s1 * s2 * … + i1 * s2 * … + … + iN-1`.
///   This matches the runtime walker's cartesian enumeration
///   (head axis varies slowest, tail nested).
/// - `Lockstep { length }` — one-axis identity:
///   `flat = mi[0]`.
/// - `Modular { axis_sizes }` — one-axis identity over `max(axis_sizes)`:
///   `flat = mi[0]`.
/// - `Concatenation { segment_sizes }` — one-axis identity
///   over `Σ segment_sizes`: `flat = mi[0]`.
/// - `Continuous` / `Hybrid` — `None`; these inputs have no
///   pre-materialized tuple list (the strategy's multi-indices
///   are quantiles, not lookups).
///
/// Returns `None` for out-of-range positions or dimension
/// mismatches.
pub fn multi_index_to_flat(idx: &IndexFn, mi: &MultiIndex) -> Option<usize> {
    match idx {
        IndexFn::Lattice { axis_sizes } => {
            if mi.len() != axis_sizes.len() {
                return None;
            }
            let mut flat: u64 = 0;
            let mut stride: u64 = 1;
            for i in (0..axis_sizes.len()).rev() {
                let pos = mi[i];
                let size = axis_sizes[i];
                if pos >= size {
                    return None;
                }
                flat = flat.checked_add(pos.checked_mul(stride)?)?;
                stride = stride.checked_mul(size)?;
            }
            Some(flat as usize)
        }
        IndexFn::Lockstep { length } => {
            if mi.len() != 1 || mi[0] >= *length {
                return None;
            }
            Some(mi[0] as usize)
        }
        IndexFn::Modular { axis_sizes } => {
            let max = axis_sizes.iter().copied().max().unwrap_or(0);
            if mi.len() != 1 || mi[0] >= max {
                return None;
            }
            Some(mi[0] as usize)
        }
        IndexFn::Concatenation { segment_sizes } => {
            let total: u64 = segment_sizes.iter().copied().sum();
            if mi.len() != 1 || mi[0] >= total {
                return None;
            }
            Some(mi[0] as usize)
        }
        IndexFn::Continuous { .. } | IndexFn::Hybrid { .. } => None,
    }
}

/// `true` when [`multi_index_to_flat`] returns a usable
/// position for in-range multi-indices over this `IndexFn`.
/// `false` for `Continuous` / `Hybrid` where the indexed
/// strategy emits quantiles, not lookups.
pub fn index_fn_supports_lookup(idx: &IndexFn) -> bool {
    !matches!(idx, IndexFn::Continuous { .. } | IndexFn::Hybrid { .. })
}

/// Cardinality of an `IndexFn`. Used by strategies to size
/// their output when no truncation is specified. Mirrors the
/// helper in `metadata.rs` but lives here to avoid a circular
/// dependency.
pub(crate) fn index_fn_size(idx: &IndexFn) -> u64 {
    match idx {
        IndexFn::Lattice { axis_sizes } => axis_sizes
            .iter()
            .copied()
            .fold(1u64, |a, b| a.saturating_mul(b)),
        IndexFn::Lockstep { length } => *length,
        IndexFn::Modular { axis_sizes } => axis_sizes.iter().copied().max().unwrap_or(0),
        IndexFn::Concatenation { segment_sizes } => segment_sizes
            .iter()
            .copied()
            .fold(0u64, |a, b| a.saturating_add(b)),
        IndexFn::Continuous { .. } | IndexFn::Hybrid { .. } => 0,
    }
}

/// Lattice dimensionality of an `IndexFn`. Used by strategies
/// that branch on dimensionality (Extrema's corner count,
/// Lhs's per-axis stratification).
pub(crate) fn index_fn_dim(idx: &IndexFn) -> usize {
    match idx {
        IndexFn::Lattice { axis_sizes } => axis_sizes.len(),
        IndexFn::Continuous { intervals, .. } => intervals.len(),
        IndexFn::Hybrid {
            discrete_axes,
            continuous_axes,
            ..
        } => discrete_axes.len() + continuous_axes.len(),
        // A zip and a union are one axis of positions, which is what
        // `multi_index_to_flat` reads from them.
        IndexFn::Lockstep { .. } | IndexFn::Modular { .. } | IndexFn::Concatenation { .. } => 1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn for_name_dispatches_to_correct_strategy() {
        assert_eq!(for_name(StrategyName::Lex).name(), StrategyName::Lex);
        assert_eq!(for_name(StrategyName::Halton).name(), StrategyName::Halton);
        assert_eq!(
            for_name(StrategyName::Extrema).name(),
            StrategyName::Extrema
        );
    }

    #[test]
    fn index_fn_size_lattice() {
        let idx = IndexFn::Lattice {
            axis_sizes: vec![3, 4, 5],
        };
        assert_eq!(index_fn_size(&idx), 60);
    }

    #[test]
    fn index_fn_size_concatenation() {
        let idx = IndexFn::Concatenation {
            segment_sizes: vec![10, 20, 30],
        };
        assert_eq!(index_fn_size(&idx), 60);
    }

    #[test]
    fn index_fn_dim_classifies_correctly() {
        assert_eq!(
            index_fn_dim(&IndexFn::Lattice {
                axis_sizes: vec![3, 4]
            }),
            2
        );
        assert_eq!(index_fn_dim(&IndexFn::Lockstep { length: 10 }), 1);
        assert_eq!(
            index_fn_dim(&IndexFn::Concatenation {
                segment_sizes: vec![1, 2, 3]
            }),
            1
        );
    }

    /// Every strategy over a zip or a union emits positions within the
    /// input, one axis as long as the input, and never fails.
    #[test]
    fn one_axis_inputs_select_within_their_length() {
        let inputs = [
            IndexFn::Modular {
                axis_sizes: vec![2, 7, 3],
            },
            IndexFn::Concatenation {
                segment_sizes: vec![2, 3, 4],
            },
            IndexFn::Lockstep { length: 9 },
        ];
        for idx in &inputs {
            let total = index_fn_size(idx);
            for name in [
                StrategyName::Lex,
                StrategyName::ReverseLex,
                StrategyName::Diagonal,
                StrategyName::Antidiagonal,
                StrategyName::Extrema,
                StrategyName::Shells,
                StrategyName::Halton,
                StrategyName::Sobol,
                StrategyName::Lhs,
                StrategyName::Shuffle,
            ] {
                let full: Vec<u64> = for_name(name)
                    .select(idx, total, None, None)
                    .iter()
                    .collect();
                let mut sorted = full.clone();
                sorted.sort_unstable();
                sorted.dedup();
                assert!(
                    full.iter().all(|p| *p < total),
                    "{name:?} over {idx:?}: {full:?}"
                );
                if !matches!(name, StrategyName::Halton | StrategyName::Sobol) {
                    assert_eq!(
                        sorted.len() as u64,
                        total,
                        "{name:?} over {idx:?} reaches every position: {full:?}"
                    );
                }
            }
        }
    }
}
