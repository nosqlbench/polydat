// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Metadata algebra (comprehension_forms.md §10.7).
//!
//! Every well-formed comprehension AST node carries a four-field
//! [`Metadata`] bundle computed bottom-up from its children's
//! metadata and its own scalar parameters. The bundle is a
//! monoid: propagation composes under composition, and every
//! field is either a closed enum (capability bit) or a
//! closed-form numeric/symbolic descriptor.
//!
//! This module owns:
//!
//! - [`Metadata`] — the four-field bundle.
//! - [`IndexFn`] — closed-form addressing schemes (six variants
//!   covering cartesian, zip Strict/Truncate, zip Cycle, union,
//!   continuous, hybrid).
//! - [`NaturalOrder`] — how a node enumerates by default.
//! - [`Materialization`] — streaming or sized-barrier
//!   classification (comprehension_forms.md §6.2).
//! - [`Comprehension::metadata`] — propagation entry point.
//!
//! The propagation rules are total, constant-time per node, and
//! cannot fail. Dependent-source cartesians produce
//! `index_addressable = None`; this is the **only** place
//! metadata propagation consults child-internal information
//! beyond the published bundles — and it does so at the
//! cartesian node, by walking the children's source expressions
//! for back-references to earlier-axis names.

use serde::{Deserialize, Serialize};

use super::ast::Comprehension;
use super::cardinality::{CardinalityClass, Hybrid, Interval, ProductMeasure};
use super::source::Source;
use super::strategy::{StrategyName, ZipMode};

/// The metadata bundle carried by every well-formed AST node.
///
/// Computed bottom-up; never mutated after propagation. Each
/// field is a closed enum or a closed-form descriptor — no
/// callbacks, no fail-able analyses.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Metadata {
    /// Cardinality class (comprehension_forms.md §6.1).
    pub cardinality: CardinalityClass,

    /// Closed-form bijection from `0..|c|` to the node's
    /// dispensed tuples. `None` when the node has no
    /// addressable index space (raw filter output, dependent
    /// cartesian, a truncated `Lex` order over either). An order
    /// other than an untruncated `Lex` is a one-axis `Lattice` of its
    /// selection: position `i` is the input's tuple at the `i`-th
    /// selected position.
    pub index_addressable: Option<IndexFn>,

    /// How this node enumerates by default.
    pub natural_order: NaturalOrder,

    /// Streaming-vs-barrier classification (comprehension_forms.md §6.2).
    pub materialization: Materialization,
}

/// Closed-form addressing schemes (comprehension_forms.md §10.7.1).
///
/// Six variants. Each describes the bijection from a
/// `0..cardinality` index range to the node's tuple shape.
/// `Continuous` and `Hybrid` carry the cardinality's
/// interval+measure descriptors directly so the R2 push-down
/// rule (comprehension_forms.md §10.2) dispatches on them without
/// recomputing.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum IndexFn {
    /// Discrete cartesian. `axis_sizes[i]` is the i-th axis's
    /// element count. Multi-index `(i₀, i₁, …)` maps to the
    /// per-axis tuple at those positions.
    Lattice {
        /// Element count per axis.
        axis_sizes: Vec<u64>,
    },

    /// Zip Strict / Truncate. One index `i ∈ 0..length` maps
    /// to the per-child tuple at position i.
    Lockstep {
        /// The common length.
        length: u64,
    },

    /// Zip Cycle. Modular addressing — index `i` maps to each
    /// child at `i mod child.cardinality`. At least one child
    /// must be bounded (the cycling target). The index range is
    /// [`cycle_length`] of the sizes: the longest child's, or empty
    /// when any child is empty.
    Modular {
        /// Element count per child.
        axis_sizes: Vec<u64>,
    },

    /// Union of index-addressable children. Index `i ∈
    /// 0..Σsegment_sizes` maps to segment k where k is the
    /// smallest such that `Σ₀^k segment_sizes > i`, position
    /// `i - Σ₀^{k-1} segment_sizes` within that segment.
    Concatenation {
        /// Element count per segment, in order.
        segment_sizes: Vec<u64>,
    },

    /// Continuous K-D box. Strategy push-down rules (Halton /
    /// Sobol / Lhs / Extrema on Continuous) draw from this
    /// directly; the discrete-to-continuous mapping is
    /// strategy-specific.
    Continuous {
        /// The interval of each axis.
        intervals: Vec<Interval>,
        /// The measure drawn from.
        measure: ProductMeasure,
    },

    /// Mixed discrete × continuous cartesian. Discrete axes get
    /// integer indexing; continuous axes get measure-weighted
    /// sampling. Strategy push-down dispatches per-axis.
    Hybrid {
        /// Element count per discrete axis.
        discrete_axes: Vec<u64>,
        /// The interval of each continuous axis.
        continuous_axes: Vec<Interval>,
        /// The measure over the continuous axes.
        measure: ProductMeasure,
    },
}

impl IndexFn {
    /// `true` if this index function carries any continuous
    /// axis. Used by per-strategy V4 checks to reject
    /// strategies that don't accept continuous inputs.
    pub fn has_continuous_axis(&self) -> bool {
        matches!(self, IndexFn::Continuous { .. } | IndexFn::Hybrid { .. })
    }

    /// `true` if this index function is a multi-axis Lattice
    /// (discrete cartesian with ≥2 axes). Required by
    /// lattice-geometric strategies (Extrema / Shells /
    /// Diagonal / Antidiagonal) for non-degenerate behavior.
    pub fn is_multi_axis_lattice(&self) -> bool {
        matches!(self, IndexFn::Lattice { axis_sizes } if axis_sizes.len() >= 2)
    }
}

/// Natural enumeration order (comprehension_forms.md §10.7.1).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum NaturalOrder {
    /// Lex order — rightmost axis varies fastest. Produced by
    /// cartesian, single-axis clause, and `order(_, Lex, _)`.
    Lex,

    /// Lockstep — zip's natural order. One tuple per i, all
    /// children at position i.
    Lockstep,

    /// Sequential — union's natural order. Drain child 0,
    /// then child 1, etc.
    Sequential,

    /// Strategy-driven — produced by `order(_, non-Lex, _)`.
    /// The wrapped strategy determines the emission order.
    Strategy(StrategyName),

    /// A continuous source that no sampling order wraps. V8
    /// refuses to dispense it.
    PendingSampling,
}

/// Streaming-vs-barrier classification (comprehension_forms.md §6.2).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Materialization {
    /// O(operator-local state) per pull; no input materialized.
    Streaming,

    /// Holds a finite working set; size declared at compile
    /// time. The two natural barriers (comprehension_forms.md §6.3):
    /// `zip(Cycle)` shorter children + non-Lex `order`.
    BoundedBarrier {
        /// Tuples the barrier holds at most.
        working_set_size: u64,
    },

    /// Working set is unbounded. Always V6-rejected per spec
    /// §5; this variant exists for representational
    /// completeness but should never propagate through to a
    /// valid AST's metadata.
    UnboundedBarrier,
}

impl Comprehension {
    /// Compute this node's metadata bundle (comprehension_forms.md
    /// §10.7.2).
    ///
    /// Bottom-up: every child's metadata is computed first,
    /// then this node's. Constant-time per node above the
    /// child cost. Total — never fails, never partial.
    ///
    /// For non-leaf nodes the metadata is recomputed on every
    /// call (no caching at this layer); consumers that need
    /// memoization wrap it externally. The propagation cost is
    /// O(N) in the nodes, and the optimizer re-propagates after
    /// each rewrite.
    pub fn metadata(&self) -> Metadata {
        match self {
            Comprehension::Clause { source, .. } => clause_metadata(source),
            Comprehension::Cartesian { children } => cartesian_metadata(children),
            Comprehension::Zip { children, mode } => zip_metadata(children, *mode),
            Comprehension::Union { children } => union_metadata(children),
            Comprehension::Filter { child, .. } => filter_metadata(child),
            Comprehension::Order {
                child,
                strategy,
                truncation,
                ..
            } => order_metadata(child, *strategy, *truncation),
        }
    }
}

fn clause_metadata(source: &Source) -> Metadata {
    let cardinality = source.cardinality();
    let (index_addressable, natural_order) = match &cardinality {
        CardinalityClass::Bounded(n) => (
            Some(IndexFn::Lattice {
                axis_sizes: vec![*n],
            }),
            NaturalOrder::Lex,
        ),
        CardinalityClass::Continuous { intervals, measure } => (
            Some(IndexFn::Continuous {
                intervals: intervals.clone(),
                measure: measure.clone(),
            }),
            NaturalOrder::PendingSampling,
        ),
        // BoundedAtMost / Unbounded / ContinuousAtMost — no
        // closed-form addressing function exists.
        _ => (None, NaturalOrder::Lex),
    };
    Metadata {
        cardinality,
        index_addressable,
        natural_order,
        materialization: Materialization::Streaming,
    }
}

fn cartesian_metadata(children: &[Comprehension]) -> Metadata {
    // First detect dependent sources: any child whose source
    // expression references an earlier child's coordinate name.
    // Dependent → index_addressable = None.
    let dependent = detect_dependent_sources(children);

    let child_meta: Vec<Metadata> = children.iter().map(|c| c.metadata()).collect();
    let cardinality = combine_cartesian_cardinality(&child_meta);

    let index_addressable = if dependent {
        None
    } else {
        combine_cartesian_index_fn(&child_meta)
    };

    let natural_order = if matches!(
        cardinality,
        CardinalityClass::Continuous { .. } | CardinalityClass::Hybrid(_)
    ) {
        NaturalOrder::PendingSampling
    } else {
        NaturalOrder::Lex
    };

    Metadata {
        cardinality,
        index_addressable,
        natural_order,
        materialization: Materialization::Streaming,
    }
}

fn zip_metadata(children: &[Comprehension], mode: ZipMode) -> Metadata {
    let child_meta: Vec<Metadata> = children.iter().map(|c| c.metadata()).collect();
    let cardinality = combine_zip_cardinality(&child_meta, mode);
    let index_addressable = combine_zip_index_fn(&child_meta, mode);

    let materialization = match mode {
        ZipMode::Strict | ZipMode::Truncate => Materialization::Streaming,
        ZipMode::Cycle => cycle_materialization(&cycle_operands(&child_meta)),
    };

    Metadata {
        cardinality,
        index_addressable,
        natural_order: NaturalOrder::Lockstep,
        materialization,
    }
}

/// The tuple count of a `zip(Cycle)` over operands of these counts:
/// the longest operand's, or zero when any operand is empty. Every
/// tuple binds every operand's names, and an empty operand has no
/// tuple to cycle.
pub fn cycle_length(counts: &[u64]) -> u64 {
    if counts.contains(&0) {
        0
    } else {
        counts.iter().copied().max().unwrap_or(0)
    }
}

/// `true` when metadata alone shows the operand yields no tuple.
fn known_empty(m: &Metadata) -> bool {
    matches!(
        m.cardinality,
        CardinalityClass::Bounded(0) | CardinalityClass::BoundedAtMost(0)
    )
}

/// How a `zip(Cycle)` holds one operand while it cycles (comprehension_forms.md §6.2,
/// §6.3).
///
/// Cycling re-emits an operand's earlier tuples once it is exhausted
/// and a longer operand is not. An index-addressable operand is read
/// at `i mod |operand|` directly and holds nothing; one operand that
/// is not addressable streams, and is restarted when it runs out
/// before the zip does; every other operand that is not addressable
/// is buffered in full. An operand found empty empties the zip, so an
/// executor checks the indexed operands' lengths and drains the
/// buffered operands in ascending bound before it holds any tuple,
/// and stops at the first empty one.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CycleOperand {
    /// Read at `i mod |operand|` through its index function.
    Indexed,
    /// Pulled once per tuple, and restarted when it runs out before
    /// the zip does.
    Streamed,
    /// Held in full and replayed. A bound of zero marks an operand
    /// known empty: the executor drains it first and holds nothing.
    Buffered {
        /// Tuples the buffer holds at most; `None` when the operand's
        /// count is unknown before it is evaluated.
        bound: Option<u64>,
    },
}

/// The plan a `zip(Cycle)` over operands with these bundles executes:
/// an operand known empty is buffered with bound zero; any other
/// addressable discrete operand is [`CycleOperand::Indexed`]; of the
/// rest, the first whose count is unknown streams, or when every count
/// is known, the first with the largest bound; every other operand is
/// buffered.
pub fn cycle_operands(children: &[Metadata]) -> Vec<CycleOperand> {
    let indexed = |m: &Metadata| {
        !known_empty(m)
            && m.index_addressable
                .as_ref()
                .is_some_and(|idx| !idx.has_continuous_axis())
    };
    let bound = |m: &Metadata| match &m.cardinality {
        CardinalityClass::Bounded(n) | CardinalityClass::BoundedAtMost(n) => Some(*n),
        _ => None,
    };
    let rest: Vec<usize> = (0..children.len())
        .filter(|&i| !indexed(&children[i]) && !known_empty(&children[i]))
        .collect();
    let streamed = rest
        .iter()
        .copied()
        .find(|&i| bound(&children[i]).is_none())
        .or_else(|| {
            rest.iter()
                .copied()
                .rev()
                .max_by_key(|&i| bound(&children[i]))
        });
    children
        .iter()
        .enumerate()
        .map(|(i, m)| {
            if indexed(m) {
                CycleOperand::Indexed
            } else if Some(i) == streamed {
                CycleOperand::Streamed
            } else {
                CycleOperand::Buffered { bound: bound(m) }
            }
        })
        .collect()
}

/// `true` when `plan` holds an operand known empty, so the zip yields
/// no tuple and buffers nothing.
pub fn cycle_plan_is_empty(plan: &[CycleOperand]) -> bool {
    plan.contains(&CycleOperand::Buffered { bound: Some(0) })
}

/// A `zip(Cycle)`'s working set under `plan`: the buffered operands'
/// bounds summed, unbounded when one of them has no bound, and
/// streaming when nothing is buffered or an operand is known empty.
pub fn cycle_materialization(plan: &[CycleOperand]) -> Materialization {
    if cycle_plan_is_empty(plan) {
        return Materialization::Streaming;
    }
    let mut total: u64 = 0;
    let mut buffered = false;
    for operand in plan {
        if let CycleOperand::Buffered { bound } = operand {
            buffered = true;
            match bound {
                Some(n) => total = total.saturating_add(*n),
                None => return Materialization::UnboundedBarrier,
            }
        }
    }
    if buffered {
        Materialization::BoundedBarrier {
            working_set_size: total,
        }
    } else {
        Materialization::Streaming
    }
}

fn union_metadata(children: &[Comprehension]) -> Metadata {
    let child_meta: Vec<Metadata> = children.iter().map(|c| c.metadata()).collect();
    let cardinality = combine_union_cardinality(&child_meta);
    let index_addressable = combine_union_index_fn(&child_meta);
    Metadata {
        cardinality,
        index_addressable,
        natural_order: NaturalOrder::Sequential,
        materialization: Materialization::Streaming,
    }
}

fn filter_metadata(child: &Comprehension) -> Metadata {
    let child_meta = child.metadata();
    let cardinality = match &child_meta.cardinality {
        // Filtering nothing keeps exactly nothing.
        CardinalityClass::Bounded(0) | CardinalityClass::BoundedAtMost(0) => {
            CardinalityClass::Bounded(0)
        }
        CardinalityClass::Bounded(n) | CardinalityClass::BoundedAtMost(n) => {
            CardinalityClass::BoundedAtMost(*n)
        }
        CardinalityClass::Unbounded => CardinalityClass::Unbounded,
        CardinalityClass::Continuous { intervals, measure }
        | CardinalityClass::ContinuousAtMost {
            intervals,
            measure_at_most: measure,
        } => CardinalityClass::ContinuousAtMost {
            intervals: intervals.clone(),
            measure_at_most: measure.clone(),
        },
        CardinalityClass::Hybrid(h) => CardinalityClass::Hybrid(h.clone()),
    };
    Metadata {
        cardinality,
        index_addressable: None, // filter destroys the bijection
        natural_order: child_meta.natural_order,
        materialization: child_meta.materialization,
    }
}

fn order_metadata(
    child: &Comprehension,
    strategy: StrategyName,
    truncation: Option<u64>,
) -> Metadata {
    let child_meta = child.metadata();
    let cardinality = order_cardinality(child, &child_meta.cardinality, strategy, truncation);

    // An order's output is addressed through its selection: position `i`
    // is the input's tuple at the `i`-th selected position, so the
    // output is one axis as long as the selection (comprehension_forms.md
    // §3.6). The axis is sized by the order's count, or its bound.
    let selected = || match &cardinality {
        CardinalityClass::Bounded(n) | CardinalityClass::BoundedAtMost(n) => {
            Some(IndexFn::Lattice {
                axis_sizes: vec![*n],
            })
        }
        _ => None,
    };
    let (index_addressable, natural_order, materialization) = match strategy {
        StrategyName::Lex => (
            // An untruncated `Lex` passes its input through, positions
            // and all. A truncated one selects a prefix of an addressable
            // input's positions; over any other input it counts the
            // tuples as they stream and addresses nothing.
            match truncation {
                None => child_meta.index_addressable,
                Some(_) => child_meta.index_addressable.and_then(|_| selected()),
            },
            NaturalOrder::Lex,
            child_meta.materialization, // counter wrapper at most
        ),
        non_lex => {
            // Over an addressable input the strategy selects positions
            // and holds only its selection (R2); over any other input
            // the input is buffered in full first.
            let materialization = match &child_meta.index_addressable {
                Some(_) => Materialization::BoundedBarrier {
                    working_set_size: strategy_working_set(
                        non_lex,
                        &child_meta.index_addressable,
                        truncation,
                    ),
                },
                None => match &child_meta.cardinality {
                    CardinalityClass::Bounded(n) | CardinalityClass::BoundedAtMost(n) => {
                        Materialization::BoundedBarrier {
                            working_set_size: *n,
                        }
                    }
                    _ => Materialization::UnboundedBarrier,
                },
            };
            (selected(), NaturalOrder::Strategy(non_lex), materialization)
        }
    };

    Metadata {
        cardinality,
        index_addressable,
        natural_order,
        materialization,
    }
}

// ---- cardinality combinators ----

fn combine_cartesian_cardinality(children: &[Metadata]) -> CardinalityClass {
    let mut has_continuous = false;
    let mut has_discrete = false;
    let mut counts: Vec<Count> = Vec::new();
    let mut discrete_axes: Vec<u64> = Vec::new();
    let mut continuous_intervals: Vec<Interval> = Vec::new();
    let mut continuous_measures: Vec<ProductMeasure> = Vec::new();

    for m in children {
        match &m.cardinality {
            CardinalityClass::Bounded(n) | CardinalityClass::BoundedAtMost(n) => {
                has_discrete = true;
                discrete_axes.push(*n); // the count, or its upper bound
                counts.extend(Count::of(&m.cardinality));
            }
            CardinalityClass::Unbounded => {
                has_discrete = true;
                discrete_axes.push(0);
                counts.push(Count::Unknown);
            }
            CardinalityClass::Continuous { intervals, measure }
            | CardinalityClass::ContinuousAtMost {
                intervals,
                measure_at_most: measure,
            } => {
                has_continuous = true;
                continuous_intervals.extend(intervals.iter().cloned());
                continuous_measures.push(measure.clone());
            }
            CardinalityClass::Hybrid(h) => {
                has_continuous = true;
                has_discrete = true;
                discrete_axes.extend(h.discrete_axes.iter().copied());
                continuous_intervals.extend(h.continuous_axes.iter().cloned());
                continuous_measures.push(h.measure.clone());
            }
        }
    }

    if has_continuous && has_discrete {
        CardinalityClass::Hybrid(Hybrid {
            discrete_axes,
            continuous_axes: continuous_intervals,
            measure: simplify_measures(continuous_measures),
        })
    } else if has_continuous {
        CardinalityClass::Continuous {
            intervals: continuous_intervals,
            measure: simplify_measures(continuous_measures),
        }
    } else {
        cartesian_count(&counts).class()
    }
}

fn combine_cartesian_index_fn(children: &[Metadata]) -> Option<IndexFn> {
    // All children must be addressable for the cartesian to be.
    let all_addressable = children.iter().all(|m| m.index_addressable.is_some());
    if !all_addressable {
        return None;
    }

    let mut all_discrete = true;
    let mut all_continuous = true;
    let mut discrete_axes: Vec<u64> = Vec::new();
    let mut continuous_intervals: Vec<Interval> = Vec::new();
    let mut continuous_measures: Vec<ProductMeasure> = Vec::new();

    for m in children {
        match m.index_addressable.as_ref().unwrap() {
            IndexFn::Lattice { axis_sizes } => {
                all_continuous = false;
                discrete_axes.extend(axis_sizes.iter().copied());
            }
            IndexFn::Continuous { intervals, measure } => {
                all_discrete = false;
                continuous_intervals.extend(intervals.iter().cloned());
                continuous_measures.push(measure.clone());
            }
            IndexFn::Hybrid {
                discrete_axes: d,
                continuous_axes: c,
                measure,
            } => {
                all_discrete = false;
                all_continuous = false;
                discrete_axes.extend(d.iter().copied());
                continuous_intervals.extend(c.iter().cloned());
                continuous_measures.push(measure.clone());
            }
            // Lockstep / Modular / Concatenation — these don't
            // combine as cartesian axes (they're 1-D index
            // spaces of their own), and a cartesian of a zip or a
            // union has no addressing scheme here, so it has none.
            IndexFn::Lockstep { .. } | IndexFn::Modular { .. } | IndexFn::Concatenation { .. } => {
                return None;
            }
        }
    }

    if all_discrete {
        Some(IndexFn::Lattice {
            axis_sizes: discrete_axes,
        })
    } else if all_continuous {
        Some(IndexFn::Continuous {
            intervals: continuous_intervals,
            measure: simplify_measures(continuous_measures),
        })
    } else {
        Some(IndexFn::Hybrid {
            discrete_axes,
            continuous_axes: continuous_intervals,
            measure: simplify_measures(continuous_measures),
        })
    }
}

fn combine_zip_cardinality(children: &[Metadata], mode: ZipMode) -> CardinalityClass {
    // A continuous child is a V7 failure; its count is unknown here.
    let counts: Vec<Count> = children
        .iter()
        .map(|m| Count::of(&m.cardinality).unwrap_or(Count::Unknown))
        .collect();
    match mode {
        ZipMode::Strict => strict_zip_count(&counts),
        ZipMode::Truncate => truncate_zip_count(&counts),
        ZipMode::Cycle => cycle_zip_count(&counts),
    }
    .class()
}

// ---- tuple counts ----

/// A discrete tuple count as metadata knows it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Count {
    /// Exactly this many.
    Exact(u64),
    /// Between zero and this many.
    AtMost(u64),
    /// No known bound.
    Unknown,
}

impl Count {
    /// The count a discrete class states, `None` for a continuous one.
    /// At most zero is exactly zero.
    fn of(class: &CardinalityClass) -> Option<Self> {
        Some(match class {
            CardinalityClass::Bounded(n) | CardinalityClass::BoundedAtMost(n @ 0) => {
                Count::Exact(*n)
            }
            CardinalityClass::BoundedAtMost(n) => Count::AtMost(*n),
            CardinalityClass::Unbounded => Count::Unknown,
            _ => return None,
        })
    }

    fn bound(self) -> Option<u64> {
        match self {
            Count::Exact(n) | Count::AtMost(n) => Some(n),
            Count::Unknown => None,
        }
    }

    /// `n` exactly when every count it was combined from is exact, at
    /// most `n` otherwise.
    fn combined(n: u64, counts: &[Count]) -> Self {
        if counts.iter().all(|c| matches!(c, Count::Exact(_))) {
            Count::Exact(n)
        } else {
            Count::AtMost(n)
        }
    }

    fn class(self) -> CardinalityClass {
        match self {
            Count::Exact(n) | Count::AtMost(n @ 0) => CardinalityClass::Bounded(n),
            Count::AtMost(n) => CardinalityClass::BoundedAtMost(n),
            Count::Unknown => CardinalityClass::Unbounded,
        }
    }
}

/// A cartesian's count: exactly zero when an operand is exactly empty,
/// whatever the others; unknown when an operand's count is; otherwise
/// the product of the operands' counts, exact when every one is.
fn cartesian_count(counts: &[Count]) -> Count {
    if counts.contains(&Count::Exact(0)) {
        return Count::Exact(0);
    }
    let Some(bounds) = counts.iter().map(|c| c.bound()).collect::<Option<Vec<_>>>() else {
        return Count::Unknown;
    };
    Count::combined(bounds.into_iter().fold(1, u64::saturating_mul), counts)
}

/// A truncating zip's count: the shortest operand's. Exactly zero when
/// an operand is exactly empty; exact when every operand is; otherwise
/// at most the least bound, since an operand at most `m` or of unknown
/// count may end at any length before `m`. Unknown only when no operand
/// has a bound.
fn truncate_zip_count(counts: &[Count]) -> Count {
    if counts.contains(&Count::Exact(0)) {
        return Count::Exact(0);
    }
    match counts.iter().filter_map(|c| c.bound()).min() {
        Some(m) => Count::combined(m, counts),
        None => Count::Unknown,
    }
}

/// A strict zip's count when it yields: every operand's, so an exact
/// operand's count exactly; otherwise at most the least bound. Operands
/// that end apart fail the zip instead.
fn strict_zip_count(counts: &[Count]) -> Count {
    if let Some(exact) = counts.iter().find(|c| matches!(c, Count::Exact(_))) {
        return *exact;
    }
    match counts.iter().filter_map(|c| c.bound()).min() {
        Some(m) => Count::AtMost(m),
        None => Count::Unknown,
    }
}

/// A cycle zip's count ([`cycle_length`]): exactly zero when an operand
/// is exactly empty; unknown when an operand's count is; otherwise the
/// longest operand's, exact when every operand is exact, and at most
/// that when one is at most, since it may be empty at open.
fn cycle_zip_count(counts: &[Count]) -> Count {
    if counts.contains(&Count::Exact(0)) {
        return Count::Exact(0);
    }
    let Some(bounds) = counts.iter().map(|c| c.bound()).collect::<Option<Vec<_>>>() else {
        return Count::Unknown;
    };
    Count::combined(bounds.into_iter().max().unwrap_or(0), counts)
}

/// A union's count: the sum of its operands', exact when every one is,
/// unknown when one is.
fn union_count(counts: &[Count]) -> Count {
    let Some(bounds) = counts.iter().map(|c| c.bound()).collect::<Option<Vec<_>>>() else {
        return Count::Unknown;
    };
    Count::combined(bounds.into_iter().fold(0, u64::saturating_add), counts)
}

/// An order's count. `Lex` and the strategies that truncate by tuple
/// count (`reverse_lex`, `diagonal`, `antidiagonal`, `halton`, `sobol`,
/// `lhs`, `shuffle`) keep `min(count, n)` of a discrete input; `extrema`
/// and `shells` truncate by whole strata or shells, so they keep at
/// most the input's count. Over a continuous space a sampling strategy
/// draws `n` points, exactly `n` unless a filter in the space or a
/// discrete axis of inexact count may leave fewer; `extrema` takes
/// the strata of the box, each continuous axis contributing its two
/// ends.
fn order_cardinality(
    child: &Comprehension,
    child_class: &CardinalityClass,
    strategy: StrategyName,
    truncation: Option<u64>,
) -> CardinalityClass {
    let strata = matches!(strategy, StrategyName::Extrema | StrategyName::Shells);
    let Some(count) = Count::of(child_class) else {
        // A continuous space, sampled by a non-`Lex` order with a count
        // (V8); any other order over one is invalid and keeps its class.
        let Some(n) = truncation.filter(|_| !matches!(strategy, StrategyName::Lex)) else {
            return child_class.clone();
        };
        let mut space = SampledSpace::default();
        space.collect(child);
        if space.discrete.contains(&Count::Exact(0)) {
            return CardinalityClass::Bounded(0);
        }
        return if matches!(strategy, StrategyName::Extrema) {
            // `n` strata of the box: at most all of its corners.
            let mut axes = space.discrete;
            axes.extend(std::iter::repeat_n(Count::Exact(2), space.continuous));
            match cartesian_count(&axes) {
                Count::Exact(m) | Count::AtMost(m) => Count::AtMost(m),
                Count::Unknown => Count::Unknown,
            }
        } else if !space.filtered && space.discrete.iter().all(|c| matches!(c, Count::Exact(_))) {
            Count::Exact(n)
        } else {
            Count::AtMost(n)
        }
        .class();
    };
    match (count, truncation) {
        (Count::Exact(0), _) | (_, None) => count,
        (Count::Exact(c) | Count::AtMost(c), Some(_)) if strata => Count::AtMost(c),
        (Count::Exact(c), Some(n)) => Count::Exact(c.min(n)),
        (Count::AtMost(c), Some(n)) => Count::AtMost(c.min(n)),
        (Count::Unknown, Some(_)) if strata => Count::Unknown,
        (Count::Unknown, Some(n)) => Count::AtMost(n),
    }
    .class()
}

/// The axes an order over a continuous space samples, walked as the
/// runtime walks them: clauses through cartesians and filters, any
/// other node one discrete axis of its tuples.
#[derive(Default)]
struct SampledSpace {
    /// Each discrete axis's count.
    discrete: Vec<Count>,
    /// How many continuous axes.
    continuous: usize,
    /// Whether a filter sits between the order and its clauses.
    filtered: bool,
}

impl SampledSpace {
    fn collect(&mut self, c: &Comprehension) {
        match c {
            Comprehension::Clause { source, .. } => match source.cardinality() {
                CardinalityClass::Continuous { .. } => self.continuous += 1,
                class => self
                    .discrete
                    .push(Count::of(&class).unwrap_or(Count::Unknown)),
            },
            Comprehension::Cartesian { children } => {
                children.iter().for_each(|child| self.collect(child));
            }
            Comprehension::Filter { child, .. } => {
                self.filtered = true;
                self.collect(child);
            }
            other => self
                .discrete
                .push(Count::of(&other.metadata().cardinality).unwrap_or(Count::Unknown)),
        }
    }
}

fn combine_zip_index_fn(children: &[Metadata], mode: ZipMode) -> Option<IndexFn> {
    let all_addressable = children.iter().all(|m| m.index_addressable.is_some());
    if !all_addressable {
        return None;
    }
    let counts: Vec<u64> = children
        .iter()
        .filter_map(|m| match &m.cardinality {
            CardinalityClass::Bounded(n) | CardinalityClass::BoundedAtMost(n) => Some(*n),
            _ => None,
        })
        .collect();
    if counts.len() != children.len() {
        return None;
    }
    match mode {
        ZipMode::Strict | ZipMode::Truncate => {
            let length = match mode {
                ZipMode::Strict => counts[0],
                ZipMode::Truncate => *counts.iter().min().unwrap(),
                ZipMode::Cycle => unreachable!(),
            };
            Some(IndexFn::Lockstep { length })
        }
        ZipMode::Cycle => Some(IndexFn::Modular { axis_sizes: counts }),
    }
}

fn combine_union_cardinality(children: &[Metadata]) -> CardinalityClass {
    // A continuous child is a V9 failure; its count is unknown here.
    let counts: Vec<Count> = children
        .iter()
        .map(|m| Count::of(&m.cardinality).unwrap_or(Count::Unknown))
        .collect();
    union_count(&counts).class()
}

fn combine_union_index_fn(children: &[Metadata]) -> Option<IndexFn> {
    let all_addressable = children.iter().all(|m| m.index_addressable.is_some());
    if !all_addressable {
        return None;
    }
    let segment_sizes: Vec<u64> = children
        .iter()
        .filter_map(|m| match &m.cardinality {
            CardinalityClass::Bounded(n) | CardinalityClass::BoundedAtMost(n) => Some(*n),
            _ => None,
        })
        .collect();
    if segment_sizes.len() != children.len() {
        return None;
    }
    Some(IndexFn::Concatenation { segment_sizes })
}

// ---- supporting helpers ----

fn simplify_measures(measures: Vec<ProductMeasure>) -> ProductMeasure {
    match measures.len() {
        0 => ProductMeasure::Uniform,
        1 => measures.into_iter().next().unwrap(),
        _ => ProductMeasure::Product(measures),
    }
}

/// Strategy-specific working-set size for use as
/// `BoundedBarrier.working_set_size` over an addressable input: the
/// selection the strategy holds, since it reads tuples only at the
/// positions it selects (R2).
fn strategy_working_set(
    strategy: StrategyName,
    input: &Option<IndexFn>,
    truncation: Option<u64>,
) -> u64 {
    match (strategy, input, truncation) {
        // Halton / Sobol / Shuffle over an index-addressable
        // input + truncation: O(n) draws.
        (StrategyName::Halton, Some(_), Some(n))
        | (StrategyName::Sobol, Some(_), Some(n))
        | (StrategyName::Shuffle, Some(_), Some(n))
        | (StrategyName::ReverseLex, Some(_), Some(n)) => n,
        // Lhs: O(n * dim).
        (StrategyName::Lhs, Some(idx), Some(n)) => {
            let dim = lattice_dim(idx).max(1);
            n.saturating_mul(dim as u64)
        }
        // Extrema (comprehension_forms.md §3.6) and Shells rank every multi-index of
        // the input's index space before keeping the first strata or
        // shells, so they hold the whole index space.
        (StrategyName::Extrema, Some(idx), Some(_))
        | (StrategyName::Shells, Some(idx), Some(_)) => index_fn_cardinality(idx),
        // Diagonal / Antidiagonal walk the diagonals in order and stop
        // at `n`.
        (StrategyName::Diagonal, Some(_), Some(n))
        | (StrategyName::Antidiagonal, Some(_), Some(n)) => n,
        // No truncation: fall back to the input's cardinality.
        (_, Some(idx), None) => index_fn_cardinality(idx),
        // No addressable input: we can't compute a closed form;
        // use the naïve "input cardinality" placeholder so the
        // metadata still has a number (consumers should treat
        // this as a conservative upper bound).
        (_, None, Some(n)) => n,
        (_, None, None) => 0,
        // Lex with truncation over addressable input — counter
        // wrapper, working set equals output size.
        (StrategyName::Lex, Some(_), Some(n)) => n,
    }
}

fn lattice_dim(idx: &IndexFn) -> usize {
    match idx {
        IndexFn::Lattice { axis_sizes } => axis_sizes.len(),
        IndexFn::Continuous { intervals, .. } => intervals.len(),
        IndexFn::Hybrid {
            discrete_axes,
            continuous_axes,
            ..
        } => discrete_axes.len() + continuous_axes.len(),
        IndexFn::Lockstep { .. } | IndexFn::Modular { .. } | IndexFn::Concatenation { .. } => 1,
    }
}

fn index_fn_cardinality(idx: &IndexFn) -> u64 {
    match idx {
        IndexFn::Lattice { axis_sizes } => axis_sizes
            .iter()
            .copied()
            .fold(1u64, |a, b| a.saturating_mul(b)),
        IndexFn::Lockstep { length } => *length,
        IndexFn::Modular { axis_sizes } => cycle_length(axis_sizes),
        IndexFn::Concatenation { segment_sizes } => segment_sizes
            .iter()
            .copied()
            .fold(0u64, |a, b| a.saturating_add(b)),
        // Continuous index has no integer cardinality.
        IndexFn::Continuous { .. } | IndexFn::Hybrid { .. } => 0,
    }
}

/// Walk children's source expressions for back-references to
/// earlier-axis coordinate names. Used by cartesian metadata
/// propagation to detect dependent sources (comprehension_forms.md §3.2).
fn detect_dependent_sources(children: &[Comprehension]) -> bool {
    let mut prior_names: Vec<String> = Vec::new();
    for child in children {
        // First check if the child references any prior name in
        // its source(s).
        for name in collect_source_name_references(child) {
            if prior_names.contains(&name) {
                return true;
            }
        }
        // Then add this child's coordinates to the prior set.
        for n in child.coordinate_names() {
            if !prior_names.contains(&n) {
                prior_names.push(n);
            }
        }
    }
    false
}

/// Extract `{name}` interpolation references from source
/// expressions in a comprehension subtree. Sources that carry
/// raw strings (`Generator`, `WorkloadParamList`) are walked;
/// `Literal`, `IntRange`, `ContinuousInterval`, `Distribution`
/// contain no string references.
fn collect_source_name_references(c: &Comprehension) -> Vec<String> {
    let mut out = Vec::new();
    walk_source_refs(c, &mut out);
    out
}

fn walk_source_refs(c: &Comprehension, out: &mut Vec<String>) {
    match c {
        Comprehension::Clause { source, .. } => {
            extract_source_refs(source, out);
        }
        Comprehension::Cartesian { children }
        | Comprehension::Zip { children, .. }
        | Comprehension::Union { children } => {
            for c in children {
                walk_source_refs(c, out);
            }
        }
        Comprehension::Filter { child, .. } | Comprehension::Order { child, .. } => {
            walk_source_refs(child, out);
        }
    }
}

fn extract_source_refs(source: &Source, out: &mut Vec<String>) {
    let s = match source {
        Source::Generator { expr, .. } => expr.as_str(),
        Source::WorkloadParamList { name, .. } => name.as_str(),
        _ => return,
    };
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'{'
            && let Some(close) = s[i + 1..].find('}')
        {
            let name = s[i + 1..i + 1 + close].trim();
            if !name.is_empty()
                && name.chars().all(|c| c.is_alphanumeric() || c == '_')
                && !out.contains(&name.to_string())
            {
                out.push(name.to_string());
            }
            i += close + 2;
            continue;
        }
        i += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::comprehension::source::{LiteralValue, Source};

    fn clause(name: &str, vs: &[i64]) -> Comprehension {
        Comprehension::clause(
            name,
            Source::Literal {
                values: vs.iter().map(|n| LiteralValue::Int(*n)).collect(),
            },
        )
    }

    fn continuous_clause(name: &str) -> Comprehension {
        Comprehension::clause(
            name,
            Source::ContinuousInterval {
                interval: Interval::closed(0.0, 1.0),
                measure: ProductMeasure::Uniform,
            },
        )
    }

    #[test]
    fn clause_metadata_for_bounded_source() {
        let m = clause("k", &[1, 2, 3]).metadata();
        assert_eq!(m.cardinality, CardinalityClass::Bounded(3));
        assert_eq!(
            m.index_addressable,
            Some(IndexFn::Lattice {
                axis_sizes: vec![3]
            })
        );
        assert_eq!(m.natural_order, NaturalOrder::Lex);
        assert_eq!(m.materialization, Materialization::Streaming);
    }

    #[test]
    fn clause_metadata_for_continuous_source() {
        let m = continuous_clause("alpha").metadata();
        assert!(matches!(m.cardinality, CardinalityClass::Continuous { .. }));
        assert!(matches!(
            m.index_addressable,
            Some(IndexFn::Continuous { .. })
        ));
        assert_eq!(m.natural_order, NaturalOrder::PendingSampling);
        assert_eq!(m.materialization, Materialization::Streaming);
    }

    #[test]
    fn cartesian_metadata_combines_lattice_axes() {
        let c =
            Comprehension::cartesian(vec![clause("k", &[1, 2]), clause("limit", &[10, 20, 30])]);
        let m = c.metadata();
        assert_eq!(m.cardinality, CardinalityClass::Bounded(6));
        assert_eq!(
            m.index_addressable,
            Some(IndexFn::Lattice {
                axis_sizes: vec![2, 3]
            })
        );
        assert_eq!(m.natural_order, NaturalOrder::Lex);
    }

    #[test]
    fn cartesian_metadata_for_hybrid() {
        let c =
            Comprehension::cartesian(vec![clause("k", &[1, 2, 3, 4]), continuous_clause("theta")]);
        let m = c.metadata();
        match m.cardinality {
            CardinalityClass::Hybrid(h) => {
                assert_eq!(h.discrete_axes, vec![4]);
                assert_eq!(h.continuous_axes.len(), 1);
            }
            other => panic!("expected Hybrid, got {other:?}"),
        }
        assert!(matches!(m.index_addressable, Some(IndexFn::Hybrid { .. })));
        assert_eq!(m.natural_order, NaturalOrder::PendingSampling);
    }

    #[test]
    fn dependent_cartesian_produces_none_addressable() {
        // clause replicas references {k} from the prior clause.
        let dependent = Comprehension::cartesian(vec![
            clause("k", &[1, 2, 3]),
            Comprehension::clause(
                "replicas",
                Source::Generator {
                    expr: "range(0, 2 * {k})".into(),
                    cardinality_hint: Some(6),
                },
            ),
        ]);
        let m = dependent.metadata();
        assert!(m.index_addressable.is_none());
    }

    #[test]
    fn zip_strict_produces_lockstep_index_fn() {
        let c = Comprehension::zip(
            vec![clause("x", &[1, 2, 3]), clause("y", &[10, 20, 30])],
            ZipMode::Strict,
        );
        let m = c.metadata();
        assert_eq!(m.index_addressable, Some(IndexFn::Lockstep { length: 3 }));
        assert_eq!(m.natural_order, NaturalOrder::Lockstep);
        assert_eq!(m.materialization, Materialization::Streaming);
    }

    #[test]
    fn zip_cycle_produces_modular_index_fn_and_barrier() {
        let c = Comprehension::zip(
            vec![clause("k", &[1, 2, 3, 4, 5]), clause("color", &[1, 2, 3])],
            ZipMode::Cycle,
        );
        let m = c.metadata();
        match m.index_addressable {
            Some(IndexFn::Modular { axis_sizes }) => {
                assert_eq!(axis_sizes, vec![5, 3]);
            }
            other => panic!("expected Modular, got {other:?}"),
        }
        // Both operands are addressable: cycling reads the shorter one
        // at `i mod 3` and buffers nothing.
        assert_eq!(m.materialization, Materialization::Streaming);
    }

    fn unknown_count(name: &str) -> Comprehension {
        Comprehension::clause(
            name,
            Source::Generator {
                expr: "values({n})".into(),
                cardinality_hint: None,
            },
        )
    }

    /// With an operand of unknown count, that operand streams and every
    /// finite operand that is not addressable is buffered in full;
    /// addressable ones are indexed.
    #[test]
    fn zip_cycle_with_an_unknown_count_buffers_every_finite_unaddressable_operand() {
        let c = Comprehension::zip(
            vec![
                unknown_count("tick"),
                Comprehension::filter(clause("a", &(0..1000).collect::<Vec<_>>()), "{a} > 1"),
                Comprehension::filter(clause("b", &[1, 2, 3]), "{b} > 0"),
                clause("color", &[1, 2, 3]),
            ],
            ZipMode::Cycle,
        );
        let plan = cycle_operands(
            &match &c {
                Comprehension::Zip { children, .. } => children,
                _ => unreachable!(),
            }
            .iter()
            .map(Comprehension::metadata)
            .collect::<Vec<_>>(),
        );
        assert_eq!(
            plan,
            vec![
                CycleOperand::Streamed,
                CycleOperand::Buffered { bound: Some(1000) },
                CycleOperand::Buffered { bound: Some(3) },
                CycleOperand::Indexed,
            ]
        );
        assert_eq!(
            c.metadata().materialization,
            Materialization::BoundedBarrier {
                working_set_size: 1003
            }
        );
    }

    /// With every count known, the largest operand that is not
    /// addressable streams and the others are buffered.
    #[test]
    fn zip_cycle_streams_the_largest_unaddressable_operand() {
        let c = Comprehension::zip(
            vec![
                Comprehension::filter(clause("a", &[1, 2]), "{a} > 0"),
                Comprehension::filter(clause("b", &[1, 2, 3, 4]), "{b} > 0"),
                clause("k", &(0..100).collect::<Vec<_>>()),
            ],
            ZipMode::Cycle,
        );
        assert_eq!(
            c.metadata().materialization,
            Materialization::BoundedBarrier {
                working_set_size: 2
            }
        );
    }

    /// An operand known empty empties the zip: no tuple, no index, and
    /// nothing held, in any position and beside operands of unknown
    /// count.
    #[test]
    fn zip_cycle_with_an_operand_known_empty_is_empty() {
        let operands = || {
            vec![
                unknown_count("tick"),
                Comprehension::filter(clause("a", &[1, 2, 3]), "{a} > 1"),
                clause("color", &[1, 2, 3]),
            ]
        };
        let empties = [
            clause("e", &[]),
            Comprehension::filter(clause("e", &[]), "{e} > 1"),
        ];
        for empty in &empties {
            for at in 0..=3 {
                let mut children = operands();
                children.insert(at, empty.clone());
                let plan = cycle_operands(
                    &children
                        .iter()
                        .map(Comprehension::metadata)
                        .collect::<Vec<_>>(),
                );
                assert_eq!(plan[at], CycleOperand::Buffered { bound: Some(0) });
                assert!(cycle_plan_is_empty(&plan));
                let m = Comprehension::zip(children, ZipMode::Cycle).metadata();
                assert_eq!(m.cardinality, CardinalityClass::Bounded(0));
                assert_eq!(m.materialization, Materialization::Streaming);
            }
        }
        let addressable = Comprehension::zip(
            vec![clause("k", &[1, 2, 3]), clause("e", &[])],
            ZipMode::Cycle,
        );
        let m = addressable.metadata();
        assert_eq!(
            m.index_addressable,
            Some(IndexFn::Modular {
                axis_sizes: vec![3, 0]
            })
        );
        assert_eq!(
            index_fn_cardinality(m.index_addressable.as_ref().unwrap()),
            0
        );
        assert_eq!(cycle_length(&[3, 0]), 0);
        assert_eq!(cycle_length(&[3, 5]), 5);
    }

    /// An operand that may be empty at open makes the zip's count an
    /// upper bound.
    #[test]
    fn zip_cycle_over_an_operand_at_most_counts_at_most() {
        let c = Comprehension::zip(
            vec![
                clause("k", &[1, 2, 3, 4, 5]),
                Comprehension::filter(clause("a", &[1, 2]), "{a} > 1"),
            ],
            ZipMode::Cycle,
        );
        assert_eq!(c.metadata().cardinality, CardinalityClass::BoundedAtMost(5));
    }

    /// The tuple counts an operand of count `c` may have: an exact
    /// count its own, an at-most count every count up to its bound, an
    /// unknown count small and large ones.
    fn witnesses(c: Count) -> Vec<u64> {
        match c {
            Count::Exact(n) => vec![n],
            Count::AtMost(n) => (0..=n).collect(),
            Count::Unknown => (0..=7).chain([1000]).collect(),
        }
    }

    /// Assert `claim` holds for every count the operator yields over
    /// the operands' witnesses, and is tight: exact means every yield is
    /// that count, at most means none exceeds the bound and one meets
    /// it, and unknown means some yield exceeds any bound the operands
    /// state.
    fn assert_describes(claim: Count, yields: &[u64], what: &str) {
        let Some(&max) = yields.iter().max() else {
            return; // the operator never yields, as a strict zip of unequal operands
        };
        match claim {
            Count::Exact(n) => assert!(yields.iter().all(|&y| y == n), "{what}: {yields:?}"),
            Count::AtMost(n) => {
                assert!(max <= n, "{what}: {yields:?} exceed {n}");
                assert_eq!(max, n, "{what}: the bound is not tight");
                assert!(n > 0, "{what}: at most zero is exactly zero");
                assert!(
                    yields.iter().any(|&y| y != n),
                    "{what}: always {n}, so exact"
                );
            }
            Count::Unknown => assert!(max >= 1000, "{what}: bounded by {max}"),
        }
    }

    const KINDS: [Count; 5] = [
        Count::Exact(0),
        Count::Exact(3),
        Count::Exact(5),
        Count::AtMost(4),
        Count::Unknown,
    ];

    /// Every combination of the operands' witnesses.
    fn combinations(counts: &[Count]) -> Vec<Vec<u64>> {
        counts.iter().fold(vec![Vec::new()], |acc, c| {
            acc.iter()
                .flat_map(|prefix| {
                    witnesses(*c).into_iter().map(move |w| {
                        let mut next = prefix.clone();
                        next.push(w);
                        next
                    })
                })
                .collect()
        })
    }

    /// Each operator's count over every pair and triple of kinds holds
    /// for, and is tight over, what the operator yields.
    #[test]
    fn every_kind_combination_counts_what_the_operator_yields() {
        let mut shapes: Vec<Vec<Count>> = Vec::new();
        for a in KINDS {
            for b in KINDS {
                shapes.push(vec![a, b]);
                for c in KINDS {
                    shapes.push(vec![a, b, c]);
                }
            }
        }
        for counts in &shapes {
            let combos = combinations(counts);
            let product: Vec<u64> = combos.iter().map(|c| c.iter().product()).collect();
            assert_describes(
                cartesian_count(counts),
                &product,
                &format!("cartesian {counts:?}"),
            );
            let sum: Vec<u64> = combos.iter().map(|c| c.iter().sum()).collect();
            assert_describes(union_count(counts), &sum, &format!("union {counts:?}"));
            let shortest: Vec<u64> = combos.iter().map(|c| *c.iter().min().unwrap()).collect();
            assert_describes(
                truncate_zip_count(counts),
                &shortest,
                &format!("truncate {counts:?}"),
            );
            let cycled: Vec<u64> = combos.iter().map(|c| cycle_length(c)).collect();
            assert_describes(
                cycle_zip_count(counts),
                &cycled,
                &format!("cycle {counts:?}"),
            );
            let strict: Vec<u64> = combos
                .iter()
                .filter(|c| c.iter().all(|&n| n == c[0]))
                .map(|c| c[0])
                .collect();
            assert_describes(
                strict_zip_count(counts),
                &strict,
                &format!("strict {counts:?}"),
            );
        }
    }

    fn filtered(c: Comprehension) -> Comprehension {
        Comprehension::filter(c, "true")
    }

    /// The combinators report a filtered operand's bound as a bound: a
    /// product and a truncating zip over one are at most, not exactly,
    /// their count.
    #[test]
    fn an_operand_at_most_makes_a_combination_at_most() {
        let at_most = || filtered(clause("k", &[1, 2, 3, 4, 5, 6, 7, 8, 9]));
        let colors = || clause("c", &[1, 2]);
        let product = Comprehension::cartesian(vec![at_most(), colors()]);
        assert_eq!(
            product.metadata().cardinality,
            CardinalityClass::BoundedAtMost(18)
        );
        let zip = Comprehension::zip(vec![at_most(), colors()], ZipMode::Truncate);
        assert_eq!(
            zip.metadata().cardinality,
            CardinalityClass::BoundedAtMost(2)
        );
        let zip = Comprehension::zip(vec![unknown_count("u"), colors()], ZipMode::Truncate);
        assert_eq!(
            zip.metadata().cardinality,
            CardinalityClass::BoundedAtMost(2)
        );
        let product = Comprehension::cartesian(vec![unknown_count("u"), clause("e", &[])]);
        assert_eq!(product.metadata().cardinality, CardinalityClass::Bounded(0));
        let empty = filtered(clause("e", &[]));
        assert_eq!(empty.metadata().cardinality, CardinalityClass::Bounded(0));
    }

    /// An order keeps `min(count, n)` under a strategy that truncates by
    /// tuple count, at most its input's count under one that truncates
    /// by strata, and `n` samples of a continuous space unless a filter
    /// may leave fewer.
    #[test]
    fn an_order_counts_by_its_strategy() {
        let order = |c, s, t| Comprehension::order(c, s, t).metadata().cardinality;
        let ks = || clause("k", &[1, 2, 3, 4, 5, 6]);
        for s in [
            StrategyName::Lex,
            StrategyName::ReverseLex,
            StrategyName::Diagonal,
            StrategyName::Halton,
            StrategyName::Sobol,
            StrategyName::Lhs,
            StrategyName::Shuffle,
        ] {
            assert_eq!(
                order(ks(), s, Some(4)),
                CardinalityClass::Bounded(4),
                "{s:?}"
            );
            assert_eq!(
                order(ks(), s, Some(9)),
                CardinalityClass::Bounded(6),
                "{s:?}"
            );
            assert_eq!(order(ks(), s, None), CardinalityClass::Bounded(6), "{s:?}");
            assert_eq!(
                order(filtered(ks()), s, Some(4)),
                CardinalityClass::BoundedAtMost(4),
                "{s:?}"
            );
            assert_eq!(
                order(unknown_count("u"), s, Some(4)),
                CardinalityClass::BoundedAtMost(4),
                "{s:?}"
            );
        }
        for s in [StrategyName::Extrema, StrategyName::Shells] {
            assert_eq!(
                order(ks(), s, Some(1)),
                CardinalityClass::BoundedAtMost(6),
                "{s:?}"
            );
            assert_eq!(order(ks(), s, None), CardinalityClass::Bounded(6), "{s:?}");
            assert_eq!(
                order(clause("e", &[]), s, Some(1)),
                CardinalityClass::Bounded(0)
            );
        }
        let space = || Comprehension::cartesian(vec![clause("k", &[1, 2]), continuous_clause("u")]);
        assert_eq!(
            order(space(), StrategyName::Halton, Some(5)),
            CardinalityClass::Bounded(5)
        );
        assert_eq!(
            order(filtered(space()), StrategyName::Halton, Some(5)),
            CardinalityClass::BoundedAtMost(5)
        );
        assert_eq!(
            order(space(), StrategyName::Extrema, Some(1)),
            CardinalityClass::BoundedAtMost(4)
        );
        let empty_axis = Comprehension::cartesian(vec![clause("k", &[]), continuous_clause("u")]);
        assert_eq!(
            order(empty_axis, StrategyName::Sobol, Some(5)),
            CardinalityClass::Bounded(0)
        );
    }

    /// Two operands of unknown count: one streams, the other has no
    /// bound to buffer.
    #[test]
    fn zip_cycle_with_two_unknown_counts_is_unbounded() {
        let c = Comprehension::zip(vec![unknown_count("x"), unknown_count("y")], ZipMode::Cycle);
        assert_eq!(
            c.metadata().materialization,
            Materialization::UnboundedBarrier
        );
    }

    #[test]
    fn union_produces_concatenation_index_fn() {
        let a = Comprehension::cartesian(vec![clause("k", &[1, 2]), clause("limit", &[10])]);
        let b = Comprehension::cartesian(vec![clause("k", &[3, 4]), clause("limit", &[20])]);
        let u = Comprehension::union(vec![a, b]);
        let m = u.metadata();
        assert_eq!(m.cardinality, CardinalityClass::Bounded(4));
        assert_eq!(
            m.index_addressable,
            Some(IndexFn::Concatenation {
                segment_sizes: vec![2, 2]
            })
        );
        assert_eq!(m.natural_order, NaturalOrder::Sequential);
    }

    #[test]
    fn filter_destroys_addressability() {
        let inner =
            Comprehension::cartesian(vec![clause("k", &[1, 2]), clause("limit", &[10, 20])]);
        let filtered = Comprehension::filter(inner, "{k} > 0");
        let m = filtered.metadata();
        assert_eq!(m.cardinality, CardinalityClass::BoundedAtMost(4));
        assert_eq!(m.index_addressable, None);
    }

    /// An untruncated `Lex` order passes its input's addressing through;
    /// a truncated one selects a prefix of its input's positions, one
    /// axis as long as the prefix, and over a filter it addresses
    /// nothing.
    #[test]
    fn lex_order_inherits_addressability_untruncated() {
        let inner =
            Comprehension::cartesian(vec![clause("k", &[1, 2]), clause("limit", &[10, 20])]);
        let whole = Comprehension::order(inner.clone(), StrategyName::Lex, None).metadata();
        assert_eq!(whole.cardinality, CardinalityClass::Bounded(4));
        assert_eq!(
            whole.index_addressable,
            Some(IndexFn::Lattice {
                axis_sizes: vec![2, 2]
            })
        );
        assert_eq!(whole.natural_order, NaturalOrder::Lex);
        let prefix = Comprehension::order(inner.clone(), StrategyName::Lex, Some(3)).metadata();
        assert_eq!(prefix.cardinality, CardinalityClass::Bounded(3));
        assert_eq!(
            prefix.index_addressable,
            Some(IndexFn::Lattice {
                axis_sizes: vec![3]
            })
        );
        assert_eq!(prefix.natural_order, NaturalOrder::Lex);
        let streamed = Comprehension::order(
            Comprehension::filter(inner, "{k} > 1"),
            StrategyName::Lex,
            Some(3),
        )
        .metadata();
        assert_eq!(streamed.index_addressable, None);
    }

    /// Any other order addresses its output through its selection: one
    /// axis as long as the selection, over which the next order holds
    /// only its own selection.
    #[test]
    fn non_lex_order_addresses_its_selection() {
        let inner =
            Comprehension::cartesian(vec![clause("k", &[1, 2]), clause("limit", &[10, 20])]);
        let ordered = Comprehension::order(inner, StrategyName::Halton, Some(2));
        let m = ordered.metadata();
        assert_eq!(
            m.index_addressable,
            Some(IndexFn::Lattice {
                axis_sizes: vec![2]
            })
        );
        let reordered = Comprehension::order(ordered.clone(), StrategyName::Shuffle, None);
        let r = reordered.metadata();
        assert_eq!(r.cardinality, CardinalityClass::Bounded(2));
        assert_eq!(
            r.index_addressable,
            Some(IndexFn::Lattice {
                axis_sizes: vec![2]
            })
        );
        assert_eq!(
            r.materialization,
            Materialization::BoundedBarrier {
                working_set_size: 2
            }
        );
        match m.natural_order {
            NaturalOrder::Strategy(StrategyName::Halton) => {}
            other => panic!("expected Strategy(Halton), got {other:?}"),
        }
        assert_eq!(
            m.materialization,
            Materialization::BoundedBarrier {
                working_set_size: 2
            }
        );
    }

    #[test]
    fn continuous_sampling_yields_bounded_cardinality() {
        let inner =
            Comprehension::cartesian(vec![continuous_clause("alpha"), continuous_clause("beta")]);
        let ordered = Comprehension::order(inner, StrategyName::Halton, Some(100));
        let m = ordered.metadata();
        assert_eq!(m.cardinality, CardinalityClass::Bounded(100));
        assert_eq!(
            m.materialization,
            Materialization::BoundedBarrier {
                working_set_size: 100
            }
        );
    }

    #[test]
    fn metadata_propagation_is_idempotent() {
        let c = Comprehension::order(
            Comprehension::filter(
                Comprehension::cartesian(vec![clause("k", &[1, 2, 3]), clause("limit", &[10, 20])]),
                "{k} * {limit} > 5",
            ),
            StrategyName::Halton,
            Some(5),
        );
        let m1 = c.metadata();
        let m2 = c.metadata();
        assert_eq!(m1, m2);
    }

    #[test]
    fn has_continuous_axis_classifier() {
        let lat = IndexFn::Lattice {
            axis_sizes: vec![3, 4],
        };
        assert!(!lat.has_continuous_axis());

        let cont = IndexFn::Continuous {
            intervals: vec![Interval::closed(0.0, 1.0)],
            measure: ProductMeasure::Uniform,
        };
        assert!(cont.has_continuous_axis());
    }

    #[test]
    fn multi_axis_lattice_classifier() {
        assert!(
            IndexFn::Lattice {
                axis_sizes: vec![3, 4]
            }
            .is_multi_axis_lattice()
        );
        assert!(
            !IndexFn::Lattice {
                axis_sizes: vec![3]
            }
            .is_multi_axis_lattice()
        );
        assert!(
            !IndexFn::Continuous {
                intervals: vec![Interval::closed(0.0, 1.0), Interval::closed(0.0, 1.0)],
                measure: ProductMeasure::Uniform,
            }
            .is_multi_axis_lattice()
        );
    }
}
