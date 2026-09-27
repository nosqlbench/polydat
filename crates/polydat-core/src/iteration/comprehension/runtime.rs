// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Runtime evaluator — walks an algebra [`Comprehension`] AST
//! against a [`Lookup`] scope (a `PolydatKernel` or a `Layered`
//! view) to produce typed coordinate tuples.
//!
//! ## Why this is separate from the static IR interpreter
//!
//! The static IR interpreter (`super::ir::interpreter`) walks
//! a compiled stack-machine program over fully-statically-
//! resolvable `Source` variants (`IntRange`, `Literal`). It
//! has no notion of a runtime parent kernel, which is correct
//! for the spec §9.5 consumption surfaces it serves.
//!
//! Runtime comprehension evaluation is fundamentally different:
//!
//! - **Source-text evaluation requires the parent kernel.**
//!   `Source::Generator { expr: "pre_{outer}" }` and
//!   `Source::WorkloadParamList { name }` resolve against the
//!   parent kernel's chain — `{outer}` substitution via
//!   [`interpolate_via_kernel`](crate::kernel::interp::interpolate_via_kernel), `kernel.lookup(name)` for
//!   workload params.
//! - **Cartesian is dependent-tuple, not independent.** Clause
//!   N's spec text may reference iter-vars from clauses
//!   1..N-1. Prior-axis values are layered in front of the
//!   scope (`Layered`) so each clause evaluates against the
//!   correct context. This is SRD-18b §"Dependent Tuple
//!   Iteration".
//! - **Filter predicates evaluate against per-tuple scopes.**
//!   A [`CompiledPredicate`] parses the predicate once and tests
//!   each tuple: its boolean structure over the tuple's values
//!   directly, and any richer sub-expression interpolated against
//!   a `Layered` view of the scope and evaluated as a Polydat
//!   expression, charged to the scope's ledger.
//!
//! All three depend on polydat-side primitives that exist
//! today; this evaluator is the algebra-typed entry point for
//! them.
//!
//! ## What this owns
//!
//! [`evaluate_indexed`] is the public surface:
//! `(algebra AST + scope) → IndexedTuples`, the tuples addressed by
//! position and computed when asked for; [`evaluate_for_iteration`]
//! computes them all. The tuples carry polydat [`Value`]s, which the
//! caller binds into a kernel over the body's program, one per tuple.
//! [`evaluate_for_iteration_materialized`] is the reference the
//! index-addressed evaluator is held to: every node materializes.
//!
//! Order modifiers route through `Strategy::select` (spec
//! §10.7.8): each node returns its tuples paired with the
//! [`IndexFn`] they satisfy, and the Order node selects positions
//! from that shape and the tuple count. V4 fires at this site,
//! definitively.
//!
//! ## What this does NOT own
//!
//! - Per-iteration kernel construction. The evaluator returns
//!   tuples; the caller binds each into a kernel over the body's
//!   program — `TraversalStream` for a `for` statement, the
//!   renderer for a tile's projection.

#[cfg(test)]
use std::sync::Arc;

use crate::ast::Value;
use crate::iteration::comprehension::ast::Comprehension;
use crate::iteration::comprehension::cardinality::{Interval, ProductMeasure};
use crate::iteration::comprehension::eval_source::{EvalContext, SourceEval};
use crate::iteration::comprehension::measure::AxisMeasure;
use crate::iteration::comprehension::metadata::{IndexFn, cycle_length};
use crate::iteration::comprehension::predicate::CompiledPredicate;
use crate::iteration::comprehension::source::Source;
use crate::iteration::comprehension::strategies::{Selection, shape_input};
use crate::iteration::comprehension::strategy::StrategyName;
#[cfg(test)]
use crate::kernel::PolydatKernel;
use crate::kernel::interp::Lookup;

/// Runtime tuple type — polydat-Value-based to preserve Ext
/// typing (Partition / Json / etc.) through the iteration
/// pipeline. The algebra layer's
/// [`Tuple`](crate::iteration::comprehension::strategies::Tuple) uses
/// [`TupleValue`](crate::iteration::comprehension::strategies::TupleValue)
/// which is scalar-only; this `RuntimeTuple`
/// is what the executor actually wants for per-iteration
/// kernel binding via [`PolydatKernel::for_iteration`](crate::kernel::PolydatKernel::for_iteration).
pub type RuntimeTuple = Vec<(String, Value)>;

/// Per-node result of the runtime walker.
///
/// `tuples` is the materialized stream in source order
/// (matches the runtime walker's natural enumeration — head
/// axis varies slowest in cartesian, sequential in union,
/// lockstep in zip). `index_fn` is the addressing scheme the
/// stream satisfies; `None` when the stream is non-addressable
/// (filter output, dependent cartesian over context-required
/// sources whose actual shapes don't combine cleanly).
struct EvaluatedNode {
    tuples: Vec<RuntimeTuple>,
    index_fn: Option<IndexFn>,
}

/// Errors the runtime evaluator surfaces.
#[derive(Debug, Clone)]
pub enum RuntimeError {
    /// Source evaluation failed (interpolation error,
    /// eval_const_expr error, unsupported source shape, etc.).
    SourceEval {
        /// The clause's element name.
        var: String,
        /// The source text.
        source: String,
        /// The underlying reason.
        message: String,
    },
    /// Filter predicate evaluation failed.
    FilterEval {
        /// The predicate text.
        predicate: String,
        /// The underlying reason.
        message: String,
    },
    /// Strategy application failed.
    OrderEval {
        /// The strategy applied.
        strategy: StrategyName,
        /// The underlying reason.
        message: String,
    },
    /// V4 (spec §5) violation — strategy rejects the input's
    /// addressing shape at invocation time (spec §10.7.8).
    StrategyRejectsInput {
        /// The strategy applied.
        strategy: StrategyName,
        /// The input's addressing scheme, if one was claimed.
        index_fn: Option<IndexFn>,
    },
    /// The runtime evaluator encountered an algebra-AST shape
    /// it doesn't support (e.g., nested Filter under Order).
    UnsupportedShape(String),
    /// A strict zip's operands have different lengths. The traversal
    /// evaluators report it at open; a stream reports it when one
    /// operand ends before the others, after the tuples before it.
    ZipLengthMismatch {
        /// Each operand's tuple count, in operand order.
        lengths: Vec<u64>,
    },
}

impl std::fmt::Display for RuntimeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RuntimeError::SourceEval {
                var,
                source,
                message,
            } => {
                write!(f, "for_each clause '{var} in {source}': {message}")
            }
            RuntimeError::FilterEval { predicate, message } => {
                write!(f, "comprehension filter '{predicate}': {message}")
            }
            RuntimeError::OrderEval { strategy, message } => {
                write!(f, "order strategy {strategy:?}: {message}")
            }
            RuntimeError::StrategyRejectsInput { strategy, index_fn } => write!(
                f,
                "order strategy {strategy:?} rejects input shape {index_fn:?} \
                 (V4: per-strategy IndexFn contract; see spec §3.6's strategy table)"
            ),
            RuntimeError::UnsupportedShape(msg) => write!(f, "{msg}"),
            RuntimeError::ZipLengthMismatch { lengths } => {
                write!(f, "zip strict: child lengths differ ({lengths:?})")
            }
        }
    }
}

impl std::error::Error for RuntimeError {}

/// Evaluate a comprehension against a scope and produce the
/// typed coordinate-tuple list.
///
/// `scope` is where names resolve: the body's kernel with the
/// parent's cascaded wires, or any other [`Lookup`]. A clause with no
/// values yields no tuples, which is what an empty clause means.
///
/// This is [`evaluate_indexed`] with every tuple computed.
pub fn evaluate_for_iteration(
    comp: &Comprehension,
    scope: &dyn Lookup,
) -> Result<Vec<RuntimeTuple>, RuntimeError> {
    evaluate_indexed(comp, scope).map(|t| t.to_vec())
}

/// [`evaluate_for_iteration`], and what each leaf clause yielded on the
/// way ([`ClauseYield`]).
///
/// The counts come off the evaluation that already happened — the
/// evaluator stands at each clause holding its name, its source and its
/// values — so nothing is evaluated twice to produce them, and the
/// record is one entry per leaf rather than one per tuple.
///
/// This is how a host says something about an empty clause. Emptiness
/// stays a legal value here and the policy stays with the caller: read
/// the clauses whose `evaluations` is non-zero and whose `values` is
/// zero, and warn, fail, or ignore as that host requires. A clause the
/// construction could already count as empty is a validation warning
/// instead ([`super::validate::ValidationWarning::EmptySource`]).
pub fn evaluate_for_iteration_reported(
    comp: &Comprehension,
    scope: &dyn Lookup,
) -> Result<EvaluatedIteration, RuntimeError> {
    let mut state = EvalState::new(comp, scope);
    let (node, _) = state.index_node(comp, &[])?;
    Ok(EvaluatedIteration {
        tuples: IndexedTuples { node }.to_vec(),
        clauses: state.yields,
    })
}

/// Evaluate a comprehension against a scope to its tuples addressed by
/// position (comprehension_forms.md §10.2 R2).
///
/// Every source is evaluated here, so every source error surfaces
/// here, but a tuple is computed only when it is asked for. A clause
/// over a range holds its bounds, a clause over any other source holds
/// its values, an independent cartesian, a zip, and a union hold their
/// operands, and an order holds its operand and the positions its
/// strategy selected from the operand's shape. A node with no closed
/// form over its operands holds its tuples: a filter's survivors, a
/// cartesian whose sources reference an earlier axis, and an order
/// that samples a continuous space. `order halton/100` over a large
/// product therefore holds the product's axes and 100 positions, and
/// over a filter of that product it holds the axes and the positions of
/// the survivors it keeps.
///
/// The tuples are exactly those [`evaluate_for_iteration_materialized`]
/// produces, in the same order, and the clause yields it reports are
/// the same.
pub fn evaluate_indexed(
    comp: &Comprehension,
    scope: &dyn Lookup,
) -> Result<IndexedTuples, RuntimeError> {
    let mut state = EvalState::new(comp, scope);
    let (node, _) = state.index_node(comp, &[])?;
    Ok(IndexedTuples { node })
}

/// The reference evaluator: every node materializes its tuples, and a
/// cartesian evaluates each later axis once per tuple of the axes
/// before it.
///
/// [`evaluate_indexed`] must yield exactly these tuples in exactly
/// this order, and report the same clause yields; the equivalence
/// harness compares the two over every comprehension shape.
pub fn evaluate_for_iteration_materialized(
    comp: &Comprehension,
    scope: &dyn Lookup,
) -> Result<EvaluatedIteration, RuntimeError> {
    let mut state = EvalState::new(comp, scope);
    let tuples = state.evaluate_node(comp, &[])?.tuples;
    Ok(EvaluatedIteration {
        tuples,
        clauses: state.yields,
    })
}

/// A comprehension's tuples addressed by position: each tuple is
/// computed from its position when asked for ([`evaluate_indexed`]).
#[derive(Debug, Clone)]
pub struct IndexedTuples {
    node: Indexed,
}

impl IndexedTuples {
    /// How many tuples there are.
    pub fn len(&self) -> u64 {
        self.node.len()
    }

    /// Whether there are no tuples.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The tuple at position `i`, or `None` past the end.
    pub fn get(&self, i: u64) -> Option<RuntimeTuple> {
        if i >= self.len() {
            return None;
        }
        let mut out = RuntimeTuple::new();
        self.node.append_at(i, &mut out);
        Some(out)
    }

    /// Every tuple, in order.
    pub fn iter(&self) -> impl Iterator<Item = RuntimeTuple> + '_ {
        (0..self.len()).filter_map(|i| self.get(i))
    }

    /// Every tuple, computed now.
    pub fn to_vec(&self) -> Vec<RuntimeTuple> {
        self.iter().collect()
    }
}

/// One node of an evaluated comprehension, answering the tuple at a
/// position.
#[derive(Debug, Clone)]
enum Indexed {
    /// Materialized tuples: a node with no closed form over its
    /// operands.
    Tuples(Vec<RuntimeTuple>),
    /// One clause's values.
    Clause { name: String, values: ClauseValues },
    /// An independent cartesian: the children's tuples at the
    /// mixed-radix digits of the position, the last child least
    /// significant.
    Product {
        children: Vec<Indexed>,
        lens: Vec<u64>,
        len: u64,
    },
    /// A strict or truncating zip: every child's tuple at the position.
    Lockstep { children: Vec<Indexed>, len: u64 },
    /// A cycle zip: every child's tuple at the position modulo its
    /// length. An empty child empties the zip, so no position reaches
    /// one.
    Cycle { children: Vec<Indexed>, len: u64 },
    /// A union: the child whose segment holds the position.
    Concat { children: Vec<Indexed>, len: u64 },
    /// An order: the child's tuple at the selected position.
    Select {
        child: Box<Indexed>,
        selection: Selection,
    },
}

/// A clause's values, each computed from its position.
#[derive(Debug, Clone)]
enum ClauseValues {
    /// `lo, lo + step, …`, `len` of them: an integer range.
    Range { lo: i64, step: i64, len: u64 },
    /// Any other source's evaluated values.
    List(Vec<Value>),
}

impl ClauseValues {
    fn len(&self) -> u64 {
        match self {
            ClauseValues::Range { len, .. } => *len,
            ClauseValues::List(values) => values.len() as u64,
        }
    }

    fn at(&self, i: u64) -> Value {
        match self {
            // A range's values are unsigned coordinates, as its
            // evaluated source gives them.
            ClauseValues::Range { lo, step, .. } => {
                Value::U64((i128::from(*lo) + i128::from(i) * i128::from(*step)) as i64 as u64)
            }
            ClauseValues::List(values) => values[i as usize].clone(),
        }
    }
}

impl Indexed {
    fn len(&self) -> u64 {
        match self {
            Indexed::Tuples(tuples) => tuples.len() as u64,
            Indexed::Clause { values, .. } => values.len(),
            Indexed::Product { len, .. }
            | Indexed::Lockstep { len, .. }
            | Indexed::Cycle { len, .. }
            | Indexed::Concat { len, .. } => *len,
            Indexed::Select { selection, .. } => selection.len(),
        }
    }

    /// Append the bindings of the tuple at `i`, which is below
    /// [`Self::len`], to `out`.
    fn append_at(&self, i: u64, out: &mut RuntimeTuple) {
        match self {
            Indexed::Tuples(tuples) => out.extend(tuples[i as usize].iter().cloned()),
            Indexed::Clause { name, values } => out.push((name.clone(), values.at(i))),
            Indexed::Product { children, lens, .. } => {
                let mut digits = vec![0u64; lens.len()];
                let mut rest = i;
                for (d, len) in digits.iter_mut().zip(lens).rev() {
                    *d = rest % len;
                    rest /= len;
                }
                for (child, d) in children.iter().zip(digits) {
                    child.append_at(d, out);
                }
            }
            Indexed::Lockstep { children, .. } => {
                for child in children {
                    child.append_at(i, out);
                }
            }
            Indexed::Cycle { children, .. } => {
                for child in children {
                    child.append_at(i % child.len(), out);
                }
            }
            Indexed::Concat { children, .. } => {
                let mut offset = i;
                for child in children {
                    let len = child.len();
                    if offset < len {
                        child.append_at(offset, out);
                        return;
                    }
                    offset -= len;
                }
            }
            Indexed::Select { child, selection } => {
                if let Some(p) = selection.get(i) {
                    child.append_at(p, out);
                }
            }
        }
    }
}

/// Internal walker state — the scope the recursive walker resolves
/// names against, so it does not thread it through every call.
/// What one leaf clause yielded over a whole traversal.
///
/// A clause under a cartesian is evaluated once per outer tuple, so
/// emptiness is a property of the evaluations and not of the clause:
///
/// - `evaluations == 0` — never reached, because something outside it
///   was empty first. Not the cause; the cause is an earlier clause.
/// - `evaluations > 0 && values == 0` — reached and yielded nothing
///   every time. This is the clause a diagnostic should name.
/// - `values > 0` — yielded.
///
/// An empty stream is a legal value of the algebra, so none of these
/// is an error. What a host does about one is the host's policy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClauseYield {
    /// The clause's element name.
    pub var: String,
    /// The source's canonical text, when it has one.
    pub source: Option<String>,
    /// How many times this clause was evaluated.
    pub evaluations: usize,
    /// How many values it produced, summed over those evaluations.
    pub values: usize,
}

/// A traversal's tuples, and what each leaf clause yielded reaching
/// them ([`ClauseYield`]), in the tree order of the comprehension.
#[derive(Debug, Clone)]
pub struct EvaluatedIteration {
    /// The coordinate tuples, as [`evaluate_for_iteration`] returns.
    pub tuples: Vec<RuntimeTuple>,
    /// Every leaf clause of the comprehension, evaluated or not.
    pub clauses: Vec<ClauseYield>,
}

struct EvalState<'a> {
    /// Where names resolve: the body's kernel with the parent's cascaded
    /// wires bound; a tuple's own bindings are layered in front per use.
    scope: &'a dyn Lookup,
    /// Per leaf clause, what it yielded. Seeded from the tree before
    /// evaluation so a clause never reached is present with zero
    /// evaluations rather than absent.
    yields: Vec<ClauseYield>,
    /// Leaf identity → index into `yields`. Keyed by the address of the
    /// clause's own `Source` inside the borrowed tree, which is what
    /// distinguishes two clauses that share a name across the branches
    /// of a union.
    by_leaf: std::collections::HashMap<usize, usize>,
    /// How many evaluations the node being evaluated stands for. The
    /// reference evaluator evaluates a cartesian's later axis once per
    /// tuple of the axes before it; the index-addressed evaluator
    /// evaluates an independent axis once, which stands for that many.
    mult: usize,
}

impl<'a> EvalState<'a> {
    /// Seed one entry per leaf clause, in tree order.
    fn new(comp: &Comprehension, scope: &'a dyn Lookup) -> Self {
        let mut state = EvalState {
            scope,
            yields: Vec::new(),
            by_leaf: std::collections::HashMap::new(),
            mult: 1,
        };
        state.enumerate_leaves(comp);
        state
    }

    fn enumerate_leaves(&mut self, node: &Comprehension) {
        match node {
            Comprehension::Clause { name, source } => {
                self.by_leaf
                    .insert(std::ptr::from_ref(source) as usize, self.yields.len());
                self.yields.push(ClauseYield {
                    var: name.clone(),
                    source: source.to_text(),
                    evaluations: 0,
                    values: 0,
                });
            }
            Comprehension::Cartesian { children }
            | Comprehension::Zip { children, .. }
            | Comprehension::Union { children } => {
                for child in children {
                    self.enumerate_leaves(child);
                }
            }
            Comprehension::Filter { child, .. } | Comprehension::Order { child, .. } => {
                self.enumerate_leaves(child);
            }
        }
    }

    /// Record the evaluations of a leaf and the values each produced:
    /// one evaluation standing for `mult` identical ones.
    fn record_yield(&mut self, source: &Source, values: usize) {
        if let Some(&i) = self.by_leaf.get(&(std::ptr::from_ref(source) as usize)) {
            self.yields[i].evaluations = self.yields[i].evaluations.saturating_add(self.mult);
            self.yields[i].values = self.yields[i]
                .values
                .saturating_add(values.saturating_mul(self.mult));
        }
    }
}

impl EvalState<'_> {
    fn evaluate_node(
        &mut self,
        node: &Comprehension,
        prefix: &[(String, Value)],
    ) -> Result<EvaluatedNode, RuntimeError> {
        match node {
            Comprehension::Clause { name, source } => self.evaluate_clause(name, source, prefix),
            Comprehension::Cartesian { children } => self.evaluate_cartesian(children, prefix),
            Comprehension::Zip { children, mode } => self.evaluate_zip(children, *mode, prefix),
            Comprehension::Union { children } => self.evaluate_union(children, prefix),
            Comprehension::Filter { child, predicate } => {
                let inner = self.evaluate_node(child, prefix)?;
                self.apply_filter(inner, predicate)
            }
            Comprehension::Order {
                child,
                strategy,
                truncation,
                seed,
            } => {
                // A strategy that selects from the shape reads through
                // the untruncated orders under it (§7.4 O1).
                let child = shape_input(child, *strategy);
                // A continuous axis has no tuples of its own: an order
                // over one samples the child's space, discrete axes by
                // position and continuous axes through their measures
                // (spec §10.2 R2).
                if has_continuous_axis(child) {
                    return self.sample_space(child, prefix, *strategy, *truncation, *seed);
                }
                // A non-`Lex` order over a filter ranks the survivors by
                // their positions in the filter's input (§5 V5).
                if let (Comprehension::Filter { child, predicate }, false) =
                    (child, *strategy == StrategyName::Lex)
                {
                    let inner = self.evaluate_node(child, prefix)?;
                    let predicate = CompiledPredicate::new(predicate);
                    let mut survivors = Vec::new();
                    for (p, tuple) in inner.tuples.iter().enumerate() {
                        if predicate.keeps(tuple, self.scope)? {
                            survivors.push(p as u64);
                        }
                    }
                    let selection = surviving_selection(
                        *strategy,
                        inner.index_fn.as_ref(),
                        inner.tuples.len() as u64,
                        *truncation,
                        *seed,
                        &survivors,
                    )?;
                    return Ok(EvaluatedNode {
                        tuples: selection
                            .iter()
                            .map(|p| inner.tuples[p as usize].clone())
                            .collect(),
                        index_fn: None,
                    });
                }
                let inner = self.evaluate_node(child, prefix)?;
                self.apply_order(inner, *strategy, *truncation, *seed)
            }
        }
    }

    /// Evaluate one clause's source against the prefix and record what
    /// it yielded.
    fn evaluate_source(
        &mut self,
        name: &str,
        source: &Source,
        prefix: &[(String, Value)],
    ) -> Result<crate::iteration::comprehension::eval_source::EvaluatedSource, RuntimeError> {
        let ctx = EvalContext {
            var_name: name,
            scope: self.scope,
            prefix,
        };
        let evaluated = source.evaluate(Some(&ctx)).map_err(|e| match e {
            crate::iteration::comprehension::eval_source::EvalError::EvalFailed {
                var,
                source,
                message,
            } => RuntimeError::SourceEval {
                var,
                source,
                message,
            },
            crate::iteration::comprehension::eval_source::EvalError::NeedsContext => {
                RuntimeError::UnsupportedShape(format!(
                    "clause '{name}': source requires kernel context but evaluator \
                             provided none — internal bug in runtime walker"
                ))
            }
        })?;
        self.record_yield(source, evaluated.values.len());
        Ok(evaluated)
    }

    fn evaluate_clause(
        &mut self,
        name: &str,
        source: &Source,
        prefix: &[(String, Value)],
    ) -> Result<EvaluatedNode, RuntimeError> {
        let evaluated = self.evaluate_source(name, source, prefix)?;

        if evaluated.values.is_empty() {
            // A clause with no values yields no tuples, which is what
            // an empty clause means: a legal value of the algebra, not
            // an error and not a policy decision made here. The
            // evaluation is counted above, so a host that wants to say
            // something about it reads `ClauseYield` off the result.
            return Ok(EvaluatedNode {
                tuples: Vec::new(),
                index_fn: Some(evaluated.index_fn),
            });
        }
        let tuples: Vec<RuntimeTuple> = evaluated
            .values
            .into_iter()
            .map(|v| vec![(name.to_string(), v)])
            .collect();
        Ok(EvaluatedNode {
            tuples,
            index_fn: Some(evaluated.index_fn),
        })
    }

    fn evaluate_cartesian(
        &mut self,
        children: &[Comprehension],
        prefix: &[(String, Value)],
    ) -> Result<EvaluatedNode, RuntimeError> {
        if children.is_empty() {
            return Ok(EvaluatedNode {
                tuples: vec![Vec::new()],
                index_fn: Some(IndexFn::Lattice {
                    axis_sizes: vec![1],
                }),
            });
        }
        let mut child_index_fns: Vec<(Option<IndexFn>, u64)> = Vec::with_capacity(children.len());
        let mut dependent_observed = false;
        let result_tuples = self.evaluate_cartesian_rec(
            children.len(),
            children,
            prefix,
            &mut child_index_fns,
            &mut dependent_observed,
        )?;

        // Combined Lattice from observed per-clause cardinalities.
        // Dependent cartesians produce children whose
        // per-clause cardinality varies with the prefix — we
        // can't claim a clean Lattice in that case, so the
        // combined index_fn is None.
        let combined = if dependent_observed {
            None
        } else {
            let index_fns: Vec<Option<IndexFn>> =
                child_index_fns.into_iter().map(|(idx, _)| idx).collect();
            combine_cartesian_index_fn(&index_fns)
        };
        Ok(EvaluatedNode {
            tuples: result_tuples,
            index_fn: combined,
        })
    }

    fn evaluate_cartesian_rec(
        &mut self,
        child_count: usize,
        children: &[Comprehension],
        prefix: &[(String, Value)],
        child_index_fns: &mut Vec<(Option<IndexFn>, u64)>,
        dependent_observed: &mut bool,
    ) -> Result<Vec<RuntimeTuple>, RuntimeError> {
        if children.is_empty() {
            return Ok(vec![Vec::new()]);
        }
        let (head, tail) = children.split_first().unwrap();
        let head_eval = self.evaluate_node(head, prefix)?;
        let head_axis_len = head_eval.tuples.len() as u64;
        // The head's position among the cartesian's children: the
        // first evaluation at a position records its index_fn and
        // tuple count, and a later evaluation with another count
        // (a dependent cartesian) leaves the product no lattice.
        let depth = child_count - children.len();
        match child_index_fns.get(depth) {
            None => child_index_fns.push((head_eval.index_fn.clone(), head_axis_len)),
            Some((_, first)) if *first != head_axis_len => *dependent_observed = true,
            Some(_) => {}
        }

        if tail.is_empty() {
            return Ok(head_eval.tuples);
        }
        let mut out = Vec::new();
        for head_tuple in head_eval.tuples {
            let mut extended_prefix: Vec<(String, Value)> = prefix.to_vec();
            extended_prefix.extend(head_tuple.iter().cloned());
            let tail_tuples = self.evaluate_cartesian_rec(
                child_count,
                tail,
                &extended_prefix,
                child_index_fns,
                dependent_observed,
            )?;
            for tail_tuple in tail_tuples {
                let mut merged = head_tuple.clone();
                merged.extend(tail_tuple);
                out.push(merged);
            }
        }
        Ok(out)
    }

    fn evaluate_zip(
        &mut self,
        children: &[Comprehension],
        mode: crate::iteration::comprehension::strategy::ZipMode,
        prefix: &[(String, Value)],
    ) -> Result<EvaluatedNode, RuntimeError> {
        use crate::iteration::comprehension::strategy::ZipMode;
        if children.is_empty() {
            return Ok(EvaluatedNode {
                tuples: vec![Vec::new()],
                index_fn: Some(IndexFn::Lockstep { length: 1 }),
            });
        }
        let per_child: Vec<EvaluatedNode> = children
            .iter()
            .map(|c| self.evaluate_node(c, prefix))
            .collect::<Result<_, _>>()?;
        let lengths: Vec<usize> = per_child.iter().map(|n| n.tuples.len()).collect();
        let iter_count = match mode {
            ZipMode::Strict => {
                let first = lengths.first().copied().unwrap_or(0);
                if lengths.iter().any(|&n| n != first) {
                    return Err(RuntimeError::ZipLengthMismatch {
                        lengths: lengths.iter().map(|&n| n as u64).collect(),
                    });
                }
                first
            }
            ZipMode::Truncate => lengths.iter().copied().min().unwrap_or(0),
            ZipMode::Cycle => {
                let counts: Vec<u64> = lengths.iter().map(|&n| n as u64).collect();
                cycle_length(&counts) as usize
            }
        };
        let mut tuples = Vec::with_capacity(iter_count);
        for i in 0..iter_count {
            let mut bindings: RuntimeTuple = Vec::new();
            for (child, &len) in per_child.iter().zip(lengths.iter()) {
                let idx = match mode {
                    ZipMode::Cycle => i % len,
                    _ => i,
                };
                bindings.extend(child.tuples[idx].iter().cloned());
            }
            tuples.push(bindings);
        }
        let index_fn = match mode {
            ZipMode::Strict | ZipMode::Truncate => Some(IndexFn::Lockstep {
                length: iter_count as u64,
            }),
            ZipMode::Cycle => Some(IndexFn::Modular {
                axis_sizes: lengths.iter().map(|n| *n as u64).collect(),
            }),
        };
        Ok(EvaluatedNode { tuples, index_fn })
    }

    fn evaluate_union(
        &mut self,
        children: &[Comprehension],
        prefix: &[(String, Value)],
    ) -> Result<EvaluatedNode, RuntimeError> {
        let mut tuples = Vec::new();
        let mut segment_sizes = Vec::with_capacity(children.len());
        let mut all_segments_addressable = true;
        for child in children {
            let sub = self.evaluate_node(child, prefix)?;
            segment_sizes.push(sub.tuples.len() as u64);
            if sub.index_fn.is_none() {
                all_segments_addressable = false;
            }
            tuples.extend(sub.tuples);
        }
        let index_fn = if all_segments_addressable {
            Some(IndexFn::Concatenation { segment_sizes })
        } else {
            None
        };
        Ok(EvaluatedNode { tuples, index_fn })
    }

    fn apply_filter(
        &mut self,
        input: EvaluatedNode,
        predicate: &str,
    ) -> Result<EvaluatedNode, RuntimeError> {
        let predicate = CompiledPredicate::new(predicate);
        let mut out = Vec::with_capacity(input.tuples.len());
        for tuple in input.tuples {
            if predicate.keeps(&tuple, self.scope)? {
                out.push(tuple);
            }
        }
        // Filter destroys the bijection per spec §10.7.2.
        Ok(EvaluatedNode {
            tuples: out,
            index_fn: None,
        })
    }

    /// Sample an order over a space with a continuous axis (spec
    /// §10.2 R2). The child's clauses are the axes: a discrete
    /// clause is evaluated to its values, a continuous clause keeps
    /// its interval and measure. A sampling strategy (Halton, Sobol,
    /// Lhs, Shuffle) draws `truncation` multi-indices over the
    /// `Continuous` or `Hybrid` index function of those axes, and
    /// each continuous coordinate is carried onto its interval by
    /// its measure; `Extrema` takes the strata of the box, a
    /// continuous axis contributing its two ends. Filters between
    /// the order and its clauses apply to the drawn tuples, and a
    /// sequence strategy keeps drawing until `truncation` tuples
    /// pass, up to a bounded number of rounds.
    fn sample_space(
        &mut self,
        child: &Comprehension,
        prefix: &[(String, Value)],
        strategy: StrategyName,
        truncation: Option<u64>,
        seed: Option<u64>,
    ) -> Result<EvaluatedNode, RuntimeError> {
        let mut space = SampleSpace::default();
        self.collect_sample_space(child, prefix, &mut space, &mut Vec::new())?;
        let discrete_axes: Vec<u64> = space
            .axes
            .iter()
            .filter_map(|a| match a {
                SampleAxis::Discrete(tuples) => Some(tuples.len() as u64),
                SampleAxis::Continuous { .. } => None,
            })
            .collect();
        if discrete_axes.contains(&0) {
            return Ok(EvaluatedNode {
                tuples: Vec::new(),
                index_fn: None,
            });
        }
        let (intervals, measures): (Vec<Interval>, Vec<ProductMeasure>) = space
            .axes
            .iter()
            .filter_map(|a| match a {
                SampleAxis::Continuous {
                    interval, measure, ..
                } => Some((
                    interval.clone(),
                    match measure {
                        AxisMeasure::Uniform => ProductMeasure::Uniform,
                        AxisMeasure::Named { name, .. } => ProductMeasure::Named(*name),
                    },
                )),
                SampleAxis::Discrete(_) => None,
            })
            .unzip();
        let sequence = !matches!(strategy, StrategyName::Extrema);
        // A sampling strategy lays a hybrid out discrete axes first
        // (`IndexFn::Hybrid`); Extrema's strata are lex in clause
        // order, so it takes a lattice in that order, a continuous
        // axis being its two ends.
        let index_fn = if !sequence {
            IndexFn::Lattice {
                axis_sizes: space
                    .axes
                    .iter()
                    .map(|a| match a {
                        SampleAxis::Discrete(tuples) => tuples.len() as u64,
                        SampleAxis::Continuous { .. } => 2,
                    })
                    .collect(),
            }
        } else if discrete_axes.is_empty() {
            IndexFn::Continuous {
                intervals,
                measure: ProductMeasure::Product(measures),
            }
        } else {
            IndexFn::Hybrid {
                discrete_axes,
                continuous_axes: intervals,
                measure: ProductMeasure::Product(measures),
            }
        };

        let mut want = truncation;
        let mut rounds = 0;
        loop {
            let multi_indices = draw_sample(&index_fn, strategy, want, seed)?;
            let drawn = multi_indices.len() as u64;
            let tuples = multi_indices
                .iter()
                .map(|mi| space.realize(mi, strategy))
                .collect();
            let mut node = EvaluatedNode {
                tuples,
                index_fn: None,
            };
            for predicate in &space.predicates {
                node = self.apply_filter(node, predicate)?;
            }
            let (Some(n), Some(asked)) = (truncation, want) else {
                return Ok(node);
            };
            let enough = node.tuples.len() as u64 >= n;
            let exhausted = drawn < asked;
            if !sequence {
                return Ok(node);
            }
            if enough || exhausted || rounds >= SAMPLE_ROUNDS {
                node.tuples.truncate(n as usize);
                return Ok(node);
            }

            want = Some(asked.saturating_mul(2));
            rounds += 1;
        }
    }

    /// Walk `c` into `space`: clauses become axes in order, filters
    /// contribute their predicates, and any other node (a zip, union,
    /// or inner order) is evaluated and becomes one discrete axis of
    /// its tuples. `bound` is the names bound so far; a discrete
    /// clause may not reference one (a sampled cartesian is
    /// independent, spec §6.2).
    fn collect_sample_space(
        &mut self,
        c: &Comprehension,
        prefix: &[(String, Value)],
        space: &mut SampleSpace,
        bound: &mut Vec<String>,
    ) -> Result<(), RuntimeError> {
        let measure_error = |name: &str, message: String| RuntimeError::SourceEval {
            var: name.to_string(),
            source: "<continuous>".to_string(),
            message,
        };
        match c {
            Comprehension::Clause {
                name,
                source: Source::ContinuousInterval { interval, measure },
            } => {
                let measure =
                    AxisMeasure::from_product(measure, 0).map_err(|m| measure_error(name, m))?;
                space.axes.push(SampleAxis::Continuous {
                    name: name.clone(),
                    interval: interval.clone(),
                    measure,
                });
                bound.push(name.clone());
            }
            Comprehension::Clause {
                name,
                source:
                    Source::Distribution {
                        distribution,
                        support,
                        params,
                    },
            } => {
                let measure = AxisMeasure::named(*distribution, params)
                    .map_err(|m| measure_error(name, m))?;
                space.axes.push(SampleAxis::Continuous {
                    name: name.clone(),
                    interval: support.clone(),
                    measure,
                });
                bound.push(name.clone());
            }
            Comprehension::Clause { name, source } => {
                let references = c.referenced_source_names();
                if let Some(dep) = bound.iter().find(|b| references.contains(*b)) {
                    return Err(RuntimeError::UnsupportedShape(format!(
                        "clause '{name}' references '{dep}' beside a continuous axis; \
                         a sampled cartesian is independent (comprehension_forms.md §6.2)"
                    )));
                }
                let node = self.evaluate_clause(name, source, prefix)?;
                space.axes.push(SampleAxis::Discrete(node.tuples));
                bound.push(name.clone());
            }
            Comprehension::Cartesian { children } => {
                for child in children {
                    self.collect_sample_space(child, prefix, space, bound)?;
                }
            }
            Comprehension::Filter { child, predicate } => {
                self.collect_sample_space(child, prefix, space, bound)?;
                space.predicates.push(predicate.clone());
            }
            Comprehension::Zip { .. }
            | Comprehension::Union { .. }
            | Comprehension::Order { .. } => {
                let node = self.evaluate_node(c, prefix)?;
                bound.extend(c.coordinate_names());
                space.axes.push(SampleAxis::Discrete(node.tuples));
            }
        }
        Ok(())
    }

    fn apply_order(
        &mut self,
        input: EvaluatedNode,
        strategy: StrategyName,
        truncation: Option<u64>,
        seed: Option<u64>,
    ) -> Result<EvaluatedNode, RuntimeError> {
        // The strategy selects positions from the input's shape
        // alone (its `IndexFn` and tuple count), so the chosen tuples
        // are the input's tuples at those positions.
        let selection = order_selection(
            strategy,
            input.index_fn.as_ref(),
            input.tuples.len() as u64,
            truncation,
            seed,
        )?;
        let out = selection
            .iter()
            .map(|p| input.tuples[p as usize].clone())
            .collect();
        // Order may produce a different index_fn (e.g., Lex
        // preserves; non-Lex destroys), but downstream
        // consumers of evaluate_for_iteration only read tuples.
        Ok(EvaluatedNode {
            tuples: out,
            index_fn: None,
        })
    }
}

/// The index-addressed evaluator ([`evaluate_indexed`]). Each node
/// returns its tuples addressed by position together with the
/// `IndexFn` the reference evaluator claims for it, which is what an
/// enclosing order's strategy routes on.
impl EvalState<'_> {
    fn index_node(
        &mut self,
        node: &Comprehension,
        prefix: &[(String, Value)],
    ) -> Result<(Indexed, Option<IndexFn>), RuntimeError> {
        match node {
            Comprehension::Clause { name, source } => self.index_clause(name, source, prefix),
            Comprehension::Cartesian { children } => self.index_cartesian(children, prefix),
            Comprehension::Zip { children, mode } => self.index_zip(children, *mode, prefix),
            Comprehension::Union { children } => self.index_union(children, prefix),
            Comprehension::Filter { child, predicate } => {
                // A filter keeps the tuples that pass; which ones is
                // known only by testing each, so its output holds them.
                let (inner, _) = self.index_node(child, prefix)?;
                let predicate = CompiledPredicate::new(predicate);
                let mut kept = Vec::new();
                let mut tuple = RuntimeTuple::new();
                for i in 0..inner.len() {
                    tuple.clear();
                    inner.append_at(i, &mut tuple);
                    if predicate.keeps(&tuple, self.scope)? {
                        kept.push(tuple.clone());
                    }
                }
                Ok((Indexed::Tuples(kept), None))
            }
            Comprehension::Order {
                child,
                strategy,
                truncation,
                seed,
            } => {
                let child = shape_input(child, *strategy);
                if has_continuous_axis(child) {
                    let sampled =
                        self.sample_space(child, prefix, *strategy, *truncation, *seed)?;
                    return Ok((Indexed::Tuples(sampled.tuples), sampled.index_fn));
                }
                // A non-`Lex` order over a filter holds the filter's
                // input and the survivors' positions in it, which the
                // strategy ranks by those positions (§5 V5): a barrier
                // sized by the survivors.
                if let (Comprehension::Filter { child, predicate }, false) =
                    (child, *strategy == StrategyName::Lex)
                {
                    let (inner, index_fn) = self.index_node(child, prefix)?;
                    let predicate = CompiledPredicate::new(predicate);
                    let mut survivors = Vec::new();
                    let mut tuple = RuntimeTuple::new();
                    for p in 0..inner.len() {
                        tuple.clear();
                        inner.append_at(p, &mut tuple);
                        if predicate.keeps(&tuple, self.scope)? {
                            survivors.push(p);
                        }
                    }
                    let selection = surviving_selection(
                        *strategy,
                        index_fn.as_ref(),
                        inner.len(),
                        *truncation,
                        *seed,
                        &survivors,
                    )?;
                    return Ok((
                        Indexed::Select {
                            child: Box::new(inner),
                            selection,
                        },
                        None,
                    ));
                }
                let (inner, index_fn) = self.index_node(child, prefix)?;
                let selection = order_selection(
                    *strategy,
                    index_fn.as_ref(),
                    inner.len(),
                    *truncation,
                    *seed,
                )?;
                Ok((
                    Indexed::Select {
                        child: Box::new(inner),
                        selection,
                    },
                    None,
                ))
            }
        }
    }

    fn index_clause(
        &mut self,
        name: &str,
        source: &Source,
        prefix: &[(String, Value)],
    ) -> Result<(Indexed, Option<IndexFn>), RuntimeError> {
        let values = match source {
            // A range's values are its bounds; the source's own
            // evaluation would hold every value.
            Source::IntRange { lo, hi, step } => {
                let step = (*step).max(1);
                let len = if hi <= lo {
                    0
                } else {
                    ((i128::from(*hi) - i128::from(*lo)) as u128).div_ceil(step as u128) as u64
                };
                self.record_yield(source, len as usize);
                ClauseValues::Range { lo: *lo, step, len }
            }
            _ => {
                let evaluated = self.evaluate_source(name, source, prefix)?;
                let index_fn = evaluated.index_fn;
                return Ok((
                    Indexed::Clause {
                        name: name.to_string(),
                        values: ClauseValues::List(evaluated.values),
                    },
                    Some(index_fn),
                ));
            }
        };
        let len = values.len();
        Ok((
            Indexed::Clause {
                name: name.to_string(),
                values,
            },
            Some(IndexFn::Lattice {
                axis_sizes: vec![len],
            }),
        ))
    }

    /// An independent cartesian evaluates each axis once, the
    /// evaluation standing for one per tuple of the axes before it,
    /// and stops at an empty axis as the reference evaluator does. A
    /// cartesian whose sources reference an earlier axis is evaluated
    /// by the reference evaluator and holds its tuples.
    fn index_cartesian(
        &mut self,
        children: &[Comprehension],
        prefix: &[(String, Value)],
    ) -> Result<(Indexed, Option<IndexFn>), RuntimeError> {
        if children.is_empty() || references_an_earlier_axis(children) {
            let node = self.evaluate_cartesian(children, prefix)?;
            return Ok((Indexed::Tuples(node.tuples), node.index_fn));
        }
        let base = self.mult;
        let mut parts = Vec::with_capacity(children.len());
        let mut index_fns = Vec::with_capacity(children.len());
        let mut lens = Vec::with_capacity(children.len());
        let mut len: u64 = 1;
        for child in children {
            let evaluated = self.index_node(child, prefix);
            let (part, index_fn) = match evaluated {
                Ok(done) => done,
                Err(e) => {
                    self.mult = base;
                    return Err(e);
                }
            };
            let part_len = part.len();
            parts.push(part);
            index_fns.push(index_fn);
            lens.push(part_len);
            len = match len.checked_mul(part_len) {
                Some(n) => n,
                None => {
                    self.mult = base;
                    return Err(RuntimeError::UnsupportedShape(format!(
                        "cartesian of {lens:?} tuples exceeds 2^64"
                    )));
                }
            };
            if part_len == 0 {
                break;
            }
            self.mult = self
                .mult
                .saturating_mul(usize::try_from(part_len).unwrap_or(usize::MAX));
        }
        self.mult = base;
        let index_fn = combine_cartesian_index_fn(&index_fns);
        if len == 0 {
            return Ok((Indexed::Tuples(Vec::new()), index_fn));
        }
        Ok((
            Indexed::Product {
                children: parts,
                lens,
                len,
            },
            index_fn,
        ))
    }

    fn index_zip(
        &mut self,
        children: &[Comprehension],
        mode: crate::iteration::comprehension::strategy::ZipMode,
        prefix: &[(String, Value)],
    ) -> Result<(Indexed, Option<IndexFn>), RuntimeError> {
        use crate::iteration::comprehension::strategy::ZipMode;
        if children.is_empty() {
            let node = self.evaluate_zip(children, mode, prefix)?;
            return Ok((Indexed::Tuples(node.tuples), node.index_fn));
        }
        let mut parts = Vec::with_capacity(children.len());
        for child in children {
            parts.push(self.index_node(child, prefix)?.0);
        }
        let lengths: Vec<u64> = parts.iter().map(Indexed::len).collect();
        let len = match mode {
            ZipMode::Strict => {
                let first = lengths[0];
                if lengths.iter().any(|&n| n != first) {
                    return Err(RuntimeError::ZipLengthMismatch { lengths });
                }
                first
            }
            ZipMode::Truncate => lengths.iter().copied().min().unwrap_or(0),
            ZipMode::Cycle => cycle_length(&lengths),
        };
        Ok(match mode {
            ZipMode::Strict | ZipMode::Truncate => (
                Indexed::Lockstep {
                    children: parts,
                    len,
                },
                Some(IndexFn::Lockstep { length: len }),
            ),
            ZipMode::Cycle => (
                Indexed::Cycle {
                    children: parts,
                    len,
                },
                Some(IndexFn::Modular {
                    axis_sizes: lengths,
                }),
            ),
        })
    }

    fn index_union(
        &mut self,
        children: &[Comprehension],
        prefix: &[(String, Value)],
    ) -> Result<(Indexed, Option<IndexFn>), RuntimeError> {
        let mut parts = Vec::with_capacity(children.len());
        let mut segment_sizes = Vec::with_capacity(children.len());
        let mut all_segments_addressable = true;
        for child in children {
            let (part, index_fn) = self.index_node(child, prefix)?;
            all_segments_addressable &= index_fn.is_some();
            segment_sizes.push(part.len());
            parts.push(part);
        }
        let len = segment_sizes
            .iter()
            .try_fold(0u64, |acc, n| acc.checked_add(*n))
            .ok_or_else(|| {
                RuntimeError::UnsupportedShape(format!(
                    "union of {segment_sizes:?} tuples exceeds 2^64"
                ))
            })?;
        let index_fn = all_segments_addressable.then_some(IndexFn::Concatenation { segment_sizes });
        Ok((
            Indexed::Concat {
                children: parts,
                len,
            },
            index_fn,
        ))
    }
}

/// `true` when a child of a cartesian references, in its sources, a
/// name that an earlier child binds: the child's tuples then depend on
/// the tuple before it (spec §3.2).
fn references_an_earlier_axis(children: &[Comprehension]) -> bool {
    let mut bound: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    for child in children {
        if child
            .referenced_source_names()
            .iter()
            .any(|n| bound.contains(n))
        {
            return true;
        }
        collect_clause_names(child, &mut bound);
    }
    false
}

/// Every clause name in `c`, in every branch of a union.
fn collect_clause_names(c: &Comprehension, out: &mut std::collections::BTreeSet<String>) {
    match c {
        Comprehension::Clause { name, .. } => {
            out.insert(name.clone());
        }
        Comprehension::Cartesian { children }
        | Comprehension::Zip { children, .. }
        | Comprehension::Union { children } => {
            for child in children {
                collect_clause_names(child, out);
            }
        }
        Comprehension::Filter { child, .. } | Comprehension::Order { child, .. } => {
            collect_clause_names(child, out);
        }
    }
}

/// The positions `strategy` selects over an input of `cardinality`
/// tuples addressed by `index_fn`, after V4 (comprehension_forms.md
/// §10.7.8). An input the walker could not address (a dependent
/// cartesian, a truncated order) is ordered as a one-axis lattice of
/// its tuples: V4 admits only `Lex` over one, and `Lex` reads nothing
/// but the count. A non-`Lex` order over a filter selects through
/// [`surviving_selection`] instead.
fn order_selection(
    strategy: StrategyName,
    index_fn: Option<&IndexFn>,
    cardinality: u64,
    truncation: Option<u64>,
    seed: Option<u64>,
) -> Result<Selection, RuntimeError> {
    let dispatch = crate::iteration::comprehension::strategies::for_name(strategy);
    if !dispatch.accepts_input(index_fn) {
        return Err(RuntimeError::StrategyRejectsInput {
            strategy,
            index_fn: index_fn.cloned(),
        });
    }
    let fallback;
    let index_fn = match index_fn {
        Some(idx) => idx,
        None => {
            fallback = IndexFn::Lattice {
                axis_sizes: vec![cardinality],
            };
            &fallback
        }
    };
    Ok(dispatch.select(index_fn, cardinality, truncation, seed))
}

/// The positions `strategy` selects among `survivors`, the positions of
/// a filter's input its predicate keeps, ranked by their positions in
/// that input of `cardinality` tuples addressed by `index_fn`, after V4
/// against that input (comprehension_forms.md §5 V5).
fn surviving_selection(
    strategy: StrategyName,
    index_fn: Option<&IndexFn>,
    cardinality: u64,
    truncation: Option<u64>,
    seed: Option<u64>,
    survivors: &[u64],
) -> Result<Selection, RuntimeError> {
    let dispatch = crate::iteration::comprehension::strategies::for_name(strategy);
    let Some(index_fn) = index_fn.filter(|idx| dispatch.accepts_input(Some(idx))) else {
        return Err(RuntimeError::StrategyRejectsInput {
            strategy,
            index_fn: index_fn.cloned(),
        });
    };
    Ok(dispatch.select_surviving(index_fn, cardinality, truncation, seed, survivors))
}

/// How many times a sampled order redraws, doubling the count each
/// time, when its filters leave fewer tuples than asked.
const SAMPLE_ROUNDS: u32 = 6;

/// The scale of a continuous code: the strategies encode a point of
/// `[0, 1)` as a 53-bit fraction.
const UNIT_SCALE: f64 = (1u64 << 53) as f64;

/// One axis of a sampled space (spec §10.2 R2): a discrete child
/// evaluated to its tuples, or a continuous clause's interval and
/// measure.
enum SampleAxis {
    Discrete(Vec<RuntimeTuple>),
    Continuous {
        name: String,
        interval: Interval,
        measure: AxisMeasure,
    },
}

/// The space an order over a continuous axis samples: its axes in
/// clause order, and the predicates of the filters between the
/// order and its clauses.
#[derive(Default)]
struct SampleSpace {
    axes: Vec<SampleAxis>,
    predicates: Vec<String>,
}

impl SampleSpace {
    /// The tuple at a multi-index, its coordinates in clause order.
    /// Under a sampling strategy the multi-index lays the discrete
    /// positions first and the continuous codes after them
    /// (`IndexFn::Hybrid`), a code being a 53-bit fraction of the
    /// unit interval; under `Extrema` it is in clause order over the
    /// lattice, a continuous code being `0` or `1` for an end of the
    /// interval.
    fn realize(&self, mi: &[u64], strategy: StrategyName) -> RuntimeTuple {
        let extrema = matches!(strategy, StrategyName::Extrema);
        let discrete_count = self
            .axes
            .iter()
            .filter(|a| matches!(a, SampleAxis::Discrete(_)))
            .count();
        let (mut d, mut c) = (0, if extrema { 0 } else { discrete_count });
        let mut out = RuntimeTuple::new();
        for axis in &self.axes {
            match axis {
                SampleAxis::Discrete(tuples) => {
                    let pos = mi.get(d).copied().unwrap_or(0) as usize;
                    d += 1;
                    if extrema {
                        c += 1;
                    }
                    if let Some(t) = tuples.get(pos) {
                        out.extend(t.iter().cloned());
                    }
                }
                SampleAxis::Continuous {
                    name,
                    interval,
                    measure,
                } => {
                    let code = mi.get(c).copied().unwrap_or(0);
                    c += 1;
                    if extrema {
                        d += 1;
                    }
                    let x = if extrema {
                        measure.endpoint(interval, code == 1)
                    } else {
                        measure.map_unit(code as f64 / UNIT_SCALE, interval)
                    };
                    out.push((name.clone(), Value::F64(x)));
                }
            }
        }
        out
    }
}

/// The multi-indices a strategy draws over a `Continuous` or
/// `Hybrid` index function: a sampling strategy needs a count, and
/// `Extrema` takes `count` strata (every stratum for `None`). `seed`
/// is the authored seed of a seeded strategy.
fn draw_sample(
    index_fn: &IndexFn,
    strategy: StrategyName,
    count: Option<u64>,
    seed: Option<u64>,
) -> Result<Vec<Vec<u64>>, RuntimeError> {
    use crate::iteration::comprehension::strategies::{
        extrema::extrema_multi_indices, halton::try_halton_multi_indices,
        lhs::try_lhs_multi_indices, shuffle::try_shuffle_multi_indices,
        sobol::try_sobol_multi_indices,
    };
    if matches!(strategy, StrategyName::Extrema) {
        return Ok(extrema_multi_indices(index_fn, count));
    }
    let Some(n) = count else {
        return Err(RuntimeError::OrderEval {
            strategy,
            message: "a continuous source has no finite tuple set; give the order a count, \
                      as in `order halton/16`"
                .into(),
        });
    };
    // The count is the order's own (`order halton/16`), from the spec
    // text, so a count no machine can hold is refused here as the
    // order's error rather than aborting in the allocator.
    let drawn = match strategy {
        StrategyName::Halton => try_halton_multi_indices(index_fn, Some(n)),
        StrategyName::Sobol => try_sobol_multi_indices(index_fn, Some(n)),
        StrategyName::Lhs => try_lhs_multi_indices(index_fn, Some(n), seed),
        StrategyName::Shuffle => try_shuffle_multi_indices(index_fn, Some(n), seed),
        other => {
            return Err(RuntimeError::OrderEval {
                strategy: other,
                message: "a continuous source needs a sampling strategy: halton, sobol, lhs, \
                          shuffle, or extrema"
                    .into(),
            });
        }
    };
    drawn.map_err(|message| RuntimeError::OrderEval { strategy, message })
}

/// `true` when an order over `c` samples: a continuous clause is
/// reachable through cartesians and filters alone. Under a zip,
/// union, or inner order the continuous axis is that node's to
/// discharge.
pub(crate) fn has_continuous_axis(c: &Comprehension) -> bool {
    match c {
        Comprehension::Clause { source, .. } => matches!(
            source,
            Source::ContinuousInterval { .. } | Source::Distribution { .. }
        ),
        Comprehension::Cartesian { children } => children.iter().any(has_continuous_axis),
        Comprehension::Filter { child, .. } => has_continuous_axis(child),
        Comprehension::Zip { .. } | Comprehension::Union { .. } | Comprehension::Order { .. } => {
            false
        }
    }
}

fn combine_cartesian_index_fn(children: &[Option<IndexFn>]) -> Option<IndexFn> {
    let mut axis_sizes = Vec::new();
    for opt in children {
        match opt {
            Some(IndexFn::Lattice { axis_sizes: a }) => axis_sizes.extend(a.iter().copied()),
            Some(IndexFn::Lockstep { length }) => axis_sizes.push(*length),
            // Other shapes don't combine as cartesian axes
            // cleanly — fall back to None.
            _ => return None,
        }
    }
    Some(IndexFn::Lattice { axis_sizes })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::iteration::comprehension::source::LiteralValue;

    fn empty_kernel() -> Arc<PolydatKernel> {
        Arc::new(crate::dsl::compile_polydat_interpreter("\n").unwrap())
    }

    /// Canonical kernel with `extern k: u64` so the runtime
    /// evaluator can install per-clause `k` values via
    /// materialize_subscope — the shape the traversal lowering
    /// produces.
    fn canonical_with_k() -> Arc<PolydatKernel> {
        Arc::new(crate::dsl::compile_polydat_interpreter("extern k: u64\n").unwrap())
    }

    fn clause(name: &str, source: Source) -> Comprehension {
        Comprehension::Clause {
            name: name.into(),
            source,
        }
    }

    fn empty_literal() -> Source {
        Source::Literal { values: Vec::new() }
    }

    /// The count is what the clause produced, per leaf, in tree order.
    #[test]
    fn every_leaf_reports_what_it_yielded() {
        let comp = Comprehension::Cartesian {
            children: vec![
                clause(
                    "a",
                    Source::IntRange {
                        lo: 0,
                        hi: 3,
                        step: 1,
                    },
                ),
                clause(
                    "b",
                    Source::Literal {
                        values: vec![LiteralValue::Int(7), LiteralValue::Int(8)],
                    },
                ),
            ],
        };
        let scope = empty_kernel();

        let out = evaluate_for_iteration_reported(&comp, &*scope).unwrap();
        assert_eq!(out.tuples.len(), 6, "3 x 2");
        assert_eq!(out.clauses.len(), 2, "one entry per leaf, in tree order");
        assert_eq!(out.clauses[0].var, "a");
        assert_eq!(out.clauses[0].values, 3);
        assert_eq!(out.clauses[1].var, "b");
        // `b` is evaluated once per tuple of `a`, so its values sum over
        // those evaluations: emptiness is a property of the evaluations.
        assert_eq!(out.clauses[1].evaluations, 3);
        assert_eq!(out.clauses[1].values, 6);
    }

    /// The clause a diagnostic should name is the one that was reached
    /// and still produced nothing.
    #[test]
    fn an_empty_clause_is_reached_and_yields_nothing() {
        let comp = Comprehension::Cartesian {
            children: vec![
                clause(
                    "a",
                    Source::IntRange {
                        lo: 0,
                        hi: 2,
                        step: 1,
                    },
                ),
                clause("b", empty_literal()),
            ],
        };
        let scope = empty_kernel();

        let out = evaluate_for_iteration_reported(&comp, &*scope).unwrap();
        assert!(out.tuples.is_empty(), "an empty clause empties the product");
        let culprits: Vec<&str> = out
            .clauses
            .iter()
            .filter(|c| c.evaluations > 0 && c.values == 0)
            .map(|c| c.var.as_str())
            .collect();
        assert_eq!(culprits, ["b"], "only the empty clause is named");
    }

    /// A clause behind an empty one is never reached, so it is not the
    /// cause and must not read as one.
    #[test]
    fn a_clause_behind_an_empty_one_is_never_reached() {
        let comp = Comprehension::Cartesian {
            children: vec![
                clause("outer", empty_literal()),
                clause(
                    "inner",
                    Source::IntRange {
                        lo: 0,
                        hi: 9,
                        step: 1,
                    },
                ),
            ],
        };
        let scope = empty_kernel();

        let out = evaluate_for_iteration_reported(&comp, &*scope).unwrap();
        assert!(out.tuples.is_empty());
        let by = |v: &str| {
            out.clauses
                .iter()
                .find(|c| c.var == v)
                .expect("every leaf is present whether reached or not")
        };
        assert_eq!(by("outer").evaluations, 1);
        assert_eq!(by("outer").values, 0);
        assert_eq!(
            by("inner").evaluations,
            0,
            "never reached: the cause is `outer`, not this"
        );
        assert_eq!(by("inner").values, 0);
    }

    /// Two clauses can share a name across the branches of a union;
    /// they are still two leaves and are counted apart.
    #[test]
    fn clauses_sharing_a_name_across_a_union_are_counted_apart() {
        let comp = Comprehension::Union {
            children: vec![
                clause(
                    "k",
                    Source::Literal {
                        values: vec![LiteralValue::Int(1)],
                    },
                ),
                clause("k", empty_literal()),
            ],
        };
        let scope = empty_kernel();

        let out = evaluate_for_iteration_reported(&comp, &*scope).unwrap();
        assert_eq!(out.clauses.len(), 2, "two leaves, one name");
        assert_eq!(out.clauses[0].values, 1);
        assert_eq!(out.clauses[1].values, 0);
        assert_eq!(out.clauses[1].evaluations, 1, "reached, and empty");
    }

    /// The counts ride the evaluation that already happened, so the
    /// plain entry point is the reported one without its record.
    #[test]
    fn the_plain_entry_point_agrees_with_the_reported_one() {
        let comp = Comprehension::Cartesian {
            children: vec![
                clause(
                    "a",
                    Source::IntRange {
                        lo: 1,
                        hi: 4,
                        step: 1,
                    },
                ),
                clause(
                    "b",
                    Source::Literal {
                        values: vec![LiteralValue::Int(5)],
                    },
                ),
            ],
        };
        let scope = empty_kernel();

        let plain = evaluate_for_iteration(&comp, &*scope).unwrap();
        let reported = evaluate_for_iteration_reported(&comp, &*scope).unwrap();
        assert_eq!(plain, reported.tuples);
    }

    #[test]
    fn int_range_yields_values() {
        let comp = Comprehension::Clause {
            name: "k".into(),
            source: Source::IntRange {
                lo: 1,
                hi: 5,
                step: 1,
            },
        };
        let canonical = empty_kernel();

        let tuples = evaluate_for_iteration(&comp, &*canonical).unwrap();
        assert_eq!(tuples.len(), 4);
        assert_eq!(tuples[0][0].1, Value::U64(1));
        assert_eq!(tuples[3][0].1, Value::U64(4));
    }

    #[test]
    fn literal_list_yields_values() {
        let comp = Comprehension::Clause {
            name: "x".into(),
            source: Source::Literal {
                values: vec![LiteralValue::Int(10), LiteralValue::Int(20)],
            },
        };
        let canonical = empty_kernel();

        let tuples = evaluate_for_iteration(&comp, &*canonical).unwrap();
        assert_eq!(tuples.len(), 2);
    }

    #[test]
    fn cartesian_produces_product() {
        let comp = Comprehension::cartesian(vec![
            Comprehension::Clause {
                name: "x".into(),
                source: Source::IntRange {
                    lo: 1,
                    hi: 3,
                    step: 1,
                },
            },
            Comprehension::Clause {
                name: "y".into(),
                source: Source::IntRange {
                    lo: 10,
                    hi: 30,
                    step: 10,
                },
            },
        ]);
        let canonical = empty_kernel();

        let tuples = evaluate_for_iteration(&comp, &*canonical).unwrap();
        // 2 × 2 = 4
        assert_eq!(tuples.len(), 4);
    }

    #[test]
    fn union_produces_concatenation() {
        let comp = Comprehension::union(vec![
            Comprehension::Clause {
                name: "k".into(),
                source: Source::Literal {
                    values: vec![LiteralValue::Int(1)],
                },
            },
            Comprehension::Clause {
                name: "k".into(),
                source: Source::Literal {
                    values: vec![LiteralValue::Int(10), LiteralValue::Int(20)],
                },
            },
        ]);
        let canonical = empty_kernel();

        let tuples = evaluate_for_iteration(&comp, &*canonical).unwrap();
        assert_eq!(tuples.len(), 3);
    }

    #[test]
    fn filter_drops_non_matching() {
        let comp = Comprehension::filter(
            Comprehension::Clause {
                name: "k".into(),
                source: Source::IntRange {
                    lo: 1,
                    hi: 6,
                    step: 1,
                },
            },
            "{k} > 3",
        );
        let canonical = canonical_with_k();

        let tuples = evaluate_for_iteration(&comp, &*canonical).unwrap();
        // 1..6 = [1,2,3,4,5]; filter > 3 keeps [4, 5]
        assert_eq!(tuples.len(), 2);
    }

    #[test]
    fn order_lex_truncate() {
        let comp = Comprehension::order(
            Comprehension::Clause {
                name: "k".into(),
                source: Source::IntRange {
                    lo: 1,
                    hi: 100,
                    step: 1,
                },
            },
            StrategyName::Lex,
            Some(5),
        );
        let canonical = empty_kernel();

        let tuples = evaluate_for_iteration(&comp, &*canonical).unwrap();
        assert_eq!(tuples.len(), 5);
    }

    /// A cartesian's lattice has one axis per child, however many
    /// names a child binds: a two-name zip at the head is one axis, so
    /// `halton/6` over the 2 x 3 product draws all six tuples.
    #[test]
    fn a_multi_name_head_is_one_lattice_axis() {
        use crate::iteration::comprehension::strategy::ZipMode;
        let lit = |name: &str, vs: &[i64]| Comprehension::Clause {
            name: name.into(),
            source: Source::Literal {
                values: vs.iter().map(|v| LiteralValue::Int(*v)).collect(),
            },
        };
        let comp = Comprehension::order(
            Comprehension::cartesian(vec![
                Comprehension::zip(vec![lit("a", &[1, 2]), lit("b", &[3, 4])], ZipMode::Strict),
                lit("c", &[5, 6, 7]),
            ]),
            StrategyName::Halton,
            Some(6),
        );
        let tuples = evaluate_for_iteration(&comp, &*empty_kernel()).unwrap();
        assert_eq!(
            tuples.len(),
            6,
            "every tuple of the 2 x 3 product: {tuples:?}"
        );
    }

    /// PR α bug regression: a Generator-evaluated source can
    /// now claim an IndexFn::Lattice via SourceEval, so
    /// Extrema's indexed path fires and the 2-D Lattice case
    /// (cartesian of two clauses) gives the 2x2 corners, not
    /// just first/last of the cartesian product.
    #[test]
    fn extrema_over_cartesian_uses_indexed_form() {
        let comp = Comprehension::order(
            Comprehension::cartesian(vec![
                Comprehension::Clause {
                    name: "k".into(),
                    source: Source::Literal {
                        values: vec![
                            LiteralValue::Int(1),
                            LiteralValue::Int(2),
                            LiteralValue::Int(3),
                        ],
                    },
                },
                Comprehension::Clause {
                    name: "limit".into(),
                    source: Source::Literal {
                        values: vec![
                            LiteralValue::Int(10),
                            LiteralValue::Int(20),
                            LiteralValue::Int(30),
                        ],
                    },
                },
            ]),
            StrategyName::Extrema,
            // SRD-18d §214: `extrema/1` = the corner stratum. (Bare
            // `extrema`/`None` is now the full 9-tuple space reordered
            // corners-first; `/1` selects just the corners.)
            Some(1),
        );
        let canonical = empty_kernel();

        let tuples = evaluate_for_iteration(&comp, &*canonical).unwrap();
        // 3x3 lattice → 4 corners (interior count 0) via the indexed form.
        assert_eq!(tuples.len(), 4);
        // Each corner pairs an extreme k with an extreme limit.
        for t in &tuples {
            assert_eq!(t.len(), 2);
            let k = match &t[0].1 {
                Value::U64(n) => *n,
                other => panic!("expected u64 k, got {other:?}"),
            };
            let lim = match &t[1].1 {
                Value::U64(n) => *n,
                other => panic!("expected u64 limit, got {other:?}"),
            };
            assert!(k == 1 || k == 3, "expected extreme k, got {k}");
            assert!(lim == 10 || lim == 30, "expected extreme limit, got {lim}");
        }
    }
}
