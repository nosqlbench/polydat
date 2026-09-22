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
//!   [`interpolate_via_kernel`], `kernel.lookup(name)` for
//!   workload params.
//! - **Cartesian is dependent-tuple, not independent.** Clause
//!   N's spec text may reference iter-vars from clauses
//!   1..N-1. Prior-axis values are layered in front of the
//!   scope (`Layered`) so each clause evaluates against the
//!   correct context. This is SRD-18b §"Dependent Tuple
//!   Iteration".
//! - **Filter predicates evaluate against per-tuple scopes.**
//!   Predicates in the comprehension grammar are evaluated
//!   directly against the tuple with no kernel and no compile;
//!   anything richer is interpolated against a `Layered` view
//!   of the scope and evaluated with `eval_const_expr_for`,
//!   charged to the scope's ledger.
//!
//! All three depend on polydat-side primitives that exist
//! today; this evaluator is the algebra-typed entry point for
//! them.
//!
//! ## What this owns
//!
//! [`evaluate_for_iteration`] is the public surface:
//! `(algebra AST + scope) → Vec<RuntimeTuple>`. The returned
//! tuples carry polydat [`Value`]s, which the caller binds into a
//! kernel over the body's program, one per tuple.
//!
//! Order modifiers route through the unified
//! `Strategy::apply` (spec §10.7.8): each node returns its
//! tuples paired with the [`IndexFn`] the materialized stream
//! satisfies; the Order node assembles an [`EvaluatedInput`]
//! and invokes the strategy. V4 fires at this site,
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
use crate::dsl::compile::eval_const_expr_for;
use crate::iteration::comprehension::ast::Comprehension;
use crate::iteration::comprehension::cardinality::{Interval, ProductMeasure};
use crate::iteration::comprehension::eval_source::{EvalContext, SourceEval};
use crate::iteration::comprehension::measure::AxisMeasure;
use crate::iteration::comprehension::metadata::IndexFn;
use crate::iteration::comprehension::source::Source;
use crate::iteration::comprehension::strategies::{EvaluatedInput, Tuple, TupleValue};
use crate::iteration::comprehension::strategy::StrategyName;
#[cfg(test)]
use crate::kernel::PolydatKernel;
use crate::kernel::interp::{Layered, Lookup, interpolate_via_kernel};

/// Runtime tuple type — polydat-Value-based to preserve Ext
/// typing (Partition / Json / etc.) through the iteration
/// pipeline. The algebra layer's [`Tuple`] uses
/// [`TupleValue`] which is scalar-only; this `RuntimeTuple`
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
        }
    }
}

impl std::error::Error for RuntimeError {}

/// Evaluate a comprehension against a scope and produce the
/// typed coordinate-tuple list.
///
/// `scope` is where names resolve: the body's kernel with the
/// parent's cascaded wires, or any other [`Lookup`].
///
/// It used to take two more: a workload-parameter map, which the
/// evaluator stored and never read, and an empty-clause callback,
/// which every caller passed as "do nothing". Both date from a host
/// that decided an empty-clause policy of its own; a clause with no
/// values yields no tuples, which is what an empty clause means.
pub fn evaluate_for_iteration(
    comp: &Comprehension,
    scope: &dyn Lookup,
) -> Result<Vec<RuntimeTuple>, RuntimeError> {
    evaluate_for_iteration_reported(comp, scope).map(|e| e.tuples)
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
    let tuples = state.evaluate_node(comp, &[])?.tuples;
    Ok(EvaluatedIteration {
        tuples,
        clauses: state.yields,
    })
}

/// Evaluate a predicate in the comprehension grammar against a tuple
/// without a kernel. Returns `None` when the predicate uses anything
/// outside that grammar, or references a name the tuple does not bind,
/// so the caller can fall back to kernel interpolation.
fn fast_predicate(predicate: &str, tuple: &RuntimeTuple) -> Option<bool> {
    let p = predicate.trim();
    if p.eq_ignore_ascii_case("true") {
        return Some(true);
    }
    if p.eq_ignore_ascii_case("false") {
        return Some(false);
    }
    if let Some(inner) = p.strip_prefix('!') {
        return fast_predicate(inner, tuple).map(|b| !b);
    }
    if let Some(parts) = split_top(p, "||") {
        let mut any = false;
        for part in parts {
            any |= fast_predicate(&part, tuple)?;
        }
        return Some(any);
    }
    if let Some(parts) = split_top(p, "&&") {
        let mut all = true;
        for part in parts {
            all &= fast_predicate(&part, tuple)?;
        }
        return Some(all);
    }
    if let Some(pos) = p.find(" in ") {
        let name = curly(p[..pos].trim())?;
        let list = p[pos + 4..].trim().strip_prefix('[')?.strip_suffix(']')?;
        let needle = tuple_scalar(tuple, &name)?;
        let mut hit = false;
        for item in list.split(',') {
            let lit = literal(item.trim())?;
            hit |= scalar_eq(&needle, &lit);
        }
        return Some(hit);
    }
    for op in ["==", "!=", "<=", ">=", "<", ">"] {
        if let Some((lhs, rhs)) = split_op(p, op) {
            let lhs = lhs.trim();
            let rhs = rhs.trim();
            let a = operand(tuple, lhs)?;
            let b = operand(tuple, rhs)?;
            return Some(match op {
                "==" => scalar_eq(&a, &b),
                "!=" => !scalar_eq(&a, &b),
                "<" => scalar_cmp(&a, &b)? == std::cmp::Ordering::Less,
                ">" => scalar_cmp(&a, &b)? == std::cmp::Ordering::Greater,
                "<=" => scalar_cmp(&a, &b)? != std::cmp::Ordering::Greater,
                _ => scalar_cmp(&a, &b)? != std::cmp::Ordering::Less,
            });
        }
    }
    None
}

#[derive(Debug, Clone, PartialEq)]
enum Scalar {
    Int(i128),
    Float(f64),
    Str(String),
    Bool(bool),
}

fn operand(tuple: &RuntimeTuple, text: &str) -> Option<Scalar> {
    match curly(text) {
        Some(name) => tuple_scalar(tuple, &name),
        None => literal(text),
    }
}

fn tuple_scalar(tuple: &RuntimeTuple, name: &str) -> Option<Scalar> {
    let (_, v) = tuple.iter().find(|(n, _)| n == name)?;
    match v {
        Value::U64(n) => Some(Scalar::Int(*n as i128)),
        Value::F64(f) => Some(Scalar::Float(*f)),
        Value::Str(s) => Some(Scalar::Str(s.to_string())),
        Value::Bool(b) => Some(Scalar::Bool(*b)),
        // A JSON list's item compares as the scalar it carries.
        Value::Json(j) => match j.as_ref() {
            serde_json::Value::Number(n) if n.is_i64() => Some(Scalar::Int(n.as_i64()? as i128)),
            serde_json::Value::Number(n) if n.is_u64() => Some(Scalar::Int(n.as_u64()? as i128)),
            serde_json::Value::Number(n) => Some(Scalar::Float(n.as_f64()?)),
            serde_json::Value::String(s) => Some(Scalar::Str(s.clone())),
            serde_json::Value::Bool(b) => Some(Scalar::Bool(*b)),
            _ => None,
        },
        _ => None,
    }
}

fn literal(text: &str) -> Option<Scalar> {
    if let Some(s) = text.strip_prefix('"').and_then(|s| s.strip_suffix('"')) {
        return Some(Scalar::Str(s.to_string()));
    }
    if let Some(s) = text.strip_prefix('\'').and_then(|s| s.strip_suffix('\'')) {
        return Some(Scalar::Str(s.to_string()));
    }
    match text {
        "true" => return Some(Scalar::Bool(true)),
        "false" => return Some(Scalar::Bool(false)),
        _ => {}
    }
    if let Ok(i) = text.parse::<i128>() {
        return Some(Scalar::Int(i));
    }
    if let Ok(f) = text.parse::<f64>() {
        return Some(Scalar::Float(f));
    }
    // A bare word compares as text, matching the interpolated form
    // `load == load` a kernel evaluation would see for string elements.
    if !text.is_empty()
        && text
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
    {
        return Some(Scalar::Str(text.to_string()));
    }
    None
}

fn scalar_eq(a: &Scalar, b: &Scalar) -> bool {
    match (a, b) {
        (Scalar::Int(x), Scalar::Float(y)) | (Scalar::Float(y), Scalar::Int(x)) => {
            (*x as f64) == *y
        }
        _ => a == b,
    }
}

fn scalar_cmp(a: &Scalar, b: &Scalar) -> Option<std::cmp::Ordering> {
    match (a, b) {
        (Scalar::Int(x), Scalar::Int(y)) => Some(x.cmp(y)),
        (Scalar::Float(x), Scalar::Float(y)) => x.partial_cmp(y),
        (Scalar::Int(x), Scalar::Float(y)) => (*x as f64).partial_cmp(y),
        (Scalar::Float(x), Scalar::Int(y)) => x.partial_cmp(&(*y as f64)),
        (Scalar::Str(x), Scalar::Str(y)) => Some(x.cmp(y)),
        _ => None,
    }
}

fn curly(text: &str) -> Option<String> {
    let inner = text.strip_prefix('{')?.strip_suffix('}')?;
    (!inner.is_empty() && inner.chars().all(|c| c.is_ascii_alphanumeric() || c == '_'))
        .then(|| inner.to_string())
}

/// Split at a top-level binary token, respecting brackets and quotes.
fn split_top(s: &str, sep: &str) -> Option<Vec<String>> {
    let mut parts = Vec::new();
    let mut depth = 0i32;
    let mut quote: Option<char> = None;
    let mut start = 0;
    let bytes: Vec<char> = s.chars().collect();
    let sepc: Vec<char> = sep.chars().collect();
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i];
        if let Some(q) = quote {
            if c == q {
                quote = None;
            }
        } else {
            match c {
                '"' | '\'' => quote = Some(c),
                '(' | '[' | '{' => depth += 1,
                ')' | ']' | '}' => depth -= 1,
                _ => {}
            }
            if depth == 0 && bytes[i..].starts_with(&sepc) {
                parts.push(bytes[start..i].iter().collect::<String>());
                i += sepc.len();
                start = i;
                continue;
            }
        }
        i += 1;
    }
    if parts.is_empty() {
        return None;
    }
    parts.push(bytes[start..].iter().collect::<String>());
    Some(parts)
}

fn split_op<'a>(s: &'a str, op: &str) -> Option<(&'a str, &'a str)> {
    let mut depth = 0i32;
    let mut quote: Option<char> = None;
    let chars: Vec<(usize, char)> = s.char_indices().collect();
    for (k, &(idx, c)) in chars.iter().enumerate() {
        if let Some(q) = quote {
            if c == q {
                quote = None;
            }
            continue;
        }
        match c {
            '"' | '\'' => {
                quote = Some(c);
                continue;
            }
            '(' | '[' | '{' => depth += 1,
            ')' | ']' | '}' => depth -= 1,
            _ => {}
        }
        if depth == 0 && s[idx..].starts_with(op) {
            // Longest-match: do not split `<=` at `<`, or `!=`/`==` at `=`.
            let next = chars.get(k + op.len()).map(|(_, c)| *c);
            if (op == "<" || op == ">") && next == Some('=') {
                continue;
            }
            let prev = if k > 0 { Some(chars[k - 1].1) } else { None };
            if (op == "<" || op == ">") && matches!(prev, Some('<') | Some('>')) {
                continue;
            }
            return Some((&s[..idx], &s[idx + op.len()..]));
        }
    }
    None
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
}

impl<'a> EvalState<'a> {
    /// Seed one entry per leaf clause, in tree order.
    fn new(comp: &Comprehension, scope: &'a dyn Lookup) -> Self {
        let mut state = EvalState {
            scope,
            yields: Vec::new(),
            by_leaf: std::collections::HashMap::new(),
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

    /// Record one evaluation of a leaf and the values it produced.
    fn record_yield(&mut self, source: &Source, values: usize) {
        if let Some(&i) = self.by_leaf.get(&(std::ptr::from_ref(source) as usize)) {
            self.yields[i].evaluations += 1;
            self.yields[i].values += values;
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
                // A continuous axis has no tuples of its own: an order
                // over one samples the child's space, discrete axes by
                // position and continuous axes through their measures
                // (spec §10.2 R2).
                if has_continuous_axis(child) {
                    return self.sample_space(child, prefix, *strategy, *truncation, *seed);
                }
                let inner = self.evaluate_node(child, prefix)?;
                self.apply_order(inner, *strategy, *truncation, *seed)
            }
        }
    }

    fn evaluate_clause(
        &mut self,
        name: &str,
        source: &Source,
        prefix: &[(String, Value)],
    ) -> Result<EvaluatedNode, RuntimeError> {
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
        let mut child_index_fns: Vec<Option<IndexFn>> = Vec::with_capacity(children.len());
        let mut dependent_observed = false;
        let result_tuples = self.evaluate_cartesian_rec(
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
            combine_cartesian_index_fn(&child_index_fns)
        };
        Ok(EvaluatedNode {
            tuples: result_tuples,
            index_fn: combined,
        })
    }

    fn evaluate_cartesian_rec(
        &mut self,
        children: &[Comprehension],
        prefix: &[(String, Value)],
        child_index_fns: &mut Vec<Option<IndexFn>>,
        dependent_observed: &mut bool,
    ) -> Result<Vec<RuntimeTuple>, RuntimeError> {
        if children.is_empty() {
            return Ok(vec![Vec::new()]);
        }
        let (head, tail) = children.split_first().unwrap();
        let head_eval = self.evaluate_node(head, prefix)?;
        let head_axis_len = head_eval.tuples.len() as u64;
        // First time through, record the head's index_fn.
        if child_index_fns.len() <= prefix_depth(prefix, child_index_fns) {
            child_index_fns.push(head_eval.index_fn.clone());
        } else if let Some(prev) = child_index_fns
            .get(prefix_depth(prefix, child_index_fns))
            .cloned()
            .flatten()
        {
            // Subsequent prefix iterations of a dependent
            // cartesian: if the per-prefix child cardinality
            // differs from the first prefix's, mark dependent.
            if axis_size_of(&prev) != Some(head_axis_len) {
                *dependent_observed = true;
            }
        }

        if tail.is_empty() {
            return Ok(head_eval.tuples);
        }
        let mut out = Vec::new();
        for head_tuple in head_eval.tuples {
            let mut extended_prefix: Vec<(String, Value)> = prefix.to_vec();
            extended_prefix.extend(head_tuple.iter().cloned());
            let tail_tuples = self.evaluate_cartesian_rec(
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
                    return Err(RuntimeError::UnsupportedShape(format!(
                        "zip strict: child lengths differ ({lengths:?})"
                    )));
                }
                first
            }
            ZipMode::Truncate => lengths.iter().copied().min().unwrap_or(0),
            ZipMode::Cycle => lengths.iter().copied().max().unwrap_or(0),
        };
        let mut tuples = Vec::with_capacity(iter_count);
        for i in 0..iter_count {
            let mut bindings: RuntimeTuple = Vec::new();
            for (child, &len) in per_child.iter().zip(lengths.iter()) {
                if len == 0 {
                    continue;
                }
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
        let mut out = Vec::with_capacity(input.tuples.len());
        for tuple in input.tuples {
            // Fast path: the comprehension predicate grammar (`{name}`
            // compared to a literal or another `{name}`, joined by `&&`,
            // `||`, `!`, or `in [...]`) evaluates directly against the
            // tuple, without a kernel and without compiling (SRD 113
            // §5.2). Anything richer takes the kernel path below.
            if let Some(keep) = fast_predicate(predicate, &tuple) {
                if keep {
                    out.push(tuple);
                }
                continue;
            }
            let scope = Layered {
                prefix: &tuple,
                inner: self.scope,
            };
            let interpolated = interpolate_via_kernel(predicate, &scope).map_err(|e| {
                RuntimeError::FilterEval {
                    predicate: predicate.to_string(),
                    message: e.to_string(),
                }
            })?;
            let result = eval_const_expr_for(&interpolated, self.scope.ledger()).map_err(|e| {
                RuntimeError::FilterEval {
                    predicate: predicate.to_string(),
                    message: e.to_string(),
                }
            })?;
            let keep = match result {
                Value::Bool(b) => b,
                Value::U64(n) => n != 0,
                Value::F64(n) => n != 0.0,
                other => {
                    return Err(RuntimeError::FilterEval {
                        predicate: predicate.to_string(),
                        message: format!("expected bool/u64/f64, got {other:?}"),
                    });
                }
            };
            if keep {
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
        use crate::iteration::comprehension::strategies::{
            Strategy, antidiagonal::Antidiagonal, diagonal::Diagonal, extrema::Extrema,
            halton::Halton, lex::Lex, lhs::Lhs, reverse_lex::ReverseLex, shells::Shells,
            shuffle::Shuffle, sobol::Sobol,
        };
        use crate::iteration::comprehension::surfaces::polydat_value_to_tuple_value;

        let dispatch: Box<dyn Strategy> = match strategy {
            StrategyName::Lex => Box::new(Lex),
            StrategyName::ReverseLex => Box::new(ReverseLex),
            StrategyName::Diagonal => Box::new(Diagonal),
            StrategyName::Antidiagonal => Box::new(Antidiagonal),
            StrategyName::Extrema => Box::new(Extrema),
            StrategyName::Shells => Box::new(Shells),
            StrategyName::Halton => Box::new(Halton),
            StrategyName::Sobol => Box::new(Sobol),
            StrategyName::Lhs => Box::new(Lhs),
            StrategyName::Shuffle => Box::new(Shuffle),
        };

        // V4 fire at strategy-invocation time (spec §10.7.8).
        if !dispatch.accepts_input(input.index_fn.as_ref()) {
            return Err(RuntimeError::StrategyRejectsInput {
                strategy,
                index_fn: input.index_fn.clone(),
            });
        }

        // Build algebra tuples for the strategy in parallel
        // with the runtime tuples. Conversion preserves the
        // input's index order: post-apply we recover the
        // chosen runtime tuples via algebra-Tuple PartialEq
        // with a consumed-index bitmap so duplicate-valued
        // tuples preserve original ordering.
        let algebra_tuples: Vec<Tuple> = input
            .tuples
            .iter()
            .map(|rt| Tuple {
                bindings: rt
                    .iter()
                    .map(|(n, v)| {
                        let tv = polydat_value_to_tuple_value(v)
                            .unwrap_or(TupleValue::Str(v.to_display_string()));
                        (n.clone(), tv)
                    })
                    .collect(),
            })
            .collect();

        // Strategy needs SOME IndexFn to operate; if the
        // upstream walker couldn't claim one (filter / dependent
        // cartesian without combine), fall back to a 1-D
        // Lattice of the observed length. The strategy's
        // accepts_input still gated this via V4 above; Lex
        // accepts None and reaches here; every other strategy
        // requires Some(_) and reached here only because the
        // walker provided one.
        let index_fn = input.index_fn.clone().unwrap_or(IndexFn::Lattice {
            axis_sizes: vec![algebra_tuples.len() as u64],
        });
        let cardinality = algebra_tuples.len() as u64;
        let evaluated_input = EvaluatedInput {
            tuples: algebra_tuples.clone(),
            cardinality,
            index_fn,
        };

        let ordered = dispatch.apply_seeded(&evaluated_input, truncation, seed);

        // Map ordered algebra tuples back to runtime tuples via
        // PartialEq + consumed-index bitmap.
        let mut consumed = vec![false; algebra_tuples.len()];
        let mut out = Vec::with_capacity(ordered.len());
        for ordered_tuple in &ordered {
            let idx = algebra_tuples
                .iter()
                .enumerate()
                .find(|(i, at)| !consumed[*i] && *at == ordered_tuple)
                .map(|(i, _)| i)
                .ok_or_else(|| RuntimeError::OrderEval {
                    strategy,
                    message: "ordered tuple lost reference to runtime source — \
                              Strategy::apply must return tuples drawn from \
                              EvaluatedInput.tuples (per spec §10.7.8)"
                        .into(),
                })?;
            consumed[idx] = true;
            out.push(input.tuples[idx].clone());
        }
        // Order may produce a different index_fn (e.g., Lex
        // preserves; non-Lex destroys), but downstream
        // consumers of evaluate_for_iteration only read tuples.
        Ok(EvaluatedNode {
            tuples: out,
            index_fn: None,
        })
    }
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
        extrema::extrema_multi_indices, halton::halton_multi_indices, lhs::lhs_multi_indices,
        shuffle::shuffle_multi_indices, sobol::sobol_multi_indices,
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
    Ok(match strategy {
        StrategyName::Halton => halton_multi_indices(index_fn, Some(n)),
        StrategyName::Sobol => sobol_multi_indices(index_fn, Some(n)),
        StrategyName::Lhs => lhs_multi_indices(index_fn, Some(n), seed),
        StrategyName::Shuffle => shuffle_multi_indices(index_fn, Some(n), seed),
        other => {
            return Err(RuntimeError::OrderEval {
                strategy: other,
                message: "a continuous source needs a sampling strategy: halton, sobol, lhs, \
                          shuffle, or extrema"
                    .into(),
            });
        }
    })
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

/// Helper: prefix depth into the child_index_fns recording.
/// At runtime, each clause is evaluated against a prefix; the
/// first prefix slot per clause records its index_fn. This
/// function returns the prefix depth count = number of named
/// bindings in `prefix` that originate from the current
/// cartesian sequence — which for the simple recursive walker
/// equals the prefix length minus any names we've already
/// recorded. Conservatively returns prefix.len().
fn prefix_depth(prefix: &[(String, Value)], _recorded: &[Option<IndexFn>]) -> usize {
    prefix.len()
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

fn axis_size_of(idx: &IndexFn) -> Option<u64> {
    match idx {
        IndexFn::Lattice { axis_sizes } if axis_sizes.len() == 1 => Some(axis_sizes[0]),
        IndexFn::Lockstep { length } => Some(*length),
        _ => None,
    }
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
