// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Runtime evaluator — walks an algebra [`Comprehension`] AST
//! against a live parent kernel context to produce typed
//! coordinate tuples.
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
//!   1..N-1. Per-branch kernels (via
//!   `PolydatKernel::materialize_subscope`) carry the prior
//!   values forward so each clause evaluates against the
//!   correct context. This is SRD-18b §"Dependent Tuple
//!   Iteration".
//! - **Filter predicates evaluate against per-tuple kernels.**
//!   The predicate text is interpolated against a kernel with
//!   every tuple value installed, then run through
//!   `eval_const_expr`.
//!
//! All three depend on polydat-side primitives that exist
//! today; this evaluator is the algebra-typed entry point for
//! them.
//!
//! ## What this owns
//!
//! [`evaluate_for_iteration`] is the public surface:
//! `(algebra AST + parent kernel + canonical kernel +
//! workload params) → Vec<RuntimeTuple>`. The returned tuples
//! carry polydat [`Value`]s ready for per-iteration kernel
//! construction via [`PolydatKernel::for_iteration`].
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
//!   tuples; the caller (executor or stream surface) builds
//!   the per-iter kernel via `PolydatKernel::for_iteration`.
//! - Empty-clause policy (strict / warn). The caller passes
//!   an `on_empty` callback the same way `enumerate_tuples`
//!   does today.

use std::collections::HashMap;
use std::sync::Arc;

use crate::ast::Value;
use crate::dsl::compile::eval_const_expr;
use crate::iteration::comprehension::ast::Comprehension;
use crate::iteration::comprehension::eval_source::{EvalContext, SourceEval};
use crate::iteration::comprehension::metadata::IndexFn;
use crate::iteration::comprehension::source::Source;
use crate::iteration::comprehension::strategies::{EvaluatedInput, Tuple, TupleValue};
use crate::iteration::comprehension::strategy::StrategyName;
use crate::kernel::PolydatKernel;
use crate::kernel::interp::interpolate_via_kernel;

/// Runtime tuple type — polydat-Value-based to preserve Ext
/// typing (Partition / Json / etc.) through the iteration
/// pipeline. The algebra layer's [`Tuple`] uses
/// [`TupleValue`] which is scalar-only; this `RuntimeTuple`
/// is what the executor actually wants for per-iteration
/// kernel binding via [`PolydatKernel::for_iteration`].
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

/// Reason a clause produced no values, for the caller's
/// empty-clause policy callback.
#[derive(Debug)]
pub struct EmptyClause<'a> {
    pub var: &'a str,
    pub spec_expr: Option<&'a str>,
}

/// Errors the runtime evaluator surfaces.
#[derive(Debug, Clone)]
pub enum RuntimeError {
    /// Source evaluation failed (interpolation error,
    /// eval_const_expr error, unsupported source shape, etc.).
    SourceEval {
        var: String,
        source: String,
        message: String,
    },
    /// Filter predicate evaluation failed.
    FilterEval { predicate: String, message: String },
    /// Strategy application failed.
    OrderEval {
        strategy: StrategyName,
        message: String,
    },
    /// V4 (spec §5) violation — strategy rejects the input's
    /// addressing shape at invocation time (spec §10.7.8).
    StrategyRejectsInput {
        strategy: StrategyName,
        index_fn: Option<IndexFn>,
    },
    /// The runtime evaluator encountered an algebra-AST shape
    /// it doesn't support (e.g., nested Filter under Order).
    UnsupportedShape(String),
    /// Caller's `on_empty` callback returned an error.
    EmptyPolicy(String),
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
            RuntimeError::EmptyPolicy(msg) => write!(f, "{msg}"),
        }
    }
}

impl std::error::Error for RuntimeError {}

/// Evaluate a comprehension against the parent kernel and
/// produce the typed coordinate-tuple list.
///
/// `canonical` is the comprehension scope's kernel program
/// (used for per-branch `materialize_subscope` calls).
/// `parent` is the runtime parent — the kernel chain root for
/// all source evaluation and shadow resolution.
/// `workload_params` provides the fallback for
/// `Source::WorkloadParamList` names not yet promoted into
/// the kernel chain.
///
/// `on_empty` is called with each empty clause (zero values
/// after evaluation). Caller decides whether to abort (strict)
/// or warn-and-skip (relaxed) — same shape as
/// `enumerate_tuples`'s callback.
pub fn evaluate_for_iteration<F>(
    comp: &Comprehension,
    parent: &Arc<PolydatKernel>,
    canonical: &Arc<PolydatKernel>,
    workload_params: &HashMap<String, String>,
    on_empty: F,
) -> Result<Vec<RuntimeTuple>, RuntimeError>
where
    F: FnMut(EmptyClause<'_>) -> Result<(), String>,
{
    let mut state = EvalState {
        parent: parent.clone(),
        canonical: canonical.clone(),
        workload_params,
        on_empty,
    };
    state.evaluate_node(comp, &[]).map(|n| n.tuples)
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

/// Internal walker state — bundles the closures and shared
/// references so the recursive walker doesn't have to thread
/// them through every call.
struct EvalState<'a, F> {
    parent: Arc<PolydatKernel>,
    canonical: Arc<PolydatKernel>,
    /// Workload-param fallback. The polydat-owned
    /// `evaluate_spec` already routes through the parent
    /// kernel chain for shadow-aware resolution (SRD-21),
    /// so this is unused at the runtime evaluator level —
    /// kept on the surface for symmetry with the legacy
    /// `iterate_scope` signature in case some future
    /// `Source` variant needs param-aware evaluation that
    /// can't go through the kernel.
    #[allow(dead_code)]
    workload_params: &'a HashMap<String, String>,
    on_empty: F,
}

impl<F> EvalState<'_, F>
where
    F: FnMut(EmptyClause<'_>) -> Result<(), String>,
{
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
            } => {
                let inner = self.evaluate_node(child, prefix)?;
                // A continuous source has no tuples of its own; an order
                // strategy with a truncation samples that many points
                // from its intervals (spec §10.7.8, continuous inputs).
                if inner.tuples.is_empty()
                    && let Some(intervals) = continuous_axes(child)
                {
                    let names = child.coordinate_names();
                    return Self::sample_continuous(&names, &intervals, *strategy, *truncation);
                }
                self.apply_order(inner, *strategy, *truncation)
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
            parent: &self.parent,
            canonical: &self.canonical,
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

        if evaluated.values.is_empty() {
            let spec_text = source_display_text(source);
            (self.on_empty)(EmptyClause {
                var: name,
                spec_expr: spec_text.as_deref(),
            })
            .map_err(RuntimeError::EmptyPolicy)?;
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
            let kernel = self
                .parent
                .materialize_subscope(self.canonical.program().clone(), &tuple);
            let interpolated = interpolate_via_kernel(predicate, &kernel).map_err(|e| {
                RuntimeError::FilterEval {
                    predicate: predicate.to_string(),
                    message: e.to_string(),
                }
            })?;
            let result = eval_const_expr(&interpolated).map_err(|e| RuntimeError::FilterEval {
                predicate: predicate.to_string(),
                message: e.to_string(),
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

    /// Sample `truncation` points from continuous intervals with a
    /// space-filling strategy. The strategies encode a continuous
    /// coordinate as a 53-bit fraction of the unit interval; each is
    /// mapped onto its axis's interval and bound to the axis's name.
    fn sample_continuous(
        names: &[String],
        intervals: &[crate::iteration::comprehension::cardinality::Interval],
        strategy: StrategyName,
        truncation: Option<u64>,
    ) -> Result<EvaluatedNode, RuntimeError> {
        use crate::iteration::comprehension::strategies::{
            halton::halton_multi_indices, lhs::lhs_multi_indices, shuffle::shuffle_multi_indices,
            sobol::sobol_multi_indices,
        };
        let Some(n) = truncation else {
            return Err(RuntimeError::OrderEval {
                strategy,
                message: "a continuous source has no finite tuple set; give the order a count, as in `order halton/16`".into(),
            });
        };
        let index_fn = IndexFn::Continuous {
            intervals: intervals.to_vec(),
            measure: crate::iteration::comprehension::cardinality::ProductMeasure::Uniform,
        };
        let points = match strategy {
            StrategyName::Halton => halton_multi_indices(&index_fn, Some(n)),
            StrategyName::Sobol => sobol_multi_indices(&index_fn, Some(n)),
            StrategyName::Lhs => lhs_multi_indices(&index_fn, Some(n)),
            StrategyName::Shuffle => shuffle_multi_indices(&index_fn, Some(n)),
            other => return Err(RuntimeError::OrderEval {
                strategy: other,
                message:
                    "a continuous source needs a sampling strategy: halton, sobol, lhs, or shuffle"
                        .into(),
            }),
        };
        let scale = (1u64 << 53) as f64;
        let tuples = points
            .into_iter()
            .map(|mi| {
                mi.iter()
                    .enumerate()
                    .map(|(axis, u)| {
                        let iv = &intervals[axis.min(intervals.len().saturating_sub(1))];
                        let frac = (*u as f64) / scale;
                        let x = iv.lo + frac * (iv.hi - iv.lo);
                        let name = names
                            .get(axis)
                            .cloned()
                            .unwrap_or_else(|| format!("axis{axis}"));
                        (name, Value::F64(x))
                    })
                    .collect::<RuntimeTuple>()
            })
            .collect();
        Ok(EvaluatedNode {
            tuples,
            index_fn: None,
        })
    }

    fn apply_order(
        &mut self,
        input: EvaluatedNode,
        strategy: StrategyName,
        truncation: Option<u64>,
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

        let ordered = dispatch.apply(&evaluated_input, truncation);

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

/// The intervals of a comprehension whose every clause is continuous,
/// in coordinate order; `None` when any clause is discrete or the shape
/// is not a plain product of clauses.
pub(crate) fn continuous_axes(
    c: &Comprehension,
) -> Option<Vec<crate::iteration::comprehension::cardinality::Interval>> {
    match c {
        Comprehension::Clause {
            source: Source::ContinuousInterval { interval, .. },
            ..
        } => Some(vec![interval.clone()]),
        Comprehension::Clause {
            source: Source::Distribution { support, .. },
            ..
        } => Some(vec![support.clone()]),
        Comprehension::Clause { .. } => None,
        Comprehension::Cartesian { children } => {
            let mut out = Vec::new();
            for ch in children {
                out.extend(continuous_axes(ch)?);
            }
            Some(out)
        }
        Comprehension::Filter { child, .. } => continuous_axes(child),
        Comprehension::Zip { .. } | Comprehension::Union { .. } | Comprehension::Order { .. } => {
            None
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

fn source_display_text(source: &Source) -> Option<String> {
    match source {
        Source::Generator { expr, .. } => Some(expr.clone()),
        Source::WorkloadParamList { name, .. } => Some(format!("{{{name}}}")),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::iteration::comprehension::source::LiteralValue;

    fn empty_kernel() -> Arc<PolydatKernel> {
        Arc::new(crate::dsl::compile_polydat("\n").unwrap())
    }

    /// Canonical kernel with `extern k: u64` so the runtime
    /// evaluator can install per-clause `k` values via
    /// materialize_subscope — matches the shape
    /// `build_for_each_scope_kernel` produces in production.
    fn canonical_with_k() -> Arc<PolydatKernel> {
        Arc::new(crate::dsl::compile_polydat("extern k: u64\n").unwrap())
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
        let parent = empty_kernel();
        let canonical = empty_kernel();
        let params = HashMap::new();
        let tuples =
            evaluate_for_iteration(&comp, &parent, &canonical, &params, |_| Ok(())).unwrap();
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
        let parent = empty_kernel();
        let canonical = empty_kernel();
        let params = HashMap::new();
        let tuples =
            evaluate_for_iteration(&comp, &parent, &canonical, &params, |_| Ok(())).unwrap();
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
        let parent = empty_kernel();
        let canonical = empty_kernel();
        let params = HashMap::new();
        let tuples =
            evaluate_for_iteration(&comp, &parent, &canonical, &params, |_| Ok(())).unwrap();
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
        let parent = empty_kernel();
        let canonical = empty_kernel();
        let params = HashMap::new();
        let tuples =
            evaluate_for_iteration(&comp, &parent, &canonical, &params, |_| Ok(())).unwrap();
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
        let parent = empty_kernel();
        let canonical = canonical_with_k();
        let params = HashMap::new();
        let tuples =
            evaluate_for_iteration(&comp, &parent, &canonical, &params, |_| Ok(())).unwrap();
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
        let parent = empty_kernel();
        let canonical = empty_kernel();
        let params = HashMap::new();
        let tuples =
            evaluate_for_iteration(&comp, &parent, &canonical, &params, |_| Ok(())).unwrap();
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
        let parent = empty_kernel();
        let canonical = empty_kernel();
        let params = HashMap::new();
        let tuples =
            evaluate_for_iteration(&comp, &parent, &canonical, &params, |_| Ok(())).unwrap();
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
