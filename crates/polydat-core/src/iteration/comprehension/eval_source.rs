// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Source evaluation — spec §10.7.0, §10.7.6, §10.7.8.
//!
//! Lifts [`IndexFn`] from a static AST property to a contextual
//! query: every [`Source`] variant answers
//! `evaluate(ctx) -> EvaluatedSource` carrying its materialized
//! values, observed cardinality, and the index function the
//! emitted values actually satisfy.
//!
//! ## Why this layer exists
//!
//! Before this module, [`crate::iteration::comprehension::metadata`]
//! computed `IndexFn` at AST-construction time using only static
//! source attributes (`cardinality_hint`, declared step, etc.).
//! Two classes of sources couldn't claim a useful `IndexFn`:
//!
//! - **`Source::Generator { expr }`** — the spec-text resolves
//!   to a list whose shape is only known after evaluation. The
//!   static path conservatively declared `Lattice { axis_sizes:
//!   [N] }` from `cardinality_hint` (or `Unbounded` without
//!   it), regardless of whether the actual values form a
//!   regular arithmetic progression.
//! - **`Source::WorkloadParamList { name }`** — same: the
//!   parameter's list contents are unknown until kernel
//!   evaluation.
//!
//! Non-`Lex` strategies (Diagonal / Extrema / Shells / Halton /
//! Sobol / Lhs) need the input's real `IndexFn` shape to
//! validate V4 and dispatch their indexed-form algorithms.
//! Without this module, V4 fires (or fails to fire) against
//! a stale static estimate; with this module, V4 fires
//! against the post-evaluation truth.
//!
//! ## Eval classes
//!
//! Per spec §10.7.0, sources partition into three eval classes:
//!
//! | Class | Variants | `evaluate(None)` works? |
//! |---|---|---|
//! | [`EvalClass::Static`] | `Literal`, `IntRange`, a `Generator` whose expression references no name | yes |
//! | [`EvalClass::ContextRequired`] | `WorkloadParamList`, a `Generator` whose expression references a name | no — needs `&Context` |
//! | [`EvalClass::Distribution`] | `ContinuousInterval`, `Distribution` (in their "not yet sampled" state) | yes, but `values` is empty — enclosing `Order(_, sampling-strategy, Some(n))` materializes |
//!
//! The class of a generator is decided by its expression's free
//! names ([`Source::referenced_names`]), never by a table of
//! generator names: a context-free call evaluates in the empty
//! scope ([`crate::kernel::interp::NoScope`]), and the compile
//! flattens it into a literal of its values
//! (`comprehension::flatten`).
//!
//! [`SourceEval::eval_class`] classifies a source for callers
//! that want to know whether `evaluate(None)` will succeed; the
//! compile-time V4 check in `validate` works from AST metadata
//! and does not consult it. V4 otherwise fires at
//! strategy-invocation time per spec §10.7.8.
//!
//! ## What this module DOES NOT own
//!
//! - The runtime walker that combines per-clause
//!   `EvaluatedSource`s into the cartesian / zip / union views
//!   strategies actually consume — that lives in
//!   [`crate::iteration::comprehension::runtime`].
//! - The strategy invocation itself — see
//!   [`crate::iteration::comprehension::strategies::Strategy::apply`].
//! - The compile-time V4 fire — see
//!   [`mod@crate::iteration::comprehension::validate`].

use std::sync::Arc;

use crate::ast::Value;
use crate::iteration::comprehension::cardinality::ProductMeasure;
use crate::iteration::comprehension::metadata::IndexFn;
use crate::iteration::comprehension::source::{LiteralValue, Source};
use crate::kernel::interp::{Layered, Lookup};

/// Result of evaluating one clause's source.
///
/// `values` carries the materialized stream (one [`Value`] per
/// output position). `cardinality` is the count of values
/// (`values.len() as u64`, equivalent to the `IndexFn`'s axis
/// total for discrete sources; `0` for un-sampled continuous
/// sources). `index_fn` is the addressing scheme the emitted
/// values actually satisfy — derived from observed shape for
/// `Generator` / `WorkloadParamList`, declared for static
/// variants.
#[derive(Debug, Clone)]
pub struct EvaluatedSource {
    /// The values, in dispense order.
    pub values: Vec<Value>,
    /// How many values; zero for an unsampled continuous source.
    pub cardinality: u64,
    /// The addressing scheme the values satisfy.
    pub index_fn: IndexFn,
}

/// Spec §10.7.0 partitioning.
///
/// Tells a caller whether a source can be materialized with
/// `ctx = None`. The compile-time V4 check in `validate` works
/// from AST metadata and does not consult this; V4 otherwise
/// fires at strategy-invocation time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvalClass {
    /// Statically evaluable with no kernel / param context.
    /// `evaluate(None)` returns a fully-populated
    /// [`EvaluatedSource`].
    Static,

    /// Requires a kernel context to resolve interpolation
    /// references or workload-param lookups.
    /// `evaluate(None)` returns [`EvalError::NeedsContext`].
    ContextRequired,

    /// Continuous measure / distribution. `evaluate(None)`
    /// succeeds but emits an empty `values` vector; the
    /// `IndexFn` is `Continuous`. The enclosing sampling
    /// `Order(_, strategy, Some(n))` materializes draws.
    Distribution,
}

/// Errors returned by [`SourceEval::evaluate`].
#[derive(Debug, Clone)]
pub enum EvalError {
    /// The source needs a kernel context that wasn't provided.
    NeedsContext,

    /// Evaluation against the supplied context failed. `var`
    /// names the clause; `source` is the spec-text or
    /// description; `message` carries the underlying reason.
    EvalFailed {
        /// The clause's element name.
        var: String,
        /// The source text or description.
        source: String,
        /// The underlying reason.
        message: String,
    },
}

impl std::fmt::Display for EvalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EvalError::NeedsContext => f.write_str("source evaluation needs a kernel context"),
            EvalError::EvalFailed {
                var,
                source,
                message,
            } => {
                write!(f, "source '{var} in {source}': {message}")
            }
        }
    }
}

impl std::error::Error for EvalError {}

/// Per-evaluation context for context-required sources.
///
/// Carries the live kernel against which `Source::Generator`
/// spec-text and `Source::WorkloadParamList` lookups resolve.
/// `var_name` lets the source synthesise a useful error
/// message; `prefix` is the prior-axis bindings the evaluator
/// layers in front of `scope` (via `Layered`) so dependent
/// sources see earlier-axis values.
pub struct EvalContext<'a> {
    /// The clause's element name, for messages.
    pub var_name: &'a str,
    /// Where the source's names resolve: the body's scope with the
    /// parent's cascaded wires.
    pub scope: &'a dyn Lookup,
    /// The prior-axis bindings, in axis order.
    pub prefix: &'a [(String, Value)],
}

/// The source-evaluation surface.
///
/// Each [`Source`] variant implements this. The trait is
/// object-safe but typically called through the inherent
/// [`Source`] methods below.
pub trait SourceEval {
    /// Classify this source for the IR planner per spec
    /// §10.7.0. See [`EvalClass`].
    fn eval_class(&self) -> EvalClass;

    /// Materialize this source.
    ///
    /// Literal / IntRange (`Static`), a context-free Generator
    /// (`Static`, evaluated in the empty scope), and
    /// ContinuousInterval / Distribution (`Distribution`) accept
    /// `ctx = None`. A Generator that references a name and a
    /// WorkloadParamList (`ContextRequired`) require `Some(ctx)` and
    /// return [`EvalError::NeedsContext`] otherwise.
    fn evaluate(&self, ctx: Option<&EvalContext<'_>>) -> Result<EvaluatedSource, EvalError>;
}

impl SourceEval for Source {
    fn eval_class(&self) -> EvalClass {
        match self {
            Source::Literal { .. } | Source::IntRange { .. } => EvalClass::Static,
            Source::ContinuousInterval { .. } | Source::Distribution { .. } => {
                EvalClass::Distribution
            }
            // A generator's class is its expression's: context-free
            // when it references no name (spec §10.7.0).
            Source::Generator { .. } if self.referenced_names().is_empty() => EvalClass::Static,
            Source::Generator { .. } => EvalClass::ContextRequired,
            Source::WorkloadParamList { .. } => EvalClass::ContextRequired,
        }
    }

    fn evaluate(&self, ctx: Option<&EvalContext<'_>>) -> Result<EvaluatedSource, EvalError> {
        match self {
            Source::Literal { values } => {
                let vals: Vec<Value> = values.iter().map(literal_to_value).collect();
                let n = vals.len() as u64;
                Ok(EvaluatedSource {
                    values: vals,
                    cardinality: n,
                    // Literal lists carry no shape claim other
                    // than length — call them a 1-axis Lattice
                    // of that length. Strategies that need
                    // arithmetic progression shape (e.g. Halton
                    // over a Lattice axis) still get useful
                    // behavior because the lookup is by index,
                    // not by value.
                    index_fn: IndexFn::Lattice {
                        axis_sizes: vec![n],
                    },
                })
            }
            Source::IntRange { lo, hi, step } => {
                let step = (*step).max(1);
                let mut vals = Vec::new();
                let mut cur = *lo;
                while cur < *hi {
                    vals.push(Value::U64(cur as u64));
                    cur += step;
                }
                let n = vals.len() as u64;
                Ok(EvaluatedSource {
                    values: vals,
                    cardinality: n,
                    index_fn: IndexFn::Lattice {
                        axis_sizes: vec![n],
                    },
                })
            }
            Source::Generator { .. } | Source::WorkloadParamList { .. } => {
                let spec_text = match self {
                    Source::Generator { expr, .. } => expr.clone(),
                    Source::WorkloadParamList { name, .. } => format!("{{{name}}}"),
                    _ => unreachable!(),
                };
                // A context-free generator evaluates in the empty
                // scope; anything that references a name needs the
                // caller's.
                let empty = crate::kernel::interp::NoScope::new();
                let (var_name, scope): (&str, Layered<'_>) = match ctx {
                    Some(ctx) => (
                        ctx.var_name,
                        Layered {
                            prefix: ctx.prefix,
                            inner: ctx.scope,
                        },
                    ),
                    None if self.eval_class() == EvalClass::Static => (
                        "<context-free>",
                        Layered {
                            prefix: &[],
                            inner: &empty,
                        },
                    ),
                    None => return Err(EvalError::NeedsContext),
                };
                let vals = crate::iteration::comprehension::eval::evaluate_spec(&spec_text, &scope)
                    .map_err(|e| EvalError::EvalFailed {
                        var: var_name.to_string(),
                        source: spec_text,
                        message: e.to_string(),
                    })?;
                let n = vals.len() as u64;
                let index_fn = classify_observed_values(&vals);
                Ok(EvaluatedSource {
                    values: vals,
                    cardinality: n,
                    index_fn,
                })
            }
            Source::ContinuousInterval { interval, measure } => Ok(EvaluatedSource {
                values: Vec::new(),
                cardinality: 0,
                index_fn: IndexFn::Continuous {
                    intervals: vec![interval.clone()],
                    measure: measure.clone(),
                },
            }),
            Source::Distribution {
                distribution,
                support,
                ..
            } => Ok(EvaluatedSource {
                values: Vec::new(),
                cardinality: 0,
                // The parameters travel on the AST carrier; the
                // runtime's sampler reads them there (spec §10.7.6).
                index_fn: IndexFn::Continuous {
                    intervals: vec![support.clone()],
                    measure: ProductMeasure::Named(*distribution),
                },
            }),
        }
    }
}

/// Classify a materialized value list by observed shape.
///
/// The "expand-then-classify" stage of spec §10.7.6 / §10.7.8:
/// a numeric arithmetic progression →
/// `Lattice { axis_sizes: [N] }` reflecting the regular stride.
/// Non-numeric or non-progression value lists → a plain
/// `Lattice { axis_sizes: [N] }` whose only shape claim is
/// length. Either way the strategy gets a useful 1-axis Lattice
/// for indexed-form dispatch.
///
/// A static generator catalogue that declares shape from args
/// without expansion is not implemented.
fn classify_observed_values(vals: &[Value]) -> IndexFn {
    let n = vals.len() as u64;
    IndexFn::Lattice {
        axis_sizes: vec![n],
    }
}

fn literal_to_value(lv: &LiteralValue) -> Value {
    match lv {
        LiteralValue::Int(n) => Value::U64(*n as u64),
        LiteralValue::Float(f) => Value::F64(*f),
        LiteralValue::String(s) => Value::Str(Arc::from(s.as_str())),
        LiteralValue::Bool(b) => Value::Bool(*b),
        LiteralValue::Json(j) => Value::Json(Arc::new(j.clone())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::iteration::comprehension::cardinality::{Interval, MeasureName, ProductMeasure};
    use crate::iteration::comprehension::source::LiteralValue;

    #[test]
    fn literal_evaluates_without_context() {
        let s = Source::Literal {
            values: vec![
                LiteralValue::Int(1),
                LiteralValue::Int(2),
                LiteralValue::Int(3),
            ],
        };
        assert_eq!(s.eval_class(), EvalClass::Static);
        let ev = s.evaluate(None).unwrap();
        assert_eq!(ev.cardinality, 3);
        assert_eq!(ev.values.len(), 3);
        assert!(matches!(ev.index_fn, IndexFn::Lattice { axis_sizes: ref a } if a == &vec![3]));
    }

    #[test]
    fn int_range_evaluates_without_context() {
        let s = Source::IntRange {
            lo: 0,
            hi: 10,
            step: 2,
        };
        assert_eq!(s.eval_class(), EvalClass::Static);
        let ev = s.evaluate(None).unwrap();
        // 0, 2, 4, 6, 8 = 5 values
        assert_eq!(ev.cardinality, 5);
        assert!(matches!(ev.index_fn, IndexFn::Lattice { axis_sizes: ref a } if a == &vec![5]));
    }

    #[test]
    fn a_context_free_generator_evaluates_without_context() {
        let s = Source::Generator {
            expr: "fib(6)".into(),
            cardinality_hint: None,
        };
        assert_eq!(s.eval_class(), EvalClass::Static);
        let ev = s.evaluate(None).unwrap();
        assert_eq!(ev.cardinality, 6);
    }

    #[test]
    fn generator_without_context_errors() {
        let s = Source::Generator {
            expr: "range(0, {n})".into(),
            cardinality_hint: Some(10),
        };
        assert_eq!(s.eval_class(), EvalClass::ContextRequired);
        match s.evaluate(None) {
            Err(EvalError::NeedsContext) => {}
            other => panic!("expected NeedsContext, got {other:?}"),
        }
    }

    #[test]
    fn workload_param_list_without_context_errors() {
        let s = Source::WorkloadParamList {
            name: "k_values".into(),
            len_hint: Some(5),
        };
        assert_eq!(s.eval_class(), EvalClass::ContextRequired);
        assert!(matches!(s.evaluate(None), Err(EvalError::NeedsContext)));
    }

    #[test]
    fn continuous_interval_yields_continuous_index_fn() {
        let s = Source::ContinuousInterval {
            interval: Interval::closed(0.0, 1.0),
            measure: ProductMeasure::Uniform,
        };
        assert_eq!(s.eval_class(), EvalClass::Distribution);
        let ev = s.evaluate(None).unwrap();
        assert_eq!(ev.cardinality, 0);
        assert!(ev.values.is_empty());
        match ev.index_fn {
            IndexFn::Continuous { intervals, .. } => assert_eq!(intervals.len(), 1),
            other => panic!("expected Continuous, got {other:?}"),
        }
    }

    #[test]
    fn distribution_yields_continuous_index_fn() {
        let s = Source::Distribution {
            distribution: MeasureName::Normal,
            support: Interval {
                lo: f64::NEG_INFINITY,
                hi: f64::INFINITY,
                lo_open: true,
                hi_open: true,
            },
            params: vec![0.0, 1.0],
        };
        assert_eq!(s.eval_class(), EvalClass::Distribution);
        let ev = s.evaluate(None).unwrap();
        assert_eq!(ev.cardinality, 0);
        assert!(matches!(
            ev.index_fn,
            IndexFn::Continuous {
                measure: ProductMeasure::Named(MeasureName::Normal),
                ..
            }
        ));
    }

    #[test]
    fn generator_with_context_evaluates_to_lattice() {
        let canonical = Arc::new(crate::dsl::compile_polydat("\n").unwrap());
        let s = Source::Generator {
            expr: "1, 2, 3, 4, 5".into(),
            cardinality_hint: Some(5),
        };
        let ctx = EvalContext {
            var_name: "k",
            scope: &*canonical,
            prefix: &[],
        };
        let ev = s.evaluate(Some(&ctx)).unwrap();
        assert_eq!(ev.cardinality, 5);
        assert!(matches!(ev.index_fn, IndexFn::Lattice { .. }));
    }
}
