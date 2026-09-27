// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Compile-time flattening of context-free sources
//! (comprehension_forms.md §10.7.0, §10.7.7). Whether a generator
//! call can be evaluated before traversal is a property of its
//! expression, not of its name: a call that references no name, and
//! none that `scope` cannot resolve, is evaluated once here, and its
//! clause becomes a literal of the values it produced. The values are
//! then what the runtime binds, computed once instead of on every
//! activation, and the clause's cardinality is exact, so the V-axioms
//! that need a bound (V4, V6) fire at compile with the real shape.
//!
//! A call whose values are not literal-representable (a partition
//! list, for one) keeps its call and gains the evaluated count as its
//! cardinality hint. A call the compile cannot evaluate (a node call
//! the const evaluator does not fold, one a traversal binds through
//! its kernel) is kept as well. A call that references a coordinate
//! the comprehension binds, or a name `scope` does not resolve, is
//! left for the traversal, where the runtime's scope binds those
//! names.
//!
//! A named generator's values depend on its arguments alone, so a
//! context-free call of one that refuses its arguments fails at every
//! evaluation. [`first_refused_generator`] finds such a call for the
//! compile to report as its error (comprehension_forms.md §3.1.3).

use std::collections::BTreeSet;

use crate::ast::Value;
use crate::iteration::comprehension::ast::Comprehension;
use crate::iteration::comprehension::eval_source::{EvalContext, SourceEval};
use crate::iteration::comprehension::source::{LiteralValue, Source};
use crate::kernel::interp::Lookup;

/// Flatten the context-free sources of `ast` against `scope`. Every
/// clause the compile cannot evaluate is kept as written, so this
/// never fails: the traversal evaluates what is left.
pub fn flatten_static_sources(ast: &Comprehension, scope: &dyn Lookup) -> Comprehension {
    let bound: BTreeSet<String> = ast.coordinate_names().into_iter().collect();
    flatten_node(ast, scope, &bound)
}

fn flatten_node(c: &Comprehension, scope: &dyn Lookup, bound: &BTreeSet<String>) -> Comprehension {
    match c {
        Comprehension::Clause { name, source } => Comprehension::Clause {
            name: name.clone(),
            source: flatten_source(name, source, scope, bound),
        },
        Comprehension::Cartesian { children } => Comprehension::Cartesian {
            children: flatten_all(children, scope, bound),
        },
        Comprehension::Zip { children, mode } => Comprehension::Zip {
            children: flatten_all(children, scope, bound),
            mode: *mode,
        },
        Comprehension::Union { children } => Comprehension::Union {
            children: flatten_all(children, scope, bound),
        },
        Comprehension::Filter { child, predicate } => Comprehension::Filter {
            child: Box::new(flatten_node(child, scope, bound)),
            predicate: predicate.clone(),
        },
        Comprehension::Order {
            child,
            strategy,
            truncation,
            seed,
        } => Comprehension::Order {
            child: Box::new(flatten_node(child, scope, bound)),
            strategy: *strategy,
            truncation: *truncation,
            seed: *seed,
        },
    }
}

fn flatten_all(
    children: &[Comprehension],
    scope: &dyn Lookup,
    bound: &BTreeSet<String>,
) -> Vec<Comprehension> {
    children
        .iter()
        .map(|c| flatten_node(c, scope, bound))
        .collect()
}

/// The source as the compile leaves it: a generator whose names are
/// all resolvable here is evaluated, and everything else is kept.
fn flatten_source(
    name: &str,
    source: &Source,
    scope: &dyn Lookup,
    bound: &BTreeSet<String>,
) -> Source {
    let Source::Generator { expr, .. } = source else {
        return source.clone();
    };
    if context_required(source, scope, bound) {
        return source.clone();
    }
    let ctx = EvalContext {
        var_name: name,
        scope,
        prefix: &[],
    };
    // The compile's evaluator folds const expressions; a traversal's
    // scope can evaluate more (a node call bound through its kernel,
    // for one), so a call the compile cannot evaluate is left to it,
    // and is the traversal's error if it cannot either.
    let Ok(evaluated) = source.evaluate(Some(&ctx)) else {
        return source.clone();
    };
    let count = evaluated.cardinality;
    if count == 0 {
        // An empty result is the traversal's empty-clause policy to
        // decide, not a literal with nothing in it.
        return Source::Generator {
            expr: expr.clone(),
            cardinality_hint: Some(0),
        };
    }
    let literals: Option<Vec<LiteralValue>> = evaluated.values.iter().map(literal_of).collect();
    match literals {
        Some(values) => Source::Literal { values },
        None => Source::Generator {
            expr: expr.clone(),
            cardinality_hint: Some(count),
        },
    }
}

/// Whether `source` reads a coordinate the comprehension binds or a
/// name `scope` does not resolve ([`Source::names_read`]).
fn context_required(source: &Source, scope: &dyn Lookup, bound: &BTreeSet<String>) -> bool {
    source
        .names_read()
        .iter()
        .any(|n| bound.contains(n) || scope.lookup(n).is_none())
}

/// The first clause of `ast` whose source calls a named generator that
/// refuses its arguments, with the refusal, when every name the source
/// references resolves in `scope` (comprehension_forms.md §3.1.3). Such
/// a call fails wherever it is evaluated, so a compile that flattens
/// against `scope` reports it as its own error rather than keeping the
/// call for the traversal. A source that references a coordinate the
/// comprehension binds, or a name `scope` does not resolve, is checked
/// when the traversal evaluates it.
pub fn first_refused_generator(
    ast: &Comprehension,
    scope: &dyn Lookup,
) -> Option<(String, String)> {
    let bound: BTreeSet<String> = ast.coordinate_names().into_iter().collect();
    refused_in(ast, scope, &bound)
}

fn refused_in(
    c: &Comprehension,
    scope: &dyn Lookup,
    bound: &BTreeSet<String>,
) -> Option<(String, String)> {
    match c {
        Comprehension::Clause { name, source } => match source {
            Source::Generator { expr, .. } if !context_required(source, scope, bound) => {
                crate::iteration::comprehension::eval::refused_generator_call(expr, scope)
                    .map(|message| (name.clone(), message))
            }
            _ => None,
        },
        Comprehension::Cartesian { children }
        | Comprehension::Zip { children, .. }
        | Comprehension::Union { children } => {
            children.iter().find_map(|c| refused_in(c, scope, bound))
        }
        Comprehension::Filter { child, .. } | Comprehension::Order { child, .. } => {
            refused_in(child, scope, bound)
        }
    }
}

/// The literal a value binds as, when it has a literal form.
fn literal_of(v: &Value) -> Option<LiteralValue> {
    Some(match v {
        Value::U64(n) => LiteralValue::unsigned(*n),
        Value::I64(n) => LiteralValue::Int(*n),
        Value::F64(x) => LiteralValue::Float(*x),
        Value::Bool(b) => LiteralValue::Bool(*b),
        Value::Str(s) => LiteralValue::String(s.to_string()),
        Value::Json(j) => LiteralValue::Json(j.as_ref().clone()),
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::iteration::comprehension::source::Source;
    use crate::kernel::interp::NoScope;

    fn generator(name: &str, expr: &str) -> Comprehension {
        Comprehension::clause(
            name,
            Source::Generator {
                expr: expr.into(),
                cardinality_hint: None,
            },
        )
    }

    #[test]
    fn a_context_free_generator_becomes_a_literal_of_its_values() {
        let out = flatten_static_sources(&generator("k", "fib(6)"), &NoScope::new());
        let Comprehension::Clause {
            source: Source::Literal { values },
            ..
        } = out
        else {
            panic!("expected a literal clause, got {out:?}");
        };
        assert_eq!(
            values,
            vec![
                LiteralValue::Int(1),
                LiteralValue::Int(1),
                LiteralValue::Int(2),
                LiteralValue::Int(3),
                LiteralValue::Int(5),
                LiteralValue::Int(8),
            ]
        );
    }

    #[test]
    fn a_generator_over_a_bound_coordinate_or_an_unresolved_name_is_kept() {
        // `{k}` is bound by the sibling clause: dependent, evaluated
        // at traversal.
        let ast =
            Comprehension::cartesian(vec![generator("k", "fib(3)"), generator("j", "pow2({k})")]);
        let out = flatten_static_sources(&ast, &NoScope::new());
        let Comprehension::Cartesian { children } = &out else {
            panic!("{out:?}");
        };
        assert!(matches!(
            &children[0],
            Comprehension::Clause {
                source: Source::Literal { .. },
                ..
            }
        ));
        assert!(matches!(
            &children[1],
            Comprehension::Clause {
                source: Source::Generator { expr, cardinality_hint: None },
                ..
            } if expr == "pow2({k})"
        ));
        // A name the scope does not resolve: context-required.
        let out = flatten_static_sources(&generator("k", "pow2(n)"), &NoScope::new());
        assert!(matches!(
            out,
            Comprehension::Clause {
                source: Source::Generator {
                    cardinality_hint: None,
                    ..
                },
                ..
            }
        ));
    }

    #[test]
    fn values_without_a_literal_form_keep_the_call_and_gain_the_count() {
        let out =
            flatten_static_sources(&generator("p", "partitions(\"*/4\", 100)"), &NoScope::new());
        assert!(
            matches!(
                out,
                Comprehension::Clause {
                    source: Source::Generator {
                        cardinality_hint: Some(4),
                        ..
                    },
                    ..
                }
            ),
            "{out:?}"
        );
    }

    #[test]
    fn a_call_the_compile_cannot_evaluate_is_left_to_the_traversal() {
        let out = flatten_static_sources(&generator("k", "fib(-1)"), &NoScope::new());
        assert_eq!(out, generator("k", "fib(-1)"));
    }

    /// A generator value above `i64::MAX` keeps its value as an
    /// unsigned literal, so the clause is a literal like any other.
    #[test]
    fn a_value_above_i64_max_flattens_to_an_unsigned_literal() {
        let out = flatten_static_sources(&generator("k", "pow2(64)"), &NoScope::new());
        let Comprehension::Clause {
            source: Source::Literal { values },
            ..
        } = out
        else {
            panic!("expected a literal clause, got {out:?}");
        };
        assert_eq!(values.len(), 64);
        assert_eq!(values[62], LiteralValue::Int(1 << 62));
        assert_eq!(values[63], LiteralValue::UInt(1 << 63));
    }

    /// A named generator that refuses its arguments is found, by clause,
    /// when its arguments resolve; one that references a bound
    /// coordinate or an unresolved name is left for the traversal.
    #[test]
    fn a_refused_generator_is_found_when_its_arguments_resolve() {
        let scope = NoScope::new();
        let ast = Comprehension::cartesian(vec![
            generator("a", "fib(3)"),
            generator("b", "binomial(70)"),
        ]);
        let flat = flatten_static_sources(&ast, &scope);
        let (name, message) = first_refused_generator(&flat, &scope).unwrap();
        assert_eq!(name, "b");
        assert!(
            message.starts_with("binomial(70): term C(70, "),
            "{message}"
        );
        let dependent =
            Comprehension::cartesian(vec![generator("k", "fib(3)"), generator("j", "fib({k})")]);
        assert_eq!(first_refused_generator(&dependent, &scope), None);
        assert_eq!(
            first_refused_generator(&generator("k", "fib(n)"), &scope),
            None
        );
    }
}
