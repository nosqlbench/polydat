// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Evaluating a comprehension's `where` predicate against a tuple.
//!
//! A [`CompiledPredicate`] parses the predicate once
//! ([`parse_predicate`]) and tests each tuple against the tree. `||`,
//! `&&`, `!`, the comparisons, and membership evaluate directly over
//! the tuple's values. `&&` and `||` evaluate their operands left to
//! right and stop at the first that decides the result, so filters
//! folded into one conjunction (R6) evaluate exactly what the chain
//! did; `!` negates its operand's truth. A
//! `{name}` the tuple does not bind resolves in the scope the tuple was
//! drawn in. Any other expression (arithmetic, a function call, a cast)
//! is interpolated against the tuple over that scope and evaluated as a
//! Polydat expression, charged to the scope's ledger.
//!
//! Values compare as scalars: integers and floats compare numerically
//! with each other, strings and booleans with their own kind, and a
//! value of one kind is never equal to a value of another, so
//! `"s0" != 2` holds. Ordering values of different kinds is an error.
//! A predicate that does not parse is evaluated whole as a Polydat
//! expression. The value a predicate yields is true when it is `true`
//! or a non-zero number.

use crate::ast::Value;
use crate::dsl::compile::eval_const_expr_for;
use crate::iteration::comprehension::ast::Comprehension;
use crate::iteration::comprehension::runtime::{RuntimeError, RuntimeTuple};
use crate::iteration::comprehension::source::{LiteralValue, Source};
use crate::kernel::interp::{Layered, Lookup, interpolate_via_kernel};
use polydat_grammar::comprehension::predicate::{
    Comparison, Predicate, PredicateKind, PredicateLiteral, parse_predicate,
};

/// A predicate parsed once, testing tuples.
#[derive(Debug, Clone)]
pub struct CompiledPredicate {
    text: String,
    tree: Predicate,
}

/// A value a predicate compares.
#[derive(Debug, Clone, PartialEq)]
enum Scalar {
    Int(i128),
    Float(f64),
    Str(String),
    Bool(bool),
}

impl std::fmt::Display for Scalar {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Scalar::Int(n) => write!(f, "{n}"),
            Scalar::Float(x) => write!(f, "{x}"),
            Scalar::Str(s) => write!(f, "{s:?}"),
            Scalar::Bool(b) => write!(f, "{b}"),
        }
    }
}

impl CompiledPredicate {
    /// Parse `text`.
    pub fn new(text: &str) -> Self {
        let tree = parse_predicate(text).unwrap_or(Predicate {
            kind: PredicateKind::Expr,
            span: 0..text.len(),
        });
        Self {
            text: text.to_string(),
            tree,
        }
    }

    /// The predicate as written.
    pub fn text(&self) -> &str {
        &self.text
    }

    /// Whether `tuple`, drawn in `scope`, passes.
    pub fn keeps(&self, tuple: &RuntimeTuple, scope: &dyn Lookup) -> Result<bool, RuntimeError> {
        self.eval(&self.tree, tuple, scope)
            .and_then(|v| truth(&v))
            .map_err(|message| RuntimeError::FilterEval {
                predicate: self.text.clone(),
                message,
            })
    }

    fn eval(
        &self,
        node: &Predicate,
        tuple: &RuntimeTuple,
        scope: &dyn Lookup,
    ) -> Result<Scalar, String> {
        Ok(match &node.kind {
            PredicateKind::Or(parts) => {
                for part in parts {
                    if truth(&self.eval(part, tuple, scope)?)? {
                        return Ok(Scalar::Bool(true));
                    }
                }
                Scalar::Bool(false)
            }
            PredicateKind::And(parts) => {
                for part in parts {
                    if !truth(&self.eval(part, tuple, scope)?)? {
                        return Ok(Scalar::Bool(false));
                    }
                }
                Scalar::Bool(true)
            }
            PredicateKind::Not(inner) => Scalar::Bool(!truth(&self.eval(inner, tuple, scope)?)?),
            PredicateKind::Compare(op, a, b) => {
                let a = self.eval(a, tuple, scope)?;
                let b = self.eval(b, tuple, scope)?;
                Scalar::Bool(compare(*op, &a, &b)?)
            }
            PredicateKind::In(needle, items) => {
                let needle = self.eval(needle, tuple, scope)?;
                let mut hit = false;
                for item in items {
                    hit |= scalar_eq(&needle, &self.eval(item, tuple, scope)?);
                }
                Scalar::Bool(hit)
            }
            PredicateKind::Element(name) => {
                let value = match tuple.iter().find(|(n, _)| n == name) {
                    Some((_, v)) => v.clone(),
                    None => scope
                        .lookup(name)
                        .ok_or_else(|| format!("`{{{name}}}` is not bound"))?,
                };
                scalar(&value).ok_or_else(|| format!("`{{{name}}}` is {value:?}, not a scalar"))?
            }
            PredicateKind::Literal(literal) => match literal {
                PredicateLiteral::Int(n) => Scalar::Int(*n),
                PredicateLiteral::Float(f) => Scalar::Float(*f),
                PredicateLiteral::Str(s) => Scalar::Str(s.clone()),
                PredicateLiteral::Bool(b) => Scalar::Bool(*b),
            },
            PredicateKind::Arith(..) | PredicateKind::Expr => {
                let text = node.text(&self.text);
                let layered = Layered {
                    prefix: tuple,
                    inner: scope,
                };
                let interpolated =
                    interpolate_via_kernel(text, &layered).map_err(|e| e.to_string())?;
                let value = eval_const_expr_for(&interpolated, scope.ledger())
                    .map_err(|e| e.to_string())?;
                scalar(&value).ok_or_else(|| format!("`{text}` is {value:?}, not a scalar"))?
            }
        })
    }
}

/// The kind every value of an element has, as its source declares it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValueKind {
    /// Unsigned integers: a range, or a list of integers.
    Int,
    /// Floats: a list of floats, or a continuous interval.
    Float,
    /// Strings.
    Str,
    /// Booleans.
    Bool,
}

impl CompiledPredicate {
    /// Whether no evaluation of the predicate can fail over tuples whose
    /// elements have the kinds `kind_of` gives, `None` for an element of
    /// unknown or mixed kind (comprehension_forms.md §10.2 R5).
    ///
    /// The answer is read off the predicate's tree. `||`, `&&`, and `!`
    /// are total over operands that are total and have a truth value (a
    /// boolean or a number, never a string). `==`, `!=`, and membership
    /// are total over total operands, since values of different kinds
    /// are unequal; an ordering comparison is total only when both sides
    /// are numbers, both strings, or both booleans. Arithmetic is total
    /// over numeric operands, dividing only by a non-zero constant. An
    /// element is total when its kind is known, a literal always is, and
    /// any other expression (a function call, a cast, a bitwise
    /// operator) can fail.
    pub fn is_total(&self, kind_of: &dyn Fn(&str) -> Option<ValueKind>) -> bool {
        total_kind(&self.tree, kind_of).is_some_and(has_truth)
    }
}

/// The kind of value `node` yields when no evaluation of it can fail,
/// or `None` when one can.
fn total_kind(node: &Predicate, kind_of: &dyn Fn(&str) -> Option<ValueKind>) -> Option<ValueKind> {
    use polydat_grammar::ast::BinOpKind;
    let truthful = |p: &Predicate| total_kind(p, kind_of).is_some_and(has_truth);
    match &node.kind {
        PredicateKind::Or(parts) | PredicateKind::And(parts) => {
            parts.iter().all(truthful).then_some(ValueKind::Bool)
        }
        PredicateKind::Not(inner) => truthful(inner).then_some(ValueKind::Bool),
        PredicateKind::Compare(op, a, b) => {
            let (a, b) = (total_kind(a, kind_of)?, total_kind(b, kind_of)?);
            let ordered = matches!(
                (a, b),
                (
                    ValueKind::Int | ValueKind::Float,
                    ValueKind::Int | ValueKind::Float
                ) | (ValueKind::Str, ValueKind::Str)
                    | (ValueKind::Bool, ValueKind::Bool)
            );
            (matches!(op, Comparison::Eq | Comparison::Ne) || ordered).then_some(ValueKind::Bool)
        }
        PredicateKind::In(needle, items) => {
            total_kind(needle, kind_of)?;
            for item in items {
                total_kind(item, kind_of)?;
            }
            Some(ValueKind::Bool)
        }
        PredicateKind::Element(name) => kind_of(name),
        PredicateKind::Literal(literal) => Some(match literal {
            PredicateLiteral::Int(_) => ValueKind::Int,
            PredicateLiteral::Float(_) => ValueKind::Float,
            PredicateLiteral::Str(_) => ValueKind::Str,
            PredicateLiteral::Bool(_) => ValueKind::Bool,
        }),
        PredicateKind::Arith(op, a, b) => {
            // An operand is written into the expression's text, so a
            // literal is one the language reads back as itself: a
            // non-negative integer within `u64`, or a non-negative float.
            let operand = |p: &Predicate| {
                let fits = match &p.kind {
                    PredicateKind::Literal(PredicateLiteral::Int(n)) => u64::try_from(*n).is_ok(),
                    PredicateKind::Literal(PredicateLiteral::Float(f)) => *f >= 0.0,
                    _ => true,
                };
                total_kind(p, kind_of)
                    .filter(|k| fits && matches!(k, ValueKind::Int | ValueKind::Float))
            };
            let (ka, kb) = (operand(a)?, operand(b)?);
            if matches!(op, BinOpKind::Div | BinOpKind::Mod) {
                let non_zero_constant = match &b.kind {
                    PredicateKind::Literal(PredicateLiteral::Int(n)) => *n != 0,
                    PredicateKind::Literal(PredicateLiteral::Float(f)) => *f != 0.0,
                    _ => false,
                };
                if !non_zero_constant {
                    return None;
                }
            }
            Some(
                if ka == ValueKind::Int && kb == ValueKind::Int && *op != BinOpKind::Pow {
                    ValueKind::Int
                } else {
                    ValueKind::Float
                },
            )
        }
        PredicateKind::Expr => None,
    }
}

/// Whether a value of this kind has a truth value.
fn has_truth(kind: ValueKind) -> bool {
    kind != ValueKind::Str
}

/// The kind of every value `name` takes in `c`: the kind its clause's
/// source declares, the same in every branch of a union, or `None` when
/// a source's values are of unknown or mixed kind or `c` does not bind
/// `name`.
pub fn element_kind(c: &Comprehension, name: &str) -> Option<ValueKind> {
    let mut kinds = Vec::new();
    collect_kinds(c, name, &mut kinds);
    let first = (*kinds.first()?)?;
    kinds.iter().all(|k| *k == Some(first)).then_some(first)
}

fn collect_kinds(c: &Comprehension, name: &str, out: &mut Vec<Option<ValueKind>>) {
    match c {
        Comprehension::Clause { name: n, source } if n == name => out.push(source_kind(source)),
        Comprehension::Clause { .. } => {}
        Comprehension::Cartesian { children }
        | Comprehension::Zip { children, .. }
        | Comprehension::Union { children } => {
            for child in children {
                collect_kinds(child, name, out);
            }
        }
        Comprehension::Filter { child, .. } | Comprehension::Order { child, .. } => {
            collect_kinds(child, name, out);
        }
    }
}

/// The kind of every value `source` yields, when it declares one.
fn source_kind(source: &Source) -> Option<ValueKind> {
    match source {
        Source::IntRange { .. } => Some(ValueKind::Int),
        Source::ContinuousInterval { .. } | Source::Distribution { .. } => Some(ValueKind::Float),
        Source::Literal { values } => {
            let kind = |v: &LiteralValue| match v {
                LiteralValue::Int(_) => Some(ValueKind::Int),
                LiteralValue::Float(_) => Some(ValueKind::Float),
                LiteralValue::String(_) => Some(ValueKind::Str),
                LiteralValue::Bool(_) => Some(ValueKind::Bool),
                LiteralValue::Json(_) => None,
            };
            let first = kind(values.first()?)?;
            values
                .iter()
                .all(|v| kind(v) == Some(first))
                .then_some(first)
        }
        Source::Generator { .. } | Source::WorkloadParamList { .. } => None,
    }
}

/// The scalar a value carries; a JSON list's item compares as its own
/// scalar.
fn scalar(value: &Value) -> Option<Scalar> {
    Some(match value {
        Value::U64(n) => Scalar::Int(i128::from(*n)),
        Value::I64(n) => Scalar::Int(i128::from(*n)),
        Value::F64(f) => Scalar::Float(*f),
        Value::Str(s) => Scalar::Str(s.to_string()),
        Value::Bool(b) => Scalar::Bool(*b),
        Value::Json(j) => match j.as_ref() {
            serde_json::Value::Number(n) if n.is_i64() => Scalar::Int(i128::from(n.as_i64()?)),
            serde_json::Value::Number(n) if n.is_u64() => Scalar::Int(i128::from(n.as_u64()?)),
            serde_json::Value::Number(n) => Scalar::Float(n.as_f64()?),
            serde_json::Value::String(s) => Scalar::Str(s.clone()),
            serde_json::Value::Bool(b) => Scalar::Bool(*b),
            _ => return None,
        },
        _ => return None,
    })
}

/// A value's truth: `true`, or a non-zero number.
fn truth(value: &Scalar) -> Result<bool, String> {
    match value {
        Scalar::Bool(b) => Ok(*b),
        Scalar::Int(n) => Ok(*n != 0),
        Scalar::Float(f) => Ok(*f != 0.0),
        Scalar::Str(s) => Err(format!("expected bool/u64/f64, got {s:?}")),
    }
}

fn scalar_eq(a: &Scalar, b: &Scalar) -> bool {
    match (a, b) {
        (Scalar::Int(x), Scalar::Float(y)) | (Scalar::Float(y), Scalar::Int(x)) => {
            (*x as f64) == *y
        }
        _ => a == b,
    }
}

fn compare(op: Comparison, a: &Scalar, b: &Scalar) -> Result<bool, String> {
    use std::cmp::Ordering;
    let ordering = match op {
        Comparison::Eq => return Ok(scalar_eq(a, b)),
        Comparison::Ne => return Ok(!scalar_eq(a, b)),
        _ => match (a, b) {
            (Scalar::Int(x), Scalar::Int(y)) => Some(x.cmp(y)),
            (Scalar::Float(x), Scalar::Float(y)) => x.partial_cmp(y),
            (Scalar::Int(x), Scalar::Float(y)) => (*x as f64).partial_cmp(y),
            (Scalar::Float(x), Scalar::Int(y)) => x.partial_cmp(&(*y as f64)),
            (Scalar::Str(x), Scalar::Str(y)) => Some(x.cmp(y)),
            (Scalar::Bool(x), Scalar::Bool(y)) => Some(x.cmp(y)),
            _ => return Err(format!("cannot order {a} and {b}")),
        },
    };
    // A NaN is unordered: no ordering comparison holds.
    Ok(ordering.is_some_and(|o| match op {
        Comparison::Lt => o == Ordering::Less,
        Comparison::Le => o != Ordering::Greater,
        Comparison::Gt => o == Ordering::Greater,
        _ => o != Ordering::Less,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::interp::NoScope;
    use std::sync::Arc;

    fn tuple(bindings: &[(&str, Value)]) -> RuntimeTuple {
        bindings
            .iter()
            .map(|(n, v)| ((*n).to_string(), v.clone()))
            .collect()
    }

    fn keeps(predicate: &str, t: &RuntimeTuple) -> bool {
        CompiledPredicate::new(predicate)
            .keeps(t, &NoScope::new())
            .unwrap()
    }

    /// `!` binds tighter than `&&` and `||`: `!{done} || {retry}` keeps
    /// a tuple that is not done, and one that is done and retried.
    #[test]
    fn not_binds_tighter_than_or() {
        for (done, retry, kept) in [
            (false, false, true),
            (false, true, true),
            (true, false, false),
            (true, true, true),
        ] {
            let t = tuple(&[("done", Value::Bool(done)), ("retry", Value::Bool(retry))]);
            assert_eq!(keeps("!{done} || {retry}", &t), kept, "{done} {retry}");
            assert_eq!(keeps("!({done} || {retry})", &t), !done && !retry);
            assert_eq!(keeps("!{done} && {retry}", &t), !done && retry);
        }
    }

    /// Each operator pair evaluates as the one precedence table groups
    /// it, checked against the grouping written out, over every
    /// assignment of three booleans and small integers.
    #[test]
    fn every_operator_pair_evaluates_as_grouped() {
        let pairs = [
            ("{a} || {b} && {c}", "{a} || ({b} && {c})"),
            ("{a} && {b} || {c}", "({a} && {b}) || {c}"),
            ("!{a} || {b}", "(!{a}) || {b}"),
            ("!{a} && {b}", "(!{a}) && {b}"),
            ("{a} == {b} || {c}", "({a} == {b}) || {c}"),
            ("{a} || {b} == {c}", "{a} || ({b} == {c})"),
            ("{a} != {b} && {c}", "({a} != {b}) && {c}"),
            (
                "{x} < 2 || {y} >= 1 && {a}",
                "({x} < 2) || (({y} >= 1) && {a})",
            ),
            ("{x} + 1 > {y} + {y}", "({x} + 1) > ({y} + {y})"),
            (
                "{x} + {x} == {y} + 1 || {c}",
                "(({x} + {x}) == ({y} + 1)) || {c}",
            ),
            ("{x} < {y} == {a}", "({x} < {y}) == {a}"),
            ("{x} in [0, 2] || {a}", "({x} in [0, 2]) || {a}"),
            ("!{a} == {b}", "(!{a}) == {b}"),
        ];
        for bits in 0..8u8 {
            for x in 0..3u64 {
                for y in 0..3u64 {
                    let t = tuple(&[
                        ("a", Value::Bool(bits & 1 != 0)),
                        ("b", Value::Bool(bits & 2 != 0)),
                        ("c", Value::Bool(bits & 4 != 0)),
                        ("x", Value::U64(x)),
                        ("y", Value::U64(y)),
                    ]);
                    for (bare, grouped) in pairs {
                        assert_eq!(keeps(bare, &t), keeps(grouped, &t), "{bare} at {t:?}");
                    }
                }
            }
        }
    }

    /// Totality is read off the tree against the elements' kinds: what
    /// can fail is an ordering of unlike kinds, the truth of a string,
    /// arithmetic over a string or a negative literal, division by
    /// anything but a non-zero constant, an element of unknown kind, and
    /// any call or cast. (The integration suite evaluates each total
    /// predicate here over values of its kinds, with the node library
    /// that arithmetic calls into.)
    #[test]
    fn totality_is_read_off_the_tree() {
        let kind_of = |name: &str| match name {
            "k" | "m" => Some(ValueKind::Int),
            "x" => Some(ValueKind::Float),
            "w" => Some(ValueKind::Str),
            "b" => Some(ValueKind::Bool),
            _ => None,
        };
        let total = [
            "{k} > 1",
            "{k} < {x}",
            "{w} == 2",
            "{w} != {k} && {b}",
            "{w} >= \"m\" || !{b}",
            "{k} in [1, \"a\", true]",
            "{k} * 2 + 1 > {m}",
            "{k} - {m} >= 0",
            "{k} / 2 == 1",
            "{x} % 1.5 < 1",
            "{k} ** 2 > {x}",
            "{k} + 1",
            "{x}",
        ];
        let partial = [
            "{w} > 2",
            "{b} < 1",
            "{w}",
            "{w} && {b}",
            "{k} / {m} == 1",
            "{k} % 0 == 1",
            "{w} + 1 > 2",
            "{k} + -1 > 2",
            "{z} > 1",
            "u64_add({k}, 1) > 2",
            "{x} as u64 > 2",
            "{k} & 1 == 1",
        ];
        for p in total {
            assert!(CompiledPredicate::new(p).is_total(&kind_of), "{p}");
        }
        for p in partial {
            assert!(!CompiledPredicate::new(p).is_total(&kind_of), "{p}");
        }
    }

    /// Values of different kinds are unequal, and ordering them is an
    /// error naming both.
    #[test]
    fn mixed_kinds_are_unequal_and_unordered() {
        let t = tuple(&[("c", Value::Str(Arc::from("s0")))]);
        assert!(keeps("{c} != 2", &t));
        assert!(!keeps("{c} == 2", &t));
        assert!(keeps("{c} == s0", &t));
        // An operand after the one that decides is not evaluated.
        assert!(!keeps("{c} == 2 && {c} > 2", &t));
        assert!(keeps("{c} != 2 || {c} > 2", &t));
        let error = CompiledPredicate::new("{c} > 2")
            .keeps(&t, &NoScope::new())
            .unwrap_err();
        assert!(
            error.to_string().contains("cannot order \"s0\" and 2"),
            "{error}"
        );
    }

    /// A function call evaluates as a Polydat expression over the tuple.
    #[test]
    fn a_call_evaluates_through_the_kernel() {
        let t = tuple(&[("a", Value::U64(3)), ("b", Value::U64(5))]);
        assert!(keeps("u64_add({a}, {b}) > 7", &t));
        assert!(!keeps("u64_add({a}, {b}) > 8", &t));
        assert!(keeps("!(u64_add({a}, {b}) > 8)", &t));
    }

    #[test]
    fn an_unbound_name_is_an_error() {
        let error = CompiledPredicate::new("{z} > 1")
            .keeps(&tuple(&[]), &NoScope::new())
            .unwrap_err();
        assert!(error.to_string().contains("`{z}` is not bound"), "{error}");
    }
}
