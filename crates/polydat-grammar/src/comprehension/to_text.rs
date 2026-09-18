// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! The canonical text of a source and of a comprehension
//! (comprehension_forms.md §8): the rendering that
//! [`spec::parse_source`](crate::comprehension::spec::parse_source) and
//! [`spec::parse_comprehension_algebra`](crate::comprehension::spec::parse_comprehension_algebra)
//! read back to an equal source and an equal tree, so a comprehension
//! built programmatically carries the text a written one carries.
//!
//! Both renderings answer `None` rather than a spelling the parser
//! would read back as something else. A source has no text when a
//! literal holds a value the text cannot write: a JSON value, or a
//! string carrying a quote, a comma, or a bracket. A tree has no text
//! when its shape is outside the text grammar's — a clause list or a
//! bracketed union, each with an optional `where` and `order` — so a
//! filter under a cartesian, an order under a zip, and a nested
//! cartesian have none.

use super::ast::Comprehension;
use super::source::{LiteralValue, Source};
use super::strategy::{StrategyName, ZipMode};

impl Source {
    /// The source's canonical text, what a clause writes after `in`,
    /// or `None` when the text cannot write it.
    pub fn to_text(&self) -> Option<String> {
        Some(match self {
            Source::IntRange { lo, hi, step } => {
                if *step == 1 {
                    format!("{lo}..{hi}")
                } else {
                    format!("{lo}..{hi}..{step}")
                }
            }
            // The bare list the grammar's own examples write, when
            // every value reads back from it; the bracketed, quoted
            // form otherwise, which carries a string a bare list
            // cannot.
            Source::Literal { values } => literal_list_text(values)?,
            Source::Generator { expr, .. } => expr.clone(),
            Source::WorkloadParamList { name, .. } => format!("{{{name}}}"),
            Source::ContinuousInterval { interval, .. } => {
                // `{:?}` (not `{}`) keeps the decimal point so the
                // endpoints re-parse as floats — `{}` on `1.0f64` prints
                // "1", round-tripping a continuous `[1.0,5.0)` to
                // "1..5", which the source parser would re-classify as
                // an integer range (typing the element `u64`).
                if interval.hi_open {
                    format!("{:?}..{:?}", interval.lo, interval.hi)
                } else {
                    format!("{:?}..={:?}", interval.lo, interval.hi)
                }
            }
            Source::Distribution {
                distribution,
                support,
                params,
            } => {
                // The parameters in the measure's own order, and the
                // interval only when it narrows the measure's own
                // support.
                let args = params
                    .iter()
                    .map(|p| format!("{p:?}"))
                    .collect::<Vec<_>>()
                    .join(", ");
                let call = format!("{}({args})", distribution.text());
                if *support == distribution.support(params) {
                    call
                } else if support.hi_open {
                    format!("{call} on {:?}..{:?}", support.lo, support.hi)
                } else {
                    format!("{call} on {:?}..={:?}", support.lo, support.hi)
                }
            }
        })
    }
}

/// A literal list as the text writes it: the bare comma list the
/// grammar's own examples use when every value reads back from it, and
/// the bracketed, quoted form otherwise. `None` when neither reads
/// back: a JSON value, or a string carrying a quote, a comma, or a
/// bracket.
fn literal_list_text(values: &[LiteralValue]) -> Option<String> {
    if values.is_empty() {
        return None;
    }
    // A bare single string is an identifier, which reads back as a
    // generator call, so one string takes the bracketed form; one
    // number or boolean is a scalar of its own.
    let bare_ok = values.iter().all(bare_value_is_unambiguous)
        && (values.len() > 1 || !matches!(values[0], LiteralValue::String(_)));
    if bare_ok {
        let items = values
            .iter()
            .map(bare_value_text)
            .collect::<Option<Vec<_>>>()?;
        return Some(items.join(", "));
    }
    let items = values
        .iter()
        .map(quoted_value_text)
        .collect::<Option<Vec<_>>>()?;
    Some(format!("[{}]", items.join(", ")))
}

/// Whether a value reads back from a bare list: a number or a boolean
/// always does, and a string does when it carries none of the
/// characters the bare form refuses and does not read back as a number
/// or a boolean instead.
fn bare_value_is_unambiguous(v: &LiteralValue) -> bool {
    match v {
        LiteralValue::Int(_) | LiteralValue::Float(_) | LiteralValue::Bool(_) => true,
        LiteralValue::String(s) => {
            let trimmed = s.trim();
            !trimmed.is_empty()
                && trimmed == s
                && !s.contains([
                    '(', ')', '[', ']', '{', '}', '\'', '"', '+', '*', '/', '%', '=', '<', '>',
                    '!', '&', '|', '~', '^', '?', ',', '\n',
                ])
                && s.parse::<i64>().is_err()
                && s.parse::<f64>().is_err()
                && !s.eq_ignore_ascii_case("true")
                && !s.eq_ignore_ascii_case("false")
        }
        LiteralValue::Json(_) => false,
    }
}

/// One value as a bare list writes it.
fn bare_value_text(v: &LiteralValue) -> Option<String> {
    Some(match v {
        LiteralValue::Int(i) => i.to_string(),
        LiteralValue::Float(f) => format!("{f:?}"),
        LiteralValue::Bool(b) => b.to_string(),
        LiteralValue::String(s) => s.clone(),
        LiteralValue::Json(_) => return None,
    })
}

/// One value as a bracketed list writes it, strings quoted. `None`
/// when the text cannot: a JSON value has no literal spelling, and a
/// string carrying a quote, a comma, or a bracket is not read back as
/// one value.
fn quoted_value_text(v: &LiteralValue) -> Option<String> {
    Some(match v {
        LiteralValue::String(s) => {
            if s.contains(['"', '\'', ',', '[', ']', '{', '}', '(', ')', '\n']) {
                return None;
            }
            format!("\"{s}\"")
        }
        other => bare_value_text(other)?,
    })
}

impl Comprehension {
    /// The comprehension's canonical text (comprehension_forms.md §8),
    /// or `None` when the tree's shape, or a source in it, is outside
    /// the text grammar.
    ///
    /// Parsing the text this returns yields a tree equal to this one,
    /// and rendering that tree returns the same text.
    pub fn to_text(&self) -> Option<String> {
        match self {
            Comprehension::Order {
                child,
                strategy,
                truncation,
                seed,
            } => {
                let head = match &**child {
                    Comprehension::Filter { child, predicate } => {
                        format!("{} where {predicate}", child.body_text()?)
                    }
                    other => other.body_text()?,
                };
                Some(format!(
                    "{head} order {}",
                    order_text(*strategy, *truncation, *seed)
                ))
            }
            Comprehension::Filter { child, predicate } => {
                Some(format!("{} where {predicate}", child.body_text()?))
            }
            other => other.body_text(),
        }
    }

    /// The text of a comprehension's body: the clause list or the
    /// bracketed union, without a `where` or an `order`, which only
    /// the outermost level carries.
    fn body_text(&self) -> Option<String> {
        match self {
            Comprehension::Clause { .. } | Comprehension::Zip { .. } => self.clause_text(),
            Comprehension::Cartesian { children } => children
                .iter()
                .map(Comprehension::clause_text)
                .collect::<Option<Vec<_>>>()
                .map(|clauses| clauses.join(", ")),
            Comprehension::Union { children } => {
                let members = children
                    .iter()
                    .map(|c| c.to_text().map(|text| format!("for {text}")))
                    .collect::<Option<Vec<_>>>()?;
                Some(format!("[ {} ]", members.join(", ")))
            }
            // A filter or an order inside a body: the text carries each
            // once, at the end, so there is no place to write it.
            Comprehension::Filter { .. } | Comprehension::Order { .. } => None,
        }
    }

    /// The text of one clause of a clause list: `name in source`, or
    /// the tuple form `(a, b) in (…)` for a zip of single clauses.
    fn clause_text(&self) -> Option<String> {
        match self {
            Comprehension::Clause { name, source } => {
                Some(format!("{name} in {}", source.to_text()?))
            }
            Comprehension::Zip { children, mode } => {
                if children.len() < 2 {
                    return None;
                }
                let mut names = Vec::with_capacity(children.len());
                let mut sources = Vec::with_capacity(children.len());
                for child in children {
                    let Comprehension::Clause { name, source } = child else {
                        return None;
                    };
                    names.push(name.clone());
                    sources.push(source.to_text()?);
                }
                let inner = sources.join(", ");
                let rhs = match mode {
                    ZipMode::Strict => format!("({inner})"),
                    ZipMode::Truncate => format!("zip_truncate({inner})"),
                    ZipMode::Cycle => format!("zip_cycle({inner})"),
                };
                Some(format!("({}) in {rhs}", names.join(", ")))
            }
            _ => None,
        }
    }
}

/// An order's text: the terse `name/n` form, and the keyword form when
/// a seeded strategy carries an authored seed, which is what the order
/// parser reads back.
fn order_text(strategy: StrategyName, truncation: Option<u64>, seed: Option<u64>) -> String {
    let name = strategy.as_str();
    match (truncation, seed) {
        (Some(n), Some(s)) => format!("{name}(count={n}, seed={s})"),
        (None, Some(s)) => format!("{name}(seed={s})"),
        (Some(n), None) => format!("{name}/{n}"),
        (None, None) => name.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::comprehension::cardinality::{Interval, MeasureName, ProductMeasure};
    use crate::comprehension::spec::{parse_comprehension_algebra, parse_source};

    /// Every source the text can write round-trips through its text.
    #[test]
    fn every_source_round_trips_through_its_text() {
        let sources = [
            Source::IntRange {
                lo: 1,
                hi: 10,
                step: 1,
            },
            Source::IntRange {
                lo: 0,
                hi: 20,
                step: 5,
            },
            Source::Literal {
                values: vec![LiteralValue::Int(1), LiteralValue::Int(2)],
            },
            Source::Literal {
                values: vec![LiteralValue::String("load".into())],
            },
            Source::Literal {
                values: vec![
                    LiteralValue::String("load".into()),
                    LiteralValue::String("verify".into()),
                ],
            },
            Source::Literal {
                values: vec![LiteralValue::Float(1.5), LiteralValue::Float(2.0)],
            },
            Source::Literal {
                values: vec![LiteralValue::Bool(true), LiteralValue::Bool(false)],
            },
            Source::Generator {
                expr: "partitions(\"*/4\", 100)".into(),
                cardinality_hint: None,
            },
            Source::WorkloadParamList {
                name: "total".into(),
                len_hint: None,
            },
            Source::ContinuousInterval {
                interval: Interval::half_open(0.0, 1.0),
                measure: ProductMeasure::Uniform,
            },
            Source::Distribution {
                distribution: MeasureName::Normal,
                support: MeasureName::Normal.support(&[0.0, 1.0]),
                params: vec![0.0, 1.0],
            },
            Source::Distribution {
                distribution: MeasureName::Exponential,
                support: Interval::half_open(0.0, 1.0),
                params: vec![1.0],
            },
        ];
        for source in sources {
            let text = source.to_text().expect("the text writes this source");
            let back = parse_source(&text).unwrap_or_else(|e| panic!("`{text}`: {e:?}"));
            assert_eq!(back, source, "`{text}`");
            assert_eq!(back.to_text(), Some(text));
        }
    }

    /// A value the text cannot write has no text, rather than one that
    /// reads back as something else.
    #[test]
    fn a_source_the_text_cannot_write_has_no_text() {
        for value in [
            LiteralValue::Json(serde_json::json!({"a": 1})),
            LiteralValue::String("a,b".into()),
            LiteralValue::String("say \"hi\"".into()),
            LiteralValue::String("[bracketed]".into()),
        ] {
            let source = Source::Literal {
                values: vec![value.clone()],
            };
            assert_eq!(source.to_text(), None, "{value:?}");
        }
    }

    /// Canonical text is what the tree renders to and what parses back
    /// to it: text → tree → text is stable, and tree → text → tree is
    /// the identity.
    #[test]
    fn canonical_text_round_trips_through_the_grammar() {
        let texts = [
            "k in 1..4",
            "k in 1..10..2",
            "k in 1, 2, 4",
            "k in load, verify",
            "x in 0.0..1.0",
            "x in normal(0.0, 1.0)",
            "x in exponential(1.0) on 0.0..1.0",
            "k in 1..4, limit in 10, 20",
            "(a, b) in (1..4, 10..13)",
            "(a, b) in zip_truncate(1..4, 10..20)",
            "(a, b) in zip_cycle(1..4, 10..20)",
            "k in 1..9 where {k} > 2",
            "k in 1..9 order lex",
            "k in 1..9 order halton/3",
            "k in 1..9 order shuffle(count=3, seed=42)",
            "k in 1..9 order lhs(seed=7)",
            "k in 1..9 where {k} > 2 order halton/2",
            "[ for k in 1..4, for k in 10..13 ]",
            "[ for k in 1..4 where {k} > 1, for k in 10..13 order lex/2 ]",
            "[ for k in 1..4, for k in 10..13 ] where {k} > 2 order halton/2",
        ];
        for text in texts {
            let tree =
                parse_comprehension_algebra(text).unwrap_or_else(|e| panic!("`{text}`: {e}"));
            let rendered = tree
                .to_text()
                .unwrap_or_else(|| panic!("`{text}` has no canonical text"));
            assert_eq!(rendered, text, "canonical text is not the written text");
            let back = parse_comprehension_algebra(&rendered)
                .unwrap_or_else(|e| panic!("`{rendered}`: {e}"));
            assert_eq!(back, tree, "`{rendered}` parses to a different tree");
            assert_eq!(back.to_text().as_deref(), Some(rendered.as_str()));
        }
    }

    /// A tree the text grammar cannot spell has no text.
    #[test]
    fn a_tree_outside_the_text_grammar_has_no_text() {
        let clause = |name: &str| {
            Comprehension::clause(
                name,
                Source::IntRange {
                    lo: 1,
                    hi: 4,
                    step: 1,
                },
            )
        };
        // A filter under a cartesian: the text carries one `where`, at
        // the end, over the whole clause list.
        let inner_filter = Comprehension::cartesian(vec![
            Comprehension::filter(clause("k"), "{k} > 1"),
            clause("j"),
        ]);
        assert_eq!(inner_filter.to_text(), None);
        // An order under a zip, a nested cartesian, and a one-child zip.
        let ordered_zip = Comprehension::zip(
            vec![
                Comprehension::order(clause("k"), StrategyName::Lex, Some(2)),
                clause("j"),
            ],
            ZipMode::Strict,
        );
        assert_eq!(ordered_zip.to_text(), None);
        let nested = Comprehension::cartesian(vec![
            Comprehension::cartesian(vec![clause("k"), clause("j")]),
            clause("m"),
        ]);
        assert_eq!(nested.to_text(), None);
        assert_eq!(
            Comprehension::zip(vec![clause("k")], ZipMode::Strict).to_text(),
            None
        );
    }
}
