// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! The clause form → algebra AST converter.
//!
//! Reuses the crate-internal text parser
//! for structural shape
//! recognition, then converts the
//! flat-struct AST to the new algebra-layer operator-tree
//! [`crate::comprehension::ast::Comprehension`].
//!
//! Source-string typing is handled by
//! [`super::source_parser::parse_source`] — the clauses AST
//! carries source expressions as raw strings; the algebra
//! layer requires typed [`crate::comprehension::source::Source`]
//! values at AST construction time so the validator and
//! metadata propagator can do their work statically.
//!
//! This converter is the "single bridge" the audit calls for:
//! every clauses AST funnels through here on the way to the
//! algebra layer. nb-workload's parser remains responsible for
//! turning YAML / text into clauses ASTs; polydat owns the
//! conversion onward.

use crate::comprehension::ast::Comprehension as AlgebraAst;
use crate::comprehension::clause_ast::{
    Clause as ClauseForm, ClauseSource as ClauseSourceForm, Comprehension as ClauseAst,
    ComprehensionMode as ModeForm, Subspace as SubspaceForm, TraversalOrder as OrderForm,
    ZipMode as ZipModeForm,
};
use crate::comprehension::strategy::{StrategyName, ZipMode as AlgebraZipMode};

use super::source_parser::{SourceParseError, parse_source};

/// Errors produced when converting a clauses AST to algebra.
#[derive(Debug, Clone, PartialEq)]
pub enum ConvertError {
    /// A clause's source string didn't parse to a typed `Source`.
    SourceParse {
        /// The clause's element name.
        clause_var: String,
        /// The source text.
        source: String,
        /// Why it did not parse.
        cause: SourceParseError,
    },
    /// An empty cartesian or empty union mode.
    EmptyComprehension,
    /// A union sub-space was empty.
    EmptyUnionSubspace,
    /// A parallel clause's vars and exprs had mismatched lengths
    /// (should be caught by the parser, but defensive here).
    ParallelArityMismatch {
        /// Names bound.
        vars: usize,
        /// Expressions given.
        exprs: usize,
    },
}

impl std::fmt::Display for ConvertError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConvertError::SourceParse {
                clause_var,
                source,
                cause,
            } => write!(
                f,
                "clause {clause_var:?} source {source:?} failed to parse: {cause}"
            ),
            ConvertError::EmptyComprehension => f.write_str("comprehension has no clauses"),
            ConvertError::EmptyUnionSubspace => f.write_str("union has empty sub-space"),
            ConvertError::ParallelArityMismatch { vars, exprs } => {
                write!(f, "parallel clause vars={vars} != exprs={exprs}")
            }
        }
    }
}

impl std::error::Error for ConvertError {}

/// Convert the flat parse form to the algebra-layer
/// [`AlgebraAst`].
///
/// Handles:
/// - `mode` → cartesian / union
/// - `filter` → wrapping `Filter` node
/// - `order` → wrapping `Order` node
/// - `Clause::Single` source → typed `Source` via
///   [`parse_source`]
/// - `Clause::Parallel` source → algebra `Zip` of single-var
///   clauses (the algebra layer represents parallel iteration
///   as zip; the clauses parallel-clause shape is an inline
///   form of the same thing)
// The bridge runs one way. The algebra is what everything
// downstream reads: the evaluator consumes it directly and the
// fixtures walk it natively. This is the forward direction,
// `clauses_to_algebra`, which `ComprehensionSpec::into_algebra`
// calls to turn parser output into algebra shape.
pub fn clauses_to_algebra(clauses: &ClauseAst) -> Result<AlgebraAst, ConvertError> {
    let body = match &clauses.mode {
        ModeForm::Cartesian(clauses) => convert_cartesian(clauses)?,
        ModeForm::Union(subspaces) => convert_union(subspaces)?,
    };

    let with_filter = if let Some(pred) = &clauses.filter {
        AlgebraAst::filter(body, pred.clone())
    } else {
        body
    };

    let with_order = if let Some(order) = &clauses.order {
        let (strategy, truncation, seed) = convert_order(order)?;
        AlgebraAst::order_seeded(with_filter, strategy, truncation, seed)
    } else {
        with_filter
    };

    Ok(with_order)
}

fn convert_cartesian(clauses: &[ClauseForm]) -> Result<AlgebraAst, ConvertError> {
    if clauses.is_empty() {
        return Err(ConvertError::EmptyComprehension);
    }
    let algebra_children: Vec<AlgebraAst> = clauses
        .iter()
        .map(convert_clause)
        .collect::<Result<_, _>>()?;
    if algebra_children.len() == 1 {
        // Single clause = the clause itself (R0a I2 would
        // eliminate the singleton cartesian anyway; produce
        // the canonical form upfront).
        Ok(algebra_children.into_iter().next().unwrap())
    } else {
        Ok(AlgebraAst::cartesian(algebra_children))
    }
}

fn convert_union(subspaces: &[SubspaceForm]) -> Result<AlgebraAst, ConvertError> {
    if subspaces.is_empty() {
        return Err(ConvertError::EmptyComprehension);
    }
    let algebra_children: Vec<AlgebraAst> = subspaces
        .iter()
        .map(|s| {
            if s.is_empty() {
                Err(ConvertError::EmptyUnionSubspace)
            } else {
                convert_cartesian(&s.clauses)
            }
        })
        .collect::<Result<_, _>>()?;
    if algebra_children.len() == 1 {
        Ok(algebra_children.into_iter().next().unwrap())
    } else {
        Ok(AlgebraAst::union(algebra_children))
    }
}

fn convert_clause(clause: &ClauseForm) -> Result<AlgebraAst, ConvertError> {
    match &clause.source {
        ClauseSourceForm::Single(source_str) => {
            let var = clause
                .single_var()
                .unwrap_or_else(|| clause.first_var())
                .to_string();
            let source = parse_source(source_str).map_err(|cause| ConvertError::SourceParse {
                clause_var: var.clone(),
                source: source_str.clone(),
                cause,
            })?;
            Ok(AlgebraAst::clause(var, source))
        }
        ClauseSourceForm::Parallel { mode, exprs } => {
            if clause.vars.len() != exprs.len() {
                return Err(ConvertError::ParallelArityMismatch {
                    vars: clause.vars.len(),
                    exprs: exprs.len(),
                });
            }
            // Parallel iteration in the clause form is zip in the algebra.
            // Build a single-var clause per (var, expr) pair,
            // wrap in a Zip with the converted mode.
            let mut children = Vec::with_capacity(clause.vars.len());
            for (var, expr) in clause.vars.iter().zip(exprs.iter()) {
                let source = parse_source(expr).map_err(|cause| ConvertError::SourceParse {
                    clause_var: var.clone(),
                    source: expr.clone(),
                    cause,
                })?;
                children.push(AlgebraAst::clause(var.clone(), source));
            }
            let zip_mode = convert_zip_mode(*mode);
            Ok(AlgebraAst::zip(children, zip_mode))
        }
    }
}

fn convert_zip_mode(clauses: ZipModeForm) -> AlgebraZipMode {
    match clauses {
        ZipModeForm::Strict => AlgebraZipMode::Strict,
        ZipModeForm::Truncate => AlgebraZipMode::Truncate,
        ZipModeForm::Cycle => AlgebraZipMode::Cycle,
    }
}

/// Convert a clauses [`OrderForm`] into the algebra's
/// `(StrategyName, Option<u64>)` pair.
///
/// The clauses `Custom { function }` form is rejected — per
/// spec §3.6, custom orderings are no longer supported.
pub(crate) fn convert_order(
    order: &OrderForm,
) -> Result<(StrategyName, Option<u64>, Option<u64>), ConvertError> {
    let triple = match order {
        OrderForm::Lex { count } => (StrategyName::Lex, count.map(|n| n as u64), None),
        OrderForm::ReverseLex { count } => {
            (StrategyName::ReverseLex, count.map(|n| n as u64), None)
        }
        OrderForm::Diagonal { count } => (StrategyName::Diagonal, count.map(|n| n as u64), None),
        OrderForm::Antidiagonal { count } => {
            (StrategyName::Antidiagonal, count.map(|n| n as u64), None)
        }
        OrderForm::Extrema { strata } => (StrategyName::Extrema, strata.map(|n| n as u64), None),
        OrderForm::Shells { depth, .. } => (StrategyName::Shells, depth.map(|n| n as u64), None),
        OrderForm::Halton { count } => (StrategyName::Halton, count.map(|n| n as u64), None),
        OrderForm::Sobol { count } => (StrategyName::Sobol, count.map(|n| n as u64), None),
        OrderForm::Lhs { count, seed } => (StrategyName::Lhs, count.map(|n| n as u64), *seed),
        OrderForm::Shuffle { count, seed } => {
            (StrategyName::Shuffle, count.map(|n| n as u64), *seed)
        }
    };
    Ok(triple)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::comprehension::source::{LiteralValue, Source};

    fn clause(var: &str, source: &str) -> ClauseForm {
        ClauseForm::new(var, source)
    }

    #[test]
    fn cartesian_single_clause_collapses_to_clause() {
        let clauses = ClauseAst {
            mode: ModeForm::Cartesian(vec![clause("k", "1..10")]),
            filter: None,
            order: None,
        };
        let algebra = clauses_to_algebra(&clauses).unwrap();
        match algebra {
            AlgebraAst::Clause { name, source } => {
                assert_eq!(name, "k");
                assert!(matches!(
                    source,
                    Source::IntRange {
                        lo: 1,
                        hi: 10,
                        step: 1
                    }
                ));
            }
            other => panic!("expected Clause, got {other:?}"),
        }
    }

    #[test]
    fn multi_clause_cartesian_becomes_algebra_cartesian() {
        let clauses = ClauseAst {
            mode: ModeForm::Cartesian(vec![
                clause("k", "1..10"),
                clause("limit", "[10, 100, 1000]"),
            ]),
            filter: None,
            order: None,
        };
        let algebra = clauses_to_algebra(&clauses).unwrap();
        match algebra {
            AlgebraAst::Cartesian { children } => {
                assert_eq!(children.len(), 2);
                // First clause: int range
                match &children[0] {
                    AlgebraAst::Clause { name, source } => {
                        assert_eq!(name, "k");
                        assert!(matches!(
                            source,
                            Source::IntRange {
                                lo: 1,
                                hi: 10,
                                step: 1
                            }
                        ));
                    }
                    other => panic!("expected Clause, got {other:?}"),
                }
                // Second clause: literal list
                match &children[1] {
                    AlgebraAst::Clause { name, source } => {
                        assert_eq!(name, "limit");
                        match source {
                            Source::Literal { values } => {
                                assert_eq!(values.len(), 3);
                                assert_eq!(values[0], LiteralValue::Int(10));
                            }
                            other => panic!("expected Literal, got {other:?}"),
                        }
                    }
                    other => panic!("expected Clause, got {other:?}"),
                }
            }
            other => panic!("expected Cartesian, got {other:?}"),
        }
    }

    #[test]
    fn filter_wraps_body() {
        let clauses = ClauseAst {
            mode: ModeForm::Cartesian(vec![clause("k", "1..10")]),
            filter: Some("{k} > 5".to_string()),
            order: None,
        };
        let algebra = clauses_to_algebra(&clauses).unwrap();
        assert!(matches!(algebra, AlgebraAst::Filter { .. }));
    }

    #[test]
    fn order_lex_with_count_round_trips() {
        let clauses = ClauseAst {
            mode: ModeForm::Cartesian(vec![clause("k", "1..10")]),
            filter: None,
            order: Some(OrderForm::Lex { count: Some(5) }),
        };
        let algebra = clauses_to_algebra(&clauses).unwrap();
        match algebra {
            AlgebraAst::Order {
                strategy: StrategyName::Lex,
                truncation: Some(5),
                ..
            } => {}
            other => panic!("expected Order(Lex, Some(5)), got {other:?}"),
        }
    }

    #[test]
    fn order_halton_with_count() {
        let clauses = ClauseAst {
            mode: ModeForm::Cartesian(vec![clause("k", "1..10"), clause("limit", "1..100")]),
            filter: None,
            order: Some(OrderForm::Halton { count: Some(20) }),
        };
        let algebra = clauses_to_algebra(&clauses).unwrap();
        match algebra {
            AlgebraAst::Order {
                strategy: StrategyName::Halton,
                truncation: Some(20),
                ..
            } => {}
            other => panic!("expected Order(Halton, Some(20)), got {other:?}"),
        }
    }

    #[test]
    fn union_of_subspaces() {
        let clauses = ClauseAst {
            mode: ModeForm::Union(vec![
                SubspaceForm::new(vec![clause("k", "10"), clause("limit", "[1, 2, 3]")]),
                SubspaceForm::new(vec![clause("k", "100"), clause("limit", "[10, 20, 30]")]),
            ]),
            filter: None,
            order: None,
        };
        let algebra = clauses_to_algebra(&clauses).unwrap();
        match algebra {
            AlgebraAst::Union { children } => assert_eq!(children.len(), 2),
            other => panic!("expected Union, got {other:?}"),
        }
    }

    #[test]
    fn parallel_clause_becomes_zip() {
        let parallel = ClauseForm::parallel(["x", "y"], ["1..3", "10..30"]);
        let clauses = ClauseAst {
            mode: ModeForm::Cartesian(vec![parallel]),
            filter: None,
            order: None,
        };
        let algebra = clauses_to_algebra(&clauses).unwrap();
        // After the singleton-cartesian elide, the Zip
        // surfaces at the top level.
        match algebra {
            AlgebraAst::Zip {
                children,
                mode: AlgebraZipMode::Strict,
            } => {
                assert_eq!(children.len(), 2);
            }
            other => panic!("expected Zip, got {other:?}"),
        }
    }

    #[test]
    fn unparseable_source_falls_back_to_generator() {
        // parse_source now treats unrecognized text as a
        // Generator expression (runtime evaluates). So
        // "totally nonsense" round-trips through algebra as
        // a Source::Generator. No conversion error.
        let clauses = ClauseAst {
            mode: ModeForm::Cartesian(vec![clause("k", "totally nonsense")]),
            filter: None,
            order: None,
        };
        let algebra = clauses_to_algebra(&clauses).unwrap();
        match algebra {
            AlgebraAst::Clause { source, .. } => match source {
                crate::comprehension::source::Source::Generator { expr, .. } => {
                    assert_eq!(expr, "totally nonsense");
                }
                other => panic!("expected Generator, got {other:?}"),
            },
            other => panic!("expected Clause, got {other:?}"),
        }
    }

    // (algebra → clauses back-converter tests retired with the
    // bridge in 9c-4b phase 2. The forward direction
    // (`clauses_to_algebra`) tests above remain.)

    /// The authored seed of a seeded order reaches the algebra; the
    /// other strategies lower without one.
    #[test]
    fn a_seeded_order_keeps_its_seed_in_the_algebra() {
        let clauses = ClauseAst {
            mode: ModeForm::Cartesian(vec![clause("k", "1..10")]),
            filter: None,
            order: Some(OrderForm::Shuffle {
                count: Some(3),
                seed: Some(42),
            }),
        };
        let algebra = clauses_to_algebra(&clauses).unwrap();
        assert!(
            matches!(
                algebra,
                AlgebraAst::Order {
                    strategy: StrategyName::Shuffle,
                    truncation: Some(3),
                    seed: Some(42),
                    ..
                }
            ),
            "{algebra:?}"
        );
        assert_eq!(
            convert_order(&OrderForm::Lhs {
                count: None,
                seed: Some(7)
            })
            .unwrap(),
            (StrategyName::Lhs, None, Some(7))
        );
        assert_eq!(
            convert_order(&OrderForm::Halton { count: Some(4) }).unwrap(),
            (StrategyName::Halton, Some(4), None)
        );
    }
}
