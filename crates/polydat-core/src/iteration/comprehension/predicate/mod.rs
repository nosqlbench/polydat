// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Filter predicates — spec §10.9: their evaluation and their
//! analysis.
//!
//! A predicate is the text carried by
//! [`crate::iteration::comprehension::ast::Comprehension::Filter`],
//! parsed by [`parse_predicate`] with the language's one precedence
//! table. [`CompiledPredicate`] evaluates it against a tuple on every
//! path that filters. The analyzer produces [`PredicateInfo`] from the
//! same tree for the optimizer's R5 (per-axis filter pushdown) and the
//! deferred R8 / R9 / R10 rules.
//!
//! ## Module layout
//!
//! - [`eval`] — `CompiledPredicate`, the evaluator.
//! - [`info`] — `PredicateInfo` and supporting enums.
//! - [`coordset`] — `CoordSet` carrying per-coord discrete /
//!   continuous classification.
//! - [`recognizers`] — the §10.9.5 pattern catalog.
//! - [`analyzer`] — entry point + dispatch.

pub mod analyzer;
pub mod coordset;
pub mod eval;
pub mod info;
pub mod recognizers;

pub use analyzer::analyze;
pub use coordset::{CoordInfo, CoordKind, CoordSet};
pub use eval::CompiledPredicate;
pub use info::{
    Determinism, Factorization, Monotonicity, OpaqueReason, PerAxisMap, PredicateInfo,
    RangeConstraint,
};
pub use polydat_grammar::comprehension::predicate::{
    Comparison, Predicate, PredicateKind, PredicateLiteral, parse_predicate,
};
