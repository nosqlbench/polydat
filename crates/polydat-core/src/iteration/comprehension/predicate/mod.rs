// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Filter predicates — comprehension_forms.md §10.9: their
//! evaluation (§10.9.1) and their analysis.
//!
//! A predicate is the text carried by
//! [`crate::iteration::comprehension::ast::Comprehension::Filter`],
//! parsed by [`parse_predicate`] with the language's one precedence
//! table. [`CompiledPredicate`] evaluates it against a tuple on every
//! path that filters and decides whether it is total. The analyzer
//! produces [`PredicateInfo`] from the same tree for the optimizer's
//! R5 (per-axis filter pushdown).
//!
//! ## Module layout
//!
//! - [`eval`] — `CompiledPredicate`, the evaluator, and its totality
//!   check.
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
pub use eval::{CompiledPredicate, ValueKind, element_kind};
pub use info::{
    Determinism, Factorization, Monotonicity, OpaqueReason, PerAxisMap, PredicateInfo,
    RangeConstraint,
};
pub use polydat_grammar::comprehension::predicate::{
    Comparison, Predicate, PredicateKind, PredicateLiteral, parse_predicate,
};
