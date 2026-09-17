// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Comprehensions — the formal model of iteration shape in GK.
//!
//! ## What it is
//!
//! A *comprehension* is a structured description of the
//! iteration position a scope occupies — the variables it
//! binds, where their value lists come from, and how those
//! lists combine. The algebra of six constructors
//! (`clause`, `cartesian`, `zip`, `union`, `filter`, `order`),
//! closed under composition, is the canonical representation;
//! see `polydat/docs/design/comprehension_forms.md` for the
//! full spec.
//!
//! It's the static-shape counterpart to the run-time
//! [`crate::kernel::ScopeCoord`]: the comprehension says
//! "this scope binds `k` and `limit`, drawn from `{k_values}`
//! and `{k_{k}_limits}`"; the scope coordinate says "right
//! now `k=10` and `limit=20`."
//!
//! ## Module layout
//!
//! The algebra modules ([`ast`], [`source`], [`strategy`],
//! [`spec`], [`runtime`], [`surfaces`], [`ir`], [`optimize`],
//! [`predicate`], [`metadata`], [`validate`](fn@validate), [`cardinality`],
//! [`strategies`]) are the canonical comprehension layer.
//! Top-level re-exports surface the common types
//! ([`Comprehension`], [`Source`], [`ZipMode`], etc.) for
//! ergonomic consumer access.
//!
//! [`ast_legacy`] and [`parse`] retain the older flat-struct
//! comprehension types as parse-pipeline implementation
//! details: the YAML loader uses [`parse::parse_clause_list`]
//! etc. to lex the textual form, then
//! [`spec::ComprehensionSpec::into_algebra`] converts to the
//! canonical algebra AST via [`spec::legacy_to_algebra`].
//! [`eval`] is the runtime-evaluation helper module
//! ([`eval::evaluate_spec`], [`eval::pre_evaluate_clause`]) that
//! both the scope-walker and the runtime evaluator consume.
//!
//! ## Why Polydat owns it
//!
//! Comprehensions cut across three subsystems:
//!
//! - The **YAML parser** (in the host) needs to recognise
//!   the textual shapes (`for_each`, `for_combinations`,
//!   `for_each_union`).
//! - The **scope synthesiser** (in the host)
//!   needs to emit the Polydat source for each comprehension's child
//!   kernel — extern declarations for the coordinates, final
//!   injections for workload params the spec interpolates, etc.
//! - The **executor** (in the host) needs to
//!   enumerate the iteration tuples, drive the per-iteration
//!   `PolydatKernel::for_iteration` (which wires the parent scope
//!   internally), and run the children.
//!
//! All three flow through this module's canonical algebra AST.

// --- The sub-language: its text form, the flat form text parses to,
// the canonical algebra with its sources, strategies, cardinalities,
// and metadata, and the spec forms between them. These live in
// `polydat_grammar`, reachable here at the paths they always had.
pub use polydat_grammar::comprehension::{
    ast, ast_legacy, cardinality, metadata, parse, source, spec, strategy,
};

// --- The runtime's reading of the algebra.
pub mod eval_source;
pub mod flatten;
pub mod ir;
pub mod measure;
pub mod optimize;
pub mod predicate;
pub mod runtime;
pub mod source_values;
pub mod strategies;
pub mod surfaces;
pub mod validate;

// --- Parse-pipeline support modules. `ast_legacy` and `parse`
// produce the older flat-struct form that the YAML parser
// generates; `spec::ComprehensionSpec::into_algebra` converts
// that into the canonical algebra AST above via
// `spec::legacy_to_algebra`. `eval` is the runtime-evaluation
// helper used by both the algebra runtime evaluator and the
// scope-walker.
pub mod eval;
pub mod streamer_value;
pub use streamer_value::StreamerValue;

// --- Canonical algebra re-exports — `polydat::iteration::comprehension::Comprehension`
// resolves to the algebra type; same for Source, ZipMode, etc.
pub use ast::Comprehension;
pub use cardinality::{CardinalityClass, Hybrid, Interval, MeasureName, ProductMeasure};
pub use eval_source::{EvalClass, EvalContext, EvalError, EvaluatedSource, SourceEval};
pub use metadata::{IndexFn, Materialization, Metadata, NaturalOrder};
pub use source::Source;
pub use strategy::{StrategyName, ZipMode};
pub use validate::{Mode, ValidationError, ValidationReport, ValidationWarning, validate};

// --- Parse-pipeline support re-exports. These are evaluator
// utilities used by the algebra runtime evaluator and the
// scope-walker — not part of the comprehension AST surface.
//
// `parse_list_with_types` is not re-exported at this level; it
// remains reachable as `eval::parse_list_with_types`.
pub use eval::{evaluate_spec, pre_evaluate_clause, value_to_polydat_type_name};
