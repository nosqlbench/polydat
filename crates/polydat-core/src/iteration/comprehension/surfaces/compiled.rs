// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! `CompiledComprehension` — the entry point for the three
//! consumption surfaces.
//!
//! Holds an `Arc<Program>` (immutable IR per spec §9.1). Each
//! factory method on this handle (`coordinate_stream`,
//! `scoped_kernel_stream`, `scope_once`) returns a fresh
//! streamer with its own dispense state but shares the
//! `Arc<Program>` — no recompilation across siblings (spec
//! §9.5.2's "IR-sharing test" property).

use std::sync::Arc;

use crate::iteration::comprehension::ast::Comprehension;
use crate::iteration::comprehension::ir::{Program, compile as compile_to_ir};
use crate::iteration::comprehension::optimize::optimize;
use crate::iteration::comprehension::validate::{
    Mode, ValidationError, ValidationReport, validate,
};

use super::coord_stream::CoordinateStream;
use super::instance::{KernelScope, ScopedKernelInstance};
use super::scope_once::scope_once_with;
use super::scoped_stream::ScopedKernelStream;

/// A comprehension that has been compiled to immutable IR
/// and is ready to dispense. The single source of truth for
/// the underlying program across the consumption surfaces.
///
/// Construction is via [`from_ast`](Self::from_ast) (compiles once) or
/// [`from_program`](Self::from_program) (when the IR was compiled elsewhere).
/// Cloning a `CompiledComprehension` is cheap — just an
/// `Arc::clone` on the program.
#[derive(Debug, Clone)]
pub struct CompiledComprehension {
    program: Arc<Program>,
}

impl CompiledComprehension {
    /// Compile an AST: validation (§5, V1–V9) first, then the §10
    /// optimizer, then the AST → IR pass, once. Validation and
    /// optimization are stages of this compile
    /// (comprehension_forms.md §5, §9.4, §10.6): a tree that
    /// violates a V-axiom is refused here, and the §9.3 resource
    /// bounds hold for the program this returns.
    pub fn from_ast(ast: &Comprehension) -> Result<Self, ValidationError> {
        Self::from_ast_with(ast, Mode::Permissive).map(|(compiled, _)| compiled)
    }

    /// [`from_ast`](Self::from_ast) under a validation mode
    /// (comprehension_forms.md §5.8), with the validator's report: in
    /// `Permissive` mode a degenerate composition is a warning in the
    /// report and the comprehension compiles; in `Strict` mode it is
    /// the error.
    pub fn from_ast_with(
        ast: &Comprehension,
        mode: Mode,
    ) -> Result<(Self, ValidationReport), ValidationError> {
        let report = validate(ast, mode)?;
        Ok((
            Self {
                program: Arc::new(compile_to_ir(&optimize(ast.clone()))),
            },
            report,
        ))
    }

    /// Wrap an already-compiled program (tests use this for
    /// hand-built IR).
    pub fn from_program(program: Arc<Program>) -> Self {
        Self { program }
    }

    /// Access the underlying compiled program (immutable per
    /// spec §9.1).
    pub fn program(&self) -> &Program {
        &self.program
    }

    /// Clone the `Arc<Program>` for sharing with other
    /// handles. Used internally by the streamer factories.
    pub(crate) fn program_arc(&self) -> Arc<Program> {
        Arc::clone(&self.program)
    }

    /// **First-order surface** (spec §9.5).
    ///
    /// Return a fresh [`CoordinateStream`]. Each call
    /// allocates new per-streamer state; siblings share the
    /// underlying IR but dispense independently per spec
    /// §9.5.2's independence contract.
    pub fn coordinate_stream(&self) -> CoordinateStream {
        CoordinateStream::new(self.program_arc())
    }

    /// **Second-order surface** (spec §9.5).
    ///
    /// Return a fresh [`ScopedKernelStream`] wrapping the
    /// supplied parent kernel. Each `advance()` pulls one
    /// coord tuple from the underlying IR and applies
    /// `parent.scope(&coords)` to produce a
    /// [`ScopedKernelInstance`].
    ///
    /// Independence: pulling from this stream does NOT
    /// advance any [`CoordinateStream`] obtained from the
    /// same `CompiledComprehension`.
    pub fn scoped_kernel_stream<K: KernelScope>(&self, parent: K) -> ScopedKernelStream<K> {
        ScopedKernelStream::new(self.program_arc(), parent)
    }

    /// **One-shot surface** (spec §9.5.3).
    ///
    /// Apply `parent.scope(coords)` directly, without
    /// constructing any streamer. Pure function — no
    /// cursor consulted, no dispense state advanced. Used
    /// for replay, debugging, and point queries where a
    /// specific coord tuple is already known.
    pub fn scope_once<K: KernelScope>(
        &self,
        parent: &K,
        coords: &crate::iteration::comprehension::strategies::Tuple,
    ) -> ScopedKernelInstance<K::Scoped> {
        scope_once_with(parent, coords)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::iteration::comprehension::source::{LiteralValue, Source};

    fn clause(name: &str, vs: &[i64]) -> Comprehension {
        Comprehension::clause(
            name,
            Source::Literal {
                values: vs.iter().map(|n| LiteralValue::Int(*n)).collect(),
            },
        )
    }

    #[test]
    fn from_ast_compiles_once() {
        let ast = clause("k", &[1, 2, 3]);
        let compiled = CompiledComprehension::from_ast(&ast).unwrap();
        assert!(!compiled.program().is_empty());
    }

    /// Optimization is mandatory before IR compilation
    /// (comprehension_forms.md §9.4, §10.6): `from_ast` compiles the
    /// optimized tree, so a program it returns is the program of the
    /// optimized AST, and where a rule fires it is not the program of
    /// the raw one.
    #[test]
    fn from_ast_compiles_the_optimized_tree() {
        // A nested cartesian: R0b (A2) flattens it, so the raw and
        // optimized trees compile to different programs.
        let inner = Comprehension::cartesian(vec![clause("a", &[1, 2]), clause("b", &[3])]);
        let ast = Comprehension::cartesian(vec![inner, clause("c", &[4])]);
        let compiled = CompiledComprehension::from_ast(&ast).unwrap();
        assert_eq!(*compiled.program(), compile_to_ir(&optimize(ast.clone())));
        assert_ne!(*compiled.program(), compile_to_ir(&ast));
    }

    /// Validation is a stage of the compile (comprehension_forms.md
    /// §5): a tree that violates a V-axiom is refused by `from_ast`
    /// with the axiom's error, not compiled.
    #[test]
    fn from_ast_refuses_a_tree_that_violates_a_v_axiom() {
        // V1: a cartesian whose children bind the same name.
        let ast = Comprehension::cartesian(vec![clause("k", &[1, 2]), clause("k", &[3, 4])]);
        let err = CompiledComprehension::from_ast(&ast).unwrap_err();
        assert!(
            matches!(err, ValidationError::V1DuplicateName { ref name, .. } if name == "k"),
            "{err}"
        );
        assert!(err.to_string().starts_with("V1:"), "{err}");
    }

    /// Validation modes (comprehension_forms.md §5.8): a degenerate
    /// composition is a warning in the permissive report and the error
    /// of a strict compile.
    #[test]
    fn from_ast_with_reports_or_refuses_a_degenerate_composition() {
        use crate::iteration::comprehension::strategy::StrategyName;
        use crate::iteration::comprehension::validate::ValidationWarning;
        let ast = Comprehension::order(clause("k", &[1, 2, 3]), StrategyName::Extrema, Some(1));
        let (_, report) = CompiledComprehension::from_ast_with(&ast, Mode::Permissive).unwrap();
        assert!(matches!(
            report.warnings.as_slice(),
            [ValidationWarning::DegenerateGeometric { .. }]
        ));
        let err = CompiledComprehension::from_ast_with(&ast, Mode::Strict).unwrap_err();
        assert!(matches!(err, ValidationError::StrictWarning(_)), "{err}");
        assert!(err.to_string().starts_with("strict mode:"), "{err}");
    }

    #[test]
    fn cloning_compiled_shares_arc() {
        let ast = clause("k", &[1, 2, 3]);
        let a = CompiledComprehension::from_ast(&ast).unwrap();
        let b = a.clone();
        // Same Arc — strong_count goes up.
        let count = Arc::strong_count(&a.program);
        assert!(count >= 2, "expected shared Arc, count = {count}");
        drop(b);
    }

    #[test]
    fn two_coordinate_streams_share_program() {
        let ast = clause("k", &[1, 2, 3]);
        let compiled = CompiledComprehension::from_ast(&ast).unwrap();
        let _s1 = compiled.coordinate_stream();
        let _s2 = compiled.coordinate_stream();
        // Both streams hold an Arc; count is at least 3 (compiled +
        // two streamers, possibly more if internal clones happen).
        let count = Arc::strong_count(&compiled.program);
        assert!(
            count >= 3,
            "expected shared program across streamers, count = {count}"
        );
    }
}
