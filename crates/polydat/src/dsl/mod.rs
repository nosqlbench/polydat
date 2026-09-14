// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Polydat DSL: lexer, parser, and AST for `.polydat` kernel definition files.

// The language itself lives in `polydat_grammar`: the lexer, the
// parser, the AST, the pretty-printer, the free-name collector,
// pragmas, the diagnostic types, and the tile template
// parsers. Each is reachable here at the path it always had.
pub use polydat_grammar::{
    ast, error, lexer, parser, pprint, pragmas, refs, tile, tile_structural,
};

pub mod compile;
pub mod const_constraints;
pub mod cursor_sugar;
pub mod events;
pub mod factories;
/// External-facing factory module. Made `pub` (was `pub(crate)`) so
/// crates that host polydat nodes outside the polydat crate
/// (host runtimes) can reach `ConstArg`, `compile_ctx`,
/// and `build_node` from their `register_nodes!` invocations.
pub mod factory;
pub mod registry;
pub mod tile_lower;
pub mod transform;
pub mod traversal;
pub(crate) mod validate;

pub mod stub;

/// Re-exported for external crates that register Polydat nodes via `register_nodes!`.
pub use factory::ConstArg;
mod binding;
mod modules;

pub use compile::{
    CompileOptions, compile_ast_with_engine, compile_ast_with_options, compile_polydat,
    compile_polydat_checked, compile_polydat_kernel, compile_polydat_kernel_with_options,
    compile_polydat_kernel_with_tiles, compile_polydat_with, compile_polydat_with_engine,
    compile_polydat_with_options, eval_const_expr,
};
// The deprecated forms stay reachable at their old paths; a caller sees
// the deprecation at its own use.
#[allow(deprecated)]
pub use compile::{
    compile_polydat_strict, compile_polydat_with_libs, compile_polydat_with_libs_and_limit,
    compile_polydat_with_outputs, compile_polydat_with_path,
};

/// Collect identifier references from an `Expr` tree into `out`.
///
/// Walks every `Ident`, function call argument, binary/unary
/// operand, array element, and field-access source. String-
/// literal placeholders (`{name}` form) contribute their
/// identifier-shaped placeholder bodies.
///
/// Cross-crate consumers (e.g. the host's SRD-13f
/// synthesizer) use this to discover transitive wire refs from
/// a binding's RHS without depending on the private `validate`
/// module.
pub fn collect_expr_references(expr: &ast::Expr, out: &mut std::collections::HashSet<String>) {
    validate::collect_references(expr, out);
}

/// Return the embedded standard library module sources.
///
/// Each entry is `(filename, source_text)` — the same data used by the
/// compiler's module resolver at build time.
pub fn stdlib_sources() -> &'static [(&'static str, &'static str)] {
    compile::stdlib_sources()
}
