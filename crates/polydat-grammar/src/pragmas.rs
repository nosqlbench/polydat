// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Pragmas for Polydat source.
//!
//! Pragmas are first-class Polydat statements that opt a scope into
//! compile-time checks (polydat_grammar.md §14). They cover the
//! strict-wire modes that complement the const-constraint metadata
//! (graph_compiler.md §2):
//!
//! ```polydat
//! pragma strict_values
//! pragma strict_types
//! pragma strict          // both, and strict name checking
//!
//! id := mod(hash(cycle), 1000)
//! ```
//!
//! `pragma` is a reserved keyword in the Polydat grammar; pragmas are
//! [`Statement::Pragma`] in the AST and walked by the compiler the
//! same way other statements are. They are a distinct syntactic
//! construct, not comments.
//!
//! [`Statement::Pragma`]: crate::ast::Statement::Pragma
//!
//! ## Recognised pragma names
//!
//! - `strict_values` — check every wire into a port that declares a
//!   value constraint: a compile-time constant source is checked at
//!   build, and any other source gets a value assertion node.
//! - `strict_types` — accepted and acknowledged, with no effect on the
//!   graph. Wires are statically typed and a resolved wire's type is
//!   the sink port's type, so a runtime type assertion has nothing to
//!   catch.
//! - `strict` — a pragma of its own: it implies `strict_types` and
//!   `strict_values`, and it turns on strict name checking, under which
//!   a comprehension that reads a name nothing binds is refused
//!   (comprehension_forms.md §5 V3). Outside it such a name compiles
//!   with a warning and reads None.
//!
//! Unknown pragmas are recorded and warned about, not errored:
//! pragmas are forward-compatible by design so old binaries can
//! parse modules that opt into newer features they don't
//! support.
//!
//! ## Scoping
//!
//! Scoping is lexical. A pragma applies to the scope it is written in
//! and to every scope nested in it: a program, a `for` body, and a
//! module body are scopes. A `for` body is nested in the scope it is
//! written in and compiles under [`PragmaSet::nested`], its enclosing
//! set plus the pragmas its own statements declare. A module body is
//! nested in no host and compiles under its own pragmas alone. A tile
//! is not a scope: it compiles under the set of the scope it is
//! written in. Pragmas are presence-only, so a nested scope can add to
//! the set and never conflicts with it.

use crate::ast::Statement;

/// One pragma entry parsed from the source.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Pragma {
    /// Bare pragma name (e.g. `"strict_values"`).
    pub name: String,
    /// Whitespace-separated arguments after the name, if any.
    pub args: Vec<String>,
    /// 1-based line number where the pragma appeared, for diagnostics.
    pub line: usize,
}

/// The pragmas in force in one Polydat scope: those the scope declares
/// and those of every scope enclosing it.
#[derive(Debug, Clone, Default)]
pub struct PragmaSet {
    /// The pragmas in force, the enclosing scopes' first, in order.
    pub entries: Vec<Pragma>,
}

impl PragmaSet {
    /// Returns true if the named pragma is in force.
    pub fn contains(&self, name: &str) -> bool {
        self.entries.iter().any(|p| p.name == name)
    }

    /// Returns true if either `strict_types` or `strict`, which implies
    /// it, is in force.
    pub fn strict_types(&self) -> bool {
        self.contains("strict_types") || self.contains("strict")
    }

    /// Returns true if either `strict_values` or `strict`, which implies
    /// it, is in force.
    pub fn strict_values(&self) -> bool {
        self.contains("strict_values") || self.contains("strict")
    }

    /// Returns true if strict name checking is in force: `strict`, and no
    /// other pragma. A comprehension read in this scope that reads a
    /// name nothing binds is then refused (comprehension_forms.md §5 V3).
    pub fn strict_names(&self) -> bool {
        self.contains("strict")
    }

    /// Iterate the pragmas in force that the compiler doesn't
    /// recognise.
    pub fn unknown(&self) -> impl Iterator<Item = &Pragma> {
        self.entries.iter().filter(|p| !is_known(&p.name))
    }

    /// The set a scope nested in this one compiles under: every pragma
    /// in force here, then the pragmas `statements` (the nested
    /// scope's own) declare. Pragmas in scopes nested inside
    /// `statements` are not collected; each applies when its own scope
    /// compiles.
    pub fn nested(&self, statements: &[Statement]) -> PragmaSet {
        let mut entries = self.entries.clone();
        entries.extend(declared_in(statements));
        PragmaSet { entries }
    }
}

/// Recognised pragma names. Add new names here as features land.
pub fn is_known(name: &str) -> bool {
    matches!(name, "strict_types" | "strict_values" | "strict")
}

/// The pragmas `statements` declare at their own level.
pub fn declared_in(statements: &[Statement]) -> impl Iterator<Item = Pragma> + '_ {
    statements.iter().filter_map(|stmt| match stmt {
        Statement::Pragma { name, span } => Some(Pragma {
            name: name.clone(),
            args: Vec::new(),
            line: span.line,
        }),
        _ => None,
    })
}

/// Walk a parsed program and collect the pragmas it declares at its
/// top level into a [`PragmaSet`]: the set the program's own bindings
/// compile under. Pragmas inside a `for` body or a module body belong
/// to that scope.
pub fn collect_from_ast(file: &crate::ast::PolydatFile) -> PragmaSet {
    PragmaSet::default().nested(&file.statements)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::lex;
    use crate::parser::parse;

    fn pragmas_from(src: &str) -> PragmaSet {
        let tokens = lex(src).expect("lex");
        let ast = parse(tokens).expect("parse");
        collect_from_ast(&ast)
    }

    fn statements(src: &str) -> Vec<Statement> {
        parse(lex(src).expect("lex")).expect("parse").statements
    }

    #[test]
    fn strict_implies_both_modes_and_strict_names() {
        let set = pragmas_from("pragma strict\nid := cycle\n");
        assert!(set.strict_types());
        assert!(set.strict_values());
        assert!(set.strict_names());
    }

    /// Only `strict` checks names strictly: the two modes together do not.
    #[test]
    fn parse_individual_modes() {
        let set = pragmas_from("pragma strict_types\npragma strict_values\nid := cycle\n");
        assert!(set.strict_types());
        assert!(set.strict_values());
        assert!(!set.strict_names());
    }

    #[test]
    fn unknown_pragmas_are_collected() {
        let set = pragmas_from("pragma warp_drive\npragma strict\nid := cycle\n");
        assert!(set.strict_types());
        let unknown: Vec<_> = set.unknown().collect();
        assert_eq!(unknown.len(), 1);
        assert_eq!(unknown[0].name, "warp_drive");
    }

    #[test]
    fn nested_scope_inherits_the_enclosing_set() {
        let outer = pragmas_from("pragma strict_values\nid := cycle\n");
        let inner = outer.nested(&statements("x := cycle\n"));
        assert!(inner.strict_values());
    }

    #[test]
    fn nested_scope_adds_without_changing_the_enclosing_set() {
        let outer = pragmas_from("pragma strict_types\nid := cycle\n");
        let inner = outer.nested(&statements("pragma strict_values\nx := cycle\n"));
        assert!(inner.strict_types() && inner.strict_values());
        assert!(!outer.strict_values());
    }
}
