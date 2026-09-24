// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Program transforms: host intentions expressed as changes to the
//! program rather than as runtime behavior around it.
//!
//! A host that wants to inject a value, observe a scope, or otherwise
//! shape a run does so by rewriting the program and compiling the
//! result. The program's own typing, lifecycle classification, and
//! engine selection then apply to the host's additions exactly as they
//! apply to the author's.

use super::ast::{Expr, ExternPort, PolydatFile, Statement, TileDef, TilePiece};

/// Assign `name=value` text to externs and inputs by rewriting their
/// declarations.
///
/// - An `extern name: T` gets `"value"` as its default. The compiler
///   fuses the string literal to `T` through the same coercions that
///   apply when a str wire feeds a typed port, so a bad value is a
///   compile error carrying the program's own diagnostic.
/// - An `input name: T` becomes `extern name: T = "value"`: for this
///   run the coordinate is fixed, so it is no longer a coordinate.
/// - A name that is neither is an error listing what the program
///   declares.
///
/// The transform is order preserving and leaves every other statement
/// untouched.
pub fn assign_values(
    file: &mut PolydatFile,
    assignments: &[(String, String)],
) -> Result<(), String> {
    for (name, raw) in assignments {
        let mut found = false;
        // Written over the statement walker, so the transform is the
        // rewrite and the walk is not its business. It assigns at this
        // scope only: a name inside a module or a `for` body belongs to
        // that scope, and the walker would reach it.
        each_statement::<std::convert::Infallible>(&mut file.statements, &mut |stmt| {
            match stmt {
                Statement::ExternPort(port) if &port.name == name => {
                    port.default = Some(Expr::StringLit(raw.clone(), port.span));
                    found = true;
                }
                Statement::InputDecl(decl) if &decl.name == name => {
                    let span = decl.span;
                    let typ = decl.ty.clone().unwrap_or_else(|| "u64".to_string());
                    *stmt = Statement::ExternPort(ExternPort {
                        name: name.clone(),
                        typ,
                        default: Some(Expr::StringLit(raw.clone(), span)),
                        span,
                    });
                    found = true;
                }
                _ => {}
            }
            Ok(())
        })
        .expect("the closure never fails");
        if !found {
            let declared: Vec<&str> = file
                .statements
                .iter()
                .filter_map(|s| match s {
                    Statement::ExternPort(p) => Some(p.name.as_str()),
                    Statement::InputDecl(d) => Some(d.name.as_str()),
                    _ => None,
                })
                .collect();
            return Err(format!(
                "cannot assign '{name}': no extern or input by that name; declared: {}",
                if declared.is_empty() {
                    "(none)".to_string()
                } else {
                    declared.join(", ")
                }
            ));
        }
    }
    Ok(())
}

/// Let the program's `extern name` take a value of any type, converted
/// in front of each reader by a converter node (input_variance.md §6).
///
/// The extern is declared `dyn`; the compiler places one converter per
/// type its readers read, reports each in the compile log, and a value
/// the converter cannot convert fails the converter, attributed as any
/// node's failure. For one input a host knows varies, without opening
/// every input the author left untyped. A name that is not a top-level
/// `extern` is an error: a coordinate is positioned with `set_inputs`
/// and never varies in type.
pub fn convert_input(file: &mut PolydatFile, name: &str) -> Result<(), String> {
    for stmt in &mut file.statements {
        if let Statement::ExternPort(port) = stmt
            && port.name == name
        {
            port.typ = crate::ast::PortType::Dyn.to_keyword().to_string();
            return Ok(());
        }
    }
    let externs: Vec<&str> = file
        .statements
        .iter()
        .filter_map(|s| match s {
            Statement::ExternPort(p) => Some(p.name.as_str()),
            _ => None,
        })
        .collect();
    Err(format!(
        "cannot convert '{name}': no extern by that name; declared externs: {}",
        if externs.is_empty() {
            "(none)".to_string()
        } else {
            externs.join(", ")
        }
    ))
}

/// Split `name=value` text into its parts.
pub fn parse_assignment(text: &str) -> Result<(String, String), String> {
    let (name, value) = text
        .split_once('=')
        .ok_or_else(|| format!("expected NAME=VALUE, got '{text}'"))?;
    let name = name.trim();
    if name.is_empty() || !name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
        return Err(format!("'{name}' is not a valid wire name in '{text}'"));
    }
    Ok((name.to_string(), value.trim().to_string()))
}

/// Add `tiles` a host built from what it holds to the program, as
/// `tile` statements at its top level.
///
/// A tile a host supplies is the same thing as a tile the author
/// wrote: it is one `tile` statement either way, and from here on the
/// program's own typing, lifecycle classification and engine selection
/// apply to it exactly as they apply to the author's. That is why this
/// is a transform and not a compile entry point — a host that also
/// wants [`assign_values`], or a rewrite of its own, applies them to
/// the same tree in whatever order it means, and compiles once.
///
/// A tile whose name the program already declares is an error naming
/// it, rather than a second declaration for the compiler to resolve.
/// The check reaches module and `for` bodies, so a host cannot shadow
/// a nested tile without knowing it.
pub fn add_tiles(file: &mut PolydatFile, tiles: Vec<TileDef>) -> Result<(), String> {
    let mut declared: Vec<String> = Vec::new();
    each_tile::<std::convert::Infallible>(&mut file.statements, &mut |tile| {
        declared.push(tile.name.clone());
        Ok(())
    })
    .expect("the closure never fails");
    for tile in &tiles {
        if declared.contains(&tile.name) {
            return Err(format!(
                "the program already declares a tile named '{}'; a host tile cannot \
                 replace one the program declares. Rename the host tile, or rewrite \
                 the declared one in place with `tile_named`.",
                tile.name,
            ));
        }
        declared.push(tile.name.clone());
    }
    file.statements
        .extend(tiles.into_iter().map(Statement::Tile));
    Ok(())
}

// ── Addressing a subtree ────────────────────────────────────────────
//
// A transform reads and rewrites the parsed program, never its source
// text: the text is what produced the tree and has no authority over
// it afterwards. These walk the two nesting axes so a transform can be
// written once and applied to a whole program or to one part of it.
//
// Statements nest through module bodies and `for` bodies; a template's
// pieces nest through projection bodies and branch arms. Each walker
// takes the slice to walk rather than the file, so the caller chooses
// the subtree: pass `&mut file.statements` for the program, a module's
// `body` for that module, or one tile's `pieces` for that tile.

/// Apply `f` to every statement in `statements`, then to every
/// statement nested in a module body or a `for` body, depth first.
///
/// `f` sees a statement before its own nested bodies are walked, so a
/// transform may rewrite a statement and have the walk continue into
/// what it wrote.
pub fn each_statement<E>(
    statements: &mut [Statement],
    f: &mut impl FnMut(&mut Statement) -> Result<(), E>,
) -> Result<(), E> {
    for stmt in statements.iter_mut() {
        f(stmt)?;
        match stmt {
            Statement::ModuleDef(m) => each_statement(&mut m.body, f)?,
            Statement::For(s) => each_statement(&mut s.body, f)?,
            _ => {}
        }
    }
    Ok(())
}

/// Apply `f` to every tile the subtree declares, module and `for`
/// bodies included.
pub fn each_tile<E>(
    statements: &mut [Statement],
    f: &mut impl FnMut(&mut TileDef) -> Result<(), E>,
) -> Result<(), E> {
    each_statement(statements, &mut |stmt| match stmt {
        Statement::Tile(t) => f(t),
        _ => Ok(()),
    })
}

/// The tile bound to `name` in the subtree, for a transform qualified
/// to one definition.
pub fn tile_named<'a>(statements: &'a mut [Statement], name: &str) -> Option<&'a mut TileDef> {
    for stmt in statements.iter_mut() {
        match stmt {
            Statement::Tile(t) if t.name == name => return Some(t),
            Statement::ModuleDef(m) => {
                if let Some(t) = tile_named(&mut m.body, name) {
                    return Some(t);
                }
            }
            Statement::For(s) => {
                if let Some(t) = tile_named(&mut s.body, name) {
                    return Some(t);
                }
            }
            _ => {}
        }
    }
    None
}

/// Apply `f` to every piece of a template, then to every piece nested
/// in a projection body or a branch arm, depth first.
///
/// Rewriting a piece rewrites the tile: a tile's text is rendered from
/// its pieces, so what the program projects after a transform is what
/// it renders, with no second copy to keep in step.
pub fn each_piece<E>(
    pieces: &mut [TilePiece],
    f: &mut impl FnMut(&mut TilePiece) -> Result<(), E>,
) -> Result<(), E> {
    for piece in pieces.iter_mut() {
        f(piece)?;
        match piece {
            TilePiece::Projection { body, .. } => each_piece(body, f)?,
            TilePiece::Branch {
                then, otherwise, ..
            } => {
                each_piece(then, f)?;
                if let Some(arm) = otherwise {
                    each_piece(arm, f)?;
                }
            }
            TilePiece::Static(_) | TilePiece::Hole(_) => {}
        }
    }
    Ok(())
}

/// What a scope of a parsed program binds, by name.
///
/// A host that rewrites a program before compiling it often needs to
/// know what the program will call things — which wires a scope has,
/// so an added binding can name them. The compiled program knows
/// exactly; a host that only has the source would otherwise work it
/// out again, and a second implementation of "what does this scope
/// bind" is a second answer to a question with one.
///
/// `declared_wires` is that answer, read from the AST. It is checked
/// against the compiler's own in the suite.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ScopeWires {
    /// Names bound in this scope, in declaration order: bindings,
    /// tiles, and — inside a traversal body — the element names the
    /// traversal's source binds.
    pub names: Vec<String>,
}

impl ScopeWires {
    fn push(&mut self, name: &str) {
        if !name.starts_with("__") && !self.names.iter().any(|n| n == name) {
            self.names.push(name.to_string());
        }
    }

    /// Whether this scope binds `name`.
    pub fn binds(&self, name: &str) -> bool {
        self.names.iter().any(|n| n == name)
    }
}

/// Every scope of a parsed program, by name.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DeclaredWires {
    /// The root scope's bindings and tiles.
    pub root: ScopeWires,
    /// The names the root declares as inputs or extern ports. Each is
    /// a slot and a passthrough output under the same name, so it can
    /// be read like a binding; it is listed separately because a host
    /// selecting "the program's outputs" usually means the bindings.
    pub inputs: Vec<String>,
    /// One entry per top-level `for` statement, in source order.
    pub traversal_bodies: Vec<ScopeWires>,
}

/// Read [`DeclaredWires`] from a parsed program.
///
/// A traversal body's element names come from its source: a
/// comprehension binds its own, a bare producer name or a derivation
/// of one binds the producer's, which is resolved against the
/// bindings of the enclosing scope.
pub fn declared_wires(file: &PolydatFile) -> DeclaredWires {
    let mut out = DeclaredWires::default();
    for stmt in &file.statements {
        match stmt {
            Statement::Binding(b) => {
                for t in &b.targets {
                    out.root.push(t);
                }
            }
            Statement::Tile(t) => out.root.push(&t.name),
            Statement::InputDecl(i) => out.inputs.push(i.name.clone()),
            Statement::ExternPort(p) => out.inputs.push(p.name.clone()),
            _ => {}
        }
    }
    // Bodies in a second pass, because a body sees the scope that
    // encloses it: an outer wire the body names cascades in as an
    // input of the child program and is readable there under the same
    // name. Its own names come first, so a listing reads body-first.
    for stmt in &file.statements {
        if let Statement::For(f) = stmt {
            let mut body = ScopeWires::default();
            for name in source_element_names(&f.source, file) {
                body.push(&name);
            }
            collect_scope(&f.body, file, &mut body);
            for name in out.root.names.iter().chain(out.inputs.iter()) {
                body.push(name);
            }
            out.traversal_bodies.push(body);
        }
    }
    out
}

/// The bindings and tiles of a statement list, appended to `into`.
fn collect_scope(body: &[Statement], file: &PolydatFile, into: &mut ScopeWires) {
    for stmt in body {
        match stmt {
            Statement::Binding(b) => {
                for t in &b.targets {
                    into.push(t);
                }
            }
            Statement::Tile(t) => into.push(&t.name),
            Statement::For(f) => {
                // A nested traversal is its own scope; its element
                // names and bindings do not reach this one.
                let _ = (f, file);
            }
            _ => {}
        }
    }
}

/// The element names a `for` source binds.
fn source_element_names(source: &crate::dsl::ast::ForSource, file: &PolydatFile) -> Vec<String> {
    use crate::dsl::ast::ForSourceKind;
    match &source.kind {
        ForSourceKind::Comprehension(c) => {
            c.coordinate_specs().into_iter().map(|(v, _)| v).collect()
        }
        ForSourceKind::Producer(name) => producer_element_names(name, file),
        ForSourceKind::Derived { base, .. } => producer_element_names(base, file),
    }
}

/// The element names of the producer bound to `name` at the root:
/// `name := for <comprehension>`, or a derivation of another producer,
/// which is followed to its base.
fn producer_element_names(name: &str, file: &PolydatFile) -> Vec<String> {
    let mut seen = 0usize;
    let mut target = name.to_string();
    // A derivation chain is finite; the bound is the statement count,
    // so a cycle a malformed program could write ends the walk rather
    // than spinning.
    while seen <= file.statements.len() {
        seen += 1;
        let found = file.statements.iter().find_map(|s| match s {
            Statement::Binding(b) if b.targets.contains(&target) => match &b.value {
                Expr::For(src) => Some(src),
                _ => None,
            },
            _ => None,
        });
        let Some(src) = found else { return Vec::new() };
        use crate::dsl::ast::ForSourceKind;
        match &src.kind {
            ForSourceKind::Comprehension(c) => {
                return c.coordinate_specs().into_iter().map(|(v, _)| v).collect();
            }
            ForSourceKind::Producer(next) | ForSourceKind::Derived { base: next, .. } => {
                target = next.clone();
            }
        }
    }
    Vec::new()
}
