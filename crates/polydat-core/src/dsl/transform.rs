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
