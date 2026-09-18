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

use super::ast::{Expr, ExternPort, PolydatFile, Statement, TileOptions};

/// Give every tile that declares no delimiters or sigil of its own the
/// host's defaults (SRD 114 §5.6, §10).
///
/// **Prefer
/// [`parser::parse_with_tile_defaults`](crate::dsl::parser::parse_with_tile_defaults).**
/// A tile body is raw text until it is read, and what it means depends
/// on the delimiters it is read under, so the host's belong at the
/// parse, not after it. This function re-reads a tile that has already
/// been read once, by rendering its template back to text under the
/// options it was read with and admitting that text under the new ones.
/// The round trip is exact except where static text contains the old
/// open delimiter, which the render escapes by doubling and the new
/// reading does not undo.
///
/// A tile that names any option keeps all of them; only what the author
/// left to the default is filled in. Tiles inside module bodies are
/// rewritten too, since they compile in this program.
#[deprecated(
    note = "parse under the host's delimiters instead: dsl::parser::parse_with_tile_defaults"
)]
pub fn apply_tile_defaults(file: &mut PolydatFile, defaults: &TileOptions) -> Result<(), String> {
    fn visit(statements: &mut [Statement], defaults: &TileOptions) -> Result<(), String> {
        let stock = TileOptions::default();
        for stmt in statements.iter_mut() {
            match stmt {
                Statement::Tile(t) => {
                    let untouched = t.options.open == stock.open
                        && t.options.close == stock.close
                        && t.options.sigil == stock.sigil;
                    if !untouched {
                        continue;
                    }
                    let as_written = t.body_text();
                    let strict = t.options.strict;
                    t.options = defaults.clone();
                    t.options.strict = strict || defaults.strict;
                    t.pieces = super::tile::parse_template(&as_written, &t.options, t.span)
                        .map_err(|e| format!("tile '{}' under host delimiters: {e}", t.name))?;
                }
                Statement::ModuleDef(m) => visit(&mut m.body, defaults)?,
                _ => {}
            }
        }
        Ok(())
    }
    visit(&mut file.statements, defaults)
}

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
        for stmt in file.statements.iter_mut() {
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
        }
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
