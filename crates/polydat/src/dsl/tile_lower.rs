// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Lowering a `tile` statement to a program (SRD 114 §6).
//!
//! Each hole becomes a `tile_encode` binding in the enclosing scope
//! whose constant spec carries the tile's encoding, the hole's position,
//! its declared type, its format, and its raw flag. Static runs fold
//! into single instructions. Projections compile their body into a
//! child program text whose holes are `tile_encode` bindings too; the
//! render node compiles that body once at setup and re-runs it per
//! tuple. The tile itself becomes a `tile_render` call over the encoded
//! hole wires and any wires a body imports.

use std::collections::BTreeSet;

use crate::ast::PortType;
use crate::compile::assembly::PolydatAssembler;
use crate::iteration::comprehension::StreamerValue;
use crate::library::tile_render::{ChildSpec, HoleEncoding, HolePosition, HoleSource, TileOp, TileSpec};

use super::ast::{Arg, CallExpr, Expr, ForSourceKind, TileDef, TilePiece};
use super::compile::Compiler;
use super::refs::collect_expr_refs;

impl Compiler {
    /// Lower a tile into encode bindings and a `tile_render` binding
    /// named after the tile. Records the tile so later tiles can splice
    /// it.
    pub(super) fn compile_tile(&mut self, asm: &mut PolydatAssembler, tile: &TileDef) -> Result<(), String> {
        let encoding = tile.encoding.clone().unwrap_or_else(|| "text".to_string());
        let mut lowering = TileLowering {
            tile_name: tile.name.clone(),
            encoding: encoding.clone(),
            inputs: Vec::new(),
            children: Vec::new(),
            hole_counter: 0,
            in_string: false,
            span: tile.span,
        };
        let pieces = self.splice_tiles(&tile.pieces, &tile.name, &tile.encoding, 0)?;
        let ops = lowering.lower_pieces(self, asm, &pieces, None)?;
        let spec = TileSpec { name: tile.name.clone(), encoding, ops, children: lowering.children };
        let mut args = vec![Arg::Positional(Expr::StringLit(spec.to_json(), tile.span))];
        for name in &lowering.inputs {
            args.push(Arg::Positional(Expr::Ident(name.clone(), tile.span)));
        }
        let call = Expr::Call(CallExpr { func: "tile_render".into(), args, span: tile.span });
        self.compile_binding(asm, std::slice::from_ref(&tile.name), &call)?;
        if lowering.inputs.is_empty() {
            // No holes: the tile is a constant, and says so.
            asm.mark_const_output(&tile.name);
            asm.set_output_modifier(&tile.name, super::ast::BindingModifier::CONST);
        }
        self.tiles.push(tile.clone());
        Ok(())
    }

    /// Replace splice holes, `${name}` where `name` is an earlier tile,
    /// with that tile's pieces, recursively.
    fn splice_tiles(&self, pieces: &[TilePiece], current: &str, encoding: &Option<String>, depth: usize) -> Result<Vec<TilePiece>, String> {
        if depth > 16 {
            return Err(format!("tile '{current}': splice nesting too deep; is there a cycle?"));
        }
        let mut out = Vec::with_capacity(pieces.len());
        for piece in pieces {
            match piece {
                TilePiece::Hole(h) => {
                    if let Expr::Ident(name, _) = &h.expr
                        && h.decl_type.is_none()
                        && h.format.is_none()
                        && let Some(other) = self.tiles.iter().rev().find(|t| &t.name == name)
                    {
                        if name == current {
                            return Err(format!("tile '{current}' splices itself"));
                        }
                        // Pieces are spliced only into a tile of the same
                        // encoding, where they mean the same thing. Across
                        // encodings the inner tile is an ordinary wire:
                        // its rendered text enters through the hole and
                        // is encoded (or, with `!`, inlined) as any string.
                        if other.encoding.as_deref().unwrap_or("text") == encoding.as_deref().unwrap_or("text") {
                            out.extend(self.splice_tiles(&other.pieces, name, encoding, depth + 1)?);
                            continue;
                        }
                    }
                    out.push(piece.clone());
                }
                TilePiece::Projection { source, sep, body, span } => out.push(TilePiece::Projection {
                    source: source.clone(),
                    sep: sep.clone(),
                    body: self.splice_tiles(body, current, encoding, depth + 1)?,
                    span: *span,
                }),
                TilePiece::Branch { cond, then, otherwise, span } => out.push(TilePiece::Branch {
                    cond: cond.clone(),
                    then: self.splice_tiles(then, current, encoding, depth + 1)?,
                    otherwise: match otherwise {
                        Some(o) => Some(self.splice_tiles(o, current, encoding, depth + 1)?),
                        None => None,
                    },
                    span: *span,
                }),
                other => out.push(other.clone()),
            }
        }
        Ok(out)
    }
}

/// State for one tile's lowering.
struct TileLowering {
    tile_name: String,
    encoding: String,
    /// Wire names passed to `tile_render`, in input order.
    inputs: Vec<String>,
    children: Vec<ChildSpec>,
    hole_counter: usize,
    /// Whether the static text so far leaves a `json` skeleton inside a
    /// string literal.
    in_string: bool,
    span: super::lexer::Span,
}

/// A projection body under construction.
struct BodyContext {
    elements: Vec<(String, PortType)>,
    bindings: Vec<String>,
    /// Outer names the body references: `(name, node input index, type keyword)`.
    cascade: Vec<(String, usize, String)>,
    counter: usize,
}

impl TileLowering {
    fn lower_pieces(
        &mut self,
        compiler: &mut Compiler,
        asm: &mut PolydatAssembler,
        pieces: &[TilePiece],
        mut body: Option<&mut BodyContext>,
    ) -> Result<Vec<TileOp>, String> {
        let mut ops: Vec<TileOp> = Vec::new();
        let mut static_buf = String::new();
        let flush = |buf: &mut String, ops: &mut Vec<TileOp>| {
            if !buf.is_empty() {
                ops.push(TileOp::Static(std::mem::take(buf)));
            }
        };
        for piece in pieces {
            match piece {
                TilePiece::Static(s) => {
                    self.track_string_state(s);
                    static_buf.push_str(s);
                }
                TilePiece::Hole(h) => {
                    flush(&mut static_buf, &mut ops);
                    let enc = HoleEncoding {
                        encoding: self.encoding.clone(),
                        position: self.position(),
                        ty: h.decl_type.clone(),
                        format: h.format.clone(),
                        raw: h.raw,
                        cond: false,
                    };
                    let source = match body.as_deref_mut() {
                        Some(ctx) => self.body_hole(asm, &h.expr, &enc, ctx)?,
                        None => self.wire_hole(compiler, asm, &h.text, &h.expr, &enc)?,
                    };
                    ops.push(TileOp::Hole(source));
                }
                TilePiece::Branch { cond, then, otherwise, .. } => {
                    flush(&mut static_buf, &mut ops);
                    let enc = HoleEncoding {
                        encoding: self.encoding.clone(),
                        position: HolePosition::Text,
                        ty: None,
                        format: None,
                        raw: true,
                        cond: true,
                    };
                    let source = match body.as_deref_mut() {
                        Some(ctx) => self.body_hole(asm, cond, &enc, ctx)?,
                        None => self.wire_hole(compiler, asm, "condition", cond, &enc)?,
                    };
                    let saved = self.in_string;
                    let then_ops = self.lower_pieces(compiler, asm, then, body.as_deref_mut())?;
                    self.in_string = saved;
                    let else_ops = match otherwise {
                        Some(o) => self.lower_pieces(compiler, asm, o, body.as_deref_mut())?,
                        None => Vec::new(),
                    };
                    self.in_string = saved;
                    ops.push(TileOp::Branch { cond: source, then: then_ops, otherwise: else_ops });
                }
                TilePiece::Projection { source, sep, body: proj_body, .. } => {
                    if body.is_some() {
                        return Err(format!("tile '{}': a projection inside a projection body is not supported yet", self.tile_name));
                    }
                    flush(&mut static_buf, &mut ops);
                    let comprehension = match &source.kind {
                        ForSourceKind::Comprehension(c) => c.clone(),
                        ForSourceKind::Producer(name) => compiler
                            .producer_comprehension(name)
                            .ok_or_else(|| format!("tile '{}': producer '{name}' is not bound before this tile", self.tile_name))?,
                        ForSourceKind::Derived { .. } => {
                            return Err(format!(
                                "tile '{}': a derived comprehension (`for base where ...`) is not supported in a projection yet; bind it as a producer first",
                                self.tile_name
                            ));
                        }
                    };
                    let stream = StreamerValue::new(source.text.clone(), comprehension.clone()).to_json();
                    let mut probe = |_: &str| -> Result<PortType, String> {
                        Err("generator-call sources are not supported in tile projections yet".to_string())
                    };
                    let elements = super::traversal::element_types(&comprehension, &mut probe)
                        .map_err(|e| format!("tile '{}': projection: {e}", self.tile_name))?;
                    let mut ctx = BodyContext { elements, bindings: Vec::new(), cascade: Vec::new(), counter: 0 };
                    let saved = self.in_string;
                    let body_ops = self.lower_pieces(compiler, asm, proj_body, Some(&mut ctx))?;
                    self.in_string = saved;
                    // The body's own input is the tuple index; the
                    // program's `cycle` reaches it as a cascaded extern
                    // like any other outer wire.
                    let mut src = String::from("input __tuple: u64\n");
                    for (name, ty) in &ctx.elements {
                        src.push_str(&format!("extern {name}: {}\n", ty.to_keyword()));
                    }
                    for (name, _, ty) in &ctx.cascade {
                        src.push_str(&format!("extern {name}: {ty}\n"));
                    }
                    for b in &ctx.bindings {
                        src.push_str(b);
                        src.push('\n');
                    }
                    // Compile the body once here so a bad reference is a
                    // compile error of the enclosing program, not a
                    // failure inside `tile_render` construction.
                    super::compile_polydat(&src).map_err(|e| {
                        format!("tile '{}': projection body: {e}", self.tile_name)
                    })?;
                    self.children.push(ChildSpec { source: src, cascade: ctx.cascade });
                    let child = self.children.len() - 1;
                    let default_sep = match self.encoding.as_str() {
                        "json" | "csv" => ",",
                        _ => "",
                    };
                    ops.push(TileOp::Repeat { stream, child, sep: sep.clone().unwrap_or_else(|| default_sep.to_string()), body: body_ops });
                }
            }
        }
        flush(&mut static_buf, &mut ops);
        Ok(ops)
    }

    /// A hole in the tile's own scope: `__tile_<name>_hN := tile_encode(expr, spec)`.
    fn wire_hole(&mut self, compiler: &mut Compiler, asm: &mut PolydatAssembler, text: &str, expr: &Expr, enc: &HoleEncoding) -> Result<HoleSource, String> {
        validate_type(text, enc)?;
        let name = self.next_name("h");
        let call = encode_call(expr, enc, self.span);
        compiler
            .compile_binding(asm, std::slice::from_ref(&name), &call)
            .map_err(|e| format!("tile '{}': hole `{text}`: {e}", self.tile_name))?;
        let index = self.push_input(name);
        Ok(HoleSource::Wire(index))
    }

    /// A hole inside a projection body: a `tile_encode` binding of the
    /// child program. Outer names it references cascade in as render
    /// node inputs, typed by the parent's wire.
    fn body_hole(&mut self, asm: &mut PolydatAssembler, expr: &Expr, enc: &HoleEncoding, ctx: &mut BodyContext) -> Result<HoleSource, String> {
        validate_type("body hole", enc)?;
        let mut refs = BTreeSet::new();
        collect_expr_refs(expr, &mut refs);
        for r in refs {
            if ctx.elements.iter().any(|(n, _)| *n == r) || ctx.cascade.iter().any(|(n, _, _)| *n == r) {
                continue;
            }
            let ty = asm
                .node_output_type(&r)
                .or_else(|| if asm.input_names().iter().any(|n| *n == r) { Some(PortType::U64) } else { None });
            if let Some(ty) = ty {
                let idx = self.push_input(r.clone());
                ctx.cascade.push((r, idx, ty.to_keyword().to_string()));
            }
            // Names the parent does not define are left for the child
            // compile to report as unknown wires.
        }
        let name = format!("__b{}", ctx.counter);
        ctx.counter += 1;
        let call = encode_call(expr, enc, self.span);
        ctx.bindings.push(format!("{name} := {}", super::pprint::pp_expr(&call)));
        Ok(HoleSource::Child(name))
    }

    fn push_input(&mut self, name: String) -> usize {
        if let Some(i) = self.inputs.iter().position(|n| *n == name) {
            return i;
        }
        self.inputs.push(name);
        self.inputs.len() - 1
    }

    fn next_name(&mut self, kind: &str) -> String {
        let n = self.hole_counter;
        self.hole_counter += 1;
        format!("__tile_{}_{kind}{n}", self.tile_name)
    }

    fn position(&self) -> HolePosition {
        if self.encoding != "json" {
            HolePosition::Text
        } else if self.in_string {
            HolePosition::InString
        } else {
            HolePosition::Value
        }
    }

    /// Track whether static JSON text ends inside a string literal.
    fn track_string_state(&mut self, s: &str) {
        if self.encoding != "json" {
            return;
        }
        let mut escaped = false;
        for c in s.chars() {
            if self.in_string {
                if escaped {
                    escaped = false;
                } else if c == '\\' {
                    escaped = true;
                } else if c == '"' {
                    self.in_string = false;
                }
            } else if c == '"' {
                self.in_string = true;
            }
        }
    }
}

fn validate_type(text: &str, enc: &HoleEncoding) -> Result<(), String> {
    if let Some(kw) = &enc.ty
        && PortType::from_keyword(kw).is_none()
    {
        return Err(format!("hole `{text}`: unknown type '{kw}'"));
    }
    Ok(())
}

/// `tile_encode(expr, "<spec>")`.
fn encode_call(expr: &Expr, enc: &HoleEncoding, span: super::lexer::Span) -> Expr {
    Expr::Call(CallExpr {
        func: "tile_encode".into(),
        args: vec![Arg::Positional(expr.clone()), Arg::Positional(Expr::StringLit(enc.to_spec(), span))],
        span,
    })
}

