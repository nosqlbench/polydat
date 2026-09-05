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
            strict: self.strict || tile.options.strict,
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
    /// Reject implicit adapters at holes (SRD 114 §4.4).
    strict: bool,
    span: super::lexer::Span,
}

/// How one hole was typed (SRD 114 §4): the wire's compile-time type,
/// the declared type if any, the type the encoder sees, and the
/// adapter between them when the declaration asks for one.
struct HoleTyping {
    wire: PortType,
    declared: Option<PortType>,
    effective: PortType,
    adapter: Option<(PortType, PortType)>,
    expectation: &'static str,
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
                        Some(ctx) => self.body_hole(compiler, asm, &h.text, &h.expr, enc, ctx)?,
                        None => self.wire_hole(compiler, asm, &h.text, &h.expr, enc)?,
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
                    let text = format!("if {}", super::pprint::pp_expr(cond));
                    let source = match body.as_deref_mut() {
                        Some(ctx) => self.body_hole(compiler, asm, &text, cond, enc, ctx)?,
                        None => self.wire_hole(compiler, asm, &text, cond, enc)?,
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

    /// The compile-time type of a hole expression (SRD 114 §4.2): the
    /// same inference every binding gets, with projection elements and
    /// cascaded outer wires resolved from the body context.
    fn infer_type(&self, compiler: &Compiler, asm: &PolydatAssembler, expr: &Expr, ctx: Option<&BodyContext>) -> PortType {
        if let Expr::Ident(name, _) = expr
            && let Some(ctx) = ctx
        {
            if let Some((_, t)) = ctx.elements.iter().find(|(n, _)| n == name) {
                return *t;
            }
            if let Some((_, _, t)) = ctx.cascade.iter().find(|(n, _, _)| n == name) {
                return PortType::from_keyword(t).unwrap_or(PortType::U64);
            }
        }
        super::binding::infer_expr_type(expr, asm, &compiler.input_names)
    }

    /// Type one hole (SRD 114 §4): the declared type wins and must be
    /// reachable from the wire type through the assembler's own adapter
    /// catalog; otherwise the wire type stands. Strict mode rejects the
    /// adapter a declaration would need.
    fn type_hole(&self, text: &str, wire: PortType, declared: Option<&str>, position: HolePosition, cond: bool) -> Result<HoleTyping, String> {
        let tile = &self.tile_name;
        let expectation = if cond {
            "a truth value"
        } else {
            match (self.encoding.as_str(), position) {
                ("json", HolePosition::Value) => "any JSON value",
                ("json", HolePosition::InString) => "text inside a JSON string",
                ("csv", _) => "a CSV field",
                _ => "text",
            }
        };
        if matches!(
            wire,
            PortType::Ext
                | PortType::Handle
                | PortType::Reg128
                | PortType::RegI8x16
                | PortType::RegI16x8
                | PortType::RegI32x4
                | PortType::RegI64x2
                | PortType::RegF16x8
                | PortType::RegF32x4
                | PortType::RegF64x2
        ) {
            return Err(format!(
                "tile '{tile}': hole `{text}`: a {} wire has no text form and cannot fill a hole",
                wire.to_keyword()
            ));
        }
        let declared_ty = match declared {
            Some(kw) => Some(
                PortType::from_keyword(kw).ok_or_else(|| format!("tile '{tile}': hole `{text}`: unknown type '{kw}'"))?,
            ),
            None => None,
        };
        let mut adapter = None;
        if let Some(to) = declared_ty
            && to != wire
        {
            if crate::compile::assembly::auto_adapter(wire, to).is_none() {
                return Err(format!(
                    "tile '{tile}': hole `{text}`: no conversion from the wire type {} to the declared type {}; \
                     write the conversion explicitly in the expression",
                    wire.to_keyword(),
                    to.to_keyword()
                ));
            }
            if self.strict {
                return Err(format!(
                    "tile '{tile}': hole `{text}`: strict mode rejects the implicit {} -> {} adapter the declaration needs; \
                     write the conversion explicitly in the expression",
                    wire.to_keyword(),
                    to.to_keyword()
                ));
            }
            adapter = Some((wire, to));
        }
        Ok(HoleTyping { wire, declared: declared_ty, effective: declared_ty.unwrap_or(wire), adapter, expectation })
    }

    /// Record how a hole was typed for `explain tiles` (SRD 114 §4.4).
    fn record(&self, compiler: &mut Compiler, text: &str, typing: &HoleTyping, enc: &HoleEncoding) {
        let kw = typing.effective.to_keyword();
        let mut encoder = if enc.cond {
            "truth value as 1 or 0".to_string()
        } else if enc.raw {
            "raw text, no escaping".to_string()
        } else {
            match (enc.encoding.as_str(), enc.position) {
                ("json", HolePosition::Value) => match typing.effective {
                    PortType::Str | PortType::Bytes => "json string, quoted and escaped".to_string(),
                    PortType::Bool => "json boolean".to_string(),
                    PortType::Json => "json value, serialized".to_string(),
                    t if is_numeric(t) => "json number".to_string(),
                    _ => "json string, quoted and escaped".to_string(),
                },
                ("json", HolePosition::InString) => "json escaped text".to_string(),
                ("csv", _) => "csv field, quoted when needed".to_string(),
                _ => "display text".to_string(),
            }
        };
        if let Some(f) = &enc.format {
            encoder.push_str(&format!(", format {f}"));
        }
        compiler.tile_events.push(super::events::CompileEvent::TileHoleTyped {
            tile: self.tile_name.clone(),
            hole: text.to_string(),
            wire_type: typing.wire.to_keyword().to_string(),
            declared: typing.declared.map(|t| t.to_keyword().to_string()),
            expectation: format!("{} ({kw})", typing.expectation),
            encoder,
            adapter: typing.adapter.map(|(f, t)| format!("{} -> {}", f.to_keyword(), t.to_keyword())),
        });
    }

    /// A hole in the tile's own scope: `__tile_<name>_hN := tile_encode(expr, spec)`,
    /// with the declaration's adapter node between when one is needed.
    fn wire_hole(&mut self, compiler: &mut Compiler, asm: &mut PolydatAssembler, text: &str, expr: &Expr, mut enc: HoleEncoding) -> Result<HoleSource, String> {
        let wire = self.infer_type(compiler, asm, expr, None);
        let typing = self.type_hole(text, wire, enc.ty.as_deref(), enc.position, enc.cond)?;
        enc.ty = Some(typing.effective.to_keyword().to_string());
        let name = self.next_name("h");
        let err = |e: String| format!("tile '{}': hole `{text}`: {e}", self.tile_name);
        let value = match typing.adapter {
            Some((from, to)) => {
                let vname = format!("{name}_v");
                compiler.compile_binding(asm, std::slice::from_ref(&vname), expr).map_err(err)?;
                let aname = format!("{name}_a");
                let node = crate::compile::assembly::auto_adapter(from, to).expect("adapter checked by type_hole");
                asm.add_node(&aname, node, vec![crate::compile::assembly::WireRef::node(&vname)]);
                compiler.all_names.push(aname.clone());
                Expr::Ident(aname, self.span)
            }
            None => expr.clone(),
        };
        let call = encode_call(&value, &enc, self.span);
        compiler.compile_binding(asm, std::slice::from_ref(&name), &call).map_err(err)?;
        self.record(compiler, text, &typing, &enc);
        let index = self.push_input(name);
        Ok(HoleSource::Wire(index))
    }

    /// A hole inside a projection body: a `tile_encode` binding of the
    /// child program. Outer names it references cascade in as render
    /// node inputs, typed by the parent's wire.
    fn body_hole(
        &mut self,
        compiler: &mut Compiler,
        asm: &mut PolydatAssembler,
        text: &str,
        expr: &Expr,
        mut enc: HoleEncoding,
        ctx: &mut BodyContext,
    ) -> Result<HoleSource, String> {
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
        let wire = self.infer_type(compiler, asm, expr, Some(ctx));
        let typing = self.type_hole(text, wire, enc.ty.as_deref(), enc.position, enc.cond)?;
        enc.ty = Some(typing.effective.to_keyword().to_string());
        let value = match typing.adapter {
            // The child is source text, so the adapter is the `as`
            // fusion, which covers the widening the catalog allows here.
            Some((PortType::U64, PortType::F64)) => Expr::Cast(Box::new(expr.clone()), PortType::F64, self.span),
            Some((from, to)) if to == PortType::Str || to == PortType::Bool => {
                // Display and truth readings need no node: the encoder
                // applies them from the effective type.
                let _ = from;
                expr.clone()
            }
            Some((from, to)) => {
                return Err(format!(
                    "tile '{}': hole `{text}`: the {} -> {} adapter is not supported inside a projection body yet; \
                     write the conversion explicitly in the expression",
                    self.tile_name,
                    from.to_keyword(),
                    to.to_keyword()
                ));
            }
            None => expr.clone(),
        };
        let name = format!("__b{}", ctx.counter);
        ctx.counter += 1;
        let call = encode_call(&value, &enc, self.span);
        ctx.bindings.push(format!("{name} := {}", super::pprint::pp_expr(&call)));
        self.record(compiler, text, &typing, &enc);
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

fn is_numeric(t: PortType) -> bool {
    matches!(
        t,
        PortType::U64
            | PortType::I64
            | PortType::F64
            | PortType::U32
            | PortType::I32
            | PortType::F32
            | PortType::U16
            | PortType::I16
            | PortType::U8
            | PortType::I8
            | PortType::F16
            | PortType::U128
            | PortType::I128
    )
}

/// `tile_encode(expr, "<spec>")`.
fn encode_call(expr: &Expr, enc: &HoleEncoding, span: super::lexer::Span) -> Expr {
    Expr::Call(CallExpr {
        func: "tile_encode".into(),
        args: vec![Arg::Positional(expr.clone()), Arg::Positional(Expr::StringLit(enc.to_spec(), span))],
        span,
    })
}

