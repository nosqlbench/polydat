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

use super::ast::{Arg, CallExpr, Expr, ForSource, ForSourceKind, TileDef, TileOptions, TilePiece};
use super::compile::Compiler;
use super::refs::collect_expr_refs;
use super::tile::render_template;

impl Compiler {
    /// Lower a tile into encode bindings and a `tile_render` binding
    /// named after the tile. Records the tile so later tiles can splice
    /// it.
    pub(super) fn compile_tile(&mut self, asm: &mut PolydatAssembler, tile: &TileDef) -> Result<(), String> {
        let encoding = tile.encoding.clone().unwrap_or_else(|| "text".to_string());
        let mut lowering = TileLowering {
            tile_name: tile.name.clone(),
            encoding: encoding.clone(),
            options: tile.options.clone(),
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
        let mut shape = SkeletonShape::default();
        shape.count(&spec.ops);
        self.tile_events.push(super::events::CompileEvent::TileCompiled {
            tile: tile.name.clone(),
            encoding: spec.encoding.clone(),
            statics: shape.statics,
            static_bytes: shape.static_bytes,
            holes: shape.holes,
            branches: shape.branches,
            projections: shape.projections,
            bodies: spec.children.iter().map(|c| c.source.clone()).collect(),
        });
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
    options: TileOptions,
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
    /// Producers re-bound inside the body program for nested projections.
    producers: Vec<String>,
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
                    flush(&mut static_buf, &mut ops);
                    if let Some(ctx) = body.as_deref_mut() {
                        let op = self.nested_projection(compiler, asm, source, sep, proj_body, ctx)?;
                        ops.push(op);
                        continue;
                    }
                    // The same resolution the `for` construct uses, so
                    // inline text, bound producers, and derivations
                    // (`base where ... order ...`) all project.
                    let comprehension = super::traversal::resolve_source(source, &compiler.producers_seen)
                        .map_err(|e| format!("tile '{}': projection: {e}", self.tile_name))?;
                    self.check_bounded(&comprehension, &source.text)?;
                    self.check_predicates(&comprehension, &source.text)?;
                    let stream = StreamerValue::new(source.text.clone(), comprehension.clone()).to_json();
                    let mut probe = |expr: &str| self.generator_type(compiler, asm, expr, None);
                    let elements = super::traversal::element_types(&comprehension, &mut probe)
                        .map_err(|e| format!("tile '{}': projection `for {}`: {e}", self.tile_name, source.text))?;
                    // Generator-call sources are expressions over the
                    // enclosing scope: each compiles to a wire here and
                    // its value reaches the render node as an input, so
                    // the projection needs no kernel of its own.
                    let mut generators = Vec::new();
                    for (name, expr_text) in generator_clauses(&comprehension) {
                        if expr_text.contains('{') {
                            return Err(format!(
                                "tile '{}': projection `for {}`: generator `{expr_text}` uses a `{{name}}` placeholder; \
                                 in a tile the generator is an expression over the scope's wires, so name the wire directly",
                                self.tile_name, source.text
                            ));
                        }
                        let expr = super::tile::parse_hole_expr(&expr_text)
                            .map_err(|e| format!("tile '{}': projection `for {}`: generator `{expr_text}`: {e}", self.tile_name, source.text))?;
                        let wire = self.next_name("g");
                        compiler
                            .compile_binding(asm, std::slice::from_ref(&wire), &expr)
                            .map_err(|e| format!("tile '{}': projection `for {}`: generator `{expr_text}`: {e}", self.tile_name, source.text))?;
                        let idx = self.push_input(wire);
                        let ty = elements.iter().find(|(n, _)| *n == name).map(|(_, t)| t.to_keyword()).unwrap_or("str");
                        generators.push((name, idx, ty.to_string()));
                    }
                    let mut ctx = BodyContext { elements, bindings: Vec::new(), cascade: Vec::new(), producers: Vec::new(), counter: 0 };
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
                    ops.push(TileOp::Repeat {
                        stream,
                        child,
                        sep: sep.clone().unwrap_or_else(|| default_sep.to_string()),
                        body: body_ops,
                        generators,
                    });
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

    /// The element type of a generator-call source. In a tile the
    /// generator is an expression over the enclosing scope, so it is
    /// typed by the same inference as a hole; the isolated probe the
    /// `for` construct uses is the fallback when it does not parse.
    fn generator_type(&self, compiler: &Compiler, asm: &PolydatAssembler, expr: &str, ctx: Option<&BodyContext>) -> Result<PortType, String> {
        match super::tile::parse_hole_expr(expr) {
            Ok(e) => Ok(self.infer_type(compiler, asm, &e, ctx)),
            Err(_) => compiler.probe_element_type(expr),
        }
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
            self.cascade_ref(asm, r, ctx);
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

    /// Make an outer name the body references available inside the body
    /// program as a cascaded extern typed by the parent's wire. Names
    /// the body defines itself, and names the parent does not define,
    /// are left alone; the latter surface as unknown wires when the
    /// body compiles.
    fn cascade_ref(&mut self, asm: &PolydatAssembler, r: String, ctx: &mut BodyContext) {
        if ctx.elements.iter().any(|(n, _)| *n == r)
            || ctx.cascade.iter().any(|(n, _, _)| *n == r)
            || ctx.producers.contains(&r)
        {
            return;
        }
        // An extern carries its declared type; a coordinate input is u64.
        let ty = asm.node_output_type(&r).or_else(|| {
            if asm.input_names().iter().any(|n| *n == r) {
                Some(asm.input_type(&r).unwrap_or(PortType::U64))
            } else {
                None
            }
        });
        if let Some(ty) = ty {
            let idx = self.push_input(r.clone());
            ctx.cascade.push((r, idx, ty.to_keyword().to_string()));
        }
    }

    /// SRD 114 §5.4: a projection's comprehension must have bounded
    /// cardinality.
    fn check_bounded(&self, c: &crate::iteration::comprehension::Comprehension, text: &str) -> Result<(), String> {
        use crate::iteration::comprehension::cardinality::CardinalityClass as C;
        match c.metadata().cardinality {
            C::Bounded(_) | C::BoundedAtMost(_) => Ok(()),
            // A generator or parameter list with no cardinality hint
            // reports unbounded because its count is unknown, not
            // infinite. Its finiteness is the source's own contract, as
            // it is for the `for` construct, so it projects.
            C::Unbounded if has_unhinted_source(c) => Ok(()),
            C::Unbounded => Err(format!(
                "tile '{}': projection `for {text}` has unbounded cardinality; a projection renders once per tuple and needs a finite source",
                self.tile_name
            )),
            // The comprehension runtime does not sample continuous
            // intervals into tuples yet, so a projection over one would
            // render nothing; say so instead.
            C::Continuous { .. } | C::ContinuousAtMost { .. } | C::Hybrid(_) => Err(format!(
                "tile '{}': projection `for {text}` ranges over a continuous source, which has no finite tuple set; \
                 project over a discrete range or list, or over values sampled by an expression",
                self.tile_name
            )),
        }
    }

    /// A filter predicate may name the comprehension's own elements;
    /// a `{name}` for a wire outside it has no value at render time.
    fn check_predicates(&self, c: &crate::iteration::comprehension::Comprehension, text: &str) -> Result<(), String> {
        let elements = c.coordinate_names();
        for pred in filter_predicates(c) {
            for name in placeholder_names(&pred) {
                if !elements.contains(&name) {
                    return Err(format!(
                        "tile '{}': projection `for {text}`: predicate placeholder `{{{name}}}` names a wire outside the comprehension; \
                         a projection's predicate sees only its elements",
                        self.tile_name
                    ));
                }
            }
        }
        Ok(())
    }

    /// A projection inside a projection body (SRD 114 §5.4). The body
    /// program is Polydat source, so the nested projection becomes a
    /// `tile` statement inside it, compiled by this same lowering one
    /// level down; the outer body reads the nested tile's text through a
    /// raw hole. Producers the nested source names are re-bound inside
    /// the body program from their original text.
    fn nested_projection(
        &mut self,
        compiler: &mut Compiler,
        asm: &mut PolydatAssembler,
        source: &ForSource,
        sep: &Option<String>,
        body: &[TilePiece],
        ctx: &mut BodyContext,
    ) -> Result<TileOp, String> {
        if self.in_string {
            return Err(format!(
                "tile '{}': a nested projection inside a string position is not supported yet; \
                 project the inner text into a wire of its own and reference it",
                self.tile_name
            ));
        }
        // Validate the nested source here, in the enclosing scope, so a
        // bad source is this tile's error and its element names are known.
        let comprehension = super::traversal::resolve_source(source, &compiler.producers_seen)
            .map_err(|e| format!("tile '{}': nested projection: {e}", self.tile_name))?;
        self.check_bounded(&comprehension, &source.text)?;
        self.check_predicates(&comprehension, &source.text)?;
        let mut probe = |expr: &str| self.generator_type(compiler, asm, expr, Some(&*ctx));
        let nested_elements = super::traversal::element_types(&comprehension, &mut probe)
            .map_err(|e| format!("tile '{}': nested projection `for {}`: {e}", self.tile_name, source.text))?;
        // Generator expressions of the nested source read the scope's
        // wires; they must reach the body program too.
        for (_, expr_text) in generator_clauses(&comprehension) {
            if let Ok(expr) = super::tile::parse_hole_expr(&expr_text) {
                let mut refs = BTreeSet::new();
                collect_expr_refs(&expr, &mut refs);
                for r in refs {
                    self.cascade_ref(asm, r, ctx);
                }
            }
        }
        // Producers the source names, directly or as a derivation base,
        // are re-bound in the body program from their original text.
        let producer = match &source.kind {
            ForSourceKind::Producer(n) | ForSourceKind::Derived { base: n, .. } => Some(n.clone()),
            ForSourceKind::Comprehension(_) => None,
        };
        if let Some(name) = producer
            && !ctx.producers.contains(&name)
        {
            let p = compiler
                .producers_seen
                .iter()
                .rev()
                .find(|p| p.name == name)
                .cloned()
                .ok_or_else(|| format!("tile '{}': producer '{name}' is not bound before this tile", self.tile_name))?;
            for r in placeholder_names(&p.source_text) {
                self.cascade_ref(asm, r, ctx);
            }
            ctx.producers.push(name.clone());
            ctx.bindings.push(format!("{name} := for {}", p.source_text));
        }
        for r in placeholder_names(&source.text) {
            self.cascade_ref(asm, r, ctx);
        }
        // Every outer wire the nested body reads must reach the body
        // program; the nested tile then cascades it one level further.
        let mut refs = BTreeSet::new();
        collect_piece_refs(body, &compiler.producers_seen, &mut refs);
        for r in refs {
            if nested_elements.iter().any(|(n, _)| *n == r) {
                continue;
            }
            self.cascade_ref(asm, r, ctx);
        }
        let piece = TilePiece::Projection { source: source.clone(), sep: sep.clone(), body: body.to_vec(), span: self.span };
        let text = render_template(std::slice::from_ref(&piece), &self.options);
        let name = format!("__b{}", ctx.counter);
        ctx.counter += 1;
        let defaults = TileOptions::default();
        let mut opts = Vec::new();
        if self.options.open != defaults.open || self.options.close != defaults.close {
            opts.push(format!("delims \"{}\" \"{}\"", escape_polydat_string(&self.options.open), escape_polydat_string(&self.options.close)));
        }
        if self.options.sigil != defaults.sigil {
            opts.push(format!("sigil \"{}\"", escape_polydat_string(&self.options.sigil)));
        }
        if self.strict {
            opts.push("strict".to_string());
        }
        let opts = if opts.is_empty() { String::new() } else { format!(" ({})", opts.join(", ")) };
        ctx.bindings.push(format!("tile {name} : {}{opts} := \"{}\"", self.encoding, escape_polydat_string(&text)));
        Ok(TileOp::Hole(HoleSource::Child(name)))
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

/// Counts over a skeleton for `explain tiles`.
#[derive(Default)]
struct SkeletonShape {
    statics: usize,
    static_bytes: usize,
    holes: usize,
    branches: usize,
    projections: usize,
}

impl SkeletonShape {
    fn count(&mut self, ops: &[TileOp]) {
        for op in ops {
            match op {
                TileOp::Static(s) => {
                    self.statics += 1;
                    self.static_bytes += s.len();
                }
                TileOp::Hole(_) => self.holes += 1,
                TileOp::Branch { then, otherwise, .. } => {
                    self.branches += 1;
                    self.count(then);
                    self.count(otherwise);
                }
                TileOp::Repeat { body, .. } => {
                    self.projections += 1;
                    self.count(body);
                }
            }
        }
    }
}

/// Wire names a run of pieces references: hole expressions, branch
/// conditions, `{name}` placeholders in nested projection sources, and
/// the wires their generator expressions read.
fn collect_piece_refs(pieces: &[TilePiece], producers: &[super::traversal::Producer], out: &mut BTreeSet<String>) {
    for p in pieces {
        match p {
            TilePiece::Static(_) => {}
            TilePiece::Hole(h) => collect_expr_refs(&h.expr, out),
            TilePiece::Branch { cond, then, otherwise, .. } => {
                collect_expr_refs(cond, out);
                collect_piece_refs(then, producers, out);
                if let Some(o) = otherwise {
                    collect_piece_refs(o, producers, out);
                }
            }
            TilePiece::Projection { source, body, .. } => {
                out.extend(placeholder_names(&source.text));
                if let ForSourceKind::Producer(n) | ForSourceKind::Derived { base: n, .. } = &source.kind {
                    out.insert(n.clone());
                }
                if let Ok(c) = super::traversal::resolve_source(source, producers) {
                    for (_, expr_text) in generator_clauses(&c) {
                        if let Ok(expr) = super::tile::parse_hole_expr(&expr_text) {
                            collect_expr_refs(&expr, out);
                        }
                    }
                }
                collect_piece_refs(body, producers, out);
            }
        }
    }
}

/// `(element, expression)` for each generator-call clause.
fn generator_clauses(c: &crate::iteration::comprehension::Comprehension) -> Vec<(String, String)> {
    use crate::iteration::comprehension::source::Source;
    use crate::iteration::comprehension::Comprehension as K;
    let mut out = Vec::new();
    match c {
        K::Clause { name, source: Source::Generator { expr, .. } } => out.push((name.clone(), expr.clone())),
        K::Clause { .. } => {}
        K::Cartesian { children } | K::Zip { children, .. } | K::Union { children } => {
            for ch in children {
                out.extend(generator_clauses(ch));
            }
        }
        K::Filter { child, .. } | K::Order { child, .. } => out.extend(generator_clauses(child)),
    }
    out
}

/// Every filter predicate in a comprehension.
fn filter_predicates(c: &crate::iteration::comprehension::Comprehension) -> Vec<String> {
    use crate::iteration::comprehension::Comprehension as K;
    let mut out = Vec::new();
    match c {
        K::Clause { .. } => {}
        K::Cartesian { children } | K::Zip { children, .. } | K::Union { children } => {
            for ch in children {
                out.extend(filter_predicates(ch));
            }
        }
        K::Filter { child, predicate } => {
            out.push(predicate.clone());
            out.extend(filter_predicates(child));
        }
        K::Order { child, .. } => out.extend(filter_predicates(child)),
    }
    out
}

/// Whether a comprehension draws on a generator call or a workload
/// parameter list that carries no cardinality hint.
fn has_unhinted_source(c: &crate::iteration::comprehension::Comprehension) -> bool {
    use crate::iteration::comprehension::source::Source;
    use crate::iteration::comprehension::Comprehension as K;
    match c {
        K::Clause { source, .. } => matches!(
            source,
            Source::Generator { cardinality_hint: None, .. } | Source::WorkloadParamList { len_hint: None, .. }
        ),
        K::Cartesian { children } | K::Zip { children, .. } | K::Union { children } => {
            children.iter().any(has_unhinted_source)
        }
        K::Filter { child, .. } | K::Order { child, .. } => has_unhinted_source(child),
    }
}

/// `{name}` placeholders in comprehension text.
fn placeholder_names(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let chars: Vec<char> = text.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        if chars[i] == '{' {
            let start = i + 1;
            let mut j = start;
            while j < chars.len() && (chars[j].is_ascii_alphanumeric() || chars[j] == '_') {
                j += 1;
            }
            if j > start && chars.get(j) == Some(&'}') {
                out.push(chars[start..j].iter().collect());
                i = j;
            }
        }
        i += 1;
    }
    out
}

/// Escape text for a Polydat string literal.
fn escape_polydat_string(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"").replace('\n', "\\n").replace('\t', "\\t")
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

