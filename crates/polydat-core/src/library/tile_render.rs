// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! The tile nodes (SRD 114 §6, §7.1).
//!
//! The compiler lowers a `tile` statement to one `tile_render` binding
//! over the hole wires: the render node carries each hole's encoding
//! spec and encodes the value at the hole, concatenates static runs
//! with encoded holes, selects branches, and re-runs projection bodies
//! per tuple over a scratch state, using a skeleton it parses once at
//! setup.
//!
//! `tile_encode` encodes one value under one hole's spec. The compiler
//! emitted one per hole until encoding moved into the renderer; it is
//! a library node now, which a program may call by name and nothing
//! generates.

use std::collections::HashMap;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::ast::SlotShape;
use crate::ast::{PortType, Value, ValueRef};
use crate::iteration::comprehension::StreamerValue;
use crate::iteration::comprehension::runtime::{RuntimeTuple, evaluate_for_iteration};
use crate::kernel::{Kernel, KernelProgram, PolydatKernel, PolydatProgram};
use crate::library::support::float_text;

/// Where a hole sits in a `json` skeleton, which decides its encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HolePosition {
    /// A JSON value position: numbers bare, strings quoted.
    Value,
    /// Inside a JSON string literal: escaped text only.
    InString,
    /// Plain text (text and csv encodings).
    Text,
}

/// The encoder for one hole: what the renderer applies at the hole,
/// and what the `tile_encode` node takes as its spec.
/// Serialized compactly as `encoding|position|type|format|flags`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HoleEncoding {
    /// The tile's encoding.
    pub encoding: String,
    /// Where in the output the hole sits: a value, inside a string, or text.
    pub position: HolePosition,
    /// The declared type, if any.
    pub ty: Option<String>,
    /// The format, if any.
    pub format: Option<String>,
    /// Whether the value is emitted without the encoding's escaping.
    pub raw: bool,
    /// Encode as a branch condition: `1` or `0`.
    pub cond: bool,
}

impl HoleEncoding {
    /// The compact spec form, `encoding|position|type|format|flags`.
    pub fn to_spec(&self) -> String {
        let pos = match self.position {
            HolePosition::Value => "value",
            HolePosition::InString => "string",
            HolePosition::Text => "text",
        };
        let mut flags = String::new();
        if self.raw {
            flags.push('r');
        }
        if self.cond {
            flags.push('c');
        }
        format!(
            "{}|{}|{}|{}|{}",
            self.encoding,
            pos,
            self.ty.as_deref().unwrap_or(""),
            self.format.as_deref().unwrap_or(""),
            flags
        )
    }

    /// The encoding a spec names; a missing part takes its default.
    pub fn from_spec(spec: &str) -> Self {
        let mut parts = spec.splitn(5, '|');
        let encoding = parts.next().unwrap_or("text").to_string();
        let position = match parts.next().unwrap_or("text") {
            "value" => HolePosition::Value,
            "string" => HolePosition::InString,
            _ => HolePosition::Text,
        };
        let ty = parts.next().filter(|s| !s.is_empty()).map(str::to_string);
        let format = parts.next().filter(|s| !s.is_empty()).map(str::to_string);
        let flags = parts.next().unwrap_or("");
        HoleEncoding {
            encoding,
            position,
            ty,
            format,
            raw: flags.contains('r'),
            cond: flags.contains('c'),
        }
    }
}

/// Where a hole's encoded text comes from at render time.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum HoleSource {
    /// The render node's wire input at this index, encoded per `spec`
    /// (a [`HoleEncoding`] spec) as it is rendered.
    Wire {
        /// The input's index among the render node's wires.
        index: usize,
        /// The hole's encoding spec.
        spec: String,
    },
    /// An output of the enclosing projection's body program, encoded
    /// per `spec` as it is rendered.
    Child {
        /// The body program's output.
        name: String,
        /// The hole's encoding spec.
        spec: String,
    },
}

/// One skeleton instruction. Holes are values, encoded at the hole.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TileOp {
    /// Text copied as is.
    Static(String),
    /// A hole, already encoded.
    Hole(HoleSource),
    /// A projection: the body rendered once per tuple.
    Repeat {
        /// The comprehension, as a serialized [`StreamerValue`].
        stream: String,
        /// Index into [`TileSpec::children`].
        child: usize,
        /// The separator between tuples.
        sep: String,
        /// The body skeleton.
        body: Vec<TileOp>,
        /// Generator-call clauses whose expressions compiled to wires of
        /// the enclosing program: `(element, node input index, type)`.
        /// At render the input's value stands in for the clause.
        #[serde(default)]
        generators: Vec<(String, usize, String)>,
    },
    /// A branch on a condition hole.
    Branch {
        /// The condition, encoded as `1` or `0`.
        cond: HoleSource,
        /// The skeleton when the condition holds.
        then: Vec<TileOp>,
        /// The skeleton otherwise.
        otherwise: Vec<TileOp>,
    },
}

/// A projection body: a program compiled once at setup, and the outer
/// wires it imports from the render node's inputs, each with the type
/// its extern declares so the transported text can be re-typed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChildSpec {
    /// The body program's source.
    pub source: String,
    /// `(extern name, render-node input index, port-type keyword)`.
    pub cascade: Vec<(String, usize, String)>,
}

/// The serialized skeleton.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TileSpec {
    /// The tile's name.
    pub name: String,
    /// The tile's encoding.
    pub encoding: String,
    /// The skeleton.
    pub ops: Vec<TileOp>,
    /// The projection body programs, by index.
    pub children: Vec<ChildSpec>,
}

impl TileSpec {
    /// The spec as JSON, the form the `tile_render` node's spec argument carries.
    pub fn to_json(&self) -> String {
        serde_json::to_string(self).expect("TileSpec serializes")
    }
}

/// One skeleton instruction in its runtime form (SRD 114 §6): static
/// runs are interned once at build and copied from the interner, the
/// comprehension of a projection is parsed once, and separators are
/// interned too.
#[derive(Debug)]
enum RtOp {
    /// Copy an interned static run.
    Copy(&'static str),
    /// Encode a value at the hole.
    Hole(RtSource, HoleEncoding),
    Repeat {
        stream: Arc<StreamerValue>,
        child: usize,
        sep: &'static str,
        body: Vec<RtOp>,
        generators: Vec<(String, usize, String)>,
    },
    Branch {
        cond: RtSource,
        then: Vec<RtOp>,
        otherwise: Vec<RtOp>,
    },
}

/// Where a hole's value comes from at render time.
#[derive(Debug)]
enum RtSource {
    Wire(usize),
    /// A body output and its ordinal among the body's holes, which a
    /// body kernel's entry resolves to an output index once.
    Child(String, usize),
}

fn lower_source(source: &HoleSource) -> (RtSource, HoleEncoding) {
    match source {
        HoleSource::Wire { index, spec } => (RtSource::Wire(*index), HoleEncoding::from_spec(spec)),
        HoleSource::Child { name, spec } => (
            RtSource::Child(name.clone(), 0),
            HoleEncoding::from_spec(spec),
        ),
    }
}

/// Intern every static run and separator of a skeleton and parse every
/// projection stream, once, at construction.
fn lower_ops(ops: &[TileOp]) -> Vec<RtOp> {
    use crate::kernel::StaticInterner;
    ops.iter()
        .map(|op| match op {
            TileOp::Static(s) => RtOp::Copy(StaticInterner::intern(s)),
            TileOp::Hole(h) => {
                let (source, enc) = lower_source(h);
                RtOp::Hole(source, enc)
            }
            TileOp::Repeat {
                stream,
                child,
                sep,
                body,
                generators,
            } => RtOp::Repeat {
                stream: Arc::new(StreamerValue::from_json(stream)),
                child: *child,
                sep: StaticInterner::intern(sep),
                body: lower_ops(body),
                generators: generators.clone(),
            },
            TileOp::Branch {
                cond,
                then,
                otherwise,
            } => RtOp::Branch {
                cond: lower_source(cond).0,
                then: lower_ops(then),
                otherwise: lower_ops(otherwise),
            },
        })
        .collect()
}

/// The runtime form: the spec, compiled body programs, and parsed streams.
pub struct TileProgram {
    /// The serialized skeleton.
    pub spec: TileSpec,
    /// The skeleton with statics interned and streams parsed.
    ops: Vec<RtOp>,
    /// Each projection body, as the one carrier a `for` body uses:
    /// its statements, the settings it compiles under, and its
    /// program per engine, built on first use.
    ///
    /// This used to be two eager compiles per body — one interpreter
    /// program and one on `Engine::default()` — made at node
    /// construction whether or not either engine ever rendered it.
    pub bodies: Vec<Arc<crate::dsl::traversal::BodySource>>,
    /// The body programs for the interpreter, which construction
    /// needs: the canonical kernels are built over them, and
    /// memoisation runs against those.
    pub children: Vec<Arc<PolydatProgram>>,
    /// One kernel over each body program, the canonical kernel the
    /// comprehension evaluator installs tuple values into.
    canonicals: Vec<Arc<PolydatKernel>>,
    /// Per body, its projection's tuples when the comprehension is the
    /// same every render: no generator clause and no placeholder in
    /// its sources. Evaluated once at construction.
    memo: Vec<Option<Arc<[RuntimeTuple]>>>,
}

impl std::fmt::Debug for TileProgram {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TileProgram")
            .field("spec", &self.spec)
            .field("ops", &self.ops)
            .field("children", &self.children.len())
            .finish_non_exhaustive()
    }
}

/// Give every body hole an ordinal within its body, so a body kernel's
/// entry can resolve the hole's output index once and keep it by
/// position (SRD 117 step 3).
fn number_child_holes(ops: &mut [RtOp]) {
    fn walk(ops: &mut [RtOp], next: &mut usize) {
        for op in ops.iter_mut() {
            match op {
                RtOp::Hole(RtSource::Child(_, k), _) => {
                    *k = *next;
                    *next += 1;
                }
                RtOp::Branch {
                    cond,
                    then,
                    otherwise,
                } => {
                    if let RtSource::Child(_, k) = cond {
                        *k = *next;
                        *next += 1;
                    }
                    walk(then, next);
                    walk(otherwise, next);
                }
                RtOp::Repeat { body, .. } => {
                    let mut inner = 0;
                    walk(body, &mut inner);
                }
                _ => {}
            }
        }
    }
    let mut top = 0;
    walk(ops, &mut top);
}

/// Evaluate every projection whose tuples cannot change between
/// renders, once.
fn memoize(
    ops: &[RtOp],
    canonicals: &[Arc<PolydatKernel>],
    memo: &mut [Option<Arc<[RuntimeTuple]>>],
) {
    for op in ops {
        match op {
            RtOp::Repeat {
                stream,
                child,
                body,
                generators,
                ..
            } => {
                if generators.is_empty()
                    && !stream.text.contains('{')
                    && let Ok(tuples) = evaluate_for_iteration(
                        &stream.ast,
                        &*canonicals[*child],
                        &HashMap::new(),
                        |_| Ok(()),
                    )
                {
                    memo[*child] = Some(tuples.into());
                }
                memoize(body, canonicals, memo);
            }
            RtOp::Branch {
                then, otherwise, ..
            } => {
                memoize(then, canonicals, memo);
                memoize(otherwise, canonicals, memo);
            }
            _ => {}
        }
    }
}

impl TileProgram {
    /// A program from a skeleton and the bodies its projections run.
    ///
    /// The compiler's path: it lowered the bodies, so it hands them
    /// over as they are. Each body carries the settings the parent
    /// compiled under, and its program per engine is built when a
    /// render on that engine first asks for it.
    pub fn from_parts(
        spec: TileSpec,
        bodies: Vec<Arc<crate::dsl::traversal::BodySource>>,
    ) -> Result<Self, String> {
        // Construction needs the interpreter's program: the canonical
        // kernel the comprehension evaluator installs tuples into is
        // built over it, and memoisation runs there. Every other
        // engine's is built on the first render that asks for it.
        let mut children: Vec<Arc<PolydatProgram>> = Vec::with_capacity(bodies.len());
        for (i, body) in bodies.iter().enumerate() {
            let program = body
                .program_on(crate::Engine::Interpreter(crate::JitMode::Auto))
                .map_err(|e| format!("projection body {i} failed to compile: {e}"))?;
            children.push(
                program
                    .as_interpreter()
                    .ok_or_else(|| format!("projection body {i} is not an interpreter program"))?,
            );
        }
        let canonicals: Vec<Arc<PolydatKernel>> = children
            .iter()
            .map(|p| Arc::new(PolydatKernel::from_program(p.clone())))
            .collect();
        let mut ops = lower_ops(&spec.ops);
        number_child_holes(&mut ops);
        let mut memo = vec![None; children.len()];
        memoize(&ops, &canonicals, &mut memo);
        Ok(TileProgram {
            spec,
            ops,
            bodies,
            children,
            canonicals,
            memo,
        })
    }

    /// A program from a serialized skeleton, whose projection bodies
    /// are rebuilt from the source text it carries under the default
    /// settings.
    ///
    /// The route for a skeleton that reaches the runtime as text — a
    /// host that stored one, a test that wrote one. The compiler's own
    /// path is [`Self::from_parts`], which hands the bodies over
    /// rather than describing them.
    pub fn from_json(json: &str) -> Self {
        let spec: TileSpec = serde_json::from_str(json)
            .unwrap_or_else(|e| panic!("tile_render: malformed skeleton payload: {e}"));
        let name = spec.name.clone();
        let bodies: Vec<Arc<crate::dsl::traversal::BodySource>> = spec
            .children
            .iter()
            .enumerate()
            .map(|(i, c)| {
                Arc::new(
                    crate::dsl::traversal::BodySource::from_source(
                        &c.source,
                        &format!("tile '{name}' :: projection body {i}"),
                    )
                    .unwrap_or_else(|e| {
                        panic!(
                            "tile '{name}': projection body failed to parse: {e}\n{}",
                            c.source
                        )
                    }),
                )
            })
            .collect();
        Self::from_parts(spec, bodies).unwrap_or_else(|e| panic!("tile '{name}': {e}"))
    }

    /// The body program of projection `child` for a render on
    /// `engine` — the engine the kernel doing the rendering runs on,
    /// which is the rule a `for` body already followed.
    ///
    /// A tile's body used to render on `Engine::default()` whatever
    /// engine the kernel was, because both of its programs were built
    /// at construction and the default one was the only compiled
    /// program there was. An engine that refuses the body falls back
    /// to the interpreter's, as before, and says so once.
    fn body_program_on(&self, child: usize, engine: crate::Engine) -> Arc<dyn KernelProgram> {
        if matches!(engine, crate::Engine::Interpreter(_)) {
            return self.children[child].clone();
        }
        self.bodies[child].program_on(engine).unwrap_or_else(|e| {
            crate::library::support::audit::debug(&format!(
                "tile '{}': projection body {child} renders on the interpreter: {e}",
                self.spec.name
            ));
            self.children[child].clone()
        })
    }

    /// True when any op re-runs a projection body.
    pub fn has_projections(&self) -> bool {
        fn walk(ops: &[RtOp]) -> bool {
            ops.iter().any(|op| match op {
                RtOp::Repeat { .. } => true,
                RtOp::Branch {
                    then, otherwise, ..
                } => walk(then) || walk(otherwise),
                _ => false,
            })
        }
        walk(&self.ops)
    }

    /// Render with the node's wire inputs, the hole values, on the
    /// interpreter, over `bodies`, the rendering state's own kernels
    /// for the projection bodies.
    pub fn render(&self, inputs: &[Value], bodies: &mut BodyKernels) -> String {
        let refs: Vec<ValueRef<'_>> = inputs.iter().map(ValueRef::from).collect();
        let mut out = String::new();
        self.render_into(
            &refs,
            crate::Engine::Interpreter(crate::JitMode::Auto),
            bodies,
            &mut out,
        );
        out
    }

    /// Render into any text sink from borrowed views of the hole
    /// values: a `String` at P1, a step's own scratch in a compiled
    /// closure. Every hole is encoded here, from the view straight into
    /// the sink, and a projection's body runs on `engine`, the engine
    /// of the kernel rendering, in a kernel the rendering state owns
    /// (`bodies`) and reuses across renders.
    pub fn render_into<W: std::fmt::Write>(
        &self,
        inputs: &[ValueRef<'_>],
        engine: crate::Engine,
        bodies: &mut BodyKernels,
        out: &mut W,
    ) {
        self.render_ops(&self.ops, inputs, engine, bodies, None, out);
    }

    fn render_ops<W: std::fmt::Write>(
        &self,
        ops: &[RtOp],
        inputs: &[ValueRef<'_>],
        engine: crate::Engine,
        bodies: &mut BodyKernels,
        mut child: Option<&mut BodyEntry>,
        out: &mut W,
    ) {
        for op in ops {
            match op {
                // `Copy`: a memcpy from the static interner (SRD 114 §6,
                // SRD 115 step 3). The bytes were interned at build.
                RtOp::Copy(s) => out.put(s),
                RtOp::Hole(source, enc) => match source {
                    RtSource::Wire(i) => {
                        encode_ref(inputs.get(*i).copied().unwrap_or(ValueRef::None), enc, out)
                    }
                    RtSource::Child(name, k) => {
                        if let Some(entry) = child.as_deref_mut()
                            && let Some(i) = entry.hole(*k, name)
                        {
                            let v = entry.kernel.pull_at(i);
                            encode_ref(ValueRef::from(&v), enc, out)
                        }
                    }
                },
                RtOp::Branch {
                    cond,
                    then,
                    otherwise,
                } => {
                    let c = self.truthy(cond, inputs, child.as_deref_mut());
                    let branch = if c { then } else { otherwise };
                    self.render_ops(branch, inputs, engine, bodies, child.as_deref_mut(), out);
                }
                RtOp::Repeat {
                    stream,
                    child: child_idx,
                    sep,
                    body,
                    generators,
                } => {
                    // The tuples: memoized when the comprehension is the
                    // same every render, otherwise evaluated now with the
                    // same evaluator the `for` construct opens a traversal
                    // with, which applies order strategies, samples
                    // continuous sources, and runs predicates over the
                    // tuple. Generators are bound first, so the parent
                    // kernel it sees is empty and the canonical kernel is
                    // the body program.
                    let memoized = self.memo[*child_idx].clone();
                    let tuples: std::borrow::Cow<'_, [RuntimeTuple]> = match &memoized {
                        Some(t) => std::borrow::Cow::Borrowed(&t[..]),
                        None => {
                            let mut streamer = (**stream).clone();
                            if !generators.is_empty() {
                                streamer.ast = bind_generators(&streamer.ast, generators, inputs);
                            }
                            std::borrow::Cow::Owned(
                                evaluate_for_iteration(
                                    &streamer.ast,
                                    &*self.canonicals[*child_idx],
                                    &HashMap::new(),
                                    |_| Ok(()),
                                )
                                .unwrap_or_else(|e| {
                                    panic!(
                                        "tile '{}': projection `for {}` failed at render: {e}",
                                        self.spec.name, streamer.text
                                    )
                                }),
                            )
                        }
                    };
                    let child_spec = &self.spec.children[*child_idx];
                    // The body runs on the engine the kernel rendering
                    // it runs on, which is the rule a `for` body
                    // already followed. The kernel over its program is
                    // owned by the rendering state and reused across
                    // its renders.
                    let program = self.body_program_on(*child_idx, engine);
                    let mut first = true;
                    let fail = |name: &str, e: crate::kernel::WriteError| -> ! {
                        panic!(
                            "tile '{}': projection body input `{name}`: {e}",
                            self.spec.name
                        )
                    };
                    bodies.with(&program, engine, |entry, bodies| {
                        for (index, tuple) in tuples.iter().enumerate() {
                            if !first {
                                out.put(sep);
                            }
                            first = false;
                            {
                                // The body's inputs by index: the names are
                                // resolved on the first tuple and kept.
                                let BodyEntry {
                                    kernel,
                                    elements,
                                    cascade,
                                    ..
                                } = &mut *entry;
                                kernel.set_inputs(&[index as u64]);
                                let elements = elements.get_or_insert_with(|| {
                                    tuple.iter().map(|(n, _)| kernel.input_index(n)).collect()
                                });
                                for (k, (name, v)) in tuple.iter().enumerate() {
                                    if let Some(i) = elements.get(k).copied().flatten() {
                                        kernel
                                            .set_input_at(i, v.clone())
                                            .unwrap_or_else(|e| fail(name, e));
                                    }
                                }
                                let cascade = cascade.get_or_insert_with(|| {
                                    child_spec
                                        .cascade
                                        .iter()
                                        .map(|(n, _, _)| kernel.input_index(n))
                                        .collect()
                                });
                                for (k, (name, input_idx, ty)) in
                                    child_spec.cascade.iter().enumerate()
                                {
                                    if let Some(i) = cascade.get(k).copied().flatten()
                                        && let Some(v) = inputs.get(*input_idx)
                                    {
                                        kernel
                                            .set_input_at(i, typed_for(&owned(*v), ty))
                                            .unwrap_or_else(|e| fail(name, e));
                                    }
                                }
                            }
                            self.render_ops(body, inputs, engine, bodies, Some(entry), out);
                        }
                    });
                }
            }
        }
    }

    /// A branch condition's truth, as the `cond` encoding decides it.
    fn truthy(
        &self,
        source: &RtSource,
        inputs: &[ValueRef<'_>],
        child: Option<&mut BodyEntry>,
    ) -> bool {
        match source {
            RtSource::Wire(i) => truthy_of(inputs.get(*i).copied().unwrap_or(ValueRef::None)),
            RtSource::Child(name, k) => match child {
                Some(entry) => match entry.hole(*k, name) {
                    Some(i) => truthy_of(ValueRef::from(&entry.kernel.pull_at(i))),
                    None => false,
                },
                None => false,
            },
        }
    }
}

/// A borrowed view as an owned value, for the paths that bind values
/// into a body program or a comprehension.
fn owned(v: ValueRef<'_>) -> Value {
    match v {
        ValueRef::U64(n) => Value::U64(n),
        ValueRef::I64(n) => Value::I64(n),
        ValueRef::F64(f) => Value::F64(f),
        ValueRef::Bool(b) => Value::Bool(b),
        ValueRef::Str(s) => Value::Str(Arc::from(s)),
        ValueRef::Bytes(b) => Value::Bytes(Arc::from(b)),
        ValueRef::Json(j) => Value::Json(Arc::new(j.clone())),
        ValueRef::None => Value::None,
        ValueRef::Other(v) => v.clone(),
    }
}

/// A cascaded value as the body's extern expects it. Values arrive on
/// the render node's inputs as they are, so this is the value itself;
/// text is parsed only when a `Str` reaches a non-string extern.
fn typed_for(v: &Value, ty: &str) -> Value {
    match (v, PortType::from_keyword(ty)) {
        (Value::Str(_), Some(t)) if t != PortType::Str => retype(v, ty),
        _ => v.clone(),
    }
}

/// Replace each generator-call clause with the literal values its wire
/// carries at this render: a list value (a stream, a vector, a JSON
/// array) contributes its items, a scalar contributes itself. Text that
/// spells a JSON array is read as one.
fn bind_generators(
    c: &crate::iteration::comprehension::Comprehension,
    generators: &[(String, usize, String)],
    inputs: &[ValueRef<'_>],
) -> crate::iteration::comprehension::Comprehension {
    use crate::iteration::comprehension::Comprehension as K;
    use crate::iteration::comprehension::source::{LiteralValue, Source};
    match c {
        K::Clause {
            name,
            source: Source::Generator { .. },
        } => {
            let Some((_, idx, ty)) = generators.iter().find(|(n, _, _)| n == name) else {
                return c.clone();
            };
            let raw = inputs.get(*idx).map(|v| owned(*v)).unwrap_or(Value::None);
            let items: Vec<Value> =
                match crate::iteration::comprehension::source_values::iteration_interior(&raw) {
                    Some(interior) => interior,
                    None => match &raw {
                        Value::Str(text) => {
                            match serde_json::from_str::<serde_json::Value>(text.trim()) {
                                Ok(serde_json::Value::Array(items)) => items
                                    .iter()
                                    .map(|j| {
                                        retype(
                                            &Value::Str(j.to_string().trim_matches('"').into()),
                                            ty,
                                        )
                                    })
                                    .collect(),
                                _ => vec![typed_for(&raw, ty)],
                            }
                        }
                        _ => vec![raw.clone()],
                    },
                };
            // An element declared `json` takes every item as the JSON
            // value it is, its kind kept, so the body's extern receives
            // what it declares; another declared type takes the scalar.
            let json_items = ty == "json";
            let values = items
                .iter()
                .map(|v| {
                    if json_items {
                        return LiteralValue::Json(json_of(v));
                    }
                    match v {
                        Value::U64(n) => LiteralValue::Int(*n as i64),
                        Value::I64(n) => LiteralValue::Int(*n),
                        Value::F64(f) => LiteralValue::Float(*f),
                        Value::Bool(b) => LiteralValue::Bool(*b),
                        // JSON scalars carry their own kind.
                        Value::Json(j) => match j.as_ref() {
                            serde_json::Value::Number(n) if n.is_u64() => {
                                LiteralValue::Int(n.as_u64().unwrap_or(0) as i64)
                            }
                            serde_json::Value::Number(n) if n.is_i64() => {
                                LiteralValue::Int(n.as_i64().unwrap_or(0))
                            }
                            serde_json::Value::Number(n) => {
                                LiteralValue::Float(n.as_f64().unwrap_or(0.0))
                            }
                            serde_json::Value::Bool(b) => LiteralValue::Bool(*b),
                            serde_json::Value::String(s) => LiteralValue::String(s.clone()),
                            other => LiteralValue::String(other.to_string()),
                        },
                        other => LiteralValue::String(other.to_display_string()),
                    }
                })
                .collect();
            K::Clause {
                name: name.clone(),
                source: Source::Literal { values },
            }
        }
        K::Clause { .. } => c.clone(),
        K::Cartesian { children } => K::Cartesian {
            children: children
                .iter()
                .map(|ch| bind_generators(ch, generators, inputs))
                .collect(),
        },
        K::Zip { children, mode } => K::Zip {
            children: children
                .iter()
                .map(|ch| bind_generators(ch, generators, inputs))
                .collect(),
            mode: *mode,
        },
        K::Union { children } => K::Union {
            children: children
                .iter()
                .map(|ch| bind_generators(ch, generators, inputs))
                .collect(),
        },
        K::Filter { child, predicate } => K::Filter {
            child: Box::new(bind_generators(child, generators, inputs)),
            predicate: predicate.clone(),
        },
        K::Order {
            child,
            strategy,
            truncation,
            seed,
        } => K::Order {
            child: Box::new(bind_generators(child, generators, inputs)),
            strategy: *strategy,
            truncation: *truncation,
            seed: *seed,
        },
    }
}

/// Recover a typed value from the display text a cascaded wire arrives
/// as, using the child extern's declared type.
/// A generator item as a JSON value: a JSON item as it is, a scalar as
/// the JSON of its kind.
fn json_of(v: &Value) -> serde_json::Value {
    match v {
        Value::Json(j) => j.as_ref().clone(),
        Value::U64(n) => serde_json::Value::from(*n),
        Value::I64(n) => serde_json::Value::from(*n),
        Value::F64(f) => serde_json::Number::from_f64(*f)
            .map(serde_json::Value::Number)
            .unwrap_or(serde_json::Value::Null),
        Value::Bool(b) => serde_json::Value::Bool(*b),
        Value::Str(s) => serde_json::Value::String(s.to_string()),
        Value::None => serde_json::Value::Null,
        other => serde_json::Value::String(other.to_display_string()),
    }
}

fn retype(v: &Value, ty: &str) -> Value {
    let text = v.to_display_string();
    match PortType::from_keyword(ty) {
        Some(PortType::U64) => text.parse().map(Value::U64).unwrap_or(Value::None),
        Some(PortType::F64) => text.parse().map(Value::F64).unwrap_or(Value::None),
        Some(PortType::Bool) => Value::Bool(matches!(text.trim(), "true" | "1")),
        Some(PortType::Str) | None => Value::Str(text.into()),
        Some(_) => v.clone(),
    }
}

/// A cached body kernel: the program it was created from, the kernel,
/// and the body's names resolved to indices once (SRD 117 step 3), so
/// a tuple is bound and its holes read with no lookup per tuple.
struct BodyEntry {
    program: Arc<dyn KernelProgram>,
    kernel: Box<dyn Kernel>,
    /// The tuple elements' input indices, by position in the tuple;
    /// `None` for an element the body does not declare.
    elements: Option<Vec<Option<usize>>>,
    /// The cascade's input indices, by position in the cascade.
    cascade: Option<Vec<Option<usize>>>,
    /// The body holes' output indices, by ordinal.
    holes: Vec<Option<Option<usize>>>,
}

impl BodyEntry {
    /// The output index of body hole `k`, named `name`, resolved once.
    fn hole(&mut self, k: usize, name: &str) -> Option<usize> {
        if self.holes.len() <= k {
            self.holes.resize(k + 1, None);
        }
        if self.holes[k].is_none() {
            self.holes[k] = Some(self.kernel.output_index(name));
        }
        self.holes[k].flatten()
    }
}

/// The kernels one rendering state keeps over its projection bodies:
/// one per body program and engine, created on the first render that
/// reaches the body and reused by every render after, so a projection
/// creates nothing per tuple. A tile render node owns one of these in
/// its scratch (axiom S3): the storage belongs to the state that
/// renders, never to the node, which every state of the program
/// shares. A clone is empty, since a clone of a state is a new state.
#[derive(Default)]
pub struct BodyKernels {
    entries: HashMap<(usize, crate::Engine), BodyEntry>,
    /// Kernels created so far, for the tests.
    created: u64,
}

impl Clone for BodyKernels {
    fn clone(&self) -> Self {
        Self::default()
    }
}

// SAFETY: the kernels are reached only through `&mut self` (`with`),
// which the owning state holds exclusively; every `&self` method
// (`created`, `clone`, `Debug`) reads a count and touches no kernel. A
// set inside a program shared across threads is therefore never used
// from more than one thread, and a state created from that program
// starts with an empty set of its own.
unsafe impl Sync for BodyKernels {}

impl std::fmt::Debug for BodyKernels {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BodyKernels")
            .field("entries", &self.entries.len())
            .field("created", &self.created)
            .finish()
    }
}

/// The engine a body kernel is kept under: the interpreter's body
/// program is one program whatever the enclosing kernel's cone mode.
fn body_engine_key(engine: crate::Engine) -> crate::Engine {
    match engine {
        crate::Engine::Interpreter(_) => crate::Engine::Interpreter(crate::JitMode::Auto),
        other => other,
    }
}

impl BodyKernels {
    /// Kernels this state has created so far.
    pub fn created(&self) -> u64 {
        self.created
    }

    /// The count and a clone, for the unit test of both.
    #[cfg(test)]
    fn clone_for_test(&self) -> (u64, BodyKernels) {
        (self.created, self.clone())
    }

    /// Run `f` over the kernel for `program` on `engine`, created on
    /// first use. The entry is taken out for the call, so a nested
    /// projection's body finds the set free for its own kernels.
    fn with(
        &mut self,
        program: &Arc<dyn KernelProgram>,
        engine: crate::Engine,
        f: impl FnOnce(&mut BodyEntry, &mut BodyKernels),
    ) {
        let engine = body_engine_key(engine);
        // The entry pins its program so the address cannot be reused by
        // a later program while a kernel built for this one is kept.
        let key = (Arc::as_ptr(program) as *const () as usize, engine);
        let mut entry = self
            .entries
            .remove(&key)
            .filter(|e| Arc::ptr_eq(&e.program, program))
            .unwrap_or_else(|| {
                self.created += 1;
                BodyEntry {
                    program: program.clone(),
                    kernel: program.clone().create_kernel(),
                    elements: None,
                    cascade: None,
                    holes: Vec::new(),
                }
            });
        f(&mut entry, self);
        self.entries.insert(key, entry);
    }
}

/// The tile render node's state: its projection bodies' kernels.
pub(crate) mod render_state {
    use super::{BodyKernels, TileRender};
    use crate::ast::{ScratchBuf, ScratchElem, Value};

    pub(crate) fn layout(_node: &TileRender) -> Vec<ScratchElem> {
        vec![ScratchElem::Kernels]
    }

    pub(crate) fn eval(
        node: &TileRender,
        scratch: &mut [ScratchBuf],
        inputs: &[Value],
        outputs: &mut [Value],
    ) {
        let bodies = bodies_of(&mut scratch[0]);
        outputs[0] = Value::Str(node.program.render(inputs, bodies).into());
    }

    /// The body kernel set a scratch entry holds.
    pub(crate) fn bodies_of(entry: &mut ScratchBuf) -> &mut BodyKernels {
        match entry {
            ScratchBuf::Kernels(b) => b,
            other => panic!("a tile render's scratch holds {other:?}, not its body kernels"),
        }
    }
}

/// A text sink that cannot fail: a `String`, or the `BytesSink` over a
/// step's string scratch in the compiled closure. `fmt::Write`'s
/// results are ignored because neither sink reports an error.
pub(crate) trait Sink: std::fmt::Write {
    fn put(&mut self, s: &str) {
        let _ = self.write_str(s);
    }
    fn put_char(&mut self, c: char) {
        let _ = self.write_char(c);
    }
}

impl<W: std::fmt::Write> Sink for W {}

/// Encode one value per a hole's encoding, into any text sink.
pub fn encode<W: std::fmt::Write>(value: &Value, enc: &HoleEncoding, out: &mut W) {
    encode_ref(ValueRef::from(value), enc, out)
}

/// Encode a borrowed view of a value (SRD 115 §6.1): the compiled
/// closure calls this on its slot without owning a `Value`, and a
/// string hole is encoded from its producer's scratch in place.
pub fn encode_ref<W: std::fmt::Write>(value: ValueRef<'_>, enc: &HoleEncoding, out: &mut W) {
    if enc.cond {
        out.put_char(if truthy_of(value) { '1' } else { '0' });
        return;
    }
    let ty = enc.ty.as_deref();
    // A number with no format, or a float under a `.N` precision,
    // writes its digits straight into the sink (SRD 117 step 3):
    // digits, a sign, and a point need no escaping in any encoding or
    // position, and a numeric type is written bare in a JSON value
    // position, so the text is the same as the general path's, without
    // the `String` the general path builds. The float writer is
    // byte-identical to `format!` (`support::float_text`, proved by
    // `tests/float_text.rs`).
    if is_numeric_keyword(ty.unwrap_or("u64")) {
        match (enc.format.as_deref(), value) {
            (None, ValueRef::U64(n)) => {
                put_u64(n, out);
                return;
            }
            (None, ValueRef::I64(n)) => {
                if n < 0 {
                    out.put_char('-');
                }
                put_u64(n.unsigned_abs(), out);
                return;
            }
            (None, ValueRef::F64(f)) => {
                let _ = float_text::write_shortest(f, out);
                return;
            }
            (Some(fmt), ValueRef::F64(_) | ValueRef::U64(_)) => {
                if let (Some(prec), Some(f)) = (precision_of(fmt), as_f64(value)) {
                    let _ = float_text::write_fixed(f, prec, out);
                    return;
                }
            }
            _ => {}
        }
    }
    let text = formatted_text(value, ty, enc.format.as_deref());
    if enc.raw {
        out.put(&text);
        return;
    }
    match (enc.encoding.as_str(), enc.position) {
        ("json", HolePosition::InString) => push_json_escaped(&text, out),
        ("json", HolePosition::Value) => {
            let kind = ty.unwrap_or_else(|| value.port_type().to_keyword());
            match (kind, value) {
                (_, ValueRef::None) => out.put("null"),
                ("bool", _) => out.put(if truthy_of(value) { "true" } else { "false" }),
                ("json", ValueRef::Json(j)) => {
                    let _ = write!(out, "{j}");
                }
                ("str", _) | ("String", _) | ("string", _) => {
                    out.put_char('"');
                    push_json_escaped(&text, out);
                    out.put_char('"');
                }
                (k, _) if is_numeric_keyword(k) => out.put(&text),
                (_, ValueRef::Json(j)) => {
                    let _ = write!(out, "{j}");
                }
                (_, ValueRef::Bool(b)) => out.put(if b { "true" } else { "false" }),
                (_, ValueRef::U64(_)) | (_, ValueRef::F64(_)) => out.put(&text),
                _ => {
                    out.put_char('"');
                    push_json_escaped(&text, out);
                    out.put_char('"');
                }
            }
        }
        ("csv", _) => {
            if text.contains([',', '"', '\n']) {
                out.put_char('"');
                for (i, piece) in text.split('"').enumerate() {
                    if i > 0 {
                        out.put("\"\"");
                    }
                    out.put(piece);
                }
                out.put_char('"');
            } else {
                out.put(&text);
            }
        }
        _ => out.put(&text),
    }
}

/// The decimal digits of `n`, written without an allocation.
fn put_u64<W: std::fmt::Write>(mut n: u64, out: &mut W) {
    if n == 0 {
        out.put_char('0');
        return;
    }
    let mut buf = [0u8; 20];
    let mut i = buf.len();
    while n > 0 {
        i -= 1;
        buf[i] = b'0' + (n % 10) as u8;
        n /= 10;
    }
    // SAFETY-free: the buffer holds ASCII digits only.
    out.put(std::str::from_utf8(&buf[i..]).expect("ascii digits"));
}

fn truthy_of(v: ValueRef<'_>) -> bool {
    match v {
        ValueRef::Bool(b) => b,
        ValueRef::U64(n) => n != 0,
        ValueRef::F64(f) => f != 0.0,
        ValueRef::Str(s) => !s.is_empty() && s != "0" && s != "false",
        ValueRef::None => false,
        _ => true,
    }
}

fn is_numeric_keyword(k: &str) -> bool {
    matches!(
        k,
        "u64"
            | "i64"
            | "f64"
            | "f32"
            | "u32"
            | "i32"
            | "u16"
            | "i16"
            | "u8"
            | "i8"
            | "u128"
            | "i128"
            | "f16"
    )
}

/// Display text for a value under an optional printf-style format:
/// `.N` precision for floats, `0N` zero-padded width, `N` width, `>N`
/// and `<N` alignment, `x`/`X` hex for integers.
fn formatted_text<'a>(
    value: ValueRef<'a>,
    ty: Option<&str>,
    format: Option<&str>,
) -> std::borrow::Cow<'a, str> {
    use std::borrow::Cow;
    // A string with no format is borrowed as it is; everything else is
    // owned text. The base text is produced only where a format needs
    // it: a precision or a hex format writes the number once, itself.
    let base = |value: ValueRef<'a>| -> Cow<'a, str> {
        match (ty, value) {
            (Some("bool"), v) => Cow::Owned(truthy_of(v).to_string()),
            // Text quotes nothing: a JSON string in a text position is
            // its text, as a `str` hole is.
            (_, ValueRef::Json(serde_json::Value::String(s))) => Cow::Owned(s.clone()),
            (_, ValueRef::Json(j)) => Cow::Owned(j.to_string()),
            (_, v) => v.display(),
        }
    };
    let Some(fmt) = format else {
        return base(value);
    };
    let fmt = fmt.trim();
    if let Some(prec) = precision_of(fmt) {
        if let Some(f) = as_f64(value) {
            return Cow::Owned(float_text::fixed_string(f, prec));
        }
        return base(value);
    }
    if fmt == "x" || fmt == "X" {
        if let ValueRef::U64(n) = value {
            return Cow::Owned(if fmt == "x" {
                format!("{n:x}")
            } else {
                format!("{n:X}")
            });
        }
        return base(value);
    }
    let base = base(value);
    if let Some(w) = fmt.strip_prefix('0').and_then(|w| w.parse::<usize>().ok()) {
        return Cow::Owned(format!("{base:0>w$}"));
    }
    if let Some(w) = fmt.strip_prefix('>').and_then(|w| w.parse::<usize>().ok()) {
        return Cow::Owned(format!("{base:>w$}"));
    }
    if let Some(w) = fmt.strip_prefix('<').and_then(|w| w.parse::<usize>().ok()) {
        return Cow::Owned(format!("{base:<w$}"));
    }
    if let Ok(w) = fmt.parse::<usize>() {
        return Cow::Owned(format!("{base:>w$}"));
    }
    base
}

fn as_f64(v: ValueRef<'_>) -> Option<f64> {
    match v {
        ValueRef::F64(f) => Some(f),
        ValueRef::U64(n) => Some(n as f64),
        _ => None,
    }
}

/// The `N` of a `.N` precision format, after trimming.
fn precision_of(fmt: &str) -> Option<usize> {
    fmt.trim()
        .strip_prefix('.')
        .and_then(|p| p.parse::<usize>().ok())
}

fn push_json_escaped<W: std::fmt::Write>(s: &str, out: &mut W) {
    for c in s.chars() {
        match c {
            '"' => out.put("\\\""),
            '\\' => out.put("\\\\"),
            '\n' => out.put("\\n"),
            '\r' => out.put("\\r"),
            '\t' => out.put("\\t"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.put_char(c),
        }
    }
}

/// Encode one value under a hole's spec
/// (`encoding|position|type|format|flags`).
///
/// A library node a program may call. The compiler emitted one of
/// these per hole until encoding moved into the renderer, which
/// encodes each hole where it stands in the skeleton; nothing
/// generates this node now.
#[crate::polydat_node(category = Formatting)]
fn tile_encode(
    value: Value,
    spec: Const<&str>,
    #[poly_const(HoleEncoding::from_spec, from = spec)] enc: &HoleEncoding,
) -> String {
    let mut out = String::new();
    encode(&value, enc, &mut out);
    out
}

/// The closure-tier form of `tile_render` (SRD 117 step 1): every hole
/// value is read from its slot as a borrowed view, by the wire type the
/// kernel fixed, and the document is rendered straight into the step's
/// own string scratch through a `BytesSink`; nothing is decoded into
/// an owned `Value` on the way. A wire
/// wider than one slot, or of a kind without a view, is read as a value
/// through the typed decoder.
fn tile_render_compiled(node: &TileRender, wire_types: &[PortType]) -> crate::ast::CompiledSlotKit {
    // Native code bakes this address, so it must outlive every kernel
    // compiled from the program: one reference count of the node's own
    // `Arc` is given up here and never taken back. This used to be a
    // process-wide table keyed by the whole JSON payload, which made
    // "the same tile" mean "the same bytes of JSON".
    let program: &'static TileProgram = unsafe { &*std::sync::Arc::into_raw(node.program.clone()) };
    // Per wire: its first slot and its type; a one-slot carrier or a
    // `Ref2` kind is viewed in place, a two-slot immediate is decoded.
    let mut reads: Vec<(usize, PortType)> = Vec::with_capacity(wire_types.len());
    let mut offset = 0usize;
    for &ty in wire_types {
        reads.push((offset, ty));
        offset += ty.slot_width().max(1);
    }
    crate::ast::CompiledSlotKit {
        scratch: vec![
            crate::ast::ScratchElem::Str,
            crate::ast::ScratchElem::Kernels,
        ],
        op: Box::new(
            move |inputs: &[u64], outputs: &mut [u64], scratch: &mut [crate::ast::ScratchBuf]| {
                // Owned values only for the two-slot immediates; they keep
                // their positions, so the views are built once they are
                // all in place.
                let owned_values: Vec<Value> = reads
                    .iter()
                    .filter(|(_, ty)| ty.slot_color() == crate::ast::SlotColor::Imm2)
                    .map(|&(offset, ty)| crate::compile::marshal::decode_output(inputs, offset, ty))
                    .collect();
                let mut next_owned = 0usize;
                let refs: Vec<ValueRef<'_>> = reads
                    .iter()
                    .map(|&(offset, ty)| {
                        if ty.slot_color() == crate::ast::SlotColor::Imm2 {
                            let v = ValueRef::from(&owned_values[next_owned]);
                            next_owned += 1;
                            v
                        } else {
                            // SAFETY: a pair in the buffer was published by
                            // a producer whose storage is alive (S3, S4).
                            unsafe { crate::compile::marshal::arg_ref(ty, &inputs[offset..]) }
                        }
                    })
                    .collect();
                // The document is rendered straight into this step's own
                // scratch (axiom S3). A body runs compiled wherever the
                // kernel rendering is compiled: this closure serves the
                // closure tier and a hybrid kernel's closure steps alike,
                // so the body takes the default engine.
                let (text, bodies) = scratch.split_at_mut(1);
                let crate::ast::ScratchBuf::Str(buf) = &mut text[0] else {
                    unreachable!("the render step owns a string entry");
                };
                let bodies = render_state::bodies_of(&mut bodies[0]);
                buf.clear();
                let mut w = BytesSink(buf);
                program.render_into(&refs, crate::Engine::default(), bodies, &mut w);
                let (p, l) = scratch[0].ptr_len();
                outputs[0] = p;
                outputs[1] = l;
            },
        ),
    }
}

/// A text sink over the bytes of a step's string scratch: what a
/// compiled render writes into.
pub(crate) struct BytesSink<'a>(pub(crate) &'a mut Vec<u8>);

impl std::fmt::Write for BytesSink<'_> {
    fn write_str(&mut self, s: &str) -> std::fmt::Result {
        self.0.extend_from_slice(s.as_bytes());
        Ok(())
    }
}

/// Render a compiled tile skeleton over its encoded hole texts.
///
/// The compiler emits this for a `tile` statement and hands it the
/// skeleton it built, projection bodies and all. The skeleton used to
/// travel as JSON in a string constant, which this node parsed back
/// and compiled at construction: a malformed payload was a panic here
/// rather than a compile error, the body's source lived in three
/// places, and what the compiler knew about the body that JSON cannot
/// carry — its source directory, library paths, strict flag, pragmas,
/// and the modules the program had resolved — was lost on the way.
#[crate::polydat_node(
    category = Formatting,
    variadic_min = 0,
    compiled_slot = tile_render_compiled,
    state = render_state
)]
fn tile_render(program: Const<Arc<TileProgram>>, values: &[Value]) -> String {
    // A render without a state's scratch (a node evaluated on its
    // own): body kernels of the call's own.
    program.render(values, &mut BodyKernels::default())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A rendering state's body kernels are created on the first
    /// render that reaches a projection and reused by every render
    /// after; a clone of the set is a new, empty set.
    #[test]
    fn body_kernels_are_created_once_per_state_and_reused() {
        let src =
            "input cycle: u64\ntile t : text := \"@for k in 0..3 sep \\\",\\\" {${k + cycle}}\"\n";
        let mut k = crate::dsl::compile_polydat_interpreter(src).unwrap();
        let program = k.program();
        let node = (0..program.node_count())
            .find(|&i| program.node_meta(i).name == "tile_render")
            .expect("the tile's render node");
        let bodies_of = |k: &mut PolydatKernel| match &k.state().core.node_scratch[node][0] {
            crate::ast::ScratchBuf::Kernels(b) => b.clone_for_test(),
            other => panic!("{other:?}"),
        };
        assert_eq!(bodies_of(&mut k).0, 0, "nothing before the first render");
        k.set_inputs(&[10]);
        assert_eq!(k.pull_ref("t").as_str(), "10,11,12");
        assert_eq!(bodies_of(&mut k).0, 1, "one kernel for the body");
        for c in 0..5u64 {
            k.set_inputs(&[c]);
            let _ = k.pull_ref("t");
        }
        let (created, clone) = bodies_of(&mut k);
        assert_eq!(created, 1, "reused across renders");
        assert_eq!(clone.created(), 0, "a clone is a new state's empty set");
    }

    fn enc(
        encoding: &str,
        position: HolePosition,
        ty: Option<&str>,
        format: Option<&str>,
        raw: bool,
    ) -> HoleEncoding {
        HoleEncoding {
            encoding: encoding.into(),
            position,
            ty: ty.map(str::to_string),
            format: format.map(str::to_string),
            raw,
            cond: false,
        }
    }

    #[test]
    fn json_value_and_string_positions_encode_by_type() {
        let mut out = String::new();
        encode(
            &Value::Str("a\"b".into()),
            &enc("json", HolePosition::Value, Some("str"), None, false),
            &mut out,
        );
        assert_eq!(out, "\"a\\\"b\"");
        out.clear();
        encode(
            &Value::U64(7),
            &enc("json", HolePosition::Value, None, None, false),
            &mut out,
        );
        assert_eq!(out, "7");
        out.clear();
        encode(
            &Value::Str("x\ny".into()),
            &enc("json", HolePosition::InString, None, None, false),
            &mut out,
        );
        assert_eq!(out, "x\\ny");
        out.clear();
        encode(
            &Value::F64(2.0 / 3.0),
            &enc("json", HolePosition::Value, None, Some(".2"), false),
            &mut out,
        );
        assert_eq!(out, "0.67");
        out.clear();
        encode(
            &Value::None,
            &enc("json", HolePosition::Value, None, None, false),
            &mut out,
        );
        assert_eq!(out, "null");
    }

    #[test]
    fn spec_round_trips() {
        let e = enc(
            "json",
            HolePosition::InString,
            Some("u64"),
            Some(".2"),
            true,
        );
        assert_eq!(HoleEncoding::from_spec(&e.to_spec()), e);
        let c = HoleEncoding {
            cond: true,
            ..enc("text", HolePosition::Text, None, None, false)
        };
        assert_eq!(HoleEncoding::from_spec(&c.to_spec()), c);
    }

    #[test]
    fn csv_quotes_when_needed_and_raw_skips_escaping() {
        let mut out = String::new();
        encode(
            &Value::Str("a,b".into()),
            &enc("csv", HolePosition::Text, None, None, false),
            &mut out,
        );
        assert_eq!(out, "\"a,b\"");
        out.clear();
        encode(
            &Value::Str("a\"b".into()),
            &enc("json", HolePosition::Value, None, None, true),
            &mut out,
        );
        assert_eq!(out, "a\"b");
    }

    #[test]
    fn formats_apply_before_encoding() {
        assert_eq!(formatted_text(ValueRef::U64(5), None, Some("03")), "005");
        assert_eq!(formatted_text(ValueRef::U64(255), None, Some("x")), "ff");
        assert_eq!(
            formatted_text(ValueRef::Str("ab"), None, Some(">4")),
            "  ab"
        );
        assert_eq!(
            formatted_text(ValueRef::F64(0.295), None, Some(".2")),
            "0.29"
        );
        assert_eq!(
            formatted_text(ValueRef::U64(7), None, Some(" .3 ")),
            "7.000"
        );
    }

    /// A float hole's bytes are `format!`'s, on the direct path and on
    /// the general one, in every encoding and position.
    #[test]
    fn float_holes_write_rust_text() {
        let cases: [(f64, Option<&str>, &str); 8] = [
            (100.0, None, "100.0"),
            (0.1, None, "0.1"),
            (5e-5, None, "5e-5"),
            (1e16, None, "1e16"),
            (-0.0, None, "-0.0"),
            (2.0 / 3.0, Some(".2"), "0.67"),
            (0.295, Some(".2"), "0.29"),
            (2.5, Some(".0"), "2"),
        ];
        for (f, fmt, want) in cases {
            for (encoding, position) in [
                ("text", HolePosition::Text),
                ("json", HolePosition::Value),
                ("json", HolePosition::InString),
                ("csv", HolePosition::Text),
            ] {
                for ty in [None, Some("f64")] {
                    let mut out = String::new();
                    encode(
                        &Value::F64(f),
                        &enc(encoding, position, ty, fmt, false),
                        &mut out,
                    );
                    assert_eq!(out, want, "{f:?} {fmt:?} {encoding} {position:?} {ty:?}");
                }
            }
            // A non-numeric declared type takes the general path; the
            // text is the same, quoted where the position quotes.
            let mut out = String::new();
            encode(
                &Value::F64(f),
                &enc("json", HolePosition::Value, Some("str"), fmt, false),
                &mut out,
            );
            assert_eq!(out, format!("\"{want}\""));
        }
    }

    /// A tile's ops say whether rendering it re-runs a projection body
    /// (`TileProgram::has_projections`): a body inside a branch arm
    /// counts, and the answer agrees with the compiled body programs.
    #[test]
    fn a_tile_reports_whether_any_op_re_runs_a_projection() {
        fn wire(index: usize) -> HoleSource {
            HoleSource::Wire {
                index,
                spec: "text|text".into(),
            }
        }
        fn program(ops: Vec<TileOp>, children: Vec<ChildSpec>) -> TileProgram {
            TileProgram::from_json(
                &TileSpec {
                    name: "t".into(),
                    encoding: "text".into(),
                    ops,
                    children,
                }
                .to_json(),
            )
        }
        let body = || ChildSpec {
            source: "input cycle: u64\nextern k: u64\nout := u64_add(k, 0)\n".to_string(),
            cascade: Vec::new(),
        };
        let repeat = |child: usize| TileOp::Repeat {
            stream: StreamerValue::parse_text("k in 0..3").unwrap().to_json(),
            child,
            sep: ",".into(),
            body: vec![TileOp::Hole(HoleSource::Child {
                name: "out".into(),
                spec: "text|text".into(),
            })],
            generators: Vec::new(),
        };

        // Statics, holes, and a branch over them: nothing re-runs.
        let flat = program(
            vec![
                TileOp::Static("a=".into()),
                TileOp::Hole(wire(0)),
                TileOp::Branch {
                    cond: wire(1),
                    then: vec![TileOp::Static("yes".into())],
                    otherwise: vec![TileOp::Hole(wire(0))],
                },
            ],
            Vec::new(),
        );
        assert!(!flat.has_projections());
        assert!(flat.children.is_empty());

        // A projection at the top level.
        let top = program(vec![TileOp::Static("[".into()), repeat(0)], vec![body()]);
        assert!(top.has_projections());
        assert_eq!(top.children.len(), 1);

        // A projection inside a branch arm: the walk recurses, so the
        // arm that holds it is found whichever arm it is.
        for (then, otherwise) in [
            (vec![repeat(0)], vec![TileOp::Static("none".into())]),
            (vec![TileOp::Static("none".into())], vec![repeat(0)]),
        ] {
            let branched = program(
                vec![TileOp::Branch {
                    cond: wire(0),
                    then,
                    otherwise,
                }],
                vec![body()],
            );
            assert!(branched.has_projections());
        }
    }
}
