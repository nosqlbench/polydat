// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! The tile nodes (SRD 114 §6, §7.1).
//!
//! The compiler lowers a `tile` statement to one `tile_encode` binding
//! per hole and one `tile_render` binding for the tile. `tile_encode`
//! takes the hole's value on a polymorphic wire and produces the
//! encoded text for the hole's position under the tile's encoding, its
//! declared or observed type, its format, and its raw flag. Encoding is
//! therefore an ordinary node on the graph, visible to provenance and
//! to the engines. `tile_render` then concatenates static runs with
//! encoded holes, selects branches, and re-runs projection bodies per
//! tuple over a scratch state, using a skeleton it parses once at setup.

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

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

/// The encoder for one hole, carried as the `tile_encode` node's spec.
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

    /// The encoding for a spec, interned for the process (SRD 115 §6)
    /// so the compiled lowering of `tile_encode` can bake its address.
    pub fn interned(spec: &str) -> &'static HoleEncoding {
        use std::sync::RwLock;
        static ENCODINGS: RwLock<Option<HashMap<String, &'static HoleEncoding>>> =
            RwLock::new(None);
        if let Some(e) = ENCODINGS
            .read()
            .unwrap()
            .as_ref()
            .and_then(|m| m.get(spec).copied())
        {
            return e;
        }
        let mut guard = ENCODINGS.write().unwrap();
        let map = guard.get_or_insert_with(HashMap::new);
        if let Some(e) = map.get(spec).copied() {
            return e;
        }
        let leaked: &'static HoleEncoding = Box::leak(Box::new(Self::from_spec(spec)));
        map.insert(spec.to_string(), leaked);
        leaked
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
            TileOp::Static(s) => {
                let handle = StaticInterner::intern(s);
                RtOp::Copy(StaticInterner::resolve_handle(handle).expect("just interned"))
            }
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
            } => {
                let sep_handle = StaticInterner::intern(sep);
                RtOp::Repeat {
                    stream: Arc::new(StreamerValue::from_json(stream)),
                    child: *child,
                    sep: StaticInterner::resolve_handle(sep_handle).expect("just interned"),
                    body: lower_ops(body),
                    generators: generators.clone(),
                }
            }
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
    /// The body programs, compiled once at setup, for the interpreter.
    pub children: Vec<Arc<PolydatProgram>>,
    /// One kernel over each body program, the canonical kernel the
    /// comprehension evaluator installs tuple values into.
    canonicals: Vec<Arc<PolydatKernel>>,
    /// Per body, its program on the default engine (SRD 117 step 2),
    /// compiled here, at construction, never inside a cycle: a kernel's
    /// build folds its constants in a root cycle of its own, which would
    /// reset the arena a render is writing (SRD 115, H5). `None` where
    /// the engine refused the body, which then renders interpreted.
    compiled: Vec<Option<Arc<dyn KernelProgram>>>,
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
    /// Parse a skeleton and compile its projection bodies. Panics with
    /// the compiler's diagnostic on a malformed payload, which only the
    /// compiler produces.
    pub fn from_json(json: &str) -> Self {
        let spec: TileSpec = serde_json::from_str(json)
            .unwrap_or_else(|e| panic!("tile_render: malformed skeleton payload: {e}"));
        let children: Vec<Arc<PolydatProgram>> = spec
            .children
            .iter()
            .map(|c| {
                crate::dsl::compile_polydat(&c.source)
                    .unwrap_or_else(|e| {
                        panic!(
                            "tile '{}': projection body failed to compile: {e}\n{}",
                            spec.name, c.source
                        )
                    })
                    .into_program()
            })
            .collect();
        // Both kernels serve the comprehension evaluator inside a render,
        // so neither is a root of its own cycle (SRD 115, axiom H5).
        let canonicals: Vec<Arc<PolydatKernel>> = children
            .iter()
            .map(|p| Arc::new(PolydatKernel::from_program_nested(p.clone())))
            .collect();
        let mut ops = lower_ops(&spec.ops);
        number_child_holes(&mut ops);
        let mut memo = vec![None; children.len()];
        memoize(&ops, &canonicals, &mut memo);
        let compiled = spec
            .children
            .iter()
            .enumerate()
            .map(|(i, c)| {
                match crate::dsl::compile::compile_polydat_with(&c.source, crate::Engine::default())
                {
                    Ok(kernel) => Some(kernel.into_program()),
                    Err(e) => {
                        crate::library::support::audit::debug(&format!(
                            "tile '{}': projection body {i} renders on the interpreter: {e}",
                            spec.name
                        ));
                        None
                    }
                }
            })
            .collect();
        TileProgram {
            spec,
            ops,
            children,
            canonicals,
            compiled,
            memo,
        }
    }

    /// The body program of projection `child` for a render on `engine`:
    /// the interpreter's for the interpreter, the default engine's for
    /// every compiled kernel, and the interpreter's again where the
    /// default engine refused the body.
    fn body_program_on(&self, child: usize, engine: crate::Engine) -> Arc<dyn KernelProgram> {
        if engine == crate::Engine::Interpreter {
            return self.children[child].clone();
        }
        self.compiled[child]
            .clone()
            .unwrap_or_else(|| self.children[child].clone())
    }

    /// The program for a skeleton payload, interned for the process
    /// (SRD 115 §6): the compiled lowering of `tile_render` bakes its
    /// address, so it must outlive every kernel compiled from it, and
    /// the same payload is parsed and its bodies compiled once.
    pub fn interned(spec: &str) -> &'static TileProgram {
        use std::sync::RwLock;
        static PROGRAMS: RwLock<Option<HashMap<String, usize>>> = RwLock::new(None);
        let found = PROGRAMS
            .read()
            .unwrap()
            .as_ref()
            .and_then(|m| m.get(spec).copied());
        if let Some(p) = found {
            // SAFETY: the address was leaked below and is never freed.
            return unsafe { &*(p as *const TileProgram) };
        }
        // Built with no lock held: constructing a program compiles its
        // projection bodies, and a body's own tile interns its program
        // through this same table (SRD 117 step 2). Two threads may
        // build the same program at once; the first to insert wins and
        // the other's build is dropped.
        let built = Box::new(Self::from_json(spec));
        let mut guard = PROGRAMS.write().unwrap();
        let map = guard.get_or_insert_with(HashMap::new);
        if let Some(&p) = map.get(spec) {
            // SAFETY: as above.
            return unsafe { &*(p as *const TileProgram) };
        }
        let leaked: &'static TileProgram = Box::leak(built);
        map.insert(spec.to_string(), leaked as *const TileProgram as usize);
        leaked
    }

    /// True when any op re-runs a projection body: such a skeleton
    /// stays on P1 until projection bodies activate as `for` bodies do.
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
    /// interpreter.
    pub fn render(&self, inputs: &[Value]) -> String {
        let refs: Vec<ValueRef<'_>> = inputs.iter().map(ValueRef::from).collect();
        let mut out = String::new();
        self.render_into(&refs, crate::Engine::Interpreter, &mut out);
        out
    }

    /// Render into any text sink from borrowed views of the hole
    /// values: a `String` at P1, the cycle arena writer in a compiled
    /// closure or helper (SRD 115 §6). Every hole is encoded here, from
    /// the view straight into the sink, and a projection's body runs
    /// on `engine`, the engine of the kernel rendering.
    pub fn render_into<W: std::fmt::Write>(
        &self,
        inputs: &[ValueRef<'_>],
        engine: crate::Engine,
        out: &mut W,
    ) {
        self.render_ops(&self.ops, inputs, engine, None, out);
    }

    fn render_ops<W: std::fmt::Write>(
        &self,
        ops: &[RtOp],
        inputs: &[ValueRef<'_>],
        engine: crate::Engine,
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
                    self.render_ops(branch, inputs, engine, child.as_deref_mut(), out);
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
                    // The body runs compiled wherever the kernel rendering
                    // is compiled, as a nested kernel over the body's program
                    // for the default engine, reused across renders on this
                    // thread; on the interpreter it runs interpreted.
                    let engine = match engine {
                        crate::Engine::Interpreter => engine,
                        _ => crate::Engine::default(),
                    };
                    let program = self.body_program_on(*child_idx, engine);
                    let mut first = true;
                    let fail = |name: &str, e: String| -> ! {
                        panic!(
                            "tile '{}': projection body input `{name}`: {e}",
                            self.spec.name
                        )
                    };
                    with_body_kernel(&program, engine, |entry| {
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
                            self.render_ops(body, inputs, engine, Some(entry), out);
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
                match crate::iteration::comprehension::source::iteration_interior(&raw) {
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
            let values = items
                .iter()
                .map(|v| match v {
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
        } => K::Order {
            child: Box::new(bind_generators(child, generators, inputs)),
            strategy: *strategy,
            truncation: *truncation,
        },
    }
}

/// Recover a typed value from the display text a cascaded wire arrives
/// as, using the child extern's declared type.
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

thread_local! {
    /// One kernel per projection body program and engine per thread,
    /// reused across renders so a projection creates nothing per tuple.
    static BODY_KERNELS: RefCell<HashMap<(usize, crate::Engine), BodyEntry>> = RefCell::new(HashMap::new());
    /// Body kernels created on this thread, by engine: a diagnostic for
    /// the tests.
    static BODIES_CREATED: RefCell<HashMap<crate::Engine, u64>> = RefCell::new(HashMap::new());
}

/// Distinct body programs one thread will keep kernels for before
/// starting over. Bounds the cache when programs are compiled and
/// dropped in a loop.
const SCRATCH_LIMIT: usize = 64;

/// Projection body kernels created on this thread for `engine`.
#[doc(hidden)]
pub fn body_kernels_created(engine: crate::Engine) -> u64 {
    BODIES_CREATED.with(|m| m.borrow().get(&engine).copied().unwrap_or(0))
}

fn with_body_kernel(
    program: &Arc<dyn KernelProgram>,
    engine: crate::Engine,
    f: impl FnOnce(&mut BodyEntry),
) {
    // The entry pins its program so the address cannot be reused by a
    // later program while a kernel built for this one is still cached.
    let key = (Arc::as_ptr(program) as *const () as usize, engine);
    let mut entry = BODY_KERNELS
        .with(|m| m.borrow_mut().remove(&key))
        .filter(|e| Arc::ptr_eq(&e.program, program))
        .unwrap_or_else(|| {
            // A body runs inside the enclosing cycle: it must never
            // reset the thread's arena (SRD 115, axiom H5).
            let kernel = program.clone().create_nested_kernel();
            BODIES_CREATED.with(|m| *m.borrow_mut().entry(engine).or_insert(0) += 1);
            BodyEntry {
                program: program.clone(),
                kernel,
                elements: None,
                cascade: None,
                holes: Vec::new(),
            }
        });
    f(&mut entry);
    BODY_KERNELS.with(|m| {
        let mut m = m.borrow_mut();
        if m.len() >= SCRATCH_LIMIT {
            m.clear();
        }
        m.insert(key, entry);
    });
}

/// A text sink that cannot fail: a `String`, or the cycle arena writer
/// in a compiled helper. `fmt::Write`'s results are ignored because
/// neither sink reports an error.
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
/// helper calls this on its slot without owning a `Value`, and a
/// string hole is encoded from the arena in place.
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

/// Encode one hole's value per its spec (`encoding|position|type|format|flags`).
/// Authors do not call this directly; the compiler emits it for each hole.
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
/// kernel fixed, and the document is rendered straight into the cycle
/// arena; nothing is decoded into an owned `Value` on the way. A wire
/// wider than one slot, or of a kind without a view, is read as a value
/// through the typed decoder.
fn tile_render_compiled(
    node: &TileRender,
    _entry_base: usize,
    wire_types: &[PortType],
) -> crate::ast::CompiledU64Op {
    let program: &'static TileProgram = TileProgram::interned(&node.spec);
    // Per wire: its first slot, its type, and its view code where the
    // wire is one slot of a kind `arg_ref` reads.
    let mut reads: Vec<(usize, PortType, Option<u8>)> = Vec::with_capacity(wire_types.len());
    let mut offset = 0usize;
    for &ty in wire_types {
        let code = if ty.slot_width() == 1 {
            crate::compile::marshal::type_code(ty)
        } else {
            None
        };
        reads.push((offset, ty, code));
        offset += ty.slot_width().max(1);
    }
    Box::new(move |inputs: &[u64], outputs: &mut [u64]| {
        // Owned values only for the wires without a view; they keep
        // their positions, so the views are built once they are all in
        // place.
        let owned_values: Vec<Value> = reads
            .iter()
            .filter(|(_, _, code)| code.is_none())
            .map(|&(offset, ty, _)| {
                crate::kernel::with_current_value_table(|t| {
                    crate::compile::marshal::decode_output(inputs, offset, ty, t)
                })
            })
            .collect();
        let mut next_owned = 0usize;
        let refs: Vec<ValueRef<'_>> = reads
            .iter()
            .map(|&(offset, _, code)| match code {
                Some(code) => crate::compile::marshal::arg_ref(code, inputs[offset]),
                None => {
                    let v = ValueRef::from(&owned_values[next_owned]);
                    next_owned += 1;
                    v
                }
            })
            .collect();
        // A body runs compiled wherever the kernel rendering is compiled:
        // this closure serves the closure tier and a hybrid kernel's
        // closure steps alike, so the body takes the default engine.
        let mut w = crate::kernel::ArenaWriter::new();
        program.render_into(&refs, crate::Engine::default(), &mut w);
        outputs[0] = w.finish();
    })
}

/// Render a compiled tile skeleton over its encoded hole texts. Authors
/// do not call this directly; the compiler emits it for `tile` statements.
#[crate::polydat_node(
    category = Formatting,
    variadic_min = 0,
    compiled_handle = tile_render_compiled
)]
fn tile_render(
    spec: Const<&str>,
    #[poly_const(TileProgram::from_json, from = spec)] program: &TileProgram,
    values: &[Value],
) -> String {
    program.render(values)
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
