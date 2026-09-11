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
use crate::iteration::comprehension::runtime::evaluate_for_iteration;
use crate::kernel::{PolydatKernel, PolydatProgram, PolydatState};

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
    Child(String),
}

fn lower_source(source: &HoleSource) -> (RtSource, HoleEncoding) {
    match source {
        HoleSource::Wire { index, spec } => (RtSource::Wire(*index), HoleEncoding::from_spec(spec)),
        HoleSource::Child { name, spec } => {
            (RtSource::Child(name.clone()), HoleEncoding::from_spec(spec))
        }
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
#[derive(Debug)]
pub struct TileProgram {
    /// The serialized skeleton.
    pub spec: TileSpec,
    /// The skeleton with statics interned and streams parsed.
    ops: Vec<RtOp>,
    /// The body programs, compiled once at setup.
    pub children: Vec<Arc<PolydatProgram>>,
    /// One kernel over each body program, the canonical kernel the
    /// comprehension evaluator installs tuple values into.
    canonicals: Vec<Arc<PolydatKernel>>,
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
        let canonicals = children
            .iter()
            .map(|p| Arc::new(PolydatKernel::from_program_nested(p.clone())))
            .collect();
        let ops = lower_ops(&spec.ops);
        TileProgram {
            spec,
            ops,
            children,
            canonicals,
        }
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
        let mut guard = PROGRAMS.write().unwrap();
        let map = guard.get_or_insert_with(HashMap::new);
        if let Some(&p) = map.get(spec) {
            // SAFETY: as above.
            return unsafe { &*(p as *const TileProgram) };
        }
        let leaked: &'static TileProgram = Box::leak(Box::new(Self::from_json(spec)));
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

    /// Render with the node's wire inputs, the hole values.
    pub fn render(&self, inputs: &[Value]) -> String {
        let refs: Vec<ValueRef<'_>> = inputs.iter().map(ValueRef::from).collect();
        let mut out = String::new();
        self.render_into(&refs, &mut out);
        out
    }

    /// Render into any text sink from borrowed views of the hole
    /// values: a `String` at P1, the cycle arena writer in a compiled
    /// closure or helper (SRD 115 §6). Every hole is encoded here, from
    /// the view straight into the sink.
    pub fn render_into<W: std::fmt::Write>(&self, inputs: &[ValueRef<'_>], out: &mut W) {
        self.render_ops(&self.ops, inputs, None, out);
    }

    fn render_ops<W: std::fmt::Write>(
        &self,
        ops: &[RtOp],
        inputs: &[ValueRef<'_>],
        mut child: Option<(&Arc<PolydatProgram>, &mut PolydatState)>,
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
                    RtSource::Child(name) => {
                        if let Some((program, state)) = child.as_mut() {
                            encode_ref(ValueRef::from(state.pull(program, name)), enc, out)
                        }
                    }
                },
                RtOp::Branch {
                    cond,
                    then,
                    otherwise,
                } => {
                    let c = self.truthy(cond, inputs, child.as_mut());
                    let branch = if c { then } else { otherwise };
                    match child.as_mut() {
                        Some((p, s)) => self.render_ops(branch, inputs, Some((p, s)), out),
                        None => self.render_ops(branch, inputs, None, out),
                    }
                }
                RtOp::Repeat {
                    stream,
                    child: child_idx,
                    sep,
                    body,
                    generators,
                } => {
                    let mut streamer = (**stream).clone();
                    if !generators.is_empty() {
                        streamer.ast = bind_generators(&streamer.ast, generators, inputs);
                    }
                    let program = &self.children[*child_idx];
                    let child_spec = &self.spec.children[*child_idx];
                    // The same evaluator the `for` construct opens a
                    // traversal with: it applies order strategies,
                    // samples continuous sources, and runs predicates
                    // over the tuple. Generators were bound above, so
                    // the parent kernel it sees is empty and the
                    // canonical kernel is the body program.
                    let tuples = evaluate_for_iteration(
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
                    });
                    let mut first = true;
                    with_scratch(program, |state| {
                        for (index, tuple) in tuples.iter().enumerate() {
                            if !first {
                                out.put(sep);
                            }
                            first = false;
                            state.set_inputs(&[index as u64]);
                            for (name, v) in tuple {
                                if let Some(idx) = program.find_input(name) {
                                    state.set_input(idx, v.clone());
                                }
                            }
                            for (name, input_idx, ty) in &child_spec.cascade {
                                if let (Some(idx), Some(v)) =
                                    (program.find_input(name), inputs.get(*input_idx))
                                {
                                    state.set_input(idx, typed_for(&owned(*v), ty));
                                }
                            }
                            self.render_ops(body, inputs, Some((program, state)), out);
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
        child: Option<&mut (&Arc<PolydatProgram>, &mut PolydatState)>,
    ) -> bool {
        match source {
            RtSource::Wire(i) => truthy_of(inputs.get(*i).copied().unwrap_or(ValueRef::None)),
            RtSource::Child(name) => match child {
                Some((program, state)) => truthy_of(ValueRef::from(state.pull(program, name))),
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

thread_local! {
    /// One scratch state per projection body program per thread, reused
    /// across renders so a projection allocates nothing per tuple.
    static SCRATCH: RefCell<HashMap<usize, (Arc<PolydatProgram>, PolydatState)>> = RefCell::new(HashMap::new());
}

/// Distinct child programs one thread will keep scratch states for
/// before starting over. Bounds the cache when programs are compiled
/// and dropped in a loop.
const SCRATCH_LIMIT: usize = 64;

fn with_scratch(program: &Arc<PolydatProgram>, f: impl FnOnce(&mut PolydatState)) {
    // The entry pins its program so the address cannot be reused by a
    // later program while a state built for this one is still cached.
    let key = Arc::as_ptr(program) as usize;
    let mut state = SCRATCH
        .with(|m| m.borrow_mut().remove(&key))
        .filter(|(p, _)| Arc::ptr_eq(p, program))
        .map(|(_, s)| s)
        .unwrap_or_else(|| {
            // A body runs inside the enclosing cycle: it must never
            // reset the thread's arena (SRD 115, axiom H5).
            let mut s = program.create_state();
            s.mark_nested();
            s
        });
    f(&mut state);
    SCRATCH.with(|m| {
        let mut m = m.borrow_mut();
        if m.len() >= SCRATCH_LIMIT {
            m.clear();
        }
        m.insert(key, (Arc::clone(program), state));
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
    if let Some(prec) = fmt.strip_prefix('.').and_then(|p| p.parse::<usize>().ok()) {
        if let Some(f) = as_f64(value) {
            return Cow::Owned(format!("{f:.prec$}"));
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
        let mut w = crate::kernel::ArenaWriter::new();
        program.render_into(&refs, &mut w);
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
    }
}
