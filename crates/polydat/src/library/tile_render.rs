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

use crate::ast::{PortType, Value};
use crate::iteration::comprehension::runtime::evaluate_for_iteration;
use crate::iteration::comprehension::StreamerValue;
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
    pub encoding: String,
    pub position: HolePosition,
    pub ty: Option<String>,
    pub format: Option<String>,
    pub raw: bool,
    /// Encode as a branch condition: `1` or `0`.
    pub cond: bool,
}

impl HoleEncoding {
    pub fn to_spec(&self) -> String {
        let pos = match self.position {
            HolePosition::Value => "value",
            HolePosition::InString => "string",
            HolePosition::Text => "text",
        };
        let mut flags = String::new();
        if self.raw { flags.push('r'); }
        if self.cond { flags.push('c'); }
        format!("{}|{}|{}|{}|{}", self.encoding, pos, self.ty.as_deref().unwrap_or(""), self.format.as_deref().unwrap_or(""), flags)
    }

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
        HoleEncoding { encoding, position, ty, format, raw: flags.contains('r'), cond: flags.contains('c') }
    }
}

/// Where a hole's encoded text comes from at render time.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum HoleSource {
    /// The render node's wire input at this index.
    Wire(usize),
    /// An output of the enclosing projection's body program.
    Child(String),
}

/// One skeleton instruction. Holes arrive already encoded.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TileOp {
    Static(String),
    Hole(HoleSource),
    Repeat {
        /// The comprehension, as a serialized [`StreamerValue`].
        stream: String,
        /// Index into [`TileSpec::children`].
        child: usize,
        sep: String,
        body: Vec<TileOp>,
        /// Generator-call clauses whose expressions compiled to wires of
        /// the enclosing program: `(element, node input index, type)`.
        /// At render the input's value stands in for the clause.
        #[serde(default)]
        generators: Vec<(String, usize, String)>,
    },
    Branch {
        cond: HoleSource,
        then: Vec<TileOp>,
        otherwise: Vec<TileOp>,
    },
}

/// A projection body: a program compiled once at setup, and the outer
/// wires it imports from the render node's inputs, each with the type
/// its extern declares so the transported text can be re-typed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChildSpec {
    pub source: String,
    /// `(extern name, render-node input index, port-type keyword)`.
    pub cascade: Vec<(String, usize, String)>,
}

/// The serialized skeleton.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TileSpec {
    pub name: String,
    pub encoding: String,
    pub ops: Vec<TileOp>,
    pub children: Vec<ChildSpec>,
}

impl TileSpec {
    pub fn to_json(&self) -> String {
        serde_json::to_string(self).expect("TileSpec serializes")
    }
}

/// The runtime form: the spec, compiled body programs, and parsed streams.
#[derive(Debug)]
pub struct TileProgram {
    pub spec: TileSpec,
    pub children: Vec<Arc<PolydatProgram>>,
    /// One kernel over each body program, the canonical kernel the
    /// comprehension evaluator installs tuple values into.
    canonicals: Vec<Arc<PolydatKernel>>,
    /// The empty parent kernel: every scope value a projection needs
    /// arrives through the node's inputs instead.
    empty: Arc<PolydatKernel>,
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
                    .unwrap_or_else(|e| panic!("tile '{}': projection body failed to compile: {e}\n{}", spec.name, c.source))
                    .into_program()
            })
            .collect();
        let canonicals = children.iter().map(|p| Arc::new(PolydatKernel::from_program(p.clone()))).collect();
        let empty = Arc::new(crate::dsl::compile_polydat("\n").expect("the empty program compiles"));
        TileProgram { spec, children, canonicals, empty }
    }

    /// Render with the node's wire inputs, each already encoded text.
    pub fn render(&self, inputs: &[Value]) -> String {
        let mut out = String::new();
        self.render_ops(&self.spec.ops, inputs, None, &mut out);
        out
    }

    fn render_ops(&self, ops: &[TileOp], inputs: &[Value], mut child: Option<(&Arc<PolydatProgram>, &mut PolydatState)>, out: &mut String) {
        for op in ops {
            match op {
                TileOp::Static(s) => out.push_str(s),
                TileOp::Hole(source) => out.push_str(&self.text_of(source, inputs, child.as_mut())),
                TileOp::Branch { cond, then, otherwise } => {
                    let c = self.text_of(cond, inputs, child.as_mut());
                    let branch = if c.trim() == "1" { then } else { otherwise };
                    match child.as_mut() {
                        Some((p, s)) => self.render_ops(branch, inputs, Some((p, s)), out),
                        None => self.render_ops(branch, inputs, None, out),
                    }
                }
                TileOp::Repeat { stream, child: child_idx, sep, body, generators } => {
                    let mut streamer = StreamerValue::from_json(stream);
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
                        &self.empty,
                        &self.canonicals[*child_idx],
                        &HashMap::new(),
                        |_| Ok(()),
                    )
                    .unwrap_or_else(|e| panic!("tile '{}': projection `for {}` failed at render: {e}", self.spec.name, streamer.text));
                    let mut first = true;
                    with_scratch(program, |state| {
                        for (index, tuple) in tuples.iter().enumerate() {
                            if !first {
                                out.push_str(sep);
                            }
                            first = false;
                            state.set_inputs(&[index as u64]);
                            for (name, v) in tuple {
                                if let Some(idx) = program.find_input(name) {
                                    state.set_input(idx, v.clone());
                                }
                            }
                            for (name, input_idx, ty) in &child_spec.cascade {
                                if let (Some(idx), Some(v)) = (program.find_input(name), inputs.get(*input_idx)) {
                                    state.set_input(idx, typed_for(v, ty));
                                }
                            }
                            self.render_ops(body, inputs, Some((program, state)), out);
                        }
                    });
                }
            }
        }
    }

    fn text_of(&self, source: &HoleSource, inputs: &[Value], child: Option<&mut (&Arc<PolydatProgram>, &mut PolydatState)>) -> String {
        match source {
            HoleSource::Wire(i) => inputs.get(*i).map(|v| v.to_display_string()).unwrap_or_default(),
            HoleSource::Child(name) => match child {
                Some((program, state)) => state.pull(program, name).to_display_string(),
                None => String::new(),
            },
        }
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
    inputs: &[Value],
) -> crate::iteration::comprehension::Comprehension {
    use crate::iteration::comprehension::source::{LiteralValue, Source};
    use crate::iteration::comprehension::Comprehension as K;
    match c {
        K::Clause { name, source: Source::Generator { .. } } => {
            let Some((_, idx, ty)) = generators.iter().find(|(n, _, _)| n == name) else {
                return c.clone();
            };
            let raw = inputs.get(*idx).cloned().unwrap_or(Value::None);
            let items: Vec<Value> = match crate::iteration::comprehension::source::iteration_interior(&raw) {
                Some(interior) => interior,
                None => match &raw {
                    Value::Str(text) => match serde_json::from_str::<serde_json::Value>(text.trim()) {
                        Ok(serde_json::Value::Array(items)) => items
                            .iter()
                            .map(|j| retype(&Value::Str(j.to_string().trim_matches('"').into()), ty))
                            .collect(),
                        _ => vec![typed_for(&raw, ty)],
                    },
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
                        serde_json::Value::Number(n) if n.is_u64() => LiteralValue::Int(n.as_u64().unwrap_or(0) as i64),
                        serde_json::Value::Number(n) if n.is_i64() => LiteralValue::Int(n.as_i64().unwrap_or(0)),
                        serde_json::Value::Number(n) => LiteralValue::Float(n.as_f64().unwrap_or(0.0)),
                        serde_json::Value::Bool(b) => LiteralValue::Bool(*b),
                        serde_json::Value::String(s) => LiteralValue::String(s.clone()),
                        other => LiteralValue::String(other.to_string()),
                    },
                    other => LiteralValue::String(other.to_display_string()),
                })
                .collect();
            K::Clause { name: name.clone(), source: Source::Literal { values } }
        }
        K::Clause { .. } => c.clone(),
        K::Cartesian { children } => K::Cartesian { children: children.iter().map(|ch| bind_generators(ch, generators, inputs)).collect() },
        K::Zip { children, mode } => K::Zip { children: children.iter().map(|ch| bind_generators(ch, generators, inputs)).collect(), mode: *mode },
        K::Union { children } => K::Union { children: children.iter().map(|ch| bind_generators(ch, generators, inputs)).collect() },
        K::Filter { child, predicate } => K::Filter { child: Box::new(bind_generators(child, generators, inputs)), predicate: predicate.clone() },
        K::Order { child, strategy, truncation } => K::Order {
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
        .unwrap_or_else(|| program.create_state());
    f(&mut state);
    SCRATCH.with(|m| {
        let mut m = m.borrow_mut();
        if m.len() >= SCRATCH_LIMIT {
            m.clear();
        }
        m.insert(key, (Arc::clone(program), state));
    });
}

/// Encode one value per a hole's encoding.
pub fn encode(value: &Value, enc: &HoleEncoding, out: &mut String) {
    if enc.cond {
        out.push(if truthy_of(value) { '1' } else { '0' });
        return;
    }
    let ty = enc.ty.as_deref();
    let text = formatted_text(value, ty, enc.format.as_deref());
    if enc.raw {
        out.push_str(&text);
        return;
    }
    match (enc.encoding.as_str(), enc.position) {
        ("json", HolePosition::InString) => push_json_escaped(&text, out),
        ("json", HolePosition::Value) => {
            let kind = ty.map(str::to_string).unwrap_or_else(|| value.port_type().to_keyword().to_string());
            match (kind.as_str(), value) {
                (_, Value::None) => out.push_str("null"),
                ("bool", _) => out.push_str(if truthy_of(value) { "true" } else { "false" }),
                ("json", Value::Json(j)) => out.push_str(&j.to_string()),
                ("str", _) | ("String", _) | ("string", _) => {
                    out.push('"');
                    push_json_escaped(&text, out);
                    out.push('"');
                }
                (k, _) if is_numeric_keyword(k) => out.push_str(&text),
                (_, Value::Json(j)) => out.push_str(&j.to_string()),
                (_, Value::Bool(b)) => out.push_str(if *b { "true" } else { "false" }),
                (_, Value::U64(_)) | (_, Value::F64(_)) => out.push_str(&text),
                _ => {
                    out.push('"');
                    push_json_escaped(&text, out);
                    out.push('"');
                }
            }
        }
        ("csv", _) => {
            if text.contains([',', '"', '\n']) {
                out.push('"');
                out.push_str(&text.replace('"', "\"\""));
                out.push('"');
            } else {
                out.push_str(&text);
            }
        }
        _ => out.push_str(&text),
    }
}

fn truthy_of(v: &Value) -> bool {
    match v {
        Value::Bool(b) => *b,
        Value::U64(n) => *n != 0,
        Value::F64(f) => *f != 0.0,
        Value::Str(s) => !s.is_empty() && s.as_ref() != "0" && s.as_ref() != "false",
        Value::None => false,
        _ => true,
    }
}

fn is_numeric_keyword(k: &str) -> bool {
    matches!(k, "u64" | "i64" | "f64" | "f32" | "u32" | "i32" | "u16" | "i16" | "u8" | "i8" | "u128" | "i128" | "f16")
}

/// Display text for a value under an optional printf-style format:
/// `.N` precision for floats, `0N` zero-padded width, `N` width, `>N`
/// and `<N` alignment, `x`/`X` hex for integers.
fn formatted_text(value: &Value, ty: Option<&str>, format: Option<&str>) -> String {
    let base = match (ty, value) {
        (Some("bool"), v) => truthy_of(v).to_string(),
        (_, Value::Json(j)) => j.to_string(),
        (_, v) => v.to_display_string(),
    };
    let Some(fmt) = format else { return base };
    let fmt = fmt.trim();
    if let Some(prec) = fmt.strip_prefix('.').and_then(|p| p.parse::<usize>().ok()) {
        if let Some(f) = as_f64(value) {
            return format!("{f:.prec$}");
        }
        return base;
    }
    if fmt == "x" || fmt == "X" {
        if let Value::U64(n) = value {
            return if fmt == "x" { format!("{n:x}") } else { format!("{n:X}") };
        }
        return base;
    }
    if let Some(w) = fmt.strip_prefix('0').and_then(|w| w.parse::<usize>().ok()) {
        return format!("{base:0>w$}");
    }
    if let Some(w) = fmt.strip_prefix('>').and_then(|w| w.parse::<usize>().ok()) {
        return format!("{base:>w$}");
    }
    if let Some(w) = fmt.strip_prefix('<').and_then(|w| w.parse::<usize>().ok()) {
        return format!("{base:<w$}");
    }
    if let Ok(w) = fmt.parse::<usize>() {
        return format!("{base:>w$}");
    }
    base
}

fn as_f64(v: &Value) -> Option<f64> {
    match v {
        Value::F64(f) => Some(*f),
        Value::U64(n) => Some(*n as f64),
        _ => None,
    }
}

fn push_json_escaped(s: &str, out: &mut String) {
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
}

/// Encode one hole's value per its spec (`encoding|position|type|format|flags`).
/// Authors do not call this directly; the compiler emits it for each hole.
#[crate::polydat_node(category = Formatting)]
fn tile_encode(value: Value, spec: Const<&str>) -> String {
    let enc = HoleEncoding::from_spec(&spec);
    let mut out = String::new();
    encode(&value, &enc, &mut out);
    out
}

/// Render a compiled tile skeleton over its encoded hole texts. Authors
/// do not call this directly; the compiler emits it for `tile` statements.
#[crate::polydat_node(category = Formatting, variadic_min = 0)]
fn tile_render(
    spec: Const<&str>,
    #[poly_const(TileProgram::from_json, from = spec)]
    program: &TileProgram,
    values: &[Value],
) -> String {
    program.render(values)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn enc(encoding: &str, position: HolePosition, ty: Option<&str>, format: Option<&str>, raw: bool) -> HoleEncoding {
        HoleEncoding { encoding: encoding.into(), position, ty: ty.map(str::to_string), format: format.map(str::to_string), raw, cond: false }
    }

    #[test]
    fn json_value_and_string_positions_encode_by_type() {
        let mut out = String::new();
        encode(&Value::Str("a\"b".into()), &enc("json", HolePosition::Value, Some("str"), None, false), &mut out);
        assert_eq!(out, "\"a\\\"b\"");
        out.clear();
        encode(&Value::U64(7), &enc("json", HolePosition::Value, None, None, false), &mut out);
        assert_eq!(out, "7");
        out.clear();
        encode(&Value::Str("x\ny".into()), &enc("json", HolePosition::InString, None, None, false), &mut out);
        assert_eq!(out, "x\\ny");
        out.clear();
        encode(&Value::F64(2.0 / 3.0), &enc("json", HolePosition::Value, None, Some(".2"), false), &mut out);
        assert_eq!(out, "0.67");
        out.clear();
        encode(&Value::None, &enc("json", HolePosition::Value, None, None, false), &mut out);
        assert_eq!(out, "null");
    }

    #[test]
    fn spec_round_trips() {
        let e = enc("json", HolePosition::InString, Some("u64"), Some(".2"), true);
        assert_eq!(HoleEncoding::from_spec(&e.to_spec()), e);
        let c = HoleEncoding { cond: true, ..enc("text", HolePosition::Text, None, None, false) };
        assert_eq!(HoleEncoding::from_spec(&c.to_spec()), c);
    }

    #[test]
    fn csv_quotes_when_needed_and_raw_skips_escaping() {
        let mut out = String::new();
        encode(&Value::Str("a,b".into()), &enc("csv", HolePosition::Text, None, None, false), &mut out);
        assert_eq!(out, "\"a,b\"");
        out.clear();
        encode(&Value::Str("a\"b".into()), &enc("json", HolePosition::Value, None, None, true), &mut out);
        assert_eq!(out, "a\"b");
    }

    #[test]
    fn formats_apply_before_encoding() {
        assert_eq!(formatted_text(&Value::U64(5), None, Some("03")), "005");
        assert_eq!(formatted_text(&Value::U64(255), None, Some("x")), "ff");
        assert_eq!(formatted_text(&Value::Str("ab".into()), None, Some(">4")), "  ab");
    }
}
