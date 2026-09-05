// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! The structural front end and the host boundary of Polytile (SRD 114
//! §3, §5.6).
//!
//! A host may hold a template as text, as JSON text, or as a JSON value
//! it has already parsed. All three arrive here and leave as a
//! [`TileDef`], the same statement the `tile` keyword produces, so the
//! compiler has one tile to compile. The structural form is turned into
//! template text first: strings become value holes, string holes, or
//! statics by the rules of §3.1, and directive arrays and objects
//! become `@for` and `@if` blocks by the rules of §3.2. The textual
//! parser then yields the pieces, which is what makes the two forms
//! equivalent by construction.

use serde_json::{Map, Value};

use super::ast::{TileBodyKind, TileDef, TileOptions, TilePiece};
use super::lexer::Span;
use super::tile::parse_template;

/// The encodings a tile may declare.
pub const ENCODINGS: &[&str] = &["json", "text", "csv"];

fn check_encoding(name: &str, encoding: &str) -> Result<(), String> {
    if ENCODINGS.contains(&encoding) {
        Ok(())
    } else {
        Err(format!("tile '{name}': unknown encoding '{encoding}'; encodings are json, text, csv"))
    }
}

/// A tile from template text a host holds: the body of a `tile`
/// statement without the statement. `text` may be multi-line; it is
/// taken exactly, as a heredoc body is.
pub fn tile_from_text(name: &str, encoding: &str, text: &str, options: &TileOptions, span: Span) -> Result<TileDef, String> {
    check_encoding(name, encoding)?;
    let pieces = parse_template(text, options, span).map_err(|e| format!("tile '{name}': {e}"))?;
    Ok(TileDef {
        name: name.to_string(),
        encoding: Some(encoding.to_string()),
        options: options.clone(),
        body_kind: TileBodyKind::Heredoc,
        body: text.to_string(),
        pieces,
        span,
    })
}

/// A tile from a structural template in JSON text (§3). The encoding is
/// `json`.
pub fn tile_from_json_text(name: &str, json: &str, options: &TileOptions, span: Span) -> Result<TileDef, String> {
    let value: Value =
        serde_json::from_str(json).map_err(|e| format!("tile '{name}': structural template is not JSON: {e}"))?;
    tile_from_json_value(name, &value, options, span)
}

/// A tile from a structural template a host has already parsed (§3).
/// The encoding is `json`.
pub fn tile_from_json_value(name: &str, value: &Value, options: &TileOptions, span: Span) -> Result<TileDef, String> {
    let body = template_text_from_value(value, options).map_err(|e| format!("tile '{name}': {e}"))?;
    let pieces = parse_template(&body, options, span).map_err(|e| format!("tile '{name}': {e}"))?;
    Ok(TileDef {
        name: name.to_string(),
        encoding: Some("json".to_string()),
        options: options.clone(),
        body_kind: TileBodyKind::Block,
        body,
        pieces,
        span,
    })
}

/// Turn a structural template into template text under `options`.
/// The text is what an author would have written in a `tile ... : json`
/// block for the same document.
pub fn template_text_from_value(value: &Value, options: &TileOptions) -> Result<String, String> {
    let mut t = Textualizer { opts: options, out: String::new() };
    t.value(value)?;
    Ok(t.out)
}

struct Textualizer<'a> {
    opts: &'a TileOptions,
    out: String,
}

/// Where a directive member's separating comma goes.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Comma {
    /// A static member: the ordinary `, ` between members.
    Between,
    /// A directive with a member before it: `, ` opens each repetition.
    Leading,
    /// A directive with members after it: `, ` closes each repetition.
    Trailing,
    /// A lone directive: the encoding's separator between repetitions.
    Sep,
}

/// A directive string: `@for <header>`, `@if <cond>`, or `@else`.
enum Directive<'s> {
    For(&'s str),
    If(&'s str),
    Else,
}

impl Textualizer<'_> {
    fn directive<'s>(&self, s: &'s str) -> Option<Directive<'s>> {
        let rest = s.trim().strip_prefix(self.opts.sigil.as_str())?;
        if let Some(h) = rest.strip_prefix("for")
            && h.starts_with(char::is_whitespace)
        {
            return Some(Directive::For(h.trim()));
        }
        if let Some(c) = rest.strip_prefix("if")
            && c.starts_with(char::is_whitespace)
        {
            return Some(Directive::If(c.trim()));
        }
        if rest == "else" {
            return Some(Directive::Else);
        }
        None
    }

    fn value(&mut self, v: &Value) -> Result<(), String> {
        match v {
            Value::Null | Value::Bool(_) | Value::Number(_) => self.out.push_str(&v.to_string()),
            Value::String(s) => self.string(s, false)?,
            Value::Array(items) => self.array(items)?,
            Value::Object(map) => self.object(map)?,
        }
        Ok(())
    }

    /// A string node (§3.1). As a member name, `key` is set and the
    /// string is always in string position.
    fn string(&mut self, s: &str, key: bool) -> Result<(), String> {
        if self.directive(s).is_some() {
            return Err(format!(
                "directive string `{s}` outside a directive position; a `{}for` or `{}if` string leads an array, or is an object key",
                self.opts.sigil, self.opts.sigil
            ));
        }
        let pieces = parse_template(s, self.opts, Span { line: 0, col: 0 }).map_err(|e| format!("in string `{s}`: {e}"))?;
        if pieces.iter().any(|p| matches!(p, TilePiece::Projection { .. } | TilePiece::Branch { .. })) {
            return Err(format!(
                "string `{s}` contains a directive; in the structural form directives are arrays and object keys"
            ));
        }
        if !key
            && let [TilePiece::Hole(h)] = pieces.as_slice()
            && h.decl_type.as_deref() != Some("str")
        {
            // A value hole: the node is the value, encoded by type.
            self.out.push_str(&self.opts.open);
            self.out.push_str(&h.text);
            self.out.push_str(&self.opts.close);
            return Ok(());
        }
        // A string hole or a static string.
        self.out.push('"');
        for piece in &pieces {
            match piece {
                TilePiece::Static(text) => {
                    let escaped = serde_json::to_string(text).expect("string serializes");
                    let inner = &escaped[1..escaped.len() - 1];
                    self.out.push_str(&inner.replace(&self.opts.open, &format!("{}{}", self.opts.open, self.opts.open)));
                }
                TilePiece::Hole(h) => {
                    self.out.push_str(&self.opts.open);
                    self.out.push_str(&h.text);
                    self.out.push_str(&self.opts.close);
                }
                _ => unreachable!("directives rejected above"),
            }
        }
        self.out.push('"');
        Ok(())
    }

    fn array(&mut self, items: &[Value]) -> Result<(), String> {
        self.out.push('[');
        match items.first().and_then(|f| f.as_str()).and_then(|s| self.directive(s)) {
            Some(Directive::For(header)) => {
                self.out.push_str(&format!("{}for {header} {{ ", self.opts.sigil));
                self.items(&items[1..])?;
                self.out.push_str(" }");
            }
            Some(Directive::If(cond)) => {
                let split = items[1..]
                    .iter()
                    .position(|v| v.as_str().is_some_and(|s| matches!(self.directive(s), Some(Directive::Else))));
                let (then, otherwise) = match split {
                    Some(i) => (&items[1..1 + i], Some(&items[2 + i..])),
                    None => (&items[1..], None),
                };
                self.out.push_str(&format!("{}if {cond} {{ ", self.opts.sigil));
                self.items(then)?;
                self.out.push_str(" }");
                if let Some(o) = otherwise {
                    self.out.push_str(&format!(" {}else {{ ", self.opts.sigil));
                    self.items(o)?;
                    self.out.push_str(" }");
                }
            }
            Some(Directive::Else) => return Err(format!("`{}else` without a leading `{}if`", self.opts.sigil, self.opts.sigil)),
            None => self.items(items)?,
        }
        self.out.push(']');
        Ok(())
    }

    fn items(&mut self, items: &[Value]) -> Result<(), String> {
        for (i, item) in items.iter().enumerate() {
            if i > 0 {
                self.out.push_str(", ");
            }
            self.value(item)?;
        }
        Ok(())
    }

    /// An object. A directive member beside static members carries the
    /// comma that separates it inside its own body, so a projection that
    /// renders zero tuples or a branch that renders nothing leaves no
    /// dangling separator: a directive with a member before it puts the
    /// comma first in every repetition, a leading directive with members
    /// after it puts the comma last, and a lone directive uses the
    /// encoding's separator.
    fn object(&mut self, map: &Map<String, Value>) -> Result<(), String> {
        self.out.push('{');
        let entries: Vec<(&String, &Value)> = map.iter().collect();
        let mut i = 0;
        let mut emitted_static = false;
        while i < entries.len() {
            let (key, value) = entries[i];
            let directive = self.directive(key);
            let is_directive = directive.is_some();
            let has_before = emitted_static;
            // Any later member other than this directive's own `@else`.
            let has_after = entries[i + 1..].iter().any(|(k, _)| !matches!(self.directive(k), Some(Directive::Else)));
            let comma = match (is_directive, has_before, has_after) {
                (false, _, _) => Comma::Between,
                (true, true, _) => Comma::Leading,
                (true, false, true) => Comma::Trailing,
                (true, false, false) => Comma::Sep,
            };
            if !is_directive && emitted_static {
                self.out.push_str(", ");
            }
            match directive {
                Some(Directive::For(header)) => {
                    // A member projection: the value's members, per tuple.
                    self.out.push_str(&format!("{}for {header}", self.opts.sigil));
                    if matches!(comma, Comma::Leading | Comma::Trailing) {
                        self.out.push_str(" sep \"\"");
                    }
                    self.out.push_str(" { ");
                    self.members(key, value, comma)?;
                    self.out.push_str(" }");
                }
                Some(Directive::If(cond)) => {
                    self.out.push_str(&format!("{}if {cond} {{ ", self.opts.sigil));
                    self.members(key, value, comma)?;
                    self.out.push_str(" }");
                    if let Some((next_key, next_value)) = entries.get(i + 1)
                        && matches!(self.directive(next_key), Some(Directive::Else))
                    {
                        self.out.push_str(&format!(" {}else {{ ", self.opts.sigil));
                        self.members(next_key, next_value, comma)?;
                        self.out.push_str(" }");
                        i += 1;
                    }
                }
                Some(Directive::Else) => {
                    return Err(format!("`{}else` key without a preceding `{}if` key", self.opts.sigil, self.opts.sigil))
                }
                None => {
                    self.string(key, true)?;
                    self.out.push_str(": ");
                    self.value(value)?;
                    emitted_static = true;
                }
            }
            i += 1;
        }
        self.out.push('}');
        Ok(())
    }

    /// The members of a directive key's object value, as `"k": v` runs,
    /// with the separator the position calls for.
    fn members(&mut self, key: &str, value: &Value, comma: Comma) -> Result<(), String> {
        let Some(obj) = value.as_object() else {
            return Err(format!("the value under directive key `{key}` must be an object of members"));
        };
        if matches!(comma, Comma::Leading) {
            self.out.push_str(", ");
        }
        for (i, (k, v)) in obj.iter().enumerate() {
            if i > 0 {
                self.out.push_str(", ");
            }
            self.string(k, true)?;
            self.out.push_str(": ");
            self.value(v)?;
        }
        if matches!(comma, Comma::Trailing) {
            self.out.push_str(", ");
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text(json: &str) -> String {
        let v: Value = serde_json::from_str(json).unwrap();
        template_text_from_value(&v, &TileOptions::default()).unwrap()
    }

    #[test]
    fn strings_classify_as_value_string_or_static() {
        assert_eq!(text(r#"{"ts": "${ts}", "name": "row-${row}", "s": "${ts: str}", "k": "plain"}"#),
            r#"{"ts": ${ts}, "name": "row-${row}", "s": "${ts: str}", "k": "plain"}"#);
    }

    #[test]
    fn directive_arrays_and_objects_become_blocks() {
        assert_eq!(text(r#"["@for s in 0..4", {"n": "${s}"}]"#), r#"[@for s in 0..4 { {"n": ${s}} }]"#);
        assert_eq!(text(r#"["@if v", 1, "@else", 2]"#), r#"[@if v { 1 } @else { 2 }]"#);
        assert_eq!(text(r#"{"@for t in a,b": {"${t}": true}}"#), r#"{@for t in a,b { "${t}": true }}"#);
    }

    #[test]
    fn scalars_and_escapes_pass_through() {
        assert_eq!(text(r#"{"a": null, "b": true, "c": 1.5, "d": "q\"uote ${x}"}"#), r#"{"a": null, "b": true, "c": 1.5, "d": "q\"uote ${x}"}"#);
    }

    #[test]
    fn directive_in_value_position_is_an_error() {
        let v: Value = serde_json::from_str(r#"{"a": "@for s in 0..4"}"#).unwrap();
        let e = template_text_from_value(&v, &TileOptions::default()).unwrap_err();
        assert!(e.contains("outside a directive position"), "{e}");
    }
}
