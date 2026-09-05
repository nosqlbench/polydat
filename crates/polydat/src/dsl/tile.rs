// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! The tile template grammar (SRD 114 §2.2): holes, projections,
//! branches, and the doubled-open escape, parsed over configurable
//! delimiters and sigil. The textual front end for Polytile; the
//! structural front end (§3) lowers to the same pieces.

use crate::ast::PortType;

use super::ast::{Expr, TileHole, TileOptions, TilePiece};
use super::lexer::{lex, Span};
use super::parser::{for_source_from_text, parse_expression};

/// Parse a tile body into pieces under `opts`. `span` is the tile
/// statement's position, used for diagnostics.
pub fn parse_template(text: &str, opts: &TileOptions, span: Span) -> Result<Vec<TilePiece>, String> {
    let mut p = TemplateParser { chars: text.chars().collect(), pos: 0, opts, span };
    let pieces = p.pieces(false)?;
    if p.pos < p.chars.len() {
        return Err(p.err("unexpected `}` closing a block that was never opened"));
    }
    Ok(pieces)
}

struct TemplateParser<'a> {
    chars: Vec<char>,
    pos: usize,
    opts: &'a TileOptions,
    span: Span,
}

impl TemplateParser<'_> {
    fn err(&self, msg: &str) -> String {
        format!(
            "tile at line {}, col {}: {msg} (template offset {})",
            self.span.line, self.span.col, self.pos
        )
    }

    fn starts_with(&self, s: &str) -> bool {
        let sc: Vec<char> = s.chars().collect();
        self.chars[self.pos..].starts_with(&sc)
    }

    fn take(&mut self, s: &str) {
        self.pos += s.chars().count();
    }

    fn rest(&self) -> String {
        self.chars[self.pos..].iter().collect()
    }

    /// Parse pieces until end of text, or until an unmatched `}` when
    /// `in_block` (the caller consumes it).
    fn pieces(&mut self, in_block: bool) -> Result<Vec<TilePiece>, String> {
        let open = self.opts.open.clone();
        let doubled = format!("{open}{open}");
        let sigil = self.opts.sigil.clone();
        let mut out: Vec<TilePiece> = Vec::new();
        let mut static_buf = String::new();
        // Braces inside static text (JSON objects, for instance) are
        // balanced within the block; only an unmatched `}` ends it.
        let mut static_depth = 0i32;
        let flush = |buf: &mut String, out: &mut Vec<TilePiece>| {
            if !buf.is_empty() {
                out.push(TilePiece::Static(std::mem::take(buf)));
            }
        };
        while self.pos < self.chars.len() {
            if self.starts_with(&doubled) {
                self.take(&doubled);
                static_buf.push_str(&open);
                continue;
            }
            if self.starts_with(&open) {
                flush(&mut static_buf, &mut out);
                out.push(TilePiece::Hole(self.hole()?));
                continue;
            }
            if self.starts_with(&sigil) {
                let after: String = self.rest().chars().skip(sigil.chars().count()).collect();
                if after.starts_with("for") && after[3..].starts_with(char::is_whitespace) {
                    flush(&mut static_buf, &mut out);
                    out.push(self.projection()?);
                    continue;
                }
                if after.starts_with("if") && after[2..].starts_with(char::is_whitespace) {
                    flush(&mut static_buf, &mut out);
                    out.push(self.branch()?);
                    continue;
                }
            }
            let c = self.chars[self.pos];
            if c == '{' {
                static_depth += 1;
            } else if c == '}' {
                if in_block && static_depth == 0 {
                    break;
                }
                static_depth -= 1;
            }
            static_buf.push(c);
            self.pos += 1;
        }
        flush(&mut static_buf, &mut out);
        Ok(out)
    }

    /// `${ expr [: type] [| format] [!] }`. Positioned at the open delimiter.
    fn hole(&mut self) -> Result<TileHole, String> {
        let start = self.pos;
        self.take(&self.opts.open.clone());
        let inner = self.until_close()?;
        let text = inner.trim().to_string();
        if text.is_empty() {
            return Err(self.err("empty hole"));
        }
        let (mut body, raw) = match text.strip_suffix('!') {
            Some(b) => (b.trim().to_string(), true),
            None => (text.clone(), false),
        };
        let mut format = None;
        if let Some(idx) = rfind_top_level(&body, '|')
            && !body[..idx].ends_with('|')
            && body[idx + 1..].trim().chars().all(|c| c.is_ascii_alphanumeric() || ".<>^-+#0_".contains(c))
            && !body[idx + 1..].trim().is_empty()
        {
            format = Some(body[idx + 1..].trim().to_string());
            body = body[..idx].trim().to_string();
        }
        let mut decl_type = None;
        if let Some(idx) = rfind_top_level(&body, ':')
            && let Some(kw) = body.get(idx + 1..).map(str::trim)
            && PortType::from_keyword(kw).is_some()
        {
            decl_type = Some(kw.to_string());
            body = body[..idx].trim().to_string();
        }
        if body.is_empty() {
            return Err(self.err(&format!("hole `{text}` has no expression")));
        }
        let expr = parse_hole_expr(&body).map_err(|e| {
            // `${x: integer}`: the suffix looked like a declaration but
            // is not a type keyword, so it stayed in the expression.
            if let Some(idx) = rfind_top_level(&body, ':')
                && let Some(word) = body.get(idx + 1..).map(str::trim)
                && !word.is_empty()
                && word.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
            {
                return self.err(&format!(
                    "hole `{text}`: unknown type '{word}'; types are the port-type keywords (u64, i64, f64, str, bool, json, bytes, ...)"
                ));
            }
            self.err(&format!("hole `{text}`: {e}"))
        })?;
        let _ = start;
        Ok(TileHole { text, expr, decl_type, format, raw, span: self.span })
    }

    /// Consume up to and including the close delimiter at bracket depth
    /// zero, returning the enclosed text.
    fn until_close(&mut self) -> Result<String, String> {
        let close = self.opts.close.clone();
        let mut depth = 0i32;
        let mut quote: Option<char> = None;
        let mut buf = String::new();
        while self.pos < self.chars.len() {
            let c = self.chars[self.pos];
            if let Some(q) = quote {
                buf.push(c);
                self.pos += 1;
                if c == '\\' && self.pos < self.chars.len() {
                    buf.push(self.chars[self.pos]);
                    self.pos += 1;
                } else if c == q {
                    quote = None;
                }
                continue;
            }
            if depth == 0 && self.starts_with(&close) {
                self.take(&close);
                return Ok(buf);
            }
            match c {
                '"' | '\'' => quote = Some(c),
                '(' | '[' | '{' => depth += 1,
                ')' | ']' | '}' => depth -= 1,
                _ => {}
            }
            buf.push(c);
            self.pos += 1;
        }
        Err(self.err(&format!("unterminated hole; expected `{close}`")))
    }

    /// `@for <source> [sep "..."] { body }`. Positioned at the sigil.
    fn projection(&mut self) -> Result<TilePiece, String> {
        self.take(&self.opts.sigil.clone());
        self.take("for");
        let header = self.header_until_brace()?;
        let (source_text, sep) = split_sep(&header);
        let source = for_source_from_text(source_text.trim(), self.span, true)
            .map_err(|e| self.err(&format!("projection: {e}")))?;
        let body = self.block()?;
        Ok(TilePiece::Projection { source, sep, body, span: self.span })
    }

    /// `@if cond { body } [@else { body }]`. Positioned at the sigil.
    fn branch(&mut self) -> Result<TilePiece, String> {
        self.take(&self.opts.sigil.clone());
        self.take("if");
        let header = self.header_until_brace()?;
        let cond = parse_hole_expr(header.trim()).map_err(|e| self.err(&format!("branch condition `{}`: {e}", header.trim())))?;
        let then = self.block()?;
        let save = self.pos;
        self.skip_ws();
        let else_kw = format!("{}else", self.opts.sigil);
        let otherwise = if self.starts_with(&else_kw) {
            self.take(&else_kw);
            self.skip_ws();
            Some(self.block()?)
        } else {
            self.pos = save;
            None
        };
        Ok(TilePiece::Branch { cond, then, otherwise, span: self.span })
    }

    /// Text up to a `{` at bracket depth zero that is not a `{name}`
    /// placeholder. A hole cannot appear in a directive header, so the
    /// open delimiter is not considered here. Leaves the brace
    /// unconsumed.
    fn header_until_brace(&mut self) -> Result<String, String> {
        let mut depth = 0i32;
        let mut quote: Option<char> = None;
        let mut buf = String::new();
        while self.pos < self.chars.len() {
            let c = self.chars[self.pos];
            if let Some(q) = quote {
                buf.push(c);
                self.pos += 1;
                if c == q { quote = None; }
                continue;
            }
            // A hole cannot appear in a directive header; a delimiter
            // here means the block never came. (Delimiters that begin
            // with `{` are checked as blocks below instead.)
            if depth == 0 && !self.opts.open.starts_with('{') && self.starts_with(&self.opts.open.clone()) {
                return Err(self.err("directive has no `{` block; a hole cannot appear in a directive header"));
            }
            if depth == 0 && c == '{' {
                // `{name}` is an interpolation placeholder only when it
                // is attached to header text (`1..{n}`, `{a}..{b}`). A
                // free-standing `{name}` after whitespace is the block:
                // `@if x {plain}`.
                // A free-standing one followed by an operator continues
                // the header (`where {k} > 0 {`); followed by whitespace
                // and then text, a sigil, a delimiter, or another `{`,
                // it is the block itself.
                const HEADER_PUNCT: &str = "<>=!+-*/%.,()[]&|?:";
                let placeholder = placeholder_len(&self.chars, self.pos).is_some_and(|len| {
                    let before = buf.chars().last().is_some_and(|b| HEADER_PUNCT.contains(b));
                    let after = self.chars.get(self.pos + len).is_some_and(|a| HEADER_PUNCT.contains(*a));
                    if before || after {
                        return true;
                    }
                    let rest: String = self.chars[self.pos + len..].iter().collect();
                    let rest = rest.trim_start();
                    if rest.is_empty() || rest.starts_with(&self.opts.sigil) || rest.starts_with(&self.opts.open) {
                        return false;
                    }
                    rest.starts_with(|c: char| HEADER_PUNCT.contains(c) && !"()[]".contains(c))
                });
                if !placeholder {
                    return Ok(buf);
                }
            }
            match c {
                '"' | '\'' => quote = Some(c),
                '(' | '[' => depth += 1,
                ')' | ']' => depth -= 1,
                _ => {}
            }
            buf.push(c);
            self.pos += 1;
        }
        Err(self.err("directive has no `{` block"))
    }

    /// A `{ ... }` block; returns its pieces. Positioned at `{`.
    fn block(&mut self) -> Result<Vec<TilePiece>, String> {
        if self.pos >= self.chars.len() || self.chars[self.pos] != '{' {
            return Err(self.err("expected `{`"));
        }
        self.pos += 1;
        let mut body = self.pieces(true)?;
        if self.pos >= self.chars.len() || self.chars[self.pos] != '}' {
            return Err(self.err("unterminated block; expected `}`"));
        }
        self.pos += 1;
        trim_block(&mut body);
        Ok(body)
    }

    fn skip_ws(&mut self) {
        while self.pos < self.chars.len() && self.chars[self.pos].is_whitespace() {
            self.pos += 1;
        }
    }
}

/// The braces of a directive block delimit it; the whitespace that pads
/// them for readability is not part of the body. `@if x { "hot" }`
/// renders `"hot"`, and `@for` items concatenate without stray blanks.
fn trim_block(body: &mut Vec<TilePiece>) {
    if let Some(TilePiece::Static(s)) = body.first_mut() {
        let t = s.trim_start().to_string();
        *s = t;
    }
    if let Some(TilePiece::Static(s)) = body.last_mut() {
        let t = s.trim_end().to_string();
        *s = t;
    }
    body.retain(|p| !matches!(p, TilePiece::Static(s) if s.is_empty()));
}

/// Split a trailing `sep "<text>"` off a directive header. Inside a JSON
/// string literal the quotes arrive escaped as `\"`; both spellings are
/// accepted.
fn split_sep(header: &str) -> (String, Option<String>) {
    let unescaped = header.replace("\\\"", "\"");
    let t = unescaped.trim_end();
    if let Some(q) = t.strip_suffix('"')
        && let Some(open_quote) = q.rfind('"')
        && q[..open_quote].trim_end().ends_with(" sep")
    {
        let sep = q[open_quote + 1..].to_string();
        let head = q[..open_quote].trim_end();
        let head = head[..head.len() - 3].trim_end();
        return (head.to_string(), Some(sep));
    }
    (t.to_string(), None)
}

fn parse_hole_expr(text: &str) -> Result<Expr, String> {
    let tokens = lex(text)?;
    parse_expression(tokens)
}

/// Index of the last `needle` at bracket depth zero and outside quotes.
fn rfind_top_level(s: &str, needle: char) -> Option<usize> {
    let mut depth = 0i32;
    let mut quote: Option<char> = None;
    let mut found = None;
    for (i, c) in s.char_indices() {
        if let Some(q) = quote {
            if c == q { quote = None; }
            continue;
        }
        match c {
            '"' | '\'' => quote = Some(c),
            '(' | '[' | '{' => depth += 1,
            ')' | ']' | '}' => depth -= 1,
            _ if depth == 0 && c == needle => found = Some(i),
            _ => {}
        }
    }
    found
}

/// Length of a `{identifier}` placeholder at `pos`, if present.
fn placeholder_len(chars: &[char], pos: usize) -> Option<usize> {
    let mut i = pos + 1;
    let first = *chars.get(i)?;
    if !(first.is_ascii_alphabetic() || first == '_') {
        return None;
    }
    while i < chars.len() && (chars[i].is_ascii_alphanumeric() || chars[i] == '_') {
        i += 1;
    }
    (chars.get(i) == Some(&'}')).then_some(i + 1 - pos)
}

/// Render pieces back to template text under `opts`, the inverse of
/// [`parse_template`] up to whitespace inside directives.
pub fn render_template(pieces: &[TilePiece], opts: &TileOptions) -> String {
    let mut out = String::new();
    for piece in pieces {
        match piece {
            TilePiece::Static(s) => out.push_str(&s.replace(&opts.open, &format!("{}{}", opts.open, opts.open))),
            TilePiece::Hole(h) => {
                out.push_str(&opts.open);
                out.push_str(&h.text);
                out.push_str(&opts.close);
            }
            TilePiece::Projection { source, sep, body, .. } => {
                out.push_str(&opts.sigil);
                out.push_str("for ");
                out.push_str(&source.text);
                if let Some(s) = sep {
                    out.push_str(&format!(" sep \"{s}\""));
                }
                out.push_str(" { ");
                out.push_str(&render_template(body, opts));
                out.push_str(" }");
            }
            TilePiece::Branch { cond, then, otherwise, .. } => {
                out.push_str(&opts.sigil);
                out.push_str("if ");
                out.push_str(&super::pprint::pp_expr(cond));
                out.push_str(" { ");
                out.push_str(&render_template(then, opts));
                out.push_str(" }");
                if let Some(o) = otherwise {
                    out.push(' ');
                    out.push_str(&opts.sigil);
                    out.push_str("else { ");
                    out.push_str(&render_template(o, opts));
                    out.push_str(" }");
                }
            }
        }
    }
    out
}
