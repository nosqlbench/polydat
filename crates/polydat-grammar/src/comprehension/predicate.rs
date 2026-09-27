// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! A comprehension's `where` predicate, parsed to a tree.
//!
//! A predicate is a Polydat boolean expression over the tuple's
//! elements, each written `{name}` (comprehension_forms.md §10.9.1). It
//! parses with the language's one precedence table,
//! [`crate::parser::binding_power`]: unary `!` and `-` bind tighter
//! than every binary operator, arithmetic binds tighter than
//! comparison, comparison tighter than `&&`, and `&&` tighter than
//! `||`, so `!{done} || {retry}` is `(!{done}) || {retry}`. Membership,
//! `{name} in [v1, v2, …]`, binds as a relational comparison.
//!
//! The tree holds the structure the runtime evaluates directly and the
//! predicate analyzer factorizes: `||`, `&&`, `!`, the six comparisons,
//! and membership, over element references and literals. Any other
//! expression (arithmetic, a function call, a cast) is an
//! [`PredicateKind::Expr`] leaf, which the runtime evaluates as a
//! Polydat expression from its text. Every node carries the byte range
//! of its text in the predicate, enclosing parentheses included.

use std::ops::Range;

use crate::ast::BinOpKind;
use crate::parser::binding_power;

/// One node of a parsed predicate and the byte range of its text.
#[derive(Debug, Clone, PartialEq)]
pub struct Predicate {
    /// What the node is.
    pub kind: PredicateKind,
    /// Its text's byte range in the predicate, enclosing parentheses
    /// included.
    pub span: Range<usize>,
}

/// The kinds of predicate node.
#[derive(Debug, Clone, PartialEq)]
pub enum PredicateKind {
    /// `p || q || …`: true when any operand is.
    Or(Vec<Predicate>),
    /// `p && q && …`: true when every operand is.
    And(Vec<Predicate>),
    /// `!p`: true when the operand is not.
    Not(Box<Predicate>),
    /// `a OP b` for one of the six comparisons.
    Compare(Comparison, Box<Predicate>, Box<Predicate>),
    /// `a in [v1, v2, …]`: true when `a` equals any item.
    In(Box<Predicate>, Vec<Predicate>),
    /// `{name}`: the value the tuple, or the scope it was drawn in,
    /// binds to `name`.
    Element(String),
    /// A literal value. A bare word is its own text.
    Literal(PredicateLiteral),
    /// Any other Polydat expression, evaluated from its text.
    Expr,
}

/// A comparison operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Comparison {
    /// `==`
    Eq,
    /// `!=`
    Ne,
    /// `<`
    Lt,
    /// `<=`
    Le,
    /// `>`
    Gt,
    /// `>=`
    Ge,
}

impl Comparison {
    /// The operator's spelling.
    pub fn as_str(self) -> &'static str {
        match self {
            Comparison::Eq => "==",
            Comparison::Ne => "!=",
            Comparison::Lt => "<",
            Comparison::Le => "<=",
            Comparison::Gt => ">",
            Comparison::Ge => ">=",
        }
    }

    /// The comparison with its operands swapped: `a < b` is `b > a`.
    pub fn swapped(self) -> Self {
        match self {
            Comparison::Lt => Comparison::Gt,
            Comparison::Le => Comparison::Ge,
            Comparison::Gt => Comparison::Lt,
            Comparison::Ge => Comparison::Le,
            other => other,
        }
    }
}

/// A literal in a predicate.
#[derive(Debug, Clone, PartialEq)]
pub enum PredicateLiteral {
    /// An integer, negative when written with a leading `-`.
    Int(i128),
    /// A float.
    Float(f64),
    /// A quoted string, or a bare word such as `load`.
    Str(String),
    /// `true` or `false`.
    Bool(bool),
}

impl Predicate {
    /// This node's text in `source`, the predicate it was parsed from.
    pub fn text<'a>(&self, source: &'a str) -> &'a str {
        &source[self.span.clone()]
    }
}

/// Parse a predicate. An error names what the predicate grammar does
/// not accept; the caller may still evaluate the whole text as a
/// Polydat expression.
pub fn parse_predicate(text: &str) -> Result<Predicate, String> {
    let tokens = tokenize(text)?;
    let mut parser = Parser { tokens, pos: 0 };
    let predicate = parser.expression(0)?;
    match parser.tokens.get(parser.pos) {
        None => Ok(predicate),
        Some(token) => Err(format!(
            "unexpected `{}` in predicate `{text}`",
            &text[token.span.clone()]
        )),
    }
}

#[derive(Debug, Clone)]
enum Tok {
    Element(String),
    Ident(String),
    Int(i128),
    Float(f64),
    Str(String),
    Op(BinOpKind),
    Bang,
    LParen,
    RParen,
    LBracket,
    RBracket,
    Comma,
    Other,
}

#[derive(Debug, Clone)]
struct Token {
    tok: Tok,
    span: Range<usize>,
}

fn is_ident_start(c: char) -> bool {
    c.is_ascii_alphabetic() || c == '_'
}

fn is_ident_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_'
}

fn tokenize(text: &str) -> Result<Vec<Token>, String> {
    const OPERATORS: [(&str, Option<BinOpKind>); 22] = [
        ("||", Some(BinOpKind::Or)),
        ("&&", Some(BinOpKind::And)),
        ("==", Some(BinOpKind::Eq)),
        ("!=", Some(BinOpKind::Ne)),
        ("<=", Some(BinOpKind::Le)),
        (">=", Some(BinOpKind::Ge)),
        ("<<", Some(BinOpKind::Shl)),
        (">>", Some(BinOpKind::Shr)),
        ("**", Some(BinOpKind::Pow)),
        ("|", Some(BinOpKind::BitOr)),
        ("&", Some(BinOpKind::BitAnd)),
        ("^", Some(BinOpKind::BitXor)),
        ("<", Some(BinOpKind::Lt)),
        (">", Some(BinOpKind::Gt)),
        ("+", Some(BinOpKind::Add)),
        ("-", Some(BinOpKind::Sub)),
        ("*", Some(BinOpKind::Mul)),
        ("/", Some(BinOpKind::Div)),
        ("%", Some(BinOpKind::Mod)),
        ("!", None),
        ("(", None),
        (")", None),
    ];
    let mut tokens = Vec::new();
    let bytes = text.as_bytes();
    let mut i = 0;
    'scan: while i < text.len() {
        let c = text[i..].chars().next().expect("in bounds");
        let start = i;
        if c.is_whitespace() {
            i += c.len_utf8();
            continue;
        }
        if c == '{' {
            let rest = &text[i + 1..];
            let name_len = rest
                .find(|ch: char| !is_ident_char(ch))
                .unwrap_or(rest.len());
            if name_len > 0 && rest[name_len..].starts_with('}') {
                tokens.push(Token {
                    tok: Tok::Element(rest[..name_len].to_string()),
                    span: start..i + name_len + 2,
                });
                i += name_len + 2;
                continue;
            }
        }
        if c == '"' || c == '\'' {
            let mut value = String::new();
            let mut j = i + 1;
            loop {
                let Some(ch) = text[j..].chars().next() else {
                    return Err(format!("unterminated string in predicate `{text}`"));
                };
                j += ch.len_utf8();
                if ch == c {
                    break;
                }
                if ch == '\\'
                    && let Some(escaped) = text[j..].chars().next()
                {
                    value.push(escaped);
                    j += escaped.len_utf8();
                    continue;
                }
                value.push(ch);
            }
            tokens.push(Token {
                tok: Tok::Str(value),
                span: start..j,
            });
            i = j;
            continue;
        }
        if c.is_ascii_digit() {
            let mut j = i;
            while j < text.len() && bytes[j].is_ascii_digit() {
                j += 1;
            }
            let mut float = false;
            if j + 1 < text.len() && bytes[j] == b'.' && bytes[j + 1].is_ascii_digit() {
                float = true;
                j += 1;
                while j < text.len() && bytes[j].is_ascii_digit() {
                    j += 1;
                }
            }
            if j < text.len() && (bytes[j] == b'e' || bytes[j] == b'E') {
                let mut k = j + 1;
                if k < text.len() && (bytes[k] == b'+' || bytes[k] == b'-') {
                    k += 1;
                }
                if k < text.len() && bytes[k].is_ascii_digit() {
                    float = true;
                    j = k;
                    while j < text.len() && bytes[j].is_ascii_digit() {
                        j += 1;
                    }
                }
            }
            let digits = &text[i..j];
            let tok = if float {
                Tok::Float(
                    digits
                        .parse()
                        .map_err(|e| format!("bad number `{digits}`: {e}"))?,
                )
            } else {
                Tok::Int(
                    digits
                        .parse()
                        .map_err(|e| format!("bad number `{digits}`: {e}"))?,
                )
            };
            tokens.push(Token { tok, span: i..j });
            i = j;
            continue;
        }
        if is_ident_start(c) {
            let rest = &text[i..];
            let len = rest
                .find(|ch: char| !is_ident_char(ch))
                .unwrap_or(rest.len());
            tokens.push(Token {
                tok: Tok::Ident(rest[..len].to_string()),
                span: i..i + len,
            });
            i += len;
            continue;
        }
        for (spelling, op) in OPERATORS {
            if text[i..].starts_with(spelling) {
                let tok = match (spelling, op) {
                    (_, Some(op)) => Tok::Op(op),
                    ("!", None) => Tok::Bang,
                    ("(", None) => Tok::LParen,
                    _ => Tok::RParen,
                };
                tokens.push(Token {
                    tok,
                    span: i..i + spelling.len(),
                });
                i += spelling.len();
                continue 'scan;
            }
        }
        let tok = match c {
            '[' => Tok::LBracket,
            ']' => Tok::RBracket,
            ',' => Tok::Comma,
            _ => Tok::Other,
        };
        tokens.push(Token {
            tok,
            span: i..i + c.len_utf8(),
        });
        i += c.len_utf8();
    }
    Ok(tokens)
}

struct Parser {
    tokens: Vec<Token>,
    pos: usize,
}

/// An infix operator of the predicate grammar.
#[derive(Clone, Copy)]
enum Infix {
    Binary(BinOpKind),
    In,
}

impl Parser {
    fn peek(&self) -> Option<&Tok> {
        self.tokens.get(self.pos).map(|t| &t.tok)
    }

    fn peek_at(&self, offset: usize) -> Option<&Tok> {
        self.tokens.get(self.pos + offset).map(|t| &t.tok)
    }

    fn next(&mut self) -> Result<Token, String> {
        let token = self
            .tokens
            .get(self.pos)
            .cloned()
            .ok_or_else(|| "the predicate ends early".to_string())?;
        self.pos += 1;
        Ok(token)
    }

    /// A Pratt parse at `min_bp`, with the binding powers of
    /// [`binding_power`].
    fn expression(&mut self, min_bp: u8) -> Result<Predicate, String> {
        let mut lhs = self.prefixed()?;
        lhs = self.postfix_as(lhs)?;
        loop {
            let infix = match self.peek() {
                Some(Tok::Op(op)) => Infix::Binary(*op),
                Some(Tok::Ident(word)) if word == "in" => Infix::In,
                _ => break,
            };
            let (l_bp, r_bp) = match infix {
                Infix::Binary(op) => binding_power(op),
                Infix::In => binding_power(BinOpKind::Lt),
            };
            if l_bp < min_bp {
                break;
            }
            self.pos += 1;
            lhs = match infix {
                Infix::In => {
                    let (items, end) = self.list()?;
                    Predicate {
                        span: lhs.span.start..end,
                        kind: PredicateKind::In(Box::new(lhs), items),
                    }
                }
                Infix::Binary(op) => {
                    let rhs = self.expression(r_bp)?;
                    combine(op, lhs, rhs)
                }
            };
        }
        Ok(lhs)
    }

    /// An atom behind any number of unary operators.
    fn prefixed(&mut self) -> Result<Predicate, String> {
        match self.peek() {
            Some(Tok::Bang) => {
                let start = self.next()?.span.start;
                let operand = self.prefixed()?;
                Ok(Predicate {
                    span: start..operand.span.end,
                    kind: PredicateKind::Not(Box::new(operand)),
                })
            }
            Some(Tok::Op(BinOpKind::Sub)) => {
                let start = self.next()?.span.start;
                let operand = self.prefixed()?;
                let span = start..operand.span.end;
                let kind = match operand.kind {
                    PredicateKind::Literal(PredicateLiteral::Int(n)) => {
                        PredicateKind::Literal(PredicateLiteral::Int(-n))
                    }
                    PredicateKind::Literal(PredicateLiteral::Float(f)) => {
                        PredicateKind::Literal(PredicateLiteral::Float(-f))
                    }
                    _ => PredicateKind::Expr,
                };
                Ok(Predicate { kind, span })
            }
            _ => self.atom(),
        }
    }

    /// `<expr> as <type>` casts, each an expression leaf.
    fn postfix_as(&mut self, mut operand: Predicate) -> Result<Predicate, String> {
        while matches!(self.peek(), Some(Tok::Ident(w)) if w == "as")
            && matches!(self.peek_at(1), Some(Tok::Ident(_)))
        {
            self.pos += 1;
            let end = self.next()?.span.end;
            operand = Predicate {
                span: operand.span.start..end,
                kind: PredicateKind::Expr,
            };
        }
        Ok(operand)
    }

    fn atom(&mut self) -> Result<Predicate, String> {
        let token = self.next()?;
        let span = token.span.clone();
        let kind = match token.tok {
            Tok::LParen => {
                let inner = self.expression(0)?;
                let close = self.next()?;
                if !matches!(close.tok, Tok::RParen) {
                    return Err("expected `)` in predicate".to_string());
                }
                return Ok(Predicate {
                    kind: inner.kind,
                    span: span.start..close.span.end,
                });
            }
            Tok::Element(name) => PredicateKind::Element(name),
            Tok::Int(n) => PredicateKind::Literal(PredicateLiteral::Int(n)),
            Tok::Float(f) => PredicateKind::Literal(PredicateLiteral::Float(f)),
            Tok::Str(s) => PredicateKind::Literal(PredicateLiteral::Str(s)),
            Tok::Ident(word) if matches!(self.peek(), Some(Tok::LParen)) => {
                // A call: its arguments are the callee's, evaluated
                // with the call.
                let _ = word;
                let end = self.skip_group()?;
                return Ok(Predicate {
                    kind: PredicateKind::Expr,
                    span: span.start..end,
                });
            }
            Tok::Ident(word) => match word.as_str() {
                "true" => PredicateKind::Literal(PredicateLiteral::Bool(true)),
                "false" => PredicateKind::Literal(PredicateLiteral::Bool(false)),
                _ => PredicateKind::Literal(PredicateLiteral::Str(word)),
            },
            _ => return Err("expected a value in predicate".to_string()),
        };
        Ok(Predicate { kind, span })
    }

    /// Skip a parenthesized group, returning the end of its `)`.
    fn skip_group(&mut self) -> Result<usize, String> {
        let mut depth = 0usize;
        loop {
            let token = self.next()?;
            match token.tok {
                Tok::LParen => depth += 1,
                Tok::RParen => {
                    depth -= 1;
                    if depth == 0 {
                        return Ok(token.span.end);
                    }
                }
                _ => {}
            }
        }
    }

    /// `[item, item, …]` after `in`, and the end of its `]`.
    fn list(&mut self) -> Result<(Vec<Predicate>, usize), String> {
        if !matches!(self.next()?.tok, Tok::LBracket) {
            return Err("expected `[` after `in`".to_string());
        }
        let mut items = Vec::new();
        if matches!(self.peek(), Some(Tok::RBracket)) {
            return Ok((items, self.next()?.span.end));
        }
        loop {
            items.push(self.expression(0)?);
            let token = self.next()?;
            match token.tok {
                Tok::Comma => {}
                Tok::RBracket => return Ok((items, token.span.end)),
                _ => return Err("expected `,` or `]` in an `in` list".to_string()),
            }
        }
    }
}

/// `lhs op rhs` as a node: `||` and `&&` gather their chains,
/// comparisons compare, and any other operator is an expression leaf.
fn combine(op: BinOpKind, lhs: Predicate, rhs: Predicate) -> Predicate {
    let span = lhs.span.start..rhs.span.end;
    let comparison = match op {
        BinOpKind::Eq => Comparison::Eq,
        BinOpKind::Ne => Comparison::Ne,
        BinOpKind::Lt => Comparison::Lt,
        BinOpKind::Le => Comparison::Le,
        BinOpKind::Gt => Comparison::Gt,
        BinOpKind::Ge => Comparison::Ge,
        BinOpKind::Or => {
            let parts = gather(lhs, rhs, |k| match k {
                PredicateKind::Or(parts) => Some(parts),
                _ => None,
            });
            return Predicate {
                kind: PredicateKind::Or(parts),
                span,
            };
        }
        BinOpKind::And => {
            let parts = gather(lhs, rhs, |k| match k {
                PredicateKind::And(parts) => Some(parts),
                _ => None,
            });
            return Predicate {
                kind: PredicateKind::And(parts),
                span,
            };
        }
        _ => {
            return Predicate {
                kind: PredicateKind::Expr,
                span,
            };
        }
    };
    Predicate {
        kind: PredicateKind::Compare(comparison, Box::new(lhs), Box::new(rhs)),
        span,
    }
}

/// The operands of a chain: `lhs`'s own operands when it is the same
/// chain written without parentheses, then `rhs`.
fn gather(
    lhs: Predicate,
    rhs: Predicate,
    parts_of: impl Fn(PredicateKind) -> Option<Vec<Predicate>>,
) -> Vec<Predicate> {
    let span = lhs.span.clone();
    let mut parts = match parts_of(lhs.kind.clone()) {
        Some(parts) if parts.first().map(|p| p.span.start) == Some(span.start) => parts,
        _ => vec![lhs],
    };
    parts.push(rhs);
    parts
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The predicate with its structure written out: every node that is
    /// not a leaf parenthesized.
    fn shape(text: &str) -> String {
        fn walk(p: &Predicate, text: &str) -> String {
            match &p.kind {
                PredicateKind::Or(parts) => format!(
                    "({})",
                    parts
                        .iter()
                        .map(|q| walk(q, text))
                        .collect::<Vec<_>>()
                        .join(" || ")
                ),
                PredicateKind::And(parts) => format!(
                    "({})",
                    parts
                        .iter()
                        .map(|q| walk(q, text))
                        .collect::<Vec<_>>()
                        .join(" && ")
                ),
                PredicateKind::Not(inner) => format!("(!{})", walk(inner, text)),
                PredicateKind::Compare(c, a, b) => {
                    format!("({} {} {})", walk(a, text), c.as_str(), walk(b, text))
                }
                PredicateKind::In(a, items) => format!(
                    "({} in [{}])",
                    walk(a, text),
                    items
                        .iter()
                        .map(|q| walk(q, text))
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
                PredicateKind::Element(name) => format!("{{{name}}}"),
                PredicateKind::Literal(PredicateLiteral::Int(n)) => n.to_string(),
                PredicateKind::Literal(PredicateLiteral::Float(f)) => f.to_string(),
                PredicateKind::Literal(PredicateLiteral::Str(s)) => format!("'{s}'"),
                PredicateKind::Literal(PredicateLiteral::Bool(b)) => b.to_string(),
                PredicateKind::Expr => format!("<{}>", p.text(text)),
            }
        }
        walk(&parse_predicate(text).unwrap(), text)
    }

    /// Unary `!` binds tighter than `&&`, which binds tighter than
    /// `||`.
    #[test]
    fn not_binds_tighter_than_and_and_or() {
        assert_eq!(shape("!{done} || {retry}"), "((!{done}) || {retry})");
        assert_eq!(shape("!{a} && {b}"), "((!{a}) && {b})");
        assert_eq!(shape("{a} || !{b}"), "({a} || (!{b}))");
        assert_eq!(shape("!({a} || {b})"), "(!({a} || {b}))");
        assert_eq!(shape("!!{a}"), "(!(!{a}))");
    }

    /// Every pair of adjacent precedence levels, in both orders.
    #[test]
    fn every_operator_pair_binds_by_the_one_table() {
        let cases = [
            ("{a} || {b} && {c}", "({a} || ({b} && {c}))"),
            ("{a} && {b} || {c}", "(({a} && {b}) || {c})"),
            ("{a} && {b} && {c}", "({a} && {b} && {c})"),
            ("{a} || {b} || {c}", "({a} || {b} || {c})"),
            ("({a} || {b}) && {c}", "(({a} || {b}) && {c})"),
            ("{a} == 1 && {b} != 2", "(({a} == 1) && ({b} != 2))"),
            ("{a} < 1 || {b} >= 2", "(({a} < 1) || ({b} >= 2))"),
            ("{a} < {b} == {c}", "(({a} < {b}) == {c})"),
            ("{a} == {b} < {c}", "({a} == ({b} < {c}))"),
            ("{a} <= 1 && {a} > -1", "(({a} <= 1) && ({a} > -1))"),
            ("!{a} == {b}", "((!{a}) == {b})"),
            ("!{a} < 3", "((!{a}) < 3)"),
            ("{a} + 1 > {b} * 2", "(<{a} + 1> > <{b} * 2>)"),
            ("{a} + {b} * 2 == 7", "(<{a} + {b} * 2> == 7)"),
            (
                "{a} % 2 == 0 || {b} ** 2 > 3",
                "((<{a} % 2> == 0) || (<{b} ** 2> > 3))",
            ),
            ("{a} & 1 == 1", "(<{a} & 1> == 1)"),
            ("{a} | 1 < 3", "(<{a} | 1> < 3)"),
            ("{a} << 1 >= 4 && true", "((<{a} << 1> >= 4) && true)"),
            (
                "{a} in [1, 2] && {b} == x",
                "(({a} in [1, 2]) && ({b} == 'x'))",
            ),
            ("{a} in [1] || !{b}", "(({a} in [1]) || (!{b}))"),
            ("u64_add({a}, {b}) > 7", "(<u64_add({a}, {b})> > 7)"),
            ("!is_even({a}) && {b}", "((!<is_even({a})>) && {b})"),
            ("{a} as f64 > 1.5", "(<{a} as f64> > 1.5)"),
            ("-{a} < 0", "(<-{a}> < 0)"),
            ("\"s0\" != 2", "('s0' != 2)"),
        ];
        for (text, expected) in cases {
            assert_eq!(shape(text), expected, "{text}");
        }
    }

    /// A node's span is its own text, parentheses included, so the
    /// analyzer can hand an operand on as written.
    #[test]
    fn spans_cover_each_operand_as_written() {
        let text = "({a} > 1 || {b} < 2) && !{c}";
        let p = parse_predicate(text).unwrap();
        let PredicateKind::And(parts) = &p.kind else {
            panic!("{p:?}")
        };
        assert_eq!(parts[0].text(text), "({a} > 1 || {b} < 2)");
        assert_eq!(parts[1].text(text), "!{c}");
        assert_eq!(p.text(text), text);
    }

    #[test]
    fn what_the_grammar_does_not_accept_is_an_error() {
        for text in ["", "{a} >", "({a} > 1", "{a} in 3", "{a} . b", "\"open"] {
            assert!(parse_predicate(text).is_err(), "{text}");
        }
    }
}
