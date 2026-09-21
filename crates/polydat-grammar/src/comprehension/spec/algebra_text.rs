// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! The comprehension text as the algebra reads it (comprehension_forms.md
//! §8.1, §8.2): a flat clause list with its trailing `where` and `order`
//! goes through the clause parser and [`clauses_to_algebra`]; a bracketed
//! union, `[ for <member>, for <member>, … ]`, lowers each member by the
//! same rule, so any position that takes a comprehension takes every
//! form, and the union's own trailing `where` and `order` apply to the
//! whole.

use crate::comprehension::ast::Comprehension;
use crate::comprehension::parse::{
    parse_comprehension_text, parse_order_spec, split_at_order, split_at_where,
};

use super::from_clauses::{clauses_to_algebra, convert_order};

/// Parse comprehension text to the algebra. A text beginning with `[`
/// is a union of the `for`-introduced members inside the brackets,
/// each parsed by this function; anything else is a clause list with
/// its modifiers.
pub fn parse_comprehension_algebra(text: &str) -> Result<Comprehension, String> {
    let trimmed = text.trim();
    let Some(inside_and_tail) = trimmed.strip_prefix('[') else {
        let clauses = parse_comprehension_text(trimmed)?;
        return clauses_to_algebra(&clauses).map_err(|e| e.to_string());
    };
    let close = matching_close(inside_and_tail)
        .ok_or_else(|| "union: missing the `]` that closes `[`".to_string())?;
    let inside = &inside_and_tail[..close];
    let tail = &inside_and_tail[close + 1..];

    let members = split_members(inside)?;
    if members.is_empty() {
        return Err("union: no members between `[` and `]`; each is introduced by `for`".into());
    }
    let members = members
        .iter()
        .map(|m| parse_comprehension_algebra(m))
        .collect::<Result<Vec<_>, _>>()?;
    let mut comp = Comprehension::union(members);

    // The union's own modifiers, after `]`, in the same order as on a
    // clause list: `where` binds tighter than `order`.
    let (head, order_text) = split_at_order(tail);
    let (rest, filter) = split_at_where(&head);
    if !rest.trim().is_empty() {
        return Err(format!(
            "union: unexpected text after `]`: `{}`",
            rest.trim()
        ));
    }
    if let Some(predicate) = filter {
        comp = Comprehension::filter(comp, predicate);
    }
    if let Some(spec) = order_text {
        let order = parse_order_spec(&spec)?;
        let (strategy, truncation, seed) = convert_order(&order).map_err(|e| e.to_string())?;
        comp = Comprehension::order_seeded(comp, strategy, truncation, seed);
    }
    Ok(comp)
}

/// The byte index of the `]` that closes the `[` just before `s`,
/// skipping nested brackets and quoted text.
fn matching_close(s: &str) -> Option<usize> {
    let mut depth = 0usize;
    let mut quote: Option<char> = None;
    for (i, c) in s.char_indices() {
        match quote {
            Some(q) => {
                if c == q {
                    quote = None;
                }
            }
            None => match c {
                '"' | '\'' => quote = Some(c),
                '(' | '[' | '{' => depth += 1,
                ')' | '}' => depth = depth.saturating_sub(1),
                ']' => {
                    if depth == 0 {
                        return Some(i);
                    }
                    depth -= 1;
                }
                _ => {}
            },
        }
    }
    None
}

/// Split the inside of a union's brackets into its members: each is
/// introduced by the keyword `for` at bracket depth zero, and a comma
/// may end it.
fn split_members(inside: &str) -> Result<Vec<String>, String> {
    let mut starts: Vec<usize> = Vec::new();
    let mut depth = 0usize;
    let mut quote: Option<char> = None;
    let bytes = inside.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i] as char;
        match quote {
            Some(q) => {
                if c == q {
                    quote = None;
                }
            }
            None => match c {
                '"' | '\'' => quote = Some(c),
                '(' | '[' | '{' => depth += 1,
                ')' | ']' | '}' => depth = depth.saturating_sub(1),
                'f' if depth == 0 && inside[i..].starts_with("for") => {
                    let before_ok =
                        i == 0 || matches!(bytes[i - 1] as char, ' ' | '\t' | '\n' | '\r' | ',');
                    let after_ok = inside[i + 3..]
                        .chars()
                        .next()
                        .is_none_or(|n| n.is_whitespace() || n == '[');
                    if before_ok && after_ok {
                        starts.push(i);
                        i += 3;
                        continue;
                    }
                }
                _ => {}
            },
        }
        i += 1;
    }
    let Some(&first) = starts.first() else {
        if inside.trim().is_empty() {
            return Ok(Vec::new());
        }
        return Err(format!(
            "union: members are introduced by `for`; found `{}`",
            inside.trim()
        ));
    };
    if !inside[..first].trim().is_empty() {
        return Err(format!(
            "union: members are introduced by `for`; found `{}` before the first",
            inside[..first].trim()
        ));
    }
    let mut members = Vec::with_capacity(starts.len());
    for (n, &start) in starts.iter().enumerate() {
        let end = starts.get(n + 1).copied().unwrap_or(inside.len());
        let member = inside[start + 3..end].trim().trim_end_matches(',').trim();
        if member.is_empty() {
            return Err("union: a `for` with no comprehension after it".into());
        }
        members.push(member.to_string());
    }
    Ok(members)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_flat_text_is_a_clause_list() {
        let c = parse_comprehension_algebra("k in 1..3, limit in 10,20").unwrap();
        assert!(matches!(c, Comprehension::Cartesian { .. }));
    }

    #[test]
    fn a_bracketed_text_is_a_union_of_its_members() {
        let c = parse_comprehension_algebra(
            "[ for k in 1..3 where {k} > 1, for k in 10..12 order lex/1, ]",
        )
        .unwrap();
        let Comprehension::Union { children } = c else {
            panic!("expected a union, got {c:?}");
        };
        assert_eq!(children.len(), 2);
        assert!(matches!(children[0], Comprehension::Filter { .. }));
        assert!(matches!(children[1], Comprehension::Order { .. }));
    }

    #[test]
    fn a_union_takes_its_own_modifiers_after_the_bracket() {
        let c = parse_comprehension_algebra(
            "[ for k in 1..3, for k in 10..12 ] where {k} > 2 order halton/2",
        )
        .unwrap();
        let Comprehension::Order { child, .. } = c else {
            panic!("expected an order, got {c:?}");
        };
        let Comprehension::Filter { child, .. } = *child else {
            panic!("expected a filter under the order");
        };
        assert!(matches!(*child, Comprehension::Union { .. }));
    }

    #[test]
    fn unions_nest() {
        let c =
            parse_comprehension_algebra("[ for [ for k in 1..2, for k in 3..4 ], for k in 9..9 ]")
                .unwrap();
        let Comprehension::Union { children } = c else {
            panic!("expected a union, got {c:?}");
        };
        assert_eq!(children.len(), 2);
        assert!(matches!(children[0], Comprehension::Union { .. }));
    }

    #[test]
    fn members_must_be_introduced_by_for() {
        let err = parse_comprehension_algebra("[ k in 1..3, for k in 10..12 ]").unwrap_err();
        assert!(err.contains("introduced by `for`"), "{err}");
        let err = parse_comprehension_algebra("[ for k in 1..3").unwrap_err();
        assert!(err.contains("missing the `]`"), "{err}");
    }
}
