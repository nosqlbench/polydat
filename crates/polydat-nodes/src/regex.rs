// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Regex processing nodes.

use regex::Regex;

/// Regex replace: substitute all matches of a pattern with the
/// replacement string. The compiled `Regex` is cached at
/// construction.
#[polydat::polydat_node(category = Regex)]
fn regex_replace(
    input: &str,
    pattern: polydat::derive_support::Const<&str>,
    replacement: polydat::derive_support::Const<&str>,
    #[poly_const(compile_regex, from = pattern)] re: &Regex,
) -> String {
    let _ = pattern;
    re.replace_all(input, replacement.0).into_owned()
}

/// Build a Regex from a pattern. Panics on invalid pattern;
/// the const-arg constraint (registered in the FuncSig the
/// macro emits is forthcoming) catches malformed patterns at
/// workload-compile-time, so the panic here is a true bug
/// indicator only.
fn compile_regex(pattern: &str) -> Regex {
    Regex::new(pattern).expect("invalid regex")
}

/// Regex match: test if input matches a pattern.
#[polydat::polydat_node(category = Regex)]
fn regex_match(
    input: &str,
    pattern: polydat::derive_support::Const<&str>,
    #[poly_const(compile_regex, from = pattern)] re: &Regex,
) -> bool {
    let matched = re.is_match(input);
    if polydat::library::debug_nodes_enabled() {
        let snippet: String = input.chars().take(200).collect();
        let ellipsis = if input.len() > snippet.len() {
            "…"
        } else {
            ""
        };
        polydat::library::support::audit::debug(&format!(
            "regex_match: pattern={:?} input.len={} matched={matched} input.snippet={:?}{ellipsis}",
            re.as_str(),
            input.len(),
            snippet,
        ));
    }
    matched
}

/// Regex extract: extract the first capture group (or full match).
#[polydat::polydat_node(category = Regex)]
fn regex_extract(
    input: &str,
    pattern: polydat::derive_support::Const<&str>,
    #[poly_const(compile_regex, from = pattern)] re: &Regex,
) -> String {
    if let Some(caps) = re.captures(input) {
        caps.get(1)
            .or_else(|| caps.get(0))
            .map(|m| m.as_str().to_string())
            .unwrap_or_default()
    } else {
        String::new()
    }
}

// ---------------------------------------------------------------------------
// Pattern promotion (literal / glob / regex) + the `pattern_match` node
// ---------------------------------------------------------------------------
//
// A *pattern* is a source string lifted into an anchored regex by the
// shape of the source. Shared with `nbrs-runtime`'s phase-name filter
// (which delegates to `compile_pattern`) so `phases=…` and a workload's
// `pattern_match(...)` agree on what a pattern means.

pub use polydat::library::support::pattern::{PatternDialect, compile_pattern, promote_pattern};

/// Build-time helper for [`pattern_match`]: promote + compile the const
/// pattern once. Panics on an un-compilable pattern, like
/// [`compile_regex`].
fn compile_promoted(pattern: &str) -> Regex {
    compile_pattern(pattern).unwrap_or_else(|e| panic!("{e}")).0
}

/// Promoted pattern match: `true` if `input` matches `pattern` under the
/// literal / glob / regex promotion rules. Sibling to `regex_match`
/// (which takes a *raw* regex) — use this when the pattern may be a
/// strict string, a `*` glob, or a regex and the kind should be
/// auto-detected from its shape.
#[polydat::polydat_node(category = Regex)]
fn pattern_match(
    input: &str,
    pattern: polydat::derive_support::Const<&str>,
    #[poly_const(compile_promoted, from = pattern)] re: &Regex,
) -> bool {
    re.is_match(input)
}

#[cfg(test)]
mod pattern_tests {
    use super::*;

    #[test]
    fn literal_is_strict_anchored() {
        let (re, d) = compile_pattern("schema").unwrap();
        assert_eq!(d, PatternDialect::Literal);
        assert!(re.is_match("schema"));
        assert!(!re.is_match("schema_v2"));
        assert!(!re.is_match("pre_schema"));
    }

    #[test]
    fn glob_star_distinguishes_suffix_series() {
        // The motivating case: `*m` vs `*mi`.
        let (m, dm) = compile_pattern("*m").unwrap();
        assert_eq!(dm, PatternDialect::Glob);
        assert!(m.is_match("100m"));
        assert!(!m.is_match("100mi"));
        let (mi, _) = compile_pattern("*mi").unwrap();
        assert!(mi.is_match("100mi"));
        assert!(!mi.is_match("100m"));
    }

    #[test]
    fn regex_metachar_promotes() {
        let (re, d) = compile_pattern("(100|200)m").unwrap();
        assert_eq!(d, PatternDialect::Regex);
        assert!(re.is_match("100m"));
        assert!(re.is_match("200m"));
        assert!(!re.is_match("300m"));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use polydat::ast::{PolydatNode, Value};

    #[test]
    fn regex_replace_basic() {
        let node = RegexReplace::new(r"\d+".to_string(), "NUM".to_string());
        let mut out = [Value::None];
        node.eval(&[Value::Str("abc 123 def 456".into())], &mut out);
        assert_eq!(out[0].as_str(), "abc NUM def NUM");
    }

    #[test]
    fn regex_replace_no_match() {
        let node = RegexReplace::new(r"\d+".to_string(), "NUM".to_string());
        let mut out = [Value::None];
        node.eval(&[Value::Str("no numbers here".into())], &mut out);
        assert_eq!(out[0].as_str(), "no numbers here");
    }

    #[test]
    fn regex_match_true() {
        let node = RegexMatch::new(r"^\d{3}-\d{4}$".to_string());
        let mut out = [Value::None];
        node.eval(&[Value::Str("123-4567".into())], &mut out);
        assert!(out[0].as_bool());
    }

    #[test]
    fn regex_match_false() {
        let node = RegexMatch::new(r"^\d{3}-\d{4}$".to_string());
        let mut out = [Value::None];
        node.eval(&[Value::Str("hello".into())], &mut out);
        assert!(!out[0].as_bool());
    }

    #[test]
    fn regex_extract_capture_group() {
        let node = RegexExtract::new(r"name=(\w+)".to_string());
        let mut out = [Value::None];
        node.eval(&[Value::Str("name=Alice age=30".into())], &mut out);
        assert_eq!(out[0].as_str(), "Alice");
    }

    #[test]
    fn regex_extract_no_group() {
        let node = RegexExtract::new(r"\d+".to_string());
        let mut out = [Value::None];
        node.eval(&[Value::Str("abc 42 def".into())], &mut out);
        assert_eq!(out[0].as_str(), "42");
    }

    #[test]
    fn regex_extract_no_match() {
        let node = RegexExtract::new(r"\d+".to_string());
        let mut out = [Value::None];
        node.eval(&[Value::Str("no digits".into())], &mut out);
        assert_eq!(out[0].as_str(), "");
    }
}
