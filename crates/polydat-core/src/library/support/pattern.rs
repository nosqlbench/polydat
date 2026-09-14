// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Pattern promotion: a literal, a `*` glob, or a regex, told apart by
//! shape and compiled to one anchored regex. The `pattern_match` node
//! and the vector catalog's profile filter share it.

use regex::Regex;

/// Which dialect [`promote_pattern`] / [`compile_pattern`] picked.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PatternDialect {
    /// No regex metachars and no `*` — strict, anchored full-string
    /// match (`^src$`).
    Literal,
    /// `*` (and no other regex metachar) — each `*` expands to `.*`,
    /// anchored.
    Glob,
    /// Any regex metachar — a full Rust regex, anchored `^(?:src)$`.
    Regex,
}

impl PatternDialect {
    /// The dialect's name, as `pattern_dialect` spells it.
    pub fn as_str(self) -> &'static str {
        match self {
            PatternDialect::Literal => "literal",
            PatternDialect::Glob => "glob",
            PatternDialect::Regex => "regex",
        }
    }
}

/// Metachars whose presence means "already a regex — don't glob-expand".
/// Excludes `*`, which is the glob marker.
const PATTERN_REGEX_METACHARS: &[char] = &[
    '.', '+', '?', '(', ')', '|', '[', ']', '{', '}', '^', '$', '\\',
];

/// Classify `source` and return its anchored regex source + dialect.
/// Pure — does not compile. Promotion rules:
/// - any regex metachar → full regex, `^(?:src)$`
/// - else if it contains `*` → glob, each `*` → `.*`, `^…$`
/// - else → literal, `^escape(src)$`
pub fn promote_pattern(source: &str) -> (String, PatternDialect) {
    let has_regex_metachar = source.chars().any(|c| PATTERN_REGEX_METACHARS.contains(&c));
    let has_glob_star = source.contains('*');
    if has_regex_metachar {
        (format!("^(?:{source})$"), PatternDialect::Regex)
    } else if has_glob_star {
        // regex::escape renders `*` as `\*`; turn those back into `.*`.
        let pattern = regex::escape(source).replace("\\*", ".*");
        (format!("^{pattern}$"), PatternDialect::Glob)
    } else {
        (
            format!("^{}$", regex::escape(source)),
            PatternDialect::Literal,
        )
    }
}

/// Promote and compile `source` into an anchored [`Regex`].
pub fn compile_pattern(source: &str) -> Result<(Regex, PatternDialect), String> {
    let (anchored, dialect) = promote_pattern(source);
    let re = Regex::new(&anchored).map_err(|e| {
        format!(
            "pattern '{source}' did not compile (dialect={}): {e}",
            dialect.as_str()
        )
    })?;
    Ok((re, dialect))
}
