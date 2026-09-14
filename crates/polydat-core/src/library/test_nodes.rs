// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Fixture nodes for the core's own tests. The core links no node
//! library in its test build (linking `polydat-nodes` there would put
//! a second copy of the core under the nodes), so the tests that
//! exercise the compiler, the kernels, and the subcontext seal
//! through Polydat source name these instead: the same functions,
//! with the same bodies, as the node library defines them. They exist
//! only under `cfg(test)` and register only into the test build's
//! registry.

use regex::Regex;

/// `hash(input)`: the SplitMix64 mixer, as the library's `hash`.
#[crate::polydat_node(category = Hashing)]
fn hash(input: u64) -> u64 {
    crate::numeric::hash::splitmix64_u64(input)
}

/// `mul(input, factor)`: a wrapping product, as the library's `mul`.
#[crate::polydat_node(category = Arithmetic)]
fn mul(input: u64, factor: crate::derive_support::Const<u64>) -> u64 {
    input.wrapping_mul(*factor)
}

/// `u64_add(a, b)`: a wrapping sum, as the library's `u64_add`.
#[crate::polydat_node(category = Arithmetic)]
fn u64_add(a: u64, b: u64) -> u64 {
    a.wrapping_add(b)
}

/// `u64_eq(a, b)`: 1 when equal, as the library's `u64_eq`.
#[crate::polydat_node(category = Comparison)]
fn u64_eq(a: u64, b: u64) -> u64 {
    if a == b { 1 } else { 0 }
}

fn compile_regex(pattern: &str) -> Regex {
    Regex::new(pattern).unwrap_or_else(|e| panic!("regex_match: pattern '{pattern}': {e}"))
}

/// `regex_match(input, pattern)`: whether the pattern matches, as the
/// library's `regex_match` without its debug trace.
#[crate::polydat_node(category = Regex)]
fn regex_match(
    input: &str,
    pattern: crate::derive_support::Const<&str>,
    #[poly_const(compile_regex, from = pattern)] re: &Regex,
) -> bool {
    let _ = pattern;
    re.is_match(input)
}

/// `add(input, addend)`: a wrapping sum, as the library's `add`.
#[crate::polydat_node(category = Arithmetic)]
fn add(input: u64, addend: crate::derive_support::Const<u64>) -> u64 {
    input.wrapping_add(*addend)
}

/// `u64_gt(a, b)`: 1 when `a > b`, as the library's `u64_gt`.
#[crate::polydat_node(category = Comparison)]
fn u64_gt(a: u64, b: u64) -> u64 {
    if a > b { 1 } else { 0 }
}
