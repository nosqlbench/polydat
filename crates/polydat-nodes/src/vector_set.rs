// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Element and set operations over the integer vector carriers.
//!
//! The float vector nodes ([`super::vector_math`]) compute over
//! embeddings; these read integer vectors as *lists*: how many
//! elements, what is at an index, does a value appear, where, and how
//! two lists compare as sets. A host holding a list of ordinals — the
//! records matching a label, the queries matching a predicate — asks
//! these the questions it would otherwise ask a counter.
//!
//! That is the point of them. A rank ("how many matching records come
//! before this one") and a cardinality ("how many match at all") look
//! like things a counter produces as records go by, but both are pure
//! functions of the list and the record: the rank is the position, the
//! cardinality is the length. Written that way the answer is owned by
//! its inputs, replays from its coordinates, and agrees on every engine
//! and every fiber, where a counter's answer depends on how many times
//! it was called ([Runtime Model](../../polydat/docs/design/runtime_model.md)
//! §9.1, "No state across pulls").
//!
//! Each operation exists for `vec_i32` and `vec_i64`, since the carrier
//! is part of the wire's type and polydat does not silently widen one
//! to the other. An index or a value that is out of range answers
//! rather than panicking, because a miss is an ordinary result here:
//! `vec_position` returns the length when the value is absent, which is
//! "past the end" and the same convention `vec_len` bounds.

/// `vec_len_i32(v)` — how many elements the list holds.
#[polydat::polydat_node(category = Arithmetic)]
fn vec_len_i32(v: &[i32]) -> u64 {
    v.len() as u64
}

/// `vec_len_i64(v)` — how many elements the list holds.
#[polydat::polydat_node(category = Arithmetic)]
fn vec_len_i64(v: &[i64]) -> u64 {
    v.len() as u64
}

/// `vec_at_i32(v, i)` — the element at `i`, or `0` past the end.
///
/// Out of range answers rather than failing: a caller that cares
/// compares `i` with `vec_len_i32` first, and one that is walking a
/// list it already sized does not pay for a check it cannot fail.
#[polydat::polydat_node(category = Arithmetic)]
fn vec_at_i32(v: &[i32], i: u64) -> i64 {
    v.get(i as usize).copied().unwrap_or(0) as i64
}

/// `vec_at_i64(v, i)` — the element at `i`, or `0` past the end.
#[polydat::polydat_node(category = Arithmetic)]
fn vec_at_i64(v: &[i64], i: u64) -> i64 {
    v.get(i as usize).copied().unwrap_or(0)
}

/// `vec_max_i32(v)` — the largest element, or `0` for an empty list.
#[polydat::polydat_node(category = Arithmetic)]
fn vec_max_i32(v: &[i32]) -> i64 {
    v.iter().copied().max().unwrap_or(0) as i64
}

/// `vec_max_i64(v)` — the largest element, or `0` for an empty list.
#[polydat::polydat_node(category = Arithmetic)]
fn vec_max_i64(v: &[i64]) -> i64 {
    v.iter().copied().max().unwrap_or(0)
}

/// `vec_min_i32(v)` — the smallest element, or `0` for an empty list.
#[polydat::polydat_node(category = Arithmetic)]
fn vec_min_i32(v: &[i32]) -> i64 {
    v.iter().copied().min().unwrap_or(0) as i64
}

/// `vec_min_i64(v)` — the smallest element, or `0` for an empty list.
#[polydat::polydat_node(category = Arithmetic)]
fn vec_min_i64(v: &[i64]) -> i64 {
    v.iter().copied().min().unwrap_or(0)
}

/// `vec_contains_i32(v, x)` — `1` when `x` appears in the list, else `0`.
#[polydat::polydat_node(category = Comparison)]
fn vec_contains_i32(v: &[i32], x: i64) -> u64 {
    u64::from(i32::try_from(x).is_ok_and(|x| v.contains(&x)))
}

/// `vec_contains_i64(v, x)` — `1` when `x` appears in the list, else `0`.
#[polydat::polydat_node(category = Comparison)]
fn vec_contains_i64(v: &[i64], x: i64) -> u64 {
    u64::from(v.contains(&x))
}

/// `vec_position_i32(v, x)` — the index of the first `x`, or the list's
/// length when it does not appear.
///
/// The length is "past the end", so a caller can branch on
/// `pos == vec_len_i32(v)` without a second sentinel, and a rank over a
/// sorted list of ordinals reads directly as the position.
#[polydat::polydat_node(category = Comparison)]
fn vec_position_i32(v: &[i32], x: i64) -> u64 {
    match i32::try_from(x) {
        Ok(x) => v.iter().position(|e| *e == x).unwrap_or(v.len()) as u64,
        Err(_) => v.len() as u64,
    }
}

/// `vec_position_i64(v, x)` — the index of the first `x`, or the list's
/// length when it does not appear.
#[polydat::polydat_node(category = Comparison)]
fn vec_position_i64(v: &[i64], x: i64) -> u64 {
    v.iter().position(|e| *e == x).unwrap_or(v.len()) as u64
}

/// `vec_count_below_i32(v, x)` — how many elements are less than `x`.
///
/// The rank of a record among the matching ones, without requiring the
/// list to be sorted or the record to be in it, which
/// `vec_position_i32` both do.
#[polydat::polydat_node(category = Comparison)]
fn vec_count_below_i32(v: &[i32], x: i64) -> u64 {
    v.iter().filter(|e| i64::from(**e) < x).count() as u64
}

/// `vec_count_below_i64(v, x)` — how many elements are less than `x`.
#[polydat::polydat_node(category = Comparison)]
fn vec_count_below_i64(v: &[i64], x: i64) -> u64 {
    v.iter().filter(|e| **e < x).count() as u64
}

/// `vec_intersect_count_i32(a, b)` — how many *distinct* values appear
/// in both lists.
///
/// Distinct, because this answers "how much do these two sets overlap",
/// which a repeated value would otherwise inflate. A caller wanting the
/// multiset count writes the fold it means.
#[polydat::polydat_node(category = Comparison)]
fn vec_intersect_count_i32(a: &[i32], b: &[i32]) -> u64 {
    let set: std::collections::BTreeSet<i32> = b.iter().copied().collect();
    let mine: std::collections::BTreeSet<i32> = a.iter().copied().collect();
    mine.intersection(&set).count() as u64
}

/// `vec_intersect_count_i64(a, b)` — how many distinct values appear in
/// both lists.
#[polydat::polydat_node(category = Comparison)]
fn vec_intersect_count_i64(a: &[i64], b: &[i64]) -> u64 {
    let set: std::collections::BTreeSet<i64> = b.iter().copied().collect();
    let mine: std::collections::BTreeSet<i64> = a.iter().copied().collect();
    mine.intersection(&set).count() as u64
}

/// `vec_set_eq_i32(a, b)` — `1` when both lists hold the same set of
/// values, whatever their order or repetition.
#[polydat::polydat_node(category = Comparison)]
fn vec_set_eq_i32(a: &[i32], b: &[i32]) -> u64 {
    let x: std::collections::BTreeSet<i32> = a.iter().copied().collect();
    let y: std::collections::BTreeSet<i32> = b.iter().copied().collect();
    u64::from(x == y)
}

/// `vec_set_eq_i64(a, b)` — `1` when both lists hold the same set of
/// values, whatever their order or repetition.
#[polydat::polydat_node(category = Comparison)]
fn vec_set_eq_i64(a: &[i64], b: &[i64]) -> u64 {
    let x: std::collections::BTreeSet<i64> = a.iter().copied().collect();
    let y: std::collections::BTreeSet<i64> = b.iter().copied().collect();
    u64::from(x == y)
}

// ── Reading a list from text ────────────────────────────────────────

/// `str_to_vec_i32(s)` — a JSON array of integers as a `vec_i32` wire.
///
/// The public name for a parse the crate already performed internally
/// at the boundary: a host holding a list as text — a column it
/// selected, a facet it loaded, a literal in a workload — gets it onto
/// a wire the list operations read. The `as` cast does not do this,
/// because text to a typed vector can fail on its input and the
/// adapter catalog only inserts conversions that cannot ([Type
/// System](type_system.md) §3); naming it is the caller saying it
/// accepts that.
///
/// Fails by name on anything that is not a JSON array of integers in
/// range, rather than skipping the element: a list silently short one
/// entry is a wrong count everywhere downstream.
#[polydat::polydat_node(category = Conversions)]
fn str_to_vec_i32(s: &str) -> Vec<i32> {
    parse_int_array("str_to_vec_i32", s)
        .into_iter()
        .map(|n| {
            i32::try_from(n).unwrap_or_else(|_| {
                panic!(
                    "str_to_vec_i32: element {n} is outside i32 [{}, {}]",
                    i32::MIN,
                    i32::MAX
                )
            })
        })
        .collect()
}

/// `str_to_vec_i64(s)` — a JSON array of integers as a `vec_i64` wire.
#[polydat::polydat_node(category = Conversions)]
fn str_to_vec_i64(s: &str) -> Vec<i64> {
    parse_int_array("str_to_vec_i64", s)
}

/// The shared parse: a JSON array of integers, or a failure naming the
/// node, the text, and what was wrong with it.
fn parse_int_array(node: &str, s: &str) -> Vec<i64> {
    let raw = s.trim();
    let parsed: serde_json::Value = serde_json::from_str(raw)
        .unwrap_or_else(|e| panic!("{node}: cannot parse {raw:?} as a JSON array: {e}"));
    let arr = parsed
        .as_array()
        .unwrap_or_else(|| panic!("{node}: parsed JSON is not an array: {raw:?}"));
    arr.iter()
        .map(|j| {
            j.as_i64()
                .unwrap_or_else(|| panic!("{node}: element {j} is not an integer in {raw:?}"))
        })
        .collect()
}
