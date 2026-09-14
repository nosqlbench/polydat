// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! The runtime's reading of a value as a comprehension source: which
//! values peel into elements, and how a string source strips into
//! typed tokens. The grammar's separator rule
//! (`source::split_string_comprehension`) is the one both sides use.

use super::source::split_string_comprehension;

/// The canonical "is this value peelable, and into what?" predicate
/// (SRD-18f §4). Returns `Some(interior)` when `v` has an iteration
/// interior — peeling it one level yields these elements — and
/// `None` when `v` is an iteration scalar (relaxed wraps it; an
/// explicit `[v…]` destructure errors).
///
/// This is the single place the peel/wrap decision is made,
/// replacing the scattered per-type special cases (the
/// `as_partition_list()`-peel / `Ok(other)`-wrap arms) that used
/// to live in `eval::evaluate_spec_internal`.
///
/// `Value::Str` is iterable — its interior is its
/// **string-comprehension tokens** ([`strip_string_tokens`]). The
/// single-quoted *atomic* form is resolved earlier, at the
/// source-text layer (the parser produces a one-element literal),
/// so by the time a bare `Value::Str` reaches this predicate the
/// intent is "iterate it." A whole-string binding is reached via
/// the no-peel form `x in [s]`.
pub fn iteration_interior(v: &crate::ast::Value) -> Option<Vec<crate::ast::Value>> {
    use crate::ast::Value;
    match v {
        Value::VecF32(s) => Some(s.as_slice().iter().map(|x| Value::F64(*x as f64)).collect()),
        Value::VecF64(s) => Some(s.as_slice().iter().map(|x| Value::F64(*x)).collect()),
        Value::VecF16(s) => Some(
            s.as_slice()
                .iter()
                .map(|x| Value::F64(x.to_f64()))
                .collect(),
        ),
        // Signed lanes peel to the honest signed carrier — a
        // VecI64 holding -5 iterates as I64(-5), not the unsigned
        // bit-reinterpretation (type_system_alignment.md §5).
        Value::VecI32(s) => Some(s.as_slice().iter().map(|x| Value::I64(*x as i64)).collect()),
        Value::VecI64(s) => Some(s.as_slice().iter().map(|x| Value::I64(*x)).collect()),
        Value::VecI16(s) => Some(s.as_slice().iter().map(|x| Value::I64(*x as i64)).collect()),
        Value::VecI8(s) => Some(s.as_slice().iter().map(|x| Value::I64(*x as i64)).collect()),
        // A JSON array peels to its elements (each carried as a
        // Json value); a non-array JSON is an iteration scalar.
        Value::Json(j) => j.as_array().map(|arr| {
            arr.iter()
                .map(|e| Value::Json(std::sync::Arc::new(e.clone())))
                .collect()
        }),
        // Ext carrying a PartitionList peels into its partitions
        // (SRD-71). Other Ext values are opaque scalars.
        Value::Ext(_) => v.as_partition_list().map(|list| {
            list.as_slice()
                .iter()
                .map(|p| Value::from_partition(*p))
                .collect()
        }),
        // Lane-typed register views peel like the Vec* family
        // (a reg_f32x4 is a fixed-width typed vector); the Raw
        // view is an opaque buffer-state word — an iteration
        // scalar.
        Value::Reg128(b, view) => {
            use crate::ast::RegLanes;
            match view {
                RegLanes::Raw => None,
                RegLanes::I8x16 => {
                    Some(b.lanes_i8().iter().map(|x| Value::I64(*x as i64)).collect())
                }
                RegLanes::I16x8 => Some(
                    b.lanes_i16()
                        .iter()
                        .map(|x| Value::I64(*x as i64))
                        .collect(),
                ),
                RegLanes::I32x4 => Some(
                    b.lanes_i32()
                        .iter()
                        .map(|x| Value::I64(*x as i64))
                        .collect(),
                ),
                RegLanes::I64x2 => Some(b.lanes_i64().iter().map(|x| Value::I64(*x)).collect()),
                RegLanes::F16x8 => Some(
                    b.lanes_f16()
                        .iter()
                        .map(|x| Value::F64(x.to_f64()))
                        .collect(),
                ),
                RegLanes::F32x4 => Some(
                    b.lanes_f32()
                        .iter()
                        .map(|x| Value::F64(*x as f64))
                        .collect(),
                ),
                RegLanes::F64x2 => Some(b.lanes_f64().iter().map(|x| Value::F64(*x)).collect()),
            }
        }
        // A string's interior is its comprehension tokens.
        Value::Str(s) => Some(strip_string_tokens(s)),
        // Iteration scalars — relaxed wraps, `[v…]` errors.
        Value::U64(_)
        | Value::I64(_)
        | Value::U128(_)
        | Value::I128(_)
        | Value::F64(_)
        | Value::Bool(_)
        | Value::Bytes(_)
        | Value::Handle(_)
        | Value::None => None,
    }
}

/// Split a string-comprehension source into its token values
/// (SRD-18f §3.2). Separators are runs of comma, semicolon, and
/// ASCII whitespace; every other character — notably `:` (k:v /
/// `a:b:c` tuples), `.` (floats), `-` (negatives / hyphenated
/// labels), `/` — stays inside the token. Each token is typed
/// like a literal-list element (u64 / f64 / bool / else Str), so
/// `"1, 2, 3"` yields numeric values and `"a, b"` yields strings.
///
/// A single-token string (`"OTHER"`) yields a one-element vec, so
/// the single-value case degenerates to the no-peel binding for
/// free.
pub fn strip_string_tokens(s: &str) -> Vec<crate::ast::Value> {
    use crate::ast::Value;
    split_string_comprehension(s)
        .into_iter()
        .map(|t| {
            if let Ok(n) = t.parse::<u64>() {
                Value::U64(n)
            } else if let Ok(f) = t.parse::<f64>() {
                Value::F64(f)
            } else if t == "true" {
                Value::Bool(true)
            } else if t == "false" {
                Value::Bool(false)
            } else {
                Value::Str(t.to_string().into())
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::Value;

    #[test]
    fn string_strips_on_comma_semicolon_whitespace_retaining_colons() {
        // SRD-18f §3.2: separators are comma / semicolon / ws;
        // colons (and dots, dashes) stay in-token.
        let got = strip_string_tokens("a:1, b:2; c:3 d:4");
        assert_eq!(
            got,
            vec![
                Value::Str("a:1".into()),
                Value::Str("b:2".into()),
                Value::Str("c:3".into()),
                Value::Str("d:4".into()),
            ]
        );
    }

    #[test]
    fn string_tokens_are_typed_like_literals() {
        // Floats survive (dot retained), ints type as U64.
        assert_eq!(
            strip_string_tokens("1, 2, 3"),
            vec![Value::U64(1), Value::U64(2), Value::U64(3)]
        );
        assert_eq!(
            strip_string_tokens("1.5, 2.5"),
            vec![Value::F64(1.5), Value::F64(2.5)]
        );
    }

    #[test]
    fn single_token_string_degenerates_to_singleton() {
        assert_eq!(
            strip_string_tokens("OTHER"),
            vec![Value::Str("OTHER".into())]
        );
    }

    #[test]
    fn iteration_interior_string_is_its_tokens() {
        let v = Value::Str("x, y, z".into());
        assert_eq!(
            iteration_interior(&v),
            Some(vec![
                Value::Str("x".into()),
                Value::Str("y".into()),
                Value::Str("z".into())
            ])
        );
    }

    #[test]
    fn iteration_interior_vector_peels_to_elements() {
        let v = Value::VecI32(crate::ast::SliceArc::from_vec(vec![10, 20, 30]));
        assert_eq!(
            iteration_interior(&v),
            Some(vec![Value::I64(10), Value::I64(20), Value::I64(30)])
        );
    }

    #[test]
    fn iteration_interior_signed_vector_peels_signed() {
        // The honest-I64 alignment fix: negative lane elements
        // iterate as their signed values, not the unsigned
        // bit-reinterpretation (type_system_alignment.md §5).
        let v = Value::VecI64(crate::ast::SliceArc::from_vec(vec![-5_i64, 7]));
        assert_eq!(
            iteration_interior(&v),
            Some(vec![Value::I64(-5), Value::I64(7)])
        );
    }

    #[test]
    fn iteration_interior_scalars_are_none() {
        assert_eq!(iteration_interior(&Value::U64(5)), None);
        assert_eq!(iteration_interior(&Value::F64(1.5)), None);
        assert_eq!(iteration_interior(&Value::Bool(true)), None);
    }
}
