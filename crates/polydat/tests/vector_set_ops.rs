// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Element and set operations over integer vectors, the parses that
//! get a list onto a wire, and the two string readings that go with
//! them.
//!
//! These exist so a host can ask "what is this record's rank among the
//! matching ones" and "how many match" as functions of a list, rather
//! than as a counter advancing over a visit order that polydat does not
//! promise (docs/design/runtime_model.md §9.1).

use polydat::dsl::compile_polydat_kernel;

/// Compile on the default engine, set the coordinate, read the outputs.
fn run(src: &str, names: &[&str]) -> Vec<String> {
    let mut k = compile_polydat_kernel(src).unwrap_or_else(|e| panic!("{e}\n{src}"));
    k.set_inputs(&[0]);
    names
        .iter()
        .map(|n| k.pull(n).to_display_string())
        .collect()
}

const ORDS: &str = concat!(
    "input cycle: u64\n",
    "list := \"[3, 9, 14, 20]\"\n",
    "ords := str_to_vec_i32(list)\n",
);

#[test]
fn a_list_answers_its_length_elements_and_bounds() {
    let src = format!(
        "{ORDS}{}",
        concat!(
            "n := vec_len_i32(ords)\n",
            "first := vec_at_i32(ords, 0)\n",
            "third := vec_at_i32(ords, 2)\n",
            "past := vec_at_i32(ords, 99)\n",
            "hi := vec_max_i32(ords)\n",
            "lo := vec_min_i32(ords)\n",
        )
    );
    assert_eq!(
        run(&src, &["n", "first", "third", "past", "hi", "lo"]),
        // Past the end answers rather than failing.
        vec!["4", "3", "14", "0", "20", "3"]
    );
}

#[test]
fn a_rank_is_a_position_not_a_counter() {
    // The rank of ordinal 14 among the matching ones is 2, asked either
    // way: its index in the sorted list, or how many matching ordinals
    // precede it. Neither reads a visit order.
    let src = format!(
        "{ORDS}{}",
        concat!(
            "rank_by_position := vec_position_i32(ords, to_i64(14))\n",
            "rank_by_count := vec_count_below_i32(ords, to_i64(14))\n",
            "absent := vec_position_i32(ords, to_i64(7))\n",
            "below_absent := vec_count_below_i32(ords, to_i64(7))\n",
            "matches := vec_len_i32(ords)\n",
        )
    );
    assert_eq!(
        run(
            &src,
            &[
                "rank_by_position",
                "rank_by_count",
                "absent",
                "below_absent",
                "matches"
            ]
        ),
        // A value not in the list positions past the end (4, the
        // length), while the count below it is still meaningful.
        vec!["2", "2", "4", "1", "4"]
    );
}

#[test]
fn membership_and_set_comparisons_ignore_order_and_repetition() {
    let src = concat!(
        "input cycle: u64\n",
        "a := str_to_vec_i32(\"[1, 2, 3]\")\n",
        "b := str_to_vec_i32(\"[3, 2, 1, 1]\")\n",
        "c := str_to_vec_i32(\"[3, 4]\")\n",
        "has := vec_contains_i32(a, to_i64(2))\n",
        "hasnt := vec_contains_i32(a, to_i64(42))\n",
        "same := vec_set_eq_i32(a, b)\n",
        "differ := vec_set_eq_i32(a, c)\n",
        "overlap := vec_intersect_count_i32(a, c)\n",
        "overlap_dup := vec_intersect_count_i32(a, b)\n",
    );
    assert_eq!(
        run(
            src,
            &["has", "hasnt", "same", "differ", "overlap", "overlap_dup"]
        ),
        // `b` reorders `a` and repeats a value: still the same set, and
        // the overlap counts distinct values rather than pairs.
        vec!["1", "0", "1", "0", "1", "3"]
    );
}

#[test]
fn a_name_yields_the_number_in_its_tail() {
    let src = concat!(
        "input cycle: u64\n",
        "whole := str_to_u64(\"4242\")\n",
        "malformed := str_to_u64(\"4242x\")\n",
        "label := str_digit_suffix(\"label-17\")\n",
        "under := str_digit_suffix(\"profile_8\")\n",
        "embedded := str_digit_suffix(\"sift1m\")\n",
        "none := str_digit_suffix(\"plain\")\n",
    );
    assert_eq!(
        run(
            src,
            &["whole", "malformed", "label", "under", "embedded", "none"]
        ),
        // A malformed whole-string parse is a miss, not a partial read.
        // `sift1m` ends in `m`, so the digit run ends before it.
        vec!["4242", "0", "17", "8", "0", "0"]
    );
}

#[test]
fn the_i64_forms_read_the_same_way() {
    let src = concat!(
        "input cycle: u64\n",
        "v := str_to_vec_i64(\"[10, 20, 30]\")\n",
        "n := vec_len_i64(v)\n",
        "at := vec_at_i64(v, 1)\n",
        "hi := vec_max_i64(v)\n",
        "lo := vec_min_i64(v)\n",
        "has := vec_contains_i64(v, to_i64(20))\n",
        "pos := vec_position_i64(v, to_i64(30))\n",
        "below := vec_count_below_i64(v, to_i64(25))\n",
        "same := vec_set_eq_i64(v, v)\n",
        "overlap := vec_intersect_count_i64(v, v)\n",
    );
    assert_eq!(
        run(
            src,
            &[
                "n", "at", "hi", "lo", "has", "pos", "below", "same", "overlap"
            ]
        ),
        vec!["3", "20", "30", "10", "1", "2", "2", "1", "3"]
    );
}

/// The two quantities the host asked a counter for, written as
/// functions of the list: the rank of a record among the matching
/// ones, and how many match. Same answer on every engine, because
/// neither reads an evaluation order.
#[test]
fn the_recall_audit_quantities_are_functions_of_the_list() {
    use polydat::{Engine, JitMode, Provenance};
    let src = concat!(
        "input cycle: u64\n",
        // The ordinals matching one label, and the query's own ordinal.
        "matching := str_to_vec_i32(\"[3, 9, 14, 20, 27]\")\n",
        "ordinal := to_i64(14)\n",
        "rank := vec_count_below_i32(matching, ordinal)\n",
        "cardinality := vec_len_i32(matching)\n",
    );
    for engine in [
        Engine::Interpreter(JitMode::Off),
        Engine::Interpreter(JitMode::Auto),
        Engine::Closures(Provenance::Auto),
        Engine::Native(Provenance::Auto),
    ] {
        let Ok(mut k) = polydat::dsl::compile_polydat_with(src, engine) else {
            continue; // a build without the jit feature refuses native
        };
        k.set_inputs(&[0]);
        assert_eq!(k.pull("rank").as_u64(), 2, "{engine}");
        assert_eq!(k.pull("cardinality").as_u64(), 5, "{engine}");
    }
}

// ── vec_eq ──────────────────────────────────────────────────────────

/// One engine of each kind: the interpreter, the closure tier, native,
/// and pure native when the build has the jit.
fn vec_eq_engines() -> Vec<polydat::Engine> {
    use polydat::{Engine, JitMode, Provenance};
    let mut all = vec![
        Engine::Interpreter(JitMode::Off),
        Engine::Closures(Provenance::Auto),
        Engine::Native(Provenance::Auto),
    ];
    if cfg!(feature = "jit") {
        all.push(Engine::PureNative(Provenance::PushPull));
    }
    all
}

/// Build `vec_eq_<ty>(a, b)` over two extern vectors on every engine,
/// and check each `(label, a, b, expected)` case on one kernel per
/// engine, setting the externs anew for each case.
fn check_vec_eq(ty: &str, cases: &[(&str, polydat::ast::Value, polydat::ast::Value, bool)]) {
    let src = format!(
        "input cycle: u64\n\
         extern a: vec_{ty}\n\
         extern b: vec_{ty}\n\
         out := vec_eq_{ty}(a, b)\n"
    );
    for engine in vec_eq_engines() {
        let asm = polydat::dsl::compile::compile_polydat_to_assembler(&src)
            .unwrap_or_else(|e| panic!("{e}\n{src}"));
        let mut k = asm
            .compile_with(engine)
            .unwrap_or_else(|e| panic!("{engine}: {e}\n{src}"));
        k.set_inputs(&[0]);
        for (label, a, b, expected) in cases {
            k.set_input("a", a.clone())
                .unwrap_or_else(|e| panic!("{engine}: set a: {e}"));
            k.set_input("b", b.clone())
                .unwrap_or_else(|e| panic!("{engine}: set b: {e}"));
            assert_eq!(
                k.pull("out").to_display_string(),
                expected.to_string(),
                "{engine}: vec_eq_{ty} {label}"
            );
        }
    }
}

/// The cases every carrier answers, over three elements of its type
/// built from small integers.
fn common_cases<T: Copy>(
    wrap: fn(Vec<T>) -> polydat::ast::Value,
    of: fn(i8) -> T,
) -> Vec<(&'static str, polydat::ast::Value, polydat::ast::Value, bool)> {
    let v = |xs: &[i8]| wrap(xs.iter().map(|x| of(*x)).collect());
    vec![
        ("equal", v(&[1, 2, 3]), v(&[1, 2, 3]), true),
        ("differing element", v(&[1, 2, 3]), v(&[1, 7, 3]), false),
        ("reordered", v(&[1, 2, 3]), v(&[3, 2, 1]), false),
        ("longer", v(&[1, 2, 3]), v(&[1, 2, 3, 4]), false),
        ("shorter", v(&[1, 2, 3]), v(&[1, 2]), false),
        ("empty vs non-empty", v(&[]), v(&[1]), false),
        ("empty vs empty", v(&[]), v(&[]), true),
    ]
}

/// The float cases: bit identity, not IEEE equality.
fn float_cases<T: Copy>(
    wrap: fn(Vec<T>) -> polydat::ast::Value,
    of: fn(i8) -> T,
    neg_zero: T,
    nan: T,
    other_nan: T,
) -> Vec<(&'static str, polydat::ast::Value, polydat::ast::Value, bool)> {
    let mut cases = common_cases(wrap, of);
    let with = |x: T| wrap(vec![of(1), x]);
    cases.extend([
        ("-0.0 vs 0.0", with(neg_zero), with(of(0)), false),
        ("-0.0 vs -0.0", with(neg_zero), with(neg_zero), true),
        ("NaN vs the same NaN", with(nan), with(nan), true),
        ("NaN vs another payload", with(nan), with(other_nan), false),
    ]);
    cases
}

#[test]
fn vec_eq_f32_compares_bits() {
    use polydat::ast::{SliceArc, Value};
    let (nan, other) = (f32::from_bits(0x7fc0_0000), f32::from_bits(0x7fc0_0001));
    assert!(nan.is_nan() && other.is_nan());
    check_vec_eq(
        "f32",
        &float_cases(
            |v| Value::VecF32(SliceArc::from_vec(v)),
            f32::from,
            -0.0,
            nan,
            other,
        ),
    );
}

#[test]
fn vec_eq_f64_compares_bits() {
    use polydat::ast::{SliceArc, Value};
    let (nan, other) = (
        f64::from_bits(0x7ff8_0000_0000_0000),
        f64::from_bits(0x7ff8_0000_0000_0001),
    );
    assert!(nan.is_nan() && other.is_nan());
    check_vec_eq(
        "f64",
        &float_cases(
            |v| Value::VecF64(SliceArc::from_vec(v)),
            f64::from,
            -0.0,
            nan,
            other,
        ),
    );
}

#[test]
fn vec_eq_f16_compares_bits() {
    use polydat::ast::{SliceArc, Value};
    use polydat::half::f16;
    let (nan, other) = (f16::from_bits(0x7e00), f16::from_bits(0x7e01));
    assert!(nan.is_nan() && other.is_nan());
    check_vec_eq(
        "f16",
        &float_cases(
            |v| Value::VecF16(SliceArc::from_vec(v)),
            |x| f16::from_f32(f32::from(x)),
            f16::NEG_ZERO,
            nan,
            other,
        ),
    );
}

#[test]
fn vec_eq_integer_forms_compare_values() {
    use polydat::ast::{SliceArc, Value};
    check_vec_eq(
        "i8",
        &common_cases(|v| Value::VecI8(SliceArc::from_vec(v)), |x| x),
    );
    check_vec_eq(
        "i16",
        &common_cases(|v| Value::VecI16(SliceArc::from_vec(v)), i16::from),
    );
    check_vec_eq(
        "i32",
        &common_cases(|v| Value::VecI32(SliceArc::from_vec(v)), i32::from),
    );
    check_vec_eq(
        "i64",
        &common_cases(|v| Value::VecI64(SliceArc::from_vec(v)), i64::from),
    );
}

/// Equality is order-sensitive where `vec_set_eq_i32` is not: the same
/// set in another order is a different vector.
#[test]
fn vec_eq_differs_from_set_equality_on_order() {
    let src = concat!(
        "input cycle: u64\n",
        "a := str_to_vec_i32(\"[1, 2, 3]\")\n",
        "b := str_to_vec_i32(\"[3, 2, 1]\")\n",
        "same_set := vec_set_eq_i32(a, b)\n",
        "same_vec := vec_eq_i32(a, b)\n",
        "self_eq := vec_eq_i32(a, a)\n",
    );
    assert_eq!(
        run(src, &["same_set", "same_vec", "self_eq"]),
        vec!["1", "false", "true"]
    );
}
