// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Named generators as comprehension sources (comprehension_forms.md
//! §3.1.3), on every surface.
//!
//! A generator's values are typed: an integer above `i64::MAX` is a
//! `u64` wherever it travels. Over generator sources whose values reach
//! and pass `i64::MAX`, alone and composed, the traversal evaluators,
//! the scope-less stream, a producer's stream, and a DSL traversal yield
//! the same values in the same order, and each reports the count it
//! yields. A call past a generator's limit, or with an argument out of
//! range, is a compile error naming the call and the argument when its
//! arguments are constants, and an error when it opens when they come
//! from the program's inputs.

use polydat::ast::Value;
use polydat::iteration::comprehension::ast::Comprehension;
use polydat::iteration::comprehension::cardinality::CardinalityClass;
use polydat::iteration::comprehension::runtime::{
    evaluate_for_iteration_materialized, evaluate_indexed,
};
use polydat::iteration::comprehension::source::{LiteralValue, Source};
use polydat::iteration::comprehension::strategies::{Tuple, TupleValue};
use polydat::iteration::comprehension::strategy::{StrategyName, ZipMode};

/// One bound value, exactly: an integer as its mathematical value, so a
/// `u64` above `i64::MAX` read back as a negative `i64` differs.
#[derive(Debug, Clone, PartialEq)]
enum Cell {
    Int(i128),
    Float(u64),
    Other(String),
}

fn value_cell(v: &Value) -> Cell {
    match v {
        Value::U64(n) => Cell::Int(i128::from(*n)),
        Value::I64(n) => Cell::Int(i128::from(*n)),
        Value::F64(f) => Cell::Float(f.to_bits()),
        other => Cell::Other(other.to_display_string()),
    }
}

fn tuple_cell(v: &TupleValue) -> Cell {
    match v {
        TupleValue::U64(n) => Cell::Int(i128::from(*n)),
        TupleValue::I64(n) => Cell::Int(i128::from(*n)),
        TupleValue::F64(f) => Cell::Float(f.to_bits()),
        other => Cell::Other(format!("{other:?}")),
    }
}

type Row = Vec<(String, Cell)>;

fn stream_rows(tuples: &[Tuple]) -> Vec<Row> {
    tuples
        .iter()
        .map(|t| {
            t.bindings
                .iter()
                .map(|(n, v)| (n.clone(), tuple_cell(v)))
                .collect()
        })
        .collect()
}

/// Every integer above `i64::MAX` a stream dispenses is a `U64`.
fn assert_unsigned_above_i64(tuples: &[Tuple], what: &str) {
    for t in tuples {
        for (name, v) in &t.bindings {
            if let TupleValue::I64(n) = v {
                assert!(
                    *n >= 0,
                    "{what}: `{name}` dispensed as the negative I64 {n}"
                );
            }
        }
    }
}

fn generator(name: &str, expr: &str) -> Comprehension {
    Comprehension::clause(
        name,
        Source::Generator {
            expr: expr.into(),
            cardinality_hint: None,
        },
    )
}

fn ints(name: &str, vs: &[i64]) -> Comprehension {
    Comprehension::clause(
        name,
        Source::Literal {
            values: vs.iter().map(|v| LiteralValue::Int(*v)).collect(),
        },
    )
}

fn scope() -> polydat::kernel::PolydatKernel {
    polydat::dsl::compile_polydat_interpreter("input cycle: u64\n").unwrap()
}

/// The traversal's rows for `ast`, with both evaluators and the stream
/// held to them, and the reported count to the yield.
fn traverse_and_stream(ast: &Comprehension) -> Vec<Row> {
    let scope = scope();
    let reference = evaluate_for_iteration_materialized(ast, &scope)
        .unwrap_or_else(|e| panic!("{ast:?}: {e}"))
        .tuples;
    let indexed = evaluate_indexed(ast, &scope)
        .unwrap_or_else(|e| panic!("{ast:?}: {e}"))
        .to_vec();
    assert_eq!(indexed, reference, "the evaluators differ for {ast:?}");
    let rows: Vec<Row> = reference
        .iter()
        .map(|t| t.iter().map(|(n, v)| (n.clone(), value_cell(v))).collect())
        .collect();
    let compiled = polydat::iteration::comprehension::surfaces::compile(ast)
        .unwrap_or_else(|e| panic!("{ast:?}: {e}"));
    let tuples: Vec<Tuple> = compiled
        .coordinate_stream()
        .map(|t| t.unwrap_or_else(|e| panic!("{ast:?}: {e}")))
        .collect();
    assert_unsigned_above_i64(&tuples, &format!("{ast:?}"));
    assert_eq!(
        stream_rows(&tuples),
        rows,
        "the stream's tuples differ from the traversal's for {ast:?}"
    );
    if let CardinalityClass::Bounded(n) =
        polydat::iteration::comprehension::flatten::flatten_static_sources(
            ast,
            &polydat::kernel::interp::NoScope::new(),
        )
        .metadata()
        .cardinality
    {
        assert_eq!(n, rows.len() as u64, "the count reported for {ast:?}");
    }
    rows
}

/// The generator calls whose values reach and pass `i64::MAX`, with
/// their counts and last values.
const NEAR_AND_ABOVE: &[(&str, usize, u64)] = &[
    ("pow2(63)", 63, 1 << 62),
    ("pow2(64)", 64, 1 << 63),
    ("pow2_until(18446744073709551615)", 64, 1 << 63),
    ("fib(92)", 92, 7_540_113_804_746_346_429),
    ("fib(93)", 93, 12_200_160_415_121_876_738),
    (
        "fib_until(18446744073709551615)",
        93,
        12_200_160_415_121_876_738,
    ),
    ("binomial(66)", 67, 1),
    ("binomial(67)", 68, 1),
];

#[test]
fn every_surface_yields_the_values_of_a_generator_above_i64_max() {
    for (call, count, last) in NEAR_AND_ABOVE {
        let rows = traverse_and_stream(&generator("k", call));
        assert_eq!(rows.len(), *count, "{call}");
        assert_eq!(
            rows.last().unwrap()[0].1,
            Cell::Int(i128::from(*last)),
            "{call}"
        );
        let producer = producer_rows(call);
        assert_eq!(producer, rows, "the producer over {call} differs");
        let body = dsl_traversal(call);
        assert_eq!(
            body,
            rows.iter()
                .map(|r| match r[0].1 {
                    Cell::Int(n) => u64::try_from(n).unwrap(),
                    ref other => panic!("{call}: {other:?}"),
                })
                .collect::<Vec<_>>(),
            "the DSL traversal over {call} differs"
        );
    }
}

/// Composed with literals, zipped, unioned, ordered, and filtered, a
/// generator above `i64::MAX` yields the same tuples on the stream as
/// on the traversal.
#[test]
fn compositions_over_a_generator_above_i64_max_agree() {
    let shapes = [
        Comprehension::cartesian(vec![generator("k", "pow2(64)"), ints("j", &[1, 2])]),
        Comprehension::cartesian(vec![ints("j", &[1, 2]), generator("k", "fib(93)")]),
        Comprehension::zip(
            vec![generator("k", "pow2(64)"), generator("f", "fib(64)")],
            ZipMode::Strict,
        ),
        Comprehension::zip(
            vec![generator("k", "binomial(67)"), ints("j", &[1, 2, 3])],
            ZipMode::Cycle,
        ),
        Comprehension::union(vec![generator("k", "pow2(64)"), generator("k", "fib(93)")]),
        Comprehension::order(generator("k", "fib(93)"), StrategyName::Lex, Some(90)),
        Comprehension::order_seeded(
            generator("k", "pow2(64)"),
            StrategyName::Shuffle,
            Some(10),
            Some(7),
        ),
        Comprehension::order(generator("k", "binomial(67)"), StrategyName::Extrema, None),
        Comprehension::filter(generator("k", "pow2(64)"), "{k} > 1000"),
    ];
    for ast in &shapes {
        let rows = traverse_and_stream(ast);
        assert!(!rows.is_empty(), "{ast:?}");
    }
}

/// The values a producer `s := for k in <call>` dispenses, after
/// checking the count it reports.
fn producer_rows(call: &str) -> Vec<Row> {
    let src = format!("input cycle: u64\ns := for k in {call}\n");
    let mut kernel =
        polydat::dsl::compile_polydat_interpreter(&src).unwrap_or_else(|e| panic!("{e}\n{src}"));
    kernel.set_inputs(&[0]);
    let value = kernel.pull_ref("s").clone();
    let producer = value.as_streamer().expect("a producer is a streamer");
    let tuples: Vec<Tuple> = producer
        .coordinate_stream()
        .unwrap_or_else(|e| panic!("{call}: {e}"))
        .map(|t| t.unwrap_or_else(|e| panic!("{call}: {e}")))
        .collect();
    assert_unsigned_above_i64(&tuples, call);
    assert_eq!(
        producer.cardinality(),
        CardinalityClass::Bounded(tuples.len() as u64),
        "the producer over {call}"
    );
    stream_rows(&tuples)
}

/// The values a DSL traversal `for k in <call> { v := u64_add(k, 0) }`
/// binds, one per activation.
fn dsl_traversal(call: &str) -> Vec<u64> {
    let src = format!("input cycle: u64\nfor k in {call} {{\n    v := u64_add(k, 0)\n}}\n");
    let mut kernel =
        polydat::dsl::compile_polydat_interpreter(&src).unwrap_or_else(|e| panic!("{e}\n{src}"));
    kernel.set_inputs(&[0]);
    let mut stream = kernel.traverse(0).unwrap_or_else(|e| panic!("{call}: {e}"));
    let mut seen = Vec::new();
    while let Some(mut a) = stream.advance().unwrap_or_else(|e| panic!("{call}: {e}")) {
        seen.push(a.cycle(0).pull("v").as_u64());
    }
    seen
}

/// Calls a compile refuses, each with the text its error must carry:
/// the call's own refusal, which names the argument or the first term
/// past `u64::MAX` and the largest valid argument.
const REFUSED: &[(&str, &str)] = &[
    (
        "fib(94)",
        "fib(94): term 94 is past u64::MAX; fib.n is at most 93",
    ),
    (
        "pow2(65)",
        "pow2(65): term 65, 2^64, is past u64::MAX; pow2.n is at most 64",
    ),
    (
        "binomial(68)",
        "binomial(68): term C(68, 31) is past u64::MAX; binomial.n is at most 67",
    ),
    ("fib(-1)", "fib.n: expected non-negative integer, got '-1'"),
    ("fib(1, 2)", "fib(n): expected 1 argument, got 2"),
    (
        "geometric(1, 0, 4)",
        "geometric.factor: expected a positive, finite number, got 0",
    ),
    (
        "geometric_until(1, 1, 100)",
        "geometric_until.factor: expected a finite number greater than 1, got 1",
    ),
    (
        "log_steps(0, 10, 3)",
        "log_steps.start: expected a positive number, got 0",
    ),
    (
        "concat(1..3, pow2(65))",
        "pow2(65): term 65, 2^64, is past u64::MAX",
    ),
];

/// A call whose arguments are constants is evaluated when the
/// comprehension compiles: on the DSL `for` path and in a producer, a
/// refused call is the compile's error, naming the statement and the
/// call's refusal.
#[test]
fn a_constant_call_that_fails_is_a_compile_error() {
    for (call, message) in REFUSED {
        for src in [
            format!("input cycle: u64\nfor k in {call} {{\n    v := u64_add(k, 0)\n}}\n"),
            format!("input cycle: u64\ns := for k in {call}\n"),
        ] {
            let err = match polydat::dsl::compile_polydat_interpreter(&src) {
                Ok(_) => panic!("the compile accepted {call}:\n{src}"),
                Err(e) => e.to_string(),
            };
            assert!(
                err.contains(message) && err.contains(&format!("for k in {call}")),
                "{call}: {err}"
            );
        }
        // The scope-less stream refuses it at its compile too.
        let err = polydat::iteration::comprehension::surfaces::compile(&generator("k", call))
            .expect_err(call)
            .to_string();
        assert!(err.contains(message), "{call}: {err}");
    }
}

/// A call whose argument comes from an input compiles, inline and in a
/// producer, and is checked when the traversal over it opens.
#[test]
fn a_call_over_an_input_is_checked_when_it_opens() {
    for src in [
        "input cycle: u64\ninput n: u64\nfor k in fib({n}) {\n    v := u64_add(k, 0)\n}\n",
        "input cycle: u64\ninput n: u64\ns := for k in fib({n})\nfor s {\n    v := u64_add(k, 0)\n}\n",
    ] {
        let mut kernel =
            polydat::dsl::compile_polydat_interpreter(src).unwrap_or_else(|e| panic!("{e}\n{src}"));
        kernel.set_inputs(&[0, 93]);
        let mut stream = kernel.traverse(0).unwrap();
        let mut last = 0;
        let mut n = 0;
        while let Some(mut a) = stream.advance().unwrap() {
            last = a.cycle(0).pull("v").as_u64();
            n += 1;
        }
        assert_eq!((n, last), (93, 12_200_160_415_121_876_738), "{src}");
        kernel.set_inputs(&[0, 94]);
        let err = kernel
            .traverse(0)
            .and_then(|mut s| {
                while s.advance().map_err(|e| e.to_string())?.is_some() {}
                Ok(())
            })
            .expect_err("fib(94) opened");
        assert!(
            err.contains("fib(94): term 94 is past u64::MAX; fib.n is at most 93"),
            "{src}\n{err}"
        );
    }
}

/// A call whose argument comes from an extern compiles, and is checked
/// when the traversal over it opens, with the extern's value then.
#[test]
fn a_call_over_an_extern_is_checked_when_it_opens() {
    let src =
        "input cycle: u64\nextern n: u64 = 94\nfor k in fib({n}) {\n    v := u64_add(k, 0)\n}\n";
    let mut kernel =
        polydat::dsl::compile_polydat_interpreter(src).unwrap_or_else(|e| panic!("{e}\n{src}"));
    kernel.set_inputs(&[0]);
    let err = kernel
        .traverse(0)
        .and_then(|mut s| {
            while s.advance().map_err(|e| e.to_string())?.is_some() {}
            Ok(())
        })
        .expect_err("fib(94) opened");
    assert!(
        err.contains("fib(94): term 94 is past u64::MAX; fib.n is at most 93"),
        "{err}"
    );
}
