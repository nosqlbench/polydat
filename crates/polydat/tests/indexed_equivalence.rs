// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! The index-addressed evaluator against the reference evaluator
//! (comprehension_forms.md §10.2 R2, §15.1).
//!
//! `evaluate_indexed` holds a comprehension's tuples addressed by
//! position and computes one only when it is asked for;
//! `evaluate_for_iteration_materialized` materializes every node. For
//! every comprehension shape the two must yield exactly the same
//! tuples in exactly the same order, fail with the same error, and
//! report the same clause yields. The shapes here are fixed cases for
//! each constructor and a seeded generator over their compositions:
//! cartesians (independent and dependent), the three zip modes, unions,
//! every order strategy with and without truncation and seed, filters,
//! and sampled continuous spaces. On a shape with discrete sources and
//! constant filters that the scope-less streaming surface compiles,
//! every tuple binds every name of the shape and that surface yields
//! the same count of tuples binding the same names; where the traversal
//! refuses a shape, a stream that reaches the fault fails with the same
//! error after the tuples before it.

use polydat::iteration::comprehension::ast::Comprehension;
use polydat::iteration::comprehension::cardinality::{Interval, ProductMeasure};
use polydat::iteration::comprehension::runtime::{
    RuntimeError, evaluate_for_iteration_materialized, evaluate_for_iteration_reported,
    evaluate_indexed,
};
use polydat::iteration::comprehension::source::{LiteralValue, Source};
use polydat::iteration::comprehension::strategies::Tuple;
use polydat::iteration::comprehension::strategy::{StrategyName, ZipMode};

fn scope() -> polydat::kernel::PolydatKernel {
    polydat::dsl::compile_polydat_interpreter("input cycle: u64\n").unwrap()
}

/// Assert the two evaluators agree on `ast`, returning the tuple count.
fn assert_equivalent(ast: &Comprehension, scope: &polydat::kernel::PolydatKernel) -> usize {
    let reference = evaluate_for_iteration_materialized(ast, scope);
    let indexed = evaluate_indexed(ast, scope);
    match (reference, indexed) {
        (Ok(reference), Ok(indexed)) => {
            assert_eq!(
                indexed.len(),
                reference.tuples.len() as u64,
                "tuple count differs for {ast:?}"
            );
            assert_eq!(
                indexed.to_vec(),
                reference.tuples,
                "tuples differ for {ast:?}"
            );
            // Random access answers every position alike, in any order.
            for i in (0..indexed.len()).rev().step_by(7) {
                assert_eq!(
                    indexed.get(i).as_ref(),
                    reference.tuples.get(i as usize),
                    "position {i} differs for {ast:?}"
                );
            }
            assert_eq!(indexed.get(indexed.len()), None);
            let reported = evaluate_for_iteration_reported(ast, scope).unwrap();
            assert_eq!(
                reported.clauses, reference.clauses,
                "clause yields differ for {ast:?}"
            );
            assert_eq!(reported.tuples, reference.tuples);
            if let Some(streamed) = constant_filters(ast).then(|| streamed(ast)).flatten() {
                assert!(
                    streamed.error.is_none(),
                    "the streaming surface failed where the traversal did not, for {ast:?}: {:?}",
                    streamed.error
                );
                let names = shape_names(ast);
                for tuple in &reference.tuples {
                    assert_eq!(
                        tuple_names(tuple.iter().map(|(n, _)| n)),
                        names,
                        "a tuple does not bind every name of {ast:?}"
                    );
                }
                assert_eq!(
                    streamed.tuples.len(),
                    reference.tuples.len(),
                    "the streaming surface's tuple count differs for {ast:?}"
                );
                assert!(
                    streamed.names().iter().all(|t| *t == names),
                    "a streamed tuple does not bind every name of {ast:?}"
                );
            }
            reference.tuples.len()
        }
        (Err(reference), Err(indexed)) => {
            assert_eq!(
                indexed.to_string(),
                reference.to_string(),
                "errors differ for {ast:?}"
            );
            // A strict mismatch is the one refusal the streaming surface
            // shares; others (a strategy refusing its input's shape)
            // are the traversal's alone.
            if matches!(reference, RuntimeError::ZipLengthMismatch { .. })
                && let Some(streamed) = constant_filters(ast).then(|| streamed(ast)).flatten()
            {
                assert_stream_failure(ast, &streamed, &reference);
            }
            0
        }
        (reference, indexed) => panic!(
            "one evaluator failed for {ast:?}:\n  reference: {:?}\n  indexed: {:?}",
            reference.map(|r| r.tuples.len()),
            indexed.map(|t| t.len())
        ),
    }
}

/// A shape's names, sorted.
fn shape_names(ast: &Comprehension) -> Vec<String> {
    tuple_names(ast.coordinate_names().iter())
}

fn tuple_names<'a>(names: impl Iterator<Item = &'a String>) -> Vec<String> {
    let mut names: Vec<String> = names.cloned().collect();
    names.sort();
    names
}

/// What the scope-less streaming surface dispensed: its tuples, and the
/// error that ended it, if one did.
struct Streamed {
    tuples: Vec<Tuple>,
    error: Option<RuntimeError>,
}

impl Streamed {
    /// The sorted names of each tuple.
    fn names(&self) -> Vec<Vec<String>> {
        self.tuples
            .iter()
            .map(|t| tuple_names(t.bindings.iter().map(|(n, _)| n)))
            .collect()
    }
}

/// Drain the scope-less streaming surface over `ast`, or `None` when
/// that surface refuses it: an invalid shape, or a source that needs a
/// scope.
fn streamed(ast: &Comprehension) -> Option<Streamed> {
    let compiled = polydat::iteration::comprehension::surfaces::compile(ast).ok()?;
    let mut stream = compiled.coordinate_stream();
    let mut tuples = Vec::new();
    loop {
        match stream.advance() {
            Ok(Some(t)) => tuples.push(t),
            Ok(None) => {
                return Some(Streamed {
                    tuples,
                    error: None,
                });
            }
            Err(e) => {
                // A failed stream keeps failing.
                assert!(stream.advance().is_err(), "the failure did not persist");
                return Some(Streamed {
                    tuples,
                    error: Some(e),
                });
            }
        }
    }
}

fn strict_zips(c: &Comprehension) -> usize {
    match c {
        Comprehension::Clause { .. } => 0,
        Comprehension::Zip { children, mode } => {
            usize::from(*mode == ZipMode::Strict) + children.iter().map(strict_zips).sum::<usize>()
        }
        Comprehension::Cartesian { children } | Comprehension::Union { children } => {
            children.iter().map(strict_zips).sum()
        }
        Comprehension::Filter { child, .. } | Comprehension::Order { child, .. } => {
            strict_zips(child)
        }
    }
}

/// `ast` with every strict zip made truncating: the same tuples up to
/// the point a strict zip's operands end apart.
fn truncated(c: &Comprehension) -> Comprehension {
    match c {
        Comprehension::Clause { .. } => c.clone(),
        Comprehension::Cartesian { children } => {
            Comprehension::cartesian(children.iter().map(truncated).collect())
        }
        Comprehension::Zip { children, mode } => Comprehension::zip(
            children.iter().map(truncated).collect(),
            match mode {
                ZipMode::Strict => ZipMode::Truncate,
                other => *other,
            },
        ),
        Comprehension::Union { children } => {
            Comprehension::union(children.iter().map(truncated).collect())
        }
        Comprehension::Filter { child, predicate } => {
            Comprehension::filter(truncated(child), predicate.clone())
        }
        Comprehension::Order {
            child,
            strategy,
            truncation,
            seed,
        } => Comprehension::order_seeded(truncated(child), *strategy, *truncation, *seed),
    }
}

/// A stream over a shape the traversal refuses with `expected`: when it
/// fails, it fails with the same kind of error, after a prefix of the
/// tuples the shape dispenses with every strict zip truncating. A stream
/// reports a mismatch only where it pulls one, so a stream that never
/// reaches the mismatch ends without it; with several strict zips, the
/// traversal, which evaluates every operand at open, and the stream,
/// which pulls, can reach different ones first, so the lengths the
/// error names must match only when the shape has one strict zip.
/// Returns whether the stream failed.
fn assert_stream_failure(
    ast: &Comprehension,
    streamed: &Streamed,
    expected: &RuntimeError,
) -> bool {
    let Some(error) = &streamed.error else {
        return false;
    };
    assert_eq!(
        std::mem::discriminant(error),
        std::mem::discriminant(expected),
        "the stream's error kind differs for {ast:?}: {error} and {expected}"
    );
    if strict_zips(ast) == 1 {
        assert_eq!(
            error.to_string(),
            expected.to_string(),
            "the stream's error differs for {ast:?}"
        );
    }
    let reference = self::streamed(&truncated(ast))
        .expect("the truncated shape compiles")
        .tuples;
    assert!(
        streamed.tuples.len() <= reference.len()
            && streamed.tuples[..] == reference[..streamed.tuples.len()],
        "the stream delivered other tuples before its error for {ast:?}"
    );
    true
}

/// Whether every filter in `c` is the constant `true` or `false` and
/// no clause is continuous: the shapes on which the streaming surface
/// and the traversal evaluators share predicate and sampling semantics,
/// so their tuple counts are comparable.
fn constant_filters(c: &Comprehension) -> bool {
    match c {
        Comprehension::Clause { source, .. } => {
            !matches!(source, Source::ContinuousInterval { .. })
        }
        Comprehension::Cartesian { children }
        | Comprehension::Zip { children, .. }
        | Comprehension::Union { children } => children.iter().all(constant_filters),
        Comprehension::Filter { child, predicate } => {
            matches!(predicate.as_str(), "true" | "false") && constant_filters(child)
        }
        Comprehension::Order { child, .. } => constant_filters(child),
    }
}

fn ints(name: &str, vs: &[i64]) -> Comprehension {
    Comprehension::clause(
        name,
        Source::Literal {
            values: vs.iter().map(|v| LiteralValue::Int(*v)).collect(),
        },
    )
}

fn range(name: &str, lo: i64, hi: i64, step: i64) -> Comprehension {
    Comprehension::clause(name, Source::IntRange { lo, hi, step })
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

fn unit(name: &str) -> Comprehension {
    Comprehension::clause(
        name,
        Source::ContinuousInterval {
            interval: Interval::closed(0.0, 1.0),
            measure: ProductMeasure::Uniform,
        },
    )
}

const STRATEGIES: [StrategyName; 10] = [
    StrategyName::Lex,
    StrategyName::ReverseLex,
    StrategyName::Diagonal,
    StrategyName::Antidiagonal,
    StrategyName::Extrema,
    StrategyName::Shells,
    StrategyName::Halton,
    StrategyName::Sobol,
    StrategyName::Lhs,
    StrategyName::Shuffle,
];

/// Every constructor, and every strategy over each combinator.
#[test]
fn every_shape_indexes_as_it_materializes() {
    let scope = scope();
    let product = || Comprehension::cartesian(vec![range("a", 0, 4, 1), ints("b", &[5, 6, 7])]);
    let dependent = || {
        Comprehension::cartesian(vec![
            range("a", 1, 5, 1),
            generator("b", "0..{a}"),
            ints("c", &[1, 2]),
        ])
    };
    let lockstep =
        |mode| Comprehension::zip(vec![range("x", 0, 5, 1), ints("y", &[1, 2, 3])], mode);
    let union = || {
        Comprehension::union(vec![
            Comprehension::cartesian(vec![ints("k", &[1, 2]), range("m", 0, 3, 1)]),
            Comprehension::cartesian(vec![ints("k", &[9]), range("m", 5, 7, 1)]),
        ])
    };
    let mut shapes = vec![
        range("a", 0, 10, 3),
        range("a", 5, 5, 1),
        range("a", -3, 4, 2),
        ints("a", &[]),
        ints("a", &[3, 3, 1]),
        generator("a", "10..14"),
        product(),
        dependent(),
        Comprehension::cartesian(vec![ints("a", &[]), range("b", 0, 3, 1)]),
        Comprehension::cartesian(vec![
            range("a", 0, 3, 1),
            ints("b", &[]),
            range("c", 0, 3, 1),
        ]),
        Comprehension::cartesian(vec![
            lockstep(ZipMode::Strict),
            Comprehension::cartesian(vec![ints("p", &[1, 2]), ints("q", &[3])]),
        ]),
        Comprehension::cartesian(vec![
            range("o", 0, 3, 1),
            Comprehension::union(vec![
                Comprehension::cartesian(vec![ints("k", &[1]), generator("m", "0..{o}")]),
                Comprehension::cartesian(vec![ints("k", &[2]), range("m", 0, 2, 1)]),
            ]),
        ]),
        lockstep(ZipMode::Strict),
        lockstep(ZipMode::Truncate),
        lockstep(ZipMode::Cycle),
        Comprehension::zip(vec![range("x", 0, 5, 1), ints("y", &[])], ZipMode::Cycle),
        Comprehension::zip(
            vec![
                Comprehension::filter(range("x", 0, 9, 1), "{x} > 5"),
                product(),
                ints("z", &[1, 2, 3, 4, 5]),
            ],
            ZipMode::Cycle,
        ),
        union(),
        Comprehension::filter(product(), "{a} != 2 && {b} > 5"),
        Comprehension::filter(dependent(), "{b} < 2"),
        Comprehension::filter(product(), "u64_add({a}, {b}) > 7"),
        Comprehension::order(
            Comprehension::cartesian(vec![range("a", 0, 3, 1), unit("u")]),
            StrategyName::Halton,
            Some(5),
        ),
        Comprehension::order(
            Comprehension::filter(
                Comprehension::cartesian(vec![unit("u"), unit("v")]),
                "{u} > 0.5",
            ),
            StrategyName::Sobol,
            Some(4),
        ),
    ];
    for strategy in STRATEGIES {
        for truncation in [None, Some(1), Some(4), Some(100)] {
            for seed in [None, Some(7)] {
                for input in [
                    product(),
                    dependent(),
                    lockstep(ZipMode::Truncate),
                    lockstep(ZipMode::Cycle),
                    union(),
                    Comprehension::filter(product(), "{b} != 6"),
                    Comprehension::order(product(), StrategyName::Shuffle, Some(9)),
                ] {
                    shapes.push(Comprehension::order_seeded(
                        input, strategy, truncation, seed,
                    ));
                }
            }
        }
    }
    for shape in &shapes {
        assert_equivalent(shape, &scope);
    }
}

/// An empty operand empties a cycle zip on every path: every tuple
/// binds every name, and an empty operand has no tuple to cycle. The
/// operand may be empty as written, after a filter, or as a generator
/// that yields nothing, in any position, and the zip may sit inside a
/// cartesian or a union.
#[test]
fn an_empty_operand_empties_a_cycle_zip() {
    use polydat::iteration::comprehension::CardinalityClass;
    let scope = scope();
    let cycle = |children| Comprehension::zip(children, ZipMode::Cycle);
    let empties: Vec<fn(&str) -> Comprehension> = vec![
        |n| ints(n, &[]),
        |n| range(n, 3, 3, 1),
        |n| Comprehension::filter(range(n, 0, 6, 1), format!("{{{n}}} > 9")),
        |n| Comprehension::filter(ints(n, &[1, 2]), "false"),
        |n| generator(n, "4..4"),
    ];
    let mut shapes = Vec::new();
    for empty in &empties {
        let zips = [
            cycle(vec![range("k", 1, 5, 1), empty("color")]),
            cycle(vec![empty("color"), range("k", 1, 5, 1)]),
            cycle(vec![
                range("k", 1, 5, 1),
                empty("color"),
                ints("size", &[7, 8]),
            ]),
            cycle(vec![
                Comprehension::filter(range("k", 0, 9, 1), "{k} > 2"),
                ints("size", &[7, 8]),
                empty("color"),
            ]),
        ];
        // The zip's metadata never counts a tuple it cannot yield.
        for zip in &zips {
            match zip.metadata().cardinality {
                CardinalityClass::Bounded(0)
                | CardinalityClass::BoundedAtMost(_)
                | CardinalityClass::Unbounded => {}
                other => panic!("{other:?} for {zip:?}"),
            }
        }
        shapes.extend(zips);
        shapes.push(Comprehension::cartesian(vec![
            ints("p", &[1, 2]),
            cycle(vec![range("k", 1, 5, 1), empty("color")]),
        ]));
        shapes.push(Comprehension::union(vec![
            cycle(vec![range("k", 1, 5, 1), empty("color")]),
            cycle(vec![range("k", 1, 5, 1), empty("color")]),
        ]));
        shapes.push(Comprehension::order(
            cycle(vec![empty("color"), range("k", 1, 5, 1)]),
            StrategyName::Shuffle,
            Some(3),
        ));
    }
    for shape in &shapes {
        assert_eq!(assert_equivalent(shape, &scope), 0, "{shape:?}");
        assert_eq!(evaluate_indexed(shape, &scope).unwrap().len(), 0);
        // The streaming surface yields nothing where it compiles the
        // shape; it refuses a non-Lex order over an operand that is not
        // addressable.
        let streamed = streamed(shape);
        assert!(
            streamed
                .as_ref()
                .is_none_or(|s| s.tuples.is_empty() && s.error.is_none()),
            "{:?} for {shape:?}",
            streamed.as_ref().map(Streamed::names)
        );
        if !matches!(shape, Comprehension::Order { .. }) {
            assert!(
                streamed.is_some(),
                "the streaming surface refused {shape:?}"
            );
        }
    }

    // Emptiness known from the operand's metadata empties the zip's
    // metadata and holds nothing.
    let known = cycle(vec![
        Comprehension::filter(range("k", 0, 9, 1), "{k} > 2"),
        Comprehension::filter(ints("size", &[7, 8]), "{size} > 0"),
        ints("color", &[]),
    ]);
    let meta = known.metadata();
    assert_eq!(meta.cardinality, CardinalityClass::Bounded(0));
    assert_eq!(
        meta.materialization,
        polydat::iteration::comprehension::Materialization::Streaming
    );
    let bounds = polydat::iteration::comprehension::ir::check_bounds(
        &polydat::iteration::comprehension::ir::compile(&known),
    );
    assert!(bounds.barriers.is_empty(), "{bounds:?}");
    // An operand that may be empty at open makes the count an upper
    // bound.
    let maybe = cycle(vec![
        range("k", 0, 5, 1),
        Comprehension::filter(ints("size", &[7, 8]), "{size} > 7"),
    ]);
    assert_eq!(
        maybe.metadata().cardinality,
        CardinalityClass::BoundedAtMost(5)
    );

    // A traversal and a producer over the same zip, written in the
    // language.
    let src = "input cycle: u64\n\
               sweep := for (k, color) in zip_cycle(1..5, 5..5)\n\
               for (k, color) in zip_cycle(1..5, 5..5) {\n    \
               s := u64_add(k, color)\n}\n";
    let mut kernel = polydat::dsl::compile_polydat_interpreter(src).unwrap();
    kernel.set_inputs(&[0]);
    let mut stream = kernel.traverse(0).unwrap();
    assert_eq!(stream.len(), 0);
    assert!(stream.advance().unwrap().is_none());
    let sweep = kernel.pull_ref("sweep").clone();
    let streamer = sweep.as_streamer().unwrap();
    assert_eq!(streamer.cardinality(), CardinalityClass::Bounded(0));
    assert_eq!(streamer.coordinate_stream().unwrap().count(), 0);
}

/// A strict zip whose operands end apart fails on every path with the
/// same error naming each operand's length. The traversal evaluators
/// fail at open; a stream delivers the tuples before the mismatch and
/// fails where it finds it. An empty operand is a mismatch unless every
/// operand is empty.
#[test]
fn a_strict_mismatch_fails_on_every_path() {
    let scope = scope();
    let strict = |children| Comprehension::zip(children, ZipMode::Strict);
    let names = || ints("c", &[10, 20]);
    // Each shape, the lengths its error names, and the tuples a stream
    // delivers before the error.
    let cases: Vec<(Comprehension, &str, usize)> = vec![
        (strict(vec![range("k", 1, 5, 1), names()]), "[4, 2]", 2),
        (strict(vec![names(), range("k", 1, 5, 1)]), "[2, 4]", 2),
        (
            strict(vec![ints("a", &[1, 2, 3]), ints("b", &[4, 5, 6]), names()]),
            "[3, 3, 2]",
            2,
        ),
        (strict(vec![ints("a", &[]), names()]), "[0, 2]", 0),
        (strict(vec![names(), ints("a", &[])]), "[2, 0]", 0),
        (
            strict(vec![
                range("k", 0, 4, 1),
                Comprehension::filter(range("f", 0, 6, 1), "{f} > 2"),
            ]),
            "[4, 3]",
            3,
        ),
        (
            strict(vec![
                range("k", 0, 4, 1),
                Comprehension::filter(range("f", 0, 6, 1), "false"),
            ]),
            "[4, 0]",
            0,
        ),
        // A cartesian caches its later axes before it emits, so a
        // mismatch there fails before any tuple; as its first axis the
        // zip's tuples come first.
        (
            Comprehension::cartesian(vec![
                ints("p", &[1, 2]),
                strict(vec![range("k", 1, 5, 1), names()]),
            ]),
            "[4, 2]",
            0,
        ),
        (
            Comprehension::cartesian(vec![
                strict(vec![range("k", 1, 5, 1), names()]),
                ints("p", &[1, 2]),
            ]),
            "[4, 2]",
            4,
        ),
        (
            Comprehension::union(vec![
                strict(vec![ints("k", &[1, 2]), ints("c", &[3, 4])]),
                strict(vec![ints("k", &[5, 6, 7]), ints("c", &[8])]),
            ]),
            "[3, 1]",
            3,
        ),
        (
            Comprehension::filter(strict(vec![range("k", 1, 5, 1), names()]), "true"),
            "[4, 2]",
            2,
        ),
        (
            Comprehension::order(
                strict(vec![range("k", 1, 5, 1), names()]),
                StrategyName::Shuffle,
                Some(3),
            ),
            "[4, 2]",
            0,
        ),
        (
            Comprehension::zip(
                vec![
                    strict(vec![range("k", 1, 5, 1), names()]),
                    ints("z", &[1, 2, 3]),
                ],
                ZipMode::Cycle,
            ),
            "[4, 2]",
            0,
        ),
    ];
    for (shape, lengths, before) in &cases {
        let expected = format!("zip strict: child lengths differ ({lengths})");
        let traversal = evaluate_for_iteration_materialized(shape, &scope).unwrap_err();
        assert!(
            matches!(traversal, RuntimeError::ZipLengthMismatch { .. }),
            "{traversal:?}"
        );
        assert_eq!(traversal.to_string(), expected, "{shape:?}");
        assert_eq!(assert_equivalent(shape, &scope), 0);
        let streamed = streamed(shape).expect("the streaming surface compiles");
        assert!(
            assert_stream_failure(shape, &streamed, &traversal),
            "the stream did not fail for {shape:?}"
        );
        assert_eq!(streamed.tuples.len(), *before, "{shape:?}");
        assert_eq!(
            streamed.error.as_ref().map(ToString::to_string),
            Some(expected),
            "{shape:?}"
        );
    }

    // Operands that end together, empty ones included, end the zip
    // without an error.
    for shape in [
        strict(vec![ints("a", &[]), ints("b", &[])]),
        strict(vec![range("k", 1, 3, 1), names()]),
        strict(vec![
            Comprehension::filter(range("k", 0, 4, 1), "{k} > 1"),
            names(),
        ]),
    ] {
        let count = assert_equivalent(&shape, &scope);
        let streamed = streamed(&shape).unwrap();
        assert!(streamed.error.is_none(), "{shape:?}");
        assert_eq!(streamed.tuples.len(), count);
    }

    // The same zip written in the language: the traversal fails at
    // open, and the producer's stream after two tuples.
    let src = "input cycle: u64\n\
               sweep := for (k, c) in (1..5, 5..7)\n";
    let mut kernel = polydat::dsl::compile_polydat_interpreter(src).unwrap();
    kernel.set_inputs(&[0]);
    let sweep = kernel.pull_ref("sweep").clone();
    let mut stream = sweep.as_streamer().unwrap().coordinate_stream().unwrap();
    assert!(stream.advance().unwrap().is_some());
    assert!(stream.advance().unwrap().is_some());
    assert_eq!(
        stream.advance().unwrap_err().to_string(),
        "zip strict: child lengths differ ([4, 2])"
    );
    let src = "input cycle: u64\n\
               for (k, c) in (1..5, 5..7) {\n    \
               s := u64_add(k, c)\n}\n";
    let mut kernel = polydat::dsl::compile_polydat_interpreter(src).unwrap();
    kernel.set_inputs(&[0]);
    let error = kernel
        .traverse(0)
        .err()
        .expect("the traversal fails at open");
    assert!(
        error
            .to_string()
            .contains("zip strict: child lengths differ ([4, 2])"),
        "{error}"
    );
}

/// A small PRNG, so the generated shapes are the same on every run.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }

    fn coin(&mut self, percent: u64) -> bool {
        self.below(100) < percent
    }
}

/// Seeded comprehension shapes over every constructor, with each
/// tuple count bounded so the reference evaluator stays quick.
struct Shapes {
    rng: Rng,
    names: u64,
    /// Cycle zips given an operand that keeps nothing.
    emptied: u64,
}

impl Shapes {
    fn name(&mut self) -> String {
        self.names += 1;
        format!("n{}", self.names)
    }

    fn clause(&mut self, bound: &[String]) -> Comprehension {
        let name = self.name();
        let source = match self.rng.below(5) {
            0 => Source::Literal {
                values: (0..self.rng.below(5))
                    .map(|_| LiteralValue::Int(self.rng.below(6) as i64))
                    .collect(),
            },
            1 => Source::Literal {
                values: (0..1 + self.rng.below(3))
                    .map(|i| LiteralValue::String(format!("s{i}")))
                    .collect(),
            },
            2 if !bound.is_empty() => {
                // A source over an earlier axis: the product depends
                // on the tuple before it.
                let over = &bound[self.rng.below(bound.len() as u64) as usize];
                Source::Generator {
                    expr: format!("0..{{{over}}}"),
                    cardinality_hint: None,
                }
            }
            2 => Source::Generator {
                expr: format!("{}..{}", self.rng.below(3), 2 + self.rng.below(4)),
                cardinality_hint: None,
            },
            _ => {
                let lo = self.rng.below(4) as i64 - 1;
                Source::IntRange {
                    lo,
                    hi: lo + self.rng.below(6) as i64,
                    step: 1 + self.rng.below(2) as i64,
                }
            }
        };
        Comprehension::clause(name, source)
    }

    fn shape(&mut self, depth: u32, bound: &[String]) -> Comprehension {
        if depth == 0 {
            return self.clause(bound);
        }
        match self.rng.below(10) {
            0 | 1 => self.clause(bound),
            2 | 3 => {
                let mut children = Vec::new();
                let mut seen = bound.to_vec();
                for _ in 0..2 + self.rng.below(2) {
                    let child = self.shape(depth - 1, &seen);
                    seen.extend(child.coordinate_names());
                    children.push(child);
                }
                Comprehension::cartesian(children)
            }
            4 => {
                let mode = match self.rng.below(3) {
                    0 => ZipMode::Strict,
                    1 => ZipMode::Truncate,
                    _ => ZipMode::Cycle,
                };
                let mut children: Vec<Comprehension> = (0..2 + self.rng.below(2))
                    .map(|_| self.shape(depth - 1, bound))
                    .collect();
                // A cycle zip often gets an operand empty at open, in
                // any position, besides the empty sources and filters
                // the operands draw on their own.
                if mode == ZipMode::Cycle && self.rng.coin(30) {
                    let at = self.rng.below(children.len() as u64) as usize;
                    let child = children.remove(at);
                    children.insert(at, Comprehension::filter(child, "false"));
                    self.emptied += 1;
                }
                Comprehension::zip(children, mode)
            }
            5 => {
                let template = self.shape(depth - 1, bound);
                let other = if self.rng.coin(50) {
                    template.clone()
                } else {
                    self.shape(depth - 1, bound)
                };
                Comprehension::union(vec![template, other])
            }
            6 | 7 => {
                let child = self.shape(depth - 1, bound);
                let names = child.coordinate_names();
                let predicate = match (names.first(), self.rng.below(4)) {
                    (Some(n), 0) => format!("{{{n}}} > {}", self.rng.below(3)),
                    (Some(n), 1) => format!("{{{n}}} != {}", self.rng.below(3)),
                    (Some(n), 2) if names.len() > 1 => format!("{{{n}}} <= {{{}}}", names[1]),
                    _ => if self.rng.coin(50) { "true" } else { "false" }.to_string(),
                };
                Comprehension::filter(child, predicate)
            }
            _ => {
                let child = self.shape(depth - 1, bound);
                let strategy = STRATEGIES[self.rng.below(STRATEGIES.len() as u64) as usize];
                let truncation = self.rng.coin(60).then(|| 1 + self.rng.below(8));
                let seed = self.rng.coin(30).then(|| self.rng.below(1000));
                Comprehension::order_seeded(child, strategy, truncation, seed)
            }
        }
    }
}

/// An upper bound on a shape's tuple count, a dependent source
/// counted at the most it can yield here.
fn bound(c: &Comprehension) -> u64 {
    match c {
        Comprehension::Clause { source, .. } => match source {
            Source::Generator { .. } => 6,
            other => match other.cardinality() {
                polydat::iteration::comprehension::CardinalityClass::Bounded(n) => n,
                _ => 6,
            },
        },
        Comprehension::Cartesian { children } => children
            .iter()
            .map(bound)
            .fold(1u64, |a, b| a.saturating_mul(b)),
        Comprehension::Zip { children, .. } => children.iter().map(bound).max().unwrap_or(1),
        Comprehension::Union { children } => children.iter().map(bound).sum(),
        Comprehension::Filter { child, .. } => bound(child),
        Comprehension::Order {
            child, truncation, ..
        } => truncation.map_or(bound(child), |t| t.min(bound(child))),
    }
}

/// Seeded compositions of every constructor: the two evaluators agree
/// on each one, and the streaming surface agrees with them, failing
/// where a strict zip's operands end apart.
#[test]
fn generated_shapes_index_as_they_materialize() {
    let scope = scope();
    let cases: u64 = std::env::var("POLYDAT_INDEXED_CASES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1500);
    let (mut compared, mut tuples, mut emptied, mut mismatched) = (0u64, 0usize, 0u64, 0u64);
    for case in 0..cases {
        let mut shapes = Shapes {
            rng: Rng(0x5EED_0000 + case),
            names: 0,
            emptied: 0,
        };
        let shape = shapes.shape(3, &[]);
        if bound(&shape) > 4000 {
            continue;
        }
        tuples += assert_equivalent(&shape, &scope);
        compared += 1;
        emptied += shapes.emptied;
        // Strict zips over operands of random lengths often end apart;
        // count the streams that reached such a mismatch.
        if constant_filters(&shape)
            && streamed(&shape)
                .is_some_and(|s| matches!(s.error, Some(RuntimeError::ZipLengthMismatch { .. })))
        {
            mismatched += 1;
        }
    }
    assert!(compared > cases / 2, "only {compared} of {cases} compared");
    assert!(tuples > 0, "no generated shape produced a tuple");
    assert!(
        emptied >= cases / 100,
        "only {emptied} compared cycle zips had an emptied operand"
    );
    assert!(
        mismatched >= cases / 100,
        "only {mismatched} compared streams reached a strict mismatch"
    );
}

/// The open cost of a large product: the reference evaluator builds
/// every tuple before its order selects, the index-addressed one
/// holds the axes and the selection.
#[test]
fn a_large_product_opens_at_the_cost_of_its_selection() {
    let scope = scope();
    let sweep = |n: i64| {
        Comprehension::order(
            Comprehension::cartesian(vec![range("a", 0, n, 1), range("b", 0, n, 1)]),
            StrategyName::Halton,
            Some(100),
        )
    };
    let started = std::time::Instant::now();
    let reference = evaluate_for_iteration_materialized(&sweep(300), &scope).unwrap();
    let reference_time = started.elapsed();
    let started = std::time::Instant::now();
    let indexed = evaluate_indexed(&sweep(300), &scope).unwrap();
    let indexed_time = started.elapsed();
    assert_eq!(indexed.to_vec(), reference.tuples);
    eprintln!(
        "open, 300 x 300 order halton/100: materialized {reference_time:?}, indexed {indexed_time:?}"
    );

    // A product no machine could materialize opens as fast.
    let started = std::time::Instant::now();
    let huge = evaluate_indexed(&sweep(1_000_000), &scope).unwrap();
    let huge_time = started.elapsed();
    assert_eq!(huge.len(), 100);
    assert!(huge.get(99).is_some());
    eprintln!("open, 10^6 x 10^6 order halton/100: indexed {huge_time:?}");
}

/// A traversal over a 10^12-tuple product opens, reports its length,
/// and activates any tuple directly.
#[test]
fn a_traversal_over_a_huge_product_activates_by_position() {
    let src = "input cycle: u64\n\
               for a in 0..1000000, b in 0..1000000 order halton/10 {\n    \
               s := u64_add(a, b)\n}\n";
    let mut kernel = polydat::dsl::compile_polydat_interpreter(src).unwrap();
    kernel.set_inputs(&[0]);
    let started = std::time::Instant::now();
    let mut stream = kernel.traverse(0).unwrap();
    eprintln!(
        "traverse, 10^6 x 10^6 order halton/10: {:?}",
        started.elapsed()
    );
    assert_eq!(stream.len(), 10);
    let mut last = stream.activation(9).unwrap();
    let expected = last.coord("a").unwrap().as_u64() + last.coord("b").unwrap().as_u64();
    assert_eq!(last.cycle(0).pull("s").as_u64(), expected);
    let mut count = 0;
    while stream.advance().unwrap().is_some() {
        count += 1;
    }
    assert_eq!(count, 10);
}
