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
//! every order strategy with and without truncation and seed, filters
//! over comparisons, mixed kinds, negation, membership, and function
//! calls, and sampled continuous spaces. On every shape the validator
//! accepts as written, the optimizer's rewrite validates too and no
//! traversal refuses the shape at its strategy. On every shape the
//! scope-less streaming surface compiles, the rewritten tree the stream
//! compiles ends as the shape as written does, every tuple binds every
//! name of the shape, and the stream yields exactly the traversal's
//! tuples in the same order; where the traversal refuses a shape, a
//! stream that reaches the fault fails with the same kind of error, a
//! strict mismatch after the tuples before it. No rewrite is excused.
//! On every shape a traversal yields, the cardinality its metadata
//! reports holds: an exact count is the count yielded, and an at-most
//! count is never exceeded.

use polydat::iteration::comprehension::ast::Comprehension;
use polydat::iteration::comprehension::cardinality::{Interval, ProductMeasure};
use polydat::iteration::comprehension::runtime::{
    RuntimeError, evaluate_for_iteration_materialized, evaluate_for_iteration_reported,
    evaluate_indexed,
};
use polydat::iteration::comprehension::source::{LiteralValue, Source};
use polydat::iteration::comprehension::strategies::Tuple;
use polydat::iteration::comprehension::strategy::{StrategyName, ZipMode};
use polydat::iteration::comprehension::surfaces::CompiledComprehension;
use polydat::iteration::comprehension::validate::{
    Mode, Surface, ValidationError, ValidationWarning, check_names, unresolved_names, validate,
};

/// The traversals' scope: an input, a cursor whose extent `all(row)`
/// reads, and the lists composed names read (`{k_{k}_limits}`,
/// `{t_{n}_vals}`), some compositions of which nothing binds.
fn scope() -> polydat::kernel::PolydatKernel {
    polydat::dsl::compile_polydat_interpreter(SCOPE).unwrap()
}

const SCOPE: &str = "input cycle: u64\ncursor row = range(0, 5)\n\
                     const k_values := \"1, 10\"\nconst k_1_limits := \"1, 2\"\n\
                     const k_10_limits := \"10, 20, 30\"\n\
                     const t_0_vals := \"0, 1\"\nconst t_2_vals := \"5\"\n\
                     const t_s1_vals := \"7, 8\"\n";

/// The names [`scope`] has.
fn in_scope(name: &str) -> bool {
    matches!(
        name,
        "cycle"
            | "__cursor_extent_row_start"
            | "__cursor_extent_row_end"
            | "k_values"
            | "k_1_limits"
            | "k_10_limits"
            | "t_0_vals"
            | "t_2_vals"
            | "t_s1_vals"
    )
}

/// Assert the two evaluators agree on `ast`, and the streaming surface
/// with them, returning the tuple count.
fn assert_equivalent(ast: &Comprehension, scope: &polydat::kernel::PolydatKernel) -> usize {
    compare(ast, scope).tuples
}

/// What comparing one shape found.
struct Compared {
    /// The traversal's tuple count; 0 when it fails.
    tuples: usize,
    /// Whether the streaming surface compiled the shape.
    streamed: bool,
}

/// [`assert_equivalent`], reporting what it found.
fn compare(ast: &Comprehension, scope: &polydat::kernel::PolydatKernel) -> Compared {
    let reference = evaluate_for_iteration_materialized(ast, scope);
    let indexed = evaluate_indexed(ast, scope);
    let outcome = match (reference, indexed) {
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
            assert_counts(ast, reference.tuples.len());
            Ok(reference.tuples)
        }
        (Err(reference), Err(indexed)) => {
            assert_eq!(
                indexed.to_string(),
                reference.to_string(),
                "errors differ for {ast:?}"
            );
            Err(reference)
        }
        (reference, indexed) => panic!(
            "one evaluator failed for {ast:?}:\n  reference: {:?}\n  indexed: {:?}",
            reference.map(|r| r.tuples.len()),
            indexed.map(|t| t.len())
        ),
    };
    let tuples = outcome.as_ref().map_or(0, Vec::len);
    // Acceptance is decided on the tree as written (comprehension_forms.md
    // §5): a tree the validator accepts still validates once the
    // optimizer has rewritten it, and no traversal of it refuses it at
    // its strategy (V4).
    let flat = polydat::iteration::comprehension::flatten::flatten_static_sources(
        ast,
        &polydat::kernel::interp::NoScope::new(),
    );
    // V3 (§5): a stream supplies no name and a traversal the names of
    // its scope. A shape that reads a name the stream does not bind is
    // refused by a strict stream compile, and a permissive one that
    // compiles it warns and reads the name as None, as the traversal
    // does a name its scope does not have.
    let unresolved = unresolved_names(&flat, Surface::Stream);
    if !unresolved.is_empty() {
        assert!(
            matches!(
                CompiledComprehension::from_ast_with(ast, Mode::Strict),
                Err(ValidationError::V3UnresolvedNames { .. })
            ),
            "the strict stream compiles {ast:?}, which reads a name it does not bind"
        );
        if let Ok((_, report)) = CompiledComprehension::from_ast_with(ast, Mode::Permissive) {
            assert!(
                matches!(
                    report.warnings.first(),
                    Some(ValidationWarning::UnresolvedNames { reads }) if *reads == unresolved
                ),
                "the stream compiles {ast:?} without its V3 warning"
            );
        }
    }
    // The stream is held to the traversal in its own scope, which has no
    // name: where the shape reads a name the traversal's scope has, the
    // stream reads it as None.
    let no_scope = polydat::kernel::interp::NoScope::new();
    let reads_scope = unresolved.iter().any(|r| !r.bare && in_scope(&r.name));
    let stream_scope: &dyn polydat::kernel::interp::Lookup =
        if reads_scope { &no_scope } else { scope };
    let stream_outcome = if reads_scope {
        evaluate_indexed(ast, &no_scope).map(|t| t.to_vec())
    } else {
        outcome.clone()
    };
    if validate(&flat, Mode::Permissive).is_ok() {
        let rewritten = polydat::iteration::comprehension::optimize::optimize(flat.clone());
        if let Err(e) = validate(&rewritten, Mode::Permissive) {
            panic!("the validator accepts {ast:?}, and refuses its rewrite {rewritten:?}: {e}");
        }
        assert!(
            !matches!(outcome, Err(RuntimeError::StrategyRejectsInput { .. })),
            "the validator accepts {ast:?}, which the traversal refuses at its strategy: {:?}",
            outcome.as_ref().err()
        );
    }
    let Some(streamed) = streamed(ast) else {
        return Compared {
            tuples,
            streamed: false,
        };
    };
    // The streaming surface compiles the optimized tree (§10.6), whose
    // traversal ends as the traversal of the shape as written does, and
    // the stream is held to it.
    let optimized_ast = polydat::iteration::comprehension::optimize::optimize(ast.clone());
    let optimized = evaluate_indexed(&optimized_ast, stream_scope).map(|t| t.to_vec());
    assert!(
        same_outcome(&stream_outcome, &optimized),
        "the optimizer's rewrite changes the outcome of {ast:?}:\n  as written: {:?}\n  \
         optimized: {:?}",
        stream_outcome.as_ref().map(Vec::len),
        optimized.as_ref().map(Vec::len)
    );
    match &stream_outcome {
        Ok(expected) => {
            assert_counts(&optimized_ast, expected.len());
            // On a shape that validates, which the streaming surface
            // compiles, every tuple binds every name.
            let names = shape_names(ast);
            for tuple in expected {
                assert_eq!(
                    tuple_names(tuple.iter().map(|(n, _)| n)),
                    names,
                    "a tuple does not bind every name of {ast:?}"
                );
            }
            assert!(
                streamed.error.is_none(),
                "the streaming surface failed where the traversal did not, for {ast:?}: {:?}",
                streamed.error
            );
            assert_eq!(
                streamed.tuples.iter().map(stream_row).collect::<Vec<_>>(),
                expected
                    .iter()
                    .map(|t| traversal_row(t))
                    .collect::<Vec<_>>(),
                "the streaming surface's tuples differ for {ast:?}"
            );
        }
        Err(expected) => {
            // A stream fails where it reaches a fault: the traversal's,
            // a strict mismatch after the tuples before it, or another
            // fault of the shape that it pulls first, which the
            // traversal raises once every strict zip truncates.
            if let Some(error) = &streamed.error {
                let kind = std::mem::discriminant(error);
                if kind == std::mem::discriminant(expected) {
                    if matches!(expected, RuntimeError::ZipLengthMismatch { .. }) {
                        assert_stream_failure(ast, &streamed, expected);
                    }
                } else {
                    let another_fault = match error {
                        RuntimeError::ZipLengthMismatch { .. } => strict_zips(ast) > 0,
                        _ => evaluate_indexed(&truncated(ast), stream_scope)
                            .err()
                            .is_some_and(|e| std::mem::discriminant(&e) == kind),
                    };
                    assert!(
                        another_fault,
                        "the stream failed with a fault the shape does not have, for {ast:?}: \
                         {error}, where the traversal failed with {expected}"
                    );
                }
            }
        }
    }
    Compared {
        tuples,
        streamed: true,
    }
}

/// Assert the cardinality `ast`'s metadata reports holds for the
/// `yielded` tuples: an exact count is the count, and an at-most count
/// is never exceeded.
fn assert_counts(ast: &Comprehension, yielded: usize) {
    use polydat::iteration::comprehension::CardinalityClass;
    match ast.metadata().cardinality {
        CardinalityClass::Bounded(n) => assert_eq!(
            yielded as u64, n,
            "the metadata reports exactly {n} tuples for {ast:?}"
        ),
        CardinalityClass::BoundedAtMost(n) => assert!(
            yielded as u64 <= n,
            "the metadata reports at most {n} tuples for {ast:?}, which yields {yielded}"
        ),
        _ => {}
    }
}

/// Whether two traversals end alike: the same tuples, or the same kind
/// of error.
fn same_outcome(
    a: &Result<Vec<polydat::iteration::comprehension::runtime::RuntimeTuple>, RuntimeError>,
    b: &Result<Vec<polydat::iteration::comprehension::runtime::RuntimeTuple>, RuntimeError>,
) -> bool {
    match (a, b) {
        (Ok(a), Ok(b)) => a == b,
        (Err(a), Err(b)) => std::mem::discriminant(a) == std::mem::discriminant(b),
        _ => false,
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

/// A value as both surfaces carry it: the traversal's `U64` and the
/// stream's `I64` are one integer, a float compares by its bits, and
/// JSON by its text.
#[derive(Debug, PartialEq)]
enum Cell {
    Int(u64),
    Float(u64),
    Str(String),
    Bool(bool),
}

fn traversal_row(tuple: &[(String, polydat::ast::Value)]) -> Vec<(String, Cell)> {
    use polydat::ast::Value;
    tuple
        .iter()
        .map(|(name, value)| {
            let cell = match value {
                Value::U64(n) => Cell::Int(*n),
                Value::I64(n) => Cell::Int(*n as u64),
                Value::F64(f) => Cell::Float(f.to_bits()),
                Value::Bool(b) => Cell::Bool(*b),
                Value::Str(s) => Cell::Str(s.to_string()),
                Value::Json(j) => Cell::Str(j.to_string()),
                other => Cell::Str(other.to_display_string()),
            };
            (name.clone(), cell)
        })
        .collect()
}

fn stream_row(tuple: &Tuple) -> Vec<(String, Cell)> {
    use polydat::iteration::comprehension::strategies::TupleValue;
    tuple
        .bindings
        .iter()
        .map(|(name, value)| {
            let cell = match value {
                TupleValue::U64(n) => Cell::Int(*n),
                TupleValue::I64(n) => Cell::Int(*n as u64),
                TupleValue::F64(f) => Cell::Float(f.to_bits()),
                TupleValue::Bool(b) => Cell::Bool(*b),
                TupleValue::Str(s) => Cell::Str(s.clone()),
            };
            (name.clone(), cell)
        })
        .collect()
}

fn ints(name: &str, vs: &[i64]) -> Comprehension {
    Comprehension::clause(
        name,
        Source::Literal {
            values: vs.iter().map(|v| LiteralValue::Int(*v)).collect(),
        },
    )
}

fn words(name: &str, vs: &[&str]) -> Comprehension {
    Comprehension::clause(
        name,
        Source::Literal {
            values: vs
                .iter()
                .map(|v| LiteralValue::String((*v).to_string()))
                .collect(),
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

/// A clause over the list a `{name}` placeholder reads, `name` composed
/// or not.
fn param(name: &str, list: &str) -> Comprehension {
    Comprehension::clause(
        name,
        Source::WorkloadParamList {
            name: list.into(),
            len_hint: None,
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
        // Mixed kinds: a string is never equal to a number, and
        // ordering the two fails on every path.
        Comprehension::filter(
            Comprehension::cartesian(vec![words("w", &["s0", "s1", "s2"]), ints("n", &[1, 2])]),
            "{w} != 2 && {n} != \"s1\"",
        ),
        Comprehension::filter(words("w", &["s0", "s1"]), "{w} in [\"s1\", 2, 3.5]"),
        // A bare word is a name, which fails to resolve on every path.
        Comprehension::filter(words("w", &["s0", "s1"]), "{w} == s1"),
        Comprehension::filter(words("w", &["s0", "s1"]), "{w} > 2"),
        // Negation binds to its operand.
        Comprehension::filter(product(), "!({a} == 1) || {b} == 5"),
        Comprehension::filter(product(), "!({a} < 2) && !({b} > 6) || {a} == 0"),
        // Function calls, over numbers and over strings.
        Comprehension::filter(product(), "u64_mul({a}, 2) + 1 > {b}"),
        Comprehension::filter(product(), "u64_mod(u64_add({a}, {b}), 3) == 0 || {a} == 3"),
        Comprehension::filter(words("w", &["s0", "s1"]), "u64_add({w}, 1) > 1"),
        // Orders over continuous spaces, with every sampling strategy,
        // alone and inside other combinators.
        Comprehension::cartesian(vec![
            ints("p", &[1, 2]),
            Comprehension::order(unit("u"), StrategyName::Lhs, Some(3)),
        ]),
        Comprehension::zip(
            vec![
                Comprehension::order_seeded(unit("u"), StrategyName::Shuffle, Some(4), Some(9)),
                range("k", 0, 6, 1),
            ],
            ZipMode::Truncate,
        ),
        Comprehension::union(vec![
            Comprehension::order(unit("u"), StrategyName::Halton, Some(2)),
            Comprehension::order(unit("u"), StrategyName::Sobol, Some(3)),
        ]),
        // `all(<cursor>)` reads the cursor's extent (§10.9.1), and over
        // a cursor nothing declares reads None.
        generator("xval", "all(row)"),
        generator("xval", "all(zz)"),
        // A composed name reads what it composes to with the earlier
        // axis bound, braced and bare-prior; a composition nothing binds
        // yields nothing for its tuple (§5 V3).
        Comprehension::cartesian(vec![param("k", "k_values"), param("limit", "k_{k}_limits")]),
        Comprehension::cartesian(vec![
            generator("k", "k_values"),
            param("limit", "k_{k}_limits"),
        ]),
        Comprehension::cartesian(vec![ints("n", &[0, 1, 2]), param("v", "t_{n}_vals")]),
        Comprehension::filter(
            Comprehension::cartesian(vec![
                generator("xval", "all(row)"),
                param("limit", "k_{xval}_limits"),
            ]),
            "{limit} != 2",
        ),
    ];
    for strategy in [
        StrategyName::Halton,
        StrategyName::Sobol,
        StrategyName::Lhs,
        StrategyName::Shuffle,
        StrategyName::Extrema,
    ] {
        for truncation in [Some(1), Some(5)] {
            shapes.push(Comprehension::order(
                Comprehension::cartesian(vec![ints("p", &[1, 2, 3]), unit("u")]),
                strategy,
                truncation,
            ));
            shapes.push(Comprehension::order(
                Comprehension::filter(
                    Comprehension::cartesian(vec![unit("u"), unit("v")]),
                    "{u} + {v} > 1.0",
                ),
                strategy,
                truncation,
            ));
        }
    }
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
                    // An untruncated order under the order: a strategy
                    // that selects from the shape reads through it, and
                    // any other runs after it (§7.4 O1).
                    Comprehension::order(product(), StrategyName::Shuffle, None),
                    Comprehension::order_seeded(product(), StrategyName::Lhs, None, Some(3)),
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
    // The cursor and composed-name forms yield what they read.
    for (shape, count) in [
        (generator("xval", "all(row)"), 5),
        (generator("xval", "all(zz)"), 0),
        (
            Comprehension::cartesian(vec![param("k", "k_values"), param("limit", "k_{k}_limits")]),
            5,
        ),
        (
            Comprehension::cartesian(vec![
                generator("k", "k_values"),
                param("limit", "k_{k}_limits"),
            ]),
            5,
        ),
        (
            Comprehension::cartesian(vec![ints("n", &[0, 1, 2]), param("v", "t_{n}_vals")]),
            3,
        ),
    ] {
        assert_eq!(assert_equivalent(&shape, &scope), count, "{shape:?}");
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
        // shape; the validator refuses a non-Lex order over a zip with
        // an operand that has no index function (V4), which the
        // traversal, reading the zip's evaluated shape, accepts.
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
    /// Orders over a continuous axis.
    sampled: u64,
    /// Filters whose predicate calls a function.
    called: u64,
    /// Filters whose predicate reads a name its input does not bind.
    outer: u64,
    /// Sources over a cursor's extent or a composed name.
    composed: u64,
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
            2 if self.rng.coin(20) => {
                // A source over a name nothing binds, which reads None
                // and yields nothing (§5 V3).
                self.outer += 1;
                Source::Generator {
                    expr: "0..{zz}".into(),
                    cardinality_hint: None,
                }
            }
            2 if self.rng.coin(15) => {
                // A cursor's extent (§10.9.1).
                self.composed += 1;
                Source::Generator {
                    expr: "all(row)".into(),
                    cardinality_hint: None,
                }
            }
            2 if !bound.is_empty() && self.rng.coin(30) => {
                // A name composed over an earlier axis: the scope binds
                // some compositions and not others, which read None and
                // yield nothing (§5 V3).
                self.composed += 1;
                let over = &bound[self.rng.below(bound.len() as u64) as usize];
                Source::WorkloadParamList {
                    name: format!("t_{{{over}}}_vals"),
                    len_hint: None,
                }
            }
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
        match self.rng.below(11) {
            0 | 1 => self.clause(bound),
            10 => {
                // A sampled space: a continuous axis, perhaps beside a
                // discrete clause and under a filter on the sample.
                let u = self.name();
                let mut axes = vec![unit(&u)];
                if self.rng.coin(60) {
                    let at = self.rng.below(2) as usize;
                    axes.insert(at, self.clause(&[]));
                }
                let mut space = if axes.len() == 1 {
                    axes.remove(0)
                } else {
                    Comprehension::cartesian(axes)
                };
                if self.rng.coin(40) {
                    space = Comprehension::filter(
                        space,
                        format!("{{{u}}} > 0.{}", 1 + self.rng.below(8)),
                    );
                }
                let strategy = [
                    StrategyName::Halton,
                    StrategyName::Sobol,
                    StrategyName::Lhs,
                    StrategyName::Shuffle,
                    StrategyName::Extrema,
                ][self.rng.below(5) as usize];
                let seed = self.rng.coin(30).then(|| self.rng.below(1000));
                self.sampled += 1;
                Comprehension::order_seeded(space, strategy, Some(1 + self.rng.below(8)), seed)
            }
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
                let predicate = match (names.first(), self.rng.below(8)) {
                    (Some(n), 0) => format!("{{{n}}} > {}", self.rng.below(3)),
                    (Some(n), 1) => format!("{{{n}}} != {}", self.rng.below(3)),
                    (Some(n), 2) if names.len() > 1 => format!("{{{n}}} <= {{{}}}", names[1]),
                    (Some(n), 3) => {
                        self.called += 1;
                        format!("u64_mod(u64_add({{{n}}}, 1), 3) != 0")
                    }
                    (Some(n), 4) => {
                        format!("!({{{n}}} == {}) || {{{n}}} == \"s0\"", self.rng.below(3))
                    }
                    (Some(n), 5) => format!("{{{n}}} in [0, 2, \"s1\"]"),
                    (Some(n), 6) if names.len() > 1 => {
                        format!("{{{n}}} != \"s1\" && {{{}}} != 2", names[1])
                    }
                    // A name of the traversal's scope, or one nothing
                    // binds (§5 V3).
                    (Some(n), 7) if self.rng.coin(40) => {
                        self.outer += 1;
                        let outer = if self.rng.coin(50) { "cycle" } else { "zz" };
                        format!("{{{n}}} != {{{outer}}}")
                    }
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
/// on each one, and the streaming surface yields the same tuples,
/// failing where a strict zip's operands end apart.
#[test]
fn generated_shapes_index_as_they_materialize() {
    let scope = scope();
    let cases: u64 = std::env::var("POLYDAT_INDEXED_CASES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1500);
    let (mut compared, mut tuples, mut emptied, mut mismatched) = (0u64, 0usize, 0u64, 0u64);
    let (mut streamed_tuples, mut sampled, mut called) = (0usize, 0u64, 0u64);
    let (mut outer, mut chained) = (0u64, 0u64);
    let (mut composed, mut composed_none) = (0u64, 0u64);
    for case in 0..cases {
        let mut shapes = Shapes {
            rng: Rng(0x5EED_0000 + case),
            names: 0,
            emptied: 0,
            sampled: 0,
            called: 0,
            outer: 0,
            composed: 0,
        };
        let shape = shapes.shape(3, &[]);
        if bound(&shape) > 4000 {
            continue;
        }
        let found = compare(&shape, &scope);
        tuples += found.tuples;
        compared += 1;
        emptied += shapes.emptied;
        sampled += shapes.sampled;
        called += shapes.called;
        outer += shapes.outer;
        composed += shapes.composed;
        if shapes.composed > 0
            && let Ok(reported) = evaluate_for_iteration_reported(&shape, &scope)
        {
            composed_none += reported
                .clauses
                .iter()
                .filter(|c| !c.reads_none.is_empty())
                .count() as u64;
        }
        if found.tuples > 0 && order_chains(&shape) > 0 {
            chained += 1;
        }
        if found.streamed {
            streamed_tuples += found.tuples;
            // Strict zips over operands of random lengths often end
            // apart; count the streams that reached such a mismatch.
            if streamed(&shape)
                .is_some_and(|s| matches!(s.error, Some(RuntimeError::ZipLengthMismatch { .. })))
            {
                mismatched += 1;
            }
        }
    }
    eprintln!(
        "{compared} shapes compared, {tuples} tuples, {streamed_tuples} through streams, \
         {chained} yielding through order chains, {outer} predicates and sources reading outer \
         names, {composed} sources over a cursor or a composed name, {composed_none} clauses \
         reading None"
    );
    assert!(
        composed >= cases / 20,
        "only {composed} compared sources read a cursor or a composed name"
    );
    assert!(
        composed_none >= cases / 100,
        "only {composed_none} compared clauses reported a read of None"
    );
    assert!(compared > cases / 2, "only {compared} of {cases} compared");
    assert!(tuples > 0, "no generated shape produced a tuple");
    assert!(
        streamed_tuples > tuples / 4,
        "streams yielded only {streamed_tuples} of {tuples} compared tuples"
    );
    assert!(
        emptied >= cases / 100,
        "only {emptied} compared cycle zips had an emptied operand"
    );
    assert!(
        mismatched >= cases / 100,
        "only {mismatched} compared streams reached a strict mismatch"
    );
    assert!(
        sampled >= cases / 20,
        "only {sampled} compared shapes sampled a continuous space"
    );
    assert!(
        called >= cases / 50,
        "only {called} compared filters called a function"
    );
    assert!(
        outer >= cases / 100,
        "only {outer} compared filters read a name their input does not bind"
    );
    assert!(
        chained >= cases / 50,
        "only {chained} compared shapes yielded through an order over an order"
    );
}

/// How many orders in `c` order another order's output, directly or
/// through one filter.
fn order_chains(c: &Comprehension) -> usize {
    let own = match c {
        Comprehension::Order { child, .. } => match child.as_ref() {
            Comprehension::Order { .. } => 1,
            Comprehension::Filter { child, .. } => {
                usize::from(matches!(child.as_ref(), Comprehension::Order { .. }))
            }
            _ => 0,
        },
        _ => 0,
    };
    own + c.children().map(order_chains).sum::<usize>()
}

/// A predicate groups by the language's one precedence table on every
/// path: unary `!` binds tighter than `&&`, `&&` tighter than `||`,
/// comparison tighter than `&&`, and arithmetic tighter than
/// comparison. Each predicate keeps exactly the tuples its grouping
/// written out keeps.
#[test]
fn predicates_group_by_the_one_precedence_table() {
    let scope = scope();
    let flags = |name: &str| {
        Comprehension::clause(
            name,
            Source::Literal {
                values: vec![LiteralValue::Bool(true), LiteralValue::Bool(false)],
            },
        )
    };
    let space = || {
        Comprehension::cartesian(vec![
            flags("a"),
            flags("b"),
            flags("c"),
            range("x", 0, 3, 1),
            range("y", 0, 3, 1),
        ])
    };
    let traversal = |predicate: &str| {
        evaluate_indexed(&Comprehension::filter(space(), predicate), &scope)
            .unwrap()
            .to_vec()
    };
    let stream = |predicate: &str| {
        let s = streamed(&Comprehension::filter(space(), predicate)).unwrap();
        assert!(s.error.is_none(), "{predicate}: {:?}", s.error);
        s.tuples
    };
    let pairs = [
        ("!true || {x} == 1", "(!true) || {x} == 1"),
        ("!false && {x} == 1", "(!false) && {x} == 1"),
        ("{x} == 1 || !true", "{x} == 1 || (!true)"),
        (
            "{x} == 1 || {y} == 2 && {x} > 0",
            "{x} == 1 || ({y} == 2 && {x} > 0)",
        ),
        (
            "{x} == 1 && {y} == 2 || {x} == 0",
            "({x} == 1 && {y} == 2) || {x} == 0",
        ),
        ("{x} < 2 && {y} >= 1", "({x} < 2) && ({y} >= 1)"),
        ("!{a} || {b}", "(!{a}) || {b}"),
        ("!{a} && {b}", "(!{a}) && {b}"),
        ("{a} || {b} && {c}", "{a} || ({b} && {c})"),
        ("{a} && {b} || {c}", "({a} && {b}) || {c}"),
        (
            "{x} == 1 || {y} == 2 && {a}",
            "({x} == 1) || (({y} == 2) && {a})",
        ),
        ("{x} < {y} == {a}", "({x} < {y}) == {a}"),
        ("!{a} == {b}", "(!{a}) == {b}"),
        ("{x} + 1 > {y} * 2", "({x} + 1) > ({y} * 2)"),
        (
            "{x} * 2 == {y} + 1 || {c}",
            "(({x} * 2) == ({y} + 1)) || {c}",
        ),
        ("{x} in [0, 2] || {a}", "({x} in [0, 2]) || {a}"),
    ];
    for (bare, grouped) in pairs {
        let kept = traversal(bare);
        assert_eq!(kept, traversal(grouped), "{bare}");
        let streamed: Vec<_> = stream(bare).iter().map(stream_row).collect();
        assert_eq!(
            streamed,
            kept.iter().map(|t| traversal_row(t)).collect::<Vec<_>>(),
            "{bare}"
        );
        assert_eq!(stream(grouped), stream(bare), "{bare}");
    }
    // The grouping `!` over the whole disjunction keeps other tuples.
    assert_ne!(traversal("!{a} || {b}"), traversal("!({a} || {b})"));
    assert_ne!(stream("!{a} || {b}"), stream("!({a} || {b})"));
}

/// Acceptance is decided on the tree as written (comprehension_forms.md
/// §5), before any rewrite, so it is the same for the stream, which
/// compiles the rewritten tree, and the traversal, which evaluates the
/// tree as written: a child the optimizer unwraps is judged as it is
/// judged wrapped, and a tree the validator accepts is accepted at every
/// strategy.
#[test]
fn acceptance_is_decided_on_the_tree_as_written() {
    let scope = scope();
    let refused = |c: &Comprehension| {
        assert!(validate(c, Mode::Permissive).is_err(), "{c:?}");
        assert!(streamed(c).is_none(), "{c:?}");
    };
    // V6: a strict zip over an operand of unknown count, bare or under
    // an untruncated `Lex` order that R0a drops.
    let dependent = || generator("d", "0..{a}");
    for operand in [
        dependent(),
        Comprehension::order(dependent(), StrategyName::Lex, None),
    ] {
        refused(&Comprehension::cartesian(vec![
            range("a", 1, 3, 1),
            Comprehension::zip(vec![operand, ints("z", &[1, 2])], ZipMode::Strict),
        ]));
    }
    // A truncated `Lex` order selects a prefix of its input's positions,
    // which another strategy ranks as one axis, directly or through one
    // filter (§3.6, §5 V5); over a filter it counts tuples as they stream
    // and has no positions, and a strategy over it is refused (V4).
    let prefix = || Comprehension::order(range("a", 0, 5, 1), StrategyName::Lex, Some(3));
    for input in [prefix(), Comprehension::filter(prefix(), "{a} > 0")] {
        let shape = Comprehension::order(input, StrategyName::ReverseLex, None);
        validate(&shape, Mode::Permissive).unwrap();
        assert!(streamed(&shape).is_some(), "{shape:?}");
        assert!(assert_equivalent(&shape, &scope) > 0);
    }
    let rows = |c: &Comprehension| evaluate_indexed(c, &scope).unwrap().to_vec();
    let reversed = Comprehension::order(prefix(), StrategyName::ReverseLex, None);
    let mut expected = rows(&prefix());
    expected.reverse();
    assert_eq!(rows(&reversed), expected);
    let streaming = Comprehension::order(
        Comprehension::order(
            Comprehension::filter(range("a", 0, 5, 1), "{a} > 0"),
            StrategyName::Lex,
            Some(3),
        ),
        StrategyName::ReverseLex,
        None,
    );
    refused(&streaming);
    assert!(matches!(
        evaluate_indexed(&streaming, &scope),
        Err(RuntimeError::StrategyRejectsInput { .. })
    ));
    // An untruncated `Lex` order passes its input's positions through,
    // so a strategy over it is accepted on every path.
    let through = Comprehension::order(
        Comprehension::order(range("a", 0, 5, 1), StrategyName::Lex, None),
        StrategyName::ReverseLex,
        Some(2),
    );
    validate(&through, Mode::Permissive).unwrap();
    assert_eq!(assert_equivalent(&through, &scope), 2);
    // A filter that is always true is accepted under a strategy as
    // written and once R0a drops it, and orders alike.
    let trivially = Comprehension::order(
        Comprehension::filter(range("a", 0, 5, 1), "true"),
        StrategyName::Halton,
        Some(2),
    );
    validate(&trivially, Mode::Permissive).unwrap();
    assert_eq!(
        evaluate_indexed(&trivially, &scope).unwrap().to_vec(),
        evaluate_indexed(
            &Comprehension::order(range("a", 0, 5, 1), StrategyName::Halton, Some(2)),
            &scope
        )
        .unwrap()
        .to_vec()
    );
    assert_eq!(assert_equivalent(&trivially, &scope), 2);
}

/// A predicate the totality check calls total (comprehension_forms.md
/// §10.2 R5) evaluates over every tuple of its elements' kinds without
/// error, arithmetic included, on the traversal and the stream.
#[test]
fn a_total_predicate_never_fails() {
    use polydat::iteration::comprehension::predicate::{CompiledPredicate, element_kind};
    let scope = scope();
    let space = || {
        Comprehension::cartesian(vec![
            ints("k", &[0, 1, 7, -1]),
            ints("m", &[0, 3]),
            Comprehension::clause(
                "x",
                Source::Literal {
                    values: vec![
                        LiteralValue::Float(0.0),
                        LiteralValue::Float(0.5),
                        LiteralValue::Float(3.25),
                    ],
                },
            ),
            words("w", &["a", "zz", ""]),
            Comprehension::clause(
                "b",
                Source::Literal {
                    values: vec![LiteralValue::Bool(false), LiteralValue::Bool(true)],
                },
            ),
        ])
    };
    for p in [
        "{k} > 1",
        "{k} < {x}",
        "{w} == 2",
        "{w} != {k} && {b}",
        "{w} >= \"m\" || !{b}",
        "{k} in [1, \"a\", true]",
        "{k} * 2 + 1 > {m}",
        "{k} - {m} >= 0",
        "{k} / 2 == 1",
        "{x} % 1.5 < 1",
        "{k} ** 2 > {x}",
        "{k} + 1",
        "{x}",
    ] {
        let shape = Comprehension::filter(space(), p);
        let total = CompiledPredicate::new(p).is_total(&|n| element_kind(&space(), n));
        assert!(total, "{p}");
        evaluate_indexed(&shape, &scope).unwrap_or_else(|e| panic!("{p}: {e}"));
        let s = streamed(&shape).expect("the streaming surface compiles");
        assert!(s.error.is_none(), "{p}: {:?}", s.error);
        assert_equivalent(&shape, &scope);
    }
}

/// A non-`Lex` order over a filter (comprehension_forms.md §5 V5): the
/// strategy ranks only the survivors, by their positions in the
/// filter's input, and keeps its truncation's worth of them. `extrema/1`
/// is the most extreme stratum any survivor is in, and `halton/n` is `n`
/// survivors when at least `n` exist. The traversal, the stream, and a
/// `for` statement agree, and the metadata counts at most the
/// truncation.
#[test]
fn an_order_over_a_filter_ranks_the_survivors() {
    use polydat::iteration::comprehension::CardinalityClass;
    let scope = scope();
    let grid = || Comprehension::cartesian(vec![range("a", 0, 3, 1), range("b", 0, 3, 1)]);
    let pairs = |c: &Comprehension| -> Vec<(u64, u64)> {
        evaluate_indexed(c, &scope)
            .unwrap()
            .iter()
            .map(|t| (t[0].1.as_u64(), t[1].1.as_u64()))
            .collect()
    };
    // No corner survives: the edges are the most extreme stratum left.
    let edges = Comprehension::order(
        Comprehension::filter(grid(), "{a} == 1 || {b} == 1"),
        StrategyName::Extrema,
        Some(1),
    );
    assert_eq!(pairs(&edges), vec![(0, 1), (1, 0), (1, 2), (2, 1)]);
    // One corner filtered out: the other three.
    let corners = Comprehension::order(
        Comprehension::filter(grid(), "{a} != 0 || {b} != 0"),
        StrategyName::Extrema,
        Some(1),
    );
    assert_eq!(pairs(&corners), vec![(0, 2), (2, 0), (2, 2)]);
    // Four of the five survivors, in Halton's order over the grid.
    let sampled = Comprehension::order(
        Comprehension::filter(grid(), "{a} == 1 || {b} == 1"),
        StrategyName::Halton,
        Some(4),
    );
    let kept = pairs(&sampled);
    assert_eq!(kept.len(), 4);
    assert!(kept.iter().all(|(a, b)| *a == 1 || *b == 1), "{kept:?}");
    // Every survivor, whatever the strategy, when none is cut.
    for strategy in STRATEGIES {
        for truncation in [None, Some(2), Some(100)] {
            let shape = Comprehension::order(
                Comprehension::filter(grid(), "{a} == 1 || {b} == 1"),
                strategy,
                truncation,
            );
            let yielded = assert_equivalent(&shape, &scope);
            let expected = match (strategy, truncation) {
                (_, None) | (_, Some(100)) => 5,
                (StrategyName::Extrema, Some(2)) => 5,
                (_, Some(n)) => n as usize,
            };
            assert_eq!(yielded, expected, "{strategy:?} {truncation:?}");
            assert!(
                matches!(
                    shape.metadata().cardinality,
                    CardinalityClass::BoundedAtMost(_)
                ),
                "{strategy:?}"
            );
            assert!(streamed(&shape).is_some(), "{strategy:?}");
        }
    }
    // Every tuple surviving selects what the order selects unfiltered.
    for strategy in STRATEGIES {
        let unfiltered = Comprehension::order(grid(), strategy, Some(3));
        let filtered =
            Comprehension::order(Comprehension::filter(grid(), "{a} < 9"), strategy, Some(3));
        assert_eq!(pairs(&filtered), pairs(&unfiltered), "{strategy:?}");
    }

    // The same order written in the language.
    let text = "a in 0..3, b in 0..3 where {a} == 1 || {b} == 1 order extrema/1";
    let src = format!(
        "input cycle: u64\nsweep := for {text}\nfor {text} {{\n    s := u64_add(a, b)\n}}\n"
    );
    let mut kernel = polydat::dsl::compile_polydat_interpreter(&src).unwrap();
    kernel.set_inputs(&[0]);
    let sweep = kernel.pull_ref("sweep").clone();
    let streamer = sweep.as_streamer().unwrap();
    assert_eq!(streamer.coordinate_stream().unwrap().count(), 4);
    let mut stream = kernel.traverse(0).unwrap();
    assert_eq!(stream.len(), 4);
    let mut sums = Vec::new();
    while let Some(mut activation) = stream.advance().unwrap() {
        sums.push(activation.cycle(0).pull("s").as_u64());
    }
    assert_eq!(sums, vec![1, 1, 3, 3]);
}

/// The worked examples of comprehension_forms.md §11.2 and §11.6: an
/// `extrema/1` over a filter keeps the most extreme stratum any
/// survivor is in, and the same order before the filter keeps the
/// corners the filter then drops.
#[test]
fn the_filter_and_order_examples_of_section_11_yield_what_they_say() {
    let scope = scope();
    let pairs = |c: &Comprehension| -> Vec<(u64, u64)> {
        evaluate_indexed(c, &scope)
            .unwrap()
            .iter()
            .map(|t| (t[0].1.as_u64(), t[1].1.as_u64()))
            .collect()
    };
    let grid =
        |n: i64| Comprehension::cartesian(vec![range("k", 1, n, 1), range("limit", 1, n, 1)]);
    // §11.2
    let corners = Comprehension::order(
        Comprehension::filter(grid(100), "{k} * {limit} <= 1000"),
        StrategyName::Extrema,
        Some(1),
    );
    assert_eq!(pairs(&corners), vec![(1, 1), (1, 99), (99, 1)]);
    // §11.6
    let band = "{k} * {limit} > 50 && {k} * {limit} < 80";
    let form_a = Comprehension::filter(
        Comprehension::order(grid(10), StrategyName::Extrema, Some(1)),
        band,
    );
    let form_b = Comprehension::order(
        Comprehension::filter(grid(10), band),
        StrategyName::Extrema,
        Some(1),
    );
    assert_eq!(pairs(&form_a), vec![]);
    assert_eq!(
        pairs(&form_b),
        vec![(6, 9), (7, 9), (8, 9), (9, 6), (9, 7), (9, 8)]
    );
    assert_equivalent(&form_a, &scope);
    assert_equivalent(&form_b, &scope);
}

/// An order over an untruncated order (comprehension_forms.md §7.4
/// O1): a strategy that selects from the shape chooses what it chooses
/// over the shape beneath the inner order, which has no effect, and a
/// strategy that selects from the sequence runs after the inner order.
/// Both evaluators, the stream, and the validator agree.
#[test]
fn an_order_chain_folds_only_under_a_shape_strategy() {
    let scope = scope();
    let product = || Comprehension::cartesian(vec![range("a", 0, 4, 1), ints("b", &[5, 6, 7])]);
    let shuffled = || Comprehension::order_seeded(product(), StrategyName::Shuffle, None, Some(5));
    let rows = |c: &Comprehension| evaluate_indexed(c, &scope).unwrap().to_vec();
    let streamed_rows = |c: &Comprehension| {
        let s = streamed(c).expect("the streaming surface compiles");
        assert!(s.error.is_none(), "{:?}", s.error);
        s.tuples.iter().map(stream_row).collect::<Vec<_>>()
    };
    for outer in [
        StrategyName::Halton,
        StrategyName::Sobol,
        StrategyName::Lhs,
        StrategyName::Extrema,
        StrategyName::Shells,
        StrategyName::Diagonal,
        StrategyName::Antidiagonal,
    ] {
        let chain = Comprehension::order(shuffled(), outer, Some(3));
        let direct = Comprehension::order(product(), outer, Some(3));
        validate(&chain, Mode::Permissive).unwrap();
        assert_eq!(rows(&chain), rows(&direct), "{outer:?}");
        assert_eq!(streamed_rows(&chain), streamed_rows(&direct), "{outer:?}");
        assert_equivalent(&chain, &scope);
    }
    // `lex/2` after a shuffle is the shuffle's first two tuples.
    let chain = Comprehension::order(shuffled(), StrategyName::Lex, Some(2));
    let expected: Vec<_> = rows(&shuffled()).into_iter().take(2).collect();
    assert_eq!(rows(&chain), expected);
    assert_ne!(
        rows(&chain),
        rows(&Comprehension::order(product(), StrategyName::Lex, Some(2)))
    );
    assert_eq!(
        streamed_rows(&chain),
        expected
            .iter()
            .map(|t| traversal_row(t))
            .collect::<Vec<_>>()
    );
    assert_equivalent(&chain, &scope);
}

/// An order's output is addressed through its selection (§3.6): position
/// `i` is its input's tuple at the `i`-th selected position, one axis as
/// long as the selection. Every strategy orders every strategy's output,
/// with and without truncation and seed: a strategy that selects from the
/// shape reads through an untruncated inner order (§7.4 O1), as it does
/// through an untruncated `Lex`, and otherwise the outer strategy's
/// selection over that one axis picks from the inner order's tuples. The
/// validator accepts every chain, both evaluators and the stream agree,
/// and the metadata counts the tuples and holds the outer selection.
#[test]
fn every_strategy_orders_every_order() {
    use polydat::iteration::comprehension::metadata::{IndexFn, Materialization};
    use polydat::iteration::comprehension::strategies::for_name;
    let scope = scope();
    let product = || Comprehension::cartesian(vec![range("a", 0, 4, 1), ints("b", &[5, 6, 7])]);
    let rows = |c: &Comprehension| evaluate_indexed(c, &scope).unwrap().to_vec();
    let seeded = |s: StrategyName| matches!(s, StrategyName::Shuffle | StrategyName::Lhs);
    let mut chains = 0;
    for inner in STRATEGIES {
        for inner_cut in [None, Some(5)] {
            for inner_seed in [None, Some(11)]
                .into_iter()
                .filter(|s| s.is_none() || seeded(inner))
            {
                let first = || Comprehension::order_seeded(product(), inner, inner_cut, inner_seed);
                let selected = rows(&first());
                let inner_bound = match first().metadata().cardinality {
                    polydat::iteration::comprehension::CardinalityClass::Bounded(n)
                    | polydat::iteration::comprehension::CardinalityClass::BoundedAtMost(n) => n,
                    other => panic!("{other:?}"),
                };
                for outer in STRATEGIES {
                    for outer_cut in [None, Some(3)] {
                        for outer_seed in [None, Some(29)]
                            .into_iter()
                            .filter(|s| s.is_none() || seeded(outer))
                        {
                            let chain =
                                Comprehension::order_seeded(first(), outer, outer_cut, outer_seed);
                            validate(&chain, Mode::Permissive).unwrap_or_else(|e| {
                                panic!("{outer:?} over {inner:?}/{inner_cut:?}: {e}")
                            });
                            let reads_through = inner_cut.is_none()
                                && (inner == StrategyName::Lex
                                    || for_name(outer).selects_from_shape());
                            let expected = if reads_through {
                                rows(&Comprehension::order_seeded(
                                    product(),
                                    outer,
                                    outer_cut,
                                    outer_seed,
                                ))
                            } else {
                                let len = selected.len() as u64;
                                let axis = IndexFn::Lattice {
                                    axis_sizes: vec![len],
                                };
                                for_name(outer)
                                    .select(&axis, len, outer_cut, outer_seed)
                                    .iter()
                                    .map(|p| selected[p as usize].clone())
                                    .collect()
                            };
                            assert_eq!(
                                rows(&chain),
                                expected,
                                "{outer:?}/{outer_cut:?} over {inner:?}/{inner_cut:?}"
                            );
                            assert_eq!(assert_equivalent(&chain, &scope), expected.len());
                            assert!(streamed(&chain).is_some(), "{chain:?}");
                            if !reads_through {
                                use polydat::iteration::comprehension::CardinalityClass;
                                let m = chain.metadata();
                                let (CardinalityClass::Bounded(bound)
                                | CardinalityClass::BoundedAtMost(bound)) = m.cardinality
                                else {
                                    panic!("{chain:?}: {:?}", m.cardinality)
                                };
                                assert_eq!(
                                    m.index_addressable,
                                    Some(IndexFn::Lattice {
                                        axis_sizes: vec![bound]
                                    }),
                                    "{chain:?}"
                                );
                                assert!(
                                    matches!(
                                        m.materialization,
                                        Materialization::BoundedBarrier { working_set_size }
                                            if working_set_size <= inner_bound
                                    ) || outer == StrategyName::Lex,
                                    "{chain:?}: {:?}",
                                    m.materialization
                                );
                            }
                            chains += 1;
                        }
                    }
                }
            }
        }
    }
    assert!(chains > 400, "{chains}");

    // The 50 Halton points of a continuous square, shuffled: the same
    // points in the shuffle's order of the 50 positions.
    let square = || Comprehension::cartesian(vec![unit("u"), unit("v")]);
    let points = Comprehension::order(square(), StrategyName::Halton, Some(50));
    let shuffled =
        Comprehension::order_seeded(points.clone(), StrategyName::Shuffle, None, Some(3));
    validate(&shuffled, Mode::Permissive).unwrap();
    let drawn = rows(&points);
    let order = for_name(StrategyName::Shuffle).select(
        &IndexFn::Lattice {
            axis_sizes: vec![50],
        },
        50,
        None,
        Some(3),
    );
    let expected: Vec<_> = order.iter().map(|p| drawn[p as usize].clone()).collect();
    assert_eq!(rows(&shuffled), expected);
    assert_ne!(rows(&shuffled), drawn);
    assert_eq!(assert_equivalent(&shuffled, &scope), 50);
    let m = shuffled.metadata();
    assert_eq!(
        m.cardinality,
        polydat::iteration::comprehension::CardinalityClass::Bounded(50)
    );
    assert_eq!(
        m.index_addressable,
        Some(IndexFn::Lattice {
            axis_sizes: vec![50]
        })
    );
    assert_eq!(
        m.materialization,
        Materialization::BoundedBarrier {
            working_set_size: 50
        }
    );

    // Through a filter, a strategy that selects from the shape ranks the
    // survivors by their positions beneath an untruncated order (§5 V5,
    // §7.4 O1), so the filter `true`, which R0a drops, changes nothing.
    for predicate in ["true", "{b} != 6"] {
        for outer in [
            StrategyName::Halton,
            StrategyName::Lhs,
            StrategyName::Extrema,
        ] {
            let through = Comprehension::order(
                Comprehension::filter(
                    Comprehension::order_seeded(product(), StrategyName::Shuffle, None, Some(5)),
                    predicate,
                ),
                outer,
                Some(3),
            );
            let direct =
                Comprehension::order(Comprehension::filter(product(), predicate), outer, Some(3));
            assert_eq!(rows(&through), rows(&direct), "{outer:?} {predicate}");
            assert_equivalent(&through, &scope);
        }
    }

    // The same in the language: a derivation shuffles a producer's
    // Halton points, and a traversal over it yields each point once.
    let src = "input cycle: u64\nsweep := for u in 0.0..1.0 order halton/50\n\
               mixed := for sweep order shuffle\nfor mixed {\n    x := hash(cycle)\n}\n";
    let mut kernel = polydat::dsl::compile_polydat_interpreter(src).unwrap();
    kernel.set_inputs(&[0]);
    let mut points = |wire: &str| -> Vec<u64> {
        let value = kernel.pull_ref(wire).clone();
        let mut us: Vec<u64> = value
            .as_streamer()
            .unwrap()
            .coordinate_stream()
            .unwrap()
            .map(|t| match &t.unwrap().bindings[0].1 {
                polydat::iteration::comprehension::strategies::TupleValue::F64(f) => f.to_bits(),
                other => panic!("{other:?}"),
            })
            .collect();
        us.sort_unstable();
        us
    };
    assert_eq!(points("mixed").len(), 50);
    assert_eq!(points("mixed"), points("sweep"));
    assert_eq!(kernel.traverse(0).unwrap().len(), 50);

    // Orders over orders inside a cartesian and a cycle zip, which read
    // the ordered operand by position on the stream.
    for shape in [
        Comprehension::cartesian(vec![
            Comprehension::order(
                Comprehension::order(product(), StrategyName::Sobol, Some(4)),
                StrategyName::ReverseLex,
                None,
            ),
            ints("c", &[1, 2]),
        ]),
        Comprehension::zip(
            vec![
                Comprehension::order_seeded(
                    Comprehension::order(product(), StrategyName::Halton, Some(5)),
                    StrategyName::Shuffle,
                    Some(3),
                    Some(1),
                ),
                range("k", 0, 7, 1),
            ],
            ZipMode::Cycle,
        ),
        Comprehension::order(
            Comprehension::cartesian(vec![
                Comprehension::order(unit("u"), StrategyName::Sobol, Some(3)),
                ints("c", &[1, 2]),
            ]),
            StrategyName::Extrema,
            Some(1),
        ),
    ] {
        validate(&shape, Mode::Permissive).unwrap();
        assert!(streamed(&shape).is_some(), "{shape:?}");
        assert!(assert_equivalent(&shape, &scope) > 0, "{shape:?}");
    }
}

/// V3 (§5): every name a source or predicate reads is bound by the
/// comprehension where it is read or supplied by the surface. A stream
/// supplies nothing; a `for` traversal supplies the names of the scope it
/// opens in, and captures them when it opens. A name resolved nowhere is
/// V3 on both: refused under strictness, and otherwise a warning, with
/// the name read as None, alike on the stream and the traversal. A name
/// only the enclosing scope has is the stream's `ContextRequired` for a
/// source and `PredicateContextRequired` for a predicate.
#[test]
fn names_resolve_in_the_comprehension_or_the_surface() {
    use polydat::iteration::comprehension::surfaces::compile;
    let scope = scope();
    let compile_program = |pragma: &str, text: &str| {
        polydat::dsl::compile_polydat_interpreter(&format!(
            "{pragma}input cycle: u64\nsweep := for {text}\nfor {text} {{\n    s := u64_add(k, 1)\n}}\n"
        ))
    };
    let compile_traversal = |pragma: &str, text: &str| {
        polydat::dsl::compile_polydat_interpreter(&format!(
            "{pragma}input cycle: u64\nfor {text} {{\n    s := u64_add(k, 1)\n}}\n"
        ))
    };
    let parse = |text: &str| {
        polydat::iteration::comprehension::spec::parse_comprehension_algebra(text).unwrap()
    };

    // Resolved nowhere: V3 on every surface, naming the name and where
    // it is read. Strict surfaces refuse it; lax ones read it as None,
    // and the stream and the traversal dispense the same tuples.
    for (text, name, site) in [
        ("k in 1..5 where {k} > {zz}", "zz", "predicate `{k} > {zz}`"),
        ("k in pow2({zz})", "zz", "clause 'k'"),
        ("k in zz_values", "zz_values", "clause 'k'"),
        ("k in 1..5 where {k} != s1", "s1", "predicate `{k} != s1`"),
        ("k in 1..5, j in pow2({m}), m in 1..3", "m", "clause 'j'"),
    ] {
        let ast = parse(text);
        for err in [
            check_names(&ast, Surface::Stream).unwrap_err(),
            check_names(&ast, Surface::Traversal(&in_scope)).unwrap_err(),
            CompiledComprehension::from_ast_with(&ast, Mode::Strict).unwrap_err(),
        ] {
            assert!(
                matches!(&err, ValidationError::V3UnresolvedNames { reads }
                    if reads.iter().any(|r| r.name == name)),
                "{text}: {err}"
            );
            assert!(err.to_string().contains(site), "{text}: {err}");
        }
        for program in [
            compile_traversal("pragma strict\n", text),
            compile_program("pragma strict\n", text),
        ] {
            let Err(err) = program else {
                panic!("{text} compiles under pragma strict")
            };
            let err = err.to_string();
            assert!(err.contains("V3:"), "{text}: {err}");
            assert!(err.contains(&format!("`{name}`")), "{text}: {err}");
        }
        for program in [compile_traversal("", text), compile_program("", text)] {
            let kernel = program.unwrap_or_else(|e| panic!("{text}: {e}"));
            let warnings = kernel.program().ledger().unresolved_names();
            assert!(
                warnings
                    .iter()
                    .all(|w| w.reads.iter().any(|r| r.name == name)),
                "{text}: {warnings:?}"
            );
        }
        let (_, report) = CompiledComprehension::from_ast_with(&ast, Mode::Permissive)
            .unwrap_or_else(|e| panic!("{text}: {e}"));
        assert!(
            matches!(report.warnings.first(),
                Some(ValidationWarning::UnresolvedNames { reads })
                    if reads.iter().any(|r| r.name == name)),
            "{text}"
        );
        assert!(
            compare(&ast, &scope).streamed,
            "{text} does not stream outside strictness"
        );
    }

    // Resolved in the scope: the traversal captures the name when it
    // opens, and a stream of the producer's wire refuses it. The same
    // comprehension compiled with no scope resolves it nowhere and reads
    // it as None.
    for (text, body, expected, predicate) in [
        (
            "k in 1..6 where {k} > {cycle}",
            "s := u64_add(k, 1)",
            vec![4, 5, 6],
            true,
        ),
        (
            "p in partitions(\"*/2\", {cycle})",
            "s := cardinality(p)",
            vec![1, 1],
            false,
        ),
    ] {
        let ast = parse(text);
        check_names(&ast, Surface::Traversal(&in_scope)).unwrap();
        assert!(matches!(
            CompiledComprehension::from_ast_with(&ast, Mode::Strict),
            Err(ValidationError::V3UnresolvedNames { .. })
        ));
        assert_eq!(compile(&ast).unwrap().coordinate_stream().count(), 0);
        let mut kernel = polydat::dsl::compile_polydat_interpreter(&format!(
            "input cycle: u64\nsweep := for {text}\nfor {text} {{\n    {body}\n}}\n"
        ))
        .unwrap_or_else(|e| panic!("{text}: {e}"));
        kernel.set_inputs(&[2]);
        let mut stream = kernel.traverse(0).unwrap();
        let mut values = Vec::new();
        while let Some(mut activation) = stream.advance().unwrap() {
            values.push(activation.cycle(0).pull("s").as_u64());
        }
        assert_eq!(values, expected, "{text}");
        let sweep = kernel.pull_ref("sweep").clone();
        let Err(err) = sweep.as_streamer().unwrap().coordinate_stream() else {
            panic!("{text} streams")
        };
        if predicate {
            assert!(
                matches!(err, ValidationError::PredicateContextRequired { ref references, .. }
                    if references == &["cycle".to_string()]),
                "{text}: {err}"
            );
        } else {
            assert!(
                matches!(err, ValidationError::ContextRequired { ref references, .. }
                    if references == &["cycle".to_string()]),
                "{text}: {err}"
            );
        }
    }

    // Bound by the comprehension: an earlier axis for a source, the
    // tuple for a predicate. The traversal accepts both; the stream
    // evaluates no dependent source.
    let dependent = parse("k in 1..4, j in pow2({k}) where {j} < {k}");
    check_names(&dependent, Surface::Stream).unwrap();
    assert!(matches!(
        compile(&dependent),
        Err(ValidationError::ContextRequired { .. })
    ));
    assert_eq!(assert_equivalent(&dependent, &scope), 3);
}

/// A shape whose count the metadata bounds but does not know reports
/// that bound, and a traversal over it counts what it dispenses: its
/// length is the number of tuples its evaluation kept at open.
#[test]
fn a_traversal_over_an_at_most_shape_counts_what_it_dispenses() {
    use polydat::iteration::comprehension::CardinalityClass;
    let scope = scope();
    let ast = Comprehension::cartesian(vec![
        Comprehension::filter(range("k", 1, 10, 1), "{k} > 3"),
        words("c", &["a", "b"]),
    ]);
    assert_eq!(
        ast.metadata().cardinality,
        CardinalityClass::BoundedAtMost(18)
    );
    assert_eq!(evaluate_indexed(&ast, &scope).unwrap().len(), 12);
    assert_eq!(assert_equivalent(&ast, &scope), 12);

    let text = "k in 1..10, c in a,b where {k} > 3";
    let src = format!(
        "input cycle: u64\nsweep := for {text}\nfor {text} {{\n    s := u64_add(k, 1)\n}}\n"
    );
    let mut kernel = polydat::dsl::compile_polydat_interpreter(&src).unwrap();
    kernel.set_inputs(&[0]);
    let sweep = kernel.pull_ref("sweep").clone();
    let streamer = sweep.as_streamer().unwrap();
    assert_eq!(streamer.cardinality(), CardinalityClass::BoundedAtMost(18));
    assert_eq!(streamer.coordinate_stream().unwrap().count(), 12);
    let mut stream = kernel.traverse(0).unwrap();
    assert_eq!(stream.len(), 12);
    let mut count = 0;
    while stream.advance().unwrap().is_some() {
        count += 1;
    }
    assert_eq!(count, 12);
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
