// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Context-free generators are flattened at compile
//! (comprehension_forms.md §10.7.0, §10.7.7): a call that references
//! no name is evaluated once by the compile and traversed as the
//! literal of its values, so its cardinality is exact and an order
//! over it validates; a call that references a name is left to the
//! traversal.

use polydat::dsl::compile_polydat;
use polydat::iteration::comprehension::cardinality::CardinalityClass;

fn traversed(src: &str) -> Vec<u64> {
    let mut k = compile_polydat(src).unwrap_or_else(|e| panic!("{e}\n{src}"));
    k.set_inputs(&[0]);
    let mut stream = k.traverse(0).unwrap();
    let mut seen = Vec::new();
    while let Some(mut a) = stream.advance().unwrap() {
        seen.push(a.cycle(0).pull("v").as_u64());
    }
    seen
}

/// An order over a context-free generator validates and samples: the
/// call is a literal by the time V6 looks at it.
#[test]
fn an_order_over_a_context_free_generator_validates_and_samples() {
    let seen = traversed(
        "input cycle: u64\nfor k in fib(8) order halton/4 {\n    v := u64_add(k, 0)\n}\n",
    );
    assert_eq!(seen.len(), 4);
    assert!(
        seen.iter().all(|v| [1, 2, 3, 5, 8, 13, 21].contains(v)),
        "{seen:?}"
    );
    let seen = traversed(
        "input cycle: u64\nfor k in pow2(5) order extrema/1 {\n    v := u64_add(k, 0)\n}\n",
    );
    assert_eq!(seen, vec![1, 16]);
}

/// A producer over a context-free generator has an exact cardinality,
/// and its stream is the generator's values.
#[test]
fn a_producer_over_a_context_free_generator_is_bounded_by_its_values() {
    let mut k = compile_polydat("input cycle: u64\nfibs := for k in fib(8)\n").unwrap();
    k.set_inputs(&[0]);
    let streamer = k.pull("fibs").clone();
    let streamer = streamer.as_streamer().unwrap();
    assert!(
        matches!(streamer.cardinality(), CardinalityClass::Bounded(8)),
        "{:?}",
        streamer.cardinality()
    );
    let mut stream = streamer.coordinate_stream().unwrap();
    let mut n = 0;
    while stream.advance().is_some() {
        n += 1;
    }
    assert_eq!(n, 8);
}

/// A generator that references a name the compile cannot bind is left
/// to the traversal, so an order over it is still the validator's to
/// refuse: unbounded and without a closed-form index function (V4).
#[test]
fn a_generator_over_a_runtime_name_stays_unbounded_at_compile() {
    let err = compile_polydat(
        "input cycle: u64\ninput n: u64\nfor k in pow2({n}) order halton/2 {\n    v := u64_add(k, 0)\n}\n",
    )
    .unwrap_err();
    assert!(err.contains("V4"), "{err}");
    // Without an order the traversal evaluates it with the wire bound.
    let mut k = compile_polydat(
        "input cycle: u64\nextern total: u64 = 100\nfor p in partitions(\"*/2\", {total}) {\n    v := cardinality(p)\n}\n",
    )
    .unwrap();
    k.set_inputs(&[0]);
    let mut stream = k.traverse(0).unwrap();
    let mut seen = Vec::new();
    while let Some(mut a) = stream.advance().unwrap() {
        seen.push(a.cycle(0).pull("v").as_u64());
    }
    assert_eq!(seen, vec![50, 50]);
}
