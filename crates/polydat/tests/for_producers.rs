// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! SRD 113 step 3: producer wires. `name := for ...` binds a Streamer
//! value; derived forms filter and order a bound producer; streams from
//! one wire are independent; cardinality metadata is exposed.

use polydat::ast::PortType;
use polydat::dsl::compile_polydat_interpreter;
use polydat::iteration::comprehension::cardinality::CardinalityClass;
use polydat::iteration::comprehension::strategies::TupleValue;

fn compile(src: &str) -> polydat::kernel::PolydatKernel {
    compile_polydat_interpreter(src).unwrap_or_else(|e| panic!("compile failed: {e}\n{src}"))
}

fn tuples(stream: polydat::iteration::comprehension::surfaces::CoordinateStream) -> Vec<Vec<i64>> {
    stream
        .map(|t| {
            t.bindings
                .iter()
                .map(|(_, v)| match v {
                    TupleValue::I64(n) => *n,
                    TupleValue::U64(n) => *n as i64,
                    other => panic!("unexpected {other:?}"),
                })
                .collect()
        })
        .collect()
}

#[test]
fn producer_binding_is_a_const_streamer_wire() {
    let mut k = compile("input cycle: u64\nsweep := for k in 1..4, limit in 10,20,30\n");
    let p = k.program();
    assert!(p.output_names().contains(&"sweep"));
    assert!(p.const_outputs().contains(&"sweep"));
    assert_eq!(p.output_port_type("sweep"), Some(PortType::Ext));
    k.set_inputs(&[0]);
    let v = k.pull_ref("sweep").clone();
    let s = v.as_streamer().expect("streamer value");
    assert_eq!(s.text, "k in 1..4, limit in 10, 20, 30");
    assert_eq!(s.element_names(), vec!["k", "limit"]);
    assert_eq!(s.cardinality(), CardinalityClass::Bounded(9));
    assert_eq!(v.to_display_string(), "for k in 1..4, limit in 10, 20, 30");
}

#[test]
fn streams_from_one_wire_advance_independently() {
    let mut k = compile("input cycle: u64\nsweep := for k in 1..4, limit in 10,20,30\n");
    k.set_inputs(&[0]);
    let v = k.pull_ref("sweep").clone();
    let s = v.as_streamer().unwrap();
    let mut a = s.coordinate_stream().unwrap();
    let b = s.coordinate_stream().unwrap();
    a.next();
    a.next();
    let b_all = tuples(b);
    assert_eq!(b_all.len(), 9);
    assert_eq!(b_all[0], vec![1, 10]);
    let a_rest: Vec<Vec<i64>> = tuples(a);
    assert_eq!(a_rest.len(), 7);
    assert_eq!(a_rest[0], vec![1, 30]);
}

#[test]
fn derived_where_filters_and_keeps_the_base_shape() {
    let mut k = compile(
        "input cycle: u64\nbase := for k in 1..4, limit in 10,20,30\nedges := for base where {k} == 1 || {k} == 3\n",
    );
    k.set_inputs(&[0]);
    let v = k.pull_ref("edges").clone();
    let s = v.as_streamer().unwrap();
    assert_eq!(s.element_names(), vec!["k", "limit"]);
    assert!(matches!(
        s.cardinality(),
        CardinalityClass::BoundedAtMost(9)
    ));
    let got = tuples(s.coordinate_stream().unwrap());
    assert_eq!(
        got,
        vec![
            vec![1, 10],
            vec![1, 20],
            vec![1, 30],
            vec![3, 10],
            vec![3, 20],
            vec![3, 30]
        ]
    );
}

#[test]
fn derived_order_permutes_and_truncates() {
    let mut k = compile(
        "input cycle: u64\nbase := for k in 1..4, limit in 10,20,30\nlast := for base order reverse_lex/2\n",
    );
    k.set_inputs(&[0]);
    let v = k.pull_ref("last").clone();
    let got = tuples(v.as_streamer().unwrap().coordinate_stream().unwrap());
    assert_eq!(got, vec![vec![3, 30], vec![3, 20]]);
}

#[test]
fn derivations_chain_and_each_wire_is_distinct() {
    let mut k = compile(
        "input cycle: u64\nbase := for k in 1..4, limit in 10,20,30\nedges := for base where {k} != 2\ntail := for edges order reverse_lex\n",
    );
    k.set_inputs(&[0]);
    let base = tuples(
        k.pull_ref("base")
            .clone()
            .as_streamer()
            .unwrap()
            .coordinate_stream()
            .unwrap(),
    );
    let edges = tuples(
        k.pull_ref("edges")
            .clone()
            .as_streamer()
            .unwrap()
            .coordinate_stream()
            .unwrap(),
    );
    let tail = tuples(
        k.pull_ref("tail")
            .clone()
            .as_streamer()
            .unwrap()
            .coordinate_stream()
            .unwrap(),
    );
    assert_eq!(base.len(), 9);
    assert_eq!(edges.len(), 6);
    assert_eq!(tail.len(), 6);
    assert_eq!(tail[0], vec![3, 30]);
    assert_eq!(edges[0], vec![1, 10]);
}

#[test]
fn traversal_over_a_derived_producer_resolves_elements() {
    let k = compile(
        "input cycle: u64\nbase := for k in 1..4, limit in 10,20,30\nedges := for base where {k} == 1\nfor edges {\n    f := u64_add(k, limit)\n}\n",
    );
    let p = k.program();
    assert_eq!(p.producers().len(), 2);
    assert_eq!(
        p.traversals()[0].elements,
        vec![
            ("k".to_string(), PortType::U64),
            ("limit".to_string(), PortType::U64)
        ]
    );
    assert!(matches!(
        p.traversals()[0].comprehension,
        polydat::iteration::comprehension::Comprehension::Filter { .. }
    ));
}

#[test]
fn derived_form_over_an_unknown_base_is_an_error() {
    let err =
        compile_polydat_interpreter("input cycle: u64\nedges := for nowhere where {k} == 1\n")
            .unwrap_err();
    assert!(err.to_string().contains("nowhere"), "{err}");
    let err = compile_polydat_interpreter(
        "input cycle: u64\nbase := for k in 1..4\nbad := for base order zigzag\n",
    )
    .unwrap_err();
    assert!(err.to_string().contains("zigzag"), "{err}");
}

#[test]
fn producer_wire_is_visible_to_ordinary_bindings() {
    // The wire is a value like any other: interpolation renders its
    // display form.
    let mut k = compile("input cycle: u64\nsweep := for k in 1..4\nlabel := \"plan={sweep}\"\n");
    k.set_inputs(&[0]);
    assert_eq!(k.pull_ref("label").as_str(), "plan=for k in 1..4");
}

/// The bracketed union form (comprehension_forms.md §8.1, §8.2): each
/// member is a full comprehension with its own modifiers, and the union
/// takes modifiers of its own after the bracket.
#[test]
fn a_bracketed_union_carries_each_members_modifiers() {
    let mut k = compile(
        "input cycle: u64\nmixed := for [\n    for k in 1..4 where {k} > 1,\n    for k in 10..13 order lex/2,\n]\n",
    );
    k.set_inputs(&[0]);
    let v = k.pull_ref("mixed").clone();
    let got = tuples(v.as_streamer().unwrap().coordinate_stream().unwrap());
    assert_eq!(got, vec![vec![2], vec![3], vec![10], vec![11]]);
}

/// `shuffle/n` is a text spelling of the algebra's `Shuffle`, and an
/// order over a union is the algebra's to accept (V4), not the text
/// front end's to refuse.
#[test]
fn shuffle_and_an_ordered_union_are_text_forms() {
    let mut k = compile(
        "input cycle: u64\nsome := for k in 1..10 order shuffle/3\nboth := for k in 1..3, k in 10..12 order halton/2\n",
    );
    k.set_inputs(&[0]);
    let some = tuples(
        k.pull_ref("some")
            .clone()
            .as_streamer()
            .unwrap()
            .coordinate_stream()
            .unwrap(),
    );
    assert_eq!(some.len(), 3);
    assert!(some.iter().all(|t| (1..10).contains(&t[0])));
    let both = tuples(
        k.pull_ref("both")
            .clone()
            .as_streamer()
            .unwrap()
            .coordinate_stream()
            .unwrap(),
    );
    assert_eq!(both.len(), 2);
}

/// A producer over a context-required source traverses, where the
/// scope binds its names, and has no coordinate stream: the
/// scope-less surface refuses it by name (comprehension_forms.md
/// §9.5.2, §10.7.0) instead of dispensing nothing.
#[test]
fn a_context_required_producer_traverses_but_has_no_coordinate_stream() {
    let mut k = compile(
        "input cycle: u64\nextern total: u64 = 100\nparts := for p in partitions(\"*/2\", {total})\nfor parts {\n    n := cardinality(p)\n}\n",
    );
    k.set_inputs(&[0]);
    let mut stream = k.traverse(0).unwrap();
    let mut seen = Vec::new();
    while let Some(mut a) = stream.advance().unwrap() {
        seen.push(a.cycle(0).pull_ref("n").as_u64());
    }
    assert_eq!(seen, vec![50, 50]);
    let streamer = k.pull_ref("parts").clone();
    let err = streamer
        .as_streamer()
        .unwrap()
        .coordinate_stream()
        .expect_err("a context-required source has no coordinate stream");
    let text = err.to_string();
    assert!(text.contains("'p'") && text.contains("total"), "{text}");
}

/// A traversal captures its sources' references from the enclosing
/// scope when it opens (for_traversal.md §3.1): a coordinate input as
/// much as an extern, read once, so an open traversal is a snapshot and
/// the next open reads the new values.
#[test]
fn a_traversal_captures_its_sources_references_when_it_opens() {
    let mut k = compile(
        "input cycle: u64\nparts := for p in partitions(\"*/2\", {cycle})\nfor parts {\n    n := cardinality(p)\n}\n",
    );
    let counts = |k: &mut polydat::kernel::PolydatKernel| -> Vec<u64> {
        let mut stream = k.traverse(0).unwrap();
        let mut seen = Vec::new();
        while let Some(mut a) = stream.advance().unwrap() {
            seen.push(a.cycle(0).pull_ref("n").as_u64());
        }
        seen
    };
    k.set_inputs(&[8]);
    assert_eq!(counts(&mut k), vec![4, 4]);
    k.set_inputs(&[100]);
    assert_eq!(counts(&mut k), vec![50, 50]);
    // An open traversal is a snapshot: a change after the open does not
    // reach it.
    k.set_inputs(&[8]);
    let mut stream = k.traverse(0).unwrap();
    k.set_inputs(&[100]);
    let mut seen = Vec::new();
    while let Some(mut a) = stream.advance().unwrap() {
        seen.push(a.cycle(0).pull_ref("n").as_u64());
    }
    assert_eq!(seen, vec![4, 4]);
}
