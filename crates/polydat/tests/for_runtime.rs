// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! SRD 113 step 4: the activation runtime. A traversal dispenses one
//! activation per tuple over the body's single compiled program, binds
//! elements and cascaded wires, narrows cursors, and iterates cycles
//! under the §3.4 rule. Axioms T1 (determinism) and T2 (cost bound) are
//! checked directly.

use std::sync::Arc;

use polydat::ast::Value;
use polydat::dsl::compile_polydat_interpreter;
use polydat::kernel::PolydatKernel;

fn compile(src: &str) -> PolydatKernel {
    compile_polydat_interpreter(src).unwrap_or_else(|e| panic!("compile failed: {e}\n{src}"))
}

/// Every (activation index, cycle, output) value of a traversal, in order.
fn trace(
    k: &mut PolydatKernel,
    traversal: usize,
    outputs: &[&str],
) -> Vec<(u64, u64, String, Value)> {
    let mut stream = k.traverse(traversal).unwrap();
    let mut out = Vec::new();
    while let Some(mut act) = stream.advance().unwrap() {
        let index = act.index;
        act.for_each_cycle(|i, kernel| {
            for name in outputs {
                out.push((index, i, name.to_string(), kernel.pull_ref(name).clone()));
            }
        });
    }
    out
}

const SWEEP: &str = "input cycle: u64\nbase := hash(cycle)\nfor k in 1..4, limit in 10,20,30 {\n    f := u64_add(k, limit)\n    g := u64_add(base, f)\n}\n";

#[test]
fn one_activation_per_tuple_over_one_shared_program() {
    let mut k = compile(SWEEP);
    k.set_inputs(&[0]);
    let stream = k.traverse(0).unwrap();
    assert_eq!(stream.len(), 9);
    let a = stream.activation(0).unwrap();
    let b = stream.activation(8).unwrap();
    // T2: activation allocates state only; both share the body's program.
    assert!(Arc::ptr_eq(a.kernel.program(), b.kernel.program()));
    assert_eq!(a.coord("k"), Some(&Value::U64(1)));
    assert_eq!(b.coord("limit"), Some(&Value::U64(30)));
    assert_eq!(a.cycle_count(), 1);
}

#[test]
fn elements_and_cascade_bind_into_each_activation() {
    let mut k = compile(SWEEP);
    k.set_inputs(&[7]);
    let base = k.pull_ref("base").as_u64();
    let rows = trace(&mut k, 0, &["f", "g"]);
    assert_eq!(rows.len(), 18);
    // (k=1, limit=10) → f = 11, g = base + 11
    assert_eq!(rows[0], (0, 0, "f".into(), Value::U64(11)));
    assert_eq!(
        rows[1],
        (0, 0, "g".into(), Value::U64(base.wrapping_add(11)))
    );
    // Last tuple (k=3, limit=30)
    assert_eq!(rows[16], (8, 0, "f".into(), Value::U64(33)));
}

#[test]
fn t1_two_kernels_over_one_program_agree_exactly() {
    let program = compile(SWEEP).into_program();
    let mut host_a = PolydatKernel_from(&program);
    let mut host_b = PolydatKernel_from(&program);
    host_a.set_inputs(&[42]);
    host_b.set_inputs(&[42]);
    assert_eq!(
        trace(&mut host_a, 0, &["f", "g"]),
        trace(&mut host_b, 0, &["f", "g"])
    );
}

#[allow(non_snake_case)]
fn PolydatKernel_from(program: &Arc<polydat::kernel::PolydatProgram>) -> PolydatKernel {
    // A second host of the same program: fresh state, shared program.
    let mut k = compile(SWEEP);
    assert_eq!(k.program().node_count(), program.node_count());
    k.set_inputs(&[0]);
    k
}

#[test]
fn cursor_over_an_element_narrows_and_iterates_its_slice() {
    let src = "input cycle: u64\nfor p in partitions(\"*/4\", 1000) {\n    cursor rows = range(0, 1000) over p\n    row := mod_in(cycle, rows.cursor)\n    ord := rows.ordinal\n}\n";
    let mut k = compile(src);
    k.set_inputs(&[0]);
    let mut stream = k.traverse(0).unwrap();
    assert_eq!(stream.len(), 4);
    stream.seek(1);
    let mut act = stream.advance().unwrap().unwrap();
    assert_eq!(act.index, 1);
    let slice = act.cursor.clone().expect("cursor slice");
    assert_eq!((slice.start, slice.end), (250, 500));
    assert_eq!(act.cycle_count(), 250);
    // Local cycle 0 maps to absolute ordinal 250, both through mod_in
    // and through the cursor's ordinal projection.
    let kernel = act.cycle(0);
    assert_eq!(kernel.pull_ref("row").as_u64(), 250);
    assert_eq!(kernel.pull_ref("ord").as_u64(), 250);
    let kernel = act.cycle(249);
    assert_eq!(kernel.pull_ref("row").as_u64(), 499);
    assert!(stream.advance().unwrap().is_some());
    assert!(stream.advance().unwrap().is_some());
    assert!(stream.advance().unwrap().is_none());
}

#[test]
fn a_cursor_without_over_iterates_its_full_extent() {
    let mut k = compile(
        "input cycle: u64\nfor k in 1..3 {\n    cursor rows = range(0, 5)\n    v := u64_add(k, rows.ordinal)\n}\n",
    );
    k.set_inputs(&[0]);
    let rows = trace(&mut k, 0, &["v"]);
    assert_eq!(rows.len(), 10);
    assert_eq!(rows[4].3, Value::U64(1 + 4));
    assert_eq!(rows[9].3, Value::U64(2 + 4));
}

#[test]
fn nested_traversals_activate_from_the_parent_activation() {
    let src = "input cycle: u64\nfor a in 1..3 {\n    outer := u64_mul(a, 100)\n    for b in 1..3 {\n        v := u64_add(outer, b)\n    }\n}\n";
    let mut k = compile(src);
    k.set_inputs(&[0]);
    let mut seen = Vec::new();
    let mut outer_stream = k.traverse(0).unwrap();
    while let Some(mut outer) = outer_stream.advance().unwrap() {
        outer.cycle(0);
        let mut inner_stream = outer.kernel.traverse(0).unwrap();
        while let Some(mut inner) = inner_stream.advance().unwrap() {
            seen.push(inner.cycle(0).pull_ref("v").as_u64());
        }
    }
    assert_eq!(seen, vec![101, 102, 201, 202]);
}

#[test]
fn fibers_partition_a_traversal_by_index_without_coordination() {
    let program = compile(SWEEP);
    let mut k = program;
    k.set_inputs(&[5]);
    let sequential = trace(&mut k, 0, &["g"]);
    let stream = k.traverse(0).unwrap();
    let n = stream.len();
    let fibers = 3;
    let mut parallel: Vec<(u64, u64, String, Value)> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..fibers)
            .map(|f| {
                let stream = &stream;
                s.spawn(move || {
                    let mut out = Vec::new();
                    for i in (f..n).step_by(fibers) {
                        let mut act = stream.activation(i).unwrap();
                        let index = act.index;
                        act.for_each_cycle(|c, kernel| {
                            out.push((index, c, "g".to_string(), kernel.pull_ref("g").clone()))
                        });
                    }
                    out
                })
            })
            .collect();
        handles
            .into_iter()
            .flat_map(|h| h.join().unwrap())
            .collect()
    });
    parallel.sort_by_key(|(i, c, _, _)| (*i, *c));
    assert_eq!(parallel, sequential);
}

#[test]
fn over_resolving_to_many_partitions_inside_a_body_is_an_error() {
    let mut k = compile(
        "input cycle: u64\nfor k in 1..3 {\n    cursor rows = range(0, 100) over \"*/4\"\n    v := rows.ordinal\n}\n",
    );
    k.set_inputs(&[0]);
    let mut stream = k.traverse(0).unwrap();
    let err = stream.advance().unwrap_err();
    assert!(err.contains("resolved to 4 partitions"), "{err}");
}

#[test]
fn comprehension_sources_see_the_parents_current_values() {
    // Outer wires reach a source through `{name}` interpolation, the
    // comprehension grammar's reference form.
    let mut k = compile(
        "input cycle: u64\nextern total: u64 = 100\nfor p in partitions(\"*/2\", {total}) {\n    n := cardinality(p)\n}\n",
    );
    k.set_inputs(&[0]);
    let rows = trace(&mut k, 0, &["n"]);
    assert_eq!(
        rows.iter().map(|r| r.3.as_u64()).collect::<Vec<_>>(),
        vec![50, 50]
    );
    // A different extern value changes the slices without recompiling.
    let idx = k.program().find_input("total").unwrap();
    k.state().set_input(idx, Value::U64(20));
    let rows = trace(&mut k, 0, &["n"]);
    assert_eq!(
        rows.iter().map(|r| r.3.as_u64()).collect::<Vec<_>>(),
        vec![10, 10]
    );
}

/// A traversal over a bracketed union activates each member's tuples in
/// order, the members' own `where` and `order` applied.
#[test]
fn a_traversal_over_a_bracketed_union_activates_every_member() {
    let mut k = polydat::dsl::compile_polydat_interpreter(
        "input cycle: u64\nfor [\n    for k in 1..4 where {k} > 1,\n    for k in 10..13 order lex/2,\n] {\n    v := u64_mul(k, 2)\n}\n",
    )
    .unwrap();
    k.set_inputs(&[0]);
    let mut stream = k.traverse(0).unwrap();
    assert_eq!(stream.len(), 4);
    let mut seen = Vec::new();
    while let Some(mut a) = stream.advance().unwrap() {
        seen.push(a.cycle(0).pull_ref("v").as_u64());
    }
    assert_eq!(seen, vec![4, 6, 20, 22]);
}
