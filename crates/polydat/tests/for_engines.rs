// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Engine parity, step 8 (A5): a traversal's activations run on any
//! engine. The parent opens the traversal on the interpreter; each
//! activation is a fresh kernel over the body's program for the engine
//! the host chose, compiled once per engine, driven through the
//! `Kernel` trait with the same elements, cascade, and cursor narrowing,
//! and computing what the interpreter's activation computes.

use std::sync::Arc;

use polydat::ast::Value;
use polydat::dsl::compile_polydat;
use polydat::kernel::PolydatKernel;
use polydat::{Engine, Provenance};

fn compile(src: &str) -> PolydatKernel {
    compile_polydat(src).unwrap_or_else(|e| panic!("compile failed: {e}\n{src}"))
}

fn engines() -> Vec<Engine> {
    let mut all = vec![Engine::Closures(Provenance::Auto)];
    if cfg!(feature = "jit") {
        all.push(Engine::Native(Provenance::Auto));
    }
    all
}

/// Every (activation index, cycle, output) value of a traversal.
type Trace = Vec<(u64, u64, String, Value)>;

/// The trace on the interpreter and on `engine`.
fn traces(src: &str, coords: &[u64], outputs: &[&str], engine: Engine) -> (Trace, Trace) {
    let mut k = compile(src);
    k.set_inputs(coords);
    let stream = k.traverse(0).unwrap();
    let mut want = Vec::new();
    let mut got = Vec::new();
    for index in 0..stream.len() {
        let mut p1 = stream.activation(index).unwrap();
        p1.for_each_cycle(|i, kernel| {
            for name in outputs {
                want.push((index as u64, i, name.to_string(), kernel.pull(name).clone()));
            }
        });
        let mut act = stream
            .activation_on(index, engine)
            .unwrap_or_else(|e| panic!("{engine}: {e}"));
        assert_eq!(act.index, index as u64);
        assert_eq!(act.coords, p1.coords, "{engine}: coordinates");
        assert_eq!(act.cursor, p1.cursor, "{engine}: cursor slice");
        act.for_each_cycle(|i, kernel| {
            for name in outputs {
                got.push((index as u64, i, name.to_string(), kernel.pull(name)));
            }
        });
    }
    (want, got)
}

const SWEEP: &str = "input cycle: u64\nbase := hash(cycle)\nfor k in 1..4, limit in 10,20,30 {\n    f := u64_add(k, limit)\n    g := u64_add(base, f)\n}\n";

#[test]
fn activations_on_every_engine_compute_what_the_interpreters_do() {
    for engine in engines() {
        let (want, got) = traces(SWEEP, &[7], &["f", "g"], engine);
        assert_eq!(want.len(), 18);
        assert_eq!(got, want, "{engine}");
    }
}

const SLICED: &str = "input cycle: u64\nfor p in partitions(\"*/4\", 1000) {\n    cursor q = range(0, 1000) over p\n    n := cardinality(q.cursor)\n    o := q.cursor.start_ordinal\n    v := u64_add(q.ordinal, cycle)\n}\n";

#[test]
fn cursors_narrow_and_iterate_their_slice_on_every_engine() {
    for engine in engines() {
        let (want, got) = traces(SLICED, &[0], &["n", "o", "v"], engine);
        // Four partitions of 250 ordinals, one cycle per ordinal.
        assert_eq!(want.len(), 4 * 250 * 3);
        assert_eq!(got, want, "{engine}");
    }
}

#[test]
fn a_body_compiles_once_per_engine_and_every_activation_shares_it() {
    let mut k = compile(SWEEP);
    k.set_inputs(&[1]);
    let stream = k.traverse(0).unwrap();
    for engine in engines() {
        let a = stream.activation_on(0, engine).unwrap();
        let built = polydat::kernel::programs_built();
        let b = stream.activation_on(8, engine).unwrap();
        // T2: the second activation builds nothing.
        assert_eq!(polydat::kernel::programs_built(), built, "{engine}");
        assert_eq!(a.kernel.engine(), b.kernel.engine());
        let first = stream.traversal().program_on(engine).unwrap();
        let again = stream.traversal().program_on(engine).unwrap();
        assert!(
            Arc::ptr_eq(&first, &again),
            "{engine}: one program per engine"
        );
    }
}

#[test]
fn a_body_with_its_own_traversal_is_an_interpreter_activation() {
    let src = "input cycle: u64\nfor a in 1..3 {\n    x := u64_add(a, cycle)\n    for b in 1..3 {\n        y := u64_add(x, b)\n    }\n}\n";
    let mut k = compile(src);
    k.set_inputs(&[0]);
    let stream = k.traverse(0).unwrap();
    // The interpreter activation opens the inner traversal.
    let mut outer = stream.activation(0).unwrap();
    let inner = outer.kernel.traverse(0).unwrap();
    assert_eq!(inner.len(), 2);
    // Every inner activation may run on any engine.
    for engine in engines() {
        let mut act = inner.activation_on(1, engine).unwrap();
        assert_eq!(act.cycle(0).pull("y"), Value::U64(3));
    }
    // The outer body declares a traversal, so no other engine runs it.
    for engine in engines() {
        let err = stream
            .activation_on(0, engine)
            .err()
            .unwrap_or_else(|| panic!("{engine}: expected a refusal"));
        assert!(err.contains("interpreter activation"), "{engine}: {err}");
    }
}
