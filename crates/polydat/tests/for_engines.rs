// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Engine parity, step 8 (A5): traversals open and run on any engine.
//! A kernel on any engine opens the traversals its program declares
//! through the `Kernel` trait, against the values it holds; each
//! activation is a fresh kernel over the body's program for the engine
//! the host chose, compiled once per engine, with the same elements,
//! cascade, and cursor narrowing, computing what the interpreter's
//! activation computes; and a body's own traversals open from that
//! activation, so every level of a nest runs compiled.

use std::sync::Arc;

use polydat::ast::Value;
use polydat::dsl::compile::{CompileOptions, compile_polydat_with_engine};
use polydat::dsl::compile_polydat_interpreter;
use polydat::kernel::PolydatKernel;
use polydat::kernel::activation::TraversalStream;
use polydat::{Engine, JitMode, Kernel, Provenance};

fn compile(src: &str) -> PolydatKernel {
    compile_polydat_interpreter(src).unwrap_or_else(|e| panic!("compile failed: {e}\n{src}"))
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
                want.push((
                    index as u64,
                    i,
                    name.to_string(),
                    kernel.pull_ref(name).clone(),
                ));
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
        let ledger = k.program().ledger().clone();
        let built = ledger.programs();
        let b = stream.activation_on(8, engine).unwrap();
        // T2: the second activation builds nothing.
        assert_eq!(ledger.programs(), built, "{engine}");
        assert_eq!(a.kernel.engine(), b.kernel.engine());
        let first = stream.traversal().program_on(engine).unwrap();
        let again = stream.traversal().program_on(engine).unwrap();
        assert!(
            Arc::ptr_eq(&first, &again),
            "{engine}: one program per engine"
        );
    }
}

/// The trace of `stream` with every activation on `engine`.
fn trace_on(stream: &TraversalStream, outputs: &[&str], engine: Engine) -> Trace {
    let mut got = Vec::new();
    for index in 0..stream.len() {
        let mut act = stream
            .activation_on(index, engine)
            .unwrap_or_else(|e| panic!("{engine}: {e}"));
        act.for_each_cycle(|i, kernel| {
            for name in outputs {
                got.push((index as u64, i, name.to_string(), kernel.pull(name)));
            }
        });
    }
    got
}

fn compile_on(src: &str, engine: Engine) -> Box<dyn Kernel> {
    compile_polydat_with_engine(src, engine, &CompileOptions::default(), None)
        .unwrap_or_else(|e| panic!("{engine}: compile failed: {e}\n{src}"))
}

fn same_tier(a: Engine, b: Engine) -> bool {
    matches!(
        (a, b),
        (Engine::Interpreter(_), Engine::Interpreter(_))
            | (Engine::Closures(_), Engine::Closures(_))
            | (Engine::Native(_), Engine::Native(_))
    )
}

#[test]
fn a_compiled_parent_opens_a_traversal_as_the_interpreter_does() {
    for src in [SWEEP, SLICED] {
        let outputs: &[&str] = if src == SWEEP {
            &["f", "g"]
        } else {
            &["n", "o", "v"]
        };
        let mut p1 = compile(src);
        p1.set_inputs(&[7]);
        let want_stream = p1.traverse(0).unwrap();
        let want = trace_on(&want_stream, outputs, Engine::Interpreter(JitMode::Auto));
        for engine in engines() {
            let mut root = compile_on(src, engine);
            assert!(
                same_tier(root.engine(), engine),
                "{engine}: {}",
                root.engine()
            );
            root.set_inputs(&[7]);
            let stream = root.traverse(0).unwrap_or_else(|e| panic!("{engine}: {e}"));
            assert_eq!(stream.len(), want_stream.len(), "{engine}: activations");
            for index in 0..stream.len() {
                let a = stream.activation(index).unwrap();
                let b = want_stream.activation(index).unwrap();
                assert_eq!(a.coords, b.coords, "{engine}: coordinates of {index}");
                assert_eq!(a.cursor, b.cursor, "{engine}: cursor of {index}");
            }
            assert_eq!(trace_on(&stream, outputs, engine), want, "{engine}");
            assert_eq!(root.traverse_all().unwrap().len(), 1, "{engine}");
        }
    }
}

const NEST: &str = "input cycle: u64\nbase := u64_add(cycle, 100)\nfor a in 1..3 {\n    x := u64_add(a, base)\n    for b in 1..3 {\n        y := u64_add(x, b)\n        for c in 1..3 {\n            z := u64_add(y, c)\n        }\n    }\n}\n";

/// One (a, b, c, z) row of the nest.
type NestRow = (u64, u64, u64, Value);

/// Every row of the nest, opening each level on `engine` from the
/// activation above it, with the engine every level reports.
fn nest_on(engine: Engine, coord: u64) -> (Vec<NestRow>, Vec<Engine>) {
    let mut root = compile_on(NEST, engine);
    root.set_inputs(&[coord]);
    let mut rows = Vec::new();
    let mut seen = vec![root.engine()];
    let outer = root.traverse(0).unwrap_or_else(|e| panic!("{engine}: {e}"));
    for i in 0..outer.len() {
        let mut a = outer.activation_on(i, engine).unwrap();
        seen.push(a.kernel.engine());
        let a_val = a.coord("a").unwrap().clone();
        let middle = a
            .kernel
            .traverse(0)
            .unwrap_or_else(|e| panic!("{engine}: {e}"));
        for j in 0..middle.len() {
            let mut b = middle.activation_on(j, engine).unwrap();
            seen.push(b.kernel.engine());
            let b_val = b.coord("b").unwrap().clone();
            let inner = b
                .kernel
                .traverse(0)
                .unwrap_or_else(|e| panic!("{engine}: {e}"));
            for k in 0..inner.len() {
                let mut c = inner.activation_on(k, engine).unwrap();
                seen.push(c.kernel.engine());
                let c_val = c.coord("c").unwrap().clone();
                let (Value::U64(av), Value::U64(bv), Value::U64(cv)) = (&a_val, &b_val, &c_val)
                else {
                    panic!("{engine}: element types {a_val:?} {b_val:?} {c_val:?}");
                };
                rows.push((*av, *bv, *cv, c.cycle(0).pull("z")));
            }
        }
    }
    (rows, seen)
}

#[test]
fn a_nest_runs_compiled_at_every_level() {
    let (want, p1) = nest_on(Engine::Interpreter(JitMode::Auto), 5);
    assert_eq!(want.len(), 8);
    assert!(p1.iter().all(|e| *e == Engine::Interpreter(JitMode::Auto)));
    for (a, b, c, z) in &want {
        assert_eq!(*z, Value::U64(105 + a + b + c));
    }
    for engine in engines() {
        let (got, seen) = nest_on(engine, 5);
        assert_eq!(got, want, "{engine}");
        // The root, both outer activations, four middle ones, eight
        // inner ones: every kernel in the nest is on the chosen engine.
        assert_eq!(seen.len(), 1 + 2 + 4 + 8, "{engine}");
        for e in seen {
            assert!(same_tier(e, engine), "{engine}: a level ran on {e}");
        }
    }
}

#[test]
fn a_body_with_its_own_traversal_opens_it_from_any_activation() {
    let src = "input cycle: u64\nfor a in 1..3 {\n    x := u64_add(a, cycle)\n    for b in 1..3 {\n        y := u64_add(x, b)\n    }\n}\n";
    let mut k = compile(src);
    k.set_inputs(&[0]);
    let stream = k.traverse(0).unwrap();
    for engine in engines() {
        let mut outer = stream.activation_on(0, engine).unwrap();
        let inner = outer.kernel.traverse(0).unwrap();
        assert_eq!(inner.len(), 2);
        assert_eq!(
            inner
                .traversal()
                .cascade
                .iter()
                .map(|(n, _)| n.as_str())
                .collect::<Vec<_>>(),
            ["x"],
            "{engine}: the body cascades what it references"
        );
        for e in engines()
            .into_iter()
            .chain([Engine::Interpreter(JitMode::Auto)])
        {
            let mut act = inner.activation_on(1, e).unwrap();
            assert_eq!(act.cycle(0).pull("y"), Value::U64(3), "{engine} -> {e}");
        }
    }
}

#[test]
fn a_kernel_created_from_a_compiled_program_opens_its_traversals() {
    for engine in engines() {
        let root = compile_on(SWEEP, engine);
        let program = root.into_program();
        let mut a = program.clone().create_kernel();
        let mut b = program.create_kernel();
        a.set_inputs(&[3]);
        b.set_inputs(&[4]);
        assert_eq!(a.traversals().len(), 1, "{engine}");
        let sa = a.traverse(0).unwrap_or_else(|e| panic!("{engine}: {e}"));
        let sb = b.traverse(0).unwrap_or_else(|e| panic!("{engine}: {e}"));
        assert_eq!(sa.len(), 9, "{engine}");
        assert_eq!(sb.len(), 9, "{engine}");
        // The cascade is each kernel's own: `base` differs by coordinate.
        let ga = trace_on(&sa, &["g"], engine);
        let gb = trace_on(&sb, &["g"], engine);
        assert_ne!(ga, gb, "{engine}");
        let mut p1 = compile(SWEEP);
        p1.set_inputs(&[3]);
        let want = trace_on(
            &p1.traverse(0).unwrap(),
            &["g"],
            Engine::Interpreter(JitMode::Auto),
        );
        assert_eq!(ga, want, "{engine}");
    }
}

/// The cascade is a snapshot, and a `shared` wire is no exception.
///
/// A body that reads an outer `shared` wire receives the value the cell
/// held when the traversal opened, not the cell itself: the body lowers
/// every cascaded wire as a plain extern
/// ([scope_model.md](../docs/design/scope_model.md) §4,
/// [for_traversal.md](../docs/design/for_traversal.md)). So a write to
/// the cell while a traversal is open reaches no activation of it, and
/// re-opening picks the new value up. Every engine, the same answer —
/// F-K5a recorded this as untested, which it was.
///
/// Capture rather than freeze: nothing about the parent's wire is made
/// read-only, and the parent goes on reading and writing its cell.
#[test]
fn a_body_reading_an_outer_shared_wire_sees_the_value_at_open() {
    const SRC: &str = "input cycle: u64\n\
        shared scale := 10\n\
        for k in 1..3 {\n\
          y := k * scale\n\
        }\n";

    let mut all = vec![Engine::Interpreter(JitMode::Auto)];
    all.extend(engines());
    for engine in all {
        let mut k = compile_polydat_with_engine(SRC, engine, &CompileOptions::default(), None)
            .unwrap_or_else(|e| panic!("{engine}: {e}"));
        k.set_inputs(&[0]);
        let cell = k.shared_cells()[0].cell.clone();

        let y_at = |stream: &TraversalStream, i: usize| -> u64 {
            let mut a = stream.activation(i).unwrap();
            a.kernel.pull_ref("y").as_u64()
        };

        let stream = k.traverse(0).unwrap();
        assert_eq!(y_at(&stream, 0), 10, "{engine}: k=1 at the opening value");

        // Moving the cell mid-traversal reaches no activation of the
        // stream already open, whether one already read or not.
        cell.publish(Value::U64(100));
        assert_eq!(y_at(&stream, 1), 20, "{engine}: k=2 still at the snapshot");
        assert_eq!(y_at(&stream, 0), 10, "{engine}: k=1 re-read, unchanged");
        drop(stream);

        // Re-opening captures the cell as it now stands.
        let stream = k.traverse(0).unwrap();
        assert_eq!(y_at(&stream, 0), 100, "{engine}: k=1 after re-opening");
        assert_eq!(y_at(&stream, 1), 200, "{engine}: k=2 after re-opening");
        drop(stream);

        // And the parent's own read of the wire is the cell, live.
        k.set_inputs(&[0]);
        assert_eq!(k.pull("scale"), Value::U64(100), "{engine}: the parent");
    }
}
