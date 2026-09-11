// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! One kernel API for every engine (engine_parity.md, step 4). A host
//! names an engine, gets a `Box<dyn Kernel>` or one error type, and
//! drives every engine through the same calls: coordinates, externs,
//! cursors, evaluation, typed reads, and sharing across threads. Every
//! engine that accepts a program computes what the interpreter does.

use polydat::ast::Value;
use polydat::dsl::compile::{compile_polydat_to_assembler, compile_polydat_with};
use polydat::{Engine, Kernel, KernelError, Provenance};

/// Every engine and provenance mode a host can name.
fn engines() -> Vec<Engine> {
    let modes = [
        Provenance::Raw,
        Provenance::Push,
        Provenance::Pull,
        Provenance::PushPull,
        Provenance::Auto,
    ];
    let mut all = vec![Engine::Interpreter];
    for m in modes {
        all.push(Engine::Closures(m));
        all.push(Engine::Native(m));
    }
    all
}

/// A program over every input kind: a coordinate, externs with and
/// without a default, a seeded cursor, and outputs of scalar, string,
/// and JSON types.
const SRC: &str = r#"
input cycle: u64
extern scale: u64 = 10
extern region: str = "us-east"
cursor q = range(0, 1000) over "*/4"
h := hash(cycle)
id := mod_wire(h, scale)
key := "{region}/{id}"
n := cardinality(q.cursor)
s := start_of(q.cursor)
doc := json_object(json_with("id", id), json_with("key", key))
"#;

const OUTPUTS: [&str; 6] = ["h", "id", "key", "n", "s", "doc"];

fn partition(idx: usize) -> polydat::iteration::cursor_partition::Partition {
    compile_polydat_to_assembler(SRC).unwrap().cursor_schemas()[0]
        .partitions
        .as_ref()
        .unwrap()[idx]
}

/// Drive one kernel through the trait and return every output per
/// cycle, after the same extern and cursor writes on each.
fn drive(k: &mut dyn Kernel, cycles: &[u64]) -> Vec<Vec<Value>> {
    k.set_input("scale", Value::U64(1000)).unwrap();
    k.set_input("region", Value::Str("eu-west".into())).unwrap();
    k.set_cursor("q", &partition(2)).unwrap();
    cycles
        .iter()
        .map(|&c| {
            k.set_inputs(&[c]);
            OUTPUTS.iter().map(|o| k.pull(o)).collect()
        })
        .collect()
}

fn same(want: &[Vec<Value>], got: &[Vec<Value>], engine: Engine) {
    for (c, (w, g)) in want.iter().zip(got).enumerate() {
        for (i, o) in OUTPUTS.iter().enumerate() {
            assert_eq!(
                w[i].port_type(),
                g[i].port_type(),
                "{engine}: `{o}` at row {c}: type"
            );
            assert_eq!(
                w[i].to_display_string(),
                g[i].to_display_string(),
                "{engine}: `{o}` at row {c}"
            );
        }
    }
}

#[test]
fn every_engine_is_driven_alike_and_agrees_with_the_interpreter() {
    let cycles = [0u64, 1, 7, 250, 999];
    let mut p1 = compile_polydat_with(SRC, Engine::Interpreter).unwrap();
    assert_eq!(p1.engine(), Engine::Interpreter);
    let want = drive(p1.as_mut(), &cycles);
    // The extern writes took: the key names the region the host set.
    assert!(want[0][2].to_display_string().starts_with("eu-west/"));
    for engine in engines() {
        // Every engine runs it: the partition family has no native
        // lowering, and P3 runs those nodes as closures (step 7). A build
        // without native code refuses P3 by name.
        let kernel = compile_polydat_with(SRC, engine);
        if cfg!(not(feature = "jit")) && matches!(engine, Engine::Native(_)) {
            match kernel {
                Err(KernelError::Refused { engine: e, reason }) => {
                    assert_eq!(e, engine);
                    assert!(!reason.is_empty(), "{engine}");
                }
                other => panic!("{engine}: expected a refusal, got {:?}", other.map(|_| ())),
            }
            continue;
        }
        let mut k = kernel.unwrap_or_else(|e| panic!("{engine}: {e}"));
        assert!(matches!(
            (k.engine(), engine),
            (Engine::Interpreter, Engine::Interpreter)
                | (Engine::Closures(_), Engine::Closures(_))
                | (Engine::Native(_), Engine::Native(_))
        ));
        assert_eq!(
            {
                let mut names = k.input_names();
                names.sort();
                names
            },
            {
                let mut names = p1.input_names();
                names.sort();
                names
            },
            "{engine}: inputs"
        );
        assert_eq!(
            k.output_type("key"),
            p1.output_type("key"),
            "{engine}: output type"
        );
        assert_eq!(
            k.externs()
                .iter()
                .map(|(n, _)| n.clone())
                .collect::<std::collections::BTreeSet<_>>(),
            p1.externs()
                .iter()
                .map(|(n, _)| n.clone())
                .collect::<std::collections::BTreeSet<_>>(),
            "{engine}: externs"
        );
        assert_eq!(k.cursor_schemas().len(), 1, "{engine}: cursors");
        let got = drive(k.as_mut(), &cycles);
        same(&want, &got, engine);
    }
}

#[test]
fn a_program_is_shared_across_threads_on_every_engine() {
    let cycles = [3u64, 4, 5];
    let mut p1 = compile_polydat_with(SRC, Engine::Interpreter).unwrap();
    let want = drive(p1.as_mut(), &cycles);
    for engine in engines() {
        let Ok(kernel) = compile_polydat_with(SRC, engine) else {
            continue;
        };
        let program = kernel.into_program();
        assert!(matches!(
            (program.engine(), engine),
            (Engine::Interpreter, Engine::Interpreter)
                | (Engine::Closures(_), Engine::Closures(_))
                | (Engine::Native(_), Engine::Native(_))
        ));
        let results: Vec<Vec<Vec<Value>>> = (0..3)
            .map(|_| {
                let program = std::sync::Arc::clone(&program);
                std::thread::spawn(move || {
                    let mut k = program.create_kernel();
                    drive(k.as_mut(), &cycles)
                })
            })
            .collect::<Vec<_>>()
            .into_iter()
            .map(|h| h.join().unwrap())
            .collect();
        for got in &results {
            same(&want, got, engine);
        }
    }
}

#[test]
fn a_refusal_names_the_engine_and_the_reason() {
    // A vector-typed extern has no compiled passthrough, so every
    // compiled engine refuses it by name.
    let src = "input cycle: u64\nextern v: vec_f32\nout := cycle\n";
    assert!(compile_polydat_with(src, Engine::Interpreter).is_ok());
    for engine in engines().into_iter().skip(1) {
        match compile_polydat_with(src, engine) {
            Err(KernelError::Refused { engine: e, reason }) => {
                assert_eq!(e, engine);
                assert!(
                    reason.contains("no compiled form")
                        || reason.contains("spans")
                        || reason.contains("jit"),
                    "{engine}: {reason}"
                );
                let text = format!("{}", KernelError::Refused { engine: e, reason });
                assert!(
                    text.starts_with("the ") && text.contains("refuses"),
                    "{text}"
                );
            }
            other => panic!("{engine}: expected a refusal, got {:?}", other.map(|_| ())),
        }
    }
    // A source error is the front end's message on every engine.
    for engine in engines() {
        match compile_polydat_with("input cycle: u64\nout := no_such_node(cycle)\n", engine) {
            Err(KernelError::Source(msg)) => {
                assert!(msg.contains("no_such_node"), "{engine}: {msg}")
            }
            other => panic!(
                "{engine}: expected a source error, got {:?}",
                other.map(|_| ())
            ),
        }
    }
}

#[test]
fn the_older_constructors_build_the_same_kernels() {
    let asm = || compile_polydat_to_assembler(SRC).unwrap();
    let mut via_trait = asm()
        .compile_with(Engine::Closures(Provenance::PushPull))
        .unwrap();
    let mut direct = asm().try_compile().unwrap_or_else(|_| panic!("closures"));
    direct.set_input("scale", Value::U64(1000)).unwrap();
    direct
        .set_input("region", Value::Str("eu-west".into()))
        .unwrap();
    direct.set_cursor("q", &partition(2)).unwrap();
    let want = drive(via_trait.as_mut(), &[7]);
    direct.eval(&[7]);
    for (i, o) in OUTPUTS.iter().enumerate() {
        assert_eq!(
            want[0][i].to_display_string(),
            direct.get_value(o).to_display_string(),
            "{o}"
        );
    }
}

#[test]
fn the_compile_log_reaches_every_engine() {
    use polydat::dsl::compile::{CompileOptions, compile_polydat_with_engine};
    for engine in [Engine::Interpreter, Engine::Closures(Provenance::Auto)] {
        let mut log = polydat::dsl::events::CompileEventLog::default();
        compile_polydat_with_engine(SRC, engine, &CompileOptions::default(), Some(&mut log))
            .unwrap_or_else(|e| panic!("{engine}: {e}"));
        assert!(!log.events().is_empty(), "{engine}: no compile events");
    }
}

#[test]
fn the_default_engine_is_compiled_code() {
    let want = if cfg!(feature = "jit") {
        Engine::Native(Provenance::Auto)
    } else {
        Engine::Closures(Provenance::Auto)
    };
    assert_eq!(Engine::default(), want);
    let mut k = polydat::dsl::compile::compile_polydat_kernel(SRC).unwrap();
    assert!(matches!(
        (k.engine(), want),
        (Engine::Native(_), Engine::Native(_)) | (Engine::Closures(_), Engine::Closures(_))
    ));
    let mut p1 = compile_polydat_with(SRC, Engine::Interpreter).unwrap();
    k.set_inputs(&[7]);
    p1.set_inputs(&[7]);
    for name in p1.output_names() {
        assert_eq!(k.pull(&name), p1.pull(&name), "{name}");
    }
    let asm = compile_polydat_to_assembler(SRC).unwrap();
    assert_eq!(asm.compile_kernel().unwrap().engine(), k.engine());
}
