// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! One kernel API for every engine (engines.md §3.5). A host
//! names an engine, gets a `Box<dyn Kernel>` or one error type, and
//! drives every engine through the same calls: coordinates, externs,
//! cursors, evaluation, typed reads, and sharing across threads. Every
//! engine that accepts a program computes what the interpreter does.

use polydat::ast::Value;
use polydat::dsl::compile::{compile_polydat_to_assembler, compile_polydat_with};
use polydat::{Engine, JitMode, Kernel, KernelError, Provenance};

/// Every engine and provenance mode a host can name **and get**.
/// `Native(Push)` is absent because native code has no push-only
/// kernel and the factory refuses the pair rather than substituting
/// push-pull; `a_config_that_cannot_be_realized_is_refused` covers it.
fn engines() -> Vec<Engine> {
    let modes = [
        Provenance::Raw,
        Provenance::Push,
        Provenance::Pull,
        Provenance::PushPull,
        Provenance::Auto,
    ];
    let mut all = vec![Engine::Interpreter(JitMode::Auto)];
    for m in modes {
        all.push(Engine::Closures(m));
        if m != Provenance::Push {
            all.push(Engine::Native(m));
        }
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
    let mut p1 = compile_polydat_with(SRC, Engine::Interpreter(JitMode::Auto)).unwrap();
    assert_eq!(p1.engine(), Engine::Interpreter(JitMode::Auto));
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
            (Engine::Interpreter(_), Engine::Interpreter(_))
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
    let mut p1 = compile_polydat_with(SRC, Engine::Interpreter(JitMode::Auto)).unwrap();
    let want = drive(p1.as_mut(), &cycles);
    for engine in engines() {
        let Ok(kernel) = compile_polydat_with(SRC, engine) else {
            continue;
        };
        let program = kernel.into_program();
        assert!(matches!(
            (program.engine(), engine),
            (Engine::Interpreter(_), Engine::Interpreter(_))
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

/// A node with no compiled form: its output count is decided at
/// construction (`DynamicOutputs`), a shape outside every kit, so
/// every compiled engine refuses a program that uses it.
#[polydat::polydat_node(category = Diagnostic)]
fn interpreter_only_double(
    n: u64,
    widths: polydat::derive_support::Const<Vec<u64>>,
) -> polydat::derive_support::DynamicOutputs<u64> {
    polydat::derive_support::DynamicOutputs(widths.iter().map(|w| n * 2 + w).collect())
}

/// The engine is a preference on the compile options, and leaving it
/// alone is the normative path: a caller that never mentions an engine
/// gets the most compiled form the build has, and can still read back
/// what it got. Naming one is for testing and demonstration.
#[test]
fn a_caller_that_names_no_engine_gets_the_most_compiled_form() {
    use polydat::dsl::compile::{CompileOptions, compile_polydat_kernel_with_options};

    let src = "input cycle: u64\ny := hash(cycle)\n";

    // The default options name no engine.
    let options = CompileOptions::default();
    assert_eq!(options.engine, Engine::default());

    let k = compile_polydat_kernel_with_options(src, &options, None)
        .expect("the default options compile");

    // What it runs is observable, and it is compiled, never the
    // interpreter: native where the build has the `jit` feature, the
    // closure tier where it does not.
    let reported = k.engine();
    if cfg!(feature = "jit") {
        assert!(matches!(reported, Engine::Native(_)), "{reported}");
    } else {
        assert!(matches!(reported, Engine::Closures(_)), "{reported}");
    }

    // The same program with the preference set is the testing path, and
    // it reports the engine that was asked for.
    let named = compile_polydat_kernel_with_options(
        src,
        &CompileOptions {
            engine: Engine::Interpreter(JitMode::Off),
            ..CompileOptions::default()
        },
        None,
    )
    .expect("the interpreter is nameable");
    assert_eq!(named.engine(), Engine::Interpreter(JitMode::Off));
}

/// A kernel reports the configuration it runs, so a configuration the
/// factory cannot realize is refused rather than quietly replaced by a
/// neighbouring one. Native code has no push-only kernel: push-side
/// invalidation without the cone guard has no native form. Asking for
/// it used to build the push-pull kernel, which then reported
/// `Native(PushPull)` to a caller that asked for `Native(Push)`.
#[test]
fn a_config_that_cannot_be_realized_is_refused() {
    let src = "input cycle: u64\ny := hash(cycle)\n";

    match compile_polydat_with(src, Engine::Native(Provenance::Push)) {
        Err(KernelError::Refused { engine, reason }) => {
            assert_eq!(engine, Engine::Native(Provenance::Push));
            assert!(reason.contains("push-only"), "{reason}");
            // The refusal names what the caller can ask for instead.
            assert!(
                reason.contains("pushpull") && reason.contains("auto"),
                "{reason}"
            );
        }
        other => panic!("expected a refusal, got {:?}", other.map(|k| k.engine())),
    }

    // Push alone is realizable on the closure tier, so it is not refused
    // there: the refusal is about this engine's forms, not the mode.
    let k = compile_polydat_with(src, Engine::Closures(Provenance::Push))
        .expect("closures has a push kernel");
    assert_eq!(k.engine(), Engine::Closures(Provenance::Push));

    // Every pair the factory accepts reports back what was asked for,
    // except `Auto`, which is a request for the factory to choose and
    // reports the choice.
    for engine in engines() {
        let k = compile_polydat_with(src, engine)
            .unwrap_or_else(|e| panic!("{engine} should build: {e}"));
        let reported = k.engine();
        match engine {
            Engine::Closures(Provenance::Auto) => {
                assert!(matches!(reported, Engine::Closures(_)), "{reported}")
            }
            Engine::Native(Provenance::Auto) => {
                assert!(matches!(reported, Engine::Native(_)), "{reported}")
            }
            named => assert_eq!(reported, named, "asked {named}, got {reported}"),
        }
    }
}

#[test]
fn a_refusal_names_the_engine_and_the_reason() {
    // A node with no compiled form is refused by every compiled engine,
    // by name.
    let src = "input cycle: u64\n(a, b) := interpreter_only_double(cycle, 1, 2)\n";
    assert!(compile_polydat_with(src, Engine::Interpreter(JitMode::Auto)).is_ok());
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
fn the_compile_log_is_the_same_on_every_engine() {
    use polydat::dsl::compile::{CompileOptions, compile_polydat_with_engine};
    // Tile typings, an extern without a default, and two constants to
    // fold: a literal and a node no input reaches.
    let src = "input cycle: u64\nextern doc: json\nh := hash(cycle)\nf := to_f64(h) / 3.0\ns := \"n-{h}\"\nc := u64_add(2, 3)\ntile t : json := {\"h\": ${h}, \"f\": ${f | .2}, \"s\": ${s}, \"c\": ${c}}\n";
    let events = |engine: Engine| {
        let mut log = polydat::dsl::events::CompileEventLog::default();
        compile_polydat_with_engine(src, engine, &CompileOptions::default(), Some(&mut log))
            .unwrap_or_else(|e| panic!("{engine}: {e}"));
        log.events()
            .iter()
            .map(|e| format!("{e:?}"))
            .collect::<Vec<_>>()
    };
    let want = events(Engine::Interpreter(JitMode::Auto));
    assert!(
        want.iter().any(|e| e.starts_with("ConstantFolded")),
        "{want:?}"
    );
    assert!(
        want.iter().any(|e| e.starts_with("ExternWithoutDefault")),
        "{want:?}"
    );
    for engine in [
        Engine::Closures(Provenance::Auto),
        Engine::Native(Provenance::Auto),
    ] {
        if cfg!(not(feature = "jit")) && matches!(engine, Engine::Native(_)) {
            continue;
        }
        assert_eq!(events(engine), want, "{engine}");
    }
}

#[test]
fn invalidate_all_reruns_a_cycle_whose_inputs_did_not_move() {
    // A side channel fires once per pull of a fresh cycle; with nothing
    // moved it is current and silent, and `invalidate_all` runs it again.
    let src = "input cycle: u64\nextern tag: str = \"t\"\n__emit := emit_row(\"map\", \"cycle,tag\", cycle, tag)\n";
    for engine in [
        Engine::Interpreter(JitMode::Auto),
        Engine::Closures(Provenance::Auto),
        Engine::Native(Provenance::Auto),
    ] {
        let Ok(mut k) = compile_polydat_with(src, engine) else {
            continue;
        };
        polydat::library::emit::take_rows();
        k.set_inputs(&[3]);
        k.pull("__emit");
        k.set_inputs(&[3]);
        k.pull("__emit");
        let quiet = polydat::library::emit::take_rows().len();
        k.invalidate_all();
        k.pull("__emit");
        k.invalidate_all();
        k.pull("__emit");
        let rows = polydat::library::emit::take_rows();
        assert_eq!(rows.len(), 2, "{engine}: {rows:?} after {quiet} quiet");
        assert!(
            rows.iter().all(|r| r == "cycle=3 tag=t"),
            "{engine}: {rows:?}"
        );
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
    let mut p1 = compile_polydat_with(SRC, Engine::Interpreter(JitMode::Auto)).unwrap();
    k.set_inputs(&[7]);
    p1.set_inputs(&[7]);
    for name in p1.output_names() {
        assert_eq!(k.pull(&name), p1.pull(&name), "{name}");
    }
    let asm = compile_polydat_to_assembler(SRC).unwrap();
    assert_eq!(asm.compile_kernel().unwrap().engine(), k.engine());
}

#[test]
fn outputs_are_listed_in_declaration_order_on_every_engine() {
    let p1 = compile_polydat_with(SRC, Engine::Interpreter(JitMode::Auto)).unwrap();
    for engine in [
        Engine::Closures(Provenance::Auto),
        Engine::Native(Provenance::Auto),
        Engine::default(),
    ] {
        let Ok(k) = compile_polydat_with(SRC, engine) else {
            continue;
        };
        assert_eq!(k.output_names(), p1.output_names(), "{engine}");
    }
}

#[test]
fn the_default_engine_forms_take_options_tiles_and_activations() {
    use polydat::dsl::compile::{
        CompileOptions, compile_polydat_kernel, compile_polydat_kernel_with_options,
        compile_polydat_kernel_with_tiles,
    };
    use polydat::dsl::events::CompileEventLog;
    use polydat::tile::{Span, TileOptions, tile_from_text};

    // Options: the outputs to keep and the compile log.
    let mut log = CompileEventLog::new();
    let options = CompileOptions {
        required_outputs: vec!["key".into()],
        context: "kernel api".into(),
        ..CompileOptions::default()
    };
    let mut k = compile_polydat_kernel_with_options(SRC, &options, Some(&mut log)).unwrap();
    // The same pruning the interpreter path applies: `h`, `id`, and `n`
    // are gone, `key` and the declared surface remain.
    let mut p1 = polydat::dsl::compile_polydat_with_options(SRC, &options, None).unwrap();
    assert_eq!(k.output_names(), p1.output_names());
    assert!(
        !k.output_names()
            .iter()
            .any(|n| n == "h" || n == "id" || n == "n")
    );
    assert!(!log.events().is_empty());
    k.set_inputs(&[3]);
    p1.set_inputs(&[3]);
    assert_eq!(k.pull("key"), *p1.pull("key"));
    assert_eq!(k.engine(), compile_polydat_kernel(SRC).unwrap().engine());

    // Tiles from host data.
    let tile = tile_from_text(
        "doc",
        "json",
        r#"{"n": ${cycle}, "twice": ${cycle * 2}}"#,
        &TileOptions::default(),
        Span { line: 0, col: 0 },
    )
    .unwrap();
    let mut k = compile_polydat_kernel_with_tiles("input cycle: u64\n", vec![tile]).unwrap();
    k.set_inputs(&[4]);
    assert_eq!(k.pull("doc").as_str(), r#"{"n": 4, "twice": 8}"#);

    // Activations on the default engine.
    let mut k = compile_polydat_kernel(
        "input cycle: u64\nfor a in 1..4 {\n    x := u64_add(a, cycle)\n}\n",
    )
    .unwrap();
    k.set_inputs(&[10]);
    let stream = k.traverse(0).unwrap();
    for index in 0..stream.len() {
        let mut act = stream.activate(index).unwrap();
        // The same tier as the root; the selector picks each body's provenance.
        assert!(
            matches!(
                (act.kernel.engine(), k.engine()),
                (Engine::Native(_), Engine::Native(_)) | (Engine::Closures(_), Engine::Closures(_))
            ),
            "activation {index}: {} vs {}",
            act.kernel.engine(),
            k.engine()
        );
        let mut want = stream.activation(index).unwrap();
        assert_eq!(act.cycle(0).pull("x"), *want.cycle(0).pull("x"));
    }
}

#[test]
fn every_engine_reports_its_plan() {
    let p1 = compile_polydat_with(SRC, Engine::Interpreter(JitMode::Auto)).unwrap();
    let nodes = polydat::dsl::compile_polydat_interpreter(SRC)
        .unwrap()
        .program()
        .node_count();
    let plan = p1.plan();
    assert_eq!(plan.closure_steps, 0);
    assert_eq!(plan.native_segments + plan.interpreted_nodes, nodes);
    let p2 = compile_polydat_with(SRC, Engine::Closures(Provenance::Auto)).unwrap();
    let plan = p2.plan();
    assert!(plan.closure_steps > 0, "{plan}");
    assert_eq!((plan.native_segments, plan.interpreted_nodes), (0, 0));
    assert_eq!(
        plan.to_string(),
        format!("{} closure step(s)", plan.closure_steps)
    );
    if let Ok(p3) = compile_polydat_with(SRC, Engine::Native(Provenance::Auto)) {
        let plan = p3.plan();
        assert!(plan.native_segments > 0, "{plan}");
        assert_eq!(plan.interpreted_nodes, 0);
        assert!(
            plan.to_string()
                .starts_with(&format!("{} native segment(s)", plan.native_segments))
        );
    }
    assert_eq!(polydat::EnginePlan::default().to_string(), "nothing");
}

#[test]
fn the_index_keyed_calls_agree_with_the_named_ones_on_every_engine() {
    // SRD 117 step 3: a host that binds the same inputs and reads the
    // same outputs every cycle resolves the names once.
    for engine in [
        Engine::Interpreter(JitMode::Auto),
        Engine::Closures(Provenance::Auto),
        Engine::Native(Provenance::Auto),
    ] {
        let Ok(mut by_name) = compile_polydat_with(SRC, engine) else {
            continue;
        };
        let mut by_index = compile_polydat_with(SRC, engine).unwrap();
        let scale = by_index.input_index("scale").unwrap();
        let region = by_index.input_index("region").unwrap();
        assert_eq!(by_index.input_index("cycle"), Some(0), "{engine}");
        assert_eq!(by_index.input_index("nope"), None, "{engine}");
        let key = by_index.output_index("key").unwrap();
        let n = by_index.output_index("n").unwrap();
        assert_eq!(by_index.output_index("nope"), None, "{engine}");
        assert_eq!(
            by_index.output_names()[key],
            "key",
            "{engine}: index among output_names"
        );
        for cycle in 0..4u64 {
            by_name.set_inputs(&[cycle]);
            by_name.set_input("scale", Value::U64(100 + cycle)).unwrap();
            by_name
                .set_input("region", Value::Str(format!("r{cycle}").into()))
                .unwrap();
            by_index.set_inputs(&[cycle]);
            by_index
                .set_input_at(scale, Value::U64(100 + cycle))
                .unwrap();
            by_index
                .set_input_at(region, Value::Str(format!("r{cycle}").into()))
                .unwrap();
            assert_eq!(
                by_index.pull_at(key),
                by_name.pull("key"),
                "{engine} {cycle}"
            );
            assert_eq!(by_index.pull_at(n), by_name.pull("n"), "{engine} {cycle}");
        }
        // A coordinate is not set by index through an extern write, and
        // the wrong type is refused by name.
        assert!(by_index.set_input_at(0, Value::U64(1)).is_err(), "{engine}");
        let err = by_index.set_input_at(scale, Value::Str("x".into()));
        assert!(err.is_err(), "{engine}: {err:?}");
    }
}

/// A kernel created from a program starts from the program on every
/// engine: the externs at their declared defaults, whatever the kernel
/// that became the program had been set to. The interpreter always
/// built its created kernels that way; the compiled engines cloned the
/// set values along.
#[test]
fn a_created_kernel_starts_from_the_programs_defaults_on_every_engine() {
    for engine in engines() {
        let Ok(mut kernel) = compile_polydat_with(SRC, engine) else {
            continue;
        };
        kernel.set_input("scale", Value::U64(1000)).unwrap();
        kernel
            .set_input("region", Value::Str("eu-west".into()))
            .unwrap();
        kernel.set_inputs(&[3]);
        let set_key = kernel.pull("key");
        let program = kernel.into_program();
        let mut fresh = program.create_kernel();
        assert_eq!(fresh.input_value("scale"), Some(Value::U64(10)), "{engine}");
        assert_eq!(
            fresh.input_value("region"),
            Some(Value::Str("us-east".into())),
            "{engine}"
        );
        fresh.set_inputs(&[3]);
        let fresh_key = fresh.pull("key");
        assert_ne!(set_key, fresh_key, "{engine}");
        assert!(
            fresh_key.as_str().starts_with("us-east/"),
            "{engine}: {fresh_key:?}"
        );
    }
}

/// A program whose inputs span more than sixty-four slots builds and
/// runs on every engine, and an extern past the sixty-fourth slot still
/// invalidates what it reaches. The P3 kernel's provenance mask was one
/// word, and shifted out of range at build for such a program.
#[test]
fn a_program_with_many_inputs_runs_on_every_engine() {
    let count = 70;
    let mut src = String::from("input cycle: u64\n");
    for i in 0..count {
        src.push_str(&format!("extern e{i}: u64 = {i}\n"));
    }
    src.push_str(&format!("s := e0 + e{} + cycle\n", count - 1));
    for engine in engines() {
        let mut k = match compile_polydat_with(&src, engine) {
            Ok(k) => k,
            Err(KernelError::Refused { .. }) => continue,
            Err(e) => panic!("{engine}: {e}"),
        };
        k.set_inputs(&[1]);
        assert_eq!(k.pull("s"), Value::U64(70), "{engine}");
        k.set_input(&format!("e{}", count - 1), Value::U64(1000))
            .unwrap();
        k.set_inputs(&[1]);
        assert_eq!(k.pull("s"), Value::U64(1001), "{engine}");
        // The same coordinates again: the guard finds the cone clean.
        k.set_inputs(&[1]);
        assert_eq!(k.pull("s"), Value::U64(1001), "{engine}");
        k.set_inputs(&[2]);
        assert_eq!(k.pull("s"), Value::U64(1002), "{engine}");
    }
}

/// One write rule on every engine: an extern takes a value of its
/// declared type, a bit-stuffed carrier form of it, or `None`, and
/// refuses anything else with the same message; a coordinate is set
/// with `set_inputs`, never as an extern.
#[test]
fn the_write_rule_is_the_same_on_every_engine() {
    let src = "input cycle: u64\nextern n: u64 = 1\nextern f: f64 = 1.0\nextern s: str = \"a\"\nout := to_f64(n) + f\n";
    let mut messages: Vec<(Engine, String, String)> = Vec::new();
    for engine in engines() {
        let mut k = match compile_polydat_with(src, engine) {
            Ok(k) => k,
            Err(KernelError::Refused { .. }) => continue,
            Err(e) => panic!("{engine}: {e}"),
        };
        assert!(k.set_input("n", Value::U64(5)).is_ok(), "{engine}");
        assert!(k.set_input("f", Value::F64(2.5)).is_ok(), "{engine}");
        assert!(k.set_input("s", Value::Str("b".into())).is_ok(), "{engine}");
        let wrong = k.set_input("f", Value::Str("x".into())).unwrap_err();
        let coordinate = k.set_input("cycle", Value::U64(1)).unwrap_err();
        messages.push((engine, wrong, coordinate));
        k.set_inputs(&[3]);
        assert_eq!(k.pull("out"), Value::F64(7.5), "{engine}");
    }
    let (_, wrong, coordinate) = &messages[0];
    for (engine, w, c) in &messages {
        assert_eq!(w, wrong, "{engine}");
        assert_eq!(c, coordinate, "{engine}");
    }
}
