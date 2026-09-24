// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! The node-by-engine matrix of engine equivalence (docs/design/engines.md
//! §7), pinned.
//!
//! Every registered public node is compiled and run through one cycle
//! on each engine from the same program the coverage test uses, and the
//! outcome per engine (`ok`, `refused` at compile, `failed` at run) is
//! compared with the checked-in table `tests/engine_parity.txt`. Any
//! change in either direction fails the test: a regression on an engine
//! that accepted a node, and a step of the true-up plan that lands
//! without recording its gain. Regenerate the table after an intended
//! change with `ENGINE_PARITY=overwrite cargo test --test suite engine_parity::`
//! and review the diff.
//!
//! The matrix needs every engine, so it runs with the `jit` feature.

#![cfg(feature = "jit")]

use super::common;

use polydat::dsl::compile::compile_polydat_to_assembler;
use polydat::dsl::compile::compile_polydat_with;
use polydat::{Engine, JitMode, Kernel, Provenance};
use std::path::Path;

/// The text of a panic payload.
fn payload_text(p: Box<dyn std::any::Any + Send>) -> String {
    p.downcast_ref::<String>()
        .cloned()
        .or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_default()
}

/// A failure message without its `panicked at` line, which names the
/// code that raised the panic and so differs by engine.
fn without_location(msg: &str) -> String {
    msg.lines()
        .filter(|l| !l.trim_start().starts_with("↳ panicked at"))
        .collect::<Vec<_>>()
        .join("\n")
}

/// One engine's outcome for one program.
fn outcome<T>(r: std::thread::Result<Result<T, String>>) -> (&'static str, String) {
    match r {
        Ok(Ok(_)) => ("ok", String::new()),
        Ok(Err(e)) => ("refused", e.lines().next().unwrap_or("").to_string()),
        Err(p) => {
            let msg = payload_text(p);
            ("failed", msg.lines().next().unwrap_or("").to_string())
        }
    }
}

/// The four outcomes for one program: interpreter, closure tier, P3
/// (native code where a node lowers, closures elsewhere), and pure
/// native code, the differential tier behind P3. Each engine compiles, runs one cycle, and
/// reads every output, as a host would.
fn row(src: &str) -> [(&'static str, String); 4] {
    let p1 = outcome(std::panic::catch_unwind(|| {
        let mut asm = compile_polydat_to_assembler(src).map_err(|e| e.to_string())?;
        asm.set_jit_mode(polydat::JitMode::Off);
        let mut k = asm.compile().map_err(|e| e.to_string())?;
        k.set_inputs(&[3]);
        let outs: Vec<String> = k.output_names().iter().map(|s| s.to_string()).collect();
        for o in &outs {
            let _ = k.pull_ref(o);
        }
        Ok::<(), String>(())
    }));
    let p2 = outcome(std::panic::catch_unwind(|| {
        let mut k = compile_polydat_to_assembler(src)
            .map_err(|e| e.to_string())?
            .compile_slots(polydat::Engine::Closures(polydat::Provenance::Raw))
            .map_err(|_| "no closure form for some node".to_string())?;
        k.eval_at(&[3]);
        let outs: Vec<String> = k.output_names().iter().map(|s| s.to_string()).collect();
        for o in &outs {
            let _ = k.get_value(o);
        }
        Ok::<(), String>(())
    }));
    let p3 = outcome(std::panic::catch_unwind(|| {
        let mut k = compile_polydat_to_assembler(src)
            .map_err(|e| e.to_string())?
            .compile_slots(polydat::Engine::Native(polydat::Provenance::PushPull))
            .map_err(|e| e.to_string())?;
        k.eval_at(&[3]);
        let outs: Vec<String> = k.output_names().iter().map(|s| s.to_string()).collect();
        for o in &outs {
            let _ = k.get_value(o);
        }
        Ok::<(), String>(())
    }));
    let pure = outcome(std::panic::catch_unwind(|| {
        let mut k = compile_polydat_to_assembler(src)
            .map_err(|e| e.to_string())?
            .compile_slots(polydat::Engine::PureNative(polydat::Provenance::PushPull))
            .map_err(|e| e.to_string())?;
        k.eval_at(&[3]);
        let outs: Vec<String> = k.output_names().iter().map(|s| s.to_string()).collect();
        for o in &outs {
            let _ = k.get_value(o);
        }
        Ok::<(), String>(())
    }));
    [p1, p2, p3, pure]
}

#[test]
fn the_node_by_engine_matrix_is_as_recorded() {
    let table = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/engine_parity.txt");
    let mut lines = vec![
        "# node	P1	P2	P3	pure — outcomes per engine (pure: native code alone, the differential tier); regenerate with ENGINE_PARITY=overwrite"
            .to_string(),
    ];
    let mut detail = Vec::new();
    let trace = std::env::var("ENGINE_PARITY_TRACE").is_ok();
    for (name, src) in common::coverage_cases::programs() {
        if trace {
            eprintln!("engine_parity: {name}");
        }
        let r = row(&src);
        lines.push(format!(
            "{name}\t{}\t{}\t{}\t{}",
            r[0].0, r[1].0, r[2].0, r[3].0
        ));
        for (engine, (k, msg)) in ["P1", "P2", "P3", "pure"].iter().zip(r.iter()) {
            if *k != "ok" {
                detail.push(format!("  {name} on {engine}: {k}: {msg}"));
            }
        }
    }
    if trace {
        eprintln!("{}", detail.join("\n"));
    }
    let current = lines.join("\n") + "\n";
    let overwrite = std::env::var("ENGINE_PARITY").as_deref() == Ok("overwrite");
    // The same matrix, rendered into the node reference, so the documented
    // feature set and the tested one are one file (engine parity, step 10).
    check_reference_section(&lines, overwrite);
    if overwrite {
        std::fs::write(&table, &current).unwrap();
        return;
    }
    let recorded = std::fs::read_to_string(&table)
        .unwrap_or_default()
        .replace("\r\n", "\n");
    if recorded != current {
        let old: std::collections::BTreeSet<&str> = recorded.lines().collect();
        let new: std::collections::BTreeSet<&str> = current.lines().collect();
        let gone: Vec<&&str> = old.difference(&new).collect();
        let added: Vec<&&str> = new.difference(&old).collect();
        panic!(
            "the node-by-engine matrix changed.\n\nrecorded but no longer so:\n{}\n\nnow, not recorded:\n{}\n\nrefusals and failures now:\n{}\n\nIf the change is intended, regenerate with ENGINE_PARITY=overwrite; a new refusal needs its reason in docs/design/engines.md §8.",
            gone.iter()
                .map(|l| format!("  {l}"))
                .collect::<Vec<_>>()
                .join("\n"),
            added
                .iter()
                .map(|l| format!("  {l}"))
                .collect::<Vec<_>>()
                .join("\n"),
            detail.join("\n"),
        );
    }
}

/// Every output of every program, on every engine that ran it, as the
/// interpreter computed it. The matrix above pins what each engine
/// accepts; this pins that what it accepts it computes alike. Nodes
/// declared nondeterministic (clocks, entropy, thread identity) are
/// left out; a run failure the interpreter reports must read the same on
/// every engine that fails too (step 6, A7).
#[test]
fn the_engines_agree_on_every_node() {
    use polydat::ast::{Purity, Value};
    let cycles: [u64; 3] = [3, 4, 11];
    let mut disagreements = Vec::new();
    for (name, src) in common::coverage_cases::programs() {
        let Ok(mut asm) = compile_polydat_to_assembler(&src) else {
            continue;
        };
        asm.set_jit_mode(polydat::JitMode::Off);
        let Ok(mut p1) = asm.compile() else {
            continue;
        };
        let program = p1.program();
        if (0..program.node_count()).any(|i| {
            matches!(
                program.node_ref(i).purity(),
                Purity::Nondeterministic { .. }
            )
        }) {
            continue;
        }
        let outs: Vec<String> = p1.output_names().iter().map(|s| s.to_string()).collect();
        let mut p2 = compile_polydat_to_assembler(&src)
            .unwrap()
            .compile_slots(polydat::Engine::Closures(polydat::Provenance::Raw))
            .ok();
        let mut p3 = compile_polydat_to_assembler(&src)
            .unwrap()
            .compile_slots(polydat::Engine::Native(polydat::Provenance::PushPull))
            .ok();
        let mut pure = compile_polydat_to_assembler(&src)
            .unwrap()
            .compile_slots(polydat::Engine::PureNative(polydat::Provenance::PushPull))
            .ok();
        for &c in &cycles {
            // An assertion node fails on some cycles by design; the
            // interpreter's failure is the oracle's, and an engine that
            // fails too must fail with the same message, location
            // aside (the matrix pins the failures at cycle 3).
            let want = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                p1.set_inputs(&[c]);
                outs.iter()
                    .map(|o| p1.pull_ref(o).clone())
                    .collect::<Vec<Value>>()
            }))
            .map_err(payload_text);
            let mut got: Vec<(&str, Result<Vec<Value>, String>)> = Vec::new();
            if let Some(k) = p2.as_mut() {
                let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    k.eval_at(&[c]);
                    outs.iter().map(|o| k.get_value(o)).collect::<Vec<_>>()
                }));
                got.push(("P2", r.map_err(payload_text)));
            }
            if let Some(k) = p3.as_mut() {
                let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    k.eval_at(&[c]);
                    outs.iter().map(|o| k.get_value(o)).collect::<Vec<_>>()
                }));
                got.push(("P3", r.map_err(payload_text)));
            }
            if let Some(k) = pure.as_mut() {
                let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    k.eval_at(&[c]);
                    outs.iter().map(|o| k.get_value(o)).collect::<Vec<_>>()
                }));
                got.push(("pure", r.map_err(payload_text)));
            }
            for (engine, result) in got {
                match (&want, result) {
                    (Ok(want), Ok(values)) => {
                        for (i, o) in outs.iter().enumerate() {
                            let (w, g) = (&want[i], &values[i]);
                            if w.port_type() != g.port_type()
                                || w.to_display_string() != g.to_display_string()
                            {
                                disagreements.push(format!(
                                    "  {name} on {engine}: `{o}` at cycle {c}: interpreter {:?} {}, \
                                     {engine} {:?} {}",
                                    w.port_type(),
                                    w.to_display_string(),
                                    g.port_type(),
                                    g.to_display_string()
                                ));
                            }
                        }
                    }
                    (Err(want), Err(msg)) => {
                        if without_location(&msg) != without_location(want) {
                            disagreements.push(format!(
                                "  {name} on {engine} at cycle {c}: the failure reads \
                                 differently\n    interpreter: {}\n    {engine}: {}",
                                want.replace('\n', "\n    "),
                                msg.replace('\n', "\n    ")
                            ));
                        }
                    }
                    // What runs where is the matrix's to pin.
                    (Ok(_), Err(_)) | (Err(_), Ok(_)) => {}
                }
            }
        }
    }
    assert!(
        disagreements.is_empty(),
        "engines disagree with the interpreter:\n{}",
        disagreements.join("\n")
    );
}

/// The interpreter accepts every program; that is the oracle the plan
/// measures the other engines against.
/// A nondeterministic node and a side channel run on pure native code
/// as they run on every other engine: the never-current step reruns
/// after every write, so a counter advances once per write on each
/// engine, and a side channel fires once per evaluation in which it
/// is not current, so the rows it emits agree in number when every
/// engine is driven through the same writes and evaluations.
#[cfg(feature = "jit")]
#[test]
fn nondeterministic_and_side_channel_nodes_run_alike_on_pure_native_code() {
    let src = "input cycle: u64\n\
        n := counter()\n\
        h := hash(cycle)\n\
        line := \"{n}:{h}\"\n\
        rows := emit_row(\"csv\", \"n,h\", n, h)\n";
    let mut kernels: Vec<(&str, Box<dyn Kernel>)> = vec![
        (
            "interpreter",
            compile_polydat_with(src, Engine::Interpreter(JitMode::Off)).unwrap(),
        ),
        (
            "closures",
            compile_polydat_with(src, Engine::Closures(Provenance::PushPull)).unwrap(),
        ),
        (
            "native",
            compile_polydat_with(src, Engine::Native(Provenance::PushPull)).unwrap(),
        ),
        (
            "pure native",
            // Already a box, and `SlotKernel` extends `Kernel`, so it
            // upcasts here rather than needing a second one.
            compile_polydat_to_assembler(src)
                .unwrap()
                .compile_slots(polydat::Engine::PureNative(polydat::Provenance::PushPull))
                .expect("pure native code runs a counter and a side channel"),
        ),
    ];
    let _ = polydat::library::emit::take_rows();
    // The same writes on every engine, repeats included: a write is a
    // write to a never-current step.
    let writes = [1u64, 2, 2, 3, 3, 3, 4];
    let mut counts: Vec<(String, Vec<u64>, usize)> = Vec::new();
    for (name, k) in kernels.iter_mut() {
        let _ = polydat::library::emit::take_rows();
        let mut seen = Vec::new();
        for &c in &writes {
            k.set_inputs(&[c]);
            k.eval();
            seen.push(k.pull("n").as_u64());
            assert_eq!(
                k.pull("line").as_str(),
                format!("{}:{}", seen.last().unwrap(), k.pull("h").as_u64()),
                "{name}: the line reads the counter of this write"
            );
        }
        counts.push((
            name.to_string(),
            seen,
            polydat::library::emit::take_rows().len(),
        ));
    }
    let (_, want_seen, want_rows) = &counts[0];
    assert_eq!(
        want_seen,
        &[0u64, 1, 2, 3, 4, 5, 6],
        "the interpreter counts every write"
    );
    for (name, seen, rows) in &counts[1..] {
        assert_eq!(
            seen, want_seen,
            "{name}: the counter advances once per write"
        );
        assert_eq!(
            rows, want_rows,
            "{name}: the side channel fires once per evaluation"
        );
    }
}

#[test]
fn the_interpreter_accepts_every_node() {
    let mut failures = Vec::new();
    for (name, src) in common::coverage_cases::programs() {
        let (k, msg) = row(&src)[0].clone();
        if k != "ok" {
            failures.push(format!("  {name}: {k}: {msg}"));
        }
    }
    assert!(
        failures.is_empty(),
        "the interpreter refused or failed:\n{}",
        failures.join("\n")
    );
}

const REFERENCE_START: &str = "<!-- engine matrix: generated by tests/engine_parity.rs; regenerate with ENGINE_PARITY=overwrite -->";
const REFERENCE_END: &str = "<!-- /engine matrix -->";

/// The matrix as the node reference states it: which nodes every engine
/// runs, and which the pure native tier refuses. Compared against
/// `docs/reference/nodes.md` between the markers, or written there.
fn check_reference_section(lines: &[String], overwrite: bool) {
    let mut refused_by_engine: std::collections::BTreeMap<&str, Vec<&str>> = Default::default();
    let mut pure_refused: Vec<&str> = Vec::new();
    let mut total = 0usize;
    for line in lines.iter().skip(1) {
        let cols: Vec<&str> = line.split('\t').collect();
        total += 1;
        for (engine, col) in ["the interpreter", "the closure tier", "P3"]
            .iter()
            .zip(&cols[1..4])
        {
            if *col != "ok" {
                refused_by_engine.entry(engine).or_default().push(cols[0]);
            }
        }
        if cols[4] != "ok" {
            pure_refused.push(cols[0]);
        }
    }
    // Listed by name, whatever order the cases ran in.
    for names in refused_by_engine.values_mut() {
        names.sort_unstable();
        names.dedup();
    }
    pure_refused.sort_unstable();
    pure_refused.dedup();
    let list = |names: &[&str]| {
        names
            .iter()
            .map(|n| format!("`{n}`"))
            .collect::<Vec<_>>()
            .join(", ")
    };
    let mut section = String::new();
    section.push_str(&format!(
        "\nThe engine parity suite drives {total} coverage programs, one per node \
         or node form, through every engine a host can choose.\n"
    ));
    if refused_by_engine.is_empty() {
        section.push_str(
            "Every one of them runs on the interpreter, the closure tier, and P3 \
             (native code where a node has a lowering, its closure elsewhere).\n",
        );
    } else {
        for (engine, names) in &refused_by_engine {
            section.push_str(&format!("Not on {engine}: {}.\n", list(names)));
        }
    }
    if pure_refused.is_empty() {
        section.push_str(
            "\nPure native code, the differential tier behind P3, runs every one of \
             them as well.\n\n",
        );
    } else {
        section.push_str(&format!(
            "\nPure native code, the differential tier behind P3, has no lowering for \
             these {} nodes, which P3 runs as closure steps:\n\n{}.\n\n",
            pure_refused.len(),
            list(&pure_refused)
        ));
    }
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("docs/reference/nodes.md");
    let doc = std::fs::read_to_string(&path)
        .unwrap_or_default()
        .replace("\r\n", "\n");
    let (Some(start), Some(end)) = (doc.find(REFERENCE_START), doc.find(REFERENCE_END)) else {
        panic!("docs/reference/nodes.md lacks the engine matrix markers");
    };
    let inner_start = start + REFERENCE_START.len();
    if overwrite {
        let updated = format!("{}{}{}", &doc[..inner_start], section, &doc[end..]);
        std::fs::write(&path, updated).unwrap();
        return;
    }
    let recorded = &doc[inner_start..end];
    assert_eq!(
        recorded, section,
        "the engine matrix in docs/reference/nodes.md is not the tested one; regenerate with ENGINE_PARITY=overwrite"
    );
}

/// The 128-bit carriers on every engine (F-C16).
///
/// `U128`/`I128` are the one scalar family that cannot ride a single
/// slot, so they cross the compiled tiers as a limb pair and have no
/// named native lowering. That made them easy to describe as
/// interpreter-only and easy to leave untested: the matrix above is
/// built from DSL programs, and these adapters are `__`-prefixed and
/// assembler-inserted, so no program in it reaches one.
///
/// This builds the widening and every projection back out of the
/// carrier through the programmatic assembler, and asserts the four
/// engines return the same `Value` for each.
#[test]
fn the_128_bit_carriers_agree_on_every_engine() {
    use polydat::ast::Value;
    use polydat::compile::assembly::{PolydatAssembler, WireRef};
    use polydat::library::polyfill_128 as W;

    // One program per family: widen the cycle into the carrier, then
    // read it back out through each projection the catalog has.
    let build = |signed: bool| {
        let mut asm = PolydatAssembler::new(vec!["cycle".into()]);
        if signed {
            asm.add_node(
                "wide",
                Box::new(W::U64ToI128::new()),
                vec![WireRef::input("cycle")],
            );
            asm.add_node(
                "back",
                Box::new(W::I128ToI64::new()),
                vec![WireRef::node("wide")],
            );
            asm.add_node(
                "f",
                Box::new(W::I128ToF64::new()),
                vec![WireRef::node("wide")],
            );
            asm.add_node(
                "s",
                Box::new(W::I128ToString::new()),
                vec![WireRef::node("wide")],
            );
            asm.add_node(
                "b",
                Box::new(W::I128ToBytes::new()),
                vec![WireRef::node("wide")],
            );
            asm.add_node(
                "j",
                Box::new(W::I128ToJson::new()),
                vec![WireRef::node("wide")],
            );
            asm.add_node(
                "t",
                Box::new(W::I128ToBool::new()),
                vec![WireRef::node("wide")],
            );
        } else {
            asm.add_node(
                "wide",
                Box::new(W::U64ToU128::new()),
                vec![WireRef::input("cycle")],
            );
            asm.add_node(
                "back",
                Box::new(W::U128ToU64::new()),
                vec![WireRef::node("wide")],
            );
            asm.add_node(
                "f",
                Box::new(W::U128ToF64::new()),
                vec![WireRef::node("wide")],
            );
            asm.add_node(
                "s",
                Box::new(W::U128ToString::new()),
                vec![WireRef::node("wide")],
            );
            asm.add_node(
                "b",
                Box::new(W::U128ToBytes::new()),
                vec![WireRef::node("wide")],
            );
            asm.add_node(
                "j",
                Box::new(W::U128ToJson::new()),
                vec![WireRef::node("wide")],
            );
            asm.add_node(
                "t",
                Box::new(W::U128ToBool::new()),
                vec![WireRef::node("wide")],
            );
        }
        for out in ["wide", "back", "f", "s", "b", "j", "t"] {
            asm.add_output(out, WireRef::node(out));
        }
        asm
    };
    let outs = ["wide", "back", "f", "s", "b", "j", "t"];
    // Which tiers built the program at all, so the comparison below
    // cannot pass by having been skipped.
    let mut built: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();

    for signed in [false, true] {
        // The signed family narrows back to `i64`, whose range the
        // top of the `u64` range is outside of, so that value belongs
        // to the unsigned case.
        let cycles: &[u64] = if signed {
            &[0, 1, 7, i64::MAX as u64]
        } else {
            &[0, 1, 7, u64::MAX]
        };
        for &cycle in cycles {
            let mut p1 = {
                let mut asm = build(signed);
                asm.set_jit_mode(JitMode::Off);
                asm.compile().expect("the interpreter builds it")
            };
            p1.set_inputs(&[cycle]);
            let want: Vec<Value> = outs.iter().map(|o| p1.pull(o)).collect();
            // The round trip and the carrier itself, so the test also
            // states what the value is rather than only that the
            // engines agree about it.
            let back = if signed {
                want[1].as_i64() as u64
            } else {
                want[1].as_u64()
            };
            assert_eq!(back, cycle, "round trip, signed={signed}");

            for engine in [
                Engine::Closures(Provenance::Raw),
                Engine::Native(Provenance::PushPull),
                Engine::PureNative(Provenance::PushPull),
            ] {
                let Ok(mut k) = build(signed).compile_slots(engine) else {
                    // A tier with no lowering for these nodes declines
                    // the whole program; the matrix above records the
                    // same for every fallback-only node.
                    continue;
                };
                built.insert(format!("{engine:?}"));
                k.eval_at(&[cycle]);
                for (i, o) in outs.iter().enumerate() {
                    assert_eq!(
                        k.get_value(o),
                        want[i],
                        "{engine:?} disagrees on {o}, signed={signed}, cycle={cycle}"
                    );
                }
            }
        }
    }
    // Every compiled tier carries these, the pure one included: the
    // carrier is two immediate slots and each node is a slot call of
    // its kit, which is what "no named native lowering" means and is
    // not what "interpreter-only" would have meant.
    assert_eq!(
        built.into_iter().collect::<Vec<_>>(),
        vec![
            format!("{:?}", Engine::Closures(Provenance::Raw)),
            format!("{:?}", Engine::Native(Provenance::PushPull)),
            format!("{:?}", Engine::PureNative(Provenance::PushPull)),
        ]
    );
}

/// The tier a node reports and the tier it runs on are the same
/// question, and `compile_level_of` used to answer it by asking only
/// whether the node had a scalar `compiled_u64` op — so a node whose
/// compiled form is a slot kit reported `Phase1` while the closure
/// tier ran it as a closure step and the hybrid as a slot call. The
/// binary prints that answer as "P1 interpreter".
#[test]
fn a_slot_kit_node_reports_the_tier_it_runs_on() {
    let k = polydat::dsl::compile_polydat_interpreter(
        "input cycle: u64\ntext := printf(\"n=%d\", cycle)\n",
    )
    .expect("compile");
    let program = k.program();
    let mut reported = Vec::new();
    for i in 0..program.node_count() {
        reported.push((
            program.node_meta(i).name.clone(),
            program.node_compile_level(i),
        ));
    }
    // Every node of this program has a compiled form: none should
    // report Phase1.
    let p1: Vec<&String> = reported
        .iter()
        .filter(|(_, l)| *l == polydat::ast::CompileLevel::Phase1)
        .map(|(n, _)| n)
        .collect();
    assert!(p1.is_empty(), "reported as P1 with a compiled form: {p1:?}");
}

/// A variadic node called with no wires answers its declared identity,
/// on every engine.
///
/// `FuncSig.identity` is where the identity is declared, and the native
/// lowering writes it as a constant — a second copy of the same fact,
/// and the copy drifted: `min()` answered `u64::MAX` on the interpreter
/// and `0` on native, because `0` is `max`'s identity and had been
/// written into both arms. Found by the fuzzer 2026-09-22.
///
/// The registry is the oracle here rather than a table in this file,
/// so a variadic added with an identity is covered the day it is added.
#[test]
fn a_variadic_with_no_wires_is_its_identity_on_every_engine() {
    use polydat::dsl::compile::compile_polydat_with;
    use polydat::dsl::registry;
    use polydat::{Engine, JitMode, KernelError, Provenance};

    let variadics: Vec<(&str, u64)> = registry::registry()
        .into_iter()
        .filter(|s| !s.name.starts_with("__"))
        .filter_map(|s| s.identity.map(|id| (s.name, id)))
        .collect();
    assert!(
        variadics.len() >= 4,
        "expected sum/product/min/max at least, found {variadics:?}"
    );

    let mut engines = vec![
        Engine::Interpreter(JitMode::Off),
        Engine::Interpreter(JitMode::Auto),
        Engine::Closures(Provenance::Raw),
    ];
    if cfg!(feature = "jit") {
        engines.push(Engine::Native(Provenance::Raw));
        engines.push(Engine::PureNative(Provenance::Auto));
    }

    for (name, identity) in variadics {
        let src = format!("input cycle: u64\nout := {name}()\n");
        for engine in &engines {
            let mut k = match compile_polydat_with(&src, *engine) {
                Ok(k) => k,
                Err(KernelError::Refused { .. }) => continue,
                // A variadic that refuses zero wires refuses it on
                // every engine; that is the arity rule, not this one.
                Err(_) => break,
            };
            k.set_inputs(&[1]);
            assert_eq!(
                k.pull("out"),
                polydat::ast::Value::U64(identity),
                "{name}() on {engine} is not the declared identity"
            );
        }
    }
}

/// A polymorphic node passes a value of each kind of carrier through
/// unchanged on every engine: a register and a 128-bit word (two slots),
/// narrow integers and `f32` riding a 64-bit carrier, and a string (a
/// pair). `log_warn(reg_splat_f64(cycle))` was the fuzzer's find: every
/// compiled engine handed the register back as its low limb.
#[test]
fn every_carrier_passes_through_a_polymorphic_node_on_every_engine() {
    use polydat::dsl::compile::compile_polydat_with;
    use polydat::{Engine, JitMode, Provenance};
    let producers = [
        "reg_splat_f64(cycle)",
        "__u64_to_u128(cycle)",
        "__u64_to_u32(cycle)",
        "__u64_to_i32(cycle)",
        "__u64_to_f32(cycle)",
        "format_u64(cycle)",
    ];
    for p in producers {
        let src = format!("input cycle: u64\nv := {p}\nout := log_warn(v)\n");
        let want = {
            let mut k = compile_polydat_with(&src, Engine::Interpreter(JitMode::Off))
                .unwrap_or_else(|e| panic!("{p}: {e}"));
            k.set_inputs(&[7]);
            k.pull("out")
        };
        for engine in [
            Engine::Interpreter(JitMode::Auto),
            Engine::Closures(Provenance::Raw),
            Engine::Closures(Provenance::Auto),
            Engine::Native(Provenance::Raw),
            Engine::Native(Provenance::Auto),
            Engine::PureNative(Provenance::Raw),
        ] {
            let mut k = compile_polydat_with(&src, engine).unwrap();
            k.set_inputs(&[7]);
            let got = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| k.pull("out")))
                .map_err(payload_text);
            assert_eq!(got, Ok(want.clone()), "{p} on {engine}");
        }
    }
}

/// Every node reads the same on every engine — values, at the edges.
///
/// The matrix above records whether each node *runs* on each engine: it
/// drives one cycle at `3` and discards what comes back. So nothing
/// compared a node's answer across engines, and the four drifts found
/// on 2026-09-22 all lived exactly there — a native lowering carrying a
/// second copy of the node's rule and disagreeing at an edge: `shuffle`
/// at a zero range, `blend` and `unfair_coin` at an out-of-range
/// constant, `min` with no wires. `cycle = 3` reaches none of those.
///
/// This drives each node's coverage program at inputs chosen for where
/// arithmetic breaks — zero, one, the 32- and 53-bit boundaries, the
/// signed and unsigned maxima — and compares every output against the
/// interpreter with no cones, which runs the node's own body. Both
/// failing is agreement (the message is `failure_parity`'s concern);
/// one failing and one answering is the finding, and so is two answers
/// that differ. Nondeterministic nodes are skipped, since differing is
/// what they are for; so is pure native's unset-extern trap, the one
/// runtime exception engines.md §8 names.
///
/// The whole sweep runs in about a second, so a node that makes it slow
/// is a finding too: `date_components` once walked 584 million years at
/// `u64::MAX`, and a node that sized a buffer by its input once aborted
/// the process here.
#[test]
fn every_node_reads_the_same_on_every_engine_at_the_edges() {
    const EDGES: &[u64] = &[
        0,
        1,
        3,
        255,
        // Big enough to cross the u16/i16 edges, small enough that a node
        // sizing a buffer by its input allocates it in a blink. The huge
        // edges below are refused by `buffer_for` rather than allocated.
        65536,
        (1 << 53) + 1,
        i64::MAX as u64,
        u64::MAX,
    ];
    let mut findings = Vec::new();
    let mut compared = 0usize;
    for (name, src) in common::coverage_cases::programs() {
        let (found, n) = engine_disagreements(&src, EDGES);
        compared += n;
        findings.extend(found.into_iter().map(|f| format!("{name} {f}")));
    }

    assert!(
        compared > 1000,
        "only {compared} comparisons ran; the sweep is not reaching the nodes"
    );
    assert!(
        findings.is_empty(),
        "{} node/engine/edge disagreements ({compared} compared):\n\n{}",
        findings.len(),
        findings.join("\n")
    );
}

/// Every disagreement between an engine and the node bodies over `src`
/// at each input in `inputs`, and how many reads were compared. The
/// oracle is the interpreter with no cones, which runs every node's own
/// body. Both failing is agreement (the message is `failure_parity`'s
/// concern); one failing and one answering is a finding, and so is two
/// answers that differ. A program the interpreter does not compile, or
/// one that is nondeterministic, compares nothing. Pure native's
/// unset-extern trap is the one runtime exception engines.md §8 names.
fn engine_disagreements(src: &str, inputs: &[u64]) -> (Vec<String>, usize) {
    use polydat::dsl::compile::{compile_polydat_interpreter, compile_polydat_with};
    use polydat::{Engine, JitMode, KernelError, Provenance};

    type Read = Result<Vec<polydat::ast::Value>, String>;
    // One cycle's outputs, a panic caught as its message. A kernel that
    // panicked is not trusted for the next input, so the caller rebuilds.
    fn read(k: &mut dyn polydat::Kernel, names: &[String], c: u64) -> Read {
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            k.set_inputs(&[c]);
            names.iter().map(|n| k.pull(n)).collect::<Vec<_>>()
        }))
        .map_err(payload_text)
    }
    fn agree(a: &[polydat::ast::Value], b: &[polydat::ast::Value]) -> bool {
        // `NaN != NaN`: equally undefined on both sides is agreement.
        a == b || format!("{a:?}") == format!("{b:?}")
    }

    // A node that hangs or aborts names nothing; `FUZZ_TRACE` names each
    // program before it runs.
    if std::env::var_os("FUZZ_TRACE").is_some() {
        eprintln!("engine_disagreements: {}", src.replace('\n', " ; "));
    }
    let mut findings = Vec::new();
    let mut compared = 0usize;
    let Ok(probe) = compile_polydat_interpreter(src) else {
        return (findings, compared);
    };
    if !probe.program().is_deterministic() {
        return (findings, compared);
    }
    let names: Vec<String> = polydat::Kernel::output_names(&probe);

    let build = |e: Engine| compile_polydat_with(src, e);
    let Ok(mut oracle) = build(Engine::Interpreter(JitMode::Off)) else {
        return (findings, compared);
    };
    let mut want: Vec<Read> = Vec::new();
    for &c in inputs {
        let r = read(oracle.as_mut(), &names, c);
        if r.is_err() {
            oracle = build(Engine::Interpreter(JitMode::Off)).expect("it built once");
        }
        want.push(r);
    }

    for engine in [
        Engine::Interpreter(JitMode::Auto),
        Engine::Closures(Provenance::Raw),
        Engine::Native(Provenance::Raw),
        Engine::PureNative(Provenance::Raw),
    ] {
        let mut k = match build(engine) {
            Ok(k) => k,
            Err(KernelError::Refused { .. }) => continue,
            Err(e) => {
                findings.push(format!("on {engine}: does not build: {e}"));
                continue;
            }
        };
        for (i, &c) in inputs.iter().enumerate() {
            let got = read(k.as_mut(), &names, c);
            compared += 1;
            match (&want[i], &got) {
                (Ok(a), Ok(b)) if agree(a, b) => {}
                (Err(_), Err(_)) => {}
                (_, Err(e)) if e.contains("cannot carry a `None`") => {}
                (a, b) => findings.push(format!(
                    "at cycle={c} on {engine}:\n    interpreter: {a:?}\n    {engine}: {b:?}"
                )),
            }
            if got.is_err() {
                k = match build(engine) {
                    Ok(k) => k,
                    Err(_) => break,
                };
            }
        }
    }
    (findings, compared)
}

/// Every node reads the same on every engine as each of its constants
/// moves through its edges.
///
/// A native lowering can read a node's constants as well as its wires,
/// and the sweeps above hold every constant at one value per position.
/// `weighted_pick`, the noise family and the distribution tables lower
/// through their constants, and a drift that shows only at a zero
/// weight, a negative scale or a huge count would pass them. This takes
/// every node with a numeric constant, not a list of the ones known to
/// lower, and substitutes each edge value into one constant position at
/// a time while the rest keep the coverage program's values, which are
/// known to build. A value the node's declared constraint refuses fails
/// the build the same way on every engine and compares nothing. The
/// programs run at a handful of cycle inputs, so a constant that only
/// matters against some wire values still meets them.
///
/// It found no lowering that disagreed at a constant edge, but it found
/// constants the nodes had never bounded (pinned by
/// `a_constant_outside_its_node_s_contract_is_refused_at_build`):
/// native code for `discretize` underflowed compiling zero buckets,
/// and several constants sized a node's work per call, so a huge value
/// was a hang rather than an error.
#[test]
fn every_constant_reads_the_same_on_every_engine() {
    use polydat::ast::SlotType;
    const U64_EDGES: &[&str] = &[
        "0",
        "1",
        "2",
        "7",
        "255",
        "65536",
        // Not 2^32. A constant that sizes a node's buffer is allocated
        // when it fits, however long filling it takes (`buffer_for`):
        // `byte_image_extract` builds a 4 GiB image there. The values
        // past it are refused at once, and they still set the high bits
        // a truncating lowering would drop (2^53 + 1 as `u32` is 1).
        "9007199254740993",
        "18446744073709551615",
    ];
    const F64_EDGES: &[&str] = &[
        "0.0", "-0.0", "0.5", "1.0", "-1.0", "3.5", "1e-300", "1e300", "-1e300",
    ];
    const CYCLES: &[u64] = &[0, 1, 3, 65536, (1 << 53) + 1, u64::MAX];

    let mut findings = Vec::new();
    let mut compared = 0usize;
    let mut programs = 0usize;
    for sig in polydat::dsl::registry::registry() {
        if sig.name.starts_with("__")
            || sig.category == polydat::dsl::registry::FuncCategory::RealData
        {
            continue;
        }
        let Some(base) = common::coverage_cases::synthesized_call(&sig, "cycle") else {
            continue;
        };
        let Some(args) = base
            .strip_prefix(sig.name)
            .and_then(|s| s.strip_prefix('('))
            .and_then(|s| s.strip_suffix(')'))
        else {
            continue;
        };
        let args: Vec<&str> = if args.is_empty() {
            Vec::new()
        } else {
            args.split(", ").collect()
        };
        if args.len() != sig.params.len() {
            continue;
        }
        for (pos, p) in sig.params.iter().enumerate() {
            let edges = match p.slot_type {
                SlotType::ConstU64 => U64_EDGES,
                SlotType::ConstF64 => F64_EDGES,
                _ => continue,
            };
            for v in edges {
                let mut call = args.clone();
                call[pos] = v;
                let src = format!("input cycle: u64\nout := {}({})", sig.name, call.join(", "));
                programs += 1;
                let (found, n) = engine_disagreements(&src, CYCLES);
                compared += n;
                findings.extend(
                    found.into_iter().map(|f| {
                        format!("{} with {} = {v} {f}\n    source: {src}", sig.name, p.name)
                    }),
                );
            }
        }
    }

    assert!(
        programs > 500 && compared > 5000,
        "only {programs} programs and {compared} reads; the sweep is not reaching the constants"
    );
    const SHOWN: usize = 40;
    assert!(
        findings.is_empty(),
        "{} node/constant/engine disagreements ({compared} compared, first {SHOWN} shown):\n\n{}",
        findings.len(),
        findings
            .iter()
            .take(SHOWN)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n")
    );
}

/// Constants the constant sweep found outside what their nodes can
/// answer are refused when the program is built, on every engine,
/// rather than hanging, panicking, or overflowing where only some
/// engines notice.
#[test]
fn a_constant_outside_its_node_s_contract_is_refused_at_build() {
    use polydat::{Engine, JitMode, Provenance};
    let cases = [
        // Native code computed `buckets - 1` while compiling.
        ("discretize(cycle, 100.0, 0)", "buckets"),
        // The body clamped to `range - EPSILON`, "min > max" below it.
        ("discretize(cycle, 0.0, 10)", "range"),
        // `[min, min + size)` past `u64::MAX` overflowed.
        (
            "shuffle(cycle, 100, 101, 18446744073709551615)",
            "must fit in u64",
        ),
        // An octave count is work per call, and zero octaves is NaN.
        ("fractal_noise_1d(cycle, 100, 1.0, 0)", "octaves"),
        ("fractal_noise_2d(cycle, cycle, 100, 1.0, 65)", "octaves"),
        // `n_of` hashes the whole window on every call.
        ("n_of(cycle, 1, 65537)", "m"),
    ];
    for (call, names) in cases {
        let src = format!("input cycle: u64\nout := {call}\n");
        for engine in [
            Engine::Interpreter(JitMode::Off),
            Engine::Closures(Provenance::Raw),
            Engine::Native(Provenance::Raw),
            Engine::PureNative(Provenance::Raw),
        ] {
            let err = compile_polydat_with(&src, engine)
                .err()
                .unwrap_or_else(|| panic!("{call} built on {engine}"))
                .to_string();
            assert!(err.contains(names), "{call} on {engine}: {err}");
        }
    }
}

/// A pull runs the requested output's cone and nothing else, on every
/// engine (engines.md §3.1). `counter()` counts its own runs, and it is
/// outside the cone of `a`: however many times `a` is pulled, the first
/// pull of `n` finds the counter never run, on every engine and in both
/// of pure native code's modes. Pure native code used to run its one
/// function for the whole program on every pull after a write, so each
/// pull of `a` advanced a counter nothing had asked for.
#[test]
fn a_pull_runs_its_own_cone_and_nothing_else_on_every_engine() {
    use polydat::dsl::compile::compile_polydat_with;
    use polydat::{Engine, JitMode, Provenance};
    let src = "input cycle: u64\na := hash(cycle)\nn := counter()\n";
    for engine in [
        Engine::Interpreter(JitMode::Off),
        Engine::Interpreter(JitMode::Auto),
        Engine::Closures(Provenance::Raw),
        Engine::Closures(Provenance::PushPull),
        Engine::Native(Provenance::Raw),
        Engine::Native(Provenance::PushPull),
        Engine::PureNative(Provenance::Raw),
        Engine::PureNative(Provenance::PushPull),
    ] {
        let mut k = compile_polydat_with(src, engine).unwrap();
        for c in 0..5u64 {
            k.set_inputs(&[c]);
            let _ = k.pull("a");
        }
        k.set_inputs(&[5]);
        assert_eq!(
            k.pull("n"),
            polydat::ast::Value::U64(0),
            "{engine}: pulling `a` ran `counter()`, which is outside its cone"
        );
    }
}

/// Native segments follow the graph's connections, not the statements'
/// order (SRD-105, compile::fusion_units): three chains that share
/// nothing, written interleaved, are three segments, so pulling one
/// runs one. A run of consecutive native nodes would have made them a
/// single segment, and every pull would have run all three.
#[cfg(feature = "jit")]
#[test]
fn independent_chains_are_separate_native_segments() {
    use polydat::dsl::compile::compile_polydat_with;
    use polydat::{Engine, Provenance};
    let src = "input cycle: u64\n\
               a1 := hash(cycle)\nb1 := hash(cycle)\nc1 := hash(cycle)\n\
               a2 := hash(a1)\nb2 := hash(b1)\nc2 := hash(c1)\n\
               a3 := hash(a2)\nb3 := hash(b2)\nc3 := hash(c2)\n";
    // Against one chain, so whatever else the program plans (the
    // literals of a node's defaults fold into one constant segment)
    // cancels out: each further chain is one further segment.
    let one = "input cycle: u64\na1 := hash(cycle)\na2 := hash(a1)\na3 := hash(a2)\n";
    for provenance in [Provenance::Raw, Provenance::PushPull] {
        let plan = compile_polydat_with(src, Engine::Native(provenance))
            .unwrap()
            .plan();
        let base = compile_polydat_with(one, Engine::Native(provenance))
            .unwrap()
            .plan();
        assert_eq!(
            (plan.native_segments, plan.closure_steps),
            (base.native_segments + 2, 0),
            "{provenance:?}: three chains plan {plan}, one plans {base}"
        );
    }
    // The same program answers alike on every engine, pulled one chain
    // at a time in an order that crosses them.
    for engine in [
        Engine::Native(Provenance::Raw),
        Engine::Native(Provenance::PushPull),
        Engine::PureNative(Provenance::Raw),
        Engine::PureNative(Provenance::PushPull),
    ] {
        let mut want = compile_polydat_with(src, Engine::Closures(Provenance::Raw)).unwrap();
        let mut k = compile_polydat_with(src, engine).unwrap();
        for c in 0..6u64 {
            want.set_inputs(&[c]);
            k.set_inputs(&[c]);
            for name in ["c3", "a1", "b3", "a3", "c2"] {
                assert_eq!(k.pull(name), want.pull(name), "{engine}: {name} at {c}");
            }
        }
    }
}
