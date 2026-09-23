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
