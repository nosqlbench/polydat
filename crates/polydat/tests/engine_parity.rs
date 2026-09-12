// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! The node-by-engine matrix of docs/design/engine_parity.md, pinned.
//!
//! Every registered public node is compiled and run through one cycle
//! on each engine from the same program the coverage test uses, and the
//! outcome per engine (`ok`, `refused` at compile, `failed` at run) is
//! compared with the checked-in table `tests/engine_parity.txt`. Any
//! change in either direction fails the test: a regression on an engine
//! that accepted a node, and a step of the true-up plan that lands
//! without recording its gain. Regenerate the table after an intended
//! change with `ENGINE_PARITY=overwrite cargo test --test engine_parity`
//! and review the diff.
//!
//! The matrix needs every engine, so it runs with the `jit` feature.

#![cfg(feature = "jit")]

mod common;

use polydat::dsl::compile::compile_polydat_to_assembler;
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
        let mut asm = compile_polydat_to_assembler(src)?;
        asm.set_jit_mode(polydat::JitMode::Off);
        let mut k = asm.compile().map_err(|e| e.to_string())?;
        k.set_inputs(&[3]);
        let outs: Vec<String> = k.output_names().iter().map(|s| s.to_string()).collect();
        for o in &outs {
            let _ = k.pull(o);
        }
        Ok::<(), String>(())
    }));
    let p2 = outcome(std::panic::catch_unwind(|| {
        let mut k = compile_polydat_to_assembler(src)?
            .try_compile_raw()
            .map_err(|_| "no closure form for some node".to_string())?;
        k.eval(&[3]);
        let outs: Vec<String> = k.output_names().iter().map(|s| s.to_string()).collect();
        for o in &outs {
            let _ = k.get_value(o);
        }
        Ok::<(), String>(())
    }));
    let p3 = outcome(std::panic::catch_unwind(|| {
        let mut k = compile_polydat_to_assembler(src)?.try_compile_jit()?;
        k.eval(&[3]);
        let outs: Vec<String> = k.output_names().iter().map(|s| s.to_string()).collect();
        for o in &outs {
            let _ = k.get_value(o);
        }
        Ok::<(), String>(())
    }));
    let pure = outcome(std::panic::catch_unwind(|| {
        let mut k = compile_polydat_to_assembler(src)?.try_compile_pure_jit()?;
        k.eval(&[3]);
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
            "the node-by-engine matrix changed.\n\nrecorded but no longer so:\n{}\n\nnow, not recorded:\n{}\n\nrefusals and failures now:\n{}\n\nIf the change is intended, regenerate with ENGINE_PARITY=overwrite and record the step in docs/design/engine_parity.md.",
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
            .try_compile_raw()
            .ok();
        let mut p3 = compile_polydat_to_assembler(&src)
            .unwrap()
            .try_compile_jit()
            .ok();
        let mut pure = compile_polydat_to_assembler(&src)
            .unwrap()
            .try_compile_pure_jit()
            .ok();
        for &c in &cycles {
            // An assertion node fails on some cycles by design; the
            // interpreter's failure is the oracle's, and an engine that
            // fails too must fail with the same message, location
            // aside (the matrix pins the failures at cycle 3).
            let want = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                p1.set_inputs(&[c]);
                outs.iter()
                    .map(|o| p1.pull(o).clone())
                    .collect::<Vec<Value>>()
            }))
            .map_err(payload_text);
            let mut got: Vec<(&str, Result<Vec<Value>, String>)> = Vec::new();
            if let Some(k) = p2.as_mut() {
                let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    k.eval(&[c]);
                    outs.iter().map(|o| k.get_value(o)).collect::<Vec<_>>()
                }));
                got.push(("P2", r.map_err(payload_text)));
            }
            if let Some(k) = p3.as_mut() {
                let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    k.eval(&[c]);
                    outs.iter().map(|o| k.get_value(o)).collect::<Vec<_>>()
                }));
                got.push(("P3", r.map_err(payload_text)));
            }
            if let Some(k) = pure.as_mut() {
                let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    k.eval(&[c]);
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
    section.push_str(&format!(
        "\nPure native code, the differential tier behind P3, has no lowering for \
         these {} nodes, which P3 runs as closure steps:\n\n{}.\n\n",
        pure_refused.len(),
        list(&pure_refused)
    ));
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
