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

/// One engine's outcome for one program.
fn outcome<T>(r: std::thread::Result<Result<T, String>>) -> (&'static str, String) {
    match r {
        Ok(Ok(_)) => ("ok", String::new()),
        Ok(Err(e)) => ("refused", e.lines().next().unwrap_or("").to_string()),
        Err(p) => {
            let msg = p
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_default();
            ("failed", msg.lines().next().unwrap_or("").to_string())
        }
    }
}

/// The four outcomes for one program: interpreter, closure tier, hybrid
/// kernel, pure native code. Each engine compiles, runs one cycle, and
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
    let hybrid = outcome(std::panic::catch_unwind(|| {
        let mut k = compile_polydat_to_assembler(src)?.compile_hybrid()?;
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
    [p1, p2, hybrid, p3]
}

#[test]
fn the_node_by_engine_matrix_is_as_recorded() {
    let table = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/engine_parity.txt");
    let mut lines = vec![
        "# node\tP1\tP2\thybrid\tP3 — outcomes per engine; regenerate with ENGINE_PARITY=overwrite"
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
        for (engine, (k, msg)) in ["P1", "P2", "hybrid", "P3"].iter().zip(r.iter()) {
            if *k != "ok" {
                detail.push(format!("  {name} on {engine}: {k}: {msg}"));
            }
        }
    }
    if trace {
        eprintln!("{}", detail.join("\n"));
    }
    let current = lines.join("\n") + "\n";
    if std::env::var("ENGINE_PARITY").as_deref() == Ok("overwrite") {
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
