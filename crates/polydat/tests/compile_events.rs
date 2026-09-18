// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! The compile event log retells a compile: every step that the
//! `explain` command reports reports itself, from the parse to the
//! summary, so a phase never says "nothing happened" about a step that
//! happened.

use std::process::Command;

use polydat::dsl::compile::{compile_polydat_interpreter_with_log, compile_polydat_to_assembler};
use polydat::dsl::events::{CompileEvent, CompileEventLog};
use polydat::{Engine, Provenance};

const SRC: &str = "input cycle: u64\n\
    pair_sum(a: u64, b: u64) -> (total: u64) := {\n\
        total := a + b\n\
    }\n\
    s := pair_sum(cycle, 10)\n\
    f := f64_add(s, 1.5)\n";

#[test]
fn the_log_retells_a_compile_from_parse_to_summary() {
    let mut log = CompileEventLog::new();
    compile_polydat_interpreter_with_log(SRC, &mut log).unwrap();
    let events = log.events();
    assert!(
        matches!(events.first(), Some(CompileEvent::Parsed { statements: 4 })),
        "{:?}",
        events.first()
    );
    assert!(
        events.iter().any(|e| matches!(
            e,
            CompileEvent::ModuleInlined { name, nodes_added } if name == "pair_sum" && *nodes_added > 0
        )),
        "{events:?}"
    );
    assert!(
        events.iter().any(|e| matches!(
            e,
            CompileEvent::BindingResolved { name, node_type } if name == "f" && node_type == "f64_add"
        )),
        "{events:?}"
    );
    // `s` is a u64 fed to an f64 port: a lossless widening, reported as
    // one and not as a generic adapter.
    assert!(
        events.iter().any(|e| matches!(
            e,
            CompileEvent::TypeWidening { from: "u64", to: "f64", context } if context.contains("__adapt") || context.contains("f")
        )),
        "{events:?}"
    );
    for out in ["s", "f"] {
        assert!(
            events
                .iter()
                .any(|e| matches!(e, CompileEvent::OutputDeclared { name } if name == out)),
            "{out}: {events:?}"
        );
    }
    assert!(
        events.iter().any(|e| matches!(
            e,
            CompileEvent::CompileLevelSelected { node, level } if node == "f" && level == "native"
        )),
        "{events:?}"
    );
    assert!(
        matches!(
            events.last(),
            Some(CompileEvent::Summary { nodes, outputs, .. }) if *nodes > 0 && *outputs >= 2
        ),
        "{:?}",
        events.last()
    );
}

/// A compiled tier's log reports the same forms as the interpreter's:
/// the form is the node's, not the engine's (engines.md §7).
#[test]
fn the_compiled_tiers_report_each_nodes_form() {
    let asm = compile_polydat_to_assembler(SRC).unwrap();
    let mut log = CompileEventLog::new();
    asm.compile_engine_with_log(Engine::Closures(Provenance::Auto), Some(&mut log))
        .unwrap();
    let levels: Vec<(String, String)> = log
        .events()
        .iter()
        .filter_map(|e| match e {
            CompileEvent::CompileLevelSelected { node, level } => {
                Some((node.clone(), level.clone()))
            }
            _ => None,
        })
        .collect();
    assert!(levels.iter().any(|(n, _)| n == "f"), "{levels:?}");
    assert!(
        levels.iter().all(|(_, l)| {
            [
                "native",
                "compiled u64 op",
                "slot kit",
                "slot copy",
                "interpreted",
            ]
            .contains(&l.as_str())
        }),
        "{levels:?}"
    );
}

/// `explain` reads the events: the modules phase names the module that
/// was inlined instead of saying none was.
#[test]
fn explain_names_the_module_that_was_inlined() {
    let dir = std::env::temp_dir().join(format!("polydat_explain_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("inlined.polydat");
    std::fs::write(&path, SRC).unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_polydat"))
        .args(["explain", path.to_str().unwrap(), "modules"])
        .output()
        .expect("polydat binary runs");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        stdout.contains("module pair_sum was inlined")
            && !stdout.contains("No modules were inlined"),
        "{stdout}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}
