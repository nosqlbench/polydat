// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Pragma scoping for the type round-trip lint (graph_compiler.md §2.3,
//! polydat_grammar.md §14.1). A round trip is a warning, and an error
//! when the scope the restoring binding is written in has
//! `strict_values` on: a module body under its own pragmas, a `for`
//! body under its enclosing scope's plus its own, and the program under
//! its own. Each case compiles on every engine.

use polydat::dsl::compile::{CompileOptions, compile_polydat_with_engine};
use polydat::dsl::events::{CompileEvent, CompileEventLog};
use polydat::{Engine, JitMode, Provenance};

fn engines() -> Vec<Engine> {
    let mut engines = vec![
        Engine::Interpreter(JitMode::Off),
        Engine::Interpreter(JitMode::Auto),
        Engine::Closures(Provenance::Auto),
    ];
    if cfg!(feature = "jit") {
        engines.push(Engine::Native(Provenance::Auto));
        engines.push(Engine::PureNative(Provenance::PushPull));
        engines.push(Engine::PureNative(Provenance::Raw));
    }
    engines
}

/// A `u64` sent through `i64` and back: the round trip the lint names.
const TRIP: &str = "__i64_to_u64(__u64_to_i64(a))";

/// The round-trip warnings building `source` on `engine` logged, or
/// the compile error.
fn warnings(source: &str, engine: Engine) -> Result<usize, String> {
    let mut log = CompileEventLog::new();
    compile_polydat_with_engine(source, engine, &CompileOptions::default(), Some(&mut log))
        .map_err(|e| e.to_string())?;
    Ok(log
        .events()
        .iter()
        .filter(|e| matches!(e, CompileEvent::Warning { message } if message.contains("type round trip")))
        .count())
}

fn fails(source: &str, engine: Engine) {
    let err = warnings(source, engine)
        .err()
        .unwrap_or_else(|| panic!("{engine}: expected a round-trip error\n{source}"));
    assert!(err.contains("type round trip"), "{engine}: {err}\n{source}");
}

fn warns(source: &str, engine: Engine) {
    let found = warnings(source, engine).unwrap_or_else(|e| panic!("{engine}: {e}\n{source}"));
    assert!(
        found >= 1,
        "{engine}: expected a round-trip warning\n{source}"
    );
}

/// A program with a module `m` and a host binding `h`, the pragma in
/// the host, the module, or neither, and the round trip in the module's
/// body or the host's.
fn program(host_pragma: bool, module_pragma: bool, trip_in_module: bool) -> String {
    let module_body = if trip_in_module { TRIP } else { "a" };
    let host_body = if trip_in_module {
        "cycle".to_string()
    } else {
        TRIP.replace("(a)", "(cycle)")
    };
    format!(
        "{}input cycle: u64\n\
         m(a: u64) -> (out: u64) := {{\n\
             {}out := {module_body}\n\
         }}\n\
         s := m(cycle)\n\
         h := {host_body}\n",
        if host_pragma {
            "pragma strict_values\n"
        } else {
            ""
        },
        if module_pragma {
            "pragma strict_values\n"
        } else {
            ""
        },
    )
}

/// The program's own round trip is an error under its pragma and a
/// warning without it.
#[test]
fn a_program_pragma_decides_its_own_round_trips() {
    for engine in engines() {
        fails(&program(true, false, false), engine);
        warns(&program(false, false, false), engine);
    }
}

/// A strict module in a lax host: the module's round trip is linted
/// under the module's pragma and fails, and the host's is linted under
/// the host's and warns.
#[test]
fn a_strict_module_in_a_lax_host_is_strict_and_its_host_is_not() {
    for engine in engines() {
        fails(&program(false, true, true), engine);
        warns(&program(false, true, false), engine);
    }
}

/// A lax module in a strict host: the module's round trip is linted
/// under the module's own set and warns, and the host's fails.
#[test]
fn a_lax_module_in_a_strict_host_is_lax_and_its_host_is_not() {
    for engine in engines() {
        warns(&program(true, false, true), engine);
        fails(&program(true, false, false), engine);
    }
}

/// A `for` body's pragma applies to the body's round trip, and an
/// enclosing program's pragma reaches the body.
#[test]
fn a_for_body_is_linted_under_its_scope() {
    let body = TRIP.replace("(a)", "(k)");
    for engine in engines() {
        fails(
            &format!(
                "input cycle: u64\nfor k in 1..3 {{\n    pragma strict_values\n    c := {body}\n}}\n"
            ),
            engine,
        );
        fails(
            &format!(
                "pragma strict_values\ninput cycle: u64\nfor k in 1..3 {{\n    c := {body}\n}}\n"
            ),
            engine,
        );
        warnings(
            &format!("input cycle: u64\nfor k in 1..3 {{\n    c := {body}\n}}\n"),
            engine,
        )
        .unwrap_or_else(|e| panic!("{engine}: {e}"));
    }
}
