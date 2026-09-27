// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Pragma scoping for tiles (polytile.md §6, polydat_grammar.md §14.1).
//! A tile is not a pragma scope of its own: its holes and its
//! projection bodies compile under the pragmas of the scope the tile is
//! written in, a program or a module body, as a `for` body does. Each
//! case compiles on every engine a tile renders on.

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

/// The runtime guards `strict_values` inserted building `source` on
/// `engine`, or the compile error.
fn guards(source: &str, engine: Engine) -> Result<usize, String> {
    let mut log = CompileEventLog::new();
    compile_polydat_with_engine(source, engine, &CompileOptions::default(), Some(&mut log))
        .map_err(|e| e.to_string())?;
    Ok(log
        .events()
        .iter()
        .filter(|e| matches!(e, CompileEvent::AssertionInserted { .. }))
        .count())
}

fn fails_strict(source: &str, engine: Engine) {
    let err = guards(source, engine)
        .err()
        .unwrap_or_else(|| panic!("{engine}: expected a strict_values error\n{source}"));
    assert!(
        err.contains("strict_values") && err.contains("'divisor'"),
        "{engine}: {err}\n{source}"
    );
}

/// Under a program's `strict_values`, a constant that fails a
/// constraint inside a tile fails the build, in a hole and in a
/// projection body alike; without the pragma both build. The body
/// reads `cycle`, so its value is not computed at build.
#[test]
fn a_program_pragma_checks_the_constants_of_its_tiles() {
    let hole = "input cycle: u64\ntile t : text := \"n=${mod_wire(cycle, 0)}\"\n";
    let body = "input cycle: u64\ntile t : text := \"@for k in 1..3 sep \\\",\\\" {${mod_wire(k + cycle, 0)}}\"\n";
    for engine in engines() {
        for source in [hole, body] {
            fails_strict(&format!("pragma strict_values\n{source}"), engine);
            guards(source, engine).unwrap_or_else(|e| panic!("{engine}: {e}\n{source}"));
        }
    }
}

/// Under a program's `strict_values`, a nondeterministic source inside
/// a tile's hole gets a runtime guard.
#[test]
fn a_program_pragma_guards_a_nondeterministic_source_in_a_tile() {
    let source = "input cycle: u64\ntile t : text := \"n=${mod_wire(cycle, counter())}\"\n";
    for engine in engines() {
        let strict = format!("pragma strict_values\n{source}");
        assert_eq!(guards(&strict, engine), Ok(1), "{engine}");
        assert_eq!(guards(source, engine), Ok(0), "{engine}");
    }
}

/// A tile in a strict module is checked under the module's pragmas,
/// in its holes and its projection bodies, and the non-strict host
/// that calls the module is not.
#[test]
fn a_tile_in_a_strict_module_is_strict_and_its_host_is_not() {
    let program = |tile: &str| {
        format!(
            "input cycle: u64\n\
             checked(a: u64) -> (doc: str) := {{\n\
                 pragma strict_values\n\
                 tile doc : text := \"{tile}\"\n\
             }}\n\
             s := checked(cycle)\n\
             b := mod_wire(cycle, counter())\n\
             z := mod_wire(cycle, 0)\n"
        )
    };
    for engine in engines() {
        fails_strict(&program("${mod_wire(a, 0)}"), engine);
        fails_strict(
            &program("@for k in 1..3 sep \\\",\\\" {${mod_wire(k + a, 0)}}"),
            engine,
        );
        // The module's guard, and none for the host's `counter()`.
        assert_eq!(
            guards(&program("${mod_wire(a, counter())}"), engine),
            Ok(1),
            "{engine}"
        );
    }
}

/// A tile in a module that declares no pragma compiles under the
/// module's own set, so a strict host's pragma does not check it; the
/// host's own bindings stay checked.
#[test]
fn a_tile_in_a_non_strict_module_is_not_strict_under_a_strict_host() {
    let program = |tile: &str, host: &str| {
        format!(
            "pragma strict_values\n\
             input cycle: u64\n\
             lax(a: u64) -> (doc: str) := {{\n\
                 tile doc : text := \"{tile}\"\n\
             }}\n\
             s := lax(cycle)\n\
             {host}\n"
        )
    };
    for engine in engines() {
        for tile in [
            "${mod_wire(a, 0)}",
            "@for k in 1..3 sep \\\",\\\" {${mod_wire(k + a, 0)}}",
        ] {
            guards(&program(tile, "b := cycle"), engine)
                .unwrap_or_else(|e| panic!("{engine}: {e}"));
        }
        // No guard for the module's `counter()`, one for the host's.
        assert_eq!(
            guards(
                &program(
                    "${mod_wire(a, counter())}",
                    "b := mod_wire(cycle, counter())"
                ),
                engine
            ),
            Ok(1),
            "{engine}"
        );
        fails_strict(
            &program("${mod_wire(a, 7)}", "b := mod_wire(cycle, 0)"),
            engine,
        );
    }
}
