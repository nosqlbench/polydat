// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! One carrier for a body, and the engine it runs on (F-L2).
//!
//! A `for` body and a tile's projection body are the same thing: a
//! statement list compiled once per engine and shared by every use.
//! They reach that through one type, `BodySource`, and one method,
//! `program_on` (for_traversal.md §4, polytile.md §7.2).
//!
//! A body compiles for an engine on that engine's first request, and
//! a projection body renders on the engine of the kernel it belongs
//! to rather than on `Engine::default()`.

use polydat::dsl::traversal::BodySource;
use polydat::{Engine, JitMode, Provenance};

const BODY: &str = "input __tuple: u64\nextern k: u64\nn := u64_mul(k, 3)\n";

#[test]
fn a_body_compiles_once_per_engine_and_is_shared() {
    let body = BodySource::from_source(BODY, "a test body").expect("it parses");

    let interpreted = body
        .program_on(Engine::Interpreter(JitMode::Off))
        .expect("the interpreter takes it");
    assert!(
        interpreted.clone().as_interpreter().is_some(),
        "the interpreter's program is an interpreter program"
    );
    // The same engine asked twice is the same program, not a second
    // compile: the carrier holds one per engine.
    let again = body
        .program_on(Engine::Interpreter(JitMode::Off))
        .expect("cached");
    assert!(
        std::sync::Arc::ptr_eq(&interpreted, &again),
        "a second ask for one engine returns the program already built"
    );

    // Another engine is another program, built when it is asked for
    // and not before.
    let closures = body
        .program_on(Engine::Closures(Provenance::Raw))
        .expect("the closure tier takes it");
    assert!(
        !std::sync::Arc::ptr_eq(&interpreted, &closures),
        "each engine has its own program"
    );
}

/// A tile whose projection body is rendered gives the same text on
/// every engine. The body is compiled for the engine the kernel
/// rendering it runs on, so this is the check that the engine it
/// lands on is one that agrees with the rest.
#[test]
fn a_projection_renders_the_same_on_every_engine() {
    let src = "input cycle: u64\n\
               base := mod(cycle, 5)\n\
               tile rows : csv := <<<@for k in 1..4 sep \",\" {${k}:${base}}>>>\n";
    let mut engines = vec![
        Engine::Interpreter(JitMode::Off),
        Engine::Interpreter(JitMode::Auto),
        Engine::Closures(Provenance::Raw),
        Engine::Native(Provenance::PushPull),
    ];
    if cfg!(feature = "jit") {
        engines.push(Engine::PureNative(Provenance::PushPull));
        engines.push(Engine::PureNative(Provenance::Raw));
    }
    let mut answers = Vec::new();
    for engine in engines {
        let mut k = polydat::dsl::compile::compile_polydat_with(src, engine)
            .unwrap_or_else(|e| panic!("{engine:?}: {e}"));
        let mut rows = Vec::new();
        for c in 0..4u64 {
            k.set_inputs(&[c]);
            rows.push(k.pull("rows").as_str().to_string());
        }
        answers.push((engine, rows));
    }
    let (_, first) = &answers[0];
    assert!(!first[0].is_empty(), "the tile rendered something");
    for (engine, rows) in &answers[1..] {
        assert_eq!(rows, first, "{engine:?} renders a projection differently");
    }
}

/// A tile's projection body compiles under the settings the program
/// around it compiles under (F-L4).
///
/// The compiler hands the render node the body it lowered, with the
/// source directory, library paths, strict flag, pragmas, and module
/// table it used, rather than source text compiled with defaults. A
/// module the program itself defines is therefore callable inside the
/// body, as it is in every other scope of the same program.
#[test]
fn a_projection_body_sees_the_program_s_modules() {
    let src = "input cycle: u64\n\
               dbl(a: u64) -> (o: u64) := { o := u64_mul(a, 2) }\n\
               tile rows : csv := <<<@for k in 1..4 sep \",\" {${dbl(k)}}>>>\n";
    let mut k = polydat::dsl::compile::compile_polydat_interpreter(src)
        .expect("a module the program defines resolves inside the body");
    k.set_inputs(&[1]);
    assert_eq!(polydat::Kernel::pull(&mut k, "rows").as_str(), "2,4,6");
}
