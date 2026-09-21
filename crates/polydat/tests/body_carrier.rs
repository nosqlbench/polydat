// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! One carrier for a body, and the engine it runs on (F-L2, decision 3).
//!
//! A `for` body and a tile's projection body are the same thing: a
//! statement list compiled once per engine and shared by every use.
//! They reach that through one type now, `BodySource`, and one method,
//! `program_on`.
//!
//! The tile path used to compile each body twice when the node was
//! constructed — once for the interpreter and once on
//! `Engine::default()` — whether or not either engine ever rendered
//! it, and then render on the default engine whatever engine the
//! kernel it belonged to was running.

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
    let engines = [
        Engine::Interpreter(JitMode::Off),
        Engine::Interpreter(JitMode::Auto),
        Engine::Closures(Provenance::Raw),
        Engine::Native(Provenance::PushPull),
    ];
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
/// The body used to reach the render node as source text inside a
/// JSON payload, and was compiled there with defaults: no source
/// directory, no library paths, no strict flag, no pragmas, and an
/// empty module table. So a module the program itself defines — which
/// the compiler had already resolved — was an unknown function inside
/// the body, while every other scope of the same program could call
/// it. The compiler hands the node the body it lowered now, settings
/// and all.
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
