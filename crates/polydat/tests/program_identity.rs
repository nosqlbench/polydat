// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Program identity (scope_model.md §8): one program built on every
//! engine has one canonical hash, kernels created from or forked off it
//! report the same one, and the instance hash chains canonical hashes
//! the same way whichever engine produced them.

use polydat::dsl::compile::{CompileOptions, compile_polydat_with_engine};
use polydat::kernel::instance_hash_of;
use polydat::{Engine, JitMode, Kernel, Provenance};

fn every_engine() -> Vec<Engine> {
    let mut all = vec![
        Engine::Interpreter(JitMode::Off),
        Engine::Interpreter(JitMode::Force),
        Engine::Closures(Provenance::Auto),
        Engine::Closures(Provenance::Raw),
    ];
    if cfg!(feature = "jit") {
        all.push(Engine::Native(Provenance::Auto));
        all.push(Engine::PureNative(Provenance::Auto));
        all.push(Engine::PureNative(Provenance::Raw));
    }
    all
}

fn build(src: &str, engine: Engine) -> Box<dyn Kernel> {
    compile_polydat_with_engine(src, engine, &CompileOptions::default(), None)
        .unwrap_or_else(|e| panic!("{engine}: {e}\n{src}"))
}

/// Programs that exercise what an engine changes at build: a const the
/// interpreter folds, a cone the native tiers fuse, a declared extern,
/// a modifier, and a cursor.
const PROGRAMS: &[&str] = &[
    "input cycle: u64\nh := hash(cycle)\nout := mod(h, 1000)\n",
    "input cycle: u64\nconst c := 40 + 2\nv := u64_mul(cycle, 3)\nw := u64_add(v, c)\n",
    "input cycle: u64\nextern limit: u64 = 7\nvolatile r := u64_add(cycle, limit)\n",
    "input cycle: u64\ncursor rows = range(0, 100)\nout := u64_mul(rows.ordinal, 3)\n",
    "input cycle: u64\nx := cycle as f64\ny := x * 2.5\nname := \"row-{cycle}\"\n",
    "input cycle: u64\nbase := hash(cycle)\nfor k in 1..4 {\n    g := u64_add(base, k)\n}\n",
];

#[test]
fn what_the_program_computes_moves_the_hash_and_its_spelling_does_not() {
    let base =
        "input cycle: u64\nextern limit: u64 = 7\nfor k in 1..4 {\n    g := u64_add(limit, k)\n}\n";
    let changes = [
        // An extern's default, where a host fixes a value.
        base.replace("= 7", "= 8"),
        // A `for` body.
        base.replace("u64_add(limit, k)", "u64_mul(limit, k)"),
        // A comprehension.
        base.replace("1..4", "1..5"),
    ];
    let spellings = [format!("# a comment\n{base}"), base.replace(":= ", ":=  ")];
    for engine in every_engine() {
        let reference = build(base, engine).canonical_hash();
        for changed in &changes {
            assert_ne!(
                build(changed, engine).canonical_hash(),
                reference,
                "{engine}\n{changed}"
            );
        }
        for spelled in &spellings {
            assert_eq!(
                build(spelled, engine).canonical_hash(),
                reference,
                "{engine}\n{spelled}"
            );
        }
    }
}

#[test]
fn one_program_on_every_engine_has_one_canonical_hash() {
    for src in PROGRAMS {
        let reference = build(src, Engine::Interpreter(JitMode::Off)).canonical_hash();
        for engine in every_engine() {
            let k = build(src, engine);
            assert_eq!(k.canonical_hash(), reference, "{engine}\n{src}");
        }
    }
}

#[test]
fn created_and_forked_kernels_report_their_programs_hash() {
    for engine in every_engine() {
        let k = build(PROGRAMS[1], engine);
        let hash = k.canonical_hash();
        assert_eq!(k.fork().canonical_hash(), hash, "{engine}: fork");
        let program = k.into_program();
        assert_eq!(program.canonical_hash(), hash, "{engine}: program");
        assert_eq!(
            program.create_kernel().canonical_hash(),
            hash,
            "{engine}: created"
        );
    }
}

#[test]
fn different_programs_hash_differently_on_every_engine() {
    for engine in every_engine() {
        let a = build(PROGRAMS[0], engine).canonical_hash();
        let b = build(&PROGRAMS[0].replace("1000", "1001"), engine).canonical_hash();
        assert_ne!(a, b, "{engine}");
    }
}

#[test]
fn the_instance_hash_chains_canonical_hashes_from_any_engine() {
    let child_src = "input cycle: u64\nconst y := 42\n";
    let parent_src = "input cycle: u64\nconst ds := \"v1\"\n";
    let child = polydat::dsl::compile_polydat_interpreter(child_src).unwrap();
    let parent = polydat::dsl::compile_polydat_interpreter(parent_src).unwrap();
    let from_programs = child.program().instance_hash(&[parent.program().as_ref()]);
    for engine in every_engine() {
        let own = build(child_src, engine).canonical_hash();
        let ancestor = build(parent_src, engine).canonical_hash();
        assert_eq!(
            instance_hash_of(own, &[ancestor]),
            from_programs,
            "{engine}"
        );
    }
}
