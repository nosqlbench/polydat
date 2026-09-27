// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Program identity (scope_model.md §8): one program built on every
//! engine has one canonical hash, kernels created from or forked off it
//! report the same one, and the instance hash chains canonical hashes
//! the same way whichever engine produced them. A child's inherited
//! outputs count on every engine, and a graph built by the raw tier
//! constructors hashes as the same graph through the normal path.

use polydat::ast::PortType;
use polydat::compile::assembly::{PolydatAssembler, WireRef};
use polydat::dsl::compile::{CompileOptions, compile_polydat_with_engine};
use polydat::dsl::factory::{BuildContext, ConstArg, build_node};
use polydat::kernel::instance_hash_of;
use polydat::kernel::subcontext::PolydatMatter;
use polydat::library::identity::PortPassthrough;
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

// ── A child with inherited outputs ─────────────────────────────────

const PARENT: &str = "input cycle: u64\nextern limit: u64 = 7\nbase := hash(cycle)\n";
const CHILD: &str = "input cycle: u64\nextern limit: u64\nextern base: u64\n\
                     relay := u64_add(limit, 0)\nown := u64_add(base, 1)\n";
/// A child that adds nothing: no output, and only inputs its parent
/// declares.
const EMPTY_CHILD: &str = "input cycle: u64\nextern limit: u64\n";

fn child_under(parent: &dyn Kernel, src: &str, inherited: &[&str]) -> Box<dyn Kernel> {
    PolydatMatter::builder()
        .label("child")
        .source(src)
        .inherited_outputs(inherited.iter().map(|s| s.to_string()).collect())
        .build()
        .expect("matter")
        .build_under(parent)
        .unwrap_or_else(|e| panic!("{}: {e}\n{src}", parent.engine()))
}

#[test]
fn a_childs_inherited_outputs_are_part_of_its_identity_on_every_engine() {
    let reference_parent = build(PARENT, Engine::Interpreter(JitMode::Off));
    let reference = child_under(reference_parent.as_ref(), CHILD, &["relay"]);
    let unmarked = child_under(reference_parent.as_ref(), CHILD, &[]);
    assert_ne!(
        reference.canonical_hash(),
        unmarked.canonical_hash(),
        "an inherited mark changes the program"
    );
    let program = reference.as_interpreter().expect("interpreter").program();
    let parent_program = reference_parent
        .as_interpreter()
        .expect("interpreter")
        .program();
    for engine in every_engine() {
        let parent = build(PARENT, engine);
        let child = child_under(parent.as_ref(), CHILD, &["relay"]);
        assert_eq!(
            child.engine(),
            parent.engine(),
            "{engine}: the child's engine"
        );
        assert_eq!(
            child.canonical_hash(),
            reference.canonical_hash(),
            "{engine}"
        );
        assert!(child.is_equivalent_to(reference.as_ref()), "{engine}");
        assert!(!child.is_equivalent_to(unmarked.as_ref()), "{engine}");
        assert_eq!(
            child.instance_hash(&[parent.as_ref()]),
            program.instance_hash(&[parent_program.as_ref()]),
            "{engine}"
        );
        assert_eq!(
            child.is_subset_of(parent.as_ref()),
            program.is_subset_of(parent_program),
            "{engine}"
        );
        assert_eq!(
            child.fork().canonical_hash(),
            reference.canonical_hash(),
            "{engine}: fork"
        );
        let created = child.into_program().create_kernel();
        assert_eq!(
            created.canonical_hash(),
            reference.canonical_hash(),
            "{engine}: created"
        );
    }
}

#[test]
fn equivalence_and_subset_answer_alike_on_every_engine() {
    let reference_parent = build(PARENT, Engine::Interpreter(JitMode::Off));
    let parent_program = reference_parent
        .as_interpreter()
        .expect("interpreter")
        .program();
    let empty = child_under(reference_parent.as_ref(), EMPTY_CHILD, &[]);
    let empty_program = empty.as_interpreter().expect("interpreter").program();
    assert!(empty_program.is_subset_of(parent_program));
    for engine in every_engine() {
        let parent = build(PARENT, engine);
        let child = child_under(parent.as_ref(), EMPTY_CHILD, &[]);
        assert!(child.is_subset_of(parent.as_ref()), "{engine}");
        assert!(parent.is_subset_of(parent.as_ref()), "{engine}: itself");
        assert!(
            !build(CHILD, engine).is_subset_of(parent.as_ref()),
            "{engine}: a child with outputs of its own"
        );
        assert!(
            parent.is_equivalent_to(reference_parent.as_ref()),
            "{engine}"
        );
    }
}

// ── Graphs built without the DSL ───────────────────────────────────

/// `out := mod(hash(cycle), m)`, assembled by hand.
fn hand_built(m: u64) -> PolydatAssembler {
    let ctx = BuildContext::default();
    let mut asm = PolydatAssembler::new(vec!["cycle".into()]);
    let hash = build_node(&ctx, "hash", &[WireRef::input("cycle")], &[], &[]).expect("hash");
    asm.add_node("h", hash, vec![WireRef::input("cycle")]);
    let modulo =
        build_node(&ctx, "mod", &[WireRef::node("h")], &[], &[ConstArg::Int(m)]).expect("mod");
    asm.add_node("out", modulo, vec![WireRef::node("h")]);
    asm.add_output("out", WireRef::node("out"));
    asm
}

#[test]
fn a_hand_built_graph_hashes_as_the_program_that_spells_it() {
    let src = "input cycle: u64\nh := hash(cycle)\nout := mod(h, 1000)\n";
    let from_source = build(src, Engine::Interpreter(JitMode::Off)).canonical_hash();
    // The source also outputs `h`, and outputs its input `cycle` by name
    // through a port passthrough, as the compiler does every input; the
    // hand-built graph hashes alike once it declares the same outputs,
    // and differs until then.
    for engine in every_engine() {
        let without = hand_built(1000)
            .compile_with(engine)
            .unwrap_or_else(|e| panic!("{engine}: {e}"))
            .canonical_hash();
        assert_ne!(without, from_source, "{engine}");
        let mut asm = hand_built(1000);
        asm.add_output("h", WireRef::node("h"));
        asm.add_node(
            "__port_cycle",
            Box::new(PortPassthrough::new("cycle", PortType::U64)),
            vec![WireRef::input("cycle")],
        );
        asm.add_output("cycle", WireRef::node("__port_cycle"));
        let spelled = asm
            .compile_with(engine)
            .unwrap_or_else(|e| panic!("{engine}: {e}"))
            .canonical_hash();
        assert_eq!(spelled, from_source, "{engine}");
    }
}

// ── The raw tier constructors ──────────────────────────────────────

/// The raw constructors build a compiled kernel from the graph an
/// assembler holds. That graph is the one the normal path hashes, so a
/// raw kernel reports the canonical hash the same program has on every
/// engine. The assembler holds no `for` traversal (the DSL compile
/// attaches those after assembly), so a program with one is compared
/// by its graph alone and is left out here.
#[cfg(feature = "bench-tiers")]
mod raw {
    use super::*;
    use polydat::dsl::compile::compile_polydat_to_assembler;

    fn raw_hashes(asm: impl Fn() -> PolydatAssembler) -> Vec<(&'static str, [u8; 32])> {
        let mut all = vec![(
            "closures",
            asm()
                .compile_closures_raw()
                .expect("closures")
                .canonical_hash(),
        )];
        #[cfg(feature = "jit")]
        {
            all.push((
                "native",
                asm().compile_native_raw().expect("native").canonical_hash(),
            ));
            all.push((
                "pure native",
                asm()
                    .compile_pure_native_raw()
                    .expect("pure native")
                    .canonical_hash(),
            ));
        }
        all
    }

    #[test]
    fn a_raw_built_program_hashes_as_the_same_program_through_the_dsl() {
        for src in PROGRAMS.iter().filter(|src| !src.contains("for ")) {
            let reference = build(src, Engine::Interpreter(JitMode::Off)).canonical_hash();
            for (tier, hash) in raw_hashes(|| compile_polydat_to_assembler(src).expect("assemble"))
            {
                assert_eq!(hash, reference, "{tier}\n{src}");
            }
        }
    }

    #[test]
    fn two_different_raw_built_graphs_hash_differently() {
        let a = raw_hashes(|| hand_built(1000));
        let b = raw_hashes(|| hand_built(1001));
        for ((tier, a), (_, b)) in a.iter().zip(&b) {
            assert_ne!(a, b, "{tier}");
            assert_ne!(*a, [0; 32], "{tier}: a raw kernel has a graph digest");
        }
        // The same graph on every raw tier and on the normal path.
        let normal = hand_built(1000)
            .compile_with(Engine::Closures(Provenance::Auto))
            .expect("closures")
            .canonical_hash();
        for (tier, hash) in &a {
            assert_eq!(*hash, normal, "{tier}");
        }
    }
}
