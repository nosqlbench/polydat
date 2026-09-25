// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Scope trees on all four engines (native_scope_trees.md §8).
//!
//! A host that runs workloads as a tree of scopes (params, a `set:`
//! scope, a phase, fibers, per-op children, iteration children) builds
//! it through the `Kernel` trait alone, on whichever engines it names,
//! and gets the same answers everywhere. Each test here is written in
//! the host's terms and runs every parent engine against every child
//! engine, so a tree mid-migration, part interpreter and part compiled,
//! is covered too.

use polydat::ast::Value;
use polydat::dsl::compile::compile_polydat_with;
use polydat::kernel::interp::{KernelLookup, Lookup};
use polydat::kernel::subcontext::{
    BodyFragment, Child, RootMarker, ScopeModule, SourceContext, SubcontextBuilder,
};
use polydat::kernel::{bind_under, propagate_inputs};
use polydat::{Engine, JitMode, Kernel, Provenance};

fn engines() -> Vec<Engine> {
    let mut all = vec![
        Engine::Interpreter(JitMode::Auto),
        Engine::Closures(Provenance::PushPull),
        Engine::Closures(Provenance::Raw),
    ];
    if cfg!(feature = "jit") {
        all.push(Engine::Native(Provenance::PushPull));
        all.push(Engine::Native(Provenance::Raw));
    }
    all
}

fn compile(src: &str, engine: Engine) -> Box<dyn Kernel> {
    compile_polydat_with(src, engine).unwrap_or_else(|e| panic!("{engine}: {e}"))
}

/// A module of `source` built under `parent`.
fn module(parent: &dyn Kernel, label: &str, source: &str) -> ScopeModule<Child<RootMarker>> {
    let mut b = SubcontextBuilder::under(parent);
    b.context(SourceContext::new(label));
    b.body(BodyFragment::PolydatSource(source.to_string()));
    b.finalize().unwrap_or_else(|e| panic!("{label}: {e:?}"))
}

fn str_value(s: &str) -> Value {
    Value::Str(s.into())
}

const ROOT: &str = "const mode := \"default\"\nconst size := \"small\"\nextern shard: u64 = 3\n";
const SET: &str = "extern size: String\nconst mode := \"mode_for_{size}\"\n";
const PHASE: &str = "input cycle: u64\nextern mode: String\nextern size: String\n\
                     extern shard: u64\nextern retries: u64 = 2\n\
                     label := \"{mode}/{size}/{shard}\"\ntries := u64_add(retries, cycle)\n";
const ITER: &str = "extern mode: String\nextern i: u64\nrow := \"{mode}-{i}\"\n";

/// params root → `set:` scope → phase → fork → iteration child, the
/// shape nmbrs builds, with a const in the `set:` scope that shadows a
/// parameter (the 2026-09-24 const-shadow case), a parameter carried by
/// `propagate_inputs`, a reset, and an iteration binding.
#[test]
fn a_scope_tree_answers_alike_on_every_engine_pair() {
    for parent_engine in engines() {
        for child_engine in engines() {
            let pair = format!("{parent_engine} → {child_engine}");
            let root = compile(ROOT, parent_engine);

            let set = module(root.as_ref(), "set", SET)
                .instantiate_under(root.as_ref(), child_engine, &[])
                .unwrap_or_else(|e| panic!("{pair}: {e}"));
            assert_eq!(
                KernelLookup::new(set.as_ref()).lookup("mode"),
                Some(str_value("mode_for_small")),
                "{pair}: the set scope's own const"
            );

            let mut phase = module(set.as_ref(), "phase", PHASE)
                .instantiate_under(set.as_ref(), child_engine, &[])
                .unwrap_or_else(|e| panic!("{pair}: {e}"));
            propagate_inputs(root.as_ref(), phase.as_mut())
                .unwrap_or_else(|e| panic!("{pair}: {e}"));
            phase.set_inputs(&[1]);
            assert_eq!(
                phase.pull("label"),
                str_value("mode_for_small/small/3"),
                "{pair}: the phase reads the shadowing const and the propagated parameter"
            );

            // A fork carries the state and the program.
            let mut fiber = phase.fork();
            assert_eq!(fiber.program_id(), phase.program_id(), "{pair}");
            assert_eq!(
                fiber.pull("label"),
                str_value("mode_for_small/small/3"),
                "{pair}"
            );

            // A reset puts externs back and leaves coordinates alone.
            let retries = fiber.input_index("retries").expect("declared");
            fiber
                .set_input_at(retries, Value::U64(9))
                .unwrap_or_else(|e| panic!("{pair}: {e}"));
            assert_eq!(
                fiber.pull("tries"),
                Value::U64(10),
                "{pair}: after the write"
            );
            fiber.reset_inputs();
            assert_eq!(fiber.input_value_at(retries), Some(Value::U64(2)), "{pair}");
            assert_eq!(
                fiber.pull("tries"),
                Value::U64(3),
                "{pair}: after the reset"
            );
            assert_eq!(
                fiber.input_value_at(0),
                Some(Value::U64(1)),
                "{pair}: the coordinate"
            );

            // The fork's writes are its own.
            assert_eq!(phase.pull("tries"), Value::U64(3), "{pair}: the original");

            // An iteration child per tuple.
            let iter = module(phase.as_ref(), "iter", ITER);
            for i in 0..3u64 {
                let mut child = iter
                    .instantiate_under(phase.as_ref(), child_engine, &[("i".into(), Value::U64(i))])
                    .unwrap_or_else(|e| panic!("{pair}: {e}"));
                assert_eq!(
                    child.pull("row"),
                    str_value(&format!("mode_for_small-{i}")),
                    "{pair}: iteration {i}"
                );
            }
        }
    }
}

/// A plan of pre-resolved indices, sealed on a program's identity,
/// holds for every kernel bound from that program and every fork, and
/// a second compile of the same source is a different program.
#[test]
fn program_identity_seals_a_plan_across_binds_and_forks() {
    for engine in engines() {
        let parent = compile(ROOT, engine);
        let child_src = "extern shard: u64\nout := u64_add(shard, 1)\n";
        let program = compile(child_src, engine).into_program();
        let bound = bind_under(parent.as_ref(), program.clone(), &[]).expect("binds");
        assert_eq!(bound.program_id(), program.program_id(), "{engine}");
        assert_eq!(bound.fork().program_id(), program.program_id(), "{engine}");
        let again = compile(child_src, engine).into_program();
        assert_ne!(again.program_id(), program.program_id(), "{engine}");
    }
}

/// The analysis program a module holds and every engine's kernel of it
/// agree on input and output positions, so an index resolved on one is
/// valid on the other.
#[test]
fn a_module_s_program_and_its_kernels_agree_on_positions() {
    let root = compile(ROOT, Engine::Interpreter(JitMode::Auto));
    let phase = module(root.as_ref(), "phase", PHASE);
    let inputs: Vec<String> = phase.program().input_names();
    let outputs: Vec<String> = phase
        .program()
        .output_names()
        .into_iter()
        .map(String::from)
        .collect();
    for engine in engines() {
        let kernel = phase
            .instantiate_under(root.as_ref(), engine, &[])
            .unwrap_or_else(|e| panic!("{engine}: {e}"));
        assert_eq!(kernel.input_names(), inputs, "{engine}: inputs");
        assert_eq!(kernel.output_names(), outputs, "{engine}: outputs");
        assert_eq!(kernel.coord_count(), 1, "{engine}: `cycle`");
        assert_eq!(
            kernel.input_default_at(kernel.input_index("retries").unwrap()),
            Some(Value::U64(2)),
            "{engine}"
        );
    }
}

/// A value the child's declared input refuses is an error naming both,
/// not a skip.
#[test]
fn propagate_inputs_reports_a_refusal() {
    for engine in engines() {
        let parent = compile("extern shard: String = \"a\"\n", engine);
        let mut child = compile("extern shard: u64 = 1\nout := shard\n", engine);
        let err = propagate_inputs(parent.as_ref(), child.as_mut())
            .expect_err("a string into a u64 input");
        assert!(format!("{err}").contains("shard"), "{engine}: {err}");
    }
}

/// A Rule 2 write-through commits into the parent's shared cell on
/// every engine.
#[test]
fn a_write_through_commits_into_the_parent_s_cell() {
    for parent_engine in engines() {
        for child_engine in engines() {
            let pair = format!("{parent_engine} → {child_engine}");
            let mut root = compile("input cycle: u64\nshared total := 0\n", parent_engine);
            let mut b = SubcontextBuilder::under(root.as_ref());
            b.context(SourceContext::new("op"));
            b.body(BodyFragment::PolydatSource(
                "input cycle: u64\n".to_string(),
            ));
            b.add_result_bindings("total := u64_add(total, 5)\n")
                .unwrap_or_else(|e| panic!("{pair}: {e:?}"));
            let op = b.finalize().unwrap_or_else(|e| panic!("{pair}: {e:?}"));
            let mut child = op
                .instantiate_under(root.as_ref(), child_engine, &[])
                .unwrap_or_else(|e| panic!("{pair}: {e}"));
            child.set_inputs(&[0]);
            child
                .commit_write_throughs()
                .unwrap_or_else(|e| panic!("{pair}: {e}"));
            root.set_inputs(&[0]);
            assert_eq!(root.pull("total"), Value::U64(5), "{pair}");
        }
    }
}

/// A child bound to a parent's output reads `None` until the parent
/// computes it, then what the parent computed, whether the parent was
/// pulled by name or by index, on every engine pair.
#[test]
fn a_parent_s_output_reaches_its_child_by_either_pull() {
    let parent_src = "input cycle: u64\ntotal := u64_add(cycle, 20)\n";
    for parent_engine in engines() {
        for child_engine in engines() {
            let pair = format!("{parent_engine} → {child_engine}");
            for by_index in [false, true] {
                let mut root = compile(parent_src, parent_engine);
                let child = module(root.as_ref(), "op", "extern total: u64\n")
                    .instantiate_under(root.as_ref(), child_engine, &[])
                    .unwrap_or_else(|e| panic!("{pair}: {e}"));
                let total = child.input_index("total").expect("declared");
                assert_eq!(
                    child.input_value_at(total),
                    Some(Value::None),
                    "{pair}: before the parent computes it"
                );
                root.set_inputs(&[1]);
                let value = if by_index {
                    let at = root.output_index("total").expect("an output");
                    root.pull_at(at)
                } else {
                    root.pull("total")
                };
                assert_eq!(value, Value::U64(21), "{pair}");
                assert_eq!(
                    child.input_value_at(total),
                    Some(Value::U64(21)),
                    "{pair}: after a pull {}",
                    if by_index { "by index" } else { "by name" }
                );
            }
        }
    }
}

/// One parent shared across threads, with children bound and forks
/// taken under it from all of them at once.
#[test]
fn many_threads_bind_and_fork_under_one_shared_parent() {
    for engine in engines() {
        let mut root = compile(ROOT, engine);
        root.set_inputs(&[0]);
        let root: std::sync::Arc<dyn Kernel> = root.into();
        let iter = std::sync::Arc::new(module(
            root.as_ref(),
            "iter",
            "extern mode: String\nextern i: u64\nrow := \"{mode}-{i}\"\n",
        ));
        let handles: Vec<_> = (0..8u64)
            .map(|t| {
                let root = root.clone();
                let iter = iter.clone();
                std::thread::spawn(move || {
                    let mut child = iter
                        .instantiate_under(root.as_ref(), engine, &[("i".into(), Value::U64(t))])
                        .expect("binds");
                    let row = child.pull("row");
                    let mut fork = root.fork();
                    (row, fork.pull("mode"))
                })
            })
            .collect();
        for (t, h) in handles.into_iter().enumerate() {
            let (row, mode) = h.join().expect("no panic");
            assert_eq!(row, str_value(&format!("default-{t}")), "{engine}");
            assert_eq!(mode, str_value("default"), "{engine}");
        }
    }
}
