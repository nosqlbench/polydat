// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Expression stubs and scoped expressions on every engine
//! (polydat_grammar_programmatic.md §11): a stub parsed once and bound
//! under a parent runs on the parent's engine, `set` refuses what
//! `set_input` refuses, and a typed extern starts at its own type's zero.

use polydat::ast::{PortType, Value};
use polydat::dsl::compile::{CompileOptions, compile_polydat_with_engine};
use polydat::dsl::stub::{ExprStub, GraphMatter, ScopedExpr};
use polydat::kernel::WriteError;
use polydat::{Engine, JitMode, Kernel, Provenance};

fn every_engine() -> Vec<Engine> {
    let mut all = vec![
        Engine::Interpreter(JitMode::Auto),
        Engine::Closures(Provenance::Auto),
    ];
    if cfg!(feature = "jit") {
        all.push(Engine::Native(Provenance::Auto));
        all.push(Engine::PureNative(Provenance::Auto));
    }
    all
}

fn parent(engine: Engine) -> Box<dyn Kernel> {
    compile_polydat_with_engine(
        "input cycle: u64\nbase := 40\n",
        engine,
        &CompileOptions::default(),
        None,
    )
    .unwrap_or_else(|e| panic!("{engine}: {e}"))
}

fn predicate(parent: &dyn Kernel) -> ScopedExpr {
    let mut matter = GraphMatter::new();
    matter.extern_wire::<u64>("threshold").bind(
        ExprStub::parse("__pred", "threshold > 50")
            .expect("parse")
            .returning::<u64>()
            .volatile(),
    );
    ScopedExpr::bind(parent, "__pred", matter).expect("bind")
}

#[test]
fn a_scoped_expression_runs_on_its_parents_engine() {
    for engine in every_engine() {
        let parent = parent(engine);
        let mut pred = predicate(parent.as_ref());
        assert_eq!(pred.kernel().engine(), parent.engine(), "{engine}");
        assert!(
            pred.set("threshold", Value::U64(100)).unwrap().is_true(),
            "{engine}"
        );
        assert!(
            !pred.set("threshold", Value::U64(10)).unwrap().is_true(),
            "{engine}"
        );
        // A value of another type converts by the one conversion rule.
        assert!(
            pred.set("threshold", Value::Str("70".into()))
                .unwrap()
                .is_true(),
            "{engine}"
        );
    }
}

#[test]
fn set_refuses_what_set_input_refuses() {
    for engine in every_engine() {
        let parent = parent(engine);
        let mut pred = predicate(parent.as_ref());
        assert!(
            matches!(
                pred.set("nope", Value::U64(1)),
                Err(WriteError::UnknownWire { .. })
            ),
            "{engine}: an unknown name"
        );
        // An expression that reads the parent's coordinate has it as a
        // coordinate of its own.
        let mut matter = GraphMatter::new();
        matter.bind(ExprStub::parse("__c", "u64_add(cycle, 1)").expect("parse"));
        let mut reads_cycle = ScopedExpr::bind(parent.as_ref(), "__c", matter).expect("bind");
        let coordinate = reads_cycle.set("cycle", Value::U64(1)).err();
        assert!(
            matches!(coordinate, Some(WriteError::CoordinateSlot { .. })),
            "{engine}: a coordinate: {coordinate:?}"
        );
        assert!(
            matches!(
                pred.set("threshold", Value::Str("not a number".into())),
                Err(WriteError::TypeMismatch {
                    expected: PortType::U64,
                    got: PortType::Str,
                    ..
                })
            ),
            "{engine}: a value that does not convert"
        );
    }
}

#[test]
fn a_typed_extern_starts_at_its_own_types_zero() {
    for engine in every_engine() {
        let parent = parent(engine);
        let mut matter = GraphMatter::new();
        matter
            .extern_wire::<u64>("n")
            .extern_wire::<f64>("x")
            .extern_wire_typed("s", PortType::Str)
            .extern_wire_typed("b", PortType::Bool)
            .bind(ExprStub::parse("__n", "n").unwrap().returning::<u64>())
            .bind(ExprStub::parse("__x", "x").unwrap().returning::<f64>())
            .bind(ExprStub::parse("__s", "s").unwrap())
            .bind(ExprStub::parse("__b", "b").unwrap());
        let mut scoped = ScopedExpr::bind(parent.as_ref(), "__s", matter)
            .unwrap_or_else(|e| panic!("{engine}: {e}"));
        let k = scoped.kernel();
        assert_eq!(k.pull("__n"), Value::U64(0), "{engine}");
        assert_eq!(k.pull("__x"), Value::F64(0.0), "{engine}");
        assert_eq!(k.pull("__s"), Value::Str("".into()), "{engine}");
        assert_eq!(k.pull("__b"), Value::Bool(false), "{engine}");
        scoped.set("s", Value::Str("set".into())).unwrap();
        assert_eq!(scoped.eval(), Value::Str("set".into()), "{engine}");
    }
}

#[test]
fn returning_casts_and_is_true_reads_each_truth_form() {
    for engine in every_engine() {
        let parent = parent(engine);
        let mut matter = GraphMatter::new();
        matter
            .extern_wire::<u64>("n")
            .bind(ExprStub::parse("__half", "n").unwrap().returning::<f64>());
        let mut half = ScopedExpr::bind(parent.as_ref(), "__half", matter).unwrap();
        assert!(!half.is_true(), "{engine}: F64 zero is false");
        assert_eq!(
            half.set("n", Value::U64(3)).unwrap().eval(),
            Value::F64(3.0),
            "{engine}"
        );
        assert!(half.is_true(), "{engine}: F64 nonzero is true");
    }
}
