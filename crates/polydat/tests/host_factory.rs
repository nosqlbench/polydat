// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! A host's own `NodeFactory` (library_catalog.md, "Host-registered
//! nodes"): linked into the registry with a `FactoryRegistration`, it is
//! listed with the built-in library, and the compiler builds every call
//! to one of its functions through its `build`, on every engine.
//!
//! The factory is linked into the process's registry, so this file is a
//! test binary of its own.

use std::sync::Mutex;

use polydat::ast::{PolydatNode, PortType, SlotType, Value};
use polydat::compile::assembly::WireRef;
use polydat::dsl::compile::{CompileOptions, compile_polydat_with_engine};
use polydat::dsl::factories::{NodeFactory, PolydatRuntime};
use polydat::dsl::factory::{BuildContext, ConstArg, build_node};
use polydat::dsl::registry::{
    Arity, FactoryRegistration, FuncCategory, FuncSig, OutputType, ParamSpec, lookup,
};
use polydat::{Engine, JitMode, Provenance};

/// `counted_mod(x, m)`: `mod(x, m)`, built by the host. The factory
/// records the binding each build was for, so a test sees which
/// compiles called it, and refuses a zero modulus in `validate`.
struct CountingFactory {
    built_for: Mutex<Vec<String>>,
}

static SIGS: &[FuncSig] = &[FuncSig {
    name: "counted_mod",
    category: FuncCategory::Arithmetic,
    outputs: 1,
    description: "mod, built by a host factory that records its builds",
    help: "",
    identity: None,
    variadic_ctor: None,
    params: &[
        ParamSpec {
            name: "x",
            slot_type: SlotType::Wire,
            required: true,
            example: "cycle",
            constraint: None,
        },
        ParamSpec {
            name: "m",
            slot_type: SlotType::ConstU64,
            required: true,
            example: "7",
            constraint: None,
        },
    ],
    arity: Arity::Fixed,
    commutativity: polydat::ast::Commutativity::Positional,
    default_resolver: None,
    output_type: OutputType::Fixed,
    output_port: Some(PortType::U64),
}];

impl NodeFactory for CountingFactory {
    fn signatures(&self) -> &[FuncSig] {
        SIGS
    }

    fn validate(&self, _name: &str, consts: &[ConstArg]) -> Result<(), String> {
        match consts.first() {
            Some(ConstArg::Int(0)) => Err("the modulus must not be zero".into()),
            _ => Ok(()),
        }
    }

    fn build(
        &self,
        ctx: &BuildContext,
        _name: &str,
        wires: &[WireRef],
        wire_types: &[PortType],
        consts: &[ConstArg],
    ) -> Result<Box<dyn PolydatNode>, String> {
        self.built_for
            .lock()
            .unwrap()
            .push(ctx.binding().unwrap_or("").to_string());
        build_node(ctx, "mod", wires, wire_types, consts)
    }
}

static FACTORY: CountingFactory = CountingFactory {
    built_for: Mutex::new(Vec::new()),
};

polydat::inventory::submit! {
    FactoryRegistration { factory: &FACTORY }
}

fn every_engine() -> Vec<Engine> {
    let mut all = vec![
        Engine::Interpreter(JitMode::Off),
        Engine::Closures(Provenance::Auto),
    ];
    if cfg!(feature = "jit") {
        all.push(Engine::Native(Provenance::Auto));
        all.push(Engine::PureNative(Provenance::Auto));
    }
    all
}

fn built_for(binding: &str) -> bool {
    FACTORY
        .built_for
        .lock()
        .unwrap()
        .iter()
        .any(|b| b == binding)
}

#[test]
fn the_compiler_builds_a_host_factorys_node_through_it_on_every_engine() {
    for engine in every_engine() {
        // A binding name of its own per engine, so the record says which
        // compile called the factory whatever else runs alongside.
        let binding = format!(
            "r_{}",
            format!("{engine}").replace(['(', ')', ' ', '-'], "_")
        );
        let src = format!("input cycle: u64\n{binding} := counted_mod(cycle, 7)\n");
        let mut k = compile_polydat_with_engine(&src, engine, &CompileOptions::default(), None)
            .unwrap_or_else(|e| panic!("{engine}: {e}"));
        assert!(
            built_for(&binding),
            "{engine}: the factory built `{binding}`"
        );
        k.set_inputs(&[23]);
        assert_eq!(k.pull(&binding), Value::U64(23 % 7), "{engine}");
    }
}

#[test]
fn a_host_factorys_validate_refuses_a_constant_before_build() {
    let src = "input cycle: u64\nrefused := counted_mod(cycle, 0)\n";
    for engine in every_engine() {
        let err = compile_polydat_with_engine(src, engine, &CompileOptions::default(), None)
            .err()
            .unwrap_or_else(|| panic!("{engine}: a zero modulus compiles"));
        assert!(
            err.to_string().contains("the modulus must not be zero"),
            "{engine}: {err}"
        );
    }
    assert!(!built_for("refused"), "build ran after validate refused");
}

#[test]
fn a_host_factorys_functions_are_listed_with_the_library() {
    assert!(lookup("counted_mod").is_some());
    let grouped = PolydatRuntime::new().by_category();
    let arithmetic = grouped
        .iter()
        .find(|(c, _)| *c == FuncCategory::Arithmetic)
        .expect("an arithmetic category");
    assert!(arithmetic.1.iter().any(|s| s.name == "counted_mod"));
    assert!(arithmetic.1.iter().any(|s| s.name == "mod"));
}
