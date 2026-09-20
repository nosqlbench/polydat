// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! One compile path behind every entry point. A source string, a parsed
//! file, or an assembler a host built by hand reaches the same graph and
//! the same kernel on every engine; the options mean the same thing
//! whichever entry point carries them; and what the interpreter refuses
//! under strict, every engine refuses.

use polydat::ast::Value;
use polydat::compile::assembly::{PolydatAssembler, WireRef};
use polydat::dsl::compile::{
    CompileOptions, compile_polydat_interpreter, compile_polydat_interpreter_with_log,
    compile_polydat_interpreter_with_options, compile_polydat_kernel,
    compile_polydat_kernel_with_options, compile_polydat_to_assembler,
    compile_polydat_to_assembler_with, compile_polydat_with, compile_polydat_with_engine,
};
use polydat::dsl::events::CompileEventLog;
use polydat::{Engine, JitMode, Kernel, KernelError, Provenance};

const SRC: &str = "input cycle: u64\nh := hash(cycle)\nk := mod_wire(h, 7)\ns := \"k={k}\"\n";

/// Every engine a build can name; a refusal is skipped, as elsewhere.
fn engines() -> Vec<Engine> {
    let mut all = vec![
        Engine::Interpreter(JitMode::Auto),
        Engine::Closures(Provenance::Auto),
        Engine::Closures(Provenance::Raw),
    ];
    if cfg!(feature = "jit") {
        all.push(Engine::Native(Provenance::Auto));
        all.push(Engine::Native(Provenance::Raw));
        all.push(Engine::PureNative(Provenance::Auto));
        all.push(Engine::PureNative(Provenance::Raw));
    }
    all
}

fn values(k: &mut dyn Kernel, cycle: u64) -> Vec<Value> {
    k.set_inputs(&[cycle]);
    k.output_names().iter().map(|o| k.pull(o)).collect()
}

#[test]
fn every_entry_point_reaches_the_same_kernel() {
    let mut oracle = compile_polydat_interpreter(SRC).unwrap();
    let want: Vec<Value> = values(&mut oracle, 5);
    let options = CompileOptions::default();
    let mut log = CompileEventLog::new();
    let mut kernels: Vec<(&str, Box<dyn Kernel>)> = vec![
        (
            "compile_polydat_kernel",
            compile_polydat_kernel(SRC).unwrap(),
        ),
        (
            "compile_polydat_kernel_with_options",
            compile_polydat_kernel_with_options(SRC, &options, Some(&mut log)).unwrap(),
        ),
        (
            "compile_polydat_interpreter_with_options",
            Box::new(compile_polydat_interpreter_with_options(SRC, &options, None).unwrap()),
        ),
        (
            "compile_polydat_interpreter_with_log",
            Box::new(
                compile_polydat_interpreter_with_log(SRC, &mut CompileEventLog::new()).unwrap(),
            ),
        ),
        (
            "compile_polydat_to_assembler",
            compile_polydat_to_assembler(SRC)
                .unwrap()
                .compile_kernel()
                .unwrap(),
        ),
        (
            "compile_polydat_to_assembler_with",
            Box::new(
                compile_polydat_to_assembler_with(SRC, &options)
                    .unwrap()
                    .compile()
                    .unwrap(),
            ),
        ),
    ];
    for engine in engines() {
        if let Ok(k) = compile_polydat_with(SRC, engine) {
            kernels.push(("compile_polydat_with", k));
        }
        if let Ok(k) = compile_polydat_with_engine(SRC, engine, &options, None) {
            kernels.push(("compile_polydat_with_engine", k));
        }
        if let Ok(k) = compile_polydat_to_assembler(SRC)
            .unwrap()
            .compile_with(engine)
        {
            kernels.push(("PolydatAssembler::compile_with", k));
        }
    }
    for (name, k) in kernels.iter_mut() {
        assert_eq!(
            k.output_names(),
            oracle.output_names(),
            "{name} on {}",
            k.engine()
        );
        assert_eq!(values(k.as_mut(), 5), want, "{name} on {}", k.engine());
    }
}

/// The assembler a host builds by hand is the same first-class road to a
/// kernel: nodes and wires added one by one compile on every engine to
/// what the same graph from source computes.
#[test]
fn a_hand_built_assembler_compiles_on_every_engine() {
    fn build() -> PolydatAssembler {
        let mut asm = PolydatAssembler::new(vec!["cycle".into()]);
        use polydat::ast::PortType;
        let hash = polydat::dsl::factory::build_node(
            "hash",
            &[WireRef::input("cycle")],
            &[PortType::U64],
            &[],
        )
        .unwrap();
        asm.add_node("h", hash, vec![WireRef::input("cycle")]);
        let m = polydat::dsl::factory::build_node(
            "mod",
            &[WireRef::node("h")],
            &[PortType::U64],
            &[polydat::dsl::ConstArg::Int(7)],
        )
        .unwrap();
        asm.add_node("k", m, vec![WireRef::node("h")]);
        asm.add_output("h", WireRef::node("h"));
        asm.add_output("k", WireRef::node("k"));
        asm
    }
    fn hk(k: &mut dyn Kernel) -> (Value, Value) {
        k.set_inputs(&[9]);
        (k.pull("h"), k.pull("k"))
    }
    let mut from_source =
        compile_polydat_interpreter("input cycle: u64\nh := hash(cycle)\nk := mod(h, 7)\n")
            .unwrap();
    let want = hk(&mut from_source);
    let mut p1 = build().compile().unwrap();
    assert_eq!(hk(&mut p1), want);
    for engine in engines() {
        let mut k = match build().compile_with(engine) {
            Ok(k) => k,
            Err(KernelError::Refused { .. }) => continue,
            Err(e) => panic!("{engine}: {e}"),
        };
        assert_eq!(hk(k.as_mut()), want, "{engine}");
    }
    let mut default = build().compile_kernel().unwrap();
    // The default engine, in the provenance mode the selector chose.
    assert!(matches!(
        (default.engine(), Engine::default()),
        (Engine::Native(_), Engine::Native(_)) | (Engine::Closures(_), Engine::Closures(_))
    ));
    assert_eq!(hk(default.as_mut()), want);
}

/// What strict refuses, it refuses on every engine: an implicit adapter
/// between a float wire and an integer port is an error under
/// `strict: true` whichever engine the options reach.
#[test]
fn strict_refuses_implicit_coercions_on_every_engine() {
    let src = "input cycle: u64\nh := hash(cycle)\nf := sqrt(h)\n";
    let strict = CompileOptions {
        strict: true,
        ..CompileOptions::default()
    };
    assert!(compile_polydat_interpreter_with_options(src, &strict, None).is_err());
    for engine in engines() {
        let outcome = compile_polydat_with_engine(src, engine, &strict, None);
        assert!(
            matches!(
                outcome,
                Err(KernelError::Assembly(_)) | Err(KernelError::Source(_))
            ),
            "{engine}: strict must refuse the implicit coercion, got {:?}",
            outcome.map(|k| k.engine())
        );
        let lax = compile_polydat_with_engine(src, engine, &CompileOptions::default(), None);
        assert!(
            lax.is_ok() || matches!(lax, Err(KernelError::Refused { .. })),
            "{engine}: {:?}",
            lax.err()
        );
    }
}

/// The extended API is a subtrait, not a hidden corner of the core one.
///
/// `compile_slots` hands back a `Box<dyn SlotKernel>`, so a caller
/// reaches the slot surface — a buffer slot, a raw `u64`, an evaluation
/// that returns one — without naming a kernel type, and the same box
/// upcasts to `Box<dyn Kernel>` wherever the ordinary surface will do.
/// The interpreter is refused rather than half-answered: it holds typed
/// `Value` buffers and has no slot to name.
#[test]
fn the_slot_surface_is_a_subtrait_reachable_without_a_kernel_type() {
    use polydat::SlotKernel;

    let mut oracle = compile_polydat_interpreter(SRC).unwrap();
    let want = values(&mut oracle, 5);

    for engine in engines() {
        if matches!(engine, Engine::Interpreter(_)) {
            // Honestly refused, with a reason that says why and what to
            // ask for instead.
            match compile_polydat_to_assembler(SRC)
                .unwrap()
                .compile_slots(engine)
            {
                Err(KernelError::Refused { reason, .. }) => {
                    assert!(reason.contains("slot"), "{reason}")
                }
                Err(e) => panic!("expected a refusal naming the slot buffer, got {e}"),
                Ok(_) => panic!("the interpreter has no slot buffer to hand back"),
            }
            continue;
        }
        let mut k = match compile_polydat_to_assembler(SRC)
            .unwrap()
            .compile_slots(engine)
        {
            Ok(k) => k,
            Err(KernelError::Refused { .. }) => continue,
            Err(e) => panic!("{engine}: {e}"),
        };
        // The extended surface: a slot, then an evaluation that returns
        // the raw word rather than a `Value`.
        let slot = SlotKernel::resolve_output(k.as_ref(), "k").expect("a named output");
        let raw = k.eval_for_slot(&[5], slot);
        assert_eq!(raw, SlotKernel::get(k.as_ref(), "k"));

        // And the core surface, on the same box, because the extended
        // trait extends it. No second build, no downcast.
        assert_eq!(k.engine(), engine_of(engine, k.engine()));
        assert_eq!(values(k.as_mut(), 5), want, "{engine}");
        assert_eq!(k.pull("k").as_u64(), raw, "{engine}");
    }
}

/// `Auto` delegates the provenance choice, so the reported engine is
/// the tier that was asked for with whatever mode the selector took.
fn engine_of(asked: Engine, reported: Engine) -> Engine {
    match (asked, reported) {
        (Engine::Closures(Provenance::Auto), r @ Engine::Closures(_)) => r,
        (Engine::Native(Provenance::Auto), r @ Engine::Native(_)) => r,
        (Engine::PureNative(Provenance::Auto), r @ Engine::PureNative(_)) => r,
        _ => asked,
    }
}

/// The pure native tier is an engine a host names like any other: it
/// arrives through the same options, it reports itself rather than the
/// hybrid it is the differential for, and a provenance mode it has no
/// kernel for is refused instead of answered with another.
#[test]
#[cfg(feature = "jit")]
fn the_pure_tier_names_itself_and_refuses_the_modes_it_lacks() {
    let mut oracle = compile_polydat_interpreter(SRC).unwrap();
    let want = values(&mut oracle, 5);

    for prov in [Provenance::Auto, Provenance::Raw, Provenance::PushPull] {
        let engine = Engine::PureNative(prov);
        let mut k = compile_polydat_with(SRC, engine).unwrap_or_else(|e| panic!("{engine}: {e}"));
        // The tier it ran, not the `Native` tier it stands behind.
        let ran = k.engine();
        assert!(
            matches!(ran, Engine::PureNative(_)),
            "{engine}: reported {ran}",
        );
        // A named mode is the mode; `Auto` delegated the choice.
        if prov != Provenance::Auto {
            assert_eq!(ran, engine, "a named mode is honoured or refused");
        }
        assert_eq!(values(k.as_mut(), 5), want, "{engine}");
    }

    // Push and pull have no pure kernel, so they are refusals naming
    // the engine, not silent substitutions of push+pull.
    for prov in [Provenance::Push, Provenance::Pull] {
        let engine = Engine::PureNative(prov);
        match compile_polydat_with(SRC, engine) {
            Err(KernelError::Refused {
                engine: named,
                reason,
            }) => {
                assert!(matches!(named, Engine::PureNative(_)), "{named}");
                assert!(reason.contains("pure native"), "{reason}");
            }
            other => panic!(
                "{engine}: expected a refusal, got {:?}",
                other.map(|k| k.engine())
            ),
        }
    }
}

/// What is knowable at build is known at build, and fails at build. A
/// step no input reaches runs once while the kernel is being built, so
/// a constant that cannot be computed reports it then rather than on
/// the first pull.
///
/// Every engine does this and reports it the same way: the same
/// `KernelError` variant, and a message carrying the node's own
/// failure. It is `ConstantFold` and not `Refused` because no engine is
/// declining a program the others accept — the value cannot be
/// computed anywhere. The pure tier keeps no step list, so its
/// constants are compiled a second time into an entry of their own and
/// run over the same buffer to reach the same rule (F-E15).
#[test]
fn every_engine_folds_its_constants_at_build() {
    let src = "input cycle: u64\nn := mod_wire(10, 0)\nv := n + cycle\n";
    for engine in engines() {
        let built = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            compile_polydat_with(src, engine).map(|k| k.engine())
        }))
        .unwrap_or_else(|_| panic!("{engine}: a build failure is an error, never a panic"));
        match built {
            Err(KernelError::ConstantFold { reason }) => assert!(
                reason.contains("zero") && reason.contains("mod_wire"),
                "{engine}: the message carries the node's own failure: {reason}",
            ),
            other => panic!(
                "{engine}: a constant that cannot be computed is a ConstantFold error at \
                 build, got {other:?}",
            ),
        }
    }
    // The same node with the divisor on a wire is not a constant, so
    // nothing runs it at build and every engine accepts the program.
    // The rule is about what is knowable at build, not about the node.
    let dynamic = "input cycle: u64\nn := mod_wire(10, cycle)\n";
    for engine in engines() {
        assert!(
            compile_polydat_with(dynamic, engine).is_ok(),
            "{engine}: a dynamic divisor is not a compile-time constant",
        );
    }
}

/// `Native` and `PureNative` differ in one thing: what becomes of a node
/// with no native form. `Native` runs that node's closure and always
/// succeeds, so it can never tell a host whether its program went fully
/// native; the pure tier refuses and names the node, which is the whole
/// reason to ask for it. Every node in this library lowers today, so the
/// two agree here — what this pins is that they agree in value and that
/// each reports its own tier.
#[test]
#[cfg(feature = "jit")]
fn the_two_native_tiers_agree_in_value_and_differ_in_name() {
    let mut hybrid = compile_polydat_with(SRC, Engine::Native(Provenance::Auto)).unwrap();
    let mut pure = compile_polydat_with(SRC, Engine::PureNative(Provenance::Auto)).unwrap();
    assert!(matches!(hybrid.engine(), Engine::Native(_)));
    assert!(matches!(pure.engine(), Engine::PureNative(_)));
    assert_eq!(values(hybrid.as_mut(), 9), values(pure.as_mut(), 9));
}

/// The logged interpreter compile is the plain one with a log: a program
/// with a traversal compiles through it, as it does everywhere else.
#[test]
fn the_logged_compile_accepts_a_traversal() {
    let src = "input cycle: u64\nfor k in 1..3 {\n  y := k * 10\n}\n";
    let mut log = CompileEventLog::new();
    let k = compile_polydat_interpreter_with_log(src, &mut log)
        .expect("a for statement compiles with a log");
    assert_eq!(k.traversals().len(), 1);
}

/// A cursor extent computed from a constant expression is resolved on
/// every engine, not only on the interpreter: `cursor_schemas` reports
/// the same extent whichever engine built the kernel.
#[test]
fn deferred_cursor_extents_resolve_on_every_engine() {
    let src = "input cycle: u64\nn := 10 + 5\ncursor q = range(0, n)\nv := hash(q.ordinal)\n";
    let want = compile_polydat_interpreter(src).unwrap().cursor_schemas()[0].extent;
    assert_eq!(want, Some(15));
    for engine in engines() {
        let k = match compile_polydat_with(src, engine) {
            Ok(k) => k,
            Err(KernelError::Refused { .. }) => continue,
            Err(e) => panic!("{engine}: {e}"),
        };
        assert_eq!(k.cursor_schemas()[0].extent, want, "{engine}");
    }
}

/// A module the program defines is visible inside its `for` bodies
/// whichever way the program was compiled: from a string with no source
/// directory, the body resolves the module from the program, not from a
/// file on disk.
#[test]
fn a_program_module_is_visible_inside_a_traversal_body() {
    let src = "\
input cycle: u64
twice(x: u64) -> (y: u64) := {
  y := x * 2
}
for k in 1..4 {
  d := twice(k)
}
";
    for engine in engines() {
        let mut k = match compile_polydat_with(src, engine) {
            Ok(k) => k,
            Err(KernelError::Refused { .. }) => continue,
            Err(e) => panic!("{engine}: {e}"),
        };
        k.set_inputs(&[0]);
        let streams = k.traverse_all().unwrap();
        let mut seen = Vec::new();
        for s in &streams {
            for i in 0..s.len() {
                let mut act = s.activation(i).unwrap();
                act.kernel.set_inputs(&[0]);
                seen.push(act.kernel.pull_ref("d").as_u64());
            }
        }
        assert_eq!(seen, vec![2, 4, 6], "{engine}");
    }
}
