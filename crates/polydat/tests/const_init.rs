// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! A `const` is evaluated once, when its kernel is initialized, and is
//! fixed for the kernel's life (evaluation_model.md, "Const Binding
//! Contract"). These tests hold that on all four engines, through every
//! way a kernel comes into existence.

use polydat::ast::Value;
use polydat::dsl::compile::{compile_polydat_interpreter, compile_polydat_with};
use polydat::kernel::subcontext::{BodyFragment, SubcontextBuilder};
use polydat::{Engine, JitMode, Provenance};

/// All four engines, in their default provenance mode.
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

/// A const over a volatile source is captured at initialization: reads
/// return the captured value, and only `init` captures again.
#[test]
fn a_const_captures_a_volatile_value_at_init() {
    let src = "const first := counter()\nnow := counter()\n";
    for engine in every_engine() {
        let mut k = compile_polydat_with(src, engine).unwrap_or_else(|e| panic!("{engine}: {e}"));
        let captured = k.pull("first").as_u64();
        let _ = k.pull("now");
        let _ = k.pull("now");
        assert_eq!(
            k.pull("first").as_u64(),
            captured,
            "{engine}: reads do not re-evaluate it"
        );
        k.init().unwrap();
        assert_ne!(
            k.pull("first").as_u64(),
            captured,
            "{engine}: init captures again"
        );
    }
}

/// A child scope reads its parent's captured value through the binder,
/// on every engine and however much later it is created: one session
/// origin for the whole tree, owned by the host's own binding.
#[test]
fn a_child_sees_its_parents_capture() {
    let mut parent = compile_polydat_interpreter(
        "input cycle: u64\nconst session_start := current_epoch_millis()\n",
    )
    .unwrap();
    let origin = parent.pull_ref("session_start").as_u64();
    std::thread::sleep(std::time::Duration::from_millis(20));
    let mut b = SubcontextBuilder::under(&parent);
    b.body(BodyFragment::PolydatSource(
        "input cycle: u64\nextern session_start: u64\nseen := session_start + 0\n".into(),
    ));
    let module = b.finalize().unwrap();
    for engine in every_engine() {
        let mut child = module
            .instantiate_under(&parent, engine, &[])
            .unwrap_or_else(|e| panic!("{engine}: {e}"));
        assert_eq!(
            child.pull("seen").as_u64(),
            origin,
            "{engine}: the parent's capture"
        );
    }
}

/// A const a host marks on an assembler it built is captured at
/// initialization as a const written in source is, on every engine:
/// every reader of its wire reads the captured value, a write to what it
/// reads changes nothing until `init`, and the capture is listed.
#[test]
fn a_programmatically_marked_const_is_captured_at_init() {
    use polydat::compile::assembly::{PolydatAssembler, WireRef};
    use polydat::library::arithmetic::Mod;
    use polydat::library::hash::Hash;
    for engine in every_engine() {
        let mut asm = PolydatAssembler::new(vec!["cycle".into()]);
        asm.add_input(
            "n",
            Value::U64(7),
            polydat::ast::PortType::U64,
            polydat::kernel::InputKind::ExternalWrite,
        );
        asm.add_node("h", Box::new(Hash::new()), vec![WireRef::input("n")]);
        asm.add_node("m", Box::new(Mod::new(1000)), vec![WireRef::node("h")]);
        asm.add_output("x", WireRef::node("h"));
        asm.add_output("y", WireRef::node("m"));
        asm.mark_const_output("x");
        let mut k = asm
            .compile_with(engine)
            .unwrap_or_else(|e| panic!("{engine}: {e}"));
        assert_eq!(
            k.const_inits()
                .iter()
                .map(|c| c.name.as_str())
                .collect::<Vec<_>>(),
            ["x"],
            "{engine}"
        );
        k.set_inputs(&[0]);
        let captured = k.pull("x").as_u64();
        assert_eq!(k.pull("y").as_u64(), captured % 1000, "{engine}");
        k.set_input("n", Value::U64(8)).unwrap();
        assert_eq!(
            k.pull("x").as_u64(),
            captured,
            "{engine}: a write does not change it"
        );
        assert_eq!(
            k.pull("y").as_u64(),
            captured % 1000,
            "{engine}: a reader of its wire reads the capture"
        );
        k.init().unwrap();
        let recaptured = k.pull("x").as_u64();
        assert_ne!(recaptured, captured, "{engine}: init captures again");
        assert_eq!(k.pull("y").as_u64(), recaptured % 1000, "{engine}");
    }
}

/// `shared x := <expr>` starts the register at the expression's value,
/// taken when the declaring kernel is initialized, on every engine. A
/// write to the register is what readers see, and initializing again
/// does not seed it a second time.
#[test]
fn a_shared_register_takes_a_computed_start_at_init() {
    let src =
        "input cycle: u64\nextern base: u64 = 20\nshared rolling := base * 2\nout := rolling + 1\n";
    for engine in every_engine() {
        let mut k = compile_polydat_with(src, engine).unwrap_or_else(|e| panic!("{engine}: {e}"));
        k.set_inputs(&[0]);
        assert_eq!(k.pull("rolling"), Value::U64(40), "{engine}");
        assert_eq!(k.pull("out"), Value::U64(41), "{engine}");
        k.set_input("rolling", Value::U64(7)).unwrap();
        assert_eq!(k.pull("out"), Value::U64(8), "{engine}");
        k.set_input("base", Value::U64(5)).unwrap();
        k.init().unwrap();
        assert_eq!(
            k.pull("rolling"),
            Value::U64(7),
            "{engine}: a written register is not seeded again"
        );
    }
}

/// A register with a computed start is read like any extern once
/// initialization seeds it. A host clearing it to `None` makes a pull
/// that reads it answer `None`, and pure native, which cannot carry a
/// `None`, refuses that pull naming the register; a pull that does not
/// read it is served on every engine.
#[test]
fn a_seeded_register_cleared_to_none_is_unset_for_its_readers() {
    let src = "input cycle: u64\nextern base: u64 = 20\nshared rolling := base * 2\n\
               out := rolling + 1\nother := base + cycle\n";
    let mut engines = vec![
        Engine::Interpreter(JitMode::Auto),
        Engine::Closures(Provenance::Auto),
    ];
    if cfg!(feature = "jit") {
        engines.push(Engine::Native(Provenance::Auto));
        engines.push(Engine::PureNative(Provenance::PushPull));
        engines.push(Engine::PureNative(Provenance::Raw));
    }
    for engine in engines {
        let mut k = compile_polydat_with(src, engine).unwrap_or_else(|e| panic!("{engine}: {e}"));
        k.set_inputs(&[1]);
        assert_eq!(k.pull("out"), Value::U64(41), "{engine}: seeded at init");
        k.set_input("rolling", Value::None)
            .unwrap_or_else(|e| panic!("{engine}: {e}"));
        assert_eq!(k.pull("other"), Value::U64(21), "{engine}: a non-reader");
        let pulled = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| k.pull("out")));
        if matches!(engine, Engine::PureNative(_)) {
            let payload = pulled.expect_err("pure native refuses a pull reading `rolling`");
            let text = payload
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| payload.downcast_ref::<&str>().map(|s| (*s).to_string()))
                .unwrap_or_default();
            assert!(
                text.contains("extern 'rolling'") && text.contains("pure native"),
                "{engine}: {text}"
            );
        } else {
            assert_eq!(
                pulled.unwrap_or_else(|_| panic!("{engine}: a None, not a panic")),
                Value::None,
                "{engine}"
            );
        }
        k.set_input("rolling", Value::U64(5))
            .unwrap_or_else(|e| panic!("{engine}: {e}"));
        assert_eq!(k.pull("out"), Value::U64(6), "{engine}: set again");
    }
}

/// A child that declares the same `shared` binding and is attached to
/// its parent's register never seeds it: it reads the parent's start,
/// and a write on either side is what both read.
#[test]
fn a_child_attached_to_a_register_does_not_seed_it() {
    let parent_src = "input cycle: u64\nextern base: u64 = 20\nshared rolling := base * 2\n";
    let child_src = "input cycle: u64\nextern base: u64 = 1\nshared rolling := base + 1000\nseen := rolling + 0\n";
    for engine in every_engine() {
        let mut parent =
            compile_polydat_with(parent_src, engine).unwrap_or_else(|e| panic!("{engine}: {e}"));
        let child_program = compile_polydat_with(child_src, engine)
            .unwrap_or_else(|e| panic!("{engine}: {e}"))
            .into_program();
        let mut child = polydat::kernel::bind_under(parent.as_ref(), child_program, &[])
            .unwrap_or_else(|e| panic!("{engine}: {e}"));
        child.set_inputs(&[0]);
        assert_eq!(
            child.pull("seen"),
            Value::U64(40),
            "{engine}: the parent's start, not the child's"
        );
        child.set_input("rolling", Value::U64(3)).unwrap();
        assert_eq!(parent.pull("rolling"), Value::U64(3), "{engine}");
        child.init().unwrap();
        assert_eq!(child.pull("seen"), Value::U64(3), "{engine}");
    }
}

/// `shared const` is refused: a value fixed for the kernel's life cannot
/// also be a register other scopes write.
#[test]
fn shared_const_is_refused() {
    for src in [
        "shared const x := n + 1\nextern n: u64 = 1\n",
        "const shared x := 1\n",
    ] {
        let err = compile_polydat_with(src, Engine::default())
            .err()
            .unwrap_or_else(|| panic!("`{src}` compiles"));
        let text = err.to_string();
        assert!(
            text.contains("const") && text.contains("shared") && text.contains("register"),
            "{text}"
        );
    }
}

/// Each `for` activation initializes its body, so a const over a tuple
/// element holds that element's value.
#[test]
fn each_activation_initializes_its_consts() {
    let src = "input cycle: u64\nfor i in 1..4 {\n    const doubled := i * 2\n    out := doubled + 0\n}\n";
    for engine in every_engine() {
        let mut k = compile_polydat_with(src, engine).unwrap_or_else(|e| panic!("{engine}: {e}"));
        k.set_inputs(&[0]);
        let stream = k.traverse(0).unwrap();
        for index in 0..stream.len() {
            let mut act = stream.activation(index).unwrap();
            let i = act.coord("i").expect("the tuple's element").as_u64();
            let k = act.cycle(0);
            assert_eq!(
                k.pull("out").as_u64(),
                i * 2,
                "{engine}: activation {index}"
            );
        }
    }
}

/// A fork copies an initialized state, capture included; a kernel
/// created from the program is initialized afresh.
#[test]
fn a_fork_keeps_the_capture_and_a_created_kernel_captures_its_own() {
    let src = "const first := counter()\n";
    for engine in every_engine() {
        let mut k = compile_polydat_with(src, engine).unwrap_or_else(|e| panic!("{engine}: {e}"));
        let captured = k.pull("first").as_u64();
        let mut forked = k.fork();
        assert_eq!(
            forked.pull("first").as_u64(),
            captured,
            "{engine}: the fork keeps it"
        );
        let program = k.into_program();
        let mut created = program.clone().create_kernel();
        let _ = created.pull("first");
        let mut created_again = program.create_kernel();
        assert_ne!(
            created.pull("first").as_u64(),
            created_again.pull("first").as_u64(),
            "{engine}: each created kernel is initialized, and captures its own"
        );
    }
}

/// A const over a nondeterministic read is the acknowledgment of it, as
/// `volatile` is: it compiles under strict on every engine, and without
/// strict it compiles with no warning.
#[test]
fn a_const_capture_acknowledges_its_nondeterministic_read() {
    let src = "input cycle: u64\nconst session_start := current_epoch_millis()\n";
    for engine in every_engine() {
        for strict in [true, false] {
            let options = polydat::dsl::compile::CompileOptions {
                strict,
                engine,
                ..Default::default()
            };
            let mut log = polydat::dsl::events::CompileEventLog::new();
            polydat::dsl::compile::compile_polydat_kernel_with_options(
                src,
                &options,
                Some(&mut log),
            )
            .unwrap_or_else(|e| panic!("{engine}, strict {strict}: {e}"));
            assert!(
                log.warnings().is_empty(),
                "{engine}, strict {strict}: {:?}",
                log.warnings()
            );
        }
    }
}

/// Only initialization writes a const's slot.
#[test]
fn a_consts_slot_refuses_a_write() {
    let src = "extern n: u64 = 3\nconst c := n + 1\n";
    for engine in every_engine() {
        let mut k = compile_polydat_with(src, engine).unwrap_or_else(|e| panic!("{engine}: {e}"));
        let err = k
            .set_input("__const_c", Value::U64(9))
            .expect_err("a const's slot is written only by init");
        assert!(
            matches!(err, polydat::kernel::WriteError::ConstSlot { .. }),
            "{engine}: {err}"
        );
        assert_eq!(k.pull("c").as_u64(), 4, "{engine}");
    }
}

/// Consts that read each other cannot be ordered, and a const that
/// fails makes initialization fail, naming it.
#[test]
fn a_const_cycle_and_a_failing_const_are_errors() {
    let cyclic = "extern n: u64 = 1\nconst a := b + n\nconst b := a + n\n";
    for engine in every_engine() {
        let err = compile_polydat_with(cyclic, engine)
            .err()
            .unwrap_or_else(|| panic!("{engine}: a const cycle compiles"));
        assert!(err.to_string().contains("cycle"), "{engine}: {err}");
    }
    let failing = "extern text: str = \"x\"\nconst n := vec_len_i32(str_to_vec_i32(text))\n";
    for engine in every_engine() {
        let err = compile_polydat_with(failing, engine)
            .err()
            .unwrap_or_else(|| panic!("{engine}: a failing const builds"));
        assert!(
            err.to_string().contains("const 'n'"),
            "{engine}: the error names the const: {err}"
        );
    }
}

/// A scope's lookup answers for the values fixed for the kernel's life
/// (a folded value and a const) and not for a computed output, before a
/// pull or after one, on every engine (native_scope_trees.md §6).
#[test]
fn lookup_answers_for_fixed_values_and_not_for_computed_outputs() {
    use polydat::kernel::interp::{KernelLookup, Lookup};
    let src = "input cycle: u64\nextern n: u64 = 5\nfolded := 42\nconst c := n + 1\ncyc_dep := hash(cycle)\n";
    for engine in every_engine() {
        let mut k = compile_polydat_with(src, engine).unwrap_or_else(|e| panic!("{engine}: {e}"));
        k.set_inputs(&[7]);
        for pulled in [false, true] {
            if pulled {
                let _ = k.pull("cyc_dep");
            }
            let lookup = KernelLookup::new(k.as_ref());
            assert_eq!(
                lookup.lookup("folded"),
                Some(Value::U64(42)),
                "{engine}, pulled {pulled}"
            );
            assert_eq!(
                lookup.lookup("c"),
                Some(Value::U64(6)),
                "{engine}, pulled {pulled}"
            );
            assert_eq!(lookup.lookup("cyc_dep"), None, "{engine}, pulled {pulled}");
        }
    }
}
