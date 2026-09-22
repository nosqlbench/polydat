// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Engine parity, step 6 (A7): a node that fails at evaluation fails
//! with the same attributed message on every engine. The interpreter
//! enriches a node's panic with the node's name, the outputs it
//! feeds, the program's context, and the input values; a compiled
//! kernel catches the step's panic at the step boundary and enriches
//! it the same way. Only the `panicked at` line differs, since it
//! names the code that raised the panic, which is the engine's own.

use polydat::dsl::compile::compile_polydat_to_assembler;
use polydat::{Engine, JitMode, Kernel, KernelError, Provenance};

fn engines() -> Vec<Engine> {
    let mut all = vec![Engine::Interpreter(JitMode::Off)];
    for m in [
        Provenance::Raw,
        Provenance::Push,
        Provenance::Pull,
        Provenance::PushPull,
    ] {
        all.push(Engine::Closures(m));
        // Native has no push-only kernel, so the factory refuses that
        // pair rather than building push-pull under its name.
        #[cfg(feature = "jit")]
        if m != Provenance::Push {
            all.push(Engine::Native(m));
        }
    }
    all
}

fn payload_text(p: Box<dyn std::any::Any + Send>) -> String {
    p.downcast_ref::<String>()
        .cloned()
        .or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_else(|| "<non-string payload>".into())
}

/// The message without its `panicked at` line.
fn without_location(msg: &str) -> String {
    msg.lines()
        .filter(|l| !l.trim_start().starts_with("↳ panicked at"))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Compile on `engine`, drive with `drive`, pull every output; the
/// panic's message, or `None` where the engine refused the program.
/// Cones are off, so the failing node is the program's own on every
/// engine; `a_cone_names_the_member_that_failed` covers them.
fn failure(src: &str, engine: Engine, drive: &dyn Fn(&mut dyn Kernel)) -> Option<String> {
    failure_with(src, engine, JitMode::Off, drive)
}

fn failure_with(
    src: &str,
    engine: Engine,
    mode: JitMode,
    drive: &dyn Fn(&mut dyn Kernel),
) -> Option<String> {
    let mut asm = compile_polydat_to_assembler(src).expect("the program compiles");
    asm.set_jit_mode(mode);
    let mut k = match asm.compile_with(engine) {
        Ok(k) => k,
        Err(KernelError::Refused { .. }) => return None,
        Err(e) => panic!("{engine}: {e}"),
    };
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        drive(&mut *k);
        let outs: Vec<String> = k.output_names().iter().map(|s| s.to_string()).collect();
        for o in &outs {
            let _ = k.pull(o);
        }
    }));
    match r {
        Ok(()) => panic!("{engine}: the program did not fail"),
        Err(p) => Some(payload_text(p)),
    }
}

/// Every engine that accepts the program fails with the interpreter's
/// message, location aside, and the message carries the attribution.
fn assert_same_failure(src: &str, drive: &dyn Fn(&mut dyn Kernel), expect: &[&str]) {
    let want = failure(src, Engine::Interpreter(JitMode::Off), drive)
        .expect("the interpreter runs everything");
    for e in expect {
        assert!(want.contains(e), "interpreter message lacks {e:?}:\n{want}");
    }
    assert!(
        want.contains("↳ panicked at"),
        "interpreter message lacks the location:\n{want}"
    );
    for engine in engines() {
        let Some(got) = failure(src, engine, drive) else {
            continue;
        };
        assert_eq!(
            without_location(&got),
            without_location(&want),
            "{engine} reports the failure differently from the interpreter"
        );
    }
}

#[test]
fn a_predicate_violation_reads_the_same_on_every_engine() {
    let src = "input cycle: u64\ndoubled := mul(cycle, 2)\nchecked := in_range(doubled, 0, 5)\n";
    assert_same_failure(
        src,
        &|k| k.set_inputs(&[10]),
        &[
            "in_range",
            "↳ in node `in_range` (output checked)",
            "↳ inputs: [[0]=U64(20)]",
        ],
    );
}

#[test]
fn a_string_coercion_failure_reads_the_same_on_every_engine() {
    let src = "input cycle: u64\nextern s: str = \"12\"\nn := __str_to_u64(s)\n";
    assert_same_failure(
        src,
        &|k| {
            k.set_inputs(&[3]);
            k.set_input("s", polydat::ast::Value::Str("twelve".into()))
                .expect("s is an extern");
        },
        &[
            "value \"twelve\" is not a whole number.",
            "↳ in node `__str_to_u64` (output n)",
            "↳ inputs: [[0]=Str(\"twelve\")]",
        ],
    );
}

#[test]
fn the_attribution_names_the_context_and_the_original_message() {
    let src = "input cycle: u64\nchecked := in_range(cycle, 0, 5)\n";
    for engine in engines() {
        let Some(got) = failure(src, engine, &|k| k.set_inputs(&[7])) else {
            continue;
        };
        let first = got.lines().next().unwrap_or("");
        assert!(
            first.starts_with("in_range: value 7 outside [0, 5]"),
            "{engine}: the original message is not first:\n{got}"
        );
        assert!(
            got.contains("while evaluating"),
            "{engine}: no context line:\n{got}"
        );
    }
}

/// With cones on, the interpreter's program holds the fused node; its
/// failure names the member that failed, that member's outputs as the
/// program names them, and the program, exactly as the same failure
/// reads on the native engine (A7). The cone is not a frame of its own.
#[cfg(feature = "jit")]
#[test]
fn a_cone_names_the_member_that_failed() {
    let src = "input cycle: u64\ndoubled := mul(cycle, 2)\nchecked := in_range(doubled, 0, 5)\n";
    let got = failure_with(
        src,
        Engine::Interpreter(JitMode::Force),
        JitMode::Force,
        &|k| k.set_inputs(&[10]),
    )
    .expect("the interpreter runs everything");
    for e in [
        "in_range: value 20 outside [0, 5]",
        "↳ in node `in_range` (output checked) while evaluating (polydat)",
        "↳ inputs: [[0]=U64(20)]",
    ] {
        assert!(got.contains(e), "cone failure lacks {e:?}:\n{got}");
    }
    assert!(!got.contains("jit_cone"), "the cone is not a frame:\n{got}");
    let native = failure_with(
        src,
        Engine::Native(Provenance::Auto),
        JitMode::Force,
        &|k| k.set_inputs(&[10]),
    )
    .expect("native code runs everything");
    assert_eq!(
        got, native,
        "the fused interpreter and the native engine read alike"
    );
}

/// A compile-constant step that cannot be computed fails at *build*
/// with the same sentence on every engine — and that sentence is a
/// diagnosis, not a crash report.
///
/// Evaluation failures keep the `panicked at` line, and the module doc
/// above says why: it names the code that raised, which is the engine's
/// own. A build failure is a different thing. The reason is a property
/// of the program (`bad spec 's97'`), the user has none of polydat's
/// source, and a file and line there reads as an internal defect. The
/// three compiled engines never carried one; the interpreter did, which
/// is how the same program failed two ways depending on the engine, and
/// how a message the fuzzer classifies as cryptic reached a user
/// (F-E16: `ConstantFold` on every engine, an error and never a panic).
#[test]
fn a_fold_failure_reads_the_same_on_every_engine() {
    let src = "p := partitions(\"s97\")\n";
    let mut seen: Vec<(String, String)> = Vec::new();

    for engine in engines() {
        let outcome = compile_polydat_to_assembler(src)
            .expect("it assembles")
            .compile_with(engine);
        let msg = match outcome {
            Err(KernelError::Refused { .. }) => continue,
            Err(e) => e.to_string(),
            Ok(_) => panic!("{engine}: an unparseable partition spec must fail at build"),
        };
        assert!(
            msg.contains("bad spec"),
            "{engine}: the node's own reason must survive: {msg}"
        );
        assert!(
            !msg.to_lowercase().contains("panic"),
            "{engine}: a build error must not read as a crash: {msg}"
        );
        seen.push((format!("{engine}"), msg));
    }

    assert!(seen.len() > 1, "expected more than one engine to build");
    let (first_name, first) = &seen[0];
    for (name, msg) in &seen[1..] {
        assert_eq!(msg, first, "{name} differs from {first_name}");
    }
}
