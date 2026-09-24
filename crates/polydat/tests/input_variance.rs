// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Input variance on every engine (input_variance.md §10).
//!
//! A write into a kernel is never converted. A declared input refuses a
//! value of another type under every setting; an input whose type the
//! compiler inferred is open, and `CompileOptions::input_variance`
//! decides whether it keeps its inferred type, stops the build, or takes
//! any value through a converter node reported in the compile log.

use polydat::ast::{PortType, Value};
use polydat::dsl::compile::{CompileOptions, InputVariance, compile_polydat_kernel_with_options};
use polydat::dsl::events::{CompileEvent, CompileEventLog, EventLevel};
use polydat::kernel::TypeOrigin;
use polydat::kernel::subcontext::{
    BodyFragment, CompileOptions as ScopeOptions, SourceContext, SubcontextBuilder,
};
use polydat::{Engine, JitMode, Kernel, KernelError, Provenance};

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

/// `n` is an extern the caller says it typed by inference, the way a
/// program synthesizer's externs are; `out` reads it as a `u64`.
const OPEN: &str = "extern n: u64 = 1\nout := u64_add(n, 1)\n";

fn compile(
    src: &str,
    engine: Engine,
    variance: InputVariance,
    log: Option<&mut CompileEventLog>,
) -> Result<Box<dyn Kernel>, KernelError> {
    let options = CompileOptions {
        engine,
        input_variance: variance,
        inferred_externs: vec!["n".to_string()],
        ..CompileOptions::default()
    };
    compile_polydat_kernel_with_options(src, &options, log)
}

/// A declared input refuses a value of another type under every
/// setting, on every engine.
#[test]
fn a_declared_input_refuses_a_mismatch_under_every_setting() {
    let declared = "extern n: u64 = 1\nout := u64_add(n, 1)\n";
    for engine in engines() {
        for variance in [
            InputVariance::Fixed,
            InputVariance::Error,
            InputVariance::Warn,
            InputVariance::Info,
        ] {
            let options = CompileOptions {
                engine,
                input_variance: variance,
                ..CompileOptions::default()
            };
            let mut k = compile_polydat_kernel_with_options(declared, &options, None)
                .unwrap_or_else(|e| panic!("{engine} {variance:?}: {e}"));
            assert!(
                k.set_input("n", Value::Str("5".into())).is_err(),
                "{engine} {variance:?}: a declared u64 took a string"
            );
            assert_eq!(k.input_type_origin("n"), Some(TypeOrigin::Declared));
        }
    }
}

/// `Fixed`, the default, keeps an open input at its inferred type and
/// refuses a mismatch, as before this setting existed.
#[test]
fn fixed_keeps_an_open_input_at_its_inferred_type() {
    for engine in engines() {
        let mut k = compile(OPEN, engine, InputVariance::Fixed, None).expect("builds");
        assert_eq!(
            k.input_type_origin("n"),
            Some(TypeOrigin::Inferred),
            "{engine}"
        );
        assert!(
            k.set_input("n", Value::Str("5".into())).is_err(),
            "{engine}"
        );
    }
}

/// `Error` stops construction and names every open input.
#[test]
fn error_names_every_open_input() {
    for engine in engines() {
        let err = compile(OPEN, engine, InputVariance::Error, None)
            .err()
            .unwrap_or_else(|| panic!("{engine}: an open input built under Error"));
        let text = err.to_string();
        assert!(text.contains("n (inferred u64)"), "{engine}: {text}");
    }
}

/// `Warn` and `Info` open the input: any value is written as it is, a
/// converter turns it into what the reader reads, the input still
/// reports the reader's type, and the log reports the converter at the
/// configured level.
#[test]
fn warn_and_info_convert_an_open_input_and_report_it() {
    for (variance, level) in [
        (InputVariance::Warn, EventLevel::Warning),
        (InputVariance::Info, EventLevel::Info),
    ] {
        for engine in engines() {
            let mut log = CompileEventLog::new();
            let mut k = compile(OPEN, engine, variance, Some(&mut log))
                .unwrap_or_else(|e| panic!("{engine} {variance:?}: {e}"));
            assert_eq!(k.input_port_type("n"), Some(PortType::U64), "{engine}");
            assert!(
                log.events().iter().any(|e| matches!(
                    e,
                    CompileEvent::InputConverterInserted { input, to, level: l, .. }
                        if input == "n" && to == "u64" && *l == level
                )),
                "{engine} {variance:?}: no converter event at {level:?}:\n{}",
                log.format()
            );

            k.set_inputs(&[0]);
            k.set_input("n", Value::Str("41".into()))
                .unwrap_or_else(|e| panic!("{engine}: {e}"));
            assert_eq!(k.pull("out"), Value::U64(42), "{engine}: text converted");
            k.set_input("n", Value::U64(9)).expect("a u64 as is");
            assert_eq!(k.pull("out"), Value::U64(10), "{engine}: a u64 as is");
        }
    }
}

/// A value the converter cannot convert fails the converter, naming the
/// input, on every engine.
#[test]
fn an_unconvertible_value_fails_the_converter_naming_the_input() {
    for engine in engines() {
        let mut k = compile(OPEN, engine, InputVariance::Warn, None).expect("builds");
        k.set_inputs(&[0]);
        k.set_input("n", Value::Str("not a number".into()))
            .expect("an open input takes any value");
        let failure = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| k.pull("out")))
            .expect_err("the converter fails");
        let text = failure
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| failure.downcast_ref::<&str>().map(|s| s.to_string()))
            .unwrap_or_default();
        assert!(text.contains("'n'"), "{engine}: {text}");
    }
}

/// The host-side conversion is the converter's: what one does, the other
/// does.
#[test]
fn convert_to_port_agrees_with_the_converter() {
    use polydat::convert::to_port;
    assert_eq!(
        to_port(Value::Str("41".into()), PortType::U64),
        Ok(Value::U64(41))
    );
    assert_eq!(to_port(Value::U64(7), PortType::U64), Ok(Value::U64(7)));
    assert_eq!(to_port(Value::U64(7), PortType::F64), Ok(Value::F64(7.0)));
    assert_eq!(to_port(Value::None, PortType::U64), Ok(Value::None));
    let err = to_port(Value::Str("not a number".into()), PortType::U64).expect_err("unparseable");
    assert_eq!((err.from, err.to), (PortType::Str, PortType::U64));

    for engine in engines() {
        let mut k = compile(OPEN, engine, InputVariance::Info, None).expect("builds");
        k.set_inputs(&[0]);
        for text in ["0", "41", "18446744073709551614"] {
            k.set_input("n", Value::Str(text.into())).expect("takes it");
            let want = to_port(Value::Str(text.into()), PortType::U64).expect("converts");
            let Value::U64(n) = want else { unreachable!() };
            assert_eq!(
                k.pull("out"),
                Value::U64(n.wrapping_add(1)),
                "{engine}: {text}"
            );
        }
    }
}

/// `transform::convert_input` opens one declared extern, whatever the
/// setting, and the log reports its converter as info.
#[test]
fn convert_input_opens_one_extern() {
    let mut file = polydat::dsl::compile::parse_polydat(
        "extern n: u64 = 1\nextern m: u64 = 2\nout := u64_add(n, m)\n",
    )
    .expect("parses");
    polydat::dsl::transform::convert_input(&mut file, "n").expect("n is an extern");
    assert!(polydat::dsl::transform::convert_input(&mut file, "nope").is_err());
    for engine in engines() {
        let options = CompileOptions {
            engine,
            ..CompileOptions::default()
        };
        let mut log = CompileEventLog::new();
        let mut k = polydat::dsl::compile::compile_ast_with_engine(
            &file,
            "",
            &options,
            Some(&mut log),
            engine,
        )
        .unwrap_or_else(|e| panic!("{engine}: {e}"));
        assert!(
            log.events().iter().any(|e| matches!(
                e,
                CompileEvent::InputConverterInserted { input, level: EventLevel::Info, .. }
                    if input == "n"
            )),
            "{engine}:\n{}",
            log.format()
        );
        k.set_inputs(&[0]);
        k.set_input("n", Value::Str("40".into()))
            .expect("n takes text");
        assert_eq!(k.pull("out"), Value::U64(42), "{engine}");
        assert!(
            k.set_input("m", Value::Str("3".into())).is_err(),
            "{engine}: m stays declared"
        );
    }
}

/// nmbrs's case: a scope's result binding reads the builder's `count`
/// extern, and the host writes the count as text. Compiled at `Warn`,
/// the scope converts it on every engine.
#[test]
fn a_scope_s_synthesized_result_extern_converts_under_warn() {
    for engine in engines() {
        let root = polydat::dsl::compile::compile_polydat_with("input cycle: u64\n", engine)
            .unwrap_or_else(|e| panic!("{engine}: {e}"));
        let mut b = SubcontextBuilder::under(root.as_ref());
        b.context(SourceContext::new("op"));
        b.with_compile_options(ScopeOptions {
            input_variance: InputVariance::Warn,
            ..ScopeOptions::default()
        });
        b.body(BodyFragment::PolydatSource(
            "input cycle: u64\n".to_string(),
        ));
        b.add_result_bindings("total := u64_add(count, 1)\n")
            .unwrap_or_else(|e| panic!("{engine}: {e:?}"));
        let op = b.finalize().unwrap_or_else(|e| panic!("{engine}: {e:?}"));
        let mut k = op
            .instantiate_under(root.as_ref(), engine, &[])
            .unwrap_or_else(|e| panic!("{engine}: {e}"));
        assert_eq!(
            k.input_type_origin("count"),
            Some(TypeOrigin::Inferred),
            "{engine}"
        );
        k.set_inputs(&[0]);
        k.set_input("count", Value::Str("41".into()))
            .unwrap_or_else(|e| panic!("{engine}: {e}"));
        assert_eq!(k.pull("total"), Value::U64(42), "{engine}");
    }
}
