// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! `name=value` assignment as a program transform: text fuses to the
//! declared type inside the program, inputs become fixed externs, and
//! bad values fail at compile time with the program's diagnostic.

use polydat::dsl::transform::{assign_values, parse_assignment};
use polydat::dsl::{CompileOptions, compile_ast_interpreter_with_options};

fn compile(
    src: &str,
    assigns: &[(&str, &str)],
) -> Result<polydat::kernel::PolydatKernel, polydat::KernelError> {
    let tokens = polydat::dsl::lexer::lex(src).map_err(polydat::KernelError::Source)?;
    let mut ast = polydat::dsl::parser::parse(tokens).map_err(polydat::KernelError::Source)?;
    let pairs: Vec<(String, String)> = assigns
        .iter()
        .map(|(n, v)| (n.to_string(), v.to_string()))
        .collect();
    assign_values(&mut ast, &pairs).map_err(polydat::KernelError::Source)?;
    compile_ast_interpreter_with_options(&ast, src, &CompileOptions::default(), None)
}

const SRC: &str = "input cycle: u64\ninput seed: u64\nextern interval_ms: u64 = 1000\nextern label: String = \"x\"\nextern scale: f64 = 1.0\nextern on: bool = false\nts := cycle * interval_ms + seed\nout := \"{label}:{scale}:{on}\"";

#[test]
fn string_text_fuses_to_declared_extern_types() {
    let mut k = compile(
        SRC,
        &[
            ("interval_ms", "60000"),
            ("label", "run-1"),
            ("scale", "2.5"),
            ("on", "true"),
        ],
    )
    .unwrap();
    k.set_inputs(&[3, 4]);
    assert_eq!(k.pull("ts").as_u64(), 3 * 60000 + 4);
    assert_eq!(k.pull("out").as_str(), "run-1:2.5:true");
}

#[test]
fn assigned_input_becomes_a_fixed_extern() {
    let mut k = compile(SRC, &[("seed", "7")]).unwrap();
    // `seed` is no longer a coordinate: only `cycle` is.
    assert_eq!(k.program().coord_count(), 1);
    k.set_inputs(&[2]);
    assert_eq!(k.pull("ts").as_u64(), 2 * 1000 + 7);
}

#[test]
fn non_numeric_text_for_a_numeric_extern_is_a_compile_error() {
    let err = compile(SRC, &[("interval_ms", "soon")]).unwrap_err();
    assert!(err.to_string().contains("interval_ms"), "{err}");
}

#[test]
fn unknown_name_is_rejected_with_declared_names() {
    let err = compile(SRC, &[("nope", "1")]).unwrap_err();
    assert!(
        err.to_string().contains("nope") && err.to_string().contains("interval_ms"),
        "{err}"
    );
}

#[test]
fn assignment_text_parses_and_validates_names() {
    assert_eq!(
        parse_assignment("a_b=1 2").unwrap(),
        ("a_b".to_string(), "1 2".to_string())
    );
    assert!(parse_assignment("noequals").is_err());
    assert!(parse_assignment("bad-name=1").is_err());
}
