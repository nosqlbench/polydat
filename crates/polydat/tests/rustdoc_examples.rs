// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! The examples in the crate's rustdoc, as tests. The crate does not
//! run doctests (`doctest = false`: the suite runs under nextest,
//! which has none), so each example that rustdoc shows is mirrored
//! here, in the same words, and a change to one is a change to both.

/// `src/lib.rs`, "From DSL source".
#[test]
fn from_dsl_source() {
    use polydat::dsl::compile_polydat_kernel;

    let mut kernel = compile_polydat_kernel(
        r#"
        input cycle: u64
        hashed := hash(cycle)
        user_id := mod(hashed, 1000000)
    "#,
    )
    .unwrap();

    kernel.set_inputs(&[42]);
    let user_id = kernel.pull("user_id").as_u64();
    assert!(user_id < 1_000_000);
}

/// `src/lib.rs`, "From the assembler API".
#[test]
fn from_the_assembler_api() {
    use polydat::compile::assembly::{PolydatAssembler, WireRef};
    use polydat::library::arithmetic::Mod;
    use polydat::library::hash::Hash;

    let mut asm = PolydatAssembler::new(vec!["cycle".into()]);
    asm.add_node(
        "hashed",
        Box::new(Hash::new()),
        vec![WireRef::input("cycle")],
    );
    asm.add_node(
        "user_id",
        Box::new(Mod::new(1_000_000)),
        vec![WireRef::node("hashed")],
    );
    asm.add_output("user_id", WireRef::node("user_id"));

    let mut kernel = asm.compile().unwrap();
    kernel.set_inputs(&[42]);
    assert!(kernel.pull_ref("user_id").as_u64() < 1_000_000);
}

/// `dsl::compile::eval_const_expr`, "Examples".
#[test]
fn eval_const_expr_examples() {
    use polydat::dsl::compile::eval_const_expr;
    let v = eval_const_expr("4 * 4").unwrap();
    assert_eq!(v.as_u64(), 16); // both int literals → u64_mul
    let v = eval_const_expr("4.0 * 4.0").unwrap();
    assert_eq!(v.as_f64(), 16.0); // both float literals → f64_mul
}
