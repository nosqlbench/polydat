// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Illustration: polydat is a function library AND a function loader.
//!
//! The 230 built-in nodes are one library. Workload-author functions
//! written in `.polydat` files are another — the compiler loads them
//! from the library paths in `CompileOptions` and they're callable from
//! your DSL by name as if they were built in.
//!
//! This example loads the shipped `stdlib/identity.polydat` module and
//! calls its `hashed_id` function from the workload-author DSL above.

use polydat::dsl::{CompileOptions, compile_polydat_kernel_with_options};

fn main() {
    // Path to the stdlib module shipped with the crate.
    let stdlib = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("stdlib")
        .join("identity.polydat");

    let mut kernel = compile_polydat_kernel_with_options(
        r#"
            input cycle: u64
            // Calls `hashed_id` from identity.polydat — not a built-in,
            // loaded from disk just above.
            uid := hashed_id(cycle, 1000000)
        "#,
        &CompileOptions {
            lib_paths: vec![stdlib.clone()], // explicit library paths
            context: "library_and_loader example".into(),
            ..CompileOptions::default() // no source directory, every output, not strict
        },
        None,
    )
    .expect("compile failed");

    println!("library loaded from: {}", stdlib.display());
    for c in 0..5u64 {
        kernel.set_inputs(&[c]);
        let uid = kernel.pull("uid").as_u64();
        println!("  cycle={c} → uid={uid}");
        assert!(uid < 1_000_000);
    }
}
