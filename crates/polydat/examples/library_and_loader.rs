// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Illustration: polydat is a function library AND a function loader.
//!
//! The 230 built-in nodes are one library. Workload-author functions
//! written in `.polydat` files are another — the compiler loads them
//! from the library paths in `CompileOptions` and they're callable from
//! your DSL by name as if they were built in.
//!
//! This example names the shipped `stdlib/` directory as a library
//! path and calls its `hashed_id` function from the workload-author
//! DSL above. The same modules are embedded in the compiler, so the
//! path is how a host adds its own modules ahead of them.

use polydat::dsl::{CompileOptions, compile_polydat_kernel_with_options};

fn main() {
    // The stdlib directory shipped with the compiler crate.
    let stdlib = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("polydat-core")
        .join("stdlib");

    let mut kernel = compile_polydat_kernel_with_options(
        r#"
            input cycle: u64
            // Calls `hashed_id` from identity.polydat, a library
            // module rather than a built-in node.
            uid := hashed_id(cycle, 1000000)
        "#,
        &CompileOptions {
            lib_paths: vec![stdlib.clone()], // a directory of modules, searched first
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
