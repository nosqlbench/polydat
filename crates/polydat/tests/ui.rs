// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Compile-fail tests for `#[polydat_node]`. Every diagnostic the
//! attribute can emit has a case under `tests/ui/fail/`, and the
//! expected compiler output lives beside it in a `.stderr` file. A
//! change to a message shows up as a diff here; regenerate the
//! snapshots with `TRYBUILD=overwrite cargo test --test ui` and review
//! the diff. `tests/ui/pass/` holds the shapes that must keep
//! compiling.

#[test]
fn node_attribute_diagnostics() {
    let t = trybuild::TestCases::new();
    t.pass("tests/ui/pass/*.rs");
    t.compile_fail("tests/ui/fail/*.rs");
}
