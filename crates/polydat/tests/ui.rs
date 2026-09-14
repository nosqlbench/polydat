// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Compile-fail tests for `#[polydat_node]`. Every diagnostic the
//! attribute can emit has a case under `tests/ui/fail/`, and the
//! expected compiler output lives beside it in a `.stderr` file. A
//! change to a message shows up as a diff here; regenerate the
//! snapshots with `TRYBUILD=overwrite cargo test --test ui` and review
//! the diff. `tests/ui/pass/` holds the shapes that must keep
//! compiling.
//!
//! Each case is a separate compile, so the suite takes most of a
//! minute. It is ignored by default and run explicitly:
//!
//! ```sh
//! cargo test --test ui -- --ignored
//! ```
//!
//! CI runs it on every push.

#[test]
#[ignore = "one compile per case; run with --ignored, as CI does"]
fn node_attribute_diagnostics() {
    let t = trybuild::TestCases::new();
    t.pass("tests/ui/pass/*.rs");
    t.compile_fail("tests/ui/fail/*.rs");
}

/// The subcontext seal (`kernel/subcontext`): the two crate-private
/// entry points a caller could use to bypass the typed child
/// construction must not compile from outside the crate. Two compiles,
/// so this one runs by default.
#[test]
fn kernel_seal() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/ui/seal/*.rs");
}
