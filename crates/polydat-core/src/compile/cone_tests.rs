// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Cone extraction correctness.
//!
//! The load-bearing invariant is differential: for any program, a
//! `Force`-compiled kernel produces bit-identical outputs to an
//! `Off`-compiled one. These tests pin that invariant on the boundary
//! types the cones marshal (U64/F64), the mixed-graph case (fallback
//! node kept on the interpreter), and the panic attribution contract
//! for predicate violations inside native code. The cone mode is a
//! property of each compile, so the tests need no serialization.

/// Reduced from the workspace fuzz (`fuzz_type_adapters`, seed
/// 0xDEADBEEF, iteration 261): connected-component cone selection
/// grouped eligible nodes whose only path between them ran
/// through INELIGIBLE nodes, so the kept middle became both a
/// consumer of the cone and one of its producers — a cycle in
/// the spliced graph that tripped the rebuild topo-sort assert.
/// A non-convex component must take the SRD-105 fallback (stay
/// on the interpreter), never panic the compile.
#[test]
fn non_convex_components_stay_on_the_interpreter() {
    let src = "input cycle: u64\n\
               b0 := u64_not(cycle)\n\
               b1 := str_eq(b0, b0)\n\
               b2 := u64_xor(b1, b0)\n\
               b3 := ln(b2)\n\
               b4 := tan(b3)\n\
               b5 := str_ne(b0, b2)\n\
               b6 := const_f64()\n\
               b7 := regex_replace(b6, \"s62\", \"s17\")\n\
               b8 := closest_decade(cycle)\n\
               b9 := dist_normal(cycle, 23.70, 55.80)\n";
    // The result may be Ok or a clean Err (the fuzz feeds garbage
    // types on purpose); the invariant under test is NO PANIC in
    // cone extraction/splicing under the default (auto) mode.
    let _ = crate::dsl::compile::compile_polydat_interpreter(src);
}
