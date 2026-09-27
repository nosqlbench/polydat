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
/// 0xDEADBEEF, iteration 261): eligible nodes whose only path
/// between them runs through ineligible nodes form a non-convex
/// component. Spliced as one cone, the kept middle would be both a
/// consumer of the cone and one of its producers, a cycle in the
/// spliced graph that trips the rebuild topo-sort assert. A
/// non-convex component is split into convex pieces and never panics
/// the compile.
#[test]
fn non_convex_components_split_into_convex_pieces() {
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

/// A node whose native form is withdrawn after the planner admits it:
/// it offers a compiled closure to its first `offers` classifications
/// and none after, so cone eligibility admits it and the cone's layout,
/// which classifies it again, rejects it late. No library node's native
/// form differs between those two classifications, so this node is the
/// program property that makes a planned cone's native build fail.
struct LateRejected {
    meta: crate::ast::NodeMeta,
    offers: std::sync::atomic::AtomicUsize,
}

impl LateRejected {
    fn new(offers: usize) -> Self {
        use crate::ast::{Port, Slot};
        Self {
            meta: crate::ast::NodeMeta {
                name: "late_rejected".into(),
                outs: vec![Port::u64("output")],
                ins: vec![Slot::Wire(Port::u64("input"))],
            },
            offers: std::sync::atomic::AtomicUsize::new(offers),
        }
    }
}

impl crate::ast::PolydatNode for LateRejected {
    fn meta(&self) -> &crate::ast::NodeMeta {
        &self.meta
    }

    fn eval(&self, inputs: &[crate::ast::Value], outputs: &mut [crate::ast::Value]) {
        outputs[0] = crate::ast::Value::U64(inputs[0].as_u64().wrapping_mul(3));
    }

    fn compiled_u64(&self) -> Option<crate::ast::CompiledU64Op> {
        use std::sync::atomic::Ordering;
        self.offers
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_sub(1))
            .ok()?;
        Some(Box::new(|i: &[u64], o: &mut [u64]| {
            o[0] = i[0].wrapping_mul(3)
        }))
    }
}

/// `late_rejected` feeding a copy: one cone under `Auto` or `Force`.
fn late_rejected_program(mode: super::cone::JitMode) -> super::assembly::PolydatAssembler {
    use super::assembly::{PolydatAssembler, WireRef};
    let mut asm = PolydatAssembler::new(vec!["x".into()]);
    asm.add_node(
        "tripled",
        Box::new(LateRejected::new(1)),
        vec![WireRef::input("x")],
    );
    asm.add_node(
        "copied",
        Box::new(crate::library::identity::Identity::new(
            crate::ast::PortType::U64,
        )),
        vec![WireRef::node("tripled")],
    );
    asm.add_output("tripled", WireRef::node("tripled"));
    asm.add_output("copied", WireRef::node("copied"));
    asm.set_jit_mode(mode);
    asm
}

/// Under `Auto` a cone whose native build fails stays on the
/// interpreter, the compile succeeds with the interpreter's results, and
/// the tree's ledger records the cone's members, outputs, boundary, and
/// the error text (engines.md §2.1).
#[test]
fn a_failed_cone_build_under_auto_is_recorded_and_interpreted() {
    use super::cone::JitMode;
    let mut auto = late_rejected_program(JitMode::Auto)
        .compile()
        .expect("Auto compiles whatever the cone's build does");
    let fallbacks = auto.program().ledger().cone_fallbacks();
    assert_eq!(fallbacks.len(), 1, "{fallbacks:?}");
    let f = &fallbacks[0];
    assert_eq!(f.kind, crate::kernel::ConeFallbackKind::Codegen);
    assert_eq!(f.members, ["late_rejected", "identity"]);
    assert_eq!(f.outputs, ["copied", "tripled"]);
    assert_eq!(f.boundary_inputs, 1);
    assert!(f.reason.contains("late_rejected"), "{}", f.reason);
    let program = auto.program();
    assert!(
        (0..program.node_count()).all(|i| !program.node_meta(i).name.starts_with("jit_cone[")),
        "the cone stays on the interpreter"
    );
    let mut off = late_rejected_program(JitMode::Off)
        .compile()
        .expect("Off compiles");
    for x in [0u64, 1, 7, u64::MAX] {
        auto.set_inputs(&[x]);
        off.set_inputs(&[x]);
        for out in ["tripled", "copied"] {
            assert_eq!(
                auto.pull_ref(out).clone(),
                off.pull_ref(out).clone(),
                "{out} at {x}"
            );
        }
    }
}

/// Under `Force` a cone whose native build fails fails the compile,
/// naming the cone and the error (engines.md §2.1).
#[test]
fn a_failed_cone_build_under_force_fails_the_compile() {
    let err = late_rejected_program(super::cone::JitMode::Force)
        .compile()
        .map(|_| ())
        .expect_err("Force builds native code or fails");
    match &err {
        super::assembly::AssemblyError::NativeCone { cone, reason } => {
            assert!(cone.contains("late_rejected"), "{cone}");
            assert!(reason.contains("late_rejected"), "{reason}");
        }
        other => panic!("expected NativeCone, got {other:?}"),
    }
}
