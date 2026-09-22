// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Phase 3: Cranelift JIT compilation of Polydat Kernels.
//!
//! Generates native machine code from the DAG. The entire kernel
//! becomes a single function over the state's slot buffer and
//! scratch: `fn(coords: *const u64, buffer: *mut u64, scratch: *mut ScratchBuf)`
//! (plus a clean-flag pointer on the provenance variant).
//! Arithmetic is inlined over the buffer; a node with no named
//! lowering runs its own slot kit through one helper call
//! (`JitOp::SlotCall`), with the inputs gathered into the native
//! frame and the outputs scattered back.
//!
//! The buffer is `Vec<u64>`. For f64 values, they are stored as their
//! bit representation (`f64::to_bits()` / `f64::from_bits()`). The JIT
//! uses Cranelift `bitcast` (free, no instruction emitted) to convert
//! between i64 and f64 representations when crossing type boundaries.
//!
//! Feature-gated behind `jit`.
//!
//! Simple ops are fully inlined (hash is an inline splitmix64); ops
//! with a body Cranelift cannot express (xxhash3, shuffle, interleave,
//! the math functions) call an extern helper, and any other node with
//! a kit calls the kit in place.

#[cfg(feature = "jit")]
mod codegen;
#[cfg(feature = "jit")]
pub mod host_isa;
#[cfg(feature = "jit")]
mod kernels;
#[cfg(feature = "jit")]
pub mod simd;

#[cfg(feature = "jit")]
pub use codegen::*;
#[cfg(feature = "jit")]
pub use kernels::*;

#[cfg(all(test, feature = "jit"))]
mod tests {
    use super::*;

    #[test]
    fn test_inventory_tier_distribution() {
        use crate::ast::PortType;
        use crate::compile::assembly::WireRef;
        use crate::dsl::factory::{ConstArg, build_node};
        use crate::dsl::registry::registry;

        let reg = registry();
        let total = reg.len();

        let mut p1_count = 0;
        let mut p2_count = 0;
        let mut p3_count = 0;
        let mut unbuilt = 0;

        for sig in &reg {
            let mut consts = Vec::new();
            for p in sig.params {
                match p.slot_type {
                    crate::ast::SlotType::ConstU64 => consts.push(ConstArg::Int(1)),
                    crate::ast::SlotType::ConstF64 => consts.push(ConstArg::Float(1.0)),
                    crate::ast::SlotType::ConstStr => consts.push(ConstArg::Str("test".into())),
                    crate::ast::SlotType::ConstVecU64 => consts.push(ConstArg::Int(1)),
                    crate::ast::SlotType::ConstVecF64 => consts.push(ConstArg::Float(1.0)),
                    crate::ast::SlotType::ConstVec => consts.push(ConstArg::Int(1)),
                    crate::ast::SlotType::Wire => {}
                }
            }
            let wires = vec![WireRef::Input("cycle".to_string()); sig.wire_input_count().max(1)];
            let wire_types = vec![PortType::U64; wires.len()];

            let node_res = build_node(sig.name, &wires, &wire_types, &consts);
            if let Ok(node) = node_res {
                // Every compiled form, not only the scalar one: a node
                // whose form is a slot kit belongs in the P2 column.
                let p2_eligible = crate::compile::node_tier(node.as_ref(), &wire_types)
                    != crate::ast::CompileLevel::Phase1;
                let p3_eligible = classify_node(node.as_ref()) != JitOp::Fallback;
                if p3_eligible {
                    p3_count += 1;
                } else if p2_eligible {
                    p2_count += 1;
                } else {
                    p1_count += 1;
                }
            } else {
                unbuilt += 1;
                p1_count += 1;
            }
        }

        println!("\n=== COMPILER OPTIMIZATION INVENTORY SUMMARY ===");
        println!("Total Registered Functions: {total}");
        println!(
            "Phase 3 (Full Native JIT):  {p3_count} ({:.1}%)",
            (p3_count as f64 / total as f64) * 100.0
        );
        println!(
            "Phase 2 (Captured Closure): {p2_count} ({:.1}%)",
            (p2_count as f64 / total as f64) * 100.0
        );
        println!(
            "Phase 1 (Interpreter Cones):{p1_count} ({:.1}%) (unbuilt fallback: {unbuilt})",
            (p1_count as f64 / total as f64) * 100.0
        );
        println!("===============================================\n");

        // The tier counts are a claim about the library, so one of
        // them is asserted rather than printed. Asking only
        // `compiled_u64` put 251 of 457 nodes in the P1 column and 2
        // in P2, which said polydat interprets more than half of its
        // own node library; it compiles all but a small tail. A
        // proportion rather than a number, so adding nodes does not
        // fail it, and no list of names, so a node that loses its
        // compiled form is not excused by being on one.
        let p1_share = p1_count as f64 / total as f64;
        assert!(
            p1_share < 0.15,
            "{p1_count} of {total} registered nodes reach no compiled form \
             ({:.1}%); the tier predicate has narrowed",
            p1_share * 100.0
        );
    }
}
