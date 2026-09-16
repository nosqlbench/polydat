// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! SRD-105 lattice report — what extraction actually did.
//!
//! Walks a compiled program and reports its engine mix: which
//! cones formed (members, boundary shape), which nodes stayed on
//! the interpreter, and the lattice headroom per node — P3
//! classifiability and whether the node carries a `compiled_u64`
//! op, the cheapest P2 form (a node with a slot kit runs on P2
//! too). The headroom column is the standing evidence feed for the
//! parked "P2 closures at cone boundaries" extension (SRD-105
//! §Rejected alternatives addendum): nodes that are u64-capable but
//! not JIT-classifiable currently run P1 dyn dispatch.
//!
//! Used by the cone-formation tests (`tests/core_with_library.rs`).

use crate::kernel::PolydatProgram;

/// One fused cone in the compiled program.
pub struct ConeEntry {
    /// The cone node's diagnostic label (`jit_cone[…]`).
    pub label: String,
    /// Member function names, in fusion order.
    pub members: Vec<String>,
    /// Boundary inputs the cone reads.
    pub boundary_in: usize,
    /// Boundary outputs it writes.
    pub boundary_out: usize,
}

/// One node left on the interpreter.
pub struct ResidueEntry {
    /// The node's function name.
    pub name: String,
    /// The P3 classifier can lower this node (it stayed unfused
    /// for lifecycle / threshold / boundary reasons).
    pub p3_classifiable: bool,
    /// The node carries a `compiled_u64` op, the cheapest P2 form,
    /// so the P2 middle rung could run it even though P3 can't. A
    /// node with a slot kit runs on P2 too but is not counted here.
    pub p2_capable: bool,
}

/// Engine-mix report for one compiled program.
pub struct LatticeReport {
    /// The fused cones, in program order.
    pub cones: Vec<ConeEntry>,
    /// Total nodes fused into cones (sum of members).
    pub fused_nodes: usize,
    /// The nodes left on the interpreter, in program order.
    pub residue: Vec<ResidueEntry>,
    /// Residue nodes with a `compiled_u64` op but no P3
    /// classification — the "P2 closures at cone boundaries"
    /// candidate set.
    pub p2_headroom: usize,
    /// Residue nodes the P3 classifier CAN lower that still ended
    /// up interpreted (const/scope-init lifecycle, threshold,
    /// boundary types).
    pub p3_unfused: usize,
}

#[cfg(feature = "jit")]
fn p3_classifiable(node: &dyn crate::ast::PolydatNode) -> bool {
    !matches!(
        crate::compile::jit::classify_node(node),
        crate::compile::jit::JitOp::Fallback
    )
}

#[cfg(not(feature = "jit"))]
fn p3_classifiable(_node: &dyn crate::ast::PolydatNode) -> bool {
    false
}

/// Walk `program` and report its engine mix.
pub fn lattice_report(program: &PolydatProgram) -> LatticeReport {
    let mut cones = Vec::new();
    let mut residue = Vec::new();
    let mut fused_nodes = 0;
    for i in 0..program.node_count() {
        let node = program.node_ref(i);
        if let Some(sub) = node.fusion_subgraph() {
            let members: Vec<String> = sub.members.iter().map(|m| m.meta().name.clone()).collect();
            fused_nodes += members.len();
            cones.push(ConeEntry {
                label: node.meta().name.clone(),
                members,
                boundary_in: node.meta().wire_inputs().len(),
                boundary_out: node.meta().outs.len(),
            });
        } else {
            residue.push(ResidueEntry {
                name: node.meta().name.clone(),
                p3_classifiable: p3_classifiable(node),
                p2_capable: node.compiled_u64().is_some(),
            });
        }
    }
    let p2_headroom = residue
        .iter()
        .filter(|r| r.p2_capable && !r.p3_classifiable)
        .count();
    let p3_unfused = residue.iter().filter(|r| r.p3_classifiable).count();
    LatticeReport {
        cones,
        fused_nodes,
        residue,
        p2_headroom,
        p3_unfused,
    }
}

impl std::fmt::Display for LatticeReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "lattice: {} cone{} ({} node{} fused), {} interpreter node{}",
            self.cones.len(),
            if self.cones.len() == 1 { "" } else { "s" },
            self.fused_nodes,
            if self.fused_nodes == 1 { "" } else { "s" },
            self.residue.len(),
            if self.residue.len() == 1 { "" } else { "s" },
        )?;
        for cone in &self.cones {
            writeln!(
                f,
                "  cone {} — {} member{}, boundary {}→{}",
                cone.label,
                cone.members.len(),
                if cone.members.len() == 1 { "" } else { "s" },
                cone.boundary_in,
                cone.boundary_out,
            )?;
        }
        for r in &self.residue {
            let tier = match (r.p3_classifiable, r.p2_capable) {
                (true, _) => "p3-classifiable, unfused",
                (false, true) => "p2-capable (headroom)",
                (false, false) => "p1-only",
            };
            writeln!(f, "  interp {:30} [{tier}]", r.name)?;
        }
        if self.p2_headroom > 0 {
            writeln!(
                f,
                "  headroom: {} node{} p2-capable without p3 — candidates \
                 for P2-at-cone-boundaries (SRD-105)",
                self.p2_headroom,
                if self.p2_headroom == 1 { "" } else { "s" },
            )?;
        }
        Ok(())
    }
}
