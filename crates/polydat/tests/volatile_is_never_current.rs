// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! `volatile` is the author's declaration that a wire's value is not
//! a function of its inputs — the case polydat cannot see for itself,
//! because the node says it is pure. evaluation_model.md
//! §"Non-Deterministic Nodes" states what follows: a node feeding a
//! `volatile` output is excluded from the fold *and* from
//! effectively-const classification, the exclusion is contagious
//! downstream, and "an engine never treats them as current".
//!
//! The fold half held. The runtime half did not: the never-current
//! set was computed before the output modifiers were installed, so it
//! saw only the nodes that declare `Purity::Nondeterministic`, and a
//! volatile binding was cached like any other (F-C5).

use polydat::ast::{NodeMeta, PolydatNode, Port, Slot, Value};
use polydat::compile::assembly::{PolydatAssembler, WireRef};
use polydat::dsl::ast::BindingModifier;
use std::sync::atomic::{AtomicU64, Ordering};

/// A node whose value changes on every evaluation while declaring
/// itself pure — a stand-in for the host node that reads something
/// polydat cannot see, which is what `volatile` is for.
struct Ticker {
    meta: NodeMeta,
    n: AtomicU64,
}

impl Ticker {
    fn new() -> Self {
        Self {
            meta: NodeMeta {
                name: "ticker".into(),
                ins: vec![Slot::Wire(Port::u64("seed"))],
                outs: vec![Port::u64("out")],
            },
            n: AtomicU64::new(0),
        }
    }
}

impl PolydatNode for Ticker {
    fn meta(&self) -> &NodeMeta {
        &self.meta
    }
    fn eval(&self, _inputs: &[Value], outputs: &mut [Value]) {
        outputs[0] = Value::U64(self.n.fetch_add(1, Ordering::Relaxed));
    }
}

fn program(volatile: bool) -> polydat::kernel::PolydatKernel {
    let mut asm = PolydatAssembler::new(vec!["cycle".into()]);
    asm.add_node(
        "k",
        Box::new(polydat::library::identity::ConstU64::new(7)),
        vec![],
    );
    asm.add_node("t", Box::new(Ticker::new()), vec![WireRef::node("k")]);
    asm.add_output("t", WireRef::node("t"));
    if volatile {
        asm.set_output_modifier("t", BindingModifier::VOLATILE);
    }
    asm.compile().expect("it assembles")
}

#[test]
fn a_volatile_binding_is_re_evaluated_every_cycle() {
    use polydat::Kernel;
    let mut k = program(true);
    let mut seen = Vec::new();
    for c in 0..3u64 {
        k.set_inputs(&[c]);
        seen.push(k.pull("t").as_u64());
    }
    assert_eq!(
        seen,
        vec![0, 1, 2],
        "a volatile binding must never be treated as current"
    );
}

#[test]
fn the_same_binding_without_the_modifier_is_cached_per_cycle() {
    use polydat::Kernel;
    let mut k = program(false);
    k.set_inputs(&[1]);
    let first = k.pull("t").as_u64();
    // Two reads inside one cycle are one evaluation, volatile or not;
    // what volatile changes is whether a new cycle re-runs the node.
    assert_eq!(k.pull("t").as_u64(), first);
}
