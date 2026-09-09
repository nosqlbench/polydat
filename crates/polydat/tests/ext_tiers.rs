// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Extension values on the closure tier. A host type that implements
//! `ReflectedValue` rides a wire as `Value::Ext`; nodes take and return
//! it through `Ext<T>`. Such a node has a handle closure (SRD 115 §7)
//! that reads and writes the value table, so the closure tier and the
//! hybrid kernel run it, while pure native code still refuses it.

use polydat::ast::{ReflectedValue, Value};
use polydat::derive_support::Ext;
use polydat::dsl::compile::compile_polydat_to_assembler;

#[derive(Debug, Clone, PartialEq)]
struct Span {
    lo: u64,
    hi: u64,
}

impl ReflectedValue for Span {
    fn type_name(&self) -> &str {
        "Span"
    }
    fn display(&self) -> String {
        format!("[{}, {})", self.lo, self.hi)
    }
    fn clone_reflected(&self) -> Box<dyn ReflectedValue> {
        Box::new(self.clone())
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[polydat::polydat_node(category = Math, struct_name = SpanOf)]
fn span_of(center: u64, width: u64) -> Ext<Span> {
    Ext(Span { lo: center.saturating_sub(width), hi: center.saturating_add(width) })
}

#[polydat::polydat_node(category = Math)]
fn span_len(span: Ext<Span>) -> u64 {
    span.hi - span.lo
}

#[polydat::polydat_node(category = Math)]
fn span_shift(span: Ext<Span>, by: u64) -> Ext<Span> {
    Ext(Span { lo: span.lo + by, hi: span.hi + by })
}

#[polydat::polydat_node(category = String)]
fn span_text(span: Ext<Span>) -> String {
    span.display()
}

const SRC: &str = "input cycle: u64\n\
    h := hash(cycle)\n\
    s := span_of(mod(h, 1000), mod(cycle, 7))\n\
    t := span_shift(s, 10)\n\
    n := span_len(t)\n\
    label := span_text(t)\n\
    doc := json_object(json_with(\"n\", n), json_with(\"label\", label))\n";

#[test]
fn extension_nodes_agree_between_interpreter_closures_and_hybrid() {
    let mut p1 = compile_polydat_to_assembler(SRC).unwrap();
    p1.set_jit_mode(polydat::JitMode::Off);
    let mut p1 = p1.compile().expect("P1");
    let mut p2 = compile_polydat_to_assembler(SRC).unwrap().try_compile_raw().unwrap_or_else(|_| panic!("the closure tier refused a program with extension nodes"));
    let mut hybrid = compile_polydat_to_assembler(SRC).unwrap().compile_hybrid().expect("hybrid");
    for cycle in 0..64u64 {
        p1.set_inputs(&[cycle]);
        let n = p1.pull("n").as_u64();
        let label = p1.pull("label").as_str().to_string();
        let doc = p1.pull("doc").to_display_string();
        let Value::Ext(t) = p1.pull("t").clone() else { panic!("t is an Ext wire") };
        let t = t.as_any().downcast_ref::<Span>().expect("a Span").clone();

        p2.eval(&[cycle]);
        assert_eq!(p2.get("n"), n, "cycle {cycle}: P2 n");
        assert_eq!(p2.get_value("label").as_str(), label, "cycle {cycle}: P2 label");
        assert_eq!(p2.get_value("doc").to_display_string(), doc, "cycle {cycle}: P2 doc");
        let Value::Ext(t2) = p2.get_value("t") else { panic!("P2 t is an Ext wire") };
        assert_eq!(t2.as_any().downcast_ref::<Span>(), Some(&t), "cycle {cycle}: P2 t");

        hybrid.eval(&[cycle]);
        assert_eq!(hybrid.get("n"), n, "cycle {cycle}: hybrid n");
        assert_eq!(hybrid.get_value("label").as_str(), label, "cycle {cycle}: hybrid label");
        assert_eq!(hybrid.get_value("doc").to_display_string(), doc, "cycle {cycle}: hybrid doc");
        let Value::Ext(t3) = hybrid.get_value("t") else { panic!("hybrid t is an Ext wire") };
        assert_eq!(t3.as_any().downcast_ref::<Span>(), Some(&t), "cycle {cycle}: hybrid t");
    }
}

#[test]
fn extension_nodes_are_closure_steps_never_native() {
    let hybrid = compile_polydat_to_assembler(SRC).unwrap().compile_hybrid().expect("hybrid");
    let (native, closures) = hybrid.engine_counts();
    assert!(closures >= 3, "the three extension nodes and the text node should be closure steps, got {closures}");
    assert!(native >= 1, "the scalar prefix should still be native, got {native}");
    #[cfg(feature = "jit")]
    assert!(compile_polydat_to_assembler(SRC).unwrap().try_compile_jit().is_err(), "pure native code has no form for an extension node");
}
