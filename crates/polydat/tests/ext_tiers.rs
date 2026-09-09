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
    #[cfg(feature = "jit")]
    {
        assert!(native >= 1, "the scalar prefix should still be native, got {native}");
        assert!(compile_polydat_to_assembler(SRC).unwrap().try_compile_jit().is_err(), "pure native code has no form for an extension node");
    }
    #[cfg(not(feature = "jit"))]
    assert_eq!(native, 0, "nothing is native without the jit feature");
}

// ── Fallible construction and tuple returns on the closure tier ──

/// A fallible node runs its body once at construction and replays the
/// cached value; the closure kits write that value every run.
#[polydat::polydat_node(category = Math)]
fn fixed_seed(base: polydat::derive_support::Const<u64>) -> Result<u64, String> {
    if *base == 0 { Err("fixed_seed: base must be non-zero".into()) } else { Ok(base.wrapping_mul(0x9E37_79B9_7F4A_7C15)) }
}

#[polydat::polydat_node(category = String)]
fn fixed_label(prefix: polydat::derive_support::Const<&str>) -> Result<String, String> {
    if prefix.is_empty() { Err("fixed_label: empty prefix".into()) } else { Ok(format!("{}-fixed", *prefix)) }
}

#[polydat::polydat_node(category = Math, struct_name = FixedSpanNode)]
fn fixed_span(lo: polydat::derive_support::Const<u64>, hi: polydat::derive_support::Const<u64>) -> Result<Ext<Span>, String> {
    if *lo >= *hi { Err("fixed_span: empty".into()) } else { Ok(Ext(Span { lo: *lo, hi: *hi })) }
}

#[polydat::polydat_node(category = Math)]
fn fixed_doc(n: polydat::derive_support::Const<u64>) -> Result<std::sync::Arc<serde_json::Value>, String> {
    Ok(std::sync::Arc::new(serde_json::json!({ "n": *n, "twice": *n * 2 })))
}

/// A tuple return with one element of every shape the handle kit
/// writes: a carrier, a string, an extension value, and JSON.
#[polydat::polydat_node(category = Math, output_names(len, text, moved, doc))]
fn span_parts(span: Ext<Span>, by: u64) -> (u64, String, Ext<Span>, std::sync::Arc<serde_json::Value>) {
    let moved = Span { lo: span.lo + by, hi: span.hi + by };
    let doc = std::sync::Arc::new(serde_json::json!({ "lo": moved.lo, "hi": moved.hi }));
    (span.hi - span.lo, span.display(), Ext(moved), doc)
}

const SHAPES: &str = "input cycle: u64\n\
    h := hash(cycle)\n\
    seed := fixed_seed(7)\n\
    label := fixed_label(\"job\")\n\
    fs := fixed_span(10, 20)\n\
    fd := fixed_doc(3)\n\
    s := span_of(mod(h, 1000), mod(cycle, 7))\n\
    (len, text, moved, doc) := span_parts(s, mod(seed, 100))\n\
    tail := span_len(moved)\n\
    line := \"{label}/{len}/{text}/{tail}/{fs}/{fd}/{doc}\"\n";

#[test]
fn fallible_and_tuple_nodes_agree_between_interpreter_closures_and_hybrid() {
    let mut p1 = compile_polydat_to_assembler(SHAPES).unwrap();
    p1.set_jit_mode(polydat::JitMode::Off);
    let mut p1 = p1.compile().expect("P1");
    let mut p2 = compile_polydat_to_assembler(SHAPES).unwrap().try_compile_raw().unwrap_or_else(|k| {
        let p = k.program();
        let names: Vec<String> = (0..p.node_count()).map(|i| p.node_meta(i).name.clone()).collect();
        panic!("the closure tier refused a fallible or tuple node; nodes: {names:?}")
    });
    let mut hybrid = compile_polydat_to_assembler(SHAPES).unwrap().compile_hybrid().expect("hybrid");
    let outputs = ["seed", "label", "fs", "fd", "len", "text", "moved", "doc", "tail", "line"];
    for cycle in 0..64u64 {
        p1.set_inputs(&[cycle]);
        let want: Vec<(polydat::ast::PortType, String)> = outputs.iter().map(|o| { let v = p1.pull(o); (v.port_type(), v.to_display_string()) }).collect();
        p2.eval(&[cycle]);
        for (i, o) in outputs.iter().enumerate() {
            let v = p2.get_value(o);
            assert_eq!((v.port_type(), v.to_display_string()), want[i], "cycle {cycle}: P2 `{o}`");
        }
        hybrid.eval(&[cycle]);
        for (i, o) in outputs.iter().enumerate() {
            let v = hybrid.get_value(o);
            assert_eq!((v.port_type(), v.to_display_string()), want[i], "cycle {cycle}: hybrid `{o}`");
        }
    }
    let (_, closures) = hybrid.engine_counts();
    assert!(closures >= 5, "the fallible and tuple nodes run as closure steps, got {closures}");
}

#[test]
fn a_failing_construction_is_a_compile_error_on_every_tier() {
    let src = "input cycle: u64\nseed := fixed_seed(0)\n";
    let err = compile_polydat_to_assembler(src).err().expect("construction fails at compile time");
    assert!(err.contains("base must be non-zero"), "{err}");
}

// ── What the compiled engines cannot take ──

/// An extern is a host-settable input slot. The compiled engines are
/// driven by coordinates alone and seed nothing else, so the assembler
/// entry point refuses a program with one and says where to go.
#[test]
fn the_assembler_entry_refuses_externs_with_direction() {
    let err = compile_polydat_to_assembler("input cycle: u64\nextern label: str = \"lbl\"\nx := str_concat(label, \"!\")\n")
        .err()
        .expect("the assembler entry refuses externs");
    assert!(err.contains("extern 'label'") && err.contains("compile_polydat"), "{err}");
    // The kernel path takes the same program.
    let mut k = polydat::dsl::compile::compile_polydat("input cycle: u64\nextern label: str = \"lbl\"\nx := str_concat(label, \"!\")\n").expect("kernel path");
    k.set_inputs(&[1]);
    assert_eq!(k.pull("x").as_str(), "lbl!");
}
