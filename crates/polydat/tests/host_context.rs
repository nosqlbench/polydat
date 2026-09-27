// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! What a host-registered node learns from its build context
//! (library_catalog.md, "Host-registered nodes"): the bindings under
//! construction when it was built, and the program tree's resource
//! scope.
//!
//! The nodes here are registered by hand, as a host registers one that
//! needs its build context, so this file is a test binary of its own.

use std::any::Any;
use std::sync::Arc;

use polydat::ast::{NodeMeta, PolydatNode, Port, PortType, Purity, Slot, SlotType, Value};
use polydat::compile::assembly::WireRef;
use polydat::dsl::compile::{CompileOptions, compile_polydat_interpreter_with_options};
use polydat::dsl::factory::{BuildContext, ConstArg};
use polydat::dsl::registry::{
    Arity, FuncCategory, FuncSig, NodeRegistration, OutputType, ParamSpec,
};
use polydat::dsl::stub::{ExprStub, GraphMatter, ScopedExpr};
use polydat::{Engine, Kernel, Provenance, ResourceAccessor, ResourceScope};

// ── The host's nodes ───────────────────────────────────────────────

/// `host_binding_chain()`: the build context's binding chain, joined
/// with `/`. `host_binding_tag(s)`: the chain, then its input in angle
/// brackets, so a nested call shows both constructions.
struct Attribution {
    meta: NodeMeta,
    chain: String,
}

impl PolydatNode for Attribution {
    fn meta(&self) -> &NodeMeta {
        &self.meta
    }
    fn eval(&self, inputs: &[Value], outputs: &mut [Value]) {
        let text = match inputs.first() {
            Some(Value::Str(s)) => format!("{}<{s}>", self.chain),
            _ => self.chain.clone(),
        };
        outputs[0] = Value::Str(text.as_str().into());
    }
}

/// `host_resource("key")`: the `u64` the tree's accessor holds under
/// `key`, or 0 when it holds none. Nondeterministic, as a node reading
/// a live resource is.
struct ResourceRead {
    meta: NodeMeta,
    key: String,
    resources: ResourceScope,
}

impl PolydatNode for ResourceRead {
    fn meta(&self) -> &NodeMeta {
        &self.meta
    }
    fn eval(&self, _inputs: &[Value], outputs: &mut [Value]) {
        let found = self
            .resources
            .lookup(&self.key)
            .and_then(|v| v.downcast::<u64>().ok())
            .map(|v| *v)
            .unwrap_or(0);
        outputs[0] = Value::U64(found);
    }
    fn purity(&self) -> Purity {
        Purity::Nondeterministic {
            reason: "reads a live host resource",
        }
    }
}

const fn sig(name: &'static str, params: &'static [ParamSpec], output: PortType) -> FuncSig {
    FuncSig {
        name,
        category: FuncCategory::Context,
        outputs: 1,
        description: "a host node reading its build context",
        help: "",
        identity: None,
        variadic_ctor: None,
        params,
        arity: Arity::Fixed,
        commutativity: polydat::ast::Commutativity::Positional,
        default_resolver: None,
        output_type: OutputType::Fixed,
        output_port: Some(output),
    }
}

static SIGS: &[FuncSig] = &[
    sig("host_binding_chain", &[], PortType::Str),
    sig(
        "host_binding_tag",
        &[ParamSpec {
            name: "s",
            slot_type: SlotType::Wire,
            required: true,
            example: "\"a\"",
            constraint: None,
        }],
        PortType::Str,
    ),
    sig(
        "host_resource",
        &[ParamSpec {
            name: "key",
            slot_type: SlotType::ConstStr,
            required: true,
            example: "\"session\"",
            constraint: None,
        }],
        PortType::U64,
    ),
];

fn signatures() -> &'static [FuncSig] {
    SIGS
}

fn build(
    ctx: &BuildContext,
    name: &str,
    _wires: &[WireRef],
    _wire_types: &[PortType],
    consts: &[ConstArg],
) -> Option<Result<Box<dyn PolydatNode>, String>> {
    let chain = ctx.bindings().join("/");
    let node: Box<dyn PolydatNode> = match name {
        "host_binding_chain" => Box::new(Attribution {
            meta: NodeMeta {
                name: name.into(),
                ins: vec![],
                outs: vec![Port::new("out", PortType::Str)],
            },
            chain,
        }),
        "host_binding_tag" => Box::new(Attribution {
            meta: NodeMeta {
                name: name.into(),
                ins: vec![Slot::Wire(Port::new("s", PortType::Str))],
                outs: vec![Port::new("out", PortType::Str)],
            },
            chain,
        }),
        "host_resource" => {
            let key = consts.first().map(|c| c.as_str()).unwrap_or("").to_string();
            Box::new(ResourceRead {
                meta: NodeMeta {
                    name: name.into(),
                    ins: vec![Slot::Const {
                        name: "key".into(),
                        value: polydat::ast::ConstValue::Str(key.clone()),
                    }],
                    outs: vec![Port::new("out", PortType::U64)],
                },
                key,
                resources: ctx.resources().clone(),
            })
        }
        _ => return None,
    };
    Some(Ok(node))
}

polydat::inventory::submit! {
    NodeRegistration { signatures, build, validate: None }
}

// ── The host's accessor ────────────────────────────────────────────

/// One `u64` under one key.
struct Holds(&'static str, u64);

impl ResourceAccessor for Holds {
    fn lookup(&self, key: &str) -> Option<Arc<dyn Any + Send + Sync>> {
        (key == self.0).then(|| Arc::new(self.1) as Arc<dyn Any + Send + Sync>)
    }
}

fn scope_holding(key: &'static str, value: u64) -> ResourceScope {
    ResourceScope::with_accessor(Arc::new(Holds(key, value)))
}

fn interpreter(src: &str, resources: Option<ResourceScope>) -> Box<dyn Kernel> {
    let options = CompileOptions {
        resources,
        ..CompileOptions::default()
    };
    Box::new(
        compile_polydat_interpreter_with_options(src, &options, None)
            .unwrap_or_else(|e| panic!("compile failed: {e}\n{src}")),
    )
}

fn text(k: &mut dyn Kernel, name: &str) -> String {
    match k.pull(name) {
        Value::Str(s) => s.to_string(),
        other => panic!("{name}: expected a string, got {other:?}"),
    }
}

// ── Attribution ────────────────────────────────────────────────────

#[test]
fn a_node_is_attributed_to_the_binding_that_builds_it() {
    let mut k = interpreter("input cycle: u64\nrate_adj := host_binding_chain()\n", None);
    assert_eq!(text(k.as_mut(), "rate_adj"), "rate_adj");
}

#[test]
fn a_node_built_inside_another_binding_sees_the_whole_chain() {
    // The argument compiles as an intermediate binding named after the
    // enclosing one, so the inner node's chain is two deep and the
    // outer node's is one.
    let mut k = interpreter(
        "input cycle: u64\nouter := host_binding_tag(host_binding_chain())\n",
        None,
    );
    let got = text(k.as_mut(), "outer");
    let inner = got
        .strip_prefix("outer<")
        .and_then(|s| s.strip_suffix('>'))
        .unwrap_or_else(|| panic!("the outer node's chain is `outer`: {got}"));
    let chain: Vec<&str> = inner.split('/').collect();
    assert_eq!(chain.len(), 2, "{got}");
    assert_eq!(chain[0], "outer", "{got}");
    assert!(chain[1].starts_with("outer__anon_"), "{got}");
}

#[test]
fn a_module_body_is_attributed_under_the_binding_that_calls_it() {
    let src = "input cycle: u64\n\
               tagged(x: u64) -> (label: str) := {\n    label := host_binding_chain()\n}\n\
               caller := tagged(x: cycle)\n";
    let mut k = interpreter(src, None);
    let got = text(k.as_mut(), "caller");
    let chain: Vec<&str> = got.split('/').collect();
    // The module's binding compiles under the call's prefix,
    // `__<module>_<n>_`, beneath the binding that called it.
    assert_eq!(chain.len(), 2, "{got}");
    assert_eq!(chain[0], "caller", "{got}");
    assert!(
        chain[1].starts_with("__tagged_") && chain[1].ends_with("_label"),
        "{got}"
    );
}

#[test]
fn attribution_is_exact_on_concurrent_compiles() {
    let handles: Vec<_> = (0..8)
        .map(|i| {
            std::thread::spawn(move || {
                let name = format!("b{i}");
                let src = format!("input cycle: u64\n{name} := host_binding_chain()\n");
                let mut k = interpreter(&src, None);
                (name.clone(), text(k.as_mut(), &name))
            })
        })
        .collect();
    for h in handles {
        let (name, got) = h.join().unwrap();
        assert_eq!(got, name);
    }
}

// ── Resources ──────────────────────────────────────────────────────

const READS: &str = "input cycle: u64\nsession := host_resource(\"session\")\n";

#[test]
fn two_trees_in_one_process_see_their_own_accessors() {
    let mut a = interpreter(READS, Some(scope_holding("session", 11)));
    let mut b = interpreter(READS, Some(scope_holding("session", 22)));
    assert_eq!(a.pull("session"), Value::U64(11));
    assert_eq!(b.pull("session"), Value::U64(22));
    assert!(!a.resources().same_scope(b.resources()));
}

#[test]
fn a_tree_without_an_accessor_reads_none_until_one_is_installed() {
    let mut k = interpreter(READS, None);
    assert_eq!(k.pull("session"), Value::U64(0));
    k.resources()
        .install(Arc::new(Holds("session", 5)))
        .unwrap_or_else(|_| panic!("the tree had no accessor"));
    k.invalidate_all();
    assert_eq!(k.pull("session"), Value::U64(5));
    assert!(
        k.resources()
            .install(Arc::new(Holds("session", 6)))
            .is_err(),
        "a tree has one accessor"
    );
}

#[test]
fn created_and_forked_kernels_share_the_trees_scope() {
    let scope = scope_holding("session", 3);
    let k = interpreter(READS, Some(scope.clone()));
    let mut forked = k.fork();
    assert!(forked.resources().same_scope(&scope));
    assert_eq!(forked.pull("session"), Value::U64(3));
    let program = k.into_program();
    assert!(program.resources().same_scope(&scope));
    let mut created = program.create_kernel();
    assert!(created.resources().same_scope(&scope));
    assert_eq!(created.pull("session"), Value::U64(3));
}

#[test]
fn a_subscope_reads_through_its_parents_accessor() {
    let parent = interpreter(
        "input cycle: u64\nx := 5\n",
        Some(scope_holding("session", 9)),
    );
    let mut matter = GraphMatter::new();
    matter.bind(ExprStub::parse("__r", "host_resource(\"session\")").expect("parse"));
    let mut child = ScopedExpr::bind(parent.as_ref(), "__r", matter).expect("bind");
    assert!(child.kernel().resources().same_scope(parent.resources()));
    assert_eq!(child.eval(), Value::U64(9));
}

#[test]
fn a_for_body_reads_through_its_parents_accessor() {
    let src = "input cycle: u64\nfor k in 1..3 {\n    s := host_resource(\"session\")\n}\n";
    let mut parent = interpreter(src, Some(scope_holding("session", 4)));
    let stream = parent.traverse(0).expect("traverse");
    let mut act = stream.activation(0).expect("activation");
    assert!(act.kernel.resources().same_scope(parent.resources()));
    assert_eq!(act.kernel.pull("s"), Value::U64(4));
}

/// A skeleton with one projection over `k in 0..3` whose body reads
/// the tree's `session` resource, rendered as text.
fn resource_skeleton() -> String {
    use polydat::library::tile_render::{ChildSpec, HoleSource, TileOp, TileSpec};
    TileSpec {
        name: "t".into(),
        encoding: "text".into(),
        ops: vec![TileOp::Repeat {
            stream: polydat::iteration::comprehension::StreamerValue::parse_text("k in 0..3")
                .expect("comprehension")
                .to_json(),
            child: 0,
            sep: ",".into(),
            body: vec![TileOp::Hole(HoleSource::Child {
                name: "s".into(),
                spec: "text|text".into(),
            })],
            generators: Vec::new(),
        }],
        children: vec![ChildSpec {
            source: "input cycle: u64\nextern k: u64\ns := host_resource(\"session\")\n".into(),
            cascade: Vec::new(),
        }],
    }
    .to_json()
}

/// Every piece of a rendered projection, which is one value per tuple.
fn pieces(rendered: &str) -> Vec<&str> {
    assert!(!rendered.is_empty(), "the projection renders its tuples");
    rendered.split(',').collect()
}

#[test]
fn a_tile_loaded_for_a_kernel_reads_its_trees_accessor_and_records_on_its_ledger() {
    use polydat::library::tile_render::{BodyKernels, TileProgram};
    let k = interpreter(READS, Some(scope_holding("session", 4)));
    let before = k.ledger().programs();
    let tile = TileProgram::from_json_for(&resource_skeleton(), k.as_ref());
    assert!(
        k.ledger().programs() > before,
        "the loaded body's program is recorded on the kernel's ledger"
    );
    let rendered = tile.render(&[], &mut BodyKernels::default());
    assert!(pieces(&rendered).iter().all(|p| *p == "4"), "{rendered}");
}

#[test]
fn a_standalone_tile_has_a_tree_of_its_own() {
    use polydat::library::tile_render::{BodyKernels, TileProgram};
    let k = interpreter(READS, Some(scope_holding("session", 4)));
    let before = k.ledger().programs();
    let tile = TileProgram::from_json(&resource_skeleton());
    assert_eq!(
        k.ledger().programs(),
        before,
        "no kernel's ledger records it"
    );
    let rendered = tile.render(&[], &mut BodyKernels::default());
    assert!(pieces(&rendered).iter().all(|p| *p == "0"), "{rendered}");
}

#[test]
fn every_engine_carries_the_trees_scope_to_its_children() {
    let src = "input cycle: u64\nh := hash(cycle)\nfor k in 1..3 {\n    g := u64_add(k, h)\n}\n";
    let mut engines = vec![
        Engine::Interpreter(polydat::JitMode::Off),
        Engine::Closures(Provenance::Auto),
    ];
    if cfg!(feature = "jit") {
        engines.push(Engine::Native(Provenance::Auto));
    }
    for engine in engines {
        let scope = ResourceScope::new();
        let options = CompileOptions {
            resources: Some(scope.clone()),
            ..CompileOptions::default()
        };
        let mut k = polydat::dsl::compile::compile_polydat_with_engine(src, engine, &options, None)
            .unwrap_or_else(|e| panic!("{engine}: {e}"));
        assert!(k.resources().same_scope(&scope), "{engine}: root");
        assert!(k.fork().resources().same_scope(&scope), "{engine}: fork");
        let stream = k.traverse(0).expect("traverse");
        let act = stream.activation_on(0, engine).expect("activation");
        assert!(
            act.kernel.resources().same_scope(&scope),
            "{engine}: for body"
        );
        let created = k.into_program().create_kernel();
        assert!(created.resources().same_scope(&scope), "{engine}: created");
    }
}
