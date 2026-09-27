// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! What a host-registered node learns from its build context
//! (library_catalog.md, "Host-registered nodes"): the bindings under
//! construction when it was built, and the program tree's resource
//! scope.
//!
//! The nodes here are registered by hand and through `#[polydat_node]`
//! with a setup over the build context, as a host registers one that
//! needs its build context, so this file is a test binary of its own.
//! It also covers a separately compiled image bound under a tree, whose
//! resource scope joins the tree's at bind (scope_model.md §4).

use std::any::Any;
use std::sync::Arc;

use polydat::ast::{NodeMeta, PolydatNode, Port, PortType, Purity, Slot, SlotType, Value};
use polydat::compile::assembly::WireRef;
use polydat::dsl::compile::{
    CompileOptions, compile_polydat_interpreter_with_options, compile_polydat_with_engine,
};
use polydat::dsl::factory::{BuildContext, ConstArg};
use polydat::dsl::registry::{
    Arity, FuncCategory, FuncSig, NodeRegistration, OutputType, ParamSpec,
};
use polydat::dsl::stub::{ExprStub, GraphMatter, ScopedExpr};
use polydat::kernel::bind_under;
use polydat::kernel::subcontext::{BodyFragment, SourceContext, SubcontextBuilder};
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

// ── Macro nodes that read their build context ──────────────────────

/// What a control-setting node keeps from its build context: the
/// binding chain it was built under, outermost first, joined with `/`.
#[derive(Clone)]
pub struct ControlOrigin {
    chain: String,
}

impl ControlOrigin {
    fn capture(ctx: &BuildContext) -> Self {
        Self {
            chain: ctx.bindings().join("/"),
        }
    }
}

/// `macro_control_set(name, value)`, modeled on the host's
/// `control_set`: the binding chain that set the control, then
/// `name=value`, as `chain:name=value`.
#[polydat::polydat_node(category = Context)]
fn macro_control_set(
    name: Const<&str>,
    value: u64,
    #[poly_const(ControlOrigin::capture, from = ctx)] origin: &ControlOrigin,
) -> String {
    format!("{}:{}={value}", origin.chain, name.0)
}

/// `macro_control_tag(s)`: its own binding chain around its input, so
/// a nested call shows both constructions.
#[polydat::polydat_node(category = Context)]
fn macro_control_tag(
    s: &str,
    #[poly_const(ControlOrigin::capture, from = ctx)] origin: &ControlOrigin,
) -> String {
    format!("{}<{s}>", origin.chain)
}

/// The host's live session, as its accessor hands it out.
pub struct SessionHandle {
    id: u64,
}

/// What a session-reading node keeps: the key it was configured with
/// and the resource scope of the tree it was built in.
#[derive(Clone)]
pub struct SessionRef {
    key: String,
    resources: ResourceScope,
}

impl SessionRef {
    fn capture(ctx: &BuildContext, key: &str) -> Self {
        Self {
            key: key.to_string(),
            resources: ctx.resources().clone(),
        }
    }
}

/// `macro_cql_session(key)`, modeled on the host's `cql_session`: the
/// id of the session the tree's accessor holds under `key`, or 0 when
/// it holds none.
#[polydat::polydat_node(
    category = Context,
    purity = Nondeterministic("reads a live host resource"),
)]
fn macro_cql_session(
    key: Const<&str>,
    #[poly_const(SessionRef::capture, from = (ctx, key))] session: &SessionRef,
) -> u64 {
    session
        .resources
        .lookup(&session.key)
        .and_then(|v| v.downcast::<SessionHandle>().ok())
        .map(|h| h.id)
        .unwrap_or(0)
}

/// One session under one key.
struct Sessions(&'static str, u64);

impl ResourceAccessor for Sessions {
    fn lookup(&self, key: &str) -> Option<Arc<dyn Any + Send + Sync>> {
        (key == self.0)
            .then(|| Arc::new(SessionHandle { id: self.1 }) as Arc<dyn Any + Send + Sync>)
    }
}

fn sessions(key: &'static str, id: u64) -> ResourceScope {
    ResourceScope::with_accessor(Arc::new(Sessions(key, id)))
}

/// The four engines: the interpreter, the closure tier, native, and
/// pure native (the last two with the `jit` feature).
fn four_engines() -> Vec<Engine> {
    let mut all = vec![
        Engine::Interpreter(polydat::JitMode::Off),
        Engine::Closures(Provenance::Auto),
    ];
    if cfg!(feature = "jit") {
        all.push(Engine::Native(Provenance::Auto));
        all.push(Engine::PureNative(Provenance::Auto));
    }
    all
}

fn compile_on(src: &str, engine: Engine, resources: Option<ResourceScope>) -> Box<dyn Kernel> {
    let options = CompileOptions {
        resources,
        ..CompileOptions::default()
    };
    compile_polydat_with_engine(src, engine, &options, None)
        .unwrap_or_else(|e| panic!("{engine}: {e}\n{src}"))
}

#[test]
fn a_macro_node_captures_its_binding_on_every_engine() {
    let src = "input cycle: u64\nrate_adj := macro_control_set(\"rate\", cycle)\n";
    for engine in four_engines() {
        let mut k = compile_on(src, engine, None);
        k.set_inputs(&[3]);
        assert_eq!(text(k.as_mut(), "rate_adj"), "rate_adj:rate=3", "{engine}");
        let mut forked = k.fork();
        forked.set_inputs(&[4]);
        assert_eq!(
            text(forked.as_mut(), "rate_adj"),
            "rate_adj:rate=4",
            "{engine}: fork"
        );
    }
}

#[test]
fn a_nested_macro_node_captures_the_whole_chain_on_every_engine() {
    let src = "input cycle: u64\nouter := macro_control_tag(macro_control_set(\"rate\", cycle))\n";
    for engine in four_engines() {
        let mut k = compile_on(src, engine, None);
        k.set_inputs(&[5]);
        let got = text(k.as_mut(), "outer");
        let inner = got
            .strip_prefix("outer<")
            .and_then(|s| s.strip_suffix(":rate=5>"))
            .unwrap_or_else(|| panic!("{engine}: the outer node's chain is `outer`: {got}"));
        let chain: Vec<&str> = inner.split('/').collect();
        assert_eq!(chain.len(), 2, "{engine}: {got}");
        assert_eq!(chain[0], "outer", "{engine}: {got}");
        assert!(chain[1].starts_with("outer__anon_"), "{engine}: {got}");
    }
}

#[test]
fn a_macro_node_looks_its_resource_up_when_it_evaluates_on_every_engine() {
    let src = "input cycle: u64\nsession := macro_cql_session(\"cluster-a\")\n";
    for engine in four_engines() {
        let mut k = compile_on(src, engine, Some(sessions("cluster-a", 17)));
        assert_eq!(k.pull("session"), Value::U64(17), "{engine}");
        assert_eq!(k.fork().pull("session"), Value::U64(17), "{engine}: fork");

        // Installed after the compile: the node reads it on the next
        // evaluation, so nothing was folded at build.
        let mut late = compile_on(src, engine, None);
        assert_eq!(late.pull("session"), Value::U64(0), "{engine}: none yet");
        late.resources()
            .install(Arc::new(Sessions("cluster-a", 23)))
            .unwrap_or_else(|_| panic!("{engine}: the tree had no accessor"));
        late.set_inputs(&[1]);
        assert_eq!(late.pull("session"), Value::U64(23), "{engine}: installed");
    }
}

#[test]
fn a_macro_node_is_built_directly_with_a_context() {
    let node = MacroControlSet::new(&BuildContext::with_binding("rate_adj"), "rate".into());
    let mut out = [Value::None];
    node.eval(&[Value::U64(2)], &mut out);
    assert_eq!(out[0], Value::Str("rate_adj:rate=2".into()));

    let ctx = BuildContext::new(Vec::new(), sessions("k", 5));
    let node = MacroCqlSession::new(&ctx, "k".into());
    node.eval(&[], &mut out);
    assert_eq!(out[0], Value::U64(5));

    // Both have a compiled form: the kit clones what the node captured.
    let engine = Engine::Closures(Provenance::Auto);
    assert!(node.compiled_slot(&[], engine).is_some());
    let node = MacroControlSet::new(&BuildContext::with_binding("r"), "rate".into());
    assert!(node.compiled_slot(&[PortType::U64], engine).is_some());
}

// ── Images bound under a tree ──────────────────────────────────────

const SESSION_READ: &str = "input cycle: u64\nsession := macro_cql_session(\"cluster-a\")\n";

#[test]
fn an_image_bound_under_a_root_reads_the_roots_resource_on_every_engine() {
    for engine in four_engines() {
        let root = compile_on(
            "input cycle: u64\n",
            engine,
            Some(sessions("cluster-a", 31)),
        );
        // Compiled on its own, as a host compiles a fiber image.
        let image = compile_on(SESSION_READ, engine, None).into_program();
        assert!(!image.resources().is_installed(), "{engine}");

        let mut child = bind_under(root.as_ref(), image.clone(), &[])
            .unwrap_or_else(|e| panic!("{engine}: {e}"));
        assert!(
            image
                .resources()
                .parent()
                .is_some_and(|p| p.same_scope(root.resources())),
            "{engine}: the image's scope joined the root's"
        );
        assert_eq!(child.pull("session"), Value::U64(31), "{engine}: bound");
        assert_eq!(
            child.fork().pull("session"),
            Value::U64(31),
            "{engine}: a fork of the bound child"
        );
        assert_eq!(
            image.clone().create_kernel().pull("session"),
            Value::U64(31),
            "{engine}: a kernel created from the image's program"
        );

        // A second bind under the same tree, through a fork of the root.
        let again = bind_under(root.fork().as_ref(), image.clone(), &[]);
        assert!(again.is_ok(), "{engine}: rebinding under the same tree");
    }
}

#[test]
fn an_image_with_its_own_accessor_keeps_it() {
    for engine in four_engines() {
        let root = compile_on(
            "input cycle: u64\n",
            engine,
            Some(sessions("cluster-a", 31)),
        );
        let image =
            compile_on(SESSION_READ, engine, Some(sessions("cluster-a", 47))).into_program();
        let mut child = bind_under(root.as_ref(), image.clone(), &[])
            .unwrap_or_else(|e| panic!("{engine}: {e}"));
        assert!(image.resources().parent().is_none(), "{engine}");
        assert_eq!(child.pull("session"), Value::U64(47), "{engine}");
    }
}

#[test]
fn an_image_resolves_through_the_first_tree_it_is_bound_under() {
    let engine = Engine::Closures(Provenance::Auto);
    let a = compile_on("input cycle: u64\n", engine, Some(sessions("cluster-a", 1)));
    let b = compile_on("input cycle: u64\n", engine, Some(sessions("cluster-a", 2)));
    let image = compile_on(SESSION_READ, engine, None).into_program();
    bind_under(a.as_ref(), image.clone(), &[]).expect("the first tree");
    let mut under_b = bind_under(b.as_ref(), image.clone(), &[]).expect("a second tree binds");
    assert_eq!(
        under_b.pull("session"),
        Value::U64(1),
        "the first bind wins"
    );
    assert!(
        image
            .resources()
            .parent()
            .is_some_and(|p| p.same_scope(a.resources()))
    );
}

#[test]
fn a_root_bound_under_its_own_descendant_stays_a_root() {
    let engine = Engine::Closures(Provenance::Auto);
    let root = compile_on(SESSION_READ, engine, None);
    let image = compile_on("input cycle: u64\n", engine, None).into_program();
    let child = bind_under(root.as_ref(), image.clone(), &[]).expect("child");
    // The root's own program, bound under a kernel of its child.
    let root_program = root.fork().into_program();
    let mut again =
        bind_under(child.as_ref(), root_program.clone(), &[]).expect("no cycle is made");
    assert!(root_program.resources().parent().is_none());
    root.resources()
        .install(Arc::new(Sessions("cluster-a", 9)))
        .unwrap_or_else(|_| panic!("the root had no accessor"));
    assert_eq!(again.pull("session"), Value::U64(9));
}

#[test]
fn a_module_instantiated_under_a_root_reads_the_roots_resource_on_every_engine() {
    // The module is built against an analysis tree of its own, as a
    // host that analyses on the interpreter builds it, and instantiated
    // under the tree the host runs.
    let analysis = interpreter("input cycle: u64\n", None);
    let mut b = SubcontextBuilder::under(analysis.as_ref());
    b.context(SourceContext::new("sessions"));
    b.body(BodyFragment::PolydatSource(SESSION_READ.to_string()));
    let module = b.finalize().unwrap_or_else(|e| panic!("{e:?}"));
    let root = interpreter("input cycle: u64\n", Some(sessions("cluster-a", 55)));
    for engine in four_engines() {
        let mut child = module
            .instantiate_under(root.as_ref(), engine, &[])
            .unwrap_or_else(|e| panic!("{engine}: {e}"));
        assert_eq!(child.pull("session"), Value::U64(55), "{engine}");
        assert_eq!(
            child.fork().pull("session"),
            Value::U64(55),
            "{engine}: fork"
        );
    }
}
