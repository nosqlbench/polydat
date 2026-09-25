---
type: specification
title: Library Catalog
timestamp: 2026-09-25
description: What a node is, the authoring contract, cost classes, and why the node registry is open.
tags: [library]
---

# Library Catalog

This document specifies the Polydat node library: what a node is, the
authoring contract, the wire cost classes, registration, and why the
registry is open. The library provides deterministic, composable functions
for data generation; nodes are registered in the DSL compiler's function
registry and are available by name in `.polydat` source. The library lives
in `polydat-nodes/src/`; `polydat-core/src/library/` holds the adapters,
passthroughs, constants, assertions, and tile nodes the compiler
synthesizes, plus the conversion, formatting, JSON, data-file, diagnostic,
context, logging, vector, and fixed-value nodes the runtime ships itself.
This document is not a listing; the nodes themselves are in the
[node reference](../reference/nodes.md).

**Related specifications:** the node-metadata contract every entry
satisfies is in
[composition_substrate.md §2 (slot contract)](composition_substrate.md);
fusion of catalog nodes is in
[graph_compiler.md §5](graph_compiler.md); virtual-node
registration is in
[expression_engine.md §5.5](expression_engine.md).

## The node contract

A node is a value implementing `PolydatNode` (`polydat-core/src/ast.rs`).
Two methods are required and every other one has a default. A node must
state what it *is* (`meta`) and what it *computes* (`eval`); it may
additionally declare properties that the compiler and the faster
engines use to optimize it. A node that declares nothing beyond the two
required methods still runs correctly on all four engines (the
interpreter, the closure tier, native, and pure native), through the
interpreter's typed path: `eval` over `Value`s.

```rust
pub trait PolydatNode: Send + Sync {
    fn meta(&self) -> &NodeMeta;
    fn eval(&self, inputs: &[Value], outputs: &mut [Value]);

    fn scratch_layout(&self) -> Vec<ScratchElem> { … }
    fn eval_in(&self, scratch: &mut [ScratchBuf], inputs: &[Value], outputs: &mut [Value]) { … }
    fn commutativity(&self) -> Commutativity { … }
    fn accepts_none_inputs(&self) -> bool { … }
    fn compiled_u64(&self) -> Option<CompiledU64Op> { … }
    fn compiled_slot(&self, wire_types: &[PortType]) -> Option<CompiledSlotKit> { … }
    fn jit_constants(&self) -> Vec<u64> { … }
    fn purity(&self) -> Purity { … }
    fn simd_variant(&self) -> Option<SimdVariant> { … }
    fn fusion_subgraph(&self) -> Option<FusionSubgraph<'_>> { … }
}
```

**The two required methods.** `meta()` returns the `NodeMeta` that
names the node for the DSL and for diagnostics and declares its ports:
`ins` as `Slot`s (each a wire or a const) and `outs` as `Port`s, each
typed. `eval(inputs, outputs)` computes the node: it reads one `Value`
per input from `inputs` and writes one `Value` per output into
`outputs`. The port declarations are the slot contract every entry in
the catalog satisfies, specified in
[composition_substrate.md §2](composition_substrate.md). The assembler
checks arity and types against them, so `eval` may assume its slices
have the right length and hold the right variants, and need not
re-check them.

**The optional declarations**, grouped by the part of the system that
reads them:

- **The runtime reads `purity()` and `accepts_none_inputs()`.**
  `purity()` states whether the node's output is a function of its
  inputs (D2): `Pure` by default, `SideChannel` for a node with an
  observable effect (logging, file writes), and `Nondeterministic` for
  one that keeps state across evaluations. The runtime caches the output
  of `Pure` and `SideChannel` nodes and returns it without re-running
  the node while it is current (the clean-flag cache of
  [runtime_model.md](runtime_model.md) R1); a `Nondeterministic` node's
  output is kept only within one read, and every read that reaches the
  node runs it again (R1.v). A node that declares the wrong purity
  therefore returns values that depend on how often it was pulled.
  `accepts_none_inputs()` returns `true` to opt the node out of the
  kernel's None-in-None-out propagation, under which any `None` input
  makes the output `None` without running the node. The coalescing nodes
  opt out, since their purpose is to distinguish a present value from an
  absent one ([none_semantics.md](none_semantics.md)).
- **The compiler reads `commutativity()` and `fusion_subgraph()`.**
  `commutativity()` states which inputs are interchangeable, so a
  rewrite may reorder them. `fusion_subgraph()` is implemented by the
  synthetic nodes the compiler creates when it fuses several nodes into
  one, and returns the member nodes the fusion replaced. Program-identity
  hashing walks through the fusion node into those members, so the same
  source produces the same hash whichever mix of engines compiled it.
- **The faster engines read `compiled_u64()`, `compiled_slot()`,
  `jit_constants()`, and `simd_variant()`.** The first two return the
  node's **kit**: the compiled closure the closure tier runs and native
  code calls for a node it has no native lowering for (see
  [The kit each shape yields](#the-kit-each-shape-yields)).
  `jit_constants()` returns the constants a native lowering bakes into
  its code, and `simd_variant()` names a vectorized variant of the node.
  Each is optional: a node that returns `None` runs on the typed path
  and loses nothing else. The engines consult them in order: the u64
  closure first, since a pure-scalar node needs nothing else; then the
  slot kit, for nodes with typed-slice ports; then the JIT constants;
  then the SIMD variant, which the planner still validates against
  types, purity, and lowering before it promotes anything.
- **The kernel reads `scratch_layout()` and `eval_in()`.**
  `scratch_layout()` declares the per-kernel working storage (scratch)
  the node needs, and `eval_in(scratch, inputs, outputs)` is `eval` with
  that storage passed in. Scratch storage belongs to the evaluating
  kernel, never to the node, because one node value is shared by every
  kernel created from the program. A node that works over `Value`s alone
  declares no scratch, which is every node but a native cone.

## Wire cost classes

Some node inputs are **configuration wires** — changing them
invalidates expensive internal state (e.g., recomputing a
lookup table for weighted selection). Other inputs are
**data wires** — cheap per-cycle values that drive the
node's primary computation.

Node metadata declares the cost class of each input port through
`WireCost`:

| Class | Semantics | Example |
|-------|-----------|---------|
| `config` | Expensive to change. Initializes internal state (LUT, distribution table). Expected to be wired to effectively-const sources (per [Evaluation Model](evaluation_model.md): compile-const, scope-init, or iteration externs). | `weighted_strings` weights parameter |
| `data` | Cheap dynamic input. The node's primary computation path. | `hash` input value, `mod` dividend |

The compiler warns when a `config` wire is connected to a
cycle-time source. Strict mode promotes that diagnostic to a
compile error. `WireCost` is a `Port` field (`Port::wire_cost`) rather than a
runtime type rule; a data wire and a configuration wire pass
values through the same typed slot ABI.

---

## Node design notes

### Branched dispatch and structured-body assertions

`pick` and `exactly_one_value` back the runtime-feature-detection
pattern, which a host exposes as a workload surface: probe once,
record what was found in booleans, and dispatch on them later.

#### `pick` — semantics

`pick(b0, …, bN-1, v0, …, vN-1)` is a variadic dispatcher over 2N
wires: N selectors and N values. Exactly one selector must be `true`;
the node returns the matching value.

- Evaluates all 2N inputs (no short-circuit; Polydat is data-flow).
- Counts how many of `b0..bN-1` are `true`.
  - Exactly one true → return the corresponding `vi`.
  - Zero true → eval-time error: "pick: no selector matched
    (all N booleans false); workload author guarantees one of
    {b0, …, bN-1} is true at this point."
  - Two or more true → eval-time error: "pick: multiple
    selectors matched (b1, b3, …); selectors must be mutually
    exclusive."
- Errors are reported through the failure contract, so the operator
  sees the function name, the result-wire context, and the
  inputs, on all four engines (the interpreter, the closure tier,
  native, and pure native).

**Argument shape.** Selectors and values are split into two
halves rather than interleaved `(b0,v0,b1,v1,…)`. The split
form scans more cleanly for long lists
(`pick(has_sai, has_idx, has_dse, "tbl_a", "tbl_b", "tbl_c")`),
composes naturally with construction helpers (operators pull
the two lists from separate variables), and catches a missing
pair at compile time as "odd total" — a more direct diagnostic
than the interleaved form's "value missing in last pair".

**Type rules.**

- All `bi` MUST be `Bool` at compile time. A non-bool slot is
  rejected by the parameter validator.
- All `vi` MUST share a common type at compile time
  (`Str`+`Str`, `U64`+`U64`, etc. — no implicit promotion). The
  output type is that common type.
- Mixing types in the value slots is a compile-time error
  pointing at the first mismatched index.

**Variadic registration.** Registers through the macro's
split-halves variadic shape (two `&[T]` arguments,
`variadic_min = 1`), which advertises
`Arity::VariadicWires { min_wires: 2 }`, checks the even total
at assembly, and hands the body the two halves directly:
selectors at `inputs[0..N]`, values at `inputs[N..2N]`.

**Diagnostic guidance** is a static suffix added by the
`pick` node's panic handler — generic enough to fit every
misuse without guessing the workload's structure:

```text
pick: no selector matched (all N=2 booleans false)
  ↳ in node `pick` (output `target_index_table`)
     while evaluating <op-template `indexes_present`>
  ↳ inputs: [Bool(false), Bool(false), Str("system_views.sai_column_indexes"), Str("system_views.indexes")]
  ↳ hint: did the probe phase that sets these booleans run
     before this phase? Check scenario-tree DFS order or
     declare a `detect_*` phase ahead of consumers.
```

#### `exactly_one_value` — motivation and semantics

The motivating use case is a query whose result body is a single
row × single text column carrying schema text. To regex-match
against that text, the workload needs to **unwrap** the structural
body — extract the one-and-only text value — before applying
`regex_match`.

`exactly_one_value(body)` asserts the shape explicitly. The library
does not provide an implicit projection in which a host's
`body.to_text()` would choose unary extraction or JSON
stringification based on the body's shape: a body that silently
changed shape would change the text and the match with it, and the
failure would surface as a wrong result rather than an error.

**Semantics.**

- Walk the body's rows (must be one), columns (must be one),
  cells (must be one).
- If exactly one row × one column × one cell, return the
  cell value.
- Otherwise eval-time error: `"exactly_one_value: expected
  unary structure (1 row × 1 column), found <r> rows × <c>
  columns"`.

The error flows through the failure contract; the operator
sees the function name, the result-wire context, and the
body's actual shape. Matching a regex against such a body takes two
steps: extract the unary value with `exactly_one_value`, then apply
`regex_match` to it. `exactly_one_value` is type-agnostic at the substrate layer:
the body is a structural value wide enough to round-trip through
the JSON representation.

### Host-registered nodes

The node registry is **open**. A host application registers additional nodes via
`inventory` to project its *own* runtime state — live controls, metric series, the
executing phase, the current cycle, and similar — into the DSL as named,
side-effect-free projections. polydat itself provides only the deterministic library;
nodes that read host runtime state live in (and are documented by)
the host, since they depend on host services polydat does not. The engine marks such
nodes nondeterministic so the constant-folder leaves them in place (their value changes per
pull by definition). A read-side projection is a context node; a writable value is a
control — neither is a template / env-var / global.

### Parameter resolution and validation

The predicates (`is_positive`, `in_range`, `matches`, `is_one_of`)
and the resolvers (`this_or`, `required`) are ordinary pass-through
nodes: each returns its input unchanged when the condition holds and
panics with a diagnostic otherwise. They operate on the same wires
everything else does, so a `required(...)` on a workload parameter is
the same mechanism as `required(...)` on an externally written wire
or a runtime control. They stack: several can be applied to the same
value. A predicate is evaluated at the earliest time its input is
known — at compile time for a const-folded value, at scope init for
a parameter, at cycle time for a live read — and a violation is
reported the same way on all four engines. The interpreter and the closure
tier panic in the node's body; native code raises the same failure
through the longjmp path of [JIT Boundary](jit_boundary.md), where the
scalar predicates (`is_positive`, `in_range`, `is_one_of`) have native
lowerings and dedicated fail helpers. Every engine adds the node, its
outputs, and its inputs to the message through the shared failure
contract ([Engines](engines.md)).

<a id="partition-values"></a>
### Partition values

`Partition` and `PartitionList` values are passed on wires as
`Value::Ext` reflected values (`iteration/cursor_partition.rs`).
Workload code reads and derives them with the partition nodes in
`polydat-nodes/src/partition.rs`, which run on all four engines through the slot kit's `Ext<T>`
shape, which native code calls in place. The partition
value is effectively-const for a scope activation, so each eval
reduces to constant arithmetic. The partition grammar and the axioms
behind the nodes are in [Cursor Partitions](cursor_partitions.md);
the nodes are listed in the [node reference](../reference/nodes.md).

---

## Registration

Library nodes are authored via the `#[polydat_node]`
attribute macro and self-register through the `inventory`
crate at link time. The macro is the SOLE authoring path for
workload-callable library nodes; every recognised shape is in the
table below, and a new node uses one of them.

```rust
// Scalar — body returns the output value; macro reads
// `category` + `purity` / `commutativity` per attribute.
#[polydat_node(category = Hashing)]
fn hash(input: u64) -> u64 {
    splitmix64_u64(input)
}

// Const arg — `Const<T>` wraps a workload-supplied literal.
#[polydat_node(category = Arithmetic)]
fn r#mod(input: u64, modulus: Const<u64>) -> u64 {
    input % modulus.0
}
```

The macro reads everything from the function signature. Per-argument
metadata comes from the `Wire` trait's consts (`PORT`, `JIT`,
`RESOLVER`, `WIRE_COST`), so a carrier argument needs no attribute;
marker wrappers and borrow shapes cover every other kind. Cross-crate
registration works identically: a host crate declares
`#[polydat_node]` functions in its own source and its nodes appear in
the Polydat registry at link time.

### Shapes

**Argument kinds** (`crates/polydat-derive/src/lib.rs`, `ArgKind` and
the kit plans):

| Argument | Meaning |
|---|---|
| `T` where `T: Wire` | A carrier: `u64`, `i64`, `f64`, `bool`, the narrow widths, `u128`/`i128`, the `Reg*` register views, `String`/`Arc<str>`, and every other type with a `Wire` impl. `<T as Wire>::PORT` is the port type. |
| `&str`, `&[u8]`, `&serde_json::Value`, `&[f32]` and the other typed slices | Borrow shapes: the body borrows the value for the call. |
| `Arc<[u8]>` / `Vec<u8>`, `Arc<serde_json::Value>`, `SliceArc<T>` / `Vec<T>`, `Handle` | Owned wrapper wires for bytes, JSON, vectors, and opaque handles. |
| `Const<T>` (`u64`, `f64`, `bool`, `&str`) | A workload-supplied literal, fixed at construction. |
| `Const<&[C]>` | A trailing list of literals (`Arity::VariadicConsts`), collected from the tail of the const arguments. The list is built once at construction and the body borrows it, as `Const<&str>` borrows a string literal. Prefer this form. |
| `Const<Vec<C>>` | The same list, handed to the body as an owned clone. Accepted; it allocates once per evaluation for a list that never changes after construction, so write `Const<&[C]>` unless the body genuinely needs to own the elements. |
| `#[poly_const(path, from = arg)] name: &T` | Setup state computed once at construction by `path(arg)` from a const; `from = ()` names a session-static value that is not a function of the consts. |
| `Value` | A polymorphic wire whose port type is resolved at construction; with a `Value` return the output type is `SameAsInput`. |
| `&[T]` (one argument), two `&[T]` arguments | A variadic wire list (`Arity::VariadicWires`), or split halves; element types `u64`, `bool`, `&str`, `String`, `Value`. |
| `Option<T>` | A carrier that may be `None` on the interpreter; a compiled slot never carries `None`, so the closure reads `Some` and the kernel's `None` mask skips the step (see [Compiled By-Reference Slots](compiled_handles.md) §5). |
| `Ext<T>` | A host-defined value carried opaquely as `Value::Ext`. |
| `Config<T>` | A carrier declared a configuration wire (`WIRE_COST = Config`). |
| `Resolved<R, T>` | A value supplied by the default resolver `R` (`FuncSig.default_resolver`, read from `Wire::RESOLVER`) when the workload names none. |

**Return shapes:** a single `T: Wire`; a tuple with
`output_names(...)` naming each element; `Value` (polymorphic);
`DynamicOutputs<T>` (one output port per element, the count fixed at
construction from the node's one `Const<Vec<C>>` argument);
`Result<T, E>` (a fallible body over const arguments
only, run once at construction, whose cached value every evaluation
returns).

**Attributes.** Registration: `category = <FuncCategory>` (required),
`struct_name = <Ident>`. Semantics:
`purity = <Purity>`, `identity = <expr>`, `commutativity =
<Commutativity>`, `variadic_min = <int>`. Shapes: `output_names(...)`.
Engines: `compiled_u64 = <path>`,
`compiled_slot = <path>`, `state = <path>`, `jit_constants = <path>`,
`decompose = <path>`, `simd = "<node>"`, `simd_total`. Validation:
`validate = <path>`, a
`fn(&str, &[ConstArg]) -> Result<(), String>` the factory calls with
the node's constant arguments after every per-parameter constraint has
passed, for a rule that relates two of them — `n_of`'s `n <= m`, or
`in_range`'s `lo <= hi`. Per argument: `#[constraint(...)]` on a wire
argument, `#[poly_default(...)]` on a const argument, and
`#[poly_const(...)]` as above.

A rule over constants belongs in one of those two places rather than
in the body. A body assertion runs only on the engines that run the
body: a native lowering does not run it, so under native code an
assertion such as `n_of`'s `n <= m` never fires and the program builds
and returns a value instead. The factory reads both forms when it
constructs the node, so they hold on all four engines (the interpreter,
the closure tier, native, and pure native) and report the error against
the program that wrote the literal rather than at a later cycle.

### The kit each shape yields

Every node runs on the interpreter through the macro's `eval`. Its
form on the compiled engines is derived from the signature, in this
order:

- **The u64 kit** (`compiled_u64`) covers a node whose every argument
  and return is an immediate: carriers, consts, and a `&[u64]`
  variadic; a tuple return of such elements; no setup argument; not
  fallible.
- **The slot kit** (`compiled_slot(wire_types)`) covers every other
  node whose shape it can read: carriers of one or two slots, typed
  slices, `&str` or owned strings, `&[u8]` or owned byte strings, JSON
  ports, `Ext<T>`, `Value`, variadics (split included), `Option<T>`
  over a carrier, `Config<T>` over a carrier or an owned string or
  byte string, consts, const lists, and setups; a return of a carrier,
  a vector, a string, a byte string, JSON, `Ext`, `Value`, or a tuple
  of these. It declares one scratch entry per by-reference output, and
  its bodies hold the `Ref2` dereferences the S-axioms of [JIT
  Boundary](jit_boundary.md) govern; its contracts with the kernel are
  in [Compiled By-Reference Slots](compiled_handles.md) §5.
- **A fallible body** runs once at construction; the kit its return
  shape names replays the cached value every run.
- **Outside every kit:** `DynamicOutputs<T>` and a node
  that downcasts a `Handle`. Such a node runs on the interpreter only.

**Overrides.** `compiled_u64 = <path>` makes the macro emit
`compiled_u64(&self)` as `Some(<path>(self))`, with `<path>:
fn(&Node) -> CompiledU64Op`; it exists for a closure that needs
setup-derived state the shape kits do not capture (a cycle walk's
half-bits, a mixed radix's table). `compiled_slot = <path>` makes
the macro emit `compiled_slot(&self, wire_types)` as
`Some(<path>(self, wire_types))`, with `<path>: fn(&Node,
&[PortType]) -> CompiledSlotKit`; it exists for a node whose closure
must read its slots as borrowed views rather than as owned body
arguments, or that keeps state of its own in its scratch — the string
constant (`const_str_compiled`), the tile renderer, and
`dynamic_weighted_select` are the users. `state = <path>` names a module with
`layout(&Node) -> Vec<ScratchElem>` and `eval(&Node, &mut
[ScratchBuf], &[Value], &mut [Value])`, the node's per-state storage
and its interpreter evaluation over it (`PolydatNode::scratch_layout`
and `eval_in`); a `ScratchElem::State` entry holds whatever the node
types for itself, filled on first use and empty in a clone, which is
where a memo of the node's last derivation belongs
(`dynamic_weighted_select` is the user). `jit_constants = <path>` supplies the constants a
native lowering bakes, in the order that lowering reads them. An
override wins over eligibility.

### Where a native form lives

A node's closure form is declared by its shape and emitted by the
macro; its native form is not. Native lowerings are matched by node
name in the classifier (`classify_node` and `classify_node_typed` in
`polydat-core/src/compile/jit/codegen.rs`): each arm names a `JitOp`, reads the
node's `jit_constants()` positionally, and, for a node whose body
dispatches on `Value` variants, decides by the types of its wires. A
node with no arm runs its kit from native code through the slot-call
helper ([Compiled By-Reference Slots](compiled_handles.md) §6), a
nondeterministic node or a side channel in a segment of its own on the
P3 kernel. The macro emits `jit_constants` in declaration order
for a node the u64 kit covers; a node whose lowering reads its
constants in another order supplies `jit_constants = <path>`.

The tile hole encoder, `tile_encode`, is a library node covered by the
slot kit. The renderer encodes each hole at its position in the
skeleton ([Polytile](polytile.md) §7), and the compiler emits no
encoder of its own. `tile_encode` is also callable as a node.

### Carve-outs from the canonical path

Hand-written `impl PolydatNode for X` blocks exist only where the
node cannot be expressed as a function of typed arguments. The
hand-written-impl invariant test holds the allowlist of these impls;
any new hand-written impl under `polydat-nodes/src/**` or
`polydat-core/src/library/**`
outside it fails that test. The families outside the attribute, and
why:

- **Compiler-synthesised nodes dispatched on a runtime port type or
  constraint** — `PortPassthrough`, `ConstHandle`, `ConstExt`
  (`polydat-core/src/library/identity.rs`), `AssertType`, `AssertValue`
  (`polydat-core/src/library/assertions.rs`). The macro fixes a node's port types from
  its signature; these take theirs from the value or wire the compiler
  is synthesising for, and are not DSL-callable.
- **Rust-internal composition primitives** — `LutSample`
  (`polydat-nodes/src/sampling/lut.rs`), the primitive behind the `dist_*` family, which
  has no DSL surface of its own; the `dist_*` functions are the
  workload-callable wrappers.

The test reads every line, including lines inside a `macro_rules!`
body, so a macro that generates hand-written impls fails it the same
way a literal impl does. The dataset accessor nodes in `vectors.rs`
are attribute nodes: each declares its dataset facet in its handle
argument's type, which is also where its auto-resolver comes from, so
the facet is stated once rather than in both the signature and the
body.
