---
type: specification
title: The Composition Substrate
timestamp: 2026-09-25
description: The three pillars of free graph composition (context synthesis, type safety, state layering), their S, T, and L axioms, and the boundary handlers between them.
tags: [runtime, scopes, types]
---

# The Composition Substrate

This document specifies the composition substrate: the three
pillars (Context Synthesis, Type Safety, and State Layering)
under which a typed function node runs inside a layered scope
without per-composition negotiation between the node and the
scope. It states the axioms of each pillar (S1–S5, T1–T3,
L1–L2), the slot contract the three pillars jointly guarantee,
and the handlers at the boundaries between them. The mechanisms
are specified in the companion documents; this document states
which axioms each mechanism preserves.

**Related specifications:** [Scope Model](scope_model.md)
(parent-gated materialization),
[Wire Materialization](wire_materialization.md) (the
cross-scope read invariant and write contract),
[Cross-Fiber Cell Invalidation](cross_fiber_invalidation.md)
(the cell protocol), [Subcontext Construction](subcontext_construction.md)
(the host-composed child), [None Semantics](none_semantics.md),
the [Evaluation Model](evaluation_model.md) (program/state
split, two lifecycles), and the [Runtime Model](runtime_model.md)
(R- and D-axioms, and the terms this document uses).

---

## 1. Scope

This document uses the terms of
[runtime_model.md](runtime_model.md) §Terms and the following:

- **Slot.** A declared, named, `PortType`-typed position of a
  kernel: an input slot (an entry of `input_defs`) or a node's
  output port.
- **Node tier.** The typed function nodes, which read their
  input ports and write their output ports.
- **Scope tier.** The layered scope state: a root scope, nested
  child scopes at any depth, and traversal activations, each
  held by its own kernel.
- **Chain.** The parent-child tree of kernels built by
  parent-gated construction, through which an outer scope's
  values are delivered into an inner kernel's input slots.
- **Scope-init.** The construction of a child kernel, during
  which its input slots are bound from the outer scope.
- **Cycle time.** The period after scope-init during which the
  host writes inputs and pulls outputs. "Per cycle" means once
  per coordinate write.
- **Engine.** One of the four ways a program runs: the
  interpreter (P1), the closure tier (P2), the native tier (P3,
  native segments and closure steps in one hybrid kernel), and
  pure native. The last three are the *compiled engines*. A
  statement that holds on fewer than all four names them.

The composition substrate has three pillars:

```text
┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐
│ Context Synthesis   │  │  Type Safety        │  │  State Layering     │
│ (S-axioms)          │  │  (T-axioms)         │  │  (L-axioms)         │
├─────────────────────┤  ├─────────────────────┤  ├─────────────────────┤
│ Host context →      │  │ Every slot has a    │  │ Scope state is      │
│ kernel input slots, │  │ declared PortType.  │  │ layered: root →     │
│ synthesised by the  │  │ Mismatches caught   │  │ nested scopes at    │
│ chain via auto-     │  │ at construction or  │  │ any depth.          │
│ extern + materialize│  │ healed by edge      │  │ Lifecycle bridges   │
│ + typed writes.     │  │ adapters.           │  │ layers.             │
└──────────┬──────────┘  └──────────┬──────────┘  └──────────┬──────────┘
           │                        │                        │
           └────────────────────────┴────────────────────────┘
                                    │
                                    ▼
                       The Slot Contract — §2
                       (the consequence)
                                    │
                                    ▼
                  Free composition at every layer
                  Context Fusion / Node Fusion / parallel safety
                                  (§9)
```

Together S, T, and L produce the **slot contract**, the
interface between the node tier and the scope tier. A node does
not read scope state directly, and a scope does not read or
write a node's internals; slots are the only interface between
the two tiers. A value moves through a slot in one direction at
a time, and the slot's classification fixes that direction.

No single pillar provides the slot contract, and neither does
any pair of them; §6 states what each pair lacks.

---

## 2. The slot contract — the consequence

The slot contract is the substrate's externally visible
product. A kernel exposes:

```text
input_defs:    declared slots — name + PortType + InputKind
inputs:        the slot registers at evaluation time; a cell-bound
               slot's register is its SharedCell
node buffers:  per-step output values
```

On the interpreter the registers are `Value`s. On the three
compiled engines (closure tier, native, and pure native) they
are one flat `u64` slot buffer; each slot is typed by the
static slot colour of its port type and decoded through the
kernel's typed readers. The contract is the same on all four
engines: the host sees named, typed slots.

Each `InputDef` declares one slot's identity (name), its type
(`PortType`), and its origin (`InputKind`: `Coordinate`,
`IterationExtern`, `ExternalWrite`). `ExternalWrite` is the
polydat surface that hosts use for runtime injection; hosts
give those injection patterns their own names.

The slot contract makes three guarantees, one from each pillar:

| Guarantee | Pillar | What it promises |
|---|---|---|
| Slot is **filled** | Context Synthesis (S) | At evaluation time, every declared slot holds a value, synthesised by the chain from scope state according to the slot's `InputKind`. |
| Slot is **typed** | Type Safety (T) | The value matches the slot's declared `PortType`. A mismatch fails at construction or is healed by an auto-inserted adapter. |
| Slot is **layer-aware** | State Layering (L) | The value's lifecycle (effectively-const from scope-init, or dynamic per cycle) is determined by the slot's `InputKind` and its upstream chain. Nodes consume values according to the lifecycle; they do not enforce it. |

**A node consumes inputs through slots.** It reads from a
declared input port, which is typed, named, and layer-aware,
and writes to a declared output port. It does not look up names
in a scope, request values from the kernel chain, or enumerate
parent state.

---

## 3. Pillar 1 — Context Synthesis

The chain synthesises host-provided scope state into the
kernel's declared input slots. Synthesis happens at three
times: at compilation (auto-extern), at scope-init
(binding-time materialisation), and at cycle time (the typed
writes).

### Axiom S1 — Auto-extern as the synthesis surface discovery rule

**At kernel compilation, the compiler walks the body, finds
every identifier reference that resolves to an outer-scope
binding, and synthesises an `extern X: T` slot for it. The
inner kernel's `input_defs` therefore contains a typed slot for
every outer-scope value the body consumes. The set of
discovered externs is the *synthesis surface*: exactly the
layered-state values the chain must deliver at scope-init.**

The Scope Model defines the discovery rule and the compiler
executes it. The workload author does not declare these slots.
For example, the author writes `query[id={k}]`, the compiler
finds that `{k}` refers to the outer iteration variable, and
the slot is created.

### Axiom S2 — Binding-time materialisation as the synthesis fill rule

**At scope-init, parent-gated subcontext construction, which
drives the private binder `materialize_wiring_from_outer`,
iterates the kernel's extern slots and looks up each one's
binding in the outer chain. The binding is classified by the
Wire Materialization gradient as inlined-constant,
value-only-cell, or read-write-shared-cell, and the chain fills
the slot according to that classification. After binding-time
materialisation, every declared slot holds a value.**

The binder is crate-private and is called only from
parent-gated construction ([scope_model.md](scope_model.md)
§2, §4). It is therefore the only path by which a whole
child's slots are bound to an outer scope. A traversal
activation is bound by the same rule in a smaller form: the
tuple and the cascaded wires are typed writes into the body's
declared externs.

### Axiom S3 — The typed writes as the coordinate-time synthesis advance

**The cycle-time slot mutations are the `Kernel` trait's typed
writes, and each one invalidates exactly its own dependents
([runtime_model.md](runtime_model.md) R4). `set_inputs` writes
exactly the slots whose `InputKind` is `Coordinate`;
`set_input` and `set_input_at` write one named extern;
`set_cursor` writes one cursor's `Ext` slot and its six scalar
projections. Every slot not written, including externs from the
outer scope and effectively-const bindings, keeps its
scope-init value or its last written value. Each cycle-time
write is narrow, named, and typed.**

On all four engines (interpreter, closure tier, native, and
pure native), the `Kernel` trait's writes are the only
mutations of a slot during a scope's lifetime. The
two-lifecycle classification (L2) distinguishes coordinate
inputs, which are dynamic and written per coordinate, from
every other input, which is effectively-const for the scope's
lifetime unless a host writes it.

**Coordinate/cell exclusion.** `set_inputs` writes coordinate
values directly into the coordinate prefix. It does not consult
attached cells and is not a synchronisation point for
cell-bound slots. Coordinate slots and shared cells are
therefore mutually exclusive: a cell is attached only to a
`shared` binding's slot, and `attach_shared_cell` refuses any
other slot. Cell-bound state updates are independent of
coordinate writes.

### Axiom S4 — External-write synthesis as the open-granularity fill path

**A polydat kernel exposes port-typed input slots that
external producers may write at any granularity: at scope-init
construction, at cycle time, or at any other time. These slots
are ordinary wires. Nodes read them through the standard port
read; provenance tracking (R2) marks downstream steps not
current when an external write changes a slot; and R1
re-evaluates those steps on their next pull. S4 specifies the
existence and contract of the external-write surface. Cross-tier
writes, in which an inner-scope writer writes an outer scope's
`shared` wire, are governed by S5's SharedCell write-through.**

The typed-write entry points are `Kernel::set_input` and
`set_input_at`, on all four engines (interpreter, closure
tier, native, and pure native). The write rule is that the
value must satisfy the slot's declared `PortType` or be `None`,
which clears the slot to unset; nothing converts a value at the
write. An input whose type may vary is converted by a converter
node that the compiler places in front of its readers, and only
when the host requests it (input_variance.md). The interpreter's
`Dataflow::set_wire` / `set_wire_idx`, which run the boundary
adapter catalog at the write, are deprecated. Two other write
sites consult the catalog until they are retired
(input_variance.md §11): the binder's value copies into a
child's declared inputs, and the write-through commit's
widening of a narrower numeric type. A mismatch that nothing
heals is an error at the write site, naming the slot, its
declared type, and the type given.

All four engines check every extern write: the value must satisfy
the declared type (`satisfies_slot`, bit-stuffed forms
included) or be `None`. T1 and T2 are enforced at this
boundary. The kernel makes no assumption about who the producer
is, when it writes, or what host-level meaning the write has;
the contract is generic population of an external port, and
downstream steps re-evaluate under the standard provenance
rules whenever an input changes.

S1 discovers the surface, S2 fills it at scope-init, and S3
writes the coordinates; all three are fill paths internal to
the chain. S4 is the surface for fills that originate outside
the kernel. The producer may be host code, an iteration driver,
an event source, or any other component unknown to the polydat
layer. The synthesis contract still holds, because the
typed-write API is the chain's entry point for external
producers and not a path around the chain.

**Ordering and visibility.** For external writes the substrate
guarantees:

1. **Per-port atomicity.** A write to one port is atomic with
   respect to concurrent reads of that port. Non-cell slots
   are protected by the writing kernel's `&mut` exclusion;
   cell-bound slots by the cell's `Mutex`.
2. **Same-fiber program order.** All writes issued on a fiber
   are observed in program order by later reads on the same
   fiber.
3. **Local invalidation on write.** A typed write marks not
   current every step whose provenance includes the written
   slot, on the writing kernel. The next pull on that kernel
   observes the new value.
4. **No cross-port ordering across cells.** Writing port A
   and then port B does not guarantee that a sibling kernel
   observes A before B. Each cell is an independent
   consistency domain; a producer that needs multi-port
   atomicity writes through a single port or coordinates the
   writes at a higher layer.

Publication and invalidation through a cell, the cross-kernel
part of the contract, is specified by S5.

**Volatility opt-in (cross-reference to R1).** A wire whose
value is not a function of its declared inputs (a temporal
source, an entropy source, or any other nondeterministic
producer) is *volatile*. Volatility has two sources:

- **Intrinsic.** Certain library nodes declare themselves
  volatile (e.g., `current_epoch_millis`, `counter`,
  `elapsed_millis`, `thread_id`). The library imposes the
  volatility; no user opt-in is required, and the workload
  author cannot remove the marker.
- **User opt-in.** A wire is declared volatile with the
  `volatile` modifier on its binding. The author thereby marks
  the wire as recomputed after every write.

Both sources have the same runtime effect: the wire is not
memoized across writes, and volatility spreads to its
dependents (see R1's volatile sub-axiom in runtime_model.md).
Ordinary S4 external writes do not need volatility; provenance
tracking re-evaluates the affected steps when an input changes.
Volatility is the explicit marker for the nondeterministic
case, where input-change tracking is insufficient because the
value is not a function of the declared inputs.

### Axiom S5 — Compile-emit write-through as the cross-tier synthesis path

**`SharedCell` write-through is the only mechanism by which an
inner-tier node's output can change outer-tier state. The
rewrite happens at compilation: a node that declares a write to
a wire held by an outer scope's shared cell has its local
output rewritten to a write-through call. The node itself is
unchanged and still produces a value on its declared output
port. The chain intercepts that output and routes it to the
outer cell, and the outer cell's slot is filled with the value
through the standard slot-filling contract.**

S5 is an S-axiom because a cross-tier write is a slot-filling
operation: the inner-tier node's output value fills the
outer-tier cell's slot. The compiler emits the write-through
call, and at runtime the outer cell's slot is updated through
that routing. The layer-ownership guarantee, that the outer
cell remains the canonical holder of the state and no two
layers contend for write authority, is L1's. S5 preserves it
because write-through goes through the typed-slot surface, so
the outer cell's scope sees an ordinary slot write rather than
a cross-tier mutation.

The Wire Materialization classification decides which wires
are cells, and the write-through rewrite in
[subcontext_construction.md](subcontext_construction.md) §3.1
emits the call. The cell is the only register for its wire,
and write-through routing is the only cross-tier write surface.

**Cross-fiber validity tracking.** A producer's write to a
`SharedCell` is observed by every consumer kernel on its next
read, with no host-layer action. A write to a cell-bound port
publishes the value, increments the cell's revision, and sets
the cell's bit on the intent word of the scope that created the
cell. Every consumer re-reads the cell when a revision it has
seen changes, and marks every step that reads the cell's slot
not current. Modification order is defined per cell; no total
order across distinct cells is defined. The mechanism and both
consumer realisations (the interpreter's revision check at
every memoized read, and the per-cell poll of the three
compiled engines) are specified in
[cross_fiber_invalidation.md](cross_fiber_invalidation.md),
the sole validity-tracking specification for S5's cross-tier
writes. On all four engines (interpreter, closure tier,
native, and pure native), readers never make host-side refresh
calls.

The following diagram shows one cell write, from the producer
kernel's write to a consumer kernel on another fiber reading
the new value.

![Sequence of a SharedCell write: the producer writes under the cell's mutex, the cell's revision is incremented and its intent bit set, the consumer's next read sees the revision move, the consumer marks steps reading the cell's slot not current, and its pull re-reads the published value](../diagrams/composition_substrate-cell-write.png)

**Sub-axiom S5.r — `shared` grants write permission only.**
The `shared` modifier on a wire declares that the wire is
writable across tier boundaries through write-through. It does
not affect read access: an inner-scope read of any visible
cross-tier wire returns the current outer-tier value under the
uniform read invariant (L1), whether or not the wire is marked
`shared`. Reads are wired by the chain at construction; the
`shared` declaration only lets the inner tier write to the
wire, by enabling the write-through path. Uniform reads are
guaranteed by the substrate; write permission is an explicit
opt-in.

---

## 4. Pillar 2 — Type Safety

Every slot in the kernel, whether input, output, or externally
written, has a declared `PortType`. The chain guarantees a
value of that type at every read, and the compiler rejects
mismatches at construction or heals them with auto-inserted
edge adapters. The type contract is enforced at construction
and at write boundaries.

### Axiom T1 — Every slot is typed

**Every entry in `input_defs` has a declared `PortType`
(U64, F64, Bool, Str, VecF32, VecI32, Bytes, Json, …). Every
node output port has a declared `PortType`. No slot, input or
output, is untyped. There is no "any" type at the slot tier.**

The `InputDef` and `Port` types hold the declaration, and the
[Type System](type_system.md) defines the types. On the three
compiled engines the flat `u64` buffer does not weaken this
axiom, because every slot's colour and decoding are fixed by
its port type at build.

### Axiom T2 — Type mismatches are construction-time or auto-healed

The adapter catalog is used at these sites: intra-graph wire
validation (assembly's wire resolution, which inserts edge
adapters from the conversion library); the converter nodes in
front of an input whose type may vary; and
`polydat::convert::to_port`, the host-side form of the same
conversion (input_variance.md). Until they are retired, it is
also used by the binder's value copies (`adapt_boundary_value`)
and by the write-through commit's type-stability check on a
shared cell. The deprecated `set_wire_idx` also uses the
boundary copy. The catalog is the single source of truth for
all of these sites.

**The assembly pass validates every wire's source `PortType`
against its consumer's expected type. A direct mismatch fails
construction with a typed `AssemblyError::TypeMismatch`. A
mismatch with a known auto-conversion (e.g., U64 → Str via
`U64ToString`, F64 → Str via `F64ToString`, U64 → F64 via
`U64ToF64`) is healed by inserting the adapter node in line.
After assembly, every wire's source type matches its
consumer's expected type, either directly or through a chain of
catalog adapters. Nodes never receive a value of the wrong
type.**

The catalog of known conversions is finite and explicit; a new
conversion requires a new catalog entry. The substrate does not
coerce silently.

### Axiom T3 — Compiled engines preserve the slot type contract

**A compiled step receives its inputs through the same typed
slots. Native code neither bypasses type checks nor coerces
silently: it reads slot positions whose colour and type the
compiler validated against the producing node's declared
`PortType`, and a segment's native signature is determined by
its declared input and output types. On the native tier, a node
that has no native lowering for the selected ISA, or whose
declared purity keeps it out of a native segment, runs as a
closure step in the same kernel, over the same slots; the
closure form is derived from the node's signature and exists
for every node. Nothing falls back to interpretation.**

Pure native has no closure steps: every step is a fusion unit,
and a program with a node it cannot lower is refused at
construction.

Native eligibility is determined by construction and reported,
not inferred: a node runs natively when it has a lowering for
the effective ISA and its lifecycle and purity admit it to a
segment. `Kernel::plan` reports how much of a program runs as
native segments, as closure steps, and on the interpreter, so a
host can see what the engine chose. The native boundary is
specified in [jit_boundary.md](jit_boundary.md) and the engine
lattice in [engines.md](engines.md).

---

## 5. Pillar 3 — State Layering

Scope state is layered, and each layer holds its own state. The
chain composes the layers, and the lifecycle classification
connects them. No state passes between layers except through
declared slots, filled according to their lifecycle.

### The layer taxonomy

```text
Outer scope ──────────────── chain-cascaded params, top-level externs
   ▼
Nested scopes (any depth) ── per-scope bindings, shared cells, externs
   ▼
Cycle-time ───────────────── per-pull state + external-write injection
```

Hosts give their own names to the layers of this generic
structure; polydat itself does not name them.

Inner layers read outer-layer state through auto-extern and
binding-time materialisation (S1 and S2). Outer layers do not
read inner-layer state; there is no callback up the chain.
Cross-tier writes are limited to SharedCell write-through (S5).

### Axiom L1 — Each layer owns its own state

**A scope instance (a root, a nested scope at any depth, or a
traversal activation) holds its own set of state. State written
at one layer is not visible at outer layers. State read at one
layer comes from that layer's own bindings, or from outer
layers through slots synthesised by the chain. There is no
cross-tier shared mutable state other than the SharedCell
write-through mechanism specified by S5 (compile-emit
write-through as the cross-tier synthesis path).**

The kernel chain is a parent-child tree built by parent-gated
construction ([scope_model.md](scope_model.md) §2).
Binding-time materialisation is the only surface through which
state passes between layers, and the Wire Materialization
classification governs read and write semantics. S5 specifies
the only cross-tier write surface. L1 requires that the outer
cell remain the canonical holder of the state regardless of
which inner tier issues the write, so layer ownership is
preserved under S5's routing.

### Axiom L2 — Two-lifecycle classification bridges layers

**Every input slot has one of two lifecycles: *effectively-
const* (resolved once at scope-init and fixed for the scope's
lifetime) or *dynamic* (resolved on pull at cycle time). The
lifecycle is structural: it is determined by the slot's
`InputKind` and its upstream wire chain, not by a runtime flag.
This classification connects the layers: the chain fills an
effectively-const slot at scope-init from an outer layer's
binding, and fills a dynamic slot per coordinate from the
current layer's writes.**

The classification belongs to the program and is computed by
one classifier shared by all four engines
([runtime_model.md](runtime_model.md) §3). The const-binding
contract of the [Evaluation Model](evaluation_model.md) checks
it twice: by a compile-time wire-chain check (Plan A) and by the
scope-init pull (Plan B). The classification is known before
any node receives a value, and the chain enforces it by filling
slots according to each input's lifecycle.

**Sub-axiom L2.f — Failed const materialisation falls
through to the outer chain (L2 ⊓ T1).** When an
effectively-const binding's scope-init evaluation yields
`Value::None` (under the None propagation contract of
[none_semantics.md](none_semantics.md)), the slot is
*unfilled at this layer*. The read invariant (L1) then returns
the outer scope's value for the same name through the standard
lookup chain. The effectively-const guarantee still holds at
the outer layer: the outer binding is itself effectively-const
for the scope's lifetime, so the value the inner reader
observes is stable across the activation. This combination of
L2 (the inner binding's lifecycle) with T1 (typed slots include
the None sentinel) gives the conditional-shadow semantics that
a host's `set:`-style sugar relies on: an intermediate-layer
`const X := <expr>` that yields a real value shadows the outer
X, and one that yields None leaves the outer X visible.

**Strict mode.** Silent fall-through on an intermediate-layer
None can hide an authoring error: the layer may have meant to
provide a shadow that happens to compute to None, or it may
have meant to declare `extern X` and omitted it. Under the
`strict` flag of `subcontext::CompileOptions`, `build_subscope`
calls `PolydatKernel::find_l2f_violations` after scope-init and
raises any const output materialised to `Value::None` as
`ContractViolation::StrictNonePropagation`. The diagnostic
names each offending binding and tells the author either to
make the binding yield a defined value, or to remove the
binding and declare `extern <name>` explicitly if fall-through
to the outer value was intended. The substrate provides the
mechanism; the caller chooses which builds use strict mode
through the flag.

### Note on cross-tier writes

The cross-tier write mechanism (`SharedCell` write-through) is
a synthesis concern and is specified by S5 in §3: it fills
outer-tier slots from inner-tier nodes' typed outputs, so it is
grouped with the other slot-filling axioms. The layer-ownership
guarantee that S5 preserves (the outer cell remains the
canonical holder of the state, and inner-tier writes go through
the typed-slot surface rather than as ad-hoc cross-tier
mutation) is L1's; see §"Axiom L1 — Each layer owns its own
state" above, which names SharedCell write-through as the only
exception to single-layer state ownership.

---

## 6. The three pillars compose — the substrate as product

Each of S, T, and L provides one guarantee, and the slot
contract (§2) requires all three. The three together are
sufficient for the slot contract, and no two of them are.

| If you have | But lack | The slot contract... |
|---|---|---|
| T + L | S | ...cannot be filled. Slots are typed and layer-aware, but the chain has no synthesis mechanism, so slots stay empty or take ad-hoc values. |
| S + L | T | ...is fillable but unsafe. The chain delivers values, but nothing guarantees their type, so nodes must check types defensively or coerce silently. |
| S + T | L | ...is fillable and typed but flat. Without layering there are no nested scope tiers and no lifecycle distinction. S5's cross-tier write mechanism would still route values, but without L1's layer-ownership guarantee a written cell would have no stable owner. |
| **S + T + L** | — | **...holds.** Slots are filled, typed, and layer-aware, and free composition follows. |

The substrate is exactly S + T + L. A further pillar (for
external-write timing, determinism, or cross-tier write
semantics, for example) could strengthen specific guarantees,
but the slot contract holds under S + T + L alone.

---

## 7. The slot contract enables free graph composition

The substrate has this consequence:

**A pure typed function node composes inside a layered
stateful scope without either tier depending on the other,
because the chain has synthesised the scope's relevant state
into the node's typed input slots. The node consumes typed
slots, the scope holds layered state, and the chain connects
them. Neither tier does per-composition wiring.**

Polydat graph composition therefore has no per-composition
negotiation cost. This is a property derived from S + T + L,
not a separate primitive claim.

---

## 8. Boundary mechanisms — named handlers

Certain timing and identity boundaries need named handlers.
Each handler preserves the S, T, and L axioms.

### 8.1 The external-write boundary

Under S4, external producers write port-typed slots through
the kernel's typed writes. The producer decides when to write,
not the kernel. The kernel guarantees that any write causes
standard provenance invalidation (R2) and that affected steps
re-evaluate on their next pull (R1). Reads of the slot are
ordinary port reads, and T1 and T2 ensure type safety at the
write boundary.

A host with specific timing requirements (a runtime injecting
values at defined synchronisation points, a producer driving
values from an event stream, or any other host-level
write-timing semantics) defines that contract at the host
layer. The polydat kernel guarantees only that writes cause
normal invalidation, so polydat embeds no host's write-timing
semantics and can serve diverse host runtimes.

### 8.2 SharedCell write-through (§3, S5)

A node's output value is written to an outer tier through
write-through routing. The node is unchanged, and the chain
performs the routing. S5 makes this the only mechanism; T2
ensures the type check holds across the cell; S2's synthesis
at the outer scope's next read takes the updated cell value;
and L1's layer-ownership invariant holds because the outer cell
remains the canonical holder of the state.

Cross-kernel visibility of the written value is specified by
[cross_fiber_invalidation.md](cross_fiber_invalidation.md).
The producer's mutex-protected write is accompanied by a
revision increment and an intent-bit set on the cell's defining
scope. Every consumer observes the change on its next read,
the interpreter through its cone check and a compiled kernel
through its poll. No host-side refresh call is required.

### 8.3 Const lifecycle violations

When a `const X := <expr>` binding's right-hand side depends on
a dynamic input, it violates L2's structural classification.
The Evaluation Model's const-binding contract detects this:
Plan A (compile-time wire-chain analysis) catches structural
violations, and Plan B (the scope-init pull under
`catch_unwind`) catches semantic violations. The node tier
never sees a violation; a node receives either a value from
the chain or an error from the construction layer.

### 8.4 Native and closure steps (T3)

To the substrate, compiled steps are ordinary slot consumers:
they have declared inputs, declared outputs, and typed
`PortType`s, and they read a slot buffer. When a node has no
native lowering, or its purity or lifecycle keeps it out of a
segment, the hybrid kernel runs it as a closure step and
compiles the rest into native segments ([engines.md](engines.md)
states what a segment may contain). T3 makes the mix sound,
because both kinds of step read and write the same typed slots.

### 8.5 Diagnostic and side-channel node observable side effects

Some diagnostic nodes (`log_info`, `log_debug`, etc.) write to
stderr or to a log buffer during `eval`. To the substrate, the
node's returned value is still a function of its inputs (T1 and
T2 hold); the side effect is observable but untyped, because it
does not pass through a slot. `Purity::SideChannel { sink }`
declares such a side effect; `Purity::Nondeterministic`
declares return values that vary with external state or
history. Neither kind of node is merged into a native segment
with other nodes (on pure native, a side-channel node is a
fusion unit of its own), so each runs under the same currency
rule on all four engines. Typed output
ports remain governed by T1 and T2, and ordering and
cross-fiber observability are governed by
[runtime_model.md](runtime_model.md) D2. No implicit
event-output port is synthesised.

---

## 9. Rationale

The substrate is required by three polydat capabilities:

- **Context Fusion**, the scope-init synthesis that fills the
  graph's declared slots from outer context, is S1 and S2 at
  scope-init; T1 and T2 guarantee the values are typed, and L1
  and L2 supply their lifecycle.
- **Node Fusion**, the compile-time rewriting that inserts
  adapters at wire resolution and fuses subgraphs into
  segments, is sound when each rewrite preserves the slot
  contract: the same input slots, output slots, typed values,
  and lifecycle classification. T1 and T2 give the rewriter a
  typed graph, so fusion correctness is a closure property over
  T2 and L2.
- **Parallel evaluation.** Polydat kernels run in concurrent
  fibers, one kernel per fiber, each created from the shared
  program (`KernelProgram::create_kernel`). L1 together with one
  kernel per fiber means there is no shared mutable state at
  the node tier across fibers; the only shared registers are
  cells, attached by an explicit act. Parallel safety therefore
  follows directly from L1 and T1 and needs no separate
  concurrency proof.

---

## 10. What this document does NOT specify

- **The grammar productions.** [polydat_grammar.md §21](polydat_grammar.md#sec-productions)
  specifies the formal productions; this document requires the
  grammar to expose typed input ports.
- **The compilation pipeline.**
  [graph_compiler.md](graph_compiler.md) specifies the graph
  compiler, kernel hoisting, and the fusion passes. This
  document requires the compiler to enforce S1 (auto-extern),
  T1 and T2 (typed slot construction), and L2 (lifecycle
  classification).
- **The expression system as host utility.**
  [expression_engine.md](expression_engine.md) specifies the
  host-facing expression engine. This document requires
  expression evaluation to be a special case of node
  evaluation, under the same slot contract and chain.
- **The full kernel-composition algebra.** The
  [Scope Model](scope_model.md), [Wire Materialization](wire_materialization.md),
  and [Subcontext Construction](subcontext_construction.md)
  specify the mechanics; this document names the substrate they
  form together and does not re-derive their machinery.
