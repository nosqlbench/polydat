# The Composition Substrate — Polydat Design

**Subtitle:** Context Synthesis, Type Safety, State Layering.

Formalises the substrate that makes polydat's free graph
composition work. Names the three pillars, the axioms under
each, and the boundary handlers that connect them. The
mechanism documents describe individual mechanisms — the
[Scope Model](scope_model.md) (parent-gated materialization),
[Wire Materialization](wire_materialization.md) (the
cross-scope read invariant and write contract),
[Cross-Fiber Cell Invalidation](cross_fiber_invalidation.md)
(the cell protocol), [Subcontext Construction](subcontext_construction.md)
(the host-composed child), [None Semantics](none_semantics.md),
the [Evaluation Model](evaluation_model.md) (program/state
split, two lifecycles), and the [Runtime Model](runtime_model.md)
(R- and D-axioms); this doc names the substrate they
collectively form and states the axioms each preserves.

The forcing question: **given that polydat is a graph
compiler producing kernels that run in concurrent fibers and
host user-typed function nodes inside layered scopes — what
substrate makes free composition possible at every layer
without per-composition negotiation between the node tier and
the scope tier?** This doc says: three pillars composed —
Context Synthesis, Type Safety, State Layering. The pillars
are not independent; they reinforce each other. The
*substrate* is the three pillars in composition, not any one
in isolation.

---

## 1. The claim

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

S, T, and L compose into the **slot contract**, the abstraction
barrier between the node tier and the scope tier. Below the
barrier: typed-port function nodes consuming inputs and
producing outputs. Above the barrier: layered scope state
synthesised into typed slots by the chain. The node never
reaches up; the scope never reaches down; the slot is the only
crossing, and it crosses *one direction at a time per
classification*.

The substrate is **not** any single pillar. S alone gives you
context delivery without type guarantees. T alone gives you
type checking without layered composition. L alone gives you
layered state without a chain to deliver it. The three
together — and only the three together — make the slot
contract durable across layers, types, and scopes.

---

## 2. The slot contract — the consequence

The substrate's externally-visible product is the **slot
contract**. A kernel exposes:

```text
input_defs:    declared slots — name + PortType + InputKind
inputs:        the slot registers at evaluation time; a cell-bound
               slot's register is its SharedCell
node buffers:  per-step output values
```

On the interpreter the registers are `Value`s; on the
compiled engines they are one flat `u64` slot buffer, each
slot typed by the static slot colour of its port type and
decoded through the kernel's typed readers. The contract is
the same: the host sees named, typed slots on every engine.

Each `InputDef` declares one slot's identity (name), its type
(PortType), and its origin (InputKind: Coordinate,
IterationExtern, ExternalWrite). The `ExternalWrite` kind is the
polydat-side surface that hosts use for runtime injection
patterns; hosts give those patterns their own names.

The slot contract has three guarantees, one from each pillar:

| Guarantee | Pillar | What it promises |
|---|---|---|
| Slot is **filled** | Context Synthesis (S) | At evaluation time, every declared slot holds a value. The chain has synthesised it from scope state per the slot's `InputKind`. |
| Slot is **typed** | Type Safety (T) | The value matches the slot's declared `PortType`. Mismatches were caught at construction or healed by an auto-inserted adapter. |
| Slot is **layered-aware** | State Layering (L) | The value's lifecycle (effectively-const at scope-init vs dynamic per cycle) is determined by the slot's `InputKind` and upstream chain. Nodes consume per the lifecycle; they do not enforce it. |

**A node consumes inputs through slots.** It reads from a
declared input port — typed, named, layer-aware — and writes
to a declared output port. It does not look up names in a
scope, does not request values from the kernel chain, does
not enumerate parent state. The slot is the abstraction
barrier; the substrate is what makes the barrier work.

---

## 3. Pillar 1 — Context Synthesis

The chain *synthesises* host-provided scope state into the
kernel's declared input slots. This is an active construction
process at three timings: compile (auto-extern), scope-init
(binding-time materialisation), and coordinate time (the
typed writes).

### Axiom S1 — Auto-extern as the synthesis surface discovery rule

**At kernel compilation, the compiler walks the body, finds
every identifier reference that resolves to an outer-scope
binding, and synthesises an `extern X: T` slot for it. The
inner kernel's `input_defs` accordingly contains a typed slot
for every outer-scope value the body consumes. The set of
discovered externs is the *synthesis surface* — the precise
set of layered-state values the chain must deliver at
scope-init time.**

The Scope Model defines the discovery rule; the compiler
executes it. The workload author does not declare these slots
manually; the compiler discovers them. This is what makes the
substrate *free for the author*: they write `query[id={k}]`,
the compiler discovers `{k}` references the outer iter-var,
and the slot appears.

### Axiom S2 — Binding-time materialisation as the synthesis fill rule

**At scope-init time, parent-gated subcontext construction
(driving the private binder `materialize_wiring_from_outer`)
iterates the kernel's extern slots and for each looks up the
corresponding binding in the outer chain. Per the Wire
Materialization gradient, the binding is classified as
inlined-constant, value-only-cell, or read-write-shared-cell;
the chain fills the slot per classification. After
binding-time materialisation, every declared slot holds a
value.**

The binder is crate-private and reached only through
parent-gated construction ([scope_model.md](scope_model.md)
§2, §4), so this is the *only* path by which a whole child's
slots are bound to an outer scope — there is no second channel.
A traversal activation is bound by the same rule in its
smaller form: the tuple and the cascaded wires are typed writes
into the body's declared externs.

### Axiom S3 — The typed writes as the coordinate-time synthesis advance

**The coordinate-time slot mutations are the `Kernel` trait's typed
writes, and each of them invalidates exactly its own dependents
([runtime_model.md](runtime_model.md) R4): `set_inputs`
mutates exactly the slots whose `InputKind` is `Coordinate`;
`set_input` and `set_input_at` mutate one named extern;
`set_cursor` mutates one cursor's `Ext` slot and its six
scalar projections. Every slot not written — externs from the
outer scope, effectively-const bindings — retains its
scope-init or last-write value. Coordinate-time advance is narrow,
named, and typed.**

The `Kernel` trait is the whole write surface, on every
engine; nothing else mutates a slot during a scope's lifetime.
The two-lifecycle classification (L2) is what distinguishes
coordinate inputs (dynamic, written per coordinate) from every other input
(effectively-const for the scope's lifetime unless a host
writes it).

**Coordinate/cell exclusion.** `set_inputs` writes coordinate
values directly into the coordinate prefix and does not
consult attached cells, and it is not a synchronisation point
for cell-bound slots. Coordinate slots and shared cells are
therefore mutually exclusive: a cell is attached only to a
`shared` binding's slot, and `attach_shared_cell` refuses any
other. Cell-bound state updates are independent of coordinate
advance.

### Axiom S4 — External-write synthesis as the open-granularity fill path

**A polydat kernel exposes port-typed input slots that
external producers may populate at any granularity —
construction-time scope-init, cycle-time injection, or
arbitrary external-write-time. These slots are ordinary
wires: nodes reading from them use the standard port-read
mechanism; provenance tracking (per R2) marks downstream
consumers not current when an external write changes a slot;
currency (per R1) re-evaluates them on next pull. S4 names
the existence and contract of the external-write surface
itself; cross-tier write semantics — an inner-scope writer
populating an outer-scope's `shared` wire — are governed by
S5's SharedCell write-through mechanism.**

The typed-write entry points are `Kernel::set_input` and
`set_input_at`, on every engine. The write rule: the value
must satisfy the slot's declared `PortType`, or be `None`,
which clears the slot to unset; nothing converts a value at
the write. An input whose type may vary is converted by a
converter node the compiler places in front of its readers,
and only when the host asks (input_variance.md). The
interpreter's `Dataflow::set_wire` / `set_wire_idx`, which ran
the boundary adapter catalog at the write, are deprecated.
Two write sites still consult the catalog until they are
retired (input_variance.md §11): the binder's value copies
into a child's declared inputs, and the write-through commit's
widening of a narrower numeric type. A mismatch nothing heals
is an error at the write site, naming the slot, its declared
type, and the type given.
Every engine checks every extern write: the value satisfies the
declared type (`satisfies_slot`, bit-stuffed forms included) or
is `None`. T1 + T2 are enforced at this boundary; the
kernel makes no assumption about who the producer is, when it
writes, or what host-level semantic the write carries — the
contract is generic external-port population, and downstream
nodes re-evaluate per the standard provenance rules whenever an
input changes.

Where the other S-axioms describe chain-internal fill
paths (S1 discovers the surface; S2 fills at scope-init;
S3 advances the coordinates), S4 names the open-granularity
surface for fills *originated outside the kernel*. The
producer is host code, an iteration driver, an event
source, or any other component the polydat layer does not
need to know about. The synthesis contract is preserved
because the typed-write API is the chain's entry point for
external producers, not a side channel around the chain.

**Ordering and visibility.** The substrate guarantees, for
external writes:

1. **Per-port atomicity.** A write to one port is atomic with
   respect to concurrent reads of that port. Non-cell slots
   are protected by the writing kernel's `&mut` exclusion;
   cell-bound slots by the cell's `Mutex`.
2. **Same-fiber program order.** All writes issued on a fiber
   are observed in program order by subsequent reads on the
   same fiber.
3. **Local invalidation on write.** A typed write marks not
   current every step whose provenance includes the written
   slot, on the writing kernel. Its next pull observes the
   new value.
4. **No cross-port ordering across cells.** Writing port A
   then port B on the producer does not guarantee that a
   sibling observes A before B. Each cell is an independent
   consistency domain; producers requiring multi-port
   atomicity batch through a single port or coordinate at a
   higher layer.

The cross-kernel half of the contract, publication and
invalidation through a cell, is S5's.

**Volatility opt-in (cross-reference to R1).** When a
wire's value is not a function of its declared inputs —
temporal sources, entropy sources, or any genuinely
non-deterministic producer — the wire is *volatile*.
Volatility arises from two distinct sources:

- **Intrinsic.** Certain library nodes declare themselves
  volatile (e.g., `current_epoch_millis`, `counter`,
  `elapsed_millis`, `thread_id`). The library imposes
  volatility; no user opt-in is required, and the marker
  cannot be removed by the workload author.
- **User opt-in.** A wire is declared `volatile` via the
  modifier on its binding. The author marks the wire as
  must-recompute-after-every-write.

Both sources produce the same runtime effect: opt-out of
memoization across writes, and contagion through dependents
(see R1's volatile sub-axiom in runtime_model.md).
Volatility is not required for ordinary S4 external
writes; the provenance machinery handles re-evaluation
correctly via input-change tracking. Volatility is the
explicit marker for the genuinely-non-deterministic case
where input-change tracking is insufficient because the
value is not a function of the declared inputs.

### Axiom S5 — Compile-emit write-through as the cross-tier synthesis path

**The `SharedCell` write-through is the *only* mechanism by
which an inner-tier node's output can mutate outer-tier
state. The rewrite happens at the *compilation* layer: a node
that declares a write to a wire owned by an outer scope's
shared cell has its local output rewritten to a write-through
call. The node itself is unchanged — it still produces a
value to its declared output port. The chain intercepts the
output and routes it to the outer cell; the outer cell's slot
gets filled with the inner-tier write through the standard
slot-filling contract.**

S5 sits on the S-axis because cross-tier writes are
fundamentally a slot-filling operation: the inner-tier
node's output value fills the outer-tier cell's slot. The
mechanism is compile-emit synthesis (the compiler emits the
write-through call); the runtime effect is that the outer
cell's slot has its value updated through the routing. The
layer-ownership guarantee — that the outer cell remains the
canonical state holder and no two layers race for write
authority — is L1's claim, not S5's. S5 names how the
mechanism preserves L1's invariant: write-through routes
through the typed-slot surface, so the outer cell's owner
sees an ordinary slot-write rather than a cross-tier
mutation.

The Wire Materialization classification decides which wires
are cells; the write-through rewrite in
[subcontext_construction.md](subcontext_construction.md) §3.1
emits the call; the cell is the one register for the wire,
and write-through routing is the only cross-tier write
surface.

**Cross-fiber validity tracking** — the substrate
guarantees that a producer's write to a `SharedCell` is
observed by *every* consumer kernel on its next read,
without any host-layer ceremony. A write to a cell-bound
port publishes the value, bumps the cell's revision, and
sets the cell's bit on the intent word of the scope that
created the cell; every consumer re-reads the cell when a
revision it has seen moves, and marks every step over the
cell's slot not current. Per-cell modification order is
defined; no total order across distinct cells is. The
mechanism and both consumer realizations (the interpreter's
revision check at every memoized read, the compiled
kernels' per-cell poll) are specified in
[cross_fiber_invalidation.md](cross_fiber_invalidation.md),
the sole validity-tracking spec for S5's cross-tier writes.
The reader contract (no host-side refresh calls) is
preserved by construction, on every engine.

**Sub-axiom S5.r — `shared` carries write permission only.**
The `shared` modifier on a wire declares that the wire is
*writable across tier boundaries* via the write-through
mechanism described above. It does NOT affect read access:
inner-scope reads of any visible cross-tier wire return the
current outer-tier value via the uniform read invariant
(per L1), whether or not the wire is marked `shared`. Read
mediation is governed by the chain's wiring synthesis at
construction time; the `shared` declaration only enables
the inner tier to *write* to the wire, opening the
write-through path. Read uniformity is the substrate's
guarantee; write permission is the explicit opt-in.

---

## 4. Pillar 2 — Type Safety

Every slot in the kernel — input, output, externally-written
— has a declared `PortType`. The chain guarantees a value of that
type at every read; the compiler catches mismatches at
construction or heals them with auto-inserted edge adapters.
The type contract is enforced at construction and write boundaries.

### Axiom T1 — Every slot is typed

**Every entry in `input_defs` carries a declared `PortType`
(U64, F64, Bool, Str, VecF32, VecI32, Bytes, Json, …). Every
node output port carries a declared `PortType`. No slot,
input or output, is untyped. There is no "any" type at the
slot tier.**

The `InputDef` and `Port` types carry the declaration; the
[Type System](type_system.md) owns the definitions. This
axiom is the substrate's claim that nothing escapes the
typing: on the compiled engines the flat `u64` buffer does
not weaken it, since every slot's colour and decoding are
fixed by its port type at build.

### Axiom T2 — Type mismatches are construction-time or auto-healed

The adapter catalog operates at these sites: intra-graph wire
validation (assembly's wire resolution, inserting edge
adapters from the conversion library), the converter nodes in
front of an input whose type may vary, and
`polydat::convert::to_port`, the host-side form of the same
conversion (input_variance.md); and, until they are retired,
the binder's value copies (`adapt_boundary_value`) and the
write-through commit's type-stability check on a shared cell.
The deprecated `set_wire_idx` still uses the boundary copy. The
catalog is the single source of truth across all of them.

**The assembly pass validates every wire's source `PortType`
against its consumer's expectation. A direct mismatch fails
construction with a typed `AssemblyError::TypeMismatch`. A
mismatch with a known auto-conversion edge (e.g., U64 → Str
via `U64ToString`, F64 → Str via `F64ToString`, U64 → F64 via
`U64ToF64`) is healed by inserting the adapter node in line.
After assembly, every wire's source type matches its
consumer's expectation, either directly or via a justified
adapter chain. Nodes never see a value of the wrong type.**

The catalog of known conversions is finite and explicit;
novel conversions require adding to the catalog. The
substrate does not silently coerce.

### Axiom T3 — Compiled engines preserve the slot type contract

**A compiled step receives its inputs through the same typed
slots. Native code does not bypass type checks and does not
coerce silently: it reads slot positions whose colour and
type the compiler validated against the producing node's
declared `PortType`, and the native signature of a segment is
a function of its declared input and output types. A node
that has no native lowering for the selected ISA, or whose
declared purity keeps it out of a native segment, runs as a
closure step in the same kernel, over the same slots; the
closure form is derived from the node's signature and exists
for every node. Nothing falls back to interpretation.**

Eligibility is constructive and observed, not inferred: a
node runs natively when it has a lowering for the effective
ISA and its lifecycle and purity admit it to a segment;
`Kernel::plan` reports how much of a program runs as native
segments, as closure steps, and on the interpreter, so a host
can see what the engine decided. The native boundary itself
is specified in [jit_boundary.md](jit_boundary.md); the
engine lattice in [engines.md](engines.md).

---

## 5. Pillar 3 — State Layering

Scope state is *layered*. Each layer owns its own state. The
chain composes layers; the lifecycle classification bridges
them. Nothing crosses between layers except through declared
slots populated per lifecycle.

### The layer taxonomy

```text
Outer scope ──────────────── chain-cascaded params, top-level externs
   ▼
Nested scopes (any depth) ── per-scope bindings, shared cells, externs
   ▼
Cycle-time ───────────────── per-pull state + external-write injection
```

Hosts impose their own layer names on this generic structure;
polydat itself remains layer-name-agnostic.

Each layer owns its own state. Inner layers see outer-layer
state via auto-extern + binding-time materialisation (S1+S2).
Outer layers do not see inner-layer state (no callback up the
chain). Cross-tier writes are bounded to the SharedCell
write-through mechanism (S5).

### Axiom L1 — Each layer owns its own state

**A scope instance — a root, a nested scope at any depth, a
traversal activation — owns its own state set. State written
at one layer is not visible at outer layers; state read at one
layer comes from that layer's own bindings or from outer layers
via chain-synthesised slots. There is no cross-tier shared
mutable state outside the named SharedCell write-through
mechanism specified by S5 (compile-emit write-through as the
cross-tier synthesis path).**

The kernel chain is a parent-child tree constructed through
parent-gated construction ([scope_model.md](scope_model.md)
§2); binding-time materialisation is the only state-crossing
surface; the Wire Materialization classification governs
read/write semantics. S5 specifies the only cross-tier write
surface; this axiom's claim is that the outer cell remains the
canonical state holder regardless of which inner tier issues
the write, so layer ownership is preserved across S5's routing.

### Axiom L2 — Two-lifecycle classification bridges layers

**Every input slot has one of two lifecycles: *effectively-
const* (resolved once at scope-init, frozen for the scope's
lifetime) or *dynamic* (resolved per pull at cycle time). The
lifecycle is *structural* — determined by the slot's
`InputKind` and its upstream wire chain, not by a runtime
flag. This classification is the layer-bridging mechanism: an
effectively-const slot is filled by the chain at scope-init
from an outer layer's binding; a dynamic slot is filled per
coordinate from the current layer's state advance.**

The classification is the program's, computed by one
classifier every engine shares
([runtime_model.md](runtime_model.md) §3), and the
const-binding contract of the [Evaluation Model](evaluation_model.md)
checks it twice: a compile-time wire-chain check (Plan A) and
the scope-init pull (Plan B). The classification is *known*
before the node tier ever sees a value; the chain enforces it
by populating slots according to each input's lifecycle.

**Sub-axiom L2.f — Failed const materialisation falls
through to the outer chain (L2 ⊓ T1).** When an
effectively-const binding's scope-init evaluation yields
`Value::None` (per the None propagation contract — see
[none_semantics.md](none_semantics.md)), the slot is
considered *unfilled at this layer*. The read invariant
(L1) then returns the outer scope's value for the same
name via the standard lookup chain. The effectively-const
guarantee is preserved at the outer layer: the outer
binding is itself effectively-const for the scope's
lifetime, so the value the inner reader observes is
stable across the activation. This composition of L2 (the
inner binding's lifecycle) with T1 (typed slots include
the None sentinel) gives the conditional-shadow semantic
on which a host's `set:`-style sugar relies: an
intermediate-layer `const X := <expr>` that yields a real
value shadows the outer X; one that yields None leaves the
outer X visible.

**Strict mode.** Silent fall-through on intermediate-layer
None can mask author intent: did the layer mean to provide a
shadow that happens to compute to None, or did it mean to
declare an `extern X` and forget to? Under the `strict` flag
of `subcontext::CompileOptions`, `build_subscope` queries
`PolydatKernel::find_l2f_violations` after scope-init and
escalates any const output materialised to `Value::None`
into `ContractViolation::StrictNonePropagation`. The
diagnostic names each offending binding and directs the
author to either ensure the binding yields a defined value
or remove the binding and declare `extern <name>` explicitly
if fall-through to outer was intended. The substrate provides
the mechanism; the policy (which builds get strict mode) is
the caller's choice through the flag.

### Note on cross-tier writes

The cross-tier-write mechanism (`SharedCell` write-through)
is a **synthesis-axis** concern and lives at S5 in §3 — the
mechanism fills outer-tier slots via inner-tier nodes' typed
output emission, so it belongs with the other slot-filling
axioms. The **layer-ownership guarantee** that S5 preserves
— that the outer cell remains the canonical state holder
and inner-tier writes flow through the typed-slot surface
rather than as ad-hoc cross-tier mutation — is L1's claim
(see §"Axiom L1 — Each layer owns its own state" above,
which names the SharedCell write-through as the only
permitted exception to single-layer state ownership).

---

## 6. The three pillars compose — the substrate as product

S, T, and L each provide one guarantee. The slot contract
(§2) is the product of all three. The substrate's claim is
that the three together are *sufficient* to make the slot
contract durable — and that *no two of three* is sufficient.

| If you have | But lack | The slot contract... |
|---|---|---|
| T + L | S | ...cannot be filled. Slots are typed and layer-aware but the chain has no synthesis mechanism — slots stay empty or take ad-hoc values. |
| S + L | T | ...is fillable but unsafe. The chain delivers values, but nothing guarantees type. Nodes do defensive type checking or coerce silently. |
| S + T | L | ...is fillable and typed but flat. Without layering, scope composition collapses; no lifecycle distinction, no nested scope tiers; S5's cross-tier write mechanism would still route values but L1's layer-ownership guarantee would be absent — written-to cells would have no stable owner. |
| **S + T + L** | — | **...is the durable slot contract.** Slots are filled, typed, layer-aware. Free composition follows. |

The substrate is *exactly* S + T + L. Adding more (e.g., a
fourth pillar for external-write timing, determinism, or
cross-tier write semantics)
would strengthen specific guarantees but the slot contract
holds under S + T + L alone.

---

## 7. The slot contract enables free graph composition

From the substrate, this consequence:

**A pure-typed function node composes inside a layered
stateful scope without either tier knowing about the other,
because the chain has synthesised the scope's relevant state
into the node's typed input slots. The node consumes typed
slots; the scope owns layered state; the chain mediates. No
per-composition wiring overhead at either tier.**

This consequence is what was previously informally called
"the overlap" or "the embedding property" — the load-bearing
fact that polydat graph composition has zero per-composition
negotiation cost. The substrate establishes it as a derivable
property of S + T + L, not a primitive claim.

---

## 8. Boundary mechanisms — named handlers

Certain timing and identity boundaries need named handlers.
These aren't substrate violations — they're substrate
extensions, each preserving the S/T/L axioms.

### 8.1 The external-write boundary

Per S4, external producers populate port-typed slots through
the kernel's typed writes. The timing of writes is determined
by the producer, not by the kernel; the kernel's contract is
that any write triggers standard provenance invalidation per
R2, and consumers re-evaluate on next pull per R1. Reads from
the slot are ordinary port reads; T1 + T2 ensure type safety
at the write boundary.

Hosts with specific timing requirements (runtimes injecting
values at well-defined synchronisation points, producers
driving values from event streams, or any other host-level
write-timing semantic) own that contract at the host layer.
The polydat kernel itself makes no commitment beyond "writes
trigger normal invalidation." This separation is what allows
polydat to serve as a substrate for diverse host runtimes
without embedding any host's specific write-timing
semantics.

### 8.2 SharedCell write-through (§3, S5)

A node's output value crosses an outer-tier boundary via
write-through routing. The node is unchanged; the chain
performs the routing. S5 names this as the only mechanism;
T2 ensures the type-check holds across the cell; S2's
synthesis at the outer scope's next read takes the updated
cell value; L1's layer-ownership invariant is preserved
because the outer cell remains the canonical state holder.

Cross-kernel visibility of the written value is owned by
the validity-tracking spec ([cross_fiber_invalidation.md](cross_fiber_invalidation.md)):
the producer's mutex write is accompanied by a revision
bump and an intent-bit set on the cell's defining scope;
every consumer observes the change on its next read, the
interpreter through its cone check and a compiled kernel
through its poll. No host-side refresh call is required.

### 8.3 Const lifecycle violations

When a `const X := <expr>` binding's RHS depends on a dynamic
input, L2's structural classification fails. The Evaluation
Model's const-binding contract owns the detection: Plan A
(compile-time wire-chain analysis) catches structural
violations; Plan B (the scope-init pull under `catch_unwind`)
catches semantic violations. The node tier never sees a
violation — it sees a value from the chain or an error from
the construction layer.

### 8.4 Native and closure steps (T3)

Compiled steps are ordinary slot consumers from the
substrate's perspective — declared inputs, declared outputs,
typed `PortType`s, consuming a slot buffer. When a node has no
native lowering, or its purity or lifecycle keeps it out of a
segment, the hybrid kernel runs it as a closure step and
compiles the rest into native segments (what a segment may
contain is [engines.md](engines.md)'s rule). T3 is the axiom
that makes the mix sound: both step kinds read and write the
same typed slots.

### 8.5 Diagnostic and side-channel node observable side effects

Some diagnostic nodes (`log_info`, `log_debug`, etc.) write
to stderr or to a log buffer during `eval`. From the
substrate's perspective: the node's *returned value* is still
a function of its inputs (T1, T2 preserved); the side effect
is *observable* but not *typed* — it does not flow through a
slot. `Purity::SideChannel { sink }` declares such an
observable; `Purity::Nondeterministic` declares externally or
historically varying return behavior. Neither kind joins a
native segment, so each fires under the same currency rule on
every engine; typed output ports remain governed by T1/T2,
while ordering and cross-fiber observability are governed by
[runtime_model.md](runtime_model.md) D2. No implicit
event-output port is synthesized.

---

## 9. Why this substrate matters

The substrate is what makes three of polydat's distinctive
capabilities work freely:

### 9.1 Context Fusion depends on the substrate

**Context Fusion** is the scope-init-time synthesis where
outer context fuses into the graph's declared slots. S1 + S2
are the synthesis mechanism; T1 + T2 guarantee the values
arrive typed; L1 + L2 carry the layered lifecycle. Context
Fusion is the substrate in motion at scope-init.

### 9.2 Node Fusion is sound under the substrate

**Node Fusion** is the compile-time rewriting where the
compiler inserts adapters at wire resolution and fuses
subgraphs into segments. Soundness reduces to "the rewrite
preserves the slot contract" — same input slots, same output
slots, same typed values, same lifecycle classification. T1 +
T2 give the rewriter a typed substrate; the rewrite preserves
the slot contract by construction. Fusion correctness is a
closure property over T2 + L2.

### 9.3 Parallel evaluation is safe under the substrate

Polydat kernels run in concurrent fibers, one kernel per
fiber, each created from the shared program
(`KernelProgram::create_kernel`). L1 (each layer owns its
state) plus one kernel per fiber means there is no shared
mutable state at the node tier across fibers; the only shared
registers are cells, attached by an explicit act. The
substrate is what makes the parallel safety claim cheap — it
follows directly from L1 + T1 (typed slots, layer-owned
state), not as a separate concurrency proof.

---

## 10. What this document does NOT specify

- **The grammar productions.** [polydat_grammar.md §21](polydat_grammar.md#sec-productions) owns
  the formal productions; this doc
  relies on the grammar exposing typed input ports.
- **The compilation pipeline mechanics.**
  [graph_compiler.md](graph_compiler.md)
  owns the graph compiler, kernel hoisting, and fusion
  passes.
  This doc relies on the compiler enforcing S1 (auto-extern),
  T1+T2 (typed slot construction), L2 (lifecycle
  classification).
- **The expression system as host utility.**
  [expression_engine.md](expression_engine.md) owns the
  host-facing expression engine. This doc
  relies on expression evaluation being a special case of
  node evaluation — same slot contract, same chain mediation.
- **The kernel-composition algebra in full.** The
  [Scope Model](scope_model.md), [Wire Materialization](wire_materialization.md),
  and [Subcontext Construction](subcontext_construction.md)
  cover the mechanics in detail; this doc names the
  substrate they collectively form but does not re-derive
  their machinery.
