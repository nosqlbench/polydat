---
type: specification
title: Wire Materialization
timestamp: 2026-09-25
description: "Cross-scope wire flow: the uniform read invariant, the write contract, and the materialization gradient from shared cells to resolvers."
tags: [scopes, runtime]
---

# Wire Materialization

This specification defines how a wire declared in one scope is made
readable, and where permitted writable, in descendant scopes. It
specifies the architectural model (one logical graph whose scope
boundaries partition lifecycle and access), the read invariant, the
write contract, and the materialization gradient: the ordered forms
(shared cell, value copy through the outer lookup, broadcast cell,
plain value copy, extern resolver) the materializer applies to each
wire, plus shadow suppression. A wire's *materialization* on a kernel
is the storage and wiring behind that kernel's local slot for the
wire. *Matter* is the binding source a scope's program is compiled
from (the matter AST); its declarations and modifiers classify each
wire.

**Related specifications:** the cell protocol (revision counter,
intent bits, the consumer's poll) is
[cross_fiber_invalidation.md](cross_fiber_invalidation.md);
the scope chain and its visibility rules are
[scope_model.md](scope_model.md).

Materialization is performed when the inner scope's kernel is
constructed, by interpreting the matter AST; it is not a public
bind a caller makes afterwards. Parent-gated construction (a child
is created only through a call on its parent,
`materialize_subscope`) calls the private
`materialize_wiring_from_outer` chokepoint in `kernel/state.rs`.
That function, called *the materializer* below, installs the
value, broadcast, shared-cell, and transit wiring that each
visible binding's classification prescribes.

---

## Architectural model

### One logical graph; scope boundaries partition lifecycle and access

The Polydat matter spanning a workload is *one logical graph*. Scope
boundaries (workload, phase, for_each iteration, op-template,
per-fiber) are not value-isolation barriers. They partition two
things:

- **Lifecycle.** A wire's value lives no longer than the scope
  that owns it: the phase scope owns phase bindings, fiber state
  owns per-cycle coordinates, and so on. Outer scopes outlive
  inner scopes, and ending an inner scope releases the wires it
  owns.
- **Access plane.** Each scope's kernel exposes a *local handle*
  for every wire it is permitted to read, and a write handle for
  every wire it is permitted to write. A handle is the inner
  scope's slot for a logical wire defined further out; reading
  the handle returns the current value of that logical wire.

A wire keeps its *identity* across scopes. Its *materialization*
on each scope's kernel (inlined as a constant, stored in a cell,
or wired to an upstream pull) is an implementation detail of how
the materializer builds that kernel's access plane.

### The read invariant

> Reading an inner-side handle for a cross-scope wire returns the
> same value that reading the wire on its owning kernel would
> return at the same moment.

The invariant holds for every wire, whether it is per-cycle,
per-iteration, constant, or otherwise, and whether or not it is
`shared`. The materializer must materialize each handle so that
the invariant holds without the *caller* (the wires layer, the
dispenser, an adapter) doing anything beyond a local read.

### The write contract

> Writing through an inner-side handle is permitted only when the
> wire's matter classifies it as `shared` (or equivalent
> cross-scope-writable form). When permitted, the write goes
> through the same storage the reads observe, mutex-gated,
> last-write-wins. Other scopes' readers see the new value on
> subsequent reads.

The `shared` modifier only grants write permission. It does *not*
change how reads are mediated; read access is uniform.

### Local reads are O(1)

Inner kernel reads do not walk scope chains; the read surface is
the local kernel's slot for the handle. Whatever backs the slot
(a literal constant, a shared cell with a valid bit, or a
value-only cell subscribed to upstream invalidation) is set up at
*construction*. A read checks the slot's validity and returns the
value without visiting ancestor kernels. Recomputable wires extend
the per-node valid-bit discipline across the scope boundary, so
invalidation propagates without traversal at read time.

---

## Materialization gradient

At inner kernel construction the materializer chooses one of these
forms for each visible cross-scope wire. The wire's matter
classification decides the form: the outer binding's modifier
(`const`, `shared`) and whether the name is an input slot or a
computed output of the outer program. The caller does *not*
decide it. `materialize_wiring_from_outer` applies the forms in
this order, and the first form that applies to a name binds it.
The diagram shows the decision for one inner input slot `NAME`.

![Decision chain for one inner input slot NAME: if a cell named NAME is visible at the outer scope, it is discarded when the inner scope declares const NAME and otherwise attached (form 1); else the value of outer.lookup is copied when NAME is an outer input slot or const output (form 2); else a computed outer output's broadcast cell is attached (form 3); else a value the outer lookup yields is copied once (form 4); else the registered extern resolver is asked, and failing that the declared default is kept (form 5)](../diagrams/wire_materialization-gradient.png)

### 1. Shared-cell cascade

A shared cell is one mutex-protected value register that several
kernels read and write for a `shared` binding. Every cell visible
at the outer scope is attached to the inner input slot of the same
name. The visible cells are those on the outer scope's own input
slots (its `shared` declarations and cells it inherited onto
slots) plus its transit cells, which are cells the outer scope
holds only to pass to descendants because it has no slot of that
name. The inner kernel
and the defining scope's kernel then hold local handles backed by
the same cell. The mutex serializes writes, and every write
increments the cell's revision counter and sets the cell's bit in
the defining scope's intent word (a per-scope bitmask that tells
readers which of the scope's cells have been written) per
[cross_fiber_invalidation.md](cross_fiber_invalidation.md).
Inner writes are visible to the outer scope, and to siblings
sharing the cell, on the next read, with no host-side refresh.

A cell whose name the inner scope declares `const` is discarded:
it is neither attached nor forwarded (transit suppression, below).
A cell with no matching inner slot is stored on the inner kernel
as a transit cell, so that a deeper descendant can attach it.

This form materializes `shared X := …` declarations and any other
binding that opts into cross-scope write-back.

### 2. Value copy through the outer lookup

For each output of the outer program that names an inner input
slot not bound by step 1, the first of these applies.

When the name is an input slot on the outer program too (a
passthrough extern) or is a `const` output of the outer program,
the value is copied from `outer.lookup(name)`, the two-tier read
that consults the outer's own const first and falls through to
its wired-in input when the const is `None`
([none_semantics.md](none_semantics.md)). The copy passes
through `adapt_boundary_value` (the boundary adapter catalog of
[type_system.md](type_system.md) §6.2). No cell is used. A
`const` is evaluated once, at the outer kernel's initialization,
and does not change for that kernel's life, so a copy is
semantically equivalent to a cell and costs no cell traffic.
Initialization has already applied the conditional shadow: a
const whose own expression yielded `None` holds the value its
fallback input received from the grandparent, so the copy carries
the grandparent's value down the chain.

### 3. Broadcast cell for computed outputs

A broadcast cell is a cell through which a kernel publishes one
of its computed outputs each time it pulls that output. Otherwise,
when the outer output is computed (node-backed, with no input slot
on the outer program) and the outer kernel publishes it through a
broadcast cell, that cell is attached to the inner slot. Each time the outer kernel re-evaluates the wire, per cycle,
it writes the new value into the cell, bumps the revision, and
sets the cell's intent bit. Every consumer detects the change on
its next read through the bulk-mask and per-cell revision compare
protocol of
[cross_fiber_invalidation.md](cross_fiber_invalidation.md).
The inner side cannot write to this cell.

This form materializes phase bindings that descendants reference
(`load := add(cycle, 1)`) and any other computed binding that is
visible across scope boundaries and is not `shared`.

### 4. Plain value copy

Otherwise, when the outer lookup yields a value but no cell
exists, the value is copied once, through `adapt_boundary_value`.

### 5. The registered extern resolver

When no scope in the outer chain binds the name, the host's
resolver (`dsl::factories::register_extern_resolver`) is
consulted, and a value it returns is copied through
`adapt_boundary_value`. A slot that no form fills keeps its
declared default (`None` when it has none).

After these steps, the inner kernel is initialized: every `const`
of the inner program is evaluated once against the populated slots
and fixed for the kernel's life
([evaluation_model.md](evaluation_model.md), "Const Binding
Contract"), and a const whose expression fails makes the
construction fail, naming the const. Then the inner scope's
coordinate path becomes its own coordinates followed by the
outer's.

### Inlined constants and the parameter scope

A wire whose value is statically known (a literal right-hand side,
including a `const` with a literal right-hand side, or a node with
no input in its provenance) is folded into the program at build.
All four engines (the interpreter, the closure tier, native, and
pure native) perform the same compile-constant fold, and step 2
copies the folded value into a descendant. Such a wire has no cell
and no valid bit; reads are constant lookups.

Workload parameters are *not* folded into the program that
declares the workload. A parameter scope at the root of the chain
holds one `const NAME := <literal>` per parameter. The workload
program declares each `NAME` as an extern and inherits it as a
passthrough, so it cascades through descendants as an
auto-passthrough slot rather than as one of the program's own
coordinates. Every descendant scope likewise declares it as an
extern, and the value flows down through step 2.

The rationale is **lexical shadowing**: an intermediate scope must
be able to redeclare a parameter name for its subtree without
rewriting the root program. Because the parameter scope is a
separate lexical layer, an intermediate
`const NAME := <override>` shadows the cascaded value, and
descendants resolve `NAME` through the standard `extern NAME`
lookup. If the parameter were folded into the root program,
lookup would return it through `get_constant` before consulting
the chain wiring, and the override would be silently masked. The
extra indirection is a deliberate trade-off in favour of general
lexical-scope semantics.

### Rationale: matter-driven materialization

The caller (the scope synthesizer, the wires layer, the
dispenser) does not choose the materialization. The matter AST
classifies each wire, and the materializer materializes it
accordingly. As a result:

- No caller site decides whether a wire is per-cycle; the matter
  records it.
- Callers compose no external fallback chains, because the inner
  kernel's local read is correct on its own.
- A wire's materialization can change (for example, from literal
  to computed) without touching consumers; only the matter and
  the materializer change.

### Local-authoritative shadow (transit suppression)

When an inner scope declares `const NAME := …` for a name that an
outer scope in the chain *also* exports (as a folded constant or a
passthrough output), the inner declaration becomes the
authoritative writer of that name for its subtree. The chain must
not pass the upstream value beyond this scope, or descendants
would read it instead of the local declaration.

The materializer enforces this through **transit suppression**:
during the cell cascade (step 1), any cell visible at the outer
scope whose name matches a local `const` output is discarded. It
is neither attached to a local slot nor forwarded to descendants.
Step 2 then runs normally: for names that are not shadowed it
copies the outer's view from `outer.lookup(name)` into the local
slot, and for shadowed names it leaves the name to the local
folded buffer, which the scope owns.

Every scope-tree node uses this mechanism: phases, op-templates,
comprehensions, do-loops, and intermediate binding scopes. The
synthesizer does not coordinate it; the materializer enforces it
uniformly. The result is standard lexical shadowing: the closest
declaration wins, transit cells stop at the first redeclaration,
and the chain stays self-consistent.

### The gradient on the four engines

The forms above describe the interpreter kernel's materializer.
The same forms exist on the compiled engines through the `Kernel`
trait, and the gradient holds there, except that pure native
makes no broadcast cells (form 3;
[cross_fiber_invalidation.md](cross_fiber_invalidation.md) §3.1).
`shared` bindings are bound to cells on the interpreter, the
closure tier, and the native engine alike
(`Kernel::attach_shared_cell`, `Kernel::shared_cells`), so a write
through any kernel holding the cell is what the others read next.
A value copy is `Kernel::set_input`, and the inlined-constant form
is the compile-constant fold all four engines perform at build. A traversal's activation snapshots the
cascaded wires into the body's kernel on whichever engine runs it
([for_traversal.md](for_traversal.md)).

### Value-clone economy on the chain

A value copy costs what cloning the `Value` costs. The
container variants are Arc-backed — `Str(Arc<str>)`,
`Bytes(Arc<[u8]>)`, `Json(Arc<serde_json::Value>)`,
`Handle(Arc<dyn Any + Send + Sync>)`, and the `Vec*` lane
family over `SliceArc<T>` — so a clone is one atomic increment
and no allocation. `Ext` is boxed and clones through its own
reflected clone, and the scalar variants (`U64`, `I64`, `F64`,
`Bool`, the two-limb `U128`/`I128`, `Reg128`) are copied. A
workload that reads the same cross-scope wire every cycle
therefore pays one atomic increment per read and never a heap
allocation on that wire, whatever its variant. Consumers that
need an owned tree (a `Json` body they mutate before serialising)
deep-clone explicitly at the consume site (`Value::to_json_value`
does so for the `Json` variant); the chain itself never
deep-clones.
