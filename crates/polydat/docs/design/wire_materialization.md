# Wire Materialization

The mechanism for cross-scope wire flow: architectural model
(one logical graph; scope boundaries partition lifecycle and
access; the uniform read invariant; the write contract) and
the materialization gradient the materializer applies (shared
cell / value copy through the outer lookup / broadcast cell /
resolver, plus shadow suppression). The cell protocol itself —
revision counter, intent bits, the consumer's poll — is
[cross_fiber_invalidation.md](cross_fiber_invalidation.md);
the scope chain and its visibility rules are
[scope_model.md](scope_model.md).

The construction-time wiring operation is general matter-AST
interpretation, not a public post-hoc bind. Parent-gated
construction (`materialize_subscope`) drives the private
`materialize_wiring_from_outer` chokepoint in
`kernel/state.rs`, which installs the value, broadcast,
shared-cell, and transit wiring prescribed by each visible
binding's classification.

---

## Architectural model

### One logical graph; scope boundaries partition lifecycle and access

The Polydat matter spanning a workload is *one logical graph*. Scope
boundaries (workload, phase, for_each iteration, op-template,
per-fiber) are not value-isolation barriers — they partition:

- **Lifecycle.** A wire's value-instance lifetime is bounded by
  the scope that owns it (phase scope owns phase bindings; fiber
  state owns per-cycle coordinates; etc.). Outer scopes outlive
  inner; inner scope ends release inner-owned wires.
- **Access plane.** Each scope's kernel exposes a *local handle*
  for every wire it is permitted to read (and a write handle for
  every wire it is permitted to write to). The handle is the
  inner-side surface for the same logical wire defined further
  out; reading the handle returns the current value of the
  logical wire.

The wire's *identity* is preserved across scopes. The wire's
*materialization* on each scope's kernel — whether the value is
inlined as a constant, stored in a cell, or wired to chain to an
upstream pull — is an implementation detail of how the
materializer builds the access plane on that scope.

### The read invariant

> Reading an inner-side handle for a cross-scope wire returns the
> same value that reading the wire on its owning kernel would
> return at the same moment.

This is uniform. It does not depend on whether the wire is
per-cycle, per-iteration, constant, or otherwise. It does not
depend on whether the wire is `shared`. The materializer is
responsible for materializing each handle so the invariant holds
without the *caller* (the wires layer, the dispenser, an adapter)
doing anything beyond a local read.

### The write contract

> Writing through an inner-side handle is permitted only when the
> wire's matter classifies it as `shared` (or equivalent
> cross-scope-writable form). When permitted, the write goes
> through the same storage the reads observe, mutex-gated,
> last-write-wins. Other scopes' readers see the new value on
> subsequent reads.

The `shared` modifier is purely a write-permission flag. It does
*not* control read-mediation — read access is uniform.

### Local reads are O(1)

Inner kernel reads do not walk scope chains. The local kernel's
slot for the handle is the read surface. Whatever wiring backs
the slot (literal constant, shared cell with valid bit, value-only
cell with subscription to upstream invalidation) is set up at
*construction*. Reads check the slot's validity and return the
value, taking no special path through ancestor kernels at read
time. Recomputable wires use the existing per-node valid-bit
discipline extended across the scope boundary so invalidation
propagates without read-time traversal.

---

## Materialization gradient

The materializer chooses one of these forms for each visible
cross-scope wire at inner kernel construction. The choice is
dictated by the wire's matter classification — the outer
binding's modifier (`const`, `shared`) and whether the name is an
input slot or a computed output on the outer program — *not* by
the caller. `materialize_wiring_from_outer` applies them in this
order.

### 1. Shared-cell cascade

Every cell visible at the outer scope — the cells on its own
input slots (its `shared` declarations and cells it inherited
onto slots) plus its transit cells — attaches to the inner input
slot of the same name. Storage is shared with the defining
scope's kernel; both sides hold local handles backed by the same
cell; the mutex serializes writes; the cell's revision counter
and the defining scope's intent-dirty bit are bumped on every
write per
[cross_fiber_invalidation.md](cross_fiber_invalidation.md).
Inner writes are visible to the outer scope (and to siblings
sharing the cell) on the next read, without any host-side
refresh ceremony.

A cell whose name the inner scope declares `const` is dropped —
neither attached nor forwarded (transit suppression, below). A
cell with no matching inner slot is stored on the inner kernel
as transit so a deeper descendant can pick it up.

This is the materialization for `shared X := …` declarations and
for anything else that opts into cross-scope write-back.

### 2. Value copy through the outer lookup

For each output of the outer program that names an inner input
slot not bound by step 1, the first of these applies.

When the name is an input slot on the outer program too (a
passthrough extern) or is a `const` output of the outer program,
the value is copied from `outer.lookup(name)` — the two-tier read
that consults the outer's own folded constant first and falls
through to its wired-in input when the constant is `None`
([none_semantics.md](none_semantics.md)). The copy passes
through `adapt_boundary_value` (the boundary adapter catalog of
[type_system.md](type_system.md) §6.2). No cell: a const is
effectively-const for the scope's lifetime, so a copy is
semantically equivalent to a cell and costs no cell traffic; and
a const whose buffer is `None` must not be broadcast, or the
chain walk that gives descendants the grandparent's value would
be defeated.

### 3. Broadcast cell for computed outputs

Otherwise, when the outer output is computed (node-backed, no
input slot on the outer program) and the outer kernel publishes
it through a broadcast cell, the cell attaches to the inner
slot. The outer's per-cycle re-evaluation of the wire writes
its new value into the cell, bumps the revision, and sets the
cell's intent bit; every consumer's cone walker observes the
change on its next read via the bulk-mask + per-cell-revision
compare protocol of
[cross_fiber_invalidation.md](cross_fiber_invalidation.md).
The inner side has no write surface to this cell.

This is the materialization for phase bindings that descendants
reference (`load := add(cycle, 1)`) and any computed binding
visible across scope boundaries that isn't `shared`.

### 4. Plain value copy

Otherwise, when the outer lookup yields a value but no cell
exists, the value is copied once, through `adapt_boundary_value`.

### 5. The registered extern resolver

When the outer chain has no binding for the name at all, the
host's resolver (`dsl::factories::register_extern_resolver`) is
consulted; a value it returns is copied through
`adapt_boundary_value`. A slot no form fills stays at its
declared default (`None` when it has none).

After the walk, every `const` output of the inner program is
pulled once against the populated slots
([evaluation_model.md](evaluation_model.md), Plan B), and the
inner scope's coordinate path becomes its own coordinates
followed by the outer's.

### Inlined constants and the parameter scope

A wire whose value is statically known — a literal RHS, a folded
const binding — is folded into the program at build; every
engine performs the same compile-constant fold, and the folded
value is what step 2 copies into a descendant. No cell, no valid
bit; reads are constant lookups.

Workload parameters are deliberately *not* folded into the
program that declares the workload. A parameter scope at the root
of the chain holds one `const NAME := <literal>` per parameter;
the workload program externs each `NAME` and inherits it as a
passthrough so it cascades through descendants as an auto-
passthrough slot rather than as one of the program's own
coordinates; every descendant scope likewise externs it, and the
value flows down through step 2.

The reason is **lexical shadowing**: an intermediate scope must
be able to redeclare a parameter name for its subtree without
rewriting the root program. With the parameter scope as a
separate lexical layer, an intermediate `const NAME := <override>`
shadows the cascaded value, and descendants resolve `NAME`
through the standard `extern NAME` lookup. Folding the
parameter into the root program instead would short-circuit
lookup through `get_constant` before the chain wiring is
consulted, and the override would be silently masked. That
design was tried and rejected for exactly this reason; the
indirection is a deliberate trade-off in favour of the general
lexical-scope semantics.

### Why the gradient is matter-driven, not caller-driven

The caller (the scope synthesizer, the wires layer, the
dispenser) does not pick the materialization. The matter AST
classifies each wire; the materializer materializes
accordingly. This:

- Removes the "is this wire per-cycle or not" question from
  every caller site — the matter knows.
- Eliminates external fallback chains composed by the caller —
  the inner kernel's local read is correct on its own because
  the materializer set the wiring up.
- Lets the same wire's materialization change (literal →
  computed) without touching consumers — only the matter and
  the materializer change.

### Local-authoritative shadow (transit suppression)

When an inner scope declares `const NAME := …` for a name that
is *also* exported by an outer scope in the chain (as a folded
constant or a passthrough output), the inner declaration is the
new authoritative writer for that name over its subtree. The
chain must not carry the upstream value past this scope, or
descendants would read it instead of the local declaration.

The materializer enforces this through **transit suppression**:
during the cell cascade (step 1), any cell visible at the outer
scope whose name matches a local `const` output is dropped on
the floor — not attached to a local slot, not forwarded to
descendants. Step 2's value-copy path then runs normally and
copies the outer's view via `outer.lookup(name)` into the local
slot (for non-shadow names) or leaves the name to the local
folded buffer (for shadow names, which the scope owns).

This is the same mechanism every scope-tree node uses —
phases, op-templates, comprehensions, do-loops, intermediate
binding scopes. The synthesizer doesn't need to coordinate; the
materializer enforces the invariant uniformly. The result is
standard lexical-scope shadowing: the closest declaration
wins, transit cells stop at the first redeclaration, and the
chain remains self-consistent.

### The gradient on every engine

The forms above are the interpreter kernel's materializer. The
same forms exist on every engine through the `Kernel` trait, and
the gradient holds there: `shared` bindings are bound to cells on
the interpreter, the closure tier, and the native engine alike
(`Kernel::attach_shared_cell`, `Kernel::shared_cells`), so a
write on any holder is what the others read next; a value copy
is `Kernel::set_input`; and the inlined-constant form is the
compile-constant fold every engine performs at build. A
traversal's activation snapshots the cascaded wires into the
body's kernel on whichever engine runs it
([for_traversal.md](for_traversal.md)).

### Value-clone economy on the chain

A value copy costs what cloning the `Value` costs. The
container variants are Arc-backed — `Str(Arc<str>)`,
`Bytes(Arc<[u8]>)`, `Json(Arc<serde_json::Value>)`,
`Handle(Arc<dyn Any + Send + Sync>)`, and the `Vec*` lane
family over `SliceArc<T>` — so a clone is one atomic increment
and no allocation; `Ext` is boxed and clones through its own
reflected clone; the scalar variants (`U64`, `I64`, `F64`,
`Bool`, the two-limb `U128`/`I128`, `Reg128`) copy. A workload
that reads the same cross-scope wire every cycle therefore pays
an atomic increment per read and never a heap allocation on that
wire, whichever variant it carries. Consumers that need an owned
tree (a `Json` body they mutate before serialising) deep-clone
explicitly at the consume site (`Value::to_json_value` does so
for the `Json` variant); the chain itself never does.
