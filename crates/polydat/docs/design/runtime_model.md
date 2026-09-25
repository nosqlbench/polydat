---
type: specification
title: The Runtime Model
timestamp: 2026-09-25
description: The R-axioms of data flow, currency, invalidation, and output ownership, and the D-axioms of determinism, on all four engines.
tags: [runtime]
---

# The Runtime Model

This document specifies the runtime contract of a polydat kernel
on all four engines: the interpreter (P1), the closure tier (P2),
native (P3), and pure native. The R-axioms specify the runtime
mechanics: data flows only along declared wires (R3); a step
stays current until an input in its provenance changes (R1); a
write marks the dependent steps not current, and a pull
re-evaluates only the not-current steps its output depends on
(R2); and every output is kept until an input in its provenance
changes (R4). The D-axioms (D1 through
D4) specify the determinism a host can rely on as a consequence.
The D-axioms hold per fiber, since each fiber holds its own
kernel; coordination across fibers is the host's concern.

**Related specifications:** the
[Composition Substrate](composition_substrate.md) (the static
slot contract, S/T/L axioms, that this runtime implements); the
[Graph Compiler](graph_compiler.md) (the construction passes,
H/CF/NF axioms, that produce a program); the
[Expression Engine](expression_engine.md) (cites D1/D2/D3 as its
bounded determinism); the [Polydat Grammar](polydat_grammar.md)
(G4 port-typed expressions underwrite D1; G5 structural
lifecycle classification underwrites R1 and D3).

---

## Terms

Each term is defined before it is used here and in every later
section.

- **Program.** The immutable result of compilation: nodes,
  wiring, input definitions, and the output map. One program is
  shared by every kernel created from it.
- **Kernel.** One state over a program: its input values, its
  outputs, and the storage behind them. A kernel is owned by one
  fiber at a time.
- **Node.** A function in the program's graph, with typed input
  and output ports and a declared purity (`PolydatNode::purity`).
- **Wire.** A named value, connecting one node's output port to
  the input ports that read it.
- **Input.** A slot whose value comes from outside the graph: a
  coordinate, written together with the others by `set_inputs`,
  or an extern, written by name through `set_input`. An extern
  bound to a shared cell takes the cell's published value (§5).
- **Write.** A host act that sets one or more inputs:
  `set_inputs`, `set_input`, or a cell publication a kernel
  observes at its next evaluation.
- **Change.** An input changes when it is written. On the
  interpreter every write is a change; on the compiled engines a
  coordinate written with the value it already holds does not
  change.
- **Pull.** A host request for one named output. A pull
  evaluates what the output needs and returns its value.
- **Cone.** The steps an output transitively depends on, in
  topological order. A pull runs only its output's cone.
- **Provenance.** The set of inputs a node transitively depends
  on, fixed at build (§2).
- **Fusion unit.** A connected, convex group of nodes with
  native lowerings, compiled to one native function that runs
  whole ([engines.md](engines.md) §8).
- **Engine.** One of the four ways a program runs: the
  interpreter (P1), the closure tier (P2), the native tier (P3),
  and pure native. "Every engine" in this specification means
  all four; a statement that holds on fewer names them.
- **Step.** The unit an engine evaluates and caches. On the
  interpreter and the closure tier a step is one node. On the
  native tier a step is a fusion unit or, for a node without a
  lowering, that node's closure. On pure native every step is a
  fusion unit. A step's provenance is the union of its nodes'.
- **Current.** A step is current when its cached output may be
  returned without running it. Rule R1 (§3) states when a step
  is current.
- **Round.** The span from the first evaluation after a write
  to the next write. Within one round each step runs at most
  once, whatever the number of pulls.
- **Volatile.** A step whose value is not a function of its
  provenance, because a node in it is nondeterministic or its
  wire carries the `volatile` modifier. Sub-axiom R1.v (§3)
  states how volatile steps are evaluated.

---

## 1. Data flow along the wire chain

Every value in a polydat kernel flows along a **declared
wire**. The compiled program's wiring is the data-dependency
graph: wire `w` connects node `u`'s output port to node `v`'s
input port iff the assembled DAG (per the Graph Compiler's
pipeline) declared that connection. No data flows outside the
wire chain — nodes do not write to or read from shared state,
do not consult global registries not named in their declared
inputs, do not observe timing or order-of-evaluation beyond
their declared input slots.

The substrate calls this "data linearisation embedded in graph
structure": the wiring itself fixes the evaluation order, and no
separate execution-order plan exists beside it. A compiled
engine's step order is a topological order of the same graph,
derived from the wiring and never declared separately.

Concrete consequences:

- Two pulls of the same pure output with unchanged complete
  upstream state produce the same value.
- Kernels created from one shared program produce equal pure
  outputs when their complete state and declared dependencies
  are equal, as formalized by D4.
- A pure node's output is a function of its inputs and
  configuration. Side-channel and nondeterministic nodes declare
  the additional observables or dependencies through `purity()`.

---

## 2. Dependency tracking

Each wire's **upstream cone** — the set of nodes whose outputs
(transitively) feed into the wire — is known at build,
structurally. The Graph Compiler's hoisting analysis (§3 of
[graph_compiler.md](graph_compiler.md)) computes the cone for
every wire as part of lifecycle classification: H1 guarantees
totality, H2 guarantees monotonicity under fan-in.

The program records, for every node, the exact set of inputs it
transitively depends on: its **provenance**, a multi-word mask
with one bit per input slot, so a program with more than 64
inputs is tracked exactly. Provenance is computed once, when
the program is constructed, and all four engines derive their
invalidation plans from it: the interpreter inverts it into
per-input dependent lists, and a compiled kernel inverts it
into per-input dependent step lists and per-output cone orders.

**For any output, the set of inputs it is a function of is
exact, computed at build, and constant across evaluations.**
Dependencies are never discovered at runtime; they follow from
the graph's structure.

---

## 3. Currency — the one evaluation rule

All four engines evaluate under one rule:

> A step is current until an input in its provenance changes.
> A nondeterministic step is never current. A step with no
> input in its provenance is compile-constant and is folded at
> build. Every other step runs at the first pull whose cone
> contains it after it stopped being current.

A step is the engine's unit of evaluation (Terms). A host may
rely on this rule. The bookkeeping that implements it differs by
engine, as the table shows, and never changes a result:

| Engine | Realisation |
|---|---|
| Interpreter | A clean flag per node (`node_clean`); per-input dependent lists (`input_dependents`) cleared on every write; a list of nondeterministic nodes cleared on every write; a cell-revision check at every memoized read (§5). A write is itself the change: the interpreter does not compare the new value with the old, so a same-value rewrite re-runs the dependents, which a side channel in the cone must observe. |
| Closure tier and native tier | An `Invalidation` plan derived from provenance (per-input dependent steps and per-output cone orders); a clean flag per step; a round number per step recording the evaluation round it last ran in, which never clears an output; the volatile steps never current. A coordinate counts as changed only when its value differs from the one it replaces. |
| Pure native | A clean flag per fusion unit; per-input dependent units (`pushpull`) or every unit (`raw`) cleared on a write; per-output cone orders of units, closed over each unit's producers; the units holding volatile steps cleared on every write. The program's single native function is entered with the pulled cone's precomputed order and the flags, tests each unit's flag in native code, and dispatches the stale ones. |

The rule depends on lifecycle classification, which has one
classifier, `PolydatProgram::classify_lifecycle`, shared by the
interpreter's fold and the three compiled engines: a node is
compile-constant when its provenance contains no coordinate or
external-write input and nothing upstream is nondeterministic or
`volatile`; scope-init when its provenance contains only
iteration externs; dynamic otherwise. The compile-constant fold
therefore runs at build on all four engines: every value that
can be computed at build is computed at build, and a failure to
compute one is a build error.

The **effectively-const** steps (per the Graph Compiler's
hoisting analysis) are the special case of the rule with no
dynamic input in their provenance: computed once at scope-init
and current for the scope's lifetime. Provenance modes on the
compiled engines (`Raw`, `Push`, `Pull`, `PushPull`, with
`Auto` choosing among them; [engines.md](engines.md) §4)
decide how much of the bookkeeping
a kernel keeps; they are optimisations over this rule and never
change what a pull returns.

### Axiom R1 — Per-eval currency memoization

**A step is evaluated at most once between any two events that
make it not current in a given kernel. Multiple pulls touching
the step between such events use the cached result; the cache
is reset only when an input in the step's provenance changes,
or by `invalidate_all`.**

Rationale: without memoization a pull would re-evaluate its
whole cone every time, and a side channel in the cone would fire
once per pull rather than once per change. With it, the cost of
a pull is bounded by what changed (D3), and a step's side
effects are observed once per event that made it not current
(D2). Because each layer owns its state (substrate L1), each
kernel holds its own currency, and fibers never contend for a
cache.

### Sub-axiom R1.v — Volatility carves out clean-flag memoization

**A wire is *volatile* when its value is not a function of its
declared inputs. Volatile wires opt out of clean-flag
memoization across writes: every write (every `set_inputs` or
`set_input`) re-evaluates the producing step on next pull,
regardless of whether any of the step's declared inputs
changed. Between two writes, the step is evaluated at most once
and the result is cached for subsequent reads — this gives
consumers consistency between writes. Volatility is contagious
— the lifecycle classifier propagates it through the wire chain
so every step whose dependency cone touches a volatile producer
is itself treated as volatile.**

Volatility arises from two distinct sources:

- **Intrinsic.** A library node declares itself volatile by
  returning `Purity::Nondeterministic { reason }` from
  `PolydatNode::purity`. Examples: `current_epoch_millis`,
  `counter`, `elapsed_millis`, `thread_id`,
  `session_start_millis`, entropy sources, and any node
  whose output is not a pure function of its declared
  inputs. The library imposes volatility; no user opt-in is
  required, and the workload author cannot remove the marker.
- **User opt-in.** A wire's binding declares the `volatile`
  modifier, marking the wire as must-not-be-const-folded.
  The author is asserting that the value should never be
  cached across writes even though the compiler can't infer
  it from the wire chain (e.g., a node that reads external
  mutable state the polydat layer cannot see).

Both sources produce the same values on every engine, by the
same mechanism — with one visible difference in *when* a read
happens, which "Read granularity" below states:

- The wire is excluded from the compile-constant fold — the
  canonical workload hash sees node type and wiring shape but
  never the value, keeping workload identity stable across
  processes.
- The lifecycle classifier marks the producing node dynamic
  and nondeterministic; its fixed-point propagation marks
  every downstream consumer the same. The interpreter clears
  those nodes' clean flags on every write; a compiled kernel
  clears its volatile steps at the first evaluation after a
  write and never records them as current.
- The intrinsic declaration is authoritative: an absent user
  modifier does not override a library-declared volatile
  node, and a present user modifier on a wire that reads a
  library-declared-pure node still makes the wire itself
  volatile (and contaminates its downstream).

**Consistency between writes.** Between two writes, a volatile
step is evaluated at most once and cached — consumers reading
the same value multiple times between writes observe a
consistent value. The "every write re-evaluates" guarantee is
at the granularity of the write, not per individual pull. This
is the correct semantic for temporal nodes (an op reading
`current_epoch_millis` several times between writes sees one
timestamp) and matches the mechanism every engine delivers.

**Read granularity is the step's, and the step is the
engine's.** The paragraph above is per *step*, and a step
is one node on the interpreter and the closure tier but a whole
fusion unit on native code (Terms). Two volatile wires are therefore
two steps on one engine and may be one on another, and that is
observable — it is the only way an engine's realisation shows
through, because a volatile step is the one step whose value is
not a function of anything the engine can see:

| Engine | Two volatile wires pulled in one write |
|---|---|
| Interpreter, closure tier | Two steps. Each reads when its own output is first pulled, so a change made between the two pulls is visible to the second. |
| Native tier, pure native | Two steps when no wire connects them, since native code fuses only connected, convex groups of nodes (engines.md §8), and then each reads as above. When a wire connects them they are one fusion unit, a segment or a block of the one function: both read together at the first pull of either, so a change made between the pulls is not visible until the next write. |

So, normatively:

- **Guaranteed.** Each volatile step is evaluated at most once
  per write, re-evaluated on the next write, and answers every
  read within that write with the value it read. No engine may
  carry a volatile value across a write (the fold and clean-flag
  exclusions above are what enforce it).
- **Not guaranteed.** That two volatile reads within one write
  are simultaneous, or that they are distinct. A host that needs
  two readings to come from one instant must take them in one
  node and return both, which makes them one step on every
  engine; a host that needs two distinct readings must put a
  write between them.

This is a consequence of R1's step definition rather than a
concession to it, and it cannot be removed by making the engines
agree: native code fuses connected nodes into units that run
whole, so it has no smaller unit to read at. Pinned by
`tests/host_node_tiers.rs`.

What volatility is NOT for: ordinary external-write inputs (per
composition_substrate S4). Provenance handles re-evaluation
correctly via input-change tracking when an external write
modifies a slot — the consumers re-evaluate on next pull through
the standard R2 mechanism. Volatility is the explicit marker for
the genuinely nondeterministic case where input-change tracking
is insufficient because the value does not depend on the
declared inputs at all. The grammar's `volatile` modifier (G2)
is the syntactic surface.

---

## 4. Invalidation effects

A write to an input slot marks not current every step whose
provenance includes that slot. A pull then runs, in order, the
not-current steps of the pulled output's cone and no others.
This is the **hybrid push/pull invalidation model**: the dirty
*signal* is push-side (a write marks its dependents at once);
the dirty *response* is pull-side (a step is re-evaluated only
when a pull's cone contains it). The figure shows one write
followed by two pulls of the same output.

![Push/pull invalidation: the host writes input x and the kernel marks not current every step with x in its provenance; the host pulls out1 and the kernel runs the not-current steps of out1's cone in order and returns an owned copy; a second pull with no write in between runs no step and returns the same value](../diagrams/runtime_model-push-pull.png)

The model has three named properties:

- **Lazy at the pull side.** Unused outputs are never
  recomputed. If a host pulls only output `out1`, the steps of
  `out2`'s cone that `out1`'s cone shares are evaluated, but
  the steps only `out2` needs are not visited even if a write
  marked them not current.
- **Eager at the push side.** A write does its dirty marking
  through the precomputed per-input dependent list. The lookup
  is O(dependents of the written input), not O(total steps);
  the dependent list is structural, derived from provenance at
  build.
- **Forward-only.** Dirty marks propagate forward along the
  wire chain (an input write dirties downstream consumers, not
  upstream producers). Invalidation crosses a scope boundary
  only through S5's `SharedCell` write-through, the only
  permitted cross-tier write path, and other kernels learn of
  the write by seeing the cell's revision number change (§5).

### Axiom R2 — Hybrid push/pull invalidation

**Invalidation in polydat is a hybrid: a write to an input slot
marks not current every step whose provenance includes the
written input; subsequent pulls then lazily re-evaluate only the
not-current steps in the pulled output's cone. A step in no
pulled cone is never re-evaluated, regardless of upstream
writes.**

Rationale: the push half lets a pull decide what to run without
scanning the graph, and the pull half means an unused output
costs nothing. Either half alone reintroduces one of those two
costs. On all four engines the two halves use one plan from two
sides: the interpreter walks the cone recursively and stops at a
clean node, and a compiled kernel walks the output's precomputed
cone order and skips a step that is current.

### Axiom R3 — Forward-only data flow

**Data flow in a kernel evaluation is forward-only along
declared wires from input slots to output slots. Invalidation
never propagates backward; cross-tier writes are restricted to
the substrate's S5 SharedCell write-through mechanism; there is
no out-of-band data channel between nodes or between scopes.**

Rationale: the wire graph is acyclic (the assembler rejects
a cycle as `AssemblyError::CycleDetected`), so a topological
order exists and all four engines evaluate in one; S5 is the only
cross-tier write surface, and the parent-gated binder
([scope_model.md](scope_model.md) §4) is the only path that
binds a child's slots to an outer scope, so no construction can
introduce a backward or out-of-band channel.

### Axiom R4 — Outputs are owned by their provenance

**A step's output, immediate or by reference, is kept and
returned by every pull until an input in its provenance
changes. No other event discards an output: no evaluation
round, generation, epoch, or thread boundary invalidates
anything in the provenance rules, and no step is exempt from
R1. Each kernel owns the storage behind its outputs (the
interpreter's `Value` buffers; a compiled kernel's slot buffer
and the scratch entries its steps publish `Ref2` pairs into),
and no storage belongs to a thread. A read returns an owned
copy to the reader.**

Rationale: R1 is the complete caching contract, and under it
invalidation is per input. A reset that discarded every string
at every write would be an all-or-none invalidation, which the
model does not have, and a step that had to rerun to survive
such a reset would break R1's caching. Storage owned by the
kernel whose step wrote it lives exactly as long as the output
does (L1 at runtime, §5); storage owned by a thread would need a
separate lifetime rule, and any such rule would contradict this
one.

Enforcement: `compiled_handles.md` §3 states the owner of every
`Ref2` pair a compiled slot can hold; the S3/S4/S9 axioms of
`jit_boundary.md` are the mechanism and its validator; the
kernels' write-epoch bookkeeping (`ran`, `epoch`) decides only
whether a step has run since the last write, and never what it
holds.

---

## 5. State layering at runtime

The substrate's L-axioms state how state is divided between
layers, where a layer is one scope instance: a root, a nested
scope at any depth, or a traversal activation. At runtime they
are implemented as follows:

| L-axiom | Runtime realisation |
|---|---|
| **L1** (each layer owns its state) | One kernel per fiber, on whichever engine the host chose. The program is shared read-only through `KernelProgram` (an `Arc`); every kernel created from it owns its inputs, buffers, currency flags, and cells. No cross-fiber state sharing at the node tier. |
| **L2** (two-lifecycle classification bridges layers) | Effectively-const steps are evaluated once at scope-init; dynamic steps on demand after each write. The classification belongs to the program (§3), so it is the same on all four engines. |

At every evaluation, each layer's state is owned by its layer
(L1); every read of an outer scope's value goes through an
`extern` slot the compiler synthesised in the inner program for
that value and filled when the scope was initialised (S1+S2); and
every write to an outer scope goes through the single
write-through path of S5. Compilation establishes this layering,
and the runtime preserves it.

The S-axioms beyond S1 and S2 are implemented at runtime as
follows, alongside R1, R2, and R3:

| S-axiom | Runtime realisation |
|---|---|
| **S4** (external-write synthesis, open granularity) | External-write input slots are populated through the kernel's typed writes (`set_input`, `set_input_at`, `set_cursor`) at any granularity the producer chooses; provenance (R2) marks consumers not current on write; currency (R1) re-evaluates on next pull. Volatility (R1.v) is the explicit marker for wires whose value is not a function of declared inputs and so cannot be cached even between writes. |
| **S5** (compile-emit write-through, cross-tier path) | `SharedCell` write-through routes a writing node's output to a parent-tier cell at compile-emit time (per [subcontext_construction.md](subcontext_construction.md) §3.1, the shared write-through rewrite, and §5); at runtime the write fires as an ordinary node output, intercepted by the chain and published through the cell. The outer cell's slot is filled through the standard slot-filling contract. L1 still holds, because the outer cell remains the single owner of the value. |

### Shared cells on all four engines

A `shared` binding lets several kernels read and write one
value. On all four engines (the interpreter, the closure tier,
native, and pure native), a `shared` binding's input slot is
bound to a cell, and the cell holds the slot's value:
`set_input` on any kernel holding the cell publishes a new value
through it, and every holder's next read takes the cell's
value. `Kernel::shared_cells` lists a kernel's cells, and
`Kernel::attach_shared_cell` binds a `shared` binding to a cell
another kernel holds, so two kernels, on the same engine or on
different ones, read and write one value. A kernel created from
a shared program starts with a cell of its own per `shared`
binding, so sharing happens only when a host attaches a cell.

A second kind of cell, the broadcast cell of a computed output
(`Kernel::output_cell`), lets a descendant kernel read the value
each of this kernel's pulls publishes, rather than a copy taken
when the descendant was built
([cross_fiber_invalidation.md](cross_fiber_invalidation.md)
§3.1). Broadcast cells exist on the interpreter, the closure
tier, and native; pure native has none, and its `output_cell`
returns `None`.

The reading side follows R1: a step is current until an input
in its provenance changes, and a publication by another kernel
is such a change. The interpreter detects it at every memoized
read by comparing each cell's revision number with the last one
it saw. A compiled kernel compares every cell's revision number
at the first evaluation after a write and before each pull, and
marks the dependents of a slot whose cell changed not current. The protocol, its memory ordering, and
its costs are in
[cross_fiber_invalidation.md](cross_fiber_invalidation.md).

---

## 6. The `Kernel` surface in R-terms

A host drives a kernel through the `Kernel` trait without
knowing which engine it has. All four engines implement the
trait, and each call below has the same meaning on all four.

- **Writing inputs.** A host writes inputs to give the program
  new values to compute from. `set_inputs(coords)` writes the
  coordinates, the leading `u64` input slots, from a slice. `set_input(name, value)` writes one extern,
  named by its declared name; `set_input_at(index, value)` does
  the same by input index, for a host that resolves the name
  once with `input_index`. `set_cursor(name, partition)` writes
  the cursor named `name` to cover the given partition, which
  sets the cursor's `Ext` slot and its six scalar projections.
  A value must satisfy the slot's declared type; a value of
  another type is an error at the write. `None` clears a slot to
  unset. A write marks every step that depends on the written
  slots not current and changes nothing else; nothing runs until
  the next pull (S4, R2, R4).
- **`pull(name)`** computes and returns one output. `name` is
  the output's declared name. The kernel runs, in order, the
  not-current steps the output depends on, and returns the
  output's value as an owned `Value` that the caller keeps; a
  slot pair is never returned to a host. Where native code fuses
  several nodes into a unit that runs whole (a native or
  pure-native fusion unit, or one of the interpreter's native
  cones), running one of its steps runs the whole unit, so a
  pull may also run other nodes in the same unit (R1, R2).
  `pull_at(index)` is the same call by output index, for a host
  that resolves the name once with `output_index`.
- **`eval()`** brings every output up to date without returning
  any. The interpreter pulls every output; a compiled kernel
  runs every step that has not run since the last write. A host
  calls it when it wants every step evaluated without reading
  outputs, and what a side channel observes under `eval` is the
  same on all four engines.
- **`invalidate_all()`** forces the next pulls to recompute
  everything while keeping every input value. It marks every
  step not current, so every step, a side channel included, runs
  again at its next pull as if every input had been written. A
  host uses it to observe a nondeterministic program again
  without writing an input; a host that wants the inputs back at
  their defaults resets them separately.
- **`shared_cells()`** lists the cells this kernel's `shared`
  bindings are bound to, and **`attach_shared_cell(name, cell)`**
  binds the `shared` binding `name` to a cell another kernel
  holds, so the two kernels read and write one value (§5). A
  name that is not a `shared` binding is an error naming the
  ones that are.
- **`cursor_schemas()`** lists the cursors the program declares,
  with the partitions the compiler resolved where it could, so a
  host can decide how to partition work before writing cursors
  ([cursor_partitions.md](cursor_partitions.md) §7.2).
- **`traverse(index)`** starts one of the program's `for`
  traversals. `index` selects which: the top-level `for`
  statements, numbered from 0 in source order, which
  `traversals()` lists; an index with no traversal is an error.
  Starting a traversal copies the current value of every outer
  wire the body reads, so every run of the body sees those
  values even if this kernel changes later, and evaluates the
  header comprehension to produce the tuples the traversal
  visits. It returns a `TraversalStream`. To run the body for
  one tuple, the host asks the stream for an activation with
  `activation_on(i, engine)`: a new kernel on the named engine
  whose inputs are the tuple's element values plus the copied
  outer values, with every cursor declared `over` an element
  narrowed to that element's partition. The host drives the
  activation like any other kernel (R4;
  [for_traversal.md](for_traversal.md) §5.2).
- **`into_program()`** turns a kernel into a `KernelProgram`
  that can be shared across threads. Each thread then calls
  `create_kernel()` on the program to get a kernel of its own.
  A created kernel starts from the program, not from the kernel
  that became the program: every input at its declared default,
  every `shared` binding with a cell of its own, nothing current,
  and every reference pair pointing into its own storage.

---

## 7. Determinism — the D-axiom suite

The R-axioms describe the runtime mechanics. The D-axioms
describe the **determinism guarantees** a host can rely on as
consequences of those mechanics. There are four distinct,
named bounds.

### Axiom D1 — Typed Return Determinism

**Every declared output returns a value satisfying its
`PortType`. For a cone containing only `Purity::Pure` nodes,
fixed program, inputs, and immutable referenced state produce
the same value on repeated evaluations. A
`Purity::SideChannel` node preserves that typed-value rule and
must document whether its return is a pure function of inputs.
`Purity::Nondeterministic` explicitly opts its cone out of
value-determinism.**

D1 is the composition of R1 (currency), R3 (forward-only data
flow), T1+T2 (typed slot contract), and the substrate's L1
(per-kernel state ownership). The compiler's H3 (hoisting
preserves value) guarantees that construction does not break
it. The engines' equivalence contract
([engines.md](engines.md) §7) extends it across all four
engines: the same program, inputs, and pull sequence yield the
same values on each.

D1 separates type integrity, which is unconditional, from
value equality, which depends on the cone's declared purity.

### Axiom D2 — Side-Channel Determinism

**Impure constituent nodes' side channels (logging output,
file I/O, network calls, etc.) are deterministic
*conditional on each impure node's declared semantics*. A
node that writes "X" to stderr for input `i` produces an
identical stderr line for input `i` every time. A node
whose side channel depends on external state (e.g., wall
clock, process ID, network state) produces side effects
deterministic only modulo that external state.**

The substrate's slot contract bounds impurity: side effects
are additional observables outside the typed return. Side
channels are ordered only by actual node invocation order
inside one synchronous pull. Independent nodes, fibers, and
sinks have no combined total-order guarantee.

Every node exposes `PolydatNode::purity()`.
`Purity::SideChannel` names its sink; `Purity::Nondeterministic`
names the reason that input-only determinism does not hold. A
side-channel step is never fused into a native segment, so it fires under the same currency rule on all four
engines.

### Axiom D3 — Cost Determinism

**The number of steps that can perform work on a pull is
bounded by the reached cone size minus the currently-current
steps in that cone. Each not-current step is processed at most
once before becoming current. `None` short-circuiting may
avoid calling a node's `eval`; a fused segment may represent
several scalar nodes. Engine choice, cache state, and declared
node behavior therefore refine the structural upper bound
rather than changing the graph dependency bound.**

D3 follows from R1 (currency bounds processing to one per
not-current-to-current transition), R2 (no work for unreached
cones), and the structural cone size, which is computed from
provenance at build. The host can bound evaluation work from
the program's structure, the engine's plan (`Kernel::plan`),
and the current cache profile.

For example, a three-node cone performs work at no more than
three step visits on a cold kernel and no evaluation work on a
fully current one. `None` propagation can reduce actual `eval`
calls below that bound.

### Axiom D4 — Cross-fiber equivalence

**Two kernels produce equal pure-cone outputs when they use the
same immutable program and begin from equal complete state:
coordinate inputs, ordinary inputs, buffer contents, currency
flags, shared-cell values and revisions, and any other declared
dependency.**

Per-kernel state ownership prevents accidental coupling. D4
does not equate kernels that observe different shared-cell
snapshots, external state, nondeterministic nodes, or
side-channel interleavings. A by-reference output is compared
by the value `pull` returns, owned (R4), never by the slots
that name it.

---

## 8. The R-axioms and D-axioms compose

```text
       ┌──────────────────────────────────────┐
       │  R1 — currency memoization           │   ≤1 eval per change
       │  R2 — hybrid push/pull invalidation  │   push-dirty + pull-lazy
       │  R3 — forward-only flow              │   wire chain is the path
       │  R4 — outputs owned by provenance    │   storage per state, no reset
       └────────────────┬─────────────────────┘
                        │
                        ▼  yields
                        │
       ┌───────────────────────────┐
       │  D1 — typed return        │   typed; pure cones repeat
       │  D2 — side channels       │   conditional on metadata
       │  D3 — cost                │   structurally bounded
       │  D4 — cross-fiber         │   equal complete state
       └───────────────────────────┘
```

The R-axioms specify **how** the runtime evaluates, and the
D-axioms specify **what** a host can rely on as a consequence;
together they form the runtime contract. §7 states the
guarantees, and §3–§6 state the mechanism that provides them
(the R-axioms, state layering, and the `Kernel` surface).

---

## 9. Runtime boundaries

- Evaluation of a dependency cone is synchronous within one
  fiber. There is no intra-pull parallel evaluator in this
  runtime contract.
- Each deterministic step is either current or not current. A
  not-current step is evaluated by the next pull whose cone
  contains it and becomes current; a current step stays cached
  until an input in its provenance changes or the host calls
  `invalidate_all`. Volatile and
  nondeterministic steps follow R1.v instead.
- Side-channel ordering is the local invocation order defined
  by D2. Hosts requiring a global order must serialize or
  aggregate those effects outside the graph.
- A by-reference output is kept and returned by every pull
  until an input in its provenance changes (R4). A host holds
  the owned `Value` that `pull` returned, never a slot.

### 9.1 No state across pulls

A kernel keeps no value that depends on how many times a step
ran or was pulled. Every output is a function of its declared
inputs and its provenance (R4), so it replays from its
coordinates and agrees on all four engines and in every fiber.
A running count is therefore not a runtime facility. A shared
cell does not provide one either: it is typed and
first-writer-wins ([Scope Model](scope_model.md) §6.1), and it
publishes a value computed once rather than accumulating. A
`for` activation has no previous activation to read a count
from, because a traversal materializes its tuples when it
opens ([The for Construct](for_traversal.md) §3.1).

The quantities a counter is usually wanted for are pure
functions of data the host already holds:

- **The rank of a record among the matching records** is the
  number of matching ordinals smaller than the record's
  ordinal: `vec_count_below_i32(matching, ordinal)` (or the
  `_i64` form) over the list of matching ordinals, or
  `vec_position_i32` when that list is sorted and contains the
  ordinal.
- **The number of matching records** is the length of that
  list, `vec_len_i32(matching)`, or, over a dataset facet,
  `predicate_count_of(handle, value)` and
  `metadata_count_of(handle, value)`, which a scope computes
  once when it loads the facet.

Neither needs a visit order, and a visit order is what a
counter would add: polydat does not promise one.

An accumulation facility for a need that no pure formulation
covers is admissible only when all four of these hold:

1. Its value is a function of declared inputs and provenance,
   never of pull count or fiber schedule.
2. It replays: the same coordinates give the same value on all
   four engines and after `invalidate_all`.
3. It adds no special-cased wire and no per-cycle notion to the
   scope rules, as `cycle` is an ordinary coordinate.
4. It is a program transform or a node, not a runtime mode.

A fold over a materialized traversal, computed once at scope
init from the tuple set the traversal already holds, is the
shape most likely to meet all four.
