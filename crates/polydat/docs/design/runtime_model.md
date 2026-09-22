# The Runtime Model — Polydat Design

**Subtitle:** Data Flow, Caching, Invalidation, and the
Determinism Suite.

The runtime contract of a polydat kernel on every engine: the
interpreter, the closure tier, and the native tier alike. The
R-axioms state the runtime mechanics (data flow, currency,
invalidation, output ownership); the D-axioms state the
determinism guarantees those mechanics deliver to a host. The
[Composition Substrate](composition_substrate.md) states the
static slot contract (S/T/L axioms) this mechanism realises;
the [Graph Compiler](graph_compiler.md) states the construction
passes (H/CF/NF axioms) that produce a program; the
[Expression Engine](expression_engine.md) cites D1/D2/D3 as its
bounded determinism; the [Polydat Grammar](polydat_grammar.md) supplies
the language-level commitments (G4 port-typed expressions
underwrites D1; G5 structural lifecycle classification
underwrites R1 and D3). Cross-fiber concerns are the host's:
the D-axioms hold per fiber, each fiber holding its own
kernel.

The forcing question: **given a compiled program and a kernel
over it, how do values flow, what does the kernel keep current,
how does a write invalidate, and what determinism does the
composition of those mechanics deliver?** This doc says: data
flows along declared wires alone (R3); a step stays current
until an input in its provenance changes (R1); invalidation is
push on the write and pull on the read (R2); every output is
owned by its provenance (R4); and the determinism the
runtime delivers has four named bounds (D1 through D4).

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

This is what the substrate calls "data linearisation embedded
in graph structure" — the graph IS the linearisation. There is
no separate execution-order plan overlaying it. A compiled
engine's step order is a topological order of the same graph;
it is derived from the wiring, never declared beside it.

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
the program is constructed, and every engine's invalidation
plan is a projection of it: the interpreter inverts it into
per-input dependent lists; a compiled kernel inverts it into
per-input dependent step lists and per-output cone orders.

**The host can ask of any output: which inputs is this a
function of?** The answer is exact, computed at build, constant
across evaluations. There is no runtime discovery of
dependencies; everything is structural.

---

## 3. Currency — the one evaluation rule

Every engine evaluates under one rule:

> A step is current until an input in its provenance changes.
> A nondeterministic step is never current. A step no input
> reaches is compile-constant and is folded at build.
> Everything else runs at first pull after it stopped being
> current.

A *step* is a node on the interpreter and a compiled node or
fused segment on a compiled engine. The rule is what a host may
rely on; the bookkeeping that realises it differs by engine and
never changes a result:

| Engine | Realisation |
|---|---|
| Interpreter | A clean flag per node (`node_clean`); per-input dependent lists (`input_dependents`) cleared on every write; a list of nondeterministic nodes cleared on every write; a cell-revision check at every memoized read (§5). A write is itself the change: the interpreter does not compare the new value with the old, so a same-value rewrite re-runs the dependents, which a side channel in the cone must observe. |
| Closure tier and native tier | An `Invalidation` plan derived from provenance (per-input dependent steps and per-output cone orders); a clean flag per step; a round number per step recording the evaluation round it last ran in, bookkeeping that wipes nothing; the volatile steps never current. A coordinate counts as changed only when its value differs from the one it replaces. |

The lifecycle classification the rule rests on has one
classifier, `PolydatProgram::classify_lifecycle`, shared by the
interpreter's fold and every compiled engine: a node is
compile-constant when no coordinate or external-write input
reaches it and nothing upstream is nondeterministic or
`volatile`; scope-init when only iteration externs reach it;
dynamic otherwise. The compile-constant fold therefore runs at
build on every engine, so what is knowable at build is known
at build and fails at build.

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

Why: without it a pull would cost the whole cone every time,
and a side channel in the cone would fire once per pull rather
than once per change. With it the cost of a pull is bounded by
what changed (D3), and a step's side effects are observed once
per event that made it not current (D2). The substrate's L1
(each layer owns its state) guarantees that currency is owned
by the kernel, so there is no cross-fiber cache contention.

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
engine's.** The paragraph above is per *step*, and R1 defines a
step as a node on the interpreter and a compiled node *or fused
segment* on a compiled engine. Two volatile wires are therefore
two steps on one engine and may be one on another, and that is
observable — it is the only way an engine's realisation shows
through, because a volatile step is the one step whose value is
not a function of anything the engine can see:

| Engine | Two volatile wires pulled in one write |
|---|---|
| Interpreter, closure tier | Two steps. Each reads when its own output is first pulled, so a change made between the two pulls is visible to the second. |
| Native tier, pure native | One fused segment, and on pure native one function is the whole program. Both read together at the first pull of either, so a change made between the pulls is not visible until the next write. |

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
agree: pure native compiles one function for the whole program,
so it has no smaller unit to read at. Pinned by
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
*signal* is push-side (a write proactively marks dependents);
the dirty *response* is pull-side (a step is re-evaluated only
when reached by a pull). The model has three named properties:

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
  upstream producers). Invalidation never crosses a scope
  boundary unguarded: S5's `SharedCell` write-through is the
  only legitimate cross-tier write surface, and a cell's
  revision is what carries the signal across kernels (§5).

### Axiom R2 — Hybrid push/pull invalidation

**Invalidation in polydat is a hybrid: a write to an input slot
marks not current every step whose provenance includes the
written input; subsequent pulls then lazily re-evaluate only the
not-current steps that the pulled output's cone reaches. Steps
not reached by any pull are never re-evaluated, regardless of
upstream writes.**

Why: the push half is what makes a pull cheap to decide (no
scan of the graph on read); the pull half is what makes an
unused output free. Either half alone gives one of the two
costs back. On every engine the two halves are the same plan
read from two sides: the interpreter walks the cone recursively
and stops at a clean node; a compiled kernel walks the output's
precomputed cone order and skips a step that is current.

### Axiom R3 — Forward-only data flow

**Data flow in a kernel evaluation is forward-only along
declared wires from input slots to output slots. Invalidation
never propagates backward; cross-tier writes are restricted to
the substrate's S5 SharedCell write-through mechanism; there is
no out-of-band data channel between nodes or between scopes.**

Why: the wire-chain structure is acyclic (the assembler rejects
a cycle as `AssemblyError::CycleDetected`), so a topological
order exists and every engine evaluates in one; S5 is the only
cross-tier write surface, and the parent-gated binder
([scope_model.md](scope_model.md) §4) is the only path that
binds a child's slots to an outer scope, so no construction can
introduce a backward or out-of-band channel.

### Axiom R4 — Outputs are owned by their provenance

**Every output, immediate or by reference, stands from the
run that produced it until an input in its provenance is
written. Nothing reclaims an output on any other occasion:
there is no evaluation round, generation, epoch, or thread
boundary with a meaning of its own in the provenance rules,
and no step is exempt from R1. Each state owns the storage
behind its outputs (the interpreter's `Value` buffers; a
compiled kernel's slot buffer and the scratch entries its
steps publish `Ref2` pairs into), and no storage belongs to a
thread. A read hands the reader an owned copy.**

Why: R1 is the whole caching contract, and its value is that
invalidation is per input. A reset that reclaimed every string
at every write would be an all-or-none invalidation the model
does not have, and a step that had to rerun to survive it would
be a hole in the model's caching. Storage owned by the state
whose step wrote it has the output's own lifetime for free (L1
at runtime, §5); storage owned by a thread has to have its
lifetime legislated, and any such rule contradicts this one.

Enforcement: `compiled_handles.md` §3 states the owner of every
`Ref2` pair a compiled slot can hold; the S3/S4/S9 axioms of
`jit_boundary.md` are the mechanism and its validator; the
kernels' write-epoch bookkeeping (`ran`, `epoch`) decides only
whether a step has run since the last write, and never what it
holds.

---

## 5. State layering at runtime

The substrate's L-axioms hold at runtime with these specific
realisations:

| L-axiom | Runtime realisation |
|---|---|
| **L1** (each layer owns its state) | One kernel per fiber, on whichever engine the host chose. The program is shared read-only through `KernelProgram` (an `Arc`); every kernel created from it owns its inputs, buffers, currency flags, and cells. No cross-fiber state sharing at the node tier. |
| **L2** (two-lifecycle classification bridges layers) | Effectively-const steps are evaluated once at scope-init; dynamic steps on demand after each write. The classification is the program's (§3), so it is the same on every engine. |

The runtime model is the *enactment* of the substrate's layered
state contract: at every evaluation, every layer's state is
owned by its layer (L1); every cross-tier read goes through
synthesised slots (S1+S2); every cross-tier write goes through
S5's chokepoint. The runtime mechanism preserves the layering
inherited from compilation.

The S-axis axioms beyond S1+S2 have their own runtime
realisations alongside R1/R2/R3:

| S-axiom | Runtime realisation |
|---|---|
| **S4** (external-write synthesis, open granularity) | External-write input slots are populated through the kernel's typed writes (`set_input`, `set_input_at`, `set_cursor`) at any granularity the producer chooses; provenance (R2) marks consumers not current on write; currency (R1) re-evaluates on next pull. Volatility (R1.v) is the explicit marker for wires whose value is not a function of declared inputs and so cannot be cached even between writes. |
| **S5** (compile-emit write-through, cross-tier path) | `SharedCell` write-through routes a writing node's output to a parent-tier cell at compile-emit time (per [subcontext_construction.md](subcontext_construction.md) §3.1, the shared write-through rewrite, and §5); at runtime the write fires as an ordinary node output, intercepted by the chain and published through the cell. The outer cell's slot is filled through the standard slot-filling contract; L1's layer-ownership guarantee holds because the outer cell remains the canonical state holder. |

### Shared cells on every engine

A `shared` binding's input slot is bound to a cell on every
engine, and the cell is the slot's register: `set_input` on any
holder publishes through it, and every holder's next read takes
the cell's value. `Kernel::shared_cells` lists a kernel's cells
and `Kernel::attach_shared_cell` binds a `shared` binding to a
cell another kernel holds, so two kernels, on the same engine
or different ones, read and write one register. A kernel
created from a shared program starts with a cell of its own per
`shared` binding; sharing is explicit.

The consumer side follows R1: a step is current until an input
in its provenance changes, and a cell publish is such a change
made by another kernel. The interpreter detects it at every
memoized read by comparing each cell's revision with the last
one it saw; a compiled kernel polls every cell's revision at
the first evaluation after a write and before each pull and marks the dependents of a
moved slot not current. The protocol, its memory ordering, and
its costs are in
[cross_fiber_invalidation.md](cross_fiber_invalidation.md).

---

## 6. The `Kernel` surface in R-terms

The `Kernel` trait is the surface a host drives an engine
through without knowing which one it has. Each call has one
meaning under the axioms above, on every engine:

- **The writes.** `set_inputs(coords)`
  writes the coordinate prefix; `set_input(name, value)` and
  its index-keyed form `set_input_at(index, value)` write one
  extern; `set_cursor(name, partition)` writes a cursor's
  `Ext` slot and its six scalar projections. Each is a typed
  write to declared slots (S4): a value of another type is an
  error at the write, and `None` clears a slot to unset. Each
  marks the written slots' dependents not current (R2) and
  changes nothing else (R4).
- **`pull(name)`** runs the not-current steps of one output's
  cone, in order, and returns the output's value, owned: a
  slot pair is never handed to a host. On the interpreter, the
  closure tier, and the hybrid kernel the cone is exactly the
  output's; pure native code, being one function, evaluates
  the program. `pull_at(index)` is the same by output index,
  with the name resolved once through `output_index`.
- **`eval()`** runs every step: the interpreter pulls every
  output, a compiled kernel runs every step that has not run
  since the last write. What a side channel observes under `eval` is
  the same on every engine.
- **`invalidate_all()`** makes nothing current and keeps every
  input where it is: every step, a side channel included, runs
  again at the next pull as if every input had been written. A host that
  wants the inputs back at their defaults resets them
  separately.
- **`shared_cells()` and `attach_shared_cell(name, cell)`** are
  the cell surface of §5.
- **`cursor_schemas()`** reports the cursors the program
  declares, with the partitions the compiler resolved where it
  could, on every kernel ([cursor_partitions.md](cursor_partitions.md)
  §7.2).
- **`traverse(index)`** opens a `for` traversal against the
  kernel's current values, on every engine: the cascaded wires
  are snapshotted through `pull` and `input_value`, and the
  comprehension is evaluated in the body's scope. An activation
  is a kernel of its own (R4) created from the body's program
  on the engine the host asks for
  (`TraversalStream::activation_on`), with the tuple, the
  cascade, and every cursor narrowing bound through this same
  surface ([for_traversal.md](for_traversal.md)).
- **`into_program()`** turns a kernel into a shared
  `KernelProgram`; `create_kernel()` on it yields a kernel for
  the calling thread that starts from the program: every input
  at its declared default, every `shared` binding with a cell
  of its own, nothing current, and every reference pair
  pointing into its own storage.

---

## 7. Determinism — the D-axiom suite

The R-axioms describe the runtime mechanics. The D-axioms
describe the **determinism guarantees** the runtime delivers as
consequences of those mechanics. Four distinct bounds, each
named.

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
preserves value) seals the property at the construction tier.
The engines' equivalence contract ([engines.md](engines.md) §7)
extends it across engines: the same program, inputs, and pull
sequence yield the same values on each.

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
side-channel step is never fused into a native segment, so it
fires under the same currency rule on every engine.

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

R-axioms describe **how** the runtime evaluates. D-axioms
describe **what guarantees** the host can rely on as a
consequence. Together they form the runtime contract:
mechanism plus guarantees, neither one alone sufficient.

A reader who wants to understand "what does polydat runtime
evaluation deliver?" reads §7 (D-axioms). A reader who wants to
understand "how does polydat make those guarantees real?" reads
§3–§6 (R-axioms, state layering, and the `Kernel` surface).

---

## 9. Runtime boundaries

- Evaluation of a dependency cone is synchronous within one
  fiber. There is no intra-pull parallel evaluator in this
  runtime contract.
- Cache warmup has two states per relevant step: not-current
  steps evaluate on the next reached pull and become current;
  current deterministic steps remain cached until invalidated.
  Volatile/nondeterministic steps follow R1.v instead.
- Side-channel ordering is the local invocation order defined
  by D2. Hosts requiring a global order must serialize or
  aggregate those effects outside the graph.
- A by-reference output stands until an input in its
  provenance is written (R4). A host holds the owned `Value`
  that `pull` returned, never a slot.
