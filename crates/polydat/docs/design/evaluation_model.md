# Evaluation Model

The mechanism contract for Polydat evaluation: program/state split,
provenance-based invalidation, the two-lifecycle classification
(effectively-const vs dynamic), the const binding contract
(Plan A at build, Plan B at scope activation), the
non-deterministic node exclusion, input spaces, and
externally-written inputs. The axioms it gives mechanism to are
R1–R4 of [runtime_model.md](runtime_model.md), L2/S4/T1 of
[composition_substrate.md](composition_substrate.md), and G2/G5
of [polydat_grammar.md §18](polydat_grammar.md#sec-gaxioms); the engines that run the model are
[engines.md](engines.md).

The Polydat evaluation model separates the immutable program
(shared) from mutable per-thread state (private). This
allows ordinary per-thread evaluation without shared cache locks.
Explicit `SharedCell` inputs retain their own synchronization
contract. The split holds on every engine.

---

## Program / State Split

The split is the `KernelProgram` / `Kernel` pair of the kernel
API: a program is an `Arc<dyn KernelProgram>`, immutable and
shared across threads, and each thread creates its own
`Kernel` from it (`create_kernel`). A kernel becomes its
program again with `Kernel::into_program`; a kernel created
from a program starts from the program on every engine — every
input at its declared default, every `shared` binding with a
cell of its own, nothing current.

```
KernelProgram (Arc, immutable, shared across threads)
  interpreter realisation — PolydatProgram:
    ├── nodes[]          — node instances
    ├── wiring[]         — input source tables
    ├── input_defs[]     — typed names, defaults, and lifecycle kinds
    ├── output_map       — name → (node_idx, port_idx)
    └── provenance/dependents — exact multi-word masks and reverse lists
  compiled realisation — the closure plan or the hybrid segments
    over one slot layout, with the same provenance and dependents

Kernel (per thread, mutable, private)
  interpreter realisation — PolydatState:
    ├── core: EngineCore
    │   ├── buffers[][]        — per-node output value slots
    │   ├── node_clean[]       — per-node cache validity (bool)
    │   ├── inputs[]           — non-cell input registers
    │   ├── input_defaults[]   — reset values
    │   ├── shared_cells[]     — optional cell register per input
    │   └── cell-cone state
    ├── input_dependents[] — per-input transitive dependents
    └── nondeterministic_nodes[] — never cached
  compiled realisation — one u64 slot buffer, its None mask,
    the per-step current-ness of the provenance mode, and the
    scratch entries its steps publish pairs into
```

The interpreter's `PolydatProgram` is created once at compilation
time and shared via `Arc`; `PolydatState` is created per thread
via `program.create_state()`.

---

## Provenance-Based Invalidation

Each node has a compile-time exact multi-word **provenance mask**: bit i is set
if the node transitively depends on graph input i. On input
change, only nodes whose provenance overlaps the changed inputs
are invalidated. Nodes depending on unchanged inputs stay cached.

```
1. kernel.set_inputs(&[cycle])
   → write each coordinate input in the leading coordinate prefix
   → dirty every transitive dependent of each written coordinate
   → dirty every non-deterministic node

2. kernel.pull("user_id")
   → if the node is current → return the cached value
   → recursively evaluate dirty upstream nodes
   → gather inputs, evaluate, mark the node current
   → return the value
```

Nodes that do not depend on a written input stay current. The
interpreter treats the write itself as the invalidation signal
and does not compare rich `Value` instances for equality.

This is the one evaluation rule, and it holds on every engine:
a step is current until an input in its provenance changes; a
nondeterministic node is never current; compile-constant nodes
are folded once at build; `set_inputs` writes the coordinate
prefix and `pull` evaluates the named output's cone and no
more — on the
interpreter, the closure tier, and the hybrid kernel alike
(pure native code, being one function, evaluates the program).
The provenance mode a compiled engine is built with (`Raw`,
`Push`, `Pull`, `PushPull`, or the selector's `Auto`) changes
what is recomputed, never a value: it is an optimization.

**Diamond optimization:** In a diamond-shaped DAG where only one
input branch is written, the unchanged branch stays cached.

**Memoization granularity:** Every node's output buffer is cached.
This uniform rule permits a later pull of any downstream cone to
reuse every still-clean intermediate, including shared branches in
a diamond. The normative rule and its invariants are
[The Runtime Model §3–§4](runtime_model.md) (R1 and R2).

---

## Two Evaluation Lifecycles

A Polydat node's *lifecycle* is the granularity at which it is
re-evaluated. Two are recognised:

| Lifecycle | When evaluated | Re-evaluated when… |
|-----------|----------------|---------------------|
| **effectively-const** | Once, for the duration of a scope activation. Two implementation paths: (a) **compile-fold** — evaluated during the build and replaced with a leaf const node; (b) **scope-init pull** — evaluated once after parent materialization populates iteration-variable externs, then frozen for the activation. The choice between (a) and (b) is decided by the compiler based on the wire chain; the author writes `const NAME := <expr>` in both cases. | Never within an activation. The enclosing comprehension advancing to its next iteration ([comprehension_forms.md](comprehension_forms.md) §9.5) triggers a fresh activation, which re-runs scope-init pull (compile-folded leaves are immutable across activations). |
| **dynamic** | Once per pull, on demand at execution time | Whenever a transitively dependent input changes (provenance-based invalidation). Includes per-cycle pulls *and* intra-stanza recomputation when external-write inputs or `do_while`/`do_until` counters tick. |

The `const` modifier is the single author-facing surface for
effectively-const bindings. Compile-fold and scope-init pull are
implementation paths for the same semantic contract: materialize
once and freeze for the scope activation. Authors do not select
between the paths.

### Effectively-Const Nodes

A node is **effectively-const** at a given scope-init point if
it produces exactly one value for the entire activation of the
owning scope. The set is closed under upstream traversal: a node
whose every upstream wire reaches an effectively-const producer
is itself effectively-const.

| Producer | Effectively-const? | Why |
|----------|-------------------|-----|
| Literal in source | Yes | Resolved at parse / compile. |
| Compile-const fold result | Yes | Already a leaf const node. |
| Workload param (`const` binding) | Yes | Bound once at workload-kernel init, never reassigned. |
| `for` traversal element / `extern` with no default (iteration extern) | Yes — *for one activation* | Bound when the activation is created (`activation_on`) or the child is materialised; fixed for every coordinate of that activation. |
| `do_while` / `do_until` counter | **No** | Dynamic — ticks within the scope's own evaluation; not stable for the activation. |
| Graph input (e.g. `cycle`) | **No** | Dynamic — changes every cycle. |
| External-write input | **No** | Dynamic — written by the host between pulls. |
| Non-deterministic source (`counter`, `current_epoch_millis`, `elapsed_millis`, `thread_id`) | **No** | Excluded by construction even when wires would suggest otherwise. |

The iteration-extern row is the load-bearing case. The body of
`for profile in ..., table in ... { ... }` sees `profile` and
`table` as input slots; a purely data-flow view would flag any
binding downstream of those slots as dynamic and refuse to
fold it. But `profile` is bound exactly once per activation and
held fixed for every coordinate — the same stability guarantee
as a folded literal. Treating iteration externs as
effectively-const is what permits
`const prebuffered := dataset_prebuffer("{dataset}:{profile}")`
to be a legal const binding inside such a body.

### Compile-Time Constant Folding

Compile-time fold is the compile-fold implementation path for
the effectively-const lifecycle. It runs once per build, on
every engine, before the program is shared:

```
Phase 1: Classify each node — PolydatProgram::classify_lifecycle
  - Graph input / external-write input / non-deterministic source
                                  → dynamic
  - NodeOutput whose source is dynamic
                                  → dynamic (propagates)
  - Wire to an iteration extern (`for` element / `extern` with no default)
                                  → scope-init: not foldable at
                                    build. Extern values are unknown
                                    until scope activation; folding is
                                    deferred to the scope-init pull.
  - Everything else               → compile-constant

Phase 2: Evaluate the compile-constant nodes once

Phase 3: Replace evaluated nodes with leaf const nodes
         (ConstU64, ConstF64, ConstStr, ConstHandle, …)
```

On the interpreter the three phases are
`PolydatProgram::fold_init_constants_impl`. The closure tier and
the native engine build from the same classification: the
compile-constant nodes run once when the kernel is built and
their slots hold the folded values thereafter. The
`ConstantFolded` compile event is recorded for the same nodes
with the same values on every engine.

Type adapter nodes (`__u64_to_f64`, etc.) participate. A chain
like `ConstU64(42) → __u64_to_f64 → sin` folds to
`ConstF64(sin(42.0))` — the whole chain is evaluated once and
replaced with a single constant.

Folded constants are readable by name (`get_constant` on the
interpreter kernel; `Kernel::pull` on any engine) for
activity config resolution (cycles, concurrency from dataset
metadata).

### Scope-Init Pull

The scope-init pull is the scope-init-pull implementation path
for the effectively-const lifecycle. It runs once per scope
activation, *after* parent materialization has populated the
kernel's iteration-extern input slots and *before* the scope is
used.

```
For each const-modifier output b in this scope's program:
  Pull b's name on the activation kernel. The standard pull walks
  back through b's subgraph, evaluating each upstream node against
  the populated externs and caching the result in the kernel's
  per-node buffer (marked current).
```

Every later read of the binding, from any cycle and from every
kernel created from the activation, sees that one value; the
binding's eval does not re-fire. This is the runtime side of
the const-binding contract: one eval per scope activation,
regardless of how many cycles or threads traverse it.

---

## Const Binding Contract

`const <name> := <expr>` is the canonical surface for an
effectively-const binding: it asserts that `<expr>` evaluates
to a single value for the entire activation of the enclosing
scope. The compiler and runtime together enforce two checks:

### Compile-Time Check (Plan A)

During the build, after wire resolution and topological sort:

> For every binding declared `const`, every node in its upstream
> wire chain must be effectively-const (either compile-foldable
> or an iteration extern that materialise-wiring populates at
> scope activation).

If any upstream node is non-effectively-const — a graph input,
an external-write input, a `do_while`/`do_until` counter, a chain through
a non-deterministic source — compilation **fails** with a
diagnostic naming the const binding and the offending wire.
There is no soft fall-through to dynamic evaluation.

This check runs in the fold pass. Effectively-const
classification (above) and the const-binding check share the
same upstream walk (`classify_lifecycle`); the const check simply
demands the binding's node not be dynamic.

### Scope-Activation Pull (Plan B)

After the outer chain has populated the scope's input slots, the
materializer pulls every `const` output once, so that its value
is captured for the lifetime of the scope and every later read
of the binding sees that one value.

A pull that panics is caught, not swallowed: the materializer
prints one warning naming the binding and the panic text, and
leaves the binding's buffer at `Value::None`. The scope still
activates. The failure surfaces when the binding is consumed: a
read of the `None` buffer falls through to the wired-in input
where the conditional shadow allows it, and otherwise re-raises
the node's failure with the consumer's full context. The scope
does not refuse to start on a failed const pull, because a
`const` may depend on resolution that is not ready until the
workload runs (`dataset_prebuffer` and its kind), and a warning
at activation plus the failure in context at first use is the
diagnostic pair an operator can act on.

Plan A is the type-system-style check that runs at compile time
when iteration-extern values are unknown but the wire structure
is fully visible. Plan B is the one pull at scope activation when
the values are known and the fold pass has had its chance.
Together: a const binding either evaluates exactly once per scope
activation, or every use of it reports the one failure.

### Why Both Checks

Plan A alone catches structural errors at workload-author time
(no need to wait for runtime; failures travel with the source).
But it cannot catch runtime conditions — a remote facet that
returns 403, an opaque eval panic, a `Value::None` from an
otherwise-valid scope-init pull — because those depend on real
extern values.

Plan B alone is robust against runtime conditions but defers
clear structural errors (e.g. a const binding that wires through
a `cycle`-dependent node) to runtime, where the failure surface
is larger and the diagnostic less localized to the source line.

Both are cheap. Both run at most once per scope activation. The
pair is the contract.

### Diagnostic Format

Plan A fails the build with

```
init binding '<name>' violates the init contract: <offending>
(init bindings must be effectively-const at scope-init time per the init contract, evaluation_model.md))
```

where the message's `init` names the `const` modifier (the
keyword changed; the message did not), and `<offending>` names
the wire the fold found first:

- **`wire on node '<n>' reaches coordinate input '<name>'
  (dynamic; changes every cycle)`** — const binding wired to a
  graph input declared by `input ...: u64`.
- **`wire on node '<n>' reaches external-write port '<name>'
  (dynamic; mutated by op execution)`** — const binding
  wired to an `extern X: T = default` input (the polydat
  external-write surface; hosts use it for runtime injection
  patterns).
- **`wire on node '<n>' reaches non-deterministic source '<name>'
  (dynamic by construction)`** — `counter`, `current_epoch_millis`,
  `elapsed_millis`, `session_start_millis`, or `thread_id`.
- **`wire on node '<n>' reaches dynamic node '<upstream>'
  upstream`** — fallback when the chain is dynamic but the
  immediate seed isn't one of the patterns above (e.g. a chain
  through a `do_while` counter).

Plan B warns at scope activation with `warning: scope-init const
pull failed for '<name>': <panic text>`, from
`materialize_wiring_from_outer` step 3, and the node's own
failure message, enriched with the consumer's context, is what a
later read of the binding raises.

---

## Non-Deterministic Nodes

A node declared `Purity::Nondeterministic` — `counter`,
`current_epoch_millis`, `elapsed_millis`, `thread_id` and their
kind — and any node feeding a `volatile` output are excluded
from compile-fold *and* from effectively-const classification
regardless of their input wires, and the exclusion is contagious
downstream. They are inherently dynamic even when a static
analysis would suggest otherwise, and an engine never treats
them as current. A `const` binding that depends on one of these
fails the Plan A check.

*When* one reads, within a write, is the engine's: two such
wires are two steps on the interpreter and the closure tier and
one fused segment on the native tiers, so a change made between
two pulls of one write is visible to the second on the first
pair and not on the second. Every engine re-reads on the next
write, which is the guarantee; simultaneity within a write is
not. Two readings that must come from one instant belong in one
node returning both — see
[runtime_model.md](runtime_model.md) R1.v, "Read granularity".

---

## Input Spaces

Most workloads use a single `cycle` input. Multi-dimensional
inputs enable nested iteration:

```
input cycle: u64

// Mixed-radix decomposition: flat cycle → nested indices
row := mixed_radix(cycle, 1000, 0)     // cycle / 1000
col := mixed_radix(cycle, 1000, 1)     // cycle % 1000
```

The input space is defined inside Polydat, not in the activity
layer. This enables composition with other nodes and keeps the
executor simple (it just passes `[cycle]`).

---

## External-Write Inputs

An `extern name: type = default` declaration produces an
`InputDef` with `InputKind::ExternalWrite`. External producers
write these inputs through the typed `Dataflow::set_wire` API
(the interpreter kernel) or `Kernel::set_input` (any engine).
The boundary accepts `Value::None`, accepts a value that
satisfies the declared slot type, applies a registered boundary
adapter when one exists, and otherwise returns an error.

```
Producer writes to slot "user_name"
  → kernel.set_wire("user_name", value)?
  → input slot in the kernel holds the value

Consumer pulls a binding that reads {user_name}
  → standard input read → returns the value
```

External-write inputs persist across `set_inputs()` calls,
which update only the leading coordinate-input prefix. A host
resets non-coordinate inputs to their declared defaults at a
scope boundary with `PolydatState::reset_inputs_from`, normally
passing `program.coord_count()`. Cell-bound inputs are skipped
because their lifecycle belongs to the shared cell's owning
scope. Rebuilding or invalidating the whole state also restores
ordinary input defaults.

Hosts give external writes their own application-level names
(a host that flows one op's result into the next op's inputs
does so through external-write inputs, for example); the
polydat mechanism is generic external-input population.

---

## Compilation Levels

The host chooses an `Engine` (`compile_polydat_with`,
`PolydatAssembler::compile_with`): the interpreter, the closure
tier (P2), or native code where a node has a lowering and its
closure elsewhere (P3). `Engine::default()` — what every
engine-less entry point and the binary build — is P3 with the
`jit` feature and P2 without. Every engine evaluates under the
one rule of this document, computes the same values for the
same inputs, and refuses at construction, with a reason, a
program it cannot run; `Kernel::plan` reports what the engine
chose for the program. Per-node costs, eligibility, the
provenance-mode selector, and the JIT call-boundary contract
live in [engines.md](engines.md) and
[jit_boundary.md](jit_boundary.md).

This file covers what *evaluation* is —
program/state split, lifecycles, provenance — independent
of which engine runs it.
