---
type: specification
title: Evaluation Model
timestamp: 2026-09-25
description: The program/state split, provenance-based invalidation, the two evaluation lifecycles, the const binding contract, input spaces, and external-write inputs.
tags: [runtime, compiler]
---

# Evaluation Model

This document specifies how a polydat program is evaluated,
independently of which engine runs it: the split between the
immutable program and per-thread kernel state, provenance-based
invalidation, the two evaluation lifecycles (effectively-const
and dynamic), the const binding contract (a const is evaluated
at kernel initialization), the per-read evaluation of
nondeterministic nodes, input spaces, and externally written
inputs. It
provides the mechanism for axioms R1–R4 of
[runtime_model.md](runtime_model.md), L2, S4, and T1 of
[composition_substrate.md](composition_substrate.md), and G2 and
G5 of [polydat_grammar.md §18](polydat_grammar.md#sec-gaxioms).

**Related specifications:** [runtime_model.md](runtime_model.md)
(the normative runtime rules and the terms used here),
[composition_substrate.md](composition_substrate.md),
[engines.md](engines.md) (the engines that run this model), and
[jit_boundary.md](jit_boundary.md).

This document uses the terms of
[runtime_model.md](runtime_model.md) §Terms (program, kernel,
node, wire, input, write, change, pull, cone, provenance, step,
current, read, volatile, fulcrum, initialization) and these:

- **Engine.** One of the four ways a program runs: the
  interpreter (P1), the closure tier (P2), the native tier (P3,
  native segments and closure steps in one hybrid kernel), and
  pure native. The last three are the *compiled engines*.
- **Cycle.** One coordinate write (`set_inputs`) and the pulls
  that follow it. The conventional coordinate input is named
  `cycle`.
- **Scope activation.** One instance of a scope's kernel, bound
  to its outer scope's values: a child materialised by
  parent-gated construction, or a `for` traversal activation.
  Scope-init is the start of an activation, and ends with the
  activation kernel's initialization.

The program is immutable and shared; each thread's kernel state
is mutable and private. Threads therefore evaluate without
locking a shared cache, and only explicit `SharedCell` inputs
are shared, under their own synchronisation contract. The split
holds on all four engines.

---

## Program / State Split

The split is the `KernelProgram` / `Kernel` pair of the kernel
API. A program is an `Arc<dyn KernelProgram>`, immutable and
shared across threads, and each thread creates its own `Kernel`
from it with `create_kernel`. `Kernel::into_program` turns a
kernel back into its program. On all four engines, a kernel
created from a program starts from the program's initial state:
every input at its declared default, every `shared` binding
with a cell of its own, every `const` evaluated from those
defaults, and no other step current.

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
    └── nondeterministic_nodes[] — marked not current at every read
  compiled realisation — one u64 slot buffer, its None mask,
    the per-step current-ness of the provenance mode, and the
    scratch entries its steps publish pairs into
```

On the interpreter, the `PolydatProgram` is created once at
compilation and shared through an `Arc`, and a `PolydatState`
is created per thread with `program.create_state()`.

---

## Provenance-Based Invalidation

Each node has an exact multi-word **provenance mask**, computed
at compile time: bit i is set if the node transitively depends
on graph input i. When inputs change, only the nodes whose
provenance includes a changed input are marked not current.
Nodes that depend only on unchanged inputs keep their cached
values.

```
1. kernel.set_inputs(&[cycle])
   → write each coordinate input in the leading coordinate prefix
   → dirty every transitive dependent of each written coordinate

2. kernel.pull("user_id")
   → dirty every volatile node (every read does this)
   → if the node is current → return the cached value
   → recursively evaluate dirty upstream nodes
   → gather inputs, evaluate, mark the node current
   → return the value
```

In this sequence, "dirty" means not current. The interpreter
treats the write itself as the invalidation signal and does not
compare rich `Value` instances for equality.

The following diagram shows the same sequence for a write to
one input and a pull of one output.

![A write to one input marks only the steps whose provenance includes that input not current; a later pull of an output first marks the volatile steps not current, then runs only the not-current steps in that output's cone and returns cached values for the rest](../diagrams/evaluation_model-write-pull.png)

The evaluation rule is the same on all four engines
(interpreter, closure tier, native, and pure native):

- A step is current until an input in its provenance changes.
- A volatile step runs again at every read whose cone reaches
  it, and at most once within one read.
- Compile-constant nodes are folded once at build.
- A `const` is evaluated once, at initialization.
- `set_inputs` writes the coordinate prefix.
- `pull` evaluates the named output's cone and nothing else.

A compiled engine is built with a provenance mode (`Raw`,
`Push`, `Pull`, `PushPull`, or the selector's `Auto`); the
closure tier offers all four named modes, native offers `Raw`,
`Pull`, and `PushPull`, and pure native offers `Raw` and
`PushPull`. The mode changes which steps are recomputed, never
a value: it is an optimization.

**Diamond optimization.** In a diamond-shaped DAG where only one
input branch is written, the unchanged branch keeps its cached
value.

**Memoization granularity.** Every node's output buffer is
cached. Because the rule is uniform, a later pull of any
downstream cone reuses every intermediate that is still
current, including shared branches in a diamond. The normative
rule and its invariants are R1 and R2 of
[The Runtime Model §3–§4](runtime_model.md).

---

## Two Evaluation Lifecycles

A node's *lifecycle* is how often it is re-evaluated. There are
two:

| Lifecycle | When evaluated | Re-evaluated when… |
|-----------|----------------|---------------------|
| **effectively-const** | Once for the life of a kernel. Two implementation paths: (a) **compile-fold**, for a node with no input in its provenance and for a `const` whose right-hand side is a literal, which is evaluated during the build and replaced with a leaf const node; (b) **initialization**, for every other `const`, which is evaluated when the kernel is initialized, after the binder has written the enclosing scope's values, and held in the kernel's const slot. The author writes `const NAME := <expr>` in both cases. | Never within a kernel's life, except by `Kernel::init`. A new activation (the enclosing comprehension advancing to its next iteration, [comprehension_forms.md](comprehension_forms.md) §9.5, or the next `for` tuple) is a new kernel, which is initialized; compile-folded leaves are the same in every kernel. |
| **dynamic** | Once per pull, on demand at execution time | Whenever a transitively dependent input changes (provenance-based invalidation). Includes per-cycle pulls *and* intra-stanza recomputation when external-write inputs or `do_while`/`do_until` counters tick. A volatile node is dynamic and runs again at every read (see Non-Deterministic Nodes). |

The `const` modifier is the only author-facing way to declare an
effectively-const binding. Compile-fold and initialization are
two implementations of the same contract: evaluate once and keep
the value for the kernel's life. The compiler chooses between
them from the right-hand side; the author does not.

### Effectively-Const Nodes

A node is **effectively-const** at a given scope-init if it
produces exactly one value for the entire activation of its
scope. The set is closed under upstream traversal: a node whose
every upstream wire comes from an effectively-const producer is
itself effectively-const.

| Producer | Effectively-const? | Why |
|----------|-------------------|-----|
| Literal in source | Yes | Resolved at parse / compile. |
| Compile-const fold result | Yes | Already a leaf const node. |
| `const` binding | Yes | Evaluated once at kernel initialization and held in a const slot that only initialization writes. |
| `for` traversal element / `extern` with no default (iteration extern) | Yes — *for one activation* | Bound when the activation is created (`activation_on`) or the child is materialised; fixed for every coordinate of that activation. |
| `do_while` / `do_until` counter | **No** | Dynamic — ticks within the scope's own evaluation; not stable for the activation. |
| Graph input (e.g. `cycle`) | **No** | Dynamic — changes every cycle. |
| External-write input | **No** | Dynamic — written by the host between pulls. |
| Non-deterministic source (`counter`, `current_epoch_millis`, `thread_id`) | **No** | Excluded by construction even when wires would suggest otherwise. A `const` over one captures its value at initialization. |

The iteration-extern row is the case that needs explanation.
The body of `for profile in ..., table in ... { ... }` sees
`profile` and `table` as input slots, and a purely data-flow
view would classify every binding downstream of those slots as
dynamic and refuse to fold it. But `profile` is bound exactly
once per activation and held fixed for every coordinate, which
is the same stability guarantee a folded literal has. Because
iteration externs are effectively-const,
`const prebuffered := dataset_prebuffer("{dataset}:{profile}")`
is a legal const binding inside such a body.

### Compile-Time Constant Folding

Compile-time folding is the compile-fold path of the
effectively-const lifecycle. It runs once per build, on all four
engines, before the program is shared:

```
Phase 1: Classify each node — PolydatProgram::classify_lifecycle
  - Graph input / external-write input / non-deterministic source
                                  → dynamic
  - NodeOutput whose source is dynamic
                                  → dynamic (propagates)
  - Wire to an iteration extern (`for` element / `extern` with no default)
    or to a const slot (`__const_<name>`)
                                  → scope-init: not foldable at
                                    build. These values are unknown
                                    until the kernel is bound and
                                    initialized.
  - Everything else               → compile-constant

Phase 2: Evaluate the compile-constant nodes once

Phase 3: Replace evaluated nodes with leaf const nodes
         (ConstU64, ConstF64, ConstStr, ConstHandle, …)
```

On the interpreter the three phases are
`PolydatProgram::fold_init_constants_impl`. The three compiled
engines use the same classification: the compile-constant nodes
run once when the kernel is built, and their slots hold the
folded values from then on. The `ConstantFolded` compile event
is recorded for the same nodes with the same values on all four
engines.

Type adapter nodes (`__u64_to_f64`, etc.) are folded like any
other node. A chain such as `ConstU64(42) → __u64_to_f64 → sin`
folds to `ConstF64(sin(42.0))`: the whole chain is evaluated
once and replaced with a single constant.

Folded constants can be read by name (`get_constant` on the
interpreter kernel, or `Kernel::pull` on all four engines), for
example to resolve activity configuration such as cycles or
concurrency from dataset metadata.

### Scope-Init Pull

The scope-init path of the effectively-const lifecycle is
initialization (`Kernel::init`). It runs once for every kernel
that comes into existence, after the binder has filled the
kernel's iteration-extern and fallback input slots, and before
the kernel is handed to its user.

```
For each ConstInit c in the kernel's const_inits(), in order
(a const after the consts it reads):
  1. Pull c.source (the output __init_<name>). The standard pull
     evaluates the const's expression against the bound inputs.
  2. If the value is None and c.fallback names an input, take
     that input's value instead (the conditional shadow).
  3. Write the value to c.slot (__const_<name>) through
     init_input_at, and pull c.name so its output is current.
```

Every later read of the binding, from any cycle, returns that
one value, and the binding's expression does not run again. A
kernel forked from an initialized kernel (`fork`) copies the
consts and is not initialized again. This is the runtime side of
the const-binding contract: one evaluation per initialization,
however many cycles read it.

---

## Const Binding Contract

`const <name> := <expr>` declares a binding that is evaluated
at kernel initialization: `<expr>` is evaluated once when the
kernel is initialized, from the kernel's inputs as they are at
that moment, and the value is fixed for the kernel's life.
Nothing re-evaluates it, and no step runs again because of it,
until `Kernel::init` initializes the kernel again. The contract
holds on all four engines (the interpreter, the closure tier,
native, and pure native).

### Compilation of a const

A const whose right-hand side is a literal (a number, a string,
`true` or `false`, a negated or cast literal, or a list of
literals) folds at build and needs no initialization. Every
other const is compiled as a program transform:

- its expression becomes the output `__init_<name>`;
- its value lives in the input slot `__const_<name>`, of kind
  `InputKind::Const`;
- `<name>` is a passthrough of that slot, so every reader of the
  const reads the captured value, and a step that reads it is
  scope-init in the lifecycle classification;
- a `ConstInit { name, slot, source, fallback }` record lists the
  const in the program's `const_inits()`, in dependency order: a
  const comes after every const it reads, directly or through
  plain bindings.

The compiler refuses two shapes, on all four engines:

- A const that reads a coordinate, directly or through plain
  bindings, fails the build with `const '<name>' reads the
  coordinate '<coord>': a const is evaluated once when the kernel
  is initialized, and a coordinate advances every cycle.` A
  coordinate changes every cycle, so no single value represents
  it.
- Consts that read each other in a cycle fail the build, since
  none can be evaluated first.

A const may read an extern, another const, an iteration extern,
or a volatile expression. An extern a const reads is read at
initialization: a later write to that extern does not change the
const until `Kernel::init` runs again. A const over a volatile
expression is a capture (see Non-Deterministic Nodes).

### Initialization failure

A const whose expression fails (its evaluation panics) makes
initialization fail, and the kernel is not handed out:
`Kernel::init` returns `KernelError::ConstInit { name, reason }`,
naming the const and the failure; the interpreter's build path
reports `AssemblyError::ConstInit`; and
`KernelProgram::create_kernel` panics with the same message. On
pure native a const whose value is `None` after initialization is
refused with `KernelError::Refused`, since pure native code
cannot hold `None`. A const whose expression is slow makes
initialization slow.

### Writing a const slot

Only initialization writes a const slot. `set_input` or
`set_input_at` naming `__const_<name>` is refused with
`WriteError::ConstSlot` on all four engines; `init_input_at` is
the write initialization uses.

### Conditional shadow and strict mode

A const whose right-hand side references a name also gets an
input slot of its own name, which the binder fills with the
enclosing scope's value. When the const's own value is `None` at
initialization, the const takes that fallback value, so a const
that yields nothing leaves the outer binding visible
([none_semantics.md](none_semantics.md), "Conditional-shadow
semantics for `const`"). Strict mode
([composition_substrate.md](composition_substrate.md) L2.f)
detects such a silent fall-through by reading `__init_<name>`,
the const's own value before the fallback.

---

## Non-Deterministic Nodes

A node declared `Purity::Nondeterministic` (`counter`,
`current_epoch_millis`, `thread_id`, and similar nodes), and any
node feeding a `volatile` output, is excluded from compile-fold
*and* from effectively-const classification regardless of its
input wires, and the exclusion propagates downstream. Such nodes
are volatile: always dynamic, even when a static analysis of
their wires would suggest otherwise.

Every read (every `pull` and every `eval`) whose cone reaches a
volatile node runs it again, whether or not an input was written
since the last read, and runs it at most once within that read.
The most upstream volatile node on a path is the fulcrum. The
fulcrum and everything downstream of it run on every read that
reaches them; everything upstream of the fulcrum keeps ordinary
provenance currency and is cached. No native code unit (a fusion
unit of native or pure native, or an interpreter native cone)
contains both a volatile node and a non-volatile node, so the
upstream steps stay cached on native code too. Two volatile reads
therefore behave the same on all four engines (the interpreter,
the closure tier, native, and pure native). Two readings that
must come from one instant belong in one node that returns both;
see [runtime_model.md](runtime_model.md) R1.v.

A `const` over a volatile expression is legal and is a capture:
the expression runs once, at initialization, and the const holds
that value for the kernel's life. Volatility stops at the
capture, so a step downstream of the const is not volatile
unless it has another volatile input. For example, a host that
wants one clock origin for a whole session declares
`const session_start := current_epoch_millis()` in its root
scope, and a child scope reads it through
`extern session_start: u64`; polydat ships no session-timestamp
node.

The library's `is_stable` node is pure: it takes a window of
samples (`vec_f64`), a margin, and a minimum sample count, and
returns the settled value and whether the window is stable. The
host keeps the window, for example as a JSON array it converts
with `str_to_vec_f64`.

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

The input space is defined inside polydat, not in the activity
layer. It can therefore be composed with other nodes, and the
executor only passes `[cycle]`.

---

## External-Write Inputs

An `extern name: type = default` declaration produces an
`InputDef` with `InputKind::ExternalWrite`. External producers
write these inputs through the deprecated `Dataflow::set_wire`
API (on the interpreter kernel) or through `Kernel::set_input`
(on all four engines). The boundary accepts `Value::None`,
accepts a value that satisfies the declared slot type, applies
a registered boundary adapter when one exists, and otherwise
returns an error.

```
Producer writes to slot "user_name"
  → kernel.set_wire("user_name", value)?
  → input slot in the kernel holds the value

Consumer pulls a binding that reads {user_name}
  → standard input read → returns the value
```

External-write inputs keep their values across `set_inputs()`
calls, which write only the leading coordinate-input prefix. A
host resets the non-coordinate inputs to their declared
defaults at a scope boundary with
`PolydatState::reset_inputs_from`, normally passing
`program.coord_count()`. Cell-bound inputs are skipped, because
their lifecycle is controlled by the scope that owns the shared
cell. Rebuilding or invalidating the whole state also restores
ordinary inputs to their defaults.

Hosts give external writes their own application-level names;
for example, a host that passes one operation's result into the
next operation's inputs does so through external-write inputs.
The polydat mechanism is generic population of external inputs.

---

## Compilation Levels

The host chooses an `Engine` (`compile_polydat_with`,
`PolydatAssembler::compile_with`): the interpreter; the closure
tier (P2); native code (P3), which runs a node natively where it
has a lowering and as a closure elsewhere; or pure native (with
the `jit` feature), which compiles the whole program to native
code. `Engine::default()`, which every engine-less entry point
and the binary use, is P3 with the `jit` feature and P2 without
it. All four engines evaluate under the rule of this document
and compute the same values for the same inputs; each refuses
at construction, with a reason, a program it cannot run; and
`Kernel::plan` reports what the engine chose for the program.
Per-node costs, eligibility, the provenance-mode selector, and
the JIT call-boundary contract are specified in
[engines.md](engines.md) and [jit_boundary.md](jit_boundary.md).
