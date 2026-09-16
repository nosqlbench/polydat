# The Graph Compiler — Polydat Design

**Subtitle:** Hoisting and Graph Fusion (Context Fusion + Node
Fusion).

Formalises the scope-aware compiler that produces ready
kernels. Names hoisting as the lifecycle analysis, Graph Fusion
as the two-phase accommodation pipeline, and the axioms each
preserves over the
[composition substrate](composition_substrate.md). The
[Runtime Model](runtime_model.md) owns what the compiled
program *does* at runtime; the
[Evaluation Model](evaluation_model.md) owns the two-lifecycle
classification that hoisting applies and the const-binding
contract it enforces.

The forcing question: **a polydat workload declares its
intent — clauses, bindings, ops — but never declares the
machinery (which values live where, which subgraphs collapse,
which inputs are constant for which scope's lifetime). The
compiler infers all of it. What is the compiler doing, and
what invariants does it preserve?** This doc says: the
compiler performs *hoisting* (lifecycle analysis that
identifies which computation moves to which scope layer) and
*Graph Fusion* in two phases (Context Fusion at scope-init,
Node Fusion at compile time). Together they produce a
ready-kernel that honours the substrate's slot contract.

---

## 1. The claim

The compiler is **scope-aware**: it does not treat a kernel
as a closed compilation unit but as a participant in a chain
of nested scopes, each with its own lifecycle classification
and synthesis surface. The compiler's load-bearing
construction work is three coordinated mechanisms:

```text
                  AUTHORED GRAPH
                        │
                        ▼
                ┌──────────────────┐
                │   Node Fusion    │   compile-time graph
                │   (Phase 2 GF)   │   rewriting
                └────────┬─────────┘
                         │
                         ▼
                ┌──────────────────┐
                │     Hoisting     │   compile-time lifecycle
                │     (analysis)   │   classification
                └────────┬─────────┘
                         │
                         ▼
                ┌──────────────────┐
                │   kernel on the  │   the program is built
                │   chosen Engine  │   and ready to instantiate
                └────────┬─────────┘
                         │
                         │   (per scope-init)
                         ▼
                ┌──────────────────┐
                │  Context Fusion  │   scope-init-time
                │  (Phase 1 GF)    │   slot filling
                └────────┬─────────┘
                         │
                         ▼
                  READY KERNEL
                  (slot contract held)
```

**Hoisting** is the lifecycle analysis: which input is
effectively-const for which scope, what computation can be
folded once at build or at scope-init versus per-cycle.

**Graph Fusion Phase 2 (Node Fusion)** is compile-time graph
rewriting: type-coercion edge adapters inserted at wire
resolution, and fused-node substitution for recognised
subgraphs.

**Graph Fusion Phase 1 (Context Fusion)** is scope-init-time
slot filling: extern slots get bound from the outer scope,
the chain materialises the synthesis surface auto-extern
declared.

Together they form the *construction pipeline*. The substrate
gives them the typed slot contract to preserve; the compiler
preserves it through every pass.

---

## 2. Pipeline overview

The pipeline, in the order the code runs it. Everything up to
the resolved graph is engine-neutral: it runs once, the same
way, whichever engine the host names; the engine then decides
what runs the resolved graph.

| Pass | Where | What it does |
|---|---|---|
| Parse | `polydat_grammar::lexer` / `parser` (re-exported as `dsl::lexer`, `dsl::parser`) | Source to AST. |
| Prologue | `dsl::compile` (`Prepared`) | One prologue for every entry point: the compile options, the required outputs, the pragmas, the data-file base directory. |
| Bind | `dsl::compile::assemble_parent` | One lowering for every entry point: bindings become assembler nodes; every referenced name not defined locally becomes an input slot (the conditional-shadow rule of [none_semantics.md](none_semantics.md)); tiles are typed. `for` statements are lifted out first and their bodies compiled as traversals. |
| Wire resolution + adapter insertion | `compile::assembly::resolve_with_log` | Arity is checked; each wire's producer type is compared with the consumer's declared input type; a mismatch heals through `auto_adapter` (the adapter node is inserted and `TypeAdapterInserted` logged) or fails as `AssemblyError::TypeMismatch`; strict mode refuses the implicit coercion instead. |
| Strict-wire assertions | same pass | Under `strict_values`, an `AssertValue` node is spliced in front of every constrained sink port whose source is not already proven (`AssertionInserted` / `AssertionSkipped`). |
| Node Fusion | same pass, `compile::fusion::apply_fusions` | `fusion::default_rules` (every rule the linked node crates register, in priority order) applied to a fixpoint; `FusionApplied` logged. |
| Dead-code elimination | same pass | Nodes not reachable from a declared output are dropped; the side-channel `log_*` nodes are pinned alive. |
| Topological sort | same pass | Kahn's algorithm over the live nodes; a cycle is `CycleDetected`. |
| Round-trip lint | same pass, `compile::roundtrip_lint` | A value modulated `T → Y → … → T` through pure conversion machinery is a warning, and an error under `strict_values`. |
| → `ResolvedDag` | | The engine-neutral product: nodes in topological order, wiring, input definitions, output map. |
| Hoisting | `PolydatProgram::classify_lifecycle` | Every node compile-constant, scope-init, or dynamic (§3); run by the engine build. |
| Engine build | `PolydatAssembler::compile_with(Engine)` | Interpreter: native cone extraction, then the fold (`fold_init_constants_impl`). Closure tier: the closure plan (`build_p2_layout`). Native: the hybrid kernel's segments. The strict refusals (`refuse_strict`) and the compile-constant fold happen on every engine, and `ConstantFolded` is logged on every engine. |
| Context Fusion | `materialize_subscope` | Slot synthesis at scope-init (§4). |

---

## 3. Hoisting

### 3.0 What "hoisting" means in polydat

In compiler theory, *hoisting* often means cross-scope code
motion — moving an inner-loop-invariant computation to
before the loop. In polydat, hoisting has a more specific
meaning:

> **Hoisting is the per-node lifecycle classification within
> a single kernel's program, used to partition the program's
> execution into code paths built into the same program: a
> compile-constant fold that runs once at build, a scope-init
> evaluation that runs once when the kernel materialises, and
> a per-cycle dispatch that runs each `set_inputs` advance.**

It is within-kernel partitioning, not cross-kernel code
motion. Kernels are optimization boundaries: computations
are not moved between parent and child kernels. Values that
cross that boundary use the scope materialization protocol.

This narrower meaning is what makes the pipeline ordering
unambiguous: hoisting's classification is a property of the
*final* wire chain (which inputs each node reaches through
which computed nodes); Node Fusion changes wire structure;
therefore Node Fusion precedes Hoisting. The classification
sees the post-fusion graph.

### 3.1 The hoisting algebra

Per the [Evaluation Model](evaluation_model.md), every node
in a kernel has one of two lifecycles:

- **Effectively-const** — resolved once, at build (compile-
  constant) or at scope-init, frozen for the scope's
  lifetime.
- **Dynamic** — resolved per pull at cycle time.

Hoisting analysis walks the kernel's wire chain and classifies
every node by lifecycle. The classification is structural:

```text
classify(node) =
    seed from the inputs the node reads directly:
      a Coordinate input slot:        Dynamic
      an ExternalWrite input slot:    Dynamic
      an IterationExtern input slot:  ScopeInit
      no input at all:                CompileConst
    a node declared Purity::Nondeterministic, or feeding a
    `volatile` output:                Dynamic (and nondeterministic)
    then the join with every upstream node:
      the maximum of its own seed and its producers'
      (Dynamic > ScopeInit > CompileConst); nondeterminism
      is contagious downstream.
```

A computed node's classification is the *join* of its
upstream nodes' classifications. This is monotonic and
bottom-up; the propagation is a fixpoint over the wiring.

The classification is `PolydatProgram::classify_lifecycle`,
the one rule shared by the interpreter's fold, the strict
refusal, the closure plan, and the hybrid builder, so every
engine folds, defers, and recomputes the same nodes. (The
interpreter's native-cone extraction re-derives the same
three classes to keep cones inside per-cycle work.)

### 3.2 The hoisting boundary

The hoisting boundary in a kernel is the set of wires that
mark "everything upstream is Effectively-const; everything
downstream is Dynamic." The build emits the evaluation
codepaths accordingly:

- **Compile-constant path**: evaluate every compile-constant
  node once at build and replace it with a constant.
- **Scope-init path**: evaluate every scope-init node once
  at scope-init, store in slot or buffer.
- **Per-cycle path**: evaluate every Dynamic node per
  `set_inputs` advance, reading Effectively-const upstream
  values from their folded constants or pre-evaluated
  buffers.

The boundary is *structural*, not declared. A node author
does not say "I'm hoistable"; the analysis derives it from
the upstream wire chain.

### 3.3 Hoisting eligibility — what moves to scope-init

A computed node is hoisting-eligible iff its full upstream
cone reaches *only* Effectively-const inputs. Any Dynamic
upstream pins the node to the per-cycle path.

This is a property of the wire chain, not of the node.
Concrete consequences:

- A `hash(k)` node where `k` is a `for` traversal element is
  hoistable — `k` is `IterationExtern` (Effectively-const for
  the scope's lifetime).
- A `hash(cycle)` node is *not* hoistable — `cycle` is a
  `Coordinate` (Dynamic).
- A `const X := <expr>` binding where the RHS uses only
  iter-vars and other consts is hoistable — and the
  const-binding contract's compile-time check (Plan A)
  catches any violation.
- A `const X := hash(cycle)` binding is *rejected* by Plan A
  — the const surface cannot host a Dynamic upstream.

### 3.4 The Axiom suite — H-axioms

The substrate's axioms (S1–S5, T1–T3, L1–L2) are general
guarantees about slots and layers; hoisting adds specific
guarantees about the analysis itself.

#### Axiom H1 — Classification is total

**Every node in a well-formed graph has exactly one
lifecycle classification (compile-constant, scope-init, or
dynamic). The classification function is total — there is no
"unknown" or "deferred" state.**

Enforcement: the structural walk in §3.1 is exhaustive; an
unresolved wire fails at construction (`UnknownWire`); strict
mode refuses what lax compilation only warns about.

#### Axiom H2 — Classification is monotonic and stable under composition

**Node `n`'s classification is a monotonic function of its
upstream cone's classifications: if any upstream is Dynamic,
`n` is Dynamic; otherwise `n` is Effectively-const. The
classification is preserved under fan-in composition (joining
upstream cones) and under fan-out (multiple consumers of
`n`).**

Enforcement: the join rule in §3.1. Adding an upstream wire
can change `n` from Effectively-const to Dynamic but never
the reverse; this monotonicity is what makes the analysis
stable under graph rewrites (Node Fusion, §5).

#### Axiom H3 — Hoisting boundary is preservation-of-determinism

**A node `n` moved from the per-cycle path to the scope-init
or compile-constant path produces the same value at
evaluation time as it would have produced per-cycle, for
every cycle. That is, hoisting does not change the semantic
value of any wire; it only changes *when* the value is
computed.**

Enforcement: H1's totality + S3 (the typed writes advance
coordinates only) + L2 (lifecycle bridging). The
Effectively-const lifecycle is precisely the property that
"value doesn't change after scope-init"; hoisting only
applies the lifecycle's promise.

---

## 4. Graph Fusion Phase 1 — Context Fusion

Context Fusion is the scope-init-time fulfilment of the
substrate's Synthesis pillar (S1, S2). When a scope
materialises, its declared extern slots are populated from
the outer scope's bindings; its Effectively-const buffer is
evaluated; its dispatch state is initialised.

### 4.1 The synthesis surface

Per S1, the compiler discovers extern slots via auto-extern.
The set of discovered externs is the **synthesis surface** —
the precise set of slot names + types that the chain must
fill from outer state at scope-init.

The synthesis surface is encoded in the kernel's
`input_defs`: `IterationExtern` and relevant `ExternalWrite`
entries are the non-coordinate slots filled by the applicable
construction or external-write boundary.

### 4.2 The synthesis act

Per S2, parent-gated construction is `materialize_subscope`:
it creates the child kernel from the shared program, seeds
the iteration bindings into their slots, and drives the
private `materialize_wiring_from_outer` pass against `outer`.
That pass, in order:

1. **Cell cascade.** Every cell visible at `outer` — the
   cells on its own input slots and its transit cells —
   attaches to the inner input slot of the same name. A cell
   whose name the inner scope declares `const` is dropped
   (transit suppression, [wire_materialization.md](wire_materialization.md));
   a cell with no matching inner slot is recorded on the inner
   kernel's transit list so a deeper descendant can pick it
   up.
2. **Outer outputs into inner slots.** For each output of
   `outer` that names an inner input slot not already bound
   to a cell, the first applicable form:
   - *value copy through `outer.lookup`* when the name is an
     input slot on `outer` (a passthrough extern) or a `const`
     output of `outer` — the two-tier read, so a `None` const
     falls through to the grandparent's value;
   - *cell attach* when `outer` publishes the output through a
     broadcast cell (a computed, per-cycle output);
   - *plain value copy* of `outer.lookup(name)` otherwise;
   - *the registered extern resolver*
     (`dsl::factories::register_extern_resolver`) when the
     outer chain has no binding at all.
   Every copied value passes through `adapt_boundary_value`,
   the boundary adapter catalog of
   [type_system.md](type_system.md) §6.2.
3. **Const pull.** Every `const` output is pulled once against
   the populated slots (the Evaluation Model's Plan B).
4. **Scope coordinates.** The inner scope's path becomes its
   own coordinates followed by `outer`'s.

After the walk, every declared extern slot in the child has a
value or a `SharedCell` handle, and transit cells are queued
for downstream materialisations.

(The output **manifest** — the typed contract a program
exposes to descendant synthesizers — is a separate
read-only summary produced by `kernel::extract_manifest`. It
is consumed by synthesizers *before* compiling an inner
program, to decide what auto-externs the inner program may
declare. It is not fired during synthesis; it informs the
shape the synthesis surface will take.)

The same forms exist on every engine through the `Kernel`
trait: a value copy is `set_input`, a cell is
`attach_shared_cell`, and the constants are folded at build.
A traversal's activation snapshots the parent's values into
the body's kernel on whichever engine runs it
([for_traversal.md](for_traversal.md)).

### 4.3 The Axiom suite — CF-axioms

#### Axiom CF1 — Surface completeness

**Every extern slot in the kernel's `input_defs` is present
in the synthesis surface, and Context Fusion fills every slot
before scope-init evaluation begins. No "lazy" slots, no
"resolved on first read" — every declared slot is filled at
scope-init time.**

Enforcement: `materialize_wiring_from_outer` walks every
cell and every outer output against `input_defs`; parent-
gated construction ([subcontext_construction.md](subcontext_construction.md))
is the only path that materialises a child, so no alternative
synthesis path exists.

#### Axiom CF2 — Deterministic synthesis

**For a fixed outer scope state and a fixed program, Context
Fusion produces the same slot values every time. There is no
nondeterministic ordering, no synthesis-time random choice,
no implicit context not derivable from the outer chain.**

Enforcement: the synthesis walk is structural over the outer
program's outputs and `input_defs` (defined ordering) and the
outer chain lookup is shadow-aware (transit suppression).
Identical inputs produce identical slot vectors.

#### Axiom CF3 — Gradient honouring

**The classification of each slot during synthesis (value
copy / cell / transit) is *exactly* the materialization
gradient of [wire_materialization.md](wire_materialization.md),
derived from the outer binding's modifier and ownership.
Context Fusion does not "promote" or "demote"
classifications; it honours them.**

Enforcement: the classification is read from the outer
program's binding modifiers (`const`, `shared`) and input
kinds, fixed at the outer program's build, and applied by the
materializer at scope-init. No re-classification at
synthesis.

#### Axiom CF4 — Synthesis happens once per scope-init

**Context Fusion fires once at scope-init: when the kernel
is being instantiated as a fresh scope. It does not fire
per coordinate. State advance between coordinates is the typed writes (S3),
which is narrow and named.**

Enforcement: the private materializer runs once per child
construction; `set_inputs` is the per-cycle surface and
mutates only coordinate slots. There is no "re-fuse the
context mid-scope" surface.

### 4.4 Interaction with the substrate

Context Fusion is the runtime mechanism that gives the
Synthesis pillar (S1, S2) its concrete realisation. Where
the substrate says "the chain synthesises scope state into
slots," Context Fusion is the synthesis act.

T1, T2 (type-checked slots) are enforced *during* synthesis:
the slot's declared `PortType` is verified against the
outer binding's value type; mismatches use the boundary
adapter catalog (T2). L2 (lifecycle bridging) is the
classification that drives the synthesis path — only slots
of Effectively-const lifecycle (per H1) are filled by
Context Fusion; Dynamic slots are filled by S3 per cycle.

---

## 5. Graph Fusion Phase 2 — Node Fusion

Node Fusion is the compile-time rewriting of the graph. It
runs during wire resolution, **before** hoisting, so its
rewrites are visible to the lifecycle classification and the
classification reflects the rewritten structure. There are
two mechanisms.

### 5.1 Adapter insertion at wire resolution

A wire whose producer `PortType` differs from its consumer's
declared input type is healed while the wire is resolved by a
node from the adapter catalog
(`compile::assembly::auto_adapter`, the class-A conversions of
[type_system.md](type_system.md) §3): the adapter is inserted
between producer and consumer and `TypeAdapterInserted` is
logged. A pair with no catalog entry is
`AssemblyError::TypeMismatch`. Under strict mode the implicit
coercion is refused, with a message naming the explicit
conversion to write. A few variadic, type-polymorphic nodes
(`printf`, `pick`, the `log_*` family, `exactly_one_value`)
declare placeholder port types and skip the check; they
validate their inputs at eval.

### 5.2 Subgraph fusion

After every wire is resolved,
`compile::fusion::apply_fusions` rewrites recognised
subgraphs into single fused nodes. The catalog is whatever
the linked node crates register through
`FusionRuleRegistration` (inventory), collected by
`fusion::default_rules` in ascending priority, ties by name;
polydat-nodes contributes `hash_mod_to_hash_range` (10),
`hash_unit_lerp_to_hash_interval` (20), and
`unit_lerp_to_scale_range` (30). A `FusionRule` is a pattern, a
replacement factory, and the names of the captured wires that
become the fused node's inputs. Every fused node implements
`FusedNode::decomposed`, which rebuilds the unfused subgraph
it replaces; that is the equivalence contract NF2 rests on
and what the equivalence tests exercise. A node directly
referenced by a declared output is never consumed as an
interior node. The pass runs to a fixpoint (§8.3).

Both mechanisms are *graph rewrites*: nodes are added,
removed, or reconnected. The graph after Node Fusion is a
structurally different graph from the one the parser
produced; the classification that follows sees the rewritten
graph, not the original.

### 5.3 The Axiom suite — NF-axioms

#### Axiom NF1 — Slot contract preservation

**A Node Fusion rewrite must preserve the slot contract:
the input slots, output slots, and their declared
`PortType`s of the rewritten subgraph match the input
slots, output slots, and types of the original subgraph
modulo type-compatible edge adapters.**

Enforcement: the assembler ([`compile::assembly`]) resolves
every wire of the rewritten graph against T1+T2. A fusion
that introduces a slot-contract violation fails assembly.

#### Axiom NF2 — Determinism preservation

**A Node Fusion rewrite must preserve evaluation determinism:
for every input vector `v`, the rewritten subgraph produces
the same output vector as the original subgraph.**

Enforcement: `FusedNode::decomposed`. Every fused node can
rebuild the subgraph it replaced, and the equivalence tests
evaluate both on a representative input space.

#### Axiom NF3 — Lifecycle preservation or weakening

**A Node Fusion rewrite may preserve or *weaken* the
lifecycle classification of any wire, but may not strengthen
it. That is: Effectively-const → Effectively-const (preserve)
or Effectively-const → Dynamic (weaken) is allowed; Dynamic
→ Effectively-const requires explicit proof and lifecycle
pinning.**

Enforcement: hoisting (§3) classifies the fused graph from
scratch; a fused node's lifecycle is the join of the inputs
the fused subgraph reads, which are the inputs its members
read, so the rewrite cannot reach a stronger class than its
members did.

#### Axiom NF4 — Compositional closure

**The result of a Node Fusion rewrite is itself a subgraph
eligible for further Node Fusion. The fusion pass iterates
until no further patterns apply.**

Enforcement: `apply_fusions` is a fixpoint iteration over the
graph with the registered rules; a rewritten graph re-enters
the matcher pool.

### 5.4 Interaction with hoisting and Context Fusion

**Node Fusion precedes hoisting.** The rewrites are visible
to the lifecycle classification; the analysis sees the
*fused* graph, not the original. NF3 ensures that
fusion-induced lifecycle changes are visible and monotonic.

**Node Fusion is invisible to Context Fusion.** Context
Fusion operates on the synthesis surface (the `input_defs`
of the built program), which is determined post-fusion. The
synthesis act doesn't know which nodes were fused; it just
fills declared slots.

**Adapter insertion interacts with the slot contract at
construction.** A wire with a `U64` source type connecting
to a `Str` consumer gets a `U64 → Str` adapter. The resolved
wire chain has both source and consumer typed correctly; the
substrate's T1 + T2 hold across the inserted adapter.

---

## 6. The pipeline as ordered composition

```text
                  ┌──────────────────────────────┐
                  │   Parse + Bind               │   one prologue (Prepared),
                  │   (assemble_parent)          │   one lowering; auto-externs;
                  │                              │   tiles typed; for bodies lifted
                  └─────────────┬────────────────┘
                                │ assembler
                                ▼
                  ┌──────────────────────────────┐
                  │   resolve_with_log           │
                  │   - wire resolution +        │   auto_adapter; strict refuses
                  │     adapter insertion        │   the implicit coercion
                  │   - strict-wire assertions   │   AssertValue under strict_values
                  │   - Node Fusion (§5)         │   default_rules to a fixpoint
                  │   - dead-code elimination    │   reachable from outputs
                  │   - topological sort         │   Kahn; CycleDetected
                  │   - round-trip lint          │   warning; error under strict_values
                  └─────────────┬────────────────┘
                                │ ResolvedDag (engine-neutral)
                                ▼
                  ┌──────────────────────────────┐
                  │   compile_with(Engine)       │   classify_lifecycle (§3) on
                  │   interpreter: cones + fold  │   every engine; strict refusals;
                  │   closures: closure plan     │   compile-constant fold;
                  │   native: hybrid segments    │   ConstantFolded logged
                  └─────────────┬────────────────┘
                                │ kernel on the chosen engine
                                ▼
                       --- compile end ---

                       --- scope-init begin ---

                                │
                                ▼
                  ┌──────────────────────────────┐
                  │   Context Fusion (§4)        │   materialize_subscope:
                  │   - cell cascade             │   cells, value copies,
                  │   - value copy / cell attach │   resolver, const pull,
                  │   - const pull               │   scope coordinates
                  └─────────────┬────────────────┘
                                │ ready kernel
                                ▼
                       --- READY KERNEL ---
                       (slot contract held)
```

The pipeline's correctness is the composition of every pass's
axioms with the substrate's. The substrate guarantees the
slot contract holds at every cross-pass boundary; this doc's
axioms (H, CF, NF) guarantee each pass preserves the contract.
The pipeline is engine-neutral up to `ResolvedDag`; the
`Engine` the host names ([engines.md](engines.md)) selects
what runs the resolved graph and changes nothing about what
it computes.

---

## 7. Why this matters

The graph compiler is what makes polydat's *zero-overhead
composition* claim concrete. Without the three mechanisms in
this doc, the substrate's slot contract would be a static
property that workload authors would have to manually
honour. With them, the contract is *machinery*: the compiler
holds the substrate's axioms across every authored graph,
automatically.

Three specific properties depend on this:

### 7.1 Workload authors don't write synthesis boilerplate

Per S1 + auto-extern, the compiler discovers what slots
need filling. Per S2 + binding-time materialisation, the
chain fills them. The author writes `query[id={k}]`; the
compiler discovers `{k}` references the outer iter-var;
Context Fusion fills the slot. No "I declare that this
kernel reads k from the outer scope" syntax exists, and none
is needed.

### 7.2 Performance comes from compile-time analysis, not runtime cleverness

Per H1 + H2 + H3, hoisting moves work from per-cycle to
build time or scope-init. The compiler does the analysis
once per program; the runtime executes the partitioned code
paths straight-line. There is no per-cycle "is this value
still the same?" check — the lifecycle classification
carries the proof.

### 7.3 Pattern-level optimisations compose with the substrate

Per NF1–NF4, Node Fusion rewrites preserve the slot
contract. A fusion catalog entry doesn't have to argue from
first principles that its rewrite is sound across every
upstream / downstream combination; it argues from NF1–NF4.
Per-fusion soundness follows from the substrate-level
guarantees + the per-pattern equivalence contract.

---

## 8. Compiler boundaries and deterministic ordering

### 8.1 Adapter catalog

The conversions the compiler inserts are exactly the entries
of `auto_adapter`. There is no separate implicit or
host-discovered conversion namespace; a pair with no entry is
a type error the author resolves with an explicit conversion
node.

### 8.2 Kernel boundary

Hoisting does not move work across scope kernels. Each child
is compiled and classified independently, and parent values
enter through explicit scope inputs. This preserves kernel
ownership, invalidation, and per-fiber state boundaries.

### 8.3 Node Fusion order

Fusion is deterministic for a fixed graph and rule catalog.
Each fixed-point round examines rules in ascending `priority`
(ties by name) and nodes in ascending graph index, applies
the first valid match, then recomputes consumer counts and
restarts. Interior nodes with external consumers or
named-output references are not consumed. A rule's
`priority` is therefore part of the compiler contract; a new
rule must be equivalence-correct under every lower-priority
rule.

---

[`ast`]: ../../../polydat-core/src/ast.rs
[`kernel`]: ../../../polydat-core/src/kernel/mod.rs
[`compile::assembly`]: ../../../polydat-core/src/compile/assembly.rs
[`compile::fusion`]: ../../../polydat-core/src/compile/fusion.rs
[`compile::hybrid`]: ../../../polydat-core/src/compile/hybrid.rs
[`compile::select`]: ../../../polydat-core/src/compile/select.rs
