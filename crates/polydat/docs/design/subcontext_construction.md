---
type: specification
title: Parent-Gated Subcontext Construction
timestamp: 2026-09-25
description: The typed construction boundary for a child scope, enforcing lifecycle isolation and cross-tier write-through.
tags: [scopes, host]
---

# Parent-Gated Subcontext Construction

This specification defines the typed API through which a host builds a child
scope under a parent scope: a builder collects the child's source and
contract, `finalize` compiles and closes it into a module, and the parent's
`spawn` creates the child kernel and binds it to the parent's wires. The
construction is *parent-gated*: a child is created only through a call on its
parent, and that call binds it. This API enforces the lifecycle isolation and
cross-tier write-through that [The Composition
Substrate](composition_substrate.md) requires. A *write-through* is a child
assignment that is published into a shared cell owned by an ancestor scope
(§3.1, §5).

**Related specifications:** [The Composition
Substrate](composition_substrate.md), [The Scope Model](scope_model.md), and
[Wire Materialization](wire_materialization.md).

The governing rule is:

> A child `ScopeKernel` is constructed from a closed `ScopeModule` by its
> parent `ScopeKernel`. `finalize` closes and compiles module matter; `spawn`
> is the single point at which parent/child wiring becomes live.

---

## 1. Public model

```mermaid
flowchart LR
    P[Arc ScopeKernel P] -->|subcontext_builder| B[SubcontextBuilder P]
    B -->|finalize| M[ScopeModule Child P]
    P -->|spawn name, module| C[ScopeKernel Child P]
    C -->|subcontext_builder| G[Next child builder]
```

The construction types are:

- `ScopeKernel<M>` — a typed wrapper around a live `PolydatKernel`, with a
  structured name, source context, named-child registry, pull consumers, and
  write-through bindings;
- `SubcontextBuilder<P>` — an accumulator that owns an `Arc` to the parent and
  records imports, exports, body fragments, consumers, diagnostics, and compile
  options;
- `ScopeModule<Child<P>>` — the closed artifact produced by `finalize` and
  consumed by `ScopeKernel<P>::spawn`;
- `ScopeContract<M>` — typed import and export handle bundles branded by `M`;
- `ImportSpec` and `ExportSpec` — named declarations carrying port type and
  lifecycle classification;
- `ChildName` — a hierarchical registry key; and
- `SourceContext` — a diagnostic label with optional file and line range.

`Child<P>` records the parent type in the child type. This prevents handles for
different module brands from being interchanged at compile time. Runtime child
identity is additionally enforced by `ChildName` in the parent's registry.

---

## 2. Builder inputs

### 2.1 Body fragments

`SubcontextBuilder::body` accepts either:

```rust,ignore
BodyFragment::PolydatSource(String)
BodyFragment::Statements(Vec<Statement>)
```

Source fragments are lexed and parsed during `finalize`. Statement fragments
enter the same compiler without a text round trip. Multiple fragments are
concatenated in registration order. An empty body is a contract error.

### 2.2 Imports and exports

Imports classify a name as `CompileConst`, `Extern`, `Shared`, or
`IterationExtern`. Exports classify a name as `Local`, `Final`, `Shared`,
`Coordinate`, or `Volatile` and carry the corresponding binding modifier.

`finalize` enforces exactly these rules:

- `finalize` requires every declared import name to exist among the parent's
  input or output names;
- the compiled body must type-check under the ordinary Polydat compiler;
- an import unused by the compiled body is retained in the module contract and
  emits a module diagnostic;
- a child export cannot shadow a same-named parent `const` output; and
- a child export matching a shared cell visible at the parent becomes a typed
  write-through binding.

`ImportSpec::port_type` and `ImportSpec::classification` are preserved in the
public `ScopeContract`, but `finalize` does not independently compare them with
a typed parent-export manifest. Actual child input slots and shared-cell writes
remain protected by the compiler and kernel slot type checks.

### 2.3 Compile options

The subcontext builder's `CompileOptions` (`kernel::subcontext::CompileOptions`,
distinct from `dsl::compile::CompileOptions`, into which `finalize` maps it)
holds the compile-time inputs that affect the child program:

```text
workload_dir
polydat_lib_paths
strict
required_outputs
context_label
cursor_limit
kernel_opt
```

Every option set compiles through the ordinary Polydat compiler:

- default options compile the statements under the default
  `dsl::compile::CompileOptions`, charged to the parent's ledger;
- non-default options are mapped into `dsl::compile::CompileOptions` and
  compile the (possibly rewritten) statements, or the re-joined source when
  no rewrite fired and every fragment is source.

`KernelOptLevel` also controls whether result-binding support keeps only the
injected slots the body references or retains all diagnostic slots (§3.2).

### 2.4 Pull consumers

`register_pull` records an `Arc<dyn PullConsumer>`, a host object that will
pull named outputs of the child, and carries it into the closed module and
the spawned kernel. A consumer supplies a stable diagnostic label and an
ordered list of output names. Resolving those names into a host pull plan is
the consumer layer's job; this protocol only collects the registrations and
passes them on.

---

## 3. Finalization

`SubcontextBuilder::finalize` performs these operations in order:

1. Snapshot the parent input and output name sets.
2. Reject declared imports that have no matching parent name.
3. Reject a child export that shadows a parent `const` output.
4. Discover shared cells visible at the parent, including cells inherited from
   an ancestor.
5. Parse every source body fragment and concatenate all statements.
6. Rewrite shared export collisions into explicit write-through form.
7. Compile the rewritten statements with the requested compile options,
   charging the compile to the parent's `CompileLedger`; the module's
   program carries that ledger.
8. Apply inherited-output marking before the program `Arc` is shared.
9. Bake write-through metadata into the compiled program.
10. Diagnose declared imports not represented in the compiled input/output
    closure.
11. Verify every write-through has both its child input slot and synthetic
    output.
12. Seal program, contract, context, consumers, write-through bindings, and
    diagnostics into `ScopeModule<Child<P>>`.

The resulting artifact has no live parent reference. Its parent-dependent
decisions have already been made by the builder that held the parent.

### 3.1 Shared write-through rewrite

When a child export `X := expr` matches a shared cell named `X` visible at the
parent, finalization rewrites the single-target binding conceptually as:

```polydat
extern X: <cell-type>
__write_X := expr
```

It also records:

```text
WriteThroughBinding {
  export_name: "X",
  source_output: "__write_X"
}
```

Tuple-target bindings are not write-through shapes. They are left to ordinary
compiler validation and cannot silently update one shared cell.

### 3.2 Result bindings

`add_result_bindings` parses result-source statements and applies the same
write-through path. It has these additional rules:

- empty source is a no-op;
- `body: Json`, `count: u64`, and `ok: bool` externs are injected only when
  referenced under release optimization;
- diagnostic optimization retains all three slots;
- assignments to `body`, `count`, or `ok` are rejected because those names are
  runtime-injected inputs; and
- only result LHS names that collide with an in-scope shared cell are declared
  as shared exports for rewriting.

Non-colliding result bindings remain ordinary child outputs with their inferred
types.

The names `body`, `count`, and `ok` and their types are a host convention: a
host that binds an operation's result into a child scope injects them. Polydat
hosts the convention here because the write-through rewrite is the mechanism
it needs; nothing in the language depends on the three names.

---

## 4. Spawn

`ScopeKernel<P>::spawn(name, module)` is the only call that creates a live
typed child. `name` is the `ChildName` the child is registered under in the
parent, and `module` is the `ScopeModule<Child<P>>` that `finalize` produced;
the call consumes the module and returns the new `ScopeKernel<Child<P>>`, or a
`ContractViolation` error. It performs these steps:

1. It locks the parent registry and rejects an existing `ChildName`, reporting
   the prior and current `SourceContext` values.
2. It records the child name.
3. It creates a kernel for the closed child program and binds it under the
   parent. The child is a separate kernel that owns its outputs and their
   storage ([Runtime Model](runtime_model.md) R4). Shared cells visible at the
   parent attach to matching child slots and remain available for transitive
   descendant wiring.
4. It transfers context, consumer registrations, and write-through bindings to
   the new `ScopeKernel<Child<P>>`.

No subsequent API changes the child's parent or cross-binding shape.

### 4.1 Named-child registry

`ChildName` is a segment vector with constructors for phases, operations,
iterations, and composed paths. Segment equality defines identity.

A second live registration under the same name returns
`ContractViolation::DuplicateChild`. `release_child(name)` removes only the
registry entry; it does not invalidate an already returned child kernel. After
release, the parent may spawn a replacement under that name. This explicit
transition supports iteration re-traversal without weakening duplicate-spawn
detection.

### 4.2 Cross-generation sharing

`shared_cells_in_scope` enumerates both cells owned by the parent and cells
attached from its ancestors. Spawn forwards this visibility through silent
intermediate scopes. A root shared cell can therefore bind a grandchild even
when the intermediate body's graph never reads the name.

This transit behavior affects cell availability only. Ordinary non-shared
values follow normal parent/child import and materialization rules.

---

## 5. Write-through after evaluation

`ScopeKernel::commit_write_throughs` is called after the child has produced its
values for a coordinate. It is a no-op when the module has no write-through
bindings.

For each binding it:

1. pulls the synthetic `__write_X` output;
2. resolves the child input slot for `X`;
3. validates or applies the permitted boundary conversion against the slot's
   fixed `PortType`; and
4. writes the value through that input slot to the attached `SharedCell`.

The implementation uses a two-pass pull-then-write sequence to avoid competing
mutable borrows of the kernel. An unhealable type mismatch returns an error at
the write site. Shared-cell concurrency remains last-write-wins by mutex
acquisition order as specified in [The Scope Model](scope_model.md).

---

## 6. Lifecycle and ownership

```text
parent live
  └─ builder open
       ├─ module matter mutable
       └─ finalize
            └─ closed ScopeModule
                 └─ parent.spawn
                      └─ live child ScopeKernel
```

- The builder owns an `Arc` to the parent for its complete lifetime.
- `finalize` consumes the builder.
- `spawn` consumes the module artifact.
- A compiled `PolydatProgram` is immutable and shareable.
- The spawned `ScopeKernel` owns one synchronized mutable kernel state.
- Independent kernels for other fibers are created from the immutable program
  through `KernelProgram::create_kernel`; they are not created by repeating
  `spawn` for the same named child.
- Hot rebinding and multi-parent construction are unsupported.

Direct compilation (`compile_polydat_with`) creates a root kernel on any of
the four engines (the interpreter, the closure tier, native, and pure
native), but it does not create a typed child relationship. The second
sanctioned construction path is `PolydatKernel::build_subscope(PolydatMatter)`.
Its argument is matter built by `PolydatMatter::builder()` from exactly one
of source, pre-parsed statements, or a compiled program. Source and statement
matter go through a transient typed parent (`wrap_root_kernel`) and this
protocol's `finalize`; program matter is bound directly with its iteration
bindings. There is no free-function construction path.

---

## 7. Error contract

The active construction errors are:

| Error | Boundary |
| --- | --- |
| `UnboundImport` | A declared import name is absent from the parent input/output closure. |
| `FinalShadow` | A child export collides with an immutable parent output. |
| `DuplicateChild` | The parent registry already contains the child name. |
| `Compile` | Parsing, rewriting, type checking, or write-through structural validation fails. |
| `StrictNonePropagation` | Strict intermediate-scope materialization would silently fall through after a `const` binding produced `None`. |

A `const` binding whose expression fails when the child is initialized fails
the construction. `kernel::bind_under` returns it as
`KernelError::ConstInit`, naming the const; the interpreter's own spawn path
(`materialize_wiring_from_outer`) has no error return, so it panics with the
same message.

`SourceContext` accompanies construction diagnostics so failures identify their
logical scope and, when supplied, source file and line range.

---

## 8. Normative invariants

**SC1 — Parent gate.** A typed child kernel is produced only by its parent from
a finalized child module.

**SC2 — Closed artifact.** Module body, program, contract, consumers,
write-throughs, and diagnostics cannot be amended after `finalize`.

**SC3 — One binding point.** Parent/child cell attachment occurs during
`spawn`; no public hot-rebind path exists.

**SC4 — Name closure.** Every declared import resolves to a parent-visible name
before body execution.

**SC5 — Immutable shadow protection.** A child cannot redefine a same-named
parent `const` output.

**SC6 — Explicit shared writes.** A child assignment updates an ancestral
shared cell only when finalization produced a `WriteThroughBinding` and
`commit_write_throughs` publishes it.

**SC7 — Type-stable publication.** Every write-through is checked against the
fixed target slot type before publication.

**SC8 — Transitive cells.** A shared cell remains visible through intermediate
scopes that do not consume it.

**SC9 — Named ownership.** Duplicate live child names are errors; reuse requires
an explicit registry release.

**SC10 — Program/state separation.** Program sharing does not imply mutable
state sharing between independent kernel instances.

---

## 9. Relation to traversal activation

Polydat has two ways to produce a child kernel bound to outer wires, and this
protocol is one of them. Both produce a separate child kernel (R4) over a
program compiled once, both bind the child's declared externs to the outer
scope by name through typed writes, and both start the child with fresh
buffers and nothing current. They differ in who composes the child and what
is transferred to it:

- **Subcontext construction** is for a host-composed child with a contract:
  the host supplies imports, exports, body fragments, consumers, and result
  bindings, and the parent's `spawn` binds the child through the full binder
  ([Scope Model](scope_model.md) §4) — shared cells and transit cells
  attached, computed parent outputs attached as broadcast cells, the child
  initialized so its `const` bindings are evaluated from the bound values,
  scope coordinates threaded. The child is an interpreter kernel.
- **Traversal activation** is for the language's own `for`: the body is
  compiled with the parent, and `TraversalStream::activation_on` creates one
  kernel per tuple on the engine the host asks for, binding the
  tuple's elements and a snapshot of the cascaded wires by value,
  narrowing every cursor, and then initializing the activation
  ([for_traversal.md](for_traversal.md)).

In both forms, the iteration does not decide what the child contains. A
comprehension determines the order and the values of the tuples; a host's
scope walker decides the imports, the exports, and the body fragments that
make a child from one tuple; and this protocol's `finalize` validates the
contract between them. Polydat compiles the module matter and binds the named
child. The host releases a replaceable iteration child, through
`release_child`.

The subcontext path through `spawn` produces an interpreter child and
transfers cells, while the traversal path runs on all four engines (the
interpreter, the closure tier, native, and pure native) and transfers values.
The design direction is that both call one binder expressed over the `Kernel`
trait (`shared_cells`, `attach_shared_cell`, `input_value`, `pull`,
`set_input`), so that a host-composed child can run on any of the four
engines and a traversal body can share a cell rather than a snapshot. The
surface a host needs around that binder to run a whole scope tree on a
compiled engine, the per-cycle operations in particular, is specified in
[native_scope_trees.md](native_scope_trees.md).
