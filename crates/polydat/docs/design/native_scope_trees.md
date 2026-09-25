---
type: specification
title: Scope Trees on All Four Engines
timestamp: 2026-09-25
description: Building and driving a tree of child scopes on all four engines through the Kernel trait, including the binder and writes of varying type.
tags: [scopes, engines, host]
---

# Scope Trees on All Four Engines

This specification defines the `Kernel` trait surface a host uses to build
and drive a tree of child scopes on any of the four engines (the interpreter,
the closure tier, native, and pure native): the per-cycle operations (§3),
`fork` (§4), thread safety (§4a), the construction forms (§5), and lookup
(§6). A *scope tree* is a hierarchy of kernels in which each child is bound
to its parent's wires; the *binder* is the code that performs that binding
(`wire_child_under`, reached through `bind_under`). Polydat provides the
surface of §3–§6 and the conformance suite of §8. A host that moves a scope
tree from the interpreter's concrete type `PolydatKernel` onto this surface
is responsible for the obligations of §9.

**Related specifications:**
[subcontext_construction.md](subcontext_construction.md) (the binder),
[scope_model.md](scope_model.md), and
[input_variance.md](input_variance.md) (writes of varying type).

## 1. Scope

A host that runs workloads as a tree of scopes (params, workload, phase,
fiber, per-op) and builds the tree through `PolydatKernel` runs every cycle
on the interpreter, with at most its cones native. Binding a child under a
parent does not depend on the engine: `wire_child_under` takes
`&mut dyn Kernel` on both sides, `ParentView::of_kernel` describes a parent
on any of the four engines, and
`ScopeModule::instantiate_under(parent, engine, bindings)` returns a
`Box<dyn Kernel>`. This specification covers what surrounds the binding: the
per-cycle operations a host performs on a bound kernel's state, and
any-engine forms of the two construction forms that `PolydatKernel` offers.

The diagram shows a typical scope tree and the call that creates each kernel
in it.

![A scope tree: a params root kernel; a workload scope and a phase scope below it, each bound with bind_under or instantiate_under; per-fiber kernels created from the phase kernel with fork; per-op and iteration children bound under a fiber kernel with instantiate_under or bind_under](../diagrams/native_scope_trees-scope-tree.png)

The surface covers the concrete-API uses of a scope-tree host: a
per-cycle path of raw `PolydatState` operations by
pre-resolved index, scope synthesis that reads program structure at load,
and adapters that name `Arc<PolydatKernel>`.

## 2. What stays on the program

Scope synthesis at load reads graph structure that is the same on all four
engines: which inputs are coordinates, whether an output is statically
known, the statements a child re-emits from its parent, the outputs a scope
declares itself, checkpoint identity, and the describe views. None of it
runs per cycle, and all of it is determined by the resolved program, not by
a kernel's state.

This analysis stays on the program. A `ScopeModule` pairs the analysis
program (`module.program()`, a `PolydatProgram`) with its per-engine images
(`program_on(engine)`). Synthesis reads the former, and execution
instantiates the latter. This specification adds no surface for it.

## 3. The per-cycle surface

Each row maps an interpreter raw-state operation to the `Kernel` method
that replaces it; all four engines implement every method in the table.
Index arguments are positions in `input_names` or `output_names`, resolved
once by the host.

| Interpreter raw operation | `Kernel` surface | What it does |
|---|---|---|
| `pull_by_index` | `pull_at` | |
| `read_input_value(idx)` | `input_value_at(idx)` | returns the value input `idx` holds now (`None` past the end); `input_value(name)` is the by-name form |
| raw `set_input(idx, v)` | `set_input_at` | typed; a value that must be converted is converted first (input_variance.md §6) |
| `set_inputs` on coordinates | `set_inputs` | `set_input_at` refuses a coordinate |
| `coord_count` | `coord_count()` | returns how many inputs are coordinates (they come first in `input_names`); also derivable from `input_names` and `externs` |
| `input_default_by_idx` | `input_default_at(idx)` | returns the value input `idx` starts with: an extern's declared default, `U64(0)` for a coordinate; `None` past the end |
| `reset_inputs_from(coord_count)` | `reset_inputs()` | sets every non-coordinate input back to its default, leaves cell-bound slots untouched, and marks dependents not current |
| `shared_cell(idx).is_none()` | `input_is_cell_bound(idx)` | returns whether input `idx` is bound to a shared cell |
| `seed_node_buffer` / `node_buffer` for const capture; `cell_scope_snapshot`; `for_iteration(k, k, &[])` as a scratch copy | `fork() -> Box<dyn Kernel>` | see §4 |
| `advance_broadcasts` | `publish_broadcasts()` | pulls every output that has a broadcast cell, so descendants read current values |
| `commit_write_throughs` | `commit_write_throughs()` | pulls each synthetic `__write_<name>` output and writes it through the parent's shared cell (Rule 2 write-through) |
| `PullPlan` sealed by `Arc<PolydatProgram>` identity | `program_id() -> ProgramId`, on `Kernel` and on `KernelProgram` | see below |
| settle pulse: raw `set_input` on `cycle` to advance the generation | `set_inputs` then `invalidate_all` | |

**Identity, not a program handle.** A compiled kernel's program is the
kernel itself (`into_program` wraps it), so a borrowing
`program() -> Arc<dyn KernelProgram>` would clone the kernel on every call
and could not serve as an identity. A host seals a plan against the
identity instead: `program_id()` is equal for every kernel created from one
program, for every fork, and for the `KernelProgram` itself, and it differs
between any two programs, including one source compiled twice.
`ScopeModule::program_on(engine)` returns the same `Arc` on every call, so a
host seals once per module and engine, on
`module.program_on(engine)?.program_id()`.

**Positions agree.** A module's analysis program (`module.program()`)
and its kernel on each of the four engines list inputs and outputs in the
same order, so an index a host resolves on one is valid on the others
(`a_module_s_program_and_its_kernels_agree_on_positions`).

## 4. `fork`

`fork(&self) -> Box<dyn Kernel>` returns a new kernel over the same program
with this kernel's state. It serves three host operations:

- A fiber starts from its scope's kernel with its `const` bindings
  already initialized, so it does not evaluate them again; `fork` does
  not initialize the new kernel (on
  `PolydatKernel`, copying const-output buffers by (node, port) with
  `seed_node_buffer`).
- An activation scope shares its parent's cells (on `PolydatKernel`,
  `cell_scope_snapshot`).
- A probe evaluates a copy without disturbing the original (on
  `PolydatKernel`, `for_iteration(k, k, &[])`).

On all four engines `fork` is the engine's `Clone`: a new state of the same
program (engines.md §3.5), with the inputs (the const slots included),
the current outputs, and the attached cells as they are. Cells stay shared, because a cell is a register
the scope owns and not a value it holds. Transit cells, which the kernel
holds only to pass to descendants because it has no slot of their name,
are copied into the fork. Broadcast cells stay with the original, because
descendants are bound to the original. On the interpreter the fork also
copies the current output buffers, so a scope-init constant materialized
once is current in every fork.

## 4a. Threads

`Kernel: Send + Sync`. A scope parent is shared across threads (a host
typically holds it in an `Arc` that every fiber task reads), and
`fork(&self)`, `bind_under(parent: &dyn Kernel, …)`, and `instantiate_under`
are sound when many threads call them at once on one parent.

The basis differs per engine. The compiled kernels (closure tier, native,
and pure native) are `Sync` by construction; the closure tier and native
create their on-demand broadcast cells under a mutex. The interpreter's
`EngineCore` is `Sync` by one stated invariant, that no `&self` method
mutates the core; its `unsafe impl` states that invariant, so a `&self`
cache added later cannot break it silently. The finalized native modules
and a tile's body-kernel set are immutable or reached only through
`&mut self`.

## 5. Construction

| Concrete call | Any-engine form |
|---|---|
| `build_subscope` with source matter (label, source, inherited outputs, options, result bindings) | `SubcontextBuilder::under(parent: &dyn Kernel)` … `finalize()` → `ScopeModule::instantiate_under(parent, engine, bindings) -> Result<Box<dyn Kernel>, KernelError>` |
| `build_subscope` with program matter, `for_iteration(canonical, parent, bindings)` | `kernel::bind_under(parent, program: Arc<dyn KernelProgram>, bindings) -> Result<Box<dyn Kernel>, KernelError>`; the child is on `program`'s engine |
| `propagate_inputs_into(child)` | `kernel::propagate_inputs(parent: &dyn Kernel, child: &mut dyn Kernel) -> Result<(), WriteError>` |

`bind_under(parent, program, bindings)` creates an uninitialized kernel of
`program` (`create_uninitialized`), writes the iteration bindings `bindings`
(name/value pairs) into its inputs first, so the child's own scope
coordinates include them, and then binds it under `parent`: the parent's
cells are attached and its values copied in. Last, it initializes the child
(`Kernel::init`), so each `const` in the child is evaluated once from the
bound values. It returns the child, `KernelError::Write` for a binding the
child's input refuses, or `KernelError::ConstInit` naming a const whose
expression failed. The child's `program_id()` is `program`'s.

`instantiate_under(parent, engine, bindings)` is `bind_under` over
`program_on(engine)`, plus the module's Rule 2 write-throughs, which it
hands to every instance so that `commit_write_throughs` knows them on all
four engines. An iteration binding the child refuses is returned as
`KernelError::Write`.

`propagate_inputs(parent, child)` writes each of the parent's inputs that has
a value into the child's input of the same name with `set_input_at`. It
skips the child's coordinates, which a host positions with `set_inputs`, and
the child's cell-bound inputs, because writing one would publish into a
register the scope shares. A value the child's declared input refuses is
returned as an error naming it, not skipped.

The caller names a child's engine. The parent's engine is the usual choice,
since a child belongs to the kernel it was bound under. A tree may mix
engines, for example during a migration, because every binder step is over
`dyn Kernel`.

## 6. Lookup

`KernelLookup::new(&dyn Kernel).lookup(name)` is the lookup for a kernel on
any of the four engines. It returns a `const` output's own value first,
then the value of an input slot of that name, then a value the build folded
(interp.rs). It differs from `PolydatKernel::lookup` in one way: it does not
return a non-`const` output that was computed by a pull, because a compiled
kernel keeps no record of which outputs have been pulled. A host that reads
a computed output pulls it.

## 7. Performance contract

The per-cycle path is the purpose of this surface, so the surface is held to
these costs: `input_value_at`, `set_input_at`, `pull_at`, `reset_inputs`,
and `input_is_cell_bound` allocate nothing and do no name lookup;
`publish_broadcasts` does nothing on a kernel with no broadcast cells; and
`fork` allocates the new state and nothing per cycle afterwards. The cone
spectrum suite and the engine ladder measure the native engines this path
runs on (performance.md).

## 8. Conformance

`crates/polydat/tests/scope_trees.rs` builds the shapes a scope-tree host
builds: a params root, a `set:` scope with a const that shadows a parameter,
a phase, a fiber fork, a reset, an iteration child per tuple, and a per-op
child with result bindings committing into a shared cell. It builds them on
the interpreter, the closure tier, and native (pure native is not in the
suite), with every parent-child engine pair, and requires every result to
equal the interpreter's at each step. It also checks the program identity
across binds and forks, the positions a module's program and its kernels
agree on, `propagate_inputs`'s refusal, and eight threads binding and
forking under one shared parent at once. One of its steps is a
const-shadow case (`set: { mode: "mode_for_{size}" }`).

The suite also holds the compiled engines to two rules. A compiled kernel's
`coord_count()` returns the number of coordinates, not the number of buffer
slots the inputs occupy. A cell refresh on the closure and native tiers
clears the slot's `None` mark when it writes the cell's value into an
extern's slot, so an extern with no default that is bound to a parent's cell
reads the cell's value.

## 9. Order

Polydat provides the surface of §3–§5 on all four engines, with §8's test.
A host that moves a scope tree off `PolydatKernel` is responsible for these steps, in this order:

1. Move the fiber and per-op kernels, which run every cycle, onto a native
   engine, leaving scope synthesis on the program (§2).
2. Change its adapter API from `Arc<PolydatKernel>` to `dyn Kernel`, then
   move the activation-time kernels.
