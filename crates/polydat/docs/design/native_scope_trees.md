# Scope trees on any engine

Status: proposed 2026-09-24. Builds on
[subcontext_construction.md](subcontext_construction.md) (the binder),
[scope_model.md](scope_model.md), and
[input_variance.md](input_variance.md) (writes of varying type).

## 1. The problem

A host that runs workloads as a tree of scopes (params, workload, phase,
fiber, per-op) builds the tree through the interpreter's concrete type,
`PolydatKernel`, and so runs every cycle on the interpreter, with at
best its cones native. The binding of a child under a parent already
works on any engine: `wire_child_under` takes `&mut dyn Kernel` on both
sides, `ParentView::of_kernel` describes a parent of any engine, and
`ScopeModule::instantiate_under(parent, engine, bindings)` returns a
`Box<dyn Kernel>`. What keeps a tree on the interpreter is everything
around the binding: the per-cycle operations a host performs on a bound
kernel's raw state, and two construction forms that only the concrete
type offers.

nmbrs's map of its concrete-API use (2026-09-24, nmbrs @ 379c613) is the
inventory this document answers. Its per-cycle path uses raw
`PolydatState` operations by pre-resolved index; its scope synthesis
reads program structure at load; its adapters name `Arc<PolydatKernel>`.

## 2. What stays on the program

Scope synthesis at load reads graph structure that is the same on every
engine: which inputs are coordinates, whether an output is statically
known, the statements a child re-emits from its parent, the outputs a
scope declares itself, checkpoint identity, and the describe views.
None of it runs per cycle, and all of it is a property of the resolved
program, not of a kernel's state.

It stays where it is. A `ScopeModule` already pairs the analysis program
(`module.program()`, a `PolydatProgram`) with its per-engine images
(`program_on(engine)`). Synthesis reads the former; execution
instantiates the latter. This document adds nothing for it.

## 3. The per-cycle surface

Each row replaces a raw-state operation with a `Kernel` method every
engine implements. Index arguments are positions in `input_names` or
`output_names`, resolved once by the host.

| Raw operation today | `Kernel` surface | Notes |
|---|---|---|
| `pull_by_index` | `pull_at` (exists) | |
| `read_input_value(idx)` | `input_value_at(idx)` | new; `input_value(name)` exists |
| raw `set_input(idx, v)` | `set_input_at` (exists) | typed; a value that must be converted is converted first (input_variance.md §6) |
| `set_inputs` on coordinates | `set_inputs` (exists) | `set_input_at` refuses a coordinate |
| `coord_count` | `coord_count()` | new; today derivable from `input_names` and `externs` |
| `input_default_by_idx` | `input_default_at(idx)` | new |
| `reset_inputs_from(coord_count)` | `reset_inputs()` | new: every non-coordinate input back to its default, cell-bound slots untouched, dependents marked not current |
| `shared_cell(idx).is_none()` | `input_is_cell_bound(idx)` | new |
| `seed_node_buffer` / `node_buffer` for const capture; `cell_scope_snapshot`; `for_iteration(k, k, &[])` as a scratch copy | `fork() -> Box<dyn Kernel>` | new; see §4 |
| `advance_broadcasts` | `publish_broadcasts()` | new: pull every output with a broadcast cell so descendants read current values |
| `commit_write_throughs` | `commit_write_throughs()` | new: Rule 2 write-through into the parent's shared cells |
| `PullPlan` sealed by `Arc<PolydatProgram>` identity | `program() -> Arc<dyn KernelProgram>` | new, non-consuming; `Arc::ptr_eq` on it is the seal |
| settle pulse: raw `set_input` on `cycle` to advance the generation | `set_inputs` then `invalidate_all` (exist) | |

`into_program` stays as it is; `program()` is the borrowing form a host
needs to recognize a kernel's program without giving the kernel up.

## 4. `fork`

Three host operations are one idea: a new kernel over the same program
with this kernel's state.

- A fiber starts from its scope's kernel with the scope-init constants
  already materialized, so it does not evaluate them again (today:
  copying const-output buffers by (node, port) with `seed_node_buffer`).
- An activation scope shares its parent's cells (today:
  `cell_scope_snapshot`).
- A probe evaluates a copy without disturbing the original (today:
  `for_iteration(k, k, &[])`).

`fork` is the operation every engine already has behind `Clone`: a new
state of the same program (engines.md §3.5), with the inputs, the
current outputs, and the attached cells as they are. Cells stay shared,
since a cell is a register the scope owns and not a value it holds. A
host that needs a copy that does not share cells detaches through the
cell API, which is the rarer need and says so.

## 5. Construction

| Concrete call | Any-engine form |
|---|---|
| `build_subscope` with source matter (label, source, inherited outputs, options, result bindings) | `SubcontextBuilder::new(ParentView::of_kernel(parent))` … `finalize()` → `ScopeModule::instantiate_under(parent, engine, &[])` (exists) |
| `build_subscope` with program matter, `for_iteration(canonical, parent, bindings)` | `bind_under(parent, program: Arc<dyn KernelProgram>, engine_of(program), bindings) -> Box<dyn Kernel>` (new; what `instantiate_under` does after `program_on`) |
| `propagate_inputs_into(child)` | `propagate_inputs(parent: &dyn Kernel, child: &mut dyn Kernel)` (new): each of the parent's inputs with a value and a same-named child input is written to the child with `set_input_at`, and a refusal is an error, not a skip |

A child's engine is the caller's to name, and the parent's is the usual
answer (a child belongs to the kernel it was bound under). A tree may mix
engines during a migration: every binder step is over `dyn Kernel`.

## 6. Lookup

`KernelLookup::new(&dyn Kernel).lookup(name)` is the any-engine lookup: a
`const` output's own value first, then an input slot of the name, then a
value the build folded (interp.rs; the const-first order since
2026-09-24). It differs from `PolydatKernel::lookup` in one way: it does
not return a non-`const` output that was computed by a pull, because a
compiled kernel keeps no record of which outputs have been pulled. A host
that reads a computed output pulls it.

## 7. Performance contract

The point of the move is the per-cycle path, so the surface is held to
it: `input_value_at`, `set_input_at`, `pull_at`, `reset_inputs`, and
`input_is_cell_bound` allocate nothing and do no name lookup;
`publish_broadcasts` does nothing on a kernel with no broadcast cells;
`fork` allocates the new state and nothing per cycle afterwards. The
cone spectrum suite and the engine ladder measure the native engines
this path runs on (performance.md).

## 8. Conformance

One test builds the shapes a scope-tree host builds (params root, a
`set:` scope with a const that shadows a parameter, a phase, a fiber
fork, a per-op child with result bindings and a shared-cell write-
through, an iteration child per tuple) on every engine, with every
parent-child engine pair, and requires the interpreter's lookups, pulls,
resets, and cell values at each step. The const-shadow case of
2026-09-24 (nmbrs's `set: { mode: "mode_for_{size}" }`) is one of its
steps.

## 9. Order

1. polydat: the surface of §3–§5 on every engine, with §8's test.
2. nmbrs: the fiber and per-op kernels, which run every cycle, onto a
   native engine; synthesis unchanged (§2).
3. nmbrs: the adapter API from `Arc<PolydatKernel>` to `dyn Kernel`, then
   the activation-time kernels.
