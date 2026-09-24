# Scope trees on any engine

Status: implemented 2026-09-24 (§3–§6, §8); the nmbrs migration (§9,
steps 2–3) is the host's. Builds on
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
| `PullPlan` sealed by `Arc<PolydatProgram>` identity | `program_id() -> ProgramId`, on `Kernel` and on `KernelProgram` | new; see below |
| settle pulse: raw `set_input` on `cycle` to advance the generation | `set_inputs` then `invalidate_all` (exist) | |

**Identity, not a program handle.** A compiled kernel's program is the
kernel itself (`into_program` wraps it), so a borrowing
`program() -> Arc<dyn KernelProgram>` would clone the kernel per call and
could not be an identity. What a host seals a plan against is the
identity: `program_id()` is equal for every kernel created from one
program, for every fork, and for the `KernelProgram` itself, and differs
between any two programs, including one source compiled twice.
`ScopeModule::program_on(engine)` returns the same `Arc` on every call,
so a host seals once per module and engine, on
`module.program_on(engine)?.program_id()`.

**Positions agree.** A module's analysis program (`module.program()`)
and every engine's kernel of it list inputs and outputs in the same
order, so an index a host resolves on one is valid on the other
(`a_module_s_program_and_its_kernels_agree_on_positions`).

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
since a cell is a register the scope owns and not a value it holds, and
transit cells (those the kernel carries for descendants without a slot
of its own) travel with the fork. Broadcast cells stay with the
original, which is what descendants are bound to. On the interpreter the
fork also copies the current output buffers, so a scope-init constant
materialized once is current in every fork.

## 4a. Threads

`Kernel: Send + Sync`. A scope parent is shared across threads (nmbrs
holds it in an `Arc` read by every fiber task), and `fork(&self)`,
`bind_under(parent: &dyn Kernel, …)`, and `instantiate_under` are sound
when many threads call them at once on one parent.

The basis per engine: the compiled kernels are `Sync` by construction
(their on-demand broadcast cells are made under a mutex); the
interpreter's `EngineCore` is `Sync` by one stated invariant, that no
`&self` method mutates the core, which its `unsafe impl` now names so a
later `&self` cache cannot break it silently. The finalized native
modules and a tile's body-kernel set are immutable or reached only
through `&mut self`.

## 5. Construction

| Concrete call | Any-engine form |
|---|---|
| `build_subscope` with source matter (label, source, inherited outputs, options, result bindings) | `SubcontextBuilder::under(parent: &dyn Kernel)` (new; the builder's constructor was crate-private) … `finalize()` → `ScopeModule::instantiate_under(parent, engine, bindings) -> Result<Box<dyn Kernel>, KernelError>` |
| `build_subscope` with program matter, `for_iteration(canonical, parent, bindings)` | `kernel::bind_under(parent, program: Arc<dyn KernelProgram>, bindings) -> Result<Box<dyn Kernel>, WriteError>` (new); the child is on `program`'s engine |
| `propagate_inputs_into(child)` | `kernel::propagate_inputs(parent: &dyn Kernel, child: &mut dyn Kernel) -> Result<(), WriteError>` (new) |

`instantiate_under` is `bind_under` over `program_on(engine)`, plus the
module's Rule 2 write-throughs, which it hands to every instance so
`commit_write_throughs` knows them on every engine. An iteration binding
the child refuses is `KernelError::Write`, where it used to be dropped.

`propagate_inputs` writes each of the parent's inputs that has a value
into the child's input of the same name with `set_input_at`, skipping the
child's coordinates (a host positions those with `set_inputs`) and its
cell-bound inputs (writing one would publish into a register the scope
shares). A value the child's declared input refuses is an error naming
it, not a skip.

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

`crates/polydat/tests/scope_trees.rs` builds the shapes a scope-tree host
builds (params root, a `set:` scope with a const that shadows a
parameter, a phase, a fiber fork, a reset, an iteration child per tuple,
a per-op child with result bindings committing into a shared cell) on
every engine, with every parent-child engine pair, and requires the
interpreter's answers at each step. It also pins the program identity
across binds and forks, the positions a module's program and its kernels
agree on, `propagate_inputs`'s refusal, and eight threads binding and
forking under one shared parent at once. The const-shadow case of
2026-09-24 (nmbrs's `set: { mode: "mode_for_{size}" }`) is one of its
steps.

Writing it found two defects in the compiled engines, both fixed with
it: a compiled kernel's `coord_count()` answered the number of buffer
slots every input occupies rather than the number of coordinates, and a
cell refresh on the closure and native tiers wrote the cell's value into
an extern's slot without clearing the slot's `None` mark, so an extern
with no default bound to a parent's cell read `None` forever.

## 9. Order

1. polydat: the surface of §3–§5 on every engine, with §8's test.
2. nmbrs: the fiber and per-op kernels, which run every cycle, onto a
   native engine; synthesis unchanged (§2).
3. nmbrs: the adapter API from `Arc<PolydatKernel>` to `dyn Kernel`, then
   the activation-time kernels.
