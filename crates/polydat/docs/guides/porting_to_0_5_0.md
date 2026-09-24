# Porting a host from polydat 0.4 to 0.5.0

Written against nmbrs, the host that drove this release, and measured the
same way as the [0.3.2 guide](porting_to_0_3_2.md): by building that host
against the tree, not by reading diffs. nmbrs's whole workspace (128
suites, 3177 tests) passes against it with two one-line changes, both in
[Part 1](#part-1-what-stops-compiling). The rest of this guide is the
full list, for hosts that use more of the surface than nmbrs does.

The release adds two things a host builds on and one it gets for free:

- **Scope trees on any engine.** A tree of scopes (params, phases,
  fibers, per-op and iteration children) can be built and driven through
  the `Kernel` trait alone, on any engine, parent and child on different
  engines if it likes ([native_scope_trees.md](../design/native_scope_trees.md);
  embedding guide §4, "Scope trees on any engine").
- **Inputs of varying type**, converted by a node in the program, never
  at the write, and only where the host asks
  ([input_variance.md](../design/input_variance.md); embedding guide §3,
  "When an input's type varies").
- **Native code that follows the graph.** The native engines fuse
  connected groups of nodes, so a pull runs only its own cone, and pure
  native code checks what is stale in native code. Pulling one output of
  a wide program no longer costs the whole program (performance guide,
  "The cone spectrum").

## Part 1: what stops compiling

| Change | What breaks | Fix |
|---|---|---|
| `dsl::compile::CompileOptions` gained `input_variance` and `inferred_externs` | a struct literal naming every field (E0063) | end it with `..Default::default()` |
| `kernel::subcontext::CompileOptions` gained `input_variance` | the same | the same (nmbrs: `scope.rs`) |
| `InputDef` gained `type_origin` and `converts_to` | a literal building an `InputDef`, e.g. for `PolydatProgram::with_inputs` | add `type_origin: TypeOrigin::Declared, converts_to: None`, which keep today's meaning |
| `PortType` gained `Dyn` | a `match` over every variant (E0004) | name types with `to_keyword()` or `Display` and do not match (embedding guide §6, "Naming port types without matching them"); where the match is per-type behavior, add the arm (nmbrs: `describe.rs`) |
| `KernelError` gained `Write` | an exhaustive `match` on `KernelError` | add the arm or a wildcard |
| `AssemblyError` gained `OpenInputs` | an exhaustive `match` on `AssemblyError` | the same |
| `CompileEvent` gained `InputConverterInserted` | an exhaustive `match` over compile events, such as a host's own formatter | the same, or `CompileEventLog::format()` |

`PortType` stays exhaustive on purpose: a `match` over every variant that
stops compiling until it decides what a new type means is what polydat
relies on in its own code. A host that only labels types should not
match at all.

## Part 2: what warns

`Dataflow::set_wire` and `set_wire_idx` are deprecated. They converted a
value at the write through the boundary adapter catalog, which no other
write does. Each call is a warning, and an error only in a crate that
denies warnings. Replace each by what it relied on:

- **A value whose type already matches**: `Kernel::set_input_at` (or
  `set_input` by name). Nothing changes.
- **A value that must be converted, to a type the host knows**:
  `polydat::convert::to_port(value, kernel.input_port_type(name)?)`, then
  `set_input_at`. `to_port` applies the same catalog the old write did and
  returns a `ConvertError` where the old write healed silently or failed.
  nmbrs's command-line phase-parameter override is this case.
- **An input whose type varies over the kernel's life**: compile with
  `CompileOptions::input_variance` at `Warn` or `Info`, and the input takes
  any value through a converter node reported in the compile log. The
  scope builder's result externs (`body`, `count`, `ok`, captures) are
  this case, and `Warn` opens them.

The deprecated writes are removed in the release after this one.

## Part 3: what changes quietly

These compile unchanged and behave differently. Each is a correction.

- **`set_inputs` writes the coordinates and nothing past them.** Given
  more values than a program has coordinates, every engine used to write
  the extras into its externs, and a compiled engine into an extern's
  slots, where a by-reference extern keeps a pointer. A host that relied
  on the positional spill should write those inputs by name.
- **`coord_count()` on a compiled kernel is the number of coordinates.**
  It answered the number of buffer slots every input occupies.
- **`ScopeModule::instantiate_under` reports a refused iteration binding**
  as `KernelError::Write`; the value used to be dropped.
- **An extern with no default, bound to a parent's cell, reads the cell's
  value** on the closure and native engines. It read `None` for good once
  its slot had been unset.
- **`dyn` is a type keyword.** `extern x: dyn` declares an input that takes
  any value, which failed to compile before.
- **Two volatile reads that no wire connects are two steps on every
  engine.** The native tier used to fuse them into one segment and read
  both at the first pull of either (runtime_model.md R1.v).
- **A pull by index publishes to descendants**, as a pull by name always
  did, on every engine. A child bound to a parent's output used to keep
  the last value a pull by name published while the host drove the
  parent with `pull_at`.
- **A child bound to an output the parent has not computed reads
  `None`** on every engine. The compiled engines gave it the type's zero.
- **`input_value_at` on a cell-bound input reads the cell** on the closure
  and native engines, as the interpreter does. It answered the copy the
  child took at its last pull.

## Not breaking, though it looks it

- **`Kernel` gained required methods** (`fork`, `coord_count`,
  `input_value_at`, `program_id`, and others). The trait is sealed, so
  nothing outside polydat implements it, and a caller only gains methods.
- **`Kernel` is `Send + Sync`.** Every engine already was; the bound lets
  a scope parent be shared across threads, which a caller only gains.

## Known issues

- **The scope binder still converts a value it copies into a child's
  declared input**, and the write-through commit still widens a narrower
  numeric type. Both retire with the deprecated writes
  ([input_variance.md](../design/input_variance.md) §11).

## Checklist

1. Patch and relock, and confirm the patch was used.
2. Apply Part 1 until every crate compiles.
3. Replace each `set_wire` call as Part 2 describes it, choosing by what
   the call relied on.
4. Read Part 3 against the host: a positional `set_inputs` spill into
   externs, and a count taken from a compiled kernel's `coord_count()`,
   are the two that change a result. A host that called
   `publish_broadcasts()` after driving a parent by index can keep the
   call, which is now redundant, or drop it.
