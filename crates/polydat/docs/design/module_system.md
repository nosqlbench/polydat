# Polydat Module System

This specification defines file-backed module discovery, typed module
interfaces, call resolution, and graph inlining. Module definitions use the
grammar in [grammar.md](grammar.md); their compiled composition semantics use
the inline mode in [scope_model.md](scope_model.md).

## 1. Module definition

A formal module is a typed function-shaped definition inside a `.polydat`
source file:

```polydat
hash_range(input: u64, max: u64) -> (value: u64) := {
    h := hash(input)
    value := mod(h, max)
}
```

The signature is explicit:

- parameters are ordered `(name: type)` entries;
- outputs are ordered `(name: type)` entries; and
- the body is a sequence of ordinary Polydat statements.

A caller invokes the module with normal function-call syntax. There is no
separate `use` or import statement:

```polydat
bounded := hash_range(input: cycle, max: 1000000)
```

Tuple destructuring binds multiple declared outputs in declaration order.
Requesting more targets than the module declares is an error.

## 2. Resolution

Every formal module defined in the program being compiled is registered by
name before any statement compiles, so it resolves without a source
directory and shadows a native node or registered factory of the same name:
an author's `pick(...)` is the author's. For every other call name, native
nodes and registered factories have their normal call-resolution precedence,
and when a name is not resolved there the compiler attempts to resolve a
Polydat module with the same name. Module lookup is deterministic:

1. the compiler's per-compilation module cache, which holds the program's
   own definitions and every module resolved so far;
2. `<name>.polydat` in the source directory and compatible subgraph exports
   discoverable there;
3. the same lookup in each configured Polydat library directory, in caller
   order; and
4. the embedded standard library.

Directory lookups read and parse each `.polydat` file once per process and
reuse the parse while the file's modification time is unchanged, so a cold
compile of a large directory pays for each file once rather than once per
unresolved name, and compilers for `for` bodies and probes share the work.
Files within a directory are visited in name order, so resolution does not
depend on the platform's directory ordering.

Failure to resolve the name through either the node registry or the module
chain is a compile error. File read, parse, signature, and inline errors retain
the module name in the diagnostic.

The public construction surfaces select the available lookup roots:

- `compile_polydat` uses registered nodes and the embedded standard library;
- `compile_polydat_with_path` additionally supplies a source directory; and
- `compile_polydat_with_libs` additionally supplies ordered library paths.

## 3. Formal and compatibility modules

Resolution first searches a file for a formal `ModuleDef` whose name matches
the call. Its parameter and output names, types, and body define the complete
interface.

For source compatibility, a file without a matching formal definition may
export a binding with the requested name. The resolver extracts that binding's
transitive dependency subgraph, infers unbound references as inputs, and
exposes the requested binding as the sole output. Inferred inputs are sorted
to make positional compatibility calls deterministic. This extraction form
has no declared boundary types; normal graph assembly still type-checks all
resulting wires.

## 4. Argument contract

Named arguments map by parameter name. Positional arguments map in signature
order. Formal modules enforce exact parameter coverage, known argument names,
literal compatibility with declared parameter types, and output arity. Wire
arguments are checked by the assembler after inlining.

When the compilation entry point's `strict` flag is true:

- graph inputs require explicit declarations;
- every module argument must be named; and
- every module input must be supplied by the caller rather than falling
  through to an outer coordinate.

The strict compilation flag is distinct from `pragma strict_types` and
`pragma strict_values`. Those pragmas control wire assertions and compose
through the pragma rules in [grammar.md](grammar.md).

## 5. Inlining

Module calls are a compile-time graph operation. For each call, the compiler:

1. allocates a unique prefix for internal binding names;
2. maps module parameters to caller expressions;
3. rewrites and compiles body bindings in source order;
4. verifies that every declared output was produced; and
5. installs typed passthrough nodes from the internal outputs to the caller's
   target names.

Input declarations are satisfied by the caller and are not copied into the
host graph. Nested module definitions, extern declarations, and cursor
declarations are not inlined as body nodes. Module pragmas contribute to the
combined pragma set. The resulting nodes participate in the host program's
single wiring validation, lifecycle analysis, topological ordering, dead-code
elimination, fusion, and engine selection passes. No runtime module boundary
remains.

## 6. Invariants

1. A module call has the same observable value sequence as its inlined body.
2. Every generated internal name is unique to one call site.
3. Formal parameter and output order is stable and controls positional
   mapping.
4. A formal output that is not produced by the body is a compile error.
5. Module resolution and inlining complete before program assembly; runtime
   evaluation never performs filesystem lookup.
6. The compiled DAG, not the source file boundary, owns caching,
   invalidation, purity, and execution-tier behavior.
