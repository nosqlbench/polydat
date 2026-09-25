---
type: specification
title: Module System
timestamp: 2026-09-25
description: File-backed module discovery, typed module interfaces, call resolution, and graph inlining.
tags: [language, compiler]
---

# Module System

This specification defines how Polydat modules are written, found, called,
and compiled. A module is a named, reusable group of bindings with a typed
signature. A program calls it like a function, the compiler finds its
definition in the program itself, in `.polydat` files on disk, or in the
embedded standard library, and the compiler then copies the module's body
into the calling program (inlining), so no module boundary remains at
runtime.

**Related specifications:** module definitions use the grammar in
[polydat_grammar.md §13](polydat_grammar.md#sec-modules); their compiled
composition semantics use the inline mode in
[scope_model.md](scope_model.md).

## 1. Module definition

A formal module is a typed, function-shaped definition inside a `.polydat`
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

A caller invokes the module with normal function-call syntax, passing one
argument per parameter, and binds the module's outputs to wires of its own.
There is no separate `use` or import statement:

```polydat
bounded := hash_range(input: cycle, max: 1000000)
```

Tuple destructuring binds multiple declared outputs in declaration order.
Requesting more targets than the module declares is an error.

## 2. Resolution

When the compiler meets a call, it decides what the called name refers to
in the following order.

1. **The program's own modules.** Every formal module defined in the
   program being compiled is registered by name before any statement
   compiles. It therefore resolves without a source directory, and it
   shadows a native node or registered factory of the same name: a program
   that defines `pick` calls its own `pick`.
2. **Native nodes and registered factories.** Every other call name is
   resolved by the normal call-resolution precedence.
3. **The module chain.** A name not resolved by the node registry is
   looked up as a Polydat module, in this deterministic order:
   1. the compiler's per-compilation module cache, which holds the
      program's own definitions and every module resolved so far;
   2. `<name>.polydat` in the source directory, and compatible subgraph
      exports discoverable there (§3);
   3. the same lookup in each configured Polydat library directory, in
      caller order; and
   4. the embedded standard library.

The following flowchart shows this order for one call name.

![Resolution of a call name: a formal module defined in the program, then native nodes and registered factories, then the module cache, the source directory, the library directories in caller order, and the embedded standard library; a name found nowhere is a compile error](../diagrams/module_system-resolution.png)

Directory lookups read and parse each `.polydat` file once per process and
reuse the parse while the file's modification time is unchanged. A cold
compile of a large directory therefore parses each file once rather than
once per unresolved name, and the compilers for `for` bodies and for
element-type probes (a generator expression compiled on its own to learn
the type of its elements) reuse the same parses. Files within a directory
are visited in name order, so resolution does not depend on the platform's
directory ordering.

Failure to resolve the name through either the node registry or the module
chain is a compile error. File read, parse, signature, and inline errors
include the module name in the diagnostic.

The public construction surfaces select the available lookup roots:

- `compile_polydat` uses registered nodes and the embedded standard library;
- `compile_polydat_interpreter_with_options` with `CompileOptions::source_dir`
  additionally supplies a source directory, and with
  `CompileOptions::lib_paths` (the binary's `--lib`) ordered library paths.

## 3. Formal and compatibility modules

Resolution first searches a file for a formal `ModuleDef` whose name matches
the call. Its parameter and output names, types, and body define the
complete interface.

For source compatibility, a file without a matching formal definition may
export a binding with the requested name. The resolver extracts that
binding's transitive dependency subgraph, treats the subgraph's unbound
references as inputs, and exposes the requested binding as the sole output.
Inferred inputs are sorted, so positional calls to a compatibility module
are deterministic. This extraction form has no declared boundary types;
normal graph assembly still type-checks all resulting wires.

## 4. Argument contract

Named arguments map by parameter name. Positional arguments map in
signature order. Formal modules enforce exact parameter coverage, known
argument names, literal compatibility with declared parameter types, and
output arity. Wire arguments are checked by the assembler after inlining.

Parameter and output types are the port-type keywords of the type system,
compared as types rather than as spellings: `str` and `String` name the
same type, so a caller's `Str` wire satisfies a parameter declared either
way.

When the compilation entry point's `strict` flag is true:

- graph inputs require explicit declarations;
- every module argument must be named; and
- every module input must be supplied by the caller rather than falling
  through to an outer coordinate.

The strict compilation flag is distinct from `pragma strict_types` and
`pragma strict_values`. Those pragmas control wire assertions and combine
through the pragma rules in
[polydat_grammar.md §14](polydat_grammar.md#sec-pragmas).

## 5. Inlining

A module call is a compile-time graph operation. For each call, the
compiler:

1. allocates a unique prefix for internal binding names;
2. maps module parameters to caller expressions;
3. rewrites and compiles body bindings in source order;
4. verifies that every declared output was produced; and
5. installs typed passthrough nodes from the internal outputs to the
   caller's target names.

Input declarations are satisfied by the caller and are not copied into the
host graph. Nested module definitions, extern declarations, and cursor
declarations are not inlined as body nodes. Module pragmas are added to the
combined pragma set. The resulting nodes go through the host program's
single wiring validation, lifecycle analysis, topological ordering,
dead-code elimination, fusion, and engine selection passes. No runtime
module boundary remains.

## 6. Invariants

1. A module call has the same observable value sequence as its inlined body.
2. Every generated internal name is unique to one call site.
3. Formal parameter and output order is stable and controls positional
   mapping.
4. A formal output that is not produced by the body is a compile error.
5. Module resolution and inlining complete before program assembly; runtime
   evaluation never performs filesystem lookup.
6. Caching, invalidation, purity, and execution-tier behavior are
   determined by the compiled DAG, not by the source file boundary.

## 7. Modules and bodies

A `for` body ([The `for` Construct](for_traversal.md) §4) compiles as a
child program with the settings its parent used: the source directory,
the ordered library paths, the strict flag, the diagnostic context label,
the cursor limit, and the pragma set. The child compiler also starts with
the parent's module cache as it stood when the body was lowered. That
cache holds the program's own formal definitions (registered before any
statement compiled, §2) and every module the parent had resolved by then.
The body therefore resolves a module defined in the program itself, and a
module the parent already found on disk, without a source directory of its
own. The body is compiled from this same record on each of the four
engines (the interpreter, the closure tier, native, and pure native), so a
body sees the same modules on all four.

A tile projection body ([Polytile](polytile.md) §6) compiles with the
default settings, registered nodes and the embedded standard library. It
does not inherit the enclosing program's directory, library paths,
pragmas, or program-local modules.

A tile declared inside a module body is inlined with the call under the
call's prefix, as every other statement of the body is. A producer bound
in a module body (`name := for ...`) is bound under the prefix as the
`streamer` constant the `for` expression lowers to.
