---
type: guide
title: Porting to 0.6.0
timestamp: 2026-09-27
description: "The 0.6.0 release notes as a host reads them, covering every change since the published 0.5.0: what stops compiling and its fix, the behavior that changes quietly, the new capabilities, and the known issues."
tags: [release, host]
---

# Porting to 0.6.0

This guide covers porting a host from polydat 0.5.0 to 0.6.0. It lists
every change a host sees between the published 0.5.0 and 0.6.0,
including the ones the 0.5.1 to 0.5.4 patch releases carried: a host
still on 0.5.0 meets them all at once, and a host that took the patch
releases has already met the const and `is_stable` changes of Part 1
and Part 2.

The release makes three things true that a host builds on:

- **A const is a value computed once, at kernel initialization.** Every
  way a kernel comes into existence initializes it, `Kernel::init`
  initializes it again on request, and a const may read externs and
  volatile expressions
  ([evaluation_model.md](../design/evaluation_model.md), "Const Binding
  Contract").
- **A scope tree runs on its parent's engine.** Subscopes, spawned
  children, and scoped expressions build on the engine of the kernel
  they are bound under, pure native included, and a parent is any
  `Box<dyn Kernel>`
  ([native_scope_trees.md](../design/native_scope_trees.md) §5).
- **Streams and traversals of one comprehension agree.** Predicates
  parse with the language's precedence, strict zips fail on every
  surface, orders select by position, and a traversal opens in time
  proportional to its shape rather than its tuple count
  ([comprehension_forms.md](../design/comprehension_forms.md) §9.5.2,
  [for_traversal.md](../design/for_traversal.md) §5.2).

## Part 1: what stops compiling

### In Rust

| Change | What breaks | Fix |
|---|---|---|
| `Dataflow::set_wire` and `set_wire_idx` are removed | every call (deprecated since 0.5.0) | `Kernel::set_input` or `set_input_at`, converting first with `polydat::convert::to_port` where the type differs; see the example below |
| `WireKey::describe` is removed | a call to it | name the key yourself; `WireKey` is sealed, so nothing implemented it |
| `RawState` and `PolydatProgram::create_raw_state` are removed | a benchmark baseline built on them | compare against a kernel from `create_kernel` |
| `PolydatKernel::build_subscope` and `Construction::subscope` return `Box<dyn Kernel>` | code that used the result as a `PolydatKernel` | drive the child through `Kernel`; reach interpreter extras with `as_interpreter()` |
| `wrap_root_kernel` takes `Box<dyn Kernel>` | a call passing a `PolydatKernel` | `wrap_root_kernel(Box::new(kernel), label)`, or pass any engine's kernel |
| `ScopeKernel::lock_inner` guards a `Box<dyn Kernel>` | interpreter-only calls through the guard | `Kernel` methods, or `guard.as_interpreter()` |
| `ScopedExpr::bind` takes `&dyn Kernel`, and `ScopedExpr::dataflow()` is renamed `kernel()` and returns `&mut dyn Kernel` | a call to `dataflow()`, and code that used its `PolydatKernel` | `kernel()`, through `Kernel` |
| `TupleStream::advance`, `CoordinateStream::advance`, and `ScopedKernelStream::advance` return `Result<Option<_>, RuntimeError>`, and the streams iterate `Result` items | every `advance` loop and `for` loop over a stream | propagate or handle the error; see the example below |
| `TupleStream` gains the required method `rewind` | a host's own `TupleStream` implementation | implement `rewind` to restart at the first tuple |
| `Strategy` requires `select`, which returns a `Selection` of positions; `apply` and `apply_seeded` are provided | a host's own `Strategy` implementation | implement `select` from the input's `IndexFn`, count, truncation, and seed ([comprehension_forms.md](../design/comprehension_forms.md) §3.6) |
| `KernelProgram` requires `create_uninitialized`; `create_kernel` is provided and initializes | a host's own `KernelProgram` implementation | implement `create_uninitialized`, returning the kernel without calling `init` |
| `Op::OrderMaterialize` gains `input` and `Op::Zip` gains `operands` | a pattern or literal naming every field of either | end the pattern with `..`; `OrderMaterialize` now holds its input and pops nothing from the stack |
| New variants: `KernelError::ConstInit`, `AssemblyError::ConstInit`, `AssemblyError::NativeCone`, `WriteError::ConstSlot`, `WriteError::FromParent`, `ContractViolation::Bind`, `RuntimeError::ZipLengthMismatch`, `ValidationError::PredicateContextRequired`, `InputKind::Const` | an exhaustive `match` on any of these enums (E0004) | add the arm or a wildcard |
| The node types `SessionStartMillis` and `ElapsedMillis` are removed | code that named them | a const capture; see the next table |
| `polydat_grammar::PragmaSet` loses its `parent` field, `attach_to`, and `PragmaConflict` | code that chained pragma sets by hand | `PragmaSet::nested` builds a nested scope's set from its enclosing one ([polydat_grammar.md](../design/polydat_grammar.md) §14.1) |
| `#[polydat_node]` refuses a signature that mixes a const list (`Const<Vec<C>>`) with a wire variadic (`&[T]`), and an unknown `from = (…)` setup source | a host node written that way, which the macro accepted before | split it into two nodes, or pass the constants as individual `Const` arguments ([library_catalog.md](../design/library_catalog.md), "Shapes") |

A healing write becomes a conversion the host asks for:

```rust
// 0.5.0
kernel.set_wire("limit", Value::Str("10".into()))?;

// 0.6.0
let ty = kernel.input_port_type("limit").expect("limit is an input");
let value = polydat::convert::to_port(Value::Str("10".into()), ty)?;
kernel.set_input("limit", value)?;
```

An input whose type varies over the kernel's life is opened with
`CompileOptions::input_variance` instead
([input_variance.md](../design/input_variance.md) §4).

A scope tree takes any engine's kernel:

```rust
// 0.5.0
let child: PolydatKernel = parent.build_subscope(matter)?;
let root = wrap_root_kernel(parent, "root");
let program = root.lock_inner().program().clone();

// 0.6.0
let child: Box<dyn Kernel> = matter.build_under(&parent)?;
let root = wrap_root_kernel(Box::new(parent), "root");
let program = root
    .lock_inner()
    .as_interpreter()
    .map(|k| k.program().clone());
```

A stream reports the error that ends it:

```rust
// 0.5.0
while let Some(tuple) = stream.advance() { use_tuple(tuple); }
for tuple in coordinate_stream { use_tuple(tuple); }

// 0.6.0
while let Some(tuple) = stream.advance()? { use_tuple(tuple); }
for tuple in coordinate_stream { use_tuple(tuple?); }
```

An iterator yields the error once and then ends; `CoordinateStream`
returns it again on every later `advance`.

### In a Polydat program

| Change | What breaks | Fix |
|---|---|---|
| `shared const` is a parse error, as `const volatile` is | a binding with both modifiers, in either order | `const` for a value fixed for the kernel's life; `shared x := <expr>` for a register with a computed start ([polydat_grammar.md](../design/polydat_grammar.md) §5.1) |
| A const that reads a coordinate is refused on all four engines | such a const; the compiled engines accepted it before | read an extern or another const, or drop `const` ([evaluation_model.md](../design/evaluation_model.md), "Compilation of a const") |
| `session_start_millis()` and `elapsed_millis()` are removed | a call to either (unknown function) | `const session_start := current_epoch_millis()` in the root scope, read in children through `extern session_start: u64`; elapsed time is `current_epoch_millis() - session_start` in a volatile binding |
| `is_stable` takes a window: `is_stable(samples: vec_f64, margin, min_samples)` | the 0.5.0 form `is_stable(value, margin, min_samples, horizon)` | keep the recent samples in the host and pass them, for example as JSON through `str_to_vec_f64` ([evaluation_model.md](../design/evaluation_model.md), "Non-Deterministic Nodes") |
| A predicate compiled without a scope may name only what its tuples bind | a coordinate stream whose `where` names an enclosing wire, now `ValidationError::PredicateContextRequired` | traverse it with `for`, which captures those names when it opens |
| `order reverse_lex` over a filter is refused on streams, as it was on traversals | a stream of such an order (V4) | order before filtering, or use `lex` ([comprehension_forms.md](../design/comprehension_forms.md) §5, V4) |
| Under `pragma strict_values`, a constant that violates the constraint of the port it feeds fails the build | a program such as `mod_wire(cycle, 0)`, which compiled before because a constant source was skipped unchecked | pass a value the constraint accepts; the error names the port, the constraint, and the constant ([graph_compiler.md](../design/graph_compiler.md) §2.3) |

## Part 2: what changes quietly

These compile unchanged and behave differently. Each is a correction or
a rule the specifications now state and every engine follows.

### Consts, volatiles, and shared registers

- **A const is evaluated once, when its kernel is initialized, and is
  fixed for the kernel's life.** In 0.5.0 the interpreter refused a
  const over an extern and pulled each const once at scope activation.
  Now a const may read externs, other consts, iteration externs, and
  volatile expressions, and reads them as they are at initialization.
  With `extern base: u64 = 20` and `const limit := u64_add(base, 32)`,
  writing `base` leaves `limit` at 52 until the host calls
  `Kernel::init` again
  ([evaluation_model.md](../design/evaluation_model.md), "Const
  Binding Contract").
- **A failing const fails initialization.** `Kernel::init` returns
  `KernelError::ConstInit` naming the const, the interpreter's build
  returns `AssemblyError::ConstInit`, and `KernelProgram::create_kernel`
  panics with the same message. In 0.5.0 the scope printed a warning and
  the const read `None`. On pure native a const whose value is `None`
  after initialization is refused with `KernelError::Refused`.
- **Writing a const's slot is refused.** `set_input` naming
  `__const_<name>` returns `WriteError::ConstSlot`; to change a const,
  write the inputs it reads and call `init()`.
  `PolydatKernel::scope_values` leaves const slots out, so a host that
  copies a scope's values into another kernel does not meet that
  refusal.
- **A const whose value is known at build folds.** A const whose cone
  reads no input and no nondeterministic node, such as
  `const width := u64_mul(4, 8)`, folds with the other constants and
  does not appear in `const_inits()`.
- **A const over a nondeterministic read acknowledges it.**
  `const session_start := current_epoch_millis()` compiles without the
  unacknowledged-read warning, and under strict mode without an error.
- **A volatile step runs again on every read that reaches it**, on all
  four engines, where 0.5.0 ran it once per write. The most upstream
  volatile node is the fulcrum; it and everything downstream run on
  every `pull` and `eval` that reaches them, and everything upstream
  stays cached. Two readings that must come from one instant belong in
  one node that returns both
  ([runtime_model.md](../design/runtime_model.md) R1.v).
- **`shared x := <expr>` takes any expression as the starting value.**
  A literal stays the slot's default. Any other expression is evaluated
  once at the declaring kernel's initialization and written only while
  the register's cell is unwritten, so an attached child, a host that
  attached a cell, and a second `init` never seed it again
  ([evaluation_model.md](../design/evaluation_model.md), "Shared
  registers and their starting value").
- **A scope lookup answers only for values fixed for the kernel's
  life.** Through `KernelLookup`, the last tier, the folded value,
  answers for const outputs and outputs folded at build. In 0.5.0 it
  answered for any output that held a value, so after a pull a
  per-cycle output resolved as a scope value
  ([expression_engine.md](../design/expression_engine.md) §3.2.2).

### Writes and scope trees

- **A binder copy the child cannot take fails binding.** A parent value
  copied into a child input the child declares with another type is
  refused with `KernelError::Write(WriteError::FromParent { slot,
  expected, got })`, which `spawn` and `build_under` return as
  `ContractViolation::Bind`. In 0.5.0 the binder converted the value, or
  skipped the copy and left the input unset. A parent with
  `const limit := "10"` and a child with `extern limit: u64` now fail to
  bind; declare the child's input as `str`, or bind `limit` to a `u64`
  in the parent. An input whose type the compiler inferred still takes
  a value the adapter catalog converts
  ([input_variance.md](../design/input_variance.md) §7).
- **No write converts.** With `set_wire` gone, the write-through commit
  and every binder copy are written under the host-write rule. The only
  conversions are `convert::to_port`, which the host calls, and the
  converter nodes `CompileOptions::input_variance` inserts
  ([input_variance.md](../design/input_variance.md) §2).
  `ScopedExpr::set` still converts through `to_port`, and ignores a
  value that does not convert and a coordinate.
- **Subscopes and spawned children run on the parent's engine.** In
  0.5.0 they were interpreter kernels whatever the parent was. A parent
  built with `Engine::default()` now has native children, and program
  matter keeps the engine its program was compiled for
  ([native_scope_trees.md](../design/native_scope_trees.md) §5).

### Engines

- **Pure native has broadcast output cells.** A child bound under a pure
  native parent reads a computed output as a live link, as it does
  under the other three engines
  ([engines.md](../design/engines.md) §3.6).
- **An unset extern answers for exactly the outputs that depend on it,
  on all four engines.** Pure native refused every pull while any
  extern its native code reads was unset; it now refuses only a pull
  whose output depends on one. Fusion units split where their extern
  dependencies differ, so native and the interpreter's cones no longer
  answer `None` for an output that shares a unit with an extern's
  reader. With `t := hash(x)`, `a := hash(t)`, and
  `b := u64_mul(t, mode)`, clearing `mode` leaves `a` served on every
  engine ([engines.md](../design/engines.md) §3.3). A report of a
  program's nodes can change where a unit splits, as the `for` body
  example in [Illustrations](../tutorials/illustrations.md) shows.
- **The dataset nodes run on all four engines.** `dataset_open`,
  `dataset_group_open`, `dataset_prebuffer`, and every accessor have a
  kit, so a program that uses them builds on the default engine, where
  0.5.0 refused it by the node's name
  ([engines.md](../design/engines.md) §8).
- **A prebuffered handle answers as an opened one.**
  `metadata_content_count` answered 0 on a prebuffered handle, and the
  group accessors (`dataset_distance_function`, `dataset_facets`, the
  profile enumerators, `profile_partitions`,
  `matching_profile_name_at`) panicked on one. Every accessor now gives
  the same answer on either handle.
- **A cone that cannot be built natively is recorded.** Under
  `JitMode::Auto`, a cone whose native code fails to build stays on the
  interpreter, as before, and the program's `CompileLedger` now records
  it (`cone_fallbacks()`: the members, the outputs they produce, the
  input count, and the error). Under `JitMode::Force` the compile fails
  with `AssemblyError::NativeCone` instead of falling back
  ([engines.md](../design/engines.md) §2.1).
- **A cone reading more than 64 inputs runs natively in pieces.** Such a
  cone was left interpreted; it is now cut into convex pieces of at
  most 64 inputs, each compiled natively. A single node reading more
  than 64 inputs stays interpreted and is recorded on the ledger
  ([engines.md](../design/engines.md) §2.2).
- **A cone group that is not convex runs natively in pieces.** A group of
  eligible nodes that a path leaves through an interpreted node and
  re-enters was left interpreted; it is now split into the same convex
  pieces native code forms from it, each built as a cone of its own
  ([engines.md](../design/engines.md) §2).

### Comprehensions and traversals

- **Predicates use the language's precedence.** Unary `!` binds tighter
  than every binary operator, comparison tighter than `&&`, and `&&`
  tighter than `||`, on every surface. `where !{done} || {retry}` now
  means `(!{done}) || {retry}`; 0.5.0 read it as `!({done} || {retry})`.
  `where !{k} > 3` is `(!{k}) > 3`
  ([polydat_grammar.md](../design/polydat_grammar.md) §6.1).
- **In a predicate, `!` is logical and `&&` and `||` short-circuit**,
  stopping at the operand that decides the result, so filters the
  optimizer folds into one conjunction evaluate what the chain did.
  Outside predicates nothing changes: in an ordinary expression `!` is
  bitwise NOT and `&&` and `||` evaluate both sides
  ([polydat_grammar.md](../design/polydat_grammar.md) §6.3, §7).
- **A bare word in a predicate is one identifier.** `{region} == us-east`
  parses as `us - east`; quote it: `{region} == "us-east"`. Bare words
  without a hyphen, such as `load`, still compare as text.
- **A stream evaluates predicates and orders as a traversal does.**
  `"s0" != 2` holds on a stream, a predicate that calls a function
  filters instead of passing every tuple, and an order over a continuous
  space samples it instead of dispensing nothing.
- **A strict zip fails on a stream when its operands end apart.** It
  yields the tuples before the mismatch and then
  `RuntimeError::ZipLengthMismatch`, naming each operand's length. In
  0.5.0 a stream stopped quietly at the shortest operand; the traversal
  already failed, and now returns the same variant in place of an
  `UnsupportedShape` message
  ([comprehension_forms.md](../design/comprehension_forms.md) §3.3).
- **An empty operand empties a cycle zip** on every path, whether it is
  empty as written, after a filter, or as a generator that yields
  nothing. The traversal used to yield the longest operand's count with
  the empty operand's names missing.
- **Orders over unions, zips, and products of products select across
  the whole input.** `halton`, `sobol`, and `lhs` over a zip or union
  drew only within the first operand, `shuffle` over a cycle zip emitted
  nothing, and `lex`, `diagonal`, and `shells` over a union panicked.
  An independent product of products was refused by every strategy but
  `lex`, and is now accepted. An order over an empty discrete input
  yields nothing.
- **Cardinality reports combine by kind.** A product or truncating zip
  with a filtered operand reports at most, not exactly:
  `cartesian(k in 1..10 where {k} > 3, c in a,b)` reports at most 18
  and yields 12. A traversal's `len` is the count its evaluation keeps
  at open, never the metadata's bound
  ([comprehension_forms.md](../design/comprehension_forms.md) §6.1).
- **A traversal opens in time proportional to its shape.** Opening
  evaluates every source, so every source error still surfaces there,
  and then addresses tuples by position instead of materializing them.
  `order halton/100` over a large product computes 100 tuples, not the
  product
  ([for_traversal.md](../design/for_traversal.md) §5.2, "Opening
  cost").

### Pragmas

- **A pragma applies to the scope it is written in and every scope
  nested in it.** A `pragma` inside a `for` body was parsed and ignored;
  it now applies to that body. A module's pragmas were added to the
  program that imports the module, so importing a strict module made
  the whole host strict; they now apply to the module's own bindings
  only ([polydat_grammar.md](../design/polydat_grammar.md) §14.1).
- **`strict_values` guards a nondeterministic source at run time.** A
  source with no wire inputs was treated as a constant and never
  checked. A constant is now checked at build (Part 1), and a
  nondeterministic or volatile source, such as `counter()`, gets the
  runtime assertion every other wire gets
  ([graph_compiler.md](../design/graph_compiler.md) §2.3).

### Release requirements

- **The facade requires the newest published version of each internal
  crate.** Updating `polydat` moves a host's lockfile to every crate the
  release publishes; in 0.5.x a host could keep an older `polydat-core`
  under a newer facade ([releasing.md](releasing.md)).

## Part 3: new capabilities

- **`Kernel::init`** evaluates every const again from the inputs as they
  are now, and seeds computed `shared` starts that are still unwritten.
  `Kernel::const_inits` lists the captured consts in the order `init`
  evaluates them; each `ConstInit` record names the const, its slot,
  its source output, its fallback input, and whether it is a register
  start. A host reads records but cannot construct one.
- **`PolydatMatter::build_under(&dyn Kernel)`** binds source, statement,
  or program matter under a parent of any engine.
- **`Kernel::as_interpreter` and `as_interpreter_mut`** reach the
  interpreter's own extras (`Metadata`, `Dataflow`, its program and
  state) from a boxed kernel, and return `None` on a compiled engine.
- **Handle ports and path-qualified types in node kits.** The node macro
  gives a node with `Arc<T>` handle arguments or returns a kit, and
  recognizes `Ext<T>` and `Arc<T>` however their path is qualified, so a
  host's own nodes written that way run on the compiled engines
  ([compiled_handles.md](../design/compiled_handles.md)).
- **`str_to_vec_f64`** turns a JSON array of numbers into a `vec_f64`,
  such as the sample window `is_stable` reads.
- **Comprehension building blocks:** `parse_predicate` and
  `CompiledPredicate` evaluate a predicate as every surface does;
  `evaluate_indexed` and `IndexedTuples` address a comprehension's
  tuples by position; `Strategy::select` and `Selection` give a
  strategy's order as positions; `cycle_length` gives a cycle zip's
  count.
- **Source factories are specified.** `rewind_for_poll` restarts a
  source between polling phases; the range source rewinds to its
  start, and the extending range to its start and base chunk. A rewind
  is safe while other threads read the source
  ([cursor_partitions.md](../design/cursor_partitions.md) §8).

## Not breaking, though it looks it

- **`Kernel` gained `const_inits`, `init_input_at`, `init`,
  `as_interpreter`, and `as_interpreter_mut`.** The trait is sealed, so
  nothing outside polydat implements it, and a caller only gains
  methods.
- **`PolydatMatterBuilder::program` takes `Arc<dyn KernelProgram>`.** An
  `Arc<PolydatProgram>` argument coerces, so existing calls compile; the
  child now runs on the engine the program was compiled for, and a
  program of any engine is accepted.
- **`pragma strict_types` is accepted and has no effect.** Statically
  typed wires already make every resolved wire's type match its sink, so
  a runtime type assertion would never fire; a program that sets the
  pragma compiles to the same graph as one that does not
  ([graph_compiler.md](../design/graph_compiler.md) §2.3).
- **A wire type mismatch inside a node reads differently.** The panic
  `Wire::extract` raises when a compiled wire's value does not have the
  node's argument type now names both types
  (`derive_support::extract_mismatch`), and the engine's failure report
  names the node. The compiler's typing keeps it from firing; only the
  text of a report changes.

## Known issues

- **The known issue of 0.5.0 is resolved.** The scope binder no longer
  converts a copy into a declared input, and the write-through commit no
  longer widens a numeric type.
- **`ScopedExpr::set` ignores a value it cannot write.** A name the
  expression does not read, a coordinate, and a value that does not
  convert are dropped without an error.

## Pending for 0.6.0

**Pending.** The items below are in flight and are not part of the tree
this guide was written against. Each bullet is completed, or removed,
when its change lands.

- Optimizer rules: O1 restricted, validation before rewrite, R5 pushdown
  only for total predicates, and an order over a filter ranks the
  survivors.
- Binding attribution moves from `compile_ctx::current_binding()` to the
  construction context.
- The resource accessor moves from a process global to the kernel tree.
- `ScopedExpr::set` returns a `Result`.
- Extern defaults are typed.
- Named generators are specified.

## Checklist

1. Update `polydat`, and confirm with `cargo tree -i polydat-core` that
   every internal crate resolved to the version this release published.
2. Apply the Rust table in Part 1 until every crate compiles: the
   `set_wire` replacements, `Box<dyn Kernel>` in scope trees, and the
   fallible stream advances are the common ones.
3. Compile every Polydat program the host ships, and apply the program
   table in Part 1: `shared const`, consts over coordinates, the removed
   session nodes, and `is_stable` fail there.
4. Read Part 2 against the host. A const over an extern the host writes
   after creating the kernel, a volatile read the host relied on being
   cached within a write, a binder copy between differently typed
   scopes, a `where` predicate that starts with `!` or compares to a
   hyphenated word, and a stream over a strict zip are the ones that
   change a result.
