---
type: specification
title: The for Construct
timestamp: 2026-09-25
description: "Comprehension producers and traversal scopes in the grammar: typing, compilation, one program per lexical position, affine activation, and cursor narrowing."
tags: [iteration, language, scopes]
---

# The for Construct

This document specifies the `for` construct: its grammar, its two
readings (a producer that binds a comprehension as a value, and a
traversal that runs a body once per tuple), how element names are typed,
how bodies compile, and the runtime that opens a traversal and runs its
body for each tuple on the host's chosen engine.

**Ownership:** Polydat owns the grammar, the compiled form, the activation
runtime, and the consumption surfaces defined here. Hosts own scheduling
policy, side effects, and any surface sugar over the canonical text form.

**Related specifications:**
[Comprehension Forms](comprehension_forms.md) (the algebra),
[Cursor Partitions](cursor_partitions.md) (partition values and `over`),
[Scope Model](scope_model.md) (activation, materialization, coordinates),
[Runtime Model](runtime_model.md) (the D-axioms),
[The Polydat Grammar](polydat_grammar.md) (productions this document extends),
[Engines](engines.md) (the engines an activation runs on).

## 1. Scope

A Polydat kernel is a pure function of its coordinates. The `for`
construct lets a program name a traversal over a comprehension and lets
the runtime run it: Polydat parses and evaluates the comprehension, binds
each tuple into a child scope, resolves every `over` clause, writes the
cursor slots, and positions each cycle. The host only schedules the work
and consumes the outputs.

The construct is one keyword, `for`, with two readings that share one
compiled form:

- **Producer.** `name := for ...` binds a comprehension as a value. The wire
  has type `Streamer`. Derived comprehensions compose on it.
- **Traversal.** `for ... { body }` with no l-value runs the body once per
  tuple, each time in a new child scope called an **activation** (§3.2).
  The comprehension's element names are wires inside the body.

A traversal is the producer's tuple stream with the body activated at each
tuple. Both readings are deterministic enumerations, so the D-axioms hold
across both.

## 2. Grammar

Extends [polydat_grammar.md §21](polydat_grammar.md#sec-productions).

```ebnf
statement      ::= ...existing...
                |  for_stmt

binding        ::= modifier* ident ":=" expr
                |  ...existing...

expr           ::= ...existing...
                |  for_expr

for_expr       ::= "for" comprehension_text

for_stmt       ::= "for" for_source "{" statement* "}"

for_source     ::= comprehension_text        (* inline comprehension *)
                |  ident                     (* a bound producer *)

comprehension_text ::= clause ("," clause)* ("where" predicate)? ("order" order_spec)?
                |  "[" ("for" comprehension_text ","?)+ "]" ("where" predicate)? ("order" order_spec)?   (* union *)
order_spec     ::= strategy ("/" int)?                          (* bare or terse *)
                |  strategy "(" (ident "=" value) ("," ident "=" value)* ")"   (* keyword; `seed=` for shuffle and lhs *)
clause         ::= ident "in" source
                |  "(" ident ("," ident)+ ")" "in" source
```

`comprehension_text` is the canonical text form accepted by the
comprehension parser. `source` is the comprehension's source surface:
literal lists, `lo..hi` ranges, generator calls such as
`partitions("*/4", n)` and `subdivide(p, n)`, string comprehensions, and
continuous sources: a float range or a named measure such as
`normal(0, 1)`, optionally restricted with `on <interval>`
([Polydat Grammar](polydat_grammar.md) §16.2).
`predicate` and `strategy` are the algebra's. The grammar adds no new
source forms.

Examples:

```text
sweep := for k in 1..4, limit in 10,20,30 order halton/5
edges := for sweep where {k} == 1 || {k} == 3

for sweep {
    f := myfunc(k)
    g := otherfunc(limit, k)
}

for phase in load,verify, p in partitions("*/4", 1000000) {
    cursor rows = range(0, 1000000) over p
    row := mod_in(cycle, rows.cursor)
    stmt := select_str(str_eq(phase, "load"), load_stmt, verify_stmt)
}
```

`for` is a hard keyword. Programs cannot use it as an identifier, because
the comprehension parser already reserves it.

The lexer captures the comprehension text as one token: everything after
`for` up to whichever comes first of the end of the line at bracket depth
zero or a `{` at bracket depth zero, with string literals skipped whole.
A `{name}` coordinate reference inside a `where` predicate is part of the
text, since a block brace is never immediately followed by an identifier
and a closing brace. A traversal's `{` therefore opens on the same line
as its comprehension. The captured text is handed to the comprehension
parser unchanged, so the comprehension grammar is defined in one place.

## 3. Semantics

### 3.1 Producer

A `for` expression evaluates to a compiled comprehension. The binding is
`const`: the value is computed once at scope init and never re-evaluated
within the scope. The wire type is `Streamer`. A `Streamer` holds the
comprehension AST after validation and optimization, plus the metadata
the algebra computes: tuple shape, cardinality class, and index
addressability.

Derived forms apply the algebra's modifiers to a bound producer. They
begin with `for` like every other comprehension value, so every
comprehension is introduced by the same keyword and the lexer captures
its text the same way:

```text
base     := for k in 1..100, limit in 1..100
boundary := for base where {k} == 1 || {k} == 100
sampled  := for base order halton/50
```

A derivation is resolved at compile time against the producer bound
earlier in the same scope: the filter and the order are applied to the
base producer's comprehension AST (`resolve_source` in
`polydat-core/src/dsl/traversal.rs`). Derivations therefore chain, a
derivation of a derivation is a comprehension like any other, and a
traversal over a derivation resolves its element names from the base. A
producer expression is comprehension text or a derivation. A bare
producer name alone (`x := for sweep`) is an error, since only a
traversal may name a producer without modifying it.

Each derived wire is a distinct comprehension, and streams obtained from
them never share dispense state (the position of the next tuple to
produce). This is the independence contract in
[Comprehension Forms](comprehension_forms.md) §9.5.2, applied to wires.

**Realization.** `Streamer` uses the type system's extension type: the
wire's port type is `Ext` and its value is a reflected `StreamerValue`
whose type name is `Streamer`. It holds the text and the resolved
algebra AST and exposes `compiled`, `coordinate_stream`, `metadata`, and
`cardinality`. The compiler lowers `name := for ...` to
`const name := streamer("<payload>")`, where `streamer` is an ordinary
registered node whose payload is the resolved comprehension. Authors may
call `streamer` with comprehension text directly. No contract in this
document needs a dedicated port-type variant; the reflected type name is
the wire-level tag.

A producer's sources may reference wires of the enclosing scope through
the comprehension grammar's reference form, `{name}`, as in
`partitions("*/4", {total})`. The reference is resolved against the
enclosing kernel's current values when the traversal is opened, so a
host can change an extern and re-open the traversal without recompiling.
Every wire a source references is read at that moment, whatever its
provenance: a coordinate input as much as an extern, taken from the
enclosing kernel's values when the traversal opens, even where the body
declares the same name. The tuple set is materialized at open, so a later
change to a referenced wire does not affect a traversal that is already
open; the next open reads the new values.

### 3.2 Traversal

A `for` statement declares a traversal scope. The body is a statement list
compiled as a child program. For each tuple the comprehension dispenses,
the runtime creates one **activation**: a fresh kernel over the child
program, with the tuple's values bound to the body's inputs.

Inside the body:

- **Element names are wires.** Each comprehension variable is an input slot
  of kind `IterationExtern`, typed from the source. The full type plane is
  allowed: `u64`, `f64`, strings, booleans, `Ext` values such as
  `Partition`, and `Streamer` when a source yields comprehensions.
- **Every statement kind is allowed,** including cursors, `const`
  bindings, module calls, tiles, nested `for` statements, and nested
  producers.
- **Outer wires are visible** under the visibility rules in
  [Scope Model](scope_model.md) §5. The body is a child; the enclosing
  scope is its parent. Every outer wire the body references, and every
  outer wire the comprehension's own sources reference through `{name}`,
  becomes a **cascade extern** of the child program: an extern input,
  typed from the parent, whose value is copied from the parent when the
  traversal opens (§5.2).
- **Modules are visible.** The body sees every module the parent had
  resolved when the body was lowered, the parent's own definitions
  included, on whichever engine the body compiles.
- **Outputs are the body's named bindings.** A host reads them by
  pulling from the activation's kernel.

A traversal over a bound producer, `for sweep { ... }`, is identical to a
traversal over the producer's inline text. The body sees the producer's
element names. Traversing the same producer twice creates two independent
streams.

### 3.3 Typing of element names

Element types are determined at compile time from the source:

| Source form | Element type |
| --- | --- |
| Integer literal list or `lo..hi` range | `u64` |
| Float literal list or continuous interval | `f64` |
| String literal list or string comprehension | `Str` |
| Boolean literal list | `Bool` |
| `partitions(...)`, `subdivide(...)`, `<name>.partitions` | `Ext` carrying `Partition` |
| A generator node call | The node's declared return type |
| A generator expression naming a `Streamer` wire | `Ext` (the probe's type) |

An integer among floats widens the element to `f64`; a list that mixes
numbers and strings is a compile error. A generator call is typed by
compiling a one-binding probe program, `__probe := <expr>`, with the
parent's library paths and module cache, and reading the binding's port
type; the same probe types projection sources in tiles. The body is
type-checked against these types before any activation exists, so a
mismatch is reported once, at compile time, with the source span of the
offending clause.

### 3.4 Cursors and cycles inside a traversal

A body may declare cursors. An `over <name>` clause may name an element of
the enclosing comprehension. At activation, Polydat resolves the `over`
value to a `Partition`, narrows the cursor to that interval (restricts
the ordinals it iterates to the partition's `[start, end)`), and records
that interval, the **slice**, on the activation.

A **cycle** is one evaluation position within an activation, written to
the body's implicit `cycle` input. The number of cycles an activation has
is:

- **A body with one or more cursors** iterates its narrowest cursor
  extent. Each ordinal in the slice is one cycle. The cursor's ordinal is
  written before each pull.
- **A body with no cursor** has exactly one cycle per activation. The
  tuple is the complete coordinate.

A traversal therefore has a two-level coordinate: the tuple selects the
activation, and the cursor ordinal selects the cycle. Both are pure
functions of position, so any activation and any cycle can be
regenerated in isolation, and activations can be distributed across
fibers without coordination.

### 3.5 Nesting

A `for` statement inside a body declares a grandchild traversal. Its
comprehension may reference the parent tuple's elements, so
`for inner in subdivide(p, 4) { ... }` inside `for p in partitions(...)`
is the nested-partition form the cursor partition specification names.
Coordinate paths compose root-first, matching
[Scope Model](scope_model.md) §7.

### 3.6 Consumption surfaces

Polydat offers a host three ways to consume a traversal, extending
[Comprehension Forms](comprehension_forms.md) §9.5:

```text
CoordinateStream        tuples only
ScopedKernelStream<K>   one bound kernel per tuple
TraversalStream         one Activation per tuple

Activation<K> {
    index:   u64                   // position in the dispense order
    coords:  Vec<(String, Value)>  // the tuple, in element order
    kernel:  K                     // a fresh kernel over the shared program
    cursor:  Option<CursorSlice>   // narrowed extent, if the body has one
}
```

`TraversalStream::advance` returns the next activation, or `None` after
the last. `Activation::cycle(i)` positions the activation's kernel at
cycle `i` under the rule in §3.4: it writes `i` to the `cycle` input and,
when the body has a cursor, the absolute ordinal `slice.start + i` to the
cursor's ordinal slot. It returns the kernel ready to pull.
`for_each_cycle(f)` calls `cycle(i)` for every cycle in order and passes
`i` and the kernel to `f`. A host that wants only tuples uses the first
surface; a host that wants to schedule work uses the third.

A `Streamer` wire pulled from a kernel exposes `compiled` and
`coordinate_stream`, so a host holding a compiled program can enumerate
any producer it names; every call returns a fresh stream with its own
dispense state.

## 4. Compilation

1. **Parse.** `for_expr` produces `Expr::For(Box<ForSource>)`. `for_stmt`
   produces `Statement::For(ForStmt)`, where `source` is
   comprehension text parsed to the algebra, a producer reference, or a
   derivation.
2. **Strip.** The compiler removes every `for` statement from the parent
   file and rewrites every producer binding to
   `const name := streamer("<payload>")`, resolving derivations against
   the producers bound before them. The parent compiles without the
   bodies' wires.
3. **Element typing.** Each clause source is typed per §3.3.
4. **Body compilation.** The body compiles to a child program exactly as
   a module body does, from a child file that declares an implicit
   `cycle` input, one `IterationExtern` per element, and one cascade
   extern per outer wire the body or the comprehension references and
   the parent exposes. A body may not declare an input other than
   `cycle`. This happens once, at parent compile time, on the
   interpreter; the child program is stored on the parent program keyed
   by the `for` statement's lexical position, per §5.1.
5. **Body source.** Beside the interpreter's program the parent keeps the
   body as it lowered it (`BodySource` in `polydat-core/src/dsl/traversal.rs`): the
   child file, its source text, and the compiler settings the parent
   used, which are the source directory, the library paths, strictness,
   the diagnostic context label, the cursor limit, the pragma set, and
   the parent's resolved modules. The body compiles on any other engine
   (the closure tier, native, or pure native) from this record, with the
   same settings, on the first request for that engine (§5.2).

The body's cursors compile as cursors do anywhere, allocating the `over`
slots. No node type exists for cursor narrowing; the runtime writes the
existing slots.

A body's compile error reports the `for` source text and the statement's
line and column, followed by the child compiler's diagnostic. Tile events
raised inside a body are recorded in the parent's compile log.

## 5. Runtime

### 5.1 One program per lexical position

A body's compiled program is determined by where the `for` body appears
in the program text, not by the tuples it runs for. The body's wiring,
types, lifecycle classification, fusion, and native code are the same for
every tuple, because a tuple only supplies values for input slots the
program already declares.

Polydat therefore compiles each `for` body exactly once per engine and
caches the program keyed on the body's lexical position. The cache key
is the position rather than the tuple because a body's program does not
depend on its coordinates.

For nesting, this means a traversal of any depth compiles one program per
level:

```text
for p in partitions("*/4", 1000000) {            // program A, compiled once
    for tenant in 0..20 {                        // program B, compiled once
        for device in 0..50 {                    // program C, compiled once
            ...
        }
    }
}
```

Three programs exist for the life of the root program, however many
tuples the traversal dispenses. The 4 × 20 × 50 = 4000 innermost
activations share program C. Each activation is a fresh kernel over that
program with its own tuple bound.

A host can verify this. Every program built for a tree (the root program
and its bodies at every depth) is recorded in the tree's `CompileLedger`,
obtained through `ledger()` on the kernel or the program, and
`program_count(program)` counts the root plus one program per body at
every depth. Two trees never share a ledger, whatever thread or process
runs them; a host that wants several trees on one ledger passes it in
`CompileOptions::ledger`.

### 5.2 Affine activation

Activation never compiles. It clones the program `Arc`, creates a nested
kernel over it, binds the tuple and the cascade externs, and narrows the
cursors. The first activation of a body on an engine pays that engine's
compilation once; every later activation costs the same constant, which
depends only on the size of the state. This is the affine property:
compile once, then a per-activation cost that is independent of how many
coordinates precede it.

A host may additionally reuse a live activation when it revisits the same
tuple under the same parent, for replay or for a second phase over the
same slice. That is a state-reuse policy, not a compilation concern, and
Polydat leaves it to the host. Correctness never depends on it; the
program cache in §5.1 is what Polydat guarantees.

**Engines.** Opening and activation work the same way on all four engines
(the interpreter, the closure tier, native, and pure native). Pure native
refuses a body containing a node without a native lowering, as it refuses
any such program; `activation_on` then returns an error naming the
engine.

The diagram below shows how a host opens a traversal and then runs its
body once per tuple and once per cycle.

![Flow of a traversal: traverse(index) copies the cascaded and referenced wire values and evaluates the comprehension to its tuples once; then, for each tuple, advance() gets the body program for the engine, creates a kernel, binds the tuple and cascade, and narrows the cursors; then, for each cycle, cycle(i) writes the cycle and cursor ordinal and the host pulls outputs](../diagrams/for_traversal-open-and-activate.png)

- **Opening a traversal.** `traverse(index)` is a method of the `Kernel`
  trait, so a kernel on any of the four engines opens the traversals its
  program declares. `index` selects among the program's top-level `for`
  statements, numbered from 0 in source order; a nested `for` belongs to
  its body's program and is opened from an activation of that body. An
  index with no traversal is an error that states how many the program
  declares. Opening works from the values the kernel holds at that
  moment. First it takes a **snapshot**, a copy of the current value of
  every cascaded wire and every wire a source references, read through
  the trait (`pull` for a wire the parent computes, `input_value` for an
  input or extern). Then it evaluates the comprehension against that
  snapshot alone, through the evaluator's `Lookup` view: the captured
  source references first, then the cascaded wires, over an empty scope
  (`NoScope`) charged to the body program's ledger, with a tuple's own
  elements resolved first as the tuple is built (`Layered` in
  `polydat-core/src/kernel/interp.rs`). Opening builds no kernel state
  for the body, and it needs no kernel of the engine that opened it. `traverse` returns a
  `TraversalStream` holding the tuples, the snapshot, and the opening
  kernel's engine.
- **Running the body for one tuple.**
  `TraversalStream::activation_on(index, engine)` returns the activation
  for tuple `index` on `engine`, without moving the dispense position.
  It compiles the body for that engine once, on the first request, from
  the `BodySource` through the same assembler all four engines use
  (`Traversal::program_on`), and every later activation on that engine
  shares the program, as interpreter activations share theirs: one
  program per engine per position. The activation is a new kernel
  created from that program and driven through the `Kernel` trait:
  elements and cascade externs are written by name through `set_input`,
  cursors are narrowed through one routine (§5.3), and it computes what
  the interpreter's activation computes. `activation(index)` is
  `activation_on` on the stream's engine, the engine of the kernel that
  opened the traversal. `advance()` returns `activation` at the dispense
  position and moves the position forward.
- **Nested traversals.** A body's own `for` statements are compiled with
  the body on every engine it compiles on and are opened from its
  activation, so every level of a nest (the root, the outer bodies, and
  the innermost body where the cycles run) runs on the engine the host
  chose.

**The cascade is a snapshot.** A cascaded wire enters the body as the
value it had when the traversal was opened, bound into the activation's
extern slot. A wire that changes on the parent after that does not change
inside the activations already opened, and re-opening the traversal sees
the new value. This holds for a `shared` wire as for any other: the body
receives the cell's value at open, not the cell. Cells the body's own
`shared` bindings create follow [Scope Model](scope_model.md) §8 within
the activation.

**A state of its own.** An activation is a kernel state like any other,
created with `create_kernel`: it owns its inputs, its outputs, and the
storage behind them, and follows the provenance rules as if it were the
only state ([Runtime Model](runtime_model.md) R4). It does not read or
write the kernel that opened it except through the cells it attaches.

**Opening cost.** Opening a traversal evaluates its comprehension. Ranges
and literal lists evaluate directly. A generator-call source such as
`partitions("*/4", {total})` evaluates through the constant-expression
path, which compiles a one-binding program for the expression text. That
path is cached by text, so the same source text compiles once per
process and every later open is compile-free. Filter predicates in the
comprehension grammar (`{name}` compared to a literal or another element
and joined by `&&`, `||`, `!`, or `in [...]`) evaluate directly against
each tuple with no kernel and no compilation. Only a predicate outside
that grammar takes the kernel path, which is likewise cached by
interpolated text. A source or predicate that does compile is charged
to the ledger of the tree that opened it, so a host can verify these
properties on the kernel it holds.

Hosts may also elide a traversal whose body adds no matter beyond its
parent. Elision is likewise host policy; the runtime provides the
program-identity hashes it needs.

### 5.3 Cursor narrowing

At activation, Polydat processes each cursor in the body through the
`Kernel` trait, so the steps are the same on all four engines:

1. With an `over` clause: resolve the compiled `over` expression on the
   activation kernel. Accept a string spec, a `Partition`, a
   `PartitionSpec`, or a `PartitionList` with exactly one element. A
   list that resolves to no partition, or to more than one, is an
   activation error; inside a traversal the list is bound with an
   enclosing `for p in ...` and the cursor declared `over p`.
2. Resolve against the cursor's extent as the cursor partition
   specification defines, honoring open-extent cursors, and narrow the
   cursor through `set_cursor`.
3. Without an `over` clause: the slice is the cursor's full extent.
4. Record the narrowest slice on the `Activation`. Each cycle writes the
   cursor's ordinal (`<cursor>__ordinal`) before the pull.

The resolution routines (`cursor_over_partitions_on`, `cursor_extent_on`)
are in `iteration::cursor_partition`; the `polydat` binary resolves its
cursors with `cursor_over_partitions_on` and narrows them with `set_cursor`
at fiber setup.

### 5.4 Fibers

Activations are independent. A host may hand consecutive activations to
different fibers, or partition the traversal's index space and give each
fiber a range of tuple indices. Because every strategy is a decidable
permutation, a fiber can seek to its tuple index without dispensing the
tuples before it. Polydat provides `TraversalStream::seek(index)` and
random-access `activation(index)` for this.

## 6. Axioms

- **T1, Traversal determinism.** For a fixed program and parent state, the
  sequence of tuples and the state of every activation at every cycle is
  the same on every run, on every host, and on all four engines. Follows
  from D1 and D4 in the runtime model and from the strategy determinism
  in the algebra.
- **T2, Cost bound.** The work of a full traversal is bounded by the tuple
  cardinality times the per-activation cone cost from D3, plus a constant
  activation cost. Nothing in a traversal is unbounded unless a cursor or
  a comprehension source is declared unbounded.
- **T3, Purity of the construct.** `for` introduces no mutation and no
  ordering dependence between activations. Side effects in a body are
  governed by D2 exactly as in any other kernel.
- **T4, No control flow inside a cycle.** A body is a scope, not a loop
  body. A cycle remains a pure pull. `for` cannot appear inside an
  expression that varies per cycle.

## 7. Boundaries

Not specified here, and deliberately left to hosts:

- when to create activations, how many to keep live, and how to schedule
  them across threads;
- any YAML or command-line sugar over the canonical text form;
- stop conditions, rate control, and metrics;
- side-effect sinks.

Not supported, and reported as compile errors rather than silently
accepted:

- a body that declares an `input` other than `cycle`;
- `over` naming a wire that is not `Partition`-typed at compile time.

## 8. Worked example

A toy test definition
([`examples/toy_test_definition.polydat`](../../examples/toy_test_definition.polydat)
is the full form, with a tile for the document):

```text
extern base_epoch_ms: u64 = 1700000000000

const dataset := "iot-readings-toy"
const schema_stmt := "CREATE TABLE toy.readings (...)"

reading_model(seed: u64) -> (temp_c: f64, humidity: f64, status: String) := {
    temp_c   := normal_sample(input: seed, mean: 21.5, stddev: 2.0)
    humidity := uniform_sample(input: hash(seed), min: 30.0, max: 70.0)
    status   := weighted_strings(hash(hash(seed)), "ok:0.97;degraded:0.02;error:0.01")
}

flow := for phase in load,verify, interval_ms in 1000,60000, p in partitions("*/4", 1000000)

for flow {
    cursor rows = range(0, 1000000) over p
    row := mod_in(cycle, rows.cursor)
    (tenant, device, reading) := mixed_radix(row, 20, 50, 0)

    tenant_id  := hashed_id(input: tenant, bound: 1000000)
    device_key := interleave(tenant, device)
    device_id  := hashed_uuid(device_key)
    (temp_c, humidity, status) := reading_model(hash(interleave(device_key, reading)))
    ts := base_epoch_ms + reading * interval_ms

    load_stmt   := "INSERT INTO toy.readings (...) VALUES ({tenant_id}, '{device_id}', {ts}, {temp_c}, {humidity}, '{status}')"
    verify_stmt := "expect temp_c = {temp_c}, humidity = {humidity}, status = '{status}'"
    stmt := select_str(str_eq(phase, "load"), load_stmt, verify_stmt)
}
```

The traversal has sixteen activations, one per tuple, and each iterates
its 250000-row slice. `phase`, `interval_ms`, and `p` are wires in the
body; `base_epoch_ms` is a cascade extern copied from the parent; and
`reading_model` is visible in the body because the parent defined it.
The complete host code that runs every cycle is:

```rust
let mut traversal = kernel.traverse(0)?;
while let Some(mut activation) = traversal.advance()? {
    activation.for_each_cycle(|_, kernel| execute(kernel.pull("stmt").as_str()));
}
```

The `polydat` binary is that host. Its optional behaviors are graph
transforms, not runtime decorators: a feature that needs to observe a
scope is expressed by inserting a node into that scope, where it has
ordinary wired access to everything the scope can see. Under traversal
the inserted binding goes inside the affected block, so every activation
includes it with the block's element names in scope.
