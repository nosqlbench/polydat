# The `for` Construct — Comprehension Producers and Traversal Scopes

**Status:** Implemented. Proposed SRD 113. All six steps of §11 have
landed: both `for` forms lex, parse, and pretty-print; every traversal
body compiles once into a child program with typed element externs and
cascade externs, keyed by lexical position; producer bindings are
`const` wires carrying a `Streamer` value with independent stream
factories, and derivations filter and order them; the activation runtime
dispenses fresh states over those programs and iterates cycles under the
cursor rule; the one-program-per-position property is measured; and the
toy test definition, the illustrations, and the README use the construct.

**Ownership:** Polydat owns the grammar, the compiled form, the activation
runtime, and the consumption surfaces defined here. Hosts own scheduling
policy, side effects, and any surface sugar over the canonical text form.

**Companion documents:**
[Comprehension Forms](comprehension_forms.md) (the algebra),
[Cursor Partitions](cursor_partitions.md) (partition values and `over`),
[Scope Model](scope_model.md) (activation, materialization, coordinates),
[Runtime Model](runtime_model.md) (the D-axioms),
[Grammar](grammar.md) (productions this document extends).

## 1. The claim

Today a Polydat kernel is a pure function of its coordinates, and every
loop that drives it lives in the host. The comprehension algebra, cursor
partitions, and per-iteration scope materialization all exist in the crate,
but nothing in the grammar can name a traversal and nothing in the runtime
can run one. The host must parse the comprehension, evaluate it, bind each
tuple, resolve every `over` clause, write cursor slots, and advance cycles.
nmbrs does exactly this in its executor.

This document adds one construct, `for`, with two readings that share one
compiled form:

- **Producer.** `name := for ...` binds a comprehension as a value. The wire
  has type `Streamer`. Derived comprehensions compose on it.
- **Traversal.** `for ... { body }` with no l-value activates one child scope
  per tuple. The comprehension's element names are wires inside the body.

The traversal reading is a functor over the producer reading: a traversal
is the image of a coordinate stream under "activate the body at this
tuple". Both readings are deterministic enumerations, so the D-axioms
compose across them.

## 2. Grammar

Extends [Grammar](grammar.md) §2.

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

comprehension_text ::= clause ("," clause)* ("where" predicate)? ("order" strategy ("/" int)?)?
clause         ::= ident "in" source
                |  "(" ident ("," ident)+ ")" "in" source
```

`comprehension_text` is the canonical text form already accepted by the
comprehension parser. `source` is the existing source surface: literal
lists, `lo..hi` ranges, generator calls such as `partitions("*/4", n)` and
`subdivide(p, n)`, and string comprehensions. `predicate` and `strategy`
are unchanged. The grammar adds no new source forms.

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

`for` is a hard keyword. Existing programs do not use it as an identifier
because the comprehension parser already reserved it.

The lexer captures the comprehension text as one token: everything after
`for` up to the end of the line or a `{` at bracket depth zero, whichever
comes first, with string literals skipped whole. A `{name}` coordinate
reference inside a `where` predicate is part of the text, since a block
brace is never immediately followed by an identifier and a closing brace.
A traversal's `{` therefore opens on the same line as its comprehension.
The captured text is handed to the comprehension parser unchanged, so the
traversal grammar has exactly one owner.

## 3. Semantics

### 3.1 Producer

A `for` expression evaluates to a compiled comprehension. The binding is
effectively `const`: the value is computed once at scope init and never
re-evaluated within the scope. The wire type is `Streamer`. A `Streamer`
carries the comprehension AST after validation and optimization, plus the
metadata the algebra already computes: tuple shape, cardinality class, and
index addressability.

Derived forms apply the algebra's modifiers to a bound producer. They are
introduced by `for` like every other comprehension value, so one keyword
opens every comprehension and the lexer owns the text capture:

```text
base     := for k in 1..100, limit in 1..100
boundary := for base where {k} == 1 || {k} == 100
sampled  := for base order halton/50
```

Each derived wire is a distinct comprehension. Streams obtained from them
never share dispense state. This is the independence contract in
[Comprehension Forms](comprehension_forms.md) §9.5.2, applied to wires.

**Realization.** `Streamer` rides the type plane's extension point: the
wire's port type is `Ext` and its value is a reflected `StreamerValue`
whose type name is `Streamer`, carrying the text and the resolved
algebra AST and exposing `coordinate_stream`, `compiled`, `metadata`,
and `cardinality`. The compiler lowers `name := for ...` to
`const name := streamer("<payload>")`, where `streamer` is an ordinary
registered node whose payload is the resolved comprehension. Authors may
call `streamer` with comprehension text directly. A dedicated port-type
variant is not needed for any contract in this document; the reflected
type name is the wire-level tag.

A producer's sources may reference wires of the enclosing scope through
the comprehension grammar's reference form, `{name}`, as in
`partitions("*/4", {total})`. The reference is resolved against the
enclosing kernel's current values when the traversal is opened, so a
host can change an extern and re-open the traversal without recompiling.
A source that depends on a per-cycle wire is an error at that point: a
producer is a scope-init value and cannot vary per cycle.

### 3.2 Traversal

A `for` statement declares a traversal scope. The body is a statement list
compiled as a child program. For each tuple the comprehension dispenses,
the runtime activates one child scope: a fresh state over the child
program with the tuple's values bound.

Inside the body:

- **Element names are wires.** Each comprehension variable is an input slot
  of kind `IterationExtern`, typed from the source. The full type plane is
  allowed: `u64`, `f64`, strings, booleans, `Ext` values such as
  `Partition`, and `Streamer` when a source yields comprehensions.
- **Every statement kind is allowed,** including cursors, `const`
  bindings, module calls, nested `for` statements, and nested producers.
- **Outer wires are visible** under the visibility rules in
  [Scope Model](scope_model.md) §5. The body is a child; the enclosing
  scope is its parent. Parent values reach the body through the existing
  parent-gated materialization.
- **Outputs are the body's named bindings.** A host observes them by
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
| String literal list or string comprehension | `String` |
| Boolean literal list | `Bool` |
| `partitions(...)`, `subdivide(...)`, `<name>.partitions` | `Ext` carrying `Partition` |
| A generator node call | The node's declared return type |
| A bound `Streamer` used as a source | `Streamer` |

Mixed literal lists are a compile error. The body is type-checked against
these types before any activation exists, so a mismatch is reported once,
at compile time, with the source span of the offending clause.

### 3.4 Cursors and cycles inside a traversal

A body may declare cursors. An `over <name>` clause may name an element of
the enclosing comprehension. At activation, Polydat resolves the `over`
value to a `Partition`, narrows the cursor to that interval, and writes the
`<cursor>.cursor` value and its scalar projections. This is the step the
compiler comments today describe as "written by the executor at phase
setup"; it moves into the crate and becomes part of activation.

The cycle rule for an activation:

- **A body with one or more cursors** iterates its narrowest cursor
  extent. Each ordinal in the slice is one cycle. The cursor's ordinal and
  field projections are written before each pull, exactly as
  `inject_into_state` does now.
- **A body with no cursor** has exactly one cycle per activation. The
  tuple is the complete coordinate.

A traversal therefore has a two-level coordinate: the tuple selects the
activation, and the cursor ordinal selects the cycle. Both are pure
functions of position. Any activation and any cycle can be regenerated in
isolation, which is what makes activations distributable across fibers
without coordination.

### 3.5 Nesting

A `for` statement inside a body declares a grandchild traversal. Its
comprehension may reference the parent tuple's elements, so
`for inner in subdivide(p, 4) { ... }` inside `for p in partitions(...)`
is the nested-partition form the cursor partition specification already
names. Coordinate paths compose root-first, matching
[Scope Model](scope_model.md) §7.

### 3.6 Consumption surfaces

Polydat exposes three surfaces, extending
[Comprehension Forms](comprehension_forms.md) §9.5:

```text
CoordinateStream        tuples only                       (exists)
ScopedKernelStream<K>   one bound kernel per tuple        (exists)
TraversalStream         one Activation per tuple          (new)

Activation {
    coords:  ScopeCoord            // the tuple, root-first path available
    kernel:  PolydatKernel         // fresh state over the cached program
    cursor:  Option<CursorSlice>   // narrowed extent, if the body has one
}
```

`TraversalStream::advance` yields the next activation. `Activation::cycles`
yields cycles under the rule in §3.4, writing cursor projections and
returning the kernel ready to pull. A host that wants only tuples uses the
first surface; one that wants to schedule work uses the third.

`Streamer` wires pulled from a kernel expose the same three factories, so
a host holding a compiled program can traverse any producer it names.

## 4. Compilation

1. **Parse.** `for_expr` produces `Expr::For(ComprehensionAst)`. `for_stmt`
   produces `Statement::For { source, body }` where `source` is either a
   comprehension AST or a wire reference.
2. **Element typing.** The compiler evaluates each clause source's type per
   §3.3. Sources that reference outer wires are resolved through the
   auto-extern pass first.
3. **Body compilation.** The body compiles to a child `PolydatProgram`
   exactly as a module body does, with one `IterationExtern` input per
   element and cascade externs for every outer name it references. This
   happens once, at parent compile time. The child program is stored on
   the parent program keyed by the `for` statement's lexical position,
   per §5.1.
4. **Comprehension compilation.** The comprehension AST is validated,
   optimized, and lowered to the IR. The IR is stored alongside the child
   program.
5. **Producer wires.** A `for_expr` binding compiles to a `const` node
   whose value is the compiled comprehension. Derived `where` and `order`
   forms compile to nodes that wrap the parent `Streamer`.

The body's cursors compile as they do today, allocating the `over` slots.
No new node types are required for cursor narrowing; the runtime writes
the existing slots.

## 5. Runtime

### 5.1 One program per lexical position

A compiled program is a property of where a `for` body appears in the
program text, not of which coordinates reach it. The body's wiring, types,
lifecycle classification, fusion, and native code are the same for every
tuple, because the tuple only supplies values for slots the program already
declares. Coordinates vary; the program does not.

Polydat therefore compiles each `for` body exactly once, at parent compile
time, and caches it keyed on the body's lexical position. The cache is not
an optimization over tuple identity. It is the statement that a kernel
program is invariant under its coordinates.

The consequence for nesting is what makes deep traversals cheap:

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
activations share program C. Each activation is a fresh state over that
program with its own tuple bound.

### 5.2 Affine activation

Activation never compiles. It does what `for_iteration` does now: clone
the program `Arc`, allocate fresh state, inject the tuple, and run parent
materialization. The first activation of a body pays compilation once;
every activation after that costs the same constant, which depends only on
the size of the state. This is the affine property: compile once, then a
per-activation cost that is independent of how many coordinates precede
it.

Shared cells and transit cells remain shared across activations per
[Scope Model](scope_model.md) §8. Ordinary buffers and clean flags are
fresh.

A host may additionally reuse a live activation when it revisits the same
tuple under the same parent, for replay or for a second phase over the
same slice. That is a state-reuse policy, not a compilation concern, and
Polydat leaves it to the host. Correctness never depends on it; the
program cache in §5.1 is what Polydat guarantees.

**Opening cost.** Opening a traversal evaluates its comprehension. Ranges
and literal lists evaluate directly. A generator-call source such as
`partitions("*/4", {total})` evaluates through the constant-expression
path, which compiles a one-binding program for the expression text; that
path is cached by text, so the same source text compiles once per
process and every later open is compile-free. Filter predicates in the
comprehension grammar, `{name}` compared to a literal or another element
and joined by `&&`, `||`, `!`, or `in [...]`, evaluate directly against
each tuple with no kernel and no compilation; only a predicate outside
that grammar takes the kernel path, which is likewise cached by
interpolated text. `kernel::programs_built()` exposes the process-wide
build count so a host can verify these properties.

Hosts may also elide a traversal whose body adds no matter beyond its
parent. Elision is likewise host policy; the runtime provides the
program-identity hashes it needs.

### 5.3 Cursor narrowing

At activation, for each cursor in the body with an `over` clause:

1. Pull the compiled `over` expression on the activation kernel. Accept a
   string spec, a `Partition`, a `PartitionSpec`, or a `PartitionList`
   with exactly one element. A multi-element list is an activation error.
2. Resolve against the cursor's extent using the existing `resolve`
   routine, honoring open-extent cursors as the cursor partition
   specification defines.
3. Write `<cursor>__cursor` and its six scalar projections.
4. Record the narrowed interval on the `Activation`.

The resolution and slot-writing steps already live in the crate as
`resolve_over`, `cursor_over_partitions`, and `narrow_cursor` in the
cursor partition module, ported from the nmbrs executor unchanged in
behavior. The `polydat` binary calls them at fiber setup today; activation
calls them once the construct lands.

### 5.4 Fibers

Activations are independent. A host may hand consecutive activations to
different fibers, or partition the traversal's index space and give each
fiber a range of tuple indices. Because every strategy is a decidable
permutation, a fiber can seek to its tuple index without dispensing the
tuples before it. Polydat exposes `TraversalStream::seek(index)` for this.

## 6. Axioms

- **T1, Traversal determinism.** For a fixed program and parent state, the
  sequence of tuples and the state of every activation at every cycle is
  the same on every run, on every host. Follows from D1 and D4 in the
  runtime model and from the strategy determinism in the algebra.
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

Not supported in this revision, and reported as compile errors rather than
silently accepted:

- a producer whose source depends on a per-cycle wire;
- a body that declares an `input` other than `cycle`;
- `over` naming a wire that is not `Partition`-typed at compile time.

## 8. Relationship to nmbrs

What moves into Polydat: the per-tuple activation loop, `over`
resolution and cursor-slot writes, the activation cache, and the seekable
traversal surface. nmbrs keeps its scope tree, elision policy, phase
scheduling, and YAML surface, and calls `TraversalStream` where it
currently calls `evaluate_for_iteration` plus `for_iteration` plus its own
slot writes.

The migration is additive. Existing nmbrs paths keep working while it
adopts the new surface one call site at a time.

## 9. Worked example

The toy test definition in its target form:

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

Sixteen activations, one per tuple. Each iterates its 250000-row slice.
`phase`, `interval_ms`, and `p` are wires in the body; `base_epoch_ms`
reaches it from the parent. The host's entire job is:

```rust
let mut traversal = kernel.traverse(0)?;
while let Some(mut activation) = traversal.advance()? {
    activation.for_each_cycle(|_, kernel| execute(kernel.pull("stmt").as_str()));
}
```

The shipped form is `examples/toy_test_definition.polydat`, and the
`polydat` binary is that host:

```text
polydat run toy_test_definition.polydat --fibers 4 --cycles 1000 --emit csv
```

## 10. The `polydat` binary

The crate's binary is the reference host for this construct. Its optional
behaviors are graph transforms, not runtime decorators: a feature that
needs to observe a scope is expressed by inserting a node into that scope,
where it has ordinary wired access to everything the scope can see.

- **Emission.** `--emit` appends `__emit := emit_row(format, names, ...)`
  to the program; under traversal the same binding is inserted inside
  the affected block, so every activation emits its own rows with the
  block's element names in scope. `--emit tile:<name>` selects a tile as
  the row ([Polytile](polytile.md) §10). The node buffers per thread;
  the harness drains buffers at chunk boundaries.
- **Assignment.** A bare `name=value` argument (or `--set name=value`)
  rewrites the named extern's default, or turns the named input into an
  extern with that default, before compilation. The text fuses to the
  declared type through the program's own string coercions, so a bad
  value is a compile error with the program's diagnostic. Nothing is
  written to a state at runtime, and the run loop advances only the
  program's coordinate inputs each cycle: an assigned input is an extern
  now and keeps its value.
- **Ordering.** Fibers claim work in chunks with sequence numbers. The
  writer restores cycle order by default and emits in completion order
  with `--unordered`. Under traversal, the sequence number is the tuple
  index, which every strategy makes decidable.
- **Diagnostics.** `check` reports statistics and the manifest. `explain`
  narrates each compilation phase from the compiler's event log and the
  compiled program: tokens, statements, inputs, modules, wires, types,
  lifecycle, constants, fusion, engine selection, provenance, and outputs.
  Traversal adds two phases: the comprehension plan and the per-body
  program table showing one compiled program per lexical position.
- **Timing.** `--timing` reports compile time, wall time, throughput, and
  per-fiber busy time. It wraps the run from outside because it measures
  the kernel rather than participating in it.

## 11. Implementation plan

1. **Parser and AST.** Done. `TokenKind::For(text)`, `Expr::For`,
   `Statement::For` with `ForSource` and `ForStmt`, block bodies, and
   pretty-printer round trip. Tests in `tests/for_syntax.rs` parse and
   print every form in §2, including nesting and `where` placeholders, and
   check the compiler's interim rejection. `tests/fuzz_for_syntax.rs`
   generates random programs over the whole `for` surface and random
   byte-level mutants of them, and checks that the front end never
   panics, element names match, the printer reaches a fixed point, and
   the compiler declines by name; an ignored superfuzz sweeps many seeds.
2. **Element typing and body compilation.** Done. `dsl::traversal`
   strips `for` forms from the parent, types each element from its source
   per §3.3 (generator calls are typed by compiling a one-line probe
   program), builds the body as a child file with an implicit `cycle`
   input, one `IterationExtern` per element, and one cascade extern per
   outer wire the parent exposes, and compiles it once with the parent's
   library paths and pragmas. `PolydatProgram::traversals()` and
   `producers()` expose the result; the binary's `explain traversals`
   phase prints the per-position program table. Tests in
   `tests/for_compile.rs` cover each row of the §3.3 table, widening and
   mixed-literal errors, outer-wire cascade with types, producer
   resolution, three-level nesting with one program per body, body type
   errors, unknown outer names, the extra-coordinate rule, and cursors
   over elements.
3. **Producer wires.** Done. `StreamerValue` in the comprehension
   module is the reflected value; the `streamer` node in the library
   materializes it as a `const` wire; `ForSourceKind::Derived` parses
   `for base where ... order ...` and `resolve_source` applies the filter
   and order to the base producer's AST, so derivations chain and
   traversals over them resolve their elements. Interpolating a producer
   renders its `for` text. Tests in `tests/for_producers.rs` cover the
   const wire and its value, stream independence, `where` and `order`
   derivations, chaining, traversal over a derivation, error cases, and
   cardinality metadata.
4. **Activation runtime.** Done. `kernel::activation` provides
   `PolydatKernel::traverse(i)` and `traverse_all()`, which snapshot the
   cascade, evaluate the comprehension against the kernel's current
   values through the kernel-aware runtime evaluator, and return a
   `TraversalStream` with `len`, `seek`, `advance`, and random-access
   `activation(i)`. An `Activation` is a fresh state over the body's
   shared program with elements and cascade bound and every cursor
   narrowed, exposing `cycle_count`, `cycle(i)`, and `for_each_cycle`
   under the §3.4 rule; nested traversals open from the activation's
   kernel. The binary runs programs with top-level traversals in
   traversal mode, inserting the emit binding into each body and striding
   activations across fibers. Tests in `tests/for_runtime.rs` cover one
   program per position (T2), T1 across two hosts, element and cascade
   binding, cursor narrowing and ordinal projection, full-extent cursors,
   nesting, fiber partitioning by index against a sequential trace,
   multi-partition `over` errors, and outer references in sources. The
   fuzzer activates every traversal of each generated program with
   bounded activations and cycles and checks T1 between two hosts.
5. **Program invariance.** Done. `kernel::programs_built()` counts every
   program constructed in the process and `program_count(program)` counts
   the root plus one program per body at every depth; `check --stats`
   prints the latter. Tests in `tests/for_invariance.rs` compile a
   three-level traversal, assert exactly four programs, activate all
   4000 innermost tuples while the build counter stays flat and every
   innermost activation shares one program by pointer, show a second host
   over the same program building nothing, show a generator-sourced
   traversal compiling its source once and re-opening compile-free, and
   show producers and derivations adding no programs per tuple. Making
   those tests pass required two runtime changes, both recorded in §5.2:
   a text-keyed cache for constant-expression evaluation, and a
   compile-free evaluator for filter predicates in the comprehension
   grammar. An ignored measurement test reports per-activation cost
   against bare state allocation and asserts it does not grow with the
   tuple index.
6. **Examples and docs.** Done. `examples/toy_test_definition.polydat`
   binds its flow as a producer and traverses it, and the `polydat`
   binary runs it in traversal mode; `docs/tutorials/toy_test_definition.md` shows
   the grammar and real output. `examples/for_producer.rs` and
   `examples/for_traversal.rs` back two new sections of
   `docs/tutorials/illustrations.md`. The README describes the construct in its
   iteration section and links here.

Each step lands with its tests and leaves the previous surfaces working.
