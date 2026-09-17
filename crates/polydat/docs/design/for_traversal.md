# The `for` Construct — Comprehension Producers and Traversal Scopes

**Ownership:** Polydat owns the grammar, the compiled form, the activation
runtime, and the consumption surfaces defined here. Hosts own scheduling
policy, side effects, and any surface sugar over the canonical text form.

**Companion documents:**
[Comprehension Forms](comprehension_forms.md) (the algebra),
[Cursor Partitions](cursor_partitions.md) (partition values and `over`),
[Scope Model](scope_model.md) (activation, materialization, coordinates),
[Runtime Model](runtime_model.md) (the D-axioms),
[Grammar](grammar.md) (productions this document extends),
[Engines](engines.md) (the engines an activation runs on).

## 1. The claim

A Polydat kernel is a pure function of its coordinates. Without a
traversal construct, every loop that drives it lives in the host: the
comprehension algebra, cursor partitions, and per-iteration scope
materialization exist in the crate, but nothing in the grammar can name
a traversal and nothing in the runtime can run one. The host has to
parse the comprehension, evaluate it, bind each tuple, resolve every
`over` clause, write cursor slots, and advance cycles.

The construct is one keyword, `for`, with two readings that share one
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

`comprehension_text` is the canonical text form accepted by the
comprehension parser. `source` is the comprehension's source surface:
literal lists, `lo..hi` ranges, generator calls such as
`partitions("*/4", n)` and `subdivide(p, n)`, and string comprehensions.
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

`for` is a hard keyword. Programs do not use it as an identifier because
the comprehension parser already reserved it.

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
`const`: the value is computed once at scope init and never re-evaluated
within the scope. The wire type is `Streamer`. A `Streamer` carries the
comprehension AST after validation and optimization, plus the metadata
the algebra computes: tuple shape, cardinality class, and index
addressability.

Derived forms apply the algebra's modifiers to a bound producer. They are
introduced by `for` like every other comprehension value, so one keyword
opens every comprehension and the lexer owns the text capture:

```text
base     := for k in 1..100, limit in 1..100
boundary := for base where {k} == 1 || {k} == 100
sampled  := for base order halton/50
```

A derivation is resolved at compile time against the producer bound
earlier in the same scope: the filter and the order are applied to the
base producer's comprehension AST (`resolve_source` in
`polydat-core/src/dsl/traversal.rs`), so derivations chain, a derivation of a
derivation is a comprehension like any other, and a traversal over a
derivation resolves its element names from the base. A producer
expression is comprehension text or a derivation; a bare producer name
alone (`x := for sweep`) is an error, since only a traversal may name a
producer without modifying it.

Each derived wire is a distinct comprehension. Streams obtained from them
never share dispense state. This is the independence contract in
[Comprehension Forms](comprehension_forms.md) §9.5.2, applied to wires.

**Realization.** `Streamer` rides the type plane's extension point: the
wire's port type is `Ext` and its value is a reflected `StreamerValue`
whose type name is `Streamer`, carrying the text and the resolved
algebra AST and exposing `compiled`, `coordinate_stream`, `metadata`,
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
the runtime activates one child scope: a fresh kernel over the child
program with the tuple's values bound.

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
  becomes a cascade extern of the child program, typed from the parent.
- **Modules are visible.** The body sees every module the parent had
  resolved when the body was lowered, the parent's own definitions
  included, wherever and on whatever engine the body compiles.
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
value to a `Partition`, narrows the cursor to that interval, and records
the slice on the activation.

The cycle rule for an activation:

- **A body with one or more cursors** iterates its narrowest cursor
  extent. Each ordinal in the slice is one cycle. The cursor's ordinal is
  written before each pull.
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
is the nested-partition form the cursor partition specification names.
Coordinate paths compose root-first, matching
[Scope Model](scope_model.md) §7.

### 3.6 Consumption surfaces

Polydat exposes three surfaces, extending
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

`TraversalStream::advance` yields the next activation. `Activation::cycle(i)`
positions the kernel at cycle `i` under the rule in §3.4, writing the
coordinate and the cursor ordinal, and returns it ready to pull;
`for_each_cycle` runs every cycle in order. A host that wants only tuples
uses the first surface; one that wants to schedule work uses the third.

A `Streamer` wire pulled from a kernel exposes `compiled` and
`coordinate_stream`, so a host holding a compiled program can enumerate
any producer it names; every call is a fresh stream with its own
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
   from this record, with the same settings, on the first request for
   that engine (§5.2).

The body's cursors compile as cursors do anywhere, allocating the `over`
slots. No node type exists for cursor narrowing; the runtime writes the
existing slots.

A body's compile error carries the `for` source text and the statement's
line and column, then the child compiler's diagnostic. Tile events raised
inside a body reach the parent's compile log.

## 5. Runtime

### 5.1 One program per lexical position

A compiled program is a property of where a `for` body appears in the
program text, not of which coordinates reach it. The body's wiring, types,
lifecycle classification, fusion, and native code are the same for every
tuple, because the tuple only supplies values for slots the program already
declares. Coordinates vary; the program does not.

Polydat therefore compiles each `for` body exactly once per engine, and
caches it keyed on the body's lexical position. The cache is not an
optimization over tuple identity. It is the statement that a kernel
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
activations share program C. Each activation is a fresh kernel over that
program with its own tuple bound. Every program built for a tree is
recorded in the tree's `CompileLedger`, reached through `ledger()` on
the kernel or the program, and `program_count(program)` counts the root
plus one program per body at every depth, so a host can verify the
property. Two trees never share a ledger, whatever thread or process
runs them; a host that wants several trees on one ledger passes it in
`CompileOptions::ledger`.

### 5.2 Affine activation

Activation never compiles. It clones the program `Arc`, creates a nested
kernel over it, binds the tuple and the cascade, and narrows the cursors.
The first activation of a body on an engine pays that engine's
compilation once; every activation after that costs the same constant,
which depends only on the size of the state. This is the affine property:
compile once, then a per-activation cost that is independent of how many
coordinates precede it.

A host may additionally reuse a live activation when it revisits the same
tuple under the same parent, for replay or for a second phase over the
same slice. That is a state-reuse policy, not a compilation concern, and
Polydat leaves it to the host. Correctness never depends on it; the
program cache in §5.1 is what Polydat guarantees.

**Engines.** Opening and activation are the same on every engine.

- `traverse` is a method of the `Kernel` trait: a kernel on any engine
  opens the traversals its program declares against the values it holds.
  Opening snapshots the cascade through the trait (`pull` for a wire the
  parent computes, `input_value` for an input or extern), builds the
  body's scope, the body's interpreter program with the cascaded wires
  bound, and evaluates the comprehension there through the evaluator's
  `Lookup` view, with a tuple's own elements layered in front of the
  scope as the tuple is built (`Layered` in `polydat-core/src/kernel/interp.rs`). A
  source or predicate resolves every name it can reference in that
  scope, so the opening kernel contributes only the snapshot and opening
  needs no kernel of the engine that opened it.
- `TraversalStream::activation_on(index, engine)` compiles the body for
  that engine once, on the first request, from the `BodySource` through
  the same assembler every engine uses (`Traversal::program_on`), and
  every later activation on that engine shares the program, as
  interpreter activations share theirs: one program per engine per
  position. The activation is a kernel of its own created from that program
  and driven through the `Kernel` trait: elements and cascade bind by
  name through `set_input`, cursors narrow through one routine, and it
  computes what the interpreter's activation computes.
  `activate(index)` is `activation_on` on `Engine::default()`;
  `activation(index)` is the interpreter's.
- A body's own `for` statements are compiled with the body on every
  engine and open from its activation, so every level of a nest, root,
  outer bodies, and the innermost body where the cycles run, is on the
  engine the host chose.

**The cascade is a snapshot.** A cascaded wire reaches the body as the
value it had when the traversal was opened, bound into the activation's
extern slot; a wire that changes on the parent after that does not change
inside the activations already opened, and re-opening the traversal sees
the new value. This holds for a `shared` wire as for any other: the body
receives the cell's value at open, not the cell. Cells the body's own
`shared` bindings create follow [Scope Model](scope_model.md) §8 within
the activation.

**A state of its own.** An activation is a kernel state like any other,
created with `create_kernel`: it owns its inputs, its outputs, and the
storage behind them, and observes the provenance rules as if it were
the only state ([Runtime Model](runtime_model.md) R4). Nothing it does
reaches into the kernel that opened it beyond the cells it attaches.

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
interpolated text. A source or predicate that does compile is charged
to the ledger of the tree that opened it, so a host can verify these
properties on the kernel it holds.

Hosts may also elide a traversal whose body adds no matter beyond its
parent. Elision is likewise host policy; the runtime provides the
program-identity hashes it needs.

### 5.3 Cursor narrowing

At activation, for each cursor in the body, through the `Kernel` trait
and therefore the same on every engine:

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
live in `iteration::cursor_partition`; the `polydat` binary resolves its
cursors with `cursor_over_partitions_on` and narrows them with `set_cursor`
at fiber setup.

### 5.4 Fibers

Activations are independent. A host may hand consecutive activations to
different fibers, or partition the traversal's index space and give each
fiber a range of tuple indices. Because every strategy is a decidable
permutation, a fiber can seek to its tuple index without dispensing the
tuples before it. Polydat exposes `TraversalStream::seek(index)` and
random-access `activation(index)` for this.

## 6. Axioms

- **T1, Traversal determinism.** For a fixed program and parent state, the
  sequence of tuples and the state of every activation at every cycle is
  the same on every run, on every host, on every engine. Follows from D1
  and D4 in the runtime model and from the strategy determinism in the
  algebra.
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

- a producer whose source depends on a per-cycle wire;
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

Sixteen activations, one per tuple. Each iterates its 250000-row slice.
`phase`, `interval_ms`, and `p` are wires in the body; `base_epoch_ms`
reaches it from the parent; `reading_model` is visible in the body because
the parent defined it. The host's entire job is:

```rust
let mut traversal = kernel.traverse(0)?;
while let Some(mut activation) = traversal.advance()? {
    activation.for_each_cycle(|_, kernel| execute(kernel.pull("stmt").as_str()));
}
```

The `polydat` binary is that host. Its optional behaviors are graph
transforms, not runtime decorators: a feature that needs to observe a
scope is expressed by inserting a node into that scope, where it has
ordinary wired access to everything the scope can see; under traversal
the inserted binding goes inside the affected block, so every activation
carries it with the block's element names in scope.
