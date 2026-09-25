---
type: tutorial
title: Illustrations
timestamp: 2026-09-25
description: Runnable examples of the DSL, the assembler, function graphs, libraries, type safety, parameter spaces, partitions, traversal, and tiles.
tags: [language, host]
---

# Illustrations

This page shows what polydat does through small runnable examples,
rather than explaining how it works.

Polydat turns named `u64` coordinate tuples into typed output values
(variates) by evaluating a compiled directed acyclic graph (DAG) of
function nodes. The same coordinate always produces the same outputs,
so results are reproducible and can be computed in parallel with no
shared mutable state.

The two entry points below are the two ways a caller builds a program:
the Polydat DSL, a small expression language you write as a string, and
the Assembler API, which builds the graph node by node in Rust. Both
produce the same kind of compiled kernel, a running instance of the
program that takes coordinates and returns outputs; pick whichever
suits the calling code.

For all of these features working together in one grammar file, see
[A toy test definition](toy_test_definition.md).

Every code block in this file is the body of a runnable example under
[`polydat/examples/`](../../examples/). They all compile under
`cargo build --examples` and produce the printed outputs you see
quoted below. If a snippet ever drifts from what the example actually
does, the example is the source of truth.

## From the Polydat DSL

```rust
use polydat::dsl::compile_polydat_kernel;

let mut kernel = compile_polydat_kernel(r#"
    input cycle: u64
    hashed := hash(cycle)
    user_id := mod(hashed, 1000000)
"#).unwrap();

kernel.set_inputs(&[42]);
let user_id = kernel.pull("user_id").as_u64();
assert!(user_id < 1_000_000);
```

See [`examples/basic.rs`](../../examples/basic.rs) for the full runnable form.

## From the Assembler API

```rust
use polydat::compile::assembly::{PolydatAssembler, WireRef};
use polydat::library::hash::Hash;
use polydat::library::arithmetic::Mod;

let mut asm = PolydatAssembler::new(vec!["cycle".into()]);
asm.add_node("hashed", Box::new(Hash::new()), vec![WireRef::input("cycle")]);
asm.add_node("user_id", Box::new(Mod::new(1_000_000)), vec![WireRef::node("hashed")]);
asm.add_output("user_id", WireRef::node("user_id"));

let mut kernel = asm.compile_kernel().unwrap();
kernel.set_inputs(&[42]);
assert!(kernel.pull("user_id").as_u64() < 1_000_000);
```

See [`examples/assembler.rs`](../../examples/assembler.rs).

---

The README's capabilities table says what polydat does. Each section
below shows one capability in a small, runnable form, so each claim can
be checked by running it.

## A function graph grammar

Polydat source is a literal grammar for declaring function graphs. Each
`name := expr` line is a graph node; the expressions reference other
nodes by name, forming directed edges. The compiler topologically
sorts the result and refuses cycles.

```rust
let mut kernel = polydat::dsl::compile_polydat_kernel(r#"
    input cycle: u64

    // Decompose one coordinate into two dimensions (device, reading).
    (device, reading) := mixed_radix(cycle, 100, 0)

    // Derive an identity hash and split it into independent streams.
    device_h := hash(device)
    h_temp := hash(device_h)
    h_humid := hash(h_temp)

    // Convert each independent stream into a unit-interval quantile.
    q_temp := unit_interval(h_temp)
    q_humid := unit_interval(h_humid)
"#).expect("compile failed");

kernel.set_inputs(&[12_345]);
// device=45, reading=123, q_temp=0.019101, q_humid=0.155169
```

The grammar fits on one page (see [`polydat-grammar/src/`](../../../polydat-grammar/src/) or [the grammar](../design/polydat_grammar.md)) but the
graphs you can build are arbitrarily wide and deep. The compiler
tracks each wire's port type (u64, f64, str, bool, bytes, json,
vectors…) and rejects mismatches at compile time, before the kernel
ever runs.

Note: Polydat numeric literals don't accept Rust-style underscores —
`1000000` is fine, `1_000_000` is a parse error.

See [`examples/function_graph_grammar.rs`](../../examples/function_graph_grammar.rs).

## A procedural generation kernel

A compiled kernel is a pure function of its coordinate inputs: the same
coordinate always produces the same outputs, with no shared state.
Because of this, a multi-thread benchmark can generate billions of
distinct, reproducible variates in parallel. Each thread gets its own
kernel, and the program is shared through an `Arc`.

```rust
let kernel = polydat::dsl::compile_polydat_kernel(r#"
    input cycle: u64
    user_id := mod(hash(cycle), 1000000)
"#).unwrap();

let program = kernel.into_program();

// Two threads, no locks, identical results for the same coord.
let results: Vec<u64> = std::thread::scope(|s| {
    let handles: Vec<_> = (0..2).map(|_| {
        let program = program.clone();
        s.spawn(move || {
            let mut kernel = program.create_kernel();
            kernel.set_inputs(&[42]);
            kernel.pull("user_id").as_u64()
        })
    }).collect();
    handles.into_iter().map(|h| h.join().unwrap()).collect()
});
// All threads produce the same value (275413 for coord=42).
```

See [`examples/generation_kernel.rs`](../../examples/generation_kernel.rs)
and [`examples/multi_thread.rs`](../../examples/multi_thread.rs) (the
larger benchmark form).

## A function library and loader

The 230 built-in nodes (see [nodes.md](../reference/nodes.md)) are one library.
Functions that workload authors write in `.polydat` files are another:
the library paths in `CompileOptions` load them from disk, and the DSL
calls them by name as if they were built in.

```rust
let stdlib = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
    .join("..").join("polydat-core").join("stdlib");

let mut kernel = polydat::dsl::compile_polydat_kernel_with_options(
    r#"
        input cycle: u64
        // `hashed_id` comes from the embedded standard library.
        uid := hashed_id(cycle, 1000000)
    "#,
    &CompileOptions {
        lib_paths: vec![stdlib],   // a directory of modules, searched first
        context: "library example".into(),
        ..CompileOptions::default() // no source directory, every output, not strict
    },
    None,
).expect("compile failed");
```

`hashed_id` comes from the embedded standard library and needs no path;
a `lib_paths` directory adds the host's own modules ahead of it. The
shipped [`stdlib/`](../../../polydat-core/stdlib/) directory has more examples:
`identity.polydat`, `distributions.polydat`, `hashing.polydat`, `modeling.polydat`, etc.
Each is loadable the same way.

See [`examples/library_and_loader.rs`](../../examples/library_and_loader.rs).

## A runtime type-safety reifier

Polydat's `Value` enum has one variant per port type, so the types the
compiler checked are still visible when the host reads a value. The
compiler checks port types when it wires nodes together (for example,
`mod` takes two u64 inputs and returns a u64); at run time each value
keeps its type until the host reads it through a typed accessor or a
`match`.

```rust
use polydat::ast::Value;

let mut kernel = polydat::dsl::compile_polydat_kernel(r#"
    input cycle: u64
    n := mod(hash(cycle), 1000)             // u64
    p := unit_interval(hash(cycle))         // f64 in [0.0, 1.0)
    label := number_to_words(n)             // Str
    is_big := u64_gt(n, 500)                // u64 (0 or 1)
"#).unwrap();

kernel.set_inputs(&[42]);
match kernel.pull("n")      { Value::U64(_) => {}, _ => unreachable!() }
match kernel.pull("p")      { Value::F64(_) => {}, _ => unreachable!() }
match kernel.pull("label")  { Value::Str(_) => {}, _ => unreachable!() }
match kernel.pull("is_big") { Value::U64(_) => {}, _ => unreachable!() }
```

Connecting incompatible port types — e.g., feeding the u64 result of
`mod(...)` directly into `sin(...)` which expects f64 — is a compile
error, not a silent conversion. Type checking also holds in Phase 3
native code generation: each Cranelift function signature follows the
port types, so a type mistake cannot reach the generated native code.

Note: polydat's comparison family (`u64_gt`, `u64_lt`, `f64_eq`, …)
returns `u64` (0 or 1) rather than `Bool`. The `Bool` variant of
`Value` exists and is used by `regex_match`, `random_bool`, `const_bool`,
and a few others, but most comparison-driven workloads stay in u64 for
fast-path math.

See [`examples/type_safety.rs`](../../examples/type_safety.rs).

## A parameter space projection system

A single integer coordinate (`cycle`) is a one-dimensional parameter
space. `mixed_radix` maps it onto a space of several dimensions, which
is useful when you want "100 devices × N readings" but have only one
cycle counter to drive everything.

```rust
let mut kernel = polydat::dsl::compile_polydat_kernel(r#"
    input cycle: u64
    (device, reading) := mixed_radix(cycle, 100, 0)
"#).unwrap();

for cycle in [0u64, 1, 99, 100, 49_999, 50_000] {
    kernel.set_inputs(&[cycle]);
    let d = kernel.pull("device").as_u64();
    let r = kernel.pull("reading").as_u64();
    println!("cycle={cycle} → device={d} reading={r}");
}
```

Actual output (from the runnable example):

```text
cycle  device  reading
-----  ------  -------
    0       0        0
    1       1        0
   99      99        0
  100       0        1
49999      99      499
50000       0      500
```

Devices fill before readings advance: the first dimension wraps at its
cardinality of 100 and the second increments once per wrap.

The trailing `0` in `mixed_radix(cycle, 100, 0)` declares the second
dimension as unbounded — it grows indefinitely rather than wrapping.
Replace it with a finite cardinality to cap the total space.

`partition` nodes (see [nodes.md §Determinism](../reference/nodes.md#determinism-and-sampling))
give type-checked access to the resulting cells.

See [`examples/parameter_space.rs`](../../examples/parameter_space.rs).

## Partitioning a domain across fibers

A cursor names an ordinal domain, a range of row numbers. A partition
spec splits that domain into fixed intervals with absolute bounds, so
each fiber (one of the host's worker loops) can take one interval
without coordinating with the others. The `partitions` node takes a
spec and the size of the domain, resolves the spec, and returns the
list of intervals as a value:

```rust
let mut k = compile_polydat_kernel(r#"
    input cycle: u64
    parts := partitions("20%,30%,*", 1000000)
"#).expect("compile");
k.set_inputs(&[0]);
let list = k.pull("parts").as_partition_list().expect("list").clone();
for p in list.0.iter() {
    println!("p{}  [{:>7}, {:>7})  {:>6} ordinals", p.idx, p.start_ord, p.end_ord, p.end_ord - p.start_ord);
}
```

```text
p0  [      0,  200000)  200000 ordinals
p1  [ 200000,  500000)  300000 ordinals
p2  [ 500000, 1000000)  500000 ordinals
```

A fiber claims its slice with `over`. The kernel exposes the resolved
partition as `q.cursor` and its bounds as metadata wires. Resolving the
`over` spec and writing those values is the host's job when it sets up
the scope. `cursor_over_partitions_on` takes the kernel and a cursor's
schema and returns the cursor's resolved partitions, and `set_cursor`
takes the cursor's name and one partition and narrows the cursor to it.
Both work the same way on all four engines. This example performs the
step for fiber 1:

```rust
let mut k = compile_polydat_kernel(r#"
    input cycle: u64
    cursor q = range(0, 1000000) over "20%,30%,*"
    start := q.cursor.start_ordinal
    end   := q.cursor.end_ordinal
    size  := cardinality(q.cursor)
    slot  := mod_in(cycle, q.cursor)
    row   := mod(hash(slot), 1000000)
    sub   := subdivide(q.cursor, 4)
"#).expect("compile2");

// The host resolves the `over` spec and hands fiber 1 its partition.
let schema = k.cursor_schemas()[0].clone();
let parts = cursor_over_partitions_on(k.as_mut(), &schema).expect("resolve");
k.set_cursor("q", &parts[1]).expect("narrow");

for cycle in [0u64, 1, 299_999, 300_000] {
    k.set_inputs(&[cycle]);
    // ... pull start, end, size, slot, row
}
```

```text
cycle=0      window=[200000, 500000) size=300000 slot=200000 row=70708
cycle=1      window=[200000, 500000) size=300000 slot=200001 row=145486
cycle=299999 window=[200000, 500000) size=300000 slot=499999 row=795364
cycle=300000 window=[200000, 500000) size=300000 slot=200000 row=70708

sub0  [200000, 275000)
sub1  [275000, 350000)
sub2  [350000, 425000)
sub3  [425000, 500000)
```

`mod_in` maps the fiber's local cycle onto an ordinal inside its
absolute slice, wrapping at the slice's end, so the fiber never touches
another fiber's ordinals and every row it produces
can be regenerated from its ordinal alone. `subdivide` splits the slice
again with the same boundary math, which is how a fiber hands work to
worker threads. The full spec language, including recipes, windows,
gaps, and ordering, is in
[Cursor Partitions](../design/cursor_partitions.md).

See [`examples/cursor_partitions.rs`](../../examples/cursor_partitions.rs).

## Traversing a parameter space

A comprehension names a parameter space and how to walk it. The
algebra has four constructors, `cartesian`, `zip`, `union`, and
`filter`, plus `order`, which applies a named traversal strategy.
In the text form that hosts embed, a space and two variations of it are
written like this:

```polydat
for k in 1..4, limit in 10..40 step 10
for k in 1..4, limit in 10..40 step 10 order halton/5
for k in 1..4, limit in 10..40 step 10 where {k} >= 2 && {limit} != 20
```

The same space can be built directly in Rust and drained as coordinate
tuples:

```rust
let base = Comprehension::cartesian(vec![
    Comprehension::clause("k", Source::IntRange { lo: 1, hi: 4, step: 1 }),
    Comprehension::clause("limit", Source::IntRange { lo: 10, hi: 40, step: 10 }),
]);
show("lex", &base);
show("reverse", &Comprehension::order(base.clone(), StrategyName::ReverseLex, None));
show("diagonal", &Comprehension::order(base.clone(), StrategyName::Diagonal, None));
show("shells", &Comprehension::order(base.clone(), StrategyName::Shells, None));
show("extrema/1", &Comprehension::order(base.clone(), StrategyName::Extrema, Some(1)));
show("halton/5", &Comprehension::order(base.clone(), StrategyName::Halton, Some(5)));
show("where", &Comprehension::filter(base.clone(), "{k} >= 2 && {limit} != 20"));
```

where `show` compiles the comprehension, drains its coordinate stream,
and prints each tuple as `(k,limit)`:

```text
lex             9  (1,10) (1,20) (1,30) (2,10) (2,20) (2,30) (3,10) (3,20) (3,30)
reverse         9  (3,30) (3,20) (3,10) (2,30) (2,20) (2,10) (1,30) (1,20) (1,10)
diagonal        9  (1,10) (1,20) (2,10) (1,30) (2,20) (3,10) (2,30) (3,20) (3,30)
shells          9  (1,10) (1,20) (1,30) (2,10) (2,30) (3,10) (3,20) (3,30) (2,20)
extrema/1       4  (1,10) (1,30) (3,10) (3,30)
halton/5        5  (2,20) (1,30) (3,10) (1,20) (2,30)
where           4  (2,10) (2,30) (3,10) (3,30)
```

Every strategy is a decidable permutation of the same nine points, so
the optimizer can reason about it and a run can be replayed from its
position. `extrema` visits the corners first, `shells` works inward
from the boundary, and `halton` is a low-discrepancy sample that
covers the space evenly however early it is cut off. A
`CoordinateStream` yields the tuples; a `ScopedKernelStream` yields a
kernel instance per tuple with the coordinates already bound. Both are specified in
[Comprehension Forms](../design/comprehension_forms.md).

See [`examples/parameter_space_traversal.rs`](../../examples/parameter_space_traversal.rs).

## A comprehension producer

`name := for ...` binds a comprehension as a value. The wire's value is
a Streamer, which holds the comprehension's text and its validated
algebra, can create any number of independent streams over the tuples,
and reports how many tuples there are. A derived producer is written
with the same `for` keyword applied to a bound producer, as `corners`
and `sampled` are below.

```rust
let mut kernel = polydat::dsl::compile_polydat_kernel(r#"
    input cycle: u64

    base    := for k in 1..4, limit in 10,20,30
    corners := for base where {k} == 1 || {k} == 3
    sampled := for base order halton/4
    label   := "plan: {base}"
"#).expect("compile failed");

kernel.set_inputs(&[0]);
println!("{}", kernel.pull("label").as_str());
for name in ["base", "corners", "sampled"] {
    let value = kernel.pull(name);
    show(name, value.as_streamer().expect("streamer"));
}

// Two streams from one wire never share a cursor.
let value = kernel.pull("base");
let base = value.as_streamer().unwrap();
let mut a = base.coordinate_stream();
let b = base.coordinate_stream();
a.next();
a.next();
println!("after two pulls on a: a has {} left, b has {}", a.count(), b.count());
```

where `show` prints the cardinality class and dispenses the stream:

```text
plan: for k in 1..4, limit in 10, 20, 30
base     Bounded(9)   9 tuples  (1,10) (1,20) (1,30) (2,10) (2,20) (2,30) (3,10) (3,20) (3,30)
corners  BoundedAtMost(9)   6 tuples  (1,10) (1,20) (1,30) (3,10) (3,20) (3,30)
sampled  Bounded(4)   4 tuples  (2,20) (1,30) (3,10) (1,20)
after two pulls on a: a has 7 left, b has 9
```

The producer is computed once, when the kernel is initialized, so it
costs nothing per cycle, and interpolating it into a string yields its
`for` text. See
[`examples/for_producer.rs`](../../examples/for_producer.rs).

## A traversal

`for <comprehension> { body }` activates one child scope per tuple. The
body compiles once, at parent compile time, into its own program. Each
activation is a new kernel over that program, with the tuple's elements
bound as typed wires and the parent's wires that the body references
passed in. A cursor declared `over` an element is narrowed for each
activation, and its slice sets how many cycles the activation runs.

```rust
let mut kernel = polydat::dsl::compile_polydat_kernel(r#"
    input cycle: u64
    extern total: u64 = 1000
    base := hash(cycle)

    for p in partitions("*/4", {total}), scale in 1,100 {
        cursor rows = range(0, 1000) over p
        row  := mod_in(cycle, rows.cursor)
        v    := u64_add(u64_mul(row, scale), base)
    }
"#).expect("compile failed");
kernel.set_inputs(&[7]);

let stream = kernel.traverse(0).expect("open traversal");
// The body compiles for the engine on the first activation; every
// activation after it shares that program.
drop(stream.activation(0).expect("first activation"));
let ledger = kernel.ledger().clone();
let built_before = ledger.programs();
for index in 0..stream.len() {
    let mut act = stream.activation(index).expect("activation");
    let slice = act.cursor.clone().expect("cursor slice");
    let scale = act.coord("scale").unwrap().as_u64();
    let kernel = act.cycle(0);
    // ... print index, slice, scale, cycle count, row, v
}
println!("programs built after the first activation: {}", ledger.programs() - built_before);
```

```text
body program: 16 nodes, compiled once
8 activations from `p in partitions("*/4", {total}), scale in 1, 100`

act  p          scale  cycles  first row  first v
  0  [  0, 250)      1     250          0  7191089600892374487
  1  [  0, 250)    100     250          0  7191089600892374487
  2  [250, 500)      1     250        250  7191089600892374737
  3  [250, 500)    100     250        250  7191089600892399487
  4  [500, 750)      1     250        500  7191089600892374987
  5  [500, 750)    100     250        500  7191089600892424487
  6  [750,1000)      1     250        750  7191089600892375237
  7  [750,1000)    100     250        750  7191089600892449487

programs built after the first activation: 0
```

The comprehension's source reads the parent's `total` through `{total}`
when the traversal is opened, so a host can change the extern and open
it again without recompiling. Fibers divide a traversal among
themselves by taking activations by index. The full contract is [The `for`
Construct](../design/for_traversal.md). See
[`examples/for_traversal.rs`](../../examples/for_traversal.rs).

## Tiles: documents as wires

A `tile` is a template whose holes are expressions. It compiles into
the graph and renders per cycle as a string wire, with the encoding
deciding how each hole is written.

```rust
let mut kernel = compile_polydat_kernel(r#"
    input cycle: u64
    base := cycle * 100
    tile samples : json := {
        "base": ${base},
        "points": [ @for i in 0..3 { {"i": ${i}, "v": ${base + i}} } ]
    }
"#).unwrap();
kernel.set_inputs(&[2]);
println!("{}", kernel.pull("samples").as_str());
```

```json
{
    "base": 200,
    "points": [ {"i": 0, "v": 200},{"i": 1, "v": 201},{"i": 2, "v": 202} ]
}
```

Numbers render bare and strings render quoted because the encoder
reads the wire's type; `@for` repeats its body over a comprehension.
The walk-through from a one-line text tile to a statement carrying a
JSON document is [the Polytile tutorial](polytile_tutorial.md). See
[`examples/polytile_tutorial.rs`](../../examples/polytile_tutorial.rs).

## A context layering API

A library function can be composed into a parent kernel as a subgraph.
Each call site provides its own inputs and gets its own outputs: the
same compiled function, applied in a different context at each call.

```rust
let stdlib = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
    .join("..").join("polydat-core").join("stdlib");

let mut kernel = polydat::dsl::compile_polydat_kernel_with_options(
    r#"
        input cycle: u64
        (tenant, device) := mixed_radix(cycle, 100, 0)
        // Same compiled function, two layered call sites.
        tenant_id := hashed_id(tenant, 10000)
        device_id := hashed_id(device, 10000)
    "#,
    &CompileOptions { lib_paths: vec![stdlib], context: "context layering".into(), ..CompileOptions::default() },
    None,
).expect("compile failed");
```

Within one polydat program, composition happens in a single file, as
above. The larger `polydat::kernel::subcontext` API (`SubcontextBuilder`,
`ScopeKernel`, `PolydatMatter`) is what a host uses to layer entire
scope trees over parent kernels, with typed import and export contracts
at every boundary. The principle is the same; the multi-scope case
needs more bookkeeping.

See [`examples/context_layering.rs`](../../examples/context_layering.rs).

## An expression language

The `:=` lines in Polydat are full expressions, not just direct node calls.
You can nest, chain, and combine without naming every intermediate
wire — useful for one-line derivations where naming the intermediate
would just be noise.

```rust
let mut kernel = polydat::dsl::compile_polydat_kernel(r#"
    input cycle: u64

    // Nested expressions — no named intermediates needed.
    user_id := mod(hash(cycle), 1000000)

    // Multi-arg, mixed-type composition.
    word := str_upper(number_to_words(mod(cycle, 10)))

    // Inline conditional via select_u64(cond, then, else).
    bucket := select_u64(u64_lt(user_id, 500000), 0, 1)
"#).unwrap();
```

Sample output (from the runnable example):

```text
cycle  user_id  word     bucket
-----  -------  -------  ------
    0   607535  ZERO          1
    1   822465  ONE           1
    2   348110  TWO           0
    3   139053  THREE         0
    4   603978  FOUR          1
```

See [`examples/expression_language.rs`](../../examples/expression_language.rs).
