# Polydat Illustrations

It may actually be easier to show what polydat can do than to explain
what it is or how it actually works.

Polydat transforms named `u64` coordinate tuples into typed output
variates via a compiled DAG of composable function nodes. The same
coordinate always produces the same outputs — deterministic,
reproducible, and parallelizable with zero shared mutable state.

The two entry points below are the two surfaces polydat presents to a
caller: the Polydat DSL (a small expression grammar you write as a string)
and the Assembler API (programmatic node-by-node construction). They
produce the same kind of compiled kernel; pick the one that fits your
call-site shape.

For all of these features working together in one grammar file, see
[A toy test definition](toy_test_definition.md).

Every code block in this file is the body of a runnable example under
[`polydat/examples/`](../examples/). They all compile under
`cargo build --examples` and produce the printed outputs you see
quoted below. If a snippet ever drifts from what the example actually
does, the example is the source of truth.

## From the Polydat DSL

```rust
use polydat::dsl::compile_polydat;

let mut kernel = compile_polydat(r#"
    input cycle: u64
    hashed := hash(cycle)
    user_id := mod(hashed, 1000000)
"#).unwrap();

kernel.set_inputs(&[42]);
let user_id = kernel.pull("user_id").as_u64();
assert!(user_id < 1_000_000);
```

See [`examples/basic.rs`](../examples/basic.rs) for the full runnable form.

## From the Assembler API

```rust
use polydat::assembly::{GkAssembler, WireRef};
use polydat::nodes::hash::Hash64;
use polydat::nodes::arithmetic::ModU64;

let mut asm = GkAssembler::new(vec!["cycle".into()]);
asm.add_node("hashed", Box::new(Hash64::new()), vec![WireRef::coord("cycle")]);
asm.add_node("user_id", Box::new(ModU64::new(1_000_000)), vec![WireRef::node("hashed")]);
asm.add_output("user_id", WireRef::node("user_id"));

let mut kernel = asm.compile().unwrap();
kernel.set_inputs(&[42]);
assert!(kernel.pull("user_id").as_u64() < 1_000_000);
```

See [`examples/assembler.rs`](../examples/assembler.rs).

---

The README claims polydat is seven things at once. Each section below
shows one of those facets in a small, runnable form so the claim isn't
just an assertion.

## A function graph grammar

GK source is a literal grammar for declaring function graphs. Each
`name := expr` line is a graph node; the expressions reference other
nodes by name, forming directed edges. The compiler topologically
sorts the result and refuses cycles.

```rust
let mut kernel = polydat::dsl::compile_polydat(r#"
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
// device=45, reading=123, q_temp=0.071753, q_humid=0.452280
```

The grammar fits on one page (see [`src/dsl/`](../src/dsl/)) but the
graphs you can build are arbitrarily wide and deep. The compiler
tracks each wire's port type (u64, f64, str, bool, bytes, json,
vectors…) and rejects mismatches at compile time, before the kernel
ever runs.

Note: Polydat numeric literals don't accept Rust-style underscores —
`1000000` is fine, `1_000_000` is a parse error.

See [`examples/function_graph_grammar.rs`](../examples/function_graph_grammar.rs).

## A procedural generation kernel

A compiled kernel is a pure function of its coordinate inputs. Same
coordinate in, same outputs out, every time, no shared state. That
property is what lets a multi-thread benchmark generate billions of
distinct, reproducible variates in parallel — each thread gets its
own state, the program is shared via `Arc`.

```rust
let kernel = polydat::dsl::compile_polydat(r#"
    input cycle: u64
    user_id := mod(hash(cycle), 1000000)
"#).unwrap();

let program = kernel.into_program();

// Two threads, no locks, identical results for the same coord.
let results: Vec<u64> = std::thread::scope(|s| {
    let handles: Vec<_> = (0..2).map(|_| {
        let program = program.clone();
        s.spawn(move || {
            let mut state = program.create_state();
            state.set_inputs(&[42]);
            state.pull(&program, "user_id").as_u64()
        })
    }).collect();
    handles.into_iter().map(|h| h.join().unwrap()).collect()
});
// All threads produce the same value (915720 for coord=42).
```

See [`examples/generation_kernel.rs`](../examples/generation_kernel.rs)
and [`examples/multi_thread.rs`](../examples/multi_thread.rs) (the
larger benchmark form).

## A function library and loader

The 230 built-in nodes (see [nodes.md](nodes.md)) are one library.
Workload-author functions written in `.polydat` files are another —
`compile_polydat_with_libs` loads them from disk and they're callable from
your DSL by name as if they were built in.

```rust
let stdlib = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
    .join("stdlib").join("identity.polydat");

let mut kernel = polydat::dsl::compile_polydat_with_libs(
    r#"
        input cycle: u64
        // `hashed_id` is loaded from identity.polydat, not a built-in.
        uid := hashed_id(cycle, 1000000)
    "#,
    None,                 // source_dir — no implicit ./*.gk lookup
    vec![stdlib],         // explicit library paths
    &[],                  // required_outputs — compile everything
    false,                // strict
    "library example",
).expect("compile failed");
```

The shipped [`stdlib/`](../stdlib/) directory has more examples:
`identity.polydat`, `distributions.polydat`, `hashing.polydat`, `modeling.polydat`, etc.
Each is loadable the same way.

See [`examples/library_and_loader.rs`](../examples/library_and_loader.rs).

## A runtime type-safety reifier

Polydat's `Value` enum reifies the port-type lattice at the runtime
boundary. The compiler enforces port-type contracts when wiring nodes
(e.g., `mod` accepts two u64 inputs, returns u64); the runtime carries
the typed payload through to the consumer, who reads it through a
typed accessor.

```rust
use polydat::node::Value;

let mut kernel = polydat::dsl::compile_polydat(r#"
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
error, not a silent coercion. The type discipline survives through
Phase 3 JIT codegen: each Cranelift signature mirrors the port-type
shape, so a typed mistake can't survive into the generated native
code.

Note: polydat's comparison family (`u64_gt`, `u64_lt`, `f64_eq`, …)
returns `u64` (0 or 1) rather than `Bool`. The `Bool` variant of
`Value` exists and is used by `regex_match`, `control_bool`, and a
few others, but most comparison-driven workloads stay in u64 for
fast-path math.

See [`examples/type_safety.rs`](../examples/type_safety.rs).

## A parameter space projection system

A single integer coordinate (`cycle`) is a 1-D parameter space.
`mixed_radix` projects it onto an N-D space — useful when you want
"100 devices × N readings" but only have one cycle counter to
drive everything.

```rust
let mut kernel = polydat::dsl::compile_polydat(r#"
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
cycle=0      → device=0  reading=0
cycle=1      → device=1  reading=0    (devices fill before readings advance)
cycle=99     → device=99 reading=0
cycle=100    → device=0  reading=1
cycle=49999  → device=99 reading=499
cycle=50000  → device=0  reading=500  (second dim unbounded with radix `0`)
```

The trailing `0` in `mixed_radix(cycle, 100, 0)` declares the second
dimension as unbounded — it grows indefinitely rather than wrapping.
Replace it with a finite cardinality to cap the total space.

`partition` nodes (see [nodes.md §Determinism](nodes.md#determinism-and-sampling))
give type-checked access to the resulting cells.

See [`examples/parameter_space.rs`](../examples/parameter_space.rs).

## Partitioning a domain across fibers

A cursor names an ordinal domain. A partition spec splits that domain
into stable, absolute intervals that fibers can own without talking to
each other. The `partitions` node resolves a spec against an extent
and returns the list as a value:

```rust
let mut k = compile_polydat(r#"
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
`over` spec and writing those slots is the host's job at scope setup;
`cursor_over_partitions` and `narrow_cursor` are that step, and this
example performs it for fiber 1:

```rust
let mut k = compile_polydat(r#"
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
let program = k.program().clone();
let schema = &program.cursor_schemas()[0];
let parts = cursor_over_partitions(&program, k.state(), schema).expect("resolve");
narrow_cursor(&program, k.state(), "q", &parts[1]);

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

`mod_in` wraps the fiber's local cycle into its absolute slice, so the
fiber never touches another fiber's ordinals and every row it produces
can be regenerated from its ordinal alone. `subdivide` splits the slice
again with the same boundary math, which is how a fiber hands work to
worker threads. The full spec language, including recipes, windows,
gaps, and ordering, is in
[Cursor Partitions](design/cursor_partitions.md).

See [`examples/cursor_partitions.rs`](../examples/cursor_partitions.rs).

## Traversing a parameter space

A comprehension names a parameter space and how to walk it. The
algebra has four constructors, `cartesian`, `zip`, `union`, and
`filter`, plus `order`, which applies a named traversal strategy.
In the text form that hosts embed, the same space reads:

```text
for k in 1..4, limit in 10..40 step 10
for k in 1..4, limit in 10..40 step 10 order halton/5
for k in 1..4, limit in 10..40 step 10 where {k} >= 2 && {limit} != 20
```

Built directly, dispensed as coordinate tuples:

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
covers the space evenly at any truncation. A `CoordinateStream` yields
the tuples; a `ScopedKernelStream` yields a kernel instance per tuple
with the coordinates already bound. Both are specified in
[Comprehension Forms](design/comprehension_forms.md).

See [`examples/parameter_space_traversal.rs`](../examples/parameter_space_traversal.rs).

## A comprehension producer

`name := for ...` binds a comprehension as a value. The wire carries a
Streamer: the comprehension's text and validated algebra, with stream
factories and cardinality metadata. Derivations are written with the
same keyword over a bound producer.

```rust
let mut kernel = polydat::dsl::compile_polydat(r#"
    input cycle: u64

    base    := for k in 1..4, limit in 10,20,30
    corners := for base where {k} == 1 || {k} == 3
    sampled := for base order halton/4
    label   := "plan: {base}"
"#).expect("compile failed");

kernel.set_inputs(&[0]);
println!("{}", kernel.pull("label").as_str());
for name in ["base", "corners", "sampled"] {
    let value = kernel.pull(name).clone();
    show(name, value.as_streamer().expect("streamer"));
}

// Two streams from one wire never share a cursor.
let value = kernel.pull("base").clone();
let base = value.as_streamer().unwrap();
let mut a = base.coordinate_stream();
let b = base.coordinate_stream();
a.next();
a.next();
println!("after two pulls on a: a has {} left, b has {}", a.count(), b.count());
```

where `show` prints the cardinality class and dispenses the stream:

```text
plan: for k in 1..4, limit in 10,20,30
base     Bounded(9)   9 tuples  (1,10) (1,20) (1,30) (2,10) (2,20) (2,30) (3,10) (3,20) (3,30)
corners  BoundedAtMost(9)   6 tuples  (1,10) (1,20) (1,30) (3,10) (3,20) (3,30)
sampled  Bounded(4)   4 tuples  (2,20) (1,30) (3,10) (1,20)
after two pulls on a: a has 7 left, b has 9
```

The producer is an init-time constant, so it costs nothing per cycle,
and it interpolates as its `for` text. See
[`examples/for_producer.rs`](../examples/for_producer.rs).

## A traversal

`for <comprehension> { body }` activates one child scope per tuple. The
body compiles once, at parent compile time, into its own program. Each
activation is a fresh state over that program with the tuple's elements
bound as typed wires and the parent's referenced wires cascaded in. A
cursor declared `over` an element is narrowed per activation, and its
slice sets how many cycles the activation runs.

```rust
let mut kernel = polydat::dsl::compile_polydat(r#"
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

let built_before = programs_built();
let mut stream = kernel.traverse(0).expect("open traversal");
while let Some(mut act) = stream.advance().expect("activation") {
    let slice = act.cursor.clone().expect("cursor slice");
    let scale = act.coord("scale").unwrap().as_u64();
    let kernel = act.cycle(0);
    // ... print index, slice, scale, cycle count, row, v
}
println!("programs built while activating: {}", programs_built() - built_before);
```

```text
body program: 18 nodes, compiled once
8 activations from `p in partitions("*/4", {total}), scale in 1,100`

act  p          scale  cycles  first row  first v
  0  [  0, 250)      1     250          0  7191089600892374487
  1  [  0, 250)    100     250          0  7191089600892374487
  2  [250, 500)      1     250        250  7191089600892374737
  3  [250, 500)    100     250        250  7191089600892399487
  4  [500, 750)      1     250        500  7191089600892374987
  5  [500, 750)    100     250        500  7191089600892424487
  6  [750,1000)      1     250        750  7191089600892375237
  7  [750,1000)    100     250        750  7191089600892449487

programs built while activating: 0
```

The comprehension's source reads the parent's `total` through `{total}`
when the traversal is opened, so a host can change the extern and open
it again without recompiling. Fibers partition a traversal by taking
activations by index. The full contract is [The `for`
Construct](design/for_traversal.md). See
[`examples/for_traversal.rs`](../examples/for_traversal.rs).

## A context layering API

A library function can be composed into a parent kernel as a sub-DAG.
Each invocation site provides its own inputs and gets its own
outputs — same compiled function, layered over a different context
at each call.

```rust
let stdlib = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
    .join("stdlib").join("identity.polydat");

let mut kernel = polydat::dsl::compile_polydat_with_libs(
    r#"
        input cycle: u64
        (tenant, device) := mixed_radix(cycle, 100, 0)
        // Same compiled function, two layered call sites.
        tenant_id := hashed_id(tenant, 10000)
        device_id := hashed_id(device, 10000)
    "#,
    None, vec![stdlib], &[], false, "context layering",
).expect("compile failed");
```

The polydat-direct form is single-file kernel composition like the
above. The heavier `polydat::subcontext` API (`SubcontextBuilder`,
`ScopeKernel`, `PolydatMatter`) is what nbrs uses for layering entire
scope-trees over parent kernels — with typed import/export contracts
at every boundary. The principle is the same; the multi-scope case
just needs more bookkeeping.

See [`examples/context_layering.rs`](../examples/context_layering.rs).

## An expression language

The `:=` lines in Polydat are full expressions, not just direct node calls.
You can nest, chain, and combine without naming every intermediate
wire — useful for one-line derivations where naming the intermediate
would just be noise.

```rust
let mut kernel = polydat::dsl::compile_polydat(r#"
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
    0   527897  ZERO          1
    1   460078  ONE           0
    2   564547  TWO           1
    3   960189  THREE         1
    4   862456  FOUR          1
```

See [`examples/expression_language.rs`](../examples/expression_language.rs).
