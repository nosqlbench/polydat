# Embedding Polydat

Polydat is a library first. The binary is one host among many, and
everything it does is available to any Rust program that links the
crate. This guide shows what a host is responsible for, what Polydat is
responsible for, the APIs at the boundary between them, and the points
where a host extends the kernel. Every program and every line of output
below is produced by `examples/embedding_guide.rs`:

```sh
cargo run --all-features --example embedding_guide
```

## 1. Who owns what

Polydat owns the meaning of a program: given a coordinate, a compiled
program produces the same values everywhere, on every engine, on every
thread. The host owns everything that touches the outside world.

| Concern | Owner | Where it shows up |
| --- | --- | --- |
| Which coordinates to visit, in what order, on which threads | host | §3, §4, §9 |
| Parsing, typing, and compiling the program | Polydat | §2, §8 |
| Selecting engines (native code, closures, the interpreter) | Polydat, unless the host names one | §11 |
| Per-coordinate values and their determinism | Polydat | §2, §4 |
| Extern values and their defaults | host sets, Polydat types | §3 |
| Functions the program may call | Polydat's library plus the host's registered nodes | §5, §6 |
| Modules on disk | host names the directories, Polydat resolves and compiles | §8 |
| Templates that arrive as data | host hands them in, Polydat lowers them | §10 |
| Side effects: sinks, files, network | host | §12 |
| Feature switches: emit formats, defaults, overrides | host, as program transforms | §3, §12 |

The last row is the rule that shapes the rest: a host feature is a
rewrite of the program text or AST that the compiler then sees. Nothing
is written into a running kernel from outside except coordinates and
extern values, and both of those are declared slots with declared
types. See [Runtime Model](../design/runtime_model.md) for the axioms
this rule serves.

## 2. Compile and drive

The smallest host compiles source text, sets a coordinate, and reads
named outputs. `compile_polydat_kernel` returns a `Box<dyn Kernel>` on
the default engine, native code where the build has the `jit` feature
and the closure tier otherwise, ready to run on the calling thread.

```rust
use polydat::dsl::compile_polydat_kernel;

let mut kernel = compile_polydat_kernel(r#"
    input cycle: u64
    user_id := mod(hash(cycle), 1000000)
    score   := unit_interval(hash(user_id))
    label   := "user-{user_id}"
"#)?;
println!("engine: {}", kernel.engine());
println!("inputs: {:?}", kernel.input_names());
for cycle in [0u64, 1, 2] {
    kernel.set_inputs(&[cycle]);
    let user_id = kernel.pull("user_id").as_u64();
    let score = kernel.pull("score").as_f64();
    let label = kernel.pull("label").as_str().to_string();
    println!("cycle {cycle}: user_id={user_id} score={score:.3} label={label}");
}
```

```text
engine: native (Raw)
inputs: ["cycle"]
outputs: ["cycle", "user_id", "score", "label"]
cycle 0: user_id=607535 score=0.601 label=user-607535
cycle 1: user_id=822465 score=0.230 label=user-822465
cycle 2: user_id=348110 score=0.219 label=user-348110
```

`set_inputs` writes the coordinate tuple; `pull` evaluates the output's
cone on demand and caches until an input the output depends on is
written again, so pulling the same wire twice costs one evaluation.
Values come back as
an owned `Value` with typed accessors (`as_u64`, `as_f64`, `as_str`,
`as_bool`) that panic on a type mismatch, which is a host bug: the
program's types are known at compile time and the host is expected to
read what it declared. `engine()` reports what the selector built, here
native code with push-pull provenance; §11 names the engines.

`pull` looks its name up on every call. A host that reads the same
outputs every cycle resolves each name once with `output_index` and
pulls with `pull_at`, so no lookup runs per cycle. The index is the
output's position in `output_names`, which lists the anonymous
intermediate wires too, so the indices are not dense. The binary's
fibers drive their runs this way.

```rust
let at: Vec<usize> = ["user_id", "score", "label"]
    .iter()
    .map(|n| kernel.output_index(n).expect("a named output"))
    .collect();
println!("output indices: {at:?}");
for cycle in [3u64, 4] {
    kernel.set_inputs(&[cycle]);
    let user_id = kernel.pull_at(at[0]).as_u64();
    let score = kernel.pull_at(at[1]).as_f64();
    let label = kernel.pull_at(at[2]).as_str().to_string();
    println!("cycle {cycle} by index: user_id={user_id} score={score:.3} label={label}");
}
```

```text
output indices: [2, 4, 5]
cycle 3 by index: user_id=139053 score=0.350 label=user-139053
cycle 4 by index: user_id=603978 score=0.869 label=user-603978
```

`output_index` returns `None` for a name the program does not bind;
`pull_at` with an index past the outputs is a panic, a host bug of the
same kind as a typed accessor on the wrong variant.

The other entry points differ in what they take:

| Function | Adds |
| --- | --- |
| `compile_polydat_kernel_with_options(src, &options, log)` | `CompileOptions`: the source directory, so relative `import` paths resolve; library directories (§8); required outputs; strict typing, where implicit adapters are errors; a context label for errors; a cursor limit; and the compile ledger to charge (`ledger`), by default a new one for the program tree, or the parent's when a host compiles a subscope. And the compile event log (§13). |
| `parse_polydat(src)` | stops after the parse and returns the program tree, the form a transform rewrites (§9); `parse_polydat_with_tile_defaults(src, &defaults)` sets the delimiters a tile body is read with |
| `compile_polydat_with(src, engine)` | the engine by name (§11); `compile_polydat_with_engine(src, engine, &options, log)` takes the options and the log as well |
| `compile_polydat(src)` | the interpreter through the trait: the reference implementation every other engine is checked against (§11). |
| `compile_polydat_interpreter(src)`, `compile_polydat_interpreter_with_options(src, &options, log)`, `compile_polydat_interpreter_with_log(src, log)` | the interpreter kernel, `PolydatKernel`, as a concrete type. Every one says `interpreter`, because the concrete type is for observing the interpreter's own internals in testing and diagnostics (§13), not for running a program. |
| `compile_ast_interpreter_with_options(&ast, src, &options, log)` and `compile_ast_with_engine(&ast, src, &options, log, engine)` | the same compiles from a parsed, possibly transformed, program (§9) |
| `compile_polydat_to_assembler(src)` | stops before engine selection and returns the assembler (§7, §11), the graph a host may extend by hand and build on any engine |
| `compile_polydat_to_assembler_with(src, &options)` | the assembler built with the same `CompileOptions` |

Every one of these is one compile path: the same prologue applies the
options, the same assembly builds the graph, the same lowering attaches
the traversals and resolves the cursor extents, on every engine. The
options mean the same thing whichever entry point carries them, and
`strict` refuses the same programs on every engine.

## 3. Externs

An `extern` is a typed input slot with a default. The host may overwrite
it per kernel, and a program transform may fix it before compilation.
Both roads lead to the same slot, so the kernel never learns which one
the host took.

```rust
let src = r#"
    input cycle: u64
    extern region: str = "us-east"
    extern scale: u64 = 10
    id := mod_wire(hash(cycle), scale)
    key := "{region}/{id}"
"#;
let mut kernel = compile_polydat_kernel(src)?;
kernel.set_inputs(&[7]);
println!("defaults: {}", kernel.pull("key").as_str());

kernel.set_input("region", Value::Str("eu-west".into()))?;
kernel.set_input("scale", Value::U64(1000))?;
kernel.set_inputs(&[7]);
println!("overridden: {}", kernel.pull("key").as_str());

// The binary's `name=value` arguments are this transform.
let transformed = src.replace(r#"extern region: str = "us-east""#, r#"extern region: str = "ap-south""#);
let mut fixed = compile_polydat_kernel(&transformed)?;
fixed.set_inputs(&[7]);
println!("transformed: {}", fixed.pull("key").as_str());

// The interpreter has the same slots behind the same calls.
let mut p1 = compile_polydat_with(src, Engine::Interpreter(JitMode::Auto))?;
p1.set_input("region", Value::Str("eu-west".into()))?;
p1.set_input("scale", Value::U64(1000))?;
p1.set_inputs(&[7]);
println!("interpreter, overridden: {}", p1.pull("key").as_str());
```

```text
defaults: us-east/7
overridden: eu-west/487
transformed: ap-south/7
interpreter, overridden: eu-west/487
```

Every engine treats an extern the same way. A compiled kernel gives it
slots after the coordinates, seeds the default when the kernel is
built, and `set_input` by name writes a number or boolean into its slot
at once and a string, JSON, or extension value into the value the
kernel stores for it, with the slots pointing at that value until the
next write; the interpreter keeps it as a value in the state. Setting
an extern marks
everything downstream of it for recomputation. `set_input` checks the
value against the declared port type on every engine and refuses a
mismatch by name with `WriteError::TypeMismatch`; nothing converts a
value at the write (below, "When an input's type varies"). An extern declared without
a default, such as `extern doc: json`, is `None` until the host sets
it, and every consumer reads `None` through it, on the interpreter, the
closure tier, and the hybrid kernel alike; pure native code cannot
carry `None` and refuses to run until it is set. The compile log (§13)
names every extern without a default, so a host knows what it must
set. A constant no input reaches is folded when a kernel is built, on
every engine, so a failure there surfaces at build; what depends on an
extern is computed at the first pull that needs it and kept until the
extern changes.

Prefer the transform when the value is fixed for the run: the compiler
then sees a constant, folds it, and the native segments carry it as an
immediate. Use `set_input` when the value genuinely varies per kernel,
such as a per-thread shard label.

When the value varies per cycle, resolve the slot once with
`input_index` and write it with `set_input_at`. The index is the slot's
position in `input_names`, the coordinates first, so an extern's index
follows them. The write is the same typed write as `set_input`, with
the same refusal of a mismatch, and no lookup runs per cycle.

```rust
let region = kernel.input_index("region").expect("a declared extern");
let key = kernel.output_index("key").expect("a named output");
println!("input index of region: {region}");
for (cycle, name) in [(8u64, "us-east"), (9, "eu-west"), (10, "ap-south")] {
    kernel.set_inputs(&[cycle]);
    kernel.set_input_at(region, Value::Str(name.into()))?;
    println!("cycle {cycle} by index: {}", kernel.pull_at(key).as_str());
}
```

```text
input index of region: 1
cycle 8 by index: us-east/622
cycle 9 by index: eu-west/228
cycle 10 by index: ap-south/466
```

### When an input's type varies

The design is [input_variance.md](../design/input_variance.md). The
interpreter's `Dataflow::set_wire`, which converted at the write, is
deprecated in favor of what follows.

A declared input's type is fixed for the kernel's lifetime, and a write
that does not match it is a bug the refusal names. Some hosts write
values whose type legitimately varies: a parameter that arrives as text
in one run and as a number in the next, or a parent scope's value copied
into a child that typed it differently. Polydat converts such a value in
the program, never at the write, and only where the host asked for it:

- **Convert before writing** when you know the target type.
  `polydat::convert::to_port(value, port_type)` applies the same
  conversions the compiler's adapters use and returns an error you can
  handle at the write, instead of a node failure at the pull. This is the
  trusted surface for host-side conversion.
- **Convert one input in the program** with the transform
  `transform::convert_input(&mut file, "name")`. The input then accepts
  any value, and a converter node, a closure step, converts it for its
  consumers. The compile log reports the converter.
- **Open every input the author left untyped** with
  `CompileOptions::input_variance`. An input whose type the compiler
  inferred rather than the author wrote is *open*: an auto-extern, or an
  extern a synthesizer declared and named in
  `CompileOptions::inferred_externs` (the scope builder names its result
  and write-through externs this way; a host that writes source with
  types it knows are exact leaves them out). Coordinates are never open.
  The setting decides what the compiler does with open inputs:

  | `input_variance` | Open inputs |
  |---|---|
  | `Fixed` (default) | take their inferred type; a mismatched write is refused |
  | `Error` | stop construction, with every open input named |
  | `Warn` | get a converter each, reported as a warning |
  | `Info` | get a converter each, reported as info |

Declared inputs are never converted under any setting. A converter costs
a closure step on the compiled engines, so conversion is something you
ask for, input by input or by setting, and never assumed. A converted
input still reports the type its readers see through `input_port_type`,
and `input_type_origin` says whether it was declared or inferred. A host
that drives scope trees with parameters of varying type, as nmbrs does,
compiles with `Warn`, converts what it can through `convert::to_port`,
and reads the compile log to see every input it relies on converting. A
host that compiles many scopes may compile at `Info` instead and report
each open input once from `input_type_origin`.

A `cursor` declared `over` a literal spec is resolved at build. The
assembler and every kernel list each cursor with its partitions through
`cursor_schemas`; a clause that denotes one partition seeds the cursor,
so the program runs on every engine with no host call; a clause that
denotes several is narrowed with `set_cursor(name, &partition)`, the
same call on every engine. [Cursor Partitions](../design/cursor_partitions.md) §7.2 has the
rules.

## 4. Share a program across threads

A program is immutable once compiled and is shared through an `Arc`:
`into_program` turns a kernel into an `Arc<dyn KernelProgram>`, and
`create_kernel` gives each thread a kernel of its own over the shared
steps or native code. There are no locks on the evaluation path, and
because values depend only on the coordinate, the partition of cycles
across threads is invisible in the results.

```rust
let program = compile_polydat_kernel("input cycle: u64\nv := mod(hash(cycle), 1000)\n")?.into_program();
let sums: Vec<u64> = std::thread::scope(|s| {
    let handles: Vec<_> = (0..threads).map(|t| {
        let program = program.clone();
        s.spawn(move || {
            let mut kernel = program.create_kernel();
            let mut sum = 0u64;
            for c in t * per_thread..(t + 1) * per_thread {
                kernel.set_inputs(&[c]);
                sum += kernel.pull("v").as_u64();
            }
            sum
        })
    }).collect();
    handles.into_iter().map(|h| h.join().unwrap()).collect()
});
```

```text
4 threads x 25000 cycles: sum 49908479; serial sum 49908479; equal: true
```

The serial sum is the same cycles on one kernel. This is the whole
concurrency story for a host: decide which thread gets which
coordinates, create one kernel per thread, and never share a kernel.
`examples/multi_thread.rs` shows the same pattern at a million cycles
per thread, with timing. The interpreter's program has the same shape
under its own names, `PolydatProgram` and one `PolydatState` per
thread through `create_state`.

### Scope trees on any engine

A host that runs workloads as a tree of scopes (parameters, a phase,
fibers, per-operation children, one child per iteration tuple) binds
each child under its parent. Binding attaches the parent's shared cells,
copies in the values the child imports, and materializes the child's
scope-init constants. It works on every engine, and a tree may mix them,
since every step is over `dyn Kernel`:

| To | Call |
|---|---|
| build a child from source under a parent | `SubcontextBuilder::under(parent)`, then `.body(…)`, `.add_result_bindings(…)`, `.finalize()` to a `ScopeModule` |
| instantiate it, on an engine you name | `module.instantiate_under(parent, engine, &iteration_bindings)` |
| bind a precompiled program under a parent | `kernel::bind_under(parent, program, &iteration_bindings)` |
| carry a parent's extern values into a child | `kernel::propagate_inputs(parent, child)` |
| start a fiber from a scope's kernel | `kernel.fork()` |
| read a name the way a scope resolves it | `KernelLookup::new(kernel).lookup(name)` |

A module compiles once per engine and each instance costs only a state,
so instantiating one per iteration tuple compiles nothing after the
first. Per cycle, a scope kernel is driven by index through `pull_at`,
`input_value_at`, `set_input_at`, `reset_inputs`, `publish_broadcasts`,
and `commit_write_throughs`. None of these looks a name up. A plan of
indices resolved once is sealed against `program_id()`, which is equal
for every kernel of one program and for its forks.

The rule above, one kernel per thread, is about kernels you evaluate. A
scope *parent* is different: every `Kernel` is `Sync`, so a parent may be
held in an `Arc` and have children bound and forks taken under it from
many threads at once. What each thread then evaluates is its own child
or fork. [Scope trees on any engine](../design/native_scope_trees.md)
is the design, and `tests/scope_trees.rs` walks a whole tree on every
pair of engines.

## 5. Host-defined nodes

The function library is open. A host defines a node with the same
attribute the crate's own library uses; the attribute generates the
node's metadata, its interpreter body, and its closure form, and
registers it at link time under the function's name. Programs compiled
anywhere in the process can then call it.

```rust
#[polydat::polydat_node(category = Math)]
fn host_checksum(a: u64, b: u64) -> u64 {
    a.rotate_left(7) ^ b.wrapping_mul(0x9E37_79B9_7F4A_7C15)
}

#[polydat::polydat_node(category = String)]
fn host_tag(prefix: &str, n: u64) -> String {
    format!("{prefix}-{n:04}")
}

let mut kernel = compile_polydat_kernel(r#"
    input cycle: u64
    c := host_checksum(cycle, hash(cycle))
    t := host_tag("job", mod(c, 10000))
"#)?;
```

```text
cycle 1: c=11416446865527871317 t=job-1317
cycle 2: c=16311599965266896870 t=job-6870
```

Parameter types follow the wire types: `u64`, `f64`, `bool`, `&str`,
`Value`, `Const<T>` for arguments that must be literals, `Ext<T>` for
extension values, and `&[Value]` for a variadic tail. The return type
is the output wire's type. Attribute options declare the category, the
purity (the default is pure; `SideChannel` marks a node that writes
somewhere the kernel does not see, as the emit node in §12 does), and
constraints such as `#[constraint(NonZeroU64)]` on a parameter. A rule
that relates two constants rather than bounding one goes in
`validate = <path>`, a function the factory calls with the node's
constant arguments. Both are read when the node is built, so they hold
on every engine; a check written into the body instead only fires on
the engines that run the body.

Every node the attribute accepts runs on the interpreter, the closure
tier, and hybrid kernels: the interpreter calls the body, and the
closure tier calls the generated closure, which carries scalars in
slots and strings, JSON, and host values as the reference pairs of
[Compiled By-Reference Slots](../design/compiled_handles.md) (§5
there describes the closure kits). Pure native code runs a node only
when it has a native lowering; a host never needs one for correctness,
only for speed, and [Engines](../design/engines.md) §8 records
what each engine accepts.

## 6. Host-defined value types

Nodes are not the only thing a host can add. A host type becomes a wire
value by implementing `ReflectedValue`; the wire's port type is `Ext`,
nodes take and return it through `Ext<T>`, and everything generic in the
kernel reaches it through the trait: string interpolation calls
`display`, JSON encoding calls `to_json_value`, and cloning goes through
`clone_reflected`. The host gets the concrete type back with `as_any`.

```rust
use polydat::ast::ReflectedValue;
use polydat::derive_support::Ext;

#[derive(Debug, Clone, PartialEq)]
struct GeoCell { lat_deg: f64, lon_deg: f64, level: u64 }

impl ReflectedValue for GeoCell {
    fn type_name(&self) -> &str { "GeoCell" }
    fn display(&self) -> String { format!("cell({:.3}, {:.3}, L{})", self.lat_deg, self.lon_deg, self.level) }
    fn to_json_value(&self) -> serde_json::Value {
        serde_json::json!({ "lat": self.lat_deg, "lon": self.lon_deg, "level": self.level })
    }
    fn clone_reflected(&self) -> Box<dyn ReflectedValue> { Box::new(self.clone()) }
    fn as_any(&self) -> &dyn std::any::Any { self }
}

#[polydat::polydat_node(category = Math, struct_name = GeoCellNode)]
fn geo_cell(lat: f64, lon: f64, level: u64) -> Ext<GeoCell> {
    Ext(GeoCell { lat_deg: lat, lon_deg: lon, level })
}

#[polydat::polydat_node(category = String)]
fn cell_token(cell: Ext<GeoCell>) -> String {
    let scale = (1u64 << cell.level) as f64;
    let row = ((cell.lat_deg + 90.0) / 180.0 * scale) as u64;
    let col = ((cell.lon_deg + 180.0) / 360.0 * scale) as u64;
    format!("L{}:{row}:{col}", cell.level)
}

let mut kernel = compile_polydat_kernel(r#"
    input cycle: u64
    lat  := unit_interval(hash(cycle)) * 180.0 - 90.0
    lon  := unit_interval(hash(cycle + 1000)) * 360.0 - 180.0
    cell := geo_cell(lat, lon, 6)
    tok  := cell_token(cell)
    line := "{tok} is {cell}"
"#)?;
for cycle in [0u64, 1] {
    kernel.set_inputs(&[cycle]);
    println!("cycle {cycle}: {}", kernel.pull("line").as_str());
    // The host reads the wire as its own type again.
    let Value::Ext(boxed) = kernel.pull("cell") else { panic!("cell is an Ext wire") };
    let cell = boxed.as_any().downcast_ref::<GeoCell>().expect("a GeoCell");
    println!("cycle {cycle}: level {} at ({:.1}, {:.1}); json {}", cell.level, cell.lat_deg, cell.lon_deg, boxed.to_json_value());
}
println!("cell wire type: {:?}", kernel.output_type("cell"));
```

```text
cycle 0: L6:56:15 is cell(68.996, -95.456, L6)
cycle 0: level 6 at (69.0, -95.5); json {"lat":68.9959454784557,"lon":-95.45620227479236,"level":6}
cycle 1: L6:36:20 is cell(11.981, -62.941, L6)
cycle 1: level 6 at (12.0, -62.9); json {"lat":11.981083531010583,"lon":-62.94065304500829,"level":6}
cell wire type: Some(Ext)
```

The attribute generates a struct named after the function in PascalCase,
which for `geo_cell` would be `GeoCell`, the value type itself. The
`struct_name` parameter picks another Rust name; the DSL name is always
the function name.

Three things to know about extension values:

- **The type check is by name at compile time and by downcast at run
  time.** The compiler sees only `Ext`, so wiring a `GeoCell` into a node
  that expects a `Partition` compiles. The downcast in `Ext<T>::extract`
  then panics with both type names. Give distinct host types distinct
  producers and consumers, and keep them in one crate so the downcast
  can see the concrete type.
- **Extension nodes run on every engine through their closure.** An
  extension value rides the compiled engines as a reference pair to
  the value its producing step owns, the same way JSON does. The
  closure tier runs a node with an `Ext` signature as a closure step,
  and native code calls the same closure in place, inside the segment
  with the hashing and scaling around it, passing the value between
  the nodes as the pair described in
  [Compiled By-Reference Slots](../design/compiled_handles.md).
- **The value is cloned on every read.** `Ext<T>::extract` clones the
  boxed value, so a large host type should hold its payload in an `Arc`.
  The crate's own `Partition` and `Streamer` values do exactly that.

### Naming port types without matching them

`PortType` is exhaustive on purpose. The language adds a type now and
then, and a `match` over every variant stops compiling until it decides
what the new type means, which is what polydat wants of its own code and
of host code that behaves differently per type. A host that only labels,
parses, or classifies types should not match on the enum: `to_keyword()`
and `Display` name a type, `from_keyword()` parses one, and
`numeric_domain()` and the `SlotShape` queries classify one. Code written
that way keeps compiling when a type is added, and it labels the new type
correctly without a change.

## 7. The assembler API

Source text is one front end. The assembler builds the same graph from
node instances and wire references, which suits hosts that generate
programs from their own configuration rather than from text.

```rust
use polydat::compile::assembly::{PolydatAssembler, WireRef};
use polydat::library::{arithmetic::Mod, hash::Hash};

let mut asm = PolydatAssembler::new(vec!["cycle".into()]);
asm.add_node("hashed", Box::new(Hash::new()), vec![WireRef::input("cycle")]);
asm.add_node("user_id", Box::new(Mod::new(1_000_000)), vec![WireRef::node("hashed")]);
asm.add_output("user_id", WireRef::node("user_id"));
let mut kernel = asm.compile_kernel()?;
kernel.set_inputs(&[42]);
```

```text
user_id at cycle 42: 275413
```

The DSL compiler produces exactly this assembler, so
`compile_polydat_to_assembler` is the point where a host can inspect or
adjust the graph between parsing and engine selection. `compile_kernel()`
builds on the default engine, `compile_with(engine)` names one, and
`compile()` builds the interpreter kernel, with `set_jit_mode` choosing
how much of that graph runs as native cones.

## 8. Modules from a library directory

Modules are `.polydat` files whose top-level definitions become callable
functions. The host names the directories; the compiler resolves calls
against them, compiles each module once, and inlines it with the
caller's types.

```rust
std::fs::write(dir.join("bucketed.polydat"), r#"
bucketed(input: u64, buckets: u64) -> (bucket: u64, label: str) := {
    bucket := mod(hash(input), buckets)
    label := "b{bucket}"
}
"#)?;
let mut kernel = compile_polydat_kernel_with_options(
    "input cycle: u64\n(b, l) := bucketed(input: cycle, buckets: 8)\n",
    &CompileOptions {
        lib_paths: vec![dir],   // library directories, searched in order
        context: "embedding guide".into(),
        ..CompileOptions::default()
    },
    None,
)?;
```

```text
cycle 0: bucket=7 label=b7
cycle 1: bucket=1 label=b1
cycle 2: bucket=6 label=b6
```

The binary's `--lib` flag is this option. See
[Module System](../design/module_system.md) for resolution order and the
rules for named and positional arguments.

## 9. Traversal

A `for` statement declares a traversal: a comprehension over coordinates
with a body that compiles once. The host opens it, receives one
activation per tuple, and drives each activation's cycles itself. This
is the division of labor from §1 made concrete: Polydat enumerates the
tuples deterministically and types the body, the host decides when and
where each activation runs.

```rust
let mut kernel = compile_polydat_kernel(r#"
    input cycle: u64
    for shard in 0..3, phase in load,verify {
        row := mod(hash(cycle), 100) + shard * 100
        stmt := "{phase} shard {shard} row {row}"
    }
"#)?;
kernel.set_inputs(&[0]);
let stream = kernel.traverse(0)?;
for index in 0..stream.len() {
    let mut act = stream.activate(index)?;
    let k = act.cycle(1);
    println!("  activation {index}: {}", k.pull("stmt").as_str());
}
```

```text
6 activations of `shard in 0..3, phase in load, verify`
  activation 0: load shard 0 row 65
  activation 1: verify shard 0 row 65
  activation 2: load shard 1 row 165
  activation 3: verify shard 1 row 165
  activation 4: load shard 2 row 265
  activation 5: verify shard 2 row 265
```

`activate(index)` builds the activation at `index` on the default
engine; `activation_on(index, engine)` names one, and `activation(index)`
and `advance()` build the interpreter's. Every activation is a kernel
over the body's program, compiled once per engine and driven through
the `Kernel` trait, computing what the interpreter's computes. Fibers
partition a traversal by activating disjoint index ranges. `traverse(i)`
opens the i-th traversal in the program; it is a `Kernel` trait method,
so a kernel on any engine opens its traversals the same way, and an
activation opens the `for` statements of its own body. Each activation
exposes its coordinates (`act.coord("shard")`), its cursor slice when
the body declares a cursor, its cycle count, and `act.cycle(n)`, which
returns the activation's kernel positioned at cycle `n`. The body
program is compiled once per traversal position and engine, not once
per activation; `examples/for_traversal.rs` measures that.
[The `for` Construct](../design/for_traversal.md) has the full contract.

## 10. Tiles from host data

A tile is a template whose holes are wires. In source it is a `tile`
statement; at the host boundary it may arrive as text, as JSON text, or
as an already-parsed JSON value, and each form becomes the same tile
statement before compilation. This is how a host lets its users write
templates in their own configuration format without teaching the host
anything about rendering.

```rust
use polydat::dsl::{CompileOptions, compile_ast_with_engine, parse_polydat};
use polydat::tile::{add_tiles, tile_from_json_value, Span, TileOptions};

let template = serde_json::json!({
    "id": "${cycle}",
    "label": "row-${cycle}",
    "points": [ "@for s in 0..2", { "n": "${s}", "v": "${cycle + s}" } ]
});
let tile = tile_from_json_value("doc", &template, &TileOptions::default(), Span { line: 0, col: 0 })?;
let source = "input cycle: u64\n";
let mut program = parse_polydat(source)?;
add_tiles(&mut program, vec![tile])?;
let mut kernel = compile_ast_with_engine(
    &program, source, &CompileOptions::default(), None, polydat::Engine::default())?;
kernel.set_inputs(&[4]);
println!("doc: {}", kernel.pull("doc").as_str());
```

```text
doc: {"id": 4, "label": "row-4", "points": [{"n": 0, "v": 4},{"n": 1, "v": 5}]}
```

`tile_from_text` and `tile_from_json_text` are the other two entry
points. The [Polytile tutorial](../tutorials/polytile_tutorial.md) covers
the template language; [Polytile](../design/polytile.md) §3 specifies the
structural JSON form the value above uses.

## 11. Compiled kernels

A host normally lets `Engine::default()` choose: P3 with the `jit`
feature, the closure tier without. The engines are named by one type,
`Engine`. `Native` is P3, native code for every node that has a lowering
with the node's closure elsewhere; `Closures` is P2, the closure tier;
each takes a `Provenance` that says how much work repeated inputs skip
(`Auto` lets the selector choose); and `Interpreter` is P1, the oracle.
One constructor, `compile_polydat_with(src, engine)` or
`compile_with(engine)` on the assembler, builds a `Box<dyn Kernel>` on
any of them or returns one error type, `KernelError`, whose `Refused`
variant names the engine and the node or construct it cannot run. The
`Kernel` trait is the same calls on every engine: `set_inputs`,
`set_input`, `set_cursor`, `eval`, `pull`, `traverse`, and the name and
type listings. The program below includes `host_tag` from §5 and the two
extension nodes from §6, which have no native lowering of their own; P3
reaches them through their kits from native code:

```rust
let src = r#"
    input cycle: u64
    h := hash(cycle)
    name := "user-{h}"
    tag := host_tag("job", mod(h, 10000))
    cell := geo_cell(to_f64(mod(h, 180)) - 90.0, to_f64(mod(h, 360)) - 180.0, 4)
    tok := cell_token(cell)
    tile j : json := {"h": ${h}, "name": ${name}, "tag": ${tag}, "cell": ${tok}}
"#;
let mut kernels: Vec<Box<dyn Kernel>> = Vec::new();
for engine in [
    Engine::Closures(Provenance::Auto),
    Engine::Native(Provenance::Auto),
] {
    match compile_polydat_with(src, engine) {
        Ok(k) => kernels.push(k),
        Err(e) => println!("{e}"),
    }
}
let p3 = compile_polydat_kernel(src)?;
println!("Engine::default() is {}", p3.engine());
println!("P3 plan: {}", p3.plan());
let mut p1 = compile_polydat_with(src, Engine::Interpreter(JitMode::Auto))?;
for cycle in [0u64, 1] {
    p1.set_inputs(&[cycle]);
    let want = p1.pull("j").to_display_string();
    println!("cycle {cycle}: j={want}");
    for k in kernels.iter_mut() {
        k.set_inputs(&[cycle]);
        let got = k.pull("j").to_display_string();
        println!("  {} agrees: {}", k.engine(), got == want);
    }
}
```

```text
Engine::default() is native (Pull)
P3 plan: 2 native segment(s)
cycle 0: j={"h": 16294208416658607535, "name": "user-16294208416658607535", "tag": "job-7535", "cell": "L4:10:13"}
  closures (Pull) agrees: true
  native (Pull) agrees: true
cycle 1: j={"h": 10451216379200822465, "name": "user-10451216379200822465", "tag": "job-2465", "cell": "L4:0:8"}
  closures (Pull) agrees: true
  native (Pull) agrees: true
```

Both engines accept the program and compute what the interpreter
computes; `engine()` on a kernel reports the provenance the selector
chose, the same choice on either compiled engine: this graph has one
input, so the cone guard alone (`Pull`); the smaller first example above
gets `Raw`. P3 runs this program as two native segments: the literals,
folded once at build, and everything else, which is one connected
group. The three host nodes are part of it: each has a kit, and native
code calls a kit where it would otherwise emit an instruction, with the
extension value passing between two of them as a reference pair.
`plan()` is the only planning detail
a kernel exposes, on every engine, so a host can see whether a program is mostly native before
deciding to care. An engine that cannot run a program at all, either
compiled engine on an extern of a 128-bit integer or register word
type for one, says so as the error, with the engine and the port
named. A `shared` binding runs on every
engine with the interpreter's cell: `shared_cells` on a kernel lists
its cells and `attach_shared_cell` binds one kernel's cell into
another, so both read and write one register.

`pull` evaluates the output's cone and no more on every engine: a side
channel in the
cone fires when the output is pulled, and a failing node fails when
pulled, as on the interpreter, with the same message: the original
panic, the node's name, the outputs it feeds, the program's context,
and its input values (only the `panicked at` line names the engine's
own code). A constant no input reaches is folded
when the kernel is built, on every engine, so a failure there surfaces
at build; a step that depends on an extern is computed at the first
pull that needs it and kept until that extern changes. An
extern without a default is `None` until the host sets it, and every
consumer reads `None` through it; the compile log names each such
extern.

`pull` returns an owned value on every engine: a string, JSON document,
or rendered tile is copied out of the kernel's own storage, so a host
holding a `dyn Kernel` never holds a reference into a kernel, and a
value read before a write is intact after it. The concrete compiled
kernel types keep their raw readers as extras (`get` for a scalar slot,
`get_value` for a typed copy, `eval(&[u64])`); `get` refuses a
by-reference slot, and `get_value` copies it out.

A kernel on any engine shares its program across threads the way §4
shows for the interpreter: `into_program` gives an
`Arc<dyn KernelProgram>`, and `create_kernel` on it gives each thread a
kernel of its own over the shared steps or native code.

The interpreter kernel of `compile()` and `compile_polydat`,
`PolydatKernel`, runs the interpreter over a graph whose native-eligible
regions are cones, per `set_jit_mode` on the assembler (`Auto`, `Off`,
`Force`) and the `jit` Cargo feature; it is the oracle every engine is
checked against, and its program carries the metadata §13 inspects. The
engine a host gets when it names none, `Engine::default()`, is P3 with
the `jit` feature and the closure tier without: `compile_polydat_kernel`
and its `_with_options` and `_with_tiles` forms, `compile_kernel()` on
the assembler, and the `polydat` binary build on it, and the binary opens
every level of a traversal nest on it. [Compilation levels](compilation.md)
describes each engine, [Engines](../design/engines.md) the selection
rules, and [Engines](../design/engines.md) §8 what each engine
accepts.

## 12. Program transforms

A host feature is a rewrite of the program. The emit facility the binary
exposes as `--emit` is nothing more than an appended binding to a
side-channel node, plus a thread-local buffer the host drains.

```rust
let src = "input cycle: u64\nid := mod(hash(cycle), 1000)\nname := \"user-{id}\"\n";
let with_emit = format!("{src}__emit := emit_row(\"jsonl\", \"cycle,id,name\", cycle, id, name)\n");
let mut kernel = compile_polydat_kernel(&with_emit)?;
for cycle in [0u64, 1, 2] {
    kernel.set_inputs(&[cycle]);
    kernel.pull("__emit");   // the pull is what emits
}
// Drain this thread's buffer. Nothing here refers to the kernel.
for row in polydat::library::emit::take_rows() {
    println!("{row}");
}
```

```text
{"cycle":0,"id":535,"name":"user-535"}
{"cycle":1,"id":465,"name":"user-465"}
{"cycle":2,"id":110,"name":"user-110"}
```

Notice that `take_rows` takes no kernel. The emit node never writes
into the state that evaluated it: when the kernel evaluates `__emit`,
the node renders the row and pushes it onto a buffer that is
thread-local to the emit module, and `take_rows` drains the calling
thread's buffer. The only link between the kernel and the rows is that
both ran on the same thread. Four consequences follow:

- **The pull is the trigger.** Nothing is emitted for a cycle unless the
  host pulls the emit wire, or a wire downstream of it, in that cycle.
  A host that forgets the pull gets an empty buffer and no error.
- **Drain on the thread that ran the kernel.** A kernel driven on a
  worker thread leaves its rows in that worker's buffer; `take_rows`
  from another thread returns nothing.
- **Kernels on one thread share one buffer.** Two kernels with emit
  bindings on the same thread interleave their rows in evaluation order.
  There is no per-kernel channel; a host that needs one runs each kernel
  on its own thread or drains between them.
- **Rows accumulate until drained.** A kernel that emits every cycle
  under a host that never drains grows the buffer without bound. The
  binary drains after each cycle.

This shape is deliberate. The node needs no reference back to the state
that evaluated it, so emitting costs one push, and the host gets the
feature with no kernel API for it. The scope is the thread because that
is the unit the host already owns (§4).

The pattern generalizes. To pin a default, rewrite the `extern` line
(§3). To add a derived output, append a binding. To attach a template,
add a tile statement (§10). To restrict what runs, pass required outputs
and let the compiler prune. A transform operates on the **parsed program**, never on the text it
was parsed from. The text is what produced the tree and has no
authority over it afterwards, so re-reading it is not a rewrite of the
program but a second, unrelated parse — which is why a tile keeps no
copy of its body (§10).

`dsl::transform` holds the rewrites the crate ships and the walkers a
host writes its own over:

| | |
|---|---|
| `assign_values(&mut file, &pairs)` | pin an `extern` default, or turn an `input` into one |
| `each_statement(&mut statements, f)` | every statement, descending into module and `for` bodies |
| `each_tile(&mut statements, f)` | every tile the subtree declares |
| `tile_named(&mut statements, "doc")` | one tile, for a transform qualified to it |
| `each_piece(&mut pieces, f)` | every piece of a template, descending into projection bodies and branch arms |

Each takes the slice to walk rather than the whole file, so the caller
chooses the subtree: the program's statements, one module's body, or
one tile's pieces. A rewrite projects forward — a tile's text is
rendered from the pieces a transform touched, so what the program
prints after a rewrite is what it renders.

What a host should not do is reach into a compiled program and change
it: the compiler's provenance, purity, and fusion decisions were made
against the program it saw.

Side effects belong on the host side of the emit buffer. A node marked
`SideChannel` may write to a thread-local the host drains, as `emit_row`
does; it must not read anything that varies between runs, because the
kernel's determinism contract does not know it exists.

## 13. Diagnostics

The compile log records what the compiler did to a program: what it
inlined, folded, fused, and why. The binary's `explain` command narrates
it; a host gets the same events through the log parameter of
`compile_polydat_kernel_with_options`, and `compile_polydat_interpreter_with_log`
fills the same log while building the interpreter's program, whose
metadata answers the questions a host usually has at run time.

```rust
let mut log = CompileEventLog::new();
let kernel = compile_polydat_interpreter_with_log(src, &mut log)?;
println!("compile events: {}", log.events().len());
for event in log.events() {
    println!("  {:?}: {event:?}", event.level());
}
let program = kernel.program();
println!("nodes: {}, deterministic: {}", program.node_count(), program.is_deterministic());
let names: Vec<_> = (0..program.node_count()).map(|i| program.node_meta(i).name.clone()).collect();

let mut compiled_log = CompileEventLog::new();
compile_polydat_kernel_with_options(src, &CompileOptions::default(), Some(&mut compiled_log))?;
println!("compile events on the default engine: {}", compiled_log.events().len());
```

```text
compile events: 18
  Info: Parsed { statements: 4 }
  Info: BindingResolved { name: "h", node_type: "hash" }
  Info: BindingResolved { name: "f", node_type: "f64_div" }
  Info: TileHoleTyped { tile: "t", hole: "h", wire_type: "u64", declared: None, expectation: "any JSON value (u64)", encoder: "json number", adapter: None }
  Info: TileHoleTyped { tile: "t", hole: "f | .2", wire_type: "f64", declared: None, expectation: "any JSON value (f64)", encoder: "json number, format .2", adapter: None }
  Info: TileCompiled { tile: "t", encoding: "json", statics: 3, static_bytes: 14, holes: 2, branches: 0, projections: 0, bodies: [] }
  Info: OutputDeclared { name: "h" }
  Info: OutputDeclared { name: "f__anon_0" }
  Info: OutputDeclared { name: "f" }
  Info: OutputDeclared { name: "t" }
  Info: CompileLevelSelected { node: "const_f64", level: "native" }
  Info: CompileLevelSelected { node: "h", level: "native" }
  Info: CompileLevelSelected { node: "f__anon_0", level: "native" }
  Info: CompileLevelSelected { node: "f", level: "native" }
  Info: CompileLevelSelected { node: "t", level: "native" }
  Info: CompileLevelSelected { node: "cycle", level: "native" }
  Info: ConstantFolded { node: "const_f64", value: "3.0" }
  Info: Summary { nodes: 6, outputs: 5, constants_folded: 1 }
nodes: 3, deterministic: true
node names: ["const_f64", "__port_cycle", "jit_cone[hash+to_f64+f64_div+tile_render]"]
compile events on the default engine: 18
```

The program in this section is a hash, a division, and a JSON tile with
two holes. `CompileEvent` is an enum with one variant per kind of
decision, each carrying its particulars, and `level` classifies it as
info, advisory, or warning. The events here retell the compile in order:
the parse, each binding and the node it resolved to, the two hole
typings, each naming the wire type it saw, what the hole's position
expects, and the encoder it chose; the tile's compiled shape; the outputs
the program exposes; the compiled form each node has, native here for
every one; the one constant the compiler folded; and a summary of the
resolved graph. Advisories, such as an implicit type
widening, and warnings, such as an unknown pragma, arrive in the same
list, so a host that wants a strict build can fail on any event whose
level is a warning. Cone fusion is not an
event: the node list shows what the interpreter's program is actually
running, one constant, the passthrough that exposes the coordinate as
an output, and one native cone that fused the hash, the conversion,
the division, and the tile renderer, whose closure the cone calls in
place over the state's own scratch (compiled_handles.md §6). The log
is the
same on every engine: the default engine records the same events,
from the parse to the summary, the folds its own build made among them,
and every extern without a default is named on every engine. `is_deterministic`
is false when any node's purity is nondeterministic, such as a
wall-clock or a true random source, which is the check a host should
make before relying on replay. Side-channel nodes such as `emit_row`
do not clear it: they write outward but read nothing that varies.

The compiler and the data-source nodes also write an audit log. With no
sink installed it goes to stderr, which is where the `DBG jit cone`
lines come from when the example above runs. A host that has its own
logger installs it once with `library::support::audit::set_log_fn`,
which receives a severity and a line and is called from every thread.

## 14. Where to go next

- [Illustrations](../tutorials/illustrations.md) runs the DSL and the
  assembler through more complete examples.
- [Compilation levels](compilation.md) explains the engines a host is
  choosing between when it names an `Engine`.
- [Runtime Model](../design/runtime_model.md) states the ownership and
  determinism axioms that this guide's division of labor implements.
- `examples/multi_thread.rs`, `examples/for_traversal.rs`, and
  `examples/embedding_guide.rs` are the runnable versions of §4, §9,
  and this whole guide.
