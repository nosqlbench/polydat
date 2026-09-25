---
type: guide
title: Embedding Polydat
timestamp: 2026-09-25
description: What a host owns and what polydat owns, the APIs for compiling, driving, sharing, and extending a kernel, and the extension points.
tags: [host]
---

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
program produces the same values on all four engines (the interpreter,
the closure tier, native, and pure native) and on every thread. The host
owns everything that touches the outside world.

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

The last row shapes most of this guide: a host feature is a rewrite of
the program text or syntax tree, made before the compiler sees the
program. The host writes nothing into a running kernel except
coordinates and extern values, and both are declared inputs with
declared types. See [Runtime Model](../design/runtime_model.md) for the
axioms behind this rule.

## 2. Compile and drive

The smallest host compiles source text, sets a coordinate, and reads
named outputs. A *kernel* is one running instance of a compiled program:
it holds the current input values and the outputs computed from them.
`compile_polydat_kernel` takes the program's source text, compiles it,
and returns a `Box<dyn Kernel>` ready to run on the calling thread, or
an error if the source does not compile. The kernel runs on the default
engine: native code when the build has the `jit` feature, and the
closure tier otherwise.

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

`set_inputs` takes one value per coordinate (the program's `input`
declarations, in the order they are declared) and writes them into the
kernel. `pull` takes the name of an output, evaluates only the steps
that output depends on (its *cone*), and returns the output's value.
The kernel keeps each computed value and returns it again on later
pulls until an input the output depends on changes, so pulling the same
wire twice evaluates it once. The value comes back as an owned `Value`
with typed accessors (`as_u64`, `as_f64`, `as_str`, `as_bool`) that
panic on a type mismatch. A mismatch is a host bug: the program's types
are known at compile time, and the host is expected to read the types
it declared. `engine()` returns the name of the engine the kernel runs
on and, in parentheses, its provenance mode, which decides how much
work the kernel skips when inputs repeat. Here that is native code with
`Raw` provenance, which the `Auto` selector picks for a program this
small; §11 names the engines and the provenance modes.

`pull` looks the name up on every call. A host that reads the same
outputs every cycle can resolve each name once with `output_index`,
which takes the name and returns the output's index, and then pull by
index with `pull_at`, so no lookup runs per cycle. The index is the
output's position in `output_names`, which lists the anonymous
intermediate wires too, so the indices of the named outputs are not
consecutive. The binary's fibers (the worker loops that drive a run,
one kernel each) pull this way.

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

`output_index` returns `None` for a name the program does not bind.
`pull_at` with an index past the last output panics; like a typed
accessor on the wrong variant, that is a host bug.

The other compile entry points differ from `compile_polydat_kernel` in
what they take and what they return:

| Function | Adds |
| --- | --- |
| `compile_polydat_kernel_with_options(src, &options, log)` | `CompileOptions`: the source directory, so relative `import` paths resolve; library directories (§8); required outputs; strict typing, where implicit adapters are errors; a context label for errors; a cursor limit; and the compile ledger to charge (`ledger`), by default a new one for the program tree, or the parent's when a host compiles a subscope. And the compile event log (§13). |
| `parse_polydat(src)` | stops after the parse and returns the program tree, the form a transform rewrites (§12); `parse_polydat_with_tile_defaults(src, &defaults)` sets the delimiters a tile body is read with |
| `compile_polydat_with(src, engine)` | builds on the engine the caller names with an `Engine` value (§11); `compile_polydat_with_engine(src, engine, &options, log)` takes the options and the log as well |
| `compile_polydat(src)` | the same as `compile_polydat_kernel(src)`: builds on the default engine and returns a `Box<dyn Kernel>`. A host that wants the interpreter, the reference implementation the other three engines are checked against (§11), names it with `compile_polydat_with(src, Engine::Interpreter(..))` or the `interpreter` entry points below. |
| `compile_polydat_interpreter(src)`, `compile_polydat_interpreter_with_options(src, &options, log)`, `compile_polydat_interpreter_with_log(src, log)` | the interpreter kernel, `PolydatKernel`, as a concrete type. Every one says `interpreter`, because the concrete type is for observing the interpreter's own internals in testing and diagnostics (§13), not for running a program. |
| `compile_ast_interpreter_with_options(&ast, src, &options, log)` and `compile_ast_with_engine(&ast, src, &options, log, engine)` | the same compiles from a parsed, possibly transformed, program (§12) |
| `compile_polydat_to_assembler(src)` | stops before engine selection and returns the assembler (§7, §11), the graph a host may extend by hand and then build on any of the four engines |
| `compile_polydat_to_assembler_with(src, &options)` | the assembler built with the same `CompileOptions` |

All of these entry points share one compile path on all four engines:
the same prologue applies the options, the same assembly builds the
graph, and the same lowering attaches the traversals and resolves the
cursor extents. The options mean the same thing whichever entry point
receives them, and `strict` refuses the same programs on all four
engines.

## 3. Externs

An `extern` declares a typed input with a default value. The host may
overwrite the value in each kernel at run time, or a program transform
may fix it in the source before compilation. Both set the same input,
and the kernel cannot tell which of the two the host used.

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

All four engines treat an extern the same way. `set_input` takes the
extern's name and a `Value`, writes the value, and returns an error if
the write is refused. Setting an extern marks everything downstream of
it for recomputation at the next pull. `set_input` checks the value
against the extern's declared port type and refuses a mismatch with
`WriteError::TypeMismatch`, naming the extern; nothing converts a value
at the write (see "When an input's type varies" below).

Internally, a compiled kernel stores each extern in slots after the
coordinates and writes the default into them when the kernel is built.
`set_input` writes a number or boolean straight into its slot. A
string, JSON, or extension value is stored as a value the kernel owns,
and the slots point at that value until the next write. The
interpreter keeps every extern as a value in its state.

An extern declared without a default, such as `extern doc: json`, is
`None` until the host sets it, and every consumer reads `None` through
it on the interpreter, the closure tier, and native (P3). Pure native
code has no way to represent `None`, so it refuses at the pull until
the extern is set. The compile log (§13) names every extern without a
default, so a host knows what it must set.

A constant that no input feeds is computed once when a kernel is built,
on all four engines, so a failure in it surfaces at build. A value that
depends on an extern is computed at the first pull that needs it and
kept until the extern changes.

A `const` binding is evaluated when the kernel is initialized, and
every kernel you receive from a build, from `create_kernel`, from a
scope binder, or from `activation_on` is initialized already, from the
input values it had at that moment. Its value then stays fixed: writing
an extern the const reads does not change the const. When you set such
an extern after creating the kernel and want the const recomputed, call
`kernel.init()`. It evaluates every const again, in dependency order,
from the inputs as they are now, keeps every input's value, and returns
`Ok(())`, or `KernelError::ConstInit` naming the const whose expression
failed. `create_kernel` panics with that message instead, since it has
no error return. Writing a const's own slot (`__const_<name>`) with
`set_input` is refused with `WriteError::ConstSlot`; `init` is the only
way a const changes.

A `volatile` binding, and anything that reads a nondeterministic node
such as `current_epoch_millis`, is evaluated again at every `pull` or
`eval` that needs it, while the steps it reads stay cached. To take one
reading for a kernel's life, capture it in a `const`. A session clock is
the usual case, and polydat ships no session-timestamp node, so the host
defines the origin: its root scope declares
`const session_start := current_epoch_millis()`, each child declares
`extern session_start: u64` and receives the root's captured value when
it is bound, and elapsed time is
`current_epoch_millis() - session_start`.

Prefer the transform when the value is fixed for the run: the compiler
then sees a constant, folds it, and native code embeds it as an
immediate operand. Use `set_input` when the value varies from kernel to
kernel, such as a per-thread shard label.

When the value varies per cycle, look the extern up once with
`input_index`, which takes the name and returns the input's index, and
write it each cycle with `set_input_at`, which takes that index and the
value. The index is the input's position in `input_names`, which lists
the coordinates first, so an extern's index follows them. The write is
the same typed write as `set_input`, with the same refusal of a
mismatch, and no lookup runs per cycle.

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

A declared input's type is fixed for the kernel's lifetime. A write of
a value of another type is a host bug, and the write is refused with
the input named. Some hosts write
values whose type legitimately varies: a parameter that arrives as text
in one run and as a number in the next, or a parent scope's value copied
into a child that typed it differently. Polydat converts such a value in
the program, never at the write, and only where the host asked for it:

- **Convert before writing** when you know the target type.
  `polydat::convert::to_port(value, port_type)` takes a value and the
  port type to convert it to, and returns the converted value or a
  `ConvertError`. It applies the same conversions the compiler's
  adapters use, so a failure is an error you handle at the write rather
  than a node failure at the pull. This is the supported way for a host
  to convert values itself.
- **Convert one input in the program** with the transform
  `transform::convert_input(&mut file, "name")`, which takes the parsed
  program and the input's name and rewrites the program. The input then
  accepts any value, and a converter node, which runs as a closure step,
  converts it for the nodes that read it. The compile log reports the
  converter.
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
that drives scope trees with parameters of varying type
compiles with `Warn`, converts what it can through `convert::to_port`,
and reads the compile log to see every input it relies on converting. A
host that compiles many scopes may compile at `Info` instead and report
each open input once from `input_type_origin`.

A `cursor` describes an ordinal domain, such as the range
`[0, 1_000_000)`, and a partition is a fixed sub-range of it assigned to
one scope or fiber. A cursor declared `over` a literal partition spec is
resolved when the kernel is built. The assembler and every kernel list
each cursor with its partitions through `cursor_schemas`. When the
`over` clause names exactly one partition, that partition is written
into the cursor at build, and the program runs on all four engines with
no host call. When the clause names several, the host picks one with
`set_cursor(name, &partition)`, which takes the cursor's name and the
partition and narrows the cursor to it; the call is the same on all
four engines. [Cursor Partitions](../design/cursor_partitions.md) §7.2
has the rules.

## 4. Share a program across threads

A compiled program does not change and is shared through an `Arc`.
`into_program` consumes a kernel and returns its program as an
`Arc<dyn KernelProgram>`. `create_kernel`, called on that program,
returns a new kernel with its own state that shares the program's
compiled steps or native code; each thread creates one. There are no
locks on the evaluation path, and because values depend only on the
coordinate, how the cycles are split across threads does not change
the results.

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

The serial sum runs the same cycles on one kernel. This is all a host
does for concurrency: decide which thread gets which coordinates,
create one kernel per thread, and never share a kernel between threads.
`examples/multi_thread.rs` shows the same pattern at a million cycles
per thread, with timing. The interpreter's program has the same shape
under its own names, `PolydatProgram` and one `PolydatState` per
thread through `create_state`.

### Scope trees on all four engines

A host that runs workloads as a tree of scopes (parameters, a phase,
fibers, per-operation children, one child per iteration tuple) binds
each child under its parent. Binding connects the child to the parent's
shared cells (values the parent publishes for its children to read),
copies in the values the child imports from the parent, and then
initializes the child, evaluating each of its `const` bindings once from
those values. Binding works on the interpreter, the closure tier, and
native (P1, P2, and P3), and one tree may mix those engines, because
every call below works through the `Kernel` trait. `kernel::bind_under`
returns `KernelError::ConstInit`, naming the const, when one fails, and
a fork copies an initialized kernel, consts included, without evaluating
them again.

| To | Call |
|---|---|
| build a child from source under a parent | `SubcontextBuilder::under(parent)`, then `.body(…)`, `.add_result_bindings(…)`, `.finalize()` to a `ScopeModule` |
| instantiate it, on an engine you name | `module.instantiate_under(parent, engine, &iteration_bindings)` |
| bind a precompiled program under a parent | `kernel::bind_under(parent, program, &iteration_bindings)` |
| carry a parent's extern values into a child | `kernel::propagate_inputs(parent, child)` |
| start a fiber from a scope's kernel | `kernel.fork()` |
| read a name the way a scope resolves it | `KernelLookup::new(kernel).lookup(name)` |

A module is compiled once per engine, and each instance adds only a new
kernel state, so instantiating one per iteration tuple compiles nothing
after the first. Per cycle, a host drives a scope kernel by index
through `pull_at`, `input_value_at`, `set_input_at`, `reset_inputs`,
`publish_broadcasts`, and `commit_write_throughs`, none of which looks
a name up. A host that resolves its indices once records the
`program_id()` it resolved them against and may reuse them on any
kernel with that id: the id is equal for every kernel of one program
and for its forks.

The rule above, one kernel per thread, applies to kernels you evaluate.
A scope *parent* is different: every `Kernel` is `Sync`, so a parent may
be held in an `Arc`, and children may be bound and forks taken under it
from many threads at once. Each thread then evaluates only its own
child or fork. [Scope trees on all four engines](../design/native_scope_trees.md)
is the design, and `tests/scope_trees.rs` walks a whole tree on every
pair of the three engines P1, P2, and P3.

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
constant arguments. Both are checked when the node is built, so they
are enforced on all four engines; a check written into the body instead
fires only on the engines that run the body.

Every node the attribute accepts runs on the interpreter, the closure
tier, and native (P3), which runs a node without a native lowering as
its closure. The interpreter calls the body, and the closure tier calls
the generated closure, which passes scalars in slots and passes
strings, JSON, and host values as the reference pairs of
[Compiled By-Reference Slots](../design/compiled_handles.md) (§5
there describes the closure kits). Pure native code runs a node only
when it has a native lowering; a host never needs one for correctness,
only for speed, and [Engines](../design/engines.md) §8 records
what each engine accepts.

## 6. Host-defined value types

Nodes are not the only thing a host can add. A host type becomes a wire
value by implementing `ReflectedValue`; the wire's port type is `Ext`,
nodes take and return it through `Ext<T>`, and the kernel's generic
code uses it only through the trait: string interpolation calls
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
- **Extension nodes run on all four engines through their closure.** On
  the compiled engines an extension value is passed as a reference pair
  pointing at the value its producing step stores, the same way JSON
  is. The closure tier runs a node with an `Ext` signature as a closure
  step, and native code calls the same closure directly from inside the
  segment, with the hashing and scaling around it, passing the value
  between the nodes as the pair described in
  [Compiled By-Reference Slots](../design/compiled_handles.md).
- **The value is cloned on every read.** `Ext<T>::extract` clones the
  boxed value, so a large host type should hold its payload in an `Arc`.
  The crate's own `Partition` and `Streamer` values do exactly that.

### Naming port types without matching them

`PortType` is exhaustive on purpose. When the language adds a type, a
`match` over every variant stops compiling until its author decides
what the new type means. Polydat wants that for its own code and for
host code that behaves differently per type. A host that only labels,
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

`PolydatAssembler::new` takes the names of the coordinates. `add_node`
takes the name of the node's output wire, the node instance, and the
wires it reads, which are either inputs (`WireRef::input`) or other
nodes' outputs (`WireRef::node`). `add_output` exposes a wire as a named
output of the program. The DSL compiler produces exactly this
assembler, so `compile_polydat_to_assembler` is the point where a host
can inspect or adjust the graph between parsing and engine selection.
Three methods build a kernel from it: `compile_kernel()` builds on the
default engine, `compile_with(engine)` builds on the engine the caller
names, and `compile()` builds the interpreter kernel, with
`set_jit_mode` choosing how much of that graph the interpreter runs as
native cones.

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

A `for` statement declares a traversal: a comprehension over one or
more variables, with a body that compiles once. Each combination of
the variables' values is a *tuple*, and an *activation* is a kernel
over the body for one tuple. The host opens the traversal, gets one
activation per tuple, and drives each activation's cycles itself. This
is the division of labor from §1 in practice: Polydat enumerates the
tuples in a deterministic order and types the body, and the host
decides when and where each activation runs.

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
    let mut act = stream.activation(index)?;
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

`traverse(i)` takes the position of a `for` statement in the program,
counting from 0, and returns a stream over that traversal's tuples, or
an error. It is a `Kernel` trait method, so a kernel on any of the four
engines opens its traversals the same way, and an activation opens the
`for` statements of its own body the same way. On the stream,
`len()` is the number of tuples, and `activation(index)` builds the
activation for the tuple at `index` on the stream's engine, which is
the engine of the kernel that opened the traversal;
`activation_on(index, engine)` builds it on the engine the caller
names, and `advance()` builds the next one in order on the stream's
engine.
Every activation is a kernel over the body's program, driven through
the `Kernel` trait, and computes what the interpreter's activation
computes. Fibers divide a traversal among themselves by activating
disjoint index ranges.

Each activation exposes its tuple's values (`act.coord("shard")`), its
cursor slice when the body declares a cursor, its cycle count, and
`act.cycle(n)`, which returns the activation's kernel positioned at
cycle `n`. The body program is compiled once per traversal position and
engine, not once per activation; `examples/for_traversal.rs` measures
that. [The `for` Construct](../design/for_traversal.md) has the full
contract.

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

`tile_from_json_value` takes the tile's name, which becomes the name of
its output wire, the template as a JSON value, the tile options, and a
source position that error messages report; it returns the tile
statement or an error. `add_tiles` appends tile statements to a parsed
program, and the program is then compiled like any other.
`tile_from_text` and `tile_from_json_text` are the other two entry
points. The [Polytile tutorial](../tutorials/polytile_tutorial.md) covers
the template language; [Polytile](../design/polytile.md) §3 specifies the
structural JSON form the value above uses.

## 11. Compiled kernels

A host normally lets `Engine::default()` choose: P3 with the `jit`
feature, the closure tier without. The engines are named by one type,
`Engine`, with four variants:

- `Interpreter` is P1, the reference implementation the other engines
  are checked against. It takes a `JitMode` that says how much of the
  graph it runs as native cones.
- `Closures` is P2, the closure tier: every node runs a generated
  closure over one buffer of slots.
- `Native` is P3: native code for every node that has a native
  lowering, and the node's closure elsewhere.
- `PureNative` is native code with no closure fallback. It refuses a
  program containing a node that has neither a native lowering nor a
  kit, which makes it the way to ask whether a program is fully native.
  It accepts only `Raw` or `PushPull` provenance (`Auto` resolves to one
  of the two), and a build without the `jit` feature refuses it.

The three compiled variants take a `Provenance`, which says how much
work the kernel skips when inputs repeat; `Auto` lets the selector
choose. One constructor, `compile_polydat_with(src, engine)` or
`compile_with(engine)` on the assembler, builds a `Box<dyn Kernel>` on
any of them or returns one error type, `KernelError`, whose `Refused`
variant names the engine and the node or construct it cannot run. The
`Kernel` trait offers the same calls on all four engines: `set_inputs`,
`set_input`, `set_cursor`, `eval`, `pull`, `traverse`, and the name and
type listings. The program below includes `host_tag` from §5 and the two
extension nodes from §6, which have no native lowering of their own; P3
calls them from native code through their kits. A kit is a function the
`#[polydat_node]` attribute generates for each node, which runs the
node's body directly on the kernel's slot buffer:

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

Both compiled engines accept the program and compute what the
interpreter computes. `engine()` on a kernel reports the provenance mode
the selector chose, and the selector makes the same choice on either
compiled engine: this graph has one input, so it gets the cone guard
alone (`Pull`), while the smaller first example above gets `Raw`. P3
runs this program as two native segments: the literals, folded once at
build, and everything else, which is one connected group. The three
host nodes are part of that group: each has a kit, and native code
calls the kit where it would otherwise emit an instruction, with the
extension value passed between two of them as a reference pair.

`plan()` returns a short description of how the kernel runs the
program, such as the number of native segments; it is the only
planning detail a kernel exposes, and all four engines provide it, so a
host can see whether a program is mostly native before deciding whether
that matters. An engine that cannot run a program at all returns an
error that names the engine and the port; for example, either compiled
engine refuses an extern of a 128-bit integer or register word type. A
`shared` binding runs on all four engines with the interpreter's cell:
`shared_cells` on a kernel lists its cells, and `attach_shared_cell`
binds one kernel's cell into another, so both read and write one
register.

On all four engines, `pull` evaluates the output's cone and nothing
more. A side channel in the cone fires when the output is pulled, and a
failing node fails when it is pulled, as on the interpreter, with the
same message: the original panic, the node's name, the outputs it
feeds, the program's context, and its input values (only the
`panicked at` line names the engine's own code). A constant that no
input feeds is computed once when the kernel is built, on all four
engines, so a failure in it surfaces at build; a step that depends on
an extern is computed at the first pull that needs it and kept until
that extern changes. An extern without a default is `None` until the
host sets it, and on the interpreter, the closure tier, and native
(P3) every consumer reads `None` through it; pure native refuses at the
pull instead (§3). The compile log names each such extern.

On all four engines, `pull` returns an owned value: a string, JSON
document, or rendered tile is copied out of the kernel's own storage,
so a host holding a `dyn Kernel` never holds a reference into a kernel,
and a value read before a write is unchanged after it. The concrete
compiled kernel types also offer raw readers (`get` for a scalar slot,
`get_value` for a typed copy, `eval(&[u64])`); `get` refuses a
by-reference slot, and `get_value` copies it out.

A kernel on any of the four engines shares its program across threads
the way §4 shows: `into_program` returns an `Arc<dyn KernelProgram>`,
and `create_kernel` on it returns a kernel of its own for each thread,
over the shared steps or native code.

The interpreter kernel that `compile()` and the `interpreter` entry
points build, `PolydatKernel`, runs the interpreter over a graph whose native-eligible
regions are fused into native cones, as `set_jit_mode` on the assembler
(`Auto`, `Off`, `Force`) and the `jit` Cargo feature allow. It is the
reference the other three engines are checked against, and its program
holds the metadata §13 inspects. The engine a host gets when it names
none, `Engine::default()`, is P3 with the `jit` feature and the closure
tier without. `compile_polydat_kernel` and its `_with_options` and
`_with_tiles` forms, `compile_kernel()` on the assembler, and the
`polydat` binary all build on it, and the binary opens every level of a
traversal nest on it. [Compilation levels](compilation.md)
describes each engine, [Engines](../design/engines.md) the selection
rules, and [Engines](../design/engines.md) §8 what each engine
accepts.

## 12. Program transforms

A host feature is a rewrite of the program. The emit facility the binary
exposes as `--emit` is only a binding to a side-channel node appended to
the program, plus a thread-local buffer the host drains. A side-channel
node is one that writes somewhere the kernel does not see.

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
and let the compiler prune. A transform operates on the **parsed
program**, never on the text it was parsed from. Once the text is
parsed, the tree is the program and the text does not define it:
parsing the text again does not rewrite the program but produces a
second, unrelated one. That is why a tile keeps no copy of its body
(§10).

`dsl::transform` holds the rewrites the crate ships and the walkers a
host uses to write its own:

| | |
|---|---|
| `assign_values(&mut file, &pairs)` | pin an `extern` default, or turn an `input` into one |
| `each_statement(&mut statements, f)` | every statement, descending into module and `for` bodies |
| `each_tile(&mut statements, f)` | every tile the subtree declares |
| `tile_named(&mut statements, "doc")` | one tile, for a transform qualified to it |
| `each_piece(&mut pieces, f)` | every piece of a template, descending into projection bodies and branch arms |

Each takes the slice to walk rather than the whole file, so the caller
chooses the subtree: the program's statements, one module's body, or
one tile's pieces. A tile's text is rendered from its pieces as the
transform left them, so what the program prints after a rewrite matches
what it renders.

A host should never modify a compiled program: the compiler's
provenance, purity, and fusion decisions were made for the program it
compiled.

Side effects belong on the host side of the emit buffer. A node marked
`SideChannel` may write to a thread-local the host drains, as `emit_row`
does; it must not read anything that varies between runs, because the
kernel's determinism contract does not know it exists.

## 13. Diagnostics

The compile log records what the compiler did to a program: what it
inlined, folded, fused, and why. The binary's `explain` command narrates
it; a host gets the same events through the log parameter of
`compile_polydat_kernel_with_options`, and `compile_polydat_interpreter_with_log`
fills the same log while building the interpreter's program. That
program's metadata (node count, node names, determinism) covers what a
host usually needs to know at run time.

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
decision, each carrying its details, and `level` classifies an event
as info, advisory, or warning. The events here describe the compile in
order: the parse; each binding and the node it resolved to; the two
hole typings, each naming the wire type it saw, what the hole's
position expects, and the encoder it chose; the tile's compiled shape;
the outputs the program exposes; the compiled form of each node, native
for every one here; the one constant the compiler folded; and a summary
of the resolved graph. Advisories, such as an implicit type widening,
and warnings, such as an unknown pragma, arrive in the same list, so a
host that wants a strict build can fail on any event whose level is a
warning.

Cone fusion is not an event. The node list shows what the interpreter's
program actually runs: one constant, the passthrough that exposes the
coordinate as an output, and one native cone that fused the hash, the
conversion, the division, and the tile renderer; the cone calls the
renderer's closure directly, over the state's own scratch storage
(compiled_handles.md §6).

The log does not depend on the engine. The default engine records the
same events, from the parse to the summary, including the constants its
own build folded, and every extern without a default is named on all
four engines.

`is_deterministic` returns false when any node is nondeterministic,
such as a wall clock or a true random source; a host should check it
before relying on replay. Side-channel nodes such as `emit_row` do not
make it false: they write outward but read nothing that varies.

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
