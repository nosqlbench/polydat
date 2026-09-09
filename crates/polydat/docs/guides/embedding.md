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
| Selecting engines (interpreter, closures, native code) | Polydat | §11 |
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
named outputs. `compile_polydat` returns a `PolydatKernel`: a program
plus one state, ready to run on the calling thread.

```rust
use polydat::dsl::compile::compile_polydat;

let mut kernel = compile_polydat(r#"
    input cycle: u64
    user_id := mod(hash(cycle), 1000000)
    score   := unit_interval(hash(user_id))
    label   := "user-{user_id}"
"#)?;
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
inputs: ["cycle"]
outputs: ["cycle", "user_id", "score", "label"]
cycle 0: user_id=607535 score=0.601 label=user-607535
cycle 1: user_id=822465 score=0.230 label=user-822465
cycle 2: user_id=348110 score=0.219 label=user-348110
```

`set_inputs` writes the coordinate tuple and advances the cycle; `pull`
evaluates on demand and caches within the cycle, so pulling the same
wire twice costs one evaluation. Values come back as `&Value` with typed
accessors (`as_u64`, `as_f64`, `as_str`, `as_bool`) that panic on a
type mismatch, which is a host bug: the program's types are known at
compile time and the host is expected to read what it declared.

The other compile entry points differ only in what they take:

| Function | Adds |
| --- | --- |
| `compile_polydat_with_path(src, path)` | the source directory, so relative `import` paths resolve |
| `compile_polydat_with_libs(src, dir, libs, outputs, strict, ctx)` | library directories, required outputs, strict typing, and a context label for errors |
| `compile_polydat_strict(src, dir, strict)` | strict mode: implicit adapters are errors |
| `compile_polydat_with_log(src, &mut log)` | the compile event log (§13) |
| `compile_polydat_with_tiles(src, tiles)` | tile statements built from host data (§10) |
| `compile_polydat_to_assembler(src)` | stops before engine selection and returns the assembler (§7, §11) |

## 3. Externs

An `extern` is a typed input slot with a default. The host may overwrite
it per state, and a program transform may fix it before compilation.
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
let program = compile_polydat(src)?.into_program();
let mut state = program.create_state();
state.set_inputs(&[7]);
println!("defaults: {}", state.pull(&program, "key").as_str());

let region = program.find_input("region").unwrap();
let scale = program.find_input("scale").unwrap();
state.set_input(region, Value::Str("eu-west".into()));
state.set_input(scale, Value::U64(1000));
state.set_inputs(&[7]);
println!("overridden: {}", state.pull(&program, "key").as_str());

// The binary's `name=value` arguments are this transform.
let transformed = src.replace(r#"extern region: str = "us-east""#, r#"extern region: str = "ap-south""#);
let mut fixed = compile_polydat(&transformed)?;
fixed.set_inputs(&[7]);
println!("transformed: {}", fixed.pull("key").as_str());
```

```text
defaults: us-east/7
overridden: eu-west/487
transformed: ap-south/7
```

Prefer the transform when the value is fixed for the run: the compiler
then sees a constant, folds it, and the fused native cones in §11 carry
it as an immediate. Use `set_input` when the value genuinely varies per
state, such as a per-thread shard label. `set_input` takes a `Value` and
writes it without re-checking the slot's declared type, so passing the
wrong variant is a host bug that surfaces when a consumer reads it.

## 4. Share a program across threads

A `PolydatProgram` is immutable once compiled and is shared through an
`Arc`. Each thread creates its own `PolydatState` over it. There are no
locks on the evaluation path, and because values depend only on the
coordinate, the partition of cycles across threads is invisible in the
results.

```rust
let program = compile_polydat("input cycle: u64\nv := mod(hash(cycle), 1000)\n")?.into_program();
let sums: Vec<u64> = std::thread::scope(|s| {
    let handles: Vec<_> = (0..threads).map(|t| {
        let program = program.clone();
        s.spawn(move || {
            let mut state = program.create_state();
            let mut sum = 0u64;
            for c in t * per_thread..(t + 1) * per_thread {
                state.set_inputs(&[c]);
                sum += state.pull(&program, "v").as_u64();
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

The serial sum is the same cycles on one state. This is the whole
concurrency story for a host: decide which thread gets which
coordinates, create one state per thread, and never share a state.
`examples/multi_thread.rs` shows the same pattern at a million cycles
per thread, with timing.

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

let mut kernel = compile_polydat(r#"
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
constraints such as `#[constraint(NonZeroU64)]` on a parameter.

A node with a scalar signature runs on every engine: the interpreter
calls the body, the closure tier calls the generated closure, and native
cones call the closure through a fixed entry. A node whose signature
uses strings or `Value` runs on the interpreter and the closure tier
and, since [Compiled Non-Scalar Slots](../design/compiled_handles.md)
landed, inside native kernels through the handle boundary. The macro
kit that makes a node native-capable is described there in §7; a host
never needs it for correctness, only for speed.

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

let mut kernel = compile_polydat(r#"
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
    let Value::Ext(boxed) = kernel.pull("cell").clone() else { panic!("cell is an Ext wire") };
    let cell = boxed.as_any().downcast_ref::<GeoCell>().expect("a GeoCell");
    println!("cycle {cycle}: level {} at ({:.1}, {:.1}); json {}", cell.level, cell.lat_deg, cell.lon_deg, boxed.to_json_value());
}
println!("cell wire type: {:?}", kernel.program().output_port_type("cell"));
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
- **Extension nodes run on the interpreter.** A node with an `Ext`
  signature has neither a closure form nor a native one, so the closure
  and native kernels of §11 refuse a program that contains one, and the
  production kernel runs it on the interpreter. The scalar work around it
  is still fused: in the run above the hashing and scaling fused into two
  native cones while the two host nodes ran interpreted, and a native
  neighbour reads an extension value through the table handle described
  in [Compiled Non-Scalar Slots](../design/compiled_handles.md).
- **The value is cloned on every read.** `Ext<T>::extract` clones the
  boxed value, so a large host type should hold its payload in an `Arc`.
  The crate's own `Partition` and `Streamer` values do exactly that.

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
let mut kernel = asm.compile()?;
kernel.set_inputs(&[42]);
```

```text
user_id at cycle 42: 275413
```

The DSL compiler produces exactly this assembler, so
`compile_polydat_to_assembler` is the point where a host can inspect or
adjust the graph between parsing and engine selection, and where it can
set the JIT mode before calling `compile()`.

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
let mut kernel = compile_polydat_with_libs(
    "input cycle: u64\n(b, l) := bucketed(input: cycle, buckets: 8)\n",
    None,            // source directory for relative imports
    vec![dir],       // library directories, searched in order
    &[],             // required outputs, if the host wants a check
    false,           // strict typing
    "embedding guide",
)?;
```

```text
cycle 0: bucket=7 label=b7
cycle 1: bucket=1 label=b1
cycle 2: bucket=6 label=b6
```

The binary's `--lib` flag is this argument. See
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
let mut kernel = compile_polydat(r#"
    input cycle: u64
    for shard in 0..3, phase in load,verify {
        row := mod(hash(cycle), 100) + shard * 100
        stmt := "{phase} shard {shard} row {row}"
    }
"#)?;
kernel.set_inputs(&[0]);
let mut stream = kernel.traverse(0)?;
while let Some(mut act) = stream.advance()? {
    let index = act.index;
    let k = act.cycle(1);
    println!("  activation {index}: {}", k.pull("stmt").as_str());
}
```

```text
6 activations of `shard in 0..3, phase in load,verify`
  activation 0: load shard 0 row 65
  activation 1: verify shard 0 row 65
  activation 2: load shard 1 row 165
  activation 3: verify shard 1 row 165
  activation 4: load shard 2 row 265
  activation 5: verify shard 2 row 265
```

`traverse(i)` opens the i-th traversal in the program. Each activation
exposes its coordinates (`act.coord("shard")`), its cursor slice when
the body declares a cursor, its cycle count, and `act.cycle(n)`, which
returns the activation's kernel positioned at cycle `n`. The body
program is compiled once per traversal position, not once per
activation; `examples/for_traversal.rs` measures that.
[The `for` Construct](../design/for_traversal.md) has the full contract.

## 10. Tiles from host data

A tile is a template whose holes are wires. In source it is a `tile`
statement; at the host boundary it may arrive as text, as JSON text, or
as an already-parsed JSON value, and each form becomes the same tile
statement before compilation. This is how a host lets its users write
templates in their own configuration format without teaching the host
anything about rendering.

```rust
use polydat::tile::{compile_polydat_with_tiles, tile_from_json_value, Span, TileOptions};

let template = serde_json::json!({
    "id": "${cycle}",
    "label": "row-${cycle}",
    "points": [ "@for s in 0..2", { "n": "${s}", "v": "${cycle + s}" } ]
});
let tile = tile_from_json_value("doc", &template, &TileOptions::default(), Span { line: 0, col: 0 })?;
let mut kernel = compile_polydat_with_tiles("input cycle: u64\n", vec![tile])?;
kernel.set_inputs(&[4]);
println!("doc: {}", kernel.pull("doc").as_str());
```

```text
doc: {"id": 4, "label": "row-4", "points": [{"n": 0, "v": 4},{"n": 1, "v": 5}]}
```

`tile_from_text` and `tile_from_json_text` are the other two entry
points. The [Polytile tutorial](../tutorials/polytile_tutorial.md) covers
the template language; [Polytile](../design/polytile.md) §6 specifies the
structural JSON form the value above uses.

## 11. Compiled kernels

A host normally lets `compile()` choose engines. The production kernel
runs the interpreter over a graph in which every native-eligible region
has been fused into a cone, and everything else runs through closures
or the interpreter. Three direct forms exist for hosts that want a whole
kernel compiled, such as benchmarks and the differential tests: the
closure tier, pure native code, and the hybrid kernel that lowers what
it can and runs the rest as closures. The program below includes
`host_tag` from §5, which has a closure form but no native one, so the
three forms behave differently:

```rust
let src = r#"
    input cycle: u64
    h := hash(cycle)
    name := "user-{h}"
    tag := host_tag("job", mod(h, 10000))
    tile j : json := {"h": ${h}, "name": ${name}, "tag": ${tag}}
"#;
let asm = || compile_polydat_to_assembler(src).unwrap();
let mut p2 = asm().try_compile_raw().unwrap_or_else(|_| panic!("P2"));   // closures
match asm().try_compile_jit() {                                          // pure native code
    Ok(_) => println!("pure P3: compiled"),
    Err(e) => println!("pure P3 refused: {e}"),
}
let mut hybrid = asm().compile_hybrid()?;                                // native where possible
let (native, closures) = hybrid.engine_counts();
println!("hybrid plan: {native} native segment(s), {closures} closure step(s)");
for cycle in [0u64, 1] {
    p2.eval(&[cycle]);
    let p2_h = p2.get("h");                              // scalar read
    let p2_j = p2.get_value("j").to_display_string();    // handle read: copies out
    hybrid.eval(&[cycle]);
    let hy_h = hybrid.get("h");
    let hy_j = hybrid.get_value("j").to_display_string();
    println!("cycle {cycle}: hybrid agrees: {}", p2_h == hy_h && p2_j == hy_j);
}
let mut mixed = asm().compile()?;   // what a host should normally use
```

```text
pure P3 refused: some nodes cannot be JIT-compiled
hybrid plan: 8 native segment(s), 1 closure step(s)
cycle 0: P2 h=16294208416658607535 j={"h": 16294208416658607535, "name": "user-16294208416658607535", "tag": "job-7535"}
cycle 0: hybrid agrees: true
cycle 1: P2 h=10451216379200822465 j={"h": 10451216379200822465, "name": "user-10451216379200822465", "tag": "job-2465"}
cycle 1: hybrid agrees: true
production kernel j: {"h": 10451216379200822465, "name": "user-10451216379200822465", "tag": "job-2465"}
```

Pure native code refuses the program because one node has no native
form. The hybrid kernel accepts it: eight steps run as native segments,
the host node runs as one closure step, and every value matches the
closure tier. `engine_counts` is the only planning detail the hybrid
kernel exposes, and it exists so a host can see whether a program is
mostly native before deciding to care.

Scalars come back by value from `get`. Strings, JSON, and rendered tiles
are handles into a per-thread arena that lives for one cycle of the
kernel that produced them; `get_value` copies the referent out. That
gives the one rule a host must follow when it drives compiled kernels
directly: read a kernel's handle outputs before running another root
kernel on the same thread, because the next kernel's cycle reclaims the
arena. The production kernel and the traversal runtime follow the rule
internally, so it only reaches a host through `try_compile_raw`,
`try_compile_jit`, and `compile_hybrid`.
[Compiled Non-Scalar Slots](../design/compiled_handles.md) §4 states it
as axiom H3.

Engine choice is not a host concern beyond `set_jit_mode` on the
assembler (`Auto`, `Off`, `Force`) and the `jit` Cargo feature.
[Compilation levels](compilation.md) describes each engine and
[Engines](../design/engines.md) the selection rules.

## 12. Program transforms

A host feature is a rewrite of the program. The emit facility the binary
exposes as `--emit` is nothing more than an appended binding to a
side-channel node, plus a thread-local buffer the host drains.

```rust
let src = "input cycle: u64\nid := mod(hash(cycle), 1000)\nname := \"user-{id}\"\n";
let with_emit = format!("{src}__emit := emit_row(\"jsonl\", \"cycle,id,name\", cycle, id, name)\n");
let mut kernel = compile_polydat(&with_emit)?;
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
and let the compiler prune. The `dsl::transform` module holds the
rewrites the crate ships, such as `apply_tile_defaults`, and a host adds its own
by operating on the source text or the parsed `PolydatFile` before
compiling. What a host should not do is reach into a compiled program
and change it: the compiler's provenance, purity, and fusion decisions
were made against the program it saw.

Side effects belong on the host side of the emit buffer. A node marked
`SideChannel` may write to a thread-local the host drains, as `emit_row`
does; it must not read anything that varies between runs, because the
kernel's determinism contract does not know it exists.

## 13. Diagnostics

The compile log records what the compiler did to a program: what it
inlined, folded, fused, and why. The binary's `explain` command narrates
it; a host gets the same events from `compile_polydat_with_log`. The
program itself answers the questions a host usually has at run time.

```rust
let mut log = CompileEventLog::new();
let kernel = compile_polydat_with_log(src, &mut log)?;
println!("compile events: {}", log.events().len());
for event in log.events() {
    println!("  {:?}: {event:?}", event.level());
}
let program = kernel.program();
println!("nodes: {}, deterministic: {}", program.node_count(), program.is_deterministic());
let names: Vec<_> = (0..program.node_count()).map(|i| program.node_meta(i).name.clone()).collect();
```

```text
compile events: 4
  Info: TileHoleTyped { tile: "t", hole: "h", wire_type: "u64", declared: None, expectation: "any JSON value (u64)", encoder: "json number", adapter: None }
  Info: TileHoleTyped { tile: "t", hole: "f | .2", wire_type: "f64", declared: None, expectation: "any JSON value (f64)", encoder: "json number, format .2", adapter: None }
  Info: TileCompiled { tile: "t", encoding: "json", statics: 3, static_bytes: 14, holes: 2, branches: 0, projections: 0, bodies: [] }
  Info: ConstantFolded { node: "const_f64", value: "3.0" }
nodes: 2, deterministic: true
node names: ["const_f64", "jit_cone[hash+tile_encode+to_f64+f64_div+tile_encode+tile_render]"]
```

The program in this section is a hash, a division, and a JSON tile with
two holes. `CompileEvent` is an enum with one variant per kind of
decision, each carrying its particulars, and `level` classifies it as
info, advisory, or warning. The four events here are the two hole
typings, each naming the wire type it saw, what the hole's position
expects, and the encoder it chose; the tile's compiled shape; and the
one constant the compiler folded. Advisories, such as an implicit type
widening, and warnings, such as an unknown pragma, arrive in the same
list, so a host that wants a strict build can fail on any event whose
level is a warning. Cone fusion is not an
event: the node list shows what the host is actually running, one
constant and one native cone that fused the hash, the conversion, the
division, both hole encoders, and the tile renderer. `is_deterministic`
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
  choosing between when it calls `set_jit_mode`.
- [Runtime Model](../design/runtime_model.md) states the ownership and
  determinism axioms that this guide's division of labor implements.
- `examples/multi_thread.rs`, `examples/for_traversal.rs`, and
  `examples/embedding_guide.rs` are the runnable versions of §4, §9,
  and this whole guide.
