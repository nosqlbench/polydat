# Polydat

Polydat is a compiler and runtime for deterministic procedural data in
Rust. You declare typed inputs and a graph of named functions; Polydat
compiles the graph into a reusable kernel whose named outputs are pulled
on demand, as native machine code where the graph permits it.

The same input always yields the same values, on any thread, any host,
and any engine, with no state carried between evaluations. That is the
whole contract, and everything else in the crate serves it: the DSL,
the node library, the type system, the parameter-space comprehensions,
the tile templates, and the host embedding surface.

Polydat powers the variates subsystem for
[nmbrs](https://github.com/nosqlbench/nmbrs), the successor to
NoSQLBench, and is built to embed in any host.

## Quick start

```toml
[dependencies]
polydat = "0.5"
```

Compile a graph from source and pull named results:

```rust
use polydat::dsl::compile_polydat_kernel;

fn main() -> Result<(), polydat::KernelError> {
    let mut kernel = compile_polydat_kernel(r#"
        input cycle: u64

        hashed := hash(cycle)
        user_id := mod(hashed, 1000000)
        bucket := mod(hashed, 64)
    "#)?;

    for cycle in 0..5 {
        kernel.set_inputs(&[cycle]);
        let user_id = kernel.pull("user_id").as_u64();
        let bucket = kernel.pull("bucket").as_u64();
        println!("{cycle:>5}  {user_id:>7}  {bucket:>6}");
    }
    Ok(())
}
```

```text
    0   607535      47
    1   822465       1
    2   348110      14
    3   139053      45
    4   603978      10
```

The same program with a header and ten rows is `examples/basic.rs`; run
it with `cargo run -p polydat --example basic`. The
[examples directory](examples) also covers the programmatic
assembler, modules, context layering, parameter-space projection, cursor
partitions, traversal strategies, type safety, sharing a kernel across
threads, and defining your own nodes.

## Engines

One program compiles to any of three engines and gives the same values
on each. The host picks with `Engine`; the default is the fastest the
build has.

| Engine | Mechanism | Feature |
| --- | --- | --- |
| Interpreter | Boxed nodes over typed value buffers | always |
| Closures | One generated closure per node over a flat slot buffer | always |
| Native | Cranelift machine code where a node has a lowering, its closure elsewhere | `jit` |

## Cargo features

| Feature | Default | Purpose |
| --- | ---: | --- |
| `jit` | yes | The native engine, on Cranelift. |
| `cli` | yes | The `polydat` command-line harness. |
| `vectordata` | no | Vector-dataset access nodes for ML/AI-oriented workloads. |

For an interpreter-and-closures library build without Cranelift or the
binary:

```toml
[dependencies]
polydat = { version = "0.5", default-features = false }
```

## Why one subsystem

Polydat is the reduction of NoSQLBench's core features that a Rust port
forced into a minimal shape. The lesson that shaped it: managing many
testing and simulation parameters across different places and formats
is overwhelming. Workload parameters, template parameters, procedural
recipes, op-template fields, and dimensional labels are all values in
one graph under one grammar, so a single set of precedence and
referencing rules applies everywhere, and any value can be asked where
it comes from and how it is computed.

## Documentation

The rustdoc on [docs.rs](https://docs.rs/polydat) covers the API. The
[documentation index](docs/README.md) lists the narrative documentation
by section.

Tutorials, each backed by a runnable example:

- [Illustrations](docs/tutorials/illustrations.md): the DSL and the
  programmatic assembler, function graphs, libraries, parameter spaces,
  partitions, traversal, and tiles.
- [Polytile tutorial](docs/tutorials/polytile_tutorial.md): templates
  whose holes are wires, rendering JSON, CSV, and text per cycle.
- [A toy test definition](docs/tutorials/toy_test_definition.md): one
  grammar for coordinates, a workload, and the documents it emits.

Guides:

- [Embedding Polydat](docs/guides/embedding.md): what a host owns and
  what Polydat owns, the APIs for compiling, driving, sharing, and
  extending a kernel, and the extension points.
- [Compilation](docs/guides/compilation.md): the three engines, how to
  choose one, and the Cargo features.
- [Engine-ladder performance](docs/guides/performance.md): one typed
  graph measured on every engine.

Reference and design:

- [Nodes](docs/reference/nodes.md): the built-in function nodes by
  family.
- [Design documents](docs/design): the specifications the code
  implements, with their axioms.

## Crates

`polydat` is the facade. It re-exports three crates at the paths they
always had and holds the binary, the tests, the benches, the examples,
and the docs.

- [`polydat-grammar`](https://crates.io/crates/polydat-grammar): the
  language without the runtime.
- [`polydat-core`](https://crates.io/crates/polydat-core): the runtime,
  compiler, and engines.
- [`polydat-nodes`](https://crates.io/crates/polydat-nodes): the node
  library.
- [`polydat-derive`](https://crates.io/crates/polydat-derive): the
  `#[polydat_node]` attribute.

## License

Apache-2.0. See [LICENSE](LICENSE).
