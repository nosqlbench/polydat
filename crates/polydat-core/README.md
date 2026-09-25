# polydat-core

The Polydat runtime: the value model, the graph compiler, the four
execution engines, the kernels, the comprehension runtime, the node
macro's support surface, the nodes the compiler synthesizes itself
(adapters, passthroughs, constants, tile rendering) and the runtime's
own node modules (formatting, JSON, data files, diagnostics, context,
logging, vector datasets), and the numeric bodies the native lowerings
share with the node library.

Most programs should depend on [`polydat`](https://crates.io/crates/polydat),
the facade that re-exports this crate together with the node library
and the language at the paths they always had. Depend on `polydat-core`
directly when you are assembling your own node set and do not want the
standard library linked, or when you are building a runtime tool that
needs the compiler and engines alone.

## Example

The runtime compiles any program whose functions are linked. The
standard functions such as `hash` live in
[`polydat-nodes`](https://crates.io/crates/polydat-nodes) and register
at link time, so a program that calls them needs that crate too:

```toml
[dependencies]
polydat-core = "0.5"
polydat-nodes = "0.5"
```

```rust
use polydat_core::dsl::compile_polydat_with;
use polydat_core::{Engine, Provenance};

fn main() -> Result<(), polydat_core::KernelError> {
    // Linking the node library is what makes `hash` callable.
    let _ = &polydat_nodes::hash::Hash::new;

    let mut kernel = compile_polydat_with(
        r#"
            input cycle: u64
            id := mod(hash(cycle), 1000)
        "#,
        Engine::Closures(Provenance::PushPull),
    )?;

    kernel.set_inputs(&[7]);
    println!("{}", kernel.pull("id").as_u64());
    Ok(())
}
```

`Engine::default()` is the native engine with the `jit` feature and the
closure engine without it. The interpreter, the closure engine, and
the native engine accept every program and yield the same values; the
pure native engine accepts only programs whose every node has a native
lowering, and yields the same values for those.

## Cargo features

| Feature | Default | Purpose |
| --- | ---: | --- |
| `jit` | yes | The native engine, on Cranelift. |
| `vectordata` | no | Vector-dataset access nodes for ML/AI-oriented workloads. |

## Documentation

The rustdoc on [docs.rs](https://docs.rs/polydat-core) covers the API.
The design documents that this crate implements, with their axioms, are
in the repository under
[`crates/polydat/docs/design`](https://github.com/nosqlbench/polydat/tree/main/crates/polydat/docs/design);
the [runtime model](https://github.com/nosqlbench/polydat/blob/main/crates/polydat/docs/design/runtime_model.md),
the [graph compiler](https://github.com/nosqlbench/polydat/blob/main/crates/polydat/docs/design/graph_compiler.md),
and the [engines](https://github.com/nosqlbench/polydat/blob/main/crates/polydat/docs/design/engines.md)
are the ones to read first.

## License

Apache-2.0.
