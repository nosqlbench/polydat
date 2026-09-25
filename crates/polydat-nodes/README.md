# polydat-nodes

The Polydat node library: every function a program can call that the
compiler does not synthesize itself. Hashing, arithmetic, sampling and
distributions, strings and encodings, digests, noise, dates, vectors,
data files, partitions, and more, each a `#[polydat_node]` function
over the runtime's value model.

Linking the crate is what makes its functions callable: every node
registers into the runtime's registry at link time. Most programs get
this crate through [`polydat`](https://crates.io/crates/polydat), which
re-exports it under `polydat::library`. Depend on `polydat-nodes`
directly only when you depend on
[`polydat-core`](https://crates.io/crates/polydat-core) directly.

## Writing your own nodes

A third-party node crate is built the same way as this one. Each node
is a typed free function under the attribute; the macro derives the
node's metadata, its evaluation adapters for the interpreter and for
the closure steps of the compiled engines, and its link-time
registration from the signature. A derived node has no native lowering,
so the native engine runs it as a closure and pure native refuses it:

```rust
use polydat::dsl::compile_polydat_kernel;

#[polydat::polydat_node(category = Math)]
fn host_checksum(a: u64, b: u64) -> u64 {
    a.rotate_left(7) ^ b.wrapping_mul(0x9E37_79B9_7F4A_7C15)
}

#[polydat::polydat_node(category = String)]
fn host_tag(prefix: &str, n: u64) -> String {
    format!("{prefix}-{n:04}")
}

fn main() -> Result<(), polydat::KernelError> {
    let mut kernel = compile_polydat_kernel(r#"
        input cycle: u64
        c := host_checksum(cycle, hash(cycle))
        t := host_tag("job", mod(c, 10000))
    "#)?;
    kernel.set_inputs(&[1]);
    println!("{}", kernel.pull("t").as_str());
    Ok(())
}
```

The attribute emits paths under `polydat::`, so a node crate depends on
the `polydat` facade. This crate is the one exception: it aliases
`polydat_core` as `polydat` because the facade depends on it.

Wire arguments take scalars, strings, bytes, JSON, typed vectors, SIMD
registers, and host types; `Const<T>` arguments capture workload
constants at construction; a tuple return names several outputs. The
[embedding guide](https://github.com/nosqlbench/polydat/blob/main/crates/polydat/docs/guides/embedding.md)
walks through host-defined nodes and value types, and the
[`polydat-derive`](https://docs.rs/polydat-derive) rustdoc lists every
argument and return shape the attribute accepts.

## Documentation

The [node reference](https://github.com/nosqlbench/polydat/blob/main/crates/polydat/docs/reference/nodes.md)
lists every built-in node by family with its signature. The rustdoc on
[docs.rs](https://docs.rs/polydat-nodes) covers the same nodes as Rust
items.

## License

Apache-2.0.
