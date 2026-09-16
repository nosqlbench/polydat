# polydat-derive

The `#[polydat_node]` procedural macro for
[Polydat](https://crates.io/crates/polydat). It turns one typed free
function into a Polydat node: the node struct, its constructor, the
`PolydatNode` implementation with the evaluation forms the signature
allows on every engine, and the link-time registration that makes the
function callable from Polydat source.

Most users never depend on this crate directly. The `polydat` facade
re-exports the attribute as `polydat::polydat_node`, and the macro
emits paths under `polydat::`, so a crate that defines nodes depends on
`polydat` and writes:

```rust
/// A rotated-xor checksum of two words.
#[polydat::polydat_node(category = Math)]
fn host_checksum(a: u64, b: u64) -> u64 {
    a.rotate_left(7) ^ b.wrapping_mul(0x9E37_79B9_7F4A_7C15)
}
```

The function's doc comment becomes the node's description and help.
Each argument is classified by its type: per-cycle wires (scalars,
strings, bytes, JSON, typed vectors, SIMD registers, host types),
`Const<T>` workload constants captured at construction, `&[T]` variadic
arguments, and setup state derived once from the constants. A single
return is one output, a tuple is several named outputs, `Result` runs
once at construction, and `Value` is polymorphic.

## Documentation

The rustdoc on [docs.rs](https://docs.rs/polydat-derive) lists every
argument shape, return shape, and attribute parameter the macro
accepts. The
[embedding guide](https://github.com/nosqlbench/polydat/blob/main/crates/polydat/docs/guides/embedding.md)
shows host-defined nodes and value types in a running program.

## License

Apache-2.0.
