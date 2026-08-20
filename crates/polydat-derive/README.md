# polydat-derive

`polydat-derive` provides the `#[polydat_node]` procedural macro used by
[`polydat`](https://github.com/nosqlbench/polydat). It generates Polydat node
metadata, evaluation adapters, function signatures, and inventory registration
from typed Rust functions.

Most users should depend on `polydat`, which re-exports the macro as
`polydat::polydat_node`.

Licensed under Apache-2.0.
