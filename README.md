# Polydat

Polydat is a variates construction engine: a typed function-graph grammar,
procedural generation kernel, expression language, function library, and
workload simulation compiler.

This repository contains two crates:

- [`polydat`](polydat) — the public engine, DSL, standard library, and runtime.
- [`polydat-derive`](polydat-derive) — the procedural macro implementation for
  `#[polydat_node]`, re-exported by `polydat`.

See the [Polydat crate README](polydat/README.md) for an overview and links to
the design and usage documentation.

## License

Apache-2.0. See [LICENSE](LICENSE).
