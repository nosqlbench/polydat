## What is Polydat?

Polydat is a variates construction engine.

More specifically, Polydat is all of:

* function graph grammar
* procedural generation kernel
* function library and loader
* runtime type-safety reifier
* parameter space projection system
* context layering API
* expression language
* workload simulation compiler

## Why? It's too many things!

Polydat is, ultimately, the distilled and sharpened variate generation system
needed for systems like nb-rs. In fact, Polydat wasn't designed as a system
against a set of new requirements. It evolved as a reduction of nosqlbench's
core features during a rust port. Polydat is what you get when you take the
essence of a complex but capable system like nosqlbench and compress it down
with heat and pressure until it is forced to take a minimal, elemental shape.

## Motivation

A lesson learned while evolving nosqlbench steered Polydat to where it is now:

__Managing many testing and simulation parameters can be overwhelming!__

In nosqlbench, users have to manage a variety of parameters in different places
with different formats. Polydat's design categorically avoids this problem,
since all the values used in a workload simulation are owned by the same
subsystem.

* Workload parameters on the command line -> Polydat
* Template parameters in the workload -> Polydat
* Procedural generation recipes -> Polydat
* Op template fields and bindings -> Polydat
* Dimensional configuration parameters and labels -> Polydat

What this means in practice is that a single set of rules applies everywhere,
and since everything is owned by the same subsystem, you can always ask it
exactly where a value comes from or how it is computed. All variables used in a
simulation are connected through the same substrate with a supporting grammar
and a consistent set of precedence and referencing rules.

## Documentation

The [documentation index](docs/README.md) lists everything by section.

Tutorials, each backed by a runnable example:

- [Illustrations](docs/tutorials/illustrations.md) — the DSL and the
  programmatic Assembler API, function graphs, libraries, parameter
  spaces, partitions, traversal, and tiles.
- [Polytile tutorial](docs/tutorials/polytile_tutorial.md) — templates whose
  holes are wires: JSON, CSV, and text documents rendered per cycle, with
  projections, branches, and the corner cases.
- [A toy test definition](docs/tutorials/toy_test_definition.md) — one
  grammar for coordinates, a workload, and the documents it emits.

Guides:

- [Embedding Polydat](docs/guides/embedding.md) — host and kernel
  responsibilities, the APIs for compiling, driving, sharing, and
  extending a kernel, and the extension points, each with real output.
- [Compilation](docs/guides/compilation.md) — Phase 1 / 2 / 3 levels, Hybrid
  mode, throughput numbers, and the `jit` / `vectordata` Cargo features.
- [Engine-ladder performance](docs/guides/performance.md) — one typed graph
  measured on every engine.

Reference and design:

- [Nodes](docs/reference/nodes.md) — the 250+ built-in function nodes.
- [Design documents](docs/design) — the specifications the code
  implements, with their axioms and landing records.
- [License](docs/license.md) — Apache-2.0; full text at [LICENSE](LICENSE).

Part of the [nb-rs](https://github.com/nosqlbench/nb-rs) workspace.
