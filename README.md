# Polydat

## What is Polydat?

Polydat is a variates construction engine.

More specifically, Polydat is all of:

- function graph grammar
- procedural generation kernel
- function library and loader
- runtime type-safety reifier
- parameter space projection system
- context layering API
- expression language
- workload simulation compiler

## Why? It's too many things!

Polydat is, ultimately, the distilled and sharpened variate generation system
needed for systems like nb-rs. In fact, Polydat wasn't designed as a system
against a set of new requirements. It evolved as a reduction of nosqlbench's
core features during a Rust port. Polydat is what you get when you take the
essence of a complex but capable system like nosqlbench and compress it down
with heat and pressure until it is forced to take a minimal, elemental shape.

## Motivation

A lesson learned while evolving nosqlbench steered Polydat to where it is now:

**Managing many testing and simulation parameters can be overwhelming!**

In nosqlbench, users have to manage a variety of parameters in different places
with different formats. Polydat's design categorically avoids this problem,
since all the values used in a workload simulation are owned by the same
subsystem.

- Workload parameters on the command line → Polydat
- Template parameters in the workload → Polydat
- Procedural generation recipes → Polydat
- Op template fields and bindings → Polydat
- Dimensional configuration parameters and labels → Polydat

What this means in practice is that a single set of rules applies everywhere,
and since everything is owned by the same subsystem, you can always ask it
exactly where a value comes from or how it is computed. All variables used in a
simulation are connected through the same substrate with a supporting grammar
and a consistent set of precedence and referencing rules.

## Workspace

This repository contains two crates:

- [`polydat`](crates/polydat) — the public engine, DSL, standard library, and runtime.
- [`polydat-derive`](crates/polydat-derive) — the procedural macro implementation for
  `#[polydat_node]`, re-exported by `polydat`.

## Documentation

- [Illustrations](crates/polydat/docs/illustrations.md) — runnable examples through the
  Polydat DSL and the programmatic Assembler API.
- [Compilation](crates/polydat/docs/compilation.md) — Phase 1 / 2 / 3 levels, Hybrid
  mode, throughput numbers, and the `jit` / `vectordata` Cargo features.
- [Nodes](crates/polydat/docs/nodes.md) — the 250+ built-in function nodes.
- [License](crates/polydat/docs/license.md) — Apache-2.0; full text at
  [LICENSE](LICENSE).

Polydat was extracted from the
[nb-rs](https://github.com/nosqlbench/nb-rs) workspace.
