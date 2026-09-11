# Polydat documentation

The documents are in four sections. Start with a tutorial, read a guide
when you need to know how a part works, look up the reference when you
need a name, and read the design documents when you need the reasoning
and the invariants.

## Tutorials

Walk-throughs with real output. Every program in them is run by an
example under `examples/` or by the binary, and `tests/guide_output.rs`
checks that every quoted `text` block in the tutorials and the embedding
guide is output of that run. The same test checks that the performance
guide quotes its graph file exactly and describes it correctly, that the
compilation guide names real Cargo features, and that every relative
link in the documentation resolves.

- [Illustrations](tutorials/illustrations.md): the DSL, the assembler
  API, function graphs, procedural generation, libraries, type safety,
  parameter spaces, partitions, traversal, tiles, and an expression
  language, each as a runnable example.
- [Polytile tutorial](tutorials/polytile_tutorial.md): templates whose
  holes are Polydat expressions, from a one-line text tile to nested
  projections, then the corner cases: escapes, encodings, empty
  projections, modules, the compiler's refusals, and one tile on every
  engine.
- [A toy test definition](tutorials/toy_test_definition.md): one grammar
  for coordinates, a workload, and the documents it emits, run by the
  binary.

## Guides

How the parts work and how to use them from a host program.

- [Embedding Polydat](guides/embedding.md): what a host owns and what
  Polydat owns, the APIs for compiling, driving, sharing, and extending
  a kernel, and the extension points, each with a running example.
- [Compilation levels](guides/compilation.md): the interpreter, the
  closure tier, the hybrid kernel, and native code; features and
  trade-offs.
- [Engine-ladder performance](guides/performance.md): one typed graph
  measured on every engine, with the measurement contract.

## Reference

- [Node library](reference/nodes.md): the built-in function families and
  where they live.
- [License](license.md).

## Design

The specifications (SRDs) that the code implements, with their axioms,
tripwires, and landing records: [`design/`](design/). The ones a reader
of the guides most often needs are [Runtime Model](design/runtime_model.md),
[Graph Compiler](design/graph_compiler.md), [Engines](design/engines.md),
[The `for` Construct](design/for_traversal.md), [Polytile](design/polytile.md),
and [Compiled Non-Scalar Slots](design/compiled_handles.md).
[Engine Parity](design/engine_parity.md) is the review of every way the
compilation levels differed in anything other than performance, and the
plan that removed each difference; every step has landed.
