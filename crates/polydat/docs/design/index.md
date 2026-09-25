# Design specifications

* [Compiled By-Reference Slots](compiled_handles.md) - How Str, Bytes, Json, Ext, and Handle values occupy compiled slots as reference pairs into step-owned scratch, and what crosses the interpreter boundary.
* [The Composition Substrate](composition_substrate.md) - The three pillars of free graph composition (context synthesis, type safety, state layering), their S, T, and L axioms, and the boundary handlers between them.
* [Comprehension Forms](comprehension_forms.md) - The comprehension algebra: constructors, closure and validity axioms, boundedness, algebraic equivalences, the operator IR, and dispense semantics.
* [Cross-Fiber Cell Invalidation](cross_fiber_invalidation.md) - The SharedCell publish and consume protocol: revisions, intent bits, memory ordering, and happens-before across fibers on all four engines.
* [Cursor Partitions](cursor_partitions.md) - Partition values, the partition-spec language, resolution and rounding, ordering, metadata wires, and cursor narrowing with over.
* [Execution Engines](engines.md) - The P1, P2, P3, and pure-native engines, the rules every engine follows, provenance modes, the selector, slot representation, and engine equivalence.
* [Evaluation Model](evaluation_model.md) - The program/state split, provenance-based invalidation, the two evaluation lifecycles, the const binding contract, input spaces, and external-write inputs.
* [The Expression Engine](expression_engine.md) - Polydat as a host-embeddable evaluator: the expression surfaces, the E-axioms of the embedding contract, and the host and polydat obligations at the boundary.
* [The for Construct](for_traversal.md) - Comprehension producers and traversal scopes in the grammar: typing, compilation, one program per lexical position, affine activation, and cursor narrowing.
* [The Graph Compiler](graph_compiler.md) - The compiler pipeline and its ordering: wire resolution, adapter insertion, node fusion, hoisting, context fusion, and the H, CF, and NF axioms.
* [Input Variance](input_variance.md) - Inputs whose written type varies are served by converter nodes compiled into the graph, not by conversions at the write.
* [Comprehension IR Architecture](ir_architecture.md) - The comprehension IR as a stack machine over stream operands: interpretation, materialization barriers, stack effects, and adding an opcode.
* [JIT Boundary](jit_boundary.md) - The native call boundary: function signatures, predicate failures and their recovery, the invoke_with_catch contract, and invalidation across the boundary.
* [Library Catalog](library_catalog.md) - What a node is, the authoring contract, cost classes, and why the node registry is open.
* [Module System](module_system.md) - File-backed module discovery, typed module interfaces, call resolution, and graph inlining.
* [Scope Trees on All Four Engines](native_scope_trees.md) - Building and driving a tree of child scopes on all four engines through the Kernel trait, including the binder and writes of varying type.
* [None Semantics](none_semantics.md) - How Value::None flows through the kernel and the language, including the conditional-shadow const rule.
* [The Polydat Grammar](polydat_grammar.md) - The normative surface language: lexical rules, productions, operators, type inference, casts, modules, and the G-axioms.
* [Programmatic Construction of the Grammar](polydat_grammar_programmatic.md) - Building the grammar specification's example kernels through the public AST types, each verified to project to the same canonical syntax.
* [Polytile](polytile.md) - Compiled variate templates: static skeletons with typed holes, encodings, projections over comprehensions, and rendering on all four engines.
* [The Runtime Model](runtime_model.md) - The R-axioms of data flow, currency, invalidation, and output ownership, and the D-axioms of determinism, on all four engines.
* [Scope Model](scope_model.md) - Scope identity, parent-gated construction, visibility, lifecycle ownership, shared mutation, and scope-coordinate paths.
* [SIMD ISA Selection and Scalar-Flow Promotion](simd_isa_autopromotion.md) - Effective native-ISA discovery, typed SIMD node variants, promotion qualification, ordinal packet execution, ordered scalar drain, and recovery.
* [Parent-Gated Subcontext Construction](subcontext_construction.md) - The typed construction boundary for a child scope, enforcing lifecycle isolation and cross-tier write-through.
* [Type System](type_system.md) - The static PortType contract on wires, its runtime Value representation, and the adapter catalog between types.
* [Type-System Alignment](type_system_alignment.md) - How static wire types, runtime values, compiled slots, Cranelift 0.116, and serde JSON align.
* [Wire Materialization](wire_materialization.md) - Cross-scope wire flow: the uniform read invariant, the write contract, and the materialization gradient from shared cells to resolvers.
