# Comprehension Integration Surfaces

This document specifies the boundaries between Polydat's
canonical comprehension algebra and its parsers, kernel scope,
execution surfaces, and external hosts. It contains no migration
or cutover procedure.

Companion specifications:

- [Comprehension Forms](comprehension_forms.md)
- [Comprehension Implementation](comprehension_implementation_plan.md)
- [Subcontext Construction](subcontext_construction.md)
- [Expression Engine](expression_engine.md)

## Ownership summary

| Surface | Owner | Canonical API |
|---|---|---|
| Text and serde input | Polydat | `spec::ComprehensionSpec`, `spec::parse_text`, `parse_inline` |
| Algebra AST | Polydat | `Comprehension`, `Source`, `StrategyName`, `ZipMode` |
| Validation and metadata | Polydat | `validate`, `Metadata` |
| Optimization and IR | Polydat | `optimize`, `ir::compile`, `ir::Program` |
| Static consumption | Polydat | `surfaces::*` |
| Runtime tuple evaluation | Polydat | `runtime::evaluate_for_iteration` |
| Kernel interpolation | Polydat | `kernel::interp::interpolate_via_kernel` |
| Child-scope construction | Polydat | `kernel::subcontext::SubcontextBuilder` |
| Workload traversal and child policy | External host | Calls the Polydat surfaces above |

## 1. Text and serde input

`ComprehensionSpec` is the author-facing serde shape.
`parse_text` is the full text-block entry point, and
`parse_inline` handles the inline form supported by the spec.
Every successful input path ends in
`ComprehensionSpec::into_algebra()`.

`ast_legacy`, `parse`, and `spec::legacy_convert` (in
`polydat-grammar`) are compatibility internals used to
recognize established grammar. Their types must not cross the
canonical public boundary or be retained in a workload model.
`eval::evaluate_spec` (in `polydat-core`) is the runtime's
clause-source evaluator; only `eval::enumerate_tuples` is legacy
and unused on the production path.

Input-shape detection belongs at this surface. Downstream
stages receive one canonical operator tree and must not branch
on whether the author used YAML, JSON, inline text, or a legacy
spelling.

## 2. Canonical AST

The canonical data model is the six-form `Comprehension`
operator tree:

```text
Clause
Cartesian(children)
Zip(mode, children)
Union(children)
Filter(predicate, child)
Order(strategy, child)
```

Hosts store this type when they need to retain a comprehension.
They do not store parser clauses, flattened subspaces, or an
independent host-side algebra.

## 3. Validation, metadata, and optimization

Before execution, callers validate the canonical tree and use
the metadata and optimizer defined by Polydat. A host must not
duplicate V1–V9, infer a competing cardinality model, or skip
mandatory optimization while claiming the canonical compiled
semantics.

Validation and optimization are caller-invoked passes. The
in-tree consumers (`surfaces::compile`,
`CompiledComprehension::from_ast`, `evaluate_for_iteration`) do
not run them: the surfaces go straight to `ir::compile`, and the
runtime evaluator enforces V4 at strategy invocation
(`Strategy::accepts_input`).

Optimization is conservative: lack of proof leaves the tree
unchanged. An optimizer finding is diagnostic data, not a new
host-facing execution language.

## 4. Execution

Two execution families intentionally coexist:

- The IR compiler/interpreter and `surfaces::*` implement
  algebra-native streams for supported static sources.
- `runtime::evaluate_for_iteration` evaluates executor-facing
  tuples against a `Lookup` scope (`PolydatKernel` or
  `Layered`), so it is engine-neutral, including dependent
  Cartesian sources, interpolation, filtering, and ordering.

The runtime evaluator is the authoritative choice when source
meaning depends on the current kernel scope. The static
interpreter's conservative handling of unsupported sources or
predicates must not leak into production execution.

Each consumer owns independent dispense state. Sharing an AST,
metadata value, or immutable IR program does not authorize
sharing cursors, strategy state, mutable buffers, or per-fiber
kernel state.

## 5. Interpolation

Kernel-aware string interpolation is owned by
`polydat::kernel::interp`, not by the comprehension parser.
Comprehension source evaluation calls
`interpolate_via_kernel` when source or predicate text refers
to scope bindings.

This keeps one name-resolution contract:

```text
placeholder
    │
    ▼
kernel.lookup(name)
    │
    ├── defined constant output
    └── cell-aware input / inherited binding
```

Unresolved names are typed errors. Nested placeholders follow
the bounded fixed-point rules in the expression/interpolation
spec; a host must not add a second interpolation dialect.

## 6. Child-scope construction

The comprehension layer determines tuple order and values. It
does not own workload-specific child contents. An external
scope walker translates a tuple and its workload body into
calls on the parent's `SubcontextBuilder`.

The ownership split is:

| Operation | Owner |
|---|---|
| Select next coordinate tuple | Polydat comprehension runtime |
| Decide workload imports, exports, and body fragments | Host scope walker |
| Validate parent/child contracts | `SubcontextBuilder::finalize` |
| Compile module matter | Polydat |
| Bind/spawn named child | Parent `ScopeKernel` |
| Release a replaceable iteration child | Host, through `release_child` |

No comprehension-specific direct call to `compile_polydat`,
`from_program`, or the private materializer may bypass the parent
construction gate.

## 7. Literal formatting

Formatting host-owned values as authored Polydat source is a
host concern. The host must emit syntactically valid typed
literals and must not place parser compatibility types into the
canonical AST.

Formatting values already inside a kernel is a Polydat concern
and uses the ordinary `Value`, interpolation, and adapter
contracts.

## 8. Removed boundaries

There is no separate comprehension `iteration` module or
`order` module. Iteration is owned by `runtime` and
`surfaces`; strategies own ordering. There is no
comprehension-specific source-synthesis module: child program
construction uses the general subcontext protocol.

These absences are architectural constraints. Reintroducing a
parallel owner would create competing semantics and requires a
new specification change, not a compatibility helper.

## 9. Error ownership

- Parse/serde shape errors originate in `spec`.
- Algebra validity errors originate in `validate`.
- Bound and IR-shape errors originate in `ir`.
- Runtime source, interpolation, predicate, and strategy errors
  originate in `runtime`.
- Parent/child contract errors originate in
  `kernel::subcontext`.

Hosts may add source locations and workload context, but must
preserve the underlying error category and causal chain.
