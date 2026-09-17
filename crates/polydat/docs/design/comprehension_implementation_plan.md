# Comprehension Implementation Specification

This document specifies how the comprehension algebra is
implemented. The filename is retained because other design
documents link to it; delivery sequencing and PR history are
not part of the specification.

The semantic authority is
[comprehension_forms.md](comprehension_forms.md). This document
maps that algebra to modules, compiler stages, execution
surfaces, and verification obligations.

## Canonical module

The canonical public namespace is
`polydat::iteration::comprehension` (re-exported through the
crate's public iteration surface). Its operator-tree AST is
`Comprehension` in `ast.rs`. The language half lives in
`polydat-grammar`, the runtime half in `polydat-core`; both are
reachable at `polydat::iteration::comprehension::*`.

The files `ast_legacy.rs`, `parse.rs`, and
`spec/legacy_convert.rs` are parse-pipeline compatibility
internals. They may accept established source forms, but no
runtime or host model may retain a legacy AST after conversion
to the canonical `Comprehension` tree. `eval.rs` is the
runtime's clause-source evaluator (`eval::evaluate_spec`);
`eval::enumerate_tuples` enumerates the `ast_legacy` clause form
directly; polydat's own paths call `runtime::evaluate_for_iteration`.

## Compilation pipeline

```text
source text / serde form
        │
        ▼
ComprehensionSpec
        │ parse + legacy normalization
        ▼
canonical Comprehension AST
        │
        ├── validate (V1–V9)
        ├── derive metadata
        └── optimize to fixed point
              │
              ▼
        immutable IR Program
              │
              ├── resource-bound check
              ├── static consumption surfaces
              └── runtime iteration evaluator
```

Each stage either returns a well-formed value for the next
stage or a typed error. Later stages do not repair an invalid
earlier representation.

The stages are separate passes over the canonical AST:
`validate`, `optimize` (AST to AST, to a fixed point),
`ir::compile` (the only AST-to-IR path), and `check_bounds`.
`surfaces::compile` and `CompiledComprehension::from_ast` run
only `ir::compile`; a caller that wants validation,
optimization, or bounds runs those passes first.

## Representation ownership

| Concern | Module | Contract |
|---|---|---|
| Operator-tree AST | `ast.rs` | Six forms: clause, Cartesian, zip, union, filter, order. |
| Source vocabulary | `source.rs` | Literal, integer range, generator, workload-param list, continuous interval, distribution. |
| Cardinality | `cardinality.rs` | Finite/infinite/continuous/hybrid classification and measure constraints. |
| Strategy vocabulary | `strategy.rs`, `strategies/` | Closed strategy and zip-mode enums with deterministic implementations. |
| Validation | `validate.rs` | V1–V9 structural and semantic checks. |
| Metadata | `metadata.rs` | Cardinality, index function, natural order, and materialization behavior. |
| Predicate analysis | `predicate/` | Conservative coordinate-dependency and recognizer catalog. |
| Optimization | `optimize/` | R0a, R0b, R3, R4, R5, R6, and R7 only. |
| IR | `ir/` | Immutable stack program, resource bounds, and static interpreter. |
| Evaluated sources | `eval_source.rs` | Runtime values plus runtime-query index behavior. |
| Runtime iteration | `runtime.rs` | Dependent-product evaluation against a kernel scope. |
| Consumption | `surfaces/` | Independent coordinate, instance, scope-once, compiled, and scoped-stream consumers. |

The optimizer has no R8, R9, or R10 semantics. A rule is part
of the implementation only when it has a `RuleId`, a concrete
module, an equivalence precondition, and verification coverage.

## Core invariants

### Stream-first execution

Clause sources are stream producers. A clause is not
normalized to `Vec<Value>` as its semantic representation.
Materialization occurs only at an IR operation whose metadata
declares a barrier, such as a non-streaming order strategy.

### Dependent Cartesian product

Cartesian evaluation is a dependent product. Each downstream
source is evaluated in the environment formed by the tuple
prefix already selected. If no source depends on that prefix,
the result is the ordinary independent Cartesian product.

Non-lexicographic Cartesian ordering requires independence;
validation rejects a strategy that would need random access
over a dependent source.

### Closed-enum behavior

`IndexFn`, `NaturalOrder`, `Materialization`,
`StrategyName`, and `ZipMode` are closed. Runtime callbacks
and process-level strategy registration are not part of the
comprehension contract. Adding a variant requires exhaustive
updates to metadata, validation, IR compilation, execution,
serde, and tests.

### Immutable public IR

`ir::Program` is constructed by the compiler and exposed by
value through read-only operations. Consumers do not mutate
opcodes or stack effects. `Program::stack_depth()` and the
resource-bound pass must agree with every opcode's declared
stack effect.

### Independent consumers

Consumption surfaces may share an AST or IR program, but each
owns its dispense cursor, strategy state, buffers, and kernel
state. Advancing one consumer cannot advance or invalidate
another.

### Deterministic seeded strategies

PRNG strategies (`Shuffle`, `Lhs`) seed from a module constant
plus the input length; there is no authored seed, so equal
inputs give equal sequences. Thread scheduling, address
layout, and iteration among sibling consumers must not alter
the sequence.

## Evaluation surfaces

The static IR interpreter executes literal/index-addressable
algebra and the predicate recognizer subset documented in
[ir_architecture.md](ir_architecture.md). It is suitable for
algebra-native consumption surfaces.

`runtime::evaluate_for_iteration` is the executor-facing path.
It evaluates sources against a `Lookup` scope (`PolydatKernel`
or `Layered`), preserves dependent-product environments, applies filters and strategies,
and returns runtime tuples.

`EvaluatedSource` distinguishes source values from source
indexability. `IndexFn` is a runtime query over the evaluated
source, not a promise inferred solely from source syntax.
Callers must consult the evaluated contract before selecting an
index-sampling strategy.

## Optimization contract

Optimization is a semantics-preserving pass the caller runs
before compiling (see the compilation pipeline above);
`surfaces::compile` and `CompiledComprehension::from_ast` do
not run it on the caller's behalf. When run, it applies the
implemented rules to a deterministic fixed point. A rule
may fire only when its structural, cardinality, order,
dependency, predicate, and materialization preconditions are
proven by current metadata.

Every finding records:

- the applied `RuleId`;
- the source location or AST locus;
- the replacement tree;
- the proof metadata used by the precondition.

If a proof is unavailable, the optimizer leaves the tree
unchanged. Conservative non-optimization is correct behavior.

## Error contract

The implementation rejects:

- invalid coordinate-name ownership or incompatible tuple
  shapes;
- illegal zip cardinality/mode combinations;
- non-integrable continuous measures;
- non-index-addressable strategies applied to sources that
  require indexing;
- non-lexicographic ordering of dependent Cartesian products;
- unbounded materialization without an accepted resource
  bound;
- unresolved runtime sources or predicates on the production
  path.

The static IR interpreter's documented conservative predicate
and unsupported-source behavior is not a substitute for the
production runtime checks.

## Verification contract

Changes to this subsystem must preserve:

1. parse/serde round trips into the canonical AST;
2. V1–V9 validation coverage;
3. metadata propagation for every AST variant;
4. optimizer before/after dispense-sequence equivalence;
5. IR stack effects and resource-bound accounting;
6. independent state across all consumption surfaces;
7. seeded-strategy repeatability;
8. agreement between static and runtime evaluation wherever
   their supported domains overlap;
9. end-to-end workload comprehension behavior.

## Integration ownership

Polydat owns parsing normalization, the canonical algebra,
optimization, IR, and runtime tuple evaluation. Host crates own
their scope walker and the policy for turning each tuple into a
child kernel. Child-kernel construction must go through
[subcontext_construction.md](subcontext_construction.md); the
comprehension layer does not recreate a parallel scope-binding
protocol.

See [comprehension_cutover_contact_surfaces.md](comprehension_cutover_contact_surfaces.md)
for the current host integration boundaries and
[comprehension_migration_gate.md](comprehension_migration_gate.md)
for the permanent regression obligations.
