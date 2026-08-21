# Comprehension Regression Contract

This document defines the permanent acceptance gate for changes
to Polydat's comprehension subsystem. It replaces the historical
cutover checklist; fixed test counts and PR sequencing are not
specification.

## Acceptance rule

A comprehension change is acceptable only when all of the
following remain true:

1. Every public input form normalizes to the canonical
   `Comprehension` AST.
2. Validation enforces V1–V9 without host-side substitutes.
3. Metadata is total for every accepted AST node.
4. Optimization preserves the exact dispense sequence and
   tuple values for every rule precondition.
5. IR compilation preserves stack shape and passes resource
   bounds before execution.
6. Static and runtime evaluators agree on their common
   supported domain.
7. Dependent Cartesian products evaluate downstream sources
   against the selected prefix environment.
8. Seeded order strategies repeat exactly for equal authored
   inputs and seeds.
9. Independent consumers do not share mutable dispense state.
10. Scope-dependent execution uses the canonical kernel lookup
    and subcontext construction protocols.

Any violation rejects the change even if a compatibility parser
or host adapter can hide it.

## Required verification surfaces

The regression suite is organized by contract, not by a frozen
number of tests:

| Contract | Primary integration coverage |
|---|---|
| Spec/serde/text parity | `spec_surface_parity.rs` |
| Worked algebra examples | `spec_section_11_worked_examples.rs` |
| Optimizer equivalence | `optimizer_worked_examples.rs` plus optimizer unit/property tests |
| Predicate soundness | `predicate_analyzer_soundness.rs` |
| IR compilation and execution | `ir_end_to_end.rs` |
| Resource bounds | `resource_bounds_verification.rs` |
| Consumption isolation | `surfaces_independence.rs` |
| Kernel and scope integration | `scope_composition.rs`, comprehension runtime unit tests |
| Source semantics | `source_tests.rs`, `sampling_test.rs` |
| Whole-crate behavior | the complete `polydat` test suite |

Test names and counts may evolve. Removing or weakening coverage
for one of these contracts requires replacement coverage in the
same change.

## Equivalence oracle

For every implemented optimization rule, the unoptimized tree
is the semantic oracle. Verification compares:

- tuple count;
- tuple coordinate names and values;
- dispense order;
- termination/error class;
- deterministic strategy state where applicable.

For bounded discrete trees, exhaustive comparison is preferred.
Property-generated trees must use bounded sources and stable
seeds so failures reproduce exactly.

Continuous or externally evaluated sources use contract-level
tests for cardinality, integrability, indexing, and sampling
preconditions; they are not coerced into a discrete exhaustive
oracle.

## Parser compatibility gate

Legacy parsing helpers may remain only as internal
normalization machinery. The gate fails if:

- a legacy AST type appears in a public retained model;
- two source forms normalize to semantically different
  canonical trees without an authored distinction;
- a downstream stage branches on the original source format;
- compatibility parsing bypasses canonical validation.

## Runtime integration gate

The runtime gate fails if:

- `evaluate_for_iteration` and a host produce different tuple
  order from the same canonical tree;
- scope lookup is duplicated outside
  `kernel::interp`/`PolydatKernel::lookup`;
- child kernels bypass `SubcontextBuilder`;
- one consumer's advance changes another consumer's result;
- an unsupported static-interpreter shortcut is used as
  production semantics.

## Validation command

The authoritative local gate is the complete crate suite:

```text
cargo test -p polydat
```

Targeted suites may be run while developing, but they do not
replace the complete gate at handoff.

## Change evidence

A behavior-changing patch must identify:

- the specification clause changed;
- the affected canonical representation or stage;
- the old and new dispense/error behavior;
- the tests that prove the new contract;
- whether parser compatibility or serialized forms change.

Performance-only changes must still pass the same semantic gate.
Benchmark improvement never authorizes a different tuple stream.
