---
type: guide
title: Porting to 0.8.0
timestamp: 2026-11-05
description: "The 0.8.0 release notes as a host reads them, covering every change since the published 0.7.0: what stops compiling and its fix, the behavior that changes quietly, the new capabilities, and the known issues."
tags: [release, host]
---

# Porting to 0.8.0

This guide covers porting a host from polydat 0.7.0 to 0.8.0. It lists
every change a host sees between the published 0.7.0 and 0.8.0.

The release makes two things true that a host builds on:

- **A vector operation computes at the width of its operands.**
  `vec_add`, `vec_scale`, `vec_norm`, `vec_dot`, `vec_l2`, `vec_cosine`
  and `vec_distance` each have an f32 and an f64 form, and the
  unsuffixed name builds the form its operands' types agree with
  ([library_catalog.md](../design/library_catalog.md), "Vector identity
  and facet ordinal lists";
  [polydat_grammar.md](../design/polydat_grammar.md) §18.1,
  `T-Call-OverloadResolve`).
- **A signature can gain a field without breaking a host.** `FuncSig`
  and `ParamSpec` are `#[non_exhaustive]` and built through `const`
  constructors ([library_catalog.md](../design/library_catalog.md),
  "Host-registered nodes").

## Part 1: what stops compiling

### In Rust

| Change | What breaks | Fix |
|---|---|---|
| `FuncSig` and `ParamSpec` are `#[non_exhaustive]` | a `FuncSig { … }` or `ParamSpec { … }` literal in a host factory (E0639) | build them with `FuncSig::new(name, category, params)` and `ParamSpec::new(name, slot_type)` or `ParamSpec::wire(name)`, then the `const` `with_` methods; see the example below. The fields stay public, so code that reads them, or assigns one on a value it built, is unchanged |
| The vector node types are renamed for their f32 form: `VecAdd`, `VecScale`, `VecNorm`, `VecDot`, `VecL2` and `VecCosine` become `VecAddF32`, `VecScaleF32`, `VecNormF32`, `VecDotF32`, `VecL2F32` and `VecCosineF32` | code that names one of them, such as a test that builds a node directly | the new name; the f64 forms are `VecAddF64` and so on |
| `iteration::source::TimeElapsedPolicy` is removed | code that named it | `UntilElapsedPolicy { min_ms, delta: base }`, which it wrapped |

A host signature that was a literal becomes a constructor chain, and
stays a `static`:

```rust
use polydat::ast::{PortType, SlotType};
use polydat::dsl::registry::{FuncCategory, FuncSig, ParamSpec};

static SIGS: &[FuncSig] = &[FuncSig::new(
    "counted_mod",
    FuncCategory::Arithmetic,
    &[
        ParamSpec::wire("x"),
        ParamSpec::new("m", SlotType::ConstU64).with_example("7"),
    ],
)
.with_description("mod, built by a host factory that records its builds")
.with_output_port(PortType::U64)];
```

`FuncSig::new` starts from one output, a fixed arity, positional
inputs, and no description, help, identity, variadic constructor,
default resolver or output port; `ParamSpec::new` starts required, with
no constraint, and with the example `cycle` for a wire and none for a
constant. Each `with_` method, and `ParamSpec::optional`, changes one
of these.

### In programs

Nothing that compiled under 0.7.0 stops compiling.

## Part 2: what changes quietly

| Change | Who notices | What to do |
|---|---|---|
| A call to an unsuffixed vector name builds a suffixed node, so the node's `meta().name`, a panic, and an explanation name `vec_dot_f32` where 0.7.0 named `vec_dot` | a host or test matching on a node name or a panic message, such as `vec_dot: operand lengths differ (1 vs 2)` | match the suffixed name; the program text is unchanged |
| `registry::registry()` lists `vec_dot_f32` and `vec_dot_f64` where it listed `vec_dot`, and likewise for `vec_add`, `vec_scale`, `vec_norm`, `vec_l2` and `vec_cosine`; the unsuffixed names are listed by `registry::overloads()` | a host that lists or looks up the library by name | look an unsuffixed name up with `registry::lookup_call`, which answers for an overloaded name with its first form's signature, or resolve it for given wire types with `registry::resolve_call` |
| The README install lines name the current release | nobody's program; 0.7.0's lines named 0.6, and `readme_install_lines_name_the_current_release` failed on it | none |

## Part 3: new capabilities

- **Overloaded names.** A node module gives one name several typed
  forms with `inventory::submit! { Overload { name, forms } }`, each
  `OverloadForm` naming a registered function and its wire types
  (`dsl::registry::Overload`). A call builds the form agreeing with the
  most of its operands' types, position by position, the earliest among
  equals, and operands no form agrees with build the first form, which
  the adapters reach. `registry::overloads()`, `overload(name)`,
  `resolve_call(name, wire_types)` and `lookup_call(name)` answer for
  them, and the node reference lists them under "Overloaded names".
- **f64 vector operations.** `vec_add_f64`, `vec_scale_f64`,
  `vec_norm_f64`, `vec_dot_f64`, `vec_l2_f64`, `vec_cosine_f64` and
  `vec_distance_f64`. Under 0.7.0 an f64 vector passed to `vec_dot` was
  refused (`cannot connect vec_f64 output to vec_f32 input`); it now
  builds the f64 form and is never narrowed. `vec_scale(v, 2)` over a
  `vec_f64` builds `vec_scale_f64` and widens the integer factor.
- **Ground-truth distances.** `vec_distance(metric, a, b)` recomputes a
  distance as a dataset's ground truth stores it, smaller being closer:
  squared Euclidean for `EUCLIDEAN`, `1 − cos θ` for `COSINE`, `−a·b`
  for `DOT_PRODUCT`, and `Σ|aᵢ − bᵢ|` for `MANHATTAN`, accumulated in
  f64. `metric` is a name as `dataset_distance_function` returns it.
  Compare a stored distance with it, never with `vec_l2` or
  `vec_cosine`, which answer the textbook quantities.
- **Element access on every vector carrier.** `vec_len_<t>` and
  `vec_at_<t>` for `f16`, `f32`, `f64`, `i8` and `i16`, beside the
  `i32` and `i64` forms. Integer elements come out as `i64` and float
  ones as `f64`, both exactly, so two stored f32 distances that are
  equal compare equal through `f64_eq`. Past the end the element is `0`
  or `0.0`.
- **`str_to_vec_f32(s)`** puts a JSON array of numbers onto a `vec_f32`
  wire, each element the f32 nearest the f64 its text parses as. A
  number too large for an f32 fails by name rather than becoming an
  infinity.

## Part 4: known issues

- **Mixed-width operands are refused with an error about the f32
  form.** `vec_dot(a_f64, b_f32)` agrees with each form in one position,
  so it builds the first form, and the assembler refuses the f64
  operand: `cannot connect vec_f64 output to vec_f32 input`. The call
  is correct to refuse, since no conversion keeps both operands, but
  the message names a port the author did not choose. Convert one
  operand explicitly.
- **`PortType::Str` displays as `String`**, while its keyword is `str`.
  Both parse, and source polydat prints through `Display` (the pretty
  printer, `viz`, and tile lowering) writes `extern x: String`, so the
  `String` spelling is not yet retired.
