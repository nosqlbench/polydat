---
type: guide
title: Porting to 0.9.0
timestamp: 2026-11-10
description: "The 0.9.0 release notes as a host reads them, covering every change since 0.8.2: what stops compiling and its fix, the behavior that changes quietly, the new capabilities, and the known issues."
tags: [release, host]
---

# Porting to 0.9.0

This guide covers porting a host from polydat 0.8 to 0.9.0. It lists
every change a host sees between 0.8.2 and 0.9.0. The changes in 0.8.1
and 0.8.2 are in [Porting to 0.8.0](porting_to_0_8_0.md).

The release makes three things true that a host builds on:

- **The registry's types can grow without breaking a host.**
  `FuncCategory`, `Overload` and `OverloadForm` are `#[non_exhaustive]`,
  as `FuncSig` and `ParamSpec` became in 0.8.0
  ([library_catalog.md](../design/library_catalog.md), "Host-registered
  nodes").
- **A port type prints as the keyword that parses back to it.**
  `PortType`'s `Display` is its keyword, so `str` where 0.8 printed
  `String` ([polydat_grammar.md](../design/polydat_grammar.md) §15).
- **A vector call never narrows an operand.** An overloaded call builds
  a form every operand reaches without narrowing when one exists, so
  `vec_dot(a_f64, b_f32)` computes in f64 instead of being refused
  ([polydat_grammar.md](../design/polydat_grammar.md) §18.1,
  `T-Call-OverloadResolve`).

## Part 1: what stops compiling

### In Rust

| Change | What breaks | Fix |
|---|---|---|
| `FuncCategory` is `#[non_exhaustive]` and has a new variant, `Uncategorised` | a `match` over `FuncCategory` outside polydat with no `_` arm (E0004) | add a `_` arm; `display_name()` names any category, so a listing that only needs a heading can use it instead of a `match` |
| `Overload` and `OverloadForm` are `#[non_exhaustive]` | an `Overload { name, forms }` or `OverloadForm { wires, func }` literal in a host's `inventory::submit!` (E0639) | `Overload::new(name, forms)` and `OverloadForm::new(wires, func)`, both `const`, so the submission is unchanged otherwise; see the example below. The fields stay public |
| `Overload::category()` returns `FuncCategory`, not `Option<FuncCategory>` | code that matched `Some`/`None` on it | use the value; `FuncCategory::Uncategorised` is what `None` was, an overload whose first form is not registered |
| `#[polydat_node(category = Uncategorised)]` is refused | a node that declared it, which 0.8 had no such variant for | name the category the node belongs to |
| `registry::lookup_call`, deprecated in 0.8.1, is removed | a call to it (E0425) | call `registry::lookup`, which answers the same for every name |

A host overload becomes a constructor call:

```rust
use polydat::ast::PortType;
use polydat::dsl::registry::{Overload, OverloadForm};

polydat::inventory::submit! {
    Overload::new(
        "my_dot",
        &[
            OverloadForm::new(&[PortType::VecF32, PortType::VecF32], "my_dot_f32"),
            OverloadForm::new(&[PortType::VecF64, PortType::VecF64], "my_dot_f64"),
        ],
    )
}
```

### In programs

Nothing that compiled under 0.8 stops compiling.

## Part 2: what changes quietly

| Change | Who notices | What to do |
|---|---|---|
| `PortType`'s `Display` prints `str` for `PortType::Str`, where 0.8 printed `String`; every other type printed its keyword already | a host that formats a `PortType` into source, such as `format!("extern {name}: {ty}")`, or into a message it matches; source polydat prints (the pretty printer, `viz`, tile lowering) now writes `extern x: str` | nothing for synthesized source: `str` and `String` both parse to the same type. Match `str` where a test or a log filter matched `String` |
| A type-mismatch message names a string port `str`: `cannot connect str output to handle input`, and `Wire<String>::extract: expected str, got u64` | a host or test matching on those messages | match `str` |
| `type_of(x)` over a string answers `"str"`, where 0.8 answered `"String"` | a program that compares `type_of`'s result with `"String"` | compare it with `"str"` |
| `overloads_by_category()` lists an overload whose first form names no registered function, last, under `FuncCategory::Uncategorised`; 0.8.2 left it out | a host whose listing is built from it, if it registers such an overload | none, or fix the registration `unlisted_overloads()` names |
| An overloaded call builds, among the forms every operand reaches by its own type or through a lossless adapter, the one agreeing with the most operand types, where 0.8 considered every form. A call 0.8 compiled builds the same form | nobody whose program compiled under 0.8 | none |

## Part 3: new capabilities

- **Mixed-width vector operands.** `vec_dot`, `vec_l2`, `vec_cosine`,
  `vec_add` and `vec_distance` over one `vec_f64` and one `vec_f32`
  operand, in either order, build the f64 form and widen the `vec_f32`
  exactly. Under 0.8 the call was refused with `cannot connect vec_f64
  output to vec_f32 input`.
- **`FuncCategory::Uncategorised`**, the category of an overload whose
  first form is not registered, so a listing shows a broken
  registration instead of hiding it. No registered function has it.

## Part 4: known issues

- **`String`, `Json` and `Ext` still parse as type keywords.** `String`
  is how polydat printed `str` before 0.9, so source a host synthesized
  then uses it; `Json` and `Ext` are older capitalised spellings. A
  later release retires them; a host that writes them should write
  `str`, `json` and `ext`.
