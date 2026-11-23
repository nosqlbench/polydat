---
type: guide
title: Porting to 0.10.0
timestamp: 2026-11-23
description: "The 0.10.0 release notes as a host reads them, covering every change since 0.9.0: what stops compiling and its fix, the behavior that changes quietly, the new capabilities, and the known issues."
tags: [release, host]
---

# Porting to 0.10.0

This guide covers porting a host from polydat 0.9 to 0.10.0. It lists
every change a host sees between 0.9.0 and 0.10.0. The changes in 0.9.0
are in [Porting to 0.9.0](porting_to_0_9_0.md).

The release makes one thing true that a host builds on:

- **Every type has exactly one keyword.** The keyword is the one
  `PortType`'s `Display` and `to_keyword()` print, so source polydat
  prints and source a host writes name a type the same way
  ([polydat_grammar.md](../design/polydat_grammar.md) §15).

## Part 1: what stops compiling

### In programs

| Change | What breaks | Fix |
|---|---|---|
| The type spellings `String`, `Str`, `Json` and `Ext` are refused, wherever a type is named: `extern x: T`, `input x: T`, a module's parameters and outputs, `shared x: T := …`, `… as T`, and a tile hole `${x:T}` | a program that writes one, which 0.9 read as `str`, `json` or `ext`. The error names the keyword to write: `` `String` is no longer a type keyword; write `str` `` | write `str`, `json` or `ext`. Source polydat printed before 0.9.0 wrote `String`, so check stored source a host generated then, as well as source written by hand |

### In Rust

| Change | What breaks | Fix |
|---|---|---|
| `PortType::from_keyword` accepts only the keyword `to_keyword` prints: `from_keyword("String")`, `"Str"`, `"Json"` and `"Ext"` answer `None` | a host that parses type names itself, such as a workload schema's `id: String`, and relied on the old spellings | write the keyword; to tell a user what to write, `PortType::retired_keyword` names it (`Some("str")` for `"String"`), and `PortType::retired_keyword_message` gives polydat's own diagnostic. `from_workload_name` follows `from_keyword` |

## Part 2: what changes quietly

Nothing. A program or call that compiles under 0.10.0 behaves as it
did under 0.9.

## Part 3: new capabilities

- **`PortType::retired_keyword` and `PortType::retired_keyword_message`**,
  so a host that reads type names from its own formats can refuse a
  retired spelling with the same message polydat gives.

## Part 4: known issues

None.
