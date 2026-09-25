---
type: guide
title: Porting to 0.3.2
timestamp: 2026-09-25
description: "Every surface change from 0.3.1: the renames a host applies, the type change that fails elsewhere, and the decisions polydat stopped making for a host."
tags: [release, host]
---

# Porting to 0.3.2

This guide covers porting a host from polydat 0.3.1 to 0.3.2. It lists the
full set of surface changes between the published 0.3.1 and 0.3.2,
found by building a host against the new tree rather than by reading
diffs.

The changes fall into three parts. [Part 1](#part-1-mechanical) is
renames and signature edits; apply them and move on.
[Part 2](#part-2-the-two-that-change-quietly) is two changes that do not
fail where they are written: one breaks somewhere else, and one does not
break at all. Read that part before you start.
[Part 3](#part-3-what-the-host-rationalizes) is the short list of places
where polydat deliberately stopped deciding something, and the host now
decides it.

## What it costs, and where

The port produced forty-three errors over the whole host, and they are
not spread evenly. Two crates fail first, with four of them; all of the
remaining thirty-nine are in the single crate that drives kernels. So
the port is about an afternoon's work in a few closely related files,
not a sweep across the host, and the crate that only *registers* nodes
compiles once two lines change.

The count was measured again on 2026-09-22 against the tree at
`c7e16c7`, after the empty-clause and shuffle changes were merged: the
set below is unchanged by either, so neither adds anything to a port
already planned against it.

## Checking as you go

Build the host against a polydat working tree without editing the
host's manifest:

```sh
cargo check --workspace --all-targets --keep-going \
  --config 'patch.crates-io.polydat.path="/path/to/polydat/crates/polydat"'
```

Watch for two traps. First, the lockfile pins the published version, so
cargo ignores the patch until you relock. It then prints `patch … was
not used in the crate graph` and otherwise builds normally, so check for
that line before trusting a clean run:

```sh
cargo update -p polydat --config 'patch.crates-io.polydat.path="…"'
```

Second, errors surface in waves: a crate that fails hides the errors of
every crate downstream of it. `--keep-going` gets the independent ones in one pass,
but expect a second and third round as each layer starts compiling.

## Part 1: mechanical

| 0.3.1 | 0.3.2 | note |
|---|---|---|
| `PolydatKernel::pull` | `pull_ref` | see [part 2](#pull-changes-type-without-moving) |
| `PolydatKernel::pull_by_index` | `pull_ref_at` | same |
| `Box<dyn Kernel>::program()` | `into_program()`, or ask the kernel directly | `into_program` consumes; most callers wanted one accessor, not the program |
| `Box<dyn Kernel>::output_port_type(n)` | `Kernel::output_type(n)` | also `input_port_type`, `input_port_type_by_idx` |
| `dsl::compile::compile_polydat_with_options` | `compile_polydat_kernel_with_options` | same signature; the engine comes from `options.engine` — see [the engine](#the-kernel-is-the-path-the-engine-is-configuration) |
| `comprehension::spec::parse_comprehension_text` | `spec::parse_comprehension_algebra` | |
| `comprehension::spec::parse_clause`, `parse_clause_list` | `spec::serde_form::parse_inline` | see [clause text](#clause-text-is-polydats-to-parse) |
| `comprehension::runtime::EmptyClause` | — | see [empty clauses](#empty-clauses-are-reported-not-decided) |
| `evaluate_for_iteration(comp, scope, params, on_empty)` | `evaluate_for_iteration(comp, scope)` | the dropped params were a map the evaluator never read, and the callback |
| `dyn Kernel: Debug` | — | format the error, not the kernel |
| `library::register_view::RegView::new(to)` (also re-exported from `polydat_nodes::register`) | `register_view::reg_view(to)` | returns `Option<Box<dyn PolydatNode>>`, `None` for a non-register type; each view is now a registered node, `__reg_view_raw` … `__reg_view_f64x2`, callable by name |

Additive, so the fix is an extra field or arm:

- `Comprehension::Order` gained `seed: Option<u64>` — add `..` to the
  pattern, or read it if you render the comprehension back to text.
- `WriteError` gained `CoordinateSlot { .. }` — a write aimed at a
  coordinate slot, which used to be reported as something less precise.
  **This one fails silently if the host ignores the `Result`.** A named
  write (`set_input`, `set_input_at`) to a coordinate is now refused on
  all four engines, where some paths used to write it. A host that declared a
  fed value as `input x: T` (a coordinate) and writes it by name, and
  discards the write's result, keeps running on the stale value with no
  message. Declare such a value `extern x: T`, which a named write sets,
  and never discard a write's `Result`.
- `ValidationError` gained `SourceFailed { name, message }`. A producer
  over a context-free generator that cannot be evaluated is now refused
  when its stream opens. It used to open as an empty stream, so a host
  that treated "no coordinates" as a normal outcome may have been
  reading a failure.
- `derive_support` gained `buffer_for`, `string_for` and `reserve_for`,
  and `try_buffer_for` for a caller that reports failure as a value.
  A host node that sizes a buffer by a wire or constant should allocate
  through them. A size the machine cannot hold then fails as that node,
  caught and attributed like any other failure, instead of aborting the
  process in the allocator.
- `CompileOptions` gained `engine` and `ledger`. It derives `Default`,
  so a struct literal ending in `..Default::default()` compiles. Read
  about `engine` before moving on, though: it is where the host states
  which engine it prefers, and its default is compiled code rather than
  the interpreter.

Some node constants are now bounded, and a workload outside a bound is
refused when it compiles, naming the node and the parameter. Each of
these used to hang, panic, overflow or return NaN, depending on the
engine, so no working workload relied on it. Grep the workload text for
these nodes:

| node | now refused | was |
|---|---|---|
| `discretize(x, range, buckets)` | `range` not positive and finite; `buckets` of 0 | a panic for a range below `f64::EPSILON`; native compile underflowed at 0 buckets |
| `shuffle(x, feedback, size, min)` | `min + size - 1` past `u64::MAX` | an overflow panic in debug, a wrapped value in release |
| `fractal_noise_1d`, `fractal_noise_2d` | `octaves` outside 1 to 64 | a hang for large counts, NaN at 0, and `as u32` truncation |
| `n_of(x, n, m)` | `m` above 65 536 | `m` hashes per cycle, a hang for large `m` |

Partition recipes and comprehension generators whose count no machine
can hold (`linear:N`, `fib(n)`, `order halton/N`, …) are compile errors
naming the spec instead of process aborts. A count that fits is still
materialized in full.

## Part 2: the two that change quietly

Everything in part 1 fails at the line you have to edit. These two do
not, so read them before you start rather than after something behaves
unexpectedly.

### `compile_polydat` builds a different engine

It returned `Box<dyn Kernel>` and built the interpreter. It still
returns `Box<dyn Kernel>`, and now builds
[`Engine::default`](#the-kernel-is-the-path-the-engine-is-configuration)
— the most native form the build has.

**There is no compile error for this one at all.** A host that changes
nothing still moves from the interpreter to native code, at every call
site, as soon as it takes the new version. For most hosts that is the
upgrade they wanted, and native code runs roughly ten times faster than
the interpreter, so the change is worth having; but a host should make
it deliberately rather than discover it. Two things follow:

- If a call site needs the interpreter, say so with `options.engine`,
  or call `compile_polydat_interpreter` when the call site needs the
  concrete interpreter type. Using the interpreter as the reference in
  a differential test is the legitimate case, and naming the engine
  there is the fix.
- A program that compiled on the interpreter and has never run on a
  compiled tier runs on one for the first time here. Everything in
  polydat's own suite computes the same values on both, so the
  expectation is that values match; but a host node registered from
  outside that suite has never been tested that way, and this change is
  where it first is.

#### The one behaviour that does change with the engine

Values do not change with the tier. *When a nondeterministic node
reads* does, and this migration changes it for every host that was on
`compile_polydat`.

A volatile step (a node declaring `Purity::Nondeterministic`, or one
under a `volatile` binding) is read at most once per write and read
again after the next write, on all four engines. But the step is the
*engine's* step, and on the native engines a step is a fusion unit: a
connected group of nodes compiled together. Two volatile wires that
share nothing are two units, and are read as two steps on all four
engines; two that a wire connects are one unit on the native engines:

```
w := clock_reading()      # unconnected: two steps everywhere
x := clock_reading()

t := clock_reading()      # connected: one unit on native code
u := t + clock_reading()
```

| engine | `w` and `x` | `t` and `u` |
|---|---|---|
| interpreter, closure tier | read separately, when each is first pulled | read separately |
| native (the default), pure native | read separately, when each is first pulled | read together, at the first pull of either |

So a host moving from the interpreter to the default engine sees a
difference only where volatile reads are wired together, and there it
gets one reading where it got two. For sampling something that moves,
such as a clock or a metric, that is usually the better behavior: the
connected outputs of one cycle come from one instant.

What to check in a port:

- Code that **relied on two connected readings differing** within one
  cycle — a per-pull counter or sequence feeding another, say — stops
  differing. That is the breaking direction, and it is silent.
- Code that **wanted one instant** and worked around not having it
  (reading once and passing the value along) can keep the workaround;
  it is correct on all four engines and stays correct.

The contract guarantees neither: two volatile reads within one write
are not promised to be simultaneous, nor promised to be distinct
([runtime_model.md](../design/runtime_model.md) R1.v, "Read
granularity"). Two readings that must come from one instant belong in
one node returning both, which is one step on all four engines. Two
that must differ need a write between them.

### `pull` changes type without moving

`PolydatKernel::pull` was an inherent method returning `&Value`. It
shadowed `Kernel::pull`, which returns an owned `Value`. The inherent
method is renamed to `pull_ref`, so a `.pull()` call you do not change
now resolves to the trait method: it still compiles, and now returns
`Value` instead of `&Value`.

Nothing fails at the call site. It fails wherever the result was matched
and the bindings were dereferenced:

```
error[E0614]: type `u64` cannot be dereferenced
```

That error will point at a match arm some distance from the `.pull()`
that caused it. If you see a cluster of `cannot be dereferenced` on
`u64`, `f64`, or `bool`, look upstream for a `.pull()` that used to hand
back a reference. Decide per site: `pull_ref` to keep the borrow, or
keep the trait's `pull` and drop the `*`.

These two cover both ways a change can go unnoticed: one fails
somewhere other than where it was caused, and the other does not fail
at all.

## Part 3: what the host rationalizes

These are not renames. In each, polydat stopped making a decision that
belongs to the host, and the host makes it now.

### The kernel is the path; the engine is configuration

There is one call path, and it returns a `Box<dyn Kernel>`. The engine
that builds the kernel is a *value* the host configures, not a function
it picks: it is the `CompileOptions.engine` field, which defaults to the
most native form the build has, so a host that never names an engine
gets compiled code rather than the interpreter.

| what a host called | what it calls now |
|---|---|
| `compile_polydat(src)` | unchanged — but it builds a different engine, [see part 2](#compile_polydat-builds-a-different-engine) |
| `compile_polydat_with_options(src, &o, log)` | `compile_polydat_kernel_with_options(src, &o, log)` |

The new function has the signature the old one had and reads
`o.engine`, which is one of the two fields `CompileOptions` gained, so
setting it is part of the same edit that makes the struct literal
compile again.

A host that wants a particular engine sets `options.engine`, which
states a preference on the one call path instead of creating a second
path.

The `compile_polydat_interpreter*` entry points still exist and are
**not** a porting strategy. Adopting them gives the host an
interpreter-specific call path, which is exactly what this migration
removes: the engine stops being a branch in the code and becomes a
field. A host on that path also gives up the compiled engines, which
run roughly ten times faster than the interpreter.

When a host needs to know which engine it got, it asks the kernel after
building it, for the specific detail it needs, rather than calling a
different function:

- `Kernel::engine()` returns the tier this kernel runs on.
- `KernelProgram::plan()` returns what that engine decided: native
  segments, closure steps, interpreted nodes.
- `KernelProgram::as_interpreter()` returns interpreter-level detail. It
  is `Some` only when the engine is the interpreter, because only that
  tier can answer the question.

That is the design to port to: one way in, the engine as
configuration, and inspection after the build for the few places that
need it.

### Empty clauses are reported, not decided

0.3.1 took an `on_empty` callback and called it while evaluating. It
was removed on the assumption that every caller's policy was to do
nothing, which was not true of a host that warned, and failed under
`strict`. The callback is not coming back, because it had polydat
calling the host's policy in the middle of evaluation and made the
evaluator's result depend on a closure. Instead, whether a clause is
empty is now reported at both points where it can be known.

**At construction**, a source that polydat can already count as empty
is reported as a degenerate composition, alongside a filter that is
trivially false:

```rust
use polydat::iteration::comprehension::{Mode, validate};

let report = validate(&comp, if strict { Mode::Strict } else { Mode::Permissive })?;
for w in &report.warnings {
    log::warn!("{w}");            // Display writes the diagnostic
}
```

`validate` takes the comprehension and a mode and returns a report
whose `warnings` list the degenerate parts. `Mode::Strict` turns the
first warning into the returned error, so a host's own `strict` flag
maps onto it directly rather than being re-implemented.
This catches `x in []`, `x in 5..5`, and a context-free generator the
compile evaluated to nothing. A source whose count is *not* known at
construction is `Unbounded`, never `Bounded(0)`, so an interpolated call
or a parameter without a declared length never warns here.

**At evaluation**, which is where a selector that matched nothing
becomes visible, call the reported form. It returns the tuples together
with one record per clause, counting how many times the clause was
evaluated and how many values it yielded:

```rust
use polydat::iteration::comprehension::evaluate_for_iteration_reported;

let out = evaluate_for_iteration_reported(&comp, scope)?;
for c in &out.clauses {
    if c.evaluations > 0 && c.values == 0 {
        // reached, and produced nothing: this is the clause to name
        let label = match &c.source {
            Some(text) => format!("clause '{} in {}'", c.var, text),
            None => format!("clause '{}'", c.var),
        };
        // host policy: warn, or fail under strict
    }
}
let tuples = out.tuples;
```

The distinction to keep is between the two zero cases. A clause under a
cartesian is evaluated once per outer tuple, so:

- `evaluations > 0 && values == 0` — reached every time and yielded
  nothing. **This is the cause**, and the one to put in a message.
- `evaluations == 0` — never reached, because something outside it was
  empty first. Reporting this one names a symptom and buries the cause.

The counts are collected during the evaluation itself, so asking for
them evaluates nothing twice, and the record has one entry per leaf
clause rather than one per tuple. `evaluate_for_iteration` is unchanged and
wraps the reported form, so callers that do not want diagnostics keep
the simpler signature.

### Clause text is polydat's to parse

The clause-text parser and the flat form it produces are internal to
`polydat_grammar` now (comprehension_forms.md §14.8). A host that
called `parse_clause` / `parse_clause_list` to split `"var in expr"`
now parses the text with the public entry point, `parse_inline`, and
walks the resulting comprehension tree:

```rust
use polydat::iteration::comprehension::{Comprehension, spec::serde_form::parse_inline};

fn clause_pairs(c: &Comprehension, out: &mut Vec<(String, String)>) {
    match c {
        Comprehension::Clause { name, source } => {
            out.push((name.clone(), source.to_text().unwrap_or_default()));
        }
        Comprehension::Cartesian { children }
        | Comprehension::Zip { children, .. }
        | Comprehension::Union { children } => {
            for ch in children {
                clause_pairs(ch, out);
            }
        }
        Comprehension::Filter { child, .. } | Comprehension::Order { child, .. } => {
            clause_pairs(child, out);
        }
    }
}

let mut pairs = Vec::new();
clause_pairs(&parse_inline(text)?, &mut pairs);
```

`Source::to_text` round-trips: parsing what it returns yields a tree
equal to the one it came from. For text that arrived as text — which is
the case a YAML front end has — the pair you get back is the pair you
would have got from `Clause::var` and `Clause::expr`.

There is deliberately no `text -> (var, expr)` helper. The tree already
holds that information, and a second entry point for the same question
would be a second grammar to keep consistent with the first.

## Checklist

1. Patch and relock, and confirm the patch was actually used.
2. Apply [part 1](#part-1-mechanical) until the first crate compiles.
3. Expect new waves; repeat.
4. Grep for `.pull(` and decide `pull_ref` or drop-the-`*` per site
   ([part 2](#pull-changes-type-without-moving)).
5. Hold `Box<dyn Kernel>`, and know that `compile_polydat` now builds
   the default engine whether or not you touch it
   ([part 2](#compile_polydat-builds-a-different-engine)). If the host
   wants a different engine, set `options.engine` and say why there —
   not by calling a different function.
6. Run the host's own suite against the compiled tier before reading
   anything into a behaviour change: that is the first time a
   host-registered node runs anywhere but the interpreter.
7. Find every `Nondeterministic` node the host registers and every
   `volatile` binding it writes, and check nothing depends on two of
   their reads *differing* within one cycle — on the default engine
   they now come from one instant
   ([read granularity](#the-one-behaviour-that-does-change-with-the-engine)).
8. Move the empty-clause policy onto `validate` plus the per-clause
   yields, and check that a workload whose sweep resolves empty still
   says so — that is the behaviour this migration is most likely to
   drop silently.
