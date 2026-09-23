# Porting a host from polydat 0.3.1 to 0.3.2

Written against nmbrs, the host that drove this review, but nothing here
is nmbrs-specific: it is the full set of surface changes between the
published 0.3.1 and the current tree, measured by building that host
against this one rather than by reading diffs.

The changes divide in three. [Part 1](#part-1-mechanical) is renames and
signature edits — apply them and move on.
[Part 2](#part-2-the-two-that-change-quietly) is two changes that do not
fail where they are written: one breaks somewhere else, and one does not
break at all. Read that part before you start.
[Part 3](#part-3-what-the-host-rationalizes) is the short list of places
where polydat deliberately stopped deciding something, and the host now
decides it.

## What it costs, and where

Forty-three errors over the whole host, and they are not spread evenly.
Two crates fail first on four of them; every one of the remaining
thirty-nine is in the single crate that drives kernels. So the port is
one afternoon in one file's worth of neighbourhood, not a sweep — and
the crate that only *registers* nodes clears completely once two lines
change.

Re-measured 2026-09-22 against the tree at `c7e16c7`, after the
empty-clause work and the shuffle consolidation landed: the set below is
unchanged by either, so neither adds anything to a port already planned
against it.

## Checking as you go

Build the host against a polydat working tree without editing the
host's manifest:

```sh
cargo check --workspace --all-targets --keep-going \
  --config 'patch.crates-io.polydat.path="/path/to/polydat/crates/polydat"'
```

Two traps. The lockfile pins the published version, so the patch is
ignored until you relock — cargo says `patch … was not used in the crate
graph` and otherwise builds normally, so check for that line before
trusting a clean run:

```sh
cargo update -p polydat --config 'patch.crates-io.polydat.path="…"'
```

And errors surface in waves: a crate that fails hides every crate
downstream of it. `--keep-going` gets the independent ones in one pass,
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
- `derive_support` gained `buffer_for`, `string_for` and `reserve_for`.
  A host node that sizes a buffer by a wire or constant should allocate
  through them. A size the machine cannot hold then fails as that node,
  caught and attributed like any other failure, instead of aborting the
  process in the allocator.
- `CompileOptions` gained `engine` and `ledger`. It derives `Default`,
  so a struct literal takes `..Default::default()` and compiles. Do
  read `engine` before moving on, though: it is where the engine
  preference lives now, and its default is compiled code rather than
  the interpreter.

## Part 2: the two that change quietly

Everything in part 1 fails at the line you have to edit. These two do
not, which is why they are worth reading before you start rather than
after something is strange.

### `compile_polydat` builds a different engine

It returned `Box<dyn Kernel>` and built the interpreter. It still
returns `Box<dyn Kernel>`, and now builds
[`Engine::default`](#the-kernel-is-the-path-the-engine-is-configuration)
— the most native form the build has.

**There is no compile error for this one at all.** A host that changes
nothing still moves from the interpreter to native code, at every call
site, the moment it takes the new version. For most hosts that is the
upgrade they wanted and the ladder is roughly ten to one, so the change
is worth having — but it is worth *taking*, not discovering. Two things
follow from it:

- If a call site needs the interpreter, say so: `options.engine`, or
  `compile_polydat_interpreter` when it is the concrete type it needs.
  A differential oracle is the honest case, and naming it is the fix.
- If a program compiled on the interpreter and has never run on a
  compiled tier, this is when it first does. Everything in polydat's
  own suite computes the same values on both, so the expectation is
  parity — but a host node registered from outside that suite has not
  been under that test, and this is the change that puts it there.

#### The one behaviour that does change with the engine

Values do not change with the tier. *When a nondeterministic node
reads* does, and this migration changes it for every host that was on
`compile_polydat`.

A volatile step — a node declaring `Purity::Nondeterministic`, or one
under a `volatile` binding — is read at most once per write and re-read
on the next, on every engine. But "per step" is per the *engine's*
step, and a compiled engine's step is a fused segment. Two volatile
wires are two steps on the interpreter and the closure tier, and one
segment on the native tiers:

```
w := clock_reading()      # two volatile wires,
x := clock_reading()      # pulled in one write
```

| engine | `w` and `x` |
|---|---|
| interpreter, closure tier | read separately, when each is first pulled |
| native, pure native | read together, at the first pull of either |

So a host moving from the interpreter to the default engine goes from
two readings to one. For sampling something that moves — a clock, a
metric — that is usually the better semantics, and it is the direction
this change moves you: the outputs of one cycle now come from one
instant rather than from as many instants as there were pulls.

What to check in a port:

- Code that **relied on two readings differing** within one cycle — a
  per-pull counter or sequence, say — stops differing. That is the
  breaking direction, and it is silent.
- Code that **wanted one instant** and worked around not having it
  (reading once and passing the value along) can keep the workaround;
  it is correct on every engine and stays correct.

Neither is guaranteed by the contract: two volatile reads within one
write are not promised simultaneous *nor* promised distinct
([runtime_model.md](../design/runtime_model.md) R1.v, "Read
granularity"). Two readings that must come from one instant belong in
one node returning both — that is one step on every engine. Two that
must differ need a write between them.

### `pull` changes type without moving

`PolydatKernel::pull` was an inherent method returning `&Value`. It
shadowed `Kernel::pull`, which returns an owned `Value`. Renaming the
inherent one to `pull_ref` un-shadows the trait — so a call you do not
change still compiles, and now returns `Value` instead of `&Value`.

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

Between them these two cover both ways a change can hide: one fails
somewhere other than where it was caused, the other does not fail at
all.

## Part 3: what the host rationalizes

These are not renames. In each, polydat stopped making a decision that
was not its to make, and the host makes it now.

### The kernel is the path; the engine is configuration

There is one call path and it hands back `Box<dyn Kernel>`. Which engine
built that kernel is a *value* the host configures, not a function it
picks: `CompileOptions.engine` carries it, and it already defaults to
the most native form the build has, so a host that never names an engine
gets compiled code rather than the interpreter.

| what a host called | what it calls now |
|---|---|
| `compile_polydat(src)` | unchanged — but it builds a different engine, [see part 2](#compile_polydat-builds-a-different-engine) |
| `compile_polydat_with_options(src, &o, log)` | `compile_polydat_kernel_with_options(src, &o, log)` |

The second has the signature the old one had, and reads `o.engine` —
which is one of the two fields `CompileOptions` gained, so setting it is
the same edit as making the struct literal compile again.

A host that wants a particular engine sets `options.engine` and says so,
which is the difference between a preference and a fork.

The `compile_polydat_interpreter*` entry points still exist and are
**not** a porting strategy. Adopting them is precisely how a host
acquires an interpreter-specific call path, which is the shape this
migration removes: the engine stops being a branch in the code and
becomes a field. A host on that path also opts out of the ladder, and
the ladder is roughly ten to one from interpreter to native.

When a host needs to know what it actually got, it asks **afterward and
surgically** rather than by having called differently:

- `Kernel::engine()` — the tier this kernel runs.
- `KernelProgram::plan()` — what that engine decided: native segments,
  closure steps, interpreted nodes.
- `KernelProgram::as_interpreter()` — interpreter-level detail, `Some`
  only when that is the engine, which is the honest shape for a question
  only one tier can answer.

That is the division to port to: one way in, the engine as
configuration, and introspection after the fact for the few places that
genuinely need it.

### Empty clauses are reported, not decided

0.3.1 took an `on_empty` callback and called it while evaluating. That
was removed on the premise that every caller's policy was to do nothing,
which was not true of a host that warned and failed under `strict`. The
callback is not coming back — it had polydat calling the host's policy
mid-evaluation, and made the evaluator's result depend on a closure —
but the fact it carried is now available at both levels where emptiness
is knowable.

**At construction**, a source polydat can already count as empty is a
degenerate composition, beside the trivially-false filter:

```rust
use polydat::iteration::comprehension::{Mode, validate};

let report = validate(&comp, if strict { Mode::Strict } else { Mode::Permissive })?;
for w in &report.warnings {
    log::warn!("{w}");            // Display writes the diagnostic
}
```

`Mode::Strict` makes the first warning the error, so a host's own
`strict` flag maps onto it directly rather than being re-implemented.
This catches `x in []`, `x in 5..5`, and a context-free generator the
compile evaluated to nothing. A source whose count is *not* known at
construction is `Unbounded`, never `Bounded(0)`, so an interpolated call
or a parameter without a declared length never warns here.

**At evaluation**, which is where a selector that matched nothing shows
up, ask for the reported form:

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

The counts come off the evaluation that already happened, so asking for
them evaluates nothing twice, and the record is one entry per leaf
rather than one per tuple. `evaluate_for_iteration` is unchanged and
wraps the reported form, so callers that do not want diagnostics keep
the simpler signature.

### Clause text is polydat's to parse

The clause-text parser and the flat form it produces are internal to
`polydat_grammar` now (comprehension_forms.md §14.8). A host that was
calling `parse_clause` / `parse_clause_list` to split `"var in expr"`
goes through the public spec entry and reads the algebra:

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

There is deliberately no `text -> (var, expr)` helper. The algebra
already says it, and a second entry point for the same question is a
second grammar to keep honest.

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
