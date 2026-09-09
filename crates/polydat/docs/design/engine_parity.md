# Engine Parity — Review and True-Up Plan

**Status:** Proposed SRD 116. This document is a review, recorded on
2026-09-09, of every place the compilation levels differ in anything
other than performance, and the plan that removes each difference.
Nothing in it has landed; §5 is the plan, §4 the findings it answers.

**Ownership.** Polydat owns the feature set, the engines, and the public
API that names them. A host chooses an engine for speed and for nothing
else.

**Companion documents.** [Engines](engines.md) (the lattice and the
equivalence contract this document tightens), [Compiled Non-Scalar
Slots](compiled_handles.md) (the handle tiers), [Runtime
Model](runtime_model.md), [The `for` Construct](for_traversal.md),
[Cursor Partitions](cursor_partitions.md), and the
[embedding guide](../guides/embedding.md), which is the documented API a
host sees.

## 1. The ideal

There is one feature set. It is defined by the documented public API and
by the language: every declaration, node, construct, and runtime
interaction a program can express. Every engine, the interpreter (P1),
the closure tier (P2), the hybrid kernel, and native code (P3), accepts
every program in that set and produces the same values, the same
`None`s, the same failures, and the same side effects for the same
inputs and the same reads. Choosing an engine changes how fast a program
runs and nothing else.

[Engines](engines.md) §6 states this for "any graph accepted by two
engine forms". The qualifier is the gap: an engine that refuses a graph
is not equivalent to one that accepts it, and a host that wants the
fast engine must first learn which programs it may write. §7 there
lists the refusals as unsupported combinations. This document treats
each as a defect to remove, and adds the differences that are not
refusals at all: divergent semantics, divergent failure reporting, and a
public API whose shape differs by engine.

## 2. Method

Three surveys, all reproducible from the tree at this revision:

- **A node-by-engine matrix.** Every registered public node (96, the
  `__` adapters and file-backed nodes excluded) compiled and run through
  one cycle on each engine, from one program per node: the explicit case
  in `tests/function_coverage.rs` where it has one, otherwise the same
  synthesized call that test uses. The engines are the interpreter with
  cones off, the raw closure kernel, the hybrid kernel, and the pure
  native push-pull kernel. The probe that produced it is described in
  §5 step 1, which makes it a permanent test.
- **Constructs and runtime interactions**, probed one program each:
  register inputs, vectors, cursors, `shared` bindings, `for`
  traversals and producers, side-channel nodes, externs with and
  without defaults, and host-set extension values.
- **The API inventory**, from the public items of `kernel::state`,
  `kernel::engines`, `compile::assembly`, `compile::closures`,
  `compile::hybrid`, `compile::jit::kernels`, and `compile::select`,
  and the count of public items with no documentation
  (`RUSTDOCFLAGS=-W missing_docs`).

The matrix at this revision:

| Engine | Accepts | Refuses or fails |
| --- | ---: | ---: |
| Interpreter (P1) | 96 | 0 |
| Closure tier (P2) | 59 | 37 |
| Hybrid | 69 | 27 |
| Pure native (P3) | 39 | 57 |

The 27 the hybrid kernel refuses are the 17 of A1 and the 10 cursor
consumers of A3; the 37 the closure tier refuses add the 10 of A2.

## 3. The public API today

What a host holds after compiling, by engine. "Same" means the same
name, signature, and meaning as the interpreter kernel.

| Surface | `PolydatKernel` (P1) | P2 kernels (4 variants) | Hybrid (3 variants) | P3 kernels (4 variants) |
| --- | --- | --- | --- | --- |
| Construct | `compile_polydat*`, `compile()` | `try_compile*` → `Result<_, Box<PolydatKernel>>` | `compile_hybrid` → `Result<_, String>` | `try_compile_jit*` → `Result<_, String>` |
| Coordinates | `set_inputs(&[u64])` then `pull` | `eval(&[u64])` | `eval` | `eval` |
| Externs | `set_input(name, Value)`, `get_input` | `set_input`, `externs()` | same | same |
| Read a value | `pull(name) -> &Value` (lazy) | `get_value(name) -> Value` after `eval` | same | same |
| Read raw bits | none | `get`, `get_slot` (refuse handle slots) | same | same |
| Read vectors | through `Value` | `read_vec_*` | `read_vec_*` | none (refused at compile) |
| Introspect | `program()`, `input_names`, `output_names`, `get_constant`, `lookup` | `output_names`, `resolve_output`, `coord_count` (counts slots) | same plus `engine_counts` | `output_names`, `resolve_output`, `coord_count` |
| Share across threads | `into_program()` then `create_state()` per thread | none: compile once per thread | none | none |
| Traversal | `traverse(i)`, activations, `for_iteration` | none | none | none |
| Cursors | host `cursor_over_partitions` on the state; activations seed their bodies | none | none | none |
| Shared bindings | cells, write-through, broadcast | an ordinary input | same | same |
| Failure of a node | panic enriched with node name and inputs | raw panic from the closure or helper | same | same |
| Engine selection | `set_jit_mode` (production kernel mixes cones) | `auto_compile_p2` → `P2Engine` | one form | `auto_compile_p3` → `P3Engine` |

The `P2Engine` and `P3Engine` selectors expose `eval`, `eval_for_slot`,
`get_slot`, `resolve_output`, and `coord_count` only: no typed read and
no extern setter.

## 4. Aberrations

Each item states the difference, where it is in the code, how it
reaches a host, and what removes it. Severity: **functional** (an
engine refuses or cannot express something), **semantic** (the same
program behaves differently), **API** (the same thing is spelled
differently or is missing), **documentation**.

### A1. Seventeen nodes run on the interpreter only — functional

`to_hex`, `from_hex`, `sha256`, `md5`, `to_base64`, `from_base64`,
`histribution`, `weighted_strings`, `weighted_u64`, `one_of_weighted`,
`regex_match`, `regex_replace`, `counter`, `elapsed_millis`,
`fft_analyze`, `body_column_i32`, and `reg_shuffle_bytes` have no
compiled form: the closure tier and the hybrid kernel refuse a program
that contains one (`hybrid.rs`: "has no compiled form and can't be
JIT-compiled"), and the production kernel runs them interpreted. Five
causes, each in the node attribute's kit eligibility
(`polydat-derive/src/lib.rs`, `jit_eligible`, `slot_eligible`, and the
handle plan):

- A byte-string argument (`&[u8]`) or return (`Vec<u8>`) has no kit.
  The `Bytes` handle exists; the kits do not read or write it.
- A setup argument derived from consts (`#[poly_const(..., from =
  spec)]`) is admitted by the handle kit only when the node also has a
  handle shape, and never by the u64 kit. `histribution`, the weighted
  family, and the regex nodes are scalar-or-string nodes with a setup,
  so no kit takes them.
- A session-static setup (`from = ()`) is refused by every kit.
  `counter` and `elapsed_millis` are the two cases; `session_start_millis`
  is the same shape but has a native lowering, so it is missing on P2
  only (A2).
- A `Value` argument with a `Vec<i32>` return (`body_column_i32`)
  matches neither the handle kit (vector return) nor the slot kit
  (polymorphic argument).
- A `Const<Vec<u64>>` with a 128-bit register wire (`reg_shuffle_bytes`)
  is a const-list shape the u64 kit does not capture.

True-up: extend the u64 kit to capture setup arguments the way the
handle kit does (recomputed from the captured consts at closure
creation), including the session-static form, which is a construction
capture like a fallible node's cached value; give both kits the `Bytes`
handle through `put_thread_bytes` and `resolve_thread_bytes`; let the
slot kit take a `Value` argument decoded by wire type; and let the u64
kit capture `Const<Vec<_>>`. Each is a plan in the derive crate with a
differential in `tests/ext_tiers.rs` and a fuzzer arm.

### A2. Ten nodes run compiled only with the JIT — functional

`dist_normal`, `dist_exponential`, `dist_uniform`, `dist_pareto`,
`dist_zipf`, `dist_empirical`, `perlin_2d`, `simplex_2d`,
`fractal_noise_2d`, and `session_start_millis` have a native lowering
(`jit_constants` overrides bake their lookup tables) but no closure, so
a build without the `jit` feature, or a hybrid kernel on it, runs them
interpreted, and the raw closure kernel refuses them. The cause is the
same setup-argument rule as A1. True-up: the u64 kit's setup capture
(A1) covers all ten.

### A3. Cursors are an interpreter feature — functional and semantic

A `cursor` declaration becomes an `ExternalWrite` input named
`<name>__cursor` with a `Value::None` default (`dsl/compile.rs`,
`process_cursor`), plus the extent bindings the compiler defers. No
engine seeds it at a root scope: a host resolves the partition with
`cursor_over_partitions` on a `PolydatState`, as the binary does, and
the traversal runtime does the same for each activation's body. A
compiled kernel offers no such call, and the difference then shows in
two ways. The ten partition consumers in the matrix (`cardinality`,
`start_of`, `end_of`, `idx_of`, `count_of`, `mod_in`, `at`, `clamp_in`,
`random_in`, `subdivide`) compile on every engine; a compiled kernel
stops at its first run with "extern has no value", while the
interpreter propagates `None` through the consumer and returns `None`
for the output (probe: `q__cursor` is `None` and `out` is `None` on
both the assembler and the kernel path). That is A12's difference again,
and the host has no way to close it on a compiled kernel.

True-up: give every kernel the same cursor resolution the state offers
through `cursor_over_partitions`, as a `set_cursor(name, Partition)` on
the `Kernel` trait of A8 that writes through `compile::externs`, and
have the compiler seed a cursor whose `over` clause and extent are
constant at build, so the common case needs no host call on any engine.

### A4. Pure native code refuses half the library — functional

Thirty nodes with no native lowering and every vector-bearing node
(`Ref2` ports) make `try_compile_jit*` fail, while the hybrid kernel
accepts all of them: `json_merge`, `escape_json`, `html_encode`,
`html_decode`, `url_encode`, `url_decode`, `date_components`,
`format_u64`, `random_range`, `random_f64`, `env`, `partitions`,
`partition_count`, `partition_at`, the seven `vec_*` and `lid_mle`,
`reg_gather_f32`, `vec_to_reg_f32`, `reg_to_vec_f32`, the register lane
accessors, `reg_mul_i8`, `reg_dot_f32`, every side-channel node, and
every host node without a native form. `str_concat` with a non-string
wire falls back as well.

This is by design of the pure tier: it is the differential oracle for
native lowering. But it is also a public entry point a host can reach,
and a host that reaches it gets a refusal where the hybrid kernel would
have run the same program with the same native code for every node
that has one. True-up: make the hybrid kernel the public P3. The pure
native kernels stay as `pub(crate)` or `#[doc(hidden)]` surfaces for the
differential suites, `try_compile_jit*` becomes the hybrid constructor
family, and the `P3Engine` selector wraps hybrid variants. The
performance guide's ladder keeps measuring the pure kernel, which is
what it is for.

### A5. Traversals and producers are a kernel-path feature — functional

A `for` statement or producer compiles only through `compile_polydat*`,
which builds activation programs and a runtime the host drives with
`traverse`. The assembler entry point fails on either with a message
that has been wrong since SRD 113 step 2 landed ("the `for` construct is
parsed but not compiled yet"; `dsl/compile.rs`, two sites). A compiled
kernel cannot be a traversal's parent or its body.

True-up, in two parts. First, the message: the assembler entry point
should say that traversals are driven through `PolydatKernel::traverse`
and point at the guide. Second, the capability: an activation's body is
a program like any other, so `ActivationStream` should be able to
compile bodies with the engine the host chose and hand back a kernel
with the same drive and read API (A8). This is the largest item in the
plan and depends on A8.

### A6. Evaluation is lazy on the interpreter and eager everywhere else — semantic

`PolydatState::pull` evaluates the cone of the requested output and
nothing else. `eval` on every compiled kernel runs every step, or every
dirty step, whether or not the host reads the result; `eval_for_slot`
skips only when a cone guard proves the slot unaffected. Three
observable consequences:

- A side-channel node such as `emit_row` fires once per `eval` on a
  compiled kernel and once per pull of its wire on the interpreter.
  [Engines](engines.md) §6 property 4 forbids moving such a node where
  caching "would suppress required observations"; it says nothing about
  producing observations the interpreter would not, and that is what
  happens.
- A node that fails, `at` out of range for one, fails on a compiled
  kernel even when the host never reads its output; on the interpreter
  it fails only when pulled.
- An unset extern or cursor stops a compiled kernel at the start of the
  run (A12); on the interpreter it is a `None` that reaches only the
  consumers the host pulls.

True-up: give every engine the same evaluation contract, and make it the
lazy one, since that is the semantics the language defines and the one
the guide documents. The compiled kernels already carry per-slot
provenance masks and per-step clean flags; a `pull(name)` on a compiled
kernel is "run the steps in the cone of this slot that are not clean",
which is `eval_for_slot` with the step set restricted to the cone. Eager
`eval` stays as the fast path for hosts that read everything, and is
documented as evaluating every output. Side-channel steps then fire
exactly when their wire is pulled on every engine.

### A7. Failures are reported differently — semantic

The interpreter catches a node's panic and re-raises it with the node's
name, its position, and its input values (`kernel/engines.rs`,
`enrich_eval_panic`). A compiled kernel raises the closure's or
helper's panic as it is, and a native kernel raises it through the
longjmp catch with no node attribution at all. The message a host logs
for the same failing program therefore differs by engine, and the
compiled one does not say which node failed.

True-up: every compiled kernel keeps a step-to-node map (the hybrid
kernel already retains its nodes); the eval loop catches a panic at the
step boundary, on the failure path only, and re-raises it enriched the
same way, with the decoded input values where the slot types allow.
Native segments attribute to the segment's node list.

### A8. The API is spelled differently by engine — API

- **Construction.** `try_compile*` return the interpreter kernel as the
  error; `try_compile_jit*` and `compile_hybrid` return a string. Four
  P2 variants, four P3 variants, and one hybrid form are nine
  constructors for one decision, provenance mode, that the selector can
  make.
- **Driving.** `set_inputs` and `pull` on the interpreter; `eval` and
  `get_value` on the rest. `coord_count` on a compiled kernel counts
  input slots, externs included, not coordinates.
- **Reading.** `pull` returns a borrowed `Value`; `get_value` an owned
  one. `get`, `get_slot`, and `read_vec_*` have no interpreter
  counterpart. `P2Engine` and `P3Engine` have no `get_value` and no
  `set_input`.
- **Introspection.** `program()`, `input_names`, `get_constant`, and
  `lookup` exist on the interpreter only; `engine_counts` on the hybrid
  only; `retain_nodes` and `into_parts` are public but internal.
- **Sharing.** `into_program` and `create_state` exist on the
  interpreter only; a host runs a compiled kernel on many threads by
  compiling it on each.

True-up: one `Kernel` trait, implemented by every engine, with
`set_inputs`, `set_input`, `pull(name) -> Value`, `eval`, `input_names`,
`output_names`, `output_type(name)`, and `externs`; one constructor,
`compile_with(Engine)` on the assembler and `compile_polydat_with(src,
Engine)` on the DSL entry, where `Engine` names interpreter, closures,
hybrid, or native with an optional provenance mode; one error type. The
raw-bit readers stay as engine-specific extras, documented as such.
Sharing across threads becomes `into_program` on every engine, with a
per-thread state that owns the buffer, the table, and the externs.

### A9. The assembler entry point takes fewer options — API

`compile_polydat_to_assembler` has no source directory, no library
paths, no strict flag, and no compile log, so modules from disk and
strict typing are reachable only through the kernel path. True-up:
`compile_polydat_to_assembler_with(src, CompileOptions)` sharing the
options struct with the kernel entry points, which then reduce to
wrappers.

### A10. `shared` bindings lose their cells off the interpreter — semantic

A `shared` binding compiles on every engine (probe: register, closure,
hybrid, and native all accept `shared counter := 0`), but only the
interpreter state attaches the cross-fiber cell, commits write-throughs,
and advances broadcasts. On a compiled kernel the binding is an
ordinary input that `set_input` overwrites and nothing publishes.
True-up: either the shared-cell protocol reaches compiled kernels
through `compile::externs` (a cell-bound slot is materialized from the
cell at run start and published at run end), or `shared` is refused at
compile time on engines that cannot honor it. The first is the ideal;
the second is the honest interim and belongs in step 2 of the plan.

### A11. Vectors have no native form — functional, by design

Every `Ref2` port keeps a node off pure native code
(`build_jit_layout`: "pure-P3 kernels carry no reference slots"). The
hybrid kernel runs vector nodes as closures and reads them with
`read_vec_*`. Under A4 this stops being host-visible; it remains a
limit of the pure tier and belongs in the SIMD and register documents.

### A12. An unset extern is a `None` on the interpreter and a refusal elsewhere — semantic

An `extern` without a default and never set is `Value::None` on the
interpreter, which propagates to its consumers' outputs. On a compiled
kernel the same extern stops the first run with "extern has no value".
The compiled behavior is the clearer one and the interpreter's is the
defined one. True-up: under A6 the compiled kernels materialize an unset
table-kind extern as a `None` entry that decodes to `None`, so the
consumer sees what the interpreter's consumer sees, and the refusal
becomes a compile-time diagnostic that names every extern without a
default so a host knows what it must set.

### A13. The public API is largely undocumented — documentation

2,249 public items have no rustdoc. In the host-facing modules:
`dsl::events` 57, `dsl::compile` 35, `library::tile_render` 30,
`compile::select` 21, `compile::jit::codegen` 18, `compile::closures`
18, `compile::assembly` 15, `kernel::program` 9; the rest are library
nodes and internal types that `pub` exposes. The compilation guide says
the compiled artifact "is the same shape at each level", which §3 shows
is not so. [Engines](engines.md) §7 lists categories of refusal, not the
nodes, and no document lists what each engine accepts.

True-up: `#![warn(missing_docs)]` on the crate with a burn-down to zero,
enforced in CI like the other rustdoc lints once it reaches zero; the
node-by-engine matrix generated by a test into the node reference, so
the documented feature set and the tested one are the same file; the
compilation guide's claim replaced by the `Kernel` trait once A8 lands.

### A14. The benchmark's reference run predates the passthrough steps — documentation

Recorded in the performance guide on 2026-09-09. A re-record on a quiet
machine closes it.

## 5. The plan

Ordered so each step makes the next one testable, with the test that
proves it.

1. **Pin the matrix.** A permanent `tests/engine_parity.rs` builds the
   node-by-engine matrix of §2 from the coverage cases, compares it to a
   checked-in table, and fails on any change in either direction. The
   table starts as the state recorded here and shrinks as steps land;
   the test is the definition of done for A1 through A4. Also fix the
   traversal message (A5, first part) and make the assembler entry point
   take the kernel path's options (A9).
2. **Close the closure tier.** Setup capture in the u64 kit, the
   `Bytes` handle in both kits, `Value` arguments in the slot kit,
   `Const<Vec<_>>` capture, and the session-static form (A1, A2). Refuse
   `shared` at compile time on compiled engines until A10's protocol
   lands. Every node then has a closure, and the JIT-less build runs the
   whole library compiled. Differentials in `tests/ext_tiers.rs`, arms
   in the fuzzer.
3. **Seed cursors on every engine** (A3): resolve at build, seed through
   `compile::externs`, `set_cursor` on every kernel. Differential over
   the partition family with narrowed cursors.
4. **One kernel API** (A8): the `Kernel` trait, `compile_with(Engine)`,
   one error type, `into_program` and per-thread states for compiled
   engines. The guide's compiled-kernel sections rewrite to it. The old
   constructors remain for one release as documented aliases.
5. **Lazy evaluation everywhere** (A6, A12): `pull` on compiled kernels
   runs the requested cone; unset externs decode to `None`; side
   channels fire per pull. The fuzzer's `check` switches from `eval` to
   `pull` for every engine and adds a side-channel counter to the
   comparison.
6. **Attributed failures** (A7): step-to-node maps and the shared
   enrichment on the failure path, with a test that the same failing
   program produces the same message on every engine.
7. **Hybrid becomes P3** (A4, A11): the pure native kernels retreat to
   the differential suites; `try_compile_jit*` and `P3Engine` become
   hybrid. The performance guide is re-recorded (A14) against the new
   ladder.
8. **Traversals on compiled engines** (A5, second part): activation
   bodies compiled with the host's engine choice behind the `Kernel`
   trait.
9. **Shared cells on compiled engines** (A10) through `compile::externs`
   materialization and publication.
10. **Documentation to zero** (A13): `missing_docs` burn-down under CI,
    the generated matrix in the node reference, the guide and the
    engines document updated to state that every engine accepts every
    program.

Steps 1 through 3 are contained in the derive crate, the assembler, and
the extern plumbing that landed on 2026-09-09, and change no public
signature. Step 4 is the API change and should land as one release.
Steps 5 through 7 change semantics and belong behind the equivalence
harness that step 5 extends. Steps 8 and 9 are new capability.

## 6. What does not change

The interpreter remains the oracle. No step coerces a type, moves a
node into an engine whose caching would suppress an observation, or
changes a public port type or named output. The determinism axioms of
[Runtime Model](runtime_model.md) hold on every engine before and after
this plan; what the plan adds is that a host no longer has to know which
engine it is on.
