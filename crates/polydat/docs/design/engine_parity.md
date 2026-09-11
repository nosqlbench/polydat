# Engine Parity — Review and True-Up Plan

**Status:** SRD 116, landed. This document is a review, recorded on
2026-09-09, of every place the compilation levels differed in anything
other than performance, and the plan that removed each difference. §4
holds the findings, each with the note that closed it; §5 holds the
plan, each step with its landing record. Steps 1 through 3 landed on
2026-09-09 and steps 4 through 10 on 2026-09-10. Every engine a host
can choose now accepts every program the interpreter accepts and
computes what it computes, behind one trait; the node-by-engine matrix
the parity suite maintains in the [node reference](../reference/nodes.md)
is the standing proof, and `tests/engine_parity.rs` fails on any change
to it.

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

- **A node-by-engine matrix.** Every registered public node (the `__`
  adapters and the real-data category excluded) compiled and run through
  one cycle on each engine, from one program per node: the explicit case
  in `tests/common/coverage_cases.rs` where it has one, otherwise the
  same synthesized call the coverage test uses. The engines are the
  interpreter with cones off, the raw closure kernel, the hybrid kernel,
  and the pure native push-pull kernel. The review's first pass probed
  the 96 nodes with explicit cases; the permanent test of §5 step 1
  covers all 294 registered names (296 programs, two names being
  registered twice) and is the matrix recorded below.
- **Constructs and runtime interactions**, probed one program each:
  register inputs, vectors, cursors, `shared` bindings, `for`
  traversals and producers, side-channel nodes, externs with and
  without defaults, and host-set extension values.
- **The API inventory**, from the public items of `kernel::state`,
  `kernel::engines`, `compile::assembly`, `compile::closures`,
  `compile::hybrid`, `compile::jit::kernels`, and `compile::select`,
  and the count of public items with no documentation
  (`RUSTDOCFLAGS=-W missing_docs`).

The matrix as pinned in `tests/engine_parity.txt` (296 programs), as
the review found it and after step 2:

| Engine | Accepts | Refuses at compile | Fails at run | After step 2 |
| --- | ---: | ---: | ---: | --- |
| Interpreter (P1) | 296 | 0 | 0 | 296 / 0 / 0 |
| Closure tier (P2) | 215 | 71 | 10 | 286 / 0 / 10 |
| Hybrid | 231 | 55 | 10 | 286 / 0 / 10 |
| Pure native (P3) | 169 | 127 | 0 | 169 / 127 / 0 |

The 55 the hybrid kernel refused were the 53 nodes of A1 (two of them
with two cases); the 71 the closure tier refused added the 16 of A2;
the 10 run failures on both are the cursor consumers of A3; the 127
pure native refusals add the 62 of A4. The same test now also compares
every output of every program across the engines that ran it with the
interpreter's value (`the_engines_agree_on_every_node`).

## 3. The public API the review found

*The API today is one trait, `Kernel`, built by one constructor,
`compile_with(Engine)` on the assembler and `compile_polydat_with(src,
Engine)` on the DSL entry, with one error type, `KernelError` (step 4),
and the engines are the interpreter, the closure tier, and P3, the
hybrid kernel (step 7). The table below records the surfaces the review
found on 2026-09-09; the closure and hybrid extras remain as documented
aliases, and the pure native kernels of the last column are the
differential tier behind P3, hidden and reachable through
`try_compile_pure_jit*` only.*

What a host held after compiling, by engine. "Same" means the same
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
| Cursors | `cursor_schemas` (with the partitions resolved at build), `set_cursor`; `cursor_over_partitions` for the run-time cases; activations seed their bodies | `cursor_schemas`, `set_cursor` | same | same |
| Shared bindings | cells, write-through, broadcast | same, through the cell bound to the binding's extern slot (step 9) | same | same |
| Failure of a node | panic enriched with node name, outputs, context, and inputs | same | same | same |
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

### A1. Fifty-three nodes run on the interpreter only — functional

*Closed by step 2.* The record below is the state the review found.

Fifty-three nodes had no compiled form: the closure tier and the
hybrid kernel refuse a program that contains one (`hybrid.rs`: "has no
compiled form and can't be JIT-compiled"), and the production kernel
runs them interpreted. The review's first pass found seventeen; the
pinned matrix found the rest. Seven causes, all but one in the node
attribute's kit eligibility (`polydat-derive/src/lib.rs`,
`jit_eligible`, `slot_eligible`, and the handle plan):

- A byte-string argument (`&[u8]`) or return (`Vec<u8>`) has no kit:
  `to_hex`, `from_hex`, `sha256`, `md5`, `to_base64`, `from_base64`,
  `to_base32`, `byte_slice`. The `Bytes` handle exists; the kits do not
  read or write it.
- A setup argument derived from consts (`#[poly_const(..., from =
  spec)]`) is admitted by the handle kit only when the node also has a
  handle shape, and never by the u64 kit. Twenty-four scalar-or-string
  nodes have such a setup: `histribution`, `weighted_strings`,
  `weighted_u64`, `one_of_weighted`, `alias_sample`, `is_stable`,
  `regex_match`, `regex_replace`, `regex_extract`, `pattern_match`,
  `matches`, `combinations`, `char_buf`, `hashed_line_to_string`,
  `char_image_extract`, `byte_image_extract`, `env_or`, and the
  file-backed `csv_row`, `csv_field`, `csv_row_count`, `jsonl_row`,
  `jsonl_field`, `jsonl_row_count`, `file_line_at`.
- A session-static setup (`from = ()`) is refused by every kit:
  `counter`, `elapsed_millis`, `tmp_dir`. `session_start_millis` is the
  same shape but has a native lowering, so it is missing on P2 only
  (A2).
- A `Value` or `Option<u64>` argument matches neither the handle kit nor
  the slot kit: `body_column_i32` (with its `Vec<i32>` return), `pick`
  (variadic `&[bool]` and `&[Value]`), `this_or`, `required`, the
  side-channel nodes `log_debug`, `log_info`, `log_warn`, `log_error`,
  `inspect`, and `fft_analyze`.
- A `Const<Vec<_>>` is a const-list shape the u64 kit does not capture:
  `reg_shuffle_bytes` (with a 128-bit register wire), `fixed_values_u64`,
  `fixed_values_str`, `fixed_values_f64`, `is_one_of`, `one_of`.
- A `Config<_>` wire (`dynamic_weighted_select`) has no kit form.
- `limit` is registered by hand in `library/context.rs` rather than
  through the node attribute, so no kit ever sees it.

True-up: extend the u64 kit to capture setup arguments the way the
handle kit does (recomputed from the captured consts at closure
creation), including the session-static form, which is a construction
capture like a fallible node's cached value; give both kits the `Bytes`
handle through `put_thread_bytes` and `resolve_thread_bytes`; let the
slot kit take a `Value` argument decoded by wire type; and let the u64
kit capture `Const<Vec<_>>` and a `Config<_>` wire; and move `limit`
onto the node attribute. Each is a plan in the derive crate with a
differential in `tests/ext_tiers.rs` and a fuzzer arm.

### A2. Sixteen nodes run compiled only with the JIT — functional

*Closed by step 2.* The record below is the state the review found.

`dist_normal`, `dist_exponential`, `dist_uniform`, `dist_pareto`,
`dist_zipf`, `dist_empirical`, `icd_normal`, `icd_exponential`,
`perlin_1d`, `perlin_2d`, `simplex_2d`, `fractal_noise_1d`,
`fractal_noise_2d`, `coin_flip`, `session_start_millis`, and
`default_or` have a native lowering (`jit_constants` overrides bake the
lookup tables of the first fourteen) but no closure, so a build without
the `jit` feature, or a hybrid kernel on it, runs them interpreted, and
the raw closure kernel refuses them. The cause is the same
setup-argument rule as A1 for all but `default_or`, whose `Value`
arguments are the fourth cause there. True-up: the u64 kit's setup
capture and the slot kit's `Value` argument (A1) cover all sixteen.

### A3. Cursors are an interpreter feature — functional and semantic

*Closed by step 3 for every cursor whose `over` clause is a literal
spec over an extent known at build, and by step 5 for the rest of the
difference: a cursor not yet narrowed is `None` on the closure tier and
in a hybrid kernel as it is on the interpreter. The record below is the
state the review found. What remains: a cursor whose `over` clause is
computed or whose extent is known only at run time is resolved by the
host through `cursor_over_partitions` on an interpreter state, as
before, and pure native code runs such a program only after the host
narrows it with `set_cursor`.*

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

*Closed by step 7: the hybrid kernel is the public P3 (`Engine::Native`,
`try_compile_jit*`, `P3Engine`), and pure native code is the differential
tier behind it, reachable through the hidden `try_compile_pure_jit*`. The
record below is the state the review found.*

Sixty-two programs the closure tier accepts fail on `try_compile_jit*`,
while the hybrid kernel accepts all of them: the nodes with no native
lowering and every vector-bearing node (`Ref2` ports), among them
`json_merge`, `escape_json`, `html_encode`,
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

*Closed by step 8 for the bodies: an activation runs on any engine
through `TraversalStream::activation_on`. Opening a traversal remains
the interpreter's work, so the root kernel and the activation of any
body that itself contains a `for` statement are interpreter kernels;
the innermost bodies, where the cycles run, activate on any engine. The
record below is the state the review found.*

A `for` statement or producer compiles only through `compile_polydat*`,
which builds activation programs and a runtime the host drives with
`traverse`. The assembler entry point fails on either with a message
that had been wrong since SRD 113 step 2 landed ("the `for` construct is
parsed but not compiled yet"; `dsl/compile.rs`, two sites). A compiled
kernel cannot be a traversal's parent or its body.

True-up, in two parts. First, the message (landed in step 1): the
assembler entry point says that a traversal compiles through
`compile_polydat` and runs through `PolydatKernel::traverse`, and points
here. Second, the capability: an activation's body is
a program like any other, so `ActivationStream` should be able to
compile bodies with the engine the host chose and hand back a kernel
with the same drive and read API (A8). This is the largest item in the
plan and depends on A8.

### A6. Evaluation is lazy on the interpreter and eager everywhere else — semantic

*Closed by step 5 for the closure tier and the hybrid kernel; the
record below is the state the review found. What remains: pure native
code evaluates the whole program at a pull, since one native function
is the program (it retires from the public path in step 7).*

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

*Closed by step 6; the record below is the state the review found.*

The interpreter catches a node's panic and re-raises it with the node's
name, its position, and its input values (`kernel/engines.rs`,
`enrich_eval_panic`). A compiled kernel raises the closure's or
helper's panic as it is, and a native kernel raises it through the
longjmp catch with no node attribution at all. The message a host logs
for the same failing program therefore differs by engine, and the
compiled one does not say which node failed.

Worse than a different message: a panic inside a native helper is a
process abort unless the helper body runs under `guarded`
(`jit/codegen.rs`), which converts the panic to the longjmp the kernel
catches. Rust cannot unwind through Cranelift frames, and the `extern
"C"` boundary turns the unwind into an abort. Step 1 hit this while
pinning the matrix: `fractal_noise_1d` at 100 octaves saturated the
lattice coordinate and overflowed `xi + 1` in `library/noise.rs`, which
the interpreter reported as an enriched panic and the native kernel
turned into a crash of the test process. The lattice arithmetic now
wraps and the five noise helpers run under `guarded`; 64 of the 72
helpers still do not.

True-up: every helper body under `guarded`, so a node failure is a
caught failure on every engine, then the attribution below. Every
compiled kernel keeps a step-to-node map (the hybrid
kernel already retains its nodes); the eval loop catches a panic at the
step boundary, on the failure path only, and re-raises it enriched the
same way, with the decoded input values where the slot types allow.
Native segments attribute to the segment's node list.

### A8. The API is spelled differently by engine — API

*Closed by step 4; the record below is the state the review found.*

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

*Closed by steps 1 and 4: the compile log reaches every engine through
`compile_engine_with_log` and `compile_polydat_with_engine`.*

`compile_polydat_to_assembler` had no source directory, no library
paths, no strict flag, and no compile log, so modules from disk and
strict typing were reachable only through the kernel path. True-up
(landed in step 1): `compile_polydat_to_assembler_with(src, &options)`
takes the kernel path's `CompileOptions`, and the kernel path's
`compile_parent` is the same assembly routine the assembler entry point
uses, where before it was a second copy of it. The compile log remains
a kernel-path argument (`compile_polydat_with_log`) until A8's single
constructor takes it.

### A10. `shared` bindings lose their cells off the interpreter — semantic

*Closed by step 9: the cell protocol reaches every compiled kernel
through `compile::externs`, and the `Kernel` trait carries the cells.
The record below is the state the review found.*

A `shared` binding compiles on every engine (probe: register, closure,
hybrid, and native all accept `shared counter := 0`), but only the
interpreter state attaches the cross-fiber cell, commits write-throughs,
and advances broadcasts. On a compiled kernel the binding is an
ordinary input that `set_input` overwrites and nothing publishes.
True-up: either the shared-cell protocol reaches compiled kernels
through `compile::externs` (a cell-bound slot is materialized from the
cell at run start and published at run end), or `shared` is refused at
compile time on engines that cannot honor it. The first is the ideal;
the second is the honest interim and landed in step 2: every compiled
constructor refuses a graph with a `shared` binding, naming the
bindings and this item (`assembly::shared_binding_refusal`).

### A11. Vectors have no native form — functional, by design

*Closed by step 7 with A4: no host-visible engine refuses a vector node.*

Every `Ref2` port keeps a node off pure native code
(`build_jit_layout`: "pure-P3 kernels carry no reference slots"). The
hybrid kernel runs vector nodes as closures and reads them with
`read_vec_*`. Under A4 this stops being host-visible; it remains a
limit of the pure tier and belongs in the SIMD and register documents.

### A12. An unset extern is a `None` on the interpreter and a refusal elsewhere — semantic

*Closed by step 5 for the closure tier and the hybrid kernel, which
carry a `None` mask per slot and propagate it as SRD-74 Rule 1 says;
the compile log names every extern without a default
(`CompileEvent::ExternWithoutDefault`). Pure native code cannot carry
`None` and still refuses to run with an unset table-kind extern.*

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

*Closed by step 10: every public item is documented, `missing_docs` is
enforced by the rustdoc and clippy gates, the node reference carries the
generated matrix, and the guides say what the engines do. The record
below is the state the review found.*

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

*Closed by step 7: re-recorded on 2026-09-10 against the ladder with the
hybrid kernel as P3 and pure native code as a fourth rung.*

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

   *Landed 2026-09-09.* The coverage cases moved to
   `tests/common/coverage_cases.rs`, shared by `function_coverage` and
   the new `tests/engine_parity.rs`, whose table `tests/engine_parity.txt`
   is the matrix in §2 (regenerate with `ENGINE_PARITY=overwrite`, list
   every non-`ok` cell with `ENGINE_PARITY_TRACE=1 -- --nocapture`). The
   matrix corrected the review's counts: 53 interpreter-only nodes, not
   17 (A1), and 16 JIT-only, not 10 (A2). Pinning also found that five
   coverage programs never ran on any engine (the assertion nodes and
   `pick` were called with arguments that fail their own checks; the
   coverage test only compiled them), that a comparison such as
   `cycle == 0` is a u64 truth value and not the `Bool` wire `pick`
   requires, and the native-helper abort recorded under A7. The
   traversal message and `compile_polydat_to_assembler_with` landed as
   planned; `output_names` was added to the hybrid and pure native
   kernels, which lacked it (A8). Folding the kernel path's assembly
   onto the assembler's (A9) exposed one more difference, now closed:
   the assembler entry point accepted a non-literal `shared`
   initializer as an ordinary binding where the kernel path rejected
   it; both reject it now.

   *Step 2 landed 2026-09-09.* The `compiled_handle` kit became the
   general closure: it takes every node the u64 kit does not, and its
   plan covers setups (recomputed from consts, or cloned from the node
   when session-static), byte-string arguments, const lists, `Config`
   and `Option` wires, split variadics, two-slot carriers, and
   polymorphic returns (`kernel::encode_arg`); the slot kit takes a
   polymorphic argument; `limit` has a closure by hand. Every
   registered node now compiles on the closure tier and in hybrid
   kernels, and the JIT-less build runs the whole library compiled;
   the ten cursor consumers still fail at run (A3). The `&[u8]`
   argument had been refused by a whitespace mismatch in the macro's
   type matching, not by any missing kit. Comparing values across
   engines for every program (the new `the_engines_agree_on_every_node`
   and three fuzzer arms in `tests/handle_tiers.rs`) found six
   defects that step 1's outcome matrix could not, all fixed:
   - the compiled kernels' `get_value` read a register or 128-bit
     output as its first limb, a signed narrow output as unsigned, and a
     vector output as its pointer (`marshal::decode_output`,
     `ScratchBuf::to_value`);
   - the constant fold replaced a Bool-valued init node with a
     `const_u64`, so the interpreter read a U64 on a Bool wire
     (`fold_init_constants`);
   - `blend` lowered as a numeric conversion where the body
     reinterprets bits, `coin_flip` lowered as a hashed coin where the
     body compares the raw input against its threshold (and as a fair
     coin when it had no constant), and `default_or` lowered as a
     three-way select that returned the fallback for any non-zero
     value; a native select between two handles also tripped the
     handle discipline, so `select` over handles stays a closure.
   `shared` is refused on every compiled engine (A10).

   *Step 3 landed 2026-09-09.* The compiler resolves a cursor's `over`
   clause at build when the clause is a literal spec and the extent is
   known (`SourceSchema::partitions`), and seeds the cursor's `Ext`
   slot and six scalar projections when the clause denotes exactly one
   partition, so such a program runs on every engine with no host
   call; the ten partition consumers in the matrix now run on the
   closure tier and in hybrid kernels. The cursors reach every kernel
   through the assembler (`PolydatAssembler::set_cursor_schemas`) and
   the extern plumbing: every compiled kernel and the interpreter
   kernel offer `cursor_schemas` and `set_cursor(name, &Partition)`,
   which writes the same seven inputs `narrow_cursor` writes.
   `cursor_over_partitions` returns the resolved list without a pull.
   `tests/cursor_tiers.rs` is the differential over the partition
   family with narrowed cursors, and the fuzzer declares seeded
   cursors. A clause that denotes several partitions is still unset
   until the host or the traversal runtime narrows it; on the
   interpreter that reads as `None`, on a compiled kernel as a refusal
   to run (A12, step 5).

   *Step 4 landed 2026-09-10.* The `Kernel` trait (`kernel::api`,
   re-exported at the crate root) is implemented by the interpreter
   kernel and all eleven compiled kernel types: `engine`, `set_inputs`,
   `set_input`, `set_cursor`, `eval`, `pull` (owned, never a handle),
   `input_names`, `output_names`, `output_type`, `externs`,
   `cursor_schemas`, and `into_program`. `Engine` names the interpreter,
   the closure tier, the hybrid kernel, or native code, each with a
   `Provenance` (`Raw`, `Push`, `Pull`, `PushPull`, or `Auto` for the
   selector's choice); `PolydatAssembler::compile_with(Engine)` and
   `compile_polydat_with(src, Engine)` build a `Box<dyn Kernel>` or a
   `KernelError` (`Source`, `Assembly`, or `Refused { engine, reason }`),
   and `compile_engine_with_log` and `compile_polydat_with_engine` take
   the options and the compile log for every engine (A9). A shared
   program is `into_program`, an `Arc<dyn KernelProgram>` whose
   `create_kernel` gives each thread its own kernel: the interpreter's
   program as before, and for a compiled kernel a clone that shares the
   steps, the nodes, and the native code (now `Arc`-held) and owns its
   buffer, table, scratch, and externs. The older constructors remain as
   documented aliases; the raw slot readers, `eval(&[u64])`, and
   `engine_counts` remain as engine-specific extras. On the way the
   closure tier's refusal gained a reason (`build_p2_layout` returns it),
   and the hybrid kernel's raw and pull-only forms, which nothing had
   constructed, are reachable as `Engine::Hybrid(Raw)` and
   `Engine::Hybrid(Pull)`. `tests/kernel_api.rs` drives every engine
   through the trait against the interpreter, shares each across
   threads, and checks the error type; the embedding guide's compiled
   kernels section is written to the trait.

   *Step 5 landed 2026-09-10.* The closure tier and the hybrid kernel
   evaluate under the runtime model's one rule, the classification the
   interpreter's fold itself calls (`classify_lifecycle`): a step is
   current until an input in its provenance changes, whichever call
   changed it, a coordinate through `set_inputs` or an extern or cursor
   through `set_input`; a nondeterministic step, or one downstream of
   one, is never current, and is invalidated at every cycle as the
   interpreter invalidates it at every `set_inputs`; a handle-writing
   step runs every cycle (SRD 115 §4); a compile-constant step, one no
   input reaches, is folded at build, once, on every engine, which is
   the interpreter's fold at the same moment, so what is knowable at
   build is known at build and fails at build; and everything else
   runs at first pull. `set_inputs` (or a changed extern or cursor)
   opens a cycle, `pull(name)` runs the steps of that output's cone
   that are not current and have not run in the cycle, and `eval` runs
   every step. A provenance mode is an optimization on that rule and
   never a change to it: a mode without per-step skipping may recompute
   a pure step redundantly, which nothing observes, but a side-channel
   step is skipped when current in every mode, since its run is
   observed. The bookkeeping is one data structure,
   `compile::Invalidation`: per input slot the steps it invalidates,
   per named output the steps it needs. The evaluation loops consume
   only that, and provenance derives it today; a host that knows its
   write and read patterns may supply a narrower plan later (explicit
   dirty registers instead of cone invalidation) without touching the
   loops. A per-slot `None` mask carries SRD-74 through the compiled
   kernels: an unset extern is `None`, a step whose node does not
   accept `None` emits `None` on every output without running, and
   `pull` returns `None` for such a slot; a node downstream of an
   extern without a default runs as a closure in a hybrid kernel, since
   native code cannot carry `None`. The handle validators (H4, S9)
   check only the slots the cycle wrote. The compile log names every
   extern without a default. The fuzzer drives every engine through
   `Kernel::pull` output by output and counts the rows a side channel
   emits per engine per cycle, with an `emit_row` arm; it caught, in
   turn, a plan that took every input slot as a coordinate, a
   construction-time flattening of extern-dependent steps that the
   interpreter does not do (a semantic difference by engine, removed),
   a raw mode that re-fired a memoized side channel, and a nullary
   nondeterministic node (`tmp_dir`) that the compiled engines classed
   volatile while the fold classed it compile-constant, which is why
   the classification is now one function shared with the fold. Pure native
   kernels still evaluate the whole program per pull (A6).
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

   *Step 6 landed 2026-09-10.* A node that fails at evaluation fails
   with one message on every engine. The enrichment is one function,
   `kernel::engines::enrich_panic`, which the interpreter's `eval_node`
   and every compiled kernel call: the original payload, the location
   the capture guard recorded, the node's name, the outputs it feeds,
   the program's context, and its input values. Each compiled kernel
   carries a `compile::Attribution`, built by the assembler from the
   resolved graph (step index is node index on every engine): per
   node its name, its outputs, and `(first slot, port type)` per
   input, so the failure path decodes the inputs from the buffer as
   `get_value` decodes an output (`None` through the mask, a vector by
   its type). The closure tier and the hybrid kernel catch at the step
   boundary with the step index in hand. Pure native code, being one
   function, names the step it is in by storing the index to a slot
   past the layout before each helper call, the one way native code
   fails; the store is removed again from a step of inline arithmetic,
   so a step that cannot fail pays nothing. A cone (SRD-105) does the
   same for its members and re-raises attributed to the member, and
   the interpreter's own enrichment then names the cone, so a failure
   inside a fused node reads member first, cone second. Every native
   helper runs under `guarded` now, except the three predicate-fail
   helpers that are the longjmp themselves, so a helper's panic is the
   longjmp the kernel catches and never a process abort; the longjmp
   wrapper re-raises with `resume_unwind`, so the hook does not record
   its own location over the helper's. On the way the native string
   parsers were found to differ from the library's adapters in both
   message and, for booleans, in the spellings accepted (`t`, `yes`,
   `y`, and anything else read as false); they call the adapters'
   parse functions now, so the diagnostic is the library's on every
   engine. `tests/failure_parity.rs` drives a predicate violation and
   a string coercion failure through the thirteen engine-and-mode
   combinations and requires the interpreter's message, the `panicked
   at` line aside, since that line names the engine's own code; the
   agreement test compares failure messages the same way for every
   coverage program that fails. Cost: a `catch_unwind` frame per
   helper call and per run of a step loop; the ladder is re-recorded
   at step 7 (A14).
7. **Hybrid becomes P3** (A4, A11): the pure native kernels retreat to
   the differential suites; `try_compile_jit*` and `P3Engine` become
   hybrid. The performance guide is re-recorded (A14) against the new
   ladder.

   *Step 7 landed 2026-09-10.* `Engine::Native`, `try_compile_jit*`,
   `P3Engine`, and `auto_compile_p3` build the hybrid kernel: native
   code for every node that has a lowering and the node's closure
   elsewhere, so P3 accepts every program the closure tier accepts
   (A4, A11). `Engine::Hybrid` is gone, being the same thing. The pure
   native kernels are the differential tier behind P3, `#[doc(hidden)]`
   and reachable only through `try_compile_pure_jit*`; the matrix's
   columns are P1, P2, P3, and `pure`, and the differential suites
   (`handle_tiers`, `ext_tiers`, `slot_state_axioms`, `value_table`,
   `variadic_lowering`, the register and SIMD lowerings) drive the pure
   tier by that name. On the way the ladder benchmark showed the
   hybrid kernel no faster than the interpreter and the closure tier at
   twice its step 4 time, both of which the design forbids: an engine
   below another on the ladder must be faster, and lazy evaluation is
   an optimization over eager evaluation, never a cost. Three causes,
   three fixes, each an optimization over the runtime model's rule
   with the same result. The hybrid kernel compiled every native node
   as its own segment behind its own setjmp; it now compiles each run
   of consecutive native-eligible nodes of one lifecycle as one segment
   (a compile-constant node never joins a segment that is not, or a
   constant step downstream of it would run at build before its
   producer, which the fuzzer caught), and a segment names the member
   it is in through the tracker slot, so the attribution of step 6
   still reads per node. The closure tier's and the hybrid's evaluation
   loops paid per-step bookkeeping on every step of every cycle; a
   fresh cycle in a mode without per-step skipping, with no `None` in
   play, now runs every step straight through (`run_fresh`), a step's
   cycle number replaces a `ran` flag that had to be cleared, the raw
   and pull-only modes dirty only the side channels an input reaches
   since nothing else consults a pure step's currency, the value table
   is installed in place instead of moved out and back, and a
   passthrough (`identity`, `__port_`) is an inline slot copy rather
   than a closure call. And the step runners were not being inlined:
   `#[inline(always)]` on `run_step` and `run_hybrid_step` alone
   returned a quarter of the closure tier's time. The engine ladder,
   re-recorded in the performance guide (A14), now reads interpreter,
   closures, P3, and pure native code in that order, with P3 equal to
   pure native code on a graph where every node lowers.
8. **Traversals on compiled engines** (A5, second part): activation
   bodies compiled with the host's engine choice behind the `Kernel`
   trait.

   *Step 8 landed 2026-09-10.* A traversal's activations run on any
   engine. The parent compiles each body once for the interpreter, as
   before, and now also keeps the body as it lowered it, the child file
   with the compiler settings (`dsl::traversal::BodySource`);
   `Traversal::program_on(engine)` compiles that body through the
   assembler on the first request for an engine and caches the
   `KernelProgram`, so a body is one program per engine as it is one
   program per position (SRD 113 §5.1), and
   `TraversalStream::activation_on(index, engine)` is a fresh nested
   kernel from that program, driven through the `Kernel` trait: the
   elements and the cascade bind by name through `set_input`, the
   cursors narrow through one routine for every engine
   (`cursor_over_partitions_on`, `cursor_extent_on`, `set_cursor`),
   which the interpreter's activation now uses too, and `cycle(i)`
   sets the coordinate and the cursor ordinal as before. `Activation`
   is generic over its kernel, the interpreter's by default, so every
   host of the old form is unchanged. A nested kernel is a trait
   matter now: `KernelProgram::create_nested_kernel` and
   `Kernel::nest` mark a kernel as running inside the cycle of the
   kernel that opened it (SRD 115 §4), on every engine. What stays on
   the interpreter is opening a traversal, since the comprehension's
   sources evaluate against the kernel that opens it. So the root kernel
   is an interpreter kernel, and so is the activation of any body that
   itself contains a `for` statement or producer binding, because that
   activation is what opens the nested traversal; the other engines
   refuse such a body by name. Only a body with no `for` of its own, the
   innermost body of a nest, where the cycles run, activates on another
   engine. The `polydat` binary still activates on the interpreter.
   `tests/for_engines.rs` traces every activation and cycle of a sweep
   and of a partition-sliced cursor body on every engine against the
   interpreter's, checks one program per engine with the build counter
   flat across activations, and checks the nested-body refusal and
   that inner activations of an interpreter activation run anywhere.
9. **Shared cells on compiled engines** (A10) through `compile::externs`
   materialization and publication.

   *Step 9 landed 2026-09-10.* A `shared` binding runs on every engine
   under the interpreter's cell protocol. The binding was already an
   extern slot on the compiled kernels; `compile::externs` now binds
   that slot to a `SharedCell`, the same type the interpreter attaches
   (`kernel::engines::SharedCellInner`, with the scope's intent word
   and bit): `set` publishes through the cell, every run's
   materialization and every pull inside a cycle take the cell's value
   where its revision moved and mark the slot's dependents through the
   plan, so a write by any holder of the cell is what the others read
   next, as the interpreter's revision-aware cone check gives it, and a
   write outside every kernel, through the cell itself, reaches them
   all the same way. Pure native code, which evaluates the program per
   pull, treats a moved revision as a changed input. The `Kernel` trait
   carries the protocol: `shared_cells` lists a kernel's cells and
   `attach_shared_cell` binds one kernel's cell into another, on the
   interpreter and the compiled engines alike, and a kernel created
   from a shared program starts with cells of its own, as an
   interpreter state does. The refusal step 2 landed
   (`shared_binding_refusal`) is gone. `tests/shared_tiers.rs` drives
   the default, the write-through, two attached kernels writing and
   reading one register inside open cycles, a publication from outside,
   program-created kernels' independence, and type stability on every
   engine and provenance mode.
10. **Documentation to zero** (A13): `missing_docs` burn-down under CI,
    the generated matrix in the node reference, the guide and the
    engines document updated to state that every engine accepts every
    program.

    *Step 10 landed 2026-09-10.* The crate declares
    `#![warn(missing_docs)]`, and since CI runs rustdoc and clippy with
    warnings as errors on every feature set, an undocumented public item
    fails the build. The count went from 2,251 to zero in two moves.
    The node macro emits every node's struct, constructor, and constant
    argument fields; it now forwards the node function's own doc
    comments onto the struct, names the node where the function has
    none, and describes each constructor and field, which took the
    count to 1,123 without a hand-written line. The rest is
    hand-written: the value and port types, the DSL and comprehension
    ASTs, the compile events and errors, the engine selectors and the
    kernel types, the SIMD tiers, the cursor and traversal types, the
    scope-composition contracts, and every library type that `pub`
    exposes. The node-by-engine matrix is generated by the parity suite
    into the node reference (`docs/reference/nodes.md`, between
    markers), compared on every run and rewritten with
    `ENGINE_PARITY=overwrite`, so the documented feature set and the
    tested one are one file: today it states that every one of the 296
    coverage programs runs on the interpreter, the closure tier, and P3,
    and names the 127 nodes pure native code has no lowering for, which
    P3 runs as closure steps. The engines document's §7 now describes
    placements within an engine rather than refusals, and the
    compilation guide's "same shape at each level" claim is the `Kernel`
    trait. With this step the plan's ten steps have landed.

Steps 1 through 3 are contained in the derive crate, the assembler, and
the extern plumbing that landed on 2026-09-09, and change no public
signature (step 1 adds two: the options-taking assembler entry point and
`output_names` on the hybrid and pure native kernels). Step 4 is the API change and should land as one release.
Steps 5 through 7 change semantics and belong behind the equivalence
harness that step 5 extends. Steps 8 and 9 are new capability.

## 6. What does not change

The interpreter remains the oracle. No step coerces a type, moves a
node into an engine whose caching would suppress an observation, or
changes a public port type or named output. The determinism axioms of
[Runtime Model](runtime_model.md) hold on every engine before and after
this plan; what the plan adds is that a host no longer has to know which
engine it is on.
