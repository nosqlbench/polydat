---
type: specification
title: JIT Boundary
timestamp: 2026-09-25
description: "The native call boundary: function signatures, predicate failures and their recovery, the invoke_with_catch contract, and invalidation across the boundary."
tags: [native, engines]
---

# JIT Boundary

This document specifies polydat's internal contract for native code: how
the machine code Cranelift generates is called from Rust, what happens
when that code fails, how a failure is reported, how invalidation and
extern resolution cross the boundary, and the rules for the slot buffer
and scratch that native code and Rust share (the slot-state axioms
S1–S10). Native code runs on the native tier (P3), on pure
native, and inside the interpreter's native cones; the closure tier runs
no native code.

**Related specifications:** [Engines](engines.md) and
[graph_compiler.md §6 (ordered composition)](graph_compiler.md) (which
engine runs a program and which of its nodes run natively);
[runtime_model.md](runtime_model.md) (the R-axioms: the clean-flag and
memoization model native code preserves across the boundary).

---

## Call boundary overview

Native code compiles to a function over the slot buffer and the
evaluating kernel's scratch: the storage entries into which steps
write by-reference values such as strings and vectors
(compiled_handles.md §3, §6). The provenance variant also takes the
steps' clean flags:

```text
fn(coords: *const u64, buffer: *mut u64,
   scratch: *mut ScratchBuf)                    // raw
fn(coords: *const u64, buffer: *mut u64,
   scratch: *mut ScratchBuf, clean: *mut u8)    // provenance variant
```

The Rust side owns the buffer and the scratch and calls the function
pointer. Native code runs in three places (the table below). Each of
them calls through `codegen::invoke_with_catch` when the code can
fail, which is when it calls a helper (a Rust function the generated
code calls); `JitCode::fallible` records this, decided at finalization
by whether the function contains a call instruction. Code with no
call is arithmetic over the buffer and cannot fail, so the call site
runs it directly, without the jump buffer, the panic capture, or the
unwind guard. Those make up most of a small evaluation's cost when
the generated code is a few arithmetic instructions.

| Site | What runs natively | Where |
|---|---|---|
| An embedded cone | A fused subgraph of an interpreter kernel, one native function per cone, over a slot buffer and the members' scratch entries, which the evaluating state owns as the cone node's scratch | `compile/cone.rs`, the cone node's `eval_in` |
| A segment of the P3 kernel | A fusion unit: a connected, convex group of native-eligible nodes of one lifecycle and one volatility, with a side channel always a segment by itself, one native function per segment, all of a kernel's segments in one compiled module, over the kernel's own buffer and scratch; the nodes outside segments run as closure steps | `compile/fusion_units.rs`, `compile/hybrid.rs`, the step runner |
| The pure native tier | The whole program as one native function over the kernel's buffer and scratch, built of the same fusion units, one block each, and entered with a cone order of units and their clean flags, which the code tests itself, running the stale units and marking each current as it ends | `compile/jit/kernels.rs`, `JitCore::run_units` |

The pure native tier is the differential reference for native
lowering and the carrier of the Tier-1 register kernel
([SIMD ISA Selection](simd_isa_autopromotion.md)); a host ordinarily
runs native code through the P3 kernel. No direct `(code_fn)(...)`
invocation exists outside the three sites; a search for that pattern
finds only them, each inside the wrapper.

---

## Why predicate violations are a problem at this boundary

The library's predicates (`is_positive`, `in_range`, `is_one_of`;
[Library Catalog](library_catalog.md)) lower natively and can fail
during evaluation. When one fails, the native code calls an extern
helper (`jit_is_positive_fail`, `jit_in_range_fail`,
`jit_is_one_of_fail`) that must report the violation and stop the
current evaluation.

Panicking from the extern helper and catching the panic in the Rust
caller does not work with Cranelift-generated frames:

- Cranelift emits DWARF `.eh_frame` entries when
  `unwind_info=true`, and registers them with the platform
  unwinder via `JITModule::finalize_definitions()`.
- Those frames do **not** carry Rust's panic personality
  routine. The libunwind walker can traverse them, but
  `_Unwind_RaiseException` never finds a catch block and the
  panic runtime aborts with "failed to initiate panic, error 5
  (`_URC_END_OF_STACK`)".
- Declaring the extern `extern "C-unwind"` does not fix this,
  because the problem is the missing personality routine, not the
  ABI.

Making Cranelift emit `.gcc_except_table` entries that reference
`rust_eh_personality`, and re-registering frames through a
personality shim, is outside this crate's scope.

---

## The setjmp / longjmp workaround

A native failure does not unwind *through* the JIT frame; it jumps
*past* it. Before calling native code, the Rust wrapper records a jump
target with `_setjmp`; a failing helper stores its message and calls
`_longjmp` to that target, and the wrapper then raises an ordinary Rust
panic with the message.

### Flow

```text
        ┌─────────── Rust caller (eval) ───────────┐
        │                                          │
        │  invoke_with_catch(|| {                  │
        │      _setjmp(&jmp_buf)         ← record  │
        │      (code_fn)(buf, mut_buf)   ← JIT ───┐│
        │  })                            return   ││
        │                                          │
        └──────────────────────────────────────────┘
                                                  ││
                                              JIT frame (machine code,
                                               no Rust personality)
                                                  ││
                                                  ▼▼
                                         extern "C" fn
                                         jit_is_positive_fail(v)
                                            │
                                            ├─ stash message in TLS
                                            └─ _longjmp(&jmp_buf, 1)
                                                  ▲
                                            control jumps HERE
                                                  │
        ┌─────────── Rust caller (eval) ───────────┐
        │  _setjmp returned non-zero:              │
        │  read the stashed message                │
        │  resume_unwind(message) ← Rust-land panic│
        └──────────────────────────────────────────┘
                     │
          ordinary Rust unwind, proper personality
                     ▼
          std::panic::catch_unwind in the caller
```

The longjmp skips the JIT frame entirely — no unwinding through
it, no personality lookup, no catch-block walk. Control returns
into Rust land where a normal panic propagates through
Rust-personality FDEs the way any other panic would.

### Safety

- **Resources in the JIT frame.** Cranelift-generated code is
  pure machine code with no Drop obligations, no heap-owning
  values, no locks to release. Skipping it is safe.
- **Resources in the extern fail functions.** Each helper
  only `format!`s a message, stashes it in a TLS slot, and
  calls `_longjmp`. The `String` allocated by `format!` is
  moved into the TLS slot *before* the longjmp, so nothing is
  dropped implicitly across the jump.
- **Thread locality.** The jmp_buf pointer and the message
  slot are both `thread_local!`. Concurrent kernels on
  different worker threads don't share state.
- **Nesting.** `invoke_with_catch` saves the outer
  `JIT_JMP_BUF` slot in a stack-local variable, installs its
  own buffer, and restores the outer on every exit path. An
  inner longjmp jumps to the innermost buffer; the outer
  regains the slot when the inner frame unwinds. A projection
  body's native kernel rendering inside another kernel's render
  step is such a nesting.
- **SIMD register state.** `_setjmp` on glibc preserves only
  the core register set that `longjmp` restores. The Rust
  wrapper keeps no live SIMD state across the JIT call, so
  nothing is lost. Live SIMD data held across a predicate
  violation is not preserved by this mechanism.

### Platform-portable jmp_buf shim

`libc` does not expose `jmp_buf`, `setjmp`, or `longjmp`, which are
generally considered unsafe to call from Rust, so the crate declares
them directly:

```rust
#[repr(C, align(16))]
struct JitJmpBuf([u8; 512]);          // 512 > glibc (~200) > macOS (~192)

#[cfg(not(windows))]
unsafe extern "C" {
    fn _setjmp(env: *mut JitJmpBuf) -> i32;
    fn _longjmp(env: *mut JitJmpBuf, val: i32) -> !;
}

#[cfg(windows)]
unsafe extern "C" {
    fn _setjmp(env: *mut JitJmpBuf, frame: *mut std::ffi::c_void) -> i32;
    #[link_name = "longjmp"]
    fn _longjmp(env: *mut JitJmpBuf, val: i32) -> !;
}
```

On glibc and macOS the crate links against `_setjmp` and `_longjmp`
rather than plain `setjmp` and `longjmp`, because the plain variants
are glibc macros that expand to `__sigsetjmp(env, 0)`, which saves the
signal mask. The wrapper does not need the signal mask; `_setjmp`
saves registers only and is faster.

The MSVC CRT names the pair differently: it exports `longjmp` (there
is no `_longjmp`), and its x64 `_setjmp` takes a second argument
recorded as the jmp_buf's `Frame` field, which a C compiler fills in
by intrinsic. The wrapper passes NULL explicitly, for two reasons. It
keeps the second argument register from writing garbage into the
buffer, and a zero `Frame` makes `longjmp` do a plain register
restore instead of an `RtlUnwindEx` unwind. The plain restore is
required, because the frames being skipped are JIT code with no
unwind tables registered, which is the problem this path avoids.

---

## `invoke_with_catch` contract

```rust
pub(crate) fn invoke_with_catch<F: FnOnce()>(f: F)
```

- Installs a stack-local jmp_buf into the thread-local
  `JIT_JMP_BUF` slot.
- Runs `f()`.
- If `f()` returns normally, a [`JmpBufGuard`] restores the
  outer slot on drop.
- If `f()` triggers a native failure, the extern helper
  `_longjmp`s back; the wrapper reads the TLS message and
  re-raises it with `std::panic::resume_unwind`, not `panic!`:
  the panic hook already saw the original panic under `guarded`
  (below) and recorded its location for the enrichment the kernel
  adds, and a second hook call would overwrite it. The guard still
  runs (on the unwind path inside the wrapper frame) and restores
  the outer slot.
- If `f()` panics for a non-JIT reason (a bug in a non-JIT
  sub-path; a panic from a closure step in the P3 kernel),
  the panic unwinds through the wrapper's frame. The guard's
  `Drop` restores the outer slot before the unwind
  continues. Subsequent `invoke_with_catch` calls see a clean
  sentinel.

### RAII guard

```rust
struct JmpBufGuard { prev: Option<*mut JitJmpBuf> }

impl Drop for JmpBufGuard {
    fn drop(&mut self) {
        JIT_JMP_BUF.with(|b| b.set(self.prev));
    }
}
```

The guard is the only thing that writes the previous slot
back. Every exit path from `invoke_with_catch` — return,
setjmp-return-then-unwind, or panic-through-wrapper — runs
through `Drop`.

### `guarded`: helpers whose body may panic

A predicate helper fails on purpose and calls
`jit_violation_longjmp` itself. A helper whose body may panic as its
interpreter node panics (`printf` with a placeholder and no argument,
a tile whose projection body fails at render) cannot let that panic
unwind out of an `extern "C"` frame, where it would abort. Such a
helper runs its body under `guarded`:

```rust
fn guarded<T>(body: impl FnOnce() -> T) -> T
```

which runs the body under `catch_unwind`, extracts the payload's
message, and re-raises it through `jit_violation_longjmp`. The panic
hook fires once, inside `guarded`, and records the location; the
wrapper's `resume_unwind` then propagates the message without a
second hook call. `guarded` is part of the ABI: a helper that can
panic and does not use it is a defect.

---

## No-sentinel fallback

`jit_violation_longjmp` checks the thread-local for an installed
jmp_buf before attempting to jump:

```rust
fn jit_violation_longjmp(msg: String) -> ! {
    JIT_VIOLATION_MSG.with(|m| *m.borrow_mut() = Some(msg.clone()));
    match JIT_JMP_BUF.with(|b| b.get()) {
        Some(ptr) => unsafe { _longjmp(ptr, 1) },
        None => {
            // Native code invoked outside a wrapper.
            eprintln!("{msg}");
            std::process::abort();
        }
    }
}
```

This fallback is the last resort. It runs only when a native
function pointer is called without going through one of the three
wrapped sites. It prints the message and aborts the process, since
no Rust caller has installed a jump target to catch the failure.

---

## The failure contract

A failure in native code produces exactly the message the same
failure produces on the interpreter, so the message is the same on
all four engines ([Engines](engines.md) §3.4 states the contract;
this section specifies its native part).

- **The tracker slot.** Every native function has one slot past the
  layout, the tracker. Before a step that calls a helper, generated
  code stores the step's index there; a step of inline arithmetic
  cannot fail and has no store (codegen removes the store when the
  step emitted no call). The runner sets the tracker to `u64::MAX`
  before each run, so a failure before any store names no step.
- **Attribution.** The runner arms an `EvalPanicCaptureGuard` for the
  run, so the panic the hook sees under `guarded` or at a predicate
  helper is recorded quietly rather than printed. When the wrapped
  call unwinds, the runner reads the tracker, maps the step to the
  program node it belongs to (a cone or segment keeps its members in
  step order), decodes that node's input slots by their port types,
  and re-raises through `enrich_panic` with the original payload, the
  recorded location, the node's name, every output it feeds, the
  program's diagnostic context, and the formatted inputs. The
  interpreter's re-raise builds the same message from its `Value`s.
- **The P3 kernel** does the same for a closure step, and for a
  segment reports the member whose index native code stored in the
  tracker.

---

## Extern-helper table

Each predicate has one dedicated fail helper. The helpers live
in `compile/jit/codegen.rs` and are registered with Cranelift's JIT
symbol table so the emitted native code can call them.

| Extern | Arity | Called from |
|---|---|---|
| `jit_is_positive_fail` | `(u64, ptr, len) -> u64` (value, control name) | `JitOp::IsPositiveCheck` |
| `jit_in_range_fail` | `(u64, u64, u64) -> u64` (value, lo, hi) | `JitOp::InRangeCheck` |
| `jit_is_one_of_fail` | `(u64, ptr, len) -> u64` (value, allowed set) | `JitOp::IsOneOfCheck` |
| `jit_div_zero_fail` | `(kind) -> u64` (0 a quotient, 1 a remainder); the words Rust's `/` and `%` fail with | `JitOp::U64DivWire`, `JitOp::U64ModWire`, `JitOp::DivConst(0)`, `JitOp::ModConst(0)` |

The `u64` return type matches the extern-function ABI the JIT
uses; since each helper ends in `_longjmp` (which is `-> !`),
the return is unreachable.

### Bit-exactness

A named lowering computes the bits the node's body computes, not a
value near them. Where Cranelift's instruction differs from the Rust
operation the body uses, the lowering spells the body's operation out
or calls it: `f64::round` rounds half away from zero, so `round_to_u64`,
`round_u64`, and `quantize` truncate and step by a signed one where the
fraction reaches a half rather than using `nearest`, which rounds to
even; `f64::clamp` keeps a negative zero and a NaN, so `clamp_f64`,
`inv_lerp`, and `discretize` select on two comparisons rather than
using `fmax`/`fmin`, which return the bound's zero; a division in the
body is a division in the code (`remap`), never a product with a
reciprocal, which differs in the last bit; `div_ceil` is the quotient
plus one on a nonzero remainder, never `(v + m - 1) / m`, whose sum
overflows near the top (`ceil_to_multiple`, `multiples_at_least`);
`f64_mod` calls the body itself, since Rust's `%` on floats has no
Cranelift equivalent; and `div_wire`, `mod_wire`, `div`, and `mod` fail
on a zero divisor in the body's words, where `u64_div` and `u64_mod`,
whose bodies check, yield zero. The corner suite verifies these
cases.

### The slot-call helper

Every node with no named lowering and a kit runs its kit from native
code through one helper ([Compiled By-Reference
Slots](compiled_handles.md) §6):

| Extern | Arity | Called from |
|---|---|---|
| `jit_slot_call` | `(kit, inputs ptr, n_in, outputs ptr, n_out, scratch ptr, base, n_scratch)`, no return | `JitOp::SlotCall` |
| `jit_u64_to_str`, `jit_i64_to_str`, `jit_f64_to_str` | `(scratch ptr, base, buffer ptr, out slot, bits)`, no return; the digits into the step's entry, the pair published | `JitOp::U64ToStr` and siblings |
| `jit_str_concat` | `(scratch ptr, base, buffer ptr, out slot, pairs ptr, n)`, no return; every pair's bytes appended into the entry | `JitOp::StrConcat` |
| `jit_json_to_str` | `(scratch ptr, base, buffer ptr, out slot, ptr, len)`, no return; the compact serialization into the entry | `JitOp::JsonToStr` |
| `jit_vec_add`, `jit_vec_scale`, `jit_vec_norm`, `jit_hash_vec`, `jit_xxhash3_vec`, `jit_reg_to_vec_f32` | `(scratch ptr, base, buffer ptr, out slot, w0, w1, w2, w3)`, no return; the input words in order, zero past the last; the vector into the step's `F32` entry, the pair published | `JitOp::VecProduce` |
| `jit_vec_dot`, `jit_vec_l2`, `jit_vec_cosine`, `jit_lid_mle` | `(w0, w1, w2, w3) -> bits` of the `f64` result | `JitOp::VecReduce` |
| `jit_reg_lane_f32`, `jit_reg_lane_i16`, `jit_reg_lane_i64` | `(lo, hi, i) -> word`; the lane as its port stores it, the bound checked | `JitOp::RegLane` |
| `jit_reg_with_lane_f32`, `jit_reg_gather_f32`, `jit_vec_to_reg_f32`, `jit_reg_mul_i8` | `(buffer ptr, out slot, w0, w1, w2, w3)`, no return; the word into the two output slots | `JitOp::RegProduce` |
| `jit_f64_mod` | `(bits, bits) -> bits`; the body's `%` | `JitOp::F64Mod` |

The generated code stores the step's input slots into a frame array,
calls the helper with the kit's address (an immediate; the kit is
shared by every kernel compiled from the program and kept alive
beside the code), the frame, and the state's scratch with the index of
the step's first entry, then loads the outputs from the frame into
their slots. The helper runs the kit's closure under the same panic
guard as every other helper, so a node's failure surfaces as the
interpreter surfaces it (A7). A `Ref2` pair enters and leaves the
helper as two slots; only the kit dereferences it (S7).

The message formatting happens at the Rust side, inside the
helper:

```rust
extern "C" fn jit_in_range_fail(value: u64, lo: u64, hi: u64) -> u64 {
    jit_violation_longjmp(
        format!("in_range: value {value} outside [{lo}, {hi}]"),
    );
}
```

---

## Operator-visible semantics

To a caller, a predicate violation in native code behaves exactly
like a predicate violation on the interpreter or the closure tier:

- `#[should_panic(expected = "must be > 0")]` on the caller
  works.
- `std::panic::catch_unwind` catches the panic and returns `Err`.
- The panic message contains the violating value (and, for
  `in_range`, the configured bounds), enriched with the node and its
  inputs as above.
- The kernel remains usable after the panic is caught. The failing
  step leaves the slot buffer partially written, and later
  evaluations overwrite it.

The helper ABI passes the `is_positive` control name and the
`is_one_of` allowed set as stable pointers into node metadata, and
passes `in_range`'s numeric bounds as arguments. These values remain
valid for the compiled kernel lifetime because the native core and the
cone node retain the originating nodes.

---

## Unwind boundary constraint

Native failures cross generated frames through the documented
setjmp/longjmp trampoline. Generated code MUST NOT allow a Rust
panic to unwind through a Cranelift frame because the emitted
object does not register a compatible Rust unwind personality for
that path. The TLS jump-buffer guard, `guarded`, and the Rust-side
catch boundary are therefore part of the ABI, not an optional
implementation detail.

---

## SIMD compute kernels

`compile/jit/simd.rs` compiles four f32-lane kernels once per
process through the same Cranelift engine, using Cranelift's SIMD
types (`F32X4`): `dot_f32`, `l2sq_f32`, `add_f32`, and
`scale_f32`. Each processes the body of a slice in 128-bit chunks,
with unaligned loads because `SliceArc<f32>` data is only
4-byte-aligned, followed by a scalar loop for the remaining
elements; the reducing kernels finish with an `extractlane`
horizontal sum. The `vec_*` nodes in
`polydat-nodes/src/vector_math.rs` use these kernels, and fall back
to scalar Rust loops when the `jit` feature is off or building for
the host ISA fails. The kernels are also usable through the slot
ABI ([Type-System Alignment](type_system_alignment.md) §6): typed
slice values pass between compiled steps as `(ptr, len)` slot pairs,
and `CompiledSlotOp` publishes vector results into kernel-owned
scratch. Every native function, scalar or not, takes the state's
scratch beside its buffer (`fn(coords, buffer, scratch)`); a
slice-bearing step runs its kit through `jit_slot_call` or a named
helper that publishes into the step's entry, under the slot-state
axioms below.

SIMD accumulation reassociates float addition, so reduced results
may differ from the scalar reference in the final ulps; the
equivalence tests compare with relative tolerance.

## Scalar buffer conventions

Every scalar value is stored in one `u64` slot, or two for 128-bit
values, as follows:

- `u64` is stored as-is; `i64` is stored `as u64`, which is
  bit-identical, because Cranelift integers are sign-agnostic and
  signedness is a property of the operations.
- `f64` is stored via `to_bits()`; Cranelift's `bitcast` converts
  it at no cost.
- `bool` is stored as 0 or 1 in a `u64` slot; codegen uses
  `types::I8` loads and constants for flag values, the only scalar
  type other than I64 and F64 that kernel codegen emits.
- Narrow widths (u8/i8/u16/i16/u32/i32/f32/f16) are stored zero- or
  sign-extended, or bit-stuffed, in the `u64` slot according to the
  static `PortType`. The `#[polydat_node]` macro's buffer tokens are
  width-aware (its internal `JitType` has one variant per
  width), so narrow-typed nodes get `compiled_u64` closures whose
  casts mirror the Wire storage conventions exactly; the typed
  readers of every compiled kernel sign-extend narrow signed
  outputs on the way out.
- 128-bit integers and register words are stored as `Imm2`, two
  immediate limbs in consecutive slots; the typed readers reassemble
  them (`marshal::decode_output`).
- Strings, byte strings, JSON, extension values, and handles are
  stored as `Ref2` pairs, per [Compiled By-Reference
  Slots](compiled_handles.md); every read copies out.

---

## Slot-state axioms (S1–S10)

These axioms are the normative contract for the state of a compiled
kernel's buffer and scratch, under the heap-slice plane and
slot-color contract (`type_system_alignment.md` §4, §6). They apply
to the three compiled engines (the closure tier, native, and pure
native) and to the interpreter's native cones. Each axiom is cited by
the SAFETY comments that rely on it and is enforced by a runtime
assertion or a CI check, not by comments alone.

**S1 — Slot color is static, total, and three-valued.** Every
`PortType` maps at kernel-build time to exactly one color:
`Imm1` (one slot, immediate value), `Imm2` (two slots, immediate
limbs — register words, u128/i128), `Ref2` (two slots,
`(ptr, len)` reference — heap slices). Width derives from color.
No runtime tags; no per-value color. Immediate slots never
contain an address, so buffer + port table is a complete state
description for everything except Ref data. *Chokepoint:
`PortType::slot_color()`; `slot_width()` derives from it.*

**S2 — Pointer containment.** A Ref pair is meaningful only
inside the engine's gather→op→scatter path. Raw readers (`get`,
`get_slot`, `eval_for_slot`) PANIC on Ref-colored slots; external
access goes through borrow-checked scratch accessors
(`read_vec_f32(&self, slot) -> &[f32]`, lifetime tied to `&self`
so holding a slice across the next `eval(&mut self)` is a compile
error) or owned copy-out.

**S3 — Single-writer scratch.** Each scratch entry is owned by
exactly one (step, output port); only the owning op mutates it,
and every execution republishes the `(ptr, len)` pair within that
execution. The engine passes each op only its own scratch range —
ownership by sub-slice, not discipline.

**S4 — Publish-before-read.** Steps execute sequentially in
topological order within an eval pass. A Ref pair's validity
interval is [owning op's scatter completes, owning op's next
invocation begins); every consumer gather occurs inside the
producer's current interval.

**S5 — Skip coherence.** If a producer executes in a pass, every
transitive consumer executes later in the same pass. Mechanical
oracle: the Raw (never-skip) engine and the Push/Pull/PushPull
(skip) engines must produce identical outputs for arbitrary
input-change sequences.

**S6 — Sequential-by-axiom; parallelism is a redesign gate.**
Kernel state (buffer + scratch) is single-threaded by
construction: one state per thread; cross-thread sharing only via
a shared program (`into_program`, an `Arc<dyn KernelProgram>`;
`PolydatProgram` on the interpreter); values cross threads only as
owned copies.
Intra-kernel parallel step execution is FORBIDDEN until S4 is
replaced with a new ordering proof (epochs/generations).

**S7 — One static dereference.** Ref access compiles to exactly
one pointer dereference; color/offset resolution completes at
kernel build. Index tables, handles resolved through a lookup, runtime
color dispatch, and any second hop are forbidden — in interpreter
closures and in JIT-generated code alike (a segment loads the
pointer from its slot and passes it).

**S8 — The interpreter is the semantic oracle.** Typed eval defines
meaning; every compiled engine must be bit-identical to it
(cross-lane float reductions per their declared fixed-shape
contracts). No node or op shape is added without an equivalence test
across all four engines.

**S9 — Deterministic runtime validation.** (a) In debug/test
builds, after every eval pass the engine asserts every
scratch-backed Ref pair equals its owning entry's current
`(as_ptr(), len())` — forgot-to-republish / wrong-slot /
dangling failures name the slot deterministically. (b) The
slice transport is checked under Miri, in a build without native
code, since Miri cannot execute native code: Stacked Borrows accepts
the `from_raw_parts` pattern and the leak check passes, so the leak
check also guards against regressions. Miri warns on the
integer-to-pointer casts inherent to passing pointers through `u64`
slots, where pointer provenance is necessarily reconstructed, so the
check runs under permissive integer-pointer semantics rather than
strict provenance.

**S10 — Unsafe is enumerable, annotated, and tripwired.** Every
Ref-deref `unsafe` lives in macro-generated `compiled_slot`
bodies (emitted from `polydat-derive`) or named engine accessor
sites; each SAFETY comment cites the axioms it relies on (S3,
S4). A CI tripwire fails when `from_raw_parts` appears outside
the allowlisted files.

**Native corollary.** Native code holds a `Ref2` pair as two
words and never dereferences it; the helper performs S7's single
dereference. A pure native kernel and a P3 segment hold Ref slots
wherever a slot call or a named string/vector lowering produces one,
and
`build_jit_layout` rejects only a node with neither a lowering nor
a kit.

**Forwarding boundary.** Native code never optimizes by passing a
Ref pair through from input to output. A Ref output is backed by its
step's scratch, and S9(a)'s validator applies to it, unless its pair
names storage that outlives the kernel (a string constant interned
for the process) or a boundary value borrowed for one call; a copy
step never forwards a pair.

**By-reference values.** `Str`, `Bytes`, `Json`, `Ext`, and `Handle`
are `Ref2` (S1): a string or byte string as the pair of its bytes, a
JSON, extension, or handle value as a one-element slice holding the
`Value`. The producing step owns the scratch entry the pair names (S3),
the pair is valid until that step runs again (S4), the closure or the
boundary decode makes the one dereference (S7), raw readers refuse the
slot (S2), and every read that leaves the compiled tier copies out.
Who owns every pair a compiled slot can hold is stated in
[Compiled By-Reference Slots](compiled_handles.md) §3; there are no
axioms beyond S1–S10 for these types.
