# JIT Boundary

The polydat-internal contract for the Phase-3 native kernel.
This doc specifies *how* the Cranelift-generated machine code
plugs into the rest of the runtime — the call boundary
between Rust and the native function, what happens when that
code raises a predicate violation, and how invalidation and
extern resolution cross the boundary.

For the engine-selection side (*which* compilation level the
compiler picks for a given subgraph), see
[graph_compiler.md §6 (ordered composition)](graph_compiler.md).
The clean-flag and memoization model the JIT preserves across
the boundary is in [runtime_model.md](runtime_model.md) (R-axioms).

---

## Call boundary overview

A Phase-3 kernel compiles to a single native function:

```text
fn(coords: *const u64, buffer: *mut u64)        // raw
fn(coords: *const u64, buffer: *mut u64,
   clean:  *mut u8)                             // provenance variant
```

The Rust wrapper owns a `Vec<u64>` buffer and calls the function
pointer each cycle. Four kernel variants dispatch the same way
but apply different optimizations:

| Kernel | Optimization |
|---|---|
| `JitKernelRaw` | Runs every node unconditionally |
| `JitKernelPush` | Per-node dirty tracking (push-side step skip) |
| `JitKernelPull` | Cone guard for per-slot eval (pull-side skip) |
| `JitKernelPushPull` | Both |

The `HybridKernelRaw` / `HybridKernelPull` / `HybridKernelPushPull`
variants mix Phase-3 JIT segments with Phase-2 closure steps
inside one kernel, using the same buffer.

All of these share one entry rule: **every call into native code
goes through `codegen::invoke_with_catch`.** There is no direct
`(code_fn)(...)` invocation anywhere in the library; grep for
that pattern finds only the wrapper itself.

---

## Why predicate violations are a problem at this boundary

SRD 12 §"Parameter resolution and validation" lists three JIT-
lowered predicates that can fail at cycle time: `is_positive`,
`in_range`, `is_one_of`. When they fail the JIT emits a call to
an extern helper (`jit_is_positive_fail`, `jit_in_range_fail`,
`jit_is_one_of_fail`) that must report the violation and stop
the current evaluation.

The obvious shape — `panic!` from the extern helper and catch
it upstream — does not work with Cranelift-generated frames:

- Cranelift emits DWARF `.eh_frame` entries when
  `unwind_info=true`, and registers them with the platform
  unwinder via `JITModule::finalize_definitions()`.
- Those frames do **not** carry Rust's panic personality
  routine. The libunwind walker can traverse them, but
  `_Unwind_RaiseException` never finds a catch block and the
  panic runtime aborts with "failed to initiate panic, error 5
  (`_URC_END_OF_STACK`)".
- Switching the extern to `extern "C-unwind"` alone doesn't
  fix this — the issue is the missing personality, not the
  ABI flag.

Teaching Cranelift to emit `.gcc_except_table` entries referencing
`rust_eh_personality`, plus re-registering frames via a personality
shim, is an integration project that isn't in this crate's scope.

---

## The setjmp / longjmp workaround

Rather than unwinding *through* the JIT frame, we jump *past* it.

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
        │  panic!(message)   ← Rust-land panic     │
        └──────────────────────────────────────────┘
                     │
          ordinary Rust unwind, proper personality
                     ▼
          std::panic::catch_unwind in the caller
```

The longjmp skips the JIT frame entirely — no unwinding through
it, no personality lookup, no catch-block walk. Control returns
into Rust land where a normal `panic!` propagates through
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
  different tokio worker threads don't share state.
- **Nesting.** `invoke_with_catch` saves the outer
  `JIT_JMP_BUF` slot in a stack-local variable, installs its
  own buffer, and restores the outer on every exit path. An
  inner longjmp jumps to the innermost buffer; the outer
  regains the slot when the inner frame unwinds.
- **SIMD register state.** `_setjmp` on glibc preserves only
  the core register set that `longjmp` restores. The Rust
  wrapper doesn't keep live SIMD state across the JIT call,
  so this is fine. Workloads that wanted to keep live SIMD
  data across a predicate violation would have a larger
  problem.

### Platform-portable jmp_buf shim

`libc` doesn't expose `jmp_buf` / `setjmp` / `longjmp` (they're
generally considered unsafe to reach from Rust). We declare
them directly:

```rust
#[repr(C, align(16))]
struct JitJmpBuf([u8; 512]);          // 512 > glibc (~200) > macOS (~192)

unsafe extern "C" {
    fn _setjmp(env: *mut JitJmpBuf) -> i32;
    fn _longjmp(env: *mut JitJmpBuf, val: i32) -> !;
}
```

We link against `_setjmp` / `_longjmp` (rather than plain
`setjmp` / `longjmp`) because the plain variants are glibc
macros that expand to `__sigsetjmp(env, 0)` — saving the
signal mask, which we don't need. `_setjmp` saves registers
only and is faster.

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
- If `f()` triggers a JIT predicate violation, the extern
  helper `_longjmp`s back; the wrapper reads the TLS message
  and raises `panic!`. The guard still runs (on the panic
  unwind path inside the wrapper frame) and restores the
  outer slot.
- If `f()` panics for a non-JIT reason (a bug in a non-JIT
  sub-path; a panic from a closure step in a hybrid kernel),
  the panic unwinds through the wrapper's frame. The guard's
  `Drop` restores the outer slot before the unwind
  continues. Subsequent `invoke_with_catch` calls see a clean
  sentinel. This is covered by the test
  `invoke_with_catch_restores_slot_after_foreign_panic`.

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
setjmp-return-then-panic, or panic-through-wrapper — runs
through `Drop`.

---

## Where the wrapper is applied

Every `eval` / `eval_for_slot` on every JIT and Hybrid kernel
variant uses the wrapper:

| Caller | Uses |
|---|---|
| `JitKernelRaw::eval` | ✓ |
| `JitKernelPush::eval` | ✓ |
| `JitKernelPull::eval`, `eval_for_slot` | ✓ (both invocation sites) |
| `JitKernelPushPull::eval`, `eval_for_slot` | ✓ (both) |
| `HybridCore::eval_all_hybrid_steps` (used by `HybridKernelRaw::eval`, `HybridKernelPull::eval`, `HybridKernelPull::eval_for_slot`'s dirty path) | ✓ (each per-step JIT segment) |
| `HybridKernelPushPull::eval`, `eval_for_slot` | ✓ (each per-step JIT segment) |

The `JitKernelRaw::into_parts` accessor remains a raw-pointer
export for hybrid-kernel integration. Callers of `into_parts`
are expected to either build a hybrid kernel (which wraps every
call) or install their own wrapper before invoking the pointer;
calling the pointer directly without either would abort on
violation via the no-sentinel fallback.

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
            // Raw code_fn invoked outside a wrapper.
            eprintln!("{msg}");
            std::process::abort();
        }
    }
}
```

This is the last-line defense. In practice the only way to
reach it is to call a JIT function pointer without going through
one of the wrapped kernels (e.g. a test that retrieves the raw
pointer via `into_parts` and invokes it directly). The fallback
prints the message and aborts — the same behavior the wrapper
replaces in the normal path, but without the catch-unwind
integration.

---

## Extern-helper table

Each predicate has one dedicated fail helper. The helpers live
in `polydat/src/compile/jit/codegen.rs` and are registered with
Cranelift's JIT symbol table so the emitted native code can
call them.

| Extern | Arity | Called from |
|---|---|---|
| `jit_is_positive_fail` | `(u64, ptr, len) -> u64` (value, control name) | `JitOp::IsPositiveCheck` |
| `jit_in_range_fail` | `(u64, u64, u64) -> u64` (value, lo, hi) | `JitOp::InRangeCheck` |
| `jit_is_one_of_fail` | `(u64, ptr, len) -> u64` (value, allowed set) | `JitOp::IsOneOfCheck` |

The `u64` return type matches the extern-function ABI the JIT
uses; since each helper ends in `_longjmp` (which is `-> !`),
the return is unreachable.

### Handle helpers (SRD 115 §6)

The non-scalar lowerings call helpers over `u64` handles, registered
the same way. Every argument and return is an `I64`: bits, a handle,
an interned address, or a type code.

| Extern | Arity | Called from |
|---|---|---|
| `jit_u64_to_str`, `jit_i64_to_str`, `jit_f64_to_str`, `jit_bool_to_str` | `(bits) -> arena handle` | `JitOp::U64ToString` and siblings |
| `jit_str_to_u64`, `jit_str_to_i64`, `jit_str_to_f64`, `jit_str_to_bool` | `(handle) -> bits`; a failed parse longjmps | `JitOp::StringToU64` and siblings |
| `jit_str_lower`, `jit_str_upper`, `jit_str_trim`, `jit_str_len` | `(handle) -> handle or u64` | `JitOp::StrLower` and siblings |
| `jit_str_concat` | `(handle, handle) -> arena handle` | `JitOp::StrConcat` |
| `jit_u64_to_json`, `jit_i64_to_json`, `jit_f64_to_json`, `jit_bool_to_json`, `jit_str_to_json` | `(entry, arg) -> table handle` | `JitOp::U64ToJson` and siblings |
| `jit_json_to_str` | `(handle) -> arena handle` | `JitOp::JsonToStr` |
| `jit_printf`, `jit_tile_render` | `(interned address, type codes, args ptr) -> arena handle` | `JitOp::Printf`, `JitOp::TileRender` |
| `jit_json_array`, `jit_json_object` | `(entry, type codes, args ptr) -> table handle` | `JitOp::JsonArray`, `JitOp::JsonObject` |
| `jit_to_json` | `(entry, type code, bits) -> table handle` | `JitOp::ToJson` |
| `jit_json_text` | `(type code, bits) -> arena handle` | `JitOp::JsonText` |
| `jit_tile_encode` | `(interned address, type code, bits) -> arena handle` | `JitOp::TileEncode` |

A producer of a table handle is told the entry it owns as an immediate
and writes through the table the engine installed around the call
(`with_value_table`); a helper that runs with no table installed
panics. The variadic helpers read their arguments from an array the
generated code stores into its own frame, decoding each by a one-byte
type code interned as a static string.

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

From the outside looking in, a predicate violation in JIT code
behaves exactly like a predicate violation in Phase-1 or Phase-2:

- `#[should_panic(expected = "must be > 0")]` on the caller
  works.
- `std::panic::catch_unwind` catches and returns `Err`.
- The panic message carries the violating value (and, for
  `in_range`, the configured bounds).
- The workload can continue — the kernel survives catches;
  the per-cycle buffer is left partially written for the
  failing step but subsequent evals overwrite cleanly.

The helper ABI preserves the `is_positive` control name and the
`is_one_of` allowed set through stable pointers into node metadata.
`in_range` carries its numeric bounds directly. These values remain
valid for the compiled kernel lifetime because the JIT core retains
the originating nodes.

---

## Tests

`polydat/src/compile/jit/codegen.rs` carries unit coverage:

- Per-predicate happy path: value passes through.
- Per-predicate catchable-panic path: violation fires and
  `catch_unwind` returns `Err` with the expected message.
- `jit_kernel_survives_multiple_violations` — repeated
  caught violations followed by a happy-path eval all work
  on the same kernel instance, proving no state leaks
  across longjmp.
- `invoke_with_catch_restores_slot_after_foreign_panic` —
  a non-JIT panic inside the closure still restores the
  TLS slot, so a subsequent legitimate JIT violation is
  caught cleanly. This is the specific regression `JmpBufGuard`
  protects against.

---

## Unwind boundary constraint

JIT predicate violations cross generated frames through the
documented setjmp/longjmp trampoline. Generated code MUST NOT
allow a Rust panic to unwind through a Cranelift frame because
the emitted object does not register a compatible Rust unwind
personality for that path. The TLS jump-buffer guard and
Rust-side catch boundary are therefore part of the ABI, not an
optional implementation detail.

---

## SIMD compute kernels (alignment §8.2)

`compile/jit/simd.rs` compiles four f32-lane kernels once per
process through the same cranelift engine, using real cranelift
SIMD types (`F32X4`): `dot_f32`, `l2sq_f32`, `add_f32`,
`scale_f32`. Each processes the slice body in 128-bit chunks
(unaligned loads — `SliceArc<f32>` data is only 4-aligned) with a
scalar tail loop, and reducing kernels finish with an
`extractlane` horizontal sum. Consumers are the `vec_*` nodes in
`library/vector_math.rs`, which fall back to scalar Rust loops
when the `jit` feature is off or host-ISA construction fails.
This is also usable through the slot ABI. Typed slice values
cross compiled steps as `(ptr, len)` slot pairs and
`CompiledSlotOp` publishes vector results through kernel-owned
scratch. Scalar-only P3 segments retain the compact
`fn(coords, buffer)` shape; slice-bearing steps use the wider
compiled-op contract described by the slot-state axioms below.

SIMD accumulation reassociates float addition, so reduced results
may differ from the scalar reference in the final ulps; the
equivalence tests compare with relative tolerance.

## Scalar buffer conventions

- `u64` rides as-is; `i64` is `as u64` (bit-identical — cranelift
  integers are sign-agnostic, signedness lives in the ops).
- `f64` rides via `to_bits()`; cranelift `bitcast` converts for
  free.
- `bool` rides as 0/1 in a u64 slot; codegen uses `types::I8`
  loads and constants for flag values — the only non-{I64, F64}
  scalar type the kernel codegen emits.
- Narrow widths (u8/i8/u16/i16/u32/i32/f32/f16) ride zero-/sign-
  extended or bit-stuffed in the u64 slot per the static
  `PortType`. The `#[polydat_node]` macro's buffer tokens are
  width-aware (its internal `JitType` carries one variant per
  width), so narrow-typed nodes get `compiled_u64` closures whose
  casts mirror the Wire storage conventions exactly — pinned by
  the P1↔P2 equivalence tests in `polydat_node_macro.rs`.
- 128-bit integers do not ride at all (interpreter-only) until a
  two-slot protocol exists.

---

## Slot-state axioms (S1–S10) — RATIFIED 2026-06-12

The normative contract for compiled-kernel buffer state under the
§8.4 vector substrate (`type_system_alignment.md`). These are
axioms in the SYSREF sense: load-bearing, cited by SAFETY
comments, and enforced by tripwires rather than comments.

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
error) or owned copy-out. A walled-off API in the established
sense.

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
input-change sequences (`tests/slot_state_axioms.rs`).

**S6 — Sequential-by-axiom; parallelism is a redesign gate.**
Kernel state (buffer + scratch) is single-threaded by
construction: one state per thread; cross-thread sharing only via
`Arc<PolydatProgram>`; values cross threads only as owned copies.
Intra-kernel parallel step execution is FORBIDDEN until S4 is
replaced with a new ordering proof (epochs/generations).

**S7 — One static dereference.** Ref access compiles to exactly
one pointer dereference; color/offset resolution completes at
kernel build. Index tables, arena handles, runtime color
dispatch, and any second hop are forbidden — in interpreter
closures and in JIT-generated code alike (a segment loads the
pointer from its slot and passes it).

**S8 — P1 is the semantic oracle.** Typed eval defines meaning;
every compiled tier must be bit-identical to it (cross-lane float
reductions per their declared fixed-shape contracts). No node or
op shape lands without a P1↔P2(↔P3) equivalence test.

**S9 — Deterministic runtime validation.** (a) In debug/test
builds, after every eval pass the engine asserts every
scratch-backed Ref pair equals its owning entry's current
`(as_ptr(), len())` — forgot-to-republish / wrong-slot /
dangling failures name the slot deterministically. (b) The
slice-transport tests run under Miri (no-jit configuration —
Miri cannot execute JIT'd native code) to adjudicate the formal
aliasing validity of the `from_raw_parts` pattern. Lane command:

```sh
MIRIFLAGS=-Zmiri-ignore-leaks cargo +nightly miri test \
    -p polydat --no-default-features --test slot_state_axioms
```

Lane command (clean leak-checking, no suppression flags):

```sh
cargo +nightly miri test -p polydat --no-default-features \
    --test slot_state_axioms
```

ADJUDICATED 2026-06-12: Stacked Borrows accepts the pattern (the
S5 oracle passes under Miri across all five engines) AND the leak
check passes clean. One caveat, recorded rather than hidden: Miri
warns on the integer-to-pointer casts (inherent to a u64-slot
transport — provenance is necessarily reconstructed, so the check
runs under permissive int-ptr semantics rather than strict
provenance). The earlier `-Zmiri-ignore-leaks` requirement is
gone: the ~200-byte-per-compile leak it masked was a real bug in
`registry::lookup` (it `Box::leak`'d a clone to fabricate a
`'static` instead of returning the already-`'static` inventory
entry) plus a latent twin in the fusion matcher (`Box::leak` on a
synthesized bind name, now `Cow::Owned`); both fixed 2026-06-12,
so the leak check itself is now the regression guard.

**S10 — Unsafe is enumerable, annotated, and tripwired.** Every
Ref-deref `unsafe` lives in macro-generated `compiled_slot`
bodies (emitted from `polydat-derive`) or named engine accessor
sites; each SAFETY comment cites the axioms it relies on (S3,
S4). A CI tripwire fails when `from_raw_parts` appears outside
the allowlisted files.

**P3 corollary.** Pure-P3 kernels contain no Ref slots by
construction (slice-bearing nodes classify `Fallback`, and
`build_jit_layout` rejects Fallback); the JIT builders enforce
this defensively. Hybrid kernels carry Ref slots only in closure
steps.

**Forwarding boundary.** Ref-pair pass-through is not a P3
optimization. Every Ref output is scratch-backed and S9(a)'s
validator mapping applies to every Ref output without exemption.

**Handle slots (SRD 115).** A fourth color, `Hdl1`, carries `Str`,
`Bytes`, `Json`, `Ext`, and `Handle` values as one-slot handles. S1's
statement that immediate slots never contain an address is unchanged:
`Hdl1` is not an immediate color. S7's ban on arena handles applies to
`Ref2` data, where a handle would be a second dereference; for `Hdl1`
the handle is the value's representation and the single dereference S7
protects is the helper's. Raw readers refuse `Hdl1` slots as they
refuse `Ref2` slots. Byte-string handles (`Str`, `Bytes`) cross a cone
boundary by SRD 115 §5: the input is copied into the cycle arena and the
output is copied out to an owned value, so P1 never holds a handle.
Table handles (`Json`, `Ext`, `Handle`) cross by the engine-owned value
table of SRD 115 §3: a cone borrows one for the eval and releases it, a
whole kernel owns one and validates it after every run, and helpers
reach it only through the installation the engine makes around its
native call. The handle axioms are stated normatively in
[Compiled Non-Scalar Slots](compiled_handles.md) §8 and summarised here
so a SAFETY comment can cite them beside the S-axioms:

- **H1 — A handle is a name, not an address.** Generated code loads,
  stores, and passes handles; only helpers, closures, and the engine
  decode. *Chokepoint: no `Hdl1` slot is an operand of arithmetic or
  memory instructions in emitted IR.*
- **H2 — Three places, decided at build.** Byte strings are static or
  arena handles, everything else a table handle, by `PortType`.
  *Chokepoint: `PortType::handle_kind()`.*
- **H3 — Handles live one cycle.** Arena handles until the root cycle's
  reset, a cone's until its eval returns, table handles within the
  generation that wrote them. *Tripwire: the generation stamp in every
  table handle, checked on every read; handle-producing steps are never
  skipped as clean.*
- **H4 — One writer per table entry.** Exactly one (step, port) owns an
  entry, is told its number at compile time, and republishes it every
  execution. *Tripwire: the post-run validator in the P2 and P3
  kernels, the S9(a) assertion restated for handles.*
- **H5 — Only the root resets.** The arena resets at a root kernel's
  cycle advance and nowhere else. *Chokepoint: the nested flag the
  activation, subscope, and tile constructors set.*
- **H6 — P1 never holds a handle.** Every `Hdl1` boundary output and
  every `get_value` read copies out to an owned `Value`.
- **H7 — Equivalence.** P1, P2, and P3 produce identical bytes.
  *Tripwire: `tests/handle_tiers.rs`.*
