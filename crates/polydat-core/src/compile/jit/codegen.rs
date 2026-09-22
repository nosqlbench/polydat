// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! JIT codegen: Cranelift IR generation, operation classification, and
//! extern "C" runtime helpers called from JIT-compiled code.
//!
//! `JitOp` classifies each DAG node into an inline IR pattern or an
//! extern call. `compile_jit_impl` lowers a slice of `(JitOp, inputs,
//! outputs)` steps into a single native function via Cranelift.
//! The four `compile_jit_*` constructors wrap the result in the
//! appropriate kernel struct from `kernels`.

use std::collections::HashMap;
use std::mem;

use cranelift_codegen::ir::{self, AbiParam, InstBuilder, types};
use cranelift_codegen::settings::{self, Configurable};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module};

use crate::ast::PolydatNode;
use crate::ast::SlotShape;

use super::kernels::{JitCore, JitKernelPushPull, JitKernelRaw};

// ── Extern "C" runtime helpers ─────────────────────────────

/// Extern function: xxhash3 of a u64 (called from JIT code).
extern "C" fn jit_xxh3_hash(value: u64) -> u64 {
    guarded(|| xxhash_rust::xxh3::xxh3_64(&value.to_le_bytes()))
}

/// Extern function: interleave bits of two u64 values (called from JIT code).
extern "C" fn jit_interleave(a: u64, b: u64) -> u64 {
    guarded(|| {
        let mut result: u64 = 0;
        for i in 0..32 {
            result |= ((a >> i) & 1) << (2 * i);
            result |= ((b >> i) & 1) << (2 * i + 1);
        }
        result
    })
}

/// Extern function: interpolating LUT sample (called from JIT code).
///
/// Input is f64 bits in [0,1]. LUT pointer + length are baked constants.
/// Returns f64 result as u64 bits.
extern "C" fn jit_lut_sample(input_bits: u64, lut_ptr: u64, lut_len: u64) -> u64 {
    guarded(|| {
        let u = f64::from_bits(input_bits).clamp(0.0, 1.0);
        let n = (lut_len - 1) as f64;
        let pos = u * n;
        let idx = (pos as usize).min(lut_len as usize - 2);
        let frac = pos - idx as f64;
        let result = unsafe {
            let ptr = lut_ptr as *const f64;
            let a = *ptr.add(idx);
            let b = *ptr.add(idx + 1);
            a * (1.0 - frac) + b * frac
        };
        result.to_bits()
    })
}

/// Extern function: LFSR shuffle (called from JIT code).
extern "C" fn jit_shuffle(input: u64, feedback: u64, size: u64, min: u64) -> u64 {
    guarded(|| {
        let mut register = (input % size) + 1;
        loop {
            let lsb = register & 1;
            register >>= 1;
            if lsb != 0 {
                register ^= feedback;
            }
            if register <= size {
                break;
            }
        }
        (register - 1) + min
    })
}

// Extern functions for math operations (called from JIT code).
extern "C" fn jit_sin(bits: u64) -> u64 {
    guarded(|| f64::from_bits(bits).sin().to_bits())
}
extern "C" fn jit_cos(bits: u64) -> u64 {
    guarded(|| f64::from_bits(bits).cos().to_bits())
}
extern "C" fn jit_tan(bits: u64) -> u64 {
    guarded(|| f64::from_bits(bits).tan().to_bits())
}
extern "C" fn jit_asin(bits: u64) -> u64 {
    guarded(|| f64::from_bits(bits).asin().to_bits())
}
extern "C" fn jit_acos(bits: u64) -> u64 {
    guarded(|| f64::from_bits(bits).acos().to_bits())
}
extern "C" fn jit_atan(bits: u64) -> u64 {
    guarded(|| f64::from_bits(bits).atan().to_bits())
}
extern "C" fn jit_sqrt(bits: u64) -> u64 {
    guarded(|| f64::from_bits(bits).sqrt().to_bits())
}
extern "C" fn jit_abs_f64(bits: u64) -> u64 {
    guarded(|| f64::from_bits(bits).abs().to_bits())
}
extern "C" fn jit_ln(bits: u64) -> u64 {
    guarded(|| f64::from_bits(bits).ln().to_bits())
}
extern "C" fn jit_exp(bits: u64) -> u64 {
    guarded(|| f64::from_bits(bits).exp().to_bits())
}
extern "C" fn jit_floor_base10(bits: u64) -> u64 {
    guarded(|| {
        use crate::numeric::round_numbers::*;
        let x = f64::from_bits(bits);
        let r = if !positive_finite(x) {
            0.0
        } else {
            floor_pow10(x)
        };
        r.to_bits()
    })
}
extern "C" fn jit_ceiling_base10(bits: u64) -> u64 {
    guarded(|| {
        use crate::numeric::round_numbers::*;
        let x = f64::from_bits(bits);
        let r = if !positive_finite(x) {
            0.0
        } else {
            let lo = floor_pow10(x);
            if lo == x { lo } else { lo * 10.0 }
        };
        r.to_bits()
    })
}
extern "C" fn jit_closest_base10(bits: u64) -> u64 {
    guarded(|| {
        use crate::numeric::round_numbers::*;
        let x = f64::from_bits(bits);
        let r = if !positive_finite(x) {
            0.0
        } else {
            let lo = floor_pow10(x);
            let hi = if lo == x { lo } else { lo * 10.0 };
            pick_closest(x, lo, hi)
        };
        r.to_bits()
    })
}
extern "C" fn jit_floor_decade(bits: u64) -> u64 {
    guarded(|| {
        use crate::numeric::round_numbers::*;
        let x = f64::from_bits(bits);
        let r = if !positive_finite(x) {
            0.0
        } else {
            let base = floor_pow10(x);
            (x / base).floor() * base
        };
        r.to_bits()
    })
}
extern "C" fn jit_ceiling_decade(bits: u64) -> u64 {
    guarded(|| {
        use crate::numeric::round_numbers::*;
        let x = f64::from_bits(bits);
        let r = if !positive_finite(x) {
            0.0
        } else {
            let base = floor_pow10(x);
            (x / base).ceil() * base
        };
        r.to_bits()
    })
}
extern "C" fn jit_closest_decade(bits: u64) -> u64 {
    guarded(|| {
        use crate::numeric::round_numbers::*;
        let x = f64::from_bits(bits);
        let r = if !positive_finite(x) {
            0.0
        } else {
            let base = floor_pow10(x);
            (x / base).round() * base
        };
        r.to_bits()
    })
}
extern "C" fn jit_floor_binomial(bits: u64) -> u64 {
    guarded(|| {
        use crate::numeric::round_numbers::*;
        let x = f64::from_bits(bits);
        let r = if !positive_finite(x) {
            0.0
        } else {
            floor_pow2(x)
        };
        r.to_bits()
    })
}
extern "C" fn jit_ceiling_binomial(bits: u64) -> u64 {
    guarded(|| {
        use crate::numeric::round_numbers::*;
        let x = f64::from_bits(bits);
        let r = if !positive_finite(x) {
            0.0
        } else {
            let lo = floor_pow2(x);
            if lo == x { lo } else { lo * 2.0 }
        };
        r.to_bits()
    })
}
extern "C" fn jit_closest_binomial(bits: u64) -> u64 {
    guarded(|| {
        use crate::numeric::round_numbers::*;
        let x = f64::from_bits(bits);
        let r = if !positive_finite(x) {
            0.0
        } else {
            let lo = floor_pow2(x);
            let hi = if lo == x { lo } else { lo * 2.0 };
            pick_closest(x, lo, hi)
        };
        r.to_bits()
    })
}
extern "C" fn jit_floor_fibonacci(bits: u64) -> u64 {
    guarded(|| {
        use crate::numeric::round_numbers::*;
        let x = f64::from_bits(bits);
        let r = if !positive_finite(x) {
            0.0
        } else {
            floor_fibonacci_val(x)
        };
        r.to_bits()
    })
}
extern "C" fn jit_ceiling_fibonacci(bits: u64) -> u64 {
    guarded(|| {
        use crate::numeric::round_numbers::*;
        let x = f64::from_bits(bits);
        let r = if !positive_finite(x) {
            0.0
        } else {
            ceiling_fibonacci_val(x)
        };
        r.to_bits()
    })
}
extern "C" fn jit_closest_fibonacci(bits: u64) -> u64 {
    guarded(|| {
        use crate::numeric::round_numbers::*;
        let x = f64::from_bits(bits);
        let r = if !positive_finite(x) {
            0.0
        } else {
            pick_closest(x, floor_fibonacci_val(x), ceiling_fibonacci_val(x))
        };
        r.to_bits()
    })
}

extern "C" fn jit_atan2(y_bits: u64, x_bits: u64) -> u64 {
    guarded(|| {
        f64::from_bits(y_bits)
            .atan2(f64::from_bits(x_bits))
            .to_bits()
    })
}
extern "C" fn jit_pow(base_bits: u64, exp_bits: u64) -> u64 {
    guarded(|| {
        f64::from_bits(base_bits)
            .powf(f64::from_bits(exp_bits))
            .to_bits()
    })
}
extern "C" fn jit_round_nearest(x_bits: u64, iv_bits: u64) -> u64 {
    guarded(|| {
        let x = f64::from_bits(x_bits);
        let interval = f64::from_bits(iv_bits);
        let r = if !(interval.is_finite() && interval > 0.0) {
            x
        } else {
            (x / interval).round() * interval
        };
        r.to_bits()
    })
}
extern "C" fn jit_round_floor(x_bits: u64, iv_bits: u64) -> u64 {
    guarded(|| {
        let x = f64::from_bits(x_bits);
        let interval = f64::from_bits(iv_bits);
        let r = if !(interval.is_finite() && interval > 0.0) {
            x
        } else {
            (x / interval).floor() * interval
        };
        r.to_bits()
    })
}
extern "C" fn jit_round_ceiling(x_bits: u64, iv_bits: u64) -> u64 {
    guarded(|| {
        let x = f64::from_bits(x_bits);
        let interval = f64::from_bits(iv_bits);
        let r = if !(interval.is_finite() && interval > 0.0) {
            x
        } else {
            (x / interval).ceil() * interval
        };
        r.to_bits()
    })
}

extern "C" fn jit_pcg(input: u64, seed: u64, stream: u64) -> u64 {
    guarded(|| {
        let inc = 2u64.wrapping_mul(stream).wrapping_add(1);
        crate::numeric::pcg::pcg_seek(seed, inc, input)
    })
}
extern "C" fn jit_pcg_stream(input: u64, stream: u64, seed: u64) -> u64 {
    guarded(|| {
        let inc = 2u64.wrapping_mul(stream).wrapping_add(1);
        crate::numeric::pcg::pcg_seek(seed, inc, input)
    })
}
extern "C" fn jit_n_of(input: u64, n: u64, m: u64) -> u64 {
    guarded(|| {
        if m == 0 {
            return 0;
        }
        crate::numeric::n_of_m::n_of_m_eval(input, n, m)
    })
}

extern "C" fn jit_cycle_walk(pos: u64, range: u64, seed: u64, inc: u64) -> u64 {
    guarded(|| {
        let stream = inc.saturating_sub(1) / 2;
        let state = crate::numeric::pcg::build_cycle_walk_state(range, seed, stream);
        crate::numeric::pcg::cycle_walk_inner(
            pos,
            range,
            state.half_bits,
            state.half_mask,
            &state.round_keys,
        )
    })
}

extern "C" fn jit_perlin_1d(input: u64, perm_ptr: u64, freq_bits: u64) -> u64 {
    guarded(|| {
        let perm = unsafe { &*(perm_ptr as *const crate::numeric::noise::PermTable) };
        let freq = f64::from_bits(freq_bits);
        let r = crate::numeric::noise::perlin_1d_algo(perm, input as f64 * freq);
        r.to_bits()
    })
}

extern "C" fn jit_perlin_2d(x: u64, y: u64, perm_ptr: u64, freq_bits: u64) -> u64 {
    guarded(|| {
        let perm = unsafe { &*(perm_ptr as *const crate::numeric::noise::PermTable) };
        let freq = f64::from_bits(freq_bits);
        let r = crate::numeric::noise::perlin_2d_algo(perm, x as f64 * freq, y as f64 * freq);
        r.to_bits()
    })
}

extern "C" fn jit_simplex_2d(x: u64, y: u64, perm_ptr: u64, freq_bits: u64) -> u64 {
    guarded(|| {
        let perm = unsafe { &*(perm_ptr as *const crate::numeric::noise::PermTable) };
        let freq = f64::from_bits(freq_bits);
        let r = crate::numeric::noise::simplex_2d_algo(perm, x as f64 * freq, y as f64 * freq);
        r.to_bits()
    })
}

extern "C" fn jit_fractal_noise_1d(input: u64, perm_ptr: u64, freq_bits: u64, octaves: u64) -> u64 {
    guarded(|| {
        let perm = unsafe { &*(perm_ptr as *const crate::numeric::noise::PermTable) };
        let freq = f64::from_bits(freq_bits);
        let r = crate::numeric::noise::fbm_1d(perm, input as f64, freq, octaves as u32);
        r.to_bits()
    })
}

extern "C" fn jit_fractal_noise_2d(
    x: u64,
    y: u64,
    perm_ptr: u64,
    freq_bits: u64,
    octaves: u64,
) -> u64 {
    guarded(|| {
        let perm = unsafe { &*(perm_ptr as *const crate::numeric::noise::PermTable) };
        let freq = f64::from_bits(freq_bits);
        let r = crate::numeric::noise::fbm_2d(perm, x as f64, y as f64, freq, octaves as u32);
        r.to_bits()
    })
}

extern "C" fn jit_thread_id() -> u64 {
    guarded(|| {
        let id = std::thread::current().id();
        let id_str = format!("{id:?}");
        let num = id_str.trim_start_matches("ThreadId(").trim_end_matches(')');
        num.parse().unwrap_or(0)
    })
}

extern "C" fn jit_current_epoch_millis() -> u64 {
    guarded(|| {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64
    })
}

// ── Catchable predicate violations via setjmp/longjmp ─────────
//
// Cranelift-JIT emits DWARF unwind info (`unwind_info=true`) but
// does not call `__register_frame`; teaching the system
// unwinder about JIT frames needs either an upstream Cranelift
// change or a personality-routine shim that's a project on its
// own. We take the self-contained route instead: a setjmp
// sentinel installed by the Rust eval wrapper, and extern
// helpers that `longjmp` back to it on violation.
//
// The longjmp skips over the JIT frame entirely — no unwind,
// no personality lookup, no catch-block walk. Control returns
// to the Rust wrapper which reads the violation message from a
// thread-local and raises a normal Rust `panic!`. That panic
// unwinds through the Rust caller's frames (which have proper
// `rust_eh_personality` FDEs) and `catch_unwind` catches it
// like any other panic. Fail-path callers no longer lose the
// entire process to an abort.
//
// Safety
//   - longjmp skips C-level destructors. The JIT code is pure
//     machine code with no Drop semantics, so nothing leaks.
//     The extern helpers themselves hold no resources.
//   - The thread-local buffer is per-thread, so concurrent
//     kernels on different tokio worker threads don't share
//     state. A nested evaluation on the same thread installs
//     its own buffer and `JmpBufGuard` restores the enclosing
//     one on every exit path, so the slot behaves as a stack.

/// Platform-independent jmp_buf shim. Allocated oversize (512
/// bytes, 16-aligned) so the biggest real platform buffer
/// (glibc Linux: ~200 bytes, macOS: ~192) fits with margin.
/// We link against the C library's `_setjmp` / `_longjmp`
/// symbols directly — the `setjmp` macro in the glibc header
/// expands to `__sigsetjmp`, which saves the signal mask; we
/// don't need that and `_setjmp` is faster.
#[repr(C, align(16))]
struct JitJmpBuf([u8; 512]);

#[cfg(not(windows))]
unsafe extern "C" {
    fn _setjmp(env: *mut JitJmpBuf) -> i32;
    fn _longjmp(env: *mut JitJmpBuf, val: i32) -> !;
}

// MSVC CRT spelling of the same pair: it exports `longjmp`
// (no underscore — `_longjmp` doesn't exist there, LNK2019)
// and an x64 `_setjmp` whose second register argument is
// recorded as the jmp_buf's `Frame` field. The C compiler
// normally fills that in via intrinsic; calling from Rust we
// pass NULL explicitly, which is load-bearing twice over: it
// keeps rdx from carrying garbage into the buffer, and a zero
// `Frame` makes `longjmp` do a plain register restore instead
// of an `RtlUnwindEx` unwind — mandatory here because the
// frames being skipped are JIT code with no unwind tables
// registered (the exact problem this setjmp path exists to
// avoid; see the module comment above).
#[cfg(windows)]
unsafe extern "C" {
    fn _setjmp(env: *mut JitJmpBuf, frame: *mut std::ffi::c_void) -> i32;
    #[link_name = "longjmp"]
    fn _longjmp(env: *mut JitJmpBuf, val: i32) -> !;
}

use std::cell::{Cell, RefCell};
thread_local! {
    /// Set by [`invoke_with_catch`] before entering JIT code;
    /// cleared on return. The extern longjmp helpers consult
    /// this slot to find their return target. `None` means "no
    /// wrapper installed" → fall back to abort so violations
    /// outside a catching wrapper still terminate cleanly
    /// rather than triggering undefined behavior.
    static JIT_JMP_BUF: Cell<Option<*mut JitJmpBuf>> = const { Cell::new(None) };
    /// Populated by the extern helpers right before the
    /// longjmp; drained by the wrapper after setjmp returns
    /// non-zero.
    static JIT_VIOLATION_MSG: RefCell<Option<String>> = const { RefCell::new(None) };
}

/// Store the violation message and longjmp back to the wrapper.
/// Used by every predicate extern on the fail path. If no
/// wrapper is installed on the current thread (e.g. someone
/// calling the JIT code directly without `invoke_with_catch`),
/// prints the message and aborts — matches the original
/// behavior for that call pattern.
fn jit_violation_longjmp(msg: String) -> ! {
    JIT_VIOLATION_MSG.with(|m| *m.borrow_mut() = Some(msg.clone()));
    let buf_ptr: Option<*mut JitJmpBuf> = JIT_JMP_BUF.with(|b| b.get());
    match buf_ptr {
        Some(ptr) => unsafe { _longjmp(ptr, 1) },
        None => {
            let mut err = std::io::stderr().lock();
            use std::io::Write;
            let _ = writeln!(err, "{msg}");
            let _ = err.flush();
            std::process::abort();
        }
    }
}

/// RAII restore of the enclosing thread-local `JIT_JMP_BUF`
/// slot. Ensures the wrapper's buffer pointer doesn't outlive
/// its stack frame — even if the wrapped closure panics for a
/// reason unrelated to the JIT predicate (a bug in a
/// non-JIT sub-path, an OOM, etc.) the guard's `Drop`
/// reinstates the previous slot so the next `invoke_with_catch`
/// call doesn't see a dangling pointer.
struct JmpBufGuard {
    prev: Option<*mut JitJmpBuf>,
}

impl Drop for JmpBufGuard {
    fn drop(&mut self) {
        JIT_JMP_BUF.with(|b| b.set(self.prev));
    }
}

/// Wrapper the kernels, cones, and hybrid segments run fallible
/// native code under (code that calls a helper); code with no
/// helper call runs bare. Sets up the setjmp sentinel, runs the
/// closure (which calls into JIT code), and translates a longjmp
/// return into a Rust panic carrying the violation message. The panic happens in Rust
/// land, so `catch_unwind` catches it normally.
///
/// Both entry/exit paths flow through the [`JmpBufGuard`] so a
/// panic from inside `f()` that isn't a JIT violation still
/// restores the outer slot correctly.
pub(crate) fn invoke_with_catch<F: FnOnce()>(f: F) {
    use std::mem::MaybeUninit;
    let mut buf: MaybeUninit<JitJmpBuf> = MaybeUninit::uninit();
    let buf_ptr = buf.as_mut_ptr();
    // Install the jmp_buf for the duration of the call. The
    // guard restores the previous slot on every exit path
    // (normal return, longjmp, or non-JIT panic unwinding
    // through our frame).
    let prev: Option<*mut JitJmpBuf> = JIT_JMP_BUF.with(|b| b.replace(Some(buf_ptr)));
    let _guard = JmpBufGuard { prev };
    #[cfg(not(windows))]
    let jmpval = unsafe { _setjmp(buf_ptr) };
    // NULL frame → non-unwinding longjmp; see the extern block.
    #[cfg(windows)]
    let jmpval = unsafe { _setjmp(buf_ptr, std::ptr::null_mut()) };
    if jmpval == 0 {
        f();
    } else {
        // longjmp return. The guard will restore the outer
        // slot when this frame exits; drain the violation
        // message and raise a normal Rust panic so the
        // caller's `catch_unwind` can see it.
        let msg = JIT_VIOLATION_MSG
            .with(|m| m.borrow_mut().take())
            .unwrap_or_else(|| "JIT predicate violation (no message)".into());
        // Not `panic!`: the hook already saw the original panic (under
        // `guarded`) and recorded its location for the enrichment the
        // kernel adds; a second hook call would overwrite it.
        std::panic::resume_unwind(Box::new(msg));
    }
}

/// Extern function: longjmp back to the enclosing wrapper with
/// an `is_positive` violation message. Called from JIT code on
/// the predicate-fail path.
extern "C" fn jit_is_positive_fail(value: u64, name_ptr: u64, name_len: u64) -> u64 {
    // The pointer targets the `name` const in the node's NodeMeta;
    // the node is kept alive for the life of the compiled code by
    // `JitCore::_nodes` / the cone node's `members`, so the str
    // data is stable. (ptr, len) == (0, 0) means the default name.
    let name = if name_ptr != 0 {
        unsafe {
            std::str::from_utf8_unchecked(std::slice::from_raw_parts(
                name_ptr as *const u8,
                name_len as usize,
            ))
        }
    } else {
        "value"
    };
    jit_violation_longjmp(format!(
        "is_positive({name}): value must be > 0, got {value}"
    ));
}

/// Extern function: longjmp back to the enclosing wrapper with
/// an `in_range` violation message.
extern "C" fn jit_in_range_fail(value: u64, lo: u64, hi: u64) -> u64 {
    jit_violation_longjmp(format!("in_range: value {value} outside [{lo}, {hi}]"));
}

/// Extern function: the failure a node whose body divides by a wire
/// or constant raises on a zero divisor, in the words the interpreter
/// raises it (Rust's own): `kind` 0 for a quotient, 1 for a remainder.
extern "C" fn jit_div_zero_fail(kind: u64) -> u64 {
    jit_violation_longjmp(
        if kind == 0 {
            "attempt to divide by zero"
        } else {
            "attempt to calculate the remainder with a divisor of zero"
        }
        .to_string(),
    );
}

/// `f64_mod` natively: the body itself, since Rust's `%` on floats is
/// the truncated remainder with the dividend's sign, which no sequence
/// of Cranelift float instructions reproduces for every input.
extern "C" fn jit_f64_mod(a_bits: u64, b_bits: u64) -> u64 {
    let (a, b) = (f64::from_bits(a_bits), f64::from_bits(b_bits));
    (if b != 0.0 { a % b } else { 0.0 }).to_bits()
}

/// Extern function: longjmp back to the enclosing wrapper with
/// an `is_one_of` violation message, carrying the allow-list
/// contents so the message matches the interpreter's byte for
/// byte. The pointer targets the node's meta VecU64 const; the
/// node is kept alive for the life of the compiled code by
/// `JitCore::_nodes` / the cone node's members, so the data is
/// stable. (ptr, len) == (0, 0) degrades to an elided set.
extern "C" fn jit_is_one_of_fail(value: u64, set_ptr: u64, set_len: u64) -> u64 {
    let msg = if set_ptr != 0 {
        let set = unsafe { std::slice::from_raw_parts(set_ptr as *const u64, set_len as usize) };
        format!("is_one_of: value {value} not in allowed set {set:?}")
    } else {
        format!("is_one_of: value {value} not in allowed set [..]")
    };
    jit_violation_longjmp(msg);
}

/// Extern function: weighted pick via alias table (called from JIT code).
///
/// Performs O(1) alias sampling and value lookup. All array pointers
/// are baked as i64 immediates in the JIT code.
extern "C" fn jit_weighted_pick(
    input: u64,
    values_ptr: u64,
    biases_ptr: u64,
    primaries_ptr: u64,
    aliases_ptr: u64,
    n: u64,
) -> u64 {
    guarded(|| {
        let n = n as usize;
        let slot = (input as usize) % n;
        let bias_test = ((input >> 32) as f64) / (u32::MAX as f64);
        unsafe {
            let biases = std::slice::from_raw_parts(biases_ptr as *const f64, n);
            let primaries = std::slice::from_raw_parts(primaries_ptr as *const u64, n);
            let aliases = std::slice::from_raw_parts(aliases_ptr as *const u64, n);
            let values = std::slice::from_raw_parts(values_ptr as *const u64, n);
            let index = if bias_test < biases[slot] {
                primaries[slot]
            } else {
                aliases[slot]
            };
            values[index as usize]
        }
    })
}

/// Run a body that may panic as its P1 node panics inside an
/// `extern "C"` helper, where a panic would abort: the panic is caught
/// and re-raised through the longjmp path, so it surfaces in Rust land
/// as the same panic the P1 node raises.
fn guarded<T>(body: impl FnOnce() -> T) -> T {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(body)) {
        Ok(v) => v,
        Err(payload) => {
            let msg = payload
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_else(|| "panic in a compiled helper".to_string());
            jit_violation_longjmp(msg)
        }
    }
}

/// A node's slot kit as native code holds it: shared by every kernel
/// compiled from the program, compared by identity.
#[derive(Clone)]
pub struct SlotKitRef(pub std::sync::Arc<crate::ast::CompiledSlotKit>);

impl SlotKitRef {
    fn new(kit: crate::ast::CompiledSlotKit) -> Self {
        SlotKitRef(std::sync::Arc::new(kit))
    }
}

impl std::fmt::Debug for SlotKitRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "SlotKitRef({:p}, {} scratch)",
            std::sync::Arc::as_ptr(&self.0),
            self.0.scratch.len()
        )
    }
}

impl PartialEq for SlotKitRef {
    fn eq(&self, other: &Self) -> bool {
        std::sync::Arc::ptr_eq(&self.0, &other.0)
    }
}

/// The helper behind [`JitOp::SlotCall`]: run a kit's closure over
/// inputs gathered into the native frame, outputs to scatter from it,
/// and the state's scratch entries from `base`. A panic in the closure
/// is the node's own failure and is re-raised through the longjmp path
/// like every other helper's.
///
/// # Safety
/// Called only from generated code, which passes the kit the step was
/// compiled with (kept alive by the code that calls it), frame arrays
/// of the stated lengths, and the scratch the calling state owns, laid
/// out as the builder placed the kit's entries.
extern "C" fn jit_slot_call(
    kit: *const crate::ast::CompiledSlotKit,
    inputs: *const u64,
    n_in: u64,
    outputs: *mut u64,
    n_out: u64,
    scratch: *mut crate::ast::ScratchBuf,
    base: u64,
    n_scratch: u64,
) {
    guarded(|| unsafe {
        let kit = &*kit;
        let ins = std::slice::from_raw_parts(inputs, n_in as usize);
        let outs = std::slice::from_raw_parts_mut(outputs, n_out as usize);
        let sc = std::slice::from_raw_parts_mut(scratch.add(base as usize), n_scratch as usize);
        (kit.op)(ins, outs, sc)
    })
}

impl JitOp {
    /// The kit a slot call runs, if this is one.
    pub(crate) fn slot_kit(&self) -> Option<&SlotKitRef> {
        match self {
            JitOp::SlotCall { kit, .. } => Some(kit),
            _ => None,
        }
    }

    /// The scratch entries the step needs in the state that runs it.
    pub(crate) fn scratch_elems(&self) -> &[crate::ast::ScratchElem] {
        const STR_ENTRY: [crate::ast::ScratchElem; 1] = [crate::ast::ScratchElem::Str];
        const F32_ENTRY: [crate::ast::ScratchElem; 1] = [crate::ast::ScratchElem::F32];
        match self {
            JitOp::SlotCall { kit, .. } => &kit.0.scratch,
            JitOp::U64ToStr { .. }
            | JitOp::I64ToStr { .. }
            | JitOp::F64ToStr { .. }
            | JitOp::StrConcat { .. }
            | JitOp::JsonToStr { .. } => &STR_ENTRY,
            JitOp::VecProduce { .. } => &F32_ENTRY,
            _ => &[],
        }
    }

    /// Place the step's scratch entries at `base` in the state's
    /// scratch; the builder that lays the state out calls this once.
    pub(crate) fn place_scratch(&mut self, base: usize) {
        match self {
            JitOp::SlotCall { scratch_base, .. }
            | JitOp::U64ToStr { scratch_base }
            | JitOp::I64ToStr { scratch_base }
            | JitOp::F64ToStr { scratch_base }
            | JitOp::StrConcat { scratch_base }
            | JitOp::JsonToStr { scratch_base }
            | JitOp::VecProduce { scratch_base, .. } => *scratch_base = base,
            _ => {}
        }
    }
}

/// The vector producers with a named lowering (compiled_handles.md
/// §6): each writes its result into the step's own `F32` entry and
/// publishes the pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VecProducer {
    /// `vec_add(a, b)` over two `vec_f32` wires.
    Add,
    /// `vec_scale(a, k)` over a `vec_f32` and an `f64` wire.
    Scale,
    /// `vec_norm(a)` over a `vec_f32` wire.
    Norm,
    /// `hash_vec(seed, dim)`.
    HashVec,
    /// `xxhash3_vec(seed, dim)`.
    XxHash3Vec,
    /// `reg_to_vec_f32(r)`.
    RegToVec,
}

/// The vector reductions with a named lowering: each returns the
/// bits of its `f64` result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VecReducer {
    /// `vec_dot(a, b)`.
    Dot,
    /// `vec_l2(a, b)`.
    L2,
    /// `vec_cosine(a, b)`.
    Cosine,
    /// `lid_mle(distances, k)`.
    LidMle,
}

/// The register lane reads with a named lowering: each returns the
/// lane as the slot word its port stores.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegLaneRead {
    /// `reg_lane_f32(r, i)`, widened to f64.
    F32,
    /// `reg_lane_i16(r, i)`, sign-extended.
    I16,
    /// `reg_lane_i64(r, i)`.
    I64,
}

/// The register producers that run through a helper: a bounds check
/// on a wire index, or a lane operation Cranelift has no instruction
/// for. Each writes its word into the output slots.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegProducer {
    /// `reg_with_lane_f32(r, i, v)`.
    WithLaneF32,
    /// `reg_gather_f32(v, offset)`.
    GatherF32,
    /// `vec_to_reg_f32(v)`.
    VecToRegF32,
    /// `reg_mul_i8(a, b)`.
    MulI8,
}

/// The `vec_f32` slice a wire's pair names.
///
/// # Safety
/// The pair was published by its producing step into storage alive
/// until that step reruns (axioms S3, S4), and the wire is a `vec_f32`.
unsafe fn vec_f32_of<'a>(ptr: u64, len: u64) -> &'a [f32] {
    if len == 0 {
        &[]
    } else {
        unsafe { std::slice::from_raw_parts(ptr as usize as *const f32, len as usize) }
    }
}

/// Write a vector into the step's own `F32` entry and publish its
/// pair into the output slots: what every vector producer's lowering
/// does around its arithmetic. `f` fills the entry.
///
/// # Safety
/// As for [`write_str_entry`].
unsafe fn write_f32_entry(
    scratch: *mut crate::ast::ScratchBuf,
    base: u64,
    buffer: *mut u64,
    out_slot: u64,
    f: impl FnOnce(&mut Vec<f32>),
) {
    unsafe {
        let entry = &mut *scratch.add(base as usize);
        let crate::ast::ScratchBuf::F32(v) = entry else {
            panic!("a vector lowering's scratch entry is not an f32 vector");
        };
        f(v);
        *buffer.add(out_slot as usize) = v.as_ptr() as usize as u64;
        *buffer.add(out_slot as usize + 1) = v.len() as u64;
    }
}

/// The vector producers natively: the step's input words in order
/// (`w0..w3`, zero past the last), the same body the node runs, into
/// the entry.
macro_rules! vec_producer {
    ($name:ident, |$out:ident, $w0:ident, $w1:ident, $w2:ident, $w3:ident| $body:expr) => {
        extern "C" fn $name(
            scratch: *mut crate::ast::ScratchBuf,
            base: u64,
            buffer: *mut u64,
            out_slot: u64,
            $w0: u64,
            $w1: u64,
            $w2: u64,
            $w3: u64,
        ) {
            guarded(|| unsafe {
                let _ = ($w2, $w3);
                write_f32_entry(scratch, base, buffer, out_slot, |$out| $body)
            })
        }
    };
}

vec_producer!(jit_vec_add, |out, a_ptr, a_len, b_ptr, b_len| {
    let (a, b) = (vec_f32_of(a_ptr, a_len), vec_f32_of(b_ptr, b_len));
    crate::numeric::vector::check_lens("vec_add", a.len(), b.len());
    crate::numeric::vector::add_f32_into(a, b, out)
});
vec_producer!(jit_vec_scale, |out, a_ptr, a_len, k_bits, _z| {
    let a = vec_f32_of(a_ptr, a_len);
    crate::numeric::vector::scale_f32_into(a, f64::from_bits(k_bits) as f32, out)
});
vec_producer!(jit_vec_norm, |out, a_ptr, a_len, _y, _z| {
    crate::numeric::vector::norm_f32_into(vec_f32_of(a_ptr, a_len), out)
});
vec_producer!(jit_hash_vec, |out, seed, dim, _y, _z| {
    crate::numeric::vector::hash_vec_into(seed, dim, out)
});
vec_producer!(jit_xxhash3_vec, |out, seed, dim, _y, _z| {
    crate::numeric::vector::xxhash3_vec_into(seed, dim, out)
});
vec_producer!(jit_reg_to_vec_f32, |out, lo, hi, _y, _z| {
    out.clear();
    out.extend_from_slice(&crate::ast::Bits128([lo, hi]).lanes_f32())
});

/// The vector reductions natively: the input words in order, the
/// result's bits back.
macro_rules! vec_reducer {
    ($name:ident, |$w0:ident, $w1:ident, $w2:ident, $w3:ident| $body:expr) => {
        extern "C" fn $name($w0: u64, $w1: u64, $w2: u64, $w3: u64) -> u64 {
            guarded(|| unsafe {
                let _ = ($w2, $w3);
                let r: f64 = $body;
                r.to_bits()
            })
        }
    };
}

vec_reducer!(jit_vec_dot, |a_ptr, a_len, b_ptr, b_len| {
    let (a, b) = (vec_f32_of(a_ptr, a_len), vec_f32_of(b_ptr, b_len));
    crate::numeric::vector::check_lens("vec_dot", a.len(), b.len());
    crate::numeric::vector::dot_f32(a, b) as f64
});
vec_reducer!(jit_vec_l2, |a_ptr, a_len, b_ptr, b_len| {
    let (a, b) = (vec_f32_of(a_ptr, a_len), vec_f32_of(b_ptr, b_len));
    crate::numeric::vector::check_lens("vec_l2", a.len(), b.len());
    (crate::numeric::vector::l2sq_f32(a, b) as f64).sqrt()
});
vec_reducer!(jit_vec_cosine, |a_ptr, a_len, b_ptr, b_len| {
    let (a, b) = (vec_f32_of(a_ptr, a_len), vec_f32_of(b_ptr, b_len));
    crate::numeric::vector::check_lens("vec_cosine", a.len(), b.len());
    crate::numeric::vector::cosine_f32(a, b)
});
vec_reducer!(jit_lid_mle, |d_ptr, d_len, k_bits, _z| {
    crate::numeric::vector::lid_mle_of(vec_f32_of(d_ptr, d_len), f64::from_bits(k_bits))
});

/// `reg_lane_f32` natively: the lane widened, as f64 bits.
extern "C" fn jit_reg_lane_f32(lo: u64, hi: u64, i: u64) -> u64 {
    guarded(|| crate::numeric::register::lane_f32(crate::ast::Bits128([lo, hi]), i).to_bits())
}

/// `reg_lane_i16` natively: the lane sign-extended into the slot word.
extern "C" fn jit_reg_lane_i16(lo: u64, hi: u64, i: u64) -> u64 {
    guarded(|| crate::numeric::register::lane_i16(crate::ast::Bits128([lo, hi]), i) as i64 as u64)
}

/// `reg_lane_i64` natively.
extern "C" fn jit_reg_lane_i64(lo: u64, hi: u64, i: u64) -> u64 {
    guarded(|| crate::numeric::register::lane_i64(crate::ast::Bits128([lo, hi]), i) as u64)
}

/// The register producers natively: the input words in order, the
/// word into the two output slots.
macro_rules! reg_producer {
    ($name:ident, |$w0:ident, $w1:ident, $w2:ident, $w3:ident| $body:expr) => {
        extern "C" fn $name(
            buffer: *mut u64,
            out_slot: u64,
            $w0: u64,
            $w1: u64,
            $w2: u64,
            $w3: u64,
        ) {
            guarded(|| unsafe {
                let _ = ($w2, $w3);
                let r: crate::ast::Bits128 = $body;
                *buffer.add(out_slot as usize) = r.0[0];
                *buffer.add(out_slot as usize + 1) = r.0[1];
            })
        }
    };
}

reg_producer!(jit_reg_with_lane_f32, |lo, hi, i, v_bits| {
    crate::numeric::register::with_lane_f32(
        crate::ast::Bits128([lo, hi]),
        i,
        f64::from_bits(v_bits),
    )
});
reg_producer!(jit_reg_gather_f32, |v_ptr, v_len, offset, _z| {
    crate::numeric::register::gather_f32(vec_f32_of(v_ptr, v_len), offset)
});
reg_producer!(jit_vec_to_reg_f32, |v_ptr, v_len, _y, _z| {
    crate::numeric::register::to_reg_f32(vec_f32_of(v_ptr, v_len))
});
reg_producer!(jit_reg_mul_i8, |a_lo, a_hi, b_lo, b_hi| {
    crate::numeric::register::mul_i8(
        crate::ast::Bits128([a_lo, a_hi]),
        crate::ast::Bits128([b_lo, b_hi]),
    )
});

/// Write a string into the step's own entry and publish its pair into
/// the output slots: what every named string lowering does around its
/// formatting. `f` fills the cleared entry.
///
/// # Safety
/// Called only from the helpers below, with the state's scratch and
/// the entry index the builder placed for the step, and the buffer and
/// output slot the step writes.
unsafe fn write_str_entry(
    scratch: *mut crate::ast::ScratchBuf,
    base: u64,
    buffer: *mut u64,
    out_slot: u64,
    f: impl FnOnce(&mut Vec<u8>),
) {
    unsafe {
        let entry = &mut *scratch.add(base as usize);
        let crate::ast::ScratchBuf::Str(v) = entry else {
            panic!("a string lowering's scratch entry is not a string");
        };
        v.clear();
        f(v);
        *buffer.add(out_slot as usize) = v.as_ptr() as usize as u64;
        *buffer.add(out_slot as usize + 1) = v.len() as u64;
    }
}

/// `__u64_to_string` natively: the digits straight into the entry.
extern "C" fn jit_u64_to_str(
    scratch: *mut crate::ast::ScratchBuf,
    base: u64,
    buffer: *mut u64,
    out_slot: u64,
    value: u64,
) {
    use std::io::Write;
    guarded(|| unsafe {
        write_str_entry(scratch, base, buffer, out_slot, |v| {
            write!(v, "{value}").expect("a vector accepts every write")
        })
    })
}

/// `__i64_to_string` natively.
extern "C" fn jit_i64_to_str(
    scratch: *mut crate::ast::ScratchBuf,
    base: u64,
    buffer: *mut u64,
    out_slot: u64,
    value: u64,
) {
    use std::io::Write;
    guarded(|| unsafe {
        write_str_entry(scratch, base, buffer, out_slot, |v| {
            write!(v, "{}", value as i64).expect("a vector accepts every write")
        })
    })
}

/// `__f64_to_string` natively: `Display`, which is what `to_string`
/// writes on the interpreter.
extern "C" fn jit_f64_to_str(
    scratch: *mut crate::ast::ScratchBuf,
    base: u64,
    buffer: *mut u64,
    out_slot: u64,
    bits: u64,
) {
    use std::io::Write;
    guarded(|| unsafe {
        write_str_entry(scratch, base, buffer, out_slot, |v| {
            write!(v, "{}", f64::from_bits(bits)).expect("a vector accepts every write")
        })
    })
}

/// `str_concat` over string wires natively: every pair's bytes
/// appended in order. `pairs` holds `n` `(ptr, len)` pairs the
/// generated code stored into its frame.
extern "C" fn jit_str_concat(
    scratch: *mut crate::ast::ScratchBuf,
    base: u64,
    buffer: *mut u64,
    out_slot: u64,
    pairs: *const u64,
    n: u64,
) {
    guarded(|| unsafe {
        let words = std::slice::from_raw_parts(pairs, 2 * n as usize);
        write_str_entry(scratch, base, buffer, out_slot, |v| {
            for pair in words.as_chunks::<2>().0 {
                // SAFETY: each pair was published by its producing step
                // into storage alive until that step reruns (axioms S3,
                // S4), and the wire is a string.
                let bytes =
                    std::slice::from_raw_parts(pair[0] as usize as *const u8, pair[1] as usize);
                v.extend_from_slice(bytes);
            }
        })
    })
}

/// `json_to_str` natively: the compact serialization straight into
/// the entry, the bytes `serde_json::Value::to_string` produces.
extern "C" fn jit_json_to_str(
    scratch: *mut crate::ast::ScratchBuf,
    base: u64,
    buffer: *mut u64,
    out_slot: u64,
    ptr: u64,
    len: u64,
) {
    guarded(|| unsafe {
        let pair = [ptr, len];
        let value = crate::derive_support::ref_value(&pair);
        let json = match value {
            crate::ast::Value::Json(j) => j.as_ref(),
            other => panic!("expected Json wire, got {other:?}"),
        };
        write_str_entry(scratch, base, buffer, out_slot, |v| {
            serde_json::to_writer(v, json).expect("a vector accepts every write")
        })
    })
}

/// Classify a node with the types of its wire inputs known. A node
/// with a named native lowering takes it; any other node with a kit
/// is a [`JitOp::SlotCall`] of that kit, so a reference pair on either
/// side is no bar to native code, and neither is nondeterminism or a
/// side effect: the kernels that run the code keep such a step's
/// currency its own (a segment of its own on the hybrid kernel, a
/// never-current step on pure native code). A node with neither stays
/// interpreted.
pub fn classify_node_typed(node: &dyn PolydatNode, wire_types: &[crate::ast::PortType]) -> JitOp {
    use crate::ast::PortType as PT;
    let is_ref = |t: &crate::ast::PortType| t.slot_color() == crate::ast::SlotColor::Ref2;
    let vec_produce = |kind: VecProducer| JitOp::VecProduce {
        kind,
        scratch_base: 0,
    };
    let ref_copy = |ty: crate::ast::PortType| {
        crate::compile::assembly::ref_copy_kit(ty)
            .map(|kit| JitOp::SlotCall {
                kit: SlotKitRef::new(kit),
                scratch_base: 0,
            })
            .unwrap_or(JitOp::Fallback)
    };
    let meta = node.meta();
    let named = match meta.name.as_str() {
        // The string producers with a lowering that writes straight
        // into the step's entry (compiled_handles.md §6).
        "__u64_to_string" => JitOp::U64ToStr { scratch_base: 0 },
        "__i64_to_string" => JitOp::I64ToStr { scratch_base: 0 },
        "__f64_to_string" => JitOp::F64ToStr { scratch_base: 0 },
        "json_to_str" if wire_types == [crate::ast::PortType::Json] => {
            JitOp::JsonToStr { scratch_base: 0 }
        }
        "str_concat"
            if !wire_types.is_empty()
                && wire_types.iter().all(|t| *t == crate::ast::PortType::Str) =>
        {
            JitOp::StrConcat { scratch_base: 0 }
        }
        // The vector group: each lowering runs the node's own body on
        // the wires' slices and writes into the step's entry or
        // returns its scalar. Keyed on the exact wire types the body
        // is written for; any adapted shape takes the kit.
        "vec_add" if wire_types == [PT::VecF32, PT::VecF32] => vec_produce(VecProducer::Add),
        "vec_scale" if wire_types == [PT::VecF32, PT::F64] => vec_produce(VecProducer::Scale),
        "vec_norm" if wire_types == [PT::VecF32] => vec_produce(VecProducer::Norm),
        "hash_vec" if wire_types == [PT::U64, PT::U64] => vec_produce(VecProducer::HashVec),
        "xxhash3_vec" if wire_types == [PT::U64, PT::U64] => vec_produce(VecProducer::XxHash3Vec),
        "reg_to_vec_f32" if wire_types == [PT::RegF32x4] => vec_produce(VecProducer::RegToVec),
        "vec_dot" if wire_types == [PT::VecF32, PT::VecF32] => JitOp::VecReduce(VecReducer::Dot),
        "vec_l2" if wire_types == [PT::VecF32, PT::VecF32] => JitOp::VecReduce(VecReducer::L2),
        "vec_cosine" if wire_types == [PT::VecF32, PT::VecF32] => {
            JitOp::VecReduce(VecReducer::Cosine)
        }
        "lid_mle" if wire_types == [PT::VecF32, PT::F64] => JitOp::VecReduce(VecReducer::LidMle),
        // The register group: the lane reads and the producers with a
        // bounds check run through a helper; the dot product and the
        // byte shuffle are inline vector instructions.
        "reg_lane_f32" if wire_types == [PT::RegF32x4, PT::U64] => JitOp::RegLane(RegLaneRead::F32),
        "reg_lane_i16" if wire_types == [PT::RegI16x8, PT::U64] => JitOp::RegLane(RegLaneRead::I16),
        "reg_lane_i64" if wire_types == [PT::RegI64x2, PT::U64] => JitOp::RegLane(RegLaneRead::I64),
        "reg_with_lane_f32" if wire_types == [PT::RegF32x4, PT::U64, PT::F64] => {
            JitOp::RegProduce(RegProducer::WithLaneF32)
        }
        "reg_gather_f32" if wire_types == [PT::VecF32, PT::U64] => {
            JitOp::RegProduce(RegProducer::GatherF32)
        }
        "vec_to_reg_f32" if wire_types == [PT::VecF32] => {
            JitOp::RegProduce(RegProducer::VecToRegF32)
        }
        "reg_mul_i8" if wire_types == [PT::RegI8x16, PT::RegI8x16] => {
            JitOp::RegProduce(RegProducer::MulI8)
        }
        "reg_dot_f32" if wire_types == [PT::RegF32x4, PT::RegF32x4] => JitOp::RegDotF32,
        // The compiler's input passthrough and `default_or(value,
        // fallback)` (`value` unless it is `None`, which a compiled slot
        // never carries; engines.md §3.3): a slot copy of an
        // immediate, a copy into the step's own scratch of a reference
        // value (axiom S3: a pair is never forwarded).
        n if n.starts_with("__port_") || n == "default_or" => match meta.outs.first() {
            Some(o) if is_ref(&o.typ) => return ref_copy(o.typ),
            _ => JitOp::Identity,
        },
        // The named selects are native only between one-slot
        // immediates; any other shape takes the node's kit below.
        "select" | "select_u64" if wire_types.iter().skip(1).any(|t| t.slot_width() != 1) => {
            JitOp::Fallback
        }
        _ if wire_types.iter().any(is_ref) || meta.outs.iter().any(|o| is_ref(&o.typ)) => {
            JitOp::Fallback
        }
        _ => classify_node(node),
    };
    if !matches!(named, JitOp::Fallback) {
        return named;
    }
    if let Some(kit) = node.compiled_slot(
        wire_types,
        crate::compile::select::Engine::Native(crate::compile::select::Provenance::Auto),
    ) {
        return JitOp::SlotCall {
            kit: SlotKitRef::new(kit),
            scratch_base: 0,
        };
    }
    if let Some(op) = node.compiled_u64() {
        return JitOp::SlotCall {
            kit: SlotKitRef::new(crate::ast::CompiledSlotKit {
                scratch: Vec::new(),
                op: Box::new(move |inputs, outputs, _| op(inputs, outputs)),
            }),
            scratch_base: 0,
        };
    }
    JitOp::Fallback
}

// ── JitOp ──────────────────────────────────────────────────

/// Description of a JIT step — what operation to generate.
///
/// For f64 operations, values are stored in the u64 buffer as their
/// bit representation. Cranelift `bitcast` converts between i64/f64.
#[derive(Debug, Clone, PartialEq)]
pub enum JitOp {
    // --- u64 integer ops ---
    /// `output[i] = input[i]` for every slot the port spans  (identity / copy)
    Identity,
    /// `output[0] = input[0] + constant`
    AddConst(u64),
    /// `output[0] = input[0] * constant`
    MulConst(u64),
    /// `output[0] = input[0] / constant`
    DivConst(u64),
    /// `output[0] = input[0] % constant`
    ModConst(u64),
    /// `output[0] = clamp(input[0], min, max)`  (unsigned)
    ClampConst(u64, u64),
    /// `output[0] = interleave_bits(input[0], input[1])`  (extern call)
    Interleave,
    /// `output[i] = mixed-radix decomposition of input[0]`  (inline urem/udiv)
    MixedRadixConst(Vec<u64>),
    /// `output[0] = xxh3_hash(input[0])`  (extern call)
    Hash,
    /// `output[0] = splitmix64(input[0]) (fully inlined 64-bit ALU bit mixer)`
    SplitMix64,
    /// `output[0] = shuffle(input[0])`  (extern call: feedback, size, min)
    ShuffleConst(u64, u64, u64),

    // --- f64 ops (values stored as u64 bits in buffer) ---
    /// `output[0] = input[0] as f64 / u64::MAX as f64`  (u64 → f64 bits)
    UnitInterval,
    /// `output[0] = f64::from_bits(input[0]) as u64`  (f64 bits → u64, truncate)
    F64ToU64,
    /// `output[0] = f64::from_bits(input[0]).round() as u64`: half
    /// away from zero, as Rust rounds, then the saturating conversion.
    RoundToU64,
    /// `output[0] = f64::from_bits(input[0]).floor() as u64`
    FloorToU64,
    /// `output[0] = f64::from_bits(input[0]).ceil() as u64`
    CeilToU64,
    /// `output[0] = clamp(f64::from_bits(input[0]), min, max)`  → f64 bits
    ClampF64Const(u64, u64), // min.to_bits(), max.to_bits()
    /// `output[0] = a + (b - a) * f64::from_bits(input[0])`  → f64 bits
    LerpConst(u64, u64), // a.to_bits(), b.to_bits()
    /// `output[0] = min + range * (input[0] as f64 / MAX)`  → f64 bits  (u64 input)
    ScaleRangeConst(u64, u64), // min.to_bits(), range.to_bits()
    /// `output[0] = round(f64::from_bits(input[0]) / step) * step`  → f64 bits
    QuantizeConst(u64), // step.to_bits()
    /// `output[0] = discretize(f64 input, range, buckets)`  → u64
    DiscretizeConst(u64, u64), // range.to_bits(), buckets
    /// `output[0] = lut_sample(f64 input, lut_ptr, lut_len)`  → f64 bits  (extern call)
    LutSampleConst(u64, u64), // lut_ptr as u64, lut_len
    /// `output[0] = weighted_pick(input, values_ptr, biases_ptr, primaries_ptr, aliases_ptr, n)`
    WeightedPickConst(u64, u64, u64, u64, u64), // values_ptr, biases_ptr, primaries_ptr, aliases_ptr, n

    /// Unary f64 math function via extern call. The u8 identifies which function.
    /// 0=sin 1=cos 2=tan 3=asin 4=acos 5=atan 6=sqrt 7=abs 8=ln 9=exp
    /// 10=floor_base10 11=ceiling_base10 12=closest_base10
    /// 13=floor_decade 14=ceiling_decade 15=closest_decade
    /// 16=floor_binomial 17=ceiling_binomial 18=closest_binomial
    /// 19=floor_fibonacci 20=ceiling_fibonacci 21=closest_fibonacci
    MathUnary(u8),
    /// Binary f64 math function via extern call.
    /// 0=atan2 1=pow 2=round_nearest 3=round_floor 4=round_ceiling
    MathBinary(u8),

    // --- Two-wire u64 integer ops ---
    /// output = `input[0]` + `input[1]`  (wrapping)
    U64Add2,
    /// output = `input[0]` - `input[1]`  (wrapping)
    U64Sub2,
    /// output = `input[0]` * `input[1]`  (wrapping)
    U64Mul2,
    /// output = `input[0]` / `input[1]`  (0 if divisor is 0)
    U64Div2,
    /// output = `input[0]` % `input[1]`  (0 if divisor is 0)
    U64Mod2,
    /// output = `input[0]` & `input[1]`
    U64And,
    /// output = `input[0]` | `input[1]`
    U64Or,
    /// output = `input[0]` ^ `input[1]`
    U64Xor,
    /// output = `input[0]` << `input[1]`
    U64Shl,
    /// output = `input[0]` >> `input[1]`  (logical)
    U64Shr,
    /// output = !`input[0]`  (unary bitwise NOT)
    U64Not,

    // --- Inline binary f64 arithmetic (no extern call) ---
    /// output = input as f64 (integer to float conversion, not bit reinterpret)
    ToF64,

    /// output = f64(a) + f64(b)
    F64Add,
    /// output = f64(a) - f64(b)
    F64Sub,
    /// output = f64(a) * f64(b)
    F64Mul,
    /// output = f64(a) / f64(b) (0 if b==0)
    F64Div,
    /// output = f64(a) % f64(b) (0 if b==0), through `jit_f64_mod`
    F64Mod,
    /// `output[0] = input[0] / input[1]`, failing on a zero divisor as
    /// the body's `/` does (`div_wire`)
    U64DivWire,
    /// `output[0] = input[0] % input[1]`, failing on a zero divisor as
    /// the body's `%` does (`mod_wire`)
    U64ModWire,

    /// A call of the node's own slot kit from native code
    /// (compiled_handles.md §6): the inputs are gathered into the
    /// frame, `jit_slot_call` runs the kit's closure over them and the
    /// state's scratch entries at `scratch_base`, and the outputs are
    /// scattered back. Every node with a kit lowers this way, so a
    /// reference pair rides through a segment or a cone as it rides
    /// through a closure step.
    SlotCall {
        /// The kit, shared by every kernel compiled from the program
        /// and kept alive by the code that calls it.
        kit: SlotKitRef,
        /// Index of the kit's first scratch entry in the state's
        /// scratch, assigned by the builder that lays the state out.
        scratch_base: usize,
    },

    // --- Named lowerings that write a string into the step's own
    // entry (compiled_handles.md §6): no intermediate `String`, no
    // frame, the pair published by the helper. Each owns one `Str`
    // scratch entry at `scratch_base`.
    /// `output = decimal digits of input[0] as a u64`
    U64ToStr {
        /// The step's string entry in the state's scratch.
        scratch_base: usize,
    },
    /// `output = decimal digits of input[0] as an i64`
    I64ToStr {
        /// The step's string entry in the state's scratch.
        scratch_base: usize,
    },
    /// `output = Display form of input[0] as an f64`
    F64ToStr {
        /// The step's string entry in the state's scratch.
        scratch_base: usize,
    },
    /// `output = the concatenation of every input pair's bytes`, for
    /// a `str_concat` whose wires are all strings.
    StrConcat {
        /// The step's string entry in the state's scratch.
        scratch_base: usize,
    },
    /// `output = compact serialization of the JSON value input[0..2] names`
    JsonToStr {
        /// The step's string entry in the state's scratch.
        scratch_base: usize,
    },

    // --- The vector and register groups (compiled_handles.md §6) ---
    /// `output = a vec_f32 written into the step's own `F32` entry`
    /// by the producer's body over the input words.
    VecProduce {
        /// Which producer.
        kind: VecProducer,
        /// The step's `F32` entry in the state's scratch.
        scratch_base: usize,
    },
    /// `output[0] = f64 bits of the reduction over the input words`
    VecReduce(VecReducer),
    /// `output[0] = lane input[2] of the register word input[0..2]`,
    /// bounds-checked by the helper.
    RegLane(RegLaneRead),
    /// `output[0..2] = the producer's word over the input words`
    RegProduce(RegProducer),
    /// `output[0] = ((a0*b0 + a1*b1) + (a2*b2 + a3*b3)) as f64`, the
    /// fixed tree of `reg_dot_f32`, over f32x4 words: one `fmul`,
    /// four lane extracts, three adds, one promotion.
    RegDotF32,
    /// `output[0..2] = byte permutation of input[0..2]` by a baked
    /// 16-entry mask, one `shuffle`.
    RegShuffleConst([u8; 16]),

    /// Parameter predicate: pass `input[0]` through to `output[0]`;
    /// if `input[0]` == 0, call `jit_is_positive_fail` (panics)
    /// with the configured predicate name — (ptr, len) into the
    /// node's meta const, (0, 0) for the default. Message parity
    /// with the interpreter's `is_positive({name}): …` is asserted
    /// by the SRD-105 battery.
    IsPositiveCheck {
        /// Address of the predicate's name, or 0 for the default.
        name_ptr: u64,
        /// Its length in bytes.
        name_len: u64,
    },
    /// Parameter predicate: pass `input[0]` through to `output[0]`;
    /// if `input[0]` < lo or `input[0]` > hi, call
    /// `jit_in_range_fail` (panics). Stored as (lo, hi).
    InRangeCheck(u64, u64),
    /// Parameter predicate: pass `input[0]` through to `output[0]`;
    /// if `input[0]` is not in the allow-list, call
    /// `jit_is_one_of_fail` (panics) with the allow-list contents
    /// — (ptr, len) into the node's meta VecU64 const, (0, 0)
    /// when unavailable. Message parity with the interpreter's
    /// `is_one_of: … not in allowed set […]` is asserted by the
    /// SRD-105 battery. Inline comparisons use the baked vector.
    IsOneOfCheck {
        /// The allow-list, baked into the comparisons.
        allowed: Vec<u64>,
        /// Address of the node's allow-list constant for the message, or 0.
        set_ptr: u64,
        /// Its length.
        set_len: u64,
    },

    // --- Register-plane ops (type_system_alignment.md §3, native tier of §7) ---
    // A register value occupies two consecutive u64 slots; the
    // codegen emits one unaligned 128-bit load/store per value
    // (buffer is only 8-aligned) and a single vector instruction.
    /// Element-wise register binop. (lane_ty index, arith index)
    /// — lanes: 0=i8x16 1=i16x8 2=i32x4 3=i64x2 4=f32x4 5=f64x2;
    /// arith: 0=add 1=sub 2=mul.
    RegBinOp(u8, u8),
    /// View retag / two-slot copy (`__reg_view_*`): one 128-bit
    /// load + store; the lane typing is static, so no instruction
    /// beyond the move.
    RegCopy,
    /// Broadcast a scalar wire into all lanes. Same lane index
    /// vocabulary as `RegBinOp`; float lanes read the f64 slot
    /// and demote as needed, integer lanes reduce from u64.
    RegSplat(u8),

    // --- Comparisons & selections (SRD 110) ---
    /// Integer comparison: `output[0]` = if a `<cond>` b { 1 } else { 0 }
    U64Cmp(ir::condcodes::IntCC),
    /// Float comparison: `output[0]` = if a `<cond>` b { 1 } else { 0 }
    F64Cmp(ir::condcodes::FloatCC),
    /// Conditional select for u64: `output[0]` = if cond != 0 { a } else { b }
    SelectU64,
    /// Conditional select for f64: `output[0]` = if cond != 0 { a } else { b }
    SelectF64,

    // --- Type conversions & lattice adapters (SRD 110) ---
    /// Signed integer to float: `output[0]` = (`input[0]` as i64 as f64).to_bits()
    I64ToF64,
    /// Float to signed integer: `output[0]` = (f64::from_bits(`input[0]`) as i64) as u64
    F64ToI64,
    /// Sign-extend 32-bit integer: `output[0]` = ((`input[0]` as i32) as i64) as u64
    SignExtendI32,
    /// Sign-extend 16-bit integer: `output[0]` = ((`input[0]` as i16) as i64) as u64
    SignExtendI16,
    /// Sign-extend 8-bit integer: `output[0]` = ((`input[0]` as i8) as i64) as u64
    SignExtendI8,
    /// Zero-extend 32-bit integer: `output[0]` = (`input[0]` as u32) as u64
    ZeroExtendU32,
    /// Zero-extend 16-bit integer: `output[0]` = (`input[0]` as u16) as u64
    ZeroExtendU16,
    /// Zero-extend 8-bit integer: `output[0]` = (`input[0]` as u8) as u64
    ZeroExtendU8,
    /// Truthiness boolean coercion: `output[0]` = if `input[0]` != 0 { 1 } else { 0 }
    ToBool,
    /// Constant u64: `output[0]` = val
    ConstU64(u64),
    /// Constant f64: `output[0]` = val_bits
    ConstF64(u64),

    // --- Interpolation & Hashing (SRD 110) ---
    /// Hash range: `output[0]` = if max == 0 { 0 } else { hash(`input[0]`) % max }
    HashRangeConst(u64),
    /// Hash interval: `output[0]` = min + (hash(`input[0]`) / MAX) * (max - min)
    HashIntervalConst(u64, u64),
    /// Inverse lerp: `output[0]` = ((`input[0]` - a) / (b - a)).clamp(0, 1)
    InvLerpConst(u64, u64),
    /// Remap: `output[0]` = out_min + ((`input[0]` - in_min) / (in_max - in_min)) * (out_max - out_min)
    RemapConst(u64, u64, u64, u64),

    // --- Context & Datetime (SRD 110) ---
    /// Epoch offset: `output[0]` = `input[0]`.wrapping_add(base)
    EpochOffsetConst(u64),
    /// Epoch scale: `output[0]` = `input[0]`.wrapping_mul(factor)
    EpochScaleConst(u64),
    /// OS thread ID
    ThreadId,
    /// Wall clock millis
    CurrentEpochMillis,

    // --- Coherent Noise (SRD 110) ---
    /// `output[0] = jit_perlin_1d(input[0], perm, freq)`: (permutation table address, frequency bits).
    Perlin1dConst(u64, u64),
    /// `jit_perlin_2d` over two inputs: (permutation table address, frequency bits).
    Perlin2dConst(u64, u64),
    /// `jit_simplex_2d` over two inputs: (permutation table address, frequency bits).
    Simplex2dConst(u64, u64),
    /// `jit_fractal_noise_1d`: (permutation table address, frequency bits, octaves).
    FractalNoise1dConst(u64, u64, u64),
    /// `jit_fractal_noise_2d` over two inputs: (permutation table address, frequency bits, octaves).
    FractalNoise2dConst(u64, u64, u64),

    // --- Variadics & wire arithmetic (SRD 110) ---
    /// Variadic sum across all inputs
    VariadicSum,
    /// Variadic product across all inputs
    VariadicProduct,
    /// Variadic minimum across all inputs (unsigned)
    VariadicMin,
    /// Variadic maximum across all inputs (unsigned)
    VariadicMax,
    /// Checked unsigned addition: `output[0]` = a.checked_add(b).unwrap_or(0)
    CheckedAdd,
    /// Saturating unsigned subtraction: `output[0]` = a.saturating_sub(b)
    CheckedSub,
    /// Checked unsigned multiplication: `output[0]` = a.checked_mul(b).unwrap_or(0)
    CheckedMul,
    /// Smallest multiple of multiple >= value: `output[0]` = if m == 0 { v } else { v.div_ceil(m).saturating_mul(m) }
    CeilToMultiple,
    /// Multiples at least: `output[0]` = if m == 0 { 0 } else { v.div_ceil(m) }
    MultiplesAtLeast,

    // --- Probability & permutations (SRD 110) ---
    /// Fair coin flip: `output[0]` = `input[0]` & 1
    FairCoin,
    /// Float blend with constant mix: `output[0]` = (fa * (1 - mix) + fb * mix).round() as u64
    BlendConst(u64),
    /// LFSR advance step with constant feedback polynomial:
    /// `output[0] = (input[0] >> 1) ^ (if input[0] & 1 != 0 { feedback } else { 0 })`
    LfsrStepConst(u64),
    /// PCG random with constant seed and stream: (seed, stream)
    PcgConst(u64, u64),
    /// PCG random with wire stream and constant seed: (seed)
    PcgStreamConst(u64),
    /// Cycle walk: (range, seed, inc)
    CycleWalkConst(u64, u64, u64),
    /// Unfair coin with constant probability: (p_bits)
    UnfairCoinConst(u64),
    /// `coin_flip`: the input compared unsigned against a threshold the
    /// node computed from its probability at construction; no hash.
    CoinFlipConst(u64),
    /// Chance with constant probability: (p_bits)
    ChanceConst(u64),
    /// N-of-M selection with constant n and m: (n, m)
    NOfConst(u64, u64),

    /// Fallback: no native lowering; the node runs as a closure step
    /// on the hybrid kernel and stays interpreted otherwise.
    Fallback,
}

// ── Node classification ────────────────────────────────────

/// Classify a Polydat node into a JIT-able operation.
///
/// Uses `jit_constants()` to extract assembly-time constants
/// directly from the node — no probing hacks needed.
pub fn classify_node(node: &dyn PolydatNode) -> JitOp {
    let name = node.meta().name.as_str();
    let consts = node.jit_constants();

    match name {
        "identity" => JitOp::Identity,
        "hash" | "splitmix64" | "scatter" => JitOp::SplitMix64,
        "fair_coin" => JitOp::FairCoin,
        "unfair_coin" => {
            if let Some(&p) = consts.first() {
                JitOp::UnfairCoinConst(p)
            } else {
                JitOp::Fallback
            }
        }
        "chance" => {
            if let Some(&p) = consts.first() {
                JitOp::ChanceConst(p)
            } else {
                JitOp::Fallback
            }
        }
        "xxhash3" | "xxh3" => JitOp::Hash,
        "hash_range" => {
            if let Some(&c) = consts.first() {
                JitOp::HashRangeConst(c)
            } else {
                JitOp::Fallback
            }
        }
        "hash_interval" => {
            if consts.len() >= 2 {
                JitOp::HashIntervalConst(consts[0], consts[1])
            } else {
                JitOp::Fallback
            }
        }
        "add" => {
            if let Some(&c) = consts.first() {
                JitOp::AddConst(c)
            } else {
                JitOp::Fallback
            }
        }
        "mul" => {
            if let Some(&c) = consts.first() {
                JitOp::MulConst(c)
            } else {
                JitOp::Fallback
            }
        }
        "div" => {
            if let Some(&c) = consts.first() {
                JitOp::DivConst(c)
            } else {
                JitOp::Fallback
            }
        }
        "mod" => {
            if let Some(&c) = consts.first() {
                JitOp::ModConst(c)
            } else {
                JitOp::Fallback
            }
        }
        "clamp" => {
            if consts.len() >= 2 {
                JitOp::ClampConst(consts[0], consts[1])
            } else {
                JitOp::Fallback
            }
        }
        "interleave" => JitOp::Interleave,
        "mixed_radix" => {
            if consts.is_empty() {
                JitOp::Fallback
            } else {
                JitOp::MixedRadixConst(consts)
            }
        }
        "shuffle" => {
            if consts.len() >= 3 {
                JitOp::ShuffleConst(consts[0], consts[1], consts[2])
            } else {
                JitOp::Fallback
            }
        }
        // f64 ops
        "unit_interval" => JitOp::UnitInterval,
        "f64_to_u64" => JitOp::F64ToU64,
        "round_to_u64" => JitOp::RoundToU64,
        "floor_to_u64" => JitOp::FloorToU64,
        "ceil_to_u64" => JitOp::CeilToU64,
        "clamp_f64" => {
            if consts.len() >= 2 {
                JitOp::ClampF64Const(consts[0], consts[1])
            } else {
                JitOp::Fallback
            }
        }
        "lerp" => {
            if consts.len() >= 2 {
                JitOp::LerpConst(consts[0], consts[1])
            } else {
                JitOp::Fallback
            }
        }
        "scale_range" => {
            if consts.len() >= 2 {
                JitOp::ScaleRangeConst(consts[0], consts[1])
            } else {
                JitOp::Fallback
            }
        }
        "quantize" => {
            if let Some(&c) = consts.first() {
                JitOp::QuantizeConst(c)
            } else {
                JitOp::Fallback
            }
        }
        "discretize" => {
            if consts.len() >= 2 {
                JitOp::DiscretizeConst(consts[0], consts[1])
            } else {
                JitOp::Fallback
            }
        }
        "lut_sample" | "dist_normal" | "icd_normal" | "dist_exponential" | "icd_exponential"
        | "dist_uniform" | "dist_pareto" | "dist_zipf" | "dist_empirical" => {
            if consts.len() >= 2 {
                JitOp::LutSampleConst(consts[0], consts[1])
            } else {
                JitOp::Fallback
            }
        }
        // Math functions
        "sin" => JitOp::MathUnary(0),
        "cos" => JitOp::MathUnary(1),
        "tan" => JitOp::MathUnary(2),
        "asin" => JitOp::MathUnary(3),
        "acos" => JitOp::MathUnary(4),
        "atan" => JitOp::MathUnary(5),
        "sqrt" => JitOp::MathUnary(6),
        "abs_f64" => JitOp::MathUnary(7),
        "ln" => JitOp::MathUnary(8),
        "exp" => JitOp::MathUnary(9),
        "floor_base10" => JitOp::MathUnary(10),
        "ceiling_base10" => JitOp::MathUnary(11),
        "closest_base10" => JitOp::MathUnary(12),
        "floor_decade" => JitOp::MathUnary(13),
        "ceiling_decade" => JitOp::MathUnary(14),
        "closest_decade" => JitOp::MathUnary(15),
        "floor_binomial" => JitOp::MathUnary(16),
        "ceiling_binomial" => JitOp::MathUnary(17),
        "closest_binomial" => JitOp::MathUnary(18),
        "floor_fibonacci" => JitOp::MathUnary(19),
        "ceiling_fibonacci" => JitOp::MathUnary(20),
        "closest_fibonacci" => JitOp::MathUnary(21),
        "atan2" => JitOp::MathBinary(0),
        "pow" => JitOp::MathBinary(1),
        "round_nearest" => JitOp::MathBinary(2),
        "round_floor" => JitOp::MathBinary(3),
        "round_ceiling" => JitOp::MathBinary(4),
        "to_f64" => JitOp::ToF64,
        // Two-wire u64 ops (no constants)
        "u64_add" => JitOp::U64Add2,
        "u64_sub" => JitOp::U64Sub2,
        "u64_mul" => JitOp::U64Mul2,
        "u64_div" => JitOp::U64Div2,
        "u64_mod" => JitOp::U64Mod2,
        "u64_and" => JitOp::U64And,
        "u64_or" => JitOp::U64Or,
        "u64_xor" => JitOp::U64Xor,
        "u64_shl" => JitOp::U64Shl,
        "u64_shr" => JitOp::U64Shr,
        "u64_not" => JitOp::U64Not,

        // ── Register plane (type_system_alignment.md §3, native tier of §7) ──
        "reg_add_i8" => JitOp::RegBinOp(0, 0),
        "reg_sub_i8" => JitOp::RegBinOp(0, 1),
        // `imul.i8x16` has no cranelift lowering (x86 has no
        // byte-lane multiply short of AVX-512; cranelift 0.116
        // rejects it in ISLE); `classify_node_typed` lowers
        // `reg_mul_i8` through its helper.
        "reg_shuffle_bytes" => {
            let mut mask = [0u8; 16];
            if consts.len() == 16 && consts.iter().all(|&m| m < 16) {
                for (m, &c) in mask.iter_mut().zip(consts.iter()) {
                    *m = c as u8;
                }
                JitOp::RegShuffleConst(mask)
            } else {
                JitOp::Fallback
            }
        }
        "reg_add_i16" => JitOp::RegBinOp(1, 0),
        "reg_sub_i16" => JitOp::RegBinOp(1, 1),
        "reg_mul_i16" => JitOp::RegBinOp(1, 2),
        "reg_add_i32" => JitOp::RegBinOp(2, 0),
        "reg_sub_i32" => JitOp::RegBinOp(2, 1),
        "reg_mul_i32" => JitOp::RegBinOp(2, 2),
        "reg_add_i64" => JitOp::RegBinOp(3, 0),
        "reg_sub_i64" => JitOp::RegBinOp(3, 1),
        "reg_mul_i64" => JitOp::RegBinOp(3, 2),
        "reg_add_f32" => JitOp::RegBinOp(4, 0),
        "reg_sub_f32" => JitOp::RegBinOp(4, 1),
        "reg_mul_f32" => JitOp::RegBinOp(4, 2),
        "reg_add_f64" => JitOp::RegBinOp(5, 0),
        "reg_sub_f64" => JitOp::RegBinOp(5, 1),
        "reg_mul_f64" => JitOp::RegBinOp(5, 2),
        "__reg_view_raw" | "__reg_view_i8x16" | "__reg_view_i16x8" | "__reg_view_i32x4"
        | "__reg_view_i64x2" | "__reg_view_f16x8" | "__reg_view_f32x4" | "__reg_view_f64x2" => {
            JitOp::RegCopy
        }
        "reg_splat_i8" => JitOp::RegSplat(0),
        "reg_splat_i16" => JitOp::RegSplat(1),
        "reg_splat_i32" => JitOp::RegSplat(2),
        "reg_splat_i64" => JitOp::RegSplat(3),
        "reg_splat_f32" => JitOp::RegSplat(4),
        "reg_splat_f64" => JitOp::RegSplat(5),

        "f64_add" => JitOp::F64Add,
        "f64_sub" => JitOp::F64Sub,
        "f64_mul" => JitOp::F64Mul,
        "f64_div" => JitOp::F64Div,
        "f64_mod" => JitOp::F64Mod,

        // ── Comparisons & Selections (SRD 110) ───────────────────
        "u64_eq" => JitOp::U64Cmp(ir::condcodes::IntCC::Equal),
        "u64_ne" => JitOp::U64Cmp(ir::condcodes::IntCC::NotEqual),
        "u64_lt" => JitOp::U64Cmp(ir::condcodes::IntCC::UnsignedLessThan),
        "u64_le" => JitOp::U64Cmp(ir::condcodes::IntCC::UnsignedLessThanOrEqual),
        "u64_gt" => JitOp::U64Cmp(ir::condcodes::IntCC::UnsignedGreaterThan),
        "u64_ge" => JitOp::U64Cmp(ir::condcodes::IntCC::UnsignedGreaterThanOrEqual),
        "f64_eq" => JitOp::F64Cmp(ir::condcodes::FloatCC::Equal),
        "f64_ne" => JitOp::F64Cmp(ir::condcodes::FloatCC::NotEqual),
        "f64_lt" => JitOp::F64Cmp(ir::condcodes::FloatCC::LessThan),
        "f64_le" => JitOp::F64Cmp(ir::condcodes::FloatCC::LessThanOrEqual),
        "f64_gt" => JitOp::F64Cmp(ir::condcodes::FloatCC::GreaterThan),
        "f64_ge" => JitOp::F64Cmp(ir::condcodes::FloatCC::GreaterThanOrEqual),
        "select_u64" | "select" => JitOp::SelectU64,
        "select_f64" => JitOp::SelectF64,

        // ── Wire Arithmetic & Multiples (SRD 110) ────────────────
        "div_wire" => JitOp::U64DivWire,
        "mod_wire" => JitOp::U64ModWire,
        "ceil_to_multiple" => JitOp::CeilToMultiple,
        "multiples_at_least" => JitOp::MultiplesAtLeast,
        "checked_add" => JitOp::CheckedAdd,
        "checked_sub" => JitOp::CheckedSub,
        "checked_mul" => JitOp::CheckedMul,

        // ── Variadics (SRD 110) ──────────────────────────────────
        "sum" => JitOp::VariadicSum,
        "product" => JitOp::VariadicProduct,
        "min" => JitOp::VariadicMin,
        "max" => JitOp::VariadicMax,

        // ── PRNG & Probability (SRD 110) ─────────────────────────
        "blend" => {
            if let Some(&c) = consts.first() {
                JitOp::BlendConst(c)
            } else {
                JitOp::Fallback
            }
        }
        "lfsr_step" => {
            if let Some(&fb) = consts.first() {
                JitOp::LfsrStepConst(fb)
            } else {
                JitOp::Fallback
            }
        }
        "pcg" => {
            if consts.len() >= 2 {
                JitOp::PcgConst(consts[0], consts[1])
            } else {
                JitOp::Fallback
            }
        }
        "pcg_stream" => {
            if let Some(&seed) = consts.first() {
                JitOp::PcgStreamConst(seed)
            } else {
                JitOp::Fallback
            }
        }
        "n_of" => {
            if consts.len() >= 2 {
                JitOp::NOfConst(consts[0], consts[1])
            } else {
                JitOp::Fallback
            }
        }

        "cycle_walk" => {
            if consts.len() >= 3 {
                JitOp::CycleWalkConst(consts[0], consts[1], consts[2])
            } else {
                JitOp::Fallback
            }
        }
        "coin_flip" => {
            // The body is `input < threshold` over the raw input
            // (library/fixed.rs), not a hashed unit interval as
            // `unfair_coin` is; the node bakes its threshold as its
            // one constant.
            if let Some(&threshold) = consts.first() {
                JitOp::CoinFlipConst(threshold)
            } else {
                JitOp::Fallback
            }
        }
        // `default_or` without wire types: the typed classifier decides
        // (a copy of the value, since a compiled slot is never `None`).
        "default_or" => JitOp::Identity,
        "const_u64" | "const_bool" | "session_start_millis" => {
            if let Some(&c) = consts.first() {
                JitOp::ConstU64(c)
            } else {
                JitOp::Fallback
            }
        }
        "const_f64" => {
            if let Some(&c) = consts.first() {
                JitOp::ConstF64(c)
            } else {
                JitOp::Fallback
            }
        }
        "inv_lerp" => {
            if consts.len() >= 2 {
                JitOp::InvLerpConst(consts[0], consts[1])
            } else {
                JitOp::Fallback
            }
        }
        "remap" => {
            if consts.len() >= 4 {
                JitOp::RemapConst(consts[0], consts[1], consts[2], consts[3])
            } else {
                JitOp::Fallback
            }
        }
        "epoch_offset" => {
            if let Some(&c) = consts.first() {
                JitOp::EpochOffsetConst(c)
            } else {
                JitOp::Fallback
            }
        }
        "epoch_scale" => {
            if let Some(&c) = consts.first() {
                JitOp::EpochScaleConst(c)
            } else {
                JitOp::Fallback
            }
        }
        "thread_id" => JitOp::ThreadId,
        "current_epoch_millis" => JitOp::CurrentEpochMillis,
        "perlin_1d" => {
            if consts.len() >= 2 {
                JitOp::Perlin1dConst(consts[0], consts[1])
            } else {
                JitOp::Fallback
            }
        }
        "perlin_2d" => {
            if consts.len() >= 2 {
                JitOp::Perlin2dConst(consts[0], consts[1])
            } else {
                JitOp::Fallback
            }
        }
        "simplex_2d" => {
            if consts.len() >= 2 {
                JitOp::Simplex2dConst(consts[0], consts[1])
            } else {
                JitOp::Fallback
            }
        }
        "fractal_noise_1d" => {
            if consts.len() >= 3 {
                JitOp::FractalNoise1dConst(consts[0], consts[1], consts[2])
            } else {
                JitOp::Fallback
            }
        }
        "fractal_noise_2d" => {
            if consts.len() >= 3 {
                JitOp::FractalNoise2dConst(consts[0], consts[1], consts[2])
            } else {
                JitOp::Fallback
            }
        }

        // ── Type Conversion Lattice (SRD 110) ────────────────────
        "__u64_to_f64" | "__u32_to_f64" | "__bool_to_f64" | "__bool_to_f32" | "__f32_to_f64"
        | "__u64_to_f32" | "__u32_to_f32" | "__u16_to_f32" | "__u8_to_f32" | "__u16_to_f64"
        | "__u8_to_f64" | "__u128_to_f64" | "__u128_to_f32" | "__u128_to_f16" => JitOp::ToF64,

        "__i64_to_f64" | "__i32_to_f64" | "__i64_to_f32" | "__i32_to_f32" | "__i16_to_f32"
        | "__i8_to_f32" | "__i16_to_f64" | "__i8_to_f64" | "__i128_to_f64" | "__i128_to_f32"
        | "__i128_to_f16" => JitOp::I64ToF64,

        "__f64_to_u64_checked"
        | "__f64_to_u32"
        | "__f32_to_u64"
        | "__f32_to_u32"
        | "__f64_to_u16"
        | "__f64_to_u8"
        | "__f32_to_u16"
        | "__f32_to_u8"
        | "__f16_to_u64"
        | "__f16_to_u32"
        | "__f16_to_u16"
        | "__f16_to_u8"
        | "__f64_to_u128"
        | "__f32_to_u128"
        | "__f16_to_u128"
        | "trunc_u64" => JitOp::F64ToU64,
        // `round_u64` rounds half away from zero before the saturating
        // conversion, which is what `round_to_u64` does too.
        "round_u64" => JitOp::RoundToU64,

        "__f64_to_i64" | "__f64_to_i32" | "__f32_to_i64" | "__f32_to_i32" | "__f64_to_i16"
        | "__f64_to_i8" | "__f32_to_i16" | "__f32_to_i8" | "__f16_to_i64" | "__f16_to_i32"
        | "__f16_to_i16" | "__f16_to_i8" | "__f64_to_i128" | "__f32_to_i128" | "__f16_to_i128" => {
            JitOp::F64ToI64
        }

        "__f64_to_f32" | "__f16_to_f32" | "__f16_to_f64" | "__f32_to_f16" | "__f64_to_f16" => {
            JitOp::Identity
        }

        "__u32_to_u64" | "__u64_to_u32" | "__u32_to_i32" | "__i32_to_u32" | "__u64_to_i64"
        | "__i64_to_u64" | "__bool_to_u64" | "__bool_to_i64" | "__bool_to_u32"
        | "__bool_to_i32" | "__u64_to_u16" | "__u64_to_u8" | "__u64_to_i16" | "__u64_to_i8"
        | "__i64_to_u32" | "__i64_to_u16" | "__i64_to_u8" | "__i64_to_i16" | "__i64_to_i8"
        | "__u32_to_u16" | "__u32_to_u8" | "__u32_to_i16" | "__u32_to_i8" | "__i32_to_u16"
        | "__i32_to_u8" | "__i32_to_i16" | "__i32_to_i8" | "__u16_to_u8" | "__u16_to_i8"
        | "__i16_to_u8" | "__i16_to_i8" | "__u128_to_u64" | "__u128_to_i64" | "__i128_to_u64"
        | "__i128_to_i64" | "__u128_to_u32" | "__u128_to_u16" | "__u128_to_u8"
        | "__u128_to_i32" | "__u128_to_i16" | "__u128_to_i8" | "__i128_to_u32"
        | "__i128_to_u16" | "__i128_to_u8" | "__i128_to_i32" | "__i128_to_i16" | "__i128_to_i8"
        | "__u64_to_u128" | "__u64_to_i128" | "__i64_to_u128" | "__i64_to_i128"
        | "__u128_to_i128" | "__i128_to_u128" | "__bool_to_u16" | "__bool_to_u8"
        | "__bool_to_i16" | "__bool_to_i8" | "__bool_to_u128" | "__bool_to_i128"
        | "__u8_to_f16" | "__u16_to_f16" | "__i8_to_f16" | "__i16_to_f16" | "__u64_to_f16"
        | "__i64_to_f16" | "__u32_to_f16" | "__i32_to_f16" | "__bool_to_f16" => JitOp::Identity,

        "__i32_to_i64" | "__i32_to_u64" | "__u32_to_i64" | "__u32_to_u128" | "__u32_to_i128"
        | "__i32_to_u128" | "__i32_to_i128" => JitOp::SignExtendI32,

        "__i16_to_i32" | "__i16_to_i64" | "__i16_to_u32" | "__i16_to_u64" | "__i16_to_u128"
        | "__i16_to_i128" => JitOp::SignExtendI16,

        "__i8_to_i16" | "__i8_to_i32" | "__i8_to_i64" | "__i8_to_u16" | "__i8_to_u32"
        | "__i8_to_u64" | "__i8_to_u128" | "__i8_to_i128" => JitOp::SignExtendI8,

        "__u16_to_u32" | "__u16_to_u64" | "__u16_to_i32" | "__u16_to_i64" | "__u16_to_u128"
        | "__u16_to_i128" | "__u16_to_i16" | "__i16_to_u16" => JitOp::ZeroExtendU16,

        "__u8_to_u16" | "__u8_to_u32" | "__u8_to_u64" | "__u8_to_i16" | "__u8_to_i32"
        | "__u8_to_i64" | "__u8_to_u128" | "__u8_to_i128" | "__u8_to_i8" | "__i8_to_u8" => {
            JitOp::ZeroExtendU8
        }

        "__u64_to_i32" | "__i64_to_i32" => JitOp::ZeroExtendU32,

        "__u64_to_bool" | "__i64_to_bool" | "__u32_to_bool" | "__i32_to_bool" | "__f64_to_bool"
        | "__f32_to_bool" | "__f16_to_bool" | "__u8_to_bool" | "__u16_to_bool" | "__i8_to_bool"
        | "__i16_to_bool" | "__u128_to_bool" | "__i128_to_bool" => JitOp::ToBool,

        "weighted_pick" => {
            if consts.len() >= 5 {
                JitOp::WeightedPickConst(consts[0], consts[1], consts[2], consts[3], consts[4])
            } else {
                JitOp::Fallback
            }
        }

        // ── Parameter helpers (SRD 12) ─────────────────────────
        // `is_positive` / `in_range` are JIT-lowered inline: one
        // comparison on the happy path, an extern call on the
        // fail path (which panics). The pass-through is a plain
        // store, no function call overhead on the typical cycle.
        "is_positive" => {
            let name = node.meta().ins.iter().find_map(|slot| match slot {
                crate::ast::Slot::Const {
                    name,
                    value: crate::ast::ConstValue::Str(v),
                } if name == "name" => Some(v),
                _ => None,
            });
            match name {
                Some(v) => JitOp::IsPositiveCheck {
                    name_ptr: v.as_ptr() as u64,
                    name_len: v.len() as u64,
                },
                None => JitOp::IsPositiveCheck {
                    name_ptr: 0,
                    name_len: 0,
                },
            }
        }
        "in_range" => {
            if consts.len() >= 2 {
                JitOp::InRangeCheck(consts[0], consts[1])
            } else {
                JitOp::Fallback
            }
        }
        "is_one_of" => {
            if consts.is_empty() {
                JitOp::Fallback
            } else {
                let set = node.meta().ins.iter().find_map(|slot| match slot {
                    crate::ast::Slot::Const {
                        name,
                        value: crate::ast::ConstValue::VecU64(v),
                    } if name == "allowed" => Some(v),
                    _ => None,
                });
                let (set_ptr, set_len) = match set {
                    Some(v) => (v.as_ptr() as u64, v.len() as u64),
                    None => (0, 0),
                };
                JitOp::IsOneOfCheck {
                    allowed: consts,
                    set_ptr,
                    set_len,
                }
            }
        }
        // Every other node with a kit (`required`, `this_or`,
        // `matches`, the context nodes) takes `JitOp::SlotCall` in
        // `classify_node_typed`; only a node with no kit stays
        // interpreted.
        _ => JitOp::Fallback,
    }
}

// ── Kernel constructors ────────────────────────────────────

/// Compile a set of JIT steps into a raw (no-provenance) native kernel.
///
/// Each step has: jit_op, input_slots (buffer indices), output_slots.
/// The generated function reads coords from the buffer, executes
/// all steps in order, and writes results to the buffer.
#[doc(hidden)]
pub fn compile_jit_raw(
    coord_count: usize,
    total_slots: usize,
    steps: Vec<(JitOp, Vec<usize>, Vec<usize>)>,
    output_map: HashMap<String, usize>,
    nodes: Vec<Box<dyn PolydatNode>>,
) -> Result<JitKernelRaw, String> {
    compile_jit_raw_with(
        coord_count,
        total_slots,
        steps,
        output_map,
        nodes,
        crate::compile::externs::Externs::default(),
        super::kernels::ScratchPlan::default(),
        Vec::new(),
    )
}

/// `compile_jit_raw` for a graph with extern inputs: their defaults
/// are written through into the buffer at build.
#[allow(clippy::too_many_arguments)]
pub(crate) fn compile_jit_raw_with(
    coord_count: usize,
    total_slots: usize,
    steps: Vec<(JitOp, Vec<usize>, Vec<usize>)>,
    output_map: HashMap<String, usize>,
    nodes: Vec<Box<dyn PolydatNode>>,
    externs: crate::compile::externs::Externs,
    scratch: super::kernels::ScratchPlan,
    volatile: Vec<usize>,
) -> Result<JitKernelRaw, String> {
    let (raw_fn, _, code) = compile_jit_impl(&steps, false, Some(total_slots))?;
    let mut core = JitCore::new(
        total_slots,
        coord_count,
        output_map,
        code,
        nodes,
        scratch,
        volatile,
    );
    core.set_externs(externs);
    core.engine =
        crate::compile::select::Engine::PureNative(crate::compile::select::Provenance::Raw);
    Ok(JitKernelRaw {
        core,
        code_fn: raw_fn,
    })
}

/// A compiled segment for an engine that owns its own buffer: the
/// entry point and the module that keeps it alive (SRD-105 cones,
/// hybrid JIT segments).
pub(crate) type JitSegmentCode = (NativeFn, super::kernels::JitCode);

/// An entry with no kernel wrapper: a cone node or a hybrid segment
/// owns the function pointer and code, and the state evaluating it
/// provides the buffer and the scratch.
pub(crate) fn compile_jit_entry(
    steps: &[(JitOp, Vec<usize>, Vec<usize>)],
    tracker: Option<usize>,
) -> Result<JitSegmentCode, String> {
    let (raw_fn, _, code) = compile_jit_impl(steps, false, tracker)?;
    Ok((raw_fn, code))
}

/// Compile a set of JIT steps into a push+pull (full optimization) native kernel.
#[allow(clippy::too_many_arguments)]
pub(crate) fn compile_jit_push_pull(
    coord_count: usize,
    total_slots: usize,
    steps: Vec<(JitOp, Vec<usize>, Vec<usize>)>,
    output_map: HashMap<String, usize>,
    nodes: Vec<Box<dyn PolydatNode>>,
    input_dependents: Vec<Vec<usize>>,
    externs: crate::compile::externs::Externs,
    scratch: super::kernels::ScratchPlan,
    volatile: Vec<usize>,
) -> Result<JitKernelPushPull, String> {
    let step_count = steps.len();
    let buffer_len = total_slots;
    let (_, prov_fn, code) = compile_jit_impl(&steps, true, Some(total_slots))?;
    let step_outs: Vec<&[usize]> = steps.iter().map(|(_, _, o)| o.as_slice()).collect();
    let slot_provenance =
        crate::compile::slot_provenance(coord_count, buffer_len, &step_outs, &input_dependents);
    let mut core = JitCore::new(
        total_slots,
        coord_count,
        output_map,
        code,
        nodes,
        scratch,
        volatile,
    );
    core.set_externs(externs);
    Ok(JitKernelPushPull {
        core,
        code_fn_prov: prov_fn,
        node_clean: vec![0u8; step_count],
        input_dependents,
        slot_provenance,
        changed_mask: crate::kernel::ProvMask::all_below(coord_count),
        force_run: false,
    })
}

// ── Core Cranelift IR generation ───────────────────────────

/// A native entry point over a state's slot buffer and scratch.
pub type NativeFn = unsafe fn(*const u64, *mut u64, *mut crate::ast::ScratchBuf);
/// The provenance variant: a clean flag per step follows the scratch.
pub type NativeProvFn = unsafe fn(*const u64, *mut u64, *mut crate::ast::ScratchBuf, *mut u8);

/// `(raw_fn, prov_fn, code)` — produced by the core JIT compile: the
/// scalar entry point, the provenance-tracking entry point, and the
/// finalized code that keeps both alive with the kits they call.
type JitCompiled = (NativeFn, NativeProvFn, super::kernels::JitCode);

/// Core JIT compilation. Returns (raw_fn, prov_fn, code).
/// If provenance=false, prov_fn is a dummy transmute of raw_fn.
/// If provenance=true, raw_fn is a dummy transmute of prov_fn.
fn compile_jit_impl(
    steps: &[(JitOp, Vec<usize>, Vec<usize>)],
    provenance: bool,
    tracker: Option<usize>,
) -> Result<JitCompiled, String> {
    let mut flag_builder = settings::builder();
    flag_builder.set("opt_level", "speed").unwrap();
    // Unwind tables and frame pointers are kept for debuggers and
    // profilers walking JIT frames; failures never unwind through
    // native code, they longjmp past it (see the setjmp section
    // above).
    flag_builder.set("unwind_info", "true").unwrap();
    flag_builder.set("preserve_frame_pointers", "true").unwrap();
    let isa = super::host_isa::build_host_isa(flag_builder)?;

    let mut jit_builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());

    // Register extern functions
    jit_builder.symbol("jit_xxh3_hash", jit_xxh3_hash as *const u8);
    jit_builder.symbol("jit_interleave", jit_interleave as *const u8);
    jit_builder.symbol("jit_shuffle", jit_shuffle as *const u8);
    jit_builder.symbol("jit_lut_sample", jit_lut_sample as *const u8);
    jit_builder.symbol("jit_weighted_pick", jit_weighted_pick as *const u8);
    jit_builder.symbol("jit_pcg", jit_pcg as *const u8);
    jit_builder.symbol("jit_pcg_stream", jit_pcg_stream as *const u8);
    jit_builder.symbol("jit_n_of", jit_n_of as *const u8);
    jit_builder.symbol("jit_cycle_walk", jit_cycle_walk as *const u8);
    jit_builder.symbol("jit_perlin_1d", jit_perlin_1d as *const u8);
    jit_builder.symbol("jit_perlin_2d", jit_perlin_2d as *const u8);
    jit_builder.symbol("jit_simplex_2d", jit_simplex_2d as *const u8);
    jit_builder.symbol("jit_fractal_noise_1d", jit_fractal_noise_1d as *const u8);
    jit_builder.symbol("jit_fractal_noise_2d", jit_fractal_noise_2d as *const u8);
    jit_builder.symbol("jit_thread_id", jit_thread_id as *const u8);
    jit_builder.symbol(
        "jit_current_epoch_millis",
        jit_current_epoch_millis as *const u8,
    );
    // Parameter-helper predicates (SRD 12 §"Parameter resolution
    // and validation"): happy path is inline, violation is an
    // extern call that never returns.
    jit_builder.symbol("jit_is_positive_fail", jit_is_positive_fail as *const u8);
    jit_builder.symbol("jit_in_range_fail", jit_in_range_fail as *const u8);
    jit_builder.symbol("jit_is_one_of_fail", jit_is_one_of_fail as *const u8);
    // A node's slot kit, called from native code (compiled_handles.md §6),
    // and the string producers that write into the step's entry directly.
    jit_builder.symbol("jit_slot_call", jit_slot_call as *const u8);
    jit_builder.symbol("jit_u64_to_str", jit_u64_to_str as *const u8);
    jit_builder.symbol("jit_i64_to_str", jit_i64_to_str as *const u8);
    jit_builder.symbol("jit_f64_to_str", jit_f64_to_str as *const u8);
    jit_builder.symbol("jit_str_concat", jit_str_concat as *const u8);
    jit_builder.symbol("jit_json_to_str", jit_json_to_str as *const u8);
    jit_builder.symbol("jit_vec_add", jit_vec_add as *const u8);
    jit_builder.symbol("jit_vec_scale", jit_vec_scale as *const u8);
    jit_builder.symbol("jit_vec_norm", jit_vec_norm as *const u8);
    jit_builder.symbol("jit_hash_vec", jit_hash_vec as *const u8);
    jit_builder.symbol("jit_xxhash3_vec", jit_xxhash3_vec as *const u8);
    jit_builder.symbol("jit_reg_to_vec_f32", jit_reg_to_vec_f32 as *const u8);
    jit_builder.symbol("jit_vec_dot", jit_vec_dot as *const u8);
    jit_builder.symbol("jit_vec_l2", jit_vec_l2 as *const u8);
    jit_builder.symbol("jit_vec_cosine", jit_vec_cosine as *const u8);
    jit_builder.symbol("jit_lid_mle", jit_lid_mle as *const u8);
    jit_builder.symbol("jit_reg_lane_f32", jit_reg_lane_f32 as *const u8);
    jit_builder.symbol("jit_reg_lane_i16", jit_reg_lane_i16 as *const u8);
    jit_builder.symbol("jit_reg_lane_i64", jit_reg_lane_i64 as *const u8);
    jit_builder.symbol("jit_reg_with_lane_f32", jit_reg_with_lane_f32 as *const u8);
    jit_builder.symbol("jit_reg_gather_f32", jit_reg_gather_f32 as *const u8);
    jit_builder.symbol("jit_vec_to_reg_f32", jit_vec_to_reg_f32 as *const u8);
    jit_builder.symbol("jit_reg_mul_i8", jit_reg_mul_i8 as *const u8);
    // Math externs
    jit_builder.symbol("jit_sin", jit_sin as *const u8);
    jit_builder.symbol("jit_cos", jit_cos as *const u8);
    jit_builder.symbol("jit_tan", jit_tan as *const u8);
    jit_builder.symbol("jit_asin", jit_asin as *const u8);
    jit_builder.symbol("jit_acos", jit_acos as *const u8);
    jit_builder.symbol("jit_atan", jit_atan as *const u8);
    jit_builder.symbol("jit_sqrt", jit_sqrt as *const u8);
    jit_builder.symbol("jit_abs_f64", jit_abs_f64 as *const u8);
    jit_builder.symbol("jit_ln", jit_ln as *const u8);
    jit_builder.symbol("jit_exp", jit_exp as *const u8);
    jit_builder.symbol("jit_floor_base10", jit_floor_base10 as *const u8);
    jit_builder.symbol("jit_ceiling_base10", jit_ceiling_base10 as *const u8);
    jit_builder.symbol("jit_closest_base10", jit_closest_base10 as *const u8);
    jit_builder.symbol("jit_floor_decade", jit_floor_decade as *const u8);
    jit_builder.symbol("jit_ceiling_decade", jit_ceiling_decade as *const u8);
    jit_builder.symbol("jit_closest_decade", jit_closest_decade as *const u8);
    jit_builder.symbol("jit_floor_binomial", jit_floor_binomial as *const u8);
    jit_builder.symbol("jit_ceiling_binomial", jit_ceiling_binomial as *const u8);
    jit_builder.symbol("jit_closest_binomial", jit_closest_binomial as *const u8);
    jit_builder.symbol("jit_floor_fibonacci", jit_floor_fibonacci as *const u8);
    jit_builder.symbol("jit_ceiling_fibonacci", jit_ceiling_fibonacci as *const u8);
    jit_builder.symbol("jit_closest_fibonacci", jit_closest_fibonacci as *const u8);
    jit_builder.symbol("jit_atan2", jit_atan2 as *const u8);
    jit_builder.symbol("jit_pow", jit_pow as *const u8);
    jit_builder.symbol("jit_round_nearest", jit_round_nearest as *const u8);
    jit_builder.symbol("jit_round_floor", jit_round_floor as *const u8);
    jit_builder.symbol("jit_round_ceiling", jit_round_ceiling as *const u8);
    jit_builder.symbol("jit_f64_mod", jit_f64_mod as *const u8);
    jit_builder.symbol("jit_div_zero_fail", jit_div_zero_fail as *const u8);

    let mut module = JITModule::new(jit_builder);

    // Declare extern: hash(u64) -> u64
    let hash_func_id = {
        let mut sig = module.make_signature();
        sig.params.push(AbiParam::new(types::I64));
        sig.returns.push(AbiParam::new(types::I64));
        module
            .declare_function("jit_xxh3_hash", Linkage::Import, &sig)
            .map_err(|e| format!("declare hash: {e}"))?
    };

    // Declare extern: interleave(u64, u64) -> u64
    let interleave_func_id = {
        let mut sig = module.make_signature();
        sig.params.push(AbiParam::new(types::I64));
        sig.params.push(AbiParam::new(types::I64));
        sig.returns.push(AbiParam::new(types::I64));
        module
            .declare_function("jit_interleave", Linkage::Import, &sig)
            .map_err(|e| format!("declare interleave: {e}"))?
    };

    // Declare extern: shuffle(u64, u64, u64, u64) -> u64
    let shuffle_func_id = {
        let mut sig = module.make_signature();
        for _ in 0..4 {
            sig.params.push(AbiParam::new(types::I64));
        }
        sig.returns.push(AbiParam::new(types::I64));
        module
            .declare_function("jit_shuffle", Linkage::Import, &sig)
            .map_err(|e| format!("declare shuffle: {e}"))?
    };

    // Declare extern: lut_sample(u64, u64, u64) -> u64
    let lut_sample_func_id = {
        let mut sig = module.make_signature();
        for _ in 0..3 {
            sig.params.push(AbiParam::new(types::I64));
        }
        sig.returns.push(AbiParam::new(types::I64));
        module
            .declare_function("jit_lut_sample", Linkage::Import, &sig)
            .map_err(|e| format!("declare lut_sample: {e}"))?
    };

    // Declare extern: weighted_pick(u64, u64, u64, u64, u64, u64) -> u64
    let weighted_pick_func_id = {
        let mut sig = module.make_signature();
        for _ in 0..6 {
            sig.params.push(AbiParam::new(types::I64));
        }
        sig.returns.push(AbiParam::new(types::I64));
        module
            .declare_function("jit_weighted_pick", Linkage::Import, &sig)
            .map_err(|e| format!("declare weighted_pick: {e}"))?
    };

    let pcg_func_id = {
        let mut sig = module.make_signature();
        for _ in 0..3 {
            sig.params.push(AbiParam::new(types::I64));
        }
        sig.returns.push(AbiParam::new(types::I64));
        module
            .declare_function("jit_pcg", Linkage::Import, &sig)
            .map_err(|e| format!("declare pcg: {e}"))?
    };
    let pcg_stream_func_id = {
        let mut sig = module.make_signature();
        for _ in 0..3 {
            sig.params.push(AbiParam::new(types::I64));
        }
        sig.returns.push(AbiParam::new(types::I64));
        module
            .declare_function("jit_pcg_stream", Linkage::Import, &sig)
            .map_err(|e| format!("declare pcg_stream: {e}"))?
    };
    let n_of_func_id = {
        let mut sig = module.make_signature();
        for _ in 0..3 {
            sig.params.push(AbiParam::new(types::I64));
        }
        sig.returns.push(AbiParam::new(types::I64));
        module
            .declare_function("jit_n_of", Linkage::Import, &sig)
            .map_err(|e| format!("declare n_of: {e}"))?
    };
    let cycle_walk_func_id = {
        let mut sig = module.make_signature();
        for _ in 0..4 {
            sig.params.push(AbiParam::new(types::I64));
        }
        sig.returns.push(AbiParam::new(types::I64));
        module
            .declare_function("jit_cycle_walk", Linkage::Import, &sig)
            .map_err(|e| format!("declare cycle_walk: {e}"))?
    };
    let perlin_1d_func_id = {
        let mut sig = module.make_signature();
        for _ in 0..3 {
            sig.params.push(AbiParam::new(types::I64));
        }
        sig.returns.push(AbiParam::new(types::I64));
        module
            .declare_function("jit_perlin_1d", Linkage::Import, &sig)
            .map_err(|e| format!("declare perlin_1d: {e}"))?
    };
    let perlin_2d_func_id = {
        let mut sig = module.make_signature();
        for _ in 0..4 {
            sig.params.push(AbiParam::new(types::I64));
        }
        sig.returns.push(AbiParam::new(types::I64));
        module
            .declare_function("jit_perlin_2d", Linkage::Import, &sig)
            .map_err(|e| format!("declare perlin_2d: {e}"))?
    };
    let simplex_2d_func_id = {
        let mut sig = module.make_signature();
        for _ in 0..4 {
            sig.params.push(AbiParam::new(types::I64));
        }
        sig.returns.push(AbiParam::new(types::I64));
        module
            .declare_function("jit_simplex_2d", Linkage::Import, &sig)
            .map_err(|e| format!("declare simplex_2d: {e}"))?
    };
    let fractal_noise_1d_func_id = {
        let mut sig = module.make_signature();
        for _ in 0..4 {
            sig.params.push(AbiParam::new(types::I64));
        }
        sig.returns.push(AbiParam::new(types::I64));
        module
            .declare_function("jit_fractal_noise_1d", Linkage::Import, &sig)
            .map_err(|e| format!("declare fractal_noise_1d: {e}"))?
    };
    let fractal_noise_2d_func_id = {
        let mut sig = module.make_signature();
        for _ in 0..5 {
            sig.params.push(AbiParam::new(types::I64));
        }
        sig.returns.push(AbiParam::new(types::I64));
        module
            .declare_function("jit_fractal_noise_2d", Linkage::Import, &sig)
            .map_err(|e| format!("declare fractal_noise_2d: {e}"))?
    };
    let thread_id_func_id = {
        let mut sig = module.make_signature();
        sig.returns.push(AbiParam::new(types::I64));
        module
            .declare_function("jit_thread_id", Linkage::Import, &sig)
            .map_err(|e| format!("declare thread_id: {e}"))?
    };
    let current_epoch_millis_func_id = {
        let mut sig = module.make_signature();
        sig.returns.push(AbiParam::new(types::I64));
        module
            .declare_function("jit_current_epoch_millis", Linkage::Import, &sig)
            .map_err(|e| format!("declare current_epoch_millis: {e}"))?
    };

    // Declare math externs: unary (u64) -> u64
    let math_unary_names = [
        "jit_sin",
        "jit_cos",
        "jit_tan",
        "jit_asin",
        "jit_acos",
        "jit_atan",
        "jit_sqrt",
        "jit_abs_f64",
        "jit_ln",
        "jit_exp",
        "jit_floor_base10",
        "jit_ceiling_base10",
        "jit_closest_base10",
        "jit_floor_decade",
        "jit_ceiling_decade",
        "jit_closest_decade",
        "jit_floor_binomial",
        "jit_ceiling_binomial",
        "jit_closest_binomial",
        "jit_floor_fibonacci",
        "jit_ceiling_fibonacci",
        "jit_closest_fibonacci",
    ];
    let mut math_unary_ids = Vec::new();
    for name in &math_unary_names {
        let mut sig = module.make_signature();
        sig.params.push(AbiParam::new(types::I64));
        sig.returns.push(AbiParam::new(types::I64));
        math_unary_ids.push(
            module
                .declare_function(name, Linkage::Import, &sig)
                .map_err(|e| format!("declare {name}: {e}"))?,
        );
    }

    // Declare param-helper extern:
    // jit_is_positive_fail(u64, name_ptr, name_len) -> u64
    // (never returns, but the ABI requires a return type).
    let is_positive_fail_id = {
        let mut sig = module.make_signature();
        sig.params.push(AbiParam::new(types::I64));
        sig.params.push(AbiParam::new(types::I64));
        sig.params.push(AbiParam::new(types::I64));
        sig.returns.push(AbiParam::new(types::I64));
        module
            .declare_function("jit_is_positive_fail", Linkage::Import, &sig)
            .map_err(|e| format!("declare is_positive_fail: {e}"))?
    };

    // Declare param-helper extern: jit_in_range_fail(u64, u64, u64) -> u64
    let in_range_fail_id = {
        let mut sig = module.make_signature();
        for _ in 0..3 {
            sig.params.push(AbiParam::new(types::I64));
        }
        sig.returns.push(AbiParam::new(types::I64));
        module
            .declare_function("jit_in_range_fail", Linkage::Import, &sig)
            .map_err(|e| format!("declare in_range_fail: {e}"))?
    };

    // Declare param-helper extern:
    // jit_is_one_of_fail(u64, set_ptr, set_len) -> u64
    let is_one_of_fail_id = {
        let mut sig = module.make_signature();
        sig.params.push(AbiParam::new(types::I64));
        sig.params.push(AbiParam::new(types::I64));
        sig.params.push(AbiParam::new(types::I64));
        sig.returns.push(AbiParam::new(types::I64));
        module
            .declare_function("jit_is_one_of_fail", Linkage::Import, &sig)
            .map_err(|e| format!("declare is_one_of_fail: {e}"))?
    };

    // Declare math externs: binary (u64, u64) -> u64
    let math_binary_names = [
        "jit_atan2",
        "jit_pow",
        "jit_round_nearest",
        "jit_round_floor",
        "jit_round_ceiling",
        "jit_f64_mod",
    ];
    const F64_MOD_HELPER: usize = 5;

    // Declare the zero-divisor failure: jit_div_zero_fail(kind) -> u64
    let div_zero_fail_id = {
        let mut sig = module.make_signature();
        sig.params.push(AbiParam::new(types::I64));
        sig.returns.push(AbiParam::new(types::I64));
        module
            .declare_function("jit_div_zero_fail", Linkage::Import, &sig)
            .map_err(|e| format!("declare div_zero_fail: {e}"))?
    };
    let mut math_binary_ids = Vec::new();
    for name in &math_binary_names {
        let mut sig = module.make_signature();
        sig.params.push(AbiParam::new(types::I64));
        sig.params.push(AbiParam::new(types::I64));
        sig.returns.push(AbiParam::new(types::I64));
        math_binary_ids.push(
            module
                .declare_function(name, Linkage::Import, &sig)
                .map_err(|e| format!("declare {name}: {e}"))?,
        );
    }

    // Declare extern: jit_slot_call(kit, inputs, n_in, outputs, n_out,
    // scratch, base, n_scratch)
    let slot_call_id = {
        let mut sig = module.make_signature();
        for _ in 0..8 {
            sig.params.push(AbiParam::new(types::I64));
        }
        module
            .declare_function("jit_slot_call", Linkage::Import, &sig)
            .map_err(|e| format!("declare jit_slot_call: {e}"))?
    };

    // Declare the string producers: (scratch, base, buffer, out_slot,
    // value) for the scalar conversions, (…, ptr, len) for the JSON
    // serialization and (…, pairs ptr, n) for the concatenation.
    let mut declare_str = |name: &str, args: usize| -> Result<cranelift_module::FuncId, String> {
        let mut sig = module.make_signature();
        for _ in 0..args {
            sig.params.push(AbiParam::new(types::I64));
        }
        module
            .declare_function(name, Linkage::Import, &sig)
            .map_err(|e| format!("declare {name}: {e}"))
    };
    let u64_to_str_id = declare_str("jit_u64_to_str", 5)?;
    let i64_to_str_id = declare_str("jit_i64_to_str", 5)?;
    let f64_to_str_id = declare_str("jit_f64_to_str", 5)?;
    let str_concat_id = declare_str("jit_str_concat", 6)?;
    let json_to_str_id = declare_str("jit_json_to_str", 6)?;

    // Declare the vector and register helpers: a producer takes
    // (scratch, base, buffer, out_slot, w0..w3), a reducer (w0..w3)
    // and returns bits, a lane read (lo, hi, i) and returns the word,
    // a register producer (buffer, out_slot, w0..w3).
    let mut declare_words =
        |name: &str, args: usize, returns: bool| -> Result<cranelift_module::FuncId, String> {
            let mut sig = module.make_signature();
            for _ in 0..args {
                sig.params.push(AbiParam::new(types::I64));
            }
            if returns {
                sig.returns.push(AbiParam::new(types::I64));
            }
            module
                .declare_function(name, Linkage::Import, &sig)
                .map_err(|e| format!("declare {name}: {e}"))
        };
    let vec_producer_ids = [
        (VecProducer::Add, declare_words("jit_vec_add", 8, false)?),
        (
            VecProducer::Scale,
            declare_words("jit_vec_scale", 8, false)?,
        ),
        (VecProducer::Norm, declare_words("jit_vec_norm", 8, false)?),
        (
            VecProducer::HashVec,
            declare_words("jit_hash_vec", 8, false)?,
        ),
        (
            VecProducer::XxHash3Vec,
            declare_words("jit_xxhash3_vec", 8, false)?,
        ),
        (
            VecProducer::RegToVec,
            declare_words("jit_reg_to_vec_f32", 8, false)?,
        ),
    ];
    let vec_reducer_ids = [
        (VecReducer::Dot, declare_words("jit_vec_dot", 4, true)?),
        (VecReducer::L2, declare_words("jit_vec_l2", 4, true)?),
        (
            VecReducer::Cosine,
            declare_words("jit_vec_cosine", 4, true)?,
        ),
        (VecReducer::LidMle, declare_words("jit_lid_mle", 4, true)?),
    ];
    let reg_lane_ids = [
        (
            RegLaneRead::F32,
            declare_words("jit_reg_lane_f32", 3, true)?,
        ),
        (
            RegLaneRead::I16,
            declare_words("jit_reg_lane_i16", 3, true)?,
        ),
        (
            RegLaneRead::I64,
            declare_words("jit_reg_lane_i64", 3, true)?,
        ),
    ];
    let reg_producer_ids = [
        (
            RegProducer::WithLaneF32,
            declare_words("jit_reg_with_lane_f32", 6, false)?,
        ),
        (
            RegProducer::GatherF32,
            declare_words("jit_reg_gather_f32", 6, false)?,
        ),
        (
            RegProducer::VecToRegF32,
            declare_words("jit_vec_to_reg_f32", 6, false)?,
        ),
        (
            RegProducer::MulI8,
            declare_words("jit_reg_mul_i8", 6, false)?,
        ),
    ];

    // Function signature depends on provenance mode:
    // Without: fn(coords: *const u64, buffer: *mut u64, scratch: *mut ScratchBuf)
    // With:    fn(coords, buffer, scratch, clean: *mut u8)
    let mut sig = module.make_signature();
    sig.params.push(AbiParam::new(types::I64)); // coords ptr
    sig.params.push(AbiParam::new(types::I64)); // buffer ptr
    sig.params.push(AbiParam::new(types::I64)); // scratch ptr
    if provenance {
        sig.params.push(AbiParam::new(types::I64)); // clean ptr
    }
    let func_id = module
        .declare_function("polydat_kernel", Linkage::Local, &sig)
        .map_err(|e| format!("declare kernel: {e}"))?;

    let mut ctx = module.make_context();
    ctx.func.signature = sig;

    let mut fb_ctx = FunctionBuilderContext::new();
    {
        let mut builder = FunctionBuilder::new(&mut ctx.func, &mut fb_ctx);
        let block = builder.create_block();
        builder.append_block_params_for_function_params(block);
        builder.switch_to_block(block);
        builder.seal_block(block);

        let _coords_ptr = builder.block_params(block)[0];
        let buffer_ptr = builder.block_params(block)[1];
        let scratch_ptr = builder.block_params(block)[2];
        let clean_ptr = if provenance {
            Some(builder.block_params(block)[3])
        } else {
            None
        };

        // Import extern functions for calls
        let hash_func_ref = module.declare_func_in_func(hash_func_id, builder.func);
        let interleave_func_ref = module.declare_func_in_func(interleave_func_id, builder.func);
        let shuffle_func_ref = module.declare_func_in_func(shuffle_func_id, builder.func);
        let lut_sample_func_ref = module.declare_func_in_func(lut_sample_func_id, builder.func);
        let weighted_pick_func_ref =
            module.declare_func_in_func(weighted_pick_func_id, builder.func);
        let is_positive_fail_ref = module.declare_func_in_func(is_positive_fail_id, builder.func);
        let in_range_fail_ref = module.declare_func_in_func(in_range_fail_id, builder.func);
        let div_zero_fail_ref = module.declare_func_in_func(div_zero_fail_id, builder.func);
        let is_one_of_fail_ref = module.declare_func_in_func(is_one_of_fail_id, builder.func);
        let slot_call_ref = module.declare_func_in_func(slot_call_id, builder.func);
        let u64_to_str_ref = module.declare_func_in_func(u64_to_str_id, builder.func);
        let i64_to_str_ref = module.declare_func_in_func(i64_to_str_id, builder.func);
        let f64_to_str_ref = module.declare_func_in_func(f64_to_str_id, builder.func);
        let str_concat_ref = module.declare_func_in_func(str_concat_id, builder.func);
        let json_to_str_ref = module.declare_func_in_func(json_to_str_id, builder.func);
        let vec_producer_refs: Vec<(VecProducer, ir::FuncRef)> = vec_producer_ids
            .iter()
            .map(|(k, id)| (*k, module.declare_func_in_func(*id, builder.func)))
            .collect();
        let vec_reducer_refs: Vec<(VecReducer, ir::FuncRef)> = vec_reducer_ids
            .iter()
            .map(|(k, id)| (*k, module.declare_func_in_func(*id, builder.func)))
            .collect();
        let reg_lane_refs: Vec<(RegLaneRead, ir::FuncRef)> = reg_lane_ids
            .iter()
            .map(|(k, id)| (*k, module.declare_func_in_func(*id, builder.func)))
            .collect();
        let reg_producer_refs: Vec<(RegProducer, ir::FuncRef)> = reg_producer_ids
            .iter()
            .map(|(k, id)| (*k, module.declare_func_in_func(*id, builder.func)))
            .collect();
        let pcg_func_ref = module.declare_func_in_func(pcg_func_id, builder.func);
        let pcg_stream_func_ref = module.declare_func_in_func(pcg_stream_func_id, builder.func);
        let n_of_func_ref = module.declare_func_in_func(n_of_func_id, builder.func);
        let cycle_walk_func_ref = module.declare_func_in_func(cycle_walk_func_id, builder.func);
        let perlin_1d_func_ref = module.declare_func_in_func(perlin_1d_func_id, builder.func);
        let perlin_2d_func_ref = module.declare_func_in_func(perlin_2d_func_id, builder.func);
        let simplex_2d_func_ref = module.declare_func_in_func(simplex_2d_func_id, builder.func);
        let fractal_noise_1d_func_ref =
            module.declare_func_in_func(fractal_noise_1d_func_id, builder.func);
        let fractal_noise_2d_func_ref =
            module.declare_func_in_func(fractal_noise_2d_func_id, builder.func);
        let thread_id_func_ref = module.declare_func_in_func(thread_id_func_id, builder.func);
        let current_epoch_millis_func_ref =
            module.declare_func_in_func(current_epoch_millis_func_id, builder.func);
        let math_unary_refs: Vec<_> = math_unary_ids
            .iter()
            .map(|id| module.declare_func_in_func(*id, builder.func))
            .collect();
        let math_binary_refs: Vec<_> = math_binary_ids
            .iter()
            .map(|id| module.declare_func_in_func(*id, builder.func))
            .collect();
        // Generate code for each step
        for (step_idx, (jit_op, input_slots, output_slots)) in steps.iter().enumerate() {
            // Provenance guard: if clean[step_idx] != 0, skip this node.
            let skip_block = if let Some(cp) = clean_ptr {
                let skip = builder.create_block();
                let cont = builder.create_block();
                // Load clean[step_idx] (u8)
                let offset = builder.ins().iconst(types::I64, step_idx as i64);
                let addr = builder.ins().iadd(cp, offset);
                let flag = builder.ins().load(types::I8, ir::MemFlags::new(), addr, 0);
                let zero = builder.ins().iconst(types::I8, 0);
                let is_clean = builder
                    .ins()
                    .icmp(ir::condcodes::IntCC::NotEqual, flag, zero);
                builder.ins().brif(is_clean, skip, &[], cont, &[]);
                builder.switch_to_block(cont);
                builder.seal_block(cont);
                Some(skip)
            } else {
                None
            };
            // A7: name the step for the failure path. The store stays only
            // when the step calls a helper, the one way native code fails;
            // a step of inline arithmetic pays nothing.
            let tracker_store = tracker.map(|t| {
                let idx = builder.ins().iconst(types::I64, step_idx as i64);
                let inst = store_slot(&mut builder, buffer_ptr, t, idx);
                (inst, builder.func.dfg.num_insts())
            });
            match jit_op {
                JitOp::Identity => {
                    // A copy of every slot the port spans: one for a
                    // carrier or handle, two for a 128-bit immediate.
                    for (&i, &o) in input_slots.iter().zip(output_slots.iter()) {
                        let val = load_slot(&mut builder, buffer_ptr, i);
                        store_slot(&mut builder, buffer_ptr, o, val);
                    }
                }
                JitOp::AddConst(c) => {
                    let val = load_slot(&mut builder, buffer_ptr, input_slots[0]);
                    let c_val = builder.ins().iconst(types::I64, *c as i64);
                    let result = builder.ins().iadd(val, c_val);
                    store_slot(&mut builder, buffer_ptr, output_slots[0], result);
                }
                JitOp::MulConst(c) => {
                    let val = load_slot(&mut builder, buffer_ptr, input_slots[0]);
                    let c_val = builder.ins().iconst(types::I64, *c as i64);
                    let result = builder.ins().imul(val, c_val);
                    store_slot(&mut builder, buffer_ptr, output_slots[0], result);
                }
                JitOp::DivConst(c) | JitOp::ModConst(c) => {
                    // The body's `/` or `%` by the constant: a zero
                    // constant fails at every evaluation as it does.
                    let is_div = matches!(jit_op, JitOp::DivConst(_));
                    let val = load_slot(&mut builder, buffer_ptr, input_slots[0]);
                    if *c == 0 {
                        let kind = builder.ins().iconst(types::I64, if is_div { 0 } else { 1 });
                        let _ = builder.ins().call(div_zero_fail_ref, &[kind]);
                        store_slot(&mut builder, buffer_ptr, output_slots[0], val);
                    } else {
                        let c_val = builder.ins().iconst(types::I64, *c as i64);
                        let result = if is_div {
                            builder.ins().udiv(val, c_val)
                        } else {
                            builder.ins().urem(val, c_val)
                        };
                        store_slot(&mut builder, buffer_ptr, output_slots[0], result);
                    }
                }
                JitOp::U64DivWire | JitOp::U64ModWire => {
                    // The body's `/` or `%` by the wire: a zero divisor
                    // fails as it does there; `udiv` and `urem` trap on
                    // one, so the failure branches first.
                    let is_div = matches!(jit_op, JitOp::U64DivWire);
                    let a = load_slot(&mut builder, buffer_ptr, input_slots[0]);
                    let b = load_slot(&mut builder, buffer_ptr, input_slots[1]);
                    let zero = builder.ins().iconst(types::I64, 0);
                    let is_zero = builder.ins().icmp(ir::condcodes::IntCC::Equal, b, zero);
                    let fail_block = builder.create_block();
                    let ok_block = builder.create_block();
                    builder.ins().brif(is_zero, fail_block, &[], ok_block, &[]);
                    builder.switch_to_block(fail_block);
                    builder.seal_block(fail_block);
                    let kind = builder.ins().iconst(types::I64, if is_div { 0 } else { 1 });
                    let _ = builder.ins().call(div_zero_fail_ref, &[kind]);
                    builder.ins().jump(ok_block, &[]);
                    builder.switch_to_block(ok_block);
                    builder.seal_block(ok_block);
                    let result = if is_div {
                        builder.ins().udiv(a, b)
                    } else {
                        builder.ins().urem(a, b)
                    };
                    store_slot(&mut builder, buffer_ptr, output_slots[0], result);
                }
                JitOp::ClampConst(min, max) => {
                    let val = load_slot(&mut builder, buffer_ptr, input_slots[0]);
                    let min_val = builder.ins().iconst(types::I64, *min as i64);
                    let max_val = builder.ins().iconst(types::I64, *max as i64);
                    let clamped_lo = builder.ins().umax(val, min_val);
                    let clamped = builder.ins().umin(clamped_lo, max_val);
                    store_slot(&mut builder, buffer_ptr, output_slots[0], clamped);
                }
                JitOp::Interleave => {
                    let a = load_slot(&mut builder, buffer_ptr, input_slots[0]);
                    let b = load_slot(&mut builder, buffer_ptr, input_slots[1]);
                    let call = builder.ins().call(interleave_func_ref, &[a, b]);
                    let result = builder.inst_results(call)[0];
                    store_slot(&mut builder, buffer_ptr, output_slots[0], result);
                }
                JitOp::MixedRadixConst(radixes) => {
                    // Unrolled: for each radix, emit urem + udiv
                    let mut remainder = load_slot(&mut builder, buffer_ptr, input_slots[0]);
                    for (i, &radix) in radixes.iter().enumerate() {
                        if radix == 0 {
                            // Unbounded: output = remainder
                            store_slot(&mut builder, buffer_ptr, output_slots[i], remainder);
                        } else {
                            let r = builder.ins().iconst(types::I64, radix as i64);
                            let digit = builder.ins().urem(remainder, r);
                            store_slot(&mut builder, buffer_ptr, output_slots[i], digit);
                            remainder = builder.ins().udiv(remainder, r);
                        }
                    }
                }
                JitOp::Hash => {
                    let val = load_slot(&mut builder, buffer_ptr, input_slots[0]);
                    let call = builder.ins().call(hash_func_ref, &[val]);
                    let result = builder.inst_results(call)[0];
                    store_slot(&mut builder, buffer_ptr, output_slots[0], result);
                }
                JitOp::SplitMix64 => {
                    let x0 = load_slot(&mut builder, buffer_ptr, input_slots[0]);
                    let c_gamma = builder
                        .ins()
                        .iconst(types::I64, 0x9e3779b97f4a7c15u64 as i64);
                    let x1 = builder.ins().iadd(x0, c_gamma);
                    let s30 = builder.ins().ushr_imm(x1, 30);
                    let x2 = builder.ins().bxor(x1, s30);
                    let c_m1 = builder
                        .ins()
                        .iconst(types::I64, 0xbf58476d1ce4e5b9u64 as i64);
                    let x3 = builder.ins().imul(x2, c_m1);
                    let s27 = builder.ins().ushr_imm(x3, 27);
                    let x4 = builder.ins().bxor(x3, s27);
                    let c_m2 = builder
                        .ins()
                        .iconst(types::I64, 0x94d049bb133111ebu64 as i64);
                    let x5 = builder.ins().imul(x4, c_m2);
                    let s31 = builder.ins().ushr_imm(x5, 31);
                    let result = builder.ins().bxor(x5, s31);
                    store_slot(&mut builder, buffer_ptr, output_slots[0], result);
                }
                JitOp::FairCoin => {
                    let x0 = load_slot(&mut builder, buffer_ptr, input_slots[0]);
                    let c_gamma = builder
                        .ins()
                        .iconst(types::I64, 0x9e3779b97f4a7c15u64 as i64);
                    let x1 = builder.ins().iadd(x0, c_gamma);
                    let s30 = builder.ins().ushr_imm(x1, 30);
                    let x2 = builder.ins().bxor(x1, s30);
                    let c_m1 = builder
                        .ins()
                        .iconst(types::I64, 0xbf58476d1ce4e5b9u64 as i64);
                    let x3 = builder.ins().imul(x2, c_m1);
                    let s27 = builder.ins().ushr_imm(x3, 27);
                    let x4 = builder.ins().bxor(x3, s27);
                    let c_m2 = builder
                        .ins()
                        .iconst(types::I64, 0x94d049bb133111ebu64 as i64);
                    let x5 = builder.ins().imul(x4, c_m2);
                    let s31 = builder.ins().ushr_imm(x5, 31);
                    let h = builder.ins().bxor(x5, s31);
                    let one = builder.ins().iconst(types::I64, 1);
                    let result = builder.ins().band(h, one);
                    store_slot(&mut builder, buffer_ptr, output_slots[0], result);
                }
                JitOp::CoinFlipConst(threshold) => {
                    let x = load_slot(&mut builder, buffer_ptr, input_slots[0]);
                    let thr = builder.ins().iconst(types::I64, *threshold as i64);
                    let cmp = builder
                        .ins()
                        .icmp(ir::condcodes::IntCC::UnsignedLessThan, x, thr);
                    let zero = builder.ins().iconst(types::I64, 0);
                    let one = builder.ins().iconst(types::I64, 1);
                    let result = builder.ins().select(cmp, one, zero);
                    store_slot(&mut builder, buffer_ptr, output_slots[0], result);
                }
                JitOp::UnfairCoinConst(p_bits) => {
                    let x0 = load_slot(&mut builder, buffer_ptr, input_slots[0]);
                    let c_gamma = builder
                        .ins()
                        .iconst(types::I64, 0x9e3779b97f4a7c15u64 as i64);
                    let x1 = builder.ins().iadd(x0, c_gamma);
                    let s30 = builder.ins().ushr_imm(x1, 30);
                    let x2 = builder.ins().bxor(x1, s30);
                    let c_m1 = builder
                        .ins()
                        .iconst(types::I64, 0xbf58476d1ce4e5b9u64 as i64);
                    let x3 = builder.ins().imul(x2, c_m1);
                    let s27 = builder.ins().ushr_imm(x3, 27);
                    let x4 = builder.ins().bxor(x3, s27);
                    let c_m2 = builder
                        .ins()
                        .iconst(types::I64, 0x94d049bb133111ebu64 as i64);
                    let x5 = builder.ins().imul(x4, c_m2);
                    let s31 = builder.ins().ushr_imm(x5, 31);
                    let h = builder.ins().bxor(x5, s31);

                    let fval = builder.ins().fcvt_from_uint(types::F64, h);
                    let max_f = builder.ins().f64const(u64::MAX as f64);
                    let unit = builder.ins().fdiv(fval, max_f);
                    let p_f = builder.ins().f64const(f64::from_bits(*p_bits));
                    let cmp = builder
                        .ins()
                        .fcmp(ir::condcodes::FloatCC::LessThan, unit, p_f);
                    let zero = builder.ins().iconst(types::I64, 0);
                    let one = builder.ins().iconst(types::I64, 1);
                    let result = builder.ins().select(cmp, one, zero);
                    store_slot(&mut builder, buffer_ptr, output_slots[0], result);
                }
                JitOp::ChanceConst(p_bits) => {
                    let x0 = load_slot(&mut builder, buffer_ptr, input_slots[0]);
                    let c_gamma = builder
                        .ins()
                        .iconst(types::I64, 0x9e3779b97f4a7c15u64 as i64);
                    let x1 = builder.ins().iadd(x0, c_gamma);
                    let s30 = builder.ins().ushr_imm(x1, 30);
                    let x2 = builder.ins().bxor(x1, s30);
                    let c_m1 = builder
                        .ins()
                        .iconst(types::I64, 0xbf58476d1ce4e5b9u64 as i64);
                    let x3 = builder.ins().imul(x2, c_m1);
                    let s27 = builder.ins().ushr_imm(x3, 27);
                    let x4 = builder.ins().bxor(x3, s27);
                    let c_m2 = builder
                        .ins()
                        .iconst(types::I64, 0x94d049bb133111ebu64 as i64);
                    let x5 = builder.ins().imul(x4, c_m2);
                    let s31 = builder.ins().ushr_imm(x5, 31);
                    let h = builder.ins().bxor(x5, s31);

                    let fval = builder.ins().fcvt_from_uint(types::F64, h);
                    let max_f = builder.ins().f64const(u64::MAX as f64);
                    let unit = builder.ins().fdiv(fval, max_f);
                    let p_f = builder.ins().f64const(f64::from_bits(*p_bits));
                    let cmp = builder
                        .ins()
                        .fcmp(ir::condcodes::FloatCC::LessThan, unit, p_f);
                    let zero_bits = builder.ins().iconst(types::I64, 0.0_f64.to_bits() as i64);
                    let one_bits = builder.ins().iconst(types::I64, 1.0_f64.to_bits() as i64);
                    let result = builder.ins().select(cmp, one_bits, zero_bits);
                    store_slot(&mut builder, buffer_ptr, output_slots[0], result);
                }
                JitOp::ShuffleConst(feedback, size, min) => {
                    let val = load_slot(&mut builder, buffer_ptr, input_slots[0]);
                    let fb = builder.ins().iconst(types::I64, *feedback as i64);
                    let sz = builder.ins().iconst(types::I64, *size as i64);
                    let mn = builder.ins().iconst(types::I64, *min as i64);
                    let call = builder.ins().call(shuffle_func_ref, &[val, fb, sz, mn]);
                    let result = builder.inst_results(call)[0];
                    store_slot(&mut builder, buffer_ptr, output_slots[0], result);
                }

                // --- f64 ops ---
                JitOp::UnitInterval => {
                    // u64 → f64: input as f64 / u64::MAX as f64
                    let val = load_slot(&mut builder, buffer_ptr, input_slots[0]);
                    let fval = builder.ins().fcvt_from_uint(types::F64, val);
                    let max_f = builder.ins().f64const(u64::MAX as f64);
                    let result = builder.ins().fdiv(fval, max_f);
                    store_slot_f64(&mut builder, buffer_ptr, output_slots[0], result);
                }
                JitOp::F64ToU64 => {
                    let fval = load_slot_f64(&mut builder, buffer_ptr, input_slots[0]);
                    let result = builder.ins().fcvt_to_uint_sat(types::I64, fval);
                    store_slot(&mut builder, buffer_ptr, output_slots[0], result);
                }
                JitOp::RoundToU64 => {
                    let fval = load_slot_f64(&mut builder, buffer_ptr, input_slots[0]);
                    let rounded = round_half_away(&mut builder, fval);
                    let result = builder.ins().fcvt_to_uint_sat(types::I64, rounded);
                    store_slot(&mut builder, buffer_ptr, output_slots[0], result);
                }
                JitOp::FloorToU64 => {
                    let fval = load_slot_f64(&mut builder, buffer_ptr, input_slots[0]);
                    let floored = builder.ins().floor(fval);
                    let result = builder.ins().fcvt_to_uint_sat(types::I64, floored);
                    store_slot(&mut builder, buffer_ptr, output_slots[0], result);
                }
                JitOp::CeilToU64 => {
                    let fval = load_slot_f64(&mut builder, buffer_ptr, input_slots[0]);
                    let ceiled = builder.ins().ceil(fval);
                    let result = builder.ins().fcvt_to_uint_sat(types::I64, ceiled);
                    store_slot(&mut builder, buffer_ptr, output_slots[0], result);
                }
                JitOp::ClampF64Const(min_bits, max_bits) => {
                    let fval = load_slot_f64(&mut builder, buffer_ptr, input_slots[0]);
                    let fmin = builder.ins().f64const(f64::from_bits(*min_bits));
                    let fmax = builder.ins().f64const(f64::from_bits(*max_bits));
                    let clamped = clamp_ir(&mut builder, fval, fmin, fmax);
                    store_slot_f64(&mut builder, buffer_ptr, output_slots[0], clamped);
                }
                JitOp::LerpConst(a_bits, b_bits) => {
                    // a + t * (b - a)
                    let t = load_slot_f64(&mut builder, buffer_ptr, input_slots[0]);
                    let a = builder.ins().f64const(f64::from_bits(*a_bits));
                    let b = builder.ins().f64const(f64::from_bits(*b_bits));
                    let diff = builder.ins().fsub(b, a);
                    let scaled = builder.ins().fmul(t, diff);
                    let result = builder.ins().fadd(a, scaled);
                    store_slot_f64(&mut builder, buffer_ptr, output_slots[0], result);
                }
                JitOp::ScaleRangeConst(min_bits, range_bits) => {
                    // min + range * (input as f64 / u64::MAX as f64)
                    let val = load_slot(&mut builder, buffer_ptr, input_slots[0]);
                    let fval = builder.ins().fcvt_from_uint(types::F64, val);
                    let max_f = builder.ins().f64const(u64::MAX as f64);
                    let t = builder.ins().fdiv(fval, max_f);
                    let fmin = builder.ins().f64const(f64::from_bits(*min_bits));
                    let frange = builder.ins().f64const(f64::from_bits(*range_bits));
                    let scaled = builder.ins().fmul(t, frange);
                    let result = builder.ins().fadd(fmin, scaled);
                    store_slot_f64(&mut builder, buffer_ptr, output_slots[0], result);
                }
                JitOp::QuantizeConst(step_bits) => {
                    // round(val / step) * step
                    let fval = load_slot_f64(&mut builder, buffer_ptr, input_slots[0]);
                    let step = builder.ins().f64const(f64::from_bits(*step_bits));
                    let divided = builder.ins().fdiv(fval, step);
                    let rounded = round_half_away(&mut builder, divided);
                    let result = builder.ins().fmul(rounded, step);
                    store_slot_f64(&mut builder, buffer_ptr, output_slots[0], result);
                }

                JitOp::LutSampleConst(lut_ptr, lut_len) => {
                    // Extern call: jit_lut_sample(input_bits, lut_ptr, lut_len) -> f64 bits
                    let input = load_slot(&mut builder, buffer_ptr, input_slots[0]);
                    let ptr_val = builder.ins().iconst(types::I64, *lut_ptr as i64);
                    let len_val = builder.ins().iconst(types::I64, *lut_len as i64);
                    let call = builder
                        .ins()
                        .call(lut_sample_func_ref, &[input, ptr_val, len_val]);
                    let result = builder.inst_results(call)[0];
                    store_slot(&mut builder, buffer_ptr, output_slots[0], result);
                }
                JitOp::DiscretizeConst(range_bits, buckets) => {
                    // clamp(input, 0.0, range - eps) / range * buckets → u64
                    let fval = load_slot_f64(&mut builder, buffer_ptr, input_slots[0]);
                    let range = f64::from_bits(*range_bits);
                    let fzero = builder.ins().f64const(0.0);
                    let frange_m_eps = builder.ins().f64const(range - f64::EPSILON);
                    let frange = builder.ins().f64const(range);
                    let fbuckets = builder.ins().f64const(*buckets as f64);
                    let clamped = clamp_ir(&mut builder, fval, fzero, frange_m_eps);
                    let divided = builder.ins().fdiv(clamped, frange);
                    let scaled = builder.ins().fmul(divided, fbuckets);
                    let as_u64 = builder.ins().fcvt_to_uint_sat(types::I64, scaled);
                    let max_bucket = builder.ins().iconst(types::I64, (*buckets - 1) as i64);
                    let result = builder.ins().umin(as_u64, max_bucket);
                    store_slot(&mut builder, buffer_ptr, output_slots[0], result);
                }

                JitOp::WeightedPickConst(values_ptr, biases_ptr, primaries_ptr, aliases_ptr, n) => {
                    // Extern call: jit_weighted_pick(input, values_ptr, biases_ptr, primaries_ptr, aliases_ptr, n)
                    let input = load_slot(&mut builder, buffer_ptr, input_slots[0]);
                    let v_ptr = builder.ins().iconst(types::I64, *values_ptr as i64);
                    let b_ptr = builder.ins().iconst(types::I64, *biases_ptr as i64);
                    let p_ptr = builder.ins().iconst(types::I64, *primaries_ptr as i64);
                    let a_ptr = builder.ins().iconst(types::I64, *aliases_ptr as i64);
                    let n_val = builder.ins().iconst(types::I64, *n as i64);
                    let call = builder.ins().call(
                        weighted_pick_func_ref,
                        &[input, v_ptr, b_ptr, p_ptr, a_ptr, n_val],
                    );
                    let result = builder.inst_results(call)[0];
                    store_slot(&mut builder, buffer_ptr, output_slots[0], result);
                }

                JitOp::MathUnary(idx) => {
                    let input = load_slot(&mut builder, buffer_ptr, input_slots[0]);
                    let func_ref = math_unary_refs[*idx as usize];
                    let call = builder.ins().call(func_ref, &[input]);
                    let result = builder.inst_results(call)[0];
                    store_slot(&mut builder, buffer_ptr, output_slots[0], result);
                }

                JitOp::MathBinary(idx) => {
                    let a = load_slot(&mut builder, buffer_ptr, input_slots[0]);
                    let b = load_slot(&mut builder, buffer_ptr, input_slots[1]);
                    let func_ref = math_binary_refs[*idx as usize];
                    let call = builder.ins().call(func_ref, &[a, b]);
                    let result = builder.inst_results(call)[0];
                    store_slot(&mut builder, buffer_ptr, output_slots[0], result);
                }

                JitOp::ToF64 => {
                    let val = load_slot(&mut builder, buffer_ptr, input_slots[0]);
                    let fval = builder.ins().fcvt_from_uint(types::F64, val);
                    store_slot_f64(&mut builder, buffer_ptr, output_slots[0], fval);
                }

                // ── Register plane: one vector instruction per op ──
                JitOp::RegBinOp(lane, arith) => {
                    let vt = reg_lane_type(*lane);
                    let a = load_reg128(&mut builder, buffer_ptr, input_slots[0], vt);
                    let b = load_reg128(&mut builder, buffer_ptr, input_slots[2], vt);
                    let is_float = matches!(*lane, 4 | 5);
                    let r = match (arith, is_float) {
                        (0, false) => builder.ins().iadd(a, b),
                        (1, false) => builder.ins().isub(a, b),
                        (2, false) => builder.ins().imul(a, b),
                        (0, true) => builder.ins().fadd(a, b),
                        (1, true) => builder.ins().fsub(a, b),
                        (2, true) => builder.ins().fmul(a, b),
                        _ => unreachable!("RegBinOp arith index out of range"),
                    };
                    store_reg128(&mut builder, buffer_ptr, output_slots[0], r);
                }
                JitOp::RegCopy => {
                    let v = load_reg128(&mut builder, buffer_ptr, input_slots[0], types::I64X2);
                    store_reg128(&mut builder, buffer_ptr, output_slots[0], v);
                }
                JitOp::RegSplat(lane) => {
                    let vt = reg_lane_type(*lane);
                    let scalar = match *lane {
                        // Integer lanes: u64 slot reduced to lane width.
                        0 => {
                            let v = load_slot(&mut builder, buffer_ptr, input_slots[0]);
                            builder.ins().ireduce(types::I8, v)
                        }
                        1 => {
                            let v = load_slot(&mut builder, buffer_ptr, input_slots[0]);
                            builder.ins().ireduce(types::I16, v)
                        }
                        2 => {
                            let v = load_slot(&mut builder, buffer_ptr, input_slots[0]);
                            builder.ins().ireduce(types::I32, v)
                        }
                        3 => load_slot(&mut builder, buffer_ptr, input_slots[0]),
                        // Float lanes: f64 slot, demoted for f32.
                        4 => {
                            let f = load_slot_f64(&mut builder, buffer_ptr, input_slots[0]);
                            builder.ins().fdemote(types::F32, f)
                        }
                        5 => load_slot_f64(&mut builder, buffer_ptr, input_slots[0]),
                        _ => unreachable!("RegSplat lane index out of range"),
                    };
                    let v = builder.ins().splat(vt, scalar);
                    store_reg128(&mut builder, buffer_ptr, output_slots[0], v);
                }

                // Two-wire u64 integer ops — pure Cranelift, no extern call
                JitOp::U64Add2 => {
                    let a = load_slot(&mut builder, buffer_ptr, input_slots[0]);
                    let b = load_slot(&mut builder, buffer_ptr, input_slots[1]);
                    let result = builder.ins().iadd(a, b);
                    store_slot(&mut builder, buffer_ptr, output_slots[0], result);
                }
                JitOp::U64Sub2 => {
                    let a = load_slot(&mut builder, buffer_ptr, input_slots[0]);
                    let b = load_slot(&mut builder, buffer_ptr, input_slots[1]);
                    let result = builder.ins().isub(a, b);
                    store_slot(&mut builder, buffer_ptr, output_slots[0], result);
                }
                JitOp::U64Mul2 => {
                    let a = load_slot(&mut builder, buffer_ptr, input_slots[0]);
                    let b = load_slot(&mut builder, buffer_ptr, input_slots[1]);
                    let result = builder.ins().imul(a, b);
                    store_slot(&mut builder, buffer_ptr, output_slots[0], result);
                }
                JitOp::U64Div2 => {
                    let a = load_slot(&mut builder, buffer_ptr, input_slots[0]);
                    let b = load_slot(&mut builder, buffer_ptr, input_slots[1]);
                    // Guard: if b == 0, store 0; else store a / b.
                    // Must branch because udiv traps on zero divisor.
                    let zero = builder.ins().iconst(types::I64, 0);
                    let is_zero = builder.ins().icmp(ir::condcodes::IntCC::Equal, b, zero);
                    let div_block = builder.create_block();
                    let merge_block = builder.create_block();
                    builder.append_block_param(merge_block, types::I64);
                    builder
                        .ins()
                        .brif(is_zero, merge_block, &[zero], div_block, &[]);
                    builder.switch_to_block(div_block);
                    builder.seal_block(div_block);
                    let div_result = builder.ins().udiv(a, b);
                    builder.ins().jump(merge_block, &[div_result]);
                    builder.switch_to_block(merge_block);
                    builder.seal_block(merge_block);
                    let result = builder.block_params(merge_block)[0];
                    store_slot(&mut builder, buffer_ptr, output_slots[0], result);
                }
                JitOp::U64Mod2 => {
                    let a = load_slot(&mut builder, buffer_ptr, input_slots[0]);
                    let b = load_slot(&mut builder, buffer_ptr, input_slots[1]);
                    // Guard: if b == 0, store 0; else store a % b.
                    // Must branch because urem traps on zero divisor.
                    let zero = builder.ins().iconst(types::I64, 0);
                    let is_zero = builder.ins().icmp(ir::condcodes::IntCC::Equal, b, zero);
                    let rem_block = builder.create_block();
                    let merge_block = builder.create_block();
                    builder.append_block_param(merge_block, types::I64);
                    builder
                        .ins()
                        .brif(is_zero, merge_block, &[zero], rem_block, &[]);
                    builder.switch_to_block(rem_block);
                    builder.seal_block(rem_block);
                    let rem_result = builder.ins().urem(a, b);
                    builder.ins().jump(merge_block, &[rem_result]);
                    builder.switch_to_block(merge_block);
                    builder.seal_block(merge_block);
                    let result = builder.block_params(merge_block)[0];
                    store_slot(&mut builder, buffer_ptr, output_slots[0], result);
                }
                JitOp::U64And => {
                    let a = load_slot(&mut builder, buffer_ptr, input_slots[0]);
                    let b = load_slot(&mut builder, buffer_ptr, input_slots[1]);
                    let result = builder.ins().band(a, b);
                    store_slot(&mut builder, buffer_ptr, output_slots[0], result);
                }
                JitOp::U64Or => {
                    let a = load_slot(&mut builder, buffer_ptr, input_slots[0]);
                    let b = load_slot(&mut builder, buffer_ptr, input_slots[1]);
                    let result = builder.ins().bor(a, b);
                    store_slot(&mut builder, buffer_ptr, output_slots[0], result);
                }
                JitOp::U64Xor => {
                    let a = load_slot(&mut builder, buffer_ptr, input_slots[0]);
                    let b = load_slot(&mut builder, buffer_ptr, input_slots[1]);
                    let result = builder.ins().bxor(a, b);
                    store_slot(&mut builder, buffer_ptr, output_slots[0], result);
                }
                JitOp::U64Shl => {
                    let a = load_slot(&mut builder, buffer_ptr, input_slots[0]);
                    let b = load_slot(&mut builder, buffer_ptr, input_slots[1]);
                    let result = builder.ins().ishl(a, b);
                    store_slot(&mut builder, buffer_ptr, output_slots[0], result);
                }
                JitOp::U64Shr => {
                    let a = load_slot(&mut builder, buffer_ptr, input_slots[0]);
                    let b = load_slot(&mut builder, buffer_ptr, input_slots[1]);
                    let result = builder.ins().ushr(a, b);
                    store_slot(&mut builder, buffer_ptr, output_slots[0], result);
                }
                JitOp::U64Not => {
                    let a = load_slot(&mut builder, buffer_ptr, input_slots[0]);
                    let result = builder.ins().bnot(a);
                    store_slot(&mut builder, buffer_ptr, output_slots[0], result);
                }

                // Inline binary f64 arithmetic — pure Cranelift, no extern call
                JitOp::F64Add => {
                    let a = load_slot_f64(&mut builder, buffer_ptr, input_slots[0]);
                    let b = load_slot_f64(&mut builder, buffer_ptr, input_slots[1]);
                    let result = builder.ins().fadd(a, b);
                    store_slot_f64(&mut builder, buffer_ptr, output_slots[0], result);
                }
                JitOp::F64Sub => {
                    let a = load_slot_f64(&mut builder, buffer_ptr, input_slots[0]);
                    let b = load_slot_f64(&mut builder, buffer_ptr, input_slots[1]);
                    let result = builder.ins().fsub(a, b);
                    store_slot_f64(&mut builder, buffer_ptr, output_slots[0], result);
                }
                JitOp::F64Mul => {
                    let a = load_slot_f64(&mut builder, buffer_ptr, input_slots[0]);
                    let b = load_slot_f64(&mut builder, buffer_ptr, input_slots[1]);
                    let result = builder.ins().fmul(a, b);
                    store_slot_f64(&mut builder, buffer_ptr, output_slots[0], result);
                }
                JitOp::F64Div => {
                    let a = load_slot_f64(&mut builder, buffer_ptr, input_slots[0]);
                    let b = load_slot_f64(&mut builder, buffer_ptr, input_slots[1]);
                    // Guard: if b == 0, result = 0; else result = a / b
                    let zero = builder.ins().f64const(0.0);
                    let is_zero = builder.ins().fcmp(ir::condcodes::FloatCC::Equal, b, zero);
                    let div_result = builder.ins().fdiv(a, b);
                    let result = builder.ins().select(is_zero, zero, div_result);
                    store_slot_f64(&mut builder, buffer_ptr, output_slots[0], result);
                }
                JitOp::F64Mod => {
                    // The body through its helper: Rust's `%` on floats.
                    let a = load_slot(&mut builder, buffer_ptr, input_slots[0]);
                    let b = load_slot(&mut builder, buffer_ptr, input_slots[1]);
                    let call = builder
                        .ins()
                        .call(math_binary_refs[F64_MOD_HELPER], &[a, b]);
                    let result = builder.inst_results(call)[0];
                    store_slot(&mut builder, buffer_ptr, output_slots[0], result);
                }

                JitOp::IsPositiveCheck { name_ptr, name_len } => {
                    // if input == 0: call jit_is_positive_fail (panics);
                    // else: store input → output.
                    // The branch splits to a fail block for the
                    // violation path; the merge reads through the
                    // common path after either branch completes.
                    let val = load_slot(&mut builder, buffer_ptr, input_slots[0]);
                    let zero = builder.ins().iconst(types::I64, 0);
                    let is_zero = builder.ins().icmp(ir::condcodes::IntCC::Equal, val, zero);
                    let fail_block = builder.create_block();
                    let ok_block = builder.create_block();
                    builder.ins().brif(is_zero, fail_block, &[], ok_block, &[]);

                    builder.switch_to_block(fail_block);
                    builder.seal_block(fail_block);
                    let np = builder.ins().iconst(types::I64, *name_ptr as i64);
                    let nl = builder.ins().iconst(types::I64, *name_len as i64);
                    let _ = builder.ins().call(is_positive_fail_ref, &[val, np, nl]);
                    // Extern panics — this is unreachable. Jump to
                    // ok_block to keep the IR well-formed; the
                    // branch never runs in practice.
                    builder.ins().jump(ok_block, &[]);

                    builder.switch_to_block(ok_block);
                    builder.seal_block(ok_block);
                    store_slot(&mut builder, buffer_ptr, output_slots[0], val);
                }

                JitOp::InRangeCheck(lo, hi) => {
                    // if input < lo || input > hi: call
                    // jit_in_range_fail (panics); else store
                    // input → output.
                    let val = load_slot(&mut builder, buffer_ptr, input_slots[0]);
                    let lo_v = builder.ins().iconst(types::I64, *lo as i64);
                    let hi_v = builder.ins().iconst(types::I64, *hi as i64);
                    let below =
                        builder
                            .ins()
                            .icmp(ir::condcodes::IntCC::UnsignedLessThan, val, lo_v);
                    let above =
                        builder
                            .ins()
                            .icmp(ir::condcodes::IntCC::UnsignedGreaterThan, val, hi_v);
                    let out_of_range = builder.ins().bor(below, above);

                    let fail_block = builder.create_block();
                    let ok_block = builder.create_block();
                    builder
                        .ins()
                        .brif(out_of_range, fail_block, &[], ok_block, &[]);

                    builder.switch_to_block(fail_block);
                    builder.seal_block(fail_block);
                    let _ = builder.ins().call(in_range_fail_ref, &[val, lo_v, hi_v]);
                    builder.ins().jump(ok_block, &[]);

                    builder.switch_to_block(ok_block);
                    builder.seal_block(ok_block);
                    store_slot(&mut builder, buffer_ptr, output_slots[0], val);
                }

                JitOp::IsOneOfCheck {
                    allowed,
                    set_ptr,
                    set_len,
                } => {
                    // Unroll the allow-list as N inline eq
                    // comparisons OR'd together. Fast-path is
                    // 1–8 values (the common case); pathologically
                    // large allow-lists still JIT but cost N
                    // comparisons per cycle.
                    let val = load_slot(&mut builder, buffer_ptr, input_slots[0]);
                    let mut any_match = builder.ins().iconst(types::I8, 0);
                    for allow in allowed.iter() {
                        let c = builder.ins().iconst(types::I64, *allow as i64);
                        let eq = builder.ins().icmp(ir::condcodes::IntCC::Equal, val, c);
                        any_match = builder.ins().bor(any_match, eq);
                    }
                    let fail_block = builder.create_block();
                    let ok_block = builder.create_block();
                    // If any_match == 0 (no equality hit),
                    // branch to the fail extern. Otherwise
                    // jump straight to ok_block.
                    builder
                        .ins()
                        .brif(any_match, ok_block, &[], fail_block, &[]);

                    builder.switch_to_block(fail_block);
                    builder.seal_block(fail_block);
                    let sp = builder.ins().iconst(types::I64, *set_ptr as i64);
                    let sl = builder.ins().iconst(types::I64, *set_len as i64);
                    let _ = builder.ins().call(is_one_of_fail_ref, &[val, sp, sl]);
                    builder.ins().jump(ok_block, &[]);

                    builder.switch_to_block(ok_block);
                    builder.seal_block(ok_block);
                    store_slot(&mut builder, buffer_ptr, output_slots[0], val);
                }

                JitOp::U64Cmp(cc) => {
                    let a = load_slot(&mut builder, buffer_ptr, input_slots[0]);
                    let b = load_slot(&mut builder, buffer_ptr, input_slots[1]);
                    let cmp = builder.ins().icmp(*cc, a, b);
                    let zero = builder.ins().iconst(types::I64, 0);
                    let one = builder.ins().iconst(types::I64, 1);
                    let result = builder.ins().select(cmp, one, zero);
                    store_slot(&mut builder, buffer_ptr, output_slots[0], result);
                }
                JitOp::F64Cmp(cc) => {
                    let a = load_slot_f64(&mut builder, buffer_ptr, input_slots[0]);
                    let b = load_slot_f64(
                        &mut builder,
                        buffer_ptr,
                        if input_slots.len() > 1 {
                            input_slots[1]
                        } else {
                            input_slots[0]
                        },
                    );
                    let cmp = builder.ins().fcmp(*cc, a, b);
                    let zero = builder.ins().iconst(types::I64, 0);
                    let one = builder.ins().iconst(types::I64, 1);
                    let result = builder.ins().select(cmp, one, zero);
                    store_slot(&mut builder, buffer_ptr, output_slots[0], result);
                }
                JitOp::SelectU64 => {
                    let cond = load_slot(&mut builder, buffer_ptr, input_slots[0]);
                    let a = load_slot(
                        &mut builder,
                        buffer_ptr,
                        if input_slots.len() > 1 {
                            input_slots[1]
                        } else {
                            input_slots[0]
                        },
                    );
                    let b = load_slot(
                        &mut builder,
                        buffer_ptr,
                        if input_slots.len() > 2 {
                            input_slots[2]
                        } else {
                            input_slots[0]
                        },
                    );
                    let zero = builder.ins().iconst(types::I64, 0);
                    let is_nonzero = builder
                        .ins()
                        .icmp(ir::condcodes::IntCC::NotEqual, cond, zero);
                    let result = builder.ins().select(is_nonzero, a, b);
                    store_slot(&mut builder, buffer_ptr, output_slots[0], result);
                }
                JitOp::SelectF64 => {
                    let cond = load_slot(&mut builder, buffer_ptr, input_slots[0]);
                    let a = load_slot_f64(
                        &mut builder,
                        buffer_ptr,
                        if input_slots.len() > 1 {
                            input_slots[1]
                        } else {
                            input_slots[0]
                        },
                    );
                    let b = load_slot_f64(
                        &mut builder,
                        buffer_ptr,
                        if input_slots.len() > 2 {
                            input_slots[2]
                        } else {
                            input_slots[0]
                        },
                    );
                    let zero = builder.ins().iconst(types::I64, 0);
                    let is_nonzero = builder
                        .ins()
                        .icmp(ir::condcodes::IntCC::NotEqual, cond, zero);
                    let result = builder.ins().select(is_nonzero, a, b);
                    store_slot_f64(&mut builder, buffer_ptr, output_slots[0], result);
                }

                JitOp::I64ToF64 => {
                    let val = load_slot(&mut builder, buffer_ptr, input_slots[0]);
                    let fval = builder.ins().fcvt_from_sint(types::F64, val);
                    store_slot_f64(&mut builder, buffer_ptr, output_slots[0], fval);
                }
                JitOp::F64ToI64 => {
                    let fval = load_slot_f64(&mut builder, buffer_ptr, input_slots[0]);
                    let ival = builder.ins().fcvt_to_sint(types::I64, fval);
                    store_slot(&mut builder, buffer_ptr, output_slots[0], ival);
                }
                JitOp::SignExtendI32 => {
                    let val = load_slot(&mut builder, buffer_ptr, input_slots[0]);
                    let i32_val = builder.ins().ireduce(types::I32, val);
                    let sext_val = builder.ins().sextend(types::I64, i32_val);
                    store_slot(&mut builder, buffer_ptr, output_slots[0], sext_val);
                }
                JitOp::SignExtendI16 => {
                    let val = load_slot(&mut builder, buffer_ptr, input_slots[0]);
                    let i16_val = builder.ins().ireduce(types::I16, val);
                    let sext_val = builder.ins().sextend(types::I64, i16_val);
                    store_slot(&mut builder, buffer_ptr, output_slots[0], sext_val);
                }
                JitOp::SignExtendI8 => {
                    let val = load_slot(&mut builder, buffer_ptr, input_slots[0]);
                    let i8_val = builder.ins().ireduce(types::I8, val);
                    let sext_val = builder.ins().sextend(types::I64, i8_val);
                    store_slot(&mut builder, buffer_ptr, output_slots[0], sext_val);
                }
                JitOp::ZeroExtendU32 => {
                    let val = load_slot(&mut builder, buffer_ptr, input_slots[0]);
                    let mask = builder.ins().iconst(types::I64, 0xFFFFFFFFu64 as i64);
                    let result = builder.ins().band(val, mask);
                    store_slot(&mut builder, buffer_ptr, output_slots[0], result);
                }
                JitOp::ZeroExtendU16 => {
                    let val = load_slot(&mut builder, buffer_ptr, input_slots[0]);
                    let mask = builder.ins().iconst(types::I64, 0xFFFFu64 as i64);
                    let result = builder.ins().band(val, mask);
                    store_slot(&mut builder, buffer_ptr, output_slots[0], result);
                }
                JitOp::ZeroExtendU8 => {
                    let val = load_slot(&mut builder, buffer_ptr, input_slots[0]);
                    let mask = builder.ins().iconst(types::I64, 0xFFu64 as i64);
                    let result = builder.ins().band(val, mask);
                    store_slot(&mut builder, buffer_ptr, output_slots[0], result);
                }
                JitOp::ToBool => {
                    let val = load_slot(&mut builder, buffer_ptr, input_slots[0]);
                    let zero = builder.ins().iconst(types::I64, 0);
                    let one = builder.ins().iconst(types::I64, 1);
                    let cmp = builder
                        .ins()
                        .icmp(ir::condcodes::IntCC::NotEqual, val, zero);
                    let result = builder.ins().select(cmp, one, zero);
                    store_slot(&mut builder, buffer_ptr, output_slots[0], result);
                }
                JitOp::ConstU64(v) | JitOp::ConstF64(v) => {
                    let result = builder.ins().iconst(types::I64, *v as i64);
                    store_slot(&mut builder, buffer_ptr, output_slots[0], result);
                }
                JitOp::HashRangeConst(max) => {
                    let input = load_slot(&mut builder, buffer_ptr, input_slots[0]);
                    let c_gamma = builder
                        .ins()
                        .iconst(types::I64, 0x9e3779b97f4a7c15u64 as i64);
                    let x1 = builder.ins().iadd(input, c_gamma);
                    let s30 = builder.ins().ushr_imm(x1, 30);
                    let x2 = builder.ins().bxor(x1, s30);
                    let c_m1 = builder
                        .ins()
                        .iconst(types::I64, 0xbf58476d1ce4e5b9u64 as i64);
                    let x3 = builder.ins().imul(x2, c_m1);
                    let s27 = builder.ins().ushr_imm(x3, 27);
                    let x4 = builder.ins().bxor(x3, s27);
                    let c_m2 = builder
                        .ins()
                        .iconst(types::I64, 0x94d049bb133111ebu64 as i64);
                    let x5 = builder.ins().imul(x4, c_m2);
                    let s31 = builder.ins().ushr_imm(x5, 31);
                    let h = builder.ins().bxor(x5, s31);
                    if *max == 0 {
                        let zero = builder.ins().iconst(types::I64, 0);
                        store_slot(&mut builder, buffer_ptr, output_slots[0], zero);
                    } else {
                        let m = builder.ins().iconst(types::I64, *max as i64);
                        let rem = builder.ins().urem(h, m);
                        store_slot(&mut builder, buffer_ptr, output_slots[0], rem);
                    }
                }
                JitOp::HashIntervalConst(min_bits, max_bits) => {
                    let input = load_slot(&mut builder, buffer_ptr, input_slots[0]);
                    let c_gamma = builder
                        .ins()
                        .iconst(types::I64, 0x9e3779b97f4a7c15u64 as i64);
                    let x1 = builder.ins().iadd(input, c_gamma);
                    let s30 = builder.ins().ushr_imm(x1, 30);
                    let x2 = builder.ins().bxor(x1, s30);
                    let c_m1 = builder
                        .ins()
                        .iconst(types::I64, 0xbf58476d1ce4e5b9u64 as i64);
                    let x3 = builder.ins().imul(x2, c_m1);
                    let s27 = builder.ins().ushr_imm(x3, 27);
                    let x4 = builder.ins().bxor(x3, s27);
                    let c_m2 = builder
                        .ins()
                        .iconst(types::I64, 0x94d049bb133111ebu64 as i64);
                    let x5 = builder.ins().imul(x4, c_m2);
                    let s31 = builder.ins().ushr_imm(x5, 31);
                    let h = builder.ins().bxor(x5, s31);

                    let h_f = builder.ins().fcvt_from_uint(types::F64, h);
                    let denom = builder.ins().f64const(u64::MAX as f64);
                    let unit = builder.ins().fdiv(h_f, denom);
                    let min_f = f64::from_bits(*min_bits);
                    let max_f = f64::from_bits(*max_bits);
                    let span = builder.ins().f64const(max_f - min_f);
                    let min_val = builder.ins().f64const(min_f);
                    let scaled = builder.ins().fmul(unit, span);
                    let res_f = builder.ins().fadd(min_val, scaled);
                    let res = builder
                        .ins()
                        .bitcast(types::I64, ir::MemFlags::new(), res_f);
                    store_slot(&mut builder, buffer_ptr, output_slots[0], res);
                }
                JitOp::InvLerpConst(a_bits, b_bits) => {
                    let input = load_slot(&mut builder, buffer_ptr, input_slots[0]);
                    let in_f = builder
                        .ins()
                        .bitcast(types::F64, ir::MemFlags::new(), input);
                    let a_f = f64::from_bits(*a_bits);
                    let b_f = f64::from_bits(*b_bits);
                    let a_val = builder.ins().f64const(a_f);
                    // The body's operations in its order: the reciprocal
                    // of the span (infinite for an empty one), the
                    // product, the clamp.
                    let inv_span = builder.ins().f64const(1.0 / (b_f - a_f));
                    let diff = builder.ins().fsub(in_f, a_val);
                    let t = builder.ins().fmul(diff, inv_span);
                    let zero = builder.ins().f64const(0.0);
                    let one = builder.ins().f64const(1.0);
                    let res_f = clamp_ir(&mut builder, t, zero, one);
                    let res = builder
                        .ins()
                        .bitcast(types::I64, ir::MemFlags::new(), res_f);
                    store_slot(&mut builder, buffer_ptr, output_slots[0], res);
                }
                JitOp::RemapConst(in_min_bits, in_max_bits, out_min_bits, out_max_bits) => {
                    let input = load_slot(&mut builder, buffer_ptr, input_slots[0]);
                    let in_f = builder
                        .ins()
                        .bitcast(types::F64, ir::MemFlags::new(), input);
                    let in_min = f64::from_bits(*in_min_bits);
                    let in_max = f64::from_bits(*in_max_bits);
                    let out_min = f64::from_bits(*out_min_bits);
                    let out_max = f64::from_bits(*out_max_bits);
                    // The body's operations in its order: a division by
                    // the span (not a product with its reciprocal, which
                    // differs in the last bit), then the affine step.
                    let in_span_val = builder.ins().f64const(in_max - in_min);
                    let in_min_val = builder.ins().f64const(in_min);
                    let out_min_val = builder.ins().f64const(out_min);
                    let out_span_val = builder.ins().f64const(out_max - out_min);
                    let diff = builder.ins().fsub(in_f, in_min_val);
                    let t = builder.ins().fdiv(diff, in_span_val);
                    let scaled = builder.ins().fmul(t, out_span_val);
                    let res_f = builder.ins().fadd(out_min_val, scaled);
                    let res = builder
                        .ins()
                        .bitcast(types::I64, ir::MemFlags::new(), res_f);
                    store_slot(&mut builder, buffer_ptr, output_slots[0], res);
                }
                JitOp::EpochOffsetConst(base) => {
                    let val = load_slot(&mut builder, buffer_ptr, input_slots[0]);
                    let b = builder.ins().iconst(types::I64, *base as i64);
                    let res = builder.ins().iadd(val, b);
                    store_slot(&mut builder, buffer_ptr, output_slots[0], res);
                }
                JitOp::EpochScaleConst(factor) => {
                    let val = load_slot(&mut builder, buffer_ptr, input_slots[0]);
                    let f = builder.ins().iconst(types::I64, *factor as i64);
                    let res = builder.ins().imul(val, f);
                    store_slot(&mut builder, buffer_ptr, output_slots[0], res);
                }
                JitOp::ThreadId => {
                    let call = builder.ins().call(thread_id_func_ref, &[]);
                    let res = builder.inst_results(call)[0];
                    store_slot(&mut builder, buffer_ptr, output_slots[0], res);
                }
                JitOp::CurrentEpochMillis => {
                    let call = builder.ins().call(current_epoch_millis_func_ref, &[]);
                    let res = builder.inst_results(call)[0];
                    store_slot(&mut builder, buffer_ptr, output_slots[0], res);
                }
                JitOp::Perlin1dConst(perm_ptr, freq_bits) => {
                    let input = load_slot(&mut builder, buffer_ptr, input_slots[0]);
                    let p = builder.ins().iconst(types::I64, *perm_ptr as i64);
                    let fb = builder.ins().iconst(types::I64, *freq_bits as i64);
                    let call = builder.ins().call(perlin_1d_func_ref, &[input, p, fb]);
                    let res = builder.inst_results(call)[0];
                    store_slot(&mut builder, buffer_ptr, output_slots[0], res);
                }
                JitOp::Perlin2dConst(perm_ptr, freq_bits) => {
                    let x = load_slot(&mut builder, buffer_ptr, input_slots[0]);
                    let y = load_slot(&mut builder, buffer_ptr, input_slots[1]);
                    let p = builder.ins().iconst(types::I64, *perm_ptr as i64);
                    let fb = builder.ins().iconst(types::I64, *freq_bits as i64);
                    let call = builder.ins().call(perlin_2d_func_ref, &[x, y, p, fb]);
                    let res = builder.inst_results(call)[0];
                    store_slot(&mut builder, buffer_ptr, output_slots[0], res);
                }
                JitOp::Simplex2dConst(perm_ptr, freq_bits) => {
                    let x = load_slot(&mut builder, buffer_ptr, input_slots[0]);
                    let y = load_slot(&mut builder, buffer_ptr, input_slots[1]);
                    let p = builder.ins().iconst(types::I64, *perm_ptr as i64);
                    let fb = builder.ins().iconst(types::I64, *freq_bits as i64);
                    let call = builder.ins().call(simplex_2d_func_ref, &[x, y, p, fb]);
                    let res = builder.inst_results(call)[0];
                    store_slot(&mut builder, buffer_ptr, output_slots[0], res);
                }
                JitOp::FractalNoise1dConst(perm_ptr, freq_bits, octaves) => {
                    let input = load_slot(&mut builder, buffer_ptr, input_slots[0]);
                    let p = builder.ins().iconst(types::I64, *perm_ptr as i64);
                    let fb = builder.ins().iconst(types::I64, *freq_bits as i64);
                    let oct = builder.ins().iconst(types::I64, *octaves as i64);
                    let call = builder
                        .ins()
                        .call(fractal_noise_1d_func_ref, &[input, p, fb, oct]);
                    let res = builder.inst_results(call)[0];
                    store_slot(&mut builder, buffer_ptr, output_slots[0], res);
                }
                JitOp::FractalNoise2dConst(perm_ptr, freq_bits, octaves) => {
                    let x = load_slot(&mut builder, buffer_ptr, input_slots[0]);
                    let y = load_slot(&mut builder, buffer_ptr, input_slots[1]);
                    let p = builder.ins().iconst(types::I64, *perm_ptr as i64);
                    let fb = builder.ins().iconst(types::I64, *freq_bits as i64);
                    let oct = builder.ins().iconst(types::I64, *octaves as i64);
                    let call = builder
                        .ins()
                        .call(fractal_noise_2d_func_ref, &[x, y, p, fb, oct]);
                    let res = builder.inst_results(call)[0];
                    store_slot(&mut builder, buffer_ptr, output_slots[0], res);
                }
                JitOp::CycleWalkConst(range, seed, inc) => {
                    let pos = load_slot(&mut builder, buffer_ptr, input_slots[0]);
                    let r = builder.ins().iconst(types::I64, *range as i64);
                    let s = builder.ins().iconst(types::I64, *seed as i64);
                    let i = builder.ins().iconst(types::I64, *inc as i64);
                    let call = builder.ins().call(cycle_walk_func_ref, &[pos, r, s, i]);
                    let res = builder.inst_results(call)[0];
                    store_slot(&mut builder, buffer_ptr, output_slots[0], res);
                }

                JitOp::VariadicSum => {
                    if input_slots.is_empty() {
                        let zero = builder.ins().iconst(types::I64, 0);
                        store_slot(&mut builder, buffer_ptr, output_slots[0], zero);
                    } else {
                        let mut acc = load_slot(&mut builder, buffer_ptr, input_slots[0]);
                        for &slot in &input_slots[1..] {
                            let v = load_slot(&mut builder, buffer_ptr, slot);
                            acc = builder.ins().iadd(acc, v);
                        }
                        store_slot(&mut builder, buffer_ptr, output_slots[0], acc);
                    }
                }
                JitOp::VariadicProduct => {
                    if input_slots.is_empty() {
                        let one = builder.ins().iconst(types::I64, 1);
                        store_slot(&mut builder, buffer_ptr, output_slots[0], one);
                    } else {
                        let mut acc = load_slot(&mut builder, buffer_ptr, input_slots[0]);
                        for &slot in &input_slots[1..] {
                            let v = load_slot(&mut builder, buffer_ptr, slot);
                            acc = builder.ins().imul(acc, v);
                        }
                        store_slot(&mut builder, buffer_ptr, output_slots[0], acc);
                    }
                }
                JitOp::VariadicMin => {
                    if input_slots.is_empty() {
                        let zero = builder.ins().iconst(types::I64, 0);
                        store_slot(&mut builder, buffer_ptr, output_slots[0], zero);
                    } else {
                        let mut acc = load_slot(&mut builder, buffer_ptr, input_slots[0]);
                        for &slot in &input_slots[1..] {
                            let v = load_slot(&mut builder, buffer_ptr, slot);
                            let cmp =
                                builder
                                    .ins()
                                    .icmp(ir::condcodes::IntCC::UnsignedLessThan, v, acc);
                            acc = builder.ins().select(cmp, v, acc);
                        }
                        store_slot(&mut builder, buffer_ptr, output_slots[0], acc);
                    }
                }
                JitOp::VariadicMax => {
                    if input_slots.is_empty() {
                        let zero = builder.ins().iconst(types::I64, 0);
                        store_slot(&mut builder, buffer_ptr, output_slots[0], zero);
                    } else {
                        let mut acc = load_slot(&mut builder, buffer_ptr, input_slots[0]);
                        for &slot in &input_slots[1..] {
                            let v = load_slot(&mut builder, buffer_ptr, slot);
                            let cmp = builder.ins().icmp(
                                ir::condcodes::IntCC::UnsignedGreaterThan,
                                v,
                                acc,
                            );
                            acc = builder.ins().select(cmp, v, acc);
                        }
                        store_slot(&mut builder, buffer_ptr, output_slots[0], acc);
                    }
                }

                JitOp::CeilToMultiple => {
                    let val = load_slot(&mut builder, buffer_ptr, input_slots[0]);
                    let m = load_slot(&mut builder, buffer_ptr, input_slots[1]);
                    let zero = builder.ins().iconst(types::I64, 0);
                    let one = builder.ins().iconst(types::I64, 1);
                    let is_zero = builder.ins().icmp(ir::condcodes::IntCC::Equal, m, zero);
                    let calc_block = builder.create_block();
                    let merge_block = builder.create_block();
                    builder.append_block_param(merge_block, types::I64);
                    builder
                        .ins()
                        .brif(is_zero, merge_block, &[val], calc_block, &[]);
                    builder.switch_to_block(calc_block);
                    builder.seal_block(calc_block);
                    // `div_ceil` without the sum that overflows near the
                    // top, then the saturating product: the body's.
                    let div = div_ceil(&mut builder, val, m, one);
                    let high = builder.ins().umulhi(div, m);
                    let low = builder.ins().imul(div, m);
                    let zero_hi = builder.ins().iconst(types::I64, 0);
                    let overflows =
                        builder
                            .ins()
                            .icmp(ir::condcodes::IntCC::NotEqual, high, zero_hi);
                    let max = builder.ins().iconst(types::I64, -1);
                    let mul = builder.ins().select(overflows, max, low);
                    builder.ins().jump(merge_block, &[mul]);
                    builder.switch_to_block(merge_block);
                    builder.seal_block(merge_block);
                    let result = builder.block_params(merge_block)[0];
                    store_slot(&mut builder, buffer_ptr, output_slots[0], result);
                }
                JitOp::CheckedAdd => {
                    let a = load_slot(&mut builder, buffer_ptr, input_slots[0]);
                    let b = load_slot(&mut builder, buffer_ptr, input_slots[1]);
                    let sum = builder.ins().iadd(a, b);
                    let is_overflow =
                        builder
                            .ins()
                            .icmp(ir::condcodes::IntCC::UnsignedLessThan, sum, a);
                    let zero = builder.ins().iconst(types::I64, 0);
                    let result = builder.ins().select(is_overflow, zero, sum);
                    store_slot(&mut builder, buffer_ptr, output_slots[0], result);
                }
                JitOp::CheckedSub => {
                    let a = load_slot(&mut builder, buffer_ptr, input_slots[0]);
                    let b = load_slot(&mut builder, buffer_ptr, input_slots[1]);
                    let is_lt = builder
                        .ins()
                        .icmp(ir::condcodes::IntCC::UnsignedLessThan, a, b);
                    let diff = builder.ins().isub(a, b);
                    let zero = builder.ins().iconst(types::I64, 0);
                    let result = builder.ins().select(is_lt, zero, diff);
                    store_slot(&mut builder, buffer_ptr, output_slots[0], result);
                }
                JitOp::CheckedMul => {
                    let a = load_slot(&mut builder, buffer_ptr, input_slots[0]);
                    let b = load_slot(&mut builder, buffer_ptr, input_slots[1]);
                    let prod = builder.ins().imul(a, b);
                    let zero = builder.ins().iconst(types::I64, 0);
                    let a_is_zero = builder.ins().icmp(ir::condcodes::IntCC::Equal, a, zero);
                    let div_block = builder.create_block();
                    let merge_block = builder.create_block();
                    builder.append_block_param(merge_block, types::I64);
                    builder
                        .ins()
                        .brif(a_is_zero, merge_block, &[zero], div_block, &[]);
                    builder.switch_to_block(div_block);
                    builder.seal_block(div_block);
                    let div = builder.ins().udiv(prod, a);
                    let ok = builder.ins().icmp(ir::condcodes::IntCC::Equal, div, b);
                    let mul_res = builder.ins().select(ok, prod, zero);
                    builder.ins().jump(merge_block, &[mul_res]);
                    builder.switch_to_block(merge_block);
                    builder.seal_block(merge_block);
                    let result = builder.block_params(merge_block)[0];
                    store_slot(&mut builder, buffer_ptr, output_slots[0], result);
                }
                JitOp::MultiplesAtLeast => {
                    let val = load_slot(&mut builder, buffer_ptr, input_slots[0]);
                    let m = load_slot(&mut builder, buffer_ptr, input_slots[1]);
                    let zero = builder.ins().iconst(types::I64, 0);
                    let one = builder.ins().iconst(types::I64, 1);
                    let is_zero = builder.ins().icmp(ir::condcodes::IntCC::Equal, m, zero);
                    let calc_block = builder.create_block();
                    let merge_block = builder.create_block();
                    builder.append_block_param(merge_block, types::I64);
                    builder
                        .ins()
                        .brif(is_zero, merge_block, &[zero], calc_block, &[]);
                    builder.switch_to_block(calc_block);
                    builder.seal_block(calc_block);
                    let div = div_ceil(&mut builder, val, m, one);
                    builder.ins().jump(merge_block, &[div]);
                    builder.switch_to_block(merge_block);
                    builder.seal_block(merge_block);
                    let result = builder.block_params(merge_block)[0];
                    store_slot(&mut builder, buffer_ptr, output_slots[0], result);
                }

                JitOp::BlendConst(mix_bits) => {
                    // The body reinterprets both inputs' bits as f64
                    // and returns the mix's bits (`blend` in
                    // polydat-nodes `probability.rs`); the lowering does the
                    // same, not a numeric conversion.
                    let a = load_slot(&mut builder, buffer_ptr, input_slots[0]);
                    let b = load_slot(&mut builder, buffer_ptr, input_slots[1]);
                    let fa = builder.ins().bitcast(types::F64, ir::MemFlags::new(), a);
                    let fb = builder.ins().bitcast(types::F64, ir::MemFlags::new(), b);
                    let mix_f64 = f64::from_bits(*mix_bits);
                    let mix_val = builder.ins().f64const(mix_f64);
                    let one = builder.ins().f64const(1.0);
                    let one_minus_mix = builder.ins().fsub(one, mix_val);
                    let a_part = builder.ins().fmul(fa, one_minus_mix);
                    let b_part = builder.ins().fmul(fb, mix_val);
                    let sum = builder.ins().fadd(a_part, b_part);
                    let result = builder.ins().bitcast(types::I64, ir::MemFlags::new(), sum);
                    store_slot(&mut builder, buffer_ptr, output_slots[0], result);
                }
                JitOp::LfsrStepConst(feedback) => {
                    let val = load_slot(&mut builder, buffer_ptr, input_slots[0]);
                    let feedback = builder.ins().iconst(types::I64, *feedback as i64);
                    let one = builder.ins().iconst(types::I64, 1);
                    let zero = builder.ins().iconst(types::I64, 0);
                    let shifted = builder.ins().ushr(val, one);
                    let lsb = builder.ins().band(val, one);
                    let is_odd = builder
                        .ins()
                        .icmp(ir::condcodes::IntCC::NotEqual, lsb, zero);
                    let fb_mask = builder.ins().select(is_odd, feedback, zero);
                    let result = builder.ins().bxor(shifted, fb_mask);
                    store_slot(&mut builder, buffer_ptr, output_slots[0], result);
                }
                JitOp::PcgConst(seed, stream) => {
                    let input = load_slot(&mut builder, buffer_ptr, input_slots[0]);
                    let s = builder.ins().iconst(types::I64, *seed as i64);
                    let st = builder.ins().iconst(types::I64, *stream as i64);
                    let call = builder.ins().call(pcg_func_ref, &[input, s, st]);
                    let result = builder.inst_results(call)[0];
                    store_slot(&mut builder, buffer_ptr, output_slots[0], result);
                }
                JitOp::PcgStreamConst(seed) => {
                    let input = load_slot(&mut builder, buffer_ptr, input_slots[0]);
                    let st = load_slot(&mut builder, buffer_ptr, input_slots[1]);
                    let s = builder.ins().iconst(types::I64, *seed as i64);
                    let call = builder.ins().call(pcg_stream_func_ref, &[input, st, s]);
                    let result = builder.inst_results(call)[0];
                    store_slot(&mut builder, buffer_ptr, output_slots[0], result);
                }
                JitOp::NOfConst(n, m) => {
                    let input = load_slot(&mut builder, buffer_ptr, input_slots[0]);
                    let n_val = builder.ins().iconst(types::I64, *n as i64);
                    let m_val = builder.ins().iconst(types::I64, *m as i64);
                    let call = builder.ins().call(n_of_func_ref, &[input, n_val, m_val]);
                    let result = builder.inst_results(call)[0];
                    store_slot(&mut builder, buffer_ptr, output_slots[0], result);
                }

                JitOp::SlotCall { kit, scratch_base } => {
                    // Gather the inputs into the frame, call the kit
                    // over them and the state's scratch, scatter the
                    // outputs back. The kit's address is an immediate:
                    // the kit is shared by every kernel compiled from
                    // the program and outlives the code.
                    let n_in = input_slots.len();
                    let n_out = output_slots.len();
                    let frame = |builder: &mut FunctionBuilder, n: usize| {
                        builder.create_sized_stack_slot(ir::StackSlotData::new(
                            ir::StackSlotKind::ExplicitSlot,
                            (n.max(1) * 8) as u32,
                            3,
                        ))
                    };
                    let in_frame = frame(&mut builder, n_in);
                    let out_frame = frame(&mut builder, n_out);
                    for (k, &s) in input_slots.iter().enumerate() {
                        let v = load_slot(&mut builder, buffer_ptr, s);
                        builder.ins().stack_store(v, in_frame, (k * 8) as i32);
                    }
                    let kit_ptr = builder
                        .ins()
                        .iconst(types::I64, std::sync::Arc::as_ptr(&kit.0) as usize as i64);
                    let in_ptr = builder.ins().stack_addr(types::I64, in_frame, 0);
                    let n_in_v = builder.ins().iconst(types::I64, n_in as i64);
                    let out_ptr = builder.ins().stack_addr(types::I64, out_frame, 0);
                    let n_out_v = builder.ins().iconst(types::I64, n_out as i64);
                    let base_v = builder.ins().iconst(types::I64, *scratch_base as i64);
                    let n_sc_v = builder.ins().iconst(types::I64, kit.0.scratch.len() as i64);
                    builder.ins().call(
                        slot_call_ref,
                        &[
                            kit_ptr,
                            in_ptr,
                            n_in_v,
                            out_ptr,
                            n_out_v,
                            scratch_ptr,
                            base_v,
                            n_sc_v,
                        ],
                    );
                    for (k, &s) in output_slots.iter().enumerate() {
                        let v = builder
                            .ins()
                            .stack_load(types::I64, out_frame, (k * 8) as i32);
                        store_slot(&mut builder, buffer_ptr, s, v);
                    }
                }

                JitOp::U64ToStr { scratch_base }
                | JitOp::I64ToStr { scratch_base }
                | JitOp::F64ToStr { scratch_base } => {
                    // The helper writes the digits into the step's entry
                    // and publishes the pair into the output slots.
                    let func = match jit_op {
                        JitOp::U64ToStr { .. } => u64_to_str_ref,
                        JitOp::I64ToStr { .. } => i64_to_str_ref,
                        _ => f64_to_str_ref,
                    };
                    let value = load_slot(&mut builder, buffer_ptr, input_slots[0]);
                    let base_v = builder.ins().iconst(types::I64, *scratch_base as i64);
                    let out_v = builder.ins().iconst(types::I64, output_slots[0] as i64);
                    builder
                        .ins()
                        .call(func, &[scratch_ptr, base_v, buffer_ptr, out_v, value]);
                }
                JitOp::JsonToStr { scratch_base } => {
                    let ptr = load_slot(&mut builder, buffer_ptr, input_slots[0]);
                    let len = load_slot(&mut builder, buffer_ptr, input_slots[1]);
                    let base_v = builder.ins().iconst(types::I64, *scratch_base as i64);
                    let out_v = builder.ins().iconst(types::I64, output_slots[0] as i64);
                    builder.ins().call(
                        json_to_str_ref,
                        &[scratch_ptr, base_v, buffer_ptr, out_v, ptr, len],
                    );
                }
                JitOp::StrConcat { scratch_base } => {
                    // The input pairs go into the frame in order; the
                    // helper appends each one's bytes into the entry.
                    let n_words = input_slots.len();
                    let frame = builder.create_sized_stack_slot(ir::StackSlotData::new(
                        ir::StackSlotKind::ExplicitSlot,
                        (n_words.max(1) * 8) as u32,
                        3,
                    ));
                    for (k, &s) in input_slots.iter().enumerate() {
                        let v = load_slot(&mut builder, buffer_ptr, s);
                        builder.ins().stack_store(v, frame, (k * 8) as i32);
                    }
                    let pairs_ptr = builder.ins().stack_addr(types::I64, frame, 0);
                    let n_v = builder.ins().iconst(types::I64, (n_words / 2) as i64);
                    let base_v = builder.ins().iconst(types::I64, *scratch_base as i64);
                    let out_v = builder.ins().iconst(types::I64, output_slots[0] as i64);
                    builder.ins().call(
                        str_concat_ref,
                        &[scratch_ptr, base_v, buffer_ptr, out_v, pairs_ptr, n_v],
                    );
                }

                JitOp::VecProduce { kind, scratch_base } => {
                    // The helper runs the node's body over the input
                    // words and publishes the pair from the step's
                    // entry.
                    let func = func_of(&vec_producer_refs, *kind);
                    let base_v = builder.ins().iconst(types::I64, *scratch_base as i64);
                    let out_v = builder.ins().iconst(types::I64, output_slots[0] as i64);
                    let words = load_words(&mut builder, buffer_ptr, input_slots, 4);
                    let mut args = vec![scratch_ptr, base_v, buffer_ptr, out_v];
                    args.extend(words);
                    builder.ins().call(func, &args);
                }
                JitOp::VecReduce(kind) => {
                    let func = func_of(&vec_reducer_refs, *kind);
                    let words = load_words(&mut builder, buffer_ptr, input_slots, 4);
                    let call = builder.ins().call(func, &words);
                    let bits = builder.inst_results(call)[0];
                    store_slot(&mut builder, buffer_ptr, output_slots[0], bits);
                }
                JitOp::RegLane(kind) => {
                    let func = func_of(&reg_lane_refs, *kind);
                    let words = load_words(&mut builder, buffer_ptr, input_slots, 3);
                    let call = builder.ins().call(func, &words);
                    let word = builder.inst_results(call)[0];
                    store_slot(&mut builder, buffer_ptr, output_slots[0], word);
                }
                JitOp::RegProduce(kind) => {
                    let func = func_of(&reg_producer_refs, *kind);
                    let out_v = builder.ins().iconst(types::I64, output_slots[0] as i64);
                    let words = load_words(&mut builder, buffer_ptr, input_slots, 4);
                    let mut args = vec![buffer_ptr, out_v];
                    args.extend(words);
                    builder.ins().call(func, &args);
                }
                JitOp::RegDotF32 => {
                    // The products at f32 precision, then the fixed
                    // tree ((p0+p1)+(p2+p3)) at f32, then widened: the
                    // node's body, operation for operation.
                    let a = load_reg128(&mut builder, buffer_ptr, input_slots[0], types::F32X4);
                    let b = load_reg128(&mut builder, buffer_ptr, input_slots[2], types::F32X4);
                    let p = builder.ins().fmul(a, b);
                    let p0 = builder.ins().extractlane(p, 0);
                    let p1 = builder.ins().extractlane(p, 1);
                    let p2 = builder.ins().extractlane(p, 2);
                    let p3 = builder.ins().extractlane(p, 3);
                    let s01 = builder.ins().fadd(p0, p1);
                    let s23 = builder.ins().fadd(p2, p3);
                    let s = builder.ins().fadd(s01, s23);
                    let wide = builder.ins().fpromote(types::F64, s);
                    store_slot_f64(&mut builder, buffer_ptr, output_slots[0], wide);
                }
                JitOp::RegShuffleConst(mask) => {
                    // Output byte i is input byte mask[i]: the word's
                    // bytes lie in memory in little-endian order, which
                    // is the order `shuffle` numbers its lanes.
                    let x = load_reg128(&mut builder, buffer_ptr, input_slots[0], types::I8X16);
                    let imm = builder
                        .func
                        .dfg
                        .immediates
                        .push(ir::ConstantData::from(&mask[..]));
                    let r = builder.ins().shuffle(x, x, imm);
                    store_reg128(&mut builder, buffer_ptr, output_slots[0], r);
                }

                JitOp::Fallback => {
                    // Can't JIT this node — skip (caller should
                    // not include fallback ops in JIT steps)
                }
            }
            if let Some((inst, mark)) = tracker_store {
                let calls = (mark..builder.func.dfg.num_insts()).any(|i| {
                    builder.func.dfg.insts[ir::Inst::from_u32(i as u32)]
                        .opcode()
                        .is_call()
                });
                if !calls {
                    builder.func.layout.remove_inst(inst);
                }
            }

            // Provenance: set clean[step_idx] = 1, then jump to skip block
            if let (Some(cp), Some(skip)) = (clean_ptr, skip_block) {
                let offset = builder.ins().iconst(types::I64, step_idx as i64);
                let addr = builder.ins().iadd(cp, offset);
                let one = builder.ins().iconst(types::I8, 1);
                builder.ins().store(ir::MemFlags::new(), one, addr, 0);
                builder.ins().jump(skip, &[]);
                builder.switch_to_block(skip);
                builder.seal_block(skip);
            }
        }

        builder.ins().return_(&[]);
        builder.finalize();
    }
    // Code that calls nothing cannot fail: no helper, no longjmp, no
    // panic. The kernel that runs it skips the catch.
    let fallible = ctx.func.layout.blocks().any(|block| {
        ctx.func
            .layout
            .block_insts(block)
            .any(|inst| ctx.func.dfg.insts[inst].opcode().is_call())
    });

    module
        .define_function(func_id, &mut ctx)
        .map_err(|e| format!("define function: {e}"))?;
    module.clear_context(&mut ctx);
    module
        .finalize_definitions()
        .map_err(|e| format!("finalize: {e}"))?;

    let code_ptr = module.get_finalized_function(func_id);
    // The kits the code calls, kept alive beside it.
    let kits: Vec<SlotKitRef> = steps
        .iter()
        .filter_map(|(op, _, _)| op.slot_kit().cloned())
        .collect();
    let code = super::kernels::JitCode::new(module, kits, fallible);

    if provenance {
        let prov_fn: NativeProvFn = unsafe { mem::transmute(code_ptr) };
        let dummy_raw: NativeFn = unsafe { mem::transmute(code_ptr) };
        Ok((dummy_raw, prov_fn, code))
    } else {
        let raw_fn: NativeFn = unsafe { mem::transmute(code_ptr) };
        let dummy_prov: NativeProvFn = unsafe { mem::transmute(code_ptr) };
        Ok((raw_fn, dummy_prov, code))
    }
}

// ── Buffer slot helpers ────────────────────────────────────

/// Load a u64 from buffer[slot].
fn load_slot(builder: &mut FunctionBuilder, buffer_ptr: ir::Value, slot: usize) -> ir::Value {
    let offset = (slot * 8) as i32;
    builder
        .ins()
        .load(types::I64, ir::MemFlags::trusted(), buffer_ptr, offset)
}

/// Store a u64 to buffer[slot].
fn store_slot(
    builder: &mut FunctionBuilder,
    buffer_ptr: ir::Value,
    slot: usize,
    value: ir::Value,
) -> ir::Inst {
    let offset = (slot * 8) as i32;
    builder
        .ins()
        .store(ir::MemFlags::trusted(), value, buffer_ptr, offset)
}

/// Cranelift vector type for a register lane index (the
/// `RegBinOp`/`RegSplat` vocabulary).
fn reg_lane_type(lane: u8) -> ir::Type {
    match lane {
        0 => types::I8X16,
        1 => types::I16X8,
        2 => types::I32X4,
        3 => types::I64X2,
        4 => types::F32X4,
        5 => types::F64X2,
        _ => unreachable!("register lane index out of range"),
    }
}

/// Load a 128-bit register value from its two consecutive slots
/// (an `Imm2` port occupies two consecutive slots, axiom S1). The buffer is only
/// 8-aligned, so the load must NOT carry the aligned flag —
/// `MemFlags::new()` permits unaligned 128-bit access.
fn load_reg128(
    builder: &mut FunctionBuilder,
    buffer_ptr: ir::Value,
    first_slot: usize,
    vt: ir::Type,
) -> ir::Value {
    let offset = (first_slot * 8) as i32;
    builder
        .ins()
        .load(vt, ir::MemFlags::new(), buffer_ptr, offset)
}

/// Store a 128-bit register value into its two consecutive slots.
fn store_reg128(
    builder: &mut FunctionBuilder,
    buffer_ptr: ir::Value,
    first_slot: usize,
    value: ir::Value,
) {
    let offset = (first_slot * 8) as i32;
    builder
        .ins()
        .store(ir::MemFlags::new(), value, buffer_ptr, offset);
}

/// `x` rounded half away from zero, as `f64::round` rounds: the
/// truncation, plus one in the sign of `x` when the fraction's
/// magnitude reaches a half. Exact: where the fraction is nonzero the
/// truncation is below 2^52, so the step is representable.
fn round_half_away(builder: &mut FunctionBuilder, x: ir::Value) -> ir::Value {
    let t = builder.ins().trunc(x);
    let frac = builder.ins().fsub(x, t);
    let mag = builder.ins().fabs(frac);
    let half = builder.ins().f64const(0.5);
    let reaches = builder
        .ins()
        .fcmp(ir::condcodes::FloatCC::GreaterThanOrEqual, mag, half);
    let one = builder.ins().f64const(1.0);
    let step = builder.ins().fcopysign(one, x);
    let up = builder.ins().fadd(t, step);
    builder.ins().select(reaches, up, t)
}

/// `x.clamp(lo, hi)` as `f64::clamp` computes it: `lo` when `x < lo`,
/// `hi` when `x > hi`, else `x` itself, so a negative zero and a NaN
/// pass through as they do there (`fmax`/`fmin` would return the
/// bound's zero for `-0.0`).
fn clamp_ir(
    builder: &mut FunctionBuilder,
    x: ir::Value,
    lo: ir::Value,
    hi: ir::Value,
) -> ir::Value {
    let below = builder.ins().fcmp(ir::condcodes::FloatCC::LessThan, x, lo);
    let above = builder
        .ins()
        .fcmp(ir::condcodes::FloatCC::GreaterThan, x, hi);
    let capped = builder.ins().select(above, hi, x);
    builder.ins().select(below, lo, capped)
}

/// `val.div_ceil(m)` for a nonzero `m`: the quotient, plus one when
/// the remainder is nonzero, with no sum that can overflow.
fn div_ceil(
    builder: &mut FunctionBuilder,
    val: ir::Value,
    m: ir::Value,
    one: ir::Value,
) -> ir::Value {
    let q = builder.ins().udiv(val, m);
    let r = builder.ins().urem(val, m);
    let zero = builder.ins().iconst(types::I64, 0);
    let inexact = builder.ins().icmp(ir::condcodes::IntCC::NotEqual, r, zero);
    let q1 = builder.ins().iadd(q, one);
    builder.ins().select(inexact, q1, q)
}

/// The function reference declared for one helper of a group.
fn func_of<K: PartialEq + Copy>(refs: &[(K, ir::FuncRef)], key: K) -> ir::FuncRef {
    refs.iter()
        .find(|(k, _)| *k == key)
        .map(|(_, r)| *r)
        .expect("every helper of the group is declared")
}

/// Load a step's input words in order, zero past the last, as the
/// arguments of a helper that takes a fixed count of words.
fn load_words(
    builder: &mut FunctionBuilder,
    buffer_ptr: ir::Value,
    input_slots: &[usize],
    n: usize,
) -> Vec<ir::Value> {
    (0..n)
        .map(|k| match input_slots.get(k) {
            Some(&s) => load_slot(builder, buffer_ptr, s),
            None => builder.ins().iconst(types::I64, 0),
        })
        .collect()
}

/// Load an f64 from buffer[slot] (bitcast from i64).
fn load_slot_f64(builder: &mut FunctionBuilder, buffer_ptr: ir::Value, slot: usize) -> ir::Value {
    let i64_val = load_slot(builder, buffer_ptr, slot);
    builder
        .ins()
        .bitcast(types::F64, ir::MemFlags::new(), i64_val)
}

/// Store an f64 to buffer[slot] (bitcast to i64).
fn store_slot_f64(
    builder: &mut FunctionBuilder,
    buffer_ptr: ir::Value,
    slot: usize,
    value: ir::Value,
) {
    let i64_val = builder
        .ins()
        .bitcast(types::I64, ir::MemFlags::new(), value);
    store_slot(builder, buffer_ptr, slot, i64_val);
}

// ── Tests ──────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jit_identity() {
        let steps = vec![(JitOp::Identity, vec![0], vec![1])];
        let mut output_map = HashMap::new();
        output_map.insert("out".into(), 1);
        let mut kernel = compile_jit_raw(1, 2, steps, output_map, Vec::new()).unwrap();
        kernel.eval(&[42]);
        assert_eq!(kernel.get("out"), 42);
    }

    #[test]
    fn jit_add_const() {
        let steps = vec![(JitOp::AddConst(100), vec![0], vec![1])];
        let mut output_map = HashMap::new();
        output_map.insert("out".into(), 1);
        let mut kernel = compile_jit_raw(1, 2, steps, output_map, Vec::new()).unwrap();
        kernel.eval(&[5]);
        assert_eq!(kernel.get("out"), 105);
    }

    #[test]
    fn jit_mul_const() {
        let steps = vec![(JitOp::MulConst(7), vec![0], vec![1])];
        let mut output_map = HashMap::new();
        output_map.insert("out".into(), 1);
        let mut kernel = compile_jit_raw(1, 2, steps, output_map, Vec::new()).unwrap();
        kernel.eval(&[6]);
        assert_eq!(kernel.get("out"), 42);
    }

    #[test]
    fn jit_mod_const() {
        let steps = vec![(JitOp::ModConst(100), vec![0], vec![1])];
        let mut output_map = HashMap::new();
        output_map.insert("out".into(), 1);
        let mut kernel = compile_jit_raw(1, 2, steps, output_map, Vec::new()).unwrap();
        kernel.eval(&[542]);
        assert_eq!(kernel.get("out"), 42);
    }

    #[test]
    fn jit_hash() {
        let steps = vec![(JitOp::Hash, vec![0], vec![1])];
        let mut output_map = HashMap::new();
        output_map.insert("out".into(), 1);
        let mut kernel = compile_jit_raw(1, 2, steps, output_map, Vec::new()).unwrap();

        kernel.eval(&[42]);
        let v1 = kernel.get("out");

        // Verify it matches the Rust xxh3 implementation
        let expected = xxhash_rust::xxh3::xxh3_64(&42u64.to_le_bytes());
        assert_eq!(v1, expected);
    }

    #[test]
    fn jit_hash_deterministic() {
        let steps = vec![(JitOp::Hash, vec![0], vec![1])];
        let mut output_map = HashMap::new();
        output_map.insert("out".into(), 1);
        let mut kernel = compile_jit_raw(1, 2, steps, output_map, Vec::new()).unwrap();

        kernel.eval(&[42]);
        let v1 = kernel.get("out");
        kernel.eval(&[42]);
        let v2 = kernel.get("out");
        assert_eq!(v1, v2);
    }

    #[test]
    fn jit_chain_hash_mod() {
        // hash(cycle) → mod(result, 1000000)
        let steps = vec![
            (JitOp::Hash, vec![0], vec![1]), // slot 1 = hash(coord 0)
            (JitOp::ModConst(1_000_000), vec![1], vec![2]), // slot 2 = slot 1 % 1M
        ];
        let mut output_map = HashMap::new();
        output_map.insert("user_id".into(), 2);
        let mut kernel = compile_jit_raw(1, 3, steps, output_map, Vec::new()).unwrap();

        kernel.eval(&[42]);
        let uid = kernel.get("user_id");
        assert!(uid < 1_000_000, "got {uid}");
    }

    #[test]
    fn jit_clamp_const() {
        let steps = vec![(JitOp::ClampConst(10, 50), vec![0], vec![1])];
        let mut output_map = HashMap::new();
        output_map.insert("out".into(), 1);
        let mut kernel = compile_jit_raw(1, 2, steps, output_map, Vec::new()).unwrap();

        kernel.eval(&[5]);
        assert_eq!(kernel.get("out"), 10); // below min

        kernel.eval(&[30]);
        assert_eq!(kernel.get("out"), 30); // in range

        kernel.eval(&[100]);
        assert_eq!(kernel.get("out"), 50); // above max
    }

    #[test]
    fn jit_interleave() {
        let steps = vec![(JitOp::Interleave, vec![0, 1], vec![2])];
        let mut output_map = HashMap::new();
        output_map.insert("out".into(), 2);
        let mut kernel = compile_jit_raw(2, 3, steps, output_map, Vec::new()).unwrap();

        kernel.eval(&[0b101, 0b010]);
        // Same as the Interleave node test: result = 0b011001
        assert_eq!(kernel.get("out"), 0b01_10_01);
    }

    #[test]
    fn jit_mixed_radix() {
        // 100 × 1000 × unbounded
        let steps = vec![(
            JitOp::MixedRadixConst(vec![100, 1000, 0]),
            vec![0],
            vec![1, 2, 3],
        )];
        let mut output_map = HashMap::new();
        output_map.insert("d0".into(), 1);
        output_map.insert("d1".into(), 2);
        output_map.insert("d2".into(), 3);
        let mut kernel = compile_jit_raw(1, 4, steps, output_map, Vec::new()).unwrap();

        // 4201337 → (37, 13, 42)
        kernel.eval(&[4_201_337]);
        assert_eq!(kernel.get("d0"), 37);
        assert_eq!(kernel.get("d1"), 13);
        assert_eq!(kernel.get("d2"), 42);
    }

    #[test]
    fn jit_unit_interval() {
        let steps = vec![(JitOp::UnitInterval, vec![0], vec![1])];
        let mut output_map = HashMap::new();
        output_map.insert("out".into(), 1);
        let mut kernel = compile_jit_raw(1, 2, steps, output_map, Vec::new()).unwrap();

        kernel.eval(&[0]);
        let v = f64::from_bits(kernel.get("out"));
        assert!((v - 0.0).abs() < 1e-10);

        kernel.eval(&[u64::MAX]);
        let v = f64::from_bits(kernel.get("out"));
        assert!((v - 1.0).abs() < 1e-10);
    }

    #[test]
    fn jit_f64_to_u64() {
        // Store 3.7 as f64 bits in coord slot, convert to u64
        let steps = vec![(JitOp::F64ToU64, vec![0], vec![1])];
        let mut output_map = HashMap::new();
        output_map.insert("out".into(), 1);
        let mut kernel = compile_jit_raw(1, 2, steps, output_map, Vec::new()).unwrap();

        kernel.eval(&[3.7f64.to_bits()]);
        assert_eq!(kernel.get("out"), 3); // truncate toward zero
    }

    #[test]
    fn jit_round_to_u64() {
        let steps = vec![(JitOp::RoundToU64, vec![0], vec![1])];
        let mut output_map = HashMap::new();
        output_map.insert("out".into(), 1);
        let mut kernel = compile_jit_raw(1, 2, steps, output_map, Vec::new()).unwrap();

        kernel.eval(&[3.7f64.to_bits()]);
        assert_eq!(kernel.get("out"), 4);

        kernel.eval(&[3.2f64.to_bits()]);
        assert_eq!(kernel.get("out"), 3);
    }

    #[test]
    fn jit_clamp_f64() {
        let steps = vec![(
            JitOp::ClampF64Const(0.0f64.to_bits(), 1.0f64.to_bits()),
            vec![0],
            vec![1],
        )];
        let mut output_map = HashMap::new();
        output_map.insert("out".into(), 1);
        let mut kernel = compile_jit_raw(1, 2, steps, output_map, Vec::new()).unwrap();

        kernel.eval(&[(-0.5f64).to_bits()]);
        assert_eq!(f64::from_bits(kernel.get("out")), 0.0);

        kernel.eval(&[0.5f64.to_bits()]);
        assert_eq!(f64::from_bits(kernel.get("out")), 0.5);

        kernel.eval(&[1.5f64.to_bits()]);
        assert_eq!(f64::from_bits(kernel.get("out")), 1.0);
    }

    #[test]
    fn jit_lerp() {
        let steps = vec![(
            JitOp::LerpConst(10.0f64.to_bits(), 20.0f64.to_bits()),
            vec![0],
            vec![1],
        )];
        let mut output_map = HashMap::new();
        output_map.insert("out".into(), 1);
        let mut kernel = compile_jit_raw(1, 2, steps, output_map, Vec::new()).unwrap();

        kernel.eval(&[0.0f64.to_bits()]);
        assert_eq!(f64::from_bits(kernel.get("out")), 10.0);

        kernel.eval(&[1.0f64.to_bits()]);
        assert_eq!(f64::from_bits(kernel.get("out")), 20.0);

        kernel.eval(&[0.5f64.to_bits()]);
        assert_eq!(f64::from_bits(kernel.get("out")), 15.0);
    }

    #[test]
    fn jit_scale_range() {
        let steps = vec![(
            JitOp::ScaleRangeConst(10.0f64.to_bits(), 10.0f64.to_bits()),
            vec![0],
            vec![1],
        )];
        let mut output_map = HashMap::new();
        output_map.insert("out".into(), 1);
        let mut kernel = compile_jit_raw(1, 2, steps, output_map, Vec::new()).unwrap();

        kernel.eval(&[0]);
        let v = f64::from_bits(kernel.get("out"));
        assert!((v - 10.0).abs() < 0.001);

        kernel.eval(&[u64::MAX]);
        let v = f64::from_bits(kernel.get("out"));
        assert!((v - 20.0).abs() < 0.001);
    }

    #[test]
    fn jit_quantize() {
        let steps = vec![(JitOp::QuantizeConst(10.0f64.to_bits()), vec![0], vec![1])];
        let mut output_map = HashMap::new();
        output_map.insert("out".into(), 1);
        let mut kernel = compile_jit_raw(1, 2, steps, output_map, Vec::new()).unwrap();

        kernel.eval(&[13.0f64.to_bits()]);
        assert_eq!(f64::from_bits(kernel.get("out")), 10.0);

        kernel.eval(&[17.0f64.to_bits()]);
        assert_eq!(f64::from_bits(kernel.get("out")), 20.0);
    }

    #[test]
    fn jit_discretize() {
        let steps = vec![(
            JitOp::DiscretizeConst(100.0f64.to_bits(), 10),
            vec![0],
            vec![1],
        )];
        let mut output_map = HashMap::new();
        output_map.insert("out".into(), 1);
        let mut kernel = compile_jit_raw(1, 2, steps, output_map, Vec::new()).unwrap();

        kernel.eval(&[0.0f64.to_bits()]);
        assert_eq!(kernel.get("out"), 0);

        kernel.eval(&[55.0f64.to_bits()]);
        assert_eq!(kernel.get("out"), 5);

        kernel.eval(&[99.0f64.to_bits()]);
        assert_eq!(kernel.get("out"), 9);

        // Clamp above range
        kernel.eval(&[200.0f64.to_bits()]);
        assert_eq!(kernel.get("out"), 9);
    }

    #[test]
    fn jit_chain_unit_interval_lerp() {
        // u64 → unit_interval → lerp(100, 200)
        let steps = vec![
            (JitOp::UnitInterval, vec![0], vec![1]),
            (
                JitOp::LerpConst(100.0f64.to_bits(), 200.0f64.to_bits()),
                vec![1],
                vec![2],
            ),
        ];
        let mut output_map = HashMap::new();
        output_map.insert("out".into(), 2);
        let mut kernel = compile_jit_raw(1, 3, steps, output_map, Vec::new()).unwrap();

        kernel.eval(&[0]);
        let v = f64::from_bits(kernel.get("out"));
        assert!((v - 100.0).abs() < 0.001);

        kernel.eval(&[u64::MAX]);
        let v = f64::from_bits(kernel.get("out"));
        assert!((v - 200.0).abs() < 0.001);
    }

    #[test]
    fn jit_multi_step_chain() {
        // cycle → add(10) → mul(3) → mod(100)
        let steps = vec![
            (JitOp::AddConst(10), vec![0], vec![1]),
            (JitOp::MulConst(3), vec![1], vec![2]),
            (JitOp::ModConst(100), vec![2], vec![3]),
        ];
        let mut output_map = HashMap::new();
        output_map.insert("out".into(), 3);
        let mut kernel = compile_jit_raw(1, 4, steps, output_map, Vec::new()).unwrap();

        kernel.eval(&[5]);
        // (5 + 10) * 3 = 45, 45 % 100 = 45
        assert_eq!(kernel.get("out"), 45);
    }

    // ── Parameter helper predicates ────────────────────────────

    #[test]
    fn jit_is_positive_check_passes_positive() {
        let steps = vec![(
            JitOp::IsPositiveCheck {
                name_ptr: 0,
                name_len: 0,
            },
            vec![0],
            vec![1],
        )];
        let mut output_map = HashMap::new();
        output_map.insert("out".into(), 1);
        let mut kernel = compile_jit_raw(1, 2, steps, output_map, Vec::new()).unwrap();
        kernel.eval(&[42]);
        assert_eq!(kernel.get("out"), 42);
        // Large values pass through unchanged — happy path is a
        // bare store, not a clamp.
        kernel.eval(&[u64::MAX]);
        assert_eq!(kernel.get("out"), u64::MAX);
    }

    #[test]
    fn jit_in_range_check_passes_interior() {
        let steps = vec![(JitOp::InRangeCheck(10, 100), vec![0], vec![1])];
        let mut output_map = HashMap::new();
        output_map.insert("out".into(), 1);
        let mut kernel = compile_jit_raw(1, 2, steps, output_map, Vec::new()).unwrap();
        kernel.eval(&[50]);
        assert_eq!(kernel.get("out"), 50);
        // Boundaries are inclusive.
        kernel.eval(&[10]);
        assert_eq!(kernel.get("out"), 10);
        kernel.eval(&[100]);
        assert_eq!(kernel.get("out"), 100);
    }

    // Violation paths longjmp back to `invoke_with_catch` and
    // surface as ordinary panics; the tests below catch them
    // in-process.

    #[test]
    fn jit_is_one_of_check_passes_allowed_values() {
        let steps = vec![(
            JitOp::IsOneOfCheck {
                allowed: vec![1, 2, 3, 5, 8],
                set_ptr: 0,
                set_len: 0,
            },
            vec![0],
            vec![1],
        )];
        let mut output_map = HashMap::new();
        output_map.insert("out".into(), 1);
        let mut kernel = compile_jit_raw(1, 2, steps, output_map, Vec::new()).unwrap();
        // Every allowed value passes straight through.
        for v in [1u64, 2, 3, 5, 8] {
            kernel.eval(&[v]);
            assert_eq!(kernel.get("out"), v);
        }
    }

    #[test]
    fn jit_is_one_of_check_accepts_single_element_allow_list() {
        // Degenerate case — one-value allow-list reduces to an
        // equality check with panic on mismatch.
        let steps = vec![(
            JitOp::IsOneOfCheck {
                allowed: vec![42],
                set_ptr: 0,
                set_len: 0,
            },
            vec![0],
            vec![1],
        )];
        let mut output_map = HashMap::new();
        output_map.insert("out".into(), 1);
        let mut kernel = compile_jit_raw(1, 2, steps, output_map, Vec::new()).unwrap();
        kernel.eval(&[42]);
        assert_eq!(kernel.get("out"), 42);
    }

    // ── Catchable panic from JIT predicate fails ──────────────
    //
    // The extern fail helpers use `_longjmp` back to the Rust
    // wrapper, which then raises a Rust `panic!` carrying the
    // violation message. The panic originates in Rust land
    // (the JIT frame has already been jumped past), so its
    // unwind works through Rust-personality FDEs and
    // `std::panic::catch_unwind` catches it normally.

    fn extract_panic_msg(payload: Box<dyn std::any::Any + Send + 'static>) -> String {
        payload
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
            .unwrap_or_else(|| "(non-string panic)".into())
    }

    #[test]
    fn jit_is_positive_violation_is_catchable() {
        let steps = vec![(
            JitOp::IsPositiveCheck {
                name_ptr: 0,
                name_len: 0,
            },
            vec![0],
            vec![1],
        )];
        let mut output_map = HashMap::new();
        output_map.insert("out".into(), 1);
        let mut kernel = compile_jit_raw(1, 2, steps, output_map, Vec::new()).unwrap();
        let err = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| kernel.eval(&[0])))
            .expect_err("JIT violation should panic");
        assert!(extract_panic_msg(err).contains("must be > 0"));
    }

    #[test]
    fn jit_in_range_violation_is_catchable() {
        let steps = vec![(JitOp::InRangeCheck(10, 100), vec![0], vec![1])];
        let mut output_map = HashMap::new();
        output_map.insert("out".into(), 1);
        let mut kernel = compile_jit_raw(1, 2, steps, output_map, Vec::new()).unwrap();

        let err = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| kernel.eval(&[5])))
            .expect_err("below-range should panic");
        assert!(extract_panic_msg(err).contains("outside [10, 100]"));

        let err = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| kernel.eval(&[500])))
            .expect_err("above-range should panic");
        assert!(extract_panic_msg(err).contains("outside [10, 100]"));
    }

    #[test]
    fn jit_is_one_of_violation_is_catchable() {
        let steps = vec![(
            JitOp::IsOneOfCheck {
                allowed: vec![1, 3, 5],
                set_ptr: 0,
                set_len: 0,
            },
            vec![0],
            vec![1],
        )];
        let mut output_map = HashMap::new();
        output_map.insert("out".into(), 1);
        let mut kernel = compile_jit_raw(1, 2, steps, output_map, Vec::new()).unwrap();
        let err = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| kernel.eval(&[2])))
            .expect_err("disallowed value should panic");
        assert!(extract_panic_msg(err).contains("not in allowed set"));
    }

    #[test]
    fn invoke_with_catch_restores_slot_after_foreign_panic() {
        // A non-JIT panic from inside `f()` (simulating a bug
        // in a hybrid-closure step or any other non-longjmp
        // path that may run between setjmp and return) must
        // still leave the thread-local JIT_JMP_BUF slot in a
        // consistent state. The next `invoke_with_catch` that
        // actually calls into JIT code should see a clean
        // sentinel.
        let caught = std::panic::catch_unwind(|| {
            invoke_with_catch(|| panic!("foreign panic"));
        });
        assert!(caught.is_err(), "foreign panic should propagate out");

        // Subsequent legitimate JIT violation is still caught.
        let steps = vec![(
            JitOp::IsPositiveCheck {
                name_ptr: 0,
                name_len: 0,
            },
            vec![0],
            vec![1],
        )];
        let mut output_map = HashMap::new();
        output_map.insert("out".into(), 1);
        let mut kernel = compile_jit_raw(1, 2, steps, output_map, Vec::new()).unwrap();
        let err = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| kernel.eval(&[0])))
            .expect_err("JIT violation should panic cleanly after foreign panic");
        assert!(extract_panic_msg(err).contains("must be > 0"));

        // And the happy path too — no stale pointer lingering.
        kernel.eval(&[42]);
        assert_eq!(kernel.get("out"), 42);
    }

    #[test]
    fn jit_kernel_survives_multiple_violations() {
        // After a caught violation the kernel remains usable —
        // the jmp_buf slot is correctly cleared and a
        // subsequent happy-path eval returns normally.
        let steps = vec![(
            JitOp::IsPositiveCheck {
                name_ptr: 0,
                name_len: 0,
            },
            vec![0],
            vec![1],
        )];
        let mut output_map = HashMap::new();
        output_map.insert("out".into(), 1);
        let mut kernel = compile_jit_raw(1, 2, steps, output_map, Vec::new()).unwrap();

        for _ in 0..3 {
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| kernel.eval(&[0])))
                .expect_err("violation should still panic");
        }
        // Happy path still works.
        kernel.eval(&[42]);
        assert_eq!(kernel.get("out"), 42);
    }
}
