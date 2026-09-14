// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! PCG-RXS-M-XS 64/64 random number generator nodes.
//!
//! These nodes implement the PCG (Permuted Congruential Generator) family
//! algorithm with the RXS-M-XS output permutation. The key property is
//! O(log N) seek: any position in the sequence can be computed directly
//! without iterating from the beginning. This makes it ideal for
//! deterministic parallel workloads where each thread jumps to its own
//! region of the sequence.
//!
//! Three nodes are provided:
//!
//! - [`Pcg`] — fixed seed and stream, position is the wire input
//! - [`PcgStream`] — fixed seed, both position and stream are wire inputs
//! - [`CycleWalk`] — bijective permutation of `[0, range)` via cycle-walking
//!
//! `CycleWalk` uses multi-source `#[poly_const]` to derive a
//! `CycleWalkState` from `(range, seed, stream)` at construction
//! time, plus `compiled_u64 = ...` / `jit_constants = ...` overrides
//! that capture the cached Feistel state by Copy and publish
//! `[range, seed, inc]` to the JIT classifier.

use polydat::ast::CompiledU64Op;
#[cfg(test)]
use polydat::ast::{PolydatNode, Value};

// =================================================================
// PCG-RXS-M-XS 64/64 core algorithm
// =================================================================

pub use polydat::numeric::pcg::{
    CycleWalkState, FEISTEL_ROUNDS, MULT, build_cycle_walk_state, cycle_walk_inner, pcg_output,
    pcg_seek,
};

/// PCG-RXS-M-XS 64/64 random number generator with fixed seed and stream.
///
/// Signature: `pcg(input: u64, seed: u64, stream: u64) -> u64`
///
/// The `seed` and `stream` are init-time constants baked into the node.
/// The `input` wire selects which element of the sequence to return.
/// Seeking is O(log N) so any position can be accessed directly.
///
/// Use this when every thread/cycle needs an independent, deterministic
/// random value from the same generator. The output is a full 64-bit
/// pseudo-random value suitable for feeding into range reduction,
/// unit-interval mapping, or distribution sampling.
///
/// JIT level: P2 (auto-emitted `compiled_u64` closure with captured
/// `seed` and `stream`; the body recomputes `inc = 2*stream + 1`
/// per call — a single mul+add, negligible vs the seek work).
/// `jit_constants`: `[seed, stream]` in declaration order.
#[polydat::polydat_node(category = Permutation)]
fn pcg(
    input: u64,
    #[poly_default(0u64)] seed: Const<u64>,
    #[poly_default(0u64)] stream: Const<u64>,
) -> u64 {
    let inc = 2u64.wrapping_mul(*stream).wrapping_add(1);
    pcg_seek(*seed, inc, input)
}

/// PCG-RXS-M-XS 64/64 with runtime stream selection.
///
/// Signature: `pcg_stream(input: u64, stream: u64, seed: u64) -> u64`
///
/// Like [`Pcg`], but the stream is a wire input rather than a constant.
/// This allows each row or partition to use a different stream while
/// sharing the same seed, producing independent sequences that are
/// statistically uncorrelated.
///
/// Use this when the stream identity is data-dependent (e.g., derived
/// from a partition key) and cannot be fixed at assembly time.
///
/// JIT level: P2 (auto-emitted `compiled_u64` closure with captured
/// `seed`; `inc` derives from the wire-fed `stream` each call).
#[polydat::polydat_node(category = Permutation)]
fn pcg_stream(input: u64, stream: u64, #[poly_default(0u64)] seed: Const<u64>) -> u64 {
    let inc = 2u64.wrapping_mul(stream).wrapping_add(1);
    pcg_seek(*seed, inc, input)
}

/// `compiled_u64` override — captures the pre-computed Feistel
/// state from `&Self` by Copy and returns a closure that walks
/// the input through the bijection. The override receives `&Self`
/// so setup-derived state is reachable without exposing the
/// macro-internal struct shape to user code.
fn cycle_walk_jit(node: &CycleWalk) -> CompiledU64Op {
    let range = node.range;
    let half_bits = node.state.half_bits;
    let half_mask = node.state.half_mask;
    let round_keys = node.state.round_keys;
    Box::new(move |inputs, outputs| {
        outputs[0] = cycle_walk_inner(inputs[0], range, half_bits, half_mask, &round_keys);
    })
}

fn cycle_walk_jit_constants(node: &CycleWalk) -> Vec<u64> {
    vec![node.range, node.seed, node.state.inc]
}

/// Bijective permutation of `[0, range)` via cycle-walking over PCG.
///
/// Signature: `cycle_walk(position: u64, range: u64, seed: u64, stream: u64) -> u64`
///
/// Maps every integer in `[0, range)` to a unique integer in `[0, range)`
/// (a permutation). Internally uses a 6-round Feistel network operating
/// on the bit-width of range, with PCG-derived round keys, then
/// cycle-walks: if the Feistel output is >= range, it is fed back as
/// input. Because the Feistel cipher is a bijection on the power-of-two
/// domain and the mask is at most 2x range, each cycle-walk iteration
/// has >= 50% chance of landing in range, giving fast expected
/// termination (~2 iterations).
///
/// Use this when you need a shuffle or bijective mapping: e.g., visiting
/// every row in a table exactly once in a pseudo-random order, or
/// generating unique IDs without a tracking structure.
///
/// The `range`, `seed`, and `stream` are init-time constants.
///
/// JIT level: P2 — macro-authored via `compiled_u64 = ...` /
/// `jit_constants = ...` overrides that capture the pre-computed
/// `CycleWalkState`. Exposes `jit_constants`: `[range, seed, inc]`.
#[polydat::polydat_node(
    category = Permutation,
    compiled_u64 = cycle_walk_jit,
    jit_constants = cycle_walk_jit_constants,
)]
fn cycle_walk(
    position: u64,
    range: Const<u64>,
    #[poly_default(0u64)] seed: Const<u64>,
    #[poly_default(0u64)] stream: Const<u64>,
    #[poly_const(build_cycle_walk_state, from = (range, seed, stream))] state: &CycleWalkState,
) -> u64 {
    let _ = seed;
    let _ = stream;
    cycle_walk_inner(
        position,
        *range,
        state.half_bits,
        state.half_mask,
        &state.round_keys,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    // ----- pcg_seek / pcg_output unit tests -----

    #[test]
    fn pcg_output_deterministic() {
        // Same state must always produce the same output.
        let a = pcg_output(123456789);
        let b = pcg_output(123456789);
        assert_eq!(a, b);
    }

    #[test]
    fn pcg_seek_position_zero_vs_one() {
        let seed = 42u64;
        let inc = 1u64; // stream 0
        let v0 = pcg_seek(seed, inc, 0);
        let v1 = pcg_seek(seed, inc, 1);
        assert_ne!(v0, v1, "different positions must produce different values");
    }

    #[test]
    fn pcg_seek_deterministic() {
        let seed = 0xDEAD_BEEF;
        let inc = 3;
        let a = pcg_seek(seed, inc, 1000);
        let b = pcg_seek(seed, inc, 1000);
        assert_eq!(a, b);
    }

    #[test]
    fn pcg_seek_sequential_matches_step() {
        // Verify that seek(N) produces the same result as stepping
        // through the LCG N times.
        let seed = 77u64;
        let inc = 5u64;
        let n = 50u64;

        // Step through manually
        let mut state = seed;
        for _ in 0..n {
            state = state.wrapping_mul(MULT).wrapping_add(inc);
        }
        let stepped = pcg_output(state);

        let seeked = pcg_seek(seed, inc, n);
        assert_eq!(
            stepped, seeked,
            "seek({n}) must match {n} sequential LCG steps"
        );
    }

    // ----- Pcg node tests -----

    #[test]
    fn pcg_node_deterministic() {
        let node = Pcg::new(42, 0);
        let mut out = [Value::None];
        node.eval(&[Value::U64(100)], &mut out);
        let first = out[0].as_u64();
        node.eval(&[Value::U64(100)], &mut out);
        assert_eq!(
            first,
            out[0].as_u64(),
            "same position must give same result"
        );
    }

    #[test]
    fn pcg_node_different_positions() {
        let node = Pcg::new(42, 0);
        let mut out1 = [Value::None];
        let mut out2 = [Value::None];
        node.eval(&[Value::U64(0)], &mut out1);
        node.eval(&[Value::U64(1)], &mut out2);
        assert_ne!(out1[0].as_u64(), out2[0].as_u64());
    }

    #[test]
    fn pcg_node_different_seeds() {
        let a = Pcg::new(1, 0);
        let b = Pcg::new(2, 0);
        let mut out_a = [Value::None];
        let mut out_b = [Value::None];
        a.eval(&[Value::U64(50)], &mut out_a);
        b.eval(&[Value::U64(50)], &mut out_b);
        assert_ne!(
            out_a[0].as_u64(),
            out_b[0].as_u64(),
            "different seeds should produce different values"
        );
    }

    #[test]
    fn pcg_node_different_streams() {
        let a = Pcg::new(42, 0);
        let b = Pcg::new(42, 1);
        let mut out_a = [Value::None];
        let mut out_b = [Value::None];
        a.eval(&[Value::U64(50)], &mut out_a);
        b.eval(&[Value::U64(50)], &mut out_b);
        assert_ne!(
            out_a[0].as_u64(),
            out_b[0].as_u64(),
            "different streams should produce different values"
        );
    }

    #[test]
    fn pcg_compiled_matches_eval() {
        let node = Pcg::new(99, 7);
        let compiled = node.compiled_u64().expect("Pcg must provide compiled_u64");
        for pos in 0..100u64 {
            let mut eval_out = [Value::None];
            node.eval(&[Value::U64(pos)], &mut eval_out);
            let mut comp_out = [0u64];
            compiled(&[pos], &mut comp_out);
            assert_eq!(
                eval_out[0].as_u64(),
                comp_out[0],
                "compiled and eval must agree at position {pos}"
            );
        }
    }

    #[test]
    fn pcg_jit_constants() {
        // Macro auto-emits jit_constants in declaration order:
        // [seed, stream]. Phase-3 classifier doesn't special-case
        // pcg (it rides the Fallback path), so the precise layout
        // is informational; the contract is "values the closure
        // depends on", and the body recomputes inc from stream.
        let node = Pcg::new(42, 7);
        let consts = node.jit_constants();
        assert_eq!(consts.len(), 2);
        assert_eq!(consts[0], 42, "first constant is seed");
        assert_eq!(
            consts[1], 7,
            "second constant is stream (inc = 2*stream+1 derived in body)"
        );
    }

    // ----- PcgStream node tests -----

    #[test]
    fn pcg_stream_deterministic() {
        let node = PcgStream::new(42);
        let mut out = [Value::None];
        node.eval(&[Value::U64(100), Value::U64(3)], &mut out);
        let first = out[0].as_u64();
        node.eval(&[Value::U64(100), Value::U64(3)], &mut out);
        assert_eq!(first, out[0].as_u64());
    }

    #[test]
    fn pcg_stream_independence() {
        let node = PcgStream::new(42);
        let mut out_a = [Value::None];
        let mut out_b = [Value::None];
        node.eval(&[Value::U64(50), Value::U64(0)], &mut out_a);
        node.eval(&[Value::U64(50), Value::U64(1)], &mut out_b);
        assert_ne!(
            out_a[0].as_u64(),
            out_b[0].as_u64(),
            "different stream_ids should produce different values"
        );
    }

    #[test]
    fn pcg_stream_matches_fixed_pcg() {
        // PcgStream with a fixed stream_id should produce the same
        // output as Pcg constructed with that stream.
        let fixed = Pcg::new(42, 5);
        let dynamic = PcgStream::new(42);
        for pos in 0..50u64 {
            let mut f_out = [Value::None];
            let mut d_out = [Value::None];
            fixed.eval(&[Value::U64(pos)], &mut f_out);
            dynamic.eval(&[Value::U64(pos), Value::U64(5)], &mut d_out);
            assert_eq!(
                f_out[0].as_u64(),
                d_out[0].as_u64(),
                "PcgStream must match Pcg for same seed/stream at position {pos}"
            );
        }
    }

    #[test]
    fn pcg_stream_compiled_matches_eval() {
        let node = PcgStream::new(99);
        let compiled = node
            .compiled_u64()
            .expect("PcgStream must provide compiled_u64");
        for pos in 0..50u64 {
            for stream in 0..5u64 {
                let mut eval_out = [Value::None];
                node.eval(&[Value::U64(pos), Value::U64(stream)], &mut eval_out);
                let mut comp_out = [0u64];
                compiled(&[pos, stream], &mut comp_out);
                assert_eq!(
                    eval_out[0].as_u64(),
                    comp_out[0],
                    "compiled and eval must agree at pos={pos}, stream={stream}"
                );
            }
        }
    }

    // ----- CycleWalk node tests -----

    #[test]
    fn cycle_walk_bounded() {
        let node = CycleWalk::new(100, 42, 0);
        let mut out = [Value::None];
        for i in 0..200u64 {
            node.eval(&[Value::U64(i)], &mut out);
            assert!(
                out[0].as_u64() < 100,
                "output {} >= range 100",
                out[0].as_u64()
            );
        }
    }

    #[test]
    fn cycle_walk_deterministic() {
        let node = CycleWalk::new(1000, 42, 0);
        let mut out = [Value::None];
        node.eval(&[Value::U64(77)], &mut out);
        let first = out[0].as_u64();
        node.eval(&[Value::U64(77)], &mut out);
        assert_eq!(first, out[0].as_u64());
    }

    #[test]
    fn cycle_walk_bijective_small() {
        // For inputs [0, range), the mapping must be a permutation:
        // every output is unique and within [0, range).
        let range = 50u64;
        let node = CycleWalk::new(range, 42, 0);
        let mut seen = HashSet::new();
        let mut out = [Value::None];
        for i in 0..range {
            node.eval(&[Value::U64(i)], &mut out);
            let v = out[0].as_u64();
            assert!(v < range, "output {v} out of range [0, {range})");
            assert!(seen.insert(v), "duplicate output {v} at position {i}");
        }
        assert_eq!(
            seen.len(),
            range as usize,
            "must produce exactly {range} distinct values"
        );
    }

    #[test]
    fn cycle_walk_bijective_power_of_two() {
        // Powers of two are a common edge case.
        let range = 64u64;
        let node = CycleWalk::new(range, 123, 7);
        let mut seen = HashSet::new();
        let mut out = [Value::None];
        for i in 0..range {
            node.eval(&[Value::U64(i)], &mut out);
            let v = out[0].as_u64();
            assert!(v < range);
            assert!(seen.insert(v), "duplicate at {i}");
        }
        assert_eq!(seen.len(), range as usize);
    }

    #[test]
    fn cycle_walk_compiled_matches_eval() {
        let node = CycleWalk::new(200, 42, 3);
        let compiled = node
            .compiled_u64()
            .expect("CycleWalk must provide compiled_u64");
        for pos in 0..200u64 {
            let mut eval_out = [Value::None];
            node.eval(&[Value::U64(pos)], &mut eval_out);
            let mut comp_out = [0u64];
            compiled(&[pos], &mut comp_out);
            assert_eq!(
                eval_out[0].as_u64(),
                comp_out[0],
                "compiled and eval must agree at position {pos}"
            );
        }
    }

    #[test]
    fn cycle_walk_jit_constants() {
        let node = CycleWalk::new(500, 42, 7);
        let consts = node.jit_constants();
        assert_eq!(consts.len(), 3);
        assert_eq!(consts[0], 500, "first constant is range");
        assert_eq!(consts[1], 42, "second constant is seed");
        assert_eq!(consts[2], 2 * 7 + 1, "third constant is inc");
    }

    #[test]
    #[should_panic(expected = "range must be > 0")]
    fn cycle_walk_zero_range_panics() {
        CycleWalk::new(0, 42, 0);
    }

    #[test]
    fn cycle_walk_range_one() {
        // With range=1, every input must map to 0.
        let node = CycleWalk::new(1, 42, 0);
        let mut out = [Value::None];
        for i in 0..10u64 {
            node.eval(&[Value::U64(i)], &mut out);
            assert_eq!(out[0].as_u64(), 0);
        }
    }
}
