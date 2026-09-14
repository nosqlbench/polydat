// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! The PCG-RXS-M-XS generator, its O(log N) seek, and the Feistel
//! cycle walk: the bodies of `pcg`, `pcg_stream`, and `cycle_walk`,
//! which the native lowerings and the comprehension strategies call.

use crate::derive_support::PolydatSetup;

/// LCG multiplier for the 64-bit state.
/// The LCG multiplier of the 64-bit state.
pub const MULT: u64 = 6364136223846793005;

/// Apply the RXS-M-XS output permutation to an LCG state.
///
/// This is the bit-mixing function that turns correlated LCG state
/// into high-quality pseudo-random output.
#[inline]
pub fn pcg_output(state: u64) -> u64 {
    let word = ((state >> ((state >> 59) + 5)) ^ state).wrapping_mul(12605985483714917081);
    (word >> 43) ^ word
}

/// Seek to an arbitrary position in the PCG sequence in O(log N) time.
///
/// Uses the "distance" algorithm that exponentiates the LCG recurrence
/// via repeated squaring, equivalent to computing `state_N` directly
/// from `seed` without iterating through positions 0..N.
///
/// - `seed`: initial LCG state
/// - `inc`: LCG increment (must be odd; typically `2 * stream + 1`)
/// - `position`: the sequence index to seek to
#[inline]
pub fn pcg_seek(seed: u64, inc: u64, position: u64) -> u64 {
    let mut cur_mult = MULT;
    let mut cur_plus = inc;
    let mut acc_mult: u64 = 1;
    let mut acc_plus: u64 = 0;
    let mut delta = position;
    while delta > 0 {
        if delta & 1 != 0 {
            acc_mult = acc_mult.wrapping_mul(cur_mult);
            acc_plus = acc_plus.wrapping_mul(cur_mult).wrapping_add(cur_plus);
        }
        cur_plus = cur_mult.wrapping_add(1).wrapping_mul(cur_plus);
        cur_mult = cur_mult.wrapping_mul(cur_mult);
        delta >>= 1;
    }
    let state = acc_mult.wrapping_mul(seed).wrapping_add(acc_plus);
    pcg_output(state)
}

// =================================================================
// Polydat Nodes
// =================================================================

/// Number of Feistel rounds. 6 rounds provides good diffusion.
pub const FEISTEL_ROUNDS: usize = 6;

/// Pre-computed Feistel state for the `cycle_walk` node. Built once at
/// construction from `(range, seed, stream)` via the multi-source
/// `#[poly_const]` setup; consumed read-only by every cycle and by
/// the `compiled_u64` override's captured closure.
pub struct CycleWalkState {
    /// Number of bits per Feistel half (total domain is 2^(2*half_bits)).
    pub half_bits: u32,
    /// Bitmask for each half: `(1 << half_bits) - 1`.
    pub half_mask: u64,
    /// PCG-derived LCG increment, `2 * stream + 1`. Published to
    /// the JIT classifier as the third element of `jit_constants`.
    pub inc: u64,
    /// Pre-computed round keys derived from seed and stream.
    pub round_keys: [u64; FEISTEL_ROUNDS],
}

impl PolydatSetup for CycleWalkState {}

/// Joint Feistel-state derivation. Single-call construction-time
/// invocation per node instance; the macro emits the call inside
/// the generated `CycleWalk::new(range, seed, stream)`.
///
/// Panics if `range` is 0 — preserves the construction-time
/// validation contract from the pre-Phase-E hand-written form.
pub fn build_cycle_walk_state(range: u64, seed: u64, stream: u64) -> CycleWalkState {
    assert!(range > 0, "CycleWalk range must be > 0");
    let inc = 2u64.wrapping_mul(stream).wrapping_add(1);

    // Compute the total bit width needed, then round up to even
    // so the Feistel halves are balanced.
    let min_bits = if range <= 1 {
        2 // minimum 2 bits for a balanced Feistel
    } else {
        let b = 64 - (range - 1).leading_zeros();
        if !b.is_multiple_of(2) {
            b + 1
        } else {
            b.max(2)
        }
    };
    let half_bits = min_bits / 2;
    let half_mask = (1u64 << half_bits) - 1;

    // Derive round keys from seed and inc using the PCG itself.
    let mut round_keys = [0u64; FEISTEL_ROUNDS];
    for (i, key) in round_keys.iter_mut().enumerate() {
        *key = pcg_seek(seed, inc, i as u64 + 1_000_000_000);
    }

    CycleWalkState {
        half_bits,
        half_mask,
        inc,
        round_keys,
    }
}

#[inline]
fn feistel_round_fn(half: u64, round_key: u64) -> u64 {
    let x = half
        .wrapping_mul(0x9E3779B97F4A7C15)
        .wrapping_add(round_key);
    let x = ((x >> 32) ^ x).wrapping_mul(0xD6E8FEB86659FD93);
    (x >> 32) ^ x
}

/// Apply a balanced Feistel network: a bijection on `[0, 2^total_bits)`.
///
/// The value is split into two halves of `half_bits` each (total_bits
/// is always even -- we round up). Standard 6-round balanced Feistel
/// with pre-computed round keys ensures bijectivity.
#[inline]
fn feistel_encrypt(
    value: u64,
    half_bits: u32,
    half_mask: u64,
    round_keys: &[u64; FEISTEL_ROUNDS],
) -> u64 {
    let mut left = (value >> half_bits) & half_mask;
    let mut right = value & half_mask;

    for key in round_keys.iter() {
        let new_right = left ^ (feistel_round_fn(right, *key) & half_mask);
        left = right;
        right = new_right;
    }

    (left << half_bits) | right
}

/// Apply cycle-walking with the Feistel bijection.
///
/// The input `value` is first reduced to `[0, range)` via modular
/// reduction so that out-of-range inputs are accepted gracefully.
/// Starting from a value in `[0, range)`, cycle-walking is guaranteed
/// to terminate because the Feistel permutation's cycle through that
/// value must re-enter `[0, range)`.
#[inline]
pub fn cycle_walk_inner(
    mut value: u64,
    range: u64,
    half_bits: u32,
    half_mask: u64,
    round_keys: &[u64; FEISTEL_ROUNDS],
) -> u64 {
    if range == 1 {
        return 0;
    }
    // Ensure we start in [0, range) so cycle-walk terminates.
    value %= range;
    loop {
        value = feistel_encrypt(value, half_bits, half_mask, round_keys);
        if value < range {
            return value;
        }
    }
}
