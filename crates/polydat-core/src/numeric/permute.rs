// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! The Galois LFSR permutation: one body for the `shuffle` node and for
//! `jit_shuffle`, the helper native code calls.
//!
//! The two carried a copy each until 2026-09-22, and the copies drifted
//! the way copies do — a zero `size` divided by zero in both, and a
//! guard written into one would have made the engines disagree about
//! the same program. They are one function now, which is what this
//! module is for.
//!
//! The feedback polynomials come with it. A shuffle whose polynomial
//! does not match its register width is not a permutation, so the table
//! and the body are one thing to keep honest, not two; splitting them
//! across the crates is what let the bodies diverge in the first place.
//! `polydat-nodes` re-exports the selection helpers under their old
//! paths.

/// Number of banks (feedback polynomials) stored per register width.
const BANKS_PER_WIDTH: usize = 8;

/// Galois LFSR feedback polynomials, 8 banks per register width 4..64.
/// Indexed as FEEDBACK_BANKS[(width - 4) * 8 + bank].
/// Widths with fewer than 8 known polynomials repeat the last one.
const FEEDBACK_BANKS: [u64; 61 * BANKS_PER_WIDTH] = include!("metashift_banks.inc");

/// Return the feedback polynomial for a given register width and bank.
///
/// `width` must be 4..=64. `bank` selects among different polynomials
/// for the same width (modulo the number of available banks). Different
/// banks produce different permutation orderings over the same range.
pub fn feedback_for_width_and_bank(width: u32, bank: usize) -> u64 {
    assert!(
        (4..=64).contains(&width),
        "LFSR width must be 4..64, got {width}"
    );
    let base = (width as usize - 4) * BANKS_PER_WIDTH;
    FEEDBACK_BANKS[base + (bank % BANKS_PER_WIDTH)]
}

/// Return the default (bank 0) feedback polynomial for a given width.
pub fn feedback_for_width(width: u32) -> u64 {
    feedback_for_width_and_bank(width, 0)
}

/// Return the minimum register width needed to represent `period` values.
pub fn width_for_period(period: u64) -> u32 {
    assert!(period > 0, "period must be positive");
    let bits = 64 - period.leading_zeros();
    bits.max(4) // minimum 4-bit LFSR
}

/// Convenience: derive a bank-0 feedback polynomial directly from a
/// shuffle `size`. Callers building a `Shuffle` node from an outer
/// "size" parameter use this rather than tracking width / bank
/// manually.
pub fn feedback_for_size(size: u64) -> u64 {
    feedback_for_width_and_bank(width_for_period(size), 0)
}

/// One step of the Galois LFSR: shift right, and fold in the feedback
/// polynomial when the bit shifted out was set.
///
/// Bijective on the non-zero registers, which is what makes the shuffle
/// above it a permutation rather than a hash. Zero is a fixed point and
/// is kept out of range by the 1-based normalization.
#[inline]
pub fn lfsr_step(register: u64, feedback: u64) -> u64 {
    let lsb = register & 1;
    let shifted = register >> 1;
    // The `-lsb` trick: lsb=1 gives all ones, so the mask passes the
    // feedback; lsb=0 gives zero, so it blocks it. Branch-free, and the
    // same instruction sequence native code would have emitted inline.
    shifted ^ (lsb.wrapping_neg() & feedback)
}

/// Map `input` into `[min, min + size)`, visiting every value of that
/// range exactly once per cycle.
///
/// The LFSR cannot produce zero, so the range is worked 1-based and
/// denormalized on the way out. A register that steps past `size`
/// is rejected and stepped again, which is what makes a range that is
/// not a power of two come out whole.
///
/// # Degenerate pairs
///
/// Three pairs describe no permutation, and all three answer `min`, the
/// range's floor: the empty range below, a `feedback` that walks the
/// register to zero, and one that cycles without ever landing in range.
/// Each is reachable with the node's own defaults or with the arbitrary
/// constants a fuzzer supplies, so each is answered in bounded time
/// rather than trapped. The result is always inside `[min, min + size)`,
/// or `min` when that interval is empty.
///
/// `size == 0` is the empty range `[min, min)`, which has no value to
/// permute onto, and it is reachable without being asked for: the
/// node's `size` defaults to zero, so `shuffle(x)` and
/// `shuffle(x, feedback)` both land here. It answers `min`, the range's
/// own floor, as `hash_range` answers `0` for the same degenerate
/// bound. Left unanswered it is not merely undefined but a trap:
/// `input % size` divides by zero, and were that defined the rejection
/// loop could never terminate, since the register starts at 1 and the
/// exit condition wants `register <= 0`.
#[inline]
pub fn shuffle_bounded(input: u64, feedback: u64, size: u64, min: u64) -> u64 {
    if size == 0 {
        return min;
    }

    // Normalize to the 1-based LFSR range (the LFSR cannot produce 0).
    let mut register = (input % size) + 1;

    // A maximal polynomial of width `w` walks every non-zero register
    // below `2^w` before it repeats, and `width_for_period` picks `w`
    // so that `size < 2^w`. A well-formed pair therefore lands in
    // `[1, size]` within `2^w` steps. Spending more proves the step is
    // cycling in a subset that misses the range, and a deterministic
    // map over finite state that has repeated will repeat forever — so
    // the budget is not a timeout but the point past which "has not
    // landed" and "cannot land" are the same statement.
    let width = width_for_period(size);
    let budget = if width >= 63 { u64::MAX } else { 1u64 << width };
    let mut steps = 0u64;

    // Rejection sampling: step until the register lands in range.
    loop {
        register = lfsr_step(register, feedback);
        if register == 0 {
            // The sequence collapsed. Zero is the LFSR's fixed point
            // and sits outside the 1-based range, so there is nothing
            // to denormalize — `register - 1` would wrap. It is
            // reachable whenever `feedback` is not a polynomial for
            // this width, and above all when it is zero, which is the
            // node's default: the step degenerates to a plain shift and
            // walks to zero. Small ranges reach it at once, `size == 1`
            // for every input. Answer the floor, as the empty range
            // does.
            return min;
        }
        if register <= size {
            break;
        }
        steps += 1;
        if steps >= budget {
            // Not a polynomial for this width. Same answer as the
            // other degenerate pairs, and reached in bounded time.
            return min;
        }
    }

    // Denormalize back into [min, min + size).
    (register - 1) + min
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The property the node exists for: every value of the range is
    /// visited exactly once.
    #[test]
    fn a_bounded_shuffle_is_a_permutation() {
        let size = 1000u64;
        let feedback = feedback_for_size(size);
        let mut seen = vec![false; size as usize];
        for input in 0..size {
            let out = shuffle_bounded(input, feedback, size, 0);
            assert!(out < size, "{out} out of range for size {size}");
            assert!(!seen[out as usize], "{out} visited twice");
            seen[out as usize] = true;
        }
        assert!(seen.iter().all(|b| *b), "not every value was visited");
    }

    /// The degenerate bound, answered rather than trapped.
    #[test]
    fn an_empty_range_answers_its_floor() {
        for min in [0u64, 7, u64::MAX] {
            for input in [0u64, 1, 42, u64::MAX] {
                assert_eq!(shuffle_bounded(input, 0, 0, min), min);
            }
        }
    }

    /// A one-value range is the only value, whatever the input.
    #[test]
    fn a_single_value_range_is_that_value() {
        let feedback = feedback_for_size(1);
        for input in [0u64, 1, 42, u64::MAX] {
            assert_eq!(shuffle_bounded(input, feedback, 1, 5), 5);
        }
    }

    /// Zero is the `feedback` default, and it is not a polynomial: the
    /// step degenerates to a plain shift that walks the register to
    /// zero, which is outside the 1-based range. Answered, not wrapped
    /// — `register - 1` underflowed here before.
    #[test]
    fn a_collapsed_sequence_answers_the_floor() {
        for size in [1u64, 2, 3, 16, 1000] {
            for input in [0u64, 1, 42, u64::MAX] {
                let out = shuffle_bounded(input, 0, size, 9);
                assert!(
                    (9..9 + size).contains(&out),
                    "size={size} input={input} left the range with {out}"
                );
            }
        }
    }

    /// The budget must never cut a well-formed pair short. Every bank
    /// is a polynomial for its width, so every one of them still walks
    /// the whole range: if the bound were too tight this is what would
    /// break, and it would break as a lost value rather than as a
    /// panic.
    #[test]
    fn a_valid_polynomial_never_exhausts_the_budget() {
        for size in [1u64, 2, 5, 16, 17, 100, 255, 256, 1000] {
            let width = width_for_period(size);
            for bank in 0..8 {
                let feedback = feedback_for_width_and_bank(width, bank);
                let mut seen = vec![false; size as usize];
                for input in 0..size {
                    let out = shuffle_bounded(input, feedback, size, 0);
                    assert!(out < size, "size={size} bank={bank} left range: {out}");
                    assert!(
                        !seen[out as usize],
                        "size={size} bank={bank} repeated {out}"
                    );
                    seen[out as usize] = true;
                }
                assert!(
                    seen.iter().all(|b| *b),
                    "size={size} bank={bank}: the budget truncated a valid permutation"
                );
            }
        }
    }

    /// A feedback that is not a polynomial for the width may cycle
    /// among registers that all miss the range. The step is
    /// deterministic over finite state, so once it has repeated it
    /// never lands — the bound is where that becomes provable, and the
    /// answer is the floor. Terminating is the assertion.
    #[test]
    fn a_feedback_that_cannot_land_terminates() {
        for size in [1u64, 2, 3, 7, 64, 1000] {
            for feedback in [
                u64::MAX,
                1u64 << 63,
                (1u64 << 63) | 1,
                0xFFFF_0000_FFFF_0000,
            ] {
                for input in [0u64, 1, 42] {
                    let out = shuffle_bounded(input, feedback, size, 11);
                    assert!(
                        (11..11 + size).contains(&out),
                        "size={size} feedback={feedback:#x} input={input} gave {out}"
                    );
                }
            }
        }
    }

    /// Whatever the constants, the answer is in range and there is no
    /// panic: the fuzzer reaches this node with arbitrary ones.
    #[test]
    fn arbitrary_constants_stay_in_range() {
        for feedback in [0u64, 1, 2, 7, 0xB400, u64::MAX] {
            for size in [0u64, 1, 2, 5, 64] {
                for input in [0u64, 1, 9, u64::MAX] {
                    let out = shuffle_bounded(input, feedback, size, 3);
                    let upper = 3 + size.max(1);
                    assert!(
                        (3..upper).contains(&out),
                        "feedback={feedback} size={size} input={input} gave {out}"
                    );
                }
            }
        }
    }

    #[test]
    fn lfsr_step_leaves_zero_alone() {
        assert_eq!(lfsr_step(0, 0xB400), 0);
    }
}
