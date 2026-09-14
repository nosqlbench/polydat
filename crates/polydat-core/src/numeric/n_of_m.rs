// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! The n-of-m selection: within each window of `m` inputs, the `n`
//! whose position hashes lowest are selected; the body of `n_of` and
//! of its native lowering.

/// Core n-of-m evaluation: hash the input's position within its window
/// and check whether its rank falls within the selected n.
///
/// Algorithm: within each window of m consecutive inputs, hash each
/// position (0..m) and sort by hash. The n positions with the smallest
/// hashes are selected. To avoid sorting at runtime, we count how many
/// of the m positions hash lower than the current one — if fewer than
/// n do, this position is selected.
#[inline]
pub fn n_of_m_eval(input: u64, n: u64, m: u64) -> u64 {
    let window = input / m;
    let pos = input % m;
    // Hash this position within the window using fast register mix
    let my_hash = crate::numeric::hash::splitmix64_u64(
        window.wrapping_mul(0x517cc1b727220a95) ^ pos.wrapping_mul(0x9e3779b97f4a7c15),
    );
    // Count how many positions in the same window hash lower
    let mut rank: u64 = 0;
    for i in 0..m {
        if i == pos {
            continue;
        }
        let other_hash = crate::numeric::hash::splitmix64_u64(
            window.wrapping_mul(0x517cc1b727220a95) ^ i.wrapping_mul(0x9e3779b97f4a7c15),
        );
        if other_hash < my_hash || (other_hash == my_hash && i < pos) {
            rank += 1;
        }
    }
    // Selected if rank < n (i.e., among the n smallest hashes)
    if rank < n { 1 } else { 0 }
}
