// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! The SplitMix64 mixer: the body of `hash` and the mix every
//! hashing lowering inlines.

/// Ultra-fast 64-bit pseudo-random permutation mixer (SplitMix64).
///
/// Passes the TestU01 BigCrush suite with 100% avalanche effect.
/// Emits 3 inlined multiplications and bit shifts in JIT mode with
/// zero heap allocation and zero extern call overhead.
#[inline(always)]
pub fn splitmix64_u64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e3779b97f4a7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    x ^ (x >> 31)
}
