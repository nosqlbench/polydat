// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Hash function nodes.
//!
//! Every node here is JIT-eligible: each arg and return is a scalar
//! that rides one slot, so the macro emits `compiled_u64()` and
//! `jit_constants()` (carrying the captured `Const<...>` field
//! values) alongside `eval()`.

use polydat::compile::fusion::{DecomposedGraph, DecomposedWire, FusedNode};
use xxhash_rust::xxh3::xxh3_64;

pub use polydat::numeric::hash::splitmix64_u64;

/// 64-bit hash using SplitMix64 (high-speed scalar integer mixer).
///
/// Signature: `hash(input: u64) -> (u64)`
///
/// The fundamental entropy source for deterministic data generation.
/// Place at the head of nearly every pipeline to scatter sequential
/// cycle counters into uniformly distributed u64 values. Fully inlined
/// into native Cranelift ALU instructions in Phase 3 JIT.
///
/// JIT level: P3 (Fully inlined Cranelift native IR).
#[polydat::polydat_node(category = Hashing)]
fn hash(input: u64) -> u64 {
    splitmix64_u64(input)
}

/// Explicit SplitMix64 integer permutation node.
///
/// Signature: `splitmix64(input: u64) -> (u64)`
#[polydat::polydat_node(category = Hashing)]
fn splitmix64(input: u64) -> u64 {
    splitmix64_u64(input)
}

/// Scatter sequential cycle counters across the 64-bit integer space.
///
/// Signature: `scatter(input: u64) -> (u64)`
#[polydat::polydat_node(category = Hashing)]
fn scatter(input: u64) -> u64 {
    splitmix64_u64(input)
}

/// Canonical 64-bit xxHash3 digest.
///
/// Signature: `xxhash3(input: u64) -> (u64)`
///
/// Use when exact compatibility with the xxHash3 algorithm is required.
#[polydat::polydat_node(category = Hashing)]
fn xxhash3(input: u64) -> u64 {
    xxh3_64(&input.to_le_bytes())
}

/// Canonical 64-bit xxHash3 digest (short alias).
///
/// Signature: `xxh3(input: u64) -> (u64)`
#[polydat::polydat_node(category = Hashing)]
fn xxh3(input: u64) -> u64 {
    xxh3_64(&input.to_le_bytes())
}

/// Hash a u64 into a bounded range `[0, max)`.
///
/// Signature: `hash_range(input: u64, max: u64) -> (u64)`
///
/// Combines hashing and modular reduction in a single node. Use when
/// you need a bounded integer directly, for example selecting a row
/// index: `hash_range(cycle, 1_000_000)` gives a uniformly distributed
/// key in [0, 1M).
///
/// JIT level: P3 (Fully inlined Cranelift native IR).
#[polydat::polydat_node(category = Hashing)]
fn hash_range(input: u64, max: Const<u64>) -> u64 {
    if *max == 0 {
        0
    } else {
        splitmix64_u64(input) % *max
    }
}

impl FusedNode for HashRange {
    /// `hash_range(x, K)` decomposes to `mod(hash(x), K)`.
    fn decomposed(&self) -> DecomposedGraph {
        use crate::arithmetic::Mod;
        let mut g = DecomposedGraph::new(1);
        let h = g.add_node(Box::new(Hash::new()), vec![DecomposedWire::Input(0)]);
        let m = g.add_node(
            Box::new(Mod::new(self.max)),
            vec![DecomposedWire::Node(h, 0)],
        );
        g.set_outputs(vec![DecomposedWire::Node(m, 0)]);
        g
    }
}

/// Hash a u64 into a float interval `[min, max)`.
///
/// Signature: `hash_interval(input: u64, min: f64, max: f64) -> (f64)`
///
/// Convenience node that hashes, normalizes to [0,1), and scales in one
/// step. Useful when a uniform f64 in a specific range is needed without
/// wiring separate `hash` + `unit_interval` + `lerp` nodes.
///
/// JIT level: P3 (Fully inlined Cranelift native IR).
#[polydat::polydat_node(category = Hashing)]
fn hash_interval(input: u64, min: Const<f64>, max: Const<f64>) -> f64 {
    let h = splitmix64_u64(input);
    let unit = (h as f64) / (u64::MAX as f64);
    *min + unit * (*max - *min)
}

impl FusedNode for HashInterval {
    /// `hash_interval(x, lo, hi)` decomposes to `lerp(unit_interval(hash(x)), lo, hi)`.
    fn decomposed(&self) -> DecomposedGraph {
        use crate::lerp::Lerp;
        use crate::sampling::icd::UnitInterval;
        let mut g = DecomposedGraph::new(1);
        let h = g.add_node(Box::new(Hash::new()), vec![DecomposedWire::Input(0)]);
        let ui = g.add_node(
            Box::new(UnitInterval::new()),
            vec![DecomposedWire::Node(h, 0)],
        );
        let lerp = g.add_node(
            Box::new(Lerp::new(self.min, self.max)),
            vec![DecomposedWire::Node(ui, 0)],
        );
        g.set_outputs(vec![DecomposedWire::Node(lerp, 0)]);
        g
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use polydat::ast::{PolydatNode, Value};

    #[test]
    fn hash_deterministic() {
        let node = Hash::new();
        let mut out = [Value::None];
        node.eval(&[Value::U64(42)], &mut out);
        let first = out[0].as_u64();
        node.eval(&[Value::U64(42)], &mut out);
        assert_eq!(
            first,
            out[0].as_u64(),
            "same input must produce same output"
        );
    }

    #[test]
    fn hash_different_inputs_differ() {
        let node = Hash::new();
        let mut out1 = [Value::None];
        let mut out2 = [Value::None];
        node.eval(&[Value::U64(0)], &mut out1);
        node.eval(&[Value::U64(1)], &mut out2);
        assert_ne!(out1[0].as_u64(), out2[0].as_u64());
    }

    #[test]
    fn hash_range_bounded() {
        let node = HashRange::new(100);
        let mut out = [Value::None];
        for i in 0..1000 {
            node.eval(&[Value::U64(i)], &mut out);
            assert!(out[0].as_u64() < 100);
        }
    }

    #[test]
    fn hash_interval_bounded() {
        let node = HashInterval::new(10.0, 20.0);
        let mut out = [Value::None];
        for i in 0..1000 {
            node.eval(&[Value::U64(i)], &mut out);
            let v = out[0].as_f64();
            assert!((10.0..20.0).contains(&v), "got {v}");
        }
    }

    #[test]
    fn splitmix64_and_xxhash3_distinguishable() {
        let sm = Splitmix64::new();
        let xh = Xxhash3::new();
        let mut out_sm = [Value::None];
        let mut out_xh = [Value::None];
        sm.eval(&[Value::U64(12345)], &mut out_sm);
        xh.eval(&[Value::U64(12345)], &mut out_xh);
        assert_ne!(out_sm[0].as_u64(), 0);
        assert_ne!(out_xh[0].as_u64(), 0);
        assert_ne!(out_sm[0].as_u64(), out_xh[0].as_u64());
    }
}

// ── Fusion rules ───────────────────────────────────────────────
//
// The compiler knows no node by name: the rules that fuse a hash
// with its consumers are registered here, beside the nodes they
// build (compile::fusion::FusionRuleRegistration).

use polydat::compile::fusion::{FusionPattern, FusionRule, FusionRuleRegistration};

/// `mod(hash(x), K)` → `hash_range(x, K)`: hashing and bounded
/// reduction in one node, with no buffer slot for the hash between.
fn hash_mod_to_hash_range() -> FusionRule {
    FusionRule {
        name: "hash_mod_to_hash_range",
        pattern: FusionPattern::node(
            "mod",
            vec![FusionPattern::node(
                "hash",
                vec![FusionPattern::any("x")],
                "hash_node",
            )],
            "mod_node",
        ),
        replacement: |m| {
            let max = m.const_u64("mod_node");
            Box::new(HashRange::new(max))
        },
        input_bindings: &["x"],
    }
}

/// `lerp(unit_interval(hash(x)), lo, hi)` → `hash_interval(x, lo, hi)`:
/// one hash and one scaled float in one step.
fn hash_unit_lerp_to_hash_interval() -> FusionRule {
    FusionRule {
        name: "hash_unit_lerp_to_hash_interval",
        pattern: FusionPattern::node(
            "lerp",
            vec![FusionPattern::node(
                "unit_interval",
                vec![FusionPattern::node(
                    "hash",
                    vec![FusionPattern::any("x")],
                    "hash_node",
                )],
                "ui_node",
            )],
            "lerp_node",
        ),
        replacement: |m| {
            let consts = m.const_vec("lerp_node");
            let lo = f64::from_bits(consts[0]);
            let hi = f64::from_bits(consts[1]);
            Box::new(HashInterval::new(lo, hi))
        },
        input_bindings: &["x"],
    }
}

polydat::inventory::submit! {
    FusionRuleRegistration { priority: 10, build: hash_mod_to_hash_range }
}
polydat::inventory::submit! {
    FusionRuleRegistration { priority: 20, build: hash_unit_lerp_to_hash_interval }
}
