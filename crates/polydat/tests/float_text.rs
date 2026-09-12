// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! The proof for `library::support::float_text`: the tile float writer
//! is byte-identical to `format!("{f:?}")` (an unformatted hole) and to
//! `format!("{f:.N}")` for every precision 0 through 9 (a `.N` hole),
//! over named edge values, three arithmetic series, and a million
//! seeded random bit patterns. A tile's bytes are its contract on
//! every engine, so a mismatch here is a defect in the writer, not a
//! tolerance; every mismatch is reported with its bit pattern.

use polydat::library::support::float_text::{
    fixed_is_fast, fixed_string, shortest_is_fast, shortest_string, write_fixed, write_shortest,
};

const PRECISIONS: std::ops::RangeInclusive<usize> = 0..=9;

/// The named edge values of the task statement, plus the class
/// boundaries the writer's layout depends on.
fn edge_values() -> Vec<f64> {
    let mut v = vec![
        0.0,
        -0.0,
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::MIN_POSITIVE,
        f64::MAX,
        f64::MIN,
        f64::EPSILON,
        0.1,
        0.2,
        0.3,
        0.125,
        0.295,
        0.5,
        1.5,
        2.5,
        100.0,
        1e21,
        1e22,
        // 123456789.123456789, as the nearest double reads.
        123_456_789.123_456_79,
        4.35,
        2.675,
        1.005,
        // Layout class boundaries of `Debug`: 1e-4 and 1e16, and ryu's
        // own at 1e-5, each with its neighbours.
        1e-5,
        5e-5,
        9.9e-5,
        1e-4,
        1e-3,
        1e15,
        9999999999999998.0,
        1e16,
        // The fast path's magnitude limit, around 2^75 at .0 and 2^45
        // at .9, and the shift range limit near 2^-128.
        1e29,
        1e30,
        3e22,
        4e13,
        2.0f64.powi(74),
        2.0f64.powi(75),
        2.0f64.powi(-127),
        2.0f64.powi(-128),
        2.0f64.powi(-129),
    ];
    // Subnormals: the smallest, the largest, and a few in between.
    v.extend(
        [
            1u64,
            2,
            3,
            0x000f_ffff_ffff_ffff,
            0x0008_0000_0000_0000,
            0x0000_0000_1234_5678,
        ]
        .into_iter()
        .map(f64::from_bits),
    );
    // Neighbours of every value above, on both sides.
    let base: Vec<f64> = v.clone();
    for f in base {
        if f.is_finite() {
            v.push(f64::from_bits(f.to_bits().wrapping_add(1)));
            v.push(f64::from_bits(f.to_bits().wrapping_sub(1)));
        }
    }
    // Powers of ten from 1e-30 to 1e30, and their negatives.
    for e in -30..=30 {
        v.push(10f64.powi(e));
        v.push(-(10f64.powi(e)));
    }
    v
}

/// The three arithmetic series of the task statement.
fn series_values() -> impl Iterator<Item = f64> {
    (1..=100_000u32).flat_map(|n| [n as f64 / 7.0, n as f64 / 3.0, n as f64 / 100.0])
}

/// A seeded xorshift64 over all bit patterns, NaN included.
struct XorShift64(u64);

impl Iterator for XorShift64 {
    type Item = u64;
    fn next(&mut self) -> Option<u64> {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        Some(x)
    }
}

/// One differential run over `values`: every mismatch, for the
/// shortest form and every precision, with the bit pattern that
/// produced it, plus the counts of checks each fast path served.
#[derive(Default)]
struct Report {
    checked: u64,
    ties: u64,
    fast: u64,
    slow: u64,
    mismatches: Vec<String>,
}

impl Report {
    fn check(&mut self, f: f64, mine: &mut String, theirs: &mut String) {
        use std::fmt::Write as _;
        self.checked += 1;
        mine.clear();
        theirs.clear();
        let _ = write_shortest(f, mine);
        let _ = write!(theirs, "{f:?}");
        if mine != theirs {
            self.record(f, "shortest", mine, theirs);
        }
        if !shortest_is_fast(f) {
            self.ties += 1;
        }
        for prec in PRECISIONS {
            mine.clear();
            theirs.clear();
            let _ = write_fixed(f, prec, mine);
            let _ = write!(theirs, "{f:.prec$}");
            if mine != theirs {
                self.record(f, &format!(".{prec}"), mine, theirs);
            }
            if fixed_is_fast(f, prec) {
                self.fast += 1;
            } else {
                self.slow += 1;
            }
        }
    }

    fn record(&mut self, f: f64, form: &str, mine: &str, theirs: &str) {
        if self.mismatches.len() < 64 {
            self.mismatches.push(format!(
                "bits {:#018x} ({f:e}) form {form}: writer {mine:?}, format! {theirs:?}",
                f.to_bits()
            ));
        } else if self.mismatches.len() == 64 {
            self.mismatches.push("... further mismatches elided".into());
        }
    }

    fn assert_clean(&self, what: &str) {
        assert!(
            self.mismatches.is_empty(),
            "{what}: {} mismatches over {} values ({} shortest ties; fixed: {} fast, {} \
             fallback):\n{}",
            self.mismatches.len(),
            self.checked,
            self.ties,
            self.fast,
            self.slow,
            self.mismatches.join("\n")
        );
        // The counts are worth seeing on a passing run too
        // (`--nocapture`); they say how much of the sweep the fast
        // paths served.
        println!(
            "{what}: {} values, {} shortest ties, fixed checks {} fast / {} fallback",
            self.checked, self.ties, self.fast, self.slow
        );
    }
}

fn run<I: IntoIterator<Item = f64>>(values: I) -> Report {
    let mut report = Report::default();
    let (mut mine, mut theirs) = (String::new(), String::new());
    for f in values {
        report.check(f, &mut mine, &mut theirs);
    }
    report
}

#[test]
fn edge_values_match_format() {
    run(edge_values()).assert_clean("edge values");
}

#[test]
fn series_match_format() {
    run(series_values()).assert_clean("series n/7, n/3, n/100");
}

/// A quarter million random bit patterns from one seed: the harness
/// runs the four quarters on separate threads, which keeps the debug
/// build's wall clock down without shrinking the million.
fn random_quarter(seed: u64) {
    let report = run(XorShift64(seed).take(250_000).map(f64::from_bits));
    report.assert_clean(&format!("random bit patterns, seed {seed:#x}"));
    // The random sweep must exercise the fast path, not only the
    // fallback: uniformly random exponents put most values far outside
    // the exact range, so this bounds the check from below.
    assert!(
        report.fast > 10_000,
        "fast path covered only {} of {} fixed checks",
        report.fast,
        report.fast + report.slow
    );
}

#[test]
fn random_bit_patterns_quarter_1() {
    random_quarter(0x9E37_79B9_7F4A_7C15);
}

#[test]
fn random_bit_patterns_quarter_2() {
    random_quarter(0xBF58_476D_1CE4_E5B9);
}

#[test]
fn random_bit_patterns_quarter_3() {
    random_quarter(0x94D0_49BB_1331_11EB);
}

#[test]
fn random_bit_patterns_quarter_4() {
    random_quarter(0x2545_F491_4F6C_DD1D);
}

/// Random doubles in the magnitude band a tile is likely to hold,
/// where the fixed fast path serves every value: mantissas at random
/// with exponents from 2^-40 to 2^40. Half a million from each seed.
fn in_band_half(seed: u64) {
    let report = run(XorShift64(seed).take(500_000).map(|bits| {
        let exp = 1023 + (bits >> 52) % 81 - 40;
        f64::from_bits((bits & 0x800f_ffff_ffff_ffff) | (exp << 52))
    }));
    report.assert_clean(&format!("random in-band values, seed {seed:#x}"));
    assert_eq!(report.slow, 0, "the band must sit inside the fast path");
}

#[test]
fn random_in_band_values_half_1() {
    in_band_half(0xD1B5_4A32_D192_ED03);
}

#[test]
fn random_in_band_values_half_2() {
    in_band_half(0x6A09_E667_F3BC_C909);
}

/// The tie class on its own: odd multiples of 2^-2 and 2^-3 near 2^50
/// and 2^49, whose shortest forms are ties, alongside their
/// neighbours, which are not.
#[test]
fn shortest_ties_match_format() {
    let mut values = Vec::new();
    for base in [
        2.0f64.powi(50),
        2.0f64.powi(49),
        2.0f64.powi(46),
        2.0f64.powi(43),
    ] {
        for step in 1..2000u32 {
            let f = base + step as f64 * 1.25;
            values.push(f);
            values.push(-f);
            values.push(f64::from_bits(f.to_bits() + 1));
        }
    }
    let report = run(values);
    report.assert_clean("shortest ties");
    assert!(report.ties > 0, "the tie class must be present");
}

#[test]
fn string_forms_agree_with_writers() {
    for f in [0.295, 1e-5, 100.0, f64::NAN] {
        assert_eq!(shortest_string(f), format!("{f:?}"));
        assert_eq!(fixed_string(f, 2), format!("{f:.2}"));
    }
}

/// The bigger sweep: sixteen million random bit patterns.
#[test]
#[ignore]
fn sixteen_million_random_bit_patterns_match_format() {
    run(XorShift64(0x3C6E_F372_FE94_F82B)
        .take(16_000_000)
        .map(f64::from_bits))
    .assert_clean("random bit patterns (large sweep)");
}
