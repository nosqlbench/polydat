// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Float text, byte-identical to Rust's formatting and faster.
//!
//! A tile encodes an `f64` hole as `format!("{f:?}")` when the hole has
//! no format and as `format!("{f:.N}")` under a `.N` precision (SRD 114
//! §7.2). Those bytes are the tile's contract on every engine, so a
//! faster writer is only admissible if it produces the same bytes for
//! every value. This module is that writer, and
//! `tests/float_text.rs` is the proof: a differential over edge values,
//! arithmetic series, and a million seeded bit patterns, for the
//! shortest form and every precision 0 through 9.
//!
//! **Shortest form** ([`write_shortest`]). Rust's `Debug` for `f64`
//! prints the shortest round-trip digits, as a decimal with at least
//! one fractional digit (`100.0`, `0.1`, `-0.0`) when the magnitude is
//! in `[1e-4, 1e16)` and otherwise in exponent form (`1e16`,
//! `1.5e-7`). The `ryu` crate produces the same shortest digits and,
//! for every class but one, the same layout; the exception is
//! `[1e-5, 1e-4)`, which ryu lays out as `0.00005` and Rust as `5e-5`.
//! The writer takes ryu's text and re-lays out that one class from
//! ryu's digits. The digits themselves agree except on a tie: a value
//! whose exact decimal expansion is one digit longer than its shortest
//! form and ends in 5 (`2231889947293916.25`, whose shortest forms
//! `…916.2` and `…916.3` both round-trip), where ryu rounds to the
//! even digit and Rust rounds up. The writer detects a tie exactly from
//! the float's odd mantissa and exponent and falls back to `format!`
//! for it, so the proof's random sweep is what establishes that no
//! other class differs.
//!
//! **Fixed precision** ([`write_fixed`]). `format!("{f:.N}")` rounds
//! the exact binary value to `N` fractional digits, half to even on the
//! exact decimal expansion. Rounding the shortest digits is not the
//! same operation (0.295 is below the tie in binary, so `.2` gives
//! `0.29`, where rounding the text `0.295` half-even gives `0.30`).
//! The writer decodes the float to `m * 2^e` and computes
//! `round(m * 10^N * 2^e)` in `u128` arithmetic: for `e < 0` the
//! quotient and remainder of a shift, compared against the half; for
//! `e >= 0` a left shift with no rounding at all. That is exact
//! wherever `m * 10^N * 2^max(e,0)` fits in 128 bits, which covers
//! `N <= 22` and magnitudes below about `2^(75 - 3.33 N)` (`1.6e29` at
//! `N = 9`); every other case falls back to `format!`, so the output is
//! Rust's own where the fast path does not reach. A magnitude below the
//! fast path's shift range is exactly zero at any supported precision
//! and is written as such without a fallback.

use std::fmt;

/// Powers of ten that fit `u128` alongside a 53-bit mantissa.
const POW10: [u128; 23] = {
    let mut t = [1u128; 23];
    let mut i = 1;
    while i < 23 {
        t[i] = t[i - 1] * 10;
        i += 1;
    }
    t
};

/// The largest precision the exact fast path serves.
const MAX_FAST_PRECISION: usize = POW10.len() - 1;

/// Write `f` exactly as `format!("{f:?}")` does.
pub fn write_shortest<W: fmt::Write>(f: f64, out: &mut W) -> fmt::Result {
    if f.is_nan() {
        return out.write_str("NaN");
    }
    if f.is_infinite() {
        return out.write_str(if f.is_sign_negative() { "-inf" } else { "inf" });
    }
    let mut buf = ryu::Buffer::new();
    let text = buf.format_finite(f);
    if shortest_is_tie(f, text) {
        return write!(out, "{f:?}");
    }
    // ryu lays out every class as Rust does except `[1e-5, 1e-4)`,
    // where ryu writes `0.0000d...` and Rust writes `d.dddde-5`.
    let (sign, body) = match text.strip_prefix('-') {
        Some(rest) => ("-", rest),
        None => ("", text),
    };
    let Some(digits) = body.strip_prefix("0.0000") else {
        return out.write_str(text);
    };
    out.write_str(sign)?;
    out.write_str(&digits[..1])?;
    if digits.len() > 1 {
        out.write_char('.')?;
        out.write_str(&digits[1..])?;
    }
    out.write_str("e-5")
}

/// Whether the shortest digits of finite, non-zero `f` are a tie: the
/// exact binary value lies exactly halfway between two shortest
/// candidates, where ryu rounds to the even digit and Rust's `Debug`
/// rounds up (`flt2dec::strategy::dragon::format_shortest`). `text` is
/// ryu's rendering of `f`.
///
/// The exact value is `m' * 2^e'` with `m'` odd. When `e' >= 0` its
/// decimal expansion is an integer that the shortest form reproduces
/// or ends in an even digit, so there is no tie. When `e' < 0` the
/// expansion is `X = m' * 5^j` over `10^j`, an odd multiple of five,
/// so it ends in 5; it is a tie exactly when the shortest form has one
/// digit fewer than `X`. `X` above `10^18` has more digits than any
/// shortest form plus one, so only a small `X` needs the count.
#[inline]
fn shortest_is_tie(f: f64, text: &str) -> bool {
    let bits = f.to_bits();
    let exp_bits = ((bits >> 52) & 0x7ff) as i32;
    let frac = bits & ((1u64 << 52) - 1);
    let (m, e) = if exp_bits == 0 {
        (frac, -1074)
    } else {
        (frac | (1u64 << 52), exp_bits - 1075)
    };
    let tz = m.trailing_zeros() as i32;
    let e = e + tz;
    if e >= 0 {
        return false;
    }
    let j = (-e) as u32;
    // `5^j` for `j > 26` is above `10^18` on its own.
    if j > 26 {
        return false;
    }
    let x = (m >> tz) as u128 * 5u128.pow(j);
    if x >= POW10[18] {
        return false;
    }
    let n = significant_digits(text);
    POW10[n] <= x && x < POW10[n + 1]
}

/// The count of significant digits in ryu's text: its digits before
/// any exponent, without leading zeros and without the trailing zeros
/// the fixed layouts add (`12340000000.0`, `100.0`).
#[inline]
fn significant_digits(text: &str) -> usize {
    let bytes = text.as_bytes();
    let end = bytes.iter().position(|&b| b == b'e').unwrap_or(bytes.len());
    let mantissa = &bytes[..end];
    let Some(first) = mantissa.iter().position(|b| (b'1'..=b'9').contains(b)) else {
        return 0;
    };
    let last = mantissa
        .iter()
        .rposition(|b| (b'1'..=b'9').contains(b))
        .expect("a first nonzero digit implies a last");
    mantissa[first..=last]
        .iter()
        .filter(|b| b.is_ascii_digit())
        .count()
}

/// Whether [`write_shortest`] takes the ryu path for `f`, or falls
/// back to `format!` on a tie. Exposed so the proof can report its
/// fallback rate.
pub fn shortest_is_fast(f: f64) -> bool {
    if !f.is_finite() || f == 0.0 {
        return true;
    }
    let mut buf = ryu::Buffer::new();
    !shortest_is_tie(f, buf.format_finite(f))
}

/// Write `f` exactly as `format!("{f:.prec$}")` does.
pub fn write_fixed<W: fmt::Write>(f: f64, prec: usize, out: &mut W) -> fmt::Result {
    if f.is_nan() {
        return out.write_str("NaN");
    }
    if f.is_infinite() {
        return out.write_str(if f.is_sign_negative() { "-inf" } else { "inf" });
    }
    let Some(q) = fixed_scaled(f, prec) else {
        return write!(out, "{f:.prec$}");
    };
    let mut buf = [0u8; 48];
    let len = layout_fixed(f.is_sign_negative(), q, prec, &mut buf);
    out.write_str(std::str::from_utf8(&buf[len..]).expect("ascii"))
}

/// `round_half_even(|f| * 10^prec)` as an integer, or `None` where the
/// value does not fit the exact `u128` path and `format!` must serve.
#[inline]
fn fixed_scaled(f: f64, prec: usize) -> Option<u128> {
    if prec > MAX_FAST_PRECISION {
        return None;
    }
    let bits = f.to_bits();
    let exp_bits = ((bits >> 52) & 0x7ff) as i32;
    let frac = bits & ((1u64 << 52) - 1);
    let (m, e) = if exp_bits == 0 {
        (frac, -1074)
    } else {
        (frac | (1u64 << 52), exp_bits - 1075)
    };
    if m == 0 {
        return Some(0);
    }
    // `m < 2^53` and `10^22 < 2^74`, so `t < 2^127`.
    let t = m as u128 * POW10[prec];
    if e >= 0 {
        // An integer-valued float: `t << e` has no fraction to round.
        if e as u32 > t.leading_zeros() {
            return None;
        }
        return Some(t << e);
    }
    let s = (-e) as u32;
    if s >= 128 {
        // `t < 2^127 <= 2^(s-1)`, so the scaled value is below one
        // half and rounds to zero; it cannot be a tie.
        return Some(0);
    }
    let q = t >> s;
    let rem = t & ((1u128 << s) - 1);
    let half = 1u128 << (s - 1);
    Some(if rem > half || (rem == half && q & 1 == 1) {
        q + 1
    } else {
        q
    })
}

/// Lay `q / 10^prec` out as sign, integer digits, and `prec`
/// fractional digits, right-aligned in `buf`; returns the start index.
#[inline]
fn layout_fixed(negative: bool, q: u128, prec: usize, buf: &mut [u8; 48]) -> usize {
    // The same digit loop in `u64` where the scaled value fits (every
    // value a tile is likely to hold) and in `u128` otherwise; a
    // `u128` division is several times a `u64` one.
    macro_rules! put_scaled {
        ($q:expr, $scale:expr) => {{
            let mut i = buf.len();
            let mut q = $q;
            let mut frac = q % $scale;
            q /= $scale;
            for _ in 0..prec {
                i -= 1;
                buf[i] = b'0' + (frac % 10) as u8;
                frac /= 10;
            }
            if prec > 0 {
                i -= 1;
                buf[i] = b'.';
            }
            loop {
                i -= 1;
                buf[i] = b'0' + (q % 10) as u8;
                q /= 10;
                if q == 0 {
                    break;
                }
            }
            i
        }};
    }
    let mut i = if q <= u64::MAX as u128 {
        put_scaled!(q as u64, POW10[prec] as u64)
    } else {
        put_scaled!(q, POW10[prec])
    };
    if negative {
        i -= 1;
        buf[i] = b'-';
    }
    i
}

/// [`write_shortest`] into a new `String`.
pub fn shortest_string(f: f64) -> String {
    let mut s = String::with_capacity(24);
    let _ = write_shortest(f, &mut s);
    s
}

/// [`write_fixed`] into a new `String`.
pub fn fixed_string(f: f64, prec: usize) -> String {
    let mut s = String::with_capacity(24 + prec);
    let _ = write_fixed(f, prec, &mut s);
    s
}

/// Whether [`write_fixed`] takes the exact fast path for `f` at
/// `prec`, or falls back to `format!`. Exposed so the proof can report
/// its fallback rate.
pub fn fixed_is_fast(f: f64, prec: usize) -> bool {
    f.is_finite() && fixed_scaled(f, prec).is_some()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shortest_matches_debug_on_named_values() {
        for f in [
            0.0,
            -0.0,
            1.0,
            100.0,
            0.1,
            1e-5,
            5e-5,
            9.99e-5,
            1e-4,
            1e15,
            1e16,
            1.5e300,
            5e-324,
            f64::MAX,
            f64::MIN,
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
            -0.000012345,
        ] {
            assert_eq!(
                shortest_string(f),
                format!("{f:?}"),
                "{:#018x}",
                f.to_bits()
            );
        }
    }

    #[test]
    fn fixed_matches_format_on_named_values() {
        for f in [
            0.0,
            -0.0,
            0.5,
            1.5,
            2.5,
            0.295,
            4.35,
            2.675,
            1.005,
            1e21,
            1e29,
            1e30,
            1e300,
            5e-324,
            -1e-30,
            // 123456789.123456789, as the nearest double reads.
            123_456_789.123_456_79,
        ] {
            for prec in 0..=9 {
                assert_eq!(
                    fixed_string(f, prec),
                    format!("{f:.prec$}"),
                    "{:#018x} .{prec}",
                    f.to_bits()
                );
            }
        }
        assert!(!fixed_is_fast(1e300, 2));
        assert!(fixed_is_fast(1e21, 9));
        assert!(fixed_is_fast(5e-324, 9));
    }
}
