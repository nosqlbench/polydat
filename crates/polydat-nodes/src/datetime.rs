// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Datetime and epoch function nodes.

/// Scale a u64 to epoch milliseconds by multiplying by a factor.
///
/// Signature: `(input: u64) -> (u64)`
/// Param: `factor: u64` — milliseconds per input unit.
///
/// Example: `EpochScale(1000)` treats input as seconds → millis.
#[polydat::polydat_node(category = Datetime)]
fn epoch_scale(
    input: u64,
    #[poly_default(1u64)] factor: polydat::derive_support::Const<u64>,
) -> u64 {
    input.wrapping_mul(*factor)
}

impl EpochScale {
    /// Scale by 1: milliseconds stay milliseconds.
    pub fn millis() -> Self {
        Self::new(1)
    }
    /// Scale by 1,000: seconds to milliseconds.
    pub fn seconds() -> Self {
        Self::new(1_000)
    }
    /// Scale by 60,000: minutes to milliseconds.
    pub fn minutes() -> Self {
        Self::new(60_000)
    }
    /// Scale by 3,600,000: hours to milliseconds.
    pub fn hours() -> Self {
        Self::new(3_600_000)
    }
}

/// Add a base epoch offset to a u64 value.
#[polydat::polydat_node(category = Datetime)]
fn epoch_offset(
    input: u64,
    #[poly_default(0u64)] base_epoch_ms: polydat::derive_support::Const<u64>,
) -> u64 {
    input.wrapping_add(*base_epoch_ms)
}

impl EpochOffset {
    /// 2024-01-01T00:00:00Z in epoch millis.
    pub fn from_2024() -> Self {
        Self::new(1_704_067_200_000)
    }
    /// 2025-01-01T00:00:00Z in epoch millis.
    pub fn from_2025() -> Self {
        Self::new(1_735_689_600_000)
    }
}

/// Format an epoch-millis u64 as an ISO-8601-like timestamp string.
///
/// Signature: `(input: u64) -> (String)`
///
/// Produces: `"YYYY-MM-DDThh:mm:ss.mmmZ"`
/// Uses a simple arithmetic calendar (no timezone, no leap second handling).
#[polydat::polydat_node(category = Datetime)]
fn to_timestamp(input: u64) -> String {
    epoch_ms_to_iso(input)
}

/// Decompose epoch millis into date/time components.
#[polydat::polydat_node(
    category = Datetime,
    output_names(year, month, day, hour, minute, second, millis),
)]
fn date_components(input: u64) -> (u64, u64, u64, u64, u64, u64, u64) {
    decompose_epoch_ms(input)
}

// --- Calendar arithmetic (simplified, no leap seconds) ---

const MILLIS_PER_SEC: u64 = 1_000;
#[allow(dead_code)]
const MILLIS_PER_MIN: u64 = 60_000;
#[allow(dead_code)]
const MILLIS_PER_HOUR: u64 = 3_600_000;
#[allow(dead_code)]
const MILLIS_PER_DAY: u64 = 86_400_000;

fn decompose_epoch_ms(epoch_ms: u64) -> (u64, u64, u64, u64, u64, u64, u64) {
    let mut remaining = epoch_ms;
    let ms = remaining % MILLIS_PER_SEC;
    remaining /= MILLIS_PER_SEC;
    let sec = remaining % 60;
    remaining /= 60;
    let min = remaining % 60;
    remaining /= 60;
    let hour = remaining % 24;
    let (year, month, day) = civil_from_days(remaining / 24);
    (year, month, day, hour, min, sec, ms)
}

/// Days since 1970-01-01 to a proleptic Gregorian `(year, month, day)`,
/// in constant time.
///
/// The calendar used to be walked a year at a time, which is exact but
/// linear in the year: at `u64::MAX` milliseconds that is some 584
/// million iterations, one evaluation taking tens of seconds. This is
/// the closed form (H. Hinnant, "chrono-Compatible Low-Level Date
/// Algorithms"): count 400-year eras of 146 097 days from 0000-03-01,
/// so the leap day falls at the end of each year of the era, then read
/// the year, day of year, and month out of the era arithmetically. The
/// input never precedes 1970, so every step stays in `u64`, and the
/// largest input (`u64::MAX / 86_400_000` days) is far from overflow.
fn civil_from_days(days: u64) -> (u64, u64, u64) {
    let z = days + 719_468; // days since 0000-03-01
    let era = z / 146_097;
    let doe = z - era * 146_097; // [0, 146096]
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365; // [0, 399]
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365], from March 1
    let mp = (5 * doy + 2) / 153; // [0, 11], March = 0
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + u64::from(month <= 2);
    (year, month, day)
}

fn epoch_ms_to_iso(epoch_ms: u64) -> String {
    let (y, mo, d, h, mi, s, ms) = decompose_epoch_ms(epoch_ms);
    format!("{y:04}-{mo:02}-{d:02}T{h:02}:{mi:02}:{s:02}.{ms:03}Z")
}

#[cfg(test)]
mod tests {
    use super::*;
    use polydat::ast::{PolydatNode, Value};

    /// Every day from 1970 through the year 10 000 against the calendar
    /// counted forward a day at a time: slow as a node, plainly right,
    /// and cheap here because the count is carried along rather than
    /// restarted. The range crosses the 100- and 400-year rules (2000
    /// leaps, 2100 does not) and some twenty whole 400-year eras, the
    /// period the closed form repeats with.
    #[test]
    fn the_closed_form_calendar_agrees_with_the_walk() {
        let mut days = 0u64;
        let (mut y, mut m, mut d) = (1970u64, 1u64, 1u64);
        let leap =
            |y: u64| (y.is_multiple_of(4) && !y.is_multiple_of(100)) || y.is_multiple_of(400);
        while y <= 10_000 {
            assert_eq!(civil_from_days(days), (y, m, d), "day {days}");
            let len = [
                31,
                if leap(y) { 29 } else { 28 },
                31,
                30,
                31,
                30,
                31,
                31,
                30,
                31,
                30,
                31,
            ][m as usize - 1];
            (y, m, d) = if d < len {
                (y, m, d + 1)
            } else if m < 12 {
                (y, m + 1, 1)
            } else {
                (y + 1, 1, 1)
            };
            days += 1;
        }
    }

    /// The top of the range answers at once and in range: the input
    /// that used to walk 584 million years. The bound is loose; the walk
    /// took tens of seconds and the closed form takes nanoseconds.
    #[test]
    fn the_largest_epoch_decomposes_in_constant_time() {
        let t = std::time::Instant::now();
        let (y, mo, d, h, mi, s, ms) = decompose_epoch_ms(u64::MAX);
        assert!(t.elapsed() < std::time::Duration::from_secs(1));
        assert!(y > 584_000_000 && (1..=12).contains(&mo) && (1..=31).contains(&d));
        assert!(h < 24 && mi < 60 && s < 60 && ms < 1000);
    }

    #[test]
    fn epoch_scale_seconds() {
        let node = EpochScale::seconds();
        let mut out = [Value::None];
        node.eval(&[Value::U64(5)], &mut out);
        assert_eq!(out[0].as_u64(), 5000);
    }

    #[test]
    fn epoch_offset_basic() {
        let node = EpochOffset::new(1_000_000);
        let mut out = [Value::None];
        node.eval(&[Value::U64(500)], &mut out);
        assert_eq!(out[0].as_u64(), 1_000_500);
    }

    #[test]
    fn to_timestamp_epoch_zero() {
        let node = ToTimestamp::new();
        let mut out = [Value::None];
        node.eval(&[Value::U64(0)], &mut out);
        assert_eq!(out[0].as_str(), "1970-01-01T00:00:00.000Z");
    }

    #[test]
    fn to_timestamp_known_date() {
        let node = ToTimestamp::new();
        let mut out = [Value::None];
        // 2024-01-01T00:00:00.000Z = 1704067200000
        node.eval(&[Value::U64(1_704_067_200_000)], &mut out);
        assert_eq!(out[0].as_str(), "2024-01-01T00:00:00.000Z");
    }

    #[test]
    fn date_components_epoch_zero() {
        let node = DateComponents::new();
        let mut out = vec![Value::None; 7];
        node.eval(&[Value::U64(0)], &mut out);
        assert_eq!(out[0].as_u64(), 1970);
        assert_eq!(out[1].as_u64(), 1);
        assert_eq!(out[2].as_u64(), 1);
        assert_eq!(out[3].as_u64(), 0);
        assert_eq!(out[4].as_u64(), 0);
        assert_eq!(out[5].as_u64(), 0);
        assert_eq!(out[6].as_u64(), 0);
    }

    #[test]
    fn date_components_known() {
        let node = DateComponents::new();
        let mut out = vec![Value::None; 7];
        // 2024-03-15T14:30:45.123Z
        // Manually: days from epoch to 2024-03-15 = 19797
        // 19797 * 86400000 + 14*3600000 + 30*60000 + 45*1000 + 123
        let epoch = 19797u64 * MILLIS_PER_DAY
            + 14 * MILLIS_PER_HOUR
            + 30 * MILLIS_PER_MIN
            + 45 * MILLIS_PER_SEC
            + 123;
        node.eval(&[Value::U64(epoch)], &mut out);
        assert_eq!(out[0].as_u64(), 2024);
        assert_eq!(out[1].as_u64(), 3);
        assert_eq!(out[2].as_u64(), 15);
        assert_eq!(out[3].as_u64(), 14);
        assert_eq!(out[4].as_u64(), 30);
        assert_eq!(out[5].as_u64(), 45);
        assert_eq!(out[6].as_u64(), 123);
    }

    #[test]
    fn epoch_scale_compiled() {
        let node = EpochScale::seconds();
        let op = node.compiled_u64().unwrap();
        let mut out = [0u64];
        op(&[5], &mut out);
        assert_eq!(out[0], 5000);
    }
}
