// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! The budget and combination order every superfuzz shares.
//!
//! A superfuzz names its combinatoric dimensions (each a count of the
//! values it can take) and a check to run on one combination. [`run`]
//! walks the full product of the dimensions in a stratified order and
//! stops at whichever comes first of `SUPERFUZZ_MAX_COMBINATIONS`
//! combinations (default 1000) and `SUPERFUZZ_MAX_SECONDS` seconds of
//! wall time (default 60). A value of 0, or no value, takes the
//! default; a large number such as `18446744073709551615` lifts the
//! limit. The time is checked between combinations, so a run ends
//! within one combination of the limit.
//!
//! The order is a bijection of the product in which every dimension
//! changes on every step. The dimensions are ranked by size, largest
//! first, and combination `i` is written in mixed radix with the
//! largest dimension as the fastest digit `e0`. Dimension `m` then
//! takes the value `(e_m + e_0 + … + e_(m-1)) mod n_m`. The first
//! `n_0` combinations therefore take every value of every dimension,
//! depths included, and each later round of `n_0` shifts the smaller
//! dimensions against one another, so a budget cut never leaves a
//! dimension or a depth unvisited once the budget reaches the size of
//! the largest dimension.
//!
//! At the end the run prints the combinations it covered of the total,
//! the values of each dimension it reached, why it stopped, and the
//! violations it found, and it fails on any violation.

use std::time::{Duration, Instant};

/// The default count of combinations a superfuzz covers.
pub const DEFAULT_MAX_COMBINATIONS: u64 = 1000;
/// The default wall time of a superfuzz, in seconds.
pub const DEFAULT_MAX_SECONDS: u64 = 60;

/// A numeric environment variable, or `default` when it is unset,
/// unparsable, or 0.
pub fn env_nonzero(name: &str, default: u64) -> u64 {
    match std::env::var(name)
        .ok()
        .and_then(|s| s.trim().parse::<u64>().ok())
    {
        Some(0) | None => default,
        Some(v) => v,
    }
}

/// Seeds run per combination: `SUPERFUZZ_SEEDS`, default 1.
pub fn seeds_per_combination() -> u64 {
    env_nonzero("SUPERFUZZ_SEEDS", 1)
}

/// The seed of the `k`th run of combination `index`, decorrelated
/// from `base` by a splitmix64 finalizer so neighbouring combinations
/// draw unrelated tails.
pub fn seed_for(base: u64, index: u64, k: u64) -> u64 {
    let mut z = base
        .wrapping_add(index.wrapping_mul(0x9E37_79B9_7F4A_7C15))
        .wrapping_add(k.wrapping_mul(0xD1B5_4A32_D192_ED03));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// The count and time limits of one run.
#[derive(Clone, Copy, Debug)]
pub struct Budget {
    pub max_combinations: u64,
    pub max_time: Duration,
}

impl Budget {
    /// `SUPERFUZZ_MAX_COMBINATIONS` and `SUPERFUZZ_MAX_SECONDS`, each
    /// defaulted when unset or 0.
    pub fn from_env() -> Self {
        Budget {
            max_combinations: env_nonzero("SUPERFUZZ_MAX_COMBINATIONS", DEFAULT_MAX_COMBINATIONS),
            max_time: Duration::from_secs(env_nonzero(
                "SUPERFUZZ_MAX_SECONDS",
                DEFAULT_MAX_SECONDS,
            )),
        }
    }
}

/// The stratified order over the product of dimension sizes.
pub struct Stratified {
    sizes: Vec<usize>,
    /// Dimension indices, largest size first.
    rank: Vec<usize>,
    total: u64,
}

impl Stratified {
    pub fn new(sizes: &[usize]) -> Self {
        assert!(
            sizes.iter().all(|&n| n > 0),
            "a superfuzz dimension has no values: {sizes:?}"
        );
        let mut rank: Vec<usize> = (0..sizes.len()).collect();
        rank.sort_by_key(|&d| std::cmp::Reverse(sizes[d]));
        let total = sizes
            .iter()
            .try_fold(1u64, |acc, &n| acc.checked_mul(n as u64))
            .unwrap_or(u64::MAX);
        Stratified {
            sizes: sizes.to_vec(),
            rank,
            total,
        }
    }

    /// The size of the full product, saturated at `u64::MAX`.
    pub fn total(&self) -> u64 {
        self.total
    }

    /// The value of each dimension, in declaration order, of the
    /// `index`th combination.
    pub fn digits(&self, index: u64) -> Vec<usize> {
        let mut out = vec![0; self.sizes.len()];
        let mut rest = index;
        let mut shift = 0u64;
        for &d in &self.rank {
            let n = self.sizes[d] as u64;
            let e = rest % n;
            rest /= n;
            out[d] = ((e + shift) % n) as usize;
            shift += e;
        }
        out
    }
}

/// Why a run stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stop {
    Exhausted,
    Count,
    Time,
}

/// Walk the product of `dims` (name, size) in the stratified order,
/// calling `check(values, index)` on each combination until the
/// budget runs out, then print the coverage and fail on any violation.
pub fn run(
    name: &str,
    dims: &[(&str, usize)],
    mut check: impl FnMut(&[usize], u64) -> Vec<String>,
) {
    let budget = Budget::from_env();
    let sizes: Vec<usize> = dims.iter().map(|d| d.1).collect();
    let order = Stratified::new(&sizes);
    let total = order.total();
    let mut reached: Vec<Vec<bool>> = sizes.iter().map(|&n| vec![false; n]).collect();
    let mut violations: Vec<String> = Vec::new();
    let start = Instant::now();
    let mut covered = 0u64;
    let stop = loop {
        if covered >= total {
            break Stop::Exhausted;
        }
        if covered >= budget.max_combinations {
            break Stop::Count;
        }
        if start.elapsed() >= budget.max_time {
            break Stop::Time;
        }
        let values = order.digits(covered);
        for (d, &v) in values.iter().enumerate() {
            reached[d][v] = true;
        }
        let found = check(&values, covered);
        if !found.is_empty() {
            let named: Vec<String> = dims
                .iter()
                .zip(&values)
                .map(|((n, _), v)| format!("{n}={v}"))
                .collect();
            let tag = format!("[combination {covered}: {}]", named.join(" "));
            violations.extend(found.into_iter().map(|f| format!("{tag} {f}")));
        }
        covered += 1;
        if covered.is_multiple_of(250) {
            eprintln!(
                "{name}: {covered} combinations in {:.1}s, {} violation(s) so far",
                start.elapsed().as_secs_f64(),
                violations.len()
            );
        }
    };
    let elapsed = start.elapsed().as_secs_f64();
    let why = match stop {
        Stop::Exhausted => "every combination was covered".to_string(),
        Stop::Count => format!(
            "the combination limit of {} was reached (SUPERFUZZ_MAX_COMBINATIONS)",
            budget.max_combinations
        ),
        Stop::Time => format!(
            "the time limit of {}s was reached (SUPERFUZZ_MAX_SECONDS)",
            budget.max_time.as_secs()
        ),
    };
    let per_dim: Vec<String> = dims
        .iter()
        .zip(&reached)
        .map(|((n, size), r)| format!("{n} {}/{size}", r.iter().filter(|b| **b).count()))
        .collect();
    eprintln!(
        "{name}: covered {covered} of {total} combinations in {elapsed:.1}s; stopped because \
         {why}.\n{name}: dimension values reached: {}\n{name}: {} violation(s)",
        per_dim.join(", "),
        violations.len()
    );
    let mut report = violations.join("\n---\n");
    const MAX_REPORT: usize = 30_000;
    if report.len() > MAX_REPORT {
        let mut cut = MAX_REPORT;
        while !report.is_char_boundary(cut) {
            cut -= 1;
        }
        report.truncate(cut);
        report.push_str("\n… (report truncated)");
    }
    assert!(
        violations.is_empty(),
        "{name}: invariants violated ({} failures over {covered} of {total} \
         combinations):\n\n{report}",
        violations.len()
    );
}

#[test]
fn stratified_order_is_a_bijection_that_visits_every_value_first() {
    let sizes = [3usize, 5, 2, 4];
    let order = Stratified::new(&sizes);
    assert_eq!(order.total(), 120);
    let mut seen = std::collections::HashSet::new();
    for i in 0..order.total() {
        assert!(seen.insert(order.digits(i)), "combination {i} repeats");
    }
    // The first round, as long as the largest dimension, reaches every
    // value of every dimension.
    for (d, &n) in sizes.iter().enumerate() {
        let hit: std::collections::HashSet<usize> = (0..5).map(|i| order.digits(i)[d]).collect();
        assert_eq!(hit.len(), n, "dimension {d} not covered in the first round");
    }
}
