// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Standard Polydat function-node library + sampling primitives + library-internal support.
//!
//! - Function-node modules (`arithmetic`, `string`, `hash`, …):
//!   the 250+ built-in [`crate::ast::PolydatNode`] implementations
//!   workload authors compose into kernels.
//! - [`sampling`]: alias tables, LUT interpolation, ICD —
//!   variate-sampling building blocks consumed by
//!   distribution-emitting nodes (weighted, probability, etc.).
//! - [`support`]: library-internal infrastructure
//!   ([`support::cache`], [`support::audit`]) used by nodes
//!   like `vectors` for caching dataset handles and
//!   diagnosing data-source mismatches.

pub mod sampling;
pub mod support;

pub mod arithmetic;
pub mod assertions;
pub mod bitwise;
pub mod bytebuf;
pub mod compare;
pub mod context;
pub mod convert;
pub mod datafile;
pub mod datetime;
pub mod diagnostic;
pub mod digest;
pub mod emit;
pub mod encoding;
pub mod exactly_one;
pub mod fixed;
pub mod format;
pub mod hash;
pub mod identity;
pub mod json;
pub mod lerp;
pub mod log_levels;
pub mod math;
pub mod noise;
pub mod param_helpers;
pub mod partition;
pub mod pcg;
pub mod pick;
pub mod polyfill;
pub mod polyfill_128;
pub mod polyfill_complete;
pub mod polyfill_narrow;
pub mod probability;
pub mod random;
pub mod realer;
pub mod regex;
pub mod register;
pub mod round_numbers;
pub mod stability;
pub mod streamer;
pub mod string;
pub mod tile_render;
pub mod vector_math;
#[cfg(feature = "vectordata")]
pub mod vectors;
pub mod weighted;

/// Env-gated debug-level diagnostic for selection / matching nodes.
///
/// Returns true when `NBRS_DEBUG_NODES` is set to a non-empty,
/// non-`"0"` value. Cached on first read so the variable can be set
/// once at process start and the per-cycle check is a load.
///
/// Used by `regex_match`, `exactly_one_value`, and `pick` to emit
/// pre-eval / pre-panic context (input shape, match result, selector
/// states) when probe phases produce surprising values. The user's
/// expected workflow:
///
/// ```sh
/// NBRS_DEBUG_NODES=1 nbrs run my-workload …
/// ```
///
/// then read the stderr trace to see what each matching node saw.
pub fn debug_nodes_enabled() -> bool {
    use std::sync::OnceLock;
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("NBRS_DEBUG_NODES")
            .map(|v| !v.is_empty() && v != "0")
            .unwrap_or(false)
    })
}
