// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! The Polydat node library: every function a program can call that
//! the compiler does not synthesize itself. Each node is a
//! `#[polydat_node]` function over the core's value model and
//! registers into the registry at link time, so linking this crate is
//! what makes its functions callable; a third-party node crate is
//! built the same way. The numeric bodies the native lowerings share
//! live in `polydat_core::numeric`, and the nodes here that have a
//! lowering are thin wrappers over them.
//!
//! `polydat` re-exports every module here under `polydat::library`,
//! beside the nodes the core keeps, at the paths they always had.

#![cfg_attr(test, allow(clippy::approx_constant))]

// The node macro emits `polydat::…` paths; here they resolve to the
// core, as they do inside the core itself.
extern crate polydat_core as polydat;

pub mod arithmetic;
pub mod bitwise;
pub mod bytebuf;
pub mod compare;
pub mod datetime;
pub mod digest;
pub mod emit;
pub mod encoding;
pub mod hash;
pub mod lerp;
pub mod math;
pub mod noise;
pub mod param_helpers;
pub mod partition;
pub mod pcg;
pub mod pick;
pub mod probability;
pub mod random;
pub mod realer;
pub mod regex;
pub mod register;
pub mod round_numbers;
pub mod sampling;
pub mod stability;
pub mod streamer;
pub mod string;
pub mod vector_math;
pub mod vector_set;
pub mod weighted;

#[cfg(test)]
mod core_tests;
