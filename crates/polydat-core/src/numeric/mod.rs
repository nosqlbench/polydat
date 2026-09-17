// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! The numeric bodies the node library and the native lowerings
//! share. A node in `polydat-nodes` is a thin wrapper over a function
//! here, and the JIT helper for that node calls the same function, so
//! the interpreter and native code compute the same bits by
//! construction (compiled_handles.md §6, the corner suite).

pub mod hash;
pub mod n_of_m;
pub mod noise;
pub mod pcg;
pub mod register;
pub mod round_numbers;
pub mod special;
pub mod vector;
