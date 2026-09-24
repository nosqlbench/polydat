// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Binding a child kernel under a parent, on any engine
//! (native_scope_trees.md §5).
//!
//! A scope tree is built by binding each child under its parent: the
//! parent's cells attached, its values copied in, the child's scope-init
//! constants materialized. The wiring is over `dyn Kernel` on both sides,
//! so a child of any engine binds under a parent of any engine, and a
//! parent shared across threads may have children bound under it from
//! several at once (`Kernel: Sync`).

use std::sync::Arc;

use crate::ast::Value;
use crate::kernel::{Kernel, KernelProgram, PolydatKernel, WriteError};

/// A kernel of `program` bound under `parent`: `iter_bindings` written
/// first, so the child's own scope coordinates see them, then the
/// parent's cells attached and its values copied in.
///
/// The kernel's [`Kernel::program_id`] is `program`'s, whichever engine
/// `program` is on, so a host that sealed a plan of indices against the
/// program can use it on every kernel bound from it.
pub fn bind_under(
    parent: &dyn Kernel,
    program: Arc<dyn KernelProgram>,
    iter_bindings: &[(String, Value)],
) -> Result<Box<dyn Kernel>, WriteError> {
    let mut child = program.create_kernel();
    for (var, value) in iter_bindings {
        child.set_input(var, value.clone())?;
    }
    PolydatKernel::wire_child_under(child.as_mut(), parent);
    Ok(child)
}

/// Copy `parent`'s input values into `child`'s inputs of the same name:
/// what carries a cascade of externs down a scope tree past what the
/// child imports as outputs.
///
/// Coordinates are not copied; a host positions a child with
/// `set_inputs`. An input with no value on the parent is skipped. A
/// value the child's declared input refuses is an error naming both,
/// not a skip (input_variance.md §7).
pub fn propagate_inputs(parent: &dyn Kernel, child: &mut dyn Kernel) -> Result<(), WriteError> {
    let child_coords = child.coord_count();
    for (index, name) in parent.input_names().iter().enumerate() {
        let Some(value) = parent.input_value_at(index) else {
            continue;
        };
        if matches!(value, Value::None) {
            continue;
        }
        let Some(child_index) = child.input_index(name) else {
            continue;
        };
        // A coordinate is positioned with `set_inputs`; a cell-bound slot
        // reads its cell, and writing it would publish into a register
        // the whole scope shares.
        if child_index < child_coords || child.input_is_cell_bound(child_index) {
            continue;
        }
        child.set_input_at(child_index, value)?;
    }
    Ok(())
}
