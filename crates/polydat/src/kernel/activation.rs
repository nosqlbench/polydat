// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! The activation runtime for `for` traversals (SRD 113 §3.4, §3.6,
//! §5).
//!
//! A [`TraversalStream`] dispenses one [`Activation`] per tuple of a
//! compiled traversal. An activation is a fresh state over the body's
//! program, compiled once at parent compile time, with the tuple's
//! elements and the parent's cascaded wires bound and every `over`
//! cursor narrowed. Activation never compiles: the cost of the second
//! activation is the cost of the first minus nothing, because the first
//! did not compile either.
//!
//! Cycles follow the rule in §3.4: a body with a cursor iterates its
//! narrowest cursor extent, one cycle per ordinal, with the cursor's
//! ordinal written before each pull; a body without a cursor has one
//! cycle per activation.

use std::collections::HashMap;
use std::sync::Arc;

use crate::ast::Value;
use crate::dsl::traversal::Traversal;
use crate::iteration::comprehension::runtime::{RuntimeTuple, evaluate_for_iteration};
use crate::iteration::cursor_partition::{cursor_extent_on, cursor_over_partitions_on};
use crate::kernel::Kernel;

use super::{PolydatKernel, PolydatProgram};

/// The interval of ordinals an activation's cursor iterates.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CursorSlice {
    /// The cursor's name.
    pub cursor: String,
    /// The first ordinal of the slice.
    pub start: u64,
    /// One past the last ordinal.
    pub end: u64,
}

impl CursorSlice {
    /// Ordinals in the slice.
    pub fn len(&self) -> u64 {
        self.end.saturating_sub(self.start)
    }

    /// Whether the slice has no ordinal.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// One child scope of a traversal: the tuple it was activated for, a
/// fresh kernel over the body's program, and its cursor slice if the
/// body declares a cursor. The kernel is the interpreter's by default;
/// [`TraversalStream::activation_on`] builds one on any engine behind
/// the [`Kernel`] trait (engine parity, step 8).
pub struct Activation<K = PolydatKernel> {
    /// Position of this activation's tuple in the traversal's dispense
    /// order.
    pub index: u64,
    /// The tuple, in element order.
    pub coords: Vec<(String, Value)>,
    /// A fresh kernel over the body's shared program.
    pub kernel: K,
    /// The narrowest cursor slice, when the body declares a cursor.
    pub cursor: Option<CursorSlice>,
}

impl std::fmt::Debug for Activation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Activation")
            .field("index", &self.index)
            .field("coords", &self.coords)
            .field("cursor", &self.cursor)
            .field("program_nodes", &self.kernel.program().node_count())
            .finish()
    }
}

impl std::fmt::Debug for Activation<Box<dyn Kernel>> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Activation")
            .field("index", &self.index)
            .field("coords", &self.coords)
            .field("cursor", &self.cursor)
            .field("engine", &self.kernel.engine())
            .finish()
    }
}

impl<K> Activation<K> {
    /// Cycles this activation runs under §3.4: the cursor slice length,
    /// or one when the body has no cursor.
    pub fn cycle_count(&self) -> u64 {
        match &self.cursor {
            Some(slice) => slice.len(),
            None => 1,
        }
    }

    /// The value of one coordinate.
    pub fn coord(&self, name: &str) -> Option<&Value> {
        self.coords.iter().find(|(n, _)| n == name).map(|(_, v)| v)
    }
}

impl Activation<Box<dyn Kernel>> {
    /// Position the kernel at cycle `i` and return it ready to pull, as
    /// the interpreter activation's `cycle` does.
    pub fn cycle(&mut self, i: u64) -> &mut dyn Kernel {
        self.kernel.set_inputs(&[i]);
        if let Some(slice) = &self.cursor {
            let ordinal = slice.start.saturating_add(i);
            let slot = format!("{}__ordinal", slice.cursor);
            // A body without the projection has no slot to write.
            let _ = self.kernel.set_input(&slot, Value::U64(ordinal));
        }
        self.kernel.as_mut()
    }

    /// Run `f` once per cycle, in order.
    pub fn for_each_cycle(&mut self, mut f: impl FnMut(u64, &mut dyn Kernel)) {
        for i in 0..self.cycle_count() {
            let kernel = self.cycle(i);
            f(i, kernel);
        }
    }
}

impl Activation {
    /// Position the kernel at cycle `i` and return it ready to pull.
    /// The body's `cycle` coordinate is the local index; the cursor's
    /// ordinal slot receives the absolute ordinal.
    pub fn cycle(&mut self, i: u64) -> &mut PolydatKernel {
        self.kernel.set_inputs(&[i]);
        if let Some(slice) = &self.cursor {
            let ordinal = slice.start.saturating_add(i);
            let slot = format!("{}__ordinal", slice.cursor);
            if let Some(idx) = self.kernel.program().find_input(&slot) {
                self.kernel.state().set_input(idx, Value::U64(ordinal));
            }
        }
        &mut self.kernel
    }

    /// Run `f` once per cycle, in order.
    pub fn for_each_cycle(&mut self, mut f: impl FnMut(u64, &mut PolydatKernel)) {
        for i in 0..self.cycle_count() {
            let kernel = self.cycle(i);
            f(i, kernel);
        }
    }
}

/// Dispenses activations for one traversal of a kernel.
pub struct TraversalStream {
    traversal: Traversal,
    tuples: Vec<RuntimeTuple>,
    cascade: Vec<(String, Value)>,
    next: usize,
}

impl TraversalStream {
    /// Number of activations the traversal dispenses.
    pub fn len(&self) -> usize {
        self.tuples.len()
    }

    /// Whether the traversal dispenses no activation.
    pub fn is_empty(&self) -> bool {
        self.tuples.is_empty()
    }

    /// The traversal this stream dispenses.
    pub fn traversal(&self) -> &Traversal {
        &self.traversal
    }

    /// Move the dispense position. Every strategy is a decidable
    /// permutation, so seeking costs nothing beyond the index.
    pub fn seek(&mut self, index: usize) {
        self.next = index.min(self.tuples.len());
    }

    /// Current dispense position.
    pub fn position(&self) -> usize {
        self.next
    }

    /// The next activation, or `None` when exhausted.
    pub fn advance(&mut self) -> Result<Option<Activation>, String> {
        if self.next >= self.tuples.len() {
            return Ok(None);
        }
        let i = self.next;
        self.next += 1;
        self.activation(i).map(Some)
    }

    /// Build the activation at `index` without moving the dispense
    /// position. Fibers partition a traversal by calling this over
    /// disjoint index ranges.
    pub fn activation(&self, index: usize) -> Result<Activation, String> {
        let tuple = self.tuples.get(index).ok_or_else(|| {
            format!(
                "activation index {index} is out of range; traversal has {} tuples",
                self.tuples.len()
            )
        })?;
        let program = self.traversal.program.clone();
        // An activation runs inside the root's cycle (SRD 115, H5).
        let mut kernel = PolydatKernel::from_program_nested(program);
        bind_by_name(&mut kernel, tuple);
        bind_by_name(&mut kernel, &self.cascade);
        let cursor = narrow_cursors(&mut kernel)?;
        Ok(Activation {
            index: index as u64,
            coords: tuple.clone(),
            kernel,
            cursor,
        })
    }

    /// [`Self::activation`] on `engine` (engine parity, step 8): a fresh
    /// kernel over the body's program for that engine, compiled once
    /// per engine and shared by every activation after, driven through
    /// the [`Kernel`] trait with the same elements, cascade, and cursor
    /// narrowing. An engine that cannot run the body says so by name.
    pub fn activation_on(
        &self,
        index: usize,
        engine: crate::Engine,
    ) -> Result<Activation<Box<dyn Kernel>>, String> {
        let tuple = self.tuples.get(index).ok_or_else(|| {
            format!(
                "activation index {index} is out of range; traversal has {} tuples",
                self.tuples.len()
            )
        })?;
        let program = self
            .traversal
            .program_on(engine)
            .map_err(|e| e.to_string())?;
        // An activation runs inside the root's cycle (SRD 115, H5).
        let mut kernel = program.create_nested_kernel();
        bind_by_name_on(kernel.as_mut(), tuple)?;
        bind_by_name_on(kernel.as_mut(), &self.cascade)?;
        let cursor = narrow_cursors_on(kernel.as_mut())?;
        Ok(Activation {
            index: index as u64,
            coords: tuple.clone(),
            kernel,
            cursor,
        })
    }
}

/// Bind the inputs the body declares among `values`, through the trait.
fn bind_by_name_on(kernel: &mut dyn Kernel, values: &[(String, Value)]) -> Result<(), String> {
    let declared: std::collections::HashSet<String> = kernel.input_names().into_iter().collect();
    for (name, value) in values {
        if declared.contains(name) {
            kernel.set_input(name, value.clone())?;
        }
    }
    Ok(())
}

fn bind_by_name(kernel: &mut PolydatKernel, values: &[(String, Value)]) {
    for (name, value) in values {
        if let Some(idx) = kernel.program().find_input(name) {
            kernel.state().set_input(idx, value.clone());
        }
    }
}

/// Resolve every `over` clause in the body and narrow its cursor. Returns
/// the narrowest slice, or the full extent of the first cursor when none
/// has an `over` clause. One routine for every engine, through the
/// trait.
fn narrow_cursors(kernel: &mut PolydatKernel) -> Result<Option<CursorSlice>, String> {
    narrow_cursors_on(kernel)
}

fn narrow_cursors_on(kernel: &mut dyn Kernel) -> Result<Option<CursorSlice>, String> {
    let schemas: Vec<crate::iteration::source::SourceSchema> = kernel.cursor_schemas().to_vec();
    let mut narrowest: Option<CursorSlice> = None;
    for schema in &schemas {
        let slice = if schema.partition_output.is_some() {
            let parts = cursor_over_partitions_on(kernel, schema)?;
            let partition = match parts.len() {
                1 => parts[0],
                0 => {
                    return Err(format!(
                        "cursor '{}': its `over` value resolved to no partitions",
                        schema.name
                    ));
                }
                n => {
                    return Err(format!(
                        "cursor '{}': its `over` value resolved to {n} partitions; inside a traversal, bind the list \
                     with an enclosing `for p in ...` and declare the cursor `over p`",
                        schema.name
                    ));
                }
            };
            kernel.set_cursor(&schema.name, &partition)?;
            CursorSlice {
                cursor: schema.name.clone(),
                start: partition.start_ord,
                end: partition.end_ord,
            }
        } else {
            let extent = cursor_extent_on(kernel, schema);
            CursorSlice {
                cursor: schema.name.clone(),
                start: 0,
                end: extent,
            }
        };
        narrowest = Some(match narrowest {
            Some(prev) if prev.len() <= slice.len() => prev,
            _ => slice,
        });
    }
    Ok(narrowest)
}

impl PolydatKernel {
    /// A fresh kernel over a shared, already compiled program: the host
    /// side of one program, many states.
    pub fn over(program: Arc<PolydatProgram>) -> Self {
        PolydatKernel::from_program(program)
    }

    /// Open the traversal at `index` among this program's top-level
    /// `for` statements, evaluated against this kernel's current values.
    ///
    /// Comprehension sources that reference this kernel's wires see the
    /// values currently set on it. Cascade externs are snapshotted from
    /// this kernel now and bound into every activation.
    pub fn traverse(&mut self, index: usize) -> Result<TraversalStream, String> {
        let program = self.program().clone();
        let traversal = program.traversals().get(index).cloned().ok_or_else(|| {
            format!(
                "no traversal at index {index}; the program declares {}",
                program.traversals().len()
            )
        })?;

        // Snapshot the cascade: pull each outer wire the body imports.
        let mut cascade = Vec::with_capacity(traversal.cascade.len());
        for (name, _) in &traversal.cascade {
            let value = if program.output_index(name).is_some() {
                self.pull(name).clone()
            } else if let Some(idx) = program.find_input(name) {
                self.state().get_input(idx)
            } else {
                Value::None
            };
            cascade.push((name.clone(), value));
        }

        // Evaluate the comprehension with this kernel's values in scope.
        // The parent is a snapshot of this kernel; the canonical kernel is
        // the body's program, whose element externs let filter predicates
        // and dependent sources see the tuple being built.
        let mut snapshot = PolydatKernel::from_program_nested(program.clone());
        self.propagate_inputs_into(&mut snapshot);
        let parent = Arc::new(snapshot);
        let mut canonical_kernel = PolydatKernel::from_program_nested(traversal.program.clone());
        bind_by_name(&mut canonical_kernel, &cascade);
        let canonical = Arc::new(canonical_kernel);
        let params: HashMap<String, String> = HashMap::new();
        let tuples = evaluate_for_iteration(
            &traversal.comprehension,
            &parent,
            &canonical,
            &params,
            |_| Ok(()),
        )
        .map_err(|e| {
            format!(
                "`for {}` at line {}, col {}: {e}",
                traversal.source_text, traversal.span.line, traversal.span.col
            )
        })?;

        Ok(TraversalStream {
            traversal,
            tuples,
            cascade,
            next: 0,
        })
    }

    /// Open every top-level traversal, in document order.
    pub fn traverse_all(&mut self) -> Result<Vec<TraversalStream>, String> {
        let n = self.program().traversals().len();
        (0..n).map(|i| self.traverse(i)).collect()
    }
}

/// Identity of the program an activation runs over, for callers that
/// want to assert the one-program-per-position property.
pub fn program_identity(kernel: &PolydatKernel) -> *const PolydatProgram {
    Arc::as_ptr(kernel.program())
}
