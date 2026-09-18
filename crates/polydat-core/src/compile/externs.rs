// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Extern on the compiled engines.
//!
//! An `extern` is a typed input slot with a default that a host may
//! overwrite. The interpreter keeps it as a `Value` in the state; the
//! compiled engines keep every input in the flat slot buffer, so an
//! extern needs a slot, an encoding of its value into that slot, and a
//! way for the host to change it. This module is that plumbing, shared
//! by the closure, hybrid, and native kernels:
//!
//! - **Layout.** Every input beyond the coordinates owns its slots in
//!   the buffer, after the coordinates, at the width its port type
//!   names. A `Ref2`-kind extern (a string, a byte string, JSON, an
//!   extension value, a handle) owns two slots: the pair into the
//!   value it stores.
//! - **Writing through.** Every write of an extern reaches the buffer
//!   at once: a carrier (u64, i64, f64, bool) as its bits, a `Ref2`
//!   kind (a string, a byte string, JSON, an extension value, a
//!   handle) as a `(ptr, len)` pair into the value the extern stores,
//!   which stands until the host writes the extern again (axioms S3,
//!   S4). Nothing is rewritten on any other occasion: an extern is an
//!   input, and its slot stands until it is set.
//! - **Setting.** `set` type-checks the value against the declared
//!   port type, stores it, and writes it through; the kernel that owns
//!   the buffer marks the slot's dependents dirty.
//! - **Cells.** A `shared` binding's slot is bound to a `SharedCell`
//!   (engine parity, step 9), the same cell type the interpreter
//!   attaches: the cell is the register. `set` publishes through it,
//!   every run and every pull refresh the slot from it when its
//!   revision moved, and a host attaches one kernel's cell to another
//!   through the `Kernel` trait so both read and write one register.

use std::collections::HashMap;

use crate::ast::SlotShape;
use crate::ast::{PortType, Value};
use crate::kernel::InputDef;
use crate::kernel::WriteError;

/// One extern input of a compiled kernel.
#[derive(Clone)]
pub(crate) struct ExternSlot {
    pub name: String,
    /// First buffer slot: one slot for a carrier, two for a `Ref2`
    /// kind's pair.
    pub slot: usize,
    pub ty: PortType,
    /// The current value: the declared default until the host sets it.
    /// A `Ref2` kind's pair points into this value, so it is written
    /// only through `set_slot`, which republishes the pair.
    pub value: Value,
    /// The declared default, what a kernel created from the program
    /// starts with.
    pub default: Value,
    /// The shared cell a `shared` binding's slot is bound to.
    pub cell: Option<crate::kernel::SharedCell>,
    /// The cell revision the slot last took its value from.
    pub seen: Option<u64>,
}

/// The extern inputs of one compiled kernel.
#[derive(Clone, Default)]
pub(crate) struct Externs {
    slots: Vec<ExternSlot>,
    by_name: HashMap<String, usize>,
    /// Every input by name, the coordinates first, as the interpreter
    /// program lists them.
    input_names: Vec<String>,
    /// Per input index, the extern slot it names; `None` for a
    /// coordinate. The index-keyed set (SRD 117 step 3).
    by_index: Vec<Option<usize>>,
    /// Every named output in declaration order, as the interpreter
    /// program lists them.
    output_names: Vec<String>,
    /// The cursors the program declares (engines.md §3.5):
    /// each is an `Ext` extern plus six scalar ones, and its schema
    /// carries the partitions the compiler resolved at build.
    cursors: Vec<crate::iteration::source::SourceSchema>,
    /// This kernel's intent-dirty word, shared by the cells it seeds
    /// (cross_fiber_invalidation.md §3.1).
    intent: std::sync::Arc<std::sync::atomic::AtomicU64>,
    /// The next bit of `intent` to give a cell.
    next_bit: u8,
    /// Slots whose value a cell refresh changed, for the kernel to mark
    /// dirty; drained after every refresh.
    changed: Vec<usize>,
    /// The compile ledger of the tree this kernel's program belongs
    /// to, recorded in at build: what the compiled engines keep of
    /// the program tree's identity.
    ledger: std::sync::Arc<crate::kernel::CompileLedger>,
}

impl Externs {
    /// The externs among `input_defs` (every input after the first
    /// `coord_count`), each at the slot `input_starts` gives it. An
    /// extern is a one-slot carrier or a `Ref2` kind; a two-slot
    /// immediate has no compiled form.
    pub(crate) fn new(
        input_defs: &[InputDef],
        coord_count: usize,
        input_starts: &[usize],
        cursors: &[crate::iteration::source::SourceSchema],
        shared: &[&str],
        ledger: std::sync::Arc<crate::kernel::CompileLedger>,
    ) -> Result<Self, String> {
        let mut slots = Vec::new();
        let mut by_name = HashMap::new();
        let mut by_index = vec![None; input_defs.len()];
        for (i, def) in input_defs.iter().enumerate().skip(coord_count) {
            if def.port_type.slot_color() == crate::ast::SlotColor::Imm2 {
                return Err(format!(
                    "extern '{}' has type {}, a two-slot immediate; the compiled engines carry \
                     one-slot carriers and by-reference externs (strings, byte strings, JSON, \
                     extension values, handles)",
                    def.name, def.port_type,
                ));
            }
            by_name.insert(def.name.clone(), slots.len());
            by_index[i] = Some(slots.len());
            slots.push(ExternSlot {
                name: def.name.clone(),
                slot: input_starts[i],
                ty: def.port_type,
                value: def.default.clone(),
                default: def.default.clone(),
                cell: None,
                seen: None,
            });
        }
        let mut externs = Self {
            slots,
            by_name,
            input_names: input_defs.iter().map(|d| d.name.clone()).collect(),
            by_index,
            output_names: Vec::new(),
            cursors: cursors.to_vec(),
            intent: std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0)),
            next_bit: 0,
            changed: Vec::new(),
            ledger,
        };
        for name in shared {
            if let Some(&i) = externs.by_name.get(*name) {
                let cell = externs.new_cell(externs.slots[i].value.clone());
                externs.slots[i].seen = Some(cell.snapshot().1);
                externs.slots[i].cell = Some(cell);
            }
        }
        // One compiled program of the tree, whichever engine it is on.
        externs.ledger.record();
        Ok(externs)
    }

    /// The compile ledger of the tree this kernel's program belongs to.
    pub(crate) fn ledger(&self) -> &std::sync::Arc<crate::kernel::CompileLedger> {
        &self.ledger
    }

    /// A cell of this kernel's scope holding `initial`, with the next
    /// bit of the intent word. The compiled kernel keeps one intent
    /// word, so cells past the 64th share bit 63, where the
    /// interpreter opens a new word (`allocate_cell_bit`).
    fn new_cell(&mut self, initial: Value) -> crate::kernel::SharedCell {
        let bit = self.next_bit;
        self.next_bit = self.next_bit.saturating_add(1).min(63);
        std::sync::Arc::new(crate::kernel::SharedCellInner::new(
            initial,
            self.intent.clone(),
            bit,
        ))
    }

    /// Give every `shared` slot a cell of its own holding its current
    /// value: what a kernel created from a shared program starts with,
    /// as an interpreter state seeds its own cells.
    pub(crate) fn reseed_cells(&mut self) {
        self.intent = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
        self.next_bit = 0;
        for i in 0..self.slots.len() {
            if self.slots[i].cell.is_none() {
                continue;
            }
            let cell = self.new_cell(self.slots[i].value.clone());
            self.slots[i].seen = Some(cell.snapshot().1);
            self.slots[i].cell = Some(cell);
        }
    }

    /// Bind the `shared` binding `name` to `cell`: from now on this
    /// kernel reads and writes that register, as every other holder of
    /// the cell does. Returns the slot, for the caller's dirty marking;
    /// the value arrives at the next refresh.
    pub(crate) fn attach_cell(
        &mut self,
        name: &str,
        cell: crate::kernel::SharedCell,
    ) -> Result<usize, String> {
        let Some(&i) = self.by_name.get(name) else {
            let known: Vec<&str> = self
                .slots
                .iter()
                .filter(|s| s.cell.is_some())
                .map(|s| s.name.as_str())
                .collect();
            return Err(format!(
                "no `shared` binding named '{name}'; this kernel's shared bindings are {known:?}"
            ));
        };
        let s = &mut self.slots[i];
        if s.cell.is_none() {
            return Err(format!(
                "'{name}' is an extern, not a `shared` binding; only a `shared` binding takes a cell"
            ));
        }
        s.cell = Some(cell);
        s.seen = None;
        Ok(s.slot)
    }

    /// The cells this kernel's `shared` bindings are bound to.
    pub(crate) fn shared_cells(&self) -> Vec<crate::kernel::SharedCellEntry> {
        self.slots
            .iter()
            .filter_map(|s| {
                s.cell.as_ref().map(|cell| crate::kernel::SharedCellEntry {
                    name: s.name.clone(),
                    port_type: s.ty,
                    cell: cell.clone(),
                })
            })
            .collect()
    }

    /// Whether any cell has been published to since this kernel last
    /// took its value: one Acquire load per cell.
    pub(crate) fn cells_dirty(&self) -> bool {
        self.slots.iter().any(|s| match (&s.cell, s.seen) {
            (Some(cell), seen) => {
                Some(cell.revision.load(std::sync::atomic::Ordering::Acquire)) != seen
            }
            (None, _) => false,
        })
    }

    /// Take every cell's current value where its revision moved since
    /// this kernel last read it: the value is stored and written
    /// through, and the slot is recorded for the kernel to mark dirty
    /// (`take_changed`).
    pub(crate) fn refresh_cells(&mut self, buffer: &mut [u64]) {
        for s in &mut self.slots {
            let Some(cell) = &s.cell else {
                continue;
            };
            if Some(cell.revision.load(std::sync::atomic::Ordering::Acquire)) == s.seen {
                continue;
            }
            let (value, revision) = cell.snapshot();
            s.value = value;
            s.seen = Some(revision);
            write_through(s, buffer);
            self.changed.push(s.slot);
        }
    }

    /// Whether the last refresh changed any slot.
    #[inline]
    pub(crate) fn has_changed(&self) -> bool {
        !self.changed.is_empty()
    }

    /// The slots the last refresh changed, once.
    pub(crate) fn take_changed(&mut self) -> Vec<usize> {
        std::mem::take(&mut self.changed)
    }

    /// Give back the drained list so its allocation is reused.
    pub(crate) fn return_changed(&mut self, mut list: Vec<usize>) {
        list.clear();
        self.changed = list;
    }

    /// Every input by name, the coordinates first.
    pub(crate) fn input_names(&self) -> &[String] {
        &self.input_names
    }

    /// Record the named outputs in declaration order.
    pub(crate) fn set_output_names(&mut self, names: &[String]) {
        self.output_names = names.to_vec();
    }

    /// Every named output in declaration order.
    pub(crate) fn output_names(&self) -> &[String] {
        &self.output_names
    }

    /// The slots of the externs that have no value at build: they are
    /// `None` until a host sets them, and their consumers propagate it.
    #[cfg(feature = "jit")]
    pub(crate) fn unset_slots(&self) -> Vec<usize> {
        self.slots
            .iter()
            .filter(|s| s.value == Value::None)
            .map(|s| s.slot)
            .collect()
    }

    /// The cursors the program declares, with their partitions where
    /// the compiler resolved them.
    pub(crate) fn cursor_schemas(&self) -> &[crate::iteration::source::SourceSchema] {
        &self.cursors
    }

    /// Record the extent of cursor `index`, resolved after the build from
    /// the constants the graph folded.
    pub(crate) fn set_cursor_extent(&mut self, index: usize, extent: u64) {
        if let Some(schema) = self.cursors.get_mut(index) {
            schema.extent = Some(extent);
        }
    }

    /// The writes that narrow cursor `name` to `partition`: its `Ext`
    /// slot and its six scalar projections, each an extern of this
    /// kernel. An unknown cursor is an error naming the known ones.
    pub(crate) fn cursor_writes(
        &self,
        name: &str,
        partition: &crate::iteration::cursor_partition::Partition,
    ) -> Result<Vec<(String, Value)>, WriteError> {
        if !self.cursors.iter().any(|c| c.name == name) {
            return Err(WriteError::UnknownWire {
                key: name.to_string(),
                known: self.cursors.iter().map(|c| c.name.clone()).collect(),
            });
        }
        Ok(
            crate::iteration::cursor_partition::cursor_slot_writes(name, partition)
                .into_iter()
                .filter(|(slot, _)| self.by_name.contains_key(slot))
                .collect(),
        )
    }

    /// Write every extern into the buffer, and mark the unset ones in
    /// `none` where the kernel keeps a mask; returns whether any is
    /// unset. What a build and a reset do.
    pub(crate) fn seed(&self, buffer: &mut [u64], mut none: Option<&mut [bool]>) -> bool {
        let mut any_none = false;
        for s in &self.slots {
            write_through(s, buffer);
            let unset = s.value == Value::None;
            any_none |= unset;
            if let Some(mask) = none.as_deref_mut() {
                mask[s.slot] = unset;
            }
        }
        any_none
    }

    /// Whether any extern has no value (A12): the kernel that keeps a
    /// `None` mask propagates it as the interpreter does; a native
    /// kernel, which cannot, refuses to run.
    pub(crate) fn any_unset(&self) -> bool {
        self.slots.iter().any(|s| s.value == Value::None)
    }

    /// The name and type of an unset extern, for a native kernel's
    /// refusal.
    #[cfg(feature = "jit")]
    pub(crate) fn first_unset(&self) -> Option<(&str, PortType)> {
        self.slots
            .iter()
            .find(|s| s.value == Value::None)
            .map(|s| (s.name.as_str(), s.ty))
    }

    /// Set an extern by name. The value must be of the declared port
    /// type; `Value::None` clears it to unset. The slot is written
    /// through now. Returns the extern's first slot, for the caller's
    /// dirty marking, and whether it is now unset.
    pub(crate) fn set(
        &mut self,
        name: &str,
        value: Value,
        buffer: &mut [u64],
    ) -> Result<(usize, bool), WriteError> {
        let Some(&i) = self.by_name.get(name) else {
            if self.input_names.iter().any(|n| n == name) {
                return Err(WriteError::CoordinateSlot {
                    slot: name.to_string(),
                });
            }
            return Err(WriteError::UnknownWire {
                key: name.to_string(),
                known: self.slots.iter().map(|s| s.name.clone()).collect(),
            });
        };
        self.set_slot(i, value, buffer)
    }

    /// [`Self::set`] by input index, the index among every input with
    /// the coordinates first, as `input_names` lists them.
    pub(crate) fn set_at(
        &mut self,
        index: usize,
        value: Value,
        buffer: &mut [u64],
    ) -> Result<(usize, bool), WriteError> {
        match self.by_index.get(index) {
            Some(Some(i)) => self.set_slot(*i, value, buffer),
            Some(None) => Err(WriteError::CoordinateSlot {
                slot: self.input_names[index].clone(),
            }),
            None => Err(WriteError::UnknownWire {
                key: format!("wire[{index}]"),
                known: self.input_names.clone(),
            }),
        }
    }

    /// The one write rule of every engine: the value satisfies the
    /// declared type (a carrier's bit-stuffed forms included) or is
    /// `None`, which clears the extern.
    fn set_slot(
        &mut self,
        i: usize,
        value: Value,
        buffer: &mut [u64],
    ) -> Result<(usize, bool), WriteError> {
        let s = &mut self.slots[i];
        if !value.satisfies_slot(s.ty) {
            return Err(WriteError::TypeMismatch {
                slot: s.name.clone(),
                expected: s.ty,
                got: value.port_type(),
            });
        }
        s.value = value;
        // A `shared` binding's slot writes through its cell, so every
        // holder of the cell reads this value; the revision is this
        // kernel's own and needs no refresh.
        if let Some(cell) = &s.cell {
            cell.publish(s.value.clone());
            s.seen = Some(cell.revision.load(std::sync::atomic::Ordering::Acquire));
        }
        write_through(s, buffer);
        Ok((s.slot, s.value == Value::None))
    }

    /// Start over from the program: every extern back at its declared
    /// default, written through into `buffer`, and every `shared`
    /// binding with a cell of its own holding that default. What a
    /// kernel created from a shared program starts with, whatever the
    /// kernel it was cloned from had been set to. Also what a clone
    /// needs before its first run: its pairs must point into its own
    /// stored values, not the original's.
    pub(crate) fn reset_to_program(&mut self, buffer: &mut [u64]) {
        for s in &mut self.slots {
            s.value = s.default.clone();
            s.seen = None;
            write_through(s, buffer);
        }
        self.reseed_cells();
    }

    /// The current value of the extern `name`, if there is one.
    pub(crate) fn value(&self, name: &str) -> Option<Value> {
        self.by_name.get(name).map(|&i| self.slots[i].value.clone())
    }

    /// The externs by name and declared type, for diagnostics.
    pub(crate) fn names(&self) -> Vec<(&str, PortType)> {
        self.slots.iter().map(|s| (s.name.as_str(), s.ty)).collect()
    }
}

/// Write an extern's current value into its slots: a carrier as its
/// bits, a `Ref2` kind as the pair into the value the slot stores. An
/// unset `Ref2` kind is an empty pair, which a string consumer reads
/// as empty where nothing keeps a `None` mask and a value consumer
/// reads as `None`.
fn write_through(s: &ExternSlot, buffer: &mut [u64]) {
    match s.ty.slot_color() {
        crate::ast::SlotColor::Ref2 => {
            let (p, l) = match &s.value {
                Value::None => crate::compile::marshal::empty_pair(),
                v => crate::compile::marshal::borrow_pair(v).unwrap_or_else(|| {
                    panic!(
                        "extern '{}' ({}) holds a {} value, which has no slot form",
                        s.name,
                        s.ty,
                        v.port_type()
                    )
                }),
            };
            buffer[s.slot] = p;
            buffer[s.slot + 1] = l;
        }
        _ => buffer[s.slot] = carrier_bits(&s.value),
    }
}

/// A carrier value's slot bits; an unset carrier reads as zero.
fn carrier_bits(v: &Value) -> u64 {
    match v {
        Value::U64(n) => *n,
        Value::I64(n) => *n as u64,
        Value::F64(f) => f.to_bits(),
        Value::Bool(b) => u64::from(*b),
        _ => 0,
    }
}
