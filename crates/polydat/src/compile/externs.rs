// Copyright 2024-2026 Jonathan Shook
// SPDX-Licene-Ientifier: Apache-2.0

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
//!   names. A table-kind extern (JSON, an extension value, a handle)
//!   also owns one entry of the kernel's value table, numbered after
//!   the entries the nodes own.
//! - **Seeding.** A carrier (u64, i64, f64, bool) is written into its
//!   slot once, when the kernel is built or the host sets it.
//! - **Materialization.** A handle kind is written at the start of
//!   every run: a string or byte string into the cycle arena, a table
//!   kind into its entry, so the slot holds a handle of the current
//!   generation (SRD 115 axiom H3) that names its own entry (H4). The
//!   value itself lives here between runs.
//! - **Setting.** `set` type-checks the value against the declared
//!   port type, stores it, and writes a carrier through immediately;
//!   the kernel that owns the buffer marks the slot's dependents dirty.

use std::collections::HashMap;

use crate::ast::{PortType, Value};
use crate::kernel::{InputDef, ValueTable};

/// One extern input of a compiled kernel.
#[derive(Clone)]
pub(crate) struct ExternSlot {
    pub name: String,
    /// First buffer slot; every supported extern is one slot wide.
    pub slot: usize,
    pub ty: PortType,
    /// The value-table entry a table-kind extern owns.
    pub entry: Option<usize>,
    /// The current value: the declared default until the host sets it.
    pub value: Value,
}

/// The extern inputs of one compiled kernel.
#[derive(Clone, Default)]
pub(crate) struct Externs {
    slots: Vec<ExternSlot>,
    by_name: HashMap<String, usize>,
    /// Every input by name, the coordinates first, as the interpreter
    /// program lists them.
    input_names: Vec<String>,
    /// The cursors the program declares (engine_parity.md, step 3):
    /// each is an `Ext` extern plus six scalar ones, and its schema
    /// carries the partitions the compiler resolved at build.
    cursors: Vec<crate::iteration::source::SourceSchema>,
}

impl Externs {
    /// The externs among `input_defs` (every input after the first
    /// `coord_count`), each at the slot `input_starts` gives it. Table
    /// kinds take entries from `entry_base` on, in input order. An
    /// extern wider than one slot has no compiled form.
    pub(crate) fn new(
        input_defs: &[InputDef],
        coord_count: usize,
        input_starts: &[usize],
        entry_base: usize,
        cursors: &[crate::iteration::source::SourceSchema],
    ) -> Result<Self, String> {
        let mut slots = Vec::new();
        let mut by_name = HashMap::new();
        let mut next_entry = entry_base;
        for (i, def) in input_defs.iter().enumerate().skip(coord_count) {
            if def.port_type.slot_width() != 1 {
                return Err(format!(
                    "extern '{}' has type {}, which spans {} slots; the compiled engines carry \
                     one-slot externs (carriers, strings, JSON, extension values)",
                    def.name,
                    def.port_type,
                    def.port_type.slot_width()
                ));
            }
            let entry = if def.port_type.handle_kind() == Some(crate::ast::HandleKind::Table) {
                let e = next_entry;
                next_entry += 1;
                Some(e)
            } else {
                None
            };
            by_name.insert(def.name.clone(), slots.len());
            slots.push(ExternSlot {
                name: def.name.clone(),
                slot: input_starts[i],
                ty: def.port_type,
                entry,
                value: def.default.clone(),
            });
        }
        Ok(Self {
            slots,
            by_name,
            input_names: input_defs.iter().map(|d| d.name.clone()).collect(),
            cursors: cursors.to_vec(),
        })
    }

    /// Every input by name, the coordinates first.
    pub(crate) fn input_names(&self) -> &[String] {
        &self.input_names
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

    /// The writes that narrow cursor `name` to `partition`: its `Ext`
    /// slot and its six scalar projections, each an extern of this
    /// kernel. An unknown cursor is an error naming the known ones.
    pub(crate) fn cursor_writes(
        &self,
        name: &str,
        partition: &crate::iteration::cursor_partition::Partition,
    ) -> Result<Vec<(String, Value)>, String> {
        if !self.cursors.iter().any(|c| c.name == name) {
            let known: Vec<&str> = self.cursors.iter().map(|c| c.name.as_str()).collect();
            return Err(format!(
                "no cursor named '{name}'; this program's cursors are {known:?}"
            ));
        }
        Ok(
            crate::iteration::cursor_partition::cursor_slot_writes(name, partition)
                .into_iter()
                .filter(|(slot, _)| self.by_name.contains_key(slot))
                .collect(),
        )
    }

    /// Renumber the table-kind externs' entries from `base`, in input
    /// order, for a kernel that learns how many entries its nodes own
    /// only after they are compiled.
    pub(crate) fn renumber_entries(&mut self, base: usize) {
        let mut next = base;
        for s in &mut self.slots {
            if s.entry.is_some() {
                s.entry = Some(next);
                next += 1;
            }
        }
    }

    /// `(slot, entry)` for every table-kind extern, for the kernel's
    /// H4 validator list.
    pub(crate) fn table_entries(&self) -> Vec<(usize, usize)> {
        self.slots
            .iter()
            .filter_map(|s| s.entry.map(|e| (s.slot, e)))
            .collect()
    }

    /// The slots that hold handles, for a native verifier's list of
    /// handle slots a caller fills before the call.
    #[cfg(feature = "jit")]
    pub(crate) fn handle_slots(&self) -> Vec<usize> {
        self.slots
            .iter()
            .filter(|s| s.ty.slot_color() == crate::ast::SlotColor::Hdl1)
            .map(|s| s.slot)
            .collect()
    }

    /// Write every carrier extern into the buffer. Handle kinds wait
    /// for `materialize`, which runs at the start of every run.
    pub(crate) fn seed(&self, buffer: &mut [u64]) {
        for s in &self.slots {
            if s.ty.slot_color() != crate::ast::SlotColor::Hdl1 {
                buffer[s.slot] = carrier_bits(&s.value);
            }
        }
    }

    /// Write every handle-kind extern for the run that is starting:
    /// strings and byte strings into the arena, table kinds into their
    /// entries. Called after the run's generation is set and before
    /// any step reads an input.
    /// Returns whether any slot was marked `None`, so a kernel knows
    /// without scanning its mask.
    pub(crate) fn materialize(
        &self,
        buffer: &mut [u64],
        table: &mut ValueTable,
        mut none: Option<&mut [bool]>,
    ) -> bool {
        let mut any_none = false;
        for s in &self.slots {
            // An unset extern is `None` (A12): the kernel that keeps a
            // `None` mask marks the slot and its consumers propagate
            // it as the interpreter does; a native kernel, which
            // cannot, refuses to run.
            if s.value == Value::None {
                match none.as_deref_mut() {
                    Some(mask) => {
                        mask[s.slot] = true;
                        any_none = true;
                    }
                    None if s.entry.is_some() => panic!(
                        "extern '{}' ({}) has no value: it has no default, so set it with \
                         set_input before the first run (native code cannot carry `None`; \
                         docs/design/engine_parity.md, A12)",
                        s.name, s.ty
                    ),
                    None => {}
                }
            } else if let Some(mask) = none.as_deref_mut() {
                mask[s.slot] = false;
            }
            if s.ty.slot_color() != crate::ast::SlotColor::Hdl1 {
                continue;
            }
            buffer[s.slot] = match (&s.value, s.entry) {
                (Value::Str(text), None) => crate::kernel::put_thread_str(text),
                (Value::Bytes(bytes), None) => crate::kernel::put_thread_bytes(bytes),
                // An unset string extern reads as empty where nothing
                // keeps a `None` mask.
                (Value::None, None) => crate::kernel::put_thread_str(""),
                (other, None) => crate::kernel::put_thread_str(&other.to_display_string()),
                // A table kind: the entry is written every run so the
                // slot's handle names it in this generation; an unset
                // one holds `None`, which decodes as `None`.
                (v, Some(entry)) => table.write(entry, v.clone()),
            };
        }
        any_none
    }

    /// Set an extern by name. The value must be of the declared port
    /// type; `Value::None` clears it to unset. A carrier is written
    /// into `buffer` now; a handle kind is written at the next run.
    /// Returns the extern's first slot, for the caller's dirty marking.
    pub(crate) fn set(
        &mut self,
        name: &str,
        value: Value,
        buffer: &mut [u64],
    ) -> Result<usize, String> {
        let Some(&i) = self.by_name.get(name) else {
            let known: Vec<&str> = self.slots.iter().map(|s| s.name.as_str()).collect();
            return Err(format!(
                "no extern named '{name}'; this kernel's externs are {known:?}"
            ));
        };
        let s = &mut self.slots[i];
        if value != Value::None && value.port_type() != s.ty {
            return Err(format!(
                "extern '{name}' is declared {} but was set to a {} value",
                s.ty,
                value.port_type()
            ));
        }
        s.value = value;
        if s.ty.slot_color() != crate::ast::SlotColor::Hdl1 {
            buffer[s.slot] = carrier_bits(&s.value);
        }
        Ok(s.slot)
    }

    /// The externs by name and declared type, for diagnostics.
    pub(crate) fn names(&self) -> Vec<(&str, PortType)> {
        self.slots.iter().map(|s| (s.name.as_str(), s.ty)).collect()
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
