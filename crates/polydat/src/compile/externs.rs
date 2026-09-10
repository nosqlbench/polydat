// Copyright 2024-2026 Jonathan Shook
// SPDX-Licene-Identifier: Apache-2.0

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
#[derive(Default)]
pub(crate) struct Externs {
    slots: Vec<ExternSlot>,
    by_name: HashMap<String, usize>,
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
        Ok(Self { slots, by_name })
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
    pub(crate) fn materialize(&self, buffer: &mut [u64], table: &mut ValueTable) {
        for s in &self.slots {
            if s.ty.slot_color() != crate::ast::SlotColor::Hdl1 {
                continue;
            }
            buffer[s.slot] = match (&s.value, s.entry) {
                (Value::Str(text), None) => crate::kernel::put_thread_str(text),
                (Value::Bytes(bytes), None) => crate::kernel::put_thread_bytes(bytes),
                // An unset string extern reads as empty, as a missing
                // default does on the interpreter after `as_str`.
                (Value::None, None) => crate::kernel::put_thread_str(""),
                (other, None) => crate::kernel::put_thread_str(&other.to_display_string()),
                // A table kind without a value has nothing its
                // consumers could decode; the engines run eagerly, so
                // the run cannot start. The interpreter fails at the
                // consumer instead, which is later and less clear.
                (Value::None, Some(_)) => panic!(
                    "extern '{}' ({}) has no value: it has no default, so set it with \
                     set_input before the first run",
                    s.name, s.ty
                ),
                // A table kind: the entry is written every run so the
                // slot's handle names it in this generation.
                (v, Some(entry)) => table.write(entry, v.clone()),
            };
        }
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
