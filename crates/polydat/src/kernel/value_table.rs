// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! The value table (SRD 115 §3): where `Json`, `Ext`, and `Handle`
//! values live while native code holds them by handle.
//!
//! A table is owned by the engine that runs the native code, never by
//! the thread: a whole compiled kernel owns one for its lifetime, an
//! embedded cone borrows one for the duration of a single eval. Its
//! length is fixed at compile time, one entry per table-kind slot the
//! native code can write, and each entry is written in place by the
//! one helper that owns it (axiom H4). Native code reaches the table
//! through the context the engine installs around the call
//! ([`with_value_table`]); a helper that runs outside such a context is
//! a bug and panics.
//!
//! ## Handle format
//!
//! `[tag:2 = table][kind:6][generation:24][entry:32]`
//!
//! The kind is the value's variant, so a reader can refuse a handle of
//! the wrong kind without touching the entry. The generation is the low
//! bits of the owning engine's cycle generation when the entry was
//! written, so a handle that outlives its cycle is recognised when it is
//! read (axiom H3) rather than silently naming a newer value.

use std::cell::Cell;

use super::arena::{TAG_MASK, TAG_RES};
use crate::ast::Value;

const KIND_SHIFT: u32 = 56;
const KIND_MASK: u64 = 0x3F;
const GEN_SHIFT: u32 = 32;
const GEN_MASK: u64 = 0xFF_FFFF;
const ENTRY_MASK: u64 = 0xFFFF_FFFF;

/// Kind codes carried in a table handle.
pub const KIND_JSON: u64 = 1;
pub const KIND_EXT: u64 = 2;
pub const KIND_HANDLE: u64 = 3;
/// Any other variant the engine admits at a boundary.
pub const KIND_OTHER: u64 = 0;

fn kind_of_value(v: &Value) -> u64 {
    match v {
        Value::Json(_) => KIND_JSON,
        Value::Ext(_) => KIND_EXT,
        Value::Handle(_) => KIND_HANDLE,
        _ => KIND_OTHER,
    }
}

/// Assemble a table handle.
#[inline]
pub fn encode_table_handle(kind: u64, generation: u64, entry: usize) -> u64 {
    TAG_RES
        | ((kind & KIND_MASK) << KIND_SHIFT)
        | ((generation & GEN_MASK) << GEN_SHIFT)
        | (entry as u64 & ENTRY_MASK)
}

/// Take a table handle apart: `(kind, generation, entry)`.
#[inline]
pub fn decode_table_handle(handle: u64) -> (u64, u64, usize) {
    (
        (handle >> KIND_SHIFT) & KIND_MASK,
        (handle >> GEN_SHIFT) & GEN_MASK,
        (handle & ENTRY_MASK) as usize,
    )
}

/// A fixed set of entries, one per table-kind slot of the engine that
/// owns it.
#[derive(Clone)]
pub struct ValueTable {
    entries: Vec<Option<Value>>,
    generation: u64,
}

impl ValueTable {
    /// A table with `len` unwritten entries.
    pub fn new(len: usize) -> Self {
        Self {
            entries: (0..len).map(|_| None).collect(),
            generation: 0,
        }
    }

    /// Number of entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// True when the table has no entries.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Drop every value and set the entry count. Capacity is retained.
    pub fn resize(&mut self, len: usize) {
        self.entries.clear();
        self.entries.resize_with(len, || None);
    }

    /// Drop every value; the entries stay allocated.
    pub fn clear(&mut self) {
        for e in &mut self.entries {
            *e = None;
        }
    }

    /// The generation stamped into handles written from now on.
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// Stamp handles written from now on with `generation`.
    pub fn set_generation(&mut self, generation: u64) {
        self.generation = generation;
    }

    /// True once `entry` holds a value.
    pub fn is_written(&self, entry: usize) -> bool {
        self.entries.get(entry).is_some_and(|e| e.is_some())
    }

    /// Replace `entry` with `v` and return the handle that names it.
    #[inline]
    pub fn write(&mut self, entry: usize, v: Value) -> u64 {
        let kind = kind_of_value(&v);
        match self.entries.get_mut(entry) {
            Some(slot) => *slot = Some(v),
            None => panic!(
                "value table entry {entry} does not exist in a {}-entry table; \
                 the layout assigns one entry per table-kind slot (SRD 115 §3)",
                self.entries.len()
            ),
        }
        encode_table_handle(kind, self.generation, entry)
    }

    /// The value a handle names, borrowed. Refuses a handle that is not
    /// a table handle, that names a missing or unwritten entry, or that
    /// was written in another generation (axiom H3).
    #[inline]
    pub fn get(&self, handle: u64) -> &Value {
        if handle & TAG_MASK != TAG_RES {
            panic!("{handle:#x} is not a value-table handle");
        }
        let (_, generation, entry) = decode_table_handle(handle);
        let current = self.generation & GEN_MASK;
        if generation != current {
            panic!(
                "table handle {handle:#x} was written in cycle generation {generation}, but the table is at {current}; \
                 a value-table handle lives one cycle (SRD 115, axiom H3)"
            );
        }
        match self.entries.get(entry) {
            Some(Some(v)) => v,
            Some(None) => panic!(
                "table handle {handle:#x} names entry {entry}, which has not been written this cycle"
            ),
            None => panic!(
                "table handle {handle:#x} names entry {entry} of a {}-entry table",
                self.entries.len()
            ),
        }
    }

    /// The value a handle names, cloned out (axiom H6: the caller never
    /// holds the handle afterwards).
    #[inline]
    pub fn read(&self, handle: u64) -> Value {
        self.get(handle).clone()
    }
}

impl Default for ValueTable {
    fn default() -> Self {
        Self::new(0)
    }
}

thread_local! {
    /// The table native code on this thread writes through, installed
    /// by the engine around each native call. Null outside a call.
    static CURRENT_TABLE: Cell<*mut ValueTable> = const { Cell::new(std::ptr::null_mut()) };
}

/// An installation of a table as the one compiled code on this thread
/// writes through (see [`install_value_table`]). Dropping it restores
/// the previous installation.
pub struct TableInstallation<'a> {
    previous: *mut ValueTable,
    _table: std::marker::PhantomData<&'a mut ValueTable>,
}

impl Drop for TableInstallation<'_> {
    fn drop(&mut self) {
        CURRENT_TABLE.set(self.previous);
    }
}

/// Install `table` as the one compiled code on this thread writes
/// through until the returned guard drops. The borrow of `table` lasts
/// as long as the guard, so the engine cannot touch the table while
/// compiled code may.
#[inline]
pub fn install_value_table(table: &mut ValueTable) -> TableInstallation<'_> {
    TableInstallation {
        previous: CURRENT_TABLE.replace(table as *mut ValueTable),
        _table: std::marker::PhantomData,
    }
}

/// Run `f` with `table` installed as the table native code writes
/// through. The previous installation, if any, is restored afterwards,
/// including on unwind, so an engine nested inside another engine's
/// call installs and releases its own table cleanly.
#[inline]
pub fn with_value_table<R>(table: &mut ValueTable, f: impl FnOnce() -> R) -> R {
    let _installed = install_value_table(table);
    f()
}

// ── The compiled closure ABI (SRD 115 §7) ────────────────────────
// A P2 closure over handle slots reaches the table the same way a
// native helper does, through the installation its kernel made. These
// are the entry points the `#[polydat_node]` macro's `compiled_handle`
// kit calls; they are public because generated code in other crates
// calls them, not because hosts should.

/// Slot bits as the owned `Value` a port of type `ty` carries: scalars
/// from bits, byte strings copied out of the arena, table kinds read
/// from the installed table.
#[inline]
pub fn decode_arg(ty: crate::ast::PortType, bits: u64) -> Value {
    match ty.handle_kind() {
        Some(crate::ast::HandleKind::Table) => with_current_value_table(|t| t.read(bits)),
        _ => crate::compile::marshal::decode_slot(bits, ty, &ValueTable::new(0)),
    }
}

/// An owned `Value` as the slot bits a port of type `ty` carries: the
/// inverse of [`decode_arg`], for a polymorphic return. A table kind
/// is written to `entry` of the installed table, which the kernel
/// assigned to the output slot. The value must be of the port's type:
/// the graph colored the slot by the node's resolved output type, and
/// a value of another type would be read by every consumer as
/// something it is not, where the interpreter would have carried it.
/// A `None` has no scalar slot form on a compiled engine
/// (engine_parity.md, A12).
#[inline]
pub fn encode_arg(ty: crate::ast::PortType, v: Value, entry: Option<usize>) -> u64 {
    if v.port_type() != ty {
        panic!(
            "a node produced a {:?} on an output the graph typed {:?}; a compiled engine \
             cannot carry a value of another type than the slot's (engine_parity.md, A7)",
            v.port_type(),
            ty
        );
    }
    match v {
        Value::U64(x) => x,
        Value::I64(x) => x as u64,
        Value::F64(x) => x.to_bits(),
        Value::Bool(b) => b as u64,
        Value::Str(s) => crate::kernel::put_thread_str(&s),
        Value::Bytes(b) => crate::kernel::put_thread_bytes(&b),
        Value::Json(_) | Value::Ext(_) | Value::Handle(_) => {
            let entry = entry.unwrap_or_else(|| {
                panic!("a {ty:?} output has no value-table entry assigned (SRD 115 §3)")
            });
            write_table_entry(entry, v)
        }
        other => panic!(
            "a {:?} value has no one-slot compiled form",
            other.port_type()
        ),
    }
}

/// The value a table handle names, borrowed from the installed table
/// for the rest of the current native call. The entry is written only
/// by its one producing step (H4), which ran before any consumer, and
/// the table is cleared or dropped only after the call returns, so a
/// helper may hold this for its own duration and no longer.
#[inline]
pub fn current_table_value(bits: u64) -> &'static Value {
    with_current_value_table(|t| {
        // SAFETY: see above; the same interval `resolve_thread_str`
        // relies on for arena bytes, restated for table entries.
        unsafe { std::mem::transmute::<&Value, &'static Value>(t.get(bits)) }
    })
}

/// The JSON value a table handle names, shared.
#[inline]
pub fn read_table_json(bits: u64) -> std::sync::Arc<serde_json::Value> {
    with_current_value_table(|t| match t.get(bits) {
        Value::Json(j) => j.clone(),
        other => panic!("a Json port holds a {:?} table entry", other.port_type()),
    })
}

/// Write `v` to `entry` of the installed table and return its handle.
#[inline]
pub fn write_table_entry(entry: usize, v: Value) -> u64 {
    with_current_value_table(|t| t.write(entry, v))
}

/// Access the installed table from a native helper. Panics when no
/// engine has installed one: a table write outside an engine's native
/// call has no owner and is a bug.
#[inline]
pub fn with_current_value_table<R>(f: impl FnOnce(&mut ValueTable) -> R) -> R {
    let p = CURRENT_TABLE.get();
    assert!(
        !p.is_null(),
        "no value table is installed on this thread; native code that touches a table entry \
         runs only inside the owning engine's with_value_table (SRD 115 §3)"
    );
    // SAFETY: `p` was installed by `with_value_table` from a `&mut
    // ValueTable` that outlives the native call this helper runs
    // inside, and the engine touches nothing else in the table until
    // the call returns, so this is the only live access.
    f(unsafe { &mut *p })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn handle_fields_round_trip() {
        let h = encode_table_handle(KIND_EXT, 0xABCDEF, 7);
        assert_eq!(h & TAG_MASK, TAG_RES);
        assert_eq!(decode_table_handle(h), (KIND_EXT, 0xABCDEF, 7));
    }

    #[test]
    fn entries_are_replaced_in_place() {
        let mut t = ValueTable::new(2);
        t.set_generation(5);
        let a = t.write(1, Value::U64(1));
        let b = t.write(1, Value::U64(2));
        assert_eq!(
            a, b,
            "the same entry yields the same handle within a generation"
        );
        assert_eq!(t.read(b).as_u64(), 2);
        assert!(!t.is_written(0));
    }

    #[test]
    fn a_helper_needs_an_installed_table() {
        let r = std::panic::catch_unwind(|| with_current_value_table(|t| t.len()));
        assert!(r.is_err());
        let mut t = ValueTable::new(3);
        let n = with_value_table(&mut t, || with_current_value_table(|t| t.len()));
        assert_eq!(n, 3);
        let r = std::panic::catch_unwind(|| with_current_value_table(|t| t.len()));
        assert!(r.is_err(), "the installation is released after the call");
    }
}
