// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! The static string interner: workload-compile-time string constants
//! with process lifetime. Every string literal, `Const<&str>` argument,
//! and tile static run is interned at kernel build; a compiled step
//! that produces such a constant publishes a `(ptr, len)` pair to the
//! interned bytes, which never move and are never freed, so the pair
//! has a proven owner for the life of the process (jit_boundary.md,
//! axiom S7). Interning the same text twice yields the same bytes.

use std::sync::RwLock;

/// Global static string interner for compile-time constants.
pub struct StaticInterner;

struct StaticTable {
    entries: Vec<&'static str>,
    index: std::collections::HashMap<&'static str, u32>,
}

static STATIC_STRINGS: RwLock<Option<StaticTable>> = RwLock::new(None);

impl StaticInterner {
    /// Intern a string and return its static slice.
    pub fn intern(s: &str) -> &'static str {
        // One read lock, released before the write lock is taken: a
        // second read taken while a writer waits would deadlock.
        let found: Option<&'static str> = STATIC_STRINGS
            .read()
            .unwrap()
            .as_ref()
            .and_then(|t| t.index.get(s).map(|&id| t.entries[id as usize]));
        if let Some(text) = found {
            return text;
        }
        let mut guard = STATIC_STRINGS.write().unwrap();
        let table = guard.get_or_insert_with(|| StaticTable {
            entries: Vec::new(),
            index: std::collections::HashMap::new(),
        });
        if let Some(&id) = table.index.get(s) {
            return table.entries[id as usize];
        }
        let leaked: &'static str = Box::leak(s.to_string().into_boxed_str());
        let id = table.entries.len() as u32;
        table.entries.push(leaked);
        table.index.insert(leaked, id);
        leaked
    }

    /// The interned string with id `id`; empty when there is none.
    pub fn resolve(id: u32) -> &'static str {
        STATIC_STRINGS
            .read()
            .unwrap()
            .as_ref()
            .and_then(|t| t.entries.get(id as usize).copied())
            .unwrap_or("")
    }

    /// Number of distinct strings interned so far.
    pub fn len() -> usize {
        STATIC_STRINGS
            .read()
            .unwrap()
            .as_ref()
            .map_or(0, |t| t.entries.len())
    }
}

/// The `(ptr, len)` pair of a static string, as a `Ref2` output slot
/// pair publishes it.
#[inline]
pub fn static_pair(s: &'static str) -> (u64, u64) {
    (s.as_ptr() as usize as u64, s.len() as u64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interning_dedups_and_keeps_bytes_in_place() {
        let a = StaticInterner::intern("intern-test-constant");
        let b = StaticInterner::intern("intern-test-constant");
        assert_eq!(
            a.as_ptr(),
            b.as_ptr(),
            "the same text interns to the same bytes"
        );
        assert_eq!(a, "intern-test-constant");
        let (p, l) = static_pair(a);
        assert_eq!(p, a.as_ptr() as usize as u64);
        assert_eq!(l, a.len() as u64);
    }
}
