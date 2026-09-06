// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Thread-local and cycle-scoped bump allocator and handle encoding
//! for non-scalar types in JIT execution frames (SRD 111).
//!
//! ## 64-bit Handle Format
//!
//! Non-scalar values (`String`, `Vec<u8>`, `serde_json::Value`, `&str`)
//! are encoded as 64-bit integer values in JIT buffer slots:
//!
//! - **Tag (bits 62..64)**:
//!   - `0b00`: Static string interner handle (`[Tag: 2][Unused: 30][InternerId: 32]`)
//!   - `0b01`: Dynamic cycle arena slice (`[Tag: 2][Offset: 31][Length: 31]`)
//!   - `0b10`: Value-table handle (`[Tag: 2][Kind: 6][Generation: 24][Entry: 32]`, see `value_table`)
//!
//! This enables non-scalar data to flow through flat 64-bit slot registers
//! without per-operation heap allocations or pointer invalidation risks.

use std::cell::RefCell;
use std::sync::RwLock;

/// Handle Tag constants
pub const TAG_STATIC: u64 = 0b00 << 62;
pub const TAG_ARENA: u64  = 0b01 << 62;
pub const TAG_RES: u64    = 0b10 << 62;
pub const TAG_MASK: u64   = 0b11 << 62;

/// Default initial size for thread-local cycle bump arena (64KB).
const DEFAULT_ARENA_CAPACITY: usize = 64 * 1024;

/// Thread-local cycle bump arena for dynamic strings, byte buffers, and JSON.
pub struct CycleArena {
    buffer: Vec<u8>,
    cursor: usize,
}

impl Default for CycleArena {
    fn default() -> Self {
        Self::new()
    }
}

impl CycleArena {
    pub fn new() -> Self {
        Self {
            buffer: vec![0u8; DEFAULT_ARENA_CAPACITY],
            cursor: 0,
        }
    }

    /// Reset cursor in 1 instruction at cycle boundaries.
    #[inline(always)]
    pub fn reset(&mut self) {
        self.cursor = 0;
    }

    /// Bytes allocated since the last reset.
    #[inline(always)]
    pub fn used(&self) -> usize {
        self.cursor
    }

    /// Allocate raw bytes in the arena and return the mutable slice.
    #[inline]
    pub fn alloc_bytes(&mut self, len: usize) -> &mut [u8] {
        if self.cursor + len > self.buffer.len() {
            let new_cap = (self.buffer.len() * 2).max(self.cursor + len);
            self.buffer.resize(new_cap, 0);
        }
        let start = self.cursor;
        self.cursor += len;
        &mut self.buffer[start..self.cursor]
    }

    /// Copy a byte slice into the arena and return its 64-bit handle.
    #[inline]
    pub fn put_bytes(&mut self, bytes: &[u8]) -> u64 {
        let len = bytes.len();
        let offset = self.cursor;
        let dest = self.alloc_bytes(len);
        dest.copy_from_slice(bytes);
        encode_arena_handle(offset as u32, len as u32)
    }

    /// Copy a string into the arena and return its 64-bit handle.
    #[inline]
    pub fn put_str(&mut self, s: &str) -> u64 {
        self.put_bytes(s.as_bytes())
    }

    /// Resolve a string from a 64-bit handle (static or arena).
    #[inline]
    pub fn resolve_str(&self, handle: u64) -> &str {
        match handle & TAG_MASK {
            TAG_STATIC => StaticInterner::resolve(handle as u32),
            TAG_ARENA => {
                let (offset, len) = decode_arena_handle(handle);
                let bytes = &self.buffer[offset as usize..(offset + len) as usize];
                unsafe { std::str::from_utf8_unchecked(bytes) }
            }
            _ => "",
        }
    }

    /// Resolve bytes from a 64-bit handle (static or arena).
    #[inline]
    pub fn resolve_bytes(&self, handle: u64) -> &[u8] {
        match handle & TAG_MASK {
            TAG_STATIC => StaticInterner::resolve(handle as u32).as_bytes(),
            TAG_ARENA => {
                let (offset, len) = decode_arena_handle(handle);
                &self.buffer[offset as usize..(offset + len) as usize]
            }
            _ => &[],
        }
    }
}

/// Encode arena offset and length into a 64-bit handle (Tag = `0b01`).
#[inline(always)]
pub fn encode_arena_handle(offset: u32, len: u32) -> u64 {
    TAG_ARENA | ((offset as u64 & 0x7FFF_FFFF) << 31) | (len as u64 & 0x7FFF_FFFF)
}

/// Decode arena handle into `(offset, length)`.
#[inline(always)]
pub fn decode_arena_handle(handle: u64) -> (u32, u32) {
    let offset = ((handle >> 31) & 0x7FFF_FFFF) as u32;
    let len = (handle & 0x7FFF_FFFF) as u32;
    (offset, len)
}

thread_local! {
    /// Thread-local cycle arena.
    pub static THREAD_CYCLE_ARENA: RefCell<CycleArena> = RefCell::new(CycleArena::new());
}

/// Helper to execute a closure with the thread-local cycle arena.
#[inline]
pub fn with_cycle_arena<R>(f: impl FnOnce(&mut CycleArena) -> R) -> R {
    THREAD_CYCLE_ARENA.with(|arena| f(&mut arena.borrow_mut()))
}

thread_local! {
    /// The thread's cycle generation: incremented by every root cycle
    /// advance (SRD 115 §4). An arena handle is valid only within the
    /// generation that produced it.
    static CYCLE_GENERATION: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// Begin a root cycle on this thread (SRD 115 §4, axiom H5): reset the
/// cycle arena so its bytes are reused,
/// and advance the generation so a handle held across the boundary can
/// be recognised as stale. Only a root state calls this; nested kernels
/// (traversal activations, projection bodies, materialized subscopes)
/// run inside the root's cycle and never reset.
#[inline]
pub fn begin_root_cycle() -> u64 {
    THREAD_CYCLE_ARENA.with(|arena| arena.borrow_mut().reset());
    CYCLE_GENERATION.with(|g| {
        let next = g.get().wrapping_add(1);
        g.set(next);
        next
    })
}


/// The current cycle generation on this thread.
#[inline]
pub fn cycle_generation() -> u64 {
    CYCLE_GENERATION.with(|g| g.get())
}

/// Bytes currently allocated in this thread's cycle arena; zero right
/// after a root cycle begins.
#[inline]
pub fn cycle_arena_used() -> usize {
    THREAD_CYCLE_ARENA.with(|arena| arena.borrow().used())
}

/// Resolve a string from a 64-bit handle using the thread-local cycle arena.
#[inline]
pub fn resolve_thread_str(handle: u64) -> &'static str {
    THREAD_CYCLE_ARENA.with(|arena| {
        let a = arena.borrow();
        match handle & TAG_MASK {
            TAG_STATIC => StaticInterner::resolve(handle as u32),
            TAG_ARENA => {
                let (offset, len) = decode_arena_handle(handle);
                let bytes = &a.buffer[offset as usize..(offset + len) as usize];
                unsafe { std::mem::transmute::<&str, &'static str>(std::str::from_utf8_unchecked(bytes)) }
            }
            _ => "",
        }
    })
}

/// Resolve bytes from a 64-bit handle using the thread-local cycle arena.
#[inline]
pub fn resolve_thread_bytes(handle: u64) -> &'static [u8] {
    THREAD_CYCLE_ARENA.with(|arena| {
        let a = arena.borrow();
        match handle & TAG_MASK {
            TAG_STATIC => StaticInterner::resolve(handle as u32).as_bytes(),
            TAG_ARENA => {
                let (offset, len) = decode_arena_handle(handle);
                let bytes = &a.buffer[offset as usize..(offset + len) as usize];
                unsafe { std::mem::transmute::<&[u8], &'static [u8]>(bytes) }
            }
            _ => &[],
        }
    })
}

/// Put a string into the thread-local cycle arena and return its handle.
#[inline]
pub fn put_thread_str(s: &str) -> u64 {
    THREAD_CYCLE_ARENA.with(|arena| arena.borrow_mut().put_str(s))
}

/// Put bytes into the thread-local cycle arena and return its handle.
#[inline]
pub fn put_thread_bytes(b: &[u8]) -> u64 {
    THREAD_CYCLE_ARENA.with(|arena| arena.borrow_mut().put_bytes(b))
}

/// Global static string interner for workload-compile-time constants
/// (SRD 115 §2.2, tag `static`). Every string literal, `Const<&str>`
/// argument, and tile static run is interned at kernel build, so
/// constants never enter the cycle arena. Interned bytes are immutable
/// and live for the process; interning the same text twice yields the
/// same handle.
pub struct StaticInterner;

struct StaticTable {
    entries: Vec<&'static str>,
    index: std::collections::HashMap<&'static str, u32>,
}

static STATIC_STRINGS: RwLock<Option<StaticTable>> = RwLock::new(None);

impl StaticInterner {
    /// Intern a string and return its static handle.
    pub fn intern(s: &str) -> u64 {
        if let Some(id) = STATIC_STRINGS.read().unwrap().as_ref().and_then(|t| t.index.get(s).copied()) {
            return TAG_STATIC | u64::from(id);
        }
        let mut guard = STATIC_STRINGS.write().unwrap();
        let table = guard.get_or_insert_with(|| StaticTable { entries: Vec::new(), index: std::collections::HashMap::new() });
        if let Some(&id) = table.index.get(s) {
            return TAG_STATIC | u64::from(id);
        }
        let leaked: &'static str = Box::leak(s.to_string().into_boxed_str());
        let id = table.entries.len() as u32;
        table.entries.push(leaked);
        table.index.insert(leaked, id);
        TAG_STATIC | u64::from(id)
    }

    /// Resolve a static handle's id back to its string slice.
    pub fn resolve(id: u32) -> &'static str {
        STATIC_STRINGS
            .read()
            .unwrap()
            .as_ref()
            .and_then(|t| t.entries.get(id as usize).copied())
            .unwrap_or("")
    }

    /// Resolve a whole handle, when it is a static handle.
    pub fn resolve_handle(handle: u64) -> Option<&'static str> {
        if handle & TAG_MASK != TAG_STATIC {
            return None;
        }
        let id = handle as u32;
        STATIC_STRINGS.read().unwrap().as_ref().and_then(|t| t.entries.get(id as usize).copied())
    }

    /// Number of distinct strings interned so far.
    pub fn len() -> usize {
        STATIC_STRINGS.read().unwrap().as_ref().map_or(0, |t| t.entries.len())
    }
}

/// The cycle arena's cursor, for an engine that scopes its arena use to
/// one native call (SRD 115 §3, embedded cones): take the mark before
/// the call and release to it once every output is copied out, so a
/// cycle's arena use is bounded by its largest cone eval rather than
/// the sum of them.
#[inline]
pub fn cycle_arena_mark() -> usize {
    cycle_arena_used()
}

/// Release the cycle arena back to `mark`. Bytes past the mark are
/// dead: every handle into them was decoded before the release.
#[inline]
pub fn cycle_arena_release(mark: usize) {
    THREAD_CYCLE_ARENA.with(|arena| {
        let mut a = arena.borrow_mut();
        debug_assert!(mark <= a.cursor, "arena release past the cursor");
        a.cursor = mark;
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arena_alloc_and_resolve() {
        let mut arena = CycleArena::new();
        let h1 = arena.put_str("hello");
        let h2 = arena.put_str("world");
        assert_eq!(arena.resolve_str(h1), "hello");
        assert_eq!(arena.resolve_str(h2), "world");

        arena.reset();
        let h3 = arena.put_str("fresh");
        assert_eq!(arena.resolve_str(h3), "fresh");
    }

    #[test]
    fn static_interner_roundtrip() {
        let h1 = StaticInterner::intern("test_constant");
        let h2 = StaticInterner::intern("test_constant");
        assert_eq!(h1, h2, "interning same string must return identical handle");

        let arena = CycleArena::new();
        assert_eq!(arena.resolve_str(h1), "test_constant");
    }
}
