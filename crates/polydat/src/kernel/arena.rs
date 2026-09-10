// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Thread-local, cycle-scoped bump allocator and handle encoding for
//! non-scalar types in compiled execution frames (SRD 111, SRD 115).
//!
//! ## 64-bit handle format
//!
//! Non-scalar values are encoded as 64-bit integers in slots:
//!
//! - **Tag (bits 62..64)**:
//!   - `0b00`: Static string interner handle (`[Tag: 2][Unused: 30][InternerId: 32]`)
//!   - `0b01`: Dynamic cycle arena slice (`[Tag: 2][Offset: 31][Length: 31]`)
//!   - `0b10`: Value-table handle (`[Tag: 2][Kind: 6][Generation: 24][Entry: 32]`, see `value_table`)
//!
//! ## The arena is chunked
//!
//! The arena is a list of fixed-size chunks rather than one growable
//! buffer, so bytes already written never move within a cycle. That is
//! what makes two things sound: a resolved `&str` stays valid while a
//! helper allocates more (a concat, a case change, a render), and an
//! [`ArenaWriter`] can write a result straight into the arena while it
//! is produced, with no intermediate `String`. An arena handle's offset
//! is `chunk << 16 | position`; a chunk holds either small allocations
//! bumped within its first 64 KiB or one large allocation at position
//! zero. Chunks are retained across resets, so a steady cycle allocates
//! nothing.

use std::cell::RefCell;
use std::sync::RwLock;

/// Handle Tag constants
pub const TAG_STATIC: u64 = 0b00 << 62;
pub const TAG_ARENA: u64 = 0b01 << 62;
pub const TAG_RES: u64 = 0b10 << 62;
pub const TAG_MASK: u64 = 0b11 << 62;

/// Chunk size and the offset split: `offset = chunk << CHUNK_SHIFT | position`.
const CHUNK_SHIFT: u32 = 16;
const CHUNK_SIZE: usize = 1 << CHUNK_SHIFT;
const POSITION_MASK: usize = CHUNK_SIZE - 1;

/// A position in the arena, taken with [`CycleArena::mark`] and restored
/// with [`CycleArena::release`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArenaMark {
    chunk: usize,
    cursor: usize,
    total: usize,
}

/// Thread-local cycle bump arena for dynamic strings and byte buffers.
pub struct CycleArena {
    chunks: Vec<Vec<u8>>,
    /// The chunk allocations currently go into.
    chunk: usize,
    /// The next free byte within `chunk`.
    cursor: usize,
    /// Bytes allocated since the last reset.
    total: usize,
}

impl Default for CycleArena {
    fn default() -> Self {
        Self::new()
    }
}

impl CycleArena {
    pub fn new() -> Self {
        Self {
            chunks: vec![vec![0u8; CHUNK_SIZE]],
            chunk: 0,
            cursor: 0,
            total: 0,
        }
    }

    /// Reset at a cycle boundary: back to the first chunk, chunks kept.
    #[inline(always)]
    pub fn reset(&mut self) {
        self.chunk = 0;
        self.cursor = 0;
        self.total = 0;
    }

    /// Bytes allocated since the last reset.
    #[inline(always)]
    pub fn used(&self) -> usize {
        self.total
    }

    /// The current position, to release back to.
    #[inline]
    pub fn mark(&self) -> ArenaMark {
        ArenaMark {
            chunk: self.chunk,
            cursor: self.cursor,
            total: self.total,
        }
    }

    /// Release back to `mark`. Bytes past it are dead: every handle into
    /// them was decoded before the release.
    #[inline]
    pub fn release(&mut self, mark: ArenaMark) {
        debug_assert!(
            (mark.chunk, mark.cursor) <= (self.chunk, self.cursor),
            "arena release past the cursor"
        );
        self.chunk = mark.chunk;
        self.cursor = mark.cursor;
        self.total = mark.total;
    }

    /// Where the next allocation of `len` bytes will land, moving to a
    /// chunk that can hold it first. Small allocations bump within a
    /// chunk's first 64 KiB; a larger one takes a chunk of its own at
    /// position zero, replacing a retained chunk that is too small.
    fn place(&mut self, len: usize) -> (usize, usize) {
        if len <= CHUNK_SIZE {
            if self.cursor + len > CHUNK_SIZE {
                self.chunk += 1;
                self.cursor = 0;
                if self.chunk == self.chunks.len() {
                    self.chunks.push(vec![0u8; CHUNK_SIZE]);
                }
            }
        } else {
            if self.cursor != 0 {
                self.chunk += 1;
                self.cursor = 0;
            }
            if self.chunk == self.chunks.len() {
                self.chunks.push(vec![0u8; len]);
            } else if self.chunks[self.chunk].len() < len {
                self.chunks[self.chunk] = vec![0u8; len];
            }
        }
        (self.chunk, self.cursor)
    }

    /// A mutable view of exactly `[start, start + len)` of a chunk.
    ///
    /// Never `&mut self.chunks[chunk][..]`: indexing a `Vec` mutably
    /// reborrows the whole chunk, which under the aliasing model
    /// invalidates every shared reference a helper still holds into that
    /// chunk (a resolved `&str` being copied or formatted). A view built
    /// from the raw pointer covers only the fresh bytes, so it coexists
    /// with those references; the Miri lane checks this.
    ///
    /// # Safety
    /// The range must lie within the chunk and must not overlap any live
    /// reference, which holds for fresh bytes past the cursor.
    #[inline]
    unsafe fn fresh(&mut self, chunk: usize, start: usize, len: usize) -> &mut [u8] {
        debug_assert!(start + len <= self.chunks[chunk].len());
        unsafe { std::slice::from_raw_parts_mut(self.chunks[chunk].as_mut_ptr().add(start), len) }
    }

    /// Allocate `len` bytes and return the mutable slice.
    #[inline]
    pub fn alloc_bytes(&mut self, len: usize) -> &mut [u8] {
        let (chunk, start) = self.place(len);
        self.cursor = start + len;
        self.total += len;
        // SAFETY: fresh bytes past the cursor.
        unsafe { self.fresh(chunk, start, len) }
    }

    /// Copy a byte slice into the arena and return its 64-bit handle.
    /// `bytes` may borrow the arena itself: chunks never move, and the
    /// destination view covers only fresh bytes.
    #[inline]
    pub fn put_bytes(&mut self, bytes: &[u8]) -> u64 {
        let len = bytes.len();
        let (chunk, start) = self.place(len);
        self.cursor = start + len;
        self.total += len;
        // SAFETY: fresh bytes past the cursor; `bytes` lies before the
        // cursor or outside the arena, so the ranges cannot overlap.
        unsafe { self.fresh(chunk, start, len).copy_from_slice(bytes) };
        encode_arena_handle(((chunk << CHUNK_SHIFT) | start) as u32, len as u32)
    }

    /// Copy a string into the arena and return its 64-bit handle.
    #[inline]
    pub fn put_str(&mut self, s: &str) -> u64 {
        self.put_bytes(s.as_bytes())
    }

    /// The bytes an arena handle names: a shared view of exactly that
    /// range, built from the raw pointer for the same reason `fresh` is.
    #[inline]
    fn arena_slice(&self, handle: u64) -> &[u8] {
        let (offset, len) = decode_arena_handle(handle);
        let (chunk, start, len) = (
            (offset as usize) >> CHUNK_SHIFT,
            (offset as usize) & POSITION_MASK,
            len as usize,
        );
        debug_assert!(start + len <= self.chunks[chunk].len());
        // SAFETY: the range was allocated by `place` within this chunk and
        // chunks never move or shrink within a cycle.
        unsafe { std::slice::from_raw_parts(self.chunks[chunk].as_ptr().add(start), len) }
    }

    /// Resolve a string from a 64-bit handle (static or arena).
    #[inline]
    pub fn resolve_str(&self, handle: u64) -> &str {
        match handle & TAG_MASK {
            TAG_STATIC => StaticInterner::resolve(handle as u32),
            TAG_ARENA => unsafe { std::str::from_utf8_unchecked(self.arena_slice(handle)) },
            _ => "",
        }
    }

    /// Resolve bytes from a 64-bit handle (static or arena).
    #[inline]
    pub fn resolve_bytes(&self, handle: u64) -> &[u8] {
        match handle & TAG_MASK {
            TAG_STATIC => StaticInterner::resolve(handle as u32).as_bytes(),
            TAG_ARENA => self.arena_slice(handle),
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

/// The cycle arena's position, for an engine that scopes its arena use
/// to one native call (SRD 115 §3, embedded cones): take the mark before
/// the call and release to it once every output is copied out, so a
/// cycle's arena use is bounded by its largest cone eval rather than
/// the sum of them.
#[inline]
pub fn cycle_arena_mark() -> ArenaMark {
    THREAD_CYCLE_ARENA.with(|arena| arena.borrow().mark())
}

/// Release the cycle arena back to `mark`. Bytes past the mark are
/// dead: every handle into them was decoded before the release.
#[inline]
pub fn cycle_arena_release(mark: ArenaMark) {
    THREAD_CYCLE_ARENA.with(|arena| arena.borrow_mut().release(mark));
}

/// Resolve a string from a 64-bit handle using the thread-local cycle
/// arena. The reference is valid until the arena's next reset or a
/// release past the handle's bytes: chunks never move within a cycle.
#[inline]
pub fn resolve_thread_str(handle: u64) -> &'static str {
    THREAD_CYCLE_ARENA.with(|arena| {
        let a = arena.borrow();
        // SAFETY: arena bytes stay at their address until the arena is
        // reset or released past them (SRD 115 axiom H3), and every
        // holder of the reference is inside that interval.
        unsafe { std::mem::transmute::<&str, &'static str>(a.resolve_str(handle)) }
    })
}

/// Resolve bytes from a 64-bit handle using the thread-local cycle
/// arena; the same lifetime as [`resolve_thread_str`].
#[inline]
pub fn resolve_thread_bytes(handle: u64) -> &'static [u8] {
    THREAD_CYCLE_ARENA.with(|arena| {
        let a = arena.borrow();
        // SAFETY: as in `resolve_thread_str`.
        unsafe { std::mem::transmute::<&[u8], &'static [u8]>(a.resolve_bytes(handle)) }
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

/// A result being written straight into the thread's cycle arena
/// (SRD 115 §6): a helper appends the bytes it produces and, when done,
/// takes the handle of the whole. No intermediate `String` exists.
///
/// The writer extends its allocation at the arena's cursor. If something
/// else allocated in between (a projection body's kernels running inside
/// a tile render), or the current chunk is full, the bytes so far are
/// moved to a fresh allocation and writing continues there; the result
/// is always one contiguous range. Only one writer is open at a time on
/// a thread; nested helpers finish theirs before returning.
pub struct ArenaWriter {
    chunk: usize,
    start: usize,
    len: usize,
}

impl Default for ArenaWriter {
    fn default() -> Self {
        Self::new()
    }
}

impl ArenaWriter {
    /// Open a writer at the arena's cursor.
    pub fn new() -> Self {
        let (chunk, start) = with_cycle_arena(|a| (a.chunk, a.cursor));
        ArenaWriter {
            chunk,
            start,
            len: 0,
        }
    }

    /// Append bytes.
    pub fn push(&mut self, bytes: &[u8]) {
        if bytes.is_empty() {
            return;
        }
        with_cycle_arena(|a| {
            let contiguous = a.chunk == self.chunk && a.cursor == self.start + self.len;
            if contiguous
                && self.start + self.len + bytes.len()
                    <= a.chunks[self.chunk]
                        .len()
                        .min(CHUNK_SIZE)
                        .max(self.start + self.len)
            {
                let dest = a.alloc_bytes(bytes.len());
                dest.copy_from_slice(bytes);
                self.len += bytes.len();
            } else {
                // Relocate: one fresh allocation for everything so far
                // plus the new bytes. The old bytes stay where they are
                // until the cycle ends; nothing names them.
                let new_len = self.len + bytes.len();
                let (chunk, start) = a.place(new_len);
                a.cursor = start + new_len;
                a.total += new_len;
                if self.len > 0 {
                    let old = a.chunks[self.chunk].as_ptr();
                    // SAFETY: the old range is before the new one (the new
                    // allocation is past the cursor, in this chunk or a
                    // later one), so the ranges are disjoint, and the
                    // destination view covers only the fresh bytes.
                    unsafe {
                        let old = old.add(self.start);
                        let new = a.fresh(chunk, start, self.len);
                        std::ptr::copy_nonoverlapping(old, new.as_mut_ptr(), self.len);
                    }
                }
                // SAFETY: fresh bytes past the copied prefix.
                unsafe {
                    a.fresh(chunk, start + self.len, bytes.len())
                        .copy_from_slice(bytes)
                };
                self.chunk = chunk;
                self.start = start;
                self.len = new_len;
            }
        });
    }

    /// Bytes written so far.
    pub fn len(&self) -> usize {
        self.len
    }

    /// True when nothing has been written.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Close the writer and return the handle of what it wrote.
    pub fn finish(self) -> u64 {
        encode_arena_handle(
            ((self.chunk << CHUNK_SHIFT) | self.start) as u32,
            self.len as u32,
        )
    }
}

impl std::fmt::Write for ArenaWriter {
    fn write_str(&mut self, s: &str) -> std::fmt::Result {
        self.push(s.as_bytes());
        Ok(())
    }
}

impl std::io::Write for ArenaWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.push(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
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
        if let Some(id) = STATIC_STRINGS
            .read()
            .unwrap()
            .as_ref()
            .and_then(|t| t.index.get(s).copied())
        {
            return TAG_STATIC | u64::from(id);
        }
        let mut guard = STATIC_STRINGS.write().unwrap();
        let table = guard.get_or_insert_with(|| StaticTable {
            entries: Vec::new(),
            index: std::collections::HashMap::new(),
        });
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
        STATIC_STRINGS
            .read()
            .unwrap()
            .as_ref()
            .and_then(|t| t.entries.get(id as usize).copied())
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::fmt::Write as _;

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

    /// Bytes never move: a string resolved before the arena grows past
    /// its first chunk is still readable at the same address after.
    #[test]
    fn chunks_keep_earlier_bytes_in_place() {
        let mut arena = CycleArena::new();
        let h = arena.put_str("anchor");
        let p = arena.resolve_str(h).as_ptr();
        for _ in 0..40 {
            arena.put_bytes(&[7u8; 5000]);
        }
        let big = arena.put_bytes(&vec![9u8; 3 * CHUNK_SIZE]);
        assert_eq!(arena.resolve_str(h).as_ptr(), p);
        assert_eq!(arena.resolve_str(h), "anchor");
        assert_eq!(arena.resolve_bytes(big).len(), 3 * CHUNK_SIZE);
        assert!(arena.resolve_bytes(big).iter().all(|&b| b == 9));
        // A small allocation after a large one lands in a fresh chunk.
        let after = arena.put_str("after");
        assert_eq!(arena.resolve_str(after), "after");
    }

    #[test]
    fn a_source_inside_the_arena_can_be_copied_into_it() {
        let mut arena = CycleArena::new();
        let h = arena.put_str("copy me");
        let s: *const str = arena.resolve_str(h);
        // SAFETY: chunks never move; the source stays valid across the put.
        let h2 = arena.put_str(unsafe { &*s });
        assert_eq!(arena.resolve_str(h2), "copy me");
    }

    #[test]
    fn mark_and_release_restore_the_position_across_chunks() {
        let mut arena = CycleArena::new();
        arena.put_str("kept");
        let mark = arena.mark();
        for _ in 0..30 {
            arena.put_bytes(&[1u8; 4000]);
        }
        assert!(arena.chunk > 0);
        arena.release(mark);
        assert_eq!(arena.mark(), mark);
        let h = arena.put_str("next");
        assert_eq!(decode_arena_handle(h).0 as usize & POSITION_MASK, 4);
    }

    #[test]
    fn the_writer_builds_one_contiguous_result() {
        begin_root_cycle();
        let mut w = ArenaWriter::new();
        let n = 12;
        write!(w, "{n}-ab").unwrap();
        w.push(b"!");
        let h = w.finish();
        assert_eq!(resolve_thread_str(h), "12-ab!");
    }

    #[test]
    fn the_writer_relocates_when_something_allocates_between_pushes() {
        begin_root_cycle();
        let mut w = ArenaWriter::new();
        w.push(b"head");
        let other = put_thread_str("interleaved");
        w.push(b"-tail");
        let h = w.finish();
        assert_eq!(resolve_thread_str(h), "head-tail");
        assert_eq!(resolve_thread_str(other), "interleaved");
    }

    #[test]
    fn the_writer_crosses_a_chunk_boundary() {
        begin_root_cycle();
        put_thread_bytes(&vec![0u8; CHUNK_SIZE - 10]);
        let mut w = ArenaWriter::new();
        w.push(b"12345");
        w.push(&[b'x'; 20]);
        let h = w.finish();
        let s = resolve_thread_str(h);
        assert_eq!(s.len(), 25);
        assert!(s.starts_with("12345") && s.ends_with("xxxxx"));
    }
}
