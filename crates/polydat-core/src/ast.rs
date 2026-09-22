// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Core types for Polydat nodes: values, ports, metadata, and the evaluation trait.
//!
//! The Polydat type system has three layers:
//!
//! 1. **Runtime values** ([`Value`]) — the enum that flows through
//!    the DAG at evaluation time. Every interpreter buffer slot holds
//!    a `Value`; compiled kernels carry the same values as typed
//!    `u64` slots.
//!
//! 2. **Port types** ([`PortType`]) — compile-time type tags on
//!    node input/output ports. The assembler validates that wiring
//!    connects compatible types and auto-inserts adapters when not.
//!
//! 3. **Slot types** ([`SlotType`]) — distinguishes wire inputs
//!    (cycle-time values) from constant parameters (baked at
//!    construction). The DSL compiler uses these to decide whether
//!    a literal in a function call is a wire promotion or a const arg.
//!
//! The [`PolydatNode`] trait is what every node function implements.
//! A node declares its port metadata via [`NodeMeta`] and evaluates
//! via `eval(&[Value], &mut [Value])`.

use std::fmt;
use std::ops::Deref;
use std::sync::Arc;

/// Arc-managed typed slice. Holds a borrow into a parent Arc'd
/// owner — typically either an owned backing buffer (`Arc<[T]>`)
/// or a long-lived resource like an mmap'd dataset. Cloning is
/// one `Arc::clone` (atomic increment, zero allocations); the
/// owner is type-erased as `Arc<dyn Any + Send + Sync>` so the
/// same `SliceArc<T>` shape covers both modes.
///
/// Used by [`Value::VecF32`] / [`Value::VecI32`] to flow vector
/// data on wires from accessors to native-binding adapters with:
///   - zero per-cycle allocation when the source supports
///     zero-copy reads (mmap-backed `VectorReader::get_slice`),
///   - exactly one allocation when it doesn't (a `Vec<T>` from
///     `VectorReader::get`, wrapped into an `Arc<[T]>`).
///
/// See SRD 53 §"Native Vector Binding".
pub struct SliceArc<T: 'static> {
    /// Keeps the storage alive. For owned data this is an
    /// `Arc<OwnedSlice<T>>`; for mmap-backed data this is an
    /// `Arc<UniformDataset<T>>` (or any other type whose Arc
    /// keeps the underlying memory mapped).
    _owner: Arc<dyn std::any::Any + Send + Sync>,
    ptr: *const T,
    len: usize,
}

// Send/Sync: the raw pointer is treated as a borrow into memory
// owned by `_owner`, which is itself Send+Sync. T must be
// Send+Sync for the slice contents to be safely shared.
unsafe impl<T: Send + Sync + 'static> Send for SliceArc<T> {}
unsafe impl<T: Send + Sync + 'static> Sync for SliceArc<T> {}

/// Type-erasable wrapper for an owned `Arc<[T]>`. Used as the
/// owner when the source isn't zero-copy — `Arc<[T]>` is unsized
/// so it can't be cast to `Arc<dyn Any>` directly, but
/// `OwnedSlice<T>` is sized and the cast works.
// Field is unused at the type level — its only job is to keep the
// Arc<[T]> reference count alive while the SliceArc holds the raw
// pointer into the buffer. Hence the `dead_code` allow.
#[allow(dead_code)]
pub(crate) struct OwnedSlice<T: 'static>(pub(crate) Arc<[T]>);

impl<T: Send + Sync + 'static> SliceArc<T> {
    /// Build from an owned `Vec<T>`. One heap allocation
    /// (`Vec → Arc<[T]>`); cloning the resulting `SliceArc<T>` is
    /// one atomic increment.
    pub fn from_vec(v: Vec<T>) -> Self {
        let arc: Arc<[T]> = Arc::from(v);
        let ptr = arc.as_ptr();
        let len = arc.len();
        let owner: Arc<dyn std::any::Any + Send + Sync> = Arc::new(OwnedSlice(arc));
        Self {
            _owner: owner,
            ptr,
            len,
        }
    }

    /// Build from a `&[T]` borrowed from `owner`'s data.
    ///
    /// # Safety
    ///
    /// `slice` must point into memory owned by `owner` and
    /// remain valid for at least as long as `owner` (i.e., until
    /// the last clone of this Arc is dropped). The caller asserts
    /// this — typical use is mmap-backed readers where the slice
    /// is a view into a memory-mapped page kept alive by the
    /// dataset Arc.
    pub unsafe fn from_borrowed(owner: Arc<dyn std::any::Any + Send + Sync>, slice: &[T]) -> Self {
        Self {
            _owner: owner,
            ptr: slice.as_ptr(),
            len: slice.len(),
        }
    }
}

impl<T: 'static> SliceArc<T> {
    /// Borrow as `&[T]`. The borrow lives as long as `&self`.
    /// Defined here without Send+Sync bounds so it's reachable
    /// from `Deref`/`PartialEq`/`Debug` impls that don't carry
    /// those bounds.
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        // SAFETY: `_owner` keeps the storage alive; `ptr`/`len`
        // were validated at construction. The returned reference
        // is bounded by `&self`'s lifetime.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

impl<T: Send + Sync + 'static> Clone for SliceArc<T> {
    fn clone(&self) -> Self {
        Self {
            _owner: self._owner.clone(),
            ptr: self.ptr,
            len: self.len,
        }
    }
}

impl<T: 'static> Deref for SliceArc<T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        // SAFETY: identical reasoning to as_slice().
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

impl<T: PartialEq + 'static> PartialEq for SliceArc<T> {
    fn eq(&self, other: &Self) -> bool {
        // Pointer-equal pair → trivially equal (zero-copy from the
        // same source). Otherwise compare contents — two unrelated
        // SliceArcs may hold equal data.
        if std::ptr::eq(self.ptr, other.ptr) && self.len == other.len {
            return true;
        }
        self.as_slice() == other.as_slice()
    }
}

impl<T: fmt::Debug + 'static> fmt::Debug for SliceArc<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SliceArc")
            .field("len", &self.len)
            .field("first", &self.as_slice().first())
            .finish_non_exhaustive()
    }
}

/// Two-limb carrier for 128-bit integers inside [`Value`].
///
/// Limbs are little-endian (`[lo, hi]`). Using `[u64; 2]` instead
/// of a raw `u128`/`i128` field keeps `Value`'s alignment at 8 and
/// its size inside the 40-byte buffer-slot envelope; reassembly is
/// two register moves.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Bits128(pub [u64; 2]);

impl Bits128 {
    #[inline]
    /// The two-word form of a `u128`, low word first.
    pub fn from_u128(v: u128) -> Self {
        Self([v as u64, (v >> 64) as u64])
    }
    #[inline]
    /// The two-word form of an `i128`, low word first.
    pub fn from_i128(v: i128) -> Self {
        Self::from_u128(v as u128)
    }
    /// The word as a `u128`.
    #[inline]
    pub fn as_u128(self) -> u128 {
        (self.0[0] as u128) | ((self.0[1] as u128) << 64)
    }
    /// The word as an `i128`.
    #[inline]
    pub fn as_i128(self) -> i128 {
        self.as_u128() as i128
    }

    #[inline]
    /// The word's sixteen bytes, little-endian.
    pub fn to_le_bytes(self) -> [u8; 16] {
        self.as_u128().to_le_bytes()
    }

    #[inline]
    /// A word from sixteen little-endian bytes.
    pub fn from_le_bytes(b: [u8; 16]) -> Self {
        Self::from_u128(u128::from_le_bytes(b))
    }
}

/// Lane-codec macro: `[T; N]` views over the 16-byte word,
/// little-endian lane order (lane 0 = lowest address).
macro_rules! bits128_lanes {
    ($to:ident, $from:ident, $t:ty, $n:expr) => {
        impl Bits128 {
            #[inline]
            /// The word as lanes of one element type, lane 0 at the lowest address.
            pub fn $to(self) -> [$t; $n] {
                let b = self.to_le_bytes();
                let mut out = [<$t>::default(); $n];
                let w = core::mem::size_of::<$t>();
                for (i, lane) in out.iter_mut().enumerate() {
                    let mut lb = [0u8; core::mem::size_of::<$t>()];
                    lb.copy_from_slice(&b[i * w..(i + 1) * w]);
                    *lane = <$t>::from_le_bytes(lb);
                }
                out
            }
            #[inline]
            /// A word from lanes of one element type, lane 0 at the lowest address.
            pub fn $from(lanes: [$t; $n]) -> Self {
                let mut b = [0u8; 16];
                let w = core::mem::size_of::<$t>();
                for (i, lane) in lanes.iter().enumerate() {
                    b[i * w..(i + 1) * w].copy_from_slice(&lane.to_le_bytes());
                }
                Self::from_le_bytes(b)
            }
        }
    };
}

bits128_lanes!(lanes_i8, from_lanes_i8, i8, 16);
bits128_lanes!(lanes_i16, from_lanes_i16, i16, 8);
bits128_lanes!(lanes_i32, from_lanes_i32, i32, 4);
bits128_lanes!(lanes_i64, from_lanes_i64, i64, 2);
bits128_lanes!(lanes_f32, from_lanes_f32, f32, 4);
bits128_lanes!(lanes_f64, from_lanes_f64, f64, 2);

impl Bits128 {
    /// f16 lanes go through the bit-pattern codec (`half::f16`
    /// has no `to_le_bytes`).
    #[inline]
    pub fn lanes_f16(self) -> [half::f16; 8] {
        self.lanes_i16().map(|b| half::f16::from_bits(b as u16))
    }
    #[inline]
    /// A word from eight `f16` lanes, through the bit-pattern codec.
    pub fn from_lanes_f16(lanes: [half::f16; 8]) -> Self {
        Self::from_lanes_i16(lanes.map(|f| f.to_bits() as i16))
    }
}

/// Lane-typing view tag for [`Value::Reg128`] — which
/// interpretation a 128-bit register word currently carries
/// (type_system_alignment.md §8.4 layer 2). `Raw` is the
/// algorithm-defined buffer-state view (heterogeneous lane
/// roles); the typed views are homogeneous `[T; N]` readings.
/// All views are free bitcasts of one another.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RegLanes {
    /// The algorithm-defined view: heterogeneous lane roles, no element type.
    Raw,
    /// Sixteen `i8` lanes.
    I8x16,
    /// Eight `i16` lanes.
    I16x8,
    /// Four `i32` lanes.
    I32x4,
    /// Two `i64` lanes.
    I64x2,
    /// Eight `f16` lanes.
    F16x8,
    /// Four `f32` lanes.
    F32x4,
    /// Two `f64` lanes.
    F64x2,
}

#[derive(Debug, Clone)]
/// A typed value on a wire: what a node reads and produces on the
/// interpreter, and what a host sets and pulls on every engine.
pub enum Value {
    /// Unsigned 64-bit integer. The workhorse type for deterministic
    /// data generation: hash outputs, modular arithmetic, bit
    /// manipulation, cycle counters, primary keys.
    U64(u64),
    /// Unsigned 128-bit integer (cranelift I128, unsigned
    /// interpretation). Carried as two u64 limbs ([`Bits128`],
    /// little-endian limb order) so `Value` keeps alignment 8 —
    /// see the `value_size_probe` test. Carried as two immediate
    /// slots (`SlotColor::Imm2`) in compiled kernels. JSON
    /// projection is a decimal string (JSON Number cannot carry
    /// 128-bit magnitude).
    U128(Bits128),
    /// Signed 128-bit integer (cranelift I128, signed
    /// interpretation). Same limb carrier and conventions as
    /// [`Value::U128`].
    I128(Bits128),
    /// 128-bit SIMD register word (type_system_alignment.md
    /// §8.4 layer 2). The [`RegLanes`] tag records the current
    /// view — a homogeneous lane typing (`[f32; 4]`, `[i16; 8]`,
    /// …) or `Raw` (algorithm-defined buffer state with
    /// heterogeneous lane roles). Views are free bitcasts; the
    /// word is a plain value (two u64 slots in compiled buffers,
    /// no pointers, no lifetime).
    Reg128(Bits128, RegLanes),
    /// Signed 64-bit integer. The honest runtime carrier for
    /// `PortType::I64` (and sign-extended `I32`) slots — matching
    /// `serde_json::Number`'s `NegInt` leaf so display and JSON
    /// projection render negatives as negatives instead of their
    /// unsigned bit-reinterpretation. At the JIT boundary the bits
    /// ride the same u64 slot (`i64 as u64` is a free bitcast), so
    /// signedness costs nothing in compiled kernels. See
    /// `polydat/docs/design/type_system_alignment.md` §5.
    I64(i64),
    /// IEEE 754 double-precision float. Used for distributions,
    /// noise functions, trigonometry, interpolation, and any
    /// computation that needs fractional precision.
    F64(f64),
    /// Boolean. Used for conditional ops (`if:` field), selection
    /// nodes, and flag computation.
    Bool(bool),
    /// Shared, immutable UTF-8 string. Used for formatted output,
    /// weighted string selection, template interpolation, and any
    /// value that will appear directly in an op statement. Backed
    /// by `Arc<str>` so cloning is one atomic increment with no
    /// allocation — the per-cycle reads that materialize a `final`
    /// or `init` string into op-template substitution are
    /// pointer-share, not heap-copy.
    Str(Arc<str>),
    /// Shared, immutable raw byte buffer. Used for cryptographic
    /// digests, binary encoding/decoding, and byte-level data
    /// generation. Backed by `Arc<[u8]>` so cloning is one atomic
    /// increment.
    Bytes(Arc<[u8]>),
    /// Shared, immutable structured JSON value. Used for
    /// vector representations (JSON arrays), complex structured
    /// data, and JSON merge ops. Backed by `Arc<serde_json::Value>`
    /// so cloning is one atomic increment — the per-cycle reads
    /// of result-body JSON wires (capture extraction, recall
    /// evaluation, column projection) share the underlying
    /// allocation rather than deep-cloning the tree. Consumers
    /// that need an owned `serde_json::Value` (mutation,
    /// serialization sinks) explicitly deep-clone via
    /// `(*v).clone()` at the consume site.
    Json(Arc<serde_json::Value>),
    /// Adapter-contributed reflected value. Carries type info and
    /// standard access methods (display, JSON, string, bytes).
    /// Enables protocol-native types (UUIDs, timestamps, inet
    /// addresses) to flow through Polydat without boxing to strings.
    Ext(Box<dyn ReflectedValue>),
    /// Type-erased Arc handle to a resolved resource (dataset,
    /// prepared statement, ...). Cloning during input gather is one
    /// `Arc::clone` — a single atomic increment, zero allocations.
    /// Produced by resolver nodes (e.g. `dataset_open`) and consumed
    /// by reader nodes that downcast to the concrete type. See
    /// SRD 53 §"Dataset Handles" for the canonical use case.
    Handle(Arc<dyn std::any::Any + Send + Sync>),
    /// Typed `f32` vector carrier. Flows from vector accessors to
    /// native-binding adapters without string formatting or byte
    /// serialization on the cycle path. Cloning is one `Arc::clone`,
    /// zero allocations. The underlying [`SliceArc`] supports both
    /// owned (allocated `Arc<[f32]>`) and zero-copy (borrow into a
    /// long-lived owner like an mmap'd dataset) storage modes.
    /// `to_display_string()` renders as JSON array.
    VecF32(SliceArc<f32>),
    /// Typed `i32` vector carrier (e.g. neighbor indices). Same
    /// shape as VecF32 — typed slice on the wire.
    VecI32(SliceArc<i32>),
    /// Typed `f64` vector carrier (`Arc<[f64]>`). Same shape as
    /// VecF32. Used for double-precision embeddings / dense
    /// numeric features bound to CQL `vector<double, N>` etc.
    VecF64(SliceArc<f64>),
    /// Typed `i64` vector carrier (`Arc<[i64]>`). 64-bit integer
    /// vectors for CQL `vector<bigint, N>`.
    VecI64(SliceArc<i64>),
    /// Typed half-precision float vector (`Arc<[half::f16]>`).
    /// 16-bit float carrier — stays at f16 on the wire so
    /// embeddings stored as half-precision aren't widened on the
    /// kernel side.
    VecF16(SliceArc<half::f16>),
    /// Typed `i16` vector carrier (`Arc<[i16]>`). 16-bit signed
    /// integer vectors for CQL `vector<smallint, N>`.
    VecI16(SliceArc<i16>),
    /// Typed `i8` vector carrier (`Arc<[i8]>`). 8-bit signed
    /// integer vectors (CQL `vector<tinyint, N>`); completes the
    /// cranelift lane family {i8, i16, i32, i64, f16, f32, f64}
    /// (type_system_alignment.md §8.2). Unsigned byte buffers are
    /// spelled `Bytes`.
    VecI8(SliceArc<i8>),
    /// The absent value (SRD-74): fresh buffer slots start as
    /// `None`, and the kernel propagates it through nodes that do
    /// not `accepts_none_inputs`.
    None,
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Value::U64(a), Value::U64(b)) => a == b,
            (Value::I64(a), Value::I64(b)) => a == b,
            (Value::U128(a), Value::U128(b)) => a == b,
            (Value::I128(a), Value::I128(b)) => a == b,
            (Value::Reg128(a, av), Value::Reg128(b, bv)) => a == b && av == bv,
            (Value::F64(a), Value::F64(b)) => a == b,
            (Value::Bool(a), Value::Bool(b)) => a == b,
            // Arc-backed variants: pointer-eq fast path before
            // any content compare. Hot per-cycle callers
            // (notably `PolydatState::reset_inputs_from`'s
            // "still at default?" probe) typically test a slot
            // against a value that was Arc-cloned from the same
            // source — `Arc::ptr_eq` is O(1) and lets the deep
            // compare drop out of the per-cycle path.
            (Value::Str(a), Value::Str(b)) => Arc::ptr_eq(a, b) || a == b,
            (Value::Bytes(a), Value::Bytes(b)) => Arc::ptr_eq(a, b) || a == b,
            (Value::Json(a), Value::Json(b)) => Arc::ptr_eq(a, b) || a == b,
            (Value::None, Value::None) => true,
            (Value::Ext(a), Value::Ext(b)) => {
                a.type_name() == b.type_name() && a.display() == b.display()
            }
            (Value::Handle(a), Value::Handle(b)) => Arc::ptr_eq(a, b),
            (Value::VecF32(a), Value::VecF32(b)) => a == b,
            (Value::VecI32(a), Value::VecI32(b)) => a == b,
            (Value::VecF64(a), Value::VecF64(b)) => a == b,
            (Value::VecI64(a), Value::VecI64(b)) => a == b,
            (Value::VecF16(a), Value::VecF16(b)) => a == b,
            (Value::VecI16(a), Value::VecI16(b)) => a == b,
            (Value::VecI8(a), Value::VecI8(b)) => a == b,
            _ => false,
        }
    }
}

/// Trait for adapter-contributed value types.
///
/// Any type that flows through the Polydat Kernel as `Value::Ext` must
/// implement this. It provides standard access patterns that work
/// across adapter boundaries — stdout can display it, HTTP can
/// serialize it, model adapter can capture it — without needing
/// the concrete type.
///
/// The producing adapter can downcast via `as_any()` when it needs
/// native protocol access (e.g., CQL binding a `uuid::Uuid`).
pub trait ReflectedValue: Send + Sync + std::fmt::Debug {
    /// Type name for diagnostics and describe output.
    fn type_name(&self) -> &str;

    /// Human-readable string representation.
    /// Used by stdout adapter, logging, and diagnostics.
    fn display(&self) -> String;

    /// JSON representation for serialization and HTTP bodies.
    fn to_json_value(&self) -> serde_json::Value {
        serde_json::Value::String(self.display())
    }

    /// Try to represent as a string. Many types have a canonical
    /// string form (UUIDs, timestamps, IP addresses).
    fn try_as_str(&self) -> Option<String> {
        Some(self.display())
    }

    /// Try to represent as u64.
    fn try_as_u64(&self) -> Option<u64> {
        None
    }

    /// Try to represent as f64.
    fn try_as_f64(&self) -> Option<f64> {
        None
    }

    /// Try to represent as bytes.
    fn try_as_bytes(&self) -> Option<&[u8]> {
        None
    }

    /// Downcast to the concrete type. Only works when the consuming
    /// code has the concrete type in scope (same crate or shared dep).
    fn as_any(&self) -> &dyn std::any::Any;

    /// Clone into a new boxed trait object.
    fn clone_reflected(&self) -> Box<dyn ReflectedValue>;
}

impl Clone for Box<dyn ReflectedValue> {
    fn clone(&self) -> Self {
        self.clone_reflected()
    }
}

impl Value {
    /// The `U64` payload; panics on any other variant, naming both types.
    #[inline]
    pub fn as_u64(&self) -> u64 {
        match self {
            Value::U64(v) => *v,
            _ => panic!("expected U64, got {}", self.type_name()),
        }
    }

    /// Read a signed 64-bit integer. Accepts the honest `Value::I64`
    /// carrier and — during the bit-stuffed-to-honest migration —
    /// a legacy `Value::U64` whose bits are reinterpreted (the
    /// pre-alignment storage convention for `PortType::I64` slots).
    #[inline]
    pub fn as_i64(&self) -> i64 {
        match self {
            Value::I64(v) => *v,
            Value::U64(v) => *v as i64,
            _ => panic!("expected I64, got {}", self.type_name()),
        }
    }

    /// Read an unsigned 128-bit integer. Accepts the honest
    /// `Value::U128` carrier plus zero-extended `U64` (widening
    /// is implicit at read sites the way `as_i64` accepts the
    /// legacy stuffed form).
    #[inline]
    pub fn as_u128(&self) -> u128 {
        match self {
            Value::U128(b) => b.as_u128(),
            Value::U64(v) => *v as u128,
            _ => panic!("expected U128, got {}", self.type_name()),
        }
    }

    /// Read a signed 128-bit integer. Accepts `Value::I128` plus
    /// sign-extended `I64` and zero-extended `U64`.
    #[inline]
    pub fn as_i128(&self) -> i128 {
        match self {
            Value::I128(b) => b.as_i128(),
            Value::I64(v) => *v as i128,
            Value::U64(v) => *v as i128,
            _ => panic!("expected I128, got {}", self.type_name()),
        }
    }

    /// Read a 128-bit register word under any view (views are
    /// free bitcasts — a consumer declaring a different lane
    /// typing than the producer is the intended use).
    #[inline]
    pub fn as_reg_bits(&self) -> Bits128 {
        match self {
            Value::Reg128(b, _) => *b,
            _ => panic!("expected Reg128, got {}", self.type_name()),
        }
    }

    /// The `F64` payload; panics on any other variant, naming both types.
    #[inline]
    pub fn as_f64(&self) -> f64 {
        match self {
            Value::F64(v) => *v,
            _ => panic!("expected F64, got {}", self.type_name()),
        }
    }

    /// The `Bool` payload; panics on any other variant, naming both types.
    #[inline]
    pub fn as_bool(&self) -> bool {
        match self {
            Value::Bool(v) => *v,
            _ => panic!("expected Bool, got {}", self.type_name()),
        }
    }

    /// The `Str` payload as a string slice; panics on any other variant.
    #[inline]
    pub fn as_str(&self) -> &str {
        match self {
            Value::Str(v) => v,
            _ => panic!("expected Str, got {}", self.type_name()),
        }
    }

    /// The `Bytes` payload as a byte slice; panics on any other variant.
    #[inline]
    pub fn as_bytes(&self) -> &[u8] {
        match self {
            Value::Bytes(v) => v,
            _ => panic!("expected Bytes, got {}", self.type_name()),
        }
    }

    /// The `Json` payload by reference; panics on any other variant.
    #[inline]
    pub fn as_json(&self) -> &serde_json::Value {
        match self {
            Value::Json(v) => v,
            _ => panic!("expected Json, got {}", self.type_name()),
        }
    }

    /// Borrow the inner `Arc<serde_json::Value>` from a
    /// `Value::Json` variant. Use when a consumer wants to
    /// share the JSON tree across kernels without deep-cloning
    /// the structure — e.g. capture extraction that writes the
    /// same JSON wire to multiple downstream slots. Panics on
    /// type mismatch.
    #[inline]
    pub fn as_json_arc(&self) -> &Arc<serde_json::Value> {
        match self {
            Value::Json(v) => v,
            _ => panic!("expected Json, got {}", self.type_name()),
        }
    }

    /// Return the `PortType` corresponding to this value's variant.
    #[inline]
    pub fn port_type(&self) -> PortType {
        match self {
            Value::U64(_) => PortType::U64,
            Value::I64(_) => PortType::I64,
            Value::U128(_) => PortType::U128,
            Value::I128(_) => PortType::I128,
            Value::Reg128(_, v) => match v {
                RegLanes::Raw => PortType::Reg128,
                RegLanes::I8x16 => PortType::RegI8x16,
                RegLanes::I16x8 => PortType::RegI16x8,
                RegLanes::I32x4 => PortType::RegI32x4,
                RegLanes::I64x2 => PortType::RegI64x2,
                RegLanes::F16x8 => PortType::RegF16x8,
                RegLanes::F32x4 => PortType::RegF32x4,
                RegLanes::F64x2 => PortType::RegF64x2,
            },
            Value::F64(_) => PortType::F64,
            Value::Bool(_) => PortType::Bool,
            Value::Str(_) => PortType::Str,
            Value::Bytes(_) => PortType::Bytes,
            Value::Json(_) => PortType::Json,
            Value::Ext(_) => PortType::Ext,
            Value::Handle(_) => PortType::Handle,
            Value::VecF32(_) => PortType::VecF32,
            Value::VecI32(_) => PortType::VecI32,
            Value::VecF64(_) => PortType::VecF64,
            Value::VecI64(_) => PortType::VecI64,
            Value::VecF16(_) => PortType::VecF16,
            Value::VecI16(_) => PortType::VecI16,
            Value::VecI8(_) => PortType::VecI8,
            // `None` is the absence of a value, which no port type
            // names. `U64` is what this has always answered, and
            // callers that care read it through
            // [`Self::type_name`] or test for `None` first
            // ([`Self::satisfies_slot`] does).
            Value::None => PortType::U64,
        }
    }

    /// The name of this value's type, for a diagnostic.
    ///
    /// Distinct from [`Self::port_type`] in the one case that
    /// matters: an absent value reads as "none" rather than as the
    /// `u64` its port type answers. A reader told "expected Handle,
    /// got U64" goes looking for a number; the value was not there
    /// at all, which is a different fault with a different cause.
    pub fn type_name(&self) -> String {
        match self {
            Value::None => "none".to_string(),
            other => other.port_type().to_string(),
        }
    }

    /// Borrow a `VecF32` value as `&[f32]`. Panics on type mismatch.
    #[inline]
    pub fn as_vec_f32(&self) -> &[f32] {
        match self {
            Value::VecF32(arc) => arc,
            _ => panic!("expected VecF32, got {}", self.type_name()),
        }
    }

    /// Test whether this value's runtime variant is acceptable
    /// to a slot declaring `slot_type`. `port_type() == slot_type`
    /// is the strict case; this method also accepts the
    /// **bit-stuffing equivalences** documented in
    /// `polydat/docs/design/type_system.md` §1:
    ///
    /// - `Value::U64` is the runtime storage for `PortType` `U64`,
    ///   `U32`, `I64`, and `I32` (narrow integers carry their
    ///   bits in the low part of the u64; sign-extension for
    ///   `I32` is part of the producer convention).
    /// - `Value::F64` is the runtime storage for `PortType` `F64`
    ///   and `F32` (`F32` carries its bits in the low 32 via
    ///   `f32::to_bits() as u64`-style stuffing — but float
    ///   stuffing uses `Value::F64` for the materialised float
    ///   value, not the bit pattern).
    /// - `Value::None` is acceptable for every slot type
    ///   (SRD-74 absent sentinel).
    ///
    /// Used at the typed-write residual check
    /// (`Dataflow::set_wire_idx`) AFTER the boundary adapter has
    /// already converted/validated the value — see
    /// `kernel/api_impl.rs`. The pre-adapter check in
    /// `adapt_boundary_value` stays strict (`port_type ==
    /// slot_type`) so an unadapted Value::U64 can never silently
    /// truncate into a narrower slot.
    #[inline]
    pub fn satisfies_slot(&self, slot_type: PortType) -> bool {
        if matches!(self, Value::None) {
            return true;
        }
        let value_type = self.port_type();
        if value_type == slot_type {
            return true;
        }
        matches!(
            (value_type, slot_type),
            // Bit-stuffed forms: U8/U16/U32 zero-extend into U64
            // storage, the signed narrow types may still arrive as
            // U64 storage from a pre-alignment producer, and F32 and
            // F16 ride their bit patterns in U64 (`Wire for f32` and
            // `Wire for f16` inject them so).
            (PortType::U64, PortType::U32 | PortType::I64 | PortType::I32
                | PortType::U8 | PortType::U16 | PortType::I8 | PortType::I16
                | PortType::F32 | PortType::F16)
                | (PortType::F64, PortType::F32 | PortType::F16)
                // Honest signed carrier: I64 storage serves the
                // I64 slot and the sign-extended narrow signed
                // projections.
                | (PortType::I64, PortType::I32 | PortType::I8 | PortType::I16)
                // Register views are free bitcasts: a word under
                // any view satisfies a slot declaring any other
                // (the consumer's declared lane typing IS the
                // bitcast).
                | (
                    PortType::Reg128 | PortType::RegI8x16 | PortType::RegI16x8
                        | PortType::RegI32x4 | PortType::RegI64x2
                        | PortType::RegF16x8 | PortType::RegF32x4 | PortType::RegF64x2,
                    PortType::Reg128 | PortType::RegI8x16 | PortType::RegI16x8
                        | PortType::RegI32x4 | PortType::RegI64x2
                        | PortType::RegF16x8 | PortType::RegF32x4 | PortType::RegF64x2,
                )
        )
    }

    /// Borrow a `VecI32` value as `&[i32]`. Panics on type mismatch.
    #[inline]
    pub fn as_vec_i32(&self) -> &[i32] {
        match self {
            Value::VecI32(arc) => arc,
            _ => panic!("expected VecI32, got {}", self.type_name()),
        }
    }

    /// Borrow a `VecF64` value as `&[f64]`. Panics on type mismatch.
    #[inline]
    pub fn as_vec_f64(&self) -> &[f64] {
        match self {
            Value::VecF64(arc) => arc,
            _ => panic!("expected VecF64, got {}", self.type_name()),
        }
    }

    /// Borrow a `VecI64` value as `&[i64]`. Panics on type mismatch.
    #[inline]
    pub fn as_vec_i64(&self) -> &[i64] {
        match self {
            Value::VecI64(arc) => arc,
            _ => panic!("expected VecI64, got {}", self.type_name()),
        }
    }

    /// Borrow a `VecF16` value as `&[half::f16]`. Panics on type mismatch.
    #[inline]
    pub fn as_vec_f16(&self) -> &[half::f16] {
        match self {
            Value::VecF16(arc) => arc,
            _ => panic!("expected VecF16, got {}", self.type_name()),
        }
    }

    /// Borrow a `VecI16` value as `&[i16]`. Panics on type mismatch.
    #[inline]
    pub fn as_vec_i16(&self) -> &[i16] {
        match self {
            Value::VecI16(arc) => arc,
            _ => panic!("expected VecI16, got {}", self.type_name()),
        }
    }

    /// Borrow a `VecI8` value as `&[i8]`. Panics on type mismatch.
    #[inline]
    pub fn as_vec_i8(&self) -> &[i8] {
        match self {
            Value::VecI8(arc) => arc,
            _ => panic!("expected VecI8, got {}", self.type_name()),
        }
    }

    /// Downcast a Handle value to a borrowed reference of its concrete
    /// type. Panics if the variant isn't `Handle` or the type doesn't
    /// match. Used by reader nodes that consume a typed-handle wire
    /// produced by a resolver node (see SRD 53 §"Dataset Handles").
    ///
    /// The borrow lasts as long as `self` (the buffer slot's `Value`
    /// is what holds the `Arc`). For per-cycle reads this is the
    /// expected pattern — call methods on the borrowed dataset, then
    /// return.
    #[inline]
    pub fn as_handle<T: std::any::Any + Send + Sync>(&self) -> &T {
        match self {
            Value::Handle(arc) => arc.downcast_ref::<T>().unwrap_or_else(|| {
                panic!(
                    "Handle downcast failed: expected {}",
                    std::any::type_name::<T>()
                )
            }),
            _ => panic!("expected Handle, got {}", self.type_name()),
        }
    }

    /// Construct a `Value::Handle` from a typed `Arc<T>`. Convenience
    /// wrapper that performs the type-erasure to `Arc<dyn Any + Send + Sync>`.
    pub fn handle<T: std::any::Any + Send + Sync>(arc: Arc<T>) -> Self {
        Value::Handle(arc as Arc<dyn std::any::Any + Send + Sync>)
    }

    /// Best-effort string representation for any value.
    /// Works across all variants including Ext.
    pub fn to_display_string(&self) -> String {
        match self {
            Value::U64(v) => v.to_string(),
            Value::I64(v) => v.to_string(),
            Value::U128(b) => b.as_u128().to_string(),
            Value::I128(b) => b.as_i128().to_string(),
            // Lane-typed register views render like the Vec*
            // display forms; the raw view renders as 32 hex
            // digits (the full word as buffer state).
            Value::Reg128(b, view) => match view {
                RegLanes::Raw => format!("{:032x}", b.as_u128()),
                RegLanes::I8x16 => format!("{:?}", b.lanes_i8()),
                RegLanes::I16x8 => format!("{:?}", b.lanes_i16()),
                RegLanes::I32x4 => format!("{:?}", b.lanes_i32()),
                RegLanes::I64x2 => format!("{:?}", b.lanes_i64()),
                RegLanes::F16x8 => format!("{:?}", b.lanes_f16().map(|f| f.to_f32())),
                RegLanes::F32x4 => format!("{:?}", b.lanes_f32()),
                RegLanes::F64x2 => format!("{:?}", b.lanes_f64()),
            },
            // `{v:?}` (Rust Debug) for f64 always includes at
            // least one fractional digit, so whole-number floats
            // render as `1.0` instead of `1` — distinguishing
            // them from integers in CQL OPTIONS strings, plot
            // labels, and other surfaces where the type matters.
            // Display-formatted (`v.to_string()`) strips the
            // trailing zero, conflating ints with whole-number
            // floats. Both forms produce identical output for
            // non-whole floats (`1.5 → "1.5"`).
            Value::F64(v) => format!("{v:?}"),
            Value::Bool(v) => v.to_string(),
            Value::Str(v) => v.to_string(),
            Value::Bytes(v) => v.iter().map(|b| format!("{b:02x}")).collect(),
            Value::Json(v) => v.to_string(),
            Value::Ext(v) => v.display(),
            Value::Handle(arc) => format!("<handle:{:?}>", arc.type_id()),
            Value::VecF32(arc) => {
                // JSON-array text. Per-element format-write into a
                // pre-sized String avoids the intermediate Vec<String>.
                // Debug formatter (`{v:?}`) matches the F64 element
                // rule above: whole-number floats render as `1.0`
                // so VecF32 stays distinguishable from VecI32 at the
                // display surface.
                let mut s = String::with_capacity(arc.len() * 8 + 2);
                s.push('[');
                let mut first = true;
                for v in arc.iter() {
                    if !first {
                        s.push(',');
                    }
                    first = false;
                    use std::fmt::Write;
                    let _ = write!(&mut s, "{v:?}");
                }
                s.push(']');
                s
            }
            Value::VecI32(arc) => {
                let mut s = String::with_capacity(arc.len() * 4 + 2);
                s.push('[');
                let mut first = true;
                for v in arc.iter() {
                    if !first {
                        s.push(',');
                    }
                    first = false;
                    use std::fmt::Write;
                    let _ = write!(&mut s, "{v}");
                }
                s.push(']');
                s
            }
            Value::VecF64(arc) => {
                let mut s = String::with_capacity(arc.len() * 8 + 2);
                s.push('[');
                let mut first = true;
                for v in arc.iter() {
                    if !first {
                        s.push(',');
                    }
                    first = false;
                    use std::fmt::Write;
                    let _ = write!(&mut s, "{v:?}");
                }
                s.push(']');
                s
            }
            Value::VecI64(arc) => {
                let mut s = String::with_capacity(arc.len() * 4 + 2);
                s.push('[');
                let mut first = true;
                for v in arc.iter() {
                    if !first {
                        s.push(',');
                    }
                    first = false;
                    use std::fmt::Write;
                    let _ = write!(&mut s, "{v}");
                }
                s.push(']');
                s
            }
            Value::VecF16(arc) => {
                let mut s = String::with_capacity(arc.len() * 6 + 2);
                s.push('[');
                let mut first = true;
                for v in arc.iter() {
                    if !first {
                        s.push(',');
                    }
                    first = false;
                    use std::fmt::Write;
                    // Render as the f32 widening so the JSON form
                    // is the standard "1.0" / "1.5" surface — f16
                    // Display has its own form but it isn't valid
                    // JSON, so widening makes the array shape
                    // parseable downstream.
                    let _ = write!(&mut s, "{:?}", v.to_f32());
                }
                s.push(']');
                s
            }
            Value::VecI16(arc) => {
                let mut s = String::with_capacity(arc.len() * 4 + 2);
                s.push('[');
                let mut first = true;
                for v in arc.iter() {
                    if !first {
                        s.push(',');
                    }
                    first = false;
                    use std::fmt::Write;
                    let _ = write!(&mut s, "{v}");
                }
                s.push(']');
                s
            }
            Value::VecI8(arc) => {
                let mut s = String::with_capacity(arc.len() * 4 + 2);
                s.push('[');
                let mut first = true;
                for v in arc.iter() {
                    if !first {
                        s.push(',');
                    }
                    first = false;
                    use std::fmt::Write;
                    let _ = write!(&mut s, "{v}");
                }
                s.push(']');
                s
            }
            Value::None => String::new(),
        }
    }

    /// Strict-render variant of [`Self::to_display_string`] for use
    /// at wire-protocol render sites (op-template substitution,
    /// adapter byte-emission paths).
    ///
    /// Returns `None` for [`Value::None`] instead of converting it
    /// to `""`. The empty-string mapping in `to_display_string` is
    /// convenient for diagnostic / log contexts but lethal at the
    /// wire boundary — it silently coerces "absent" into "present
    /// but empty," corrupting downstream bytes (e.g. sending
    /// `'source_model': ''` to a CQL cluster when the intended
    /// shadow didn't bind). Render paths use this primitive and
    /// surface a clear error when an unresolved bind-point reaches
    /// them. See `crates/polydat/docs/design/none_semantics.md`
    /// (the render-refuses-silent-None rule).
    pub fn to_display_strict(&self) -> Option<String> {
        match self {
            Value::None => None,
            other => Some(other.to_display_string()),
        }
    }

    /// JSON representation for any value. Works across all variants.
    pub fn to_json_value(&self) -> serde_json::Value {
        match self {
            Value::U64(v) => serde_json::Value::from(*v),
            Value::I64(v) => serde_json::Value::from(*v),
            // JSON Number is bounded by u64/i64/f64 leaves
            // (serde_json without arbitrary_precision); 128-bit
            // magnitudes project as decimal strings, the same
            // string-convention family as Bytes-as-hex.
            Value::U128(b) => serde_json::Value::String(b.as_u128().to_string()),
            Value::I128(b) => serde_json::Value::String(b.as_i128().to_string()),
            // Lane-typed views project as homogeneous arrays
            // (same shape as the matching Vec*); the raw view as
            // a hex string (lane roles are algorithm-defined, so
            // no numeric reading exists).
            Value::Reg128(b, view) => match view {
                RegLanes::Raw => serde_json::Value::String(format!("{:032x}", b.as_u128())),
                RegLanes::I8x16 => serde_json::Value::Array(
                    b.lanes_i8()
                        .iter()
                        .map(|i| serde_json::Value::from(*i as i32))
                        .collect(),
                ),
                RegLanes::I16x8 => serde_json::Value::Array(
                    b.lanes_i16()
                        .iter()
                        .map(|i| serde_json::Value::from(*i as i32))
                        .collect(),
                ),
                RegLanes::I32x4 => serde_json::Value::Array(
                    b.lanes_i32()
                        .iter()
                        .map(|i| serde_json::Value::from(*i))
                        .collect(),
                ),
                RegLanes::I64x2 => serde_json::Value::Array(
                    b.lanes_i64()
                        .iter()
                        .map(|i| serde_json::Value::from(*i))
                        .collect(),
                ),
                RegLanes::F16x8 => serde_json::Value::Array(
                    b.lanes_f16()
                        .iter()
                        .map(|f| serde_json::json!(f.to_f32()))
                        .collect(),
                ),
                RegLanes::F32x4 => serde_json::Value::Array(
                    b.lanes_f32()
                        .iter()
                        .map(|f| serde_json::json!(*f))
                        .collect(),
                ),
                RegLanes::F64x2 => serde_json::Value::Array(
                    b.lanes_f64()
                        .iter()
                        .map(|f| serde_json::json!(*f))
                        .collect(),
                ),
            },
            Value::F64(v) => serde_json::json!(*v),
            Value::Bool(v) => serde_json::Value::from(*v),
            Value::Str(v) => serde_json::Value::from(&**v),
            Value::Bytes(v) => {
                serde_json::Value::from(v.iter().map(|b| format!("{b:02x}")).collect::<String>())
            }
            Value::Json(v) => (**v).clone(),
            Value::Ext(v) => v.to_json_value(),
            Value::Handle(_) => serde_json::Value::Null,
            Value::VecF32(arc) => {
                serde_json::Value::Array(arc.iter().map(|f| serde_json::json!(*f)).collect())
            }
            Value::VecI32(arc) => {
                serde_json::Value::Array(arc.iter().map(|i| serde_json::Value::from(*i)).collect())
            }
            Value::VecF64(arc) => {
                serde_json::Value::Array(arc.iter().map(|f| serde_json::json!(*f)).collect())
            }
            Value::VecI64(arc) => {
                serde_json::Value::Array(arc.iter().map(|i| serde_json::Value::from(*i)).collect())
            }
            Value::VecF16(arc) => serde_json::Value::Array(
                arc.iter().map(|f| serde_json::json!(f.to_f32())).collect(),
            ),
            Value::VecI16(arc) => serde_json::Value::Array(
                arc.iter()
                    .map(|i| serde_json::Value::from(*i as i32))
                    .collect(),
            ),
            Value::VecI8(arc) => serde_json::Value::Array(
                arc.iter()
                    .map(|i| serde_json::Value::from(*i as i32))
                    .collect(),
            ),
            Value::None => serde_json::Value::Null,
        }
    }
}

pub use polydat_grammar::{NumericDomain, PortType};

/// What a port type means to a compiled buffer: its slot color, the
/// width that follows from it, and the scratch element a by-reference
/// producer owns. The type itself is the grammar's
/// (`polydat_grammar::PortType`); these are the runtime's reading of
/// it, and every layout, codegen, and guard decision derives from
/// them.
pub trait SlotShape {
    /// Slot color in compiled (P2/P3/hybrid) kernel buffers —
    /// axiom S1 (`jit_boundary.md` §"Slot-state axioms"). The
    /// single chokepoint: width and every layout/codegen/guard
    /// decision derive from this, never restate it.
    fn slot_color(&self) -> SlotColor;
    /// The scratch element a `Ref2`-colored port's producer owns
    /// (axiom S3); `None` for an immediate color.
    fn scratch_elem(&self) -> Option<ScratchElem>;
    /// Buffer slots this type occupies — derived from
    /// [`Self::slot_color`] per axiom S1.
    fn slot_width(&self) -> usize;
}

impl SlotShape for PortType {
    #[inline]
    fn slot_color(&self) -> SlotColor {
        match self {
            // 128-bit immediates: two slots of limb DATA —
            // register words and 128-bit integers are values,
            // never addresses.
            Self::U128
            | Self::I128
            | Self::Reg128
            | Self::RegI8x16
            | Self::RegI16x8
            | Self::RegI32x4
            | Self::RegI64x2
            | Self::RegF16x8
            | Self::RegF32x4
            | Self::RegF64x2 => SlotColor::Imm2,
            // Heap slices: a (ptr, len) reference pair viewing
            // kernel-owned scratch (§8.4 layer 3). A string and a
            // byte string are slices of bytes; a JSON, extension, or
            // handle value is a one-element slice holding the value.
            Self::VecF32
            | Self::VecI32
            | Self::VecF64
            | Self::VecI64
            | Self::VecF16
            | Self::VecI16
            | Self::VecI8
            | Self::Str
            | Self::Bytes
            | Self::Json
            | Self::Ext
            | Self::Handle => SlotColor::Ref2,
            // Everything else (incl. all narrow widths riding
            // their 64-bit carriers): one slot of immediate data.
            _ => SlotColor::Imm1,
        }
    }

    #[inline]
    fn scratch_elem(&self) -> Option<ScratchElem> {
        Some(match self {
            Self::VecF32 => ScratchElem::F32,
            Self::VecF64 => ScratchElem::F64,
            Self::VecF16 => ScratchElem::F16,
            Self::VecI8 => ScratchElem::I8,
            Self::VecI16 => ScratchElem::I16,
            Self::VecI32 => ScratchElem::I32,
            Self::VecI64 => ScratchElem::I64,
            Self::Str => ScratchElem::Str,
            Self::Bytes => ScratchElem::Bytes,
            Self::Json | Self::Ext | Self::Handle => ScratchElem::Value,
            _ => return None,
        })
    }

    #[inline]
    fn slot_width(&self) -> usize {
        match self.slot_color() {
            SlotColor::Imm1 => 1,
            SlotColor::Imm2 | SlotColor::Ref2 => 2,
        }
    }
}

/// The lifecycle of a port's value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lifecycle {
    /// Cycle-time: value changes per evaluation.
    Cycle,
    /// Init-time: value is frozen at assembly, immutable at runtime.
    /// Wiring a cycle-time value to an init port is an assembly error.
    Init,
}

/// Cost class for an input wire, indicating how expensive it is
/// to change the value on this port.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WireCost {
    /// Data wire: cheap per-cycle input. The node's primary
    /// computation path. Default for most ports.
    #[default]
    Data,
    /// Config wire: changing this input invalidates expensive
    /// internal state (LUT, distribution table). Expected to be
    /// wired to init-time constants or rarely-changing values.
    /// The compiler warns when a config wire connects to a
    /// cycle-time binding.
    Config,
}

/// Descriptor for a single input or output port on a node.
#[derive(Debug, Clone)]
pub struct Port {
    /// The port's name, as bindings and diagnostics refer to it.
    pub name: String,
    /// The port's declared type.
    pub typ: PortType,
    /// When the port's value changes: per cycle, at init, or as configuration.
    pub lifecycle: Lifecycle,
    /// Cost class for input ports. Ignored for output ports.
    pub wire_cost: WireCost,
    /// Optional value contract this wire must satisfy at runtime
    /// (SRD 15 §"Strict Wire Mode"). The compiler uses this to
    /// decide whether to auto-insert a value assertion when the
    /// upstream source can't statically be proven to deliver a
    /// satisfying value. `None` = no constraint declared.
    ///
    /// Constraints reuse the same vocabulary as
    /// [`crate::dsl::const_constraints::ConstConstraint`] — the
    /// difference is just where the value comes from (a literal
    /// for `ConstU64`, a wire for `Slot::Wire`).
    pub constraint: Option<crate::dsl::const_constraints::ConstConstraint>,
    /// Whether this port takes the wire's value as it is, whatever
    /// type the wire carries — in which case [`Self::typ`] is a
    /// nominal placeholder and the assembler inserts no adapter into
    /// this port.
    ///
    /// The one shape that needs it is an element of a `&[Value]`
    /// variadic: the node inspects the `Value` variant itself, so
    /// converting the wire to the port's declared type would change
    /// what the node sees — `json_array(cycle)` would hold the text
    /// of a number rather than the number. A plain `Value` argument
    /// does not need it, because the assembler resolves that port's
    /// type from its wire and hands it to the constructor.
    ///
    /// The assembler used to decide this from a list of thirteen node
    /// names, which was both a name-keyed table and the wrong
    /// granularity: `pick`'s selector wires must be `Bool` while its
    /// value wires are polymorphic, and one flag per node cannot say
    /// that.
    pub accepts_any_type: bool,
}

impl Port {
    /// A cycle-lifecycle port of the given type with no constraint.
    pub fn new(name: impl Into<String>, typ: PortType) -> Self {
        Self {
            name: name.into(),
            typ,
            lifecycle: Lifecycle::Cycle,
            wire_cost: WireCost::Data,
            constraint: None,
            accepts_any_type: false,
        }
    }

    /// This port, taking the wire as it is whatever its type. See
    /// [`Self::accepts_any_type`].
    pub fn any_type(mut self) -> Self {
        self.accepts_any_type = true;
        self
    }

    /// Create a port with explicit lifecycle.
    pub fn with_lifecycle(name: impl Into<String>, typ: PortType, lifecycle: Lifecycle) -> Self {
        Self {
            name: name.into(),
            typ,
            lifecycle,
            wire_cost: WireCost::Data,
            constraint: None,
            accepts_any_type: false,
        }
    }

    /// A `u64` port.
    pub fn u64(name: impl Into<String>) -> Self {
        Self::new(name, PortType::U64)
    }

    /// An `f64` port.
    pub fn f64(name: impl Into<String>) -> Self {
        Self::new(name, PortType::F64)
    }

    /// A string port.
    pub fn str(name: impl Into<String>) -> Self {
        Self::new(name, PortType::Str)
    }

    /// A boolean port.
    pub fn bool(name: impl Into<String>) -> Self {
        Self::new(name, PortType::Bool)
    }

    /// A JSON port.
    pub fn json(name: impl Into<String>) -> Self {
        Self::new(name, PortType::Json)
    }

    /// A handle port.
    pub fn handle(name: impl Into<String>) -> Self {
        Self::new(name, PortType::Handle)
    }

    /// An `f32` vector port.
    pub fn vec_f32(name: impl Into<String>) -> Self {
        Self::new(name, PortType::VecF32)
    }

    /// An `i32` vector port.
    pub fn vec_i32(name: impl Into<String>) -> Self {
        Self::new(name, PortType::VecI32)
    }

    /// Create an init-time port (frozen at assembly).
    pub fn init(name: impl Into<String>, typ: PortType) -> Self {
        Self::with_lifecycle(name, typ, Lifecycle::Init)
    }

    /// Attach a value constraint. Used by node authors that want
    /// to declare "this wire must satisfy X" so strict-wire-mode
    /// can auto-insert the right value assertion. See SRD 15
    /// §"Strict Wire Mode".
    pub fn with_constraint(mut self, c: crate::dsl::const_constraints::ConstConstraint) -> Self {
        self.constraint = Some(c);
        self
    }

    /// Mark this port as a config wire (expensive to change).
    pub fn config(mut self) -> Self {
        self.wire_cost = WireCost::Config;
        self
    }

    /// Set the wire cost directly. Used by the macro to thread
    /// `Wire::WIRE_COST` from the trait through to the slot.
    pub fn with_cost(mut self, cost: WireCost) -> Self {
        self.wire_cost = cost;
        self
    }
}

// ---------------------------------------------------------------------------
// Unified slot model (SRD 36 §Variadic)
// ---------------------------------------------------------------------------

/// The type discriminant for a slot: wire or typed constant.
///
/// This is the shared vocabulary between `FuncSig` (static registry)
/// and `NodeMeta` (owned instance). It replaces the former `ParamKind`,
/// `ConstType`, and `SlotKind` enums with a single type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SlotType {
    /// A runtime wire input carrying a value each cycle.
    Wire,
    /// A u64 constant literal.
    ConstU64,
    /// An f64 constant literal.
    ConstF64,
    /// A string constant literal.
    ConstStr,
    /// A `Vec<u64>` constant (from array literal).
    ConstVecU64,
    /// A `Vec<f64>` constant (from array literal).
    ConstVecF64,
    /// SRD-80b Phase C — typed-element variadic-const slot for
    /// `Const<Vec<C>>` operator-side shape. Element type
    /// discrimination is emitted inline by the macro at the
    /// build-closure call site, from the element type it read out of
    /// the signature; the slot tag only signals "this is a list" to
    /// the DSL type-checker.
    ConstVec,
}

impl SlotType {
    /// Whether this is a constant (not a wire).
    pub fn is_const(self) -> bool {
        !matches!(self, SlotType::Wire)
    }

    /// Whether this is a wire (not a constant).
    pub fn is_wire(self) -> bool {
        matches!(self, SlotType::Wire)
    }
}

/// A concrete constant value stored in node metadata.
///
/// Assembly-time values baked into the node at construction. The
/// variant determines the `SlotType` — no separate type discriminant
/// is needed.
#[derive(Debug, Clone, PartialEq)]
pub enum ConstValue {
    /// An unsigned integer.
    U64(u64),
    /// A floating-point number.
    F64(f64),
    /// A string.
    Str(String),
    /// A list of unsigned integers.
    VecU64(Vec<u64>),
    /// A list of floating-point numbers.
    VecF64(Vec<f64>),
}

impl ConstValue {
    /// Return the `SlotType` for this value.
    pub fn slot_type(&self) -> SlotType {
        match self {
            ConstValue::U64(_) => SlotType::ConstU64,
            ConstValue::F64(_) => SlotType::ConstF64,
            ConstValue::Str(_) => SlotType::ConstStr,
            ConstValue::VecU64(_) => SlotType::ConstVecU64,
            ConstValue::VecF64(_) => SlotType::ConstVecF64,
        }
    }

    /// Encode to the JIT's u64 representation.
    pub fn to_jit_u64s(&self) -> Vec<u64> {
        match self {
            ConstValue::U64(v) => vec![*v],
            ConstValue::F64(v) => vec![v.to_bits()],
            ConstValue::Str(_) => vec![],
            ConstValue::VecU64(v) => v.clone(),
            ConstValue::VecF64(v) => v.iter().map(|f| f.to_bits()).collect(),
        }
    }
}

/// A single logical input to a node: either a runtime wire or an
/// assembly-time constant. The positional order in `NodeMeta.slots`
/// matches the function call syntax in the DSL.
#[derive(Debug, Clone)]
pub enum Slot {
    /// A runtime wire input carrying a value each cycle.
    Wire(Port),
    /// An assembly-time constant, baked into the node at construction.
    Const {
        /// The constant's name, as the node's signature calls it.
        name: String,
        /// The baked value.
        value: ConstValue,
    },
}

impl Slot {
    /// Return the `SlotType` discriminant for this slot.
    pub fn slot_type(&self) -> SlotType {
        match self {
            Slot::Wire(_) => SlotType::Wire,
            Slot::Const { value, .. } => value.slot_type(),
        }
    }

    /// Create a wire slot.
    pub fn wire(port: Port) -> Self {
        Slot::Wire(port)
    }

    /// Create a u64 constant slot.
    pub fn const_u64(name: impl Into<String>, v: u64) -> Self {
        Slot::Const {
            name: name.into(),
            value: ConstValue::U64(v),
        }
    }

    /// Create an f64 constant slot.
    pub fn const_f64(name: impl Into<String>, v: f64) -> Self {
        Slot::Const {
            name: name.into(),
            value: ConstValue::F64(v),
        }
    }

    /// Create a string constant slot.
    pub fn const_str(name: impl Into<String>, v: impl Into<String>) -> Self {
        Slot::Const {
            name: name.into(),
            value: ConstValue::Str(v.into()),
        }
    }

    /// Create a `Vec<u64>` constant slot.
    pub fn const_vec_u64(name: impl Into<String>, v: Vec<u64>) -> Self {
        Slot::Const {
            name: name.into(),
            value: ConstValue::VecU64(v),
        }
    }

    /// Create a `Vec<f64>` constant slot.
    pub fn const_vec_f64(name: impl Into<String>, v: Vec<f64>) -> Self {
        Slot::Const {
            name: name.into(),
            value: ConstValue::VecF64(v),
        }
    }
}

/// Declares which inputs of a node are interchangeable.
///
/// Used by the fusion pattern matcher to recognize equivalent
/// subgraphs regardless of operand order, and by future passes
/// (e.g., canonical ordering, common subexpression elimination).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum Commutativity {
    /// Input order matters. No permutations attempted during
    /// pattern matching. This is the default for unary nodes and
    /// any node where operand order affects the result.
    ///
    /// Examples: `mod(dividend, divisor)`, `div(x, K)`,
    /// `concat(left, right)`, `sub(a, b)`.
    #[default]
    Positional,

    /// All inputs are interchangeable, including variadic.
    /// For small arity (2-3), the matcher tries all permutations.
    /// For larger arity, it uses set-matching.
    ///
    /// Examples: `sum(a, b, ..., n)`, `product(a, b, ..., n)`,
    /// `min(a, b, ..., n)`, `max(a, b, ..., n)`.
    AllCommutative,

    /// Specific groups of input port indices are interchangeable
    /// within each group. Inputs not listed in any group are
    /// positional.
    ///
    /// Example: `fma(x, y, z) = x + y * z`
    /// The multiplicands `y` (index 1) and `z` (index 2) commute,
    /// but the addend `x` (index 0) does not.
    /// `Groups(vec![vec![1, 2]])`
    Groups(Vec<Vec<usize>>),
}

/// Metadata describing a node's interface: its input slots and output ports.
///
/// Generated per-node-type and queryable at runtime for assembly-time
/// validation, compilation, optimization passes, and describe output.
///
/// Wire inputs are `Slot::Wire(Port)`. Constants are `Slot::Const { name, value }`.
/// Use `wire_inputs()` to extract just the wire ports.
#[derive(Debug, Clone)]
pub struct NodeMeta {
    /// The node's function name, as programs call it.
    pub name: String,
    /// All inputs in positional order: wires and constants.
    pub ins: Vec<Slot>,
    /// The output ports, in positional order.
    pub outs: Vec<Port>,
}

impl NodeMeta {
    /// Wire-only input ports extracted from `ins`.
    pub fn wire_inputs(&self) -> Vec<&Port> {
        self.ins
            .iter()
            .filter_map(|s| match s {
                Slot::Wire(p) => Some(p),
                Slot::Const { .. } => None,
            })
            .collect()
    }

    /// Constant names and values extracted from `ins`.
    pub fn const_slots(&self) -> Vec<(&str, &ConstValue)> {
        self.ins
            .iter()
            .filter_map(|s| match s {
                Slot::Const { name, value } => Some((name.as_str(), value)),
                Slot::Wire(_) => None,
            })
            .collect()
    }

    /// Encode all constants from `ins` to JIT u64 representation.
    pub fn jit_constants_from_slots(&self) -> Vec<u64> {
        self.const_slots()
            .iter()
            .flat_map(|(_, v)| v.to_jit_u64s())
            .collect()
    }
}

/// A compiled u64-only evaluation step.
///
/// The closure captures all assembly-time parameters. At runtime it
/// reads from input slots and writes to output slots in a flat `[u64]`
/// buffer — no `Value` enum, no virtual dispatch.
pub type CompiledU64Op = Box<dyn Fn(&[u64], &mut [u64]) + Send + Sync>;

/// Element type of one kernel-owned scratch buffer
/// (type_system_alignment.md §8.4 layer 3). One entry per
/// `Ref2`-colored output port of a slot-compiled node: a typed
/// vector, a string, a byte string, or a value held by reference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScratchElem {
    /// `f32` elements.
    F32,
    /// `f64` elements.
    F64,
    /// `f16` elements.
    F16,
    /// `i8` elements.
    I8,
    /// `i16` elements.
    I16,
    /// `i32` elements.
    I32,
    /// `i64` elements.
    I64,
    /// The UTF-8 bytes of a string.
    Str,
    /// The bytes of a byte string.
    Bytes,
    /// One value held by reference (`Json`, `Ext`, `Handle`): the
    /// pair is `(&Value, 1)`.
    Value,
    /// A buffer of 64-bit slots: a native cone's own slot buffer,
    /// owned by the state that evaluates it.
    Slots,
    /// The kernels a tile render keeps over its projection bodies,
    /// owned by the state that renders.
    Kernels,
    /// State a node defines for itself per evaluating kernel state, a
    /// memo of what it last derived from its inputs, created by the
    /// node on first use; a clone starts empty.
    State,
}

/// Node-defined state held by a kernel state (`ScratchElem::State`):
/// what a node keeps between its evaluations in one state, typed by
/// the node and never shared between states. Empty until the node
/// first fills it; a clone is empty, since a clone of a state is a
/// new state (compiled_handles.md §3).
#[derive(Default)]
pub struct NodeState(Option<Box<dyn std::any::Any + Send + Sync>>);

impl NodeState {
    /// The state as `T`, created by `init` when the entry is empty or
    /// holds another type.
    pub fn get_or_insert_with<T: std::any::Any + Send + Sync>(
        &mut self,
        init: impl FnOnce() -> T,
    ) -> &mut T {
        if !self.0.as_ref().is_some_and(|b| b.is::<T>()) {
            self.0 = Some(Box::new(init()));
        }
        self.0
            .as_mut()
            .and_then(|b| b.downcast_mut::<T>())
            .expect("the entry holds a T")
    }

    /// The state as `T`, if the node has filled it with one.
    pub fn get<T: std::any::Any + Send + Sync>(&self) -> Option<&T> {
        self.0.as_ref().and_then(|b| b.downcast_ref::<T>())
    }
}

impl Clone for NodeState {
    fn clone(&self) -> Self {
        NodeState(None)
    }
}

impl std::fmt::Debug for NodeState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "NodeState({})",
            if self.0.is_some() { "filled" } else { "empty" }
        )
    }
}

/// Slot color of a `PortType` in compiled kernel buffers —
/// axiom S1: static, total, three-valued. `Imm*` slots carry
/// immediate data only (never addresses); `Ref2` pairs carry a
/// `(ptr, len)` reference to storage with a proven owner: the
/// step's own scratch, an extern's stored value, an interned
/// constant, or a boundary value alive for the call. They are
/// engine-internal per axiom S2.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SlotColor {
    /// One slot of immediate data.
    Imm1,
    /// Two slots of immediate limb data (128-bit values).
    Imm2,
    /// Two slots holding a (ptr, len) reference pair.
    Ref2,
}

/// One kernel-owned scratch buffer. A `Ref2` output port's
/// `(ptr, len)` buffer slots view its scratch — the kernel owns
/// the allocation, so the pointer is valid exactly as long as the
/// producing step doesn't rerun (and a rerun rewrites the slots
/// before any consumer reads them). No Arc traffic, no allocation
/// after warmup: a string or byte string is rewritten in place, a
/// value is replaced.
#[derive(Debug, Clone)]
pub enum ScratchBuf {
    /// An `f32` buffer.
    F32(Vec<f32>),
    /// An `f64` buffer.
    F64(Vec<f64>),
    /// An `f16` buffer.
    F16(Vec<half::f16>),
    /// An `i8` buffer.
    I8(Vec<i8>),
    /// An `i16` buffer.
    I16(Vec<i16>),
    /// An `i32` buffer.
    I32(Vec<i32>),
    /// An `i64` buffer.
    I64(Vec<i64>),
    /// The UTF-8 bytes of a string.
    Str(Vec<u8>),
    /// The bytes of a byte string.
    Bytes(Vec<u8>),
    /// One value held by reference; empty until the step first runs.
    Value(Vec<Value>),
    /// A buffer of 64-bit slots (a native cone's own).
    Slots(Vec<u64>),
    /// The kernels a tile render keeps over its projection bodies. A
    /// clone is empty: a new state builds its own.
    Kernels(crate::library::tile_render::BodyKernels),
    /// State a node defines for itself, per kernel state. A clone is
    /// empty: a new state derives its own.
    State(NodeState),
}

impl ScratchBuf {
    /// The `(ptr, len)` pair this entry currently publishes —
    /// the ground truth axiom S9(a)'s validator compares buffer
    /// slots against.
    pub fn ptr_len(&self) -> (u64, u64) {
        match self {
            ScratchBuf::F32(v) => (v.as_ptr() as usize as u64, v.len() as u64),
            ScratchBuf::F64(v) => (v.as_ptr() as usize as u64, v.len() as u64),
            ScratchBuf::F16(v) => (v.as_ptr() as usize as u64, v.len() as u64),
            ScratchBuf::I8(v) => (v.as_ptr() as usize as u64, v.len() as u64),
            ScratchBuf::I16(v) => (v.as_ptr() as usize as u64, v.len() as u64),
            ScratchBuf::I32(v) => (v.as_ptr() as usize as u64, v.len() as u64),
            ScratchBuf::I64(v) => (v.as_ptr() as usize as u64, v.len() as u64),
            ScratchBuf::Str(v) | ScratchBuf::Bytes(v) => {
                (v.as_ptr() as usize as u64, v.len() as u64)
            }
            ScratchBuf::Value(v) => (v.as_ptr() as usize as u64, v.len() as u64),
            ScratchBuf::Slots(v) => (v.as_ptr() as usize as u64, v.len() as u64),
            ScratchBuf::Kernels(_) | ScratchBuf::State(_) => (0, 0),
        }
    }

    /// What this entry holds as an owned `Value`, copied out: the
    /// typed read of a `Ref2` output on a compiled kernel, which is
    /// what the interpreter's `pull` returns for the same port. A
    /// value entry that has not been written reads as `None`.
    pub fn to_value(&self) -> Value {
        match self {
            ScratchBuf::F32(v) => Value::VecF32(SliceArc::from_vec(v.clone())),
            ScratchBuf::F64(v) => Value::VecF64(SliceArc::from_vec(v.clone())),
            ScratchBuf::F16(v) => Value::VecF16(SliceArc::from_vec(v.clone())),
            ScratchBuf::I8(v) => Value::VecI8(SliceArc::from_vec(v.clone())),
            ScratchBuf::I16(v) => Value::VecI16(SliceArc::from_vec(v.clone())),
            ScratchBuf::I32(v) => Value::VecI32(SliceArc::from_vec(v.clone())),
            ScratchBuf::I64(v) => Value::VecI64(SliceArc::from_vec(v.clone())),
            // SAFETY: a `Str` entry is written only from `&str` bytes.
            ScratchBuf::Str(v) => {
                Value::Str(Arc::from(unsafe { std::str::from_utf8_unchecked(v) }))
            }
            ScratchBuf::Bytes(v) => Value::Bytes(Arc::from(&v[..])),
            ScratchBuf::Value(v) => v.first().cloned().unwrap_or(Value::None),
            ScratchBuf::Slots(_) => panic!("a slot buffer is not a value"),
            ScratchBuf::Kernels(_) => panic!("a body kernel set is not a value"),
            ScratchBuf::State(_) => panic!("a node's own state is not a value"),
        }
    }

    /// The node-defined state this entry holds. The entry must be a
    /// `State` entry.
    pub fn node_state(&mut self) -> &mut NodeState {
        match self {
            ScratchBuf::State(s) => s,
            other => panic!("scratch entry holds {other:?}, not a node's state"),
        }
    }

    /// Replace the string this entry holds, reusing its allocation.
    /// The entry must be a `Str` entry.
    #[inline]
    pub fn set_str(&mut self, s: &str) {
        match self {
            ScratchBuf::Str(v) => {
                v.clear();
                v.extend_from_slice(s.as_bytes());
            }
            other => panic!("scratch entry holds {other:?}, not a string"),
        }
    }

    /// Replace the byte string this entry holds, reusing its
    /// allocation. The entry must be a `Bytes` entry.
    #[inline]
    pub fn set_bytes(&mut self, b: &[u8]) {
        match self {
            ScratchBuf::Bytes(v) => {
                v.clear();
                v.extend_from_slice(b);
            }
            other => panic!("scratch entry holds {other:?}, not a byte string"),
        }
    }

    /// Replace the value this entry holds. The entry must be a
    /// `Value` entry.
    #[inline]
    pub fn set_value(&mut self, value: Value) {
        match self {
            ScratchBuf::Value(v) => {
                v.clear();
                v.push(value);
            }
            other => panic!("scratch entry holds {other:?}, not a value"),
        }
    }

    /// An empty buffer of the element type.
    pub fn new(elem: ScratchElem) -> Self {
        match elem {
            ScratchElem::F32 => ScratchBuf::F32(Vec::new()),
            ScratchElem::F64 => ScratchBuf::F64(Vec::new()),
            ScratchElem::F16 => ScratchBuf::F16(Vec::new()),
            ScratchElem::I8 => ScratchBuf::I8(Vec::new()),
            ScratchElem::I16 => ScratchBuf::I16(Vec::new()),
            ScratchElem::I32 => ScratchBuf::I32(Vec::new()),
            ScratchElem::I64 => ScratchBuf::I64(Vec::new()),
            ScratchElem::Str => ScratchBuf::Str(Vec::new()),
            ScratchElem::Bytes => ScratchBuf::Bytes(Vec::new()),
            ScratchElem::Value => ScratchBuf::Value(Vec::new()),
            ScratchElem::Slots => ScratchBuf::Slots(Vec::new()),
            ScratchElem::Kernels => ScratchBuf::Kernels(Default::default()),
            ScratchElem::State => ScratchBuf::State(NodeState::default()),
        }
    }
}

/// Compiled closure for a node with typed-slice ports (§8.4
/// layer 3). Same calling shape as [`CompiledU64Op`] plus the
/// step's scratch buffers: slice inputs arrive as `(ptr, len)`
/// slot pairs in `inputs`; vector outputs are written into
/// scratch and their `(ptr, len)` into `outputs`.
pub type CompiledSlotOp = Box<dyn Fn(&[u64], &mut [u64], &mut [ScratchBuf]) + Send + Sync>;

/// A slot-compiled node's closure plus its scratch declaration
/// (one [`ScratchElem`] per vector-producing output, in port
/// order). Returned by [`PolydatNode::compiled_slot`].
pub struct CompiledSlotKit {
    /// The closure: slice inputs as slot pairs, vector outputs into scratch.
    pub op: CompiledSlotOp,
    /// One element type per vector-producing output, in port order.
    pub scratch: Vec<ScratchElem>,
}

/// Per-node purity classification per
/// [`runtime_model.md`'s D2 axiom][spec] and
/// [`composition_substrate.md`'s T1+T2 axioms][substrate].
///
/// Every node declares its purity status via
/// [`PolydatNode::purity`]. The default is [`Purity::Pure`]; nodes
/// with observable side channels (logging, file I/O, network)
/// or eval-call-spanning state override to declare
/// [`Purity::SideChannel`] or [`Purity::Nondeterministic`].
///
/// **D1 (Typed Return Determinism) holds for every purity
/// class.** The slot contract carries only typed return
/// values; impure nodes still produce typed-deterministic
/// returns. What varies between purity classes is the
/// *observable side channels* (D2): pure nodes have none;
/// SideChannel nodes have declared side channels; Stateful
/// nodes additionally have internal eval-call-spanning state
/// that affects future evaluations.
///
/// [spec]: https://github.com/nosqlbench/polydat/blob/main/crates/polydat/docs/design/runtime_model.md
/// [substrate]: https://github.com/nosqlbench/polydat/blob/main/crates/polydat/docs/design/composition_substrate.md
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Purity {
    /// Pure function — `eval(inputs)` is a function of inputs,
    /// no observable side effects, byte-identical determinism
    /// across calls with identical inputs.
    Pure,

    /// Has an observable side channel (logging, file I/O,
    /// network, etc.) but the typed return value is still a
    /// function of inputs. Hosts that care about side-channel
    /// observability examine the `sink` to know what
    /// observable surface this node writes to.
    SideChannel {
        /// The observable surface the node writes to.
        sink: SideChannelSink,
    },

    /// The typed return value is not a function of declared
    /// inputs alone — it depends on external sources (system
    /// clock, entropy, thread identity, environment) or on
    /// eval-call-spanning internal state mutated by prior calls.
    /// In either case, the runtime's `node_clean` caching model
    /// must opt the node out of within-cycle memoization
    /// suppression; the assembler's lifecycle classes mark the node
    /// as nondeterministic (`PolydatProgram::nondeterministic`).
    /// The `reason` string documents the source of
    /// non-determinism (e.g., "reads system clock",
    /// "monotonic counter incremented per call",
    /// "accumulates signal buffer across calls").
    ///
    /// This is the intrinsic-volatility marker referenced by
    /// runtime_model.md R1.v: certain library nodes declare
    /// themselves volatile via this variant; no user opt-in is
    /// required, and the workload author cannot remove the
    /// marker. User-opt-in volatility via the `volatile`
    /// modifier is a separate surface that produces the same
    /// runtime effect (see R1.v).
    Nondeterministic {
        /// The source of the non-determinism, for diagnostics.
        reason: &'static str,
    },
}

/// Where a [`Purity::SideChannel`] node writes its observable
/// side effects. Hosts reasoning about side-channel
/// determinism (D2) pattern-match on this to know what
/// observable surface to expect.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SideChannelSink {
    /// Writes to the process's stderr.
    Stderr,
    /// Writes to the process's stdout.
    Stdout,
    /// Writes to a log buffer (e.g. tracing/log crate sink).
    LogBuffer,
    /// Writes to a file path determined at construction time.
    File,
    /// Writes to a network endpoint determined at
    /// construction time.
    Network,
    /// Writes to an observable surface not covered by the
    /// other variants. The host should consult the node's
    /// documentation for the specific contract.
    Other,
}

/// Semantic contract for a scalar node's explicitly registered SIMD variant.
///
/// This metadata is deliberately attached to the scalar node rather than
/// inferred from function names. A promotion pass may use it only after it
/// also validates the scalar/register port shapes and proves that the complete
/// vector cone lowers for the effective host ISA.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SimdVariant {
    /// DSL name of the register-typed, lane-wise equivalent node.
    pub vector_node: &'static str,
    /// Whether every lane is exactly equivalent to one scalar invocation.
    pub exact: bool,
    /// Whether evaluation is total for every bit pattern admitted by the
    /// scalar input types. Tier-1 padded execution requires this flag.
    pub total: bool,
    /// Whether one lane can be evaluated without reading or changing another
    /// lane. Scalar-flow auto-promotion requires this flag.
    pub lane_independent: bool,
}

impl SimdVariant {
    /// Exact, total, element-wise variant used by the first promotion tier.
    pub const fn exact_total(vector_node: &'static str) -> Self {
        Self {
            vector_node,
            exact: true,
            total: true,
            lane_independent: true,
        }
    }

    /// Exact element-wise variant which may fault for some lane values.
    ///
    /// Such a variant can be used only when the planner proves the admitted
    /// value range or implements ordered lane-error attribution.
    pub const fn exact_fallible(vector_node: &'static str) -> Self {
        Self {
            vector_node,
            exact: true,
            total: false,
            lane_independent: true,
        }
    }
}

/// Runtime evaluation interface for a Polydat node.
///
/// Every engine drives this trait: the interpreter through `eval`,
/// the closure and native engines through `compiled_u64` /
/// `compiled_slot` where a node offers them and the node's own
/// closure elsewhere.
pub trait PolydatNode: Send + Sync {
    /// Return this node's metadata (port names and types).
    fn meta(&self) -> &NodeMeta;

    /// Evaluate the node: read from `inputs`, write to `outputs`.
    ///
    /// The assembly phase guarantees that `inputs` and `outputs` have
    /// the correct length and types matching `meta()`.
    fn eval(&self, inputs: &[Value], outputs: &mut [Value]);

    /// The scratch entries a state owns for this node's evaluation
    /// (axiom S3), one per entry in the order the node expects them
    /// in [`Self::eval_in`]. Empty for a node that evaluates over
    /// `Value`s alone, which is every node but a native cone.
    fn scratch_layout(&self) -> Vec<ScratchElem> {
        Vec::new()
    }

    /// [`Self::eval`] with the node's scratch, which the evaluating
    /// state owns and hands in: storage belongs to the state, never to
    /// the node, which is shared by every state of the program.
    fn eval_in(&self, scratch: &mut [ScratchBuf], inputs: &[Value], outputs: &mut [Value]) {
        let _ = scratch;
        self.eval(inputs, outputs)
    }

    /// Declare which inputs are interchangeable for this node.
    ///
    /// Override for commutative operations like `sum`, `product`,
    /// `min`, `max`. The default is `Positional` (order matters).
    fn commutativity(&self) -> Commutativity {
        Commutativity::Positional
    }

    /// True iff this node should receive `Value::None` inputs
    /// directly rather than have the kernel propagate None through
    /// it. Default: false — most nodes follow SRD-74 Rule 1
    /// (None in → None out, no eval invocation).
    ///
    /// Override to true for nodes whose semantics explicitly
    /// consume None: coalesce-style fallbacks (`default_or`),
    /// optional/maybe handlers, anything that distinguishes
    /// "present" from "absent" as part of its contract.
    /// Override-true nodes are responsible for handling
    /// `Value::None` in their own `eval` implementation.
    ///
    /// See `crates/polydat/docs/design/none_semantics.md`
    /// (string-interpolation propagates None) — the
    /// rule is general (lifted to the kernel level) rather than
    /// per-node; this flag is the opt-out for legitimate None-
    /// aware operators.
    fn accepts_none_inputs(&self) -> bool {
        false
    }

    /// Return a compiled u64-only evaluation closure, if this node
    /// operates entirely in u64 space.
    ///
    /// The closure reads from an input slice and writes to an output
    /// slice, both `&[u64]` / `&mut [u64]`. Assembly-time parameters
    /// are captured in the closure.
    ///
    /// Return `None` if the node has non-u64 ports or cannot be
    /// compiled. The assembly phase will fall back to Phase 1.
    fn compiled_u64(&self) -> Option<CompiledU64Op> {
        None
    }

    /// Return a slot-compiled closure for nodes with typed-slice
    /// ports (§8.4 layer 3): slice inputs read `(ptr, len)` slot
    /// pairs; vector outputs write into kernel-owned scratch.
    /// Checked by the compiled-kernel builders AFTER
    /// [`Self::compiled_u64`] — pure-scalar nodes never need it.
    /// Default `None`: the node stays on typed eval.
    ///
    /// `engine` is the engine the kit is being built for, which a node
    /// needs when its closure runs a program of its own: a tile's
    /// projection body belongs to the kernel rendering it, the way a
    /// `for` body belongs to the kernel that opened it, and the kit is
    /// the only place a closure can learn which that is.
    fn compiled_slot(
        &self,
        _wire_types: &[PortType],
        _engine: crate::compile::select::Engine,
    ) -> Option<CompiledSlotKit> {
        None
    }

    /// Return assembly-time constants for JIT compilation.
    ///
    /// Nodes with baked-in constants (Mod's modulus, Add's addend, etc.)
    /// override this to expose their constants to the JIT compiler.
    /// Returns a list of u64 constants in the order the JIT expects.
    ///
    /// Default: empty (no constants to expose).
    fn jit_constants(&self) -> Vec<u64> {
        Vec::new()
    }

    /// Declare this node's purity status per the
    /// [`runtime_model.md`'s D2 axiom][spec]. Default:
    /// [`Purity::Pure`]. Override to declare an observable
    /// side channel ([`Purity::SideChannel`]) or
    /// eval-call-spanning state ([`Purity::Nondeterministic`]).
    ///
    /// **What this affects:**
    ///
    /// - The runtime's `node_clean` cache (R1) holds for
    ///   `Purity::Pure` and `Purity::SideChannel`. The
    ///   typed return value is cached after one eval;
    ///   subsequent pulls with identical inputs reuse the
    ///   cache. For `SideChannel` nodes, this means the
    ///   side channel fires once per dirty-to-clean
    ///   transition (not on every pull).
    /// - `Purity::Nondeterministic` nodes opt out of `node_clean`
    ///   caching at the construction tier (the assembler's
    ///   lifecycle classes mark them as nondeterministic,
    ///   `PolydatProgram::nondeterministic`).
    /// - Hosts inspecting an expression's determinism
    ///   profile via D2 read this declaration to know
    ///   whether the constituent node has side channels.
    ///
    /// Default: `Purity::Pure`. Most nodes are pure
    /// functions over their inputs.
    ///
    /// [spec]: https://github.com/nosqlbench/polydat/blob/main/crates/polydat/docs/design/runtime_model.md
    fn purity(&self) -> Purity {
        Purity::Pure
    }

    /// Explicit SIMD-native implementation of this scalar node, if one has
    /// been registered with a semantic contract.
    ///
    /// Returning metadata does not itself make a node promotable. The planner
    /// must still validate types, purity, source replay, packet ownership, and
    /// successful lowering by the same Cranelift ISA used for code generation.
    fn simd_variant(&self) -> Option<SimdVariant> {
        None
    }

    /// A synthetic fusion node's view of the subgraph it stands in
    /// for (SRD-105 cone extraction). Program-identity hashing
    /// (`PolydatProgram::canonical_hash`) walks THROUGH fusion
    /// nodes into this subgraph, so identity is invariant to the
    /// engine mix: `jit=off` and `jit=auto` compiles of the same
    /// source hash identically, and resume-skip matching survives
    /// mode changes. Default `None`: ordinary nodes hash as
    /// themselves.
    fn fusion_subgraph(&self) -> Option<FusionSubgraph<'_>> {
        None
    }
}

/// Borrowed view of the subgraph a fusion node replaced. Local
/// wiring convention: `WireSource::Input(i)` refers to the fusion
/// node's i-th input wire in the OUTER graph; `NodeOutput(j, p)`
/// refers to member `j`'s port `p`.
pub struct FusionSubgraph<'a> {
    /// The original member nodes, verbatim.
    pub members: &'a [Box<dyn PolydatNode>],
    /// Per-member local wiring (see convention above).
    pub wiring: &'a [Vec<crate::kernel::WireSource>],
    /// Per fusion output port: `(member index, member port)` —
    /// the original producer behind that port.
    pub out_ports: &'a [(usize, usize)],
}

/// The compile level of a node, given the types of the wires feeding
/// it. One call to [`crate::compile::node_tier`], which is the order
/// every builder walks; the types are needed because a node's slot kit
/// is offered per call site with the types the kernel fixed.
///
/// Prefer [`crate::kernel::PolydatProgram::node_compile_level`], which
/// reads the types out of the program rather than asking the caller
/// for them.
pub fn compile_level_of(node: &dyn PolydatNode, wire_types: &[PortType]) -> CompileLevel {
    crate::compile::node_tier(node, wire_types)
}

/// The maximum compilation level a node supports.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompileLevel {
    /// Runtime interpreter: `dyn PolydatNode` + `Value` enum.
    Phase1,
    /// Compiled closure: `Box<dyn Fn(&[u64], &mut [u64])>`.
    Phase2,
    /// JIT native code via Cranelift.
    Phase3,
}

#[cfg(test)]
mod purity_tests {
    use super::*;

    /// A minimal pure node — defaults to `Purity::Pure` via
    /// the trait default impl.
    struct DefaultPureNode {
        meta: NodeMeta,
    }

    impl PolydatNode for DefaultPureNode {
        fn meta(&self) -> &NodeMeta {
            &self.meta
        }
        fn eval(&self, _inputs: &[Value], outputs: &mut [Value]) {
            outputs[0] = Value::U64(42);
        }
    }

    /// A node that explicitly declares a side channel.
    struct SideChannelNode {
        meta: NodeMeta,
    }

    impl PolydatNode for SideChannelNode {
        fn meta(&self) -> &NodeMeta {
            &self.meta
        }
        fn eval(&self, _inputs: &[Value], _outputs: &mut [Value]) {}
        fn purity(&self) -> Purity {
            Purity::SideChannel {
                sink: SideChannelSink::Stderr,
            }
        }
    }

    /// A node that explicitly declares stateful behaviour.
    struct StatefulNode {
        meta: NodeMeta,
    }

    impl PolydatNode for StatefulNode {
        fn meta(&self) -> &NodeMeta {
            &self.meta
        }
        fn eval(&self, _inputs: &[Value], _outputs: &mut [Value]) {}
        fn purity(&self) -> Purity {
            Purity::Nondeterministic {
                reason: "test fixture",
            }
        }
    }

    fn empty_meta() -> NodeMeta {
        NodeMeta {
            name: "test".into(),
            ins: vec![],
            outs: vec![Port::u64("out")],
        }
    }

    #[test]
    fn default_purity_is_pure() {
        let n = DefaultPureNode { meta: empty_meta() };
        assert_eq!(n.purity(), Purity::Pure);
    }

    #[test]
    fn side_channel_declaration_is_observable() {
        let n = SideChannelNode { meta: empty_meta() };
        match n.purity() {
            Purity::SideChannel { sink } => assert_eq!(sink, SideChannelSink::Stderr),
            other => panic!("expected SideChannel, got {other:?}"),
        }
    }

    #[test]
    fn stateful_declaration_is_observable() {
        let n = StatefulNode { meta: empty_meta() };
        match n.purity() {
            Purity::Nondeterministic { reason } => assert_eq!(reason, "test fixture"),
            other => panic!("expected Stateful, got {other:?}"),
        }
    }

    #[test]
    fn inspect_node_declares_stderr_side_channel() {
        let n = crate::library::diagnostic::Inspect::new(PortType::U64, "x".to_string());
        match n.purity() {
            Purity::SideChannel { sink } => assert_eq!(sink, SideChannelSink::Stderr),
            other => panic!("inspect should declare Stderr SideChannel, got {other:?}"),
        }
    }

    #[test]
    fn log_passthrough_declares_log_buffer_side_channel() {
        let n = crate::library::log_levels::LogInfo::new(PortType::U64);
        match n.purity() {
            Purity::SideChannel { sink } => assert_eq!(sink, SideChannelSink::LogBuffer),
            other => panic!("log_passthrough should declare LogBuffer SideChannel, got {other:?}"),
        }
    }
}

#[cfg(test)]
mod value_size_probe {
    /// The `Value` enum rides per-slot in every node buffer; its
    /// size is a load-bearing budget: 40 bytes (the `SliceArc`
    /// borrow shape) at alignment 8. The 128-bit integer variants
    /// deliberately ride as two u64 limbs ([`super::Bits128`])
    /// instead of raw `u128`/`i128` payloads — a native 128-bit
    /// field would force the enum to alignment 16 and grow every
    /// buffer slot to 48 bytes for a rarely-carried type
    /// (type_system_alignment.md §8.1). This test pins the
    /// envelope so an accidental payload regression is caught at
    /// the door.
    #[test]
    fn value_fits_size_envelope() {
        assert!(
            std::mem::size_of::<super::Value>() <= 40,
            "Value grew past the 40-byte envelope: {}",
            std::mem::size_of::<super::Value>()
        );
        assert_eq!(
            std::mem::align_of::<super::Value>(),
            8,
            "Value alignment must stay 8 — a 16-aligned payload \
             (raw u128/i128?) snuck in"
        );
    }
}

/// A borrowed view of a [`Value`] (SRD 115 §6.1): what a compiled helper
/// or closure sees for an argument it does not own. A scalar is carried
/// by value, a string or byte string by reference into the arena or the
/// interner, a JSON value by reference into the value table, and any
/// other variant by reference to the `Value` itself. The P1 nodes build
/// the same view from their `Value` inputs, so one body serves both
/// tiers without copying a string argument to inspect it.
#[derive(Clone, Copy, Debug)]
pub enum ValueRef<'a> {
    /// An unsigned integer.
    U64(u64),
    /// A signed integer.
    I64(i64),
    /// A float.
    F64(f64),
    /// A boolean.
    Bool(bool),
    /// A string, borrowed from the arena or the interner.
    Str(&'a str),
    /// A byte string, borrowed.
    Bytes(&'a [u8]),
    /// A JSON value, by reference into the value table.
    Json(&'a serde_json::Value),
    /// No value.
    None,
    /// Any other variant, by reference to the value.
    Other(&'a Value),
}

impl<'a> From<&'a Value> for ValueRef<'a> {
    fn from(v: &'a Value) -> Self {
        match v {
            Value::U64(x) => ValueRef::U64(*x),
            Value::I64(x) => ValueRef::I64(*x),
            Value::F64(x) => ValueRef::F64(*x),
            Value::Bool(b) => ValueRef::Bool(*b),
            Value::Str(s) => ValueRef::Str(s),
            Value::Bytes(b) => ValueRef::Bytes(b),
            Value::Json(j) => ValueRef::Json(j),
            Value::None => ValueRef::None,
            other => ValueRef::Other(other),
        }
    }
}

impl<'a> ValueRef<'a> {
    /// The port type of the value viewed.
    pub fn port_type(&self) -> PortType {
        match self {
            ValueRef::U64(_) => PortType::U64,
            ValueRef::I64(_) => PortType::I64,
            ValueRef::F64(_) => PortType::F64,
            ValueRef::Bool(_) => PortType::Bool,
            ValueRef::Str(_) => PortType::Str,
            ValueRef::Bytes(_) => PortType::Bytes,
            ValueRef::Json(_) => PortType::Json,
            ValueRef::None => Value::None.port_type(),
            ValueRef::Other(v) => v.port_type(),
        }
    }

    /// The display form, exactly as [`Value::to_display_string`] gives
    /// it; a string is borrowed rather than copied.
    pub fn display(&self) -> std::borrow::Cow<'a, str> {
        use std::borrow::Cow;
        match self {
            ValueRef::Str(s) => Cow::Borrowed(s),
            ValueRef::U64(v) => Cow::Owned(v.to_string()),
            ValueRef::I64(v) => Cow::Owned(v.to_string()),
            ValueRef::F64(v) => Cow::Owned(format!("{v:?}")),
            ValueRef::Bool(v) => Cow::Owned(v.to_string()),
            ValueRef::Bytes(b) => Cow::Owned(b.iter().map(|b| format!("{b:02x}")).collect()),
            ValueRef::Json(j) => Cow::Owned(j.to_string()),
            ValueRef::None => Cow::Owned(Value::None.to_display_string()),
            ValueRef::Other(v) => Cow::Owned(v.to_display_string()),
        }
    }

    /// The display form as an owned string.
    pub fn to_display_string(&self) -> String {
        self.display().into_owned()
    }

    /// The JSON projection, exactly as [`Value::to_json_value`] gives it.
    pub fn to_json_value(&self) -> serde_json::Value {
        match self {
            ValueRef::U64(v) => serde_json::Value::from(*v),
            ValueRef::I64(v) => serde_json::Value::from(*v),
            ValueRef::F64(v) => serde_json::json!(*v),
            ValueRef::Bool(v) => serde_json::Value::from(*v),
            ValueRef::Str(s) => serde_json::Value::from(*s),
            ValueRef::Bytes(b) => {
                serde_json::Value::from(b.iter().map(|b| format!("{b:02x}")).collect::<String>())
            }
            ValueRef::Json(j) => (*j).clone(),
            ValueRef::None => Value::None.to_json_value(),
            ValueRef::Other(v) => v.to_json_value(),
        }
    }
}

#[cfg(test)]
mod satisfies_slot_tests {
    use super::*;

    /// A float node output rides its bit pattern in `Value::U64`
    /// (`Wire for f32` / `Wire for f16` inject it so), and a host may
    /// write the materialised `Value::F64` instead; a float slot
    /// accepts both, and a `U64` slot does not accept a float.
    #[test]
    fn float_slots_accept_the_bit_stuffed_and_materialised_forms() {
        let f32_bits = Value::U64(1.5f32.to_bits() as u64);
        let f16_bits = Value::U64(half::f16::from_f32(1.5).to_bits() as u64);
        assert!(f32_bits.satisfies_slot(PortType::F32));
        assert!(f16_bits.satisfies_slot(PortType::F16));
        assert!(Value::F64(1.5).satisfies_slot(PortType::F32));
        assert!(Value::F64(1.5).satisfies_slot(PortType::F16));
        assert!(!Value::F64(1.5).satisfies_slot(PortType::U64));
        assert!(!Value::Str("1.5".into()).satisfies_slot(PortType::F32));
    }
}
