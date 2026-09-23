// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Trait surface that the `#[polydat_node]` proc-macro
//! (`polydat-derive`) calls into for boxing / unboxing wire
//! values.
//!
//! ## Canonical trait surface (SRD-80b)
//!
//! - [`Wire`] — `Sized + 'static` Rust-type ↔ [`Value`] bridge.
//!   Owned types only; the macro recognises borrow shapes
//!   (`&str`, `&[u8]`, `&[T]`, `&serde_json::Value`)
//!   syntactically and emits direct `match`-on-`Value`
//!   extraction at the eval call site — no trait dispatch, no
//!   `unsafe` lifetime transmute.
//!
//! Combinator [`Wire`] impls cover [`Option<T>`] (`None`-aware
//! pass-through) and [`Ext<T>`] (downcast through
//! [`ReflectedValue`]).
//!
//! A `Const<T>` position is not a `Wire`: the macro classifies it
//! syntactically into a `ConstShape` and emits the extraction from
//! the `ConstArg` inline, so there is no trait for it to dispatch
//! through. A `ConstSource` trait once carried that, and stayed here
//! after the macro stopped emitting calls to it.
//!
//! ## Why the trait surface lives here
//!
//! `polydat-derive` is a proc-macro crate — it can't define
//! traits that are visible at the call site, only emit token
//! streams referencing traits defined elsewhere. The macro
//! emits `<T as polydat::derive_support::Wire>::extract(...)`
//! paths; this module is what those paths resolve to.

use std::sync::Arc;

use crate::ast::SlotShape;
use crate::ast::{PortType, ReflectedValue, SliceArc, Value};

// =====================================================================
// Wire — Rust-type ↔ Value bridge (owned types only)
// =====================================================================

/// Rust-type ↔ `Value` bridge.
///
/// Every owned Rust type the macro accepts in a wire position
/// implements this trait. `PORT` is the static [`PortType`] the
/// DSL type-checker uses to route a wire to this slot; which slots
/// a value rides and how wide they are follows from that port type
/// through [`SlotShape`], the one place that mapping lives. A second
/// `JIT: Option<JitType>` const here said the same thing about the
/// compiled buffer and nothing ever read it.
///
/// Borrow shapes (`&str`, `&[u8]`, `&[T]`,
/// `&serde_json::Value`) and polymorphic `Value`-typed wires
/// are NOT covered here — the macro recognises them
/// syntactically and emits direct `match`-on-`Value` extraction
/// at the eval call site. This keeps the trait surface free of
/// lifetime parameters.
///
/// `extract` panics on type mismatch — the DSL type-checker is
/// responsible for routing well-typed `Value`s to each slot
/// before `eval` runs. A panic here is a "type-checker was
/// lied to" bug, not a normal path.
pub trait Wire: Sized + 'static {
    /// Static port type for the DSL type-checker.
    const PORT: PortType;

    /// SRD-53 §"Source-string call-site sugar" — auto-resolver
    /// for `Str`-typed upstream wires feeding this slot. `None`
    /// (the default) disables auto-promotion; the workload must
    /// supply the wire's actual port type directly. Set via the
    /// [`Resolved<R, T>`] marker wrapper.
    const RESOLVER: Option<crate::dsl::registry::DefaultResolver> = None;

    /// SRD-15 §"WireCost::Config" — cost class for this wire.
    /// Defaults to [`WireCost::Data`](crate::ast::WireCost::Data) (cheap per-cycle input).
    /// Set to [`WireCost::Config`](crate::ast::WireCost::Config) via the [`Config<T>`] marker
    /// wrapper to signal that the wire is rarely-changing and
    /// the compiler should warn on cycle-time binding.
    const WIRE_COST: crate::ast::WireCost = crate::ast::WireCost::Data;

    /// Pull a typed value out of a `Value` wire.
    fn extract(v: &Value) -> Self;

    /// Push a typed value back into the `Value` outputs stream.
    fn inject(self) -> Value;
}

// ── Scalar primitives ─────────────────────────────────────────

impl Wire for u64 {
    const PORT: PortType = PortType::U64;
    fn extract(v: &Value) -> Self {
        v.as_u64()
    }
    fn inject(self) -> Value {
        Value::U64(self)
    }
}

impl Wire for u32 {
    const PORT: PortType = PortType::U32;
    fn extract(v: &Value) -> Self {
        v.as_u64() as u32
    }
    fn inject(self) -> Value {
        Value::U64(self as u64)
    }
}

impl Wire for i32 {
    const PORT: PortType = PortType::I32;
    // Lenient extract: honest `Value::I64` (sign-extended I32
    // storage convention) plus the legacy bit-stuffed `Value::U64`
    // form during the alignment migration — same precedent as
    // `Wire<bool>` accepting `U64(n != 0)`.
    fn extract(v: &Value) -> Self {
        v.as_i64() as i32
    }
    fn inject(self) -> Value {
        Value::I64(self as i64)
    }
}

impl Wire for i64 {
    const PORT: PortType = PortType::I64;
    // Lenient extract: see `Wire<i32>` note above.
    fn extract(v: &Value) -> Self {
        v.as_i64()
    }
    fn inject(self) -> Value {
        Value::I64(self)
    }
}

impl Wire for u8 {
    const PORT: PortType = PortType::U8;
    fn extract(v: &Value) -> Self {
        v.as_u64() as u8
    }
    fn inject(self) -> Value {
        Value::U64(self as u64)
    }
}

impl Wire for u16 {
    const PORT: PortType = PortType::U16;
    fn extract(v: &Value) -> Self {
        v.as_u64() as u16
    }
    fn inject(self) -> Value {
        Value::U64(self as u64)
    }
}

impl Wire for i8 {
    const PORT: PortType = PortType::I8;
    // Lenient extract through as_i64 (honest I64 or legacy
    // stuffed U64), narrowed by truncation — sign survives
    // because the storage convention is sign-extension.
    fn extract(v: &Value) -> Self {
        v.as_i64() as i8
    }
    fn inject(self) -> Value {
        Value::I64(self as i64)
    }
}

impl Wire for i16 {
    const PORT: PortType = PortType::I16;
    fn extract(v: &Value) -> Self {
        v.as_i64() as i16
    }
    fn inject(self) -> Value {
        Value::I64(self as i64)
    }
}

impl Wire for u128 {
    const PORT: PortType = PortType::U128;
    // No named native lowering: a 128-bit value cannot ride the
    // one-u64 JIT slot, so it crosses the compiled tiers as a limb
    // pair (`Imm2`) and a node over it runs its closure on the
    // closure tier and a slot call of its kit on the native engines
    // (type_system_alignment.md §2). The two-slot ride is about the
    // register, not about which engines carry the type — every one
    // of them does, which `the_128_bit_carriers_agree_on_every_engine`
    // pins.
    fn extract(v: &Value) -> Self {
        v.as_u128()
    }
    fn inject(self) -> Value {
        Value::U128(crate::ast::Bits128::from_u128(self))
    }
}

impl Wire for i128 {
    const PORT: PortType = PortType::I128;
    fn extract(v: &Value) -> Self {
        v.as_i128()
    }
    fn inject(self) -> Value {
        Value::I128(crate::ast::Bits128::from_i128(self))
    }
}

// ── 128-bit register words (type_system_alignment.md §8.4 L2) ──
//
// The raw view extracts/injects the word itself; the lane-typed
// `[T; N]` views extract through the free-bitcast rule (any
// register view satisfies any register slot) and inject tagged
// with their own lane typing.

impl Wire for crate::ast::Bits128 {
    const PORT: PortType = PortType::Reg128;
    fn extract(v: &Value) -> Self {
        v.as_reg_bits()
    }
    fn inject(self) -> Value {
        Value::Reg128(self, crate::ast::RegLanes::Raw)
    }
}

macro_rules! impl_wire_reg {
    ($arr:ty, $port:ident, $view:ident, $to:ident, $from:ident) => {
        impl Wire for $arr {
            const PORT: PortType = PortType::$port;
            fn extract(v: &Value) -> Self {
                v.as_reg_bits().$to()
            }
            fn inject(self) -> Value {
                Value::Reg128(
                    crate::ast::Bits128::$from(self),
                    crate::ast::RegLanes::$view,
                )
            }
        }
    };
}

impl_wire_reg!([i8; 16], RegI8x16, I8x16, lanes_i8, from_lanes_i8);
impl_wire_reg!([i16; 8], RegI16x8, I16x8, lanes_i16, from_lanes_i16);
impl_wire_reg!([i32; 4], RegI32x4, I32x4, lanes_i32, from_lanes_i32);
impl_wire_reg!([i64; 2], RegI64x2, I64x2, lanes_i64, from_lanes_i64);
impl_wire_reg!([half::f16; 8], RegF16x8, F16x8, lanes_f16, from_lanes_f16);
impl_wire_reg!([f32; 4], RegF32x4, F32x4, lanes_f32, from_lanes_f32);
impl_wire_reg!([f64; 2], RegF64x2, F64x2, lanes_f64, from_lanes_f64);

impl Wire for f64 {
    const PORT: PortType = PortType::F64;
    fn extract(v: &Value) -> Self {
        v.as_f64()
    }
    fn inject(self) -> Value {
        Value::F64(self)
    }
}

impl Wire for f32 {
    const PORT: PortType = PortType::F32;
    fn extract(v: &Value) -> Self {
        f32::from_bits(v.as_u64() as u32)
    }
    fn inject(self) -> Value {
        Value::U64(self.to_bits() as u64)
    }
}

impl Wire for half::f16 {
    const PORT: PortType = PortType::F16;
    // Same bit-stuffing convention as f32: the binary16 pattern
    // rides the low 16 bits of the u64 carrier.
    fn extract(v: &Value) -> Self {
        half::f16::from_bits(v.as_u64() as u16)
    }
    fn inject(self) -> Value {
        Value::U64(self.to_bits() as u64)
    }
}

impl Wire for bool {
    const PORT: PortType = PortType::Bool;
    fn extract(v: &Value) -> Self {
        match v {
            Value::Bool(b) => *b,
            Value::U64(n) => *n != 0,
            other => panic!(
                "Wire<bool>::extract: type-checker routed {other:?} \
                 to a Bool slot"
            ),
        }
    }
    fn inject(self) -> Value {
        Value::Bool(self)
    }
}

impl Wire for String {
    const PORT: PortType = PortType::Str;
    fn extract(v: &Value) -> Self {
        // SRD-80b: panic on shape mismatch — the type-checker is
        // responsible for routing well-typed values to each slot,
        // and a non-Str input here is a "type system was lied to"
        // bug, not a coercion opportunity. Nodes that want a
        // display rendering of an arbitrary `Value` take a
        // `Value`-typed (PolyWire) arg instead.
        match v {
            Value::Str(s) => s.to_string(),
            other => panic!("Wire<String>::extract: expected Str, got {other:?}"),
        }
    }
    fn inject(self) -> Value {
        Value::Str(self.into())
    }
}

/// `Arc<str>` — zero-copy shared string handle. Reading
/// extracts the existing `Arc<str>` from `Value::Str` (refcount
/// bump only); injecting wraps directly. Nodes whose hot path
/// emits the same string per cycle (lookup table outputs,
/// fixed-value selectors) should use this instead of `String`
/// to avoid the per-cycle `to_string()` allocation.
impl Wire for std::sync::Arc<str> {
    const PORT: PortType = PortType::Str;
    fn extract(v: &Value) -> Self {
        match v {
            Value::Str(s) => s.clone(),
            other => panic!("Wire<Arc<str>>::extract: expected Str, got {other:?}"),
        }
    }
    fn inject(self) -> Value {
        Value::Str(self)
    }
}

/// `Arc<dyn Any + Send + Sync>` — opaque Handle wire. The body
/// receives the runtime-typed handle directly; downcast is the
/// operator's responsibility. Use [`Resolved<R, T>`] when the
/// node wants a typed Handle with SRD-53 source-string
/// auto-promotion sugar; use this raw shape when the body
/// needs to handle multiple inner types via runtime dispatch.
impl Wire for std::sync::Arc<dyn std::any::Any + Send + Sync> {
    const PORT: PortType = PortType::Handle;
    fn extract(v: &Value) -> Self {
        match v {
            Value::Handle(arc) => arc.clone(),
            other => panic!("Wire<Arc<dyn Any>>::extract: expected Handle, got {other:?}"),
        }
    }
    fn inject(self) -> Value {
        Value::Handle(self)
    }
}

/// `Box<dyn ReflectedValue>` — Ext (adapter-typed) wire with
/// dynamic downcast left to the body. Use [`Ext<T>`] when the
/// inner type is known at codegen; use this when a node needs
/// to dispatch on the runtime ReflectedValue::type_name.
impl Wire for Box<dyn ReflectedValue> {
    const PORT: PortType = PortType::Ext;
    fn extract(v: &Value) -> Self {
        match v {
            Value::Ext(b) => b.clone_reflected(),
            other => panic!("Wire<Box<dyn ReflectedValue>>::extract: expected Ext, got {other:?}"),
        }
    }
    fn inject(self) -> Value {
        Value::Ext(self)
    }
}

// ── Bytes ──────────────────────────────────────────────────────

impl Wire for Arc<[u8]> {
    const PORT: PortType = PortType::Bytes;
    fn extract(v: &Value) -> Self {
        match v {
            Value::Bytes(b) => b.clone(),
            other => panic!("Wire<Arc<[u8]>>::extract: expected Bytes, got {other:?}"),
        }
    }
    fn inject(self) -> Value {
        Value::Bytes(self)
    }
}

impl Wire for Vec<u8> {
    const PORT: PortType = PortType::Bytes;
    fn extract(v: &Value) -> Self {
        match v {
            Value::Bytes(b) => b.to_vec(),
            other => panic!("Wire<Vec<u8>>::extract: expected Bytes, got {other:?}"),
        }
    }
    fn inject(self) -> Value {
        Value::Bytes(self.into())
    }
}

// ── Json ───────────────────────────────────────────────────────

impl Wire for Arc<serde_json::Value> {
    const PORT: PortType = PortType::Json;
    fn extract(v: &Value) -> Self {
        match v {
            Value::Json(j) => j.clone(),
            other => panic!("Wire<Arc<Json>>::extract: expected Json, got {other:?}"),
        }
    }
    fn inject(self) -> Value {
        Value::Json(self)
    }
}

// ── Typed-element vectors ──────────────────────────────────────

macro_rules! impl_wire_vec {
    ($elem:ty, $variant:ident, $port:ident) => {
        impl Wire for SliceArc<$elem> {
            const PORT: PortType = PortType::$port;
            fn extract(v: &Value) -> Self {
                match v {
                    Value::$variant(arc) => arc.clone(),
                    other => panic!(
                        concat!(
                            "Wire<SliceArc<",
                            stringify!($elem),
                            ">>::extract: expected ",
                            stringify!($variant),
                            ", got {:?}"
                        ),
                        other
                    ),
                }
            }
            fn inject(self) -> Value {
                Value::$variant(self)
            }
        }

        impl Wire for Vec<$elem> {
            const PORT: PortType = PortType::$port;
            fn extract(v: &Value) -> Self {
                match v {
                    Value::$variant(arc) => arc.as_slice().to_vec(),
                    other => panic!(
                        concat!(
                            "Wire<Vec<",
                            stringify!($elem),
                            ">>::extract: expected ",
                            stringify!($variant),
                            ", got {:?}"
                        ),
                        other
                    ),
                }
            }
            fn inject(self) -> Value {
                Value::$variant(SliceArc::from_vec(self))
            }
        }
    };
}

impl_wire_vec!(f32, VecF32, VecF32);
impl_wire_vec!(i32, VecI32, VecI32);
impl_wire_vec!(f64, VecF64, VecF64);
impl_wire_vec!(i64, VecI64, VecI64);
impl_wire_vec!(half::f16, VecF16, VecF16);
impl_wire_vec!(i16, VecI16, VecI16);
impl_wire_vec!(i8, VecI8, VecI8);

// ── Phase C combinators ────────────────────────────────────────

/// None-aware wire combinator. Macro auto-emits
/// `accepts_none_inputs() -> true` when any arg is `Option<_>`.
impl<T: Wire> Wire for Option<T> {
    const PORT: PortType = T::PORT;
    fn extract(v: &Value) -> Self {
        match v {
            Value::None => None,
            _ => Some(T::extract(v)),
        }
    }
    fn inject(self) -> Value {
        match self {
            None => Value::None,
            Some(t) => t.inject(),
        }
    }
}

/// Operator-side wrapper for adapter-typed wire arguments.
/// `Ext<T>` signals "this arg comes from `Value::Ext(Box<dyn
/// ReflectedValue>)`; downcast it to `T`." Implements `Deref` /
/// `DerefMut` like [`Const<T>`] so the body can use `.method()`
/// directly.
#[derive(Clone)]
pub struct Ext<T>(pub T);

impl<T> std::ops::Deref for Ext<T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.0
    }
}

impl<T> std::ops::DerefMut for Ext<T> {
    fn deref_mut(&mut self) -> &mut T {
        &mut self.0
    }
}

impl<T: ReflectedValue + Clone + 'static> Wire for Ext<T> {
    const PORT: PortType = PortType::Ext;
    fn extract(v: &Value) -> Self {
        match v {
            Value::Ext(boxed) => {
                let any = boxed.as_any();
                match any.downcast_ref::<T>() {
                    Some(t) => Ext(t.clone()),
                    None => panic!(
                        "Wire<Ext<{}>>::extract: ReflectedValue downcast failed; \
                         got runtime type {:?}",
                        std::any::type_name::<T>(),
                        boxed.type_name()
                    ),
                }
            }
            other => panic!("Wire<Ext>::extract: expected Ext, got {other:?}"),
        }
    }
    fn inject(self) -> Value {
        Value::Ext(Box::new(self.0))
    }
}

// ── DynamicOutputs<T> — variable output port count ────────────

/// Marker wrapper for node return types whose output port
/// COUNT is determined at construction time from a
/// const-list arg's length, not at codegen time.
/// SRD-80b shape extension covering nodes like `mixed_radix`
/// that emit one output per radix where `radix` count is a
/// workload-supplied list.
///
/// Operator writes:
///
/// ```ignore
/// #[polydat_node(category = Arithmetic)]
/// fn mixed_radix(
///     value: u64,
///     radixes: Const<&[u64]>,
/// ) -> DynamicOutputs<u64> {
///     // body returns DynamicOutputs(Vec<u64>) with len == radixes.len()
/// }
/// ```
///
/// The macro emits one output port per element (named `d0`,
/// `d1`, ...) at construction time using the const-list
/// arg's length. `FuncSig.outputs` is `0` signalling dynamic.
/// Requires exactly one const-list arg per function — `Const<&[C]>`
/// or its owned spelling `Const<Vec<C>>` — and the
/// macro errors at compile time otherwise.
pub struct DynamicOutputs<T>(pub Vec<T>);

impl<T> std::ops::Deref for DynamicOutputs<T> {
    type Target = Vec<T>;
    fn deref(&self) -> &Vec<T> {
        &self.0
    }
}

// ── Config<T> — wire arg marked as config-cost ────────────────

/// Marker wrapper signalling that the wrapped wire is a
/// configuration input — expensive to change because the node
/// keeps internal state (LUTs, alias tables, parsed specs)
/// derived from it. The macro emits the matching slot with
/// `Port::config()` (SRD 15 §"WireCost::Config") so the
/// compiler warns on cycle-time binding.
///
/// In-spirit replacement for a `#[wire_cost(Config)]` arg-level
/// attribute — operator declares the cost intent via the type
/// system. Body unwraps with `.0` or via `Deref`.
pub struct Config<T>(pub T);

impl<T> std::ops::Deref for Config<T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.0
    }
}

impl<T: Wire> Wire for Config<T> {
    const PORT: PortType = T::PORT;
    const RESOLVER: Option<crate::dsl::registry::DefaultResolver> = T::RESOLVER;
    const WIRE_COST: crate::ast::WireCost = crate::ast::WireCost::Config;
    fn extract(v: &Value) -> Self {
        Config(T::extract(v))
    }
    fn inject(self) -> Value {
        self.0.inject()
    }
}

// ── Resolved<R, T> — Handle wire with SRD-53 auto-resolver ────

/// SRD-80b in-spirit replacement for the `default_resolver`
/// attribute. The `R` parameter (a [`ResolverKind`] impl) carries
/// the auto-resolver kind; the `T` parameter is the concrete
/// `Handle`-inner type the body sees.
///
/// Operators write:
///
/// ```ignore
/// fn matching_profiles(
///     group: Resolved<GroupResolver, DatasetHandle>,
///     prefix: &str,
/// ) -> Vec<String> {
///     let group: &TestDataGroup = group_of(&group);
///     // ... use group methods directly
/// }
/// ```
///
/// The macro reads `<Resolved<GroupResolver, T> as Wire>::RESOLVER`
/// at codegen time and emits the matching `FuncSig.default_resolver`.
/// No `#[polydat_node(default_resolver = ...)]` attribute is
/// involved — the resolver information lives in the function
/// signature where it belongs.
pub struct Resolved<R: ResolverKind, T: 'static + Send + Sync> {
    inner: std::sync::Arc<T>,
    _r: std::marker::PhantomData<fn() -> R>,
}

impl<R: ResolverKind, T: 'static + Send + Sync> std::ops::Deref for Resolved<R, T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.inner
    }
}

impl<R: ResolverKind, T: 'static + Send + Sync> Resolved<R, T> {
    /// Construct from a pre-resolved Arc — useful for tests
    /// and programmatic graph assembly that bypasses the DSL
    /// auto-resolver.
    pub fn from_arc(inner: std::sync::Arc<T>) -> Self {
        Self {
            inner,
            _r: std::marker::PhantomData,
        }
    }
    /// Borrow the inner Arc.
    pub fn as_arc(&self) -> &std::sync::Arc<T> {
        &self.inner
    }
}

/// Marker trait that names a kind of source-string auto-resolver
/// for [`Resolved<R, T>`] wire args. The variants here mirror
/// [`crate::dsl::registry::DefaultResolver`]; each impl picks
/// one of them.
pub trait ResolverKind: 'static {
    /// The resolver this kind names.
    const RESOLVER: crate::dsl::registry::DefaultResolver;
}

/// Splice `dataset_group_open(<source>)` upstream when the wire
/// source is a `Str` (SRD-53 `DefaultResolver::Group`).
pub struct GroupResolver;
impl ResolverKind for GroupResolver {
    const RESOLVER: crate::dsl::registry::DefaultResolver =
        crate::dsl::registry::DefaultResolver::Group;
}

impl<R: ResolverKind, T: 'static + Send + Sync> Wire for Resolved<R, T> {
    const PORT: PortType = PortType::Handle;
    const RESOLVER: Option<crate::dsl::registry::DefaultResolver> =
        Some(<R as ResolverKind>::RESOLVER);
    fn extract(v: &Value) -> Self {
        match v {
            Value::Handle(arc) => {
                let inner = arc.clone().downcast::<T>().unwrap_or_else(|_| {
                    panic!(
                        "Wire<Resolved<_, {}>>::extract: Handle downcast failed",
                        std::any::type_name::<T>()
                    )
                });
                Resolved {
                    inner,
                    _r: std::marker::PhantomData,
                }
            }
            // The common fault, and the one worth naming: an
            // upstream open did not resolve, so the handle this
            // node reads is absent rather than wrong.
            Value::None => panic!(
                "a resolved handle is None — the upstream open failed to \
                 resolve. The audit log carries the underlying error and \
                 the name it was opening: a catalog miss, a facet missing \
                 on disk, or a transport failure. This is the most common \
                 fault when a workload runs on a system whose catalog is \
                 not configured for the source it asks for."
            ),
            other => panic!("Wire<Resolved>::extract: expected Handle, got {other:?}"),
        }
    }
    fn inject(self) -> Value {
        Value::Handle(self.inner)
    }
}

// FromValue / IntoValue retired 2026-06-05 — the `#[polydat_node]`
// macro now dispatches every owned type through `<T as Wire>::extract`
// / `::inject` and emits direct `match`-on-`Value` extraction for
// borrow shapes (`&str`, `&[u8]`, `&[T]`, `&serde_json::Value`).
// Per SRD-80b Phase B; the old trait pair plus their borrow-impls'
// `unsafe { transmute }` lifetime-extension hack are gone.
//
// [PLACEHOLDER_PHASE_B_DELETE]

/// SRD-80 PR B.5 — marker wrapper for const arguments in
/// `#[polydat_node]` function signatures.
///
/// Use in arg position to signal that the value is captured at
/// node-construction time (assembly-time) rather than read
/// per-cycle from a wire. The macro detects `Const<T>` in arg
/// position and:
///
/// - Emits `Slot::Const { ... }` (not `Slot::Wire`) in the
///   node's NodeMeta.
/// - Emits `SlotType::ConstU64` / `ConstF64` / `ConstStr` in
///   the corresponding `FuncSig.params` entry (the const
///   variant matching `T`).
/// - Adds a struct field to hold the captured value.
/// - Generates a `new(const_values...)` constructor.
/// - Wires the build closure to pull from `consts: &[ConstArg]`
///   and pass values to `new()`.
/// - Constructs a `Const<T>(...)` wrapper around the struct
///   field at eval time so the user's function body sees the
///   wrapped type matching its signature.
///
/// Body code accesses the wrapped value via `.0` or via the
/// `Deref` impl below:
///
/// ```ignore
/// #[polydat_node(category = String)]
/// fn combinations(input: u64, pattern: Const<&str>) -> String {
///     apply(input, pattern.0)  // pattern.0 is &str
/// }
/// ```
///
/// Type-shape dispatch table:
///
/// | `Const<T>` form | `SlotType` variant | Struct field type | ConstArg accessor |
/// |---|---|---|---|
/// | `Const<u64>`  | `ConstU64`  | `u64`    | `as_u64()`  |
/// | `Const<f64>`  | `ConstF64`  | `f64`    | `as_f64()`  |
/// | `Const<bool>` | `ConstU64`  | `bool`   | `as_u64() != 0` |
/// | `Const<&str>` | `ConstStr`  | `String` | `as_str().to_string()` |
pub struct Const<T>(pub T);

impl<T> std::ops::Deref for Const<T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.0
    }
}

impl<T> std::ops::DerefMut for Const<T> {
    fn deref_mut(&mut self) -> &mut T {
        &mut self.0
    }
}

/// SRD-80 PR B.6 — construction-time setup contract for nodes
/// that derive a pre-computed runtime state from their const
/// args (e.g. `combinations` parsing a charset pattern into
/// segments + modulus, `regex_match` compiling a pattern,
/// `histribution` parsing a distribution spec).
///
/// **The contract**: the operator-provided setup function is
/// called EXACTLY ONCE per node instance, at construction time
/// (`new()`). Its result is stored in a struct field; eval-
/// time access is a plain `&T` borrow.
///
/// **Type-level enforcement**: the `#[poly_const(...)]`
/// attribute on a `&T` argument tells the macro to generate
/// this construction pattern. The macro is the sole party
/// emitting `setup_fn(...)` calls and it generates the call
/// exactly once inside `new()`. The contract is inviolable
/// because no other code path can reach the setup function —
/// the macro hides it inside the constructor.
///
/// In effect, the function pointer behaves as `FnOnce` —
/// invoked one time, by one site, never again. The FnOnce
/// semantics aren't expressed as a trait bound because they
/// don't need to be: the macro is the only caller, and the
/// macro respects single-call by construction.
///
/// Library author idiom:
///
/// ```ignore
/// pub struct ParsedPattern {
///     pub segments: Vec<Segment>,
///     pub modulus: u64,
/// }
///
/// impl ParsedPattern {
///     /// Single-call setup. Macro invokes once in `new()`.
///     fn from_pattern(pattern: &str) -> Self { /* parse */ }
/// }
///
/// #[polydat_node(category = String)]
/// fn combinations(
///     input: u64,
///     pattern: Const<&str>,
///     #[poly_const(ParsedPattern::from_pattern, from = pattern)]
///     parsed: &ParsedPattern,
/// ) -> String {
///     // parsed is a borrow of the cached struct field —
///     // no recomputation, no clone, no ceremony at the call site.
///     let mut r = input % parsed.modulus;
///     /* ... */
/// }
/// ```
///
/// Marker trait — purely a documentation handle for types
/// intended to be polydat-setup targets. The macro doesn't
/// dispatch on this; the attribute is the dispatch surface.
/// Implementing the trait gives library authors a way to
/// signal intent and improve `cargo doc` discoverability.
pub trait PolydatSetup {}

// ── The slot kit's run-time helpers ──────────────────────────────
// Called by the closures `#[polydat_node]` emits for its `compiled_slot`
// kit; public because generated code in other crates calls them, not
// because hosts should.

/// The value a `Ref2` pair at the head of `slots` holds by reference:
/// the one-element slice a JSON, extension, or handle producer
/// published (jit_boundary.md, axiom S7: one dereference).
///
/// The slots must hold a pair a producer published into storage that
/// is alive: its own scratch, an extern's stored value, or a boundary
/// value alive for the call (axioms S3, S4). A pair of length zero,
/// an unset extern, reads as [`Value::None`].
#[inline]
pub fn ref_value(slots: &[u64]) -> &Value {
    static NONE: Value = Value::None;
    if slots.get(1).copied().unwrap_or(0) == 0 {
        return &NONE;
    }
    // SAFETY: as documented; the producer's storage outlives the read.
    unsafe { &*(slots[0] as usize as *const Value) }
}

/// An empty buffer with room for `n` elements, for a node whose buffer
/// size comes from a *value* — a wire, a constant, or arithmetic on
/// either.
///
/// `Vec::with_capacity(n as usize)` is the obvious spelling and it is
/// wrong twice over for a value the node did not choose. A size the
/// machine cannot hold makes the allocator **abort the process**, which
/// no `catch_unwind` sees: not an error on the program that asked, but
/// the host gone. And `as usize` truncates on a 32-bit target, so a
/// large size silently becomes a small one. Here a size that does not
/// fit `usize`, or cannot be reserved, is a failure of the node like
/// any other — caught, attributed to the node and its inputs, and the
/// same on every engine.
///
/// There is no cap. A size that *can* be allocated is allocated, however
/// slow filling it is; how large a string a host asks for is the host's
/// business. What this refuses is only what could never have
/// succeeded. Arithmetic on a size belongs in `u64` with
/// `saturating_add`, so an overflow reaches here as a size that cannot
/// be reserved rather than wrapping to a small one first.
pub fn buffer_for<T>(n: u64, what: &str) -> Vec<T> {
    let mut v = Vec::new();
    match usize::try_from(n) {
        Ok(n) if v.try_reserve_exact(n).is_ok() => v,
        _ => refuse_size(n, what),
    }
}

/// [`buffer_for`] for text: an empty `String` with room for `n` bytes.
pub fn string_for(n: u64, what: &str) -> String {
    let mut s = String::new();
    match usize::try_from(n) {
        Ok(n) if s.try_reserve_exact(n).is_ok() => s,
        _ => refuse_size(n, what),
    }
}

/// [`buffer_for`] for a buffer that is reused across evaluations:
/// `out` is cleared and then has room for `n` elements. A native
/// producer writing into step-owned scratch takes this shape, so its
/// refusal reads the same as the node's.
pub fn reserve_for<T>(out: &mut Vec<T>, n: u64, what: &str) {
    out.clear();
    match usize::try_from(n) {
        Ok(n) if out.try_reserve_exact(n).is_ok() => {}
        _ => refuse_size(n, what),
    }
}

#[cold]
fn refuse_size(n: u64, what: &str) -> ! {
    panic!("{what}: a buffer of {n} elements cannot be allocated on this machine")
}

/// A polymorphic port's slots as the owned `Value` the wire type
/// names: a scalar from its bits, a `Ref2` kind copied out of the
/// pair its producer published.
#[inline]
pub fn read_poly(ty: PortType, slots: &[u64]) -> Value {
    crate::compile::marshal::decode_slot(slots, ty)
}

/// A polymorphic return written by the node's resolved output type: a
/// scalar as its bits into `outputs[0]`, a `Ref2` kind into
/// `scratch[0]` with its pair republished (axiom S3). The value must
/// be the port type's carrier: the graph colored the slot by the
/// node's resolved output type, and a value of another type would be
/// read by every consumer as something it is not, where the
/// interpreter would have carried it. A `None` has no slot form on a
/// compiled engine (engines.md §3.3).
///
/// The comparison is with the carrier, not the port type, because a
/// `u32` or `f32` value *is* a `U64` in flight. Comparing with the port
/// type refused every narrow integer and small float that reached a
/// polymorphic node, on every compiled engine.
#[inline]
pub fn write_poly(
    ty: PortType,
    v: Value,
    scratch: &mut [crate::ast::ScratchBuf],
    outputs: &mut [u64],
) {
    if v.port_type() != crate::compile::marshal::carrier_port(ty) {
        panic!(
            "a node produced a {:?} on an output the graph typed {:?}; a compiled engine \
             cannot carry a value of another type than the slot's (engines.md §3.4)",
            v.port_type(),
            ty
        );
    }
    // The slot form is a property of the *type*, and the type is
    // resolved: `ty` is what the graph coloured this output. So the
    // write is chosen by the colour, of which there are three, rather
    // than by the value's variant — a match over `Value` has to be
    // extended every time the language grows a carrier, and a missing
    // arm is not a compile error but a panic at the first pull. This
    // one went through three rounds of that: the by-reference singles,
    // then the numeric vectors, then the 128-bit words.
    match ty.slot_color() {
        // One slot of immediate data: the carrier's bits.
        crate::ast::SlotColor::Imm1 => outputs[0] = carrier_slot_bits(&v),
        // Two slots of immediate limb data, low word first
        // (`Bits128`), which is what every 128-bit and register value
        // is underneath.
        crate::ast::SlotColor::Imm2 => {
            // `as_reg_bits` is the register reading and refuses a
            // 128-bit integer, so the limbs are taken from whichever
            // two-slot carrier this is.
            let words = match v {
                Value::U128(b) | Value::I128(b) | Value::Reg128(b, _) => b.0,
                other => unreachable!("a {:?} is not a two-slot carrier", other.port_type()),
            };
            outputs[0] = words[0];
            outputs[1] = words[1];
        }
        // A (ptr, len) pair into the step's own scratch entry. Filling
        // the entry can move its buffer, so the pair is republished
        // every time — a slot left pointing at the old allocation is
        // what the S9 validator catches (axiom S9(a)).
        crate::ast::SlotColor::Ref2 => {
            scratch[0].set_from_value(&v);
            let (p, l) = scratch[0].ptr_len();
            outputs[0] = p;
            outputs[1] = l;
        }
    }
}

/// The slot bits of a one-slot carrier.
///
/// `Value::None` has no slot form on a compiled engine (engines.md
/// §3.3); it never reaches here, because an unset output is tracked
/// beside the buffer rather than written into it.
#[inline]
fn carrier_slot_bits(v: &Value) -> u64 {
    match v {
        Value::U64(x) => *x,
        Value::I64(x) => *x as u64,
        Value::F64(x) => x.to_bits(),
        Value::Bool(b) => *b as u64,
        other => panic!(
            "a {:?} value is not a one-slot carrier; the graph coloured its output Imm1",
            other.port_type()
        ),
    }
}

// SRD-80 PR B.2/B.3 — macro-generated nodes register through
// the existing `NodeRegistration` inventory channel
// (`polydat::dsl::registry::NodeRegistration`), the same
// channel `register_nodes!` already uses. The proc-macro
// emits a `NodeRegistration` per `#[polydat_node]` site, so
// every consumer that already iterates the registry
// (`registry()`, `lookup()`, the compile pipeline's
// `factory::build_node`) sees macro-generated nodes
// automatically — no parallel collection, no separate dispatch
// surface. See `polydat::dsl::registry` for the load-bearing
// data structures.

#[cfg(test)]
mod tests {
    use super::*;

    fn refusal(f: impl FnOnce() + std::panic::UnwindSafe) -> String {
        let err = std::panic::catch_unwind(f).expect_err("the size must be refused");
        err.downcast_ref::<String>().cloned().unwrap_or_default()
    }

    /// A size no machine can hold is a caught failure naming the node,
    /// not an allocator abort that takes the host down with it.
    #[test]
    fn an_impossible_size_is_refused_not_aborted() {
        let m = refusal(|| drop(buffer_for::<u32>(u64::MAX, "probe")));
        assert!(
            m.starts_with("probe: a buffer of 18446744073709551615"),
            "{m}"
        );
        let m = refusal(|| drop(string_for((1 << 53) + 1, "probe")));
        assert!(m.contains("cannot be allocated"), "{m}");
        let m = refusal(|| reserve_for(&mut vec![1.0f32; 4], u64::MAX, "probe"));
        assert!(m.contains("cannot be allocated"), "{m}");
    }

    #[derive(Debug, Clone)]
    struct Probe;
    impl ReflectedValue for Probe {
        fn type_name(&self) -> &str {
            "probe"
        }
        fn display(&self) -> String {
            "probe".into()
        }
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
        fn clone_reflected(&self) -> Box<dyn ReflectedValue> {
            Box::new(self.clone())
        }
    }

    /// A value of every port type, as its producer's `Wire` impl would
    /// carry it. The match has no wildcard, so a type added to the
    /// language does not compile here until it has one.
    fn sample(ty: PortType) -> Value {
        use crate::ast::{Bits128, RegLanes};
        let limbs = Bits128([0x0123_4567_89ab_cdef, 0xfedc_ba98_7654_3210]);
        let reg = |lanes| Value::Reg128(limbs, lanes);
        match ty {
            PortType::U64 | PortType::U32 | PortType::U16 | PortType::U8 => Value::U64(200),
            PortType::F32 => Value::U64(1.5f32.to_bits() as u64),
            PortType::F16 => Value::U64(half::f16::from_f32(1.5).to_bits() as u64),
            PortType::I64 | PortType::I32 | PortType::I16 | PortType::I8 => Value::I64(-7),
            PortType::F64 => Value::F64(-2.25),
            PortType::Bool => Value::Bool(true),
            PortType::U128 => Value::U128(limbs),
            PortType::I128 => Value::I128(limbs),
            PortType::Reg128 => reg(RegLanes::Raw),
            PortType::RegI8x16 => reg(RegLanes::I8x16),
            PortType::RegI16x8 => reg(RegLanes::I16x8),
            PortType::RegI32x4 => reg(RegLanes::I32x4),
            PortType::RegI64x2 => reg(RegLanes::I64x2),
            PortType::RegF16x8 => reg(RegLanes::F16x8),
            PortType::RegF32x4 => reg(RegLanes::F32x4),
            PortType::RegF64x2 => reg(RegLanes::F64x2),
            PortType::Str => Value::Str("héllo".into()),
            PortType::Bytes => Value::Bytes(vec![0u8, 1, 255].into()),
            PortType::Json => Value::Json(Arc::new(serde_json::json!({"k": [1, 2]}))),
            PortType::Ext => Value::Ext(Box::new(Probe)),
            PortType::Handle => Value::Handle(Arc::new(42u32)),
            PortType::VecF32 => Value::VecF32(SliceArc::from_vec(vec![1.0, -0.5])),
            PortType::VecI32 => Value::VecI32(SliceArc::from_vec(vec![-3, 4])),
            PortType::VecF64 => Value::VecF64(SliceArc::from_vec(vec![1e300, -0.0])),
            PortType::VecI64 => Value::VecI64(SliceArc::from_vec(vec![i64::MIN, 9])),
            PortType::VecF16 => Value::VecF16(SliceArc::from_vec(vec![half::f16::from_f32(0.5)])),
            PortType::VecI16 => Value::VecI16(SliceArc::from_vec(vec![-300i16, 300])),
            PortType::VecI8 => Value::VecI8(SliceArc::from_vec(vec![-8i8, 8])),
        }
    }

    /// Reading a polymorphic port inverts writing one, for every port
    /// type the language has: what a polymorphic node returns is what
    /// the next polymorphic node takes, on every compiled engine.
    ///
    /// A register reaching a polymorphic node's input used to arrive as
    /// its low limb typed `U64`, because the read had a `U64` fallback
    /// arm and the limb reassembly lived only in the output read the
    /// host makes. The fuzzer found it on `log_warn(reg_splat_f64(..))`.
    #[test]
    fn every_port_type_reads_back_what_was_written() {
        for &ty in PortType::ALL {
            let v = sample(ty);
            assert_eq!(
                v.port_type(),
                crate::compile::marshal::carrier_port(ty),
                "{ty:?}: the sample is not what the port carries"
            );
            let mut scratch: Vec<crate::ast::ScratchBuf> = ty
                .scratch_elem()
                .map(crate::ast::ScratchBuf::new)
                .into_iter()
                .collect();
            let mut slots = [0u64; 2];
            // Written as the macro writes it: by the port's resolved
            // type, which for a narrow or float-in-carrier type is not
            // the port type of the carrier value its `Wire` impl injects.
            write_poly(ty, v.clone(), &mut scratch, &mut slots);
            let back = read_poly(ty, &slots[..ty.slot_width()]);
            assert_eq!(back, v, "{ty:?}");
        }
    }

    /// A size that fits is reserved exactly, and a reused buffer comes
    /// back cleared.
    #[test]
    fn a_feasible_size_is_reserved() {
        let v: Vec<u8> = buffer_for(1000, "probe");
        assert!(v.is_empty() && v.capacity() >= 1000);
        let mut w = vec![7u8; 3];
        reserve_for(&mut w, 64, "probe");
        assert!(w.is_empty() && w.capacity() >= 64);
    }
}
