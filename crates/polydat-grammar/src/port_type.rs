// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! The port type vocabulary: every type a wire, a port, a cast, or a
//! declaration can name, with its keyword. What a type means to a
//! compiled buffer (its slot color, width, and scratch element) is
//! the runtime's, defined on this type by `polydat-core`.

use std::fmt;

/// Compile-time type tag for a port on a Polydat node.
///
/// **Narrow types and runtime storage:**
///
/// `PortType` includes narrow integer and float variants (U8/U16/U32,
/// I8/I16/I32, F16/F32) that have no `Value` of their own. At
/// runtime, narrow values are stored in the wide `Value` of their
/// kind, with the assumption that the bits fit:
///
/// - unsigned (`u8`, `u16`, `u32`) → zero-extended in `Value::U64`
/// - signed (`i8`, `i16`, `i32`) → sign-extended in `Value::I64`
/// - `f32` → losslessly widened in `Value::F64` (`f16` rides
///   `Value::U64` as its bit pattern; see [`PortType::F16`])
///
/// The narrow `PortType` variants exist for compile-time type
/// checking and auto-adapter insertion (`U32ToU64`, `F32ToF64`).
/// P2/P3 compiled kernels use flat u64 buffers where this packing
/// is natural. The `Value` enum stays small — no combinatorial
/// explosion of narrow variant types.
///
/// Every input and output port declares its `PortType`. The assembler
/// uses these to validate wiring and auto-insert type adapters (e.g.,
/// `u64 → f64` widening). At runtime, the corresponding `Value`
/// variant is used.
///
/// **Widening rules** (auto-inserted by the assembler):
/// - `U32 → U64`, `I32 → I64`, `F32 → F64` (lossless widening)
/// - `U64 → F64` (lossless for values < 2^53)
/// - `Bool → U64` (true=1, false=0)
/// - Any type → `Str` (via display conversion)
///
/// **Narrowing** is never implicit — use explicit cast functions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PortType {
    /// 64-bit unsigned integer. The primary numeric type.
    U64,
    /// 64-bit IEEE 754 float. Used for math, distributions, noise.
    F64,
    /// 32-bit unsigned integer. Widens to U64 automatically.
    U32,
    /// 32-bit signed integer. Widens to I64 automatically.
    I32,
    /// 64-bit signed integer.
    I64,
    /// 32-bit IEEE 754 float. Widens to F64 automatically.
    F32,
    /// 8-bit unsigned integer (cranelift I8 lane, unsigned
    /// interpretation). Zero-extended in `Value::U64`; widens to
    /// U64 automatically.
    U8,
    /// 8-bit signed integer (cranelift I8 lane, signed
    /// interpretation). Sign-extended in `Value::I64`; widens to
    /// I64 automatically.
    I8,
    /// 16-bit unsigned integer (cranelift I16 lane, unsigned
    /// interpretation). Zero-extended in `Value::U64`; widens to
    /// U64 automatically.
    U16,
    /// 16-bit signed integer (cranelift I16 lane, signed
    /// interpretation). Sign-extended in `Value::I64`; widens to
    /// I64 automatically.
    I16,
    /// 16-bit IEEE 754-2008 binary16 float (cranelift F16).
    /// Carried as its bit pattern in `Value::U64` (low 16 bits),
    /// the same stuffing convention as `F32`; widens to F32/F64
    /// automatically (every f16 is exactly representable in both).
    F16,
    /// 128-bit unsigned integer (cranelift I128, unsigned
    /// interpretation). Real `Value::U128` two-limb carrier — a
    /// 128-bit value cannot ride a 64-bit slot. Rides two
    /// consecutive u64 slots (a limb pair) on the compiled engines.
    U128,
    /// 128-bit signed integer (cranelift I128, signed
    /// interpretation). Same carrier story as `U128`.
    I128,
    /// 128-bit SIMD register word, raw view — the full word as
    /// algorithm-defined buffer state (heterogeneous lane
    /// roles). Free bitcast to/from every lane-typed view.
    Reg128,
    /// Register word viewed as 16 × i8 lanes.
    RegI8x16,
    /// Register word viewed as 8 × i16 lanes.
    RegI16x8,
    /// Register word viewed as 4 × i32 lanes.
    RegI32x4,
    /// Register word viewed as 2 × i64 lanes.
    RegI64x2,
    /// Register word viewed as 8 × f16 lanes.
    RegF16x8,
    /// Register word viewed as 4 × f32 lanes.
    RegF32x4,
    /// Register word viewed as 2 × f64 lanes.
    RegF64x2,
    /// Boolean (true/false). Widens to U64 (1/0).
    Bool,
    /// Heap-allocated string. Any type auto-converts to Str.
    Str,
    /// Raw byte buffer.
    Bytes,
    /// Structured JSON value.
    Json,
    /// Adapter-contributed reflected type (e.g., CQL UUID).
    Ext,
    /// Type-erased Arc handle to a resolved resource (dataset,
    /// prepared statement, ...). The producer node populates an
    /// `Arc<dyn Any + Send + Sync>`; the consumer node downcasts to
    /// the concrete type via `Value::as_handle::<T>()`.
    Handle,
    /// Typed `f32` vector slice (`Arc<[f32]>`). Bound natively by
    /// adapters that understand `[f32]` (CQL `vector<float, N>`).
    VecF32,
    /// Typed `i32` vector slice (`Arc<[i32]>`).
    VecI32,
    /// Typed `f64` vector slice (`Arc<[f64]>`). Bound natively
    /// for CQL `vector<double, N>`.
    VecF64,
    /// Typed `i64` vector slice (`Arc<[i64]>`). Bound natively
    /// for CQL `vector<bigint, N>`.
    VecI64,
    /// Typed half-precision float vector (`Arc<[half::f16]>`).
    /// Bound natively for CQL `vector<half_float, N>`-style
    /// columns; stays at f16 on the wire so embeddings stored
    /// as 16-bit floats don't widen to f32 at the boundary.
    VecF16,
    /// Typed `i16` vector slice (`Arc<[i16]>`). Bound natively
    /// for CQL `vector<smallint, N>`.
    VecI16,
    /// Typed `i8` vector slice (`Arc<[i8]>`). Completes the
    /// cranelift lane family; CQL `vector<tinyint, N>`.
    VecI8,
    /// Any value, as written: the slot of an input whose type may vary
    /// over a kernel's lifetime (input_variance.md). Only a converter
    /// node reads it, turning the value into the type its consumers
    /// read; no other port has this type, and no value is typed `Dyn`.
    Dyn,
}

impl fmt::Display for PortType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PortType::U64 => write!(f, "u64"),
            PortType::F64 => write!(f, "f64"),
            PortType::U32 => write!(f, "u32"),
            PortType::I32 => write!(f, "i32"),
            PortType::I64 => write!(f, "i64"),
            PortType::F32 => write!(f, "f32"),
            PortType::U8 => write!(f, "u8"),
            PortType::I8 => write!(f, "i8"),
            PortType::U16 => write!(f, "u16"),
            PortType::I16 => write!(f, "i16"),
            PortType::F16 => write!(f, "f16"),
            PortType::U128 => write!(f, "u128"),
            PortType::I128 => write!(f, "i128"),
            PortType::Reg128 => write!(f, "reg128"),
            PortType::RegI8x16 => write!(f, "reg_i8x16"),
            PortType::RegI16x8 => write!(f, "reg_i16x8"),
            PortType::RegI32x4 => write!(f, "reg_i32x4"),
            PortType::RegI64x2 => write!(f, "reg_i64x2"),
            PortType::RegF16x8 => write!(f, "reg_f16x8"),
            PortType::RegF32x4 => write!(f, "reg_f32x4"),
            PortType::RegF64x2 => write!(f, "reg_f64x2"),
            PortType::Bool => write!(f, "bool"),
            PortType::Str => write!(f, "String"),
            PortType::Bytes => write!(f, "bytes"),
            PortType::Json => write!(f, "json"),
            PortType::Ext => write!(f, "ext"),
            PortType::Handle => write!(f, "handle"),
            PortType::VecF32 => write!(f, "vec_f32"),
            PortType::VecI32 => write!(f, "vec_i32"),
            PortType::VecF64 => write!(f, "vec_f64"),
            PortType::VecI64 => write!(f, "vec_i64"),
            PortType::VecF16 => write!(f, "vec_f16"),
            PortType::VecI16 => write!(f, "vec_i16"),
            PortType::VecI8 => write!(f, "vec_i8"),
            PortType::Dyn => write!(f, "dyn"),
        }
    }
}

/// The set of numbers a scalar port type can carry, described by the
/// properties that decide whether one of them holds every value of
/// another: how many bits the representation has, and how it spends
/// them.
///
/// `Bool` is the one-bit unsigned domain, which is what the type
/// system already says of it — it widens to `U64` as 1 and 0.
/// Non-scalar types (`Str`, `Bytes`, `Json`, the vectors, the
/// register views, `Ext`, `Handle`) have no domain.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum NumericDomain {
    /// Non-negative integers in `0 ..= 2^bits - 1`.
    Unsigned {
        /// Width of the representation.
        bits: u32,
    },
    /// Two's-complement integers, one of whose bits is the sign.
    Signed {
        /// Width of the representation, sign bit included.
        bits: u32,
    },
    /// An IEEE 754 binary float.
    Float {
        /// Significand bits, the implicit leading one included: the
        /// largest integer represented exactly is `2^mantissa`.
        mantissa: u32,
        /// Exponent bits, which fix the magnitude range.
        exponent: u32,
    },
}

impl NumericDomain {
    /// Whether every value of this domain is a value of `other` —
    /// that is, whether a conversion into `other` is lossless.
    ///
    /// Integers fit by counting the bits each spends on magnitude: an
    /// unsigned domain needs a signed one strictly wider, a signed
    /// domain never fits an unsigned one. An integer fits a float
    /// when its magnitude bits fit the float's significand, which is
    /// why `U64 → F64` does not: 64 magnitude bits do not fit 53, and
    /// the values above `2^53` round. A float fits a wider float when
    /// both its significand and its exponent do.
    pub fn fits_in(self, other: Self) -> bool {
        use NumericDomain::{Float, Signed, Unsigned};
        match (self, other) {
            (Unsigned { bits: a }, Unsigned { bits: b }) => a <= b,
            (Unsigned { bits: a }, Signed { bits: b }) => a < b,
            (Signed { bits: a }, Signed { bits: b }) => a <= b,
            (Signed { .. }, Unsigned { .. }) => false,
            (Unsigned { bits }, Float { mantissa, .. }) => bits <= mantissa,
            (Signed { bits }, Float { mantissa, .. }) => bits - 1 <= mantissa,
            (Float { .. }, Unsigned { .. } | Signed { .. }) => false,
            (
                Float {
                    mantissa: m1,
                    exponent: e1,
                },
                Float {
                    mantissa: m2,
                    exponent: e2,
                },
            ) => m1 <= m2 && e1 <= e2,
        }
    }
}

impl PortType {
    /// Every port type, once.
    ///
    /// A test that must hold for *all* types can walk this rather than
    /// name the ones its author thought of. That is the difference
    /// between a mapping that is complete and one that is complete so
    /// far: the value↔slot writers were extended three times by finding
    /// a type they had missed at run time, each found by a program that
    /// happened to use it (2026-09-22).
    ///
    /// [`Self::every_variant_is_listed`] keeps this honest — it is an
    /// exhaustive `match`, so adding a variant without adding it here
    /// fails to compile rather than quietly shrinking every sweep that
    /// walks this list.
    pub const ALL: &'static [PortType] = &[
        PortType::U64,
        PortType::F64,
        PortType::U32,
        PortType::I32,
        PortType::I64,
        PortType::F32,
        PortType::U8,
        PortType::I8,
        PortType::U16,
        PortType::I16,
        PortType::F16,
        PortType::U128,
        PortType::I128,
        PortType::Reg128,
        PortType::RegI8x16,
        PortType::RegI16x8,
        PortType::RegI32x4,
        PortType::RegI64x2,
        PortType::RegF16x8,
        PortType::RegF32x4,
        PortType::RegF64x2,
        PortType::Bool,
        PortType::Str,
        PortType::Bytes,
        PortType::Json,
        PortType::Ext,
        PortType::Handle,
        PortType::VecF32,
        PortType::VecI32,
        PortType::VecF64,
        PortType::VecI64,
        PortType::VecF16,
        PortType::VecI16,
        PortType::VecI8,
        PortType::Dyn,
    ];

    /// `true` for every variant, by an exhaustive match: the compiler
    /// refuses this function when a variant is added, and the test
    /// beside it checks [`Self::ALL`] carries the one that was added.
    #[doc(hidden)]
    pub fn every_variant_is_listed(self) -> bool {
        match self {
            PortType::U64
            | PortType::F64
            | PortType::U32
            | PortType::I32
            | PortType::I64
            | PortType::F32
            | PortType::U8
            | PortType::I8
            | PortType::U16
            | PortType::I16
            | PortType::F16
            | PortType::U128
            | PortType::I128
            | PortType::Reg128
            | PortType::RegI8x16
            | PortType::RegI16x8
            | PortType::RegI32x4
            | PortType::RegI64x2
            | PortType::RegF16x8
            | PortType::RegF32x4
            | PortType::RegF64x2
            | PortType::Bool
            | PortType::Str
            | PortType::Bytes
            | PortType::Json
            | PortType::Ext
            | PortType::Handle
            | PortType::VecF32
            | PortType::VecI32
            | PortType::VecF64
            | PortType::VecI64
            | PortType::VecF16
            | PortType::VecI16
            | PortType::VecI8
            | PortType::Dyn => Self::ALL.contains(&self),
        }
    }

    /// The numbers this type can carry, for the types that carry
    /// numbers. `None` for every other type.
    ///
    /// This is what decides whether a conversion between two types
    /// keeps the value, so that the answer is read off the types
    /// themselves rather than kept in a list of pairs beside them.
    pub fn numeric_domain(self) -> Option<NumericDomain> {
        use NumericDomain::{Float, Signed, Unsigned};
        Some(match self {
            Self::Bool => Unsigned { bits: 1 },
            Self::U8 => Unsigned { bits: 8 },
            Self::U16 => Unsigned { bits: 16 },
            Self::U32 => Unsigned { bits: 32 },
            Self::U64 => Unsigned { bits: 64 },
            Self::U128 => Unsigned { bits: 128 },
            Self::I8 => Signed { bits: 8 },
            Self::I16 => Signed { bits: 16 },
            Self::I32 => Signed { bits: 32 },
            Self::I64 => Signed { bits: 64 },
            Self::I128 => Signed { bits: 128 },
            Self::F16 => Float {
                mantissa: 11,
                exponent: 5,
            },
            Self::F32 => Float {
                mantissa: 24,
                exponent: 8,
            },
            Self::F64 => Float {
                mantissa: 53,
                exponent: 11,
            },
            Self::Str
            | Self::Bytes
            | Self::Json
            | Self::Ext
            | Self::Handle
            | Self::Reg128
            | Self::RegI8x16
            | Self::RegI16x8
            | Self::RegI32x4
            | Self::RegI64x2
            | Self::RegF16x8
            | Self::RegF32x4
            | Self::RegF64x2
            | Self::VecF32
            | Self::VecI32
            | Self::VecF64
            | Self::VecI64
            | Self::VecF16
            | Self::VecI16
            | Self::VecI8
            | Self::Dyn => return None,
        })
    }

    /// The canonical lowercase keyword for this `PortType`.
    ///
    /// This is the single source of truth for the str↔PortType
    /// mapping used by every synthesizer and parser in the
    /// workspace — synthesized polydat source (`extern <name>:
    /// <keyword>`), the workload-author `{name:<keyword>}` lvalue
    /// spec, and reverse parsing via [`Self::from_keyword`].
    /// Inverse of [`Self::from_keyword`].
    ///
    /// Exhaustive over the enum — adding a new `PortType` variant
    /// is a compile error here, forcing the addition of its
    /// canonical keyword and the round-trip closure to update.
    pub fn to_keyword(&self) -> &'static str {
        match self {
            Self::U64 => "u64",
            Self::F64 => "f64",
            Self::U32 => "u32",
            Self::I32 => "i32",
            Self::I64 => "i64",
            Self::F32 => "f32",
            Self::U8 => "u8",
            Self::I8 => "i8",
            Self::U16 => "u16",
            Self::I16 => "i16",
            Self::F16 => "f16",
            Self::U128 => "u128",
            Self::I128 => "i128",
            Self::Reg128 => "reg128",
            Self::RegI8x16 => "reg_i8x16",
            Self::RegI16x8 => "reg_i16x8",
            Self::RegI32x4 => "reg_i32x4",
            Self::RegI64x2 => "reg_i64x2",
            Self::RegF16x8 => "reg_f16x8",
            Self::RegF32x4 => "reg_f32x4",
            Self::RegF64x2 => "reg_f64x2",
            Self::Bool => "bool",
            Self::Str => "str",
            Self::Bytes => "bytes",
            Self::Json => "json",
            Self::Ext => "ext",
            Self::Handle => "handle",
            Self::VecF32 => "vec_f32",
            Self::VecI32 => "vec_i32",
            Self::VecF64 => "vec_f64",
            Self::VecI64 => "vec_i64",
            Self::VecF16 => "vec_f16",
            Self::VecI16 => "vec_i16",
            Self::VecI8 => "vec_i8",
            Self::Dyn => "dyn",
        }
    }

    /// Parse a polydat type keyword into a `PortType`.
    ///
    /// Inverse of [`Self::to_keyword`]: accepts every keyword
    /// that `to_keyword` emits, plus a small set of legacy aliases
    /// (`"String"`, `"Json"`, `"Ext"`) that survive in older
    /// hand-written workload source. Returns `None` for any
    /// unrecognized keyword so callers can surface a loud
    /// diagnostic rather than silently coercing to a default.
    ///
    /// Used by the DSL `extern <name>: <keyword>` parser
    /// (`polydat-core/src/dsl/compile.rs`). Round-trips cleanly with
    /// any source `to_keyword` emits.
    pub fn from_keyword(name: &str) -> Option<Self> {
        match name {
            "u64" => Some(Self::U64),
            "f64" => Some(Self::F64),
            "u32" => Some(Self::U32),
            "i32" => Some(Self::I32),
            "i64" => Some(Self::I64),
            "f32" => Some(Self::F32),
            "u8" => Some(Self::U8),
            "i8" => Some(Self::I8),
            "u16" => Some(Self::U16),
            "i16" => Some(Self::I16),
            "f16" => Some(Self::F16),
            "u128" => Some(Self::U128),
            "i128" => Some(Self::I128),
            "reg128" => Some(Self::Reg128),
            "reg_i8x16" => Some(Self::RegI8x16),
            "reg_i16x8" => Some(Self::RegI16x8),
            "reg_i32x4" => Some(Self::RegI32x4),
            "reg_i64x2" => Some(Self::RegI64x2),
            "reg_f16x8" => Some(Self::RegF16x8),
            "reg_f32x4" => Some(Self::RegF32x4),
            "reg_f64x2" => Some(Self::RegF64x2),
            "bool" => Some(Self::Bool),
            "str" | "Str" | "String" => Some(Self::Str),
            "bytes" => Some(Self::Bytes),
            "json" | "Json" => Some(Self::Json),
            "ext" | "Ext" => Some(Self::Ext),
            "handle" => Some(Self::Handle),
            "vec_f32" => Some(Self::VecF32),
            "vec_i32" => Some(Self::VecI32),
            "vec_f64" => Some(Self::VecF64),
            "vec_i64" => Some(Self::VecI64),
            "vec_f16" => Some(Self::VecF16),
            "vec_i16" => Some(Self::VecI16),
            "vec_i8" => Some(Self::VecI8),
            "dyn" => Some(Self::Dyn),
            _ => None,
        }
    }

    /// Workload-author-facing parser for the `{name:<keyword>}`
    /// lvalue-spec surface. Strict subset of [`Self::from_keyword`]
    /// — `handle` and `ext` are rejected because they're
    /// internal-only types a workload author should never assert.
    ///
    /// Returns `None` for any unrecognized name; the caller
    /// surfaces the unknown spec as a workload-shape diagnostic.
    pub fn from_workload_name(name: &str) -> Option<Self> {
        match Self::from_keyword(name)? {
            Self::Handle | Self::Ext => None,
            pt => Some(pt),
        }
    }
}
