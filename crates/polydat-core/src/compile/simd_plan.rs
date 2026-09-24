// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Static type and node-variant qualification for scalar-flow SIMD promotion.
//!
//! This module does not rewrite a graph. It establishes the production
//! boundary a rewrite must pass before it may form a packet plan: an explicit
//! scalar-node contract, a supported scalar/register lane shape, pure nodes,
//! and exact register-typed I/O. Final Cranelift compilation remains the
//! authoritative lowering probe.

use std::fmt;

use crate::ast::{Lifecycle, PolydatNode, PortType, Purity};

/// Interpretation of scalar values sharing one physical register lane shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SimdLaneKind {
    /// Unsigned integer lanes.
    UnsignedInteger,
    /// Signed integer lanes.
    SignedInteger,
    /// Floating-point lanes.
    Float,
}

/// A scalar type's fixed 128-bit Polydat register representation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SimdTypeShape {
    /// The scalar type.
    pub scalar: PortType,
    /// The register type that carries a packet of it.
    pub register: PortType,
    /// How the lanes are interpreted.
    pub lane_kind: SimdLaneKind,
    /// Bits per lane.
    pub lane_bits: u8,
    /// Lanes per register.
    pub lanes: u8,
}

impl SimdTypeShape {
    /// Bits per packet: lanes times lane bits.
    pub const fn packet_bits(self) -> u16 {
        self.lane_bits as u16 * self.lanes as u16
    }
}

/// Physically representable fixed-width shapes for the first promotion tier.
///
/// Unsigned integer types deliberately share the corresponding `RegI*x*`
/// physical shape. Add/sub/mul and raw bitwise operations are bit-identical;
/// signedness remains semantic metadata for comparisons, division, shifts,
/// and conversion recipes. A returned shape is necessary, not sufficient:
/// every member node still needs explicit variant metadata and a successful
/// whole-cone backend compilation.
pub const fn promotable_type_shape(scalar: PortType) -> Option<SimdTypeShape> {
    use PortType::*;
    use SimdLaneKind::*;

    let shape = match scalar {
        U8 => SimdTypeShape {
            scalar,
            register: RegI8x16,
            lane_kind: UnsignedInteger,
            lane_bits: 8,
            lanes: 16,
        },
        I8 => SimdTypeShape {
            scalar,
            register: RegI8x16,
            lane_kind: SignedInteger,
            lane_bits: 8,
            lanes: 16,
        },
        U16 => SimdTypeShape {
            scalar,
            register: RegI16x8,
            lane_kind: UnsignedInteger,
            lane_bits: 16,
            lanes: 8,
        },
        I16 => SimdTypeShape {
            scalar,
            register: RegI16x8,
            lane_kind: SignedInteger,
            lane_bits: 16,
            lanes: 8,
        },
        U32 => SimdTypeShape {
            scalar,
            register: RegI32x4,
            lane_kind: UnsignedInteger,
            lane_bits: 32,
            lanes: 4,
        },
        I32 => SimdTypeShape {
            scalar,
            register: RegI32x4,
            lane_kind: SignedInteger,
            lane_bits: 32,
            lanes: 4,
        },
        U64 => SimdTypeShape {
            scalar,
            register: RegI64x2,
            lane_kind: UnsignedInteger,
            lane_bits: 64,
            lanes: 2,
        },
        I64 => SimdTypeShape {
            scalar,
            register: RegI64x2,
            lane_kind: SignedInteger,
            lane_bits: 64,
            lanes: 2,
        },
        F32 => SimdTypeShape {
            scalar,
            register: RegF32x4,
            lane_kind: Float,
            lane_bits: 32,
            lanes: 4,
        },
        F64 => SimdTypeShape {
            scalar,
            register: RegF64x2,
            lane_kind: Float,
            lane_bits: 64,
            lanes: 2,
        },
        // F16X8 exists in Polydat's type plane, but the installed Cranelift
        // x64 path is not a production arithmetic lowering. It enters this
        // table only after a backend probe and operation catalog justify it.
        F16 | U128 | I128 | Bool | Str | Bytes | Json | Ext | Handle | VecF32 | VecF64 | VecF16
        | VecI8 | VecI16 | VecI32 | VecI64 | Reg128 | RegI8x16 | RegI16x8 | RegI32x4 | RegI64x2
        | RegF16x8 | RegF32x4 | RegF64x2 | Dyn => return None,
    };
    Some(shape)
}

/// A scalar node whose declared register variant has passed static shape and
/// semantic validation. It still needs whole-cone Cranelift compilation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ValidatedSimdVariant {
    /// The scalar node's name.
    pub scalar_node: String,
    /// The register node that computes a packet of it.
    pub vector_node: &'static str,
    /// The lane shape.
    pub shape: SimdTypeShape,
    /// Wire inputs the scalar node takes.
    pub wire_inputs: u8,
}

#[derive(Clone, Debug, PartialEq, Eq)]
/// Why a scalar node's declared register variant was not accepted.
pub enum SimdVariantError {
    /// The node declares no register variant.
    Undeclared,
    /// The scalar node is not pure.
    ScalarNodeNotPure,
    /// The variant is not declared exact.
    VariantNotExact,
    /// The variant is not declared total.
    VariantNotTotal,
    /// The variant is not declared lane-independent.
    VariantNotLaneIndependent,
    /// The scalar type has no fixed register shape.
    UnsupportedScalarShape(PortType),
    /// The scalar node's inputs and output are not all of one type.
    ScalarSignatureNotUniform,
    /// The scalar node's ports are not all cycle-lifecycle.
    ScalarLifecycleNotCycle,
    /// The register node is not registered.
    VectorNodeUnavailable(String),
    /// The register node is not pure.
    VectorNodeNotPure,
    /// The register node's signature does not match the scalar node's shape.
    VectorSignatureMismatch,
    /// The register node's ports are not all cycle-lifecycle.
    VectorLifecycleNotCycle,
}

impl fmt::Display for SimdVariantError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Undeclared => f.write_str("scalar node declares no SIMD variant"),
            Self::ScalarNodeNotPure => f.write_str("scalar node is not pure"),
            Self::VariantNotExact => f.write_str("SIMD variant is not exact"),
            Self::VariantNotTotal => f.write_str("SIMD variant is not total"),
            Self::VariantNotLaneIndependent => f.write_str("SIMD variant is not lane-independent"),
            Self::UnsupportedScalarShape(t) => write!(f, "unsupported scalar SIMD shape {t}"),
            Self::ScalarSignatureNotUniform => {
                f.write_str("scalar node inputs and output do not share one scalar lane type")
            }
            Self::ScalarLifecycleNotCycle => {
                f.write_str("scalar node has an init-lifecycle data port")
            }
            Self::VectorNodeUnavailable(e) => write!(f, "register variant unavailable: {e}"),
            Self::VectorNodeNotPure => f.write_str("register variant is not pure"),
            Self::VectorSignatureMismatch => {
                f.write_str("register variant signature does not match the scalar lane shape")
            }
            Self::VectorLifecycleNotCycle => {
                f.write_str("register variant has an init-lifecycle data port")
            }
        }
    }
}

impl std::error::Error for SimdVariantError {}

/// Validate an instantiated scalar node and its declared register node.
///
/// Tier 1 currently accepts uniform element-wise operations: all scalar wire
/// inputs and the sole output have one type, and the vector variant has the
/// same arity under that type's register shape. Broadcast-vs-varying input
/// roles are a property of the eventual cone plan, not the node variant.
pub fn validate_simd_variant(
    scalar: &dyn PolydatNode,
) -> Result<ValidatedSimdVariant, SimdVariantError> {
    let variant = scalar.simd_variant().ok_or(SimdVariantError::Undeclared)?;
    if scalar.purity() != Purity::Pure {
        return Err(SimdVariantError::ScalarNodeNotPure);
    }
    if !variant.exact {
        return Err(SimdVariantError::VariantNotExact);
    }
    if !variant.total {
        return Err(SimdVariantError::VariantNotTotal);
    }
    if !variant.lane_independent {
        return Err(SimdVariantError::VariantNotLaneIndependent);
    }

    let scalar_meta = scalar.meta();
    let [scalar_output] = scalar_meta.outs.as_slice() else {
        return Err(SimdVariantError::ScalarSignatureNotUniform);
    };
    let shape = promotable_type_shape(scalar_output.typ)
        .ok_or(SimdVariantError::UnsupportedScalarShape(scalar_output.typ))?;
    let scalar_inputs = scalar_meta.wire_inputs();
    if scalar_inputs.is_empty() || scalar_inputs.iter().any(|port| port.typ != shape.scalar) {
        return Err(SimdVariantError::ScalarSignatureNotUniform);
    }
    if scalar_output.lifecycle != Lifecycle::Cycle
        || scalar_inputs
            .iter()
            .any(|port| port.lifecycle != Lifecycle::Cycle)
    {
        return Err(SimdVariantError::ScalarLifecycleNotCycle);
    }

    let wires = vec![crate::compile::assembly::WireRef::input("__simd_probe"); scalar_inputs.len()];
    let wire_types = vec![shape.register; scalar_inputs.len()];
    let vector = crate::dsl::factory::build_node(variant.vector_node, &wires, &wire_types, &[])
        .map_err(|e| SimdVariantError::VectorNodeUnavailable(e.to_string()))?;
    if vector.purity() != Purity::Pure {
        return Err(SimdVariantError::VectorNodeNotPure);
    }
    let vector_meta = vector.meta();
    let [vector_output] = vector_meta.outs.as_slice() else {
        return Err(SimdVariantError::VectorSignatureMismatch);
    };
    let vector_inputs = vector_meta.wire_inputs();
    if vector_output.typ != shape.register
        || vector_inputs.len() != scalar_inputs.len()
        || vector_inputs.iter().any(|port| port.typ != shape.register)
    {
        return Err(SimdVariantError::VectorSignatureMismatch);
    }
    if vector_output.lifecycle != Lifecycle::Cycle
        || vector_inputs
            .iter()
            .any(|port| port.lifecycle != Lifecycle::Cycle)
    {
        return Err(SimdVariantError::VectorLifecycleNotCycle);
    }

    Ok(ValidatedSimdVariant {
        scalar_node: scalar_meta.name.clone(),
        vector_node: variant.vector_node,
        shape,
        wire_inputs: scalar_inputs.len() as u8,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn common_numeric_shapes_cover_one_128_bit_word() {
        for ty in [
            PortType::U8,
            PortType::I8,
            PortType::U16,
            PortType::I16,
            PortType::U32,
            PortType::I32,
            PortType::U64,
            PortType::I64,
            PortType::F32,
            PortType::F64,
        ] {
            assert_eq!(promotable_type_shape(ty).unwrap().packet_bits(), 128);
        }
        assert!(promotable_type_shape(PortType::F16).is_none());
        assert!(promotable_type_shape(PortType::U128).is_none());
        assert!(promotable_type_shape(PortType::Str).is_none());
    }
}
