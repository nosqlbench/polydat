// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! End-to-end Tier-1 scalar-flow SIMD promotion.
//!
//! This module deliberately implements only the sealed perfect-ordinal slice:
//! one selected scalar output, one `u64` ordinal input that varies inside an
//! owned source lease, pure exact total lane-independent member nodes, and
//! scope-stable scalar broadcasts. The canonical scalar DAG is retained as the
//! oracle and fragment/recovery path; the synthesized register DAG is a
//! parallel execution plan, not a mutation of scalar graph semantics.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fmt;
use std::sync::Arc;

use crate::ast::{Bits128, PortType, Purity, Value};
use crate::compile::assembly::{PolydatAssembler, ResolvedDag, WireRef};
use crate::compile::jit::JitKernelRaw;
use crate::compile::jit::host_isa::EffectiveIsa;
use crate::compile::simd_plan::{SimdTypeShape, validate_simd_variant};
use crate::iteration::simd_ordinal::{OrdinalLaneClock, OrdinalPacketStamp};
use crate::iteration::source::OrdinalBatchLease;
use crate::kernel::{InputKind, PolydatKernel, PolydatProgram, WireSource};

const TIER1_U64_LANES: usize = 2;
const TIER1_MIN_MEMBERS: usize = 2;

#[derive(Clone, Debug, PartialEq, Eq)]
/// Why a Tier-1 SIMD plan could not be built, started, or advanced;
/// `Display` renders each with its names.
pub enum Tier1SimdError {
    /// The selected output does not exist.
    OutputNotFound(String),
    /// The driving input does not exist.
    DrivingInputNotFound(String),
    /// The selected output does not depend on the driving input.
    DrivingInputNotUsed(String),
    /// The selected node has more than one output port.
    OutputPortUnsupported,
    /// Fewer nodes could be promoted than the plan requires.
    TooFewMembers {
        /// Promotable nodes found.
        found: usize,
        /// The minimum required.
        minimum: usize,
    },
    /// A node in the cone cannot be promoted.
    NodeIneligible {
        /// The node.
        node: String,
        /// Why it cannot be promoted.
        reason: String,
    },
    /// A node changes the lane shape.
    ShapeMismatch {
        /// The node.
        node: String,
    },
    /// A node cannot serve as a packet boundary.
    BoundaryUnsupported {
        /// The node.
        node: String,
    },
    /// A boundary does not carry the scalar type the plan requires.
    BoundaryTypeMismatch {
        /// The boundary's name.
        boundary: String,
        /// The scalar type required.
        expected: PortType,
    },
    /// An input other than the driving ordinal is externally writable.
    MutableBroadcast(String),
    /// A node accepts `None` inputs, for which packets have no validity semantics.
    NoneAwareNode(String),
    /// The output has a visibility modifier and cannot be packetized speculatively.
    OutputModifierUnsupported(String),
    /// The output is an init binding, not a scalar stream.
    ConstOutputUnsupported(String),
    /// The lease driver supports `u64` over two `i64` lanes only, not this type.
    RuntimeShapeUnsupported(PortType),
    /// A constant boundary node could not be evaluated safely.
    ConstantEvaluationFailed(String),
    /// The register-typed graph could not be built.
    VectorGraphBuild(String),
    /// The effective native ISA declined the graph.
    VectorCompilation(String),
    /// Native ISA detection failed.
    CapabilityDetection(String),
    /// The previous ordinal lease is not fully drained.
    ActiveLease,
    /// The lease targets an input other than the plan's driving input.
    LeaseInputMismatch {
        /// The plan's driving input index.
        expected: usize,
        /// The lease's input index.
        got: usize,
    },
    /// The lease arrived out of reservation order.
    LeaseSequenceMismatch {
        /// The sequence expected.
        expected: u64,
        /// The sequence received.
        got: u64,
    },
    /// The lease's stream or generation changed within one activation.
    LeaseStreamMismatch,
    /// The activation epoch moved backwards.
    ActivationWentBackwards {
        /// The epoch before.
        previous: u64,
        /// The epoch received.
        got: u64,
    },
    /// The name is not a scope-stable broadcast input of the plan.
    UnknownBroadcast(String),
    /// A broadcast value has the wrong type.
    BroadcastTypeMismatch {
        /// The broadcast's name.
        name: String,
        /// The type expected.
        expected: PortType,
        /// The type received.
        got: PortType,
    },
}

impl fmt::Display for Tier1SimdError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OutputNotFound(name) => write!(f, "SIMD output '{name}' does not exist"),
            Self::DrivingInputNotFound(name) => {
                write!(f, "SIMD driving input '{name}' does not exist")
            }
            Self::DrivingInputNotUsed(name) => {
                write!(
                    f,
                    "selected output does not depend on driving input '{name}'"
                )
            }
            Self::OutputPortUnsupported => {
                f.write_str("Tier-1 SIMD requires the selected node's sole output port")
            }
            Self::TooFewMembers { found, minimum } => write!(
                f,
                "Tier-1 SIMD requires at least {minimum} promoted nodes, found {found}"
            ),
            Self::NodeIneligible { node, reason } => {
                write!(f, "node '{node}' is not SIMD-promotable: {reason}")
            }
            Self::ShapeMismatch { node } => {
                write!(f, "node '{node}' changes the SIMD lane shape")
            }
            Self::BoundaryUnsupported { node } => {
                write!(
                    f,
                    "node '{node}' cannot be used as a Tier-1 packet boundary"
                )
            }
            Self::BoundaryTypeMismatch { boundary, expected } => write!(
                f,
                "SIMD boundary '{boundary}' does not carry the required {expected} scalar type"
            ),
            Self::MutableBroadcast(name) => write!(
                f,
                "input '{name}' is externally writable; only the driving ordinal may change inside a lease"
            ),
            Self::NoneAwareNode(node) => write!(
                f,
                "node '{node}' accepts None inputs and needs explicit packet validity semantics"
            ),
            Self::OutputModifierUnsupported(name) => write!(
                f,
                "output '{name}' has a visibility modifier and cannot be speculatively packetized"
            ),
            Self::ConstOutputUnsupported(name) => write!(
                f,
                "output '{name}' is an init binding and cannot be advanced as a scalar stream"
            ),
            Self::RuntimeShapeUnsupported(typ) => write!(
                f,
                "the landed Tier-1 lease driver supports u64/RegI64x2, not {typ}"
            ),
            Self::ConstantEvaluationFailed(node) => {
                write!(
                    f,
                    "constant boundary node '{node}' could not be evaluated safely"
                )
            }
            Self::VectorGraphBuild(reason) => {
                write!(f, "failed to build the register-typed SIMD graph: {reason}")
            }
            Self::VectorCompilation(reason) => {
                write!(
                    f,
                    "the effective native ISA declined the SIMD graph: {reason}"
                )
            }
            Self::CapabilityDetection(reason) => {
                write!(f, "native ISA detection failed: {reason}")
            }
            Self::ActiveLease => f.write_str("the previous ordinal lease is not fully drained"),
            Self::LeaseInputMismatch { expected, got } => write!(
                f,
                "ordinal lease targets input {got}, but the SIMD plan drives input {expected}"
            ),
            Self::LeaseSequenceMismatch { expected, got } => write!(
                f,
                "ordinal lease arrived out of reservation order: expected {expected}, got {got}"
            ),
            Self::LeaseStreamMismatch => {
                f.write_str("ordinal lease stream/generation changed within one activation")
            }
            Self::ActivationWentBackwards { previous, got } => write!(
                f,
                "activation epoch moved backwards from {previous} to {got}"
            ),
            Self::UnknownBroadcast(name) => {
                write!(
                    f,
                    "'{name}' is not a scope-stable broadcast input of this SIMD plan"
                )
            }
            Self::BroadcastTypeMismatch {
                name,
                expected,
                got,
            } => write!(f, "broadcast '{name}' expects {expected}, got {got}"),
        }
    }
}

impl std::error::Error for Tier1SimdError {}

/// Recoverable lease-start failure. Ownership of the reserved ordinal range is
/// returned to the caller so it can be processed through another scalar path.
#[derive(Debug)]
pub struct Tier1LeaseStartError {
    reason: Tier1SimdError,
    lease: OrdinalBatchLease,
}

impl Tier1LeaseStartError {
    /// Why the lease could not start.
    pub const fn reason(&self) -> &Tier1SimdError {
        &self.reason
    }

    /// The reason and the lease, whose ordinal range is the caller's again.
    pub fn into_parts(self) -> (Tier1SimdError, OrdinalBatchLease) {
        (self.reason, self.lease)
    }
}

impl fmt::Display for Tier1LeaseStartError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.reason.fmt(f)
    }
}

impl std::error::Error for Tier1LeaseStartError {}

#[derive(Clone, Debug, PartialEq, Eq)]
/// What a Tier-1 SIMD plan promotes: the output, the driving input, the
/// lane shape, the member nodes, and the broadcast inputs.
pub struct Tier1SimdDescriptor {
    /// The output the plan computes.
    pub output: String,
    /// The input whose ordinals the packets stride.
    pub driving_input: String,
    /// The scalar type of the stream.
    pub scalar_type: PortType,
    /// The register type a packet carries.
    pub register_type: PortType,
    /// Lanes per packet.
    pub lanes: u8,
    /// The nodes promoted into the vector graph.
    pub member_nodes: Vec<String>,
    /// Scope-stable inputs broadcast across the lanes.
    pub broadcast_inputs: Vec<String>,
    /// The native ISA the plan was compiled for.
    pub effective_isa_fingerprint: String,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
/// Counters of a plan's execution.
pub struct Tier1SimdStats {
    /// Ordinal leases drained to completion.
    pub leases_completed: u64,
    /// Packets computed by the vector graph.
    pub vector_packets: u64,
    /// Lanes of a partial packet, computed one at a time.
    pub scalar_fragment_lanes: u64,
    /// Lanes computed one at a time after a lease fell back to the scalar path.
    pub scalar_recovery_lanes: u64,
    /// Values handed out.
    pub values_drained: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum BoundaryKey {
    Input(usize),
    Constant(usize),
}

#[derive(Clone, Debug)]
enum BoundaryValue {
    Driving,
    Broadcast { input_index: usize, name: String },
    Constant { value: Value },
}

struct ReadyPacket {
    stamp: OrdinalPacketStamp,
    lanes: [Value; TIER1_U64_LANES],
}

struct ActiveLease {
    lease: OrdinalBatchLease,
    activation_epoch: u64,
    next_ordinal: u64,
    scalar_only: bool,
    ready: Option<ReadyPacket>,
}

/// Per-fiber Tier-1 executor. The scalar and native register kernels share no
/// mutable state with another fiber.
pub struct Tier1SimdExecutor {
    descriptor: Tier1SimdDescriptor,
    shape: SimdTypeShape,
    driving_input_index: usize,
    scalar: PolydatKernel,
    vector: JitKernelRaw,
    vector_output_slot: usize,
    boundaries: Vec<BoundaryValue>,
    active: Option<ActiveLease>,
    activation_epoch: Option<u64>,
    expected_lease_sequence: u64,
    stream_identity: Option<(u64, u64)>,
    stats: Tier1SimdStats,
}

impl Tier1SimdExecutor {
    /// The plan's description.
    pub fn descriptor(&self) -> &Tier1SimdDescriptor {
        &self.descriptor
    }

    /// The counters so far.
    pub const fn stats(&self) -> Tier1SimdStats {
        self.stats
    }

    /// The scalar program the plan falls back to.
    pub fn scalar_program(&self) -> &Arc<PolydatProgram> {
        self.scalar.program()
    }

    /// Whether a lease is open.
    pub const fn has_active_lease(&self) -> bool {
        self.active.is_some()
    }

    /// Values left in the open lease, or zero.
    pub fn remaining(&self) -> u64 {
        self.active
            .as_ref()
            .map(|active| active.lease.range().end - active.next_ordinal)
            .unwrap_or(0)
    }

    /// Freeze one scope-stable input value for subsequent packet evaluation.
    /// Rebinding during an owned lease is rejected rather than invalidating
    /// already-reserved logical inputs.
    pub fn set_broadcast(&mut self, name: &str, value: Value) -> Result<(), Tier1SimdError> {
        if self.active.is_some() {
            return Err(Tier1SimdError::ActiveLease);
        }
        let Some(input_index) = self.boundaries.iter().find_map(|boundary| match boundary {
            BoundaryValue::Broadcast {
                input_index,
                name: candidate,
            } if candidate == name => Some(*input_index),
            _ => None,
        }) else {
            return Err(Tier1SimdError::UnknownBroadcast(name.to_string()));
        };
        if !value.satisfies_slot(self.shape.scalar) || value.port_type() != self.shape.scalar {
            return Err(Tier1SimdError::BroadcastTypeMismatch {
                name: name.to_string(),
                expected: self.shape.scalar,
                got: value.port_type(),
            });
        }
        self.scalar.state().set_input(input_index, value);
        Ok(())
    }

    /// Accept an owned source reservation. A rejection returns the lease in
    /// [`Tier1LeaseStartError`] so the caller can scalar-process it.
    pub fn begin_lease(
        &mut self,
        lease: OrdinalBatchLease,
        activation_epoch: u64,
    ) -> Result<(), Tier1LeaseStartError> {
        let validation = self.validate_lease(&lease, activation_epoch);
        if let Err(reason) = validation {
            return Err(Tier1LeaseStartError { reason, lease });
        }

        if self.activation_epoch != Some(activation_epoch) {
            self.activation_epoch = Some(activation_epoch);
            self.expected_lease_sequence = 0;
            self.stream_identity = None;
        }
        self.stream_identity = Some((lease.stream_id(), lease.source_generation()));
        let next_ordinal = lease.range().start;
        self.active = Some(ActiveLease {
            lease,
            activation_epoch,
            next_ordinal,
            scalar_only: false,
            ready: None,
        });
        Ok(())
    }

    fn validate_lease(
        &self,
        lease: &OrdinalBatchLease,
        activation_epoch: u64,
    ) -> Result<(), Tier1SimdError> {
        if self.active.is_some() {
            return Err(Tier1SimdError::ActiveLease);
        }
        if lease.input_index() != self.driving_input_index {
            return Err(Tier1SimdError::LeaseInputMismatch {
                expected: self.driving_input_index,
                got: lease.input_index(),
            });
        }
        if let Some(previous) = self.activation_epoch {
            if activation_epoch < previous {
                return Err(Tier1SimdError::ActivationWentBackwards {
                    previous,
                    got: activation_epoch,
                });
            }
            if activation_epoch == previous {
                if lease.sequence() != self.expected_lease_sequence {
                    return Err(Tier1SimdError::LeaseSequenceMismatch {
                        expected: self.expected_lease_sequence,
                        got: lease.sequence(),
                    });
                }
                if let Some(identity) = self.stream_identity
                    && identity != (lease.stream_id(), lease.source_generation())
                {
                    return Err(Tier1SimdError::LeaseStreamMismatch);
                }
            } else if lease.sequence() != 0 {
                return Err(Tier1SimdError::LeaseSequenceMismatch {
                    expected: 0,
                    got: lease.sequence(),
                });
            }
        } else if lease.sequence() != 0 {
            return Err(Tier1SimdError::LeaseSequenceMismatch {
                expected: 0,
                got: lease.sequence(),
            });
        }
        Ok(())
    }

    /// Discard an uncommitted register packet and process the remainder of the
    /// active lease through the retained scalar graph. The consumer frontier
    /// does not move and the shared source cursor is not touched.
    pub fn force_scalar_recovery(&mut self) {
        if let Some(active) = self.active.as_mut() {
            active.ready = None;
            active.scalar_only = true;
        }
    }

    /// Drain committed scalar values in ordinal order. Calls may use arbitrary
    /// burst sizes; a partly consumed register packet remains in `BatchState`.
    pub fn drain_into(&mut self, output: &mut [Value]) -> usize {
        if output.is_empty() || self.active.is_none() {
            return 0;
        }

        let mut written = 0;
        while written < output.len() && self.active.is_some() {
            let needs_packet = {
                let active = self.active.as_ref().expect("checked above");
                active
                    .ready
                    .as_ref()
                    .is_none_or(|ready| !ready.stamp.contains(active.next_ordinal))
            };
            if needs_packet {
                self.materialize_packet();
            }

            let mut completed = false;
            {
                let active = self.active.as_mut().expect("packet materialized");
                let ready = active.ready.as_ref().expect("packet materialized");
                let lane = (active.next_ordinal - ready.stamp.base_ordinal) as usize;
                output[written] = ready.lanes[lane].clone();
                written += 1;
                self.stats.values_drained += 1;
                active.next_ordinal += 1;

                if active.next_ordinal >= ready.stamp.base_ordinal + self.shape.lanes as u64 {
                    active.ready = None;
                }
                if active.next_ordinal >= active.lease.range().end {
                    completed = true;
                }
            }

            if completed {
                self.active = None;
                self.expected_lease_sequence = self.expected_lease_sequence.wrapping_add(1);
                self.stats.leases_completed += 1;
            }
        }
        written
    }

    fn materialize_packet(&mut self) {
        let (
            base_ordinal,
            range,
            consumer_frontier,
            scalar_only,
            activation_epoch,
            stream_id,
            generation,
        ) = {
            let active = self.active.as_ref().expect("active lease");
            (
                OrdinalLaneClock::<TIER1_U64_LANES>::packet_base(active.next_ordinal),
                active.lease.range(),
                active.next_ordinal,
                active.scalar_only,
                active.activation_epoch,
                active.lease.stream_id(),
                active.lease.source_generation(),
            )
        };
        let mut valid_mask = 0u16;
        for lane in 0..TIER1_U64_LANES {
            let ordinal = base_ordinal + lane as u64;
            if ordinal >= range.start && ordinal >= consumer_frontier && ordinal < range.end {
                valid_mask |= 1 << lane;
            }
        }

        let full_mask = (1u16 << TIER1_U64_LANES) - 1;
        let lanes = if valid_mask == full_mask && !scalar_only {
            self.stats.vector_packets += 1;
            self.eval_vector_packet(base_ordinal)
        } else {
            let mut lanes = [Value::U64(0), Value::U64(0)];
            for (lane, slot) in lanes.iter_mut().enumerate() {
                if valid_mask & (1 << lane) != 0 {
                    *slot = self.eval_scalar_ordinal(base_ordinal + lane as u64);
                    if scalar_only {
                        self.stats.scalar_recovery_lanes += 1;
                    } else {
                        self.stats.scalar_fragment_lanes += 1;
                    }
                }
            }
            lanes
        };

        self.active.as_mut().expect("active lease").ready = Some(ReadyPacket {
            stamp: OrdinalPacketStamp {
                stream_id,
                source_generation: generation,
                activation_epoch,
                dependency_epoch: 0,
                base_ordinal,
                lane_count: TIER1_U64_LANES as u8,
                valid_mask,
            },
            lanes,
        });
    }

    fn eval_vector_packet(&mut self, base_ordinal: u64) -> [Value; TIER1_U64_LANES] {
        let mut coords = Vec::with_capacity(self.boundaries.len() * 2);
        for boundary in &self.boundaries {
            let bits = match boundary {
                BoundaryValue::Driving => {
                    Bits128::from_lanes_i64([base_ordinal as i64, (base_ordinal + 1) as i64])
                }
                BoundaryValue::Broadcast { input_index, .. } => {
                    let value = self.scalar.state_ref().get_input(*input_index).as_u64();
                    Bits128::from_lanes_i64([value as i64; TIER1_U64_LANES])
                }
                BoundaryValue::Constant { value } => {
                    let value = value.as_u64();
                    Bits128::from_lanes_i64([value as i64; TIER1_U64_LANES])
                }
            };
            coords.extend_from_slice(&bits.0);
        }
        self.vector.eval(&coords);
        let bits = Bits128([
            self.vector.get_slot(self.vector_output_slot),
            self.vector.get_slot(self.vector_output_slot + 1),
        ]);
        bits.lanes_i64().map(|lane| Value::U64(lane as u64))
    }

    fn eval_scalar_ordinal(&mut self, ordinal: u64) -> Value {
        self.scalar
            .state()
            .set_input(self.driving_input_index, Value::U64(ordinal));
        self.scalar.pull(&self.descriptor.output).clone()
    }
}

fn is_input_alias(resolved: &ResolvedDag, node_idx: usize) -> Option<usize> {
    let node = &resolved.nodes[node_idx];
    let meta = node.meta();
    if !meta.name.starts_with("__port_") || meta.wire_inputs().len() != 1 || meta.outs.len() != 1 {
        return None;
    }
    match resolved.wiring[node_idx].as_slice() {
        [WireSource::Input(input_idx)] => Some(*input_idx),
        _ => None,
    }
}

fn is_constant_boundary(resolved: &ResolvedDag, node_idx: usize) -> bool {
    resolved.wiring[node_idx].is_empty()
        && resolved.nodes[node_idx].meta().wire_inputs().is_empty()
        && resolved.nodes[node_idx].meta().outs.len() == 1
        && resolved.nodes[node_idx].purity() == Purity::Pure
}

fn evaluate_constant_boundary(
    resolved: &ResolvedDag,
    node_idx: usize,
) -> Result<Value, Tier1SimdError> {
    let node = &resolved.nodes[node_idx];
    let mut output = [Value::None];
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        node.eval(&[], &mut output);
    }))
    .map_err(|_| Tier1SimdError::ConstantEvaluationFailed(node.meta().name.clone()))?;
    Ok(output[0].clone())
}

fn boundary_key_for_source(resolved: &ResolvedDag, source: &WireSource) -> Option<BoundaryKey> {
    match source {
        WireSource::Input(input_idx) => Some(BoundaryKey::Input(*input_idx)),
        WireSource::NodeOutput(node_idx, port) if *port == 0 => {
            if let Some(input_idx) = is_input_alias(resolved, *node_idx) {
                Some(BoundaryKey::Input(input_idx))
            } else if is_constant_boundary(resolved, *node_idx) {
                Some(BoundaryKey::Constant(*node_idx))
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Discover, synthesize, and compile the first production SIMD slice.
pub(crate) fn compile_tier1_ordinal(
    resolved: ResolvedDag,
    driving_input: &str,
    output: &str,
) -> Result<Tier1SimdExecutor, Tier1SimdError> {
    let effective_isa = EffectiveIsa::detect().map_err(Tier1SimdError::CapabilityDetection)?;
    let driving_input_index = resolved
        .input_defs
        .iter()
        .position(|input| input.name == driving_input)
        .ok_or_else(|| Tier1SimdError::DrivingInputNotFound(driving_input.to_string()))?;
    let (output_node, output_port) = resolved
        .output_map
        .get(output)
        .copied()
        .ok_or_else(|| Tier1SimdError::OutputNotFound(output.to_string()))?;
    if output_port != 0 || resolved.nodes[output_node].meta().outs.len() != 1 {
        return Err(Tier1SimdError::OutputPortUnsupported);
    }
    if resolved.output_modifiers.contains_key(output) {
        return Err(Tier1SimdError::OutputModifierUnsupported(
            output.to_string(),
        ));
    }
    if resolved.const_outputs.contains(output) {
        return Err(Tier1SimdError::ConstOutputUnsupported(output.to_string()));
    }

    let mut members = BTreeSet::new();
    let mut boundary_keys = BTreeSet::new();
    let mut stack = vec![output_node];
    while let Some(node_idx) = stack.pop() {
        if is_input_alias(&resolved, node_idx).is_some()
            || is_constant_boundary(&resolved, node_idx)
        {
            continue;
        }
        if !members.insert(node_idx) {
            continue;
        }
        for source in &resolved.wiring[node_idx] {
            if let Some(key) = boundary_key_for_source(&resolved, source) {
                boundary_keys.insert(key);
            } else if let WireSource::NodeOutput(upstream, port) = source {
                if *port != 0 {
                    return Err(Tier1SimdError::BoundaryUnsupported {
                        node: resolved.nodes[*upstream].meta().name.clone(),
                    });
                }
                stack.push(*upstream);
            }
        }
    }
    if members.len() < TIER1_MIN_MEMBERS {
        return Err(Tier1SimdError::TooFewMembers {
            found: members.len(),
            minimum: TIER1_MIN_MEMBERS,
        });
    }

    let mut shape = None;
    let mut variants = BTreeMap::new();
    let mut member_names = Vec::with_capacity(members.len());
    for &node_idx in &members {
        let node = &resolved.nodes[node_idx];
        if node.accepts_none_inputs() {
            return Err(Tier1SimdError::NoneAwareNode(node.meta().name.clone()));
        }
        let validated = validate_simd_variant(node.as_ref()).map_err(|reason| {
            Tier1SimdError::NodeIneligible {
                node: node.meta().name.clone(),
                reason: reason.to_string(),
            }
        })?;
        if let Some(expected) = shape {
            if expected != validated.shape {
                return Err(Tier1SimdError::ShapeMismatch {
                    node: node.meta().name.clone(),
                });
            }
        } else {
            shape = Some(validated.shape);
        }
        member_names.push(format!("{node_idx}:{}", node.meta().name));
        variants.insert(node_idx, validated);
    }
    let shape = shape.expect("minimum member count establishes a shape");
    if shape.scalar != PortType::U64 {
        return Err(Tier1SimdError::RuntimeShapeUnsupported(shape.scalar));
    }

    if !boundary_keys.contains(&BoundaryKey::Input(driving_input_index)) {
        return Err(Tier1SimdError::DrivingInputNotUsed(
            driving_input.to_string(),
        ));
    }

    let mut boundary_values = Vec::with_capacity(boundary_keys.len());
    let mut boundary_names = Vec::with_capacity(boundary_keys.len());
    let mut boundary_to_name = BTreeMap::new();
    let mut broadcast_inputs = Vec::new();
    for (position, key) in boundary_keys.iter().enumerate() {
        let vector_name = format!("__simd_boundary_{position}");
        boundary_to_name.insert(key.clone(), vector_name.clone());
        boundary_names.push(vector_name);
        match key {
            BoundaryKey::Input(input_index) => {
                let input = &resolved.input_defs[*input_index];
                if input.port_type != shape.scalar {
                    return Err(Tier1SimdError::BoundaryTypeMismatch {
                        boundary: input.name.clone(),
                        expected: shape.scalar,
                    });
                }
                if *input_index == driving_input_index {
                    boundary_values.push(BoundaryValue::Driving);
                } else {
                    if input.kind == InputKind::ExternalWrite {
                        return Err(Tier1SimdError::MutableBroadcast(input.name.clone()));
                    }
                    broadcast_inputs.push(input.name.clone());
                    boundary_values.push(BoundaryValue::Broadcast {
                        input_index: *input_index,
                        name: input.name.clone(),
                    });
                }
            }
            BoundaryKey::Constant(node_idx) => {
                let value = evaluate_constant_boundary(&resolved, *node_idx)?;
                if value.port_type() != shape.scalar {
                    return Err(Tier1SimdError::BoundaryTypeMismatch {
                        boundary: resolved.nodes[*node_idx].meta().name.clone(),
                        expected: shape.scalar,
                    });
                }
                boundary_values.push(BoundaryValue::Constant { value });
            }
        }
    }

    let mut vector_assembler = PolydatAssembler::new(boundary_names.clone());
    for name in &boundary_names {
        vector_assembler.set_input_type(name, shape.register);
    }
    vector_assembler.set_context(&resolved.source, "(Tier-1 SIMD register plan)");

    let mut vector_node_names = HashMap::new();
    for &node_idx in &members {
        let validated = &variants[&node_idx];
        let mut wires = Vec::with_capacity(resolved.wiring[node_idx].len());
        for source in &resolved.wiring[node_idx] {
            if let Some(key) = boundary_key_for_source(&resolved, source) {
                wires.push(WireRef::input(
                    boundary_to_name
                        .get(&key)
                        .expect("discovered boundary has vector input"),
                ));
            } else if let WireSource::NodeOutput(upstream, port) = source {
                let upstream_name = vector_node_names.get(upstream).ok_or_else(|| {
                    Tier1SimdError::BoundaryUnsupported {
                        node: resolved.nodes[*upstream].meta().name.clone(),
                    }
                })?;
                wires.push(WireRef::node_port(upstream_name, *port));
            } else {
                return Err(Tier1SimdError::VectorGraphBuild(
                    "unresolved scalar boundary".to_string(),
                ));
            }
        }
        let wire_types = vec![shape.register; wires.len()];
        let vector_node =
            crate::dsl::factory::build_node(validated.vector_node, &wires, &wire_types, &[])
                .map_err(|error| Tier1SimdError::VectorGraphBuild(error.to_string()))?;
        let name = format!("__simd_node_{node_idx}");
        vector_assembler.add_node(&name, vector_node, wires);
        vector_node_names.insert(node_idx, name);
    }
    let vector_output_node =
        vector_node_names
            .get(&output_node)
            .ok_or_else(|| Tier1SimdError::BoundaryUnsupported {
                node: resolved.nodes[output_node].meta().name.clone(),
            })?;
    vector_assembler.add_output(output, WireRef::node_port(vector_output_node, output_port));
    let vector = vector_assembler
        .try_compile_pure_jit_raw()
        .map_err(|e| Tier1SimdError::VectorCompilation(e.to_string()))?;
    let vector_output_slot = vector.resolve_output(output).ok_or_else(|| {
        Tier1SimdError::VectorCompilation("compiled output slot is missing".to_string())
    })?;

    let selected_output_map = HashMap::from([(output.to_string(), (output_node, output_port))]);
    let scalar_program = Arc::new(PolydatProgram::with_inputs(
        resolved.nodes,
        resolved.wiring,
        resolved.input_defs,
        resolved.coord_count,
        selected_output_map,
        vec![output.to_string()],
        &resolved.source,
        &resolved.context,
        resolved.ledger.clone(),
    ));
    let scalar = PolydatKernel::from_program(scalar_program);

    Ok(Tier1SimdExecutor {
        descriptor: Tier1SimdDescriptor {
            output: output.to_string(),
            driving_input: driving_input.to_string(),
            scalar_type: shape.scalar,
            register_type: shape.register,
            lanes: shape.lanes,
            member_nodes: member_names,
            broadcast_inputs,
            effective_isa_fingerprint: effective_isa.fingerprint(),
        },
        shape,
        driving_input_index,
        scalar,
        vector,
        vector_output_slot,
        boundaries: boundary_values,
        active: None,
        activation_epoch: None,
        expected_lease_sequence: 0,
        stream_identity: None,
        stats: Tier1SimdStats::default(),
    })
}
