// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Offset-stamped SIMD batching for stable ordinal streams.
//!
//! This is the Tier-1 execution primitive from
//! `docs/design/simd_isa_autopromotion.md`: one stable-by-ordinal source, one
//! lane-independent transform over a power-of-two 128-bit lane shape, one
//! ordered consumer, and explicit handling for lease fragments. It is not a
//! general scalar↔register type adapter; the compiler supplies only qualified
//! scalar/register variants.

use std::fmt;
use std::ops::Range;

/// Number of `i32` lanes in Polydat's 128-bit register plane.
pub const I32X4_LANES: usize = 4;

/// Stateless packet pacing derived directly from an ordinal cursor.
///
/// Every supported Polydat lane shape has a power-of-two lane count, so the
/// cursor's low bits are the lane selector and the remaining high bits are the
/// packet number. The runtime therefore does not need a second packet counter,
/// fill count, or consumed-lane bitmap for an ordered stable stream.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct OrdinalLaneClock<const LANES: usize>;

impl<const LANES: usize> OrdinalLaneClock<LANES> {
    #[inline]
    /// Whether the lane count is a power of two between 1 and 16.
    pub const fn is_supported() -> bool {
        LANES > 0 && LANES <= 16 && LANES.is_power_of_two()
    }

    /// Packet containing `ordinal`.
    #[inline]
    pub const fn packet_number(ordinal: u64) -> u64 {
        debug_assert!(Self::is_supported());
        ordinal >> LANES.trailing_zeros()
    }

    /// First ordinal in the packet containing `ordinal`.
    #[inline]
    pub const fn packet_base(ordinal: u64) -> u64 {
        debug_assert!(Self::is_supported());
        ordinal & !((LANES as u64) - 1)
    }

    /// Low-order cursor bits interpreted as a lane selector.
    #[inline]
    pub const fn lane_index(ordinal: u64) -> u8 {
        debug_assert!(Self::is_supported());
        (ordinal & ((LANES as u64) - 1)) as u8
    }

    /// Lanes at or after the cursor within its current packet.
    #[inline]
    pub const fn remaining_mask(ordinal: u64) -> u16 {
        debug_assert!(Self::is_supported());
        full_mask::<LANES>() << Self::lane_index(ordinal)
    }
}

/// Stable source which can render any owned ordinal without advancing mutable
/// cursor state.
pub trait StableOrdinalSource<T, const LANES: usize> {
    /// The stream this source belongs to.
    fn stream_id(&self) -> u64;
    /// The generation of the stream's values.
    fn generation(&self) -> u64;
    /// The value at one ordinal.
    fn value_at(&self, ordinal: u64) -> T;

    /// Materialize one aligned ingress packet. Sources with an affine or other
    /// closed form may override this to eliminate individual scalar renders.
    fn packet_at(&self, base_ordinal: u64) -> [T; LANES] {
        core::array::from_fn(|lane| self.value_at(base_ordinal + lane as u64))
    }
}

/// Closure-backed stable ordinal renderer used by typed promotion plans.
pub struct RenderedOrdinalSource<R> {
    /// The stream this source belongs to.
    pub stream_id: u64,
    /// The generation of its values.
    pub generation: u64,
    render: R,
}

impl<R> RenderedOrdinalSource<R> {
    /// A source rendering each ordinal with `render`.
    pub const fn new(stream_id: u64, generation: u64, render: R) -> Self {
        Self {
            stream_id,
            generation,
            render,
        }
    }
}

impl<T, R, const LANES: usize> StableOrdinalSource<T, LANES> for RenderedOrdinalSource<R>
where
    R: Fn(u64) -> T,
{
    fn stream_id(&self) -> u64 {
        self.stream_id
    }

    fn generation(&self) -> u64 {
        self.generation
    }

    fn value_at(&self, ordinal: u64) -> T {
        (self.render)(ordinal)
    }
}

/// A stable-by-ordinal affine scalar source.
///
/// Values use `i32` wrapping arithmetic. The origin may be placed anywhere in
/// the ordinal space, allowing a reserved stanza to be described without
/// materializing preceding values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AffineI32Source {
    /// The stream this source belongs to.
    pub stream_id: u64,
    /// The generation of its values.
    pub generation: u64,
    /// The ordinal the origin value sits at.
    pub origin_ordinal: u64,
    /// The value at the origin ordinal.
    pub origin_value: i32,
    /// The difference between consecutive values, wrapping.
    pub step: i32,
}

impl AffineI32Source {
    /// An affine source with the given origin and step.
    pub const fn new(
        stream_id: u64,
        generation: u64,
        origin_ordinal: u64,
        origin_value: i32,
        step: i32,
    ) -> Self {
        Self {
            stream_id,
            generation,
            origin_ordinal,
            origin_value,
            step,
        }
    }

    /// Render one ordinal without advancing any cursor.
    #[inline]
    pub fn value_at(self, ordinal: u64) -> i32 {
        if ordinal >= self.origin_ordinal {
            let delta = ordinal - self.origin_ordinal;
            self.origin_value
                .wrapping_add(self.step.wrapping_mul(delta as i32))
        } else {
            let delta = self.origin_ordinal - ordinal;
            self.origin_value
                .wrapping_sub(self.step.wrapping_mul(delta as i32))
        }
    }

    /// Synthesize the four consecutive values beginning at an aligned packet
    /// base. This is the pack-elimination surface for affine integer sources.
    #[inline]
    pub fn packet_at(self, base_ordinal: u64) -> [i32; I32X4_LANES] {
        let base = self.value_at(base_ordinal);
        [
            base,
            base.wrapping_add(self.step),
            base.wrapping_add(self.step.wrapping_mul(2)),
            base.wrapping_add(self.step.wrapping_mul(3)),
        ]
    }
}

impl StableOrdinalSource<i32, I32X4_LANES> for AffineI32Source {
    fn stream_id(&self) -> u64 {
        self.stream_id
    }

    fn generation(&self) -> u64 {
        self.generation
    }

    fn value_at(&self, ordinal: u64) -> i32 {
        (*self).value_at(ordinal)
    }

    fn packet_at(&self, base_ordinal: u64) -> [i32; I32X4_LANES] {
        (*self).packet_at(base_ordinal)
    }
}

/// Provenance and ownership stamp carried beside one SIMD value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OrdinalPacketStamp {
    /// The stream the value came from.
    pub stream_id: u64,
    /// The generation of the source's values.
    pub source_generation: u64,
    /// The activation the packet belongs to.
    pub activation_epoch: u64,
    /// The dependency epoch the packet was computed under.
    pub dependency_epoch: u64,
    /// The ordinal of lane 0.
    pub base_ordinal: u64,
    /// Logical lanes in this packet. Limited to 16 by `valid_mask`.
    pub lane_count: u8,
    /// Lane `i` is valid only when bit `i` is set.
    pub valid_mask: u16,
}

impl OrdinalPacketStamp {
    #[inline]
    /// The packet's index in the stream: its base ordinal over the lane count.
    pub const fn packet_number(self) -> u64 {
        debug_assert!(self.lane_count.is_power_of_two());
        self.base_ordinal >> self.lane_count.trailing_zeros()
    }

    #[inline]
    /// Whether the packet holds a valid lane for `ordinal`.
    pub const fn contains(self, ordinal: u64) -> bool {
        let Some(delta) = ordinal.checked_sub(self.base_ordinal) else {
            return false;
        };
        delta < self.lane_count as u64 && (self.valid_mask & (1 << delta)) != 0
    }
}

/// Durable state needed to resume an ordered consumer. Register payloads are
/// intentionally absent: a stable source can reconstruct them by ordinal.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OrdinalBatchCheckpoint {
    /// The stream.
    pub stream_id: u64,
    /// The generation of the source's values.
    pub source_generation: u64,
    /// The activation epoch.
    pub activation_epoch: u64,
    /// The dependency epoch.
    pub dependency_epoch: u64,
    /// The first ordinal of the lease.
    pub lease_start: u64,
    /// One past the last ordinal of the lease.
    pub lease_end: u64,
    /// The next ordinal the consumer will take.
    pub next_ordinal: u64,
}

/// Counters used by correctness tests and the local performance study.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct OrdinalBatchStats {
    /// Packets computed by the vector path.
    pub vector_packets: u64,
    /// Vector packets with invalid lanes past the lease end.
    pub padded_vector_packets: u64,
    /// Lanes computed by the scalar path.
    pub scalar_fragment_lanes: u64,
    /// Values handed to the consumer.
    pub values_drained: u64,
    /// Packets served again from the ready register.
    pub burst_reuses: u64,
    /// Times a dependency change dropped the ready register.
    pub dependency_invalidations: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
/// Why a batch could not be built or resumed.
pub enum OrdinalBatchError {
    /// The lease's end precedes its start.
    InvalidLease {
        /// The lease's first ordinal.
        start: u64,
        /// One past its last.
        end: u64,
    },
    /// The lane count is not a supported power of two.
    InvalidLaneCount {
        /// The lane count given.
        lanes: usize,
    },
    /// The checkpoint names another stream, generation, or epoch.
    CheckpointIdentityMismatch,
    /// The checkpoint's lease is not this batch's.
    CheckpointLeaseMismatch,
    /// The checkpoint's cursor lies outside the lease.
    CheckpointCursorOutsideLease,
}

impl fmt::Display for OrdinalBatchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidLease { start, end } => {
                write!(f, "invalid ordinal lease [{start}, {end})")
            }
            Self::InvalidLaneCount { lanes } => {
                write!(
                    f,
                    "SIMD lane count must be a power of two in 1..=16, got {lanes}"
                )
            }
            Self::CheckpointIdentityMismatch => {
                f.write_str("checkpoint stream/generation/epoch identity mismatch")
            }
            Self::CheckpointLeaseMismatch => f.write_str("checkpoint lease mismatch"),
            Self::CheckpointCursorOutsideLease => {
                f.write_str("checkpoint consumer cursor is outside its lease")
            }
        }
    }
}

impl std::error::Error for OrdinalBatchError {}

#[derive(Clone, Copy, Debug)]
struct ReadyPacket<T, const LANES: usize> {
    stamp: OrdinalPacketStamp,
    lanes: [T; LANES],
}

/// Treatment of a packet intersecting only part of the owned stanza.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PacketFragmentPolicy {
    /// Evaluate only owned lanes through the scalar oracle. This is the safe
    /// default for opaque sources and operations which may fault or observe
    /// evaluation.
    #[default]
    Scalarize,
    /// Synthesize and evaluate the complete aligned vector, then expose only
    /// lanes in the ownership mask. This requires a mathematically extensible
    /// perfect sequence and a pure, total, lane-independent vector pipeline.
    PadAndVectorize,
}

/// Burst-resumable typed SIMD executor for one owned ordinal stanza.
///
/// `V` is the explicit SIMD-native graph variant and `S` is its scalar oracle.
/// Both must be pure, total, lane-independent, and semantically equivalent.
/// The type cannot prove those laws; construction is intentionally kept at the
/// executor/compiler boundary where variant metadata will eventually enforce
/// them.
pub struct OrdinalSimdStream<T, const LANES: usize, R, V, S>
where
    T: Copy + Default,
    R: StableOrdinalSource<T, LANES>,
    V: FnMut([T; LANES]) -> [T; LANES],
    S: FnMut(T) -> T,
{
    source: R,
    lease: Range<u64>,
    activation_epoch: u64,
    dependency_epoch: u64,
    next_ordinal: u64,
    ready: Option<ReadyPacket<T, LANES>>,
    fragment_policy: PacketFragmentPolicy,
    vector: V,
    scalar: S,
    stats: OrdinalBatchStats,
}

/// Backward-compatible name for the original affine `i32x4` prototype.
pub type OrdinalI32x4Stream<V, S> = OrdinalSimdStream<i32, I32X4_LANES, AffineI32Source, V, S>;

impl<T, const LANES: usize, R, V, S> OrdinalSimdStream<T, LANES, R, V, S>
where
    T: Copy + Default,
    R: StableOrdinalSource<T, LANES>,
    V: FnMut([T; LANES]) -> [T; LANES],
    S: FnMut(T) -> T,
{
    /// A batch over `lease` of `source`, computing packets with `vector` and
    /// single lanes with `scalar`; an inverted lease is an error.
    pub fn new(
        source: R,
        lease: Range<u64>,
        activation_epoch: u64,
        dependency_epoch: u64,
        vector: V,
        scalar: S,
    ) -> Result<Self, OrdinalBatchError> {
        if lease.start > lease.end {
            return Err(OrdinalBatchError::InvalidLease {
                start: lease.start,
                end: lease.end,
            });
        }
        if !OrdinalLaneClock::<LANES>::is_supported() {
            return Err(OrdinalBatchError::InvalidLaneCount { lanes: LANES });
        }
        let next_ordinal = lease.start;
        Ok(Self {
            source,
            lease,
            activation_epoch,
            dependency_epoch,
            next_ordinal,
            ready: None,
            fragment_policy: PacketFragmentPolicy::Scalarize,
            vector,
            scalar,
            stats: OrdinalBatchStats::default(),
        })
    }

    /// Select how unaligned lease fragments are evaluated. The compiler may
    /// choose `PadAndVectorize` only after proving its documented source and
    /// node requirements.
    pub fn with_fragment_policy(mut self, fragment_policy: PacketFragmentPolicy) -> Self {
        self.fragment_policy = fragment_policy;
        self
    }

    /// Ordered scalar consumer frontier. The allocation frontier is external
    /// to this object and is never rewound by packet recovery.
    #[inline]
    pub const fn next_ordinal(&self) -> u64 {
        self.next_ordinal
    }

    #[inline]
    /// The ordinal range leased.
    pub fn lease(&self) -> Range<u64> {
        self.lease.clone()
    }

    #[inline]
    /// The counters so far.
    pub const fn stats(&self) -> OrdinalBatchStats {
        self.stats
    }

    #[inline]
    /// The stamp of the packet ready to drain, if any.
    pub fn current_stamp(&self) -> Option<OrdinalPacketStamp> {
        self.ready.map(|p| p.stamp)
    }

    /// Drop derived register state after a relevant dependency changes. The
    /// source key and ordered consumer frontier remain intact.
    pub fn invalidate_dependency_epoch(&mut self, dependency_epoch: u64) {
        if self.dependency_epoch != dependency_epoch {
            self.dependency_epoch = dependency_epoch;
            self.ready = None;
            self.stats.dependency_invalidations += 1;
        }
    }

    /// Produce a durable, payload-free consumer checkpoint.
    pub fn checkpoint(&self) -> OrdinalBatchCheckpoint {
        OrdinalBatchCheckpoint {
            stream_id: self.source.stream_id(),
            source_generation: self.source.generation(),
            activation_epoch: self.activation_epoch,
            dependency_epoch: self.dependency_epoch,
            lease_start: self.lease.start,
            lease_end: self.lease.end,
            next_ordinal: self.next_ordinal,
        }
    }

    /// Restore a checkpoint and deliberately discard any process-local SIMD
    /// payload. The caller must ensure values after the checkpoint were not
    /// externally committed; this is state restoration, not cursor rewind.
    pub fn restore_checkpoint(
        &mut self,
        checkpoint: OrdinalBatchCheckpoint,
    ) -> Result<(), OrdinalBatchError> {
        if checkpoint.stream_id != self.source.stream_id()
            || checkpoint.source_generation != self.source.generation()
            || checkpoint.activation_epoch != self.activation_epoch
            || checkpoint.dependency_epoch != self.dependency_epoch
        {
            return Err(OrdinalBatchError::CheckpointIdentityMismatch);
        }
        if checkpoint.lease_start != self.lease.start || checkpoint.lease_end != self.lease.end {
            return Err(OrdinalBatchError::CheckpointLeaseMismatch);
        }
        if checkpoint.next_ordinal < self.lease.start || checkpoint.next_ordinal > self.lease.end {
            return Err(OrdinalBatchError::CheckpointCursorOutsideLease);
        }
        self.next_ordinal = checkpoint.next_ordinal;
        self.ready = None;
        Ok(())
    }

    /// Drain up to `output.len()` committed scalar values. Calls may use any
    /// burst size; an incompletely drained packet remains available to the next
    /// call.
    pub fn drain_into(&mut self, output: &mut [T]) -> usize {
        if output.is_empty() || self.next_ordinal >= self.lease.end {
            return 0;
        }

        if self
            .ready
            .is_some_and(|p| p.stamp.contains(self.next_ordinal))
        {
            self.stats.burst_reuses += 1;
        }

        let mut written = 0;
        while written < output.len() && self.next_ordinal < self.lease.end {
            self.ensure_ready();
            let packet = self.ready.expect("ensure_ready must materialize a packet");
            debug_assert!(packet.stamp.contains(self.next_ordinal));

            let lane = (self.next_ordinal - packet.stamp.base_ordinal) as usize;
            output[written] = packet.lanes[lane];
            written += 1;
            self.stats.values_drained += 1;
            self.next_ordinal += 1;

            let packet_end = packet.stamp.base_ordinal.saturating_add(LANES as u64);
            if self.next_ordinal >= packet_end || self.next_ordinal >= self.lease.end {
                self.ready = None;
            }
        }
        written
    }

    fn ensure_ready(&mut self) {
        if self.ready.is_some_and(|p| {
            p.stamp.contains(self.next_ordinal)
                && p.stamp.source_generation == self.source.generation()
                && p.stamp.activation_epoch == self.activation_epoch
                && p.stamp.dependency_epoch == self.dependency_epoch
        }) {
            return;
        }

        let base_ordinal = OrdinalLaneClock::<LANES>::packet_base(self.next_ordinal);
        let valid_mask = valid_mask::<LANES>(base_ordinal, &self.lease);
        let full_mask = full_mask::<LANES>();
        debug_assert_ne!(valid_mask, 0);

        let ingress = self.source.packet_at(base_ordinal);
        let vectorize = valid_mask == full_mask
            || self.fragment_policy == PacketFragmentPolicy::PadAndVectorize;
        let lanes = if vectorize {
            self.stats.vector_packets += 1;
            if valid_mask != full_mask {
                self.stats.padded_vector_packets += 1;
            }
            (self.vector)(ingress)
        } else {
            let mut lanes = [T::default(); LANES];
            for lane in 0..LANES {
                if valid_mask & (1 << lane) != 0 {
                    lanes[lane] = (self.scalar)(ingress[lane]);
                    self.stats.scalar_fragment_lanes += 1;
                }
            }
            lanes
        };

        self.ready = Some(ReadyPacket {
            stamp: OrdinalPacketStamp {
                stream_id: self.source.stream_id(),
                source_generation: self.source.generation(),
                activation_epoch: self.activation_epoch,
                dependency_epoch: self.dependency_epoch,
                base_ordinal,
                lane_count: LANES as u8,
                valid_mask,
            },
            lanes,
        });
    }
}

#[inline]
fn valid_mask<const LANES: usize>(base_ordinal: u64, lease: &Range<u64>) -> u16 {
    let mut mask = 0;
    for lane in 0..LANES {
        let Some(ordinal) = base_ordinal.checked_add(lane as u64) else {
            continue;
        };
        if ordinal >= lease.start && ordinal < lease.end {
            mask |= 1 << lane;
        }
    }
    mask
}

#[inline]
const fn full_mask<const LANES: usize>() -> u16 {
    if LANES == 16 {
        u16::MAX
    } else {
        (1u16 << LANES) - 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[inline]
    fn scalar_pipeline(mut x: i32) -> i32 {
        for _ in 0..4 {
            x = x.wrapping_mul(3).wrapping_add(17);
        }
        x
    }

    fn vector_pipeline(mut x: [i32; 4]) -> [i32; 4] {
        for _ in 0..4 {
            x = x.map(|v| v.wrapping_mul(3).wrapping_add(17));
        }
        x
    }

    fn stream(
        lease: Range<u64>,
    ) -> OrdinalI32x4Stream<impl FnMut([i32; 4]) -> [i32; 4], impl FnMut(i32) -> i32> {
        OrdinalI32x4Stream::new(
            AffineI32Source::new(7, 3, 0, -11, 5),
            lease,
            9,
            0,
            vector_pipeline,
            scalar_pipeline,
        )
        .unwrap()
    }

    #[test]
    fn affine_source_supports_origins_on_either_side() {
        let source = AffineI32Source::new(1, 0, 10, 100, 7);
        assert_eq!(source.value_at(8), 86);
        assert_eq!(source.value_at(10), 100);
        assert_eq!(source.value_at(13), 121);
        assert_eq!(source.packet_at(8), [86, 93, 100, 107]);
    }

    #[test]
    fn ordinal_low_bits_are_the_lane_clock_for_every_common_shape() {
        macro_rules! check_clock {
            ($lanes:expr) => {{
                for ordinal in 0..(3 * $lanes as u64) {
                    assert_eq!(
                        OrdinalLaneClock::<$lanes>::packet_number(ordinal),
                        ordinal / $lanes as u64
                    );
                    assert_eq!(
                        OrdinalLaneClock::<$lanes>::packet_base(ordinal),
                        ordinal - ordinal % $lanes as u64
                    );
                    assert_eq!(
                        OrdinalLaneClock::<$lanes>::lane_index(ordinal),
                        (ordinal % $lanes as u64) as u8
                    );
                    let lane = OrdinalLaneClock::<$lanes>::lane_index(ordinal);
                    assert_eq!(
                        OrdinalLaneClock::<$lanes>::remaining_mask(ordinal),
                        full_mask::<$lanes>() << lane
                    );
                }
            }};
        }

        check_clock!(2);
        check_clock!(4);
        check_clock!(8);
        check_clock!(16);
    }

    #[test]
    fn every_lease_alignment_matches_the_scalar_oracle() {
        for start in 0..8 {
            for len in 0..17 {
                let end = start + len;
                let mut batch = stream(start..end);
                let mut got = vec![0; len as usize];
                assert_eq!(batch.drain_into(&mut got), len as usize);

                let source = AffineI32Source::new(7, 3, 0, -11, 5);
                let want: Vec<_> = (start..end)
                    .map(|ordinal| scalar_pipeline(source.value_at(ordinal)))
                    .collect();
                assert_eq!(got, want, "lease={start}..{end}");
            }
        }
    }

    #[test]
    fn arbitrary_bursts_reuse_packets_without_sequence_drift() {
        let mut batch = stream(2..31);
        let bursts = [1usize, 7, 2, 3, 8, 1, 4, 16];
        let mut got = Vec::new();
        let mut burst_index = 0;
        while batch.next_ordinal() < batch.lease().end {
            let mut buf = vec![0; bursts[burst_index % bursts.len()]];
            let n = batch.drain_into(&mut buf);
            got.extend_from_slice(&buf[..n]);
            burst_index += 1;
        }

        let source = AffineI32Source::new(7, 3, 0, -11, 5);
        let want: Vec<_> = (2..31)
            .map(|ordinal| scalar_pipeline(source.value_at(ordinal)))
            .collect();
        assert_eq!(got, want);
        assert!(batch.stats().burst_reuses > 0);
        assert_eq!(batch.stats().values_drained, 29);
    }

    #[test]
    fn unaligned_lease_scalarizes_only_its_fragments() {
        let mut batch = stream(1..11);
        let mut out = [0; 10];
        assert_eq!(batch.drain_into(&mut out), 10);
        let stats = batch.stats();
        assert_eq!(stats.vector_packets, 1);
        assert_eq!(stats.scalar_fragment_lanes, 6);
    }

    #[test]
    fn perfect_sequence_can_pad_unaligned_fragments_without_exposing_them() {
        let mut batch = stream(1..11).with_fragment_policy(PacketFragmentPolicy::PadAndVectorize);
        let mut got = [0; 10];
        assert_eq!(batch.drain_into(&mut got), 10);

        let source = AffineI32Source::new(7, 3, 0, -11, 5);
        let want =
            core::array::from_fn::<_, 10, _>(|i| scalar_pipeline(source.value_at(1 + i as u64)));
        assert_eq!(got, want);
        assert_eq!(batch.stats().vector_packets, 3);
        assert_eq!(batch.stats().padded_vector_packets, 2);
        assert_eq!(batch.stats().scalar_fragment_lanes, 0);
    }

    #[test]
    fn checkpoint_discards_payload_and_rematerializes_by_ordinal() {
        let mut original = stream(0..20);
        let mut prefix = [0; 3];
        assert_eq!(original.drain_into(&mut prefix), 3);
        assert!(original.current_stamp().is_some());
        let checkpoint = original.checkpoint();

        let mut resumed = stream(0..20);
        resumed.restore_checkpoint(checkpoint).unwrap();
        assert!(resumed.current_stamp().is_none());
        let mut suffix = [0; 17];
        assert_eq!(resumed.drain_into(&mut suffix), 17);

        let source = AffineI32Source::new(7, 3, 0, -11, 5);
        let want: Vec<_> = (0..20)
            .map(|ordinal| scalar_pipeline(source.value_at(ordinal)))
            .collect();
        assert_eq!([prefix.as_slice(), suffix.as_slice()].concat(), want);
    }

    #[test]
    fn dependency_invalidation_keeps_the_consumer_frontier() {
        let mut batch = stream(0..12);
        let mut first = [0];
        batch.drain_into(&mut first);
        let next = batch.next_ordinal();
        assert!(batch.current_stamp().is_some());

        batch.invalidate_dependency_epoch(1);
        assert_eq!(batch.next_ordinal(), next);
        assert!(batch.current_stamp().is_none());

        let mut rest = [0; 11];
        assert_eq!(batch.drain_into(&mut rest), 11);
        assert_eq!(batch.stats().dependency_invalidations, 1);
    }

    #[test]
    fn checkpoint_identity_and_lease_are_validated() {
        let batch = stream(4..12);
        let mut wrong_identity = batch.checkpoint();
        wrong_identity.source_generation += 1;
        let mut restore = stream(4..12);
        assert_eq!(
            restore.restore_checkpoint(wrong_identity),
            Err(OrdinalBatchError::CheckpointIdentityMismatch)
        );

        let wrong_lease = batch.checkpoint();
        let mut restore = stream(4..16);
        assert_eq!(
            restore.restore_checkpoint(wrong_lease),
            Err(OrdinalBatchError::CheckpointLeaseMismatch)
        );
    }

    #[test]
    fn common_scalar_lane_shapes_share_the_cursor_clock() {
        macro_rules! check_shape {
            ($ty:ty, $lanes:expr, $render:expr, $transform:expr) => {{
                let source = RenderedOrdinalSource::new(91, 4, $render);
                let mut batch = OrdinalSimdStream::<$ty, $lanes, _, _, _>::new(
                    source,
                    3..41,
                    2,
                    0,
                    |lanes| lanes.map($transform),
                    $transform,
                )
                .unwrap()
                .with_fragment_policy(PacketFragmentPolicy::PadAndVectorize);

                let bursts = [1usize, 3, 7, 2, 19];
                let mut got = Vec::new();
                let mut burst = 0;
                while batch.next_ordinal() < batch.lease().end {
                    let mut output = vec![<$ty>::default(); bursts[burst % bursts.len()]];
                    let written = batch.drain_into(&mut output);
                    got.extend_from_slice(&output[..written]);
                    burst += 1;
                }
                let want: Vec<$ty> = (3..41)
                    .map(|ordinal| $transform(($render)(ordinal)))
                    .collect();
                assert_eq!(got, want, "{}x{}", stringify!($ty), $lanes);
                assert!(batch.stats().vector_packets > 0);
                assert!(batch.stats().burst_reuses > 0);
            }};
        }

        check_shape!(u8, 16, |o| o as u8, |x: u8| x
            .wrapping_mul(3)
            .wrapping_add(1));
        check_shape!(i8, 16, |o| o as i8 - 20, |x: i8| x
            .wrapping_mul(3)
            .wrapping_add(1));
        check_shape!(u16, 8, |o| o as u16, |x: u16| x
            .wrapping_mul(5)
            .wrapping_add(7));
        check_shape!(i16, 8, |o| o as i16 - 20, |x: i16| x
            .wrapping_mul(5)
            .wrapping_add(7));
        check_shape!(u32, 4, |o| o as u32, |x: u32| x
            .wrapping_mul(11)
            .wrapping_add(9));
        check_shape!(
            i32,
            4,
            |o: u64| i32::try_from(o).expect("ordinal fits") - 20,
            |x: i32| x.wrapping_mul(11).wrapping_add(9)
        );
        check_shape!(u64, 2, |o| o, |x: u64| x.wrapping_mul(13).wrapping_add(3));
        check_shape!(i64, 2, |o| o as i64 - 20, |x: i64| x
            .wrapping_mul(13)
            .wrapping_add(3));
        check_shape!(f32, 4, |o| o as f32 * 0.5, |x: f32| x * 1.5 + 2.0);
        check_shape!(f64, 2, |o| o as f64 * 0.5, |x: f64| x * 1.5 + 2.0);
    }

    #[test]
    fn unsupported_lane_counts_are_rejected() {
        let result = OrdinalSimdStream::<u8, 3, _, _, _>::new(
            RenderedOrdinalSource::new(1, 0, |o| o as u8),
            0..3,
            0,
            0,
            |lanes| lanes,
            |lane| lane,
        );
        assert!(matches!(
            result,
            Err(OrdinalBatchError::InvalidLaneCount { lanes: 3 })
        ));
    }
}
