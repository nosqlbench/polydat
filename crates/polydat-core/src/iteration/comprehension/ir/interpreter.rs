// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Stack-machine IR interpreter — spec §9.1 option (a) +
//! §9.2 correctness contract.
//!
//! Walks an IR `Program` linearly, maintaining a stack of
//! tuple-stream operands. Each opcode either pushes a new
//! stream ([`Op::PushClause`]) or combines / wraps the top-N
//! ([`Op::Cartesian`], `Zip`, `Union`, `Filter`,
//! `OrderStreaming`, `OrderMaterialize`). `Dispense` marks
//! the top stream as the output.
//!
//! Returns a [`TupleStream`] — a lazy producer the consumer
//! pulls from. The stream graph is built at
//! [`interpret`]-time; tuple production happens on-demand.
//!
//! A stream over an index-addressable subtree (clauses over literal
//! lists and ranges, combined by cartesian, zip, union, and `Lex`
//! order) also answers the tuple at a position without pulling. A
//! cartesian over such children computes each tuple from its
//! position instead of caching its tail axes, a cycle zip reads such
//! an operand at `i mod |operand|` instead of buffering it, and a
//! non-`Lex` order over such an input selects its positions and
//! computes only the selected tuples (spec §6.2, §10.2 R2).

use crate::iteration::comprehension::metadata::{CycleOperand, IndexFn, cycle_length};
use crate::iteration::comprehension::source::{LiteralValue, Source};
use crate::iteration::comprehension::strategies::{Selection, Tuple, TupleValue};
use crate::iteration::comprehension::strategy::{StrategyName, ZipMode};

use super::op::{Op, OrderStreamingKind};
use super::program::Program;

/// A lazy tuple stream — `advance` returns the next tuple or
/// `None` when the stream is exhausted.
pub trait TupleStream {
    /// The next tuple, or `None` once the stream is exhausted.
    fn advance(&mut self) -> Option<Tuple>;

    /// Restart at the first tuple: the stream dispenses the same
    /// tuples again, keeping whatever it built the first time.
    fn rewind(&mut self);

    /// How many tuples the stream dispenses, when its tuples are
    /// addressable by position; `None` otherwise.
    fn indexed_len(&self) -> Option<u64> {
        None
    }

    /// The tuple the stream dispenses at position `i`, when its tuples
    /// are addressable, whatever the dispense position; `None` past
    /// the end or when they are not addressable.
    fn tuple_at(&self, i: u64) -> Option<Tuple> {
        let _ = i;
        None
    }
}

/// Append `tuple`'s bindings to `out`.
fn extend(out: &mut Tuple, tuple: Tuple) {
    out.bindings.extend(tuple.bindings);
}

/// Boxed stream alias used throughout the interpreter's stack
/// manipulation.
type BoxedStream = Box<dyn TupleStream>;

/// Interpret a compiled `Program` and return the result
/// stream. Per spec §9.1 the final opcode must be
/// `Op::Dispense`; if it's missing the function panics
/// (programs not produced by the compiler are caller-error).
pub fn interpret(program: &Program) -> BoxedStream {
    let mut stack: Vec<BoxedStream> = Vec::new();
    for op in program.ops() {
        match op {
            Op::PushClause { name, source } => {
                stack.push(Box::new(ClauseStream::new(name.clone(), source.clone())));
            }
            Op::Cartesian { n } => {
                let children = pop_n(&mut stack, *n);
                stack.push(Box::new(CartesianStream::new(children)));
            }
            Op::Zip { n, mode, operands } => {
                let children = pop_n(&mut stack, *n);
                stack.push(Box::new(ZipStream::new(children, *mode, operands.clone())));
            }
            Op::Union { n } => {
                let children = pop_n(&mut stack, *n);
                stack.push(Box::new(UnionStream::new(children)));
            }
            Op::Filter { predicate } => {
                let inner = stack.pop().expect("Filter on empty stack");
                stack.push(Box::new(FilterStream::new(inner, predicate.clone())));
            }
            Op::OrderStreaming { kind, truncation } => {
                let inner = stack.pop().expect("OrderStreaming on empty stack");
                stack.push(Box::new(OrderStreamingStream::new(
                    inner,
                    *kind,
                    *truncation,
                )));
            }
            Op::OrderMaterialize {
                strategy,
                truncation,
                input_index_fn,
                seed,
            } => {
                let inner = stack.pop().expect("OrderMaterialize on empty stack");
                stack.push(Box::new(OrderMaterializeStream::new(
                    inner,
                    *strategy,
                    *truncation,
                    input_index_fn.clone(),
                    *seed,
                )));
            }
            Op::Dispense => {
                // No-op at the interpreter level; the top of
                // stack is the result.
            }
        }
    }
    stack.pop().expect("Program produced no result stream")
}

fn pop_n(stack: &mut Vec<BoxedStream>, n: usize) -> Vec<BoxedStream> {
    assert!(
        stack.len() >= n,
        "stack underflow: needed {n}, have {}",
        stack.len()
    );
    let split_at = stack.len() - n;
    stack.split_off(split_at)
}

// ---- ClauseStream ----

/// Streams a Source's values as single-name tuples.
/// Continuous / Distribution sources are not interpretable
/// at this layer (they require a sampling order to be
/// meaningful per V8); attempting to advance one returns
/// `None` so the interpreter doesn't deadlock.
struct ClauseStream {
    name: String,
    values: ClauseValues,
    pos: u64,
}

/// A clause's values, each computed from its position.
enum ClauseValues {
    LiteralList(Vec<LiteralValue>),
    /// `lo, lo + step, …`, `len` of them.
    IntRange {
        lo: i64,
        step: i64,
        len: u64,
    },
    Exhausted,
}

impl ClauseStream {
    fn new(name: String, source: Source) -> Self {
        let values = match source {
            Source::Literal { values } => ClauseValues::LiteralList(values),
            Source::IntRange { lo, hi, step } => {
                let step = step.max(1);
                let len = if hi <= lo {
                    0
                } else {
                    ((i128::from(hi) - i128::from(lo)) as u128).div_ceil(step as u128) as u64
                };
                ClauseValues::IntRange { lo, step, len }
            }
            // Generator / WorkloadParamList: a context-free one is a
            // literal by the time the IR is compiled, and a
            // context-required one is refused by `from_ast`
            // (`ValidationError::ContextRequired`); a hand-built
            // program carrying one dispenses nothing.
            // ContinuousInterval / Distribution: must be sampled
            // via an enclosing order; bare clause is not pulled
            // in valid programs.
            _ => ClauseValues::Exhausted,
        };
        Self {
            name,
            values,
            pos: 0,
        }
    }
}

impl TupleStream for ClauseStream {
    fn advance(&mut self) -> Option<Tuple> {
        let t = self.tuple_at(self.pos)?;
        self.pos += 1;
        Some(t)
    }

    fn rewind(&mut self) {
        self.pos = 0;
    }

    fn indexed_len(&self) -> Option<u64> {
        Some(match &self.values {
            ClauseValues::LiteralList(values) => values.len() as u64,
            ClauseValues::IntRange { len, .. } => *len,
            ClauseValues::Exhausted => 0,
        })
    }

    fn tuple_at(&self, i: u64) -> Option<Tuple> {
        let value = match &self.values {
            ClauseValues::LiteralList(values) => {
                literal_to_tuple_value(values.get(usize::try_from(i).ok()?)?)
            }
            ClauseValues::IntRange { lo, step, len } => {
                if i >= *len {
                    return None;
                }
                TupleValue::I64((i128::from(*lo) + i128::from(i) * i128::from(*step)) as i64)
            }
            ClauseValues::Exhausted => return None,
        };
        Some(Tuple::new().with(self.name.clone(), value))
    }
}

fn literal_to_tuple_value(lv: &LiteralValue) -> TupleValue {
    match lv {
        LiteralValue::Int(n) => TupleValue::I64(*n),
        LiteralValue::Float(f) => TupleValue::F64(*f),
        LiteralValue::String(s) => TupleValue::Str(s.clone()),
        LiteralValue::Bool(b) => TupleValue::Bool(*b),
        LiteralValue::Json(j) => TupleValue::Str(j.to_string()),
    }
}

// ---- CartesianStream ----

/// Enumerates the cross product of N child streams in Lex
/// order (rightmost varies fastest).
///
/// When every child is addressable, the tuple at position `i` is
/// its children's tuples at the mixed-radix digits of `i`, and the
/// stream holds nothing but its position. Otherwise the first advance
/// pulls child 0 once and children 1..N to exhaustion, caching them,
/// and later advances iterate over the cached cross product.
struct CartesianStream {
    children: Vec<BoxedStream>,
    /// Every child's tuple count, when every child is addressable and
    /// the product fits.
    lens: Option<Vec<u64>>,
    /// The product of `lens`.
    total: u64,
    /// The dispense position, when addressable.
    pos: u64,
    /// Cached values for axes 1..N (axis 0 streams).
    cached: Vec<Vec<Tuple>>,
    /// Current cursor for axis 0 (lazy pull).
    current_a0: Option<Tuple>,
    /// Cursor positions for axes 1..N.
    cursors: Vec<usize>,
    /// True once we've initialized — first advance() needs to
    /// cache children 1..N and pull initial child 0.
    initialized: bool,
    done: bool,
}

impl CartesianStream {
    fn new(children: Vec<BoxedStream>) -> Self {
        let n = children.len();
        let lens: Option<Vec<u64>> = if children.is_empty() {
            None
        } else {
            children.iter().map(|c| c.indexed_len()).collect()
        };
        let total = lens
            .as_ref()
            .and_then(|l| l.iter().try_fold(1u64, |acc, n| acc.checked_mul(*n)));
        Self {
            children,
            lens: total.and(lens),
            total: total.unwrap_or(0),
            pos: 0,
            cached: Vec::with_capacity(n.saturating_sub(1)),
            current_a0: None,
            cursors: vec![0; n.saturating_sub(1)],
            initialized: false,
            done: false,
        }
    }

    fn initialize(&mut self) {
        if self.children.is_empty() {
            self.done = true;
            return;
        }
        // Cache all children 1..N to exhaustion.
        for i in 1..self.children.len() {
            let mut v = Vec::new();
            while let Some(t) = self.children[i].advance() {
                v.push(t);
            }
            self.cached.push(v);
        }
        // Pull first axis 0 value.
        self.current_a0 = self.children[0].advance();
        if self.current_a0.is_none() || self.cached.iter().any(|v| v.is_empty()) {
            // Any empty axis → empty cartesian.
            self.done = true;
        }
    }
}

impl TupleStream for CartesianStream {
    fn indexed_len(&self) -> Option<u64> {
        self.lens.as_ref().map(|_| self.total)
    }

    fn tuple_at(&self, i: u64) -> Option<Tuple> {
        let lens = self.lens.as_ref()?;
        if i >= self.total {
            return None;
        }
        // Mixed-radix digits of `i`, the last axis least significant.
        let mut digits = vec![0u64; lens.len()];
        let mut rest = i;
        for (d, len) in digits.iter_mut().zip(lens).rev() {
            *d = rest % len;
            rest /= len;
        }
        let mut out = Tuple::new();
        for (child, d) in self.children.iter().zip(digits) {
            extend(&mut out, child.tuple_at(d)?);
        }
        Some(out)
    }

    fn rewind(&mut self) {
        self.pos = 0;
        if self.lens.is_some() || !self.initialized || self.children.is_empty() {
            return;
        }
        self.children[0].rewind();
        self.cursors.iter_mut().for_each(|c| *c = 0);
        self.current_a0 = self.children[0].advance();
        self.done = self.current_a0.is_none() || self.cached.iter().any(|v| v.is_empty());
    }

    fn advance(&mut self) -> Option<Tuple> {
        if self.lens.is_some() {
            let t = self.tuple_at(self.pos)?;
            self.pos += 1;
            return Some(t);
        }
        if !self.initialized {
            self.initialize();
            self.initialized = true;
        }
        if self.done {
            return None;
        }
        // Compose current cursor + current axis-0 value.
        let mut out = Tuple::new();
        if let Some(a0) = self.current_a0.as_ref() {
            for (k, v) in &a0.bindings {
                out.bindings.push((k.clone(), v.clone()));
            }
        }
        for (i, cursor) in self.cursors.iter().enumerate() {
            let tup = &self.cached[i][*cursor];
            for (k, v) in &tup.bindings {
                out.bindings.push((k.clone(), v.clone()));
            }
        }

        // Advance cursors (rightmost-fastest).
        let n_cached = self.cursors.len();
        let mut overflow = true;
        for i in (0..n_cached).rev() {
            self.cursors[i] += 1;
            if self.cursors[i] < self.cached[i].len() {
                overflow = false;
                break;
            }
            self.cursors[i] = 0;
        }
        if overflow {
            // Advance axis 0.
            self.current_a0 = self.children[0].advance();
            if self.current_a0.is_none() {
                self.done = true;
            }
        }
        Some(out)
    }
}

// ---- ZipStream ----

/// Lockstep over N child streams. Strict and Truncate pull one tuple
/// from each child and stop when any child runs out. Cycle runs to its
/// longest operand and holds each operand as its plan says
/// ([`CycleOperand`]): an indexed operand is read at `i mod |operand|`,
/// a buffered one is drained once and replayed, and the streamed one
/// is pulled once per tuple and rewound when it runs out first. An
/// empty operand empties the zip.
struct ZipStream {
    children: Vec<BoxedStream>,
    mode: ZipMode,
    /// Under Cycle, the plan the op carries; empty for a program built
    /// without one.
    plan: Vec<CycleOperand>,
    /// Under Cycle, how each operand is held, once initialized.
    holds: Vec<Hold>,
    /// Under Cycle, the streamed operand's position among the
    /// children.
    streamed: Option<usize>,
    /// The streamed operand's length, known once it first runs out.
    streamed_len: Option<u64>,
    /// The longest indexed or buffered operand's length.
    known_len: u64,
    /// Under Cycle, the position of the next tuple.
    step: u64,
    /// Under Cycle, an indexed or buffered operand was found empty.
    empty: bool,
    initialized: bool,
    done: bool,
}

/// How a cycle zip holds one operand.
enum Hold {
    /// Read at `i mod len` through the child's `tuple_at`.
    Indexed(u64),
    /// Drained once and replayed.
    Buffered(Vec<Tuple>),
    /// Pulled once per tuple.
    Streamed,
}

impl ZipStream {
    fn new(children: Vec<BoxedStream>, mode: ZipMode, plan: Vec<CycleOperand>) -> Self {
        Self {
            children,
            mode,
            plan,
            holds: Vec::new(),
            streamed: None,
            streamed_len: None,
            known_len: 0,
            step: 0,
            empty: false,
            initialized: false,
            done: false,
        }
    }

    /// Hold each operand as the plan says. An operand the plan indexes
    /// but whose stream is not addressable is buffered; without a
    /// plan, addressable operands are indexed and the first other
    /// operand streams. An empty operand empties the zip, so the
    /// indexed operands' lengths are checked first and the buffered
    /// operands are drained in ascending bound, stopping at the first
    /// empty one before any other is held.
    fn initialize_cycle(&mut self) {
        let planned = self.plan.len() == self.children.len();
        let mut holds: Vec<Option<Hold>> = Vec::with_capacity(self.children.len());
        let mut buffered: Vec<(usize, Option<u64>)> = Vec::new();
        for (i, child) in self.children.iter().enumerate() {
            let want = if planned {
                self.plan[i].clone()
            } else if child.indexed_len().is_some() {
                CycleOperand::Indexed
            } else if self.streamed.is_none() {
                CycleOperand::Streamed
            } else {
                CycleOperand::Buffered { bound: None }
            };
            let hold = match (want, child.indexed_len()) {
                (CycleOperand::Indexed, Some(len)) => Some(Hold::Indexed(len)),
                (CycleOperand::Streamed, _) if self.streamed.is_none() => {
                    self.streamed = Some(i);
                    Some(Hold::Streamed)
                }
                (CycleOperand::Buffered { bound }, _) => {
                    buffered.push((i, bound));
                    None
                }
                _ => {
                    buffered.push((i, None));
                    None
                }
            };
            holds.push(hold);
        }
        if holds.iter().any(|h| matches!(h, Some(Hold::Indexed(0)))) {
            self.empty = true;
            self.done = true;
            return;
        }
        buffered.sort_by_key(|&(i, bound)| (bound.is_none(), bound, i));
        for (i, _) in buffered {
            let child = &mut self.children[i];
            let mut buf = Vec::new();
            while let Some(t) = child.advance() {
                buf.push(t);
            }
            if buf.is_empty() {
                self.empty = true;
                self.done = true;
                return;
            }
            holds[i] = Some(Hold::Buffered(buf));
        }
        let holds: Vec<Hold> = holds.into_iter().flatten().collect();
        for hold in &holds {
            let len = match hold {
                Hold::Indexed(len) => *len,
                Hold::Buffered(buf) => buf.len() as u64,
                Hold::Streamed => continue,
            };
            self.known_len = self.known_len.max(len);
        }
        self.holds = holds;
    }

    fn advance_cycle(&mut self) -> Option<Tuple> {
        let i = self.step;
        if let Some(len) = self.streamed_len
            && i >= len.max(self.known_len)
        {
            self.done = true;
            return None;
        }
        let mut pulled = None;
        match self.streamed {
            Some(s) => {
                pulled = self.children[s].advance();
                if pulled.is_none() {
                    // The first run-out is the streamed operand's length.
                    let len = *self.streamed_len.get_or_insert(i);
                    if len == 0 || i >= len.max(self.known_len) {
                        self.done = true;
                        return None;
                    }
                    self.children[s].rewind();
                    pulled = self.children[s].advance();
                    if pulled.is_none() {
                        self.done = true;
                        return None;
                    }
                }
            }
            None => {
                if i >= self.known_len {
                    self.done = true;
                    return None;
                }
            }
        }
        let mut out = Tuple::new();
        for (child, hold) in self.children.iter().zip(&self.holds) {
            let t = match hold {
                Hold::Indexed(len) => child.tuple_at(i % len)?,
                Hold::Buffered(buf) => buf[(i % buf.len() as u64) as usize].clone(),
                Hold::Streamed => pulled.take()?,
            };
            extend(&mut out, t);
        }
        self.step += 1;
        Some(out)
    }
}

impl TupleStream for ZipStream {
    fn advance(&mut self) -> Option<Tuple> {
        if self.done {
            return None;
        }
        match self.mode {
            ZipMode::Strict | ZipMode::Truncate => {
                // Pull one tuple from each child; if any returns
                // None, this stream is exhausted.
                let mut out = Tuple::new();
                for child in &mut self.children {
                    match child.advance() {
                        Some(t) => extend(&mut out, t),
                        None => {
                            self.done = true;
                            return None;
                        }
                    }
                }
                Some(out)
            }
            ZipMode::Cycle => {
                if !self.initialized {
                    self.initialize_cycle();
                    self.initialized = true;
                    if self.done {
                        return None;
                    }
                }
                self.advance_cycle()
            }
        }
    }

    fn rewind(&mut self) {
        match self.mode {
            ZipMode::Strict | ZipMode::Truncate => {
                self.children.iter_mut().for_each(|c| c.rewind());
                self.done = false;
            }
            ZipMode::Cycle => {
                if !self.initialized {
                    return;
                }
                self.step = 0;
                if let Some(s) = self.streamed {
                    self.children[s].rewind();
                }
                self.done = self.empty || self.streamed_len == Some(0);
            }
        }
    }

    fn indexed_len(&self) -> Option<u64> {
        if self.children.is_empty() {
            return None;
        }
        let lens: Vec<u64> = self
            .children
            .iter()
            .map(|c| c.indexed_len())
            .collect::<Option<_>>()?;
        Some(match self.mode {
            ZipMode::Strict | ZipMode::Truncate => lens.iter().copied().min().unwrap_or(0),
            ZipMode::Cycle => cycle_length(&lens),
        })
    }

    fn tuple_at(&self, i: u64) -> Option<Tuple> {
        if i >= self.indexed_len()? {
            return None;
        }
        let mut out = Tuple::new();
        for child in &self.children {
            let at = match self.mode {
                ZipMode::Strict | ZipMode::Truncate => i,
                ZipMode::Cycle => i % child.indexed_len()?,
            };
            extend(&mut out, child.tuple_at(at)?);
        }
        Some(out)
    }
}

// ---- UnionStream ----

/// Drain children in order: child 0 fully, then child 1, etc.
struct UnionStream {
    children: Vec<BoxedStream>,
    active_idx: usize,
}

impl UnionStream {
    fn new(children: Vec<BoxedStream>) -> Self {
        Self {
            children,
            active_idx: 0,
        }
    }
}

impl TupleStream for UnionStream {
    fn advance(&mut self) -> Option<Tuple> {
        loop {
            if self.active_idx >= self.children.len() {
                return None;
            }
            if let Some(t) = self.children[self.active_idx].advance() {
                return Some(t);
            }
            // Advance to next child.
            self.active_idx += 1;
        }
    }

    fn rewind(&mut self) {
        self.children.iter_mut().for_each(|c| c.rewind());
        self.active_idx = 0;
    }

    fn indexed_len(&self) -> Option<u64> {
        self.children
            .iter()
            .try_fold(0u64, |acc, c| acc.checked_add(c.indexed_len()?))
    }

    fn tuple_at(&self, i: u64) -> Option<Tuple> {
        let mut offset = i;
        for child in &self.children {
            let len = child.indexed_len()?;
            if offset < len {
                return child.tuple_at(offset);
            }
            offset -= len;
        }
        None
    }
}

// ---- FilterStream ----

struct FilterStream {
    inner: BoxedStream,
    predicate: String,
}

impl FilterStream {
    fn new(inner: BoxedStream, predicate: String) -> Self {
        Self { inner, predicate }
    }
}

impl TupleStream for FilterStream {
    fn advance(&mut self) -> Option<Tuple> {
        loop {
            let candidate = self.inner.advance()?;
            if evaluate_predicate(&self.predicate, &candidate) {
                return Some(candidate);
            }
        }
    }

    fn rewind(&mut self) {
        self.inner.rewind();
    }
}

// ---- OrderStreamingStream ----

/// `order(c, Lex, truncation)` — pass-through optionally
/// capped at truncation tuples.
struct OrderStreamingStream {
    inner: BoxedStream,
    truncation: Option<u64>,
    emitted: u64,
}

impl OrderStreamingStream {
    fn new(inner: BoxedStream, _kind: OrderStreamingKind, truncation: Option<u64>) -> Self {
        Self {
            inner,
            truncation,
            emitted: 0,
        }
    }
}

impl TupleStream for OrderStreamingStream {
    fn advance(&mut self) -> Option<Tuple> {
        if let Some(cap) = self.truncation
            && self.emitted >= cap
        {
            return None;
        }
        let t = self.inner.advance()?;
        self.emitted += 1;
        Some(t)
    }

    fn rewind(&mut self) {
        self.inner.rewind();
        self.emitted = 0;
    }

    fn indexed_len(&self) -> Option<u64> {
        let len = self.inner.indexed_len()?;
        Some(self.truncation.map_or(len, |t| t.min(len)))
    }

    fn tuple_at(&self, i: u64) -> Option<Tuple> {
        if i >= self.indexed_len()? {
            return None;
        }
        self.inner.tuple_at(i)
    }
}

// ---- OrderMaterializeStream ----

/// MATERIALIZATION BARRIER. On first advance, the strategy selects
/// the positions it emits from the input's shape (spec §10.7.8):
/// the `input_index_fn` the compiler propagated from upstream
/// metadata, or a 1-axis Lattice of the observed length when the
/// metadata claims none.
///
/// Over an addressable input (R2) the stream holds only the
/// selection and computes each selected tuple as it is emitted;
/// over any other input it buffers the input first and emits the
/// selected tuples from the buffer.
struct OrderMaterializeStream {
    inner: BoxedStream,
    strategy: StrategyName,
    truncation: Option<u64>,
    input_index_fn: Option<IndexFn>,
    seed: Option<u64>,
    /// The emitted positions, once selected.
    selection: Option<Selection>,
    /// The input's tuples, when the input is not addressable.
    buffer: Option<Vec<Tuple>>,
    pos: u64,
}

impl OrderMaterializeStream {
    fn new(
        inner: BoxedStream,
        strategy: StrategyName,
        truncation: Option<u64>,
        input_index_fn: Option<IndexFn>,
        seed: Option<u64>,
    ) -> Self {
        Self {
            inner,
            strategy,
            truncation,
            input_index_fn,
            seed,
            selection: None,
            buffer: None,
            pos: 0,
        }
    }

    fn select(&mut self) {
        let cardinality = match (&self.input_index_fn, self.inner.indexed_len()) {
            (Some(_), Some(len)) => len,
            _ => {
                let mut buf = Vec::new();
                while let Some(t) = self.inner.advance() {
                    buf.push(t);
                }
                let len = buf.len() as u64;
                self.buffer = Some(buf);
                len
            }
        };
        let index_fn = self.input_index_fn.clone().unwrap_or(IndexFn::Lattice {
            axis_sizes: vec![cardinality],
        });
        let dispatched = crate::iteration::comprehension::strategies::for_name(self.strategy);
        self.selection =
            Some(dispatched.select(&index_fn, cardinality, self.truncation, self.seed));
    }
}

impl TupleStream for OrderMaterializeStream {
    fn advance(&mut self) -> Option<Tuple> {
        if self.selection.is_none() {
            self.select();
        }
        let p = self.selection.as_ref()?.get(self.pos)?;
        self.pos += 1;
        match &self.buffer {
            Some(buf) => buf.get(p as usize).cloned(),
            None => self.inner.tuple_at(p),
        }
    }

    fn rewind(&mut self) {
        self.pos = 0;
    }
}

// ---- Predicate evaluator ----

/// Simple predicate evaluator covering the §10.9.5 catalog.
/// Returns `true` for unrecognized predicates (the
/// conservative choice: keep tuples we can't decide on; the
/// caller's algebra-level predicate analyzer marks unknown
/// patterns Opaque so the optimizer doesn't push them
/// down; the IR interpreter then runs them per-tuple here).
///
/// Implementations:
/// - `{name} OP literal` and `literal OP {name}` for the 6
///   comparison operators.
/// - `p && q`, `p || q`, `!p` (recursive).
/// - `{name} in [v1, v2, ...]` discrete-set membership.
/// - Literal `true` / `false`.
///
/// Anything else evaluates to `true` (passes through). This
/// evaluator serves the IR surfaces; the production `runtime`
/// walker evaluates richer predicates through the scope.
fn evaluate_predicate(predicate: &str, tuple: &Tuple) -> bool {
    let trimmed = predicate.trim();
    // A predicate wrapped in one pair of parentheses, as a folded
    // filter's conjuncts are.
    if let Some(inner) = enclosed(trimmed) {
        return evaluate_predicate(inner, tuple);
    }
    if trimmed.eq_ignore_ascii_case("true") {
        return true;
    }
    if trimmed.eq_ignore_ascii_case("false") {
        return false;
    }
    // Negation.
    if let Some(inner) = trimmed.strip_prefix('!') {
        return !evaluate_predicate(inner.trim(), tuple);
    }
    // Disjunction, which binds looser than conjunction.
    if let Some(parts) = split_top_level(trimmed, "||") {
        return parts.iter().any(|p| evaluate_predicate(p, tuple));
    }
    // Conjunction.
    if let Some(parts) = split_top_level(trimmed, "&&") {
        return parts.iter().all(|p| evaluate_predicate(p, tuple));
    }
    // `{name} in [v1, v2, ...]`
    if let Some(in_pos) = trimmed.find(" in ") {
        let lhs = trimmed[..in_pos].trim();
        let rhs = trimmed[in_pos + 4..].trim();
        if let Some(name) = strip_curly(lhs)
            && let Some(inner) = rhs.strip_prefix('[').and_then(|s| s.strip_suffix(']'))
        {
            let needle = lookup(tuple, &name);
            if needle.is_none() {
                return true; // Unknown coord — pass through.
            }
            return inner.split(',').any(|item| {
                parse_literal(item.trim())
                    .map(|v| values_eq(&lit_to_tuple_value(&v), needle.unwrap()))
                    .unwrap_or(false)
            });
        }
    }
    // Comparison ops: try longest first.
    for (op, op_kind) in [
        ("==", CmpKind::Eq),
        ("!=", CmpKind::Ne),
        ("<=", CmpKind::Le),
        (">=", CmpKind::Ge),
        ("<", CmpKind::Lt),
        (">", CmpKind::Gt),
    ] {
        if let Some((lhs, rhs)) = split_top_level_op(trimmed, op) {
            let lhs = lhs.trim();
            let rhs = rhs.trim();
            // {name} OP literal
            if let (Some(name), Some(lit)) = (strip_curly(lhs), parse_literal(rhs)) {
                let val = lookup(tuple, &name);
                if val.is_none() {
                    return true;
                }
                return compare(val.unwrap(), op_kind, &lit_to_tuple_value(&lit));
            }
            // literal OP {name}
            if let (Some(name), Some(lit)) = (strip_curly(rhs), parse_literal(lhs)) {
                let val = lookup(tuple, &name);
                if val.is_none() {
                    return true;
                }
                // Invert kind: a < b iff b > a.
                let inv = invert_kind(op_kind);
                return compare(val.unwrap(), inv, &lit_to_tuple_value(&lit));
            }
            // {a} OP {b}
            if let (Some(a), Some(b)) = (strip_curly(lhs), strip_curly(rhs)) {
                let va = lookup(tuple, &a);
                let vb = lookup(tuple, &b);
                if va.is_none() || vb.is_none() {
                    return true;
                }
                return compare(va.unwrap(), op_kind, vb.unwrap());
            }
        }
    }
    true
}

#[derive(Clone, Copy)]
enum CmpKind {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

fn invert_kind(k: CmpKind) -> CmpKind {
    match k {
        CmpKind::Lt => CmpKind::Gt,
        CmpKind::Le => CmpKind::Ge,
        CmpKind::Gt => CmpKind::Lt,
        CmpKind::Ge => CmpKind::Le,
        other => other,
    }
}

fn lookup<'a>(tuple: &'a Tuple, name: &str) -> Option<&'a TupleValue> {
    tuple
        .bindings
        .iter()
        .find(|(k, _)| k == name)
        .map(|(_, v)| v)
}

fn compare(a: &TupleValue, kind: CmpKind, b: &TupleValue) -> bool {
    let ord = match (a, b) {
        (TupleValue::I64(a), TupleValue::I64(b)) => a.cmp(b),
        (TupleValue::U64(a), TupleValue::U64(b)) => a.cmp(b),
        (TupleValue::F64(a), TupleValue::F64(b)) => {
            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
        }
        (TupleValue::I64(a), TupleValue::F64(b)) => (*a as f64)
            .partial_cmp(b)
            .unwrap_or(std::cmp::Ordering::Equal),
        (TupleValue::F64(a), TupleValue::I64(b)) => a
            .partial_cmp(&(*b as f64))
            .unwrap_or(std::cmp::Ordering::Equal),
        (TupleValue::Str(a), TupleValue::Str(b)) => a.cmp(b),
        (TupleValue::Bool(a), TupleValue::Bool(b)) => a.cmp(b),
        _ => return false,
    };
    match kind {
        CmpKind::Eq => ord.is_eq(),
        CmpKind::Ne => !ord.is_eq(),
        CmpKind::Lt => ord.is_lt(),
        CmpKind::Le => ord.is_le(),
        CmpKind::Gt => ord.is_gt(),
        CmpKind::Ge => ord.is_ge(),
    }
}

fn values_eq(a: &TupleValue, b: &TupleValue) -> bool {
    compare(a, CmpKind::Eq, b)
}

fn strip_curly(s: &str) -> Option<String> {
    let s = s.trim();
    if s.starts_with('{') && s.ends_with('}') {
        let inner = &s[1..s.len() - 1];
        let trimmed = inner.trim();
        if trimmed.chars().all(|c| c.is_alphanumeric() || c == '_') && !trimmed.is_empty() {
            return Some(trimmed.to_string());
        }
    }
    None
}

fn parse_literal(s: &str) -> Option<LiteralValue> {
    let s = s.trim();
    if s.eq_ignore_ascii_case("true") {
        return Some(LiteralValue::Bool(true));
    }
    if s.eq_ignore_ascii_case("false") {
        return Some(LiteralValue::Bool(false));
    }
    if s.len() >= 2
        && ((s.starts_with('"') && s.ends_with('"')) || (s.starts_with('\'') && s.ends_with('\'')))
    {
        return Some(LiteralValue::String(s[1..s.len() - 1].to_string()));
    }
    if let Ok(n) = s.parse::<i64>() {
        return Some(LiteralValue::Int(n));
    }
    if let Ok(f) = s.parse::<f64>() {
        return Some(LiteralValue::Float(f));
    }
    None
}

fn lit_to_tuple_value(lv: &LiteralValue) -> TupleValue {
    literal_to_tuple_value(lv)
}

/// The text inside `s` when one pair of parentheses encloses all of it.
fn enclosed(s: &str) -> Option<&str> {
    let inner = s.strip_prefix('(')?.strip_suffix(')')?;
    let mut depth = 0i64;
    for b in inner.bytes() {
        match b {
            b'(' => depth += 1,
            b')' => {
                depth -= 1;
                if depth < 0 {
                    return None;
                }
            }
            _ => {}
        }
    }
    (depth == 0).then_some(inner)
}

fn split_top_level(s: &str, sep: &str) -> Option<Vec<String>> {
    let mut parts = Vec::new();
    let mut depth = 0i64;
    let mut last = 0usize;
    let bytes = s.as_bytes();
    let sep_bytes = sep.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'(' | b'[' | b'{' => depth += 1,
            b')' | b']' | b'}' => depth -= 1,
            _ => {}
        }
        if depth == 0
            && i + sep_bytes.len() <= bytes.len()
            && &bytes[i..i + sep_bytes.len()] == sep_bytes
        {
            parts.push(s[last..i].trim().to_string());
            last = i + sep_bytes.len();
            i = last;
            continue;
        }
        i += 1;
    }
    if parts.is_empty() {
        return None;
    }
    parts.push(s[last..].trim().to_string());
    Some(parts)
}

fn split_top_level_op<'a>(s: &'a str, op: &str) -> Option<(&'a str, &'a str)> {
    let mut depth = 0i64;
    let bytes = s.as_bytes();
    let op_bytes = op.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'(' | b'[' | b'{' => depth += 1,
            b')' | b']' | b'}' => depth -= 1,
            _ => {}
        }
        if depth == 0
            && i + op_bytes.len() <= bytes.len()
            && &bytes[i..i + op_bytes.len()] == op_bytes
        {
            if op.len() == 1 {
                let next = bytes.get(i + 1).copied();
                if next == Some(b'=') {
                    i += 1;
                    continue;
                }
            }
            return Some((&s[..i], &s[i + op_bytes.len()..]));
        }
        i += 1;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::iteration::comprehension::ast::Comprehension;
    use crate::iteration::comprehension::ir::compile;
    use crate::iteration::comprehension::source::{LiteralValue, Source};

    fn clause(name: &str, vs: &[i64]) -> Comprehension {
        Comprehension::clause(
            name,
            Source::Literal {
                values: vs.iter().map(|n| LiteralValue::Int(*n)).collect(),
            },
        )
    }

    fn collect(stream: &mut BoxedStream) -> Vec<Tuple> {
        let mut out = Vec::new();
        while let Some(t) = stream.advance() {
            out.push(t);
        }
        out
    }

    #[test]
    fn single_clause_dispense() {
        let ast = clause("k", &[1, 2, 3]);
        let prog = compile(&ast);
        let mut stream = interpret(&prog);
        let tuples = collect(&mut stream);
        assert_eq!(tuples.len(), 3);
        assert_eq!(tuples[0].bindings[0].1, TupleValue::I64(1));
        assert_eq!(tuples[2].bindings[0].1, TupleValue::I64(3));
    }

    #[test]
    fn cartesian_2d_lex_order() {
        let ast = Comprehension::cartesian(vec![clause("a", &[1, 2]), clause("b", &[10, 20])]);
        let prog = compile(&ast);
        let mut stream = interpret(&prog);
        let tuples = collect(&mut stream);
        assert_eq!(tuples.len(), 4);
        // Lex: (a=1, b=10), (a=1, b=20), (a=2, b=10), (a=2, b=20)
        assert_eq!(tuples[0].bindings[0].1, TupleValue::I64(1));
        assert_eq!(tuples[0].bindings[1].1, TupleValue::I64(10));
        assert_eq!(tuples[1].bindings[1].1, TupleValue::I64(20));
        assert_eq!(tuples[2].bindings[0].1, TupleValue::I64(2));
    }

    #[test]
    fn zip_strict_3() {
        let ast = Comprehension::zip(
            vec![clause("x", &[1, 2, 3]), clause("y", &[10, 20, 30])],
            ZipMode::Strict,
        );
        let prog = compile(&ast);
        let mut stream = interpret(&prog);
        let tuples = collect(&mut stream);
        assert_eq!(tuples.len(), 3);
        assert_eq!(tuples[0].bindings[0].1, TupleValue::I64(1));
        assert_eq!(tuples[0].bindings[1].1, TupleValue::I64(10));
        assert_eq!(tuples[2].bindings[1].1, TupleValue::I64(30));
    }

    #[test]
    fn zip_truncate_shortest() {
        let ast = Comprehension::zip(
            vec![clause("x", &[1, 2, 3, 4]), clause("y", &[10, 20])],
            ZipMode::Truncate,
        );
        let prog = compile(&ast);
        let mut stream = interpret(&prog);
        let tuples = collect(&mut stream);
        assert_eq!(tuples.len(), 2);
    }

    #[test]
    fn union_drains_in_order() {
        let ast = Comprehension::union(vec![clause("k", &[1, 2]), clause("k", &[10, 20])]);
        let prog = compile(&ast);
        let mut stream = interpret(&prog);
        let tuples = collect(&mut stream);
        assert_eq!(tuples.len(), 4);
        assert_eq!(tuples[0].bindings[0].1, TupleValue::I64(1));
        assert_eq!(tuples[2].bindings[0].1, TupleValue::I64(10));
    }

    #[test]
    fn filter_keeps_only_matching() {
        let cart = Comprehension::cartesian(vec![clause("k", &[1, 2, 3, 4, 5])]);
        let ast = Comprehension::filter(cart, "{k} > 2");
        let prog = compile(&ast);
        let mut stream = interpret(&prog);
        let tuples = collect(&mut stream);
        assert_eq!(tuples.len(), 3);
        for t in &tuples {
            match t.bindings[0].1 {
                TupleValue::I64(n) => assert!(n > 2),
                _ => panic!(),
            }
        }
    }

    #[test]
    fn order_streaming_lex_truncates() {
        let ast = Comprehension::order(clause("k", &[1, 2, 3, 4, 5]), StrategyName::Lex, Some(2));
        let prog = compile(&ast);
        let mut stream = interpret(&prog);
        let tuples = collect(&mut stream);
        assert_eq!(tuples.len(), 2);
        assert_eq!(tuples[0].bindings[0].1, TupleValue::I64(1));
        assert_eq!(tuples[1].bindings[0].1, TupleValue::I64(2));
    }

    #[test]
    fn order_materialize_shuffle_produces_full_set() {
        let ast = Comprehension::order(clause("k", &[1, 2, 3, 4, 5]), StrategyName::Shuffle, None);
        let prog = compile(&ast);
        let mut stream = interpret(&prog);
        let tuples = collect(&mut stream);
        assert_eq!(tuples.len(), 5);
        // All original values must be present (just permuted).
        let mut sorted: Vec<i64> = tuples
            .iter()
            .map(|t| match t.bindings[0].1 {
                TupleValue::I64(n) => n,
                _ => panic!(),
            })
            .collect();
        sorted.sort();
        assert_eq!(sorted, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn dispense_sequence_for_section_11_1() {
        // Spec §11.1: cartesian over (k in 1..=2) × (b in 10..=20 step 10).
        let ast = Comprehension::cartesian(vec![clause("k", &[1, 2]), clause("b", &[10, 20])]);
        let prog = compile(&ast);
        let mut stream = interpret(&prog);
        let tuples = collect(&mut stream);
        // 2 × 2 = 4 tuples in Lex.
        assert_eq!(tuples.len(), 4);
    }

    fn ints(tuples: &[Tuple]) -> Vec<Vec<i64>> {
        tuples
            .iter()
            .map(|t| {
                t.bindings
                    .iter()
                    .map(|(_, v)| match v {
                        TupleValue::I64(n) => *n,
                        other => panic!("expected an int, got {other:?}"),
                    })
                    .collect()
            })
            .collect()
    }

    /// The cycle zip every plan must produce: operand `c` at
    /// `i mod |c|` for every `i` below the longest operand.
    fn cycled(operands: &[&[i64]]) -> Vec<Vec<i64>> {
        let len = operands.iter().map(|o| o.len()).max().unwrap_or(0);
        if operands.iter().any(|o| o.is_empty()) {
            return Vec::new();
        }
        (0..len)
            .map(|i| operands.iter().map(|o| o[i % o.len()]).collect())
            .collect()
    }

    fn kept(name: &str, vs: &[i64]) -> Comprehension {
        // A filter that keeps everything makes the operand
        // unaddressable without changing its tuples.
        Comprehension::filter(clause(name, vs), format!("{{{name}}} > -1000"))
    }

    /// The streamed operand is shorter than an indexed one, so it is
    /// rewound and replayed rather than buffered.
    #[test]
    fn a_short_streamed_operand_is_rewound() {
        let ast = Comprehension::zip(
            vec![kept("a", &[1, 2]), clause("b", &[10, 20, 30, 40, 50])],
            ZipMode::Cycle,
        );
        let tuples = collect(&mut interpret(&compile(&ast)));
        assert_eq!(ints(&tuples), cycled(&[&[1, 2], &[10, 20, 30, 40, 50]]));
    }

    /// Indexed, buffered, and streamed operands in one zip.
    #[test]
    fn every_hold_cycles_to_the_longest_operand() {
        let shapes: [[&[i64]; 3]; 4] = [
            [&[1, 2, 3], &[1, 2, 3, 4, 5], &[7]],
            [&[1, 2, 3, 4, 5, 6, 7], &[1, 2], &[7, 8, 9]],
            [&[1], &[1, 2], &[7, 8, 9, 10, 11]],
            [&[1, 2], &[], &[7, 8, 9]],
        ];
        for [x, y, z] in shapes {
            let ast = Comprehension::zip(
                vec![kept("x", x), kept("y", y), clause("z", z)],
                ZipMode::Cycle,
            );
            let mut stream = interpret(&compile(&ast));
            let first = collect(&mut stream);
            assert_eq!(ints(&first), cycled(&[x, y, z]), "{x:?} {y:?} {z:?}");
            stream.rewind();
            assert_eq!(collect(&mut stream), first, "a rewound zip replays");
        }
    }

    /// An operand empty at open, known from its metadata or found by
    /// its filter, in any position, empties the zip, before and after
    /// a rewind.
    #[test]
    fn an_empty_operand_empties_the_zip() {
        let empties = [
            clause("e", &[]),
            Comprehension::filter(clause("e", &[1, 2]), "{e} > 5"),
            Comprehension::filter(clause("e", &[1, 2]), "false"),
        ];
        for empty in empties {
            for at in 0..3 {
                let mut children = vec![kept("a", &[1, 2, 3, 4]), clause("b", &[7, 8])];
                children.insert(at, empty.clone());
                let ast = Comprehension::zip(children, ZipMode::Cycle);
                let program = compile(&ast);
                let mut stream = interpret(&program);
                assert!(collect(&mut stream).is_empty(), "{ast:?}");
                stream.rewind();
                assert!(collect(&mut stream).is_empty(), "{ast:?}");
                // The empty literal is known empty from its metadata, so
                // the zip holds nothing.
                if matches!(empty, Comprehension::Clause { .. }) {
                    assert!(program.ops().iter().all(|op| !op.is_barrier()), "{ast:?}");
                }
            }
        }
    }

    /// A predicate in parentheses, as chained filters fold to, and a
    /// disjunction binding looser than a conjunction.
    #[test]
    fn predicates_group_as_written() {
        let ks = clause("k", &[1, 2, 3, 4, 5]);
        let folded = Comprehension::filter(Comprehension::filter(ks.clone(), "{k} > 1"), "false");
        let optimized = crate::iteration::comprehension::optimize::optimize(folded);
        assert!(collect(&mut interpret(&compile(&optimized))).is_empty());
        let mixed = Comprehension::filter(ks, "{k} == 1 || {k} > 2 && {k} < 4");
        let tuples = collect(&mut interpret(&compile(&mixed)));
        assert_eq!(ints(&tuples), vec![vec![1], vec![3]]);
    }

    /// Over an addressable input a strategy computes only the tuples
    /// it selects: `halton/5` over a 10^12-tuple product.
    #[test]
    fn an_order_over_an_addressable_product_computes_only_its_selection() {
        let range = |name: &str| {
            Comprehension::clause(
                name,
                Source::IntRange {
                    lo: 0,
                    hi: 1_000_000,
                    step: 1,
                },
            )
        };
        let ast = Comprehension::order(
            Comprehension::cartesian(vec![range("a"), range("b")]),
            StrategyName::Halton,
            Some(5),
        );
        let tuples = collect(&mut interpret(&compile(&ast)));
        assert_eq!(tuples.len(), 5);
    }

    /// A cartesian over addressable children answers any position,
    /// matching what it dispenses.
    #[test]
    fn an_addressable_cartesian_answers_positions() {
        let ast = Comprehension::cartesian(vec![
            clause("a", &[1, 2]),
            Comprehension::union(vec![clause("b", &[10]), clause("b", &[20, 30])]),
        ]);
        let mut stream = interpret(&compile(&ast));
        assert_eq!(stream.indexed_len(), Some(6));
        let at: Vec<Tuple> = (0..6).map(|i| stream.tuple_at(i).unwrap()).collect();
        assert_eq!(collect(&mut stream), at);
    }
}
