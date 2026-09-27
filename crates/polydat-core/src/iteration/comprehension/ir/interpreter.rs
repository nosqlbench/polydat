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
//!
//! Predicates and non-`Lex` orders evaluate as a traversal evaluates
//! them, in the empty scope: a filter through [`CompiledPredicate`],
//! and an order through [`evaluate_indexed`] over its input, so a
//! stream and a traversal of one comprehension yield the same tuples.

use crate::ast::Value;
use crate::iteration::comprehension::ast::Comprehension;
use crate::iteration::comprehension::metadata::{CycleOperand, cycle_length};
use crate::iteration::comprehension::predicate::CompiledPredicate;
use crate::iteration::comprehension::runtime::{
    IndexedTuples, RuntimeError, RuntimeTuple, evaluate_indexed,
};
use crate::iteration::comprehension::source::{LiteralValue, Source};
use crate::iteration::comprehension::strategies::{Tuple, TupleValue};
use crate::iteration::comprehension::strategy::ZipMode;
use crate::iteration::comprehension::surfaces::tuple_value_to_polydat_value;
use crate::kernel::interp::NoScope;

use super::op::{Op, OrderStreamingKind};
use super::program::Program;

/// A lazy tuple stream — `advance` returns the next tuple, `None`
/// when the stream is exhausted, or the error it found, at the point
/// it finds it: a strict zip whose operands end apart fails after the
/// tuples before the mismatch.
pub trait TupleStream {
    /// The next tuple, `None` once the stream is exhausted, or the
    /// error that ends it.
    fn advance(&mut self) -> Result<Option<Tuple>, RuntimeError>;

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
                stack.push(Box::new(FilterStream::new(inner, predicate)));
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
                seed,
                input,
                ..
            } => {
                stack.push(Box::new(OrderMaterializeStream::new(
                    Comprehension::order_seeded((**input).clone(), *strategy, *truncation, *seed),
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
    fn advance(&mut self) -> Result<Option<Tuple>, RuntimeError> {
        let t = self.tuple_at(self.pos);
        if t.is_some() {
            self.pos += 1;
        }
        Ok(t)
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
    /// cache children 1..N.
    initialized: bool,
    /// True when axis 0's next value is still to be pulled: at the
    /// start and after a rewind.
    pull_a0: bool,
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
            pull_a0: true,
            done: false,
        }
    }

    /// Cache children 1..N to exhaustion, in order, stopping at the
    /// first empty one: the product is empty, and the axes after it are
    /// never evaluated, as in the traversal.
    fn initialize(&mut self) -> Result<(), RuntimeError> {
        self.cached.clear();
        for i in 1..self.children.len() {
            let mut v = Vec::new();
            while let Some(t) = self.children[i].advance()? {
                v.push(t);
            }
            let empty = v.is_empty();
            self.cached.push(v);
            if empty {
                break;
            }
        }
        Ok(())
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
        self.current_a0 = None;
        self.pull_a0 = true;
        self.done = false;
    }

    fn advance(&mut self) -> Result<Option<Tuple>, RuntimeError> {
        if self.lens.is_some() {
            let t = self.tuple_at(self.pos);
            if t.is_some() {
                self.pos += 1;
            }
            return Ok(t);
        }
        if !self.initialized {
            // The first axis is pulled before the later axes are
            // cached: with no first value the product is empty and the
            // later axes are never evaluated, as in the traversal.
            if self.children.is_empty() {
                self.done = true;
            } else {
                self.current_a0 = self.children[0].advance()?;
                self.pull_a0 = false;
                if self.current_a0.is_none() {
                    self.done = true;
                } else {
                    self.initialize()?;
                    self.done = self.cached.iter().any(|v| v.is_empty());
                }
            }
            self.initialized = true;
        }
        if self.done {
            return Ok(None);
        }
        if self.pull_a0 {
            self.current_a0 = self.children[0].advance()?;
            self.pull_a0 = false;
            if self.current_a0.is_none() || self.cached.iter().any(|v| v.is_empty()) {
                // Any empty axis → empty cartesian.
                self.done = true;
                return Ok(None);
            }
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
            // Axis 0 advances on the next pull, which reports the end
            // or an error at that point.
            self.pull_a0 = true;
        }
        Ok(Some(out))
    }
}

// ---- ZipStream ----

/// Lockstep over N child streams. Strict and Truncate pull one tuple
/// from each child. Truncate stops when any child runs out; Strict
/// stops when every child runs out together, and when one runs out
/// before another it fails with the operands' lengths, the error the
/// traversal evaluators give at open, after the tuples before the
/// mismatch. Cycle runs to its
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
    /// Under Strict and Cycle, the position of the next tuple.
    step: u64,
    /// Under Cycle, an indexed or buffered operand was found empty.
    empty: bool,
    /// Under Strict, the length mismatch found, returned again on
    /// every later pull until a rewind.
    failed: Option<RuntimeError>,
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
            failed: None,
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
    fn initialize_cycle(&mut self) -> Result<(), RuntimeError> {
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
            return Ok(());
        }
        buffered.sort_by_key(|&(i, bound)| (bound.is_none(), bound, i));
        for (i, _) in buffered {
            let child = &mut self.children[i];
            let mut buf = Vec::new();
            while let Some(t) = child.advance()? {
                buf.push(t);
            }
            if buf.is_empty() {
                self.empty = true;
                self.done = true;
                return Ok(());
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
        Ok(())
    }

    fn advance_cycle(&mut self) -> Result<Option<Tuple>, RuntimeError> {
        let i = self.step;
        if let Some(len) = self.streamed_len
            && i >= len.max(self.known_len)
        {
            self.done = true;
            return Ok(None);
        }
        let mut pulled = None;
        match self.streamed {
            Some(s) => {
                pulled = self.children[s].advance()?;
                if pulled.is_none() {
                    // The first run-out is the streamed operand's length.
                    let len = *self.streamed_len.get_or_insert(i);
                    if len == 0 || i >= len.max(self.known_len) {
                        self.done = true;
                        return Ok(None);
                    }
                    self.children[s].rewind();
                    pulled = self.children[s].advance()?;
                    if pulled.is_none() {
                        self.done = true;
                        return Ok(None);
                    }
                }
            }
            None => {
                if i >= self.known_len {
                    self.done = true;
                    return Ok(None);
                }
            }
        }
        let mut out = Tuple::new();
        for (child, hold) in self.children.iter().zip(&self.holds) {
            let t = match hold {
                Hold::Indexed(len) => child.tuple_at(i % len),
                Hold::Buffered(buf) => Some(buf[(i % buf.len() as u64) as usize].clone()),
                Hold::Streamed => pulled.take(),
            };
            let Some(t) = t else {
                return Ok(None);
            };
            extend(&mut out, t);
        }
        self.step += 1;
        Ok(Some(out))
    }

    /// One strict lockstep step: every child's next tuple, the end when
    /// every child ends here, or the length mismatch when some end and
    /// others do not. The mismatch counts each operand's length by
    /// draining the ones that continue, so it names the lengths the
    /// traversal evaluators name.
    fn advance_strict(&mut self) -> Result<Option<Tuple>, RuntimeError> {
        let mut pulled = Vec::with_capacity(self.children.len());
        for child in &mut self.children {
            pulled.push(child.advance()?);
        }
        if pulled.iter().all(Option::is_some) {
            let mut out = Tuple::new();
            for t in pulled.into_iter().flatten() {
                extend(&mut out, t);
            }
            self.step += 1;
            return Ok(Some(out));
        }
        self.done = true;
        if pulled.iter().all(Option::is_none) {
            return Ok(None);
        }
        let mut lengths = Vec::with_capacity(self.children.len());
        for (child, t) in self.children.iter_mut().zip(&pulled) {
            let mut len = self.step;
            if t.is_some() {
                len += 1;
                while child.advance()?.is_some() {
                    len += 1;
                }
            }
            lengths.push(len);
        }
        let error = RuntimeError::ZipLengthMismatch { lengths };
        self.failed = Some(error.clone());
        Err(error)
    }
}

impl TupleStream for ZipStream {
    fn advance(&mut self) -> Result<Option<Tuple>, RuntimeError> {
        if let Some(error) = &self.failed {
            return Err(error.clone());
        }
        if self.done {
            return Ok(None);
        }
        match self.mode {
            ZipMode::Strict => self.advance_strict(),
            ZipMode::Truncate => {
                // Pull one tuple from each child; if any returns
                // None, this stream is exhausted.
                let mut out = Tuple::new();
                for child in &mut self.children {
                    match child.advance()? {
                        Some(t) => extend(&mut out, t),
                        None => {
                            self.done = true;
                            return Ok(None);
                        }
                    }
                }
                Ok(Some(out))
            }
            ZipMode::Cycle => {
                if !self.initialized {
                    self.initialize_cycle()?;
                    self.initialized = true;
                    if self.done {
                        return Ok(None);
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
                self.step = 0;
                self.failed = None;
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
        match self.mode {
            // Operands of different lengths are a mismatch, which only
            // a pull reports, so the zip is not addressable.
            ZipMode::Strict => lens.iter().all(|&n| n == lens[0]).then_some(lens[0]),
            ZipMode::Truncate => lens.iter().copied().min(),
            ZipMode::Cycle => Some(cycle_length(&lens)),
        }
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
    fn advance(&mut self) -> Result<Option<Tuple>, RuntimeError> {
        loop {
            if self.active_idx >= self.children.len() {
                return Ok(None);
            }
            if let Some(t) = self.children[self.active_idx].advance()? {
                return Ok(Some(t));
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

/// Keeps the tuples that pass a predicate, evaluated as a traversal
/// evaluates it ([`CompiledPredicate`]), in the empty scope: the
/// compile refuses a predicate that names anything its tuples do not
/// bind.
struct FilterStream {
    inner: BoxedStream,
    predicate: CompiledPredicate,
    scope: NoScope,
}

impl FilterStream {
    fn new(inner: BoxedStream, predicate: &str) -> Self {
        Self {
            inner,
            predicate: CompiledPredicate::new(predicate),
            scope: NoScope::new(),
        }
    }
}

impl TupleStream for FilterStream {
    fn advance(&mut self) -> Result<Option<Tuple>, RuntimeError> {
        while let Some(candidate) = self.inner.advance()? {
            let bindings: RuntimeTuple = candidate
                .bindings
                .iter()
                .map(|(name, value)| (name.clone(), tuple_value_to_polydat_value(value)))
                .collect();
            if self.predicate.keeps(&bindings, &self.scope)? {
                return Ok(Some(candidate));
            }
        }
        Ok(None)
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
    fn advance(&mut self) -> Result<Option<Tuple>, RuntimeError> {
        if let Some(cap) = self.truncation
            && self.emitted >= cap
        {
            return Ok(None);
        }
        let t = self.inner.advance()?;
        if t.is_some() {
            self.emitted += 1;
        }
        Ok(t)
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

/// MATERIALIZATION BARRIER. On first advance the order is evaluated as
/// a traversal evaluates it ([`evaluate_indexed`]), in the empty
/// scope: the strategy selects positions from the input's evaluated
/// shape and length, and V4 refuses an input shape the strategy does
/// not accept.
///
/// Over an addressable input (R2) the stream holds only the selection
/// and computes each selected tuple as it is emitted; over any other
/// input it holds the input's tuples, and over a continuous axis the
/// samples, as the traversal does.
struct OrderMaterializeStream {
    order: Comprehension,
    state: Selected,
    pos: u64,
}

/// What an order stream holds.
enum Selected {
    /// Not evaluated yet.
    Pending,
    /// The selected tuples, addressed by position.
    Ready(IndexedTuples),
    /// The error the evaluation ended with, returned on every pull.
    Failed(RuntimeError),
}

impl OrderMaterializeStream {
    fn new(order: Comprehension) -> Self {
        Self {
            order,
            state: Selected::Pending,
            pos: 0,
        }
    }
}

impl TupleStream for OrderMaterializeStream {
    fn advance(&mut self) -> Result<Option<Tuple>, RuntimeError> {
        if matches!(self.state, Selected::Pending) {
            self.state = match evaluate_indexed(&self.order, &NoScope::new()) {
                Ok(tuples) => Selected::Ready(tuples),
                Err(error) => Selected::Failed(error),
            };
        }
        match &self.state {
            Selected::Ready(tuples) => {
                let tuple = tuples.get(self.pos).map(|bindings| Tuple {
                    bindings: bindings
                        .iter()
                        .map(|(name, value)| (name.clone(), stream_value(value)))
                        .collect(),
                });
                if tuple.is_some() {
                    self.pos += 1;
                }
                Ok(tuple)
            }
            Selected::Failed(error) => Err(error.clone()),
            Selected::Pending => unreachable!("evaluated above"),
        }
    }

    fn rewind(&mut self) {
        self.pos = 0;
    }
}

/// A traversal's value as the streams carry it: an integer as the
/// `I64` a clause dispenses, and JSON as its text.
fn stream_value(value: &Value) -> TupleValue {
    match value {
        Value::U64(n) => TupleValue::I64(*n as i64),
        Value::I64(n) => TupleValue::I64(*n),
        Value::F64(f) => TupleValue::F64(*f),
        Value::Bool(b) => TupleValue::Bool(*b),
        Value::Str(s) => TupleValue::Str(s.to_string()),
        Value::Json(j) => TupleValue::Str(j.to_string()),
        other => TupleValue::Str(other.to_display_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::iteration::comprehension::ir::compile;
    use crate::iteration::comprehension::strategy::StrategyName;

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
        while let Some(t) = stream.advance().unwrap() {
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
