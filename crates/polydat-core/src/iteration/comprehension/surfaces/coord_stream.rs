// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! `CoordinateStream` — first-order consumption surface (spec
//! §9.5.1).
//!
//! Dispenses coordinate tuples (`Tuple` from the strategies
//! layer). Each instance holds its own dispense state — the
//! interpreter's tuple stream — but shares the
//! `Arc<Program>` with sibling streamers obtained from the
//! same `CompiledComprehension`.

use std::sync::Arc;

use crate::iteration::comprehension::ir::{Program, TupleStream, interpret};
use crate::iteration::comprehension::runtime::RuntimeError;
use crate::iteration::comprehension::strategies::Tuple;

/// First-order coordinate stream. Each `advance()` yields one
/// coordinate tuple, `None` when the stream is exhausted, or the
/// error that ends it where the stream finds it.
///
/// Construct via [`crate::iteration::comprehension::surfaces::CompiledComprehension::coordinate_stream`].
/// Implements [`Iterator`] over `Result<Tuple, RuntimeError>` so
/// callers can use standard Rust iterator combinators (`.take(n)`,
/// `.collect::<Result<Vec<_>, _>>()`, etc.).
pub struct CoordinateStream {
    /// Held to keep the underlying program alive while this
    /// stream's interpreter graph references it.
    #[allow(dead_code)]
    program: Arc<Program>,
    /// The per-streamer interpreter — independent dispense
    /// state per spec §9.5.2.
    stream: Box<dyn TupleStream>,
    /// The error that ended the stream, returned again by every later
    /// `advance`.
    failed: Option<RuntimeError>,
    /// Set once the iterator has yielded the error, after which it
    /// yields nothing.
    reported: bool,
}

impl CoordinateStream {
    pub(crate) fn new(program: Arc<Program>) -> Self {
        let stream = interpret(&program);
        Self {
            program,
            stream,
            failed: None,
            reported: false,
        }
    }

    /// Pull the next coordinate tuple. Returns `None` when exhausted,
    /// and the error that ends the stream where it is found: a strict
    /// zip whose operands end apart fails after the tuples before the
    /// mismatch, and keeps failing on later pulls.
    pub fn advance(&mut self) -> Result<Option<Tuple>, RuntimeError> {
        if let Some(error) = &self.failed {
            return Err(error.clone());
        }
        let pulled = self.stream.advance();
        if let Err(error) = &pulled {
            self.failed = Some(error.clone());
        }
        pulled
    }
}

impl Iterator for CoordinateStream {
    type Item = Result<Tuple, RuntimeError>;

    /// The next tuple or the error that ends the stream; after an
    /// error, `None`.
    fn next(&mut self) -> Option<Self::Item> {
        if self.reported {
            return None;
        }
        let item = self.advance().transpose();
        self.reported = matches!(item, Some(Err(_)));
        item
    }
}

impl std::fmt::Debug for CoordinateStream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CoordinateStream")
            .field("program_ops", &self.program.len())
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::iteration::comprehension::ast::Comprehension;
    use crate::iteration::comprehension::source::{LiteralValue, Source};
    use crate::iteration::comprehension::strategies::TupleValue;
    use crate::iteration::comprehension::strategy::ZipMode;
    use crate::iteration::comprehension::surfaces::compile;

    fn clause(name: &str, vs: &[i64]) -> Comprehension {
        Comprehension::clause(
            name,
            Source::Literal {
                values: vs.iter().map(|n| LiteralValue::Int(*n)).collect(),
            },
        )
    }

    #[test]
    fn advance_returns_tuples_then_none() {
        let compiled = compile(&clause("k", &[1, 2, 3])).unwrap();
        let mut stream = compiled.coordinate_stream();
        assert!(stream.advance().unwrap().is_some());
        assert!(stream.advance().unwrap().is_some());
        assert!(stream.advance().unwrap().is_some());
        assert!(stream.advance().unwrap().is_none());
        assert!(stream.advance().unwrap().is_none()); // exhausted
    }

    #[test]
    fn iterator_collect() {
        let compiled = compile(&clause("k", &[10, 20, 30])).unwrap();
        let stream = compiled.coordinate_stream();
        let tuples: Vec<Tuple> = stream.collect::<Result<_, _>>().unwrap();
        assert_eq!(tuples.len(), 3);
        assert_eq!(tuples[0].bindings[0].1, TupleValue::I64(10));
        assert_eq!(tuples[2].bindings[0].1, TupleValue::I64(30));
    }

    #[test]
    fn iterator_take_truncates() {
        let compiled = compile(&clause("k", &[1, 2, 3, 4, 5])).unwrap();
        let stream = compiled.coordinate_stream();
        let tuples: Vec<Tuple> = stream.take(2).collect::<Result<_, _>>().unwrap();
        assert_eq!(tuples.len(), 2);
    }

    /// A strict zip whose operands end apart delivers the tuples
    /// before the mismatch, then the error, on every later pull; the
    /// iterator yields the error once and ends.
    #[test]
    fn a_strict_mismatch_fails_where_it_is_found() {
        let ast = Comprehension::zip(
            vec![clause("k", &[1, 2, 3, 4]), clause("c", &[7, 8])],
            ZipMode::Strict,
        );
        let compiled = compile(&ast).unwrap();
        let mut stream = compiled.coordinate_stream();
        assert!(stream.advance().unwrap().is_some());
        assert!(stream.advance().unwrap().is_some());
        let error = stream.advance().unwrap_err();
        assert!(matches!(
            error,
            RuntimeError::ZipLengthMismatch { ref lengths } if *lengths == [4, 2]
        ));
        assert!(stream.advance().is_err());
        let items: Vec<_> = compiled.coordinate_stream().collect();
        assert_eq!(items.len(), 3);
        assert!(items[2].is_err());
    }
}
