// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! The comprehension sub-language: its text form ([`parse`]), the
//! flat form that text parses to ([`ast_legacy`]), the canonical
//! algebra ([`ast`]) with its sources, strategies, cardinalities, and
//! metadata, and the spec forms that convert between them ([`spec`]).
//! Evaluation is the runtime's.

pub mod ast;
pub mod ast_legacy;
pub mod cardinality;
pub mod metadata;
pub mod parse;
pub mod source;
pub mod spec;
pub mod strategy;

pub use ast::Comprehension;
pub use cardinality::{CardinalityClass, Hybrid, Interval, MeasureName, ProductMeasure};
pub use metadata::{IndexFn, Materialization, Metadata, NaturalOrder};
pub use source::Source;
pub use strategy::{StrategyName, ZipMode};
