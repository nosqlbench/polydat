// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! The comprehension sub-language: the canonical algebra ([`ast`])
//! with its sources, strategies, cardinalities, and metadata, and the
//! forms that reach it ([`spec`]) from text, from a specification
//! document, and from a source expression. The text parser and the
//! flat form it produces on the way are internal to this crate
//! (comprehension_forms.md §14.8). Evaluation is the runtime's.

pub mod ast;
pub mod cardinality;
pub(crate) mod clause_ast;
pub mod metadata;
pub(crate) mod parse;
pub mod source;
pub mod spec;
pub mod strategy;
pub mod to_text;

pub use ast::Comprehension;
pub use cardinality::{CardinalityClass, Hybrid, Interval, MeasureName, ProductMeasure};
pub use metadata::{IndexFn, Materialization, Metadata, NaturalOrder};
pub use source::Source;
pub use strategy::{StrategyName, ZipMode};
