// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! The value a producer wire carries (SRD 113 §3.1).
//!
//! `name := for ...` binds a comprehension as a value. The wire is a
//! `Streamer`, realized as a reflected `Ext` value so it rides the
//! existing type-erased extension point of the type plane. It carries
//! the comprehension's text and validated algebra AST, and exposes
//! `compiled()` (from which all three consumption surfaces hang) and
//! `coordinate_stream()` as factories. Every factory call compiles
//! a fresh stream, so streams obtained from one wire never share
//! dispense state (§9.5.2 of Comprehension Forms).

use serde::{Deserialize, Serialize};

use crate::ast::{ReflectedValue, Value};
use crate::iteration::comprehension::ast::Comprehension;
use crate::iteration::comprehension::cardinality::CardinalityClass;
use crate::iteration::comprehension::metadata::Metadata;
use crate::iteration::comprehension::surfaces::{CompiledComprehension, CoordinateStream, compile};
use crate::iteration::comprehension::validate::ValidationError;

/// A comprehension bound as a value.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StreamerValue {
    /// The text after `for`, as the author wrote it (or, for a derived
    /// producer, the derivation text).
    pub text: String,
    /// The comprehension, with any derivation already applied.
    pub ast: Comprehension,
}

impl StreamerValue {
    /// A streamer over `ast` with its source text.
    pub fn new(text: impl Into<String>, ast: Comprehension) -> Self {
        Self {
            text: text.into(),
            ast,
        }
    }

    /// Element names in tuple order.
    pub fn element_names(&self) -> Vec<String> {
        self.ast.coordinate_names()
    }

    /// The algebra's metadata: cardinality, index addressability, and
    /// natural order.
    pub fn metadata(&self) -> Metadata {
        self.ast.metadata()
    }

    /// The cardinality class of the tuple space.
    pub fn cardinality(&self) -> CardinalityClass {
        self.metadata().cardinality
    }

    /// Compile to the shared IR: validation, optimization, then the
    /// AST → IR pass. Each call is independent. A comprehension the
    /// `for` lowering resolved was validated then and fails here only
    /// when a source needs a scope, which this surface has none of
    /// (comprehension_forms.md §9.5.2);
    /// one built programmatically is validated here.
    pub fn compiled(&self) -> Result<CompiledComprehension, ValidationError> {
        compile(&self.ast)
    }

    /// A fresh coordinate stream with its own dispense cursor.
    pub fn coordinate_stream(&self) -> Result<CoordinateStream, ValidationError> {
        Ok(self.compiled()?.coordinate_stream())
    }

    /// Serialize for transport through a const node argument. The
    /// compiler lowers a producer binding to `streamer("<json>")`.
    pub fn to_json(&self) -> String {
        serde_json::to_string(self).expect("StreamerValue serializes")
    }

    /// Build from a node argument: either the JSON payload the compiler
    /// emits for `for` expressions, or plain comprehension text such as
    /// `k in 1..4` when an author calls `streamer` directly. Panics
    /// with the parser's diagnostic when neither form applies.
    pub fn from_json(payload: &str) -> Self {
        let trimmed = payload.trim();
        if trimmed.starts_with('{') {
            match serde_json::from_str::<Self>(trimmed) {
                Ok(v) => return v,
                Err(e) => panic!("streamer: malformed comprehension payload: {e}"),
            }
        }
        match Self::parse_text(trimmed) {
            Ok(v) => v,
            Err(e) => panic!("streamer: `{trimmed}` is not a comprehension: {e}"),
        }
    }

    /// Parse canonical comprehension text into a streamer value.
    pub fn parse_text(text: &str) -> Result<Self, String> {
        let legacy = crate::iteration::comprehension::parse::parse_comprehension_text(text)?;
        let ast = crate::iteration::comprehension::spec::legacy_to_algebra(&legacy)
            .map_err(|e| e.to_string())?;
        Ok(Self::new(text, ast))
    }
}

impl ReflectedValue for StreamerValue {
    fn type_name(&self) -> &str {
        "Streamer"
    }

    fn display(&self) -> String {
        format!("for {}", self.text)
    }

    fn to_json_value(&self) -> serde_json::Value {
        serde_json::json!({
            "for": self.text,
            "elements": self.element_names(),
            "cardinality": format!("{:?}", self.cardinality()),
        })
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn clone_reflected(&self) -> Box<dyn ReflectedValue> {
        Box::new(self.clone())
    }
}

impl Value {
    /// Wrap a [`StreamerValue`] as a `Value::Ext`.
    pub fn from_streamer(s: StreamerValue) -> Self {
        Value::Ext(Box::new(s))
    }

    /// Downcast to a [`StreamerValue`]. `None` if the value is not a
    /// streamer.
    pub fn as_streamer(&self) -> Option<&StreamerValue> {
        match self {
            Value::Ext(b) => b.as_any().downcast_ref::<StreamerValue>(),
            _ => None,
        }
    }
}
