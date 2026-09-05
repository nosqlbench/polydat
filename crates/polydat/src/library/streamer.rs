// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! The producer node (SRD 113 §3.1). `name := for ...` lowers to
//! `const name := streamer("<json>")`, where the payload is the
//! resolved comprehension serialized by [`StreamerValue::to_json`].
//! The node parses once at setup and emits the same reflected value on
//! every pull, so the wire is an init-time constant.

use crate::derive_support::Ext;
use crate::iteration::comprehension::streamer_value::StreamerValue;

/// Materialize a comprehension as a `Streamer` value. Authors do not
/// call this directly; the compiler emits it for `for` expressions.
#[crate::polydat_node(category = Variadic)]
fn streamer(
    spec: Const<&str>,
    #[poly_const(StreamerValue::from_json, from = spec)]
    parsed: &StreamerValue,
) -> Ext<StreamerValue> {
    Ext(parsed.clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{PolydatNode, Value};
    use crate::iteration::comprehension::ast::Comprehension;
    use crate::iteration::comprehension::source::Source;

    #[test]
    fn node_round_trips_the_payload() {
        let ast = Comprehension::clause("k", Source::IntRange { lo: 1, hi: 4, step: 1 });
        let payload = StreamerValue::new("k in 1..4", ast).to_json();
        let node = Streamer::new(payload);
        let mut out = [Value::None];
        node.eval(&[], &mut out);
        let s = out[0].as_streamer().expect("streamer value");
        assert_eq!(s.text, "k in 1..4");
        assert_eq!(s.element_names(), vec!["k"]);
    }
}
