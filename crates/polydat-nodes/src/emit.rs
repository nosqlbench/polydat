// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Row emission as a graph node.
//!
//! The `polydat` binary's `--emit` option is a source transform: it
//! appends one binding, `__emit := emit_row(format, names, a, b, ...)`,
//! to the program. The node sees every wire it names through ordinary
//! wiring, so emission is part of the kernel rather than a decorator
//! around it. When the `for` construct lands, the same binding is
//! inserted inside the traversed block and observes that block's
//! scope.
//!
//! Rows accumulate in a thread-local buffer so concurrent fibers never
//! contend. The harness drains each fiber's buffer with [`take_rows`]
//! at chunk boundaries and decides on ordering.

use std::cell::RefCell;

use polydat::ast::Value;

thread_local! {
    static ROWS: RefCell<Vec<String>> = const { RefCell::new(Vec::new()) };
}

/// Formats supported by `emit_row`. Parsed from the node's `format`
/// constant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmitFormat {
    /// `name=value` pairs separated by single spaces.
    Map,
    /// Comma-separated values; strings are quoted when they contain a
    /// comma, quote, or newline.
    Csv,
    /// One JSON object per row.
    Jsonl,
    /// The values' text as it is, one per line. This is how a tile is
    /// emitted: `--emit tile:<name>` selects the tile and this format.
    Text,
}

impl EmitFormat {
    /// The format named by `s`, case-insensitively, if any.
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "map" => Some(Self::Map),
            "csv" => Some(Self::Csv),
            "json" | "jsonl" => Some(Self::Jsonl),
            "text" => Some(Self::Text),
            _ => None,
        }
    }
}

/// The header line for a format, if the format has one.
pub fn header(format: EmitFormat, names: &[&str]) -> Option<String> {
    match format {
        EmitFormat::Csv => Some(names.join(",")),
        EmitFormat::Map | EmitFormat::Jsonl | EmitFormat::Text => None,
    }
}

/// Render one row without touching the buffer.
pub fn render_row(format: EmitFormat, names: &[&str], values: &[Value]) -> String {
    match format {
        EmitFormat::Text => values
            .iter()
            .map(|v| v.to_display_string())
            .collect::<Vec<_>>()
            .join("\n"),
        EmitFormat::Map => names
            .iter()
            .zip(values)
            .map(|(n, v)| format!("{n}={}", v.to_display_string()))
            .collect::<Vec<_>>()
            .join(" "),
        EmitFormat::Csv => values
            .iter()
            .map(|v| csv_cell(&v.to_display_string()))
            .collect::<Vec<_>>()
            .join(","),
        EmitFormat::Jsonl => {
            let mut map = serde_json::Map::with_capacity(names.len());
            for (n, v) in names.iter().zip(values) {
                map.insert((*n).to_string(), v.to_json_value());
            }
            serde_json::Value::Object(map).to_string()
        }
    }
}

fn csv_cell(s: &str) -> String {
    if s.contains([',', '"', '\n']) {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

/// Drain the calling thread's emitted rows.
pub fn take_rows() -> Vec<String> {
    ROWS.with(|r| std::mem::take(&mut *r.borrow_mut()))
}

/// Append one formatted row to the calling thread's buffer and return
/// the number of values emitted. `format` is `map`, `csv`, or `jsonl`;
/// `names` is the comma-separated list of wire names in the same order
/// as the variadic values.
#[polydat::polydat_node(category = Diagnostic, purity = SideChannel(Other), variadic_min = 0)]
fn emit_row(format: Const<&str>, names: Const<&str>, values: &[Value]) -> u64 {
    let fmt = EmitFormat::parse(&format).unwrap_or(EmitFormat::Map);
    let names: Vec<&str> = names
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .collect();
    let row = render_row(fmt, &names, values);
    ROWS.with(|r| r.borrow_mut().push(row));
    values.len() as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use polydat::ast::PolydatNode;

    #[test]
    fn csv_quotes_only_when_needed() {
        let vals = [
            Value::U64(1),
            Value::Str("a,b".into()),
            Value::Str("plain".into()),
        ];
        let row = render_row(EmitFormat::Csv, &["x", "y", "z"], &vals);
        assert_eq!(row, "1,\"a,b\",plain");
    }

    #[test]
    fn map_and_jsonl_shapes() {
        let vals = [Value::U64(7), Value::F64(1.5)];
        assert_eq!(render_row(EmitFormat::Map, &["a", "b"], &vals), "a=7 b=1.5");
        assert_eq!(
            render_row(EmitFormat::Jsonl, &["a", "b"], &vals),
            r#"{"a":7,"b":1.5}"#
        );
    }

    #[test]
    fn rows_accumulate_per_thread_and_drain() {
        let node = EmitRow::new("csv".to_string(), "a".to_string(), 1);
        let mut out = [Value::None];
        node.eval(&[Value::U64(3)], &mut out);
        node.eval(&[Value::U64(4)], &mut out);
        assert_eq!(out[0].as_u64(), 1);
        assert_eq!(take_rows(), vec!["3".to_string(), "4".to_string()]);
        assert!(take_rows().is_empty());
    }
}
