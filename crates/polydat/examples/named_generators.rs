// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Illustration: every named generator in source position. Each call
//! is bound as a producer, `s := for x in <call>`, and the example
//! prints the cardinality the producer reports and the values it
//! yields. A call whose arguments are out of range, or whose terms
//! pass `u64::MAX`, is a compile error.

use polydat::iteration::comprehension::strategies::TupleValue;

fn main() {
    for call in [
        "fib(8)",
        "fib_until(50)",
        "pow2(6)",
        "pow2_until(100)",
        "binomial(4)",
        "geometric(1, 10, 4)",
        "geometric_until(1, 10, 1000)",
        "linear_starts(0, 1, 4)",
        "linear_steps(0, 1, 4)",
        "log_steps(1, 1000, 4)",
        "log_steps(0, 10, 3)",
        "fib(94)",
    ] {
        let src = format!("input cycle: u64\ns := for x in {call}\n");
        let mut kernel = match polydat::dsl::compile_polydat_kernel(&src) {
            Ok(kernel) => kernel,
            Err(e) => {
                println!("{call:<30} error: {e}");
                continue;
            }
        };
        kernel.set_inputs(&[0]);
        let value = kernel.pull("s");
        let producer = value.as_streamer().expect("a producer");
        let stream = match producer.coordinate_stream() {
            Ok(stream) => stream,
            Err(e) => {
                println!("{call:<30} error: {e}");
                continue;
            }
        };
        let values: Vec<String> = stream
            .map(
                |t| match &t.expect("the producer dispenses").bindings[0].1 {
                    TupleValue::U64(n) => n.to_string(),
                    TupleValue::I64(n) => n.to_string(),
                    TupleValue::F64(f) => f.to_string(),
                    other => format!("{other:?}"),
                },
            )
            .collect();
        println!(
            "{call:<30} {:<11} {}",
            format!("{:?}", producer.cardinality()),
            values.join(", ")
        );
    }
}
