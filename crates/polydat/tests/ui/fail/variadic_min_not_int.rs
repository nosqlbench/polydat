#[polydat::polydat_node(category = Math, variadic_min = "1")]
fn f(values: &[polydat::ast::Value]) -> u64 { values.len() as u64 }
fn main() {}
