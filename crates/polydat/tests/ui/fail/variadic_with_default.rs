#[polydat::polydat_node(category = Math)]
fn f(#[poly_default(1u64)] values: &[polydat::ast::Value]) -> u64 { values.len() as u64 }
fn main() {}
