#[polydat::polydat_node(category = Math)]
fn f(#[poly_default(1u64)] v: polydat::ast::Value) -> polydat::ast::Value { v }
fn main() {}
