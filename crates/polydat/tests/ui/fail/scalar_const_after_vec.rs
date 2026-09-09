#[polydat::polydat_node(category = Math)]
fn f(a: u64, xs: polydat::derive_support::Const<Vec<u64>>, n: polydat::derive_support::Const<u64>) -> u64 { a + xs.len() as u64 + *n }
fn main() {}
