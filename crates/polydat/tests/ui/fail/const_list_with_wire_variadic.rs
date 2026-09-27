#[polydat::polydat_node(category = Math)]
fn f(xs: &[u64], weights: polydat::derive_support::Const<Vec<u64>>) -> u64 { xs.len() as u64 + weights.len() as u64 }
fn main() {}
