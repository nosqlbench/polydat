#[polydat::polydat_node(category = Math)]
fn f(a: u64, xs: polydat::derive_support::Const<Vec<u64>>, ys: polydat::derive_support::Const<Vec<u64>>) -> u64 { a + xs.len() as u64 + ys.len() as u64 }
fn main() {}
