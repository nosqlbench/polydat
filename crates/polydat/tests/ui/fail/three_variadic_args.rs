#[polydat::polydat_node(category = Math)]
fn f(a: &[u64], b: &[u64], c: &[u64]) -> u64 { (a.len() + b.len() + c.len()) as u64 }
fn main() {}
