fn build(n: u64) -> u64 { n }
#[polydat::polydat_node(category = Math)]
fn f(a: u64, #[poly_const(build, into = size)] size: polydat::derive_support::Const<u64>) -> u64 { a + *size }
fn main() {}
