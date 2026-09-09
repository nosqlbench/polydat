#[polydat::polydat_node(category = Math)]
fn f(a: u64, n: polydat::derive_support::Const<u64>) -> Result<u64, String> { Ok(a + *n) }
fn main() {}
