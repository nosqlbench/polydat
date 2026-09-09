#[polydat::polydat_node(category = Math)]
fn f(a: u64, #[poly_default(8u64)] size: u64) -> u64 { a + size }
fn main() {}
