#[polydat::polydat_node(category = Math, struct_name = Pass, instantiate(u64, f64))]
fn passthrough<T: polydat::derive_support::Wire>(a: T) -> T { a }
fn main() {}
