#[polydat::polydat_node(category = Math, instantiate())]
fn passthrough<T: polydat::derive_support::Wire>(a: T) -> T { a }
fn main() {}
