#[polydat::polydat_node(category = Math)]
fn f(a: u64) -> polydat::derive_support::DynamicOutputs<u64> { polydat::derive_support::DynamicOutputs(vec![a]) }
fn main() {}
