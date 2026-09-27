fn build(lo: u64, hi: u64) -> u64 { hi - lo }
#[polydat::polydat_node(category = Math)]
fn f(
    a: u64,
    lo: polydat::derive_support::Const<u64>,
    #[poly_const(build, from = (lo, hi))] span: &u64,
) -> u64 { a + *lo + *span }
fn main() {}
