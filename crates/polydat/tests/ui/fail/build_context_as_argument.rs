#[polydat::polydat_node(category = Context)]
fn f(a: u64, ctx: &polydat::dsl::factory::BuildContext) -> u64 {
    a + ctx.bindings().len() as u64
}
fn main() {}
