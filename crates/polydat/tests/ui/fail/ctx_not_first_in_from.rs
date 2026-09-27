use polydat::dsl::factory::BuildContext;
fn capture(key: &str, ctx: &BuildContext) -> String { format!("{key}{:?}", ctx.binding()) }
#[polydat::polydat_node(category = Context)]
fn f(
    key: polydat::derive_support::Const<&str>,
    #[poly_const(capture, from = (key, ctx))] origin: &String,
) -> String { origin.clone() }
fn main() {}
