use polydat::dsl::factory::BuildContext;
fn capture(ctx: &BuildContext) -> String { ctx.binding().unwrap_or("").to_string() }
#[polydat::polydat_node(category = Context)]
fn f(
    ctx: polydat::derive_support::Const<u64>,
    #[poly_const(capture, from = ctx)] origin: &String,
) -> String { format!("{origin}{}", *ctx) }
fn main() {}
