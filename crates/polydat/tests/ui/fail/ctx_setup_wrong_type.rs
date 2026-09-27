fn capture(name: &str) -> String { name.to_string() }
#[polydat::polydat_node(category = Context)]
fn f(a: u64, #[poly_const(capture, from = ctx)] origin: &String) -> String {
    format!("{origin}{a}")
}
fn main() {}
