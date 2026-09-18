// The subcontext seal (kernel/subcontext): `materialize_wiring_from_outer`
// is private to the kernel, so a caller cannot bypass the typed child
// construction. Must not compile.
use polydat::dsl::compile::compile_polydat_interpreter;
fn main() {
    let mut inner = compile_polydat_interpreter("input cycle: u64\n").unwrap();
    let outer = compile_polydat_interpreter("input cycle: u64\n").unwrap();
    inner.materialize_wiring_from_outer(&outer);
}
