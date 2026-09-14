// The subcontext seal (kernel/subcontext): `PolydatKernel::from_program`
// is crate-private, so a kernel is only instanced through the typed
// surface. Must not compile.
use polydat::dsl::compile::compile_polydat;
use polydat::kernel::PolydatKernel;
fn main() {
    let kernel = compile_polydat("input cycle: u64\n").unwrap();
    let _ = PolydatKernel::from_program(kernel.program().clone());
}
