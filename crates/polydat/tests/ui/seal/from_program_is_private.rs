// The subcontext seal (kernel/subcontext): `PolydatKernel::from_program`
// is crate-private, so a kernel is only instanced through the typed
// surface. Must not compile.
use polydat::dsl::compile::compile_polydat_interpreter;
use polydat::kernel::PolydatKernel;
fn main() {
    let kernel = compile_polydat_interpreter("input cycle: u64\n").unwrap();
    let _ = PolydatKernel::from_program(kernel.program().clone());
}
