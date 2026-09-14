# polydat-nodes

The Polydat node library: every function a program can call that the
compiler does not synthesize itself, as `#[polydat_node]` functions
over `polydat-core`'s value model. Linking the crate registers the
functions; `polydat` re-exports them under `polydat::library`. A
third-party node crate is built the same way.
