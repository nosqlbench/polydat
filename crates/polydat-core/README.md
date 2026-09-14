# polydat-core

The Polydat runtime: the value model, the graph compiler, the
execution engines, the kernels, the comprehension runtime, the node
macro's support surface, the nodes the compiler itself synthesizes,
and the numeric bodies the native lowerings share with the node
library. Programs link `polydat`, which re-exports this crate and the
node library at the paths they always had.
