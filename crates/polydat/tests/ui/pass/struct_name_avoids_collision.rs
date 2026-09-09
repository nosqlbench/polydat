// A node may produce a value type that carries the node's own name:
// `struct_name` keeps the generated struct out of the way.
use polydat::ast::ReflectedValue;
use polydat::derive_support::Ext;

#[derive(Debug, Clone)]
struct GeoCell { level: u64 }

impl ReflectedValue for GeoCell {
    fn type_name(&self) -> &str { "GeoCell" }
    fn display(&self) -> String { format!("L{}", self.level) }
    fn clone_reflected(&self) -> Box<dyn ReflectedValue> { Box::new(self.clone()) }
    fn as_any(&self) -> &dyn std::any::Any { self }
}

#[polydat::polydat_node(category = Math, struct_name = GeoCellNode)]
fn geo_cell(level: u64) -> Ext<GeoCell> { Ext(GeoCell { level }) }

fn main() {
    let _node: GeoCellNode = GeoCellNode::default();
}
