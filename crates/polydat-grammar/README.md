# polydat-grammar

The Polydat language without its runtime: the lexer, the parser, the
AST, the pretty-printer, the free-name collector, the diagnostic types,
the comprehension sub-language and its algebra, the tile template
parsers, the AST visualizer, and the port type vocabulary.

Depend on this crate alone when a tool only needs to read, check,
rewrite, or print Polydat source: an editor integration, a linter, a
formatter, a workload generator, or a documentation tool. A program that
compiles and runs kernels depends on
[`polydat`](https://crates.io/crates/polydat), which re-exports every
module here at the path it always had, so `polydat::dsl::ast` and
`polydat::ast::PortType` are these types.

## Example

Lex, parse, and print a program back in canonical form:

```rust
use polydat_grammar::lexer::lex;
use polydat_grammar::parser::parse;
use polydat_grammar::pprint::pp_file;

fn main() -> Result<(), String> {
    let source = r#"
        input cycle: u64
        id := mod(hash(cycle), 1000)
        label := "user-{id}"
    "#;
    let file = parse(lex(source)?)?;
    println!("{}", pp_file(&file));
    Ok(())
}
```

The printer emits the canonical spelling rather than the source text:
binary operators are fully parenthesized, string interpolation is
desugared to `printf`, and literals are normalized. The language's
verification rule is that parsing the printed form and printing it
again is a fixed point.

## What the runtime owns

What a type means to a compiled buffer, what a comprehension evaluates
to, and what a tile renders are decided by the runtime, as trait
implementations and functions over the types defined here. This crate
has no dependency on the runtime or the node library.

## Documentation

The rustdoc on [docs.rs](https://docs.rs/polydat-grammar) covers the
API. The normative language specification, machine-checked against the
parser, is
[`polydat_grammar.md`](https://github.com/nosqlbench/polydat/blob/main/crates/polydat/docs/design/polydat_grammar.md)
in the repository; the comprehension algebra is
[`comprehension_forms.md`](https://github.com/nosqlbench/polydat/blob/main/crates/polydat/docs/design/comprehension_forms.md)
beside it.

## License

Apache-2.0.
