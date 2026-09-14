# polydat-grammar

The Polydat language without its runtime: the lexer, the parser, the
AST, the pretty-printer, the comprehension sub-language and its
algebra, the tile template parsers, and the port type vocabulary.
`polydat` depends on this crate and re-exports every module at the
paths it always had (`polydat::dsl::ast`, `polydat::dsl::parser`,
`polydat::iteration::comprehension::parse`, `polydat::ast::PortType`),
so a program that compiles against `polydat` is unchanged; a tool
that only needs to read, check, or print Polydat source links this
crate alone.
