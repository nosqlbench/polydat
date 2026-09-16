// Copyright 2024-2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! The Polydat language without its runtime.
//!
//! This crate holds what a reader of Polydat source needs and nothing
//! a runner needs: the lexer and parser ([`lexer`], [`parser`]), the
//! AST ([`ast`]), its pretty-printer ([`pprint`]) and visualizer
//! ([`viz`]), the free-name collector ([`refs`]), pragmas
//! ([`pragmas`]), the diagnostic types ([`error`]), the tile
//! template parsers ([`tile`], [`tile_structural`]), the
//! comprehension sub-language with its algebra ([`comprehension`]),
//! and the port type vocabulary ([`PortType`]). `polydat-core`
//! depends on this crate and compiles what it parses; the `polydat`
//! facade re-exports every module here at the path it always had, so
//! `polydat::dsl::ast` and `polydat::ast::PortType` are these.
//!
//! What a type means to a compiled buffer, what a comprehension
//! evaluates to, and what a tile renders are all the runtime's: they
//! live in `polydat-core` (reached through `polydat`), as trait
//! implementations and functions over the types defined here.

pub mod ast;
pub mod comprehension;
pub mod error;
pub mod lexer;
pub mod parser;
pub mod pprint;
pub mod pragmas;
pub mod refs;
pub mod tile;
pub mod tile_structural;
pub mod viz;

mod port_type;
pub use port_type::PortType;
