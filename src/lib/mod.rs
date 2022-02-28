#![allow(clippy::cognitive_complexity)]
#![allow(clippy::many_single_char_names)]
#![allow(clippy::needless_doctest_main)]
#![allow(clippy::new_without_default)]
#![allow(clippy::range_plus_one)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]
#![allow(clippy::unnecessary_wraps)]
#![allow(clippy::upper_case_acronyms)]

//! `lrpar` provides a Yacc-compatible parser (where grammars can be generated at
//! compile-time or run-time). It can take in traditional `.y` files and convert
//! them into an idiomatic Rust parser. More details can be found in the [grmtools
//! book](https://softdevteam.github.io/grmtools/master/book); the
//! [quickstart guide](https://softdevteam.github.io/grmtools/master/book/quickstart.html)
//! is a good place to start.
//!
//!
//! ## Example
//!
//! Let's assume we want to statically generate a parser for a simple calculator language (and
//! let's also assume we are able to use
//! [`lrlex`](https://softdevteam.github.io/grmtools/master/book/lrlex.html) for the lexer). We
//! need to add a `build.rs` file to our project which statically compiles both the lexer and
//! parser. While we can perform both steps individually, it's easiest to use `lrlex` which does
//! both jobs for us in one go. Our `build.rs` file thus looks as follows:
//!
//! ```text
//! use cfgrammar::yacc::YaccKind;
//! use lrlex::CTLexerBuilder;
//!
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     CTLexerBuilder::new()
//!         .lrpar_config(|ctp| {
//!             ctp.yacckind(YaccKind::Grmtools)
//!                 .grammar_in_src_dir("calc.y")
//!                 .unwrap()
//!         })
//!         .lexer_in_src_dir("calc.l")?
//!         .build()?;
//!     Ok(())
//! }
//! ```
//!
//! where `src/calc.l` is as follows:
//!
//! ```text
//! %%
//! [0-9]+ "INT"
//! \+ "+"
//! \* "*"
//! \( "("
//! \) ")"
//! [\t ]+ ;
//! ```
//!
//! and `src/calc.y` is as follows:
//!
//! ```text
//! %start Expr
//! %avoid_insert "INT"
//! %%
//! Expr -> Result<u64, ()>:
//!       Expr '+' Term { Ok($1? + $3?) }
//!     | Term { $1 }
//!     ;
//!
//! Term -> Result<u64, ()>:
//!       Term '*' Factor { Ok($1? * $3?) }
//!     | Factor { $1 }
//!     ;
//!
//! Factor -> Result<u64, ()>:
//!       '(' Expr ')' { $2 }
//!     | 'INT'
//!       {
//!           let v = $1.map_err(|_| ())?;
//!           parse_int($lexer.span_str(v.span()))
//!       }
//!     ;
//! %%
//! // Any functions here are in scope for all the grammar actions above.
//!
//! fn parse_int(s: &str) -> Result<u64, ()> {
//!     match s.parse::<u64>() {
//!         Ok(val) => Ok(val),
//!         Err(_) => {
//!             eprintln!("{} cannot be represented as a u64", s);
//!             Err(())
//!         }
//!     }
//! }
//! ```
//!
//! Because we specified that our Yacc file is in `Grmtools` format, each rule has a
//! separate Rust type to which all its functions conform (in this case, all the
//! rules have the same type, but that's not a requirement).
//!
//! A simple `src/main.rs` is as follows:
//!
//! ```text
//! use std::io::{self, BufRead, Write};
//!
//! use lrlex::lrlex_mod;
//! use lrpar::lrpar_mod;
//!
//! // Using `lrlex_mod!` brings the lexer for `calc.l` into scope. By default the module name
//! // will be `calc_l` (i.e. the file name, minus any extensions, with a suffix of `_l`).
//! lrlex_mod!("calc.l");
//! // Using `lrpar_mod!` brings the parser for `calc.y` into scope. By default the module name
//! // will be `calc_y` (i.e. the file name, minus any extensions, with a suffix of `_y`).
//! lrpar_mod!("calc.y");
//!
//! fn main() {
//!     // Get the `LexerDef` for the `calc` language.
//!     let lexerdef = calc_l::lexerdef();
//!     let stdin = io::stdin();
//!     loop {
//!         print!(">>> ");
//!         io::stdout().flush().ok();
//!         match stdin.lock().lines().next() {
//!             Some(Ok(ref l)) => {
//!                 if l.trim().is_empty() {
//!                     continue;
//!                 }
//!                 // Now we create a lexer with the `lexer` method with which we can lex an input.
//!                 let lexer = lexerdef.lexer(l);
//!                 // Pass the lexer to the parser and lex and parse the input.
//!                 let (res, errs) = calc_y::parse(&lexer);
//!                 for e in errs {
//!                     println!("{}", e.pp(&lexer, &calc_y::token_epp));
//!                 }
//!                 match res {
//!                     Some(Ok(r)) => println!("Result: {}", r),
//!                     _ => eprintln!("Unable to evaluate expression.")
//!                 }
//!             }
//!             _ => break
//!         }
//!     }
//! }
//! ```
//!
//! We can now `cargo run` our project and evaluate simple expressions:
//!
//! ```text
//! >>> 2 + 3
//! Result: 5
//! >>> 2 + 3 * 4
//! Result: 14
//! >>> (2 + 3) * 4
//! Result: 20
//! ```
//!
//! `lrpar` also comes with advanced [error
//! recovery](https://softdevteam.github.io/grmtools/master/book/errorrecovery.html) built-in:
//!
//! ```text
//! >>> 2 + + 3
//! Parsing error at line 1 column 5. Repair sequences found:
//!    1: Delete +
//!    2: Insert INT
//! Result: 5
//! >>> 2 + 3 3
//! Parsing error at line 1 column 7. Repair sequences found:
//!    1: Insert *
//!    2: Insert +
//!    3: Delete 3
//! Result: 11
//! >>> 2 + 3 4 5
//! Parsing error at line 1 column 7. Repair sequences found:
//!    1: Insert *, Delete 4
//!    2: Insert +, Delete 4
//!    3: Delete 4, Delete 5
//!    4: Insert +, Shift 4, Delete 5
//!    5: Insert +, Shift 4, Insert +
//!    6: Insert *, Shift 4, Delete 5
//!    7: Insert *, Shift 4, Insert *
//!    8: Insert *, Shift 4, Insert +
//!    9: Insert +, Shift 4, Insert *
//! Result: 17
//! ```

mod cpctplus;
#[doc(hidden)]
pub mod ctbuilder;
mod dijkstra;
#[doc(hidden)]
pub mod lex_api;
#[doc(hidden)]
pub mod parser;
#[cfg(test)]
mod test_utils;

pub use crate::{
    ctbuilder::{CTParser, CTParserBuilder, Visibility},
    lex_api::{LexError, Lexeme, Lexer, NonStreamingLexer, LRNonStreamingLexer},
    parser::{LexParseError, Node, ParseError, ParseRepair, RTParserBuilder, RecoveryKind},
};

/// A convenience macro for including statically compiled `.y` files. A file `src/a/b/c.ly
/// processed by [CTParserBuilder::parser_in_src_dir] can then be used in a crate with
/// `lrpar_mod!("a/b/c.l")`.
///
/// Note that you can use `lrpar_mod` with [CTParserBuilder::output_path] if, and only if, the
/// output file was placed in [std::env::var]`("OUT_DIR")` or one of its subdirectories.
#[macro_export]
macro_rules! lrpar_mod {
    ($path:expr) => {
        include!(concat!(env!("OUT_DIR"), "/", $path, ".rs"));
    };
}

/// A `Span` records what portion of the user's input something (e.g. a lexeme or production)
/// references (i.e. the `Span` doesn't hold a reference / copy of the actual input).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Span {
    start: usize,
    end: usize,
}

impl Span {
    /// Create a new span starting at byte `start` and ending at byte `end`.
    ///
    /// # Panics
    ///
    /// If `end` is less than `start`.
    pub fn new(start: usize, end: usize) -> Self {
        if end < start {
            panic!("Span starts ({}) after it ends ({})!", start, end);
        }
        Span { start, end }
    }

    /// Byte offset of the start of the span.
    pub fn start(&self) -> usize {
        self.start
    }

    /// Byte offset of the end of the span.
    pub fn end(&self) -> usize {
        self.end
    }

    /// Length in bytes of the span.
    pub fn len(&self) -> usize {
        self.end - self.start
    }

    /// Returns `true` if this `Span` covers 0 bytes, or `false` otherwise.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
