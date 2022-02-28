#![allow(clippy::len_without_is_empty)]

use std::{cmp, error::Error, fmt, hash::Hash, marker::{self, PhantomData}, convert::TryFrom};

use num_traits::{PrimInt, Unsigned};

use crate::Span;

/// A Lexing error.
#[derive(Copy, Clone, Debug)]
pub struct LexError {
    span: Span,
}

impl LexError {
    pub fn new(span: Span) -> Self {
        LexError { span }
    }

    pub fn span(&self) -> Span {
        self.span
    }
}

impl Error for LexError {}

impl fmt::Display for LexError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "Couldn't lex input starting at byte {}",
            self.span.start()
        )
    }
}

/// The base trait which all lexers which want to interact with `lrpar` must implement.
pub trait Lexer<LexemeT: Lexeme<StorageT>, StorageT: Hash + PrimInt + Unsigned> {
    /// Iterate over all the lexemes in this lexer. Note that:
    ///   * The lexer may or may not stop after the first [LexError] is encountered.
    ///   * There are no guarantees about what happens if this function is called more than once.
    ///     For example, a streaming lexer may only produce [Lexeme]s on the first call.
    fn iter<'a>(&'a self) -> Box<dyn Iterator<Item = Result<LexemeT, LexError>> + 'a>;
}

/// A `NonStreamingLexer` is one that takes input in one go, and is then able to hand out
/// substrings to that input and calculate line and column numbers from a [Span].
pub trait NonStreamingLexer<'input, LexemeT: Lexeme<StorageT>, StorageT: Hash + PrimInt + Unsigned>:
    Lexer<LexemeT, StorageT>
{
    /// Return the user input associated with a [Span].
    ///
    /// The [Span] must be well formed:
    ///   * The start/end byte indexes must be valid UTF-8 character indexes.
    ///   * The end byte index must not exceed the input's length.
    ///
    /// If these requirements are not respected this function may panic or return unexpected
    /// portions of the input.
    fn span_str(&self, span: Span) -> &'input str;

    /// Return the lines containing the input at `span` (including *all* the text on the lines
    /// that `span` starts and ends on).
    ///
    /// The [Span] must be well formed:
    ///   * The start/end byte indexes must be valid UTF-8 character indexes.
    ///   * The end byte index must not exceed the input's length.
    ///
    /// If these requirements are not respected this function may panic or return unexpected
    /// portions of the input.
    fn span_lines_str(&self, span: Span) -> &'input str;

    /// Return `((start line, start column), (end line, end column))` for `span`. Note that column
    /// *characters* (not bytes) are returned.
    ///
    /// The [Span] must be well formed:
    ///   * The start/end byte indexes must be valid UTF-8 character indexes.
    ///   * The end byte index must not exceed the input's length.
    ///
    /// If these requirements are not respected this function may panic or return unexpected
    /// portions of the input.
    fn line_col(&self, span: Span) -> ((usize, usize), (usize, usize));
}

/// A lexeme represents a segment of the user's input that conforms to a known type: this trait
/// captures the common behaviour of all lexeme structs.
///
/// Lexemes are assumed to have a definition which describes all possible correct lexemes (e.g. the
/// regular expression `[0-9]+` defines all integer lexemes). This trait also allows "faulty"
/// lexemes to be represented -- that is, lexemes that have resulted from error recovery of some
/// sort. Faulty lexemes can violate the lexeme's type definition in any possible way (e.g. they
/// might span more or less input than the definition would suggest is possible).
pub trait Lexeme<StorageT>: fmt::Debug + fmt::Display + cmp::Eq + Hash + marker::Copy {
    /// Create a new lexeme with ID `tok_id`, a starting position in the input `start`, and length
    /// `len`.
    ///
    /// Lexemes created using this function are expected to be "correct" in the sense that they
    /// fully respect the lexeme's definition semantics. To create faulty lexemes, use
    /// [new_faulty](Lexeme::new_faulty).
    fn new(tok_id: StorageT, start: usize, len: usize) -> Self
    where
        Self: Sized;

    /// Create a new faulty lexeme with ID `tok_id` and a starting position in the input `start`.
    fn new_faulty(tok_id: StorageT, start: usize, len: usize) -> Self
    where
        Self: Sized;

    /// The token ID.
    fn tok_id(&self) -> StorageT;

    /// Obtain this `Lexeme`'s [Span].
    fn span(&self) -> Span;

    /// Returns `true` if this lexeme is "faulty" i.e. is the result of error recovery in some way.
    /// If `true`, note that the lexeme's span may be greater or less than you may expect from the
    /// lexeme's definition.
    fn faulty(&self) -> bool;
}


pub struct LRNonStreamingLexer<'lexer, 'input: 'lexer, LexemeT, StorageT: fmt::Debug> {
    s: &'input str,
    lexemes: Vec<Result<LexemeT, LexError>>,
    /// A sorted list of the byte index of the start of the following line. i.e. for the input
    /// string `" a\nb\n  c d"` this will contain `[3, 5]`.
    newlines: Vec<usize>,
    phantom: PhantomData<(&'lexer (), StorageT)>,
}


impl<
        'lexer,
        'input: 'lexer,
        LexemeT: Lexeme<StorageT>,
        StorageT: Copy + Eq + fmt::Debug + Hash + PrimInt + TryFrom<usize> + Unsigned,
    > LRNonStreamingLexer<'lexer, 'input, LexemeT, StorageT>
{
    /// Create a new `LRNonStreamingLexer` that read in: the input `s`; and derived `lexemes` and
    /// `newlines`. The `newlines` `Vec<usize>` is a sorted list of the byte index of the start of
    /// the following line. i.e. for the input string `" a\nb\n  c d"` the `Vec` should contain
    /// `[3, 5]`.
    ///
    /// Note that if one or more lexemes or newlines was not created from `s`, subsequent calls to
    /// the `LRNonStreamingLexer` may cause `panic`s.
    pub fn new(
        s: &'input str,
        lexemes: Vec<Result<LexemeT, LexError>>,
        newlines: Vec<usize>,
    ) -> LRNonStreamingLexer<'lexer, 'input, LexemeT, StorageT> {
        LRNonStreamingLexer {
            s,
            lexemes,
            newlines,
            phantom: PhantomData,
        }
    }

    pub fn iter<'a>(&'a self) -> Box<dyn Iterator<Item = Result<LexemeT, LexError>> + 'a> {
        Box::new(self.lexemes.iter().cloned())
    }

    pub fn span_str(&self, span: Span) -> &'input str {
        if span.end() > self.s.len() {
            panic!(
                "Span {:?} exceeds known input length {}",
                span,
                self.s.len()
            );
        }
        &self.s[span.start()..span.end()]
    }

    pub fn span_lines_str(&self, span: Span) -> &'input str {
        debug_assert!(span.end() >= span.start());
        if span.end() > self.s.len() {
            panic!(
                "Span {:?} exceeds known input length {}",
                span,
                self.s.len()
            );
        }

        let (st, st_line) = match self.newlines.binary_search(&span.start()) {
            Ok(j) => (self.newlines[j], j + 1),
            Err(0) => (0, 0),
            Err(j) => (self.newlines[j - 1], j),
        };
        let en = match self.newlines[st_line..].binary_search(&span.end()) {
            Ok(j) => self.newlines[st_line + j + 1] - 1,
            Err(j) if st_line + j == self.newlines.len() => self.s.len(),
            Err(j) => self.newlines[st_line + j] - 1,
        };
        &self.s[st..en]
    }

    pub fn line_col(&self, span: Span) -> ((usize, usize), (usize, usize)) {
        debug_assert!(span.end() >= span.start());
        if span.end() > self.s.len() {
            panic!(
                "Span {:?} exceeds known input length {}",
                span,
                self.s.len()
            );
        }

        /// Returns `(line byte offset, line index)`.
        fn lc_byte<LexemeT, StorageT: fmt::Debug>(
            lexer: &LRNonStreamingLexer<LexemeT, StorageT>,
            i: usize,
        ) -> (usize, usize) {
            match lexer.newlines.binary_search(&i) {
                Ok(j) => (lexer.newlines[j], j + 2),
                Err(0) => (0, 1),
                Err(j) => (lexer.newlines[j - 1], j + 1),
            }
        }

        fn lc_char<LexemeT, StorageT: Copy + Eq + fmt::Debug + Hash + PrimInt + Unsigned>(
            lexer: &LRNonStreamingLexer<LexemeT, StorageT>,
            i: usize,
            s: &str,
        ) -> (usize, usize) {
            let (line_byte, line_idx) = lc_byte(lexer, i);
            (line_idx, s[line_byte..i].chars().count() + 1)
        }

        (
            lc_char(self, span.start(), self.s),
            lc_char(self, span.end(), self.s),
        )
    }
}
