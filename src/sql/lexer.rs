//! Hand-written lexer for the ClickHouse-flavoured dialect.
//!
//! Four decisions shape everything below.
//!
//! **There is no keyword table.** `SELECT` and a column named `select` produce
//! the same [`Token::Word`]; only the parser decides which one it is, matching
//! with [`Token::is_keyword`] (ASCII case-insensitive, so `SeLeCt` works). A
//! lexer-level `Keyword` enum would have to be exhaustive, and every word it
//! reserved would stop being a legal column name -- an unaffordable trade for a
//! dialect where `date`, `count`, `min` and `key` are all plausible column
//! names. The only keyword knowledge here is [`is_reserved`]: the small set of
//! words that may not appear as a *bare alias*, which is the parser's one
//! genuine ambiguity (`SELECT a FROM t` must not alias `a` as `FROM`).
//!
//! **Quoting is remembered, not erased.** `` `x` `` and `"x"` set
//! `quoted = true`, which is exactly what lets `"select"` name a column: a
//! quoted word never compares equal to a keyword.
//!
//! **Literals are narrowed here.** An integer becomes `Value::Int` when it fits
//! in `i64` and `Value::UInt` otherwise. Doing it in the lexer means the parser
//! never re-reads source text, and the AST holds a real [`Value`] the planner
//! can constant-fold without another parse step. Note that a leading `-` is
//! *not* part of a numeric token -- the parser folds `-<literal>` back into the
//! literal, because `a-1` must lex as three tokens.
//!
//! **A decimal point makes an exact literal; an exponent does not.** `0.1` is
//! `Value::Decimal(1, 1)` and `1.5e3` is `Value::Float(1500.0)`. This is the
//! single decision behind `SELECT 0.1 + 0.2` answering `0.3` instead of
//! `0.30000000000000004`, and it is Postgres's rule: an unsuffixed decimal
//! constant is `numeric`, and only combining it with a float makes it one.
//!
//! The exponent form has to stay a float, and not only for compatibility. It is
//! what `toString` of a double produces (`1e-7`, `2.5e22`), so it is the
//! spelling that means "this came out of a binary float" -- and it is also the
//! only spelling that can name a magnitude no `Decimal64` has. Keeping it
//! inexact leaves every user of this dialect a way to *ask* for a float, which
//! a rule with no exceptions would have taken away.
//!
//! **Byte offsets, not line/column.** `Error::parse` carries a byte offset for
//! caret rendering; converting to line/column is the caller's job and only
//! costs a scan when an error actually fires.

use crate::common::{Error, Result};
use crate::types::Value;
use std::fmt;

/// One lexical token. Operators are spelled out rather than carried as text so
/// the parser matches on variants instead of comparing strings on every step.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Token {
    /// An identifier or a keyword -- the lexer does not distinguish them.
    /// `quoted` records backtick/double-quote form, which forces the
    /// identifier reading.
    Word { value: String, quoted: bool },
    /// Numeric literal, already narrowed to `Int`/`UInt`/`Float`.
    Number(Value),
    /// Single-quoted string literal with escapes already resolved.
    Str(String),

    Eq,
    NotEq,
    Lt,
    LtEq,
    Gt,
    GtEq,

    Plus,
    Minus,
    Star,
    Slash,
    Percent,
    /// `||` -- string concatenation, never logical or.
    Concat,

    LParen,
    RParen,
    Comma,
    Dot,
    Semicolon,
    /// `::` -- ClickHouse's cast operator.
    DoubleColon,
}

impl Token {
    /// Identifier text, quoted or not.
    pub fn word(&self) -> Option<&str> {
        match self {
            Token::Word { value, .. } => Some(value),
            _ => None,
        }
    }

    /// Text of an *unquoted* word -- the only thing that can be a keyword.
    pub fn bare_word(&self) -> Option<&str> {
        match self {
            Token::Word { value, quoted: false } => Some(value),
            _ => None,
        }
    }

    /// Case-insensitive keyword test. A quoted word never matches, which is
    /// what makes `"from"` usable as a column name.
    pub fn is_keyword(&self, kw: &str) -> bool {
        matches!(self, Token::Word { value, quoted: false } if value.eq_ignore_ascii_case(kw))
    }
}

impl fmt::Display for Token {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Token::Word { value, quoted } => {
                if *quoted {
                    write!(f, "`{value}`")
                } else {
                    write!(f, "{value}")
                }
            }
            Token::Number(v) => write!(f, "{}", v.render_plain()),
            Token::Str(s) => write!(f, "'{}'", s.replace('\'', "''")),
            Token::Eq => write!(f, "="),
            Token::NotEq => write!(f, "!="),
            Token::Lt => write!(f, "<"),
            Token::LtEq => write!(f, "<="),
            Token::Gt => write!(f, ">"),
            Token::GtEq => write!(f, ">="),
            Token::Plus => write!(f, "+"),
            Token::Minus => write!(f, "-"),
            Token::Star => write!(f, "*"),
            Token::Slash => write!(f, "/"),
            Token::Percent => write!(f, "%"),
            Token::Concat => write!(f, "||"),
            Token::LParen => write!(f, "("),
            Token::RParen => write!(f, ")"),
            Token::Comma => write!(f, ","),
            Token::Dot => write!(f, "."),
            Token::Semicolon => write!(f, ";"),
            Token::DoubleColon => write!(f, "::"),
        }
    }
}

/// A token plus the byte offset where it starts.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Spanned {
    pub tok: Token,
    pub pos: usize,
}

impl Spanned {
    pub fn new(tok: Token, pos: usize) -> Spanned {
        Spanned { tok, pos }
    }
}

/// Words that cannot be a *bare* alias, because seeing one after a complete
/// expression or table always means the next clause has begun. Deliberately
/// short: everything not listed here stays available as an identifier, and
/// even these can be used with an explicit `AS` or with quoting.
const RESERVED: &[&str] = &[
    "ALL", "AND", "ANTI", "ANY", "ARRAY", "AS", "ASC", "ASOF", "BETWEEN", "BY", "CASE", "CROSS",
    "DESC", "DISTINCT", "DIV", "ELSE", "END", "EXCEPT", "FINAL", "FORMAT", "FROM", "FULL",
    "GLOBAL", "GROUP", "HAVING", "ILIKE", "IN", "INNER", "INTERSECT", "IS", "JOIN", "LEFT",
    "LIKE", "LIMIT", "NOT", "NULLS", "OFFSET", "ON", "OR", "ORDER", "OUTER", "PREWHERE", "RIGHT",
    "SELECT", "SEMI", "SETTINGS", "THEN", "UNION", "USING", "VALUES", "WHEN", "WHERE", "WITH",
];

/// True if `word` may not be used as a bare (no `AS`) alias.
pub fn is_reserved(word: &str) -> bool {
    RESERVED.iter().any(|k| k.eq_ignore_ascii_case(word))
}

#[inline]
fn is_ident_start(c: u8) -> bool {
    // Bytes >= 0x80 are UTF-8 lead/continuation bytes: accepting them wholesale
    // makes non-ASCII identifiers work without decoding chars in the hot loop.
    c.is_ascii_alphabetic() || c == b'_' || c >= 0x80
}

#[inline]
fn is_ident_cont(c: u8) -> bool {
    is_ident_start(c) || c.is_ascii_digit() || c == b'$'
}

/// Split `sql` into tokens. Comments and whitespace are dropped.
pub fn tokenize(sql: &str) -> Result<Vec<Spanned>> {
    let b = sql.as_bytes();
    // ~4 bytes per token is a good guess for SQL and saves a few reallocs.
    let mut out: Vec<Spanned> = Vec::with_capacity(b.len() / 4 + 8);
    let mut i = 0usize;

    while i < b.len() {
        let c = b[i];

        if c.is_ascii_whitespace() {
            i += 1;
            continue;
        }

        // -- line comment
        if c == b'-' && i + 1 < b.len() && b[i + 1] == b'-' {
            i += 2;
            while i < b.len() && b[i] != b'\n' {
                i += 1;
            }
            continue;
        }

        // /* block comment */ -- not nested, matching ClickHouse.
        if c == b'/' && i + 1 < b.len() && b[i + 1] == b'*' {
            let start = i;
            let mut j = i + 2;
            let mut end = None;
            while j + 1 < b.len() {
                if b[j] == b'*' && b[j + 1] == b'/' {
                    end = Some(j + 2);
                    break;
                }
                j += 1;
            }
            match end {
                Some(e) => {
                    i = e;
                    continue;
                }
                None => return Err(Error::parse("unterminated block comment", start)),
            }
        }

        let start = i;

        if is_ident_start(c) {
            i += 1;
            while i < b.len() && is_ident_cont(b[i]) {
                i += 1;
            }
            out.push(Spanned::new(
                Token::Word { value: sql[start..i].to_string(), quoted: false },
                start,
            ));
            continue;
        }

        if c == b'`' || c == b'"' {
            let (value, next) = read_quoted(sql, start, c, "quoted identifier")?;
            i = next;
            out.push(Spanned::new(Token::Word { value, quoted: true }, start));
            continue;
        }

        if c == b'\'' {
            let (value, next) = read_quoted(sql, start, c, "string literal")?;
            i = next;
            out.push(Spanned::new(Token::Str(value), start));
            continue;
        }

        if c.is_ascii_digit() {
            let (tok, next) = read_number(sql, start)?;
            i = next;
            out.push(Spanned::new(tok, start));
            continue;
        }

        // Operators. Two-byte forms are tried first so `<=` never lexes as `<`.
        let (tok, len) = match c {
            b'=' if i + 1 < b.len() && b[i + 1] == b'=' => (Token::Eq, 2),
            b'=' => (Token::Eq, 1),
            b'!' if i + 1 < b.len() && b[i + 1] == b'=' => (Token::NotEq, 2),
            b'!' => return Err(Error::parse("expected `=` after `!`", i)),
            b'<' if i + 1 < b.len() && b[i + 1] == b'=' => (Token::LtEq, 2),
            b'<' if i + 1 < b.len() && b[i + 1] == b'>' => (Token::NotEq, 2),
            b'<' => (Token::Lt, 1),
            b'>' if i + 1 < b.len() && b[i + 1] == b'=' => (Token::GtEq, 2),
            b'>' => (Token::Gt, 1),
            b'|' if i + 1 < b.len() && b[i + 1] == b'|' => (Token::Concat, 2),
            b'|' => {
                return Err(Error::parse(
                    "`|` is not an operator; use `||` to concatenate or `OR` for logic",
                    i,
                ))
            }
            b':' if i + 1 < b.len() && b[i + 1] == b':' => (Token::DoubleColon, 2),
            b':' => return Err(Error::parse("expected `::`, found a single `:`", i)),
            b'+' => (Token::Plus, 1),
            b'-' => (Token::Minus, 1),
            b'*' => (Token::Star, 1),
            b'/' => (Token::Slash, 1),
            b'%' => (Token::Percent, 1),
            b'(' => (Token::LParen, 1),
            b')' => (Token::RParen, 1),
            b',' => (Token::Comma, 1),
            b'.' => (Token::Dot, 1),
            b';' => (Token::Semicolon, 1),
            _ => {
                let ch = sql[i..].chars().next().unwrap_or('?');
                return Err(Error::parse(format!("unexpected character `{ch}`"), i));
            }
        };
        i += len;
        out.push(Spanned::new(tok, start));
    }

    Ok(out)
}

/// Read a quoted run whose opening quote sits at `start`. Returns the decoded
/// contents and the index just past the closing quote.
///
/// Both doubling (`''`) and backslash escapes are accepted, because both are in
/// the wild and neither is ambiguous.
fn read_quoted(sql: &str, start: usize, quote: u8, what: &str) -> Result<(String, usize)> {
    let b = sql.as_bytes();
    let mut buf: Vec<u8> = Vec::new();
    let mut i = start + 1;
    loop {
        if i >= b.len() {
            return Err(Error::parse(format!("unterminated {what}"), start));
        }
        let c = b[i];
        if c == quote {
            if i + 1 < b.len() && b[i + 1] == quote {
                buf.push(quote);
                i += 2;
                continue;
            }
            i += 1;
            break;
        }
        if c == b'\\' {
            if i + 1 >= b.len() {
                return Err(Error::parse(format!("trailing backslash in {what}"), i));
            }
            let e = b[i + 1];
            match e {
                b'n' => buf.push(b'\n'),
                b't' => buf.push(b'\t'),
                b'r' => buf.push(b'\r'),
                b'0' => buf.push(0),
                b'a' => buf.push(0x07),
                b'b' => buf.push(0x08),
                b'f' => buf.push(0x0c),
                b'v' => buf.push(0x0b),
                b'\\' | b'\'' | b'"' | b'`' | b'/' => buf.push(e),
                // Unknown escapes keep their backslash, and the escaped byte is
                // copied by the normal path on the next turn: `LIKE 'a\%'` must
                // reach the matcher with `\%` intact, and consuming the byte
                // here would also risk splitting a UTF-8 sequence.
                _ => {
                    buf.push(b'\\');
                    i += 1;
                    continue;
                }
            }
            i += 2;
            continue;
        }
        buf.push(c);
        i += 1;
    }
    let s = String::from_utf8(buf)
        .map_err(|_| Error::parse(format!("invalid utf-8 in {what}"), start))?;
    Ok((s, i))
}

/// Read a numeric literal starting at `start` (guaranteed to be a digit).
///
/// The point/exponent split is the whole of the decimal-literal rule; see the
/// module header for why the exponent form has to stay a float.
fn read_number(sql: &str, start: usize) -> Result<(Token, usize)> {
    let b = sql.as_bytes();
    let mut i = start;
    let (mut point, mut exp) = (false, false);
    while i < b.len() && b[i].is_ascii_digit() {
        i += 1;
    }
    // A `.` only continues the number when a digit follows, so `t.1` and a
    // trailing `1.` still lex as separate tokens instead of eating the dot.
    if i + 1 < b.len() && b[i] == b'.' && b[i + 1].is_ascii_digit() {
        point = true;
        i += 1;
        while i < b.len() && b[i].is_ascii_digit() {
            i += 1;
        }
    }
    if i < b.len() && (b[i] == b'e' || b[i] == b'E') {
        let mut j = i + 1;
        if j < b.len() && (b[j] == b'+' || b[j] == b'-') {
            j += 1;
        }
        if j < b.len() && b[j].is_ascii_digit() {
            exp = true;
            i = j;
            while i < b.len() && b[i].is_ascii_digit() {
                i += 1;
            }
        }
    }
    // `123abc` and `0x1f` are typos, not two tokens; rejecting them here gives
    // a far better message than the parser's "expected an operator" would.
    if i < b.len() && is_ident_start(b[i]) {
        let mut j = i;
        while j < b.len() && is_ident_cont(b[j]) {
            j += 1;
        }
        return Err(Error::parse(
            format!("invalid numeric literal `{}`", &sql[start..j]),
            start,
        ));
    }
    let text = &sql[start..i];
    let float = |v: &str| {
        v.parse::<f64>()
            .map_err(|_| Error::parse(format!("invalid float literal `{v}`"), start))
    };
    let v = match (point, exp) {
        // The exact case. `Value::decimal_literal` declines a literal wider
        // than an `i64` lane, which then falls back to the float it has always
        // been rather than becoming an error: a number nobody asked to be exact
        // must not stop parsing.
        (true, false) => match Value::decimal_literal(text) {
            Some(v) => v,
            None => Value::Float(float(text)?),
        },
        (_, true) => Value::Float(float(text)?),
        (false, false) => {
            if let Ok(n) = text.parse::<i64>() {
                Value::Int(n)
            } else if let Ok(n) = text.parse::<u64>() {
                Value::UInt(n)
            } else {
                return Err(Error::parse(
                    format!("integer literal `{text}` does not fit in 64 bits"),
                    start,
                ));
            }
        }
    };
    Ok((Token::Number(v), i))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn toks(sql: &str) -> Vec<Token> {
        tokenize(sql).unwrap().into_iter().map(|s| s.tok).collect()
    }

    fn err_pos(sql: &str) -> usize {
        match tokenize(sql).unwrap_err() {
            Error::Parse { pos, .. } => pos,
            other => panic!("expected a parse error, got {other}"),
        }
    }

    fn word(s: &str) -> Token {
        Token::Word { value: s.into(), quoted: false }
    }

    #[test]
    fn keywords_are_case_insensitive_and_untabled() {
        for s in ["SELECT", "select", "SeLeCt"] {
            let t = &toks(s)[0];
            assert!(t.is_keyword("select"), "{s}");
            assert!(t.is_keyword("SELECT"), "{s}");
        }
        // ...and a keyword is just a word, so it can still be an identifier.
        assert_eq!(toks("select")[0], word("select"));
        assert!(!toks("selects")[0].is_keyword("select"));
    }

    #[test]
    fn quoted_identifiers_are_never_keywords() {
        let t = toks("`select`");
        assert_eq!(t[0], Token::Word { value: "select".into(), quoted: true });
        assert!(!t[0].is_keyword("select"));
        assert_eq!(toks("\"my col\"")[0].word(), Some("my col"));
        assert_eq!(toks("`a``b`")[0].word(), Some("a`b"));
        assert_eq!(toks("`a\\`b`")[0].word(), Some("a`b"));
        assert!(toks("`x`")[0].bare_word().is_none());
    }

    #[test]
    fn dotted_names_split_on_the_dot() {
        assert_eq!(
            toks("db.t.c"),
            vec![word("db"), Token::Dot, word("t"), Token::Dot, word("c")]
        );
        assert_eq!(toks("t.*"), vec![word("t"), Token::Dot, Token::Star]);
    }

    #[test]
    fn string_literals_handle_both_escape_conventions() {
        assert_eq!(toks("'abc'"), vec![Token::Str("abc".into())]);
        assert_eq!(toks("'it''s'"), vec![Token::Str("it's".into())]);
        assert_eq!(toks(r"'it\'s'"), vec![Token::Str("it's".into())]);
        assert_eq!(toks(r"'a\nb\tc\\d'"), vec![Token::Str("a\nb\tc\\d".into())]);
        assert_eq!(toks("''"), vec![Token::Str(String::new())]);
        // unknown escapes survive intact so LIKE patterns still work
        assert_eq!(toks(r"'a\%'"), vec![Token::Str(r"a\%".into())]);
        // non-ascii passes through byte-for-byte
        assert_eq!(toks("'héllo'"), vec![Token::Str("héllo".into())]);
    }

    /// Inverted when decimal literals landed: `1.5` used to be
    /// `Value::Float(1.5)` here, which is what made `SELECT 0.1 + 0.2` answer
    /// `0.30000000000000004`. The exponent forms below are the other half of
    /// the rule and did *not* change.
    #[test]
    fn numeric_literals_narrow_to_the_right_value() {
        assert_eq!(toks("42"), vec![Token::Number(Value::Int(42))]);
        assert_eq!(
            toks("9223372036854775808"),
            vec![Token::Number(Value::UInt(9_223_372_036_854_775_808))]
        );
        assert_eq!(toks("1.5"), vec![Token::Number(Value::Decimal(15, 1))]);
        assert_eq!(toks("1e3"), vec![Token::Number(Value::Float(1000.0))]);
        assert_eq!(toks("1.5E-3"), vec![Token::Number(Value::Float(0.0015))]);
        // a leading minus is a separate token: `a-1` must stay three tokens
        assert_eq!(
            toks("a-1"),
            vec![word("a"), Token::Minus, Token::Number(Value::Int(1))]
        );
    }

    /// The variant matters here and `Value`'s `Eq` is blind to it -- every
    /// spelling of 1.5 compares equal -- so these assert on `variant()` and on
    /// the unit/scale pair rather than with `assert_eq!` on a `Value`.
    #[test]
    fn a_point_makes_an_exact_literal_and_an_exponent_does_not() {
        let num = |s: &str| match toks(s).remove(0) {
            Token::Number(v) => v,
            other => panic!("{s} lexed as {other:?}"),
        };
        // The headline case: these two are what `0.1 + 0.2` is made of.
        assert_eq!(num("0.1").decimal_parts(), Some((1, 1)));
        assert_eq!(num("0.2").decimal_parts(), Some((2, 1)));
        // The scale is the digits written, trailing zeros and all, exactly as
        // for a `Decimal64(S)` column.
        assert_eq!(num("1.50").decimal_parts(), Some((150, 2)));
        assert_eq!(num("1.5").decimal_parts(), Some((15, 1)));
        assert_eq!(num("00.5").decimal_parts(), Some((5, 1)));
        assert_eq!(num("12.34").decimal_parts(), Some((1234, 2)));
        // 18 significant digits is the lane, and both edges of it are here:
        // the widest exact literal, then the first one that is not.
        assert_eq!(
            num("0.999999999999999999").decimal_parts(),
            Some((999_999_999_999_999_999, 18))
        );
        assert_eq!(num("999999999999999999.9").variant(), "Float");
        assert_eq!(num("0.0000000000000000001").variant(), "Float");
        // ...and the fallback is the float it always was, not an error.
        assert_eq!(num("999999999999999999.9").as_f64(), Some(1e18));
        // An exponent keeps the literal a float whether or not it has a point.
        for s in ["1.5e3", "1.5E-3", "1e3", "0.1e0", "1.0e18"] {
            assert_eq!(num(s).variant(), "Float", "{s}");
        }
        // An integer has no point, so it is untouched.
        assert_eq!(num("42").variant(), "Int");
    }

    #[test]
    fn every_operator_lexes() {
        assert_eq!(
            toks("= == != <> < <= > >= + - * / % || ( ) , . ; ::"),
            vec![
                Token::Eq,
                Token::Eq,
                Token::NotEq,
                Token::NotEq,
                Token::Lt,
                Token::LtEq,
                Token::Gt,
                Token::GtEq,
                Token::Plus,
                Token::Minus,
                Token::Star,
                Token::Slash,
                Token::Percent,
                Token::Concat,
                Token::LParen,
                Token::RParen,
                Token::Comma,
                Token::Dot,
                Token::Semicolon,
                Token::DoubleColon,
            ]
        );
    }

    #[test]
    fn comments_are_dropped() {
        assert_eq!(toks("a -- trailing\nb"), vec![word("a"), word("b")]);
        assert_eq!(toks("a /* mid */ b"), vec![word("a"), word("b")]);
        assert_eq!(toks("/* only a comment */"), vec![]);
        // `--` inside a string is not a comment
        assert_eq!(toks("'a -- b'"), vec![Token::Str("a -- b".into())]);
        // a lone `-` is still subtraction
        assert_eq!(toks("a - 1").len(), 3);
    }

    #[test]
    fn spans_point_at_token_starts() {
        let t = tokenize("  select  x").unwrap();
        assert_eq!(t[0].pos, 2);
        assert_eq!(t[1].pos, 10);
        let t = tokenize("/* c */ 1 + 2").unwrap();
        assert_eq!(t[0].pos, 8);
        assert_eq!(t[2].pos, 12);
    }

    #[test]
    fn lexer_errors_name_the_offset() {
        assert_eq!(err_pos("a # b"), 2);
        assert_eq!(err_pos("a | b"), 2);
        assert_eq!(err_pos("a ! b"), 2);
        assert_eq!(err_pos("select 'unterminated"), 7);
        assert_eq!(err_pos("select `unterminated"), 7);
        assert_eq!(err_pos("select /* unterminated"), 7);
        assert_eq!(err_pos("select 1abc"), 7);
        assert_eq!(err_pos("select 99999999999999999999999999"), 7);
        assert!(matches!(tokenize("a ? b"), Err(Error::Parse { .. })));
    }

    #[test]
    fn error_messages_say_what_was_wrong() {
        let e = tokenize("select 0x1f").unwrap_err().to_string();
        assert!(e.contains("invalid numeric literal `0x1f`"), "{e}");
        let e = tokenize("a | b").unwrap_err().to_string();
        assert!(e.contains("||"), "{e}");
    }

    #[test]
    fn reserved_words_are_case_insensitive_and_minimal() {
        assert!(is_reserved("from"));
        assert!(is_reserved("FROM"));
        assert!(is_reserved("Group"));
        // things that must stay usable as identifiers
        for w in ["date", "count", "key", "value", "id", "min", "total", "first", "last"] {
            assert!(!is_reserved(w), "{w} should not be reserved");
        }
    }

    #[test]
    fn display_renders_tokens_for_error_messages() {
        assert_eq!(Token::Comma.to_string(), ",");
        assert_eq!(Token::Concat.to_string(), "||");
        assert_eq!(word("x").to_string(), "x");
        assert_eq!(Token::Word { value: "x".into(), quoted: true }.to_string(), "`x`");
        assert_eq!(Token::Str("it's".into()).to_string(), "'it''s'");
        assert_eq!(Token::Number(Value::Int(-3)).to_string(), "-3");
    }

    #[test]
    fn empty_and_whitespace_only_input_is_empty() {
        assert!(tokenize("").unwrap().is_empty());
        assert!(tokenize("   \n\t ").unwrap().is_empty());
        assert!(tokenize("-- nothing").unwrap().is_empty());
    }
}
