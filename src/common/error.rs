//! One error type for the whole engine. Kept deliberately flat: the SQL
//! surface needs to render these as messages, not match on them structurally.

use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// Lexer/parser rejected the input. Carries a byte offset for carets.
    Parse { msg: String, pos: usize },
    /// Name resolution / type checking failed.
    Bind(String),
    /// The statement is well-formed but we do not implement it (yet).
    Unsupported(String),
    /// Runtime failure during execution (overflow, bad cast, divide by zero).
    Exec(String),
    /// Storage/catalog invariant violated or object missing.
    Storage(String),
    /// I/O or on-disk format problem.
    Io(String),
    /// On-disk data failed its checksum or version check.
    Corruption(String),
}

impl Error {
    pub fn parse(msg: impl Into<String>, pos: usize) -> Self {
        Error::Parse { msg: msg.into(), pos }
    }
    pub fn bind(msg: impl Into<String>) -> Self {
        Error::Bind(msg.into())
    }
    pub fn unsupported(msg: impl Into<String>) -> Self {
        Error::Unsupported(msg.into())
    }
    pub fn exec(msg: impl Into<String>) -> Self {
        Error::Exec(msg.into())
    }
    pub fn storage(msg: impl Into<String>) -> Self {
        Error::Storage(msg.into())
    }
    pub fn corruption(msg: impl Into<String>) -> Self {
        Error::Corruption(msg.into())
    }

    /// ClickHouse-style short code, handy for tests and for a wire protocol.
    pub fn code(&self) -> &'static str {
        match self {
            Error::Parse { .. } => "SYNTAX_ERROR",
            Error::Bind(_) => "UNKNOWN_IDENTIFIER",
            Error::Unsupported(_) => "NOT_IMPLEMENTED",
            Error::Exec(_) => "EXECUTION_ERROR",
            Error::Storage(_) => "STORAGE_ERROR",
            Error::Io(_) => "IO_ERROR",
            Error::Corruption(_) => "CHECKSUM_MISMATCH",
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Parse { msg, pos } => write!(f, "syntax error at offset {pos}: {msg}"),
            Error::Bind(m) => write!(f, "binding error: {m}"),
            Error::Unsupported(m) => write!(f, "not implemented: {m}"),
            Error::Exec(m) => write!(f, "execution error: {m}"),
            Error::Storage(m) => write!(f, "storage error: {m}"),
            Error::Io(m) => write!(f, "io error: {m}"),
            Error::Corruption(m) => write!(f, "corruption: {m}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e.to_string())
    }
}

impl From<std::str::Utf8Error> for Error {
    fn from(e: std::str::Utf8Error) -> Self {
        Error::Corruption(format!("invalid utf-8: {e}"))
    }
}

pub type Result<T> = std::result::Result<T, Error>;
