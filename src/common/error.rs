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
    /// On-disk data failed its checksum.
    Corruption(String),
    /// On-disk data is intact but was written by a format version this build
    /// does not read.
    ///
    /// Distinct from [`Error::Corruption`] because it is the opposite
    /// diagnosis and calls for the opposite response: nothing is damaged, no
    /// file should be deleted, and quarantining the table would file a version
    /// skew as rot. It also has to be a variant rather than a message, because
    /// [`super::store::load_catalog`] branches on it -- version skew refuses
    /// the open, damage quarantines one table.
    Version(String),
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
    pub fn version(msg: impl Into<String>) -> Self {
        Error::Version(msg.into())
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
            Error::Version(_) => "FORMAT_VERSION",
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
            Error::Version(m) => write!(f, "unsupported on-disk format: {m}"),
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
