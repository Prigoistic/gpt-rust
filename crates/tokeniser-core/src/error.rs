use std::{fmt, io};

#[derive(Debug)]
pub enum LoadError {
    Io(io::Error),
    Json(serde_json::Error),
    UnknownSymbolChar(char),
    IdOutOfRange(u32),
}

#[derive(Debug)]
pub struct UnknownToken(pub u32);

impl fmt::Display for LoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "failed to read vocab file: {e}"),
            Self::Json(e) => write!(f, "invalid encoder.json: {e}"),
            Self::UnknownSymbolChar(c) => write!(f, "vocab symbol contains unmapped char {c:?}"),
            Self::IdOutOfRange(id) => write!(f, "token id {id} exceeds vocab size"),
        }
    }
}

impl fmt::Display for UnknownToken {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "unknown token id {}", self.0)
    }
}

impl std::error::Error for LoadError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            Self::Json(e) => Some(e),
            _ => None,
        }
    }
}

impl std::error::Error for UnknownToken {}

impl From<io::Error> for LoadError {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<serde_json::Error> for LoadError {
    fn from(e: serde_json::Error) -> Self {
        Self::Json(e)
    }
}
