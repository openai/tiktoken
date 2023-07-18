use fancy_regex::Error;
use parse_display::Display;

use crate::error::TiktokenError::RegexError;

#[derive(Debug, Display)]
pub enum TiktokenError {
    #[display("RegexError: {0}")]
    RegexError(String),
    #[display("GenericError: {0}")]
    GenericError(String),
}

impl From<fancy_regex::Error> for TiktokenError {
    fn from(value: Error) -> Self {
        RegexError(value.to_string())
    }
}
