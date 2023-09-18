use crate::encoding::Encoding;
use rustc_hash::FxHashMap as HashMap;
use thiserror::Error;

use crate::load::load_tiktoken_bpe;

#[derive(Error, Debug, Clone)]
pub enum EncodingFactoryError {
    #[error("failed to load encoding")]
    FailedToLoadEncoding,
    #[error("unable to create encoding: {0}")]
    UnableToCreateEncoding(String),
}

const ENDOFTEXT: &str = "<|endoftext|>";
const FIM_PREFIX: &str = "<|fim_prefix|>";
const FIM_MIDDLE: &str = "<|fim_middle|>";
const FIM_SUFFIX: &str = "<|fim_suffix|>";
const ENDOFPROMPT: &str = "<|endofprompt|>";

const IM_START: &str = "<|im_start|>";
const IM_END: &str = "<|im_end|>";
const IM_SEP: &str = "<|im_sep|>";

#[derive(Clone, Debug, Copy)]
pub struct EncodingFactory {}
impl EncodingFactory {
    pub fn gpt2() -> Result<Encoding, EncodingFactoryError> {
        // todo!
        // vocab_bpe_file: sha256 = 1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5
        // encoder_json_file: sha256 = 196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783
        // Encoding::new()
        unimplemented!("gpt2")
    }

    pub fn r50k_base() -> Result<Encoding, EncodingFactoryError> {
        let mergeable_ranks = load_tiktoken_bpe(
            include_bytes!("../data/r50k_base.tiktoken"),
            "306cd27f03c1a714eca7108e03d66b7dc042abe8c258b44c199a7ed9838dd930",
        )
        .map_err(|_| EncodingFactoryError::FailedToLoadEncoding)?;
        let special_tokens: HashMap<String, usize> =
            [(ENDOFTEXT.to_string(), 50256)].iter().cloned().collect();
        Encoding::new(
            "r50k_base",
            r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+",
            mergeable_ranks,
            special_tokens,
            Some(50257),
        )
        .map_err(|e| EncodingFactoryError::UnableToCreateEncoding(e.to_string()))
    }

    pub fn p50k_base() -> Result<Encoding, EncodingFactoryError> {
        let mergeable_ranks = load_tiktoken_bpe(
            include_bytes!("../data/p50k_base.tiktoken"),
            "94b5ca7dff4d00767bc256fdd1b27e5b17361d7b8a5f968547f9f23eb70d2069",
        )
        .map_err(|_| EncodingFactoryError::FailedToLoadEncoding)?;
        let special_tokens: HashMap<String, usize> =
            [(ENDOFTEXT.to_string(), 50256)].iter().cloned().collect();
        Encoding::new(
            "p50k_base",
            r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+",
            mergeable_ranks,
            special_tokens,
            Some(50281),
        )
        .map_err(|e| EncodingFactoryError::UnableToCreateEncoding(e.to_string()))
    }

    pub fn cl100k_base() -> Result<Encoding, EncodingFactoryError> {
        let mergeable_ranks = load_tiktoken_bpe(
            include_bytes!("../data/cl100k_base.tiktoken"),
            "223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7",
        )
        .map_err(|_| EncodingFactoryError::FailedToLoadEncoding)?;
        let special_tokens: HashMap<String, usize> = [
            (ENDOFTEXT.to_string(), 100257),
            (FIM_PREFIX.to_string(), 100258),
            (FIM_MIDDLE.to_string(), 100259),
            (FIM_SUFFIX.to_string(), 100260),
            (ENDOFPROMPT.to_string(), 100276),
        ]
        .iter()
        .cloned()
        .collect();
        Encoding::new(
            "cl100k_base",
            r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
            mergeable_ranks,
            special_tokens,
            None,
        )
        .map_err(|e| EncodingFactoryError::UnableToCreateEncoding(e.to_string()))
    }

    pub fn cl100k_with_special_tokens(special_tokens: &[(String, usize)]) -> Result<Encoding, EncodingFactoryError> {
        let mergeable_ranks = load_tiktoken_bpe(
            include_bytes!("../data/cl100k_base.tiktoken"),
            "223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7",
        )
        .map_err(|_| EncodingFactoryError::FailedToLoadEncoding)?;
        let special_tokens: HashMap<String, usize> = special_tokens 
        .iter()
        .cloned()
        .collect();
        Encoding::new(
            "cl100k_base",
            r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
            mergeable_ranks,
            special_tokens,
            None,
        )
        .map_err(|e| EncodingFactoryError::UnableToCreateEncoding(e.to_string()))
    }
}
