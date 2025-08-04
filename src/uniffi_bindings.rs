use std::collections::{HashMap as StdHashMap, HashSet};
use std::sync::Arc;
use rustc_hash::FxHashMap as HashMap;
use base64::Engine;

use crate::{CoreBPE as CoreBPEInternal, Rank};

#[derive(Debug, thiserror::Error)]
pub enum TiktokenError {
    #[error("Value error: {0}")]
    ValueError(String),
    #[error("Key error: {0}")]
    KeyError(String),
    #[error("Decode error: {0}")]
    DecodeError(String),
}

impl From<crate::DecodeKeyError> for TiktokenError {
    fn from(err: crate::DecodeKeyError) -> Self {
        TiktokenError::KeyError(format!("Invalid token for decoding: {}", err.token))
    }
}

impl From<crate::DecodeError> for TiktokenError {
    fn from(err: crate::DecodeError) -> Self {
        TiktokenError::DecodeError(err.message)
    }
}

#[derive(Debug)]
pub struct EncodingResult {
    pub tokens: Vec<u32>,
    pub last_piece_token_len: u64,
}

#[derive(Debug)]
pub struct UnstableEncodingResult {
    pub tokens: Vec<u32>,
    pub completions: Vec<Vec<u32>>,
}

#[derive(Clone)]
pub struct CoreBpe {
    inner: Arc<CoreBPEInternal>,
}

impl CoreBpe {
    pub fn new(
        encoder: StdHashMap<String, u32>,
        special_tokens_encoder: StdHashMap<String, u32>,
        pattern: String,
    ) -> Self {
        // Convert String keys to Vec<u8> for the encoder
        // Handle base64-encoded byte sequences for non-UTF8 tokens
        let byte_encoder: HashMap<Vec<u8>, Rank> = encoder
            .into_iter()
            .map(|(k, v)| {
                if k.starts_with("base64:") {
                    // Decode base64 for non-UTF8 sequences
                    let b64_str = &k[7..];
                    match base64::engine::general_purpose::STANDARD.decode(b64_str) {
                        Ok(bytes) => (bytes, v),
                        Err(e) => {
                            eprintln!("Failed to decode base64 token {}: {}", k, e);
                            (k.into_bytes(), v)
                        }
                    }
                } else {
                    // Regular UTF-8 string
                    (k.into_bytes(), v)
                }
            })
            .collect();
        
        let special_tokens_encoder: HashMap<String, Rank> = special_tokens_encoder
            .into_iter()
            .collect();

        let inner = CoreBPEInternal::new_internal(byte_encoder, special_tokens_encoder, &pattern)
            .expect("Failed to create CoreBPE");

        Self {
            inner: Arc::new(inner),
        }
    }

    pub fn encode_ordinary(&self, text: String) -> Vec<u32> {
        self.inner.encode_ordinary(&text)
    }

    pub fn encode(&self, text: String, allowed_special: Vec<String>) -> Vec<u32> {
        let allowed_special: HashSet<&str> = allowed_special.iter().map(|s| s.as_str()).collect();
        self.inner.encode(&text, &allowed_special).0
    }

    pub fn encode_with_details(&self, text: String, allowed_special: Vec<String>) -> EncodingResult {
        let allowed_special: HashSet<&str> = allowed_special.iter().map(|s| s.as_str()).collect();
        let (tokens, last_piece_token_len) = self.inner.encode(&text, &allowed_special);
        EncodingResult {
            tokens,
            last_piece_token_len: last_piece_token_len as u64,
        }
    }

    pub fn encode_with_unstable(
        &self,
        text: String,
        allowed_special: Vec<String>,
    ) -> UnstableEncodingResult {
        let allowed_special: HashSet<&str> = allowed_special.iter().map(|s| s.as_str()).collect();
        let (tokens, completions) = self.inner._encode_unstable_native(&text, &allowed_special);
        UnstableEncodingResult {
            tokens,
            completions: completions.into_iter().collect(),
        }
    }

    pub fn encode_bytes(&self, input: Vec<u8>) -> Vec<u32> {
        match std::str::from_utf8(&input) {
            Ok(text) => self.inner.encode_ordinary(text),
            Err(e) => {
                let text = unsafe { std::str::from_utf8_unchecked(&input[..e.valid_up_to()]) };
                let (tokens, last_piece_token_len) = self.inner.encode(text, &HashSet::new());
                let (mut tokens, last_piece_token_len) = self
                    .inner
                    ._increase_last_piece_token_len(tokens, last_piece_token_len);

                let mut unstable_bytes;
                if !tokens.is_empty() && last_piece_token_len > 0 {
                    unstable_bytes = self
                        .inner
                        .decode_bytes(&tokens[tokens.len() - last_piece_token_len..])
                        .unwrap();
                    unstable_bytes.extend_from_slice(&input[e.valid_up_to()..]);
                    tokens.truncate(tokens.len() - last_piece_token_len);
                } else {
                    unstable_bytes = input[e.valid_up_to()..].to_vec();
                }

                if !unstable_bytes.is_empty() {
                    match self.inner.encoder.get(&unstable_bytes) {
                        Some(token) => tokens.push(*token),
                        None => {
                            tokens.extend(&crate::byte_pair_encode(&unstable_bytes, &self.inner.encoder))
                        }
                    }
                }
                tokens
            }
        }
    }

    pub fn encode_single_token(&self, piece: Vec<u8>) -> Result<u32, TiktokenError> {
        if let Some(token) = self.inner.encoder.get(&piece).copied() {
            return Ok(token);
        }
        if let Ok(piece_str) = std::str::from_utf8(&piece) {
            if let Some(token) = self.inner.special_tokens_encoder.get(piece_str).copied() {
                return Ok(token);
            }
        }
        Err(TiktokenError::KeyError(format!(
            "Token not found: {:?}",
            piece
        )))
    }

    pub fn encode_single_piece(&self, piece: Vec<u8>) -> Vec<u32> {
        if piece.is_empty() {
            return vec![];
        }
        if let Some(token) = self.inner.encoder.get(&piece) {
            return vec![*token];
        }
        crate::byte_pair_encode(&piece, &self.inner.encoder)
    }

    pub fn decode_bytes(&self, tokens: Vec<u32>) -> Result<Vec<u8>, TiktokenError> {
        self.inner.decode_bytes(&tokens).map_err(|e| e.into())
    }

    pub fn decode_single_token_bytes(&self, token: u32) -> Result<Vec<u8>, TiktokenError> {
        if let Some(bytes) = self.inner.decoder.get(&token) {
            return Ok(bytes.clone());
        }
        if let Some(bytes) = self.inner.special_tokens_decoder.get(&token) {
            return Ok(bytes.clone());
        }
        Err(TiktokenError::KeyError(format!("Token not found: {}", token)))
    }

    pub fn token_byte_values(&self) -> Vec<Vec<u8>> {
        self.inner.sorted_token_bytes.clone()
    }

    pub fn special_tokens(&self) -> Vec<String> {
        self.inner
            .special_tokens_encoder
            .keys()
            .cloned()
            .collect()
    }

    pub fn encode_with_special_tokens(&self, text: String) -> Vec<u32> {
        self.inner.encode_with_special_tokens(&text)
    }
    
    pub fn max_token_value(&self) -> u32 {
        // Find the maximum value among regular and special tokens
        let max_regular = self.inner.encoder.values().max().copied().unwrap_or(0);
        let max_special = self.inner.special_tokens_encoder.values().max().copied().unwrap_or(0);
        max_regular.max(max_special)
    }
    
    pub fn n_vocab(&self) -> u32 {
        // For backwards compatibility, n_vocab is max_token_value + 1
        self.max_token_value() + 1
    }
}

pub fn new_core_bpe(
    encoder: StdHashMap<String, u32>,
    special_tokens_encoder: StdHashMap<String, u32>,
    pattern: String,
) -> Result<Arc<CoreBpe>, TiktokenError> {
    // Convert String keys to Vec<u8> for the encoder
    let byte_encoder: HashMap<Vec<u8>, Rank> = encoder
        .into_iter()
        .map(|(k, v)| (k.into_bytes(), v))
        .collect();
    
    let special_tokens_encoder: HashMap<String, Rank> = special_tokens_encoder
        .into_iter()
        .collect();

    let inner = CoreBPEInternal::new_internal(byte_encoder, special_tokens_encoder, &pattern)
        .map_err(|e| TiktokenError::ValueError(e.to_string()))?;

    Ok(Arc::new(CoreBpe {
        inner: Arc::new(inner),
    }))
}

uniffi::include_scaffolding!("tiktoken");

