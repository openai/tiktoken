use crate::corebpe::CoreBPE;
use regex::Regex;
use rustc_hash::FxHashMap as HashMap;
use std::collections::HashSet;
use std::sync::Arc;

/// A struct that represents an encoding scheme based on byte-pair encoding (BPE).
pub struct Encoding {
    /// The name of the encoding.
    pub name: String,
    /// The regular expression pattern used to split text into pieces.
    pat_str: String,
    /// The map from mergeable byte sequences to their ranks.
    mergeable_ranks: HashMap<Vec<u8>, usize>,
    /// The map from special token strings to their values.
    special_tokens: HashMap<String, usize>,
    /// The maximum token value in the encoding.
    max_token_value: usize,
    /// The core BPE logic implemented in Rust.
    core_bpe: Arc<CoreBPE>,
}

pub enum SpecialTokenAction {
    /// The special token is forbidden. If it is included in the string, an error will be returned.
    Forbidden,
    /// The special token is tokenized as normal text.
    NormalText,
    /// The special token is treated as the special token it is. If the text is NOT a special token then an error will be returned.
    Special,
}
pub struct SpecialTokenHandling {
    pub default: SpecialTokenAction,
    pub overrides: Vec<(String, SpecialTokenAction)>,
}

impl Default for SpecialTokenHandling {
    fn default() -> Self {
        Self {
            default: SpecialTokenAction::Forbidden,
            overrides: vec![],
        }
    }
}

impl Encoding {
    /// Creates a new encoding from the given parameters.
    pub fn new(
        name: &str,
        pat_str: &str,
        mergeable_ranks: HashMap<Vec<u8>, usize>,
        special_tokens: HashMap<String, usize>,
        explicit_n_vocab: Option<usize>,
    ) -> Result<Self, String> {
        let max_token_value = match mergeable_ranks
            .values()
            .chain(special_tokens.values())
            .max()
            .copied()
        {
            Some(value) => value,
            None => return Err("No token values found".to_string()),
        };
        if let Some(explicit_n_vocab) = explicit_n_vocab {
            if mergeable_ranks.len() + special_tokens.len() != explicit_n_vocab {
                return Err(
                    "Mismatch between explicit vocab size and actual vocab size".to_string()
                );
            }
            if max_token_value != explicit_n_vocab - 1 {
                return Err("Mismatch between max token value and explicit vocab size".to_string());
            }
        }

        let core_bpe = CoreBPE::new(
            mergeable_ranks.clone(),
            special_tokens.clone(),
            pat_str.clone(),
        )
        .map_err(|e| format!("Error creating core BPE: {}", e))?;

        Ok(Self {
            name: name.to_string(),
            pat_str: pat_str.to_string(),
            mergeable_ranks,
            special_tokens,
            max_token_value,
            core_bpe: Arc::new(core_bpe),
        })
    }

    /// Encodes a string into tokens, ignoring special tokens.
    pub fn encode_ordinary(&self, text: &str) -> Vec<usize> {
        self.core_bpe.encode_ordinary(text)
    }

    /// Encodes a string into tokens.
    ///
    /// Special tokens are artificial tokens used to unlock capabilities from a model,
    /// such as fill-in-the-middle. So we want to be careful about accidentally encoding special
    /// tokens, since they can be used to trick a model into doing something we don't want it to do.
    pub fn encode(
        &self,
        text: &str,
        special_token_handling: &SpecialTokenHandling,
    ) -> Result<Vec<usize>, String> {
        // first check if all special tokens are valid
        for (special_token, _) in &special_token_handling.overrides {
            if !self.special_tokens.contains_key(special_token) {
                return Err(format!("Unknown special token {:?}", special_token));
            }
        }

        let special_tokens_to_recognize = match special_token_handling.default {
            SpecialTokenAction::Special => {
                &self
                    .special_tokens
                    .keys()
                    .map(|s| s.as_str())
                    .collect::<HashSet<_>>()
                    - &special_token_handling
                        .overrides
                        .iter()
                        .filter_map(|(token, action)| match action {
                            SpecialTokenAction::Special => None,
                            _ => Some(token.as_str()),
                        })
                        .collect::<HashSet<_>>()
            }
            _ => special_token_handling
                .overrides
                .iter()
                .filter_map(|(token, action)| match action {
                    SpecialTokenAction::Special => Some(token.as_str()),
                    _ => None,
                })
                .collect::<HashSet<_>>(),
        };
        let forbidden_special = match special_token_handling.default {
            SpecialTokenAction::Forbidden => {
                &self
                    .special_tokens
                    .keys()
                    .map(|s| s.as_str())
                    .collect::<HashSet<_>>()
                    - &special_token_handling
                        .overrides
                        .iter()
                        .filter_map(|(token, action)| match action {
                            SpecialTokenAction::Forbidden => None,
                            _ => Some(token.as_str()),
                        })
                        .collect::<HashSet<_>>()
            }
            _ => special_token_handling
                .overrides
                .iter()
                .filter_map(|(token, action)| match action {
                    SpecialTokenAction::Forbidden => Some(token.as_str()),
                    _ => None,
                })
                .collect::<HashSet<_>>(),
        };
        if !forbidden_special.is_empty() {
            let re = special_token_regex(&forbidden_special);
            if let Some(matched) = re.find(text) {
                return Err(format!(
                    "Encountered text corresponding to disallowed special token {:?}.",
                    matched.as_str()
                ));
            }
        }

        Ok(self.core_bpe.encode(text, &special_tokens_to_recognize))
    }

    /// Encodes a list of strings into tokens, in parallel, ignoring special tokens.
    pub fn encode_ordinary_batch(&self, text: Vec<String>) -> Vec<Vec<usize>> {
        // encoder = functools.partial(self.encode_ordinary)
        // with ThreadPoolExecutor(num_threads) as e:
        //     return list(e.map(encoder, text))
        // TODO: use rayon
        unimplemented!("todo")
    }

    /// Encodes a list of strings into tokens, in parallel.
    pub fn encode_batch(
        &self,
        text: Vec<String>,
        special_token_handling: &SpecialTokenHandling,
    ) -> Vec<Vec<usize>> {
        // with ThreadPoolExecutor(num_threads) as e:
        //     return list(e.map(encoder, text))

        // TODO: use rayon
        unimplemented!("todo")
    }

    /// Encodes a string into stable tokens and possible completion sequences.
    ///
    /// Note that the stable tokens will only represent a substring of `text`.
    ///
    /// This API should itself be considered unstable.
    pub fn encode_with_unstable(
        &self,
        text: &str,
        special_token_handling: &SpecialTokenHandling,
    ) -> Result<(Vec<usize>, HashSet<Vec<usize>>), String> {
        // first check if all special tokens are valid
        for (special_token, _) in &special_token_handling.overrides {
            if !self.special_tokens.contains_key(special_token) {
                return Err(format!("Unknown special token {:?}", special_token));
            }
        }

        let special_tokens_to_recognize = match special_token_handling.default {
            SpecialTokenAction::Special => {
                &self
                    .special_tokens
                    .keys()
                    .map(|s| s.as_str())
                    .collect::<HashSet<_>>()
                    - &special_token_handling
                        .overrides
                        .iter()
                        .filter_map(|(token, action)| match action {
                            SpecialTokenAction::Special => None,
                            _ => Some(token.as_str()),
                        })
                        .collect::<HashSet<_>>()
            }
            _ => special_token_handling
                .overrides
                .iter()
                .filter_map(|(token, action)| match action {
                    SpecialTokenAction::Special => Some(token.as_str()),
                    _ => None,
                })
                .collect::<HashSet<_>>(),
        };
        let forbidden_special = match special_token_handling.default {
            SpecialTokenAction::Forbidden => {
                &self
                    .special_tokens
                    .keys()
                    .map(|s| s.as_str())
                    .collect::<HashSet<_>>()
                    - &special_token_handling
                        .overrides
                        .iter()
                        .filter_map(|(token, action)| match action {
                            SpecialTokenAction::Forbidden => None,
                            _ => Some(token.as_str()),
                        })
                        .collect::<HashSet<_>>()
            }
            _ => special_token_handling
                .overrides
                .iter()
                .filter_map(|(token, action)| match action {
                    SpecialTokenAction::Forbidden => Some(token.as_str()),
                    _ => None,
                })
                .collect::<HashSet<_>>(),
        };
        if !forbidden_special.is_empty() {
            let re = special_token_regex(&forbidden_special);
            if let Some(matched) = re.find(text) {
                return Err(format!(
                    "Encountered text corresponding to disallowed special token {:?}.",
                    matched.as_str()
                ));
            }
        }

        Ok(self
            .core_bpe
            .encode_with_unstable(text, &special_tokens_to_recognize))
    }

    /// Encodes text corresponding to a single token to its token value.
    ///
    /// NOTE: this will encode all special tokens.
    ///
    /// Returns an error if the token is not in the vocabulary.
    pub fn encode_single_token(&self, text: &str) -> Result<usize, Vec<u8>> {
        self.encode_single_token_bytes(text.as_bytes())
    }
    pub fn encode_single_token_bytes(&self, bytes: &[u8]) -> Result<usize, Vec<u8>> {
        self.core_bpe.encode_single_token(bytes)
    }

    /// Decodes a list of tokens into bytes.
    pub fn decode_bytes(&self, tokens: &[usize]) -> Vec<u8> {
        self.core_bpe.decode_bytes(tokens)
    }

    /// Decodes a list of tokens into a string.
    ///
    /// WARNING: the default behaviour of this function is lossy, since decoded bytes are not
    /// guaranteed to be valid UTF-8.
    pub fn decode(&self, tokens: &[usize]) -> String {
        let bytes = self.core_bpe.decode_bytes(tokens);
        String::from_utf8_lossy(&bytes).to_string()
    }

    /// Decodes a token into bytes.
    ///
    /// NOTE: this will decode all special tokens.
    ///
    /// Returns an error if the token is not in the vocabulary.
    pub fn decode_single_token_bytes(&self, token: usize) -> Result<Vec<u8>, usize> {
        self.core_bpe.decode_single_token_bytes(token)
    }

    /// Decodes a list of tokens into a list of bytes.
    ///
    /// Useful for visualising tokenisation.
    ///
    /// Returns an error if any of the tokens is not in the vocabulary.
    pub fn decode_tokens_bytes(&self, tokens: Vec<usize>) -> Result<Vec<Vec<u8>>, usize> {
        tokens
            .into_iter()
            .map(|token| self.decode_single_token_bytes(token))
            .collect()
    }

    /// Returns the list of all token byte values.
    pub fn token_byte_values(&self) -> Vec<Vec<u8>> {
        self.core_bpe.token_byte_values()
    }

    /// Returns the end-of-text token.
    pub fn eot_token(&self) -> usize {
        self.special_tokens["<|endoftext|>"]
    }

    /// Encodes text corresponding to bytes without a regex split.
    ///
    /// NOTE: this will not encode any special tokens.
    fn _encode_single_piece(&self, text: &str) -> Vec<usize> {
        let text_or_bytes = text.as_bytes();
        self.core_bpe.encode_single_piece(text_or_bytes)
    }

    /// Encodes a string into tokens, but do regex splitting in Rust.
    fn _encode_only_native_bpe(&self, text: &str) -> Vec<usize> {
        let re = Regex::new(&self.pat_str).unwrap();
        let mut ret = Vec::new();
        for piece in re.find_iter(text) {
            ret.extend(self.core_bpe.encode_single_piece(piece.as_str().as_bytes()));
        }
        ret
    }

    /// Encodes bytes into tokens.
    fn _encode_bytes(&self, text: &[u8]) -> Vec<usize> {
        self.core_bpe._encode_bytes(text)
    }
}

/// Returns a regular expression that matches any of the given special tokens.
fn special_token_regex(tokens: &HashSet<&str>) -> Regex {
    let inner = tokens
        .iter()
        .map(|token| regex::escape(token))
        .collect::<Vec<_>>()
        .join("|");
    Regex::new(&format!("({})", inner)).unwrap()
}

// only use for testing!!!!
impl Default for Encoding {
    fn default() -> Self {
        crate::openai_public::EncodingFactory::cl100k_base().unwrap()
    }
}
