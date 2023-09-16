//! WARNING: This code is under active development. Functionality,
//! behavior, and the interface may change in future updates.

use std::collections::HashMap;
use once_cell::sync::Lazy;
use regex::Regex;


pub struct Encoding {
    /// The name of the encoding. It should be clear from the name of the encoding
    /// what behaviour to expect, in particular, encodings with different special tokens
    /// should have different names.
    pub name: &'static str,
    /// A regex pattern string that is used to split the input text.
    pub pat_str: Regex,
    /// A dictionary mapping mergeable token bytes to their ranks. The ranks
    /// must correspond to merge priority.
    pub mergeable_ranks: HashMap<&'static str, u32>,
    /// A dictionary mapping special token strings to their token values.
    pub special_tokens: HashMap<&'static str, u32>,
    /// The number of tokens in the vocabulary. If provided, it is checked
    /// that the number of mergeable tokens and special tokens is equal to this number.
    pub explicit_n_vocab: Option<u32>,
}

pub static GPT2: Lazy<Encoding> = Lazy::new(|| {
    let mergeable_ranks = Default::default();
    let special_tokens = [
        ("<|endoftext|>", 50256)
    ].iter().cloned().collect();

    Encoding{
        name: "gpt2",
        pat_str: Regex::new(r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+").unwrap(),
        mergeable_ranks,
        special_tokens,
        explicit_n_vocab: Some(50257),
    }
});

pub fn get_encoding() {

}

#[cfg(test)]
mod test {

    #[test]
    fn test_simple() {
        // enc = tiktoken.get_encoding("gpt2")
        // assert enc.encode("hello world") == [31373, 995]
        // assert enc.decode([31373, 995]) == "hello world"
        // assert enc.encode("hello <|endoftext|>", allowed_special="all") == [31373, 220, 50256]
        //
        // enc = tiktoken.get_encoding("cl100k_base")
        // assert enc.encode("hello world") == [15339, 1917]
        // assert enc.decode([15339, 1917]) == "hello world"
        // assert enc.encode("hello <|endoftext|>", allowed_special="all") == [15339, 220, 100257]
        //
        // for enc_name in tiktoken.list_encoding_names():
        //     enc = tiktoken.get_encoding(enc_name)
        // for token in range(10_000):
        //     assert enc.encode_single_token(enc.decode_single_token_bytes(token)) == token
    }
}