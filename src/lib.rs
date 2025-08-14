use std::collections::HashSet;
use std::thread;

use fancy_regex::Regex;
#[cfg(feature = "python")]
use pyo3::types::{PyBytes, PyList, PyTuple};
#[cfg(feature = "python")]
use pyo3::{exceptions, prelude::*, types::PyDict};
use rustc_hash::FxHashMap as HashMap;

#[cfg(feature = "python")]
mod py;

#[cfg(feature = "uniffi")]
pub mod uniffi_bindings;

// UniFfiTag is required by the scaffolding at crate root
#[cfg(feature = "uniffi")]
pub struct UniFfiTag;

pub type Rank = u32;

fn _byte_pair_merge(ranks: &HashMap<Vec<u8>, Rank>, piece: &[u8]) -> Vec<(usize, Rank)> {
    // This is a vector of (start, rank).
    // The rank is of the pair starting at position start.
    let mut parts = Vec::with_capacity(piece.len() + 1);

    // Note that we hash bytes when indexing into `ranks`, not token pairs. As long as we train BPE
    // the way we currently do, this is equivalent. An easy way to break this would be to decouple
    // merge priority from token index or to prevent specific token merges.
    let mut min_rank: (Rank, usize) = (Rank::MAX, usize::MAX);
    for i in 0..piece.len() - 1 {
        let rank = *ranks.get(&piece[i..i + 2]).unwrap_or(&Rank::MAX);
        if rank < min_rank.0 {
            min_rank = (rank, i);
        }
        parts.push((i, rank));
    }
    parts.push((piece.len() - 1, Rank::MAX));
    parts.push((piece.len(), Rank::MAX));

    let get_rank = {
        #[inline(always)]
        |parts: &Vec<(usize, Rank)>, i: usize| {
            if (i + 3) < parts.len() {
                // Similar to `piece[i..i + 2]` above. The +3 is because we haven't yet deleted
                // parts[i + 1], see comment in the main loop.
                *ranks
                    .get(&piece[parts[i].0..parts[i + 3].0])
                    .unwrap_or(&Rank::MAX)
            } else {
                Rank::MAX
            }
        }
    };

    // If you have n parts and m merges, this does O(mn) work.
    // We could do something with a heap and do O(m log n) work.
    // It's important that we're iterating over parts and not over ranks.
    // The way we iterate here, we're iterating over parts (i.e. pieces of the text).
    // If we iterated over ranks, we'd be iterating over the vocabulary.
    // Given that vocabulary is >> parts in most cases, iterating over parts is faster.
    while min_rank.0 != Rank::MAX {
        let i = min_rank.1;
        // Update parts[i] and parts[i - 1] before removing parts[i + 1], since
        // `parts.remove(i + 1)` will invalidate them.
        parts[i] = (parts[i].0, get_rank(&parts, i));
        if i > 0 {
            parts[i - 1] = (parts[i - 1].0, get_rank(&parts, i - 1));
        }

        parts.remove(i + 1);

        min_rank = (Rank::MAX, usize::MAX);
        for (i, &(_, rank)) in parts[..parts.len() - 1].iter().enumerate() {
            if rank < min_rank.0 {
                min_rank = (rank, i);
            }
        }
    }
    parts
}

pub fn byte_pair_encode(piece: &[u8], ranks: &HashMap<Vec<u8>, Rank>) -> Vec<Rank> {
    if piece.len() == 1 {
        return vec![ranks[piece]];
    }
    _byte_pair_merge(ranks, piece)
        .windows(2)
        .map(|part| ranks[&piece[part[0].0..part[1].0]])
        .collect()
}

pub fn byte_pair_split<'a>(piece: &'a [u8], ranks: &HashMap<Vec<u8>, Rank>) -> Vec<&'a [u8]> {
    assert!(piece.len() > 1);
    _byte_pair_merge(ranks, piece)
        .windows(2)
        .map(|part| &piece[part[0].0..part[1].0])
        .collect()
}

// Various performance notes:
//
// Regex
// =====
// Most of the time is spent in regex. The easiest way to speed this up is by using less fancy
// regex features. For instance, using a regex parse-able by `regex` crate is 3x faster than
// the usual regex we use.
//
// However, given that we're using a regex parse-able by `regex`, there isn't much difference
// between using the `regex` crate and using the `fancy_regex` crate.
//
// There is an important interaction between threading, `regex` and `fancy_regex`.
// When using `fancy_regex`, we hit regex.find_at. It turns out that this causes contention on
// some mutable scratch space inside the regex. This absolutely kills performance. When using plain
// old `regex`, we don't hit this, because `regex` clones the regex for each thread.
//
// Cloning the regex is expensive, so we rely on thread locals to avoid doing it too often.
// This is a bit tricky, but it's worth it for the performance boost.

fn _get_regex(regex_str: &str) -> Result<Regex, fancy_regex::Error> {
    Regex::new(regex_str)
}

#[derive(Debug, Clone)]
/// Tokenizer that doesn't have any special tokens and regex patterns
pub struct FakeTokenizer {
    encoder: HashMap<Vec<u8>, Rank>,
    decoder: HashMap<Rank, Vec<u8>>,
}

impl FakeTokenizer {
    pub fn new(encoder: HashMap<Vec<u8>, Rank>) -> Self {
        let mut decoder = HashMap::default();
        for (k, v) in &encoder {
            decoder.insert(*v, k.clone());
        }

        Self { encoder, decoder }
    }

    pub fn encode(&self, text: &str) -> Vec<Rank> {
        match self.encoder.get(text.as_bytes()) {
            Some(token) => vec![*token],
            None => byte_pair_encode(text.as_bytes(), &self.encoder),
        }
    }

    pub fn decode(&self, tokens: Vec<Rank>) -> Result<String, DecodeError> {
        let bytes = self.decode_bytes(tokens)?;
        Ok(unsafe { String::from_utf8_unchecked(bytes) })
    }

    fn decode_bytes(&self, tokens: Vec<Rank>) -> Result<Vec<u8>, DecodeError> {
        let mut output = Vec::with_capacity(tokens.len() * 2);
        for token in tokens {
            let bytes = self.decoder.get(&token).ok_or(DecodeError {
                message: format!("Invalid token: {}", token),
            })?;
            output.extend_from_slice(bytes);
        }
        Ok(output)
    }
}

fn hash_current_thread() -> usize {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    
    let id = thread::current().id();
    let mut hasher = DefaultHasher::new();
    id.hash(&mut hasher);
    hasher.finish() as usize
}

#[derive(Debug)]
pub struct DecodeKeyError {
    pub token: Rank,
}

impl fmt::Display for DecodeKeyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Invalid token for decoding: {}", self.token)
    }
}

impl std::error::Error for DecodeKeyError {}

#[derive(Debug)]
pub struct DecodeError {
    pub message: String,
}

use std::fmt;

impl fmt::Display for DecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Could not decode tokens: {}", self.message)
    }
}

impl std::error::Error for DecodeError {}

#[derive(Debug, Clone)]
pub struct EncodeError {
    pub message: String,
}

impl std::fmt::Display for EncodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "Could not encode string: {}", self.message)
    }
}

impl std::error::Error for EncodeError {}

const MAX_NUM_THREADS: usize = 128;

#[cfg_attr(feature = "python", pyclass)]
#[derive(Clone)]
pub struct CoreBPE {
    encoder: HashMap<Vec<u8>, Rank>,
    special_tokens_encoder: HashMap<String, Rank>,
    decoder: HashMap<Rank, Vec<u8>>,
    special_tokens_decoder: HashMap<Rank, Vec<u8>>,
    regex_tls: Vec<Regex>,
    special_regex_tls: Vec<Regex>,
    sorted_token_bytes: Vec<Vec<u8>>,
}

impl CoreBPE {
    fn _get_tl_regex(&self) -> &Regex {
        // See performance notes above for what this is about
        // It's also a little janky, please make a better version of it!
        // However, it's nice that this doesn't leak memory to short-lived threads
        &self.regex_tls[hash_current_thread() % MAX_NUM_THREADS]
    }

    fn _get_tl_special_regex(&self) -> &Regex {
        &self.special_regex_tls[hash_current_thread() % MAX_NUM_THREADS]
    }

    /// Decodes tokens into a list of bytes.
    ///
    /// The bytes are not gauranteed to be a valid utf-8 string.
    pub fn decode_bytes(&self, tokens: &[Rank]) -> Result<Vec<u8>, DecodeKeyError> {
        let mut ret = Vec::with_capacity(tokens.len() * 2);
        for &token in tokens {
            let token_bytes = match self.decoder.get(&token) {
                Some(bytes) => bytes,
                None => self
                    .special_tokens_decoder
                    .get(&token)
                    .ok_or(DecodeKeyError { token })?,
            };
            ret.extend(token_bytes);
        }
        Ok(ret)
    }

    pub fn encode_ordinary(&self, text: &str) -> Vec<Rank> {
        // This is the core of the encoding logic; the other functions in here
        // just make things complicated :-)
        let regex = self._get_tl_regex();
        let mut ret = vec![];
        for mat in regex.find_iter(text) {
            let piece = mat.unwrap().as_str().as_bytes();
            if let Some(token) = self.encoder.get(piece) {
                ret.push(*token);
                continue;
            }
            ret.extend(&byte_pair_encode(piece, &self.encoder));
        }
        ret
    }

    pub fn encode(
        &self,
        text: &str,
        allowed_special: &HashSet<&str>,
    ) -> Result<(Vec<Rank>, usize), EncodeError> {
        let special_regex = self._get_tl_special_regex();
        let regex = self._get_tl_regex();
        let mut ret = vec![];

        let mut start = 0;
        let mut last_piece_token_len = 0;
        loop {
            let mut next_special;
            let mut start_find = start;
            loop {
                // Find the next allowed special token, if any
                next_special = special_regex.find_from_pos(text, start_find).unwrap();
                match next_special {
                    Some(m) => {
                        if allowed_special.contains(&text[m.start()..m.end()]) {
                            break;
                        }
                        start_find = m.start() + 1;
                    }
                    None => break,
                }
            }
            let end = next_special.map_or(text.len(), |m| m.start());

            // Okay, here we go, compare this logic to encode_ordinary
            for mat_res in regex.find_iter(&text[start..end]) {
                let mat = match mat_res {
                    Ok(m) => m,
                    Err(e) => {
                        return Err(EncodeError {
                            message: format!("Regex error while tokenizing: {e}"),
                        });
                    }
                };

                let piece = mat.as_str().as_bytes();
                if let Some(token) = self.encoder.get(piece) {
                    last_piece_token_len = 1;
                    ret.push(*token);
                    continue;
                }
                let tokens = byte_pair_encode(piece, &self.encoder);
                last_piece_token_len = tokens.len();
                ret.extend(&tokens);
            }

            match next_special {
                // And here we push the special token
                Some(m) => {
                    let piece = m.as_str();
                    let token = self.special_tokens_encoder[piece];
                    ret.push(token);
                    start = m.end();
                    last_piece_token_len = 0;
                }
                None => break,
            }
        };

        // last_piece_token_len is how many tokens came from the last regex split. This is used
        // for determining unstable tokens, since you can't merge across (stable) regex splits
        Ok((ret, last_piece_token_len))
    }

    fn _increase_last_piece_token_len(
        &self,
        tokens: Vec<Rank>,
        mut last_piece_token_len: usize,
    ) -> (Vec<Rank>, usize) {
        // Unfortunately, the locations where our regex splits can be unstable.
        // For the purposes of determining unstable tokens, unstable regex splitting
        // is only a problem if a split that was present disappears, since this can
        // lead to merging of tokens otherwise thought to be stable.
        // cl100k_base makes our life hard by including the \s*[\r\n]+
        // pattern. This can e.g. cause "\n" + " " to become "\n \n".
        // Here is a quick and dirty fix:
        {
            let token_is_all_space = |token| {
                self.decoder
                    .get(token)
                    .map(|token_bytes| {
                        token_bytes
                            .iter()
                            .rev()
                            .all(|&b| [b' ', b'\n', b'\r', b'\t'].contains(&b))
                    })
                    .unwrap_or(false)
            };
            if last_piece_token_len > 0
                && token_is_all_space(&tokens[tokens.len() - last_piece_token_len])
            {
                while (last_piece_token_len < tokens.len())
                    && token_is_all_space(&tokens[tokens.len() - last_piece_token_len - 1])
                {
                    last_piece_token_len += 1;
                }
            }
        }
        debug_assert!(last_piece_token_len <= tokens.len());

        (tokens, last_piece_token_len)
    }

    fn _encode_unstable_native(
        &self,
        text: &str,
        allowed_special: &HashSet<&str>,
    ) -> (Vec<Rank>, HashSet<Vec<Rank>>) {
        let (tokens, last_piece_token_len) = self.encode(text, allowed_special).unwrap();
        if last_piece_token_len == 0 {
            // If last_piece_token_len is zero, the last token was a special token and we have
            // no unstable bytes
            return (tokens, HashSet::new());
        }
        let (mut tokens, last_piece_token_len) =
            self._increase_last_piece_token_len(tokens, last_piece_token_len);

        let unstable_bytes = self
            .decode_bytes(&tokens[tokens.len() - last_piece_token_len..])
            .unwrap();
        tokens.truncate(tokens.len() - last_piece_token_len);

        // TODO: we should try harder to find additional stable tokens
        // This would reduce the amount of retokenising when determining completions
        // Refer to the logic in an older version of this file

        let mut completions = HashSet::new();
        if unstable_bytes.is_empty() {
            return (tokens, completions);
        }

        // This is the easy bit. Just find all single tokens that start with unstable_bytes
        // (including tokens that exactly match unstable_bytes)
        // Separating this from the loop below helps with performance in a common case.
        let point = unstable_bytes.as_slice();
        for tokens in &self.sorted_token_bytes {
            let s = tokens.as_slice();
            if s < point {
                continue;
            } else if s == point {
                // s == point
                let token = self.encoder[tokens];
                completions.insert(vec![token]);
            } else {
                // s > point
                // Check whether s starts with point
                if s.starts_with(point) {
                    let token = self.encoder[tokens];
                    completions.insert(vec![token]);
                } else {
                    // Otherwise, try to skip many bytes
                    if s.len() >= point.len() {
                        // Since this optimization is complex and not critical for our use case,
                        // we'll skip it for now
                        break;
                    }
                }
            }
        }

        // This is also not straightforward. While we generally assume that regex splits are stable,
        // unfortunately, they are not. That is, if adding bytes were to make a split appear in
        // unstable_bytes, this could make tokens possible which our logic would otherwise think
        // would be merged.
        // For example, with gpt2, the use of \s+(?!\S) means that "\n\n" could
        // develop a split, e.g. "\n\n0" splits into "\n"+"\n"+"0", making "\n" a possible token.
        // Here is a quick and dirty fix:
        // This isn't right if we ever remove \s+(?!\S)
        if unstable_bytes.len() > 1 {
            let last_decoded = bstr::decode_last_utf8(unstable_bytes.as_slice());
            if unstable_bytes.len() - last_decoded.1 > 0
                && last_decoded.0.is_some_and(|c| c.is_whitespace())
            {
                let mut reencoded = byte_pair_encode(
                    &unstable_bytes[..unstable_bytes.len() - last_decoded.1],
                    &self.encoder,
                );
                reencoded.extend(byte_pair_encode(
                    &unstable_bytes[unstable_bytes.len() - last_decoded.1..],
                    &self.encoder,
                ));
                completions.insert(reencoded);
            }
        }

        // This is also a valid continuation of unstable_bytes (any token that starts with unstable_bytes)
        completions.insert(vec![]);

        (tokens, completions)
    }

    pub fn encode_with_special_tokens(&self, text: &str) -> Vec<Rank> {
        let special_regex = self._get_tl_special_regex();
        let regex = self._get_tl_regex();
        let mut ret = vec![];

        let mut start = 0;
        loop {
            let mat = special_regex.find_from_pos(text, start).unwrap();

            // First, handle any text before the special token
            let end = mat.as_ref().map_or(text.len(), |m| m.start());
            for m in regex.find_iter(&text[start..end]) {
                let piece = m.unwrap().as_str().as_bytes();
                if let Some(token) = self.encoder.get(piece) {
                    ret.push(*token);
                    continue;
                }
                ret.extend(&byte_pair_encode(piece, &self.encoder));
            }

            match mat {
                Some(m) => {
                    let piece = m.as_str();
                    if let Some(token) = self.special_tokens_encoder.get(piece) {
                        ret.push(*token);
                        start = m.end();
                    } else {
                        // This should never happen, but handle it gracefully
                        eprintln!("Special token not found: {}", piece);
                        start = m.end();
                    }
                }
                None => break,
            }
        }

        ret
    }

    fn new_internal(
        encoder: HashMap<Vec<u8>, Rank>,
        special_tokens_encoder: HashMap<String, Rank>,
        pattern: &str,
    ) -> Result<Self, fancy_regex::Error> {
        let regex_vec: Result<Vec<_>, _> = (0..MAX_NUM_THREADS)
            .map(|_| Regex::new(pattern))
            .collect();
        let regex_vec = regex_vec?;

        let special_regex_vec: Result<Vec<_>, _> = (0..MAX_NUM_THREADS)
            .map(|_| {
                let s = special_tokens_encoder
                    .keys()
                    .map(|s| fancy_regex::escape(s))
                    .collect::<Vec<_>>()
                    .join("|");
                Regex::new(&s)
            })
            .collect();
        let special_regex_vec = special_regex_vec?;

        let mut decoder: HashMap<Rank, Vec<u8>> =
            HashMap::with_capacity_and_hasher(encoder.len(), Default::default());
        for (k, v) in &encoder {
            decoder.insert(*v, k.clone());
        }

        assert!(encoder.len() == decoder.len());

        let mut special_tokens_decoder: HashMap<Rank, Vec<u8>> =
            HashMap::with_capacity_and_hasher(special_tokens_encoder.len(), Default::default());
        for (k, v) in &special_tokens_encoder {
            special_tokens_decoder.insert(*v, k.as_bytes().to_vec());
        }

        // Clone because I don't know how to tell Rust I'm not going to change the map
        let mut sorted_token_bytes: Vec<Vec<u8>> = encoder.keys().cloned().collect();
        sorted_token_bytes.sort_unstable();

        Ok(Self {
            encoder,
            special_tokens_encoder,
            decoder,
            special_tokens_decoder,
            regex_tls: regex_vec,
            special_regex_tls: special_regex_vec,
            sorted_token_bytes,
        })
    }

    pub fn new(
        encoder: HashMap<Vec<u8>, Rank>,
        special_tokens_encoder: HashMap<String, Rank>,
        pattern: &str,
    ) -> Result<Self, fancy_regex::Error> {
        Self::new_internal(encoder, special_tokens_encoder, pattern)
    }
}

#[cfg(test)]
mod tests {
    use fancy_regex::Regex;
    use rustc_hash::FxHashMap as HashMap;

    use crate::{Rank, byte_pair_split};

    fn setup_ranks() -> HashMap<Vec<u8>, Rank> {
        HashMap::from_iter([(b"ab".to_vec(), 0), (b"cd".to_vec(), 1)])
    }

    #[test]
    fn test_simple_characters() {
        let ranks = setup_ranks();
        let res = byte_pair_split(b"abcd", &ranks);
        assert_eq!(res, vec![b"ab", b"cd"]);
    }

    #[test]
    fn test_repeated_characters() {
        let ranks = setup_ranks();
        let res = byte_pair_split(b"abab", &ranks);
        assert_eq!(res, vec![b"ab", b"ab"]);
    }
}
