// This check is new and seems buggy (possibly with PyO3 interaction)
#![allow(clippy::borrow_deref_ref)]

use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::iter::successors;
use std::num::NonZeroU64;
use std::thread;

use fancy_regex::Regex;
use pyo3::exceptions;
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedStr;
use pyo3::types::{PyBytes, PyList, PyTuple};
use pyo3::PyResult;
use rustc_hash::FxHashMap as HashMap;

type Rank = u32;

const LARGE_ENCODER_CHARACTER_LIMIT: usize = 500;

fn _byte_pair_merge(ranks: &HashMap<Vec<u8>, Rank>, piece: &[u8]) -> Vec<(usize, Rank)> {
    if piece.len() < LARGE_ENCODER_CHARACTER_LIMIT {
        _byte_pair_merge_small(ranks, piece) // Quadratic, but lightweight
    } else {
        _byte_pair_merge_large(ranks, piece) // Linearithmic, but heavy
    }
}

fn _byte_pair_merge_small(ranks: &HashMap<Vec<u8>, Rank>, piece: &[u8]) -> Vec<(usize, Rank)> {
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
    // n is often very small so considerations like cache-locality outweigh the algorithmic
    // complexity downsides of the `parts` vector.
    while min_rank.0 != Rank::MAX {
        let i = min_rank.1;
        // Update parts[i] and parts[i - 1] before removing parts[i + 1], since
        // `parts.remove(i + 1)` will thrash the cache.
        if i > 0 {
            parts[i - 1].1 = get_rank(&parts, i - 1);
        }
        parts[i].1 = get_rank(&parts, i);
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

fn _byte_pair_merge_large(ranks: &HashMap<Vec<u8>, Rank>, piece: &[u8]) -> Vec<(usize, Rank)> {
    let mut rank_indexes = BTreeMap::<Rank, BTreeSet<usize>>::new();
    let mut index_rank = vec![Rank::MAX; piece.len() + 1];
    let mut index_prev = vec![usize::MAX; piece.len() + 1];
    let mut index_next = vec![usize::MAX; piece.len() + 1];

    let get_rank = |start_idx: usize, end_idx: usize| -> Rank {
        *piece.get(start_idx..end_idx)
            .and_then(|p| ranks.get(p))
            .unwrap_or(&Rank::MAX)
    };

    let mut prev_node = None;
    for i in 0..=piece.len() {
        let rank = get_rank(i, i + 2);
        index_rank[i] = rank;
        if let Some(prev) = prev_node {
            index_prev[i] = prev;
            index_next[prev] = i;
        }
        prev_node = Some(i);

        rank_indexes.entry(rank).or_default().insert(i);
    }

    while rank_indexes.len() > 1 {
        let mut skip_next = false;
        if let Some((_, nodes)) = rank_indexes.pop_first() {
            for &min_node in &nodes {
                if skip_next {
                    skip_next = false;
                    continue;
                }

                let min_rank = index_rank[min_node];

                let prev_node = index_prev[min_node];
                let next_node = index_next[min_node];
                let next_next_node = index_next[next_node];
                let next_next_next_node = index_next[next_next_node];

                if prev_node != usize::MAX {
                    let new_rank = get_rank(prev_node, next_next_node);
                    if index_rank[prev_node] != new_rank {
                        rank_indexes.get_mut(&index_rank[prev_node]).unwrap().remove(&prev_node);
                        index_rank[prev_node] = new_rank;
                        rank_indexes.entry(new_rank).or_default().insert(prev_node);
                    }
                }

                let new_rank = get_rank(min_node, next_next_next_node);
                index_rank[min_node] = new_rank;
                rank_indexes.entry(new_rank).or_default().insert(min_node);

                index_next[min_node] = next_next_node;
                index_prev[next_next_node] = min_node;

                let next_node_rank = index_rank[next_node];
                if next_node_rank == min_rank {
                    skip_next = true;
                } else if next_node_rank != Rank::MAX {
                    rank_indexes.get_mut(&next_node_rank).unwrap().remove(&next_node);
                }
            }
        }
    }

    successors(Some(0), |&n| index_next.get(n).filter(|&&x| x != usize::MAX).copied())
        .map(|n| (n, Rank::MAX))
        .collect()
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
// When using `fancy_regex`, we hit `regex.find_at`. It turns out that this causes contention on
// some mutable scratch space inside of `regex`. This absolutely kills performance. When using plain
// old `regex`, we don't hit this, because `find_iter` has a different code path.
// Related: https://github.com/rust-lang/regex/blob/master/PERFORMANCE.md
// Anyway, the way we get around this is with having a (mostly) thread local clone of the regex for
// each thread.
//
// Threading
// =========
// I tried using `rayon`. It wasn't really faster than using Python threads and releasing the GIL.
// So goodbye `rayon`! Let thread count etc be in control of our Python users.
//
// Caching
// =======
// The reference tokeniser has an lru cache over the equivalent of `byte_pair_encode`.
// Originally, we had one too! Without it, we were only vaguely faster than Python.
// I used an RWLock to protect the cache. This didn't seem to hurt single threaded performance
// noticeably, but it did affect multi-threaded performance. Weirdly, it seemed to affect
// multi-threaded performance even when I only had readers (maybed I messed something up?).
// Anyway, I realised that we could get rid of the cache, if we treat the set of tokens as a cache!
// These are exactly the set or merges that are likely to be hot. And now we don't have to think
// about interior mutability, memory use, or cloning.
//
// Hashing
// =======
// We use FxHashMap instead of the standard HashMap. This is maybe like a 5-10% win?
// The current implementation ends up doing a lot of hashing of bytes. In theory, this could be made
// to be hashing of two-tuples of ints, which looks like it may also be a couple percent faster.

pub struct FakeThreadId(NonZeroU64);

fn hash_current_thread() -> usize {
    // It's easier to use unsafe than to use nightly. Rust has this nice u64 thread id counter
    // that works great for our use case of avoiding collisions in our array. Unfortunately,
    // it's private. However, there are only so many ways you can layout a u64, so just transmute
    // https://github.com/rust-lang/rust/issues/67939
    const _: [u8; 8] = [0; std::mem::size_of::<std::thread::ThreadId>()];
    const _: [u8; 8] = [0; std::mem::size_of::<FakeThreadId>()];
    let x = unsafe {
        std::mem::transmute::<std::thread::ThreadId, FakeThreadId>(thread::current().id()).0
    };
    u64::from(x) as usize
}

#[derive(Debug, Clone)]
struct DecodeKeyError {
    token: Rank,
}

impl std::fmt::Display for DecodeKeyError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "Invalid token for decoding: {}", self.token)
    }
}

const MAX_NUM_THREADS: usize = 128;

#[pyclass]
struct CoreBPE {
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

    fn _decode_native(&self, tokens: &[Rank]) -> Result<Vec<u8>, DecodeKeyError> {
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

    fn _encode_ordinary_native(&self, text: &str) -> Vec<Rank> {
        // This is the core of the encoding logic; the other functions in here
        // just make things complicated :-)
        let regex = self._get_tl_regex();
        let mut ret = vec![];
        for mat in regex.find_iter(text) {
            let piece = mat.unwrap().as_str().as_bytes();
            match self.encoder.get(piece) {
                Some(token) => ret.push(*token),
                None => ret.extend(&byte_pair_encode(piece, &self.encoder)),
            }
        }
        ret
    }

    fn _encode_native(&self, text: &str, allowed_special: &HashSet<&str>) -> (Vec<Rank>, usize) {
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

            // Okay, here we go, compare this logic to _encode_ordinary_native
            for mat in regex.find_iter(&text[start..end]) {
                let piece = mat.unwrap().as_str().as_bytes();
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
        }

        // last_piece_token_len is how many tokens came from the last regex split. This is used
        // for determining unstable tokens, since you can't merge across (stable) regex splits
        (ret, last_piece_token_len)
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
                            .all(|&b| [b' ', b'\n', b'\t'].contains(&b))
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
        let (tokens, last_piece_token_len) = self._encode_native(text, allowed_special);
        if last_piece_token_len == 0 {
            // If last_piece_token_len is zero, the last token was a special token and we have
            // no unstable bytes
            return (tokens, HashSet::new());
        }
        let (mut tokens, last_piece_token_len) =
            self._increase_last_piece_token_len(tokens, last_piece_token_len);

        let unstable_bytes = self
            ._decode_native(&tokens[tokens.len() - last_piece_token_len..])
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
        let mut point = self
            .sorted_token_bytes
            .partition_point(|x| x.as_slice() < unstable_bytes.as_slice());
        while point < self.sorted_token_bytes.len()
            && self.sorted_token_bytes[point].starts_with(&unstable_bytes)
        {
            completions.insert(vec![
                self.encoder[self.sorted_token_bytes[point].as_slice()],
            ]);
            point += 1;
        }

        // Now apply even more brute force. At every (other) possible position for the straddling
        // token, concatenate additional bytes from that token (if any) to unstable_bytes,
        // and retokenise the whole thing and see what we get.
        for i in 1..unstable_bytes.len() {
            let prefix = &unstable_bytes[..i];
            let suffix = &unstable_bytes[i..];
            let mut point = self
                .sorted_token_bytes
                .partition_point(|x| x.as_slice() < suffix);
            // TODO: Perf optimisation if suffix starts with " "?
            while point < self.sorted_token_bytes.len()
                && self.sorted_token_bytes[point].starts_with(suffix)
            {
                let possibility = [prefix, self.sorted_token_bytes[point].as_slice()].concat();
                let encoded = match std::str::from_utf8(&possibility) {
                    // Morally, this is byte_pair_encode(&possibility, &self.encoder)
                    // But we might have introduced a regex split which would prevent merges.
                    // (particularly possible in the presence of unstable regex splits)
                    // So convert to UTF-8 and do regex splitting.
                    // E.g. with cl100k_base "  !" gets split to " " + " !",
                    // but byte_pair_encode("  !") != byte_pair_encode(" ")
                    Ok(s) => self._encode_ordinary_native(s),

                    // Technically, whether or not this arm is correct depends on whether there
                    // would be a regex split before the UTF-8 truncation point.
                    // Probably niche enough that no one will ever notice (after all, people didn't
                    // notice all the big holes in the previous unstable token implementation)
                    Err(_) => byte_pair_encode(&possibility, &self.encoder),
                    // Something like the following is intriguing but incorrect:
                    // Err(e) => self._encode_ordinary_native(unsafe {
                    //     std::str::from_utf8_unchecked(&possibility[..e.valid_up_to()])
                    // }),
                };
                let mut seq = Vec::new();
                let mut seq_len = 0;
                for token in encoded {
                    seq.push(token);
                    seq_len += self.decoder[&token].len();
                    if seq_len >= unstable_bytes.len() {
                        break;
                    }
                }
                completions.insert(seq);
                point += 1;
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
                && last_decoded.0.map_or(false, |c| c.is_whitespace())
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

        (tokens, completions)
    }
}

#[pymethods]
impl CoreBPE {
    #[new]
    fn new(
        encoder: HashMap<Vec<u8>, Rank>,
        special_tokens_encoder: HashMap<String, Rank>,
        pattern: &str,
    ) -> PyResult<Self> {
        let regex = Regex::new(pattern)
            .map_err(|e| PyErr::new::<exceptions::PyValueError, _>(e.to_string()))?;

        let special_regex = {
            let _parts = special_tokens_encoder
                .keys()
                .map(|s| fancy_regex::escape(s))
                .collect::<Vec<_>>();
            Regex::new(&_parts.join("|"))
                .map_err(|e| PyErr::new::<exceptions::PyValueError, _>(e.to_string()))?
        };

        let decoder: HashMap<Rank, Vec<u8>> =
            encoder.iter().map(|(k, v)| (*v, k.clone())).collect();

        assert!(
            encoder.len() == decoder.len(),
            "Encoder and decoder must be of equal length; maybe you had duplicate token indices in your encoder?"
        );

        let special_tokens_decoder: HashMap<Rank, Vec<u8>> = special_tokens_encoder
            .iter()
            .map(|(k, v)| (*v, k.as_bytes().to_vec()))
            .collect();

        // Clone because I don't know how to tell Rust I'm not going to change the map
        let mut sorted_token_bytes: Vec<Vec<u8>> = encoder.keys().cloned().collect();
        sorted_token_bytes.sort();

        Ok(CoreBPE {
            encoder,
            special_tokens_encoder,
            decoder,
            special_tokens_decoder,
            regex_tls: (0..MAX_NUM_THREADS).map(|_| regex.clone()).collect(),
            special_regex_tls: (0..MAX_NUM_THREADS)
                .map(|_| special_regex.clone())
                .collect(),
            sorted_token_bytes,
        })
    }

    // ====================
    // Encoding
    // ====================

    fn encode_ordinary(&self, py: Python, text: &str) -> Vec<Rank> {
        py.allow_threads(|| self._encode_ordinary_native(text))
    }

    fn encode(&self, py: Python, text: &str, allowed_special: HashSet<PyBackedStr>) -> Vec<Rank> {
        py.allow_threads(|| {
            let allowed_special: HashSet<&str> =
                allowed_special.iter().map(|s| s.as_ref()).collect();
            self._encode_native(text, &allowed_special).0
        })
    }

    fn encode_to_tiktoken_buffer(
        &self,
        py: Python,
        text: &str,
        allowed_special: HashSet<PyBackedStr>,
    ) -> Py<PyAny> {
        let tokens = py.allow_threads(|| {
            let allowed_special: HashSet<&str> =
                allowed_special.iter().map(|s| s.as_ref()).collect();
            self._encode_native(text, &allowed_special).0
        });
        let buffer = TiktokenBuffer { tokens };
        buffer.into_py(py)
    }

    fn _encode_bytes(&self, py: Python, bytes: &[u8]) -> Vec<Rank> {
        py.allow_threads(|| {
            match std::str::from_utf8(bytes) {
                Ok(text) => self._encode_ordinary_native(text),
                Err(e) => {
                    let text = unsafe { std::str::from_utf8_unchecked(&bytes[..e.valid_up_to()]) };
                    let (tokens, last_piece_token_len) = self._encode_native(text, &HashSet::new());
                    let (mut tokens, last_piece_token_len) =
                        self._increase_last_piece_token_len(tokens, last_piece_token_len);
                    if !tokens.is_empty() && last_piece_token_len > 0 {
                        // Lop off the tokens from the last piece and run BPE on the remaining bytes
                        // Somewhat niche, but this may not be correct if we'd have had a regex
                        // split between the valid UTF-8 and the invalid bytes, which is why this
                        // method is private
                        let mut unstable_bytes = self
                            ._decode_native(&tokens[tokens.len() - last_piece_token_len..])
                            .unwrap();
                        unstable_bytes.extend_from_slice(&bytes[e.valid_up_to()..]);

                        tokens.truncate(tokens.len() - last_piece_token_len);
                        match self.encoder.get(&unstable_bytes) {
                            Some(token) => tokens.push(*token),
                            None => {
                                tokens.extend(&byte_pair_encode(&unstable_bytes, &self.encoder))
                            }
                        }
                    }
                    tokens
                }
            }
        })
    }

    fn encode_with_unstable(
        &self,
        py: Python,
        text: &str,
        allowed_special: HashSet<PyBackedStr>,
    ) -> Py<PyTuple> {
        let (tokens, completions) = py.allow_threads(|| {
            let allowed_special: HashSet<&str> =
                allowed_special.iter().map(|s| s.as_ref()).collect();
            self._encode_unstable_native(text, &allowed_special)
        });
        let py_completions = PyList::new_bound(
            py,
            completions
                .iter()
                .map(|seq| PyList::new_bound(py, &seq[..])),
        );
        (tokens, py_completions).into_py(py)
    }

    fn encode_single_token(&self, piece: &[u8]) -> PyResult<Rank> {
        if let Some(token) = self.encoder.get(piece).copied() {
            return Ok(token);
        }
        if let Ok(piece_str) = std::str::from_utf8(piece) {
            if let Some(token) = self.special_tokens_encoder.get(piece_str).copied() {
                return Ok(token);
            }
        }
        Err(PyErr::new::<exceptions::PyKeyError, _>(piece.to_owned()))
    }

    fn encode_single_piece(&self, piece: &[u8]) -> Vec<Rank> {
        if let Some(token) = self.encoder.get(piece) {
            return vec![*token];
        }
        byte_pair_encode(piece, &self.encoder)
    }

    // ====================
    // Decoding
    // ====================

    fn decode_bytes(&self, py: Python, tokens: Vec<Rank>) -> Result<Py<PyBytes>, PyErr> {
        match py.allow_threads(|| self._decode_native(&tokens)) {
            Ok(bytes) => Ok(PyBytes::new_bound(py, &bytes).into()),
            Err(e) => Err(pyo3::exceptions::PyKeyError::new_err(format!("{}", e))),
        }
    }

    fn decode_single_token_bytes(&self, py: Python, token: Rank) -> PyResult<Py<PyBytes>> {
        if let Some(bytes) = self.decoder.get(&token) {
            return Ok(PyBytes::new_bound(py, bytes).into());
        }
        if let Some(bytes) = self.special_tokens_decoder.get(&token) {
            return Ok(PyBytes::new_bound(py, bytes).into());
        }
        Err(PyErr::new::<exceptions::PyKeyError, _>(token.to_string()))
    }

    // ====================
    // Miscellaneous
    // ====================

    fn token_byte_values(&self, py: Python) -> Vec<Py<PyBytes>> {
        self.sorted_token_bytes
            .iter()
            .map(|x| PyBytes::new_bound(py, x).into())
            .collect()
    }
}

#[pyclass]
struct TiktokenBuffer {
    tokens: Vec<Rank>,
}

#[pymethods]
impl TiktokenBuffer {
    // Based on https://github.com/PyO3/pyo3/blob/v0.22.2/tests/test_buffer_protocol.rs#L25
    unsafe fn __getbuffer__(
        slf: Bound<'_, Self>,
        view: *mut pyo3::ffi::Py_buffer,
        flags: std::os::raw::c_int,
    ) -> PyResult<()> {
        if view.is_null() {
            return Err(pyo3::exceptions::PyBufferError::new_err("View is null"));
        }
        if (flags & pyo3::ffi::PyBUF_WRITABLE) == pyo3::ffi::PyBUF_WRITABLE {
            return Err(pyo3::exceptions::PyBufferError::new_err(
                "Object is not writable",
            ));
        }

        (*view).obj = slf.clone().into_any().into_ptr();

        let data = &slf.borrow().tokens;
        (*view).buf = data.as_ptr() as *mut std::os::raw::c_void;
        (*view).len = (data.len() * std::mem::size_of::<Rank>()) as isize;
        (*view).readonly = 1;
        (*view).itemsize = std::mem::size_of::<Rank>() as isize;
        (*view).format = if (flags & pyo3::ffi::PyBUF_FORMAT) == pyo3::ffi::PyBUF_FORMAT {
            let msg = std::ffi::CString::new("I").unwrap();
            msg.into_raw()
        } else {
            std::ptr::null_mut()
        };
        (*view).ndim = 1;
        (*view).shape = if (flags & pyo3::ffi::PyBUF_ND) == pyo3::ffi::PyBUF_ND {
            &mut (*view).len
        } else {
            std::ptr::null_mut()
        };
        (*view).strides = if (flags & pyo3::ffi::PyBUF_STRIDES) == pyo3::ffi::PyBUF_STRIDES {
            &mut (*view).itemsize
        } else {
            std::ptr::null_mut()
        };
        (*view).suboffsets = std::ptr::null_mut();
        (*view).internal = std::ptr::null_mut();

        Ok(())
    }

    unsafe fn __releasebuffer__(&self, view: *mut pyo3::ffi::Py_buffer) {
        std::mem::drop(std::ffi::CString::from_raw((*view).format));
    }
}

#[pymodule]
fn _tiktoken(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<CoreBPE>()?;
    Ok(())
}

#[cfg(test)]
mod tests {

    use rustc_hash::FxHashMap as HashMap;

    use crate::{byte_pair_split, Rank};

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
