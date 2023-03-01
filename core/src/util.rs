use rustc_hash::FxHashMap as HashMap;

fn _byte_pair_merge<T>(
    piece: &[u8],
    ranks: &HashMap<Vec<u8>, usize>,
    f: impl Fn(std::ops::Range<usize>) -> T,
) -> Vec<T> {
    // This is a vector of (start, rank).
    // The rank is of the byte pair starting at position start.
    // The rank of the last item in the vector is not a valid value.
    let mut parts: Vec<(usize, usize)> = (0..piece.len() + 1).map(|i| (i, usize::MAX)).collect();

    // NOTE: using a macro here because a closure fails to get inlined
    // according to optimization remarks.
    // A closure also cannot capture a reference to `piece` without
    // the borrow checker complaining about the mutable borrows during
    // the assignments later in this code.
    macro_rules! get_rank {
        ($start_idx:expr, $skip:expr) => {{
            let start_idx: usize = $start_idx;
            let skip: usize = $skip;
            if (start_idx + skip + 2) < parts.len() {
                ranks
                    .get(&piece[parts[start_idx].0..parts[start_idx + skip + 2].0])
                    .map(|r| *r)
            } else {
                None
            }
        }};
        ($idx:expr) => {{
            get_rank!($idx, 0)
        }};
    }

    // We look up the ranks once in the beggining and iteratively update
    // them during each merge, which reduces the number of rank lookups.
    for i in 0..parts.len() - 2 {
        match get_rank!(i) {
            Some(rank) => {
                // usize::MAX is a sentinel value and cannot be a valid rank
                debug_assert!(rank != usize::MAX);
                parts[i].1 = rank;
            }
            None => {
                continue;
            }
        };
    }

    // If you have n parts and m merges, this does O(mn) work.
    // We could do something with a heap and do O(m log n) work.
    // It is important to consider that n is often small (<100), and as such
    // the cache-locality benefits outweigh the algorithmic complexity downsides
    // of the `parts` vector data structure above.

    // Note that we hash bytes, not token pairs. As long as we train BPE the way we
    // currently do, this is equivalent. An easy way to break this would be to decouple
    // merge priority from token index or to prevent specific token merges.
    loop {
        if parts.len() == 1 {
            break;
        }

        // usize::MAX is a sentinel rank value allowing us to
        // take the min more quickly
        let mut min_rank: (usize, usize) = (usize::MAX, 0);
        for (i, &(_, rank)) in parts[..parts.len() - 1].iter().enumerate() {
            if rank < min_rank.0 {
                min_rank = (rank, i);
            }
        }

        if min_rank.0 != usize::MAX {
            let i = min_rank.1;

            // NOTE: We are about to remove parts[i + 1]. We do not do it
            // yet because there are cache-locality benefits to updating
            // parts[i] and parts[i-1] before removing, which could thrash
            // the cache. Thus, we update the rank calculation by skipping over
            // parts[i + 1], by invoking `get_rank!` with `skip = 1`.
            parts[i].1 = get_rank!(i, 1).unwrap_or(usize::MAX);
            if i > 0 {
                parts[i - 1].1 = get_rank!(i - 1, 1).unwrap_or(usize::MAX);
            }

            parts.remove(i + 1);
        } else {
            break;
        }
    }
    let mut out: Vec<T> = Vec::with_capacity(parts.len() - 1);
    for i in 0..parts.len() - 1 {
        out.push(f(parts[i].0..parts[i + 1].0));
    }
    out
}

pub fn byte_pair_encode(piece: &[u8], ranks: &HashMap<Vec<u8>, usize>) -> Vec<usize> {
    if piece.len() == 1 {
        return vec![ranks[piece]];
    }
    _byte_pair_merge(piece, ranks, |p| ranks[&piece[p.start..p.end]])
}

pub fn byte_pair_split<'a>(piece: &'a [u8], ranks: &HashMap<Vec<u8>, usize>) -> Vec<&'a [u8]> {
    if piece.len() == 1 {
        return vec![piece];
    }
    _byte_pair_merge(piece, ranks, |p| &piece[p.start..p.end])
}

#[cfg(test)]
mod tests {
    use rustc_hash::FxHashMap as HashMap;

    use crate::util::_byte_pair_merge;
    pub fn byte_pair_split<'a>(piece: &'a [u8], ranks: &HashMap<Vec<u8>, usize>) -> Vec<&'a [u8]> {
        if piece.len() == 1 {
            return vec![piece];
        }
        _byte_pair_merge(piece, ranks)
            .iter()
            .map(|p| &piece[p.start..p.end])
            .collect()
    }

    #[test]
    fn very_simple_test() {
        let mut ranks = HashMap::default();
        ranks.insert(b"ab".to_vec(), 1);
        ranks.insert(b"cd".to_vec(), 2);

        let res = byte_pair_split(b"abcd", &ranks);
        assert_eq!(res, vec![b"ab", b"cd"]);
    }
}