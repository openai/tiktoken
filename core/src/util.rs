use rustc_hash::FxHashMap as HashMap;


pub fn _byte_pair_merge(piece: &[u8], ranks: &HashMap<Vec<u8>, usize>) -> Vec<std::ops::Range<usize>> {
    let mut parts: Vec<_> = (0..piece.len()).map(|i| i..i + 1).collect();

    // If you have n parts and m merges, this does O(mn) work
    // We could do something with a heap and do O(m log n) work

    // Note that we hash bytes, not token pairs. As long as we train BPE the way we
    // currently do, this is equivalent. An easy way to break this would be to decouple
    // merge priority from token index or to prevent specific token merges.
    loop {
        if parts.len() == 1 {
            break;
        }
        let mut min_rank: Option<(usize, usize)> = None;
        for i in 0..parts.len() - 1 {
            let rank = if let Some(r) = ranks.get(&piece[parts[i].start..parts[i + 1].end]) {
                *r
            } else {
                continue;
            };
            if min_rank.is_none() || rank < min_rank.unwrap().0 {
                min_rank = Some((rank, i));
            }
        }
        if let Some((_, i)) = min_rank {
            parts[i] = parts[i].start..parts[i + 1].end;
            parts.remove(i + 1);
        } else {
            break;
        }
    }
    parts
}

pub fn byte_pair_encode(piece: &[u8], ranks: &HashMap<Vec<u8>, usize>) -> Vec<usize> {
    if piece.len() == 1 {
        return vec![ranks[piece]];
    }
    _byte_pair_merge(piece, ranks)
        .iter()
        .map(|p| ranks[&piece[p.start..p.end]])
        .collect()
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