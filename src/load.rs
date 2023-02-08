
use rustc_hash::FxHashMap as HashMap;
use linefeed::chars::is_printable;

pub fn data_gym_to_mergeable_bpe_ranks(vocab_bpe_file: String, encoder_json_file: String) -> HashMap<Vec<u8>, usize> {
    let rank_to_intbyte = (0..256)
        .filter(|x| is_printable(x) && x != ' ')
        .collect::<Vec<u8>>();

    let mut data_gym_byte_to_byte = HashMap::default();
    for b in rank_to_intbyte.iter() {
        data_gym_byte_to_byte.insert(b, b);
    }

    let mut n = 0;
    for b in 0..256 {
        if !rank_to_intbyte.contains(b) {
            rank_to_intbyte.push(b);
            data_gym_byte_to_byte.insert(256 + n, b);
            n += 1;
        }
    }
    assert!(rank_to_intbyte.len() == 256);

    ranks
} data_gym_to_mergeable_bpe_ranks