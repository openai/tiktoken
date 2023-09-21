


fn roll_hash(old: i64, new: u8) -> i64 {
    const LEN: usize = 256;
    const PRIME: i64 = 31;
    const MODULUS: i64 = 1e9 as i64 + 7;
    ((old - (new as i64).pow(LEN as u32) % MODULUS + PRIME) * 256 + (new as i64)) % MODULUS
}

fn roll_hash_slice(slice: &[u8]) -> i64 {
    let mut hash = 0;
    for &byte in slice {
        hash = roll_hash(hash, byte);
    }
    hash
}
