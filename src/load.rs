use rustc_hash::FxHashMap as HashMap;
use sha2::Digest;
use sha2::Sha256;
// Import the base64 crate Engine trait anonymously so we can
// call its methods without adding to the namespace.
use base64::engine::general_purpose::STANDARD as BASE64;
use base64::engine::Engine as _;

// define the error
#[derive(Debug, Clone)]
pub enum Error {
    InvalidTiktokenBpe,
    ShasumMismatch,
}

pub fn load_tiktoken_bpe(
    tiktoken_bpe_contents: &[u8],
    shasum: &str,
) -> Result<HashMap<Vec<u8>, usize>, Error> {
    // check the shasum
    let mut hasher = Sha256::new();
    hasher.update(tiktoken_bpe_contents);
    let hash = hasher.finalize();
    let hash = hash.to_vec();
    if hash != hex::decode(shasum).map_err(|_| Error::ShasumMismatch)? {
        return Err(Error::ShasumMismatch);
    }

    let mut map = HashMap::default();
    for line in tiktoken_bpe_contents.split(|&b| b == b'\n') {
        if line.is_empty() {
            continue;
        }
        let mut parts = line.splitn(2, |&b| b == b' ');
        let next = parts.next().ok_or(Error::InvalidTiktokenBpe)?;
        let token = BASE64.decode(next).map_err(|_| Error::InvalidTiktokenBpe)?;
        let rank = parts
            .next()
            .ok_or(Error::InvalidTiktokenBpe)?
            .iter()
            .fold(0, |acc, &b| acc * 10 + (b - b'0') as usize);
        map.insert(token, rank);
    }
    map.shrink_to_fit();
    Ok(map)
}
