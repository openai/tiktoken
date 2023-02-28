
use rustc_hash::FxHashMap as HashMap;
use std::{env, path::PathBuf};
use sha1::{Sha1, Digest};
use std::error::Error;
use json;

type Result<T> = std::result::Result<T, Box<dyn Error>>;

fn read_file(blobpath: &str) -> Result<Vec<u8>> {
    // TODO: support blobs?

    if !(blobpath.starts_with("http") || blobpath.starts_with("https")) {
        return Ok(std::fs::read(blobpath)?);
    }

    Ok(reqwest::blocking::get(blobpath)?.bytes()?.to_vec())
}

fn get_tiktoken_cache_dir() -> PathBuf {
     match env::var_os("TIKTOKEN_CACHE_DIR") {
        Some(v) => PathBuf::from(v),
        None => {
            match env::var_os("DATA_GYM_CACHE_DIR") {
                Some(v) => PathBuf::from(v),
                None => {
                    let mut temp_dir = env::temp_dir();
                    temp_dir.push("data-gym-cache");

                    temp_dir
                }
            }
        }
    }
}

fn sha1_as_hex(s: &str) -> String {
    let mut hasher = Sha1::new();
    hasher.update(s.as_bytes());
    let result = hasher.finalize();

    format!("{:x}", result)
}

fn read_file_cached(blobpath: &str) -> Result<Vec<u8>> {
    let mut cache_path = get_tiktoken_cache_dir();

    if !cache_path.exists() {
        std::fs::create_dir_all(&cache_path)?;
    }

    cache_path.push(sha1_as_hex(blobpath));

    println!("cache_path: {:?}", cache_path);

    if cache_path.exists() {
        let catch_path_str = cache_path.into_os_string().into_string()
            .or(Err( {
                // let cache_path_lossy_str = cache_path.to_string_lossy().to_string();
                // format!("Unable to convert path {cache_path_lossy_str}")
                format!("Unable to convert path")
            }))?;
        return read_file(&catch_path_str);
    }

    let content = read_file(blobpath)?;

    std::fs::write(cache_path, &content)?;

    Ok(content)
}

fn is_printable(u: u8) -> bool {
    // printable ascii characters according to python
    !(u <= 31 || (u >= 127 && u <= 160) || u == 173)
}

pub fn data_gym_to_mergeable_bpe_ranks(vocab_bpe_file: &str, encoder_json_file: &str) -> Result<HashMap<Vec<u8>, usize>> {
    let mut rank_to_intbyte = (0..=255)
        .filter(|x| is_printable(*x) && (*x as char) != ' ')
        .collect::<Vec<u8>>();

    let mut data_gym_byte_to_byte = rank_to_intbyte
        .iter()
        .map(|&x| (x as u32, x))
        .collect::<HashMap<u32, u8>>();

    let mut n = 0;
    for b in 0..=255 {
        if !rank_to_intbyte.contains(&b) {
            rank_to_intbyte.push(b);
            data_gym_byte_to_byte.insert(256 + n, b);
            n += 1;
        }
    }
    assert!(rank_to_intbyte.len() == 256);

    // vocab_bpe contains the merges along with associated ranks
    let cached_vocab = read_file_cached(vocab_bpe_file)?;
    let vocab_bpe_contents = std::str::from_utf8(&cached_vocab)?
        .split("\n").collect::<Vec<&str>>();

    let bpe_merges = match vocab_bpe_contents[1..(vocab_bpe_contents.len() - 1)]
        .iter()
        .map(|&s| s.split_whitespace())
        .map(|mut sp| match (sp.next(), sp.next()) {
            (Some(a), Some(b)) => Some((a, b)),
            _ => None,
        })
        .collect::<Option<Vec<(&str, &str)>>>()
    {
        Some(v) => v,
        None => return Err("Unable to parse vocab_bpe file".into()),
    };

    let decode_data_gym =
        |value: &str| value.chars().map(|c| {
            data_gym_byte_to_byte[&(c as u32)]
        } ).collect::<Vec<u8>>();

    // # add the single byte tokens
    let mut bpe_ranks =
        rank_to_intbyte
            .iter()
            .enumerate()
            .map(|(i, b)| (vec![*b], i))
            .collect::<HashMap<Vec<u8>, usize>>();

    // add the merged tokens
    let mut n = bpe_ranks.len();
    for (first, second) in bpe_merges {
        bpe_ranks.insert([decode_data_gym(first), decode_data_gym(second)].concat(), n);
        n += 1;
    }

    // check that the encoder file matches the merges file
    // this sanity check is important since tiktoken assumes that ranks are ordered the same
    // as merge priority
    let cached_encoder = read_file_cached(encoder_json_file)?;
    let encoder_json = json::parse(&std::str::from_utf8(&cached_encoder)?)?;

    let mut encoder_json_loaded = encoder_json.entries()
        .map(|(k, v)| (decode_data_gym(k), v.as_usize().unwrap()))
        .collect::<HashMap<Vec<u8>, usize>>();

    // drop these two special tokens if present, since they're not mergeable bpe tokens
    encoder_json_loaded.remove(&decode_data_gym("<|endoftext|>"));
    encoder_json_loaded.remove(&decode_data_gym("<|startoftext|>"));

    assert!(bpe_ranks == encoder_json_loaded);

    Ok(bpe_ranks)
}

pub fn load_tiktoken_bpe(tiktoken_bpe_file: &str) -> Result<HashMap<Vec<u8>, usize>> {
    use base64::{engine::general_purpose, Engine as _};

    let content = read_file_cached(tiktoken_bpe_file)?;

    Ok(std::str::from_utf8(&content)?
        .lines()
        .filter(|s| s.len() > 0)
        .map(|s| s.split_whitespace())
        .map(|mut sp| (sp.next().unwrap(), sp.next().unwrap()))
        .map(|(first, second)| (general_purpose::STANDARD.decode(&first).unwrap(), second.parse::<usize>().unwrap()))
        .collect::<HashMap<Vec<u8>, usize>>())
}

