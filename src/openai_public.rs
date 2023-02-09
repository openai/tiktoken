
use rustc_hash::FxHashMap as HashMap;
use std::error::Error;

#[path = "load.rs"]
mod load;

type Result<T> = std::result::Result<T, Box<dyn Error>>;

lazy_static! {
    static ref REGISTRY: HashMap<String, EncodingLazy> = [
            EncodingLazy::new(
                "gpt2".into(),
                50257,
                r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+".into(),
                [
                    ("<|endoftext|>".into(), 50256),
                ].into_iter().collect(),
                EncoderLoadingStrategy::DataGym(
                    DataGymDef {
                        vocab_bpe_file: "https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/vocab.bpe".to_string(),
                        encoder_json_file: "https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/encoder.json".to_string()
                    }
                ))
            ]
            .into_iter()
            .map(|enc| (enc.name.clone(), enc))
            .collect::<HashMap<String, EncodingLazy>>();
}
struct DataGymDef {
    vocab_bpe_file: String,
    encoder_json_file: String,
}

enum EncoderLoadingStrategy {
    BPE(String),
    DataGym(DataGymDef),
}

struct EncodingLazy {
    name: String,
    explicit_n_vocab: usize,
    pat_str: String,
    special_tokens: HashMap<String, usize>,
    mergeable_ranks: Option<HashMap<Vec<u8>, usize>>,
    loading_strategy: EncoderLoadingStrategy,
}

impl EncodingLazy {
    fn new(name: String,
           explicit_n_vocab: usize,
           pat_str: String,
           special_tokens: HashMap<String, usize>,
           loading_strategy: EncoderLoadingStrategy) -> Self {
        EncodingLazy {
            name,
            explicit_n_vocab,
            pat_str,
            special_tokens,
            mergeable_ranks: None,
            loading_strategy
        }
    }

    fn get(&mut self) -> Result<&HashMap<Vec<u8>, usize>> {
        if self.mergeable_ranks.is_none() {
            self.mergeable_ranks = Some(match &self.loading_strategy {
                EncoderLoadingStrategy::BPE(path) => Self::load_bpe(&path)?,
                EncoderLoadingStrategy::DataGym(def) => Self::load_data_gym(&def)?,
            })
        }

        Ok(self.mergeable_ranks.as_ref().expect("mergeable_ranks should be loaded by now"))
    }

    fn load_bpe(path: &str) -> Result<HashMap<Vec<u8>, usize>> {
        load::load_tiktoken_bpe(path)
    }

    fn load_data_gym(def: &DataGymDef) -> Result<HashMap<Vec<u8>, usize>> {
        load::data_gym_to_mergeable_bpe_ranks(&def.vocab_bpe_file, &def.encoder_json_file)
    }
}



