
use rustc_hash::FxHashMap as HashMap;
use std::error::Error;

#[path = "load.rs"]
mod load;

type Result<T> = std::result::Result<T, Box<dyn Error>>;

lazy_static! {
    pub static ref REGISTRY: HashMap<String, EncodingLazy> = [
        // TODO
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

    pub static ref MODEL_TO_ENCODING: HashMap<String, String> = [
        // text
        ("text-davinci-003", "p50k_base"),
        ("text-davinci-002", "p50k_base"),
        ("text-davinci-001", "r50k_base"),
        ("text-curie-001", "r50k_base"),
        ("text-babbage-001", "r50k_base"),
        ("text-ada-001", "r50k_base"),
        ("davinci", "r50k_base"),
        ("curie", "r50k_base"),
        ("babbage", "r50k_base"),
        ("ada", "r50k_base"),
        // code
        ("code-davinci-002", "p50k_base"),
        ("code-davinci-001", "p50k_base"),
        ("code-cushman-002", "p50k_base"),
        ("code-cushman-001", "p50k_base"),
        ("davinci-codex", "p50k_base"),
        ("cushman-codex", "p50k_base"),
        // edit
        ("text-davinci-edit-001", "p50k_edit"),
        ("code-davinci-edit-001", "p50k_edit"),
        // embeddings
        ("text-embedding-ada-002", "cl100k_base"),
        // old embeddings
        ("text-similarity-davinci-001", "r50k_base"),
        ("text-similarity-curie-001", "r50k_base"),
        ("text-similarity-babbage-001", "r50k_base"),
        ("text-similarity-ada-001", "r50k_base"),
        ("text-search-davinci-doc-001", "r50k_base"),
        ("text-search-curie-doc-001", "r50k_base"),
        ("text-search-babbage-doc-001", "r50k_base"),
        ("text-search-ada-doc-001", "r50k_base"),
        ("code-search-babbage-code-001", "r50k_base"),
        ("code-search-ada-code-001", "r50k_base"),
        // open source
        ("gpt2", "gpt2"),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), v.to_string()))
    .collect::<HashMap<String, String>>();
}

struct DataGymDef {
    vocab_bpe_file: String,
    encoder_json_file: String,
}

enum EncoderLoadingStrategy {
    BPE(String),
    DataGym(DataGymDef),
}

pub struct EncodingLazy {
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



