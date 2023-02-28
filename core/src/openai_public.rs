
use rustc_hash::FxHashMap as HashMap;
use std::error::Error;
use std::sync::RwLock;
use json;

#[path = "load.rs"]
mod load;

type Result<T> = std::result::Result<T, Box<dyn Error>>;

lazy_static! {
    pub static ref REGISTRY: HashMap<String, EncodingLazy> = {
        json::parse(include_str!("../../tiktoken/registry.json"))
            .expect("Failed to parse internal JSON")
            .entries()
            .map(|(key, value)| {
                let loading_strategy = if value.has_key("data_gym_to_mergeable_bpe_ranks") {
                    EncoderLoadingStrategy::DataGym(
                        DataGymDef {
                            vocab_bpe_file: value["data_gym_to_mergeable_bpe_ranks"]["vocab_bpe_file"].as_str().expect("error").into(),
                            encoder_json_file: value["data_gym_to_mergeable_bpe_ranks"]["encoder_json_file"].as_str().expect("error").into()
                        })
                }
                else if value.has_key("load_tiktoken_bpe") {
                    EncoderLoadingStrategy::BPE(value["load_tiktoken_bpe"].as_str().expect("fail").into())
                }
                else {
                    panic!("Invalid encoding");
                };

                EncodingLazy::new(
                    key.into(),
                    value["explicit_n_vocab"].as_usize(),
                    value["pat_str"].as_str().expect("foo").into(),
                    value["special_tokens"].entries()
                        .map(|(key, value)| (key.into(), value.as_usize().expect("foo")))
                        .collect::<HashMap<String, usize>>(),
                    loading_strategy
                )
            })
            
            .map(|enc| (enc.name.clone(), enc))
            .collect::<HashMap<String, EncodingLazy>>()
        };

        pub static ref MODEL_TO_ENCODING: HashMap<String, String> = 
            json::parse(include_str!("../../tiktoken/model_to_encoding.json"))
                .expect("Failed to parse internal JSON")
                .entries()
                .map(|(k, v)| (k.into(), v.as_str().expect("foo").into()))
                .collect::<HashMap<String, String>>();
}

#[derive(Clone, PartialEq, Eq, Hash)]
struct DataGymDef {
    vocab_bpe_file: String,
    encoder_json_file: String,
}

#[derive(Clone, PartialEq, Eq, Hash)]
enum EncoderLoadingStrategy {
    BPE(String),
    DataGym(DataGymDef),
}

pub struct EncodingLazy {
    name: String,
    explicit_n_vocab: Option<usize>,
    pub pat_str: String,
    pub special_tokens: HashMap<String, usize>,
    mergeable_ranks: RwLock<Option<HashMap<Vec<u8>, usize>>>,
    loading_strategy: EncoderLoadingStrategy,
}

fn load_bpe(path: &str) -> Result<HashMap<Vec<u8>, usize>> {
    load::load_tiktoken_bpe(path)
}

fn load_data_gym(def: &DataGymDef) -> Result<HashMap<Vec<u8>, usize>> {
    load::data_gym_to_mergeable_bpe_ranks(&def.vocab_bpe_file, &def.encoder_json_file)
}

// #[memoize]
fn load_mergeable_ranks(loading_strategy: &EncoderLoadingStrategy) -> Result<HashMap<Vec<u8>, usize>>
{
    match loading_strategy {
            EncoderLoadingStrategy::BPE(path) => load_bpe(&path),
            EncoderLoadingStrategy::DataGym(def) => load_data_gym(&def),
        }
}

impl EncodingLazy {
    fn new(name: String,
           explicit_n_vocab: Option<usize>,
           pat_str: String,
           special_tokens: HashMap<String, usize>,
           loading_strategy: EncoderLoadingStrategy) -> Self {
        EncodingLazy {
            name,
            explicit_n_vocab,
            pat_str,
            special_tokens,
            mergeable_ranks: RwLock::new(None),
            loading_strategy
        }
    }

    pub fn get(&self) -> Result<HashMap<Vec<u8>, usize>> {
        {
            let read = self.mergeable_ranks.read().unwrap();
            if read.is_some() {
                return Ok(read.as_ref().unwrap().clone());
            }
        }

        let mut write = self.mergeable_ranks.write().unwrap();
        *write = Some(load_mergeable_ranks(&self.loading_strategy)?);

        Ok(write.as_ref().unwrap().clone())
    }
}



