use _tiktoken_core::CoreBPENative;

use base64::{engine::general_purpose, Engine as _};
use fancy_regex::Regex;
use rustc_hash::FxHashMap as HashMap;
use std::collections::HashSet;
use std::result::Result;
use anyhow::Error;

use magnus::{define_module, exception, function, memoize, method, prelude::*, Error as MError, RModule, Value, RString};
use serde_magnus::deserialize;

type RbResult<T> = Result<T, MError>;

const ENDOFTEXT: &'static str = "<|endoftext|>";

const FIM_PREFIX: &'static str = "<|fim_prefix|>";

const FIM_MIDDLE: &'static str = "<|fim_middle|>";

const FIM_SUFFIX: &'static str = "<|fim_suffix|>";

const ENDOFPROMPT: &'static str = "<|endofprompt|>";

struct CoreBPEConstructor {
    encoder: HashMap<Vec<u8>, usize>,
    special_tokens: HashMap<String, usize>,
    pat_str: String,
}

impl CoreBPEConstructor {
    fn new(
        tiktoken_bfe: &str,
        special_tokens: Option<HashMap<String, usize>>,
        pat_str: &str,
    ) -> Self {
        CoreBPEConstructor {
            encoder: CoreBPEConstructor::parse_bfe(tiktoken_bfe).unwrap(),
            special_tokens: special_tokens.unwrap_or_default(),
            pat_str: String::from(pat_str),
        }
    }

    fn parse_bfe(tiktoken_bfe: &str) -> Result<HashMap<Vec<u8>, usize>, Error> {
        let mut encoder = HashMap::default();
        if tiktoken_bfe.chars().next().unwrap() == '!' {
            for line in tiktoken_bfe.lines() {
                let mut parts = line.split(' ');
                parts.next().unwrap();

                let offset: i32 = parts.next().unwrap().parse()?;
                for (pos, token) in parts.enumerate() {
                    let token = &general_purpose::STANDARD.decode(token)?;
                    encoder.insert(token.clone(), (offset as usize) + pos);
                }
            }
        } else {
            for line in tiktoken_bfe.lines() {
                let mut parts = line.split(' ');
                let token = &general_purpose::STANDARD.decode(parts.next().unwrap())?;
                let rank: usize = parts.next().unwrap().parse().unwrap();
                encoder.insert(token.clone(), rank);
            }
        }

        Ok(encoder)
    }

    fn gpt2() -> Self {
        let mut special_tokens = HashMap::default();
        special_tokens.insert(String::from(ENDOFTEXT), 50256);

        CoreBPEConstructor::new(
            include_str!("../../../ranks/gpt2.compress.tiktoken"),
            Some(special_tokens),
            "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+",
        )
    }

    fn r50k_base() -> Self {
        let mut special_tokens = HashMap::default();
        special_tokens.insert(String::from(ENDOFTEXT), 50256);

        CoreBPEConstructor::new(
            include_str!("../../../ranks/r50k_base.compress.tiktoken"),
            Some(special_tokens),
            "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+",
        )
    }

    fn p50k_base() -> Self {
        let mut special_tokens = HashMap::default();
        special_tokens.insert(String::from(ENDOFTEXT), 50256);

        CoreBPEConstructor::new(
            include_str!("../../../ranks/p50k_base.compress.tiktoken"),
            Some(special_tokens),
            "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+",
        )
    }

    fn p50k_edit() -> Self {
        let mut special_tokens = HashMap::default();
        special_tokens.insert(String::from(ENDOFTEXT), 50256);
        special_tokens.insert(String::from(FIM_PREFIX), 50281);
        special_tokens.insert(String::from(FIM_MIDDLE), 50282);
        special_tokens.insert(String::from(FIM_SUFFIX), 50283);

        CoreBPEConstructor::new(
            include_str!("../../../ranks/p50k_base.compress.tiktoken"),
            Some(special_tokens),
            "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+",
        )
    }

    fn cl100k_base() -> Self {
        let mut special_tokens = HashMap::default();
        special_tokens.insert(String::from(ENDOFTEXT), 100257);
        special_tokens.insert(String::from(FIM_PREFIX), 100258);
        special_tokens.insert(String::from(FIM_MIDDLE), 100259);
        special_tokens.insert(String::from(FIM_SUFFIX), 100260);
        special_tokens.insert(String::from(ENDOFPROMPT), 100276);

        CoreBPEConstructor::new(
            include_str!("../../../ranks/cl100k_base.compress.tiktoken"),
            Some(special_tokens),
            "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
        )
    }
}

#[magnus::wrap(class = "Tiktoken::Encoder")]
pub struct Encoder {
    name: Option<String>,
    special_tokens_set: HashSet<String>,
    bpe: CoreBPENative,
}

impl Encoder {
    fn from_encoding(
        encoding: String,
        extend_special_tokens: &Option<HashMap<String, usize>>,
    ) -> RbResult<Self> {
        let mut constructor: CoreBPEConstructor = match encoding.as_str() {
            "gpt2" => Ok(CoreBPEConstructor::gpt2()),
            "r50k_base" => Ok(CoreBPEConstructor::r50k_base()),
            "p50k_base" => Ok(CoreBPEConstructor::p50k_base()),
            "p50k_edit" => Ok(CoreBPEConstructor::p50k_edit()),
            "cl100k_base" => Ok(CoreBPEConstructor::cl100k_base()),
            &_ => Err(MError::new(exception::arg_error(), "Invalid encoding")),
        }?;

        if let Some(tokens) = extend_special_tokens {
            constructor.special_tokens.extend(tokens.clone());
        }

        Ok(Encoder {
            name: Some(String::from(encoding)),
            // TODO: can we avoid cloning here?
            special_tokens_set: constructor
                .special_tokens
                .keys()
                .map(|s| s.clone())
                .collect(),
            bpe: CoreBPENative::new(
                constructor.encoder,
                constructor.special_tokens,
                &constructor.pat_str,
            )
            .unwrap(),
        })
    }

    pub fn name(&self) -> Option<String> {
        self.name.clone()
    }

    pub fn encode(
        &self,
        text: RString,
        allowed_special: Value,
        disallowed_special: Value,
    ) -> RbResult<Vec<usize>> {
        unsafe {
            let text_str = text.as_str().unwrap();
            let allowed_tokens =
                self.validate_allowed_tokens(&text_str, &allowed_special, &disallowed_special)?;

            Ok(self
                .bpe
                ._encode_native(
                    &text_str,
                    &allowed_tokens.iter().map(AsRef::as_ref).collect(),
                    None,
                )
                .0)
        }
    }

    pub fn encode_ordinary(&self, text: RString) -> RbResult<Vec<usize>> {
        unsafe {
            let text_str = text.as_str().unwrap();
            Ok(self.bpe._encode_ordinary_native(&text_str))
        }
    }

    // TODO do we need this?
    //pub fn encode_single_token(&self, bytes: &[u8]) -> usize {
    //    self.bpe.encode_single_token(&bytes).unwrap()
    //}

    pub fn decode(&self, tokens: Vec<usize>) -> Vec<u8> {
        self.bpe._decode_native(&tokens)
    }

    // TODO do we need this?
    // pub fn decode_single_token_bytes(&self, token: usize) -> Vec<u8> {
    //     self.bpe
    //         .decode_single_token_bytes(token)
    //         .unwrap()
    //         .to_vec()
    // }

    // TODO do we need this
    // pub fn token_byte_values(&self) -> Value {
    //     JsValue::from_serde(&self.bpe.token_byte_values()).unwrap_throw()
    // }

    fn validate_allowed_tokens(
        &self,
        text: &str,
        allowed_special_param: &Value,
        disallowed_special_param: &Value,
    ) -> Result<HashSet<String>, MError> {
        // If it's a string, only 'all' is allowed. Otherwise, needs to be a list of strings.
        let allowed_special: HashSet<String> = match allowed_special_param.class().inspect().as_str() {
            "String" => {
                let allowed_special_str: String = deserialize(allowed_special_param).unwrap_or_default();
                match allowed_special_str.as_str() {
                    "all" => Ok(self.special_tokens_set.clone()),
                    _ => Err(MError::new(exception::arg_error(), "Invalid value for allowed_special")),
                }
            },
            "Array" => Ok(deserialize(allowed_special_param).unwrap_or_default()),
            _ => Err(MError::new(exception::arg_error(), "Invalid type for allowed_special")),
        }?;

        let disallowed_special: HashSet<String> = match disallowed_special_param.class().inspect().as_str() {
            "String" => {
                let disallowed_special_str: String = deserialize(disallowed_special_param).unwrap_or_default();
                match disallowed_special_str.as_str() {
                    "all" => Ok(&self.special_tokens_set - &allowed_special),
                    _ => Err(MError::new(exception::arg_error(), "Invalid value for disallowed_special")),
                }
            },
            "Array" => Ok(deserialize(disallowed_special_param).unwrap_or_default()),
            _ => Err(MError::new(exception::arg_error(), "Invalid type for disallowed_special")),
        }?;

        if !disallowed_special.is_empty() {
            if let Some(found) = Encoder::special_token_regex(&disallowed_special).find(text).unwrap() {
                let err: String = format!(
                    "The text contains a special token that is not allowed: {}",
                    found.as_str()
                );
                return Err(MError::new(exception::arg_error(), err));
            }
        }

        return Ok(allowed_special);
    }

    fn special_token_regex(tokens: &HashSet<String>) -> Regex {
        let inner = tokens
            .iter()
            .map(|token| regex::escape(token))
            .collect::<Vec<String>>()
            .join("|");

        Regex::new(&format!("({})", inner)).unwrap()
    }
}

pub fn get_encoding(encoding: String, extend_special_tokens: Value) -> RbResult<Encoder> {
    let _extend_special_tokens: Option<HashMap<String, usize>> = deserialize(&extend_special_tokens).ok();

    Encoder::from_encoding(
        encoding,
        &_extend_special_tokens
    )
}

pub fn encoding_for_model(
    model: String,
    extend_special_tokens: Value,
) -> RbResult<Encoder> {
    let encoding = match model.as_str() {
        "text-davinci-003" => Ok("p50k_base"),
        "text-davinci-002" => Ok("p50k_base"),
        "text-davinci-001" => Ok("r50k_base"),
        "text-curie-001" => Ok("r50k_base"),
        "text-babbage-001" => Ok("r50k_base"),
        "text-ada-001" => Ok("r50k_base"),
        "davinci" => Ok("r50k_base"),
        "curie" => Ok("r50k_base"),
        "babbage" => Ok("r50k_base"),
        "ada" => Ok("r50k_base"),
        "code-davinci-002" => Ok("p50k_base"),
        "code-davinci-001" => Ok("p50k_base"),
        "code-cushman-002" => Ok("p50k_base"),
        "code-cushman-001" => Ok("p50k_base"),
        "davinci-codex" => Ok("p50k_base"),
        "cushman-codex" => Ok("p50k_base"),
        "text-davinci-edit-001" => Ok("p50k_edit"),
        "code-davinci-edit-001" => Ok("p50k_edit"),
        "text-embedding-ada-002" => Ok("cl100k_base"),
        "text-similarity-davinci-001" => Ok("r50k_base"),
        "text-similarity-curie-001" => Ok("r50k_base"),
        "text-similarity-babbage-001" => Ok("r50k_base"),
        "text-similarity-ada-001" => Ok("r50k_base"),
        "text-search-davinci-doc-001" => Ok("r50k_base"),
        "text-search-curie-doc-001" => Ok("r50k_base"),
        "text-search-babbage-doc-001" => Ok("r50k_base"),
        "text-search-ada-doc-001" => Ok("r50k_base"),
        "code-search-babbage-code-001" => Ok("r50k_base"),
        "code-search-ada-code-001" => Ok("r50k_base"),
        "gpt2" => Ok("gpt2"),
        "gpt-3.5-turbo" => Ok("cl100k_base"),
        "gpt-3.5-turbo-0301" => Ok("cl100k_base"),
        "gpt-4" => Ok("cl100k_base"),
        "gpt-4-32k" => Ok("cl100k_base"),
        model => Err(MError::new(exception::arg_error(),
            format!("Invalid model: {}", model.to_string()),
        )),
    }?;

    let _extend_special_tokens: Option<HashMap<String, usize>> = deserialize(&extend_special_tokens).ok();

    Encoder::from_encoding(
        encoding.to_string(),
        &_extend_special_tokens
    )
}

fn module() -> RModule {
    *memoize!(RModule: define_module("Tiktoken").unwrap())
}

#[magnus::init]
fn init() -> RbResult<()> {
    let module = module();

    module.define_module_function("_get_encoding", function!(get_encoding, 2))?;
    module.define_module_function("_encoding_for_model", function!(encoding_for_model, 2))?;

    let class = module.define_class("Encoder", Default::default())?;
    class.define_method("name", method!(Encoder::name, 0))?;
    class.define_method("_encode", method!(Encoder::encode, 3))?;
    class.define_method("_encode_ordinary", method!(Encoder::encode_ordinary, 1))?;
    class.define_method("_decode", method!(Encoder::decode, 1))?;

    Ok(())
}