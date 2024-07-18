use _tiktoken_core::CoreBPENative;
use anyhow::Error;
use base64::{engine::general_purpose, Engine as _};
use fancy_regex::Regex;
use gloo_utils::format::JsValueSerdeExt;
use rustc_hash::FxHashMap as HashMap;
use std::collections::HashSet;
use std::result::Result;
use wasm_bindgen::prelude::*;

#[cfg(feature = "inline")]
const ENDOFTEXT: &'static str = "<|endoftext|>";

#[cfg(feature = "inline")]
const FIM_PREFIX: &'static str = "<|fim_prefix|>";

#[cfg(feature = "inline")]
const FIM_MIDDLE: &'static str = "<|fim_middle|>";

#[cfg(feature = "inline")]
const FIM_SUFFIX: &'static str = "<|fim_suffix|>";

#[cfg(feature = "inline")]
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

    #[cfg(feature = "inline")]
    fn gpt2() -> Self {
        let mut special_tokens = HashMap::default();
        special_tokens.insert(String::from(ENDOFTEXT), 50256);

        CoreBPEConstructor::new(
            include_str!("./ranks/gpt2.compress.tiktoken"),
            Some(special_tokens),
            include_str!("./ranks/gpt2.regex.tiktoken"),
        )
    }

    #[cfg(feature = "inline")]
    fn r50k_base() -> Self {
        let mut special_tokens = HashMap::default();
        special_tokens.insert(String::from(ENDOFTEXT), 50256);

        CoreBPEConstructor::new(
            include_str!("./ranks/r50k_base.compress.tiktoken"),
            Some(special_tokens),
            include_str!("./ranks/r50k_base.regex.tiktoken"),
        )
    }

    #[cfg(feature = "inline")]
    fn p50k_base() -> Self {
        let mut special_tokens = HashMap::default();
        special_tokens.insert(String::from(ENDOFTEXT), 50256);

        CoreBPEConstructor::new(
            include_str!("./ranks/p50k_base.compress.tiktoken"),
            Some(special_tokens),
            include_str!("./ranks/p50k_base.regex.tiktoken"),
        )
    }

    #[cfg(feature = "inline")]
    fn p50k_edit() -> Self {
        let mut special_tokens = HashMap::default();
        special_tokens.insert(String::from(ENDOFTEXT), 50256);
        special_tokens.insert(String::from(FIM_PREFIX), 50281);
        special_tokens.insert(String::from(FIM_MIDDLE), 50282);
        special_tokens.insert(String::from(FIM_SUFFIX), 50283);

        CoreBPEConstructor::new(
            include_str!("./ranks/p50k_base.compress.tiktoken"),
            Some(special_tokens),
            include_str!("./ranks/p50k_base.regex.tiktoken"),
        )
    }

    #[cfg(feature = "inline")]
    fn cl100k_base() -> Self {
        let mut special_tokens = HashMap::default();
        special_tokens.insert(String::from(ENDOFTEXT), 100257);
        special_tokens.insert(String::from(FIM_PREFIX), 100258);
        special_tokens.insert(String::from(FIM_MIDDLE), 100259);
        special_tokens.insert(String::from(FIM_SUFFIX), 100260);
        special_tokens.insert(String::from(ENDOFPROMPT), 100276);

        CoreBPEConstructor::new(
            include_str!("./ranks/cl100k_base.compress.tiktoken"),
            Some(special_tokens),
            include_str!("./ranks/cl100k_base.regex.tiktoken"),
        )
    }

    #[cfg(feature = "inline")]
    fn o200k_base() -> Self {
        let mut special_tokens = HashMap::default();
        special_tokens.insert(String::from(ENDOFTEXT), 199999);
        special_tokens.insert(String::from(ENDOFPROMPT), 200018);

        CoreBPEConstructor::new(
            include_str!("./ranks/o200k_base.compress.tiktoken"),
            Some(special_tokens),
            include_str!("./ranks/o200k_base.regex.tiktoken"),
        )
    }
}

#[wasm_bindgen]
pub struct Tiktoken {
    name: Option<String>,
    special_tokens_set: HashSet<String>,
    bpe: CoreBPENative,
}

#[wasm_bindgen]
impl Tiktoken {
    #[wasm_bindgen(constructor)]
    pub fn new(tiktoken_bfe: &str, special_tokens: JsValue, pat_str: &str) -> Self {
        let constructor = CoreBPEConstructor::new(
            tiktoken_bfe,
            special_tokens.into_serde::<HashMap<String, usize>>().ok(),
            pat_str,
        );

        Tiktoken {
            name: None,
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
        }
    }

    #[cfg(feature = "inline")]
    fn with_encoding(
        encoding: &str,
        extend_special_tokens: &Option<HashMap<String, usize>>,
    ) -> Result<Self, JsError> {
        let mut constructor: CoreBPEConstructor = match encoding {
            "gpt2" => Ok(CoreBPEConstructor::gpt2()),
            "r50k_base" => Ok(CoreBPEConstructor::r50k_base()),
            "p50k_base" => Ok(CoreBPEConstructor::p50k_base()),
            "p50k_edit" => Ok(CoreBPEConstructor::p50k_edit()),
            "cl100k_base" => Ok(CoreBPEConstructor::cl100k_base()),
            "o200k_base" => Ok(CoreBPEConstructor::o200k_base()),
            &_ => Err(JsError::new("Invalid encoding")),
        }?;

        if let Some(tokens) = extend_special_tokens {
            constructor.special_tokens.extend(tokens.clone());
        }

        Ok(Tiktoken {
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

    #[wasm_bindgen(getter)]
    pub fn name(&self) -> Option<String> {
        self.name.clone()
    }

    pub fn encode(
        &self,
        text: &str,
        allowed_special: JsValue,
        disallowed_special: JsValue,
    ) -> Result<Vec<usize>, JsError> {
        let allowed_tokens =
            self.validate_allowed_tokens(text, &allowed_special, &disallowed_special)?;

        Ok(self
            .bpe
            ._encode_native(
                &text,
                &allowed_tokens.iter().map(AsRef::as_ref).collect(),
                None,
            )
            .0)
    }

    pub fn encode_ordinary(&self, text: &str) -> Vec<usize> {
        self.bpe._encode_ordinary_native(&text)
    }

    pub fn encode_with_unstable(
        &self,
        text: &str,
        allowed_special: JsValue,
        disallowed_special: JsValue,
    ) -> Result<JsValue, JsError> {
        let allowed_tokens =
            self.validate_allowed_tokens(text, &allowed_special, &disallowed_special)?;

        JsValue::from_serde(
            &self.bpe._encode_unstable_native(
                &text,
                &allowed_tokens.iter().map(AsRef::as_ref).collect(),
            ),
        )
        .map_err(|e| {
            JsError::new(&format!(
                "Failed to serialize encode_with_unstable result: {}",
                e
            ))
        })
    }

    pub fn encode_single_token(&self, bytes: &[u8]) -> usize {
        self.bpe.encode_single_token(&bytes).unwrap_throw()
    }

    pub fn decode(&self, tokens: Vec<usize>) -> Vec<u8> {
        self.bpe._decode_native(&tokens)
    }

    pub fn decode_single_token_bytes(&self, token: usize) -> Vec<u8> {
        self.bpe
            .decode_single_token_bytes(token)
            .unwrap_throw()
            .to_vec()
    }

    pub fn token_byte_values(&self) -> JsValue {
        JsValue::from_serde(&self.bpe.token_byte_values()).unwrap_throw()
    }

    fn validate_allowed_tokens(
        &self,
        text: &str,
        allowed_special_param: &JsValue,
        disallowed_special_param: &JsValue,
    ) -> Result<HashSet<String>, JsError> {
        let allowed_special: HashSet<String> = match allowed_special_param.as_string() {
            Some(value) => match value.as_str() {
                "all" => Ok(self.special_tokens_set.clone()),
                _ => Err(JsError::new("Invalid value for allowed_special")),
            },
            _ => Ok(JsValue::into_serde(&allowed_special_param).unwrap_or_default()),
        }?;

        let disallowed_special = JsValue::into_serde::<HashSet<String>>(&disallowed_special_param)
            .or_else(|_| {
                match disallowed_special_param
                    .as_string()
                    .unwrap_or(String::from("all"))
                    .as_str()
                {
                    "all" => Ok(&self.special_tokens_set - &allowed_special),
                    _ => Err(JsError::new("Invalid value for disallowed_special")),
                }
            })?;

        if !disallowed_special.is_empty() {
            if let Some(found) = Tiktoken::special_token_regex(&disallowed_special).find(text)? {
                return Err(JsError::new(&format!(
                    "The text contains a special token that is not allowed: {}",
                    found.as_str()
                )));
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

        Regex::new(&format!("({})", inner)).unwrap_throw()
    }
}

#[cfg(feature = "inline")]
#[wasm_bindgen(typescript_custom_section)]
const _: &'static str = r#"
export type TiktokenEncoding = "gpt2" | "r50k_base" | "p50k_base" | "p50k_edit" | "cl100k_base" | "o200k_base";

/**
 * @param {TiktokenEncoding} encoding
 * @param {Record<string, number>} [extend_special_tokens]
 * @returns {Tiktoken}
 */
export function get_encoding(encoding: TiktokenEncoding, extend_special_tokens?: Record<string, number>): Tiktoken;
"#;

#[cfg(feature = "inline")]
#[wasm_bindgen(skip_typescript)]
pub fn get_encoding(encoding: &str, extend_special_tokens: JsValue) -> Result<Tiktoken, JsError> {
    Tiktoken::with_encoding(
        encoding,
        &extend_special_tokens
            .into_serde::<HashMap<String, usize>>()
            .ok(),
    )
}

#[cfg(feature = "inline")]
#[wasm_bindgen(typescript_custom_section)]
const _: &'static str = r#"
export type TiktokenModel =
    | "davinci-002"
    | "babbage-002"
    | "text-davinci-003"
    | "text-davinci-002"
    | "text-davinci-001"
    | "text-curie-001"
    | "text-babbage-001"
    | "text-ada-001"
    | "davinci"
    | "curie"
    | "babbage"
    | "ada"
    | "code-davinci-002"
    | "code-davinci-001"
    | "code-cushman-002"
    | "code-cushman-001"
    | "davinci-codex"
    | "cushman-codex"
    | "text-davinci-edit-001"
    | "code-davinci-edit-001"
    | "text-embedding-ada-002"
    | "text-similarity-davinci-001"
    | "text-similarity-curie-001"
    | "text-similarity-babbage-001"
    | "text-similarity-ada-001"
    | "text-search-davinci-doc-001"
    | "text-search-curie-doc-001"
    | "text-search-babbage-doc-001"
    | "text-search-ada-doc-001"
    | "code-search-babbage-code-001"
    | "code-search-ada-code-001"
    | "gpt2"
    | "gpt-3.5-turbo"
    | "gpt-35-turbo"
    | "gpt-3.5-turbo-0301"
    | "gpt-3.5-turbo-0613"
    | "gpt-3.5-turbo-1106"
    | "gpt-3.5-turbo-0125"
    | "gpt-3.5-turbo-16k"
    | "gpt-3.5-turbo-16k-0613"
    | "gpt-3.5-turbo-instruct"
    | "gpt-3.5-turbo-instruct-0914"
    | "gpt-4"
    | "gpt-4-0314"
    | "gpt-4-0613"
    | "gpt-4-32k"
    | "gpt-4-32k-0314"
    | "gpt-4-32k-0613"
    | "gpt-4-turbo"
    | "gpt-4-turbo-2024-04-09"
    | "gpt-4-turbo-preview"
    | "gpt-4-1106-preview"
    | "gpt-4-0125-preview"
    | "gpt-4-vision-preview"
    | "gpt-4o"
    | "gpt-4o-2024-05-13"
    | "gpt-4o-mini"

/**
 * @param {TiktokenModel} encoding
 * @param {Record<string, number>} [extend_special_tokens]
 * @returns {Tiktoken}
 */
export function encoding_for_model(model: TiktokenModel, extend_special_tokens?: Record<string, number>): Tiktoken;
"#;

#[cfg(feature = "inline")]
#[wasm_bindgen(skip_typescript)]
pub fn encoding_for_model(
    model: &str,
    extend_special_tokens: JsValue,
) -> Result<Tiktoken, JsError> {
    let encoding = match model {
        "davinci" => Ok("p50k_base"),
        "text-davinci-003" => Ok("p50k_base"),
        "text-davinci-002" => Ok("p50k_base"),
        "text-davinci-001" => Ok("r50k_base"),
        "text-curie-001" => Ok("r50k_base"),
        "text-babbage-001" => Ok("r50k_base"),
        "text-ada-001" => Ok("r50k_base"),
        "davinci" => Ok("r50k_base"),
        "curie" => Ok("r50k_base"),
        "babbage" => Ok("r50k_base"),
        "babbage-002" => Ok("r50k_base"),
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
        "gpt-3.5-turbo-0613" => Ok("cl100k_base"),
        "gpt-3.5-turbo-16k" => Ok("cl100k_base"),
        "gpt-3.5-turbo-16k-0613" => Ok("cl100k_base"),
        "gpt-3.5-turbo-instruct" => Ok("clk100k_base"),
        "gpt-3.5-turbo-instruct-0914" => Ok("cl100k_base"),
        "gpt-4" => Ok("cl100k_base"),
        "gpt-4-0314" => Ok("cl100k_base"),
        "gpt-4-0613" => Ok("cl100k_base"),
        "gpt-4-32k" => Ok("cl100k_base"),
        "gpt-4-32k-0314" => Ok("cl100k_base"),
        "gpt-4-32k-0613" => Ok("cl100k_base"),
        "gpt-3.5-turbo-1106" => Ok("cl100k_base"),
        "gpt-35-turbo" => Ok("cl100k_base"),
        "gpt-4-1106-preview" => Ok("cl100k_base"),
        "gpt-4-vision-preview" => Ok("cl100k_base"),
        "gpt-3.5-turbo-0125" => Ok("cl100k_base"),
        "gpt-4-turbo" => Ok("cl100k_base"),
        "gpt-4-turbo-2024-04-09" => Ok("cl100k_base"),
        "gpt-4-turbo-preview" => Ok("cl100k_base"),
        "gpt-4-0125-preview" => Ok("cl100k_base"),
        "gpt-4o" => Ok("o200k_base"),
        "gpt-4o-2024-05-13" => Ok("o200k_base"),
        "gpt-4o-mini" => Ok("o200k_base"),
        model => Err(JsError::new(
            format!("Invalid model: {}", model.to_string()).as_str(),
        )),
    }?;

    Tiktoken::with_encoding(
        encoding,
        &extend_special_tokens
            .into_serde::<HashMap<String, usize>>()
            .ok(),
    )
}
