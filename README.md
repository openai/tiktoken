# ⏳ tiktoken-rs

tiktoken-rs is based on [openai/tiktoken](https://github.com/openai/tiktoken), rewritten to work as a Rust crate. It is unstable, experimental, and only half-implemented at the moment, but usable enough to count tokens in some cases.

```rust
let enc = tiktoken::EncodingFactory::cl100k_base().unwrap();
let tokens = enc.encode(
    "hello world",
    &SpecialTokenHandling {
        default: SpecialTokenAction::Forbidden,
        ..Default::default()
    }
).unwrap()
println!("Number of tokens: {}", tokens.len());
```

## Which tokenizer to use?

GPT-3 (`text-davinci-002` and earlier) is `r50k_base`.

Codex and GPT-3.5 (`code-davinci-002` and `text-davinci-003`) is `p50k_base`.

Embeddings (`text-embedding-ada-002`) is `cl100k_base`.
