# ⏳ tiktoken

tiktoken is a [BPE](https://en.wikipedia.org/wiki/Byte_pair_encoding) tokeniser for use with
OpenAI's models, forked from the original tiktoken library to provide NPM bindings for Node and other JS runtimes.

The open source version of `tiktoken` can be installed from NPM:

```
npm install @dqbd/tiktoken
```

> Please note there are some missing features which are present in the Python version but not in the JS version.

## Usage

Basic usage follows:

```typescript
import assert from "node:assert";
import { get_encoding, encoding_for_model } from "@dqbd/tiktoken";

const enc = get_encoding("gpt2");
assert(
  new TextDecoder().decode(enc.decode(enc.encode("hello world"))) ===
    "hello world"
);

// To get the tokeniser corresponding to a specific model in the OpenAI API:
const enc = encoding_for_model("text-davinci-003");

// Extend existing encoding with custom special tokens
const enc = encoding_for_model("gpt2", {
  "<|im_start|>": 100264,
  "<|im_end|>": 100265,
});
```

If desired, you can create a Tiktoken instance directly with custom ranks, special tokens and regex pattern:

```typescript
import { Tiktoken } from "../pkg";
import { readFileSync } from "fs";

const encoder = new Tiktoken(
  readFileSync("./ranks/gpt2.tiktoken").toString("utf-8"),
  { "<|endoftext|>": 50256, "<|im_start|>": 100264, "<|im_end|>": 100265 },
  "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+"
);
```

## Acknowledgements

- https://github.com/zurawiki/tiktoken-rs

## Tasks to do before creating an upstream PR

1. Add back the pyo3 bindings, so we can build both Python version and JS version at the same time
2. Allow loading of embeddings via an argument. This is needed to make the resulting WASM blob smaller, as it is currently inlined during build.
3. Examine the possibility of reintroduction of multithreading (not sure, if that is even needed however due to the sheer perf. difference between other JS libraries)
4. Feature parity match - adding special tokens support etc.
5. Investigate better packaging support for browsers and other runtimes.
