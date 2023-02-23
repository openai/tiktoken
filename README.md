# ⏳ tiktoken

tiktoken is a [BPE](https://en.wikipedia.org/wiki/Byte_pair_encoding) tokeniser for use with
OpenAI's models, forked from the original tiktoken library to provide NPM bindings for Node and other JS runtimes.

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
```

The open source version of `tiktoken` can be installed from NPM:

```
npm install @dqbd/tiktoken
```

Please note there are some missing features which are present in the Python version but not in the JS version. 

## Acknowledgements

- https://github.com/zurawiki/tiktoken-rs

## Tasks to do before creating an upstream PR

1. Add back the pyo3 bindings, so we can build both Python version and JS version at the same time
2. Allow loading of embeddings via an argument. This is needed to make the resulting WASM blob smaller, as it is currently inlined during build.
3. Examine the possibility of reintroduction of multithreading (not sure, if that is even needed however due to the sheer perf. difference between other JS libraries)
4. Feature parity match - adding special tokens support etc.
5. Investigate better packaging support for browsers and other runtimes. 