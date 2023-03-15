# ⏳ tiktoken

tiktoken is a [BPE](https://en.wikipedia.org/wiki/Byte_pair_encoding) tokeniser for use with
OpenAI's models, forked from the original tiktoken library to provide NPM bindings for Node and other JS runtimes.

The open source version of `tiktoken` can be installed from NPM:

```
npm install @dqbd/tiktoken
```

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

// don't forget to free the encoder after it is not used
enc.free();
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

## Compatibility

As this is a WASM library, there might be some issues with specific runtimes. If you encounter any issues, please open an issue.

| Runtime             | Status | Notes                                       |
| ------------------- | ------ | ------------------------------------------- |
| Node.js             | ✅     |                                             |
| Bun                 | ✅     |                                             |
| Vite                | ✅     | See [here](#vite) for notes                 |
| Next.js             | ✅     | See [here](#nextjs) for notes               |
| Vercel Edge Runtime | ✅     | See [here](#vercel-edge-runtime) for notes  |
| Cloudflare Workers  | 🚧     | See [here](#cloudflare-workers) for caveats |
| Deno                | ❌     | Currently unsupported                       |

### [Vite](#vite)

If you are using Vite, you will need to add both the `vite-plugin-wasm` and `vite-plugin-top-level-await`. Add the following to your `vite.config.js`:

```js
import wasm from "vite-plugin-wasm";
import topLevelAwait from "vite-plugin-top-level-await";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [wasm(), topLevelAwait()],
});
```

### [Next.js](#nextjs)

Both API routes and `/pages` are supported with the following `next.config.js` configuration.

```typescript
// next.config.json
const config = {
  webpack(config, { isServer, dev }) {
    config.experiments = {
      asyncWebAssembly: true,
      layers: true,
    };

    return config;
  },
};
```

Usage in pages:

```tsx
import { get_encoding } from "@dqbd/tiktoken";
import { useState } from "react";

const encoding = get_encoding("cl100k_base");

export default function Home() {
  const [input, setInput] = useState("hello world");
  const tokens = encoding.encode(input);

  return (
    <div>
      <input
        type="text"
        value={input}
        onChange={(e) => setInput(e.target.value)}
      />
      <div>{tokens.toString()}</div>
    </div>
  );
}
```

Usage in API routes:

```typescript
import { get_encoding } from "@dqbd/tiktoken";
import { NextApiRequest, NextApiResponse } from "next";

export default function handler(req: NextApiRequest, res: NextApiResponse) {
  const encoding = get_encoding("cl100k_base");
  const tokens = encoding.encode("hello world");
  encoding.free();
  return res.status(200).json({ tokens });
}
```

### [Vercel Edge Runtime](#vercel-edge-runtime)

Vercel Edge Runtime does support WASM modules by adding a `?module` suffix. Initialize the encoder with the following snippet:

```typescript
import wasm from "@dqbd/tiktoken/tiktoken_bg.wasm?module";
import { init, get_encoding } from "@dqbd/tiktoken/init";

export const config = { runtime: "edge" };

export default async function (req: Request) {
  await init((imports) => WebAssembly.instantiate(wasm, imports));

  const encoder = get_encoding("cl100k_base");
  const tokens = encoder.encode("hello world");
  encoder.free();

  return new Response(`${encoder.encode("hello world")}`);
}
```

### [Cloudflare Workers](#cloudflare-workers)

> Currently work in progress, investigating crashes and workarounds to compress ranks.

Similar to Vercel Edge Runtime, Cloudflare Workers must import the WASM binary file manually. However, users need to point directly at the WASM binary, including `node_modules` prefix in some cases.

Add the following rule to the `wrangler.toml` to upload WASM during build:

```toml
[[rules]]
globs = ["**/*.wasm"]
type = "CompiledWasm"
```

Initialize the encoder with the following snippet:

```javascript
import wasm from "./node_modules/@dqbd/tiktoken/tiktoken_bg.wasm";
import { get_encoding, init } from "@dqbd/tiktoken/init";

export default {
  async fetch() {
    await init((imports) => WebAssembly.instantiate(wasm, imports));
    const encoder = get_encoder("cl100k_base");
    const tokens = encoder.encode("hello world");
    encoder.free();
    return new Response(`${tokens}`);
  },
};
```

## Acknowledgements

- https://github.com/zurawiki/tiktoken-rs
