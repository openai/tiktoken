# ⏳ tiktoken

tiktoken is a [BPE](https://en.wikipedia.org/wiki/Byte_pair_encoding) tokeniser for use with
OpenAI's models, forked from the original tiktoken library to provide JS/WASM bindings for NodeJS and other JS runtimes.

This repository contains the following packages:

- `tiktoken` (formally hosted at `@dqbd/tiktoken`): WASM bindings for the original Python library, providing full 1-to-1 feature parity.
- `js-tiktoken`: Pure JavaScript port of the original library with the core functionality, suitable for environments where WASM is not well supported or not desired (such as edge runtimes). 

Documentation for `js-tiktoken` can be found in [here](https://github.com/dqbd/tiktoken/blob/main/js/README.md). Documentation for the `tiktoken` can be found here below.

The WASM version of `tiktoken` can be installed from NPM:

```
npm install tiktoken
```

## Usage

Basic usage follows, which includes all the OpenAI encoders and ranks:

```typescript
import assert from "node:assert";
import { get_encoding, encoding_for_model } from "tiktoken";

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

In constrained environments (eg. Edge Runtime, Cloudflare Workers), where you don't want to load all the encoders at once, you can use the lightweight WASM binary via `tiktoken/lite`.

```typescript
const { Tiktoken } = require("tiktoken/lite");
const cl100k_base = require("tiktoken/encoders/cl100k_base.json");

const encoding = new Tiktoken(
  cl100k_base.bpe_ranks,
  cl100k_base.special_tokens,
  cl100k_base.pat_str
);
const tokens = encoding.encode("hello world");
encoding.free();
```

If you want to fetch the latest ranks, use the `load` function:

```typescript
const { Tiktoken } = require("tiktoken/lite");
const { load } = require("tiktoken/load");
const registry = require("tiktoken/registry.json");
const models = require("tiktoken/model_to_encoding.json");

async function main() {
  const model = await load(registry[models["gpt-3.5-turbo"]]);
  const encoder = new Tiktoken(
    model.bpe_ranks,
    model.special_tokens,
    model.pat_str
  );
  const tokens = encoder.encode("hello world");
  encoder.free();
}

main();
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

Finally, you can a custom `init` function to override the WASM initialization logic for non-Node environments. This is useful if you are using a bundler that does not support WASM ESM integration.

```typescript
import { get_encoding, init } from "tiktoken/init";

async function main() {
  const wasm = "..."; // fetch the WASM binary somehow
  await init((imports) => WebAssembly.instantiate(wasm, imports));

  const encoding = get_encoding("cl100k_base");
  const tokens = encoding.encode("hello world");
  encoding.free();
}

main();
```

## Compatibility

As this is a WASM library, there might be some issues with specific runtimes. If you encounter any issues, please open an issue.

| Runtime                      | Status | Notes                                                                                      |
| ---------------------------- | ------ | ------------------------------------------------------------------------------------------ |
| Node.js                      | ✅     |                                                                                            |
| Bun                          | ✅     |                                                                                            |
| Vite                         | ✅     | See [here](#vite) for notes                                                                |
| Next.js                      | ✅     | See [here](#nextjs) for notes                                                              |
| Create React App (via Craco) | ✅     | See [here](#create-react-app) for notes                                                    |
| Vercel Edge Runtime          | ✅     | See [here](#vercel-edge-runtime) for notes                                                 |
| Cloudflare Workers           | ✅     | See [here](#cloudflare-workers) for notes                                                  |
| Electron                     | ✅     | See [here](#electron) for notes                                                            |
| Deno                         | ❌     | Currently unsupported (see [dqbd/tiktoken#22](https://github.com/dqbd/tiktoken/issues/22)) |
| Svelte + Cloudflare Workers  | ❌     | Currently unsupported (see [dqbd/tiktoken#37](https://github.com/dqbd/tiktoken/issues/37)) |

For unsupported runtimes, consider using [`js-tiktoken`](https://www.npmjs.com/package/js-tiktoken), which is a pure JS implementation of the tokeniser.

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
import { get_encoding } from "tiktoken";
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
import { get_encoding } from "tiktoken";
import { NextApiRequest, NextApiResponse } from "next";

export default function handler(req: NextApiRequest, res: NextApiResponse) {
  const encoding = get_encoding("cl100k_base");
  const tokens = encoding.encode("hello world");
  encoding.free();
  return res.status(200).json({ tokens });
}
```

### [Create React App](#create-react-app)

By default, the Webpack configugration found in Create React App does not support WASM ESM modules. To add support, please do the following:

1. Swap `react-scripts` with `craco`, using the guide found here: https://craco.js.org/docs/getting-started/.
2. Add the following to `craco.config.js`:

```js
module.exports = {
  webpack: {
    configure: (config) => {
      config.experiments = {
        asyncWebAssembly: true,
        layers: true,
      };

      // turn off static file serving of WASM files
      // we need to let Webpack handle WASM import
      config.module.rules
        .find((i) => "oneOf" in i)
        .oneOf.find((i) => i.type === "asset/resource")
        .exclude.push(/\.wasm$/);

      return config;
    },
  },
};
```

### [Vercel Edge Runtime](#vercel-edge-runtime)

Vercel Edge Runtime does support WASM modules by adding a `?module` suffix. Initialize the encoder with the following snippet:

```typescript
// @ts-expect-error
import wasm from "tiktoken/lite/tiktoken_bg.wasm?module";
import model from "tiktoken/encoders/cl100k_base.json";
import { init, Tiktoken } from "tiktoken/lite/init";

export const config = { runtime: "edge" };

export default async function (req: Request) {
  await init((imports) => WebAssembly.instantiate(wasm, imports));

  const encoding = new Tiktoken(
    model.bpe_ranks,
    model.special_tokens,
    model.pat_str
  );

  const tokens = encoding.encode("hello world");
  encoding.free();

  return new Response(`${tokens}`);
}
```

### [Cloudflare Workers](#cloudflare-workers)

Similar to Vercel Edge Runtime, Cloudflare Workers must import the WASM binary file manually and use the `tiktoken/lite` version to fit the 1 MB limit. However, users need to point directly at the WASM binary via a relative path (including `./node_modules/`).

Add the following rule to the `wrangler.toml` to upload WASM during build:

```toml
[[rules]]
globs = ["**/*.wasm"]
type = "CompiledWasm"
```

Initialize the encoder with the following snippet:

```javascript
import { init, Tiktoken } from "tiktoken/lite/init";
import wasm from "./node_modules/tiktoken/lite/tiktoken_bg.wasm";
import model from "tiktoken/encoders/cl100k_base.json";

export default {
  async fetch() {
    await init((imports) => WebAssembly.instantiate(wasm, imports));
    const encoder = new Tiktoken(
      model.bpe_ranks,
      model.special_tokens,
      model.pat_str
    );
    const tokens = encoder.encode("test");
    encoder.free();
    return new Response(`${tokens}`);
  },
};
```

### [Electron](#electron)

To use tiktoken in your Electron main process, you need to make sure the WASM binary gets copied into your application package.

Assuming a setup with [Electron Forge](https://www.electronforge.io) and [`@electron-forge/plugin-webpack`](https://www.npmjs.com/package/@electron-forge/plugin-webpack), add the following to your `webpack.main.config.js`:

```javascript
const CopyPlugin = require("copy-webpack-plugin");

module.exports = {
  // ...
  plugins: [
    new CopyPlugin({
      patterns: [
        { from: "./node_modules/tiktoken/tiktoken_bg.wasm" },
      ],
    }),
  ],
};
```

## Development

To build the `tiktoken` library, make sure to have:
- Rust and [`wasm-pack`](https://github.com/rustwasm/wasm-pack) installed.
- Node.js 18+ is required to build the JS bindings and fetch the latest encoder ranks via `fetch`.

Install all the dev-dependencies with `yarn install` and build both WASM binary and JS bindings with `yarn build`.

## Acknowledgements

- https://github.com/zurawiki/tiktoken-rs
