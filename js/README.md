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

| Runtime             | Status | Notes                           |
| ------------------- | ------ | ------------------------------- |
| Node.js             | ✅     |                                 |
| Bun                 | ✅     |                                 |
| Vite                | ✅     | See [here](#vite) for notes     |
| Next.js             | ✅ 🚧  | See [here](#nextjs) for caveats |
| Vercel Edge Runtime | 🚧     | Work in progress                |
| Cloudflare Workers  | 🚧     | Untested                        |
| Deno                | ❌     | Currently unsupported           |

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

Both API routes and `/pages` are supported with some caveats. To overcome issues with importing `/node` variant and incorrect `__dirname` resolution, you can import the package from `@dqbd/tiktoken/bundler` instead.

```typescript
import { get_encoding } from "@dqbd/tiktoken/bundler";
import { NextApiRequest, NextApiResponse } from "next";

export default function handler(req: NextApiRequest, res: NextApiResponse) {
  return res.status(200).json({
    // eslint-disable-next-line @typescript-eslint/no-unsafe-assignment
    message: get_encoding("gpt2").encode(`Hello World ${Math.random()}`),
  });
}
```

Additional Webpack configuration is also required, see https://github.com/vercel/next.js/issues/29362.

```typescript
class WasmChunksFixPlugin {
  apply(compiler) {
    compiler.hooks.thisCompilation.tap("WasmChunksFixPlugin", (compilation) => {
      compilation.hooks.processAssets.tap(
        { name: "WasmChunksFixPlugin" },
        (assets) =>
          Object.entries(assets).forEach(([pathname, source]) => {
            if (!pathname.match(/\.wasm$/)) return;
            compilation.deleteAsset(pathname);

            const name = pathname.split("/")[1];
            const info = compilation.assetsInfo.get(pathname);
            compilation.emitAsset(name, source, info);
          })
      );
    });
  }
}

const config = {
  webpack(config, { isServer, dev }) {
    config.experiments = {
      asyncWebAssembly: true,
      layers: true,
    };

    if (!dev && isServer) {
      config.output.webassemblyModuleFilename = "chunks/[id].wasm";
      config.plugins.push(new WasmChunksFixPlugin());
    }

    return config;
  },
};
```

To properly resolve `tsconfig.json`, use either `moduleResolution: "node16"` or `moduleResolution: "nodenext"`:

```json
{
  "compilerOptions": {
    "moduleResolution": "node16"
  }
}
```

## Acknowledgements

- https://github.com/zurawiki/tiktoken-rs
