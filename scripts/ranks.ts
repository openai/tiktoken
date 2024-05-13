import fs from "node:fs/promises";
import path from "node:path";
import outdent from "outdent";
import { fromByteArray } from "base64-js";
import registry from "../tiktoken/registry.json";
import modelToEncoding from "../tiktoken/model_to_encoding.json";

// printable ascii characters according to python
function isPrintable(u: number): boolean {
  return !(u <= 31 || (u >= 127 && u <= 160) || u == 173);
}

function dataGymToMergeableBpeRanks(
  vocal_bpe_contents: string,
  encoder_json_contents: string
) {
  const rank_to_intbyte = Array.from({ length: 2 ** 8 }, (_, i) => i).filter(
    (i) => isPrintable(i) && String.fromCharCode(i) !== " "
  );

  const data_gym_byte_to_byte = rank_to_intbyte.reduce<Record<string, number>>(
    (memo, item) => {
      memo[String.fromCharCode(item)] = item;
      return memo;
    },
    {}
  );

  let n = 0;
  for (let b = 0; b < 2 ** 8; b++) {
    if (!rank_to_intbyte.includes(b)) {
      rank_to_intbyte.push(b);
      data_gym_byte_to_byte[String.fromCharCode(2 ** 8 + n)] = b;
      n += 1;
    }
  }

  if (rank_to_intbyte.length !== 2 ** 8) {
    throw new Error("rank_to_intbyte.length must be 2**8");
  }

  // vocab_bpe contains the merges along with associated ranks
  const bpe_merges = vocal_bpe_contents
    .split("\n")
    .slice(1, -1)
    .map((merge_str) => merge_str.split(" "));

  function decodeDataGym(value: string) {
    return value.split("").map((b) => data_gym_byte_to_byte[b]);
  }

  // add the single byte tokens
  const bpe_ranks = Object.fromEntries(rank_to_intbyte.map((b, i) => [b, i]));

  // add the merged tokens
  n = rank_to_intbyte.length;
  for (const [first, second] of bpe_merges) {
    bpe_ranks[[...decodeDataGym(first), ...decodeDataGym(second)].join(",")] =
      n;
    n += 1;
  }

  // check that the encoder file matches the merges file
  // this sanity check is important since tiktoken assumes that ranks are ordered the same
  // as merge priority
  const encoder_json: Record<string, number> = JSON.parse(
    encoder_json_contents
  );

  const encoder_json_loaded = Object.fromEntries(
    Object.entries(encoder_json).map(([k, v]) => [
      decodeDataGym(k).join(","),
      v,
    ])
  );

  // drop these two special tokens if present, since they're not mergeable bpe tokens
  delete encoder_json_loaded[decodeDataGym("<|endoftext|>").join(",")];
  delete encoder_json_loaded[decodeDataGym("<|startoftext|>").join(",")];

  function normalize_map(items: Record<string, number>) {
    return JSON.stringify(
      Object.keys(items)
        .sort()
        .map((key) => [key, items[key]])
    );
  }

  if (normalize_map(bpe_ranks) !== normalize_map(encoder_json_loaded)) {
    throw new Error("bpe_ranks !== encoder_json_loaded");
  }

  return bpe_ranks;
}

function dumpTiktokenBpe(bpe_ranks: Record<string, number>) {
  return (
    Object.entries(bpe_ranks)
      .sort((a, b) => a[1] - b[1])
      .map(([token_str, rank]) =>
        [
          fromByteArray(
            new Uint8Array(
              token_str.split(",").map((i) => Number.parseInt(i, 10))
            )
          ),
          rank,
        ].join(" ")
      )
      .join("\n") + "\n"
  );
}

async function downloadBpe(
  registry: (
    | { load_tiktoken_bpe: string }
    | {
        data_gym_to_mergeable_bpe_ranks: {
          vocab_bpe_file: string;
          encoder_json_file: string;
        };
      }
  ) & {
    explicit_n_vocab?: number;
    pat_str: string;
    special_tokens: Record<string, number>;
  }
) {
  if ("data_gym_to_mergeable_bpe_ranks" in registry) {
    const [vocab_bpe, encoder_json] = await Promise.all([
      fetch(registry.data_gym_to_mergeable_bpe_ranks.vocab_bpe_file).then((a) =>
        a.text()
      ),
      fetch(registry.data_gym_to_mergeable_bpe_ranks.encoder_json_file).then(
        (a) => a.text()
      ),
    ]);

    return {
      explicit_n_vocab: registry.explicit_n_vocab,
      pat_str: registry.pat_str,
      special_tokens: registry.special_tokens,
      bpe_ranks: dumpTiktokenBpe(
        dataGymToMergeableBpeRanks(vocab_bpe, encoder_json)
      ),
    };
  } else {
    return {
      explicit_n_vocab: registry.explicit_n_vocab,
      pat_str: registry.pat_str,
      special_tokens: registry.special_tokens,
      bpe_ranks: await fetch(registry.load_tiktoken_bpe).then((a) => a.text()),
    };
  }
}

function compressTiktokenBpe(tiktoken_bpe_file: string) {
  const original = tiktoken_bpe_file
    .split("\n")
    .map((line) => line.trim() && line.split(" "))
    .filter((x): x is Array<string> => !!x && Array.isArray(x))
    .map(([token, rank]) => [token, Number.parseInt(rank, 10)] as const)
    .sort((a, b) => a[1] - b[1]);

  const newTokens = original.reduce<
    Array<{ offset: number; tokens: string[] }>
  >((memo, item) => {
    if (memo.length === 0) return [{ offset: item[1], tokens: [item[0]] }];
    const lastSplit = memo[memo.length - 1];
    const nextOffset = lastSplit.offset + lastSplit.tokens.length;

    if (nextOffset === item[1]) {
      lastSplit.tokens.push(item[0]);
      return memo;
    }

    return [...memo, { offset: item[1], tokens: [item[0]] }];
  }, []);

  const compressed = newTokens
    .map((x) => `! ${x.offset} ${x.tokens.join(" ")}`)
    .join("\n");

  // make sure the compressed and the original files are the same
  const tiktokenOld = compressed
    .split("\n")
    .filter(Boolean)
    .reduce<Record<string, number>>((memo, x) => {
      const [_, offsetStr, ...tokens] = x.split(" ");
      const offset = Number.parseInt(offsetStr, 10);
      tokens.forEach((token, i) => (memo[token] = offset + i));
      return memo;
    }, {});

  function normalizeMap(items: Record<string, number>) {
    return JSON.stringify(
      Object.keys(items)
        .sort()
        .map((key) => [key, items[key]])
    );
  }

  if (
    normalizeMap(tiktokenOld) !== normalizeMap(Object.fromEntries(original))
  ) {
    throw new Error("Invalid compression");
  }

  return compressed;
}

function combineInsensitive(value: string, acc: string[] = [""]): string[] {
  if (value.length === 0) return acc;
  if (value[0].match(/[a-zA-Z]/)) {
    return combineInsensitive(
      value.substring(1),
      acc.flatMap((i) => [
        `${i}${value[0].toLocaleLowerCase()}`,
        `${i}${value[0].toLocaleUpperCase()}`,
      ])
    );
  }

  return combineInsensitive(
    value.substring(1),
    acc.map((i) => `${i}${value[0]}`)
  );
}

async function main() {
  for (const lib of ["wasm", "js"]) {
    const targetDir = path.resolve(__dirname, "../", lib, "src/ranks");

    try {
      await fs.mkdir(targetDir, { recursive: true });
    } catch {}

    for (const name in registry) {
      console.log(name);

      const data = registry[name as keyof typeof registry];
      const bpePath = path.resolve(targetDir, `${name}.tiktoken`);
      const compressPath = path.resolve(targetDir, `${name}.compress.tiktoken`);
      const regexPath = path.resolve(targetDir, `${name}.regex.tiktoken`);
      const jsonPath = path.resolve(targetDir, `${name}.json`);
      const cjsPath = path.resolve(targetDir, `${name}.cjs`);
      const dtsPath = path.resolve(targetDir, `${name}.d.ts`);
      const mjsPath = path.resolve(targetDir, `${name}.js`);

      try {
        await Promise.all([
          fs.stat(bpePath),
          fs.stat(jsonPath),
          fs.stat(compressPath),
          fs.stat(regexPath),
          fs.stat(cjsPath),
          fs.stat(mjsPath),
          fs.stat(dtsPath),
        ]);

        continue;
      } catch {}

      const bpe = await downloadBpe(data);

      if (lib === "js") {
        bpe.pat_str = bpe.pat_str.replaceAll(
          /\(\?i:(.*?)\)/g,
          (_, match: string) =>
            `(${match
              .split("|")
              .flatMap((a) => combineInsensitive(a))
              .join("|")})`
        );

        // attempt to create a regexp
        new RegExp(bpe.pat_str, "u");
      }

      await fs.writeFile(bpePath, bpe.bpe_ranks, { encoding: "utf-8" });

      const compress = compressTiktokenBpe(bpe.bpe_ranks);
      await fs.writeFile(compressPath, compress, { encoding: "utf-8" });

      const regex = bpe.pat_str;
      await fs.writeFile(regexPath, regex, { encoding: "utf-8" });

      const json = JSON.stringify({ ...bpe, bpe_ranks: compress });
      await fs.writeFile(jsonPath, json, { encoding: "utf-8" });

      const mjs = `export default ${json};`;
      await fs.writeFile(mjsPath, mjs, { encoding: "utf-8" });

      const cjs = `module.exports = ${json};`;
      await fs.writeFile(cjsPath, cjs, { encoding: "utf-8" });

      const dts = outdent`
        declare const encoder: {
          pat_str: string;
          special_tokens: Record<string, number>;
          bpe_ranks: string;
        };
        export default encoder;
      `;

      await fs.writeFile(dtsPath, dts, { encoding: "utf-8" });
    }

    const indexPath = path.resolve(targetDir, "ranks.ts");
    const indexMjs = outdent`
      export type TiktokenEncoding = ${Object.keys(registry)
        .map((i) => `"${i}"`)
        .join(" | ")};
      export type TiktokenModel = ${Object.keys(modelToEncoding)
        .map((i) => `"${i}"`)
        .join(" | ")};
    `;
    await fs.writeFile(indexPath, indexMjs, { encoding: "utf-8" });
  }
}

main();
