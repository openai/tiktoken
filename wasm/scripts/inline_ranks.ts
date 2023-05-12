import fs from "node:fs/promises";
import path from "node:path";
import { load } from "../src/load";

function compress_tiktoken_bpe(tiktoken_bpe_file: string) {
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

  function normalize_map(items: Record<string, number>) {
    return JSON.stringify(
      Object.keys(items)
        .sort()
        .map((key) => [key, items[key]])
    );
  }

  if (
    normalize_map(tiktokenOld) !== normalize_map(Object.fromEntries(original))
  ) {
    throw new Error("Invalid compression");
  }

  return compressed;
}

async function main() {
  try {
    await fs.mkdir(path.resolve(__dirname, "../ranks"), { recursive: true });
  } catch {}

  const registry = JSON.parse(
    await fs.readFile(path.resolve(__dirname, "../../tiktoken/registry.json"), {
      encoding: "utf-8",
    })
  );

  for (const name in registry) {
    console.log(name);
    const data = registry[name];

    const tiktokenFile = path.resolve(__dirname, `../ranks/${name}.tiktoken`);
    const tiktokenCompressedFile = path.resolve(
      __dirname,
      `../ranks/${name}.compress.tiktoken`
    );
    const jsonFile = path.resolve(__dirname, `../ranks/${name}.json`);

    try {
      await Promise.all([
        fs.stat(tiktokenFile),
        fs.stat(jsonFile),
        fs.stat(tiktokenCompressedFile),
      ]);
      continue;
    } catch {}

    const result = await load(data);
    await fs.writeFile(tiktokenFile, result.bpe_ranks, { encoding: "utf-8" });

    const compress = compress_tiktoken_bpe(result.bpe_ranks);
    await fs.writeFile(tiktokenCompressedFile, compress, {
      encoding: "utf-8",
    });

    await fs.writeFile(
      jsonFile,
      JSON.stringify({ ...result, bpe_ranks: compress }),
      { encoding: "utf-8" }
    );
  }
}

main();
