import assert from "node:assert";
import fs from "node:fs/promises";
import path from "node:path";

// printable ascii characters according to python
function is_printable(u: number): boolean {
  return !(u <= 31 || (u >= 127 && u <= 160) || u == 173);
}

function data_gym_to_mergeable_bpe_ranks(
  vocal_bpe_contents: string,
  encoder_json_contents: string
) {
  const rank_to_intbyte = Array.from({ length: 2 ** 8 }, (_, i) => i).filter(
    (i) => is_printable(i) && String.fromCharCode(i) !== " "
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

  assert(
    rank_to_intbyte.length === 2 ** 8,
    "rank_to_intbyte.length must be 2**8"
  );

  // vocab_bpe contains the merges along with associated ranks
  const bpe_merges = vocal_bpe_contents
    .split("\n")
    .slice(1, -1)
    .map((merge_str) => merge_str.split(" "));

  function decode_data_gym(value: string) {
    return value.split("").map((b) => data_gym_byte_to_byte[b]);
  }

  // add the single byte tokens
  const bpe_ranks = Object.fromEntries(rank_to_intbyte.map((b, i) => [b, i]));

  // add the merged tokens
  n = rank_to_intbyte.length;
  for (const [first, second] of bpe_merges) {
    bpe_ranks[
      [...decode_data_gym(first), ...decode_data_gym(second)].join(",")
    ] = n;
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
      decode_data_gym(k).join(","),
      v,
    ])
  );

  // drop these two special tokens if present, since they're not mergeable bpe tokens
  delete encoder_json_loaded[decode_data_gym("<|endoftext|>").join(",")];
  delete encoder_json_loaded[decode_data_gym("<|startoftext|>").join(",")];

  function normalize_map(items: Record<string, number>) {
    return JSON.stringify(
      Object.keys(items)
        .sort()
        .map((key) => [key, items[key]])
    );
  }

  assert(normalize_map(bpe_ranks) === normalize_map(encoder_json_loaded));
  return bpe_ranks;
}

function load_tiktoken_bpe(tiktoken_bpe_file: string) {
  return Object.fromEntries<number>(
    tiktoken_bpe_file
      .split("\n")
      .map((line) => line.trim() && line.split(" "))
      .filter((x): x is Array<string> => !!x && Array.isArray(x))
      .map(([token, rank]) => [
        Buffer.from(token, "base64").join(","),
        Number.parseInt(rank, 10),
      ])
  );
}

function dump_tiktoken_bpe(bpe_ranks: Record<string, number>) {
  return (
    Object.entries(bpe_ranks)
      .sort((a, b) => a[1] - b[1])
      .map(([token_str, rank]) =>
        [
          Buffer.from(
            token_str.split(",").map((i) => Number.parseInt(i, 10))
          ).toString("base64"),
          rank,
        ].join(" ")
      )
      .join("\n") + "\n"
  );
}

async function requestText(url: string) {
  return await fetch(url).then((a) => a.text());
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

    const targetFile = path.resolve(__dirname, `../ranks/${name}.tiktoken`);

    try {
      await fs.stat(targetFile);
      continue;
    } catch {}

    let ranks: Record<string, number> | null = null;

    if (data.data_gym_to_mergeable_bpe_ranks) {
      ranks = data_gym_to_mergeable_bpe_ranks(
        await requestText(data.data_gym_to_mergeable_bpe_ranks.vocab_bpe_file),
        await requestText(
          data.data_gym_to_mergeable_bpe_ranks.encoder_json_file
        )
      );
    } else if (data.load_tiktoken_bpe) {
      ranks = load_tiktoken_bpe(await requestText(data.load_tiktoken_bpe));
    }

    if (ranks != null) {
      await fs.writeFile(targetFile, dump_tiktoken_bpe(ranks));
    }
  }
}

main();
