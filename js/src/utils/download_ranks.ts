function assert(condition: unknown, message?: string): asserts condition {
  if (!condition) {
    throw new Error(message);
  }
}

// printable ascii characters according to python
function is_printable(u: number): boolean {
  return !(u <= 31 || (u >= 127 && u <= 160) || u == 173);
}

export function data_gym_to_mergeable_bpe_ranks(
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
  return dump_tiktoken_bpe(bpe_ranks);
}

export function dump_tiktoken_bpe(bpe_ranks: Record<string, number>) {
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
