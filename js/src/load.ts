/**
Copyright (c) 2014 Jameson Little

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
 */
const lookup: string[] = [];
const revLookup: number[] = [];

const code = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
for (var i = 0, len = code.length; i < len; ++i) {
  lookup[i] = code[i];
  revLookup[code.charCodeAt(i)] = i;
}

// Support decoding URL-safe base64 strings, as Node.js does.
// See: https://en.wikipedia.org/wiki/Base64#URL_applications
revLookup["-".charCodeAt(0)] = 62;
revLookup["_".charCodeAt(0)] = 63;

function tripletToBase64(num: number) {
  return (
    lookup[(num >> 18) & 0x3f] +
    lookup[(num >> 12) & 0x3f] +
    lookup[(num >> 6) & 0x3f] +
    lookup[num & 0x3f]
  );
}

function encodeChunk(uint8: number[], start: number, end: number) {
  var tmp;
  var output = [];
  for (var i = start; i < end; i += 3) {
    tmp =
      ((uint8[i] << 16) & 0xff0000) +
      ((uint8[i + 1] << 8) & 0xff00) +
      (uint8[i + 2] & 0xff);
    output.push(tripletToBase64(tmp));
  }
  return output.join("");
}

function fromByteArray(uint8: number[]) {
  var tmp;
  var len = uint8.length;
  var extraBytes = len % 3; // if we have 1 byte left, pad 2 bytes
  var parts = [];
  var maxChunkLength = 16383; // must be multiple of 3

  // go through the array every three bytes, we'll deal with trailing stuff later
  for (var i = 0, len2 = len - extraBytes; i < len2; i += maxChunkLength) {
    parts.push(
      encodeChunk(
        uint8,
        i,
        i + maxChunkLength > len2 ? len2 : i + maxChunkLength
      )
    );
  }

  // pad the end with zeros, but make sure to not forget the extra bytes
  if (extraBytes === 1) {
    tmp = uint8[len - 1];
    parts.push(lookup[tmp >> 2] + lookup[(tmp << 4) & 0x3f] + "==");
  } else if (extraBytes === 2) {
    tmp = (uint8[len - 2] << 8) + uint8[len - 1];
    parts.push(
      lookup[tmp >> 10] +
        lookup[(tmp >> 4) & 0x3f] +
        lookup[(tmp << 2) & 0x3f] +
        "="
    );
  }

  return parts.join("");
}

function is_printable(u: number): boolean {
  // printable ascii characters according to python
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

  if (rank_to_intbyte.length !== 2 ** 8) {
    throw new Error("rank_to_intbyte.length must be 2**8");
  }

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

  if (normalize_map(bpe_ranks) !== normalize_map(encoder_json_loaded)) {
    throw new Error("bpe_ranks !== encoder_json_loaded");
  }

  return bpe_ranks;
}

function dump_tiktoken_bpe(bpe_ranks: Record<string, number>) {
  return (
    Object.entries(bpe_ranks)
      .sort((a, b) => a[1] - b[1])
      .map(([token_str, rank]) =>
        [
          fromByteArray(
            token_str.split(",").map((i) => Number.parseInt(i, 10))
          ),
          rank,
        ].join(" ")
      )
      .join("\n") + "\n"
  );
}

export async function load(
  registry: (
    | { load_tiktoken_bpe: string }
    | {
        data_gym_to_mergeable_bpe_ranks: {
          vocab_bpe_file: string;
          encoder_json_file: string;
        };
      }
  ) & {
    explicit_n_vocab: number;
    pat_str: string;
    special_tokens: Record<string, number>;
  },
  customFetch?: (url: string) => Promise<string>
) {
  const ofetch = customFetch
    ? customFetch
    : (url: string) => fetch(url).then((r) => r.text());

  if ("data_gym_to_mergeable_bpe_ranks" in registry) {
    const [vocab_bpe, encoder_json] = await Promise.all([
      ofetch(registry.data_gym_to_mergeable_bpe_ranks.vocab_bpe_file),
      ofetch(registry.data_gym_to_mergeable_bpe_ranks.encoder_json_file),
    ]);

    return {
      explicit_n_vocab: registry.explicit_n_vocab,
      pat_str: registry.pat_str,
      special_tokens: registry.special_tokens,
      bpe_ranks: dump_tiktoken_bpe(
        data_gym_to_mergeable_bpe_ranks(vocab_bpe, encoder_json)
      ),
    };
  } else {
    return {
      explicit_n_vocab: registry.explicit_n_vocab,
      pat_str: registry.pat_str,
      special_tokens: registry.special_tokens,
      bpe_ranks: await ofetch(registry.load_tiktoken_bpe),
    };
  }
}
