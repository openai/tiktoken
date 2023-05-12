import gpt2 from "../../wasm/dist/encoders/gpt2.json";
import p50k_base from "../../wasm/dist/encoders/p50k_base.json";
import p50k_edit from "../../wasm/dist/encoders/p50k_edit.json";
import r50k_base from "../../wasm/dist/encoders/r50k_base.json";
import cl100k_base from "../../wasm/dist/encoders/cl100k_base.json";

import base64 from "base64-js";

function never(message: string, _: never) {
  throw new Error(message);
}

function bytePairMerge(
  piece: Uint8Array,
  ranks: Map<string, number>
): Array<{ start: number; end: number }> {
  let parts: Array<{ start: number; end: number }> = Array.from(
    { length: piece.length },
    (_, i) => ({ start: i, end: i + 1 })
  );

  while (parts.length > 1) {
    let minRank: [number, number] | null = null;

    for (let i = 0; i < parts.length - 1; i++) {
      const slice = piece.slice(parts[i].start, parts[i + 1].end);
      const rank = ranks.get(slice.join(","));
      if (rank == null) continue;

      if (minRank == null || rank < minRank[0]) {
        minRank = [rank, i];
      }
    }

    if (minRank != null) {
      const i = minRank[1];
      parts[i] = { start: parts[i].start, end: parts[i + 1].end };
      parts.splice(i + 1, 1);
    } else {
      break;
    }
  }
  return parts;
}

function bytePairEncode(piece: Uint8Array, ranks: Map<string, number>) {
  if (piece.length === 1) return [ranks.get(piece.join(","))!];

  return bytePairMerge(piece, ranks)
    .map((p) => ranks.get(piece.slice(p.start, p.end).join(",")))
    .filter((x): x is number => x != null);
}

function escapeRegex(str: string) {
  return str.replace(/[\\^$*+?.()|[\]{}]/g, "\\$&");
}

export class Tiktoken {
  protected specialTokens: Record<string, number>;
  protected inverseSpecialTokens: Record<number, Uint8Array>;

  protected patStr: string;

  protected textEncoder = new TextEncoder();
  protected textDecoder = new TextDecoder("utf-8");

  protected rankMap = new Map<string, number>();
  protected textMap = new Map<number, Uint8Array>();

  constructor(
    ranks: {
      pat_str: string;
      special_tokens: Record<string, number>;
      bpe_ranks: string;
    } = cl100k_base,
    extendedSpecialTokens?: Record<string, number>
  ) {
    this.patStr = ranks.pat_str;

    const uncompressed = ranks.bpe_ranks
      .split("\n")
      .filter(Boolean)
      .reduce<Record<string, number>>((memo, x) => {
        const [_, offsetStr, ...tokens] = x.split(" ");
        const offset = Number.parseInt(offsetStr, 10);
        tokens.forEach((token, i) => (memo[token] = offset + i));
        return memo;
      }, {});

    for (const [token, rank] of Object.entries(uncompressed)) {
      const bytes = base64.toByteArray(token);
      this.rankMap.set(bytes.join(","), rank);
      this.textMap.set(rank, bytes);
    }

    this.specialTokens = { ...ranks.special_tokens, ...extendedSpecialTokens };
    this.inverseSpecialTokens = Object.entries(this.specialTokens).reduce<
      Record<number, Uint8Array>
    >((memo, [text, rank]) => {
      memo[rank] = this.textEncoder.encode(text);
      return memo;
    }, {});
  }

  encode(text: string, allowedSpecial: Set<string> | "all" = new Set()) {
    const regexes = new RegExp(this.patStr, "ug");
    const specialRegex = new RegExp(
      Object.keys(this.specialTokens)
        .map((i) => escapeRegex(i))
        .join("|"),
      "g"
    );

    const ret: number[] = [];

    const allowedSpecialSet =
      allowedSpecial === "all"
        ? new Set(Object.keys(this.specialTokens))
        : allowedSpecial;

    let start = 0;
    while (true) {
      let nextSpecial: RegExpMatchArray | null = null;
      let startFind = start;

      while (true) {
        specialRegex.lastIndex = startFind;
        nextSpecial = specialRegex.exec(text);
        if (nextSpecial == null || allowedSpecialSet.has(nextSpecial[0])) break;
        startFind = nextSpecial.index! + 1;
      }

      const end = nextSpecial?.index ?? text.length;
      for (const match of text.substring(start, end).matchAll(regexes)) {
        const piece = this.textEncoder.encode(match[0]);
        const token = this.rankMap.get(piece.join(","));

        if (token != null) {
          ret.push(token);
          continue;
        }

        ret.push(...bytePairEncode(piece, this.rankMap));
      }

      if (nextSpecial == null) break;
      let token = this.specialTokens[nextSpecial[0]];
      ret.push(token);

      start = nextSpecial.index! + nextSpecial[0].length;
    }

    return ret;
  }

  decode(tokens: number[]) {
    const res: Uint8Array[] = [];
    let length = 0;
    for (let i = 0; i < tokens.length; ++i) {
      const token = tokens[i];
      const bytes = this.textMap.get(token) ?? this.inverseSpecialTokens[token];

      if (bytes != null) {
        res.push(bytes);
        length += bytes.length;
      }
    }

    const mergedArray = new Uint8Array(length);
    let i = 0;
    for (const bytes of res) {
      mergedArray.set(bytes, i);
      i += bytes.length;
    }

    return this.textDecoder.decode(mergedArray);
  }
}

export type TiktokenEncoding =
  | "gpt2"
  | "r50k_base"
  | "p50k_base"
  | "p50k_edit"
  | "cl100k_base";

export function getEncoding(
  encoding: TiktokenEncoding,
  extendSpecialTokens?: Record<string, number>
) {
  switch (encoding) {
    case "gpt2":
      return new Tiktoken(gpt2, extendSpecialTokens);
    case "r50k_base":
      return new Tiktoken(r50k_base, extendSpecialTokens);
    case "p50k_base":
      return new Tiktoken(p50k_base, extendSpecialTokens);
    case "p50k_edit":
      return new Tiktoken(p50k_edit, extendSpecialTokens);
    case "cl100k_base":
      return new Tiktoken(cl100k_base, extendSpecialTokens);
    default:
      never("Unknown encoding", encoding);
  }
}

export type TiktokenModel =
  | "text-davinci-003"
  | "text-davinci-002"
  | "text-davinci-001"
  | "text-curie-001"
  | "text-babbage-001"
  | "text-ada-001"
  | "davinci"
  | "curie"
  | "babbage"
  | "ada"
  | "code-davinci-002"
  | "code-davinci-001"
  | "code-cushman-002"
  | "code-cushman-001"
  | "davinci-codex"
  | "cushman-codex"
  | "text-davinci-edit-001"
  | "code-davinci-edit-001"
  | "text-embedding-ada-002"
  | "text-similarity-davinci-001"
  | "text-similarity-curie-001"
  | "text-similarity-babbage-001"
  | "text-similarity-ada-001"
  | "text-search-davinci-doc-001"
  | "text-search-curie-doc-001"
  | "text-search-babbage-doc-001"
  | "text-search-ada-doc-001"
  | "code-search-babbage-code-001"
  | "code-search-ada-code-001"
  | "gpt2"
  | "gpt-4"
  | "gpt-4-0314"
  | "gpt-4-32k"
  | "gpt-4-32k-0314"
  | "gpt-3.5-turbo"
  | "gpt-3.5-turbo-0301";

export function encodingForModel(
  model: TiktokenModel,
  extendSpecialTokens?: Record<string, number>
) {
  switch (model) {
    case "gpt2": {
      return getEncoding("gpt2", extendSpecialTokens);
    }
    case "code-cushman-001":
    case "code-cushman-002":
    case "code-davinci-001":
    case "code-davinci-002":
    case "cushman-codex":
    case "davinci-codex":
    case "text-davinci-002":
    case "text-davinci-003": {
      return getEncoding("p50k_base", extendSpecialTokens);
    }
    case "code-davinci-edit-001":
    case "text-davinci-edit-001": {
      return getEncoding("p50k_edit", extendSpecialTokens);
    }
    case "ada":
    case "babbage":
    case "code-search-ada-code-001":
    case "code-search-babbage-code-001":
    case "curie":
    case "davinci":
    case "text-ada-001":
    case "text-babbage-001":
    case "text-curie-001":
    case "text-davinci-001":
    case "text-search-ada-doc-001":
    case "text-search-babbage-doc-001":
    case "text-search-curie-doc-001":
    case "text-search-davinci-doc-001":
    case "text-similarity-ada-001":
    case "text-similarity-babbage-001":
    case "text-similarity-curie-001":
    case "text-similarity-davinci-001": {
      return getEncoding("r50k_base", extendSpecialTokens);
    }
    case "gpt-3.5-turbo-0301":
    case "gpt-3.5-turbo":
    case "gpt-4-0314":
    case "gpt-4-32k-0314":
    case "gpt-4-32k":
    case "gpt-4":
    case "text-embedding-ada-002": {
      return getEncoding("cl100k_base", extendSpecialTokens);
    }
    default:
      never("Unknown model", model);
  }
}
