import base64 from "base64-js";

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

export interface TiktokenBPE {
  pat_str: string;
  special_tokens: Record<string, number>;
  bpe_ranks: string;
}

export class Tiktoken {
  /** @internal */
  protected specialTokens: Record<string, number>;

  /** @internal */
  protected inverseSpecialTokens: Record<number, Uint8Array>;

  /** @internal */
  protected patStr: string;

  /** @internal */
  protected textEncoder = new TextEncoder();

  /** @internal */
  protected textDecoder = new TextDecoder("utf-8");

  /** @internal */
  protected rankMap = new Map<string, number>();

  /** @internal */
  protected textMap = new Map<number, Uint8Array>();

  constructor(
    ranks: TiktokenBPE,
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
