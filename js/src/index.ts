import { TiktokenModel, TiktokenEncoding } from "./ranks/ranks";
import gpt2 from "./ranks/gpt2";
import p50k_base from "./ranks/p50k_base";
import p50k_edit from "./ranks/p50k_edit";
import r50k_base from "./ranks/r50k_base";
import cl100k_base from "./ranks/cl100k_base";
import o200k_base from "./ranks/o200k_base";

import { Tiktoken, getEncodingNameForModel } from "./core";
import { never } from "./utils";

export function getEncoding(
  encoding: TiktokenEncoding,
  extendSpecialTokens?: Record<string, number>
): Tiktoken {
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
    case "o200k_base":
      return new Tiktoken(o200k_base, extendSpecialTokens);
    default:
      never(encoding);
      throw new Error("Unknown encoding");
  }
}

export function encodingForModel(
  model: TiktokenModel,
  extendSpecialTokens?: Record<string, number>
) {
  return getEncoding(getEncodingNameForModel(model), extendSpecialTokens);
}

export { Tiktoken, TiktokenBPE, getEncodingNameForModel } from "./core";
export { TiktokenModel, TiktokenEncoding } from "./ranks/ranks";
