import { TiktokenModel, TiktokenEncoding } from "./ranks/ranks";
import gpt2 from "./ranks/gpt2";
import p50k_base from "./ranks/p50k_base";
import p50k_edit from "./ranks/p50k_edit";
import r50k_base from "./ranks/r50k_base";
import cl100k_base from "./ranks/cl100k_base";

import { Tiktoken } from "./core";
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
    default:
      never(encoding);
      throw new Error("Unknown encoding");
  }
}

export function getEncodingNameForModel(model: TiktokenModel) {
  switch (model) {
    case "gpt2": {
      return "gpt2";
    }
    case "code-cushman-001":
    case "code-cushman-002":
    case "code-davinci-001":
    case "code-davinci-002":
    case "cushman-codex":
    case "davinci-codex":
    case "text-davinci-002":
    case "text-davinci-003": {
      return "p50k_base";
    }
    case "code-davinci-edit-001":
    case "text-davinci-edit-001": {
      return "p50k_edit";
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
      return "r50k_base";
    }
    case "gpt-3.5-turbo-0301":
    case "gpt-3.5-turbo":
    case "gpt-4-0314":
    case "gpt-4-32k-0314":
    case "gpt-4-32k":
    case "gpt-4":
    case "text-embedding-ada-002": {
      return "cl100k_base";
    }
    default:
      never(model);
      throw new Error("Unknown model");
  }
}

export function encodingForModel(
  model: TiktokenModel,
  extendSpecialTokens?: Record<string, number>
) {
  return getEncoding(getEncodingNameForModel(model), extendSpecialTokens);
}

export { Tiktoken, TiktokenBPE } from "./core";
export { TiktokenModel, TiktokenEncoding } from "./ranks/ranks";
