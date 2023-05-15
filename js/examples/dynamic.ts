import {
  Tiktoken,
  TiktokenBPE,
  TiktokenEncoding,
  TiktokenModel,
  getEncodingNameForModel,
} from "../dist";

const cache: Record<string, TiktokenBPE> = {};

async function getEncoding(
  encoding: TiktokenEncoding,
  extendedSpecialTokens?: Record<string, number>
) {
  if (!(encoding in cache)) {
    const res = await fetch(`https://tiktoken.pages.dev/js/${encoding}.json`);

    if (!res.ok) throw new Error("Failed to fetch encoding");
    cache[encoding] = await res.json();
  }
  return new Tiktoken(cache[encoding], extendedSpecialTokens);
}

async function encodingForModel(
  model: TiktokenModel,
  extendedSpecialTokens?: Record<string, number>
) {
  return getEncoding(getEncodingNameForModel(model), extendedSpecialTokens);
}

async function main() {
  const encodings = await encodingForModel("gpt-4");
  const text = "function foo() { return 1; }";
  const tokens = encodings.encode(text);
  console.log(tokens);
}

main();
