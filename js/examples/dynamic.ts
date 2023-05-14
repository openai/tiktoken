import { Tiktoken, TiktokenBPE, TiktokenEncoding } from "../dist";

const cache: Record<string, TiktokenBPE> = {};

async function getEncoding(encoding: TiktokenEncoding) {
  if (!(encoding in cache)) {
    const res = await fetch(`https://tiktoken.pages.dev/js/${encoding}.json`);

    if (!res.ok) throw new Error("Failed to fetch encoding");
    cache[encoding] = await res.json();
  }
  return new Tiktoken(cache[encoding]);
}

async function main() {
  const encodings = await getEncoding("cl100k_base");
  const text = "function foo() { return 1; }";
  const tokens = encodings.encode(text);
  console.log(tokens);
}

main();
