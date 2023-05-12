// @ts-expect-error
import * as imports from "./tiktoken_bg";

let isInitialized = false;
export async function init(
  callback: (
    imports: WebAssembly.Imports
  ) => Promise<WebAssembly.Instance | WebAssembly.WebAssemblyInstantiatedSource>
): Promise<void> {
  if (isInitialized) return imports;
  const result = await callback({ "./tiktoken_bg.js": imports });
  const instance =
    "instance" in result && result.instance instanceof WebAssembly.Instance
      ? result.instance
      : result instanceof WebAssembly.Instance
      ? result
      : null;
  if (instance == null) throw new Error("Missing instance");
  imports.__wbg_set_wasm(instance.exports);
  isInitialized = true;
  return imports;
}

// @ts-expect-error
export * from "./tiktoken_bg";
