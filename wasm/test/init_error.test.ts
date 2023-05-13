import { it, expect } from "vitest";
import { encoding_for_model, get_encoding, Tiktoken } from "../dist/init";
import model from "../dist/encoders/cl100k_base.json";

it("use before initialization", () => {
  expect(() => encoding_for_model("gpt2")).toThrowError(
    "tiktoken: WASM binary has not been propery initialized."
  );
  expect(() => get_encoding("gpt2")).toThrowError(
    "tiktoken: WASM binary has not been propery initialized."
  );

  expect(
    () => new Tiktoken(model.bpe_ranks, model.special_tokens, model.pat_str)
  ).toThrowError(
    "tiktoken: WASM binary has not been propery initialized."
  );
});
