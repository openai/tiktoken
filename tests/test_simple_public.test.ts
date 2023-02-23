import { it, expect } from "vitest";
import { encoding_for_model, get_encoding } from "../";

it("encoding_for_model initialization", () => {
  expect(() => encoding_for_model("gpt2")).not.toThrowError();

  // @ts-expect-error
  expect(() => encoding_for_model("gpt2-unknown")).toThrowError(
    "Invalid model"
  );
});

it("get_encoding initialization", () => {
  expect(() => get_encoding("cl100k_base")).not.toThrowError();

  // @ts-expect-error
  expect(() => get_encoding("unknown")).toThrowError("Invalid encoding");
});

it("test_simple", () => {
  const enc = get_encoding("gpt2");
  expect(enc.encode("hello world")).toStrictEqual(
    new Uint32Array([31373, 995])
  );

  expect(
    new TextDecoder().decode(enc.decode(new Uint32Array([31373, 995])))
  ).toStrictEqual("hello world");

  expect(enc.encode("hello <|endoftext|>", "all")).toStrictEqual(
    new Uint32Array([31373, 220, 50256])
  );
});

it("test_simple", () => {
  const decoder = new TextDecoder();
  const enc = get_encoding("cl100k_base");
  expect(enc.encode("hello world")).toStrictEqual(
    new Uint32Array([15339, 1917])
  );
});

it("test_custom_tokens", () => {
  const enc = encoding_for_model("gpt2", {
    "<|im_start|>": 100264,
    "<|im_end|>": 100265,
  });
  expect(enc.encode("<|im_start|>test<|im_end|>", "all")).toStrictEqual(
    new Uint32Array([100264, 9288, 100265])
  );
});
